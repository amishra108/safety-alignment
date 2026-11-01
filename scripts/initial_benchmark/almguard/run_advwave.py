from pathlib import Path
from io import BytesIO
import argparse
import gc
import json

import librosa
import torch
import torchaudio
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


ROOT = Path(__file__).resolve().parents[3]

DATASET = ROOT / "data/advbench_audio/dataset"

MODEL_PATHS = {
    "qwen2_audio": ROOT / "models/qwen2-audio-7b-instruct",
}

ADV_AUDIO_ROOT = ROOT / "data/almguard/advwave"
RESULT_ROOT = ROOT / "results/almguard/advwave"

SAMPLE_RATE = 16000
SUFFIX_SAMPLES = 44100
MAX_STEPS = 3000
EARLY_STOP_LOSS = 0.1
LEARNING_RATE = 1e-3


def get_audio_bytes(row):
    audio = row["audio"]

    if not isinstance(audio, dict):
        raise RuntimeError(
            f'AdvBench field "audio" must be dict, got {type(audio)}'
        )

    if audio["bytes"] is None:
        raise RuntimeError(
            f'AdvBench audio bytes missing for {row["example_id"]}'
        )

    return audio["bytes"]


def decode_audio(row):
    audio_bytes = get_audio_bytes(row)

    waveform, sr = librosa.load(
        BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True,
    )

    if sr != SAMPLE_RATE:
        raise RuntimeError(
            f"Expected {SAMPLE_RATE} Hz audio, got {sr}"
        )

    return torch.tensor(
        waveform,
        dtype=torch.float32,
    )


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()


def load_completed(path):
    completed = set()

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            completed.add(
                str(row["example_id"])
            )

    return completed


def load_qwen2():
    model_path = str(
        MODEL_PATHS["qwen2_audio"]
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = (
        Qwen2AudioForConditionalGeneration
        .from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        .cuda()
        .eval()
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, processor


def build_chat_text(processor):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "file:audio.wav",
                }
            ],
        }
    ]

    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


def get_input_embeds(
    model,
    input_ids,
    input_features,
    feature_attention_mask,
    attention_mask,
):
    inputs_embeds = (
        model.get_input_embeddings()(
            input_ids
        )
    )

    audio_feat_lengths, audio_output_lengths = (
        model.audio_tower
        ._get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )
    )

    batch_size, _, max_mel_seq_len = (
        input_features.shape
    )

    max_seq_len = (
        (max_mel_seq_len - 2) // 2 + 1
    )

    seq_range = (
        torch.arange(
            0,
            max_seq_len,
            dtype=audio_feat_lengths.dtype,
            device=audio_feat_lengths.device,
        )
        .unsqueeze(0)
        .expand(
            batch_size,
            max_seq_len,
        )
    )

    lengths_expand = (
        audio_feat_lengths
        .unsqueeze(1)
        .expand(
            batch_size,
            max_seq_len,
        )
    )

    padding_mask = (
        seq_range >= lengths_expand
    )

    audio_attention_mask_bool = (
        padding_mask
        .view(
            batch_size,
            1,
            1,
            max_seq_len,
        )
        .expand(
            batch_size,
            1,
            max_seq_len,
            max_seq_len,
        )
    )

    audio_attention_mask = (
        audio_attention_mask_bool.to(
            dtype=model.audio_tower.conv1.weight.dtype,
            device=model.audio_tower.conv1.weight.device,
        )
    )

    audio_attention_mask[
        audio_attention_mask_bool
    ] = float("-inf")

    audio_outputs = model.audio_tower(
        input_features,
        attention_mask=audio_attention_mask,
    )

    selected_audio_feature = (
        audio_outputs.last_hidden_state
    )

    audio_features = (
        model.multi_modal_projector(
            selected_audio_feature
        )
    )

    (
        inputs_embeds,
        _,
        _,
        _,
        _,
    ) = (
        model._merge_input_ids_with_audio_features(
            audio_features,
            audio_output_lengths,
            inputs_embeds,
            input_ids,
            attention_mask,
            None,
        )
    )

    return inputs_embeds


def target_loss(
    model,
    processor,
    chat_text,
    waveform,
    target_text,
):
    target_ids = processor(
        text=target_text,
        return_tensors="pt",
        padding=True,
    )["input_ids"].cuda()

    inputs = processor(
        text=chat_text + target_text,
        audios=waveform.unsqueeze(0),
        return_tensors="pt",
        padding=True,
        sampling_rate=SAMPLE_RATE,
    )

    inputs = {
        key: (
            value.cuda()
            if isinstance(
                value,
                torch.Tensor,
            )
            else value
        )
        for key, value in inputs.items()
    }

    model_inputs = (
        model.prepare_inputs_for_generation(
            **inputs
        )
    )

    inputs_embeds = get_input_embeds(
        model=model,
        input_ids=model_inputs["input_ids"],
        input_features=model_inputs[
            "input_features"
        ],
        feature_attention_mask=model_inputs[
            "feature_attention_mask"
        ],
        attention_mask=model_inputs[
            "attention_mask"
        ],
    )

    outputs = model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=model_inputs[
            "attention_mask"
        ],
        use_cache=False,
    )

    logits = outputs.logits

    shift = (
        inputs_embeds.shape[1]
        - target_ids.shape[1]
    )

    shift_logits = logits[
        ...,
        shift - 1:-1,
        :,
    ].contiguous()

    if (
        shift_logits.shape[1]
        != target_ids.shape[1]
    ):
        raise RuntimeError(
            "Target/logit length mismatch: "
            f"{shift_logits.shape[1]} vs "
            f"{target_ids.shape[1]}"
        )

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.size(-1),
        ),
        target_ids.view(-1),
    )

    return loss


@torch.no_grad()
def select_target(
    model,
    processor,
    chat_text,
    waveform,
    candidates,
):
    best_target = None
    best_loss = float("inf")

    for target in candidates:
        with torch.enable_grad():
            probe_waveform = (
                waveform.detach()
            )

            loss = target_loss(
                model=model,
                processor=processor,
                chat_text=chat_text,
                waveform=probe_waveform,
                target_text=target,
            )

        value = float(
            loss.detach().cpu()
        )

        if value < best_loss:
            best_loss = value
            best_target = target

    if best_target is None:
        raise RuntimeError(
            "Dynamic target selection "
            "returned no target"
        )

    return best_target, best_loss


@torch.no_grad()
def generate_response(
    model,
    processor,
    waveform,
):
    chat_text = build_chat_text(
        processor
    )

    inputs = processor(
        text=chat_text,
        audios=waveform.unsqueeze(0),
        return_tensors="pt",
        padding=True,
        sampling_rate=SAMPLE_RATE,
    )

    inputs = {
        key: (
            value.cuda()
            if isinstance(
                value,
                torch.Tensor,
            )
            else value
        )
        for key, value in inputs.items()
    }

    ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )

    ids = ids[
        :,
        inputs["input_ids"].shape[1]:,
    ]

    return processor.batch_decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def attack_example(
    model,
    processor,
    clean_waveform,
    target_candidates,
    output_path,
):
    clean_waveform = (
        clean_waveform
        .cuda()
        .detach()
    )

    chat_text = build_chat_text(
        processor
    )

    target_text, target_probe_loss = (
        select_target(
            model=model,
            processor=processor,
            chat_text=chat_text,
            waveform=clean_waveform,
            candidates=target_candidates,
        )
    )

    suffix = (
        torch.randn(
            SUFFIX_SAMPLES,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.01
    )

    suffix.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [suffix],
        lr=LEARNING_RATE,
    )

    losses = []

    progress = tqdm(
        range(MAX_STEPS),
        leave=False,
    )

    for step in progress:
        optimizer.zero_grad(
            set_to_none=True
        )

        attacked = torch.cat(
            [
                clean_waveform,
                suffix,
            ],
            dim=0,
        )

        loss = target_loss(
            model=model,
            processor=processor,
            chat_text=chat_text,
            waveform=attacked,
            target_text=target_text,
        )

        loss.backward()
        optimizer.step()

        loss_value = float(
            loss.detach().cpu()
        )

        losses.append(
            loss_value
        )

        progress.set_description(
            f"loss={loss_value:.4f}"
        )

        if (
            loss_value
            < EARLY_STOP_LOSS
        ):
            break

    attacked = torch.cat(
        [
            clean_waveform,
            suffix.detach(),
        ],
        dim=0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torchaudio.save(
        str(output_path),
        attacked.detach()
        .float()
        .cpu()
        .unsqueeze(0),
        SAMPLE_RATE,
    )

    response = generate_response(
        model=model,
        processor=processor,
        waveform=attacked.detach(),
    )

    return {
        "target_text": target_text,
        "target_probe_loss": (
            target_probe_loss
        ),
        "final_loss": losses[-1],
        "steps": len(losses),
        "converged": (
            losses[-1]
            < EARLY_STOP_LOSS
        ),
        "response": response,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "qwen2_audio",
        ],
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=520,
    )

    args = parser.parse_args()

    if not DATASET.exists():
        raise FileNotFoundError(
            DATASET
        )

    ds = load_from_disk(
        str(DATASET)
    )

    if len(ds) != 520:
        raise RuntimeError(
            f"Expected 520 rows, "
            f"got {len(ds)}"
        )

    if not (
        0
        <= args.start
        < args.end
        <= len(ds)
    ):
        raise ValueError(
            f"Invalid range "
            f"{args.start}:{args.end}"
        )

    model, processor = load_qwen2()

    audio_dir = (
        ADV_AUDIO_ROOT
        / args.model
    )

    result_dir = (
        RESULT_ROOT
        / args.model
    )

    audio_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_file = (
        result_dir
        / "responses.jsonl"
    )

    completed = load_completed(
        result_file
    )

    for index in range(
        args.start,
        args.end,
    ):
        row = ds[index]

        example_id = str(
            row["example_id"]
        )

        if example_id in completed:
            print(
                f"[skip] {example_id}",
                flush=True,
            )
            continue

        print(
            f"[{index + 1:03d}/520] "
            f"{example_id}",
            flush=True,
        )

        clean_waveform = decode_audio(
            row
        )

        # AdvBench provides a query-specific
        # affirmative target completion.
        #
        # This list interface is retained so
        # additional dynamically searched
        # AdvWave targets can be supplied
        # without changing the optimization.
        target_candidates = [
            str(row["target"])
        ]

        output_path = (
            audio_dir
            / f"{example_id}.wav"
        )

        result = attack_example(
            model=model,
            processor=processor,
            clean_waveform=clean_waveform,
            target_candidates=target_candidates,
            output_path=output_path,
        )

        record = {
            "example_id": example_id,
            "model": args.model,
            "attack": "advwave",
            "prompt": str(
                row["prompt"]
            ),
            "dataset_target": str(
                row["target"]
            ),
            "attack_target": result[
                "target_text"
            ],
            "target_probe_loss": result[
                "target_probe_loss"
            ],
            "suffix_samples": (
                SUFFIX_SAMPLES
            ),
            "sample_rate": (
                SAMPLE_RATE
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "max_steps": (
                MAX_STEPS
            ),
            "early_stop_loss": (
                EARLY_STOP_LOSS
            ),
            "steps": result[
                "steps"
            ],
            "final_loss": result[
                "final_loss"
            ],
            "converged": result[
                "converged"
            ],
            "adversarial_audio": str(
                output_path
            ),
            "response": result[
                "response"
            ],
        }

        append_jsonl(
            result_file,
            record,
        )

        print(
            f"  steps={record['steps']} "
            f"loss={record['final_loss']:.6f} "
            f"converged={record['converged']}",
            flush=True,
        )

        del clean_waveform

        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("[OK] AdvWave complete")
    print(
        "Audios:",
        audio_dir,
    )
    print(
        "Responses:",
        result_file,
    )


if __name__ == "__main__":
    main()
