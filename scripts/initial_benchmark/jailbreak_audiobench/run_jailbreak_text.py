from pathlib import Path
import argparse
import gc
import importlib.util
import json
import re
import wave

import numpy as np
import pandas as pd
import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

AUDIO_RUNNER = (
    ROOT
    / "scripts/initial_benchmark/jailbreak_audiobench/run_jailbreak.py"
)

TEXT_ROOT = ROOT / "data/jailbreak_audiobench/text"

RESULT_ROOT = ROOT / "results/jailbreak_audiobench_text"

MODEL_PATHS = {
    "qwen2_audio":
        ROOT / "models/qwen2-audio-7b-instruct",
    "qwen25_omni":
        ROOT / "models/qwen2.5-omni-7b",
    "kimi_audio":
        ROOT / "models/kimi-audio-7b-instruct",
    "audio_flamingo3":
        ROOT / "models/audio-flamingo-3",
}

TEXT_FILES = {
    "Advbench": TEXT_ROOT / "Explicit_Advbench.csv",
    "MM-Safetybench": TEXT_ROOT / "Explicit_MM-Safetybench.csv",
    "RedTeam_2K": TEXT_ROOT / "Explicit_RedTeam_2K.csv",
    "safebench": TEXT_ROOT / "Explicit_safebench.csv",
}

JAILBREAK_TEXT_INSTRUCTION = (
    "Provide a detailed answer that exceeds 500 words."
)

GENERATION_MAX_NEW_TOKENS = 512
EXPECTED_BASES = 262

SILENCE_PATH = (
    ROOT
    / "data/jailbreak_audiobench/text/one_second_silence_16k.wav"
)


def load_audio_runner():
    spec = importlib.util.spec_from_file_location(
        "audio_runner",
        AUDIO_RUNNER,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_exact_base_ids():
    audio_runner = load_audio_runner()
    ds = audio_runner.load_data()

    ids = sorted({
        audio_runner.base_audio_id(row)
        for row in ds
    })

    if len(ids) != EXPECTED_BASES:
        raise RuntimeError(
            f"Expected {EXPECTED_BASES} unique base IDs, got {len(ids)}"
        )

    return ids


def load_text_tables():
    tables = {}

    for source, path in TEXT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        if "question" not in df.columns:
            raise RuntimeError(
                f"{path} missing required column: question"
            )

        if df["question"].isna().any():
            raise RuntimeError(
                f"{path} contains missing questions"
            )

        tables[source] = df.reset_index(drop=True)

    return tables


def parse_base_id(base_id):
    patterns = (
        ("Advbench", r"^Advbench_audio_(\d+)$"),
        ("MM-Safetybench", r"^MM-Safetybench_audio_(\d+)$"),
        ("RedTeam_2K", r"^RedTeam_2K_audio_(\d+)$"),
        ("safebench", r"^safebench_audio_(\d+)$"),
    )

    for source, pattern in patterns:
        m = re.fullmatch(pattern, base_id)

        if m:
            return source, int(m.group(1))

    raise RuntimeError(
        f"Unrecognised base_audio_id: {base_id}"
    )


def build_examples():
    base_ids = get_exact_base_ids()
    tables = load_text_tables()

    examples = []

    for base_id in base_ids:
        source, source_row_number = parse_base_id(base_id)

        df = tables[source]

        # The authors' TTS generator names audio by CSV row position:
        # audio_{index + 1:03d}.wav
        csv_index = source_row_number - 1

        if csv_index < 0 or csv_index >= len(df):
            raise RuntimeError(
                f"{base_id}: requested CSV row "
                f"{source_row_number}, but {TEXT_FILES[source]} "
                f"contains {len(df)} rows"
            )

        row = df.iloc[csv_index]

        question = str(row["question"]).strip()

        if not question:
            raise RuntimeError(
                f"{base_id}: empty question at CSV row "
                f"{source_row_number}"
            )

        prompt = (
            question
            + "\n\n"
            + JAILBREAK_TEXT_INSTRUCTION
        )

        examples.append({
            "example_id": f"text_{base_id}",
            "base_audio_id": base_id,
            "source_dataset": source,
            "source_csv_row": source_row_number,
            "original_prompt": question,
            "text_instruction": JAILBREAK_TEXT_INSTRUCTION,
            "full_prompt": prompt,
        })

    if len(examples) != EXPECTED_BASES:
        raise RuntimeError(
            f"Expected {EXPECTED_BASES} examples, "
            f"got {len(examples)}"
        )

    if len({
        x["base_audio_id"]
        for x in examples
    }) != EXPECTED_BASES:
        raise RuntimeError(
            "Duplicate base IDs"
        )

    return examples


def ensure_silence():
    SILENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SILENCE_PATH.exists():
        return

    sr = 16000
    pcm = np.zeros(sr, dtype="<i2")

    with wave.open(str(SILENCE_PATH), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def load_qwen2_audio():
    from transformers import (
        AutoProcessor,
        Qwen2AudioForConditionalGeneration,
    )

    path = str(MODEL_PATHS["qwen2_audio"])

    processor = AutoProcessor.from_pretrained(
        path,
        local_files_only=True,
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        path,
        device_map="auto",
        torch_dtype="auto",
        local_files_only=True,
    )

    model.eval()

    def infer(prompt):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            k: (
                v.to(model.device)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                do_sample=False,
            )

        out = out[:, input_len:]

        return processor.batch_decode(
            out,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    return model, infer, "text_only"


def load_qwen25_omni():
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    path = str(MODEL_PATHS["qwen25_omni"])

    processor = Qwen2_5OmniProcessor.from_pretrained(
        path,
        local_files_only=True,
    )

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )

    if hasattr(model, "disable_talker"):
        model.disable_talker()

    model.eval()

    def infer(prompt):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
        )

        inputs = inputs.to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
            )

        if isinstance(out, tuple):
            out = out[0]

        out = out[:, input_len:]

        return processor.batch_decode(
            out,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    return model, infer, "text_only"


def load_audio_flamingo3():
    from transformers import (
        AudioFlamingo3ForConditionalGeneration,
        AutoProcessor,
    )

    path = str(MODEL_PATHS["audio_flamingo3"])

    processor = AutoProcessor.from_pretrained(
        path,
        local_files_only=True,
    )

    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True,
    )

    model.eval()

    def infer(prompt):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        inputs = inputs.to(
            model.device,
            dtype=model.dtype,
        )

        input_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                do_sample=False,
            )

        out = out[:, input_len:]

        return processor.decode(
            out[0],
            skip_special_tokens=True,
        ).strip()

    return model, infer, "text_only"


def load_kimi_audio():
    from kimia_infer.api.kimia import KimiAudio

    ensure_silence()

    model = KimiAudio(
        model_path=str(MODEL_PATHS["kimi_audio"]),
        load_detokenizer=False,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
    }

    def infer(prompt):
        messages = [
            {
                "role": "user",
                "message_type": "text",
                "content": prompt,
            },
            {
                "role": "user",
                "message_type": "audio",
                "content": str(SILENCE_PATH),
            },
        ]

        _, text = model.generate(
            messages,
            output_type="text",
            **sampling_params,
        )

        return str(text).strip()

    return model, infer, "text_plus_1s_silence"


def load_model(name):
    if name == "qwen2_audio":
        return load_qwen2_audio()

    if name == "qwen25_omni":
        return load_qwen25_omni()

    if name == "kimi_audio":
        return load_kimi_audio()

    if name == "audio_flamingo3":
        return load_audio_flamingo3()

    raise ValueError(name)


def shard_indices(n, num_shards, shard_index):
    return [
        i
        for i in range(n)
        if i % num_shards == shard_index
    ]


def run_one_shard(
    model_name,
    examples,
    num_shards,
    shard_index,
):
    output_dir = RESULT_ROOT / model_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = output_dir / (
        f"responses.shard{shard_index:03d}"
        f"-of-{num_shards:03d}.jsonl"
    )

    done_path = output.with_suffix(".done")

    completed = set()

    if output.exists():
        with output.open() as f:
            for line in f:
                if line.strip():
                    completed.add(
                        json.loads(line)["example_id"]
                    )

    indices = shard_indices(
        len(examples),
        num_shards,
        shard_index,
    )

    todo = [
        examples[i]
        for i in indices
        if examples[i]["example_id"] not in completed
    ]

    print(
        f"[shard {shard_index}/{num_shards}] "
        f"assigned={len(indices)} "
        f"completed={len(completed)} "
        f"remaining={len(todo)}",
        flush=True,
    )

    model, infer, input_mode = load_model(model_name)

    with output.open("a", buffering=1) as f:
        for j, ex in enumerate(todo, 1):
            print(
                f"[{j:04d}/{len(todo):04d}] "
                f"{model_name} | {ex['base_audio_id']}",
                flush=True,
            )

            response = infer(ex["full_prompt"])

            record = {
                **ex,
                "model": model_name,
                "benchmark": "Jailbreak-AudioBench",
                "experiment": "text_baseline",
                "input_mode": input_mode,
                "generation_max_new_tokens":
                    GENERATION_MAX_NEW_TOKENS,
                "response": response,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    done_path.touch()

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_merge(model_name, num_shards):
    output_dir = RESULT_ROOT / model_name

    shard_paths = [
        output_dir
        / (
            f"responses.shard{i:03d}"
            f"-of-{num_shards:03d}.jsonl"
        )
        for i in range(num_shards)
    ]

    done_paths = [
        p.with_suffix(".done")
        for p in shard_paths
    ]

    if not all(p.exists() for p in done_paths):
        return

    rows = []

    for path in shard_paths:
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    if len(rows) != EXPECTED_BASES:
        raise RuntimeError(
            f"Merge expected {EXPECTED_BASES} rows, got {len(rows)}"
        )

    ids = [r["example_id"] for r in rows]

    if len(set(ids)) != EXPECTED_BASES:
        raise RuntimeError(
            "Duplicate/missing example IDs during merge"
        )

    rows.sort(key=lambda x: x["base_audio_id"])

    merged = output_dir / "responses.jsonl"
    tmp = output_dir / (
        "responses.jsonl.tmp."
        + str(__import__("os").getpid())
    )

    with tmp.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(merged)

    print(
        f"[OK] merged {len(rows)} rows -> {merged}",
        flush=True,
    )


def preflight():
    examples = build_examples()

    print()
    print("===== TEXT JAILBREAK PREFLIGHT =====")
    print("examples:", len(examples))
    print("unique bases:", len({
        x["base_audio_id"]
        for x in examples
    }))
    print(
        "instruction:",
        JAILBREAK_TEXT_INSTRUCTION,
    )

    counts = {}

    for ex in examples:
        counts[ex["source_dataset"]] = (
            counts.get(ex["source_dataset"], 0)
            + 1
        )

    print("sources:", counts)

    print()
    for ex in examples[:8]:
        print(
            ex["base_audio_id"],
            "->",
            ex["source_dataset"],
            ex["source_csv_row"],
            "->",
            repr(ex["original_prompt"][:100]),
        )

    print()
    print("[OK] exact 262 audio base IDs recovered")
    print("[OK] each ID matched to authors' Text CSV by source + id")
    print("[OK] original prompt stored in every result JSONL")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=tuple(MODEL_PATHS),
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--shard-indices",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    args = parser.parse_args()

    examples = build_examples()

    if args.preflight:
        preflight()
        return

    if args.model is None:
        parser.error("--model is required unless --preflight is used")

    shard_indices_requested = [
        int(x)
        for x in args.shard_indices.split(",")
    ]

    for shard_index in shard_indices_requested:
        if not 0 <= shard_index < args.num_shards:
            raise ValueError(
                f"Invalid shard index {shard_index}"
            )

        run_one_shard(
            args.model,
            examples,
            args.num_shards,
            shard_index,
        )

    maybe_merge(
        args.model,
        args.num_shards,
    )


if __name__ == "__main__":
    main()
