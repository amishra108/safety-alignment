from pathlib import Path
import argparse
import csv
import gc
import json
import os
import tempfile
import wave

import torch
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[3]

AUDIO_DATASET = ROOT / "data/xstest_audio/dataset"
TEXT_MANIFEST = ROOT / "data/xstest/manifest.csv"
RESULT_ROOT = ROOT / "results/xstest"

MODEL_PATHS = {
    "qwen2_audio": ROOT / "models/qwen2-audio-7b-instruct",
    "qwen25_omni": ROOT / "models/qwen2.5-omni-7b",
    "kimi_audio": ROOT / "models/kimi-audio-7b-instruct",
    "audio_flamingo3": ROOT / "models/audio-flamingo-3",
}


# ============================================================
# DATA
# ============================================================

def load_text_metadata():
    rows = []

    with TEXT_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if len(rows) != 450:
        raise RuntimeError(
            f"Expected 450 XSTest text rows, got {len(rows)}"
        )

    return rows


def get_audio_bytes(row):
    audio_bytes = row["audio_bytes"]

    if audio_bytes is None:
        raise RuntimeError(
            f'XSTest audio_bytes missing for {row["example_id"]}'
        )

    return audio_bytes, row["audio_path"]


def materialize_audio(row, example_id):
    audio_bytes, original_name = get_audio_bytes(row)

    suffix = Path(str(original_name)).suffix or ".wav"

    tmp_root = Path(
        os.environ.get(
            "TMPDIR",
            tempfile.gettempdir(),
        )
    )

    tmp_root.mkdir(parents=True, exist_ok=True)

    path = tmp_root / f"xstest_{example_id}{suffix}"
    path.write_bytes(audio_bytes)

    return path



def append_jsonl(path, record):
    """Canonical output: preserve the model response exactly."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def rebuild_readable_csv(jsonl_path, csv_path, fields):
    """
    Inspection copy only:
      - exactly one physical line per example
      - quote every field
      - encode actual CR/LF characters visibly as \\r / \\n
    Canonical evaluation data remain in JSONL.
    """
    with jsonl_path.open("r", encoding="utf-8") as src, \
         csv_path.open("w", newline="", encoding="utf-8") as dst:

        writer = csv.DictWriter(
            dst,
            fieldnames=fields,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for line in src:
            if not line.strip():
                continue

            record = json.loads(line)

            readable = dict(record)

            for k, v in readable.items():
                if isinstance(v, str):
                    readable[k] = (
                        v.replace("\r", "\\r")
                         .replace("\n", "\\n")
                    )

            writer.writerow(readable)

# ============================================================
# QWEN2-AUDIO
# ============================================================

def load_qwen2_audio():
    import librosa

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

    def move_inputs(inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
        return inputs

    def infer_text(prompt):
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

        inputs = move_inputs(inputs)

        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_length=1024,
                do_sample=False,
            )

        ids = ids[:, inputs["input_ids"].shape[1]:]

        return processor.batch_decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def infer_audio(audio_path):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "file:" + str(audio_path),
                    }
                ],
            }
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        wav, _ = librosa.load(
            str(audio_path),
            sr=processor.feature_extractor.sampling_rate,
        )

        inputs = processor(
            text=text,
            audios=[wav],
            return_tensors="pt",
            padding=True,
            sampling_rate=processor.feature_extractor.sampling_rate,
        )

        inputs = move_inputs(inputs)

        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_length=1024,
                do_sample=False,
            )

        ids = ids[:, inputs["input_ids"].shape[1]:]

        return processor.batch_decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    return model, infer_text, infer_audio


# ============================================================
# QWEN2.5-OMNI
# ============================================================

def load_qwen25_omni():
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    from qwen_omni_utils import process_mm_info

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

    def infer_text(prompt):
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

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                max_new_tokens=1024,
            )

        if isinstance(out, tuple):
            out = out[0]

        out = out[:, inputs["input_ids"].shape[1]:]

        return processor.batch_decode(
            out,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def infer_audio(audio_path):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": "file://" + str(audio_path.resolve()),
                    }
                ],
            }
        ]

        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=False,
        )

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        inputs = inputs.to(model.device).to(model.dtype)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                use_audio_in_video=False,
                max_new_tokens=1024,
            )

        if isinstance(out, tuple):
            out = out[0]

        out = out[:, inputs["input_ids"].shape[1]:]

        return processor.batch_decode(
            out,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    return model, infer_text, infer_audio


# ============================================================
# KIMI-AUDIO
#
# IMPORTANT:
# Kimi's "text" condition is TEXT + FIXED 1s SILENCE because
# its published inference implementation expects audio features.
# ============================================================

def load_kimi_audio():
    from kimia_infer.api.kimia import KimiAudio

    path = str(MODEL_PATHS["kimi_audio"])

    model = KimiAudio(
        model_path=path,
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
        "max_new_tokens": 1024,
    }

    silence_path = Path(
        os.environ.get(
            "TMPDIR",
            tempfile.gettempdir(),
        )
    ) / "kimi_text_control_silence_1s_16khz.wav"

    if not silence_path.exists():
        silence_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sr = 16000

        with wave.open(
            str(silence_path),
            "wb",
        ) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(
                b"\x00\x00" * sr
            )

    def run(messages):
        _, text = model.generate(
            messages,
            output_type="text",
            **sampling_params,
        )

        return str(text).strip()

    def infer_text(prompt):
        messages = [
            {
                "role": "user",
                "message_type": "text",
                "content": prompt,
            },
            {
                "role": "user",
                "message_type": "audio",
                "content": str(silence_path),
            },
        ]

        return run(messages)

    def infer_audio(audio_path):
        messages = [
            {
                "role": "user",
                "message_type": "audio",
                "content": str(audio_path),
            }
        ]

        return run(messages)

    return model, infer_text, infer_audio


# ============================================================
# AUDIO FLAMINGO 3
# ============================================================

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
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    def run(conversation):
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        outputs = outputs[:, input_len:]

        return processor.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

    def infer_text(prompt):
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

        return run(conversation)

    def infer_audio(audio_path, af3_audio_mode="prompted"):
        if af3_audio_mode == "bare":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": str(audio_path),
                        },
                    ],
                }
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Follow the spoken user's request and provide your response "
                                "to that request. Do not transcribe or repeat the audio."
                            ),
                        },
                        {
                            "type": "audio",
                            "path": str(audio_path),
                        },
                    ],
                }
            ]

        return run(conversation)

    return model, infer_text, infer_audio


# ============================================================
# DISPATCH
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_PATHS),
    )

    parser.add_argument(
        "--modality",
        choices=[
            "text",
            "audio",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--af3-audio-mode",
        choices=["prompted", "bare"],
        default="prompted",
    )

    args = parser.parse_args()

    if not AUDIO_DATASET.exists():
        raise FileNotFoundError(AUDIO_DATASET)

    if not TEXT_MANIFEST.exists():
        raise FileNotFoundError(TEXT_MANIFEST)

    ds = load_from_disk(
        str(AUDIO_DATASET)
    )

    metadata = load_text_metadata()

    if len(ds) != 450:
        raise RuntimeError(
            f"Expected 450 XSTest audio rows, got {len(ds)}"
        )

    # Join by example_id when possible, otherwise require exact
    # row-order/prompt agreement.
    meta_by_id = {
        str(r["example_id"]): r
        for r in metadata
    }

    examples = []

    for i, audio_row in enumerate(ds):
        eid = str(audio_row["example_id"])

        if eid not in meta_by_id:
            raise RuntimeError(
                f"Missing exact XSTest metadata for {eid}"
            )

        meta = meta_by_id[eid]

        prompt = str(audio_row["prompt"])

        if prompt != str(meta["prompt"]):
            raise RuntimeError(
                f"Prompt mismatch at row {i}: "
                f"{prompt!r} != {meta['prompt']!r}"
            )

        examples.append(
            (
                eid,
                prompt,
                str(meta["type"]),
                str(meta["is_safe"]),
                audio_row,
            )
        )

    if args.limit is not None:
        examples = examples[
            : min(
                args.limit,
                len(examples),
            )
        ]

    result_model_name = args.model

    if (
        args.model == "audio_flamingo3"
        and args.af3_audio_mode == "bare"
    ):
        result_model_name = "audio_flamingo3_true"

    outdir = RESULT_ROOT / result_model_name
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outfile = outdir / "responses.csv"
    summaryfile = outdir / "summary.json"
    jsonlfile = outdir / "responses.jsonl"

    completed = set()

    if jsonlfile.exists():
        with jsonlfile.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                completed.add(
                    (
                        str(row["example_id"]),
                        str(row["modality"]),
                    )
                )

        print(
            f"[resume] {len(completed)} completed rows from JSONL",
            flush=True,
        )

    model, infer_text, infer_audio = load_model(
        args.model
    )

    modalities = (
        ["text", "audio"]
        if args.modality == "both"
        else [args.modality]
    )

    fields = [
        "example_id",
        "model",
        "modality",
        "prompt",
        "type",
        "is_safe",
        "response",
    ]

    write_header = not outfile.exists()

    with outfile.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        if write_header:
            writer.writeheader()
            f.flush()

        total_jobs = (
            len(examples)
            * len(modalities)
        )

        job_i = 0

        for (
            eid,
            prompt,
            xtype,
            is_safe,
            audio_row,
        ) in examples:

            for modality in modalities:

                job_i += 1

                key = (
                    eid,
                    modality,
                )

                if key in completed:
                    print(
                        f"[{job_i:04d}/{total_jobs:04d}] "
                        f"skip {eid} {modality}",
                        flush=True,
                    )
                    continue

                print(
                    f"[{job_i:04d}/{total_jobs:04d}] "
                    f"{args.model} {eid} {modality}",
                    flush=True,
                )

                if modality == "text":
                    response = infer_text(
                        prompt
                    )

                else:
                    audio_path = materialize_audio(
                        audio_row,
                        eid,
                    )

                    try:
                        if args.model == "audio_flamingo3":
                            response = infer_audio(
                                audio_path,
                                args.af3_audio_mode,
                            )
                        else:
                            response = infer_audio(
                                audio_path
                            )

                    finally:
                        try:
                            audio_path.unlink()
                        except FileNotFoundError:
                            pass

                record = {
                    "example_id": eid,
                    "model": args.model,
                    "modality": modality,
                    "prompt": prompt,
                    "type": xtype,
                    "is_safe": is_safe,
                    "response": response,
                }

                append_jsonl(jsonlfile, record)

    rebuild_readable_csv(jsonlfile, outfile, fields)

    counts = {
        "text": 0,
        "audio": 0,
    }

    safe_counts = {
        "text": 0,
        "audio": 0,
    }

    unsafe_counts = {
        "text": 0,
        "audio": 0,
    }

    with outfile.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        for row in csv.DictReader(f):
            modality = row["modality"]

            if modality not in counts:
                continue

            counts[modality] += 1

            safe = str(
                row["is_safe"]
            ).lower() in {
                "true",
                "1",
                "yes",
            }

            if safe:
                safe_counts[modality] += 1
            else:
                unsafe_counts[modality] += 1

    summary = {
        "benchmark": "XSTest",
        "model": args.model,
        "expected_examples": 450,
        "expected_safe": 250,
        "expected_unsafe": 200,
        "text_rows": counts["text"],
        "audio_rows": counts["audio"],
        "safe_rows": safe_counts,
        "unsafe_rows": unsafe_counts,
        "judge_applied": False,
        "judge": (
            "Official XSTest completion classifier "
            "to be applied separately"
        ),
        "kimi_text_condition": (
            "text_plus_fixed_1s_silence"
            if args.model == "kimi_audio"
            else "text_only"
        ),
    }

    summaryfile.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("================================")
    print("[OK]", args.model)
    print("================================")
    print("Text rows:", counts["text"])
    print("Audio rows:", counts["audio"])
    print("Responses:", outfile)

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
