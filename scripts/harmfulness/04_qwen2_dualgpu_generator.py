#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import os
import socket
import tempfile
import wave
from pathlib import Path

import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

MANIFEST = (
    ROOT
    / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"
)

RESULT_ROOT = (
    ROOT
    / "results/audio_jailbreak_clean_benchmark"
)

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

MODELS = tuple(MODEL_PATHS)
MODALITIES = ("text", "audio")

EXPECTED_ROWS = 3200
MAX_NEW_TOKENS = 1024


# ============================================================
# DATA
# ============================================================

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    if not path.exists():
        return []

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSONL {path}:{ln}"
                ) from exc

    return rows


def append_jsonl(path, row):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def load_examples():
    rows = read_csv(MANIFEST)

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows; "
            f"found {len(rows)}"
        )

    required = {
        "example_id",
        "english_example_id",
        "source_id",
        "split",
        "condition",
        "language",
        "source_text",
        "english_prompt_text",
        "prompt_text",
        "translated_text",
        "translation_model",
        "translation_seed",
        "paper_faithful_condition",
        "audio_path",
        "tts_model",
        "tts_language",
        "tts_style_prompt",
        "tts_batch_size",
        "sample_rate",
        "duration_seconds",
    }

    seen = set()

    for i, row in enumerate(rows, 1):
        missing = required - set(row)

        if missing:
            raise RuntimeError(
                f"{MANIFEST}:{i} missing "
                f"{sorted(missing)}"
            )

        eid = row["example_id"]

        if eid in seen:
            raise RuntimeError(
                f"Duplicate example_id: {eid}"
            )

        seen.add(eid)

        if row["split"] not in {
            "harmful",
            "benign",
        }:
            raise RuntimeError(
                f"Bad split for {eid}: "
                f"{row['split']}"
            )

        if row["condition"] not in {
            "original",
            "literal",
            "narrative_framing",
            "content_dilution",
        }:
            raise RuntimeError(
                f"Bad condition for {eid}: "
                f"{row['condition']}"
            )

        if row["language"] not in {
            "english",
            "spanish",
            "chinese",
            "russian",
        }:
            raise RuntimeError(
                f"Bad language for {eid}: "
                f"{row['language']}"
            )

        if not row["prompt_text"].strip():
            raise RuntimeError(
                f"Empty prompt_text for {eid}"
            )

        wav = ROOT / row["audio_path"]

        if (
            not wav.exists()
            or wav.stat().st_size <= 44
        ):
            raise RuntimeError(
                f"Missing/invalid WAV for {eid}: "
                f"{wav}"
            )

    expected_languages = {
        "english": 800,
        "spanish": 800,
        "chinese": 800,
        "russian": 800,
    }

    actual_languages = {
        lang: sum(
            r["language"] == lang
            for r in rows
        )
        for lang in expected_languages
    }

    if actual_languages != expected_languages:
        raise RuntimeError(
            f"Language counts wrong: "
            f"{actual_languages}"
        )

    return rows


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

    print(f"[load] Qwen2-Audio: {path}", flush=True)

    processor = AutoProcessor.from_pretrained(
        path,
        local_files_only=True,
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    device_map = getattr(model, "hf_device_map", None)
    print(
        f"[QWEN2-AUDIO] device_map={device_map} "
        f"dtype={model.dtype} "
        f"gpu_allocated={torch.cuda.memory_allocated()/1024**3:.2f} GiB",
        flush=True,
    )

    if device_map is not None:
        bad = {
            k: v for k, v in device_map.items()
            if str(v) in {"cpu", "disk", "meta"}
        }
        if bad:
            raise RuntimeError(
                f"Qwen2-Audio CPU/disk offload detected: {bad}"
            )

    def move_inputs(inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
        return inputs

    def decode_batch(ids):
        return [
            x.strip()
            for x in processor.batch_decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

    def infer_text(rows):
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": row["prompt_text"],
                        }
                    ],
                }
            ]
            for row in rows
        ]

        texts = [
            processor.apply_chat_template(
                conv,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conv in conversations
        ]

        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        inputs = move_inputs(inputs)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        return decode_batch(
            ids[:, input_len:]
        )

    def infer_audio(rows):
        conversations = []
        audios = []

        sr = processor.feature_extractor.sampling_rate

        for row in rows:
            audio_path = ROOT / row["audio_path"]

            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio",
                                "audio_url":
                                    "file:" + str(audio_path),
                            }
                        ],
                    }
                ]
            )

            wav, _ = librosa.load(
                str(audio_path),
                sr=sr,
            )

            audios.append(wav)

        texts = [
            processor.apply_chat_template(
                conv,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conv in conversations
        ]

        inputs = processor(
            text=texts,
            audio=audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        inputs = move_inputs(inputs)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        return decode_batch(
            ids[:, input_len:]
        )

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
    print(f"[load] Qwen2.5-Omni: {path}", flush=True)

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

    def decode(out, input_len):
        if isinstance(out, tuple):
            out = out[0]
        return processor.batch_decode(
            out[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def infer_one_text(row):
        conversation = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": row["prompt_text"],
            }],
        }]

        chat_text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=chat_text,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        return decode(out, input_len)

    def infer_one_audio(row):
        audio_path = (ROOT / row["audio_path"]).resolve()

        conversation = [{
            "role": "user",
            "content": [{
                "type": "audio",
                "audio": "file://" + str(audio_path),
            }],
        }]

        chat_text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=False,
        )

        inputs = processor(
            text=chat_text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                use_audio_in_video=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        return decode(out, input_len)

    def infer_text(rows):
        return [infer_one_text(row) for row in rows]

    def infer_audio(rows):
        return [infer_one_audio(row) for row in rows]

    return model, infer_text, infer_audio


# ============================================================
# KIMI-AUDIO
# ============================================================

def load_kimi_audio():
    from kimia_infer.api.kimia import (
        KimiAudio,
    )

    path = str(
        MODEL_PATHS["kimi_audio"]
    )

    print(
        f"[load] Kimi-Audio: {path}",
        flush=True,
    )

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

        "max_new_tokens":
            MAX_NEW_TOKENS,
    }

    silence_path = (
        Path(
            os.environ.get(
                "TMPDIR",
                tempfile.gettempdir(),
            )
        )
        / "kimi_text_control_silence_1s_16khz.wav"
    )

    if not silence_path.exists():
        silence_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sample_rate = 16000

        with wave.open(
            str(silence_path),
            "wb",
        ) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(
                sample_rate
            )
            wf.writeframes(
                b"\x00\x00"
                * sample_rate
            )

    def run(messages):
        _, text = model.generate(
            messages,
            output_type="text",
            **sampling_params,
        )

        return str(text).strip()

    def infer_text(rows):
        outputs = []

        for row in rows:
            messages = [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": row["prompt_text"],
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": str(silence_path),
                },
            ]

            outputs.append(
                run(messages)
            )

        return outputs

    def infer_audio(rows):
        outputs = []

        for row in rows:
            messages = [
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": str(
                        ROOT / row["audio_path"]
                    ),
                }
            ]

            outputs.append(
                run(messages)
            )

        return outputs

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

    print(f"[load] Audio Flamingo 3: {path}", flush=True)

    processor = AutoProcessor.from_pretrained(
        path,
        local_files_only=True,
    )

    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    device_map = getattr(model, "hf_device_map", None)
    print(
        f"[AF3] device_map={device_map} "
        f"dtype={model.dtype} "
        f"gpu_allocated={torch.cuda.memory_allocated()/1024**3:.2f} GiB",
        flush=True,
    )

    if device_map is not None:
        bad = {
            k: v for k, v in device_map.items()
            if str(v) in {"cpu", "disk", "meta"}
        }
        if bad:
            raise RuntimeError(f"AF3 CPU/disk offload detected: {bad}")

    def run(conversations):
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "text_kwargs": {
                    "padding": True,
                },
                "audio_kwargs": {
                    "return_attention_mask": True,
                },
                "common_kwargs": {
                    "return_tensors": "pt",
                    "padding_side": "left",
                },
            },
        )

        moved = {}

        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                moved[k] = v
            elif torch.is_floating_point(v):
                moved[k] = v.to(
                    model.device,
                    dtype=model.dtype,
                )
            else:
                moved[k] = v.to(
                    model.device
                )

        input_len = moved["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **moved,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        outputs = outputs[:, input_len:]

        return [
            x.strip()
            for x in processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]

    def infer_text(rows):
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": row["prompt_text"],
                        }
                    ],
                }
            ]
            for row in rows
        ]

        return run(conversations)

    def infer_audio(rows):
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Follow the instruction spoken in the audio and answer it.",
                        },
                        {
                            "type": "audio",
                            "path": str(
                                ROOT / row["audio_path"]
                            ),
                        }
                    ],
                }
            ]
            for row in rows
        ]

        return run(conversations)

    return model, infer_text, infer_audio


# ============================================================
# MODEL DISPATCH
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
# OUTPUT VALIDATION
# ============================================================

def output_path(model, modality):
    return (
        RESULT_ROOT
        / model
        / f"responses.all.{modality}.jsonl"
    )


def validate_existing(
    existing,
    examples_by_id,
    model_name,
    modality,
    path,
):
    completed = set()

    for i, row in enumerate(
        existing,
        start=1,
    ):
        required = {
            "task_id",
            "model",
            "example_id",
            "modality",
            "response",
        }

        missing = required - set(row)

        if missing:
            raise RuntimeError(
                f"{path}:{i} missing "
                f"{sorted(missing)}"
            )

        eid = row["example_id"]

        if eid not in examples_by_id:
            raise RuntimeError(
                f"{path}:{i} unknown "
                f"example_id={eid}"
            )

        if row["model"] != model_name:
            raise RuntimeError(
                f"{path}:{i} wrong model"
            )

        if row["modality"] != modality:
            raise RuntimeError(
                f"{path}:{i} wrong modality"
            )

        expected_task = (
            eid + "__" + modality
        )

        if row["task_id"] != expected_task:
            raise RuntimeError(
                f"{path}:{i} wrong task_id"
            )

        if eid in completed:
            raise RuntimeError(
                f"{path}: duplicate {eid}"
            )

        completed.add(eid)

    return completed


# ============================================================
# RUNNER
# ============================================================


# ============================================================
# SHARDED RUNNER
# ============================================================

def shard_bounds(n, num_shards, shard_index):
    start = (n * shard_index) // num_shards
    end = (n * (shard_index + 1)) // num_shards
    return start, end


def result_paths(model_name, num_shards, shard_index):
    outdir = RESULT_ROOT / model_name
    shard_dir = outdir / "shards"

    shard = (
        shard_dir
        / f"responses.shard{shard_index:03d}-of-{num_shards:03d}.jsonl"
    )

    done = (
        shard_dir
        / f"shard{shard_index:03d}-of-{num_shards:03d}.done"
    )

    return outdir, shard_dir, shard, done


def all_shard_paths(model_name, num_shards):
    outdir = RESULT_ROOT / model_name
    shard_dir = outdir / "shards"

    shard_files = [
        shard_dir
        / f"responses.shard{i:03d}-of-{num_shards:03d}.jsonl"
        for i in range(num_shards)
    ]

    done_files = [
        shard_dir
        / f"shard{i:03d}-of-{num_shards:03d}.done"
        for i in range(num_shards)
    ]

    return outdir, shard_files, done_files


def modalities_for_arg(value):
    return (
        ("text", "audio")
        if value == "both"
        else (value,)
    )


def make_record(model_name, modality, row, response):
    return {
        "task_id":
            row["example_id"]
            + "__"
            + modality,

        "benchmark":
            "Audio Jailbreak Clean Benchmark",

        "model":
            model_name,

        "host":
            socket.gethostname(),

        "example_id":
            row["example_id"],

        "english_example_id":
            row["english_example_id"],

        "source_id":
            row["source_id"],

        "split":
            row["split"],

        "condition":
            row["condition"],

        "attack_type":
            (
                "no_attack"
                if row["condition"] == "original"
                else row["condition"]
            ),

        "modality":
            modality,

        "model_input_mode":
            (
                "text_plus_fixed_1s_silence"
                if (
                    model_name == "kimi_audio"
                    and modality == "text"
                )
                else (
                    "text_only"
                    if modality == "text"
                    else "audio_only"
                )
            ),

        "language":
            row["language"],

        "source_text":
            row["source_text"],

        "english_prompt_text":
            row["english_prompt_text"],

        # This is the ONLY textual request supplied to target
        # models in the text condition.
        "prompt_text":
            row["prompt_text"],

        "translated_text":
            row["translated_text"] or None,

        "translation_model":
            row["translation_model"] or None,

        "translation_seed":
            row["translation_seed"] or None,

        "audio_path":
            (
                row["audio_path"]
                if modality == "audio"
                else None
            ),

        "tts_model":
            (
                row["tts_model"]
                if modality == "audio"
                else None
            ),

        "tts_language":
            (
                row["tts_language"]
                if modality == "audio"
                else None
            ),

        "tts_style_prompt":
            (
                row["tts_style_prompt"]
                if modality == "audio"
                else None
            ),

        "audio_sample_rate":
            (
                int(row["sample_rate"])
                if modality == "audio"
                else None
            ),

        "audio_duration_seconds":
            (
                float(row["duration_seconds"])
                if modality == "audio"
                else None
            ),

        "generation_max_new_tokens":
            MAX_NEW_TOKENS,

        "generation_decoding":
            "greedy",

        "target_system_prompt":
            None,

        "target_extra_instruction":
            None,

        "response":
            response,
    }


def validate_shard_existing(path, expected_keys):
    rows = read_jsonl(path)

    seen = set()

    for i, row in enumerate(rows, 1):
        key = (
            str(row.get("example_id")),
            str(row.get("modality")),
        )

        if key not in expected_keys:
            raise RuntimeError(
                f"{path}:{i}: result does not belong "
                f"to this shard: {key}"
            )

        if key in seen:
            raise RuntimeError(
                f"{path}: duplicate result {key}"
            )

        seen.add(key)

    return seen


def merge_outputs(
    model_name,
    examples,
    modalities,
    num_shards,
):
    outdir, shard_files, done_files = (
        all_shard_paths(
            model_name,
            num_shards,
        )
    )

    waiting = [
        i
        for i, path in enumerate(done_files)
        if not path.exists()
    ]

    if waiting:
        print(
            f"[merge] waiting for shards: {waiting}",
            flush=True,
        )
        return False

    order = {
        (
            row["example_id"],
            modality,
        ): (
            example_i,
            modality_i,
        )
        for example_i, row in enumerate(examples)
        for modality_i, modality in enumerate(modalities)
    }

    merged = {}

    for path in shard_files:
        if not path.exists():
            raise RuntimeError(
                f"Done marker exists but shard missing: {path}"
            )

        for row in read_jsonl(path):
            key = (
                str(row["example_id"]),
                str(row["modality"]),
            )

            if key not in order:
                raise RuntimeError(
                    f"Unexpected merged result: {key}"
                )

            if key in merged:
                raise RuntimeError(
                    f"Duplicate merged result: {key}"
                )

            merged[key] = row

    missing = set(order) - set(merged)

    if missing:
        raise RuntimeError(
            f"Merge missing {len(missing)} tasks; "
            f"examples={sorted(missing)[:10]}"
        )

    ordered = sorted(
        merged.values(),
        key=lambda r: order[
            (
                str(r["example_id"]),
                str(r["modality"]),
            )
        ],
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for modality in modalities:
        records = [
            r
            for r in ordered
            if r["modality"] == modality
        ]

        canonical = (
            outdir
            / f"responses.all.{modality}.jsonl"
        )

        tmp = canonical.with_name(
            canonical.name
            + f".tmp.{os.getpid()}"
        )

        with tmp.open(
            "w",
            encoding="utf-8",
        ) as f:
            for row in records:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        os.replace(
            tmp,
            canonical,
        )

        expected_n = len(examples)

        if len(records) != expected_n:
            raise RuntimeError(
                f"{canonical}: expected "
                f"{expected_n}, got {len(records)}"
            )

        print(
            f"[merged] {canonical}: "
            f"{len(records)} rows",
            flush=True,
        )

    return True


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        required=True,
        choices=MODELS,
    )

    p.add_argument(
        "--modality",
        choices=[
            "text",
            "audio",
            "both",
        ],
        default="both",
    )

    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
    )

    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
    )

    p.add_argument(
        "--merge-only",
        action="store_true",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults: Qwen2=4, Omni=2, AF3=2, Kimi=1.",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    p.add_argument(
        "--preflight",
        action="store_true",
    )

    args = p.parse_args()

    if args.num_shards < 1:
        raise RuntimeError(
            "--num-shards must be >= 1"
        )

    examples = load_examples()

    if args.limit is not None:
        if args.limit < 1:
            raise RuntimeError(
                "--limit must be >= 1"
            )

        examples = examples[
            :args.limit
        ]

    modalities = modalities_for_arg(
        args.modality
    )

    default_batch_sizes = {
        "qwen2_audio": 16,
        "qwen25_omni": 16,
        "kimi_audio": 1,
        "audio_flamingo3": 16,
    }

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else default_batch_sizes[args.model]
    )

    if batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")

    if args.model == "kimi_audio" and batch_size != 1:
        raise RuntimeError(
            "KimiAudio's released generate() API is single-example; "
            "use --batch-size 1 for Kimi."
        )

    print()
    print("=" * 80)
    print("CLEAN AUDIO-JAILBREAK BENCHMARK")
    print("=" * 80)
    print("manifest    :", MANIFEST)
    print("model       :", args.model)
    print("modality    :", args.modality)
    print("examples    :", len(examples))
    print("num shards  :", args.num_shards)
    print("max tokens  :", MAX_NEW_TOKENS)
    print("batch size  :", batch_size)
    print("extra prompt: NONE")

    if (
        args.model == "kimi_audio"
        and "text" in modalities
    ):
        print(
            "Kimi text   : prompt_text + "
            "fixed 1 s silence"
        )

    print()

    if args.preflight:
        for i in range(
            args.num_shards
        ):
            start, end = shard_bounds(
                len(examples),
                args.num_shards,
                i,
            )

            print(
                f"shard {i:03d}: "
                f"examples {start}:{end} "
                f"({end-start}), "
                f"generations="
                f"{(end-start)*len(modalities)}"
            )

        return

    if args.merge_only:
        merge_outputs(
            model_name=args.model,
            examples=examples,
            modalities=modalities,
            num_shards=args.num_shards,
        )
        return

    if args.shard_index is None:
        if args.num_shards == 1:
            shard_index = 0
        else:
            raise RuntimeError(
                "With --num-shards > 1, "
                "provide --shard-index."
            )
    else:
        shard_index = args.shard_index

    if not (
        0
        <= shard_index
        < args.num_shards
    ):
        raise RuntimeError(
            f"Invalid shard index "
            f"{shard_index}/"
            f"{args.num_shards}"
        )

    start, end = shard_bounds(
        len(examples),
        args.num_shards,
        shard_index,
    )

    assigned = examples[
        start:end
    ]

    outdir, shard_dir, outfile, donefile = (
        result_paths(
            args.model,
            args.num_shards,
            shard_index,
        )
    )

    shard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_keys = {
        (
            row["example_id"],
            modality,
        )
        for row in assigned
        for modality in modalities
    }

    completed = validate_shard_existing(
        outfile,
        expected_keys,
    )

    pending_n = (
        len(expected_keys)
        - len(completed)
    )

    print(
        f"[shard]     "
        f"{shard_index}/{args.num_shards}"
    )
    print(
        f"[range]     {start}:{end}"
    )
    print(
        f"[examples]  {len(assigned)}"
    )
    print(
        f"[jobs]      {len(expected_keys)}"
    )
    print(
        f"[completed] {len(completed)}"
    )
    print(
        f"[pending]   {pending_n}"
    )
    print(
        f"[output]    {outfile}"
    )

    if pending_n == 0:
        donefile.write_text(
            "complete\n",
            encoding="utf-8",
        )

        print(
            "[COMPLETE] shard already done"
        )

        merge_outputs(
            model_name=args.model,
            examples=examples,
            modalities=modalities,
            num_shards=args.num_shards,
        )

        return

    model, infer_text, infer_audio = (
        load_model(args.model)
    )

    try:
        jobs = []

        for row in assigned:
            for modality in modalities:
                key = (
                    row["example_id"],
                    modality,
                )

                if key not in completed:
                    jobs.append(
                        (
                            row,
                            modality,
                        )
                    )

        # Batch only like-with-like. This avoids mixing text and
        # audio preprocessing in a single model invocation.
        for modality in modalities:
            modality_jobs = [
                row
                for row, mod in jobs
                if mod == modality
            ]

            infer = (
                infer_text
                if modality == "text"
                else infer_audio
            )

            requested_batch_size = batch_size
            current_batch_size = batch_size
            pos = 0
            batch_i = 0

            while pos < len(modality_jobs):
                n = min(
                    current_batch_size,
                    len(modality_jobs) - pos,
                )

                batch = modality_jobs[
                    pos:pos + n
                ]

                batch_i += 1

                print(
                    f"[batch {batch_i:04d}] "
                    f"shard={shard_index:02d} | "
                    f"{modality:5s} | "
                    f"n={len(batch)} | "
                    f"requested={requested_batch_size} | "
                    f"{batch[0]['example_id']} ...",
                    flush=True,
                )

                try:
                    responses = infer(batch)

                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if current_batch_size <= 1:
                        print(
                            "[OOM] batch size 1 also OOM; aborting",
                            flush=True,
                        )
                        raise

                    new_batch_size = max(
                        1,
                        current_batch_size // 2,
                    )

                    print(
                        f"[OOM] {modality} batch "
                        f"{current_batch_size} -> "
                        f"{new_batch_size}; retrying same examples",
                        flush=True,
                    )

                    current_batch_size = new_batch_size
                    continue

                if isinstance(responses, str):
                    responses = [responses]

                if len(responses) != len(batch):
                    raise RuntimeError(
                        f"Batch returned {len(responses)} responses "
                        f"for {len(batch)} examples"
                    )

                for row, response in zip(
                    batch,
                    responses,
                ):
                    key = (
                        row["example_id"],
                        modality,
                    )

                    record = make_record(
                        model_name=args.model,
                        modality=modality,
                        row=row,
                        response=response,
                    )

                    append_jsonl(
                        outfile,
                        record,
                    )

                    completed.add(key)

                pos += len(batch)

                # Gradually probe upward again after a successful
                # reduced batch. Never exceed requested batch size.
                if current_batch_size < requested_batch_size:
                    grown = min(
                        requested_batch_size,
                        current_batch_size * 2,
                    )

                    print(
                        f"[batch] success at {current_batch_size}; "
                        f"next try={grown}",
                        flush=True,
                    )

                    current_batch_size = grown

    finally:
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if completed != expected_keys:
        missing = (
            expected_keys
            - completed
        )

        raise RuntimeError(
            f"Shard incomplete: "
            f"{len(missing)} missing; "
            f"{sorted(missing)[:10]}"
        )

    donefile.write_text(
        "complete\n",
        encoding="utf-8",
    )

    print(
        f"[OK] shard "
        f"{shard_index}/{args.num_shards} "
        f"complete",
        flush=True,
    )

    merge_outputs(
        model_name=args.model,
        examples=examples,
        modalities=modalities,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
