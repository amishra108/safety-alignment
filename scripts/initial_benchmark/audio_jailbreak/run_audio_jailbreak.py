from pathlib import Path

import argparse
import csv
import gc
import json
import os
import socket
import subprocess
import sys
import tempfile
import wave

import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

BASE_MANIFEST = (
    ROOT / "data/audio_jailbreaks/manifests/base.csv"
)

SEMANTIC_MANIFEST = (
    ROOT / "data/audio_jailbreaks/manifests/semantic.csv"
)

AUDIO_MANIFEST = (
    ROOT / "data/audio_jailbreaks/manifests/semantic_audio.csv"
)

LANGUAGE_ABLATION_MANIFEST = (
    ROOT
    / "data/audio_jailbreaks/language_ablation/manifests/language_ablation.csv"
)

RESULT_ROOT = ROOT / "results/audio_jailbreak"

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

EXPECTED_BASE_ROWS = 200
EXPECTED_SEMANTIC_ROWS = 800
EXPECTED_AUDIO_ROWS = 800
EXPECTED_LANGUAGE_ABLATION_ROWS = 480

GENERATION_MAX_NEW_TOKENS = 1024
PAPER_CONTEXT_WINDOW = 4096


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


def parse_bool(value):
    value = str(value).strip().lower()

    if value == "true":
        return True

    if value == "false":
        return False

    raise RuntimeError(
        f"Expected boolean text true/false, found {value!r}"
    )


def load_examples():
    base_rows = read_csv(BASE_MANIFEST)
    semantic_rows = read_csv(SEMANTIC_MANIFEST)
    audio_rows = read_csv(AUDIO_MANIFEST)

    if len(base_rows) != EXPECTED_BASE_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASE_ROWS} base rows; "
            f"found {len(base_rows)}"
        )

    if len(semantic_rows) != EXPECTED_SEMANTIC_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_SEMANTIC_ROWS} semantic rows; "
            f"found {len(semantic_rows)}"
        )

    if len(audio_rows) != EXPECTED_AUDIO_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_AUDIO_ROWS} audio rows; "
            f"found {len(audio_rows)}"
        )

    required_base = {
        "source_id",
        "split",
        "source_row",
        "jbb_index",
        "goal",
        "target",
        "behavior",
        "category",
        "source",
    }

    required_semantic = {
        "example_id",
        "source_id",
        "split",
        "condition",
        "source_text",
        "prompt_text",
        "paper_faithful_condition",
    }

    required_audio = {
        "example_id",
        "source_id",
        "split",
        "condition",
        "source_text",
        "prompt_text",
        "paper_faithful_condition",
        "audio_path",
        "tts_model",
        "tts_language",
        "tts_style_prompt",
        "tts_batch_size",
        "sample_rate",
        "duration_seconds",
    }

    if set(base_rows[0]) != required_base:
        raise RuntimeError(
            "Unexpected base.csv schema.\n"
            f"Expected: {sorted(required_base)}\n"
            f"Found:    {sorted(base_rows[0])}"
        )

    if not required_semantic.issubset(
        semantic_rows[0].keys()
    ):
        raise RuntimeError(
            "semantic.csv is missing required fields.\n"
            f"Required: {sorted(required_semantic)}\n"
            f"Found:    {sorted(semantic_rows[0])}"
        )

    if not required_audio.issubset(
        audio_rows[0].keys()
    ):
        raise RuntimeError(
            "semantic_audio.csv is missing required fields.\n"
            f"Required: {sorted(required_audio)}\n"
            f"Found:    {sorted(audio_rows[0])}"
        )

    base_by_id = {}

    for row in base_rows:
        sid = str(row["source_id"])

        if sid in base_by_id:
            raise RuntimeError(
                f"Duplicate source_id in base.csv: {sid}"
            )

        base_by_id[sid] = row

    audio_by_id = {}

    for row in audio_rows:
        eid = str(row["example_id"])

        if eid in audio_by_id:
            raise RuntimeError(
                f"Duplicate example_id in audio manifest: {eid}"
            )

        audio_by_id[eid] = row

    examples = []
    seen = set()

    condition_counts = {
        "original": 0,
        "literal": 0,
        "narrative_framing": 0,
        "content_dilution": 0,
    }

    split_counts = {
        "harmful": 0,
        "benign": 0,
    }

    for row in semantic_rows:
        eid = str(row["example_id"])
        sid = str(row["source_id"])
        split = str(row["split"])
        condition = str(row["condition"])

        if eid in seen:
            raise RuntimeError(
                f"Duplicate semantic example_id: {eid}"
            )

        seen.add(eid)

        if sid not in base_by_id:
            raise RuntimeError(
                f"Missing base metadata for {sid}"
            )

        if eid not in audio_by_id:
            raise RuntimeError(
                f"Missing audio manifest row for {eid}"
            )

        base = base_by_id[sid]
        audio = audio_by_id[eid]

        if split not in split_counts:
            raise RuntimeError(
                f"Unexpected split for {eid}: {split}"
            )

        if condition not in condition_counts:
            raise RuntimeError(
                f"Unexpected condition for {eid}: {condition}"
            )

        if str(base["split"]) != split:
            raise RuntimeError(
                f"Base/semantic split mismatch for {eid}"
            )

        for field in (
            "source_id",
            "split",
            "condition",
            "source_text",
            "prompt_text",
            "paper_faithful_condition",
        ):
            if str(audio[field]) != str(row[field]):
                raise RuntimeError(
                    f"Semantic/audio mismatch for "
                    f"{eid} field={field!r}"
                )

        audio_path = ROOT / str(audio["audio_path"])

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Missing WAV for {eid}: {audio_path}"
            )

        if audio_path.stat().st_size <= 44:
            raise RuntimeError(
                f"Invalid or empty WAV for {eid}: {audio_path}"
            )

        condition_counts[condition] += 1
        split_counts[split] += 1

        examples.append({
            "example_id":
                eid,

            "source_id":
                sid,

            "split":
                split,

            "condition":
                condition,

            "source_text":
                str(row["source_text"]),

            "prompt_text":
                str(row["prompt_text"]),

            "paper_faithful_condition":
                parse_bool(
                    row["paper_faithful_condition"]
                ),

            "source_row":
                int(base["source_row"]),

            "jbb_index":
                int(base["jbb_index"]),

            "goal":
                str(base["goal"]),

            "target":
                str(base["target"]),

            "behavior":
                str(base["behavior"]),

            "category":
                str(base["category"]),

            "jbb_source":
                str(base["source"]),

            "audio_path":
                str(audio_path),

            "audio_path_relative":
                str(audio["audio_path"]),

            "tts_model":
                str(audio["tts_model"]),

            "tts_language":
                str(audio["tts_language"]),

            "tts_style_prompt":
                str(audio["tts_style_prompt"]),

            "tts_batch_size":
                int(audio["tts_batch_size"]),

            "sample_rate":
                int(audio["sample_rate"]),

            "duration_seconds":
                float(audio["duration_seconds"]),
        })

    expected_conditions = {
        "original": 200,
        "literal": 200,
        "narrative_framing": 200,
        "content_dilution": 200,
    }

    if condition_counts != expected_conditions:
        raise RuntimeError(
            f"Condition counts mismatch: {condition_counts}"
        )

    if split_counts != {
        "harmful": 400,
        "benign": 400,
    }:
        raise RuntimeError(
            f"Split counts mismatch: {split_counts}"
        )

    return examples



def load_language_ablation_examples():
    base_rows = read_csv(BASE_MANIFEST)
    rows = read_csv(LANGUAGE_ABLATION_MANIFEST)

    if len(base_rows) != EXPECTED_BASE_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_BASE_ROWS} base rows; "
            f"found {len(base_rows)}"
        )

    if len(rows) != EXPECTED_LANGUAGE_ABLATION_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_LANGUAGE_ABLATION_ROWS} "
            f"language-ablation rows; found {len(rows)}"
        )

    required = {
        "source_id",
        "split",
        "condition",
        "english_example_id",
        "english_text",
        "language",
        "translated_text",
        "translation_model",
        "translation_seed",
        "tts_supported",
        "audio_path",
        "sample_rate",
        "duration_seconds",
    }

    if not rows:
        raise RuntimeError(
            "language_ablation.csv is empty"
        )

    missing = (
        required
        - set(rows[0].keys())
    )

    if missing:
        raise RuntimeError(
            "language_ablation.csv is missing fields: "
            f"{sorted(missing)}"
        )

    base_by_id = {}

    for row in base_rows:
        sid = str(row["source_id"])

        if sid in base_by_id:
            raise RuntimeError(
                f"Duplicate source_id in base.csv: {sid}"
            )

        base_by_id[sid] = row

    expected_languages = {
        "spanish",
        "chinese",
        "russian",
    }

    language_counts = {
        language: 0
        for language in expected_languages
    }

    condition_counts = {
        "original": 0,
        "literal": 0,
        "narrative_framing": 0,
        "content_dilution": 0,
    }

    split_counts = {
        "harmful": 0,
        "benign": 0,
    }

    examples = []
    seen = set()

    for row in rows:
        sid = str(row["source_id"])
        split = str(row["split"])
        condition = str(row["condition"])
        language = str(row["language"])

        if sid not in base_by_id:
            raise RuntimeError(
                f"Missing base metadata for {sid}"
            )

        if language not in expected_languages:
            raise RuntimeError(
                f"Unexpected language: {language}"
            )

        if condition not in condition_counts:
            raise RuntimeError(
                f"Unexpected condition: {condition}"
            )

        if split not in split_counts:
            raise RuntimeError(
                f"Unexpected split: {split}"
            )

        base = base_by_id[sid]

        if str(base["split"]) != split:
            raise RuntimeError(
                f"Split mismatch for {sid}"
            )

        english_example_id = str(
            row["english_example_id"]
        )

        example_id = (
            english_example_id
            + "__"
            + language
        )

        if example_id in seen:
            raise RuntimeError(
                f"Duplicate language example_id: "
                f"{example_id}"
            )

        seen.add(example_id)

        translated_text = str(
            row["translated_text"]
        ).strip()

        if not translated_text:
            raise RuntimeError(
                f"Empty translated_text for {example_id}"
            )

        audio_rel = str(
            row["audio_path"]
        ).strip()

        if not audio_rel:
            raise RuntimeError(
                f"Missing audio_path for {example_id}"
            )

        audio_path = ROOT / audio_rel

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Missing language-ablation WAV "
                f"for {example_id}: {audio_path}"
            )

        if audio_path.stat().st_size <= 44:
            raise RuntimeError(
                f"Invalid WAV for {example_id}: "
                f"{audio_path}"
            )

        language_counts[language] += 1
        condition_counts[condition] += 1
        split_counts[split] += 1

        examples.append({
            "example_id":
                example_id,

            "english_example_id":
                english_example_id,

            "source_id":
                sid,

            "split":
                split,

            "condition":
                condition,

            "source_text":
                str(base["goal"]),

            "prompt_text":
                translated_text,

            "english_prompt_text":
                str(row["english_text"]),

            "translated_text":
                translated_text,

            "language":
                language,

            "translation_model":
                str(row["translation_model"]),

            "translation_seed":
                int(row["translation_seed"]),

            "paper_faithful_condition":
                False,

            "source_row":
                int(base["source_row"]),

            "jbb_index":
                int(base["jbb_index"]),

            "goal":
                str(base["goal"]),

            "target":
                str(base["target"]),

            "behavior":
                str(base["behavior"]),

            "category":
                str(base["category"]),

            "jbb_source":
                str(base["source"]),

            "audio_path":
                str(audio_path),

            "audio_path_relative":
                audio_rel,

            "tts_model":
                "Qwen3-TTS-12Hz-1.7B-VoiceDesign",

            "tts_language":
                language,

            "tts_style_prompt":
                None,

            "tts_batch_size":
                64,

            "sample_rate":
                int(row["sample_rate"]),

            "duration_seconds":
                float(row["duration_seconds"]),

            "experiment":
                "language_ablation",
        })

    if language_counts != {
        "spanish": 160,
        "chinese": 160,
        "russian": 160,
    }:
        raise RuntimeError(
            f"Language counts mismatch: "
            f"{language_counts}"
        )

    if condition_counts != {
        "original": 120,
        "literal": 120,
        "narrative_framing": 120,
        "content_dilution": 120,
    }:
        raise RuntimeError(
            f"Condition counts mismatch: "
            f"{condition_counts}"
        )

    if split_counts != {
        "harmful": 240,
        "benign": 240,
    }:
        raise RuntimeError(
            f"Split counts mismatch: "
            f"{split_counts}"
        )

    return examples


def load_dataset_examples(dataset):
    if dataset == "main":
        examples = load_examples()

        for row in examples:
            row["english_example_id"] = (
                row["example_id"]
            )
            row["english_prompt_text"] = (
                row["prompt_text"]
            )
            row["translated_text"] = None
            row["language"] = "english"
            row["translation_model"] = None
            row["translation_seed"] = None
            row["experiment"] = "semantic"

        return examples

    if dataset == "language_ablation":
        return load_language_ablation_examples()

    raise ValueError(dataset)

def filter_scope(examples, scope):
    if scope == "all":
        selected = examples

    elif scope == "paper":
        selected = [
            row
            for row in examples
            if row["paper_faithful_condition"]
        ]

    else:
        raise ValueError(scope)

    return selected


# ============================================================
# JSONL
# ============================================================

def append_jsonl(path, record):
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
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()


def read_jsonl(path):
    if not path.exists():
        return []

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for lineno, line in enumerate(
            f,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSONL "
                    f"{path}:{lineno}"
                ) from exc

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

    path = str(
        MODEL_PATHS["qwen2_audio"]
    )

    processor = AutoProcessor.from_pretrained(
        path,
        local_files_only=True,
    )

    model = (
        Qwen2AudioForConditionalGeneration
        .from_pretrained(
            path,
            device_map="auto",
            torch_dtype="auto",
            local_files_only=True,
        )
    )

    model.eval()

    def move_inputs(inputs):
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(
                    model.device
                )

        return inputs

    def infer_text(prompt):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        chat_text = (
            processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

        inputs = processor(
            text=chat_text,
            return_tensors="pt",
            padding=True,
        )

        inputs = move_inputs(inputs)

        input_len = (
            inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=(
                    GENERATION_MAX_NEW_TOKENS
                ),
                do_sample=False,
            )

        output = output[:, input_len:]

        return (
            processor.batch_decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
        )

    def infer_audio(audio_path):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url":
                            "file:"
                            + str(audio_path),
                    },
                ],
            },
        ]

        chat_text = (
            processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

        wav, _ = librosa.load(
            str(audio_path),
            sr=(
                processor
                .feature_extractor
                .sampling_rate
            ),
        )

        inputs = processor(
            text=chat_text,
            audios=[wav],
            sampling_rate=(
                processor
                .feature_extractor
                .sampling_rate
            ),
            return_tensors="pt",
            padding=True,
        )

        inputs = move_inputs(inputs)

        input_len = (
            inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=(
                    GENERATION_MAX_NEW_TOKENS
                ),
                do_sample=False,
            )

        output = output[:, input_len:]

        return (
            processor.batch_decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
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

    from qwen_omni_utils import (
        process_mm_info,
    )

    path = str(
        MODEL_PATHS["qwen25_omni"]
    )

    processor = (
        Qwen2_5OmniProcessor
        .from_pretrained(
            path,
            local_files_only=True,
        )
    )

    model = (
        Qwen2_5OmniForConditionalGeneration
        .from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
    )

    if hasattr(
        model,
        "disable_talker",
    ):
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
                    },
                ],
            },
        ]

        chat_text = (
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = processor(
            text=chat_text,
            return_tensors="pt",
            padding=True,
        )

        inputs = inputs.to(
            model.device
        )

        input_len = (
            inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                max_new_tokens=(
                    GENERATION_MAX_NEW_TOKENS
                ),
            )

        if isinstance(
            output,
            tuple,
        ):
            output = output[0]

        output = output[:, input_len:]

        return (
            processor.batch_decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
        )

    def infer_audio(audio_path):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio":
                            "file://"
                            + str(
                                audio_path.resolve()
                            ),
                    },
                ],
            },
        ]

        chat_text = (
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        (
            audios,
            images,
            videos,
        ) = process_mm_info(
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
        )

        inputs = (
            inputs
            .to(model.device)
            .to(model.dtype)
        )

        input_len = (
            inputs["input_ids"].shape[1]
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                use_audio_in_video=False,
                max_new_tokens=(
                    GENERATION_MAX_NEW_TOKENS
                ),
            )

        if isinstance(
            output,
            tuple,
        ):
            output = output[0]

        output = output[:, input_len:]

        return (
            processor.batch_decode(
                output,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
        )

    return model, infer_text, infer_audio


# ============================================================
# KIMI-AUDIO
# ============================================================

def load_kimi_audio():
    from kimia_infer.api.kimia import (
        KimiAudio,
    )

    model = KimiAudio(
        model_path=str(
            MODEL_PATHS["kimi_audio"]
        ),
        load_detokenizer=False,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,

        "text_temperature": 0.0,
        "text_top_k": 5,

        "audio_repetition_penalty":
            1.0,

        "audio_repetition_window_size":
            64,

        "text_repetition_penalty":
            1.0,

        "text_repetition_window_size":
            16,

        "max_new_tokens":
            GENERATION_MAX_NEW_TOKENS,
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
            wf.setframerate(sample_rate)

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
                "content":
                    str(silence_path),
            },
        ]

        return run(messages)

    def infer_audio(audio_path):
        messages = [
            {
                "role": "user",
                "message_type": "audio",
                "content":
                    str(audio_path),
            },
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

    path = str(
        MODEL_PATHS["audio_flamingo3"]
    )

    processor = (
        AutoProcessor.from_pretrained(
            path,
            local_files_only=True,
        )
    )

    model = (
        AudioFlamingo3ForConditionalGeneration
        .from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
    )

    model.eval()

    def run(conversation):
        inputs = (
            processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
        )

        inputs = inputs.to(
            model.device,
            dtype=model.dtype,
        )

        input_len = (
            inputs.input_ids.shape[1]
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=(
                    GENERATION_MAX_NEW_TOKENS
                ),
                do_sample=False,
            )

        output = output[:, input_len:]

        return (
            processor.decode(
                output[0],
                skip_special_tokens=True,
            )
            .strip()
        )

    def infer_text(prompt):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        return run(conversation)

    def infer_audio(audio_path):
        # Paper protocol supplies only the spoken request
        # to the target audio-language model.
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path":
                            str(audio_path),
                    },
                ],
            },
        ]

        return run(conversation)

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
# SHARDING
# ============================================================

def shard_bounds(
    n,
    num_shards,
    shard_index,
):
    start = (
        n * shard_index
    ) // num_shards

    end = (
        n * (shard_index + 1)
    ) // num_shards

    return start, end


def result_paths(
    outdir,
    num_shards,
    scope,
    modality,
):
    canonical = (
        outdir
        / f"responses.{scope}.{modality}.jsonl"
    )

    shard_files = [
        outdir
        / (
            f"responses.{scope}.{modality}."
            f"shard{i:03d}-of-{num_shards:03d}"
            ".jsonl"
        )
        for i in range(num_shards)
    ]

    done_files = [
        outdir
        / (
            f"{scope}_{modality}."
            f"shard{i:03d}-of-{num_shards:03d}"
            ".done"
        )
        for i in range(num_shards)
    ]

    return (
        canonical,
        shard_files,
        done_files,
    )


def visible_gpu_ids():
    visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )

    if visible:
        return [
            item.strip()
            for item in visible.split(",")
            if item.strip()
        ]

    return [
        str(i)
        for i in range(
            torch.cuda.device_count()
        )
    ]


# ============================================================
# TASKS
# ============================================================

def build_tasks(
    examples,
    modality,
):
    modalities = (
        ["text", "audio"]
        if modality == "both"
        else [modality]
    )

    tasks = []

    for example in examples:
        for input_modality in modalities:
            tasks.append({
                "example":
                    example,

                "modality":
                    input_modality,

                "task_id":
                    (
                        example["example_id"]
                        + "__"
                        + input_modality
                    ),
            })

    return tasks


# ============================================================
# WORKER
# ============================================================

def run_worker(
    model_name,
    assigned_tasks,
    outfile,
    donefile,
):
    completed = {
        (
            str(row["example_id"]),
            str(row["modality"]),
        )
        for row in read_jsonl(
            outfile
        )
    }

    remaining = [
        task
        for task in assigned_tasks
        if (
            task["example"]["example_id"],
            task["modality"],
        )
        not in completed
    ]

    print(
        f"[worker] assigned={len(assigned_tasks)} "
        f"completed={len(completed)} "
        f"remaining={len(remaining)}",
        flush=True,
    )

    if not remaining:
        donefile.write_text(
            "complete\n",
            encoding="utf-8",
        )

        print(
            "[OK] shard already complete",
            flush=True,
        )

        return

    model, infer_text, infer_audio = (
        load_model(
            model_name
        )
    )

    try:
        for position, task in enumerate(
            assigned_tasks,
            start=1,
        ):
            example = task["example"]
            modality = task["modality"]

            key = (
                example["example_id"],
                modality,
            )

            if key in completed:
                continue

            print(
                f"[{position:04d}/"
                f"{len(assigned_tasks):04d}] "
                f"{model_name} | "
                f"{example['example_id']} | "
                f"{modality}",
                flush=True,
            )

            if modality == "text":
                response = infer_text(
                    example["prompt_text"]
                )

                model_input_mode = (
                    "text_plus_fixed_1s_silence"
                    if model_name == "kimi_audio"
                    else "text_only"
                )

                inference_audio_path = None

            elif modality == "audio":
                audio_path = Path(
                    example["audio_path"]
                )

                response = infer_audio(
                    audio_path
                )

                model_input_mode = (
                    "audio_only"
                )

                inference_audio_path = (
                    example[
                        "audio_path_relative"
                    ]
                )

            else:
                raise ValueError(
                    modality
                )

            paper_faithful_example = (
                modality == "audio"
                and example[
                    "paper_faithful_condition"
                ]
            )

            record = {
                "task_id":
                    task["task_id"],

                "benchmark":
                    "Audio Jailbreaks "
                    "(arXiv:2605.30031)",

                "model":
                    model_name,

                "host":
                    socket.gethostname(),

                "example_id":
                    example["example_id"],

                "source_id":
                    example["source_id"],

                "split":
                    example["split"],

                "condition":
                    example["condition"],

                "attack_type":
                    (
                        "no_attack"
                        if (
                            example["condition"]
                            == "original"
                        )
                        else example["condition"]
                    ),

                "modality":
                    modality,

                "model_input_mode":
                    model_input_mode,

                "paper_faithful_condition":
                    example[
                        "paper_faithful_condition"
                    ],

                "paper_faithful_example":
                    paper_faithful_example,

                "language":
                    example["language"],

                "source_text":
                    example["source_text"],

                "english_prompt_text":
                    example["english_prompt_text"],

                "prompt_text":
                    example["prompt_text"],

                "translated_text":
                    example["translated_text"],

                "translation_model":
                    example["translation_model"],

                "translation_seed":
                    example["translation_seed"],

                "english_example_id":
                    example["english_example_id"],

                "experiment":
                    example["experiment"],

                "source_row":
                    example["source_row"],

                "jbb_index":
                    example["jbb_index"],

                "goal":
                    example["goal"],

                "target":
                    example["target"],

                "behavior":
                    example["behavior"],

                "category":
                    example["category"],

                "jbb_source":
                    example["jbb_source"],

                "audio_path":
                    inference_audio_path,

                "tts_model":
                    (
                        example["tts_model"]
                        if modality == "audio"
                        else None
                    ),

                "tts_language":
                    (
                        example["tts_language"]
                        if modality == "audio"
                        else None
                    ),

                "tts_style_prompt":
                    (
                        example[
                            "tts_style_prompt"
                        ]
                        if modality == "audio"
                        else None
                    ),

                "tts_batch_size":
                    (
                        example[
                            "tts_batch_size"
                        ]
                        if modality == "audio"
                        else None
                    ),

                "audio_sample_rate":
                    (
                        example["sample_rate"]
                        if modality == "audio"
                        else None
                    ),

                "audio_duration_seconds":
                    (
                        example[
                            "duration_seconds"
                        ]
                        if modality == "audio"
                        else None
                    ),

                "generation_context_window":
                    PAPER_CONTEXT_WINDOW,

                "generation_max_new_tokens":
                    GENERATION_MAX_NEW_TOKENS,

                "generation_batch_size":
                    1,

                "generation_decoding":
                    "greedy",

                "response":
                    response,
            }

            append_jsonl(
                outfile,
                record,
            )

            completed.add(key)

    finally:
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected = {
        (
            task["example"]["example_id"],
            task["modality"],
        )
        for task in assigned_tasks
    }

    actual = {
        (
            str(row["example_id"]),
            str(row["modality"]),
        )
        for row in read_jsonl(
            outfile
        )
    }

    missing = (
        expected - actual
    )

    if missing:
        raise RuntimeError(
            f"Shard has {len(missing)} "
            f"missing tasks"
        )

    donefile.write_text(
        "complete\n",
        encoding="utf-8",
    )

    print(
        "[OK] shard complete",
        flush=True,
    )


# ============================================================
# MERGE
# ============================================================

def merge_outputs(
    outdir,
    all_tasks,
    num_shards,
    scope,
    modality,
):
    (
        canonical,
        shard_files,
        _,
    ) = result_paths(
        outdir,
        num_shards,
        scope,
        modality,
    )

    expected_order = {
        (
            task["example"]["example_id"],
            task["modality"],
        ): index
        for index, task
        in enumerate(all_tasks)
    }

    merged = {}

    for shard in shard_files:
        for record in read_jsonl(
            shard
        ):
            key = (
                str(record["example_id"]),
                str(record["modality"]),
            )

            if key not in expected_order:
                continue

            if key in merged:
                raise RuntimeError(
                    f"Duplicate result key: {key}"
                )

            merged[key] = record

    missing = (
        set(expected_order)
        - set(merged)
    )

    if missing:
        raise RuntimeError(
            f"Merge missing "
            f"{len(missing)} tasks"
        )

    records = sorted(
        merged.values(),
        key=lambda row: (
            expected_order[
                (
                    str(row["example_id"]),
                    str(row["modality"]),
                )
            ]
        ),
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

    return canonical, records


def maybe_merge(
    model_name,
    outdir,
    all_tasks,
    num_shards,
    scope,
    modality,
):
    (
        _,
        _,
        donefiles,
    ) = result_paths(
        outdir,
        num_shards,
        scope,
        modality,
    )

    waiting = [
        index
        for index, path
        in enumerate(donefiles)
        if not path.exists()
    ]

    if waiting:
        print(
            "[merge] waiting for shards:",
            waiting,
            flush=True,
        )

        return False

    canonical, records = (
        merge_outputs(
            outdir,
            all_tasks,
            num_shards,
            scope,
            modality,
        )
    )

    counts = {}

    for record in records:
        key = (
            record["split"],
            record["condition"],
            record["modality"],
        )

        counts[str(key)] = (
            counts.get(
                str(key),
                0,
            )
            + 1
        )

    summary = {
        "benchmark":
            "Audio Jailbreaks "
            "(arXiv:2605.30031)",

        "model":
            model_name,

        "dataset":
            (
                "language_ablation"
                if any(
                    row.get("experiment")
                    == "language_ablation"
                    for row in records
                )
                else "main"
            ),

        "scope":
            scope,

        "modality":
            modality,

        "num_shards":
            num_shards,

        "total_generations":
            len(records),

        "context_window":
            PAPER_CONTEXT_WINDOW,

        "max_completion_tokens":
            GENERATION_MAX_NEW_TOKENS,

        "generation_batch_size":
            1,

        "decoding":
            "greedy",

        "paper_audio_only_protocol":
            (
                modality == "audio"
                and all(
                    row["paper_faithful_example"]
                    for row in records
                )
            ),

        "text_control_extension":
            modality in {
                "text",
                "both",
            },

        "kimi_text_condition":
            (
                "text_plus_fixed_1s_silence"
                if model_name
                == "kimi_audio"
                else None
            ),

        "counts":
            counts,

        "judge_applied":
            False,
    }

    (
        outdir
        / f"summary.{scope}.{modality}.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"[OK] merged "
        f"{len(records)} rows"
    )
    print(
        f"[responses] {canonical}"
    )

    return True


# ============================================================
# PREFLIGHT
# ============================================================

def preflight(
    examples,
    scope,
    modality,
):
    selected = filter_scope(
        examples,
        scope,
    )

    tasks = build_tasks(
        selected,
        modality,
    )

    harmful = sum(
        row["split"] == "harmful"
        for row in selected
    )

    benign = sum(
        row["split"] == "benign"
        for row in selected
    )

    print()
    print(
        "========================================"
    )
    print(
        "AUDIO JAILBREAK PREFLIGHT"
    )
    print(
        "========================================"
    )
    print(
        "base manifest:",
        BASE_MANIFEST,
    )
    print(
        "semantic manifest:",
        SEMANTIC_MANIFEST,
    )
    print(
        "audio manifest:",
        AUDIO_MANIFEST,
    )
    print(
        "scope:",
        scope,
    )
    print(
        "modality:",
        modality,
    )
    print(
        "selected examples:",
        len(selected),
    )
    print(
        "harmful:",
        harmful,
    )
    print(
        "benign:",
        benign,
    )
    print(
        "model generations:",
        len(tasks),
    )
    print(
        "paper context window:",
        PAPER_CONTEXT_WINDOW,
    )
    print(
        "max completion tokens:",
        GENERATION_MAX_NEW_TOKENS,
    )
    print(
        "generation batch size: 1"
    )
    print(
        "decoding: greedy"
    )
    print()

    return selected, tasks


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(
            MODEL_PATHS
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "main",
            "language_ablation",
        ],
        default="main",
    )

    parser.add_argument(
        "--scope",
        choices=[
            "all",
            "paper",
        ],
        default="all",
        help=(
            "'all' includes the 300 transformed "
            "benign matched-control extensions. "
            "'paper' keeps only rows marked "
            "paper_faithful_condition=true."
        ),
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
        "--num-shards",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--shard-indices",
        type=str,
        default=None,
        help=(
            "Comma-separated global shard indices. "
            "One worker is launched per visible GPU."
        ),
    )

    parser.add_argument(
        "--merge-only",
        action="store_true",
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    examples = load_dataset_examples(
        args.dataset
    )

    if (
        args.dataset == "language_ablation"
        and args.scope != "all"
    ):
        raise RuntimeError(
            "Language ablation is an extension and "
            "supports --scope all only."
        )

    selected = filter_scope(
        examples,
        args.scope,
    )

    if args.limit is not None:
        if args.limit < 1:
            raise RuntimeError(
                "--limit must be >= 1"
            )

        selected = selected[
            :args.limit
        ]

    all_tasks = build_tasks(
        selected,
        args.modality,
    )

    preflight(
        selected,
        args.scope,
        args.modality,
    )

    if args.preflight:
        return

    if args.model is None:
        raise RuntimeError(
            "--model is required unless "
            "--preflight is used"
        )

    if args.num_shards < 1:
        raise RuntimeError(
            "--num-shards must be >= 1"
        )

    outdir = (
        RESULT_ROOT
        / args.model
    )

    if args.dataset == "language_ablation":
        outdir = (
            outdir
            / "language_ablation"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        canonical,
        shard_files,
        donefiles,
    ) = result_paths(
        outdir,
        args.num_shards,
        args.scope,
        args.modality,
    )

    if args.merge_only:
        maybe_merge(
            args.model,
            outdir,
            all_tasks,
            args.num_shards,
            args.scope,
            args.modality,
        )
        return

    if args.shard_index is not None:
        index = args.shard_index

        if not (
            0
            <= index
            < args.num_shards
        ):
            raise RuntimeError(
                f"Invalid shard index "
                f"{index}"
            )

        start, end = shard_bounds(
            len(all_tasks),
            args.num_shards,
            index,
        )

        run_worker(
            args.model,
            all_tasks[start:end],
            shard_files[index],
            donefiles[index],
        )

        maybe_merge(
            args.model,
            outdir,
            all_tasks,
            args.num_shards,
            args.scope,
            args.modality,
        )

        return

    if args.shard_indices is not None:
        indices = [
            int(item)
            for item in (
                args.shard_indices
                .split(",")
            )
            if item.strip()
        ]

        if not indices:
            raise RuntimeError(
                "--shard-indices is empty"
            )

        if len(indices) != len(
            set(indices)
        ):
            raise RuntimeError(
                "Duplicate shard indices"
            )

        for index in indices:
            if not (
                0
                <= index
                < args.num_shards
            ):
                raise RuntimeError(
                    f"Invalid shard index "
                    f"{index}"
                )

        gpu_ids = (
            visible_gpu_ids()
        )

        if len(gpu_ids) < len(
            indices
        ):
            raise RuntimeError(
                f"Need {len(indices)} GPUs "
                f"but visible GPUs are "
                f"{gpu_ids}"
            )

        processes = []

        for index, gpu_id in zip(
            indices,
            gpu_ids,
        ):
            env = os.environ.copy()

            env[
                "CUDA_VISIBLE_DEVICES"
            ] = gpu_id

            cmd = [
                sys.executable,
                str(
                    Path(__file__)
                    .resolve()
                ),
                "--model",
                args.model,
                "--dataset",
                args.dataset,
                "--scope",
                args.scope,
                "--modality",
                args.modality,
                "--num-shards",
                str(
                    args.num_shards
                ),
                "--shard-index",
                str(index),
            ]

            if args.limit is not None:
                cmd.extend([
                    "--limit",
                    str(args.limit),
                ])

            print(
                f"[launch] "
                f"shard={index}/"
                f"{args.num_shards} "
                f"gpu={gpu_id}",
                flush=True,
            )

            processes.append(
                subprocess.Popen(
                    cmd,
                    env=env,
                )
            )

        exit_codes = [
            process.wait()
            for process in processes
        ]

        if any(
            code != 0
            for code in exit_codes
        ):
            raise RuntimeError(
                "Generation worker failure: "
                f"{exit_codes}"
            )

        maybe_merge(
            args.model,
            outdir,
            all_tasks,
            args.num_shards,
            args.scope,
            args.modality,
        )

        return

    if args.num_shards != 1:
        raise RuntimeError(
            "With --num-shards > 1, "
            "provide --shard-index or "
            "--shard-indices."
        )

    run_worker(
        args.model,
        all_tasks,
        shard_files[0],
        donefiles[0],
    )

    maybe_merge(
        args.model,
        outdir,
        all_tasks,
        1,
        args.scope,
        args.modality,
    )


if __name__ == "__main__":
    main()
