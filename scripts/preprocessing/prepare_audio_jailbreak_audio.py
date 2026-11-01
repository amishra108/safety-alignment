from pathlib import Path
import csv
import json
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data/audio_jailbreaks"
MANIFEST_ROOT = DATA_ROOT / "manifests"
AUDIO_ROOT = DATA_ROOT / "audio" / "semantic"

SEMANTIC_CSV = MANIFEST_ROOT / "semantic.csv"
AUDIO_MANIFEST = MANIFEST_ROOT / "semantic_audio.csv"

TTS_MODEL_PATH = ROOT / "models/qwen3-tts-12hz-1.7b-voicedesign"

DEFAULT_STYLE_PROMPT = (
    "Speak in English with a neutral accent, in a neutral tone, "
    "sounding like a middle-aged adult male speaker, and "
    "speak at a normal speed."
)

ORIGINAL_BATCH_SIZE = 64
SEMANTIC_BATCH_SIZE = 32

CONDITIONS = {
    "original",
    "literal",
    "narrative_framing",
    "content_dilution",
}

OUTPUT_FIELDS = [
    "example_id",
    "source_id",
    "split",
    "condition",
    "source_text",
    "prompt_text",
    "paper_faithful_condition",
    "rewriter_model",
    "audio_path",
    "tts_model",
    "tts_language",
    "tts_style_prompt",
    "tts_batch_size",
    "sample_rate",
    "duration_seconds",
]


def load_manifest():
    if not SEMANTIC_CSV.exists():
        raise FileNotFoundError(SEMANTIC_CSV)

    with SEMANTIC_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 800:
        raise RuntimeError(
            f"Expected exactly 800 rows in semantic.csv; found {len(rows)}"
        )

    conditions = {row["condition"] for row in rows}
    if conditions != CONDITIONS:
        raise RuntimeError(f"Unexpected conditions: {sorted(conditions)}")

    originals = [r for r in rows if r["condition"] == "original"]
    semantic = [r for r in rows if r["condition"] != "original"]

    if len(originals) != 200:
        raise RuntimeError(f"Expected 200 originals; found {len(originals)}")

    if len(semantic) != 600:
        raise RuntimeError(f"Expected 600 semantic rows; found {len(semantic)}")

    for condition in (
        "literal",
        "narrative_framing",
        "content_dilution",
    ):
        n = sum(r["condition"] == condition for r in semantic)
        if n != 200:
            raise RuntimeError(
                f"{condition}: expected 200 rows; found {n}"
            )

    for split in ("harmful", "benign"):
        n = sum(r["split"] == split for r in rows)
        if n != 400:
            raise RuntimeError(
                f"{split}: expected 400 rows; found {n}"
            )

    return originals, semantic


def load_tts():
    print(f"[tts model] {TTS_MODEL_PATH}", flush=True)

    return Qwen3TTSModel.from_pretrained(
        str(TTS_MODEL_PATH),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )


def wav_path(row):
    return (
        AUDIO_ROOT
        / row["condition"]
        / row["split"]
        / f'{row["example_id"]}.wav'
    )


def make_manifest_row(row, path, sr, duration, batch_size):
    return {
        **row,
        "audio_path": str(path.relative_to(ROOT)),
        "tts_model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "tts_language": "English",
        "tts_style_prompt": DEFAULT_STYLE_PROMPT,
        "tts_batch_size": batch_size,
        "sample_rate": sr,
        "duration_seconds": f"{duration:.6f}",
    }


def write_manifest(rows):
    AUDIO_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with AUDIO_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_FIELDS,
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


def synthesize(model, rows, batch_size, completed, group_name):
    existing = []
    missing = []

    for row in rows:
        path = wav_path(row)

        if path.exists() and path.stat().st_size > 0:
            try:
                info = sf.info(path)

                if info.frames <= 0 or info.samplerate <= 0:
                    raise RuntimeError("invalid WAV")

                completed[row["example_id"]] = make_manifest_row(
                    row=row,
                    path=path,
                    sr=info.samplerate,
                    duration=info.duration,
                    batch_size=batch_size,
                )
                existing.append(row)

            except Exception:
                path.unlink(missing_ok=True)
                missing.append(row)
        else:
            missing.append(row)

    print(
        f"[{group_name}] existing={len(existing)} "
        f"missing={len(missing)}",
        flush=True,
    )

    # Length bucketing improves efficiency for variable-length semantic prompts.
    if group_name == "semantic":
        missing = sorted(
            missing,
            key=lambda row: len(row["prompt_text"]),
        )

    total_batches = (
        len(missing) + batch_size - 1
    ) // batch_size

    for batch_index, start in enumerate(
        range(0, len(missing), batch_size),
        start=1,
    ):
        batch = missing[start:start + batch_size]

        print(
            f"[{group_name}] "
            f"batch {batch_index}/{total_batches} "
            f"size={len(batch)}",
            flush=True,
        )

        texts = [
            row["prompt_text"]
            for row in batch
        ]

        languages = [
            "English"
            for _ in batch
        ]

        instructs = [
            DEFAULT_STYLE_PROMPT
            for _ in batch
        ]

        t0 = time.time()

        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(
                text=texts,
                language=languages,
                instruct=instructs,
            )

        if len(wavs) != len(batch):
            raise RuntimeError(
                f"TTS returned {len(wavs)} WAVs "
                f"for batch of {len(batch)}"
            )

        for row, wav in zip(batch, wavs):
            path = wav_path(row)
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            sf.write(
                path,
                wav,
                sr,
            )

            duration = len(wav) / sr

            completed[row["example_id"]] = make_manifest_row(
                row=row,
                path=path,
                sr=sr,
                duration=duration,
                batch_size=batch_size,
            )

        elapsed = time.time() - t0

        print(
            f"    generated={len(batch)} "
            f"time={elapsed:.1f}s "
            f"avg={elapsed / len(batch):.2f}s/item "
            f"sr={sr}",
            flush=True,
        )


def main():
    originals, semantic = load_manifest()

    print("============================================")
    print("Qwen3-TTS Audio Jailbreak Preparation")
    print("============================================")
    print(f"original rows : {len(originals)}")
    print(f"semantic rows : {len(semantic)}")
    print(f"total rows    : {len(originals) + len(semantic)}")
    print(f"original batch: {ORIGINAL_BATCH_SIZE}")
    print(f"semantic batch: {SEMANTIC_BATCH_SIZE}")
    print()

    model = load_tts()
    completed = {}

    synthesize(
        model=model,
        rows=originals,
        batch_size=ORIGINAL_BATCH_SIZE,
        completed=completed,
        group_name="original",
    )

    synthesize(
        model=model,
        rows=semantic,
        batch_size=SEMANTIC_BATCH_SIZE,
        completed=completed,
        group_name="semantic",
    )

    if len(completed) != 800:
        raise RuntimeError(
            f"Expected 800 completed rows; found {len(completed)}"
        )

    canonical_order = originals + semantic

    final_rows = [
        completed[row["example_id"]]
        for row in canonical_order
    ]

    write_manifest(final_rows)

    metadata = {
        "tts_model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "tts_language": "English",
        "tts_style_prompt": DEFAULT_STYLE_PROMPT,
        "attention_backend": "sdpa",
        "original_batch_size": ORIGINAL_BATCH_SIZE,
        "semantic_batch_size": SEMANTIC_BATCH_SIZE,
        "original_rows": 200,
        "harmful_semantic_rows": 300,
        "benign_semantic_rows": 300,
        "total_rows": 800,
        "benign_semantic_extension": True,
    }

    (
        MANIFEST_ROOT / "semantic_audio_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("============================================")
    print("[OK] 800 AUDIO FILES COMPLETE")
    print("============================================")
    print(f"manifest: {AUDIO_MANIFEST}")
    print(f"audio:    {AUDIO_ROOT}")


if __name__ == "__main__":
    main()
