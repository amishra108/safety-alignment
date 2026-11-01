#!/usr/bin/env python3

import csv
import json
import os
import shutil
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(
    "/research/milsrg1/user_workspace/am3244/safety-alignment"
)

DATA_ROOT = (
    ROOT / "data/audio_jailbreak_clean_benchmark"
)

MANIFEST_ROOT = DATA_ROOT / "manifests"
AUDIO_ROOT = DATA_ROOT / "audio"

TEXT_MANIFEST = (
    MANIFEST_ROOT / "benchmark_text.csv"
)

OLD_AUDIO_MANIFEST = (
    ROOT
    / "data/audio_jailbreaks/manifests/semantic_audio.csv"
)

FINAL_MANIFEST = (
    MANIFEST_ROOT / "benchmark.csv"
)

MODEL_PATH = (
    ROOT
    / "models/qwen3-tts-12hz-1.7b-voicedesign"
)

LANGUAGE_NAMES = {
    "english": "English",
    "spanish": "Spanish",
    "chinese": "Chinese",
    "russian": "Russian",
}

BATCH_SIZE = 32

BASE_FIELDS = [
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
]

FINAL_FIELDS = BASE_FIELDS + [
    "audio_path",
    "tts_model",
    "tts_language",
    "tts_style_prompt",
    "tts_batch_size",
    "sample_rate",
    "duration_seconds",
]


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=FINAL_FIELDS,
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        w.writerows(rows)

    tmp.replace(path)


def style(language):
    name = LANGUAGE_NAMES[language]

    return (
        f"Speak in {name} with a neutral accent, "
        "in a neutral tone, sounding like a "
        "middle-aged adult male speaker, and "
        "speak at a normal speed."
    )


def wav_path(row):
    return (
        AUDIO_ROOT
        / row["language"]
        / row["condition"]
        / row["split"]
        / f'{row["example_id"]}.wav'
    )


def valid_wav(path):
    if not path.exists():
        return False

    if path.stat().st_size <= 44:
        return False

    try:
        info = sf.info(path)
        return (
            info.frames > 0
            and info.samplerate > 0
        )
    except Exception:
        return False


def final_row(row, path, batch_size):
    info = sf.info(path)

    return {
        **{
            k: row[k]
            for k in BASE_FIELDS
        },
        "audio_path":
            str(path.relative_to(ROOT)),
        "tts_model":
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "tts_language":
            LANGUAGE_NAMES[row["language"]],
        "tts_style_prompt":
            style(row["language"]),
        "tts_batch_size":
            batch_size,
        "sample_rate":
            info.samplerate,
        "duration_seconds":
            f"{info.duration:.6f}",
    }


def link_or_copy(src, dst):
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if valid_wav(dst):
        return

    if dst.exists():
        dst.unlink()

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    rows = read_csv(TEXT_MANIFEST)

    if len(rows) != 3200:
        raise RuntimeError(
            f"Expected 3200 text rows; got {len(rows)}"
        )

    old_audio = read_csv(
        OLD_AUDIO_MANIFEST
    )

    old_by_id = {
        r["example_id"]: r
        for r in old_audio
    }

    english = [
        r for r in rows
        if r["language"] == "english"
    ]

    translated = [
        r for r in rows
        if r["language"] != "english"
    ]

    if len(english) != 800:
        raise RuntimeError(
            f"Expected 800 English; got {len(english)}"
        )

    if len(translated) != 2400:
        raise RuntimeError(
            f"Expected 2400 translated; got {len(translated)}"
        )

    completed = {}

    print(
        "[english] linking existing 800 WAVs",
        flush=True,
    )

    for i, row in enumerate(
        english,
        start=1,
    ):
        english_id = row[
            "english_example_id"
        ]

        if english_id not in old_by_id:
            raise RuntimeError(
                f"Missing old English audio: {english_id}"
            )

        src = (
            ROOT
            / old_by_id[english_id]["audio_path"]
        )

        if not valid_wav(src):
            raise RuntimeError(
                f"Invalid old WAV: {src}"
            )

        dst = wav_path(row)

        link_or_copy(
            src,
            dst,
        )

        completed[row["example_id"]] = (
            final_row(
                row,
                dst,
                int(
                    old_by_id[english_id][
                        "tts_batch_size"
                    ]
                ),
            )
        )

        if i % 100 == 0:
            print(
                f"[english] {i}/800",
                flush=True,
            )

    missing = []

    for row in translated:
        path = wav_path(row)

        if valid_wav(path):
            completed[row["example_id"]] = (
                final_row(
                    row,
                    path,
                    BATCH_SIZE,
                )
            )
        else:
            if path.exists():
                path.unlink()
            missing.append(row)

    print(
        f"[translated] existing="
        f"{len(translated)-len(missing)} "
        f"missing={len(missing)}",
        flush=True,
    )

    if missing:
        print(
            f"[tts model] {MODEL_PATH}",
            flush=True,
        )

        model = (
            Qwen3TTSModel
            .from_pretrained(
                str(MODEL_PATH),
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
        )

        # Reduce padding waste.
        missing = sorted(
            missing,
            key=lambda r: (
                r["language"],
                len(r["prompt_text"]),
            ),
        )

        total_batches = (
            len(missing)
            + BATCH_SIZE - 1
        ) // BATCH_SIZE

        for bi, start in enumerate(
            range(
                0,
                len(missing),
                BATCH_SIZE,
            ),
            start=1,
        ):
            batch = missing[
                start:start + BATCH_SIZE
            ]

            # Never mix languages inside one TTS call.
            first_language = (
                batch[0]["language"]
            )

            same_language = [
                r
                for r in batch
                if r["language"]
                == first_language
            ]

            if (
                len(same_language)
                != len(batch)
            ):
                # Find the contiguous boundary.
                cut = next(
                    i
                    for i, r in enumerate(batch)
                    if r["language"]
                    != first_language
                )

                batch = batch[:cut]

            language = batch[0]["language"]

            print(
                f"[tts] batch={bi}/{total_batches} "
                f"language={language} "
                f"size={len(batch)}",
                flush=True,
            )

            texts = [
                r["prompt_text"]
                for r in batch
            ]

            languages = [
                LANGUAGE_NAMES[language]
                for _ in batch
            ]

            instructs = [
                style(language)
                for _ in batch
            ]

            t0 = time.time()

            with torch.inference_mode():
                wavs, sr = (
                    model.generate_voice_design(
                        text=texts,
                        language=languages,
                        instruct=instructs,
                    )
                )

            if len(wavs) != len(batch):
                raise RuntimeError(
                    f"TTS returned {len(wavs)} "
                    f"for batch {len(batch)}"
                )

            for row, wav in zip(
                batch,
                wavs,
            ):
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

                if not valid_wav(path):
                    raise RuntimeError(
                        f"Invalid generated WAV: {path}"
                    )

                completed[
                    row["example_id"]
                ] = final_row(
                    row,
                    path,
                    BATCH_SIZE,
                )

            print(
                f"      time={time.time()-t0:.1f}s "
                f"avg={(time.time()-t0)/len(batch):.2f}s/item",
                flush=True,
            )

            # Durable checkpoint after every batch.
            partial = [
                completed[r["example_id"]]
                for r in rows
                if r["example_id"] in completed
            ]

            write_csv(
                FINAL_MANIFEST,
                partial,
            )

            # Recompute missing so unusual language-boundary batch
            # truncation cannot silently omit rows.
            remaining_ids = {
                r["example_id"]
                for r in translated
                if r["example_id"]
                not in completed
            }

            if remaining_ids:
                # Continue outer processing; any unprocessed boundary
                # rows remain absent and are caught below.
                pass

        # One second pass handles any rows skipped at a language boundary.
        leftovers = [
            r
            for r in translated
            if r["example_id"]
            not in completed
        ]

        for row in leftovers:
            language = row["language"]

            print(
                f"[tts single fallback] "
                f"{row['example_id']}",
                flush=True,
            )

            with torch.inference_mode():
                wavs, sr = (
                    model.generate_voice_design(
                        text=[row["prompt_text"]],
                        language=[
                            LANGUAGE_NAMES[language]
                        ],
                        instruct=[
                            style(language)
                        ],
                    )
                )

            path = wav_path(row)
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            sf.write(
                path,
                wavs[0],
                sr,
            )

            completed[
                row["example_id"]
            ] = final_row(
                row,
                path,
                BATCH_SIZE,
            )

    if len(completed) != 3200:
        missing_ids = [
            r["example_id"]
            for r in rows
            if r["example_id"]
            not in completed
        ]

        raise RuntimeError(
            f"Only {len(completed)}/3200 complete; "
            f"missing={missing_ids[:10]}"
        )

    final = [
        completed[r["example_id"]]
        for r in rows
    ]

    write_csv(
        FINAL_MANIFEST,
        final,
    )

    metadata = {
        "total_rows": 3200,
        "languages": {
            "english": 800,
            "spanish": 800,
            "chinese": 800,
            "russian": 800,
        },
        "english_audio":
            "hard-linked from existing Audio Jailbreak semantic WAVs "
            "where filesystem permits; copy fallback otherwise",
        "translated_audio_rows": 2400,
        "tts_model":
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "translated_tts_batch_size":
            BATCH_SIZE,
        "tts_style":
            "same neutral middle-aged male normal-speed template "
            "with target language substituted",
    }

    (
        MANIFEST_ROOT
        / "audio_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("======================================")
    print("[OK] CLEAN BENCHMARK AUDIO COMPLETE")
    print("======================================")
    print("English : 800")
    print("Spanish : 800")
    print("Chinese : 800")
    print("Russian : 800")
    print("TOTAL   : 3200")
    print("manifest:", FINAL_MANIFEST)


if __name__ == "__main__":
    main()
