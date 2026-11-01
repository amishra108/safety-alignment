from pathlib import Path
from collections import Counter, defaultdict
import json
import shutil
import subprocess

import librosa
import soundfile as sf
from datasets import Audio, Dataset, load_from_disk
from pydub import AudioSegment
from pydub.generators import WhiteNoise


ROOT = Path(
    "/research/milsrg1/user_workspace/am3244/safety-alignment"
)

SOURCE = (
    ROOT
    / "data/jailbreak_audiobench/working_dataset"
)

OUTPUT = (
    ROOT
    / "data/jailbreak_audiobench/"
    "explicit_small_partial_compositional"
)

OUTPUT_TMP = Path(
    str(OUTPUT) + ".__tmp__"
)

STAGING = (
    ROOT
    / "data/jailbreak_audiobench/"
    "explicit_small_partial_compositional.__wav_staging__"
)

SUMMARY = (
    ROOT
    / "data/jailbreak_audiobench/"
    "explicit_small_partial_compositional_summary.json"
)

EXPECTED_BASES = 262
EXPECTED_VARIANTS_PER_BASE = 16
EXPECTED_EDITED = (
    EXPECTED_BASES
    * EXPECTED_VARIANTS_PER_BASE
)
EXPECTED_TOTAL = (
    EXPECTED_BASES
    + EXPECTED_EDITED
)

BRANCHES = (
    ("accent", "Kanye West"),
    ("accent", "Donald Trump"),
    ("emotion", "laughs"),
    ("emotion", "screams"),
)

EMPHASIS = (
    2.0,
    5.0,
)

SPEED = (
    0.5,
    1.5,
)

WHITE_NOISE_GAIN_DB = -30.0


def filepath_of(row):
    fp = row["filepath"]
    return str(fp).replace("\\", "/")


def parse_condition(filepath):
    parts = filepath.split("/")

    if len(parts) < 3:
        raise RuntimeError(
            f"Malformed filepath: {filepath}"
        )

    family = parts[1]

    if family == "audio_original":
        variant = "original"
    else:
        if len(parts) < 4:
            raise RuntimeError(
                f"Missing variant: {filepath}"
            )

        variant = parts[2]

    return family, variant


def base_audio_id(filepath):
    filename = Path(filepath).name
    stem = Path(filename).stem

    prefixes = (
        "Advbench_audio_",
        "MM-Safetybench_audio_",
        "RedTeam_2K_audio_",
        "safebench_audio_",
    )

    for prefix in prefixes:
        if not stem.startswith(prefix):
            continue

        rest = stem[len(prefix):]
        digits = ""

        for char in rest:
            if not char.isdigit():
                break
            digits += char

        if not digits:
            break

        return (
            prefix
            + str(int(digits)).zfill(3)
        )

    raise RuntimeError(
        f"Cannot recover base ID: {filepath}"
    )


def audio_bytes(row):
    audio = row["audio"]

    if not isinstance(audio, dict):
        raise RuntimeError(
            "Expected audio to be stored "
            "as an encoded-audio dictionary."
        )

    data = audio.get("bytes")

    if not data:
        raise RuntimeError(
            f"Missing audio bytes: {filepath_of(row)}"
        )

    return data


def write_source(row, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        audio_bytes(row)
    )


def apply_emphasis(
    input_path,
    output_path,
    gain,
):
    # Matches the repository implementation:
    # librosa.load() default sample rate and the
    # first 1.0 second multiplied by the gain.
    y, sr = librosa.load(
        str(input_path)
    )

    end_sample = min(
        int(1.0 * sr),
        len(y),
    )

    y[:end_sample] *= gain

    sf.write(
        str(output_path),
        y,
        sr,
    )


def apply_speed(
    input_path,
    output_path,
    speed,
):
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-filter:a",
            f"atempo={speed}",
            str(output_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "FFmpeg atempo failed:\n"
            + proc.stderr
        )


def apply_white_noise(
    input_path,
    output_path,
):
    # Matches generate_white_noise_dataset.py:
    # WhiteNoise with the same duration,
    # -30 dB gain, then overlay.
    original = AudioSegment.from_wav(
        str(input_path)
    )

    white_noise = (
        WhiteNoise()
        .to_audio_segment(
            duration=len(original)
        )
        .apply_gain(
            WHITE_NOISE_GAIN_DB
        )
    )

    combined = original.overlay(
        white_noise
    )

    combined.export(
        str(output_path),
        format="wav",
    )


if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "ffmpeg is not available on PATH."
    )


# ============================================================
# Load the already-prepared Plus subset
# ============================================================

if not SOURCE.exists():
    raise FileNotFoundError(
        SOURCE
    )

ds = load_from_disk(
    str(SOURCE)
)

ds = ds.cast_column(
    "audio",
    Audio(decode=False),
)

if len(ds) != 3144:
    raise RuntimeError(
        "Expected the existing Plus working dataset "
        f"to contain 3144 rows; found {len(ds)}."
    )


# ============================================================
# Index originals and the four released first-stage branches
# ============================================================

originals = {}
branches = defaultdict(dict)

family_counts = Counter()

for row in ds:
    fp = filepath_of(row)

    family, variant = (
        parse_condition(fp)
    )

    base = base_audio_id(fp)

    family_counts[family] += 1

    if family == "audio_original":
        if base in originals:
            raise RuntimeError(
                f"Duplicate original: {base}"
            )

        originals[base] = row

    elif (
        family,
        variant,
    ) in BRANCHES:
        key = (
            family,
            variant,
        )

        if base in branches[key]:
            raise RuntimeError(
                f"Duplicate branch: "
                f"{key} / {base}"
            )

        branches[key][base] = row


if len(originals) != EXPECTED_BASES:
    raise RuntimeError(
        f"Expected exactly 262 Plus originals; "
        f"found {len(originals)}."
    )

base_ids = set(originals)

for key in BRANCHES:
    found = set(
        branches[key]
    )

    missing = (
        base_ids
        - found
    )

    extra = (
        found
        - base_ids
    )

    if missing or extra:
        raise RuntimeError(
            f"{key} does not exactly cover "
            f"the 262 Plus bases. "
            f"missing={len(missing)} "
            f"extra={len(extra)} "
            f"first_missing={sorted(missing)[:10]}"
        )


print()
print("=" * 72)
print("SOURCE PREFLIGHT")
print("=" * 72)
print("working_dataset rows:", len(ds))
print("family counts:", dict(family_counts))
print("original bases:", len(base_ids))

for key in BRANCHES:
    print(
        f"{key}: {len(branches[key])}"
    )

print(
    "[OK] all four released first-stage "
    "branches cover all 262 Plus bases"
)


# ============================================================
# Clean staging only
# ============================================================

if STAGING.exists():
    shutil.rmtree(
        STAGING
    )

STAGING.mkdir(
    parents=True,
    exist_ok=True,
)

if OUTPUT_TMP.exists():
    shutil.rmtree(
        OUTPUT_TMP
    )


records = []


# ============================================================
# Preserve 262 original audios
# ============================================================

for base_index, base in enumerate(
    sorted(base_ids)
):
    path = (
        STAGING
        / "original"
        / f"{base}.wav"
    )

    write_source(
        originals[base],
        path,
    )

    records.append({
        "example_id":
            (
                "jab_partial_comp_"
                f"{base_index:03d}_original"
            ),

        "base_audio_id":
            base,

        "condition_family":
            "audio_original",

        "condition_variant":
            "original",

        "branch_family":
            "original",

        "branch_variant":
            "original",

        "emphasis":
            "",

        "speed":
            "",

        "noise":
            "",

        "source_filepath":
            filepath_of(
                originals[base]
            ),

        "wav_path":
            str(path),
    })


# ============================================================
# Build the reproducible half-grid
#
# branch -> emphasis -> speed -> white noise
#
# 4 × 2 × 2 × 1 = 16 variants/base
# ============================================================

edited = 0

for base_index, base in enumerate(
    sorted(base_ids),
    1,
):
    print(
        f"[{base_index:03d}/{EXPECTED_BASES}] "
        f"{base}",
        flush=True,
    )

    scratch = (
        STAGING
        / "_scratch"
        / base
    )

    scratch.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        branch_family,
        branch_variant,
    ) in BRANCHES:
        source_row = branches[
            (
                branch_family,
                branch_variant,
            )
        ][base]

        safe_variant = (
            branch_variant
            .replace(" ", "_")
        )

        branch_wav = (
            scratch
            / (
                f"{branch_family}"
                f"_{safe_variant}.wav"
            )
        )

        write_source(
            source_row,
            branch_wav,
        )

        for emphasis in EMPHASIS:
            emphasized = (
                scratch
                / (
                    f"{branch_family}"
                    f"_{safe_variant}"
                    f"_emphasis_x{emphasis:g}.wav"
                )
            )

            apply_emphasis(
                branch_wav,
                emphasized,
                emphasis,
            )

            for speed in SPEED:
                sped = (
                    scratch
                    / (
                        f"{branch_family}"
                        f"_{safe_variant}"
                        f"_emphasis_x{emphasis:g}"
                        f"_speed_x{speed:g}.wav"
                    )
                )

                apply_speed(
                    emphasized,
                    sped,
                    speed,
                )

                final_dir = (
                    STAGING
                    / "compositional"
                    / base
                )

                final_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                final_path = (
                    final_dir
                    / (
                        f"{branch_family}"
                        f"-{safe_variant}"
                        f"__emphasis-x{emphasis:g}"
                        f"__speed-x{speed:g}"
                        f"__noise-white.wav"
                    )
                )

                apply_white_noise(
                    sped,
                    final_path,
                )

                edited += 1

                records.append({
                    "example_id":
                        (
                            "jab_partial_comp_"
                            f"{base_index - 1:03d}_"
                            f"{edited:05d}"
                        ),

                    "base_audio_id":
                        base,

                    "condition_family":
                        "compositional",

                    "condition_variant":
                        (
                            f"{branch_family}:"
                            f"{branch_variant}"
                            f"|emphasis:x{emphasis:g}"
                            f"|speed:x{speed:g}"
                            "|noise:white"
                        ),

                    "branch_family":
                        branch_family,

                    "branch_variant":
                        branch_variant,

                    "emphasis":
                        f"x{emphasis:g}",

                    "speed":
                        f"x{speed:g}",

                    "noise":
                        "white",

                    "source_filepath":
                        filepath_of(
                            source_row
                        ),

                    "wav_path":
                        str(final_path),
                })

    shutil.rmtree(
        scratch
    )


# ============================================================
# Exact structural checks
# ============================================================

if edited != EXPECTED_EDITED:
    raise RuntimeError(
        f"Expected {EXPECTED_EDITED} edited audios; "
        f"generated {edited}."
    )

if len(records) != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL} total rows; "
        f"generated {len(records)}."
    )

per_base = Counter(
    row["base_audio_id"]
    for row in records
)

bad = {
    base: count
    for base, count
    in per_base.items()
    if count != 17
}

if bad:
    raise RuntimeError(
        "Every base must have exactly "
        "1 original + 16 composites. "
        f"Bad bases: {bad}"
    )

grid = Counter(
    (
        row["branch_family"],
        row["branch_variant"],
        row["emphasis"],
        row["speed"],
        row["noise"],
    )
    for row in records
    if row["condition_family"]
    == "compositional"
)

if len(grid) != 16:
    raise RuntimeError(
        f"Expected exactly 16 grid cells; "
        f"found {len(grid)}."
    )

for cell, count in grid.items():
    if count != EXPECTED_BASES:
        raise RuntimeError(
            f"Grid cell {cell} has {count} rows; "
            f"expected {EXPECTED_BASES}."
        )


# ============================================================
# Pack everything into a self-contained Arrow dataset
# ============================================================

def generate_rows():
    for row in records:
        wav = Path(
            row["wav_path"]
        )

        yield {
            "example_id":
                row["example_id"],

            "base_audio_id":
                row["base_audio_id"],

            "filepath":
                str(
                    wav.relative_to(
                        STAGING
                    )
                ),

            "condition_family":
                row["condition_family"],

            "condition_variant":
                row["condition_variant"],

            "branch_family":
                row["branch_family"],

            "branch_variant":
                row["branch_variant"],

            "emphasis":
                row["emphasis"],

            "speed":
                row["speed"],

            "noise":
                row["noise"],

            "source_filepath":
                row["source_filepath"],

            "audio": {
                "path":
                    wav.name,

                "bytes":
                    wav.read_bytes(),
            },
        }


out = Dataset.from_generator(
    generate_rows
)

if len(out) != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Arrow dataset has {len(out)} rows; "
        f"expected {EXPECTED_TOTAL}."
    )

out.save_to_disk(
    str(OUTPUT_TMP)
)

check = load_from_disk(
    str(OUTPUT_TMP)
)

check = check.cast_column(
    "audio",
    Audio(decode=False),
)

if len(check) != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Saved dataset has {len(check)} rows."
    )

for index in (
    0,
    261,
    262,
    len(check) // 2,
    len(check) - 1,
):
    if not check[index]["audio"]["bytes"]:
        raise RuntimeError(
            f"Audio bytes missing "
            f"after save at row {index}."
        )


if OUTPUT.exists():
    shutil.rmtree(
        OUTPUT
    )

OUTPUT_TMP.rename(
    OUTPUT
)


summary = {
    "benchmark":
        "Jailbreak-AudioBench",

    "subset":
        "Explicit Small partial compositional grid",

    "source":
        "Existing Plus working_dataset",

    "base_audios":
        EXPECTED_BASES,

    "variants_per_base":
        EXPECTED_VARIANTS_PER_BASE,

    "edited_audios":
        EXPECTED_EDITED,

    "total_rows":
        EXPECTED_TOTAL,

    "editing_order": [
        "released accent_or_emotion",
        "emphasis",
        "speed",
        "white_noise",
    ],

    "branches": [
        {
            "family": family,
            "variant": variant,
        }
        for family, variant
        in BRANCHES
    ],

    "emphasis": [
        2.0,
        5.0,
    ],

    "speed": [
        0.5,
        1.5,
    ],

    "noise": [
        "white",
    ],

    "white_noise_gain_db":
        WHITE_NOISE_GAIN_DB,

    "omitted_from_full_32_grid":
        "second background-noise branch",

    "output":
        str(OUTPUT),
}

SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


shutil.rmtree(
    STAGING
)


print()
print("=" * 72)
print("[OK] PARTIAL COMPOSITIONAL GRID BUILT")
print("=" * 72)
print("Bases:             262")
print("Variants/base:      16")
print("Edited:           4192")
print("Originals:         262")
print("TOTAL:            4454")
print()
print("Output:")
print(OUTPUT)
