from pathlib import Path
import csv
import shutil
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from datasets import load_from_disk, Dataset, Features, Value
from kokoro import KPipeline

SRC = Path("data/xstest/manifest.csv")
ROOT = Path("data/xstest_audio")
OLD_ARROW = ROOT / "dataset"
TMP_ARROW = ROOT / "_dataset_tmp"
WAV_DIR = ROOT / "_wav"
MANIFEST = ROOT / "manifest.csv"

if not SRC.exists():
    raise FileNotFoundError(SRC)

if not OLD_ARROW.exists():
    raise FileNotFoundError(OLD_ARROW)

df = pd.read_csv(SRC)

if len(df) != 450:
    raise RuntimeError(f"Expected 450 XSTest prompts, got {len(df)}")

if "prompt" not in df.columns:
    raise RuntimeError(f"Missing prompt column: {list(df.columns)}")

print("[1/4] Loading existing 250 synthesized audios...", flush=True)

old = load_from_disk(str(OLD_ARROW))

print("Existing rows:", len(old), flush=True)

if len(old) != 250:
    raise RuntimeError(
        f"Expected existing 250-row Arrow dataset, got {len(old)}"
    )

# Match by exact prompt text, NOT by row number.
existing = {}

for row in old:
    existing[str(row["prompt"])] = {
        "audio_path": row["audio_path"],
        "audio_bytes": row["audio_bytes"],
    }

print("Reusable existing audios:", len(existing), flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("[2/4] Loading Kokoro...", flush=True)
print("Device:", device, flush=True)

pipeline = KPipeline(
    lang_code="a",
    device=device,
)

VOICE = "af_heart"
SPEED = 1.0
SAMPLE_RATE = 24000

WAV_DIR.mkdir(parents=True, exist_ok=True)

rows = []
reused = 0
synthesized = 0

for i, r in df.iterrows():

    prompt = str(r["prompt"])
    eid = str(r["example_id"])

    if prompt in existing:

        audio_path = existing[prompt]["audio_path"]
        audio_bytes = existing[prompt]["audio_bytes"]

        reused += 1

        print(
            f"[{i+1:03d}/450] reuse      {eid}",
            flush=True,
        )

    else:

        wav_path = WAV_DIR / f"{eid}.wav"

        print(
            f"[{i+1:03d}/450] synthesize {eid}",
            flush=True,
        )

        chunks = []

        generator = pipeline(
            prompt,
            voice=VOICE,
            speed=SPEED,
        )

        for _, _, audio in generator:
            chunks.append(np.asarray(audio))

        if not chunks:
            raise RuntimeError(
                f"No audio generated for {eid}"
            )

        audio = np.concatenate(chunks)

        sf.write(
            wav_path,
            audio,
            SAMPLE_RATE,
        )

        audio_path = wav_path.name
        audio_bytes = wav_path.read_bytes()

        synthesized += 1

    row = {}

    for col in df.columns:
        value = r[col]

        if pd.isna(value):
            row[col] = None
        else:
            row[col] = str(value)

    row.update({
        "audio_path": audio_path,
        "audio_bytes": audio_bytes,
        "tts_provider": "local",
        "tts_model": "hexgrad/Kokoro-82M",
        "tts_voice": VOICE,
        "tts_speed": str(SPEED),
        "sample_rate": str(SAMPLE_RATE),
    })

    rows.append(row)

print()
print("Reused:", reused)
print("Synthesized:", synthesized)

if reused != 250:
    raise RuntimeError(
        f"Expected to reuse 250 prompts, reused {reused}. "
        "Prompt alignment between old and full XSTest differs."
    )

if synthesized != 200:
    raise RuntimeError(
        f"Expected to synthesize 200 prompts, synthesized {synthesized}."
    )

print("[3/4] Packing complete 450-row Arrow dataset...", flush=True)

features = {}

for col in df.columns:
    features[col] = Value("string")

features.update({
    "audio_path": Value("string"),
    "audio_bytes": Value("binary"),
    "tts_provider": Value("string"),
    "tts_model": Value("string"),
    "tts_voice": Value("string"),
    "tts_speed": Value("string"),
    "sample_rate": Value("string"),
})

ds = Dataset.from_list(
    rows,
    features=Features(features),
)

if TMP_ARROW.exists():
    shutil.rmtree(TMP_ARROW)

ds.save_to_disk(str(TMP_ARROW))

assert len(ds) == 450

for i in range(len(ds)):
    assert ds[i]["audio_bytes"]
    assert len(ds[i]["audio_bytes"]) > 0

print("[4/4] Writing manifest and replacing old Arrow...", flush=True)

manifest_cols = [
    c for c in ds.column_names
    if c != "audio_bytes"
]

with MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=manifest_cols,
    )

    writer.writeheader()

    for row in rows:
        writer.writerow({
            c: row[c]
            for c in manifest_cols
        })

# Only replace the old Arrow after the new one is fully verified.
shutil.rmtree(OLD_ARROW)
TMP_ARROW.rename(OLD_ARROW)

shutil.rmtree(WAV_DIR, ignore_errors=True)

print()
print("====================================")
print("[OK] COMPLETE XSTest-Audio")
print("====================================")
print("Rows:", len(ds))
print("Reused old audio:", reused)
print("Newly synthesized:", synthesized)
print("Arrow:", OLD_ARROW)
print("Manifest:", MANIFEST)
print("TTS: hexgrad/Kokoro-82M")
print("Voice:", VOICE)
print("Device:", device)
