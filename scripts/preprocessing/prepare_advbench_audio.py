from pathlib import Path
import json
import csv
import shutil

from datasets import Dataset, Features, Value
from huggingface_hub import snapshot_download

ROOT = Path("data/advbench_audio")
RAW = ROOT / "_raw"
OUT = ROOT / "dataset"
MANIFEST = ROOT / "manifest.csv"

ROOT.mkdir(parents=True, exist_ok=True)

print("[1/3] Downloading/reusing official AdvBench-Audio files...")

snapshot_download(
    repo_id="WeifeiJin/AdvBench-Audio",
    repo_type="dataset",
    local_dir=str(RAW),
)

meta_path = RAW / "AdvBench_Audio.json"

if not meta_path.exists():
    raise FileNotFoundError(meta_path)

print("[2/3] Reading official metadata and WAV files...")

rows = []

with meta_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue

        x = json.loads(line)

        audio_name = x["audio"]

        # Official repository stores WAVs here.
        wav_path = RAW / "advbench_audios" / audio_name

        if not wav_path.exists():
            raise FileNotFoundError(wav_path)

        rows.append({
            "example_id": f"advbench_{i:04d}",
            "prompt": x["prompt"],
            "target": x["target"],

            # Store encoded WAV bytes directly.
            # This deliberately avoids datasets.Audio and torchcodec.
            "audio": {
                "path": audio_name,
                "bytes": wav_path.read_bytes(),
            },
        })

assert len(rows) == 520, f"Expected 520 examples, got {len(rows)}"

features = Features({
    "example_id": Value("string"),
    "prompt": Value("string"),
    "target": Value("string"),
    "audio": {
        "path": Value("string"),
        "bytes": Value("binary"),
    },
})

ds = Dataset.from_list(
    rows,
    features=features,
)

if OUT.exists():
    shutil.rmtree(OUT)

print("[3/3] Saving self-contained Arrow dataset...")

ds.save_to_disk(str(OUT))

with MANIFEST.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "example_id",
            "prompt",
            "target",
            "audio_filename",
        ],
    )
    writer.writeheader()

    for row in rows:
        writer.writerow({
            "example_id": row["example_id"],
            "prompt": row["prompt"],
            "target": row["target"],
            "audio_filename": row["audio"]["path"],
        })

# Verify embedded bytes before removing raw files.
check = ds[0]

assert check["audio"]["bytes"]
assert len(check["audio"]["bytes"]) > 0

print()
print("[OK] AdvBench-Audio complete")
print("Rows:", len(ds))
print("Columns:", ds.column_names)
print("Arrow:", OUT)
print("Manifest:", MANIFEST)
print("First audio bytes:", len(check["audio"]["bytes"]))

# Arrow is now self-contained.
shutil.rmtree(RAW)

print("[OK] Temporary raw repository removed")
