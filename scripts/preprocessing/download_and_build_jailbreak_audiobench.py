from pathlib import Path
import csv
import json
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import snapshot_download


ROOT = Path("data/jailbreak_audiobench")

FULL = ROOT / "full_download"
SMALL = ROOT / "small_dataset"
MANIFEST = ROOT / "small_manifest.csv"
SUMMARY = ROOT / "small_summary.json"

REPO = "researchtopic/Jailbreak-AudioBench-Plus"

PREFIX = "Explicit_small_gpt4o_eval/"


def normalise_path(x):
    return str(x).replace("\\", "/")


# ============================================================
# 1. FULL DOWNLOAD
# ============================================================

print("=" * 72, flush=True)
print("STEP 1/4 — FULL JAILBREAK-AUDIOBENCH-PLUS DOWNLOAD", flush=True)
print("=" * 72, flush=True)
print("Destination:", FULL, flush=True)

FULL.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    local_dir=str(FULL),
)

print("\n[OK] Full repository downloaded", flush=True)


# ============================================================
# 2. FIND LOCAL PARQUET FILES
# ============================================================

print("\n" + "=" * 72, flush=True)
print("STEP 2/4 — LOCATING PARQUET FILES", flush=True)
print("=" * 72, flush=True)

parquets = sorted(FULL.rglob("*.parquet"))

if not parquets:
    raise RuntimeError(
        f"No parquet files found under {FULL}. "
        "Full download has been preserved for inspection."
    )

print("Parquet files:", len(parquets), flush=True)

for p in parquets[:5]:
    print(" ", p, flush=True)

if len(parquets) > 5:
    print("  ...", flush=True)


# ============================================================
# 3. EXTRACT OFFICIAL EXPLICIT SMALL SECTION
#
# Important:
#   - read raw parquet directly
#   - do NOT use datasets.Audio
#   - do NOT decode audio
#   - preserve encoded bytes exactly as released
#   - deduplicate using filepath
# ============================================================

print("\n" + "=" * 72, flush=True)
print("STEP 3/4 — EXTRACTING Explicit_small_gpt4o_eval", flush=True)
print("=" * 72, flush=True)

rows_by_filepath = {}

raw_matching_rows = 0

for fi, parquet_path in enumerate(parquets, 1):

    print(
        f"[{fi:02d}/{len(parquets):02d}] scanning {parquet_path.name}",
        flush=True,
    )

    pf = pq.ParquetFile(parquet_path)

    names = pf.schema_arrow.names

    if "filepath" not in names:
        print("  skip: no filepath column", flush=True)
        continue

    if "audio" not in names:
        print("  skip: no audio column", flush=True)
        continue

    for batch in pf.iter_batches(
        batch_size=256,
        columns=["filepath", "audio"],
    ):

        table = pa.Table.from_batches([batch])

        paths = table["filepath"].to_pylist()
        audios = table["audio"].to_pylist()

        for filepath, audio in zip(paths, audios):

            if filepath is None:
                continue

            fp = normalise_path(filepath)

            if not fp.startswith(PREFIX):
                continue

            raw_matching_rows += 1

            # Deduplicate by the benchmark's own filepath.
            if fp in rows_by_filepath:
                continue

            if audio is None:
                raise RuntimeError(
                    f"Matched row has null audio: {fp}"
                )

            # HF audio parquet representation is normally
            # {"bytes": ..., "path": ...}.
            if isinstance(audio, dict):
                audio_bytes = audio.get("bytes")
                audio_path = audio.get("path") or Path(fp).name
            else:
                raise RuntimeError(
                    f"Unexpected audio representation for {fp}: "
                    f"{type(audio)}"
                )

            if not audio_bytes:
                raise RuntimeError(
                    f"No encoded audio bytes found for {fp}"
                )

            relative = fp[len(PREFIX):]

            pieces = relative.split("/")

            condition = pieces[0] if pieces else ""

            variant = (
                pieces[1]
                if len(pieces) >= 3
                else ""
            )

            filename = pieces[-1]

            rows_by_filepath[fp] = {
                "filepath": fp,
                "relative_filepath": relative,
                "condition": condition,
                "variant": variant,
                "filename": filename,
                "audio": {
                    "path": audio_path,
                    "bytes": audio_bytes,
                },
            }


rows = list(rows_by_filepath.values())

rows.sort(key=lambda x: x["filepath"])

print()
print("Raw matching rows:", raw_matching_rows, flush=True)
print("Unique matching files:", len(rows), flush=True)

if not rows:
    raise RuntimeError(
        "No Explicit_small_gpt4o_eval files found. "
        "Full download has been preserved."
    )


# ============================================================
# CONDITION COUNTS
# ============================================================

condition_counts = {}

for row in rows:
    c = row["condition"]
    condition_counts[c] = condition_counts.get(c, 0) + 1

print("\nUnique files by condition:", flush=True)

for key in sorted(condition_counts):
    print(
        f"  {key:30s} {condition_counts[key]}",
        flush=True,
    )


# ============================================================
# 4. SAVE SELF-CONTAINED ARROW SUBSET
# ============================================================

print("\n" + "=" * 72, flush=True)
print("STEP 4/4 — SAVING SMALL ARROW DATASET", flush=True)
print("=" * 72, flush=True)

if SMALL.exists():
    shutil.rmtree(SMALL)

ds = Dataset.from_list(rows)

ds.save_to_disk(str(SMALL))

# Verify audio bytes survived Arrow packing.
assert len(ds) == len(rows)

for i in [0, len(ds) // 2, len(ds) - 1]:
    assert ds[i]["audio"]["bytes"]
    assert len(ds[i]["audio"]["bytes"]) > 0


# ============================================================
# MANIFEST
# ============================================================

with MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fields = [
        "example_id",
        "filepath",
        "relative_filepath",
        "condition",
        "variant",
        "filename",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for i, row in enumerate(rows):
        writer.writerow({
            "example_id": f"jab_small_{i:05d}",
            "filepath": row["filepath"],
            "relative_filepath": row["relative_filepath"],
            "condition": row["condition"],
            "variant": row["variant"],
            "filename": row["filename"],
        })


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "source_repo": REPO,
    "full_download": str(FULL),
    "small_dataset": str(SMALL),
    "selection_prefix": PREFIX,
    "number_of_parquet_files": len(parquets),
    "raw_matching_rows": raw_matching_rows,
    "unique_small_rows": len(rows),
    "condition_counts": condition_counts,
}

SUMMARY.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print()
print("=" * 72)
print("[OK] JAILBREAK-AUDIOBENCH OVERNIGHT PREPARATION COMPLETE")
print("=" * 72)
print("Full repository:")
print(" ", FULL)
print()
print("Small Arrow dataset:")
print(" ", SMALL)
print()
print("Small manifest:")
print(" ", MANIFEST)
print()
print("Summary:")
print(" ", SUMMARY)
print()
print("Unique small rows:", len(rows))
print()

if len(rows) == 5502:
    print("[OK] Exactly 5,502 files: matches reported Explicit Small size.")
else:
    print(
        "[NOTE] Unique count is not 5,502.",
        "Nothing has been deleted.",
        "Inspect small_summary.json before using it experimentally.",
    )
