from pathlib import Path
import csv
import shutil
from collections import Counter

from datasets import Audio, Dataset, Features, Value
from huggingface_hub import snapshot_download


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

REPO_ID = "tsinghua-ee/SACRED-Bench"

DOWNLOAD_DIR = ROOT / "data/sacred_bench/full_official"
DATASET_DIR = ROOT / "data/sacred_bench/full_dataset"
MANIFEST_PATH = ROOT / "data/sacred_bench/full_manifest.csv"

FAMILIES = (
    (
        "contextual_audio_cues",
        "Contextual_Audio_Cues/test/audio",
        717,
    ),
    (
        "multi_speaker_dialogue",
        "Multi-speaker_Dialogue/test/audio",
        1364,
    ),
    (
        "speech_overlap",
        "Speech_Overlap/test/audio",
        400,
    ),
)

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
}

EXPECTED_TOTAL = 2481

MANIFEST_FIELDS = [
    "example_id",
    "base_example_id",
    "benchmark",
    "split",
    "category",
    "condition",
    "audio_index",
    "source_path",
]


def download_release():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[download] {REPO_ID}")
    print(f"[download] destination: {DOWNLOAD_DIR}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DOWNLOAD_DIR),
        max_workers=2,
    )


def collect_rows():
    rows = []

    for category, relative_root, expected_count in FAMILIES:
        family_root = DOWNLOAD_DIR / relative_root

        if not family_root.exists():
            raise FileNotFoundError(
                f"Expected SACRED directory missing:\n{family_root}"
            )

        audio_files = sorted(
            (
                p
                for p in family_root.rglob("*")
                if p.is_file()
                and p.suffix.lower() in AUDIO_EXTENSIONS
            ),
            key=lambda p: p.relative_to(DOWNLOAD_DIR).as_posix(),
        )

        if len(audio_files) != expected_count:
            raise RuntimeError(
                f"{category}: expected {expected_count} audio files, "
                f"found {len(audio_files)} under {family_root}"
            )

        print(
            f"[family] {category}: {len(audio_files)}",
            flush=True,
        )

        for p in audio_files:
            audio_index = len(rows)
            example_id = f"sacred_full_{audio_index:04d}"
            source_path = p.relative_to(DOWNLOAD_DIR).as_posix()

            rows.append(
                {
                    "example_id": example_id,
                    "base_example_id": example_id,
                    "benchmark": "sacred_bench",
                    "split": "test",
                    "category": category,
                    "condition": category,
                    "audio_index": audio_index,
                    "source_path": source_path,
                    "_absolute_audio_path": str(p),
                }
            )

    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} SACRED examples, got {len(rows)}"
        )

    ids = [r["example_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate example_id values detected")

    source_paths = [r["source_path"] for r in rows]
    if len(source_paths) != len(set(source_paths)):
        raise RuntimeError("Duplicate source_path values detected")

    return rows


def write_manifest(rows):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=MANIFEST_FIELDS,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row[field]
                    for field in MANIFEST_FIELDS
                }
            )

    print(f"[manifest] {MANIFEST_PATH}")


def dataset_generator(rows):
    for i, row in enumerate(rows, 1):
        audio_path = Path(row["_absolute_audio_path"])

        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        yield {
            "example_id": row["example_id"],
            "base_example_id": row["base_example_id"],
            "benchmark": row["benchmark"],
            "split": row["split"],
            "category": row["category"],
            "condition": row["condition"],
            "audio_index": row["audio_index"],
            "source_path": row["source_path"],
            "audio": {
                "bytes": audio_path.read_bytes(),
                "path": row["source_path"],
            },
        }

        if i % 100 == 0 or i == len(rows):
            print(
                f"[arrow] encoded {i}/{len(rows)}",
                flush=True,
            )


def build_arrow_dataset(rows):
    features = Features(
        {
            "example_id": Value("string"),
            "base_example_id": Value("string"),
            "benchmark": Value("string"),
            "split": Value("string"),
            "category": Value("string"),
            "condition": Value("string"),
            "audio_index": Value("int64"),
            "source_path": Value("string"),
            "audio": Audio(decode=False),
        }
    )

    ds = Dataset.from_generator(
        lambda: dataset_generator(rows),
        features=features,
    )

    if len(ds) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Arrow dataset has {len(ds)} rows; "
            f"expected {EXPECTED_TOTAL}"
        )

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    ds.save_to_disk(str(DATASET_DIR))

    print(f"[dataset] {DATASET_DIR}")


def verify_saved_dataset():
    from datasets import load_from_disk

    ds = load_from_disk(str(DATASET_DIR))

    if len(ds) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Saved dataset has {len(ds)} rows; "
            f"expected {EXPECTED_TOTAL}"
        )

    expected_columns = MANIFEST_FIELDS + ["audio"]

    if ds.column_names != expected_columns:
        raise RuntimeError(
            "Unexpected saved dataset schema:\n"
            f"expected: {expected_columns}\n"
            f"got:      {ds.column_names}"
        )

    counts = Counter(ds["category"])

    expected_counts = {
        category: expected_count
        for category, _, expected_count in FAMILIES
    }

    if dict(counts) != expected_counts:
        raise RuntimeError(
            "Unexpected saved family counts:\n"
            f"expected: {expected_counts}\n"
            f"got:      {dict(counts)}"
        )

    manifest_rows = list(
        csv.DictReader(
            MANIFEST_PATH.open(
                "r",
                encoding="utf-8",
            )
        )
    )

    if len(manifest_rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Manifest has {len(manifest_rows)} rows; "
            f"expected {EXPECTED_TOTAL}"
        )

    for i, (manifest_row, dataset_row) in enumerate(
        zip(manifest_rows, ds),
    ):
        for field in MANIFEST_FIELDS:
            manifest_value = manifest_row[field]
            dataset_value = dataset_row[field]

            if field == "audio_index":
                dataset_value = str(dataset_value)

            if manifest_value != dataset_value:
                raise RuntimeError(
                    f"Manifest/dataset mismatch at row {i}, "
                    f"field {field}: "
                    f"{manifest_value!r} != {dataset_value!r}"
                )

        audio = dataset_row["audio"]

        if not isinstance(audio, dict):
            raise RuntimeError(
                f"Row {i}: unexpected audio value type "
                f"{type(audio).__name__}"
            )

        if audio.get("bytes") is None:
            raise RuntimeError(
                f"Row {i}: audio bytes are not embedded"
            )

    print()
    print("===== SACRED FULL DATASET =====")
    print(f"rows: {len(ds)}")
    print(f"families: {dict(counts)}")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"dataset: {DATASET_DIR}")
    print("[OK] manifest, Arrow rows, schema, and embedded audio verified")


def main():
    download_release()
    rows = collect_rows()
    write_manifest(rows)
    print("[dataset] using official SACRED release files directly")
    print("[dataset] official SACRED release verified; no Arrow copy required")


if __name__ == "__main__":
    main()
