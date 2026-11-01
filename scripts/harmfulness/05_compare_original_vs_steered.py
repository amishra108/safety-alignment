#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
from pathlib import Path


MODALITIES = ("audio", "text")
SPLITS = ("benign", "harmful")
DIVIDER = "=" * 80


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original and steered responses by task_id."
    )
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--steered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing comparison TXT files.",
    )
    return parser.parse_args()


def read_responses(path, modality, expected_split=None):
    if not path.is_file():
        raise FileNotFoundError(path)

    records = {}

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            record = json.loads(line)

            required = ("task_id", "split", "response")
            missing = [key for key in required if key not in record]

            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields {missing}"
                )

            task_id = record["task_id"]
            split = record["split"]

            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"{path}:{line_number}: invalid task_id")

            if split not in SPLITS:
                raise ValueError(
                    f"{path}:{line_number}: invalid split {split!r}"
                )

            if expected_split is not None and split != expected_split:
                raise ValueError(
                    f"{path}:{line_number}: expected split "
                    f"{expected_split!r}, got {split!r}"
                )

            if "modality" in record and record["modality"] != modality:
                raise ValueError(
                    f"{path}:{line_number}: modality mismatch"
                )

            if not task_id.endswith(f"__{modality}"):
                raise ValueError(
                    f"{path}:{line_number}: task_id modality mismatch"
                )

            if not isinstance(record["response"], str):
                raise ValueError(
                    f"{path}:{line_number}: response is not a string"
                )

            if task_id in records:
                raise ValueError(
                    f"{path}:{line_number}: duplicate task_id {task_id}"
                )

            records[task_id] = record

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def flatten(value):
    return " ".join(value.split())


def first_five_match(original, steered):
    original_words = flatten(original).casefold().split()
    steered_words = flatten(steered).casefold().split()

    return (
        len(original_words) >= 5
        and len(steered_words) >= 5
        and original_words[:5] == steered_words[:5]
    )


def format_pair(task_id, original, steered):
    return (
        f"TASK ID: {task_id}\n"
        f"ORIGINAL: {flatten(original)}\n"
        f"STEERED: {flatten(steered)}\n\n"
    )


def prepare_comparison(originals, steered, split, output):
    original_subset = {
        task_id: record
        for task_id, record in originals.items()
        if record["split"] == split
    }

    original_ids = set(original_subset)
    steered_ids = set(steered)

    if original_ids != steered_ids:
        missing = sorted(original_ids - steered_ids)
        extra = sorted(steered_ids - original_ids)

        raise ValueError(
            f"{output.name}: task_id mismatch: "
            f"{len(missing)} missing, {len(extra)} unexpected. "
            f"Missing examples: {missing[:3]}; "
            f"unexpected examples: {extra[:3]}"
        )

    different = []
    matching = []

    for task_id in original_subset:
        original_response = original_subset[task_id]["response"]
        steered_response = steered[task_id]["response"]

        block = format_pair(
            task_id,
            original_response,
            steered_response,
        )

        if first_five_match(original_response, steered_response):
            matching.append(block)
        else:
            different.append(block)

    content = (
        "".join(different)
        + f"\n{DIVIDER}\n"
        + "SAME FIRST FIVE WORDS\n"
        + f"{DIVIDER}\n\n"
        + "".join(matching)
    )

    return content, len(different), len(matching)


def write_atomic(path, content):
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)

        os.replace(temporary, path)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    args = parse_args()

    jobs = []

    for modality in MODALITIES:
        original_path = (
            args.original_dir
            / f"responses.all.{modality}.jsonl"
        )

        originals = read_responses(original_path, modality)

        for split in SPLITS:
            steered_path = (
                args.steered_dir
                / f"responses.all.{modality}.{split}.jsonl"
            )

            steered = read_responses(
                steered_path,
                modality,
                expected_split=split,
            )

            for task_id, record in steered.items():
                if "model" in record and record["model"] != args.model:
                    raise ValueError(
                        f"{steered_path}: {task_id}: wrong model"
                    )

            output_path = (
                args.output_dir
                / f"pairs_originalvssteered_"
                f"{args.model}_{modality}_{split}.txt"
            )

            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{output_path} already exists; use --overwrite"
                )

            content, different, matching = prepare_comparison(
                originals,
                steered,
                split,
                output_path,
            )

            jobs.append(
                (output_path, content, different, matching)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nVALIDATION PASSED")
    print("All task IDs matched between original and steered files.")
    print()

    for output_path, content, different, matching in jobs:
        write_atomic(output_path, content)

        print(f"{output_path.name}")
        print(f"  Different first five words: {different}")
        print(f"  Same first five words:      {matching}")
        print(f"  Total pairs:                {different + matching}")
        print()

    print("COMPLETE: Four comparison TXT files written.")


if __name__ == "__main__":
    main()
