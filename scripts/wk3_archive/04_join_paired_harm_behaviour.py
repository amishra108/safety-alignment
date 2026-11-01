#!/usr/bin/env python3

import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
MODEL = "qwen2_audio"
BASE = ROOT / "results/INITIAL_BENCHMARK" / MODEL / "wk3"
FOLDS = BASE / "dataset_summary/folds"
OUTPUT = BASE / "paired_harm_auroc"

COMPONENTS = (
    ("text", "decoder"),
    ("audio", "decoder"),
    ("audio", "audio_encoder"),
)

SPLITS = ("benign", "harmful")
GROUP_TYPES = ("all", "language", "condition", "language_condition")

SUMMARY_FIELDS = (
    "model", "modality", "component", "layer", "method",
    "group_type", "group", "split",
    "n", "decision_0", "decision_1",
    "decision_0_pct", "decision_1_pct",
    "mean_score_decision_0", "mean_score_decision_1",
)


def read_decisions():
    decisions = {}

    for fold in range(1, 6):
        path = FOLDS / f"fold_{fold}_test.csv"

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "task_id", "split", "modality", "language",
                "condition", "decision", "test_fold", "partition"
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing manifest fields: {path}")

            count = 0
            for row in reader:
                if row["partition"] != "test":
                    raise ValueError(f"Not a test row: {row['task_id']}")
                if int(row["test_fold"]) != fold:
                    raise ValueError(f"Fold mismatch: {row['task_id']}")
                if row["split"] not in SPLITS:
                    raise ValueError(f"Invalid split: {row['task_id']}")
                if row["decision"] not in ("0", "1"):
                    raise ValueError(f"Invalid decision: {row['task_id']}")

                key = (fold, row["task_id"])
                if key in decisions:
                    raise ValueError(f"Duplicate decision: {key}")

                decisions[key] = int(row["decision"])
                count += 1

            if count != 1280:
                raise ValueError(
                    f"Fold {fold}: expected 1280 test examples, found {count}"
                )

    if len(decisions) != 6400:
        raise ValueError("Incomplete five-fold decision coverage")

    return decisions


def task_id(split, row):
    return (
        f"jbb_{split}_{int(row['topic']):03d}__"
        f"{row['condition']}__{row['language']}__{row['modality']}"
    )


def groups_for(row):
    return (
        ("all", "all"),
        ("language", row["language"]),
        ("condition", row["condition"]),
        (
            "language_condition",
            f"{row['language']}__{row['condition']}"
        ),
    )


def process_predictions(path, decisions, aggregates=None):
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    seen = set()

    try:
        with (
            path.open(newline="", encoding="utf-8") as source,
            temporary.open("w", newline="", encoding="utf-8") as destination
        ):
            reader = csv.DictReader(source)
            original_fields = reader.fieldnames

            required = {
                "model", "modality", "component", "fold",
                "layer", "method", "topic", "language",
                "condition", "benign_score", "harmful_score",
                "paired_correct"
            }
            if not required.issubset(original_fields or []):
                raise ValueError(f"Missing prediction fields: {path}")

            new_fields = tuple(
                field for field in original_fields
                if field not in ("benign_decision", "harmful_decision")
            ) + ("benign_decision", "harmful_decision")

            writer = csv.DictWriter(destination, fieldnames=new_fields)
            writer.writeheader()

            for row in reader:
                if row["model"] != MODEL:
                    raise ValueError(f"Unexpected model: {path}")

                fold = int(row["fold"])
                if not 1 <= fold <= 5:
                    raise ValueError(f"Invalid fold: {fold}")

                pair_key = (
                    fold, row["layer"], row["method"],
                    row["topic"], row["language"], row["condition"]
                )
                if pair_key in seen:
                    raise ValueError(f"Duplicate prediction: {pair_key}")
                seen.add(pair_key)

                for split in SPLITS:
                    key = (fold, task_id(split, row))
                    if key not in decisions:
                        raise ValueError(f"Missing decision: {key}")

                    row[f"{split}_decision"] = decisions[key]

                    if aggregates is not None:
                        decision = decisions[key]
                        score = float(row[f"{split}_score"])

                        for group_type, group in groups_for(row):
                            aggregate_key = (
                                row["model"], row["modality"],
                                row["component"], int(row["layer"]),
                                row["method"], group_type, group, split
                            )
                            counts, sums = aggregates[aggregate_key]
                            counts[decision] += 1
                            sums[decision] += score

                writer.writerow(row)
                count += 1

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Updated {path}: {count:,} rows", flush=True)
    return count


def write_summary(path, aggregates):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for key, (counts, sums) in sorted(aggregates.items()):
            total = counts[0] + counts[1]
            writer.writerow({
                "model": key[0],
                "modality": key[1],
                "component": key[2],
                "layer": key[3],
                "method": key[4],
                "group_type": key[5],
                "group": key[6],
                "split": key[7],
                "n": total,
                "decision_0": counts[0],
                "decision_1": counts[1],
                "decision_0_pct": f"{100 * counts[0] / total:.2f}",
                "decision_1_pct": f"{100 * counts[1] / total:.2f}",
                "mean_score_decision_0": (
                    f"{sums[0] / counts[0]:.4f}" if counts[0] else ""
                ),
                "mean_score_decision_1": (
                    f"{sums[1] / counts[1]:.4f}" if counts[1] else ""
                ),
            })


def main():
    decisions = read_decisions()
    print(f"Loaded {len(decisions):,} adjudicated test decisions")

    aggregates = defaultdict(
        lambda: ([0, 0], [0.0, 0.0])
    )

    for modality, component in COMPONENTS:
        directory = OUTPUT / modality / component

        for fold in range(1, 6):
            process_predictions(
                directory / f"fold_{fold}_predictions.csv",
                decisions
            )

        process_predictions(
            directory / "heldout_predictions.csv",
            decisions,
            aggregates
        )

    summary = OUTPUT / "behaviour_summary.csv"
    write_summary(summary, aggregates)

    print(f"\nCOMPLETE: {summary}", flush=True)
    print(f"Summary rows: {len(aggregates):,}", flush=True)


if __name__ == "__main__":
    main()
