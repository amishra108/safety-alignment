#!/usr/bin/env python3

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
BASE = ROOT / "results/INITIAL_BENCHMARK/qwen2_audio/wk3/paired_harm_auroc"

COMPONENTS = (
    ("text", "decoder"),
    ("audio", "decoder"),
    ("audio", "audio_encoder"),
)

STAT_FIELDS = (
    "valid_folds",
    "positive_folds",
    "negative_folds",
    "mean_gap",
    "std_gap",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
    "cohens_d",
    "cohens_d_std",
    "cohens_d_valid_folds",
)


def groups_for(row):
    return (
        ("all", "all"),
        ("language", row["language"]),
        ("condition", row["condition"]),
        (
            "language_condition",
            f"{row['language']}__{row['condition']}",
        ),
    )


def main():
    # Each accumulator stores count, mean and M2 for stable variance.
    accum = defaultdict(lambda: [0, 0.0, 0.0])

    for modality, component in COMPONENTS:
        for fold in range(1, 6):
            path = (
                BASE / modality / component
                / f"fold_{fold}_predictions.csv"
            )

            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)

                required = {
                    "fold", "layer", "method", "language", "condition",
                    "benign_score", "harmful_score",
                    "benign_decision", "harmful_decision",
                }
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"Missing prediction columns: {path}")

                for row in reader:
                    if int(row["fold"]) != fold:
                        raise ValueError(f"Fold mismatch: {path}")

                    for split in ("benign", "harmful"):
                        decision = int(row[f"{split}_decision"])
                        if decision not in (0, 1):
                            raise ValueError(f"Invalid decision: {path}")

                        score = float(row[f"{split}_score"])
                        if not np.isfinite(score):
                            raise ValueError(f"Nonfinite score: {path}")

                        for group_type, group in groups_for(row):
                            key = (
                                modality,
                                component,
                                int(row["layer"]),
                                row["method"],
                                group_type,
                                group,
                                split,
                                fold,
                                decision,
                            )

                            acc = accum[key]
                            acc[0] += 1
                            delta = score - acc[1]
                            acc[1] += delta / acc[0]
                            acc[2] += delta * (score - acc[1])

            print(f"Processed {modality}/{component} fold {fold}", flush=True)

    gaps = defaultdict(dict)
    effect_sizes = defaultdict(dict)

    for key in {k[:-1] for k in accum}:
        zero = accum.get(key + (0,))
        one = accum.get(key + (1,))

        if zero is None or one is None:
            continue

        n0, mean0, m2_0 = zero
        n1, mean1, m2_1 = one

        split = key[6]
        fold = key[7]
        group_key = key[:7]

        gap = (
            mean0 - mean1
            if split == "harmful"
            else mean1 - mean0
        )

        gaps[group_key][fold] = gap

        if n0 < 2 or n1 < 2:
            continue

        pooled_variance = (
            m2_0 + m2_1
        ) / (n0 + n1 - 2)

        if pooled_variance > 0:
            effect_sizes[group_key][fold] = (
                gap / np.sqrt(pooled_variance)
            )

    summary = BASE / "behaviour_summary.csv"
    temporary = summary.with_name(summary.name + ".tmp")
    row_count = 0

    try:
        with (
            summary.open(newline="", encoding="utf-8") as source,
            temporary.open("w", newline="", encoding="utf-8") as destination
        ):
            reader = csv.DictReader(source)
            original_fields = reader.fieldnames

            required = {
                "modality", "component", "layer", "method",
                "group_type", "group", "split",
            }
            if not required.issubset(original_fields or []):
                raise ValueError("Unexpected behaviour summary schema")

            fields = [
                field for field in original_fields
                if field not in STAT_FIELDS
            ] + list(STAT_FIELDS)

            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()

            for row in reader:
                key = (
                    row["modality"],
                    row["component"],
                    int(row["layer"]),
                    row["method"],
                    row["group_type"],
                    row["group"],
                    row["split"],
                )

                fold_gaps = gaps.get(key, {})
                fold_ds = effect_sizes.get(key, {})

                gap_values = np.array(
                    list(fold_gaps.values()), dtype=float
                )
                d_values = np.array(
                    list(fold_ds.values()), dtype=float
                )

                row.update({
                    "valid_folds": len(gap_values),
                    "positive_folds": int(np.sum(gap_values > 0)),
                    "negative_folds": int(np.sum(gap_values < 0)),
                    "mean_gap": (
                        f"{gap_values.mean():.4f}"
                        if len(gap_values) else ""
                    ),
                    "std_gap": (
                        f"{gap_values.std(ddof=1):.4f}"
                        if len(gap_values) >= 2 else ""
                    ),
                    "cohens_d": (
                        f"{d_values.mean():.4f}"
                        if len(d_values) else ""
                    ),
                    "cohens_d_std": (
                        f"{d_values.std(ddof=1):.4f}"
                        if len(d_values) >= 2 else ""
                    ),
                    "cohens_d_valid_folds": len(d_values),
                })

                for fold in range(1, 6):
                    value = fold_gaps.get(fold)
                    row[f"fold_{fold}"] = (
                        f"{value:.4f}" if value is not None else ""
                    )

                writer.writerow(row)
                row_count += 1

        os.replace(temporary, summary)

    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"\nUpdated {row_count:,} rows: {summary}")

    df = pd.read_csv(summary)
    df = df[
        (df["component"] == "decoder")
        & (df["method"] == "mean")
    ].copy()

    df["folds"] = (
        df["positive_folds"].astype("Int64").astype(str)
        + "/"
        + df["valid_folds"].astype("Int64").astype(str)
    )

    df["d_result"] = df.apply(
        lambda r: (
            f"{r['cohens_d']:+.3f} +/- {r['cohens_d_std']:.3f}"
            if pd.notna(r["cohens_d_std"])
            else f"{r['cohens_d']:+.3f}"
            if pd.notna(r["cohens_d"])
            else "NA"
        ),
        axis=1,
    )

    print("\n" + "=" * 100)
    print("1. OVERALL COHEN'S d — ALL DECODER LAYERS")
    print("=" * 100)

    overall = df[
        (df["group_type"] == "all")
        & (df["group"] == "all")
    ].sort_values(["layer", "modality", "split"])

    print(
        overall[
            ["layer", "modality", "split", "cohens_d",
             "cohens_d_std", "cohens_d_valid_folds", "folds"]
        ].to_string(index=False, na_rep="NA", float_format=lambda x: f"{x:.3f}")
    )

    print("\n" + "=" * 100)
    print("2. KEY LAYERS — TEXT VS AUDIO")
    print("=" * 100)

    key_layers = overall[
        overall["layer"].isin((14, 16, 20, 24, 28, 32))
    ]

    print(
        key_layers[
            ["layer", "modality", "split",
             "mean_gap", "std_gap", "d_result", "folds"]
        ].to_string(index=False, na_rep="NA", float_format=lambda x: f"{x:.3f}")
    )

    print("\n" + "=" * 100)
    print("3. LAYER 20 — LANGUAGE AND ATTACK CONDITION")
    print("=" * 100)

    breakdown = df[
        (df["layer"] == 20)
        & df["group_type"].isin(("language", "condition"))
    ].sort_values(
        ["group_type", "group", "modality", "split"]
    )

    print(
        breakdown[
            ["group_type", "group", "modality", "split",
             "decision_1_pct", "d_result",
             "cohens_d_valid_folds", "folds"]
        ].to_string(index=False, na_rep="NA", float_format=lambda x: f"{x:.3f}")
    )

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("Harmful: positive d = higher scores for noncompliance.")
    print("Benign:  positive d = higher scores for refusal.")
    print("Larger |d| = greater standardized behavioural separation.")
    print("d +/- SD = mean and sample SD across valid held-out folds.")
    print("Compare text and audio at the SAME decoder layer.")
    print("NA = insufficient outcome counts or zero pooled variance.")


if __name__ == "__main__":
    main()
