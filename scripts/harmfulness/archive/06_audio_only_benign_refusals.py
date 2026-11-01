#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
BASE = ROOT / "results/INITIAL_BENCHMARK/qwen2_audio/wk3/paired_harm_auroc"
SUMMARY = BASE / "behaviour_summary.csv"

KEYS = ["fold", "layer", "topic", "language", "condition"]
GROUPS = ("00", "01", "10", "11")
GROUP_TYPES = ("all", "language", "condition", "language_condition")
KEY_LAYERS = (14, 16, 18, 20, 22, 24, 28, 32)

FIELDS = [
    f"matched_{g}_{field}"
    for g in GROUPS
    for field in (
        "n", "pct", "valid_folds",
        "text_z", "audio_z", "audio_minus_text_z",
        "audio_minus_text_z_std",
    )
]


def load_predictions(modality, fold):
    path = BASE / modality / "decoder" / f"fold_{fold}_predictions.csv"

    df = pd.read_csv(
        path,
        usecols=KEYS + ["method", "benign_score", "benign_decision"],
    )
    df = df.loc[df["method"] == "mean"].drop(columns="method").copy()

    if df.empty or df.duplicated(KEYS).any():
        raise ValueError(f"Empty data or duplicate matching keys: {path}")

    if not df["benign_decision"].isin([0, 1]).all():
        raise ValueError(f"Invalid benign decisions: {path}")

    if not np.isfinite(df["benign_score"].to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid projection scores: {path}")

    stats = df.groupby("layer")["benign_score"].transform

    means = stats("mean")
    stds = stats("std")

    if stds.isna().any() or (stds <= 0).any():
        raise ValueError(f"Cannot standardize projection scores: {path}")

    df["z"] = (df["benign_score"] - means) / stds

    return df[KEYS + ["benign_decision", "z"]]


def build_matched():
    frames = []

    for fold in range(1, 6):
        text = load_predictions("text", fold)
        audio = load_predictions("audio", fold)

        matched = text.merge(
            audio,
            on=KEYS,
            how="outer",
            suffixes=("_text", "_audio"),
            validate="one_to_one",
            indicator=True,
        )

        if not matched["_merge"].eq("both").all():
            raise ValueError(f"Unmatched prompts in fold {fold}")

        matched = matched.drop(columns="_merge")

        matched["outcome"] = (
            matched["benign_decision_text"].astype(int).astype(str)
            + matched["benign_decision_audio"].astype(int).astype(str)
        )

        matched["audio_minus_text_z"] = (
            matched["z_audio"] - matched["z_text"]
        )

        frames.append(matched)
        print(f"Matched and standardized fold {fold}/5", flush=True)

    result = pd.concat(frames, ignore_index=True)

    if result.duplicated(["layer", "topic", "language", "condition"]).any():
        raise ValueError("Duplicate held-out examples across folds")

    return result


def summarize(matched):
    summaries = {}

    for group_type in GROUP_TYPES:
        if group_type == "all":
            df = matched.assign(summary_group="all")
        elif group_type == "language":
            df = matched.assign(summary_group=matched["language"])
        elif group_type == "condition":
            df = matched.assign(summary_group=matched["condition"])
        else:
            df = matched.assign(
                summary_group=(
                    matched["language"] + "__" + matched["condition"]
                )
            )

        for (layer, group), subgroup in df.groupby(
            ["layer", "summary_group"], sort=True
        ):
            result = {}
            total = len(subgroup)

            for outcome in GROUPS:
                rows = subgroup.loc[subgroup["outcome"] == outcome]
                n = len(rows)

                prefix = f"matched_{outcome}_"

                result[prefix + "n"] = n
                result[prefix + "pct"] = 100.0 * n / total

                fold_means = rows.groupby("fold")[
                    ["z_text", "z_audio", "audio_minus_text_z"]
                ].mean()

                valid = len(fold_means)
                result[prefix + "valid_folds"] = valid

                if valid:
                    result[prefix + "text_z"] = fold_means["z_text"].mean()
                    result[prefix + "audio_z"] = fold_means["z_audio"].mean()
                    result[prefix + "audio_minus_text_z"] = (
                        fold_means["audio_minus_text_z"].mean()
                    )
                    result[prefix + "audio_minus_text_z_std"] = (
                        fold_means["audio_minus_text_z"].std(ddof=1)
                        if valid >= 2 else np.nan
                    )
                else:
                    for field in (
                        "text_z",
                        "audio_z",
                        "audio_minus_text_z",
                        "audio_minus_text_z_std",
                    ):
                        result[prefix + field] = np.nan

            summaries[(int(layer), group_type, group)] = result

    return summaries


def update_summary(summaries):
    df = pd.read_csv(SUMMARY)

    required = {
        "modality", "component", "layer", "method",
        "group_type", "group", "split",
    }

    if not required.issubset(df.columns):
        raise ValueError("Unexpected behaviour summary schema")

    for field in FIELDS:
        df[field] = np.nan

    count = 0

    for idx, row in df.iterrows():
        if not (
            row["component"] == "decoder"
            and row["method"] == "mean"
            and row["split"] == "benign"
        ):
            continue

        key = (
            int(row["layer"]),
            row["group_type"],
            row["group"],
        )

        if key not in summaries:
            raise ValueError(f"Missing summary group: {key}")

        for name, value in summaries[key].items():
            df.at[idx, name] = value

        count += 1

    if count == 0:
        raise ValueError("No matching summary rows were updated")

    tmp = SUMMARY.with_name(SUMMARY.name + ".tmp")

    try:
        df.to_csv(tmp, index=False, float_format="%.6f")
        os.replace(tmp, SUMMARY)
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"\nUpdated {count} existing rows: {SUMMARY}")
    return df


def print_results(df):
    overall = df.loc[
        (df["component"] == "decoder")
        & (df["method"] == "mean")
        & (df["split"] == "benign")
        & (df["group_type"] == "all")
        & (df["group"] == "all")
        & (df["modality"] == "audio")
    ].copy()

    def show(layer, group_type="all", group="all"):
        subset = df.loc[
            (df["layer"] == layer)
            & (df["component"] == "decoder")
            & (df["method"] == "mean")
            & (df["split"] == "benign")
            & (df["group_type"] == group_type)
            & (df["group"] == group)
            & (df["modality"] == "audio")
        ]

        if len(subset) != 1:
            raise ValueError(
                f"Expected one summary: {layer}, {group_type}, {group}"
            )

        r = subset.iloc[0]
        rows = []

        for g in GROUPS:
            p = f"matched_{g}_"
            rows.append({
                "outcome": g,
                "N": int(r[p + "n"]),
                "pct": r[p + "pct"],
                "valid_folds": int(r[p + "valid_folds"]),
                "text_z": r[p + "text_z"],
                "audio_z": r[p + "audio_z"],
                "audio-text": r[p + "audio_minus_text_z"],
                "delta_fold_SD": r[p + "audio_minus_text_z_std"],
            })

        return pd.DataFrame(rows)

    print("\n" + "=" * 92)
    print("MATCHED BENIGN OUTCOMES — DECODER LAYER 20")
    print("00 = both accept; 01 = audio-only refusal")
    print("10 = text-only refusal; 11 = both refuse")
    print("=" * 92)

    print(
        show(20).to_string(
            index=False, na_rep="NA",
            float_format=lambda x: f"{x:+.3f}"
        )
    )

    print("\n" + "=" * 92)
    print("ALL FOUR OUTCOMES — IMPORTANT DECODER LAYERS")
    print("=" * 92)

    for layer in KEY_LAYERS:
        print(f"\nLayer {layer}")
        print(
            show(layer)[
                ["outcome", "N", "text_z", "audio_z",
                 "audio-text", "delta_fold_SD", "valid_folds"]
            ].to_string(
                index=False, na_rep="NA",
                float_format=lambda x: f"{x:+.3f}"
            )
        )

    print("\n" + "=" * 92)
    print("LAYER 20 — LANGUAGE BREAKDOWN")
    print("=" * 92)

    for language in ("chinese", "english", "russian", "spanish"):
        print(f"\n{language}")
        print(
            show(20, "language", language)[
                ["outcome", "N", "pct", "text_z", "audio_z", "audio-text"]
            ].to_string(
                index=False, na_rep="NA",
                float_format=lambda x: f"{x:+.3f}"
            )
        )

    print("\n" + "=" * 92)
    print("LAYER 20 — ATTACK CONDITION BREAKDOWN")
    print("=" * 92)

    for condition in (
        "original", "literal",
        "narrative_framing", "content_dilution"
    ):
        print(f"\n{condition}")
        print(
            show(20, "condition", condition)[
                ["outcome", "N", "pct", "text_z", "audio_z", "audio-text"]
            ].to_string(
                index=False, na_rep="NA",
                float_format=lambda x: f"{x:+.3f}"
            )
        )

    print("\n" + "=" * 92)
    print("INTERPRETATION")
    print("=" * 92)
    print("Scores standardized against all benign prompts within each")
    print("modality, fold, and decoder layer.")
    print("Means are computed within outcome group, then averaged across folds.")
    print("Positive audio-text = higher relative harmfulness score in audio.")
    print("N and pct show the true frequency of each outcome.")
    print("Fold SD measures variation in the audio-text gap across folds.")
    print("Missing values mean the outcome did not occur.")
    print("This is an observational association, not a causal intervention.")


def main():
    matched = build_matched()
    summaries = summarize(matched)
    summary = update_summary(summaries)
    print_results(summary)


if __name__ == "__main__":
    main()
