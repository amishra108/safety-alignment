#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

MODELS = {
    "qwen2_audio": "decoder",
    "kimi_audio": "shared_decoder",
    "qwen25_omni": "thinker",
}

OUTCOMES = ("00", "01", "10", "11")
KEY_LAYERS = (14, 16, 18, 20, 22, 24, 28, 32)
KEYS = ["fold", "layer", "topic", "language", "condition"]

TESTS = {
    "score": "Audio benign harmfulness score > text",
    "separation": "Audio harmful-benign separation > text",
}

FIELDS = [
    f"matched_{outcome}_{test}_{metric}"
    for outcome in OUTCOMES
    for test in TESTS
    for metric in (
        "audio_gt_text_n",
        "audio_gt_text_pct",
        "text_gt_audio_n",
        "equal_n",
        "positive_folds",
        "valid_folds",
    )
]


def load(base, component, modality, fold):
    path = (
        base / modality / component
        / f"fold_{fold}_predictions.csv"
    )

    df = pd.read_csv(
        path,
        usecols=KEYS + [
            "method",
            "benign_score",
            "harmful_score",
            "benign_decision",
        ],
    )

    df = df.loc[df["method"] == "mean"].drop(
        columns="method"
    ).copy()

    if df.empty or df.duplicated(KEYS).any():
        raise ValueError(f"Empty data or duplicate keys: {path}")

    if df["benign_decision"].isna().any():
        raise ValueError(f"Missing decisions: {path}")

    if not df["benign_decision"].isin([0, 1]).all():
        raise ValueError(f"Invalid decisions: {path}")

    if not np.isfinite(
        df[["benign_score", "harmful_score"]].to_numpy(float)
    ).all():
        raise ValueError(f"Invalid scores: {path}")

    sizes = df.groupby("layer").size()

    if not sizes.eq(320).all():
        raise ValueError(f"Incomplete held-out layers: {path}")

    mean = df.groupby("layer")["benign_score"].transform("mean")
    sd = df.groupby("layer")["benign_score"].transform("std")

    if sd.isna().any() or (sd <= 0).any():
        raise ValueError(f"Invalid normalization: {path}")

    df["benign_z"] = (df["benign_score"] - mean) / sd

    df["separation_z"] = (
        df["harmful_score"] - df["benign_score"]
    ) / sd

    return df[
        KEYS + [
            "benign_decision",
            "benign_z",
            "separation_z",
        ]
    ]


def build_data(model, component):
    base = RESULTS / model / "wk3/paired_harm_auroc"
    frames = []

    for fold in range(1, 6):
        text = load(base, component, "text", fold)
        audio = load(base, component, "audio", fold)

        df = text.merge(
            audio,
            on=KEYS,
            how="outer",
            validate="one_to_one",
            suffixes=("_text", "_audio"),
            indicator=True,
        )

        if not df["_merge"].eq("both").all():
            raise ValueError(
                f"{model}: unmatched prompts in fold {fold}"
            )

        df = df.drop(columns="_merge")

        df["outcome"] = (
            df["benign_decision_text"].astype(int).astype(str)
            + df["benign_decision_audio"].astype(int).astype(str)
        )

        df["score_diff"] = (
            df["benign_z_audio"] - df["benign_z_text"]
        )

        df["separation_diff"] = (
            df["separation_z_audio"]
            - df["separation_z_text"]
        )

        frames.append(df)

        print(
            f"{model}: matched and normalized fold {fold}/5",
            flush=True,
        )

    result = pd.concat(frames, ignore_index=True)

    if result.duplicated(
        ["layer", "topic", "language", "condition"]
    ).any():
        raise ValueError("Duplicate test examples across folds")

    return result


def calculate(df):
    output = {}

    for group_type in (
        "all",
        "language",
        "condition",
        "language_condition",
    ):
        if group_type == "all":
            grouped = df.assign(summary_group="all")
        elif group_type == "language":
            grouped = df.assign(
                summary_group=df["language"]
            )
        elif group_type == "condition":
            grouped = df.assign(
                summary_group=df["condition"]
            )
        else:
            grouped = df.assign(
                summary_group=(
                    df["language"] + "__" + df["condition"]
                )
            )

        for (layer, group), subset in grouped.groupby(
            ["layer", "summary_group"],
            sort=True,
        ):
            record = {}

            for outcome in OUTCOMES:
                selected = subset.loc[
                    subset["outcome"] == outcome
                ]

                for test, column in (
                    ("score", "score_diff"),
                    ("separation", "separation_diff"),
                ):
                    prefix = f"matched_{outcome}_{test}_"
                    values = selected[column].to_numpy(float)

                    audio_wins = int(np.sum(values > 0))
                    text_wins = int(np.sum(values < 0))
                    equal = int(np.sum(values == 0))

                    n = len(selected)

                    if audio_wins + text_wins + equal != n:
                        raise ValueError("Unclassified examples")

                    record[prefix + "audio_gt_text_n"] = audio_wins
                    record[prefix + "text_gt_audio_n"] = text_wins
                    record[prefix + "equal_n"] = equal

                    record[prefix + "audio_gt_text_pct"] = (
                        100.0 * audio_wins / n
                        if n else np.nan
                    )

                    fold_rates = (
                        selected.assign(
                            audio_wins=selected[column] > 0
                        )
                        .groupby("fold")["audio_wins"]
                        .mean()
                    )

                    record[prefix + "valid_folds"] = len(
                        fold_rates
                    )
                    record[prefix + "positive_folds"] = int(
                        (fold_rates > 0.5).sum()
                    )

            output[(int(layer), group_type, group)] = record

    return output


def update_summary(model, component, calculations):
    path = (
        RESULTS / model
        / "wk3/paired_harm_auroc/behaviour_summary.csv"
    )

    df = pd.read_csv(path)

    for field in FIELDS:
        df[field] = np.nan

    selected = (
        (df["component"] == component)
        & (df["method"] == "mean")
        & (df["split"] == "benign")
    )

    count = 0

    for idx, row in df.loc[selected].iterrows():
        key = (
            int(row["layer"]),
            row["group_type"],
            row["group"],
        )

        if key not in calculations:
            raise ValueError(f"Missing summary group: {key}")

        values = calculations[key]

        for name, value in values.items():
            df.at[idx, name] = value

        count += 1

    if count == 0:
        raise ValueError(f"No rows updated: {model}")

    temporary = path.with_name(path.name + ".tmp")

    try:
        df.to_csv(
            temporary,
            index=False,
            float_format="%.6f",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"{model}: updated {count} rows in {path}",
        flush=True,
    )

    return df


def result_table(df, component, layer, group_type="all"):
    selected = df.loc[
        (df["component"] == component)
        & (df["method"] == "mean")
        & (df["split"] == "benign")
        & (df["modality"] == "audio")
        & (df["layer"] == layer)
        & (df["group_type"] == group_type)
    ]

    rows = []

    for _, item in selected.iterrows():
        for outcome in OUTCOMES:
            a = f"matched_{outcome}_score_"
            b = f"matched_{outcome}_separation_"

            rows.append({
                "group": item["group"],
                "outcome": outcome,
                "N": int(item[f"matched_{outcome}_n"]),
                "benign_audio_gt_text_%": item[
                    a + "audio_gt_text_pct"
                ],
                "benign_audio_wins": item[
                    a + "audio_gt_text_n"
                ],
                "separation_audio_gt_text_%": item[
                    b + "audio_gt_text_pct"
                ],
                "separation_audio_wins": item[
                    b + "audio_gt_text_n"
                ],
                "benign_positive_folds": item[
                    a + "positive_folds"
                ],
                "separation_positive_folds": item[
                    b + "positive_folds"
                ],
            })

    return pd.DataFrame(rows)


def print_table(title, table):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    print(
        table.to_string(
            index=False,
            na_rep="NA",
            float_format=lambda x: f"{x:.2f}",
        )
    )


def main():
    for model, component in MODELS.items():
        print("\n" + "#" * 110)
        print(f"MODEL: {model.upper()}")
        print("#" * 110)

        matched = build_data(model, component)
        calculated = calculate(matched)
        summary = update_summary(
            model, component, calculated
        )

        layers = sorted(matched["layer"].unique())

        main_layer = (
            20 if 20 in layers else layers[len(layers) // 2]
        )

        print_table(
            f"{model}: COMPLETE COUNTING RESULT — LAYER {main_layer}",
            result_table(summary, component, main_layer),
        )

        all_layers = []

        for layer in KEY_LAYERS:
            if layer not in layers:
                continue

            table = result_table(
                summary, component, layer
            ).copy()
            table.insert(0, "layer", layer)
            all_layers.append(table)

        if all_layers:
            print_table(
                f"{model}: IMPORTANT DECODER LAYERS",
                pd.concat(all_layers, ignore_index=True),
            )

        for group_type in ("language", "condition"):
            print_table(
                f"{model}: {group_type.upper()} — LAYER {main_layer}",
                result_table(
                    summary,
                    component,
                    main_layer,
                    group_type,
                ),
            )

    print("\n" + "=" * 110)
    print("COMPLETE — ALL THREE MODELS")
    print("=" * 110)
    print("00: both accept; 01: audio-only refusal;")
    print("10: text-only refusal; 11: both refuse.")
    print("benign_audio_gt_text_%: fraction with higher audio benign z-score.")
    print("separation_audio_gt_text_%: fraction with larger audio")
    print("harmful-minus-benign separation, scaled by benign SD.")
    print("positive_folds: folds where >50% of examples favour audio.")
    print("Both comparisons use fold-specific normalization.")
    print("Only the existing behaviour_summary.csv files were updated.")


if __name__ == "__main__":
    main()
