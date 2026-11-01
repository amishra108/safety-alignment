#!/usr/bin/env python3

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

MODELS = {
    "kimi_audio": {
        "decoder": "shared_decoder",
        "components": (
            ("text", "shared_decoder"),
            ("text", "text_mimo"),
            ("audio", "shared_decoder"),
            ("audio", "text_mimo"),
            ("audio", "audio_encoder"),
        ),
    },
    "qwen25_omni": {
        "decoder": "thinker",
        "components": (
            ("text", "thinker"),
            ("audio", "thinker"),
            ("audio", "audio_encoder"),
        ),
    },
}

OUTCOMES = ("00", "01", "10", "11")
METHODS = (
    "mean", "svd_1", "svd_2", "svd_4", "svd_8",
    "svd_16", "svd_32", "svd_64", "svd_128",
)

MATCHED_FIELDS = [
    f"matched_{outcome}_{field}"
    for outcome in OUTCOMES
    for field in (
        "n", "pct", "valid_folds",
        "text_z", "audio_z", "audio_minus_text_z",
        "audio_minus_text_z_std",
    )
]

FIELDS = [
    "model", "modality", "component", "layer", "method",
    "group_type", "group", "split",
    "n", "decision_0", "decision_1",
    "decision_0_pct", "decision_1_pct",
    "mean_score_decision_0", "mean_score_decision_1",
    "valid_folds", "positive_folds", "negative_folds",
    "mean_gap", "std_gap",
    "fold_1", "fold_2", "fold_3", "fold_4", "fold_5",
    "cohens_d", "cohens_d_std", "cohens_d_valid_folds",
] + MATCHED_FIELDS

PREDICTION_FIELDS = [
    "model", "modality", "component", "fold", "layer",
    "method", "topic", "language", "condition",
    "benign_score", "harmful_score", "paired_correct",
]

KEY_FIELDS = ("fold", "layer", "topic", "language", "condition")

BASE_PATTERN = re.compile(r"^jbb_(benign|harmful)_(\d+)$")


def groups(language, condition):
    return (
        ("all", "all"),
        ("language", language),
        ("condition", condition),
        ("language_condition", f"{language}__{condition}"),
    )


def update_moments(state, x):
    state[0] += 1
    delta = x - state[1]
    state[1] += delta / state[0]
    state[2] += delta * (x - state[1])


def safe_float(value):
    return "" if value is None or not np.isfinite(value) else f"{value:.6f}"


def load_manifest(model):
    folder = RESULTS / model / "wk3/dataset_summary/folds"
    decisions = {}
    assignments = defaultdict(set)

    for fold in range(1, 6):
        path = folder / f"fold_{fold}_test.csv"

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            required = {
                "base_id", "split", "modality", "language",
                "condition", "decision", "test_fold", "partition",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected manifest schema: {path}")

            for row in reader:
                match = BASE_PATTERN.fullmatch(row["base_id"])
                if match is None or match.group(1) != row["split"]:
                    raise ValueError(f"Invalid base_id in {path}: {row['base_id']}")

                topic = int(match.group(2))
                decision = int(row["decision"])

                if not 0 <= topic < 100 or decision not in (0, 1):
                    raise ValueError(f"Invalid topic or decision: {path}")

                if int(row["test_fold"]) != fold:
                    raise ValueError(f"Fold mismatch: {path}")

                key = (
                    fold, row["split"], row["modality"], topic,
                    row["language"], row["condition"],
                )

                if key in decisions:
                    raise ValueError(f"Duplicate manifest key: {key}")

                decisions[key] = decision
                assignments[(fold, row["modality"], row["split"])].add(
                    (topic, row["language"], row["condition"])
                )

    for fold in range(1, 6):
        for modality in ("text", "audio"):
            for split in ("benign", "harmful"):
                n = len(assignments[(fold, modality, split)])
                if n != 320:
                    raise ValueError(
                        f"{model} fold={fold} {modality}/{split}: "
                        f"expected 320 examples, found {n}"
                    )

    for fold in range(1, 6):
        for modality in ("text", "audio"):
            a = assignments[(fold, modality, "benign")]
            b = assignments[(fold, modality, "harmful")]
            if a != b:
                raise ValueError("Harmful/benign fold pairing mismatch")

        if assignments[(fold, "text", "benign")] != assignments[
            (fold, "audio", "benign")
        ]:
            raise ValueError("Text/audio fold pairing mismatch")

    if len(decisions) != 6400:
        raise ValueError(f"Unexpected manifest size: {len(decisions)}")

    print(f"Validated {len(decisions)} adjudicated manifest rows", flush=True)
    return decisions


def prediction_paths(base, components):
    for modality, component in components:
        directory = base / "paired_harm_auroc" / modality / component
        for fold in range(1, 6):
            yield modality, component, directory / f"fold_{fold}_predictions.csv"
        yield modality, component, directory / "heldout_predictions.csv"


def validate_prediction_files(base, components):
    for modality, component, path in prediction_paths(base, components):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []

            if not set(PREDICTION_FIELDS).issubset(fields):
                raise ValueError(f"Unexpected prediction schema: {path}")

            existing = set(fields) - set(PREDICTION_FIELDS)
            if not existing.issubset({
                "benign_decision", "harmful_decision"
            }):
                raise ValueError(f"Unexpected extra prediction columns: {path}")


def process_predictions(model, config, base, decisions):
    stats = defaultdict(lambda: [0, 0.0, 0.0])
    benign_matched = defaultdict(dict)
    coverage = defaultdict(set)

    main_component = config["decoder"]

    for modality, component, path in prediction_paths(
        base, config["components"]
    ):
        is_heldout = path.name == "heldout_predictions.csv"
        temporary = path.with_name(path.name + ".tmp")

        count = 0

        try:
            with (
                path.open(newline="", encoding="utf-8") as source,
                temporary.open("w", newline="", encoding="utf-8") as destination
            ):
                reader = csv.DictReader(source)
                fields = list(reader.fieldnames or [])

                for field in ("benign_decision", "harmful_decision"):
                    if field not in fields:
                        fields.append(field)

                writer = csv.DictWriter(
                    destination, fieldnames=fields
                )
                writer.writeheader()

                for row in reader:
                    fold = int(row["fold"])
                    topic = int(row["topic"])
                    layer = int(row["layer"])
                    language = row["language"]
                    condition = row["condition"]
                    method = row["method"]

                    if fold not in range(1, 6):
                        raise ValueError(f"Invalid fold: {path}")
                    if method not in METHODS:
                        raise ValueError(f"Unexpected direction method: {method}")
                    if row["model"] != model:
                        raise ValueError(f"Model mismatch: {path}")
                    if row["component"] != component:
                        raise ValueError(f"Component mismatch: {path}")
                    if row["modality"] != modality:
                        raise ValueError(f"Modality mismatch: {path}")
                    if not is_heldout and path.name != (
                        f"fold_{fold}_predictions.csv"
                    ):
                        raise ValueError(f"Fold/file mismatch: {path}")

                    for split in ("benign", "harmful"):
                        key = (
                            fold, split, modality, topic,
                            language, condition,
                        )
                        if key not in decisions:
                            raise ValueError(
                                f"Missing adjudicated label: {key}"
                            )

                        decision = decisions[key]
                        field = f"{split}_decision"

                        if row.get(field) not in (
                            None, "", str(decision)
                        ):
                            raise ValueError(
                                f"Conflicting existing decision: {path}, {key}"
                            )

                        row[field] = str(decision)

                        score = float(row[f"{split}_score"])
                        if not np.isfinite(score):
                            raise ValueError(f"Nonfinite score: {path}")

                        if not is_heldout:
                            for group_type, group in groups(
                                language, condition
                            ):
                                stat_key = (
                                    modality, component, layer, method,
                                    group_type, group, split, fold, decision,
                                )
                                update_moments(stats[stat_key], score)

                        if (
                            not is_heldout
                            and split == "benign"
                            and component == main_component
                            and method == "mean"
                        ):
                            pair_key = (
                                fold, layer, topic, language, condition
                            )

                            if pair_key in benign_matched[modality]:
                                raise ValueError(
                                    f"Duplicate matched prompt: {pair_key}"
                                )

                            benign_matched[modality][pair_key] = (
                                score, decision
                            )

                    if not is_heldout and method == "mean":
                        coverage[(modality, component, fold, layer)].add(
                            (topic, language, condition)
                        )

                    writer.writerow(row)
                    count += 1

            if count == 0:
                raise ValueError(f"Empty predictions file: {path}")

            os.replace(temporary, path)

        finally:
            if temporary.exists():
                temporary.unlink()

        print(
            f"  {modality}/{component}/{path.name}: "
            f"{count:,} rows",
            flush=True,
        )

    for key, examples in coverage.items():
        if len(examples) != 320:
            raise ValueError(
                f"Incomplete mean-direction coverage: {key}, "
                f"N={len(examples)}"
            )

    return stats, benign_matched


def describe_stats(stats):
    overall = defaultdict(dict)

    for key, state in stats.items():
        modality, component, layer, method, gt, group, split, fold, decision = key
        prefix = (modality, component, layer, method, gt, group, split)
        overall[prefix].setdefault(fold, {})[decision] = state

    rows = []
    fold_effects = {}

    for key, by_fold in sorted(overall.items()):
        modality, component, layer, method, gt, group, split = key

        fold_gaps = {}
        fold_ds = {}
        total_count = [0, 0]
        weighted_sums = [0.0, 0.0]

        for fold, outcomes in by_fold.items():
            for decision in (0, 1):
                state = outcomes.get(decision)
                if state is not None:
                    n, mean, _ = state
                    total_count[decision] += n
                    weighted_sums[decision] += n * mean

            if 0 not in outcomes or 1 not in outcomes:
                continue

            n0, mean0, m2_0 = outcomes[0]
            n1, mean1, m2_1 = outcomes[1]

            gap = (
                mean0 - mean1 if split == "harmful"
                else mean1 - mean0
            )

            fold_gaps[fold] = gap

            if n0 >= 2 and n1 >= 2:
                variance = (m2_0 + m2_1) / (n0 + n1 - 2)

                if variance > 0:
                    fold_ds[fold] = gap / np.sqrt(variance)

        count = sum(total_count)
        gap_values = np.array(list(fold_gaps.values()))
        d_values = np.array(list(fold_ds.values()))

        record = {
            "model": "",
            "modality": modality,
            "component": component,
            "layer": layer,
            "method": method,
            "group_type": gt,
            "group": group,
            "split": split,
            "n": count,
            "decision_0": total_count[0],
            "decision_1": total_count[1],
            "decision_0_pct": (
                100 * total_count[0] / count if count else np.nan
            ),
            "decision_1_pct": (
                100 * total_count[1] / count if count else np.nan
            ),
            "mean_score_decision_0": (
                weighted_sums[0] / total_count[0]
                if total_count[0] else np.nan
            ),
            "mean_score_decision_1": (
                weighted_sums[1] / total_count[1]
                if total_count[1] else np.nan
            ),
            "valid_folds": len(gap_values),
            "positive_folds": int(np.sum(gap_values > 0)),
            "negative_folds": int(np.sum(gap_values < 0)),
            "mean_gap": (
                gap_values.mean() if len(gap_values) else np.nan
            ),
            "std_gap": (
                gap_values.std(ddof=1)
                if len(gap_values) >= 2 else np.nan
            ),
            "cohens_d": (
                d_values.mean() if len(d_values) else np.nan
            ),
            "cohens_d_std": (
                d_values.std(ddof=1)
                if len(d_values) >= 2 else np.nan
            ),
            "cohens_d_valid_folds": len(d_values),
        }

        for fold in range(1, 6):
            record[f"fold_{fold}"] = fold_gaps.get(fold, np.nan)

        rows.append(record)
        fold_effects[key] = fold_ds

    return rows, fold_effects


def matched_analysis(benign_matched):
    text = benign_matched["text"]
    audio = benign_matched["audio"]

    if set(text) != set(audio):
        raise ValueError(
            "Main decoder: text/audio held-out prompt keys differ"
        )

    data = []

    for key in sorted(text):
        fold, layer, topic, language, condition = key
        text_score, text_decision = text[key]
        audio_score, audio_decision = audio[key]

        data.append({
            "fold": fold,
            "layer": layer,
            "topic": topic,
            "language": language,
            "condition": condition,
            "text_score": text_score,
            "audio_score": audio_score,
            "outcome": f"{text_decision}{audio_decision}",
        })

    df = pd.DataFrame(data)

    for modality in ("text", "audio"):
        score_col = f"{modality}_score"

        means = df.groupby(["fold", "layer"])[score_col].transform(
            "mean"
        )
        stds = df.groupby(["fold", "layer"])[score_col].transform(
            "std"
        )

        if (stds <= 0).any() or stds.isna().any():
            raise ValueError("Cannot standardize matched scores")

        df[f"{modality}_z"] = (
            df[score_col] - means
        ) / stds

    df["delta_z"] = df["audio_z"] - df["text_z"]
    df["summary_group"] = "all"

    summaries = {}

    for group_type in (
        "all", "language", "condition", "language_condition"
    ):
        if group_type == "language":
            df["summary_group"] = df["language"]
        elif group_type == "condition":
            df["summary_group"] = df["condition"]
        elif group_type == "language_condition":
            df["summary_group"] = (
                df["language"] + "__" + df["condition"]
            )
        else:
            df["summary_group"] = "all"

        for (layer, group), subset in df.groupby(
            ["layer", "summary_group"]
        ):
            result = {}
            total = len(subset)

            for outcome in OUTCOMES:
                selected = subset.loc[
                    subset["outcome"] == outcome
                ]
                prefix = f"matched_{outcome}_"

                result[prefix + "n"] = len(selected)
                result[prefix + "pct"] = (
                    100 * len(selected) / total
                )

                means = selected.groupby("fold")[
                    ["text_z", "audio_z", "delta_z"]
                ].mean()

                result[prefix + "valid_folds"] = len(means)

                for source, target in (
                    ("text_z", "text_z"),
                    ("audio_z", "audio_z"),
                    ("delta_z", "audio_minus_text_z"),
                ):
                    result[prefix + target] = (
                        means[source].mean()
                        if len(means) else np.nan
                    )

                result[prefix + "audio_minus_text_z_std"] = (
                    means["delta_z"].std(ddof=1)
                    if len(means) >= 2 else np.nan
                )

            summaries[(int(layer), group_type, group)] = result

    return summaries


def write_summary(model, base, config, rows, matched):
    summary = base / "paired_harm_auroc/behaviour_summary.csv"
    temporary = summary.with_name(summary.name + ".tmp")

    for row in rows:
        row["model"] = model

        if (
            row["component"] == config["decoder"]
            and row["method"] == "mean"
            and row["split"] == "benign"
        ):
            key = (
                row["layer"], row["group_type"], row["group"]
            )

            if key not in matched:
                raise ValueError(f"Missing matched results: {key}")

            row.update(matched[key])

    try:
        with temporary.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()

            for row in rows:
                formatted = {
                    field: (
                        safe_float(row[field])
                        if isinstance(row.get(field), (float, np.floating))
                        else row.get(field, "")
                    )
                    for field in FIELDS
                }
                writer.writerow(formatted)

        os.replace(temporary, summary)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"\nSaved {len(rows):,} summary rows: {summary}",
        flush=True,
    )
    return pd.read_csv(summary)


def print_results(model, config, summary, fold_effects):
    component = config["decoder"]

    main = summary.loc[
        (summary["component"] == component)
        & (summary["method"] == "mean")
        & (summary["group_type"] == "all")
        & (summary["group"] == "all")
    ].copy()

    layers = sorted(
        set(main.loc[main["modality"] == "text", "layer"])
        & set(main.loc[main["modality"] == "audio", "layer"])
    )

    if not layers:
        raise ValueError("No common text/audio decoder layers")

    print("\n" + "=" * 112)
    print(f"{model.upper()}: MAIN DECODER BEHAVIOUR")
    print("=" * 112)

    print(
        main[
            ["layer", "modality", "split", "decision_1_pct",
             "mean_gap", "std_gap",
             "cohens_d", "cohens_d_std",
             "positive_folds", "valid_folds"]
        ].sort_values(
            ["layer", "modality", "split"]
        ).to_string(
            index=False,
            na_rep="NA",
            float_format=lambda x: f"{x:+.3f}",
        )
    )

    paired_rows = []

    for layer in layers:
        for split in ("harmful", "benign"):
            common = []

            for fold in range(1, 6):
                def effect(modality):
                    key = (
                        modality, component, layer, "mean",
                        "all", "all", split,
                    )
                    return fold_effects[key].get(fold)

                text_d = effect("text")
                audio_d = effect("audio")

                if text_d is not None and audio_d is not None:
                    common.append(text_d - audio_d)

            if not common:
                continue

            differences = np.array(common)

            paired_rows.append({
                "layer": layer,
                "split": split,
                "text_minus_audio_d": differences.mean(),
                "fold_SD": (
                    differences.std(ddof=1)
                    if len(differences) > 1 else np.nan
                ),
                "text_wins": int((differences > 0).sum()),
                "audio_wins": int((differences < 0).sum()),
                "valid_folds": len(differences),
            })

    paired = pd.DataFrame(paired_rows)

    print("\n" + "=" * 112)
    print("PAIRED FOLD-WISE COHEN'S d: TEXT MINUS AUDIO")
    print("=" * 112)

    print(
        paired.to_string(
            index=False,
            na_rep="NA",
            float_format=lambda x: f"{x:+.3f}",
        )
    )

    chosen_layer = 20 if 20 in layers else layers[len(layers) // 2]

    matched = main.loc[
        (main["layer"] == chosen_layer)
        & (main["modality"] == "audio")
        & (main["split"] == "benign")
    ]

    if len(matched) != 1:
        raise ValueError("Missing overall benign matched summary")

    r = matched.iloc[0]

    matched_rows = []

    for outcome in OUTCOMES:
        p = f"matched_{outcome}_"

        matched_rows.append({
            "outcome": outcome,
            "N": int(r[p + "n"]),
            "pct": r[p + "pct"],
            "valid_folds": int(r[p + "valid_folds"]),
            "text_z": r[p + "text_z"],
            "audio_z": r[p + "audio_z"],
            "audio_minus_text_z": r[p + "audio_minus_text_z"],
            "delta_fold_SD": r[p + "audio_minus_text_z_std"],
        })

    print("\n" + "=" * 112)
    print(f"MATCHED BENIGN OUTCOMES — DECODER LAYER {chosen_layer}")
    print("00 both accept | 01 audio only refuses | 10 text only refuses | 11 both refuse")
    print("=" * 112)

    print(
        pd.DataFrame(matched_rows).to_string(
            index=False,
            na_rep="NA",
            float_format=lambda x: f"{x:+.3f}",
        )
    )

    breakdown = summary.loc[
        (summary["component"] == component)
        & (summary["method"] == "mean")
        & (summary["layer"] == chosen_layer)
        & summary["group_type"].isin(("language", "condition"))
    ].copy()

    print("\n" + "=" * 112)
    print(f"LANGUAGES AND ATTACK CONDITIONS — DECODER LAYER {chosen_layer}")
    print("=" * 112)

    print(
        breakdown[
            ["group_type", "group", "modality", "split",
             "decision_1_pct", "cohens_d",
             "cohens_d_std", "cohens_d_valid_folds"]
        ].sort_values(
            ["group_type", "group", "modality", "split"]
        ).to_string(
            index=False,
            na_rep="NA",
            float_format=lambda x: f"{x:+.3f}",
        )
    )

    print("\n" + "=" * 112)
    print("MATCHED BENIGN OUTCOMES BY LANGUAGE AND CONDITION")
    print("=" * 112)

    matched_subgroups = breakdown.loc[
        (breakdown["modality"] == "audio")
        & (breakdown["split"] == "benign")
    ].copy()

    display_rows = []

    for _, item in matched_subgroups.iterrows():
        for outcome in OUTCOMES:
            p = f"matched_{outcome}_"

            display_rows.append({
                "group_type": item["group_type"],
                "group": item["group"],
                "outcome": outcome,
                "N": item[p + "n"],
                "text_z": item[p + "text_z"],
                "audio_z": item[p + "audio_z"],
                "audio_minus_text_z": item[
                    p + "audio_minus_text_z"
                ],
            })

    print(
        pd.DataFrame(display_rows).to_string(
            index=False,
            na_rep="NA",
            float_format=lambda x: f"{x:+.3f}",
        )
    )

    print("\nINTERPRETATION")
    print("Harmful decision=1: successful harmful compliance (ASR).")
    print("Benign decision=1: incorrect refusal (BRR).")
    print("Positive harmful gap/d: noncompliance scores higher.")
    print("Positive benign gap/d: refusal scores higher.")
    print("Positive text-minus-audio d: stronger separation in text.")
    print("Positive audio-minus-text z: higher relative audio score.")
    print("Matched z-scores use all benign examples per fold/layer/modality.")
    print("Group means are calculated within each outcome before averaging folds.")
    print("These are descriptive associations, not causal effects.")


def main():
    for model, config in MODELS.items():
        print("\n" + "#" * 112, flush=True)
        print(f"PROCESSING {model}", flush=True)
        print("#" * 112, flush=True)

        base = RESULTS / model / "wk3"

        decisions = load_manifest(model)
        validate_prediction_files(base, config["components"])

        stats, benign_matched = process_predictions(
            model, config, base, decisions
        )

        rows, fold_effects = describe_stats(stats)
        matched = matched_analysis(benign_matched)

        summary = write_summary(
            model, base, config, rows, matched
        )

        print_results(model, config, summary, fold_effects)

        print(
            f"\nCOMPLETED {model.upper()}",
            flush=True,
        )

    print("\nBOTH MODELS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
