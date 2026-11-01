#!/usr/bin/env python3

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

COMPONENTS = {
    "qwen2_audio": "decoder",
    "qwen25_omni": "thinker",
    "kimi_audio": "shared_decoder",
}

SOURCES = ("text", "audio", "combined")
K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)

EXAMPLE_PATTERN = re.compile(
    r"jbb_(benign|harmful)_(\d{3})__"
    r"(original|literal|narrative_framing|content_dilution)__"
    r"(english|chinese|spanish|russian)"
)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalise_fold(value):
    match = re.fullmatch(r"(?:fold_)?([1-5])(?:\.0)?", str(value))
    if match is None:
        raise ValueError(f"Invalid test_fold: {value}")
    return int(match.group(1))


def build_pairs(assignments):
    required = {"task_id", "modality", "test_fold"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing assignment columns: {sorted(missing)}")

    indexed = defaultdict(dict)

    for row in assignments.itertuples(index=False):
        task_id = str(row.task_id)
        modality = str(row.modality).lower()
        match = EXAMPLE_PATTERN.search(task_id)

        if match is None:
            raise ValueError(f"Unrecognised task_id: {task_id}")

        if modality not in ("text", "audio"):
            raise ValueError(f"Invalid modality: {modality}")

        label, topic, condition, language = match.groups()
        key = (topic, condition, language, modality)
        fold = normalise_fold(row.test_fold)

        if label in indexed[key]:
            raise ValueError(f"Duplicate pair member: {key}, {label}")

        indexed[key][label] = (task_id, fold)

    pairs = {"text": [], "audio": []}

    for key, entries in sorted(indexed.items()):
        if set(entries) != {"benign", "harmful"}:
            raise ValueError(f"Incomplete matched pair: {key}")

        benign_id, benign_fold = entries["benign"]
        harmful_id, harmful_fold = entries["harmful"]

        if benign_fold != harmful_fold:
            raise ValueError(f"Pair crosses folds: {key}")

        pairs[key[3]].append(
            (benign_id, harmful_id, benign_fold)
        )

    for modality in ("text", "audio"):
        if len(pairs[modality]) != 1600:
            raise ValueError(
                f"Expected 1600 {modality} pairs; "
                f"found {len(pairs[modality])}"
            )

    return pairs


def load_difference_matrix(model, component, pairs):
    representation_dir = (
        RESULTS / model / "5fold/representations_shared"
    )

    differences = []
    folds = []

    for index, (benign_id, harmful_id, fold) in enumerate(pairs, 1):
        vectors = []

        for task_id in (benign_id, harmful_id):
            path = representation_dir / f"{task_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)

            record = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            if component not in record:
                raise KeyError(f"{path}: missing {component}")

            tensor = record[component]

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{path}: {component} is not a tensor")

            if tensor.ndim != 2:
                raise ValueError(
                    f"{path}: expected [boundaries, hidden], "
                    f"found {tuple(tensor.shape)}"
                )

            vectors.append(
                tensor.detach().to(
                    device="cpu", dtype=torch.float32
                ).numpy()
            )

        if vectors[0].shape != vectors[1].shape:
            raise ValueError(
                f"Representation shape mismatch: "
                f"{benign_id} versus {harmful_id}"
            )

        differences.append(vectors[1] - vectors[0])
        folds.append(fold)

        if index % 400 == 0:
            print(
                f"    loaded {index}/{len(pairs)} pairs",
                flush=True,
            )

    matrix = np.stack(differences, axis=0)
    fold_ids = np.asarray(folds, dtype=np.int8)

    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite representation differences")

    return matrix, fold_ids


def unit_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Direction has zero or non-finite norm")

    return vector / norm


def load_saved_directions(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as saved:
        needed = {
            "means", "vectors", "layer_boundaries"
        }
        missing = needed - set(saved.files)

        if missing:
            raise KeyError(
                f"{path}: missing arrays {sorted(missing)}"
            )

        means = np.asarray(saved["means"], dtype=np.float64)
        vectors = np.asarray(saved["vectors"], dtype=np.float64)
        boundaries = np.asarray(
            saved["layer_boundaries"], dtype=int
        )

    if means.ndim != 2 or vectors.ndim != 3:
        raise ValueError(f"{path}: unexpected saved array dimensions")

    if not (
        len(boundaries) == means.shape[0] == vectors.shape[0]
    ):
        raise ValueError(f"{path}: inconsistent layer dimensions")

    return means, vectors, boundaries


def calculate_positive_percentages(
    matrix, fold_ids, means, vectors, boundaries, fold
):
    train = matrix[fold_ids != fold]
    test = matrix[fold_ids == fold]

    expected_train = 1280 if len(matrix) == 1600 else 2560
    expected_test = 320 if len(matrix) == 1600 else 640

    if len(train) != expected_train or len(test) != expected_test:
        raise ValueError(
            f"Fold {fold}: expected {expected_train}/{expected_test} "
            f"train/test pairs, got {len(train)}/{len(test)}"
        )

    records = []

    for j, boundary in enumerate(boundaries):
        if boundary < 0 or boundary >= matrix.shape[1]:
            raise ValueError(f"Invalid saved boundary: {boundary}")

        mean = means[j]
        basis = vectors[j]

        if basis.shape[-1] != matrix.shape[-1]:
            raise ValueError(
                f"Direction dimension mismatch at layer {boundary}"
            )

        if basis.shape[0] < max(K_VALUES):
            raise ValueError(
                f"Insufficient singular vectors at layer {boundary}"
            )

        train_d = train[:, boundary, :].astype(
            np.float64, copy=False
        )
        test_d = test[:, boundary, :].astype(
            np.float64, copy=False
        )

        methods = [("mean", 0, unit_vector(mean))]

        for k in K_VALUES:
            top = basis[:k]
            projected_mean = (top @ mean) @ top
            direction = unit_vector(projected_mean)
            methods.append((f"svd_{k}", k, direction))

        for method, k, direction in methods:
            train_projection = train_d @ direction
            test_projection = test_d @ direction

            records.append({
                "layer": int(boundary),
                "method": method,
                "k": k,
                "train_positive_projection_pct": (
                    100.0 * np.mean(train_projection > 0)
                ),
                "test_positive_projection_pct": (
                    100.0 * np.mean(test_projection > 0)
                ),
            })

    return pd.DataFrame.from_records(records)


def update_fold_csv(path, new_metrics):
    old = read_csv(path)

    for column in (
        "train_positive_projection_pct",
        "test_positive_projection_pct",
    ):
        old = old.drop(columns=column, errors="ignore")

    keys = ["layer", "method", "k"]

    merged = old.merge(
        new_metrics,
        how="left",
        on=keys,
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        missing = merged.loc[
            merged["_merge"] != "both", keys
        ]
        raise ValueError(
            f"{path}: unmatched rows:\n{missing.to_string(index=False)}"
        )

    merged = merged.drop(columns="_merge")
    merged.to_csv(path, index=False)


def update_summary_csv(path, fold_metrics):
    old = read_csv(path)

    metrics = (
        "train_positive_projection_pct",
        "test_positive_projection_pct",
    )

    old = old.drop(
        columns=[
            f"{metric}_{suffix}"
            for metric in metrics
            for suffix in ("mean", "std")
        ],
        errors="ignore",
    )

    grouped = (
        pd.concat(fold_metrics, ignore_index=True)
        .groupby(["layer", "method", "k"], as_index=False)
        .agg({
            metric: ["mean", "std"]
            for metric in metrics
        })
    )

    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple) else column
        for column in grouped.columns
    ]

    merged = old.merge(
        grouped,
        on=["layer", "method", "k"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError(f"{path}: unmatched summary rows")

    merged.drop(columns="_merge").to_csv(path, index=False)


def print_summary(model, source, component, metrics):
    all_results = pd.concat(metrics, ignore_index=True)

    summary = (
        all_results
        .groupby(["layer", "method"], as_index=False)
        .agg(
            mean=("test_positive_projection_pct", "mean"),
            std=("test_positive_projection_pct", "std"),
            train_mean=("train_positive_projection_pct", "mean"),
        )
    )

    layers = sorted(summary["layer"].unique())
    chosen = sorted(set([
        layers[0],
        layers[len(layers) // 2],
        layers[-1],
    ]))

    print(
        f"\n{'=' * 75}\n"
        f"{model} | {source} | {component}\n"
        f"{'=' * 75}"
    )
    print(
        "Layer  Direction   Train positive %   "
        "Test positive % (5-fold mean ± SD)"
    )

    for layer in chosen:
        for method in ("mean", "svd_1", "svd_4", "svd_16", "svd_128"):
            selected = summary[
                (summary["layer"] == layer)
                & (summary["method"] == method)
            ]

            if len(selected) != 1:
                raise ValueError(
                    f"Missing summary: layer={layer}, method={method}"
                )

            row = selected.iloc[0]

            print(
                f"{layer:>5}  {method:<10}  "
                f"{row.train_mean:>8.2f}           "
                f"{row['mean']:>6.2f} ± {row['std']:.2f}"
            )

    print("\nBest held-out layer per direction:")

    for method in ("mean", "svd_1", "svd_4", "svd_16", "svd_128"):
        rows = summary[summary["method"] == method]
        best = rows.loc[rows["mean"].idxmax()]

        print(
            f"  {method:<8} layer {int(best.layer):>2}: "
            f"{best['mean']:.2f} ± {best['std']:.2f}%"
        )


def run_model(model):
    component = COMPONENTS[model]
    assignment_path = (
        RESULTS / model
        / "wk3/dataset_summary/fold_assignments.csv"
    )

    print(f"\nLoading assignments: {model}", flush=True)
    pairs = build_pairs(read_csv(assignment_path))

    matrices = {}
    fold_ids = {}

    for modality in ("text", "audio"):
        print(
            f"  Loading {modality} {component} representations",
            flush=True,
        )
        matrices[modality], fold_ids[modality] = (
            load_difference_matrix(
                model, component, pairs[modality]
            )
        )

    for source in SOURCES:
        if source == "combined":
            matrix = np.concatenate(
                (matrices["text"], matrices["audio"]),
                axis=0,
            )
            folds = np.concatenate(
                (fold_ids["text"], fold_ids["audio"])
            )
        else:
            matrix = matrices[source]
            folds = fold_ids[source]

        directory = (
            RESULTS / model / "wk3/paired_harm_svd"
            / source / component
        )

        per_fold_metrics = []

        for fold in range(1, 6):
            print(
                f"  {model} {source}: fold {fold}/5",
                flush=True,
            )

            means, vectors, boundaries = load_saved_directions(
                directory / f"fold_{fold}_directions.npz"
            )

            new_metrics = calculate_positive_percentages(
                matrix,
                folds,
                means,
                vectors,
                boundaries,
                fold,
            )

            update_fold_csv(
                directory / f"fold_{fold}_energy.csv",
                new_metrics,
            )

            per_fold_metrics.append(new_metrics)

        update_summary_csv(
            directory / "energy_five_fold_mean_std.csv",
            per_fold_metrics,
        )

        print_summary(
            model, source, component, per_fold_metrics
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        choices=tuple(COMPONENTS),
        default=list(COMPONENTS),
    )
    args = parser.parse_args()

    for model in args.models:
        run_model(model)


if __name__ == "__main__":
    main()
