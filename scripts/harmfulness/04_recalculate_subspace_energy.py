#!/usr/bin/env python3

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

COMPONENTS = {
    "qwen2_audio": ("decoder", 32),
    "qwen25_omni": ("thinker", 28),
    "kimi_audio": ("shared_decoder", 28),
}

DIMENSIONS = {
    "qwen2_audio": 4096,
    "qwen25_omni": 3584,
    "kimi_audio": 3584,
}

SOURCES = ("text", "audio", "combined")
LANGUAGES = ("chinese", "english", "russian", "spanish")
CONDITIONS = (
    "content_dilution",
    "literal",
    "narrative_framing",
    "original",
)

K_VALUES = (1, 4, 8, 16, 128)


def load_fold_indices(model):
    directory = RESULTS / model / "wk3/dataset_summary/folds"
    assignments = {}

    for fold in range(1, 6):
        path = directory / f"fold_{fold}_test.csv"

        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        topics = {
            int(row["base_id"].rsplit("_", 1)[1])
            for row in rows
        }

        if len(rows) != 1280 or len(topics) != 20:
            raise ValueError(f"Invalid test split: {path}")

        if any(topic in assignments for topic in topics):
            raise ValueError(f"Repeated test topics: fold {fold}")

        for topic in topics:
            assignments[topic] = fold

    if set(assignments) != set(range(100)):
        raise ValueError(f"Incomplete fold coverage: {model}")

    topics_per_example = np.repeat(np.arange(100), 16)
    folds_per_example = np.array(
        [assignments[int(topic)] for topic in topics_per_example]
    )

    result = {}

    for fold in range(1, 6):
        train = np.flatnonzero(folds_per_example != fold)
        test = np.flatnonzero(folds_per_example == fold)

        if len(train) != 1280 or len(test) != 320:
            raise ValueError(f"Invalid fold sizes: {model}, {fold}")

        result[fold] = (train, test)

    return result


def load_differences(model, modality, component, n_layers, dim):
    directory = RESULTS / model / "5fold/representations_shared"

    differences = np.empty(
        (1600, n_layers + 1, dim),
        dtype=np.float32,
    )

    index = 0

    for topic in range(100):
        for language in LANGUAGES:
            for condition in CONDITIONS:
                tensors = []

                for category in ("benign", "harmful"):
                    task_id = (
                        f"jbb_{category}_{topic:03d}__"
                        f"{condition}__{language}__{modality}"
                    )
                    path = directory / f"{task_id}.pt"

                    record = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=True,
                    )

                    if record["task_id"] != task_id:
                        raise ValueError(f"Task ID mismatch: {path}")

                    tensor = record[component]

                    if tuple(tensor.shape) != (n_layers + 1, dim):
                        raise ValueError(
                            f"Unexpected shape {tuple(tensor.shape)}: {path}"
                        )

                    tensors.append(tensor.to(dtype=torch.float32))

                differences[index] = (
                    tensors[1] - tensors[0]
                ).numpy()

                index += 1

        print(
            f"  Loaded {modality}: {topic + 1}/100 topics",
            flush=True,
        )

    if index != 1600 or not np.isfinite(differences).all():
        raise ValueError(f"Invalid representations: {model}/{modality}")

    return differences


def calculate_energy(matrix, basis, k):
    selected = basis[:k].T

    projections = matrix @ selected

    numerator = np.sum(
        np.square(projections, dtype=np.float64),
        dtype=np.float64,
    )

    denominator = np.sum(
        np.square(matrix, dtype=np.float64),
        dtype=np.float64,
    )

    if denominator <= 0:
        raise ValueError("Zero difference energy")

    return float(numerator / denominator)


def write_csv(path, dataframe):
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected existing CSV for replacement: {path}"
        )

    temporary = path.with_suffix(".csv.tmp")

    try:
        dataframe.to_csv(
            temporary,
            index=False,
            float_format="%.10f",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"  REPLACED {path.name}", flush=True)


def process_source(model, source, component, folds, differences):
    directory = (
        RESULTS / model / "wk3/paired_harm_svd"
        / source / component
    )

    summary_rows = []

    for fold in range(1, 6):
        path = directory / f"fold_{fold}_directions.npz"

        with np.load(path) as data:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            layers = np.asarray(data["layer_boundaries"], dtype=int)

        expected_layers = np.arange(1, COMPONENTS[model][1] + 1)

        if not np.array_equal(layers, expected_layers):
            raise ValueError(f"Unexpected layer boundaries: {path}")

        if vectors.shape != (
            len(layers),
            128,
            DIMENSIONS[model],
        ):
            raise ValueError(f"Unexpected vectors shape: {path}")

        train_indices, test_indices = folds[fold]
        fold_rows = []

        for layer_index, layer in enumerate(layers):
            basis = vectors[layer_index]

            gram = basis @ basis.T
            if not np.allclose(
                gram,
                np.eye(128, dtype=np.float32),
                atol=2e-3,
            ):
                raise ValueError(
                    f"Non-orthonormal SVD basis: {path}, layer {layer}"
                )

            train_parts = [
                array[train_indices, layer, :]
                for array in differences
            ]
            test_parts = [
                array[test_indices, layer, :]
                for array in differences
            ]

            train_matrix = np.concatenate(train_parts, axis=0)
            test_matrix = np.concatenate(test_parts, axis=0)

            train_energy = []
            test_energy = []

            for k in K_VALUES:
                train_value = calculate_energy(
                    train_matrix, basis, k
                )
                test_value = calculate_energy(
                    test_matrix, basis, k
                )

                train_energy.append(train_value)
                test_energy.append(test_value)

                fold_rows.append({
                    "fold": fold,
                    "layer": int(layer),
                    "method": f"svd_{k}",
                    "k": k,
                    "train_projected_energy_fraction": train_value,
                    "test_projected_energy_fraction": test_value,
                    "cumulative_explained_energy": train_value,
                })

            if (
                np.any(np.diff(train_energy) < -1e-5)
                or np.any(np.diff(test_energy) < -1e-5)
            ):
                raise ValueError(
                    f"Non-monotonic energy: {model}/{source}, "
                    f"fold {fold}, layer {layer}"
                )

        fold_df = pd.DataFrame(fold_rows)

        write_csv(
            directory / f"fold_{fold}_energy.csv",
            fold_df,
        )

        summary_rows.extend(fold_rows)

        print(
            f"  Fold {fold}/5 complete: "
            f"{len(layers)} layers, k={K_VALUES}",
            flush=True,
        )

    df = pd.DataFrame(summary_rows)

    metrics = [
        "train_projected_energy_fraction",
        "test_projected_energy_fraction",
        "cumulative_explained_energy",
    ]

    summary = (
        df.groupby(["layer", "method", "k"], sort=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        f"{metric}_{statistic}"
        if statistic else metric
        for metric, statistic in summary.columns.to_flat_index()
    ]

    write_csv(
        directory / "energy_five_fold_mean_std.csv",
        summary,
    )


def main():
    torch.set_num_threads(4)

    for model, (component, n_layers) in COMPONENTS.items():
        print(f"\n{'=' * 70}\n{model} / {component}", flush=True)

        folds = load_fold_indices(model)
        dim = DIMENSIONS[model]

        for source in SOURCES:
            print(f"\nSOURCE: {source}", flush=True)

            if source == "combined":
                modalities = ("text", "audio")
            else:
                modalities = (source,)

            differences = [
                load_differences(
                    model, modality, component, n_layers, dim
                )
                for modality in modalities
            ]

            process_source(
                model, source, component, folds, differences
            )

            del differences

    print("\nCOMPLETE: all 54 energy CSVs replaced", flush=True)


if __name__ == "__main__":
    main()
