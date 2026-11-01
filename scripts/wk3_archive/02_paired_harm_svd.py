#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

MODELS = {
    "qwen2_audio": ("decoder", "audio_encoder"),
    "qwen25_omni": ("thinker", "audio_encoder"),
    "kimi_audio": ("shared_decoder", "text_mimo", "audio_encoder"),
}

LANGUAGES = ("chinese", "english", "russian", "spanish")
CONDITIONS = (
    "content_dilution",
    "literal",
    "narrative_framing",
    "original",
)
FOLDS = (1, 2, 3, 4, 5)
K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
RANK = max(K_VALUES)

ENERGY_FIELDS = (
    "fold", "layer", "k",
    "train_energy", "test_energy",
    "mean_energy", "test_mean_signed_projection",
)

OVERLAP_FIELDS = (
    "fold", "layer_from", "layer_to", "k",
    "mean_principal_cosine",
    "mean_corresponding_vector_abs_cosine",
    "mean_direction_signed_cosine",
)

GROUP_FIELDS = (
    "fold", "layer", "component",
    "partition", "group_type", "group",
    "n", "mean_signed_score", "mean_abs_score",
    "component_energy_share",
    "component_energy_per_example",
)


def formatted(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return value


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: formatted(row[field])
                for field in fields
            })


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def task_id(split, index, condition, language, modality):
    return (
        f"jbb_{split}_{index:03d}__"
        f"{condition}__{language}__{modality}"
    )


def load_folds(model):
    fold_dir = RESULTS / model / "wk3/dataset_summary/folds"
    assignments = {}

    for fold in FOLDS:
        rows = read_csv(fold_dir / f"fold_{fold}_test.csv")

        if len(rows) != 1280:
            raise ValueError(
                f"{model} fold {fold}: expected 1280 test rows"
            )

        seen = set()
        indices = set()

        for row in rows:
            index = int(row["base_id"].rsplit("_", 1)[1])
            expected = task_id(
                row["split"], index, row["condition"],
                row["language"], row["modality"]
            )

            if row["task_id"] != expected:
                raise ValueError(f"Invalid task identity: {row}")

            if row["task_id"] in seen:
                raise ValueError(f"Duplicate task: {expected}")

            if row["partition"] != "test":
                raise ValueError(f"Invalid partition: {expected}")

            if int(row["test_fold"]) != fold:
                raise ValueError(f"Incorrect fold: {expected}")

            previous = assignments.get(index)
            if previous is not None and previous != fold:
                raise ValueError(
                    f"Topic {index} appears in multiple test folds"
                )

            assignments[index] = fold
            seen.add(expected)
            indices.add(index)

        if len(indices) != 20:
            raise ValueError(
                f"{model} fold {fold}: expected 20 topic indices"
            )

        expected_tasks = {
            task_id(split, index, condition, language, modality)
            for index in indices
            for split in ("benign", "harmful")
            for condition in CONDITIONS
            for language in LANGUAGES
            for modality in ("text", "audio")
        }

        if seen != expected_tasks:
            raise ValueError(
                f"{model} fold {fold}: incomplete matched coverage"
            )

        train_rows = read_csv(
            fold_dir / f"fold_{fold}_train.csv"
        )

        if len(train_rows) != 5120:
            raise ValueError(
                f"{model} fold {fold}: expected 5120 train rows"
            )

        train_ids = {row["task_id"] for row in train_rows}

        expected_train = {
            task_id(split, index, condition, language, modality)
            for index in range(100)
            if index not in indices
            for split in ("benign", "harmful")
            for condition in CONDITIONS
            for language in LANGUAGES
            for modality in ("text", "audio")
        }

        if train_ids != expected_train:
            raise ValueError(
                f"{model} fold {fold}: invalid training coverage"
            )

    if set(assignments) != set(range(100)):
        raise ValueError(f"{model}: expected 100 topic indices")

    return assignments


def read_representation(path, expected_id, component):
    if not path.is_file():
        raise FileNotFoundError(path)

    record = torch.load(
        path, map_location="cpu", weights_only=True
    )

    if record.get("task_id") != expected_id:
        raise ValueError(f"Task mismatch: {path}")

    tensor = record[component].float()

    if component == "audio_encoder":
        if tensor.ndim != 3 or tensor.shape[1] != 8:
            raise ValueError(f"Invalid encoder shape: {path}")
        tensor = tensor.mean(dim=1)
    elif tensor.ndim != 2:
        raise ValueError(f"Invalid decoder shape: {path}")

    result = tensor.numpy()

    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite values: {path}")

    return result


def load_differences(model, modality, component):
    directory = (
        RESULTS / model / "5fold/representations_shared"
    )

    differences = None
    topics = np.empty(1600, dtype=np.int32)
    languages = np.empty(1600, dtype=np.int8)
    conditions = np.empty(1600, dtype=np.int8)

    pos = 0

    for index in range(100):
        for language_index, language in enumerate(LANGUAGES):
            for condition_index, condition in enumerate(CONDITIONS):
                benign_id = task_id(
                    "benign", index, condition, language, modality
                )
                harmful_id = task_id(
                    "harmful", index, condition, language, modality
                )

                benign = read_representation(
                    directory / f"{benign_id}.pt",
                    benign_id,
                    component,
                )
                harmful = read_representation(
                    directory / f"{harmful_id}.pt",
                    harmful_id,
                    component,
                )

                if benign.shape != harmful.shape:
                    raise ValueError(
                        f"Shape mismatch: {benign_id}"
                    )

                if differences is None:
                    differences = np.empty(
                        (1600, *benign.shape),
                        dtype=np.float32,
                    )

                if benign.shape != differences.shape[1:]:
                    raise ValueError(
                        f"Unexpected shape: {harmful_id}"
                    )

                differences[pos] = harmful - benign
                topics[pos] = index
                languages[pos] = language_index
                conditions[pos] = condition_index
                pos += 1

        if (index + 1) % 20 == 0:
            print(
                f"  Loaded {index + 1}/100 topic indices",
                flush=True,
            )

    if pos != 1600:
        raise ValueError("Expected 1600 matched differences")

    zero_layers = np.flatnonzero(
        np.all(differences == 0, axis=(0, 2))
    ).tolist()

    if zero_layers == [0]:
        print(
            "  Boundary 0 has zero differences; excluded.",
            flush=True,
        )
        differences = differences[:, 1:, :]
        offset = 1
    elif zero_layers:
        raise ValueError(
            f"Unexpected zero-difference boundaries: {zero_layers}"
        )
    else:
        offset = 0

    return differences, topics, languages, conditions, offset


def randomized_svd(matrix, seed):
    rows, dimensions = matrix.shape

    if min(rows, dimensions) < RANK:
        raise ValueError("Insufficient matrix rank capacity")

    rng = np.random.default_rng(seed)
    sketch_size = min(RANK + 32, rows, dimensions)

    omega = rng.standard_normal(
        (dimensions, sketch_size),
        dtype=np.float32,
    )

    q, _ = np.linalg.qr(
        matrix @ omega,
        mode="reduced",
    )

    for _ in range(3):
        z, _ = np.linalg.qr(
            matrix.T @ q,
            mode="reduced",
        )
        q, _ = np.linalg.qr(
            matrix @ z,
            mode="reduced",
        )

    compressed = q.T @ matrix

    _, singular_values, vt = np.linalg.svd(
        compressed,
        full_matrices=False,
    )

    return (
        singular_values[:RANK].astype(np.float64),
        vt[:RANK].astype(np.float32),
    )


def energy_fractions(projections, total_energy):
    squared = projections.astype(np.float64) ** 2
    return np.cumsum(
        np.sum(squared, axis=0)
    ) / total_energy


def overlap(previous, current, k):
    left = previous[:k].astype(np.float64)
    right = current[:k].astype(np.float64)

    principal_cosines = np.linalg.svd(
        left @ right.T,
        compute_uv=False,
    )
    principal_cosines = np.clip(
        principal_cosines, 0.0, 1.0
    )

    corresponding = np.abs(
        np.sum(left * right, axis=1)
    )

    return (
        float(principal_cosines.mean()),
        float(corresponding.mean()),
    )


def group_statistics(
    projections,
    topics,
    languages,
    conditions,
    fold,
    layer,
    partition,
):
    rows = []

    groups = (
        ("language", languages, LANGUAGES),
        ("condition", conditions, CONDITIONS),
    )

    squared = projections.astype(np.float64) ** 2
    total_by_component = squared.sum(axis=0)

    for group_type, labels, names in groups:
        for group_index, group_name in enumerate(names):
            mask = labels == group_index
            selected = projections[mask]
            selected_squared = squared[mask]

            group_energy = selected_squared.sum(axis=0)

            for component in range(RANK):
                denominator = total_by_component[component]

                rows.append({
                    "fold": fold,
                    "layer": layer,
                    "component": component + 1,
                    "partition": partition,
                    "group_type": group_type,
                    "group": group_name,
                    "n": len(selected),
                    "mean_signed_score": float(
                        selected[:, component].mean()
                    ),
                    "mean_abs_score": float(
                        np.abs(selected[:, component]).mean()
                    ),
                    "component_energy_share": (
                        float(group_energy[component] / denominator)
                        if denominator > 0 else float("nan")
                    ),
                    "component_energy_per_example": float(
                        group_energy[component] / len(selected)
                    ),
                })

    return rows


def analyze_fold(
    differences,
    topics,
    languages,
    conditions,
    assignments,
    fold,
    offset,
    output,
    seed,
):
    is_train = np.array(
        [assignments[int(index)] != fold for index in topics],
        dtype=bool,
    )
    is_test = ~is_train

    train = differences[is_train]
    test = differences[is_test]

    if train.shape[0] != 1280 or test.shape[0] != 320:
        raise ValueError(f"Invalid sample counts: fold {fold}")

    train_topics = topics[is_train]
    test_topics = topics[is_test]

    train_languages = languages[is_train]
    test_languages = languages[is_test]

    train_conditions = conditions[is_train]
    test_conditions = conditions[is_test]

    n_layers = differences.shape[1]
    width = differences.shape[2]

    vectors = np.empty(
        (n_layers, RANK, width),
        dtype=np.float32,
    )
    means = np.empty(
        (n_layers, width),
        dtype=np.float32,
    )
    singular_values = np.empty(
        (n_layers, RANK),
        dtype=np.float64,
    )

    train_scores = np.empty(
        (n_layers, len(train), RANK),
        dtype=np.float32,
    )
    test_scores = np.empty(
        (n_layers, len(test), RANK),
        dtype=np.float32,
    )
    train_u = np.empty_like(train_scores)

    energy_rows = []
    overlap_rows = []
    group_rows = []

    previous_basis = None
    previous_mean_unit = None

    for local_layer in range(n_layers):
        layer = local_layer + offset

        x = np.ascontiguousarray(
            train[:, local_layer, :]
        )
        heldout = np.ascontiguousarray(
            test[:, local_layer, :]
        )

        train_total = float(
            np.sum(x.astype(np.float64) ** 2)
        )
        test_total = float(
            np.sum(heldout.astype(np.float64) ** 2)
        )

        if train_total <= 0 or test_total <= 0:
            raise ValueError(
                f"Zero energy: fold={fold}, layer={layer}"
            )

        mean = x.mean(axis=0, dtype=np.float64)
        mean = mean.astype(np.float32)
        mean_norm = float(np.linalg.norm(mean))

        mean_fraction = (
            len(x) * mean_norm ** 2 / train_total
        )

        if mean_norm > 0:
            mean_unit = mean / mean_norm
            test_mean_projection = heldout @ mean_unit
            signed_test_mean = float(
                test_mean_projection.mean()
            )
        else:
            mean_unit = None
            signed_test_mean = float("nan")

        s, basis = randomized_svd(
            x,
            seed + fold * 1000 + layer,
        )

        z_train = x @ basis.T
        z_test = heldout @ basis.T

        u = np.divide(
            z_train,
            s[np.newaxis, :],
            out=np.zeros_like(z_train),
            where=s[np.newaxis, :] > 0,
        )

        vectors[local_layer] = basis
        means[local_layer] = mean
        singular_values[local_layer] = s
        train_scores[local_layer] = z_train
        test_scores[local_layer] = z_test
        train_u[local_layer] = u

        train_energy = energy_fractions(
            z_train, train_total
        )
        test_energy = energy_fractions(
            z_test, test_total
        )

        if train_energy[-1] > 1.001:
            raise ValueError(
                f"Invalid training projection energy at layer {layer}"
            )
        if test_energy[-1] > 1.001:
            raise ValueError(
                f"Invalid held-out projection energy at layer {layer}"
            )

        if (
            previous_mean_unit is not None
            and mean_unit is not None
        ):
            mean_cosine = float(
                previous_mean_unit @ mean_unit
            )
        else:
            mean_cosine = float("nan")

        for k in K_VALUES:
            energy_rows.append({
                "fold": fold,
                "layer": layer,
                "k": k,
                "train_energy": float(train_energy[k - 1]),
                "test_energy": float(test_energy[k - 1]),
                "mean_energy": float(mean_fraction),
                "test_mean_signed_projection":
                    signed_test_mean,
            })

            if previous_basis is not None:
                principal, corresponding = overlap(
                    previous_basis,
                    basis,
                    k,
                )

                overlap_rows.append({
                    "fold": fold,
                    "layer_from": layer - 1,
                    "layer_to": layer,
                    "k": k,
                    "mean_principal_cosine": principal,
                    "mean_corresponding_vector_abs_cosine":
                        corresponding,
                    "mean_direction_signed_cosine":
                        mean_cosine,
                })

        group_rows.extend(
            group_statistics(
                z_train,
                train_topics,
                train_languages,
                train_conditions,
                fold,
                layer,
                "train",
            )
        )
        group_rows.extend(
            group_statistics(
                z_test,
                test_topics,
                test_languages,
                test_conditions,
                fold,
                layer,
                "test",
            )
        )

        print(
            f"    fold={fold} layer={layer:02d} "
            f"train k=1/8/32/128: "
            f"{train_energy[0]:.3f}/"
            f"{train_energy[7]:.3f}/"
            f"{train_energy[31]:.3f}/"
            f"{train_energy[127]:.3f} | "
            f"test: "
            f"{test_energy[0]:.3f}/"
            f"{test_energy[7]:.3f}/"
            f"{test_energy[31]:.3f}/"
            f"{test_energy[127]:.3f}",
            flush=True,
        )

        previous_basis = basis
        previous_mean_unit = mean_unit

    output.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output / f"fold_{fold}_directions.npz",
        vectors=vectors,
        means=means,
        singular_values=singular_values,
        layer_boundaries=np.arange(
            offset, offset + n_layers
        ),
        k_values=np.array(K_VALUES),
    )

    np.savez_compressed(
        output / f"fold_{fold}_component_scores.npz",
        train_u=train_u,
        train_scores=train_scores,
        test_scores=test_scores,
        train_topic_indices=train_topics,
        test_topic_indices=test_topics,
        train_language_indices=train_languages,
        test_language_indices=test_languages,
        train_condition_indices=train_conditions,
        test_condition_indices=test_conditions,
        languages=np.array(LANGUAGES),
        conditions=np.array(CONDITIONS),
        layer_boundaries=np.arange(
            offset, offset + n_layers
        ),
    )

    write_csv(
        output / f"fold_{fold}_energy.csv",
        energy_rows,
        ENERGY_FIELDS,
    )
    write_csv(
        output / f"fold_{fold}_overlap.csv",
        overlap_rows,
        OVERLAP_FIELDS,
    )
    write_csv(
        output / f"fold_{fold}_group_scores.csv",
        group_rows,
        GROUP_FIELDS,
    )


def aggregate(output):
    specifications = (
        ("energy.csv", "energy", ENERGY_FIELDS),
        (
            "adjacent_subspace_overlap.csv",
            "overlap",
            OVERLAP_FIELDS,
        ),
        (
            "component_group_scores.csv",
            "group_scores",
            GROUP_FIELDS,
        ),
    )

    for name, suffix, fields in specifications:
        rows = []
        for fold in FOLDS:
            rows.extend(
                read_csv(
                    output / f"fold_{fold}_{suffix}.csv"
                )
            )
        write_csv(output / name, rows, fields)


def run_component(model, modality, component, assignments, seed):
    output = (
        RESULTS / model / "wk3/paired_harm_svd"
        / modality / component
    )

    required = lambda fold: (
        output / f"fold_{fold}_directions.npz",
        output / f"fold_{fold}_component_scores.npz",
        output / f"fold_{fold}_energy.csv",
        output / f"fold_{fold}_overlap.csv",
        output / f"fold_{fold}_group_scores.csv",
    )

    if all(
        all(path.is_file() for path in required(fold))
        for fold in FOLDS
    ):
        aggregate(output)
        print(f"ALREADY COMPLETE: {output}", flush=True)
        return

    print(
        f"\nMODEL={model}, MODALITY={modality}, "
        f"COMPONENT={component}",
        flush=True,
    )

    differences, topics, languages, conditions, offset = (
        load_differences(model, modality, component)
    )

    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": model,
        "modality": modality,
        "component": component,
        "difference": "harmful_minus_benign",
        "total_topic_pairs": 100,
        "differences_per_modality": 1600,
        "train_differences_per_fold": 1280,
        "test_differences_per_fold": 320,
        "folds": 5,
        "k_values": list(K_VALUES),
        "first_boundary": offset,
        "svd_method": "uncentered_randomized_svd",
        "oversampling": 32,
        "power_iterations": 3,
        "component_scores": "D_times_V",
        "train_u": (
            "approximate_left_singular_coordinates_"
            "calculated_as_DV_divided_by_singular_values"
        ),
        "subspace_similarity": (
            "mean_cosine_of_principal_angles"
        ),
        "audio_encoder_pooling": (
            "mean_over_eight_temporal_bins"
            if component == "audio_encoder" else None
        ),
        "csv_decimal_places": 3,
        "saved_arrays_unrounded": True,
        "seed": seed,
    }

    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    for fold in FOLDS:
        if all(path.is_file() for path in required(fold)):
            print(f"  Skipping completed fold {fold}", flush=True)
            continue

        print(f"\n  ANALYZING FOLD {fold}", flush=True)

        analyze_fold(
            differences,
            topics,
            languages,
            conditions,
            assignments,
            fold,
            offset,
            output,
            seed,
        )

        print(f"  SAVED FOLD {fold}", flush=True)

    aggregate(output)

    print(f"\nCOMPLETE: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=list(MODELS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true")

    args = parser.parse_args()

    torch.set_num_threads(4)

    assignments = {
        model: load_folds(model)
        for model in args.models
    }

    reference = assignments[args.models[0]]

    for model in args.models:
        if assignments[model] != reference:
            raise ValueError(
                f"{model}: topic assignments differ across models"
            )

    for model in args.models:
        old_output = (
            RESULTS / model / "wk3/paired_harm_svd"
        )

        if args.reset and old_output.exists():
            shutil.rmtree(old_output)

        for modality in ("text", "audio"):
            for component in MODELS[model]:
                if modality == "text" and component == "audio_encoder":
                    continue

                run_component(
                    model, modality, component,
                    assignments[model], args.seed,
                )

    print("\nALL ANALYSES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
