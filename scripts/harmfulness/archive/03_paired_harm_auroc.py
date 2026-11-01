#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

COMPONENTS = {
    "qwen2_audio": ("decoder", "audio_encoder"),
    "qwen25_omni": ("thinker", "audio_encoder"),
    "kimi_audio": ("shared_decoder", "text_mimo", "audio_encoder"),
}

LANGUAGES = ("chinese", "english", "russian", "spanish")
CONDITIONS = (
    "content_dilution", "literal", "narrative_framing", "original"
)
K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
METHODS = ("mean",) + tuple(f"svd_{k}" for k in K_VALUES)

PREDICTION_FIELDS = (
    "model", "modality", "component", "fold", "layer", "method",
    "topic", "language", "condition", "benign_score", "harmful_score",
    "paired_correct",
)

METRIC_FIELDS = (
    "model", "modality", "component", "fold", "layer", "method",
    "group_type", "group", "n_pairs", "auroc", "paired_accuracy",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for field in fields:
                value = row[field]
                formatted[field] = (
                    f"{float(value):.3f}"
                    if isinstance(value, (float, np.floating))
                    else value
                )
            writer.writerow(formatted)


def make_id(split, topic, condition, language, modality):
    return (
        f"jbb_{split}_{topic:03d}__"
        f"{condition}__{language}__{modality}"
    )


def topic_folds(model):
    directory = RESULTS / model / "wk3/dataset_summary/folds"
    assignment = {}

    for fold in range(1, 6):
        test_rows = read_csv(directory / f"fold_{fold}_test.csv")
        train_rows = read_csv(directory / f"fold_{fold}_train.csv")

        if len(test_rows) != 1280 or len(train_rows) != 5120:
            raise ValueError(f"Incorrect fold size: {model}, {fold}")

        topics = set()
        observed_ids = set()

        for row in test_rows:
            topic = int(row["base_id"].rsplit("_", 1)[1])
            topics.add(topic)
            observed_ids.add(row["task_id"])

            if row["partition"] != "test":
                raise ValueError(f"Invalid partition: {row}")
            if int(row["test_fold"]) != fold:
                raise ValueError(f"Invalid fold: {row}")

            previous = assignment.get(topic)
            if previous is not None and previous != fold:
                raise ValueError(f"Topic leakage: {topic}")

            assignment[topic] = fold

        if len(topics) != 20 or len(observed_ids) != 1280:
            raise ValueError(f"Invalid test coverage in fold {fold}")

        expected_test = {
            make_id(split, topic, condition, language, modality)
            for topic in topics
            for split in ("benign", "harmful")
            for condition in CONDITIONS
            for language in LANGUAGES
            for modality in ("text", "audio")
        }

        expected_train = {
            make_id(split, topic, condition, language, modality)
            for topic in range(100) if topic not in topics
            for split in ("benign", "harmful")
            for condition in CONDITIONS
            for language in LANGUAGES
            for modality in ("text", "audio")
        }

        if observed_ids != expected_test:
            raise ValueError(f"Incorrect test tasks, fold {fold}")

        if {r["task_id"] for r in train_rows} != expected_train:
            raise ValueError(f"Incorrect training tasks, fold {fold}")

    if set(assignment) != set(range(100)):
        raise ValueError("Incomplete topic folds")

    return assignment


def load_tensor(path, expected_id, component):
    record = torch.load(
        path, map_location="cpu", weights_only=True
    )

    if record.get("task_id") != expected_id:
        raise ValueError(f"Representation identity mismatch: {path}")

    tensor = record[component].float()

    if component == "audio_encoder":
        if tensor.ndim != 3 or tensor.shape[1] != 8:
            raise ValueError(f"Unexpected audio encoder shape: {path}")
        tensor = tensor.mean(dim=1)
    elif tensor.ndim != 2:
        raise ValueError(f"Unexpected hidden-state shape: {path}")

    if not torch.isfinite(tensor).all():
        raise ValueError(f"Nonfinite representation: {path}")

    return tensor


def load_representations(model, modality, component):
    root = RESULTS / model / "5fold/representations_shared"

    benign = None
    harmful = None
    metadata = []
    position = 0

    for topic in range(100):
        for language in LANGUAGES:
            for condition in CONDITIONS:
                names = {
                    split: make_id(
                        split, topic, condition, language, modality
                    )
                    for split in ("benign", "harmful")
                }

                b = load_tensor(
                    root / f"{names['benign']}.pt",
                    names["benign"],
                    component,
                )
                h = load_tensor(
                    root / f"{names['harmful']}.pt",
                    names["harmful"],
                    component,
                )

                if b.shape != h.shape:
                    raise ValueError(f"Paired shape mismatch: {topic}")

                if benign is None:
                    benign = torch.empty(
                        (1600, *b.shape), dtype=torch.float32
                    )
                    harmful = torch.empty_like(benign)

                if tuple(b.shape) != tuple(benign.shape[1:]):
                    raise ValueError(f"Inconsistent shape: {topic}")

                benign[position] = b
                harmful[position] = h
                metadata.append((topic, language, condition))
                position += 1

        if (topic + 1) % 20 == 0:
            print(
                f"    Loaded {topic + 1}/100 topics",
                flush=True,
            )

    if position != 1600:
        raise ValueError("Expected 1600 matched pairs")

    return benign, harmful, metadata


def metrics_for_rows(rows, fold, group_type, group):
    benign = np.array(
        [float(row["benign_score"]) for row in rows],
        dtype=np.float64,
    )
    harmful = np.array(
        [float(row["harmful_score"]) for row in rows],
        dtype=np.float64,
    )

    labels = np.concatenate(
        (np.zeros(len(rows)), np.ones(len(rows)))
    )
    scores = np.concatenate((benign, harmful))

    auroc = float(roc_auc_score(labels, scores))
    paired_accuracy = float(
        np.mean(harmful > benign)
        + 0.5 * np.mean(harmful == benign)
    )

    first = rows[0]

    return {
        "model": first["model"],
        "modality": first["modality"],
        "component": first["component"],
        "fold": fold,
        "layer": first["layer"],
        "method": first["method"],
        "group_type": group_type,
        "group": group,
        "n_pairs": len(rows),
        "auroc": auroc,
        "paired_accuracy": paired_accuracy,
    }


def summarize(rows, fold):
    results = []

    group_keys = set(
        (row["layer"], row["method"]) for row in rows
    )

    for layer, method in sorted(group_keys):
        selected = [
            row for row in rows
            if row["layer"] == layer and row["method"] == method
        ]

        groups = [("all", "all", selected)]

        for language in LANGUAGES:
            groups.append((
                "language", language,
                [r for r in selected if r["language"] == language],
            ))

        for condition in CONDITIONS:
            groups.append((
                "condition", condition,
                [r for r in selected if r["condition"] == condition],
            ))

        for language in LANGUAGES:
            for condition in CONDITIONS:
                groups.append((
                    "language_condition",
                    f"{language}__{condition}",
                    [
                        r for r in selected
                        if r["language"] == language
                        and r["condition"] == condition
                    ],
                ))

        for group_type, group_name, subset in groups:
            if not subset:
                raise ValueError(
                    f"Empty group: {layer}, {method}, {group_name}"
                )
            results.append(
                metrics_for_rows(
                    subset, fold, group_type, group_name
                )
            )

    return results


def analyze(model, modality, component, assignment, device):
    base = RESULTS / model / "wk3"
    svd_root = base / "paired_harm_svd" / modality / component
    output = base / "paired_harm_auroc" / modality / component
    output.mkdir(parents=True, exist_ok=True)

    print(
        f"\nMODEL={model} MODALITY={modality} "
        f"COMPONENT={component}",
        flush=True,
    )

    benign, harmful, metadata = load_representations(
        model, modality, component
    )

    all_predictions = []
    fold_metrics = []

    for fold in range(1, 6):
        indices = [
            i for i, entry in enumerate(metadata)
            if assignment[entry[0]] == fold
        ]

        if len(indices) != 320:
            raise ValueError(
                f"Fold {fold}: expected 320 held-out pairs"
            )

        svd_path = svd_root / f"fold_{fold}_directions.npz"

        with np.load(svd_path, allow_pickle=False) as saved:
            vectors = torch.from_numpy(
                saved["vectors"].copy()
            ).to(device)
            means = torch.from_numpy(
                saved["means"].copy()
            ).to(device)
            layer_numbers = saved["layer_boundaries"].tolist()

        if (
            vectors.shape[0] != len(layer_numbers)
            or means.shape[0] != len(layer_numbers)
            or vectors.shape[1] < max(K_VALUES)
            or vectors.shape[2] != benign.shape[2]
        ):
            raise ValueError(
                f"Incompatible SVD arrays: {svd_path}"
            )

        b = benign[indices].to(device)
        h = harmful[indices].to(device)
        fold_predictions = []

        with torch.no_grad():
            for position, layer in enumerate(layer_numbers):
                if layer >= benign.shape[1]:
                    raise ValueError(f"Invalid layer {layer}")

                basis = vectors[position]
                mean = means[position]

                directions = []
                method_names = []

                mean_norm = torch.linalg.vector_norm(mean)

                if mean_norm <= 0:
                    raise ValueError(
                        f"Zero training mean: fold {fold}, layer {layer}"
                    )

                directions.append(mean / mean_norm)
                method_names.append("mean")

                for k in K_VALUES:
                    top_basis = basis[:k]
                    coordinates = top_basis @ mean
                    projected_mean = coordinates @ top_basis
                    norm = torch.linalg.vector_norm(projected_mean)

                    if norm <= 1e-10:
                        raise ValueError(
                            f"Near-zero projected mean: "
                            f"fold {fold}, layer {layer}, k={k}"
                        )

                    directions.append(projected_mean / norm)
                    method_names.append(f"svd_{k}")

                weight_matrix = torch.stack(
                    directions, dim=1
                )

                b_scores = (
                    b[:, layer, :] @ weight_matrix
                ).cpu().numpy()
                h_scores = (
                    h[:, layer, :] @ weight_matrix
                ).cpu().numpy()

                for local_index, global_index in enumerate(indices):
                    topic, language, condition = metadata[global_index]

                    for method_index, method in enumerate(method_names):
                        bs = float(
                            b_scores[local_index, method_index]
                        )
                        hs = float(
                            h_scores[local_index, method_index]
                        )

                        fold_predictions.append({
                            "model": model,
                            "modality": modality,
                            "component": component,
                            "fold": fold,
                            "layer": layer,
                            "method": method,
                            "topic": topic,
                            "language": language,
                            "condition": condition,
                            "benign_score": bs,
                            "harmful_score": hs,
                            "paired_correct": (
                                1.0 if hs > bs
                                else 0.5 if hs == bs
                                else 0.0
                            ),
                        })

                if (position + 1) % 5 == 0 or position == len(layer_numbers) - 1:
                    print(
                        f"    Fold {fold}: layer {layer}, "
                        f"{position + 1}/{len(layer_numbers)}",
                        flush=True,
                    )

        expected_rows = len(layer_numbers) * len(METHODS) * 320
        if len(fold_predictions) != expected_rows:
            raise ValueError("Incomplete fold predictions")

        write_csv(
            output / f"fold_{fold}_predictions.csv",
            fold_predictions,
            PREDICTION_FIELDS,
        )

        fold_metrics.extend(
            summarize(fold_predictions, fold)
        )
        all_predictions.extend(fold_predictions)

        del b, h, vectors, means
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"  Completed fold {fold}", flush=True)

    write_csv(
        output / "heldout_predictions.csv",
        all_predictions,
        PREDICTION_FIELDS,
    )
    write_csv(
        output / "fold_metrics.csv",
        fold_metrics,
        METRIC_FIELDS,
    )

    pooled_metrics = summarize(all_predictions, "pooled")

    write_csv(
        output / "pooled_metrics.csv",
        pooled_metrics,
        METRIC_FIELDS,
    )

    metadata_output = {
        "model": model,
        "modality": modality,
        "component": component,
        "fivefold": "seed42_topic_pair_disjoint",
        "heldout_topic_indices_per_fold": 20,
        "heldout_pairs_per_fold": 320,
        "total_out_of_fold_pairs": 1600,
        "methods": list(METHODS),
        "score": "dot_product_with_training_direction",
        "svd_k_direction": (
            "normalized_projection_of_training_harmful_minus_"
            "benign_mean_onto_training_top_k_svd_subspace"
        ),
        "auroc": (
            "heldout_harmful_vs_benign_classification"
        ),
        "paired_accuracy": (
            "fraction_harmful_score_greater_than_matched_benign_"
            "score_with_half_credit_for_ties"
        ),
        "normalization": "none_applied_to_individual_activations",
        "csv_decimal_places": 3,
    }

    (output / "metadata.json").write_text(
        json.dumps(metadata_output, indent=2) + "\n",
        encoding="utf-8",
    )

    del benign, harmful
    print(f"COMPLETE: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=tuple(COMPONENTS)
    )
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} is unavailable")

    torch.set_num_threads(4)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(
        f"Device: {torch.cuda.get_device_name(device)}",
        flush=True,
    )

    assignments = topic_folds(args.model)

    for modality in ("text", "audio"):
        for component in COMPONENTS[args.model]:
            if modality == "text" and component == "audio_encoder":
                continue

            analyze(
                args.model, modality, component,
                assignments, device
            )

    print(f"\nMODEL COMPLETE: {args.model}", flush=True)


if __name__ == "__main__":
    main()
