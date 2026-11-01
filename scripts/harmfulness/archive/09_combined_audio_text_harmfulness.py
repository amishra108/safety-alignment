#!/usr/bin/env python3

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"
SCRIPTS = ROOT / "scripts/wk3"

MODELS = {
    "qwen2_audio": "decoder",
    "kimi_audio": "shared_decoder",
    "qwen25_omni": "thinker",
}

K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
METHODS = ("mean",) + tuple(f"svd_{k}" for k in K_VALUES)
OUTCOMES = ("00", "01", "10", "11")
KEYS = ["fold", "layer", "topic", "language", "condition"]


def load_existing_module():
    path = SCRIPTS / "03_paired_harm_auroc.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location("paired_harm_auroc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("topic_folds", "load_representations"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"Required existing function missing: {name}")

    return module


def unit(x):
    norm = torch.linalg.vector_norm(x)
    if not torch.isfinite(norm) or norm <= 1e-10:
        raise ValueError("Invalid zero or nonfinite direction")
    return x / norm


def fit_directions(differences, rank, seed):
    torch.manual_seed(seed)

    mean = differences.mean(dim=0)

    if not torch.isfinite(mean).all():
        raise ValueError("Nonfinite combined mean")

    q = min(rank + 8, min(differences.shape))
    _, singular_values, v = torch.svd_lowrank(
        differences,
        q=q,
        niter=3,
    )

    basis = v[:, :rank].T.contiguous()

    if not torch.isfinite(basis).all():
        raise ValueError("Nonfinite SVD basis")

    return mean, basis, singular_values[:rank]


def method_directions(mean, basis):
    directions = {"mean": unit(mean)}

    for k in K_VALUES:
        top = basis[:k]
        projected = (top @ mean) @ top
        directions[f"svd_{k}"] = unit(projected)

    return directions


def read_saved_directions(path, device):
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as saved:
        return {
            "means": torch.from_numpy(saved["means"].copy()).to(device),
            "vectors": torch.from_numpy(saved["vectors"].copy()).to(device),
            "layers": saved["layer_boundaries"].astype(int).tolist(),
        }


def cosine(a, b):
    return float(torch.dot(unit(a), unit(b)).item())


def subspace_overlap(a, b):
    k = a.shape[0]
    return float(
        ((a @ b.T) ** 2).sum().item() / k
    )


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    if "method" in df.columns:
        method_order = {
            name: index for index, name in enumerate(METHODS)
        }
        unknown = set(df["method"].dropna()) - set(method_order)
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")

        df["_method_order"] = df["method"].map(method_order)

    order = [
        column for column in (
            "model", "component", "fold", "layer",
            "_method_order", "modality", "group_type",
            "group", "outcome", "topic", "language", "condition"
        )
        if column in df.columns
    ]

    if order:
        df = df.sort_values(order, kind="stable")

    df = df.drop(columns=["_method_order"], errors="ignore")

    for column in df.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].round(3)

    df.to_csv(path, index=False, float_format="%.3f")

    print(f"SAVED {path}", flush=True)


def get_decisions(base, component, modality, fold):
    path = (
        base / "paired_harm_auroc"
        / modality / component / f"fold_{fold}_predictions.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = set(KEYS + ["method", "benign_decision", "harmful_decision"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")

    df = df.loc[df["method"] == "mean",
                KEYS + ["benign_decision", "harmful_decision"]].copy()

    if df.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate decisions")

    if len(df) == 0:
        raise ValueError(f"{path}: no mean rows")

    return df


def add_normalized_scores(predictions):
    result = predictions.copy()

    grouping = ["fold", "layer", "method", "modality"]

    benign_mean = result.groupby(grouping)["benign_score"].transform("mean")
    benign_sd = result.groupby(grouping)["benign_score"].transform("std")

    if benign_sd.isna().any() or (benign_sd <= 0).any():
        raise ValueError("Invalid within-modality benign SD")

    result["benign_z"] = (
        result["benign_score"] - benign_mean
    ) / benign_sd

    result["harmful_z"] = (
        result["harmful_score"] - benign_mean
    ) / benign_sd

    result["separation_z"] = (
        result["harmful_score"] - result["benign_score"]
    ) / benign_sd

    result["benign_reference_mean"] = benign_mean
    result["benign_reference_sd"] = benign_sd

    return result


def aggregate_matched(matched):
    rows = []

    group_definitions = {
        "all": ["layer", "method"],
        "language": ["layer", "method", "language"],
        "condition": ["layer", "method", "condition"],
        "language_condition": [
            "layer", "method", "language", "condition"
        ],
    }

    for group_type, keys in group_definitions.items():
        for values, subset in matched.groupby(keys, sort=True):
            if not isinstance(values, tuple):
                values = (values,)

            identity = dict(zip(keys, values))

            if group_type == "all":
                group_name = "all"
            elif group_type == "language":
                group_name = identity["language"]
            elif group_type == "condition":
                group_name = identity["condition"]
            else:
                group_name = (
                    identity["language"] + "__" + identity["condition"]
                )

            for outcome in OUTCOMES:
                selected = subset.loc[subset["outcome"] == outcome]
                n = len(selected)

                row = {
                    "layer": int(identity["layer"]),
                    "method": identity["method"],
                    "group_type": group_type,
                    "group": group_name,
                    "outcome": outcome,
                    "n": n,
                    "outcome_pct": 100 * n / len(subset),
                    "valid_folds": selected["fold"].nunique(),
                }

                for field in (
                    "benign_z_text",
                    "benign_z_audio",
                    "harmful_z_text",
                    "harmful_z_audio",
                    "score_diff",
                    "separation_diff",
                    "raw_score_diff",
                ):
                    row[f"{field}_mean"] = (
                        selected[field].mean() if n else np.nan
                    )

                for prefix, field in (
                    ("score", "score_diff"),
                    ("separation", "separation_diff"),
                ):
                    positives = int((selected[field] > 0).sum())
                    negatives = int((selected[field] < 0).sum())
                    ties = int((selected[field] == 0).sum())

                    if positives + negatives + ties != n:
                        raise ValueError("Unclassified comparison")

                    row[f"{prefix}_audio_gt_text_n"] = positives
                    row[f"{prefix}_text_gt_audio_n"] = negatives
                    row[f"{prefix}_equal_n"] = ties
                    row[f"{prefix}_audio_gt_text_pct"] = (
                        100 * positives / n if n else np.nan
                    )

                    if n:
                        fold_fractions = (
                            selected.assign(
                                audio_wins=selected[field] > 0
                            )
                            .groupby("fold")["audio_wins"]
                            .mean()
                        )
                        row[f"{prefix}_positive_folds"] = int(
                            (fold_fractions > 0.5).sum()
                        )
                    else:
                        row[f"{prefix}_positive_folds"] = 0

                rows.append(row)

    return rows


def run_model(model, module, device, rank):
    component = MODELS[model]
    base = RESULTS / model / "wk3"

    print(f"\n{'#' * 100}")
    print(f"MODEL {model} COMPONENT {component}")
    print(f"{'#' * 100}", flush=True)

    fold_assignment = module.topic_folds(model)

    text_b, text_h, text_metadata = module.load_representations(
        model, "text", component
    )
    audio_b, audio_h, audio_metadata = module.load_representations(
        model, "audio", component
    )

    if text_metadata != audio_metadata:
        raise ValueError(f"{model}: text/audio metadata order differs")

    if text_b.shape != audio_b.shape or text_h.shape != audio_h.shape:
        raise ValueError(f"{model}: incompatible representation shapes")

    n, total_layers, hidden_size = text_b.shape
    if n != 1600:
        raise ValueError(f"{model}: expected 1600 matched pairs, got {n}")

    direction_root = (
        base / "paired_harm_svd" / "combined" / component
    )
    projection_root = (
        base / "paired_harm_auroc" / "combined" / component
    )
    comparison_root = (
        base / "audio_text_combined_comparison"
    )

    direction_root.mkdir(parents=True, exist_ok=True)
    projection_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_comparisons = []

    for fold in range(1, 6):
        train_indices = [
            i for i, entry in enumerate(text_metadata)
            if fold_assignment[entry[0]] != fold
        ]
        test_indices = [
            i for i, entry in enumerate(text_metadata)
            if fold_assignment[entry[0]] == fold
        ]

        if len(train_indices) != 1280 or len(test_indices) != 320:
            raise ValueError(f"{model}: invalid fold {fold} partition")

        existing = {}

        for modality in ("text", "audio"):
            path = (
                base / "paired_harm_svd" / modality / component
                / f"fold_{fold}_directions.npz"
            )
            existing[modality] = read_saved_directions(path, device)

        layers = existing["text"]["layers"]

        if layers != existing["audio"]["layers"]:
            raise ValueError(f"{model}: incompatible saved layers")

        if not layers or len(set(layers)) != len(layers):
            raise ValueError(f"{model}: invalid or duplicate layer indices")

        if any(layer < 0 or layer >= total_layers for layer in layers):
            raise ValueError(
                f"{model}: saved layers {layers} incompatible "
                f"with representation shape {text_b.shape}"
            )

        print(
            f"{model}: representation positions={total_layers}, "
            f"analyzed layers={len(layers)}, "
            f"range={min(layers)}-{max(layers)}",
            flush=True,
        )

        saved_means = []
        saved_vectors = []
        saved_singular_values = []

        fold_predictions = []

        print(
            f"\n{model} FOLD {fold}/5 "
            f"train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )

        for position, layer in enumerate(layers):
            if layer < 0 or layer >= total_layers:
                raise ValueError(f"Invalid layer {layer}")

            text_difference = (
                text_h[train_indices, layer, :]
                - text_b[train_indices, layer, :]
            ).to(device)

            audio_difference = (
                audio_h[train_indices, layer, :]
                - audio_b[train_indices, layer, :]
            ).to(device)

            combined_difference = torch.cat(
                (text_difference, audio_difference), dim=0
            )

            combined_mean, combined_basis, singular_values = (
                fit_directions(
                    combined_difference,
                    rank,
                    42000 + fold * 100 + layer,
                )
            )

            text_reference_mean = existing["text"]["means"][position]
            audio_reference_mean = existing["audio"]["means"][position]

            mean_check = (
                text_reference_mean + audio_reference_mean
            ) / 2

            relative_error = (
                torch.linalg.vector_norm(combined_mean - mean_check)
                / torch.linalg.vector_norm(combined_mean)
            ).item()

            if relative_error > 0.005:
                raise ValueError(
                    f"{model} fold={fold} layer={layer}: "
                    f"combined mean disagrees with saved modality means "
                    f"(relative error={relative_error:.6f}). "
                    f"Existing SVD inputs may use a different definition."
                )

            combined_dirs = method_directions(
                combined_mean, combined_basis
            )

            text_dirs = method_directions(
                text_reference_mean,
                existing["text"]["vectors"][position],
            )
            audio_dirs = method_directions(
                audio_reference_mean,
                existing["audio"]["vectors"][position],
            )

            for method in METHODS:
                row = {
                    "model": model,
                    "component": component,
                    "fold": fold,
                    "layer": layer,
                    "method": method,
                    "text_audio_cosine": cosine(
                        text_dirs[method], audio_dirs[method]
                    ),
                    "text_combined_cosine": cosine(
                        text_dirs[method], combined_dirs[method]
                    ),
                    "audio_combined_cosine": cosine(
                        audio_dirs[method], combined_dirs[method]
                    ),
                }

                if method.startswith("svd_"):
                    k = int(method.split("_")[1])

                    text_basis = existing["text"]["vectors"][
                        position, :k
                    ]
                    audio_basis = existing["audio"]["vectors"][
                        position, :k
                    ]
                    current_basis = combined_basis[:k]

                    row["text_audio_subspace_overlap"] = (
                        subspace_overlap(text_basis, audio_basis)
                    )
                    row["text_combined_subspace_overlap"] = (
                        subspace_overlap(text_basis, current_basis)
                    )
                    row["audio_combined_subspace_overlap"] = (
                        subspace_overlap(audio_basis, current_basis)
                    )
                else:
                    row["text_audio_subspace_overlap"] = np.nan
                    row["text_combined_subspace_overlap"] = np.nan
                    row["audio_combined_subspace_overlap"] = np.nan

                all_comparisons.append(row)

            direction_matrix = torch.stack(
                [combined_dirs[name] for name in METHODS], dim=1
            )

            for modality, benign_states, harmful_states in (
                ("text", text_b, text_h),
                ("audio", audio_b, audio_h),
            ):
                benign_test = benign_states[
                    test_indices, layer, :
                ].to(device)

                harmful_test = harmful_states[
                    test_indices, layer, :
                ].to(device)

                benign_scores = (
                    benign_test @ direction_matrix
                ).cpu().numpy()

                harmful_scores = (
                    harmful_test @ direction_matrix
                ).cpu().numpy()

                for local_index, global_index in enumerate(test_indices):
                    topic, language, condition = text_metadata[global_index]

                    for method_index, method in enumerate(METHODS):
                        bs = float(benign_scores[local_index, method_index])
                        hs = float(harmful_scores[local_index, method_index])

                        fold_predictions.append({
                            "model": model,
                            "component": component,
                            "fold": fold,
                            "layer": layer,
                            "modality": modality,
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

                del benign_test, harmful_test

            saved_means.append(
                combined_mean.cpu().numpy().astype(np.float32)
            )
            saved_vectors.append(
                combined_basis.cpu().numpy().astype(np.float32)
            )
            saved_singular_values.append(
                singular_values.cpu().numpy().astype(np.float64)
            )

            del text_difference, audio_difference, combined_difference
            del direction_matrix

            if (position + 1) % 4 == 0 or position == len(layers) - 1:
                print(
                    f"{model} fold={fold} layers="
                    f"{position + 1}/{len(layers)}",
                    flush=True,
                )

        npz_path = (
            direction_root / f"fold_{fold}_directions.npz"
        )

        np.savez_compressed(
            npz_path,
            means=np.stack(saved_means),
            vectors=np.stack(saved_vectors),
            singular_values=np.stack(saved_singular_values),
            layer_boundaries=np.asarray(layers, dtype=np.int64),
            k_values=np.asarray(K_VALUES, dtype=np.int64),
        )

        print(f"SAVED {npz_path}", flush=True)

        fold_predictions = pd.DataFrame(fold_predictions)

        decision_tables = []

        for modality in ("text", "audio"):
            decisions = get_decisions(
                base, component, modality, fold
            ).copy()

            decisions["modality"] = modality
            decision_tables.append(decisions)

        decisions = pd.concat(
            decision_tables, ignore_index=True
        )

        join_keys = KEYS + ["modality"]

        if decisions.duplicated(join_keys).any():
            raise ValueError(
                f"{model}: duplicate decision keys in fold {fold}"
            )

        if (
            "benign_decision" in fold_predictions.columns
            or "harmful_decision" in fold_predictions.columns
        ):
            raise ValueError(
                "Decision fields already present before join"
            )

        fold_predictions = fold_predictions.merge(
            decisions,
            on=join_keys,
            how="left",
            validate="many_to_one",
            sort=False,
        )

        if fold_predictions[
            ["benign_decision", "harmful_decision"]
        ].isna().any().any():
            raise ValueError(
                f"{model}: missing behavioural decisions in fold {fold}"
            )

        expected_rows = (
            len(layers) * len(METHODS) * len(test_indices) * 2
        )

        if len(fold_predictions) != expected_rows:
            raise ValueError(
                f"{model}: decision join changed row count "
                f"({len(fold_predictions)} != {expected_rows})"
            )

        for field in ("benign_decision", "harmful_decision"):
            values = set(fold_predictions[field].unique())
            if not values.issubset({0, 1}):
                raise ValueError(
                    f"{model}: invalid {field} values: {values}"
                )

        print(
            f"{model}: fold {fold} decisions matched correctly "
            f"for {len(fold_predictions)} projection rows",
            flush=True,
        )

        save_csv(
            fold_predictions.to_dict("records"),
            projection_root / f"fold_{fold}_predictions.csv",
        )

        all_predictions.append(fold_predictions)

        print(f"{model}: FOLD {fold} COMPLETE", flush=True)

    all_predictions = pd.concat(
        all_predictions, ignore_index=True
    )

    save_csv(
        all_predictions.to_dict("records"),
        projection_root / "heldout_predictions.csv",
    )

    normalized = add_normalized_scores(all_predictions)

    save_csv(
        normalized.to_dict("records"),
        projection_root / "heldout_predictions_normalized.csv",
    )

    metric_rows = []

    for (modality, fold, layer, method), subset in normalized.groupby(
        ["modality", "fold", "layer", "method"],
        sort=True,
    ):
        auroc = roc_auc_score(
            np.r_[
                np.zeros(len(subset)),
                np.ones(len(subset)),
            ],
            np.r_[
                subset["benign_score"].to_numpy(),
                subset["harmful_score"].to_numpy(),
            ],
        )

        metric_rows.append({
            "model": model,
            "component": component,
            "modality": modality,
            "fold": fold,
            "layer": layer,
            "method": method,
            "n_pairs": len(subset),
            "auroc": auroc,
            "paired_accuracy": subset["paired_correct"].mean(),
        })

    save_csv(
        metric_rows,
        projection_root / "fold_metrics.csv",
    )

    text_rows = normalized.loc[
        normalized["modality"] == "text"
    ].drop(columns=["modality"])

    audio_rows = normalized.loc[
        normalized["modality"] == "audio"
    ].drop(columns=["modality"])

    join_keys = KEYS + ["method"]

    matched = text_rows.merge(
        audio_rows,
        on=join_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_text", "_audio"),
    )

    if not matched["_merge"].eq("both").all():
        raise ValueError(f"{model}: unmatched text/audio examples")

    matched = matched.drop(columns=["_merge"])

    matched["outcome"] = (
        matched["benign_decision_text"].astype(int).astype(str)
        + matched["benign_decision_audio"].astype(int).astype(str)
    )

    if not matched["outcome"].isin(OUTCOMES).all():
        raise ValueError("Invalid matched outcome")

    matched["score_diff"] = (
        matched["benign_z_audio"] - matched["benign_z_text"]
    )

    matched["separation_diff"] = (
        matched["separation_z_audio"]
        - matched["separation_z_text"]
    )

    matched["raw_score_diff"] = (
        matched["benign_score_audio"]
        - matched["benign_score_text"]
    )

    save_csv(
        matched.to_dict("records"),
        projection_root / "matched_predictions.csv",
    )

    matched_summary = aggregate_matched(matched)

    save_csv(
        matched_summary,
        projection_root / "matched_behaviour_summary.csv",
    )

    save_csv(
        all_comparisons,
        comparison_root / "direction_comparisons.csv",
    )

    summary = pd.DataFrame(all_comparisons)

    summary = (
        summary.groupby(["model", "component", "layer", "method"])
        .agg(
            text_audio_cosine_mean=("text_audio_cosine", "mean"),
            text_audio_cosine_std=("text_audio_cosine", "std"),
            text_combined_cosine_mean=("text_combined_cosine", "mean"),
            text_combined_cosine_std=("text_combined_cosine", "std"),
            audio_combined_cosine_mean=("audio_combined_cosine", "mean"),
            audio_combined_cosine_std=("audio_combined_cosine", "std"),
            text_audio_subspace_overlap_mean=(
                "text_audio_subspace_overlap", "mean"
            ),
            text_combined_subspace_overlap_mean=(
                "text_combined_subspace_overlap", "mean"
            ),
            audio_combined_subspace_overlap_mean=(
                "audio_combined_subspace_overlap", "mean"
            ),
        )
        .reset_index()
    )

    save_csv(
        summary.to_dict("records"),
        comparison_root / "direction_comparisons_fold_summary.csv",
    )

    metadata = {
        "model": model,
        "component": component,
        "folds": 5,
        "training_topics_per_fold": 80,
        "test_topics_per_fold": 20,
        "training_pairs_per_modality": 1280,
        "combined_training_difference_rows": 2560,
        "methods": METHODS,
        "combined_fit": "equal-weight stacked text/audio matched harmful-minus-benign differences",
        "svd_method": "torch.svd_lowrank, q=rank+8, niter=3",
        "normalization": "heldout benign mean and sample SD per fold, layer, method and modality",
        "behaviour_groups": "benign text/audio decisions",
    }

    metadata_path = projection_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{model}: ALL FIVE FOLDS COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODELS),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--rank", type=int, default=128)
    args = parser.parse_args()

    if args.rank != 128:
        raise ValueError("This experiment requires rank=128")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} unavailable")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)

    print(f"DEVICE: {torch.cuda.get_device_name(device)}", flush=True)

    module = load_existing_module()

    run_model(args.model, module, device, args.rank)


if __name__ == "__main__":
    main()
