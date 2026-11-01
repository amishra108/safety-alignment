#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

FOLD_ROOT = (
    ROOT
    / "results/INITIAL_BENCHMARK/qwen25_omni/5fold"
)

REP_ROOT = (
    ROOT
    / "results/INITIAL_BENCHMARK/qwen25_omni/5fold"
    / "representations_shared"
)

OUT_ROOT = (
    ROOT
    / "results/INITIAL_BENCHMARK/qwen25_omni/linear_probe"
)

WEIGHTS_ROOT = OUT_ROOT / "weights"

CONFIGS = (
    ("benign", "text",  "thinker", 28),
    ("benign", "audio", "thinker", 28),
    ("harmful", "text",  "thinker", 28),
    ("harmful", "audio", "thinker", 28),
    ("benign", "audio", "encoder", 32),
    ("harmful", "audio", "encoder", 32),
)


def read_jsonl(path):
    rows = []

    if not path.exists():
        raise RuntimeError(f"Missing fold file: {path}")

    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSONL {path}:{ln}"
                ) from exc

    return rows


def rep_path(row):
    return REP_ROOT / f"{row['task_id']}.pt"


def load_vector(row, modality, family, layer):
    path = rep_path(row)

    if not path.exists():
        raise RuntimeError(
            f"Missing representation:\n{path}"
        )

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    # Strong identity checks so a wrong .pt can never silently enter
    # a train/test fold.
    for key in (
        "task_id",
        "safety_split",
        "modality",
    ):
        if str(obj.get(key)) != str(row[key]):
            raise RuntimeError(
                f"Metadata mismatch in {path}: "
                f"{key}: pt={obj.get(key)!r}, "
                f"fold={row[key]!r}"
            )

    if int(obj["decision"]) != int(row["decision"]):
        raise RuntimeError(
            f"Decision mismatch in {path}: "
            f"pt={obj['decision']}, "
            f"fold={row['decision']}"
        )

    if modality == "text":
        if family != "thinker":
            raise RuntimeError(
                f"Text has no {family} representation: {path}"
            )
        key = "thinker"
        expected_layers = 29
        expected_dim = 3584

    elif modality == "audio":
        if family == "thinker":
            key = "thinker"
            expected_layers = 29
            expected_dim = 3584
        elif family == "encoder":
            key = "audio_encoder"
            expected_layers = 33
            expected_dim = 1280
        else:
            raise RuntimeError(
                f"Unknown audio family: {family}"
            )

    else:
        raise RuntimeError(
            f"Unknown modality: {modality}"
        )

    if key not in obj:
        raise RuntimeError(
            f"Missing {key!r} in {path}"
        )

    tensor = obj[key]

    if not torch.is_tensor(tensor):
        raise RuntimeError(
            f"{key!r} in {path} is not a tensor"
        )

    if tuple(tensor.shape) != (
        expected_layers,
        expected_dim,
    ):
        raise RuntimeError(
            f"Wrong {key} shape in {path}: "
            f"{tuple(tensor.shape)} != "
            f"{(expected_layers, expected_dim)}"
        )

    if not 0 <= layer < expected_layers:
        raise RuntimeError(
            f"Layer {layer} out of range for {key}"
        )

    return (
        tensor[layer]
        .float()
        .contiguous()
        .numpy()
    )


def subset(rows, role, safety, modality):
    out = [
        row
        for row in rows
        if (
            row["train_test"] == role
            and row["safety_split"] == safety
            and row["modality"] == modality
        )
    ]

    if not out:
        raise RuntimeError(
            f"Empty subset: "
            f"{role}/{safety}/{modality}"
        )

    return out


def build_xy(rows, modality, family, layer):
    X = np.stack([
        load_vector(
            row,
            modality,
            family,
            layer,
        )
        for row in rows
    ])

    y = np.asarray(
        [
            int(row["decision"])
            for row in rows
        ],
        dtype=np.int64,
    )

    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError(
            f"Need both decision classes; got "
            f"{sorted(np.unique(y).tolist())}"
        )

    return X, y


def save_probe_weights(
    probe,
    fold,
    safety,
    modality,
    family,
    layer,
    feature_dim,
    C,
):
    expected_dim = 3584 if family == "thinker" else 1280

    if feature_dim != expected_dim:
        raise RuntimeError(
            f"Unexpected feature dimension for {family}: "
            f"{feature_dim} != {expected_dim}"
        )

    if probe.coef_.shape != (1, expected_dim):
        raise RuntimeError(
            f"Unexpected coef shape: {probe.coef_.shape}"
        )

    if probe.intercept_.shape != (1,):
        raise RuntimeError(
            f"Unexpected intercept shape: {probe.intercept_.shape}"
        )

    out = (
        WEIGHTS_ROOT
        / f"fold_{fold}"
        / safety
        / modality
        / family
        / f"layer_{layer:02d}.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "fold": int(fold),
        "safety_split": safety,
        "modality": modality,
        "family": family,
        "layer": int(layer),
        "feature_dim": int(expected_dim),

        # Logistic-regression decision function:
        # z = w^T h + b
        # P(decision=1) = sigmoid(z)
        "w": torch.from_numpy(
            probe.coef_[0]
            .astype(np.float32, copy=True)
        ),
        "b": float(probe.intercept_[0]),

        "classes": [
            int(x) for x in probe.classes_.tolist()
        ],
        "C": float(C),
        "penalty": "l2",
        "class_weight": "balanced",
        "solver": "liblinear",
    }

    tmp = out.with_name(out.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def evaluate(y, pred, prob):
    return {
        "accuracy":
            float(accuracy_score(y, pred)),
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
                )
            ),
        "auroc":
            float(
                roc_auc_score(
                    y,
                    prob,
                )
            ),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--fold",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
        help="Run one fold only; default runs all five",
    )

    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength",
    )

    args = p.parse_args()

    folds = (
        [args.fold]
        if args.fold is not None
        else [1, 2, 3, 4, 5]
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    WEIGHTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_rows = []
    prediction_rows = []

    for fold in folds:
        fold_path = (
            FOLD_ROOT
            / f"fold_{fold}"
            / f"fold_{fold}.jsonl"
        )

        rows = read_jsonl(
            fold_path
        )

        print()
        print("=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)

        for (
            safety,
            modality,
            family,
            max_layer,
        ) in CONFIGS:

            train_rows = subset(
                rows,
                "train",
                safety,
                modality,
            )

            test_rows = subset(
                rows,
                "test",
                safety,
                modality,
            )

            print()
            print(
                f"{safety:7s} "
                f"{modality:5s} "
                f"{family:7s} "
                f"train={len(train_rows)} "
                f"test={len(test_rows)}"
            )

            for layer in range(
                max_layer + 1
            ):
                X_train, y_train = build_xy(
                    train_rows,
                    modality,
                    family,
                    layer,
                )

                X_test, y_test = build_xy(
                    test_rows,
                    modality,
                    family,
                    layer,
                )

                # L2-regularized linear probe.
                probe = LogisticRegression(
                    penalty="l2",
                    C=args.C,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                )

                probe.fit(
                    X_train,
                    y_train,
                )

                save_probe_weights(
                    probe=probe,
                    fold=fold,
                    safety=safety,
                    modality=modality,
                    family=family,
                    layer=layer,
                    feature_dim=X_train.shape[1],
                    C=args.C,
                )

                prob = probe.predict_proba(
                    X_test
                )[:, 1]

                pred = (
                    prob >= 0.5
                ).astype(np.int64)

                metrics = evaluate(
                    y_test,
                    pred,
                    prob,
                )

                metric_rows.append({
                    "fold": fold,
                    "safety_split": safety,
                    "modality": modality,
                    "family": family,
                    "layer": layer,
                    "train_n":
                        len(train_rows),
                    "test_n":
                        len(test_rows),
                    "train_0":
                        int((y_train == 0).sum()),
                    "train_1":
                        int((y_train == 1).sum()),
                    "test_0":
                        int((y_test == 0).sum()),
                    "test_1":
                        int((y_test == 1).sum()),
                    **metrics,
                })

                for row, truth, p1, yhat in zip(
                    test_rows,
                    y_test,
                    prob,
                    pred,
                ):
                    prediction_rows.append({
                        "fold": fold,
                        "task_id":
                            row["task_id"],
                        "base_prompt_id":
                            row["base_prompt_id"],
                        "language":
                            row["language"],
                        "condition":
                            row["condition"],
                        "safety_split":
                            safety,
                        "modality":
                            modality,
                        "family":
                            family,
                        "layer":
                            layer,
                        "decision":
                            int(truth),
                        "prob_decision_0":
                            float(1.0 - p1),
                        "prob_decision_1":
                            float(p1),
                        "prediction":
                            int(yhat),
                    })

                print(
                    f"  layer {layer:02d} | "
                    f"AUROC={metrics['auroc']:.3f} | "
                    f"bal_acc="
                    f"{metrics['balanced_accuracy']:.3f}",
                    flush=True,
                )

    metrics_path = (
        OUT_ROOT
        / "metrics.csv"
    )

    predictions_path = (
        OUT_ROOT
        / "predictions.jsonl"
    )

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                metric_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            metric_rows
        )

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in prediction_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("metrics    :", metrics_path)
    print("predictions:", predictions_path)
    print("weights    :", WEIGHTS_ROOT)


if __name__ == "__main__":
    main()
