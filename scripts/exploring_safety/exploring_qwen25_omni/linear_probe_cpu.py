#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

FOLD_ROOT = (
    ROOT
    / "results/INITIAL_BENCHMARK/qwen25_omni/5fold"
)

REP_ROOT = FOLD_ROOT / "representations_shared"

OUT_ROOT = (
    ROOT
    / "results/INITIAL_BENCHMARK/qwen25_omni/linear_probe_cpu"
)

WEIGHTS_ROOT = OUT_ROOT / "weights"

CONFIGS = (
    ("benign",  "text",  "thinker", 28),
    ("benign",  "audio", "thinker", 28),
    ("harmful", "text",  "thinker", 28),
    ("harmful", "audio", "thinker", 28),
    ("benign",  "audio", "encoder", 32),
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
        raise RuntimeError(f"Missing representation:\n{path}")

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    for key in ("task_id", "safety_split", "modality"):
        if str(obj.get(key)) != str(row[key]):
            raise RuntimeError(
                f"Metadata mismatch in {path}: "
                f"{key}: pt={obj.get(key)!r}, "
                f"fold={row[key]!r}"
            )

    if int(obj["decision"]) != int(row["decision"]):
        raise RuntimeError(
            f"Decision mismatch in {path}"
        )

    if family == "thinker":
        x = obj["thinker"]

        if tuple(x.shape) != (29, 3584):
            raise RuntimeError(
                f"Wrong thinker shape in {path}: "
                f"{tuple(x.shape)} != (29, 3584)"
            )

        v = x[layer].float().reshape(-1)

        if v.numel() != 3584:
            raise RuntimeError("Thinker dimension error")

    elif family == "encoder":
        if modality != "audio":
            raise RuntimeError(
                "Encoder probe requested for non-audio row"
            )

        x = obj["audio_encoder"]

        if tuple(x.shape) != (33, 8, 1280):
            raise RuntimeError(
                f"Wrong audio_encoder shape in {path}: "
                f"{tuple(x.shape)} != (33, 8, 1280)"
            )

        # 8 temporal bins × 1280 hidden dimensions.
        v = x[layer].float().reshape(-1)

        if v.numel() != 10240:
            raise RuntimeError("Encoder dimension error")

    else:
        raise RuntimeError(f"Unknown family: {family}")

    return v.numpy()


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
            f"Empty subset: {role}/{safety}/{modality}"
        )

    return out


def build_xy(rows, modality, family, layer):
    X = np.stack([
        load_vector(row, modality, family, layer)
        for row in rows
    ]).astype(np.float32)

    y = np.asarray(
        [int(row["decision"]) for row in rows],
        dtype=np.int64,
    )

    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError(
            f"Need both classes; got {np.unique(y)}"
        )

    return X, y


def evaluate(y, pred, prob):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy":
            float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, prob)),
    }


def save_weights(
    probe,
    fold,
    safety,
    modality,
    family,
    layer,
):
    dim = 3584 if family == "thinker" else 10240

    if probe.coef_.shape != (1, dim):
        raise RuntimeError(
            f"Wrong coefficient shape: {probe.coef_.shape}"
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

    torch.save(
        {
            "backend": "sklearn_cpu",
            "fold": fold,
            "safety_split": safety,
            "modality": modality,
            "family": family,
            "layer": layer,
            "feature_dim": dim,
            "w": torch.tensor(
                probe.coef_[0],
                dtype=torch.float32,
            ),
            "b": float(probe.intercept_[0]),
        },
        out,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fold",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
    )
    args = p.parse_args()

    folds = (
        [args.fold]
        if args.fold is not None
        else [1, 2, 3, 4, 5]
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    prediction_rows = []

    for fold in folds:
        rows = read_jsonl(
            FOLD_ROOT
            / f"fold_{fold}"
            / f"fold_{fold}.jsonl"
        )

        print(f"\n{'=' * 80}")
        print(f"FOLD {fold} -- CPU")
        print("=" * 80)

        for safety, modality, family, max_layer in CONFIGS:
            train_rows = subset(
                rows, "train", safety, modality
            )
            test_rows = subset(
                rows, "test", safety, modality
            )

            print(
                f"\n{safety:7s} {modality:5s} "
                f"{family:7s} "
                f"train={len(train_rows)} "
                f"test={len(test_rows)}"
            )

            for layer in range(max_layer + 1):
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

                probe = LogisticRegression(
                    penalty="l2",
                    C=args.C,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                )

                probe.fit(X_train, y_train)

                prob = probe.predict_proba(X_test)[:, 1]
                pred = (prob >= 0.5).astype(np.int64)

                metrics = evaluate(
                    y_test,
                    pred,
                    prob,
                )

                save_weights(
                    probe,
                    fold,
                    safety,
                    modality,
                    family,
                    layer,
                )

                metrics_rows.append({
                    "fold": fold,
                    "safety_split": safety,
                    "modality": modality,
                    "family": family,
                    "layer": layer,
                    "feature_dim": X_train.shape[1],
                    "train_n": len(train_rows),
                    "test_n": len(test_rows),
                    "train_0": int((y_train == 0).sum()),
                    "train_1": int((y_train == 1).sum()),
                    "test_0": int((y_test == 0).sum()),
                    "test_1": int((y_test == 1).sum()),
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
                        "task_id": row["task_id"],
                        "base_prompt_id":
                            row["base_prompt_id"],
                        "language": row["language"],
                        "condition": row["condition"],
                        "safety_split": safety,
                        "modality": modality,
                        "family": family,
                        "layer": layer,
                        "decision": int(truth),
                        "prob_decision_0":
                            float(1.0 - p1),
                        "prob_decision_1":
                            float(p1),
                        "prediction": int(yhat),
                    })

                print(
                    f"  layer {layer:02d} | "
                    f"dim={X_train.shape[1]:5d} | "
                    f"AUROC={metrics['auroc']:.3f} | "
                    f"bal_acc="
                    f"{metrics['balanced_accuracy']:.3f}",
                    flush=True,
                )

    metrics_path = OUT_ROOT / "metrics.csv"
    predictions_path = OUT_ROOT / "predictions.jsonl"

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(metrics_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in prediction_rows:
            f.write(json.dumps(row) + "\n")

    print("\nDONE")
    print("metrics    :", metrics_path)
    print("predictions:", predictions_path)
    print("weights    :", WEIGHTS_ROOT)


if __name__ == "__main__":
    main()
