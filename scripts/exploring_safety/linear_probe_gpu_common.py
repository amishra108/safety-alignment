#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

MODEL_SPECS = {
    "qwen25_omni": {
        "families": {
            "thinker": {
                "key": "thinker",
                "layers": 29,
                "shape": (29, 3584),
                "modalities": ("text", "audio"),
                "flatten": False,
            },
            "encoder": {
                "key": "audio_encoder",
                "layers": 33,
                "shape": (33, 8, 1280),
                "modalities": ("audio",),
                "flatten": True,
            },
        },
    },
    "qwen2_audio": {
        "families": {
            "decoder": {
                "key": "decoder",
                "layers": 33,
                "shape": (33, 4096),
                "modalities": ("text", "audio"),
                "flatten": False,
            },
            "encoder": {
                "key": "audio_encoder",
                "layers": 33,
                "shape": (33, 8, 1280),
                "modalities": ("audio",),
                "flatten": True,
            },
        },
    },
    "kimi_audio": {
        "families": {
            "shared_decoder": {
                "key": "shared_decoder",
                "layers": 29,
                "shape": (29, 3584),
                "modalities": ("text", "audio"),
                "flatten": False,
            },
            "text_mimo": {
                "key": "text_mimo",
                "layers": 7,
                "shape": (7, 3584),
                "modalities": ("text", "audio"),
                "flatten": False,
            },
            "encoder": {
                "key": "audio_encoder",
                "layers": 33,
                "shape": (33, 8, 1280),
                "modalities": ("audio",),
                "flatten": True,
            },
        },
    },
}


def read_jsonl(path):
    if not path.exists():
        raise RuntimeError(f"Missing file: {path}")

    rows = []

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


def subset(rows, role, safety, modality):
    out = [
        r for r in rows
        if r["train_test"] == role
        and r["safety_split"] == safety
        and r["modality"] == modality
    ]

    if not out:
        raise RuntimeError(
            f"Empty subset: {role}/{safety}/{modality}"
        )

    return out


def load_vector(rep_root, row, spec, layer):
    path = rep_root / f"{row['task_id']}.pt"

    if not path.exists():
        raise RuntimeError(f"Missing representation: {path}")

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    for key in ("task_id", "safety_split", "modality"):
        if str(obj.get(key)) != str(row[key]):
            raise RuntimeError(
                f"{path}: metadata mismatch {key}: "
                f"{obj.get(key)!r} != {row[key]!r}"
            )

    x = obj.get(spec["key"])

    if not torch.is_tensor(x):
        raise RuntimeError(
            f"{path}: missing tensor {spec['key']}"
        )

    if tuple(x.shape) != tuple(spec["shape"]):
        raise RuntimeError(
            f"{path}: {spec['key']} shape "
            f"{tuple(x.shape)} != {spec['shape']}"
        )

    v = x[layer].float()

    if spec["flatten"]:
        v = v.reshape(-1)
    else:
        v = v.reshape(-1)

    return v


def build_xy(rep_root, rows, spec, layer):
    X = torch.stack([
        load_vector(
            rep_root,
            row,
            spec,
            layer,
        )
        for row in rows
    ]).float()

    y = torch.tensor(
        [int(row["decision"]) for row in rows],
        dtype=torch.float32,
    )

    classes = set(
        int(x)
        for x in torch.unique(y).tolist()
    )

    if classes != {0, 1}:
        raise RuntimeError(
            f"Need both decision classes; got {classes}"
        )

    return X, y


def evaluate(y, pred, prob):
    return {
        "accuracy":
            float(accuracy_score(y, pred)),
        "balanced_accuracy":
            float(balanced_accuracy_score(y, pred)),
        "auroc":
            float(roc_auc_score(y, prob)),
    }


def fit_probe(X_train, y_train, device, l2, max_iter):
    X = X_train.to(
        device,
        non_blocking=True,
    )

    y = y_train.to(
        device,
        non_blocking=True,
    )

    n, d = X.shape

    # TRAIN statistics only.
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False)

    std = torch.where(
        std < 1e-6,
        torch.ones_like(std),
        std,
    )

    Xn = (X - mean) / std

    n0 = (y == 0).sum().float()
    n1 = (y == 1).sum().float()

    # Equal total weight for both classes.
    weights = torch.where(
        y == 1,
        0.5 / n1,
        0.5 / n0,
    ) * n

    w = torch.zeros(
        d,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    b = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)

        logits = Xn @ w + b

        per_example = (
            F.binary_cross_entropy_with_logits(
                logits,
                y,
                reduction="none",
            )
        )

        loss = (
            (per_example * weights).mean()
            + 0.5 * l2 * w.square().sum()
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    # Convert back to original activation coordinates so
    # the saved classifier can be applied directly to h.
    with torch.no_grad():
        w_raw = w / std
        b_raw = b - (mean / std * w).sum()

    return (
        w_raw.detach(),
        b_raw.detach(),
    )


@torch.no_grad()
def predict_probe(X, w, b, device):
    X = X.to(
        device,
        non_blocking=True,
    )

    logits = X @ w + b

    return (
        torch.sigmoid(logits)
        .cpu()
        .numpy()
    )


def save_weights(
    root,
    model_name,
    w,
    b,
    fold,
    safety,
    modality,
    family,
    layer,
    l2,
):
    out = (
        root
        / f"fold_{fold}"
        / safety
        / modality
        / family
        / f"layer_{layer:02d}.pt"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = out.with_name(
        out.name + ".tmp"
    )

    torch.save(
        {
            "backend": "pytorch_cuda",
            "model": model_name,
            "fold": fold,
            "safety_split": safety,
            "modality": modality,
            "family": family,
            "layer": layer,
            "feature_dim": int(w.numel()),
            "w": w.detach().cpu().float(),
            "b": float(b.detach().cpu()),
            "l2": float(l2),
            "classes": [0, 1],
        },
        tmp,
    )

    tmp.replace(out)


def run(model_name):
    if model_name not in MODEL_SPECS:
        raise RuntimeError(model_name)

    p = argparse.ArgumentParser()

    p.add_argument(
        "--fold",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
    )
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--max-iter", type=int, default=250)

    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda:0")

    model_root = (
        ROOT
        / "results/INITIAL_BENCHMARK"
        / model_name
    )

    fold_root = model_root / "5fold"
    rep_root = fold_root / "representations_shared"
    out_root = model_root / "linear_probe_gpu"
    weights_root = out_root / "weights"

    spec = MODEL_SPECS[model_name]

    configs = []

    for family, family_spec in spec["families"].items():
        for safety in ("benign", "harmful"):
            for modality in family_spec["modalities"]:
                configs.append(
                    (
                        safety,
                        modality,
                        family,
                        family_spec,
                    )
                )

    folds = (
        [args.fold]
        if args.fold is not None
        else [1, 2, 3, 4, 5]
    )

    print("=" * 80)
    print("GPU LINEAR PROBE")
    print("=" * 80)
    print("model  :", model_name)
    print("device :", torch.cuda.get_device_name(0))
    print("l2     :", args.l2)
    print("reps   :", rep_root)
    print("=" * 80)

    metric_rows = []
    prediction_rows = []

    for fold in folds:
        rows = read_jsonl(
            fold_root / f"fold_{fold}.jsonl"
        )

        print(
            f"\n{'=' * 80}\n"
            f"FOLD {fold}\n"
            f"{'=' * 80}"
        )

        for (
            safety,
            modality,
            family,
            family_spec,
        ) in configs:

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

            print(
                f"\n{safety:7s} {modality:5s} "
                f"{family:14s} "
                f"train={len(train_rows)} "
                f"test={len(test_rows)}"
            )

            for layer in range(
                family_spec["layers"]
            ):
                X_train, y_train = build_xy(
                    rep_root,
                    train_rows,
                    family_spec,
                    layer,
                )

                X_test, y_test = build_xy(
                    rep_root,
                    test_rows,
                    family_spec,
                    layer,
                )

                w, b = fit_probe(
                    X_train,
                    y_train,
                    device,
                    args.l2,
                    args.max_iter,
                )

                prob = predict_probe(
                    X_test,
                    w,
                    b,
                    device,
                )

                pred = (
                    prob >= 0.5
                ).astype(np.int64)

                y_np = (
                    y_test.numpy()
                    .astype(np.int64)
                )

                metrics = evaluate(
                    y_np,
                    pred,
                    prob,
                )

                save_weights(
                    weights_root,
                    model_name,
                    w,
                    b,
                    fold,
                    safety,
                    modality,
                    family,
                    layer,
                    args.l2,
                )

                metric_rows.append({
                    "model": model_name,
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

                for (
                    row,
                    truth,
                    p1,
                    yhat,
                ) in zip(
                    test_rows,
                    y_np,
                    prob,
                    pred,
                ):
                    prediction_rows.append({
                        "model": model_name,
                        "fold": fold,
                        "task_id": row["task_id"],
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
                    f"dim={X_train.shape[1]:5d} | "
                    f"AUROC={metrics['auroc']:.3f} | "
                    f"bal_acc="
                    f"{metrics['balanced_accuracy']:.3f}",
                    flush=True,
                )

                del (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    w,
                    b,
                )

    result_root = (
        out_root / f"fold_{args.fold}"
        if args.fold is not None
        else out_root
    )

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        result_root / "metrics.csv"
    )

    predictions_path = (
        result_root / "predictions.jsonl"
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
        writer.writerows(metric_rows)

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
    print("weights    :", weights_root)
