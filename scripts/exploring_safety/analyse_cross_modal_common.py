#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
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
            },
        },
    },
    "qwen2_audio": {
        "families": {
            "decoder": {
                "key": "decoder",
                "layers": 33,
                "shape": (33, 4096),
            },
        },
    },
    "kimi_audio": {
        "families": {
            "shared_decoder": {
                "key": "shared_decoder",
                "layers": 29,
                "shape": (29, 3584),
            },
            "text_mimo": {
                "key": "text_mimo",
                "layers": 7,
                "shape": (7, 3584),
            },
        },
    },
}


def read_jsonl(path):
    rows = []

    if not path.exists():
        raise RuntimeError(f"Missing file: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def matched_key(row):
    return (
        row["base_prompt_id"],
        row["language"],
        row["condition"],
    )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run(model_name):
    spec = MODEL_SPECS[model_name]

    model_root = (
        ROOT
        / "results/INITIAL_BENCHMARK"
        / model_name
    )

    fold_root = model_root / "5fold"
    rep_root = fold_root / "representations_shared"

    probe_root = (
        model_root
        / "linear_probe_gpu"
    )

    weight_root = probe_root / "weights"

    out_root = (
        model_root
        / "cross_modal_analysis"
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    def load_rep(row, family):
        fs = spec["families"][family]

        path = (
            rep_root
            / f"{row['task_id']}.pt"
        )

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        x = obj.get(fs["key"])

        if not torch.is_tensor(x):
            raise RuntimeError(
                f"{path}: missing {fs['key']}"
            )

        if tuple(x.shape) != tuple(fs["shape"]):
            raise RuntimeError(
                f"{path}: {fs['key']} "
                f"{tuple(x.shape)} != "
                f"{fs['shape']}"
            )

        return x.float()

    def load_weight(
        fold,
        safety,
        modality,
        family,
        layer,
    ):
        path = (
            weight_root
            / f"fold_{fold}"
            / safety
            / modality
            / family
            / f"layer_{layer:02d}.pt"
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing probe weight: {path}"
            )

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        return (
            obj["w"].float().reshape(-1),
            float(obj["b"]),
        )

    transfer_rows = []
    disagreement_rows = {
        "benign": [],
        "harmful": [],
    }

    print("=" * 80)
    print("CROSS-MODAL PROBE ANALYSIS")
    print("=" * 80)
    print("model:", model_name)

    for family, fs in spec["families"].items():
        print()
        print("#" * 80)
        print("FAMILY:", family)
        print("#" * 80)

        # ====================================================
        # 1. Cross-modal transfer
        # ====================================================

        for fold in range(1, 6):
            rows = read_jsonl(
                fold_root
                / f"fold_{fold}.jsonl"
            )

            test_rows = [
                r for r in rows
                if r["train_test"] == "test"
            ]

            for safety in (
                "benign",
                "harmful",
            ):
                for source, target in (
                    ("text", "audio"),
                    ("audio", "text"),
                ):
                    target_rows = [
                        r for r in test_rows
                        if (
                            r["safety_split"]
                            == safety
                            and r["modality"]
                            == target
                        )
                    ]

                    for layer in range(
                        fs["layers"]
                    ):
                        w, b = load_weight(
                            fold,
                            safety,
                            source,
                            family,
                            layer,
                        )

                        labels = []
                        logits = []

                        for row in target_rows:
                            h = (
                                load_rep(
                                    row,
                                    family,
                                )[layer]
                                .reshape(-1)
                            )

                            logits.append(
                                float(
                                    torch.dot(
                                        w,
                                        h,
                                    )
                                    + b
                                )
                            )

                            labels.append(
                                int(row["decision"])
                            )

                        labels = np.asarray(
                            labels,
                            dtype=np.int64,
                        )

                        logits = np.asarray(
                            logits,
                            dtype=np.float64,
                        )

                        probs = sigmoid(logits)

                        preds = (
                            probs >= 0.5
                        ).astype(np.int64)

                        auroc = float(
                            roc_auc_score(
                                labels,
                                probs,
                            )
                        )

                        bal = float(
                            balanced_accuracy_score(
                                labels,
                                preds,
                            )
                        )

                        transfer_rows.append({
                            "model": model_name,
                            "family": family,
                            "fold": fold,
                            "safety_split": safety,
                            "source_probe_modality":
                                source,
                            "target_representation_modality":
                                target,
                            "layer": layer,
                            "n": len(labels),
                            "auroc": auroc,
                            "balanced_accuracy": bal,
                        })

                        print(
                            f"transfer fold={fold} "
                            f"{safety:7s} "
                            f"{source}->{target} "
                            f"{family} "
                            f"L{layer:02d} "
                            f"AUROC={auroc:.3f} "
                            f"bal={bal:.3f}",
                            flush=True,
                        )

        # ====================================================
        # CHECKPOINT: write cross-modal transfer now
        # ====================================================

        transfer_path = (
            out_root
            / "cross_modal_transfer.csv"
        )

        with transfer_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(
                    transfer_rows[0].keys()
                ),
            )
            w.writeheader()
            w.writerows(transfer_rows)

        print(
            f"[WROTE TRANSFER] {transfer_path} "
            f"rows={len(transfer_rows)}",
            flush=True,
        )

        # ====================================================
        # 2. Matched benign/harmful text-audio trajectories
        # ====================================================

        for safety in ("benign", "harmful"):
            for fold in range(1, 6):
                rows = read_jsonl(
                    fold_root / f"fold_{fold}.jsonl"
                )

                heldout = [
                    r for r in rows
                    if (
                        r["train_test"] == "test"
                        and r["safety_split"] == safety
                    )
                ]

                text = {
                    matched_key(r): r
                    for r in heldout
                    if r["modality"] == "text"
                }

                audio = {
                    matched_key(r): r
                    for r in heldout
                    if r["modality"] == "audio"
                }

                common = sorted(set(text) & set(audio))

                pairs = []

                for key in common:
                    tr = text[key]
                    ar = audio[key]

                    td = int(tr["decision"])
                    ad = int(ar["decision"])

                    if td == 0 and ad == 0:
                        group = "text0_audio0"
                    elif td == 0 and ad == 1:
                        group = "text0_audio1"
                    elif td == 1 and ad == 0:
                        group = "text1_audio0"
                    elif td == 1 and ad == 1:
                        group = "text1_audio1"
                    else:
                        raise RuntimeError(
                            f"Unexpected decisions: text={td}, audio={ad}"
                        )

                    pairs.append((group, tr, ar))

                cache = {}

                for _, tr, ar in pairs:
                    cache[tr["task_id"]] = load_rep(tr, family)
                    cache[ar["task_id"]] = load_rep(ar, family)

                print(
                    f"{safety} fold {fold} family={family} "
                    f"pairs={len(pairs)}"
                )

                for probe_modality in ("text", "audio"):
                    for layer in range(fs["layers"]):
                        w, b = load_weight(
                            fold,
                            safety,
                            probe_modality,
                            family,
                            layer,
                        )

                        grouped = {
                            "text0_audio0": [],
                            "text0_audio1": [],
                            "text1_audio0": [],
                            "text1_audio1": [],
                        }

                        for group, tr, ar in pairs:
                            ht = cache[tr["task_id"]][layer].reshape(-1)
                            ha = cache[ar["task_id"]][layer].reshape(-1)

                            text_logit = float(torch.dot(w, ht) + b)
                            audio_logit = float(torch.dot(w, ha) + b)

                            grouped[group].append(
                                (
                                    text_logit,
                                    audio_logit,
                                    audio_logit - text_logit,
                                )
                            )

                        for group, vals in grouped.items():
                            if not vals:
                                continue

                            arr = np.asarray(vals, dtype=np.float64)
                            d = arr[:, 2]

                            disagreement_rows[safety].append({
                                "model": model_name,
                                "family": family,
                                "fold": fold,
                                "safety_split": safety,
                                "probe_modality": probe_modality,
                                "layer": layer,
                                "disagreement_group": group,
                                "n": len(vals),
                                "mean_text_logit":
                                    float(arr[:, 0].mean()),
                                "mean_audio_logit":
                                    float(arr[:, 1].mean()),
                                "mean_audio_minus_text":
                                    float(d.mean()),
                                "median_audio_minus_text":
                                    float(np.median(d)),
                                "std_audio_minus_text":
                                    (
                                        float(d.std(ddof=1))
                                        if len(d) > 1
                                        else 0.0
                                    ),
                            })

    transfer_path = (
        out_root
        / "cross_modal_transfer.csv"
    )

    benign_path = (
        out_root
        / "benign_disagreement_trajectory.csv"
    )

    disagree_path = (
        out_root
        / "harmful_disagreement_trajectory.csv"
    )

    with transfer_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                transfer_rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(transfer_rows)

    for safety, path in (
        ("benign", benign_path),
        ("harmful", disagree_path),
    ):
        rows = disagreement_rows[safety]

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            w.writeheader()
            w.writerows(rows)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("transfer :", transfer_path)
    print("benign   :", benign_path)
    print("harmful  :", disagree_path)


if __name__ == "__main__":
    raise RuntimeError(
        "Run a model-specific analyse_cross_modal.py wrapper"
    )
