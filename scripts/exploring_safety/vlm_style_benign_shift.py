#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

SPECS = {
    "qwen25_omni": {
        "families": {
            "thinker": (29, 3584),
        },
    },
    "qwen2_audio": {
        "families": {
            "decoder": (33, 4096),
        },
    },
    "kimi_audio": {
        "families": {
            "shared_decoder": (29, 3584),
            "text_mimo": (7, 3584),
        },
    },
}


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Malformed JSONL {path}:{ln}") from e
    return rows


def pair_key(r):
    return (
        r["base_prompt_id"],
        r["language"],
        r["condition"],
    )


def make_pairs(rows, role):
    use = [
        r for r in rows
        if r["train_test"] == role
        and r["safety_split"] == "benign"
    ]

    text = {
        pair_key(r): r
        for r in use
        if r["modality"] == "text"
    }
    audio = {
        pair_key(r): r
        for r in use
        if r["modality"] == "audio"
    }

    if set(text) != set(audio):
        raise RuntimeError(
            f"{role}: text/audio pair mismatch: "
            f"text_only={len(set(text)-set(audio))}, "
            f"audio_only={len(set(audio)-set(text))}"
        )

    return [(k, text[k], audio[k]) for k in sorted(text)]


def summarize(vals):
    a = np.asarray(vals, dtype=np.float64)

    if len(a) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }

    return {
        "n": len(a),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows for {path}")

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tmp.replace(path)


def run_model(model, device):

    spec = SPECS[model]

    model_root = ROOT / "results/INITIAL_BENCHMARK" / model
    fold_root = model_root / "5fold"
    rep_root = fold_root / "representations_shared"
    out_root = model_root / "cross_modal_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    per_example_csv = (
        out_root / "VLM_STYLE_BENIGN_independent_direction_per_example.csv"
    )
    by_fold_csv = (
        out_root / "VLM_STYLE_BENIGN_independent_direction_by_fold.csv"
    )
    oof_csv = (
        out_root / "VLM_STYLE_BENIGN_independent_direction_OOF.csv"
    )
    contrast_csv = (
        out_root / "VLM_STYLE_BENIGN_independent_direction_contrasts.csv"
    )

    print()
    print("=" * 100, flush=True)
    print("VLM-STYLE INDEPENDENT BENIGN-COMPLIANCE-DIRECTION ANALYSIS", flush=True)
    print("model     :", model, flush=True)
    print("families  :", ", ".join(spec["families"]), flush=True)
    print("direction : TRAIN benign TEXT mean(y=1)-mean(y=0)", flush=True)
    print("offset    : TRAIN matched audio-text by language x condition", flush=True)
    print("test      : held-out benign matched pairs", flush=True)
    print("=" * 100, flush=True)

    fold_rows = {}

    for fold in range(1, 6):
        p = fold_root / f"fold_{fold}.jsonl"
        rows = read_jsonl(p)
        fold_rows[fold] = rows

        train_pairs = make_pairs(rows, "train")
        test_pairs = make_pairs(rows, "test")

        print(
            f"[FOLD {fold}] "
            f"train_pairs={len(train_pairs)} "
            f"test_pairs={len(test_pairs)}",
            flush=True,
        )

    # Every fold file contains the full benchmark, merely with different
    # train_test assignments, so fold 1 is sufficient to enumerate task IDs.
    harmful_rows = [
        r for r in fold_rows[1]
        if r["safety_split"] == "benign"
    ]

    task_ids = sorted({r["task_id"] for r in harmful_rows})

    print(
        f"\n[CACHE] loading {len(task_ids)} benign representation files...",
        flush=True,
    )

    # task_id -> family -> [layer, dim], kept float32 on CPU.
    cache = {}

    for i, task_id in enumerate(task_ids, 1):
        p = rep_root / f"{task_id}.pt"

        if not p.exists():
            raise RuntimeError(f"Missing representation: {p}")

        obj = torch.load(
            p,
            map_location="cpu",
            weights_only=True,
        )

        fam_cache = {}

        for family, (n_layers, dim) in spec["families"].items():

            if family not in obj:
                raise RuntimeError(
                    f"{task_id}: missing family {family}; "
                    f"keys={list(obj.keys())}"
                )

            x = obj[family]

            if tuple(x.shape) != (n_layers, dim):
                raise RuntimeError(
                    f"{task_id} {family}: "
                    f"shape={tuple(x.shape)} expected={(n_layers, dim)}"
                )

            fam_cache[family] = x.float().contiguous()

        cache[task_id] = fam_cache

        if i % 250 == 0 or i == len(task_ids):
            print(f"  cached {i}/{len(task_ids)}", flush=True)

    print("[CACHE] done\n", flush=True)

    per_example_rows = []
    by_fold_rows = []

    for family, (n_layers, dim) in spec["families"].items():

        print()
        print("#" * 100, flush=True)
        print(
            f"FAMILY {family} | layers={n_layers} dim={dim}",
            flush=True,
        )
        print("#" * 100, flush=True)

        for fold in range(1, 6):

            rows = fold_rows[fold]
            train_pairs = make_pairs(rows, "train")
            test_pairs = make_pairs(rows, "test")

            train_text_rows = [
                r for r in rows
                if r["train_test"] == "train"
                and r["safety_split"] == "benign"
                and r["modality"] == "text"
            ]

            n0 = sum(int(r["decision"]) == 0 for r in train_text_rows)
            n1 = sum(int(r["decision"]) == 1 for r in train_text_rows)

            if not n0 or not n1:
                raise RuntimeError(
                    f"{model} fold {fold}: text classes n0={n0} n1={n1}"
                )

            print()
            print(
                f"[{family}] FOLD {fold} | "
                f"text train y0={n0} y1={n1}",
                flush=True,
            )

            for layer in range(n_layers):

                # ==========================================================
                # 1. Independent jailbreak direction:
                #    benign TRAIN TEXT only.
                # ==========================================================

                text0 = torch.stack([
                    cache[r["task_id"]][family][layer]
                    for r in train_text_rows
                    if int(r["decision"]) == 0
                ]).to(device)

                text1 = torch.stack([
                    cache[r["task_id"]][family][layer]
                    for r in train_text_rows
                    if int(r["decision"]) == 1
                ]).to(device)

                mu0 = text0.mean(0)
                mu1 = text1.mean(0)

                v = mu1 - mu0
                v_norm = torch.linalg.vector_norm(v)

                if (
                    not torch.isfinite(v_norm)
                    or v_norm.item() <= 1e-12
                ):
                    print(
                        f"[SKIP] {family} fold={fold} "
                        f"layer={layer:02d} |v|={v_norm.item()}",
                        flush=True,
                    )
                    del text0, text1, mu0, mu1, v
                    continue

                v = v / v_norm

                del text0, text1, mu0, mu1

                # ==========================================================
                # 2. Train-only generic modality offset.
                #    No decision labels enter this estimate.
                # ==========================================================

                offset_values = defaultdict(list)

                for _, tr, ar in train_pairs:
                    ht = cache[tr["task_id"]][family][layer]
                    ha = cache[ar["task_id"]][family][layer]

                    offset_values[
                        (tr["language"], tr["condition"])
                    ].append(ha - ht)

                offsets = {
                    stratum: torch.stack(vals).mean(0).to(device)
                    for stratum, vals in offset_values.items()
                }

                # ==========================================================
                # 3. Held-out modality-induced shift.
                #    TEST audio decision is used only AFTER projection.
                # ==========================================================

                layer_records = []

                for _, tr, ar in test_pairs:

                    stratum = (tr["language"], tr["condition"])

                    if stratum not in offsets:
                        raise RuntimeError(
                            f"No offset fold={fold} "
                            f"family={family} stratum={stratum}"
                        )

                    ht = (
                        cache[tr["task_id"]][family][layer]
                        .to(device)
                    )
                    ha = (
                        cache[ar["task_id"]][family][layer]
                        .to(device)
                    )

                    centred_shift = (
                        (ha - ht)
                        - offsets[stratum]
                    )

                    projection = float(
                        torch.dot(centred_shift, v).item()
                    )

                    td = int(tr["decision"])
                    ad = int(ar["decision"])

                    rec = {
                        "model": model,
                        "family": family,
                        "fold": fold,
                        "layer": layer,
                        "base_prompt_id": tr["base_prompt_id"],
                        "language": tr["language"],
                        "condition": tr["condition"],
                        "text_decision": td,
                        "audio_decision": ad,
                        "outcome_group": f"{td}{ad}",
                        "projection_toward_text_jailbreak_direction":
                            projection,
                    }

                    per_example_rows.append(rec)
                    layer_records.append(rec)

                for group in ("00", "01", "10", "11"):
                    vals = [
                        r[
                            "projection_toward_text_jailbreak_direction"
                        ]
                        for r in layer_records
                        if r["outcome_group"] == group
                    ]

                    s = summarize(vals)

                    by_fold_rows.append({
                        "model": model,
                        "family": family,
                        "fold": fold,
                        "layer": layer,
                        "outcome_group": group,
                        "n": s["n"],
                        "mean_projection": s["mean"],
                        "median_projection": s["median"],
                        "std_projection": s["std"],
                    })

                r0 = [
                    r for r in layer_records
                    if r["text_decision"] == 0
                ]
                r1 = [
                    r for r in layer_records
                    if r["text_decision"] == 1
                ]

                def auc(rr):
                    y = [r["audio_decision"] for r in rr]
                    if len(set(y)) != 2:
                        return float("nan")
                    x = [
                        r[
                            "projection_toward_text_jailbreak_direction"
                        ]
                        for r in rr
                    ]
                    return float(roc_auc_score(y, x))

                print(
                    f"[{model}/{family}] "
                    f"fold={fold} layer={layer:02d} "
                    f"AUC|text0={auc(r0):.3f} "
                    f"AUC|text1={auc(r1):.3f}",
                    flush=True,
                )

                del v, offsets
                torch.cuda.empty_cache()

    # ==============================================================
    # Combined out-of-fold summaries.
    # ==============================================================

    oof_rows = []
    contrast_rows = []

    for family, (n_layers, _) in spec["families"].items():

        for layer in range(n_layers):

            lr = [
                r for r in per_example_rows
                if r["family"] == family
                and r["layer"] == layer
            ]

            # Degenerate/skipped layer.
            if not lr:
                for group in ("00", "01", "10", "11"):
                    oof_rows.append({
                        "model": model,
                        "family": family,
                        "layer": layer,
                        "outcome_group": group,
                        "n": 0,
                        "mean_projection": float("nan"),
                        "median_projection": float("nan"),
                        "std_projection": float("nan"),
                    })

                for contrast in (
                    "C01_minus_C00",
                    "C11_minus_C10",
                    "audio_outcome_AUROC_given_text0",
                    "audio_outcome_AUROC_given_text1",
                ):
                    contrast_rows.append({
                        "model": model,
                        "family": family,
                        "layer": layer,
                        "contrast": contrast,
                        "lhs_group": "",
                        "rhs_group": "",
                        "lhs_n": 0,
                        "rhs_n": 0,
                        "lhs_mean": float("nan"),
                        "rhs_mean": float("nan"),
                        "mean_difference": float("nan"),
                        "approx_se": float("nan"),
                        "approx_ci95_low": float("nan"),
                        "approx_ci95_high": float("nan"),
                        "low_support": 1,
                    })

                continue

            group_stats = {}

            for group in ("00", "01", "10", "11"):

                vals = [
                    r[
                        "projection_toward_text_jailbreak_direction"
                    ]
                    for r in lr
                    if r["outcome_group"] == group
                ]

                s = summarize(vals)
                group_stats[group] = s

                oof_rows.append({
                    "model": model,
                    "family": family,
                    "layer": layer,
                    "outcome_group": group,
                    "n": s["n"],
                    "mean_projection": s["mean"],
                    "median_projection": s["median"],
                    "std_projection": s["std"],
                })

            for lhs, rhs, name in (
                ("01", "00", "C01_minus_C00"),
                ("11", "10", "C11_minus_C10"),
            ):
                a = group_stats[lhs]
                b = group_stats[rhs]

                if a["n"] and b["n"]:
                    diff = a["mean"] - b["mean"]
                else:
                    diff = float("nan")

                if a["n"] > 1 and b["n"] > 1:
                    se = math.sqrt(
                        a["std"] ** 2 / a["n"]
                        + b["std"] ** 2 / b["n"]
                    )
                    lo = diff - 1.96 * se
                    hi = diff + 1.96 * se
                else:
                    se = lo = hi = float("nan")

                contrast_rows.append({
                    "model": model,
                    "family": family,
                    "layer": layer,
                    "contrast": name,
                    "lhs_group": lhs,
                    "rhs_group": rhs,
                    "lhs_n": a["n"],
                    "rhs_n": b["n"],
                    "lhs_mean": a["mean"],
                    "rhs_mean": b["mean"],
                    "mean_difference": diff,
                    "approx_se": se,
                    "approx_ci95_low": lo,
                    "approx_ci95_high": hi,
                    "low_support": int(
                        min(a["n"], b["n"]) < 20
                    ),
                })

            for td in (0, 1):

                rr = [
                    r for r in lr
                    if r["text_decision"] == td
                ]

                y = [r["audio_decision"] for r in rr]
                x = [
                    r[
                        "projection_toward_text_jailbreak_direction"
                    ]
                    for r in rr
                ]

                auc = (
                    float(roc_auc_score(y, x))
                    if len(set(y)) == 2
                    else float("nan")
                )

                n1 = sum(v == 1 for v in y)
                n0 = sum(v == 0 for v in y)

                contrast_rows.append({
                    "model": model,
                    "family": family,
                    "layer": layer,
                    "contrast":
                        f"audio_outcome_AUROC_given_text{td}",
                    "lhs_group": "",
                    "rhs_group": "",
                    "lhs_n": n1,
                    "rhs_n": n0,
                    "lhs_mean": float("nan"),
                    "rhs_mean": float("nan"),
                    "mean_difference": auc,
                    "approx_se": float("nan"),
                    "approx_ci95_low": float("nan"),
                    "approx_ci95_high": float("nan"),
                    "low_support": int(min(n1, n0) < 20),
                })

    write_csv(per_example_csv, per_example_rows)
    write_csv(by_fold_csv, by_fold_rows)
    write_csv(oof_csv, oof_rows)
    write_csv(contrast_csv, contrast_rows)

    print()
    print("=" * 100, flush=True)
    print("[DONE]", model, flush=True)
    print("per-example :", per_example_csv, flush=True)
    print("by-fold     :", by_fold_csv, flush=True)
    print("OOF         :", oof_csv, flush=True)
    print("contrasts   :", contrast_csv, flush=True)
    print("=" * 100, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=(*SPECS.keys(), "all"),
        required=True,
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)

    models = (
        list(SPECS)
        if args.model == "all"
        else [args.model]
    )

    for model in models:
        run_model(model, device)


if __name__ == "__main__":
    main()
