#!/usr/bin/env python3

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

MODEL = "qwen25_omni"
N_LAYERS = 29
DIM = 3584

MODEL_ROOT = ROOT / "results/INITIAL_BENCHMARK" / MODEL
FOLD_ROOT = MODEL_ROOT / "5fold"
REP_ROOT = FOLD_ROOT / "representations_shared"

OUT_ROOT = MODEL_ROOT / "cross_modal_analysis"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PER_EXAMPLE_CSV = OUT_ROOT / "VLM_STYLE_independent_direction_per_example.csv"
BY_FOLD_CSV = OUT_ROOT / "VLM_STYLE_independent_direction_by_fold.csv"
OOF_CSV = OUT_ROOT / "VLM_STYLE_independent_direction_OOF.csv"
CONTRAST_CSV = OUT_ROOT / "VLM_STYLE_independent_direction_contrasts.csv"


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Malformed JSONL: {path}:{ln}") from e
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
        and r["safety_split"] == "harmful"
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
            f"{role}: text/audio matched-pair keys differ: "
            f"text_only={len(set(text)-set(audio))}, "
            f"audio_only={len(set(audio)-set(text))}"
        )

    return [(k, text[k], audio[k]) for k in sorted(text)]


def sample_std(vals):
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def summarize(vals):
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }

    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": sample_std(vals),
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


print("=" * 100, flush=True)
print("VLM-STYLE INDEPENDENT JAILBREAK-DIRECTION ANALYSIS", flush=True)
print("model      :", MODEL, flush=True)
print("family     : thinker", flush=True)
print("direction  : TRAIN harmful TEXT only: mean(y=1) - mean(y=0)", flush=True)
print("offset     : TRAIN matched audio-text, by language x condition", flush=True)
print("evaluation : held-out matched harmful examples only", flush=True)
print("=" * 100, flush=True)


# ----------------------------------------------------------------------
# Read all five folds first.
# ----------------------------------------------------------------------

fold_rows = {}

for fold in range(1, 6):
    p = FOLD_ROOT / f"fold_{fold}.jsonl"
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


# ----------------------------------------------------------------------
# Cache each harmful representation once.
# Decisions always come from current fold JSONL, never from .pt metadata.
# ----------------------------------------------------------------------

task_ids = sorted({
    r["task_id"]
    for r in fold_rows[1]
    if r["safety_split"] == "harmful"
})

print(
    f"\n[CACHE] loading {len(task_ids)} harmful thinker tensors once...",
    flush=True,
)

cache = {}

for i, task_id in enumerate(task_ids, 1):
    p = REP_ROOT / f"{task_id}.pt"

    if not p.exists():
        raise RuntimeError(f"Missing representation: {p}")

    obj = torch.load(
        p,
        map_location="cpu",
        weights_only=True,
    )

    x = obj["thinker"]

    if tuple(x.shape) != (N_LAYERS, DIM):
        raise RuntimeError(
            f"{task_id}: thinker shape {tuple(x.shape)} "
            f"!= {(N_LAYERS, DIM)}"
        )

    cache[task_id] = x.float().contiguous()

    if i % 250 == 0 or i == len(task_ids):
        print(
            f"  cached {i}/{len(task_ids)}",
            flush=True,
        )

print("[CACHE] done\n", flush=True)


device = torch.device("cuda")
print("[GPU]", torch.cuda.get_device_name(device), flush=True)

per_example_rows = []
by_fold_rows = []


# ----------------------------------------------------------------------
# Main cross-validation.
# ----------------------------------------------------------------------

for fold in range(1, 6):

    rows = fold_rows[fold]

    train_pairs = make_pairs(rows, "train")
    test_pairs = make_pairs(rows, "test")

    train_text_rows = [
        r for r in rows
        if r["train_test"] == "train"
        and r["safety_split"] == "harmful"
        and r["modality"] == "text"
    ]

    n0 = sum(int(r["decision"]) == 0 for r in train_text_rows)
    n1 = sum(int(r["decision"]) == 1 for r in train_text_rows)

    if n0 == 0 or n1 == 0:
        raise RuntimeError(
            f"Fold {fold}: missing text decision class: n0={n0}, n1={n1}"
        )

    print()
    print("=" * 100, flush=True)
    print(
        f"FOLD {fold} | "
        f"text-direction train y0={n0} y1={n1} | "
        f"offset train pairs={len(train_pairs)} | "
        f"test pairs={len(test_pairs)}",
        flush=True,
    )
    print("=" * 100, flush=True)

    for layer in range(N_LAYERS):

        # --------------------------------------------------------------
        # 1. Independent jailbreak direction.
        #
        # IMPORTANT:
        # ONLY harmful TEXT TRAIN representations/decisions enter here.
        # No audio outcome is used.
        # --------------------------------------------------------------

        text0 = torch.stack([
            cache[r["task_id"]][layer].float()
            for r in train_text_rows
            if int(r["decision"]) == 0
        ]).to(device)

        text1 = torch.stack([
            cache[r["task_id"]][layer].float()
            for r in train_text_rows
            if int(r["decision"]) == 1
        ]).to(device)

        mu0 = text0.mean(dim=0)
        mu1 = text1.mean(dim=0)

        v = mu1 - mu0
        v_norm = torch.linalg.vector_norm(v)

        if not torch.isfinite(v_norm) or v_norm.item() <= 1e-12:
            print(
                f"[SKIP] fold={fold} layer={layer:02d} "
                f"degenerate text jailbreak direction |v|={v_norm.item()}",
                flush=True,
            )
            del text0, text1, mu0, mu1, v
            continue

        v = v / v_norm

        del text0, text1, mu0, mu1


        # --------------------------------------------------------------
        # 2. Train-only generic modality offsets.
        #
        # No decision labels used here.
        # One offset per language x condition.
        # --------------------------------------------------------------

        offset_values = defaultdict(list)

        for _, tr, ar in train_pairs:
            ht = cache[tr["task_id"]][layer].float()
            ha = cache[ar["task_id"]][layer].float()

            offset_values[
                (tr["language"], tr["condition"])
            ].append(ha - ht)

        offsets = {}

        for stratum, vals in offset_values.items():
            offsets[stratum] = (
                torch.stack(vals)
                .mean(dim=0)
                .to(device)
            )


        # --------------------------------------------------------------
        # 3. Held-out matched shift.
        #
        # Audio decisions are used ONLY here for post-hoc grouping.
        # They never affected v or the modality offset.
        # --------------------------------------------------------------

        layer_records = []

        for key, tr, ar in test_pairs:

            stratum = (tr["language"], tr["condition"])

            if stratum not in offsets:
                raise RuntimeError(
                    f"Missing train offset for fold={fold}, "
                    f"layer={layer}, stratum={stratum}"
                )

            ht = cache[tr["task_id"]][layer].float().to(device)
            ha = cache[ar["task_id"]][layer].float().to(device)

            raw_shift = ha - ht
            centred_shift = raw_shift - offsets[stratum]

            projection = float(
                torch.dot(centred_shift, v).item()
            )

            td = int(tr["decision"])
            ad = int(ar["decision"])

            group = f"{td}{ad}"

            rec = {
                "model": MODEL,
                "family": "thinker",
                "fold": fold,
                "layer": layer,
                "base_prompt_id": tr["base_prompt_id"],
                "language": tr["language"],
                "condition": tr["condition"],
                "text_decision": td,
                "audio_decision": ad,
                "outcome_group": group,
                "projection_toward_text_jailbreak_direction": projection,
            }

            per_example_rows.append(rec)
            layer_records.append(rec)


        # --------------------------------------------------------------
        # 4. Fold-level summaries.
        # --------------------------------------------------------------

        for group in ("00", "01", "10", "11"):

            vals = [
                r["projection_toward_text_jailbreak_direction"]
                for r in layer_records
                if r["outcome_group"] == group
            ]

            s = summarize(vals)

            by_fold_rows.append({
                "model": MODEL,
                "family": "thinker",
                "fold": fold,
                "layer": layer,
                "outcome_group": group,
                "n": s["n"],
                "mean_projection": s["mean"],
                "median_projection": s["median"],
                "std_projection": s["std"],
            })


        # Direct held-out question:
        # among prompts with same text outcome, does projection distinguish
        # whether audio ultimately complied?
        auc_text0 = float("nan")
        auc_text1 = float("nan")

        r0 = [
            r for r in layer_records
            if r["text_decision"] == 0
        ]
        r1 = [
            r for r in layer_records
            if r["text_decision"] == 1
        ]

        if len({r["audio_decision"] for r in r0}) == 2:
            auc_text0 = roc_auc_score(
                [r["audio_decision"] for r in r0],
                [
                    r["projection_toward_text_jailbreak_direction"]
                    for r in r0
                ],
            )

        if len({r["audio_decision"] for r in r1}) == 2:
            auc_text1 = roc_auc_score(
                [r["audio_decision"] for r in r1],
                [
                    r["projection_toward_text_jailbreak_direction"]
                    for r in r1
                ],
            )

        print(
            f"[fold {fold}] layer={layer:02d} "
            f"|v|={v_norm.item():.3f} "
            f"AUC(audio outcome | text=0)={auc_text0:.3f} "
            f"AUC(audio outcome | text=1)={auc_text1:.3f}",
            flush=True,
        )

        del v, offsets
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Combined out-of-fold summaries.
#
# Every base prompt appears in test exactly once across the five folds,
# so this is the proper combined OOF analysis.
# ----------------------------------------------------------------------

oof_rows = []
contrast_rows = []

for layer in range(N_LAYERS):

    lr = [
        r for r in per_example_rows
        if r["layer"] == layer
    ]

    group_stats = {}

    for group in ("00", "01", "10", "11"):

        vals = [
            r["projection_toward_text_jailbreak_direction"]
            for r in lr
            if r["outcome_group"] == group
        ]

        s = summarize(vals)
        group_stats[group] = s

        oof_rows.append({
            "model": MODEL,
            "family": "thinker",
            "layer": layer,
            "outcome_group": group,
            "n": s["n"],
            "mean_projection": s["mean"],
            "median_projection": s["median"],
            "std_projection": s["std"],
        })


    # Positive contrast means audio=1 examples shifted MORE toward the
    # independent text-derived jailbreak direction.
    for lhs, rhs, name in (
        ("01", "00", "C01_minus_C00"),
        ("11", "10", "C11_minus_C10"),
    ):
        a = group_stats[lhs]
        b = group_stats[rhs]

        contrast = a["mean"] - b["mean"]

        if a["n"] > 1 and b["n"] > 1:
            se = math.sqrt(
                (a["std"] ** 2) / a["n"]
                + (b["std"] ** 2) / b["n"]
            )
            lo = contrast - 1.96 * se
            hi = contrast + 1.96 * se
        else:
            se = lo = hi = float("nan")

        contrast_rows.append({
            "model": MODEL,
            "family": "thinker",
            "layer": layer,
            "contrast": name,
            "lhs_group": lhs,
            "rhs_group": rhs,
            "lhs_n": a["n"],
            "rhs_n": b["n"],
            "lhs_mean": a["mean"],
            "rhs_mean": b["mean"],
            "mean_difference": contrast,
            "approx_se": se,
            "approx_ci95_low": lo,
            "approx_ci95_high": hi,
            "low_support": int(min(a["n"], b["n"]) < 20),
        })


    # OOF predictive AUROC of audio outcome, conditional on text outcome.
    for td in (0, 1):

        rr = [
            r for r in lr
            if r["text_decision"] == td
        ]

        y = [r["audio_decision"] for r in rr]
        x = [
            r["projection_toward_text_jailbreak_direction"]
            for r in rr
        ]

        auc = (
            float(roc_auc_score(y, x))
            if len(set(y)) == 2
            else float("nan")
        )

        contrast_rows.append({
            "model": MODEL,
            "family": "thinker",
            "layer": layer,
            "contrast": f"audio_outcome_AUROC_given_text{td}",
            "lhs_group": "",
            "rhs_group": "",
            "lhs_n": sum(v == 1 for v in y),
            "rhs_n": sum(v == 0 for v in y),
            "lhs_mean": float("nan"),
            "rhs_mean": float("nan"),
            "mean_difference": auc,
            "approx_se": float("nan"),
            "approx_ci95_low": float("nan"),
            "approx_ci95_high": float("nan"),
            "low_support": int(min(
                sum(v == 1 for v in y),
                sum(v == 0 for v in y),
            ) < 20),
        })


write_csv(PER_EXAMPLE_CSV, per_example_rows)
write_csv(BY_FOLD_CSV, by_fold_rows)
write_csv(OOF_CSV, oof_rows)
write_csv(CONTRAST_CSV, contrast_rows)

print()
print("=" * 100, flush=True)
print("DONE", flush=True)
print("=" * 100, flush=True)
print("per-example :", PER_EXAMPLE_CSV, flush=True)
print("by-fold     :", BY_FOLD_CSV, flush=True)
print("OOF groups  :", OOF_CSV, flush=True)
print("contrasts   :", CONTRAST_CSV, flush=True)

print("\n===== FINAL-LAYER OOF =====", flush=True)

for r in oof_rows:
    if r["layer"] == N_LAYERS - 1:
        print(
            f"{r['outcome_group']} "
            f"N={r['n']:4d} "
            f"mean={r['mean_projection']:+.5f}",
            flush=True,
        )

print("\n===== FINAL-LAYER CONTRASTS =====", flush=True)

for r in contrast_rows:
    if r["layer"] == N_LAYERS - 1:
        print(
            f"{r['contrast']:36s} "
            f"value={r['mean_difference']:+.5f}",
            flush=True,
        )
