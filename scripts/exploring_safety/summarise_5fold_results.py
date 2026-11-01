#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/INITIAL_BENCHMARK")
MODELS = ["qwen25_omni", "qwen2_audio", "kimi_audio"]

def summarise_linear(model):
    root = ROOT / model / "linear_probe_gpu"

    fold_files = sorted(root.glob("fold_*/metrics.csv"))

    if fold_files:
        df = pd.concat(
            [pd.read_csv(p) for p in fold_files],
            ignore_index=True,
        )
    else:
        p = root / "metrics.csv"
        if not p.exists():
            print(f"[linear missing] {model}")
            return
        df = pd.read_csv(p)

    keys = [
        c for c in [
            "safety_split",
            "safety",
            "modality",
            "probe_modality",
            "representation_family",
            "family",
            "representation",
            "layer",
            "direction",
            "train_modality",
            "test_modality",
        ]
        if c in df.columns
    ]

    metrics = [
        c for c in [
            "accuracy",
            "balanced_accuracy",
            "auroc",
        ]
        if c in df.columns
    ]

    if not keys or not metrics:
        raise RuntimeError(
            f"{model}: cannot summarise; "
            f"keys={keys}, metrics={metrics}"
        )

    agg = (
        df.groupby(keys, dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg.columns = [
        "_".join(x).rstrip("_")
        if isinstance(x, tuple)
        else x
        for x in agg.columns
    ]

    n = (
        df.groupby(keys, dropna=False)["fold"]
        .nunique()
        .reset_index(name="n_folds")
    )

    agg = agg.merge(n, on=keys, how="left")

    out = root / "metrics_mean_std.csv"
    agg.to_csv(out, index=False)

    print(
        f"[linear] {model}: "
        f"{len(df)} fold rows -> {out}"
    )

    bad = agg[agg["n_folds"] != 5]
    if len(bad):
        print(
            f"[WARNING] {model}: "
            f"{len(bad)} summary rows do not have 5 folds"
        )
    else:
        print(f"[OK] {model}: every summary row has 5 folds")


def summarise_crossmodal_file(path):
    df = pd.read_csv(path)

    if "fold" not in df.columns:
        return

    keys = [
        c for c in [
            "safety_split",
            "safety",
            "probe_modality",
            "train_modality",
            "test_modality",
            "representation_family",
            "family",
            "representation",
            "layer",
            "direction",
            "disagreement_group",
            "group",
        ]
        if c in df.columns
    ]

    metrics = [
        c for c in [
            "accuracy",
            "balanced_accuracy",
            "auroc",
        ]
        if c in df.columns
    ]

    pieces = None

    if metrics:
        a = (
            df.groupby(keys, dropna=False)[metrics]
            .agg(["mean", "std"])
            .reset_index()
        )
        a.columns = [
            "_".join(x).rstrip("_")
            if isinstance(x, tuple)
            else x
            for x in a.columns
        ]
        pieces = a

    if "n" in df.columns:
        effect_cols = [
            c for c in [
                "mean_text_logit",
                "mean_audio_logit",
                "mean_audio_minus_text",
                "median_audio_minus_text",
            ]
            if c in df.columns
        ]

        rows = []

        for vals, g in df.groupby(keys, dropna=False):
            if not isinstance(vals, tuple):
                vals = (vals,)

            r = dict(zip(keys, vals))
            r["n_total"] = int(g["n"].sum())
            r["n_folds"] = int(g["fold"].nunique())

            for c in effect_cols:
                valid = (
                    g[c].notna()
                    & g["n"].notna()
                    & (g["n"] > 0)
                )
                gg = g.loc[valid]

                if len(gg):
                    r[c + "_weighted_mean"] = np.average(
                        gg[c].astype(float),
                        weights=gg["n"].astype(float),
                    )
                    r[c + "_fold_mean"] = gg[c].mean()
                    r[c + "_fold_std"] = gg[c].std(ddof=1)
                else:
                    r[c + "_weighted_mean"] = np.nan
                    r[c + "_fold_mean"] = np.nan
                    r[c + "_fold_std"] = np.nan

            rows.append(r)

        b = pd.DataFrame(rows)
        pieces = (
            b if pieces is None
            else pieces.merge(b, on=keys, how="outer")
        )

    if pieces is None:
        return

    if "n_folds" not in pieces.columns:
        n = (
            df.groupby(keys, dropna=False)["fold"]
            .nunique()
            .reset_index(name="n_folds")
        )
        pieces = pieces.merge(n, on=keys, how="left")

    out = path.with_name(
        path.stem + "_mean_std.csv"
    )
    pieces.to_csv(out, index=False)

    print(f"[cross-modal] {path} -> {out}")


for model in MODELS:
    summarise_linear(model)

    cm = ROOT / model / "cross_modal_analysis"
    if cm.exists():
        for p in sorted(cm.glob("*.csv")):
            if not p.name.endswith("_mean_std.csv"):
                summarise_crossmodal_file(p)

print("DONE")
