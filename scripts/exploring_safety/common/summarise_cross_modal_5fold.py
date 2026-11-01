from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/INITIAL_BENCHMARK")
MODELS = ["qwen25_omni", "qwen2_audio", "kimi_audio"]

def transfer_summary(model, root):
    src = root / "cross_modal_transfer.csv"
    if not src.exists():
        print(f"[SKIP transfer] {model}: {src} missing")
        return

    df = pd.read_csv(src)

    required = {
        "fold", "safety_split",
        "source_probe_modality",
        "target_representation_modality",
        "layer", "n", "auroc", "balanced_accuracy"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{src}: missing {sorted(missing)}")

    # Preserve representation family for models such as Kimi if present.
    GROUP = []
    if "model" in df.columns:
        GROUP.append("model")
    if "family" in df.columns:
        GROUP.append("family")

    GROUP += [
        "safety_split",
        "source_probe_modality",
        "target_representation_modality",
        "layer",
    ]

    out = (
        df.groupby(GROUP, dropna=False)
          .agg(
              auroc_mean=("auroc", "mean"),
              auroc_std=("auroc", "std"),
              balanced_accuracy_mean=("balanced_accuracy", "mean"),
              balanced_accuracy_std=("balanced_accuracy", "std"),
              n_total=("n", "sum"),
              n_folds=("fold", "nunique"),
          )
          .reset_index()
          .sort_values(GROUP)
    )

    if not (out["n_folds"] == 5).all():
        print("\nBAD TRANSFER GROUPS:")
        print(out[out["n_folds"] != 5].to_string(index=False))
        raise RuntimeError(f"{model}: transfer groups missing folds")

    if out[["auroc_std","balanced_accuracy_std"]].isna().any().any():
        raise RuntimeError(f"{model}: transfer summary contains NaN SD")

    dst = root / "cross_modal_transfer_5fold_mean_std.csv"
    out.to_csv(dst, index=False)

    print(f"\n===== {model}: TRANSFER =====")
    print("WROTE:", dst)
    print("rows:", len(out))
    print(out.head(12).to_string(index=False))


def disagreement_summary(model, root):
    src = root / "harmful_disagreement_trajectory.csv"
    if not src.exists():
        print(f"[SKIP disagreement] {model}: {src} missing")
        return

    df = pd.read_csv(src)

    required = {"fold", "layer", "n"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{src}: missing {sorted(missing)}")

    # Build identity ONLY from actual experimental-condition columns.
    GROUP = []

    for c in [
        "model",
        "family",
        "safety_split",
        "probe_modality",
        "source_probe_modality",
        "disagreement_group",
        "group",
        "layer",
    ]:
        if c in df.columns and c not in GROUP:
            GROUP.append(c)

    if "layer" not in GROUP:
        GROUP.append("layer")

    # These are fold-level quantities which should be summarised,
    # NOT used to merge/split experimental conditions.
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

    for key, g in df.groupby(GROUP, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        r = dict(zip(GROUP, key))
        r["n_total"] = int(g["n"].sum())
        r["n_folds"] = int(g["fold"].nunique())

        for col in effect_cols:
            valid = g[col].notna() & g["n"].notna() & (g["n"] > 0)
            x = g.loc[valid]

            if len(x):
                # Overall pair-weighted effect across held-out folds.
                r[col + "_weighted_mean"] = float(
                    np.average(
                        x[col].astype(float),
                        weights=x["n"].astype(float)
                    )
                )

                # Fold-level stability.
                r[col + "_fold_mean"] = float(x[col].mean())
                r[col + "_fold_std"] = (
                    float(x[col].std(ddof=1)) if len(x) > 1 else np.nan
                )
            else:
                r[col + "_weighted_mean"] = np.nan
                r[col + "_fold_mean"] = np.nan
                r[col + "_fold_std"] = np.nan

        rows.append(r)

    out = pd.DataFrame(rows).sort_values(GROUP)

    bad = out[out["n_folds"] != 5]
    if len(bad):
        print(
            f"\n[NOTE] {model}: some disagreement groups occur in fewer "
            "than 5 folds. This can be legitimate if a fold has zero "
            "examples of that disagreement direction."
        )
        print(bad[GROUP + ["n_total","n_folds"]].to_string(index=False))

    dst = root / "harmful_disagreement_trajectory_5fold_summary.csv"
    out.to_csv(dst, index=False)

    print(f"\n===== {model}: HARMFUL DISAGREEMENT =====")
    print("WROTE:", dst)
    print("rows:", len(out))
    print(out.head(12).to_string(index=False))


for model in MODELS:
    root = ROOT / model / "cross_modal_analysis"
    print(f"\n\n######## {model} ########")

    if not root.exists():
        print("[SKIP] missing directory:", root)
        continue

    transfer_summary(model, root)
    disagreement_summary(model, root)

print("\n===== ALL DONE =====")
