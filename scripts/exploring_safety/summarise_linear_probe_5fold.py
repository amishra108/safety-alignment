from pathlib import Path
import pandas as pd

ROOT = Path("results/INITIAL_BENCHMARK")
MODELS = ["qwen25_omni", "qwen2_audio", "kimi_audio"]

for model in MODELS:
    root = ROOT / model / "linear_probe_gpu"
    files = [root / f"fold_{f}" / "metrics.csv" for f in range(1, 6)]

    missing = [p for p in files if not p.exists()]
    if missing:
        print(f"\n[SKIP] {model}: missing:")
        for p in missing:
            print(" ", p)
        continue

    parts = []
    for f, p in enumerate(files, 1):
        df = pd.read_csv(p)

        required = {
            "model","fold","safety_split","modality","family","layer",
            "feature_dim","train_n","test_n","train_0","train_1",
            "test_0","test_1","accuracy","balanced_accuracy","auroc"
        }
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise RuntimeError(f"{p}: missing columns {sorted(missing_cols)}")

        # Trust directory identity, but verify CSV fold agrees.
        if not (df["fold"] == f).all():
            raise RuntimeError(f"{p}: fold column does not equal {f}")

        parts.append(df)

    df = pd.concat(parts, ignore_index=True)

    GROUP = ["model", "safety_split", "modality", "family", "layer"]

    out = (
        df.groupby(GROUP, dropna=False)
          .agg(
              feature_dim=("feature_dim", "first"),

              accuracy_mean=("accuracy", "mean"),
              accuracy_std=("accuracy", "std"),

              balanced_accuracy_mean=("balanced_accuracy", "mean"),
              balanced_accuracy_std=("balanced_accuracy", "std"),

              auroc_mean=("auroc", "mean"),
              auroc_std=("auroc", "std"),

              n_folds=("fold", "nunique"),

              # Since held-out folds are disjoint, useful total evaluation N:
              test_n_total=("test_n", "sum"),
              test_0_total=("test_0", "sum"),
              test_1_total=("test_1", "sum"),
          )
          .reset_index()
          .sort_values(["safety_split", "modality", "family", "layer"])
    )

    if not (out["n_folds"] == 5).all():
        print(out[out["n_folds"] != 5].to_string(index=False))
        raise RuntimeError(f"{model}: some configurations do not have all 5 folds")

    stdcols = ["accuracy_std", "balanced_accuracy_std", "auroc_std"]
    if out[stdcols].isna().any().any():
        raise RuntimeError(f"{model}: NaN SD found despite 5-fold aggregation")

    outfile = root / "metrics_5fold_mean_std.csv"
    out.to_csv(outfile, index=False)

    print(f"\n===== {model} =====")
    print(f"WROTE: {outfile}")
    print(f"ROWS: {len(out)}")
    print(out.to_string(index=False))

print("\n===== DONE =====")
