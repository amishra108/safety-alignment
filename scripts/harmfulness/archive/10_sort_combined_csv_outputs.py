#!/usr/bin/env python3

from pathlib import Path
import os
import tempfile

import pandas as pd


ROOT = Path(
    "/research/milsrg1/user_workspace/am3244/safety-alignment"
    "/results/INITIAL_BENCHMARK"
)

MODELS = {
    "qwen2_audio": "decoder",
    "kimi_audio": "shared_decoder",
    "qwen25_omni": "thinker",
}

METHODS = (
    "mean", "svd_1", "svd_2", "svd_4", "svd_8",
    "svd_16", "svd_32", "svd_64", "svd_128",
)

METHOD_ORDER = {
    method: index for index, method in enumerate(METHODS)
}

FILES = (
    "audio_text_combined_comparison/direction_comparisons.csv",
    "audio_text_combined_comparison/direction_comparisons_fold_summary.csv",
    "paired_harm_auroc/combined/{component}/fold_1_predictions.csv",
    "paired_harm_auroc/combined/{component}/fold_2_predictions.csv",
    "paired_harm_auroc/combined/{component}/fold_3_predictions.csv",
    "paired_harm_auroc/combined/{component}/fold_4_predictions.csv",
    "paired_harm_auroc/combined/{component}/fold_5_predictions.csv",
    "paired_harm_auroc/combined/{component}/heldout_predictions.csv",
    "paired_harm_auroc/combined/{component}/heldout_predictions_normalized.csv",
    "paired_harm_auroc/combined/{component}/fold_metrics.csv",
    "paired_harm_auroc/combined/{component}/matched_predictions.csv",
    "paired_harm_auroc/combined/{component}/matched_behaviour_summary.csv",
)

SORT_COLUMNS = (
    "model",
    "component",
    "fold",
    "layer",
    "_method_order",
    "modality",
    "group_type",
    "group",
    "outcome",
    "topic",
    "language",
    "condition",
)


def process(path):
    if not path.is_file():
        print(f"SKIP (not yet generated): {path}", flush=True)
        return

    df = pd.read_csv(path)

    original_rows = len(df)
    original_columns = list(df.columns)

    if "method" in df.columns:
        unknown = set(df["method"].dropna()) - set(METHOD_ORDER)
        if unknown:
            raise ValueError(
                f"{path}: unknown methods {sorted(unknown)}"
            )

        df["_method_order"] = df["method"].map(METHOD_ORDER)

    columns = [
        column for column in SORT_COLUMNS
        if column in df.columns
    ]

    if columns:
        df = df.sort_values(
            columns, kind="stable", na_position="last"
        )

    df = df.drop(columns=["_method_order"], errors="ignore")

    if len(df) != original_rows:
        raise RuntimeError(f"Row count changed: {path}")

    if list(df.columns) != original_columns:
        raise RuntimeError(f"Column schema changed: {path}")

    for column in df.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].round(3)

    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".csv",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(
                handle,
                index=False,
                float_format="%.3f",
            )

        os.replace(temporary, path)

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(
        f"UPDATED rows={original_rows:,} "
        f"columns={len(original_columns)} {path}",
        flush=True,
    )


def main():
    for model, component in MODELS.items():
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model}")
        print(f"{'=' * 80}", flush=True)

        base = ROOT / model / "wk3"

        for template in FILES:
            path = base / template.format(component=component)
            process(path)

    print("\nCOMPLETE: all existing combined CSVs sorted and formatted.")


if __name__ == "__main__":
    main()
