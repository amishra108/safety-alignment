#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"
OUTPUT = ROOT / "scripts/wk3/figures/refusal_auroc"

MODELS = {
    "qwen2_audio": ("decoder", 32),
    "qwen25_omni": ("thinker", 28),
    "kimi_audio": ("shared_decoder", 28),
}

KEYS = ["fold", "layer", "topic", "language", "condition"]

COMPARISONS = (
    ("harmful", "01", "00", "H: 00 vs 01"),
    ("harmful", "11", "10", "H: 10 vs 11"),
    ("benign", "00", "01", "B: 01 vs 00"),
    ("benign", "10", "11", "B: 11 vs 10"),
)


def load_projections(model, component, modality):
    path = (
        RESULTS / model / "wk3/paired_harm_projections"
        / modality / component
        / "all_folds_heldout_raw_and_z_projections.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    columns = KEYS + [
        "direction_source",
        "method",
        "harmful_score",
        "benign_score",
        "harmful_decision",
        "benign_decision",
    ]

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df = df.loc[
        (df["direction_source"] == "text")
        & (df["method"] == "svd_1"),
        KEYS + [
            "harmful_score",
            "benign_score",
            "harmful_decision",
            "benign_decision",
        ],
    ].copy()

    if df.empty or df.duplicated(KEYS).any():
        raise ValueError(f"Empty or duplicate observations: {path}")

    for column in (
        "harmful_score",
        "benign_score",
        "harmful_decision",
        "benign_decision",
    ):
        df[column] = pd.to_numeric(df[column], errors="raise")

    for category in ("harmful", "benign"):
        decision = df[f"{category}_decision"]

        if not decision.isin([0, 1]).all():
            raise ValueError(f"Invalid {category} decisions: {path}")

        if not np.isfinite(df[f"{category}_score"]).all():
            raise ValueError(f"Non-finite {category} projections: {path}")

    return df.rename(columns={
        column: f"{modality}_{column}"
        for column in (
            "harmful_score",
            "benign_score",
            "harmful_decision",
            "benign_decision",
        )
    })


def prepare_model(model, component, n_layers):
    text = load_projections(model, component, "text")
    audio = load_projections(model, component, "audio")

    df = text.merge(
        audio,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not df["_merge"].eq("both").all():
        raise ValueError(f"{model}: unmatched text/audio examples")

    df = df.drop(columns="_merge")

    expected_layers = list(range(1, n_layers + 1))

    if sorted(df["layer"].unique()) != expected_layers:
        raise ValueError(f"{model}: unexpected decoder layers")

    if len(df) != 1600 * n_layers:
        raise ValueError(f"{model}: incorrect total example count")

    if sorted(df["fold"].unique()) != [1, 2, 3, 4, 5]:
        raise ValueError(f"{model}: unexpected fold identifiers")

    for layer, part in df.groupby("layer"):
        if len(part) != 1600:
            raise ValueError(f"{model}: layer {layer} has {len(part)} rows")

    for category in ("harmful", "benign"):
        df[f"{category}_group"] = (
            df[f"text_{category}_decision"].astype(int).astype(str)
            + df[f"audio_{category}_decision"].astype(int).astype(str)
        )

        df[f"{category}_shift"] = (
            df[f"audio_{category}_score"]
            - df[f"text_{category}_score"]
        )

    return df


def evaluate_comparison(df, category, negative_group, positive_group):
    group_column = f"{category}_group"
    score_column = f"{category}_shift"

    subset = df.loc[
        df[group_column].isin([negative_group, positive_group])
    ]

    y = (subset[group_column] == positive_group).astype(int)
    scores = subset[score_column]

    n_negative = int((y == 0).sum())
    n_positive = int((y == 1).sum())

    if n_negative == 0 or n_positive == 0:
        raise ValueError(
            f"Missing class: {category}, "
            f"{negative_group} vs {positive_group}"
        )

    pooled_auc = roc_auc_score(y, scores)

    fold_aucs = []

    for fold, part in subset.groupby("fold"):
        fold_y = (part[group_column] == positive_group).astype(int)

        if fold_y.nunique() == 2:
            fold_aucs.append(
                roc_auc_score(fold_y, part[score_column])
            )

    if not fold_aucs:
        raise ValueError("No fold contains both comparison groups")

    fold_mean = float(np.mean(fold_aucs))
    fold_std = (
        float(np.std(fold_aucs, ddof=1))
        if len(fold_aucs) > 1
        else np.nan
    )

    return {
        "n_negative": n_negative,
        "n_positive": n_positive,
        "auc": float(pooled_auc),
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "valid_folds": len(fold_aucs),
    }


def calculate_results(df, n_layers):
    rows = []

    for layer in range(1, n_layers + 1):
        part = df.loc[df["layer"] == layer]

        for category, negative, positive, label in COMPARISONS:
            metrics = evaluate_comparison(
                part, category, negative, positive
            )

            rows.append({
                "layer": layer,
                "comparison": label,
                "category": category,
                **metrics,
            })

    return pd.DataFrame(rows)


def print_results(model, df, results, n_layers):
    print("\n" + "=" * 92)
    print(model.upper())
    print("=" * 92)

    print("Direction: text-trained SVD-1")
    print("Score: audio raw projection - text raw projection")
    print("Positive class: AUDIO REFUSAL for every comparison")
    print("AUROC > 0.5 supports the hypothesised direction")
    print("Harmful: 0=refusal, 1=attack success")
    print("Benign:  0=compliance, 1=refusal")

    first = df.loc[df["layer"] == 1]

    print("\nASR / BRR")
    for modality in ("text", "audio"):
        asr = 100 * (
            first[f"{modality}_harmful_decision"] == 1
        ).mean()
        brr = 100 * (
            first[f"{modality}_benign_decision"] == 1
        ).mean()

        print(
            f"  {modality:<6} ASR={asr:6.2f}% "
            f"BRR={brr:6.2f}%"
        )

    print("\nDECISION GROUP COUNTS")
    for category in ("harmful", "benign"):
        counts = first[f"{category}_group"].value_counts()

        print(
            f"  {category:<8} "
            + "  ".join(
                f"{group}={int(counts.get(group, 0))}"
                for group in ("00", "01", "10", "11")
            )
        )

    print("\nPOOLED HELD-OUT AUROC BY LAYER")
    print(
        f"{'Layer':>5} "
        f"{'H 00 vs 01':>12} "
        f"{'H 10 vs 11':>12} "
        f"{'B 01 vs 00':>12} "
        f"{'B 11 vs 10':>12}"
    )
    print("-" * 62)

    labels = [item[3] for item in COMPARISONS]

    for layer in range(1, n_layers + 1):
        part = results.loc[results["layer"] == layer]
        values = [
            float(
                part.loc[
                    part["comparison"] == label, "auc"
                ].iloc[0]
            )
            for label in labels
        ]

        print(
            f"{layer:5d} "
            + " ".join(f"{value:12.3f}" for value in values)
        )

    print("\nFINAL-LAYER RESULTS")
    print(
        f"{'Comparison':<15} {'N neg':>7} {'N pos':>7} "
        f"{'AUROC':>8} {'Fold mean ± SD':>18}"
    )
    print("-" * 66)

    for label in labels:
        row = results.loc[
            (results["layer"] == n_layers)
            & (results["comparison"] == label)
        ].iloc[0]

        print(
            f"{label:<15} "
            f"{int(row['n_negative']):7d} "
            f"{int(row['n_positive']):7d} "
            f"{row['auc']:8.3f} "
            f"{row['fold_mean']:7.3f} ± {row['fold_std']:.3f}"
            f"  ({int(row['valid_folds'])}/5 folds)"
        )

    print("\nHIGHEST OBSERVED AUROC BY COMPARISON")
    for label in labels:
        part = results.loc[results["comparison"] == label]
        best = part.loc[part["auc"].idxmax()]

        print(
            f"  {label:<15} "
            f"layer={int(best['layer']):2d} "
            f"AUROC={best['auc']:.3f} "
            f"fold={best['fold_mean']:.3f}±{best['fold_std']:.3f}"
        )



def print_fold_sample_sizes_and_best_layers(model, df, results):
    print("\n" + "=" * 92)
    print(f"{model.upper()}: FOLD SAMPLE SIZES AND BEST FIVE-FOLD MEAN AUROC")
    print("=" * 92)

    first_layer = df.loc[df["layer"] == 1]

    for category, negative, positive, label in COMPARISONS:
        print(f"\n{label}")
        print(f"  Negative class: {negative} | Positive class: {positive}")
        print(f"  {'Fold':>5} {'N negative':>12} {'N positive':>12} {'N total':>10}")

        for fold in range(1, 6):
            part = first_layer.loc[first_layer["fold"] == fold]
            counts = part[f"{category}_group"].value_counts()

            n_neg = int(counts.get(negative, 0))
            n_pos = int(counts.get(positive, 0))

            print(
                f"  {fold:5d} {n_neg:12d} {n_pos:12d} "
                f"{n_neg + n_pos:10d}"
            )

        subset = results.loc[
            results["comparison"] == label
        ].sort_values("layer")

        best = subset.loc[subset["fold_mean"].idxmax()]

        print(
            f"  BEST: layer={int(best['layer'])}, "
            f"five-fold mean AUROC={best['fold_mean']:.3f}, "
            f"SD={best['fold_std']:.3f}, "
            f"valid folds={int(best['valid_folds'])}/5"
        )

    print("\nLATEX TABLE ROWS (FIVE-FOLD MEAN MAXIMA)")
    for category, negative, positive, label in COMPARISONS:
        subset = results.loc[results["comparison"] == label]
        best = subset.loc[subset["fold_mean"].idxmax()]

        print(
            f"{label} & {best['fold_mean']:.3f} & "
            f"{best['fold_std']:.3f} & {int(best['layer'])} \\\\"
        )

def draw_panel(ax, results, labels, title):
    for label in labels:
        part = results.loc[
            results["comparison"] == label
        ].sort_values("layer")

        layers = part["layer"].to_numpy()
        values = part["fold_mean"].to_numpy(dtype=float)
        std = part["fold_std"].to_numpy(dtype=float)

        line, = ax.plot(
            layers,
            values,
            linewidth=1.9,
            label=label,
        )

        ax.fill_between(
            layers,
            np.clip(values - std, 0, 1),
            np.clip(values + std, 0, 1),
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Chance (0.5)",
    )

    ax.set_title(title)
    ax.set_xlabel("Decoder layer boundary")
    ax.set_ylim(0, 1)
    ax.set_xlim(
        int(results["layer"].min()),
        int(results["layer"].max()),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)


def plot_model(model, results):
    labels = [item[3] for item in COMPARISONS]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10.5, 4.4),
        sharex=True,
        sharey=True,
    )

    draw_panel(
        axes[0],
        results,
        labels[:2],
        "Harmful prompts",
    )

    draw_panel(
        axes[1],
        results,
        labels[2:],
        "Benign prompts",
    )

    axes[0].set_ylabel("Five-fold mean AUROC (audio refusal positive)")

    fig.tight_layout()

    path = OUTPUT / f"{model}_refusal_auroc.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSAVED PDF: {path}", flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for model, (component, n_layers) in MODELS.items():
        df = prepare_model(model, component, n_layers)
        results = calculate_results(df, n_layers)

        print_results(model, df, results, n_layers)
        print_fold_sample_sizes_and_best_layers(model, df, results)
        plot_model(model, results)

    print("\nCOMPLETE: three AUROC PDFs generated.", flush=True)


if __name__ == "__main__":
    main()
