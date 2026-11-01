#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"
OUT = ROOT / "scripts/wk3/figures/section2"

MODELS = {
    "qwen2_audio": ("decoder", "Qwen2-Audio"),
    "qwen25_omni": ("thinker", "Qwen2.5-Omni"),
    "kimi_audio": ("shared_decoder", "Kimi-Audio"),
}

MODALITIES = ("text", "audio")
ENERGY_K = (1, 4, 16, 128)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})


def results_dir(model, modality, component):
    return RESULTS / model / "wk3/paired_harm_svd" / modality / component


def energy_data(model, modality, component):
    path = results_dir(model, modality, component) / "energy_five_fold_mean_std.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fold_directions(model, modality, component, fold):
    path = results_dir(model, modality, component) / f"fold_{fold}_directions.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return (
            data["means"].copy(),
            data["vectors"].copy(),
            data["layer_boundaries"].copy(),
        )


def rows(df, method):
    result = df.loc[df["method"] == method].sort_values("layer")
    if result.empty:
        raise ValueError(f"Method {method} is absent")
    return result


def finish(fig, filename):
    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"SAVED {path}", flush=True)


def plot_energy(model, component, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, modality in zip(axes, MODALITIES):
        df = energy_data(model, modality, component)

        for k in ENERGY_K:
            subset = rows(df, f"svd_{k}")
            ax.plot(
                subset["layer"],
                subset["cumulative_explained_energy_mean"],
                label=f"$k={k}$",
            )

        ax.set_title(modality.capitalize())
        ax.set_xlabel("Decoder boundary")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Fraction of difference energy")
    axes[1].legend(frameon=False)
    fig.suptitle(f"{title}: cumulative SVD energy")
    finish(fig, f"01_energy_{model}.pdf")


def plot_generalisation(model, component, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, modality in zip(axes, MODALITIES):
        df = energy_data(model, modality, component)

        for method, name in (("mean", "Mean"), ("svd_1", "SVD-1")):
            subset = rows(df, method)

            for partition, linestyle in (("train", "-"), ("test", "--")):
                column = f"{partition}_projected_energy_fraction_mean"
                ax.plot(
                    subset["layer"],
                    subset[column],
                    linestyle=linestyle,
                    label=f"{name} {partition}",
                )

        ax.set_title(modality.capitalize())
        ax.set_xlabel("Decoder boundary")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Fraction of difference energy")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title}: training versus held-out pairs")
    finish(fig, f"02_generalisation_{model}.pdf")


def normalise(x):
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError("Invalid projected direction")
    return x / norm


def calculate_overlap(model, component):
    cosine_by_fold = []
    subspace_by_fold = []
    reference_layers = None

    for fold in range(1, 6):
        mt, vt, lt = fold_directions(model, "text", component, fold)
        ma, va, la = fold_directions(model, "audio", component, fold)

        if not np.array_equal(lt, la):
            raise ValueError(f"{model} fold {fold}: layer mismatch")

        if reference_layers is None:
            reference_layers = lt
        elif not np.array_equal(reference_layers, lt):
            raise ValueError(f"{model}: fold boundary mismatch")

        cosine = []
        overlap = []

        for i in range(len(lt)):
            wt = normalise((vt[i, :1] @ mt[i]) @ vt[i, :1])
            wa = normalise((va[i, :1] @ ma[i]) @ va[i, :1])
            cosine.append(abs(float(wt @ wa)))

            k = 8
            cross = vt[i, :k] @ va[i, :k].T
            overlap.append(float(np.linalg.norm(cross, "fro") ** 2 / k))

        cosine_by_fold.append(cosine)
        subspace_by_fold.append(overlap)

    return (
        reference_layers,
        np.asarray(cosine_by_fold),
        np.asarray(subspace_by_fold),
    )


def plot_overlap(model, title, layers, cosine, subspace):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, values, subtitle in (
        (axes[0], cosine, "SVD-1 direction |cosine|"),
        (axes[1], subspace, "Top-8 SVD subspace overlap"),
    ):
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)

        ax.plot(layers, mean)
        ax.fill_between(
            layers,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            alpha=0.2,
        )

        ax.set_title(subtitle)
        ax.set_xlabel("Decoder boundary")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Text–audio similarity")
    fig.suptitle(f"{title}: text–audio harmfulness overlap")
    finish(fig, f"03_overlap_{model}.pdf")


def safety_gaps(model):
    path = RESULTS / model / "wk3/dataset_summary/fold_assignments.csv"
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"task_id", "split", "modality", "decision"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}")
    if df["task_id"].duplicated().any():
        raise ValueError(f"Duplicate task IDs in {path}")
    if not df["decision"].isin((0, 1)).all():
        raise ValueError(f"Invalid decisions in {path}")

    gaps = {}

    for split, metric in (("harmful", "ASR"), ("benign", "BRR")):
        rates = {}

        for modality in MODALITIES:
            subset = df.loc[
                (df["split"] == split)
                & (df["modality"] == modality)
            ]
            if subset.empty:
                raise ValueError(f"Missing {split}/{modality} for {model}")

            rates[modality] = 100 * subset["decision"].mean()

        gaps[metric] = abs(rates["text"] - rates["audio"])
        print(
            f"{model} {metric}: text={rates['text']:.2f}%, "
            f"audio={rates['audio']:.2f}%, gap={gaps[metric]:.2f} pp"
        )

    return gaps


def plot_performance(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric in zip(axes, ("ASR", "BRR")):
        for entry in data:
            x = entry["overlap"]
            y = entry[metric]

            ax.scatter(x, y, s=65)
            ax.annotate(
                entry["label"], (x, y),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_xlabel("Top-8 text–audio subspace overlap")
        ax.set_ylabel(f"Absolute {metric} gap (percentage points)")
        ax.set_title(metric)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)

    fig.suptitle("Cross-model overlap versus modality safety gap")
    finish(fig, "04_overlap_vs_safety_gap.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comparisons = []

    for model, (component, title) in MODELS.items():
        plot_energy(model, component, title)
        plot_generalisation(model, component, title)

        layers, cosine, subspace = calculate_overlap(model, component)
        plot_overlap(model, title, layers, cosine, subspace)

        last_quarter = max(1, len(layers) // 4)
        mean_overlap = float(subspace[:, -last_quarter:].mean())
        gaps = safety_gaps(model)

        comparisons.append({
            "label": title,
            "overlap": mean_overlap,
            **gaps,
        })

        print(
            f"{title}: final-quarter top-8 overlap="
            f"{mean_overlap:.4f}",
            flush=True,
        )

    plot_performance(comparisons)
    print("\nCOMPLETE: 10 PDF figures", flush=True)


if __name__ == "__main__":
    main()
