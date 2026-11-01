#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"
OUTPUT = ROOT / "scripts/wk3/figures/section2"

MODELS = {
    "qwen2_audio": ("decoder", 32),
    "qwen25_omni": ("thinker", 28),
    "kimi_audio": ("shared_decoder", 28),
}

SOURCES = ("text", "audio", "combined")
ENERGY_K = (1, 4, 16, 128)
GENERALISATION_K = (1, 8)

METRICS = (
    "train_projected_energy_fraction",
    "test_projected_energy_fraction",
)


def load_data(model, component, n_layers, source):
    path = (
        RESULTS / model / "wk3/paired_harm_svd"
        / source / component / "energy_five_fold_mean_std.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "layer",
        "method",
        "k",
        "train_projected_energy_fraction_mean",
        "train_projected_energy_fraction_std",
        "test_projected_energy_fraction_mean",
        "test_projected_energy_fraction_std",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df = df.sort_values(["k", "layer"]).reset_index(drop=True)

    expected_k = {1, 4, 8, 16, 128}
    if set(df["k"]) != expected_k:
        raise ValueError(f"{path}: unexpected k values")

    expected_layers = set(range(1, n_layers + 1))

    for k in sorted(expected_k):
        part = df.loc[df["k"] == k]

        if (
            len(part) != n_layers
            or set(part["layer"]) != expected_layers
            or not part["method"].eq(f"svd_{k}").all()
        ):
            raise ValueError(f"{path}: incomplete rows for k={k}")

    if df.duplicated(["layer", "k"]).any():
        raise ValueError(f"{path}: duplicate layer/k rows")

    for metric in METRICS:
        mean = df[f"{metric}_mean"].to_numpy(dtype=float)
        std = df[f"{metric}_std"].to_numpy(dtype=float)

        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"{path}: non-finite {metric}")

        if (
            (mean < -1e-5).any()
            or (mean > 1 + 1e-5).any()
            or (std < 0).any()
        ):
            raise ValueError(f"{path}: invalid energy values")

    return df


def series(df, k, metric):
    part = df.loc[df["k"] == k].sort_values("layer")

    return (
        part["layer"].to_numpy(dtype=int),
        part[f"{metric}_mean"].to_numpy(dtype=float),
        part[f"{metric}_std"].to_numpy(dtype=float),
    )


def plot_energy(model, datasets):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(14.0, 4.0),
        sharex=True,
        sharey=True,
    )

    for ax, source in zip(axes, SOURCES):
        df = datasets[source]

        for k in ENERGY_K:
            layers, mean, std = series(
                df, k, "train_projected_energy_fraction"
            )

            line, = ax.plot(
                layers,
                mean,
                linewidth=1.8,
                label=f"Top-{k}",
            )

            ax.fill_between(
                layers,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=line.get_color(),
                alpha=0.13,
                linewidth=0,
            )

        ax.set_title(source.capitalize())
        ax.set_xlabel("Decoder layer boundary")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("Cumulative explained energy")

    fig.tight_layout()

    path = OUTPUT / f"01_energy_{model}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"REPLACED {path}", flush=True)


def plot_generalisation(model, datasets):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(14.0, 4.0),
        sharex=True,
        sharey=True,
    )

    for ax, source in zip(axes, SOURCES):
        df = datasets[source]

        for k in GENERALISATION_K:
            layers, train_mean, train_std = series(
                df, k, "train_projected_energy_fraction"
            )
            _, test_mean, test_std = series(
                df, k, "test_projected_energy_fraction"
            )

            line, = ax.plot(
                layers,
                train_mean,
                linewidth=1.8,
                linestyle="-",
                label=f"Top-{k} train",
            )

            color = line.get_color()

            ax.plot(
                layers,
                test_mean,
                linewidth=1.8,
                linestyle="--",
                color=color,
                label=f"Top-{k} test",
            )

            ax.fill_between(
                layers,
                np.clip(train_mean - train_std, 0, 1),
                np.clip(train_mean + train_std, 0, 1),
                color=color,
                alpha=0.10,
                linewidth=0,
            )

            ax.fill_between(
                layers,
                np.clip(test_mean - test_std, 0, 1),
                np.clip(test_mean + test_std, 0, 1),
                color=color,
                alpha=0.10,
                linewidth=0,
            )

        ax.set_title(source.capitalize())
        ax.set_xlabel("Decoder layer boundary")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("Cumulative explained energy")

    fig.tight_layout()

    path = OUTPUT / f"02_generalisation_{model}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"REPLACED {path}", flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_data = {}

    for model, (component, n_layers) in MODELS.items():
        all_data[model] = {
            source: load_data(model, component, n_layers, source)
            for source in SOURCES
        }

    print("Validated all nine corrected energy summaries.", flush=True)

    for model in MODELS:
        plot_energy(model, all_data[model])
        plot_generalisation(model, all_data[model])

    print("\nCOMPLETE: six energy PDFs replaced.", flush=True)


if __name__ == "__main__":
    main()
