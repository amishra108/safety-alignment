#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
OUTPUT = ROOT / "scripts/wk3/figures/section2"

MODELS = {
    "qwen2_audio": "decoder",
    "qwen25_omni": "thinker",
    "kimi_audio": "shared_decoder",
}

COMPARISONS = (
    ("text", "audio", "Text–Audio"),
    ("text", "combined", "Text–Combined"),
    ("audio", "combined", "Audio–Combined"),
)


def load_directions(model, component, modality, fold):
    path = (
        ROOT
        / "results/INITIAL_BENCHMARK"
        / model
        / "wk3/paired_harm_svd"
        / modality
        / component
        / f"fold_{fold}_directions.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float64)
        layers = np.asarray(data["layer_boundaries"], dtype=int)

    if vectors.ndim != 3 or vectors.shape[1] < 8:
        raise ValueError(f"Invalid SVD vectors in {path}")

    if len(layers) != len(vectors):
        raise ValueError(f"Layer count mismatch in {path}")

    return layers, vectors


def subspace_overlap(a, b, k):
    cross = a[:k] @ b[:k].T
    return float(np.sum(cross ** 2) / k)


def plot_model(model, component):
    results = {1: {}, 8: {}}
    reference_layers = None

    for first, second, label in COMPARISONS:
        for k in (1, 8):
            results[k][label] = []

        for fold in range(1, 6):
            layers_a, vectors_a = load_directions(
                model, component, first, fold
            )
            layers_b, vectors_b = load_directions(
                model, component, second, fold
            )

            if not np.array_equal(layers_a, layers_b):
                raise ValueError(
                    f"Layer mismatch: {model}, {label}, fold {fold}"
                )

            if reference_layers is None:
                reference_layers = layers_a
            elif not np.array_equal(reference_layers, layers_a):
                raise ValueError(
                    f"Inconsistent layer boundaries: {model}"
                )

            for k in (1, 8):
                overlap = np.asarray(
                    [
                        subspace_overlap(a, b, k)
                        for a, b in zip(vectors_a, vectors_b)
                    ]
                )
                results[k][label].append(overlap)

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True
    )

    for ax, k in zip(axes, (1, 8)):
        for _, _, label in COMPARISONS:
            values = np.stack(results[k][label], axis=0)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)

            line, = ax.plot(
                reference_layers,
                mean,
                linewidth=1.8,
                label=label,
            )
            ax.fill_between(
                reference_layers,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=line.get_color(),
                alpha=0.14,
                linewidth=0,
            )

        ax.set_title(f"Top-{k} SVD overlap")
        ax.set_xlabel("Decoder layer boundary")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("Subspace overlap")

    fig.tight_layout()

    output = OUTPUT / f"03_overlap_{model}.pdf"
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")
    print("  Left: top-1 squared cosine overlap")
    print("  Right: top-8 subspace overlap")
    print("  Shading: five-fold standard deviation")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for model, component in MODELS.items():
        plot_model(model, component)


if __name__ == "__main__":
    main()
