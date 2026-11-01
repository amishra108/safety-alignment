#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

FAMILY = {
    "qwen25_omni": "thinker",
    "qwen2_audio": "decoder",
    "kimi_audio": "shared_decoder",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=tuple(FAMILY),
        required=True,
    )
    args = p.parse_args()

    model = args.model
    family = FAMILY[model]

    root = (
        ROOT
        / "results/INITIAL_BENCHMARK"
        / model
        / "cross_modal_analysis"
    )

    inp = root / "VLM_STYLE_BENIGN_independent_direction_contrasts.csv"
    df = pd.read_csv(inp)

    d = df[df["family"] == family].copy()

    if d.empty:
        raise RuntimeError(f"No rows for {model}/{family} in {inp}")

    out = root / "VLM_STYLE_BENIGN_plots"
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. AUROC
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    curves = (
        ("audio_outcome_AUROC_given_text0", "Text refused"),
        ("audio_outcome_AUROC_given_text1", "Text complied"),
    )

    vals = []

    for key, label in curves:
        x = d[d["contrast"] == key].sort_values("layer")
        y = x["mean_difference"].to_numpy(dtype=float)

        ax.plot(
            x["layer"],
            y,
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=label,
        )

        vals.extend(y[np.isfinite(y)])

    ax.axhline(0.5, linestyle="--", linewidth=1)

    if vals:
        ymin = min(min(vals), 0.5)
        ymax = max(max(vals), 0.5)
        pad = max(0.03, 0.10 * max(ymax - ymin, 0.1))
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout(pad=1.4)

    path = out / f"{family}_VLM_STYLE_BENIGN_AUROC.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    print("[WROTE]", path)

    # ------------------------------------------------------------
    # 2. Projection contrasts
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    curves = (
        ("C01_minus_C00", r"$C_{01}-C_{00}$"),
        ("C11_minus_C10", r"$C_{11}-C_{10}$"),
    )

    for key, label in curves:
        x = (
            d[d["contrast"] == key]
            .sort_values("layer")
            .copy()
        )

        good = (
            np.isfinite(x["mean_difference"])
            & np.isfinite(x["approx_ci95_low"])
            & np.isfinite(x["approx_ci95_high"])
        )

        x = x[good]

        y = x["mean_difference"].to_numpy()
        lo = x["approx_ci95_low"].to_numpy()
        hi = x["approx_ci95_high"].to_numpy()

        ax.plot(
            x["layer"],
            y,
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=label,
        )

        ax.fill_between(
            x["layer"],
            lo,
            hi,
            alpha=0.15,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel(
        "Difference in benign-compliance\n"
        "shift projection"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout(pad=1.4)

    path = out / f"{family}_VLM_STYLE_BENIGN_CONTRASTS.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    print("[WROTE]", path)


if __name__ == "__main__":
    main()
