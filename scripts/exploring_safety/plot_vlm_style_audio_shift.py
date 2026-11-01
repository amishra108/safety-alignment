#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")


def safe_name(x):
    return str(x).replace("/", "_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=("qwen25_omni", "qwen2_audio", "kimi_audio"),
        required=True,
    )
    args = p.parse_args()

    model = args.model

    root = (
        ROOT
        / "results/INITIAL_BENCHMARK"
        / model
        / "cross_modal_analysis"
    )

    inp = root / "VLM_STYLE_independent_direction_contrasts.csv"
    df = pd.read_csv(inp)

    out = root / "VLM_STYLE_plots"
    out.mkdir(parents=True, exist_ok=True)

    for family in df["family"].unique():

        d = df[df["family"] == family].copy()

        # ==========================================================
        # 1. MAIN FIGURE: conditional AUROC
        # ==========================================================

        fig, ax = plt.subplots(figsize=(7.0, 4.3))

        names = (
            ("audio_outcome_AUROC_given_text0",
             "Text refused"),
            ("audio_outcome_AUROC_given_text1",
             "Text complied"),
        )

        for key, label in names:
            x = d[d["contrast"] == key].sort_values("layer")
            ax.plot(
                x["layer"],
                x["mean_difference"],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=label,
            )

        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC")
        # Automatic y-limits so models with stronger/weaker effects are not clipped.
        vals = []
        for key, _ in names:
            xx = d[d["contrast"] == key]["mean_difference"].to_numpy(dtype=float)
            vals.extend(xx[np.isfinite(xx)])

        if vals:
            ymin = min(min(vals), 0.5)
            ymax = max(max(vals), 0.5)
            pad = max(0.03, 0.10 * (ymax - ymin if ymax > ymin else 0.1))
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

        fig.tight_layout(pad=1.4)

        path = out / f"{safe_name(family)}_VLM_STYLE_AUROC.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        print("[WROTE]", path)

        # ==========================================================
        # 2. MECHANISTIC FIGURE: projection contrasts + 95% CI
        # ==========================================================

        fig, ax = plt.subplots(figsize=(7.0, 4.3))

        names = (
            ("C01_minus_C00", r"$C_{01}-C_{00}$"),
            ("C11_minus_C10", r"$C_{11}-C_{10}$"),
        )

        for key, label in names:

            x = (
                d[d["contrast"] == key]
                .sort_values("layer")
                .copy()
            )

            valid = np.isfinite(x["mean_difference"])

            x = x[valid]

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
            "Difference in harmful-compliance\nshift projection"
        )
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

        fig.tight_layout(pad=1.4)

        path = out / f"{safe_name(family)}_VLM_STYLE_CONTRASTS.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        print("[WROTE]", path)


if __name__ == "__main__":
    main()
