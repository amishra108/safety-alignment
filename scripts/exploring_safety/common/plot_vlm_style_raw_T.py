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

LABELS = {
    "00": r"$C_{00}$: text refuse, audio refuse",
    "01": r"$C_{01}$: text refuse, audio comply",
    "10": r"$C_{10}$: text comply, audio refuse",
    "11": r"$C_{11}$: text comply, audio comply",
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

    inp = root / "VLM_STYLE_independent_direction_per_example.csv"
    df = pd.read_csv(inp)

    df = df[df["family"] == family].copy()

    if df.empty:
        raise RuntimeError(f"No rows for {model}/{family}: {inp}")

    score = "projection_toward_text_jailbreak_direction"

    # OOF mean T and ordinary example-level SE at each layer/group.
    g = (
        df.groupby(["layer", "outcome_group"])[score]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    g["se"] = g["std"] / np.sqrt(g["count"])
    g["ci"] = 1.96 * g["se"]

    out = root / "VLM_STYLE_plots"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    for group in ("00", "01", "10", "11"):
        x = (
            g[g["outcome_group"].astype(str).str.zfill(2) == group]
            .sort_values("layer")
        )

        if x.empty:
            continue

        layer = x["layer"].to_numpy()
        mean = x["mean"].to_numpy()
        ci = x["ci"].to_numpy()

        ax.plot(
            layer,
            mean,
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=LABELS[group],
        )

        ax.fill_between(
            layer,
            mean - ci,
            mean + ci,
            alpha=0.12,
        )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("Layer")
    ax.set_ylabel(
        r"Mean modality shift projection $T$"
        "\n(+ toward harmful compliance)"
    )

    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    ax.grid(alpha=0.2)

    fig.tight_layout(pad=1.4)

    path = out / f"{family}_VLM_STYLE_RAW_T.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    print("[WROTE]", path)

    print("\n===== FINAL AVAILABLE LAYER =====")
    last = int(g["layer"].max())

    for group in ("00", "01", "10", "11"):
        x = g[
            (g["layer"] == last)
            & (g["outcome_group"].astype(str).str.zfill(2) == group)
        ]

        if len(x):
            r = x.iloc[0]
            print(
                f"{group}: N={int(r['count']):4d} "
                f"mean T={r['mean']:+.5f} "
                f"95% CI=[{r['mean']-r['ci']:+.5f}, "
                f"{r['mean']+r['ci']:+.5f}]"
            )


if __name__ == "__main__":
    main()
