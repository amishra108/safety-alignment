#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(
    "results/INITIAL_BENCHMARK/kimi_audio/decoding_layers/"
    "kimi_audio_adjudicated_layer_behaviour.pdf"
)

layers = np.arange(29)
groups = ["00", "01", "10", "11"]

# ============================================================
# BENIGN — BRR
#
# N:
# 00 = 25
# 01 = 25
# 10 = 14
# 11 = 24
#
# Layers 0-22: all 0%.
# ============================================================

benign_text = {
    "00": [0]*23 + [0, 0, 4.0, 0, 4.0, 0],
    "01": [0]*23 + [0, 20, 28, 24, 4, 0],
    "10": [0]*23 + [7.1, 28.6, 42.9, 35.7, 35.7, 92.9],
    "11": [0]*23 + [29.2, 75, 75, 91.7, 75, 100],
}

benign_audio = {
    "00": [0]*23 + [0, 0, 4.0, 4.0, 0, 0],
    "01": [0]*23 + [16, 40, 40, 40, 60, 100],
    "10": [0]*23 + [0, 7.1, 0, 14.3, 14.3, 0],
    "11": [0]*23 + [33.3, 75, 87.5, 75, 95.8, 100],
}

# ============================================================
# HARMFUL — ASR
#
# N = 25 per outcome group.
#
# Layers 0-23: all 0%.
# ============================================================

harmful_text = {
    "00": [0]*24 + [0, 0, 0, 0, 4],
    "01": [0]*24 + [8, 12, 16, 12, 8],
    "10": [0]*24 + [8, 20, 36, 60, 72],
    "11": [0]*24 + [16, 40, 56, 72, 76],
}

harmful_audio = {
    "00": [0]*24 + [0, 0, 0, 0, 0],
    "01": [0]*24 + [12, 20, 24, 36, 32],
    "10": [0]*24 + [4, 8, 12, 8, 4],
    "11": [0]*24 + [16, 44, 60, 72, 76],
}

for name, data in [
    ("benign_text", benign_text),
    ("benign_audio", benign_audio),
    ("harmful_text", harmful_text),
    ("harmful_audio", harmful_audio),
]:
    for group in groups:
        assert len(data[group]) == 29, (
            name, group, len(data[group])
        )
        assert all(0 <= x <= 100 for x in data[group])

fig, axes = plt.subplots(
    2, 2,
    figsize=(11.2, 7.4),
    sharex=True,
    sharey=True,
)

panels = [
    (axes[0, 0], "Benign — text",  "BRR (%)", benign_text),
    (axes[0, 1], "Benign — audio", "BRR (%)", benign_audio),
    (axes[1, 0], "Harmful — text",  "ASR (%)", harmful_text),
    (axes[1, 1], "Harmful — audio", "ASR (%)", harmful_audio),
]

for ax, title, ylabel, data in panels:
    for group in groups:
        ax.plot(
            layers,
            data[group],
            marker="o",
            markersize=3.0,
            linewidth=1.65,
            label=group,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Shared-decoder layer")

    ax.set_xlim(0, 28)
    ax.set_ylim(-3, 103)

    ax.set_xticks(np.arange(0, 29, 2))
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(
        True,
        alpha=0.22,
        linewidth=0.7,
    )

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Outcome group\n(text, audio)",
    loc="center right",
    bbox_to_anchor=(0.995, 0.5),
    frameon=True,
)

fig.suptitle(
    "Kimi-Audio forced-decoding safety behaviour",
    fontsize=13,
)

fig.text(
    0.5,
    0.012,
    "BRR: refusal of benign task. "
    "ASR: material compliance with harmful task. "
    "Outcome groups denote original text/audio decisions.",
    ha="center",
    fontsize=8.5,
)

fig.tight_layout(
    rect=[0, 0.045, 0.88, 0.95]
)

fig.savefig(
    OUT,
    format="pdf",
    bbox_inches="tight",
)

print(f"saved: {OUT}")
