#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

spec = importlib.util.spec_from_file_location(
    "paired_harm_svd",
    ROOT / "scripts/wk3/02_paired_harm_svd.py",
)
svd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svd)

METHODS = ("mean", "svd_1", "svd_8")


def load_directions(model, source, component, fold, device):
    path = (
        RESULTS / model / "wk3/paired_harm_svd"
        / source / component / f"fold_{fold}_directions.npz"
    )

    with np.load(path, allow_pickle=False) as saved:
        means = torch.from_numpy(saved["means"].copy()).to(device)
        vectors = torch.from_numpy(saved["vectors"].copy()).to(device)
        layers = saved["layer_boundaries"].tolist()

    return means, vectors, layers


def project_group(model, modality, component, assignments, device):
    benign, harmful, metadata = svd.load_representations(
        model, modality, component
    )

    primary = {
        "qwen2_audio": "decoder",
        "kimi_audio": "shared_decoder",
        "qwen25_omni": "thinker",
    }[model]

    sources = [modality]

    if component == primary:
        sources.append("combined")
        sources.append("audio" if modality == "text" else "text")

    output = (
        RESULTS / model / "wk3/paired_harm_projections"
        / modality / component
    )
    output.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 6):
        indices = [
            i for i, (topic, _, _) in enumerate(metadata)
            if assignments[topic] == fold
        ]

        if len(indices) != 320:
            raise ValueError(f"Expected 320 test pairs in fold {fold}")

        rows = []

        for source in sources:
            means, vectors, layers = load_directions(
                model, source, component, fold, device
            )

            if (
                len(layers) != len(means)
                or len(layers) != len(vectors)
                or any(layer >= benign.shape[1] for layer in layers)
            ):
                raise ValueError("Saved directions and layers mismatch")

            for position, layer in enumerate(layers):
                directions = torch.stack(
                    [
                        svd.direction(
                            means[position],
                            vectors[position],
                            method,
                        )
                        for method in METHODS
                    ],
                    dim=1,
                )

                b_scores = (
                    benign[indices, layer].to(device) @ directions
                ).cpu().numpy()

                h_scores = (
                    harmful[indices, layer].to(device) @ directions
                ).cpu().numpy()

                for method_index, method in enumerate(METHODS):
                    b = b_scores[:, method_index]
                    h = h_scores[:, method_index]

                    mu = b.mean()
                    sd = b.std(ddof=1)

                    if not np.isfinite(sd) or sd <= 0:
                        raise ValueError(
                            f"Invalid normalization: "
                            f"{model} {source} {fold} {layer} {method}"
                        )

                    for local_index, global_index in enumerate(indices):
                        topic, language, condition = metadata[global_index]

                        rows.append({
                            "fold": fold,
                            "layer": layer,
                            "direction_source": source,
                            "method": method,
                            "topic": topic,
                            "language": language,
                            "condition": condition,
                            "benign_score": float(b[local_index]),
                            "harmful_score": float(h[local_index]),
                            "benign_z": float(
                                (b[local_index] - mu) / sd
                            ),
                            "harmful_z": float(
                                (h[local_index] - mu) / sd
                            ),
                            "benign_reference_mean": float(mu),
                            "benign_reference_std": float(sd),
                        })

                if layer % 4 == 0 or layer == layers[-1]:
                    print(
                        f"{model} {modality}/{component} "
                        f"fold={fold} source={source} "
                        f"layer={layer}/{layers[-1]}",
                        flush=True,
                    )

        df = pd.DataFrame(rows)

        order = {
            "mean": 0,
            "svd_1": 1,
            "svd_8": 2,
        }
        df["_order"] = df["method"].map(order)

        df = df.sort_values(
            [
                "fold", "layer", "direction_source",
                "_order", "topic", "language", "condition"
            ],
            kind="stable",
        ).drop(columns="_order")

        path = output / f"fold_{fold}_heldout_raw_and_z_projections.csv"

        df.to_csv(path, index=False, float_format="%.3f")
        print(f"SAVED {path} ({len(df)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=tuple(svd.COMPONENTS)
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")

    torch.set_num_threads(4)

    assignments = svd.fold_assignments(args.model)

    for modality in ("text", "audio"):
        for component in svd.COMPONENTS[args.model][modality]:
            project_group(
                args.model,
                modality,
                component,
                assignments,
                device,
            )

    print(f"\nCOMPLETE: {args.model}", flush=True)


if __name__ == "__main__":
    main()
