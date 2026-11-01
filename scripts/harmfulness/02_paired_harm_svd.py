#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

COMPONENTS = {
    "qwen2_audio": {
        "text": ("decoder",),
        "audio": ("decoder", "audio_encoder"),
        "combined": ("decoder",),
    },
    "kimi_audio": {
        "text": ("shared_decoder", "text_mimo"),
        "audio": ("shared_decoder", "text_mimo", "audio_encoder"),
        "combined": ("shared_decoder",),
    },
    "qwen25_omni": {
        "text": ("thinker",),
        "audio": ("thinker", "audio_encoder"),
        "combined": ("thinker",),
    },
}

LANGUAGES = ("chinese", "english", "russian", "spanish")
CONDITIONS = ("content_dilution", "literal", "narrative_framing", "original")
K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
METHODS = ("mean",) + tuple(f"svd_{k}" for k in K_VALUES)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fold_assignments(model):
    directory = RESULTS / model / "wk3/dataset_summary/folds"
    assignments = {}

    for fold in range(1, 6):
        test = read_csv(directory / f"fold_{fold}_test.csv")
        train = read_csv(directory / f"fold_{fold}_train.csv")

        if len(test) != 1280 or len(train) != 5120:
            raise ValueError(f"{model}: incorrect fold {fold} sizes")

        test_topics = {
            int(row["base_id"].rsplit("_", 1)[1])
            for row in test
        }
        train_topics = {
            int(row["base_id"].rsplit("_", 1)[1])
            for row in train
        }

        if (
            len(test_topics) != 20
            or len(train_topics) != 80
            or test_topics & train_topics
            or test_topics | train_topics != set(range(100))
        ):
            raise ValueError(f"{model}: invalid topic split in fold {fold}")

        for topic in test_topics:
            if topic in assignments:
                raise ValueError(f"{model}: duplicate test topic {topic}")
            assignments[topic] = fold

    if set(assignments) != set(range(100)):
        raise ValueError(f"{model}: incomplete fold assignments")

    return assignments


def task_id(split, topic, condition, language, modality):
    return (
        f"jbb_{split}_{topic:03d}__"
        f"{condition}__{language}__{modality}"
    )


def load_representations(model, modality, component):
    directory = RESULTS / model / "5fold/representations_shared"

    benign = []
    harmful = []
    metadata = []

    for topic in range(100):
        for language in LANGUAGES:
            for condition in CONDITIONS:
                pair = []

                for split in ("benign", "harmful"):
                    identifier = task_id(
                        split, topic, condition, language, modality
                    )
                    path = directory / f"{identifier}.pt"

                    record = torch.load(
                        path, map_location="cpu", weights_only=True
                    )

                    if record["task_id"] != identifier:
                        raise ValueError(f"Task mismatch: {path}")

                    tensor = record[component].float()

                    if component == "audio_encoder":
                        if tensor.ndim != 3 or tensor.shape[1] != 8:
                            raise ValueError(f"Unexpected encoder shape: {path}")
                        tensor = tensor.mean(dim=1)

                    if tensor.ndim != 2 or not torch.isfinite(tensor).all():
                        raise ValueError(f"Invalid representation: {path}")

                    pair.append(tensor)

                if pair[0].shape != pair[1].shape:
                    raise ValueError(f"Pair shape mismatch: topic {topic}")

                benign.append(pair[0])
                harmful.append(pair[1])
                metadata.append((topic, language, condition))

        if (topic + 1) % 20 == 0:
            print(
                f"  {model} {modality}/{component}: "
                f"loaded {topic + 1}/100 topics",
                flush=True,
            )

    return torch.stack(benign), torch.stack(harmful), metadata


def fit_layer(differences, seed):
    mean = differences.mean(dim=0)

    torch.manual_seed(seed)
    q = min(136, min(differences.shape))

    _, singular_values, right_vectors = torch.svd_lowrank(
        differences, q=q, niter=3
    )

    basis = right_vectors[:, :128].T.contiguous()
    values = singular_values[:128]

    if basis.shape[0] != 128:
        raise ValueError("SVD rank is below 128")

    return mean, basis, values


def unit(vector):
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or norm <= 1e-10:
        raise ValueError("Invalid direction norm")
    return vector / norm


def direction(mean, basis, method):
    if method == "mean":
        return unit(mean)

    k = int(method.split("_")[1])
    top = basis[:k]
    return unit((top @ mean) @ top)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        path, index=False, float_format="%.3f"
    )
    print(f"SAVED {path}", flush=True)


def run_group(model, source, component, assignments, device):
    if source == "combined":
        tb, th, tm = load_representations(model, "text", component)
        ab, ah, am = load_representations(model, "audio", component)

        if tm != am or tb.shape != ab.shape or th.shape != ah.shape:
            raise ValueError("Combined representation mismatch")

        groups = ((tb, th), (ab, ah))
        metadata = tm

    else:
        b, h, metadata = load_representations(
            model, source, component
        )
        groups = ((b, h),)

    output = (
        RESULTS / model / "wk3/paired_harm_svd"
        / source / component
    )
    output.mkdir(parents=True, exist_ok=True)

    all_energy = []

    for fold in range(1, 6):
        train = [
            i for i, (topic, _, _) in enumerate(metadata)
            if assignments[topic] != fold
        ]
        test = [
            i for i, (topic, _, _) in enumerate(metadata)
            if assignments[topic] == fold
        ]

        if len(train) != 1280 or len(test) != 320:
            raise ValueError(f"Incorrect fold partition: {fold}")

        total_layers = groups[0][0].shape[1]
        layers = list(range(1, total_layers))

        means = []
        vectors = []
        singular_values = []
        energy_rows = []

        print(
            f"\n{model} {source}/{component} FOLD {fold}/5",
            flush=True,
        )

        for layer in layers:
            train_parts = []
            test_parts = []

            for benign, harmful in groups:
                train_parts.append(
                    (harmful[train, layer] - benign[train, layer]).to(device)
                )
                test_parts.append(
                    (harmful[test, layer] - benign[test, layer]).to(device)
                )

            d_train = torch.cat(train_parts)
            d_test = torch.cat(test_parts)

            mean, basis, values = fit_layer(
                d_train, seed=42000 + fold * 100 + layer
            )

            train_energy = d_train.square().sum()
            test_energy = d_test.square().sum()

            if train_energy <= 0 or test_energy <= 0:
                raise ValueError("Zero difference energy")

            cumulative = torch.cumsum(values.square(), dim=0)
            cumulative = cumulative / train_energy

            for method in METHODS:
                w = direction(mean, basis, method)

                train_projection = d_train @ w
                test_projection = d_test @ w

                if method == "mean":
                    k = 0
                    cumulative_energy = float("nan")
                else:
                    k = int(method.split("_")[1])
                    cumulative_energy = cumulative[k - 1].item()

                energy_rows.append({
                    "fold": fold,
                    "layer": layer,
                    "method": method,
                    "k": k,
                    "train_projected_energy_fraction": (
                        train_projection.square().sum() / train_energy
                    ).item(),
                    "test_projected_energy_fraction": (
                        test_projection.square().sum() / test_energy
                    ).item(),
                    "train_mean_energy_fraction": (
                        d_train.shape[0] * mean.square().sum()
                        / train_energy
                    ).item(),
                    "test_mean_signed_projection": (
                        test_projection.mean()
                    ).item(),
                    "cumulative_explained_energy": cumulative_energy,
                })

            means.append(mean.cpu().numpy().astype(np.float32))
            vectors.append(basis.cpu().numpy().astype(np.float32))
            singular_values.append(
                values.cpu().numpy().astype(np.float64)
            )

            del d_train, d_test, train_parts, test_parts

            if layer % 4 == 0 or layer == layers[-1]:
                print(
                    f"  fold {fold}: layer {layer}/{layers[-1]}",
                    flush=True,
                )

        np.savez_compressed(
            output / f"fold_{fold}_directions.npz",
            means=np.stack(means),
            vectors=np.stack(vectors),
            singular_values=np.stack(singular_values),
            layer_boundaries=np.asarray(layers, dtype=np.int64),
            k_values=np.asarray(K_VALUES, dtype=np.int64),
        )

        write_csv(output / f"fold_{fold}_energy.csv", energy_rows)
        all_energy.extend(energy_rows)

    df = pd.DataFrame(all_energy)

    measurements = [
        "train_projected_energy_fraction",
        "test_projected_energy_fraction",
        "train_mean_energy_fraction",
        "test_mean_signed_projection",
        "cumulative_explained_energy",
    ]

    summary = df.groupby(["layer", "method", "k"], sort=False)[
        measurements
    ].agg(["mean", "std"])

    summary.columns = [
        f"{name}_{statistic}"
        for name, statistic in summary.columns
    ]

    write_csv(
        output / "energy_five_fold_mean_std.csv",
        summary.reset_index().to_dict("records"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=tuple(COMPONENTS)
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")

    torch.set_num_threads(4)

    assignments = fold_assignments(args.model)

    for source, components in COMPONENTS[args.model].items():
        for component in components:
            run_group(
                args.model, source, component,
                assignments, device
            )

    print(f"\nCOMPLETE: {args.model}", flush=True)


if __name__ == "__main__":
    main()
