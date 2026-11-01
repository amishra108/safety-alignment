#!/usr/bin/env python3

import argparse
import csv
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"

MODELS = {
    "qwen25_omni": {
        "families": {
            "thinker": {
                "key": "thinker",
                "layers": 29,
                "dim": 3584,
            },
        },
    },
    "qwen2_audio": {
        "families": {
            "decoder": {
                "key": "decoder",
                "layers": 33,
                "dim": 4096,
            },
        },
    },
    "kimi_audio": {
        "families": {
            "shared_decoder": {
                "key": "shared_decoder",
                "layers": 29,
                "dim": 3584,
            },
            "text_mimo": {
                "key": "text_mimo",
                "layers": 7,
                "dim": 3584,
            },
        },
    },
}

PCA_K = (32, 64, 128, 256, 512, 1024)
REMOVE_TOP = (1, 5, 10, 20)
SEED = 1729


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing file: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Bad JSON {path}:{ln}") from exc
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[EMPTY] {path}", flush=True)
        return

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tmp.replace(path)
    print(f"[WROTE] {path} rows={len(rows)}", flush=True)


def matched_key(r):
    return (
        str(r["base_prompt_id"]),
        str(r["language"]),
        str(r["condition"]),
    )


def outcome(td, ad):
    return f"text{int(td)}_audio{int(ad)}"


# ---------------------------------------------------------------------
# REPRESENTATIONS
# ---------------------------------------------------------------------

def load_family_tensor(rep_root, rows, fs):
    """
    Load one family ONCE for all 6400 rows.

    Stored as float16 on CPU:
        [N, L, D]

    Per-layer operations convert only one [N,D] slice to float32.
    """
    n = len(rows)
    L = fs["layers"]
    D = fs["dim"]

    X = torch.empty(
        (n, L, D),
        dtype=torch.float16,
        device="cpu",
    )

    print(
        f"[LOAD] family={fs['key']} N={n} shape=({L},{D})",
        flush=True,
    )

    for i, row in enumerate(rows):
        p = rep_root / f"{row['task_id']}.pt"

        if not p.exists():
            raise RuntimeError(f"Missing representation: {p}")

        obj = torch.load(
            p,
            map_location="cpu",
            weights_only=True,
        )

        x = obj.get(fs["key"])
        if not torch.is_tensor(x):
            raise RuntimeError(
                f"{p}: missing tensor {fs['key']} "
                f"keys={list(obj.keys())}"
            )

        if tuple(x.shape) != (L, D):
            raise RuntimeError(
                f"{p}: {fs['key']} shape={tuple(x.shape)} "
                f"expected={(L,D)}"
            )

        X[i].copy_(x.to(torch.float16))

        if (i + 1) % 500 == 0 or i + 1 == n:
            print(
                f"  loaded {i+1}/{n}",
                flush=True,
            )

    return X


# ---------------------------------------------------------------------
# INDEXING
# ---------------------------------------------------------------------

def indices(rows, role=None, safety=None, modality=None,
            language=None, condition=None):
    out = []

    for i, r in enumerate(rows):
        if role is not None and r["train_test"] != role:
            continue
        if safety is not None and r["safety_split"] != safety:
            continue
        if modality is not None and r["modality"] != modality:
            continue
        if language is not None and r["language"] != language:
            continue
        if condition is not None and r["condition"] != condition:
            continue
        out.append(i)

    return out


def pair_records(rows, role, safety=None):
    text = {}
    audio = {}

    for i, r in enumerate(rows):
        if r["train_test"] != role:
            continue
        if safety is not None and r["safety_split"] != safety:
            continue

        k = matched_key(r)

        if r["modality"] == "text":
            text[k] = (i, r)
        elif r["modality"] == "audio":
            audio[k] = (i, r)

    common = sorted(set(text) & set(audio))
    pairs = []

    for k in common:
        ti, tr = text[k]
        ai, ar = audio[k]

        if tr["safety_split"] != ar["safety_split"]:
            raise RuntimeError(f"pair safety mismatch: {k}")

        td = int(tr["decision"])
        ad = int(ar["decision"])

        pairs.append({
            "key": k,
            "text_index": ti,
            "audio_index": ai,
            "text_row": tr,
            "audio_row": ar,
            "text_decision": td,
            "audio_decision": ad,
            "group": outcome(td, ad),
            "safety_split": tr["safety_split"],
            "language": tr["language"],
            "condition": tr["condition"],
        })

    return pairs


def random_partner_map(model, fold, pairs):
    """
    Random audio control within exact:
      safety × language × condition × outcome_group.

    Derangement where N>=2. N=1 -> no random control.
    """
    strata = defaultdict(list)

    for p in pairs:
        strata[
            (
                p["safety_split"],
                p["language"],
                p["condition"],
                p["group"],
            )
        ].append(p)

    mapping = {}

    for stratum, ps in sorted(strata.items(), key=lambda z: str(z[0])):
        ps = sorted(ps, key=lambda p: str(p["key"]))

        if len(ps) < 2:
            mapping[ps[0]["key"]] = None
            continue

        seed_text = (
            f"{SEED}|{model}|{fold}|"
            + "|".join(map(str, stratum))
        )
        seed = int.from_bytes(
            hashlib.sha256(seed_text.encode()).digest()[:8],
            "big",
        )

        rng = np.random.default_rng(seed)
        order = np.arange(len(ps))

        for _ in range(10000):
            shuffled = rng.permutation(len(ps))
            if np.all(shuffled != order):
                break
        else:
            shuffled = np.roll(order, -1)

        for i, j in enumerate(shuffled):
            mapping[ps[i]["key"]] = ps[int(j)]

    return mapping


# ---------------------------------------------------------------------
# PROBES
# ---------------------------------------------------------------------

def class_balanced_probe(
    X_train,
    y_train,
    device,
    l2=1e-4,
    max_iter=150,
    standardise=True,
):
    """
    Same logic as original probes:
      - train-only feature standardisation
      - equal total class weight
      - LBFGS
      - return classifier in INPUT coordinates
    """
    X = X_train.to(device=device, dtype=torch.float32)
    y = y_train.to(device=device, dtype=torch.float32)

    if standardise:
        mean = X.mean(0)
        std = X.std(0, unbiased=False)
        std = torch.where(
            std < 1e-6,
            torch.ones_like(std),
            std,
        )
        Xn = (X - mean) / std
    else:
        # Representation is already in the common geometry being tested.
        # Do NOT introduce a separate modality-specific rescaling.
        mean = torch.zeros(
            X.shape[1],
            device=device,
            dtype=X.dtype,
        )
        std = torch.ones(
            X.shape[1],
            device=device,
            dtype=X.dtype,
        )
        Xn = X

    n0 = (y == 0).sum().float()
    n1 = (y == 1).sum().float()

    if n0.item() == 0 or n1.item() == 0:
        raise RuntimeError(
            f"Probe training missing class: n0={n0.item()} n1={n1.item()}"
        )

    n = y.numel()
    sample_weight = torch.where(
        y == 1,
        0.5 / n1,
        0.5 / n0,
    ) * n

    w = torch.zeros(
        X.shape[1],
        device=device,
        requires_grad=True,
    )
    b = torch.zeros(
        (),
        device=device,
        requires_grad=True,
    )

    opt = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = Xn @ w + b
        loss0 = F.binary_cross_entropy_with_logits(
            logits,
            y,
            reduction="none",
        )
        loss = (
            (loss0 * sample_weight).mean()
            + 0.5 * l2 * w.square().sum()
        )
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        w_raw = w / std
        b_raw = b - (mean / std * w).sum()

    return (
        w_raw.detach().cpu(),
        float(b_raw.detach().cpu()),
    )


def evaluate_probe(X, y, w, b):
    logits = (
        X.float() @ w.float()
        + float(b)
    ).cpu().numpy()

    labels = y.cpu().numpy().astype(np.int64)

    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    pred = (probs >= 0.5).astype(np.int64)

    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, pred)
        ),
    }


def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)

    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)

    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")

    return float(
        torch.dot(a, b) / (na * nb)
    )


def row_labels(rows, idx):
    return torch.tensor(
        [int(rows[i]["decision"]) for i in idx],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------
# SPECTRAL TRANSFORMS
# ---------------------------------------------------------------------

@torch.no_grad()
def covariance_eigh(X_cpu, device, ridge_fraction=1e-6):
    """
    Exact full covariance eigendecomposition.

      X_cpu: [N,D]

    Returns CPU:
      mean [D]
      eigenvalues descending [D]
      eigenvectors columns [D,D]

    A tiny isotropic ridge prevents singular whitening.
    """
    X = X_cpu.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    mean = X.mean(0)
    Xc = X - mean

    cov = (Xc.T @ Xc) / max(1, X.shape[0] - 1)

    scale = torch.trace(cov) / cov.shape[0]
    ridge = torch.clamp(
        scale * ridge_fraction,
        min=1e-8,
    )

    cov.diagonal().add_(ridge)

    evals, evecs = torch.linalg.eigh(cov)

    order = torch.arange(
        evals.numel() - 1,
        -1,
        -1,
        device=device,
    )

    evals = evals[order]
    evecs = evecs[:, order]

    return (
        mean.detach().cpu(),
        evals.detach().cpu(),
        evecs.detach().cpu(),
        float(ridge.detach().cpu()),
    )


@torch.no_grad()
def apply_whitening(X, mean, evals, evecs, device):
    Xd = X.to(device=device, dtype=torch.float32)
    md = mean.to(device)
    vd = evecs.to(device)
    ld = evals.to(device)

    Z = (Xd - md) @ vd
    Z = Z * torch.rsqrt(torch.clamp(ld, min=1e-12))

    return Z.cpu()


@torch.no_grad()
def fisher_information_eigh(
    X_cpu,
    w,
    b,
    device,
    ridge_fraction=1e-2,
):
    """
    Fisher information matrix for a binary logistic safety probe,
    with respect to the probe's representation-space weight vector.

        p_i = sigmoid(h_i^T w + b)

        F = (1/N) sum_i p_i(1-p_i) h_i h_i^T

    The probe and Fisher matrix are fitted on training data only.

    Returns CPU:
      eigenvalues descending [D]
      eigenvectors columns [D,D]
      ridge
      mean p(1-p)
    """
    X = X_cpu.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    wd = w.to(
        device=device,
        dtype=torch.float32,
    )

    logits = X @ wd + float(b)
    probs = torch.sigmoid(logits)

    # Bernoulli likelihood curvature.
    curvature = probs * (1.0 - probs)

    # F = X^T diag[p(1-p)] X / N
    weighted_X = X * curvature.unsqueeze(1)

    fisher = (
        X.T @ weighted_X
    ) / max(1, X.shape[0])

    # Isotropic numerical regularisation.
    # Regularised Fisher metric:
    #
    #   F_reg = F + lambda * trace(F)/D * I
    #
    # D can exceed N here, so the empirical Fisher is rank-deficient.
    # A materially sized isotropic term prevents null/near-null Fisher
    # directions from exploding under F^{-1/2}.
    scale = torch.trace(fisher) / fisher.shape[0]

    ridge = torch.clamp(
        scale * ridge_fraction,
        min=1e-8,
    )

    fisher.diagonal().add_(ridge)

    evals, evecs = torch.linalg.eigh(fisher)

    order = torch.arange(
        evals.numel() - 1,
        -1,
        -1,
        device=device,
    )

    evals = evals[order]
    evecs = evecs[:, order]

    return (
        evals.detach().cpu(),
        evecs.detach().cpu(),
        float(ridge.detach().cpu()),
        float(curvature.mean().detach().cpu()),
    )


@torch.no_grad()
def pca_scores(X, mean, evecs, k, device):
    Xd = X.to(device=device, dtype=torch.float32)
    md = mean.to(device)
    V = evecs[:, :k].to(device)
    return ((Xd - md) @ V).cpu()


@torch.no_grad()
def remove_top_pcs(X, mean, evecs, n_remove, device):
    Xd = X.to(device=device, dtype=torch.float32)
    md = mean.to(device)
    Xc = Xd - md

    V = evecs[:, :n_remove].to(device)
    Xr = Xc - (Xc @ V) @ V.T

    return Xr.cpu()


def transformed_old_weight(w, evals, evecs):
    """
    z = Lambda^-1/2 V^T (x-mu)

    Therefore classifier covector in z-space is:
        w_z = Lambda^1/2 V^T w_x
    """
    return (
        (evecs.T @ w.float())
        * torch.sqrt(torch.clamp(evals, min=1e-12))
    )


# ---------------------------------------------------------------------
# OLD WEIGHTS
# ---------------------------------------------------------------------

def load_old_weight(model_root, fold, safety, modality, family, layer):
    p = (
        model_root
        / "linear_probe_gpu"
        / "weights"
        / f"fold_{fold}"
        / safety
        / modality
        / family
        / f"layer_{layer:02d}.pt"
    )

    if not p.exists():
        raise RuntimeError(f"Missing old probe: {p}")

    obj = torch.load(
        p,
        map_location="cpu",
        weights_only=True,
    )

    return (
        obj["w"].float().reshape(-1),
        float(obj["b"]),
    )


# ---------------------------------------------------------------------
# SUMMARY HELPERS FOR COSINE
# ---------------------------------------------------------------------

def mean_std(vals):
    vals = [
        float(v) for v in vals
        if math.isfinite(float(v))
    ]

    if not vals:
        return float("nan"), float("nan"), 0

    x = np.asarray(vals, dtype=np.float64)

    return (
        float(x.mean()),
        float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        int(len(x)),
    )


def summarise_cosine(rows, keys):
    buckets = defaultdict(list)

    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(r)

    out = []

    for key, vals in sorted(
        buckets.items(),
        key=lambda z: str(z[0]),
    ):
        matched = [x["matched_cosine"] for x in vals]
        randomc = [x["random_unmatched_cosine"] for x in vals]
        delta = [x["matched_minus_random"] for x in vals]

        mm, ms, mn = mean_std(matched)
        rm, rs, rn = mean_std(randomc)
        dm, ds, dn = mean_std(delta)

        r = dict(zip(keys, key))
        r.update({
            "n_pairs": len(vals),
            "matched_cosine_mean": mm,
            "matched_cosine_std": ms,
            "random_unmatched_cosine_mean": rm,
            "random_unmatched_cosine_std": rs,
            "matched_minus_random_mean": dm,
            "matched_minus_random_std": ds,
            "n_finite_differences": dn,
        })
        out.append(r)

    return out



def pooled_group_stats(rows):
    """
    Pool disjoint OOF summary cells using n, mean and within-cell sample SD.
    """
    if not rows:
        return None

    N = sum(int(r["n"]) for r in rows)

    if N <= 0:
        return None

    mu = (
        sum(
            int(r["n"])
            * float(r["mean_audio_minus_text"])
            for r in rows
        )
        / N
    )

    if N <= 1:
        sd = 0.0
    else:
        ss_within = 0.0
        ss_between = 0.0

        for r in rows:
            n = int(r["n"])
            m = float(r["mean_audio_minus_text"])
            sd_i = float(r["std_audio_minus_text"])

            if n > 1:
                ss_within += (n - 1) * sd_i * sd_i

            ss_between += n * (m - mu) ** 2

        sd = math.sqrt(
            max(
                0.0,
                (ss_within + ss_between) / (N - 1),
            )
        )

    return N, mu, sd


def build_trajectory_contrasts(trajectory_rows):
    """
    Pool all 5 held-out folds into OOF behavioural contrasts.

      01 - 00
      10 - 11

    Language/condition cells contribute according to their actual N.
    """
    out = []

    for safety, source_rows in trajectory_rows.items():
        grouped = defaultdict(list)

        for r in source_rows:
            key = (
                r["model"],
                r["family"],
                r["probe_modality"],
                int(r["layer"]),
                r["disagreement_group"],
            )
            grouped[key].append(r)

        bases = sorted({
            (m, f, pm, layer)
            for (m, f, pm, layer, group) in grouped
        })

        for model, family, probe_modality, layer in bases:

            def get(group):
                return pooled_group_stats(
                    grouped.get(
                        (
                            model,
                            family,
                            probe_modality,
                            layer,
                            group,
                        ),
                        [],
                    )
                )

            for lhs, rhs, name in (
                (
                    "text0_audio1",
                    "text0_audio0",
                    "01_minus_00",
                ),
                (
                    "text1_audio0",
                    "text1_audio1",
                    "10_minus_11",
                ),
            ):
                A = get(lhs)
                B = get(rhs)

                if A is None or B is None:
                    continue

                n1, m1, s1 = A
                n0, m0, s0 = B

                contrast = m1 - m0

                # Approximate two-group SE using pooled OOF SDs.
                se = math.sqrt(
                    (s1 * s1 / n1 if n1 else 0.0)
                    +
                    (s0 * s0 / n0 if n0 else 0.0)
                )

                out.append({
                    "model": model,
                    "family": family,
                    "safety_split": safety,
                    "probe_modality": probe_modality,
                    "layer": layer,
                    "contrast": name,
                    "lhs_group": lhs,
                    "rhs_group": rhs,
                    "lhs_n_total": n1,
                    "rhs_n_total": n0,
                    "lhs_mean_audio_minus_text": m1,
                    "rhs_mean_audio_minus_text": m0,
                    "contrast_mean": contrast,
                    "contrast_se_approx": se,
                    "contrast_ci95_low":
                        contrast - 1.96 * se,
                    "contrast_ci95_high":
                        contrast + 1.96 * se,
                    "low_support":
                        int(min(n1, n0) < 20),
                })

    return out


# ---------------------------------------------------------------------
# MAIN MODEL
# ---------------------------------------------------------------------

def run_model(model_name, args):
    spec = MODELS[model_name]

    model_root = RESULTS / model_name
    fold_root = model_root / "5fold"
    rep_root = fold_root / "representations_shared"

    cross_root = model_root / "cross_modal_analysis"

    cosine_root = (
        model_root
        / "linear_probe_gpu"
        / "cosine_similarity"
        / "NORMALISED_cosine_similarity"
    )

    new_root = (
        model_root
        / "normalised_analysis"
    )

    cross_root.mkdir(parents=True, exist_ok=True)
    cosine_root.mkdir(parents=True, exist_ok=True)
    new_root.mkdir(parents=True, exist_ok=True)

    trajectory_rows = {
        "benign": [],
        "harmful": [],
    }

    whitening_transfer_rows = []
    cosine_pair_rows = []
    probe_cos_rows = []
    fisher_rows = []
    pca_rows = []
    pc_remove_cos_rows = []
    pc_remove_transfer_rows = []
    spectral_rows = []

    checkpoint_path = (
        new_root / "_NORMALISED_layer_checkpoint.pt"
    )

    completed_layers = set()

    if checkpoint_path.exists() and not args.restart:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        trajectory_rows = ckpt["trajectory_rows"]
        whitening_transfer_rows = (
            ckpt["whitening_transfer_rows"]
        )
        cosine_pair_rows = ckpt["cosine_pair_rows"]
        probe_cos_rows = ckpt["probe_cos_rows"]
        fisher_rows = ckpt["fisher_rows"]
        pca_rows = ckpt["pca_rows"]
        pc_remove_cos_rows = ckpt["pc_remove_cos_rows"]
        pc_remove_transfer_rows = (
            ckpt["pc_remove_transfer_rows"]
        )
        spectral_rows = ckpt["spectral_rows"]

        completed_layers = {
            tuple(x)
            for x in ckpt["completed_layers"]
        }

        print(
            f"[RESUME] {checkpoint_path} | "
            f"completed_layers={len(completed_layers)}",
            flush=True,
        )

    device = torch.device(args.device)

    print()
    print("=" * 110)
    print("MODEL:", model_name)
    print("DEVICE:", device)
    print("=" * 110)

    for fold in range(1, 6):
        rows = read_jsonl(
            fold_root / f"fold_{fold}.jsonl"
        )

        if len(rows) != 6400:
            raise RuntimeError(
                f"{model_name} fold {fold}: expected 6400 rows "
                f"got {len(rows)}"
            )

        print()
        print("=" * 110)
        print(f"{model_name} | FOLD {fold}")
        print("=" * 110)

        train_all = indices(rows, role="train")
        test_all = indices(rows, role="test")

        test_pairs_all = pair_records(
            rows,
            role="test",
            safety=None,
        )

        random_map = random_partner_map(
            model_name,
            fold,
            test_pairs_all,
        )

        for family, fs in spec["families"].items():
            print()
            print("#" * 110)
            print(
                f"{model_name} fold={fold} family={family} "
                f"L={fs['layers']} D={fs['dim']}"
            )
            print("#" * 110)

            Hfull = load_family_tensor(
                rep_root,
                rows,
                fs,
            )

            for layer in range(fs["layers"]):
                layer_key = (
                    model_name,
                    int(fold),
                    family,
                    int(layer),
                )

                if layer_key in completed_layers:
                    print(
                        f"[SKIP COMPLETED] {model_name} "
                        f"fold={fold} {family} L{layer:02d}",
                        flush=True,
                    )
                    continue

                print()
                print(
                    f"[LAYER] {model_name} fold={fold} "
                    f"{family} L{layer:02d}",
                    flush=True,
                )

                H = Hfull[:, layer, :].float()

                # =====================================================
                # 1. RECENTER OLD TRAJECTORY EXPERIMENT
                # =====================================================

                for safety in ("benign", "harmful"):
                    test_pairs = [
                        p for p in test_pairs_all
                        if p["safety_split"] == safety
                    ]

                    languages = sorted({
                        p["language"]
                        for p in test_pairs
                    })
                    conditions = sorted({
                        p["condition"]
                        for p in test_pairs
                    })

                    offsets = {}

                    for language in languages:
                        for condition in conditions:
                            tr_t = indices(
                                rows,
                                role="train",
                                safety=safety,
                                modality="text",
                                language=language,
                                condition=condition,
                            )
                            tr_a = indices(
                                rows,
                                role="train",
                                safety=safety,
                                modality="audio",
                                language=language,
                                condition=condition,
                            )

                            if not tr_t or not tr_a:
                                raise RuntimeError(
                                    f"Missing recenter train stratum: "
                                    f"{model_name} fold={fold} {safety} "
                                    f"{language} {condition}"
                                )

                            mu_t = H[tr_t].mean(0)
                            mu_a = H[tr_a].mean(0)

                            # text -> audio offset
                            offsets[(language, condition)] = (
                                mu_a - mu_t
                            )

                    for probe_modality in ("text", "audio"):
                        old_w, old_b = load_old_weight(
                            model_root,
                            fold,
                            safety,
                            probe_modality,
                            family,
                            layer,
                        )

                        grouped = defaultdict(list)

                        for p in test_pairs:
                            ht = H[p["text_index"]]

                            delta_mod = offsets[
                                (
                                    p["language"],
                                    p["condition"],
                                )
                            ]

                            # Align audio into text coordinate system.
                            ha = (
                                H[p["audio_index"]]
                                - delta_mod
                            )

                            lt = float(
                                torch.dot(old_w, ht)
                                + old_b
                            )
                            la = float(
                                torch.dot(old_w, ha)
                                + old_b
                            )

                            key = (
                                p["language"],
                                p["condition"],
                                p["group"],
                            )

                            grouped[key].append(
                                (lt, la, la - lt)
                            )

                        for (
                            language,
                            condition,
                            group,
                        ), vals in grouped.items():
                            A = np.asarray(
                                vals,
                                dtype=np.float64,
                            )
                            d = A[:, 2]

                            trajectory_rows[safety].append({
                                "model": model_name,
                                "family": family,
                                "fold": fold,
                                "safety_split": safety,
                                "language": language,
                                "condition": condition,
                                "probe_modality":
                                    probe_modality,
                                "layer": layer,
                                "disagreement_group":
                                    group,
                                "n": len(vals),
                                "mean_text_logit":
                                    float(A[:, 0].mean()),
                                "mean_audio_logit":
                                    float(A[:, 1].mean()),
                                "mean_audio_minus_text":
                                    float(d.mean()),
                                "median_audio_minus_text":
                                    float(np.median(d)),
                                "std_audio_minus_text":
                                    float(d.std(ddof=1))
                                    if len(d) > 1
                                    else 0.0,
                            })

                # =====================================================
                # SHARED TRAIN-ONLY SPECTRAL FIT
                # =====================================================

                X_joint_train = H[train_all]

                (
                    shared_mean,
                    shared_evals,
                    shared_evecs,
                    shared_ridge,
                ) = covariance_eigh(
                    X_joint_train,
                    device,
                    ridge_fraction=args.ridge_fraction,
                )

                total_var = float(shared_evals.sum())

                spectral_rows.append({
                    "model": model_name,
                    "family": family,
                    "fold": fold,
                    "layer": layer,
                    "train_n": len(train_all),
                    "dim": fs["dim"],
                    "ridge": shared_ridge,
                    "largest_eigenvalue":
                        float(shared_evals[0]),
                    "smallest_eigenvalue":
                        float(shared_evals[-1]),
                    "condition_number":
                        float(
                            shared_evals[0]
                            / max(
                                float(shared_evals[-1]),
                                1e-20,
                            )
                        ),
                })

                # =====================================================
                # 2. SHARED WHITENING
                # =====================================================

                H_white = apply_whitening(
                    H,
                    shared_mean,
                    shared_evals,
                    shared_evecs,
                    device,
                )

                # -- controlled matched/random representation cosine
                for p in test_pairs_all:
                    ht = H_white[p["text_index"]]
                    ha = H_white[p["audio_index"]]

                    mc = cosine(ht, ha)

                    rp = random_map[p["key"]]

                    if rp is None:
                        rc = float("nan")
                        diff = float("nan")
                        random_id = ""
                    else:
                        hra = H_white[
                            rp["audio_index"]
                        ]
                        rc = cosine(ht, hra)
                        diff = mc - rc
                        random_id = "|".join(
                            map(str, rp["key"])
                        )

                    cosine_pair_rows.append({
                        "model": model_name,
                        "fold": fold,
                        "safety_split":
                            p["safety_split"],
                        "language":
                            p["language"],
                        "condition":
                            p["condition"],
                        "outcome_group":
                            p["group"].replace(
                                "text", ""
                            ).replace(
                                "_audio", ""
                            ),
                        "text_decision":
                            p["text_decision"],
                        "audio_decision":
                            p["audio_decision"],
                        "family": family,
                        "layer": layer,
                        "pair_id":
                            "|".join(map(str, p["key"])),
                        "random_audio_pair_id":
                            random_id,
                        "matched_cosine": mc,
                        "random_unmatched_cosine": rc,
                        "matched_minus_random": diff,
                    })

                # -- whitened probes + transfer + weight alignment
                for safety in ("benign", "harmful"):
                    train_t = indices(
                        rows,
                        role="train",
                        safety=safety,
                        modality="text",
                    )
                    train_a = indices(
                        rows,
                        role="train",
                        safety=safety,
                        modality="audio",
                    )
                    test_t = indices(
                        rows,
                        role="test",
                        safety=safety,
                        modality="text",
                    )
                    test_a = indices(
                        rows,
                        role="test",
                        safety=safety,
                        modality="audio",
                    )

                    yt_train = row_labels(rows, train_t)
                    ya_train = row_labels(rows, train_a)
                    yt_test = row_labels(rows, test_t)
                    ya_test = row_labels(rows, test_a)

                    wt, bt = class_balanced_probe(
                        H_white[train_t],
                        yt_train,
                        device,
                        l2=args.l2,
                        max_iter=args.max_iter,
                        standardise=False,
                    )

                    wa, ba = class_balanced_probe(
                        H_white[train_a],
                        ya_train,
                        device,
                        l2=args.l2,
                        max_iter=args.max_iter,
                        standardise=False,
                    )

                    # same modality sanity
                    t_same = evaluate_probe(
                        H_white[test_t],
                        yt_test,
                        wt,
                        bt,
                    )
                    a_same = evaluate_probe(
                        H_white[test_a],
                        ya_test,
                        wa,
                        ba,
                    )

                    # cross modal
                    t_to_a = evaluate_probe(
                        H_white[test_a],
                        ya_test,
                        wt,
                        bt,
                    )
                    a_to_t = evaluate_probe(
                        H_white[test_t],
                        yt_test,
                        wa,
                        ba,
                    )

                    for source, target, result, n in (
                        (
                            "text",
                            "text",
                            t_same,
                            len(test_t),
                        ),
                        (
                            "audio",
                            "audio",
                            a_same,
                            len(test_a),
                        ),
                        (
                            "text",
                            "audio",
                            t_to_a,
                            len(test_a),
                        ),
                        (
                            "audio",
                            "text",
                            a_to_t,
                            len(test_t),
                        ),
                    ):
                        whitening_transfer_rows.append({
                            "model": model_name,
                            "family": family,
                            "fold": fold,
                            "safety_split": safety,
                            "source_probe_modality":
                                source,
                            "target_representation_modality":
                                target,
                            "layer": layer,
                            "n": n,
                            "auroc": result["auroc"],
                            "balanced_accuracy":
                                result[
                                    "balanced_accuracy"
                                ],
                            "normalisation":
                                "shared_full_whitening",
                        })

                    # Existing old probes transformed consistently
                    # into shared-whitened coordinates.
                    old_wt, _ = load_old_weight(
                        model_root,
                        fold,
                        safety,
                        "text",
                        family,
                        layer,
                    )
                    old_wa, _ = load_old_weight(
                        model_root,
                        fold,
                        safety,
                        "audio",
                        family,
                        layer,
                    )

                    old_wt_z = transformed_old_weight(
                        old_wt,
                        shared_evals,
                        shared_evecs,
                    )
                    old_wa_z = transformed_old_weight(
                        old_wa,
                        shared_evals,
                        shared_evecs,
                    )

                    probe_cos_rows.append({
                        "model": model_name,
                        "family": family,
                        "fold": fold,
                        "safety_split": safety,
                        "layer": layer,
                        "old_raw_probe_cosine":
                            cosine(old_wt, old_wa),
                        "old_probe_cosine_after_shared_whitening":
                            cosine(
                                old_wt_z,
                                old_wa_z,
                            ),
                        "retrained_whitened_probe_cosine":
                            cosine(wt, wa),
                        "text_same_modality_auroc":
                            t_same["auroc"],
                        "audio_same_modality_auroc":
                            a_same["auroc"],
                        "text_to_audio_auroc":
                            t_to_a["auroc"],
                        "audio_to_text_auroc":
                            a_to_t["auroc"],
                    })

                # =====================================================
                # 3. TRUE FISHER-INFORMATION WHITENING
                # =====================================================

                for safety in ("benign", "harmful"):
                    tr = indices(
                        rows,
                        role="train",
                        safety=safety,
                    )

                    y = row_labels(rows, tr)
                    Xs = H[tr]

                    # -------------------------------------------------
                    # Fit one TRAIN-ONLY safety probe jointly to text
                    # and audio.  This probe defines p(y=1 | h).
                    #
                    # No feature standardisation here: the Fisher
                    # matrix is constructed directly in the original
                    # representation coordinates.
                    # -------------------------------------------------
                    w_fisher_base, b_fisher_base = (
                        class_balanced_probe(
                            Xs,
                            y,
                            device,
                            l2=args.l2,
                            max_iter=args.max_iter,
                            standardise=False,
                        )
                    )

                    (
                        fisher_evals,
                        fisher_evecs,
                        fisher_ridge,
                        fisher_mean_curvature,
                    ) = fisher_information_eigh(
                        Xs,
                        w_fisher_base,
                        b_fisher_base,
                        device,
                        ridge_fraction=args.fisher_ridge_fraction,
                    )

                    # Fisher whitening:
                    #
                    #   z = F^{-1/2} h
                    #
                    # There is deliberately no class-mean subtraction
                    # here. F is the logistic-probe Fisher information
                    # matrix, not a within-class covariance matrix.
                    zero_mean = torch.zeros(
                        fs["dim"],
                        dtype=torch.float32,
                    )

                    H_fisher = apply_whitening(
                        H,
                        zero_mean,
                        fisher_evals,
                        fisher_evecs,
                        device,
                    )

                    train_t = indices(
                        rows,
                        "train",
                        safety,
                        "text",
                    )
                    train_a = indices(
                        rows,
                        "train",
                        safety,
                        "audio",
                    )
                    test_t = indices(
                        rows,
                        "test",
                        safety,
                        "text",
                    )
                    test_a = indices(
                        rows,
                        "test",
                        safety,
                        "audio",
                    )

                    yt_train = row_labels(rows, train_t)
                    ya_train = row_labels(rows, train_a)
                    yt_test = row_labels(rows, test_t)
                    ya_test = row_labels(rows, test_a)

                    # Retrain modality-specific probes in the common
                    # Fisher-whitened representation space.
                    wt_f, bt_f = class_balanced_probe(
                        H_fisher[train_t],
                        yt_train,
                        device,
                        l2=args.l2,
                        max_iter=args.max_iter,
                        standardise=False,
                    )

                    wa_f, ba_f = class_balanced_probe(
                        H_fisher[train_a],
                        ya_train,
                        device,
                        l2=args.l2,
                        max_iter=args.max_iter,
                        standardise=False,
                    )

                    t_metric = evaluate_probe(
                        H_fisher[test_t],
                        yt_test,
                        wt_f,
                        bt_f,
                    )

                    a_metric = evaluate_probe(
                        H_fisher[test_a],
                        ya_test,
                        wa_f,
                        ba_f,
                    )

                    t_to_a_metric = evaluate_probe(
                        H_fisher[test_a],
                        ya_test,
                        wt_f,
                        bt_f,
                    )

                    a_to_t_metric = evaluate_probe(
                        H_fisher[test_t],
                        yt_test,
                        wa_f,
                        ba_f,
                    )

                    fisher_rows.append({
                        "model": model_name,
                        "family": family,
                        "fold": fold,
                        "safety_split": safety,
                        "layer": layer,
                        "train_n": len(tr),
                        "train_0":
                            int((y == 0).sum()),
                        "train_1":
                            int((y == 1).sum()),
                        "ridge": fisher_ridge,
                        "fisher_ridge_fraction":
                            args.fisher_ridge_fraction,
                        "mean_bernoulli_curvature":
                            fisher_mean_curvature,
                        "fisher_probe_cosine":
                            cosine(wt_f, wa_f),
                        "text_test_auroc":
                            t_metric["auroc"],
                        "text_test_balanced_accuracy":
                            t_metric["balanced_accuracy"],
                        "audio_test_auroc":
                            a_metric["auroc"],
                        "audio_test_balanced_accuracy":
                            a_metric["balanced_accuracy"],
                        "text_to_audio_auroc":
                            t_to_a_metric["auroc"],
                        "text_to_audio_balanced_accuracy":
                            t_to_a_metric[
                                "balanced_accuracy"
                            ],
                        "audio_to_text_auroc":
                            a_to_t_metric["auroc"],
                        "audio_to_text_balanced_accuracy":
                            a_to_t_metric[
                                "balanced_accuracy"
                            ],
                        "fisher_definition":
                            "binary_logistic_expected_fisher",
                        "fisher_formula":
                            "mean[p*(1-p)*h*hT]",
                    })

                    del H_fisher
                    gc.collect()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                # =====================================================
                # 4. PCA DIMENSIONALITY SWEEP
                # =====================================================

                max_k = min(
                    max(PCA_K),
                    fs["dim"],
                )

                scores = pca_scores(
                    H,
                    shared_mean,
                    shared_evecs,
                    max_k,
                    device,
                )

                cumulative = torch.cumsum(
                    shared_evals,
                    dim=0,
                )

                for k in PCA_K:
                    if k > max_k:
                        continue

                    explained = float(
                        cumulative[k - 1]
                        / max(total_var, 1e-20)
                    )

                    Z = scores[:, :k]

                    for safety in ("benign", "harmful"):
                        train_t = indices(
                            rows,
                            "train",
                            safety,
                            "text",
                        )
                        train_a = indices(
                            rows,
                            "train",
                            safety,
                            "audio",
                        )
                        test_t = indices(
                            rows,
                            "test",
                            safety,
                            "text",
                        )
                        test_a = indices(
                            rows,
                            "test",
                            safety,
                            "audio",
                        )

                        yt_train = row_labels(
                            rows,
                            train_t,
                        )
                        ya_train = row_labels(
                            rows,
                            train_a,
                        )
                        yt_test = row_labels(
                            rows,
                            test_t,
                        )
                        ya_test = row_labels(
                            rows,
                            test_a,
                        )

                        wt_k, bt_k = class_balanced_probe(
                            Z[train_t],
                            yt_train,
                            device,
                            l2=args.l2,
                            max_iter=args.max_iter,
                        )
                        wa_k, ba_k = class_balanced_probe(
                            Z[train_a],
                            ya_train,
                            device,
                            l2=args.l2,
                            max_iter=args.max_iter,
                        )

                        results = {
                            ("text", "text"):
                                evaluate_probe(
                                    Z[test_t],
                                    yt_test,
                                    wt_k,
                                    bt_k,
                                ),
                            ("audio", "audio"):
                                evaluate_probe(
                                    Z[test_a],
                                    ya_test,
                                    wa_k,
                                    ba_k,
                                ),
                            ("text", "audio"):
                                evaluate_probe(
                                    Z[test_a],
                                    ya_test,
                                    wt_k,
                                    bt_k,
                                ),
                            ("audio", "text"):
                                evaluate_probe(
                                    Z[test_t],
                                    yt_test,
                                    wa_k,
                                    ba_k,
                                ),
                        }

                        for (
                            source,
                            target,
                        ), result in results.items():
                            pca_rows.append({
                                "model": model_name,
                                "family": family,
                                "fold": fold,
                                "safety_split":
                                    safety,
                                "layer": layer,
                                "k": k,
                                "explained_variance_ratio":
                                    explained,
                                "source_probe_modality":
                                    source,
                                "target_representation_modality":
                                    target,
                                "n":
                                    len(test_t)
                                    if target == "text"
                                    else len(test_a),
                                "auroc":
                                    result["auroc"],
                                "balanced_accuracy":
                                    result[
                                        "balanced_accuracy"
                                    ],
                            })

                # =====================================================
                # 5. TOP-PC REMOVAL
                # =====================================================

                for n_remove in REMOVE_TOP:
                    H_removed = remove_top_pcs(
                        H,
                        shared_mean,
                        shared_evecs,
                        n_remove,
                        device,
                    )

                    # matched-vs-random cosine on heldout pairs
                    cosine_bucket = defaultdict(list)

                    for p in test_pairs_all:
                        ht = H_removed[
                            p["text_index"]
                        ]
                        ha = H_removed[
                            p["audio_index"]
                        ]

                        mc = cosine(ht, ha)

                        rp = random_map[p["key"]]

                        if rp is None:
                            rc = float("nan")
                            diff = float("nan")
                        else:
                            rc = cosine(
                                ht,
                                H_removed[
                                    rp["audio_index"]
                                ],
                            )
                            diff = mc - rc

                        key = (
                            p["safety_split"],
                            p["language"],
                            p["condition"],
                            p["group"],
                        )

                        cosine_bucket[key].append(
                            (mc, rc, diff)
                        )

                    for (
                        safety,
                        language,
                        condition,
                        group,
                    ), vals in cosine_bucket.items():
                        mm, ms, mn = mean_std(
                            [x[0] for x in vals]
                        )
                        rm, rs, rn = mean_std(
                            [x[1] for x in vals]
                        )
                        dm, ds, dn = mean_std(
                            [x[2] for x in vals]
                        )

                        pc_remove_cos_rows.append({
                            "model": model_name,
                            "family": family,
                            "fold": fold,
                            "layer": layer,
                            "n_removed_top_pcs":
                                n_remove,
                            "safety_split": safety,
                            "language": language,
                            "condition": condition,
                            "outcome_group":
                                group.replace(
                                    "text", ""
                                ).replace(
                                    "_audio", ""
                                ),
                            "n_pairs": len(vals),
                            "matched_cosine_mean": mm,
                            "matched_cosine_std": ms,
                            "random_unmatched_cosine_mean":
                                rm,
                            "random_unmatched_cosine_std":
                                rs,
                            "matched_minus_random_mean":
                                dm,
                            "matched_minus_random_std":
                                ds,
                        })

                    # transfer survival
                    for safety in ("benign", "harmful"):
                        train_t = indices(
                            rows,
                            "train",
                            safety,
                            "text",
                        )
                        train_a = indices(
                            rows,
                            "train",
                            safety,
                            "audio",
                        )
                        test_t = indices(
                            rows,
                            "test",
                            safety,
                            "text",
                        )
                        test_a = indices(
                            rows,
                            "test",
                            safety,
                            "audio",
                        )

                        yt_train = row_labels(
                            rows,
                            train_t,
                        )
                        ya_train = row_labels(
                            rows,
                            train_a,
                        )
                        yt_test = row_labels(
                            rows,
                            test_t,
                        )
                        ya_test = row_labels(
                            rows,
                            test_a,
                        )

                        wt_r, bt_r = class_balanced_probe(
                            H_removed[train_t],
                            yt_train,
                            device,
                            l2=args.l2,
                            max_iter=args.max_iter,
                        )
                        wa_r, ba_r = class_balanced_probe(
                            H_removed[train_a],
                            ya_train,
                            device,
                            l2=args.l2,
                            max_iter=args.max_iter,
                        )

                        results = {
                            ("text", "text"):
                                evaluate_probe(
                                    H_removed[test_t],
                                    yt_test,
                                    wt_r,
                                    bt_r,
                                ),
                            ("audio", "audio"):
                                evaluate_probe(
                                    H_removed[test_a],
                                    ya_test,
                                    wa_r,
                                    ba_r,
                                ),
                            ("text", "audio"):
                                evaluate_probe(
                                    H_removed[test_a],
                                    ya_test,
                                    wt_r,
                                    bt_r,
                                ),
                            ("audio", "text"):
                                evaluate_probe(
                                    H_removed[test_t],
                                    yt_test,
                                    wa_r,
                                    ba_r,
                                ),
                        }

                        for (
                            source,
                            target,
                        ), result in results.items():
                            pc_remove_transfer_rows.append({
                                "model": model_name,
                                "family": family,
                                "fold": fold,
                                "safety_split":
                                    safety,
                                "layer": layer,
                                "n_removed_top_pcs":
                                    n_remove,
                                "source_probe_modality":
                                    source,
                                "target_representation_modality":
                                    target,
                                "n":
                                    len(test_t)
                                    if target == "text"
                                    else len(test_a),
                                "auroc":
                                    result["auroc"],
                                "balanced_accuracy":
                                    result[
                                        "balanced_accuracy"
                                    ],
                            })

                    del H_removed
                    gc.collect()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                # Layer is considered complete ONLY after every
                # requested analysis above succeeded.
                completed_layers.add(layer_key)

                # Keep human-readable CSVs current after every layer.
                write_outputs(
                    model_name,
                    model_root,
                    cross_root,
                    cosine_root,
                    new_root,
                    trajectory_rows,
                    whitening_transfer_rows,
                    cosine_pair_rows,
                    probe_cos_rows,
                    fisher_rows,
                    pca_rows,
                    pc_remove_cos_rows,
                    pc_remove_transfer_rows,
                    spectral_rows,
                )

                checkpoint_tmp = checkpoint_path.with_name(
                    checkpoint_path.name + ".tmp"
                )

                torch.save(
                    {
                        "trajectory_rows":
                            trajectory_rows,
                        "whitening_transfer_rows":
                            whitening_transfer_rows,
                        "cosine_pair_rows":
                            cosine_pair_rows,
                        "probe_cos_rows":
                            probe_cos_rows,
                        "fisher_rows":
                            fisher_rows,
                        "pca_rows":
                            pca_rows,
                        "pc_remove_cos_rows":
                            pc_remove_cos_rows,
                        "pc_remove_transfer_rows":
                            pc_remove_transfer_rows,
                        "spectral_rows":
                            spectral_rows,
                        "completed_layers":
                            sorted(completed_layers),
                    },
                    checkpoint_tmp,
                )

                checkpoint_tmp.replace(
                    checkpoint_path
                )

                print(
                    f"[CHECKPOINT] fold={fold} "
                    f"{family} L{layer:02d} | "
                    f"completed={len(completed_layers)}",
                    flush=True,
                )

                del (
                    H_white,
                    scores,
                    H,
                    shared_evecs,
                    shared_evals,
                )
                gc.collect()

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del Hfull
            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Checkpoint after every fold.
        write_outputs(
            model_name,
            model_root,
            cross_root,
            cosine_root,
            new_root,
            trajectory_rows,
            whitening_transfer_rows,
            cosine_pair_rows,
            probe_cos_rows,
            fisher_rows,
            pca_rows,
            pc_remove_cos_rows,
            pc_remove_transfer_rows,
            spectral_rows,
        )

    print()
    print("=" * 110)
    print("MODEL COMPLETE:", model_name)
    print("=" * 110)


# ---------------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------------

def write_outputs(
    model_name,
    model_root,
    cross_root,
    cosine_root,
    new_root,
    trajectory_rows,
    whitening_transfer_rows,
    cosine_pair_rows,
    probe_cos_rows,
    fisher_rows,
    pca_rows,
    pc_remove_cos_rows,
    pc_remove_transfer_rows,
    spectral_rows,
):
    # Recentered trajectories beside old trajectories.
    write_csv(
        cross_root
        / "NORMALISED_benign_disagreement_trajectory.csv",
        trajectory_rows["benign"],
    )
    write_csv(
        cross_root
        / "NORMALISED_harmful_disagreement_trajectory.csv",
        trajectory_rows["harmful"],
    )

    trajectory_contrasts = build_trajectory_contrasts(
        trajectory_rows
    )

    write_csv(
        cross_root
        / "NORMALISED_trajectory_OOF_contrasts.csv",
        trajectory_contrasts,
    )

    # Shared-whitened transfer beside old transfer.
    write_csv(
        cross_root
        / "NORMALISED_cross_modal_transfer.csv",
        whitening_transfer_rows,
    )

    # Shared whitened controlled cosine experiment.
    write_csv(
        cosine_root / "controlled_pairs.csv",
        cosine_pair_rows,
    )

    if cosine_pair_rows:
        # Per-fold diagnostics.
        by_language_fold = summarise_cosine(
            cosine_pair_rows,
            [
                "model",
                "fold",
                "safety_split",
                "language",
                "outcome_group",
                "family",
                "layer",
            ],
        )

        by_condition_fold = summarise_cosine(
            cosine_pair_rows,
            [
                "model",
                "fold",
                "safety_split",
                "language",
                "condition",
                "outcome_group",
                "family",
                "layer",
            ],
        )

        # Main OOF summaries: all 5 disjoint held-out folds combined.
        by_language = summarise_cosine(
            cosine_pair_rows,
            [
                "model",
                "safety_split",
                "language",
                "outcome_group",
                "family",
                "layer",
            ],
        )

        by_condition = summarise_cosine(
            cosine_pair_rows,
            [
                "model",
                "safety_split",
                "language",
                "condition",
                "outcome_group",
                "family",
                "layer",
            ],
        )

        write_csv(
            cosine_root
            / "by_language_outcome_by_fold.csv",
            by_language_fold,
        )

        write_csv(
            cosine_root
            / "by_language_condition_outcome_by_fold.csv",
            by_condition_fold,
        )

        write_csv(
            cosine_root
            / "by_language_outcome.csv",
            by_language,
        )

        write_csv(
            cosine_root
            / "by_language_condition_outcome.csv",
            by_condition,
        )

    write_csv(
        cosine_root / "probe_weight_cosine.csv",
        probe_cos_rows,
    )

    # New experiment folder.
    write_csv(
        new_root / "fisher_information_probe_alignment.csv",
        fisher_rows,
    )
    write_csv(
        new_root / "pca_dimensionality_sweep.csv",
        pca_rows,
    )
    write_csv(
        new_root / "top_pc_removal_cosine.csv",
        pc_remove_cos_rows,
    )
    write_csv(
        new_root / "top_pc_removal_transfer.csv",
        pc_remove_transfer_rows,
    )
    write_csv(
        new_root / "shared_spectral_diagnostics.csv",
        spectral_rows,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        choices=(
            "all",
            "qwen25_omni",
            "qwen2_audio",
            "kimi_audio",
        ),
        default="all",
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--l2",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--max-iter",
        type=int,
        default=150,
    )

    p.add_argument(
        "--ridge-fraction",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--fisher-ridge-fraction",
        type=float,
        default=1e-2,
        help=(
            "Regularisation lambda for Fisher whitening: "
            "F_reg = F + lambda * trace(F)/D * I."
        ),
    )

    p.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore saved per-layer checkpoint "
            "and recompute from scratch."
        ),
    )

    args = p.parse_args()

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        print(
            "[GPU]",
            torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
            flush=True,
        )

    models = (
        list(MODELS)
        if args.model == "all"
        else [args.model]
    )

    for model in models:
        try:
            run_model(model, args)
        except Exception:
            print()
            print("=" * 110)
            print("FAILED MODEL:", model)
            print("=" * 110)
            raise

    print()
    print("=" * 110)
    print("ALL NORMALISED ANALYSES COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
