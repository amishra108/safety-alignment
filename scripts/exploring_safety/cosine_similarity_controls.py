from pathlib import Path
from collections import defaultdict
import csv
import math
import random
import torch

ROOT = Path("results/INITIAL_BENCHMARK")
SEED = 12345
RNG = random.Random(SEED)

MODEL_SPECS = {
    "qwen2_audio": {
        "families": ("decoder",),
    },
    "qwen25_omni": {
        "families": ("thinker",),
    },
    "kimi_audio": {
        "families": ("shared_decoder", "text_mimo"),
    },
}

SAFETY_SPLITS = ("benign", "harmful")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))


def cosine(a, b):
    a = torch.as_tensor(a).detach().float().cpu().reshape(-1)
    b = torch.as_tensor(b).detach().float().cpu().reshape(-1)

    if a.shape != b.shape:
        raise RuntimeError(
            f"Cosine shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        )

    na = torch.linalg.vector_norm(a).item()
    nb = torch.linalg.vector_norm(b).item()

    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")

    v = torch.dot(a, b).item() / (na * nb)
    return max(-1.0, min(1.0, float(v)))


def mean_std(vals):
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan"), 0

    x = torch.tensor(vals, dtype=torch.float64)
    mean = x.mean().item()
    std = x.std(unbiased=True).item() if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def load_probe_weight(path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected dict checkpoint: {path}")

    if "w" not in obj:
        raise RuntimeError(
            f"No w in {path}; keys={list(obj.keys())}"
        )

    return torch.as_tensor(obj["w"]).detach().float().cpu().reshape(-1)


def layer_from_path(path):
    return int(path.stem.split("_")[1])


def representation_pair_stem(path):
    stem = path.stem

    if stem.endswith("__text"):
        return stem[:-6], "text"

    if stem.endswith("__audio"):
        return stem[:-7], "audio"

    return None, None


def metadata_from_stem(base):
    # Expected:
    # jbb_benign_000__literal__english
    parts = base.split("__")

    condition = parts[-2] if len(parts) >= 3 else ""
    language = parts[-1] if len(parts) >= 2 else ""

    if base.startswith("jbb_benign_"):
        safety = "benign"
    elif base.startswith("jbb_harmful_"):
        safety = "harmful"
    else:
        safety = ""

    return safety, condition, language


def get_rep_tensor(obj, family, path):
    if family not in obj:
        raise RuntimeError(
            f"{family!r} missing from {path}; "
            f"keys={list(obj.keys())}"
        )

    x = torch.as_tensor(obj[family]).detach().float().cpu()

    if x.ndim != 2:
        raise RuntimeError(
            f"{path}: family={family} expected [layers, hidden], "
            f"got {tuple(x.shape)}"
        )

    return x


# ==================================================================
# A. PROBE WEIGHT COSINE
# ==================================================================

print()
print("=" * 100)
print("A. LINEAR PROBE WEIGHT COSINE")
print("=" * 100)

for model, spec in MODEL_SPECS.items():
    weights_root = ROOT / model / "linear_probe_gpu" / "weights"

    if not weights_root.exists():
        print(f"[SKIP] {model}: no weights root")
        continue

    out_root = (
        weights_root
        / "cosine_similarity"
        / "linear_probe_weights"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    raw_rows = []

    for fold in FOLDS:
        for safety in SAFETY_SPLITS:
            for family in spec["families"]:
                tdir = weights_root / fold / safety / "text" / family
                adir = weights_root / fold / safety / "audio" / family

                if not tdir.exists() or not adir.exists():
                    print(
                        f"[SKIP FAMILY] {model} {fold} "
                        f"{safety} {family}"
                    )
                    continue

                tpaths = {
                    layer_from_path(p): p
                    for p in tdir.glob("layer_*.pt")
                    if "_prev" not in p.stem
                }
                apaths = {
                    layer_from_path(p): p
                    for p in adir.glob("layer_*.pt")
                    if "_prev" not in p.stem
                }

                layers = sorted(set(tpaths) & set(apaths))

                if not layers:
                    continue

                # One deterministic derangement/permutation of audio layers.
                # Every control layer differs from its true text layer.
                if len(layers) > 1:
                    shifted = layers[1:] + layers[:1]
                else:
                    shifted = layers[:]

                for layer, random_audio_layer in zip(layers, shifted):
                    wt = load_probe_weight(tpaths[layer])
                    wa = load_probe_weight(apaths[layer])

                    matched = cosine(wt, wa)

                    if len(layers) > 1:
                        wa_random = load_probe_weight(
                            apaths[random_audio_layer]
                        )

                        if wt.numel() == wa_random.numel():
                            random_cos = cosine(wt, wa_random)
                        else:
                            random_cos = float("nan")
                    else:
                        random_cos = float("nan")

                    raw_rows.append({
                        "model": model,
                        "fold": fold,
                        "safety_split": safety,
                        "family": family,
                        "layer": layer,
                        "random_audio_layer": random_audio_layer,
                        "matched_cosine": matched,
                        "random_unmatched_cosine": random_cos,
                        "matched_minus_random": (
                            matched - random_cos
                            if math.isfinite(matched)
                            and math.isfinite(random_cos)
                            else float("nan")
                        ),
                        "hidden_dim": wt.numel(),
                    })

    raw_path = (
        out_root
        / "weight_cosine_matched_vs_random_by_fold.csv"
    )

    fields = [
        "model",
        "fold",
        "safety_split",
        "family",
        "layer",
        "random_audio_layer",
        "matched_cosine",
        "random_unmatched_cosine",
        "matched_minus_random",
        "hidden_dim",
    ]

    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(raw_rows)

    grouped = defaultdict(lambda: {
        "matched": [],
        "random": [],
        "diff": [],
    })

    for r in raw_rows:
        key = (
            r["model"],
            r["safety_split"],
            r["family"],
            r["layer"],
            r["hidden_dim"],
        )
        grouped[key]["matched"].append(r["matched_cosine"])
        grouped[key]["random"].append(r["random_unmatched_cosine"])
        grouped[key]["diff"].append(r["matched_minus_random"])

    summary_rows = []

    for key in sorted(grouped):
        model_, safety, family, layer, dim = key
        g = grouped[key]

        mm, ms, mn = mean_std(g["matched"])
        rm, rs, rn = mean_std(g["random"])
        dm, ds, dn = mean_std(g["diff"])

        summary_rows.append({
            "model": model_,
            "safety_split": safety,
            "family": family,
            "layer": layer,
            "hidden_dim": dim,
            "n_folds": mn,
            "matched_cosine_mean": mm,
            "matched_cosine_std": ms,
            "random_unmatched_cosine_mean": rm,
            "random_unmatched_cosine_std": rs,
            "matched_minus_random_mean": dm,
            "matched_minus_random_std": ds,
        })

    summary_path = (
        out_root
        / "weight_cosine_matched_vs_random_mean_std.csv"
    )

    summary_fields = [
        "model",
        "safety_split",
        "family",
        "layer",
        "hidden_dim",
        "n_folds",
        "matched_cosine_mean",
        "matched_cosine_std",
        "random_unmatched_cosine_mean",
        "random_unmatched_cosine_std",
        "matched_minus_random_mean",
        "matched_minus_random_std",
    ]

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[WROTE] {raw_path}")
    print(f"[WROTE] {summary_path}")


# ==================================================================
# B. REPRESENTATION COSINE
# ==================================================================

print()
print("=" * 100)
print("B. MATCHED TEXT/AUDIO REPRESENTATION COSINE")
print("=" * 100)

for model, spec in MODEL_SPECS.items():
    rep_root = ROOT / model / "5fold" / "representations_shared"
    weights_root = ROOT / model / "linear_probe_gpu" / "weights"

    if not rep_root.exists():
        print(f"[SKIP] {model}: no representations_shared")
        continue

    out_root = (
        weights_root
        / "cosine_similarity"
        / "representation_similarity"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    text_paths = {}
    audio_paths = {}

    for p in sorted(rep_root.glob("*.pt")):
        base, modality = representation_pair_stem(p)

        if base is None:
            continue

        if modality == "text":
            text_paths[base] = p
        else:
            audio_paths[base] = p

    matched_bases = sorted(set(text_paths) & set(audio_paths))

    print(
        f"[{model}] text={len(text_paths)} "
        f"audio={len(audio_paths)} matched={len(matched_bases)}"
    )

    # Build random-control pools preserving:
    # safety split + condition + language.
    pools = defaultdict(list)

    for base in matched_bases:
        safety, condition, language = metadata_from_stem(base)
        pools[(safety, condition, language)].append(base)

    random_partner = {}

    for group_key, bases in sorted(pools.items()):
        bases = sorted(bases)

        if len(bases) < 2:
            for base in bases:
                random_partner[base] = None
            continue

        # Deterministic random derangement.
        shuffled = bases[:]

        for _ in range(1000):
            RNG.shuffle(shuffled)
            if all(a != b for a, b in zip(bases, shuffled)):
                break
        else:
            # Guaranteed derangement fallback.
            shuffled = bases[1:] + bases[:1]

        for base, other in zip(bases, shuffled):
            if base == other:
                raise RuntimeError(
                    f"Random control accidentally self-matched: {base}"
                )
            random_partner[base] = other

    pair_rows = []

    for idx, base in enumerate(matched_bases, 1):
        tpath = text_paths[base]
        apath = audio_paths[base]
        random_base = random_partner.get(base)

        if random_base is None:
            continue

        random_apath = audio_paths[random_base]

        tobj = torch.load(
            tpath,
            map_location="cpu",
            weights_only=False,
        )
        aobj = torch.load(
            apath,
            map_location="cpu",
            weights_only=False,
        )
        raobj = torch.load(
            random_apath,
            map_location="cpu",
            weights_only=False,
        )

        safety = (
            tobj.get("safety_split")
            or metadata_from_stem(base)[0]
        )
        condition = (
            tobj.get("condition")
            or metadata_from_stem(base)[1]
        )
        language = (
            tobj.get("language")
            or metadata_from_stem(base)[2]
        )

        if aobj.get("safety_split", safety) != safety:
            raise RuntimeError(
                f"Matched safety mismatch for {base}"
            )

        for family in spec["families"]:
            if (
                family not in tobj
                or family not in aobj
                or family not in raobj
            ):
                print(
                    f"[SKIP FAMILY IN REP] {model} "
                    f"{family} base={base}"
                )
                continue

            ht = get_rep_tensor(tobj, family, tpath)
            ha = get_rep_tensor(aobj, family, apath)
            hra = get_rep_tensor(
                raobj,
                family,
                random_apath,
            )

            if ht.shape != ha.shape:
                raise RuntimeError(
                    f"MATCHED SHAPE ERROR {model} {family} {base}: "
                    f"text={tuple(ht.shape)} "
                    f"audio={tuple(ha.shape)}"
                )

            if ht.shape != hra.shape:
                raise RuntimeError(
                    f"RANDOM SHAPE ERROR {model} {family} {base}: "
                    f"text={tuple(ht.shape)} "
                    f"random_audio={tuple(hra.shape)}"
                )

            for layer in range(ht.shape[0]):
                matched_cos = cosine(ht[layer], ha[layer])
                random_cos = cosine(ht[layer], hra[layer])

                pair_rows.append({
                    "model": model,
                    "safety_split": safety,
                    "condition": condition,
                    "language": language,
                    "family": family,
                    "layer": layer,
                    "pair_id": base,
                    "random_audio_pair_id": random_base,
                    "hidden_dim": ht.shape[1],
                    "matched_cosine": matched_cos,
                    "random_unmatched_cosine": random_cos,
                    "matched_minus_random": (
                        matched_cos - random_cos
                        if math.isfinite(matched_cos)
                        and math.isfinite(random_cos)
                        else float("nan")
                    ),
                })

        del tobj, aobj, raobj

        if idx % 250 == 0 or idx == len(matched_bases):
            print(
                f"[{model}] representation pairs "
                f"{idx}/{len(matched_bases)}",
                flush=True,
            )

    pair_path = (
        out_root
        / "representation_cosine_pairs.csv"
    )

    pair_fields = [
        "model",
        "safety_split",
        "condition",
        "language",
        "family",
        "layer",
        "pair_id",
        "random_audio_pair_id",
        "hidden_dim",
        "matched_cosine",
        "random_unmatched_cosine",
        "matched_minus_random",
    ]

    with pair_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pair_fields)
        w.writeheader()
        w.writerows(pair_rows)

    grouped = defaultdict(lambda: {
        "matched": [],
        "random": [],
        "diff": [],
    })

    for r in pair_rows:
        key = (
            r["model"],
            r["safety_split"],
            r["family"],
            r["layer"],
            r["hidden_dim"],
        )
        grouped[key]["matched"].append(r["matched_cosine"])
        grouped[key]["random"].append(r["random_unmatched_cosine"])
        grouped[key]["diff"].append(r["matched_minus_random"])

    summary_rows = []

    for key in sorted(grouped):
        model_, safety, family, layer, dim = key
        g = grouped[key]

        mm, ms, mn = mean_std(g["matched"])
        rm, rs, rn = mean_std(g["random"])
        dm, ds, dn = mean_std(g["diff"])

        summary_rows.append({
            "model": model_,
            "safety_split": safety,
            "family": family,
            "layer": layer,
            "hidden_dim": dim,
            "n_pairs": mn,
            "matched_cosine_mean": mm,
            "matched_cosine_std": ms,
            "random_unmatched_cosine_mean": rm,
            "random_unmatched_cosine_std": rs,
            "matched_minus_random_mean": dm,
            "matched_minus_random_std": ds,
        })

    summary_path = (
        out_root
        / "representation_cosine_mean_std.csv"
    )

    summary_fields = [
        "model",
        "safety_split",
        "family",
        "layer",
        "hidden_dim",
        "n_pairs",
        "matched_cosine_mean",
        "matched_cosine_std",
        "random_unmatched_cosine_mean",
        "random_unmatched_cosine_std",
        "matched_minus_random_mean",
        "matched_minus_random_std",
    ]

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[WROTE] {pair_path}")
    print(f"[WROTE] {summary_path}")

    print()
    for r in summary_rows:
        print(
            f"{model} | {r['safety_split']:7s} | "
            f"{r['family']:14s} | L{r['layer']:02d} | "
            f"matched={r['matched_cosine_mean']:+.4f} | "
            f"random={r['random_unmatched_cosine_mean']:+.4f} | "
            f"delta={r['matched_minus_random_mean']:+.4f} | "
            f"N={r['n_pairs']}"
        )

print()
print("DONE")
