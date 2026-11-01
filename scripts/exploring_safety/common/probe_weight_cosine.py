from pathlib import Path
import csv
import math
import torch

ROOT = Path("results/INITIAL_BENCHMARK")

MODELS = [
    "qwen2_audio",
    "qwen25_omni",
    "kimi_audio",
    "audio_flamingo3",
]

SAFETY_SPLITS = ("benign", "harmful")
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))


def extract_w_b(obj, path):
    """
    Return flattened probe weight w and bias b.
    Supports the common checkpoint layouts used by these probe runs.
    """
    if isinstance(obj, dict):
        # Direct saved tensors.
        for w_key in ("w", "weight", "coef", "coef_"):
            if w_key in obj:
                w = obj[w_key]
                b = None
                for b_key in ("b", "bias", "intercept", "intercept_"):
                    if b_key in obj:
                        b = obj[b_key]
                        break

                w = torch.as_tensor(w).detach().float().cpu().reshape(-1)
                if b is not None:
                    b = torch.as_tensor(b).detach().float().cpu().reshape(-1)
                return w, b

        # Occasionally saved inside a state_dict.
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return extract_w_b(obj["state_dict"], path)

        # torch Linear style keys.
        weight_keys = [k for k in obj if k.endswith(".weight")]
        if len(weight_keys) == 1:
            wk = weight_keys[0]
            prefix = wk[:-len(".weight")]
            bk = prefix + ".bias"
            w = torch.as_tensor(obj[wk]).detach().float().cpu().reshape(-1)
            b = obj.get(bk)
            if b is not None:
                b = torch.as_tensor(b).detach().float().cpu().reshape(-1)
            return w, b

        raise RuntimeError(
            f"Cannot find probe weight in {path}\n"
            f"keys={list(obj.keys())}"
        )

    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        w = torch.as_tensor(obj[0]).detach().float().cpu().reshape(-1)
        b = None if len(obj) < 2 else torch.as_tensor(obj[1]).detach().float().cpu().reshape(-1)
        return w, b

    raise RuntimeError(f"Unsupported checkpoint type in {path}: {type(obj)}")


def layer_number(path):
    # layer_00.pt -> 0
    return int(path.stem.split("_")[1])


all_rows = []

for model in MODELS:
    weights_root = ROOT / model / "linear_probe_gpu" / "weights"

    if not weights_root.exists():
        print(f"[skip model] {model}: {weights_root} does not exist")
        continue

    out_dir = weights_root / "cosine_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = []

    print()
    print("=" * 90)
    print(model)
    print("=" * 90)

    for safety in SAFETY_SPLITS:
        for fold in FOLDS:
            fold_root = weights_root / fold / safety
            text_root = fold_root / "text"
            audio_root = fold_root / "audio"

            if not text_root.exists() or not audio_root.exists():
                print(f"[skip] {model} {fold} {safety}: missing text/audio directory")
                continue

            # Compare only representation families present in BOTH modalities.
            text_families = {
                p.name for p in text_root.iterdir()
                if p.is_dir()
            }
            audio_families = {
                p.name for p in audio_root.iterdir()
                if p.is_dir()
            }
            families = sorted(text_families & audio_families)

            only_audio = sorted(audio_families - text_families)
            only_text = sorted(text_families - audio_families)

            if only_audio:
                print(
                    f"[not comparable] {model} {fold} {safety} "
                    f"audio-only families: {', '.join(only_audio)}"
                )
            if only_text:
                print(
                    f"[not comparable] {model} {fold} {safety} "
                    f"text-only families: {', '.join(only_text)}"
                )

            for family in families:
                text_dir = text_root / family
                audio_dir = audio_root / family

                # Explicitly reject *_prev.pt.
                text_layers = {
                    layer_number(p): p
                    for p in text_dir.glob("layer_*.pt")
                    if "_prev" not in p.stem
                }
                audio_layers = {
                    layer_number(p): p
                    for p in audio_dir.glob("layer_*.pt")
                    if "_prev" not in p.stem
                }

                common_layers = sorted(set(text_layers) & set(audio_layers))

                missing_audio = sorted(set(text_layers) - set(audio_layers))
                missing_text = sorted(set(audio_layers) - set(text_layers))

                if missing_audio or missing_text:
                    print(
                        f"[layer mismatch] {model} {fold} {safety} {family}: "
                        f"text-only layers={missing_audio}, "
                        f"audio-only layers={missing_text}"
                    )

                for layer in common_layers:
                    text_path = text_layers[layer]
                    audio_path = audio_layers[layer]

                    text_obj = torch.load(text_path, map_location="cpu", weights_only=False)
                    audio_obj = torch.load(audio_path, map_location="cpu", weights_only=False)

                    wt, bt = extract_w_b(text_obj, text_path)
                    wa, ba = extract_w_b(audio_obj, audio_path)

                    if wt.shape != wa.shape:
                        raise RuntimeError(
                            f"WEIGHT SHAPE MISMATCH:\n"
                            f"  text:  {text_path} -> {tuple(wt.shape)}\n"
                            f"  audio: {audio_path} -> {tuple(wa.shape)}"
                        )

                    nt = torch.linalg.vector_norm(wt).item()
                    na = torch.linalg.vector_norm(wa).item()

                    zero_weight = (nt <= 1e-12 or na <= 1e-12)

                    if zero_weight:
                        cosine = float("nan")
                        angle = float("nan")
                        print(
                            f"[undefined cosine] {model} {fold} {safety} "
                            f"{family} L{layer:02d}: "
                            f"||w_text||={nt:.6g}, ||w_audio||={na:.6g}"
                        )
                    else:
                        cosine = torch.dot(wt, wa).item() / (nt * na)
                        cosine = max(-1.0, min(1.0, cosine))
                        angle = math.degrees(math.acos(cosine))

                    row = {
                        "model": model,
                        "fold": fold,
                        "safety_split": safety,
                        "family": family,
                        "layer": layer,
                        "hidden_dim": wt.numel(),
                        "cosine_similarity": cosine,
                        "angle_degrees": angle,
                        "text_weight_norm": nt,
                        "audio_weight_norm": na,
                        "text_bias": (
                            float(bt.reshape(-1)[0])
                            if bt is not None and bt.numel() == 1
                            else ""
                        ),
                        "audio_bias": (
                            float(ba.reshape(-1)[0])
                            if ba is not None and ba.numel() == 1
                            else ""
                        ),
                        "text_weight_path": str(text_path),
                        "audio_weight_path": str(audio_path),
                    }

                    model_rows.append(row)
                    all_rows.append(row)

    raw_path = out_dir / "cosine_similarity_by_fold.csv"

    fieldnames = [
        "model",
        "fold",
        "safety_split",
        "family",
        "layer",
        "hidden_dim",
        "cosine_similarity",
        "angle_degrees",
        "text_weight_norm",
        "audio_weight_norm",
        "text_bias",
        "audio_bias",
        "text_weight_path",
        "audio_weight_path",
    ]

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(model_rows)

    # Mean/std across the five corresponding fold cosine values.
    grouped = {}
    for r in model_rows:
        key = (
            r["model"],
            r["safety_split"],
            r["family"],
            r["layer"],
            r["hidden_dim"],
        )
        c = r["cosine_similarity"]
        if math.isfinite(c):
            grouped.setdefault(key, []).append(c)

    summary_rows = []
    for (m, safety, family, layer, dim), vals in sorted(grouped.items()):
        x = torch.tensor(vals, dtype=torch.float64)
        summary_rows.append({
            "model": m,
            "safety_split": safety,
            "family": family,
            "layer": layer,
            "hidden_dim": dim,
            "n_folds": len(vals),
            "cosine_mean": x.mean().item(),
            "cosine_std": x.std(unbiased=True).item() if len(vals) > 1 else 0.0,
            "cosine_min": x.min().item(),
            "cosine_max": x.max().item(),
        })

    summary_path = out_dir / "cosine_similarity_mean_std.csv"
    summary_fields = [
        "model",
        "safety_split",
        "family",
        "layer",
        "hidden_dim",
        "n_folds",
        "cosine_mean",
        "cosine_std",
        "cosine_min",
        "cosine_max",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[saved] {raw_path}")
    print(f"[saved] {summary_path}")

    # Concise terminal result.
    for safety in SAFETY_SPLITS:
        these = [r for r in summary_rows if r["safety_split"] == safety]
        families = sorted({r["family"] for r in these})

        for family in families:
            rr = sorted(
                [r for r in these if r["family"] == family],
                key=lambda r: r["layer"],
            )
            print(f"\n{model} | {safety} | {family}")
            for r in rr:
                print(
                    f"L{r['layer']:02d}  "
                    f"cos={r['cosine_mean']:+.4f} "
                    f"± {r['cosine_std']:.4f}  "
                    f"folds={r['n_folds']}"
                )

print()
print("DONE")
