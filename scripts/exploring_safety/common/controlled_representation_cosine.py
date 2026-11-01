from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import random
import torch

ROOT = Path("results/INITIAL_BENCHMARK")
JUDGED = ROOT / "all_judged"
SEED = 1729

# Explicit authoritative decision files.
FILES = {
    "qwen2_audio": {
        ("benign", "text"):
            JUDGED / "qwen2_audio.text.benign.adjudicated.carefully_rechecked.jsonl",
        ("benign", "audio"):
            JUDGED / "qwen2_audio.audio.benign.adjudicated.carefully_rechecked.jsonl",
        ("harmful", "text"):
            JUDGED / "qwen2_audio.text.harmful.adjudicated.carefully_rechecked.jsonl",
        ("harmful", "audio"):
            JUDGED / "qwen2_audio.audio.harmful.adjudicated.carefully_rechecked.jsonl",
    },

    "qwen25_omni": {
        ("benign", "text"):
            JUDGED / "qwen25_omni.text.benign.readjudicated.carefully_rechecked.jsonl",
        ("benign", "audio"):
            JUDGED / "qwen25_omni.audio.benign.readjudicated.carefully_rechecked.jsonl",
        ("harmful", "text"):
            JUDGED / "qwen25_omni.text.harmful.readjudicated.carefully_rechecked.jsonl",
        ("harmful", "audio"):
            JUDGED / "qwen25_omni.audio.harmful.readjudicated.carefully_rechecked.jsonl",
    },

    "kimi_audio": {
        ("benign", "text"):
            JUDGED / "kimi_audio.text.benign.adjudicated.carefully_rechecked.jsonl",
        ("benign", "audio"):
            JUDGED / "kimi_audio.audio.benign.adjudicated.carefully_rechecked.jsonl",
        ("harmful", "text"):
            JUDGED / "kimi_audio.text.harmful.adjudicated.carefully_rechecked.jsonl",
        ("harmful", "audio"):
            JUDGED / "kimi_audio.audio.harmful.adjudicated.carefully_rechecked.jsonl",
    },
}

FAMILIES = {
    "qwen2_audio": ("decoder",),
    "qwen25_omni": ("thinker",),
    "kimi_audio": ("shared_decoder", "text_mimo"),
}


def read_jsonl(path):
    if not path.exists():
        raise RuntimeError(f"Missing authoritative file: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Bad JSON: {path}:{ln}") from exc
    return rows


def base_id(task_id, modality):
    suffix = f"__{modality}"
    if not task_id.endswith(suffix):
        raise RuntimeError(
            f"Task ID does not end in {suffix}: {task_id}"
        )
    return task_id[:-len(suffix)]


def cosine(a, b):
    a = torch.as_tensor(a).float().cpu().reshape(-1)
    b = torch.as_tensor(b).float().cpu().reshape(-1)

    if a.shape != b.shape:
        raise RuntimeError(
            f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
        )

    na = torch.linalg.vector_norm(a).item()
    nb = torch.linalg.vector_norm(b).item()

    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")

    x = torch.dot(a, b).item() / (na * nb)
    return max(-1.0, min(1.0, float(x)))


def stats(values):
    x = [
        float(v) for v in values
        if math.isfinite(float(v))
    ]

    if not x:
        return float("nan"), float("nan"), 0

    t = torch.tensor(x, dtype=torch.float64)

    return (
        t.mean().item(),
        t.std(unbiased=True).item() if len(x) > 1 else 0.0,
        len(x),
    )


def summarise(rows, key_fields):
    groups = defaultdict(lambda: {
        "matched": [],
        "random": [],
        "delta": [],
        "pair_ids": set(),
        "random_pair_ids": set(),
    })

    for r in rows:
        key = tuple(r[k] for k in key_fields)
        g = groups[key]

        g["matched"].append(r["matched_cosine"])
        g["random"].append(r["random_unmatched_cosine"])
        g["delta"].append(r["matched_minus_random"])
        g["pair_ids"].add(r["pair_id"])

        if r["random_audio_pair_id"]:
            g["random_pair_ids"].add(r["random_audio_pair_id"])

    out = []

    for key in sorted(groups):
        g = groups[key]

        mm, ms, mn = stats(g["matched"])
        rm, rs, rn = stats(g["random"])
        dm, ds, dn = stats(g["delta"])

        row = dict(zip(key_fields, key))
        row.update({
            "n_pairs": len(g["pair_ids"]),
            "n_random_control_pairs": len(g["random_pair_ids"]),
            "matched_cosine_mean": mm,
            "matched_cosine_std": ms,
            "random_unmatched_cosine_mean": rm,
            "random_unmatched_cosine_std": rs,
            "matched_minus_random_mean": dm,
            "matched_minus_random_std": ds,
            "n_finite_differences": dn,
        })
        out.append(row)

    return out


for model in FILES:
    print()
    print("=" * 110)
    print(model)
    print("=" * 110)

    rep_root = (
        ROOT
        / model
        / "5fold"
        / "representations_shared"
    )

    if not rep_root.exists():
        raise RuntimeError(f"Missing representation root: {rep_root}")

    out_root = (
        ROOT
        / model
        / "linear_probe_gpu"
        / "cosine_similarity"
        / "controlled_representation_similarity"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Load authoritative decisions.
    # --------------------------------------------------------------
    records = {}

    for safety in ("benign", "harmful"):
        modality_rows = {}

        for modality in ("text", "audio"):
            path = FILES[model][(safety, modality)]
            rows = read_jsonl(path)

            print(
                f"[AUTHORITATIVE] {safety:7s} {modality:5s} "
                f"N={len(rows)} | {path.name}"
            )

            by_base = {}

            for r in rows:
                if r.get("split") != safety:
                    raise RuntimeError(
                        f"Wrong split in {path}: "
                        f"{r.get('task_id')} -> {r.get('split')}"
                    )

                if int(r["decision"]) not in (0, 1):
                    raise RuntimeError(
                        f"Bad decision in {path}: {r['decision']}"
                    )

                b = base_id(r["task_id"], modality)

                if b in by_base:
                    raise RuntimeError(
                        f"Duplicate base ID in {path}: {b}"
                    )

                by_base[b] = r

            modality_rows[modality] = by_base

        text_ids = set(modality_rows["text"])
        audio_ids = set(modality_rows["audio"])

        if text_ids != audio_ids:
            raise RuntimeError(
                f"{model} {safety}: authoritative text/audio IDs differ. "
                f"text_only={len(text_ids-audio_ids)} "
                f"audio_only={len(audio_ids-text_ids)}"
            )

        for b in sorted(text_ids):
            tr = modality_rows["text"][b]
            ar = modality_rows["audio"][b]

            # These should describe the same underlying benchmark case.
            for field in ("language", "condition"):
                if tr.get(field) != ar.get(field):
                    raise RuntimeError(
                        f"{model} {safety} {b}: "
                        f"{field} mismatch: "
                        f"text={tr.get(field)!r}, "
                        f"audio={ar.get(field)!r}"
                    )

            td = int(tr["decision"])
            ad = int(ar["decision"])

            records[b] = {
                "safety_split": safety,
                "language": tr["language"],
                "condition": tr["condition"],
                "text_decision": td,
                "audio_decision": ad,
                "outcome_group": f"{td}{ad}",
            }

    print(f"[AUTHORITATIVE MATCHED BASES] N={len(records)}")

    # --------------------------------------------------------------
    # Check representation files explicitly against authoritative IDs.
    # --------------------------------------------------------------
    text_paths = {}
    audio_paths = {}

    for b in records:
        tp = rep_root / f"{b}__text.pt"
        ap = rep_root / f"{b}__audio.pt"

        if not tp.exists():
            raise RuntimeError(f"Missing text representation: {tp}")
        if not ap.exists():
            raise RuntimeError(f"Missing audio representation: {ap}")

        text_paths[b] = tp
        audio_paths[b] = ap

    # --------------------------------------------------------------
    # RANDOM CONTROL:
    #
    # Within exact:
    #   safety_split
    #   language
    #   condition
    #   text_decision
    #   audio_decision
    #
    # Pair text prompt i against AUDIO prompt j != i.
    #
    # Thus random control has the SAME 00/01/10/11 outcome group.
    # --------------------------------------------------------------
    strata = defaultdict(list)

    for b, r in records.items():
        key = (
            r["safety_split"],
            r["language"],
            r["condition"],
            r["text_decision"],
            r["audio_decision"],
        )
        strata[key].append(b)

    random_partner = {}

    for key, ids in sorted(strata.items()):
        ids = sorted(ids)

        if len(ids) < 2:
            random_partner[ids[0]] = None
            continue

        # Reproducible but stratum-specific shuffle.
        rng = random.Random(
            f"{SEED}|{model}|{'|'.join(map(str, key))}"
        )

        shuffled = ids[:]

        for _ in range(10000):
            rng.shuffle(shuffled)

            if all(a != b for a, b in zip(ids, shuffled)):
                break
        else:
            # Guaranteed cyclic derangement.
            shuffled = ids[1:] + ids[:1]

        for a, b in zip(ids, shuffled):
            if a == b:
                raise RuntimeError(
                    f"Self match in random control: {key} {a}"
                )
            random_partner[a] = b

    print()
    print("Outcome counts before cosine:")
    count_groups = defaultdict(int)

    for r in records.values():
        count_groups[
            (
                r["safety_split"],
                r["language"],
                r["outcome_group"],
            )
        ] += 1

    for key, n in sorted(count_groups.items()):
        print(
            f"  safety={key[0]:7s} "
            f"language={key[1]:8s} "
            f"group={key[2]} "
            f"N={n}"
        )

    # --------------------------------------------------------------
    # Compute representation cosine at every shared family/layer.
    # --------------------------------------------------------------
    pair_rows = []

    bases = sorted(records)

    for i, b in enumerate(bases, 1):
        meta = records[b]
        rb = random_partner[b]

        text_obj = torch.load(
            text_paths[b],
            map_location="cpu",
            weights_only=False,
        )
        audio_obj = torch.load(
            audio_paths[b],
            map_location="cpu",
            weights_only=False,
        )

        random_obj = None
        if rb is not None:
            random_obj = torch.load(
                audio_paths[rb],
                map_location="cpu",
                weights_only=False,
            )

        for family in FAMILIES[model]:
            if family not in text_obj:
                raise RuntimeError(
                    f"{text_paths[b]} missing {family}; "
                    f"keys={list(text_obj.keys())}"
                )

            if family not in audio_obj:
                raise RuntimeError(
                    f"{audio_paths[b]} missing {family}; "
                    f"keys={list(audio_obj.keys())}"
                )

            ht = torch.as_tensor(text_obj[family]).float().cpu()
            ha = torch.as_tensor(audio_obj[family]).float().cpu()

            if ht.ndim != 2 or ha.ndim != 2:
                raise RuntimeError(
                    f"{model} {family}: expected [layers, hidden], "
                    f"text={tuple(ht.shape)} audio={tuple(ha.shape)}"
                )

            if ht.shape != ha.shape:
                raise RuntimeError(
                    f"{model} {family} matched shape mismatch "
                    f"{b}: {tuple(ht.shape)} vs {tuple(ha.shape)}"
                )

            hra = None

            if random_obj is not None:
                if family not in random_obj:
                    raise RuntimeError(
                        f"{audio_paths[rb]} missing {family}"
                    )

                hra = torch.as_tensor(
                    random_obj[family]
                ).float().cpu()

                if hra.shape != ht.shape:
                    raise RuntimeError(
                        f"{model} {family} random shape mismatch "
                        f"{b} -> {rb}: "
                        f"{tuple(ht.shape)} vs {tuple(hra.shape)}"
                    )

            for layer in range(ht.shape[0]):
                mc = cosine(ht[layer], ha[layer])

                if hra is None:
                    rc = float("nan")
                    diff = float("nan")
                else:
                    rc = cosine(ht[layer], hra[layer])
                    diff = (
                        mc - rc
                        if math.isfinite(mc)
                        and math.isfinite(rc)
                        else float("nan")
                    )

                pair_rows.append({
                    "model": model,
                    "safety_split": meta["safety_split"],
                    "language": meta["language"],
                    "condition": meta["condition"],
                    "outcome_group": meta["outcome_group"],
                    "text_decision": meta["text_decision"],
                    "audio_decision": meta["audio_decision"],
                    "family": family,
                    "layer": layer,
                    "pair_id": b,
                    "random_audio_pair_id": rb or "",
                    "matched_cosine": mc,
                    "random_unmatched_cosine": rc,
                    "matched_minus_random": diff,
                })

        if i % 250 == 0 or i == len(bases):
            print(
                f"[{model}] processed {i}/{len(bases)} matched bases",
                flush=True,
            )

    # --------------------------------------------------------------
    # Raw pair-level output.
    # --------------------------------------------------------------
    raw_path = out_root / "controlled_pairs.csv"

    raw_fields = [
        "model",
        "safety_split",
        "language",
        "condition",
        "outcome_group",
        "text_decision",
        "audio_decision",
        "family",
        "layer",
        "pair_id",
        "random_audio_pair_id",
        "matched_cosine",
        "random_unmatched_cosine",
        "matched_minus_random",
    ]

    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        w.writerows(pair_rows)

    # --------------------------------------------------------------
    # Main requested summary:
    # safety + language + 00/01/10/11 + family + layer
    #
    # Random control was STILL condition-matched internally.
    # --------------------------------------------------------------
    main_keys = [
        "model",
        "safety_split",
        "language",
        "outcome_group",
        "family",
        "layer",
    ]

    main_rows = summarise(pair_rows, main_keys)

    main_path = out_root / "by_language_outcome.csv"

    main_fields = main_keys + [
        "n_pairs",
        "n_random_control_pairs",
        "matched_cosine_mean",
        "matched_cosine_std",
        "random_unmatched_cosine_mean",
        "random_unmatched_cosine_std",
        "matched_minus_random_mean",
        "matched_minus_random_std",
        "n_finite_differences",
    ]

    with main_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=main_fields)
        w.writeheader()
        w.writerows(main_rows)

    # --------------------------------------------------------------
    # Extra most-controlled summary including rewrite condition.
    # --------------------------------------------------------------
    condition_keys = [
        "model",
        "safety_split",
        "language",
        "condition",
        "outcome_group",
        "family",
        "layer",
    ]

    condition_rows = summarise(
        pair_rows,
        condition_keys,
    )

    condition_path = (
        out_root
        / "by_language_condition_outcome.csv"
    )

    condition_fields = condition_keys + [
        "n_pairs",
        "n_random_control_pairs",
        "matched_cosine_mean",
        "matched_cosine_std",
        "random_unmatched_cosine_mean",
        "random_unmatched_cosine_std",
        "matched_minus_random_mean",
        "matched_minus_random_std",
        "n_finite_differences",
    ]

    with condition_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=condition_fields,
        )
        w.writeheader()
        w.writerows(condition_rows)

    print()
    print(f"[WROTE] {raw_path}")
    print(f"[WROTE] {main_path}")
    print(f"[WROTE] {condition_path}")

    # --------------------------------------------------------------
    # Compact terminal rollup:
    # max-depth layer for each family/group/language.
    # --------------------------------------------------------------
    print()
    print("===== FINAL-LAYER CONTROLLED SUMMARY =====")

    max_layer = {}

    for r in main_rows:
        key = (
            r["safety_split"],
            r["language"],
            r["outcome_group"],
            r["family"],
        )
        max_layer[key] = max(
            max_layer.get(key, -1),
            int(r["layer"]),
        )

    for r in main_rows:
        key = (
            r["safety_split"],
            r["language"],
            r["outcome_group"],
            r["family"],
        )

        if int(r["layer"]) != max_layer[key]:
            continue

        print(
            f"{model:12s} | "
            f"{r['safety_split']:7s} | "
            f"{r['language']:8s} | "
            f"{r['outcome_group']} | "
            f"{r['family']:14s} | "
            f"L{int(r['layer']):02d} | "
            f"N={r['n_pairs']:4d} | "
            f"Nctrl={r['n_random_control_pairs']:4d} | "
            f"matched={r['matched_cosine_mean']:+.4f} | "
            f"random={r['random_unmatched_cosine_mean']:+.4f} | "
            f"delta={r['matched_minus_random_mean']:+.4f}"
        )

print()
print("DONE")
