#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(
    "/research/milsrg1/user_workspace/am3244/safety-alignment/"
    "scripts/initial_benchmark/audio_jailbreak/eval_fast_outputs"
)

MODELS = ["qwen2_audio", "qwen25_omni"]
MODALITIES = ["text", "audio"]
CONDITIONS = [
    "original",
    "literal",
    "narrative_framing",
    "content_dilution",
]
LANGUAGES = ["english", "spanish", "chinese", "russian"]

FILES = []
for model in MODELS:
    for modality in MODALITIES:
        FILES.append((
            "main",
            modality,
            model,
            ROOT / modality / model / "responses.jsonl",
        ))
        FILES.append((
            "language_ablation",
            modality,
            model,
            ROOT / "language_ablation" / modality / model / "responses.jsonl",
        ))

rows = []

for dataset, modality, model, path in FILES:
    if not path.exists():
        print(f"[MISSING] {path}")
        continue

    n = 0
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue

            r = json.loads(line)

            if r.get("decision") not in (0, 1):
                raise ValueError(
                    f"{path}:{ln}: bad decision={r.get('decision')!r}"
                )

            if r.get("split") not in ("harmful", "benign"):
                raise ValueError(
                    f"{path}:{ln}: bad split={r.get('split')!r}"
                )

            x = dict(r)
            x["_dataset"] = dataset
            x["_file_modality"] = modality
            x["_file_model"] = model
            rows.append(x)
            n += 1

    print(f"[LOAD] {model:12s} {dataset:18s} {modality:5s} {n:4d}  {path}")

print(f"\n[TOTAL EVALUATED ROWS] {len(rows)}")


def norm_condition(r):
    c = str(r.get("condition", "")).strip().lower()

    aliases = {
        "no_attack": "original",
        "baseline": "original",
        "narrative": "narrative_framing",
        "dilution": "content_dilution",
    }

    return aliases.get(c, c)


def norm_language(r):
    return str(r.get("language") or "unknown").strip().lower()


def stat(group):
    n = len(group)
    ones = sum(int(r["decision"]) for r in group)
    pct = 100.0 * ones / n if n else 0.0
    split = group[0]["split"] if group else ""
    metric = "ASR" if split == "harmful" else "BRR"
    return n, ones, pct, metric


def print_group(title, keys, source_rows):
    grouped = defaultdict(list)

    for r in source_rows:
        key = tuple(fn(r) for _, fn in keys)
        grouped[key].append(r)

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    headers = [name for name, _ in keys]
    print(
        " | ".join(f"{h:<20}" for h in headers)
        + " | N     | decision=1 | metric | percent"
    )
    print("-" * 110)

    for key in sorted(grouped):
        g = grouped[key]
        n, ones, pct, metric = stat(g)

        print(
            " | ".join(f"{str(v):<20}" for v in key)
            + f" | {n:5d}"
            + f" | {ones:10d}"
            + f" | {metric:6s}"
            + f" | {pct:6.2f}%"
        )


# ------------------------------------------------------------------
# 1. Basic/model-level metadata:
# model × dataset × modality × harmful/benign
# ------------------------------------------------------------------
print_group(
    "1. OVERALL: MODEL × DATASET × MODALITY × SPLIT",
    [
        ("model", lambda r: r["model"]),
        ("dataset", lambda r: r["_dataset"]),
        ("modality", lambda r: r["modality"]),
        ("split", lambda r: r["split"]),
    ],
    rows,
)


# ------------------------------------------------------------------
# 2. Attack breakdown.
# Includes ORIGINAL as the useful no-attack baseline plus all 3 attacks.
# ------------------------------------------------------------------
print_group(
    "2. CONDITION: MODEL × DATASET × MODALITY × SPLIT × ATTACK",
    [
        ("model", lambda r: r["model"]),
        ("dataset", lambda r: r["_dataset"]),
        ("modality", lambda r: r["modality"]),
        ("split", lambda r: r["split"]),
        ("attack", norm_condition),
    ],
    rows,
)


# ------------------------------------------------------------------
# 3. Language ablation: collapsed over attack types.
# ------------------------------------------------------------------
lang_rows = [
    r for r in rows
    if r["_dataset"] == "language_ablation"
]

print_group(
    "3. LANGUAGE: MODEL × MODALITY × SPLIT × LANGUAGE",
    [
        ("model", lambda r: r["model"]),
        ("modality", lambda r: r["modality"]),
        ("split", lambda r: r["split"]),
        ("language", norm_language),
    ],
    lang_rows,
)


# ------------------------------------------------------------------
# 4. Full language × attack breakdown.
# ------------------------------------------------------------------
print_group(
    "4. LANGUAGE × ATTACK: MODEL × MODALITY × SPLIT × LANGUAGE × ATTACK",
    [
        ("model", lambda r: r["model"]),
        ("modality", lambda r: r["modality"]),
        ("split", lambda r: r["split"]),
        ("language", norm_language),
        ("attack", norm_condition),
    ],
    lang_rows,
)


# ------------------------------------------------------------------
# 5. Quick per-model text-vs-audio comparison, main experiment only.
# ------------------------------------------------------------------
main_rows = [
    r for r in rows
    if r["_dataset"] == "main"
]

print_group(
    "5. MAIN QUICK VIEW: MODEL × MODALITY × SPLIT",
    [
        ("model", lambda r: r["model"]),
        ("modality", lambda r: r["modality"]),
        ("split", lambda r: r["split"]),
    ],
    main_rows,
)


# ------------------------------------------------------------------
# Sanity/meta
# ------------------------------------------------------------------
print("\n" + "=" * 110)
print("META / SANITY")
print("=" * 110)

for model in MODELS:
    mr = [r for r in rows if r["model"] == model]

    print(f"\n{model}")
    print(f"  rows      : {len(mr)}")
    print(
        "  splits    : "
        + ", ".join(
            f"{x}={sum(r['split'] == x for r in mr)}"
            for x in ("harmful", "benign")
        )
    )
    print(
        "  modalities: "
        + ", ".join(
            f"{x}={sum(r['modality'] == x for r in mr)}"
            for x in MODALITIES
        )
    )

    conditions = sorted(set(norm_condition(r) for r in mr))
    languages = sorted(set(norm_language(r) for r in mr))

    print(f"  conditions: {', '.join(conditions)}")
    print(f"  languages : {', '.join(languages)}")

print()
print("INTERPRETATION:")
print("  harmful: decision=1 -> successful jailbreak -> ASR")
print("  benign : decision=1 -> benign refusal       -> BRR")
