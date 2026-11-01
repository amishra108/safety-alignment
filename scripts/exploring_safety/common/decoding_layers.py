#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RESULTS = ROOT / "results/INITIAL_BENCHMARK"
MANIFEST = ROOT / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"

MODELS = ["qwen25_omni", "qwen2_audio", "kimi_audio"]
GROUPS = [
    "harmful_text0_audio0",
    "harmful_text0_audio1",
    "harmful_text1_audio0",
    "harmful_text1_audio1",
    "benign_text0_audio0",
    "benign_text0_audio1",
    "benign_text1_audio0",
    "benign_text1_audio1",
]

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

def strip_modality(task_id):
    for suffix in ("__text", "__audio"):
        if task_id.endswith(suffix):
            return task_id[:-len(suffix)]
    raise RuntimeError(f"Bad task_id: {task_id}")

def task_folder(row):
    return strip_modality(row["task_id"])

def stable_rank(seed, key):
    x = hashlib.sha256(f"{seed}\0{key}".encode()).digest()
    return int.from_bytes(x[:16], "big")

def prepare_selections(seed=42, cap=25):
    for model in MODELS:
        fold = RESULTS / model / "5fold/fold_1.jsonl"
        rows = read_jsonl(fold)

        paired = {}
        for r in rows:
            tid = r["task_id"]
            key = strip_modality(tid)
            mod = r["modality"]
            if mod not in ("text", "audio"):
                continue
            if mod in paired.setdefault(key, {}):
                raise RuntimeError(f"{model}: duplicate {key}/{mod}")
            paired[key][mod] = r

        buckets = {g: [] for g in GROUPS}

        for key, x in paired.items():
            if set(x) != {"text", "audio"}:
                continue

            t, a = x["text"], x["audio"]

            for field in ("safety_split", "language", "condition"):
                if str(t[field]) != str(a[field]):
                    raise RuntimeError(
                        f"{model}: pair metadata mismatch {key}/{field}"
                    )

            split = t["safety_split"]
            td, ad = int(t["decision"]), int(a["decision"])
            group = f"{split}_text{td}_audio{ad}"

            if group not in buckets:
                raise RuntimeError(f"Unexpected group {group}")

            # Use TEXT member as canonical source row, like old Omni decoder.
            rec = dict(t)
            rec["paired_text_task_id"] = t["task_id"]
            rec["paired_audio_task_id"] = a["task_id"]
            rec["text_decision"] = td
            rec["audio_decision"] = ad
            rec["outcome_group"] = f"text{td}_audio{ad}"
            rec["sampling_seed"] = seed
            buckets[group].append((stable_rank(seed, key), rec))

        print(f"\n===== {model} =====")
        for group in GROUPS:
            ranked = sorted(buckets[group], key=lambda z: z[0])
            selected = [r for _, r in ranked[:cap]]

            out = RESULTS / model / "decoding_layers" / "selections" / f"{group}.jsonl"
            write_jsonl(out, selected)

            print(
                f"{group:24s} "
                f"available={len(ranked):4d} "
                f"selected={len(selected):2d} "
                f"-> {out}"
            )

def load_manifest():
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["example_id"]: r for r in rows}

def manifest_for(row, index):
    key = strip_modality(row["task_id"])
    if key not in index:
        raise RuntimeError(f"No manifest row for {key}")
    return index[key]

def local_rows(rows, num_shards, shard_index):
    n = len(rows)
    base, rem = divmod(n, num_shards)
    start = shard_index * base + min(shard_index, rem)
    count = base + (1 if shard_index < rem else 0)
    return rows[start:start + count]

def completed_layers(path):
    path = Path(path)
    if not path.exists():
        return set()
    out = set()
    for r in read_jsonl(path):
        x = int(r["layer"])
        if x in out:
            raise RuntimeError(f"Duplicate layer {x} in {path}")
        out.add(x)
    return out

def run_backend(adapter, num_shards, shard_index, max_new_tokens, groups=None, start=0, limit=None):
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise RuntimeError("Invalid shard configuration")

    groups = groups or GROUPS
    manifest = load_manifest()

    print("=" * 80)
    print("DECODING LAYERS")
    print("=" * 80)
    print("model:", adapter.model_name)
    print("shard:", f"{shard_index}/{num_shards}")
    print("max_new_tokens:", max_new_tokens)
    print("stages:", len(adapter.stages))

    for group in groups:
        source = (
            RESULTS / adapter.model_name
            / "decoding_layers" / "selections"
            / f"{group}.jsonl"
        )
        if not source.exists():
            raise RuntimeError(
                f"Missing selection {source}; run shared --prepare first"
            )

        all_rows = read_jsonl(source)
        selected_rows = all_rows[start:] if limit is None else all_rows[start:start + limit]
        rows = local_rows(selected_rows, num_shards, shard_index)

        print(
            f"\n===== {group} | "
            f"total={len(all_rows)} selected={len(selected_rows)} "
            f"range={start}:{'end' if limit is None else start + limit} "
            f"local={len(rows)} =====",
            flush=True,
        )

        for i, row in enumerate(rows, 1):
            mrow = manifest_for(row, manifest)
            outdir = (
                RESULTS / adapter.model_name
                / "decoding_layers"
                / "examples"
                / group
                / task_folder(row)
            )
            outdir.mkdir(parents=True, exist_ok=True)

            prepared = {}
            for modality in sorted({x[0] for x in adapter.stages}):
                prepared[modality] = adapter.prepare(row, mrow, modality)

            print(
                f"[{i}/{len(rows)}] {task_folder(row)}",
                flush=True,
            )

            for modality, family, layer in adapter.stages:
                outfile = outdir / adapter.output_name(modality, family)
                if layer in completed_layers(outfile):
                    print(
                        f"  [skip] {modality}/{family}/{layer:02d}",
                        flush=True,
                    )
                    continue

                print(
                    f"  [run ] {modality}/{family}/{layer:02d}",
                    flush=True,
                )

                response = adapter.generate(
                    prepared[modality],
                    modality,
                    family,
                    layer,
                    max_new_tokens,
                )

                rec = {
                    k: v for k, v in row.items()
                    if k not in {
                        "response_text", "decision_text",
                        "response_audio", "decision_audio",
                    }
                }
                rec["layer"] = layer
                rec["response"] = response

                append_jsonl(outfile, rec)

                print(
                    f"  [done] chars={len(response)}",
                    flush=True,
                )

    print("\n[OK] decoding shard complete")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap", type=int, default=25)
    args = p.parse_args()

    if args.prepare:
        prepare_selections(args.seed, args.cap)
    else:
        p.error("Use --prepare")

if __name__ == "__main__":
    main()
