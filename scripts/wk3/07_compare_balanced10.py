#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    records = {}

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)
            task_id = row["task_id"]

            if task_id in records:
                raise RuntimeError(f"Duplicate task_id: {task_id}")

            records[task_id] = row

    return records


def preview(text):
    return " ".join(str(text).splitlines()[:2])[:250]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("qwen2_audio", "qwen25_omni", "kimi_audio"),
        required=True,
    )
    parser.add_argument("--boundary", required=True, type=int)
    parser.add_argument("--strength", required=True, type=float)
    args = parser.parse_args()

    original_dir = (
        ROOT / "results/audio_jailbreak_clean_benchmark" / args.model
    )

    steered_dir = (
        ROOT / "results/INITIAL_BENCHMARK" / args.model
        / "wk3"
        / f"steering_layer{args.boundary}_text_svd1"
        / "balanced10"
        / f"strength_{args.strength:+.6f}"
    )

    totals = Counter()

    print(
        f"\nMODEL={args.model} BOUNDARY={args.boundary} "
        f"STRENGTH={args.strength:+.1f}"
    )

    for modality in ("text", "audio"):
        filename = f"responses.all.{modality}.jsonl"

        original = load(original_dir / filename)
        steered = load(steered_dir / filename)

        if len(steered) != 10:
            raise RuntimeError(
                f"Expected 10 {modality} responses, found {len(steered)}"
            )

        counts = Counter()

        print(f"\n========== {modality.upper()} ==========")

        for task_id, row in sorted(steered.items()):
            if task_id not in original:
                raise RuntimeError(f"Missing original response: {task_id}")

            split = row["split"]
            if split not in ("benign", "harmful"):
                raise RuntimeError(f"Unexpected split: {split}")

            metadata = row["steering"]

            if metadata["layer_boundary"] != args.boundary:
                raise RuntimeError(f"Wrong boundary: {task_id}")
            if metadata["strength"] != args.strength:
                raise RuntimeError(f"Wrong strength: {task_id}")
            if metadata["direction_source"] != "text":
                raise RuntimeError(f"Wrong direction source: {task_id}")
            if metadata["direction_method"] != "svd_1":
                raise RuntimeError(f"Wrong direction method: {task_id}")
            if metadata["hook_edits"] != 1:
                raise RuntimeError(f"Wrong intervention count: {task_id}")

            a = original[task_id]["response"]
            b = row["response"]

            different = a != b

            counts[split] += 1
            totals[(split, modality, "changed")] += int(different)

            print(
                f"{split.upper():7} {task_id}: "
                f"{'DIFFERENT' if different else 'SAME'}"
            )

            if different:
                print("  ORIGINAL:", preview(a))
                print("  STEERED: ", preview(b))

        if counts != {"benign": 5, "harmful": 5}:
            raise RuntimeError(
                f"Unexpected {modality} split counts: {counts}"
            )

    print("\n========== SUMMARY ==========")

    for split in ("benign", "harmful"):
        for modality in ("text", "audio"):
            n = totals[(split, modality, "changed")]
            print(f"{split:7} {modality:5}: CHANGED={n}/5")

    print(f"TOTAL CHANGED={sum(totals.values())}/20")
    print("STEERING METADATA: PASS")


if __name__ == "__main__":
    main()
