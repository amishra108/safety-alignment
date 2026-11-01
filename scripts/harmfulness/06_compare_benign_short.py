#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row["task_id"]
            if task_id in records:
                raise RuntimeError(f"Duplicate task_id: {task_id}")
            records[task_id] = row

    return records


def preview(response):
    return "\n".join(str(response).splitlines()[:2])[:350]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("qwen2_audio", "qwen25_omni"),
        required=True,
    )
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    original_dir = (
        ROOT / "results/audio_jailbreak_clean_benchmark" / args.model
    )
    steered_dir = (
        ROOT / "results/INITIAL_BENCHMARK" / args.model / "wk3"
        / f"steering_layer{args.boundary}_text_svd1"
        / "benign_audio10"
        / f"strength_{args.strength:+.6f}"
    )

    total_changed = 0

    for modality in ("text", "audio"):
        filename = f"responses.all.{modality}.jsonl"
        original = load(original_dir / filename)
        steered = load(steered_dir / filename)

        if len(steered) != 10:
            raise RuntimeError(
                f"{modality}: expected 10 results, got {len(steered)}"
            )

        changed = 0
        print(f"\n{args.model} | {modality} | boundary {args.boundary}")

        for task_id in sorted(steered):
            if task_id not in original:
                raise RuntimeError(f"Missing original: {task_id}")

            a = original[task_id]["response"]
            b = steered[task_id]["response"]
            different = a != b
            changed += different

            print(f"{task_id}: {'DIFFERENT' if different else 'SAME'}")

            if different:
                print("  ORIGINAL:", preview(a))
                print("  STEERED: ", preview(b))

        total_changed += changed
        print(f"{modality.upper()}: CHANGED={changed}/10")

    print(f"\nTOTAL CHANGED={total_changed}/20")


if __name__ == "__main__":
    main()
