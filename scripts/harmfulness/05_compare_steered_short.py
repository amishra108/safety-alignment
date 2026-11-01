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
    lines = str(response).splitlines()
    return "\n".join(lines[:2])[:350]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("qwen2_audio", "qwen25_omni", "kimi_audio"),
        required=True,
    )
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    args = parser.parse_args()

    original_dir = (
        ROOT / "results/audio_jailbreak_clean_benchmark" / args.model
    )
    steered_dir = (
        ROOT / "results/INITIAL_BENCHMARK" / args.model
        / "wk3"
        / f"steering_layer{args.boundary}_text_svd1"
        / "unsafe_audio10"
        / f"strength_{args.strength:+.6f}"
    )

    overall_same = 0
    overall_different = 0

    for modality in ("text", "audio"):
        filename = f"responses.all.{modality}.jsonl"

        original = load(original_dir / filename)
        steered = load(steered_dir / filename)

        if len(steered) != 10:
            raise RuntimeError(
                f"{modality}: expected 10 results; got {len(steered)}"
            )

        same = 0
        different = 0

        print(f"\n{'=' * 85}")
        print(
            f"{args.model} | boundary={args.boundary} "
            f"| strength={args.strength:+.1f} | {modality}"
        )
        print("=" * 85)

        for task_id in sorted(steered):
            if task_id not in original:
                raise RuntimeError(f"Original missing: {task_id}")

            a = original[task_id]["response"]
            b = steered[task_id]["response"]

            equal = a == b
            same += int(equal)
            different += int(not equal)

            print(f"\n{task_id}: {'SAME' if equal else 'DIFFERENT'}")

            if not equal:
                print("ORIGINAL:")
                print(preview(a))
                print("STEERED:")
                print(preview(b))

        overall_same += same
        overall_different += different

        print(f"\n{modality.upper()}: SAME={same}/10 DIFFERENT={different}/10")

    print(f"\n{'=' * 85}")
    print(
        f"OVERALL: SAME={overall_same}/20 "
        f"DIFFERENT={overall_different}/20"
    )


if __name__ == "__main__":
    main()
