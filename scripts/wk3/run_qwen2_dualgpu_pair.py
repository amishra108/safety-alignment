#!/usr/bin/env python3

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
RUNNER = ROOT / "scripts/wk3/04_qwen2_dualgpu_steering.py"
OUTPUT = (
    ROOT
    / "results/INITIAL_BENCHMARK/qwen2_audio/wk3"
    / "steering_layer16_text_svd1_parallel2/full_6400"
    / "strength_-1.000000"
)
MANIFEST = ROOT / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"

BASE_ARGS = [
    "--model", "qwen2_audio",
    "--selection", "default",
    "--test-boundary", "16",
    "--modality", "both",
    "--strength=-1.0",
    "--num-shards", "2",
    "--full-run",
]


def command(*extra):
    return [sys.executable, "-u", str(RUNNER), *BASE_ARGS, *extra]


def terminate(workers):
    for process in workers:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    for process in workers:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def validate():
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        examples = list(csv.DictReader(f))

    if len(examples) != 3200:
        raise RuntimeError(f"Unexpected manifest size: {len(examples)}")

    seen = set()

    for shard_index in (0, 1):
        path = (
            OUTPUT / "shards"
            / f"responses.shard{shard_index:03d}-of-002.jsonl"
        )

        if not path.is_file():
            raise RuntimeError(f"Missing shard: {path}")

        local = set()

        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                row = json.loads(line)
                task = row["task_id"]
                eid = row["example_id"]
                modality = row["modality"]
                steering = row["steering"]

                if task != f"{eid}__{modality}":
                    raise RuntimeError(f"{path}:{line_no}: invalid task ID")
                if modality not in ("text", "audio"):
                    raise RuntimeError(f"{path}:{line_no}: invalid modality")
                if row["model"] != "qwen2_audio":
                    raise RuntimeError(f"{path}:{line_no}: invalid model")
                if steering["strength"] != -1.0:
                    raise RuntimeError(f"{path}:{line_no}: invalid strength")
                if steering["layer_boundary"] != 16:
                    raise RuntimeError(f"{path}:{line_no}: invalid boundary")

                expected_ids = {
                    r["example_id"]
                    for r in examples[1600 * shard_index:1600 * (shard_index + 1)]
                }

                if eid not in expected_ids:
                    raise RuntimeError(f"{path}:{line_no}: wrong shard")
                if task in local or task in seen:
                    raise RuntimeError(f"{path}:{line_no}: duplicate {task}")

                local.add(task)

        if len(local) != 3200:
            raise RuntimeError(
                f"Shard {shard_index}: {len(local)}/3200 responses"
            )

        seen.update(local)
        print(f"[VALIDATED] shard {shard_index}: 3200 unique", flush=True)

    if len(seen) != 6400:
        raise RuntimeError(f"Expected 6400 unique tasks, got {len(seen)}")

    print("[VALIDATED] 6400 unique responses, no overlap", flush=True)


def main():
    gpu_result = subprocess.run(
        ["nvidia-smi", "--list-gpus"],
        capture_output=True, text=True, check=True,
    )
    gpu_count = len(gpu_result.stdout.splitlines())

    if gpu_count < 2:
        raise RuntimeError(f"Two GPUs required; found {gpu_count}")

    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)

    print(f"[HOST] {os.uname().nodename}", flush=True)
    print(f"[GPUS] {gpu_count}", flush=True)

    for shard in (0, 1):
        path = (
            OUTPUT / "shards"
            / f"responses.shard{shard:03d}-of-002.jsonl"
        )
        if path.exists():
            with path.open(encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        else:
            count = 0
        print(f"[RESUME] shard {shard}: {count}/3200", flush=True)

    workers = []

    try:
        for gpu, shard in ((0, 0), (1, 1)):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                command("--shard-index", str(shard)),
                cwd=ROOT,
                env=env,
                start_new_session=True,
            )
            workers.append(process)
            print(
                f"[STARTED] GPU={gpu} shard={shard} PID={process.pid}",
                flush=True,
            )

        while True:
            statuses = [p.poll() for p in workers]

            if any(rc is not None and rc != 0 for rc in statuses):
                print(f"[WORKER FAILURE] {statuses}", flush=True)
                terminate(workers)
                raise RuntimeError("A GPU worker failed; merge cancelled")

            if all(rc == 0 for rc in statuses):
                break

            time.sleep(2)

    except BaseException:
        terminate(workers)
        raise

    print("[WORKERS COMPLETE] Validating results", flush=True)
    validate()

    print("[MERGE] Starting", flush=True)
    result = subprocess.run(
        command("--merge-only"),
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Merge failed: {result.returncode}")

    print("[SUCCESS] Both shards complete and merged", flush=True)


if __name__ == "__main__":
    main()
