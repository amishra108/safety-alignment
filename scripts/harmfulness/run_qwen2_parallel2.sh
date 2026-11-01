#!/usr/bin/env bash
source /research/milsrg1/user_workspace/am3244/venvs/air213-safety-qwen/bin/activate
cd /research/milsrg1/user_workspace/am3244/safety-alignment

set -uo pipefail

SCRIPT="scripts/wk3/04_qwen2_dualgpu_steering.py"
ROOT="results/INITIAL_BENCHMARK/qwen2_audio/wk3/steering_layer16_text_svd1_parallel2/full_6400/strength_-1.000000"
LOGDIR="$ROOT/worker_logs"
mkdir -p "$LOGDIR"

if [[ $(nvidia-smi --list-gpus | wc -l) -lt 2 ]]; then
    echo "ERROR: Two visible GPUs required."
    exit 1
fi

for i in 0 1; do
    file=$(printf '%s/shards/responses.shard%03d-of-002.jsonl' "$ROOT" "$i")
    if [[ -f "$file" ]]; then
        echo "EXISTING SHARD $i: $(wc -l < "$file") records"
    else
        echo "EXISTING SHARD $i: not started"
    fi
done

run_worker() {
    local gpu="$1"
    local shard="$2"

    (
        set -o pipefail
        CUDA_VISIBLE_DEVICES="$gpu" python -u "$SCRIPT" \
            --model qwen2_audio \
            --selection default \
            --test-boundary 16 \
            --modality both \
            --strength=-1.0 \
            --num-shards 2 \
            --shard-index "$shard" \
            --full-run 2>&1 |
            sed -u "s/^/[GPU $gpu SHARD $shard] /" |
            tee "$LOGDIR/shard${shard}.log"
    )
}

echo "========== STARTING TWO INDEPENDENT WORKERS =========="

run_worker 0 0 &
pid0=$!

run_worker 1 1 &
pid1=$!

echo "SHARD 0 WORKER PID: $pid0"
echo "SHARD 1 WORKER PID: $pid1"

stop_workers() {
    kill -- "-$pid0" -- "-$pid1" 2>/dev/null || true
    kill "$pid0" "$pid1" 2>/dev/null || true
}

trap stop_workers INT TERM

wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?

trap - INT TERM

echo "SHARD 0 EXIT: $rc0"
echo "SHARD 1 EXIT: $rc1"

if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
    echo "ERROR: Worker failure. See $LOGDIR/shard0.log and shard1.log"
    exit 1
fi

echo "========== VALIDATING SHARDS =========="

python - <<'PY'
import json
from pathlib import Path

root = Path("results/INITIAL_BENCHMARK/qwen2_audio/wk3/steering_layer16_text_svd1_parallel2/full_6400/strength_-1.000000/shards")
seen = set()

for shard in range(2):
    path = root / f"responses.shard{shard:03d}-of-002.jsonl"
    assert path.is_file(), f"Missing shard: {path}"
    local = set()

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            task = row["task_id"]
            assert task not in local, f"Duplicate in shard {shard}: {task}"
            assert task not in seen, f"Overlap between shards: {task}"
            assert row["model"] == "qwen2_audio"
            assert row["steering"]["strength"] == -1.0
            assert row["steering"]["layer_boundary"] == 16
            local.add(task)

    assert len(local) == 3200, f"Shard {shard}: {len(local)}/3200"
    seen.update(local)
    print(f"SHARD {shard}: PASS, {len(local)} unique responses")

assert len(seen) == 6400
print("PASS: 6400 unique responses; no overlap.")
PY

rc=$?
if [[ "$rc" -ne 0 ]]; then
    echo "VALIDATION FAILED — merge cancelled."
    exit "$rc"
fi

echo "========== MERGING =========="

CUDA_VISIBLE_DEVICES=0 python -u "$SCRIPT" \
    --model qwen2_audio \
    --selection default \
    --test-boundary 16 \
    --modality both \
    --strength=-1.0 \
    --num-shards 2 \
    --merge-only \
    --full-run

rc=$?
echo "MERGE EXIT CODE: $rc"
exit "$rc"
