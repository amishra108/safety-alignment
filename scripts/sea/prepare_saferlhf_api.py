#!/usr/bin/env python3

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("data/saferlhf")
CHECKPOINT = OUT / "download_checkpoint.json"
API = "https://datasets-server.huggingface.co/rows"
DATASET = "PKU-Alignment/PKU-SafeRLHF"
BATCH = 100
MAX_ROWS = 73900
REQUEST_DELAY = 3
MAX_RETRIES = 12


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(temp, path)


def fetch(offset):
    params = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": BATCH,
    })

    request = urllib.request.Request(
        f"{API}?{params}",
        headers={"User-Agent": "SEA-research/1.0"},
    )

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)

        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise

            retry_after = exc.headers.get("Retry-After")
            try:
                delay = max(1, int(retry_after))
            except (TypeError, ValueError):
                delay = min(60 * (attempt + 1), 300)

            print(
                f"Rate limited (429) at offset {offset}. "
                f"Retry {attempt + 1}/{MAX_RETRIES} "
                f"in {delay}s.",
                flush=True,
            )
            time.sleep(delay)

        except (TimeoutError, urllib.error.URLError) as exc:
            delay = min(10 * (attempt + 1), 60)
            print(
                f"Connection failed: {exc}. "
                f"Retrying in {delay}s.",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Request failed after {MAX_RETRIES} attempts at offset {offset}. "
        "Progress is checkpointed; rerun to resume."
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT.exists():
        state = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        print(
            f"Resuming from offset {state['offset']}",
            flush=True,
        )
    else:
        state = {
            "offset": 0,
            "harmful": [],
            "harmless": [],
        }
        print("Starting new download.", flush=True)

    harmful = state["harmful"]
    harmless = state["harmless"]
    seen = {
        r["prompt"]
        for r in harmful + harmless
    }

    for offset in range(state["offset"], MAX_ROWS, BATCH):
        data = fetch(offset)
        rows = data["rows"]

        if not rows:
            break

        for item in rows:
            if item.get("truncated_cells"):
                raise RuntimeError(
                    f"Truncated API record at index {item['row_idx']}"
                )

            row = item["row"]
            prompt = row["prompt"].strip()

            if not prompt or prompt in seen:
                continue

            safer_id = int(row["safer_response_id"])
            rejected_id = 1 - safer_id

            chosen = row[f"response_{safer_id}"]
            rejected = row[f"response_{rejected_id}"]

            if not chosen or not rejected:
                continue

            rejected_severity = row[
                f"response_{rejected_id}_severity_level"
            ]
            severity_0 = row["response_0_severity_level"]
            severity_1 = row["response_1_severity_level"]

            if rejected_severity == 3 and len(harmful) < 2000:
                category = "harmful"
                destination = harmful

            elif (
                severity_0 == 0
                and severity_1 == 0
                and len(harmless) < 1000
            ):
                category = "harmless"
                destination = harmless

            else:
                continue

            record = dict(row)
            record.update({
                "source_index": item["row_idx"],
                "category": category,
                "chosen_response": chosen,
                "rejected_response": rejected,
                "chosen_response_id": safer_id,
                "rejected_response_id": rejected_id,
                "chosen_is_safe": row[f"is_response_{safer_id}_safe"],
                "rejected_is_safe": row[f"is_response_{rejected_id}_safe"],
            })

            destination.append(record)
            seen.add(prompt)

        state["offset"] = offset + len(rows)
        save_json(CHECKPOINT, state)

        print(
            f"Scanned {state['offset']} | "
            f"harmful {len(harmful)}/2000 | "
            f"harmless {len(harmless)}/1000",
            flush=True,
        )

        if len(harmful) == 2000 and len(harmless) == 1000:
            break

        if len(rows) < BATCH:
            break

        time.sleep(REQUEST_DELAY)

    if len(harmful) != 2000 or len(harmless) != 1000:
        raise RuntimeError(
            f"Insufficient examples: "
            f"harmful={len(harmful)}, harmless={len(harmless)}. "
            "Progress has been saved."
        )

    for filename, records in (
        ("harmful_2000.jsonl", harmful),
        ("harmless_1000.jsonl", harmless),
        ("sea_3000.jsonl", harmful + harmless),
    ):
        path = OUT / filename
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {filename}: {len(records)}", flush=True)

    CHECKPOINT.unlink(missing_ok=True)

    print("SUCCESS: 3000 examples saved.", flush=True)


if __name__ == "__main__":
    main()
