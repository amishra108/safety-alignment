#!/usr/bin/env python3
import argparse
import gc
import json
from pathlib import Path

import torch

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
ALL_JUDGED = ROOT / "results/INITIAL_BENCHMARK/all_judged"

VERSION = "prefill_last_prompt_8bin_v2"
AUDIO_BINS = 8


def load_rows(model_name):
    import csv

    manifest_path = (
        ROOT
        / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        manifest_rows = list(csv.DictReader(f))

    manifest = {}

    for m in manifest_rows:
        eid = m["example_id"]

        if eid in manifest:
            raise RuntimeError(
                f"Duplicate example_id in manifest: {eid}"
            )

        manifest[eid] = m

    rows = []

    for safety in ("benign", "harmful"):
        for modality in ("text", "audio"):
            judged_path = (
                ALL_JUDGED
                / f"{model_name}.{modality}.{safety}.adjudicated.jsonl"
            )

            if not judged_path.exists():
                raise FileNotFoundError(judged_path)

            with judged_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                for ln, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    judged = json.loads(line)

                    if "task_id" not in judged:
                        raise RuntimeError(
                            f"{judged_path}:{ln}: missing task_id"
                        )

                    task_id = judged["task_id"]
                    suffix = f"__{modality}"

                    if not task_id.endswith(suffix):
                        raise RuntimeError(
                            f"{task_id}: expected suffix {suffix}"
                        )

                    example_id = task_id[:-len(suffix)]

                    if example_id not in manifest:
                        raise RuntimeError(
                            f"{task_id}: no manifest example_id "
                            f"{example_id}"
                        )

                    m = manifest[example_id]

                    for field in (
                        "language",
                        "condition",
                    ):
                        if (
                            field in judged
                            and judged[field] is not None
                            and str(judged[field]) != str(m[field])
                        ):
                            raise RuntimeError(
                                f"{task_id}: {field} mismatch: "
                                f"judged={judged[field]!r}, "
                                f"manifest={m[field]!r}"
                            )

                    row = dict(judged)

                    row["task_id"] = task_id
                    row["example_id"] = example_id
                    row["modality"] = modality
                    row["safety_split"] = safety

                    row["prompt_text"] = m["prompt_text"]
                    row["audio_path"] = m["audio_path"]

                    for field in (
                        "base_prompt_id",
                        "language",
                        "condition",
                    ):
                        if field in m and m[field] != "":
                            row[field] = m[field]

                    row["_source_file"] = judged_path.name

                    if not row["prompt_text"].strip():
                        raise RuntimeError(
                            f"{task_id}: empty prompt_text"
                        )

                    if modality == "audio":
                        wav = ROOT / row["audio_path"]

                        if not wav.is_file():
                            raise RuntimeError(
                                f"{task_id}: missing WAV: {wav}"
                            )

                    rows.append(row)

    if len(rows) != 6400:
        raise RuntimeError(
            f"Expected 6400 rows, got {len(rows)}"
        )

    ids = [r["task_id"] for r in rows]

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate task_id")

    # Preserve deterministic source-file order.
    # This deliberately matches the Omni extraction strategy:
    # rows[shard_index::num_shards].
    return rows


def select_shard(rows, num_shards, shard_index):
    if num_shards < 1:
        raise RuntimeError("--num-shards must be >=1")

    if not 0 <= shard_index < num_shards:
        raise RuntimeError("bad --shard-index")

    # EXACT strategy used by the working Qwen2.5-Omni extractor.
    selected = rows[shard_index::num_shards]

    # For the full 6400-row benchmark with four GPUs,
    # each shard must contain exactly 1600 rows.
    if len(rows) == 6400 and num_shards == 4:
        if len(selected) != 1600:
            raise RuntimeError(
                f"Expected 1600 rows on shard, got {len(selected)}"
            )

        n_text = sum(
            r["modality"] == "text"
            for r in selected
        )
        n_audio = sum(
            r["modality"] == "audio"
            for r in selected
        )
        n_benign = sum(
            r["safety_split"] == "benign"
            for r in selected
        )
        n_harmful = sum(
            r["safety_split"] == "harmful"
            for r in selected
        )

        if (
            n_text != 800
            or n_audio != 800
            or n_benign != 800
            or n_harmful != 800
        ):
            raise RuntimeError(
                "Bad four-way shard composition: "
                f"text={n_text} audio={n_audio} "
                f"benign={n_benign} harmful={n_harmful}"
            )

    return selected


def first_tensor(x):
    if torch.is_tensor(x):
        return x

    if isinstance(x, (tuple, list)):
        for y in x:
            t = first_tensor(y)
            if t is not None:
                return t

    if isinstance(x, dict):
        for y in x.values():
            t = first_tensor(y)
            if t is not None:
                return t

    return None


def as_btd(t, batch_size, label):
    if t is None:
        raise RuntimeError(f"{label}: no tensor")

    if t.ndim != 3:
        raise RuntimeError(
            f"{label}: expected rank 3, got {tuple(t.shape)}"
        )

    # Standard HF layout [B,T,D]
    if t.shape[0] == batch_size:
        return t

    # Defensive support for [T,B,D]
    if t.shape[1] == batch_size:
        return t.transpose(0, 1)

    raise RuntimeError(
        f"{label}: cannot identify batch dimension "
        f"for shape={tuple(t.shape)} batch={batch_size}"
    )


def pool_equal_bins(seq, valid_len, bins=AUDIO_BINS):
    valid_len = int(valid_len)
    valid_len = max(1, min(valid_len, seq.shape[0]))

    seq = seq[:valid_len]

    out = []
    for b in range(bins):
        a = (b * valid_len) // bins
        z = ((b + 1) * valid_len) // bins

        if z <= a:
            a = min(a, valid_len - 1)
            z = a + 1

        out.append(seq[a:z].float().mean(0))

    return torch.stack(out)


class BoundaryCapture:
    def __init__(self, layers, family, mode):
        self.layers = list(layers)
        self.family = family
        self.mode = mode

        if not self.layers:
            raise RuntimeError(f"{family}: zero layers")

        self.state = {}
        self.batch_size = None
        self.valid_lengths = None
        self.group_sizes = None
        self.handles = []

        self.handles.append(
            self.layers[0].register_forward_pre_hook(self._pre)
        )

        for i, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.register_forward_hook(self._post_factory(i))
            )

    def reset(
        self,
        batch_size,
        valid_lengths=None,
        group_sizes=None,
    ):
        self.state.clear()
        self.batch_size = int(batch_size)

        if valid_lengths is None:
            self.valid_lengths = None
        else:
            self.valid_lengths = [
                int(x) for x in valid_lengths
            ]

        if group_sizes is None:
            self.group_sizes = None
        else:
            self.group_sizes = [
                int(x) for x in group_sizes
            ]

            if len(self.group_sizes) != self.batch_size:
                raise RuntimeError(
                    f"{self.family}: group_sizes count "
                    f"{len(self.group_sizes)} != logical batch "
                    f"{self.batch_size}"
                )

            if any(x < 1 for x in self.group_sizes):
                raise RuntimeError(
                    f"{self.family}: invalid group_sizes="
                    f"{self.group_sizes}"
                )

    def _capture(self, obj, boundary):
        t = first_tensor(obj)

        # Decoder / ordinary one-sequence-per-example path.
        if self.mode == "last":
            t = as_btd(
                t,
                self.batch_size,
                f"{self.family}[{boundary}]",
            )

            # All text processors use left padding.
            value = t[:, -1, :]

        elif self.mode == "pool8":
            if self.valid_lengths is None:
                raise RuntimeError(
                    f"{self.family}: missing encoder lengths"
                )

            # Some architectures (AF3) flatten multiple 30 s
            # windows from one logical audio example into the
            # encoder batch dimension.
            encoder_batch = (
                sum(self.group_sizes)
                if self.group_sizes is not None
                else self.batch_size
            )

            t = as_btd(
                t,
                encoder_batch,
                f"{self.family}[{boundary}]",
            )

            if len(self.valid_lengths) != encoder_batch:
                raise RuntimeError(
                    f"{self.family}: encoder lengths count "
                    f"{len(self.valid_lengths)} != encoder batch "
                    f"{encoder_batch}"
                )

            # Normal architecture: one encoder sequence per example.
            if self.group_sizes is None:
                value = torch.stack([
                    pool_equal_bins(
                        t[i],
                        self.valid_lengths[i],
                    )
                    for i in range(self.batch_size)
                ])

            # AF3: reconstruct each original audio example from its
            # consecutive 30-second encoder windows before 8-bin pooling.
            else:
                examples = []
                window_index = 0

                for n_windows in self.group_sizes:
                    pieces = []

                    for _ in range(n_windows):
                        valid_len = self.valid_lengths[window_index]
                        valid_len = max(
                            1,
                            min(valid_len, t.shape[1]),
                        )

                        pieces.append(
                            t[
                                window_index,
                                :valid_len,
                                :
                            ]
                        )

                        window_index += 1

                    reconstructed = torch.cat(
                        pieces,
                        dim=0,
                    )

                    examples.append(
                        pool_equal_bins(
                            reconstructed,
                            reconstructed.shape[0],
                        )
                    )

                if window_index != encoder_batch:
                    raise RuntimeError(
                        f"{self.family}: consumed "
                        f"{window_index}/{encoder_batch} windows"
                    )

                value = torch.stack(examples)

        else:
            raise RuntimeError(self.mode)

        self.state[boundary] = (
            value.detach()
            .to("cpu", dtype=torch.float16)
            .contiguous()
        )

    def _pre(self, module, args):
        if 0 not in self.state:
            self._capture(args, 0)

    def _post_factory(self, boundary):
        def hook(module, args, output):
            if boundary not in self.state:
                self._capture(output, boundary)
        return hook

    def stack(self):
        n = len(self.layers)
        expected = set(range(n + 1))

        if set(self.state) != expected:
            raise RuntimeError(
                f"{self.family}: incomplete capture: "
                f"got={sorted(self.state)} expected={sorted(expected)}"
            )

        # decoder: [L+1,B,D]
        # encoder: [L+1,B,8,D]
        return torch.stack(
            [self.state[i] for i in range(n + 1)]
        ).contiguous()

    def close(self):
        for h in self.handles:
            h.remove()


def resolve_attr(root, candidates, label):
    for path in candidates:
        x = root
        ok = True

        for part in path.split("."):
            if not hasattr(x, part):
                ok = False
                break
            x = getattr(x, part)

        if ok:
            print(
                f"[architecture] {label}: "
                f"{path} | layers={len(x)}",
                flush=True,
            )
            return x, path

    raise RuntimeError(
        f"Could not resolve {label}; tried {candidates}"
    )


def is_oom(exc):
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(exc).lower()
    )


def atomic_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def representation_current(path, model, task_id, modality):
    if not path.exists():
        return False

    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        return (
            obj.get("representation_version") == VERSION
            and obj.get("model") == model
            and obj.get("task_id") == task_id
            and obj.get("modality") == modality
        )

    except Exception:
        return False


def parser():
    p = argparse.ArgumentParser()

    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)

    p.add_argument(
        "--modality",
        choices=("all", "text", "audio"),
        default="all",
    )

    p.add_argument("--text-batch-size", type=int, default=16)
    p.add_argument("--audio-batch-size", type=int, default=8)

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preflight", action="store_true")

    return p


def run_generic(adapter):
    args = parser().parse_args()

    rows = load_rows(adapter.model_name)

    if args.modality != "all":
        rows = [
            r for r in rows
            if r["modality"] == args.modality
        ]

    rows = select_shard(
        rows,
        args.num_shards,
        args.shard_index,
    )

    if args.limit is not None:
        rows = rows[:args.limit]

    out_root = (
        ROOT
        / "results/INITIAL_BENCHMARK"
        / adapter.model_name
        / "5fold"
        / "representations_shared"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REPRESENTATION EXTRACTION")
    print("model      :", adapter.model_name)
    print("rows       :", len(rows))
    print("shard      :", f"{args.shard_index}/{args.num_shards}")
    print("text batch :", args.text_batch_size)
    print("audio batch:", args.audio_batch_size)
    print("output     :", out_root)
    print("decisions  : IGNORED")
    print("=" * 80)

    ctx = adapter.load()

    if args.preflight:
        print("PREFLIGHT OK")
        return

    captures = adapter.make_captures(ctx)

    saved = 0
    skipped = 0

    def process(batch):
        nonlocal saved

        modality = batch[0]["modality"]

        for r in batch:
            if r["modality"] != modality:
                raise RuntimeError("mixed-modality batch")

        prepared = adapter.prepare_batch(
            ctx,
            batch,
            modality,
        )

        for name, cap in captures.items():
            if cap is None:
                continue

            if name == "audio_encoder":
                if modality == "audio":
                    cap.reset(
                        len(batch),
                        prepared["encoder_lengths"],
                        prepared.get("encoder_group_sizes"),
                    )
                else:
                    cap.reset(len(batch))
            else:
                cap.reset(len(batch))

        adapter.forward(
            ctx,
            prepared["inputs"],
        )

        stacks = {}

        for name, cap in captures.items():
            if cap is None:
                continue

            if name == "audio_encoder" and modality != "audio":
                continue

            stacks[name] = cap.stack()

        for j, row in enumerate(batch):
            payload = {
                "representation_version": VERSION,
                "model": adapter.model_name,
                "task_id": row["task_id"],
                "base_prompt_id": row.get("base_prompt_id"),
                "language": row.get("language"),
                "condition": row.get("condition"),
                "safety_split": row["safety_split"],
                "modality": row["modality"],
                "source_file_snapshot": row["_source_file"],
                # decision deliberately NOT saved
            }

            for name, stack in stacks.items():
                payload[name] = stack[:, j].contiguous()
                payload[f"{name}_boundaries"] = stack.shape[0]

            payload.update(
                adapter.extra_metadata(ctx)
            )

            atomic_save(
                payload,
                out_root / f"{row['task_id']}.pt",
            )

            saved += 1

    def process_backoff(batch):
        try:
            process(batch)

        except Exception as exc:
            if not is_oom(exc):
                raise

            gc.collect()
            torch.cuda.empty_cache()

            if len(batch) == 1:
                print(
                    f"OOM at batch=1: {batch[0]['task_id']}",
                    flush=True,
                )
                raise

            mid = len(batch) // 2

            print(
                f"OOM batch={len(batch)} -> "
                f"{mid}+{len(batch)-mid}",
                flush=True,
            )

            process_backoff(batch[:mid])
            process_backoff(batch[mid:])

    try:
        for modality in ("text", "audio"):
            selected = [
                r for r in rows
                if r["modality"] == modality
            ]

            if not selected:
                continue

            batch_size = (
                args.text_batch_size
                if modality == "text"
                else args.audio_batch_size
            )

            for start in range(0, len(selected), batch_size):
                candidate = selected[
                    start:start + batch_size
                ]

                todo = []

                for row in candidate:
                    path = (
                        out_root
                        / f"{row['task_id']}.pt"
                    )

                    if (
                        not args.overwrite
                        and representation_current(
                            path,
                            adapter.model_name,
                            row["task_id"],
                            row["modality"],
                        )
                    ):
                        skipped += 1
                    else:
                        todo.append(row)

                if todo:
                    process_backoff(todo)

                done = min(
                    start + batch_size,
                    len(selected),
                )

                print(
                    f"[{modality}] "
                    f"{done}/{len(selected)} "
                    f"saved={saved} skipped={skipped}",
                    flush=True,
                )

                gc.collect()
                torch.cuda.empty_cache()

    finally:
        for cap in captures.values():
            if cap is not None:
                cap.close()

    print("=" * 80)
    print("DONE")
    print("saved  :", saved)
    print("skipped:", skipped)
    print("output :", out_root)
