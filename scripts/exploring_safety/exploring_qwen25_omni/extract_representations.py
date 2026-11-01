#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import os
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from qwen_omni_utils import process_mm_info


# Paths

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
MODEL_PATH = ROOT / "models/qwen2.5-omni-7b"

FOLD_FILE = (
    ROOT
    / "results/INITIAL_BENCHMARK/qwen25_omni/5fold"
    / "fold_1/fold_1.jsonl"
)

MANIFEST = (
    ROOT
    / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"
)

OUT_ROOT = (
    ROOT
    / "results/INITIAL_BENCHMARK/qwen25_omni/5fold"
    / "representations_shared"
)

EXPECTED_ROWS = 6400
N_THINKER_LAYERS = 28
N_AUDIO_LAYERS = 32
THINKER_DIM = 3584
AUDIO_DIM = 1280
AUDIO_BINS = 8


# I/O

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Bad JSON {path}:{ln}"
                    ) from exc
    return rows


def load_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_id = {}
    for row in rows:
        eid = row["example_id"]
        if eid in by_id:
            raise RuntimeError(
                f"Duplicate manifest example_id: {eid}"
            )
        by_id[eid] = row

    return by_id


def row_identity(row):
    task_id = row["task_id"]
    modality = row["modality"]
    suffix = f"__{modality}"

    if not task_id.endswith(suffix):
        raise RuntimeError(
            f"Bad task/modality: {task_id} / {modality}"
        )

    return task_id[:-len(suffix)]


def atomic_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )
    torch.save(obj, tmp)
    os.replace(tmp, path)


def move_inputs(inputs, device):
    return {
        k: (
            v.to(device)
            if torch.is_tensor(v)
            else v
        )
        for k, v in inputs.items()
    }


# Batch construction

def render_chat(processor, conversation):
    rendered = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Qwen2.5-Omni ProcessorMixin may return [str] even for one
    # conversation. Normalize that to exactly one string.
    if isinstance(rendered, str):
        return rendered

    if (
        isinstance(rendered, list)
        and len(rendered) == 1
        and isinstance(rendered[0], str)
    ):
        return rendered[0]

    raise RuntimeError(
        "Unexpected apply_chat_template output: "
        f"type={type(rendered).__name__}, "
        f"value_type="
        f"{type(rendered[0]).__name__ if isinstance(rendered, list) and rendered else None}"
    )


def prepare_text_batch(processor, rows):
    conversations = [
        [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": row["prompt_text"],
            }],
        }]
        for row in rows
    ]

    texts = [
        render_chat(processor, conv)
        for conv in conversations
    ]

    if (
        len(texts) != len(rows)
        or not all(isinstance(x, str) for x in texts)
    ):
        raise RuntimeError(
            "Bad text batch after chat-template rendering"
        )

    return processor(
        text=texts,
        return_tensors="pt",
        padding=True,
    )


def prepare_audio_batch(processor, rows, manifest):
    conversations = []
    audios = []
    audio_paths = []

    for row in rows:
        eid = row_identity(row)

        if eid not in manifest:
            raise RuntimeError(
                f"No benchmark.csv row for {eid}"
            )

        mrow = manifest[eid]

        # Strong identity checks.
        for field, expected in (
            ("language", row["language"]),
            ("condition", row["condition"]),
        ):
            if mrow[field] != expected:
                raise RuntimeError(
                    f"{eid}: manifest {field}={mrow[field]!r}, "
                    f"row={expected!r}"
                )

        audio_path = ROOT / mrow["audio_path"]

        if not audio_path.is_file():
            raise RuntimeError(
                f"Missing audio: {audio_path}"
            )

        conv = [{
            "role": "user",
            "content": [{
                "type": "audio",
                "audio": "file://" + str(audio_path.resolve()),
            }],
        }]

        audio_data, images, videos = process_mm_info(
            conv,
            use_audio_in_video=False,
        )

        if audio_data is None or len(audio_data) != 1:
            raise RuntimeError(
                f"{eid}: expected exactly one audio"
            )

        conversations.append(conv)
        audios.extend(audio_data)
        audio_paths.append(mrow["audio_path"])

    texts = [
        render_chat(processor, conv)
        for conv in conversations
    ]

    if (
        len(texts) != len(rows)
        or not all(isinstance(x, str) for x in texts)
    ):
        raise RuntimeError(
            "Bad audio batch after chat-template rendering"
        )

    inputs = processor(
        text=texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    return inputs, audio_paths


# Capture

def first_tensor(obj):
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (tuple, list)):
        for value in obj:
            found = first_tensor(value)
            if found is not None:
                return found

    if isinstance(obj, dict):
        for value in obj.values():
            found = first_tensor(value)
            if found is not None:
                return found

    return None


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--num-shards",
        type=int,
        required=True,
    )
    p.add_argument(
        "--shard-index",
        type=int,
        required=True,
    )
    p.add_argument(
        "--text-batch-size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--audio-batch-size",
        type=int,
        default=8,
    )
    p.add_argument(
        "--device-map",
        default="auto",
    )
    p.add_argument(
        "--audio-only",
        action="store_true",
        help=(
            "Process audio rows only. Existing legacy mean-pooled "
            "audio files are replaced with 8-bin representations."
        ),
    )

    args = p.parse_args()

    if not (
        0 <= args.shard_index < args.num_shards
    ):
        raise RuntimeError("Invalid shard index")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; refusing CPU execution"
        )

    rows = read_jsonl(FOLD_FILE)

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, got {len(rows)}"
        )

    if len({r["task_id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate task_id")

    manifest = load_manifest(MANIFEST)

    # Striding gives each of the four GPUs a balanced subset.
    selected = rows[
        args.shard_index :: args.num_shards
    ]

    if len(selected) != EXPECTED_ROWS // args.num_shards:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS // args.num_shards} "
            f"rows on shard, got {len(selected)}"
        )

    def representation_is_current(row):
        path = OUT_ROOT / f"{row['task_id']}.pt"

        if not path.exists():
            return False

        if row["modality"] == "text":
            # Existing text representations remain valid.
            return True

        if row["modality"] != "audio":
            raise RuntimeError(
                f"Unknown modality: {row['modality']}"
            )

        try:
            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception:
            return False

        enc = obj.get("audio_encoder")

        return (
            torch.is_tensor(enc)
            and tuple(enc.shape)
                == (33, AUDIO_BINS, AUDIO_DIM)
            and obj.get("audio_encoder_pooling")
                == "mean_8_equal_temporal_bins_aftercnn_sequence"
        )

    if args.audio_only:
        selected_for_mode = [
            r for r in selected
            if r["modality"] == "audio"
        ]
    else:
        selected_for_mode = selected

    pending = [
        r for r in selected_for_mode
        if not representation_is_current(r)
    ]

    text_rows = [
        r for r in pending
        if r["modality"] == "text"
    ]

    audio_rows = [
        r for r in pending
        if r["modality"] == "audio"
    ]

    # Reduce padding waste.
    text_rows.sort(
        key=lambda r: len(r["prompt_text"])
    )

    def audio_duration(row):
        eid = row_identity(row)
        if eid not in manifest:
            raise RuntimeError(
                f"No manifest row for {eid}"
            )
        return float(
            manifest[eid]["duration_seconds"]
        )

    audio_rows.sort(key=audio_duration)

    print("=" * 80)
    print("BATCHED QWEN2.5-OMNI REPRESENTATION EXTRACTION")
    print("=" * 80)
    print("shard       :", args.shard_index)
    print("assigned shard rows :", len(selected))
    print("selected for mode   :", len(selected_for_mode))
    print(
        "already current     :",
        len(selected_for_mode) - len(pending),
    )
    print("pending text        :", len(text_rows))
    print("pending audio       :", len(audio_rows))
    print("audio only          :", args.audio_only)
    print("text batch  :", args.text_batch_size)
    print("audio batch :", args.audio_batch_size)
    print("=" * 80, flush=True)

    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # Left padding is the safe Qwen batched convention.
    processor.tokenizer.padding_side = "left"

    model = (
        Qwen2_5OmniForConditionalGeneration
        .from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map=args.device_map,
            local_files_only=True,
        )
    )

    if hasattr(model, "disable_talker"):
        model.disable_talker()

    model.eval()

    param_devices = {
        str(p.device)
        for p in model.parameters()
    }

    bad_devices = {
        d for d in param_devices
        if not d.startswith("cuda:")
    }

    if bad_devices:
        raise RuntimeError(
            "CPU/disk model placement detected. "
            f"Devices={sorted(param_devices)}"
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print(
        "model devices:",
        sorted(param_devices),
        flush=True,
    )

    thinker = model.thinker
    thinker_text = thinker.model
    audio_tower = thinker.audio_tower

    if len(thinker_text.layers) != N_THINKER_LAYERS:
        raise RuntimeError(
            f"Expected {N_THINKER_LAYERS} Thinker layers"
        )

    if len(audio_tower.layers) != N_AUDIO_LAYERS:
        raise RuntimeError(
            f"Expected {N_AUDIO_LAYERS} audio layers"
        )

    device = next(thinker.parameters()).device

    # Per-current-batch metadata used by hooks.
    state = {
        "last_positions": None,
        "audio_aftercnn_lens": None,
        "thinker": {},
        "encoder": {},
    }

    handles = []

    def capture_thinker_tensor(x, where):
        x = first_tensor(x)

        if x is None:
            raise RuntimeError(
                f"No tensor at {where}"
            )

        if x.ndim != 3:
            raise RuntimeError(
                f"{where}: expected [B,L,H], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != THINKER_DIM:
            raise RuntimeError(
                f"{where}: hidden dim {x.shape[-1]}"
            )

        pos = state["last_positions"]

        if pos is None:
            raise RuntimeError(
                "last_positions not initialized"
            )

        if x.shape[0] != pos.numel():
            raise RuntimeError(
                f"{where}: batch mismatch "
                f"{x.shape[0]} vs {pos.numel()}"
            )

        idx = torch.arange(
            x.shape[0],
            device=x.device,
        )

        # [B, H], still on GPU.
        return (
            x[idx, pos.to(x.device), :]
            .detach()
            .clone()
        )

    def capture_encoder_tensor(x, where):
        x = first_tensor(x)

        if x is None:
            raise RuntimeError(
                f"No tensor at {where}"
            )

        if x.ndim != 2:
            raise RuntimeError(
                f"{where}: expected packed [T,H], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != AUDIO_DIM:
            raise RuntimeError(
                f"{where}: hidden dim {x.shape[-1]}"
            )

        lengths = state["audio_aftercnn_lens"]

        if lengths is None:
            raise RuntimeError(
                "audio_aftercnn_lens not initialized"
            )

        lengths_cpu = [
            int(v)
            for v in lengths.detach().cpu().tolist()
        ]

        if sum(lengths_cpu) != x.shape[0]:
            raise RuntimeError(
                f"{where}: packed length={x.shape[0]}, "
                f"expected={sum(lengths_cpu)} "
                f"from {lengths_cpu}"
            )

        pieces = torch.split(
            x,
            lengths_cpu,
            dim=0,
        )

        # Preserve coarse temporal structure:
        # split each real audio sequence into 8 contiguous,
        # approximately equal temporal bins, mean-pooling only
        # within each bin.
        binned_examples = []

        for piece in pieces:
            T = piece.shape[0]

            if T < AUDIO_BINS:
                raise RuntimeError(
                    f"{where}: audio sequence has only {T} "
                    f"positions, fewer than {AUDIO_BINS} bins"
                )

            bins = []

            for b in range(AUDIO_BINS):
                start = (b * T) // AUDIO_BINS
                end = ((b + 1) * T) // AUDIO_BINS

                if end <= start:
                    raise RuntimeError(
                        f"{where}: empty temporal bin "
                        f"{b} for sequence length {T}"
                    )

                bins.append(
                    piece[start:end]
                    .float()
                    .mean(dim=0)
                )

            binned_examples.append(
                torch.stack(bins, dim=0)
            )

        # [B, 8, 1280], still on GPU.
        return torch.stack(
            binned_examples,
            dim=0,
        ).detach()

    # Thinker boundary 0.
    def thinker_pre(module, args):
        state["thinker"][0] = (
            capture_thinker_tensor(
                args[0],
                "Thinker boundary 0",
            )
        )

    handles.append(
        thinker_text.layers[0]
        .register_forward_pre_hook(thinker_pre)
    )

    # Thinker boundaries 1..28.
    for i, layer in enumerate(
        thinker_text.layers,
        start=1,
    ):
        def make_hook(boundary):
            def hook(module, args, output):
                state["thinker"][boundary] = (
                    capture_thinker_tensor(
                        output,
                        f"Thinker boundary {boundary}",
                    )
                )
            return hook

        handles.append(
            layer.register_forward_hook(
                make_hook(i)
            )
        )

    # Encoder boundary 0.
    def encoder_pre(module, args):
        state["encoder"][0] = (
            capture_encoder_tensor(
                args[0],
                "audio encoder boundary 0",
            )
        )

    handles.append(
        audio_tower.layers[0]
        .register_forward_pre_hook(encoder_pre)
    )

    # Encoder boundaries 1..32.
    for i, layer in enumerate(
        audio_tower.layers,
        start=1,
    ):
        def make_hook(boundary):
            def hook(module, args, output):
                state["encoder"][boundary] = (
                    capture_encoder_tensor(
                        output,
                        f"audio encoder boundary {boundary}",
                    )
                )
            return hook

        handles.append(
            layer.register_forward_hook(
                make_hook(i)
            )
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0

    def run_batch(batch_rows, modality):
        nonlocal saved

        state["thinker"].clear()
        state["encoder"].clear()
        state["audio_aftercnn_lens"] = None

        if modality == "text":
            inputs = prepare_text_batch(
                processor,
                batch_rows,
            )
            audio_paths = [None] * len(batch_rows)

        elif modality == "audio":
            inputs, audio_paths = prepare_audio_batch(
                processor,
                batch_rows,
                manifest,
            )

        else:
            raise RuntimeError(modality)

        if "attention_mask" not in inputs:
            raise RuntimeError(
                "Processor returned no attention_mask"
            )

        mask = inputs["attention_mask"]

        positions = torch.arange(
            mask.shape[1],
        ).unsqueeze(0).expand_as(mask)

        # Works for left or right padding:
        # largest index whose attention mask is 1.
        positions = positions.masked_fill(
            mask == 0,
            -1,
        )

        last_positions = positions.max(
            dim=1
        ).values

        if (last_positions < 0).any():
            raise RuntimeError(
                "Empty prompt in batch"
            )

        state["last_positions"] = (
            last_positions.to(device)
        )

        if modality == "audio":
            if "feature_attention_mask" not in inputs:
                raise RuntimeError(
                    "Audio processor returned no "
                    "feature_attention_mask"
                )

            feature_lens = (
                inputs["feature_attention_mask"]
                .sum(dim=1)
                .long()
                .to(device)
            )

            # Exact lengths used by Qwen's own audio encoder
            # to split the packed layer representation back
            # into individual audio examples.
            aftercnn_lens, _ = (
                audio_tower
                ._get_feat_extract_output_lengths(
                    feature_lens
                )
            )

            state["audio_aftercnn_lens"] = (
                aftercnn_lens
            )

        inputs = move_inputs(
            inputs,
            device,
        )

        # DIRECT PREFILL ONLY.
        # No autoregressive decoding and no generate().
        with torch.inference_mode():
            thinker(
                **inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_audio_in_video=False,
            )

        expected_thinker = set(range(29))

        if set(state["thinker"]) != expected_thinker:
            raise RuntimeError(
                "Incomplete Thinker capture: "
                f"{sorted(state['thinker'])}"
            )

        # [29, B, 3584] -> one GPU->CPU transfer.
        thinker_stack = (
            torch.stack(
                [
                    state["thinker"][i]
                    for i in range(29)
                ],
                dim=0,
            )
            .to(
                "cpu",
                dtype=torch.float16,
            )
            .contiguous()
        )

        if thinker_stack.shape != (
            29,
            len(batch_rows),
            THINKER_DIM,
        ):
            raise RuntimeError(
                f"Bad Thinker shape: "
                f"{tuple(thinker_stack.shape)}"
            )

        encoder_stack = None

        if modality == "audio":
            expected_encoder = set(range(33))

            if set(state["encoder"]) != expected_encoder:
                raise RuntimeError(
                    "Incomplete encoder capture: "
                    f"{sorted(state['encoder'])}"
                )

            # [33, B, 8, 1280] -> one GPU->CPU transfer.
            encoder_stack = (
                torch.stack(
                    [
                        state["encoder"][i]
                        for i in range(33)
                    ],
                    dim=0,
                )
                .to(
                    "cpu",
                    dtype=torch.float16,
                )
                .contiguous()
            )

            if encoder_stack.shape != (
                33,
                len(batch_rows),
                AUDIO_BINS,
                AUDIO_DIM,
            ):
                raise RuntimeError(
                    f"Bad encoder shape: "
                    f"{tuple(encoder_stack.shape)}"
                )

        # Save one independent file per original row.
        for j, row in enumerate(batch_rows):
            task_id = row["task_id"]
            eid = row_identity(row)

            record = {
                "task_id": task_id,
                "example_id": eid,
                "base_prompt_id": row["base_prompt_id"],
                "safety_split": row["safety_split"],
                "language": row["language"],
                "condition": row["condition"],
                "modality": modality,
                "prompt_text": row["prompt_text"],
                "decision": int(row["decision"]),
                "model_path": str(MODEL_PATH),
                "extraction": "full_model_direct_prefill",
                "thinker_pooling":
                    "last_real_prompt_position",
                "thinker_boundaries":
                    list(range(29)),
                "thinker":
                    thinker_stack[:, j, :].clone(),
            }

            if modality == "audio":
                record.update({
                    "audio_path":
                        audio_paths[j],
                    "audio_encoder_pooling":
                        "mean_8_equal_temporal_bins_aftercnn_sequence",
                    "audio_encoder_bins":
                        AUDIO_BINS,
                    "audio_encoder_boundaries":
                        list(range(33)),
                    "audio_encoder":
                        encoder_stack[:, j, :, :].clone(),
                })

            atomic_save(
                record,
                OUT_ROOT / f"{task_id}.pt",
            )

            saved += 1

        print(
            f"[{modality.upper():5s}] "
            f"batch={len(batch_rows):2d} | "
            f"saved_total={saved} | "
            f"last={batch_rows[-1]['task_id']}",
            flush=True,
        )

        del inputs
        del thinker_stack
        if encoder_stack is not None:
            del encoder_stack

    def is_cuda_oom(exc):
        return (
            isinstance(exc, torch.OutOfMemoryError)
            or (
                isinstance(exc, RuntimeError)
                and "out of memory" in str(exc).lower()
            )
        )

    def clear_after_oom():
        state["thinker"].clear()
        state["encoder"].clear()
        state["audio_aftercnn_lens"] = None
        state["last_positions"] = None

        gc.collect()
        torch.cuda.empty_cache()

    def run_audio_with_backoff(batch_rows):
        try:
            run_batch(
                batch_rows,
                "audio",
            )
            return

        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise

            attempted = len(batch_rows)

            print(
                f"[OOM] audio batch={attempted} | "
                f"first={batch_rows[0]['task_id']} | "
                f"last={batch_rows[-1]['task_id']}",
                flush=True,
            )

            clear_after_oom()

            if attempted == 1:
                raise RuntimeError(
                    "CUDA OOM even at audio batch size 1: "
                    f"{batch_rows[0]['task_id']}"
                ) from exc

            mid = attempted // 2

            left = batch_rows[:mid]
            right = batch_rows[mid:]

            print(
                f"      retrying as "
                f"{len(left)} + {len(right)}",
                flush=True,
            )

            run_audio_with_backoff(left)
            run_audio_with_backoff(right)

    try:
        if not args.audio_only:
            bs = args.text_batch_size

            for start in range(
                0,
                len(text_rows),
                bs,
            ):
                run_batch(
                    text_rows[start:start + bs],
                    "text",
                )

        # Every NEW outer batch starts at the requested batch size.
        # An OOM only reduces the offending batch recursively.
        bs = args.audio_batch_size

        for start in range(
            0,
            len(audio_rows),
            bs,
        ):
            run_audio_with_backoff(
                audio_rows[start:start + bs]
            )

    finally:
        for handle in handles:
            handle.remove()

    print()
    print("=" * 80)
    print("SHARD COMPLETE")
    print("=" * 80)
    print("selected :", len(selected_for_mode))
    print("saved now:", saved)
    print(
        "previously current:",
        len(selected_for_mode) - len(pending),
    )
    print("output   :", OUT_ROOT)


if __name__ == "__main__":
    main()
