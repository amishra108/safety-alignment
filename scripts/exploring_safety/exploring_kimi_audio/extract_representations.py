#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import torch

from representation_extract_common import (
    ROOT,
    VERSION,
    BoundaryCapture,
    atomic_save,
    load_rows,
    representation_current,
    resolve_attr,
    select_shard,
)

MODEL = "kimi_audio"
MODEL_PATH = ROOT / "models/kimi-audio-7b-instruct"
OUT_ROOT = (
    ROOT
    / "results/INITIAL_BENCHMARK"
    / MODEL
    / "5fold"
    / "representations_shared"
)


def silence():
    p = (
        Path(
            os.environ.get(
                "TMPDIR",
                tempfile.gettempdir(),
            )
        )
        / "kimi_rep_silence_1s.wav"
    )

    if not p.exists():
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)

    return p


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--modality",
        choices=("all", "text", "audio"),
        default="all",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preflight", action="store_true")

    args = ap.parse_args()

    rows = load_rows(MODEL)

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

    from kimia_infer.api.kimia import KimiAudio

    kimi = KimiAudio(
        model_path=str(MODEL_PATH),
        load_detokenizer=False,
    )

    shared, shared_path = resolve_attr(
        kimi,
        ("alm.model.layers",),
        "Kimi shared ALM decoder",
    )

    mimo, mimo_path = resolve_attr(
        kimi,
        ("alm.model.mimo_layers",),
        "Kimi MIMO branch",
    )

    whisper, whisper_path = resolve_attr(
        kimi,
        (
            "prompt_manager.whisper_model."
            "speech_encoder.layers",
        ),
        "Kimi Whisper encoder",
    )

    print(
        "[architecture] "
        f"hidden={kimi.alm.config.hidden_size} "
        f"shared_layers={len(shared)} "
        f"mimo_layers={len(mimo)} "
        f"mimo_fork="
        f"{kimi.alm.config.kimia_mimo_transformer_from_layer_index} "
        f"whisper_layers={len(whisper)}",
        flush=True,
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.preflight:
        print("PREFLIGHT OK")
        return

    # Kimi's released API is single-example.
    # We do NOT invent fake batching.
    shared_cap = BoundaryCapture(
        shared,
        "shared_decoder",
        "last",
    )

    mimo_cap = BoundaryCapture(
        mimo,
        "text_mimo",
        "last",
    )

    # The Kimi wrapper itself trims continuous Whisper output to the
    # true token length, but hooks inside the raw Whisper stack see
    # its padded 30-second sequence. Therefore we do NOT incorrectly
    # pretend the raw layer sequence has a trustworthy per-example
    # temporal mask. Capture full-stage mean-8 bins only for real audio.
    whisper_cap = BoundaryCapture(
        whisper,
        "audio_encoder",
        "pool8",
    )

    sil = silence()

    sampling = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
        "max_new_tokens": 1,
    }

    saved = 0
    skipped = 0

    try:
        for i, row in enumerate(rows, 1):
            out = (
                OUT_ROOT
                / f"{row['task_id']}.pt"
            )

            if (
                not args.overwrite
                and representation_current(
                    out,
                    MODEL,
                    row["task_id"],
                    row["modality"],
                )
            ):
                skipped += 1
                continue

            shared_cap.reset(1)
            mimo_cap.reset(1)

            # Raw Whisper encoder is always 30-second padded internally.
            # For now use its actual observed full sequence length.
            # This is stored explicitly in metadata so we do not confuse
            # it with Qwen/AF3 true-length binning.
            # Kimi's text control also contains 1 s silence, so the
            # Whisper stack executes for BOTH text and audio examples.
            # Internally this encoder uses its padded 30 s / 1500-step
            # post-conv sequence.
            whisper_cap.reset(
                1,
                [1500],
            )

            if row["modality"] == "text":
                messages = [
                    {
                        "role": "user",
                        "message_type": "text",
                        "content": row["prompt_text"],
                    },
                    {
                        "role": "user",
                        "message_type": "audio",
                        "content": str(sil),
                    },
                ]
            else:
                messages = [{
                    "role": "user",
                    "message_type": "audio",
                    "content": str(
                        ROOT / row["audio_path"]
                    ),
                }]

            try:
                with torch.inference_mode():
                    kimi.generate(
                        messages,
                        output_type="text",
                        **sampling,
                    )

            except Exception as exc:
                if "out of memory" not in str(exc).lower():
                    raise

                gc.collect()
                torch.cuda.empty_cache()

                # Already at batch 1.
                print(
                    f"Kimi OOM at batch=1: "
                    f"{row['task_id']}",
                    flush=True,
                )
                raise

            payload = {
                "representation_version": VERSION,
                "model": MODEL,
                "task_id": row["task_id"],
                "base_prompt_id":
                    row.get("base_prompt_id"),
                "language":
                    row.get("language"),
                "condition":
                    row.get("condition"),
                "safety_split":
                    row["safety_split"],
                "modality":
                    row["modality"],
                "source_file_snapshot":
                    row["_source_file"],
                "shared_decoder":
                    shared_cap.stack()[:, 0],
                "text_mimo":
                    mimo_cap.stack()[:, 0],
                "shared_decoder_layers":
                    len(shared),
                "text_mimo_layers":
                    len(mimo),
                "shared_decoder_module":
                    shared_path,
                "text_mimo_module":
                    mimo_path,
                "mimo_fork_layer":
                    kimi.alm.config
                    .kimia_mimo_transformer_from_layer_index,
                # decision deliberately omitted
            }

            if row["modality"] == "audio":
                payload["audio_encoder"] = (
                    whisper_cap.stack()[:, 0]
                )
                payload["audio_encoder_layers"] = (
                    len(whisper)
                )
                payload["audio_encoder_module"] = (
                    whisper_path
                )
                payload["audio_encoder_pooling"] = (
                    "mean_8_bins_over_internal_"
                    "30s_padded_whisper_sequence"
                )

            atomic_save(payload, out)
            saved += 1

            print(
                f"[{i}/{len(rows)}] "
                f"{row['modality']} "
                f"{row['task_id']} "
                f"saved={saved}",
                flush=True,
            )

            gc.collect()
            torch.cuda.empty_cache()

    finally:
        shared_cap.close()
        mimo_cap.close()
        whisper_cap.close()

    print(
        f"DONE saved={saved} skipped={skipped} "
        f"output={OUT_ROOT}"
    )


if __name__ == "__main__":
    main()
