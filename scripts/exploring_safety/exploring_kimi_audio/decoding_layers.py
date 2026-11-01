#!/usr/bin/env python3

import argparse
import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

RESULTS = ROOT / "results/INITIAL_BENCHMARK"
MANIFEST = (
    ROOT
    / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"
)
MODEL_PATH = ROOT / "models/kimi-audio-7b-instruct"

MODEL_NAME = "kimi_audio"

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


# =============================================================================
# JSONL / DATA HELPERS
# =============================================================================

def read_jsonl(path):
    path = Path(path)

    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def strip_modality(task_id):
    for suffix in ("__text", "__audio"):
        if task_id.endswith(suffix):
            return task_id[:-len(suffix)]

    raise RuntimeError(
        f"Bad task_id: {task_id}"
    )


def task_folder(row):
    return strip_modality(
        row["task_id"]
    )


def load_manifest():
    with MANIFEST.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    return {
        row["example_id"]: row
        for row in rows
    }


def manifest_for(row, manifest):
    key = strip_modality(
        row["task_id"]
    )

    if key not in manifest:
        raise RuntimeError(
            f"No manifest row for {key}"
        )

    return manifest[key]


def local_rows(
    rows,
    num_shards,
    shard_index,
):
    n = len(rows)

    base, rem = divmod(
        n,
        num_shards,
    )

    start = (
        shard_index * base
        + min(shard_index, rem)
    )

    count = (
        base
        + (
            1
            if shard_index < rem
            else 0
        )
    )

    return rows[
        start:start + count
    ]


def completed_layers(path):
    path = Path(path)

    if not path.exists():
        return set()

    seen = set()

    for row in read_jsonl(path):
        layer = int(
            row["layer"]
        )

        if layer in seen:
            raise RuntimeError(
                f"Duplicate layer {layer} "
                f"in {path}"
            )

        seen.add(layer)

    return seen


# =============================================================================
# MODEL HELPERS
# =============================================================================

def dotted(root, path):
    obj = root

    for name in path.split("."):
        obj = getattr(
            obj,
            name,
        )

    return obj


def find_whisper_layers(kimi):
    candidates = [
        "prompt_manager.whisper_model.speech_encoder",
        "prompt_manager.whisper_model.encoder",
        "prompt_manager.whisper_model.model.encoder",
    ]

    for path in candidates:
        try:
            module = dotted(
                kimi,
                path,
            )
        except AttributeError:
            continue

        if (
            hasattr(module, "layers")
            and isinstance(
                module.layers,
                torch.nn.ModuleList,
            )
            and len(module.layers) == 32
        ):
            print(
                f"[resolve] "
                f"{path}.layers "
                f"-> 32 layers"
            )

            return (
                module,
                module.layers,
            )

    root = (
        kimi
        .prompt_manager
        .whisper_model
    )

    if hasattr(
        root,
        "named_modules",
    ):
        for name, module in (
            root.named_modules()
        ):
            if (
                hasattr(
                    module,
                    "layers",
                )
                and isinstance(
                    module.layers,
                    torch.nn.ModuleList,
                )
                and len(
                    module.layers
                ) == 32
            ):
                print(
                    "[resolve] "
                    f"whisper named module "
                    f"{name}.layers -> 32"
                )

                return (
                    module,
                    module.layers,
                )

    raise RuntimeError(
        "Could not locate "
        "Kimi 32-layer Whisper encoder"
    )


# =============================================================================
# KIMI ADAPTER
# =============================================================================

class KimiAdapter:

    model_name = MODEL_NAME

    def __init__(self):
        from kimia_infer.api.kimia import (
            KimiAudio,
        )

        self.kimi = KimiAudio(
            model_path=str(
                MODEL_PATH
            ),
            load_detokenizer=False,
        )

        alm_model = (
            self.kimi
            .alm
            .model
        )

        # ---------------------------------------------------------
        # Shared decoder
        # ---------------------------------------------------------

        self.full_shared = (
            alm_model.layers
        )

        if len(
            self.full_shared
        ) != 28:
            raise RuntimeError(
                "Expected 28 Kimi "
                "shared decoder layers, "
                f"found "
                f"{len(self.full_shared)}"
            )

        # ---------------------------------------------------------
        # MIMO decoder
        # ---------------------------------------------------------

        self.full_mimo = (
            alm_model.mimo_layers
        )

        if len(
            self.full_mimo
        ) != 6:
            raise RuntimeError(
                "Expected 6 Kimi "
                "MIMO layers, found "
                f"{len(self.full_mimo)}"
            )

        # ---------------------------------------------------------
        # Whisper encoder
        # ---------------------------------------------------------

        (
            self.whisper_module,
            self.full_whisper,
        ) = find_whisper_layers(
            self.kimi
        )

        # ---------------------------------------------------------
        # Original fork
        # ---------------------------------------------------------

        self.original_fork = int(
            alm_model
            .kimia_mimo_transformer_from_layer_index
        )

        print(
            "[architecture] "
            f"shared_decoder="
            f"{len(self.full_shared)} "
            f"mimo="
            f"{len(self.full_mimo)} "
            f"whisper="
            f"{len(self.full_whisper)} "
            f"original_fork="
            f"{self.original_fork}"
        )

        # ---------------------------------------------------------
        # Layer-decoding stages
        #
        # 29 = input boundary + 28 shared blocks
        #  7 = MIMO input boundary + 6 MIMO blocks
        # 33 = Whisper input boundary + 32 blocks
        # ---------------------------------------------------------

        self.stages = (
            [
                (
                    "text",
                    "shared_decoder",
                    i,
                )
                for i in range(29)
            ]
            + [
                (
                    "text",
                    "mimo",
                    i,
                )
                for i in range(7)
            ]
            + [
                (
                    "audio",
                    "encoder",
                    i,
                )
                for i in range(33)
            ]
            + [
                (
                    "audio",
                    "shared_decoder",
                    i,
                )
                for i in range(29)
            ]
            + [
                (
                    "audio",
                    "mimo",
                    i,
                )
                for i in range(7)
            ]
        )

        # ---------------------------------------------------------
        # Text-baseline silence
        # ---------------------------------------------------------

        self.silence = Path(
            "/tmp/"
            "kimi_decoding_layers_"
            "silence_1s.wav"
        )

        if not self.silence.exists():
            sf.write(
                self.silence,
                np.zeros(
                    16000,
                    dtype=np.float32,
                ),
                16000,
            )


    # =========================================================================
    # OUTPUT FILES
    # =========================================================================

    def output_name(
        self,
        modality,
        family,
    ):
        return {
            (
                "text",
                "shared_decoder",
            ):
                "textshareddecoder_responses.jsonl",

            (
                "text",
                "mimo",
            ):
                "textmimo_responses.jsonl",

            (
                "audio",
                "encoder",
            ):
                "audioencoder_responses.jsonl",

            (
                "audio",
                "shared_decoder",
            ):
                "audioshareddecoder_responses.jsonl",

            (
                "audio",
                "mimo",
            ):
                "audiomimo_responses.jsonl",

        }[
            (
                modality,
                family,
            )
        ]


    # =========================================================================
    # INPUT PREPARATION
    # =========================================================================

    def prepare(
        self,
        row,
        mrow,
        modality,
    ):
        if modality == "text":

            # Match existing Kimi text baseline:
            # real text + identical 1 s silence.

            return [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": row[
                        "prompt_text"
                    ],
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": str(
                        self.silence
                    ),
                },
            ]

        audio_path = (
            ROOT
            / mrow["audio_path"]
        ).resolve()

        return [
            {
                "role": "user",
                "message_type": "audio",
                "content": str(
                    audio_path
                ),
            }
        ]


    # =========================================================================
    # SHARED-DECODER BOUNDARY 0
    # =========================================================================

    @staticmethod
    def _boundary_zero_hook(
        module,
        inputs,
        output,
    ):
        """
        Preserve layer-0 cache / control flow,
        but make its hidden-state output equal
        its input hidden state.

        This gives the pre-shared-transformer
        boundary while still allowing Kimi's
        autoregressive cache machinery to run.
        """

        hidden_in = inputs[0]

        if isinstance(
            output,
            tuple,
        ):
            return (
                hidden_in,
            ) + tuple(
                output[1:]
            )

        return hidden_in


    # =========================================================================
    # STAGE INTERVENTION
    # =========================================================================

    @contextmanager
    def stage(
        self,
        modality,
        family,
        layer,
    ):
        alm_model = (
            self.kimi
            .alm
            .model
        )

        hook = None

        # Always start from complete architecture.
        alm_model.layers = (
            self.full_shared
        )

        alm_model.mimo_layers = (
            self.full_mimo
        )

        self.whisper_module.layers = (
            self.full_whisper
        )

        (
            alm_model
            .kimia_mimo_transformer_from_layer_index
        ) = self.original_fork

        try:

            # =====================================================
            # SHARED DECODER
            #
            # To decode shared boundary k:
            #
            # k > 0:
            #   retain first k shared blocks,
            #   move MIMO fork to block k-1.
            #
            # Therefore the MIMO text branch receives
            # exactly the shared representation at
            # boundary k.
            #
            # k == 0:
            #   keep one physical block only so Kimi
            #   still builds a KV cache and creates
            #   mimo_hidden_states, but hook its output
            #   back to its input.
            # =====================================================

            if family == "shared_decoder":

                if not (
                    0 <= layer <= 28
                ):
                    raise RuntimeError(
                        "Bad shared decoder "
                        f"boundary {layer}"
                    )

                if layer == 0:

                    alm_model.layers = (
                        torch.nn.ModuleList(
                            [
                                self.full_shared[0]
                            ]
                        )
                    )

                    (
                        alm_model
                        .kimia_mimo_transformer_from_layer_index
                    ) = 0

                    hook = (
                        self.full_shared[0]
                        .register_forward_hook(
                            self._boundary_zero_hook
                        )
                    )

                else:

                    alm_model.layers = (
                        torch.nn.ModuleList(
                            list(
                                self.full_shared
                            )[:layer]
                        )
                    )

                    # Fork from the last retained
                    # shared transformer block.
                    (
                        alm_model
                        .kimia_mimo_transformer_from_layer_index
                    ) = layer - 1


            # =====================================================
            # MIMO
            # =====================================================

            elif family == "mimo":

                if not (
                    0 <= layer <= 6
                ):
                    raise RuntimeError(
                        f"Bad MIMO boundary {layer}"
                    )

                alm_model.mimo_layers = (
                    torch.nn.ModuleList(
                        list(
                            self.full_mimo
                        )[:layer]
                    )
                )


            # =====================================================
            # WHISPER ENCODER
            # =====================================================

            elif family == "encoder":

                if not (
                    0 <= layer <= 32
                ):
                    raise RuntimeError(
                        f"Bad encoder boundary {layer}"
                    )

                self.whisper_module.layers = (
                    torch.nn.ModuleList(
                        list(
                            self.full_whisper
                        )[:layer]
                    )
                )

            else:
                raise RuntimeError(
                    f"Bad Kimi family {family}"
                )

            yield

        finally:

            if hook is not None:
                hook.remove()

            alm_model.layers = (
                self.full_shared
            )

            alm_model.mimo_layers = (
                self.full_mimo
            )

            self.whisper_module.layers = (
                self.full_whisper
            )

            (
                alm_model
                .kimia_mimo_transformer_from_layer_index
            ) = self.original_fork


    # =========================================================================
    # GENERATION
    # =========================================================================

    def generate(
        self,
        messages,
        modality,
        family,
        layer,
        max_new_tokens,
    ):
        with self.stage(
            modality,
            family,
            layer,
        ):
            with torch.inference_mode():

                _, text = (
                    self.kimi.generate(
                        messages,
                        output_type="text",

                        audio_temperature=0.0,
                        audio_top_k=5,

                        text_temperature=0.0,
                        text_top_k=5,

                        audio_repetition_penalty=1.0,
                        audio_repetition_window_size=64,

                        text_repetition_penalty=1.0,
                        text_repetition_window_size=16,

                        max_new_tokens=max_new_tokens,
                    )
                )

        return str(text).strip()


# =============================================================================
# SELF-CONTAINED BACKEND
# =============================================================================

def run_backend(
    adapter,
    num_shards,
    shard_index,
    max_new_tokens,
    groups,
    start,
    limit,
):
    if (
        num_shards < 1
        or not (
            0
            <= shard_index
            < num_shards
        )
    ):
        raise RuntimeError(
            "Invalid shard configuration"
        )

    manifest = load_manifest()

    print("=" * 80)
    print("KIMI DECODING LAYERS")
    print("=" * 80)
    print(
        "model:",
        adapter.model_name,
    )
    print(
        "shard:",
        f"{shard_index}/{num_shards}",
    )
    print(
        "max_new_tokens:",
        max_new_tokens,
    )
    print(
        "stages:",
        len(adapter.stages),
    )

    for group in groups:

        source = (
            RESULTS
            / MODEL_NAME
            / "decoding_layers"
            / "selections"
            / f"{group}.jsonl"
        )

        if not source.exists():
            raise RuntimeError(
                f"Missing selection: {source}"
            )

        all_rows = read_jsonl(
            source
        )

        if limit is None:
            selected_rows = (
                all_rows[start:]
            )
        else:
            selected_rows = (
                all_rows[
                    start:
                    start + limit
                ]
            )

        rows = local_rows(
            selected_rows,
            num_shards,
            shard_index,
        )

        print(
            "\n"
            f"===== {group} | "
            f"total={len(all_rows)} "
            f"selected="
            f"{len(selected_rows)} "
            f"range="
            f"{start}:"
            f"{'end' if limit is None else start + limit} "
            f"local={len(rows)} "
            f"=====",
            flush=True,
        )

        for i, row in enumerate(
            rows,
            1,
        ):

            mrow = manifest_for(
                row,
                manifest,
            )

            outdir = (
                RESULTS
                / MODEL_NAME
                / "decoding_layers"
                / "examples"
                / group
                / task_folder(row)
            )

            outdir.mkdir(
                parents=True,
                exist_ok=True,
            )

            prepared = {
                "text": adapter.prepare(
                    row,
                    mrow,
                    "text",
                ),
                "audio": adapter.prepare(
                    row,
                    mrow,
                    "audio",
                ),
            }

            print(
                f"[{i}/{len(rows)}] "
                f"{task_folder(row)}",
                flush=True,
            )

            for (
                modality,
                family,
                layer,
            ) in adapter.stages:

                outfile = (
                    outdir
                    / adapter.output_name(
                        modality,
                        family,
                    )
                )

                done = completed_layers(
                    outfile
                )

                if layer in done:

                    print(
                        "  [skip] "
                        f"{modality}/"
                        f"{family}/"
                        f"{layer:02d}",
                        flush=True,
                    )

                    continue

                print(
                    "  [run ] "
                    f"{modality}/"
                    f"{family}/"
                    f"{layer:02d}",
                    flush=True,
                )

                response = (
                    adapter.generate(
                        prepared[
                            modality
                        ],
                        modality,
                        family,
                        layer,
                        max_new_tokens,
                    )
                )

                rec = {
                    k: v
                    for k, v
                    in row.items()
                    if k not in {
                        "response_text",
                        "decision_text",
                        "response_audio",
                        "decision_audio",
                    }
                }

                rec[
                    "decoding_modality"
                ] = modality

                rec[
                    "decoding_family"
                ] = family

                rec[
                    "layer"
                ] = layer

                rec[
                    "response"
                ] = response

                append_jsonl(
                    outfile,
                    rec,
                )

                print(
                    "  [done] "
                    f"chars="
                    f"{len(response)}",
                    flush=True,
                )

    print(
        "\n[OK] "
        "Kimi decoding shard complete"
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
    )

    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    p.add_argument(
        "--groups",
        nargs="*",
        default=None,
    )

    args = p.parse_args()

    adapter = KimiAdapter()

    run_backend(
        adapter=adapter,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_new_tokens=args.max_new_tokens,
        groups=args.groups or GROUPS,
        start=args.start,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
