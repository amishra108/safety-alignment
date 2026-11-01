#!/usr/bin/env python3
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import torch

from representation_extract_common import (
    ROOT,
    BoundaryCapture,
    resolve_attr,
    run_generic,
)


class Adapter:
    model_name = "audio_flamingo3"

    def load(self):
        from transformers import (
            AudioFlamingo3ForConditionalGeneration,
            AutoProcessor,
        )

        path = ROOT / "models/audio-flamingo-3"

        processor = AutoProcessor.from_pretrained(
            str(path),
            local_files_only=True,
        )

        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"

        model = (
            AudioFlamingo3ForConditionalGeneration
            .from_pretrained(
                str(path),
                torch_dtype=torch.float16,
                device_map={"": 0},
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        )
        model.eval()

        decoder, decoder_path = resolve_attr(
            model,
            ("model.language_model.layers",),
            "AF3 Qwen2 decoder",
        )

        encoder, encoder_path = resolve_attr(
            model,
            ("model.audio_tower.layers",),
            "AF3 Whisper encoder",
        )

        print(
            "[architecture] "
            f"decoder_hidden={model.config.text_config.hidden_size} "
            f"audio_hidden={model.config.audio_config.hidden_size} "
            f"encoder_layers={len(encoder)} "
            f"decoder_layers={len(decoder)}",
            flush=True,
        )

        return {
            "model": model,
            "processor": processor,
            "decoder": decoder,
            "encoder": encoder,
            "decoder_path": decoder_path,
            "encoder_path": encoder_path,
        }

    def make_captures(self, ctx):
        return {
            "decoder": BoundaryCapture(
                ctx["decoder"],
                "decoder",
                "last",
            ),
            "audio_encoder": BoundaryCapture(
                ctx["encoder"],
                "audio_encoder",
                "pool8",
            ),
        }

    def prepare_batch(self, ctx, rows, modality):
        p = ctx["processor"]

        conversations = []

        for r in rows:
            if modality == "text":
                content = [{
                    "type": "text",
                    "text": r["prompt_text"],
                }]
            else:
                content = [{
                    "type": "audio",
                    "path": str(ROOT / r["audio_path"]),
                }]

            conversations.append([{
                "role": "user",
                "content": content,
            }])

        inputs = p.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "text_kwargs": {
                    "padding": True,
                },
                "audio_kwargs": {
                    "padding": "max_length",
                    "return_attention_mask": True,
                },
                "common_kwargs": {
                    "return_tensors": "pt",
                    "padding_side": "left",
                },
            },
        )

        lengths = None
        group_sizes = None

        if modality == "audio":
            import math
            import soundfile as sf

            mask = inputs["input_features_mask"]
            raw_lengths = mask.sum(-1).long()

            # Encoder lengths for every flattened 30-second window.
            lengths = (
                (raw_lengths - 1) // 2 + 1
            ).tolist()

            # AF3 processor chunks each logical audio sample into
            # feature_extractor.chunk_length-second windows and
            # flattens all windows into input_features.
            chunk_seconds = float(
                p.feature_extractor.chunk_length
            )

            max_windows = int(
                p.max_audio_len // chunk_seconds
            )

            group_sizes = []

            for r in rows:
                info = sf.info(
                    str(ROOT / r["audio_path"])
                )

                duration = (
                    float(info.frames)
                    / float(info.samplerate)
                )

                n_windows = max(
                    1,
                    math.ceil(duration / chunk_seconds),
                )

                n_windows = min(
                    n_windows,
                    max_windows,
                )

                group_sizes.append(n_windows)

            if sum(group_sizes) != len(lengths):
                raise RuntimeError(
                    "AF3 window reconstruction mismatch: "
                    f"logical_batch={len(rows)} "
                    f"group_sizes={group_sizes} "
                    f"sum_windows={sum(group_sizes)} "
                    f"processor_windows={len(lengths)}"
                )

        moved = {}

        for k, v in inputs.items():
            if not torch.is_tensor(v):
                moved[k] = v
            elif torch.is_floating_point(v):
                moved[k] = v.to(
                    ctx["model"].device,
                    dtype=ctx["model"].dtype,
                )
            else:
                moved[k] = v.to(
                    ctx["model"].device
                )

        return {
            "inputs": moved,
            "encoder_lengths": lengths,
            "encoder_group_sizes": group_sizes,
        }

    def forward(self, ctx, inputs):
        with torch.inference_mode():
            ctx["model"](
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

    def extra_metadata(self, ctx):
        return {
            "architecture":
                "AudioFlamingo3ForConditionalGeneration",
            "decoder_module":
                ctx["decoder_path"],
            "decoder_layers":
                len(ctx["decoder"]),
            "audio_encoder_module":
                ctx["encoder_path"],
            "audio_encoder_layers":
                len(ctx["encoder"]),
            "audio_encoder_pooling":
                "mean_8_equal_temporal_bins_over_reconstructed_pre_avgpool_windows",
        }


run_generic(Adapter())
