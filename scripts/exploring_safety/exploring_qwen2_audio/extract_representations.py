#!/usr/bin/env python3
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import librosa
import torch

from representation_extract_common import (
    ROOT,
    BoundaryCapture,
    resolve_attr,
    run_generic,
)


class Adapter:
    model_name = "qwen2_audio"

    def load(self):
        from transformers import (
            AutoProcessor,
            Qwen2AudioForConditionalGeneration,
        )

        path = ROOT / "models/qwen2-audio-7b-instruct"

        processor = AutoProcessor.from_pretrained(
            str(path),
            local_files_only=True,
        )

        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"

        model = (
            Qwen2AudioForConditionalGeneration
            .from_pretrained(
                str(path),
                torch_dtype=torch.float16,
                device_map={"": 0},
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        )
        model.eval()

        # ----------------------------------------------------
        # Resolve by actual layer TYPE rather than assuming a
        # transformers-version-specific dotted module path.
        # ----------------------------------------------------

        decoder_candidates = []
        encoder_candidates = []

        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.ModuleList):
                continue

            if len(module) == 0:
                continue

            child_names = {
                child.__class__.__name__
                for child in module
            }

            if child_names == {"Qwen2DecoderLayer"}:
                decoder_candidates.append((name, module))

            if child_names == {"Qwen2AudioEncoderLayer"}:
                encoder_candidates.append((name, module))

        if len(decoder_candidates) != 1:
            print(
                "[diagnostic] decoder candidates:",
                [(n, len(m)) for n, m in decoder_candidates],
                flush=True,
            )
            raise RuntimeError(
                "Expected exactly one Qwen2DecoderLayer ModuleList, "
                f"found {len(decoder_candidates)}"
            )

        if len(encoder_candidates) != 1:
            print(
                "[diagnostic] encoder candidates:",
                [(n, len(m)) for n, m in encoder_candidates],
                flush=True,
            )
            raise RuntimeError(
                "Expected exactly one Qwen2AudioEncoderLayer ModuleList, "
                f"found {len(encoder_candidates)}"
            )

        decoder_path, decoder = decoder_candidates[0]
        encoder_path, encoder = encoder_candidates[0]

        # Find the Qwen2AudioEncoder owner, needed for the exact
        # convolution/output-length calculation used by the model.
        encoder_owner = None
        encoder_owner_path = None

        for name, module in model.named_modules():
            if (
                module.__class__.__name__ == "Qwen2AudioEncoder"
                and hasattr(module, "layers")
                and module.layers is encoder
            ):
                encoder_owner = module
                encoder_owner_path = name
                break

        if encoder_owner is None:
            raise RuntimeError(
                "Found audio encoder layers but not their "
                "Qwen2AudioEncoder owner"
            )

        print(
            f"[architecture] Qwen2 decoder: "
            f"{decoder_path} | layers={len(decoder)}",
            flush=True,
        )

        print(
            f"[architecture] Qwen2 audio encoder: "
            f"{encoder_path} | layers={len(encoder)}",
            flush=True,
        )

        print(
            f"[architecture] Qwen2 audio owner: "
            f"{encoder_owner_path}",
            flush=True,
        )

        text_hidden = getattr(
            model.config.text_config,
            "hidden_size",
            None,
        )

        audio_hidden = getattr(
            model.config.audio_config,
            "d_model",
            None,
        )

        mel_bins = getattr(
            model.config.audio_config,
            "num_mel_bins",
            None,
        )

        print(
            "[architecture] "
            f"decoder_hidden={text_hidden} "
            f"audio_hidden={audio_hidden} "
            f"mel_bins={mel_bins}",
            flush=True,
        )

        # Architecture is taken from the ACTUAL loaded checkpoint.
        # Do not impose Qwen2.5-Omni dimensions on Qwen2-Audio.
        if len(decoder) != model.config.text_config.num_hidden_layers:
            raise RuntimeError(
                "Decoder layer count disagrees with checkpoint config: "
                f"modules={len(decoder)} "
                f"config={model.config.text_config.num_hidden_layers}"
            )

        if len(encoder) != model.config.audio_config.encoder_layers:
            raise RuntimeError(
                "Audio encoder layer count disagrees with checkpoint config: "
                f"modules={len(encoder)} "
                f"config={model.config.audio_config.encoder_layers}"
            )

        if text_hidden is None or int(text_hidden) <= 0:
            raise RuntimeError(
                f"Invalid decoder hidden size: {text_hidden}"
            )

        if audio_hidden is None or int(audio_hidden) <= 0:
            raise RuntimeError(
                f"Invalid audio hidden size: {audio_hidden}"
            )

        return {
            "model": model,
            "processor": processor,
            "decoder": decoder,
            "encoder": encoder,
            "encoder_owner": encoder_owner,
            "decoder_path": decoder_path,
            "encoder_path": encoder_path,
            "encoder_owner_path": encoder_owner_path,
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

        if modality == "text":
            convs = [
                [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": r["prompt_text"],
                    }],
                }]
                for r in rows
            ]

            texts = [
                p.apply_chat_template(
                    c,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for c in convs
            ]

            inputs = p(
                text=texts,
                return_tensors="pt",
                padding=True,
            )

            lengths = None

        else:
            sr = p.feature_extractor.sampling_rate

            convs = []
            wavs = []

            for r in rows:
                ap = ROOT / r["audio_path"]

                wav, _ = librosa.load(
                    str(ap),
                    sr=sr,
                )
                wavs.append(wav)

                convs.append([{
                    "role": "user",
                    "content": [{
                        "type": "audio",
                        "audio_url": "file:" + str(ap),
                    }],
                }])

            texts = [
                p.apply_chat_template(
                    c,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for c in convs
            ]

            inputs = p(
                text=texts,
                audio=wavs,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )

            fam = inputs["feature_attention_mask"]
            feat_lengths = fam.sum(-1).long()

            # First returned length is after conv downsampling,
            # exactly the temporal length used INSIDE encoder layers.
            lengths, _ = (
                ctx["encoder_owner"]
                ._get_feat_extract_output_lengths(
                    feat_lengths
                )
            )

            lengths = lengths.tolist()

        moved = {
            k: (
                v.to(ctx["model"].device)
                if torch.is_tensor(v)
                else v
            )
            for k, v in inputs.items()
        }

        return {
            "inputs": moved,
            "encoder_lengths": lengths,
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
                "Qwen2AudioForConditionalGeneration",
            "decoder_module":
                ctx["decoder_path"],
            "decoder_layers":
                len(ctx["decoder"]),
            "audio_encoder_module":
                ctx["encoder_path"],
            "audio_encoder_layers":
                len(ctx["encoder"]),
            "audio_encoder_pooling":
                "mean_8_equal_temporal_bins_pre_avgpool",
        }


run_generic(Adapter())
