#!/usr/bin/env python3
import argparse
import importlib.util
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")

# ---------------------------------------------------------------------
# Shared traversal / selections / resumability only.
# Exactly the same architecture used by Qwen2-Audio.
# ---------------------------------------------------------------------
_shared_path = ROOT / "scripts/exploring_safety/decoding_layers.py"
_spec = importlib.util.spec_from_file_location(
    "decoding_layers_shared",
    _shared_path,
)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

GROUPS = _shared.GROUPS
run_backend = _shared.run_backend

MODEL_PATH = ROOT / "models/qwen2.5-omni-7b"


class Adapter:
    model_name = "qwen25_omni"

    def __init__(self):
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
        from qwen_omni_utils import process_mm_info

        self.process_mm_info = process_mm_info

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )

        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        self.model = (
            Qwen2_5OmniForConditionalGeneration
            .from_pretrained(
                str(MODEL_PATH),
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
        )

        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

        self.model.eval()

        if not hasattr(self.model, "thinker"):
            raise RuntimeError("Expected model.thinker")

        if not hasattr(self.model.thinker, "audio_tower"):
            raise RuntimeError("Expected model.thinker.audio_tower")

        if not hasattr(self.model.thinker, "model"):
            raise RuntimeError("Expected model.thinker.model")

        self.full_audio = self.model.thinker.audio_tower.layers
        self.full_thinker = self.model.thinker.model.layers

        if len(self.full_audio) != 32:
            raise RuntimeError(
                f"Expected 32 Omni audio encoder layers, "
                f"found {len(self.full_audio)}"
            )

        if len(self.full_thinker) != 28:
            raise RuntimeError(
                f"Expected 28 Omni thinker layers, "
                f"found {len(self.full_thinker)}"
            )

        print(
            f"[architecture] "
            f"thinker={len(self.full_thinker)} "
            f"audio_encoder={len(self.full_audio)}"
        )

        # Boundary 0 + each transformer block output.
        self.stages = (
            [("text", "thinker", i) for i in range(29)]
            + [("audio", "encoder", i) for i in range(33)]
            + [("audio", "thinker", i) for i in range(29)]
        )

        self.device = self.model.device

    def output_name(self, modality, family):
        return {
            ("text", "thinker"): "textthinker_responses.jsonl",
            ("audio", "encoder"): "audioencoder_responses.jsonl",
            ("audio", "thinker"): "audiothinker_responses.jsonl",
        }[(modality, family)]

    def prepare(self, row, mrow, modality):
        if modality == "text":
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": row["prompt_text"],
                }],
            }]

            chat_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=chat_text,
                return_tensors="pt",
                padding=True,
            )

        else:
            audio_path = (ROOT / mrow["audio_path"]).resolve()

            conversation = [{
                "role": "user",
                "content": [{
                    "type": "audio",
                    "audio": "file://" + str(audio_path),
                }],
            }]

            chat_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            audios, images, videos = self.process_mm_info(
                conversation,
                use_audio_in_video=False,
            )

            inputs = self.processor(
                text=chat_text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )

        moved = {}
        for k, v in inputs.items():
            moved[k] = (
                v.to(self.device)
                if torch.is_tensor(v)
                else v
            )

        return moved

    @contextmanager
    def stage(self, modality, family, layer):
        audio_tower = self.model.thinker.audio_tower
        thinker = self.model.thinker.model

        # Always restore full architecture before applying the next stage.
        audio_tower.layers = self.full_audio
        thinker.layers = self.full_thinker

        try:
            if modality == "audio" and family == "encoder":
                audio_tower.layers = torch.nn.ModuleList(
                    list(self.full_audio)[:layer]
                )

            elif family == "thinker":
                thinker.layers = torch.nn.ModuleList(
                    list(self.full_thinker)[:layer]
                )

            else:
                raise RuntimeError(
                    f"Bad Omni stage: {modality}/{family}/{layer}"
                )

            yield

        finally:
            audio_tower.layers = self.full_audio
            thinker.layers = self.full_thinker

    def generate(
        self,
        inputs,
        modality,
        family,
        layer,
        max_new_tokens,
    ):
        input_len = inputs["input_ids"].shape[1]

        with self.stage(
            modality,
            family,
            layer,
        ):
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_do_sample=False,
                    use_audio_in_video=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

        if isinstance(out, tuple):
            out = out[0]

        return self.processor.batch_decode(
            out[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--groups", nargs="*", default=None)
    args = p.parse_args()

    adapter = Adapter()

    run_backend(
        adapter,
        args.num_shards,
        args.shard_index,
        args.max_new_tokens,
        args.groups or GROUPS,
        start=args.start,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
