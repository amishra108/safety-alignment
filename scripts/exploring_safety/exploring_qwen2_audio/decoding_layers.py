#!/usr/bin/env python3
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import librosa
import torch

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
import importlib.util

_shared_path = ROOT / "scripts/exploring_safety/decoding_layers.py"
_spec = importlib.util.spec_from_file_location("decoding_layers_shared", _shared_path)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

GROUPS = _shared.GROUPS
run_backend = _shared.run_backend

MODEL_PATH = ROOT / "models/qwen2-audio-7b-instruct"

def resolve_list(root, candidates, expected):
    for dotted in candidates:
        obj = root
        parts = dotted.split(".")
        try:
            for x in parts[:-1]:
                obj = getattr(obj, x)
            value = getattr(obj, parts[-1])
        except AttributeError:
            continue

        if isinstance(value, torch.nn.ModuleList) and len(value) == expected:
            print(f"[resolve] {dotted} -> {len(value)} layers")
            return obj, parts[-1], value

    raise RuntimeError(
        f"Could not resolve {expected}-layer ModuleList from {candidates}"
    )

class Adapter:
    model_name = "qwen2_audio"

    def __init__(self):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        self.model = (
            Qwen2AudioForConditionalGeneration
            .from_pretrained(
                str(MODEL_PATH),
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
        )
        self.model.eval()

        self.audio_parent, self.audio_attr, self.full_audio = resolve_list(
            self.model,
            [
                "audio_tower.layers",
                "model.audio_tower.layers",
            ],
            32,
        )

        self.decoder_parent, self.decoder_attr, self.full_decoder = resolve_list(
            self.model,
            [
                "language_model.model.layers",
                "model.language_model.model.layers",
                "model.language_model.layers",
            ],
            32,
        )

        self.stages = (
            [("text", "decoder", i) for i in range(33)]
            + [("audio", "encoder", i) for i in range(33)]
            + [("audio", "decoder", i) for i in range(33)]
        )

        self.device = next(self.model.parameters()).device

    def output_name(self, modality, family):
        return {
            ("text", "decoder"): "textdecoder_responses.jsonl",
            ("audio", "encoder"): "audioencoder_responses.jsonl",
            ("audio", "decoder"): "audiodecoder_responses.jsonl",
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

            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
                padding=True,
            )

        else:
            audio_path = (ROOT / mrow["audio_path"]).resolve()
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "audio",
                    "audio_url": str(audio_path),
                }],
            }]

            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            wav, _ = librosa.load(
                str(audio_path),
                sr=self.processor.feature_extractor.sampling_rate,
                mono=True,
            )

            try:
                inputs = self.processor(
                    text=prompt,
                    audio=[wav],
                    return_tensors="pt",
                    padding=True,
                )
            except TypeError:
                inputs = self.processor(
                    text=prompt,
                    audios=[wav],
                    return_tensors="pt",
                    padding=True,
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
        setattr(self.audio_parent, self.audio_attr, self.full_audio)
        setattr(self.decoder_parent, self.decoder_attr, self.full_decoder)

        try:
            if family == "encoder":
                setattr(
                    self.audio_parent,
                    self.audio_attr,
                    torch.nn.ModuleList(
                        list(self.full_audio)[:layer]
                    ),
                )
            elif family == "decoder":
                setattr(
                    self.decoder_parent,
                    self.decoder_attr,
                    torch.nn.ModuleList(
                        list(self.full_decoder)[:layer]
                    ),
                )
            else:
                raise RuntimeError(f"Bad family {family}")

            yield
        finally:
            setattr(self.audio_parent, self.audio_attr, self.full_audio)
            setattr(self.decoder_parent, self.decoder_attr, self.full_decoder)

    def generate(self, inputs, modality, family, layer, max_new_tokens):
        input_len = inputs["input_ids"].shape[1]

        with self.stage(modality, family, layer):
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

        out = out[:, input_len:]

        return self.processor.batch_decode(
            out,
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
