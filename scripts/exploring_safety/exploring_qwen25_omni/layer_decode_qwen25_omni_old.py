#!/usr/bin/env python3
import argparse
import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
SOURCE_ROOT = ROOT / "results/INITIAL_BENCHMARK/qwen25_omni"
MANIFEST = ROOT / "data/audio_jailbreak_clean_benchmark/manifests/benchmark.csv"
MODEL_PATH = ROOT / "models/qwen2.5-omni-7b"
OUT_ROOT = ROOT / "results/INITIAL_BENCHMARK/qwen25_omni"

MAX_NEW_TOKENS = 1024

DROP_FIELDS = {
    "response_text",
    "decision_text",
    "response_audio",
    "decision_audio",
}


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSONL {path}:{ln}"
                ) from exc
    return rows


def read_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def append_jsonl(path, row):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def completed_stage_keys(path):
    if not path.exists():
        return set()

    out = set()

    for row in read_jsonl(path):
        key = (
            row["task_id"],
            int(row["layer"]),
        )

        if key in out:
            raise RuntimeError(
                f"Duplicate task/layer in {path}: {key}"
            )

        out.add(key)

    return out


def base_record(row):
    return {
        k: v
        for k, v in row.items()
        if k not in DROP_FIELDS
    }


def task_folder_name(row):
    task_id = str(row["task_id"])

    for suffix in ("__text", "__audio"):
        if task_id.endswith(suffix):
            return task_id[:-len(suffix)]

    return task_id


def validate_output_name(name):
    if not name or name in {".", ".."}:
        raise RuntimeError("--output-name must be a non-empty folder name")

    if "/" in name or "\\" in name:
        raise RuntimeError("--output-name must be a single folder name")

    return name


def build_manifest_index(rows):
    out = {}

    for row in rows:
        eid = row["example_id"]

        if eid in out:
            raise RuntimeError(
                f"Duplicate manifest example_id: {eid}"
            )

        out[eid] = row

    return out


def manifest_row_for_selected(row, by_id):
    task_id = row["task_id"]
    suffix = "__text"

    if not task_id.endswith(suffix):
        raise RuntimeError(
            f"Expected task_id ending {suffix}: {task_id}"
        )

    eid = task_id[:-len(suffix)]

    if eid not in by_id:
        raise RuntimeError(
            f"No benchmark.csv row for {eid}"
        )

    m = by_id[eid]

    for key in (
        "split",
        "language",
        "condition",
        "source_text",
        "prompt_text",
    ):
        if str(row[key]) != str(m[key]):
            raise RuntimeError(
                f"Manifest mismatch for "
                f"{eid} field={key}"
            )

    wav = ROOT / m["audio_path"]

    if (
        not wav.exists()
        or wav.stat().st_size <= 44
    ):
        raise RuntimeError(
            f"Missing/invalid WAV: {wav}"
        )

    return m


FAMILY_PREFIX = {
    ("text", "thinker"): "textthinker",
    ("audio", "encoder"): "audioencoder",
    ("audio", "thinker"): "audiothinker",
}


def family_prefix(modality, family):
    key = (modality, family)

    if key not in FAMILY_PREFIX:
        raise RuntimeError(f"Unknown family: {key}")

    return FAMILY_PREFIX[key]


def canonical_output_file(outdir, modality, family):
    return (
        outdir
        / f"{family_prefix(modality, family)}_responses.jsonl"
    )


def shard_root(outdir, num_shards):
    return (
        outdir
        / ".shards"
        / f"n{num_shards:03d}"
    )


def shard_output_file(
    outdir,
    modality,
    family,
    num_shards,
    shard_index,
):
    return (
        shard_root(outdir, num_shards)
        / (
            f"{family_prefix(modality, family)}"
            f".shard{shard_index:03d}"
            f"-of-{num_shards:03d}.jsonl"
        )
    )


def shard_done_file(
    outdir,
    num_shards,
    shard_index,
):
    return (
        shard_root(outdir, num_shards)
        / (
            f"shard{shard_index:03d}"
            f"-of-{num_shards:03d}.done"
        )
    )


def expected_family_layers(modality, family):
    if (modality, family) == ("text", "thinker"):
        return range(29)

    if (modality, family) == ("audio", "encoder"):
        return range(33)

    if (modality, family) == ("audio", "thinker"):
        return range(29)

    raise RuntimeError(
        f"Unknown family: {modality}/{family}"
    )


def merge_shards(
    outdir,
    selected,
    num_shards,
):
    root = shard_root(
        outdir,
        num_shards,
    )

    done_files = [
        shard_done_file(
            outdir,
            num_shards,
            i,
        )
        for i in range(num_shards)
    ]

    waiting = [
        i
        for i, path in enumerate(done_files)
        if not path.exists()
    ]

    if waiting:
        print(
            f"[merge] waiting for shards: {waiting}",
            flush=True,
        )
        return False

    example_order = {
        row["task_id"]: i
        for i, row in enumerate(selected)
    }

    families = (
        ("text", "thinker"),
        ("audio", "encoder"),
        ("audio", "thinker"),
    )

    for modality, family in families:
        expected = {
            (row["task_id"], layer)
            for row in selected
            for layer in expected_family_layers(
                modality,
                family,
            )
        }

        merged = {}

        for shard_index in range(num_shards):
            path = shard_output_file(
                outdir,
                modality,
                family,
                num_shards,
                shard_index,
            )

            if not path.exists():
                continue

            for row in read_jsonl(path):
                key = (
                    row["task_id"],
                    int(row["layer"]),
                )

                if key not in expected:
                    raise RuntimeError(
                        f"Unexpected row in {path}: {key}"
                    )

                if key in merged:
                    raise RuntimeError(
                        f"Duplicate merged row: {key}"
                    )

                merged[key] = row

        missing = expected - set(merged)

        if missing:
            raise RuntimeError(
                f"{modality}/{family}: "
                f"missing {len(missing)} rows; "
                f"examples={sorted(missing)[:10]}"
            )

        ordered = sorted(
            merged.values(),
            key=lambda row: (
                example_order[row["task_id"]],
                int(row["layer"]),
            ),
        )

        canonical = canonical_output_file(
            outdir,
            modality,
            family,
        )

        canonical.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = canonical.with_name(
            canonical.name
            + f".tmp.{os.getpid()}"
        )

        with tmp.open(
            "w",
            encoding="utf-8",
        ) as f:
            for row in ordered:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp,
            canonical,
        )

        print(
            f"[merged] {canonical}: "
            f"{len(ordered)} rows",
            flush=True,
        )

    return True



def consolidate_individual_outputs(
    outdir,
    row,
    num_shards,
):
    """
    Consolidate any canonical rows and rows from the previous
    stage-sharded layout into the three canonical JSONL files.

    This is safe because under example sharding exactly one worker
    owns this output directory.
    """
    families = (
        ("text", "thinker"),
        ("audio", "encoder"),
        ("audio", "thinker"),
    )

    task_id = row["task_id"]

    for modality, family in families:
        canonical = canonical_output_file(
            outdir,
            modality,
            family,
        )

        merged = {}

        sources = [canonical]

        # Exact legacy shard paths from the previous run configuration.
        sources.extend(
            shard_output_file(
                outdir,
                modality,
                family,
                num_shards,
                shard_index,
            )
            for shard_index in range(num_shards)
        )

        for path in sources:
            if not path.exists():
                continue

            for record in read_jsonl(path):
                if record["task_id"] != task_id:
                    raise RuntimeError(
                        f"Unexpected task_id in {path}: "
                        f"{record['task_id']} != {task_id}"
                    )

                key = (
                    record["task_id"],
                    int(record["layer"]),
                )

                if key in merged:
                    if merged[key] != record:
                        raise RuntimeError(
                            f"Conflicting duplicate row "
                            f"for {key} between existing outputs"
                        )
                    continue

                merged[key] = record

        if not merged:
            continue

        ordered = sorted(
            merged.values(),
            key=lambda record: int(record["layer"]),
        )

        canonical.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = canonical.with_name(
            canonical.name
            + f".tmp.{os.getpid()}"
        )

        with tmp.open(
            "w",
            encoding="utf-8",
        ) as f:
            for record in ordered:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            tmp,
            canonical,
        )



def activation_file(
    outdir,
    modality,
    family,
    layer,
    task_id,
):
    name = (
        "input"
        if layer == 0
        else f"layer_{layer:02d}"
    )

    safe = task_id.replace("/", "_")

    return (
        outdir
        / "representations"
        / modality
        / family
        / name
        / f"{safe}.pt"
    )


def first_tensor(obj):
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (tuple, list)):
        for x in obj:
            t = first_tensor(x)

            if t is not None:
                return t

    if isinstance(obj, dict):
        for x in obj.values():
            t = first_tensor(x)

            if t is not None:
                return t

    return None


class CaptureOnce:
    def __init__(self):
        self.value = None

    def pre(self, module, args):
        if self.value is not None:
            return

        t = first_tensor(args)

        if t is None:
            raise RuntimeError(
                "No tensor in pre-hook for "
                f"{module.__class__.__name__}"
            )

        self.value = (
            t.detach()
            .to(
                "cpu",
                dtype=torch.float16,
            )
        )

    def post(
        self,
        module,
        args,
        output,
    ):
        if self.value is not None:
            return

        t = first_tensor(output)

        if t is None:
            raise RuntimeError(
                "No tensor in post-hook for "
                f"{module.__class__.__name__}"
            )

        self.value = (
            t.detach()
            .to(
                "cpu",
                dtype=torch.float16,
            )
        )


@contextmanager
def stage_model(
    model,
    modality,
    family,
    layer,
    full_audio_layers,
    full_thinker_layers,
):
    audio_tower = model.thinker.audio_tower
    thinker_text = model.thinker.model

    audio_tower.layers = full_audio_layers
    thinker_text.layers = full_thinker_layers

    try:
        if (
            modality == "audio"
            and family == "encoder"
        ):
            audio_tower.layers = (
                torch.nn.ModuleList(
                    list(full_audio_layers)[:layer]
                )
            )

        elif family == "thinker":
            thinker_text.layers = (
                torch.nn.ModuleList(
                    list(full_thinker_layers)[:layer]
                )
            )

        else:
            raise RuntimeError(
                f"Bad stage: "
                f"{modality}/{family}/{layer}"
            )

        yield

    finally:
        audio_tower.layers = full_audio_layers
        thinker_text.layers = full_thinker_layers


def prepare_inputs(
    processor,
    row,
    manifest_row,
    modality,
    process_mm_info,
):
    if modality == "text":
        conversation = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": row["prompt_text"],
            }],
        }]

        chat_text = (
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        return processor(
            text=chat_text,
            return_tensors="pt",
            padding=True,
        )

    audio_path = (
        ROOT
        / manifest_row["audio_path"]
    ).resolve()

    conversation = [{
        "role": "user",
        "content": [{
            "type": "audio",
            "audio":
                "file://" + str(audio_path),
        }],
    }]

    chat_text = (
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    audios, images, videos = (
        process_mm_info(
            conversation,
            use_audio_in_video=False,
        )
    )

    return processor(
        text=chat_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )


def move_inputs(
    inputs,
    device,
):
    moved = {}

    for k, v in inputs.items():
        moved[k] = (
            v.to(device)
            if torch.is_tensor(v)
            else v
        )

    return moved


def run_stage(
    model,
    processor,
    inputs,
    modality,
    family,
    layer,
    activation_path,
):
    handles = []
    captures = {}

    if family == "encoder":
        raw = CaptureOnce()
        projected = CaptureOnce()

        handles.append(
            model.thinker.audio_tower.ln_post
            .register_forward_pre_hook(
                raw.pre
            )
        )

        handles.append(
            model.thinker.audio_tower.proj
            .register_forward_hook(
                projected.post
            )
        )

        captures["raw_1280"] = raw
        captures["projected_3584"] = projected

    else:
        hidden = CaptureOnce()

        handles.append(
            model.thinker.model.norm
            .register_forward_pre_hook(
                hidden.pre
            )
        )

        captures["hidden_3584"] = hidden

    input_len = (
        inputs["input_ids"].shape[1]
    )

    try:
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                return_audio=False,
                thinker_do_sample=False,
                use_audio_in_video=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
            )

    finally:
        for h in handles:
            h.remove()

    if isinstance(out, tuple):
        out = out[0]

    response = (
        processor.batch_decode(
            out[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )

    payload = {
        "modality": modality,
        "family": family,
        "layer": layer,
        "max_new_tokens":
            MAX_NEW_TOKENS,
    }

    for name, cap in captures.items():
        if cap.value is None:
            raise RuntimeError(
                f"Capture failed: "
                f"{modality}/"
                f"{family}/"
                f"{layer}/"
                f"{name}"
            )

        payload[name] = cap.value

    activation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        activation_path,
    )

    return response


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--source-file",
        required=True,
        help="Input JSONL filename under results/INITIAL_BENCHMARK/qwen25_omni",
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
    )

    p.add_argument(
        "--limit",
        type=int,
        required=True,
    )

    p.add_argument(
        "--output-name",
        default=None,
        help="Single output folder name under results/INITIAL_BENCHMARK/qwen25_omni",
    )

    p.add_argument(
        "--individual-folders",
        action="store_true",
        help="Write each selected example to its own task-ID folder",
    )

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
        "--device-map",
        choices=[
            "auto",
            "balanced",
        ],
        default="auto",
    )

    p.add_argument(
        "--preflight",
        action="store_true",
    )

    args = p.parse_args()

    if (
        args.start < 0
        or args.limit < 1
    ):
        raise RuntimeError(
            "--start must be >=0 "
            "and --limit >=1"
        )

    if (
        args.num_shards < 1
        or not (
            0
            <= args.shard_index
            < args.num_shards
        )
    ):
        raise RuntimeError(
            "Invalid shard configuration"
        )

    if (
        not args.source_file
        or "/" in args.source_file
        or "\\" in args.source_file
        or not args.source_file.endswith(".jsonl")
    ):
        raise RuntimeError(
            "--source-file must be a JSONL filename, not a path"
        )

    source = (
        SOURCE_ROOT
        / args.source_file
    )

    if not source.exists():
        raise RuntimeError(
            f"Source JSONL does not exist: {source}"
        )

    selected_all = read_jsonl(
        source
    )

    source_examples_root = (
        OUT_ROOT
        / source.stem
    )

    selected = selected_all[
        args.start:
        args.start + args.limit
    ]

    if len(selected) != args.limit:
        raise RuntimeError(
            f"Requested {args.limit} "
            f"rows at start={args.start}; "
            f"only found {len(selected)}"
        )

    total_selected = len(selected)

    base = total_selected // args.num_shards
    remainder = total_selected % args.num_shards

    local_start = (
        args.shard_index * base
        + min(args.shard_index, remainder)
    )

    local_count = (
        base
        + (
            1
            if args.shard_index < remainder
            else 0
        )
    )

    local_end = local_start + local_count

    selected = selected[
        local_start:local_end
    ]

    print(
        f"example shard: "
        f"global[{local_start}:{local_end}] "
        f"({len(selected)} examples)",
        flush=True,
    )

    manifest_by_id = (
        build_manifest_index(
            read_csv(MANIFEST)
        )
    )

    manifest_rows = [
        manifest_row_for_selected(
            r,
            manifest_by_id,
        )
        for r in selected
    ]

            # HARD RESUME CHECK:
            # One folder = one example; one JSONL = one modality/family.
            # Re-read the physical JSONL immediately before generation.
            done_layers = set()
            if outfile.exists():
                for _r in read_jsonl(outfile):
                    if "layer" in _r:
                        done_layers.add(int(_r["layer"]))

            if int(layer) in done_layers:
                print(
                    f"[skip] "
                    f"{modality}/"
                    f"{family}/"
                    f"{layer:02d}",
                    flush=True,
                )
                continue

            print(
                f"[run ] "
                f"{modality}/"
                f"{family}/"
                f"{layer:02d}",
                flush=True,
            )

            inputs = prepare_inputs(
                processor,
                row,
                mrow,
                modality,
                process_mm_info,
            )

            inputs = move_inputs(
                inputs,
                input_device,
            )

            act = activation_file(
                row_outdir,
                modality,
                family,
                layer,
                row["task_id"],
            )

            with stage_model(
                model,
                modality,
                family,
                layer,
                full_audio_layers,
                full_thinker_layers,
            ):
                response = run_stage(
                    model,
                    processor,
                    inputs,
                    modality,
                    family,
                    layer,
                    act,
                )

            record = base_record(row)
            record["layer"] = layer
            record["response"] = response

            append_jsonl(
                outfile,
                record,
            )

            print(
                f"[done] "
                f"{modality}/"
                f"{family}/"
                f"{layer:02d} "
                f"response_chars="
                f"{len(response)}",
                flush=True,
            )

        if args.individual_folders:
            print(
                f"[example complete] "
                f"{row['task_id']} "
                f"owner_shard={args.shard_index}",
                flush=True,
            )

    # Single-folder mode uses one completion marker only after
    # this worker has finished every selected example.
    if not args.individual_folders:
        donefile = shard_done_file(
            outdir,
            args.num_shards,
            args.shard_index,
        )

        donefile.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        donefile.write_text(
            "complete\n",
            encoding="utf-8",
        )

        merge_shards(
            outdir,
            selected,
            args.num_shards,
        )

    print(
        f"\n[OK] shard "
        f"{args.shard_index}/"
        f"{args.num_shards} complete",
        flush=True,
    )



if __name__ == "__main__":
    main()
