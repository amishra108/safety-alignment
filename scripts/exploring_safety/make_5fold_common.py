#!/usr/bin/env python3

import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/research/milsrg1/user_workspace/am3244/safety-alignment")
JUDGED = ROOT / "results/INITIAL_BENCHMARK/all_judged"

N_FOLDS = 5
SEED = 42
EXPECTED_ROWS = 6400

BASE_RE = re.compile(r"^(jbb_(?:harmful|benign)_\d+)__")


def read_jsonl(path):
    if not path.exists():
        raise RuntimeError(f"Missing input: {path}")

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


def base_prompt_id(task_id):
    m = BASE_RE.match(task_id)

    if not m:
        raise RuntimeError(
            f"Cannot extract base prompt from task_id: {task_id}"
        )

    return m.group(1)


def atomic_write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )

    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def format_counter(counter):
    return ", ".join(
        f"{k}={v}"
        for k, v in sorted(counter.items())
    )


def resolve_input(model_name, modality, safety_split):
    """
    Prefer the final carefully-rechecked judgments.
    Fall back to adjudicated only when no rechecked file exists.
    """

    candidates = [
        JUDGED
        / (
            f"{model_name}.{modality}.{safety_split}"
            ".readjudicated.carefully_rechecked.jsonl"
        ),
        JUDGED
        / (
            f"{model_name}.{modality}.{safety_split}"
            ".adjudicated.carefully_rechecked.jsonl"
        ),
        JUDGED
        / (
            f"{model_name}.{modality}.{safety_split}"
            ".adjudicated.jsonl"
        ),
    ]

    existing = [p for p in candidates if p.exists()]

    if not existing:
        raise RuntimeError(
            "No judgment file found. Tried:\n  "
            + "\n  ".join(str(p) for p in candidates)
        )

    chosen = existing[0]

    print(
        f"[input] {safety_split:7s} {modality:5s} -> "
        f"{chosen.name}",
        flush=True,
    )

    return chosen


def load_model_rows(model_name):
    rows = []
    seen_task_ids = set()

    for safety_split, modality in (
        ("benign", "text"),
        ("benign", "audio"),
        ("harmful", "text"),
        ("harmful", "audio"),
    ):
        path = resolve_input(
            model_name,
            modality,
            safety_split,
        )

        source_rows = read_jsonl(path)

        for row in source_rows:
            task_id = row["task_id"]

            if task_id in seen_task_ids:
                raise RuntimeError(
                    f"Duplicate task_id across inputs: {task_id}"
                )

            seen_task_ids.add(task_id)

            expected_suffix = f"__{modality}"

            if not task_id.endswith(expected_suffix):
                raise RuntimeError(
                    f"Expected {expected_suffix}: {task_id}"
                )

            existing_split = str(
                row.get(
                    "split",
                    row.get(
                        "safety_split",
                        safety_split,
                    ),
                )
            ).lower()

            if existing_split != safety_split:
                raise RuntimeError(
                    f"Split mismatch for {task_id}: "
                    f"{existing_split} != {safety_split}"
                )

            if "decision" not in row:
                raise RuntimeError(
                    f"{task_id}: missing decision"
                )

            decision = int(row["decision"])

            if decision not in (0, 1):
                raise RuntimeError(
                    f"{task_id}: invalid decision={decision}"
                )

            r = dict(row)

            r["base_prompt_id"] = base_prompt_id(task_id)
            r["safety_split"] = safety_split
            r["modality"] = modality
            r["decision"] = decision

            rows.append(r)

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{model_name}: expected {EXPECTED_ROWS} rows, "
            f"got {len(rows)}"
        )

    return rows


def make_assignment(rows):
    groups_by_split = defaultdict(set)

    for row in rows:
        groups_by_split[
            row["safety_split"]
        ].add(
            row["base_prompt_id"]
        )

    test_groups = {
        fold: set()
        for fold in range(1, N_FOLDS + 1)
    }

    rng = random.Random(SEED)

    for safety_split in ("benign", "harmful"):
        groups = sorted(
            groups_by_split[safety_split]
        )

        rng.shuffle(groups)

        for i, group in enumerate(groups):
            fold = (i % N_FOLDS) + 1
            test_groups[fold].add(group)

    all_groups = set().union(
        *groups_by_split.values()
    )

    assignment_count = Counter()

    for groups in test_groups.values():
        for group in groups:
            assignment_count[group] += 1

    bad = {
        group: assignment_count[group]
        for group in all_groups
        if assignment_count[group] != 1
    }

    if bad:
        raise RuntimeError(
            f"Invalid fold assignment: {bad}"
        )

    return groups_by_split, test_groups, all_groups


def qwen25_reference_assignment():
    """
    Read the already-established Qwen2.5-Omni folds and recover
    which base prompt belongs to which test fold.

    This makes the other models provably use the same folds.
    """

    root = (
        ROOT
        / "results/INITIAL_BENCHMARK/qwen25_omni/5fold"
    )

    mapping = {}

    for fold in range(1, N_FOLDS + 1):
        path = root / f"fold_{fold}.jsonl"

        if not path.exists():
            raise RuntimeError(
                f"Missing Qwen2.5-Omni reference fold: {path}"
            )

        for row in read_jsonl(path):
            if row.get("train_test") != "test":
                continue

            group = row.get(
                "base_prompt_id",
                base_prompt_id(row["task_id"]),
            )

            previous = mapping.get(group)

            if previous is not None and previous != fold:
                raise RuntimeError(
                    f"Qwen2.5 reference inconsistency: "
                    f"{group} test in folds {previous} and {fold}"
                )

            mapping[group] = fold

    return mapping


def run(model_name):
    out_root = (
        ROOT
        / "results/INITIAL_BENCHMARK"
        / model_name
        / "5fold"
    )

    print()
    print("=" * 88)
    print("MAKE 5-FOLD DATASET")
    print("=" * 88)
    print("model :", model_name)
    print("seed  :", SEED)
    print("output:", out_root)
    print()

    rows = load_model_rows(model_name)

    (
        groups_by_split,
        test_groups,
        all_groups,
    ) = make_assignment(rows)

    # --------------------------------------------------------
    # Critical cross-model validation:
    # deterministic assignment must exactly equal established
    # Qwen2.5-Omni assignment.
    # --------------------------------------------------------

    reference = qwen25_reference_assignment()

    current = {}

    for fold, groups in test_groups.items():
        for group in groups:
            current[group] = fold

    if set(current) != set(reference):
        raise RuntimeError(
            "Base-prompt universe differs from Qwen2.5-Omni: "
            f"current={len(current)} reference={len(reference)}"
        )

    mismatch = {
        group: (
            reference[group],
            current[group],
        )
        for group in current
        if reference[group] != current[group]
    }

    if mismatch:
        first = list(sorted(mismatch.items()))[:10]

        raise RuntimeError(
            "Fold assignment does NOT match Qwen2.5-Omni. "
            f"First mismatches={first}"
        )

    print(
        "[OK] fold assignment exactly matches qwen25_omni"
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_lines = [
        f"MODEL: {model_name}",
        f"TOTAL ROWS: {len(rows)}",
        f"UNIQUE BASE PROMPTS: {len(all_groups)}",
        f"BENIGN BASE PROMPTS: "
        f"{len(groups_by_split['benign'])}",
        f"HARMFUL BASE PROMPTS: "
        f"{len(groups_by_split['harmful'])}",
        f"SEED: {SEED}",
        "REFERENCE FOLDS: qwen25_omni",
        "",
    ]

    for fold in range(1, N_FOLDS + 1):
        fold_rows = []

        test_set = test_groups[fold]

        for row in rows:
            out = dict(row)

            out["cv_fold"] = fold
            out["train_test"] = (
                "test"
                if row["base_prompt_id"] in test_set
                else "train"
            )

            fold_rows.append(out)

        if len(fold_rows) != EXPECTED_ROWS:
            raise RuntimeError(
                f"Fold {fold}: expected {EXPECTED_ROWS} rows, "
                f"got {len(fold_rows)}"
            )

        train_prompts = {
            r["base_prompt_id"]
            for r in fold_rows
            if r["train_test"] == "train"
        }

        test_prompts = {
            r["base_prompt_id"]
            for r in fold_rows
            if r["train_test"] == "test"
        }

        overlap = train_prompts & test_prompts

        if overlap:
            raise RuntimeError(
                f"Fold {fold} leakage: "
                f"{sorted(overlap)[:10]}"
            )

        outfile = (
            out_root
            / f"fold_{fold}.jsonl"
        )

        atomic_write_jsonl(
            outfile,
            fold_rows,
        )

        header = (
            "\n"
            + "=" * 88
            + f"\nFOLD {fold}\n"
            + "=" * 88
        )

        print(header)
        summary_lines.append(header)

        for role in ("train", "test"):
            part = [
                r
                for r in fold_rows
                if r["train_test"] == role
            ]

            prompt_count = len({
                r["base_prompt_id"]
                for r in part
            })

            language = Counter(
                r["language"]
                for r in part
            )

            condition = Counter(
                r["condition"]
                for r in part
            )

            safety = Counter(
                r["safety_split"]
                for r in part
            )

            modality = Counter(
                r["modality"]
                for r in part
            )

            decision = Counter(
                str(int(r["decision"]))
                for r in part
            )

            joint = {}

            for safety_name in (
                "benign",
                "harmful",
            ):
                for modality_name in (
                    "text",
                    "audio",
                ):
                    joint[
                        (
                            safety_name,
                            modality_name,
                        )
                    ] = Counter(
                        str(int(r["decision"]))
                        for r in part
                        if (
                            r["safety_split"]
                            == safety_name
                            and r["modality"]
                            == modality_name
                        )
                    )

            block = [
                "",
                role.upper(),
                f"  rows        : {len(part)}",
                f"  base prompts: {prompt_count}",
                f"  language    : "
                f"{format_counter(language)}",
                f"  condition   : "
                f"{format_counter(condition)}",
                f"  safety      : "
                f"{format_counter(safety)}",
                f"  modality    : "
                f"{format_counter(modality)}",
                f"  decision    : "
                f"{format_counter(decision)}",
                "",
                "  BENIGN",
                f"    text      : "
                f"{format_counter(joint[('benign', 'text')])}",
                f"    audio     : "
                f"{format_counter(joint[('benign', 'audio')])}",
                "",
                "  HARMFUL",
                f"    text      : "
                f"{format_counter(joint[('harmful', 'text')])}",
                f"    audio     : "
                f"{format_counter(joint[('harmful', 'audio')])}",
            ]

            for line in block:
                print(line)
                summary_lines.append(line)

        check = (
            f"\n  prompt overlap train/test: "
            f"{len(overlap)}"
        )

        print(check)
        summary_lines.append(check)

    summary = out_root / "summary.txt"

    summary.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print("model    :", model_name)
    print("output   :", out_root)
    print("folds    :", N_FOLDS)
    print("rows/fold:", len(rows))
    print("summary  :", summary)


if __name__ == "__main__":
    raise RuntimeError(
        "Use a model-specific make_5fold.py wrapper."
    )
