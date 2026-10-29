"""Reproducible Meeting 1 workflow, also callable without Jupyter."""

import argparse
import hashlib
import json
import platform
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from poker_decision.baselines import LinearCandidateScorer, OptionPrior, uniform
from poker_decision.data import (
    DATA_REVISION,
    SHA256,
    TRAIN_FILES,
    DecisionExample,
    Kind,
    build_examples,
    download_training,
    read_rows,
    split_examples,
)
from poker_decision.evaluation import evaluate_by_street, inspect_mistakes


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def inspect_source(rows: list[dict[str, str]]) -> dict[str, object]:
    """Bounded raw preview, inferred column types, blanks, missing cells, and labels."""
    columns: dict[str, set[str]] = defaultdict(set)
    blank: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if value is None:
                missing[key] += 1
                continue
            if not value.strip():
                blank[key] += 1
                continue
            try:
                number = float(value)
                columns[key].add("integer" if number.is_integer() else "float")
            except ValueError:
                columns[key].add("text")
    return {
        "row_count": len(rows),
        "raw_preview": rows[:3],
        "columns": {
            key: {
                "inferred_types": sorted(columns[key]),
                "missing": missing[key],
                "blank": blank[key],
            }
            for key in (rows[0] if rows else [])
        },
        "action_frequencies": dict(Counter(row.get("correct_decision", "") for row in rows)),
    }


def duplicate_report(examples: list[DecisionExample]) -> dict[str, int]:
    states: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    exact = set()
    for e in examples:
        # Order-independent menu comparison, preserving exact action amounts.
        key = e.inputs.context, tuple(sorted(e.inputs.candidates))
        states[key].add(e.target_action)
        exact.add((*key, e.target_action))
    return {
        "rows": len(examples),
        "distinct_contexts": len({e.inputs.context for e in examples}),
        "repeated_exact_examples": len(examples) - len(exact),
        "conflicting_context_menu_groups": sum(len(targets) > 1 for targets in states.values()),
    }


def prepare(
    root: Path,
    *,
    kinds: tuple[Kind, ...] = ("preflop", "postflop"),
    sample_per_kind: int | None = 5000,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Audit ALL training rows before a reproducible optional working sample and split."""
    if sample_per_kind is not None and sample_per_kind < 2:
        raise ValueError("sample_per_kind must be at least two, or None for all rows")
    if not kinds or len(set(kinds)) != len(kinds):
        raise ValueError("Choose distinct training dataset kinds")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    selected = []
    sources = {}
    processed = root / "data/processed/pokerbench_meeting1"
    for kind in kinds:
        path = download_training(kind, root / "data/raw/pokerbench")
        rows = read_rows(path)
        inspection = inspect_source(rows)
        examples, failures = build_examples(rows, kind)
        failure_path = processed / f"{kind}_parse_failures.jsonl"
        write_jsonl(failure_path, [asdict(f) for f in failures])
        print(
            f"{kind}: {len(rows):,} source rows; "
            f"{len(examples):,} valid; {len(failures):,} failures"
        )
        write_json(processed / f"{kind}_inspection.json", inspection)
        if failures:
            raise ValueError(
                f"Resolve {len(failures)} parsing failures in {failure_path} before training"
            )
        diagnostics = duplicate_report(examples)
        sample = examples
        if sample_per_kind is not None and len(examples) > sample_per_kind:
            sample = random.Random(seed).sample(examples, sample_per_kind)
        selected.extend(sample)
        sources[kind] = {
            "file": TRAIN_FILES[kind],
            "sha256": SHA256[kind],
            "source_rows": len(rows),
            "selected_rows": len(sample),
            "diagnostics": diagnostics,
        }
    train, validation = split_examples(selected, seed=seed, validation_fraction=validation_fraction)
    manifest = {
        "dataset": "RZ412/PokerBench",
        "dataset_revision": DATA_REVISION,
        "sources": sources,
        "seed": seed,
        "validation_fraction_of_contexts": validation_fraction,
        "sample_per_kind": sample_per_kind,
        "grouping": "exact answer-free context",
        "duplicate_policy": "retain and report; identical contexts stay together",
        "train_source_ids": [e.source_id for e in train],
        "validation_source_ids": [e.source_id for e in validation],
        "official_test_used": False,
        "python": platform.python_version(),
        "package_code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    write_jsonl(processed / "train.jsonl", [e.to_record() for e in train])
    write_jsonl(processed / "validation.jsonl", [e.to_record() for e in validation])
    manifest["split_sha256"] = {
        name: hashlib.sha256((processed / f"{name}.jsonl").read_bytes()).hexdigest()
        for name in ("train", "validation")
    }
    write_json(processed / "manifest.json", manifest)
    return train, validation, manifest


def run_baselines(
    root: Path, train: list[DecisionExample], validation: list[DecisionExample], *, seed: int = 42
) -> dict[str, Any]:
    prior = OptionPrior().fit(train)
    linear = LinearCandidateScorer(seed=seed).fit(train)
    metrics = {
        name: evaluate_by_street(validation, predict)
        for name, predict in [
            ("uniform", uniform),
            ("option_prior", prior.predict),
            ("linear_candidate", linear.predict),
        ]
    }
    mistakes = inspect_mistakes(validation, linear.predict, seed=seed)
    report = {
        "metrics": metrics,
        "five_mistakes": mistakes,
        "versions": {p: version(p) for p in ["numpy", "scikit-learn", "poker-decision-model"]},
        "model_settings": {
            "features": "hashed context/action interactions, case-preserved cards",
            "hash_features": 2**18,
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 500,
            "training_weight": "0.5 per positive; 0.5 total negatives per decision",
            "single_option_training": "excluded",
            "seed": seed,
        },
    }
    output = root / "artifacts/pokerbench_meeting1"
    write_json(output / "validation_report.json", report)
    write_jsonl(output / "five_errors.jsonl", mistakes)
    print(json.dumps({name: groups["overall"] for name, groups in metrics.items()}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--kind", choices=["preflop", "postflop", "both"], default="both")
    parser.add_argument(
        "--sample-per-kind", type=int, default=5000, help="0 audits and uses all rows"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    kinds: tuple[Kind, ...] = ("preflop", "postflop") if args.kind == "both" else (args.kind,)
    train, validation, _ = prepare(
        args.root,
        kinds=kinds,
        sample_per_kind=None if args.sample_per_kind == 0 else args.sample_per_kind,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    print(f"Train: {len(train):,}; validation: {len(validation):,}; official test untouched")
    if not args.skip_baseline:
        run_baselines(args.root, train, validation, seed=args.seed)


if __name__ == "__main__":
    main()
