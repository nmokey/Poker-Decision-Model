"""Training-size comparison with nested samples and fixed Pluribus validation.

Reuses the original 128-example checkpoints and the integration training functions.
The original 64 validation rows are the prefix of the expanded validation sample.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from poker_decision.data import DecisionExample
from poker_decision.holdem_history import (
    dataset_fingerprint,
    load_histories,
    shared_sample,
)
from poker_decision.holdem_integration import (
    DEFAULT_OUT,
    ROOT,
    VARIANTS,
    checkpoint_hashes,
    code_hashes,
    compare,
    digest,
    forecast,
    hero_examples,
    load_sample,
    metrics,
    neural_setup,
    save_comparison,
    validate_sample,
    write_json,
)


def expanded_samples(
    prepared: tuple[list[dict[str, Any]], list[DecisionExample], dict[str, Any]],
    reference: dict[str, Any],
    *,
    train_sizes: list[int],
    validation_size: int,
) -> dict[int, dict[str, Any]]:
    """Preserve the reference split/rows; append seeded, answer-independent rows."""
    if (
        not train_sizes
        or len(set(train_sizes)) != len(train_sizes)
        or any(type(n) is not int or n < len(reference["train"]) for n in train_sizes)
        or type(validation_size) is not int
        or validation_size < len(reference["validation"])
    ):
        raise ValueError("Sizes must be unique and at least the original sample sizes")
    validate_sample(reference)
    rows, heroes, audit = prepared
    regenerated = shared_sample(
        prepared,
        seed=reference["seed"],
        train_size=len(reference["train"]),
        validation_size=len(reference["validation"]),
        validation_fraction=reference["validation_fraction"],
    )
    for split in ("train", "validation"):
        if dataset_fingerprint(regenerated[split]) != dataset_fingerprint(reference[split]):
            raise ValueError("Regenerated source rows differ from the reference experiment")
    forecasts = {r["source_id"]: r for r in rows}
    examples = {e.source_id: e for e in heroes}
    chosen = {}
    for split, count in (("train", max(train_sizes)), ("validation", validation_size)):
        groups = set(reference[f"{split}_sessions"])
        initial = [r["source_id"] for r in reference[split]]
        initial_set = set(initial)
        remaining = sorted(
            key
            for key in examples
            if forecasts[key]["session_id"] in groups and key not in initial_set
        )
        # Independent RNG streams: increasing training size never resamples validation.
        random.Random(reference["seed"] + (0 if split == "train" else 1)).shuffle(remaining)
        if len(initial) + len(remaining) < count:
            raise ValueError(f"Not enough {split} decisions")
        chosen[split] = initial + remaining[: count - len(initial)]

    def joined(key: str) -> dict[str, Any]:
        f = forecasts[key]
        return dict(
            source_id=key,
            session_id=f["session_id"],
            table_id=f["table_id"],
            forecast=f,
            hero=examples[key].to_record(),
        )

    train = [joined(key) for key in chosen["train"]]
    validation = [joined(key) for key in chosen["validation"]]
    result = {}
    for count in sorted(train_sizes):
        data = dict(
            train=train[:count],
            validation=validation,
            seed=reference["seed"],
            validation_fraction=reference["validation_fraction"],
            source_audit=audit,
            train_sessions=reference["train_sessions"],
            validation_sessions=reference["validation_sessions"],
        )
        validate_sample(data)
        result[count] = data
    return result


def evaluate_reference(
    out: Path,
    reference_out: Path,
    *,
    threads: int,
) -> None:
    """Evaluate the saved baseline fits, and verify its original 64 probabilities."""
    from poker_decision.holdem_integration import QUESTION
    from poker_decision.training import encode_examples, load_checkpoint, predict_encoded

    data, manifest = load_sample(out)
    reference, _ = load_sample(reference_out)
    original = json.loads((reference_out / "decision_comparison.json").read_text())
    original_scores = {r["variant"]: r for r in original["scores"]}
    neural_setup(threads)
    scores, predictions = [], {}
    for name in VARIANTS:
        checkpoint = reference_out / "checkpoints" / name
        hashes = checkpoint_hashes(checkpoint)
        if hashes != original["predictions"][name]["checkpoint_sha256"]:
            raise ValueError("Original checkpoint changed")
        tokenizer, model, metadata = load_checkpoint(checkpoint)
        rows = hero_examples(data, "validation", name == "with_history")
        encoded = encode_examples(
            rows,
            tokenizer,
            max_length=metadata["max_length"],
            question=QUESTION,
        )
        p = predict_encoded(model, encoded, pad_token_id=tokenizer.pad_token_id)
        old_p = original["predictions"][name]["probabilities"]
        old_ids = original["predictions"][name]["source_ids"]
        if [r.source_id for r in rows[: len(old_ids)]] != old_ids:
            raise ValueError("Original validation rows changed")
        max_error = max(
            abs(a - b)
            for x, y in zip(p[: len(old_p)], old_p, strict=True)
            for a, b in zip(x, y, strict=True)
        )
        if max_error > 1e-7:
            raise ValueError("Original checkpoint probabilities no longer reproduce")
        scores.append(
            dict(
                variant=name,
                **metrics([r.target_index for r in rows], p),
                training_loss=original_scores[name]["training_loss"],
                original_64_max_abs_diff=max_error,
                max_tokens=max(len(e.input_ids) for e in encoded),
            )
        )
        predictions[name] = dict(
            source_ids=[r.source_id for r in rows],
            probabilities=p,
            checkpoint_sha256=hashes,
        )
        del model
    settings = copy.deepcopy(original["settings"])
    settings.update(
        sample_sha256=manifest["sample_sha256"],
        reference_checkpoint_run=str(reference_out),
        evaluation_only=True,
        original_training_examples=len(reference["train"]),
    )
    save_comparison(out, "decision_comparison", scores, manifest, predictions, settings)


def paired_change(data: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    """Descriptive paired intervals over numeric validation-session groups."""
    a, b = (comparison["predictions"][name]["probabilities"] for name in VARIANTS)
    targets = [r["hero"]["target_index"] for r in data["validation"]]
    losses = np.array(
        [
            -np.log(max(y[t], 1e-15)) + np.log(max(x[t], 1e-15))
            for x, y, t in zip(a, b, targets, strict=True)
        ]
    )
    hits = np.array(
        [
            float(np.argmax(y) == t) - float(np.argmax(x) == t)
            for x, y, t in zip(a, b, targets, strict=True)
        ]
    )
    groups = sorted({r["session_id"] for r in data["validation"]})
    membership = np.array(
        [[r["session_id"] == group for r in data["validation"]] for group in groups]
    )
    counts = membership.sum(axis=1)
    weights = np.random.default_rng(42).multinomial(
        len(groups),
        [1 / len(groups)] * len(groups),
        size=2000,
    )
    result: dict[str, Any] = dict(
        direction="with_history minus without_history",
        validation_session_groups=len(groups),
        scope="Descriptive session-cluster intervals conditional on these fits; no retraining",
    )
    for name, delta in (("accuracy", hits), ("log_loss", losses)):
        bootstrap = weights @ (membership @ delta) / (weights @ counts)
        result[name] = dict(
            estimate=float(delta.mean()),
            interval_95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        )
    return result


def summarize(out: Path, sizes: list[int], baseline_validation_size: int) -> dict[str, Any]:
    results = []
    for count in sorted(sizes):
        run = out / f"train_{count:05d}"
        data, manifest = load_sample(run)
        for kind in ("forecaster", "decision"):
            comparison = json.loads((run / f"{kind}_comparison.json").read_text())
            if comparison["sample_sha256"] != manifest["sample_sha256"]:
                raise ValueError("Stale comparison sample")
            expected = [r["source_id"] for r in data["validation"]]
            for name in VARIANTS:
                prediction = comparison["predictions"][name]
                if prediction["source_ids"] != expected:
                    raise ValueError("Misaligned comparison")
                p = prediction["probabilities"]
                targets = [r["hero"]["target_index"] for r in data["validation"]]
                for n, scope in (
                    (len(targets), "expanded"),
                    (baseline_validation_size, "original"),
                ):
                    results.append(
                        dict(
                            model=kind,
                            train_examples=count,
                            variant=name,
                            validation_scope=scope,
                            **metrics(targets[:n], p[:n]),
                        )
                    )
            write_json(run / f"{kind}_paired_change.json", paired_change(data, comparison))
    summary = dict(
        results=results,
        original_validation_size=baseline_validation_size,
        protocol=json.loads((out / "protocol.json").read_text()),
        caution=(
            "One seed, one epoch, inspected session validation; descriptive learning curve. "
            "Larger samples also entail more optimizer updates."
        ),
    )
    write_json(out / "summary.json", summary)
    lines = [
        "# Hold'em training-size comparison",
        "",
        summary["caution"],
        "",
        "| Model | Train | Validation | Variant | Accuracy | Log loss |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['train_examples']} | {row['n']} | {row['variant']} | "
            f"{row['accuracy']:.2%} | {row['log_loss']:.6f} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--reference-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/holdem_scaling")
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "data/raw/alternative_sources/pluribus_phh_v1.zip"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Choose a new output directory to preserve prior runs")
    if args.threads < 1:
        parser.error("threads must be positive")
    reference, reference_manifest = load_sample(args.reference_out)
    checkpoint = args.checkpoint or args.reference_out / "published_bert_mini/model"
    if (
        checkpoint_hashes(checkpoint)
        != json.loads((args.reference_out / "checkpoint_source.json").read_text())[
            "checkpoint_sha256"
        ]
    ):
        raise ValueError("Starting model differs from the reference")
    started = time.perf_counter()
    print("Replaying real archive and building nested samples", flush=True)
    prepared = load_histories(args.archive, hero_per_session=None)
    samples = expanded_samples(
        prepared,
        reference,
        train_sizes=args.train_sizes,
        validation_size=args.validation_size,
    )
    protocol = dict(
        host=socket.gethostname(),
        started_utc=datetime.now(UTC).isoformat(),
        train_sizes=sorted(args.train_sizes),
        validation_size=args.validation_size,
        reference_sample_sha256=reference_manifest["sample_sha256"],
        original_validation_size=len(reference["validation"]),
        seed=reference["seed"],
        split_policy="Unchanged original numeric-session groups; nested train, fixed validation",
        source_audit=prepared[2],
        code_sha256=code_hashes(),
        scaling_code_sha256=digest(Path(__file__)),
        device="cpu",
        threads=args.threads,
        epochs=1,
        batch_size=16,
        encoder_lr=2e-5,
        head_lr=1e-3,
        weight_decay=0.01,
        starting_checkpoint_sha256=checkpoint_hashes(checkpoint),
    )
    write_json(args.out / "protocol.json", protocol)
    del prepared
    for count, data in samples.items():
        out = args.out / f"train_{count:05d}"
        write_json(out / "shared_sample.json", data)
        manifest = dict(
            schema=1,
            source_audit=data["source_audit"],
            train_size=count,
            validation_size=len(data["validation"]),
            sample_sha256=digest(out / "shared_sample.json"),
            code_sha256=code_hashes(),
            scaling_code_sha256=digest(Path(__file__)),
            reference_sample_sha256=reference_manifest["sample_sha256"],
            train_sessions=data["train_sessions"],
            validation_sessions=data["validation_sessions"],
            train_source_ids=[r["source_id"] for r in data["train"]],
            validation_source_ids=[r["source_id"] for r in data["validation"]],
        )
        write_json(out / "preparation.json", manifest)
        print(f"Training size {count}; fixed validation {len(data['validation'])}", flush=True)
        run_args = argparse.Namespace(out=out, checkpoint=checkpoint, threads=args.threads)
        forecast(run_args)
        if count == len(reference["train"]):
            original_forecast = json.loads(
                (args.reference_out / "forecaster_comparison.json").read_text()
            )
            refitted_forecast = json.loads((out / "forecaster_comparison.json").read_text())
            for name in VARIANTS:
                p = refitted_forecast["predictions"][name]["probabilities"]
                old_p = original_forecast["predictions"][name]["probabilities"]
                if not np.allclose(p[: len(old_p)], old_p, atol=1e-10, rtol=0):
                    raise ValueError("Original forecaster probabilities no longer reproduce")
            evaluate_reference(out, args.reference_out, threads=args.threads)
        else:
            compare(run_args)
        print(f"Finished {count}; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    protocol["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.out / "protocol.json", protocol)
    summarize(args.out, args.train_sizes, len(reference["validation"]))
    print(f"Completed: {args.out / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
