"""Reproducible Meeting 5 workflow: one shared sample, two matched model pairs.

The transformer consumes observed opponent summaries, not the forecaster's learned
probabilities. Both tasks use recorded human actions, not optimal-policy labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.holdem_history import (
    PROFILE_SECTION,
    features,
    load_histories,
    shared_sample,
    without_history,
)
from poker_decision.source_experiments import PHH_SHA256, PHH_URL

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/holdem_integration"
QUESTION = "What action will the acting player take?"
VARIANTS = ("without_history", "with_history")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def code_hashes() -> dict[str, str]:
    return {
        name: digest(Path(__file__).with_name(name))
        for name in ("holdem_history.py", "holdem_integration.py", "training.py")
    } | {"opponents/data.py": digest(Path(__file__).with_name("opponents") / "data.py")}


def metrics(targets: list[int], probabilities: list[list[float]]) -> dict[str, Any]:
    """Local candidate-index metrics; also works for ragged neural menus."""
    if not targets or len(targets) != len(probabilities):
        raise ValueError("Need aligned nonempty targets and predictions")
    for target, p in zip(targets, probabilities, strict=True):
        if (
            not p
            or not all(math.isfinite(v) and v >= 0 for v in p)
            or not math.isclose(sum(p), 1, abs_tol=1e-7)
            or type(target) is not int
            or not 0 <= target < len(p)
        ):
            raise ValueError("Invalid target or probabilities")
    return dict(
        n=len(targets),
        accuracy=float(
            np.mean([int(np.argmax(p)) == t for t, p in zip(targets, probabilities, strict=True)])
        ),
        log_loss=float(
            np.mean(
                [-math.log(max(p[t], 1e-15)) for t, p in zip(targets, probabilities, strict=True)]
            )
        ),
    )


def hero_examples(data: dict[str, Any], split: str, with_history: bool) -> list[DecisionExample]:
    examples = []
    for row in data[split]:
        h = row["hero"]
        inputs = DecisionInputs(h["context"], tuple(h["candidates"]))
        if not with_history:
            inputs = without_history(inputs)
        examples.append(DecisionExample(h["source_id"], h["street"], inputs, h["target_index"]))
    return examples


def validate_sample(data: dict[str, Any]) -> None:
    train_groups, held_groups = set(data["train_sessions"]), set(data["validation_sessions"])
    if not train_groups or not held_groups or not train_groups.isdisjoint(held_groups):
        raise ValueError("Session groups overlap or are empty")
    seen = set()
    for split, groups in (("train", train_groups), ("validation", held_groups)):
        if not data[split]:
            raise ValueError("Empty split")
        for row in data[split]:
            f, h = row["forecast"], row["hero"]
            key = row["source_id"]
            if key in seen or key != f["source_id"] or key != h["source_id"]:
                raise ValueError("Duplicate or mismatched source IDs")
            seen.add(key)
            if (
                row["session_id"] not in groups
                or f["session_id"] != row["session_id"]
                or f["actor_alias"] == "Pluribus"
                or h["target_index"] != f["target_index"]
            ):
                raise ValueError("Invalid split membership or target alignment")
            legal = f["inputs"]["legal_action_ids"]
            categories = [
                0 if c == "fold" else 1 if c in ("check", "call") else 2 for c in h["candidates"]
            ]
            if categories != legal or not 0 <= h["target_index"] < len(legal):
                raise ValueError("Mismatched candidate menus")
            without_history(DecisionInputs(h["context"], tuple(h["candidates"])))


def load_sample(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((out / "preparation.json").read_text())
    if digest(out / "shared_sample.json") != manifest["sample_sha256"]:
        raise ValueError("Saved sample changed; rerun prepare")
    if manifest["code_sha256"] != code_hashes():
        raise ValueError("Integration code changed; rerun prepare and downstream steps")
    data = json.loads((out / "shared_sample.json").read_text())
    validate_sample(data)
    return data, manifest


def download_data(args: argparse.Namespace) -> None:
    from poker_decision.source_experiments import download_pluribus

    if args.archive.name != "pluribus_phh_v1.zip":
        raise ValueError("Download filename must be pluribus_phh_v1.zip")
    print(download_pluribus(args.archive.parent))


def download_model(args: argparse.Namespace) -> None:
    from poker_decision.demo_assets import download_assets, verify_assets

    catalog = json.loads((ROOT / "deploy/streamlit/assets.json").read_text())
    # Support both the current single-model manifest and the refactored model catalog.
    spec = catalog.get("models", {}).get("bert_mini-linear", catalog)
    destination = args.out / "published_bert_mini"
    download_assets(spec["url"], destination, expected_sha256=spec["sha256"])
    verify_assets(destination)
    checkpoint = destination / "model"
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if "bert_uncased_L-4_H-256_A-4" not in metadata.get("backbone", ""):
        raise ValueError("Expected the published BERT Mini checkpoint")
    write_json(
        args.out / "checkpoint_source.json",
        dict(
            url=spec["url"],
            archive_sha256=spec["sha256"],
            checkpoint_sha256=checkpoint_hashes(checkpoint),
        ),
    )
    print(checkpoint)


def prepare(args: argparse.Namespace) -> None:
    prepared = load_histories(
        args.archive,
        hero_per_session=None,
        max_sessions=args.max_sessions,
        hands_per_session=args.hands_per_session,
    )
    data = shared_sample(prepared, seed=args.seed)
    validate_sample(data)
    write_json(args.out / "shared_sample.json", data)
    write_json(
        args.out / "preparation.json",
        dict(
            schema=1,
            source=PHH_URL,
            archive_sha256=PHH_SHA256,
            sample_sha256=digest(args.out / "shared_sample.json"),
            code_sha256=code_hashes(),
            seed=args.seed,
            train_size=len(data["train"]),
            validation_size=len(data["validation"]),
            source_audit=prepared[2],
            train_sessions=data["train_sessions"],
            validation_sessions=data["validation_sessions"],
            train_source_ids=[r["source_id"] for r in data["train"]],
            validation_source_ids=[r["source_id"] for r in data["validation"]],
            versions={name: version(name) for name in ("numpy", "pokerkit")},
            label_semantics="Observed human action category; exact bet sizing abstracted",
        ),
    )
    row = data["validation"][0]
    h = row["hero"]
    base = without_history(DecisionInputs(h["context"], tuple(h["candidates"])))
    (args.out / "example.md").write_text(
        "# One observed decision, two views\n\n## Public-only forecaster input\n\n"
        "~~~json\n" + json.dumps(row["forecast"]["inputs"], indent=2) + "\n~~~\n\n"
        "## Hero without history\n\n"
        + base.context
        + "\n\n## Hero with history\n\n"
        + h["context"]
        + "\n\nCandidates: "
        + ", ".join(h["candidates"])
        + f"\n\nSeparate observed target: {h['candidates'][h['target_index']]}.\n"
    )
    print(f"Prepared 128 train / 64 validation decisions; inspect {args.out / 'example.md'}")


def save_comparison(
    out: Path,
    name: str,
    scores: list[dict[str, Any]],
    manifest: dict[str, Any],
    predictions: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    write_json(
        out / f"{name}.json",
        dict(
            sample_sha256=manifest["sample_sha256"],
            code_sha256=manifest["code_sha256"],
            settings=settings,
            scores=scores,
            predictions=predictions,
        ),
    )
    with (out / f"{name}.csv").open("w") as stream:
        fields = list(dict.fromkeys(k for row in scores for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scores)
    print(json.dumps(scores, indent=2))


def forecast(args: argparse.Namespace) -> None:
    from poker_decision.opponents.models import Forecaster

    data, manifest = load_sample(args.out)
    train = [r["forecast"] for r in data["train"]]
    validation = [r["forecast"] for r in data["validation"]]
    scores, predictions = [], {}
    for name in VARIANTS:
        use_history = name == "with_history"
        model = Forecaster(kind="masked_logistic", c=1.0).fit(
            train, features(train, with_history=use_history)
        )
        p = model.predict(validation, features(validation, with_history=use_history))
        # Forecaster returns global 3-class probabilities; neural model returns local menus.
        local = [
            [float(v[a]) for a in r["inputs"]["legal_action_ids"]]
            for r, v in zip(validation, p, strict=True)
        ]
        scores.append(dict(variant=name, **metrics([r["target_index"] for r in validation], local)))
        predictions[name] = dict(
            source_ids=[r["source_id"] for r in validation],
            probabilities=local,
        )
        write_json(args.out / f"forecaster_{name}.json", model.to_dict())
    save_comparison(
        args.out,
        "forecaster_comparison",
        scores,
        manifest,
        predictions,
        dict(kind="masked_logistic", c=1.0, window=64),
    )


def checkpoint_hashes(path: Path) -> dict[str, str]:
    files = {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not {"head.pt", "metadata.json"} <= files.keys():
        raise ValueError("Incomplete checkpoint")
    return files


def neural_setup(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def check(args: argparse.Namespace) -> None:
    from poker_decision.training import encode_examples, load_checkpoint, predict_encoded

    neural_setup(args.threads)
    data, manifest = load_sample(args.out)
    checkpoint = args.checkpoint or args.out / "published_bert_mini/model"
    tokenizer, model, metadata = load_checkpoint(checkpoint)
    budget = min(512, int(model.encoder.config.max_position_embeddings))
    if model.head_type != "linear" or model.encoder.config.hidden_size != 256:
        raise ValueError("Expected the saved BERT Mini linear model")
    results = []
    for name in VARIANTS:
        use_history = name == "with_history"
        examples = hero_examples(data, "train", use_history) + hero_examples(
            data, "validation", use_history
        )
        encoded = encode_examples(examples, tokenizer, max_length=budget, question=QUESTION)
        p = predict_encoded(model, encoded, pad_token_id=tokenizer.pad_token_id)
        if any(len(v) != len(e.inputs.candidates) for v, e in zip(p, examples, strict=True)):
            raise ValueError("Need one probability per candidate")
        metrics([e.target_index for e in examples], p)
        results.append(
            dict(
                variant=name,
                examples=len(encoded),
                max_tokens=max(len(e.input_ids) for e in encoded),
                valid_predictions=len(p),
            )
        )
    write_json(
        args.out / "input_checks.json",
        dict(
            sample_sha256=manifest["sample_sha256"],
            question=QUESTION,
            checks=results,
            starting_checkpoint_sha256=checkpoint_hashes(checkpoint),
            starting_backbone=metadata.get("backbone"),
        ),
    )
    print(json.dumps(results, indent=2))


def compare(args: argparse.Namespace) -> None:
    import gc

    from poker_decision.training import (
        decision_optimizer,
        encode_examples,
        load_checkpoint,
        predict_encoded,
        save_checkpoint,
        seed_everything,
        train_epoch,
    )

    neural_setup(args.threads)
    data, manifest = load_sample(args.out)
    checkpoint = args.checkpoint or args.out / "published_bert_mini/model"
    starting_hashes = checkpoint_hashes(checkpoint)
    settings = dict(
        seed=data["seed"],
        epochs=1,
        batch_size=16,
        encoder_lr=2e-5,
        head_lr=1e-3,
        weight_decay=0.01,
        max_length=512,
        question=QUESTION,
        device="cpu",
        threads=args.threads,
        starting_checkpoint_sha256=starting_hashes,
        sample_sha256=manifest["sample_sha256"],
        versions={name: version(name) for name in ("torch", "transformers", "numpy")},
        python=platform.python_version(),
    )
    scores, predictions = [], {}
    for name in VARIANTS:
        seed_everything(data["seed"])
        tokenizer, model, metadata = load_checkpoint(checkpoint)
        if model.head_type != "linear" or model.encoder.config.hidden_size != 256:
            raise ValueError("Expected the saved BERT Mini linear model")
        use_history = name == "with_history"
        train, validation = (
            hero_examples(data, split, use_history) for split in ("train", "validation")
        )
        budget = min(settings["max_length"], int(model.encoder.config.max_position_embeddings))
        train_encoded = encode_examples(train, tokenizer, max_length=budget, question=QUESTION)
        val_encoded = encode_examples(validation, tokenizer, max_length=budget, question=QUESTION)
        before = predict_encoded(model, val_encoded, pad_token_id=tokenizer.pad_token_id)
        targets = [e.target_index for e in validation]
        initial = metrics(targets, before)
        optimizer = decision_optimizer(
            model,
            encoder_lr=settings["encoder_lr"],
            head_lr=settings["head_lr"],
            weight_decay=settings["weight_decay"],
        )
        loss = train_epoch(
            model,
            train_encoded,
            [e.target_index for e in train],
            pad_token_id=tokenizer.pad_token_id,
            optimizer=optimizer,
            batch_size=settings["batch_size"],
            seed=data["seed"],
        )
        p = predict_encoded(model, val_encoded, pad_token_id=tokenizer.pad_token_id)
        destination = args.out / "checkpoints" / name
        save_checkpoint(
            destination,
            model,
            tokenizer,
            dict(
                backbone=metadata.get("backbone"),
                max_length=budget,
                question=QUESTION,
                variant=name,
                label_semantics="Observed human action category",
                settings=settings,
                code_sha256=manifest["code_sha256"],
                train_source_ids=[e.source_id for e in train],
                validation_source_ids=[e.source_id for e in validation],
            ),
        )
        del model, optimizer
        gc.collect()
        reloaded_tokenizer, restored, _ = load_checkpoint(destination)
        # Re-encode with the saved tokenizer; verify both tokenizer and weights.
        restored_encoded = encode_examples(
            validation, reloaded_tokenizer, max_length=budget, question=QUESTION
        )
        if restored_encoded != val_encoded:
            raise ValueError("Reloaded tokenizer changed the inputs")
        restored_p = predict_encoded(
            restored, restored_encoded, pad_token_id=reloaded_tokenizer.pad_token_id
        )
        reload_error = max(
            abs(a - b)
            for x, y in zip(p, restored_p, strict=True)
            for a, b in zip(x, y, strict=True)
        )
        if reload_error > 1e-7:
            raise ValueError("Reloaded checkpoint predictions differ")
        scores.append(
            dict(
                variant=name,
                **metrics(targets, p),
                training_loss=loss,
                before_accuracy=initial["accuracy"],
                before_log_loss=initial["log_loss"],
                max_tokens=max(len(e.input_ids) for e in train_encoded + val_encoded),
                reload_max_abs_diff=reload_error,
            )
        )
        predictions[name] = dict(
            source_ids=[e.source_id for e in validation],
            probabilities=p,
            before_probabilities=before,
            checkpoint_sha256=checkpoint_hashes(destination),
        )
        del restored
        gc.collect()
    if checkpoint_hashes(checkpoint) != starting_hashes:
        raise ValueError("Starting checkpoint changed during comparison")
    save_comparison(args.out, "decision_comparison", scores, manifest, predictions, settings)


def report(args: argparse.Namespace) -> None:
    """ZIP-inspired per-decision explorer built from saved predictions; never refits."""
    data, manifest = load_sample(args.out)
    comparisons = {}
    expected_ids = [r["source_id"] for r in data["validation"]]
    for kind in ("forecaster", "decision"):
        comparison = json.loads((args.out / f"{kind}_comparison.json").read_text())
        if (
            comparison["sample_sha256"] != manifest["sample_sha256"]
            or comparison["code_sha256"] != manifest["code_sha256"]
        ):
            raise ValueError("Comparisons belong to a different sample or code revision")
        for name in VARIANTS:
            predictions = comparison["predictions"][name]
            if predictions["source_ids"] != expected_ids:
                raise ValueError("Comparison predictions are not aligned")
        comparisons[kind] = comparison
    summary: dict[str, Any] = dict(
        protocol=manifest,
        comparisons={k: v["scores"] for k, v in comparisons.items()},
        limitation=(
            "128 train / 64 validation: integration check, not evidence of reliable gain or profit."
        ),
        injected_feature="Completed-hand opponent summaries; no learned forecast probabilities",
    )
    write_json(args.out / "summary.json", summary)
    markdown: list[str] = [
        "# Meeting 5: Hold'em opponent-history integration",
        "",
        summary["limitation"],
        "",
        summary["injected_feature"],
        "",
        "| Model | History | Accuracy | Log loss |",
        "| --- | --- | ---: | ---: |",
    ]
    for kind, comparison in comparisons.items():
        for row in comparison["scores"]:
            markdown.append(
                f"| {kind} | {row['variant']} | {row['accuracy']:.2%} | {row['log_loss']:.6f} |"
            )
    (args.out / "results.md").write_text("\n".join(markdown) + "\n")
    cards = []
    for i, row in enumerate(data["validation"]):
        h = row["hero"]
        details = {
            "source_id": row["source_id"],
            "session": row["session_id"],
            "public_forecast_input": row["forecast"]["inputs"],
            "Hero_without_history": h["context"].split(PROFILE_SECTION)[0],
            "Hero_opponent_profiles": h["context"].split(PROFILE_SECTION)[1],
            "candidates": h["candidates"],
            "observed_target": h["candidates"][h["target_index"]],
            "predictions": {
                kind: {
                    name: dict(
                        zip(
                            h["candidates"],
                            comparison["predictions"][name]["probabilities"][i],
                            strict=True,
                        )
                    )
                    for name in VARIANTS
                }
                for kind, comparison in comparisons.items()
            },
        }
        cards.append(
            "<details><summary>"
            + html.escape(row["source_id"])
            + "</summary><pre>"
            + html.escape(json.dumps(details, indent=2))
            + "</pre></details>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Meeting 5 results</title>"
        "<style>body{max-width:1000px;margin:3em auto;padding:1em;font:16px system-ui}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:1em}details{margin:1em 0}"
        "summary{cursor:pointer}</style><h1>Hold'em opponent-history integration</h1><pre>"
        + html.escape("\n".join(markdown[2:]))
        + "</pre>"
        "<h2>64 paired validation decisions</h2>" + "".join(cards)
    )
    (args.out / "explorer.html").write_text(page)
    print("\n".join(markdown))
    print(f"\nExplorer: {args.out / 'explorer.html'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=(
            "download-data",
            "download-model",
            "prepare",
            "forecast",
            "check",
            "compare",
            "report",
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "data/raw/alternative_sources/pluribus_phh_v1.zip"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-sessions", type=int, help="Optional numeric session-group limit")
    parser.add_argument("--hands-per-session", type=int, help="Optional per-directory hand prefix")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    actions = {
        "download-data": download_data,
        "download-model": download_model,
        "prepare": prepare,
        "forecast": forecast,
        "check": check,
        "compare": compare,
        "report": report,
    }
    actions[args.step](args)


if __name__ == "__main__":
    main()
