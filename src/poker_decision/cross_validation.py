"""Fixed-setting, fresh-fit scenario CV with portable checkpoints and verified resume."""

import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poker_decision.baselines import LinearCandidateScorer, OptionPrior, uniform
from poker_decision.data import DecisionExample
from poker_decision.diagnostics import paired_comparison, prediction_diagnostics
from poker_decision.evaluation import evaluate_by_street, inspect_mistakes, probabilities
from poker_decision.models.encoder import BACKBONES, load_pretrained
from poker_decision.scenarios import digest, make_folds, scenario_group
from poker_decision.training import (
    decision_optimizer,
    encode_examples,
    load_checkpoint,
    predict_encoded,
    prediction_lookup,
    save_checkpoint,
    seed_everything,
    train_epoch,
)


@dataclass(frozen=True)
class CVConfig:
    backbone: str = "bert_mini"
    heads: tuple[str, ...] = ("linear",)
    head_width: int = 64
    n_splits: int = 5
    seed: int = 42
    epochs: int = 1
    batch_size: int = 16
    max_length: int = 512
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        if (
            self.backbone not in BACKBONES
            or not self.heads
            or len(set(self.heads)) != len(self.heads)
        ):
            raise ValueError("Choose a known backbone and distinct nonempty heads")
        if not set(self.heads) <= {"linear", "mlp"}:
            raise ValueError("Unknown head")
        if min(self.head_width, self.n_splits, self.epochs, self.batch_size, self.max_length) < 1:
            raise ValueError("Counts must be positive")
        if not all(math.isfinite(v) and v > 0 for v in (self.encoder_lr, self.head_lr)):
            raise ValueError("Learning rates must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


def sweep_configs(
    base: CVConfig, grid: Mapping[str, Sequence[Any]], *, max_trials: int = 8
) -> list[CVConfig]:
    """Bounded development grid; keep fold membership and random seed fixed.

    These comparisons may guide tuning, but their winning score is not an
    independent test estimate. No automatic winner selection is performed.
    """
    allowed = {
        "backbone",
        "heads",
        "head_width",
        "epochs",
        "batch_size",
        "max_length",
        "encoder_lr",
        "head_lr",
        "weight_decay",
    }
    if not grid or set(grid) - allowed:
        raise ValueError("Choose training settings; seed and n_splits stay fixed")
    if max_trials < 1 or any(isinstance(v, (str, bytes)) or not v for v in grid.values()):
        raise ValueError("Each grid axis needs a nonempty sequence of choices")
    count = math.prod(len(values) for values in grid.values())
    if count > max_trials:
        raise ValueError(f"Grid requests {count} trials; explicit budget is {max_trials}")
    trials = [
        replace(base, **dict(zip(grid, values, strict=True))) for values in product(*grid.values())
    ]
    if len({digest(asdict(config)) for config in trials}) != len(trials):
        raise ValueError("Duplicate configurations in grid")
    return trials


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_hashes(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def _completed(
    folder: Path, spec: dict[str, Any], protocol_hash: str, heads: tuple[str, ...]
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    report_file, predictions_file = folder / "report.json", folder / "oof.jsonl"
    if not report_file.exists() or not predictions_file.exists():
        return None
    report = json.loads(report_file.read_text())
    if report["protocol_sha256"] != protocol_hash or report["fold"] != spec:
        raise ValueError("Saved fold protocol mismatch")
    if report.get("checkpoint_reload_verified") is not True or (
        set(report.get("checkpoint_hashes", {})) != set(heads)
        or set(report.get("checkpoint_paths", {})) != set(heads)
    ):
        raise ValueError("Saved fold is missing verified model checkpoints")
    if report["oof_sha256"] != hashlib.sha256(predictions_file.read_bytes()).hexdigest():
        raise ValueError("Saved held-out predictions changed")
    for head, hashes in report["checkpoint_hashes"].items():
        if not hashes or hashes != _file_hashes(folder / report["checkpoint_paths"][head]):
            raise ValueError("Saved checkpoint files changed or are missing")
    records = [json.loads(line) for line in predictions_file.read_text().splitlines()]
    if [r["source_id"] for r in records] != spec["test_source_ids"]:
        raise ValueError("Saved fold prediction membership mismatch")
    return report, records


def run_cross_validation(
    examples: Sequence[DecisionExample],
    output: Path,
    *,
    config: CVConfig | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit each model using only the training side. Held-out folds never select epochs.

    The original development pool has already been inspected. These folds support
    development comparisons; selecting settings later requires inner folds or a new test.
    """
    config = CVConfig() if config is None else config
    pool, folds, protocol = make_folds(examples, n_splits=config.n_splits, seed=config.seed)
    model_name, revision = BACKBONES[config.backbone]
    package = Path(__file__).parent
    protocol.update(
        {
            "settings": asdict(config),
            "encoder": model_name,
            "revision": revision,
            "architecture": "joint state/question/all-options; candidate-token mean pooling",
            "selection": "fixed settings; no held-out-fold early stopping or tuning",
            "device": device,
            "versions": {p: version(p) for p in ["torch", "transformers", "scikit-learn", "numpy"]},
            "code_sha256": {
                str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(package.rglob("*.py"))
            },
        }
    )
    # Canonicalize tuples for saved JSON equality and make changed settings a distinct run.
    protocol = json.loads(json.dumps(protocol))
    protocol_hash = digest(protocol)
    folder = output / f"scenario-cv-{protocol_hash[:16]}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Existing protocol differs")
    _write(manifest, protocol)  # Persist memberships and settings BEFORE any fitting.
    all_records, reports = [], []
    for spec, (left, right) in zip(protocol["folds"], folds, strict=True):
        fold = spec["fold"]
        fold_folder = folder / f"fold-{fold}"
        fold_folder.mkdir(exist_ok=True)
        resumed = _completed(fold_folder, spec, protocol_hash, config.heads)
        if resumed:
            report, records = resumed
            # Recompute instead of trusting old accuracy/loss fields in a report.
            test = [pool[i] for i in right]
            expected_models = {"uniform", "option_prior", "linear_candidate"} | {
                f"transformer_{head}" for head in config.heads
            }
            if any(set(r["probabilities"]) != expected_models for r in records):
                raise ValueError("Saved fold model coverage differs from protocol")
            recomputed = {
                name: evaluate_by_street(
                    test,
                    prediction_lookup(
                        test, [r["probabilities"][name] for r in records]
                    ).__getitem__,
                )
                for name in sorted(expected_models)
            }
            if report["metrics"] != recomputed:
                raise ValueError("Saved fold metrics disagree with held-out probabilities")
            print(f"Fold {fold}/{config.n_splits}: reused verified completed fold", flush=True)
        else:
            print(
                f"Fold {fold}/{config.n_splits}: fitting {len(left)}; holding out {len(right)}",
                flush=True,
            )
            fit, test = [pool[i] for i in left], [pool[i] for i in right]
            seed = config.seed + fold
            prior = OptionPrior().fit(fit)
            linear = LinearCandidateScorer(seed=seed).fit(fit)
            values = {
                "uniform": [probabilities(e.inputs, uniform) for e in test],
                "option_prior": [probabilities(e.inputs, prior.predict) for e in test],
                "linear_candidate": [probabilities(e.inputs, linear.predict) for e in test],
            }
            histories, checkpoint_paths, checkpoint_hashes = {}, {}, {}
            for head in config.heads:
                seed_everything(seed)
                tokenizer, model = load_pretrained(
                    model_name, revision, head_type=head, head_width=config.head_width
                )
                model = model.to(device)
                budget = min(config.max_length, int(model.encoder.config.max_position_embeddings))
                encoded_fit = encode_examples(fit, tokenizer, max_length=budget)
                encoded_test = encode_examples(test, tokenizer, max_length=budget)
                optimizer = decision_optimizer(
                    model,
                    encoder_lr=config.encoder_lr,
                    head_lr=config.head_lr,
                    weight_decay=config.weight_decay,
                )
                history = []
                for epoch in range(config.epochs):
                    loss = train_epoch(
                        model,
                        encoded_fit,
                        [e.target_index for e in fit],
                        pad_token_id=tokenizer.pad_token_id,
                        optimizer=optimizer,
                        batch_size=config.batch_size,
                        seed=seed + epoch,
                    )
                    history.append({"epoch": epoch + 1, "training_loss": loss})
                    print(
                        f"Fold {fold}/{config.n_splits} {config.backbone}/{head} "
                        f"epoch {epoch + 1}/{config.epochs}: training loss={loss:.6f}",
                        flush=True,
                    )
                name = f"transformer_{head}"
                values[name] = predict_encoded(
                    model, encoded_test, pad_token_id=tokenizer.pad_token_id
                )
                # Only train histories exist until the fixed training budget completes.
                histories[name] = history
                checkpoint = fold_folder / f"checkpoint-{head}"
                save_checkpoint(
                    checkpoint,
                    model,
                    tokenizer,
                    {
                        "protocol_sha256": protocol_hash,
                        "fold": fold,
                        "seed": seed,
                        "backbone": model_name,
                        "revision": revision,
                        "max_length": budget,
                        "settings": asdict(config),
                        "train_source_ids": spec["train_source_ids"],
                        "test_source_ids": spec["test_source_ids"],
                    },
                )
                loaded_tokenizer, restored, saved = load_checkpoint(checkpoint)
                restored = restored.to(device)
                checked = predict_encoded(
                    restored,
                    encode_examples(test[:8], loaded_tokenizer, max_length=budget),
                    pad_token_id=loaded_tokenizer.pad_token_id,
                )
                if saved["protocol_sha256"] != protocol_hash or any(
                    not np.allclose(a, b, atol=1e-5, rtol=1e-4)
                    for a, b in zip(checked, values[name][:8], strict=True)
                ):
                    raise ValueError("Checkpoint round-trip verification failed")
                checkpoint_paths[head] = checkpoint.name
                checkpoint_hashes[head] = _file_hashes(checkpoint)
                del model, optimizer, restored
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()
            records = [
                {
                    "source_id": e.source_id,
                    "fold": fold,
                    "group": scenario_group(e),
                    "target_index": e.target_index,
                    "probabilities": {name: p[i] for name, p in values.items()},
                }
                for i, e in enumerate(test)
            ]
            predictors = {
                name: prediction_lookup(test, p).__getitem__ for name, p in values.items()
            }
            metrics = {
                name: evaluate_by_street(test, predict) for name, predict in predictors.items()
            }
            predictions_path = fold_folder / "oof.jsonl"
            predictions_path.write_text(
                "\n".join(json.dumps(r, allow_nan=False) for r in records) + "\n"
            )
            report = {
                "protocol_sha256": protocol_hash,
                "fold": spec,
                "metrics": metrics,
                "training_history": histories,
                "checkpoint_reload_verified": True,
                "checkpoint_paths": checkpoint_paths,
                "checkpoint_hashes": checkpoint_hashes,
                "oof_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            }
            _write(fold_folder / "report.json", report)  # Complete marker is always written last.
        reports.append(report)
        all_records.extend(records)
    summary = _summarize(pool, all_records, reports, config)
    summary["protocol_sha256"] = protocol_hash
    summary["output_directory"] = str(folder)
    _write(folder / "summary.json", summary)
    return summary


def _summarize(
    pool: list[DecisionExample],
    records: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    config: CVConfig,
) -> dict[str, Any]:
    by_id = {r["source_id"]: r for r in records}
    if (
        len(records) != len(pool)
        or len(by_id) != len(pool)
        or set(by_id) != {e.source_id for e in pool}
    ):
        raise ValueError("Each example needs exactly one held-out prediction per model")
    for e in pool:
        r = by_id[e.source_id]
        if r["target_index"] != e.target_index or r["group"] != scenario_group(e):
            raise ValueError("Held-out prediction metadata differs from source example")
    summary, by_street, diagnostics, errors, caches = {}, {}, {}, {}, {}
    for name in records[0]["probabilities"]:
        predictions = [by_id[e.source_id]["probabilities"][name] for e in pool]
        lookup = prediction_lookup(pool, predictions)
        caches[name] = lookup
        metrics = evaluate_by_street(pool, lookup.__getitem__)
        fold_metrics = [r["metrics"][name]["overall"] for r in reports]
        weighted = sum(m["n"] * m["log_loss"] for m in fold_metrics) / len(pool)
        if not math.isclose(weighted, metrics["overall"]["log_loss"], abs_tol=1e-10):
            raise ValueError("Pooled and weighted fold loss disagree")
        summary[name] = {
            **metrics["overall"],
            "fold_log_loss_mean": float(np.mean([m["log_loss"] for m in fold_metrics])),
            "fold_log_loss_sd": float(np.std([m["log_loss"] for m in fold_metrics], ddof=1)),
        }
        by_street[name] = metrics
        diagnostics[name] = prediction_diagnostics(pool, lookup.__getitem__)
        errors[name] = inspect_mistakes(pool, lookup.__getitem__, seed=config.seed)
    intervals = {
        name: paired_comparison(
            pool,
            caches[name].__getitem__,
            caches["linear_candidate"].__getitem__,
            groups=[scenario_group(e) for e in pool],
            seed=config.seed,
        )
        for name in caches
        if name.startswith("transformer_")
    }
    return {
        "models": summary,
        "by_street": by_street,
        "diagnostics": diagnostics,
        "five_mistakes": errors,
        "paired_vs_linear": intervals,
        "coverage": {"n": len(pool), "held_out_predictions_per_row": 1, "folds": len(reports)},
        "interpretation": (
            "Development CV on an already inspected pool. Fold SD is descriptive, not a CI. "
            "Grouped intervals condition on fitted folds. "
            "Subsequent tuning needs inner folds or a new test."
        ),
    }
