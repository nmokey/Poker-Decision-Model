"""Shared decision-model training, inference, and checkpoint workflows."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import random
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import functional as F

from poker_decision.data import (
    DecisionExample,
    DecisionInputs,
)
from poker_decision.evaluation import evaluate_by_street
from poker_decision.models.encoder import (
    BACKBONES,
    QUESTION,
    EncodedDecision,
    PokerDecisionModel,
    collate_decisions,
    encode_inputs,
    load_pretrained,
)
from poker_decision.pipeline import write_json, write_jsonl
from poker_decision.prepared import load_prepared as load_prepared

if TYPE_CHECKING:
    from poker_decision.cross_validation import CVConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_examples(
    examples: Sequence[DecisionExample],
    tokenizer: Any,
    *,
    max_length: int,
    question: str = QUESTION,
) -> list[EncodedDecision]:
    rows, failures = [], []
    for e in examples:
        try:
            rows.append(
                encode_inputs(e.inputs, tokenizer, max_length=max_length, question=question)
            )
        except ValueError as error:
            failures.append(f"{e.source_id}: {error}")
    if failures:
        raise ValueError(
            f"{len(failures)} tokenization failures; no rows silently removed. "
            + " | ".join(failures[:5])
        )
    return rows


def batch_on_device(
    rows: Sequence[EncodedDecision], pad_token_id: int, device: torch.device
) -> dict[str, Tensor]:
    return {k: v.to(device) for k, v in collate_decisions(rows, pad_token_id=pad_token_id).items()}


def train_epoch(
    model: PokerDecisionModel,
    encoded: Sequence[EncodedDecision],
    targets: Sequence[int],
    *,
    pad_token_id: int,
    optimizer: Any,
    batch_size: int = 16,
    seed: int = 42,
) -> float:
    if not encoded or len(encoded) != len(targets) or batch_size < 1:
        raise ValueError("Need matching nonempty examples/targets and a positive batch size")
    if any(not 0 <= t < len(e.candidate_spans) for e, t in zip(encoded, targets, strict=True)):
        raise ValueError("Target outside actual candidate set")
    model.train()
    device = next(model.parameters()).device
    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    total = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = batch_on_device([encoded[i] for i in indices], pad_token_id, device)
        labels = torch.tensor([targets[i] for i in indices], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(**batch), labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / len(encoded)


def predict_encoded(
    model: PokerDecisionModel,
    encoded: Sequence[EncodedDecision],
    *,
    pad_token_id: int,
    batch_size: int = 32,
) -> list[list[float]]:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            rows = encoded[start : start + batch_size]
            logits = model(**batch_on_device(rows, pad_token_id, device))
            values = logits.float().softmax(-1).cpu().tolist()
            for e, p in zip(rows, values, strict=True):
                # Python double precision matches the shared metric normalization contract.
                actual = p[: len(e.candidate_spans)]
                predictions.append([v / sum(actual) for v in actual])
    return predictions


def prediction_lookup(
    examples: Sequence[DecisionExample], predictions: Sequence[list[float]]
) -> dict[DecisionInputs, list[float]]:
    """Adapt batched predictions to the existing answer-free metric interface."""
    return {e.inputs: p for e, p in zip(examples, predictions, strict=True)}


def score_model(
    model: PokerDecisionModel,
    examples: Sequence[DecisionExample],
    encoded: Sequence[EncodedDecision],
    *,
    pad_token_id: int,
) -> dict[str, dict[str, float]]:
    predictions = predict_encoded(model, encoded, pad_token_id=pad_token_id)
    return evaluate_by_street(examples, prediction_lookup(examples, predictions).__getitem__)


def save_checkpoint(
    path: Path, model: PokerDecisionModel, tokenizer: Any, metadata: dict[str, Any]
) -> None:
    """Save a self-contained encoder/tokenizer plus head weights and provenance."""
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
        "head_type": model.head_type,
        "head_width": model.head_width,
    }
    model.encoder.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "tokenizer")
    torch.save(model.head.state_dict(), path / "head.pt")
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def load_checkpoint(path: Path) -> tuple[Any, PokerDecisionModel, dict[str, Any]]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        path / "tokenizer", local_files_only=True
    )
    metadata = json.loads((path / "metadata.json").read_text())
    model = PokerDecisionModel(
        AutoModel.from_pretrained(path / "encoder", local_files_only=True),
        head_type=metadata.get("head_type", "linear"),
        head_width=metadata.get("head_width", 64),
    )
    model.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
    model.eval()
    return tokenizer, model, metadata


def decision_optimizer(
    model: PokerDecisionModel, *, encoder_lr: float, head_lr: float, weight_decay: float
) -> torch.optim.AdamW:
    """Use the same visible optimizer settings for holdout and scenario fits."""
    if not all(math.isfinite(v) and v > 0 for v in (encoder_lr, head_lr)):
        raise ValueError("Learning rates must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Invalid weight decay")
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def require_cuda() -> None:
    """Guard pretrained fits without changing accelerator state in offline tests."""
    if not torch.cuda.is_available():
        raise RuntimeError("Pretrained-model training requires a CUDA GPU runtime.")


def learning_check(
    examples: Sequence[DecisionExample],
    output: Path,
    *,
    config: CVConfig,
    head: str,
    n: int = 32,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Train a disposable model on distinct training contexts; never reuse its weights.

    Callers supply only their training partition (first-fold training for CV).
    The check does not choose settings or inspect held-out labels.
    """
    require_cuda()
    if min(n, epochs, batch_size) < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Tiny-check counts and learning rate must be positive")
    seen: set[str] = set()
    tiny = []
    for row in random.Random(config.seed).sample(list(examples), len(examples)):
        if len(row.inputs.candidates) > 1 and row.inputs.context not in seen:
            tiny.append(row)
            seen.add(row.inputs.context)
        if len(tiny) == n:
            break
    if len(tiny) != n:
        raise ValueError(f"Need {n} distinct multi-option training contexts")
    seed_everything(config.seed)
    tokenizer, model = load_pretrained(
        *BACKBONES[config.backbone], head_type=head, head_width=config.head_width
    )
    model = model.to("cuda")
    budget = min(config.max_length, int(model.encoder.config.max_position_embeddings))
    encoded = encode_examples(tiny, tokenizer, max_length=budget)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    history = []
    try:
        for epoch in range(epochs):
            loss = train_epoch(
                model,
                encoded,
                [e.target_index for e in tiny],
                pad_token_id=tokenizer.pad_token_id,
                optimizer=optimizer,
                batch_size=batch_size,
                seed=config.seed + epoch,
            )
            metrics = score_model(model, tiny, encoded, pad_token_id=tokenizer.pad_token_id)[
                "overall"
            ]
            history.append({"epoch": epoch + 1, "training_loss": loss, **metrics})
            if metrics["exact_accuracy"] == 1:
                break
        result = {
            "source_ids": [e.source_id for e in tiny],
            "history": history,
            "settings": {
                "backbone": config.backbone,
                "head": head,
                "head_width": config.head_width,
                "max_length": budget,
                "seed": config.seed,
                "n": n,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "required_accuracy": 0.95,
            },
        }
        write_json(output, result)
        if history[-1]["exact_accuracy"] < 0.95:
            raise RuntimeError(f"{config.backbone}/{head} failed training-only learning check")
        return result
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def fit_holdout(
    root: Path,
    output: Path,
    train: list[DecisionExample],
    validation: list[DecisionExample],
    manifest: dict[str, Any],
    *,
    config: CVConfig,
) -> dict[str, Any]:
    """Fit a fresh decision model and select its checkpoint on validation loss.

    This introductory protocol is development selection, separate from fixed-epoch
    scenario CV. A new output directory is required to avoid overwriting evidence.
    """
    require_cuda()
    if manifest.get("official_test_used") is not False:
        raise ValueError("Expected a training-only PokerBench manifest")
    if len(config.heads) != 1:
        raise ValueError("Holdout fitting needs exactly one head")
    if (
        not train
        or not validation
        or not {e.inputs.context for e in train}.isdisjoint(e.inputs.context for e in validation)
    ):
        raise ValueError("Need nonempty context-disjoint training and validation sets")
    if len({e.source_id for e in train + validation}) != len(train) + len(validation):
        raise ValueError("Repeated source IDs")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "data_manifest.json", manifest)
    for name, rows in (("train", train), ("validation", validation)):
        write_jsonl(output / f"{name}.jsonl", [e.to_record() for e in rows])
    seed_everything(config.seed)
    tokenizer, model = load_pretrained(
        *BACKBONES[config.backbone], head_type=config.heads[0], head_width=config.head_width
    )
    model = model.to("cuda")
    budget = min(config.max_length, int(model.encoder.config.max_position_embeddings))
    encoded_train = encode_examples(train, tokenizer, max_length=budget)
    encoded_validation = encode_examples(validation, tokenizer, max_length=budget)
    optimizer = decision_optimizer(
        model,
        encoder_lr=config.encoder_lr,
        head_lr=config.head_lr,
        weight_decay=config.weight_decay,
    )
    name, revision = BACKBONES[config.backbone]
    metadata = {
        "backbone": name,
        "revision": revision,
        "question": QUESTION,
        "max_length": budget,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "encoder_lr": config.encoder_lr,
        "head_lr": config.head_lr,
        "weight_decay": config.weight_decay,
        "selection": "validation log loss",
        "serialization": "joint state/question/options; candidate-token mean pooling",
        "device": "cuda",
        "hardware": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("torch", "transformers", "numpy", "scikit-learn")},
        "dataset_revision": manifest["dataset_revision"],
        "official_test_used": False,
        "split_sha256": {
            split: hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()
            for split in ("train", "validation")
        },
        "code_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src/poker_decision").rglob("*.py"))
        },
    }
    history = []
    best_loss = float("inf")
    try:
        for epoch in range(config.epochs):
            loss = train_epoch(
                model,
                encoded_train,
                [e.target_index for e in train],
                pad_token_id=tokenizer.pad_token_id,
                optimizer=optimizer,
                batch_size=config.batch_size,
                seed=config.seed + epoch,
            )
            metrics = score_model(
                model, validation, encoded_validation, pad_token_id=tokenizer.pad_token_id
            )["overall"]
            history.append({"epoch": epoch + 1, "train_loss": loss, **metrics})
            write_json(output / "training_history.json", history)
            if metrics["log_loss"] < best_loss:
                best_loss = metrics["log_loss"]
                save_checkpoint(
                    output / "best",
                    model,
                    tokenizer,
                    {
                        **metadata,
                        "best_epoch": epoch + 1,
                        "validation_log_loss": best_loss,
                    },
                )
        return {"output_directory": str(output), "history": history}
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    import argparse
    from datetime import UTC, datetime

    from poker_decision.cross_validation import CVConfig
    from poker_decision.paths import prepared_directory
    from poker_decision.pipeline import prepare

    parser = argparse.ArgumentParser(description="Fit a fresh PokerBench decision scorer")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--backbone", choices=list(BACKBONES), default="bert_mini")
    parser.add_argument("--head", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require_cuda()
    root = args.root.resolve()
    config = CVConfig(backbone=args.backbone, heads=(args.head,), epochs=args.epochs)
    if not (prepared_directory(root) / "manifest.json").exists():
        prepare(root)
    train, validation, manifest = load_prepared(root)
    output = args.output or root / "artifacts/decision_model" / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    # Check before spending compute; the fit also refuses an existing run directory.
    if output.exists():
        raise FileExistsError(output)
    learning_check(
        train, output.with_name(output.name + "-preflight.json"), config=config, head=args.head
    )
    print(json.dumps(fit_holdout(root, output, train, validation, manifest, config=config)))


if __name__ == "__main__":
    main()
