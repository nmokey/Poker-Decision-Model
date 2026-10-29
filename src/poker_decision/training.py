"""Training helpers shared by the Meeting 2 walkthrough and offline tests."""

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from poker_decision.data import (
    DATA_REVISION,
    SHA256,
    TRAIN_FILES,
    DecisionExample,
    DecisionInputs,
    build_examples,
    read_rows,
)
from poker_decision.evaluation import evaluate_by_street
from poker_decision.models.encoder import (
    EncodedDecision,
    PokerDecisionModel,
    collate_decisions,
    encode_inputs,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prepared(
    root: Path,
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Reuse Meeting 1 membership, checking provenance and context separation."""
    directory = root / "data/processed/pokerbench_meeting1"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["dataset_revision"] != DATA_REVISION or manifest["official_test_used"]:
        raise ValueError("Expected the audited training-only PokerBench preparation")
    current_hash = hashlib.sha256(Path(__file__).with_name("data.py").read_bytes()).hexdigest()
    if manifest["package_code_sha256"]["data.py"] != current_hash:
        raise ValueError("Preprocessing code changed; rerun Meeting 1 preparation")
    for kind, source in manifest["sources"].items():
        if kind not in SHA256 or source["sha256"] != SHA256[kind]:
            raise ValueError("Source provenance does not match pinned training files")
    splits = []
    for split in ("train", "validation"):
        if "split_sha256" in manifest and hashlib.sha256(
            (directory / f"{split}.jsonl").read_bytes()
        ).hexdigest() != manifest["split_sha256"].get(split):
            raise ValueError(f"{split} file content/membership checksum mismatch")
        rows = [
            json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines()
        ]
        examples = [
            DecisionExample(
                r["source_id"],
                r["street"],
                DecisionInputs(r["context"], tuple(r["candidates"])),
                r["target_index"],
            )
            for r in rows
        ]
        if not examples or [e.source_id for e in examples] != manifest[f"{split}_source_ids"]:
            raise ValueError(f"{split} examples do not match saved membership")
        splits.append(examples)
    train, validation = splits
    if not {e.inputs.context for e in train}.isdisjoint(e.inputs.context for e in validation):
        raise ValueError("Training and validation contexts overlap")
    ids = [e.source_id for e in train + validation]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated source IDs")
    if "split_sha256" not in manifest:
        # Older manifests have no content hashes. Check against pinned cached raw
        # rows before trusting targets, contexts, or menus; never silently resplit.
        actual = {e.source_id: e.to_record() for e in train + validation}
        verified: dict[str, dict[str, object]] = {}
        for kind, filename in TRAIN_FILES.items():
            if kind not in manifest["sources"]:
                continue
            raw = root / "data/raw/pokerbench" / filename
            if not raw.exists() or hashlib.sha256(raw.read_bytes()).hexdigest() != SHA256[kind]:
                raise ValueError("Legacy split requires pinned raw files; rerun preparation")
            selected_rows = []
            selected_ids = []
            for i, row in enumerate(read_rows(raw)):
                source_id = f"{kind}-train-{i}"
                if source_id in actual:
                    selected_rows.append(row)
                    selected_ids.append(source_id)
            parsed, failures = build_examples(selected_rows, kind)
            if failures:
                raise ValueError("Legacy source rows fail validation")
            for source_id, example in zip(selected_ids, parsed, strict=True):
                verified[source_id] = {**example.to_record(), "source_id": source_id}
        if verified != actual:
            raise ValueError("Legacy split contents differ from pinned source rows")
    return train, validation, manifest


def encode_examples(
    examples: Sequence[DecisionExample], tokenizer: Any, *, max_length: int
) -> list[EncodedDecision]:
    rows, failures = [], []
    for e in examples:
        try:
            rows.append(encode_inputs(e.inputs, tokenizer, max_length=max_length))
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
