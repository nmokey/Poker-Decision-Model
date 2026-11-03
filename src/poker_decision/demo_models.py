"""Serving catalogue, checkpoint-specific hands and a single resident model."""

import gc
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from poker_decision.data import DecisionExample, DecisionInputs

ENCODERS = {
    "modernbert": ("ModernBERT", "answerdotai/ModernBERT-base"),
    "distilbert": ("DistilBERT", "distilbert/distilbert-base-cased"),
    "minilm": ("MiniLM", "microsoft/MiniLM-L12-H384-uncased"),
    "bert_mini": ("BERT Mini", "google/bert_uncased_L-4_H-256_A-4"),
}
HEADS = {"linear": "Linear", "mlp": "MLP"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    encoder: str
    head: str
    backbone: str
    url: str
    sha256: str
    n_hands: int
    fold: int
    protocol_sha256: str

    @property
    def label(self) -> str:
        return f"{ENCODERS[self.encoder][0]} · {HEADS[self.head]}"


def load_catalog(path: Path) -> tuple[str, dict[str, ModelSpec]]:
    raw = json.loads(path.read_text())
    if raw.get("schema") != 2 or not isinstance(raw.get("models"), dict):
        raise ValueError("Unsupported serving catalogue")
    models = {}
    for key, record in raw["models"].items():
        spec = ModelSpec(
            key=key, **{name: record[name] for name in ModelSpec.__annotations__ if name != "key"}
        )
        if (
            spec.encoder not in ENCODERS
            or spec.head not in HEADS
            or spec.key != f"{spec.encoder}-{spec.head}"
            or spec.backbone != ENCODERS[spec.encoder][1]
            or not spec.url.startswith("https://")
            or len(spec.sha256) != 64
            or any(c not in "0123456789abcdef" for c in spec.sha256)
            or type(spec.n_hands) is not int
            or spec.n_hands < 1
            or type(spec.fold) is not int
            or spec.fold < 1
        ):
            raise ValueError("Invalid serving model specification")
        models[key] = spec
    default = raw.get("default_model")
    if default not in models:
        raise ValueError("Default model is not in the serving catalogue")
    return str(default), models


def checkpoint_examples(directory: Path, spec: ModelSpec) -> tuple[DecisionExample, ...]:
    """Reject mislabeled weights and rows not held out by this exact checkpoint.

    The caller verifies the bundle's file checksums before invoking this function.
    """
    metadata = json.loads((directory / "model/metadata.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    for record in (metadata, manifest):
        if (
            record.get("backbone") != spec.backbone
            or record.get("head_type") != spec.head
            or record.get("fold") != spec.fold
            or record.get("protocol_sha256") != spec.protocol_sha256
        ):
            raise ValueError("Checkpoint does not match the selected encoder/head")
    test_ids = set(metadata["test_source_ids"])
    train_ids = set(metadata["train_source_ids"])
    if test_ids.intersection(train_ids):
        raise ValueError("Checkpoint training and held-out membership overlap")
    examples = []
    for line in (directory / "hands.jsonl").read_text().splitlines():
        row = json.loads(line)
        if (
            not isinstance(row["context"], str)
            or not isinstance(row["candidates"], list)
            or any(not isinstance(c, str) for c in row["candidates"])
            or type(row["target_index"]) is not int
        ):
            raise ValueError("Invalid stored demo hand")
        if row["source_id"] not in test_ids or row["source_id"] in train_ids:
            raise ValueError("Demo hand was not held out by the selected checkpoint")
        example = DecisionExample(
            row["source_id"],
            row["street"],
            DecisionInputs(row["context"], tuple(row["candidates"])),
            row["target_index"],
        )
        if len(example.inputs.candidates) < 2:
            raise ValueError("Expected a multi-option demo hand")
        examples.append(example)
    if (
        len(examples) != spec.n_hands
        or len({e.source_id for e in examples}) != len(examples)
        or manifest.get("n_hands") != spec.n_hands
    ):
        raise ValueError("Incorrect or duplicate demo hand membership")
    return tuple(examples)


class SingleModelStore[T]:
    """Serialize load/inference and release the previous model before replacing it.

    All app sessions share this store. Operations return predictions, never model
    references, so idle sessions cannot retain eight large encoders in memory.
    """

    def __init__(self, loader: Callable[[str], T]) -> None:
        self._loader = loader
        self._lock = threading.RLock()
        self._key: str | None = None
        self._model: T | None = None

    def apply[R](self, key: str, operation: Callable[[T], R]) -> R:
        with self._lock:
            if key != self._key:
                self._model = None
                self._key = None
                gc.collect()
                self._model = self._loader(key)
                self._key = key
            if self._model is None:
                raise RuntimeError("The selected model was not loaded")
            return operation(self._model)

    def warm(self, key: str) -> None:
        self.apply(key, lambda _: None)
