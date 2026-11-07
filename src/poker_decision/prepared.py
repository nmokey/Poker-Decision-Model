"""Audited PokerBench split loading, independent of neural dependencies."""

import hashlib
import json
from pathlib import Path
from typing import Any

from poker_decision.data import (
    DATA_REVISION,
    SHA256,
    TRAIN_FILES,
    DecisionExample,
    DecisionInputs,
    build_examples,
    read_rows,
)
from poker_decision.paths import prepared_directory


def load_prepared(
    root: Path,
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Reuse saved membership, checking provenance and context separation."""
    directory = prepared_directory(root)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["dataset_revision"] != DATA_REVISION or manifest["official_test_used"]:
        raise ValueError("Expected the audited training-only PokerBench preparation")
    current_hash = hashlib.sha256(Path(__file__).with_name("data.py").read_bytes()).hexdigest()
    if manifest["package_code_sha256"]["data.py"] != current_hash:
        raise ValueError("Preprocessing code changed; rerun PokerBench preparation")
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
