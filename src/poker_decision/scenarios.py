"""Answer-free scenario families and deterministic group-aware evaluation protocols."""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from poker_decision.data import DecisionExample, action_type


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scenario_group(example: DecisionExample) -> str:
    """Group preflop histories or postflop flop ranks; never inspect the label."""
    if example.street == "preflop":
        match = re.search(r"Previous actions: (.*?)\. Source num_players:", example.inputs.context)
        if match is None:
            raise ValueError(f"Cannot identify preflop history: {example.source_id}")
        return "preflop-history:" + match[1]
    match = re.search(r"Board: (.*?)\. Preflop actions:", example.inputs.context)
    cards = match[1].split()[:3] if match else []
    if len(cards) != 3 or any(not re.fullmatch(r"[2-9TJQKA][cdhs]", c) for c in cards):
        raise ValueError(f"Cannot identify flop rank family: {example.source_id}")
    return "postflop-flop-rank-family:" + "".join(sorted(c[0] for c in cards))


def scenario_overlap(
    train: Sequence[DecisionExample], validation: Sequence[DecisionExample]
) -> dict[str, Any]:
    left = {scenario_group(e) for e in train}
    right = [scenario_group(e) for e in validation]
    if not right:
        raise ValueError("Empty validation set")
    return {
        "training_groups": len(left),
        "validation_groups": len(set(right)),
        "shared_groups": len(left & set(right)),
        "validation_fraction_with_training_group": sum(g in left for g in right) / len(right),
        "exact_context_overlap": len(
            {e.inputs.context for e in train} & {e.inputs.context for e in validation}
        ),
    }


def make_folds(
    examples: Sequence[DecisionExample], *, n_splits: int = 5, seed: int = 42
) -> tuple[list[DecisionExample], list[tuple[list[int], list[int]]], dict[str, Any]]:
    """Stratify by street/label type under scenario-group constraints; save all memberships."""
    from sklearn.model_selection import StratifiedGroupKFold

    pool = sorted(examples, key=lambda e: e.source_id)
    if any(re.fullmatch(r"(?:preflop|postflop)-test-\d+", e.source_id) for e in pool):
        raise ValueError("Official test examples must not enter development cross-validation")
    if n_splits < 2 or not pool or len({e.source_id for e in pool}) != len(pool):
        raise ValueError("Need unique examples and at least two folds")
    groups = np.array([scenario_group(e) for e in pool])
    if len(set(groups)) < n_splits:
        raise ValueError("Fewer scenario groups than folds")
    # Exact contexts cannot be separated just because street metadata differs.
    contexts: dict[str, str] = {}
    for e, group in zip(pool, groups, strict=True):
        if contexts.setdefault(e.inputs.context, str(group)) != group:
            raise ValueError("Identical context maps to conflicting scenario groups")
    strata = [e.street + "|" + action_type(e.target_action) for e in pool]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds, memberships = [], []
    seen = np.zeros(len(pool), dtype=int)
    for fold, (left, right) in enumerate(splitter.split(np.zeros(len(pool)), strata, groups), 1):
        if not len(left) or not len(right) or set(groups[left]) & set(groups[right]):
            raise ValueError("Empty fold or overlapping scenario groups")
        fit, test = [pool[i] for i in left], [pool[i] for i in right]
        if {e.inputs.context for e in fit} & {e.inputs.context for e in test}:
            raise ValueError("Exact context crosses fold boundary")
        seen[right] += 1
        folds.append((left.tolist(), right.tolist()))
        memberships.append(
            {
                "fold": fold,
                "train_source_ids": [e.source_id for e in fit],
                "test_source_ids": [e.source_id for e in test],
                "train_n": len(fit),
                "test_n": len(test),
                "test_streets": dict(Counter(e.street for e in test)),
                "scenario_overlap": 0,
                "exact_context_overlap": 0,
            }
        )
    if not np.all(seen == 1):
        raise ValueError("Every row must be held out exactly once")
    protocol = {
        "schema": "poker-scenario-cv-v1",
        "seed": seed,
        "n_splits": n_splits,
        "grouping": "preflop full betting history; postflop sorted first-three-board ranks",
        "stratification": "street + reference action type, approximately under group constraints",
        "pool_sha256": digest([e.to_record() for e in pool]),
        "groups_sha256": digest(groups.tolist()),
        "pool_n": len(pool),
        "folds": memberships,
        "official_test_used": (
            False
            if all(re.fullmatch(r"(?:preflop|postflop)-train-\d+", e.source_id) for e in pool)
            else None
        ),
        "provenance_scope": "Caller-provided pool; canonical notebooks verify "
        "the saved training manifest",
    }
    return pool, folds, protocol
