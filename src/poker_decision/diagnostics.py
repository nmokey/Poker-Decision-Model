"""Reliability, error structure and grouped uncertainty for reference-action predictions."""

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from poker_decision.data import DecisionExample, action_type
from poker_decision.evaluation import Predictor, predicted_index, probabilities


def grouped_interval(
    values: Sequence[float], groups: Sequence[str], *, seed: int = 42, repetitions: int = 2000
) -> dict[str, float]:
    """Row-weighted mean, resampling whole groups; use paired differences for comparisons."""
    if not values or len(values) != len(groups) or repetitions < 2:
        raise ValueError("Need matching nonempty values/groups and at least two repetitions")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        if not math.isfinite(value):
            raise ValueError("Nonfinite observation")
        grouped[group].append(value)
    stats = np.array([(sum(v), len(v)) for _, v in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        selected = stats[rng.integers(len(stats), size=len(stats))]
        draws.append(float(selected[:, 0].sum() / selected[:, 1].sum()))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean": sum(values) / len(values), "low95": float(low), "high95": float(high)}


def prediction_diagnostics(
    examples: Sequence[DecisionExample], predict: Predictor, *, bins: int = 10
) -> dict[str, Any]:
    if not examples or bins < 1:
        raise ValueError("Need examples and a positive bin count")
    reliability: dict[int, list[tuple[float, int]]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    sizing_errors = type_errors = 0
    for e in examples:
        p = probabilities(e.inputs, predict)
        index = predicted_index(e.inputs, p)
        correct = int(index == e.target_index)
        confidence = max(p)
        reliability[min(int(confidence * bins), bins - 1)].append((confidence, correct))
        truth, predicted = action_type(e.target_action), action_type(e.inputs.candidates[index])
        confusion[truth, predicted] += 1
        if not correct:
            sizing_errors += int(truth == predicted)
            type_errors += int(truth != predicted)
    rows = [
        {
            "bin": b,
            "lower": b / bins,
            "upper": (b + 1) / bins,
            "n": len(values),
            "mean_confidence": sum(v for v, _ in values) / len(values),
            "accuracy": sum(v for _, v in values) / len(values),
        }
        for b, values in sorted(reliability.items())
    ]
    return {
        "n": len(examples),
        "reliability": rows,
        "confusion": [
            {"reference": t, "predicted": p, "n": n} for (t, p), n in sorted(confusion.items())
        ],
        "same_action_type_sizing_errors": sizing_errors,
        "different_action_type_errors": type_errors,
        "ece": sum(r["n"] * abs(r["mean_confidence"] - r["accuracy"]) for r in rows)
        / len(examples),
        "semantics": "Reference-label confidence; not solver strategy frequencies. "
        "Binning dependent.",
    }


def paired_comparison(
    examples: Sequence[DecisionExample],
    left: Predictor,
    right: Predictor,
    *,
    groups: Sequence[str] | None = None,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Left minus right on identical rows; positive accuracy / negative loss favors left."""
    accuracy, loss = [], []
    for e in examples:
        a, b = probabilities(e.inputs, left), probabilities(e.inputs, right)
        accuracy.append(
            float(predicted_index(e.inputs, a) == e.target_index)
            - float(predicted_index(e.inputs, b) == e.target_index)
        )
        loss.append(
            -math.log(max(a[e.target_index], 1e-15)) + math.log(max(b[e.target_index], 1e-15))
        )
    clusters = [e.inputs.context for e in examples] if groups is None else groups
    return {
        "accuracy_difference": grouped_interval(
            accuracy, clusters, repetitions=repetitions, seed=seed
        ),
        "log_loss_difference": grouped_interval(loss, clusters, repetitions=repetitions, seed=seed),
        "groups": len(set(clusters)),
        "scope": "Conditional on these fitted models and chosen clusters; "
        "excludes training uncertainty.",
    }


def menu_information(
    training: Sequence[DecisionExample], *, permutations: int = 199, seed: int = 42
) -> dict[str, Any]:
    """Training-only within-street menu/label association, with a permutation reference."""
    from sklearn.metrics import mutual_info_score

    if not training or permutations < 1:
        raise ValueError("Need training examples and a positive permutation count")
    shuffled = list(training)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    unique = {e.inputs.context: e for e in shuffled}
    by_street: dict[str, list[DecisionExample]] = defaultdict(list)
    for e in unique.values():
        by_street[e.street].append(e)
    encoded = []
    for street in sorted(by_street):
        examples = by_street[street]
        menus = ["|".join(sorted(action_type(c) for c in e.inputs.candidates)) for e in examples]
        targets = [action_type(e.target_action) for e in examples]
        encoded.append((menus, targets))

    def association(shuffle: bool) -> float:
        total = 0.0
        for menus, targets in encoded:
            labels = rng.permutation(targets).tolist() if shuffle else targets
            total += (
                len(targets) / len(unique) * float(mutual_info_score(menus, labels)) / math.log(2)
            )
        return total

    observed = association(False)
    null = [association(True) for _ in range(permutations)]
    return {
        "unique_training_contexts": len(unique),
        "conditional_mi_bits": observed,
        "null_median_bits": float(np.median(null)),
        "permutations": permutations,
        "permutation_p": (1 + sum(v >= observed for v in null)) / (permutations + 1),
        "null_scope": "Within-street row-label shuffling; related-scenario dependence "
        "makes this exploratory, not a formal significance claim.",
        "interpretation": "Menu association includes genuine legal-action constraints, "
        "not proof of leakage.",
    }
