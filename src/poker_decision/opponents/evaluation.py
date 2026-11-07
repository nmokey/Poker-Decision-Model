"""Paired opponent-cluster uncertainty and descriptive forecasting diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from poker_decision.opponents.data import ACTIONS
from poker_decision.opponents.models import targets, validate_probabilities


def per_row_metrics(rows: list[dict[str, Any]], p: np.ndarray) -> dict[str, np.ndarray]:
    validate_probabilities(rows, p)
    y = targets(rows)
    # Match the package evaluator's lexical action tie rule; report tie expectation too.
    chosen = np.asarray(
        [
            min(r["inputs"]["legal_action_ids"], key=lambda a: (-v[a], ACTIONS[a]))
            for r, v in zip(rows, p, strict=True)
        ]
    )
    ties = p == p.max(axis=1, keepdims=True)
    return dict(
        log_loss=-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)),
        accuracy=(chosen == y).astype(float),
        tie_expected_accuracy=ties[np.arange(len(y)), y] / ties.sum(axis=1),
        brier=((p - np.eye(3)[y]) ** 2).sum(axis=1),
    )


def aggregate(rows: list[dict[str, Any]], metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    if not rows:
        return dict(n=0, sessions=0, opponents=0, descriptive_only=True)
    sessions = sorted({r["session_id"] for r in rows})
    opponents = {r["opponent_id"] for r in rows}
    return dict(
        n=len(rows),
        sessions=len(sessions),
        opponents=len(opponents),
        descriptive_only=len(opponents) < 5 or len(rows) < 30,
        **{k: float(v.mean()) for k, v in metrics.items()},
        session_macro_log_loss=float(
            np.mean(
                [
                    metrics["log_loss"][
                        [i for i, r in enumerate(rows) if r["session_id"] == s]
                    ].mean()
                    for s in sessions
                ]
            )
        ),
    )


def paired_interval(
    rows: list[dict[str, Any]], delta: np.ndarray, *, seed: int = 42, repetitions: int = 1000
) -> dict[str, Any]:
    if not rows or len(rows) != len(delta) or repetitions < 1:
        raise ValueError("Need aligned paired values and positive bootstrap repetitions")
    identities = sorted({r["opponent_id"] for r in rows})
    groups = [[i for i, r in enumerate(rows) if r["opponent_id"] == oid] for oid in identities]
    sums = np.array([delta[idx].sum() for idx in groups])
    counts = np.array([len(idx) for idx in groups])
    draws = np.random.default_rng(seed).integers(len(groups), size=(repetitions, len(groups)))
    sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(
        estimate=float(delta.mean()),
        interval_95=np.quantile(sampled, [0.025, 0.975]).tolist(),
        opponents=len(groups),
        repetitions=repetitions,
        descriptive_only=len(groups) < 5,
        scope=(
            "Paired whole-opponent bootstrap, conditional on simulated population and fitted models"
        ),
    )


def evaluate_predictions(
    rows: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    *,
    seed: int = 42,
    repetitions: int = 1000,
) -> dict[str, Any]:
    if not rows or "no_history" not in predictions:
        raise ValueError("Need test rows and a no_history reference")
    specs = {s["session_id"]: s for s in sessions}
    values = {k: per_row_metrics(rows, p) for k, p in predictions.items()}
    prior = []
    seen: dict[str, int] = {}
    for r in rows:
        sid = r["session_id"]
        prior.append(seen.get(sid, 0))
        seen[sid] = seen.get(sid, 0) + 1
    slices = {"all": list(range(len(rows)))}
    for lo, hi, label in (
        (0, 0, "0"),
        (1, 5, "1-5"),
        (6, 20, "6-20"),
        (21, 63, "21-63"),
        (64, float("inf"), "64+"),
    ):
        slices["prior_observed_decisions/" + label] = [
            i for i, n in enumerate(prior) if lo <= n <= hi
        ]
    for profile in sorted({s["profile"] for s in sessions}):
        slices["profile/" + profile] = [
            i for i, r in enumerate(rows) if specs[r["session_id"]]["profile"] == profile
        ]
    for switch_lo, switch_hi, label in (
        (-float("inf"), -1, "before"),
        (0, 4, "first_5_hands"),
        (5, 19, "hands_5_to_19"),
        (20, float("inf"), "hands_20_plus"),
    ):
        slices["switch/" + label] = [
            i
            for i, r in enumerate(rows)
            if specs[r["session_id"]]["switch_hand"] is not None
            and switch_lo <= r["hand_index"] - specs[r["session_id"]]["switch_hand"] <= switch_hi
        ]
    diagnostics = []
    for label, indices in slices.items():
        group = [rows[i] for i in indices]
        for name, metrics in values.items():
            diagnostics.append(
                dict(
                    slice=label,
                    model=name,
                    **aggregate(group, {k: v[indices] for k, v in metrics.items()}),
                )
            )
    comparisons = {}
    for name, metrics in values.items():
        if name != "no_history":
            comparisons[name] = {
                metric: paired_interval(
                    rows,
                    metrics[metric] - values["no_history"][metric],
                    seed=seed,
                    repetitions=repetitions,
                )
                for metric in ("log_loss", "accuracy")
            }
    # Include Bayesian vs strongest validation-selected summary without test-based selection.
    if "bayesian" in values and "history_selected" in values:
        comparisons["bayesian_minus_history_selected"] = {
            "log_loss": paired_interval(
                rows,
                values["bayesian"]["log_loss"] - values["history_selected"]["log_loss"],
                seed=seed,
                repetitions=repetitions,
            )
        }
    example_ids = [0]
    if "history_selected" in values:
        gain = values["no_history"]["log_loss"] - values["history_selected"]["log_loss"]
        example_ids.extend([int(gain.argmax()), int(gain.argmin())])
    example_ids.extend(np.random.default_rng(seed).permutation(len(rows)).tolist())
    unique = list(dict.fromkeys(example_ids))[:5]
    examples = [
        dict(
            session_id=rows[i]["session_id"],
            hand_index=rows[i]["hand_index"],
            inputs=rows[i]["inputs"],
            target_index=rows[i]["target_index"],
            probabilities={
                name: p[i, rows[i]["inputs"]["legal_action_ids"]].tolist()
                for name, p in predictions.items()
            },
        )
        for i in unique
    ]
    return dict(
        overall={k: aggregate(rows, v) for k, v in values.items()},
        diagnostics=diagnostics,
        comparisons=comparisons,
        examples=examples,
        slice_note=(
            "History bins count earlier observed actions, including this hand. "
            "These are descriptive slices, not controlled amounts of history."
        ),
    )
