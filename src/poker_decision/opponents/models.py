"""Matched legal-action forecasters and an online Bayesian residual-style filter."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from poker_decision.opponents.data import canonical


def targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([r["inputs"]["legal_action_ids"][r["target_index"]] for r in rows], dtype=int)


def legal_mask(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[a in r["inputs"]["legal_action_ids"] for a in range(3)] for r in rows])


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if logits.shape != mask.shape or not mask.any(axis=-1).all() or not np.isfinite(logits).all():
        raise ValueError("Invalid logits or legal mask")
    values = np.where(mask, logits, -np.inf)
    p = np.exp(values - values.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def validate_probabilities(rows: list[dict[str, Any]], p: np.ndarray) -> None:
    mask = legal_mask(rows)
    if p.shape != mask.shape or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid prediction shape/values")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-8) or np.any(p[~mask] != 0):
        raise ValueError("Prediction must normalize over exactly the legal menu")


class Forecaster:
    """Portable numeric model; optional sklearn/scipy imports happen only during fitting.

    masked_logistic optimizes cross-entropy over legal actions during training.
    menu_logistic retains separate sklearn classifiers for each legal menu.
    naive_bayes uses discretized feature tokens with a train-only vocabulary.
    """

    def __init__(self, kind: str = "masked_logistic", c: float = 1.0) -> None:
        if kind not in (
            "prior",
            "naive_bayes",
            "masked_logistic",
            "menu_logistic",
            "candidate_logistic",
        ):
            raise ValueError("Unknown forecaster")
        if not math.isfinite(c) or c <= 0:
            raise ValueError("C must be finite and positive")
        self.kind, self.c = kind, c
        self.state: dict[str, Any] = {}

    @staticmethod
    def tokens(vector: np.ndarray) -> list[str]:
        return [f"{i}:{v:.1f}" for i, v in enumerate(vector)]

    def fit(self, rows: list[dict[str, Any]], x: np.ndarray) -> Forecaster:
        if not rows or len(rows) != len(x) or x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Need finite aligned training features")
        y = targets(rows)
        counts = np.bincount(y, minlength=3).astype(float) + 1
        self.state = dict(prior=(counts / counts.sum()).tolist(), feature_count=x.shape[1])
        if self.kind == "prior":
            return self
        if self.kind == "naive_bayes":
            token_counts: list[Counter[str]] = [Counter() for _ in range(3)]
            vocabulary = set()
            for action, vector in zip(y, x, strict=True):
                tokens = self.tokens(vector)
                token_counts[action].update(tokens)
                vocabulary.update(tokens)
            self.state.update(
                token_counts=[dict(c) for c in token_counts], vocabulary=sorted(vocabulary)
            )
            return self
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-8] = 1
        self.state.update(mean=mean.tolist(), scale=scale.tolist())
        z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        if self.kind == "candidate_logistic":
            import warnings

            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import LogisticRegression

            candidates, binary_targets, weights = [], [], []
            for i, row in enumerate(rows):
                legal = row["inputs"]["legal_action_ids"]
                if len(legal) < 2:
                    continue
                for action in legal:
                    vector = np.zeros((3, z.shape[1]))
                    vector[action] = z[i]
                    candidates.append(vector.ravel())
                    correct = action == y[i]
                    binary_targets.append(int(correct))
                    weights.append(0.5 if correct else 0.5 / (len(legal) - 1))
            if not candidates:
                raise ValueError("Candidate binary scorer needs multi-option training decisions")
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model = LogisticRegression(
                    C=self.c,
                    solver="liblinear",
                    fit_intercept=False,
                    max_iter=3000,
                    random_state=42,
                )
                model.fit(np.asarray(candidates), binary_targets, sample_weight=weights)
            self.state["weights"] = model.coef_[0].reshape(3, z.shape[1]).T.tolist()
            return self
        if self.kind == "menu_logistic":
            import warnings

            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import LogisticRegression

            models = {}
            for menu in sorted({tuple(r["inputs"]["legal_action_ids"]) for r in rows}):
                ids = [
                    i for i, r in enumerate(rows) if tuple(r["inputs"]["legal_action_ids"]) == menu
                ]
                key = canonical(menu)
                classes = sorted(set(y[ids]))
                if len(classes) < 2 or set(classes) != set(menu):
                    prior = np.bincount(y[ids], minlength=3) + 1
                    models[key] = dict(prior=(prior / prior.sum()).tolist(), sparse_fallback=True)
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model = LogisticRegression(C=self.c, max_iter=3000).fit(z[ids, 1:], y[ids])
                models[key] = dict(
                    classes=model.classes_.tolist(),
                    coef=model.coef_.tolist(),
                    intercept=model.intercept_.tolist(),
                )
            self.state["menus"] = models
            return self
        from scipy.optimize import minimize  # type: ignore[import-untyped]
        from scipy.special import logsumexp  # type: ignore[import-untyped]

        mask = legal_mask(rows)
        onehot = np.eye(3)[y]
        n, d = z.shape
        penalty = 1 / (n * self.c)

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            w = flat.reshape(d, 3)
            logits = np.where(mask, z @ w, -np.inf)
            logp = logits - logsumexp(logits, axis=1, keepdims=True)
            p = np.exp(logp)
            loss = -logp[np.arange(n), y].mean() + 0.5 * penalty * (w[1:] ** 2).sum()
            grad = z.T @ (p - onehot) / n
            grad[1:] += penalty * w[1:]
            return float(loss), grad.ravel()

        result = minimize(
            objective,
            np.zeros(d * 3),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 3000, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Masked logistic failed to converge: {result.message}")
        self.state["weights"] = result.x.reshape(d, 3).tolist()
        self.state["optimizer"] = dict(
            iterations=int(result.nit), objective=float(result.fun), success=True
        )
        return self

    def predict(self, rows: list[dict[str, Any]], x: np.ndarray) -> np.ndarray:
        if not self.state:
            raise RuntimeError("Fit model first")
        if x.shape != (len(rows), self.state["feature_count"]) or not np.isfinite(x).all():
            raise ValueError("Prediction feature shape/value mismatch")
        mask = legal_mask(rows)
        if self.kind == "prior":
            logits = np.tile(np.log(self.state["prior"]), (len(rows), 1))
        elif self.kind == "naive_bayes":
            vocab = set(self.state["vocabulary"])
            counts = self.state["token_counts"]
            logits = np.tile(np.log(self.state["prior"]), (len(rows), 1))
            for i, vector in enumerate(x):
                tokens = [t for t in self.tokens(vector) if t in vocab]
                for a in range(3):
                    denominator = sum(counts[a].values()) + max(1, len(vocab))
                    logits[i, a] += sum(
                        math.log((counts[a].get(t, 0) + 1) / denominator) for t in tokens
                    )
        else:
            z = (x - np.asarray(self.state["mean"])) / np.asarray(self.state["scale"])
            if self.kind in ("masked_logistic", "candidate_logistic"):
                logits = np.column_stack([np.ones(len(x)), z]) @ np.asarray(self.state["weights"])
            else:
                logits = np.tile(np.log(self.state["prior"]), (len(rows), 1))
                for i, r in enumerate(rows):
                    model = self.state["menus"].get(canonical(r["inputs"]["legal_action_ids"]))
                    if model is None:
                        continue  # Unseen menu: smoothed train-only action prior.
                    if "prior" in model:
                        logits[i] = np.log(model["prior"])
                        continue
                    scores = z[i] @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])
                    if len(model["classes"]) == 2:
                        scores = np.array([0.0, float(scores[0])])
                    logits[i, model["classes"]] = scores
        p = masked_softmax(logits, mask)
        validate_probabilities(rows, p)
        return p

    def to_dict(self) -> dict[str, Any]:
        return dict(kind=self.kind, c=self.c, state=self.state)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Forecaster:
        model = cls(value["kind"], value["c"])
        model.state = value["state"]
        return model


def bayesian_predictions(
    rows: list[dict[str, Any]],
    base: np.ndarray,
    *,
    temperature: float = 1.0,
    hazard: float = 0.02,
    timing: str = "completed_hands",
) -> tuple[np.ndarray, np.ndarray]:
    """25-style discrete filter. Predict first, then reveal actions; no true styles read.

    Completed-hands mode releases updates at the next hand boundary. The per-hand
    hazard is applied once per elapsed hand, including hands with no opponent action.
    """
    if not 0 <= hazard < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid Bayesian filter parameters")
    if timing not in ("completed_hands", "all_prior"):
        raise ValueError("Invalid feedback timing")
    validate_probabilities(rows, base)
    styles = np.array(
        [(a, f) for a in (-1.6, -0.8, 0, 0.8, 1.6) for f in (-1.6, -0.8, 0, 0.8, 1.6)]
    )
    prior = np.exp(-0.5 * (styles / 0.95) ** 2).prod(axis=1)
    prior /= prior.sum()
    offset = np.column_stack([styles[:, 1], np.zeros(len(styles)), styles[:, 0]])
    result, trace = np.zeros_like(base), np.zeros((len(rows), 2))
    memory: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        hand = row["hand_index"]
        m = memory.setdefault(row["session_id"], dict(q=prior.copy(), hand=hand, pending=[]))
        if hand < m["hand"]:
            raise ValueError("Nonchronological Bayesian feedback")
        if hand != m["hand"]:
            for likelihood in m["pending"]:
                m["q"] *= likelihood
                m["q"] /= m["q"].sum()
            m["pending"] = []
            survival = (1 - hazard) ** (hand - m["hand"])
            m["q"] = survival * m["q"] + (1 - survival) * prior
        mask = np.tile([a in row["inputs"]["legal_action_ids"] for a in range(3)], (len(styles), 1))
        candidate = masked_softmax(
            np.log(np.maximum(base[i], 1e-12))[None, :] + temperature * offset, mask
        )
        result[i] = m["q"] @ candidate
        trace[i] = m["q"] @ styles
        action = row["inputs"]["legal_action_ids"][row["target_index"]]
        likelihood = candidate[:, action]
        if timing == "completed_hands":
            m["pending"].append(likelihood.copy())
        else:
            m["q"] *= likelihood
            m["q"] /= m["q"].sum()
        m["hand"] = hand
    validate_probabilities(rows, result)
    return result, trace
