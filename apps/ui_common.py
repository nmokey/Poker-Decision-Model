"""Helpers shared by the single-hand view and the benchmark view."""

import math
import time

from apps.contract import Engine, Prediction, Spot


def fmt_ms(ms: float) -> str:
    return f"{ms:,.2f} ms" if ms < 10 else f"{ms:,.0f} ms"


def run_safely(engine: Engine, spot: Spot) -> Prediction:
    """Engines must not raise, but if one does, the screen still mustn't crash."""
    start = time.perf_counter()
    try:
        return engine(spot)
    except Exception as err:
        return Prediction(
            probs=None,
            latency_ms=(time.perf_counter() - start) * 1000,
            ok=False,
            note=f"Engine crashed: {err!r}",
        )


def problem_with(spot: Spot, pred: Prediction) -> str | None:
    """Why this prediction can't be used as a valid distribution, or None if it can."""
    if not pred.ok:
        return pred.note or "The engine reported a failure."
    probs = pred.probs
    if probs is None:
        return "The engine returned no probabilities."
    if len(probs) != len(spot.candidates):
        return f"Expected {len(spot.candidates)} probabilities, got {len(probs)}."
    if not all(
        type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probs
    ) or not any(p > 0 for p in probs):
        return "Probabilities must be finite, nonnegative and not all zero."
    if not math.isfinite(pred.latency_ms) or pred.latency_ms < 0:
        return "Invalid latency."
    if abs(sum(probs) - 1) > 1e-6:
        return f"Probabilities sum to {sum(probs):.3f}, not 1."
    return None


def top_pick(probs: list[float]) -> int | None:
    """Index of the favourite option, or None if several options tie for it."""
    best = max(probs)
    return probs.index(best) if probs.count(best) == 1 else None
