"""Stand-in for the data team's apps/contract.py, written from the Meeting 3 spec.

Replace this file with theirs once it lands. The demo only relies on the fields below.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Spot:
    """One decision point: the shared input every engine receives."""

    context: str
    candidates: tuple[str, ...]


@dataclass
class Prediction:
    """What any engine returns: a distribution over a Spot's candidates.

    `probs` is aligned to `Spot.candidates` and sums to 1. On failure an engine sets
    `probs=None, ok=False` and explains why in `note` rather than raising to the UI.
    """

    probs: list[float] | None
    latency_ms: float
    ok: bool = True
    note: str = ""
    raw: str = ""  # optional engine-native output (e.g. the LLM's raw text)


class Engine(Protocol):
    def __call__(self, spot: Spot) -> Prediction: ...
