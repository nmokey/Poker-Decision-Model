"""Adapters to the shared candidate-scoring interface without merging label meanings.

Task metadata and split groups stay outside model inputs. These adapters do not
resplit data, fit a model, or turn observed opponent actions into Hero responses.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.opponents.data import FeatureSpec, canonical, feature_matrix, validate_inputs


@dataclass(frozen=True)
class DecisionTask:
    examples: tuple[DecisionExample, ...]
    groups: tuple[str, ...]
    question: str
    label_semantics: str

    def __post_init__(self) -> None:
        if len(self.examples) != len(self.groups) or not self.question or not self.label_semantics:
            raise ValueError("Each example needs an external group and explicit task semantics")
        if len({e.source_id for e in self.examples}) != len(self.examples):
            raise ValueError("Repeated source IDs")


def source_task(records: Sequence[dict[str, Any]], *, source: str) -> DecisionTask:
    """Adapt replayed Pluribus or Leduc teacher rows to the existing shared scorer.

    Leduc uses its declared argmax target; mixed teacher distributions and sampled
    actions remain in the original records. This is not distribution distillation.
    Preserve the source workflow's existing hand/episode membership.
    """
    if source not in ("pluribus", "leduc_teacher"):
        raise ValueError("Choose pluribus or leduc_teacher explicitly")
    examples, groups = [], []
    for row in records:
        inputs = row["inputs"]
        examples.append(
            DecisionExample(
                row["source_id"],
                row.get("street", "leduc"),
                DecisionInputs(inputs["context"], tuple(inputs["candidates"])),
                row["target_index"],
            )
        )
        groups.append(str(row["hand_id" if source == "pluribus" else "episode"]))
    observed = source == "pluribus"
    return DecisionTask(
        tuple(examples),
        tuple(groups),
        "What action will the acting player take?"
        if observed
        else "What action does the teacher policy prefer?",
        "observed action category" if observed else "solver-policy argmax",
    )


def opponent_task(rows: list[dict[str, Any]], *, spec: FeatureSpec) -> DecisionTask:
    """Expose the forecasting experiment to the shared encoder/training interface.

    Uses exactly the existing causal numeric features, serialized as JSON; current
    targets, opponent IDs, session IDs, profiles and switch times are not encoded.
    Group metadata preserves whole opponent instances. Use each saved split intact
    and pass this task's question to encode_examples. PokerBench action-type metrics
    do not apply to these abstract action categories; use opponents/evaluation.py.
    """
    previous: dict[str, tuple[int, int]] = {}
    identities: dict[str, str] = {}
    for row in rows:
        validate_inputs(row["inputs"])
        sid, oid = row["session_id"], row["opponent_id"]
        position = (row["hand_index"], row["decision_index"])
        if sid in previous and position <= previous[sid]:
            raise ValueError("Opponent rows must retain strict within-session chronology")
        if sid in identities and identities[sid] != oid:
            raise ValueError("A session cannot change opponent identity")
        previous[sid], identities[sid] = position, oid
    features = feature_matrix(rows, spec)
    examples = []
    for row, values in zip(rows, features, strict=True):
        examples.append(
            DecisionExample(
                f"{row['session_id']}:hand-{row['hand_index']}:decision-{row['decision_index']}",
                f"leduc_round_{row['inputs']['public']['round']}",
                DecisionInputs(
                    canonical({"visible": row["inputs"], "causal_features": values.tolist()}),
                    tuple(row["inputs"]["candidates"]),
                ),
                row["target_index"],
            )
        )
    return DecisionTask(
        tuple(examples),
        tuple(row["opponent_id"] for row in rows),
        "What action will the opponent take next?",
        "observed opponent action",
    )
