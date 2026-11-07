"""One answer-free observer contract, deterministic identity splits, and causal features."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

ACTIONS = ("fold", "check_call", "bet_raise")
GAME_PARAMETERS = dict(players=2, action_mapping=False, suit_isomorphism=False, starting_player=0)
PROFILES = {
    "weighted": ("tight", "passive", "aggressive", "balanced", "switch"),
    "categorical": ("passive", "aggressive", "switch"),
    "table": ("tight", "aggressive", "switch"),
    "residual": (
        "balanced",
        "loose_aggressive",
        "tight_passive",
        "calling_station",
        "tight_aggressive",
        "switch",
    ),
    "cfr": ("cfr", "tight", "loose", "tight_to_loose", "loose_to_tight"),
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def seed_for(seed: int, *parts: Any) -> int:
    return int(hashlib.sha256(canonical([seed, *parts]).encode()).hexdigest()[:16], 16)


@dataclass(frozen=True)
class Config:
    seed: int = 20261106
    policy_family: str = "weighted"
    opponents_per_profile: int = 10
    sessions_per_opponent: int = 2
    hands_per_session: int = 60
    observer_policy: str = "uniform"
    observer_card: bool = True
    cfr_iterations: int = 500

    def __post_init__(self) -> None:
        if self.policy_family not in PROFILES or self.observer_policy not in ("uniform", "cfr"):
            raise ValueError("Unknown policy family or observer policy")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            type(self.opponents_per_profile) is not int
            or self.opponents_per_profile < 10
            or self.opponents_per_profile % 10
        ):
            raise ValueError(
                "Use a multiple of 10 opponents/profile, to split both switch directions"
            )
        for name in ("sessions_per_opponent", "hands_per_session", "cfr_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hands_per_session < 4:
            raise ValueError("Need at least four hands to observe a switch")


def plan_sessions(cfg: Config) -> list[dict[str, Any]]:
    """Assign whole opponent instances (including repeated sessions) before dealing."""
    sessions = []
    for profile in PROFILES[cfg.policy_family]:
        membership = {}
        # Stratify direction as well as profile; the two directions occur in every split.
        for parity in (0, 1):
            ids = np.arange(parity, cfg.opponents_per_profile, 2)
            np.random.default_rng(seed_for(cfg.seed, "split", profile, parity)).shuffle(ids)
            for j, identity_index in enumerate(ids):
                membership[int(identity_index)] = (
                    "train"
                    if j < 3 * len(ids) // 5
                    else "validation"
                    if j < 4 * len(ids) // 5
                    else "test"
                )
        for identity in range(cfg.opponents_per_profile):
            oid = f"{profile}-{identity:03d}"
            instance_seed = seed_for(cfg.seed, oid, "instance")
            noise = np.random.default_rng(instance_seed).normal(0, 0.2, 3).tolist()
            for repetition in range(cfg.sessions_per_opponent):
                sid = f"{oid}-session-{repetition:02d}"
                seeds = {k: seed_for(cfg.seed, sid, k) for k in ("chance", "observer", "opponent")}
                switching = profile in ("switch", "tight_to_loose", "loose_to_tight")
                switch = None
                if switching:
                    rng = np.random.default_rng(seed_for(cfg.seed, sid, "switch"))
                    switch = int(
                        rng.integers(
                            max(1, cfg.hands_per_session * 3 // 10),
                            min(cfg.hands_per_session, cfg.hands_per_session * 7 // 10 + 1),
                        )
                    )
                sessions.append(
                    dict(
                        session_id=sid,
                        opponent_id=oid,
                        split=membership[identity],
                        profile=profile,
                        noise=noise,
                        direction=identity % 2,
                        switch_hand=switch,
                        seeds=seeds,
                        instance_seed=instance_seed,
                        hands=cfg.hands_per_session,
                    )
                )
    return sessions


def visible_inputs(
    state: Any, observer: int, prefix: list[dict[str, int]], *, observer_card: bool
) -> dict[str, Any]:
    """Whitelist reads public state and optionally the observer's OWN card only."""
    legal = list(state.legal_actions())
    board = state.public_card()
    public = dict(
        round=int(state.round()),
        actor_seat=int(state.current_player()),
        board_rank=None if board < 0 else int(board // 2),
        pot=int(state.pot()),
        contributions=[int(100 - x) for x in state.money()],
        prior_actions=[dict(e) for e in prefix],
    )
    result: dict[str, Any] = dict(
        public=public, legal_action_ids=legal, candidates=[ACTIONS[a] for a in legal]
    )
    if observer_card:
        result["observer_rank"] = int(state.private_card(observer) // 2)
    return result


def validate_inputs(inputs: dict[str, Any]) -> None:
    allowed = {"public", "legal_action_ids", "candidates", "observer_rank"}
    if set(inputs) - allowed or not {"public", "legal_action_ids", "candidates"} <= inputs.keys():
        raise ValueError("Unexpected input fields")
    legal = inputs["legal_action_ids"]
    if (
        not isinstance(legal, list)
        or not legal
        or any(type(a) is not int or a not in range(3) for a in legal)
    ):
        raise ValueError("Invalid legal action menu")
    if legal != sorted(set(legal)) or inputs["candidates"] != [ACTIONS[a] for a in legal]:
        raise ValueError("Menu IDs/names must be aligned and unique")
    p = inputs["public"]
    if set(p) != {"round", "actor_seat", "board_rank", "pot", "contributions", "prior_actions"}:
        raise ValueError("Unexpected public state fields")
    if p["round"] not in (1, 2) or p["actor_seat"] not in (0, 1):
        raise ValueError("Invalid round/seat")
    if p["board_rank"] not in (None, 0, 1, 2) or (p["round"] == 1 and p["board_rank"] is not None):
        raise ValueError("Invalid board")
    if "observer_rank" in inputs and inputs["observer_rank"] not in range(3):
        raise ValueError("Invalid observer card")
    if len(p["contributions"]) != 2 or sum(p["contributions"]) != p["pot"]:
        raise ValueError("Pot/contribution mismatch")
    prefix = p["prior_actions"]
    if len(prefix) > 8 or any(set(e) != {"seat", "round", "action_id"} for e in prefix):
        raise ValueError("Invalid public action prefix")
    if any(
        e["seat"] not in (0, 1)
        or e["round"] not in (1, 2)
        or e["round"] > p["round"]
        or e["action_id"] not in (1, 2)
        for e in prefix
    ):
        raise ValueError("Invalid public action event")


def validate_dataset(rows: list[dict[str, Any]], sessions: list[dict[str, Any]]) -> dict[str, int]:
    specs = {s["session_id"]: s for s in sessions}
    if len(specs) != len(sessions) or not sessions or not rows:
        raise ValueError("Need unique sessions and nonempty data")
    identity_split: dict[str, str] = {}
    for s in sessions:
        if s["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid split")
        if identity_split.setdefault(s["opponent_id"], s["split"]) != s["split"]:
            raise ValueError("Opponent crosses splits")
    last: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()
    current = None
    for r in rows:
        sid = r["session_id"]
        if sid not in specs:
            raise ValueError("Unknown session")
        s = specs[sid]
        if r["opponent_id"] != s["opponent_id"] or r["split"] != s["split"]:
            raise ValueError("Session membership mismatch")
        if sid != current:
            if sid in seen:
                raise ValueError("Session rows must be contiguous")
            seen.add(sid)
            current = sid
        order = (r["hand_index"], r["decision_index"])
        if any(type(x) is not int or x < 0 for x in order) or order[0] >= s["hands"]:
            raise ValueError("Invalid decision order")
        if sid in last and order <= last[sid]:
            raise ValueError("Duplicate or nonchronological decision")
        last[sid] = order
        validate_inputs(r["inputs"])
        y = r["target_index"]
        if type(y) is not int or not 0 <= y < len(r["inputs"]["candidates"]):
            raise ValueError("Target is not in legal menu")
    if seen != set(specs):
        raise ValueError("A declared session has no opponent decisions")
    return dict(decisions=len(rows), sessions=len(sessions), opponents=len(identity_split))


def history_summary(
    prior: list[dict[str, Any]],
    inputs: dict[str, Any],
    window: int | None,
    *,
    window_order: str = "before_match",
    match_round: bool = True,
) -> dict[str, Any]:
    """Snapshot earlier opponent decisions; never accepts a current target."""
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError("window must be positive or None")
    if window_order not in ("before_match", "after_match"):
        raise ValueError("Unknown window ordering")
    events = prior[-window:] if window is not None and window_order == "before_match" else prior
    matched = [
        e
        for e in events
        if e["legal"] == inputs["legal_action_ids"]
        and (not match_round or e["round"] == inputs["public"]["round"])
    ]
    if window is not None and window_order == "after_match":
        matched = matched[-window:]
    counts = Counter(e["action_id"] for e in matched)
    n, legal = len(matched), inputs["legal_action_ids"]
    return dict(
        observed_decisions=len(events),
        comparable_opportunities=n,
        counts=[counts[a] for a in range(3)],
        denominators=[n if a in legal else 0 for a in range(3)],
        rates=[(counts[a] + 1) / (n + len(legal)) if a in legal else 0.0 for a in range(3)],
    )


def equity(observer_rank: int, board_rank: int | None) -> float:
    """Exact showdown equity versus uniform unknown physical cards; not a range estimate."""
    if observer_rank not in range(3) or board_rank not in (None, 0, 1, 2):
        raise ValueError("Invalid rank")
    remaining = [0, 0, 1, 1, 2, 2]
    remaining.remove(observer_rank)
    if board_rank is not None:
        remaining.remove(board_rank)
        deals = [(v, board_rank) for v in remaining]
    else:
        deals = list(itertools.permutations(remaining, 2))

    def strength(rank: int, board: int) -> tuple[bool, int]:
        return rank == board, rank

    return sum(
        1.0
        if strength(observer_rank, b) > strength(v, b)
        else 0.5
        if strength(observer_rank, b) == strength(v, b)
        else 0.0
        for v, b in deals
    ) / len(deals)


def public_features(inputs: dict[str, Any], representation: str = "full") -> list[float]:
    """Read only a whitelisted input; strength controls retain the same opportunity cues."""
    p, legal = inputs["public"], inputs["legal_action_ids"]
    if representation == "strength":
        if "observer_rank" not in inputs:
            raise ValueError("Strength features require the observer's own card")
        return [
            equity(inputs["observer_rank"], p["board_rank"]),
            float(p["round"]),
            *[float(a in legal) for a in range(3)],
        ]
    if representation != "full":
        raise ValueError("Unknown feature representation")
    values = [
        float(p["round"]),
        float(p["actor_seat"]),
        math.log1p(p["pot"]),
        *[float(v) for v in p["contributions"]],
        *[float(p["board_rank"] == rank) for rank in (None, 0, 1, 2)],
        *[float(a in legal) for a in range(3)],
    ]
    if "observer_rank" in inputs:
        values.extend(float(inputs["observer_rank"] == rank) for rank in range(3))
    for j in range(8):
        e = p["prior_actions"][j] if j < len(p["prior_actions"]) else None
        values.extend(float(e is not None and e["action_id"] == a) for a in range(3))
        values.extend([float(e["seat"]) if e else -1.0, float(e["round"]) if e else 0.0])
    return values


@dataclass(frozen=True)
class FeatureSpec:
    window: int | None = 0  # 0 disables history; None = all history
    timing: str = "completed_hands"
    window_order: str = "before_match"
    match_round: bool = True
    representation: str = "full"

    def __post_init__(self) -> None:
        if self.window is not None and (type(self.window) is not int or self.window < 0):
            raise ValueError("Invalid window")
        if self.timing not in ("completed_hands", "all_prior"):
            raise ValueError("Invalid history timing")
        if self.window_order not in ("before_match", "after_match"):
            raise ValueError("Invalid window order")
        if self.representation not in ("full", "strength"):
            raise ValueError("Invalid representation")


def feature_matrix(rows: list[dict[str, Any]], spec: FeatureSpec) -> np.ndarray:
    """Past test actions are online observations; no fitting occurs here."""
    result = []
    history: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        inputs = row["inputs"]
        prior = history.setdefault(row["session_id"], [])
        values = public_features(inputs, spec.representation)
        if spec.window != 0:
            available = [
                e for e in prior if spec.timing == "all_prior" or e["hand"] < row["hand_index"]
            ]
            h = history_summary(
                available,
                inputs,
                spec.window,
                window_order=spec.window_order,
                match_round=spec.match_round,
            )
            values.extend(
                [
                    math.log1p(h["observed_decisions"]),
                    math.log1p(h["comparable_opportunities"]),
                    *[math.log1p(v) for v in h["counts"]],
                    *[math.log1p(v) for v in h["denominators"]],
                    *h["rates"],
                ]
            )
        result.append(values)
        # Read this label only AFTER recording its feature vector.
        prior.append(
            dict(
                hand=row["hand_index"],
                round=inputs["public"]["round"],
                legal=list(inputs["legal_action_ids"]),
                action_id=inputs["legal_action_ids"][row["target_index"]],
            )
        )
    return np.asarray(result, dtype=float)


def config_dict(cfg: Config) -> dict[str, Any]:
    return asdict(cfg)
