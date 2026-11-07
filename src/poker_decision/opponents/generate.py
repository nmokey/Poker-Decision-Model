"""OpenSpiel simulation adapters retaining the submissions' distinct opponent families."""

from __future__ import annotations

from typing import Any

import numpy as np

from poker_decision.opponents.data import (
    GAME_PARAMETERS,
    Config,
    plan_sessions,
    validate_dataset,
    visible_inputs,
)

WEIGHTS = {
    "tight": (1.4, 0.3, -1.3),
    "passive": (-0.6, 1.8, -1.4),
    "aggressive": (-1.3, 0.1, 1.6),
    "balanced": (0.0, 0.5, 0.3),
}
TABLES = {
    "tight": (
        ((0.85, 0.15, 0), (0.45, 0.50, 0.05), (0.02, 0.58, 0.40)),
        ((0, 0.95, 0.05), (0, 0.80, 0.20), (0, 0.45, 0.55)),
    ),
    "aggressive": (
        ((0.35, 0.35, 0.30), (0.10, 0.50, 0.40), (0, 0.30, 0.70)),
        ((0, 0.50, 0.50), (0, 0.30, 0.70), (0, 0.15, 0.85)),
    ),
}
CENTERS = {
    "balanced": (0.0, 0.0),
    "loose_aggressive": (1.3, -0.75),
    "tight_passive": (-0.95, 0.95),
    "calling_station": (-0.85, -0.9),
    "tight_aggressive": (0.65, 1.05),
}


def normalize(values: Any) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Invalid sampling weights")
    return np.asarray(p / p.sum())


def teacher_key(state: Any) -> str:
    return f"{state.current_player()}:{state.information_state_string(state.current_player())}"


def train_teacher(game: Any, iterations: int) -> tuple[dict[str, Any], dict[str, Any]]:
    import pyspiel

    solver = pyspiel.CFRPlusSolver(game)
    curve = []
    checkpoints = {0, min(10, iterations), min(100, iterations), iterations}
    for i in range(iterations + 1):
        if i in checkpoints:
            curve.append(
                dict(
                    iteration=i,
                    exploitability=float(pyspiel.exploitability(game, solver.average_policy())),
                )
            )
        if i < iterations:
            solver.evaluate_and_update_policy()
    # Export before the C++ solver is destroyed; no borrowed policy escapes this scope.
    policy = solver.average_policy()
    table: dict[str, Any] = {}
    stack = [game.new_initial_state()]
    while stack:
        state = stack.pop()
        if state.is_terminal():
            continue
        actions = (
            [a for a, _ in state.chance_outcomes()]
            if state.is_chance_node()
            else list(state.legal_actions())
        )
        if not state.is_chance_node():
            probabilities = policy.action_probabilities(state)
            entry = dict(
                actions=actions,
                probabilities=normalize([probabilities[a] for a in actions]).tolist(),
            )
            key = teacher_key(state)
            if key in table and table[key] != entry:
                raise ValueError("Teacher differs inside an information set")
            table[key] = entry
        stack.extend(state.child(a) for a in actions)
    return table, dict(
        iterations=iterations,
        convergence=curve,
        information_states=len(table),
        scope="CFR+ teacher only; not learned predictor exploitability",
    )


def teacher_probabilities(state: Any, table: dict[str, Any]) -> np.ndarray:
    entry = table[teacher_key(state)]
    if entry["actions"] != list(state.legal_actions()):
        raise ValueError("Teacher menu mismatch")
    return normalize(entry["probabilities"])


def behavior(spec: dict[str, Any], hand: int, family: str) -> str:
    profile = str(spec["profile"])
    after = spec["switch_hand"] is not None and hand >= spec["switch_hand"]
    if profile in ("tight_to_loose", "loose_to_tight"):
        first, second = profile.split("_to_")
        return second if after else first
    if profile == "switch":
        pair = {
            "weighted": ("tight", "aggressive"),
            "table": ("tight", "aggressive"),
            "categorical": ("passive", "aggressive"),
            "residual": ("tight_passive", "loose_aggressive"),
        }[family]
        if spec["direction"]:
            pair = pair[::-1]
        return pair[int(after)]
    return profile


def opponent_probabilities(
    state: Any, spec: dict[str, Any], hand: int, family: str, table: dict[str, Any]
) -> np.ndarray:
    legal = list(state.legal_actions())
    name = behavior(spec, hand, family)
    noise = np.asarray(spec["noise"])
    rank = state.private_card(state.current_player()) // 2
    board = state.public_card()
    paired = board >= 0 and rank == board // 2
    if family == "categorical":
        weights = np.array((0.15, 0.70, 0.15) if name == "passive" else (0.10, 0.25, 0.65))
        return normalize((weights * np.exp(noise))[legal])
    if family == "weighted":
        strength = (rank - 1) * 0.65 + (1.4 if paired else 0)
        logits = np.array(WEIGHTS[name]) + noise + np.array([-strength, 0.0, strength])
        return normalize(np.exp(logits[legal] - max(logits[legal])))
    if family == "table":
        strength_id = rank if board < 0 else 2 if paired else 1 if rank > board // 2 else 0
        weights = np.array(TABLES[name][0 if 0 in legal else 1][strength_id])
        return normalize((weights * np.exp(noise))[legal])
    if family == "residual":
        aggression, folding = np.array(CENTERS[name]) + noise[:2]
        strength = 0.25 + 0.28 * rank if board < 0 else 1.0 if paired else 0.15 + 0.26 * rank
        money = np.asarray(state.money())
        pressure = (money[state.current_player()] - money.min()) / state.pot()
        logits = np.array(
            [
                -0.7 + 2.8 * pressure - 2 * strength + folding,
                1 + 0.15 * strength - 0.4 * pressure,
                -0.45 + 2.1 * strength - 1.35 * pressure + aggression,
            ]
        )
        return normalize(np.exp(logits[legal] - max(logits[legal])))
    base = teacher_probabilities(state, table)
    if name == "cfr":
        return base
    strong, medium = paired or (board < 0 and rank == 2), rank == 2 or (board < 0 and rank == 1)
    if name == "tight":
        weights = np.array((0.08, 1, 2) if strong else (2.5, 1, 0.35) if medium else (10, 1, 0.08))
    else:
        weights = np.array((0.15, 1, 7 if strong else 5))
    return normalize((0.88 * base + 0.12 / len(legal)) * (np.array(weights) * np.exp(noise))[legal])


def generate(cfg: Config) -> dict[str, Any]:
    import pyspiel

    game = pyspiel.load_game("leduc_poker", GAME_PARAMETERS)
    sessions = plan_sessions(cfg)
    table: dict[str, Any] = {}
    teacher: dict[str, Any] = {}
    if cfg.policy_family == "cfr" or cfg.observer_policy == "cfr":
        table, teacher = train_teacher(game, cfg.cfr_iterations)
    rows, traces = [], []
    for spec in sessions:
        rng = {key: np.random.default_rng(value) for key, value in spec["seeds"].items()}
        for hand in range(cfg.hands_per_session):
            opponent = hand % 2
            observer = 1 - opponent
            state = game.new_initial_state()
            prefix: list[dict[str, int]] = []
            actions, probabilities = [], []
            while not state.is_terminal():
                if state.is_chance_node():
                    choices, chance_p = zip(*state.chance_outcomes(), strict=True)
                    action = int(rng["chance"].choice(choices, p=chance_p))
                else:
                    actor, legal = state.current_player(), list(state.legal_actions())
                    if actor == opponent:
                        inputs = visible_inputs(
                            state, observer, prefix, observer_card=cfg.observer_card
                        )
                        p = opponent_probabilities(state, spec, hand, cfg.policy_family, table)
                        action = int(rng["opponent"].choice(legal, p=p))
                        rows.append(
                            dict(
                                session_id=spec["session_id"],
                                opponent_id=spec["opponent_id"],
                                split=spec["split"],
                                hand_index=hand,
                                decision_index=len(prefix),
                                inputs=inputs,
                                target_index=legal.index(action),
                            )
                        )
                        probabilities.append(
                            dict(decision_index=len(prefix), probabilities=p.tolist())
                        )
                    else:
                        p = (
                            teacher_probabilities(state, table)
                            if cfg.observer_policy == "cfr"
                            else np.ones(len(legal)) / len(legal)
                        )
                        action = int(rng["observer"].choice(legal, p=p))
                    prefix.append(dict(seat=actor, round=int(state.round()), action_id=action))
                actions.append(action)
                state.apply_action(action)
            traces.append(
                dict(
                    session_id=spec["session_id"],
                    hand_index=hand,
                    actions=actions,
                    returns=state.returns(),
                    opponent_probabilities=probabilities,
                )
            )
    validate_dataset(rows, sessions)
    return dict(rows=rows, sessions=sessions, traces=traces, teacher=teacher, teacher_policy=table)


def replay_audit(dataset: dict[str, Any], cfg: Config) -> dict[str, int]:
    """Independently rebuild each visible decision/menu/label from engine actions."""
    import pyspiel

    validate_dataset(dataset["rows"], dataset["sessions"])
    game = pyspiel.load_game("leduc_poker", GAME_PARAMETERS)
    expected = {(r["session_id"], r["hand_index"], r["decision_index"]): r for r in dataset["rows"]}
    hands = set()
    count = 0
    for trace in dataset["traces"]:
        key = (trace["session_id"], trace["hand_index"])
        if key in hands:
            raise ValueError("Duplicate replay hand")
        hands.add(key)
        state = game.new_initial_state()
        prefix: list[dict[str, int]] = []
        opponent = trace["hand_index"] % 2
        for action in trace["actions"]:
            if state.is_terminal():
                raise ValueError("Action after terminal state")
            if state.is_chance_node():
                if action not in dict(state.chance_outcomes()):
                    raise ValueError("Illegal chance outcome")
            else:
                actor = state.current_player()
                if action not in state.legal_actions():
                    raise ValueError("Illegal replay action")
                if actor == opponent:
                    r = expected.pop((*key, len(prefix)))
                    inputs = visible_inputs(
                        state, 1 - opponent, prefix, observer_card=cfg.observer_card
                    )
                    if (
                        inputs != r["inputs"]
                        or inputs["legal_action_ids"][r["target_index"]] != action
                    ):
                        raise ValueError("Replay decision mismatch")
                    count += 1
                prefix.append(dict(seat=actor, round=int(state.round()), action_id=action))
            state.apply_action(action)
        if not state.is_terminal() or state.returns() != trace["returns"]:
            raise ValueError("Incomplete hand or return mismatch")
    if expected or len(hands) != sum(s["hands"] for s in dataset["sessions"]):
        raise ValueError("Missing replay hands/decisions")
    return dict(replayed_hands=len(hands), verified_decisions=count)
