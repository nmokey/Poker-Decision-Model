"""Separate observed-play and toy solver-policy experiments; optional dependencies."""

import hashlib
import json
import random
import re
import warnings
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

from poker_decision.data import DecisionInputs

PHH_URL = "https://zenodo.org/api/records/10796886/files/poker-hand-histories.zip/content"
PHH_SHA256 = "bfefe55e138fdffae90859592d765951c0904bcf83691b9f25b3c5c7581e7363"


def download_pluribus(directory: Path) -> Path:
    """Fetch the immutable PHH archive and verify bytes before use, including cache hits."""
    path = directory / "pluribus_phh_v1.zip"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != PHH_SHA256:
            raise ValueError("Cached PHH archive checksum mismatch")
        return path
    directory.mkdir(parents=True, exist_ok=True)
    with urlopen(PHH_URL, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != PHH_SHA256:
        raise ValueError("Downloaded PHH archive checksum mismatch")
    temporary = path.with_suffix(".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def pluribus_inputs(state: Any, history: list[str]) -> DecisionInputs:
    actor = state.actor_index
    if actor is None:
        raise ValueError("No acting player")
    candidates = []
    if state.can_fold():
        candidates.append("fold")
    if state.can_check_or_call():
        candidates.append("call" if state.checking_or_calling_amount else "check")
    if state.can_complete_bet_or_raise_to():
        candidates.append(
            "raise (any legal size)" if max(state.bets) > 0 else "bet (any legal size)"
        )
    own = [repr(c) for c in state.hole_cards[actor]]
    if len(own) != 2 or any("?" in c for c in own):
        raise ValueError("Unknown acting-player cards")
    board = [repr(c) for c in state.get_board_cards(0)]
    context = (
        f"Game: 6-player NLHE. Hero seat: {actor + 1}. Hero cards: {' '.join(own)}. "
        f"Board: {' '.join(board) or 'none'}. Pot: {state.total_pot_amount} chips. "
        f"Public bets: {list(state.bets)}. Public stacks: {list(state.stacks)}. "
        f"Prior public actions: {'; '.join(history) or 'none'}."
    )
    return DecisionInputs(context, tuple(candidates))


def replay_pluribus(text: str, *, hand_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    from pokerkit import HandHistory, parse_action

    hh = HandHistory.loads(text)
    if hh.players is None:
        raise ValueError("Player identities are required for the Pluribus subset")
    state = hh.create_state()
    history: list[str] = []
    records = []
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        for index, action in enumerate(hh.actions):
            parts = action.split()
            if action.startswith("d db") and state.can_burn_card():
                state.burn_card("??")
            betting = bool(re.fullmatch(r"p\d+", parts[0])) and parts[1] in {"f", "cc", "cbr"}
            if betting:
                actor = state.actor_index
                if actor != int(parts[0][1:]) - 1:
                    raise ValueError("Recorded actor disagrees with engine")
                inputs = pluribus_inputs(state, history)
                # Construct the complete input and menu BEFORE reading the target action.
                if parts[1] == "f":
                    target = "fold"
                elif parts[1] == "cc":
                    target = "call" if state.checking_or_calling_amount else "check"
                else:
                    target = (
                        "raise (any legal size)" if max(state.bets) > 0 else "bet (any legal size)"
                    )
                if inputs.candidates.count(target) != 1:
                    raise ValueError("Observed action does not match a legal category")
                board_count = len(list(state.get_board_cards(0)))
                records.append(
                    {
                        "source_id": f"{hand_id}:action-{index}",
                        "hand_id": hand_id,
                        "street": {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[board_count],
                        "inputs": {
                            "context": inputs.context,
                            "candidates": list(inputs.candidates),
                        },
                        "target_index": inputs.candidates.index(target),
                        "is_pluribus": hh.players[actor].lower() == "pluribus",
                        "label_semantics": "observed action category; exact sizing abstracted",
                    }
                )
            parse_action(state, action)
            if betting or action.startswith("d db"):
                history.append(action)
    return records, [str(w.message) for w in emitted]


def prepare_pluribus(archive_path: Path, *, max_hands: int = 10) -> dict[str, Any]:
    """Bounded feasibility replay; never silently skip an unparseable hand."""
    if max_hands < 1:
        raise ValueError("max_hands must be positive")
    records, failures = [], []
    warning_counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        paths = sorted(
            p for p in archive.namelist() if p.startswith("data/pluribus/") and p.endswith(".phh")
        )
        if not paths:
            raise ValueError("No Pluribus hands in archive")
        for path in paths[:max_hands]:
            try:
                rows, messages = replay_pluribus(archive.read(path).decode(), hand_id=path)
                records.extend(rows)
                warning_counts.update(messages)
            except (ValueError, KeyError, TypeError, AssertionError, IndexError) as error:
                failures.append({"hand_id": path, "error": str(error)})
    return {
        "records": records,
        "failures": failures,
        "warnings": dict(warning_counts),
        "archive_hands": len(paths),
        "attempted_hands": min(max_hands, len(paths)),
        "source": PHH_URL,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "label_semantics": "Observed behavior, not optimal or opponent-dependent action labels",
    }


def split_observed_hands(
    records: list[dict[str, Any]], *, seed: int = 42, validation_fraction: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Connected components of shared hand OR exact input context stay together."""
    if not records or not 0 < validation_fraction < 1:
        raise ValueError("Need records and a fraction strictly between zero and one")
    parent = {r["hand_id"]: r["hand_id"] for r in records}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return str(key)

    seen: dict[str, str] = {}
    for record in records:
        context, hand = record["inputs"]["context"], record["hand_id"]
        if context in seen:
            left, right = find(hand), find(seen[context])
            parent[max(left, right)] = min(left, right)
        else:
            seen[context] = hand
    components = sorted({find(r["hand_id"]) for r in records})
    if len(components) < 2:
        raise ValueError("Need at least two independent hand/context components")
    random.Random(seed).shuffle(components)
    cut = max(1, min(len(components) - 1, round(len(components) * validation_fraction)))
    held = set(components[:cut])
    return (
        [r for r in records if find(r["hand_id"]) not in held],
        [r for r in records if find(r["hand_id"]) in held],
    )


def leduc_inputs(state: Any) -> DecisionInputs:
    player = state.current_player()
    return DecisionInputs(
        "Game: two-player Leduc. " + state.information_state_string(player),
        tuple(state.action_to_string(player, a).lower() for a in state.legal_actions(player)),
    )


def generate_leduc(
    *, episodes: int = 2000, iterations: int = 500, seed: int = 42
) -> dict[str, Any]:
    """Approximate solver distribution targets. Exploitability measures the teacher only."""
    import pyspiel

    if episodes < 1 or iterations < 1:
        raise ValueError("Episode and iteration counts must be positive")
    game = pyspiel.load_game("leduc_poker")
    solver = pyspiel.CFRPlusSolver(game)
    convergence = []
    checkpoints = {0, min(10, iterations), min(50, iterations), min(100, iterations), iterations}
    for iteration in range(iterations + 1):
        if iteration in checkpoints:
            convergence.append(
                {
                    "iteration": iteration,
                    "teacher_exploitability": float(
                        pyspiel.exploitability(game, solver.average_policy())
                    ),
                }
            )
        if iteration < iterations:
            solver.evaluate_and_update_policy()
    policy = solver.average_policy()
    rng = np.random.default_rng(seed)
    records = []
    for episode in range(episodes):
        state = game.new_initial_state()
        decision = 0
        while not state.is_terminal():
            if state.is_chance_node():
                chance = state.chance_outcomes()
                state.apply_action(
                    int(rng.choice([a for a, _ in chance], p=[p for _, p in chance]))
                )
                continue
            inputs = leduc_inputs(state)
            actions = state.legal_actions()
            target = policy.action_probabilities(state)
            probs = np.array([target[a] for a in actions], dtype=float)
            if (
                not np.all(np.isfinite(probs))
                or np.any(probs < 0)
                or not np.isclose(probs.sum(), 1)
            ):
                raise ValueError("Invalid solver distribution")
            best = min(range(len(actions)), key=lambda j: (-probs[j], inputs.candidates[j]))
            sampled = int(rng.choice(len(actions), p=probs))
            records.append(
                {
                    "source_id": f"leduc-{episode}-{decision}",
                    "episode": episode,
                    "inputs": {"context": inputs.context, "candidates": list(inputs.candidates)},
                    "target_index": best,
                    "target_probabilities": probs.tolist(),
                    "sampled_action": inputs.candidates[sampled],
                }
            )
            state.apply_action(actions[sampled])
            decision += 1
    return {
        "records": records,
        "convergence": convergence,
        "seed": seed,
        "episodes": episodes,
        "iterations": iterations,
        "game": str(game),
        "label_semantics": "Approximate CFR+ teacher policy, "
        "not calibrated optimal-action confidence",
        "exploitability_scope": "Teacher policy only; no learned neural policy was evaluated",
    }


def write_source_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, allow_nan=False) for r in records) + "\n")
