"""Causal Pluribus histories shared by opponent forecasts and decision inputs.

Memory resets per archive directory; related letter-suffix directories share a
split group. Only completed, co-present hands enter the alias-keyed memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import warnings
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.opponents.data import history_summary
from poker_decision.source_experiments import PHH_SHA256, pluribus_inputs

ACTIONS = ("fold", "check_call", "bet_raise")
STREETS = ("preflop", "flop", "turn", "river")
PROFILE_SECTION = "\nOpponent profiles (completed observed hands): "
WINDOW = 64


def split_group(table: str) -> str:
    match = re.fullmatch(r"(\d+)[a-z]*", table)
    if match is None:
        raise ValueError(f"Unrecognized Pluribus table: {table}")
    return match[1]


def profile(events: list[dict[str, Any]], hands: int) -> dict[str, Any]:
    """Overall counts retained for the development-pilot adapter."""
    counts = [sum(e["action_id"] == a for e in events) for a in range(3)]
    opportunities = [sum(a in e["legal"] for e in events) for a in range(3)]
    return dict(
        hands=hands,
        counts=counts,
        opportunities=opportunities,
        rates=[(n + 1) / (d + 2) for n, d in zip(counts, opportunities, strict=True)],
    )


def public_snapshot(state: Any, prefix: list[list[float]], big_blind: float) -> dict[str, Any]:
    """Whitelist public state; never reads hole cards, labels, or outcomes."""
    actor = state.actor_index
    if actor is None:
        raise ValueError("No actor")
    legal = [
        a
        for a, allowed in enumerate(
            (state.can_fold(), state.can_check_or_call(), state.can_complete_bet_or_raise_to())
        )
        if allowed
    ]
    board = [repr(c) for c in state.get_board_cards(0)]
    return dict(
        actor=actor,
        street={0: 0, 3: 1, 4: 2, 5: 3}[len(board)],
        board=board,
        pot=float(state.total_pot_amount) / big_blind,
        to_call=float(state.checking_or_calling_amount) / big_blind,
        stacks=[float(x) / big_blind for x in state.stacks],
        bets=[float(x) / big_blind for x in state.bets],
        active=list(state.statuses),
        prefix=[list(e) for e in prefix],
        legal=legal,
    )


class HistoryBuilder:
    """Commit public actions transactionally, after a whole hand succeeds."""

    def __init__(self, window: int = WINDOW) -> None:
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        self.window = window
        self.events: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self.hands: Counter[tuple[str, str]] = Counter()
        self.last_hand = -1

    def snapshot(self, observer: str, target: str, public: dict[str, Any]) -> dict[str, Any]:
        key = observer, target
        # Window before matching, exactly as in the canonical Leduc experiment.
        events = list(self.events.get(key, ()))
        comparable = history_summary(
            [e for e in events if e["facing_bet"] == bool(public["to_call"])],
            {"legal_action_ids": public["legal"], "public": {"round": public["street"]}},
            None,
        )
        comparable["observed_decisions"] = len(events)
        comparable["hands"] = self.hands[key]
        return dict(overall=profile(events, self.hands[key]), comparable=comparable)

    def profile_text(self, observer: str, target: str, seat: int, public: dict[str, Any]) -> str:
        hands = self.hands[observer, target]
        if not hands:
            return f"seat {seat}: H=0, no observed history"
        parts = []
        for facing, label in ((True, "facing bet"), (False, "no bet")):
            h = self.snapshot(observer, target, {**public, "to_call": float(facing)})["comparable"]
            n = h["comparable_opportunities"]
            rates = "/".join(f"{r:.2f}" for r in h["rates"])
            parts.append(
                f"{label} n={n}, F/C/R={rates}" if n else f"{label}: no comparable decisions"
            )
        return f"seat {seat}: H={hands}; " + "; ".join(parts)

    def replay(
        self, text: str, *, hand_id: str, hand_index: int, hero_limit: int | None = 0
    ) -> tuple[list[dict[str, Any]], list[DecisionExample], list[str]]:
        from pokerkit import HandHistory, parse_action

        if hand_index <= self.last_hand:
            raise ValueError("Hands must be strictly increasing")
        hh = HandHistory.loads(text)
        if hh.hand is not None and hh.hand != hand_index:
            raise ValueError("Declared hand number disagrees with chronological index")
        players = hh.players
        if players is None or len(players) != 6 or len(set(players)) != 6:
            raise ValueError("Require six distinct player aliases")
        if players.count("Pluribus") != 1:
            raise ValueError("Forecast observer must be present")
        state = hh.create_state()
        big_blind = float(max(hh.blinds_or_straddles or [0]))
        if big_blind <= 0:
            raise ValueError("Require a positive big blind")
        prefix: list[list[float]] = []
        public_actions: list[str] = []
        pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rows: list[dict[str, Any]] = []
        heroes: list[DecisionExample] = []
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            for index, action in enumerate(hh.actions):
                parts = action.split()
                if action.startswith("d db") and state.can_burn_card():
                    state.burn_card("??")
                betting = bool(re.fullmatch(r"p\d+", parts[0])) and parts[1] in {"f", "cc", "cbr"}
                if betting:
                    public = public_snapshot(state, prefix, big_blind)
                    actor = public["actor"]
                    if actor != int(parts[0][1:]) - 1:
                        raise ValueError("Recorded actor disagrees with replay")
                    alias = players[actor]
                    history = self.snapshot("Pluribus", alias, public)
                    hero_input = None
                    if alias != "Pluribus" and (hero_limit is None or len(heroes) < hero_limit):
                        own = pluribus_inputs(state, public_actions)
                        profiles = [
                            self.profile_text(alias, opponent, seat + 1, public)
                            for seat, opponent in enumerate(players)
                            if seat != actor and public["active"][seat]
                        ]
                        hero_input = DecisionInputs(
                            own.context + PROFILE_SECTION + " | ".join(profiles), own.candidates
                        )
                    # Complete both input snapshots before inspecting this target.
                    target = {"f": 0, "cc": 1, "cbr": 2}[parts[1]]
                    if target not in public["legal"]:
                        raise ValueError("Observed action is not legal")
                    source_id = f"{hand_id}:action-{index}"
                    target_index = public["legal"].index(target)
                    rows.append(
                        dict(
                            source_id=source_id,
                            hand_id=hand_id,
                            hand_index=hand_index,
                            actor_alias=alias,
                            observer="Pluribus",
                            street=STREETS[public["street"]],
                            inputs=dict(
                                public=public,
                                history=history,
                                legal_action_ids=public["legal"],
                                candidates=[ACTIONS[a] for a in public["legal"]],
                            ),
                            target_index=target_index,
                        )
                    )
                    if hero_input is not None:
                        heroes.append(
                            DecisionExample(
                                source_id, STREETS[public["street"]], hero_input, target_index
                            )
                        )
                    pending[alias].append(
                        dict(
                            hand=hand_index,
                            round=public["street"],
                            legal=public["legal"],
                            facing_bet=bool(public["to_call"]),
                            action_id=target,
                        )
                    )
                parse_action(state, action)
                if betting:
                    # A bet size becomes public only AFTER that action.
                    amount = float(parts[2]) / big_blind if target == 2 else 0.0
                    prefix.append([actor, public["street"], target, amount])
                if betting or action.startswith("d db"):
                    public_actions.append(action)
        if state.status:
            raise ValueError("Incomplete hand; histories not committed")
        for observer in players:
            for target_alias in players:
                self.events[observer, target_alias].extend(pending[target_alias])
                self.hands[observer, target_alias] += 1
        self.last_hand = hand_index
        return rows, heroes, [str(w.message) for w in emitted]


def without_history(inputs: DecisionInputs) -> DecisionInputs:
    if inputs.context.count(PROFILE_SECTION) != 1:
        raise ValueError("Expected exactly one appended opponent-profile section")
    return DecisionInputs(inputs.context.split(PROFILE_SECTION)[0], inputs.candidates)


def load_histories(
    path: Path,
    *,
    hero_per_session: int | None = 8,
    max_sessions: int | None = None,
    hands_per_session: int | None = None,
    window: int = WINDOW,
) -> tuple[list[dict[str, Any]], list[DecisionExample], dict[str, Any]]:
    """Replay before sampling; optional bounds select a chronological prefix.

    The tuple interface and bounded Hero probes remain for the feasibility script.
    The meeting workflow requests all human Hero decisions.
    """
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive integer")
    for name, value in (
        ("hero_per_session", hero_per_session),
        ("max_sessions", max_sessions),
        ("hands_per_session", hands_per_session),
    ):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer or None")
    if hashlib.sha256(path.read_bytes()).hexdigest() != PHH_SHA256:
        raise ValueError("Source checksum mismatch")
    tables: dict[str, list[tuple[int, str]]] = defaultdict(list)
    rows, heroes = [], []
    warnings_seen: Counter[str] = Counter()
    hands_loaded = 0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = re.fullmatch(r"data/pluribus/(\d+[a-z]*)/(\d+)\.phh", name)
            if match:
                tables[match[1]].append((int(match[2]), name))
        if not tables:
            raise ValueError("No Pluribus hands")
        groups = sorted({split_group(t) for t in tables}, key=int)
        selected_groups = set(groups[:max_sessions])
        for table in sorted(tables, key=lambda t: (int(split_group(t)), t)):
            if split_group(table) not in selected_groups:
                continue
            hands = sorted(tables[table])
            numbers = [h for h, _ in hands]
            if numbers != list(range(numbers[0], numbers[-1] + 1)):
                raise ValueError(f"Ambiguous or incomplete hand order: {table}")
            hands = hands[:hands_per_session]
            selected = (
                None
                if hero_per_session is None
                else set(np.linspace(0, len(hands) - 1, hero_per_session, dtype=int))
            )
            builder = HistoryBuilder(window=window)
            for position, (hand, name) in enumerate(hands):
                current, samples, messages = builder.replay(
                    archive.read(name).decode(),
                    hand_id=name,
                    hand_index=hand,
                    hero_limit=None if selected is None or position in selected else 0,
                )
                for row in current:
                    row.update(session_id=split_group(table), table_id=table)
                rows.extend(current)
                heroes.extend(samples)
                warnings_seen.update(messages)
                hands_loaded += 1
    return (
        rows,
        heroes,
        dict(
            hands=hands_loaded,
            sessions=len(selected_groups),
            tables=len({r["table_id"] for r in rows}),
            decisions=len(rows),
            human_decisions=sum(r["actor_alias"] != "Pluribus" for r in rows),
            hero_examples=len(heroes),
            warnings=dict(warnings_seen),
            window=window,
            history_timing="completed earlier hands, observer co-present, reset per directory",
            split_unit="numeric session group; letter-suffix directories stay together",
            max_sessions=max_sessions,
            hands_per_session=hands_per_session,
        ),
    )


def history_vector(history: dict[str, Any]) -> list[float]:
    """The same 11 history features used by Leduc's feature_matrix."""
    return [
        math.log1p(history["observed_decisions"]),
        math.log1p(history["comparable_opportunities"]),
        *[math.log1p(v) for v in history["counts"]],
        *[math.log1p(v) for v in history["denominators"]],
        *history["rates"],
    ]


def features(rows: list[dict[str, Any]], *, with_history: bool) -> np.ndarray:
    """Same public-state block; optionally append the actor's causal summary."""
    values = []
    for row in rows:
        p = row["inputs"]["public"]
        ranks = Counter("23456789TJQKA".index(c[0]) for c in p["board"])
        suits = Counter(c[1] for c in p["board"])
        vector = (
            [float(p["street"] == i) for i in range(4)]
            + [float(p["actor"] == i) for i in range(6)]
            + np.log1p([p["pot"], p["to_call"]] + p["stacks"] + p["bets"]).tolist()
            + [float(v) for v in p["active"]]
            + [float(a in p["legal"]) for a in range(3)]
            + [ranks[i] / 4 for i in range(13)]
            + [max(suits.values(), default=0) / 5, len(p["prefix"]) / 20]
            + [
                sum(e[0] == seat and e[2] == a for e in p["prefix"]) / 10
                for seat in range(6)
                for a in range(3)
            ]
            + [sum(e[1] == p["street"] and e[2] == a for e in p["prefix"]) / 10 for a in range(3)]
        )
        if with_history:
            vector += history_vector(row["inputs"]["history"]["comparable"])
        values.append(vector)
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Need nonempty finite features")
    return result


def shared_sample(
    prepared: tuple[list[dict[str, Any]], list[DecisionExample], dict[str, Any]],
    *,
    seed: int = 42,
    train_size: int = 128,
    validation_size: int = 64,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Unique ID join, then grouped split and one shared human decision sample."""
    if not 0 < validation_fraction < 1 or min(train_size, validation_size) < 1:
        raise ValueError("Need positive sample sizes and a validation fraction in (0, 1)")
    rows, examples, audit = prepared
    forecasts = {r["source_id"]: r for r in rows}
    heroes = {e.source_id: e for e in examples}
    if len(forecasts) != len(rows) or len(heroes) != len(examples):
        raise ValueError("Views require unique source IDs")
    if not heroes.keys() <= forecasts.keys():
        raise ValueError("Unmatched Hero source IDs")
    joined = []
    for key in sorted(heroes):
        f, h = forecasts[key], heroes[key]
        if f["actor_alias"] == "Pluribus":
            raise ValueError("Only human Hero targets are allowed")
        if f["target_index"] != h.target_index or len(f["inputs"]["candidates"]) != len(
            h.inputs.candidates
        ):
            raise ValueError("View target/menu mismatch")
        joined.append(
            dict(
                source_id=key,
                session_id=f["session_id"],
                table_id=f["table_id"],
                forecast=f,
                hero=h.to_record(),
            )
        )
    groups = sorted({r["session_id"] for r in joined})
    if len(groups) < 2:
        raise ValueError("Need multiple session groups")
    rng = random.Random(seed)
    rng.shuffle(groups)
    held = set(groups[: min(len(groups) - 1, max(1, math.ceil(len(groups) * validation_fraction)))])
    pools = [
        [r for r in joined if (r["session_id"] in held) == held_out] for held_out in (False, True)
    ]
    if len(pools[0]) < train_size or len(pools[1]) < validation_size:
        raise ValueError("Not enough human examples; increase replay bounds")
    return dict(
        train=rng.sample(pools[0], train_size),
        validation=rng.sample(pools[1], validation_size),
        seed=seed,
        validation_fraction=validation_fraction,
        source_audit=audit,
        train_sessions=sorted(set(groups) - held),
        validation_sessions=sorted(held),
    )


def dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()
