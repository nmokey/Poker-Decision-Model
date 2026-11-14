"""Causal information and observer visibility checks for real Hold'em histories."""

import copy

import numpy as np
import pytest

from poker_decision.holdem_history import HistoryBuilder, features

PHH = """variant = 'NT'
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [1, 2, 0, 0, 0, 0]
min_bet = 2
starting_stacks = [100, 100, 100, 100, 100, 100]
players = ['a', 'b', 'Pluribus', 'd', 'e', 'f']
actions = ['d dh p1 AhAs', 'd dh p2 KcKd', 'd dh p3 QhQs',
'd dh p4 JcJd', 'd dh p5 ThTs', 'd dh p6 9c9d',
'p3 f', 'p4 f', 'p5 f', 'p6 f', 'p1 f']
"""


@pytest.fixture(autouse=True)
def optional_engine():
    pytest.importorskip("pokerkit")


def test_current_hand_is_not_in_committed_history():
    builder = HistoryBuilder()
    rows, _, _ = builder.replay(PHH, hand_id="first", hand_index=0)
    assert all(r["inputs"]["history"]["overall"]["hands"] == 0 for r in rows)
    assert all(r["inputs"]["history"]["overall"]["counts"] == [0, 0, 0] for r in rows)
    again, _, _ = builder.replay(PHH, hand_id="second", hand_index=1)
    assert again[0]["inputs"]["history"]["overall"]["hands"] == 1
    assert again[0]["inputs"]["history"]["overall"]["counts"] == [1, 0, 0]
    assert rows[0]["inputs"]["history"]["overall"]["hands"] == 0  # immutable snapshot
    with pytest.raises(ValueError, match="increasing"):
        builder.replay(PHH, hand_id="duplicate", hand_index=1)


def test_forecast_ignores_private_cards_and_current_future_targets():
    original, _, _ = HistoryBuilder().replay(PHH, hand_id="x", hand_index=0)
    private, _, _ = HistoryBuilder().replay(PHH.replace("AhAs", "2h2s"), hand_id="x", hand_index=0)
    assert [r["inputs"] for r in original] == [r["inputs"] for r in private]
    current, _, _ = HistoryBuilder().replay(
        PHH.replace("'p3 f'", "'p3 cbr 6'").replace("'p1 f']", "'p1 f', 'p2 f']"),
        hand_id="x",
        hand_index=0,
    )
    assert original[0]["inputs"] == current[0]["inputs"]
    assert original[0]["target_index"] != current[0]["target_index"]
    future, _, _ = HistoryBuilder().replay(
        PHH.replace("'p4 f'", "'p4 cbr 6'").replace("'p1 f']", "'p1 f', 'p2 f']"),
        hand_id="x",
        hand_index=0,
    )
    assert original[0]["inputs"] == future[0]["inputs"]


def test_features_use_only_declared_input_fields():
    rows, _, _ = HistoryBuilder().replay(PHH, hand_id="x", hand_index=0)
    altered = copy.deepcopy(rows)
    for r in altered:
        r.update(target_index=999, actor_alias="hidden", session_id="other", outcome=100)
    for history in (False, True):
        np.testing.assert_array_equal(
            features(rows, with_history=history), features(altered, with_history=history)
        )
    assert features(rows, with_history=True).shape[1] > features(rows, with_history=False).shape[1]


def test_alias_follows_seat_rotation_and_new_observer_has_no_past():
    builder = HistoryBuilder()
    builder.replay(PHH, hand_id="first", hand_index=0)
    # Swap two player aliases in seat metadata; everyone remains a unique person.
    rotated = PHH.replace("['a', 'b'", "['b', 'a'")
    rows, _, _ = builder.replay(rotated, hand_id="second", hand_index=1)
    # Actor 'a' now sits in seat 2 and does not act in this hand. Actor 'b' at seat 1
    # had no recorded decision in the previous hand, even though seat 1 folded.
    b = next(r for r in rows if r["actor_alias"] == "b")
    assert b["inputs"]["history"]["overall"]["counts"] == [0, 0, 0]
    public = b["inputs"]["public"]
    assert builder.snapshot("new arrival", "b", public)["overall"]["hands"] == 0
    assert builder.snapshot("Pluribus", "b", public)["overall"]["hands"] == 2
    assert HistoryBuilder().snapshot("Pluribus", "b", public)["overall"]["hands"] == 0


def test_hero_sees_own_cards_and_only_copresent_opponent_histories():
    builder = HistoryBuilder()
    builder.replay(PHH, hand_id="first", hand_index=0)
    _, heroes, _ = builder.replay(PHH, hand_id="second", hand_index=1, hero_limit=1)
    assert len(heroes) == 1
    # Pluribus folded first; the sampled human Hero is seat 4 with JcJd.
    context = heroes[0].inputs.context
    assert "Hero cards: Jc Jd" in context
    assert "Ah" not in context and "Kc" not in context and "Qh" not in context
    assert "H=1" in context
    changed = PHH.replace("'a', 'b'", "'new arrival', 'b'")
    _, heroes, _ = builder.replay(changed, hand_id="third", hand_index=2, hero_limit=1)
    assert "seat 1: H=0" in heroes[0].inputs.context


def test_repeated_actions_and_future_board_do_not_enter_earlier_history():
    text = PHH.replace(
        "'p3 f', 'p4 f', 'p5 f', 'p6 f', 'p1 f']",
        "'p3 cbr 6', 'p4 cc', 'p5 f', 'p6 f', 'p1 f', 'p2 f', "
        "'d db 2c3c4c', 'p3 cc', 'p4 cbr 4', 'p3 f']",
    )
    rows, heroes, _ = HistoryBuilder().replay(text, hand_id="x", hand_index=0, hero_limit=None)
    assert {r["inputs"]["public"]["street"] for r in rows} == {0, 1}
    assert {e.street for e in heroes} == {"preflop", "flop"}
    assert all(r["inputs"]["history"]["overall"]["counts"] == [0, 0, 0] for r in rows)
    changed, _, _ = HistoryBuilder().replay(
        text.replace("2c3c4c", "2d3c4c"), hand_id="x", hand_index=0
    )
    for before, after in zip(rows, changed, strict=True):
        if before["inputs"]["public"]["street"] == 0:
            assert before["inputs"] == after["inputs"]


def test_hero_target_and_future_changes_preserve_input_but_reach_next_history():
    changed = PHH.replace("'p4 f'", "'p4 cbr 6'").replace("'p1 f']", "'p1 f', 'p2 f']")
    a, b = HistoryBuilder(), HistoryBuilder()
    rows, heroes, _ = a.replay(PHH, hand_id="x", hand_index=0, hero_limit=None)
    modified, changed_heroes, _ = b.replay(changed, hand_id="x", hand_index=0, hero_limit=None)
    assert rows[1]["inputs"] == modified[1]["inputs"]
    assert heroes[0].inputs == changed_heroes[0].inputs
    assert heroes[0].target_index != changed_heroes[0].target_index
    following, _, _ = a.replay(PHH, hand_id="y", hand_index=1)
    following_changed, _, _ = b.replay(PHH, hand_id="y", hand_index=1)
    assert following[1]["inputs"]["history"] != following_changed[1]["inputs"]["history"]


def test_incomplete_replay_does_not_commit_and_departed_observer_misses_hand():
    builder = HistoryBuilder()
    builder.replay(PHH, hand_id="x", hand_index=0)
    before = copy.deepcopy((dict(builder.events), builder.hands, builder.last_hand))
    with pytest.raises(ValueError, match="Incomplete"):
        builder.replay(PHH.replace(", 'p1 f'", ""), hand_id="broken", hand_index=1)
    assert (dict(builder.events), builder.hands, builder.last_hand) == before
    builder.replay(PHH.replace("'d'", "'new'"), hand_id="y", hand_index=1)
    rows, _, _ = builder.replay(PHH, hand_id="z", hand_index=2)
    p = rows[0]["inputs"]["public"]
    assert builder.snapshot("d", "a", p)["overall"]["hands"] == 2
    assert builder.snapshot("new", "a", p)["overall"]["hands"] == 1


def test_recent_window_precedes_street_and_menu_matching():
    from poker_decision.opponents.data import history_summary

    builder = HistoryBuilder(window=2)
    builder.replay(PHH, hand_id="x", hand_index=0)
    rows, _, _ = builder.replay(PHH, hand_id="y", hand_index=1)
    p = rows[1]["inputs"]["public"]
    events = list(builder.events["Pluribus", "d"])
    expected = history_summary(
        events, {"legal_action_ids": p["legal"], "public": {"round": p["street"]}}, 2
    )
    summary = builder.snapshot("Pluribus", "d", p)["comparable"]
    for field in expected:
        assert expected[field] == summary[field]
    builder.replay(PHH, hand_id="z", hand_index=2)
    assert len(builder.events["Pluribus", "d"]) == 2
    assert builder.snapshot("Pluribus", "d", p)["overall"]["hands"] == 3


def test_loader_resets_suffix_history_but_groups_split_and_checks_source(tmp_path, monkeypatch):
    import hashlib
    import zipfile

    import poker_decision.holdem_history as holdem

    path = tmp_path / "histories.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/pluribus/100/0.phh", "hand = 0\n" + PHH)
        archive.writestr("data/pluribus/100b/1.phh", "hand = 1\n" + PHH)
    with pytest.raises(ValueError, match="checksum"):
        holdem.load_histories(path)
    monkeypatch.setattr(holdem, "PHH_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    rows, heroes, audit = holdem.load_histories(path, hero_per_session=None)
    assert audit["sessions"] == 1 and audit["tables"] == 2
    assert {r["session_id"] for r in rows} == {"100"}
    assert all(r["inputs"]["history"]["overall"]["hands"] == 0 for r in rows)
    assert len(heroes) == 8
