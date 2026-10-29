"""Hidden-information, target-semantics and hand-boundary checks for optional sources."""

import zipfile

import pytest

from poker_decision.source_experiments import (
    generate_leduc,
    leduc_inputs,
    prepare_pluribus,
    replay_pluribus,
    split_observed_hands,
)

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


def test_pluribus_inputs_ignore_opponent_cards_and_later_actions():
    pytest.importorskip("pokerkit")
    original, _ = replay_pluribus(PHH, hand_id="a")
    changed, _ = replay_pluribus(PHH.replace("AhAs", "2h2s"), hand_id="b")
    assert original[0]["inputs"] == changed[0]["inputs"]
    assert original[0]["inputs"]["candidates"] == ["fold", "call", "raise (any legal size)"]
    text = PHH.replace("'p3 f'", "'p3 cbr 6'")
    new_label, _ = replay_pluribus(text, hand_id="c")
    assert original[0]["inputs"] == new_label[0]["inputs"]
    assert original[0]["target_index"] != new_label[0]["target_index"]
    assert "Ah" not in original[0]["inputs"]["context"]
    assert set(original[0]["inputs"]) == {"context", "candidates"}


def test_replay_failures_are_reported_not_silently_dropped(tmp_path):
    pytest.importorskip("pokerkit")
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/pluribus/0/good.phh", PHH)
        archive.writestr("data/pluribus/0/bad.phh", "not a hand")
    audit = prepare_pluribus(path, max_hands=10)
    assert audit["attempted_hands"] == 2
    assert len(audit["failures"]) == 1 and len(audit["records"]) == 5


def test_hand_and_identical_context_components_cannot_cross_split():
    rows = [
        {"hand_id": "a", "inputs": {"context": "shared"}},
        {"hand_id": "b", "inputs": {"context": "shared"}},
        {"hand_id": "b", "inputs": {"context": "other"}},
        {"hand_id": "c", "inputs": {"context": "other"}},
        {"hand_id": "d", "inputs": {"context": "independent"}},
    ]
    fit, held = split_observed_hands(rows)
    assert {r["hand_id"] for r in fit}.isdisjoint(r["hand_id"] for r in held)
    assert {r["inputs"]["context"] for r in fit}.isdisjoint(r["inputs"]["context"] for r in held)
    assert {r["hand_id"] for r in fit} in ({"a", "b", "c"}, {"d"})
    with pytest.raises(ValueError, match="independent"):
        split_observed_hands(rows[:-1])


def test_leduc_teacher_targets_and_hidden_card_invariance():
    pyspiel = pytest.importorskip("pyspiel")
    game = pyspiel.load_game("leduc_poker")
    left, right = game.new_initial_state(), game.new_initial_state()
    left.apply_action(0)
    left.apply_action(2)
    right.apply_action(0)
    right.apply_action(4)
    assert leduc_inputs(left) == leduc_inputs(right)
    # Tiny algorithm/API test, not a scientific training run.
    generated = generate_leduc(episodes=2, iterations=1, seed=42)
    assert generated == generate_leduc(episodes=2, iterations=1, seed=42)
    assert "Teacher policy only" in generated["exploitability_scope"]
    for record in generated["records"]:
        p = record["target_probabilities"]
        assert len(p) == len(record["inputs"]["candidates"])
        assert sum(p) == pytest.approx(1) and min(p) >= 0
        assert p[record["target_index"]] == max(p)
        assert record["sampled_action"] in record["inputs"]["candidates"]
        assert "target" not in record["inputs"]
