"""Scientific contracts for the consolidated opponent forecasts; no external data needed."""

import copy
import json

import numpy as np
import pytest

from poker_decision.opponents.data import (
    ACTIONS,
    Config,
    FeatureSpec,
    equity,
    feature_matrix,
    history_summary,
    plan_sessions,
    validate_dataset,
    visible_inputs,
)
from poker_decision.opponents.evaluation import paired_interval, per_row_metrics
from poker_decision.opponents.models import Forecaster, bayesian_predictions


def row(action=1, *, sid="a", hand=0, decision=0, legal=(0, 1, 2)):
    return dict(
        session_id=sid,
        opponent_id=sid,
        split="train",
        hand_index=hand,
        decision_index=decision,
        inputs=dict(
            public=dict(
                round=1,
                actor_seat=1,
                board_rank=None,
                pot=2,
                contributions=[1, 1],
                prior_actions=[],
            ),
            observer_rank=0,
            legal_action_ids=list(legal),
            candidates=[ACTIONS[a] for a in legal],
        ),
        target_index=list(legal).index(action),
    )


def test_identity_split_keeps_repeated_sessions_and_switch_directions():
    specs = plan_sessions(Config())
    assert specs == plan_sessions(Config())
    for split in ("train", "validation", "test"):
        subset = [s for s in specs if s["split"] == split]
        assert {s["direction"] for s in subset if s["profile"] == "switch"} == {0, 1}
    for identity in {s["opponent_id"] for s in specs}:
        sessions = [s for s in specs if s["opponent_id"] == identity]
        assert len(sessions) == 2 and len({s["split"] for s in sessions}) == 1
        assert sessions[0]["noise"] == sessions[1]["noise"]
        assert sessions[0]["seeds"] != sessions[1]["seeds"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(seed=-1),
        dict(opponents_per_profile=9),
        dict(hands_per_session=3),
        dict(policy_family="unknown"),
    ],
)
def test_bad_generation_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_history_counts_denominators_and_window_order():
    inputs = row()["inputs"]
    prior = [
        dict(round=1, legal=[0, 1, 2], action_id=0),
        dict(round=1, legal=[0, 1, 2], action_id=2),
        dict(round=2, legal=[0, 1, 2], action_id=1),
    ]
    h = history_summary(prior, inputs, None)
    assert h["counts"] == [1, 0, 1] and h["denominators"] == [2, 2, 2]
    assert h["rates"] == [2 / 5, 1 / 5, 2 / 5]
    assert history_summary(prior, inputs, 1)["comparable_opportunities"] == 0
    assert history_summary(prior, inputs, 1, window_order="after_match")["counts"] == [0, 0, 1]
    empty = history_summary([], inputs, None)
    assert empty["rates"] == [1 / 3] * 3 and empty["denominators"] == [0] * 3
    assert h["counts"] == [1, 0, 1]  # A returned summary does not retain mutable prior data.


def test_current_future_and_other_session_targets_do_not_enter_features():
    rows = [row(0), row(1, decision=1), row(2, hand=1), row(1, sid="b")]
    changed = copy.deepcopy(rows)
    changed[1]["target_index"] = 2
    changed[2]["target_index"] = 0
    for timing in ("all_prior", "completed_hands"):
        spec = FeatureSpec(window=4, timing=timing)
        x, modified = feature_matrix(rows, spec), feature_matrix(changed, spec)
        np.testing.assert_array_equal(x[:2], modified[:2])
        np.testing.assert_array_equal(x[-1], modified[-1])
        if timing == "completed_hands":
            np.testing.assert_array_equal(x[0], x[1])
    np.testing.assert_array_equal(
        feature_matrix(rows, FeatureSpec()), feature_matrix(changed, FeatureSpec())
    )
    altered = copy.deepcopy(rows)
    for r in altered:
        r.update(policy="hidden", seed=123, future="raise", private_opponent_rank=2)
    np.testing.assert_array_equal(
        feature_matrix(rows, FeatureSpec(window=4)), feature_matrix(altered, FeatureSpec(window=4))
    )


def test_contract_rejects_duplicate_order_cross_identity_and_bad_labels():
    rows = [row()]
    specs = [dict(session_id="a", opponent_id="a", split="train", hands=2)]
    validate_dataset(rows, specs)
    with pytest.raises(ValueError, match="nonchronological"):
        validate_dataset(rows * 2, specs)
    with pytest.raises(ValueError, match="crosses splits"):
        validate_dataset(
            rows, specs + [dict(session_id="b", opponent_id="a", split="test", hands=2)]
        )
    changed = copy.deepcopy(rows)
    changed[0]["target_index"] = True
    with pytest.raises(ValueError, match="Target"):
        validate_dataset(changed, specs)
    changed = copy.deepcopy(rows)
    changed[0]["inputs"]["public"]["policy"] = "tight"
    with pytest.raises(ValueError, match="public state"):
        validate_dataset(changed, specs)


@pytest.mark.parametrize(
    "kind", ["prior", "naive_bayes", "masked_logistic", "menu_logistic", "candidate_logistic"]
)
def test_models_legal_finite_label_independent_and_json_reload(kind):
    if kind in ("masked_logistic", "menu_logistic", "candidate_logistic"):
        pytest.importorskip("sklearn")
    rows = [row(i % 3, hand=i) for i in range(60)]
    x = np.array([[1, i % 3] for i in range(60)], dtype=float)
    model = Forecaster(kind).fit(rows, x)
    predicted = model.predict(rows, x)
    restored = Forecaster.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(predicted, restored.predict(rows, x))
    changed = copy.deepcopy(rows)
    for r in changed:
        r["target_index"] = 999
    np.testing.assert_array_equal(predicted, model.predict(changed, x))
    assert (predicted >= 0).all() and np.isfinite(predicted).all()
    np.testing.assert_allclose(predicted.sum(axis=1), 1)
    probe = [row(1, legal=(1, 2)), row(1, legal=(1,))]
    p = model.predict(probe, x[:2])
    assert p[0, 0] == 0 and p[1].tolist() == [0, 1, 0]
    if kind in ("masked_logistic", "menu_logistic", "candidate_logistic"):
        assert (predicted.argmax(axis=1) == np.arange(60) % 3).mean() > 0.95


def test_binary_menu_numeric_reload_matches_sklearn_semantics():
    pytest.importorskip("sklearn")
    rows = [row(1 if i < 10 else 2, hand=i, legal=(1, 2)) for i in range(20)]
    x = np.arange(20, dtype=float)[:, None]
    model = Forecaster("menu_logistic").fit(rows, x)
    p = model.predict(rows, x)
    assert p[0, 1] > 0.9 and p[-1, 2] > 0.9 and (p[:, 0] == 0).all()


def test_bayesian_predicts_before_feedback_and_resets_sessions():
    rows = [row(2), row(2, decision=1), row(0, hand=1), row(2, sid="b")]
    base = np.full((len(rows), 3), 1 / 3)
    changed = copy.deepcopy(rows)
    changed[0]["target_index"] = 0
    for timing in ("completed_hands", "all_prior"):
        p, trace = bayesian_predictions(rows, base, timing=timing)
        q, _ = bayesian_predictions(changed, base, timing=timing)
        np.testing.assert_array_equal(p[0], q[0])
        np.testing.assert_array_equal(p[-1], q[-1])
        assert not np.allclose(p[2], q[2])
        if timing == "completed_hands":
            np.testing.assert_array_equal(p[1], q[1])
        assert trace.shape == (4, 2)
        np.testing.assert_allclose(p.sum(axis=1), 1)


def test_known_metrics_and_opponent_cluster_bootstrap():
    rows = [row(1), row(2, hand=1), row(0, sid="b")]
    p = np.array([[0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2]])
    m = per_row_metrics(rows, p)
    np.testing.assert_allclose(m["log_loss"], -np.log([0.5, 0.7, 0.6]))
    assert m["accuracy"].tolist() == [1, 1, 1]
    ties = per_row_metrics([row(2)], np.ones((1, 3)) / 3)
    assert ties["accuracy"].item() == 1  # bet_raise is lexical tie winner
    assert ties["tie_expected_accuracy"].item() == pytest.approx(1 / 3)
    rows[1]["session_id"] = "another-session-same-opponent"
    interval = paired_interval(rows, np.full(3, -0.2), repetitions=100)
    assert interval["opponents"] == 2
    assert interval["interval_95"] == pytest.approx([-0.2, -0.2])
    with pytest.raises(ValueError):
        per_row_metrics(rows, np.full((3, 3), 0.5))


def test_equity_card_removal_hand_values():
    assert equity(0, None) < equity(1, None) < equity(2, None)
    assert equity(0, 0) == 1.0
    assert equity(0, 2) == 0.125
    assert 0 <= equity(2, 1) <= 1


@pytest.mark.parametrize("family", ["weighted", "categorical", "table", "residual", "cfr"])
def test_engine_generation_reproducible_and_replay(family):
    pytest.importorskip("pyspiel")
    from poker_decision.opponents.generate import generate, replay_audit

    cfg = Config(
        policy_family=family, hands_per_session=4, sessions_per_opponent=1, cfr_iterations=2
    )
    first = generate(cfg)
    assert first == generate(cfg)
    audit = replay_audit(first, cfg)
    assert audit["verified_decisions"] == len(first["rows"])
    assert audit["replayed_hands"] == len(first["sessions"]) * 4
    altered = copy.deepcopy(first)
    altered["rows"][0]["inputs"]["public"]["pot"] += 2
    with pytest.raises(ValueError):
        replay_audit(altered, cfg)
    if family == "cfr":
        assert first["teacher"]["iterations"] == 2
        assert "teacher only" in first["teacher"]["scope"]


def test_engine_hidden_opponent_card_does_not_change_observer_input():
    pyspiel = pytest.importorskip("pyspiel")
    game = pyspiel.load_game("leduc_poker")
    left, right = game.new_initial_state(), game.new_initial_state()
    for s, opponent_card in ((left, 2), (right, 4)):
        s.apply_action(0)
        s.apply_action(opponent_card)
    assert visible_inputs(left, 0, [], observer_card=True) == visible_inputs(
        right, 0, [], observer_card=True
    )
    assert "observer_rank" not in visible_inputs(left, 0, [], observer_card=False)


def test_equity_matches_engine_terminal_returns_exhaustively():
    pyspiel = pytest.importorskip("pyspiel")
    game = pyspiel.load_game("leduc_poker")
    results = {}
    for hero, villain, board in __import__("itertools").permutations(range(6), 3):
        state = game.new_initial_state()
        for action in (hero, villain, 1, 1, board, 1, 1):
            state.apply_action(action)
        assert state.is_terminal()
        outcome = 1 if state.returns()[0] > 0 else 0 if state.returns()[0] < 0 else 0.5
        results.setdefault((hero // 2, board // 2), []).append(outcome)
    for (rank, board), outcomes in results.items():
        assert equity(rank, board) == pytest.approx(np.mean(outcomes))


def test_workflow_selection_excludes_test_labels_and_detects_corrupt_artifacts(
    tmp_path, monkeypatch
):
    pytest.importorskip("pyspiel")
    pytest.importorskip("sklearn")
    import poker_decision.opponents.experiment as exp

    root = exp.prepare(Config(hands_per_session=6, sessions_per_opponent=1), tmp_path)
    original_read = exp.read_jsonl

    def deny_test(path):
        assert path.name != "test.jsonl", "Selection tried to read test labels"
        return original_read(path)

    monkeypatch.setattr(exp, "read_jsonl", deny_test)
    selected = exp.select(root, c_grid=(0.1,))
    monkeypatch.setattr(exp, "read_jsonl", original_read)
    results = exp.final_evaluation(root, selected, repetitions=30)
    assert results == exp.final_evaluation(root, selected, repetitions=30)
    assert len(results["examples"]) == 5 and "bayesian" in results["overall"]
    assert selected == exp.select(root, c_grid=(0.1,))
    cv = exp.select(root, c_grid=(0.1,), selection_method="grouped_cv")
    choices = json.loads((cv / "selection.json").read_text())
    assert choices["settings"]["selection_method"] == "grouped_cv"
    assert any(len(r.get("folds", [])) == 5 for r in choices["candidates"])
    with (selected / "test_predictions.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="artifact checksum"):
        exp.final_evaluation(root, selected, repetitions=30)
    with (root / "test.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Dataset checksum"):
        exp.verify_manifest(root)
