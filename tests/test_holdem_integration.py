"""Shared sample, probability mapping, and provenance checks without downloads."""

import argparse
import copy
import json

import numpy as np
import pytest

from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.holdem_history import PROFILE_SECTION, shared_sample
from poker_decision.holdem_integration import (
    code_hashes,
    digest,
    forecast,
    load_sample,
    metrics,
    report,
    validate_sample,
    write_json,
)


def prepared():
    rows, heroes = [], []
    for group in range(10):
        for table in (str(group), f"{group}b"):
            for i in range(3):
                key = f"{table}/{i}:action-1"
                legal = [1, 2] if i == 0 else [0, 1, 2]
                candidates = (
                    ("check", "bet (any legal size)")
                    if i == 0
                    else ("fold", "call", "raise (any legal size)")
                )
                rows.append(
                    dict(
                        source_id=key,
                        session_id=str(group),
                        table_id=table,
                        actor_alias="human",
                        target_index=i % len(legal),
                        inputs=dict(
                            candidates=[("fold", "check_call", "bet_raise")[a] for a in legal],
                            legal_action_ids=legal,
                            public=dict(
                                street=0,
                                actor=1,
                                board=[],
                                pot=3.0,
                                to_call=float(i > 0),
                                stacks=[100.0] * 6,
                                bets=[0.0] * 6,
                                active=[True] * 6,
                                legal=legal,
                                prefix=[],
                            ),
                            history=dict(
                                comparable=dict(
                                    observed_decisions=3,
                                    comparable_opportunities=0,
                                    counts=[0, 0, 0],
                                    denominators=[0, 0, 0],
                                    rates=[1 / 3] * 3,
                                )
                            ),
                        ),
                    )
                )
                heroes.append(
                    DecisionExample(
                        key,
                        "preflop",
                        DecisionInputs(
                            "Public state. Hero cards: Ah As."
                            + PROFILE_SECTION
                            + "seat 1: no observed history",
                            candidates,
                        ),
                        i % len(legal),
                    )
                )
    return rows, heroes, {}


def saved(tmp_path):
    data = shared_sample(prepared(), train_size=16, validation_size=8)
    write_json(tmp_path / "shared_sample.json", data)
    manifest = dict(
        sample_sha256=digest(tmp_path / "shared_sample.json"),
        code_sha256=code_hashes(),
    )
    write_json(tmp_path / "preparation.json", manifest)
    return data, manifest


def test_join_split_and_saved_sample_checks(tmp_path):
    a = shared_sample(prepared(), train_size=16, validation_size=8)
    assert a == shared_sample(prepared(), train_size=16, validation_size=8)
    assert len(a["train_sessions"]) == 8 and len(a["validation_sessions"]) == 2
    assert {r["session_id"] for r in a["train"]}.isdisjoint(
        r["session_id"] for r in a["validation"]
    )
    validate_sample(a)
    rows, heroes, audit = prepared()
    with pytest.raises(ValueError, match="unique"):
        shared_sample((rows, heroes + [heroes[0]], audit), train_size=4, validation_size=2)
    saved(tmp_path)
    load_sample(tmp_path)
    p = tmp_path / "shared_sample.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="sample changed"):
        load_sample(tmp_path)


def test_menu_and_split_mismatches_rejected():
    a = shared_sample(prepared(), train_size=16, validation_size=8)
    changed = copy.deepcopy(a)
    changed["train_sessions"].append(a["validation_sessions"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_sample(changed)
    changed = copy.deepcopy(a)
    changed["train"][0]["hero"]["candidates"] = ["fold", "raise", "call"]
    with pytest.raises(ValueError, match="candidate"):
        validate_sample(changed)


def test_ragged_candidate_metrics_and_invalid_predictions():
    result = metrics([1, 2], [[0.1, 0.9], [0.1, 0.2, 0.7]])
    assert result["accuracy"] == 1
    assert result["log_loss"] == pytest.approx(-np.log([0.9, 0.7]).mean())
    for targets, p in (
        ([0], [[float("nan"), 0.5]]),
        ([0], [[-0.1, 1.1]]),
        ([2], [[0.5, 0.5]]),
        ([0], [[0.4, 0.4]]),
    ):
        with pytest.raises(ValueError):
            metrics(targets, p)


def test_forecast_global_to_local_mapping_and_saved_report(tmp_path):
    pytest.importorskip("scipy")
    data, manifest = saved(tmp_path)
    args = argparse.Namespace(out=tmp_path)
    forecast(args)
    result = json.loads((tmp_path / "forecaster_comparison.json").read_text())
    ids = [r["source_id"] for r in data["validation"]]
    for name in ("without_history", "with_history"):
        pred = result["predictions"][name]
        assert pred["source_ids"] == ids
        for row, p in zip(data["validation"], pred["probabilities"], strict=True):
            assert len(p) == len(row["hero"]["candidates"])
            assert sum(p) == pytest.approx(1)
    # A second paired comparison exercises report alignment without fitting a neural model.
    write_json(tmp_path / "decision_comparison.json", result)
    report(args)
    assert (tmp_path / "explorer.html").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["protocol"] == manifest
    result["predictions"]["with_history"]["source_ids"].reverse()
    write_json(tmp_path / "decision_comparison.json", result)
    with pytest.raises(ValueError, match="aligned"):
        report(args)


def test_scaling_keeps_original_rows_nested_training_and_fixed_validation():
    from poker_decision.holdem_scaling import expanded_samples

    source = prepared()
    reference = shared_sample(source, train_size=8, validation_size=4)
    samples = expanded_samples(source, reference, train_sizes=[8, 16, 32], validation_size=10)
    more = expanded_samples(source, reference, train_sizes=[8, 16, 40], validation_size=12)
    original_train = [r["source_id"] for r in reference["train"]]
    original_validation = [r["source_id"] for r in reference["validation"]]
    for count, data in samples.items():
        ids = [r["source_id"] for r in data["train"]]
        assert ids[:8] == original_train
        assert ids == [r["source_id"] for r in more[40]["train"][:count]]
        assert data["validation"] == samples[8]["validation"]
        assert data["validation"] == more[8]["validation"][:10]
        assert [r["source_id"] for r in data["validation"][:4]] == original_validation
        validate_sample(data)
    with pytest.raises(ValueError, match="Sizes"):
        expanded_samples(source, reference, train_sizes=[7], validation_size=10)
    changed = copy.deepcopy(reference)
    changed["train"][0]["hero"]["context"] += " changed input"
    with pytest.raises(ValueError, match="differ"):
        expanded_samples(source, changed, train_sizes=[16], validation_size=10)


def test_paired_scaling_intervals_preserve_zero_and_direction():
    from poker_decision.holdem_scaling import paired_change

    data = shared_sample(prepared(), train_size=8, validation_size=4)
    targets = [r["hero"]["target_index"] for r in data["validation"]]
    a, b = [], []
    for row, target in zip(data["validation"], targets, strict=True):
        size = len(row["hero"]["candidates"])
        a.append([1 / size] * size)
        p = [0.1 / (size - 1)] * size
        p[target] = 0.9
        b.append(p)
    same = {"predictions": {v: {"probabilities": a} for v in ("without_history", "with_history")}}
    assert paired_change(data, same)["log_loss"]["interval_95"] == [0.0, 0.0]
    changed = {
        "predictions": {
            "without_history": {"probabilities": a},
            "with_history": {"probabilities": b},
        }
    }
    assert paired_change(data, changed)["log_loss"]["estimate"] < 0
