"""Adversarial checks for data isolation, grouping and interpretable statistics."""

import json
from dataclasses import replace

import pytest

from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.diagnostics import (
    grouped_interval,
    menu_information,
    paired_comparison,
    prediction_diagnostics,
)
from poker_decision.scenarios import make_folds, scenario_group, scenario_overlap


def scenario_rows(n=30):
    rows = []
    for i in range(n):
        context = (
            f"Street: preflop. Hero position: HJ. Hero cards: Ah Kd. "
            f"Previous actions: UTG: raise {i // 2 + 2}bb. "
            "Source num_players: 6. Source num_bets: 1. Pot: 4 bb."
        )
        rows.append(
            DecisionExample(str(i), "preflop", DecisionInputs(context, ("fold", "call")), i % 2)
        )
    return rows


def test_scenario_key_ignores_labels_menus_suits_and_later_streets():
    pre = scenario_rows()[0]
    changed = replace(
        pre,
        source_id="other",
        target_index=0,
        inputs=DecisionInputs(pre.inputs.context.replace("Ah Kd", "2c 3s"), ("call", "raise 4bb")),
    )
    assert scenario_group(pre) == scenario_group(changed)
    post = DecisionExample(
        "p",
        "flop",
        DecisionInputs(
            "Board: Ah Kd 2c. Preflop actions: HJ: raise 2bb. Postflop actions: none.",
            ("check", "bet 2bb"),
        ),
        0,
    )
    river = replace(
        post,
        street="river",
        target_index=1,
        source_id="r",
        inputs=DecisionInputs(
            "Board: 2s Ac Kh Jd Tc. Preflop actions: BTN: raise 3bb. Postflop actions: IP: check.",
            ("fold", "call"),
        ),
    )
    assert scenario_group(post) == scenario_group(river)
    with pytest.raises(ValueError, match="Cannot identify"):
        scenario_group(replace(pre, inputs=DecisionInputs("unknown state", ("fold",))))


def test_folds_cover_every_row_once_keep_conflicting_duplicates_together():
    pytest.importorskip("sklearn")
    rows = scenario_rows()
    rows += [replace(rows[0], source_id="duplicate", target_index=1)]
    pool, folds, protocol = make_folds(rows, n_splits=3)
    again = make_folds(list(reversed(rows)), n_splits=3)
    assert again == (pool, folds, protocol)
    held = []
    for left, right in folds:
        fit, test = [pool[i] for i in left], [pool[i] for i in right]
        assert scenario_overlap(fit, test)["shared_groups"] == 0
        assert scenario_overlap(fit, test)["exact_context_overlap"] == 0
        assert {e.source_id for e in fit}.isdisjoint(e.source_id for e in test)
        held.extend(right)
        if any(e.source_id == "0" for e in test):
            assert any(e.source_id == "duplicate" for e in test)
    assert sorted(held) == list(range(len(rows)))
    with pytest.raises(ValueError, match="Official test"):
        make_folds([replace(rows[0], source_id="preflop-test-0")] + rows[1:], n_splits=3)
    with pytest.raises(ValueError, match="unique"):
        make_folds(rows + [rows[0]], n_splits=3)
    with pytest.raises(ValueError, match="Fewer scenario"):
        make_folds(rows[:2], n_splits=3)


def test_reliability_and_error_counts_have_known_answers():
    examples = [
        DecisionExample("sizing", "preflop", DecisionInputs("a", ("raise 2bb", "raise 3bb")), 1),
        DecisionExample("type", "preflop", DecisionInputs("b", ("fold", "call")), 0),
        DecisionExample("forced", "preflop", DecisionInputs("c", ("check",)), 0),
    ]
    values = {e.inputs: p for e, p in zip(examples, [[0.8, 0.2], [0.25, 0.75], [1]], strict=True)}
    report = prediction_diagnostics(examples, values.__getitem__, bins=2)
    assert report["same_action_type_sizing_errors"] == 1
    assert report["different_action_type_errors"] == 1
    assert sum(b["n"] for b in report["reliability"]) == 3
    assert report["ece"] == pytest.approx((0.8 + 0.75) / 3)
    assert report["reliability"][0]["accuracy"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="Invalid candidate"):
        prediction_diagnostics(examples, lambda x: [float("nan")])


def test_bootstrap_resamples_whole_clusters_and_paired_differences():
    # If this sampled rows instead of clusters, the interval would not be degenerate.
    result = grouped_interval([0, 1], ["one_cluster", "one_cluster"], repetitions=100)
    assert result == {"mean": 0.5, "low95": 0.5, "high95": 0.5}
    rows = scenario_rows(10)
    same = paired_comparison(rows, lambda x: [0.5, 0.5], lambda x: [0.5, 0.5], repetitions=100)
    for metric in ["accuracy_difference", "log_loss_difference"]:
        assert same[metric] == {"mean": 0, "low95": 0, "high95": 0}
    assert grouped_interval([0, 1], ["a", "b"], repetitions=100) == grouped_interval(
        [0, 1], ["a", "b"], repetitions=100
    )
    with pytest.raises(ValueError):
        grouped_interval([float("inf")], ["a"])


def test_menu_information_is_seeded_and_uses_only_passed_training_rows():
    pytest.importorskip("sklearn")
    rows = [
        DecisionExample(
            str(i),
            "preflop",
            DecisionInputs(str(i), ("fold", "call") if i < 20 else ("check", "bet 2bb")),
            0,
        )
        for i in range(40)
    ]
    report = menu_information(rows, permutations=19)
    assert report["unique_training_contexts"] == 40
    assert report["conditional_mi_bits"] == pytest.approx(1)
    assert report == menu_information(rows, permutations=19)
    assert 0 < report["permutation_p"] <= 1


@pytest.mark.parametrize(
    "field,value",
    [("target_index", 1), ("context", "poison"), ("candidates", ["fold", "raise 999bb"])],
)
def test_saved_split_content_tampering_is_rejected(tmp_path, monkeypatch, field, value):
    pytest.importorskip("torch")
    from test_pipeline import source

    from poker_decision.pipeline import prepare
    from poker_decision.training import load_prepared

    source(tmp_path, monkeypatch)
    prepare(tmp_path, kinds=("preflop",), sample_per_kind=None)
    path = tmp_path / "data/processed/pokerbench_meeting1/train.jsonl"
    records = [json.loads(s) for s in path.read_text().splitlines()]
    records[0][field] = value
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(tmp_path)


def test_legacy_split_must_match_checksum_verified_source_rows(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import hashlib
    import shutil

    from test_pipeline import source

    from poker_decision import training
    from poker_decision.data import TRAIN_FILES
    from poker_decision.pipeline import prepare

    source(tmp_path, monkeypatch)
    expected = prepare(tmp_path, kinds=("preflop",), sample_per_kind=None)
    directory = tmp_path / "data/processed/pokerbench_meeting1"
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["split_sha256"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Legacy split requires pinned raw"):
        training.load_prepared(tmp_path)
    # Substitute a controlled pinned source fixture; no download occurs.
    raw = tmp_path / "data/raw/pokerbench" / TRAIN_FILES["preflop"]
    raw.parent.mkdir(parents=True)
    shutil.copyfile(tmp_path / "fixture.csv", raw)
    fixture_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    monkeypatch.setattr(training, "SHA256", {"preflop": fixture_hash})
    manifest["sources"]["preflop"]["sha256"] = fixture_hash
    path.write_text(json.dumps(manifest))
    assert training.load_prepared(tmp_path)[:2] == expected[:2]
    split_path = directory / "train.jsonl"
    records = [json.loads(line) for line in split_path.read_text().splitlines()]
    records[0]["target_index"] = 1
    split_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(ValueError, match="contents differ from pinned source"):
        training.load_prepared(tmp_path)
