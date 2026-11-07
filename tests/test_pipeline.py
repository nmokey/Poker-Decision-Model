"""Exercise saved data contracts and fail-closed preparation without network access."""

import csv
import json

import pytest

from poker_decision.pipeline import duplicate_report, inspect_source, prepare


def source(tmp_path, monkeypatch, *, broken=False):
    path = tmp_path / "fixture.csv"
    rows = [
        {
            "prev_line": "",
            "hero_pos": "UTG",
            "hero_holding": cards,
            "correct_decision": "fold",
            "num_players": "6",
            "num_bets": "0",
            "available_moves": "['fold', '2.0bb']",
            "pot_size": "1.5",
        }
        for cards in ["AhKd", "AsKc", "2c3d", "4s5h", "6h7c", "8s9d"]
    ]
    if broken:
        rows[-1]["correct_decision"] = "call"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr("poker_decision.pipeline.download_training", lambda *args: path)
    return rows


def test_prepare_saves_reproducible_membership(tmp_path, monkeypatch):
    source(tmp_path, monkeypatch)
    train, validation, manifest = prepare(tmp_path, kinds=("preflop",), sample_per_kind=None)
    assert len(train) + len(validation) == 6
    assert manifest["official_test_used"] is False
    saved = json.loads((tmp_path / "data/processed/pokerbench/manifest.json").read_text())
    assert saved == manifest
    assert saved["train_source_ids"] == [e.source_id for e in train]
    assert prepare(tmp_path, kinds=("preflop",), sample_per_kind=None)[:2] == (train, validation)
    assert duplicate_report(train + [train[0]])["repeated_exact_examples"] == 1


def test_prepare_stops_and_saves_all_failures(tmp_path, monkeypatch):
    source(tmp_path, monkeypatch, broken=True)
    with pytest.raises(ValueError, match="Resolve 1 parsing failures"):
        prepare(tmp_path, kinds=("preflop",))
    output = tmp_path / "data/processed/pokerbench"
    failures = [
        json.loads(line)
        for line in (output / "preflop_parse_failures.jsonl").read_text().splitlines()
    ]
    assert len(failures) == 1
    assert failures[0]["source_id"] == "preflop-train-5"
    assert not (output / "train.jsonl").exists()
    assert not (output / "manifest.json").exists()


def test_inspection_preserves_blanks_and_exact_label_counts(tmp_path, monkeypatch):
    rows = source(tmp_path, monkeypatch)
    inspection = inspect_source(rows)
    assert inspection["columns"]["prev_line"]["blank"] == 6
    assert inspection["columns"]["num_players"]["inferred_types"] == ["integer"]
    assert inspection["action_frequencies"] == {"fold": 6}
