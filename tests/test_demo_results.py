import csv
import json
from pathlib import Path

import pytest

from poker_decision.demo_models import load_catalog
from poker_decision.demo_results import load_results

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "docs/results/week2-development.csv"
LATENCY = ROOT / "docs/results/week2-serving-latency.json"


def test_saved_matrix_joins_all_eight_exact_checkpoints():
    _, models = load_catalog(ROOT / "deploy/streamlit/assets.json")
    rows = load_results(QUALITY, LATENCY, models)
    assert {r.model_id for r in rows} == set(models)
    modern = next(r for r in rows if r.model_id == "modernbert-linear")
    assert modern.accuracy == 0.7192
    assert modern.log_loss == pytest.approx(0.6077869381413451)
    assert modern.n == 10000
    assert all(r.mean_ms > 0 for r in rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("asset_sha256", "stale"),
        ("protocol_sha256", "stale"),
        ("head", "wrong"),
        ("n_measurements", 1),
        ("mean_ms", 0),
    ],
)
def test_mismatched_or_incomplete_timings_are_rejected(tmp_path, field, value):
    _, models = load_catalog(ROOT / "deploy/streamlit/assets.json")
    report = json.loads(LATENCY.read_text())
    report["models"]["modernbert-linear"][field] = value
    path = tmp_path / "latency.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_results(QUALITY, path, models)


def test_duplicate_quality_row_cannot_silently_override_score(tmp_path):
    _, models = load_catalog(ROOT / "deploy/streamlit/assets.json")
    with QUALITY.open() as file:
        rows = list(csv.DictReader(file))
    path = tmp_path / "quality.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([*rows, rows[-1]])
    with pytest.raises(ValueError, match="duplicate"):
        load_results(path, LATENCY, models)
