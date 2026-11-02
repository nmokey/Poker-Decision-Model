"""Check matrix preflight membership and abort behavior without pretrained downloads."""

import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from test_scientific_integrity import scenario_rows  # noqa: E402

from poker_decision import experiments  # noqa: E402


@pytest.mark.parametrize("accuracy", [1.0, 0.0])
def test_runner_checks_training_only_and_starts_cv_after_both_heads(
    tmp_path, monkeypatch, accuracy
):
    rows = scenario_rows(100)
    pool, folds, _ = experiments.make_folds(rows, n_splits=5, seed=42)
    permitted = {pool[i].source_id for i in folds[0][0]}
    forbidden = {pool[i].source_id for i in folds[0][1]}
    manifest = tmp_path / "data/processed/pokerbench_meeting1/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiments",
            "--backbone",
            "bert_mini",
            "--root",
            str(tmp_path),
            "--output",
            str(tmp_path / "results"),
        ],
    )
    monkeypatch.setattr(experiments.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(experiments.torch.cuda, "get_device_name", lambda _: "test GPU")
    monkeypatch.setattr(experiments.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(experiments, "seed_everything", lambda _: None)
    monkeypatch.setattr(experiments, "load_prepared", lambda _: (rows[:80], rows[80:], {}))
    heads = []
    model = SimpleNamespace(to=lambda _: model, parameters=lambda: [])

    def load(*args, head_type, **kwargs):
        heads.append(head_type)
        return SimpleNamespace(pad_token_id=0), model

    monkeypatch.setattr(experiments, "load_pretrained", load)
    monkeypatch.setattr(experiments.torch.optim, "AdamW", lambda *a, **k: object())

    def encode(examples, *args, **kwargs):
        ids = {e.source_id for e in examples}
        assert len(examples) == len(ids) == 32
        assert ids <= permitted and ids.isdisjoint(forbidden)
        return examples

    monkeypatch.setattr(experiments, "encode_examples", encode)
    monkeypatch.setattr(experiments, "train_epoch", lambda *a, **k: 0.5)
    monkeypatch.setattr(
        experiments, "score_model", lambda *a, **k: {"overall": {"exact_accuracy": accuracy}}
    )
    calls = []

    def run(examples, output, *, config, device):
        assert heads == ["linear", "mlp"]
        assert examples == pool and device == "cuda" and config.epochs == 3
        calls.append(config)
        return {
            "output_directory": str(output),
            "models": {"transformer_linear": {"exact_accuracy": 0.5}},
        }

    monkeypatch.setattr(experiments, "run_cross_validation", run)
    if accuracy == 0:
        with pytest.raises(RuntimeError, match="failed training-only learning check"):
            experiments.main()
        assert not calls
    else:
        experiments.main()
        assert len(calls) == 1
        assert list((tmp_path / "results").rglob("metrics.csv"))
    checks = list((tmp_path / "results").rglob("tiny-*.json"))
    assert len(checks) == (2 if accuracy else 1)
    for path in checks:
        saved = json.loads(path.read_text())
        assert set(saved["source_ids"]) <= permitted
