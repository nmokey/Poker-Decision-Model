"""Exercise the actual CV runner offline with tiny fresh random BERTs and fit spies."""

import json
from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

import torch  # noqa: E402
from test_encoder import make_model, tokenizer  # noqa: E402, F401
from test_scientific_integrity import scenario_rows  # noqa: E402

from poker_decision import cross_validation as cv  # noqa: E402
from poker_decision.models.encoder import PokerDecisionModel  # noqa: E402


def test_all_fits_exclude_fold_test_and_each_model_starts_fresh(
    tokenizer,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    torch.set_num_threads(1)
    rows = scenario_rows(30)
    rows.append(replace(rows[0], source_id="extra"))
    pool, folds, protocol = cv.make_folds(rows, n_splits=3)
    baseline_memberships = []
    models, targets, encoded_memberships = [], [], []
    optimizers = []
    adamw = cv.torch.optim.AdamW

    def new_optimizer(*args, **kwargs):
        optimizer = adamw(*args, **kwargs)
        assert not optimizer.state
        optimizers.append(optimizer)
        return optimizer

    prior_fit, linear_fit = cv.OptionPrior.fit, cv.LinearCandidateScorer.fit
    encode, train_epoch = cv.encode_examples, cv.train_epoch

    def fit_prior(self, examples):
        baseline_memberships.append(("prior", [e.source_id for e in examples]))
        return prior_fit(self, examples)

    def fit_linear(self, examples):
        baseline_memberships.append(("linear", [e.source_id for e in examples]))
        return linear_fit(self, examples)

    def fresh(name, revision, *, head_type, head_width):
        # Persisted protocol must exist before the first model is fitted.
        assert list(tmp_path.rglob("protocol.json"))
        model = PokerDecisionModel(
            make_model(tokenizer).encoder, head_type=head_type, head_width=head_width
        )
        models.append(model)
        return tokenizer, model

    def encode_spy(examples, tokenizer, **kwargs):
        result = encode(examples, tokenizer, **kwargs)
        encoded_memberships.append(([e.source_id for e in examples], id(result)))
        return result

    def training_spy(model, encoded, y, **kwargs):
        ids = next(
            ids for ids, identity in reversed(encoded_memberships) if identity == id(encoded)
        )
        targets.append((ids, list(y)))
        return train_epoch(model, encoded, y, **kwargs)

    monkeypatch.setattr(cv.OptionPrior, "fit", fit_prior)
    monkeypatch.setattr(cv.LinearCandidateScorer, "fit", fit_linear)
    monkeypatch.setattr(cv, "load_pretrained", fresh)
    monkeypatch.setattr(cv, "encode_examples", encode_spy)
    monkeypatch.setattr(cv, "train_epoch", training_spy)
    monkeypatch.setattr(cv.torch.optim, "AdamW", new_optimizer)
    config = cv.CVConfig(heads=("linear", "mlp"), n_splits=3, epochs=1, batch_size=8)
    result = cv.run_cross_validation(rows, tmp_path, config=config)
    assert len(models) == 6 and len({id(m) for m in models}) == 6
    assert len(optimizers) == 6 and len({id(o) for o in optimizers}) == 6
    assert len({len(right) for _, right in folds}) > 1
    for i, spec in enumerate(protocol["folds"]):
        expected = spec["train_source_ids"]
        assert baseline_memberships[i * 2 : i * 2 + 2] == [
            ("prior", expected),
            ("linear", expected),
        ]
        for ids, y in targets[i * 2 : i * 2 + 2]:
            assert ids == expected
            assert set(ids).isdisjoint(spec["test_source_ids"])
            assert y == [pool[j].target_index for j in folds[i][0]]
    assert result["coverage"] == {"n": 31, "held_out_predictions_per_row": 1, "folds": 3}
    folder = tmp_path / result["output_directory"]
    assert set(result["models"]) == {
        "uniform",
        "option_prior",
        "linear_candidate",
        "transformer_linear",
        "transformer_mlp",
    }
    for model in result["models"].values():
        assert model["n"] == 31

    # A second run must not fit anything and must reproduce the result exactly.
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed valid fold was retrained")

    monkeypatch.setattr(cv, "load_pretrained", forbidden)
    monkeypatch.setattr(cv.OptionPrior, "fit", forbidden)
    assert cv.run_cross_validation(rows, tmp_path, config=config) == result
    report_path = folder / "fold-1/report.json"
    original = report_path.read_text()
    report = json.loads(original)
    report["metrics"]["uniform"]["overall"]["exact_accuracy"] = 999
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="metrics disagree"):
        cv.run_cross_validation(rows, tmp_path, config=config)
    report_path.write_text(original)
    predictions = folder / "fold-1/oof.jsonl"
    original_predictions = predictions.read_text()
    predictions.write_text(original_predictions + "\n")
    with pytest.raises(ValueError, match="predictions changed"):
        cv.run_cross_validation(rows, tmp_path, config=config)
    predictions.write_text(original_predictions)
    head_weights = folder / "fold-1/checkpoint-linear/head.pt"
    head_weights.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint files changed"):
        cv.run_cross_validation(rows, tmp_path, config=config)


def test_sweep_grid_is_bounded_validated_and_preserves_folds():
    base = cv.CVConfig(heads=("linear", "mlp"))
    trials = cv.sweep_configs(
        base,
        {
            "backbone": ["bert_mini", "minilm", "distilbert", "modernbert"],
            "encoder_lr": [1e-5, 2e-5],
        },
    )
    assert len(trials) == 8
    assert {(c.backbone, c.encoder_lr) for c in trials} == {
        (name, lr) for name in cv.BACKBONES for lr in (1e-5, 2e-5)
    }
    assert all(c.seed == base.seed and c.n_splits == base.n_splits for c in trials)
    assert base.backbone == "bert_mini" and base.encoder_lr == 2e-5
    for grid, message in [
        ({"seed": [1, 2]}, "stay fixed"),
        ({"n_splits": [3]}, "stay fixed"),
        ({"backbone": []}, "nonempty"),
        ({"backbone": "bert_mini"}, "nonempty"),
        ({"encoder_lr": [0]}, "positive"),
        ({"backbone": ["unknown"]}, "known"),
        ({"batch_size": list(range(1, 10))}, "budget"),
        ({"backbone": ["bert_mini", "bert_mini"]}, "Duplicate"),
    ]:
        with pytest.raises(ValueError, match=message):
            cv.sweep_configs(base, grid)
