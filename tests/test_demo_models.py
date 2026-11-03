import json
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

import pytest

from poker_decision.demo_models import (
    ModelSpec,
    SingleModelStore,
    checkpoint_examples,
    load_catalog,
)


def fixture_assets(tmp_path):
    spec = ModelSpec(
        "bert_mini-linear",
        "bert_mini",
        "linear",
        "google/bert_uncased_L-4_H-256_A-4",
        "https://example.com/model.tar.gz",
        "a" * 64,
        1,
        1,
        "protocol",
    )
    (tmp_path / "model").mkdir()
    metadata = {
        "backbone": spec.backbone,
        "head_type": spec.head,
        "fold": 1,
        "protocol_sha256": "protocol",
        "test_source_ids": ["held"],
        "train_source_ids": ["fit"],
    }
    (tmp_path / "model/metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "manifest.json").write_text(json.dumps({**metadata, "n_hands": 1}))
    row = {
        "source_id": "held",
        "street": "flop",
        "context": "visible state",
        "candidates": ["fold", "call"],
        "target_index": 1,
    }
    (tmp_path / "hands.jsonl").write_text(json.dumps(row) + "\n")
    return spec, metadata, row


def test_selection_rejects_mislabeled_checkpoint_or_training_hand(tmp_path):
    spec, metadata, row = fixture_assets(tmp_path)
    assert checkpoint_examples(tmp_path, spec)[0].source_id == "held"
    with pytest.raises(ValueError, match="does not match"):
        checkpoint_examples(tmp_path, replace(spec, head="mlp"))
    row["source_id"] = "fit"
    (tmp_path / "hands.jsonl").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="not held out"):
        checkpoint_examples(tmp_path, spec)
    metadata["train_source_ids"].append("held")
    (tmp_path / "model/metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="overlap"):
        checkpoint_examples(tmp_path, spec)


def test_catalogue_cannot_label_bert_weights_as_modernbert(tmp_path):
    spec, _, _ = fixture_assets(tmp_path)
    record = asdict(spec)
    record.pop("key")
    path = tmp_path / "catalogue.json"
    catalogue = {"schema": 2, "default_model": spec.key, "models": {spec.key: record}}
    path.write_text(json.dumps(catalogue))
    assert load_catalog(path)[1][spec.key].label == "BERT Mini · Linear"
    record["encoder"] = "modernbert"
    catalogue["default_model"] = "modernbert-linear"
    catalogue["models"] = {"modernbert-linear": record}
    path.write_text(json.dumps(catalogue))
    with pytest.raises(ValueError, match="specification"):
        load_catalog(path)


@dataclass
class FakeModel:
    key: str


def test_switch_evicts_before_loading_and_failed_load_cannot_reuse_old_model():
    references = []
    loaded = []

    def load(key):
        assert not any(ref() is not None for ref in references)
        loaded.append(key)
        if key == "bad":
            raise ValueError("missing weights")
        model = FakeModel(key)
        references.append(weakref.ref(model))
        return model

    store = SingleModelStore(load)
    assert store.apply("one", lambda m: m.key) == "one"
    assert store.apply("one", lambda m: m.key) == "one"
    assert loaded == ["one"]
    assert store.apply("two", lambda m: m.key) == "two"
    with pytest.raises(ValueError, match="missing"):
        store.warm("bad")
    assert store.apply("one", lambda m: m.key) == "one"
    assert loaded == ["one", "two", "bad", "one"]


def test_concurrent_sessions_never_predict_with_another_selected_model():
    active = 0
    peak = 0

    def load(key):
        return FakeModel(key)

    store = SingleModelStore(load)

    def request(key):
        def predict(model):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(0.005)
            assert model.key == key
            active -= 1
            return model.key

        return store.apply(key, predict)

    keys = ["linear", "mlp"] * 4
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(request, keys)) == keys
    assert peak == 1
