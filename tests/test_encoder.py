"""Offline neural contract checks with a tiny randomly initialized BERT."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from poker_decision.data import DecisionExample, DecisionInputs  # noqa: E402
from poker_decision.models.encoder import (  # noqa: E402
    PokerDecisionModel,
    collate_decisions,
    encode_inputs,
    model_input,
)
from poker_decision.training import (  # noqa: E402
    encode_examples,
    load_checkpoint,
    load_prepared,
    predict_encoded,
    save_checkpoint,
    seed_everything,
    train_epoch,
)


@pytest.fixture
def tokenizer(tmp_path):
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "state",
        ":",
        "question",
        "decisions",
        "what",
        "should",
        "hero",
        "do",
        "?",
        "strong",
        "weak",
        "call",
        "fold",
        "raise",
        "12",
        "##bb",
        "check",
    ]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocabulary) + "\n")
    return transformers.BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)


def make_model(tokenizer, head_type="linear"):
    config = transformers.BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    return PokerDecisionModel(transformers.BertModel(config), head_type=head_type)


def rows():
    return [
        DecisionExample("a", "preflop", DecisionInputs("strong", ("call", "fold")), 0),
        DecisionExample("b", "flop", DecisionInputs("weak", ("fold", "call", "raise 12bb")), 0),
        DecisionExample("c", "turn", DecisionInputs("weak", ("check",)), 0),
    ]


def test_answer_free_serialization_and_complete_spans(tokenizer):
    e = rows()[1]
    assert set(model_input(e.inputs)) == {"state", "question", "decisions"}
    encoded = encode_inputs(e.inputs, tokenizer, max_length=128)
    changed_target = DecisionExample(e.source_id, e.street, e.inputs, 1)
    assert encode_inputs(changed_target.inputs, tokenizer) == encoded
    for action, (start, end) in zip(e.inputs.candidates, encoded.candidate_spans, strict=True):
        assert list(encoded.input_ids[start:end]) == tokenizer.encode(
            action, add_special_tokens=False
        )
    with pytest.raises(ValueError, match="No truncation"):
        encode_inputs(e.inputs, tokenizer, max_length=8)
    with pytest.raises(ValueError, match="tokenization failures"):
        encode_examples([e], tokenizer, max_length=8)


def test_variable_candidates_padding_gradients_and_shared_head(tokenizer):
    seed_everything(42)
    model = make_model(tokenizer)
    examples = rows()
    encoded = encode_examples(examples, tokenizer, max_length=128)
    batch = collate_decisions(encoded, pad_token_id=tokenizer.pad_token_id)
    logits = model(**batch)
    assert logits.shape == (3, 3)
    probs = logits.softmax(-1)
    assert probs[0, 2].item() == 0
    assert probs[2].tolist() == [1, 0, 0]
    torch.testing.assert_close(probs.sum(-1), torch.ones(3))
    alone = model(**collate_decisions([encoded[0]], pad_token_id=tokenizer.pad_token_id))
    torch.testing.assert_close(logits[0, :2], alone[0])
    # Reordering extraction slots must apply the same head to each representation.
    swapped = dict(batch)
    swapped["candidate_token_mask"] = batch["candidate_token_mask"][:, [1, 0, 2]]
    torch.testing.assert_close(model(**swapped), logits[:, [1, 0, 2]])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0, 0]))
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    gradient = model.encoder.embeddings.word_embeddings.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    batch["candidate_token_mask"][0, 0, -1] = True
    with pytest.raises(ValueError, match="padding"):
        model(**batch)


@pytest.mark.parametrize("head_type", ["linear", "mlp"])
def test_learning_and_local_checkpoint_roundtrip(tokenizer, tmp_path, head_type):
    torch.set_num_threads(1)
    seed_everything(42)
    model = make_model(tokenizer, head_type)
    examples = rows()[:2]
    encoded = encode_examples(examples, tokenizer, max_length=128)
    batch = collate_decisions(encoded, pad_token_id=tokenizer.pad_token_id)
    targets = torch.tensor([0, 0])
    initial = torch.nn.functional.cross_entropy(model(**batch), targets).item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for step in range(20):
        train_epoch(model, encoded, [0, 0], pad_token_id=0, optimizer=optimizer, seed=step)
    model.eval()
    final = torch.nn.functional.cross_entropy(model(**batch), targets).item()
    assert final < initial * 0.5
    before = predict_encoded(model, encoded, pad_token_id=0)
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model, tokenizer, {"question": "What should Hero do?", "seed": 42})
    restored_tokenizer, restored_model, metadata = load_checkpoint(path)
    assert metadata["seed"] == 42
    assert restored_model.head_type == head_type
    assert metadata["head_type"] == head_type
    assert encode_examples(examples, restored_tokenizer, max_length=128) == encoded
    if head_type == "linear":
        # Earlier checkpoints did not persist a head type; their linear weights
        # must still reload without silently changing the architecture.
        old_metadata = {k: v for k, v in metadata.items() if k not in {"head_type", "head_width"}}
        (path / "metadata.json").write_text(json.dumps(old_metadata))
        _, legacy_model, _ = load_checkpoint(path)
        assert legacy_model.head_type == "linear"
        torch.testing.assert_close(legacy_model.head.weight, restored_model.head.weight)
    after = predict_encoded(restored_model, encoded, pad_token_id=0)
    torch.testing.assert_close(torch.tensor(before[0]), torch.tensor(after[0]))
    torch.testing.assert_close(torch.tensor(before[1]), torch.tensor(after[1]))
    with pytest.raises(ValueError, match="Target outside"):
        train_epoch(model, encoded, [2, 0], pad_token_id=0, optimizer=optimizer)


def test_saved_membership_and_provenance(tmp_path, monkeypatch):
    from test_pipeline import source

    from poker_decision.pipeline import prepare

    source(tmp_path, monkeypatch)
    expected = prepare(tmp_path, kinds=("preflop",), sample_per_kind=None)
    assert load_prepared(tmp_path) == expected
    path = tmp_path / "data/processed/pokerbench_meeting1/train.jsonl"
    data = path.read_text().splitlines()
    data.reverse()
    path.write_text("\n".join(data) + "\n")
    with pytest.raises(ValueError, match="membership"):
        load_prepared(tmp_path)
    path.write_text("\n".join(reversed(data)) + "\n")
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["official_test_used"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training-only"):
        load_prepared(tmp_path)


def test_notebook_is_cleared_and_valid():
    path = Path(__file__).parents[1] / "notebooks/02_pokerbench_decision_model.ipynb"
    nb = json.loads(path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(path), "exec")


def test_evaluation_does_not_update_weights_or_accumulate_gradients(tokenizer):
    seed_everything(42)
    model = make_model(tokenizer)
    examples = rows()
    encoded = encode_examples(examples, tokenizer, max_length=128)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    predictions = predict_encoded(model, encoded, pad_token_id=0)
    changed_labels = [
        DecisionExample(
            e.source_id, e.street, e.inputs, (e.target_index + 1) % len(e.inputs.candidates)
        )
        for e in examples
    ]
    assert encode_examples(changed_labels, tokenizer, max_length=128) == encoded
    assert predict_encoded(model, encoded, pad_token_id=0) == predictions
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize(
    "name",
    [
        "02_pokerbench_decision_model.ipynb",
        "03_scenario_evaluation.ipynb",
        "04_alternative_data_sources.ipynb",
    ],
)
def test_notebook_syntax_schema_outputs_and_colab_training_guard(name):
    import ast

    import nbformat

    path = Path(__file__).parents[1] / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    sources = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []
            ast.parse(cell.source)
            sources.append(cell.source)
    if name.startswith("02"):
        training_cells = [s for s in sources if "train_epoch(" in s]
        assert len(training_cells) == 2
        assert all("if not IN_COLAB or not torch.cuda.is_available():" in s for s in training_cells)
    elif name.startswith("03"):
        assert any("RUN_CV = False" in s for s in sources)
        assert any("if not IN_COLAB or not torch.cuda.is_available():" in s for s in sources)
    else:
        assert any("RUN_LEDUC = False" in s and "RUN_PLURIBUS = False" in s for s in sources)


@pytest.mark.parametrize("architecture", ["bert", "distilbert", "modernbert"])
@pytest.mark.parametrize("head_type", ["linear", "mlp"])
def test_encoder_families_padding_gradients_and_checkpoint(
    tokenizer, tmp_path, architecture, head_type
):
    """Exercise actual architecture APIs offline, without training pretrained weights."""
    common = {"vocab_size": len(tokenizer), "max_position_embeddings": 128, "pad_token_id": 0}
    if architecture == "distilbert":
        config = transformers.DistilBertConfig(
            **common,
            dim=16,
            hidden_dim=32,
            n_layers=1,
            n_heads=2,
            dropout=0,
            attention_dropout=0,
        )
    elif architecture == "modernbert":
        config = transformers.ModernBertConfig(
            **common,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            local_attention=16,
            cls_token_id=2,
            sep_token_id=3,
            reference_compile=False,
        )
    else:
        config = make_model(tokenizer).encoder.config
    model = PokerDecisionModel(
        transformers.AutoModel.from_config(config), head_type=head_type, head_width=8
    ).eval()
    encoded = encode_examples(rows(), tokenizer, max_length=128)
    batch = collate_decisions(encoded, pad_token_id=0)
    logits = model(**batch)
    assert logits.shape == (3, 3)
    assert logits.softmax(-1)[2].tolist() == [1, 0, 0]
    assert torch.isfinite(logits[:, 0]).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0, 0])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    before = predict_encoded(model, encoded, pad_token_id=0)
    save_checkpoint(tmp_path / "model", model, tokenizer, {})
    _, restored, _ = load_checkpoint(tmp_path / "model")
    after = predict_encoded(restored, encoded, pad_token_id=0)
    assert restored.head_width == 8
    for expected, actual in zip(before, after, strict=True):
        torch.testing.assert_close(torch.tensor(expected), torch.tensor(actual))


def test_backbone_loader_pairs_pinned_tokenizer_and_encoder(monkeypatch):
    from poker_decision.models.encoder import BACKBONES, load_pretrained

    calls = []
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name, **kwargs: calls.append(("tokenizer", name, kwargs)) or "tokenizer",
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda name, **kwargs: (
            calls.append(("encoder", name, kwargs))
            or transformers.BertModel(
                transformers.BertConfig(
                    hidden_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=32,
                )
            )
        ),
    )
    assert set(BACKBONES) == {"bert_mini", "minilm", "distilbert", "modernbert"}
    for key, (name, revision) in BACKBONES.items():
        assert len(revision) == 40
        _, model = load_pretrained(name, revision, head_type="mlp", head_width=8)
        assert calls[-2] == ("tokenizer", name, {"revision": revision})
        expected = {"revision": revision}
        if key == "modernbert":
            expected.update(attn_implementation="sdpa", reference_compile=False)
        assert calls[-1] == ("encoder", name, expected)
        assert model.head_width == 8


def test_notebook_optimizer_uses_visible_knobs():
    import ast

    path = Path(__file__).parents[1] / "notebooks/02_pokerbench_decision_model.ipynb"
    notebook = json.loads(path.read_text())
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "optimizer = torch.optim.AdamW(" in "".join(c["source"])
        and "metadata =" in "".join(c["source"])
    )
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "optimizer" for t in node.targets)
    )
    model = PokerDecisionModel(
        transformers.BertModel(
            transformers.BertConfig(
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
            )
        )
    )
    scope = dict(torch=torch, model=model, ENCODER_LR=3e-5, HEAD_LR=4e-4, WEIGHT_DECAY=0.12)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), scope)
    optimizer = scope["optimizer"]
    assert [group["lr"] for group in optimizer.param_groups] == [3e-5, 4e-4]
    assert all(group["weight_decay"] == 0.12 for group in optimizer.param_groups)
