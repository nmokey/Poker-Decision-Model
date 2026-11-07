"""Project entry points and dataset compatibility need no neural dependencies."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from poker_decision.paths import matrix_directory, prepared_directory, project_root
from poker_decision.pipeline import prepare
from poker_decision.prepared import load_prepared


def test_legacy_prepared_dataset_is_reused_without_rewriting(tmp_path, monkeypatch):
    from test_pipeline import source

    source(tmp_path, monkeypatch)
    expected = prepare(tmp_path, kinds=("preflop",), sample_per_kind=None)
    canonical = prepared_directory(tmp_path)
    assert canonical == tmp_path / "data/processed/pokerbench"
    legacy = canonical.with_name("pokerbench_meeting1")
    canonical.rename(legacy)
    assert prepared_directory(tmp_path) == legacy
    assert load_prepared(tmp_path) == expected
    # An invalid canonical dataset must fail instead of silently using the old split.
    canonical.mkdir()
    (canonical / "manifest.json").write_text("{}")
    assert prepared_directory(tmp_path) == canonical
    with pytest.raises(KeyError):
        load_prepared(tmp_path)


def test_root_discovery_and_unified_help():
    root = Path(__file__).resolve().parents[1]
    assert project_root(root / "notebooks") == root
    result = subprocess.run(
        [sys.executable, "-m", "poker_decision", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(task in result.stdout for task in ["prepare", "train", "evaluate", "opponents"])
    result = subprocess.run(
        [sys.executable, "-m", "poker_decision", "prepare", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--sample-per-kind" in result.stdout


def test_notebooks_are_task_clients_with_one_setup():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "notebooks").glob("*.ipynb"))
    assert {p.stem for p in paths} == {
        "data_pipeline",
        "decision_model",
        "evaluation",
        "data_sources",
        "opponent_modeling",
    }
    for path in paths:
        notebook = json.loads(path.read_text())
        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and cell["outputs"] == []
                source = "".join(cell["source"])
                compile(source, str(path), "exec")
                code.append(source)
        assert sum("notebook_setup.py" in source for source in code) == 1
        assert not any("pip" in source and "install" in source for source in code)
        assert "train_epoch(" not in "\n".join(code)


def test_shared_setup_from_notebook_directory():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = (
        "import runpy; from pathlib import Path; "
        "root = Path.cwd().parent; "
        "setup = runpy.run_path(str(root / 'scripts/notebook_setup.py'), "
        "init_globals={'ROOT': root, 'EXTRAS': 'analysis'}); "
        "assert Path.cwd() == root and not setup['IN_COLAB']"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root / "notebooks", env=env, check=True)


def test_source_adapters_keep_targets_groups_and_semantics_separate():
    from poker_decision.tasks import source_task

    observed = {
        "source_id": "hand-1-action-0",
        "hand_id": "hand-1",
        "street": "flop",
        "inputs": {"context": "visible state", "candidates": ["fold", "check"]},
        "target_index": 1,
    }
    teacher = {**observed, "episode": 1, "target_probabilities": [0.3, 0.7]}
    replay = source_task([observed], source="pluribus")
    solver = source_task([teacher], source="leduc_teacher")
    assert replay.examples[0].inputs == solver.examples[0].inputs
    assert replay.label_semantics != solver.label_semantics
    assert replay.question != solver.question
    assert replay.groups == ("hand-1",) and solver.groups == ("1",)
    assert solver.examples[0].target_index == teacher["target_index"]
    assert "target_probabilities" not in solver.examples[0].inputs.context


def test_opponent_adapter_reuses_causal_features_without_identity_leakage():
    import copy

    from test_opponents import row

    from poker_decision.opponents.data import FeatureSpec
    from poker_decision.tasks import opponent_task

    rows = [row(1, hand=0), row(2, hand=1), row(0, hand=2)]
    spec = FeatureSpec(window=16)
    task = opponent_task(rows, spec=spec)
    changed = copy.deepcopy(rows)
    changed[1]["target_index"] = 0
    alternate = opponent_task(changed, spec=spec)
    assert task.examples[1].inputs == alternate.examples[1].inputs
    assert task.examples[2].inputs != alternate.examples[2].inputs
    changed = copy.deepcopy(rows)
    for record in changed:
        record.update(
            opponent_id="hidden identity",
            session_id="hidden session",
            profile="hidden style",
            switch_hand=123,
        )
    renamed = opponent_task(changed, spec=spec)
    assert [e.inputs for e in renamed.examples] == [e.inputs for e in task.examples]
    assert renamed.groups == ("hidden identity",) * 3
    assert task.question == "What action will the opponent take next?"
    assert task.examples[0].inputs.candidates == tuple(rows[0]["inputs"]["candidates"])
    with pytest.raises(ValueError, match="chronology"):
        opponent_task(list(reversed(rows)), spec=spec)


def test_matrix_checkpoint_location_retains_legacy_compatibility(tmp_path):
    canonical = tmp_path / "artifacts/decision_matrix"
    legacy = tmp_path / "artifacts/week2_matrix"
    assert matrix_directory(tmp_path) == canonical
    legacy.mkdir(parents=True)
    assert matrix_directory(tmp_path) == legacy
    canonical.mkdir()
    assert matrix_directory(tmp_path) == canonical
