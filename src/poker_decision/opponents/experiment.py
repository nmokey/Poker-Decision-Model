"""Reproducible generation, validation-only selection, and explicit final evaluation.

python -m poker_decision.opponents.experiment --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from poker_decision.opponents.data import (
    Config,
    FeatureSpec,
    canonical,
    feature_matrix,
    validate_dataset,
)
from poker_decision.opponents.evaluation import evaluate_predictions, per_row_metrics
from poker_decision.opponents.generate import generate, replay_audit
from poker_decision.opponents.models import Forecaster, bayesian_predictions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hash() -> str:
    return hashlib.sha256(
        canonical({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}).encode()
    ).hexdigest()


def versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for p in ("numpy", "open-spiel", "scikit-learn", "scipy"):
        try:
            result[p] = version(p)
        except PackageNotFoundError:
            result[p] = "not installed"
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if sha256(root / name) != checksum:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    return dict(manifest)


def prepare(cfg: Config, output: Path) -> Path:
    protocol = dict(
        config=asdict(cfg),
        implementation=implementation_hash(),
        versions=versions(),
        label_semantics="observed next opponent action; not a Hero decision target",
    )
    root = output / ("leduc-" + hashlib.sha256(canonical(protocol).encode()).hexdigest()[:12])
    if (root / "manifest.json").exists():
        manifest = verify_manifest(root)
        if manifest["protocol"] != protocol:
            raise ValueError("Saved protocol differs")
        return root
    if root.exists() and any(root.iterdir()):
        raise ValueError("Incomplete run directory; preserve it and choose another output root")
    dataset = generate(cfg)
    audit = replay_audit(dataset, cfg)
    for split in ("train", "validation", "test"):
        write_jsonl(root / f"{split}.jsonl", [r for r in dataset["rows"] if r["split"] == split])
    write_json(root / "sessions.json", dataset["sessions"])
    write_jsonl(root / "audit/hands.jsonl", dataset["traces"])
    write_json(root / "audit/teacher.json", dataset["teacher"])
    write_json(root / "audit/teacher_policy.json", dataset["teacher_policy"])
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(
        root / "manifest.json",
        dict(
            protocol=protocol,
            files=files,
            audit=audit,
            counts=validate_dataset(dataset["rows"], dataset["sessions"]),
        ),
    )
    return root


def specifications(timing: str) -> dict[str, tuple[str, FeatureSpec]]:
    return {
        "prior": ("prior", FeatureSpec(timing=timing)),
        "no_history": ("masked_logistic", FeatureSpec(timing=timing)),
        **{
            f"history_{w}": ("masked_logistic", FeatureSpec(window=w, timing=timing))
            for w in (4, 16, 64)
        },
        "history_all": ("masked_logistic", FeatureSpec(window=None, timing=timing)),
        "candidate_no_history": ("candidate_logistic", FeatureSpec(timing=timing)),
        "candidate_history": ("candidate_logistic", FeatureSpec(window=20, timing=timing)),
        "menu_no_history": ("menu_logistic", FeatureSpec(timing=timing)),
        "menu_history": (
            "menu_logistic",
            FeatureSpec(window=10, timing=timing, window_order="after_match", match_round=False),
        ),
        "naive_bayes_no_history": ("naive_bayes", FeatureSpec(timing=timing)),
        "naive_bayes_history": ("naive_bayes", FeatureSpec(window=16, timing=timing)),
    }


def select(
    root: Path,
    *,
    timing: str = "completed_hands",
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    selection_method: str = "validation",
) -> Path:
    """Only train/validation labels are deserialized. Choices persist before test inference."""
    manifest = verify_manifest(root)
    sessions = json.loads((root / "sessions.json").read_text())
    train, val = read_jsonl(root / "train.jsonl"), read_jsonl(root / "validation.jsonl")
    for rows, split in ((train, "train"), (val, "validation")):
        validate_dataset(rows, [s for s in sessions if s["split"] == split])
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError("Need positive finite C grid")
    if selection_method not in ("validation", "grouped_cv"):
        raise ValueError("Invalid selection method")
    settings = dict(
        selection_method=selection_method,
        timing=timing,
        c_grid=list(c_grid),
        implementation=implementation_hash(),
        dataset_manifest=sha256(root / "manifest.json"),
        versions=versions(),
    )
    out = root / ("forecast-" + hashlib.sha256(canonical(settings).encode()).hexdigest()[:12])
    if (out / "selection.json").exists():
        saved = json.loads((out / "selection.json").read_text())
        if saved["settings"] != settings or saved["model_sha256"] != sha256(out / "models.json"):
            raise ValueError("Selection cache mismatch")
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError("Incomplete selection directory; choose a new run")
    specs = specifications(timing)
    if manifest["protocol"]["config"]["observer_card"]:
        specs.update(
            strength=("masked_logistic", FeatureSpec(timing=timing, representation="strength")),
            strength_history=(
                "masked_logistic",
                FeatureSpec(window=16, timing=timing, representation="strength"),
            ),
        )
    models, scores, val_predictions = {}, [], {}
    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    with threadpool_limits(limits=1):
        for name, (kind, spec) in specs.items():
            xt, xv = feature_matrix(train, spec), feature_matrix(val, spec)
            candidates = []
            for c in (1.0,) if kind in ("prior", "naive_bayes") else c_grid:
                model = Forecaster(kind, c).fit(train, xt)
                probs = model.predict(val, xv)
                metrics = per_row_metrics(val, probs)
                score: dict[str, Any] = dict(
                    model=name,
                    c=c,
                    validation_log_loss=float(metrics["log_loss"].mean()),
                    validation_accuracy=float(metrics["accuracy"].mean()),
                )
                if selection_method == "grouped_cv" and kind not in ("prior", "naive_bayes"):
                    from sklearn.model_selection import GroupKFold

                    groups = np.asarray([r["opponent_id"] for r in train])
                    total_loss = 0.0
                    fold_scores = []
                    for fit_ids, held_ids in GroupKFold(5).split(xt, groups=groups):
                        fit_rows = [train[i] for i in fit_ids]
                        held_rows = [train[i] for i in held_ids]
                        fold_model = Forecaster(kind, c).fit(fit_rows, xt[fit_ids])
                        fold_probs = fold_model.predict(held_rows, xt[held_ids])
                        losses = per_row_metrics(held_rows, fold_probs)["log_loss"]
                        total_loss += float(losses.sum())
                        fold_scores.append(dict(n=len(held_ids), log_loss=float(losses.mean())))
                    score["training_group_cv_log_loss"] = total_loss / len(train)
                    score["folds"] = fold_scores
                scores.append(score)
                criterion = score.get("training_group_cv_log_loss", score["validation_log_loss"])
                candidates.append((criterion, c, model, probs))
            _, _, model, probs = min(candidates, key=lambda item: (item[0], item[1]))
            models[name] = dict(feature_spec=asdict(spec), model=model.to_dict())
            val_predictions[name] = probs
    summary_names = ["history_4", "history_16", "history_64", "history_all"]
    selected_history = min(
        summary_names,
        key=lambda name: float(per_row_metrics(val, val_predictions[name])["log_loss"].mean()),
    )
    bayes_scores = []
    for temperature in (0.5, 0.9, 1.3, 1.7):
        for hazard in (0.0, 0.005, 0.02, 0.07):
            probs, _ = bayesian_predictions(
                val,
                val_predictions["no_history"],
                temperature=temperature,
                hazard=hazard,
                timing=timing,
            )
            bayes_scores.append(
                dict(
                    temperature=temperature,
                    hazard=hazard,
                    validation_log_loss=float(per_row_metrics(val, probs)["log_loss"].mean()),
                )
            )
    best_bayes = min(
        bayes_scores, key=lambda r: (r["validation_log_loss"], r["temperature"], r["hazard"])
    )
    write_json(out / "models.json", models)
    write_json(
        out / "selection.json",
        dict(
            settings=settings,
            candidates=scores,
            bayesian_candidates=bayes_scores,
            selected_history=selected_history,
            bayesian=best_bayes,
            model_sha256=sha256(out / "models.json"),
            note=(
                "C selected by declared validation/grouped training CV method; "
                "History window and Bayes parameters use validation. Fits use training rows only."
            ),
        ),
    )
    return out


def final_evaluation(root: Path, selection: Path, *, repetitions: int = 1000) -> dict[str, Any]:
    manifest = verify_manifest(root)
    saved = json.loads((selection / "selection.json").read_text())
    if saved["settings"]["dataset_manifest"] != sha256(root / "manifest.json"):
        raise ValueError("Selection belongs to another dataset")
    if (
        saved["settings"]["implementation"] != implementation_hash()
        or saved["settings"]["versions"] != versions()
    ):
        raise ValueError("Implementation/runtime changed; create a new selection")
    if saved["model_sha256"] != sha256(selection / "models.json"):
        raise ValueError("Saved model checksum mismatch")
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    protocol = dict(selection_sha256=sha256(selection / "selection.json"), repetitions=repetitions)
    result_file = selection / "results.json"
    if result_file.exists():
        result = json.loads(result_file.read_text())
        if result["evaluation_protocol"] != protocol:
            raise ValueError("Final evaluation exists with different settings")
        for name, checksum in result["artifact_sha256"].items():
            if sha256(selection / name) != checksum:
                raise ValueError("Saved evaluation artifact checksum mismatch")
        return dict(result)
    rows = read_jsonl(root / "test.jsonl")
    sessions = json.loads((root / "sessions.json").read_text())
    validate_dataset(rows, [s for s in sessions if s["split"] == "test"])
    predictions = {}
    for name, entry in json.loads((selection / "models.json").read_text()).items():
        features = feature_matrix(rows, FeatureSpec(**entry["feature_spec"]))
        model = Forecaster.from_dict(entry["model"])
        probs = model.predict(rows, features)
        predictions[name] = probs
    predictions["history_selected"] = predictions[saved["selected_history"]]
    predictions["uniform"] = np.asarray(
        [
            [
                float(a in r["inputs"]["legal_action_ids"]) / len(r["inputs"]["legal_action_ids"])
                for a in range(3)
            ]
            for r in rows
        ]
    )
    predictions["bayesian"], trace = bayesian_predictions(
        rows,
        predictions["no_history"],
        temperature=saved["bayesian"]["temperature"],
        hazard=saved["bayesian"]["hazard"],
        timing=saved["settings"]["timing"],
    )
    result = evaluate_predictions(
        rows,
        sessions,
        predictions,
        seed=manifest["protocol"]["config"]["seed"],
        repetitions=repetitions,
    )
    result["evaluation_protocol"] = protocol
    result["selected_history"] = saved["selected_history"]
    result["dataset_manifest_sha256"] = sha256(root / "manifest.json")
    result["interpretation"] = (
        "Controlled Leduc next-action prediction; no human tilt inference or Hero "
        "payoff improvement measured. All predeclared windows reported; "
        "selected_history chosen on validation."
    )
    write_jsonl(
        selection / "test_predictions.jsonl",
        [
            dict(
                session_id=r["session_id"],
                hand_index=r["hand_index"],
                decision_index=r["decision_index"],
                target_index=r["target_index"],
                probabilities={
                    k: p[i, r["inputs"]["legal_action_ids"]].tolist()
                    for k, p in predictions.items()
                },
                bayesian_style_mean=trace[i].tolist(),
            )
            for i, r in enumerate(rows)
        ],
    )
    lines = [
        "# Controlled Leduc forecast results",
        "",
        result["interpretation"],
        "",
        "| Model | Decisions | Sessions | Opponents | Accuracy | Log loss | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in result["overall"].items():
        lines.append(
            f"| {name} | {m['n']} | {m['sessions']} | {m['opponents']} | "
            f"{m['accuracy']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Validation-selected summary: {saved['selected_history']}. "
            "See results.json for all slices, examples and paired opponent-cluster intervals.",
            "Intervals condition on fitted models and the simulated population; "
            "small slices are descriptive.",
        ]
    )
    (selection / "results.md").write_text("\n".join(lines) + "\n")
    result["artifact_sha256"] = {
        name: sha256(selection / name) for name in ("test_predictions.jsonl", "results.md")
    }
    write_json(result_file, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "select", "evaluate", "audit"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/opponents"))
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument(
        "--policy-family",
        choices=("weighted", "categorical", "table", "residual", "cfr"),
        default="weighted",
    )
    parser.add_argument("--hands", type=int, default=60)
    parser.add_argument("--opponents-per-profile", type=int, default=10)
    parser.add_argument("--sessions-per-opponent", type=int, default=2)
    parser.add_argument("--observer-policy", choices=("uniform", "cfr"), default="uniform")
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--cfr-iterations", type=int, default=500)
    parser.add_argument(
        "--timing", choices=("completed_hands", "all_prior"), default="completed_hands"
    )
    parser.add_argument(
        "--selection-method", choices=("validation", "grouped_cv"), default="validation"
    )
    parser.add_argument(
        "--final", action="store_true", help="explicitly score held-out test sessions"
    )
    args = parser.parse_args()
    if args.command == "generate":
        cfg = Config(
            seed=args.seed,
            policy_family=args.policy_family,
            opponents_per_profile=args.opponents_per_profile,
            sessions_per_opponent=args.sessions_per_opponent,
            hands_per_session=args.hands,
            observer_policy=args.observer_policy,
            observer_card=not args.public_only,
            cfr_iterations=args.cfr_iterations,
        )
        print(prepare(cfg, args.output))
        return
    if args.data is None:
        parser.error("--data is required")
    if args.command == "select":
        print(select(args.data, timing=args.timing, selection_method=args.selection_method))
    elif args.command == "audit":
        manifest = verify_manifest(args.data)
        cfg = Config(**manifest["protocol"]["config"])
        dataset = dict(
            rows=sum(
                (read_jsonl(args.data / f"{s}.jsonl") for s in ("train", "validation", "test")), []
            ),
            sessions=json.loads((args.data / "sessions.json").read_text()),
            traces=read_jsonl(args.data / "audit/hands.jsonl"),
        )
        print(replay_audit(dataset, cfg))
    else:
        if not args.final or args.selection is None:
            parser.error("evaluate requires --final and --selection")
        result = final_evaluation(args.data, args.selection)
        print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
