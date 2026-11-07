"""Run the decision backbone/head matrix on a GPU, with training-only preflight checks."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from poker_decision.cross_validation import CVConfig, run_cross_validation
from poker_decision.models.encoder import BACKBONES
from poker_decision.paths import prepared_directory
from poker_decision.pipeline import prepare, write_json
from poker_decision.prepared import load_prepared
from poker_decision.scenarios import digest, make_folds, scenario_overlap
from poker_decision.training import learning_check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--backbone", choices=list(BACKBONES), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/decision_matrix"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This runner requires a CUDA GPU")
    torch.set_num_threads(4)
    config = CVConfig(backbone=args.backbone, heads=("linear", "mlp"), epochs=args.epochs)
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (prepared_directory(root) / "manifest.json").exists():
        prepare(root)
    train, validation, manifest = load_prepared(root)
    pool, folds, protocol = make_folds(train + validation, n_splits=5, seed=config.seed)
    preflight_spec = {
        "config": asdict(config),
        "pool_sha256": protocol["pool_sha256"],
        "tiny_n": 32,
        "tiny_epochs": 100,
        "tiny_lr": 1e-4,
        "tiny_batch_size": 8,
        "required_accuracy": 0.95,
        "hardware": torch.cuda.get_device_name(0),
        "data_manifest": manifest,
        "historical_overlap": scenario_overlap(train, validation),
    }
    run = output / f"{args.backbone}-{digest(preflight_spec)[:12]}"
    run.mkdir(exist_ok=True)
    write_json(run / "preflight_protocol.json", preflight_spec)
    write_json(run / "folds.json", protocol)
    print(
        json.dumps(
            {
                "backbone": args.backbone,
                "settings": asdict(config),
                "hardware": preflight_spec["hardware"],
                "rows": len(pool),
            },
            indent=2,
        ),
        flush=True,
    )
    # Only first-fold training examples reach the shared, disposable learning check.
    first_fit = [pool[i] for i in folds[0][0]]
    for head in config.heads:
        learning_check(first_fit, run / f"tiny-{head}.json", config=config, head=head)
    # The CV runner initializes new encoders/heads/optimizers after the tiny check.
    result = run_cross_validation(pool, output, config=config, device="cuda")
    write_json(run / "result.json", result)
    rows = [
        {
            "backbone": config.backbone,
            "model": model,
            **metrics,
            "output_directory": result["output_directory"],
        }
        for model, metrics in result["models"].items()
    ]
    with (run / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"complete": True, "metrics": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
