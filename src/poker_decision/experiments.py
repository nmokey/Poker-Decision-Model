"""Run the Week 2 backbone/head matrix on a GPU, with training-only preflight checks."""

import argparse
import csv
import gc
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from poker_decision.cross_validation import CVConfig, run_cross_validation
from poker_decision.models.encoder import BACKBONES, load_pretrained
from poker_decision.pipeline import prepare, write_json
from poker_decision.scenarios import digest, make_folds, scenario_overlap
from poker_decision.training import (
    encode_examples,
    load_prepared,
    score_model,
    seed_everything,
    train_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--backbone", choices=list(BACKBONES), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/week2_matrix"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This runner requires a CUDA GPU")
    torch.set_num_threads(4)
    config = CVConfig(backbone=args.backbone, heads=("linear", "mlp"), epochs=args.epochs)
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (root / "data/processed/pokerbench_meeting1/manifest.json").exists():
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
    # Select only first-fold TRAINING examples. Preflight never selects settings.
    first_fit = [pool[i] for i in folds[0][0]]
    tiny, seen = [], set()
    for row in random.Random(config.seed).sample(first_fit, len(first_fit)):
        if len(row.inputs.candidates) > 1 and row.inputs.context not in seen:
            tiny.append(row)
            seen.add(row.inputs.context)
        if len(tiny) == 32:
            break
    if len(tiny) != 32:
        raise ValueError("Need 32 distinct multi-option training contexts")
    for head in config.heads:
        seed_everything(config.seed)
        tokenizer, model = load_pretrained(
            *BACKBONES[config.backbone], head_type=head, head_width=config.head_width
        )
        model = model.to("cuda")
        encoded = encode_examples(tiny, tokenizer, max_length=config.max_length)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        history = []
        for epoch in range(100):
            loss = train_epoch(
                model,
                encoded,
                [e.target_index for e in tiny],
                pad_token_id=tokenizer.pad_token_id,
                optimizer=optimizer,
                batch_size=8,
                seed=config.seed + epoch,
            )
            metrics = score_model(model, tiny, encoded, pad_token_id=tokenizer.pad_token_id)[
                "overall"
            ]
            history.append({"epoch": epoch + 1, "training_loss": loss, **metrics})
            if epoch == 0 or (epoch + 1) % 10 == 0 or metrics["exact_accuracy"] == 1:
                print(
                    f"Tiny {head} epoch {epoch + 1}: accuracy={metrics['exact_accuracy']:.3f}",
                    flush=True,
                )
            if metrics["exact_accuracy"] == 1:
                break
        write_json(
            run / f"tiny-{head}.json",
            {"source_ids": [e.source_id for e in tiny], "history": history},
        )
        if history[-1]["exact_accuracy"] < 0.95:
            raise RuntimeError(f"{args.backbone}/{head} failed training-only learning check")
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
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
