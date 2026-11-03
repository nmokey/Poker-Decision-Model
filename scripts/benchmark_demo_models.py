"""Measure warm single-hand CPU latency for all serving checkpoints, without API calls."""

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from apps.trained_backend import ModelEngine, Spot
from poker_decision.demo_assets import sha256, verify_assets
from poker_decision.demo_models import SingleModelStore, checkpoint_examples, load_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("deploy/streamlit/assets.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.hands, args.repeats, args.warmup, args.threads) < 1:
        raise ValueError("Benchmark settings must be positive")
    if args.output.exists():
        raise ValueError("Refusing to overwrite a saved latency report")
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    _, specs = load_catalog(args.catalog)
    pools = {}
    for key, spec in specs.items():
        directory = args.assets_root / key
        verify_assets(directory)
        pools[key] = checkpoint_examples(directory, spec)
    pool = pools[next(iter(specs))]
    if any(examples != pool for examples in pools.values()):
        raise ValueError("Latency comparison requires the exact same ordered hand pool")
    if args.hands > len(pool):
        raise ValueError("Requested sample exceeds the common held-out pool")
    examples = random.Random(args.seed).sample(list(pool), args.hands)
    spots = [Spot(e.inputs.context, e.inputs.candidates) for e in examples]
    order = list(specs)
    random.Random(args.seed).shuffle(order)

    def load(key):
        engine = ModelEngine.from_checkpoint(
            args.assets_root / key / "model",
            device="cpu",
            expected_head=specs[key].head,
        )
        if not engine.ready:
            raise RuntimeError(f"Cannot load {key}")
        return engine

    store = SingleModelStore(load)
    results = {}
    for key in order:
        store.warm(key)
        for i in range(args.warmup):
            spot = spots[i % len(spots)]
            result = store.apply(key, lambda engine, spot=spot: engine(spot))
            if not result.ok:
                raise RuntimeError(result.note)
        samples = []
        repeat_means = []
        for repeat in range(args.repeats):
            indices = list(range(len(spots)))
            random.Random(args.seed + repeat).shuffle(indices)
            values = []
            for i in indices:
                result = store.apply(key, lambda engine, spot=spots[i]: engine(spot))
                if not result.ok:
                    raise RuntimeError(result.note)
                values.append(result.latency_ms)
            samples.extend(values)
            repeat_means.append(statistics.mean(values))
        spec = specs[key]
        results[key] = {
            "backbone": spec.backbone,
            "head": spec.head,
            "fold": spec.fold,
            "protocol_sha256": spec.protocol_sha256,
            "asset_sha256": spec.sha256,
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "p95_ms": statistics.quantiles(samples, n=20, method="inclusive")[18]
            if len(samples) > 1
            else samples[0],
            "n_measurements": len(samples),
            "repeat_means_ms": repeat_means,
            "latencies_ms": samples,
        }
        print(
            f"{key}: mean {results[key]['mean_ms']:.2f} ms ({len(samples)} measurements)",
            flush=True,
        )
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    report = {
        "schema": 1,
        "status": "complete",
        "measured_at": datetime.now(UTC).isoformat(),
        "hardware": {
            "cpu": cpu,
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "batch_size": 1,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "hands": args.hands,
            "pool_size": len(pool),
            "repeats": args.repeats,
            "warmup_per_model": args.warmup,
            "seed": args.seed,
            "model_order": order,
        },
        "sampling": {
            "source_ids": [e.source_id for e in examples],
            "pool_sha256": hashlib.sha256(
                json.dumps([e.to_record() for e in pool], sort_keys=True).encode()
            ).hexdigest(),
        },
        "timing_scope": (
            "Input validation, tokenization, batch preparation, CPU forward pass and "
            "probability validation; one example per call."
        ),
        "excluded": ["model loading", "warm-up", "store/worker queueing", "network", "rendering"],
        "catalog_sha256": sha256(args.catalog),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "serving_code_sha256": {
            str(p): sha256(p)
            for p in [
                Path("apps/trained_backend.py"),
                Path("src/poker_decision/demo_models.py"),
                Path("src/poker_decision/training.py"),
                Path("src/poker_decision/models/encoder.py"),
            ]
        },
        "models": {key: results[key] for key in specs},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
