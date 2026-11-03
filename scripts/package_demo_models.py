"""Package the eight trained fold-1 checkpoints as independently fetched demo assets."""

import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path

from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.demo_assets import sha256, verify_assets
from poker_decision.demo_models import ENCODERS, HEADS, ModelSpec, checkpoint_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing data/experiment root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = {}
    shared_ids: set[str] | None = None
    for encoder, (_, backbone) in ENCODERS.items():
        for head in HEADS:
            matches = []
            for path in (args.root / "artifacts/week2_matrix").glob(f"*/fold-1/checkpoint-{head}"):
                metadata = json.loads((path / "metadata.json").read_text())
                if metadata["backbone"] == backbone and metadata["head_type"] == head:
                    matches.append((path, metadata))
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one fold-1 {encoder}/{head} checkpoint")
            path, metadata = matches[0]
            test_ids = set(metadata["test_source_ids"])
            if test_ids.intersection(metadata["train_source_ids"]):
                raise ValueError("Training/held-out membership overlaps")
            shared_ids = test_ids if shared_ids is None else shared_ids.intersection(test_ids)
            checkpoints[f"{encoder}-{head}"] = (encoder, head, path, metadata)
    rows = []
    for split in ("train", "validation"):
        for line in (
            (args.root / f"data/processed/pokerbench_meeting1/{split}.jsonl")
            .read_text()
            .splitlines()
        ):
            row = json.loads(line)
            if row["source_id"] in (shared_ids or set()) and len(row["candidates"]) > 1:
                rows.append(
                    DecisionExample(
                        row["source_id"],
                        row["street"],
                        DecisionInputs(row["context"], tuple(row["candidates"])),
                        row["target_index"],
                    ).to_record()
                )
    if len(rows) < 256:
        raise ValueError("Need at least 256 multi-option hands held out by every checkpoint")
    rows = random.Random(42).sample(rows, 256)
    hand_bytes = "".join(json.dumps(row) + "\n" for row in rows)
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing release staging directory")
    args.output.mkdir(parents=True)
    models = {}
    for key, (encoder, head, checkpoint, metadata) in checkpoints.items():
        assets = args.output / key
        assets.mkdir()
        shutil.copytree(checkpoint, assets / "model")
        (assets / "hands.jsonl").write_text(hand_bytes)
        manifest = {
            "schema": 1,
            "backbone": metadata["backbone"],
            "head_type": head,
            "fold": 1,
            "protocol_sha256": metadata["protocol_sha256"],
            "n_hands": len(rows),
            "selection_seed": 42,
            "source": (
                "Common multi-option development hands held out by all eight fold-1 checkpoints"
            ),
            "files": {
                p.relative_to(assets).as_posix(): sha256(p)
                for p in sorted(assets.rglob("*"))
                if p.is_file()
            },
        }
        (assets / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_assets(assets)
        archive = args.output / f"{key}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
            for path in sorted(assets.rglob("*")):
                if path.is_file():
                    bundle.add(path, arcname=path.relative_to(assets).as_posix())
        record = {
            "encoder": encoder,
            "head": head,
            "backbone": metadata["backbone"],
            "url": f"https://github.com/nmokey/Poker-Decision-Model/releases/download/{args.release}/{archive.name}",
            "sha256": sha256(archive),
            "n_hands": len(rows),
            "fold": 1,
            "protocol_sha256": metadata["protocol_sha256"],
        }
        checkpoint_examples(assets, ModelSpec(key=key, **record))
        models[key] = record
        print(f"Packaged {key}: {archive.stat().st_size / 1e6:.1f} MB", flush=True)
    args.config.write_text(
        json.dumps({"schema": 2, "default_model": "modernbert-linear", "models": models}, indent=2)
        + "\n"
    )
    print(
        f"Saved {len(models)} models with the same {len(rows)} held-out hands: {args.config}",
        flush=True,
    )


if __name__ == "__main__":
    main()
