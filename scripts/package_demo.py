"""Package existing weights and a reproducible held-out demo sample, without secrets."""

import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path

from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.demo_assets import sha256, verify_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    output = args.root / "artifacts/deployment/pocketsolver"
    assets = output / "assets"
    if assets.exists():
        raise ValueError("An asset bundle already exists; inspect it before rebuilding")
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    test_ids = set(metadata["test_source_ids"])
    if test_ids.intersection(metadata["train_source_ids"]):
        raise ValueError("Checkpoint training and held-out memberships overlap")
    records = []
    for split in ("train", "validation"):
        path = args.root / "data/processed/pokerbench_meeting1" / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["source_id"] in test_ids and len(row["candidates"]) > 1:
                example = DecisionExample(
                    row["source_id"],
                    row["street"],
                    DecisionInputs(row["context"], tuple(row["candidates"])),
                    row["target_index"],
                )
                records.append(example.to_record())
    records = random.Random(42).sample(records, min(256, len(records)))
    if not records:
        raise ValueError("No held-out demo examples")
    assets.mkdir(parents=True)
    shutil.copytree(args.checkpoint, assets / "model")
    (assets / "hands.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    manifest = {
        "schema": 1,
        "backbone": metadata["backbone"],
        "head_type": metadata["head_type"],
        "fold": metadata["fold"],
        "protocol_sha256": metadata["protocol_sha256"],
        "n_hands": len(records),
        "selection_seed": 42,
        "source": "PokerBench training-only development pool; held out by this checkpoint",
        "files": {
            p.relative_to(assets).as_posix(): sha256(p)
            for p in sorted(assets.rglob("*"))
            if p.is_file()
        },
    }
    (assets / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_assets(assets)
    archive = output / "pocketsolver-demo-assets.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(assets.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(assets).as_posix())
    config = {
        "url": "https://github.com/nmokey/Poker-Decision-Model/releases/download/"
        "meeting3-demo-2026-11-02-v1/pocketsolver-demo-assets.tar.gz",
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "n_hands": len(records),
        "backbone": metadata["backbone"],
    }
    (args.root / "deploy/streamlit/assets.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Packaged {len(records)} held-out hands and saved weights: {archive}")
    print(f"Bundle size: {archive.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
