"""Launch the eight-combination Week 2 experiment on three GPUs and collect results."""

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/week2_matrix"


def worker(gpu: int, backbones: tuple[str, ...]) -> list[dict[str, object]]:
    outcomes = []
    for backbone in backbones:
        command = [
            sys.executable,
            "-u",
            "-m",
            "poker_decision.experiments",
            "--root",
            str(ROOT),
            "--backbone",
            backbone,
            "--output",
            str(OUT),
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
        }
        with (OUT / f"{backbone}.log").open("a") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        outcome = {"backbone": backbone, "gpu": gpu, "returncode": completed.returncode}
        (OUT / f"{backbone}-status.json").write_text(json.dumps(outcome, indent=2) + "\n")
        print(json.dumps(outcome), flush=True)
        outcomes.append(outcome)
    return outcomes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(worker, gpu, names)
            for gpu, names in [
                (0, ("bert_mini", "minilm")),
                (1, ("distilbert",)),
                (2, ("modernbert",)),
            ]
        ]
        outcomes = [result for future in futures for result in future.result()]
    rows = []
    for path in sorted(OUT.glob("*/metrics.csv")):
        with path.open() as stream:
            rows.extend(csv.DictReader(stream))
    if rows:
        with (OUT / "all_metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "matrix_status.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    if any(item["returncode"] != 0 for item in outcomes):
        raise SystemExit("Some experiments failed; see per-backbone logs and status files.")


if __name__ == "__main__":
    main()
