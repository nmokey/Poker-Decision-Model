"""Join saved development scores to timings for the exact serving checkpoints."""

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from poker_decision.demo_models import ModelSpec


@dataclass(frozen=True)
class ResultRow:
    model_id: str
    accuracy: float
    log_loss: float
    mean_ms: float
    n: int


def load_results(
    quality_path: Path, latency_path: Path, models: dict[str, ModelSpec]
) -> list[ResultRow]:
    report = json.loads(latency_path.read_text())
    if report.get("schema") != 1 or report.get("status") != "complete":
        raise ValueError("Incomplete latency report")
    expected_count = report["settings"]["hands"] * report["settings"]["repeats"]
    with quality_path.open(newline="") as file:
        quality = list(csv.DictReader(file))
    rows = []
    for key, spec in models.items():
        matches = [
            row
            for row in quality
            if row["backbone"] == spec.encoder and row["model"] == f"transformer_{spec.head}"
        ]
        if len(matches) != 1 or key not in report["models"]:
            raise ValueError(f"Missing or duplicate results for {key}")
        score, timing = matches[0], report["models"][key]
        if (
            score["protocol_sha256"] != spec.protocol_sha256
            or timing["protocol_sha256"] != spec.protocol_sha256
            or timing["asset_sha256"] != spec.sha256
            or timing["backbone"] != spec.backbone
            or timing["head"] != spec.head
            or timing["fold"] != spec.fold
        ):
            raise ValueError(f"Results do not match the serving checkpoint: {key}")
        samples = timing["latencies_ms"]
        mean_ms = float(timing["mean_ms"])
        accuracy, loss, n = (float(score[field]) for field in ("exact_accuracy", "log_loss", "n"))
        if (
            expected_count < 1
            or timing["n_measurements"] != expected_count
            or len(samples) != expected_count
            or any(not math.isfinite(value) or value <= 0 for value in samples)
            or not math.isfinite(mean_ms)
            or not math.isclose(mean_ms, statistics.mean(samples), rel_tol=1e-9)
            or not 0 <= accuracy <= 1
            or not math.isfinite(loss)
            or loss < 0
            or not math.isfinite(n)
            or n < 1
            or not n.is_integer()
        ):
            raise ValueError(f"Invalid measurements for {key}")
        rows.append(ResultRow(key, accuracy, loss, mean_ms, int(n)))
    return rows
