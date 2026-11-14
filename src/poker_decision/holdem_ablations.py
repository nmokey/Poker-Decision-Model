"""Matched GPU history ablations on fixed Meeting 5 decision memberships.

Architecture selection and subsequent feature/seed comparisons are development
analysis on already inspected validation sessions, never a new final test.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.holdem_history import (
    PROFILE_SECTION,
    dataset_fingerprint,
    load_histories,
    without_history,
)
from poker_decision.holdem_integration import (
    QUESTION,
    ROOT,
    checkpoint_hashes,
    digest,
    metrics,
    validate_sample,
    write_json,
)
from poker_decision.holdem_scaling import paired_change


@dataclass(frozen=True)
class Fit:
    backbone: str
    head: str
    window: int = 64
    representation: str = "compact"
    epochs: int = 1
    seed: int = 42

    @property
    def key(self) -> str:
        return (
            f"{self.backbone}-{self.head}-w{self.window:03d}-"
            f"{self.representation}-e{self.epochs}-s{self.seed}"
        )

    def control(self) -> Fit:
        return replace(self, window=0, representation="none")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def source_hashes() -> dict[str, str]:
    names = (
        "holdem_ablations.py",
        "holdem_history.py",
        "holdem_integration.py",
        "holdem_scaling.py",
        "training.py",
        "models/encoder.py",
        "opponents/data.py",
    )
    return {name: digest(Path(__file__).parent / name) for name in names}


def profile_parts(context: str) -> tuple[str, list[tuple[str, str]]]:
    if context.count(PROFILE_SECTION) != 1:
        raise ValueError("Need exactly one history section")
    base, block = context.split(PROFILE_SECTION)
    profiles = []
    for part in block.split(" | "):
        match = re.fullmatch(r"(seat \d+): (.+)", part)
        if match is None:
            raise ValueError("Malformed opponent profile")
        profiles.append((match[1], match[2]))
    return base, profiles


def transform_context(context: str, source_id: str, representation: str) -> str:
    """Only transform already-causal profile text; no target or donor rows."""
    base, profiles = profile_parts(context)
    if representation == "compact":
        return context
    if representation == "explicit":
        text = context[len(base + PROFILE_SECTION) :]
        text = re.sub(
            r"F/C/R=([0-9.]+)/([0-9.]+)/([0-9.]+)",
            r"fold=\1 call=\2 raise=\3",
            text,
        )
        text = re.sub(r"\bH=", "hands=", text)
        text = re.sub(r"\bn=", "count=", text)
        return base + PROFILE_SECTION + text
    if representation == "swapped":
        # A nonidentity rotation among this decision's active seats keeps every
        # value causal and preserves the profile set, without cross-split donors.
        values = [p[1] for p in profiles]
        if len(values) > 1:
            offset = 1 + int(hashlib.sha256(source_id.encode()).hexdigest(), 16) % (len(values) - 1)
            values = values[offset:] + values[:offset]
        return (
            base
            + PROFILE_SECTION
            + " | ".join(
                f"{seat}: {value}" for (seat, _), value in zip(profiles, values, strict=True)
            )
        )
    raise ValueError("Unknown history representation")


def profile_coverage(context: str) -> dict[str, int]:
    _, profiles = profile_parts(context)
    counts = [int(n) for _, value in profiles for n in re.findall(r"\bn=(\d+)", value)]
    return dict(profiles=len(profiles), comparable_opportunities=sum(counts))


def examples(data: dict[str, Any], split: str, fit: Fit) -> list[DecisionExample]:
    result = []
    for row in data[split]:
        h = row["hero"]
        inputs = DecisionInputs(h["context"], tuple(h["candidates"]))
        if fit.window == 0:
            inputs = without_history(inputs)
        else:
            inputs = DecisionInputs(
                transform_context(inputs.context, h["source_id"], fit.representation),
                inputs.candidates,
            )
        result.append(DecisionExample(h["source_id"], h["street"], inputs, h["target_index"]))
    return result


def architecture_fits(settings: dict[str, Any]) -> list[Fit]:
    return [
        Fit(
            b,
            h,
            window=w,
            representation="compact" if w else "none",
            seed=settings["training_seeds"][0],
        )
        for b in settings["backbones"]
        for h in settings["heads"]
        for w in (0, 64)
    ]


def feature_fits(selected: Fit, settings: dict[str, Any]) -> list[Fit]:
    return (
        [replace(selected, window=w) for w in settings["windows"] if w != 64]
        + [replace(selected, representation=r) for r in ("explicit", "swapped")]
        + [
            replace(selected, epochs=e, window=w, representation="compact" if w else "none")
            for e in settings["epochs"]
            if e != 1
            for w in (0, 64)
        ]
    )


def prepare(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("Use a new output directory; prior results are preserved")
    settings = read(args.config)
    reference = read(args.reference_out / "shared_sample.json")
    manifest = read(args.reference_out / "preparation.json")
    if digest(args.reference_out / "shared_sample.json") != manifest["sample_sha256"]:
        raise ValueError("Reference sample checksum mismatch")
    validate_sample(reference)
    if (len(reference["train"]), len(reference["validation"])) != (4096, 1024):
        raise ValueError("Require the verified 4096/1024 reference sample")
    protocol = dict(
        started_utc=datetime.now(UTC).isoformat(),
        host=socket.gethostname(),
        settings=settings,
        source_sha256=source_hashes(),
        reference_sample_sha256=manifest["sample_sha256"],
        reference_out=str(args.reference_out.resolve()),
        source_archive_sha256=digest(args.archive),
        sampling_seed=reference["seed"],
        train_source_ids=[r["source_id"] for r in reference["train"]],
        validation_source_ids=[r["source_id"] for r in reference["validation"]],
        train_sessions=reference["train_sessions"],
        validation_sessions=reference["validation_sessions"],
        selection=(
            "Architecture: smallest history-minus-control validation log loss. "
            "Then vary windows/format/epochs on that architecture. Repeat the "
            "best real-history pair on seeds 43/44; swapped is a control only."
        ),
        scope="Adaptive development comparisons on inspected validation; no final-test claim",
    )
    for window in sorted(set(settings["windows"]) | {64}):
        print(f"Replaying window {window}", flush=True)
        rows, heroes, audit = load_histories(args.archive, hero_per_session=None, window=window)
        forecasts = {r["source_id"]: r for r in rows}
        hero_map = {h.source_id: h.to_record() for h in heroes}
        data = {k: reference[k] for k in ("seed", "train_sessions", "validation_sessions")}
        data["source_audit"] = audit
        for split in ("train", "validation"):
            selected = []
            for original in reference[split]:
                key = original["source_id"]
                item = dict(
                    source_id=key,
                    session_id=original["session_id"],
                    table_id=original["table_id"],
                    forecast=forecasts[key],
                    hero=hero_map[key],
                )
                old, new = original["hero"], item["hero"]
                if (
                    old["target_index"] != new["target_index"]
                    or tuple(old["candidates"]) != tuple(new["candidates"])
                    or without_history(DecisionInputs(old["context"], tuple(old["candidates"])))
                    != without_history(DecisionInputs(new["context"], tuple(new["candidates"])))
                ):
                    raise ValueError("Replay changed target, base state, or candidate menu")
                selected.append(item)
            data[split] = selected
            if window == 64 and dataset_fingerprint(selected) != dataset_fingerprint(
                reference[split]
            ):
                raise ValueError("Canonical replay no longer matches the original rows exactly")
        validate_sample(data)
        write_json(args.out / "samples" / f"window_{window:03d}.json", data)
        del rows, heroes, forecasts, hero_map, data
        gc.collect()
    protocol["sample_sha256"] = {
        p.name: digest(p) for p in sorted((args.out / "samples").glob("*.json"))
    }
    protocol["checkpoint_root"] = str(args.checkpoint_root.resolve())
    protocol["checkpoints"] = {}
    from poker_decision.demo_assets import download_assets, verify_assets

    catalog = read(ROOT / "deploy/streamlit/assets.json")["models"]
    protocol["checkpoint_releases"] = {}
    for backbone in settings["backbones"]:
        for head in settings["heads"]:
            directory = args.checkpoint_root / f"{backbone}-{head}"
            spec = catalog[f"{backbone}-{head}"]
            if not directory.exists():
                download_assets(spec["url"], directory, expected_sha256=spec["sha256"])
            verify_assets(directory)
            protocol["checkpoint_releases"][f"{backbone}-{head}"] = spec
            protocol["checkpoints"][f"{backbone}-{head}"] = checkpoint_hashes(directory / "model")
    write_json(args.out / "protocol.json", protocol)
    print(f"Prepared fixed membership and three causal history windows: {args.out}", flush=True)


def load_data(out: Path, fit: Fit) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read(out / "protocol.json")
    if protocol["source_sha256"] != source_hashes():
        raise ValueError("Experiment source changed; prepare a fresh experiment")
    path = out / "samples" / f"window_{fit.window or 64:03d}.json"
    if digest(path) != protocol["sample_sha256"][path.name]:
        raise ValueError("Prepared sample checksum mismatch")
    data = read(path)
    validate_sample(data)
    for split in ("train", "validation"):
        if [r["source_id"] for r in data[split]] != protocol[f"{split}_source_ids"]:
            raise ValueError("Changed sample membership or order")
    return data, protocol


def slices(data: dict[str, Any], p: list[list[float]]) -> dict[str, Any]:
    rows = data["validation"]
    coverage = [profile_coverage(r["hero"]["context"])["comparable_opportunities"] for r in rows]
    groups = {
        s: [i for i, r in enumerate(rows) if r["hero"]["street"] == s]
        for s in ("preflop", "flop", "turn", "river")
    }
    groups |= {
        "history_0": [i for i, n in enumerate(coverage) if n == 0],
        "history_1_to_9": [i for i, n in enumerate(coverage) if 0 < n < 10],
        "history_10_plus": [i for i, n in enumerate(coverage) if n >= 10],
    }
    return {
        name: metrics([rows[i]["hero"]["target_index"] for i in indices], [p[i] for i in indices])
        for name, indices in groups.items()
        if indices
    }


def run_fit(out: Path, fit: Fit) -> None:
    import torch

    from poker_decision.training import (
        decision_optimizer,
        encode_examples,
        load_checkpoint,
        predict_encoded,
        save_checkpoint,
        seed_everything,
        train_epoch,
    )

    data, protocol = load_data(out, fit)
    settings = protocol["settings"]
    destination = out / "fits" / fit.key
    if (destination / "result.json").exists():
        existing = read(destination / "result.json")
        if existing["fit"] != asdict(fit) or existing["source_sha256"] != source_hashes():
            raise ValueError("Stale saved fit")
        print(f"Already complete: {fit.key}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("These ablations require a CUDA GPU; no CPU fallback")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(settings["gpu_memory_fraction"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    seed_everything(fit.seed)
    checkpoint = Path(protocol["checkpoint_root"]) / f"{fit.backbone}-{fit.head}" / "model"
    expected_hashes = protocol["checkpoints"][f"{fit.backbone}-{fit.head}"]
    if checkpoint_hashes(checkpoint) != expected_hashes:
        raise ValueError("Published starting checkpoint changed")
    tokenizer, model, metadata = load_checkpoint(checkpoint)
    if model.head_type != fit.head:
        raise ValueError("Checkpoint head does not match configuration")
    model.to("cuda")
    train, validation = (examples(data, split, fit) for split in ("train", "validation"))
    budget = min(settings["max_length"], int(model.encoder.config.max_position_embeddings))
    train_encoded = encode_examples(train, tokenizer, max_length=budget, question=QUESTION)
    val_encoded = encode_examples(validation, tokenizer, max_length=budget, question=QUESTION)
    optimizer = decision_optimizer(
        model,
        encoder_lr=settings["encoder_lr"],
        head_lr=settings["head_lr"],
        weight_decay=settings["weight_decay"],
    )
    started = time.perf_counter()
    losses = []
    for epoch in range(fit.epochs):
        loss = train_epoch(
            model,
            train_encoded,
            [e.target_index for e in train],
            pad_token_id=tokenizer.pad_token_id,
            optimizer=optimizer,
            batch_size=settings["batch_size"],
            seed=fit.seed + epoch,
            micro_batch_size=settings["micro_batch_size"],
        )
        losses.append(loss)
        print(
            f"{fit.key} epoch {epoch + 1}/{fit.epochs}: loss {loss:.6f}, "
            f"elapsed {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    p = predict_encoded(
        model,
        val_encoded,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=settings["micro_batch_size"],
    )
    score = metrics([e.target_index for e in validation], p)
    peak = torch.cuda.max_memory_allocated()
    model.to("cpu")
    del optimizer
    torch.cuda.empty_cache()
    save_checkpoint(
        destination / "checkpoint",
        model,
        tokenizer,
        dict(
            backbone=metadata.get("backbone"),
            question=QUESTION,
            max_length=budget,
            fit=asdict(fit),
            settings=settings,
            source_sha256=source_hashes(),
            label_semantics="Observed human action category",
        ),
    )
    del model
    gc.collect()
    restored_tokenizer, restored, _ = load_checkpoint(destination / "checkpoint")
    restored_encoded = encode_examples(
        validation, restored_tokenizer, max_length=budget, question=QUESTION
    )
    if restored_encoded != val_encoded:
        raise ValueError("Reloaded tokenizer changed inputs")
    restored.to("cuda")
    after = predict_encoded(
        restored,
        restored_encoded,
        pad_token_id=restored_tokenizer.pad_token_id,
        batch_size=settings["micro_batch_size"],
    )
    error = max(
        abs(x - y) for a, b in zip(p, after, strict=True) for x, y in zip(a, b, strict=True)
    )
    if error > 1e-7:
        raise ValueError("Saved checkpoint predictions changed after reload")
    if checkpoint_hashes(checkpoint) != expected_hashes:
        raise ValueError("Starting checkpoint changed during training")
    result = dict(
        fit=asdict(fit),
        key=fit.key,
        scores=score,
        slices=slices(data, p),
        source_ids=[e.source_id for e in validation],
        probabilities=p,
        targets=[e.target_index for e in validation],
        training_losses=losses,
        source_sha256=source_hashes(),
        starting_checkpoint_sha256=expected_hashes,
        sample_sha256=protocol["sample_sha256"][f"window_{fit.window or 64:03d}.json"],
        max_tokens=max(len(e.input_ids) for e in train_encoded + val_encoded),
        reload_max_abs_diff=error,
        gpu=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        peak_allocated_bytes=peak,
        dtype="float32",
        torch_version=torch.__version__,
        elapsed_seconds=time.perf_counter() - started,
        host=socket.gethostname(),
    )
    write_json(destination / "result.json", result)
    print(f"DONE {fit.key}: {json.dumps(score)}", flush=True)


def result(out: Path, fit: Fit) -> dict[str, Any]:
    value: dict[str, Any] = read(out / "fits" / fit.key / "result.json")
    return value


def difference(out: Path, fit: Fit) -> float:
    return float(
        result(out, fit)["scores"]["log_loss"] - result(out, fit.control())["scores"]["log_loss"]
    )


def launch(out: Path, fit: Fit, gpu: int) -> None:
    path = out / "logs" / f"{fit.key}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {fit.key} on GPU {gpu}", flush=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with path.open("a") as log:
        completed = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "poker_decision.holdem_ablations",
                "fit",
                "--out",
                str(out),
                "--fit-json",
                json.dumps(asdict(fit)),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Fit failed: {fit.key}; inspect {path}")
    print(f"DONE {fit.key}: {json.dumps(result(out, fit)['scores'])}", flush=True)


def suite(args: argparse.Namespace) -> None:
    protocol = read(args.out / "protocol.json")
    settings = protocol["settings"]
    architecture = architecture_fits(settings)
    for fit in architecture:
        launch(args.out, fit, args.gpu)
    selected = min((f for f in architecture if f.window), key=lambda f: difference(args.out, f))
    write_json(
        args.out / "architecture_selection.json",
        dict(
            selected=asdict(selected),
            criterion="minimum paired validation log-loss change",
            candidates=[
                dict(fit=asdict(f), delta_log_loss=difference(args.out, f))
                for f in architecture
                if f.window
            ],
        ),
    )
    print(f"Selected architecture for feature ablations: {selected.key}", flush=True)
    features = feature_fits(selected, settings)
    for fit in features:
        launch(args.out, fit, args.gpu)
    candidates = [selected] + [f for f in features if f.window and f.representation != "swapped"]
    best = min(candidates, key=lambda f: difference(args.out, f))
    write_json(
        args.out / "seed_selection.json",
        dict(
            selected=asdict(best),
            delta_log_loss=difference(args.out, best),
            scope="Selected on inspected validation; repeats are not independent final testing",
        ),
    )
    repeats = []
    for seed in settings["training_seeds"][1:]:
        pair = [replace(best.control(), seed=seed), replace(best, seed=seed)]
        for fit in pair:
            launch(args.out, fit, args.gpu)
        repeats.extend(pair)
    write_json(
        args.out / "completed_plan.json",
        dict(fits=[asdict(f) for f in architecture + features + repeats]),
    )
    summarize(args.out)


def summarize(out: Path) -> dict[str, Any]:
    protocol = read(out / "protocol.json")
    fits = [Fit(**read(p)["fit"]) for p in sorted((out / "fits").glob("*/result.json"))]
    rows, pairs = [], []
    for fit in fits:
        r = result(out, fit)
        if r["source_ids"] != protocol["validation_source_ids"]:
            raise ValueError("Misaligned saved predictions")
        checked = metrics(r["targets"], r["probabilities"])
        if checked != r["scores"] or r["source_sha256"] != source_hashes():
            raise ValueError("Stale results or incorrect aggregate metrics")
        rows.append(
            dict(
                key=fit.key,
                **asdict(fit),
                **checked,
                slices=r["slices"],
                reload_max_abs_diff=r["reload_max_abs_diff"],
                max_tokens=r["max_tokens"],
                gpu=r["gpu"],
                elapsed_seconds=r["elapsed_seconds"],
                peak_allocated_bytes=r["peak_allocated_bytes"],
            )
        )
        control_path = out / "fits" / fit.control().key / "result.json"
        if fit.window and control_path.exists():
            control = result(out, fit.control())
            data, _ = load_data(out, fit)
            comparison = dict(
                predictions={
                    "without_history": {"probabilities": control["probabilities"]},
                    "with_history": {"probabilities": r["probabilities"]},
                }
            )
            pairs.append(
                dict(key=fit.key, control=fit.control().key, **paired_change(data, comparison))
            )
    summary = dict(
        protocol=protocol,
        results=rows,
        paired_changes=pairs,
        architecture_selection=read(out / "architecture_selection.json")
        if (out / "architecture_selection.json").exists()
        else None,
        seed_selection=read(out / "seed_selection.json")
        if (out / "seed_selection.json").exists()
        else None,
        caution="Adaptive development sweeps, one inspected session split; no final-test estimate.",
    )
    write_json(out / "summary.json", summary)
    lines = [
        "# Hold’em history ablations",
        "",
        summary["caution"],
        "",
        "| Configuration | Accuracy | Log loss |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {r['key']} | {r['accuracy']:.2%} | {r['log_loss']:.6f} |" for r in rows)
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Verified {len(rows)} fits: {out / 'results.md'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "suite", "fit", "report"))
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/holdem_ablations")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/holdem_ablations.json")
    parser.add_argument(
        "--reference-out", type=Path, default=ROOT / "artifacts/holdem_scaling/train_04096"
    )
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "data/raw/alternative_sources/pluribus_phh_v1.zip"
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "artifacts/deployment/pocketsolver-models-v2"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fit-json")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args)
    elif args.step == "suite":
        suite(args)
    elif args.step == "report":
        summarize(args.out)
    else:
        run_fit(args.out, Fit(**json.loads(args.fit_json)))


if __name__ == "__main__":
    main()
