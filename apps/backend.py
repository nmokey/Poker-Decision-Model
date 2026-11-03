"""Portable saved-model and rate-limited Groq adapters."""

import json
import os
import random
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from apps.contract import Prediction, Spot
from apps.trained_backend import ModelEngine
from apps.trained_backend import Spot as ModelSpot
from poker_decision.data import DecisionExample, DecisionInputs
from poker_decision.demo_assets import download_assets, verify_assets
from poker_decision.demo_limits import CallBudget, CallLimitError, CallLimits

ROOT = Path(__file__).resolve().parents[1]


def setting(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    try:
        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    except FileNotFoundError:
        pass
    return str(dotenv_values(ROOT / ".env").get(name) or default).strip()


def groq_ready() -> bool:
    return bool(setting("GROQ_API_KEY"))


def groq_model_id() -> str:
    return setting("GROQ_MODEL_ID", "openai/gpt-oss-20b")


def call_limits() -> CallLimits:
    return CallLimits(
        daily=int(setting("POKER_LLM_DAILY_LIMIT", "100")),
        session_daily=int(setting("POKER_LLM_SESSION_LIMIT", "10")),
        cooldown_seconds=float(setting("POKER_LLM_COOLDOWN_SECONDS", "5")),
    )


@st.cache_resource
def asset_directory() -> Path:
    local = setting("POKER_DEMO_ASSET_DIR")
    if local:
        directory = Path(local).expanduser().resolve()
        verify_assets(directory)
        return directory
    config = json.loads((ROOT / "deploy/streamlit/assets.json").read_text())
    directory = Path(tempfile.gettempdir()) / "pocketsolver" / config["sha256"]
    download_assets(config["url"], directory, expected_sha256=config["sha256"])
    return directory


@st.cache_resource
def held_out_examples() -> tuple[DecisionExample, ...]:
    directory = asset_directory()
    metadata = json.loads((directory / "model/metadata.json").read_text())
    test_ids, train_ids = set(metadata["test_source_ids"]), set(metadata["train_source_ids"])
    if test_ids.intersection(train_ids):
        raise ValueError("Checkpoint membership overlaps")
    examples = []
    for line in (directory / "hands.jsonl").read_text().splitlines():
        row = json.loads(line)
        if not isinstance(row["context"], str) or not isinstance(row["candidates"], list):
            raise ValueError("Invalid stored demo hand")
        if any(not isinstance(c, str) for c in row["candidates"]):
            raise ValueError("Invalid stored candidate strings")
        if row["source_id"] not in test_ids or row["source_id"] in train_ids:
            raise ValueError("Demo hand was not held out by this checkpoint")
        example = DecisionExample(
            row["source_id"],
            row["street"],
            DecisionInputs(row["context"], tuple(row["candidates"])),
            row["target_index"],
        )
        if len(example.inputs.candidates) < 2:
            raise ValueError("Expected a multi-option demo hand")
        examples.append(example)
    if not examples or len({e.source_id for e in examples}) != len(examples):
        raise ValueError("Demo hands must be nonempty and unique")
    return tuple(examples)


@st.cache_resource
def loaded_model() -> tuple[ModelEngine, threading.Lock]:
    import torch

    torch.set_num_threads(2)
    first = held_out_examples()[0].inputs
    engine = ModelEngine.from_checkpoint(
        asset_directory() / "model",
        device="cpu",
        expected_head="linear",
        warmup_spot=ModelSpot(first.context, first.candidates),
    )
    if not engine.ready:
        raise RuntimeError(engine(ModelSpot(first.context, first.candidates)).note)
    return engine, threading.Lock()


def model_engine(spot: Spot) -> Prediction:
    engine, lock = loaded_model()
    with lock:
        result = engine(ModelSpot(spot.context, spot.candidates))
    return Prediction(
        result.probs if result.ok else None,
        result.latency_ms,
        ok=result.ok,
        note=result.note,
    )


@st.cache_resource
def budget() -> CallBudget:
    default = str(Path(tempfile.gettempdir()) / "pocketsolver/usage.sqlite3")
    return CallBudget(Path(setting("POKER_LLM_USAGE_DB", default)), call_limits())


@st.cache_resource
def provider_slots() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(2)


def llm_engine(spot: Spot, *, session: str) -> Prediction:
    from apps.groq_backend import llm_engine as predict

    started = time.perf_counter()
    token = setting("GROQ_API_KEY")
    if not token:
        return Prediction(None, 0, ok=False, note="Live LLM comparison is not configured.")
    slots = provider_slots()
    if not slots.acquire(blocking=False):
        return Prediction(
            None, 0, ok=False, note="Live comparisons are busy. Please try again shortly."
        )
    try:
        os.environ["GROQ_API_KEY"] = token
        os.environ["GROQ_MODEL_ID"] = groq_model_id()
        budget().reserve(session)
        result = predict(spot.context, spot.candidates)
        return Prediction(
            result.probs,
            result.latency_ms,
            ok=result.ok,
            note=result.note,
            raw=result.raw,
        )
    except CallLimitError as error:
        return Prediction(None, (time.perf_counter() - started) * 1000, ok=False, note=str(error))
    finally:
        slots.release()


class HandSource:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.queue: list[DecisionExample] = []

    def next(self) -> DecisionExample:
        if not self.queue:
            self.queue = self.rng.sample(list(held_out_examples()), len(held_out_examples()))
        return self.queue.pop()
