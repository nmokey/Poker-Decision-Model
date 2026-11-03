"""Portable saved-model and rate-limited Groq adapters."""

import os
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from apps.contract import Prediction, Spot
from apps.trained_backend import ModelEngine
from apps.trained_backend import Spot as ModelSpot
from poker_decision.data import DecisionExample
from poker_decision.demo_assets import download_assets, verify_assets
from poker_decision.demo_limits import CallBudget, CallLimitError, CallLimits
from poker_decision.demo_models import SingleModelStore, checkpoint_examples, load_catalog

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
def model_catalog():
    return load_catalog(ROOT / "deploy/streamlit/assets.json")


@st.cache_resource
def asset_directory(model_id: str) -> Path:
    _, models = model_catalog()
    spec = models[model_id]
    local_root = setting("POKER_DEMO_ASSET_ROOT")
    local = setting("POKER_DEMO_ASSET_DIR")
    if local_root or local:
        directory = (
            (Path(local_root) / model_id if local_root else Path(local)).expanduser().resolve()
        )
        verify_assets(directory)
        return directory
    directory = Path(tempfile.gettempdir()) / "pocketsolver" / spec.sha256
    download_assets(spec.url, directory, expected_sha256=spec.sha256)
    return directory


@st.cache_resource
def held_out_examples(model_id: str) -> tuple[DecisionExample, ...]:
    _, models = model_catalog()
    return checkpoint_examples(asset_directory(model_id), models[model_id])


def _load_model(model_id: str) -> ModelEngine:
    import torch

    torch.set_num_threads(2)
    _, models = model_catalog()
    spec = models[model_id]
    first = held_out_examples(model_id)[0].inputs
    engine = ModelEngine.from_checkpoint(
        asset_directory(model_id) / "model",
        device="cpu",
        expected_head=spec.head,
        warmup_spot=ModelSpot(first.context, first.candidates),
    )
    if not engine.ready:
        raise RuntimeError(engine(ModelSpot(first.context, first.candidates)).note)
    return engine


@st.cache_resource
def model_store() -> SingleModelStore[ModelEngine]:
    return SingleModelStore(_load_model)


def loaded_model(model_id: str) -> None:
    model_store().warm(model_id)


def model_engine(spot: Spot, *, model_id: str) -> Prediction:
    result = model_store().apply(
        model_id, lambda engine: engine(ModelSpot(spot.context, spot.candidates))
    )
    return Prediction(
        result.probs if result.ok else None,
        result.latency_ms,
        ok=result.ok,
        note=result.note,
    )


@st.cache_resource
def prediction_pool() -> ThreadPoolExecutor:
    # Share a bounded worker pool instead of creating new workers on every rerun.
    return ThreadPoolExecutor(max_workers=4)


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
    def __init__(self, seed: int | None = None, *, model_id: str) -> None:
        self.model_id = model_id
        self.rng = random.Random(seed)
        self.queue: list[DecisionExample] = []

    def next(self) -> DecisionExample:
        if not self.queue:
            examples = held_out_examples(self.model_id)
            self.queue = self.rng.sample(list(examples), len(examples))
        return self.queue.pop()
