"""PocketSolver: the trained decision model and live Groq on the same hand."""

import uuid
from functools import partial

import streamlit as st

from apps.backend import (
    HandSource,
    call_limits,
    groq_model_id,
    groq_ready,
    held_out_examples,
    llm_engine,
    loaded_model,
    model_catalog,
    model_engine,
    prediction_pool,
)
from apps.benchmark_view import draw_benchmark
from apps.contract import Prediction, Spot
from apps.spot_view import draw_spot
from apps.ui_common import fmt_ms, problem_with, run_safely
from poker_decision.data import DecisionInputs
from poker_decision.demo_models import ENCODERS, HEADS
from poker_decision.evaluation import predicted_index


def render_prediction(spot: Spot, prediction: Prediction) -> None:
    problem = problem_with(spot, prediction)
    if problem:
        st.warning(problem)
        return
    assert prediction.probs is not None
    top = predicted_index(DecisionInputs(spot.context, spot.candidates), prediction.probs)
    for i, (action, probability) in enumerate(zip(spot.candidates, prediction.probs, strict=True)):
        label = f"**{action}**" if i == top else action
        st.progress(probability, text=f"{label} · {probability:.1%}")
    st.caption(f"Top pick: {spot.candidates[top]}")
    st.metric("Response time", fmt_ms(prediction.latency_ms))
    if prediction.raw:
        with st.expander("Raw LLM response"):
            st.code(prediction.raw, language="json")


st.set_page_config(page_title="PocketSolver", page_icon="♠", layout="wide")
state = st.session_state
if "demo_session" not in state:
    state.demo_session = uuid.uuid4().hex
    state.hand_id = 0
    state.predictions = {}
if "prediction_jobs" not in state:
    state.prediction_jobs = {}

st.title("POCKETSOLVER")
st.caption("A trained decision model versus a generative LLM · same hand, same options")
default, models = model_catalog()
encoder_options = [key for key in ENCODERS if any(m.encoder == key for m in models.values())]
encoder_column, head_column = st.columns(2)
encoder = encoder_column.selectbox(
    "Pretrained encoder",
    encoder_options,
    index=encoder_options.index(models[default].encoder),
    format_func=lambda key: ENCODERS[key][0],
    key="encoder_selection",
)
head_options = [head for head in HEADS if f"{encoder}-{head}" in models]
head = head_column.selectbox(
    "Decision head",
    head_options,
    format_func=lambda key: HEADS[key],
    key="head_selection",
)
model_id = f"{encoder}-{head}"
spec = models[model_id]
groq_label = f"Groq · {groq_model_id()}"
limits = call_limits()
with st.sidebar:
    st.markdown("**Live LLM allowance**")
    st.caption(
        f"Up to {limits.session_daily} requests per session per day, "
        f"with {limits.daily} shared across this demo. "
        "The trained model remains available when the allowance is used."
    )
    st.caption(f"{spec.label} head · CPU")
    st.caption(f"Groq model: {groq_model_id()}")
    if not groq_ready():
        st.info("Live LLM comparison is not configured yet.")
try:
    with st.spinner(f"Preparing {spec.label}… First use may take a moment."):
        examples = held_out_examples(model_id)
        loaded_model(model_id)
except Exception:
    st.error(
        "The selected saved model could not be loaded. Choose another model or try again later."
    )
    st.stop()
st.caption(
    f"{len(examples):,} multi-option hands held out by this checkpoint · trained fold {spec.fold}"
)
st.caption(
    "Default: ModernBERT with a linear head, the highest observed score in our development matrix."
)
st.caption("Each encoder/head choice loads its own trained checkpoint.")
if state.get("selected_model") != model_id:
    old_example = state.get("example")
    eligible = {example.source_id: example for example in examples}
    same_hand = old_example is not None and eligible.get(old_example.source_id) == old_example
    for name, job in state.prediction_jobs.items():
        if not same_hand or name != groq_label:
            job.cancel()
    state.prediction_jobs = {
        name: job for name, job in state.prediction_jobs.items() if same_hand and name == groq_label
    }
    state.predictions = {
        name: result
        for name, result in state.predictions.items()
        if same_hand and name == groq_label
    }
    if not same_hand:
        state.pop("example", None)
    state.source = HandSource(model_id=model_id)
    state.hand_id += 1
    state.selected_model = model_id
    state.pop("demo_benchmark", None)

view = st.segmented_control("View", ["Single hand", "Benchmark"], default="Single hand")
engines = {
    f"{spec.label} · trained model": partial(model_engine, model_id=model_id),
    groq_label: partial(llm_engine, session=state.demo_session),
}
if view == "Benchmark":
    draw_benchmark(engines, model_id=model_id)
    st.stop()

new, repeat = st.columns([1, 3])
if new.button("New hand", type="primary"):
    state.example = state.source.next()
    state.hand_id += 1
    state.predictions = {}
    for job in state.prediction_jobs.values():
        job.cancel()
    state.prediction_jobs = {}
rerun = repeat.button("Re-run engines", disabled="example" not in state)
if "example" not in state:
    st.info("Click New hand to compare the trained model and live LLM.")
    st.stop()
if rerun:
    state.hand_id += 1
    state.predictions = {}
    for job in state.prediction_jobs.values():
        job.cancel()
    state.prediction_jobs = {}
request_id = state.hand_id
example = state.example
spot = Spot(example.inputs.context, example.inputs.candidates)
draw_spot(spot.context)
st.markdown("**Available actions:** " + " · ".join(f"`{c}`" for c in spot.candidates))
with st.expander("Exact input sent to both engines"):
    st.code(f"{spot.context}\n\nCandidates: {list(spot.candidates)}")

# Save futures before waiting, so selector reruns reuse even an in-flight Groq call.
for name, engine in engines.items():
    if name not in state.predictions and name not in state.prediction_jobs:
        state.prediction_jobs[name] = prediction_pool().submit(run_safely, engine, spot)
for name, column in zip(engines, st.columns(2), strict=True):
    with column.container(border=True):
        st.subheader(name)
        if name not in state.predictions:
            with st.spinner("Thinking…"):
                try:
                    prediction = state.prediction_jobs[name].result(timeout=35)
                except TimeoutError:
                    prediction = Prediction(None, 35000, ok=False, note="The request timed out.")
                if state.hand_id != request_id or state.selected_model != model_id:
                    st.stop()
                state.predictions[name] = prediction
                state.prediction_jobs.pop(name, None)
        render_prediction(spot, state.predictions[name])

st.divider()
st.markdown(f"**PokerBench reference action:** `{example.target_action}`")
st.caption(
    "The reference is never passed to either engine. Probabilities estimate agreement with "
    "the dataset reference; they are not validated poker strategy frequencies. "
    "Switching encoder/head keeps a compatible hand and reuses its LLM result. "
    "New hand or Re-run engines requests a new comparison."
)
