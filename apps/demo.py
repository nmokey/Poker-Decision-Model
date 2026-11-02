"""PocketSolver: the trained decision model and live Groq on the same hand."""

import uuid
from concurrent.futures import ThreadPoolExecutor
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
    model_engine,
)
from apps.benchmark_view import draw_benchmark
from apps.contract import Prediction, Spot
from apps.spot_view import draw_spot
from apps.ui_common import fmt_ms, problem_with, run_safely
from poker_decision.data import DecisionInputs
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
    state.source = HandSource()
    state.hand_id = 0
    state.predictions = {}

st.title("POCKETSOLVER")
st.caption("A small trained decision model versus a generative LLM · same hand, same options")
limits = call_limits()
with st.sidebar:
    st.markdown("**Live LLM allowance**")
    st.caption(
        f"Up to {limits.session_daily} requests per session per day, "
        f"with {limits.daily} shared across this demo. "
        "The local model remains available when the allowance is used."
    )
    st.caption("BERT Mini + trained linear decision head · CPU")
    st.caption(f"Groq model: {groq_model_id()}")
    if not groq_ready():
        st.info("Live LLM comparison is not configured yet.")
try:
    with st.spinner("Loading the saved decision model…"):
        loaded_model()
    st.caption(f"{len(held_out_examples()):,} multi-option hands held out by this checkpoint")
except Exception:
    st.error("The saved model could not be loaded. Please try again later.")
    st.stop()

view = st.segmented_control("View", ["Single hand", "Benchmark"], default="Single hand")
engines = {
    "BERT Mini · trained model": model_engine,
    f"Groq · {groq_model_id()}": partial(llm_engine, session=state.demo_session),
}
if view == "Benchmark":
    draw_benchmark(engines)
    st.stop()

new, repeat = st.columns([1, 3])
if new.button("New hand", type="primary"):
    state.example = state.source.next()
    state.hand_id += 1
    state.predictions = {}
rerun = repeat.button("Re-run engines", disabled="example" not in state)
if "example" not in state:
    st.info("Click New hand to compare the trained model and live LLM.")
    st.stop()
if rerun:
    state.predictions = {}
request_id = state.hand_id
example = state.example
spot = Spot(example.inputs.context, example.inputs.candidates)
draw_spot(spot.context)
st.markdown("**Available actions:** " + " · ".join(f"`{c}`" for c in spot.candidates))
with st.expander("Exact input sent to both engines"):
    st.code(f"{spot.context}\n\nCandidates: {list(spot.candidates)}")

# Start both computations before drawing the panels; the local result is shown first.
missing = {name: engine for name, engine in engines.items() if name not in state.predictions}
pool = ThreadPoolExecutor(max_workers=2)
jobs = {name: pool.submit(run_safely, engine, spot) for name, engine in missing.items()}
try:
    for name, column in zip(engines, st.columns(2), strict=True):
        with column.container(border=True):
            st.subheader(name)
            if name not in state.predictions:
                with st.spinner("Thinking…"):
                    try:
                        prediction = jobs[name].result(timeout=35)
                    except TimeoutError:
                        prediction = Prediction(
                            None, 35000, ok=False, note="The request timed out."
                        )
                    if state.hand_id != request_id:
                        st.stop()
                    state.predictions[name] = prediction
            render_prediction(spot, state.predictions[name])
finally:
    pool.shutdown(wait=False, cancel_futures=True)

st.divider()
st.markdown(f"**PokerBench reference action:** `{example.target_action}`")
st.caption(
    "The reference is never passed to either engine. Probabilities estimate agreement with "
    "the dataset reference; they are not validated poker strategy frequencies. "
    "Results stay saved until New hand or Re-run engines is selected."
)
