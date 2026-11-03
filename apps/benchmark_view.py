"""Small benchmark using the package's canonical metric definitions."""

import statistics
import time
from collections.abc import Callable

import pandas as pd
import streamlit as st

from apps.backend import HandSource, call_limits, groq_ready
from apps.contract import Prediction, Spot
from apps.ui_common import problem_with, run_safely
from poker_decision.evaluation import evaluate


def draw_benchmark(engines: dict[str, Callable[[Spot], Prediction]], *, model_id: str) -> None:
    st.subheader("Benchmark")
    st.caption("A development comparison on the same held-out multi-option hands.")
    live = st.checkbox("Include live Groq requests", value=False, disabled=not groq_ready())
    count = int(st.number_input("Hands", min_value=1, max_value=5 if live else 50, value=5))
    seed = int(st.number_input("Seed", min_value=0, max_value=1000000, value=42))
    if live:
        st.caption(f"This makes up to {count} live requests from the shared allowance.")
    if st.button("Run benchmark", type="primary"):
        source = HandSource(seed, model_id=model_id)
        examples = [source.next() for _ in range(count)]
        names = list(engines) if live else list(engines)[:1]
        results: dict[str, list[Prediction]] = {}
        progress = st.progress(0.0)
        for engine_number, name in enumerate(names):
            results[name] = []
            for i, example in enumerate(examples):
                if engine_number and i:
                    time.sleep(call_limits().cooldown_seconds)
                spot = Spot(example.inputs.context, example.inputs.candidates)
                results[name].append(run_safely(engines[name], spot))
                progress.progress((engine_number * count + i + 1) / (len(names) * count))
        progress.empty()
        st.session_state.demo_benchmark = {"examples": examples, "results": results}
    state = st.session_state.get("demo_benchmark")
    if state is None:
        st.info("Choose a sample and run the benchmark.")
        return
    examples, results = state["examples"], state["results"]
    common = [
        i
        for i, example in enumerate(examples)
        if all(
            problem_with(Spot(example.inputs.context, example.inputs.candidates), ps[i]) is None
            for ps in results.values()
        )
    ]
    rows = {}
    for name, predictions in results.items():
        answered = [
            p
            for e, p in zip(examples, predictions, strict=True)
            if problem_with(Spot(e.inputs.context, e.inputs.candidates), p) is None
        ]
        latencies = [p.latency_ms for p in answered]
        row = {"Answered": f"{len(answered)} / {len(examples)}"}
        if common:
            values = iter([predictions[i].probs for i in common])
            metrics = evaluate([examples[i] for i in common], lambda _, values=values: next(values))
            row.update(
                {
                    "Reference agreement": f"{metrics['exact_accuracy']:.1%}",
                    "Log loss": f"{metrics['log_loss']:.3f}",
                    "Brier score": f"{metrics['brier_score']:.3f}",
                }
            )
        if latencies:
            p95 = (
                statistics.quantiles(latencies, n=20, method="inclusive")[18]
                if len(latencies) > 1
                else latencies[0]
            )
            row["Median response"] = f"{statistics.median(latencies):.0f} ms"
            row["p95 response"] = f"{p95:.0f} ms"
        rows[name] = row
    st.table(pd.DataFrame(rows))
    st.caption(
        f"Quality metrics use the {len(common)} hands answered successfully by every included "
        "engine. Response times use each engine's successful requests. "
        "This small sample does not establish overall performance."
    )
    for name, predictions in results.items():
        for prediction in predictions:
            if not prediction.ok:
                st.warning(f"{name}: {prediction.note}")
