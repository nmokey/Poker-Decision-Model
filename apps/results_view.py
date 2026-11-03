"""Saved results view; viewing and changing its selection require no inference."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from poker_decision.demo_models import ENCODERS, HEADS, ModelSpec
from poker_decision.demo_results import load_results

ROOT = Path(__file__).resolve().parents[1]


def draw_results(model_id: str, models: dict[str, ModelSpec]) -> None:
    st.subheader("Results matrix")
    st.caption(f"Current selection: {models[model_id].label}")
    quality = ROOT / "docs/results/week2-development.csv"
    latency = ROOT / "docs/results/week2-serving-latency.json"
    try:
        rows = load_results(quality, latency, models)
        report = json.loads(latency.read_text())
    except (ValueError, KeyError, OSError):
        st.error("Saved results are unavailable or do not match the current model catalogue.")
        return
    frame = pd.DataFrame(
        [
            {
                "Selection": "● Current" if row.model_id == model_id else "",
                "Encoder": ENCODERS[models[row.model_id].encoder][0],
                "Head": HEADS[models[row.model_id].head],
                "Accuracy (%)": row.accuracy * 100,
                "Log loss": row.log_loss,
                "Avg latency (ms)": row.mean_ms,
            }
            for row in rows
        ]
    )
    selected_index = next(i for i, row in enumerate(rows) if row.model_id == model_id)
    styled = frame.style.apply(
        lambda row: (
            [
                "background-color: #173f59; color: #ffffff; font-weight: bold;"
                if row.name == selected_index
                else ""
            ]
            * len(row)
        ),
        axis=1,
    )
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        column_config={
            "Accuracy (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
            "Avg latency (ms)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "Accuracy ↑ and log loss ↓: five-fold development evaluation, "
        f"{rows[0].n:,} held-out predictions per combination. "
        "Average latency ↓: separate local CPU benchmark; not the hosted app's response time."
    )
    with st.expander("How these results were measured"):
        settings, hardware = report["settings"], report["hardware"]
        st.write(
            f"Latency uses the same {settings['hands']} multi-option hands for every model, "
            f"repeated {settings['repeats']} times, after {settings['warmup_per_model']} warm-up "
            f"calls. Each model uses its deployed fold-1 checkpoint. "
            f"Batch size {hardware['batch_size']}, {hardware['threads']} CPU threads."
        )
        st.write(f"CPU: {hardware['cpu']}. Measured: {report['measured_at'][:10]} (UTC).")
        st.write(report["timing_scope"])
        st.write("Excluded: " + ", ".join(report["excluded"]) + ".")
        st.write(
            "These development scores were used to choose the default model; they are not "
            "an untouched final test. Accuracy measures agreement with PokerBench references, "
            "and log loss measures the probability assigned to those references. "
            "No full-dataset Groq evaluation is saved, so Groq has no row here."
        )
    st.download_button(
        "Download matrix CSV",
        frame.drop(columns="Selection").to_csv(index=False),
        file_name="pocketsolver-results.csv",
        mime="text/csv",
    )
