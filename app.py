"""EEG Motor Imagery Analysis — Streamlit Application.

Upload a PhysioNet EDF file, explore EEG signals, review offline training
results, and run real-time motor imagery predictions with trained models.
"""
from __future__ import annotations

import math

import streamlit as st

from app.tab_explore import render_explore_tab
from app.tab_training import render_training_tab
from app.tab_predict import render_predict_tab
from app.utils.edf_loader import load_edf, validate_edf, has_motor_imagery_events, _file_hash
from app.utils.inference import get_available_models, MODEL_REGISTRY, run_inference

st.set_page_config(
    page_title="EEG Motor Imagery Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _apply_app_theme():
    """Small CSS layer for a more restrained, dashboard-like Streamlit UI."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        h2, h3 {
            color: #0F172A;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: -0.01em;
        }
        [data-testid="stMetric"] {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.85rem 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: #64748B;
        }
        [data-testid="stMetricValue"] {
            color: #0F172A;
        }
        .section-note {
            color: #64748B;
            font-size: 0.95rem;
            margin-top: -0.5rem;
            margin-bottom: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_app_theme()

# ── Sidebar ──────────────────────────────────────────────────────────────

edf_data = None
file_hash = ""
selected_channel = "C3"
selected_time_range = "Full recording"
selected_model = None

with st.sidebar:
    st.title("EEG Analysis")
    st.divider()

    uploaded_file = st.file_uploader("Upload EDF file", type=["edf"])
    st.caption("Supported: PhysioNet EEG Motor Movement/Imagery Dataset (Runs 4, 8, 12)")

    if uploaded_file is not None:
        try:
            file_hash = _file_hash(uploaded_file)
            edf_data = load_edf(uploaded_file, file_hash)
        except Exception as e:
            st.error(f"Could not read EDF file: {e}")
            st.stop()

        issues = validate_edf(edf_data)
        for issue in issues:
            if "No T1/T2" in issue:
                st.error(issue)
            else:
                st.warning(issue)

        if not has_motor_imagery_events(edf_data):
            st.error("This file has no motor imagery events. Please use runs 4, 8, or 12.")
            edf_data = None

    if edf_data is not None:
        st.divider()

        st.markdown("**Explore controls**")

        ch_names = edf_data["ch_names"]
        default_idx = ch_names.index("C3") if "C3" in ch_names else 0
        selected_channel = st.selectbox("Channel", options=ch_names, index=default_idx)

        duration = float(edf_data["duration"])
        time_options: list[str] = []
        step = 30
        for start in range(0, int(duration), step):
            end = min(start + step, int(math.ceil(duration)))
            time_options.append(f"{start}–{end} s")
        time_options.append("Full recording")
        selected_time_range = st.selectbox("Time range", options=time_options, index=0)

        st.divider()

        st.markdown("**Prediction controls**")
        available_models = get_available_models()

        if available_models:
            families = ["All models", "Classical", "Spatial", "Deep Learning"]
            selected_family = st.selectbox("Model family", families, key="sidebar_model_family")

            if selected_family == "All models":
                model_options = available_models
            else:
                model_options = [
                    name for name in available_models
                    if MODEL_REGISTRY[name].get("family") == selected_family
                ]

            if model_options:
                default_model = "EEGNet" if "EEGNet" in model_options else model_options[0]
                selected_model = st.selectbox(
                    "Model",
                    options=model_options,
                    index=model_options.index(default_model),
                    format_func=(
                        lambda name: f"{MODEL_REGISTRY[name].get('family', 'Other')} · {name}"
                        if selected_family == "All models" else name
                    ),
                    key=f"sidebar_model_{selected_family}",
                )
                st.caption(MODEL_REGISTRY[selected_model]["description"])

                if st.button("Run prediction", type="primary", use_container_width=True):
                    with st.spinner("Running inference..."):
                        results = run_inference(edf_data, selected_model)
                        if results is not None:
                            st.session_state["predictions"] = results
                            st.session_state["prediction_model"] = selected_model
                            st.success(
                                f"Predicted {len(results['epoch_info'])} epochs — "
                                f"Accuracy: {results['accuracy']:.1%}  F1: {results['macro_f1']:.3f}"
                            )
                        else:
                            st.error("No T1/T2 epochs found for prediction.")
            else:
                st.warning(f"No available models in {selected_family}.")
        else:
            st.warning("No model files found. Train models first.")

    st.divider()
    st.info("T1 = imagined left fist\n\nT2 = imagined right fist")

# ── Main Tabs ────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Explore", "Training", "Predict"])

with tab1:
    render_explore_tab(edf_data, selected_channel, file_hash, selected_time_range)

with tab2:
    render_training_tab()

with tab3:
    render_predict_tab(edf_data, selected_model)
