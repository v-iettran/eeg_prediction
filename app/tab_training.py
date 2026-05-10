"""Tab 2: Training Summary — offline model evaluation results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.colors import T1_BLUE

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

MODEL_DISPLAY = [
    {"key": "v1_rf", "name": "Random Forest (V1)", "family": "Classical", "type": "Random Forest", "features": "6-dim band power (mu, beta)", "channels": "C3, Cz, C4", "source_file": "rf_results.json", "source_type": "v1"},
    {"key": "v2_rf", "name": "Random Forest (V2)", "family": "Classical", "type": "Random Forest", "features": "31-dim enriched band power", "channels": "C3, Cz, C4", "source_file": "v2_classical_results.json", "source_type": "array"},
    {"key": "v2_xgb", "name": "XGBoost", "family": "Classical", "type": "Gradient Boosted Trees", "features": "31-dim enriched band power", "channels": "C3, Cz, C4", "source_file": "v2_classical_results.json", "source_type": "array"},
    {"key": "v1_csp_lda", "name": "CSP+LDA (V1)", "family": "Spatial", "type": "CSP + LDA", "features": "CSP log-variance", "channels": "All 64 EEG channels", "source_file": "csp_lda_results.json", "source_type": "v1"},
    {"key": "v2_csp_lda", "name": "CSP+LDA (V2)", "family": "Spatial", "type": "CSP + LDA", "features": "CSP log-variance (8 components)", "channels": "All 64 EEG channels", "source_file": "v2_csp_eegnet_results.json", "source_type": "dict"},
    {"key": "v1_eegnet", "name": "EEGNet (V1)", "family": "Deep Learning", "type": "Compact CNN (Lawhern 2018)", "features": "Raw EEG -> learned features", "channels": "All 64 EEG channels", "source_file": "eegnet_results.json", "source_type": "v1"},
    {"key": "v2_eegnet", "name": "EEGNet (V2)", "family": "Deep Learning", "type": "Compact CNN (Lawhern 2018)", "features": "Raw EEG -> learned features", "channels": "All 64 EEG channels", "source_file": "v2_csp_eegnet_results.json", "source_type": "dict"},
    {"key": "v2_tcnet", "name": "EEG-TCNet", "family": "Deep Learning", "type": "EEGNet + Temporal Convolutional Net", "features": "Raw EEG -> TCN temporal dynamics", "channels": "All 64 EEG channels", "source_file": "v2_tcnet_fbcnet_results.json", "source_type": "dict"},
    {"key": "v2_fbcnet", "name": "FBCNet", "family": "Deep Learning", "type": "Filter Bank CSP Network", "features": "Multi-band spatial + temporal variance", "channels": "All 64 EEG channels", "source_file": "v2_tcnet_fbcnet_results.json", "source_type": "dict"},
]

ALL_TEST_F1 = {
    "v1_rf": 0.5504075504075504,
    "v1_csp_lda": 0.5705489232878277,
    "v1_eegnet": 0.708808053077099,
    "v2_rf": 0.5434064814185746,
    "v2_xgb": 0.5776924926162512,
    "v2_csp_lda": 0.5994204219024841,
    "v2_eegnet": 0.7574701195219123,
    "v2_tcnet": 0.6745656420273466,
    "v2_fbcnet": 0.7034004398578921,
}


@st.cache_data
def _load_report(filename: str) -> dict | None:
    path = REPORTS / filename
    if path.exists():
        return json.loads(path.read_text())
    return None


def _get_model_metrics(info: dict) -> dict | None:
    """Extract metrics for a specific model from its report file."""
    report = _load_report(info["source_file"])
    if report is None:
        return None

    key = info["key"]

    if info["source_type"] == "v1":
        windows = report.get("windows_evaluated", [])
        winning_window = report.get("winning_window")
        winning = next(
            (w for w in windows if w.get("window") == winning_window),
            windows[0] if windows else {},
        )
        return {
            "val_accuracy": winning.get("val_accuracy"),
            "val_macro_f1": winning.get("val_macro_f1"),
            "val_confusion_matrix": winning.get("val_confusion_matrix"),
            "training_time_sec": winning.get("training_time_sec"),
            "n_params": report.get("model_config", {}).get("n_params"),
            "n_train_history_epochs": winning.get("n_train_history_epochs"),
        }

    models_data = report.get("models", {})

    if info["source_type"] == "dict":
        return models_data.get(key)
    elif info["source_type"] == "array" and isinstance(models_data, list):
        for m in models_data:
            if m.get("name") == key:
                return m
    return None


def _make_confusion_matrix_fig(cm: list, size: int = 220) -> go.Figure:
    """Create a small inline confusion matrix heatmap."""
    cm_arr = np.array(cm)
    cm_norm = cm_arr / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)

    labels = ["Left (T1)", "Right (T2)"]
    text = [
        [f"{cm_arr[i, j]}<br>({cm_norm[i, j] * 100:.0f}%)" for j in range(2)]
        for i in range(2)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#F1F5F9"], [1, T1_BLUE]],
        showscale=False,
        zmin=0,
        zmax=1,
    ))
    fig.update_layout(
        width=size, height=size,
        margin=dict(l=5, r=5, t=25, b=5),
        xaxis=dict(title="Predicted", side="bottom", tickfont=dict(size=10)),
        yaxis=dict(title="True", tickfont=dict(size=10), autorange="reversed"),
        font=dict(size=10),
    )
    return fig


def _render_pipeline_sankey():
    """Interactive Sankey diagram showing the full training pipeline."""
    node_labels = [
        "<b>EDF</b><br>64 ch, 160 Hz",
        "<b>Bandpass</b><br>8-30 Hz IIR",
        "<b>Avg Reference</b><br>Common average",
        "<b>Euclidean Align</b><br>Per-subject",
        "<b>Epochs</b><br>T1/T2, 0-4 s",
        "<b>Band Power</b><br>31 features",
        "<b>CSP</b><br>8 components",
        "<b>Raw Epochs</b><br>64 x 641",
        "<b>Filter Bank</b><br>5 bands",
        "<b>Classical</b><br>RF/XGB",
        "<b>CSP+LDA</b><br>F1 0.599",
        "<b>EEGNet</b><br>F1 0.757",
        "<b>EEG-TCNet</b><br>F1 0.675",
        "<b>FBCNet</b><br>F1 0.703",
    ]

    node_colors = [
        "#94A3B8", "#94A3B8", "#94A3B8", "#94A3B8", "#94A3B8",
        "#E2E8F0", "#BFDBFE", "#DBEAFE", "#DBEAFE",
        "#CBD5E1", "#93C5FD", "#3B82F6", "#60A5FA", "#60A5FA",
    ]

    sources = [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 7, 8]
    targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    values = [8, 8, 8, 8, 2, 5, 5, 5, 2, 5, 5, 5, 5]

    edge_colors = [
        "rgba(100,116,139,0.3)", "rgba(100,116,139,0.3)",
        "rgba(100,116,139,0.3)", "rgba(100,116,139,0.3)",
        "rgba(148,163,184,0.3)", "rgba(96,165,250,0.3)",
        "rgba(74,144,217,0.3)", "rgba(74,144,217,0.3)",
        "rgba(148,163,184,0.3)", "rgba(96,165,250,0.3)",
        "rgba(37,99,235,0.4)", "rgba(74,144,217,0.3)", "rgba(74,144,217,0.3)",
    ]

    fig = go.Figure(data=[go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=20,
            thickness=18,
            label=node_labels,
            color=node_colors,
            line=dict(color="#CBD5E1", width=0.7),
            x=[0.02, 0.16, 0.30, 0.44, 0.58, 0.74, 0.74, 0.74, 0.74, 0.94, 0.94, 0.94, 0.94, 0.94],
            y=[0.48, 0.48, 0.48, 0.48, 0.48, 0.05, 0.28, 0.52, 0.82, 0.05, 0.28, 0.46, 0.60, 0.82],
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=edge_colors,
        ),
    )])

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=13, color="#000000", family="sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption("**Dataset**\nPhysioNet eegmmidb v1.0.0\nRuns 4, 8, 12")
    with c2:
        st.caption("**Subjects**\n104 (5 excluded)\n80 train / 12 val / 12 test")
    with c3:
        st.caption("**Preprocessing**\n8–30 Hz IIR, avg reference\n300 µV PTP rejection")
    with c4:
        st.caption("**Evaluation**\nSubject-disjoint split\nGroupKFold CV (k=5)")


def _fmt_params_display(metrics: dict | None) -> str:
    if not metrics:
        return "–"
    n_params = metrics.get("n_params")
    if n_params is None:
        return "–"
    try:
        return f"{int(n_params):,}"
    except (TypeError, ValueError):
        return str(n_params)


def _format_test_f1(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _render_all_models_chart() -> go.Figure:
    """Grouped bar chart of test F1 for all models with available test metrics."""
    family_colors = {
        "Classical": "#94A3B8",
        "Spatial": "#60A5FA",
        "Deep Learning": "#3B82F6",
    }
    family_order = list(family_colors.keys())

    sorted_models = sorted(
        MODEL_DISPLAY,
        key=lambda m: (
            family_order.index(m["family"]),
            ALL_TEST_F1.get(m["key"]) or 0,
        ),
    )
    models_with_test = [m for m in sorted_models if ALL_TEST_F1.get(m["key"]) is not None]

    fig = go.Figure()
    best_f1 = max(ALL_TEST_F1[m["key"]] for m in models_with_test)

    for model in models_with_test:
        f1 = ALL_TEST_F1[model["key"]]
        is_best = f1 == best_f1
        fig.add_trace(go.Bar(
            y=[model["name"]],
            x=[f1],
            orientation="h",
            marker_color=family_colors[model["family"]],
            marker_line=dict(width=2.5, color="#1E293B") if is_best else dict(width=0),
            text=[f"{f1:.3f}"],
            textposition="outside",
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(title="Test Macro-F1", range=[0, 0.85]),
        yaxis=dict(title=""),
        height=40 * len(models_with_test) + 80,
        margin=dict(l=10, r=50, t=10, b=40),
        bargap=0.25,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _show_image_responsive(path: Path):
    try:
        st.image(str(path), use_container_width=True)
    except TypeError:
        st.image(str(path), use_column_width=True)


def _render_model_table():
    """Compact model comparison table."""
    rows = []
    for info in MODEL_DISPLAY:
        metrics = _get_model_metrics(info)
        test_f1 = ALL_TEST_F1.get(info["key"])
        val_f1 = np.nan
        if metrics:
            val_f1 = float(metrics.get("val_macro_f1", np.nan) or np.nan)

        rows.append({
            "Model": info["name"],
            "Family": info["family"],
            "Channels": info["channels"],
            "Val F1": val_f1,
            "Test F1": _format_test_f1(test_f1),
            "Params": _fmt_params_display(metrics),
        })

    df = pd.DataFrame(rows)

    def highlight_best(s: pd.Series):
        if s.name in ("Val F1", "Test F1"):
            is_max = s == s.max(skipna=True)
            return ["font-weight: bold; color: #2563EB" if v else "" for v in is_max]
        return [""] * len(s)

    styled = (
        df.style
        .apply(highlight_best, axis=0)
        .format({"Val F1": "{:.3f}", "Test F1": "{:.3f}"}, na_rep="-")
    )

    st.dataframe(styled, use_container_width=True, hide_index=True, height=390)


def _render_model_detail():
    """Family-grouped model detail panel."""
    st.caption("Select a family and model to inspect architecture notes and validation behavior.")
    families = ["Classical", "Spatial", "Deep Learning"]
    selected_family = st.radio(
        "Model family",
        options=families,
        horizontal=True,
        key="family_radio",
    )

    family_models = [m for m in MODEL_DISPLAY if m["family"] == selected_family]
    model_names = [m["name"] for m in family_models]
    selected = st.radio(
        "Model",
        options=model_names,
        horizontal=True,
        key="model_detail_radio",
    )

    info = family_models[model_names.index(selected)]
    metrics = _get_model_metrics(info)

    col_desc, col_cm = st.columns([2, 1], gap="large")

    with col_desc:
        st.markdown(f"#### {info['name']}")
        st.markdown(
            f"**Type:** {info['type']}  \n"
            f"**Input:** {info['features']}  \n"
            f"**Channels:** {info['channels']}"
        )
        if info["key"] in {"v2_xgb", "v2_tcnet", "v2_fbcnet"}:
            st.caption(
                "No V1 baseline is shown for this model because it was introduced in the V2 experiment set."
            )

        if metrics:
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Validation F1", f"{float(metrics.get('val_macro_f1', 0) or 0):.3f}")
            with metric_cols[1]:
                test_f1 = ALL_TEST_F1.get(info["key"])
                st.metric("Test F1", "-" if test_f1 is None else f"{test_f1:.3f}")
            with metric_cols[2]:
                st.metric("Parameters", _fmt_params_display(metrics))

            train_time = metrics.get("training_time_sec")
            if train_time:
                st.markdown(f"**Training time:** {train_time:.0f}s")

            n_epochs_trained = metrics.get("n_train_history_epochs")
            if n_epochs_trained:
                st.markdown(f"**Training epochs:** {n_epochs_trained} (early stopped)")

    with col_cm:
        if metrics:
            cm = metrics.get("val_confusion_matrix") or metrics.get("confusion_matrix")
            if cm:
                st.markdown("**Validation Confusion Matrix**")
                fig_cm = _make_confusion_matrix_fig(cm, size=280)
                st.plotly_chart(fig_cm, use_container_width=True)

    curve_map = {
        "EEGNet (V2)": "reports/figures/v2_eegnet_training_curves.png",
        "EEGNet (V1)": "reports/figures/eegnet_training_curves.png",
        "EEG-TCNet": "reports/figures/v2_tcnet_training_curves.png",
        "FBCNet": "reports/figures/v2_fbcnet_training_curves.png",
    }
    alt_viz_map = {
        "Random Forest (V1)": "reports/figures/rf_feature_importance.png",
        "Random Forest (V2)": "reports/figures/v2_feature_importance.png",
        "CSP+LDA (V1)": "reports/figures/csp_topomaps.png",
        "CSP+LDA (V2)": "reports/figures/csp_topomaps.png",
    }

    curve_path = ROOT / curve_map.get(selected, "__nonexistent__")
    alt_path = ROOT / alt_viz_map.get(selected, "__nonexistent__")
    if curve_path.exists():
        with st.expander("Training curves"):
            _show_image_responsive(curve_path)
    elif alt_path.exists():
        viz_label = "Feature importance" if "importance" in str(alt_path) else "CSP spatial patterns"
        with st.expander(viz_label):
            _show_image_responsive(alt_path)


def _render_v1_v2_comparison():
    """Show the V1 → V2 improvement story."""
    v1_v2_pairs = [
        ("Random Forest", ALL_TEST_F1["v1_rf"], ALL_TEST_F1["v2_rf"]),
        ("CSP+LDA", ALL_TEST_F1["v1_csp_lda"], ALL_TEST_F1["v2_csp_lda"]),
        ("EEGNet", ALL_TEST_F1["v1_eegnet"], ALL_TEST_F1["v2_eegnet"]),
    ]

    cols = st.columns(3)
    for i, (name, v1, v2) in enumerate(v1_v2_pairs):
        with cols[i]:
            improvement = (v2 - v1) / v1 * 100
            st.metric(
                label=f"{name} Test Macro-F1",
                value=f"{v2:.3f}",
                delta=f"{improvement:+.1f}% from V1 ({v1:.3f})",
            )

def _render_methodology():
    """Research rationale and model selection methodology."""
    
    st.subheader("Methodology")
    
    st.markdown("##### The Cross-Subject Challenge")
    st.markdown(
        "Cross-subject motor imagery classification is one of the hardest problems in EEG-based "
        "brain-computer interfaces. Each person's brain produces subtly different electrical "
        "patterns - electrode impedances vary, cortical folding differs, and the spatial "
        "distribution of motor imagery signals shifts across individuals. The discriminative "
        "signal I'm detecting (event-related desynchronization in the mu and beta frequency "
        "bands over motor cortex) is typically just 1–2 dB of power difference between "
        "hemispheres. With only ~45 usable epochs per subject across 3 runs, the model must "
        "learn from limited, highly variable data."
    )
    
    st.markdown("##### Why Three Model Families")
    st.markdown(
        "I chose models that form a progression from fully hand-designed "
        "to fully learned, each relaxing one assumption from the previous:"
    )
    
    st.dataframe(
        pd.DataFrame([
            {
                "Family": "Classical (RF, XGBoost)",
                "Hand-designed": "Features (mu/beta power), channels (C3/Cz/C4)",
                "Learned": "Decision boundaries only",
                "Rationale": "Neuroscience-informed baseline - tests whether known EEG features are sufficient",
            },
            {
                "Family": "Spatial (CSP+LDA)",
                "Hand-designed": "Frequency band (8–30 Hz)",
                "Learned": "Optimal spatial filters across 64 channels",
                "Rationale": "Classical BCI gold standard - learns WHERE to look on the scalp",
            },
            {
                "Family": "Deep Learning (EEGNet, TCNet, FBCNet)",
                "Hand-designed": "Architecture inductive biases only",
                "Learned": "Spatial filters, temporal filters, and features jointly",
                "Rationale": "End-to-end learning - tests whether the model can discover better representations",
            },
        ]),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("##### V1 → V2: Diagnosing Before Scaling")
    st.markdown(
        "My V1 pipeline achieved 70.9% F1 with EEGNet but only 55–57% with classical models. "
        "Before adding more architectures, I diagnosed three root causes:\n\n"
        "**1. Data loss from over-aggressive artifact rejection.** "
        "The 150 µV peak-to-peak threshold dropped 47% of epochs. "
        "Since the 8 - 30 Hz bandpass already removes the worst artifacts, "
        "I relaxed to 300 µV to recover around 2,200 epochs and doubling the effective training set.\n\n"
        "**2. Inter-subject covariance mismatch.** "
        "Spatial filters learned on one subject don't transfer well to another. "
        "I added Euclidean Alignment - a parameter-free whitening step that maps each subject's "
        "covariance to a common reference space.\n\n"
        "**3. Feature starvation in classical models.** "
        "V1's Random Forest had only 6 features. "
        "I expanded to 31 features including relative band powers, Hjorth parameters, "
        "inter-channel coherence, and hemispheric asymmetry indices."
    )
    
    st.markdown("##### EEG-TCNet and FBCNet: Targeted Architecture Choices")
    st.markdown(
        "**EEG-TCNet** (Ingolfsson 2020) replaces EEGNet's final block with a Temporal Convolutional "
        "Network using dilated causal convolutions, extending the temporal receptive field from "
        "~0.4s to ~3.0s to capture how desynchronization evolves over the full imagery period.\n\n"
        "**FBCNet** (Mane 2021) pre-defines a filter bank of 5 overlapping bands and learns separate "
        "spatial filters per band, mirroring CSP's log-variance pipeline in an end-to-end trainable form.\n\n"
        "Both underperformed EEGNet in cross-subject evaluation "
        "(TCNet: 0.675, FBCNet: 0.703 vs EEGNet: 0.757). "
        "This is consistent with findings that these architectures benefit most from "
        "within-subject calibration data rather than global cross-subject training."
    )
    
    st.markdown("##### Final Model Selection")
    st.markdown(
        "**EEGNet (V2) is the primary inference model** at 0.757 test macro-F1. "
        "It achieved the highest cross-subject performance with only 2,770 parameters. "
        "All other trained models remain available for comparison. "
        "A stacking ensemble of the three deep models (0.707 F1) did not improve over standalone "
        "EEGNet because the weaker base models added noise rather than complementary signal."
    )
    
    st.divider()


def _render_limitations_and_future():
    """Limitations and future research directions."""
    
    st.subheader("Limitations & Future Directions")
    
    st.markdown("##### Current Limitations")
    st.markdown(
        "**1. No subject-specific adaptation.** "
        "All models are trained globally across 80 subjects and applied to unseen subjects "
        "without calibration. Even a few seconds of per-user calibration data would "
        "dramatically improve accuracy. "
        "The 0.757 F1 ceiling likely reflects this constraint more than any architectural limitation.\n\n"
        "**2. Binary classification only.** "
        "The current pipeline classifies left vs right fist imagery. "
        "The PhysioNet dataset also contains feet and both-fists imagery (runs 6, 10, 14) "
        "that could extend this to a 4-class problem.\n\n"
        "**3. Fixed epoch window.** "
        "I use a fixed 0–4 second post-cue window. "
        "Motor imagery onset latency varies across individuals - "
        "an adaptive windowing strategy could capture each subject's optimal imagery period.\n\n"
        "**4. No ICA-based artifact removal.** "
        "I rely on bandpass filtering and a 300 µV threshold. "
        "ICA would improve signal quality at the cost of pipeline complexity."
    )
    
    st.markdown("##### Future Directions")
    st.markdown(
        "**1. Transfer learning with subject adaptation.** "
        "Fine-tune the global EEGNet on a small number of calibration epochs from a new subject. "
        "Even 10–20 labeled epochs could push accuracy above 80%.\n\n"
        "**2. Attention-based architectures.** "
        "EEG-Conformer and ATCNet use self-attention to capture "
        "long-range temporal dependencies and cross-channel relationships simultaneously.\n\n"
        "**3. Multimodal fusion for fatigue detection.** "
        "This project focused on motor imagery classification as a foundation for EEG signal processing. "
        "The natural extension is combining EEG with wearable sensor data (heart rate, "
        "skin conductance, accelerometry) for fatigue detection. The preprocessing pipeline, "
        "feature extraction, and model evaluation framework developed here transfer directly "
        "to that multimodal setting.\n\n"
        "**4. Deeper explainability.** "
        "Gradient-based saliency maps showing which time-frequency regions EEGNet attends to, "
        "CSP spatial pattern visualization, and per-epoch confidence calibration analysis."
    )

def render_training_tab():
    """Render the complete training summary tab."""
    _render_methodology()
    st.subheader("Development Journey: V1 → V2")
    st.markdown(
        "I systematically improved the pipeline across two iterations. "
        "V1 used conservative artifact rejection (150 µV), training on ~1,850 epochs. "
        "V2 relaxed rejection to 300 µV, added Euclidean Alignment for cross-subject "
        "normalization, enriched features from 6 to 31 dimensions, and introduced "
        "additional model architectures - recovering ~2,200 additional training epochs. "
        "The numbers below are held-out subject test Macro-F1 scores."
    )
    _render_v1_v2_comparison()
    st.caption(
        "XGBoost, EEG-TCNet, and FBCNet do not have V1 "
        "baselines because they were introduced only in the V2 experiment set."
    )

    st.divider()

    st.subheader("Training Pipeline")
    _render_pipeline_sankey()

    st.divider()

    st.subheader("Model Results")
    _render_model_table()

    st.divider()
    st.subheader("All Models - Test Performance")
    st.plotly_chart(_render_all_models_chart(), use_container_width=True)

    st.divider()
    st.subheader("Model Detail")
    _render_model_detail()
    st.divider()
    _render_limitations_and_future()
