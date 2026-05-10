"""Tab 3: Predict & Evaluate — run inference and display results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.colors import (
    T1_BLUE, T2_ORANGE,
    CORRECT_GREEN, INCORRECT_RED,
    LABEL_COLORS, LABEL_NAMES,
)


def _make_prediction_timeline(epoch_info: list[dict], duration: float) -> go.Figure:
    """Create dual-row timeline: ground truth (top) and prediction (bottom)."""
    fig = go.Figure()

    for info in epoch_info:
        true_color = LABEL_COLORS[info["true_label"]]
        pred_color = LABEL_COLORS[info["pred_label"]]
        correct = info["correct"]

        # Ground truth rectangle (top row)
        fig.add_shape(
            type="rect",
            x0=info["start_time"], x1=info["end_time"],
            y0=0.55, y1=0.95,
            fillcolor=true_color,
            line=dict(width=0.5, color="white"),
            opacity=0.85,
        )

        # Prediction rectangle (bottom row)
        fig.add_shape(
            type="rect",
            x0=info["start_time"], x1=info["end_time"],
            y0=0.05, y1=0.45,
            fillcolor=pred_color,
            line=dict(
                width=2.5 if not correct else 0.5,
                color=INCORRECT_RED if not correct else "white",
            ),
            opacity=0.85,
        )

        # Confidence text in prediction bar
        mid_t = (info["start_time"] + info["end_time"]) / 2
        fig.add_annotation(
            x=mid_t, y=0.25,
            text=f"{info['confidence'] * 100:.0f}%",
            showarrow=False,
            font=dict(size=9, color="white"),
        )

    fig.update_layout(
        xaxis=dict(title="Time (s)", range=[0, duration]),
        yaxis=dict(
            tickvals=[0.25, 0.75],
            ticktext=["Prediction", "Ground Truth"],
            range=[0, 1],
            fixedrange=True,
        ),
        height=200,
        margin=dict(l=80, r=20, t=30, b=40),
        title="Prediction Timeline",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Legend annotations
    fig.add_annotation(
        x=0.85, y=1.08, xref="paper", yref="paper",
        text=f"<span style='color:{T1_BLUE}'>■</span> Left (T1)  "
             f"<span style='color:{T2_ORANGE}'>■</span> Right (T2)  "
             f"<span style='color:{INCORRECT_RED}'>■</span> Wrong",
        showarrow=False, font=dict(size=11),
    )

    return fig


def _make_confusion_matrix(cm: list) -> go.Figure:
    """Create confusion matrix heatmap for this file's predictions."""
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
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        width=300, height=300,
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="True", autorange="reversed"),
        margin=dict(l=10, r=10, t=35, b=10),
        title="Confusion Matrix (this file)",
    )
    return fig


def _make_confidence_lollipop(epoch_info: list[dict]) -> go.Figure:
    """Per-epoch confidence lollipop chart, colored by correct/incorrect."""
    fig = go.Figure()

    for ep in epoch_info:
        color = CORRECT_GREEN if ep["correct"] else INCORRECT_RED
        y_label = f"Epoch {ep['epoch']}"
        status = "Correct" if ep["correct"] else "Incorrect"

        fig.add_trace(
            go.Scatter(
                x=[ep["confidence"]],
                y=[y_label],
                mode="markers",
                marker=dict(size=10, color=color, symbol="circle"),
                showlegend=False,
                hovertext=(
                    f"{status}: {ep['confidence'] * 100:.0f}%"
                    f" | True: {LABEL_NAMES[ep['true_label']]}"
                    f" | Pred: {LABEL_NAMES[ep['pred_label']]}"
                ),
                hoverinfo="text",
            )
        )

        fig.add_shape(
            type="line",
            x0=0.5,
            x1=ep["confidence"],
            y0=y_label,
            y1=y_label,
            line=dict(color=color, width=2),
        )

    fig.update_layout(
        xaxis=dict(title="Confidence", range=[0.45, 1.02], tickformat=".0%"),
        yaxis=dict(autorange="reversed", title=""),
        height=max(180, 28 * len(epoch_info)),
        margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.add_annotation(
        x=0.98,
        y=1.05,
        xref="paper",
        yref="paper",
        text=(
            f"<span style='color:{CORRECT_GREEN}'>●</span> Correct  "
            f"<span style='color:{INCORRECT_RED}'>●</span> Incorrect"
        ),
        showarrow=False,
        font=dict(size=11),
    )

    return fig


def render_predict_tab(edf_data: dict | None, selected_model: str | None):
    """Render the predict & evaluate tab."""

    if edf_data is None:
        st.markdown(
            "<div style='text-align: center; padding: 80px 20px;'>"
            "<h3>Upload an EDF file in the sidebar to begin</h3>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    results = st.session_state.get("predictions")
    prediction_model = st.session_state.get("prediction_model")

    if selected_model is None:
        st.info("Upload an EDF file and choose a model in the sidebar.")
        return

    if results is None or prediction_model != selected_model:
        st.info(f"Run prediction from the sidebar to view results for **{selected_model}**.")
        return

    epoch_info = results["epoch_info"]
    duration = edf_data["duration"]

    # Prediction timeline
    fig_timeline = _make_prediction_timeline(epoch_info, duration)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Summary metrics
    st.subheader("Summary")
    n_correct = sum(1 for e in epoch_info if e["correct"])
    n_total = len(epoch_info)
    avg_conf = float(np.mean([e["confidence"] for e in epoch_info]))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", f"{results['accuracy']:.3f}")
    with c2:
        st.metric("Macro F1", f"{results['macro_f1']:.3f}")
    with c3:
        st.metric("Correct Epochs", f"{n_correct} / {n_total}")
    with c4:
        st.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")

    st.markdown("**Per-epoch confidence**")
    st.caption("Each row maps to an epoch, making low-confidence and incorrect epochs easier to locate.")
    st.plotly_chart(_make_confidence_lollipop(epoch_info), use_container_width=True)

    # Epoch table and confusion matrix
    col_table, col_cm = st.columns([3, 1])

    with col_table:
        st.subheader("Epoch-Level Results")
        rows = []
        for info in epoch_info:
            rows.append({
                "Epoch": info["epoch"],
                "Start (s)": f"{info['start_time']:.1f}",
                "End (s)": f"{info['end_time']:.1f}",
                "Ground Truth": LABEL_NAMES[info["true_label"]],
                "Prediction": LABEL_NAMES[info["pred_label"]],
                "Confidence": float(info["confidence"] * 100),
                "Correct": bool(info["correct"]),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
                "Correct": st.column_config.CheckboxColumn(
                    "Correct",
                    disabled=True,
                ),
            },
        )

    with col_cm:
        st.subheader("Confusion Matrix")
        fig_cm = _make_confusion_matrix(results["confusion_matrix"])
        st.plotly_chart(fig_cm, use_container_width=True)
