"""Tab 1: Explore — EDF upload, data inspection, signal visualisation, and topographic explorer."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.utils.colors import (
    T1_BLUE,
    T2_ORANGE,
    T0_GRAY,
    T1_BLUE_LIGHT,
    T2_ORANGE_LIGHT,
    EVENT_COLORS,
)
from app.utils.edf_loader import (
    get_event_times,
    get_filtered_signal,
    compute_psd,
)


def _parse_time_range(time_range_str: str, duration: float) -> tuple[float, float]:
    if time_range_str.strip() == "Full recording":
        return 0.0, float(duration)
    s = time_range_str.replace(" s", "").strip()
    for sep in ("–", "-"):
        if sep in s:
            parts = s.split(sep)
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
    return 0.0, float(duration)


def _events_in_range(events: list[dict], t_start: float, t_end: float) -> list[dict]:
    return [e for e in events if t_start <= e["onset"] <= t_end]


# ── Signal plot helpers ──────────────────────────────────────────────────


def _make_signal_plot(
    times: np.ndarray,
    signal: np.ndarray,
    events: list[dict],
    channel_name: str,
) -> go.Figure:
    """Raw EEG signal plot with event markers and shaded regions."""
    s = signal * 1e6

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=s,
            mode="lines",
            line=dict(color="#1E293B", width=0.8),
            name=channel_name,
            showlegend=False,
        )
    )

    for evt in events:
        onset = evt["onset"]
        label = evt["label"]
        color = EVENT_COLORS.get(label, T0_GRAY)

        if label in ("T1", "T2"):
            fill = T1_BLUE_LIGHT if label == "T1" else T2_ORANGE_LIGHT
            fig.add_vrect(
                x0=onset,
                x1=onset + 4.0,
                fillcolor=fill,
                layer="below",
                line_width=0,
            )

        fig.add_vline(
            x=onset,
            line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=label if label != "T0" else "",
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color=color,
        )

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        title=f"Raw EEG — {channel_name}",
        height=320,
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",
    )
    return fig


def _make_raw_vs_filtered_plot(
    times: np.ndarray,
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    channel_name: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=raw_signal * 1e6,
            mode="lines",
            name="Raw",
            line=dict(color="#93C5FD", width=0.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=filtered_signal * 1e6,
            mode="lines",
            name="Filtered (8–30 Hz)",
            line=dict(color="#1D4ED8", width=1.2),
        )
    )
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        title=f"Raw vs Filtered — {channel_name}",
        height=320,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def _make_psd_plot(freqs: np.ndarray, psd: np.ndarray, channel_name: str) -> go.Figure:
    mask = freqs <= 50
    f, p = freqs[mask], psd[mask]
    p_db = 10 * np.log10(np.maximum(p, 1e-30))

    fig = go.Figure()
    fig.add_vrect(
        x0=8,
        x1=13,
        fillcolor="rgba(96,165,250,0.15)",
        line_width=0,
        annotation_text="μ",
        annotation_position="top left",
        annotation_font_size=12,
        annotation_font_color="#60A5FA",
    )
    fig.add_vrect(
        x0=13,
        x1=30,
        fillcolor="rgba(251,191,36,0.1)",
        line_width=0,
        annotation_text="β",
        annotation_position="top left",
        annotation_font_size=12,
        annotation_font_color="#F59E0B",
    )
    for bnd in (8, 13, 30):
        fig.add_vline(x=bnd, line=dict(color="#CBD5E1", width=1, dash="dot"))
    fig.add_trace(
        go.Scatter(
            x=f,
            y=p_db,
            mode="lines",
            line=dict(color="#1E293B", width=1.2),
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (dB/Hz)",
        title=f"Power Spectral Density — {channel_name}",
        height=320,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def _make_multichannel_plot(
    times: np.ndarray,
    data: np.ndarray,
    ch_names: list[str],
    events: list[dict],
    motor_channels: tuple[str, ...] = ("C3", "Cz", "C4"),
    highlight: str | None = None,
) -> go.Figure | None:
    ch_indices = []
    for ch in motor_channels:
        if ch in ch_names:
            ch_indices.append(ch_names.index(ch))
        else:
            return None

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(motor_channels),
        vertical_spacing=0.08,
    )
    for row_i, (ch, idx) in enumerate(zip(motor_channels, ch_indices), 1):
        if highlight and ch == highlight:
            line_kw = dict(width=1.5, color=T1_BLUE)
        else:
            line_kw = dict(width=0.6, color="#94A3B8")
        fig.add_trace(
            go.Scatter(
                x=times,
                y=data[idx] * 1e6,
                mode="lines",
                line=line_kw,
                showlegend=False,
            ),
            row=row_i,
            col=1,
        )
        fig.update_yaxes(title_text="µV", row=row_i, col=1)

    for evt in events:
        label = evt["label"]
        if label == "T0":
            continue
        color = EVENT_COLORS.get(label, T0_GRAY)
        for row_i in range(1, 4):
            fig.add_vline(
                x=evt["onset"],
                line=dict(color=color, width=1, dash="dash"),
                row=row_i,
                col=1,
            )

    title = "Motor Cortex Channels"
    if highlight and highlight in motor_channels:
        title = f"Motor Cortex Channels — {highlight} highlighted"

    fig.update_layout(
        height=520,
        xaxis3=dict(title="Time (s)"),
        margin=dict(l=60, r=20, t=40, b=40),
        title=title,
    )
    return fig


def _make_multichannel_plot_highlighted(
    times: np.ndarray,
    data: np.ndarray,
    ch_names: list[str],
    events: list[dict],
    motor_channels: tuple[str, ...],
    highlight: str,
) -> go.Figure | None:
    return _make_multichannel_plot(
        times, data, ch_names, events, motor_channels, highlight=highlight
    )


# ── Topographic explorer helpers ─────────────────────────────────────────


@st.cache_data(show_spinner="Computing scalp activity map...")
def _precompute_band_power(_data, sfreq: float, file_hash: str):
    """Compute mu and beta band power per channel per 1-second window via FFT."""
    n_ch, n_samples = _data.shape
    win_samples = int(sfreq)
    n_windows = n_samples // win_samples
    usable = n_windows * win_samples

    windowed = _data[:, :usable].reshape(n_ch, n_windows, win_samples)

    hann = np.hanning(win_samples)
    fft_in = windowed * hann[np.newaxis, np.newaxis, :]
    freqs = np.fft.rfftfreq(win_samples, d=1.0 / sfreq)
    fft_vals = np.fft.rfft(fft_in, axis=-1)
    psd = (2.0 * np.abs(fft_vals) ** 2) / (sfreq * (hann ** 2).sum())

    mu_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)

    mu_db = 10 * np.log10(np.maximum(psd[:, :, mu_mask].mean(axis=-1), 1e-20))
    beta_db = 10 * np.log10(np.maximum(psd[:, :, beta_mask].mean(axis=-1), 1e-20))

    t_centers = np.arange(n_windows).astype(float) + 0.5
    return mu_db, beta_db, t_centers


@st.cache_data
def _compute_sensor_positions(ch_names_tuple: tuple[str, ...]):
    """Project electrode positions to 2D for topomap display + highlighting."""
    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos_dict = montage.get_positions()["ch_pos"]

    names, pos_3d_list, orig_indices = [], [], []
    for i, ch in enumerate(ch_names_tuple):
        if ch in ch_pos_dict:
            names.append(ch)
            pos_3d_list.append(ch_pos_dict[ch])
            orig_indices.append(i)

    if not names:
        return None, None, None, None

    pos_3d = np.array(pos_3d_list)
    x, y, z = pos_3d.T
    r_safe = np.maximum(np.sqrt(x**2 + y**2 + z**2), 1e-10)
    theta = np.arccos(np.clip(z / r_safe, -1, 1))
    phi = np.arctan2(y, x)

    pos_2d = np.column_stack([theta * np.cos(phi), theta * np.sin(phi)])

    ch_to_pos = {
        name: (float(pos_2d[j, 0]), float(pos_2d[j, 1]))
        for j, name in enumerate(names)
    }

    return names, pos_2d, np.array(orig_indices), ch_to_pos


def _get_event_at_time(events: list[dict], t: float) -> str:
    """Return the event label active at time *t*."""
    current = "T0"
    for evt in events:
        if evt["onset"] <= t:
            current = evt["label"]
        else:
            break
    return current


def _render_topomap(
    values,
    pos_2d,
    selected_channel,
    ch_to_pos,
    sensor_names,
    title,
    vlim,
    figsize=(3.6, 3.4),
):
    """Render topomap with labeled key electrodes and highlighted selection."""
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")

    # ── Border points to fill the full head circle ──
    head_radius = float(np.max(np.linalg.norm(pos_2d, axis=1)) * 1.08)

    n_border = 20
    border_angles = np.linspace(0, 2 * np.pi, n_border, endpoint=False)
    border_r = head_radius * 0.99
    border_pos = np.column_stack([
        border_r * np.cos(border_angles),
        border_r * np.sin(border_angles),
    ])
    from scipy.spatial import cKDTree
    tree = cKDTree(pos_2d)
    _, nearest_idx = tree.query(border_pos)
    border_values = values[nearest_idx]

    pos_extended = np.vstack([pos_2d, border_pos])
    values_extended = np.concatenate([values, border_values])

    # ── Render with extended data ──
    im, _ = mne.viz.plot_topomap(
        values_extended,
        pos_extended,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=vlim,
        contours=6,
        sensors=False,
        outlines=None,
        extrapolate="box",
    )

    # ── Clip to head circle ──
    clip_circle = plt.Circle((0, 0), head_radius, transform=ax.transData,
                              facecolor="none", edgecolor="none")
    ax.add_patch(clip_circle)
    im.set_clip_path(clip_circle)
    for collection in ax.collections:
        collection.set_clip_path(clip_circle)
    for line in ax.lines:
        line.set_clip_path(clip_circle)

    head = plt.Circle((0, 0), head_radius, fill=False, color="#1E293B", linewidth=1.2, zorder=20)
    ax.add_patch(head)
    nose_y = head_radius
    ax.plot(
        [-0.08 * head_radius, 0, 0.08 * head_radius],
        [0.98 * nose_y, 1.12 * nose_y, 0.98 * nose_y],
        color="#1E293B",
        linewidth=1.0,
        zorder=20,
    )
    for side in (-1, 1):
        ear = plt.Circle(
            (side * head_radius * 1.03, 0),
            head_radius * 0.09,
            fill=False,
            color="#1E293B",
            linewidth=1.0,
            zorder=20,
        )
        ax.add_patch(ear)

    ax.set_xlim(-head_radius * 1.12, head_radius * 1.12)
    ax.set_ylim(-head_radius * 1.08, head_radius * 1.16)

    key_channels = {
        "Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Pz",
    }

    for ch_name in sensor_names:
        if ch_name not in ch_to_pos:
            continue
        hx, hy = ch_to_pos[ch_name]

        if ch_name == selected_channel:
            ax.plot(
                hx, hy, "o", markersize=10,
                markerfacecolor="#FACC15", markeredgecolor="#1E293B",
                markeredgewidth=1.5, zorder=15,
            )
            ax.annotate(
                ch_name, (hx, hy + 0.06),
                ha="center", fontsize=7, fontweight="bold", color="#1E293B",
                zorder=16,
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="#FACC15",
                    alpha=0.9,
                    edgecolor="none",
                ),
            )
        elif ch_name in key_channels:
            ax.plot(
                hx, hy, "o", markersize=4,
                markerfacecolor="#1E293B", markeredgecolor="none", zorder=10,
            )
            ax.annotate(
                ch_name, (hx, hy + 0.04),
                ha="center", fontsize=5.5, color="#475569", zorder=11,
            )
        else:
            ax.plot(hx, hy, ".", markersize=2, color="#94A3B8", zorder=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Power (dB)", fontsize=8)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def _render_event_status(current_event: str, time_seconds: float):
    """Prominent status bar for the event currently shown in the topomap."""
    if current_event == "T1":
        bg_color = T1_BLUE
        text_color = "white"
        label = "T1 — Imagined Left Fist"
    elif current_event == "T2":
        bg_color = T2_ORANGE
        text_color = "white"
        label = "T2 — Imagined Right Fist"
    else:
        bg_color = "#E2E8F0"
        text_color = "#475569"
        label = "T0 — Rest"

    st.markdown(
        (
            f"<div style='background:{bg_color}; color:{text_color}; padding:8px 16px; "
            "border-radius:6px; text-align:center; font-size:16px; font-weight:600; "
            "margin:0.25rem 0 1rem 0;'>"
            f"{label} (t = {time_seconds:.0f}s)"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _make_topomap_event_timeline(events: list[dict], duration: float, current_time: float) -> go.Figure:
    """Compact event timeline with the selected topomap time marked."""
    fig = go.Figure()

    for evt in events:
        label = evt["label"]
        if label not in ("T1", "T2"):
            continue

        onset = float(evt["onset"])
        end = min(onset + 4.0, duration)
        color = T1_BLUE if label == "T1" else T2_ORANGE
        fill = T1_BLUE_LIGHT if label == "T1" else T2_ORANGE_LIGHT
        label_text = "T1" if label == "T1" else "T2"

        fig.add_vrect(x0=onset, x1=end, fillcolor=fill, line_width=0, layer="below")
        fig.add_annotation(
            x=(onset + end) / 2,
            y=1.05,
            text=label_text,
            showarrow=False,
            font=dict(size=10, color=color),
        )

    fig.add_vline(
        x=current_time,
        line=dict(color="#0F172A", width=2),
        annotation_text=f"{current_time:.0f}s",
        annotation_position="bottom",
        annotation_font_size=10,
        annotation_font_color="#0F172A",
    )

    fig.update_layout(
        height=92,
        margin=dict(l=10, r=10, t=22, b=30),
        xaxis=dict(title="Recording time (s)", range=[0, duration], tickfont=dict(size=9)),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="white",
    )
    return fig


def _render_compact_power_box(label: str, value: float):
    """Small readout that fits beside the compact topomap row."""
    st.markdown(
        (
            "<div style='border:1px solid #E2E8F0; border-radius:8px; "
            "background:#F8FAFC; padding:0.55rem 0.7rem; margin:0.35rem 0;'>"
            f"<div style='font-size:0.78rem; color:#64748B; margin-bottom:0.1rem;'>{label}</div>"
            f"<div style='font-size:1.35rem; font-weight:600; color:#0F172A; line-height:1.2;'>{value:.1f} dB</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_asymmetry_indicator(
    mu_db_valid,
    beta_db_valid,
    sensor_names: list[str],
    t_idx: int,
    current_event: str,
):
    """Show C3-C4 power asymmetry as a diverging bar."""
    if "C3" not in sensor_names or "C4" not in sensor_names:
        return

    c3_idx = sensor_names.index("C3")
    c4_idx = sensor_names.index("C4")

    c3_mu = float(mu_db_valid[c3_idx, t_idx])
    c4_mu = float(mu_db_valid[c4_idx, t_idx])
    c3_beta = float(beta_db_valid[c3_idx, t_idx])
    c4_beta = float(beta_db_valid[c4_idx, t_idx])

    mu_asym = c4_mu - c3_mu
    beta_asym = c4_beta - c3_beta

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["β (13–30 Hz)", "μ (8–13 Hz)"],
            x=[beta_asym, mu_asym],
            orientation="h",
            marker_color=[
                T1_BLUE if beta_asym < 0 else T2_ORANGE,
                T1_BLUE if mu_asym < 0 else T2_ORANGE,
            ],
            text=[f"{beta_asym:+.1f} dB", f"{mu_asym:+.1f} dB"],
            textposition="outside",
            showlegend=False,
        )
    )

    fig.add_vline(x=0, line=dict(color="#1E293B", width=1.5))

    max_range = max(abs(mu_asym), abs(beta_asym), 2.0) * 1.5

    fig.update_layout(
        height=180,
        margin=dict(l=5, r=20, t=20, b=30),
        xaxis=dict(
            range=[-max_range, max_range],
            title="",
            tickfont=dict(size=9),
        ),
        yaxis=dict(tickfont=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.add_annotation(
        x=-max_range * 0.85,
        y=1.15,
        yref="paper",
        text="← C4 lower<br><sub>(left imagery)</sub>",
        showarrow=False,
        font=dict(size=9, color=T1_BLUE),
    )
    fig.add_annotation(
        x=max_range * 0.85,
        y=1.15,
        yref="paper",
        text="C3 lower →<br><sub>(right imagery)</sub>",
        showarrow=False,
        font=dict(size=9, color=T2_ORANGE),
    )

    st.plotly_chart(fig, use_container_width=True)

    if current_event == "T1":
        if mu_asym < 0:
            st.caption("Consistent with T1: C4 mu power is lower at this point.")
        else:
            st.caption("Expected C4 desynchronization is not visible at this point.")
    elif current_event == "T2":
        if mu_asym > 0:
            st.caption("Consistent with T2: C3 mu power is lower at this point.")
        else:
            st.caption("Expected C3 desynchronization is not visible at this point.")
    else:
        st.caption("Rest period — no consistent lateralization expected")


def _render_epoch_prediction_for_time(predictions: dict, t_seconds: float):
    """Show model prediction for the epoch covering *t_seconds*, if any."""
    epoch_info = predictions.get("epoch_info") or []
    for ep in epoch_info:
        if ep["start_time"] <= float(t_seconds) < ep["end_time"]:
            pred_name = "Left fist" if ep["pred_label"] == 0 else "Right fist"
            conf = ep["confidence"]
            st.markdown(f"**Model says:** {pred_name} ({conf * 100:.0f}%)")
            if ep["correct"]:
                st.caption("Prediction matches ground truth.")
            else:
                gt = "Left" if ep["true_label"] == 0 else "Right"
                st.caption(f"Ground truth: {gt}")
            break


# ── Main render function ─────────────────────────────────────────────────


def render_explore_tab(
    edf_data: dict | None,
    selected_channel: str,
    file_hash: str,
    selected_time_range: str = "Full recording",
):
    """Render the complete explore tab."""

    if edf_data is None:
        st.markdown(
            "<div style='text-align: center; padding: 80px 20px;'>"
            "<h2>Upload an EDF file to begin</h2>"
            "<p style='color: #6B7280; font-size: 1.1em;'>"
            "Upload a PhysioNet EEG Motor Movement/Imagery Dataset file (Runs 4, 8, or 12) "
            "using the sidebar to explore EEG signals, visualise brain activity, and run "
            "motor imagery predictions."
            "</p>"
            "<p><a href='https://physionet.org/content/eegmmidb/1.0.0/' target='_blank'>"
            "PhysioNet Dataset →</a></p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    data = edf_data["data"]
    times = edf_data["times"]
    ch_names = edf_data["ch_names"]
    sfreq = edf_data["sfreq"]
    duration = edf_data["duration"]
    events = get_event_times(edf_data)
    ann_types = sorted({e["label"] for e in events})

    t_start, t_end = _parse_time_range(selected_time_range, float(duration))
    mask = (times >= t_start) & (times <= t_end)
    times_view = times[mask]
    events_in_range = _events_in_range(events, t_start, t_end)

    st.subheader("File Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Sampling frequency", f"{sfreq:.0f} Hz")
    with c2:
        st.metric("Channels", str(len(ch_names)))
    with c3:
        st.metric("Duration", f"{duration:.1f} s")
    with c4:
        st.metric("Annotations", ", ".join(ann_types))

    ch_idx = ch_names.index(selected_channel) if selected_channel in ch_names else 0
    ch_name = ch_names[ch_idx]

    motor_channels = ("C3", "Cz", "C4")
    is_motor_ch = selected_channel in motor_channels

    st.subheader("Signal Viewer")
    st.markdown(
        f"<p class='section-note'>Showing {selected_time_range}. "
        "Use Plotly controls to zoom, pan, and inspect exact signal values.</p>",
        unsafe_allow_html=True,
    )

    raw_signal = data[ch_idx]
    raw_view = raw_signal[mask]

    if is_motor_ch:
        fig_multi = _make_multichannel_plot_highlighted(
            times_view,
            data[:, mask],
            ch_names,
            events_in_range,
            motor_channels,
            highlight=selected_channel,
        )
        if fig_multi:
            st.plotly_chart(fig_multi, use_container_width=True)
        else:
            st.plotly_chart(
                _make_signal_plot(times_view, raw_view, events_in_range, ch_name),
                use_container_width=True,
            )
    else:
        st.plotly_chart(
            _make_signal_plot(times_view, raw_view, events_in_range, ch_name),
            use_container_width=True,
        )
        st.markdown("**Motor Cortex Reference**")
        fig_multi = _make_multichannel_plot(
            times_view, data[:, mask], ch_names, events_in_range
        )
        if fig_multi:
            st.plotly_chart(fig_multi, use_container_width=True)
        else:
            st.info("Motor cortex channels (C3, Cz, C4) not found in this file.")

    col_left, col_right = st.columns(2)
    with col_left:
        filtered_data = get_filtered_signal(data, sfreq)
        st.plotly_chart(
            _make_raw_vs_filtered_plot(
                times_view, raw_view, filtered_data[ch_idx][mask], ch_name
            ),
            use_container_width=True,
        )
    with col_right:
        freqs, psd = compute_psd(raw_view, sfreq)
        st.plotly_chart(_make_psd_plot(freqs, psd, ch_name), use_container_width=True)

    st.subheader("Topographic Explorer")
    st.markdown(
        "<p class='section-note'>The time slider below controls only the scalp map. "
        "Signal plots above stay on the selected review window.</p>",
        unsafe_allow_html=True,
    )

    sensor_result = _compute_sensor_positions(tuple(ch_names))
    if sensor_result[0] is None:
        st.warning("Could not determine electrode positions for this file.")
        return

    sensor_names, pos_2d, orig_indices, ch_to_pos = sensor_result
    mu_db, beta_db, t_centers = _precompute_band_power(data, sfreq, file_hash)

    mu_db_valid = mu_db[orig_indices]
    beta_db_valid = beta_db[orig_indices]

    shared_power = np.concatenate([mu_db_valid.ravel(), beta_db_valid.ravel()])
    shared_vlim = (
        float(np.percentile(shared_power, 5)),
        float(np.percentile(shared_power, 95)),
    )

    max_t = int(t_centers[-1])
    st.markdown("**Time**")
    t_slider = st.slider(
        "Time (s)",
        min_value=0,
        max_value=max_t,
        value=0,
        step=1,
        key="topo_time",
        label_visibility="collapsed",
    )

    t_idx = int(np.argmin(np.abs(t_centers - t_slider)))
    current_time = float(t_slider)
    current_event = _get_event_at_time(events, current_time)

    st.plotly_chart(
        _make_topomap_event_timeline(events, duration, current_time),
        use_container_width=True,
    )
    _render_event_status(current_event, current_time)

    col_mu, col_beta, col_info = st.columns([1, 1, 0.75], gap="medium")

    with col_mu:
        fig_mu = _render_topomap(
            mu_db_valid[:, t_idx],
            pos_2d,
            selected_channel,
            ch_to_pos,
            sensor_names,
            f"μ (8–13 Hz) — {current_time:.0f}s",
            shared_vlim,
            figsize=(3.4, 3.2),
        )
        st.pyplot(fig_mu)
        plt.close(fig_mu)

    with col_beta:
        fig_beta = _render_topomap(
            beta_db_valid[:, t_idx],
            pos_2d,
            selected_channel,
            ch_to_pos,
            sensor_names,
            f"β (13–30 Hz) — {current_time:.0f}s",
            shared_vlim,
            figsize=(3.4, 3.2),
        )
        st.pyplot(fig_beta)
        plt.close(fig_beta)

    with col_info:
        st.markdown("**Current selection**")
        st.markdown(f"**Time:** {current_time:.0f} s")

        st.markdown(f"**Selected channel:** `{selected_channel}`")

        if selected_channel in ch_to_pos:
            ch_sel_idx = sensor_names.index(selected_channel)
            ch_mu = mu_db_valid[ch_sel_idx, t_idx]
            ch_beta = beta_db_valid[ch_sel_idx, t_idx]
            _render_compact_power_box("μ power", ch_mu)
            _render_compact_power_box("β power", ch_beta)

        predictions = st.session_state.get("predictions")
        if predictions:
            _render_epoch_prediction_for_time(predictions, current_time)

    if "C3" in sensor_names and "C4" in sensor_names:
        st.markdown("**Motor Cortex Asymmetry (C3 vs C4)**")
        st.caption(
            "Positive values mean C4 power is higher than C3. During left imagery (T1), "
            "C4 mu power often drops; during right imagery (T2), C3 mu power often drops."
        )
        _render_asymmetry_indicator(
            mu_db_valid, beta_db_valid, sensor_names, t_idx, current_event
        )
