"""Model loading, inference routing, and prediction for the Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from src.preprocessing import euclidean_align


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_REGISTRY = {
    "Random Forest": {
        "path": "models/v2_rf.joblib",
        "type": "sklearn",
        "input": "features_v2",
        "bandpass": (8, 30),
        "family": "Classical",
        "description": "Random forest on 31-dim enriched band power features from C3, Cz, C4.",
    },
    "XGBoost": {
        "path": "models/v2_xgb.joblib",
        "type": "sklearn",
        "input": "features_v2",
        "bandpass": (8, 30),
        "family": "Classical",
        "description": "Gradient boosted trees on 31-dim enriched band power features from C3, Cz, C4.",
    },
    "CSP+LDA": {
        "path": "models/v2_csp_lda.joblib",
        "type": "sklearn",
        "input": "raw_64ch",
        "bandpass": (8, 30),
        "family": "Spatial",
        "description": "Common Spatial Patterns (8 components) with Linear Discriminant Analysis on all 64 channels.",
    },
    "EEGNet": {
        "path": "models/v2_eegnet.pt",
        "type": "pytorch",
        "input": "raw_64ch",
        "bandpass": (8, 30),
        "arch": "EEGNet",
        "family": "Deep Learning",
        "description": "Compact CNN with temporal and depthwise spatial convolutions (Lawhern 2018).",
    },
    "EEG-TCNet": {
        "path": "models/v2_tcnet.pt",
        "type": "pytorch",
        "input": "raw_64ch",
        "bandpass": (8, 30),
        "arch": "EEGTCNet",
        "family": "Deep Learning",
        "description": "EEGNet frontend + Temporal Convolutional Network with dilated causal convolutions.",
    },
    "FBCNet": {
        "path": "models/v2_fbcnet.pt",
        "type": "pytorch",
        "input": "filter_bank_64ch",
        "bandpass": (8, 30),
        "arch": "FBCNet",
        "family": "Deep Learning",
        "description": "Filter Bank CSP Network with multi-window log-variance temporal features.",
    },
}


def get_available_models() -> list[str]:
    """Return names of models whose artifact files exist on disk."""
    available = []
    for name, cfg in MODEL_REGISTRY.items():
        if (ROOT / cfg["path"]).exists():
            available.append(name)
    return available


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str):
    """Load a model from disk. Cached for the app lifetime."""
    import joblib
    import torch

    cfg = MODEL_REGISTRY[model_name]
    path = ROOT / cfg["path"]

    if cfg["type"] == "sklearn":
        return joblib.load(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if cfg["arch"] == "EEGNet":
        from src.models import EEGNet, EEGNetConfig
        model_cfg = EEGNetConfig(**ckpt["config"])
        model = EEGNet(model_cfg)
    elif cfg["arch"] == "EEGTCNet":
        from src.tcnet import EEGTCNet, TCNetConfig
        config_dict = dict(ckpt["config"])
        if "tcn_dilations" in config_dict and isinstance(config_dict["tcn_dilations"], list):
            config_dict["tcn_dilations"] = tuple(config_dict["tcn_dilations"])
        model_cfg = TCNetConfig(**config_dict)
        model = EEGTCNet(model_cfg)
    elif cfg["arch"] == "FBCNet":
        from src.fbcnet import FBCNet, FBCNetConfig
        model_cfg = FBCNetConfig(**ckpt["config"])
        model = FBCNet(model_cfg)
    else:
        raise ValueError(f"Unknown architecture: {cfg['arch']}")

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return {"model": model, "checkpoint": ckpt}


def _preprocess_for_inference(edf_data: dict, bandpass: tuple) -> dict:
    """Preprocess an uploaded EDF for inference.

    Returns dict with:
        epochs_64ch: (n_epochs, 64, n_samples) array
        epochs_3ch: (n_epochs, 3, n_samples) array  (C3, Cz, C4)
        y_true: (n_epochs,) ground truth labels (0=T1/left, 1=T2/right)
        epoch_info: list of dicts with start_time, end_time, true_label
        sfreq: sampling frequency
    """
    import mne
    import tempfile

    sfreq = edf_data["sfreq"]
    data = edf_data["data"].copy()
    ch_names = edf_data["ch_names"]
    annotations = edf_data["annotations"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    mne_annotations = mne.Annotations(
        onset=[a[0] for a in annotations],
        duration=[a[1] for a in annotations],
        description=[a[2] for a in annotations],
    )
    raw.set_annotations(mne_annotations, verbose="ERROR")

    raw.filter(
        l_freq=bandpass[0], h_freq=bandpass[1],
        method="iir", verbose="ERROR",
    )
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    events, event_id_all = mne.events_from_annotations(raw, verbose="ERROR")

    target_event_id = {}
    for key, code in event_id_all.items():
        if key in ("T1", "T2"):
            target_event_id[key] = code

    if not target_event_id:
        return None

    epochs = mne.Epochs(
        raw, events, event_id=target_event_id,
        tmin=0.0, tmax=4.0, baseline=None, preload=True,
        reject=None, verbose="ERROR",
    )

    epochs_data = epochs.get_data()
    # EA: whiten this subject's epochs (single subject = one call)
    epochs_data = euclidean_align(epochs_data)
    event_codes = epochs.events[:, 2]

    t1_code = target_event_id.get("T1")
    t2_code = target_event_id.get("T2")
    y_true = np.where(event_codes == t1_code, 0, 1).astype(np.int64)

    motor_idx = []
    for ch in ("C3", "Cz", "C4"):
        if ch in ch_names:
            motor_idx.append(ch_names.index(ch))
    epochs_3ch = epochs_data[:, motor_idx, :] if len(motor_idx) == 3 else None

    epoch_times = epochs.events[:, 0] / sfreq
    epoch_info = []
    for i in range(len(epochs_data)):
        start_t = float(epoch_times[i])
        label_name = "T1" if y_true[i] == 0 else "T2"
        epoch_info.append({
            "epoch": i + 1,
            "start_time": start_t,
            "end_time": start_t + 4.0,
            "true_label": int(y_true[i]),
            "true_label_name": label_name,
        })

    return {
        "epochs_64ch": epochs_data,
        "epochs_3ch": epochs_3ch,
        "y_true": y_true,
        "epoch_info": epoch_info,
        "sfreq": sfreq,
    }


def run_inference(edf_data: dict, model_name: str) -> dict | None:
    """Run full inference pipeline on an uploaded EDF file.

    Returns dict with:
        epoch_info: list of dicts with timing + ground truth + prediction
        predictions: (n_epochs,) predicted labels
        probabilities: (n_epochs, 2) class probabilities
        y_true: (n_epochs,) ground truth
        accuracy: float
        macro_f1: float
        confusion_matrix: 2x2 list
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    cfg = MODEL_REGISTRY[model_name]
    prep = _preprocess_for_inference(edf_data, cfg["bandpass"])
    if prep is None:
        return None

    loaded = load_model(model_name)

    if cfg["type"] == "sklearn":
        if cfg["input"] == "features_v2":
            from src.features_v2 import extract_features_v2
            X = extract_features_v2(prep["epochs_3ch"], prep["sfreq"])
        else:
            X = prep["epochs_64ch"].astype(np.float64)

        if isinstance(loaded, dict):
            model = loaded["model"]
        else:
            model = loaded

        preds = model.predict(X)
        try:
            proba = model.predict_proba(X)
        except AttributeError:
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds] = 1.0

    elif cfg["type"] == "pytorch":
        model_obj = loaded["model"]
        ckpt = loaded["checkpoint"]

        if cfg["input"] == "filter_bank_64ch":
            from src.fbcnet import apply_filter_bank, FilterBankConfig

            fb_info = ckpt.get("filter_bank", {})
            fb_cfg = FilterBankConfig(
                bands=fb_info.get("bands", FilterBankConfig().bands),
                order=fb_info.get("order", 5),
                sfreq=fb_info.get("sfreq", prep["sfreq"]),
            )
            X_fb = apply_filter_bank(prep["epochs_64ch"], fb_cfg)

            fb_mean = ckpt.get("fb_mean")
            fb_std = ckpt.get("fb_std")
            if fb_mean is not None and fb_std is not None:
                X_fb = ((X_fb - fb_mean) / fb_std).astype(np.float32)

            X = X_fb
        else:
            X = prep["epochs_64ch"]
            ch_mean = ckpt.get("channel_mean")
            ch_std = ckpt.get("channel_std")
            if ch_mean is not None and ch_std is not None:
                from src.models import apply_zscore
                X = apply_zscore(X, ch_mean, ch_std)

        X_t = torch.from_numpy(X.astype(np.float32))
        model_obj.eval()
        with torch.no_grad():
            logits = model_obj(X_t)
            proba = torch.softmax(logits, dim=1).numpy()

        preds = np.argmax(proba, axis=1)

    y_true = prep["y_true"]
    epoch_info = prep["epoch_info"]
    for i, info in enumerate(epoch_info):
        info["pred_label"] = int(preds[i])
        info["pred_label_name"] = "T1" if preds[i] == 0 else "T2"
        info["confidence"] = float(proba[i, preds[i]])
        info["correct"] = bool(preds[i] == y_true[i])

    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="macro"))
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()

    return {
        "epoch_info": epoch_info,
        "predictions": preds,
        "probabilities": proba,
        "y_true": y_true,
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
    }
