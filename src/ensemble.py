"""Stacking ensemble utilities for V2 pipeline.

Provides helpers to collect base-model probabilities and fit/apply
a meta-learner (logistic regression on stacked probabilities).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.features_v2 import extract_features_v2
from src.models import EEGNet, EEGNetConfig, apply_zscore


@dataclass
class BaseModelSpec:
    name: str
    input_type: str          # "features_v2" | "raw_64ch"
    model_path: str
    val_macro_f1: float = 0.0
    test_macro_f1: float = 0.0


def predict_proba_classical(model_pipeline, X_3ch: np.ndarray, sfreq: float) -> np.ndarray:
    """Get class probabilities from a classical sklearn pipeline using v2 features."""
    feats = extract_features_v2(X_3ch, sfreq)
    return model_pipeline.predict_proba(feats)


def predict_proba_csp(model_pipeline, X_64ch: np.ndarray) -> np.ndarray:
    """Get class probabilities from CSP+LDA pipeline."""
    X = X_64ch.astype(np.float64)
    return model_pipeline.predict_proba(X)


def predict_proba_eegnet(
    model: EEGNet,
    X_64ch: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Get class probabilities from EEGNet via softmax."""
    X_z = apply_zscore(X_64ch, mean, std)
    model.eval()
    probas = []
    Xt = torch.from_numpy(X_z.astype(np.float32)).to(device)
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            logits = model(Xt[i : i + batch_size])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probas.append(probs)
    return np.concatenate(probas, axis=0)


def load_eegnet_from_checkpoint(path: str | Path, device: torch.device) -> tuple[EEGNet, np.ndarray, np.ndarray]:
    """Load an EEGNet model + normalization stats from a .pt checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = EEGNetConfig(**ckpt["config"])
    model = EEGNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["channel_mean"], ckpt["channel_std"]


def stack_probabilities(proba_list: list[np.ndarray]) -> np.ndarray:
    """Horizontally stack base-model probability arrays."""
    return np.hstack(proba_list)


def fit_meta_learner(stacked_proba: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression meta-learner on stacked base-model probabilities."""
    meta = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    meta.fit(stacked_proba, y)
    return meta


def save_ensemble_config(
    base_specs: list[BaseModelSpec],
    meta_path: str,
    out_path: str | Path,
) -> None:
    """Save ensemble configuration JSON for downstream inference."""
    config = {
        "meta_learner_path": meta_path,
        "base_models": [
            {
                "name": s.name,
                "input_type": s.input_type,
                "model_path": s.model_path,
            }
            for s in base_specs
        ],
    }
    Path(out_path).write_text(json.dumps(config, indent=2))
