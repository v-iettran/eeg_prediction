"""Evaluation utilities — metrics, confusion-matrix plotting, etc.

Kept tiny on purpose: every model notebook should call the same functions
so the resulting JSONs follow the schema in PIPELINE_PLAN.md sec. 8.2.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = ("T1 (left)", "T2 (right)")


def compute_metrics(y_true, y_pred) -> dict:
    """Return accuracy, macro-F1, per-class F1, and confusion matrix as a dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": [float(v) for v in f1_score(y_true, y_pred, average=None, labels=[0, 1])],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def plot_confusion_matrix(cm, title: str, save_path: str, class_names: Sequence[str] = CLASS_NAMES):
    """Save a 2x2 confusion-matrix heatmap with raw counts + row-normalised %"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    annot = np.array(
        [[f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    )
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history, save_path: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history.train_loss, label="train")
    axes[0].plot(history.val_loss, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(history.val_macro_f1, label="val macroF1", color="tab:green")
    axes[1].plot(history.val_accuracy, label="val accuracy", color="tab:orange")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, save_path: str, top_k: int = 12):
    import matplotlib.pyplot as plt

    importances = np.asarray(importances)
    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(order) + 1.5))
    ax.barh(range(len(order)), importances[order][::-1], color="tab:blue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest — top features")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
