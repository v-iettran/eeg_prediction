"""Phase E — Stacking ensemble over V2 base models.

Combines the best classical model(s) from Phase C, V2 CSP+LDA, and
V2 EEGNet via a logistic regression meta-learner trained on val-set
predictions (the simpler approach from the plan).

Produces:
    models/v2_ensemble_meta.joblib
    models/v2_ensemble_config.json
    reports/v2_ensemble_results.json
    reports/figures/v2_ensemble_comparison.png
    reports/figures/v2_ensemble_confusion.png
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ensemble import (  # noqa: E402
    BaseModelSpec,
    fit_meta_learner,
    load_eegnet_from_checkpoint,
    predict_proba_classical,
    predict_proba_csp,
    predict_proba_eegnet,
    save_ensemble_config,
    stack_probabilities,
)
from src.evaluation import compute_metrics, plot_confusion_matrix  # noqa: E402
from src.splits import load_split, split_indices_by_subject  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
SEED = 42
WIN_LABEL = "w0.0-4.0"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_v2_window():
    arr = np.load(PROCESSED / f"v2_epochs_{WIN_LABEL}.npz", allow_pickle=False)
    return {
        "X_3ch": arr["X_3ch"],
        "X_64ch": arr["X_64ch"],
        "y": arr["y"],
        "subject_ids": arr["subject_ids"],
        "sfreq": float(arr["sfreq"]),
    }


def _select_classical_models() -> list[tuple[str, str]]:
    """Pick best classical model(s) from Phase C results.

    Returns list of (name, model_path) tuples. If multiple models are
    within 1% val F1, include the top 2.
    """
    results_path = REPORTS_DIR / "v2_classical_results.json"
    with open(results_path) as f:
        classical = json.load(f)

    models = sorted(classical["models"], key=lambda m: m["val_macro_f1"], reverse=True)
    selected = [(models[0]["name"], models[0]["model_artifact"])]

    if len(models) > 1:
        gap = models[0]["val_macro_f1"] - models[1]["val_macro_f1"]
        if gap < 0.01:
            selected.append((models[1]["name"], models[1]["model_artifact"]))

    return selected


def main() -> None:
    np.random.seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = _device()
    split = load_split(PROCESSED / "splits.json")
    bundle = _load_v2_window()
    X_3ch = bundle["X_3ch"]
    X_64ch = bundle["X_64ch"]
    y = bundle["y"]
    subj = bundle["subject_ids"]
    sfreq = bundle["sfreq"]

    train_idx, val_idx, test_idx = split_indices_by_subject(subj, split)
    X_3ch_val = X_3ch[val_idx];   X_3ch_test = X_3ch[test_idx]
    X_64ch_val = X_64ch[val_idx]; X_64ch_test = X_64ch[test_idx]
    y_val = y[val_idx];           y_test = y[test_idx]
    print(f"Split: val={len(val_idx)} test={len(test_idx)}")

    # --- Load base models ---
    base_specs: list[BaseModelSpec] = []
    val_probas: list[np.ndarray] = []
    test_probas: list[np.ndarray] = []

    # Classical model(s)
    classical_picks = _select_classical_models()
    print(f"\nClassical models selected: {[n for n, _ in classical_picks]}")

    for name, rel_path in classical_picks:
        pipe = joblib.load(ROOT / rel_path)
        p_val = predict_proba_classical(pipe, X_3ch_val, sfreq)
        p_test = predict_proba_classical(pipe, X_3ch_test, sfreq)
        val_probas.append(p_val)
        test_probas.append(p_test)
        base_specs.append(BaseModelSpec(name=name, input_type="features_v2", model_path=rel_path))
        print(f"  {name}: val probas {p_val.shape}")

    # CSP+LDA
    csp_pipe = joblib.load(MODELS_DIR / "v2_csp_lda.joblib")
    p_val_csp = predict_proba_csp(csp_pipe, X_64ch_val)
    p_test_csp = predict_proba_csp(csp_pipe, X_64ch_test)
    val_probas.append(p_val_csp)
    test_probas.append(p_test_csp)
    base_specs.append(BaseModelSpec(
        name="v2_csp_lda", input_type="raw_64ch",
        model_path="models/v2_csp_lda.joblib",
    ))
    print(f"  v2_csp_lda: val probas {p_val_csp.shape}")

    # EEGNet
    eegnet, mean, std = load_eegnet_from_checkpoint(MODELS_DIR / "v2_eegnet.pt", device)
    p_val_eeg = predict_proba_eegnet(eegnet, X_64ch_val, mean, std, device)
    p_test_eeg = predict_proba_eegnet(eegnet, X_64ch_test, mean, std, device)
    val_probas.append(p_val_eeg)
    test_probas.append(p_test_eeg)
    base_specs.append(BaseModelSpec(
        name="v2_eegnet", input_type="raw_64ch",
        model_path="models/v2_eegnet.pt",
    ))
    print(f"  v2_eegnet: val probas {p_val_eeg.shape}")

    # --- Meta-learner ---
    stacked_val = stack_probabilities(val_probas)
    stacked_test = stack_probabilities(test_probas)
    print(f"\nStacked val shape: {stacked_val.shape}  test shape: {stacked_test.shape}")

    meta = fit_meta_learner(stacked_val, y_val)
    meta_path = MODELS_DIR / "v2_ensemble_meta.joblib"
    joblib.dump(meta, meta_path)
    print(f"Meta-learner saved → {meta_path}")

    # --- Ensemble test evaluation ---
    ensemble_test_pred = meta.predict(stacked_test)
    ensemble_test_metrics = compute_metrics(y_test, ensemble_test_pred)
    print(f"\nEnsemble TEST: accuracy={ensemble_test_metrics['accuracy']:.4f} macroF1={ensemble_test_metrics['macro_f1']:.4f}")

    plot_confusion_matrix(
        ensemble_test_metrics["confusion_matrix"],
        "V2 Ensemble — test confusion",
        str(FIG_DIR / "v2_ensemble_confusion.png"),
    )

    # --- Individual base model test metrics ---
    for i, spec in enumerate(base_specs):
        pred_i = test_probas[i].argmax(axis=1)
        m = compute_metrics(y_test, pred_i)
        spec.val_macro_f1 = float(compute_metrics(y_val, val_probas[i].argmax(axis=1))["macro_f1"])
        spec.test_macro_f1 = float(m["macro_f1"])
        print(f"  {spec.name} test macroF1: {spec.test_macro_f1:.4f}")

    # --- Comparison bar chart ---
    v1_baselines = {}
    for fname in ["rf_results.json", "csp_lda_results.json", "eegnet_results.json"]:
        path = REPORTS_DIR / fname
        if path.exists():
            with open(path) as f:
                r = json.load(f)
                v1_baselines[f"v1_{r['model_name']}"] = r["test_metrics"]["macro_f1"]

    all_models = dict(v1_baselines)
    for spec in base_specs:
        all_models[spec.name] = spec.test_macro_f1
    all_models["v2_ensemble"] = ensemble_test_metrics["macro_f1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(all_models.keys())
    vals = list(all_models.values())
    colors = []
    for n in names:
        if n.startswith("v1_"):
            colors.append("tab:gray")
        elif n == "v2_ensemble":
            colors.append("tab:green")
        else:
            colors.append("tab:blue")
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Model Comparison — Test Set Macro-F1")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_ensemble_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Comparison chart → {FIG_DIR / 'v2_ensemble_comparison.png'}")

    # --- Save ensemble config ---
    save_ensemble_config(
        base_specs,
        meta_path=str(meta_path.relative_to(ROOT)),
        out_path=MODELS_DIR / "v2_ensemble_config.json",
    )

    # --- Result JSON ---
    coefs = meta.coef_.tolist()
    result = {
        "model_name": "v2_ensemble",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_version": "v2",
        "rejection_threshold_uv": 300,
        "n_train_epochs": int(len(train_idx)),
        "n_val_epochs": int(len(val_idx)),
        "n_test_epochs": int(len(test_idx)),
        "base_models": [
            {
                "name": s.name,
                "input_type": s.input_type,
                "val_macro_f1": s.val_macro_f1,
                "test_macro_f1": s.test_macro_f1,
            }
            for s in base_specs
        ],
        "ensemble_test_metrics": ensemble_test_metrics,
        "meta_learner_config": {
            "type": "LogisticRegression",
            "trained_on": "val_predictions",
            "input_dim": stacked_val.shape[1],
            "coefficients": coefs,
        },
        "v1_baselines": v1_baselines,
    }
    out_path = REPORTS_DIR / "v2_ensemble_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")
    print("Phase E complete.")


if __name__ == "__main__":
    main()
