"""Phase E-revised — Deep-only stacking ensemble (EEGNet + TCNet + FBCNet).

All three models now use the same 8-30 Hz EA-aligned data, so no cross-bandpass
alignment is needed.

Produces:
    models/v2_ensemble_v2_meta.joblib
    models/v2_ensemble_v2_config.json
    reports/v2_ensemble_v2_results.json
    reports/figures/v2_final_comparison.png
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
    predict_proba_eegnet,
    stack_probabilities,
)
from src.evaluation import compute_metrics, plot_confusion_matrix  # noqa: E402
from src.models import apply_zscore, channel_zscore_stats  # noqa: E402
from src.tcnet import EEGTCNet, TCNetConfig  # noqa: E402
from src.fbcnet import FBCNet, FBCNetConfig, FilterBankConfig, apply_filter_bank  # noqa: E402
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


def _predict_proba_torch(model, X, device, batch_size=256):
    model.eval()
    out = []
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            logits = model(Xt[i:i + batch_size])
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    np.random.seed(SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = _device()
    split = load_split(PROCESSED / "splits.json")

    # --- Load unified 8-30 Hz EA-aligned data ---
    arr = np.load(PROCESSED / f"v2_epochs_{WIN_LABEL}.npz", allow_pickle=False)
    X_64ch = arr["X_64ch"]
    y = arr["y"]
    subj = arr["subject_ids"]
    sfreq = float(arr["sfreq"])

    train_idx, val_idx, test_idx = split_indices_by_subject(subj, split)
    X_val = X_64ch[val_idx]
    X_test = X_64ch[test_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    print(f"EA-aligned 8-30 Hz data: val={len(val_idx)} test={len(test_idx)}")

    # --- Base model 1: EEGNet ---
    print("\nLoading EEGNet...")
    eegnet, eeg_mean, eeg_std = load_eegnet_from_checkpoint(MODELS_DIR / "v2_eegnet.pt", device)
    p_val_eeg = predict_proba_eegnet(eegnet, X_val, eeg_mean, eeg_std, device)
    p_test_eeg = predict_proba_eegnet(eegnet, X_test, eeg_mean, eeg_std, device)
    print(f"  EEGNet: val probas {p_val_eeg.shape}")

    # --- Base model 2: EEG-TCNet ---
    print("Loading EEG-TCNet...")
    tcnet_ckpt = torch.load(MODELS_DIR / "v2_tcnet.pt", map_location=device, weights_only=False)
    tcnet_cfg_dict = tcnet_ckpt["config"]
    if "tcn_dilations" in tcnet_cfg_dict:
        tcnet_cfg_dict["tcn_dilations"] = tuple(tcnet_cfg_dict["tcn_dilations"])
    tcnet_cfg = TCNetConfig(**tcnet_cfg_dict)
    tcnet = EEGTCNet(tcnet_cfg)
    tcnet.load_state_dict(tcnet_ckpt["state_dict"])
    tcnet.to(device)
    tcnet.eval()

    tc_mean, tc_std = tcnet_ckpt["channel_mean"], tcnet_ckpt["channel_std"]
    X_val_tc = apply_zscore(X_val, tc_mean, tc_std)
    X_test_tc = apply_zscore(X_test, tc_mean, tc_std)
    p_val_tc = _predict_proba_torch(tcnet, X_val_tc, device)
    p_test_tc = _predict_proba_torch(tcnet, X_test_tc, device)
    print(f"  TCNet: val probas {p_val_tc.shape}")

    # --- Base model 3: FBCNet (same 8-30 Hz data, overlapping filter bank) ---
    print("Loading FBCNet...")
    fbc_ckpt = torch.load(MODELS_DIR / "v2_fbcnet.pt", map_location=device, weights_only=False)
    fbc_cfg = FBCNetConfig(**fbc_ckpt["config"])
    fbcnet = FBCNet(fbc_cfg)
    fbcnet.load_state_dict(fbc_ckpt["state_dict"])
    fbcnet.to(device)
    fbcnet.eval()

    fb_config = FilterBankConfig(
        bands=fbc_ckpt["filter_bank"]["bands"],
        order=fbc_ckpt["filter_bank"]["order"],
        sfreq=fbc_ckpt["filter_bank"]["sfreq"],
    )
    fb_mean = fbc_ckpt["fb_mean"]
    fb_std = fbc_ckpt["fb_std"]

    print("  Applying filter bank to val/test...")
    X_fb_val = apply_filter_bank(X_val, fb_config)
    X_fb_test = apply_filter_bank(X_test, fb_config)
    X_fb_val_z = ((X_fb_val - fb_mean) / fb_std).astype(np.float32)
    X_fb_test_z = ((X_fb_test - fb_mean) / fb_std).astype(np.float32)

    p_val_fbc = _predict_proba_torch(fbcnet, X_fb_val_z, device)
    p_test_fbc = _predict_proba_torch(fbcnet, X_fb_test_z, device)
    print(f"  FBCNet: val probas {p_val_fbc.shape}")

    # --- Stack and fit meta-learner ---
    stacked_val = stack_probabilities([p_val_eeg, p_val_tc, p_val_fbc])
    stacked_test = stack_probabilities([p_test_eeg, p_test_tc, p_test_fbc])
    print(f"\nStacked: val={stacked_val.shape} test={stacked_test.shape}")

    meta = fit_meta_learner(stacked_val, y_val)
    meta_path = MODELS_DIR / "v2_ensemble_v2_meta.joblib"
    joblib.dump(meta, meta_path)
    print(f"Meta-learner saved → {meta_path}")

    # --- Ensemble test evaluation ---
    ensemble_pred = meta.predict(stacked_test)
    ensemble_metrics = compute_metrics(y_test, ensemble_pred)
    print(f"\nDeep Ensemble TEST: accuracy={ensemble_metrics['accuracy']:.4f} macroF1={ensemble_metrics['macro_f1']:.4f}")

    plot_confusion_matrix(
        ensemble_metrics["confusion_matrix"],
        "V2 Deep Ensemble (fixed) — test confusion",
        str(FIG_DIR / "v2_deep_ensemble_confusion.png"),
    )

    # --- Individual base model test metrics ---
    base_specs = []
    for name, p_test_i, p_val_i in [
        ("v2_eegnet", p_test_eeg, p_val_eeg),
        ("v2_tcnet", p_test_tc, p_val_tc),
        ("v2_fbcnet", p_test_fbc, p_val_fbc),
    ]:
        test_m = compute_metrics(y_test, p_test_i.argmax(1))
        val_m = compute_metrics(y_val, p_val_i.argmax(1))
        spec = BaseModelSpec(
            name=name,
            input_type="raw_64ch" if "fbcnet" not in name else "filter_bank_64ch",
            model_path=f"models/{name}.pt",
            val_macro_f1=val_m["macro_f1"],
            test_macro_f1=test_m["macro_f1"],
        )
        base_specs.append(spec)
        print(f"  {name}: val macroF1={spec.val_macro_f1:.4f} test macroF1={spec.test_macro_f1:.4f}")

    # --- Save ensemble config ---
    config = {
        "meta_learner_path": str(meta_path.relative_to(ROOT)),
        "base_models": [
            {"name": s.name, "input_type": s.input_type, "model_path": s.model_path}
            for s in base_specs
        ],
    }
    config_path = MODELS_DIR / "v2_ensemble_v2_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # --- Final comparison bar chart ---
    all_models: dict[str, float] = {}

    for fname in ["rf_results.json", "csp_lda_results.json", "eegnet_results.json"]:
        path = REPORTS_DIR / fname
        if path.exists():
            with open(path) as f:
                r = json.load(f)
                all_models[f"v1_{r['model_name']}"] = r["test_metrics"]["macro_f1"]

    prev_ensemble_path = REPORTS_DIR / "v2_ensemble_results.json"
    if prev_ensemble_path.exists():
        with open(prev_ensemble_path) as f:
            prev = json.load(f)
        for bm in prev.get("base_models", []):
            all_models[bm["name"]] = bm["test_macro_f1"]
        all_models["v2_ensemble_initial"] = prev["ensemble_test_metrics"]["macro_f1"]

    for spec in base_specs:
        all_models[spec.name + "_fixed"] = spec.test_macro_f1

    all_models["v2_deep_ensemble_fixed"] = ensemble_metrics["macro_f1"]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    names = list(all_models.keys())
    vals = list(all_models.values())
    colors = []
    for n in names:
        if n.startswith("v1_"):
            colors.append("tab:gray")
        elif "ensemble" in n.lower():
            colors.append("tab:green")
        elif "fixed" in n:
            colors.append("tab:orange")
        elif "tcnet" in n or "fbcnet" in n:
            colors.append("tab:red")
        else:
            colors.append("tab:blue")
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("All Models — Test Set Macro-F1 (with EA + fixes)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_final_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Final comparison chart → {FIG_DIR / 'v2_final_comparison.png'}")

    # --- Result JSON ---
    result = {
        "model_name": "v2_deep_ensemble_fixed",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "data_version": "v2",
        "fixes_applied": [
            "euclidean_alignment", "unified_bandpass", "overlapping_filter_bands",
            "multi_window_variance", "lr_warmup", "label_smoothing", "patience_30",
        ],
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
        "ensemble_test_metrics": ensemble_metrics,
        "meta_learner_config": {
            "type": "LogisticRegression",
            "trained_on": "val_predictions",
            "input_dim": stacked_val.shape[1],
            "coefficients": meta.coef_.tolist(),
        },
        "all_model_test_f1": all_models,
    }
    out_path = REPORTS_DIR / "v2_ensemble_v2_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")
    print("Phase E-revised (fixed) complete.")


if __name__ == "__main__":
    main()
