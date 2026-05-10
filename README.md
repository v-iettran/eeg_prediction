# EEG Motor Imagery Prediction Application

An interactive Streamlit application for analysing and predicting EEG motor imagery events using the [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/). Upload an EDF file, explore brain signals, and classify whether a subject imagined moving their left or right fist.

**[Live Application →](https://eeg-prediction-viettran.streamlit.app/)**

---

## Classification Task

| Class | Label | Description |
|-------|-------|-------------|
| 0 | T1 | Imagined left fist movement |
| 1 | T2 | Imagined right fist movement |

Binary classification on runs 4, 8, and 12 of the PhysioNet eegmmidb dataset. T0 (rest) events are displayed but excluded from classification.

---

## Application Features

### Tab 1: Explore
- Upload an EDF file and inspect metadata (sampling frequency, channels, duration, annotations)
- Raw EEG signal viewer with T1/T2 event markers and time range selection
- Raw vs filtered signal comparison (8–30 Hz bandpass)
- Power spectral density plot with mu (8–13 Hz) and beta (13–30 Hz) band highlighting
- Motor cortex multi-channel view (C3, Cz, C4)
- **Interactive topographic explorer**: dual scalp heatmaps (mu and beta band power) driven by a time slider, with electrode labels, C3-C4 asymmetry indicator, and real-time event status

### Tab 2: Training
- Research methodology and model selection rationale
- V1 → V2 improvement narrative with delta metrics
- Interactive Sankey pipeline diagram showing data flow from EDF to each model
- Compact model comparison table (all models, sortable by metrics)
- Per-model detail panel with confusion matrices, training curves, and feature importance plots
- Limitations and future directions
- References

### Tab 3: Predict
- Run inference with any trained model on the uploaded EDF file
- Dual-row prediction timeline (ground truth vs prediction, with confidence percentages)
- Summary metrics (accuracy, macro F1, correct epochs, average confidence)
- Per-epoch confidence lollipop chart (correct vs incorrect)
- Epoch-level results table with progress bar confidence and checkbox correctness columns
- Per-file confusion matrix

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb v1.0.0)**

- Source: https://physionet.org/content/eegmmidb/1.0.0/
- 109 subjects, 64 EEG channels, 160 Hz sampling rate
- Runs used: 4, 8, 12 (imagined left/right fist movement)
- Subjects excluded: 88, 89, 92, 100, 104 (non-standard sampling rates or annotation issues)
- Final pool: 104 subjects

The full dataset is **not** included in this repository.

---

## Training Configuration

### Data Split
- **Train**: 80 subjects (~4,093 epochs after rejection)
- **Validation**: 12 subjects (~700 epochs)
- **Test**: 12 subjects (~700 epochs)
- Split is **subject-disjoint** - no subject appears in more than one set

### Preprocessing
- Bandpass filter: 8–30 Hz IIR Butterworth (captures mu + beta bands)
- Reference: common average
- Artifact rejection: 300 µV peak-to-peak threshold
- Euclidean Alignment: per-subject covariance whitening for cross-subject normalisation
- Epoch window: 0.0–4.0 seconds post-cue

### Feature Extraction

**Classical models (Random Forest, XGBoost)** - 31-dimensional feature vector per epoch from C3, Cz, C4:
- Log mu power (8–13 Hz) and log beta power (13–30 Hz) via Welch's method
- Relative band powers (mu/total, beta/total, mu/beta ratio)
- Variance and kurtosis (time domain)
- Hjorth mobility and complexity
- C3-C4 coherence in mu and beta bands
- C3-C4 mu and beta power asymmetry

**CSP+LDA** - 8-component Common Spatial Patterns with Ledoit-Wolf regularised covariance, log-variance features, on all 64 channels

**Deep learning models (EEGNet, EEG-TCNet, FBCNet)** - raw 64-channel epochs with per-channel z-score normalisation (fit on training set)

**FBCNet additionally** - 5 overlapping filter bank bands (8–30 Hz, 4 Hz wide, 2 Hz step)

### Evaluation Strategy
- Hyperparameter tuning: GroupKFold (k=5) with subject-disjoint folds on the training set
- Window selection: best validation macro-F1 across [0.0, 4.0] and [0.5, 2.5] second windows
- Test evaluation: single pass on held-out subjects, winning configuration only

---

## Models Trained

### V1 (initial pipeline, 150 µV rejection, ~1,850 epochs)

| Model | Channels | Features | Test Macro-F1 |
|-------|----------|----------|---------------|
| Random Forest | C3, Cz, C4 | 6-dim band power | 0.550 |
| CSP+LDA | All 64 | CSP log-variance | 0.571 |
| EEGNet | All 64 | Raw epochs | 0.709 |

### V2 (relaxed rejection, Euclidean Alignment, enriched features, ~4,093 epochs)

| Model | Channels | Features | Test Macro-F1 | vs V1 |
|-------|----------|----------|---------------|-------|
| Random Forest | C3, Cz, C4 | 31-dim enriched | 0.543 | −0.007 |
| XGBoost | C3, Cz, C4 | 31-dim enriched | 0.578 | new |
| CSP+LDA | All 64 | CSP log-variance | 0.599 | +0.029 |
| EEGNet | All 64 | Raw epochs | **0.757** | +0.049 |
| EEG-TCNet | All 64 | Raw epochs | 0.675 | new |
| FBCNet | All 64 | Filter bank + spatial | 0.703 | new |

**Best model: EEGNet (V2) - 0.757 test macro-F1** with 2,770 parameters.

---

## Project Structure

```
eeg_prediction/
├── app.py                          # Streamlit entry point
├── app/
│   ├── tab_explore.py              # Signal visualisation + topographic explorer
│   ├── tab_training.py             # Training summary, methodology, model results
│   ├── tab_predict.py              # Prediction results and evaluation
│   └── utils/
│       ├── edf_loader.py           # EDF loading, filtering, PSD computation
│       ├── inference.py            # Model loading and inference routing
│       └── colors.py               # Shared color constants
├── src/
│   ├── preprocessing.py            # Bandpass, epoching, Euclidean Alignment
│   ├── features.py                 # V1 band power features (6-dim)
│   ├── features_v2.py              # V2 enriched features (31-dim)
│   ├── models.py                   # EEGNet architecture
│   ├── tcnet.py                    # EEG-TCNet architecture
│   ├── fbcnet.py                   # FBCNet architecture + filter bank
│   ├── evaluation.py               # Metrics and plotting utilities
│   ├── data_io.py                  # EDF loading and channel standardisation
│   ├── splits.py                   # Subject split logic (deterministic)
│   └── ensemble.py                 # Stacking ensemble utilities
├── scripts/
│   ├── run_00_data_processing.py   # V1 data processing
│   ├── run_00_v2_data_processing.py # V2 data processing (relaxed rejection + EA)
│   ├── run_01_random_forest.py     # V1 RF training
│   ├── run_02_csp_lda.py           # V1 CSP+LDA training
│   ├── run_03_eegnet.py            # V1 EEGNet training
│   ├── run_04_classical_ensemble.py # V2 classical models (RF, XGB)
│   ├── run_05_v2_csp_eegnet.py     # V2 CSP+LDA + EEGNet retraining
│   ├── run_06_ensemble.py          # V2 stacking ensemble
│   ├── run_07_tcnet_fbcnet.py      # V2 EEG-TCNet + FBCNet training
│   └── build_notebook_*.py         # Notebook scaffolding scripts
├── notebooks/
│   ├── 00_data_processing.ipynb    # Run first - produces all processed data
│   ├── 01_random_forest.ipynb      # V1 RF
│   ├── 02_csp_lda.ipynb            # V1 CSP+LDA
│   ├── 03_eegnet.ipynb             # V1 EEGNet
│   ├── 04_classical_ensemble.ipynb # V2 classical models
│   ├── 05_v2_csp_eegnet.ipynb      # V2 CSP+LDA + EEGNet
│   ├── 06_ensemble.ipynb           # V2 stacking ensemble
│   └── 07_tcnet_fbcnet.ipynb       # V2 EEG-TCNet + FBCNet
├── models/                         # Trained model artifacts (.joblib, .pt)
├── reports/                        # Result JSONs + figures
├── .streamlit/config.toml          # Theme and server configuration
├── requirements.txt
└── README.md
```

---

## Running Locally

### Prerequisites
- Python 3.10+
- ~500 MB disk space for dependencies

### Installation

```bash
git clone https://github.com/v-iettran/eeg_prediction.git
cd eeg_prediction
pip install -r requirements.txt
```

### Launch the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload any EDF file from the PhysioNet eegmmidb dataset (runs 4, 8, or 12) to begin.

### Reproducing model training (optional)

1. Download the PhysioNet dataset: `mne.datasets.eegbci.load_data(subjects=range(1, 110), runs=[4, 8, 12])`
2. Run `notebooks/00_data_processing.ipynb` to generate processed epoch files
3. Run notebooks 01–07 in any order (01–03 are V1, 04–07 are V2)

---

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). The app uses CPU-only PyTorch (~200 MB) to stay within the 1 GB memory limit. All model inference runs on CPU in under 1 second per file.

---

## References

1. Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
2. Ingolfsson, T. M., et al. (2020). EEG-TCNet: An accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces. *IEEE SMC*, 2958–2965.
3. Mane, R., et al. (2021). FBCNet: A multi-view filter bank convolutional neural network for brain–computer interface. *arXiv:2104.01233*.
4. He, H. & Wu, D. (2020). Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach. *IEEE TBME*, 67(2), 399–410.
5. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
