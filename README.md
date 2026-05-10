# EEG Motor Imagery Classification

Offline training pipeline + (later) a Streamlit inference app for left- vs right-fist
imagined motor imagery on the PhysioNet EEG Motor Movement/Imagery Dataset
(eegmmidb v1.0.0, runs 4 / 8 / 12).

## Layout

```
project_root/
├── data/
│   ├── raw/          # symlink → physionet.org/files/eegmmidb/1.0.0  (gitignored)
│   └── processed/    # epoch .npz tensors + splits.json + data_manifest.json
├── notebooks/
│   ├── 00_data_processing.ipynb
│   ├── 01_random_forest.ipynb
│   ├── 02_csp_lda.ipynb
│   └── 03_eegnet.ipynb
├── src/              # shared modules used by all notebooks (and later by Streamlit)
├── models/           # trained model artifacts
├── reports/          # *_results.json + figures/
├── requirements.txt
└── README.md
```

## Run order

1. `pip install -r requirements.txt`
2. Run `notebooks/00_data_processing.ipynb` once. This produces all `data/processed/`
   artifacts. See `PIPELINE_PLAN.md` for the full design rationale.
3. Run `notebooks/01_random_forest.ipynb`, `02_csp_lda.ipynb`, `03_eegnet.ipynb`.
   These are independent of each other and can run in parallel kernels.

## Models trained

| Notebook | Model        | Channels         | Features             |
|----------|--------------|------------------|----------------------|
| 01       | RandomForest | 3 (C3, Cz, C4)   | log mu/beta band power |
| 02       | CSP + LDA    | 64               | CSP variance features  |
| 03       | EEGNet       | 64               | Raw epochs             |

Each model evaluates two epoch windows (`[0.0, 4.0]` and `[0.5, 2.5]` seconds
post-cue) and selects the winner by validation macro-F1, then reports a single
test score on the held-out subjects.

See `PIPELINE_PLAN.md` for the full specification.
