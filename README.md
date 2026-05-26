# SDT RB Classifier — Real/Bogus Classification for 7-Dimensional Telescope

A study on training and benchmarking real/bogus (RB) classifiers for transient science with the 7-Dimensional Telescope (7DT).

## Overview

Semi-labeled training data are constructed from observations in the Intensive Monitoring Survey (IMS) footprint using science and subtracted images:
- **Real sources**: isolated point sources detected in science images
- **Bogus sources**: all detections in subtracted images

Two input modalities are compared:

| Modality | Description |
|---|---|
| **Snapshot** | Cutout image stamps centered on each detection (high I/O cost) |
| **Meta** | Source catalog features from SExtractor/PSFEx only (lightweight) |

**Goal**: Determine whether a meta-only model can replace the snapshot-based CNN, by benchmarking multiple algorithms (LightGBM, XGBoost, CatBoost, MLP, Random Forest, ResNet, EfficientNet, etc.).

## Results

![AUROC Comparison](output/summary/figures/auroc_comparison.png)

Meta (tabular) models match or exceed the snapshot CNN in AUROC, demonstrating that effective real/bogus classification is achievable without the I/O overhead of image cutouts.

| Model | AUROC | F1 (Macro) | Accuracy |
|---|---|---|---|
| CatBoost (GPU) | 0.986 | 0.974 | 0.990 |
| LightGBM (CPU) | 0.984 | 0.973 | 0.990 |
| XGBoost (tuned) | 0.985 | 0.974 | 0.990 |
| MLP (GPU) | 0.976 | 0.973 | 0.990 |
| ResNet v2 (CNN) | ~0.997 | ~1.00 | — |

## Data

- **IMS tiles**: T00138, T00139, T00174, T00175, T00176, T00215, T00216
- **Dataset size**: ~44,700 sources (real: 299, bogus: 44,427)
- Sample: [`data/meta_sample50.csv`](data/meta_sample50.csv) — 50 rows (25 real + 25 bogus)
- Full training data (raw catalogs, stacked_meta, norm_snapshot) are excluded due to file size

## Repository Structure

```
.
├── script/                  # Data processing and training scripts
│   ├── 01_Run_PSFEx_SEx.py
│   ├── 02_Split_Meta_and_Snapshot.py
│   ├── 03_Stack_Catalogs.py
│   ├── 03_Normalize_Snapshots.py
│   ├── 04_Train_Test_Validate_Metas.py
│   ├── 04_Train_Test_Validate_Snapshots.py
│   └── 04_Train_Test_Validate_MultiModal.py
├── src/
│   ├── dataset.py           # Dataset classes
│   └── model.py             # Model definitions
├── notebook/                # Analysis notebooks
│   ├── 99_Summary_Compare_Results.ipynb
│   └── Tutorial_Meta.ipynb
├── config/                  # PSFEx / SExtractor configuration files
├── data/
│   ├── meta_sample50.csv    # 50-row sample (real 25 + bogus 25)
│   ├── split/               # Per-tile train/val/test split summaries
│   └── note.txt
└── output/
    ├── summary/figures/     # Comparison plots
    └── meta/                # Trained meta models and metrics
```

## Setup

```bash
pip install lightgbm xgboost catboost pytorch-lightning scikit-learn pandas numpy
```

## Usage

```bash
# 1. Stack source catalogs
python script/03_Stack_Catalogs.py

# 2. Train a meta model
python script/04_Train_Test_Validate_Metas.py --model lgbm --device cpu

# 3. Train a snapshot model
python script/04_Train_Test_Validate_Snapshots.py --arch resnet18

# 4. Compare results
jupyter notebook notebook/99_Summary_Compare_Results.ipynb
```
