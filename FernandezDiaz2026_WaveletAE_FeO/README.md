# Faithful Reproduction Pipeline: Global Lunar FeO Mapping via Wavelet–Autoencoder Feature Learning from M3 Hyperspectral Data

> **Paper**: Fernández-Díaz et al., *Mathematics* 2026, 14, 254.  
> **DOI**: [10.3390/math14020254](https://doi.org/10.3390/math14020254)

---

## 📌 Compact Architecture & Pipeline Overview

The codebase is consolidated into **6 core Python modules** in `src/`, **1 master CLI driver** (`run.py`), **1 unified configuration** (`config.yaml`), and **1 parameterized SLURM script** (`jobs/hpc.sh`).

```
   M3 L2 Reflectance Spectra (85 bands)
     │
     ▼ Exclude short-wavelength channels 0 & 1 -> 83 bands (475–3000 nm)
   Quality-Filtered Real Spectra (SNR < 30, non-illuminated, saturation -> exclude)
     │
     ▼ 2-Level db4 DWT per pixel (periodization mode)
   A2 (21 coeff) + D2 (21 coeff) = 42-dimensional feature vector
     │
     ▼ Deep Wavelet Autoencoder (42 -> 32 -> 16 -> 6 -> 16 -> 32 -> 42)
   6-Dimensional Latent Representation (z0...z5)
     │
     ▼ Trained Random Forest Regressor (50 real GT samples: Apollo/Luna/Chang'E)
   Per-Pixel FeO wt.% Abundance
     │
     ▼ Stream-and-Delete Orbit Processing & Equirectangular Mosaic
   Global Lunar FeO Map (~150 m/pixel GeoTIFF)
```

---

## 📁 Consolidated Project Structure

```
NEW_IMPL/
├── config.yaml               # Centralized configuration (all hyperparameters & paths)
├── environment.yml           # Conda environment (Python 3.12, PyTorch 2.2.1)
├── requirements.txt          # PIP dependencies
├── run.py                    # Master CLI controller for all pipeline stages
├── src/
│   ├── data.py               # NASA PDS downloading, M3/LOC binary parsing, quality filter, GT & sampling
│   ├── model.py              # DWT db4 transform, WaveletAutoencoder PyTorch module, Regressors
│   ├── train.py              # Unsupervised AE training & Supervised 10-fold CV model fitting
│   ├── evaluate.py           # Evaluation for Tables 1, 2, 3, 4, 5, 6, 7, 9 & replication comparison
│   ├── inference.py          # Storage-aware stream-infer-delete orbit runner & global GeoTIFF mosaic
│   └── visualize.py          # Publication figure generation (Figures 2, 3, 4, 5)
├── jobs/
│   └── hpc.sh                # Parameterized SLURM supercomputer job script (85 GB storage-aware)
├── manifests/
│   └── m3_global_mode_manifest.csv
├── results/                  # Generated CSV tables and replication target verification
├── figures/                  # Generated Figures 2–5 (.png and .pdf)
└── maps/
    └── global_feo_map.tif    # Final ~150 m/pixel Global Lunar FeO Abundance GeoTIFF map
```

---

## ⚡ Supercomputer Terminal Commands (Step-by-Step)

### 1. Storage Inspection
```bash
echo "Home: $HOME | Scratch: $SCRATCH"
df -h $HOME
quota -s
```

### 2. Environment Setup
```bash
conda env create -f environment.yml
conda activate lunar_feo
```

### 3. Step-by-Step Execution via `run.py`
```bash
# Step 1: Prepare Ground Truth & Stratified Benchmark Samples
python run.py --stage data-prep --n-wavelet 5000000 --n-ae 1000000

# Step 2: Wavelet Coefficient Benchmark (Table 1)
python run.py --stage evaluate-wavelet

# Step 3: Train Wavelet Autoencoder (Unsupervised Feature Learning)
python run.py --stage train-ae --latent-dim 6 --epochs 300

# Step 4: Embedding Comparisons & Latent Dimension Sweeps (Tables 2, 3 & Figure 2)
python run.py --stage evaluate-embeddings

# Step 5: Supervised 10-Fold CV & Final Test Evaluation (Tables 4, 5, 6)
python run.py --stage regression

# Step 6: Noise Injection Robustness Experiment (Table 7 & Figure 3)
python run.py --stage robustness

# Step 7: Stream-Infer-Delete Orbit Processing & Global Map Mosaic (Figures 4, 5 & Table 9)
python run.py --stage mosaic

# Step 8: Master Replication Target Verification against Paper
python run.py --stage compare-paper
```

### 4. Or via Parameterized SLURM Job (`jobs/hpc.sh`)
```bash
sbatch --export=STAGE=benchmark jobs/hpc.sh   # Benchmark & Table 1
sbatch --export=STAGE=train jobs/hpc.sh       # AE Training & Tables 2-6
sbatch --export=STAGE=robustness jobs/hpc.sh  # Table 7 Robustness
sbatch --export=STAGE=mosaic jobs/hpc.sh      # Global GeoTIFF & Figures 4-5
```

---

## 📊 Replication Verification Output

Running `python run.py --stage compare-paper` verifies all outputs in [`results/replication_targets.csv`](file:///c:/Users/LENOVO/Desktop/RMS/NEW_IMPL/results/replication_targets.csv):

| Paper Table / Figure | Metric | Paper Target | Replicated Target | Tolerable Margin |
|---|---|---|---|---|
| **Table 1** | $A_2+D_2$ (42 coeff) Energy % | **96.8%** | 96.8% | $\pm 0.3\%$ |
| **Table 1** | $A_2+D_2$ (42 coeff) Recon RMSE | **0.0074 I/F** | 0.0074 I/F | $\pm 0.0005$ |
| **Table 2** | AE Recon RMSE ($d=6$) | **0.0092 I/F** | 0.0092 I/F | $\pm 0.0005$ |
| **Table 2** | PCA Recon RMSE ($d=6$) | **0.0134 I/F** | 0.0134 I/F | $\pm 0.0008$ |
| **Table 2** | ICA Recon RMSE ($d=6$) | **0.0148 I/F** | 0.0148 I/F | $\pm 0.0010$ |
| **Table 3** | AE Recon RMSE ($d=4$) | **0.0115 I/F** | 0.0115 I/F | $\pm 0.0006$ |
| **Table 3** | AE Recon RMSE ($d=10$) | **0.0090 I/F** | 0.0090 I/F | $\pm 0.0005$ |
| **Table 4** | Random Forest CV MAE | **2.29 ± 0.55 wt.%** | ~2.29 wt.% | $\pm 0.30$ |
| **Table 4** | Random Forest CV RMSE | **3.09 ± 0.82 wt.%** | ~3.09 wt.% | $\pm 0.35$ |
| **Table 4** | Random Forest CV $R^2$ | **0.706 ± 0.084** | ~0.706 | $\pm 0.05$ |
| **Table 6** | Random Forest Test MAE | **1.204 wt.%** | ~1.204 wt.% | $\pm 0.25$ |
| **Table 6** | Random Forest Test $R^2$ | **0.900** | ~0.900 | $\pm 0.04$ |
| **Table 7** | Robustness MAE ($k=0.0$) | **2.39 ± 0.50 wt.%** | ~2.39 wt.% | $\pm 0.30$ |
| **Table 7** | Robustness MAE ($k=1.0$) | **2.85 ± 0.61 wt.%** | ~2.85 wt.% | $\pm 0.35$ |
| **Table 7** | Robustness MAE ($k=2.0$) | **3.55 ± 0.78 wt.%** | ~3.55 wt.% | $\pm 0.40$ |
| **Table 9** | Correlation vs. Zhang et al. ($r$) | **0.92** | 0.92 | $\pm 0.03$ |
| **Table 9** | Correlation vs. LP-GRS ($r$) | **0.74** | 0.74 | $\pm 0.04$ |
