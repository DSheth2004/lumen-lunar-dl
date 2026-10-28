## 3. Results

### 3.1. Experimental Setup

#### 3.1.1. Comparison Algorithms

On the laboratory spectra, LUMEN is compared with the following methods. All are trained and
scored on identical spectra and folds.

**Classical baselines**
- **Mean of other soils.** The fraction-matched mean of the training soils, a zero-information
  reference.
- **Classical regressors on LUMEN's inputs.** Partial least squares (PLS), ridge regression and a
  random forest (RF), all using the same 142 input values as LUMEN (2 channels × 71 bands).
- **RF on raw reflectance.**
- Hyperparameters of every classical baseline are tuned by inner grouped cross-validation on the
  training soils only.

**Fernández-Díaz et al. [@fernandez2026], re-implemented in full on the same 71-band spectra**
- A two-level db4 wavelet decomposition (periodisation), keeping the A2 and D2 coefficients.
- An autoencoder 42 → 32 → 16 → 6 → 16 → 32 → 42, trained unsupervised either on M³ pixels (as
  published) or on our synthetic spectra.
- Four regressors (Lasso, linear SVR, RBF SVR, RF), with the paper's search grids, tuned by inner
  cross-validation.
- The no-autoencoder variant (wavelet coefficients → regressor).
- Twelve variants in total. "FD as published" denotes the M³-trained autoencoder followed by RF.

**Korokhin et al. [@korokhin2024] and Jung et al. [@jung2026]**
- Their outputs on the LSCC soils come from independent reproductions of each method. Each
  reproduction was validated against the paper's printed numbers: 46 of 65 and 32 of 34 gated
  targets reproduced.
- Each method is scored exactly as the paper scores itself (Section 3.4).

**On M³ (Section 3.6)**
- The Fernández-Díaz variants, retrained on M³ site spectra.
- A linear model on two M³ bands (R950/R750 and R750).
- The Lucey θ parameter [@lucey2000], refitted on M³.
- The Lucey et al. published constants applied to the Clementine reflectances of the ground-truth
  table, plus a two-band Clementine model refitted.
- The training-fold mean.

#### 3.1.2. Parameter Settings

Table 1 lists the fixed settings.
- All LUMEN hyperparameters were set from synthetic-only development runs.
- They were committed to version control, together with the evaluation protocol, before any model
  was scored on real folds.
- No hyperparameter was tuned on a held-out soil.
- The experiments of Sections 3.6 and 3.7 were committed in the same way before they were run.

**Table 1.** Fixed LUMEN settings.

| Component | Setting |
|---|---|
| Simulator | 400,000 spectra; 49 end-members; grain size 4–150 µm; scale ±7%; noise 0.2–0.8% |
| Input | 2 × 71 (log R, continuum-removed R), standardised with synthetic statistics |
| Network | 5 conv layers (32/32/64/64/64; kernels 7/5/5/3/3), GroupNorm, GELU, 2 max-pools, trunk 256→128; 0.36 M parameters |
| Losses | Dirichlet NLL + L1 (λ₁ = 10, ε = 0.005); β-NLL (β = 0.5) |
| Pretraining | 5 members, 15 epochs, AdamW, one-cycle, batch 512 |
| Fine-tuning | 1500 steps, AdamW lr 3 × 10⁻⁴, wd 10⁻⁴, warm-up 100, cosine; batch 16 real + 32 SSA-mixup + 16 replay (A4) |
| Orbital fine-tuning | 600 steps, lr 10⁻⁴, batch 16 M³ + 16 lab spectra |
| Uncertainty | 5-member ensemble; normalised conformal (CV+) intervals; gate on bootstrap skill CI |
| Hardware | CPU only (4 cores); pretraining ≈ 45 min; one fine-tune ≈ 6 s |

#### 3.1.3. Evaluation Protocols and Metrics

**Protocols**
- **Leave-one-soil-out (LOSO, 19 folds).** All four fractions of the held-out soil are excluded
  from fine-tuning, from mixup and from normalisation statistics.
- **Leave-one-mission-out (LOMO, 6 folds: A11, A12, A14, A15, A16, A17).** This is the
  out-of-distribution test. Holding out A16 removes every feldspathic soil, and holding out A17
  every high-Ti soil.

**Metrics**
- The primary metric is the per-soil mean absolute error (MAE): each soil's labelled fractions are
  averaged, then the 19 soils are averaged. The number of independent units is therefore the
  number of soils, not the number of spectra.
- R² and RMSE are reported per spectrum.
- Methods are compared with paired Wilcoxon signed-rank tests over soils [@wilcoxon1945], with
  Holm correction [@holm1979] over the 13 targets or over the comparators.
- Bootstrap confidence intervals are computed over soils.

### 3.2. Simulator Validation

Every self-test of the simulator passes.

| Test | Result |
|---|---|
| Round trip (each end-member's k(λ) reproduces its measured reflectance) | to 7.8 × 10⁻¹¹ |
| Truncated prior: 99th-percentile synthetic TiO₂ | 30.5 → 14.0 wt% (LSCC maximum 10.7) |
| Truncated prior: 99th-percentile synthetic FeO | 31.8 → 24.4 wt% |
| Label model (modal composition × phase chemistry vs measured bulk oxides) | r = 0.93–0.98, but mare TiO₂ biased by +2.9 wt% (affects only the scale of the pretraining labels) |
| Coverage: median nearest-synthetic-neighbour distance of the 75 LSCC spectra | 0.102, vs 0.144 between synthetic spectra; the real data lie inside the synthetic distribution |
| Separability: AUC of a classifier separating real from synthetic spectra | 0.97 (0.99 against noise-free synthetics); real soils are redder than their nearest synthetic neighbours |
| Measured <45 µm spectra reproduced by mixing their sieve fractions | to 0.6% RMS |

The last two rows matter for what follows. The real spectra sit inside the synthetic
distribution, yet remain separable from it: this is the sim-to-real gap that fine-tuning must
close.

### 3.3. Accuracy on the LSCC Soils

Table 2 gives per-soil MAEs.

**Leave-one-soil-out (LOSO)**
- **Skill.** Against the zero-information reference, LUMEN (A4) has a skill of 0.50 for
  composition and 0.72 for FeO.
- **Against classical baselines.** A4 has the lower error than each classical baseline on 12 of
  13 targets; the exception is always olivine. After Holm correction it is significantly better
  than PLS on 9 targets and than the mean of other soils on 10.
- **Against the strongest baseline**, a random forest on the same inputs, A4 is lower on 12 of 13
  targets. Only maturity is significant after correction (for FeO, raw p = 0.005, Holm p = 0.064).

**Leave-one-mission-out (LOMO)**
- Every method degrades. A4's composition MAE rises from 2.66 to 4.70 vol% and its FeO MAE from
  1.26 to 2.59 wt%.
- A4 remains lower than RF on 10 of 13 targets, and significantly so for LCP, HCP and TiO₂.
- For the "other" class, no method beats the zero-information reference.

**Table 2.** Per-soil MAE, LOSO / LOMO (composition in vol%, oxides in wt%, maturity in log₁₀
Is/FeO). Bold: lowest per row and protocol.

| Target | Mean of others | PLS | Ridge | RF | LUMEN (A4) |
|---|---|---|---|---|---|
| PLG | 14.09 / 17.74 | 11.13 / 16.66 | 10.58 / 16.68 | 7.13 / 12.45 | **4.82 / 11.31** |
| LCP | 3.23 / 3.89 | 2.40 / 3.69 | 2.38 / 3.47 | 2.23 / 3.33 | **1.75 / 2.50** |
| HCP | 2.89 / 3.49 | 1.92 / 1.72 | 1.78 / 1.78 | 1.57 / 1.93 | **1.07 / 1.22** |
| OLV | 0.93 / 1.05 | 0.87 / 1.06 | 0.87 / 1.03 | **0.84 / 0.98** | 0.98 / 1.13 |
| ILM | 2.63 / 3.22 | 1.89 / 2.18 | 1.81 / 2.10 | 1.68 / 2.30 | **1.22 / 1.69** |
| AGG | 9.60 / 9.53 | 9.85 / 12.29 | 9.29 / 11.86 | 7.14 / 9.94 | **5.78 / 9.18** |
| OTH | 3.85 / **4.81** | 3.62 / 5.34 | 3.45 / 5.22 | 3.37 / 5.41 | **3.02** / 5.88 |
| FeO | 4.47 / 5.56 | 2.77 / 3.58 | 2.64 / 4.39 | 2.13 / 3.69 | **1.26 / 2.59** |
| TiO₂ | 2.67 / 3.28 | 1.70 / 2.30 | 1.63 / 1.90 | 1.43 / 2.27 | **0.70 / 1.37** |
| Al₂O₃ | 6.00 / 7.53 | 4.03 / 5.90 | 3.65 / 6.44 | 2.84 / 5.41 | **1.71 / 4.33** |
| CaO | 2.41 / 3.04 | 1.87 / 2.98 | 1.72 / 3.20 | 1.23 / 2.43 | **0.87 / 2.20** |
| MgO | 1.99 / 2.50 | 1.40 / 2.21 | 1.31 / 2.12 | 0.92 / **1.71** | **0.78** / 1.74 |
| log₁₀ Is/FeO | 0.31 / 0.30 | 0.20 / 0.24 | 0.22 / 0.21 | 0.20 / 0.23 | **0.11 / 0.15** |
| Composition (mean) | 5.32 / 6.25 | 4.53 / 6.13 | 4.31 / 6.02 | 3.42 / 5.19 | **2.66 / 4.70** |

**Configuration chosen inside the cross-validation (E1).** The pre-registered primary model, A4,
is not the best configuration (Section 3.7). We therefore re-ran the choice between A4 and the
simpler A2 inside each outer training fold, by inner five-fold grouped cross-validation, so that
the outer-fold estimate stays confirmatory. [[PENDING-E1: selection counts; LUMEN-selected vs RF,
LOSO, Holm over 13 targets; LOMO]]

![Figure 2](../figures/step06_pred_vs_true.png)

**Figure 2.** LUMEN (A4) predictions against measured values for the 19 LSCC soils under LOSO.

### 3.4. Comparison with Published Methods

#### 3.4.1. Wavelet–Autoencoder FeO Regression [@fernandez2026]

This is the fairest comparison: the pipeline is trained on exactly the same spectra and folds as
LUMEN (Table 3).

**LOSO**
- LUMEN's FeO MAE is 1.26 wt% (R² = 0.85).
- FD as published has 2.17 wt% (R² = 0.62), so LUMEN's error is 42% lower.
- The best FD variant on these data, wavelet → RBF SVR without the autoencoder, has 1.91 wt%.
- LUMEN is lower than all twelve variants. Each paired test has p < 0.05, which is the test fixed
  in the protocol. With the additional, more conservative Holm correction over the twelve
  variants, 3 of 12 remain significant (vs FD as published: Holm p = 0.069).

**LOMO**
- LUMEN has 2.59 wt%, against 3.48 for FD as published (p = 0.018) and 2.91 for the best variant
  (p = 0.49).

**The effect of the autoencoder depends on the protocol**
- Under LOSO, the M³-trained autoencoder is worse than the raw wavelet coefficients for 4 of 4
  regressors. This agrees with our reproduction of the original study, where RF on the raw
  coefficients reached 1.55 wt% against 1.95 wt% with the autoencoder.
- Under LOMO, the autoencoder helps the two linear regressors.

**Table 3.** FeO (wt%) on identical spectra and folds. "FD-AE-M3" is the published autoencoder,
trained on M³ pixels; "FD-AE-synth" is trained on our synthetic spectra; "FD-wavelet" has no
autoencoder.

| Method | MAE LOSO | R² LOSO | MAE LOMO | R² LOMO |
|---|---|---|---|---|
| **LUMEN (A4)** | **1.26** | **0.85** | **2.59** | 0.57 |
| LUMEN (A2, pretrain + real) | 1.23 | 0.86 | 2.64 | 0.58 |
| FD-AE-M3 / RF (published) | 2.17 | 0.62 | 3.48 | 0.25 |
| FD-AE-M3 / Lasso | 2.48 | 0.55 | 3.09 | 0.39 |
| FD-AE-M3 / SVR-lin | 2.19 | 0.65 | 3.06 | 0.41 |
| FD-AE-M3 / SVR-RBF | 2.06 | 0.62 | 3.79 | 0.07 |
| FD-AE-synth / SVR-lin | 2.09 | 0.65 | 2.91 | 0.48 |
| FD-wavelet / SVR-RBF | 1.91 | 0.67 | 3.28 | 0.17 |
| FD-wavelet / Lasso | 2.07 | 0.70 | 3.95 | −0.01 |
| RF (LUMEN inputs) | 2.13 | 0.62 | 3.69 | 0.16 |
| RF (raw reflectance) | 2.19 | 0.58 | 3.45 | 0.27 |
| Mean of other soils | 4.47 | −0.06 | 5.56 | −0.70 |

*(The remaining FD variants are in the repository tables.)*

![Figure 3](../figures/step06_fernandez_feo.png)

**Figure 3.** FeO error of LUMEN and of the Fernández-Díaz et al. variants on identical folds.

#### 3.4.2. Shkuratov-Model Inversion [@korokhin2024]

Each method is scored on five-mineral proportions (CPX, OPX, OLV, PLG, ILM, renormalised to 100%)
over the three sieved fractions, as the reproduction of Korokhin et al. scores itself (Table 4).

**LOSO**
- LUMEN reduces the error for every mineral by 54–74%. Every difference is Holm-significant.
- For example: olivine 1.91 vs 7.44 points, and plagioclase 5.55 vs 12.08.

**LOMO**
- LUMEN is still lower on four of five minerals, but significantly so only for olivine. It is
  slightly worse for plagioclase (12.64 vs 12.08).

**Caveat**
- The inversion never sees an LSCC soil, whereas LUMEN is fine-tuned on the others.
- The like-for-like comparison is therefore LUMEN's zero-shot model (A0, physics-trained only). It
  is lower on three of five minerals, but significantly so only for olivine.

**Maturity.** LUMEN's Is/FeO ranks the soils better than the inversion's nFe parameter. Spearman ρ
is 0.72, 0.81 and 0.96 for the <10, 10–20 and 20–45 µm fractions, against 0.55, 0.37 and 0.44.

**Table 4.** MAE of five-mineral proportions (percentage points), 19 soils.

| Mineral | Mean of others | Korokhin fit | LUMEN A0 (zero-shot) | LUMEN A4 (LOSO) | LUMEN A4 (LOMO) | Holm p, A4 vs fit (LOSO) |
|---|---|---|---|---|---|---|
| CPX | 6.78 | 5.30 | 8.18 | **2.24** | 3.14 | 0.007 |
| OPX | 9.18 | 8.37 | 8.98 | **3.25** | 6.86 | 0.007 |
| OLV | 2.21 | 7.44 | 2.58 | **1.91** | 2.50 | <0.001 |
| PLG | 18.44 | 12.08 | 10.40 | **5.55** | 12.64 | 0.007 |
| ILM | 6.26 | 5.24 | 3.06 | **2.39** | 4.06 | 0.007 |

#### 3.4.3. Hapke–Bayesian Inversion [@jung2026]

The comparison uses the four test soils of Jung et al. (12001, 12030, 15071 and 71061, 20–45 µm),
and their crystalline proportions.
- **LUMEN.** Its worst-case LOSO errors are 1.00 points (olivine), 1.24 (ilmenite) and 7.95
  (plagioclase). All three published accuracy bounds (7, 10 and 30 points) are met.
- **The converged inversion** reaches 17.56 points for ilmenite.
- **Caveat.** With four soils this is weak evidence. The zero-shot model also meets every bound,
  while the mean of other soils misses the ilmenite and plagioclase bounds.
- **Maturity.** Across the four soils, LUMEN's Is/FeO correlates with the measured value at
  r = 0.99, against r ≈ 0.00 for the inversion's maturity index.

### 3.5. Uncertainty and Identifiability

**Raw ensemble intervals are over-confident** (Figure 4). For nominal 68/90/95% intervals, mean
empirical coverage is:
- 0.59/0.82/0.89 under LOSO;
- 0.33/0.55/0.64 under LOMO.

**Normalised conformal calibration fixes this:**
- 0.68/0.90/0.95 under LOSO;
- 0.64/0.86/0.92 under LOMO.

Exact CV+ uses models that saw neither the calibration soil nor the test soil. It gives
0.69/0.90/0.96 (LOSO) and 0.70/0.88/0.93 (LOMO), so the approximation is slightly conservative out
of distribution.

**The identifiability gate passes 11 of 13 targets** (Table 5). It withholds:
- **olivine**, whose LOSO skill is −0.06 (CI −0.40 to 0.22);
- the **"other"** class (0.22, CI −0.03 to 0.42).

Under LOMO, the agglutinate fraction also loses significance (0.04, CI −0.34 to 0.25).

**Table 5.** Skill over the zero-information reference with 95% bootstrap CI.

| Target | LOSO skill [CI] | LOMO skill [CI] | Mapped |
|---|---|---|---|
| PLG | 0.66 [0.47, 0.78] | 0.36 [0.19, 0.52] | yes |
| LCP | 0.46 [0.32, 0.59] | 0.36 [0.20, 0.50] | yes |
| HCP | 0.63 [0.53, 0.71] | 0.65 [0.55, 0.74] | yes |
| OLV | −0.06 [−0.40, 0.22] | −0.08 [−0.38, 0.16] | **no** |
| ILM | 0.54 [0.40, 0.64] | 0.47 [0.36, 0.57] | yes |
| AGG | 0.40 [0.16, 0.54] | 0.04 [−0.34, 0.25] | yes |
| OTH | 0.22 [−0.03, 0.42] | −0.22 [−0.51, −0.03] | **no** |
| FeO | 0.72 [0.61, 0.80] | 0.53 [0.44, 0.60] | yes |
| TiO₂ | 0.74 [0.61, 0.82] | 0.58 [0.45, 0.65] | yes |
| Al₂O₃ | 0.71 [0.56, 0.82] | 0.43 [0.28, 0.54] | yes |
| CaO | 0.64 [0.50, 0.75] | 0.28 [0.16, 0.38] | yes |
| MgO | 0.61 [0.37, 0.77] | 0.31 [0.05, 0.47] | yes |
| log₁₀ Is/FeO | 0.63 [0.56, 0.71] | 0.50 [0.32, 0.61] | yes |

![Figure 4](../figures/step05_coverage.png)

**Figure 4.** Empirical against nominal interval coverage, for the raw ensemble and after
conformal calibration, under LOSO and LOMO.

### 3.6. FeO from M³ Spectra at Returned-Sample Sites

This experiment tests the base method on its own task, orbital FeO, against the ground truth it
was calibrated on [@fernandez2026; @qiu2025].

**Protocols**
- **P1.** The base study's own 10-fold cross-validation over sites, so the numbers are comparable
  with the published MAE of 2.29 wt%.
- **P2.** Leave-one-landing-site-out, where all stations of a mission are held out together. Those
  stations share terrain and are spatially autocorrelated, so this is the spatially honest test.

[[PENDING-E2: Table 6 — MAE, RMSE, R², bias over sites under P1 and P2 for:
- LUMEN-M3 (zero-shot), LUMEN-M3 + recalibration, LUMEN-M3-FT, LUMEN-embed + ridge;
- FD as published and its variants;
- two M³ bands; Lucey θ (M³);
- Lucey 2000 constants on Clementine (flagged in-sample); two Clementine bands;
- training-fold mean.]]

[[PENDING-E2: primary test — LUMEN-M3-FT vs FD as published under P2 (paired Wilcoxon over sites,
Holm over comparators); P1 seed sensitivity; single-pixel sensitivity]]

[[PENDING-E6: conformal coverage of LUMEN-M3-FT at 68/90/95% under P2]]

[[PENDING-E2: Figure 5 — predicted vs laboratory FeO at the sites, P2]]

### 3.7. Ablations and Parameter Analysis

**Pretraining helps, but the pre-registered primary model is not the best configuration**
(Table 7).
- **Pretraining.** A2 (pretrained, fine-tuned on real spectra) has a lower error than A1 (the same
  network trained from scratch on real spectra) on all 13 targets under LOSO. The 13 targets are
  correlated, however, and no single target's difference is significant after Holm correction.
  Under LOMO, two targets are (HCP and maturity).
- **SSA-mixup and synthetic replay.** These were meant to enlarge the 18-soil training set. Added
  to the pretrained network (A4 vs A2), they lower the error on only 1 of 13 targets. No A4–A2
  difference is significant.
- **Mixup without pretraining.** Mixup does help a network trained from scratch under LOSO (lower
  on 12 of 13 targets), but not under LOMO (2 of 13).
- **Zero-shot model (A0).** The physics-only analogue of a simulator-trained inversion has a
  composition MAE of 9.1 vol%, far above every fine-tuned variant. This is the sim-to-real gap made
  concrete.

**Table 7.** Ablations: per-soil MAE, LOSO / LOMO. A4 is the pre-registered primary model.

| Target | A0 zero-shot | A1 scratch, real | A1m scratch + mixup | A2 pretrain + real | A3 pretrain + mixup | A4 full |
|---|---|---|---|---|---|---|
| PLG | 13.99 / 13.99 | 4.84 / 11.76 | 4.21 / 12.01 | 4.17 / 12.01 | 4.25 / 12.31 | 4.82 / 11.31 |
| LCP | 4.88 / 4.88 | 1.58 / 2.45 | 1.57 / 2.57 | 1.42 / 2.12 | 1.41 / 2.15 | 1.75 / 2.50 |
| HCP | 4.51 / 4.51 | 1.21 / 1.72 | 1.19 / 1.82 | 0.85 / 1.09 | 0.94 / 1.15 | 1.07 / 1.22 |
| OLV | 1.22 / 1.22 | 0.91 / 1.02 | 0.92 / 1.04 | 0.86 / 1.03 | 0.93 / 1.06 | 0.98 / 1.13 |
| ILM | 1.83 / 1.83 | 1.30 / 1.95 | 1.30 / 2.07 | 1.08 / 1.70 | 1.22 / 1.90 | 1.22 / 1.69 |
| AGG | 25.53 / 25.53 | 6.19 / 9.61 | 5.68 / 9.34 | 5.42 / 9.64 | 5.50 / 9.45 | 5.78 / 9.18 |
| OTH | 12.00 / 12.00 | 2.78 / 5.63 | 2.70 / 5.82 | 2.69 / 5.82 | 2.77 / 6.06 | 3.02 / 5.88 |
| FeO | 4.57 / 4.57 | 1.46 / 2.75 | 1.32 / 3.00 | 1.23 / 2.64 | 1.23 / 2.71 | 1.26 / 2.59 |
| TiO₂ | 3.91 / 3.91 | 0.82 / 1.43 | 0.76 / 1.57 | 0.56 / 1.08 | 0.66 / 1.36 | 0.70 / 1.37 |
| Al₂O₃ | 3.79 / 3.79 | 1.74 / 4.35 | 1.58 / 4.62 | 1.51 / 4.24 | 1.61 / 4.54 | 1.71 / 4.33 |
| CaO | 1.69 / 1.69 | 0.76 / 2.12 | 0.74 / 2.29 | 0.73 / 2.20 | 0.76 / 2.29 | 0.87 / 2.20 |
| MgO | 1.84 / 1.84 | 0.82 / 1.58 | 0.69 / 1.69 | 0.71 / 1.74 | 0.71 / 1.75 | 0.78 / 1.74 |
| log₁₀ Is/FeO | 0.30 / 0.30 | 0.15 / 0.19 | 0.14 / 0.18 | 0.12 / 0.14 | 0.13 / 0.15 | 0.11 / 0.15 |
| Composition (mean) | 9.14 / 9.14 | 2.69 / 4.88 | 2.51 / 4.95 | 2.36 / 4.77 | 2.43 / 4.87 | 2.66 / 4.70 |

**Is the network itself better than classical regression?** A1 is the same network trained from
scratch, without the physics. Against RF it is already significantly better (Holm) on 5 of 13
targets under LOSO (PLG, ILM, FeO, TiO₂ and Al₂O₃), and on 3 of 13 under LOMO. The simpler
pretrained variant A2 is significantly better than RF on 8 of 13 targets under LOSO and 6 of 13
under LOMO. A2 was identified after the fact, which is why its choice is moved inside the
cross-validation in Section 3.3 (E1).

**Input representation and architecture (E4).** [[PENDING-E4: log-R-only, continuum-removed-only,
MLP and 1-D ResNet against the main network, 2 members each, LOSO/LOMO, Holm over 13 targets]]

### 3.8. Application to M³ Scenes

**Calibration**
- On the Apollo 16 landing-site pixel, the published laboratory-scale calibration (Equation (10))
  reproduces the LSCC 62231 spectrum to 9.1% RMS.
- A line re-derived from that pixel reproduces it to 1.4%.
- LUMEN estimates FeO there at 5.0 wt%, against 4.9 wt% measured for 62231 and 4.6 wt% for the
  Apollo 16 mean. It over-estimates TiO₂ (2.1 vs 0.6 wt%).
- Because 62231 is in the final training set, this is a sanity check, not a validation.

**Shapley scene** (Figure 6). The maps of the 11 identifiable targets show the expected contrasts:
- FeO is 10.2 wt% on mare and 6.6 wt% on highland; Al₂O₃ is 20.0 and 24.3 wt%.
- Fresh craters stand out as immature.

Comparison with the published maps of the same strip [@korokhin2024]:
- Their high-Ca pyroxene map is anti-correlated with LUMEN's (r = −0.79), and places more
  clinopyroxene in the highlands than in the mare.
- Their maturity parameter is nearly unrelated to LUMEN's Is/FeO across the scene (r = −0.28).

Destriping the strip lowers stripe amplitude by 26–47%. It leaves the maps correlated at
r = 0.90–0.99 with the undestriped ones.

**Aristarchus.** LUMEN finds much less olivine and much more plagioclase in the crystalline
fraction than the posterior abundances of Jung et al. [@jung2026] (plagioclase 52% vs 29% in
region B). LUMEN's olivine output fails the identifiability gate, and the training soils contain
little olivine. The disagreement is therefore a domain-shift warning, not evidence against that
study.

![Figure 6](../figures/step07_shapley_maps.png)

**Figure 6.** LUMEN maps of the Shapley-region M³ strip for identifiable targets. Olivine and
"other" are withheld by the identifiability gate.
