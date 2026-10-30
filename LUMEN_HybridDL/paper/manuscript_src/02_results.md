## 3. Results

### 3.1. Experimental Setup

#### 3.1.1. Comparison Algorithms

In the spectra in the lab, all the methods use identical spectra and folds. Classical baselines are implemented using the fraction-matched average of the training soils (zero information baseline); partial least squares (PLS), ridge regression, and random forest (RF) run on LUMEN’s 142 inputs; RF on raw reflectance, all of which were optimized with inner grouped cross-validation. The Fernández-Díaz et al. pipeline [@fernandez2026] was completely reimplemented on the identical spectra (db4 A2 + D2 wavelets coefficients; 42 → 32 → 16 → 6 → 16 → 32 → 42 autoencoder trained on M³ pixels as described in the paper or on our synthetic spectra; Lasso, linear SVR, RBF SVR, and RF with search grids from the paper; and no autoencoder version), producing 12 variants, among which "FD as published" is the one that uses M³-trained autoencoder with RF. Outputs of Korokhin et al. [@korokhin2024] and Jung et al. [@jung2026] are derived from independent replications verified against the numbers reported in the respective paper (46 of 65 and 32 of 34 gated targets, respectively).

For the M³ dataset (section 3.6), we consider the comparison between LUMEN zero-shot, zero-shot LUMEN plus a linear calibration of $y = a + b\hat y$ based on the training sites, LUMEN-M3-FT, and LUMEN-M3-ridge using the 128-dimensional trunk embedding (referred to as "LUMEN-embed + ridge"), the Fernández-Díaz variations with a re-trained model on M³ site spectra, a linear model using two M³ bands (R950/R750 and R750), Lucey's θ parameter [@lucey2000] trained on M³, Lucey et al.'s published constants with a re-trained two-band model on Clementine reflectances in the table, and the training fold mean.

#### 3.1.2. Parameter Settings

Table 1 contains the fixed parameters. All LUMEN hyperparameters were tuned using only the synthetic data development, and the evaluation method was fixed before applying the models to any real folds, and the following experiments (Sections 3.3, 3.6, and 3.7) were committed before performing them.

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

Figure 4 shows the experimental design. Using leave-one-soil-out (LOSO; 19 folds) strategy, all four fractions of the held-out soil are left out of fine-tuning, mixup, and normalization calculations. Leave-one-mission-out (LOMO; 6 folds: A11, A12, A14, A15, A16, A17) is used as an out-of-distribution test – leaving A16 out removes all feldspathic soils, while leaving A17 out removes all high-Ti soils. The main measure is the mean absolute error (MAE) per soil: the averages of the fractions within each soil are taken, followed by averaging over the 19 soils, such that the number of independent units equals the number of soils rather than the number of spectra. R² and RMSE are calculated per spectrum. Comparisons of methods are made using paired Wilcoxon signed-rank tests per soil [@wilcoxon1945], with Holm corrections [@holm1979] per 13 target minerals or per comparators, and confidence intervals are bootstrapped over soils.

![Figure 4](../figures/fig04_evaluation.png)

**Figure 4.** Evaluation design. The laboratory predictors are validated through leave-one-soil-out and leave-one-mission-out cross-validation, where the A2/A4 predictor is chosen through inner grouped five-fold cross-validation for each outer training set. The orbital FeO predictor is validated on the 49 sample-return sites through ten-fold cross-validation on the sites (P1, following [@fernandez2026]'s procedure) and leave-one-landing-site-out cross-validation (P2).

### 3.2. Simulator Validation

All simulator's self-tests are passed. The k(λ) of each end-member reproduces its measured reflectance to 7.8 × 10⁻¹¹. Truncating the composition prior reduces the 99th percentile synthetic TiO₂ from 30.5 to 14.0 wt% (LSCC maximum 10.7) and FeO from 31.8 to 24.4 wt%. The LSCC bulk oxides are reproduced by modal composition × class-average phase chemistry with r = 0.93–0.98. Mare TiO₂ is biased by +2.9 wt% which only affects the scale of the pretraining labels. Mixing of their sieve fractions reproduces the measured <45 µm spectra to 0.6% RMS. The median nearest synthetic neighbor distance of the 75 LSCC spectra is 0.102, compared to 0.144 between synthetic spectra, indicating that real data lie inside the synthetic distribution; yet a classifier separates real from synthetic spectra with an AUC of 0.97 (0.99 against noise-free synthetics), with real soils being redder than their nearest synthetic neighbors. Fine tuning must close this sim-to-real gap.

### 3.3. Accuracy on the LSCC Soils

Table 2 shows per-soil MAE values. Under LOSO, LUMEN (A4) has a skill of 0.50 over the zero-information reference for composition and 0.72 for FeO. It has a lower error than all classical baselines on 12 of 13 targets, the exception always being olivine, and after Holm correction it is significantly better than PLS on 9 targets and than the mean of other soils on 10. A4 is lower than the strongest baseline, a random forest on the same inputs, for 12 out of 13 targets, but only maturity is significant after correction (for FeO, raw p = 0.005, Holm p = 0.064). Under LOMO all methods degrade, with A4’s composition MAE increasing from 2.66 to 4.70 vol% and its FeO MAE from 1.26 to 2.59 wt%. A4 is below RF for 10 of the 13 targets, significantly so for LCP, HCP and TiO₂. The zero-information reference is not beaten by any method of “other” class.

**Table 2.** Per-soil MAE, LOSO / LOMO (composition in vol%, oxides in wt%, maturity in log₁₀ Is/FeO). Bold: lowest per row and protocol.

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

As the pre-trained primary model A4 was not optimal (Section 3.7) the A4/A2 model selection was performed using an inner 5-fold stratified cross-validation for each of the outer folds so that the outer estimates remained confirmatory. A2 was chosen in all cases (19/19 and 6/6 for LOSO and LOMO respectively). This model predicted 1.23 wt% FeO, 0.55 wt% TiO₂ and 2.35 vol%. When compared to the RF model, the A2 performs worse than A4 on 12/13 targets but is not olivine. It performs better than the RF on 8 targets when corrected for multiple comparisons using the Holm method. These include PLG, LCP, HCP, ILM, FeO (1.23 vs 2.13 wt%), TiO₂, Al₂O₃ and maturity (Holm p = 0.011). When compared to PLS and ridge regression it performs better on 11/13 targets and performs better than the mean of the other soils on 10/13 targets. The LOMO model performs better than the RF model on 6 targets (LCP, HCP, FeO (2.63 wt%), TiO₂, Al₂O₃ and maturity). No differences were found between the A4 and A2 models using either protocol. The models used here have been re-generated using the pre-trained ensemble members on a different machine. This resulted in all the MAE values for each soil being reproduced within 0.06 for all targets, models and protocols. The predictions made by LOSO protocol for 4 key targets are shown in Figure 5.

![Figure 5](../figures/fig05_pred_true.png)

**Figure 5.** Prediction of LUMEN (A4) against observed values of 19 LSCC soils for each of FeO, TiO₂, plagioclase and maturity using Leave One Out Cross Validation. Each dot represents a sieve fraction. The line shows a 1:1 relationship.

### 3.4. Comparison with Published Methods

#### 3.4.1. Wavelet–Autoencoder FeO Regression [@fernandez2026]

This is the fairest comparison, since the pipeline was trained using the identical set of spectra and folds (Table 3, Figure 6). For LOSO, LUMEN outperforms FD as published in FeO prediction with 1.26 wt% (R² = 0.85) and is better by 42% compared to FD as published (error = 2.17 wt%, R² = 0.62) and all twelve variants, the best of which was wavelet → RBF SVR without the autoencoder (error = 1.91 wt%). In each paired test, p < 0.05 (the test is fixed in the protocol); with an extra Holm correction across the twelve variants, 3 out of 12 are significant (vs FD as published, Holm p = 0.069). For LOMO, LUMEN gives 2.59 wt% versus 3.48 for FD as published (p = 0.018) and 2.91 wt% for the best variant (p = 0.49). For LOSO, the autoencoder trained on M³ data performs worse for each of four regressors, just like in our reproduction of the original paper (RF 1.55 wt% vs. 1.95 wt% with the autoencoder), while for LOMO, it is beneficial the two linear regressors.

**Table 3.** Values for FeO (wt%) have been provided for similar spectra and folds. "FD-AE-M3" is the tag associated with the published autoencoder model which is trained using M³ pixels, whereas "FD-AE-synth" means that autoencoder is trained using our synthetic spectra. "FD-wavelet" represents a model with no autoencoder. Other variants of FD can be found in tables of the repository.

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

![Figure 6](../figures/fig06_fd_feo.png)

**Figure 6.** Per-soil FeO error of LUMEN, the Fernández-Díaz et al. variants and the classical baselines on identical folds, under leave-one-soil-out (left) and leave-one-mission-out (right).

#### 3.4.2. Shkuratov-Model Inversion [@korokhin2024]

For each method, there are scores in five minerals (CPX, OPX, OLV, PLG, ILM) in proportions of 100% renormalized among three sieved fractions of soils according to the replication from Korokhin et al. (see Table 4). In the LOSO validation, LUMEN achieves an improvement in scores of the same order –54 to 74 percent for each mineral (all Holm-significant); for example, olivine gets the score 1.91 versus 7.44 and plagioclase – 5.55 versus 12.08. In the LOMO validation, LUMEN again obtains better scores for four of the five minerals, being only significant for olivine, but is slightly inferior for plagioclase with the score 12.64 versus 12.08. The inversion method does not produce LSCC soils at all, but LUMEN is fine-tuned on the rest soils; therefore, the fair comparison would be for LUMEN without training on A0 (only physics): LUMEN has lower scores in three of the five minerals (being significant for olivine). Is/FeO parameter from LUMEN soils has also greater soil ranking than the nFe parameter from the inverted soils in correlation coefficients (ρ) equal 0.72, 0.81, and 0.96 for <10, 10–20, and 20–45 µm fractions respectively against 0.55, 0.37, and 0.44.

**Table 4.** MAE of five-mineral proportions (percentage points), 19 soils.

| Mineral | Mean of others | Korokhin fit | LUMEN A0 (zero-shot) | LUMEN A4 (LOSO) | LUMEN A4 (LOMO) | Holm p, A4 vs fit (LOSO) |
|---|---|---|---|---|---|---|
| CPX | 6.78 | 5.30 | 8.18 | **2.24** | 3.14 | 0.007 |
| OPX | 9.18 | 8.37 | 8.98 | **3.25** | 6.86 | 0.007 |
| OLV | 2.21 | 7.44 | 2.58 | **1.91** | 2.50 | <0.001 |
| PLG | 18.44 | 12.08 | 10.40 | **5.55** | 12.64 | 0.007 |
| ILM | 6.26 | 5.24 | 3.06 | **2.39** | 4.06 | 0.007 |

#### 3.4.3. Hapke–Bayesian Inversion [@jung2026]

For the four soils investigated by Jung et al. (samples 12001, 12030, 15071, and 71061; grain size 20–45 µm), arranged in descending order according to the amount of crystalline phases, the LOSS error of LUMEN is 1.00 for olivine, 1.24 for ilmenite, and 7.95 for plagioclase, and thus satisfies all three known accuracy levels (7, 10, and 30 points). On the other hand, the inverted maturity index results in 17.56 points for ilmenite. Considering that only four soils have been examined, this provides lacking evidence: the zero-shot model satisfies all conditions simultaneously, while the average result for other soils does not satisfy the limits of ilmenite and plagioclase. The Is/FeO ratio from LUMEN compares to the measured one at r = 0.99, while the maturity index from the inversion is at approximately r = 0.00.

### 3.5. Uncertainty and Identifiability

Raw ensemble confidence intervals are overconfident (Figure 7): the actual 68/90/95% confidence intervals have means of 0.59/0.82/0.89 when using LOSO and 0.33/0.55/0.64 when using LOMO. Normalised conformal calibration restores the 0.68/0.90/0.95 intervals for LOSO and 0.64/0.86/0.92 for LOMO. Exact CV+, where models were trained without both calibration soil and test soil, produces 0.69/0.90/0.96 (LOSO) and 0.70/0.88/0.93 (LOMO), suggesting that the approximation is somewhat conservative when generalising out-of-distribution. Identifiability gate meets 11 of 13 targets (Table 5). It does not include olivine, which had a LOSO skill of −0.06 (−0.40 to 0.22) and “other” class soil, which had a LOSO skill of 0.22 (−0.03 to 0.42). Agglutinate fraction soil was also excluded due to its lack of significance in LOMO.

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

![Figure 7](../figures/step05_coverage.png)

**Figure 7.** Empirical against nominal interval coverage for the raw ensemble and after conformal calibration, under leave-one-soil-out and leave-one-mission-out.

### 3.6. FeO from M³ Spectra at Returned-Sample Sites

The base method is assessed for its performance for its assigned task of FeO in orbits employing its ground truth data [@fernandez2026; @qiu2025], according to two different schemes of tests, P1 and P2 (Figure 4), where P1 represents the internal 10-fold cross validation within each site of the base study, which is consistent with the MAE of 2.29 wt% reported; and P2 represents leave-one-landing site out test in which all sites from one mission is left out because of common terrain and spatial autocorrelation.

**Table 6.** The FeO (wt%) data was taken from 49 returned sample sites using both P1 (10-fold cross-validation with seed 42) and P2 (leave one landing site out) approaches based on spectra of the 3 × 3-pixel site. The bias here means the discrepancy between the predicted FeO and laboratory FeO. All models except the zero-shot model and constants by Lucey et al. were trained for each fold.

| Method | P1 MAE | P1 RMSE | P1 R² | P1 bias | P2 MAE | P2 RMSE | P2 R² | P2 bias |
|---|---|---|---|---|---|---|---|---|
| LUMEN-M3 (zero-shot) | 6.81 | 7.52 | −1.30 | −5.29 | 6.81 | 7.52 | −1.30 | −5.29 |
| LUMEN-M3 + recalibration | 4.39 | 5.17 | −0.09 | −0.04 | 5.11 | 6.34 | −0.63 | 0.17 |
| **LUMEN-M3-FT** | 2.20 | 3.01 | 0.63 | −1.19 | 3.27 | 4.02 | 0.34 | −1.01 |
| LUMEN-embed + ridge | 2.01 | 2.56 | 0.73 | −0.60 | 2.60 | 3.16 | 0.59 | −0.38 |
| FD-AE-M3 / RF (published) | 1.90 | 2.71 | 0.70 | 0.01 | 3.58 | 4.21 | 0.28 | 1.29 |
| FD-AE-M3 / Lasso | 2.11 | 2.71 | 0.70 | 0.01 | 3.07 | 3.78 | 0.42 | 0.78 |
| FD-AE-M3 / SVR-lin | 2.30 | 2.95 | 0.65 | −0.37 | 2.27 | 2.86 | 0.67 | −0.44 |
| FD-AE-M3 / SVR-RBF | 2.12 | 2.85 | 0.67 | −0.02 | 3.27 | 3.86 | 0.40 | 0.56 |
| FD-wavelet / RF | 1.96 | 2.72 | 0.70 | −0.32 | 3.21 | 3.88 | 0.39 | 0.72 |
| FD-wavelet / Lasso | 2.19 | 2.81 | 0.68 | −0.24 | 2.44 | 3.49 | 0.50 | 0.20 |
| FD-wavelet / SVR-lin | 2.49 | 3.16 | 0.59 | −0.61 | 2.54 | 3.41 | 0.53 | −0.02 |
| FD-wavelet / SVR-RBF | 2.40 | 3.39 | 0.53 | −0.03 | 4.39 | 5.94 | −0.44 | 1.31 |
| Two M³ bands (R950/R750, R750) | 1.99 | 2.43 | 0.76 | 0.00 | 2.11 | 2.54 | 0.74 | −0.16 |
| Lucey θ (M³, refitted) | 3.26 | 3.70 | 0.44 | 0.00 | 4.30 | 5.04 | −0.03 | −0.03 |
| Lucey et al. constants (Clementine) ᵃ | 1.25 | 1.72 | 0.88 | −0.17 | 1.25 | 1.72 | 0.88 | −0.17 |
| Two Clementine bands (refitted) | 1.34 | 1.74 | 0.88 | 0.03 | 1.82 | 2.32 | 0.78 | −0.31 |
| Training-fold mean | 4.34 | 5.07 | −0.04 | 0.00 | 4.78 | 5.69 | −0.32 | 0.03 |

ᵃ In-sample. The published constants [@lucey2000] were derived from returned-sample sites, and nothing is fitted here, so this row is not a held-out estimate and is not comparable with the others.

In the primary assessment (P2), LUMEN-M3-FT shows a smaller MAE than the published FD (3.27 vs 3.58 wt%), but it is more accurate than the latter only at 24 out of 49 locations, and the result does not reach statistical significance (Wilcoxon p = 0.74; Holm p = 1.00 for 13 comparisons). Statistical superiority is shown only against the training-fold mean (Holm p < 0.001), and it is statistically inferior to the two M³ bands (Holm p = 0.009), FD-AE-M3 / SVR-lin (Holm p = 0.029), the two Clementine bands (Holm p = 0.001), and the in-sample Lucey et al. constants (Holm p < 0.001). According to P1, the published FD value (1.90 wt%) correlates well with the reproduced results of the original paper (1.95 wt%) and is lower than its previously reported 2.29 wt% [@fernandez2026]. The LUMEN-M3-FT value (2.20 wt%) significantly differs from no-FD values and the two M³ bands, is significantly less accurate than the two Clementine bands (Holm p = 0.020), and at the same time, is significantly more accurate than the refitted Lucey θ on M³ (Holm p = 0.020).

For the six P1 seeds LUMEN-M3-FT gives an average of 2.13 wt% (2.08 to 2.20) and FD as published is 1.91 wt% (1.81 to 1.97) with FD being lower in every seed and with each strip's centre pixel rather than the 3 × 3 block. LUMEN-M3-FT has 3.26 wt% (vs 3.35 wt% for the published FD, Holm p = 1.00) and is still significantly worse than the two M³ bands (2.02 wt%, Holm p = 0.006). Under P2 the conformal intervals for LUMEN-M3-FT cover 0.63, 0.92 and 0.98 of the sites at the nominal 68, 90 and 95% (raw ensemble 0.51, 0.76 and 0.86) with mean half widths of 4.6 wt% at 68% and 10.6 wt% at 95%.

![Figure 8](../figures/fig05_m3_sites.png)

**Figure 8.** Predicted against lab FeO at 49 returned-sample sites for P2 (leave one landing site out). The plots show the predictions of LUMEN-M3-FT, FD (as published) and the two M³ band linear model. The symbol indicates the landing site. The red line represents 1:1.

### 3.7. Ablations and Parameter Analysis

The pretraining does improve performance, yet the pre-registered primary model is not the optimal configuration (Table 7). Pretrained A2 has a lower error than the same model trained from scratch (A1) on all 13 targets under LOSO, though, due to the correlation of the targets, none of the differences are statistically significant after Holm correction; under LOMO, two are (HCP and maturity). The SSA-mixup and synthetic replay decrease the error of the pretrained model (A4 vs A2) on only one out of 13 targets, without any statistically significant difference, and help a model trained from scratch under LOSO (12 of 13 targets), but not under LOMO (2 of 13). The zero-shot A0, the physics-only equivalent of a simulator-trained inversion, has an MAE of 9.1 vol%.

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

Even without physics, however, the network provides superior performance to classical regression methods in 5 of 13 targets for LOSO (PLG, ILM, FeO, TiO₂, and Al₂O₃) and 3 for LOMO, and in 8 and 6 targets with the pretrained network A2; this is why the network selection is shifted inside the cross-validation procedure (Section 3.3).

There are four design alternatives which were pre-trained and fine-tuned using the A4 recipe, each with two members, relative to members 0–1 of the baseline network (Table 8; FeO 1.32 wt% under LOSO and 2.62 wt% under LOMO). The log reflectance feature on its own already achieves non-significantly different performance compared to the other designs, suggesting that the continuum-removed channel does not add any significant advantage; similarly on other side, the continuum-removed channel on its own achieves significantly lower accuracy on six targets under LOSO and eight targets under LOMO (FeO 2.09 and 3.86 wt%, respectively). The multilayer perceptron (MLP) design is significantly inferior to the LCP design under LOSO, but significantly superior to it on four targets under LOMO (PLG, FeO, Al₂O₃, and CaO; FeO 2.30 wt%).

**Table 8.** Alternative designs compared with the main network (alternative: two members, target: thirteen sites, Holm adjustment performed on targets): "Lower MAE": The number of targets in which the MAE of the variant is lower than the one from the main network per soil. "Better / Worse": The number of targets where a Holm significant difference in favor of or against the variant exists.

| Variant | Parameters | LOSO: lower MAE | LOSO: better / worse | LOMO: lower MAE | LOMO: better / worse |
|---|---|---|---|---|---|
| Main network (log R + continuum-removed, CNN) | 0.36 M | — | — | — | — |
| Log R only | 0.36 M | 10 / 13 | 0 / 0 | 5 / 13 | 0 / 0 |
| Continuum-removed only | 0.36 M | 0 / 13 | 0 / 6 | 1 / 13 | 0 / 8 |
| MLP | 0.07 M | 0 / 13 | 0 / 1 | 9 / 13 | 4 / 0 |
| 1-D ResNet | 0.42 M | 7 / 13 | 0 / 0 | 8 / 13 | 0 / 0 |

### 3.8. Application to M³ Scenes

At the Apollo 16 landing-site pixel, the published calibration (eqn. 10) reproduces the LSCC 62231 spectrum to 9.1% rms and the lines re-derived from this pixel to 1.4%; LUMEN estimates 5.0 wt% FeO here (62231: 4.9 wt%; Apollo 16 mean: 4.6 wt%) and over-estimates TiO₂ (2.1 vs 0.6 wt%), only a sanity check, since 62231 is in the final training set. In the Shapley scene (Figure 9) the 11 identifiable targets show the expected contrasts (FeO 10.2 wt% on mare and 6.6 wt% on highland; Al₂O₃ 20.0 and 24.3 wt%) and fresh craters stand out as immature. The published high-Ca pyroxene map of the same strip [@korokhin2024] is anti-correlated with LUMEN's (r = −0.79), placing more clinopyroxene in the highlands than in the mare, and its maturity parameter is nearly unrelated to LUMEN's Is/FeO (r = −0.28). Destriping lowers stripe amplitude by 26–47% and leaves the maps correlated at r = 0.90–0.99 with the undestriped ones. At Aristarchus, LUMEN detects much less olivine and more plagioclase within the crystalline fraction than the posterior values of Jung et al. [@jung2026] (52% vs 29% plagioclase in region B); because the LUMEN olivine fraction does not pass the identifiability gate, this is a domain-shift warning, not a refutation of that paper.

![Figure 9](../figures/step07_shapley_maps.png)

**Figure 9.** LUMEN maps of the Shapley-region M³ strip for identifiable targets, with conformal interval half-widths. Olivine and "other" are withheld by the identifiability gate.
