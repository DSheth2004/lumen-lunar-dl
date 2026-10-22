## 2. Data

**Laboratory spectra and ground truth.** We use the Lunar Soil Characterization Consortium
(LSCC) suite: 9 mare soils (Taylor et al., 2001) and 10 highland soils (Taylor et al.,
2010).

- Each soil was sieved into <10, 10–20, 20–45 and <45 µm fractions and measured at RELAB
  (i = 30°, e = 0°).
- The 70181 10–20 µm spectrum is absent, which leaves 75 spectra.
- Ground truth:
  - modal mineralogy (point counts, vol %) for the three finer fractions (56 spectra), merged
    into seven parts: plagioclase (PLG), low-Ca pyroxene (LCP = orthopyroxene + pigeonite),
    high-Ca pyroxene (HCP), olivine (OLV), ilmenite (ILM), agglutinitic glass (AGG), and other
    (volcanic glass, K-phases, other);
  - bulk FeO, TiO₂, Al₂O₃, CaO and MgO;
  - the ferromagnetic-resonance maturity index Is/FeO for all 75 spectra.
- Every spectrum is convolved to the 71 Moon Mineralogy Mapper (M³) global-mode bands
  between 540.8 and 2497.1 nm, with Gaussian responses of FWHM equal to the band spacing.

**End-member library.** We use 49 RELAB spectra of lunar mineral separates and glasses (17
plagioclase, 7 low-Ca and 11 high-Ca pyroxene, 6 olivine, 4 ilmenite, 2 volcanic and 2
synthetic lunar glasses).

- They come from mare basalts, highland rocks, a non-LSCC soil (62241) and synthetic glasses.
- No end-member derives from any of the 19 evaluation soils. Every RELAB agglutinate separate
  comes from an LSCC soil, so agglutinates are modelled physically.

**M³ scenes.**

- The Shapley-region strip M3G20090728T214215 analysed by Korokhin et al. (2024), lines
  13918–14805.
- An Apollo 16 block, M3G20090607T110414.
- The two Aristarchus ROI blocks of Jung et al. (2026).

All are L2 reflectance.

## 3. Method

### 3.1 Physics simulator

- **Grain optics.** Each end-member's imaginary index k(λ) is obtained by inverting the Hapke
  (1993) equivalent-slab model at the separate's nominal grain size.
  - R(k) is non-monotonic: it falls to an opacity floor, then rises as surface reflection grows
    with k. The root is therefore bracketed on the physical, falling branch.
  - This removes the k = 1 clamping we found for a dark synthetic glass. Every end-member now
    reproduces its own reflectance to < 1e-10.
- **Space weathering.** Nanophase iron adds α = 36π z(λ) φ / λ (Hapke 2001; Lucey & Riner
  2011). Microphase iron is a grey absorber. Agglutinates are a glass host (a mix of a
  feldspathic and a mafic synthetic glass) loaded with npFe0 and mpFe0.
- **Synthetic soils.** Each synthetic soil draws:
  - a petrological regime (feldspathic / basaltic / broad Dirichlet priors on the crystalline
    proportions, truncated at ≤ 20 % ilmenite and ≤ 60 % olivine of the crystalline part);
  - a maturity that sets the agglutinate fraction and the npFe0 loads;
  - grain sizes from 4 to 150 µm;
  - one end-member per phase;
  - a calibration tilt, ±7 % scale and 0.2–0.8 % noise.
- **Mixing and labels.**
  - Components mix intimately in single-scattering albedo with cross-section weights vol/D.
  - Chemistry labels follow from stoichiometry: nominal phase chemistry with a molar Fe ↔ Mg
    exchange jitter, and the agglutinate carrying the bulk composition of the non-agglutinate
    part.
  - Is/FeO = 2270 · npFe0 / FeO sets only the scale of the pretraining target.
- **Set size.** We generate 400,000 spectra.

### 3.2 Network

- **Input.** Two channels × 71 bands: log reflectance and convex-hull continuum-removed
  reflectance, each standardised per band with *synthetic* statistics. No real-data statistic
  enters the input.
- **Body.** Five 1-D convolutions with GroupNorm and GELU, two max-pools, and a 256 → 128
  trunk (0.36 M parameters).
- **Heads.**
  - A Dirichlet head (concentrations α) for the 7-part composition. Its loss is the Dirichlet
    NLL plus an L1 term on the mean, whose weight was fixed on synthetic data.
  - Heteroscedastic Gaussian heads trained with β-NLL (β = 0.5; Seitzer et al., 2022) for the
    five oxides and log₁₀(Is/FeO).
  - Auxiliary grain-size and npFe0 heads, used during pretraining only.

### 3.3 Training

- **Pretraining.** Five ensemble members are pretrained on the synthetic set (15 epochs,
  AdamW, one-cycle).
- **Fine-tuning.** Each member is fine-tuned per fold for 1500 steps. Each step draws:
  - 16 real training spectra;
  - 32 **SSA-mixups**: 2–3 training spectra, mixed in single-scattering albedo with
    cross-section weights from their nominal fraction sizes, with labels mixed exactly
    (composition by volume, chemistry by mass, Is/FeO weighted by mass × FeO);
  - 16 synthetic replay spectra.

  Real and mixed spectra receive the simulator's calibration / noise randomisation.
- **Pre-registration.** The schedule was pre-registered and never tuned on held-out data.

### 3.4 Uncertainty and identifiability

- **Ensemble moments.** The mean of the member means. The variance is the mean of the member
  variances plus the variance of the member means.
- **Conformal intervals.** Normalised scores |y − μ|/σ from the out-of-fold predictions of
  the other soils (or missions) give conformal half-widths q·σ. This is a CV+-style
  construction; the calibration residuals come from fold models that included the test soil.
- **Identifiability gate.** A target's leave-one-soil-out skill, 1 − MAE / MAE(fraction-matched
  mean of the other soils), carries a bootstrap CI. Only targets whose CI excludes zero are
  mapped.

## 4. Evaluation protocol

- **Splits.** Leave-one-soil-out (19 folds) and leave-one-mission-out (6 folds: A11, A12, A14,
  A15, A16, A17).
  - Every fraction of the held-out soil or mission is excluded from fine-tuning, mixup and
    inner model selection.
  - Leave-one-mission-out is the out-of-distribution test: holding out A16 removes every
    feldspathic soil, and holding out A17 every high-Ti soil.
- **Metrics.** Per-soil MAE (each soil's labelled fractions averaged, then the 19 soils
  averaged). Paired Wilcoxon signed-rank tests over soils, Holm-corrected over the 13 targets.
- **Baselines** (same folds, hyperparameters by inner group CV on training soils):
  - the fraction-matched mean of the other soils (zero-information reference);
  - PLS, ridge and random forest on the same features;
  - a random forest on raw reflectance;
  - the full Fernández-Díaz et al. (2026) pipeline re-implemented on the same spectra:
    db4 A2 + D2 wavelets, a 6-D autoencoder trained on M³ pixels or on synthetic spectra, and
    four regressors with their search grids; plus its no-autoencoder variant.
- **Comparisons with Korokhin et al. (2024) and Jung et al. (2026).** These use the published
  methods' own outputs on the same soils, from independent reproductions, scored exactly as
  each paper scores itself.
- **Ablations.** Zero-shot (synthetic only), real-only from scratch, scratch + mixup,
  pretrain + real, pretrain + mixup, and the full model. All are reported.
