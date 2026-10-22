# LUMEN — hand-off for a fresh session

This file lets a new session (e.g. a Claude Code cloud session) resume the work with no
other context. Read it top to bottom before touching code.

## 1. The task (from the user, Dhrumil Sheth)

> Come up with a deep-learning approach we can write a paper on. Implement a proof of concept
> and validate it. Goal: **lunar mineral property identification**. Three papers are the
> reference: Fernández-Díaz et al. 2026, Jung et al. 2026, Korokhin et al. 2024. Propose a
> DL approach, implement it, and try to **prove it is better than these papers**.

"Better" must be shown **on identical data under identical, honest protocols**, with
paired statistics, and reported plainly where it is *not* better. Hyperparameters are
fixed before the final evaluation; nothing is tuned on held-out folds.

## 2. The three papers and where they fail

All three were reproduced from scratch in the parent repository (not available in a fresh
clone). The numbers below come from those reproductions; the outputs needed for the
head-to-head comparisons are staged in `data/baselines/`.

**Jung et al. (2026), PSJ 7(1) 18** — Hapke radiative transfer + CATMIP MCMC; 9 end-members.
They test on 4 LSCC mare soils (12001, 12030, 15071, 71061; the 20–45 µm fraction) and
apply the method to M³ Aristarchus.
- Accuracy is reported on crystalline proportions (OLV, PLG, PYX, ILM).
- Worst-case errors with the converged sampler: olivine 5.78 (claimed bound 7), **ilmenite 17.56** (bound 10;
  22.13 if 15071 is included), plagioclase 16.83 (bound 30).
- The prior alone, with no data, gives olivine 1.76, ilmenite 10.87 and plagioclase 16.92. Two of the three claims
  are therefore met without looking at a spectrum.
- The released sampler degenerates: 0–1 % unique particles, so its credible intervals are not posteriors.
- The maturity index I_sm disagrees with measured Is/FeO by up to 34.6 units and ranks the soils in reverse order;
  r = +0.15 over the 6 mare soils.
- Files: `data/baselines/jung2026/step05_vs_ground_truth.csv` (per-soil errors, `abs_pct_diff`), `step07_*`.

**Korokhin et al. (2024), PSS 244 105870** — Shkuratov model with 8 parameters (S, nFe, gray,
CPX, OPX, OLV, PLG, ILM). It is inverted two ways: Nelder–Mead, and an MLP 21-42-12-8 trained on synthetic spectra.
- On synthetic spectra the network recovers OPX at R² 0.06, OLV at 0.17 and gray at 0.32. The maps of these three
  follow the training ranges, not the spectra.
- The "OPX" end-member is actually a pigeonite.
- On the 19 LSCC soils, scored as 5-mineral proportions (each soil's fractions averaged, n = 19 soils), MAE in
  percentage points, Nelder–Mead fit vs the mean-of-other-soils baseline:
  - CPX: 5.30 vs 6.78
  - OPX: 8.37 vs 9.18
  - **OLV: 7.44 vs 2.21**
  - PLG: 12.08 vs 18.44
  - ILM: 5.24 vs 6.26
- Within each set the picture is worse:
  - mare: only OPX and ILM beat the baseline;
  - highland: only OPX and PLG beat it;
  - olivine is over-predicted by about 10 points.
- nFe vs Is/FeO, Spearman per fraction (<10 / 10–20 / 20–45): mare 0.57 / 0.67 / 0.85, highland 0.81 / 0.70 / 0.56.
  Pooled across both sets it drops to 0.37–0.55.
- Files: `data/baselines/korokhin2024/step10_lscc_fits.csv` (`prop_fit_*`, `prop_true_*` per soil × fraction),
  `step10_lscc_summary*.csv`, `step08_c1_recovery.csv`, `ann_full_median_iter3.npy` (scene maps), `highland_mask.npy`.

**Fernández-Díaz et al. (2026), Mathematics 14 254** — db4 wavelet (A2+D2 coefficients, 42 values) → autoencoder (6-D latent) →
Random Forest, predicting FeO only.
- Ground truth is 50 samples (`data/ground_truth/fernandez2026_ground_truth_50.csv`) but only about **15 distinct
  coordinates**; for example, 8 Apollo 12 samples share one pixel. The evaluation is therefore pseudo-replicated.
- Published: 10-fold CV MAE 2.29 ± 0.55 wt%, R² 0.706. The headline MAE of 1.204 is a **5-sample** hold-out.
- Reproduction findings:
  - feeding the 42 wavelet coefficients straight to the regressor gives MAE 1.595;
  - the full pipeline gives 1.986, so **the autoencoder loses information**;
  - two M³ bands and a fitted line give 1.685.
- The paper's Table 1 is mathematically impossible.

**Common gaps → design goals.**
- (G1) Sim-to-real gap: physics-only inversions are non-unique and mis-calibrated.
- (G2) Identifiability is never tested.
- (G3) No calibrated uncertainty.
- (G4) Weak validation: tiny hold-outs, pseudo-replication, no zero-information baseline.

## 3. The proposed method — LUMEN (LUnar Mineralogy Estimation Network)

A hybrid physics + deep-learning framework:

1. **Physics simulator** (`scripts/physics.py`):
   - Hapke equivalent-slab model, adapted from the Jung reproduction's self-tested core.
   - End-members are 49 real lunar mineral separates and glasses (`lumen_common.ENDMEMBER_LIBRARY`).
     **None is derived from the 19 evaluation soils** (leakage rule D1, checked in step00). All RELAB agglutinate
     separates come from LSCC soils, so agglutinates are modelled physically instead: a glass host plus npFe0 plus
     mpFe0.
   - Space weathering: npFe0 absorption α = 36π z(λ) φ / λ (Hapke 2001; Lucey & Riner 2011). mpFe0 is a grey
     absorber.
   - Domain randomisation over end-member choice, grain size (4–150 µm), maturity, spectral tilt, scale (±7 %) and
     noise.
   - Labels follow from stoichiometry. The agglutinate carries the bulk composition of the non-agglutinate part.
     Is/FeO = 2270 · npFe0 / FeO sets only the scale of the pretraining target (D4).
2. **Network**:
   - Compact 1-D CNN (GroupNorm, not BatchNorm, because fine-tuning batches are small).
   - Input: 2 channels × 71 M³ bands (540.84–2497.11 nm) — standardised log-reflectance and convex-hull
     continuum-removed reflectance.
   - **Dirichlet head** for the 7-part modal composition (PLG, LCP, HCP, OLV, ILM, AGG, OTH; vol %).
   - Heteroscedastic Gaussian heads (β-NLL) for FeO, TiO₂, Al₂O₃, CaO, MgO (wt %) and log₁₀(Is/FeO).
   - Auxiliary pretraining-only heads: log grain size and npFe0.
3. **Training**:
   - (a) Pretrain on about 400k synthetic spectra.
   - (b) Fine-tune per fold on the real training soils plus **SSA-space mixup**:
     - random Dirichlet-weighted mixtures of 2–3 training spectra, mixed in single-scattering albedo with
       cross-section weights vol/D (nominal fraction sizes 5 / 15 / 32.5 / 17 µm);
     - labels mix exactly: composition by volume, chemistry by mass,
       Is/FeO_mix = Σ mᵢ(Is/FeO)ᵢFeOᵢ / Σ mᵢFeOᵢ;
     - plus about 25 % synthetic replay.
   - Fixed schedule, pre-registered in `METHOD_NOTES.md` before the final run.
4. **Uncertainty**: a 5-member deep ensemble, then CV+ (jackknife+) conformal intervals. Report empirical coverage
   at 68 / 90 / 95 %.
5. **Identifiability gate**: per target, skill = 1 − MAE / MAE(mean-of-others), with a bootstrap CI over soils.
   Targets without significant skill are flagged and **not mapped**. This avoids Korokhin's failure.

## 4. Validation protocol (pre-registered)

- Data: `products/lab_dataset.npz` + `tables/step01_lab_dataset.csv`.
  - 75 spectra from 19 LSCC soils × 4 fractions. The 70181 10–20 µm spectrum is absent from the source.
  - 56 spectra have modal mineralogy (the <45 µm fraction has none). All 75 have chemistry and Is/FeO.
- **LOSO**: leave one soil out, 19 folds. All fractions of the held-out soil are removed from training, from mixup and
  from normalisation statistics.
- **LOMO**: leave one mission out, 6 folds — A11 (1 soil), A12 (2), A14 (4), A15 (2), A16 (6), A17 (4). This is the
  out-of-distribution test. Holding out A16 removes every feldspathic soil; holding out A17 removes every high-Ti soil.
- Metrics: per soil (the mean over its fractions, n = 19) and per spectrum. Paired Wilcoxon signed-rank tests over
  soils against each baseline. Bootstrap 95 % CIs.
- **Head-to-head definitions**, each matching how that paper scores itself:
  - *Korokhin*: 5-mineral proportions (HCP↔CPX, LCP↔"OPX", OLV, PLG, ILM) / sum × 100, over the <10, 10–20 and
    20–45 fractions; MAE per mineral against the mean-of-others baseline. Report pooled and by set. Also Spearman of
    the maturity output vs Is/FeO per fraction.
  - *Jung*: crystalline proportions (OLV, PLG, PYX = LCP + HCP, ILM) for 12001, 12030, 15071 and 71061 at 20–45 µm.
    Test against their three accuracy claims and the prior-only control.
  - *Fernández-Díaz*: FeO MAE / RMSE / R² under LOSO and LOMO. Their pipeline must be re-implemented on the same
    71-band spectra:
    - 2-level db4 (periodization), keep A2 + D2;
    - AE in→32→16→6→16→32→in (LeakyReLU 0.01, BN, dropout 0.3, Adam 1e-3, weight decay 1e-5, batch 256, early stop
      patience 12), trained unsupervised on M³ Shapley pixels and, as a variant, on the synthetic set;
    - Lasso / SVR-linear / SVR-RBF / RF with a grid search inside an inner CV;
    - plus the no-AE variant (wavelet → RF).
- Classic baselines on the same folds: mean-of-others, PLS, ridge, RF on raw bands.
- **Ablations**: synthetic-only (zero-shot, the Korokhin-like physics-only DL); real-only from scratch; + pretrain;
  + mixup; full. Report **all** of them, not only the best.

## 5. Status at hand-off

| stage | file | state |
|---|---|---|
| shared defs, I/O, M³ grid, targets, end-member registry | `scripts/lumen_common.py` | done |
| stage inputs + SHA-256 manifest + identity checks | `scripts/step00_stage_data.py` | **done and run**: 49/49 end-members verified from PDS4 labels; shuffled control rejected; leakage rule passes; 157 files, 120 MB. In a fresh clone it only verifies `data/` |
| labelled lab dataset | `scripts/step01_lab_dataset.py` | **done and run**: 75 spectra, 19 soils, 56 with composition |
| physics + simulator | `scripts/physics.py` | **written; smoke-tested only** — library builds in 1.3 s, 20k spectra in 0.9 s, R finite (0.015–0.93, median 0.15), compositions close to 100 %. Two open issues below |

**Open issues found by the smoke test (fix in step02):**
1. The k round-trip maximum error is **1.3e-2**, not < 1e-6. Some bands sit at the k bound (k reaches 1.0). This is
   probably an opaque ilmenite spectrum darker than the slab model's floor at its d_eff, or a separate brighter than
   a lossless grain.
   - Find which spectra and bands are affected and report them.
   - Fixes: use a larger d_eff for ilmenite, widen the k bracket, or accept the clamp and document it.
2. The composition prior is too ilmenite-rich: synthetic TiO₂ reaches **31 %** at the 99th percentile, while real lunar
   soils are ≤ ~11 %. The "broad" regime's Dirichlet can put > 30 % ILM in the crystalline part.
   - Cap ILM at ≤ 20 vol %, or lower its concentration, and re-check the percentiles.
   - Also check FeO (99th percentile 32 %) and MgO (28 %) against the real lunar range.

## 6. Next steps, in order

1. `scripts/step02_simulator.py`:
   - (a) k round-trip self-test: every end-member reproduces its own reflectance to < 1e-6.
   - (b) Label-model check: LSCC bulk FeO / TiO₂ / Al₂O₃ / CaO / MgO computed from modal composition × class-average
     phase chemistry vs the measured values. This validates the stoichiometry.
   - (c) Coverage of the real data: nearest-synthetic-neighbour distance per LSCC spectrum, and the AUC of a
     real-vs-synthetic classifier, which quantifies the sim-to-real gap.
   - (d) Generate 400k seeded spectra → `products/synthetic.npz`.
   - (e) Validate the SSA-mixup physics on real data: reconstruct each soil's measured <45 µm spectrum from its
     <10 / 10–20 / 20–45 spectra. Mass proportions are solved from the fraction chemistries; mixing uses SSA with
     cross-section weights.
2. `step03_baselines.py`: classic baselines plus the Fernández-Díaz re-implementation, under LOSO and LOMO.
3. `step04_lumen.py`: pretraining, then per-fold fine-tuning with the 5-member ensemble, under LOSO and LOMO, plus the
   ablations. CPU is enough: pretraining takes about 10–20 min, each fine-tune seconds.
4. `step05_uncertainty.py`: ensemble intervals → CV+ conformal → coverage tables.
5. `step06_head_to_head.py`: the three comparisons of §4 with paired tests.
6. `step07_m3_apply.py`: apply to M³ data and calibrate to the lab scale.
   - **Shapley scene**: `data/m3_shapley/scene_rfl.npy`, 888 lines × 85 bands × 304 samples (line-interleaved);
     raw L2 lines 13918–14805 of M3G20090728T214215; `scene_loc.npy` is 2 × 888 × 304.
   - **Calibration to the lab scale**: Korokhin Eq. 2, A_lab = 1.685·A_M3 + 0.0355, or re-derive it from the Apollo 16
     block `apollo16_rfl.npy` (61 × 85 × 304; centre line index 30, sample 177) against LSCC 62231 <45 µm. Their own
     re-derivation gave slope 1.669, intercept 0.016.
   - **Comparison with Korokhin's maps**: `data/baselines/korokhin2024/ann_full_median_iter3.npy` is 152000 × 8 on a
     500-row resampled grid. Parameter order: S, nFe, Gr, CPX, OPX, OLV, PLG, ILM.
   - Aristarchus ROIs (`data/m3_aristarchus/`) vs Jung.
   - Produce maps with uncertainty, masked by the identifiability gate.
7. Write `METHOD_NOTES.md` (D-numbers, see §7), a generated `RESULTS.md` (written by a report script, never by
   hand), `README.md` and figures. Then a paper draft (`PAPER_DRAFT.md`): abstract, method, experiments, results
   tables, limitations.
8. Optional extension: M³ landing-site FeO / TiO₂ validation.
   - The 12–15 sites need PDS HTTP range requests: one M³ L2 line is one 103360-byte record, so lines can be fetched
     individually. The Jung reproduction's `m3_fetch.py` did this.
   - Use leave-one-site-out, compared against the Fernández-Díaz pipeline on its own task.

## 7. Design decisions recorded so far

- **D1 leakage rule**: no simulator end-member derived from an evaluation soil, so agglutinates are modelled
  physically.
- **D2 composition prior**: three regimes (feldspathic / basaltic / broad) as Dirichlet priors on crystalline
  proportions. AGG = clip(0.65·maturity + N(0, 0.12), 0, 0.75); OTH = 0.25·U², rescaled so AGG + OTH ≤ 0.9. Chosen
  from general lunar petrology, **not** from the evaluation soils.
- **D3 phase chemistry**: nominal per end-member from literature values for the parent rock (`physics.NOMINAL_CHEM`),
  with ±15 % jitter per sample.
- **D4**: Is/FeO_sim = 2270 · npFe0 wt% / FeO wt%. This only sets the scale; fine-tuning re-learns it.
- **D5 input window**: M³ bands 3–73. Spectral response is Gaussian with FWHM = band spacing (global-mode binning).

## 8. Conventions (follow them)

- `data/` is checksummed. Every stage calls `lc.verify_manifest()` at the start and `lc.done()` at the end. Never edit
  `data/` by hand; if inputs must change, change `step00` and regenerate the manifest.
- Every stage takes an explicit seed; report the spread over seeds where results depend on them.
- `RESULTS.md` is generated by code. Rejected values are written as NaN, never as placeholders.
- Report failures plainly. A target LUMEN does not win is reported as such.

## 9. Repository layout and running

The GitHub repository (`DSheth2004/lumen-lunar-dl`) is the user's whole `paper-implementations/` folder.
LUMEN is the subfolder `LUMEN_HybridDL/`. Its sibling folders hold the three reproductions:
`Jung2026_SpaceWeatheredMaria/`, `Korokhin2024_ShkuratovANN/`, `FernandezDiaz2026_WaveletAE_FeO/` and
`Copernicus_M3_vs_IIRS/`.
- Their `PAPER_NOTES.md`, `RESULTS.md`, scripts and tables are the detailed source for every baseline number in §2.
  Read them when you need more than this file gives.
- The top-level `README*.md` files summarise all three.

These were left out of the repository because they exceed GitHub's file limit or can be regenerated; they exist only
on the user's machine:
- `FernandezDiaz2026_WaveletAE_FeO/data/raw/` (raw M³ L1B location images);
- the full IIRS Copernicus cube (`Studen1_INPUTdata_IIRS/IIRS_Copernicus_EQC_140m_86bands.tif`);
- `FernandezDiaz2026_WaveletAE_FeO/maps/global_feo_map.tif`;
- `Copernicus_M3_vs_IIRS/products/`.

None of these is needed by LUMEN. The RELAB cart order (`../RELAB_LUNAR_RAW`) is also outside the repository, so
`step00` runs in verify-only mode.

```
cd LUMEN_HybridDL
pip install -r requirements.txt
cd scripts
python step00_stage_data.py     # verifies data/ (fresh clone) or re-stages (original machine)
python step01_lab_dataset.py    # rebuilds products/lab_dataset.npz (~1 s)
# then continue with step02 ... as in section 6
```
