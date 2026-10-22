## 5. Results

### 5.1 Simulator

All self-tests pass.

- **k round trip.** Every end-member's k(λ) reproduces its measured reflectance to
  {rt_max:.1e}. Before the falling-branch fix, {k_issue_bands} bands of the synthetic L19
  glass were clamped at k = 1.
- **Prior.** Truncating the composition prior lowers the 99th-percentile synthetic TiO₂ from
  {tio2_p99_v0:.1f} to {tio2_p99_v1:.1f} wt % (the LSCC maximum is {tio2_lscc_max:.1f}) and
  FeO from {feo_p99_v0:.1f} to {feo_p99_v1:.1f} wt %.
- **Label model.** Modal composition × class-average phase chemistry reproduces the LSCC
  bulk oxides with r = {lab_r_min:.2f}–{lab_r_max:.2f}. It is biased for mare TiO₂
  (+{lab_mare_tio2_bias:.1f} wt %), which affects only the scale of the pretraining labels.
- **Coverage of the real data.** The 75 LSCC spectra lie inside the synthetic cloud: their
  median nearest-neighbour distance is {cov_real:.3f}, against {cov_syn:.3f} between
  synthetic spectra.
- **Sim-to-real gap.** A classifier still separates real from synthetic spectra
  (AUC {auc_rf:.2f}; {auc_rf_nf:.2f} against noise-free synthetic spectra). On average, real
  soils are redder than their nearest synthetic neighbours.
- **Mixing test.** The measured <45 µm spectra are reproduced by linear mixtures of their
  sieve fractions to {mix_oracle:.1f} % RMS. The bulk chemistry, however, could not
  discriminate between the mixing rules (Section 6).

### 5.2 Accuracy on the LSCC soils

Table 1 gives per-soil MAEs for the main methods; every method and target is in `RESULTS.md`.

**Leave-one-soil-out.**

- Against the fraction-matched mean of the other soils, the primary model (A4) has skill
  {skill_comp_loso:.2f} for composition and {skill_feo_loso:.2f} for FeO.
- Against each classic baseline separately, A4 has the lower error on {n_better_rf_loso}/13
  targets; the exception is olivine every time. Holm-corrected,
  it is significantly better than PLS on {n_sig_pls_loso} targets and than the mean of the
  other soils on {n_sig_mo_loso}.
- **Against the strongest baseline, a random forest on the same features, only maturity is
  significant after correction** (FeO: raw p = {p_feo_rf_loso:.3f}, Holm {holm_feo_rf_loso:.3f}).

**Leave-one-mission-out.**

- Every method degrades. A4's composition MAE rises from {comp_a4_loso:.2f} to
  {comp_a4_lomo:.2f} vol %, and its FeO MAE from {feo_a4_loso:.2f} to {feo_a4_lomo:.2f} wt %.
- A4 remains lower than the random forest on {n_better_rf_lomo}/13 targets, but loses on
  olivine, "other" and MgO.
- For agglutinate and "other", A4 is not significantly better than the mean of the other
  soils, and for "other" no method has a lower error than that zero-information baseline.

{table_main}

### 5.3 Ablations

**The pre-registered primary model is not the best configuration** (Table 2).

- Pretraining on the synthetic set helps: A2 beats real-only training from scratch
  (A1) on {n_a2_vs_a1_loso}/13 targets under leave-one-soil-out.
- The two other ingredients — SSA-mixup and synthetic replay — did not help:
  - adding them to the pretrained model (A4) lowered the error on only
    {n_a4_vs_a2_loso}/13 targets relative to A2;
  - none of the A4–A2 differences is significant after Holm correction.
- Mixup does help a model trained from scratch under leave-one-soil-out (A1m vs A1, lower on
  {n_a1m_vs_a1_loso}/13), but not under leave-one-mission-out ({n_a1m_vs_a1_lomo}/13). It
  partly substitutes for pretraining rather than adding to it.
- The zero-shot model (A0), the physics-only analogue of Korokhin's network, has a
  composition MAE of {comp_a0:.1f} vol %, far above every fine-tuned model. This is the
  sim-to-real gap made concrete.

{table_ablation}

### 5.4 Head-to-head with the three published methods

**Korokhin et al. (2024).** Table 3 scores each method on the 5-mineral proportions, exactly
as their reproduction scores itself.

- Under leave-one-soil-out, A4 reduces the error for every mineral (Holm-significant on
  {kor_sig_loso}), e.g. olivine {kor_olv_a4:.2f} vs {kor_olv_k:.2f} and plagioclase
  {kor_plg_a4:.2f} vs {kor_plg_k:.2f} percentage points.
- Under leave-one-mission-out, A4 is still lower on {kor_n_lomo}/5 but significant only for
  {kor_sig_lomo}, and it is slightly worse for plagioclase ({kor_plg_a4_lomo:.2f} vs
  {kor_plg_k:.2f}).
- **Caveat.** Their inversion never sees an LSCC soil, while A4 is fine-tuned on the others.
  The like-for-like physics-only comparison is the zero-shot A0. It beats their fit on
  {kor_a0_n}/5 minerals but is significant only for {kor_a0_sig}.
- **Maturity.** LUMEN's Is/FeO ranks the 19 soils better than their nFe parameter: Spearman
  ρ = {kor_rho_l} vs {kor_rho_k} for the <10 / 10–20 / 20–45 µm fractions.

{table_korokhin}

**Jung et al. (2026).** On their four test soils at 20–45 µm:

- A4's worst-case crystalline errors are {jung_olv:.2f} (olivine), {jung_ilm:.2f} (ilmenite)
  and {jung_plg:.2f} (plagioclase) percentage points under leave-one-soil-out. All three of
  their accuracy claims are met; their converged inversion has {jung_ilm_j:.2f} for ilmenite.
- With n = 4 this is weak evidence. The zero-shot A0 also meets every claim, while the mean
  of the other soils fails the ilmenite and plagioclase claims.
- **Maturity.** LUMEN's Is/FeO correlates with the measured value at r = {jung_r_l:.2f}
  across the four soils, against {jung_r_j:.2f} for their I_sm.

**Fernández-Díaz et al. (2026).** Their full pipeline is re-implemented on the same spectra
and folds (Table 4).

- Under leave-one-soil-out, A4's FeO MAE is {fd_a4_loso:.2f} wt %. The published
  configuration (M³-trained autoencoder → random forest) has {fd_pub_loso:.2f} wt %; their
  best variant on our data, wavelets → SVR-RBF without the autoencoder, has
  {fd_best_loso:.2f} wt %.
- A4 is lower than all twelve variants (raw p < 0.05 for each), but only {fd_holm_loso}/12
  comparisons survive Holm correction over the variants.
- Under leave-one-mission-out, A4 has {fd_a4_lomo:.2f} wt % against {fd_pub_lomo:.2f}
  (published configuration) and {fd_best_lomo:.2f} (best variant, p = {fd_best_lomo_p:.2f}).
- **The autoencoder's effect depends on the protocol.**
  - Under leave-one-soil-out it mostly loses information, as the original reproduction found:
    the M³-trained autoencoder is worse than the raw wavelet coefficients for
    {fd_ae_worse_loso}/4 regressors.
  - Under leave-one-mission-out it helps the two linear regressors, and is worse than the raw
    coefficients for {fd_ae_worse_lomo}/4.

{table_fd}

### 5.5 Uncertainty and identifiability

**The raw ensemble is overconfident.**

- Mean empirical coverage is {cov_raw_loso} under leave-one-soil-out and {cov_raw_lomo}
  under leave-one-mission-out, for nominal 68 / 90 / 95 %.
- Conformal calibration with normalised residuals restores nominal coverage under
  leave-one-soil-out ({cov_conf_loso}) and comes close under leave-one-mission-out
  ({cov_conf_lomo}).

**The identifiability gate passes {n_ident}/13 targets.** It withholds olivine and the
"other" class: their skill confidence intervals include zero, and olivine's point estimate
is negative. Under leave-one-mission-out the agglutinate fraction's skill also loses
significance.

### 5.6 Application to M³

**Calibration.** On the Apollo 16 landing-site pixel, Korokhin's published lab-scale
calibration reproduces the LSCC 62231 spectrum to {cal_eq2:.1f} % RMS, against
{cal_ours:.1f} % for the re-derived line.

**Apollo 16 site.** LUMEN estimates FeO {a16_feo:.1f} wt %, against {a16_feo_meas:.1f}
measured for 62231 and {a16_feo_mean:.1f} for the Apollo 16 mean. It overestimates TiO₂
({a16_tio2:.1f} vs {a16_tio2_meas:.1f}). Because 62231 is in the final training set, this is
a sanity check, not a validation.

**Shapley scene.** The maps of the {n_ident} identifiable targets show the expected
mare/highland contrasts: FeO {sh_feo_m:.1f} vs {sh_feo_h:.1f} wt %, Al₂O₃ {sh_al_m:.1f} vs
{sh_al_h:.1f} wt %. Fresh craters stand out as immature.

- Korokhin's high-Ca pyroxene map is anti-correlated with LUMEN's
  (r = {kmap_cpx:.2f}), and theirs places more clinopyroxene in the highlands than in the
  mare.
- Their maturity parameter nFe and LUMEN's Is/FeO are nearly unrelated across the scene
  (r = {kmap_nfe:.2f}).

**Aristarchus ROIs.**

- LUMEN finds much less olivine and much more plagioclase in the crystalline fraction than
  Jung's posteriors (plagioclase {ar_plg_b:.0f} % vs {ar_plg_b_j:.0f} % in ROI B).
- LUMEN's olivine output failed the identifiability gate, and the training soils contain
  little olivine. The disagreement is therefore a domain-shift warning rather than evidence
  against Jung.
