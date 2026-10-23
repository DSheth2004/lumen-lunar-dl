# LUMEN — method notes and pre-registration

This file records every design decision (D-numbers) and fixes the evaluation protocol
**before** any baseline or LUMEN model was trained or scored on the real LSCC folds.
It was committed to git before `step03_baselines.py` and `step04_lumen.py` were run
on real data, so the commit history shows the order: protocol first, numbers second.
Anything changed after that point is listed under **Deviations** at the end, with the reason.

Nothing below was tuned on held-out folds. The only quantities chosen by looking at
data are:
- D2b / D3b (simulator prior), chosen from general lunar petrology;
- D6, which repairs a numerical bracket;
- D9's composition-loss weight, chosen on **synthetic** data only (`scripts/dev_synthetic_only.py`,
  `tables/dev_synthetic_comp_loss.csv`).

The baselines tune their own hyperparameters by inner cross-validation inside each
training fold, which never touches the held-out soil.

## Decisions carried over from the hand-off

- **D1 leakage rule.** No simulator end-member is derived from any of the 19 evaluation
  soils. Agglutinates are modelled physically (glass host + npFe0 + mpFe0). This is
  checked in step00.
- **D2 composition prior.** Three regimes (feldspathic / basaltic / broad) are Dirichlet
  priors on the crystalline proportions.
  - AGG = clip(0.65·maturity + N(0, 0.12), 0, 0.75).
  - OTH = 0.25·U², rescaled so that AGG + OTH ≤ 0.9.
- **D3 phase chemistry.** Nominal values per end-member, from literature values for its
  parent rock (`physics.NOMINAL_CHEM`), jittered per sample.
- **D4 maturity scale.** Is/FeO_sim = 2270 · npFe0 wt% / FeO wt%. This only sets the
  scale of the pretraining target.
- **D5 input window.** M³ bands 3–73 (540.84–2497.11 nm). Band responses are Gaussian
  with FWHM equal to the band spacing.

## Decisions taken in step02 (simulator)

- **D2b truncated prior.** Each regime's Dirichlet is truncated by rejection to
  ILM ≤ 20 % and OLV ≤ 60 % of the crystalline part.
  - Reason: hand-off open issue 2. Synthetic TiO₂ reached 30.5 % at the 99th
    percentile, while returned soils are ≤ ~11 %.
  - After the change, TiO₂ p99 is 14.0 % and FeO p99 is 24.4 % (was 31.8 %).
  - MgO p99 stays at 26 %. That tail is the broad regime's pyroxenite- and
    troctolite-like draws. It is kept as domain randomisation and is documented
    (`tables/step02_prior_percentiles.csv`).
- **D3b chemistry jitter.** Fe–Mg variation is a molar exchange: FeO × e^N(0, 0.15),
  with MgO moved by the molar equivalent and bounded by the phase's Fe + Mg budget.
  TiO₂ × e^N(0, 0.15). Ilmenite keeps its stoichiometric TiO₂; only its Fe ↔ Mg
  exchange varies (MgO 1.5 % × e^N(0, 0.5)).
- **D6 k inversion.** The slab model's R(k) is not monotone: it falls to an opacity floor,
  then rises as surface reflection grows with k. k is therefore solved on the physical,
  falling branch [1e-9, k_floor].
  - Hand-off open issue 1 was caused by the synthetic L19 glass (c1sc50): 13 bands at
    0.54–0.87 µm are darker than R(k = 1). It was not ilmenite.
  - Round trip after the fix: 7.8e-11 for all 49 end-members.
- **D7 mixup rule kept.** Step02(e) found two things:
  - The measured <45 µm spectrum is a linear mixture of the <10 / 10–20 / 20–45 µm
    spectra, to 0.3–1.0 % RMS.
  - The bulk chemistry, however, cannot fix the fraction masses well enough to
    discriminate between mixing rules. The bootstrap sd of the <10 µm mass is 0.11, and
    the χ² of the implied masses does not differ significantly between SSA-cross-section,
    SSA-mass and reflectance-mass (p = 0.37–0.70). Only reflectance-cross-section is
    rejected (16/18 soils, p < 0.001).

  The pre-specified rule (SSA with cross-section weights vol/D) is therefore kept, as
  neither supported nor refuted.

## LUMEN model (step04)

- **D8 input and network.**
  - Input is (2, 71): log R and convex-hull continuum-removed R, each standardised per
    band with the mean and sd of the **synthetic** set. No real-data statistics enter.
  - Body: 1-D CNN with GroupNorm(8) and GELU — conv 2→32 (k7), 32→32 (k5), maxpool 2,
    32→64 (k5), 64→64 (k3), maxpool 2, 64→64 (k3); then flatten → 256 → 128 (dropout 0.1).
    About 0.36 M parameters (355,799).
  - Heads:
    - Dirichlet over the 7-part composition (PLG, LCP, HCP, OLV, ILM, AGG, OTH);
      α = softplus + 0.05, clamped to ≤ 1e4.
    - Gaussian (μ, log σ²) heads for FeO, TiO₂, Al₂O₃, CaO, MgO and log₁₀(Is/FeO).
    - Auxiliary Gaussian heads for log₁₀ D and log₁₀ npFe0, used in pretraining only.
- **D9 losses.**
  - Composition: Dirichlet NLL per part, with labels smoothed by ε = 0.005, plus
    `comp_l1` × 10 × the mean |mean − y| in fraction units, with **comp_l1 = 1**.
    - The L1 term was chosen on synthetic validation data (4.33 vs 4.46 vol % composition
      MAE, consistent over two seeds, no cost on other targets).
  - Scalars: β-NLL with β = 0.5.
  - Targets are scaled by fixed a-priori constants (centre / scale):
    - FeO 10 / 5, TiO₂ 3 / 3, Al₂O₃ 18 / 7, CaO 12 / 3, MgO 9 / 4, log Is/FeO 1.6 / 0.5;
    - auxiliary heads: log D 1.4 / 0.5, log npFe0 −0.7 / 0.6.
  - Missing labels are masked. The auxiliary loss has weight 0.5.
- **D10 pretraining.**
  - Data: the 400k-spectrum synthetic set (seed 1), split 380k train / 20k validation.
  - Optimiser: AdamW (wd 1e-4), OneCycle with max lr 2e-3 and 5 % warm-up; batch 512;
    **15 epochs**; gradient-norm clip 5.
  - **Five members** with seeds 100–104. Each ensemble member has its own pretrained
    weights.
- **D11 fine-tuning** (per fold, per member, member *k* initialised from pretrained
  member *k*):
  - Fixed **1500 steps**, AdamW (wd 1e-4), 100-step linear warm-up then cosine decay to 0.
    The learning rate is 3e-4 from pretrained weights and 1e-3 from scratch.
  - Each step draws 16 real training spectra (uniform, with replacement), 32 SSA-mixups
    and 16 synthetic replay spectra.
  - **SSA-mixup**:
    - k ∈ {2, 3} distinct training spectra (p = ½ each), mass weights ~ Dirichlet(1);
    - spectra mixed in SSA with cross-section weights m/D, where D is the nominal size
      of each source's fraction (5 / 15 / 32.5 / 17 µm);
    - composition mixed by m (equal grain density assumed), chemistry by m,
      Is/FeO_mix = Σ m Is / Σ m FeO;
    - composition labelled only if every source has one.
  - **Replay** supervises composition and chemistry. Its Is/FeO (D4 scale) is masked.
  - Real spectra and mixups receive the simulator's output randomisation:
    tilt N(0, 0.03)·(λ−1.5) + N(0, 0.02)·(λ−1.5)², scale U(0.93, 1.07), noise 0.2–0.8 %.
  - No early stopping, and no validation data are taken from real soils.
- **D12 ablations** (all reported). Each is a 5-member ensemble under the same folds.
  The number of real-spectrum draws per run is identical across configurations
  (16 × 1500).

  | name | init | per step | learning rate |
  |---|---|---|---|
  | A0 zero-shot | pretrained, no fine-tune | — | — |
  | A1 scratch + real | random | 16 real | 1e-3 |
  | A1m scratch + mixup | random | 16 real + 32 mixup | 1e-3 |
  | A2 pretrain + real | pretrained | 16 real | 3e-4 |
  | A3 pretrain + mixup | pretrained | 16 real + 32 mixup | 3e-4 |
  | **A4 full (primary)** | pretrained | 16 real + 32 mixup + 16 replay | 3e-4 |

  **A4 is the pre-registered primary model.** Every claim of "LUMEN beats X" refers to A4
  only. The other configurations are reported as ablations and are never selected after
  the fact.
- **Ensemble moments.** The prediction is the mean of the member means. The variance is
  the mean of the member variances plus the variance of the member means.
  - For the composition, member variances are the Dirichlet variances
    αᵢ(α₀−αᵢ)/(α₀²(α₀+1)).
- **Final model for M³** (step07): A4 fine-tuned on all 19 soils, 5 members.

## Baselines (step03)

- **D13 classic baselines.**
  - Input: the same 142 features, standardised on the training fold.
  - The composition model is fit on the composition-labelled training spectra; the
    scalar model (5 oxides + log Is/FeO) on all training spectra.
  - Composition outputs are clipped at 0 and renormalised to 100.
  - Methods:
    - **mean-of-others (fraction-matched):** the mean of the training spectra at the
      same sieve fraction. This is the zero-information reference, deliberately the
      stronger of the two obvious choices (the fraction is known metadata).
    - **PLS:** n_components ∈ 1..10, chosen by inner GroupKFold(5, by soil) on the
      training soils, minimising the MSE of training-standardised targets.
    - **Ridge:** α ∈ logspace(−3, 4, 15), chosen the same way.
    - **RF:** 500 trees, max_features 1/3, seed 0, no tuning.
    - **RF-raw:** the same RF on the 71 raw reflectances.
- **D14 Fernández-Díaz re-implementation** (FeO only):
  - Features: 2-level db4 DWT (periodization) of the 71-band reflectance, keeping A2 + D2
    (18 + 18 = 36 coefficients).
  - Autoencoder: in → 32 → 16 → 6 → 16 → 32 → in, LeakyReLU 0.01, BatchNorm, dropout
    0.3, Xavier initialisation, Adam 1e-3, weight decay 1e-5, batch 256, early stopping
    with patience 12 on a 10 % split, at most 200 epochs, raw coefficients as input.
    Two variants:
    - **FD-AE-M3:** trained unsupervised on the 269,952 Shapley-scene M³ pixels,
      converted to the lab scale with Korokhin Eq. 2 (A_lab = 1.685·A_M3 + 0.0355);
      non-finite or out-of-range pixels are dropped.
    - **FD-AE-synth:** trained on 270k synthetic spectra.

    The AE uses no labels and no LSCC spectra, so it is trained once rather than per fold.
  - Regressors on the 6-D latent and on the 36 coefficients directly (**FD-wavelet**,
    the no-AE variant):
    - Lasso, SVR-linear, SVR-RBF and RF, with the Table 5 grids of the reproduction;
    - StandardScaler inside the pipeline;
    - GridSearchCV with inner GroupKFold(5, by soil), scoring MAE.

    FD's published choice is RF; all four are reported.

## Evaluation (step04–step06)

- **D15 folds.**
  - **LOSO:** 19 folds, one per soil.
  - **LOMO:** 6 folds, one per mission — A11 (1 soil), A12 (2), A14 (4), A15 (2), A16 (6),
    A17 (4).
  - Every fraction of a held-out soil or mission is removed from training, mixup,
    normalisation statistics and inner CV. LUMEN's input statistics come from the
    synthetic set only; the baselines fit their scalers on the training fold.
  - Inner CV uses GroupKFold with min(5, number of training soils) splits.
- **Metrics.**
  - Per-soil error: the mean absolute error over the soil's labelled fractions.
    The headline **MAE is the mean over the 19 soils**; a per-spectrum MAE is also
    reported.
  - Composition is in vol %, oxides in wt %, maturity in log₁₀(Is/FeO).
  - Also reported: RMSE, R² (per spectrum), and Spearman ρ.
- **Skill.** skill = 1 − MAE / MAE(mean-of-others), with a bootstrap 95 % CI over soils
  (2000 resamples, percentile).
- **Paired tests.** A Wilcoxon signed-rank test (two-sided) on per-soil absolute errors,
  A4 vs each baseline. Holm correction across the 13 targets within each comparison.
  Significance level 0.05.
- **D16 identifiability gate.** A target is *identifiable* if the lower bound of its LOSO
  skill CI is > 0. Only identifiable targets are mapped on M³.
- **D17 uncertainty** (step05). Coverage is reported at 68 / 90 / 95 %, per spectrum and
  per soil, under LOSO and LOMO.
  - **(i) Raw ensemble:** μ ± z·σ (Gaussian), with the composition clipped to [0, 100].
  - **(ii) CV+-style conformal with normalised residuals:**
    - scores sᵢ = |yᵢ − μᵢ| / σᵢ, from the out-of-fold predictions of the *other*
      soils (LOSO) or missions (LOMO);
    - the interval for a held-out spectrum is μ ± q·σ, where q is the ⌈(1−α)(n+1)⌉-th
      smallest score.
  - **(iii) The same with absolute residuals** (σ ≡ 1).
  - Deviation from textbook CV+: the calibration residuals come from fold models that
    *included* the test soil. A nested scheme would need 19 × 18 × 5 models and is out of
    the compute budget. This makes (ii)/(iii) an approximation of CV+.
- **D18 head-to-head** (step06), each scored the way that paper scores itself:
  - **Korokhin 2024.**
    - 5-mineral proportions (HCP ↔ CPX, LCP ↔ "OPX", OLV, PLG, ILM) / sum × 100, on the
      <10, 10–20 and 20–45 µm fractions.
    - Per-soil mean over fractions; MAE per mineral against their Nelder–Mead fits
      (`step10_lscc_fits.csv`) and against their mean-of-other-soils baseline.
    - Pooled (n = 19) and by set; paired Wilcoxon, Holm over the 5 minerals.
    - Also A0 (zero-shot, the physics-only DL analogue of their ANN).
    - Maturity: Spearman ρ per fraction, by set and pooled, of LUMEN's Is/FeO vs measured,
      and of their nFe vs measured, recomputed from the same table.
  - **Jung 2026.**
    - Crystalline proportions OLV, PLG, PYX = LCP + HCP and ILM, normalised to their sum,
      for 12001, 12030, 15071 and 71061 at 20–45 µm.
    - The worst-case error is checked against their three claims (OLV < 7, ILM < 10 with
      and without 15071, PLG < 30) and against their prior-only control.
    - Maturity: Pearson r over the 4 staged soils, LUMEN Is/FeO vs measured, alongside
      their I_sm vs measured.
  - **Fernández-Díaz 2026.** FeO MAE, RMSE and R² under LOSO and LOMO, A4 against every
    FD variant and against the classic baselines; paired Wilcoxon over soils.
- **D19 M³ application** (step07).
  - Calibration to the lab scale uses Korokhin Eq. 2 as published (primary). The
    Apollo 16 re-derivation against LSCC 62231 <45 µm is a sensitivity analysis.
  - Maps are made with the final A4 ensemble, with conformal 90 % half-widths, and only
    for identifiable targets.
  - Compared with Korokhin's ANN maps on their 500-row grid, and with Jung's Aristarchus
    ROI posteriors.

## How the schedule was fixed

1. Architecture, loss form, pretraining length and fine-tuning recipe were written down
   a priori.
2. The only empirical choice was D9's L1 weight, made on synthetic validation data.
3. Code paths were dry-run on a synthetic pseudo-real set
   (`scripts/dev_synthetic_only.py`).
4. The first run on real folds is the reported run.

If a run crashes or a bug is found afterwards, the fix is recorded below and the whole
stage is re-run unchanged in every other respect.

## Deviations

1. **Korokhin truth definition (clarification of D18, before any LUMEN result existed).**
   The Korokhin reproduction takes the pyroxene split from LSCC Table 2 as printed. step01
   rescales it to the Table 1 pyroxene total, which keeps the 7-part composition closed.
   - The two 5-mineral truths differ by a mean of ≤ 0.23 pp per mineral, and by more than
     1 pp only for 61141 (<10, 10–20 µm) and 62231 (10–20 µm), where Table 2's pyroxene
     sum disagrees with Table 1.
   - Following D18 ("scored the way that paper scores itself"), every method in the
     Korokhin comparison is scored against **their** truth. The comparison against our
     truth is reported as a sensitivity.
   - The gap was found by a dry run of the evaluation code on placeholder predictions.
2. **Run order (no effect on results).**
   - step03 was restarted once, with the FD autoencoder pinned to one CPU thread, to stop
     it starving the concurrent pretraining. It had not yet written any output, and the
     rerun used the same seeds.
   - Fine-tuning started about 10 minutes late because of a process-wait bug in the
     launcher.
   - The fine-tuning pool was restarted once, from 3 to 4 workers. Finished jobs were kept,
     after checking every saved job file, and each job's seed depends only on (protocol,
     fold, configuration, member).
3. **Holm over the Fernández-Díaz variants (more conservative than pre-registered).**
   - D18 specified paired Wilcoxon tests of A4 against every FD variant.
   - The report additionally applies a Holm correction over the 12 variants, and shows the
     raw p alongside.
4. **Optional extension, run later as Addendum A.** The M³ landing-site validation (hand-off
   §6.8) was first blocked because the network policy denied planetarydata.jpl.nasa.gov.
   - Once the host was allowed, it was pre-registered (Addendum A) and run as step00b
     (staging) and step10 (scoring).
   - It is FeO only: the ground truth has no TiO₂.
5. **Paper draft generated from the tables.**
   - `PAPER_DRAFT.md` is assembled by `scripts/step09_paper.py` from hand-written sections
     in `paper/` and numbers read from `tables/`.
   - Every directional claim in the prose is asserted against the numbers before the draft
     is written.

## Addendum A — M³ landing-site validation (pre-registered before any site spectrum was fetched)

Added after the main study, when the PDS host became reachable. It is the hand-off's §6.8:
a quantitative test of LUMEN on real orbital spectra, against Fernández-Díaz et al.
(2026) on their own task. Everything below was fixed before any site spectrum was
downloaded.

- **D20 sites and truth.**
  - The 50 returned samples of `data/ground_truth/fernandez2026_ground_truth_50.csv`
    (FD's ground truth, FeO only) are grouped by distinct (lat, lon). One **site** is one
    coordinate, and its truth is the mean FeO of its samples.
  - All statistics are over sites, so FD's pseudo-replicates are never counted twice.
  - Coordinates are used exactly as FD lists them. Where FD gives samples from different
    stations the same coordinate (Apollo 17 LM vs LRV station 1), the site mixes them;
    this is flagged, not corrected.
- **D21 M³ spectra.**
  - **Candidate strips:** every global-mode L2 strip in `CH1M3_0004/INDEX/L2_INDEX.TAB`
    (806 products).
  - **Footprints:** probed from each strip's L1B LOC file (5 lines; range requests).
  - **Inclusion:** a strip covers a site if its nearest pixel is within 0.01° of the site
    and at least 5 samples from either swath edge.
  - **Strip spectrum:** the 3 × 3-pixel mean around the nearest pixel, after dropping
    pixels with any non-finite, ≤ 0 or ≥ 1 reflectance. A strip contributes only if
    ≥ 5 / 9 pixels remain.
  - **Site spectrum:** the per-band median over the contributing strips. Every method uses
    this spectrum. Sites with no contributing strip are reported and excluded.
  - **Storage:** the fetched spectra and their provenance are staged into
    `data/m3_sites/` by `scripts/step00b_fetch_m3_sites.py`, and the manifest is
    regenerated.
- **D22 methods** (FeO).
  - **LUMEN-M3 (primary)** — the A4 ensemble, applied with Korokhin Eq. 2 and **no M³
    training data**:
    - for an Apollo site of mission X, the 5 A4 LOMO members that held out mission X, so
      no soil from that mission was seen. Their weights are regenerated by re-running the
      identical fine-tuning jobs, and reproducibility is checked against the stored
      step04 predictions;
    - for Luna and Chang'e sites, the final all-soil ensemble.
  - **LUMEN-M3 + site recalibration** — leave-one-site-out OLS, y = a + b·ŷ, fit on the
    other sites. It uses the same information as the trained baselines.
  - **FD as published** — M³ bands 3–85 (83 bands), unscaled L2 reflectance; db4 level-2
    periodization, A2 + D2 (42 coefficients).
    - Autoencoder 42 → 32 → 16 → 6 → … trained unsupervised on the Shapley-scene pixels
      (same architecture and schedule as D14).
    - Regressors Lasso / SVR-lin / SVR-RBF / RF with the Table 5 grids.
    - Training rows are the samples of the other sites (as FD train on samples); inner
      GroupKFold by site; leave-one-site-out.
  - **FD no-AE** — the same with the 42 coefficients directly.
  - **Band ratio** — θ = arctan((R950 / R750 − 1.19) / (R750 − 0.08)) (Lucey et al.
    2000 origin), on the M³ bands nearest 750 and 950 nm, mapped to FeO by a
    leave-one-site-out linear fit.
  - **Mean of the other sites.**
- **D23 scoring.**
  - MAE, RMSE and R² over sites.
  - Paired Wilcoxon on per-site absolute errors, LUMEN-M3 against each comparator, with
    Holm over the comparators.
  - **Primary comparison:** LUMEN-M3 vs FD as published (AE → RF).
  - LUMEN's other outputs at the sites (mineralogy, TiO₂, maturity) are reported without
    truth.
  - Expected difficulties, stated in advance:
    - Chang'e-5 (FeO 22.5 wt %) lies above every LSCC soil (max 18.5), so LUMEN must
      extrapolate there;
    - one landing-site pixel per site is a small, noisy target.

## Addendum B — exact CV+ and M³ destriping (pre-registered before running)

These close the two remaining open items of the hand-off.

- **D24 exact CV+ (step11).** For every unordered pair of soils {i, j}, the 5 A4 members are
  fine-tuned without both soils: 171 pairs × 5 = 855 jobs. The recipe and seeds rule are
  those of D11; the seed hashes the pair.
  - For test soil j, the calibration scores of soil i come from model M_{ij}, which saw
    neither soil:
    - absolute residuals R = |y_i − μ_ij(x_i)| (per spectrum);
    - normalised residuals R / σ_ij(x_i).
  - CV+ interval for each spectrum of j (Barber et al. 2021):
    - absolute: [q_lo{μ_ij(x_j) − R}, q_hi{μ_ij(x_j) + R}] over every calibration spectrum
      of every i ≠ j;
    - normalised: the analogue with σ_ij(x_j) scaling.
  - The same is done for LOMO with the 15 mission pairs (75 jobs).
  - Coverage at 68 / 90 / 95 % is reported beside the approximate intervals of D17.
- **D25 M³ destriping (step07 sensitivity).** For every band, the along-track mean of each
  sample column is divided by a 31-sample running median of those column means, and each
  column is divided by that ratio. This is a column-gain flat field; it preserves the
  scene's large-scale signal.
  - The Shapley maps and the Korokhin map comparison are recomputed on the destriped cube
    and reported as a sensitivity.
  - The undestriped maps stay primary, as pre-registered in D19.

## Addendum C — orbital fine-tuning on M³ landing sites (pre-registered before running)

Addendum A showed that LUMEN trained on laboratory spectra does not transfer to M³. This
addendum tests whether a small orbital training signal fixes that. FeO only, scored on the
same 18 sites, leave-one-site-out, against the step10 comparators (unchanged).

- **D26 data.** The already-staged 3 × 3 blocks (`data/m3_sites`), no new download:
  - every valid pixel of every contributing strip is one training spectrum, labelled with
    its site's mean FeO;
  - the held-out site is predicted from its site spectrum, exactly as in step10;
  - Korokhin Eq. 2 calibration.
- **D27 LUMEN-M3-FT (primary).** For held-out site s and member k:
  - start from the same A4 member used in step10 (LOMO member of s's mission for Apollo
    sites, the final member otherwise);
  - fine-tune for 600 steps, AdamW lr 1e-4, wd 1e-4, cosine decay;
  - each batch holds 16 M³ pixel spectra from the other 17 sites (FeO supervised; the
    other targets masked) and 16 laboratory spectra from that model's own training soils
    (all targets), with the simulator's output randomisation;
  - seed = hash(site, member); prediction is the 5-member ensemble mean.
- **D28 LUMEN-embed + ridge (secondary).** The 128-d trunk embedding of the step10 model,
  averaged over members, then ridge regression on the other sites' pixel embeddings
  (α by inner leave-one-site-out over {0.1, 1, 10, 100}).
- **D29 scoring.** As D23: MAE, RMSE, R² over the 18 sites. Paired Wilcoxon of
  LUMEN-M3-FT against each step10 comparator, Holm over the comparators. The claim
  "beats Fernández-Díaz on M³" requires a lower MAE than every FD variant; significance
  is reported separately.
