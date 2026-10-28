## 4. Discussion

### 4.1. Discussion of the Laboratory Results: What Carries the Gain

**The decisive step is calibration against real soils, not the physics alone.**
- The zero-shot network, trained only on simulated soils, is the weakest LUMEN variant. For
  composition it is worse than the zero-information reference (MAE 9.1 vs 5.3 vol%).
- Any fine-tuning on the returned soils cuts the error by a factor of two to four.

This matches the simulator analysis: the 75 LSCC spectra lie inside the synthetic distribution,
yet a classifier separates them from it with an AUC of 0.97. A physics model good enough to cover
the data is not, by itself, good enough to be inverted.

**Network architecture, physics and augmentation each contribute differently.**
- **The network itself.** The same network trained from scratch on 18 soils already beats a tuned
  random forest on the same inputs, significantly on five targets. Learning a network on the whole
  spectrum is therefore a real part of the gain.
- **But the specific design is not uniquely good (E4).**
  - The continuum-removed channel adds nothing measurable to log reflectance.
  - A 1-D ResNet performs about the same as the main network.
  - A pretrained MLP has a higher error on all 13 targets in distribution (significantly only for
    LCP), but out of distribution it is significantly better on four targets, FeO included.
  - The convolutional design is therefore a reasonable choice, not an optimal one: a smaller,
    less structured network may generalise better to an unseen mission.
- **Physics pretraining.** It lowers the error on every target under leave-one-soil-out. However,
  that improvement is not significant for any single target after correction. Out of
  distribution, it is significant for high-Ca pyroxene and maturity.
- We therefore present the simulator as a source of a better starting representation, not as a
  calibrated forward model.
- **SSA-space mixup and synthetic replay** added nothing once the network was pretrained. Two
  explanations are consistent with the data:
  - pretraining already supplies the smoothness over composition space that mixup was meant to
    add;
  - the cross-section mixing weights are poorly constrained. Sieve-fraction spectra do mix
    linearly to 0.6% RMS, but the LSCC chemistry cannot pin down the mass proportions well enough
    to validate the weighting.

We kept the pre-registered configuration as the primary model and report the simpler one's better
numbers as an ablation. The nested selection of Section 3.3 turns that choice into part of the
method. Chosen inside each training fold, the configuration is A2 in every fold, and the selected
model remains significantly more accurate than the random forest on 8 of 13 targets, FeO
included. The advantage over RF therefore does not depend on picking the configuration after
seeing the results.

### 4.2. Discussion of the Comparison with Published Methods

The three comparisons are not equally informative.

**Fernández-Díaz et al. [@fernandez2026] is the fair comparison.** Its pipeline was re-implemented
and trained on exactly the same spectra and folds as LUMEN.
- LUMEN's 42% lower FeO error is significant on the test fixed in the protocol.
- Our results also agree with the reproduction of the original study: the autoencoder stage
  removes information, and the raw wavelet coefficients serve the regressor better under
  leave-one-soil-out.

**Korokhin et al. [@korokhin2024] and Jung et al. [@jung2026] are physics inversions that never
see an LSCC soil.** LUMEN's large margins over them therefore mostly measure the value of
calibrating against returned soils.
- In the like-for-like comparison, the zero-shot model beats the Shkuratov inversion significantly
  only for olivine.
- Jung et al.'s test set is four soils, too few for strong conclusions.
- Both physics methods' maturity parameters rank the soils worse than LUMEN's calibrated Is/FeO.
  This is a concrete and practically relevant difference.

### 4.3. Discussion of the M³ Results: Laboratory-to-Orbit Transfer

**The orbital result goes against LUMEN.** On the base method's own task, LUMEN-M3-FT is not the
most accurate orbital FeO estimator.

**Grouping by landing site changes the ranking.** Holding out whole landing sites (P2) raises the
error of almost every fitted method relative to random folds (P1), but by very different amounts:
- the random-forest and RBF-SVR models lose most, 55–88%; FD as published rises from 1.90 to 3.58
  wt%;
- LUMEN-M3-FT rises by 49% (2.20 → 3.27 wt%);
- the two-M³-band regression and the linear SVRs lose least: 6% (1.99 → 2.11 wt%) and −1 to +2%.

Under random folds, flexible models can learn the spectral signature of a landing site from its
neighbouring stations. P2 removes that, so random-fold errors at these sites overstate how well a
model will do at a new location. FD's published 2.29 wt% and our P1 numbers should be read in that
light.

**Against the comparators.**
- Under P2, LUMEN-M3-FT is lower than FD as published, but not significantly. It is significantly
  worse than the two-band regressions on M³ (2.11 wt%) and on Clementine reflectances (1.82 wt%),
  and than FD-AE-M3 / SVR-lin (2.27 wt%).
- Under P1, FD as published is the lower of the two for every seed tested.
- LUMEN-M3-FT's predictions are compressed toward the mean (Figure 5). It over-estimates the
  Apollo 16 highland sites by 4.8 wt% on average, and under-estimates the FeO-rich Chang'e-3,
  Chang'e-5 and Luna 24 sites by 6.3–7.5 wt%.
- With a few dozen training sites per fold, a linear model on two band parameters is the better
  use of the orbital labels. A ridge regression on LUMEN's embedding (2.60 wt%) also does better than
  fine-tuning the whole network (3.27 wt%), though this difference was not tested.

**The laboratory-to-orbit gap is large.**
- The zero-shot model, fine-tuned only on laboratory spectra, has an MAE of 6.81 wt% with a bias
  of −5.29 wt%. That is worse than predicting the training-fold mean (4.78 wt% under P2).
- A linear recalibration on the training sites removes the bias but not the error (5.11 wt%).
- Fine-tuning on orbital spectra halves the error (3.27 wt%) but does not close the gap.
- The laboratory skill (1.26 wt% on the LSCC soils) therefore does not carry over to M³ pixels
  through a linear calibration plus a few dozen orbital labels.

**What the in-sample Lucey/Clementine result shows.** The published Lucey et al. constants reach
1.25 wt% (R² = 0.88) on the Clementine reflectances of the same sites. These constants were
derived from returned-sample sites, so this is an in-sample fit, not a held-out estimate, and it
does not show that the Lucey algorithm beats the other methods at a new site. What it does show is
that a single spectral parameter, θ, carries most of the FeO signal at these sites. Refitted inside
each fold, the same two-band model is also more accurate on the table's Clementine reflectances
than on our M³ site spectra (1.82 vs 2.11 wt% under P2). Part of the remaining orbital error may
therefore lie in the M³ site spectra themselves (calibration, photometry, site localisation), not
only in the regressors.

**Calibration.** The laboratory–orbital calibration is itself uncertain. The published linear
transfer (Equation (10)) reproduces the Apollo 16 laboratory spectrum to 9.1% RMS, against 1.4%
for a line re-derived from that pixel. So a single linear calibration is unlikely to be enough,
and an orbital training signal is needed.

**Scenes.** The Shapley and Aristarchus comparisons show why identifiability matters on orbit:
- the largest disagreements with published maps occur exactly for quantities that LUMEN's gate
  withholds (olivine) or that the published network itself recovers poorly (pyroxene split,
  maturity);
- on the unflagged targets, LUMEN reproduces the expected mare–highland contrasts.

### 4.4. Uncertainty and Identifiability

**Coverage.** Deep ensembles alone are over-confident, more so out of distribution. Normalised
conformal calibration restores nominal coverage in distribution and comes close out of
distribution, at no extra training cost. Exact CV+ confirms the approximation. On orbit, the same
calibration gives LUMEN-M3-FT coverage of 0.63/0.92/0.98 at the 49 sites under P2. Those
intervals are close to nominal, but at 95% they average ±10.6 wt%, so they are honest rather than
useful.

**What can be identified.**
- The gate makes explicit a limitation that other methods leave implicit: from VNIR spectra of
  these soils, olivine abundance cannot be predicted better than by the mean of the other soils,
  by any method we tested.
- The olivine and "other" maps of physics inversions should therefore be read as properties of
  their priors and end-member sets, unless independent evidence supports them.
- Our soils contain little olivine (≤5 vol%), so this is a statement about the LSCC domain, not a
  claim that olivine is undetectable where it is abundant.

### 4.5. Limitations

**Small ground truth.**
- There are 19 laboratory soils (75 spectra, 56 with modal mineralogy) and 49 orbital sites.
- The orbital sites are unevenly spread: 20 are Apollo 17 stations, and 8 of the 12 landing-site
  groups hold a single site.
- Every significance statement is over these units. Several numerically large differences do not
  survive correction.

**Laboratory and orbital data differ.**
- Laboratory spectra are measured on sieved fractions under fixed geometry. M³ pixels integrate
  rock, regolith and topography over about 140 m.
- The orbital comparison relies on site coordinates and on a single linear calibration.

**Simulator.**
- The chemistry labels are biased for mare TiO₂ (+2.9 wt%).
- The maturity scale rests on one constant.
- Both are re-learned during fine-tuning, but they limit the zero-shot model.

**Baselines.**
- The physics baselines enter through independent reproductions, validated against their papers'
  printed numbers, not through the authors' own runs.
- The Fernández-Díaz pipeline was re-implemented from its description.

**Configuration.**
- The pre-registered configuration was not the best one. Choosing it inside the cross-validation
  (E1) resolves this for the laboratory soils.
- The orbital fine-tuning recipe was fixed in advance and not tuned. A fair search over it, or
  over smaller networks (Section 3.7), needs more orbital ground truth.

**Scope.** The method uses M³ global-mode bands. Its transfer to other sensors such as
Chandrayaan-2 IIRS, whose radiometry and band set differ, is not tested here.

## 5. Conclusions

We presented LUMEN, a physics-pretrained, sample-calibrated deep network that retrieves lunar soil
mineralogy, chemistry and maturity from M³-band reflectance, with calibrated intervals and an
explicit statement of which outputs are identifiable.

**Laboratory spectra.** On identical spectra and folds, under a protocol fixed before scoring:
- LUMEN reduces FeO error by 42% relative to a published wavelet–autoencoder pipeline;
- it reduces mineral-proportion errors by 54–74% relative to a Shkuratov-model inversion;
- it meets the published accuracy bounds of a Hapke–Bayesian inversion;
- with its configuration chosen inside the cross-validation, it is significantly more accurate
  than a tuned random forest on 8 of 13 targets, FeO included.

**How much each part contributes.** Most of the improvement over physics-only methods comes from
calibrating against returned soils. Learning a network on the whole spectrum and physics
pretraining add to it. The convolutional design is not uniquely good: an MLP is better out of
distribution on four targets. The proposed SSA-space augmentation does not help.

**Orbital M³ spectra.** On M³ spectra at 49 returned-sample sites, LUMEN does not transfer
zero-shot (6.81 wt%). Fine-tuned on orbital FeO, it reaches 3.27 wt% under leave-one-landing-site-out,
not significantly different from the published pipeline (3.58 wt%) and significantly worse than a
linear regression on two M³ bands (2.11 wt%). For orbital FeO at these sites, the deep model
therefore offers no advantage over simple band regressions.

**Identifiability.** Olivine and minor glass abundances are not identifiable from these spectra by
any method we tested. Maps of them should be treated with caution.

**Future work** should:
- enlarge the orbital ground truth, which is now the limiting resource;
- add independent map-level checks against Kaguya and Clementine FeO products;
- extend the approach to IIRS with an explicit sensor model.

---

**Author Contributions:** [To be completed by the authors, CRediT taxonomy.]

**Funding:** [To be completed.]

**Data Availability Statement:** All code, the version-controlled evaluation protocol, every intermediate table and the staged input data, with their SHA-256 manifests, are available at https://github.com/DSheth2004/lumen-lunar-dl. RELAB spectra are available from the PDS Geosciences Node spectral library, and M³ data from the NASA Planetary Data System. The returned-sample FeO table is Supplementary Table 1 of Qiu et al. [@qiu2025].

**Acknowledgments:** We thank the RELAB facility at Brown University and the Lunar Soil Characterization Consortium for the laboratory data, and the M³ team and the Planetary Data System for the orbital data.

**Conflicts of Interest:** The authors declare no conflicts of interest.

**Abbreviations**

| Abbreviation | Meaning |
|---|---|
| AGG | agglutinitic glass |
| CV+ | cross-validation-plus conformal prediction |
| FD | Fernández-Díaz et al. pipeline |
| HCP / LCP | high- / low-Ca pyroxene |
| ILM | ilmenite |
| LOSO / LOMO | leave-one-soil-out / leave-one-mission-out |
| LSCC | Lunar Soil Characterization Consortium |
| M³ | Moon Mineralogy Mapper |
| MAE | mean absolute error |
| npFe⁰ / mpFe⁰ | nanophase / microphase metallic iron |
| OLV | olivine |
| OTH | other phases |
| PLG | plagioclase |
| RELAB | Reflectance Experiment Laboratory |
| RF | random forest |
| SSA | single-scattering albedo |
