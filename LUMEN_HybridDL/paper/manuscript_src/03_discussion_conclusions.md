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
  random forest on the same inputs, significantly on five targets. The convolutional architecture
  over log and continuum-removed reflectance is therefore a real part of the gain.
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
method. [[PENDING-E1: one-sentence interpretation]]

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

[[PENDING-E2: discussion of P1 vs P2; whether LUMEN-M3-FT beats FD-as-published and the band
ratios; size of the lab-to-orbit gap; what the in-sample Lucey/Clementine result means]]

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
distribution, at no extra training cost. Exact CV+ confirms the approximation.

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

**Configuration.** The pre-registered configuration was not the best one. E1 addresses this for
the laboratory soils, but the same question for the orbital model awaits more ground truth.

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
- its advantage over a well-tuned conventional regressor is large in magnitude but only partly
  significant with 19 soils.

**How much each part contributes.** Most of the improvement over physics-only methods comes from
calibrating against returned soils. The network architecture and physics pretraining add to it,
and the proposed SSA-space augmentation does not.

**Orbital M³ spectra.** [[PENDING-E2: one-sentence orbital conclusion]]

**Identifiability.** Olivine and minor glass abundances are not identifiable from these spectra by
any method we tested. Maps of them should be treated with caution.

**Future work** should:
- enlarge the orbital ground truth, which is now the limiting resource;
- add independent map-level checks against Kaguya and Clementine FeO products;
- extend the approach to IIRS with an explicit sensor model.

---

**Author Contributions:** [To be completed by the authors, CRediT taxonomy.]

**Funding:** [To be completed.]

**Data Availability Statement:**
- **Code and data.** All code, the version-controlled evaluation protocol, every intermediate
  table and the staged input data, with their SHA-256 manifests, are available at
  https://github.com/DSheth2004/lumen-lunar-dl.
- **Source data.** RELAB spectra are available from the PDS Geosciences Node spectral library. M³
  data are available from the NASA Planetary Data System. The returned-sample FeO table is
  Supplementary Table 1 of Qiu et al. [@qiu2025].

**Acknowledgments:**
- We thank the RELAB facility at Brown University and the Lunar Soil Characterization Consortium
  for the laboratory data, and the M³ team and the Planetary Data System for the orbital data.
- **AI-assisted work.** Parts of the analysis code and the text were drafted with the assistance
  of an AI system (Anthropic Claude). All results were generated by the released code, and the
  authors reviewed and take responsibility for the content. [Authors to confirm and adapt to the
  journal's policy.]

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
