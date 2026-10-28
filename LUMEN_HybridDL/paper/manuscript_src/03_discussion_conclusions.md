## 4. Discussion

### 4.1. What Carries the Gain on Laboratory Spectra

The decisive step is calibration against real soils. The zero-shot network is the weakest LUMEN variant and, for composition, worse than the zero-information reference, whereas any fine-tuning on returned soils cuts the error by a factor of two to four. Together with the simulator analysis, in which the LSCC spectra lie inside the synthetic distribution yet remain separable from it, this shows that a physics model good enough to cover the data is not, by itself, good enough to be inverted.

The other ingredients contribute differently. Learning on the whole spectrum is a real part of the gain, since the network trained from scratch already beats a tuned random forest, but the convolutional design is a reasonable rather than an optimal choice: the continuum-removed channel adds nothing measurable, a 1-D ResNet performs alike, and an MLP generalises better to an unseen mission on four targets. Physics pretraining helps consistently but is significant only out of distribution, so we regard the simulator as a source of a better starting representation rather than a calibrated forward model. SSA-space mixup and replay added nothing once the network was pretrained, either because pretraining already supplies the smoothness over composition space that mixup was meant to add, or because the cross-section mixing weights cannot be validated: sieve-fraction spectra mix linearly to 0.6% RMS, but the LSCC chemistry does not constrain the mass proportions well enough. Finally, choosing the configuration inside each training fold keeps the significant advantage over the random forest, which therefore does not depend on picking the configuration after seeing the results.

### 4.2. Comparison with Published Methods

The three comparisons are not equally informative. Only the Fernández-Díaz et al. [@fernandez2026] pipeline was trained on exactly the same spectra and folds, so LUMEN's 42% lower FeO error is the fair comparison; it also confirms that the autoencoder stage removes information. Korokhin et al. [@korokhin2024] and Jung et al. [@jung2026] never see an LSCC soil, so LUMEN's margins over them mostly measure the value of calibrating against returned soils: like for like, the zero-shot model beats the Shkuratov inversion significantly only for olivine, and Jung et al.'s four test soils are too few for strong conclusions. A practically relevant difference is maturity, which both physics methods rank worse than LUMEN's calibrated Is/FeO.

### 4.3. Laboratory-to-Orbit Transfer

On the base method's own task the result goes against LUMEN. Holding out whole landing sites (P2) raises the error of almost every fitted method relative to random folds (P1), but by very different amounts: the random-forest and RBF-SVR models lose most, 55–88% (FD as published rises from 1.90 to 3.58 wt%), LUMEN-M3-FT 49%, and the two-M³-band regression and linear SVRs least (6% and −1 to +2%). Under random folds, flexible models can learn the spectral signature of a landing site from its neighbouring stations, so random-fold errors at these sites, including FD's published 2.29 wt%, overstate the accuracy expected at a new location.

LUMEN-M3-FT's predictions are compressed toward the mean (Figure 8), over-estimating the Apollo 16 highland sites by 4.8 wt% on average and under-estimating the FeO-rich Chang'e-3, Chang'e-5 and Luna 24 sites by 6.3–7.5 wt%. With a few dozen training sites per fold, a linear model on two band parameters makes better use of the orbital labels, and even a ridge regression on LUMEN's embedding (2.60 wt%, not tested for significance) does better than fine-tuning the whole network. The laboratory-to-orbit gap is large: zero-shot predictions are worse than the training-fold mean, linear recalibration removes their bias but not their error, and orbital fine-tuning halves the error without closing the gap. The calibration itself is uncertain, since the published transfer reproduces the Apollo 16 laboratory spectrum to 9.1% RMS against 1.4% for a line re-derived from that pixel.

The in-sample Lucey/Clementine result does not show that the Lucey algorithm beats the other methods at a new site, because its constants were derived from returned-sample sites; it does show that a single parameter, θ, carries most of the FeO signal there. Refitted inside each fold, the two-band model is also more accurate on the table's Clementine reflectances than on our M³ site spectra (1.82 vs 2.11 wt% under P2), so part of the orbital error may lie in the M³ site spectra themselves (calibration, photometry, site localisation) rather than in the regressors. In the mapped scenes, the largest disagreements with published maps occur exactly where LUMEN's gate withholds the output (olivine) or where the published network itself recovers poorly (pyroxene split, maturity).

### 4.4. Uncertainty and Identifiability

Deep ensembles alone are over-confident, more so out of distribution, and normalised conformal calibration restores near-nominal coverage at no extra training cost, also on orbit, where at 95% the intervals average ±10.6 wt% and are therefore honest rather than useful. The gate makes explicit a limitation that other methods leave implicit: olivine abundance in these soils cannot be predicted better than by the mean of the other soils, by any method we tested, so olivine and "other" maps from physics inversions reflect their priors and end-member sets unless independently supported. Because the soils contain little olivine (≤5 vol%), this is a statement about the LSCC domain, not a claim that olivine is undetectable where it is abundant.

### 4.5. Limitations

The ground truth is small (19 laboratory soils, 56 spectra with modal mineralogy, and 49 orbital sites, 20 of them Apollo 17 stations), and several numerically large differences do not survive correction. Laboratory spectra of sieved fractions under fixed geometry differ from M³ pixels that integrate rock, regolith and topography over about 140 m, and the orbital comparison relies on site coordinates and a single linear calibration. The simulator's chemistry labels are biased for mare TiO₂ (+2.9 wt%) and its maturity scale rests on one constant, which limits the zero-shot model. The physics baselines enter through independent reproductions, and the Fernández-Díaz pipeline was re-implemented from its description. The orbital fine-tuning recipe was fixed in advance and not tuned; searching it, or smaller networks, needs more orbital ground truth. Transfer to other sensors, such as Chandrayaan-2 IIRS, is not tested here.

## 5. Conclusions

LUMEN, a physics-pretrained and sample-calibrated network, retrieves lunar soil mineralogy, chemistry and maturity from M³-band reflectance with calibrated intervals and an explicit statement of which outputs are identifiable. On identical laboratory spectra and folds, under a protocol fixed before scoring, it reduces FeO error by 42% relative to a published wavelet–autoencoder pipeline and mineral-proportion errors by 54–74% relative to a Shkuratov-model inversion, meets the published accuracy bounds of a Hapke–Bayesian inversion, and, with its configuration chosen inside the cross-validation, is significantly more accurate than a tuned random forest on 8 of 13 targets. Most of the improvement over physics-only methods comes from calibrating against returned soils; learning on the whole spectrum and physics pretraining add to it, the convolutional design is not uniquely good, and the proposed SSA-space augmentation does not help.

On M³ spectra at 49 returned-sample sites, LUMEN does not transfer zero-shot, and after orbital fine-tuning it is not significantly better than the published pipeline and significantly worse than a two-band linear regression, so for orbital FeO at these sites the deep model offers no advantage over simple band regressions. Olivine and minor glass abundances are not identifiable from these spectra by any method we tested. Future work should enlarge the orbital ground truth, which is now the limiting resource, add independent map-level checks against Kaguya and Clementine FeO products, and extend the approach to IIRS with an explicit sensor model.

---

**Author Contributions:**

**Funding:**

**Data Availability Statement:** All code, the version-controlled evaluation protocol, every intermediate table and the staged input data, with their SHA-256 manifests, are available at https://github.com/DSheth2004/lumen-lunar-dl. RELAB spectra are available from the PDS Geosciences Node spectral library, and M³ data from the NASA Planetary Data System. The returned-sample FeO table is Supplementary Table 1 of Qiu et al. [@qiu2025].

**Acknowledgments:** We thank the RELAB facility at Brown University and the Lunar Soil Characterization Consortium for the laboratory data, and the M³ team and the Planetary Data System for the orbital data.

**Conflicts of Interest:**

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
