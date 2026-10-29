## 4. Discussion

### 4.1. What Carries the Gain on Laboratory Spectra

Most importantly, calibration to real soils. The zero-shot network is the weakest LUMEN model and, composition-wise, below the zero-information reference, while any fine-tuning on returned soils reduces the error by a factor of 2 to 4. This, combined with the simulator analysis, where the LSCC spectra are within the synthetic distribution while still being distinguishable from it, demonstrates that a physics model accurate enough to span the data is not necessarily accurate enough to be inverted.

The rest add in various ways. Total learning through the entire range is a real component in the performance gain since even a network trained from scratch is already better than a tuned random forest. But the convolutional architecture is a good but not an optimal design – the continuum removed channel gives no improvement at all, a one-dimensional ResNet works equally well, and an MLP better generalizes to an unseen mission on four targets. Physics pretraining is consistently beneficial but its importance is significant only for out-of-distribution tasks, which makes the simulator not a calibrated forward model, but a source for an improved starting representation instead. Neither SSA-space mixup nor replay brought anything after pretraining, whether due to smoothness over the composition space being provided by pretraining, or inability to check the validity of the cross-section mixing weight: the sieve fraction spectra mixes linearly to 0.6% RMS, but the LSCC chemistry does not limit the mass ratios strongly enough. And finally, choosing the configuration within each training fold maintains the considerable performance edge over the random forest, independent of post-hoc configuration selection.

### 4.2. Comparison with Published Methods

Three comparisons are non-equivalent. Of these, only the Fernández-Díaz et al. [@fernandez2026] pipeline was trained on the same spectra and folds, so the fact that LUMEN's FeO error is 42% less gives us an unbiased comparison, and verifies the information compression of autoencoder stage. Korokhin et al. [@korokhin2024] and Jung et al. [@jung2026] never see an LSCC soil, so the LUMEN margins over them are mostly a measure of the value of calibrating by returned soils: the zero-shot LUMEN far exceeds the Shkuratov inversion in only olivine, and Jung et al. four test soils are too few for a strong conclusion. One practically relevant property is maturity, to which physics methods in this comparison ranked worse than the calibrated Is/FeO provided by LUMEN.

### 4.3. Laboratory-to-Orbit Transfer

As for the inherent purpose of the base method, the resulting errors are contradictory to the LUMEN. In cases when whole landing sites (P2) are excluded, most of the fitted models produce greater errors than models based on randomly chosen folds (P1); however, the scale of this difference is significantly different: the maximum increase is observed in the random forest and RBF-SVR models – it is within the range between 55% and 88%, whereas the FD error rate rises from 1.90 to 3.58 wt%. The LUMEN-M3-FT model has a percentage increase in errors – by 49%, whereas the two-M³-band and linear SVR regression models have the lowest values – the difference is within the range of 6% and from −1% to +2%, respectively. If randomly chosen folds are used, flexible models can use the spectral properties of the adjacent landing site.

LUMEN-M3-FT's predictions are compressed toward the mean (Figure 8), over-estimating the Apollo 16 highland sites by 4.8 wt% on average and under-estimating the FeO-rich Chang'e-3, Chang'e-5 and Luna 24 sites by 6.3–7.5 wt%. With a few dozen training sites per fold, a linear model on two band parameters makes better use of the orbital labels, and even a ridge regression on LUMEN's embedding (2.60 wt%, not tested for significance) does better than fine-tuning the whole network. The laboratory-to-orbit gap is big: zero-shot predictions are suboptimal compared to the training-fold mean, linear recalibration eliminates their bias but not their error, and orbital fine-tuning reduces their error by half without reducing the gap. The calibration itself is uncertain, since the published transfer reproduces the Apollo 16 lab spectrum at 9.1% RMS, whereas a line back-calculated from that pixel is 1.4%.

The in-sample Lucey/Clementine result doesn’t prove that the Lucey algorithm outperforms other methods at a new site—its constants come from sites where samples were already returned. What it does show is that one parameter, θ, carries most of the FeO signal in that context. When we refit the two-band model within each fold, it actually matches the Clementine reflectances on the table better than it matches the M³ site spectra (1.82 vs 2.11 wt% under P2). That suggests some of the orbital error comes from the M³ site spectra themselves—calibration, photometry, or even site localization—rather than from the regressors. Looking at the mapped scenes, the biggest mismatches with published maps pop up exactly where LUMEN’s gate holds back the output (in olivine regions), or where the published network already struggles (with the pyroxene split or with maturity).

### 4.4. Uncertainty and Identifiability

Deep ensembles tend to be over-confident, especially when dealing with out-of-distribution data. Normalized conformal calibration helps fix this, bringing coverage back to near-nominal levels, and you get this benefit without any extra training. This holds even in orbit, where at 95% coverage, the intervals average ±10.6 wt%. So, you get honest intervals—they're accurate, though not always practically useful. The gating method clarifies something that other approaches only hint at: we can’t actually predict olivine abundance in these soils any better than simply taking the mean from other soils. None of the methods we tried did any better. As a result, maps showing olivine and "other" from physics-based inversion methods just mirror their priors and end-member sets if you don’t have independent evidence to back them up. But don't misinterpret this: these soils just happen to contain very little olivine (≤5 vol%). So, this is a specific statement about the LSCC domain. It’s not saying olivine can’t be detected at all—just that it’s rare here.

### 4.5. Limitations

The ground truth set is small: just 19 laboratory soils, 56 spectra with modal mineralogy, and 49 orbital sites (20 from Apollo 17 stations). A number of large numeric gaps vanish after correction. There's a gap between how lab spectra are collected—sieved fractions under fixed geometry—and how M³ pixels see the Moon, integrating rock, regolith, and topography over areas around 140 meters. Orbital comparisons lean on exact coordinates and a single linear calibration. The simulator’s chemistry labels show a positive bias for mare TiO₂, around +2.9 wt%, and its maturity scale pins everything to one constant, which puts constraints on the zero-shot model. Physics baselines come from independent reproductions, and we re-implemented the Fernández-Díaz pipeline based on published descriptions. The orbital fine-tuning method was set beforehand and left unchanged; trying different recipes or smaller networks calls for more orbital ground truth. Testing transfer to other sensors, like Chandrayaan-2 IIRS, isn't part of this work.

## 5. Conclusions

LUMEN is a physics-pretrained, sample-calibrated network that pulls out lunar soil mineralogy, chemistry, and maturity directly from M³-band reflectance. It does this using calibrated intervals and always makes clear which outputs you can actually identify. Tested on the same lab spectra, using the same folds, and under a protocol locked down before any scoring, LUMEN cuts FeO error by 42% compared to a published wavelet–autoencoder pipeline. For mineral-proportion errors, it reduces them by between 54% and 74% when stacked up against an inversion method based on the Shkuratov model. It matches the published accuracy range for Hapke–Bayesian inversion too. When its setup is chosen during cross-validation, it outperforms a tuned random forest on 8 of 13 targets. Nearly all the gains over pure physics-driven methods come from calibrating on real returned soils. Training across the full spectrum and doing physics-based pretraining help as well. The convolutional structure isn’t anything special here—it’s not what drives the improvement. And the proposed SSA-space augmentation turns out not to add value.

At 49 returned-sample sites with M³ spectra, LUMEN does not improve on the published pipeline by zero-shot transfer (no training on a given site), does not improve with orbital fine-tuning, and is significantly worse than a two-band linear regression, so the deep model does not beat simple band regressions for orbital FeO at the returned sites. Olivine and minor glass abundances are not inferrable from these spectra in any of the methods we tried. Further work would increase the orbital ground-truth available, presently the limiting factor, and incorporate independent map-level validation against Kaguya and Clementine FeO maps, and deploy the method on IIRS with explicit sensor model.

---

**Author Contributions:**

**Funding:**

**Data Availability Statement:** You’ll find all the code, the version-controlled evaluation protocol, every intermediate table, and the staged input data—along with their SHA-256 manifests—at https://github.com/DSheth2004/lumen-lunar-dl. The RELAB spectra come from the PDS Geosciences Node spectral library, and the M³ data sit in the NASA Planetary Data System. For the returned-sample FeO table, check Supplementary Table 1 in Qiu et al. [@qiu2025].

**Acknowledgments:** Thanks to the RELAB facility at Brown University, and the Lunar Soil Characterization Consortium for the laboratory data; to the M³ team and the Planetary Data System for the orbital data.

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
