## 1. Introduction

Mapping lunar mineralogy, chemistry and maturity from orbital reflectance spectra underpins
studies of the Moon's crustal evolution, volcanism and space weathering. Two families of
methods dominate.

- **Physics-based inversions.** These include Hapke radiative transfer with Bayesian sampling
  (Jung et al., 2026) and Shkuratov-model fitting by Nelder–Mead or by a neural network trained
  on synthetic spectra (Korokhin et al., 2024). They are interpretable, but non-unique and
  sensitive to the end-member set and to the optical-constant model.
- **Empirical regressions.** These relate spectral features to sample chemistry, for example
  wavelet features compressed by an autoencoder and fed to a random forest for FeO
  (Fernández-Díaz et al., 2026). They are accurate in-sample, but are trained on a few dozen
  ground-truth sites and rarely tested out of distribution.

Our independent reproductions of these three methods expose four common gaps:

- **(G1) a sim-to-real gap.** Physics-only inversions are mis-calibrated on real soils. On the
  19 LSCC soils, Korokhin's fits lose to a mean-of-other-soils baseline for olivine. Jung's
  worst-case ilmenite error (17.6 points) exceeds the claimed bound (10), and the prior alone
  meets two of their three accuracy claims.
- **(G2) untested identifiability.** Korokhin's network recovers orthopyroxene, olivine and the
  grey absorber with R² ≤ 0.32 on its own synthetic test set, yet these are mapped.
- **(G3) no calibrated uncertainty.** Jung's released sampler degenerates to 0–1 % unique
  particles, so its credible intervals are not posteriors.
- **(G4) weak validation.** Fernández-Díaz's 50 FeO samples sit on about 15 distinct coordinates
  and the headline MAE is a 5-sample hold-out. No study compares against a zero-information
  baseline.

We propose LUMEN, a hybrid that keeps the physics where it helps (pretraining and physically
consistent augmentation) and lets real soils correct it. It addresses each gap:

- **G1:** a domain-randomised Hapke simulator for pretraining, then fine-tuning on real soils
  with a new physically exact augmentation (SSA-space mixup);
- **G3:** a deep ensemble with conformal calibration;
- **G2:** an identifiability gate that withholds targets without demonstrable skill;
- **G4:** a pre-registered evaluation under leave-one-soil-out and leave-one-mission-out, with
  paired tests against all three methods on identical data and a zero-information baseline.

We report where LUMEN does not win as prominently as where it does.
