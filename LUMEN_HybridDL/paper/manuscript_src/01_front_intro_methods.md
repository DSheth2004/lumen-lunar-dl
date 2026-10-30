# Physics-Pretrained, Sample-Calibrated Deep Learning for Lunar Soil FeO, Mineralogy and Maturity from Moon Mineralogy Mapper Spectra

## Highlights

**What are the main findings?**

- A one-dimensional network pretrained on 400,000 Hapke-simulated lunar soils and fine-tuned on 19 returned soils predicts FeO, four other oxides, seven-part mineralogy and maturity. On the same laboratory data it reduces FeO error by 42% relative to a published wavelet-autoencoder pipeline and is significantly more accurate than a tuned random forest on 8 of 13 targets
- The ranges between similarities are not very large, and a identifiability gate removes olivine and minor glasses, which no other method could reasonably forecast better than a zero information baseline.

**What are the implications of the main findings?**

- Most of the error in physics only inversions can be eliminated through calibration against returned soils, and not by increasing model complexity.
- Laboratory skill did not transfer to orbit: at 49 returned sample sites, a LUMEN in orbit (3.27 wt% FeO) was not significantly superior to that reported (3.58 wt%) and outperformed by a two-band linear regression (2.11 wt%).

## Abstract

In many cases, lunar soil composition can be recovered from reflectance spectra using empirical regressions, radiative transfer inversions and physics-guided networks. This work presents LUMEN, a one-dimensional convolutional network trained on 400,000 soils from a domain randomised Hapke simulator with nanophase iron space weathering, fine-tuned on laboratory spectra of 19 Lunar Soil Characterization Consortium soils, with conformal prediction intervals and an identifiability gate to with skillful outcomes. Under leave-one-out evaluation with a protocol specified before scoring, LUMEN predicts FeO to 1.26 wt% and TiO₂ to 0.70 wt%, lowers the FeO error of a published wavelet–autoencoder pipeline by 42% and mineral-proportion errors of a Shkuratov-model inversion by 54–74%, and, when configured during cross-validation, is more accurate than a tuned random forest on 8 out of 13 targets. Periodictures reach nominal coverage; olivine is not identifiable. On Moon Mineralogy Mapper spectra of 49 returned sample sites, the laboratory model does not transfer (6.81 wt%), after orbital fine-tuning, it hits 3.27 wt% under leave-one-landing-out and is not any better than the published pipeline (3.58 wt%) or worse than a two-band linear regression (2.11 wt%).

**Keywords:** lunar regolith; FeO; mineral abundance; space weathering; Moon Mineralogy Mapper; Hapke model; deep learning; transfer learning; conformal prediction; uncertainty quantification

## 1. Introduction

Iron and titanium content, modal mineralogy, and space-weathering maturity of the lunar regolith record the crustal and volcanic history of the Moon and constrain its use of resources. The Moon Mineralogy Mapper (M³) on Chandrayaan-1 observed visible to near-infrared reflectance spectra over most of the Moon [@pieters2009; @green2011]. These spectra show the crystal-field absorptions of Fe²⁺ in pyroxene and olivine near 1 and 2 µm, overprinted with the darkening and reddening of nanophase metallic iron, and their quantification is a trick without solid foundation.

Several mathematical algorithms have been developed to link spectral properties with the composition of samples returned from landing sites. The Clementine band-ratio algorithms of Lucey et al. [@lucey2000] are the standard for FeO and TiO₂ and the 1 and 2 µm band parameters have been calibrated to M³ [@bhatt2012; @bhatt2015]. These methods range from partial least squares [@sun2016], neural networks [@xia2019] and support vector machines [@wang2021] to comparisons between regressors [@qiu2022], and convolutional networks on Kaguya Multiband Imager spectra have been used as global maps of oxide when Chang'e-5 [@yang2023; @zhang2023] and Chang'e-6 [@shi2026] samples were added; Diviner data [@ma2022], gradient boosting [@wu2024] and random forests [@fernandez2025] have been used. The method I build on, Fernández-Díaz et al. [@fernandez2026], uses a db4 wavelet decomposition, a deep autoencoder and a random forest regressor to produce a global map of M³ FeO at 150 m per pixel. Each is trained on about 50 sampling sites and evaluated using random splits or small hold-out sets, rarely against other methods or a zero-information baseline.

Physics-based methods model reflectance with radiative transfer. The Hapke model [@hapke1993] and the results for nanophase iron [@hapke2001; @lucey2011; @pieters2016] support inversions for mineral abundance, grain size and submicroscopic iron [@lili2011; @sunlucey2021; @trang2019]; the Shkuratov model [@shkuratov1999] has been inverted on M³ by a network trained on synthetic spectra [@korokhin2024]; Hapke inversions have been embedded into Bayesian samplers [@jung2026]; and PGU-Net [@lin2026] placed a Hapke single-scattering-albedo mixing layer within a deep autoencoder for unsupervised unmixing that validated only on M³ against Kaguya products with sample-informed priors as the next step. Such inversions are interpretable but not unique and sensitive to the end-member set and optical constants. Laboratory studies of the Lunar Soil Characterization Consortium (LSCC) soils have also combined spectra and measured mineralogy [@li2012; @grumpe2018; @surkov2025].

We also reproduce three previous methods ([@fernandez2026; @korokhin2024; @jung2026]), and find several limitations. For example, the network of Korokhin et al. recovers orthopyroxene, olivine and the grey absorber on its own synthetic test set with only R² of 0.06, 0.17 and 0.32, yet these are mapped; in Jung et al., abundances from the prior alone already meet two of the three published accuracy bounds; and in Fernández-Díaz et al., a random forest on the raw wavelet coefficients is more accurate than one on the autoencoder codes. Each point indicates how much of a returned number is estimated and how it is tested, not the physical models themselves. What we are missing is a method that uses physics as a source of training data, lets returned soils correct it, states its uncertainty and what it cannot identify, and then compare it against the published pipelines with identical data, under grouped cross-validation and against a zero-information baseline.

We propose LUMEN (LUnar Mineralogy Estimation Network) to fill this gap. The main contributions are:

(1) **A hybrid physics-learning method** A domain-randomised Hapke simulator, composed of 49 lunar mineral and glass end-members (none from an evaluation soil) with physically modelled nanophase- and microphase-iron weathering, generates 400,000 synthetic soils. A small, one-dimensional convolutional network is trained on these and fine-tuned on returned soils, and from one spectrum predicts seven-part modal mineralogy, five oxides and maturity. A deep ensemble with conformal calibration supplies intervals with verified coverage, and a gate restricts any target with apparent skill over a zero information baseline.

(2) **An identical-data, protocol-fixed comparison.** LUMEN is compared with wavelet–autoencoder regression [@fernandez2026], Shkuratov-model inversion [@korokhin2024], Hapke–Bayesian inversion [@jung2026] and classical regressors using the same 75 laboratory spectra and leave-one-soil-out and leave-one-mission-out folds. Pair tests were used to correct for multiple comparisons. Every evaluation choice was merged into version control before the first score was computed; negative results and null are reported as positive.

(3) **Test of laboratory-to-orbit transfer** Using M³ spectra from returned sample sites on which the base method was calibrated [@fernandez2026; @qiu2025], as well as its own cross-validation and an honest spatial leave-one-landing-site-out protocol, we show that without orbital labels, a small orbital fine-tuning set recovers, and simple two-band regressions remain better in orbit.

Section 2 describes the data, the physical preliminaries and the method; Section 3 reports the experiments; Section 4 discusses them, including limitations; and Section 5 concludes.

## 2. Materials and Methods

### 2.1. Data and Preprocessing

#### 2.1.1. Laboratory Spectra and Ground Truth

We used the LSCC suite of data [@taylor2001; @taylor2010] for nine mare soils (Apollo 11, 12, 15 and 17) and ten highland soils (Apollo 14 and 16) analyzed as <10, 10–20, 20–45 and <45 µm fractions measured at 30° incidence and 0° emission. The 10–20 µm spectrum of soil 70181 was not analyzed, leaving 75 spectra. Point-count modal mineralogy (vol%) data for the three finer fractions (56 spectra) was arranged into seven parts: plagioclase (PLG), low-Ca pyroxene (LCP, orthopyroxene plus pigeonite), high-Ca pyroxene (HCP), olivine (OLV), ilmenite (ILM), agglutinitic glass (AGG) and other phases (OTH). Bulk FeO, TiO₂, Al₂O₃, CaO and MgO (wt%) and maturity index Is/FeO are available for all 75 spectra, but have been checked against their column sum and totals. Each spectra is contigated in the 71 M³ global mode band (540.8–2497.1 nm) with Gaussian responses that span the band width of each channel, except for channels smaller than 2.5 µm.

#### 2.1.2. End-Member Library

The simulator contains 49 RELAB spectra of lunar mineral separates and glasses (17 plagioclase, 7 low-Ca and 11 high-Ca pyroxene, 6 olivine, 4 ilmenite, 2 volcanic and 2 synthetic glasses) from mare basalts, highland rocks, one non-LSCC soil (62241) and synthetic glasses. Each spectra has been individually verified against the PDS4 metadata. No end-members are found in any of the 19 evaluation soils; as each RELAB agglutinate separate is derived from an LSCC soil, we model agglutinates physically (Section 2.3.1).

#### 2.1.3. M³ Data

All orbital data are M³ Level-2 global-mode reflectance [@green2011]. We map the strip in the Shapley-region M3G20090728T214215 as analyzed by Korokhin et al. [@korokhin2024], the strip around the landing site of Apollo 16 (M3G20090607T110414) and the two regions of Aristarchus of interest of Jung et al. [@jung2026].

The FeO ground truth in the base method [@fernandez2026] is derived from Qiu et al. [@qiu2025jgr], which is open access as Supplementary Table 1 of Qiu et al. [@qiu2025]: 49 stations (Apollo 11–17 landing sites and traverse stations, Luna 16, 20 and 24, and Chang’e-3, -5 and -6) with laboratory FeO and Clementine reflectances at 415–1000 nm, which we confirmed against the worked example in [@fernandez2026] (Apollo 17 LRV-12, 17.4 wt% FeO). For each site, every M³ strip whose nearest pixel lies within 0.01° of the site, at least five samples from the swath edge, contributes its 3 × 3-pixel block if at least 5 of the 9 pixels are valid (finite, 0 < R < 1 on M³ bands 3–85, the band set of [@fernandez2026]); the site spectrum is the per-band median over contributing strips. All 49 sites are covered, Chang'e-5 included, by 183 strips (one to six per site, median four). All input files are recorded in a SHA-256 manifest that is validated at the beginning and end of each processing stage.

### 2.2. Preliminaries: Hapke Model, Single-Scattering Albedo and Space Weathering

To investigate an intimate particle mixture we used the Hapke bidirectional reflectance [@hapke1993];

$$ r(w) = \frac{w}{4(\mu_0+\mu)}\,H(w,\mu_0)\,H(w,\mu), \tag{1} $$

where $w$ is the single scattering albedo (SSA), $\mu_0$ and $\mu$ are the cosines of the incidence and emission angles (30° and 0° for RELAB). Ignoring the opposition effect and using an isotropic phase function is appropriate for RELAB geometry. $H$ is the approximate Chandrasekhar function

$$ H(w,x) = \left[1 - w\,x\left(r_0 + \frac{1-2r_0x}{2}\ln\frac{1+x}{x}\right)\right]^{-1},\qquad r_0=\frac{1-\gamma}{1+\gamma},\ \gamma=\sqrt{1-w}. \tag{2} $$

Equation (1) is monotonic in $w$ and is inverted numerically from a dense table.

The SSA of a grain population with absorption coefficient $\alpha$, real index $n$, imaginary index $k$ and size $D$ follows the equivalent-slab model [@hapke1993],

$$ w = S_e + (1-S_e)\,\frac{(1-S_i)\,\Theta}{1-S_i\,\Theta},\qquad \Theta = e^{-\alpha\langle D\rangle}, \tag{3} $$

with

$$ S_e = \frac{(n-1)^2+k^2}{(n+1)^2+k^2} + 0.05,\qquad S_i = 1.014 - \frac{4}{n(n+1)^2},\qquad \langle D\rangle = \frac{2}{3}\left[n^2-\frac{1}{n}(n^2-1)^{3/2}\right]D, \tag{4} $$

If we take $\alpha = 4\pi k/\lambda$ for the case of a pure phase, we use Equations (1)–(4) to solve for $k(\lambda)$ using the nominal grain size for each end-member. This inversion is limited because there is no monotonic relationship between the reflectance and $k$ – it decreases from the beginning until reaching the opacity floor and starts increasing due to surface reflections.

Nanophase metallic iron (npFe⁰) at volume fraction $\phi$ in a host of refractive index $n_h$ adds an absorption [@hapke2001; @lucey2011]

$$ \alpha_{np}(\lambda) = \frac{36\pi\, z(\lambda)\,\phi}{\lambda},\qquad z = \frac{n_h^3\, n_{Fe}\, k_{Fe}}{\left(n_{Fe}^2-k_{Fe}^2+2n_h^2\right)^2+\left(2n_{Fe}k_{Fe}\right)^2}, \tag{5} $$

where $n_{Fe}$ and $k_{Fe}$ are the optical constants of iron. Microphase iron (mpFe⁰) is a grey absorber, $\alpha_{mp} = 3\phi_{mp}/(4 r_{mp})$ with $r_{mp} = 0.1$ µm, which darkens without reddening [@lucey2011]. Components with volume fractions $f_i$, grain sizes $D_i$ and SSAs $w_i$ mix in SSA with cross-section weights,

$$ w_{mix} = \frac{\sum_i (f_i/D_i)\,w_i}{\sum_i f_i/D_i}, \tag{6} $$

and the mixed SSA is converted to reflectance with Equation (1).

### 2.3. Proposed Method: LUMEN

The stages of LUMEN are four (see Figure 1): the physics simulator produces labeled synthetic soils; the neural network is pretrained on these data; and the neural network is then finetuned on returned soils and, in the case of orbit, M³ spectra; while the uncertainty/identifiability layer decides on the output.

![Figure 1](../figures/fig01_framework.png)

**Figure 1.** Introduction to LUMEN. The soil synthetic generated through the physics simulator (see Figure 2) acts as a pre-training step for the network (see Figure 3). It is then fine-tuned with lab spectra from returned soils as well as M³ spectra from returned-sample sites. The identifiability gate decides the targets that are mapped onto.

#### 2.3.1. Domain-Randomised Physics Simulator

A synthetic soil defines its petrology regime, which may be feldspathic, basaltic, or general (Dirichlet priors for crystal fractions bounded by ≤20% of ilmenite and ≤60% of olivine) and a maturity degree which determines the fraction of agglutinates, AGG = clip(0.65 · maturity + N(0, 0.12), 0, 0.75). This is complemented with the load values of npFe⁰ and mpFe⁰, a "residual" fraction, grain size distributed between 4 and 150 µm, as well as one end-member for each phase (Figure 2). Agglutinates include feldspathic-mafic synthetic glass matrix containing the iron loadings, with calibration offset (tilt), ±7% scale, and 0.2 – 0.8% additive noise (instrument noise). Chemical nomenclatures are based on stoichiometries with an accuracy uncertainty (±15%) and molar Fe-Mg isomorphisms used in the nominal phase chemistries. The maturity goal is defined by Is/FeO = 2270 · npFe⁰ (wt %)/FeO (wt %) - a scale-only constant calibrated with fine tuning. Four hundred thousand spectra are created with the same seed value.

![Figure 2](../figures/fig02_simulator.png)

**Figure 2.** The physics simulation in the domain-randomization process. The imaginary indices k(λ) of the end-members are derived through the inversion of the equivalent slab; the effects of space weathering include npFe⁰ and mpFe⁰ absorption; the combination of the components occurs through the single-scattering albedo through cross-section weighing; then through the Hapke model they are converted to reflectance; the spectrum gets resampled to 71 M³ bands.

#### 2.3.2. Network

The network (see Figures 3 and Table 1) uses two channels: log reflectance and convex hull continuum-removed reflectance for 71 bands, where standardization takes place according to statistics that is estimated based solely on the artificial dataset. This design includes five 1D convolutional layers, followed by group normalization (a better choice than batch normalization since fine-tuning is based on very small mini-batches) and activation GELU, then two max-pooling. This results in a trunk with dimensions 256 – 128 and 0.36 million parameters. Then a Dirichlet head generates concentrations $\boldsymbol\alpha = \mathrm{softplus}(\cdot)+0.05$, where its mean $\boldsymbol\alpha/\alpha_0$, with $\alpha_0=\sum_j\alpha_j$, is equivalent to seven-part composition. There are heteroscedastic Gaussian heads generating means and logs of variance for oxides and log₁₀(Is/FeO). Auxiliary heads for grain size and npFe⁰ are used during pre-training only.

![Figure 3](../figures/fig03_architecture.png)

**Figure 3.** Network architecture. The shape of tensors is defined as channels × bands. Every convolution is followed by group normalization (8 groups) and GELU non-linearity. The auxiliary heads are used only in pre-training.

#### 2.3.3. Objective Functions

The compositional loss is equivalent to the negative log-likelihood of the smoothed labels with respect to the Dirichlet distribution where $\tilde{\mathbf y} = (\mathbf y + \epsilon)/(1 + 7\epsilon)$ and $\epsilon = 0.005$, augmented by $L_1$ regularizer on the mean:

$$ \mathcal L_{comp} = -\frac{1}{7}\left[\ln\Gamma(\alpha_0) - \sum_j \ln\Gamma(\alpha_j) + \sum_j(\alpha_j-1)\ln\tilde y_j\right] + \lambda_1\,\frac{1}{7}\sum_j\left|\frac{\alpha_j}{\alpha_0} - y_j\right|, \tag{7} $$

using $\lambda_1 = 10$ with synthetic data only. Scalar output variables use β-log likelihood [@seitzer2022] for $\beta=0.5$ to reduce the problem of underfitting of high variance data due to Gaussian likelihood.

$$ \mathcal L_{scal} = \frac{\sum_t m_t\,\mathrm{sg}\!\left[\sigma_t^{2\beta}\right]\tfrac12\left(\ln\sigma_t^2 + \frac{(y_t-\mu_t)^2}{\sigma_t^2}\right)}{\sum_t m_t}, \tag{8} $$

In places where the gradient is blocked by $\mathrm{sg}$, and $m_t$ prevents those targets from being used which are not available for that particular sample (mineral composition of the <45 µm size; all targets excluding FeO for orbital spectra). The total loss includes $\mathcal L_{comp}$ and $\mathcal L_{scal}$ along with β-NLL auxiliary terms during pretraining.

#### 2.3.4. Training

The five models comprising the ensemble were pretrained on the synthetic set with different random seed values (15 epochs, AdamW optimizer, one-cycle learning rate schedule, 20,000 spectra for validation), followed by fine-tuning for 1,500 steps per cross-validation split (cf. Table 1). For the complex setup (A4), each training step uses 16 spectra from real training samples, 16 synthetic replay spectra, and 32 spectra from mixup in SSA space, each being a combination of 2–3 training samples as defined by equation (6), precisely labeled (composition – volume fraction, chemistry – mass fractions of Is, FeO, with Is/FeO – by mass × FeO). Real and mixed spectra are passed through calibration and noise randomization in the simulator. The simple setup (A2) consists of just real training samples. The hyperparameter values for both setups were fixed based on the experiments on the synthetic data only.

#### 2.3.5. Uncertainty and Identifiability

The definition of an ensemble mean is the average of the means of all members, whereas that of an ensemble variance is the average of all member variances, plus the variance of the means of the member [@lakshminarayanan2017]. Raw ensemble prediction intervals are over-confident (Section 3.5) and hence conformally calibrated [@barber2021] using the normalized score, $s_i = |y_i - \mu_i|/\sigma_i$, that uses out-of-fold predictions of other soils to calculate the $\lceil (1-a)(n+1)\rceil$-th smallest value $q_{1-a}$, giving rise to the interval, $\mu \pm q_{1-a}\,\sigma$. The CV+ version uses models where not only the calibration soil but also the test soil was left out (930 more fine-tunes). Identifiability gates are used to determine the skill for every target.

$$ S = 1 - \frac{\mathrm{MAE}}{\mathrm{MAE}_{ref}}, \tag{9} $$

In this case, $\mathrm{MAE}_{ref}$ represents the error of the fraction-matched mean of the other soils used as zero information, with 95% bootstrap confidence interval estimated across soils. Only those targets for which the lower end of the confidence interval is above zero are considered to be mapped.

#### 2.3.6. Transfer to M³

Laboratory and M³ reflectance differ in absolute scale. In zero-shot tasks, M³ spectral data is scaled to the laboratory scale based on the linear scaling algorithm suggested by Korokhin et al. [@korokhin2024].

$$ A_{lab} = 1.685\,A_{M^3} + 0.0355, \tag{10} $$

An outline of the site of mission X of the Apollo program is created with the help of the trained ensemble without mission X soils (in the case of the Luna and Chang'e sites, the ensemble is trained with all 19 soils). During the orbital fine-tuning procedure (LUMEN-M3-FT), the neural network receives an additional training of 600 steps (using AdamW optimizer, 10⁻⁴ learning rate and weight decay, cosine decay) with the batches of 16 M³ pixel spectra of the training sites labeled with the FeO content of the site while all the other targets are masked, as well as 16 laboratory spectra of the training soils with all targets.
