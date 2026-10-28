# Physics-Pretrained, Sample-Calibrated Deep Learning for Lunar Soil FeO, Mineralogy and Maturity from Moon Mineralogy Mapper Spectra

## Highlights

**What are the main findings?**

- A one-dimensional network pretrained on 400,000 Hapke-simulated lunar soils and fine-tuned on 19 returned soils predicts FeO, four other oxides, seven-part mineralogy and maturity. On identical laboratory data it cuts FeO error by 42% relative to a published wavelet–autoencoder pipeline and is significantly more accurate than a tuned random forest on 8 of 13 targets.
- Conformal intervals reach nominal coverage, and an identifiability gate withholds olivine and minor glasses, which no tested method predicts better than a zero-information baseline.

**What are the implications of the main findings?**

- Most of the error of physics-only inversions is removed by calibrating against returned soils, not by added model complexity.
- Laboratory skill does not transfer to orbit: at 49 returned-sample sites, orbitally fine-tuned LUMEN (3.27 wt% FeO) does not significantly improve on the published pipeline (3.58 wt%) and is outperformed by a two-band linear regression (2.11 wt%).

## Abstract

Lunar soil composition is retrieved from reflectance spectra by empirical regressions, radiative-transfer inversions and physics-guided networks, which are rarely compared on identical data or given calibrated uncertainties. We present LUMEN, a one-dimensional convolutional network pretrained on 400,000 soils from a domain-randomised Hapke simulator with nanophase-iron space weathering, fine-tuned on laboratory spectra of the 19 Lunar Soil Characterization Consortium soils, with conformal prediction intervals and an identifiability gate that withholds outputs without demonstrable skill. Under leave-one-soil-out evaluation with a protocol fixed before scoring, LUMEN predicts FeO to 1.26 wt% and TiO₂ to 0.70 wt%, reduces the FeO error of a published wavelet–autoencoder pipeline by 42% and the mineral-proportion errors of a Shkuratov-model inversion by 54–74%, and, with its configuration chosen inside the cross-validation, is significantly more accurate than a tuned random forest on 8 of 13 targets. Intervals reach nominal coverage; olivine is not identifiable. On Moon Mineralogy Mapper spectra at 49 returned-sample sites, the laboratory model does not transfer (6.81 wt%); after orbital fine-tuning it reaches 3.27 wt% under leave-one-landing-site-out, not significantly better than the published pipeline (3.58 wt%) and significantly worse than a two-band linear regression (2.11 wt%).

**Keywords:** lunar regolith; FeO; mineral abundance; space weathering; Moon Mineralogy Mapper; Hapke model; deep learning; transfer learning; conformal prediction; uncertainty quantification

## 1. Introduction

The iron and titanium content, modal mineralogy and space-weathering maturity of the lunar regolith record the Moon's crustal and volcanic history and constrain the use of its resources. The Moon Mineralogy Mapper (M³) on Chandrayaan-1 acquired visible to near-infrared reflectance spectra over most of the surface [@pieters2009; @green2011]. These spectra carry the crystal-field absorptions of Fe²⁺ in pyroxene and olivine near 1 and 2 µm, overprinted by the darkening and reddening of nanophase metallic iron, and turning them into quantitative composition remains an inverse problem with little ground truth.

Empirical methods relate spectral parameters to the composition of returned samples at their landing sites. The Clementine band-ratio algorithms of Lucey et al. [@lucey2000] remain the reference for FeO and TiO₂, and band parameters of the 1 and 2 µm absorptions have been calibrated for M³ [@bhatt2012; @bhatt2015]. Regressions range from partial least squares [@sun2016], neural networks [@xia2019] and support vector machines [@wang2021] to systematic comparisons of regressors [@qiu2022], and convolutional networks on Kaguya Multiband Imager spectra produced global oxide maps once Chang'e-5 [@yang2023; @zhang2023] and Chang'e-6 [@shi2026] samples were added; Diviner data [@ma2022], gradient boosting [@wu2024] and random forests [@fernandez2025] have also been used. The method this study builds on, by Fernández-Díaz et al. [@fernandez2026], combines a db4 wavelet decomposition, a deep autoencoder and a random-forest regressor into a global M³ FeO map at 150 m per pixel. All of these methods are trained on about 50 sampling sites and evaluated on random splits or small hold-out sets, rarely on identical data alongside competing methods or against a zero-information baseline.

Physics-based methods model reflectance with radiative transfer. The Hapke model [@hapke1993] and its treatment of nanophase iron [@hapke2001; @lucey2011; @pieters2016] support inversions for mineral abundance, grain size and submicroscopic iron [@lili2011; @sunlucey2021; @trang2019]; the Shkuratov model [@shkuratov1999] has been inverted on M³ by a network trained on synthetic spectra [@korokhin2024]; Hapke inversions have been embedded in Bayesian samplers [@jung2026]; and PGU-Net [@lin2026] placed a Hapke single-scattering-albedo mixing layer inside a deep autoencoder for unsupervised unmixing, validated on M³ only indirectly, against Kaguya products, with sample-informed priors named as the next step. Such inversions are interpretable but non-unique and sensitive to the end-member set and optical constants. Laboratory studies of the Lunar Soil Characterization Consortium (LSCC) soils have also combined spectra with measured mineralogy [@li2012; @grumpe2018; @surkov2025].

Our independent reproductions of three recent methods [@fernandez2026; @korokhin2024; @jung2026] show where these approaches fall short. The network of Korokhin et al. recovers orthopyroxene, olivine and the grey absorber on its own synthetic test set with R² of only 0.06, 0.17 and 0.32, yet these quantities are mapped; in Jung et al., abundances drawn from the prior alone already meet two of the three published accuracy bounds; and in Fernández-Díaz et al., a random forest on the raw wavelet coefficients is more accurate than one on the autoencoder codes. Each point concerns how much of a retrieved number the spectrum determines and how that is tested, not the physical models themselves. What is missing is a method that uses physics as a source of training data, lets returned soils correct it, states its uncertainty and which outputs it cannot identify, and is evaluated against published pipelines on identical data, under grouped cross-validation and against a zero-information baseline.

We propose LUMEN (LUnar Mineralogy Estimation Network) to fill this gap. The main contributions are:

(1) **A hybrid physics–learning method.** A domain-randomised Hapke simulator, built from 49 lunar mineral and glass end-members (none taken from an evaluation soil) with physically modelled nanophase- and microphase-iron weathering, generates 400,000 synthetic soils. A compact one-dimensional convolutional network is pretrained on them and fine-tuned on returned soils, and from one spectrum predicts seven-part modal mineralogy, five oxides and maturity. A deep ensemble with conformal calibration supplies intervals with verified coverage, and an identifiability gate withholds every target whose skill over a zero-information baseline is not demonstrable.

(2) **An identical-data, protocol-fixed comparison.** LUMEN is compared with wavelet–autoencoder regression [@fernandez2026], Shkuratov-model inversion [@korokhin2024], Hapke–Bayesian inversion [@jung2026] and classical regressors on the same 75 laboratory spectra and the same leave-one-soil-out and leave-one-mission-out folds, with paired tests corrected for multiple comparisons. Every evaluation choice was committed to version control before the first score was computed, and negative and null results are reported alongside positive ones.

(3) **A test of laboratory-to-orbit transfer.** Using M³ spectra at the returned-sample sites on which the base method was calibrated [@fernandez2026; @qiu2025], under both that method's own cross-validation and a spatially honest leave-one-landing-site-out protocol, we show that laboratory accuracy does not carry over to orbit without orbital labels, how much a small orbital fine-tuning set recovers, and that simple two-band regressions remain more accurate on orbit.

Section 2 describes the data, the physical preliminaries and the method; Section 3 reports the experiments; Section 4 discusses them, including limitations; and Section 5 concludes.

## 2. Materials and Methods

### 2.1. Data and Preprocessing

#### 2.1.1. Laboratory Spectra and Ground Truth

We use the LSCC suite [@taylor2001; @taylor2010] of nine mare soils (Apollo 11, 12, 15 and 17) and ten highland soils (Apollo 14 and 16), each sieved into <10, 10–20, 20–45 and <45 µm fractions and measured at the Reflectance Experiment Laboratory (RELAB) [@relab2004] at 30° incidence and 0° emission; the 10–20 µm spectrum of soil 70181 is absent, leaving 75 spectra. Point-count modal mineralogy (vol%), available for the three finer fractions (56 spectra), is merged into seven parts: plagioclase (PLG), low-Ca pyroxene (LCP, orthopyroxene plus pigeonite), high-Ca pyroxene (HCP), olivine (OLV), ilmenite (ILM), agglutinitic glass (AGG) and other phases (OTH). Bulk FeO, TiO₂, Al₂O₃, CaO and MgO (wt%) and the maturity index Is/FeO are available for all 75 spectra; the tables were checked against their own column sums and totals. Every spectrum is convolved to the 71 M³ global-mode bands between 540.8 and 2497.1 nm with Gaussian responses whose width equals the band spacing, excluding the thermally affected channels beyond 2.5 µm.

#### 2.1.2. End-Member Library

The simulator draws on 49 RELAB spectra of lunar mineral separates and glasses (17 plagioclase, 7 low-Ca and 11 high-Ca pyroxene, 6 olivine, 4 ilmenite, 2 volcanic and 2 synthetic glasses) from mare basalts, highland rocks, one non-LSCC soil (62241) and synthetic glasses, each verified against its PDS4 metadata. An automatic check ensures that no end-member derives from any of the 19 evaluation soils; because every RELAB agglutinate separate comes from an LSCC soil, agglutinates are modelled physically instead (Section 2.3.1).

#### 2.1.3. M³ Data

All orbital data are M³ Level-2 global-mode reflectance [@green2011]. For mapping we use the Shapley-region strip M3G20090728T214215 analysed by Korokhin et al. [@korokhin2024], a block around the Apollo 16 landing site (M3G20090607T110414) and the two Aristarchus regions of interest of Jung et al. [@jung2026].

The FeO ground truth of the base method [@fernandez2026] is the compilation of Qiu et al. [@qiu2025jgr], available in open access as Supplementary Table 1 of Qiu et al. [@qiu2025]: 49 distinct, station-level sampling sites (Apollo 11–17 landing sites and traverse stations, Luna 16, 20 and 24, and Chang'e-3, -5 and -6) with laboratory FeO and Clementine reflectances at 415–1000 nm, which we verified against the worked example in [@fernandez2026] (Apollo 17 LRV-12, 17.4 wt% FeO). For each site, every M³ strip whose nearest pixel lies within 0.01° of the site, at least five samples from the swath edge, contributes its 3 × 3-pixel block if at least 5 of the 9 pixels are valid (finite, 0 < R < 1 on M³ bands 3–85, the band set of [@fernandez2026]); the site spectrum is the per-band median over contributing strips. All 49 sites are covered, Chang'e-5 included, by 183 strips (one to six per site, median four). Every input file is listed in a SHA-256 manifest that each processing stage verifies at start and end.

### 2.2. Preliminaries: Hapke Model, Single-Scattering Albedo and Space Weathering

For an intimate particulate mixture we use the Hapke bidirectional reflectance [@hapke1993],

$$ r(w) = \frac{w}{4(\mu_0+\mu)}\,H(w,\mu_0)\,H(w,\mu), \tag{1} $$

where $w$ is the single-scattering albedo (SSA) and $\mu_0$ and $\mu$ are the cosines of the incidence and emission angles (30° and 0° for RELAB). The phase function is isotropic and the opposition effect is neglected, which suits RELAB geometry, and $H$ is the approximate Chandrasekhar function

$$ H(w,x) = \left[1 - w\,x\left(r_0 + \frac{1-2r_0x}{2}\ln\frac{1+x}{x}\right)\right]^{-1},\qquad r_0=\frac{1-\gamma}{1+\gamma},\ \gamma=\sqrt{1-w}. \tag{2} $$

Equation (1) is monotonic in $w$ and is inverted numerically from a dense table.

The SSA of a grain population with absorption coefficient $\alpha$, real index $n$, imaginary index $k$ and size $D$ follows the equivalent-slab model [@hapke1993],

$$ w = S_e + (1-S_e)\,\frac{(1-S_i)\,\Theta}{1-S_i\,\Theta},\qquad \Theta = e^{-\alpha\langle D\rangle}, \tag{3} $$

with

$$ S_e = \frac{(n-1)^2+k^2}{(n+1)^2+k^2} + 0.05,\qquad S_i = 1.014 - \frac{4}{n(n+1)^2},\qquad \langle D\rangle = \frac{2}{3}\left[n^2-\frac{1}{n}(n^2-1)^{3/2}\right]D, \tag{4} $$

and $\alpha = 4\pi k/\lambda$ for a pure phase. For each end-member we invert Equations (1)–(4) at the separate's nominal grain size to obtain $k(\lambda)$. Because the reflectance is not monotonic in $k$, falling to an opacity floor and then rising as surface reflection grows, the root is bracketed on the physical, falling branch; every end-member then reproduces its own measured reflectance to < 10⁻¹⁰.

Nanophase metallic iron (npFe⁰) at volume fraction $\phi$ in a host of refractive index $n_h$ adds an absorption [@hapke2001; @lucey2011]

$$ \alpha_{np}(\lambda) = \frac{36\pi\, z(\lambda)\,\phi}{\lambda},\qquad z = \frac{n_h^3\, n_{Fe}\, k_{Fe}}{\left(n_{Fe}^2-k_{Fe}^2+2n_h^2\right)^2+\left(2n_{Fe}k_{Fe}\right)^2}, \tag{5} $$

where $n_{Fe}$ and $k_{Fe}$ are the optical constants of iron. Microphase iron (mpFe⁰) is a grey absorber, $\alpha_{mp} = 3\phi_{mp}/(4 r_{mp})$ with $r_{mp} = 0.1$ µm, which darkens without reddening [@lucey2011]. Components with volume fractions $f_i$, grain sizes $D_i$ and SSAs $w_i$ mix in SSA with cross-section weights,

$$ w_{mix} = \frac{\sum_i (f_i/D_i)\,w_i}{\sum_i f_i/D_i}, \tag{6} $$

and the mixed SSA is converted to reflectance with Equation (1).

### 2.3. Proposed Method: LUMEN

LUMEN has four stages (Figure 1): a physics simulator generates labelled synthetic soils; a network is pretrained on them; it is fine-tuned on returned soils and, for orbital use, on M³ spectra; and an uncertainty and identifiability layer decides what is reported.

![Figure 1](../figures/fig01_framework.png)

**Figure 1.** Overview of LUMEN. Synthetic soils from the physics simulator (Figure 2) pretrain the network (Figure 3), which is then fine-tuned on laboratory spectra of returned soils and, optionally, on M³ spectra at returned-sample sites. A five-member ensemble with conformal calibration provides prediction intervals, and the identifiability gate decides which targets are mapped.

#### 2.3.1. Domain-Randomised Physics Simulator

Each synthetic soil draws a petrological regime (feldspathic, basaltic or broad: Dirichlet priors on the crystalline proportions, truncated at ≤20% ilmenite and ≤60% olivine), a maturity that sets the agglutinate fraction, AGG = clip(0.65 · maturity + N(0, 0.12), 0, 0.75), and the npFe⁰ and mpFe⁰ loads, an "other" fraction, grain sizes of 4–150 µm and one end-member per phase (Figure 2). Agglutinates are a feldspathic–mafic synthetic-glass host carrying the iron loads, and a calibration tilt, a ±7% scale factor and 0.2–0.8% noise represent instrument effects. Chemistry labels follow from stoichiometry, with ±15% jitter and molar Fe–Mg exchange on the nominal phase chemistry, and the maturity target is set by Is/FeO = 2270 · npFe⁰ (wt%)/FeO (wt%), a constant that fixes only the scale, which fine-tuning re-learns. Four hundred thousand spectra are generated with a fixed seed.

![Figure 2](../figures/fig02_simulator.png)

**Figure 2.** The domain-randomised physics simulator. Each end-member's imaginary index k(λ) is obtained by inverting the equivalent-slab model; space weathering adds npFe⁰ and mpFe⁰ absorption; components mix in single-scattering albedo with cross-section weights and are converted to reflectance with the Hapke model; the spectrum is resampled to the 71 M³ bands. Composition, grain size, maturity and instrument effects are randomised, and labels follow from stoichiometry.

#### 2.3.2. Network

The network (Figure 3, Table 1) reads two channels on the 71 bands, log reflectance and convex-hull continuum-removed reflectance, each standardised with statistics of the synthetic set only. Five one-dimensional convolutions, each followed by group normalisation (preferred to batch normalisation because fine-tuning batches are small) and GELU, with two max-pooling steps, feed a 256–128 trunk (0.36 M parameters). A Dirichlet head outputs concentrations $\boldsymbol\alpha = \mathrm{softplus}(\cdot)+0.05$ whose mean $\boldsymbol\alpha/\alpha_0$, with $\alpha_0=\sum_j\alpha_j$, is the seven-part composition; heteroscedastic Gaussian heads give a mean and log-variance for each oxide and log₁₀(Is/FeO); and auxiliary heads for grain size and npFe⁰ are used during pretraining only.

![Figure 3](../figures/fig03_architecture.png)

**Figure 3.** Network architecture. Tensor shapes are given as channels × bands. Each convolution is followed by group normalisation (8 groups) and GELU. The auxiliary heads are used during pretraining only.

#### 2.3.3. Objective Functions

The composition loss is the Dirichlet negative log-likelihood of the label-smoothed proportions, $\tilde{\mathbf y} = (\mathbf y+\epsilon)/(1+7\epsilon)$ with $\epsilon = 0.005$, plus an $L_1$ term on the mean:

$$ \mathcal L_{comp} = -\frac{1}{7}\left[\ln\Gamma(\alpha_0) - \sum_j \ln\Gamma(\alpha_j) + \sum_j(\alpha_j-1)\ln\tilde y_j\right] + \lambda_1\,\frac{1}{7}\sum_j\left|\frac{\alpha_j}{\alpha_0} - y_j\right|, \tag{7} $$

with $\lambda_1 = 10$ fixed on synthetic data only. The scalar targets use the β-negative log-likelihood [@seitzer2022] with $\beta=0.5$, which avoids the under-fitting of high-variance points caused by the plain Gaussian likelihood,

$$ \mathcal L_{scal} = \frac{\sum_t m_t\,\mathrm{sg}\!\left[\sigma_t^{2\beta}\right]\tfrac12\left(\ln\sigma_t^2 + \frac{(y_t-\mu_t)^2}{\sigma_t^2}\right)}{\sum_t m_t}, \tag{8} $$

where $\mathrm{sg}$ stops the gradient and $m_t$ masks targets that are unavailable for a sample (modal mineralogy for the <45 µm fraction; all targets except FeO for orbital spectra). The total loss is $\mathcal L_{comp} + \mathcal L_{scal}$, plus the auxiliary β-NLL terms during pretraining.

#### 2.3.4. Training

Five ensemble members are pretrained on the synthetic set with different seeds (15 epochs, AdamW, one-cycle schedule, 20,000 spectra held out) and fine-tuned per cross-validation fold for 1500 steps (Table 1). In the full configuration (A4) each step combines 16 real training spectra, 16 synthetic replay spectra and 32 SSA-space mixup spectra, each a Dirichlet-weighted mixture of 2–3 training spectra combined with Equation (6) and labelled exactly (composition by volume, chemistry by mass, Is/FeO weighted by mass × FeO); real and mixed spectra receive the simulator's calibration and noise randomisation. The simpler configuration A2 uses the real spectra only. All hyperparameters were fixed from synthetic-only development runs and committed before any model was scored on real folds.

#### 2.3.5. Uncertainty and Identifiability

The ensemble mean is the mean of the member means, and the ensemble variance is the mean of the member variances plus the variance of the member means [@lakshminarayanan2017]. Because raw ensemble intervals are over-confident (Section 3.5), we calibrate them conformally [@barber2021]: the normalised score $s_i = |y_i - \mu_i|/\sigma_i$ is computed from out-of-fold predictions of the other soils, and its $\lceil (1-a)(n+1)\rceil$-th smallest value $q_{1-a}$ gives the interval $\mu \pm q_{1-a}\,\sigma$. The exact CV+ variant uses models from which both the calibration soil and the test soil were held out (930 additional fine-tunes). For each target, the identifiability gate computes the skill

$$ S = 1 - \frac{\mathrm{MAE}}{\mathrm{MAE}_{ref}}, \tag{9} $$

where $\mathrm{MAE}_{ref}$ is the error of the fraction-matched mean of the other soils, a zero-information reference, with a 95% bootstrap interval over soils. A target is mapped only if the lower bound of that interval is above zero under leave-one-soil-out evaluation.

#### 2.3.6. Transfer to M³

Laboratory and M³ reflectance differ in absolute scale. For zero-shot application, M³ spectra are brought to the laboratory scale with the linear calibration of Korokhin et al. [@korokhin2024],

$$ A_{lab} = 1.685\,A_{M^3} + 0.0355, \tag{10} $$

and a site of Apollo mission X is predicted by the ensemble fine-tuned with all of mission X's soils held out (for the Luna and Chang'e sites, by the ensemble fine-tuned on all 19 soils). For orbital fine-tuning (LUMEN-M3-FT), the network is further trained for 600 steps (AdamW, learning rate 10⁻⁴, weight decay 10⁻⁴, cosine decay) on batches of 16 M³ pixel spectra from the training sites, labelled with their site FeO and all other targets masked, and 16 laboratory spectra from that model's own training soils with all targets. The held-out site is predicted from its site spectrum.
