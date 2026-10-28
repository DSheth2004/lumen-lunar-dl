# Physics-Pretrained, Sample-Calibrated Deep Learning for Lunar Soil FeO, Mineralogy and Maturity from Moon Mineralogy Mapper Spectra

*[Author list, affiliations and correspondence to be completed by the authors.]*

**Article type:** Article — *Remote Sensing* (MDPI)

## Highlights

**What are the main findings?**

- LUMEN is a one-dimensional convolutional network. It is pretrained on 400,000 Hapke-simulated
  lunar soils and fine-tuned on 19 returned soils. From a spectrum on the Moon Mineralogy Mapper
  (M³) band set it predicts:
  - FeO, TiO₂, Al₂O₃, CaO and MgO;
  - a seven-part modal mineralogy;
  - soil maturity.
- On identical laboratory spectra and folds, it reduces FeO error by 42% relative to a published
  wavelet–autoencoder pipeline.
- Conformal calibration gives intervals with nominal coverage. An identifiability gate withholds
  the two targets that no method predicts better than a zero-information baseline (olivine and
  minor glasses).

**What are the implications of the main findings?**

- Most of the error of physics-only inversions is removed by calibrating against returned
  soils, not by adding model complexity.
- Laboratory skill does not transfer to orbital spectra without orbital labels. Fine-tuning on
  M³ spectra at returned-sample sites halves the error but does not make the deep model the best
  orbital FeO method: under leave-one-landing-site-out it reaches 3.27 wt% (R² = 0.34). That is
  lower than the published wavelet–autoencoder pipeline (3.58 wt%), though not significantly,
  and significantly higher than a linear regression on two M³ bands (2.11 wt%, R² = 0.74).

## Abstract

Retrievals of lunar surface composition from reflectance spectra fall into three groups:
- empirical band-ratio and machine-learning regressions calibrated on a few dozen returned-sample
  sites;
- radiative-transfer inversions;
- physics-guided neural networks.

They are rarely compared on identical data, and their uncertainties are seldom calibrated. We
present LUMEN (LUnar Mineralogy Estimation Network), a hybrid of physics simulation and deep
learning:
- a domain-randomised Hapke simulator, built from 49 lunar mineral and glass spectra and including
  nanophase-iron space weathering, generates 400,000 synthetic soils;
- a compact one-dimensional convolutional network is pretrained on them, then fine-tuned on
  laboratory spectra of the 19 Lunar Soil Characterization Consortium soils;
- a deep ensemble with conformal calibration supplies prediction intervals;
- an identifiability gate withholds outputs without demonstrable skill.

Under leave-one-soil-out evaluation, with the protocol fixed and committed before scoring, LUMEN
predicts FeO to 1.26 wt% and TiO₂ to 0.70 wt%. It reduces the FeO error of a re-implemented
wavelet–autoencoder pipeline by 42% (1.26 vs. 2.17 wt%) and the mineral-proportion errors of a
Shkuratov-model inversion by 54–74%. Conformal intervals reach nominal coverage. Olivine is not
identifiable from these spectra.

On M³ spectra at 49 returned-sample sites, a lab-only model does not transfer (FeO MAE 6.81
wt%). With orbital fine-tuning it reaches 3.27 wt% (R² = 0.34) under leave-one-landing-site-out:
below the published pipeline's 3.58 wt%, but not significantly (Holm p = 1.00). Two-band linear
regressions are significantly more accurate (M³: 2.11 wt%; Clementine: 1.82 wt%).

**Keywords:** lunar regolith; FeO; mineral abundance; space weathering; Moon Mineralogy Mapper;
Hapke model; deep learning; transfer learning; conformal prediction; uncertainty quantification

## 1. Introduction

Some properties of the lunar regolith can be read remotely, and they are central to lunar science
and to planning in situ resource use:
- its iron and titanium content;
- its modal mineralogy;
- its exposure to space weathering.

The Moon Mineralogy Mapper (M³) on Chandrayaan-1 acquired visible to near-infrared reflectance
spectra over most of the lunar surface [@pieters2009; @green2011]. These spectra carry the
crystal-field absorptions of Fe²⁺ in pyroxene and olivine near 1 and 2 µm, together with the
darkening and reddening produced by nanophase metallic iron. Turning such spectra into
quantitative composition remains an inverse problem with little ground truth.

**Empirical methods** relate spectral parameters to the composition of returned samples at their
landing sites.
- *Band ratios and band parameters.* The Clementine band-ratio algorithms of Lucey et al.
  [@lucey2000] remain the reference for FeO and TiO₂. Band-depth and band-centre parameters of
  the 1 and 2 µm absorptions have been calibrated for M³ [@bhatt2012; @bhatt2015].
- *Regression and machine learning.* Methods range from partial least squares [@sun2016] and
  neural networks [@xia2019] to support vector machines [@wang2021] and systematic comparisons of
  regressors [@qiu2022].
- *Deep learning with new ground truth.* Convolutional networks trained on Kaguya Multiband Imager
  spectra produced global oxide maps once Chang'e-5 [@yang2023; @zhang2023] and Chang'e-6
  [@shi2026] samples were added. Comparable approaches use Diviner thermal data [@ma2022],
  gradient boosting [@wu2024] and random forests [@fernandez2025].
- *The method this study builds on.* Fernández-Díaz et al. [@fernandez2026] combined a db4
  wavelet decomposition, a deep autoencoder and a random-forest regressor to produce a global M³
  FeO map at 150 m per pixel.

All of these methods are trained on about 50 sampling sites. Their accuracy is reported from
random splits or small hold-out sets, rarely on identical data alongside competing methods, and
rarely against a zero-information baseline.

**Physics-based methods** model reflectance with radiative transfer.
- *Models.* The Hapke model [@hapke1993] and its treatment of nanophase iron
  [@hapke2001; @lucey2011; @pieters2016] support inversions for mineral abundance, grain size and
  submicroscopic iron [@lili2011; @sunlucey2021; @trang2019].
- *Fast inversion with networks.* The Shkuratov model [@shkuratov1999] was inverted on M³ with a
  neural network trained on synthetic spectra [@korokhin2024].
- *Posteriors.* Hapke inversions have been embedded in Bayesian samplers to obtain posterior
  abundances [@jung2026].
- *Physics-guided unmixing.* PGU-Net [@lin2026] placed a Hapke single-scattering-albedo mixing
  layer inside a deep autoencoder for unsupervised unmixing. Its M³ results were validated
  indirectly, against Kaguya products, and the authors name sample-informed priors from returned
  materials as the next step.

Physics inversions are interpretable but non-unique, and they are sensitive to the end-member set
and to the optical-constant model. Laboratory studies of the Lunar Soil Characterization
Consortium (LSCC) soils have also combined spectra with measured mineralogy:
- hybrid partial-least-squares and neural-network models [@li2012];
- conversion of unmixing weights to mass fractions [@grumpe2018];
- multilayer-perceptron abundance mapping [@surkov2025].

**Our reproductions of three recent methods [@fernandez2026; @korokhin2024; @jung2026]**, made
independently of this study, show where these approaches fall short.
- *Korokhin et al. [@korokhin2024].* On its own synthetic test set, the network recovers
  orthopyroxene, olivine and the grey absorber with R² of only 0.06, 0.17 and 0.32. Yet these
  quantities are mapped.
- *Jung et al. [@jung2026].* Abundances drawn from the prior alone, using no spectral data,
  already meet two of the three published accuracy bounds.
- *Fernández-Díaz et al. [@fernandez2026].* The autoencoder lowers rather than raises regression
  accuracy: a random forest on the raw wavelet coefficients is more accurate than one on the
  autoencoder codes.

None of these points is a criticism of the physical models themselves. Each concerns how much of
a retrieved number is determined by the spectrum, and how that is tested.

**What is missing** is a method that does four things at once:
- uses physics where it helps, as a source of training data;
- lets returned soils correct the physics;
- states its uncertainty and which outputs it cannot identify;
- is evaluated against published physics-based and deep-learning pipelines on identical data,
  under grouped cross-validation, with a zero-information baseline.

To address this, we propose LUMEN (LUnar Mineralogy Estimation Network). The main contributions
are:

(1) **A hybrid method, LUMEN.** It joins physics simulation and supervised learning for lunar
soil composition.
- A domain-randomised Hapke simulator generates 400,000 synthetic soils. It uses 49 lunar mineral
  and glass end-members (none taken from an evaluation soil) and physically modelled nanophase-
  and microphase-iron weathering.
- A compact one-dimensional convolutional network is pretrained on them and fine-tuned on
  returned soils.
- From one spectrum the network predicts seven-part modal mineralogy, five oxides and maturity.
- A deep ensemble with conformal calibration supplies intervals with verified coverage. An
  identifiability gate withholds every target whose skill over a zero-information baseline is not
  demonstrable.

(2) **An identical-data, protocol-fixed comparison.** LUMEN is compared with three published
methods (wavelet–autoencoder regression [@fernandez2026], Shkuratov-model inversion
[@korokhin2024] and Hapke–Bayesian inversion [@jung2026]) and with classical regressors.
- All methods use the same 75 laboratory spectra and the same leave-one-soil-out and
  leave-one-mission-out folds.
- The tests are paired and corrected for multiple comparisons.
- Every evaluation choice was committed to version control before the first score was computed.
  Negative and null results are reported alongside the positive ones.

(3) **A test of laboratory-to-orbit transfer.** It uses M³ spectra at the returned-sample sites
on which the base method was calibrated [@fernandez2026; @qiu2025].
- It uses both that method's own cross-validation and a spatially honest leave-one-landing-site-out
  protocol.
- It shows that laboratory accuracy does not carry over to orbit without orbital labels, how
  much a small orbital fine-tuning set recovers, and that simple two-band regressions remain more
  accurate on orbit.

The remainder of this article is organised as follows:
- Section 2 describes the data, the physical preliminaries and the proposed method.
- Section 3 details the experimental settings and reports the results.
- Section 4 discusses them, including limitations.
- Section 5 concludes.

## 2. Materials and Methods

### 2.1. Data and Preprocessing

#### 2.1.1. Laboratory Spectra and Ground Truth

We use the LSCC suite [@taylor2001; @taylor2010]: nine mare soils (Apollo 11, 12, 15 and 17) and
ten highland soils (Apollo 14 and 16).
- **Spectra.** Each soil was sieved into <10, 10–20, 20–45 and <45 µm fractions. Each fraction
  was measured at the Reflectance Experiment Laboratory (RELAB) [@relab2004] at an incidence of
  30° and emission of 0°. The 10–20 µm spectrum of soil 70181 is absent from the archive, leaving
  75 spectra.
- **Modal mineralogy.** Point-count modal mineralogy (vol%) is available for the three finer
  fractions (56 spectra). It is merged into seven parts:
  - plagioclase (PLG);
  - low-Ca pyroxene (LCP: orthopyroxene plus pigeonite);
  - high-Ca pyroxene (HCP);
  - olivine (OLV);
  - ilmenite (ILM);
  - agglutinitic glass (AGG);
  - other (volcanic glass, K-rich phases and minor phases; OTH).
- **Chemistry and maturity.** Bulk FeO, TiO₂, Al₂O₃, CaO and MgO (wt%) and the ferromagnetic
  resonance maturity index Is/FeO are available for all 75 spectra.

The tables were transcribed from the published sources and checked against their own column sums
and totals (81 checks for the mare tables).

Every spectrum is convolved to the 71 M³ global-mode bands between 540.8 and 2497.1 nm, using
Gaussian responses with a full width at half maximum equal to the band spacing. These 71 bands
exclude the shortest-wavelength channels and the thermally affected channels beyond 2.5 µm.

#### 2.1.2. End-Member Library

The simulator draws on 49 RELAB spectra of lunar mineral separates and glasses: 17 plagioclase,
7 low-Ca and 11 high-Ca pyroxene, 6 olivine, 4 ilmenite, 2 volcanic and 2 synthetic lunar
glasses.
- They come from mare basalts, highland rocks, one non-LSCC soil (62241) and synthetic glasses.
- Every label was verified against its PDS4 metadata.
- A rule checked automatically at staging ensures that **no end-member derives from any of the 19
  evaluation soils**.
- Because every RELAB agglutinate separate comes from an LSCC soil, agglutinates are not taken
  from the library. They are modelled physically (Section 2.3.1).

#### 2.1.3. M³ Data

All orbital data are M³ Level-2 global-mode reflectance [@green2011]. Two kinds of M³ data are
used.

**Scenes for mapping:**
- the Shapley-region strip M3G20090728T214215 analysed by Korokhin et al. [@korokhin2024];
- a block around the Apollo 16 landing site (M3G20090607T110414);
- the two Aristarchus regions of interest of Jung et al. [@jung2026].

**Returned-sample sites.** The FeO ground truth of the base method [@fernandez2026] is the
compilation of Qiu et al. [@qiu2025jgr], available in open access as Supplementary Table 1 of
Qiu et al. [@qiu2025]. It lists 49 distinct, station-level sampling sites with laboratory FeO:
- Apollo 11, 12, 14, 15, 16 and 17 landing sites and traverse stations;
- Luna 16, 20 and 24;
- Chang'e-3, -5 and -6.

The table also gives Clementine reflectances at 415, 750, 900, 950 and 1000 nm for each site. We
verified it against the worked example in [@fernandez2026] (Apollo 17 LRV-12, 17.4 wt% FeO, at the
coordinates given in that paper).

**Site spectra.** For each site:
- every M³ strip whose nearest pixel lies within 0.01° of the site, and at least five samples from
  either swath edge, contributes the 3 × 3-pixel block around that pixel;
- pixels with any non-finite, non-positive or ≥1 reflectance are dropped;
- a strip contributes only if at least 5 of its 9 pixels remain;
- the site spectrum is the per-band median over the contributing strips.

All 49 sites are covered, Chang'e-5 included: 183 strips contribute, between one and six per site
(median four). Validity is assessed on M³ bands 3–85, the band set of [@fernandez2026].

Only the required image lines are downloaded, with HTTP range requests to the Planetary Data
System. Every input file is listed in a SHA-256 manifest, which each processing stage verifies at
start and end.

### 2.2. Preliminaries: Hapke Model, Single-Scattering Albedo and Space Weathering

#### 2.2.1. Reflectance and Single-Scattering Albedo

For an intimate particulate mixture we use the Hapke bidirectional reflectance [@hapke1993] in the
form

$$ r(w) = \frac{w}{4(\mu_0+\mu)}\,H(w,\mu_0)\,H(w,\mu), \tag{1} $$

where:
- $w$ is the single-scattering albedo (SSA);
- $\mu_0$ and $\mu$ are the cosines of the incidence and emission angles (30° and 0° for RELAB);
- the phase function is isotropic and the opposition effect is neglected, which suits RELAB
  geometry;
- $H$ is the approximate Chandrasekhar function,

$$ H(w,x) = \left[1 - w\,x\left(r_0 + \frac{1-2r_0x}{2}\ln\frac{1+x}{x}\right)\right]^{-1},\qquad r_0=\frac{1-\gamma}{1+\gamma},\ \gamma=\sqrt{1-w}. \tag{2} $$

Equation (1) is monotonic in $w$. It is inverted numerically from a dense table.

#### 2.2.2. Equivalent-Slab Grain Optics

The SSA of a grain population with absorption coefficient $\alpha$, real index $n$, imaginary
index $k$ and size $D$ follows the equivalent-slab model [@hapke1993]:

$$ w = S_e + (1-S_e)\,\frac{(1-S_i)\,\Theta}{1-S_i\,\Theta},\qquad \Theta = e^{-\alpha\langle D\rangle}, \tag{3} $$

with

$$ S_e = \frac{(n-1)^2+k^2}{(n+1)^2+k^2} + 0.05,\qquad S_i = 1.014 - \frac{4}{n(n+1)^2},\qquad \langle D\rangle = \frac{2}{3}\left[n^2-\frac{1}{n}(n^2-1)^{3/2}\right]D, \tag{4} $$

and $\alpha = 4\pi k/\lambda$ for a pure phase.

For each end-member we invert Equations (1)–(4) at the separate's nominal grain size to obtain
$k(\lambda)$.
- The reflectance is not monotonic in $k$: it falls to an opacity floor and then rises as surface
  reflection grows. The root is therefore bracketed on the physical, falling branch.
- After this correction, every end-member reproduces its own measured reflectance to
  < 10⁻¹⁰.

#### 2.2.3. Space Weathering

Nanophase metallic iron (npFe⁰) at volume fraction $\phi$ in a host of refractive index $n_h$ adds
an absorption [@hapke2001; @lucey2011]

$$ \alpha_{np}(\lambda) = \frac{36\pi\, z(\lambda)\,\phi}{\lambda},\qquad z = \frac{n_h^3\, n_{Fe}\, k_{Fe}}{\left(n_{Fe}^2-k_{Fe}^2+2n_h^2\right)^2+\left(2n_{Fe}k_{Fe}\right)^2}, \tag{5} $$

where $n_{Fe}$ and $k_{Fe}$ are the optical constants of iron. Microphase iron (mpFe⁰) is a grey
absorber, $\alpha_{mp} = 3\phi_{mp}/(4 r_{mp})$ with $r_{mp} = 0.1$ µm. It darkens without
reddening [@lucey2011].

#### 2.2.4. Intimate Mixing

Components with volume fractions $f_i$, grain sizes $D_i$ and SSAs $w_i$ mix in SSA with
cross-section weights:

$$ w_{mix} = \frac{\sum_i (f_i/D_i)\,w_i}{\sum_i f_i/D_i}. \tag{6} $$

The mixed SSA is then converted to reflectance with Equation (1).

### 2.3. Proposed Method: LUMEN

Figure 1 summarises the method. It has four stages:
1. a physics simulator generates labelled synthetic soils;
2. a network is pretrained on them;
3. it is fine-tuned on returned soils (and, for orbital use, on orbital spectra);
4. an uncertainty and identifiability layer decides what is reported.

![Figure 1](../figures/fig01_workflow.png)

**Figure 1.** LUMEN workflow. The simulator (left) produces labelled synthetic spectra from 49
lunar end-members with physically modelled space weathering. The network is pretrained on them,
then fine-tuned on laboratory spectra of returned soils, and optionally on M³ spectra at
returned-sample sites. A five-member ensemble with conformal calibration gives the intervals, and
the identifiability gate decides which targets are mapped.

#### 2.3.1. Domain-Randomised Physics Simulator

Each synthetic soil draws the following.

**Composition.**
- A petrological regime: feldspathic, basaltic or broad. Each is a Dirichlet prior on the
  crystalline proportions (PLG, LCP, HCP, OLV, ILM), truncated at ≤20% ilmenite and ≤60% olivine
  of the crystalline fraction.
- A maturity, which sets the agglutinate fraction (AGG = clip(0.65 · maturity + N(0, 0.12), 0,
  0.75)) and the npFe⁰ and mpFe⁰ loads. Agglutinates are a glass host (a mixture of a feldspathic
  and a mafic synthetic glass) carrying these iron loads.
- An "other" fraction.
- One end-member per mineral phase.

**Physical and instrumental variation.**
- Grain sizes between 4 and 150 µm.
- A calibration tilt, a ±7% scale factor and 0.2–0.8% noise.

**Labels.** Chemistry labels follow from stoichiometry. Each end-member carries a nominal phase
chemistry with ±15% jitter and molar Fe–Mg exchange. Agglutinates carry the bulk composition of
the non-agglutinate fraction.

**Maturity scale.** The pretraining maturity target is set by Is/FeO = 2270 · npFe⁰ (wt%)/FeO
(wt%). This constant fixes only the scale, which fine-tuning re-learns.

**Set size.** Four hundred thousand spectra are generated with a fixed seed.

#### 2.3.2. Network

**Input.** Two channels on the 71 bands: log reflectance, and convex-hull continuum-removed
reflectance. Each is standardised per band with statistics of the synthetic set only, so no
real-data statistic enters the input.

**Body.**
- Five one-dimensional convolutions with 32, 32, 64, 64 and 64 channels and kernel sizes 7, 5, 5,
  3 and 3.
- Each is followed by group normalisation (8 groups) and a GELU activation.
- Max-pooling after the second and fourth convolutions.
- Group normalisation replaces batch normalisation because fine-tuning batches are small.

**Trunk.** The 64 × 17 feature map is flattened and mapped through layers of 256 and 128 units
(GELU, dropout 0.1). The network has 0.36 M parameters.

**Heads.**
- A **Dirichlet head** outputs concentrations $\boldsymbol\alpha = \mathrm{softplus}(\cdot)+0.05$
  for the seven-part composition. Its mean $\boldsymbol\alpha/\alpha_0$, with
  $\alpha_0=\sum_j\alpha_j$, is the predicted composition.
- **Heteroscedastic Gaussian heads** output a mean and a log-variance for each oxide and for
  log₁₀(Is/FeO).
- **Auxiliary heads** for log grain size and npFe⁰ are used during pretraining only.

#### 2.3.3. Objective Functions

The composition loss is the Dirichlet negative log-likelihood of the label-smoothed proportions,
$\tilde{\mathbf y} = (\mathbf y+\epsilon)/(1+7\epsilon)$ with $\epsilon = 0.005$, plus an $L_1$
term on the mean:

$$ \mathcal L_{comp} = -\frac{1}{7}\left[\ln\Gamma(\alpha_0) - \sum_j \ln\Gamma(\alpha_j) + \sum_j(\alpha_j-1)\ln\tilde y_j\right] + \lambda_1\,\frac{1}{7}\sum_j\left|\frac{\alpha_j}{\alpha_0} - y_j\right|. \tag{7} $$

The weight $\lambda_1 = 10$ was fixed on synthetic data only.

The scalar targets use the β-negative log-likelihood [@seitzer2022], with $\beta=0.5$. It avoids
the under-fitting of high-variance points that the plain Gaussian likelihood causes:

$$ \mathcal L_{scal} = \frac{\sum_t m_t\,\mathrm{sg}\!\left[\sigma_t^{2\beta}\right]\tfrac12\left(\ln\sigma_t^2 + \frac{(y_t-\mu_t)^2}{\sigma_t^2}\right)}{\sum_t m_t}, \tag{8} $$

where:
- $\mathrm{sg}$ stops the gradient;
- $m_t$ masks targets that are unavailable for a sample (modal mineralogy for the <45 µm
  fraction; all targets except FeO for orbital spectra).

The total loss is $\mathcal L_{comp} + \mathcal L_{scal}$. During pretraining it also includes the
auxiliary β-NLL terms.

#### 2.3.4. Training

**Pretraining.** Five ensemble members are pretrained on the synthetic set with different seeds:
15 epochs, AdamW, a one-cycle learning-rate schedule, and 20,000 spectra held out for validation.

**Fine-tuning.** Each member is fine-tuned per cross-validation fold for 1500 steps: AdamW,
learning rate 3 × 10⁻⁴, weight decay 10⁻⁴, 100 warm-up steps, then cosine decay. The full
configuration (A4) draws each step:
- 16 real training spectra;
- 32 **SSA-space mixup** spectra. Each is a Dirichlet-weighted mixture of 2–3 training spectra,
  mixed with Equation (6) using their nominal fraction sizes. Labels are mixed exactly:
  composition by volume, chemistry by mass, and Is/FeO weighted by mass × FeO;
- 16 synthetic replay spectra.

Real and mixed spectra receive the simulator's calibration and noise randomisation. The simpler
configuration A2 uses the real spectra only.

All hyperparameters were fixed from synthetic-only development runs and committed before any
model was scored on real folds.

#### 2.3.5. Uncertainty and Identifiability

**Ensemble.** The ensemble mean is the mean of the member means. The ensemble variance is the mean
of the member variances plus the variance of the member means [@lakshminarayanan2017].

**Conformal intervals.** Raw ensemble intervals are over-confident (Section 3.5), so we calibrate
them conformally [@barber2021].
- The normalised score $s_i = |y_i - \mu_i|/\sigma_i$ is computed from out-of-fold predictions of
  the other soils.
- Its $\lceil (1-a)(n+1)\rceil$-th smallest value $q_{1-a}$ gives the interval
  $\mu \pm q_{1-a}\,\sigma$.
- The exact CV+ variant uses models from which both the calibration soil and the test soil were
  held out (930 additional fine-tunes).

**Identifiability gate.** For each target, the skill is

$$ S = 1 - \frac{\mathrm{MAE}}{\mathrm{MAE}_{ref}}, \tag{9} $$

where $\mathrm{MAE}_{ref}$ is the error of the fraction-matched mean of the other soils, a
zero-information reference. A 95% bootstrap interval is computed over soils. A target is mapped
only if the lower bound of that interval is above zero under leave-one-soil-out evaluation.

#### 2.3.6. Transfer to M³

Laboratory and M³ reflectance differ in absolute scale.

**Zero-shot application.** M³ spectra are brought to the laboratory scale with the linear
calibration of Korokhin et al. [@korokhin2024]:

$$ A_{lab} = 1.685\,A_{M^3} + 0.0355. \tag{10} $$

For a site of Apollo mission X, the zero-shot model is the ensemble fine-tuned with all of mission
X's soils held out. For the Luna and Chang'e sites it is the ensemble fine-tuned on all 19 soils.

**Orbital fine-tuning (LUMEN-M3-FT).** The network is further fine-tuned for 600 steps (AdamW,
learning rate 10⁻⁴, weight decay 10⁻⁴, cosine decay). Each batch holds:
- 16 M³ pixel spectra from the training sites, labelled with their site FeO, with every other
  target masked;
- 16 laboratory spectra from that model's own training soils, with all targets.

The held-out site is predicted from its site spectrum.
