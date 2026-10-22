# Jung et al. (2026) — implementation notes and deviations

**Paper.** Jung, J.-I., Lapôtre, M. G. A., Milliken, R. E., & Minson, S. E. (2026),
*Remote Compositional Analyses of Space-weathered Lunar Maria*, The Planetary Science
Journal **7**(1) 18. [doi:10.3847/PSJ/ae2b57](https://doi.org/10.3847/PSJ/ae2b57). Open access, CC-BY-4.0.

**Authors' code and data.** Jung, Lapôtre & Minson (2025), Zenodo,
[doi:10.5281/zenodo.17258920](https://doi.org/10.5281/zenodo.17258920), CC-BY-4.0.
Downloaded to `reference/space-weathering-main/`; archive MD5 `aaf8c4db0fda22ad8d550b88520f32f7`
matches the Zenodo record. Contains `hapkeRTM/hapkeRTM.py` (Appendices D/E + the band-depth
correction), `mcmc/` (the CATMIP MATLAB, adapted from Lapôtre et al. 2017), and the nine
end-member optical-constant files.

---

## The method in one page

**Step 1 — semiempirical band-depth correction.** A space-weathered spectrum is a linear
single-scattering-albedo mixture of a "depth-corrected" and a fully space-weathered component:

| Eq. | Expression |
|-----|------------|
| (1) | `ω_mix = θ_dc·ω_dc + θ_sw·ω_sw`, `θ_dc + θ_sw = 1` |
| (2) | `ω_dc = (ω_mix − θ_sw·ω_sw) / (1 − θ_sw)` |
| (3) | `θ_sw = I_sm/100 = C·Δ^3.731` |
| (4) | `Δ = [d_dc(970 nm) − d_mix(970 nm)] / [d_dc(970 nm) − d_sw(970 nm)]` |

`d(λ)` is the band depth of the continuum-removed **SSA** (not reflectance). Reference
wavelength 0.97 µm. For Apollo soils `d_dc = 0.358` (mean of pulverized fresh mare basalts
15058, 15555, 70017, 70035) and `d_sw = 0.120` (mean of 15041, 79221, 10084); `ω_sw` is the mean
SSA of those same three soils. For orbital data `d_dc` is unchanged but `d_sw`/`ω_sw` come from
the most space-weathered pixels near the ROI.

**Step 2 — Bayesian Hapke unmixing.** Unmix `ω_dc` into 9 end-members × (mass abundance, grain
size) = 18 parameters with CATMIP; then forward-model the fresh spectrum with the agglutinate
contribution removed. Appendices B (priors/likelihood), C (mixing), D (SSA ↔ reflectance),
E (optical constants).

**Measurement geometry.** i = 30°, e = 0°, g = 30°. Confirmed two ways: every one of the 22
RELAB PDS4 labels records incidence 30 / emission 0, and `hapkeRTM/examples.ipynb` instantiates
`HapkeRTM(emergence_angle=0, incidence_angle=30, phase_angle=30)`.

---

## Which spectrum is which

The paper names samples; the data are RELAB spectrum IDs. Resolved from the PDS4 labels and
confirmed against Table 1 and Table C1 (all 22 verified in `step00`, with a shuffled negative
control). This also settles a question the paper leaves open — the LSCC soils enter as
**specific RELAB size fractions**, not as the `<45 µm` bulk column of the LSCC `*Kdata.txt`
files.

| Role | RELAB ID | Code | Sample | Size (µm) |
|------|----------|------|--------|-----------|
| fresh basalt (→ `d_dc`) | c1lr152 / c1lr154 / c1lr156 / c1lr158 | LR-CMP-152/154/156/158 | 15058 / 15555 / 70017 / 70035 | 0–125 |
| space-weathered (→ `d_sw`, `ω_sw`) | c1lr15 / c1lr12 / c1lr17 | LR-CMP-015/012/017 | 10084 / 15041 / 79221 | 20–40, 20–44, 20–40 |
| **inverted (Table 1, Figs 3/4/F1)** | c1lr10 / c1lr32 / c1lr20 / c1lr57 | LR-CMP-010/032/020/057 | **12001 / 12030 / 15071 / 71061** | 20–44 / 20–44 / 20–40 / 20–45 |
| other mare soils (Fig. 2) | c1lr23 / c1lr28 | LR-CMP-023/028 | 70181 / 71501 | 20–44 |
| end-members (Table C1) | c1lr169, c1lr176, c1lr179, c1lr177, c1lr178, c1sc53, c1sc50, c1lr51, c0ls152 | — | 15555 / 70017 / 70035 / 74220 / 79221 / synthetic glasses | — |

---

## Deviations found

Ordered by how much they matter. Everything here is evidenced by a table in `tables/`.

### D1 — Eq. (E9) is misprinted in the paper *(material)*

The paper prints the photon mean free path as

```
<D> = (2/3)[ n² + (1/n²)(n² − 1)^(3/2) ] D
```

Both released implementations — `hapkeRTM.py::_D_mean` and `mcmc/nkDtowi.m` — instead use

```
<D> = (2/3)[ n² − (1/n)(n² − 1)^(3/2) ] D
```

which is the standard Hapke (1993, eq. 6.20) form. Two independent files agreeing against the
printed equation makes this a typesetting error, not a modelling choice. It is not cosmetic:
the two forms differ by up to **0.244 in SSA** across the nine end-members
(`tables/step01_eq_E9_discrepancy.csv`). **We use the code form.**

### D2 — Eq. (3) coefficient: 65.68, not 65.58 *(minor, resolved)*

Eq. (3) gives `θ_sw ≈ 0.6568·Δ^3.731` (i.e. `I_sm = 65.68·Δ^3.731`); the Fig. 2 annotation reads
`I_sm = 65.58 × Δ^3.731`. Scored against Table 1's four calculated `I_sm` values, **65.68 wins**
(mean |err| 0.080 vs 0.118). The figure annotation is the typo. `tables/step02_eq3_coefficient.csv`.

### D3 — the released optical constants disagree with Table C1's grain sizes *(material)*

`HapkeRTM.k()` recovers `k(λ)` at a stated grain size. Our `k`, derived at Table C1's `D_used`,
round-trips to the measured reflectance to **5e-9**. The authors' shipped `k` files do **not**
reproduce those same measured spectra at `D_used` — they only do so at an effective grain size
4–9× smaller:

| end-member | `D_used` (Table C1) | effective D of released `k` | ratio |
|---|---|---|---|
| olivine | 80 µm | 17.1 µm | 4.69× |
| plagioclase | 80 µm | 9.4 µm | 8.53× |
| pigeonite | 80 µm | 16.5 µm | 4.83× |
| augite | 80 µm | 16.2 µm | 4.93× |
| ilmenite | 80 µm | 19.6 µm | 4.08× |
| pyroxene glass | 50 µm | 12.2 µm | 4.09× |
| plagioclase glass | 50 µm | 12.2 µm | 4.09× |
| orange glass | 150 µm | 17.3 µm | 8.66× |
| agglutinate | 150 µm | 21.9 µm | 6.86× |

`k` and `D` enter the model very nearly as the product `k·D`, so the released constants and the
released grain-size table are internally inconsistent by that factor. The *published results*
were produced with the released `k`, so that is the **primary** library here; ours is carried as
a variant and both are run. `tables/step03_effective_grain_size.csv`.

### D4 — agglutinate `n` *(minor)*

Table C1 prints `n = 1.49` for agglutinates; the released `agglutinate.txt` uses **2.16**. The
other eight match Table C1 exactly. `jung_common.ENDMEMBERS` carries both as `n_paper`/`n_code`.

### D5 — end-member densities *(minor)*

Table C1 vs `load_data.m`: ilmenite **4270 vs 4720** kg m⁻³ (digit transposition), orange glass
**3000 vs 3300**. The other seven agree. Both carried as `rho_paper`/`rho_code`.

### D6 — the likelihood covariance is a scalar, not a matrix *(minor)*

Appendix B2 describes a full `2N_d × 2N_d` covariance `C_x`. `mcmc/likelihood.m` uses
`L = -0.5·resᵀ·Cinv·res` with `Cinv = 1/Cov`, `Cov = 2e-4` — i.i.d. Gaussian errors with
σ = 0.0141 in SSA. We follow the code.

### D7 — the released CATMIP proposal degenerates *(material — affects the posteriors)*

`catmip_metro_sample_dirichlet.m` proposes abundances as a **fresh independent draw from the
Dirichlet prior** at every Metropolis step, and sets the grain-size proposal covariance to

```
Sm0 = Sm;  Sm = eye(Nparam);  Sm(1:2,1:2) = Sm0(1:2,1:2);
```

so only the first two grain sizes use the adaptive covariance and the other seven random-walk
with unit (1 µm) variance. Reproducing this faithfully, on a synthetic mixture whose answer we
know:

- acceptance decays to **0** by cooling stage 15;
- the final ensemble holds **2 unique particles out of 2000**;
- with the ensemble degenerate, `LLK − median(LLK) ≡ 0`, so `catmip_calc_beta` sees COV(w) = 0 ≤ 1
  and sets `dβ = 1 − β` — β jumps 0.08 → 0.30 → 0.54 → 1.0 in three steps;
- the returned "posterior" is therefore not a posterior, and the reported abundances sit close to
  the prior means.

This is not a transcription error on our side: the synthetic problem **is** identifiable
(log-likelihood −191.8 at the truth against −2260 for a swapped augite/pigeonite mixture, and a
local optimisation from the truth recovers the major phases). Adding a proper random walk on the
simplex, `m' ~ Dirichlet(conc·m)` with the Hastings correction, restores healthy behaviour
(acceptance ≈ 0.25, 98% unique particles) and, at sufficient chain length, reaches the truth
likelihood exactly.

`catmip.py` therefore offers three proposal modes — `released`, `adaptive`, `rw` — and both the
faithful and the corrected runs are reported. See `tables/step04_*`.

### D8 — chain length

`load_data.m` ships `Nm = 2000` (ensemble) and `Nsteps = 10` (Metropolis steps per cooling
stage), which with ~50 cooling stages is the paper's "Markov chain length of 10⁶". On the
synthetic test that is **not enough to converge**:

| `Nsteps` | evaluations | max log-likelihood | R² | truth in 95% CI |
|---|---|---|---|---|
| 10 | 4.0e5 | −209.9 | 0.9575 | 0/9 |
| 40 | 1.4e6 | −204.3 | 0.9587 | 1/9 |
| 100 | 3.2e6 | **−191.7** (truth −191.8) | **0.9612** (= ceiling) | 5/9 |

### D9 — fit window

`load_data.m` fits 0.55–2.50 µm; the figures are drawn 0.5–2.5 µm. We use 0.55–2.50 µm.

### D10 — `d_dc` convention

`HapkeCorrection` hard-codes `d_dc = 0.6422`, a continuum-removed *value*; Table 1 quotes
`0.358`, the corresponding *depth* (1 − 0.6422 = 0.3578). Eq. (4) is invariant under the swap.
We work in depths.

---

### D13 — the calculated `I_sm` disagrees with the maturity it estimates *(reported, not gated)*

Eq. (3) returns a soil maturity index `I_sm` calibrated against Morris (1978) `Is/FeO`. The LSCC
tables supply the measured `Is/FeO` for exactly these soils and sieve fractions, so the
comparison can be made directly:

| soil | `I_sm` calculated | measured `Is/FeO`, 20–45 µm | measured, <250 µm |
|---|---|---|---|
| 12001 | 43.0 | 51 | 56 |
| 12030 | 31.3 | 12 | 14 |
| 15071 | 32.1 | 49 | 52 |
| 71061 | 43.6 | **9** | 14 |

Worst difference **34.6 units** (71061), and the rank ordering is not preserved under either
convention: 71061 is the *most* mature of the four by the calculated index and the *least* mature
by measurement, a full inversion. 12030 and 15071 likewise swap.

Two caveats, stated so this is not read as more than it is. First, this is **not** an arithmetic
failure on our side: `step02` reproduces Table 1's own calculated `I_sm` to 0.28, so the paper's
numbers are being reproduced, not contradicted. Second, Fig. 2 is a fit through scattered points
and does not claim per-soil accuracy — part of what is tabulated above is that published scatter
seen four points at a time. What it does show is that for the four soils the paper carries forward
into Step 2, the index used to set `θ_sw` is not tracking the measured maturity of those soils.

Widening the test past the four inverted soils: across **all six mare soils** that have both a
Δ and a measured `Is/FeO` — and which are independent of the calibration, because the three soils
that *define* `d_sw` are not among them — the correlation is
**r = +0.15 (p = 0.78)** at 20–45 µm and **+0.15 (p = 0.78)** for the <250 µm reference. The sign
is right but the relation is not detectable. With n = 6 this bounds the relation rather than
refuting it: a null at this sample size is weak evidence on its own, though a genuinely strong
correlation would have been unlikely to produce r = 0.15.

Figure A1(c) makes the same point from the other direction: the correlation between 970 nm band
contrast and measured `Is/FeO` across the nine LSCC mare soils is only **−0.27**, and no
wavelength in 0.60–2.40 µm exceeds **−0.38** (at 0.91 µm). Note this measures raw contrast, which
Eq. (4) deliberately normalises — so it does not by itself contradict the paper, but it does show
the 970 nm band is not strongly coupled to maturity in this sample set.

Reproduced by `tables/step07_maturity_check.csv` and
`tables/step07_maturity_correlation.csv`; drawn as the green stars in
`figures/fig02_ism_vs_delta.png` and panel (c) of `figures/figA1_paper.png`.

### D15 — our weathered-pixel selection partly tracks detector striping *(caveat on our ROI rule)*

`figures/fig05_paper.png` shows the 970 nm band-depth map for both staged blocks. The fresh
selection behaves as intended, tracing crater rims and ejecta. The weathered selection, however,
visibly follows **vertical column striping** in the band-depth map — an M3 detector artefact, not
mature regolith. Because our rule takes the shallowest-band pixels scene-wide rather than the
paper's hand-drawn circle beside each crater, it preferentially picks the striped columns.

This is a plausible contributor to our `d_sw` being low against the published values (0.052 vs
0.083 at ROI b, 0.034 vs 0.061 at ROI c), which in turn drags `I_sm` down (3.4 vs 6.4, 14.1 vs
20.6). Those quantities are already reported-not-gated for exactly this reason — the hand-drawn
ROIs are not recoverable from the publication — but the striping gives a concrete mechanism rather
than leaving the discrepancy unexplained. The gated Fig. 5 targets use the closed-form evaluation
on the paper's *own* band depths and are unaffected.

### D14 — the mineral-to-oxide conversion is not specified; drawn under our own assumption

Section 3 states that "inverted mineral compositions were converted to oxide abundances", plotted
as Figure 6(i),(j) in (SiO₂, FeO+MgO) space against Apollo mare-basalt fields from ApolloBasaltDB
(Cone et al. 2020). The paper gives **neither the oxide composition assumed for each of the nine
end-members nor the conversion itself**, Table C1 carries only ρ, n and D, and the released code
has no such routine — `wtperctof.m` converts weight fraction to cross-section fraction, not to
oxides.

The panel is drawn anyway, on an input **we chose and document**: measured Apollo mineral
chemistry from LSCC Table 4 (high-Ti mare soils, 20–45 µm, averaged over all five soils it
covers), mapped onto the nine end-members and renormalised to 100 wt%. Full construction, the
column mapping, and its limitations are in `data/ground_truth/provenance.md`; the table itself is
`data/ground_truth/endmember_oxides.csv`. Because the input is ours and not the paper's,
**everything derived from it is reported and never gated**.

What it shows, with that caveat firmly attached:

| | SiO₂ (wt %) | FeO + MgO (wt %) |
|---|---|---|
| ROI b, MAP | 38.8 | 33.4 |
| ROI b, posterior median | 44.0 | 31.4 |
| ROI c, MAP | 37.4 | 34.7 |
| ROI c, posterior median | 37.0 | 34.9 |
| LSCC mare soils, measured bulk (Table 3) | 38.4–46.1 | 24.0–29.3 |

Our inverted compositions sit at the **low-SiO₂, high-FeO+MgO edge of, or outside, the real mare
range** — 31–35 wt% FeO+MgO against a measured 24–29. That is exactly the direction an
over-predicted ilmenite drives the chemistry (ilmenite is ~43 % FeO and ~0 % SiO₂), so the oxide
panel is an independent expression of the same defect D12 identifies from the modal comparison,
arrived at through a different quantity. The paper reports the opposite — that its inverted
mineralogies "significantly overlap with ranges observed at Apollo landing sites".

Two reasons this is suggestive rather than conclusive. The comparison population is **not** the
paper's: ApolloBasaltDB was not obtained, so the reference points are LSCC mare *soils*, a
different population from Apollo mare *basalts*. And the oxide table is high-Ti mare mineral
chemistry, appropriate for these Aristarchus ROIs but still an assumption. A like-for-like test would need the Apollo mare basalt compilation
(ApolloBasaltDB) the paper uses.

Per-sample oxide values: `tables/step09_oxides_roi_b.csv`, `tables/step09_oxides_roi_c.csv`.

### D12 — two of the three stated accuracy claims are met by the prior alone *(material)*

Section 3 supports the method with three numbers: olivine agrees with the LSCC lab values to
better than 7 %, ilmenite typically to better than 10 % (15071 excepted), plagioclase to 20–30 %.
`step07` scores these against a **no-information control**: the Dirichlet abundance prior on its
own, renormalised to the crystalline phases exactly as Figure 4 does, with the spectra never
consulted. Worst-case error over the four soils, in percentage points:

| claim | stated bound | converged sampler | released sampler | **prior alone** |
|---|---|---|---|---|
| olivine | < 7 | 5.78 | 7.37 | **1.76** |
| ilmenite (15071 excepted) | < 10 | 17.56 | 12.89 | **10.87** |
| plagioclase | 20–30 | 16.83 | 22.56 | **16.92** |

The prior mean for olivine (0.07) renormalises to 6.89 % of the crystalline phases, and the four
LSCC mare soils happen to sit at 7.7–8.7 % olivine — so the prior alone reproduces the olivine
claim to 1.76 points, **three times better than either sampler achieves**. For plagioclase the
prior alone (16.92) matches the converged sampler (16.83) and beats the released one (22.56). Passing those two bounds is therefore not evidence that the inversion extracted
mineralogy from the spectrum; the bounds are too loose to distinguish the posterior from the prior
it started at.

Ilmenite is the one category where the data must do work, and it is the one that fails: no
configuration reproduces the < 10 % claim (17.56 converged, 12.89 released), and the failure is
robust to the sampler, so it is not an artefact of our D7 correction. Under the converged sampler
the worst ilmenite soil is 15071 at 22.13 points — the very soil the paper excepts — but 12001 at
17.56 also breaks the claim, so the exception does not rescue it.

This is the reproduction's main negative result, and it compounds D7: if the released ensemble
does not mix (D7), the reported abundances stay near the prior, and an accuracy claim that the
prior already satisfies cannot detect that.

Reproduced by `tables/step07_prior_baseline.csv`; the control uses a fixed seed and 200 000 draws.

### D11 — an arithmetic erratum in the LSCC source tables *(minor, does not affect any gate)*

LSCC Table 2 prints orthopyroxene = 0.82 for 12001, 20–45 µm; the four pyroxene sub-sets then sum
to **18.88** against its own printed "Total Pyroxene" of **19.88**. The 0.82 was re-read from the
PDF in layout mode to rule out a digit lost in text extraction — it is printed as 0.82. LSCC
Table 1 independently gives 19.9, agreeing with the printed total, so the total is taken as
correct and the sub-set split for that one cell as unreliable. `step07` compares pigeonite +
augite as a single "pyroxene" category exactly as Fig. 4 does, so no gated target uses the split.
Two further cross-table differences (71061 and 79221, both 20–45 µm) exceed 2-decimal rounding at
≈0.3 points and are recorded as `ok`. All 81 checks are in
`data/ground_truth/lscc_transcription_checks.csv`.

---

## Data sources

- **LSCC modal mineralogy and Is/FeO** (the stars and grey bars of Figs 4 / F1). Hosted at
  `pgi.utk.edu`, which was unreachable at the time of retrieval; the RELAB LSCC page
  <https://sites.brown.edu/relab/lscc/> carries only the `*Kdata.txt` spectra and links to the
  same host. The three tables were retrieved through the Internet Archive with the `id_`
  (unrewritten) suffix. Provenance, the transcription and its 81 consistency checks:
  `data/ground_truth/provenance.md`.

  The four inverted spectra are all the **20–45 µm** sieve fraction (from the RELAB PDS4 labels in
  `tables/step00_relab_metadata.csv`), so the matching LSCC column is fixed by the data rather
  than chosen to fit.
- **M³ over Aristarchus** — L2 reflectance (PDS Imaging Node, volume CH1M3_0004) and L1B location
  backplanes (CH1M3_0003) for two images, M3G20090208T160125 (ROI b) and M3G20090208T233940
  (ROI c), staged by HTTP range request (`tables/step00b_m3_provenance.csv`). The L2 data carry
  the standard PDS thermal correction; the Li & Milliken (2016) correction the paper names in its
  Figure 1 caption is not applied (see `LIMITATIONS.md`).
