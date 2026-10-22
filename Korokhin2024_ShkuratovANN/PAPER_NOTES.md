# Paper notes — Korokhin et al. (2024)

*Applying machine learning to a nonlinear spectral mixing model for mapping lunar soils
composition using CHANDRAYAAN-1 M³ data.* V. Korokhin, Y. Surkov, U. Mall, V. Kaydash,
S. Velichko, Y. Velikodsky, O. Shalygina. Planetary and Space Science 244 (2024) 105870.
doi:10.1016/j.pss.2024.105870.

This file records (1) what the paper specifies and how it was implemented, (2) every
place where a choice had to be made because the paper does not say (**D**-numbers), and
(3) the findings (**F**-numbers), with the evidence. Numbers quoted here come from
`tables/`; `RESULTS.md` is the scorecard.

---

## 1. The method in one page

**Forward model** (Appendix, Eqs. A1–A16) — Shkuratov et al. (1999), one-dimensional
geometric optics of a particulate surface:

- reflection/transmission coefficients from the real index *n* (A3–A7):
  R_e = r₀ + 0.055, R_b = (0.28n − 0.20)R_e, R_i = 1.04 − 1/n², T_e = 1 − R_e, T_i = 1 − R_i;
- particle albedos r_b, r_f from the optical density τ between internal reflections (A8);
- a layer's indicatrix ρ_b = q r_b, ρ_f = q r_f + 1 − q (A10); surface albedo
  A = P − √(P² − 1), P = (1 + ρ_b² − ρ_f²)/2ρ_b (A11);
- the inverse (A2) turns a lab spectrum into τ, and k = τλ/4πS (A1);
- intimate mixture: ρ_b, ρ_f are concentration-weighted sums (A12);
- space weathering: τ_mat = τ + α_Fe(λ) c_nFe d, d = 0.2 µm, with
  α_Fe = 6π(n/λ) Im{(m² − 1)/(m² + 2)}, m = ñ_Fe/n (A14–A16).

**Eight free parameters** (Sec. 4.1): grain size S (common to the four transparent
minerals), c_nFe, and the concentrations of the gray component, CPX, OPX, OLV, PLG and ILM.
Fixed: q = 0.5, n = 1.6 (transparent) or 2.0 (ilmenite, gray), gray albedo 0.15, basis
spectra converted to k at S = 30 µm.

**Inverse #1** — Nelder–Mead on the RMS deviation, with an exhaustive "corridor" search
over the starting grain size (Sec. 2.3). **Inverse #2** — an MLP 21-42-12-8 trained on
synthetic spectra from the forward model, then retrained twice on the scene's own
retrieved parameters (Sec. 5). The paper's case for the ANN is speed and "removing
ambiguities".

**Verified exactly** (`step01`): our transcription of A2 gives Table 5's six τ_Gr values
to the printed rounding (worst 0.00037); A2 inverts A8–A11 to 1.5 × 10⁻¹³; a lossless
medium gives albedo 1; every basis spectrum survives lab → k → mixture → albedo to
10⁻¹⁴; the computed α_Fe matches the paper's Fig. A1 to ~3 % below 1.94 µm.

---

## 2. Deviations and choices (D)

| # | what the paper leaves open, or what we could not obtain | what we did | evidence / sensitivity |
|---|---|---|---|
| D1 | Table 4 gives RELAB **sample** IDs, not spectrum IDs | resolved from the local PDS4 labels: c1lr215/220/212/217/218, all "D <45 µm" | `step00_identity_check.csv`, 23/23 match; shuffled control rejects 23/23 |
| D2 | the "improved M³ data" of Shkuratov et al. (2019) and Surkov et al. (2020, 2021): Gaussian λ-convolution + Fourier filtering of stripes. The recipe is reprinted in Surkov's PhD thesis (Göttingen, chs. 2 and 4) | follows the thesis: λ-convolution on the native bands, FWHM 3 band pixels (Surkov 2020), bands ≤ 2.6 µm, ends extended by odd reflection; 2-D Fourier filtering of each channel in reflectance with C = 0 inside Ω = the stripe comb, located in the power spectrum by a numeric rule fixed before any downstream result (950-nm channel: rows |k_line| ≤ 3 of 888, cross-track periods < 12 samples); linear resampling to the 21 points; Eq. 2 | the thesis's own acceptance test (every channel mean within 0.2 %) cannot be met on this scene by any kernel ≥ 1.5 px (the mean spectrum curves at 580–750 nm; worst 1.1 % at 581 nm); the comb is taken as the contiguous band of excess power from the shortest periods down, as the thesis describes (the lowest-frequency band also shows an excess, from the Crisium margin rather than stripes, and is excluded). Sensitivities reported side by side: `no_destripe` (no Fourier step) and `approx_v1` (a simpler approximation: one 0.1-µm kernel + log-domain notch on k_line = 0, periods < 25), whose full run is kept in `archive/v1_approx_preprocessing/` |
| D3 | Fe optical constants beyond 1.94 µm (Weaver et al. 1979) not openly available | Johnson & Christy (1974) + Ordal et al. (1988), joined continuously. Pre-registered rule: if the computed α_Fe leaves ±15 % of the digitized Fig. A1 over 0.54–2.54 µm, Fig. A1 itself is used | worst deviation 21 % (above 1.94 µm, Ordal region) → **the digitized Fig. A1 is primary** (`step01_alpha_fe_choice.csv`) |
| D4 | wavelength range / sampling of the lab fits (Tables 1, 7) | 0.30–2.60 µm on the native 5-nm RELAB grid, ⟨A⟩ over the same | the plain mean of C2LS45 over that range is 0.05674; Table 7's σ/σ_Rel implies 0.05677. Other ranges miss by ≥ 0.0025 |
| D5 | whether concentrations are forced to sum to 100 % in Table 1 | free, renormalised for the "restored proportions" | mixtures 1–2 come out at 99.96 / 99.93 % unforced; forcing 100 % cannot fit mixture 3 (σ_Rel 7.4 % vs 1.0 %), so the paper's fit must also have been free |
| D6 | Nelder–Mead starting values, facets, bounds, tolerances | start nFe 0.05, gray 30 %, minerals 10 %, PLG 20 %; facets S 1 µm, nFe 0.02, conc. 10 %; all parameters ≥ 0 | PLG 20 % because Table 10's 20 % row is identical to Table 9 site 1; ≥ 0 because the printed fits contain exact 0.00 |
| D7 | Eq. 2 re-derivation: which pixels, which 62231 fraction | 5 × 5 pixels at the Apollo 16 LM site in M3G20090607T110414; LSCC 62231 <45 µm, taken to the native M³ bands and through the same recipe; the other three fractions and the approx_v1 processing reported beside it (`step05_eq2_sensitivity.csv`) | slope reproduces, intercept does not, under every variant (F5) |
| D8 | the mare / highland split used for the branch statistics | Otsu threshold on ⟨A⟩ (0.217; 39 % highland) | — |
| D9 | test-site coordinates | Fig. 3 crosses mapped through the scene registration (step04), then the pixel within ±8 px whose spectrum best matches the digitized Fig. 12; the optimum is identical for ±8 and ±12 at all four sites | ⟨A⟩ within 0.4–3.7 % of the value implied by the tables; the refinement never sees the fitted parameters |
| D10 | "the hidden layers contain a bias neuron" | a bias on every hidden and output neuron | — |
| D11 | stopping rule is interactive ("decided by the researcher") | the argmin of scene-mean σ_Rel (the paper's Fig. 15b rule); iteration 1 evaluated every epoch to 500; iterations 2–3 every 10 epochs, stopped 1,500 epochs after the best or at 10,000 | paper optima 66 / 5,336 / 4,256 |
| D12 | weight initialisation, sample order, warm vs fresh start per iteration | uniform ±0.5; reshuffled each epoch; fresh weights each iteration; 5 seeds | seed spread reported |
| D13 | "small random disturbances, less than ±0.5 %" | multiplicative, uniform in ±0.5 % | — |
| D14 | (our addition) comparing retrieved and measured LSCC mineralogy | mineral proportions within the mineral subtotal; CPX ↔ Mg+Fe clinopyroxene, "OPX" ↔ orthopyroxene + pigeonite | step10 |
| D15 | MLR coefficients (Eqs. 3–5) are not printed | Eq. 3 is Eq. 1 of Shkuratov et al. (2005a) and Eq. 2 of Pieters et al. (2006), the two sources the paper cites for it; both print coefficients (log10, A in %). Used as published: **Is and OPX** from Pieters 2006 Table 3, **CPX and PLG** from Shkuratov 2005a Table 3; Is/FeO (both sources) and augite reported beside them. The reading is confirmed on our 56 LSCC fraction spectra: r = 0.93 for Is (Pieters: 0.92), 0.84 for PLG and Is/FeO, median error 0.07–0.09 dex. OLV (Eq. 5) and d (Eq. 4) have no published coefficients and are refitted on all 19 LSCC soils (mare + the highland soils of Taylor et al. 2010); the mare-only refit is reported beside it. Our LSCC Clementine-band values agree with the albedos Korokhin et al. (2008) Table 1 prints for the same fractions within 0.25 % (median, all five bands) | C_nFe_Is_mare/high and C_PLG_mare gated (tolerance unchanged, 0.10); all variants in `step09_correlations_variants.csv` |
| D16 | "bimodal" (Fig. 20) | ≥ 2 peaks with prominence ≥ 10 % of the maximum in a 60-bin, 3-bin-smoothed histogram | — |
| D17 | scene extent | 500 square-pixel rows (the paper's 152,000 = 304 × 500) centred on the Fig. 3 registration; the figure's own frame spans ~480 rows | — |
| D19 | refractive index of the agglutinate when used as a basis spectrum (Table 8) — not stated | n = 1.6: with n = 2.0 its 0.54-µm albedo (0.0405) is below the darkest value that index can represent (0.0431), so Eq. A2 has no solution | with n = 2.0 the fit cannot start |
| D18 | grain-size corridor per site | 1–100 µm in 1-µm steps plus 105–500 µm in 5-µm steps at every site ("up to 500 µm for young areas" — which site is young is itself a result) | Table 9 site 1 prints S = 138.7 µm |
| D20 | Fig. 31: the 1550-nm band-depth ilmenite of Surkov et al. (2020) | the thesis reprints it (ch. 4): upper convex hull over the native spectrum (541–2577 nm, M³ L2 reflectance), S = 1 − A/A_CH, BD = mean S at 1489/1509/1529/1549/1579 nm, ilmenite wt% = 202.8 BD | the five M³ channels match the published wavelengths exactly |
| D21 | LSCC highland ground truth (not in the mare-soil LSCC tables) | Taylor et al. (2010) JGR 115 E02002, NASA NTRS preprint 20100006633 (public domain): modal Tables 2–3, chemistry and Is/FeO Table 4, transcribed in `scripts/lscc_highland_taylor2010.py` | 100 arithmetic checks pass (oxide totals exceed the listed oxides by +0.06 on average — an unlisted component, one-signed); Al₂O₃/CaO/FeO/MgO equal Shkuratov 2005a Table 1 in 13/13 rows (shuffled: 1, the unmoved row); Table 2 and Table 3 pyroxene disagree in 4/30 columns in the source itself |
| D22 | Fig. 30: "a moderately nonlinear ANN (Korokhin et al., 2008)" — that network (6-3-1, α = 1.0, 500 epochs) has no published weights | retrained on Korokhin 2008 Table 1 (76 LSCC rows, transcribed in `scripts/korokhin2008_tio2.py`) with its published standardization (Table 2) and output scaling; no bias (as every published layer), per-sample back-propagation, learning rate 0.1, momentum 0.9, 5 seeds, map = seed median; applied to the M³ Clementine-like bands as fractions | gate: median k on Table 1 = 0.940 (published 0.95 ± 0.03). The standardization reading is proved by Table 2 itself (for AR and CBR, x0 = mid-range and xk = 2/range of Table 1 to four digits; for CIR1–3 and D they are not — set on another sample set). **The published weights (Tables 3–5 and 6–9) do not reproduce the published fits:** best k = 0.40 (Tables 3–5, published 0.99) and 0.70 (the two-net system) under every reading; no bias terms are printed, which is the likely cause. They are reported, not used |

---

## 3. Findings (F)

### F1 — The paper's scene and test sites, recovered from its own figure

The paper names only its calibration image. Its test area is **M3G20090728T214215**
(OP2C, 28 July 2009), raw lines 13918–14805, resampled along-track to 500 square rows.

- Of five ODE candidates, only two actually contain Shapley's centre; the other three
  miss it by 0.15–1.6°.
- Normalised cross-correlation with Fig. 3: 0.849 vs 0.766 for the runner-up
  (M3G20090201T085853); the pre-set margin (≥ 0.05) is met.
- **Independent check:** the strip's geolocated Shapley centre falls at Fig. 3 raster
  (100, 647), 4 px from the centre of the large crater at lower left (97, 650). In the
  runner-up it falls at (445, 706), open highland.
- Pixel shape: 261 m × 147 m. Resampled to square pixels, the figure's frame is ~480
  rows — which is how 304 × 500 = 152,000 arises. (The pixel-count check does *not*
  discriminate between the two strips; it is reported, not relied on.)

### F2 — Table 1 prints the olivine and pyroxene columns the wrong way round

Read as printed, our restored proportions disagree with the paper's by up to 17 points
(mixture 3 olivine: 23 vs 40). Read with the first two columns swapped, **all eight agree
within 2.5 points** (`T1swap_*`). The swap is corroborated by RELAB's own label for the
mixture-3 spectrum, C1SC07: *"OLV:PYX:MET 30:40:30"*, whereas the paper gives the source
proportions as 40:30:30. Our end-member assignment follows the paper's captions and the
RELAB labels (C1MS14 = olivine, C1MS15 = pyroxene).

Consequence: the size of the errors is unchanged but the mineral is not. As printed,
mixture 3 recovers olivine perfectly (40 vs 40) and under-recovers pyroxene; in fact the
pyroxene is recovered (38 vs 40) and **olivine is under-recovered (23 vs 30)**.

### F3 — The "orthopyroxene" end-member is a pigeonite

The spectrum used as OPX (c1lr220, 70035 "light-brown pyroxene", LR-CMP-220) is
described in its own RELAB label as *"Pyroxene Clinopyroxene Pigeonite, Mineral separate
from Apollo 17 basalt (LRMCC)"* (`step03_endmember_identity.csv`). Every "OPX"
concentration in the paper is therefore a low-Ca clinopyroxene abundance. The CPX
end-member (c1lr215) is an augite, as labelled.

### F4 — Arithmetic errors in the printed tables

`step00b` re-adds every printed row (72 checks; `step00b_consistency.csv`):

- **Σc for site 1 omits c_OPX in all five M³ site tables** (8, 9, 13, 14, 15): the
  printed total falls short by exactly the OPX value each time (8.56, 6.54, 1.65, 1.60,
  1.60). Every other site row adds up.
- Table 6, A_Gr = 0.50: components sum to 92.51, printed 94.19.
- Table 13, site 4: 106.77 vs 106.97.
- Everything else — means, standard deviations (ddof = 1), and t_Optim = epoch × t_Epoch
  in Table 12 — is consistent. Each site's σ/σ_Rel implies the same ⟨A⟩ in all five
  tables (0.200, 0.173, 0.358, 0.263), which became our site-location check.
- Caption slips: Fig. 12 says "see also Table 7" (Table 9 is meant); Figs. 19–20 cite the
  basic components of "Table 3" (the set used is Table 4).

None of these changes a conclusion; they are recorded so that nobody scores a
reproduction against a misprint.

### F5 — Eq. (2) re-derived: the slope reproduces, the intercept does not — under every variant

From the Apollo 16 image the paper names (5 × 5 pixels at the LM site) against LSCC 62231 <45 µm,
both put through the authors' recipe (D2): A_RELAB = **1.669** A_M3 + 0.016, relative rms 1.21 %
(paper: 1.685, 0.0355, 1.4 %). The slope agrees within 0.02; the intercept is short by 0.020.
Neither the preprocessing nor the lab fraction explains it (`step05_eq2_sensitivity.csv`): with
the approximate preprocessing the intercept is 0.014; with the other 62231 fractions it is −0.005,
0.001 or 0.042 — and the one fraction that approaches 0.0355 (<10 µm, 0.042) has slope 1.95.
No variant gives both published coefficients. The published Eq. 2 is used throughout, as printed.

### F6 — Three of the eight mapped parameters are set by the training ranges, not by the spectra

On the recipe-processed scene (`step08_c1_recovery.csv`, `step08_c2_summary.csv`); the approximate
preprocessing gives the same result:

- **Recoverability (C1).** On 5,000 fresh synthetic spectra from the paper's own Table 11 ranges,
  with 1 % noise, the iteration-1 nets recover CPX (R² 0.83), ILM (0.91) and nFe (0.85), partly S
  (0.60) and PLG (0.49) — but **not "OPX" (R² 0.06), olivine (0.17) or the gray component
  (0.32)**. Their RMSE is 0.97, 0.91 and 0.83 of that of always answering the range midpoint.
- **Range dependence (C2b).** Retraining iteration 1 with the OPX and OLV ranges moved from
  −2…5 to 0…10 and gray from 20…40 to 30…50 moves the maps with them: the fraction of the
  midpoint shift the map median follows is **0.94 (OPX), 0.79 (OLV), 0.83 (gray)**
  (approximate preprocessing: 1.01, 0.78, 0.86).
- **No-information net (C2a).** Trained on spectra shuffled against their parameters, the net
  outputs the range midpoints (OPX 1.500, OLV 1.55, gray 30.1).
- The paper's published maps (Figs. 19–20) sit exactly where this predicts: OPX 1.35–1.60 %,
  OLV 1.2–1.7 % (midpoint of −2…5 is 1.5), gray 31.4–32.4 %. Ours: OPX median 1.39, OLV 1.52.

**Consequence.** The paper's OPX, olivine and gray-component maps — including the statement
that "the content of orthopyroxene and olivine is very low; for most of the surface it is almost
zero" — are statements about the chosen training ranges. The spectra do not inform them.

### F7 — Iterations 2–3 cannot move far from iteration 1, and overfit quickly

- Iterations 2–3 train only on iteration-1 outputs ± 0.5 %, so they refine rather than relocate:
  pixel-wise correlation of iteration 3 with iteration 1 is 0.95–0.98 for S, nFe, CPX and ILM and
  0.5–0.9 for the rest (`step08_c2c_inheritance.csv`). A range-determined iteration-1 map stays
  range-determined.
- The reconstruction error falls with each iteration, as the paper says: **3.07 → 1.59 → 1.23 %**
  (seed medians, recipe-processed scene; paper 2.39 → 1.26 → 1.12 %; approximate preprocessing
  3.31 → 1.77 → 1.35 %). Iterations 2 and 3 are within tolerance; iteration 1 is not (seeds
  2.12–3.93 %). But that is spectral fit, not composition: on synthetic truth the iteration-2/3
  nets are *worse* than iteration 1 for every parameter (C1), because they have only ever seen
  the scene's own answers.
- The optimum epoch is 40 and 90 for iterations 2 and 3 (seed medians; paper 5,336 and 4,256). The
  warm-start sensitivity (D12, `--warm`, seeds 0–2, run on the approximate preprocessing,
  `archive/v1_approx_preprocessing/tables/step07_table12_warm.csv`) gave long runs for one seed
  but *worse* errors (iteration 3: 1.68 / 2.12 / 1.85 %). Fine-tuning explains neither the
  paper's late optima nor its accuracy.
- The paper's claim that highland pixels are fitted *better* than mare after iteration 2 fails
  narrowly (seed medians 1.54 % vs 1.53 %); with the approximate preprocessing it holds (1.52 vs
  1.86 %). Seed 2 alone has a
  highland error of 3.6 %; the claim rests on a difference smaller than the seed spread.

### F8 — Where the ANN and the optimizer disagree, the evidence favours the optimizer

On 1,000 stratified pixels (C3, `step08_c3_ann_vs_nm.csv`):

- CPX from the two inverses is **anti-correlated** (r = −0.68 mare, −0.46 highland); highland
  PLG is 57 % by Nelder–Mead and 26 % by the ANN. Ilmenite (r 0.92 / 0.95) and highland nFe
  (0.91) agree.
- Nelder–Mead fits the spectra better (mean σ_Rel 0.87 % on these pixels; the ANN's scene mean
  is 1.23 %), and C1 shows the ANN cannot recover PLG well.
- The same gap is in the paper's own tables: site 3 has PLG 68.2 % in Table 9 (Nelder–Mead) and
  30.6 % in Table 15 (ANN). We reproduce both sides: 84.3 % and 28.9 %.

Our ANN reproduces the paper's ANN site values closely at the highland sites (site 3: S 1.2 vs
1.4 µm, CPX 40.1 vs 38.3 %, PLG 28.9 vs 30.6 %, ILM 6.6 vs 6.8 %; site 4 similar). So the
agreement is between two ANNs trained on the same ranges, not between the ANN and the data.

### F9 — The non-uniqueness of Table 10 depends on how hard the optimizer searches

With the grain-size corridor the paper describes (every S₀ from 1 to 500 µm), all four starting
c_PLG values reach **the same** solution at site 1 (S 134.1 µm, σ_Rel 0.89 %). Without the
corridor — one simplex from S₀ = 30 µm — the starts give S from 30 to 98 µm, but also σ_Rel from
1.10 to 1.50 %, i.e. worse fits, not equivalent ones (`step06_table10*.csv`). The paper's
Table 10 (four solutions within 0.03 of σ_Rel) is not reproduced by an exhaustive search; the
case for the ANN "removing ambiguities" rests on it.

### F10 — Speed: the claim holds, by more than stated

Nelder–Mead with the corridor took 12.9 s per pixel on one core → ~550 core-hours for the 152,000-pixel scene (paper: ~35 h on a 16-core
Ryzen). The ANN took 4.6 min of training for all three iterations and 0.64 s to map the scene
(`step08_c3_timing.csv`). "Orders of magnitude faster" is correct.

### F11 — Lab soils of known make-up: trends yes, proportions mostly no

The abstract's "agreement with lunar sample analysis" is asserted, not measured, in the paper.
Applying the paper's fitting procedure to the LSCC soils — the nine mare soils of Taylor et al. (2001) and
the ten Apollo 14/16 highland soils of Taylor et al. (2010, D21) — each size fraction
separately (`step10_*.csv`; n = soils, not spectra):

- **Grain size:** S rises with the sieve fraction in 16 of 18 soils with all three fractions
  (mare 8/8, highland 8/10).
- **Maturity:** c_nFe tracks Is/FeO within each set (Spearman, per fraction <10 / 10–20 / 20–45:
  mare 0.57 / 0.67 / 0.85, highland 0.81 / 0.70 / 0.56), but much less across the two sets
  pooled (0.37–0.55).
- **Mineral proportions**, each soil against the mean of the other soils of its set:
  - mare (n = 9): better only for "OPX" and ilmenite; worse for CPX, PLG and olivine — olivine is
    over-predicted by 10 points (17.5 % vs 7.6 %);
  - highland (n = 10): better only for "OPX" and PLG; worse for CPX, olivine and ilmenite —
    ilmenite is over-predicted by 5 points in soils that hold 1.9 %.
  - Pooled (n = 19), the fit ranks the soils well for PLG, ILM and CPX (Spearman 0.80, 0.89,
    0.74) — which says it separates mare from highland soils, not that it measures proportions
    within either.

### F12 — Comparisons with other methods

Using the formulas of the sources the paper cites (D15, D20, D22):

- **Maturity (Is, Pieters 2006 coefficients):** r = 0.61 mare / 0.71 highland (paper 0.85 / 0.83)
  — failures. Is/FeO in its place gives
  ~0 and −0.5: the paper's Fig. 22 must use Is, as its text says.
- **PLG (Shkuratov 2005a coefficients), mare:** 0.46 (paper 0.92) — failure.
- **TiO₂ by the Korokhin (2008) ANN (Fig. 30b):** 0.66 over the scene (paper 0.91) — failure; in
  the mare alone −0.06. The published weights of that paper do not reproduce its own fits (D22).
- **Ilmenite by the 1550-nm band depth (Fig. 31, Surkov 2020):** mare 0.27 (paper 0.54) —
  failure; highlands −0.05, "no correlation" as the paper says. The band-depth ilmenite is small
  (median 0.8 wt % in the mare), so the ANN gives **17×** as much, not "twice".
- ILM against Lucey's TiO₂ (closed form) in the highlands: 0.68 (paper 0.87) — failure.
- S against the MLR grain size: 0.86 (paper 0.99; the mare-only refit gives 0.96); CPX against
  ⟨A⟩ in the highlands 0.97 (paper 0.98) — both correlations with albedo-like quantities.
- CPX and OPX against their published MLRs are **negative** (−0.47 / −0.81): the ANN's pyroxene
  maps do not agree with the regression maps the paper compares them to.
- Histograms: S and PLG are both bimodal as in Fig. 20 (PLG is unimodal under the approximate
  preprocessing).

### F13 — Where our numbers depend on our reconstruction

Everything on the M³ scene rests on D2 (the authors' own recipe, with Ω located by a numeric
rule in place of their visual choice) and D9 (sites recovered from a figure). Sensitivities are
reported side by side in RESULTS.md: Table 9's σ_Rel moves by ≤ 0.08 between the three cubes, but
the site compositions move a lot (site 3 gray 0 / 4 / 41 %, PLG 84 / 82 / 64 % for recipe /
no-destripe / approximate). Per-site numbers are therefore reported, not gated; only the paper's
qualitative claims are scored — and two of them flip with the preprocessing (F14).

The comparisons with other methods are less sensitive: the correlations whose comparison
formula is the same in both runs (Lucey TiO₂ and OMAT, ⟨A⟩) moved by at most 0.10 between the
approximate and the recipe cube, with the network retrained on each (RESULTS.md, "Sensitivity to the
preprocessing"). The failing comparisons, except the highland Is and Lucey TiO₂ ones, miss their
tolerance by more than that.

### F14 — What depends on the preprocessing, and the final score

Scored with the approximate preprocessing instead of the authors' recipe (RESULTS.md,
"Sensitivity to the preprocessing"):

- **Better with the recipe:** site 1 of Table 9 matches closely (S 134 vs 138.7 µm, PLG 26.7 vs
  25.8 %, gray 52.5 vs 54.4 %; approximate: 116 µm, 11.5 %, 66.4 %); ANN iteration 2 within
  tolerance (1.59 vs 1.26 %); the PLG histogram is bimodal.
- **Worse with the recipe:** two marginal claims — highland fitted better than mare after
  iteration 2 (1.54 vs 1.53 %) and CPX > OPX at every site (site 3: 2.0 vs 3.2 %, both near zero).
- **The same under both:** every finding above (F2, F3, F6, F8, F9), the Eq. 2 intercept, ANN
  iteration 1, site 2's σ_Rel (1.07 vs 0.70 %), the agglutinate's ilmenite (9.2 vs 3.2 %), and the
  comparisons with other methods.

Final score: **46 / 65** gated targets pass. Six of the 19 misses are **near misses**
(RESULTS.md): numeric targets within 1.5 × their tolerance (T1_m2_OLV, T9_sigrel_site2,
T12_sigrel_iter1, C_nFe_Is_high), and the two claims that fail by less than the reproduction's own
spread. The tolerances themselves are unchanged. Counting near misses as agreement: **52 / 65**;
the 13 clear failures are the substantive ones.
