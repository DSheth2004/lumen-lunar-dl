# Jung et al. (2026) — *Remote Compositional Analyses of Space-weathered Lunar Maria*

Independent reimplementation and validation of
[doi:10.3847/PSJ/ae2b57](https://doi.org/10.3847/PSJ/ae2b57)
(Planetary Science Journal **7**(1) 18; open access, CC-BY-4.0), checked against the
authors' released code and data
([doi:10.5281/zenodo.17258920](https://doi.org/10.5281/zenodo.17258920), CC-BY-4.0).

A plain-language summary is in `../README_Jung2026.md`, next to this folder.

**Result: 32 of 34 gated targets pass, 1 is a near miss, 1 fails.** See
[RESULTS.md](RESULTS.md) (generated, never hand-edited),
[PAPER_NOTES.md](PAPER_NOTES.md) (equations and the fifteen deviations found) and
[LIMITATIONS.md](LIMITATIONS.md) (the near miss and the failure in full, scope, data used).

The near miss is 0.0003 below a stated floor; the failure is a substantive disagreement about
ilmenite that holds under every sampler configuration tried.

## What the paper does

Two steps. **Step 1** removes the spectral effect of space weathering by treating a
weathered spectrum as a linear single-scattering-albedo mixture of a "depth-corrected"
and a fully weathered component, with the mixing fraction set by the 970 nm band depth
through an empirical maturity relation. **Step 2** unmixes the corrected spectrum into
nine mineral end-members with a Bayesian Hapke model sampled by CATMIP, returning a
posterior over modal mineralogy and grain size rather than a point estimate.

## Running it

Python 3.10+ and four packages — no build step, no configuration, no paths to edit:

```
pip install numpy pandas scipy matplotlib
```

(Developed on Python 3.13.7 with numpy 2.2.6, pandas 2.3.3, scipy 1.16.2,
matplotlib 3.10.6.)

```
python scripts/run_all.py            # everything (~35 min; the MCMC dominates)
python scripts/run_all.py --fast     # skip the MCMC stages (~1 min)
python scripts/audit_paper_targets.py
```

**This folder is self-contained.** Every input it needs is staged inside `data/` and
covered by the SHA-256 manifest, so it runs anywhere it is copied, with no network
access and nothing else from the repository it was built in. `step00` will re-copy the
RELAB spectra from their original source tree if that tree happens to be present, and
otherwise verifies the staged copies against the shipped manifest — the identity checks
against Table C1 run identically either way. `step00b` downloads only if the M3 data is
missing, which in a distributed copy it is not.

`run_all.py` aborts on the first failing stage. `audit_paper_targets.py` is the test
suite: it reads [`paper_targets.csv`](paper_targets.csv) — the registry of published
numbers, each with a tolerance and a written justification for that tolerance — and
scores the pipeline's tables against it, exiting non-zero if any gated target fails.

| stage | what it does |
|---|---|
| `step00_stage_data.py` | copies the 22 RELAB spectra into `data/`, verifies all nine end-members against Table C1 from their PDS4 labels, with a shuffled negative control |
| `step00b_stage_m3.py` | stages the two M3 Aristarchus ROIs by HTTP range request (5 MB, not 4.9 GB) |
| `step01_hapke_selftest.py` | 10 self-tests of the Hapke core against the authors' released Python |
| `step02_band_depth.py` | Step 1 of the paper; reproduces Table 1 |
| `step03_endmember_optical_constants.py` | Appendix E optical constants; reproduces Figure C1 |
| `step04_catmip_synthetic.py` | gates the sampler on a synthetic mixture with a known answer |
| `step05_unmix_lscc.py` | Step 2 on the four LSCC mare soils; reproduces Figure 3 |
| `step06_m3_aristarchus.py` | the orbital application; Figures 5 and 6 |
| `step07_ground_truth.py` | inverted mineralogy vs the LSCC lab values; Figures 4 and F1, plus the prior-only control |
| `step08_figures.py` | validation views: our value against the published one, panel by panel |
| `step09_paper_figures.py` | reproduces Figures 1, 3, 5, 6 and A1 in the paper's own panel layout |

## Data

All four validation soils and all 22 RELAB spectra the paper cites, including the nine
Table C1 end-members, are staged in `data/relab/`.

**M³ images.** The paper prints its two Aristarchus sites on its Figure 5 but does not name the
M³ images. The images used here contain those sites; only the lines around each site are
downloaded (PDS Imaging Node, L2 reflectance V01 and L1B location V03):

| ROI | M³ image | site (paper, Fig. 5) | lines used |
|---|---|---|---|
| b | **M3G20090208T160125** | 21.37° N, 320.54° E | 9068–9116 |
| c | **M3G20090208T233940** | 31.76° N, 315.82° E | 15230–15278 |

`data/` carries a SHA-256 manifest that is re-verified at the **start and end** of every
stage; any drift aborts the run before output is touched. The original `RELAB/` cart-order
folder is opened read-only and never written to — and is not required at all once `data/`
is staged, which is why this folder runs on its own.

**LSCC ground truth** — modal mineralogy and Is/FeO, the stars and grey bars of Figures 4
and F1 — is hosted at `pgi.utk.edu`, which was unreachable at the time of retrieval; it was
retrieved from the Internet Archive instead. The four inverted spectra
are all the 20–45 µm sieve fraction, so the matching LSCC column is fixed by the PDS4
labels rather than chosen to fit. Transcription, its 81 internal consistency checks, and
one arithmetic erratum found in the source tables: `data/ground_truth/provenance.md`.

## What reproduces, and what does not

Reproduced essentially exactly:

- **Table 1** — every cell. `d_dc`, `d_sw`, and `d_mix`, Δ and calculated `I_sm` for all
  four soils, to within 0.0005 in band depth and 0.02 in `I_sm`.
- **Figure C1** — the SSA of all nine end-members, 9/9 inside their published panels.
- **Table C1** — all nine end-member identities, verified from the PDS4 labels.
- **Figures 5–6 arithmetic** — Eqs. (4) and (3) evaluated on the paper's own published
  band depths return its Δ, θ_sw and `I_sm` to 0.0015, 0.0012 and 0.12.
- **Figure 3** — R²(MAP) for all four soils, max error 0.0073.

Not reproduced, with the reason established rather than guessed:

- The paper's **hand-drawn M3 ROIs** cannot be recovered from the publication. What can
  be said is measured: the published `d_mix` sits at the 92.8th and 93.7th percentile of
  the two staged blocks, and `d_sw` at the 27.1st and 18.0th.
- **The released CATMIP sampler degenerates.** Reproduced faithfully, its ensemble
  collapses to ~0.1–1% unique particles and its acceptance ratio decays to zero, so its
  credible intervals are not posteriors. This is not a transcription error: the synthetic
  problem is identifiable, and a corrected simplex proposal recovers the planted answer
  exactly. Both configurations are run and reported. See PAPER_NOTES.md D7.
- **The ilmenite accuracy claim does not reproduce under any configuration.** Worst-case
  error against the LSCC lab values is 17.6 percentage points with the corrected sampler
  and 12.9 with the released one, against a stated bound of 10 — so the failure is not an
  artefact of our D7 correction. Fit quality is not the problem: R² is 0.983–0.994, at or
  above the published values, while the mineralogy is wrong. That is the signature of a
  non-unique inversion, and it is visible in `figures/fig03_lscc_fits.png` beside
  `figures/fig04_abundances_vs_lscc.png`.
- **Two of the three stated accuracy claims are met by the prior alone.** Scored against a
  no-information control — the Dirichlet prior renormalised the same way, with the spectra
  never consulted — the olivine claim is satisfied to 1.76 points, better than either sampler (5.78, 7.37),
  and the plagioclase claim to 16.92, level with the converged sampler (16.83) and better than
  the released one (22.56). Passing those bounds is therefore
  not evidence that the inversion recovered mineralogy from the spectrum. See
  PAPER_NOTES.md D12; this is the reproduction's main negative result.
- **The oxide chemistry points the same way as the ilmenite failure.** Converted to
  (SiO₂, FeO+MgO), our inverted ROIs sit at 31–35 wt% FeO+MgO against 24–29 measured in real mare
  soils — the direction an over-predicted ilmenite (~43 % FeO, ~0 % SiO₂) drives the chemistry. So
  the same defect shows up independently in a second quantity. The paper reports the opposite,
  that its mineralogies overlap the Apollo fields. Suggestive rather than conclusive: the
  reference population is soils, not the basalts the paper compares against (D14).
- **`S3_rms_all` misses by 0.00028** — 0.9797 against a stated floor of 0.98, driven by
  71061, the least-converged of the four. Reported as the near-miss it is; the tolerance
  was deliberately not relaxed to make it pass.

Also found: **Eq. (E9) is misprinted** in the paper (both released implementations use
the standard Hapke form, and the difference reaches 0.244 in SSA); the **Eq. (3)
coefficient is 65.68**, not the 65.58 printed on Figure 2; the **released optical
constants are inconsistent with Table C1's grain sizes** by a factor of 4–9 in `k·D`;
and the **calculated `I_sm` disagrees with the measured `Is/FeO` it estimates** by up to
34.6 units, inverting the maturity ranking of the four soils (D13, reported not gated).

## Coverage of the paper

All four equations, Appendices A–F, Table 1, Table C1 and **all nine published figures** are
reproduced — `step09` draws Figures 1, 3, 5, 6 and A1 in the paper's own panel layout, `step08`
draws the rest as ours-vs-published comparisons. One panel is deliberately skipped and one is
drawn under a stated assumption:

- **Figure 5(a)** — the LROC WAC context mosaic. External imagery carrying no reproduced
  quantity; skipped.
- **Figure 6(i),(j)** — the (SiO₂, FeO+MgO) panels are drawn, but on an end-member oxide table
  **we chose**: the paper says inverted mineralogy "was converted to oxide abundances" and
  specifies neither the compositions nor the conversion, and the released code has no such
  routine. We use measured Apollo mineral chemistry (LSCC Table 4), documented in
  `data/ground_truth/provenance.md`; everything derived from it is reported, never gated. The
  comparison population is also not the paper's — the Apollo mare basalt compilation
  (ApolloBasaltDB) is not used; LSCC mare *soil* bulk chemistry is drawn instead and labelled as
  such (D14).

## Citation

Jung, J.-I., Lapôtre, M. G. A., Milliken, R. E., & Minson, S. E. (2026),
*Remote Compositional Analyses of Space-weathered Lunar Maria*,
The Planetary Science Journal 7(1) 18. doi:10.3847/PSJ/ae2b57

Jung, J.-I., Lapôtre, M., & Minson, S. (2025), *Data and Software for Remote
Compositional Analyses of Space-Weathered Lunar Maria*, v1, Zenodo.
doi:10.5281/zenodo.17258920

CATMIP: Minson, S. E., Simons, M., & Beck, J. L. (2013), Geophys. J. Int. **194**, 1701.
Bayesian Hapke unmixing: Lapôtre, M. G. A., Ehlmann, B. L., & Minson, S. E. (2017),
JGR Planets **122**, 983.
