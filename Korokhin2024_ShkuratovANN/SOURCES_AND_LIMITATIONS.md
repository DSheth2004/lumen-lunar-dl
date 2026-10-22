# Sources, limitations and practical notes

## Status

Complete. Every stage runs end to end, and the audit scores all 65 gated targets (`RESULTS.md`).
A `run_all.py` from empty outputs (2026-09-21, ~4.3 h) reproduced every audit value and table
exactly, apart from run times (`archive/cleanrun_verification/`). A full run with a simpler,
approximate M³ preprocessing is kept in `archive/v1_approx_preprocessing/` as a sensitivity test.
RESULTS.md compares it with the main run.

## Sources used besides the paper

| source | used for |
|---|---|
| Surkov, Y. (2022), *Mineralogical mapping of selected areas of the lunar surface using Moon Mineralogy Mapper data*, PhD thesis, Univ. Göttingen (open access); its chs. 2 and 4 reprint Shkuratov et al. (2019) and Surkov et al. (2020) | the authors' M³ preprocessing → step05; Fig. 31 band-depth ilmenite → step09 |
| Taylor et al. (2001), JGR 106, 27985 — LSCC mare-soil data tables | mare LSCC mineralogy, chemistry, Is/FeO → step00, step09, step10 |
| Taylor et al. (2010), *Mineralogical and chemical characterization of lunar highland soils*, JGR 115, E02002 (NASA NTRS preprint 20100006633) | highland LSCC mineralogy, Is/FeO → step00, step09, step10 |
| Shkuratov, Kaydash & Pieters (2005a), *Lunar clinopyroxene and plagioclase: surface distribution and composition*, Sol. Syst. Res. 39, 255 | published CPX, PLG, Is/FeO regression coefficients; independent check of the highland transcription |
| Pieters et al. (2006), *Lunar soil characterization consortium analyses: pyroxene and maturity estimates derived from Clementine image data*, Icarus 184, 83, doi:10.1016/j.icarus.2006.04.013 | published Is and OPX regression coefficients |
| Korokhin et al. (2008), *Prognosis of TiO₂ abundance in lunar soil using a non-linear analysis of Clementine and LSCC data*, PSS 56, 1063, doi:10.1016/j.pss.2008.02.001 | Fig. 30: training data and standardization of the TiO₂ network |
| Shkuratov et al. (2003), LPSC 34, abstract 1258 | the regression form only |
| RELAB spectral library; M³ L2 data from the PDS (Chandrayaan-1) | spectra and the image |

## Out of scope

- **A full-scene Nelder–Mead map** — ~550 core-hours at 12.9 s per pixel. The paper did not make
  one either; the 1,000-pixel comparison (control C3) answers the question.
- **Figs. 7–8 in full** — five RELAB spectra of the two rejected basis sets (CASC02, C2LS09,
  CALS35, CJB308, C2PO26) are not in the local RELAB downloads. No target depends on them.
- **The authors' own processed M³ cube and site pixels** (the paper: "available on request"). They
  would remove D2/D9 as a source of difference but are not expected to change the findings:
  switching between the approximate preprocessing and the authors' recipe moved the fixed-formula
  comparisons by ≤ 0.10 (RESULTS.md, "Sensitivity to the preprocessing"). Of the 13 clear
  failures, only E2_intercept and C_ILM_TiO2_high are within such a move of their tolerance; the
  other comparisons miss it by 0.14–0.36, and T10 depends on search depth (F9). The lab-side
  failures and the training-range, end-member and Table 1 findings do not depend on the M³ cube.

## Running from this folder alone

`data/` (inputs, sealed by `data/manifest.csv`) and `reference/` (transcribed and digitized
paper material) are included. Stages `step00` and `step00c` rebuild them from sources outside the
folder (the RELAB downloads, the LSCC spectra, the paper PDF); when those are absent,
`run_all.py` skips the two stages and uses the included copies. `step04` downloads the M³ lines
again (~1.3 h, network needed); `python scripts/run_all.py --from step05` reruns everything
after it from the included copy.

## Practical notes

- **The scene is M3G20090728T214215 (OP2C), raw lines 13918–14805.** The ODE archive returns ~80
  products for Shapley; most have 0–360° footprint artefacts, and only 2 of the 5 plausible ones
  contain the crater. Pixels are 261 × 147 m, which is why the paper's 152,000 pixels are
  304 × 500 after resampling to square pixels.
- **Sites** (rows/samples of the 500 × 304 scene): 1 = 118/243, 2 = 88/281, 3 = 361/157,
  4 = 408/114. Each site's σ/σ_Rel in the printed tables implies its mean albedo (0.200, 0.173,
  0.358, 0.263), which is a quick check that a site is right. Site 4 is featureless: 72 pixels lie
  within 1 point of the best match, so its row is loosely determined.
- **The recipe's own acceptance test (channel means within 0.2 %) cannot be met on this scene**
  by any λ-kernel ≥ 1.5 band pixels, because the mean spectrum curves at 580–750 nm. A one-sided
  kernel at the spectrum's ends biases the 541-nm channel by 6 %; odd reflection avoids that.
- **Stripe comb:** in the 2-D power spectrum, at cross-track periods < 12 samples, it stands
  0.7 dex above the off-comb rows, ±3 rows wide. The lowest frequency band also shows an
  along-track excess, from the Crisium margin rather than stripes, and is excluded.
- **The <45 µm lunar end-members are c1lr212/215/217/218/220.** The c1lr168–183 spectra are the
  <125 µm fraction of the same separates, a different set.
- **n = 2.0 cannot represent albedo below 0.0431** (0.0190 for n = 1.6): Eq. A2 has no solution
  there. The agglutinate LS-CMP-045 is darker than that at 0.54 µm, hence D19.
- **Speed:** the numpy forward model costs ~1.5 ms per call, the numba version 22 µs.
  `LunarModel.forward` is the reference; `fast` is used in loops.
- **α_Fe:** Johnson & Christy + Ordal agree with the paper's Fig. A1 to ~3 % below 1.94 µm and are
  11–21 % low above it, so the digitized Fig. A1 is used (rule fixed in step01 before comparing).
- **Fig. 12 digitizing:** the right-hand panels draw the blue curve dark (B ≈ 100), and the
  magenta difference curve's anti-aliased edges look blue, so the mask excludes R > 70.
- **Lab fits use 0.30–2.60 µm at 5 nm:** this reproduces the ⟨A⟩ implied by Table 7 to 3 × 10⁻⁵.
- **Korokhin 2008's printed network weights do not reproduce its own fits** (no biases; k 0.40
  instead of 0.99 on its own Table 1). Its Table 2 fixes the input reading (X = (x − x0)·xk, AR as
  a fraction). Retraining the 6-3-1 net on Table 1 gives k 0.94.
- **The published regression coefficients are log10 with A in %.** Pieters 2006 Is reproduces
  its own k (0.93 vs 0.92) on the LSCC spectra. The Clementine-band filters match Korokhin 2008
  Table 1 within 0.25 %.
- **ANN iterations 2–3 peak within ~100 epochs** for all seeds. Longer runs do not reach the
  paper's 5,336 / 4,256-epoch optima; the warm-start variant (`--warm`) gives worse errors.
