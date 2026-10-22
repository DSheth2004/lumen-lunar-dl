# Copernicus crater: Jung et al. (2026) and Korokhin et al. (2024) applied to M³ and IIRS

## What was done
Two published methods were applied to the same Copernicus scene from two sensors, and their outputs were compared.

- **Jung et al. (2026):** space-weathering correction plus Hapke/CATMIP unmixing.
- **Korokhin et al. (2024):** Shkuratov model plus a neural net.
- **Sensors:**
  - Chandrayaan-1 M³, `M3G20090416T122951`;
  - Chandrayaan-2 IIRS, the delivered 140 m map-projected cube.

Both methods are reused unchanged from `../Jung2026_SpaceWeatheredMaria` and `../Korokhin2024_ShkuratovANN`.

The inputs are read-only and guarded by a SHA-256 manifest. The audit passes **22/22** checks (`tables/audit.csv`).

Run order: `step01_inputs.py` → `step02_paper1.py` → `step03_paper3.py` → `step04_compare.py` → `audit.py`.
In total this takes about 3 h on 8 cores, mostly neural-net training.

## Setup
| Item | Value |
|---|---|
| Grid | Identical for both sensors: projection, 140 m, 311×1192 (checked) |
| Shared pixels | 98,421. IIRS covers a strip on the western side of the crater, reaching to about 3.6 km from the centre |
| Zones | Distance from the crater centre (9.62°N, 20.08°W, D = 96.07 km): central, floor, wall/rim, proximal ejecta |
| Common wavelengths | 0.85–2.50 µm, excluding the two IIRS gaps (1167–1285 nm, 1841–1976 nm), for both sensors |
| Configurations | M3_full (each method's own range), M3_common, IIRS_common (main test), IIRS_scaled (IIRS ÷ 1.170, a sensitivity check) |

## Findings about the data (before either method)
1. **IIRS is 10–28 % brighter than M³** at the same pixels, and the amount depends on wavelength: about 1.28 at 950 nm and about 1.10 at 2450 nm (`step01_radiometry.csv`).
2. **The two images are lit from opposite sides.**
   - Broad brightness patterns agree (r = 0.87).
   - Fine detail (slopes) is anti-correlated (r = −0.31 at zero shift). No shift within ±8 px fixes this, so it is illumination, not misregistration.
   - Consequence: per-pixel comparisons mix topographic shading with composition. **Smoothed maps and zone medians are the fair comparison.**
3. **IIRS has artificial dips at about 1.30 µm and about 2.00 µm.** Each one begins at the first band of a detector segment, right after a gap, and M³ shows nothing there. They are the deepest features in the IIRS spectrum (`figures/step04_iirs_segment_artefact.png`).

## Jung et al. (2026)
- **Almost no 970-nm band.** The band depth is 0.001–0.05, against 0.14–0.21 for the paper's mare targets.
  - The weathering correction therefore saturates: I_sm ≈ 53–65 for every zone, near the method's ceiling.
  - The scene's own "weathered" reference has depth 0.000.
  - Copernicus is outside the regime the method was built for.
- **Band-depth maps.** The smoothed maps correlate at r = 0.82 between the sensors, but IIRS depths are systematically shallower (bias −0.018).
- **Abundances.**
  - IIRS and M³ differ by up to 22.5 wt% for the same zone (median 3.9).
  - M³ full range and M³ cut range differ by about as much (up to about 22 wt%).
  - So the method's own instability is as large as the sensor difference, and **these abundances cannot separate the sensors.**
  - The fits are good (R² 0.94–0.99), which does not make the answers unique. This matches the earlier reproduction, where the prior dominated.

## Korokhin et al. (2024)
The neural net retrains cleanly on the 14 shared wavelengths. Model misfit (σ_Rel) is 1.3–1.7 % for M³ and 3.3–3.4 % for IIRS; the IIRS spectra are harder to fit, consistent with the artefacts.

Smoothed IIRS vs M³ (common range):

| Output | smoothed r | bias (IIRS − M³) | zone ranking same? |
|---|---|---|---|
| S (grain size) | 0.92 | +2.7 µm | yes |
| nFe | 0.92 | −0.019 | yes |
| CPX | 0.86 | +5.6 % | yes |
| OPX | 0.81 | +0.14 % | yes |
| OLV | 0.82 | +0.12 % | yes |
| PLG | 0.79 | +0.6 % | yes |
| Gray | 0.77 | +0.02 % | yes |
| ILM | 0.50 | −1.4 % | no |
| Ilmenite, 1550-nm band depth (Surkov) | −0.15 | −1.2 % | no |

- **Spatial patterns mostly agree.** The maps of grain size, iron (nFe) and pyroxene/plagioclase agree well once smoothed, and both sensors rank the zones the same way.
- **The levels are offset.** For example, IIRS gives +5.6 % CPX and a lower nFe.
- **The ilmenite outputs do not agree between sensors.** This holds both for the net's ILM output and for the 1550-nm band-depth method. The 1550-nm feature lies right beside the IIRS 1.30 µm artefact.
- **Removing the brightness offset (IIRS_scaled) changes little:** smoothed r 0.69–0.93. So the offsets come from spectral-shape differences, not just overall brightness.
- **Cutting M³ to the common range costs little** (M3_common vs M3_full, smoothed r 0.89–0.99), apart from PLG −2.1 % and CPX +2.8 %.
- **Some maps are set by the training ranges.** As the reproduction already found, the OPX, OLV and gray maps largely reflect the training ranges, so their agreement says little.
- **Not computable for IIRS** (they need 415/750 nm): the MLR maps, OMAT, Lucey TiO₂, and the Korokhin (2008) TiO₂ net (`step03_not_computable_for_iirs.csv`).

## Caveats
- Both methods were built on mare soils. Copernicus is fresh crater material, so every output here is an extrapolation.
- The IIRS photometric geometry is undocumented in the delivery. It was assumed to be the same as M³ (i = 30°, e = 0°).
- Eq. 2 of Korokhin et al. is an M³-specific calibration. It was applied unchanged to IIRS, because there is no IIRS equivalent.
- The neural net used 3 seeds (the paper used 5), and there was no Fourier destriping, because the data are map-projected.
