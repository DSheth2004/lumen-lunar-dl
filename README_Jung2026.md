# Reproducing Jung et al. (2026): removing space weathering from lunar spectra and estimating minerals

**Paper:** J.-I. Jung, M. G. A. Lapôtre, R. E. Milliken & S. E. Minson (2026), *Remote
Compositional Analyses of Space-weathered Lunar Maria*, The Planetary Science Journal 7(1), 18.
doi:10.3847/PSJ/ae2b57 (open access)

**Authors' released code and data:** Zenodo, doi:10.5281/zenodo.17258920

**Code and full results:** folder `Jung2026_SpaceWeatheredMaria/`

---

## 1. Aim

To rebuild the paper's method from scratch, run it on the same data, and check whether its
published numbers and claims hold. Every checkable published number was written down, with an
allowed tolerance, **before** our results existed, and then scored automatically. The authors'
released code was used as a reference to tell our own mistakes apart from real discrepancies.

## 2. What the paper does

On the Moon, micrometeorites and the solar wind slowly coat soil grains with tiny specks of iron
("space weathering"). This darkens the soil and weakens the absorption features used to identify
minerals. The paper corrects for this in two steps:

1. **Undo the weathering.** A weathered spectrum is treated as a blend of a fresh spectrum and a
   fully weathered one. The blend ratio is set from the depth of the absorption band at 970 nm,
   and the weathering is then removed.
2. **Estimate the minerals.** The corrected spectrum is split into nine ingredients (olivine,
   plagioclase, two pyroxenes, ilmenite, three glasses and agglutinates), each with its own amount
   and grain size. This uses a physical light-scattering model (Hapke) inside a statistical
   sampler (CATMIP, a Markov-chain Monte Carlo method). The sampler returns a range of likely
   answers with a stated confidence, not just one answer. That is the paper's main selling point.

It tests the method on 4 lunar soils measured in the laboratory, then applies it to Chandrayaan-1
M³ images of two young craters near Aristarchus crater.

## 3. Data used

| data | what it is |
|---|---|
| **M³ image M3G20090208T160125**, lines 9068–9116 | site "b" of the paper's Figure 5: 21.37° N, 320.54° E |
| **M³ image M3G20090208T233940**, lines 15230–15278 | site "c" of the paper's Figure 5: 31.76° N, 315.82° E |
| **RELAB laboratory spectra** (22) | the 9 mineral ingredients, 4 lunar soils (Apollo 12001, 12030, 15071, 71061; 20–45 µm fraction) and the fresh and weathered reference samples |
| **LSCC data tables** (Taylor et al. 2001) | measured mineral content, maturity and chemistry of the lunar soils: the "true answers" |
| **Authors' released code and data** | used as a reference to check our implementation |

Full PDS product IDs (NASA PDS Imaging Node; reflectance in volume CH1M3_0004, location in
CH1M3_0003):

- `M3G20090208T160125_V01_RFL` and `M3G20090208T160125_V03_LOC` — site b
- `M3G20090208T233940_V01_RFL` and `M3G20090208T233940_V03_LOC` — site c

Both M³ images are the calibrated reflectance product (Level 2) from the NASA PDS archive, with
their location files. The paper prints the two site coordinates on its Figure 5 but does not name
the images; the images used here contain those sites. Only the lines around each site were
downloaded (about 5 MB per site instead of 2–3 GB per image).

## 4. How the checking was done

- The weathering correction, the light-scattering model and the sampler were implemented from
  the paper's equations and checked with 10 self-tests, several of them directly against the
  authors' released code.
- **Scorecard:** 34 of the paper's numbers and claims are scored against tolerances fixed in
  advance. Another 11 values are reported for context.
- **Extra tests the paper did not run:**
  - Can the sampler find the right answer on a made-up mixture where the answer is known?
  - How well would you score if you ignored the spectra completely and used only the starting
    assumption the method begins from (the "prior")?
  - Is the maturity index the method computes consistent with the maturity measured in the
    laboratory?
- The paper's figures are redrawn from our results (except one context-map panel, Figure 5(a)).

## 5. Outcome

**Of the 34 scored items:**
- **32 pass;**
- **1 misses narrowly** (0.9797 against a stated minimum of 0.98);
- **1 misses clearly** (the ilmenite accuracy claim).

**What reproduces:**
- the weathering correction: the paper's Table 1, every value, band depths within 0.0005;
- the identities and optical properties of all nine mineral ingredients;
- the quality of the spectral fits (R² 0.983–0.994, at or above the published values);
- the arithmetic of the M³ application, to 3–4 decimal places.

**Main findings:**
1. **The accuracy claims are too loose to show the method works.** The paper supports the method
   with three claims: olivine within 7 points, plagioclase within 20–30, ilmenite within 10.
   Ignoring the spectra altogether and using only the starting assumption satisfies **two of the
   three**: olivine (better than the full method) and plagioclase (about as well). So passing those two
   says nothing about whether the method read anything from the data.
2. **Ilmenite, where the data must do the work, fails** (error 17.6 against the bound of 10). It
   also fails with the authors' own sampler settings (12.9). The spectra are fitted very well, but
   with the wrong mix of minerals: several different mixtures reproduce the same spectrum.
3. **The authors' sampler gets stuck.** On a made-up mixture with a known answer, it recovered
   0 of 4 mineral groups within its own confidence range, and its 2,000 trial answers collapsed
   to about 1 % distinct ones. So its stated confidence ranges are not reliable. A corrected
   version recovers 4 of 4. Both versions are reported side by side.
4. **The chemistry confirms the ilmenite problem.** Converted to rock chemistry, our results are
   too iron-rich (31–35 % FeO+MgO against 24–29 % in real mare soils), which is what too much
   ilmenite would produce.
5. **The maturity index disagrees with the laboratory.** It differs from the measured maturity
   by up to 35 units, and the ranking flips: the soil it rates most weathered (71061) is the least
   weathered by laboratory measurement.
6. **Errors in the paper:** one printed equation (E9) differs from the one in the authors' own
   code; a coefficient is printed as 65.58 where 65.68 is correct; and the released optical
   constants do not match the grain sizes listed in the paper's own table (a factor of 4–9).

## 6. Why some results do not match

| what does not match | paper | ours | reason |
|---|---|---|---|
| Ilmenite accuracy (worst soil) | within 10 | 17.6 | The spectra can be fitted equally well by different mineral mixes, and the method picks one with too much ilmenite. Also fails with the authors' own settings (12.9), so it is not caused by our changes. |
| Spectral fit quality, all soils | at least 0.98 | 0.9797 | A near miss; it rounds to 0.98. |
| Weathered and fresh pixels at the two M³ sites (reported, not scored) | — | — | The paper picked them by hand on a map, which cannot be recovered from the publication. The paper's own published band depths are used for the scored checks instead, and they reproduce. |
| Mineral chemistry comparison, Figure 6(i),(j) (reported, not scored) | overlaps Apollo basalts | outside the mare-soil range | The paper does not say how it converted minerals to chemistry, so we chose a documented method. We compared against lunar soils, not the Apollo basalt collection the paper uses. |
| Thermal correction of the M³ images | Li & Milliken (2016) | standard archive correction | The archive images carry the standard correction; the two differ mainly beyond 2 µm. |

## 7. Bottom line

The weathering correction and the physics reproduce cleanly, and the spectral fits are excellent.
But the evidence the paper offers for its mineral estimates does not show they work: two of the
three accuracy claims are met by the starting assumption alone, the one that needs the data
(ilmenite) fails, and the sampler behind the stated confidence ranges gets stuck.

**Where to look next:** in the folder, `RESULTS.md` has the full scorecard, `PAPER_NOTES.md`
records every equation, deviation and finding with its evidence, and `LIMITATIONS.md` gives the
scope, the data identifiers and practical notes.

## In short

We rebuilt the paper's method and ran it on the same laboratory soils and the same Chandrayaan-1
M³ sites near Aristarchus crater (images M3G20090208T160125 and M3G20090208T233940). 32 of the
paper's 34 checkable results match, 1 is very close, and 1 does not. The weathering correction
and the physics hold up. But two of the paper's three accuracy claims can be met without looking
at the data at all, the one that needs the data (ilmenite) fails, and the statistical sampler
gets stuck, so its confidence ranges cannot be trusted. The code, data and full results are in
the `Jung2026_SpaceWeatheredMaria` folder.
