# LSCC ground truth — provenance

Lab values for the Figure 4 / Figure F1 comparison and the three Section 3
accuracy claims: modal mineralogy, the pyroxene sub-set split, and Is/FeO
maturity, for the Lunar Soil Characterization Consortium mare soils.

## Source

Lunar Soil Characterization Consortium data tables, published by the Planetary
Geosciences Institute, University of Tennessee:

    http://pgi.utk.edu/wp-content/uploads/2016/04/Table1.pdf   modal mineralogy, mare soils
    http://pgi.utk.edu/wp-content/uploads/2016/04/Table2.pdf   pyroxene sub-sets, mare soils
    http://pgi.utk.edu/wp-content/uploads/2016/04/Table3.pdf   bulk chemistry + Is/FeO, mare soils

linked from <https://pgi.utk.edu/lunar-soil-characterization-consortium-lscc-data/>,
itself linked from the RELAB LSCC page <https://sites.brown.edu/relab/lscc/>.
Underlying publication: Taylor, Pieters, Keller, Morris & McKay (2001), *Lunar
mare soils: space weathering and the major effects of surface-correlated
nanophase Fe*, JGR 106(E11), 27985–27999.

**`pgi.utk.edu` was not reachable** at the time of retrieval — every
request failed with "Failure when receiving data from the peer", while every other
archive used by this project (PDS Imaging, PDS Geosciences, ODE, Zenodo,
sites.brown.edu) responds normally. The three PDFs were therefore retrieved
through the Internet Archive:

    https://web.archive.org/web/2018id_/http://pgi.utk.edu/wp-content/uploads/2016/04/Table{1,2,3}.pdf

The `id_` suffix requests the bytes as originally archived, with no Wayback
rewriting. Tables 4–9 were also retrieved but are not used: 4, 5 and 9 are
mineral/glass compositions, and 6–8 are highland soils, which are out of scope
(Eq. 3 is calibrated on mare basalts).

The RELAB LSCC page itself hosts only the `*Kdata.txt` reflectance spectra,
which this project already had locally; it carries no compositional data.

## Which rows are used

The four inverted spectra are all the **20–45 µm** sieve fraction — established
from the RELAB PDS4 labels in `step00` (`tables/step00_relab_metadata.csv`:
c1lr10 20–44, c1lr32 20–44, c1lr20 20–40, c1lr57 20–45 µm), not chosen to fit.
`step07` therefore reads the `20-45` column with no free choice.

## Transcription and its checks

The PDFs are text, not scans, and were transcribed by hand into
`lscc_modal_mineralogy.csv` and `lscc_is_feo.csv`. The published tables carry
their own internal redundancy, which is used as the transcription test — 81
checks, all recorded in `lscc_transcription_checks.csv`:

- every Table 1 column sums to the printed 100.0 (within the rounding residue of
  seven values printed to one decimal place);
- every Table 2 pyroxene sub-set column sums to its own printed "Total Pyroxene";
- every printed "Total Pyroxene" equals Table 1's "Pyroxene" for the same soil
  and fraction.

**One genuine erratum in the source**, recorded rather than corrected: for
12001, 20–45 µm, Table 2 prints orthopyroxene 0.82, and the four sub-sets then
sum to 18.88 against its own printed Total Pyroxene of 19.88 — a discrepancy of
exactly 1.00. The value 0.82 was re-read from the PDF in layout mode to rule out
a lost leading digit in text extraction; it is printed as 0.82. Table 1
independently gives 19.9 for that cell, agreeing with the printed total, so the
**total** is taken as correct and the sub-set split for that one cell as
unreliable. No gated target depends on the pyroxene split: `step07` compares
pigeonite + augite as a single "pyroxene" category, exactly as Figure 4 does.

Two further cross-table differences exceed two-decimal rounding but stay inside
the 0.35 percentage-point tolerance and are recorded as `ok`: 71061 20–45 µm
(Table 1 20.8 vs Table 2 20.49) and 79221 20–45 µm (13.5 vs 13.18).

Header inconsistencies in the source, not carried into the CSVs: Table 2 heads
10084 as `-84` where Tables 1 and 3 head it `-78`, and heads 71501 as `71051`.
The suffix is the Is/FeO of the <250 µm fraction; the value 78 is used, per
Tables 1 and 3.

## Caveat carried into the comparison

LSCC modal abundances are point counts, i.e. **by volume**; the sampler's
abundance parameters are **mass** fractions. The paper plots its inversion
against the published modal numbers, so that is what `step07` gates on; it also
reports a density-converted volume-basis column as a sensitivity, using the
Table C1 densities. Neither side of that choice is hidden in the table.

---

## End-member oxide compositions (`endmember_oxides.csv`)

Added to reproduce Figure 6(i),(j). The paper states that inverted mineralogy "was converted to
oxide abundances" but gives neither the per-end-member compositions nor the conversion, and the
released code has no such routine — so this table is **our choice, not the paper's**, and every
quantity derived from it is reported and never gated. See PAPER_NOTES.md D14.

Source: LSCC Table 4, "Average compositions of minerals and glasses in the finest size fractions
of high-Ti mare soils", same provenance and retrieval route as Tables 1–3 above; the PDF is kept
at `reference/lscc_tables/Table4.pdf`.

Construction, in full:

- Only the **20–45 µm** column is used, matching the sieve fraction of the inverted spectra.
- Averaged over all five high-Ti mare soils Table 4 covers: 10084, 79221, 71061, 70181, 71501.
- Values printed as `<0.04` are taken as 0.02 (half the detection limit).
- Each end-member is renormalised to 100 wt%; the raw analytical totals (97.9–99.4 wt%, the
  shortfall being elements not analysed) are retained in `analytical_total_wt_pct`.
- Parse check: every one of the 45 mineral columns sums to its printed total within rounding.

Mapping from Table 4's mineral columns to the paper's nine end-members:

| end-member | Table 4 column | note |
|---|---|---|
| olivine | Olivine | |
| plagioclase | Plag | |
| pigeonite | Pig | |
| augite | Mg-Cpx | augite is the high-Ca clinopyroxene; Fe-Cpx not used |
| ilmenite | Ilm | |
| pyroxene glass | Pig | a synthetic glass of pyroxene composition |
| plagioclase glass | Plag | a synthetic glass of plagioclase composition |
| orange glass | Vol Gls. | orange glass *is* volcanic glass (74220) |
| agglutinate | Agglut. Gls. | |

**Two limitations, stated rather than buried.** Table 4 covers high-Ti mare soils only, so applying
it to low-Ti material is an approximation — it is used for the Aristarchus M3 ROIs, which are mare
basalt, and not for a gated quantity. And the paper compares against Apollo mare-basalt fields from
ApolloBasaltDB (Cone et al. 2020), which has not been obtained; the reference points drawn instead
are LSCC mare *soil* bulk chemistry from Table 3, which is a different population and is labelled
as such on the figure.
