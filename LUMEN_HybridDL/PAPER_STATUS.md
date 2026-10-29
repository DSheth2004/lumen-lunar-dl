# LUMEN: what changed since `main` (paper work, 28–29 Oct 2026)

This branch (`paper-complete`) combines all the work done after your last commit on `main`
(`71c1023`): a set of new experiments, a correction to the orbital ground truth, and a full paper
draft for MDPI *Remote Sensing*. Nothing in the existing pipeline was rewritten; everything is
added on top.

## 1. Read this first: the orbital ground truth was wrong

`data/ground_truth/fernandez2026_ground_truth_50.csv`, used by `step10` and `step12` (Addenda A
and C), is **not** Fernández-Díaz et al.'s ground truth.
- Their paper says its labels are "49 samples compiled by [21] plus one sample from Chang'E-6".
  [21] is Qiu et al. (2025), Table S1: **49 distinct, station-level sites**.
- Our table squeezed the samples onto 18 coordinates; for example, 8 Apollo 12 samples sit at
  one point, where Qiu has a single Apollo 12 row.

Consequences:
- **The "FD is pseudo-replicated" argument is wrong and was withdrawn.** It described our
  reconstruction, not their paper.
- **The old orbital numbers** (6.36 and 2.64 wt% on 18 sites) are not comparable with FD's
  published 2.29 wt%. They are superseded by the new 49-site experiment (E2).
- **The correct table** is staged as `data/ground_truth/qiu2025_sampling_sites_feo.csv`. It was
  checked against FD's worked example (Apollo 17 LRV-12, 17.4 wt%), and M³ spectra for all 49
  sites are in `data/m3_sites_qiu/` (183 strips).

Details: `METHOD_NOTES.md`, Addendum D (D30).

## 2. New experiments (protocol committed before running: Addendum D, D30–D35)

| ID | Question | Script | Result |
|---|---|---|---|
| E1 | Does LUMEN beat RF when the A2/A4 choice is made *inside* the CV? | `step13_nested_selection.py` | **Yes.** A2 was chosen in all 25 folds; it is significantly better than RF on 8 of 13 targets LOSO (FeO 1.23 vs 2.13 wt%, Holm p = 0.011) and on 6 LOMO. |
| E2 + E3 | Orbital FeO at the 49 correct sites, under FD's own 10-fold CV (P1) and leave-one-landing-site-out (P2), with Lucey and two-band baselines | `step14_m3_qiu_sites.py` | **Against LUMEN.** Zero-shot: 6.81 wt%. Orbital fine-tuning, P2: 3.27 wt% (R² 0.34) vs FD as published 3.58 (p = 0.74, not significant); **a two-band M³ regression is significantly better (2.11 wt%).** Under P1, FD (1.90) beats LUMEN (2.20). |
| E4 | Are the design choices justified? | `step15_ablations.py` | log R alone ≈ both channels; continuum-removed alone is worse; the MLP is significantly *better* out of distribution on 4 targets; the 1-D ResNet ties. |
| E6 | Do the orbital intervals hold their coverage? | in `step14` | 0.63 / 0.92 / 0.98 at nominal 68 / 90 / 95 %, but wide (±10.6 wt% at 95 %). |
| E7 | Map consistency with Kaguya/Clementine | — | dropped for this version (future work) |

**Shared pretrained models.** All experiments use `checkpoints/pretrained/pretrained_member{0..4}.pt`,
re-pretrained in a cloud session. The regenerated synthetic set is bit-identical to the original
(same SHA-256), and validation metrics match the old pretraining to within 0.002.

**Implementation choices** that each experiment had to fix are recorded as Deviations 6–8 in
`METHOD_NOTES.md`.

## 3. What the paper now claims

**Laboratory (holds up):**
- FeO error is 42% lower than FD's pipeline on identical folds;
- mineral-proportion errors are 54–74% lower than the Shkuratov inversion;
- LUMEN meets all of Jung et al.'s accuracy bounds;
- it is significantly better than a tuned RF on 8 of 13 targets (E1);
- conformal intervals reach nominal coverage, and olivine is not identifiable.

**Orbital (negative, stated plainly):** lab skill does not transfer. Even after orbital fine-tuning,
LUMEN is not better than FD's pipeline, and it is worse than a two-band regression.

## 4. Where things are

| What | Where |
|---|---|
| Paper, MDPI *Remote Sensing* Word layout | `paper/MANUSCRIPT_MDPI.docx`, `paper/MANUSCRIPT_MDPI.pdf` (21 pages) |
| Paper text (edit here) | `paper/manuscript_src/*.md`, 40 references in `references.tsv` (checked against Crossref) |
| Figures | `figures/fig01–06_*.png`; vector diagram sources in `paper/figures_src/` (TikZ) |
| Result tables | `tables/step13_*`, `step14_*`, `step15_*`, `stepD_regeneration_check.csv` |
| Protocol and deviations | `METHOD_NOTES.md`, Addendum D and Deviations 6–8 |

**Rebuild the paper** (Windows with Word and MiKTeX), from `paper/`:
```
python make_figures.py
python build_manuscript.py
python check_numbers.py
python build_mdpi_docx.py
python build_pdf.py
```

**What the checks do:**
- `check_numbers.py` fails if any number in the text cannot be traced to the previous version or
  to `tables/*.csv`.
- `build_manuscript.py` reports any unfilled placeholder.

## 5. Open items

- **Author list, affiliations, contributions, funding and conflicts** are deliberately empty.
- **AI-use disclosure:** MDPI requires one in Materials and Methods; Parth will add it.
- **`RESULTS.md` and `PAPER_DRAFT.md`** still describe the main study and the superseded 18-site
  addenda. The paper itself (`paper/`) is the current write-up.
- **The `FernandezDiaz2026_WaveletAE_FeO/` folder** in this repo is a different FD implementation
  from Parth's audited one (in `paper-implementations/`). The paper relies on the audited one.
