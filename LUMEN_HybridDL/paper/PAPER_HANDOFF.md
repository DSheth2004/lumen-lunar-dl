# Paper hand-off — finish the LUMEN manuscript from the Addendum D results

This file lets a fresh session (e.g. a Claude Code cloud session) finish the manuscript with no
other context. Read it top to bottom before editing anything.

## 1. What exists

**Branch `paper-draft`** holds the manuscript draft for MDPI *Remote Sensing*. Its layout follows
Lin et al. 2026, *Remote Sens.* 18, 1123 (PGU-Net): Highlights → Abstract → 1 Introduction → 2
Materials and Methods → 3 Results → 4 Discussion → 5 Conclusions → back matter.

- **Sources:** `LUMEN_HybridDL/paper/manuscript_src/01_front_intro_methods.md`,
  `02_results.md`, `03_discussion_conclusions.md`.
- **Citations:** `[@key]`. The keys are in `manuscript_src/references.tsv`: 40 references, with
  metadata checked against Crossref.
- **Build:** `python LUMEN_HybridDL/paper/build_manuscript.py`. It writes `paper/MANUSCRIPT.md`,
  numbers citations in order of first appearance and lists the remaining `[[PENDING-...]]`
  placeholders. An unknown citation key is an error.
- **Figure 1:** `figures/fig01_workflow.png`, made by `paper/fig01_workflow.py`. The other figures
  are existing files in `figures/`.

**The protocol** for every new experiment is `LUMEN_HybridDL/METHOD_NOTES.md`, Addendum D
(D30–D35) on branch `paper-experiments`. It was committed before the experiments ran.

**Three cloud sessions** ran the experiments, each on its own branch:

| Experiment | Branch | Script | Result tables |
|---|---|---|---|
| E1, nested A2/A4 selection (D32) | `exp-e1` | `scripts/step13_nested_selection.py` | `tables/step13_*.csv`, `tables/stepD_regeneration_check.csv` |
| E4, input/architecture ablations (D34) | `exp-e4` | `scripts/step15_ablations.py` | `tables/step15_*.csv` |
| E2 + E3 + E6, orbital FeO on the 49 Qiu sites (D33) | `exp-e2` | `scripts/step14_m3_qiu_sites.py` | `tables/step14_*.csv` |

E7 (map consistency, D35) was **dropped from this version** for time. The text already lists it
as future work. Do not add it.

## 2. Rules (non-negotiable)

1. **Every number comes from a committed result table.** Read the CSVs; never copy numbers from a
   session's chat summary without checking them against the table. Never invent or round in a way
   that changes a conclusion.
2. **Claim strength follows the statistics exactly:**
   - Holm-corrected p < 0.05 on the pre-registered test → "significantly".
   - Lower error without Holm significance → "lower (not significant after correction)".
   - Higher error → say so plainly, in the Results, the Discussion and, if it is a headline
     result, the Abstract and Highlights.
3. **No placeholder may survive.** `build_manuscript.py` must report `pending placeholders: 0`.
4. **Withdrawn claim.** D30 withdraws the "FD is pseudo-replicated" claim. It must not appear
   anywhere. The earlier 18-site orbital results (Addenda A/C) are superseded and are not cited
   in the paper.
5. **Edit only `manuscript_src/*.md`** (and figure scripts), then rebuild. Do not hand-edit
   `MANUSCRIPT.md`.
6. **Keep line endings LF.** The repo stores files byte-exact (`* -text`).

## 3. Step by step

```bash
git fetch origin
git switch paper-draft
# bring in the result tables (and scripts, for figure code) without merging unrelated history:
git checkout origin/exp-e1 -- LUMEN_HybridDL/tables/ LUMEN_HybridDL/scripts/step13_nested_selection.py
git checkout origin/exp-e4 -- LUMEN_HybridDL/tables/ LUMEN_HybridDL/scripts/step15_ablations.py
git checkout origin/exp-e2 -- LUMEN_HybridDL/tables/ LUMEN_HybridDL/scripts/step14_m3_qiu_sites.py
# also copy any figures the sessions produced (figures/step13_*, step14_*, step15_*):
git checkout origin/exp-e1 -- LUMEN_HybridDL/figures/ || true
git checkout origin/exp-e4 -- LUMEN_HybridDL/figures/ || true
git checkout origin/exp-e2 -- LUMEN_HybridDL/figures/ || true
ls LUMEN_HybridDL/tables/step1[345]_* LUMEN_HybridDL/tables/stepD_*
```

If a branch's tables are missing or its run is unfinished, leave that placeholder as an explicit
`[Result pending: <experiment>]` note, and say so in the final report. **Never fabricate.**

### 3.1 Fill the placeholders (11 of them)

**`01_front_intro_methods.md`**
- **Highlights, "implications" bullet**
  (`[[PENDING-E2: orbital headline from step14, P2]]`): one clause with the P2 MAE and R² of
  LUMEN-M3-FT, and how it compares with FD as published and the two-band baseline.
- **Abstract** (`[[PENDING-E2: P2 MAE, R², ...]]`): one or two sentences with the P2 MAE/R² of
  LUMEN-M3-FT vs FD as published, the primary-test p (Holm), and the best simple baseline. Keep
  the Abstract under about 250 words.

**`02_results.md`**
- **Section 3.3, E1**: report:
  - how often A2 vs A4 was selected (LOSO and LOMO);
  - the outer per-soil MAEs of LUMEN-selected;
  - the primary test: LUMEN-selected vs RF, LOSO, Holm over 13 targets — the number of targets
    lower, the significant targets, and the FeO p;
  - LOMO as secondary.

  Add a short table if the results need one. Mention the regeneration check
  (`stepD_regeneration_check.csv`) in one sentence as a reproducibility statement, not a result.
- **Section 3.6**: replace the four placeholders with:
  - **Table 6**: MAE, RMSE, R² and bias under P1 and P2 for every method in D33. Flag the Lucey
    published-constants row as in-sample with a footnote;
  - the primary test (LUMEN-M3-FT vs FD as published, P2, Wilcoxon, Holm over comparators);
  - the P1 seed sensitivity and the single-pixel sensitivity, one sentence each;
  - E6 coverage at 68/90/95%;
  - **Figure 5**: predicted vs laboratory FeO at the 49 sites under P2, for LUMEN-M3-FT, FD as
    published and the two-band baseline. Use a figure from `exp-e2` if one exists; otherwise
    make it with matplotlib from `tables/step14_*predictions*.csv` and save it as
    `figures/fig05_m3_sites.png`. Use the existing figure style: white background, labelled
    axes, and a 1:1 line.

  Also compare the P1 numbers with FD's published 2.29 wt% (10-fold CV, [@fernandez2026]) and
  with the reproduction's 1.95 wt%.
- **Section 3.7, E4**: per variant (log-R only, continuum-removed only, MLP, 1-D ResNet), the
  number of targets with lower MAE than the main CNN (2 members each) and the Holm-significant
  differences, under LOSO and LOMO. Add a small table if useful.

**`03_discussion_conclusions.md`**
- **Section 4.1, E1**: one sentence interpreting whether in-fold selection keeps the advantage
  over RF.
- **Section 4.3, E2**: one paragraph covering:
  - P1 vs P2: how much grouping by landing site changes every method's error;
  - whether LUMEN-M3-FT beats FD as published and the band ratios, and how strongly;
  - the size of the lab-to-orbit gap (zero-shot vs fine-tuned);
  - what the in-sample Lucey/Clementine result does and does not show.
- **Section 5, Conclusions**: one sentence on the orbital result.

### 3.2 Also update (dependent text)

- **Highlights, "main findings"**: add an orbital bullet only if the E2 result supports one.
- **Section 4.5 Limitations**: drop or adjust any limitation that E1/E2/E4 resolved.
- **Section 4.4 / 3.5**: add the orbital coverage (E6) in one sentence.
- **Search the text for "48"** and make sure every site count matches the number of sites
  actually scored in `step14` (49 expected).

### 3.3 Build, export, check

```bash
cd LUMEN_HybridDL/paper
python build_manuscript.py                # must say: pending placeholders: 0
pip install pypandoc_binary              # PyPI wheel bundles pandoc
python -c "import pypandoc; pypandoc.convert_file('MANUSCRIPT.md','docx',outputfile='MANUSCRIPT.docx',extra_args=['--resource-path=.'])"
```

Then re-read the whole of `MANUSCRIPT.md` once, end to end. Check that:
- table and figure numbers are sequential (Tables 1–7+, Figures 1–6), and every figure file
  exists;
- every number in the Abstract, Highlights and Conclusions appears in the Results;
- no sentence claims significance that the tables do not show;
- the 18-site orbital numbers (6.36 / 2.64 wt%) and the pseudo-replication claim appear nowhere.

### 3.4 Commit and report

- Commit `manuscript_src/`, `MANUSCRIPT.md`, `MANUSCRIPT.docx`, any new figures and figure
  scripts, and the copied result tables to **`paper-draft`**, then push. Never push to `main`.
- End with a plain report:
  - the three headline results (E1, E2, E4) with their numbers;
  - every place where a result went **against** LUMEN;
  - anything left pending.
