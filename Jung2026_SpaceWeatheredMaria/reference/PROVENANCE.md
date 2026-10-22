# `reference/` — where each of these files came from

Nothing in this folder is our own work. It is third-party material, kept alongside the
reimplementation so that every claim in `PAPER_NOTES.md` can be checked against the thing it
refers to. All of it is redistributable under the licences below, and all of it is cited.

| file | what it is | source | licence |
|---|---|---|---|
| `Jung2026_PSJ_ae2b57.pdf` | the paper being reproduced | Jung, Lapôtre, Milliken & Minson (2026), *Remote Compositional Analyses of Space-weathered Lunar Maria*, Planetary Science Journal **7**(1) 18, [doi:10.3847/PSJ/ae2b57](https://doi.org/10.3847/PSJ/ae2b57) | open access, CC-BY-4.0 |
| `_paper_text.txt` | plain-text extraction of that PDF, for grepping quoted values | extracted locally from the PDF above | as above |
| `space-weathering-main/`, `space-weathering-main.zip` | **the authors' own released code and data** | Jung, Lapôtre & Minson (2025), Zenodo, [doi:10.5281/zenodo.17258920](https://doi.org/10.5281/zenodo.17258920) | CC-BY-4.0 |
| `lscc_tables/Table1–4.pdf` | Lunar Soil Characterization Consortium data tables — the laboratory ground truth | Planetary Geosciences Institute, University of Tennessee; underlying publication Taylor, Pieters, Keller, Morris & McKay (2001), JGR **106**(E11), 27985–27999. Retrieved via the Internet Archive because `pgi.utk.edu` is unreachable — see [`../data/ground_truth/provenance.md`](../data/ground_truth/provenance.md) for the full retrieval route | published data tables, cited to source |

## Why the authors' code is here

It is used as a **reference oracle**, not as a dependency — no script in `scripts/` imports it.
Our Hapke implementation was written from the equations in the paper's appendices, and then
checked against `hapkeRTM/hapkeRTM.py` from the Zenodo record. Several of the deviations recorded
in `PAPER_NOTES.md` are disagreements between the *paper as printed* and the *authors' own code*,
which is a distinction that can only be drawn by holding both.

The MCMC settings in `mcmc/` matter for the same reason: two of the reproduction's results are
reported under both the authors' released sampler settings and a converged configuration, so that
a disagreement cannot be dismissed as a sampling artefact.
