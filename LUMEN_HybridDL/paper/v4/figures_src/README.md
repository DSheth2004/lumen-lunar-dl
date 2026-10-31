# Figure sources (LUMEN v4)

- Figures 1-4 (diagrams): TikZ, `fig1_architecture.tex` ... `fig4_evaluation.tex` with shared
  styles in `common.tex`. Build each with `pdflatex figN_*.tex`; the PDF is drawn at its printed
  size (at most 525 pt wide), so text prints at 8-9 pt and the figure is pure vector.
- Figures 5-9 (data plots): `python make_plots.py <tables dir> <step07_shapley_maps.npz> <out dir>`
  using the result tables of the LUMEN repository (branch `paper-complete`, `LUMEN_HybridDL/tables`).
  Output is vector PDF, 520 pt wide, Liberation Sans; the Shapley maps are embedded at 300 dpi.
