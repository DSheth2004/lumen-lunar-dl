# LUMEN architecture diagrams (draw.io)

| File | Figure |
|---|---|
| `LUMEN_01_framework.drawio` | Overall framework: data, physics pretraining, transfer, calibrated reporting (paper Fig. 1) |
| `LUMEN_02_simulator.drawio` | Domain-randomised Hapke simulator (paper Fig. 2) |
| `LUMEN_03_network.drawio` | 1-D CNN, trunk and Dirichlet / Gaussian / auxiliary heads (paper Fig. 3) |
| `LUMEN_04_training_inference.drawio` | Pretraining, A4 fine-tuning batch, ensemble + conformal + identifiability gate, orbital transfer (new) |
| `LUMEN_05_evaluation.drawio` | LOSO / LOMO / nested E1 and orbital P1 / P2 protocols (paper Fig. 4) |
| `LUMEN_all_diagrams.drawio` | All five as pages of one file |

Open in draw.io desktop or app.diagrams.net, then File → Export as → PDF / SVG (vector, best for journals) or PNG at 300–600 dpi.
For PDF/SVG export tick "Crop" and untick "Include a copy of my diagram" if the journal wants a clean file.

All labels use Helvetica and HTML sub/superscripts (no LaTeX), so exports render identically everywhere.
Colour coding: amber = data, blue = physics, green = learning, purple = uncertainty/reporting.

To regenerate after editing numbers or text: `python make_diagrams.py <out_dir> <preview_dir>` (needs only Python 3; `dio.py` must sit next to it).
