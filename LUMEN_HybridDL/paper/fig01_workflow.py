"""Figure 1 of the manuscript: the LUMEN workflow (schematic). Writes figures/fig01_workflow.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig01_workflow.png"

INK, MUTED = "#1f2933", "#52606d"
FILL = {"physics": "#e8f1fb", "learn": "#eaf6ee", "data": "#fdf3e4", "out": "#f3eefb"}
EDGE = {"physics": "#3b6ea8", "learn": "#3c8c5a", "data": "#b7791f", "out": "#6b4fa0"}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.085, "\n".join(lines), ha="center", va="top", fontsize=7.6, color=MUTED,
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=None, dy=0.018):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=7.2, color=MUTED)


fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=200)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# row 1: physics -> pretraining
box(ax, 0.01, 0.56, 0.20, 0.40, "49 RELAB end-members",
    ["plagioclase, pyroxenes, olivine,", "ilmenite, glasses", "k(λ) by equivalent-slab inversion",
     "none from evaluation soils"], "data")
box(ax, 0.26, 0.56, 0.22, 0.40, "Hapke simulator",
    ["SSA intimate mixing (Eq. 6)", "npFe⁰ / mpFe⁰ weathering (Eq. 5)", "random regime, grain size,",
     "maturity, tilt, scale, noise", "400,000 labelled spectra"], "physics")
box(ax, 0.53, 0.56, 0.21, 0.40, "Pretraining",
    ["1-D CNN, 0.36 M parameters", "input: log R and continuum-", "removed R (2 × 71 bands)", "Dirichlet head: 7-part mode",
     "β-NLL heads: 5 oxides, Is/FeO", "5 ensemble members"], "learn")

# row 2: real data -> fine-tuning
box(ax, 0.01, 0.06, 0.20, 0.40, "Returned soils",
    ["19 LSCC soils, 75 RELAB spectra", "mode, oxides, Is/FeO", "", "M³ at 49 sampling sites",
     "(FeO, Qiu et al. 2025)"], "data")
box(ax, 0.26, 0.06, 0.22, 0.40, "Fine-tuning",
    ["per fold: real spectra", "(+ SSA-mixup, synthetic replay)", "orbital: + M³ pixel spectra",
     "of training sites (FeO only)", "LOSO / LOMO / site-grouped CV"], "learn")
box(ax, 0.53, 0.06, 0.21, 0.40, "Uncertainty",
    ["ensemble mean and variance", "normalised conformal (CV+)", "intervals at 68 / 90 / 95 %",
     "coverage verified out of fold"], "learn")

# output
box(ax, 0.79, 0.30, 0.20, 0.44, "Identifiability gate",
    ["skill = 1 − MAE / MAE(mean of", "other soils), bootstrap CI", "map a target only if CI > 0",
     "", "Output: maps + intervals", "for identifiable targets"], "out")

arrow(ax, 0.21, 0.76, 0.26, 0.76)
arrow(ax, 0.48, 0.76, 0.53, 0.76)
arrow(ax, 0.21, 0.26, 0.26, 0.26)
arrow(ax, 0.48, 0.26, 0.53, 0.26)
ax.add_patch(FancyArrowPatch((0.635, 0.56), (0.40, 0.46), connectionstyle="arc3,rad=-0.15",
                             arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK))
ax.text(0.53, 0.495, "initial weights", fontsize=7.2, color=MUTED, ha="center")
ax.add_patch(FancyArrowPatch((0.74, 0.26), (0.79, 0.42), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK))
ax.add_patch(FancyArrowPatch((0.74, 0.76), (0.79, 0.62), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK))
ax.text(0.80, 0.80, "zero-shot (ablation A0)", fontsize=6.8, color=MUTED, ha="left")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
