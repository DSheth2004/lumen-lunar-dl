"""Figure 5 of the manuscript: predicted vs laboratory FeO at the 49 Qiu et al. (2025) sites under
P2 (leave-one-landing-site-out, 3 x 3-block spectra), for LUMEN-M3-FT, FD as published and the
two-M3-band baseline. Reads tables/step14_predictions.csv and tables/step14_metrics.csv; writes
figures/fig05_m3_sites.png."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "fig05_m3_sites.png"
SHOW = [("LUMEN-M3-FT", "LUMEN-M3-FT", "#2a78d6"),
        ("FD-AE-M3/RF", "FD as published (AE → RF)", "#6250d6"),
        ("two M3 bands (R950/R750, R750)", "Two M³ bands (R950/R750, R750)", "#eb6834")]
MARKERS = "osD^v<>p*hXP"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

P = pd.read_csv(ROOT / "tables" / "step14_predictions.csv")
P = P[(P["spectra"] == "block3x3") & (P["run"] == "P2")]
M = pd.read_csv(ROOT / "tables" / "step14_metrics.csv")
M = M[(M["spectra"] == "block3x3") & (M["run"] == "P2")].set_index("method")
missions = sorted(P["mission"].unique())
assert P[P["method"] == SHOW[0][0]]["site_id"].nunique() == 49

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axs = plt.subplots(1, len(SHOW), figsize=(10.5, 3.7), sharex=True, sharey=True, facecolor="white")
lo, hi = 0, 25
for ax, (method, title, colour) in zip(axs, SHOW):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, lw=0.6)
    g = P[P["method"] == method]
    for mis, mk in zip(missions, MARKERS):
        gg = g[g["mission"] == mis]
        ax.scatter(gg["truth"], gg["pred"], s=22, marker=mk, color=colour, alpha=0.85, lw=0, label=mis)
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title("%s\nMAE %.2f wt%%, R² %.2f" % (title, M.loc[method, "mae"], M.loc[method, "r2"]), fontsize=9)
    ax.set_xlabel("Laboratory FeO (wt%)")
axs[0].set_ylabel("Predicted FeO (wt%)")
h = [Line2D([], [], ls="", marker=mk, color=INK2, ms=5) for mk in MARKERS[:len(missions)]]
h.append(Line2D([], [], color=INK2, lw=0.9))
fig.legend(h, missions + ["1:1"], loc="center right", fontsize=7, frameon=False, title="Landing site", title_fontsize=7.5)
fig.tight_layout(rect=(0, 0, 0.9, 1))
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
