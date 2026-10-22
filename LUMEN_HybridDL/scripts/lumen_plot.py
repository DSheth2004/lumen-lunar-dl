"""
Shared figure style for LUMEN: one categorical order (fixed, never cycled), thin marks,
recessive axes. Static PNGs for the paper; every figure's numbers also exist as a CSV
in tables/, which serves as the table view.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots in fixed order (validated reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# a fixed colour per method, so a method keeps its colour in every figure
METHOD_COLOR = {"LUMEN": C[0], "mean-of-others": "#8a8984", "PLS": C[1], "ridge": C[2],
                "RF": C[3], "RF-raw": C[4], "FD-wavelet-RF": C[6], "FD-AE": C[7],
                "Korokhin": C[5], "Jung": C[3]}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 5, "font.size": 9,
        "axes.titlesize": 10, "legend.frameon": False, "figure.dpi": 110,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
