"""Build the manuscript figures.

- Diagrams (Figures 1-4): D. Sheth's draw.io diagrams, figures_src/drawio/ (make_diagrams.py writes
  the .drawio files), rendered with headless Chrome by figures_src/drawio/render_drawio.py to vector
  PDF (LaTeX build) and 3x PNG, copied to figures/fig0N_*.png (Word build). The earlier TikZ
  versions (figures_src/fig_*.tex) are no longer used.
- Data figures (Figures 5-6), from the committed tables:
    fig05_pred_true.png  A4 LOSO predictions vs measured for FeO, TiO2, PLG, log10 Is/FeO
                         (tables/step04_predictions.csv + step01_lab_dataset.csv)
    fig06_fd_feo.png     per-soil FeO MAE of every method, LOSO and LOMO
                         (tables/step03_metrics.csv + step04_metrics.csv)
Palette: LUMEN blue #3B6EA8 and FD-as-published amber #B7791F (checked for CVD separation and
contrast), neutral grey for the other methods; mare/highland use the same blue/amber pair.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "figures_src"
FIG = ROOT / "figures"
TAB = ROOT / "tables"

DRAWIO = SRC / "drawio"
DIAGRAMS = {"01_framework": "fig01_framework.png", "02_simulator": "fig02_simulator.png",
            "03_network": "fig03_architecture.png", "05_evaluation": "fig04_evaluation.png"}
BLUE, AMBER, GREY, INK, MUTED, RULE = "#3B6EA8", "#B7791F", "#7B8794", "#1F2933", "#52606D", "#D9DEE3"


def build_diagrams() -> None:
    subprocess.run([sys.executable, str(DRAWIO / "render_drawio.py")], check=True)
    for stem, png in DIAGRAMS.items():
        shutil.copy(DRAWIO / "rendered" / (stem + ".png"), FIG / png)
        print("  %s <- drawio/rendered/%s.png" % (png, stem))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8, width=0.8)
    ax.grid(color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)


def per_soil_mae(pred, lab, target):
    m = pred.merge(lab, left_on="row", right_index=True)
    err = (m["pred_" + target] - m[target]).abs()
    return err.groupby(m["soil"]).mean().dropna().mean()


def fig_pred_true(plt) -> None:
    lab = pd.read_csv(TAB / "step01_lab_dataset.csv", dtype={"soil": str})
    pred = pd.read_csv(TAB / "step04_predictions.csv")
    pred = pred[(pred.protocol == "LOSO") & (pred.method == "A4_full")]
    m = pred.merge(lab, left_on="row", right_index=True)
    panels = [("FeO", "FeO (wt%)"), ("TiO2", "TiO$_2$ (wt%)"), ("PLG", "Plagioclase (vol%)"),
              ("logIsFeO", "log$_{10}$ Is/FeO")]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.05), dpi=600)
    for ax, (t, label) in zip(axes, panels):
        d = m[[t, "pred_" + t, "set"]].dropna()
        lo, hi = float(min(d[t].min(), d["pred_" + t].min())), float(max(d[t].max(), d["pred_" + t].max()))
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for kind, colour, marker in (("mare", BLUE, "o"), ("highland", AMBER, "s")):
            s = d[d["set"] == kind]
            ax.scatter(s[t], s["pred_" + t], s=16, c=colour, marker=marker, edgecolors="white",
                       linewidths=0.6, zorder=3, label=kind)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        style(ax)
        ax.set_title("%s\nMAE %.2f" % (label, per_soil_mae(pred, lab, t)), fontsize=8, color=INK, pad=4)
        ax.set_xlabel("measured", fontsize=7.5, color=INK)
        ax.locator_params(nbins=4)
    axes[0].set_ylabel("predicted (LOSO)", fontsize=7.5, color=INK)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "fig05_pred_true.png", facecolor="white")
    plt.close(fig)
    print("  fig05_pred_true.png")


LABELS = {"A4_full": "LUMEN (A4)", "A2_pretrain_real": "LUMEN (A2)", "FD-AE-M3/RF": "FD as published (AE-M3 / RF)",
          "RF": "RF (LUMEN inputs)", "RF-raw": "RF (raw reflectance)", "PLS": "PLS", "ridge": "Ridge",
          "mean-of-others": "Mean of other soils"}


def fig_fd_feo(plt) -> None:
    a = pd.read_csv(TAB / "step03_metrics.csv")
    b = pd.read_csv(TAB / "step04_metrics.csv")
    b = b[b.method.isin(["A4_full", "A2_pretrain_real"])]
    m = pd.concat([a, b])
    m = m[(m.target == "FeO") & (m.level == "soil")]
    piv = m.pivot_table(index="method", columns="protocol", values="mae")
    piv = piv.sort_values("LOSO", ascending=False)
    names = [LABELS.get(k, k.replace("/", " / ")) for k in piv.index]
    colours = [BLUE if k.startswith("A") and k[1].isdigit() else AMBER if k == "FD-AE-M3/RF" else GREY for k in piv.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9), dpi=600, sharey=True)
    for ax, proto in zip(axes, ("LOSO", "LOMO")):
        y = np.arange(len(piv))
        ax.barh(y, piv[proto], height=0.68, color=colours, edgecolor="white", linewidth=0.8)
        for yi, k, v in zip(y, piv.index, piv[proto]):
            if k in ("A4_full", "FD-AE-M3/RF", "mean-of-others") or colours[yi] == BLUE:
                ax.text(v + 0.06, yi, "%.2f" % v, va="center", fontsize=6.8, color=INK)
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.set_title("leave-one-%s-out" % ("soil" if proto == "LOSO" else "mission"), fontsize=8.5, color=INK)
        ax.set_xlabel("per-soil FeO MAE (wt%)", fontsize=7.5, color=INK)
        ax.set_xlim(0, piv[proto].max() * 1.12)
    axes[0].set_yticks(np.arange(len(piv)))
    axes[0].set_yticklabels(names, fontsize=7)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / "fig06_fd_feo.png", facecolor="white")
    plt.close(fig)
    print("  fig06_fd_feo.png")


def main() -> int:
    print("diagrams:")
    build_diagrams()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    print("data figures:")
    fig_pred_true(plt)
    fig_fd_feo(plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
