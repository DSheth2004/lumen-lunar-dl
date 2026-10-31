"""Data figures 5-9 for LUMEN_v3, as vector PDF sized to the printed width (525 pt), Nimbus Sans,
no titles or captions inside the figures (captions live in the LaTeX)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

TAB = Path(sys.argv[1]); MAPS = Path(sys.argv[2]); OUT = Path(sys.argv[3])
import glob
for f in glob.glob("/usr/share/fonts/truetype/liberation*/LiberationSans-*.ttf"):
    font_manager.fontManager.addfont(f)
W = 520 / 72                                   # printed width in inches (fits the 525 pt full-width column)
INK, MUTED, RULE = "#1F2933", "#52606D", "#D9DEE3"
BLUE, AMBER, GREY, PURPLE = "#2F6DB5", "#B7791F", "#8A96A3", "#6B4FA8"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "pdf.fonttype": 42,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})

def style(ax):
    ax.grid(color=RULE, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(labelcolor=INK, width=0.8)

def letter(ax, s, x=-0.02, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="right", va="bottom")

# ---------------- Figure 5: predicted vs measured (A4, LOSO) ----------------
lab = pd.read_csv(TAB / "step01_lab_dataset.csv", dtype={"soil": str})
pred = pd.read_csv(TAB / "step04_predictions.csv")
pred = pred[(pred.protocol == "LOSO") & (pred.method == "A4_full")]
m = pred.merge(lab, left_on="row", right_index=True)
panels = [("FeO", "FeO (wt%)"), ("TiO2", "TiO$_2$ (wt%)"), ("PLG", "Plagioclase (vol%)"), ("logIsFeO", "log$_{10}$ Is/FeO")]
fig, axes = plt.subplots(2, 2, figsize=(0.78 * W, 0.78 * W * 0.86))
axes = axes.ravel()
for k, (ax, (t, lab_)) in enumerate(zip(axes, panels)):
    d = m[[t, "pred_" + t, "set"]].dropna()
    lo = float(min(d[t].min(), d["pred_" + t].min())); hi = float(max(d[t].max(), d["pred_" + t].max()))
    pad = 0.06 * (hi - lo); lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for kind, c, mk in (("mare", BLUE, "o"), ("highland", AMBER, "s")):
        s = d[d.set == kind]
        ax.scatter(s[t], s["pred_" + t], s=20, c=c, marker=mk, edgecolors="white", linewidths=0.5, zorder=3, label=kind.capitalize())
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); style(ax); ax.locator_params(nbins=4)
    ax.set_xlabel("Measured " + lab_); ax.set_ylabel("Predicted " + lab_)
    letter(ax, "(%s)" % "abcd"[k], x=-0.2)
axes[0].legend(frameon=False, loc="lower right", handletextpad=0.1, borderaxespad=0.2)
fig.tight_layout(w_pad=2.0, h_pad=1.4); fig.savefig(OUT / "fig5_pred_true.pdf"); plt.close(fig)

# ---------------- Figure 6: per-soil FeO MAE of every method ----------------
LABELS = {"A4_full": "LUMEN (A4)", "A2_pretrain_real": "LUMEN (A2)", "FD-AE-M3/RF": "FD as published (AE-M3 / RF)",
          "RF": "RF (LUMEN inputs)", "RF-raw": "RF (raw reflectance)", "PLS": "PLS", "ridge": "Ridge",
          "mean-of-others": "Mean of other soils"}
a = pd.read_csv(TAB / "step03_metrics.csv"); b = pd.read_csv(TAB / "step04_metrics.csv")
b = b[b.method.isin(["A4_full", "A2_pretrain_real"])]
mm = pd.concat([a, b]); mm = mm[(mm.target == "FeO") & (mm.level == "soil")]
piv = mm.pivot_table(index="method", columns="protocol", values="mae").sort_values("LOSO", ascending=False)
names = [LABELS.get(k, k.replace("/", " / ")) for k in piv.index]
cols = [BLUE if k in ("A4_full", "A2_pretrain_real") else AMBER if k == "FD-AE-M3/RF" else GREY for k in piv.index]
fig, axes = plt.subplots(1, 2, figsize=(W, 4.3), sharey=True)
for k, (ax, proto) in enumerate(zip(axes, ("LOSO", "LOMO"))):
    y = np.arange(len(piv))
    ax.barh(y, piv[proto], height=0.7, color=cols, edgecolor="white", lw=0.6)
    for yi, key, v in zip(y, piv.index, piv[proto]):
        ax.text(v + 0.06, yi, "%.2f" % v, va="center", fontsize=8, color=INK)
    style(ax); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Per-soil FeO MAE (wt%%), %s" % proto); ax.set_xlim(0, piv[proto].max() * 1.15)
    letter(ax, "(%s)" % "ab"[k], x=0.0)
axes[0].set_yticks(np.arange(len(piv))); axes[0].set_yticklabels(names)
fig.tight_layout(w_pad=1.2); fig.savefig(OUT / "fig6_fd_feo.pdf"); plt.close(fig)

# ---------------- Figure 7: interval coverage ----------------
cov = pd.read_csv(TAB / "step05_coverage.csv")
fig, axes = plt.subplots(1, 2, figsize=(W, 3.15), sharey=True)
series = [("raw_ensemble", "Raw ensemble", GREY, "o"), ("conformal_normalised", "Conformal (normalised)", BLUE, "s"),
          ("conformal_absolute", "Conformal (absolute)", AMBER, "D")]
for k, (ax, proto) in enumerate(zip(axes, ("LOSO", "LOMO"))):
    c = cov[cov.protocol == proto].groupby(["method", "level"])["coverage_spectrum"]
    for meth, lab_, col, mk in series:
        v, lo, hi = c.mean().loc[meth], c.min().loc[meth], c.max().loc[meth]
        ax.fill_between(v.index, lo.values, hi.values, color=col, alpha=0.13, lw=0)
        ax.plot(v.index, v.values, marker=mk, color=col, lw=1.8, ms=5, label=lab_)
    ax.plot([0.6, 1.0], [0.6, 1.0], color=MUTED, lw=0.9, ls=(0, (4, 3)), label="Nominal")
    ax.set_xlim(0.65, 0.97); ax.set_ylim(0.2, 1.0); style(ax)
    ax.set_xlabel("Nominal coverage, %s" % proto); letter(ax, "(%s)" % "ab"[k], x=0.0)
axes[0].set_ylabel("Empirical coverage (spectra)")
hh, ll = axes[0].get_legend_handles_labels()
fig.legend(hh, ll, loc="lower center", ncol=4, frameon=False, handlelength=2.2, columnspacing=1.8)
fig.tight_layout(w_pad=1.5, rect=(0, 0.09, 1, 1)); fig.savefig(OUT / "fig7_coverage.pdf"); plt.close(fig)

# ---------------- Figure 8: orbital FeO at 49 sites (P2) ----------------
P = pd.read_csv(TAB / "step14_predictions.csv"); P = P[(P.spectra == "block3x3") & (P.run == "P2")]
SHOW = [("LUMEN-M3-FT", BLUE), ("FD-AE-M3/RF", PURPLE), ("two M3 bands (R950/R750, R750)", AMBER)]
missions = sorted(P.mission.unique()); MK = "osD^v<>p*hXP"
fig, axes = plt.subplots(1, 3, figsize=(W, 2.75), sharex=True, sharey=True)
for k, (ax, (meth, col)) in enumerate(zip(axes, SHOW)):
    g = P[P.method == meth]; assert g.site_id.nunique() == 49
    for mis, mk in zip(missions, MK):
        gg = g[g.mission == mis]
        ax.scatter(gg.truth, gg.pred, s=24, marker=mk, color=col, alpha=0.9, lw=0)
    ax.plot([0, 25], [0, 25], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlim(0, 25); ax.set_ylim(0, 25); ax.set_aspect("equal"); style(ax)
    ax.set_xlabel("Laboratory FeO (wt%)"); letter(ax, "(%s)" % "abc"[k], x=0.0)
axes[0].set_ylabel("Predicted FeO (wt%)")
h = [Line2D([], [], ls="", marker=mk, color=INK, ms=5.5) for mk in MK[:len(missions)]]
fig.legend(h, missions, loc="center right", frameon=False, title="Site", title_fontsize=8.5, handletextpad=0.2, labelspacing=0.3)
fig.tight_layout(rect=(0, 0, 0.88, 1), w_pad=1.0); fig.savefig(OUT / "fig8_m3_sites.pdf"); plt.close(fig)

# ---------------- Figure 9: Shapley-region maps ----------------
d = np.load(MAPS)
tg = [("FeO", "FeO (wt%)"), ("TiO2", "TiO$_2$ (wt%)"), ("Al2O3", "Al$_2$O$_3$ (wt%)"), ("logIsFeO", "log$_{10}$ Is/FeO")]
fig, axes = plt.subplots(2, 4, figsize=(W, 6.2))
for j, (t, lab_) in enumerate(tg):
    for i, (key, cmap, unit) in enumerate([(t, "viridis", lab_), ("hw90_" + t, "magma", "90% half-width, " + lab_)]):
        ax = axes[i, j]; v = d[key]
        im = ax.imshow(v, cmap=cmap, aspect="auto", interpolation="nearest",
                       vmin=np.nanpercentile(v, 1), vmax=np.nanpercentile(v, 99), rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(True); s.set_color(MUTED)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
        cb.set_label(unit, fontsize=8.5, color=INK); cb.ax.tick_params(labelsize=8, colors=INK)
        cb.ax.locator_params(nbins=4)
        letter(ax, "(%s)" % "abcdefgh"[i * 4 + j], x=0.0)
fig.tight_layout(h_pad=1.2, w_pad=0.8); fig.savefig(OUT / "fig9_shapley_maps.pdf", dpi=300); plt.close(fig)
print("ok")
