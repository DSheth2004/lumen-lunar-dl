"""
Step 08 -- redraw the paper's figures from our own tables.

Every panel is built from a table written earlier in the pipeline, never from a
number typed in here, so a figure cannot drift from the audit.  Where the paper
publishes a value for the same quantity it is drawn alongside ours (open marker
or dashed line) rather than described in a caption.

Figures produced, named for their counterparts in the paper:

    figC1_endmember_ssa.png      Fig. C1  single-scattering albedo of the 9 end-members
    fig02_ism_vs_delta.png       Fig. 2   maturity index against the band-depth ratio
    fig03_lscc_fits.png          Fig. 3   depth-corrected LSCC spectra and model fits
    fig04_abundances_vs_lscc.png Fig. 4   inverted mineralogy against the LSCC lab values
    figF1_posteriors.png         Fig. F1  full posterior for each soil
    fig05_m3_band_depths.png     Fig. 5   band-depth correction terms for the two M3 ROIs
    fig06_m3_fits.png            Fig. 6   M3 ROI spectra and their model fits
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jung_common as jc

STAGE = "step08"

SOILS = ["12001", "12030", "15071", "71061"]
CATS = ["olivine", "plagioclase", "pyroxene", "ilmenite"]
OURS = "#1f4e79"
PAPER = "#c1440e"
GREY = "#8a8a8a"


def _save(fig, name):
    jc.FIGURES.mkdir(parents=True, exist_ok=True)
    p = jc.FIGURES / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("    -> figures/%s" % name)
    return p


def fig_c1():
    d = pd.read_csv(jc.TABLES / "step03_figC1_endmember_ssa.csv")
    lam = d["wavelength_um"].to_numpy()
    cols = [c for c in d.columns if c.startswith("ssa_")]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True)
    for ax, c in zip(axes.ravel(), cols):
        ax.plot(lam, d[c], color=OURS, lw=1.4)
        ax.set_title(c[4:].replace("_", " "), fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("wavelength (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("single-scattering albedo")
    fig.suptitle("Figure C1 - end-member single-scattering albedo, "
                 "from the measured RELAB spectra", fontsize=11)
    fig.tight_layout()
    return _save(fig, "figC1_endmember_ssa.png")


def fig_02():
    t = pd.read_csv(jc.TABLES / "step02_table1_reproduction.csv")
    t["sample"] = t["sample"].astype(str)
    coef = pd.read_csv(jc.TABLES / "step02_eq3_coefficient.csv")
    best = float(coef.loc[coef["mean_abs_err_I_sm"].idxmin(), "coefficient"])

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    x = np.linspace(0.0, 1.05, 300)
    ax.plot(x, best * x ** 3.731, color=OURS, lw=1.6,
            label=r"our fit  $I_{sm}=%.2f\,\Delta^{3.731}$" % best)
    ax.plot(x, 65.58 * x ** 3.731, color=PAPER, lw=1.2, ls="--",
            label=r"Fig. 2 annotation  $65.58\,\Delta^{3.731}$")

    is_t = t["is_target"].astype(str).str.lower().isin(["true", "1"])
    tg = t[is_t]
    ax.scatter(t["Delta"], t["I_sm_calc"], s=26, color=GREY, zorder=3,
               label="all Apollo spectra (ours)")
    ax.scatter(tg["Delta"], tg["I_sm_calc"], s=70, facecolors="none",
               edgecolors=OURS, lw=1.6, zorder=4, label="the four inverted soils")
    if "paper_I_sm_calc" in tg.columns:
        ax.scatter(tg["paper_Delta"], tg["paper_I_sm_calc"], marker="x", s=60,
                   color=PAPER, zorder=5, label="Table 1, published")
    for _, r in tg.iterrows():
        ax.annotate(r["sample"], (r["Delta"], r["I_sm_calc"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8)

    # Measured maturity of the same sieve fraction, now that it is available.
    # This is NOT what Eq. (3) was calibrated against -- see PAPER_NOTES.md --
    # it is drawn so the difference is visible rather than asserted.
    gt = jc.DATA / "ground_truth" / "lscc_is_feo.csv"
    if gt.exists():
        m = pd.read_csv(gt)
        m["sample"] = m["sample"].astype(str)
        m = m[m["size_fraction_um"] == "20-45"].set_index("sample")["is_feo"]
        for _, r in tg.iterrows():
            if r["sample"] in m.index:
                ax.scatter([r["Delta"]], [m[r["sample"]]], marker="*", s=130,
                           color="#2e7d32", zorder=6)
        ax.scatter([], [], marker="*", s=130, color="#2e7d32",
                   label="measured Is/FeO, 20-45 um (LSCC Table 3)")
    ax.set_xlabel(r"band-depth ratio  $\Delta$  (Eq. 4)")
    ax.set_ylabel(r"soil maturity index  $I_{sm}$")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 175)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Figure 2 - maturity index against band-depth ratio", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig02_ism_vs_delta.png")


def fig_03():
    d = pd.read_csv(jc.TABLES / "step05_lscc_spectra.csv")
    u = pd.read_csv(jc.TABLES / "step05_lscc_unmixing.csv")
    u["sample"] = u["sample"].astype(str)
    u = u[u["config"] == "converged"].set_index("sample")
    lam = d["wavelength_um"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, s in zip(axes.ravel(), SOILS):
        ax.plot(lam, d["omega_dc_" + s], color="k", lw=2.0,
                label=r"depth-corrected $\omega_{dc}$")
        ax.plot(lam, d["fit_map_" + s], color=OURS, lw=1.3, label="MAP model")
        ax.plot(lam, d["fit_mode_" + s], color=PAPER, lw=1.0, ls="--",
                label="mode model")
        ax.plot(lam, d["fresh_map_" + s], color="#2e7d32", lw=1.0, ls=":",
                label="fresh model (agglutinates removed)")
        r2, pr2 = float(u.loc[s, "r2_map"]), float(u.loc[s, "paper_r2_map"])
        ax.set_title("%s   R2 = %.4f  (paper %.4f)" % (s, r2, pr2), fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("wavelength (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("single-scattering albedo")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Figure 3 - depth-corrected LSCC spectra and their model fits "
                 "(converged sampler)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig03_lscc_fits.png")


def _crystalline_percent(z, s):
    names = list(jc.ENDMEMBERS["name"])
    # theta is [9 grain sizes | 9 mass fractions]; abundances are the second half
    cry = [len(names) + names.index(n) for n in jc.CRYSTALLINE]
    f = z[s][:, cry]
    return f / f.sum(axis=1, keepdims=True) * 100.0


def fig_04():
    gt_p = jc.TABLES / "step05_vs_ground_truth.csv"
    if not gt_p.exists():
        print("    (skipping Figure 4: step05_vs_ground_truth.csv not present)")
        return None
    g = pd.read_csv(gt_p)
    g["sample"] = g["sample"].astype(str)
    z = np.load(jc.TABLES / "step05_posteriors.npz")
    grp = {"olivine": ["olivine"], "plagioclase": ["plagioclase"],
           "pyroxene": ["pigeonite", "augite"], "ilmenite": ["ilmenite"]}
    sub = {n: k for k, n in enumerate(jc.CRYSTALLINE)}

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    for ax, s in zip(axes, SOILS):
        f = _crystalline_percent(z, s)
        data = [f[:, [sub[n] for n in grp[c]]].sum(axis=1) for c in CATS]
        parts = ax.violinplot(data, positions=range(len(CATS)),
                              showextrema=False, widths=0.8)
        for b in parts["bodies"]:
            b.set_facecolor(OURS)
            b.set_alpha(0.45)
        gs = g[g["sample"] == s].set_index("category")
        for i, c in enumerate(CATS):
            ax.scatter([i], [gs.loc[c, "lab_wt_pct"]], marker="*", s=170,
                       color=PAPER, zorder=5)
            ax.scatter([i], [gs.loc[c, "our_median_wt_pct"]], marker="_", s=260,
                       color="k", zorder=6, lw=2)
        ax.set_xticks(range(len(CATS)))
        ax.set_xticklabels([c[:5] for c in CATS], rotation=30)
        ax.set_title(s, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    axes[0].set_ylabel("abundance (%, crystalline phases renormalised)")
    axes[0].scatter([], [], marker="*", s=170, color=PAPER, label="LSCC lab value")
    axes[0].scatter([], [], marker="_", s=260, color="k", lw=2,
                    label="posterior median")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 4 - inverted mineralogy against the LSCC 20-45 um lab values",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig04_abundances_vs_lscc.png")


def fig_f1():
    z = np.load(jc.TABLES / "step05_posteriors.npz")
    names = list(jc.ENDMEMBERS["name"])
    fig, axes = plt.subplots(4, 1, figsize=(10, 11))
    for ax, s in zip(axes, SOILS):
        f = z[s][:, len(names):] * 100.0
        parts = ax.violinplot([f[:, i] for i in range(len(names))],
                              positions=range(len(names)), showextrema=False,
                              widths=0.85)
        for b in parts["bodies"]:
            b.set_facecolor(OURS)
            b.set_alpha(0.45)
        for i in range(len(names)):
            ax.scatter([i], [np.median(f[:, i])], marker="_", s=240,
                       color="k", lw=2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_ylabel("wt %")
        ax.set_title("%s - full posterior, all nine end-members" % s, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    fig.suptitle("Figure F1 - posterior abundance distributions (converged sampler)",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "figF1_posteriors.png")


def fig_05():
    a = pd.read_csv(jc.TABLES / "step06_m3_aristarchus.csv").set_index("roi")
    c = pd.read_csv(jc.TABLES / "step06_closed_form_check.csv").set_index("roi")
    quant = [("d_mix", "paper_d_mix"), ("d_sw", "paper_d_sw"),
             ("Delta", "paper_Delta"), ("theta_sw", "paper_theta_sw")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, roi in zip(axes, ["roi_b", "roi_c"]):
        x = np.arange(len(quant))
        ours = [float(a.loc[roi, q]) for q, _ in quant]
        pub = [float(a.loc[roi, p]) for _, p in quant]
        ax.bar(x - 0.2, ours, 0.4, color=OURS, label="ours (our ROI rule)")
        ax.bar(x + 0.2, pub, 0.4, color=PAPER, label="published (Fig. 5)")
        # The closed-form values use the paper's own band depths, so they test
        # the arithmetic of Eqs. (3) and (4) rather than the ROI choice.
        ax.scatter([2, 3], [float(c.loc[roi, "Delta_from_paper_d"]),
                            float(c.loc[roi, "theta_sw_from_paper_d"])],
                   marker="D", s=48, color="k", zorder=5,
                   label="closed form from the paper's own d")
        ax.set_xticks(x)
        ax.set_xticklabels([r"$d_{mix}$", r"$d_{sw}$", r"$\Delta$", r"$\theta_{sw}$"])
        ax.set_title("%s  (Ism ours %.1f, paper %.1f)"
                     % (roi, float(a.loc[roi, "I_sm"]), float(a.loc[roi, "paper_I_sm"])),
                     fontsize=10)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("value")
    fig.suptitle("Figure 5 - band-depth correction terms at the two Aristarchus ROIs",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig05_m3_band_depths.png")


def fig_06():
    d = pd.read_csv(jc.TABLES / "step06_m3_spectra.csv")
    a = pd.read_csv(jc.TABLES / "step06_m3_aristarchus.csv").set_index("roi")
    lam = d["wavelength_um"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, roi in zip(axes, ["roi_b", "roi_c"]):
        ax.plot(lam, d["omega_mix_" + roi], color=GREY, lw=1.2,
                label=r"measured $\omega_{mix}$")
        ax.plot(lam, d["omega_sw_" + roi], color="#9467bd", lw=1.0, ls="-.",
                label=r"space-weathered $\omega_{sw}$")
        ax.plot(lam, d["omega_dc_" + roi], color="k", lw=2.0,
                label=r"depth-corrected $\omega_{dc}$")
        ax.plot(lam, d["fit_map_" + roi], color=OURS, lw=1.3, label="MAP model")
        ax.plot(lam, d["fresh_map_" + roi], color="#2e7d32", lw=1.0, ls=":",
                label="fresh model")
        ax.set_title("%s   R2 = %.4f  (paper %.4f)"
                     % (roi, float(a.loc[roi, "r2_map"]),
                        float(a.loc[roi, "paper_r2_map"])), fontsize=10)
        ax.set_xlabel("wavelength (um)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("single-scattering albedo")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 6 - M3 ROI spectra at Aristarchus and their model fits",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig06_m3_fits.png")


def main() -> int:
    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "redrawing the paper's figures from our tables")
    made = []
    for fn in (fig_c1, fig_02, fig_03, fig_04, fig_f1, fig_05, fig_06):
        p = fn()
        if p is not None:
            made.append(p.name)
    print("\n  %d figures written to figures/" % len(made))
    jc.verify_manifest(STAGE)
    print("\n  STEP 08 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
