"""
Step 09 -- reproduce the paper's figures in the paper's own panel layout.

step08 draws validation views: our value against the published one, panel by
panel.  This stage instead reproduces the figures *as the paper composes them*,
so a reader can put them side by side with the PDF.  Everything is built from
tables written earlier in the pipeline, or recomputed from the stored posterior
ensembles; no number is typed in here.

    fig01_paper.png   Fig. 1   (a) lab reflectance vs maturity, (b) M3 fresh vs weathered
    fig03_paper.png   Fig. 3   (a)-(l) all three rows for the four LSCC soils
    fig05_paper.png   Fig. 5   (b),(c) the two ROIs on the 970 nm reflectance base map
    fig06_paper.png   Fig. 6   (a)-(h) for the two M3 ROIs
    figA1_paper.png   Fig. A1  (a) SSA, (b) continuum-removed SSA, (c) why 970 nm

Two panels of the published figures are NOT reproduced, for stated reasons:

  * **Fig. 5(a)**, the LROC WAC context mosaic, is external imagery that carries
    no reproduced quantity.
  * **Fig. 6(i),(j)** IS drawn, but on an end-member oxide table WE chose.  The
    paper states that "inverted mineral compositions were converted to oxide
    abundances" and gives neither the end-member compositions nor the
    conversion; the released code has no such routine (`wtperctof.m` converts
    weight fraction to cross-section fraction, not to oxides).  The table used
    is measured Apollo mineral chemistry from LSCC Table 4, documented in
    `data/ground_truth/provenance.md`.  Because the input is our choice and not
    the paper's, every quantity derived from it is reported and never gated.
    The comparison population is also not the paper's: it uses Apollo mare
    *basalts* from ApolloBasaltDB, which has not been obtained, so LSCC mare
    *soil* bulk chemistry is drawn instead and labelled as such.  See D14.

Row 3 of Fig. 3 and panels (e),(f) of Fig. 6 need the spread of *accepted*
models, which is not stored as spectra, so the forward model is re-evaluated
over the whole stored ensemble here.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt

import hapke
import jung_common as jc
from step04_catmip_synthetic import build_forward

STAGE = "step09"

SOILS = ["12001", "12030", "15071", "71061"]
ROIS = ["roi_b", "roi_c"]
# Figure 1(a): the fresh basalt plus the five soils the caption names.
FIG1_FRESH = ("c1lr154", "15555")
FIG1_SOILS = [("c1lr10", "12001"), ("c1lr32", "12030"), ("c1lr57", "71061"),
              ("c1lr28", "71501"), ("c1lr17", "79221")]
ACCEPTED_LO, ACCEPTED_HI = 2.5, 97.5

RED = "#c1440e"
BLUE = "#1f4e79"
GREEN = "#2e7d32"


def _save(fig, name):
    jc.FIGURES.mkdir(parents=True, exist_ok=True)
    p = jc.FIGURES / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("    -> figures/%s" % name)
    return p


def _measured_ism(fraction: str = "<250") -> pd.Series:
    """Measured Is/FeO, or an empty Series if the ground truth is not staged.

    Defaults to the <250 um reference maturity, because that is the value the
    paper prints beside each soil in Table 1 and in the Figure 1 / Figure 3
    captions (12001 = 56, 12030 = 14, 15071 = 52, 71061 = 14).  The 20-45 um
    column is the one matching the inverted sieve fraction, and is what step07
    uses for the quantitative comparison.
    """
    src = jc.DATA / "ground_truth" / "lscc_is_feo.csv"
    if not src.exists():
        return pd.Series(dtype=float)
    m = pd.read_csv(src)
    m["sample"] = m["sample"].astype(str)
    return m[m["size_fraction_um"] == fraction].set_index("sample")["is_feo"]


def _forward():
    """The forward model step05 was run with: the RELAB wavelength grid."""
    return build_forward("released")


def _forward_m3():
    """The forward model step06 was run with: resampled onto the M3 grid.

    step06 fits on M3's own ~70 bands, not the RELAB 390, so the M3 figures must
    use the same grid or the likelihood cannot even be evaluated.  Constructed
    exactly as step06 does it.
    """
    import forward
    from step04_catmip_synthetic import load_endmember_library
    from step06_m3_aristarchus import m3_wavelengths_um

    wl_um = m3_wavelengths_um()
    lam_lib, n_lib, k_lib, rho = load_endmember_library("released")
    fitw = (wl_um >= jc.FIT_LAM_MIN_UM) & (wl_um <= jc.FIT_LAM_MAX_UM)
    k_m3 = np.vstack([np.exp(np.interp(wl_um[fitw], lam_lib, np.log(k_lib[i]),
                                       left=np.nan, right=np.nan))
                      for i in range(k_lib.shape[0])])
    keep = np.isfinite(k_m3).all(axis=0)
    lam_fit = wl_um[fitw][keep]
    return forward.MixtureForward(lam_fit, n_lib, k_m3[:, keep], rho), lam_fit


def _accepted_band(fwd, theta, drop=("agglutinate",)):
    """Percentile envelope of the fresh (agglutinate-free) model over the ensemble."""
    names = list(jc.ENDMEMBERS["name"])
    nem = len(names)
    models = fwd.mixture_ssa_excluding(theta[:, :nem], theta[:, nem:],
                                       list(drop), names)
    return (np.percentile(models, ACCEPTED_LO, axis=0),
            np.percentile(models, ACCEPTED_HI, axis=0))


# --------------------------------------------------------------------------
# Figure 1
# --------------------------------------------------------------------------
def fig_01():
    ism = _measured_ism()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    wl, r = jc.load_spectrum(FIG1_FRESH[0])
    ax.plot(wl, r, color="k", lw=1.8,
            label="%s pulverized fresh basalt" % FIG1_FRESH[1])
    cmap = plt.get_cmap("viridis")
    order = sorted(FIG1_SOILS,
                   key=lambda t: float(ism[t[1]]) if t[1] in ism.index else 0.0)
    for i, (rid, name) in enumerate(order):
        wl, r = jc.load_spectrum(rid)
        lab = name if name not in ism.index else "%s  (I$_{sm}$ = %d)" % (name, ism[name])
        ax.plot(wl, r, color=cmap(i / max(1, len(order) - 1)), lw=1.2, label=lab)
    ax.axvline(jc.BAND_REF_UM, color="grey", ls="--", lw=0.9)
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("reflectance")
    ax.set_xlim(0.3, 2.6)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("(a) laboratory reflectance vs soil maturity", fontsize=10)

    ax = axes[1]
    sp = jc.TABLES / "step06_m3_spectra.csv"
    if sp.exists():
        d = pd.read_csv(sp)
        lam = d["wavelength_um"].to_numpy()
        # the paper plots reflectance; our tables carry SSA, so convert back
        fresh = hapke.ssa_to_reflectance(d["omega_mix_roi_b"].to_numpy(), jc.MU0, jc.MU)
        weath = hapke.ssa_to_reflectance(d["omega_sw_roi_b"].to_numpy(), jc.MU0, jc.MU)
        ax.plot(lam, fresh, color=RED, lw=1.6, label="freshly exposed rock (ROI mean)")
        ax.plot(lam, weath, color=BLUE, lw=1.6, label="space-weathered regolith")
        ax.axvline(jc.BAND_REF_UM, color="grey", ls="--", lw=0.9)
        ax.legend(fontsize=8)
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("reflectance")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("(b) M3 at Aristarchus, fresh vs weathered", fontsize=10)

    fig.suptitle("Figure 1 - the effect this paper corrects for", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig01_paper.png")


# --------------------------------------------------------------------------
# Figure 3 -- all three rows
# --------------------------------------------------------------------------
def fig_03():
    dc = pd.read_csv(jc.TABLES / "step02_depth_corrected_ssa.csv")
    fit = pd.read_csv(jc.TABLES / "step05_lscc_spectra.csv")
    z = np.load(jc.TABLES / "step05_posteriors.npz")
    fwd, lam_fit = _forward()
    ism = _measured_ism()

    lam_dc = dc["wavelength_um"].to_numpy()
    lam_f = fit["wavelength_um"].to_numpy()

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True)
    for j, s in enumerate(SOILS):
        lab = s if s not in ism.index else "%s  (I$_{sm}$ = %d)" % (s, ism[s])

        ax = axes[0, j]
        ax.plot(lam_dc, dc["omega_mix_" + s], color=RED, lw=1.3,
                label=r"original $\omega_{mix}$")
        ax.plot(lam_dc, dc["omega_sw"], color=BLUE, lw=1.3,
                label=r"weathered end-member $\omega_{sw}$")
        ax.plot(lam_dc, dc["omega_dc_" + s], color="k", lw=1.8,
                label=r"depth-corrected $\omega_{dc}$")
        ax.set_title(lab, fontsize=10)

        ax = axes[1, j]
        ax.plot(lam_f, fit["omega_dc_" + s], color="k", lw=1.8, label=r"$\omega_{dc}$")
        ax.plot(lam_f, fit["fit_map_" + s], color=RED, lw=1.2, label="MAP")
        ax.plot(lam_f, fit["fit_mode_" + s], color=GREEN, lw=1.0, ls="--", label="mode")

        ax = axes[2, j]
        lo, hi = _accepted_band(fwd, z[s])
        ax.fill_between(lam_fit, lo, hi, color="0.72", alpha=0.9,
                        label="accepted models (%.1f-%.1f%%)" % (ACCEPTED_LO, ACCEPTED_HI))
        ax.plot(lam_f, fit["fresh_map_" + s], color=RED, lw=1.3, label="MAP")
        ax.plot(lam_f, fit["fresh_mode_" + s], color=GREEN, lw=1.0, ls="--", label="mode")
        ax.set_xlabel("wavelength (um)")

    for ax in axes.ravel():
        ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].set_ylabel("(a)-(d)   SSA")
    axes[1, 0].set_ylabel("(e)-(h)   model fit")
    axes[2, 0].set_ylabel("(i)-(l)   weathering-corrected SSA")
    for r in range(3):
        axes[r, 0].legend(fontsize=7.5)
    fig.suptitle("Figure 3 - the four LSCC mare soils, correction and inversion",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig03_paper.png")


# --------------------------------------------------------------------------
# Figure 5(b),(c)
# --------------------------------------------------------------------------
def fig_05():
    """The two ROIs, with the pixels step06 actually used drawn as masks.

    Top row is the 970 nm reflectance the paper shows as its base map; bottom
    row is the 970 nm band depth that our percentile rule actually thresholds
    on.  The selection is recomputed here by the same rule and asserted against
    the counts step06 recorded, so the figure cannot drift from the inversion.
    """
    from step06_m3_aristarchus import block_band_depths, FRESH_PCTL, SW_PCTL

    wl = pd.read_csv(jc.M3_STAGED / "m3_wavelengths_nm.csv")
    wcol = [c for c in wl.columns if "wave" in c.lower()][0]
    lam_nm = wl[wcol].to_numpy()
    b970 = int(np.argmin(np.abs(lam_nm - jc.BAND_REF_UM * 1000.0)))
    a = pd.read_csv(jc.TABLES / "step06_m3_aristarchus.csv").set_index("roi")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, roi in enumerate(ROIS):
        block = np.load(jc.M3_STAGED / ("%s_rfl_block.npy" % roi))      # (L, B, S)
        nl, _nb, ns = block.shape
        img = block[:, b970, :].astype(float)
        img[~np.isfinite(img) | (img <= 0)] = np.nan

        depth, _ssa, _wlw, _valid = block_band_depths(block, lam_nm / 1000.0)
        good = np.isfinite(depth)
        v = depth[good]
        fresh_m = (good & (depth >= np.percentile(v, FRESH_PCTL))).reshape(nl, ns)
        weath_m = (good & (depth <= np.percentile(v, SW_PCTL))).reshape(nl, ns)
        if (int(fresh_m.sum()) != int(a.loc[roi, "n_fresh"])
                or int(weath_m.sum()) != int(a.loc[roi, "n_sw"])):
            sys.exit("[%s] FATAL: recomputed ROI selection for %s (%d/%d) does not "
                     "match step06 (%d/%d)."
                     % (STAGE, roi, fresh_m.sum(), weath_m.sum(),
                        a.loc[roi, "n_fresh"], a.loc[roi, "n_sw"]))

        ax = axes[0, col]
        lo, hi = np.nanpercentile(img, [2, 98])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, aspect="auto",
                  interpolation="nearest")
        ax.set_title("%s   %.2f N, %.2f E   -   970 nm reflectance"
                     % (roi, a.loc[roi, "lat"], a.loc[roi, "lon"]), fontsize=10)

        ax = axes[1, col]
        dmap = depth.reshape(nl, ns).astype(float)
        dmap[~good.reshape(nl, ns)] = np.nan
        ax.imshow(dmap, cmap="magma", aspect="auto", interpolation="nearest")
        for mask, colour, lab in (
                (fresh_m, "#00e5ff",
                 "fresh, p%.0f+  (n=%d)" % (FRESH_PCTL, fresh_m.sum())),
                (weath_m, "#7CFC00",
                 "weathered, p%.0f-  (n=%d)" % (SW_PCTL, weath_m.sum()))):
            ov = np.zeros(mask.shape + (4,))
            ov[..., :3] = matplotlib.colors.to_rgb(colour)
            ov[..., 3] = np.where(mask, 0.45, 0.0)
            ax.imshow(ov, aspect="auto", interpolation="nearest")
            ax.plot([], [], color=colour, lw=6, alpha=0.7, label=lab)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        ax.set_title("970 nm band depth + selection   (I$_{sm}$ %.1f, paper %.1f)"
                     % (a.loc[roi, "I_sm"], a.loc[roi, "paper_I_sm"]), fontsize=10)
        ax.set_xlabel("sample")
    axes[0, 0].set_ylabel("line (within the staged block)")
    axes[1, 0].set_ylabel("line (within the staged block)")
    fig.suptitle("Figure 5(b),(c) - the two Aristarchus ROIs; selection is our "
                 "band-depth percentile rule, not the paper's hand-drawn circles",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig05_paper.png")


# --------------------------------------------------------------------------
# Figure 6(a)-(h)
# --------------------------------------------------------------------------
def _oxides(theta, drop=("agglutinate",)):
    """Bulk oxide wt% for every accepted sample.

    The conversion the paper does not specify (D14).  Mass fractions are
    renormalised after dropping agglutinates -- matching the weathering-
    corrected material that panels (e)-(h) describe -- and each end-member
    contributes its LSCC Table 4 composition.  Returns (DataFrame, ok) where ok
    is False if the composition table is not staged.
    """
    src = jc.DATA / "ground_truth" / "endmember_oxides.csv"
    if not src.exists():
        return None, False
    comp = pd.read_csv(src).set_index("endmember")
    names = list(jc.ENDMEMBERS["name"])
    nem = len(names)
    keep = [i for i, n in enumerate(names) if n not in drop]
    m = theta[:, nem:][:, keep]
    m = m / m.sum(axis=1, keepdims=True)
    ox = [c for c in comp.columns if c not in ("lscc_table4_column",
                                               "analytical_total_wt_pct")]
    A = comp.loc[[names[i] for i in keep], ox].to_numpy()      # (nkeep, nox)
    return pd.DataFrame(m @ A, columns=ox), True


def _log_posterior(fwd, theta, data_ssa):
    """log-likelihood + Dirichlet log-prior, so the MAP particle can be located.

    step06 stores the ensemble but not which particle was the MAP, so it is
    recovered here with the same likelihood step05/step06 sampled under
    (scalar Cinv = 1/2e-4, per likelihood.m -- deviation D6).
    """
    import forward
    from catmip import dirichlet_logpdf
    nem = len(jc.ENDMEMBERS)
    loglike = forward.make_log_likelihood(fwd, data_ssa, 1.0 / 2.0e-4)
    ll = loglike(theta)
    alpha = jc.PRIOR_MEANS * jc.PRIOR_ALPHA0
    lp = dirichlet_logpdf(np.clip(theta[:, nem:], 1e-300, None), alpha)
    return np.asarray(ll) + np.asarray(lp)


def _lscc_bulk():
    """LSCC mare-soil bulk chemistry (Table 3), as a reference population.

    NOT what the paper compares against -- it uses Apollo mare *basalts* from
    ApolloBasaltDB (Cone et al. 2020), which has not been obtained.  These are
    soils, a different population, and are labelled as such on the figure.
    """
    src = jc.DATA / "ground_truth" / "lscc_bulk_chemistry.csv"
    if not src.exists():
        return None
    b = pd.read_csv(src)
    return b[b["size_fraction_um"] == "20-45"]


def fig_06():
    d = pd.read_csv(jc.TABLES / "step06_m3_spectra.csv")
    a = pd.read_csv(jc.TABLES / "step06_m3_aristarchus.csv").set_index("roi")
    lam = d["wavelength_um"].to_numpy()
    fwd, lam_fit = _forward_m3()          # step06's grid, not step05's
    names = list(jc.ENDMEMBERS["name"])
    nem = len(names)

    fig, axes = plt.subplots(5, 2, figsize=(11, 17))
    for col, roi in enumerate(ROIS):
        ax = axes[0, col]
        ax.plot(lam, d["omega_mix_" + roi], color=RED, lw=1.3, label=r"$\omega_{mix}$")
        ax.plot(lam, d["omega_sw_" + roi], color=BLUE, lw=1.3, label=r"$\omega_{sw}$")
        ax.plot(lam, d["omega_dc_" + roi], color="k", lw=1.8, label=r"$\omega_{dc}$")
        ax.set_title("%s   (a),(b)  space-weathering correction" % roi, fontsize=10)

        ax = axes[1, col]
        ax.plot(lam, d["omega_dc_" + roi], color="k", lw=1.8, label=r"$\omega_{dc}$")
        ax.plot(lam, d["fit_map_" + roi], color=RED, lw=1.2,
                label="MAP (R2 = %.4f, paper %.4f)"
                      % (a.loc[roi, "r2_map"], a.loc[roi, "paper_r2_map"]))
        ax.set_title("(c),(d)  model fit", fontsize=10)

        theta = np.load(jc.TABLES / ("step06_posterior_%s.npz" % roi))["theta"]
        ax = axes[2, col]
        lo, hi = _accepted_band(fwd, theta)
        ax.fill_between(lam_fit, lo, hi, color="0.72", alpha=0.9,
                        label="accepted models (%.1f-%.1f%%)" % (ACCEPTED_LO, ACCEPTED_HI))
        ax.plot(lam, d["fresh_map_" + roi], color=RED, lw=1.3, label="MAP")
        ax.set_title("(e),(f)  weathering-corrected SSA", fontsize=10)

        ax = axes[3, col]
        cry = [nem + names.index(n) for n in jc.CRYSTALLINE]
        f = theta[:, cry]
        f = f / f.sum(axis=1, keepdims=True) * 100.0
        parts = ax.violinplot([f[:, i] for i in range(f.shape[1])],
                              positions=range(f.shape[1]), showextrema=False,
                              widths=0.85)
        for b in parts["bodies"]:
            b.set_facecolor(BLUE)
            b.set_alpha(0.45)
        for i in range(f.shape[1]):
            ax.scatter([i], [np.median(f[:, i])], marker="_", s=220, color="k", lw=2)
        ax.set_xticks(range(f.shape[1]))
        ax.set_xticklabels([n[:5] for n in jc.CRYSTALLINE], rotation=30, fontsize=8)
        ax.set_ylabel("abundance (%)")
        ax.set_title("(g),(h)  crystalline abundances, agglutinates excluded",
                     fontsize=10)

        # (i),(j) -- oxides.  The conversion is OUR assumption (D14), so this is
        # reported and never gated; the per-sample table is written for inspection.
        ax = axes[4, col]
        ox, have = _oxides(theta)
        if have:
            x = ox["SiO2"].to_numpy()
            y = (ox["FeO"] + ox["MgO"]).to_numpy()
            ax.hist2d(x, y, bins=60, cmap="viridis")
            ox.assign(roi=roi).to_csv(
                jc.TABLES / ("step09_oxides_%s.csv" % roi), index=False)

            lp = _log_posterior(fwd, theta, d["omega_dc_" + roi].to_numpy())
            best = ox.iloc[int(np.argmax(lp))]
            ax.scatter([best["SiO2"]], [best["FeO"] + best["MgO"]], s=110,
                       facecolors="none", edgecolors=RED, lw=2.2, zorder=6,
                       label="MAP")
            med = ox.median()
            ax.scatter([med["SiO2"]], [med["FeO"] + med["MgO"]], s=110,
                       facecolors="none", edgecolors=GREEN, lw=2.2, zorder=6,
                       label="posterior median")
            bulk = _lscc_bulk()
            if bulk is not None:
                ax.scatter(bulk["SiO2"], bulk["FeO"] + bulk["MgO"], marker="*",
                           s=140, color="w", edgecolors="k", lw=0.7, zorder=7,
                           label="LSCC mare soils, bulk (not Apollo basalts)")
                print("    %s oxides: MAP SiO2 %.1f, FeO+MgO %.1f | median %.1f, %.1f"
                      "   (LSCC mare soils %.1f-%.1f, %.1f-%.1f)"
                      % (roi, best["SiO2"], best["FeO"] + best["MgO"],
                         med["SiO2"], med["FeO"] + med["MgO"],
                         bulk["SiO2"].min(), bulk["SiO2"].max(),
                         (bulk["FeO"] + bulk["MgO"]).min(),
                         (bulk["FeO"] + bulk["MgO"]).max()))
            ax.set_xlabel("SiO$_2$ (wt %)")
            ax.set_ylabel("FeO + MgO (wt %)")
            ax.legend(fontsize=7, loc="best")
            ax.set_title("(i),(j)  oxides - the conversion is OUR choice (D14)",
                         fontsize=10)
        else:
            ax.text(0.5, 0.5, "endmember_oxides.csv not staged",
                    ha="center", va="center", transform=ax.transAxes)

    for ax in axes[:3].ravel():
        ax.set_xlabel("wavelength (um)")
        ax.set_ylabel("single-scattering albedo")
        ax.legend(fontsize=7.5)
    for ax in axes.ravel():
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Figure 6 - M3 at Aristarchus, panels (a)-(j).\n"
                 "Panels (i),(j) use an end-member oxide table WE chose (LSCC Table 4); "
                 "the paper specifies none, so they are reported, never gated (D14).",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig06_paper.png")


# --------------------------------------------------------------------------
# Figure A1 -- why 970 nm
# --------------------------------------------------------------------------
def fig_a1():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    rows = []
    for _, sp in jc.SPECTRA.iterrows():
        wl, r = jc.load_spectrum(sp["relab_id"])
        wl, r = jc.restrict(wl, r)
        ssa = hapke.reflectance_to_ssa_fast(r, jc.MU0, jc.MU)
        cr = jc.convex_hull_removal(wl, ssa)
        rows.append((sp["sample"], sp["role"], wl, ssa, cr))

    for name, role, wl, ssa, cr in rows:
        c = "k" if role == "fresh" else (BLUE if role == "sw" else RED)
        axes[0].plot(wl, ssa, color=c, lw=0.9, alpha=0.85)
        axes[1].plot(wl, cr, color=c, lw=0.9, alpha=0.85)
    for ax, lab in ((axes[0], "(a) SSA"), (axes[1], "(b) continuum-removed SSA")):
        ax.axvline(jc.BAND_REF_UM, color="grey", ls="--", lw=1.0)
        ax.set_xlabel("wavelength (um)")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("single-scattering albedo")
    axes[1].set_ylabel("continuum-removed SSA")
    for c, lab in (("k", "pulverized fresh basalt"),
                   (BLUE, "mature soil (defines d_sw)"),
                   (RED, "LSCC soil")):
        axes[0].plot([], [], color=c, lw=1.2, label=lab)
    axes[0].legend(fontsize=8)

    # (c) which wavelength's band contrast actually tracks measured maturity.
    # NOTE this uses raw contrast, which Eq. (4) deliberately normalises -- so a
    # weak correlation here does not by itself contradict the paper.  See D13.
    ism45 = _measured_ism("20-45")
    ax = axes[2]
    have = [(n, wl, cr) for n, role, wl, ssa, cr in rows if n in ism45.index]
    if len(have) >= 3:
        grid = np.arange(0.60, 2.40, 0.01)
        y = np.array([float(ism45[n]) for n, _, _ in have])
        rr = []
        for lam in grid:
            dd = np.array([1.0 - float(np.interp(lam, wl, cr)) for _, wl, cr in have])
            rr.append(np.corrcoef(dd, y)[0, 1] if np.std(dd) > 0 else np.nan)
        rr = np.asarray(rr)
        ax.plot(grid, rr, color=BLUE, lw=1.5)
        ax.axhline(0.0, color="grey", lw=0.8)
        j = int(np.argmin(np.abs(grid - jc.BAND_REF_UM)))
        ax.scatter([grid[j]], [rr[j]], marker="*", s=280, facecolors="none",
                   edgecolors=RED, lw=1.8, zorder=5,
                   label="0.97 um  (r = %+.2f)" % rr[j])
        best = int(np.nanargmin(rr))
        ax.scatter([grid[best]], [rr[best]], marker="v", s=70, color=GREEN, zorder=5,
                   label="strongest: %.2f um (r = %+.2f)" % (grid[best], rr[best]))
        ax.set_ylabel("corr(band contrast, measured Is/FeO), n = %d soils" % len(have))
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("wavelength (um)")
    ax.set_title("(c) which wavelength tracks maturity", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("Figure A1 - the basis for using the 970 nm band depth", fontsize=11)
    fig.tight_layout()
    return _save(fig, "figA1_paper.png")


def main() -> int:
    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "reproducing the paper's figures in its own layout")
    made = []
    for fn in (fig_01, fig_03, fig_05, fig_06, fig_a1):
        made.append(fn().name)
    print("\n  %d figures written to figures/" % len(made))
    print("  NOT reproduced, by decision rather than omission:")
    print("    Fig. 5(a)      LROC WAC context mosaic -- external imagery, with no")
    print("                   reproduced quantity attached to it")
    print("    Fig. 6(i),(j)  drawn, but the mineral-to-oxide conversion is OUR")
    print("                   choice (LSCC Table 4) -- the paper specifies none, so")
    print("                   it is reported, never gated; see PAPER_NOTES.md D14")
    jc.verify_manifest(STAGE)
    print("\n  STEP 09 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
