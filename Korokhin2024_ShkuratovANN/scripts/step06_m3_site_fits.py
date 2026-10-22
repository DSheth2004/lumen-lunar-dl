"""
Step 06 -- the numerical-optimization fits at the four M3 test sites.

  Table 9  -- the 8-parameter model (gray albedo 0.15) at sites 1-4; Fig. 12.
  Table 5  -- site 1 refitted for gray albedos 0.05 ... 0.50.
  Table 6  -- the same at site 4.
  Table 8  -- the agglutinate LS-CMP-045 replaces the gray component.
  Table 10 -- site 1 from starting c_PLG = 0, 10, 20, 50 %: non-uniqueness.

Grain-size corridor: S0 = 1 ... 100 um in 1-um steps, extended in 5-um steps to
500 um ("for young lunar areas, the upper limit is shifted to larger values, up to
500 um", Sec. 2.3) -- applied to every site alike, since which site is "young" is
itself a result. Start point and facets as step03 (D6). The site spectra are the
recipe-processed cube at the refined site pixels (step05); Table 9 is refitted on the
no-destripe and approx_v1 cubes as the D2 sensitivities.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import korokhin_common as kc
import shkuratov as sk
from step03_basis_set import START, fit_8param

STAGE = "step06"
S0_GRID = np.r_[np.arange(1.0, 101.0, 1.0), np.arange(105.0, 501.0, 5.0)]
A_GRAY_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
PLG_STARTS = (0.0, 10.0, 20.0, 50.0)


def site_spectra(cube_name: str) -> dict:
    cube = np.load(kc.PRODUCTS / ("scene21_%s.npy" % cube_name))
    sites = pd.read_csv(kc.PRODUCTS / "sites_rows.csv")
    return {int(s["site"]): cube[int(s["row"]), :, int(s["sample"])].astype(float)
            for _, s in sites.iterrows()}


def fit_row(model, obs, label: dict, start=START) -> dict:
    r = fit_8param(model, obs, s0_grid=S0_GRID, start=start)
    th = r["theta"]
    row = dict(label)
    row.update(dict(zip(kc.PARAMS, th)))
    row.update({"sum_c": float(th[2:].sum()), "sigma": r["sigma"],
                "sigma_rel": 100 * r["sigma"] / float(np.mean(obs)), "mean_albedo": float(np.mean(obs)),
                "best_S0": r["s0"], "seconds": r["seconds"]})
    return row, r["model"]


def show(title, df, key):
    print("\n  %s" % title)
    print("  " + " ".join("%8s" % c for c in [key] + kc.PARAMS + ["sum_c", "sig_rel"]))
    for _, r in df.iterrows():
        print("  " + " ".join("%8.3f" % r[c] for c in [key] + kc.PARAMS + ["sum_c", "sigma_rel"]))


def main() -> int:
    kc.banner(STAGE, "Nelder-Mead fits at the four M3 sites (Tables 5, 6, 8, 9, 10)")
    kc.verify_manifest(STAGE)
    lam = kc.M3_GRID_UM
    spec = site_spectra("destriped")
    base = sk.LunarModel(lam)

    # Table 9 + Fig. 12
    t9, models = [], {}
    for site, obs in spec.items():
        row, mod = fit_row(base, obs, {"site": site, "cube": "destriped"})
        t9.append(row)
        models[site] = mod
    for variant in ("no_destripe", "approx_v1"):                  # D2 sensitivities (step05)
        for site, obs in site_spectra(variant).items():
            row, _ = fit_row(base, obs, {"site": site, "cube": variant})
            t9.append(row)
    t9 = pd.DataFrame(t9)
    t9.to_csv(kc.TABLES / "step06_table9.csv", index=False)
    show("Table 9 (destriped)", t9[t9["cube"] == "destriped"], "site")

    # Tables 5 and 6
    for site, name in ((1, "table5"), (4, "table6")):
        rows = []
        for a in A_GRAY_SWEEP:
            m = sk.LunarModel(lam, a_gray=a)
            row, _ = fit_row(m, spec[site], {"A_Gr": a, "tau_Gr": float(sk.tau_from_albedo(a, 2.0))})
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(kc.TABLES / ("step06_%s.csv" % name), index=False)
        show("%s (site %d, gray-albedo sweep)" % (name.title(), site), df, "A_Gr")

    # Table 8 -- agglutinate instead of gray
    l, a = kc.relab_spectrum("c2ls45")
    agg = sk.LunarModel(lam, agglutinate=kc.resample(l, a, lam))
    rows = []
    for site, obs in spec.items():
        row, _ = fit_row(agg, obs, {"site": site})
        rows.append(row)
    t8 = pd.DataFrame(rows)
    t8.to_csv(kc.TABLES / "step06_table8.csv", index=False)
    show("Table 8 (agglutinate LS-CMP-045 in place of the gray component; 'Gr' = c_AGG)", t8, "site")

    # Table 10 -- starting c_PLG at site 1
    rows = []
    for p0 in PLG_STARTS:
        st = START.copy()
        st[kc.PARAMS.index("PLG")] = p0
        row, _ = fit_row(base, spec[1], {"initial_PLG": p0}, start=st)
        rows.append(row)
    t10 = pd.DataFrame(rows)
    t10.to_csv(kc.TABLES / "step06_table10.csv", index=False)
    show("Table 10 (site 1, starting c_PLG)", t10, "initial_PLG")

    # sensitivity: the same four starts WITHOUT the S corridor (one simplex from S0 = 30 um).
    # With the full corridor all four starts reach one solution; this shows whether the
    # paper's start-dependence appears when the search is less exhaustive.
    rows = []
    for p0 in PLG_STARTS:
        st = START.copy()
        st[kc.PARAMS.index("PLG")] = p0
        r = fit_8param(base, spec[1], s0_grid=np.array([30.0]), start=st)
        row = {"initial_PLG": p0}
        row.update(dict(zip(kc.PARAMS, r["theta"])))
        row.update({"sum_c": float(r["theta"][2:].sum()), "sigma_rel": 100 * r["sigma"] / spec[1].mean()})
        rows.append(row)
    t10s = pd.DataFrame(rows)
    t10s.to_csv(kc.TABLES / "step06_table10_single_start.csv", index=False)
    show("Table 10 sensitivity: single simplex from S0 = 30 um, no corridor", t10s, "initial_PLG")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, site in zip(axes.flat, (1, 2, 3, 4)):
        ax.plot(lam, spec[site], "b.-", label="measured")
        ax.plot(lam, models[site], "r.-", label="model")
        ax.plot(lam, models[site] - spec[site] + 0.1, "m.-", label="difference + 0.1")
        r = t9[(t9["site"] == site) & (t9["cube"] == "destriped")].iloc[0]
        ax.set_title("Fig. 12(%s) site %d: sigma_Rel %.2f%%" % ("abcd"[site - 1], site, r["sigma_rel"]),
                     fontsize=9)
        ax.set_ylim(0, 0.5); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(kc.FIGURES / "fig12_site_fits_nm.png", dpi=120); plt.close(fig)
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
