"""
Step 03 -- Appendices D and E: the nine compositional end-members.

Three products:

  1. FIG C1 -- SSA of each end-member, obtained by applying Appendix D to its
     measured RELAB reflectance.  Checked against the published panel ranges of
     paper Figure C1.

  2. OUR optical constants -- k(lambda) recovered from the measured reflectance
     by bisection on Eq. E2, with n and D fixed per Table C1:

         f(k) = r_Hapke( nk_to_ssa(n, k, D, lambda) ) - r_measured

     Self-consistency is gated on the round trip k -> r reproducing the measured
     spectrum to bisection tolerance.

  3. THE AUTHORS' optical constants, read from the Zenodo release
     ("data-this study/<mineral>.txt": wavelength in metres, n, k).  These are
     what the published MCMC actually consumed, so they are the PRIMARY library
     for reproducing Figures 3, 4 and 6; ours is carried as a variant so the
     audit can quantify the sensitivity.

FINDING -- the two libraries are not interchangeable.  The authors' k reproduces
the measured end-member spectra only at an effective grain size about 4.7x
SMALLER than the D_used printed in Table C1 (e.g. olivine ~17 um against a
tabulated 80 um); k and D enter the model almost purely as the product k*D, so
the released constants and the released grain-size table are internally
inconsistent by that factor.  Quantified per end-member in
tables/step03_effective_grain_size.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import hapke
import jung_common as jc

STAGE = "step03"

ZENODO_FILE = {
    "olivine": "olivine.txt",
    "plagioclase": "plagioclase.txt",
    "pigeonite": "pigeonite.txt",
    "augite": "augite.txt",
    "ilmenite": "ilmenite.txt",
    "pyroxene_glass": "pyroxeneglass.txt",
    "plagioclase_glass": "plagioclaseglass.txt",
    "orange_glass": "orangeglass.txt",
    "agglutinate": "agglutinate.txt",
}

# Published Figure C1 panel ranges (axis limits read from the figure), used only
# as a loose bracket: the computed SSA must lie inside the plotted panel.
FIG_C1_RANGE = {
    "olivine": (0.40, 0.95),
    "plagioclase": (0.82, 0.98),
    "pigeonite": (0.29, 0.92),
    "augite": (0.19, 0.88),
    "ilmenite": (0.32, 0.85),
    "pyroxene_glass": (0.17, 0.81),
    "plagioclase_glass": (0.09, 0.66),
    "orange_glass": (0.19, 0.79),
    "agglutinate": (0.18, 0.56),
}

GRID = np.arange(0.40, 2.5001, 0.005)          # common wavelength grid, microns


def read_zenodo_optical_constants(name: str):
    """Read the authors' released (lambda_m, n, k) table; returns lambda in microns."""
    path = jc.ZENODO / "mcmc" / "data-this study" / ZENODO_FILE[name]
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        elif len(parts) == 2:                   # k blank on a few agglutinate rows
            rows.append([float(parts[0]), float(parts[1]), np.nan])
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0] * 1e6, arr[:, 1], arr[:, 2]


def effective_grain_size(zl, zn, zk, wl, refl, lo=1.0, hi=400.0):
    """Grain size at which the authors' k reproduces the measured reflectance."""
    m = np.isfinite(zk) & (zk > 0)
    zl_m, zn_m, zk_m = zl[m], zn[m], zk[m]
    r_meas = np.interp(zl_m, wl, refl)

    def resid(D):
        ssa = hapke.nk_to_ssa(zn_m, zk_m, float(D), zl_m)
        return float(np.median(hapke.ssa_to_reflectance(ssa, jc.MU0, jc.MU) - r_meas))

    try:
        if resid(lo) * resid(hi) > 0:
            return np.nan
        return float(brentq(resid, lo, hi, xtol=1e-4))
    except (ValueError, FloatingPointError):
        return np.nan


def main() -> int:
    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "Appendices D/E: end-member SSA and optical constants")

    figc1 = {"wavelength_um": GRID}
    k_ours = {"wavelength_um": GRID}
    k_theirs = {"wavelength_um": GRID}
    ssa_ours = {"wavelength_um": GRID}
    ssa_theirs = {"wavelength_um": GRID}
    rows = []

    for _, em in jc.ENDMEMBERS.iterrows():
        name = em["name"]
        n_used = float(em["n_code"])
        D_used = float(em["D_used_um"])

        wl, refl = jc.load_spectrum(em["relab_id"])
        ok = np.isfinite(refl) & (refl > 0)
        wl, refl = wl[ok], refl[ok]

        # ---- (1) Figure C1: Appendix D applied to the measured spectrum ----
        ssa_meas = hapke.reflectance_to_ssa(refl, jc.MU0, jc.MU)
        fin = np.isfinite(ssa_meas)
        figc1["ssa_" + name] = np.interp(GRID, wl[fin], ssa_meas[fin],
                                         left=np.nan, right=np.nan)
        win = fin & (wl >= 0.40) & (wl <= 2.60)
        obs_lo, obs_hi = float(ssa_meas[win].min()), float(ssa_meas[win].max())
        pub_lo, pub_hi = FIG_C1_RANGE[name]
        in_panel = (obs_lo >= pub_lo - 0.02) and (obs_hi <= pub_hi + 0.02)

        # ---- (2) our optical constants (Eq. E2) ---------------------------
        k_meas = hapke.reflectance_to_k(refl, wl, n_used, D_used, jc.MU0, jc.MU)
        good = np.isfinite(k_meas) & (k_meas > 0)

        # round-trip self-consistency
        ssa_rt = hapke.nk_to_ssa(np.full(good.sum(), n_used), k_meas[good], D_used, wl[good])
        r_rt = hapke.ssa_to_reflectance(ssa_rt, jc.MU0, jc.MU)
        rt_err = float(np.max(np.abs(r_rt - refl[good])))

        kg = np.exp(np.interp(GRID, wl[good], np.log(k_meas[good]),
                              left=np.nan, right=np.nan))
        k_ours["k_" + name] = kg
        ssa_ours["ssa_" + name] = hapke.nk_to_ssa(np.full_like(GRID, n_used), kg, D_used, GRID)

        # ---- (3) the authors' optical constants ---------------------------
        zl, zn, zk = read_zenodo_optical_constants(name)
        zok = np.isfinite(zk) & (zk > 0)
        zkg = np.exp(np.interp(GRID, zl[zok], np.log(zk[zok]), left=np.nan, right=np.nan))
        zng = np.interp(GRID, zl, zn)
        k_theirs["k_" + name] = zkg
        ssa_theirs["ssa_" + name] = hapke.nk_to_ssa(zng, zkg, D_used, GRID)

        d_eff = effective_grain_size(zl, zn, zk, wl, refl)
        n_zen = float(np.median(zn))

        both = np.isfinite(kg) & np.isfinite(zkg)
        rel = np.abs(kg[both] - zkg[both]) / zkg[both]

        rows.append({
            "end_member": name,
            "relab_id": em["relab_id"],
            "n_table_C1": em["n_paper"],
            "n_released_file": n_zen,
            "n_agrees_with_C1": bool(abs(n_zen - em["n_paper"]) < 1e-9),
            "D_used_table_C1_um": D_used,
            "D_effective_of_released_k_um": d_eff,
            "D_ratio_C1_over_effective": D_used / d_eff if np.isfinite(d_eff) else np.nan,
            "our_k_roundtrip_max_abs_err": rt_err,
            "figC1_ssa_min": obs_lo,
            "figC1_ssa_max": obs_hi,
            "figC1_published_min": pub_lo,
            "figC1_published_max": pub_hi,
            "figC1_within_panel": in_panel,
            "k_median_rel_diff_vs_released": float(np.nanmedian(rel)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(jc.TABLES / "step03_effective_grain_size.csv", index=False)
    pd.DataFrame(figc1).to_csv(jc.TABLES / "step03_figC1_endmember_ssa.csv", index=False)
    pd.DataFrame(k_ours).to_csv(jc.TABLES / "step03_endmember_k_ours.csv", index=False)
    pd.DataFrame(k_theirs).to_csv(jc.TABLES / "step03_endmember_k_released.csv", index=False)
    pd.DataFrame(ssa_ours).to_csv(jc.TABLES / "step03_endmember_ssa_ours.csv", index=False)
    pd.DataFrame(ssa_theirs).to_csv(jc.TABLES / "step03_endmember_ssa_released.csv", index=False)

    print("\n  Figure C1 check -- SSA of each end-member from Appendix D:")
    print("  %-18s %16s %18s   %s" % ("end-member", "computed", "published panel", "in panel"))
    for _, r in df.iterrows():
        print("  %-18s %6.3f - %-6.3f %7.2f - %-7.2f   %s"
              % (r["end_member"], r["figC1_ssa_min"], r["figC1_ssa_max"],
                 r["figC1_published_min"], r["figC1_published_max"],
                 "yes" if r["figC1_within_panel"] else "NO"))

    print("\n  effective grain size of the RELEASED optical constants:")
    print("  %-18s %12s %14s %10s" % ("end-member", "D_used (C1)", "D_effective", "ratio"))
    for _, r in df.iterrows():
        print("  %-18s %10.0f um %11.1f um %9.2fx"
              % (r["end_member"], r["D_used_table_C1_um"],
                 r["D_effective_of_released_k_um"], r["D_ratio_C1_over_effective"]))

    n_bad_n = int((~df["n_agrees_with_C1"]).sum())
    if n_bad_n:
        for _, r in df[~df["n_agrees_with_C1"]].iterrows():
            print("\n  NOTE: %s uses n = %.2f in the released file but Table C1 prints %.2f"
                  % (r["end_member"], r["n_released_file"], r["n_table_C1"]))

    # ---- gates ----------------------------------------------------------
    checks = [
        ("our k round-trips to measured r", float(df["our_k_roundtrip_max_abs_err"].max()), 1e-6),
        ("Fig C1 SSA inside published panel (9/9)",
         float((~df["figC1_within_panel"]).sum()), 0.0),
    ]
    print("\n  gate:")
    n_fail = 0
    gate_rows = []
    for name, val, tol in checks:
        ok = val <= tol
        n_fail += (not ok)
        gate_rows.append({"check": name, "value": val, "tol": tol,
                          "status": "PASS" if ok else "FAIL"})
        print("    %-42s %-6s %.3e  (tol %.0e)" % (name, "PASS" if ok else "FAIL", val, tol))
    pd.DataFrame(gate_rows).to_csv(jc.TABLES / "step03_gate.csv", index=False)

    jc.verify_manifest(STAGE)
    print("\n  %s" % ("STEP 03 PASS" if n_fail == 0 else "STEP 03 FAIL (%d)" % n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
