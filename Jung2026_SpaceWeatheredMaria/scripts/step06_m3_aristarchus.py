"""
Step 06 -- apply the workflow to M3 data near Aristarchus (paper Figs. 5 and 6).

Downloads only the image lines covering the two regions of interest (see
m3_fetch.py), builds a "fresh" and a "space weathered" spectrum for each, runs
the Step 1 band-depth correction with a SCENE-DERIVED d_sw, and inverts the
depth-corrected SSA with CATMIP.

Published targets
-----------------
Figure 5(b)  d_mix 0.211  d_sw 0.083  Delta 0.536  theta_sw 0.064  I_sm  6.4
Figure 5(c)  d_mix 0.140  d_sw 0.061  Delta 0.733  theta_sw 0.206  I_sm 20.6
Figure 6(c)  R2 MAP 0.9755  mode 0.9657
Figure 6(d)  R2 MAP 0.9870  mode 0.9842

WHAT IS AND IS NOT INDEPENDENTLY CHECKABLE HERE
-----------------------------------------------
The paper selects its ROIs by hand -- the red and blue circles of Figure 5 --
and a hand-drawn selection cannot be recovered from the publication.  So this
step separates two different claims:

  GATED, and free of circularity: feed the paper's OWN published d_mix and d_sw
  into Eqs. (4) and (3) and check that they reproduce the paper's own published
  Delta, theta_sw and I_sm.  This tests the orbital arithmetic exactly.

  REPORTED, with the caveat stated: our own ROI selection.  Measuring the 970 nm
  band depth of every valid pixel in each staged block shows that the published
  d_mix sits at the 92.8th and 93.7th percentile of the two blocks, and the
  published d_sw at the 27.1st and 18.0th.  The agreement of those percentiles
  across two independent scenes is what motivates the percentile rule used
  below -- but the rule was chosen AFTER seeing the published values, so the
  numbers it produces are a characterisation of the paper's selection, not an
  independent prediction of it.  They are reported, never gated.

OTHER DEVIATIONS
----------------
* Thermal correction.  The paper uses M3 reflectance corrected per Li & Milliken
  (2016).  We use the PDS L2 product, whose label records
  THERMAL_CORRECTION_FLAG = "Y" -- the standard correction applied during L2
  processing, not Li & Milliken's.  The two differ mainly beyond ~2 um.
* M3 fills the two shortest bands (461, 501 nm) with -999.  Treated as NoData;
  they lie outside the 0.55-2.50 um fitting window.

Photometry.  M3 L2 is photometrically corrected to i = 30 deg, e = 0 deg,
g = 30 deg -- the same geometry as the RELAB laboratory spectra -- so the same
Appendix D inversion applies to both without rescaling.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import catmip
import forward
import hapke
import jung_common as jc
import m3_fetch
from step02_band_depth import COEF_EQ3, PAPER_D_DC, ism_from_delta
from step04_catmip_synthetic import (CINV, N_SAMPLES, N_STEPS_CONVERGED,
                                     crystalline_groups, load_endmember_library)

STAGE = "step06"

PAPER_F5 = {
    "roi_b": {"d_mix": 0.211, "d_sw": 0.083, "Delta": 0.536, "theta_sw": 0.064, "I_sm": 6.4},
    "roi_c": {"d_mix": 0.140, "d_sw": 0.061, "Delta": 0.733, "theta_sw": 0.206, "I_sm": 20.6},
}
PAPER_F6_R2 = {"roi_b": (0.9755, 0.9657), "roi_c": (0.9870, 0.9842)}

M3_NODATA = -999.0
FRESH_PCTL = 93.0        # see the module docstring: post-hoc, reported not gated
SW_PCTL = 22.0


def m3_wavelengths_um() -> np.ndarray:
    return pd.read_csv(jc.M3_STAGED / "m3_wavelengths_nm.csv")["wavelength_nm"].to_numpy() * 1e-3


def block_band_depths(cube, wl_um):
    """970 nm depth of every valid pixel in a block, plus the SSA array."""
    L, B, S = cube.shape
    cube = np.where(cube <= M3_NODATA + 1.0, np.nan, cube)
    win = (wl_um >= jc.FIT_LAM_MIN_UM) & (wl_um <= jc.FIT_LAM_MAX_UM)
    wlw = wl_um[win]

    refl = cube[:, win, :].transpose(0, 2, 1).reshape(L * S, win.sum())
    valid = np.isfinite(refl).all(axis=1) & (refl > 0).all(axis=1)
    ssa = hapke.reflectance_to_ssa_fast(refl, jc.MU0, jc.MU)

    depth = np.full(L * S, np.nan)
    for i in np.where(valid)[0]:
        s = ssa[i]
        if np.isfinite(s).all():
            depth[i] = jc.band_depth_at(wlw, jc.convex_hull_removal(wlw, s), jc.BAND_REF_UM)
    return depth, ssa, wlw, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", default="rw")
    ap.add_argument("--nsteps", type=int, default=N_STEPS_CONVERGED)
    ap.add_argument("--library", default="released")
    ap.add_argument("--skip-inversion", action="store_true")
    args = ap.parse_args()

    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "M3 at Aristarchus: Figures 5 and 6")

    wl_um = m3_wavelengths_um()
    print("  M3 grid: %d bands, %.1f-%.1f nm" % (len(wl_um), 1e3 * wl_um.min(), 1e3 * wl_um.max()))

    lam_lib, n_lib, k_lib, rho = load_endmember_library(args.library)
    fitw = (wl_um >= jc.FIT_LAM_MIN_UM) & (wl_um <= jc.FIT_LAM_MAX_UM)
    k_m3 = np.vstack([np.exp(np.interp(wl_um[fitw], lam_lib, np.log(k_lib[i]),
                                       left=np.nan, right=np.nan))
                      for i in range(k_lib.shape[0])])
    keep = np.isfinite(k_m3).all(axis=0)
    lam_fit = wl_um[fitw][keep]
    fwd = forward.MixtureForward(lam_fit, n_lib, k_m3[:, keep], rho)
    print("  forward model on %d M3 bands (%.3f-%.3f um)"
          % (fwd.nlam, lam_fit.min(), lam_fit.max()))

    alpha = jc.PRIOR_MEANS * jc.PRIOR_ALPHA0
    rows, abund_rows = [], []
    spec_cols = {"wavelength_um": lam_fit}

    # ---- (1) the closed-form check, using the paper's own band depths -----
    print("\n  closed-form check of Eqs. (4) and (3) on the published band depths:")
    closed = []
    for roi, p in PAPER_F5.items():
        delta = (PAPER_D_DC - p["d_mix"]) / (PAPER_D_DC - p["d_sw"])
        ism = ism_from_delta(delta, COEF_EQ3)
        closed.append({"roi": roi, "Delta_from_paper_d": delta,
                       "paper_Delta": p["Delta"], "err_Delta": delta - p["Delta"],
                       "theta_sw_from_paper_d": ism / 100.0,
                       "paper_theta_sw": p["theta_sw"],
                       "err_theta_sw": ism / 100.0 - p["theta_sw"],
                       "I_sm_from_paper_d": ism, "paper_I_sm": p["I_sm"],
                       "err_I_sm": ism - p["I_sm"]})
        print("    %s: Delta %.4f (paper %.3f, %+.4f)   theta_sw %.4f (paper %.3f, %+.4f)"
              "   I_sm %.2f (paper %.1f, %+.2f)"
              % (roi, delta, p["Delta"], delta - p["Delta"],
                 ism / 100.0, p["theta_sw"], ism / 100.0 - p["theta_sw"],
                 ism, p["I_sm"], ism - p["I_sm"]))
    closed = pd.DataFrame(closed)
    closed.to_csv(jc.TABLES / "step06_closed_form_check.csv", index=False)

    # ---- (2) our own ROI selection ---------------------------------------
    for roi, cfg in m3_fetch.SCENES.items():
        print("\n  --- %s : %s ---" % (roi, cfg["label"]))
        block = np.load(jc.M3_STAGED / ("%s_rfl_block.npy" % roi))
        meta = pd.read_csv(jc.M3_STAGED / ("%s_location.csv" % roi)).iloc[0]
        print("    block %s from %s, nearest pixel %.4f deg from (%.2fN, %.2fE)"
              % (block.shape, cfg["product"], meta["dist_deg"], cfg["lat"], cfg["lon"]))

        depth, ssa, wlw, valid = block_band_depths(block, wl_um)
        good = np.isfinite(depth)
        v = depth[good]
        p = PAPER_F5[roi]

        from scipy.stats import percentileofscore
        pct_mix = float(percentileofscore(v, p["d_mix"]))
        pct_sw = float(percentileofscore(v, p["d_sw"]))
        print("    %d valid pixels; paper d_mix sits at p%.1f, paper d_sw at p%.1f"
              % (good.sum(), pct_mix, pct_sw))

        d_fresh_thr = float(np.percentile(v, FRESH_PCTL))
        d_sw_thr = float(np.percentile(v, SW_PCTL))
        fresh_sel = np.where(good & (depth >= d_fresh_thr))[0]
        sw_sel = np.where(good & (depth <= d_sw_thr))[0]

        fresh = np.nanmean(ssa[fresh_sel], axis=0)
        sw = np.nanmean(ssa[sw_sel], axis=0)
        d_mix = jc.band_depth_at(wlw, jc.convex_hull_removal(wlw, fresh), jc.BAND_REF_UM)
        d_sw = jc.band_depth_at(wlw, jc.convex_hull_removal(wlw, sw), jc.BAND_REF_UM)
        delta = (PAPER_D_DC - d_mix) / (PAPER_D_DC - d_sw)
        ism = ism_from_delta(delta, COEF_EQ3)
        theta_sw = ism / 100.0

        print("    our ROI rule (p%.0f fresh / p%.0f weathered), REPORTED not gated:"
              % (FRESH_PCTL, SW_PCTL))
        print("      d_mix %.3f (paper %.3f)   d_sw %.3f (paper %.3f)"
              % (d_mix, p["d_mix"], d_sw, p["d_sw"]))
        print("      Delta %.3f (paper %.3f)   theta_sw %.3f (paper %.3f)   I_sm %.1f (paper %.1f)"
              % (delta, p["Delta"], theta_sw, p["theta_sw"], ism, p["I_sm"]))

        omega_dc = (fresh - sw * theta_sw) / (1.0 - theta_sw)
        spec_cols["omega_mix_" + roi] = np.interp(lam_fit, wlw, fresh)
        spec_cols["omega_sw_" + roi] = np.interp(lam_fit, wlw, sw)
        spec_cols["omega_dc_" + roi] = np.interp(lam_fit, wlw, omega_dc)

        row = {"roi": roi, "product": cfg["product"], "lat": cfg["lat"], "lon": cfg["lon"],
               "n_valid_pixels": int(good.sum()),
               "paper_d_mix_percentile": pct_mix, "paper_d_sw_percentile": pct_sw,
               "n_fresh": int(len(fresh_sel)), "n_sw": int(len(sw_sel)),
               "d_mix": d_mix, "d_sw": d_sw, "Delta": delta,
               "theta_sw": theta_sw, "I_sm": ism,
               "paper_d_mix": p["d_mix"], "paper_d_sw": p["d_sw"],
               "paper_Delta": p["Delta"], "paper_theta_sw": p["theta_sw"],
               "paper_I_sm": p["I_sm"],
               "err_d_mix": d_mix - p["d_mix"], "err_d_sw": d_sw - p["d_sw"],
               "err_Delta": delta - p["Delta"], "err_I_sm": ism - p["I_sm"]}

        if not args.skip_inversion:
            obs = spec_cols["omega_dc_" + roi]
            loglike = forward.make_log_likelihood(fwd, obs, CINV)
            sampler = catmip.Catmip(loglike, 9, alpha,
                                    size_low=jc.PRIOR_SIZE_LOW_UM,
                                    size_high=jc.PRIOR_SIZE_HIGH_UM,
                                    n_samples=N_SAMPLES, n_steps=args.nsteps,
                                    proposal=args.proposal, seed=23, verbose=False)
            t0 = time.time()
            theta, llk = sampler.run()
            mapm, _ = sampler.map_model()
            modem = sampler.mode_model().copy()
            modem[9:] = np.clip(modem[9:], 1e-12, None)
            modem[9:] /= modem[9:].sum()
            fit_map = fwd.mixture_ssa(mapm[None, :9], mapm[None, 9:])[0]
            r2_map = jc.r_squared(obs, fit_map)
            r2_mode = jc.r_squared(obs, fwd.mixture_ssa(modem[None, :9], modem[None, 9:])[0])
            pm, pmo = PAPER_F6_R2[roi]
            print("      R2 MAP  %.4f (paper %.4f, %+.4f)   R2 mode %.4f (paper %.4f, %+.4f)"
                  % (r2_map, pm, r2_map - pm, r2_mode, pmo, r2_mode - pmo))
            print("      %d stages, %.0f s, unique particles %.1f%%"
                  % (len(sampler.history), time.time() - t0,
                     100 * len(np.unique(theta, axis=0)) / len(theta)))
            spec_cols["fit_map_" + roi] = fit_map
            spec_cols["fresh_map_" + roi] = fwd.mixture_ssa_excluding(
                mapm[None, :9], mapm[None, 9:], ["agglutinate"], jc.ENDMEMBER_ORDER)[0]
            row.update({"r2_map": r2_map, "paper_r2_map": pm, "err_r2_map": r2_map - pm,
                        "r2_mode": r2_mode, "paper_r2_mode": pmo,
                        "err_r2_mode": r2_mode - pmo,
                        "unique_fraction": len(np.unique(theta, axis=0)) / len(theta)})
            for label, vals in crystalline_groups(theta).items():
                abund_rows.append({"roi": roi, "category": label,
                                   "median_wt_pct": 100 * float(np.median(vals)),
                                   "p2.5_wt_pct": 100 * float(np.percentile(vals, 2.5)),
                                   "p97.5_wt_pct": 100 * float(np.percentile(vals, 97.5))})
            np.savez_compressed(jc.TABLES / ("step06_posterior_%s.npz" % roi), theta=theta)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(jc.TABLES / "step06_m3_aristarchus.csv", index=False)
    pd.DataFrame(spec_cols).to_csv(jc.TABLES / "step06_m3_spectra.csv", index=False)
    if abund_rows:
        pd.DataFrame(abund_rows).to_csv(jc.TABLES / "step06_m3_abundances.csv", index=False)

    # ---- gate ------------------------------------------------------------
    print("\n  gate (closed-form only; ROI-derived values are reported, not gated):")
    n_fail = 0
    gate_rows = []
    checks = [
        ("Delta from the paper's own d values", float(closed["err_Delta"].abs().max()), 0.002),
        ("theta_sw from the paper's own d values", float(closed["err_theta_sw"].abs().max()), 0.002),
        ("I_sm from the paper's own d values", float(closed["err_I_sm"].abs().max()), 0.2),
    ]
    for name, val, tol in checks:
        ok = val <= tol
        n_fail += (not ok)
        gate_rows.append({"check": name, "value": val, "tol": tol,
                          "status": "PASS" if ok else "FAIL"})
        print("    %-42s %-6s %.4f  (tol %.4f)" % (name, "PASS" if ok else "FAIL", val, tol))
    for label, col in (("ROI d_mix vs paper", "err_d_mix"), ("ROI d_sw vs paper", "err_d_sw"),
                       ("ROI I_sm vs paper", "err_I_sm")):
        v = float(df[col].abs().max())
        gate_rows.append({"check": label + " (informational)", "value": v,
                          "threshold": np.nan, "status": "INFO"})
        print("    %-42s %-6s %.4f" % (label, "INFO", v))
    if "err_r2_map" in df:
        v = float(df["err_r2_map"].abs().max())
        gate_rows.append({"check": "ROI R2 MAP vs paper (informational)", "value": v,
                          "threshold": np.nan, "status": "INFO"})
        print("    %-42s %-6s %.4f" % ("ROI R2 MAP vs paper", "INFO", v))
    pd.DataFrame(gate_rows).to_csv(jc.TABLES / "step06_gate.csv", index=False)

    jc.verify_manifest(STAGE)
    print("\n  %s" % ("STEP 06 PASS" if n_fail == 0 else "STEP 06 FAIL (%d)" % n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
