"""
Step 02 -- Paper 1 (Jung et al. 2026) on Copernicus, M3 and IIRS.

The method, reused from ../Jung2026_SpaceWeatheredMaria/scripts (step02, step04,
step06) without changes to its physics or sampler:
  1. reflectance -> single-scattering albedo (Hapke, Appendix D; i = 30, e = 0 deg);
  2. 970-nm band depth of the continuum-removed SSA (one upper convex hull per
     spectrum over the configuration's window);
  3. Eqs. (4) and (3): Delta = (d_dc - d_mix) / (d_dc - d_sw),
     theta_sw = 0.6568 Delta^3.731;
  4. the depth-corrected spectrum omega_dc = (omega_mix - theta_sw omega_sw) / (1 - theta_sw);
  5. CATMIP inversion of omega_dc with the nine-end-member Hapke mixture (released
     library, converged 100-step sampler, the same settings as step06).

Targets, per configuration:
  * each zone's mean SSA spectrum (central, floor, wall_rim, proximal_ejecta), with
    the scene's weathered reference as omega_sw;
  * "scene_fresh": the paper's own ROI rule from step06 -- the mean of the pixels at or
    above the 93rd band-depth percentile of the scene.
  omega_sw = the mean of the pixels at or below the 22nd percentile (step06 rule).
Per-pixel maps (an extension -- the paper works on ROI spectra): 970-nm band depth and
I_sm with the scene omega_sw as reference.

The lab anchors d_dc and d_sw (fresh basalts 15058/15555/70017/70035 and mature soils
15041/79221/10084) are re-measured on each configuration's window from the same RELAB
spectra step02 uses, so the correction always compares like with like. On 0.55-2.50 um
this gives d_dc 0.356 (paper 0.358).

Photometry: M3 L2 is normalised to i = 30, e = 0 deg. The IIRS product's photometric
geometry is not documented in the delivery; the same geometry is assumed and flagged.
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cc_common as cc

sys.path.insert(0, str(cc.JUNG_SCRIPTS))
import catmip                                   # noqa: E402
import forward                                  # noqa: E402
import hapke                                    # noqa: E402
import jung_common as jc                        # noqa: E402
from step02_band_depth import COEF_EQ3, FRESH, SW, ism_from_delta, spectra_as_ssa   # noqa: E402
from step04_catmip_synthetic import (CINV, N_SAMPLES, N_STEPS_CONVERGED,   # noqa: E402
                                     crystalline_groups, load_endmember_library)
from step06_m3_aristarchus import FRESH_PCTL, SW_PCTL   # noqa: E402

STAGE = "step02"


def window(config: str, wl_um: np.ndarray) -> np.ndarray:
    if config == "M3_full":
        return (wl_um >= jc.FIT_LAM_MIN_UM) & (wl_um <= jc.FIT_LAM_MAX_UM)
    return cc.in_common(wl_um)


def hull_cr(wl: np.ndarray, y: np.ndarray) -> np.ndarray:
    return jc.convex_hull_removal(wl, y)


def lab_anchors(config: str) -> tuple[float, float]:
    ssa = spectra_as_ssa()
    d = {}
    for s, (wl, y, _r) in ssa.items():
        w = window(config, wl)
        d[s] = jc.band_depth_at(wl[w], hull_cr(wl[w], y[w]), jc.BAND_REF_UM)
    return float(np.mean([d[s] for s in FRESH])), float(np.mean([d[s] for s in SW]))


def scene_ssa(config: str):
    wl, cube = cc.config_cube(config)
    w = window(config, wl)
    joint = np.load(cc.PRODUCTS / "joint_mask.npy")
    refl = cube[w][:, joint].T                          # (n, bands)
    ssa = hapke.reflectance_to_ssa_fast(refl, jc.MU0, jc.MU)
    return wl[w], ssa


def pixel_depths(wl, ssa) -> np.ndarray:
    out = np.full(ssa.shape[0], np.nan)
    for i in range(ssa.shape[0]):
        s = ssa[i]
        if np.isfinite(s).all():
            out[i] = jc.band_depth_at(wl, hull_cr(wl, s), jc.BAND_REF_UM)
    return out


def build_forward(wl):
    lam_lib, n_lib, k_lib, rho = load_endmember_library("released")
    k = np.vstack([np.exp(np.interp(wl, lam_lib, np.log(k_lib[i]), left=np.nan, right=np.nan))
                   for i in range(k_lib.shape[0])])
    keep = np.isfinite(k).all(axis=0)
    return forward.MixtureForward(wl[keep], n_lib, k[:, keep], rho), keep


def invert(fwd, obs, nsteps, seed=23):
    alpha = jc.PRIOR_MEANS * jc.PRIOR_ALPHA0
    loglike = forward.make_log_likelihood(fwd, obs, CINV)
    smp = catmip.Catmip(loglike, 9, alpha, size_low=jc.PRIOR_SIZE_LOW_UM,
                        size_high=jc.PRIOR_SIZE_HIGH_UM, n_samples=N_SAMPLES,
                        n_steps=nsteps, proposal="rw", seed=seed, verbose=False)
    theta, _ = smp.run()
    mapm, _ = smp.map_model()
    fit = fwd.mixture_ssa(mapm[None, :9], mapm[None, 9:])[0]
    return theta, jc.r_squared(obs, fit), fit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=N_STEPS_CONVERGED)
    ap.add_argument("--configs", nargs="*", default=cc.CONFIGS)
    ap.add_argument("--skip-inversion", action="store_true")
    a = ap.parse_args()

    cc.banner(STAGE, "Paper 1 (Jung 2026) on Copernicus, M3 and IIRS")
    cc.verify_manifest(STAGE)
    jc.verify_manifest(STAGE)
    joint = np.load(cc.PRODUCTS / "joint_mask.npy")
    zone = np.load(cc.PRODUCTS / "zone_map.npy")[joint]

    rows, abund, spectra_out = [], [], []
    for config in a.configs:
        t0 = time.time()
        d_dc, d_sw_lab = lab_anchors(config)
        wl, ssa = scene_ssa(config)
        depth = pixel_depths(wl, ssa)
        good = np.isfinite(depth)
        thr_sw = float(np.nanpercentile(depth, SW_PCTL))
        thr_fr = float(np.nanpercentile(depth, FRESH_PCTL))
        sw = np.nanmean(ssa[good & (depth <= thr_sw)], axis=0)
        d_sw = jc.band_depth_at(wl, hull_cr(wl, sw), jc.BAND_REF_UM)
        print("\n  %s: %d bands %.3f-%.3f um; lab d_dc %.4f (d_sw lab %.4f); scene d_sw %.4f;"
              " %d/%d pixels with a depth"
              % (config, wl.size, wl[0], wl[-1], d_dc, d_sw_lab, d_sw, good.sum(), good.size))

        # per-pixel maps
        delta_px = (d_dc - depth) / (d_dc - d_sw)
        ism_px = np.where(delta_px > 0, ism_from_delta(np.clip(delta_px, 0, None), COEF_EQ3), np.nan)
        for name, v in (("depth970", depth), ("Ism", ism_px)):
            img = np.full(joint.shape, np.nan)
            img[joint] = v
            cc.save_map(cc.PRODUCTS / ("paper1_%s_%s.tif" % (config, name)), img)
        np.save(cc.PRODUCTS / ("p1_%s_depth970.npy" % config), depth.astype(np.float32))

        fwd, keep = build_forward(wl)
        targets = [(zname, good & (zone == i)) for i, (zname, _, _) in enumerate(cc.ZONES)]
        targets.append(("scene_fresh", good & (depth >= thr_fr)))
        for tname, sel in targets:
            mix = np.nanmean(ssa[sel], axis=0)
            d_mix = jc.band_depth_at(wl, hull_cr(wl, mix), jc.BAND_REF_UM)
            delta = (d_dc - d_mix) / (d_dc - d_sw)
            ism = float(ism_from_delta(max(delta, 0.0), COEF_EQ3))
            th = ism / 100.0
            omega_dc = (mix - sw * th) / (1.0 - th)
            row = {"config": config, "target": tname, "n_pixels": int(sel.sum()),
                   "d_dc_lab": d_dc, "d_sw_scene": d_sw, "d_mix": d_mix, "Delta": delta,
                   "theta_sw": th, "I_sm": ism,
                   "mean_ssa": float(np.mean(mix))}
            if not a.skip_inversion:
                t1 = time.time()
                theta, r2, fit = invert(fwd, omega_dc[keep], a.nsteps)
                row.update({"r2_map": r2, "minutes": (time.time() - t1) / 60})
                for label, vals in crystalline_groups(theta).items():
                    abund.append({"config": config, "target": tname, "category": label,
                                  "median_wt_pct": 100 * float(np.median(vals)),
                                  "p2.5_wt_pct": 100 * float(np.percentile(vals, 2.5)),
                                  "p97.5_wt_pct": 100 * float(np.percentile(vals, 97.5))})
                np.savez_compressed(cc.PRODUCTS / ("p1_posterior_%s_%s.npz" % (config, tname)),
                                    theta=theta)
                for k, lam in enumerate(wl[keep]):
                    spectra_out.append({"config": config, "target": tname, "wavelength_um": lam,
                                        "omega_mix": mix[keep][k], "omega_sw": sw[keep][k],
                                        "omega_dc": omega_dc[keep][k], "fit_map": fit[k]})
            rows.append(row)
            print("    %-16s n %6d  d_mix %.4f  Delta %.3f  I_sm %5.1f%s"
                  % (tname, sel.sum(), d_mix, delta, ism,
                     ("  R2 %.4f (%.1f min)" % (row["r2_map"], row["minutes"])) if "r2_map" in row else ""),
                  flush=True)
        print("    %s done in %.1f min" % (config, (time.time() - t0) / 60))

    suffix = "" if set(a.configs) == set(cc.CONFIGS) else "_" + "_".join(a.configs)
    pd.DataFrame(rows).to_csv(cc.TABLES / ("step02_paper1_targets%s.csv" % suffix), index=False)
    if abund:
        pd.DataFrame(abund).to_csv(cc.TABLES / ("step02_paper1_abundances%s.csv" % suffix), index=False)
        pd.DataFrame(spectra_out).to_csv(cc.TABLES / ("step02_paper1_spectra%s.csv" % suffix), index=False)

    cc.verify_manifest(STAGE)
    jc.verify_manifest(STAGE)
    print("\n  STEP 02 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
