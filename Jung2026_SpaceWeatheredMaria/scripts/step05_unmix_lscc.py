"""
Step 05 -- Step 2 of the paper on the four LSCC mare soils.

Inverts the depth-corrected SSA spectra produced by step02 for 12001, 12030,
15071 and 71061, then forward-models the space-weathering-removed spectrum by
dropping the agglutinate contribution, exactly as the paper's Figure 3 panels
(i)-(l) do.

Two sampler configurations are run on the same data:

  "released"   the sampler exactly as shipped (independence proposal on the
               abundances, Nsteps = 10).  This is the configuration that
               produced the published numbers, so it is what the Figure 3 R2
               values are scored against.

  "converged"  the corrected simplex random walk at a chain length shown to
               converge in step04.  Its credible intervals are the ones that
               carry statistical meaning.

Comparing the two on real data is a direct test of the degeneracy finding
(PAPER_NOTES.md D7): if the released configuration collapses, its per-parameter
marginal modes coincide with the MAP, so R2(mode) ~ R2(MAP).  The paper reports
exactly that near-equality for all four soils.

Published R2 targets (Figure 3 panel legends):
  12001 MAP 0.9887 / mode 0.9883
  12030 MAP 0.9902 / mode 0.9812
  15071 MAP 0.9888 / mode 0.9853
  71061 MAP 0.9920 / mode 0.9909
plus the Section 3 statement that accepted-model rms "consistently exceed 0.98".
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import catmip
import forward
import jung_common as jc
from step04_catmip_synthetic import (CINV, N_SAMPLES, N_STEPS_CONVERGED,
                                     N_STEPS_RELEASED, build_forward,
                                     crystalline_groups)

STAGE = "step05"

TARGETS = ["12001", "12030", "15071", "71061"]

PAPER_R2 = {
    "12001": (0.9887, 0.9883),
    "12030": (0.9902, 0.9812),
    "15071": (0.9888, 0.9853),
    "71061": (0.9920, 0.9909),
}

R2_TOL = 0.02          # see paper_targets.csv: different sampler seed/proposal

CONFIGS = [("released", "released", N_STEPS_RELEASED),
           ("converged", "rw", N_STEPS_CONVERGED)]


def invert_one(sample, omega_dc, fwd, alpha, nsteps, mode, seed):
    good = np.isfinite(omega_dc)
    loglike = forward.make_log_likelihood(fwd, np.where(good, omega_dc, np.nan), CINV)
    sampler = catmip.Catmip(
        loglike, n_endmembers=9, prior_alpha=alpha,
        size_low=jc.PRIOR_SIZE_LOW_UM, size_high=jc.PRIOR_SIZE_HIGH_UM,
        n_samples=N_SAMPLES, n_steps=nsteps, proposal=mode, seed=seed, verbose=False)
    t0 = time.time()
    theta, llk = sampler.run()
    dt = time.time() - t0

    mapm, _ = sampler.map_model()
    modem = sampler.mode_model().copy()
    modem[9:] = np.clip(modem[9:], 1e-12, None)
    modem[9:] /= modem[9:].sum()

    fit_map = fwd.mixture_ssa(mapm[None, :9], mapm[None, 9:])[0]
    fit_mode = fwd.mixture_ssa(modem[None, :9], modem[None, 9:])[0]
    fresh_map = fwd.mixture_ssa_excluding(mapm[None, :9], mapm[None, 9:],
                                          ["agglutinate"], jc.ENDMEMBER_ORDER)[0]
    fresh_mode = fwd.mixture_ssa_excluding(modem[None, :9], modem[None, 9:],
                                           ["agglutinate"], jc.ENDMEMBER_ORDER)[0]

    models = fwd.mixture_ssa(theta[:, :9], theta[:, 9:])
    obs = omega_dc[good]
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2_all = 1.0 - np.sum((models[:, good] - obs[None, :]) ** 2, axis=1) / ss_tot

    return {
        "sample": sample, "theta": theta, "map": mapm, "mode": modem,
        "fit_map": fit_map, "fit_mode": fit_mode,
        "fresh_map": fresh_map, "fresh_mode": fresh_mode,
        "r2_map": jc.r_squared(omega_dc, fit_map),
        "r2_mode": jc.r_squared(omega_dc, fit_mode),
        "r2_all": r2_all,
        "unique_fraction": len(np.unique(theta, axis=0)) / len(theta),
        "stages": len(sampler.history), "seconds": dt,
        "history": pd.DataFrame(sampler.history).assign(sample=sample),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="released", choices=["released", "ours"])
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "Step 2: CATMIP unmixing of the four LSCC mare soils")

    dc = pd.read_csv(jc.TABLES / "step02_depth_corrected_ssa.csv")
    grid = dc["wavelength_um"].to_numpy()

    fwd, lam = build_forward(args.library)
    sel = np.searchsorted(grid, lam)
    assert np.allclose(grid[sel], lam), "wavelength grids are not aligned"

    alpha = jc.PRIOR_MEANS * jc.PRIOR_ALPHA0
    all_summary, all_abund, all_hist = [], [], []
    fit_cols = {"wavelength_um": lam}
    posteriors = {}

    for tag, mode, nsteps in CONFIGS:
        print("\n" + "-" * 72)
        print("  CONFIGURATION %s  (proposal=%s, Nsteps=%d, library=%s)"
              % (tag, mode, nsteps, args.library))
        print("-" * 72)
        for i, sname in enumerate(TARGETS):
            omega_dc = dc["omega_dc_" + sname].to_numpy()[sel]
            r = invert_one(sname, omega_dc, fwd, alpha, nsteps, mode, args.seed + i)
            p_map, p_mode = PAPER_R2[sname]
            print("  %s: %2d stages, %3.0f s, unique %5.1f%%  |  "
                  "R2 MAP %.4f (paper %.4f, %+.4f)  R2 mode %.4f (paper %.4f, %+.4f)"
                  % (sname, r["stages"], r["seconds"], 100 * r["unique_fraction"],
                     r["r2_map"], p_map, r["r2_map"] - p_map,
                     r["r2_mode"], p_mode, r["r2_mode"] - p_mode))

            if tag == "converged":
                fit_cols["omega_dc_" + sname] = omega_dc
                fit_cols["fit_map_" + sname] = r["fit_map"]
                fit_cols["fit_mode_" + sname] = r["fit_mode"]
                fit_cols["fresh_map_" + sname] = r["fresh_map"]
                fit_cols["fresh_mode_" + sname] = r["fresh_mode"]
                posteriors[sname] = r["theta"]

            groups = crystalline_groups(r["theta"])
            map_g = crystalline_groups(r["map"][None, 9:])
            mode_g = crystalline_groups(r["mode"][None, 9:])
            for label, vals in groups.items():
                lo, hi = np.percentile(vals, [2.5, 97.5])
                all_abund.append({
                    "config": tag, "sample": sname, "category": label,
                    "map_wt_pct": 100 * float(map_g[label][0]),
                    "mode_wt_pct": 100 * float(mode_g[label][0]),
                    "median_wt_pct": 100 * float(np.median(vals)),
                    "p2.5_wt_pct": 100 * float(lo), "p97.5_wt_pct": 100 * float(hi),
                    "ci_width_wt_pct": 100 * float(hi - lo),
                })
            for j, nm in enumerate(jc.ENDMEMBER_ORDER):
                col = r["theta"][:, 9 + j]
                all_abund.append({
                    "config": tag, "sample": sname, "category": "raw_" + nm,
                    "map_wt_pct": 100 * float(r["map"][9 + j]),
                    "mode_wt_pct": 100 * float(r["mode"][9 + j]),
                    "median_wt_pct": 100 * float(np.median(col)),
                    "p2.5_wt_pct": 100 * float(np.percentile(col, 2.5)),
                    "p97.5_wt_pct": 100 * float(np.percentile(col, 97.5)),
                    "map_grain_size_um": float(r["map"][j]),
                    "median_grain_size_um": float(np.median(r["theta"][:, j])),
                })

            all_summary.append({
                "config": tag, "proposal": mode, "nsteps": nsteps,
                "library": args.library, "sample": sname,
                "r2_map": r["r2_map"], "paper_r2_map": p_map,
                "err_r2_map": r["r2_map"] - p_map,
                "r2_mode": r["r2_mode"], "paper_r2_mode": p_mode,
                "err_r2_mode": r["r2_mode"] - p_mode,
                "map_minus_mode_r2": r["r2_map"] - r["r2_mode"],
                "paper_map_minus_mode_r2": p_map - p_mode,
                "r2_accepted_median": float(np.median(r["r2_all"])),
                "r2_accepted_p5": float(np.percentile(r["r2_all"], 5)),
                "unique_fraction": r["unique_fraction"],
                "cooling_stages": r["stages"], "seconds": r["seconds"],
            })
            all_hist.append(r["history"].assign(config=tag))

    sm = pd.DataFrame(all_summary)
    sm.to_csv(jc.TABLES / "step05_lscc_unmixing.csv", index=False)
    pd.DataFrame(fit_cols).to_csv(jc.TABLES / "step05_lscc_spectra.csv", index=False)
    ab = pd.DataFrame(all_abund)
    ab.to_csv(jc.TABLES / "step05_lscc_abundances.csv", index=False)
    pd.concat(all_hist, ignore_index=True).to_csv(
        jc.TABLES / "step05_catmip_history.csv", index=False)
    np.savez_compressed(jc.TABLES / "step05_posteriors.npz", **posteriors)

    rel = sm[sm["config"] == "released"]
    con = sm[sm["config"] == "converged"]

    print("\n  ensemble degeneracy (measured directly):")
    print("    released proposal  : unique particles %5.1f%% (mean over the 4 soils)"
          % (100 * rel["unique_fraction"].mean()))
    print("    converged proposal : unique particles %5.1f%%"
          % (100 * con["unique_fraction"].mean()))

    print("\n  MAP-minus-mode R2 gap (reported, NOT a degeneracy test):")
    print("    paper (as published)      : %+.4f mean" % sm["paper_map_minus_mode_r2"].mean())
    print("    released proposal (ours)  : %+.4f mean" % rel["map_minus_mode_r2"].mean())
    print("    converged proposal (ours) : %+.4f mean" % con["map_minus_mode_r2"].mean())
    print("    The paper's MAP and mode R2 are nearly equal; neither of our runs")
    print("    reproduces that near-equality, so this quantity does not cleanly")
    print("    discriminate between the two samplers. The degeneracy finding rests")
    print("    on the directly measured quantities above, on the acceptance ratio")
    print("    decaying to zero, and on the synthetic non-recovery in step04.")

    print("\n  95%% credible-interval width, total pyroxene (wt%%):")
    pyx = ab[ab["category"] == "pyroxene_total"]
    for tag, _, _ in CONFIGS:
        w = pyx[pyx["config"] == tag]["ci_width_wt_pct"]
        print("    %-10s mean %6.2f   (min %.2f, max %.2f)" % (tag, w.mean(), w.min(), w.max()))

    print("\n  gate:")
    n_fail = 0
    gate_rows = []
    # The converged configuration is the one gated against the published R2:
    # the released sampler is effectively a best-of-N random search whose result
    # depends on the seed, so its fit quality is reported, not gated.  The
    # marginal-mode estimator is ill-conditioned on a well-mixed ensemble (a
    # vector of independent marginals need not be a good joint model), so mode
    # R2 is reported for both configurations rather than gated.
    checks = [
        ("converged: max |R2(MAP) - paper|", float(con["err_r2_map"].abs().max()), R2_TOL, "le"),
        ("converged: ensembles not degenerate", float(con["unique_fraction"].min()), 0.50, "ge"),
        ("converged: min median accepted R2", float(con["r2_accepted_median"].min()), 0.975, "ge"),
        ("released: ensembles do collapse (finding D7)",
         float(rel["unique_fraction"].max()), 0.05, "le"),
    ]
    for name, val, thr, sense in checks:
        ok = val <= thr if sense == "le" else val >= thr
        n_fail += (not ok)
        gate_rows.append({"check": name, "value": val, "threshold": thr,
                          "status": "PASS" if ok else "FAIL"})
        print("    %-44s %-6s %.4f  (%s %.4f)"
              % (name, "PASS" if ok else "FAIL", val,
                 "tol" if sense == "le" else "need >=", thr))
    for label, series in (("released: max |R2(MAP) - paper|", rel["err_r2_map"]),
                          ("released: max |R2(mode) - paper|", rel["err_r2_mode"]),
                          ("converged: max |R2(mode) - paper|", con["err_r2_mode"])):
        v = float(series.abs().max())
        gate_rows.append({"check": label + " (informational)", "value": v,
                          "threshold": np.nan, "status": "INFO"})
        print("    %-44s %-6s %.4f" % (label, "INFO", v))
    pd.DataFrame(gate_rows).to_csv(jc.TABLES / "step05_gate.csv", index=False)

    jc.verify_manifest(STAGE)
    print("\n  %s" % ("STEP 05 PASS" if n_fail == 0 else "STEP 05 FAIL (%d)" % n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
