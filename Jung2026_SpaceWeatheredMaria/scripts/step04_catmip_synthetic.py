"""
Step 04 -- gate the CATMIP sampler on a synthetic mixture with a known answer.

Before the sampler is pointed at real data it has to recover a planted mixture.
A synthetic SSA spectrum is generated with the same forward model the likelihood
uses, noise is added at the released covariance (Cov = 2e-4), and the inversion
must (a) reach the log-likelihood of the truth, (b) not degenerate, and (c) put
the planted composition inside its 95% credible interval.

Two configurations are run:

  "released"  the sampler exactly as shipped (independence proposal on the
              abundances, Nsteps = 10).  Reported, not gated -- it degenerates,
              which is itself one of the findings (see PAPER_NOTES.md, D7).

  "rw"        random walk on the simplex with the Hastings correction and a
              chain long enough to converge.  This one is gated.

Recovery is scored in the four categories Figure 4 of the paper reports:
olivine, plagioclase, TOTAL pyroxene (pigeonite + augite combined) and ilmenite.
The paper combines the two pyroxenes because "LSCC pyroxene abundances were
reported as 'total pyroxene'", and because it notes that splitting the pyroxene
chemistry across only two end-members makes the individual split unreliable --
which our synthetic test reproduces independently.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import catmip
import forward
import jung_common as jc

STAGE = "step04"

COV = 2.0e-4                      # load_data.m
CINV = 1.0 / COV
N_SAMPLES = 2000                  # load_data.m Nm
N_STEPS_RELEASED = 10             # load_data.m Nsteps
N_STEPS_CONVERGED = 100           # see PAPER_NOTES.md D8

# Planted truth: a plausible mare basalt, deliberately NOT the prior mean.
TRUTH_ABUNDANCE = np.array([0.12, 0.30, 0.10, 0.28, 0.08, 0.10, 0.01, 0.005, 0.005])
TRUTH_SIZE = np.array([60.0, 120.0, 45.0, 80.0, 35.0, 55.0, 70.0, 90.0, 100.0])

# Figure 4 reporting categories, over the crystalline phases only.
FIG4_GROUPS = {
    "olivine": ["olivine"],
    "plagioclase": ["plagioclase"],
    "pyroxene_total": ["pigeonite", "augite"],
    "ilmenite": ["ilmenite"],
}


def load_endmember_library(which="released"):
    """(lam_um, n, k, rho) for the nine end-members on the common grid."""
    suffix = "released" if which == "released" else "ours"
    k = pd.read_csv(jc.TABLES / ("step03_endmember_k_%s.csv" % suffix))
    lam = k["wavelength_um"].to_numpy()
    kk = np.vstack([k["k_" + nm].to_numpy() for nm in jc.ENDMEMBER_ORDER])
    n = jc.ENDMEMBERS["n_code"].to_numpy()
    rho = jc.ENDMEMBERS["rho_code"].to_numpy()
    return lam, n, kk, rho


def build_forward(which="released", lo=jc.FIT_LAM_MIN_UM, hi=jc.FIT_LAM_MAX_UM):
    lam, n, kk, rho = load_endmember_library(which)
    m = (lam >= lo) & (lam <= hi) & np.isfinite(kk).all(axis=0)
    return forward.MixtureForward(lam[m], n, kk[:, m], rho), lam[m]


def crystalline_groups(theta):
    """Renormalised Figure-4 categories from an ensemble of abundance vectors."""
    nem = len(jc.ENDMEMBER_ORDER)
    m = np.atleast_2d(theta)[:, nem:] if theta.shape[-1] == 2 * nem else np.atleast_2d(theta)
    idx = [jc.ENDMEMBER_ORDER.index(n) for n in jc.CRYSTALLINE]
    mc = m[:, idx]
    mc = mc / mc.sum(axis=1, keepdims=True)
    cols = {}
    for label, members in FIG4_GROUPS.items():
        j = [jc.CRYSTALLINE.index(n) for n in members]
        cols[label] = mc[:, j].sum(axis=1)
    return cols


def main() -> int:
    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "CATMIP synthetic recovery test")

    fwd, lam = build_forward("released")
    print("  forward model: %d end-members x %d bands (%.2f-%.2f um)"
          % (fwd.nem, fwd.nlam, lam.min(), lam.max()))

    truth = np.hstack([TRUTH_SIZE, TRUTH_ABUNDANCE])[None, :]
    clean = fwd.mixture_ssa(truth[:, :9], truth[:, 9:])[0]
    rng = np.random.default_rng(12345)
    noisy = clean + rng.normal(0.0, np.sqrt(COV), size=clean.shape)
    ceiling = jc.r_squared(noisy, clean)
    loglike = forward.make_log_likelihood(fwd, noisy, CINV)
    llk_truth = float(loglike(truth)[0])
    print("  planted spectrum: SSA %.3f-%.3f, noise sigma = %.4f" % (clean.min(), clean.max(), np.sqrt(COV)))
    print("  log-likelihood at the truth = %.2f ; achievable R2 ceiling = %.4f"
          % (llk_truth, ceiling))

    truth_groups = {k: float(v[0]) for k, v in
                    crystalline_groups(TRUTH_ABUNDANCE[None, :]).items()}

    alpha = jc.PRIOR_MEANS * jc.PRIOR_ALPHA0
    runs = {}
    for tag, mode, nsteps in (("released", "released", N_STEPS_RELEASED),
                              ("converged", "rw", N_STEPS_CONVERGED)):
        print("\n  --- %s (proposal=%s, Nsteps=%d) ---" % (tag, mode, nsteps))
        sampler = catmip.Catmip(
            loglike, n_endmembers=9, prior_alpha=alpha,
            size_low=jc.PRIOR_SIZE_LOW_UM, size_high=jc.PRIOR_SIZE_HIGH_UM,
            n_samples=N_SAMPLES, n_steps=nsteps, proposal=mode, seed=7, verbose=False)
        t0 = time.time()
        theta, llk = sampler.run()
        dt = time.time() - t0

        mapm, _ = sampler.map_model()
        r2 = jc.r_squared(noisy, fwd.mixture_ssa(mapm[None, :9], mapm[None, 9:])[0])
        uniq = len(np.unique(theta, axis=0)) / len(theta)
        groups = crystalline_groups(theta)

        rows = []
        n_in = 0
        for label, vals in groups.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            inside = bool(lo <= truth_groups[label] <= hi)
            n_in += inside
            rows.append({"run": tag, "category": label,
                         "truth": truth_groups[label],
                         "median": float(np.median(vals)),
                         "p2.5": float(lo), "p97.5": float(hi),
                         "truth_in_95CI": inside})
        runs[tag] = {
            "rows": pd.DataFrame(rows), "r2": r2, "uniq": uniq,
            "max_llk": float(np.nanmax(llk)), "n_in": n_in,
            "stages": len(sampler.history), "seconds": dt,
            "n_eval": len(sampler.history) * N_SAMPLES * nsteps,
            "history": pd.DataFrame(sampler.history).assign(run=tag),
        }
        print("    %d stages, %.1e evaluations, %.0f s" % (runs[tag]["stages"], runs[tag]["n_eval"], dt))
        print("    max logL = %8.2f (truth %.2f)   R2(MAP) = %.4f (ceiling %.4f)"
              % (runs[tag]["max_llk"], llk_truth, r2, ceiling))
        print("    unique particles = %.1f%%   truth inside 95%% CI = %d/4"
              % (100 * uniq, n_in))
        for _, r in runs[tag]["rows"].iterrows():
            print("      %-16s truth %.3f  median %.3f  [%.3f, %.3f] %s"
                  % (r["category"], r["truth"], r["median"], r["p2.5"], r["p97.5"],
                     "yes" if r["truth_in_95CI"] else "NO"))

    pd.concat([runs[t]["rows"] for t in runs], ignore_index=True).to_csv(
        jc.TABLES / "step04_synthetic_recovery.csv", index=False)
    pd.concat([runs[t]["history"] for t in runs], ignore_index=True).to_csv(
        jc.TABLES / "step04_catmip_history.csv", index=False)
    pd.DataFrame([
        {"run": t, "max_log_likelihood": runs[t]["max_llk"],
         "log_likelihood_at_truth": llk_truth,
         "R2_MAP": runs[t]["r2"], "R2_ceiling": ceiling,
         "unique_particle_fraction": runs[t]["uniq"],
         "truth_in_95CI_of_4": runs[t]["n_in"],
         "cooling_stages": runs[t]["stages"],
         "likelihood_evaluations": runs[t]["n_eval"],
         "seconds": runs[t]["seconds"]}
        for t in runs]).to_csv(jc.TABLES / "step04_synthetic_summary.csv", index=False)

    # ---- gate (only the converged configuration is gated) ----------------
    c = runs["converged"]
    checks = [
        ("reaches the truth log-likelihood", c["max_llk"], llk_truth - 2.0, "ge"),
        ("R2(MAP) reaches the noise ceiling", c["r2"], ceiling - 0.002, "ge"),
        ("ensemble not degenerate", c["uniq"], 0.50, "ge"),
        ("truth in 95% CI, Fig-4 categories", float(c["n_in"]), 4.0, "ge"),
    ]
    print("\n  gate (converged run):")
    n_fail = 0
    gate_rows = []
    for name, val, thr, sense in checks:
        ok = val >= thr if sense == "ge" else val <= thr
        n_fail += (not ok)
        gate_rows.append({"check": name, "value": val, "threshold": thr,
                          "status": "PASS" if ok else "FAIL"})
        print("    %-38s %-6s %.4f  (need >= %.4f)" % (name, "PASS" if ok else "FAIL", val, thr))

    print("\n  released configuration, reported not gated:")
    print("    unique particles = %.1f%%  ->  %s"
          % (100 * runs["released"]["uniq"],
             "DEGENERATE (see PAPER_NOTES.md D7)" if runs["released"]["uniq"] < 0.1 else "ok"))
    gate_rows.append({"check": "released proposal unique fraction (informational)",
                      "value": runs["released"]["uniq"], "threshold": np.nan,
                      "status": "INFO"})
    pd.DataFrame(gate_rows).to_csv(jc.TABLES / "step04_gate.csv", index=False)

    jc.verify_manifest(STAGE)
    print("\n  %s" % ("STEP 04 PASS" if n_fail == 0 else "STEP 04 FAIL (%d)" % n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
