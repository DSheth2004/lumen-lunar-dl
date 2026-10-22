"""
Step 01 -- self-tests for the Hapke core (paper Appendices C, D, E).

Checks hapke.py against internal identities and against the authors' released
implementation (reference/space-weathering-main/hapkeRTM/hapkeRTM.py), which is
transliterated here independently so the two can be diffed on a shared grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import hapke
import jung_common as jc

STAGE = "step01"


# --------------------------------------------------------------------------
# Independent transliteration of the authors' released Python, used as oracle.
# Kept deliberately close to the original text of hapkeRTM.py.
# --------------------------------------------------------------------------
def _oracle_chandrasekhar(ssa, x):
    gamma = np.sqrt(1 - ssa)
    r0 = (1 - gamma) / (1 + gamma)
    return (1 - ssa * x * (r0 + (1 - 2 * r0 * x) / 2 * np.log((1 + x) / x))) ** (-1)


def _oracle_ssa2ref(ssa, mu_0, mu):
    H_mu0 = _oracle_chandrasekhar(ssa, mu_0)
    H_mu = _oracle_chandrasekhar(ssa, mu)
    B, P = 0, 1
    return ssa / 4 / (mu_0 + mu) * ((1 + B) * P + H_mu0 * H_mu - 1)


def _oracle_nk2ssa(k, wi, s, n, D):
    d_mean = 2 / 3 * (n ** 2 - 1 / n * (n ** 2 - 1) ** (3 / 2)) * D
    a = 4 * np.pi * k / wi
    ri = (1 - np.sqrt(a / (a + s))) / (1 + np.sqrt(a / (a + s))) if s else np.zeros_like(np.asarray(a, float))
    ex = np.exp(-np.sqrt(a * (a + s)) * d_mean)
    th = (ri + ex) / (1 + ri * ex)
    si = 1.014 - 4 / (n * (n + 1) ** 2)
    se = ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2) + 0.05
    return se + (1 - se) * (1 - si) / (1 - si * th) * th


def main() -> int:
    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "Hapke core self-tests")

    mu0, mu = jc.MU0, jc.MU
    results = []

    def check(name, value, tol, detail="", signed=False):
        """signed=False: pass when |value| <= tol.  signed=True: pass when value <= tol."""
        if signed:
            ok = bool(np.isfinite(value) and value <= tol)
        else:
            ok = bool(np.isfinite(value) and abs(value) <= tol)
        results.append({"check": name, "value": float(value), "tol": tol,
                        "mode": "signed" if signed else "abs",
                        "status": "PASS" if ok else "FAIL", "detail": detail})
        print("  %-46s %-9s %s = %.3e (tol %.0e)"
              % (name, "PASS" if ok else "FAIL",
                 "value" if signed else "max|err|", value, tol))
        return ok

    # ---- 1. H function against the oracle -------------------------------
    w = np.linspace(1e-6, 1 - 1e-9, 4001)
    err_h = float(np.max(np.abs(hapke.chandrasekhar_h(w, mu0) - _oracle_chandrasekhar(w, mu0))))
    check("D3  H(ssa, mu0) vs released code", err_h, 1e-13)

    # ---- 2. SSA -> reflectance against the oracle -----------------------
    err_r = float(np.max(np.abs(hapke.ssa_to_reflectance(w, mu0, mu) - _oracle_ssa2ref(w, mu0, mu))))
    check("D4  ssa->r vs released code", err_r, 1e-13,
          "released form ((1+B)P + H*H - 1) with B=0,P=1 equals H*H")

    # ---- 3. round trip r -> ssa -> r ------------------------------------
    w_probe = np.linspace(0.01, 0.999, 400)
    r_probe = hapke.ssa_to_reflectance(w_probe, mu0, mu)
    w_back = hapke.reflectance_to_ssa(r_probe, mu0, mu)
    check("D1  ssa->r->ssa round trip", float(np.nanmax(np.abs(w_back - w_probe))), 1e-9)

    # ---- 4. monotonicity of r(ssa): guarantees a unique bisection root --
    r_mono = hapke.ssa_to_reflectance(w, mu0, mu)
    # strictly increasing => every forward difference is > 0 => -min(diff) < 0.
    check("D4  r(ssa) strictly increasing", float(-np.min(np.diff(r_mono))), 0.0,
          "no interior turning point, so the bisection root is unique", signed=True)

    # ---- 5. analytic limits ---------------------------------------------
    check("D3  H(0, mu) == 1", float(hapke.chandrasekhar_h(1e-15, mu) - 1.0), 1e-12)
    check("D4  r(ssa->0) == 0", float(hapke.ssa_to_reflectance(1e-15, mu0, mu)), 1e-14)

    # ---- 6. nk -> ssa against the oracle, per end-member -----------------
    lam = np.linspace(0.4, 2.5, 300)
    worst = 0.0
    for _, em in jc.ENDMEMBERS.iterrows():
        n = np.full_like(lam, em["n_code"])
        k = np.full_like(lam, 5e-4)
        got = hapke.nk_to_ssa(n, k, em["D_used_um"], lam)
        exp = _oracle_nk2ssa(k, lam, 0.0, em["n_code"], em["D_used_um"])
        worst = max(worst, float(np.max(np.abs(got - exp))))
    check("E3-E9  nk->ssa vs released code (9 EM)", worst, 1e-13)

    # ---- 7. k inversion round trip --------------------------------------
    n_ol = 1.83
    k_true = np.geomspace(1e-6, 1e-2, 60)
    lam_k = np.full_like(k_true, 1.0)
    ssa_t = hapke.nk_to_ssa(np.full_like(k_true, n_ol), k_true, 80.0, lam_k)
    r_t = hapke.ssa_to_reflectance(ssa_t, mu0, mu)
    k_back = hapke.reflectance_to_k(r_t, lam_k, n_ol, 80.0, mu0, mu)
    rel = np.nanmax(np.abs(k_back - k_true) / k_true)
    check("E2  k->r->k round trip (relative)", float(rel), 1e-6)

    # ---- 8. mixing identities -------------------------------------------
    rho = jc.ENDMEMBERS["rho_code"].to_numpy()
    D = jc.ENDMEMBERS["D_used_um"].to_numpy()
    wt = jc.PRIOR_MEANS
    f = hapke.mass_to_cross_section(rho, D, wt)
    check("C2  sum(f_i) == 1", float(abs(f.sum() - 1.0)), 1e-15)
    # a single-phase mixture must return that phase's own SSA
    stack = np.random.default_rng(0).random((9, 50))
    onehot = np.zeros(9)
    onehot[3] = 1.0
    check("C1  one-hot mixture returns that end-member",
          float(np.max(np.abs(hapke.mix_ssa(onehot, stack) - stack[3]))), 1e-15)

    # ---- 9. quantify the Eq. E9 discrepancy (paper text vs released code) -
    lam_e9 = np.linspace(0.55, 2.5, 200)
    rows_e9 = []
    for _, em in jc.ENDMEMBERS.iterrows():
        n = np.full_like(lam_e9, em["n_code"])
        k = np.full_like(lam_e9, 1e-3)
        a = hapke.nk_to_ssa(n, k, em["D_used_um"], lam_e9, paper_form=False)
        b = hapke.nk_to_ssa(n, k, em["D_used_um"], lam_e9, paper_form=True)
        rows_e9.append({"end_member": em["name"], "n": em["n_code"],
                        "D_um": em["D_used_um"],
                        "mean_ssa_code_form": float(a.mean()),
                        "mean_ssa_paper_form": float(b.mean()),
                        "max_abs_diff": float(np.max(np.abs(a - b)))})
    e9 = pd.DataFrame(rows_e9)
    e9.to_csv(jc.TABLES / "step01_eq_E9_discrepancy.csv", index=False)
    print("\n  Eq. E9 printed-vs-code difference (informational, not a gate):")
    print("    max |dSSA| across the 9 end-members = %.4f" % e9["max_abs_diff"].max())
    print("    -> written to tables/step01_eq_E9_discrepancy.csv")

    df = pd.DataFrame(results)
    df.to_csv(jc.TABLES / "step01_hapke_selftest.csv", index=False)
    n_fail = int((df["status"] == "FAIL").sum())
    print("\n%d/%d self-tests passed" % (len(df) - n_fail, len(df)))

    jc.verify_manifest(STAGE)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
