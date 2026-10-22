"""
Step 01 -- the Shkuratov model (shkuratov.py) against itself and against the paper.

Gates (exit non-zero if any fails):
  G1 tau_Gr(A_Gr) for the six albedos of Tables 5/6 equals the printed value to
     the printed rounding (0.0005).
  G2 Eq. A2 inverts A8-A11 exactly: A -> tau -> A round trip < 1e-10 for n = 1.6
     and 2.0 over the representable albedo range.
  G3 physical limits: a lossless medium (tau = 0) has albedo 1.
  G4 whole chain: a basis mineral, converted to k at S = 30 um (A1) and fed back
     through the mixture model alone (c = 100 %, S = 30 um, nFe = 0), reproduces
     its own lab spectrum (< 1e-10).
  G5 monotonicity: albedo falls as S grows (transparent minerals) and as nFe grows.

Reported, and used to pick the primary alpha_Fe (rule fixed in the plan):
  alpha_Fe from Eqs. A14-A15 with Fe n,k (Johnson & Christy + Ordal) vs the
  digitized Fig. A1. If the computed curve stays within +-15 % of Fig. A1 over
  0.54-2.54 um it is primary; otherwise the digitized Fig. A1 is.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import korokhin_common as kc
import shkuratov as sk

STAGE = "step01"


def main() -> int:
    kc.banner(STAGE, "Shkuratov-model self-tests and the paper's closed-form values")
    kc.verify_manifest(STAGE)
    gates = []

    # G1 -- Table 5/6 tau_Gr
    t5 = pd.read_csv(kc.PAPER_TABLES / "t05_gray_albedo_site1.csv")
    tau = sk.tau_from_albedo(t5["A_Gr"].to_numpy(), kc.N_OPAQUE)
    g1 = pd.DataFrame({"A_Gr": t5["A_Gr"], "tau_printed": t5["tau_Gr"], "tau_ours": tau,
                       "abs_diff": np.abs(tau - t5["tau_Gr"])})
    g1.to_csv(kc.TABLES / "step01_tau_gray.csv", index=False)
    gates.append(("G1 tau_Gr = Table 5 to printed rounding", float(g1["abs_diff"].max()), 0.0005))
    print(g1.round(5).to_string(index=False))

    # G2 -- round trip, G3 -- limits
    worst = 0.0
    floors = {}
    for n in (kc.N_TRANSPARENT, kc.N_OPAQUE):
        floors[n] = sk.albedo_floor(n)
        A = np.linspace(floors[n] + 1e-3, 0.95, 400)
        back = sk.single_albedo(sk.tau_from_albedo(A, n), n)
        worst = max(worst, float(np.nanmax(np.abs(back - A))))
    gates.append(("G2 A -> tau -> A round trip", worst, 1e-10))
    lossless = max(abs(float(sk.single_albedo(0.0, n)) - 1.0) for n in (1.6, 2.0))
    gates.append(("G3 lossless medium has albedo 1", lossless, 1e-12))
    print("  darkest representable albedo: n=1.6 -> %.4f, n=2.0 -> %.4f"
          % (floors[kc.N_TRANSPARENT], floors[kc.N_OPAQUE]))

    # G4 -- whole chain on the Table 4 basis, on the lab grid 0.30-2.60 um
    lam = np.round(np.arange(0.30, 2.6001, 0.005), 3)
    comps = sk.lunar_basis_components(lam, gray=False)
    chain = 0.0
    for c in comps:
        l0, a0 = kc.relab_spectrum(kc.BASIS[c.name]["spectrum"])
        a_lab = kc.resample(l0, a0, lam)
        mix = sk.Mixture(lam, [c])
        a_back = mix.forward(kc.S_LAB_UM, np.array([1.0]), 0.0)
        ok = np.isfinite(a_lab) & np.isfinite(a_back)
        d = float(np.max(np.abs(a_back[ok] - a_lab[ok])))
        print("  %s (%s): %d/%d wavelengths representable, max |A_back - A_lab| = %.2e"
              % (c.name, kc.BASIS[c.name]["spectrum"], ok.sum(), lam.size, d))
        chain = max(chain, d)
    gates.append(("G4 basis spectrum survives k-conversion + mixture", chain, 1e-10))

    # G5 -- monotonicity
    m = sk.LunarModel(kc.M3_GRID_UM, alpha_fe_um=sk.alpha_fe(kc.M3_GRID_UM))
    base = np.array([30.0, 0.05, 30.0, 25.0, 2.0, 2.0, 25.0, 10.0])
    a_s = [m.forward(np.r_[s, base[1:]]).mean() for s in (10, 30, 60, 120)]
    a_n = [m.forward(np.r_[base[0], f, base[2:]]).mean() for f in (0.0, 0.05, 0.1, 0.2)]
    mono = float(np.all(np.diff(a_s) < 0) and np.all(np.diff(a_n) < 0))
    gates.append(("G5 albedo decreases with S and with nFe", 1.0 - mono, 0.0))

    # alpha_Fe: computed vs digitized Fig. A1
    dig = pd.read_csv(kc.DIGITIZED / "figA1_alpha_fe.csv")
    comp = sk.alpha_fe(dig["lam_um"].to_numpy(), kc.N_TRANSPARENT)
    ratio = comp / dig["alpha_fe"].to_numpy()
    inside = (dig["lam_um"] >= 0.54) & (dig["lam_um"] <= 2.54)
    worst_dev = float(np.max(np.abs(ratio[inside] - 1.0)))
    primary = "computed" if worst_dev <= 0.15 else "digitized"
    pd.DataFrame({"lam_um": dig["lam_um"], "alpha_fig_a1": dig["alpha_fe"], "alpha_computed": comp,
                  "ratio": ratio}).to_csv(kc.TABLES / "step01_alpha_fe.csv", index=False)
    below = dig["lam_um"] <= 1.94
    pd.DataFrame({"primary": [primary], "worst_rel_dev_054_254": [worst_dev],
                  "median_ratio_054_194": [float(np.median(ratio[inside & below]))],
                  "median_ratio_194_254": [float(np.median(ratio[inside & ~below]))],
                  "rule": ["computed if within +-15% of Fig. A1 over 0.54-2.54 um, else digitized"]}
                 ).to_csv(kc.TABLES / "step01_alpha_fe_choice.csv", index=False)
    print("  alpha_Fe computed/Fig.A1: median %.3f below 1.94 um (Johnson & Christy), %.3f above"
          " (Ordal substitute); worst deviation %.1f%% -> primary = %s"
          % (np.median(ratio[inside & below]), np.median(ratio[inside & ~below]),
             100 * worst_dev, primary))

    g = pd.DataFrame(gates, columns=["gate", "value", "tolerance"])
    g["pass"] = g["value"] <= g["tolerance"]
    g.to_csv(kc.TABLES / "step01_gates.csv", index=False)
    print()
    for _, r in g.iterrows():
        print("  %-52s %.2e (tol %.0e)  %s" % (r["gate"], r["value"], r["tolerance"],
                                               "PASS" if r["pass"] else "FAIL"))
    kc.verify_manifest(STAGE)
    return 0 if g["pass"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
