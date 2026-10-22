"""
Step 02 -- Step 1 of the paper: semiempirical band-depth correction.

Reproduces paper Table 1 and produces the depth-corrected SSA spectra (omega_dc)
that Step 2 (CATMIP unmixing) consumes.

    (1)  omega_mix = theta_dc*omega_dc + theta_sw*omega_sw,  theta_dc + theta_sw = 1
    (2)  omega_dc  = (omega_mix - theta_sw*omega_sw) / (1 - theta_sw)
    (3)  theta_sw  = I_sm/100 = C * Delta^3.731
    (4)  Delta     = (d_dc(970) - d_mix(970)) / (d_dc(970) - d_sw(970))

d(lambda) is the band depth of the continuum-removed SINGLE SCATTERING ALBEDO.

Note on Eq. (3)'s coefficient C.  The paper prints two values -- 0.6568 in
Eq. (3) (i.e. C = 65.68) and "65.58" in the Fig. 2 annotation.  Both are carried
here and scored against Table 1; see tables/step02_eq3_coefficient.csv.

Note on d_dc.  The authors' HapkeCorrection hard-codes d_dc = 0.6422, which is a
continuum-removed VALUE; the paper's Table 1 quotes 0.358, the corresponding
DEPTH (1 - 0.6422 = 0.3578).  The two conventions agree, and Eq. (4) is invariant
under the swap.  We work in depths throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import hapke
import jung_common as jc

STAGE = "step02"

# Paper Table 1 / Fig. 2 reference values.
PAPER_D_DC = 0.358
PAPER_D_SW = 0.120
PAPER_TABLE1 = pd.DataFrame(
    [("12001", "LR-CMP-010", 56, 0.145, 0.893, 43.04),
     ("12030", "LR-CMP-032", 14, 0.163, 0.819, 31.26),
     ("15071", "LR-CMP-020", 52, 0.161, 0.826, 32.13),
     ("71061", "LR-CMP-057", 14, 0.144, 0.897, 43.85)],
    columns=["sample", "relab_code", "I_sm_reported", "d_mix", "Delta", "I_sm_calc"],
).set_index("sample")

EXPONENT = 3.731
COEF_EQ3 = 65.68        # paper Eq. (3), theta_sw = 0.6568 * Delta^3.731
COEF_FIG2 = 65.58       # paper Fig. 2 annotation

# Samples defining the two anchors (paper Section 2.1 and Table 1 note).
FRESH = ["15058", "15555", "70017", "70035"]
SW = ["15041", "79221", "10084"]


def ism_from_delta(delta, coef=COEF_EQ3):
    return coef * np.power(delta, EXPONENT)


def spectra_as_ssa() -> dict:
    """Load every Apollo spectrum and convert reflectance -> SSA (Eq. D1)."""
    out = {}
    for _, sp in jc.SPECTRA.iterrows():
        wl, refl = jc.load_spectrum(sp["relab_id"])
        ssa = hapke.reflectance_to_ssa(refl, jc.MU0, jc.MU)
        ok = np.isfinite(ssa)
        out[sp["sample"]] = (wl[ok], ssa[ok], refl[ok])
    return out


def band_depths(ssa_by_sample: dict, lo=None, hi=None) -> dict:
    """970 nm band depth of the continuum-removed SSA, one hull per spectrum."""
    d = {}
    for s, (wl, ssa, _r) in ssa_by_sample.items():
        w, y = (wl, ssa) if lo is None else jc.restrict(wl, ssa, lo, hi)
        d[s] = jc.band_depth_at(w, jc.convex_hull_removal(w, y), jc.BAND_REF_UM)
    return d


def main() -> int:
    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "Step 1: semiempirical band-depth correction")

    ssa_by_sample = spectra_as_ssa()

    # ---- band depths over the full native range (what the released code does)
    d = band_depths(ssa_by_sample)
    d_dc = float(np.mean([d[s] for s in FRESH]))
    d_sw = float(np.mean([d[s] for s in SW]))
    print("  d_dc = %.4f  (paper %.3f)   from %s" % (d_dc, PAPER_D_DC, ", ".join(FRESH)))
    print("  d_sw = %.4f  (paper %.3f)   from %s" % (d_sw, PAPER_D_SW, ", ".join(SW)))

    # ---- Table 1 ---------------------------------------------------------
    rows = []
    for _, sp in jc.SPECTRA[jc.SPECTRA["role"] == "soil"].iterrows():
        s = sp["sample"]
        delta = (d_dc - d[s]) / (d_dc - d_sw)
        row = {
            "sample": s,
            "relab_id": sp["relab_id"],
            "relab_code": sp["relab_code"],
            "d_mix": d[s],
            "Delta": delta,
            "I_sm_calc": ism_from_delta(delta),
            "theta_sw": ism_from_delta(delta) / 100.0,
            "is_target": bool(sp["is_target"]),
        }
        if s in PAPER_TABLE1.index:
            p = PAPER_TABLE1.loc[s]
            row.update({
                "paper_d_mix": p["d_mix"], "paper_Delta": p["Delta"],
                "paper_I_sm_calc": p["I_sm_calc"],
                "err_d_mix": row["d_mix"] - p["d_mix"],
                "err_Delta": row["Delta"] - p["Delta"],
                "err_I_sm": row["I_sm_calc"] - p["I_sm_calc"],
            })
        rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_csv(jc.TABLES / "step02_table1_reproduction.csv", index=False)

    tgt = t1[t1["is_target"]]
    print("\n  Table 1 reproduction (paper value in brackets):")
    print("  %-7s %16s %16s %18s" % ("sample", "d_mix", "Delta", "I_sm calculated"))
    for _, r in tgt.iterrows():
        print("  %-7s %8.4f [%.3f] %8.4f [%.3f] %9.2f [%6.2f]"
              % (r["sample"], r["d_mix"], r["paper_d_mix"], r["Delta"], r["paper_Delta"],
                 r["I_sm_calc"], r["paper_I_sm_calc"]))

    # ---- which Eq. (3) coefficient reproduces Table 1? --------------------
    coef_rows = []
    for label, coef in (("Eq3_0.6568_x100", COEF_EQ3), ("Fig2_annotation", COEF_FIG2)):
        err = [abs(ism_from_delta(r["Delta"], coef) - r["paper_I_sm_calc"]) for _, r in tgt.iterrows()]
        coef_rows.append({"source": label, "coefficient": coef,
                          "max_abs_err_I_sm": float(np.max(err)),
                          "mean_abs_err_I_sm": float(np.mean(err))})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(jc.TABLES / "step02_eq3_coefficient.csv", index=False)
    best = coef_df.loc[coef_df["mean_abs_err_I_sm"].idxmin()]
    print("\n  Eq. (3) coefficient test -- mean |err| in I_sm vs Table 1:")
    for _, r in coef_df.iterrows():
        print("    C = %.2f  (%-16s)  mean %.3f  max %.3f"
              % (r["coefficient"], r["source"], r["mean_abs_err_I_sm"], r["max_abs_err_I_sm"]))
    print("    -> adopting C = %.2f" % best["coefficient"])

    # ---- hull-window sensitivity ----------------------------------------
    sens = []
    for lo, hi in [(None, None), (0.40, 2.60), (0.50, 2.60), (0.55, 2.50), (0.70, 2.50)]:
        dw = band_depths(ssa_by_sample, lo, hi)
        sens.append({
            "window": "native" if lo is None else "%.2f-%.2f" % (lo, hi),
            "d_dc": float(np.mean([dw[s] for s in FRESH])),
            "d_sw": float(np.mean([dw[s] for s in SW])),
            **{"d_" + s: dw[s] for s in PAPER_TABLE1.index},
        })
    pd.DataFrame(sens).to_csv(jc.TABLES / "step02_hull_window_sensitivity.csv", index=False)
    print("\n  hull-window sensitivity -> tables/step02_hull_window_sensitivity.csv")

    # ---- negative control: a featureless spectrum must not look immature --
    wl_flat = np.linspace(0.35, 2.55, 441)
    ssa_flat = np.full_like(wl_flat, 0.55)
    d_flat = jc.band_depth_at(wl_flat, jc.convex_hull_removal(wl_flat, ssa_flat), jc.BAND_REF_UM)
    delta_flat = (d_dc - d_flat) / (d_dc - d_sw)
    ism_flat = ism_from_delta(delta_flat, best["coefficient"])
    nc_ok = abs(d_flat) < 1e-9 and ism_flat > 90.0
    print("  negative control (no 970 nm band): d = %.2e -> Delta = %.3f -> I_sm = %.1f  [%s]"
          % (d_flat, delta_flat, ism_flat, "PASS" if nc_ok else "FAIL"))

    # ---- depth-corrected SSA for the four target soils -------------------
    # omega_sw = mean SSA of the three most space-weathered soils, on a common grid.
    grid = np.arange(0.40, 2.5001, 0.005)
    sw_stack = np.vstack([np.interp(grid, ssa_by_sample[s][0], ssa_by_sample[s][1],
                                    left=np.nan, right=np.nan) for s in SW])
    omega_sw = np.nanmean(sw_stack, axis=0)

    out = {"wavelength_um": grid, "omega_sw": omega_sw}
    for _, r in tgt.iterrows():
        s = r["sample"]
        wl_s, ssa_s, _ = ssa_by_sample[s]
        omega_mix = np.interp(grid, wl_s, ssa_s, left=np.nan, right=np.nan)
        theta_sw = float(r["I_sm_calc"]) / 100.0
        omega_dc = (omega_mix - omega_sw * theta_sw) / (1.0 - theta_sw)   # Eq. (2)
        out["omega_mix_" + s] = omega_mix
        out["omega_dc_" + s] = omega_dc
    dc = pd.DataFrame(out)
    dc.to_csv(jc.TABLES / "step02_depth_corrected_ssa.csv", index=False)
    print("\n  depth-corrected SSA -> tables/step02_depth_corrected_ssa.csv")

    # ---- gate ------------------------------------------------------------
    checks = [
        ("d_dc vs paper 0.358", abs(d_dc - PAPER_D_DC), 5e-4),
        ("d_sw vs paper 0.120", abs(d_sw - PAPER_D_SW), 5e-4),
        ("max |err| d_mix (4 soils)", float(tgt["err_d_mix"].abs().max()), 5e-4),
        ("max |err| Delta (4 soils)", float(tgt["err_Delta"].abs().max()), 2e-3),
        ("max |err| I_sm (4 soils)", float(tgt["err_I_sm"].abs().max()), 0.30),
    ]
    print("\n  gate:")
    n_fail = 0
    gate_rows = []
    for name, val, tol in checks:
        ok = val <= tol
        n_fail += (not ok)
        gate_rows.append({"check": name, "value": val, "tol": tol,
                          "status": "PASS" if ok else "FAIL"})
        print("    %-30s %-6s %.5f  (tol %.5f)" % (name, "PASS" if ok else "FAIL", val, tol))
    if not nc_ok:
        n_fail += 1
        gate_rows.append({"check": "negative control", "value": ism_flat, "tol": 90.0,
                          "status": "FAIL"})
    else:
        gate_rows.append({"check": "negative control", "value": ism_flat, "tol": 90.0,
                          "status": "PASS"})
    pd.DataFrame(gate_rows).to_csv(jc.TABLES / "step02_gate.csv", index=False)

    jc.verify_manifest(STAGE)
    print("\n  %s" % ("STEP 02 PASS" if n_fail == 0 else "STEP 02 FAIL (%d checks)" % n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
