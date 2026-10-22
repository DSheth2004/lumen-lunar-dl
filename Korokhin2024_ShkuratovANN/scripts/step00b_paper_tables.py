"""
Step 00b -- the paper's tables, transcribed, and checked against themselves.

The transcription below is the single place the published numbers enter this
reproduction (paper_targets.csv quotes from it). Every value was read from the
published PDF; the rows flagged by the checks were re-read from the
page image a second time before being accepted as printed.

Internal-consistency checks, none of which need our code or data:
  1. Sigma c (the printed total) = sum of the printed concentrations, incl. gray/AGG.
  2. The printed Mean / StdDev rows of Tables 5 and 6 (StdDev with ddof = 1).
  3. sigma_Rel = 100 sigma / <A> implies each test site's mean albedo <A>; the
     site is the same pixel in Tables 8, 9, 13, 14, 15, so <A> must agree.
  4. Table 12: t_Optim = optimal epoch x t_Epoch.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd

import korokhin_common as kc

STAGE = "step00b"

CONC = ["Gr", "CPX", "OPX", "OLV", "PLG", "ILM"]

T01 = """mixture,components,source_pct,grain_range_um,S_for_A1_um,restored_pct,restored_S_um,sigma_rel_pct
1,OLV+PYX,50:50,0-75,38,56:44,32.9,1.6
2,OLV+PYX+ANO,33:33:33,45-75,45,39:34:27,43.3,1.1
3,OLV+PYX+MET,40:30:30,45-90,45,40:23:37,39.4,1.0
"""

T05 = """A_Gr,tau_Gr,Gr,S,nFe,CPX,OPX,OLV,PLG,ILM,sum_c,sigma_rel
0.05,1.304,129.12,137.30,0.086,16.36,6.03,6.44,23.89,6.70,188.54,0.771
0.10,0.441,74.06,137.20,0.086,16.91,6.22,6.66,24.57,6.93,135.35,0.772
0.20,0.180,45.01,137.60,0.086,17.64,6.51,6.97,25.61,7.24,108.98,0.773
0.30,0.095,35.11,137.70,0.086,17.98,6.63,7.11,26.07,7.38,100.28,0.774
0.40,0.053,30.33,137.30,0.086,18.18,6.68,7.20,26.24,7.48,96.10,0.774
0.50,0.030,27.68,138.30,0.086,18.23,6.77,7.22,26.54,7.47,93.91,0.774
"""
T05_STATS = {"Mean": {"S": 137.57, "nFe": 0.086, "CPX": 17.55, "OPX": 6.47, "OLV": 6.93,
                      "PLG": 25.49, "ILM": 7.20, "sigma_rel": 0.773},
             "StdDev": {"S": 0.41, "nFe": 0.000, "CPX": 0.76, "OPX": 0.29, "OLV": 0.32,
                        "PLG": 1.04, "ILM": 0.32, "sigma_rel": 0.001}}

T06 = """A_Gr,tau_Gr,Gr,S,nFe,CPX,OPX,OLV,PLG,ILM,sum_c,sigma_rel
0.05,1.304,90.61,41.80,0.071,10.35,1.82,0.00,48.80,11.11,162.69,0.636
0.10,0.441,51.31,39.50,0.071,11.07,1.52,0.00,48.23,11.33,123.46,0.636
0.20,0.180,30.39,43.10,0.071,10.29,2.03,0.00,50.63,11.35,104.70,0.636
0.30,0.095,24.31,42.00,0.073,10.54,2.04,0.00,50.34,11.48,98.71,0.636
0.40,0.053,20.22,41.60,0.071,10.65,2.03,0.15,50.96,11.62,95.61,0.640
0.50,0.030,16.67,43.10,0.071,10.90,1.81,0.00,51.45,11.68,94.19,0.637
"""
T06_STATS = {"Mean": {"S": 41.85, "nFe": 0.071, "CPX": 10.63, "OPX": 1.88, "OLV": 0.03,
                      "PLG": 50.07, "ILM": 11.43, "sigma_rel": 0.637},
             "StdDev": {"S": 1.32, "nFe": 0.001, "CPX": 0.31, "OPX": 0.20, "OLV": 0.06,
                        "PLG": 1.27, "ILM": 0.21, "sigma_rel": 0.001}}

T07 = """sample,S,nFe,Gr,CPX,OPX,OLV,PLG,ILM,sum_c,sigma,sigma_rel
LS-CMP-045,57.1,0.052,17.29,7.91,0.00,3.54,6.09,3.19,38.02,0.00088,1.55
"""

# Table 8: the agglutinate LS-CMP-045 replaces the gray component (column "Gr" = c_AGG)
T08 = """site,S,nFe,Gr,CPX,OPX,OLV,PLG,ILM,sum_c,sigma,sigma_rel
1,87.7,0.033,65.53,8.53,8.56,3.55,40.50,0.00,118.11,0.00186,0.93
2,49.9,0.139,109.69,0.00,4.47,0.44,30.86,2.04,147.50,0.00144,0.83
3,0.1,0.035,26.29,26.29,0.00,17.44,36.31,4.37,110.70,0.00133,0.37
4,11.9,0.045,73.98,0.00,2.63,0.54,53.72,4.62,135.49,0.00143,0.54
"""

T09 = """site,S,nFe,Gr,CPX,OPX,OLV,PLG,ILM,sum_c,sigma,sigma_rel
1,138.7,0.087,54.39,17.65,6.54,6.98,25.78,7.00,111.80,0.00155,0.77
2,88.4,0.139,43.53,18.88,0.01,0.00,30.61,13.24,106.27,0.00122,0.70
3,8.0,0.041,16.42,11.41,0.00,0.59,68.21,6.59,103.22,0.00154,0.43
4,38.1,0.071,37.89,11.53,1.48,0.00,48.40,11.57,110.87,0.00168,0.64
"""

T10 = """initial_PLG,S,nFe,Gr,CPX,OPX,OLV,PLG,ILM,sigma,sigma_rel
0,158.7,0.083,50.69,12.93,9.91,6.62,32.37,4.63,0.00161,0.80
10,137.4,0.086,54.35,17.71,6.48,7.00,25.60,7.06,0.00155,0.77
20,138.7,0.087,54.39,17.65,6.54,6.98,25.78,7.00,0.00155,0.77
50,116.9,0.084,54.74,19.00,5.58,7.52,21.93,7.88,0.00158,0.79
"""

T12 = """iteration,N1,N2,initial_rate,adaptive_k,pattern_size,sigma_rel,optimal_epoch,t_epoch_s,t_optim_min,t_map_s
1,42,12,1.000,0.01,20000,2.39,66,0.24,0.25,0.2
2,42,12,0.002,,152000,1.26,5336,0.88,78.26,0.2
3,42,12,0.010,,152000,1.12,4256,0.88,62.42,0.2
"""

T13 = """site,S,nFe,Gr,CPX,OPX,OLV,PLG,ILM,sum_c,sigma,sigma_rel
1,67.9,0.041,31.19,33.02,1.65,1.64,19.18,15.22,100.25,0.00812,4.05
2,45.6,0.103,32.31,25.46,1.43,1.55,20.96,16.29,98.00,0.00368,2.13
3,4.0,0.042,31.78,36.90,1.39,1.24,30.18,8.46,109.95,0.01315,3.67
4,17.7,0.055,31.76,32.41,1.46,1.38,26.78,12.98,106.97,0.00704,2.68
"""

T14 = """site,S,nFe,Gr,CPX,OPX,OLV,PLG,ILM,sum_c,sigma,sigma_rel
1,57.0,0.043,31.31,32.30,1.60,1.59,20.51,13.54,99.25,0.00439,2.19
2,35.1,0.096,32.30,23.86,1.38,1.48,22.15,15.78,96.95,0.00377,2.18
3,1.0,0.048,31.86,37.66,1.37,1.21,30.70,7.33,110.13,0.00496,1.38
4,9.7,0.063,31.99,29.72,1.39,1.31,28.23,13.98,106.62,0.00243,0.92
"""

T15 = """site,S,nFe,Gr,CPX,OPX,OLV,PLG,ILM,sum_c,sigma,sigma_rel
1,55.4,0.041,31.29,31.710,1.60,1.58,20.74,13.76,99.08,0.00434,2.17
2,35.6,0.095,32.31,23.29,1.37,1.48,21.75,15.97,96.17,0.00202,1.17
3,1.4,0.046,31.78,38.29,1.39,1.22,30.63,6.82,110.13,0.00224,0.63
4,10.6,0.063,31.97,29.85,1.40,1.32,28.07,13.46,106.07,0.00241,0.92
"""

TABLES = {"t01_lab_mixtures": T01, "t05_gray_albedo_site1": T05, "t06_gray_albedo_site4": T06,
          "t07_agglutinate_fit": T07, "t08_m3_sites_agglutinate": T08, "t09_m3_sites_nm": T09,
          "t10_site1_initial_plg": T10, "t12_ann_iterations": T12, "t13_m3_sites_ann_iter1": T13,
          "t14_m3_sites_ann_iter2": T14, "t15_m3_sites_ann_iter3": T15}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(kc.PAPER_TABLES / ("%s.csv" % name))


def main() -> int:
    kc.banner(STAGE, "transcribe the paper's tables and check their arithmetic")
    for name, txt in TABLES.items():
        pd.read_csv(io.StringIO(txt)).to_csv(kc.PAPER_TABLES / ("%s.csv" % name), index=False)
    rng = pd.DataFrame({"param": kc.PARAMS, "min": [kc.TABLE11_MIN[p] for p in kc.PARAMS],
                        "max": [kc.TABLE11_MAX[p] for p in kc.PARAMS]})
    rng.to_csv(kc.PAPER_TABLES / "t11_training_ranges.csv", index=False)

    checks = []

    def add(check, where, printed, recomputed, tol, note=""):
        diff = recomputed - printed
        checks.append({"check": check, "where": where, "printed": printed,
                       "recomputed": round(recomputed, 4), "diff": round(diff, 4),
                       "tolerance": tol, "consistent": abs(diff) <= tol, "note": note})

    # 1. sum of concentrations
    for name in ("t05_gray_albedo_site1", "t06_gray_albedo_site4", "t07_agglutinate_fit",
                 "t08_m3_sites_agglutinate", "t09_m3_sites_nm", "t13_m3_sites_ann_iter1",
                 "t14_m3_sites_ann_iter2", "t15_m3_sites_ann_iter3"):
        t = load(name)
        key = t.columns[0]
        for _, r in t.iterrows():
            s = float(sum(r[c] for c in CONC))
            diff = s - r["sum_c"]
            note = ""
            if abs(diff) > 0.03 and abs(abs(diff) - r["OPX"]) < 0.015:
                note = "difference equals c_OPX: the printed total omits the OPX term"
            add("sum_c", "%s %s=%s" % (name, key, r[key]), r["sum_c"], s, 0.03, note)

    # 2. Mean / StdDev rows of Tables 5 and 6
    for name, stats in (("t05_gray_albedo_site1", T05_STATS), ("t06_gray_albedo_site4", T06_STATS)):
        t = load(name)
        for col, v in stats["Mean"].items():
            add("mean", "%s %s" % (name, col), v, float(t[col].mean()),
                0.006 if col != "nFe" else 0.0006)
        for col, v in stats["StdDev"].items():
            add("stddev(ddof=1)", "%s %s" % (name, col), v, float(t[col].std(ddof=1)),
                0.006 if col != "nFe" else 0.0006)

    # 3. implied mean albedo per site, across the five site tables
    implied = []
    for name in ("t08_m3_sites_agglutinate", "t09_m3_sites_nm", "t13_m3_sites_ann_iter1",
                 "t14_m3_sites_ann_iter2", "t15_m3_sites_ann_iter3"):
        t = load(name)
        for _, r in t.iterrows():
            implied.append({"table": name, "site": int(r["site"]),
                            "mean_albedo": 100.0 * r["sigma"] / r["sigma_rel"]})
    imp = pd.DataFrame(implied)
    site_a = imp.groupby("site")["mean_albedo"].agg(["mean", "std", "min", "max"]).reset_index()
    site_a.to_csv(kc.PAPER_TABLES / "derived_site_mean_albedo.csv", index=False)
    for _, r in site_a.iterrows():
        # sigma is printed to 3 significant figures, sigma_rel to 2 decimals: spread
        # of a few thousandths is rounding
        add("implied <A> agrees across tables", "site %d" % r["site"], float(r["mean"]),
            float(r["mean"]) + float(r["max"] - r["min"]), 0.01,
            "range %.4f-%.4f over 5 tables" % (r["min"], r["max"]))
    a7 = load("t07_agglutinate_fit").iloc[0]
    (kc.PAPER_TABLES / "derived_agglutinate_mean_albedo.txt").write_text(
        "%.5f\n" % (100.0 * a7["sigma"] / a7["sigma_rel"]))

    # 4. Table 12 timing arithmetic
    for _, r in load("t12_ann_iterations").iterrows():
        add("t_optim = epoch x t_epoch", "iteration %d" % r["iteration"], r["t_optim_min"],
            r["optimal_epoch"] * r["t_epoch_s"] / 60.0, 0.02 * r["t_optim_min"] + 0.01,
            "t_epoch printed to 2 decimals")

    df = pd.DataFrame(checks)
    df.to_csv(kc.TABLES / "step00b_consistency.csv", index=False)
    bad = df[~df["consistent"]]
    print("  %d arithmetic checks on the printed tables: %d consistent, %d not"
          % (len(df), int(df["consistent"].sum()), len(bad)))
    for _, r in bad.iterrows():
        print("   INCONSISTENT %-18s %-40s printed %9.3f  recomputed %9.3f  %s"
              % (r["check"], r["where"], r["printed"], r["recomputed"], r["note"]))
    print("  implied site mean albedo <A> (RELAB system):")
    for _, r in site_a.iterrows():
        print("    site %d: %.4f  (range %.4f-%.4f)" % (r["site"], r["mean"], r["min"], r["max"]))
    print("  implied <A> of agglutinate LS-CMP-045 (Table 7): %.5f"
          % (100.0 * a7["sigma"] / a7["sigma_rel"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
