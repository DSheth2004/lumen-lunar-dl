"""
LSCC highland-soil ground truth, transcribed from Taylor et al. (2010), "Mineralogical and
chemical characterization of lunar highland soils: insights into the space weathering of
soils on airless bodies", J. Geophys. Res. 115, E02002 -- NASA NTRS preprint 20100006633
(revised 9 August 2009; public domain).

  Table 2  modal abundance of minerals and glasses (vol. %), 20-45 / 10-20 / <10 um
  Table 3  the pyroxene sub-sets (orthopyroxene, pigeonite, Mg-cpx, Fe-pyroxene)
  Table 4  bulk chemistry (wt. %) and Is/FeO, <45 / 20-45 / 10-20 / <10 um

The mare-soil tables (Taylor et al. 2001) have no highland counterpart; these ten soils
are the Apollo 14 and 16 soils whose RELAB spectra (Kdata) are already staged. The
preprint may differ from the published version in detail; no published copy is open.

check() verifies the transcription against the tables' own arithmetic:
  * every Table 2 column sums to its printed 100.0 (+-0.15, one-decimal rounding);
  * every Table 3 column's sub-sets sum to its printed total (+-0.03);
  * every Table 4 column's listed oxides fall short of its printed total by -0.05 ... +0.20.
    The first tolerance tried (+-0.06) failed on 17 of 40 columns, all but three of the
    40 residuals being positive (mean +0.056, max +0.19): the printed totals include
    something the table does not list. A transcription slip would give a residual of
    either sign; the oxides actually used (FeO, Al2O3, CaO, MgO) are verified
    independently below.
independent_check() compares FeO, Al2O3, CaO and MgO POSITIONALLY with Shkuratov et al.
(2005a) Table 1, which prints the same LSCC analyses for 13 highland soil fractions, with
a shuffled negative control.
pyroxene_crosscheck() compares Table 3 totals with Table 2 pyroxene the same way.
Disagreements that are the source's own (Table 2 vs Table 3 pyroxene) are listed, not
"fixed".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE = "Taylor et al. (2010) JGR 115 E02002, NTRS preprint 20100006633"
SOILS = ["14141", "14163", "14260", "14259", "61221", "67461", "67481", "61141", "64801", "62231"]
MODAL_FRACS = ["20-45", "10-20", "<10"]
CHEM_FRACS = ["<45", "20-45", "10-20", "<10"]

# Table 2 -- per soil, rows in this order, columns 20-45 / 10-20 / <10
T2_ROWS = ["agglutinitic_glass", "pyroxene", "plagioclase", "olivine", "ilmenite", "k_phases", "others"]
T2 = {
    "14141": [(41.0, 48.6, 45.9), (19.8, 10.9, 10.3), (26.6, 28.0, 27.6), (4.0, 1.6, 1.5),
              (1.9, 1.1, 1.7), (4.5, 7.4, 10.8), (2.2, 2.4, 2.2)],
    "14163": [(56.4, 58.5, 66.3), (16.2, 13.8, 3.8), (18.9, 18.3, 21.8), (2.4, 2.1, 0.4),
              (0.8, 0.9, 1.1), (3.8, 4.1, 3.4), (1.5, 2.3, 3.2)],
    "14260": [(64.0, 65.2, 66.5), (13.7, 12.1, 7.7), (15.6, 16.1, 16.3), (2.1, 1.5, 1.4),
              (0.9, 1.0, 1.3), (2.5, 2.6, 3.7), (1.2, 1.5, 3.1)],
    "14259": [(60.5, 68.7, 71.6), (18.2, 9.1, 5.9), (14.1, 15.4, 14.6), (2.3, 1.4, 0.7),
              (1.3, 1.2, 1.5), (2.5, 2.7, 3.4), (1.1, 1.5, 2.3)],
    "61221": [(28.9, 32.6, 41.6), (7.4, 5.3, 1.5), (58.7, 59.4, 54.4), (3.9, 2.0, 0.9),
              (0.6, 0.3, 0.9), (0.2, 0.2, 0.3), (0.3, 0.2, 0.4)],
    "67461": [(25.4, 32.4, 35.8), (7.3, 4.1, 2.8), (64.3, 61.0, 60.0), (2.5, 1.5, 0.7),
              (0.3, 0.3, 0.2), (0.1, 0.2, 0.1), (0.1, 0.5, 0.4)],
    "67481": [(27.6, 28.6, 35.2), (6.6, 5.6, 3.6), (61.2, 62.0, 58.8), (4.0, 2.9, 1.5),
              (0.1, 0.2, 0.2), (0.3, 0.4, 0.3), (0.2, 0.3, 0.4)],
    "61141": [(50.1, 53.9, 61.6), (4.4, 3.3, 1.7), (42.5, 40.3, 35.3), (2.1, 1.6, 0.5),
              (0.3, 0.3, 0.3), (0.3, 0.4, 0.3), (0.3, 0.2, 0.3)],
    "64801": [(53.6, 61.0, 63.6), (4.5, 2.8, 2.7), (39.3, 34.5, 32.3), (1.8, 1.0, 0.6),
              (0.3, 0.2, 0.2), (0.3, 0.3, 0.4), (0.2, 0.2, 0.2)],
    "62231": [(50.6, 55.0, 69.5), (5.1, 4.40, 0.9), (40.5, 37.8, 28.3), (2.9, 1.7, 0.3),
              (0.3, 0.5, 0.4), (0.3, 0.4, 0.3), (0.3, 0.2, 0.3)],
}

# Table 3 -- orthopyroxene, pigeonite, Mg-clinopyroxene, Fe-pyroxene, printed total
T3_ROWS = ["orthopyroxene", "pigeonite", "mg_clinopyroxene", "fe_clinopyroxene", "total_pyroxene_printed"]
T3 = {
    "14141": [(7.57, 4.07, 3.37), (8.08, 4.58, 4.29), (3.08, 1.85, 2.19), (1.08, 0.38, 0.44), (19.81, 10.88, 10.29)],
    "14163": [(6.50, 5.68, 1.51), (5.66, 4.94, 1.38), (3.10, 2.41, 0.91), (0.92, 0.78, 0.14), (16.18, 13.81, 3.94)],
    "14260": [(4.68, 5.14, 2.58), (4.99, 4.23, 3.15), (3.07, 2.00, 1.51), (0.94, 0.64, 0.48), (13.68, 12.01, 7.72)],
    "14259": [(7.40, 3.72, 1.92), (6.14, 3.18, 1.96), (3.04, 1.66, 1.77), (1.57, 0.50, 0.29), (18.15, 9.06, 5.94)],
    "61221": [(2.96, 1.82, 0.56), (2.24, 1.43, 0.55), (1.98, 1.95, 0.37), (0.19, 0.14, 0.02), (7.37, 5.34, 1.50)],
    "67461": [(2.96, 1.47, 1.09), (1.61, 1.07, 0.64), (2.53, 1.52, 1.06), (0.18, 0.05, 0.04), (7.28, 4.11, 2.83)],
    "67481": [(2.95, 2.55, 1.38), (1.54, 1.27, 1.05), (1.94, 1.73, 1.41), (0.17, 0.13, 0.05), (6.60, 5.68, 3.89)],
    "61141": [(1.68, 1.69, 0.22), (1.38, 2.15, 0.23), (1.11, 1.45, 0.19), (0.18, 0.04, 0.06), (4.35, 5.33, 0.70)],
    "64801": [(2.03, 1.24, 1.18), (1.15, 0.96, 0.84), (1.33, 0.60, 0.64), (0.01, 0.01, 0.00), (4.52, 2.81, 2.66)],
    "62231": [(2.08, 1.99, 0.28), (1.33, 1.55, 0.27), (1.52, 1.74, 0.30), (0.19, 0.12, 0.03), (5.12, 5.40, 0.88)],
}

# Table 4 -- oxides (wt. %), printed total, Is/FeO; columns <45 / 20-45 / 10-20 / <10
T4_ROWS = ["SiO2", "TiO2", "Al2O3", "Cr2O3", "MgO", "CaO", "MnO", "FeO", "Na2O", "K2O", "P2O5", "SO3",
           "total_printed", "is_feo"]
T4 = {
    "14141": [(47.9, 47.2, 48.4, 49.2), (1.65, 1.96, 1.71, 1.51), (17.0, 15.0, 17.2, 19.2),
              (0.22, 0.26, 0.23, 0.21), (9.28, 11.0, 9.08, 6.99), (10.7, 10.1, 10.7, 11.3),
              (0.14, 0.15, 0.13, 0.10), (9.81, 11.6, 9.46, 7.66), (0.76, 0.59, 0.71, 0.91),
              (0.70, 0.47, 0.66, 0.96), (0.50, 0.26, 0.32, 0.40), (0.10, 0.07, 0.07, 0.10),
              (98.82, 98.78, 98.68, 98.61), (9.7, 5.8, 11.6, 14.5)],
    "14163": [(47.4, 47.1, 47.4, 47.2), (1.90, 2.00, 1.88, 2.07), (17.1, 15.4, 17.0, 18.9),
              (0.20, 0.23, 0.22, 0.21), (9.49, 11.0, 9.57, 8.14), (10.9, 10.2, 10.8, 11.6),
              (0.15, 0.15, 0.13, 0.12), (9.94, 11.5, 10.1, 8.83), (0.65, 0.57, 0.67, 0.70),
              (0.51, 0.41, 0.51, 0.55), (0.35, 0.21, 0.27, 0.33), (0.10, 0.08, 0.10, 0.11),
              (98.69, 99.02, 98.72, 98.85), (66.5, 43.2, 64.8, 87.0)],
    "14260": [(47.6, 47.4, 47.5, 47.8), (1.85, 1.86, 1.98, 1.94), (17.3, 16.3, 17.3, 19.1),
              (0.21, 0.22, 0.23, 0.20), (9.46, 10.4, 9.53, 8.21), (11.0, 10.7, 11.0, 11.8),
              (0.15, 0.14, 0.13, 0.12), (9.65, 10.7, 9.84, 8.10), (0.61, 0.60, 0.60, 0.57),
              (0.49, 0.44, 0.46, 0.47), (0.32, 0.22, 0.21, 0.17), (0.12, 0.10, 0.10, 0.09),
              (98.81, 99.13, 98.97, 98.76), (93.3, 80.2, 98.9, 144.9)],
    "14259": [(47.7, 47.1, 47.5, 47.9), (1.80, 1.99, 1.96, 2.02), (17.4, 15.8, 17.4, 19.3),
              (0.20, 0.24, 0.23, 0.20), (9.47, 10.7, 9.44, 8.09), (11.1, 10.5, 11.0, 11.9),
              (0.13, 0.15, 0.13, 0.12), (9.54, 11.0, 9.71, 7.82), (0.62, 0.60, 0.63, 0.63),
              (0.47, 0.43, 0.47, 0.50), (0.30, 0.26, 0.23, 0.23), (0.11, 0.09, 0.12, 0.10),
              (98.80, 99.02, 98.87, 98.84), (108.6, 77.2, 101.8, 174.8)],
    "61221": [(44.7, 44.5, 44.5, 44.5), (0.52, 0.56, 0.54, 0.50), (27.3, 27.2, 27.5, 28.5),
              (0.09, 0.09, 0.09, 0.08), (5.29, 5.45, 5.16, 4.35), (15.9, 15.9, 16.0, 16.5),
              (0.08, 0.06, 0.05, 0.06), (4.47, 4.62, 4.40, 3.64), (0.48, 0.46, 0.45, 0.53),
              (0.09, 0.07, 0.09, 0.13), (0.06, 0.05, 0.05, 0.06), (0.07, 0.04, 0.06, 0.10),
              (99.13, 99.07, 98.93, 99.00), (13.6, 8.4, 13.89, 19.8)],
    "67461": [(44.6, 44.4, 44.1, 44.5), (0.35, 0.44, 0.39, 0.34), (28.4, 27.3, 27.8, 29.4),
              (0.08, 0.09, 0.08, 0.08), (4.46, 5.11, 4.80, 3.83), (16.5, 16.1, 16.5, 17.1),
              (0.06, 0.07, 0.08, 0.06), (4.24, 4.93, 4.64, 3.35), (0.40, 0.41, 0.39, 0.43),
              (0.06, 0.05, 0.05, 0.07), (0.04, 0.03, 0.04, 0.03), (0.06, 0.07, 0.04, 0.07),
              (99.26, 99.00, 98.90, 99.31), (29.8, 22.3, 23.9, 35.2)],
    "67481": [(44.6, 44.7, 44.4, 44.5), (0.44, 0.49, 0.40, 0.42), (28.1, 26.7, 28.4, 29.1),
              (0.10, 0.09, 0.09, 0.08), (4.91, 5.98, 4.54, 4.09), (16.2, 15.6, 16.4, 16.7),
              (0.06, 0.08, 0.05, 0.07), (4.38, 5.19, 4.04, 3.61), (0.43, 0.45, 0.45, 0.46),
              (0.06, 0.06, 0.07, 0.08), (0.04, 0.05, 0.04, 0.04), (0.04, 0.04, 0.06, 0.07),
              (99.39, 99.50, 99.08, 99.22), (33.5, 20.7, 33.0, 38.5)],
    "61141": [(45.0, 44.5, 44.6, 44.9), (0.59, 0.58, 0.64, 0.59), (26.3, 26.1, 25.6, 27.4),
              (0.12, 0.11, 0.13, 0.11), (6.39, 6.56, 6.84, 5.53), (15.3, 15.2, 15.2, 16.0),
              (0.07, 0.08, 0.08, 0.07), (4.80, 5.15, 5.14, 3.66), (0.43, 0.46, 0.41, 0.48),
              (0.11, 0.10, 0.10, 0.14), (0.06, 0.06, 0.05, 0.06), (0.09, 0.05, 0.08, 0.11),
              (99.34, 99.00, 98.91, 99.11), (94.5, 75.5, 81.6, 119.3)],
    "64801": [(45.0, 44.6, 44.5, 44.8), (0.65, 0.63, 0.68, 0.61), (26.9, 26.5, 26.3, 27.7),
              (0.10, 0.10, 0.12, 0.12), (5.83, 6.09, 6.18, 5.22), (15.6, 15.6, 15.6, 16.1),
              (0.06, 0.08, 0.08, 0.07), (4.68, 4.82, 4.78, 3.84), (0.43, 0.44, 0.41, 0.42),
              (0.12, 0.12, 0.11, 0.14), (0.07, 0.06, 0.06, 0.04), (0.09, 0.10, 0.07, 0.11),
              (99.50, 99.20, 98.99, 99.21), (92.2, 83.4, 84.9, 115.2)],
    "62231": [(45.0, 44.5, 44.7, 45.0), (0.60, 0.58, 0.61, 0.58), (26.3, 25.7, 26.3, 27.4),
              (0.11, 0.11, 0.13, 0.13), (6.20, 6.59, 6.38, 5.49), (15.4, 15.3, 15.5, 16.1),
              (0.09, 0.09, 0.07, 0.07), (4.87, 5.31, 4.86, 3.63), (0.43, 0.42, 0.41, 0.46),
              (0.12, 0.09, 0.10, 0.14), (0.07, 0.07, 0.05, 0.04), (0.09, 0.08, 0.08, 0.13),
              (99.32, 98.87, 99.22, 99.22), (116.7, 80.7, 109.9, 169.0)],
}


def modal_long() -> pd.DataFrame:
    rows = []
    for soil in SOILS:
        for j, frac in enumerate(MODAL_FRACS):
            for name, vals in zip(T2_ROWS, T2[soil]):
                rows.append((soil, frac, "Table 2", name, vals[j]))
            for name, vals in zip(T3_ROWS, T3[soil]):
                rows.append((soil, frac, "Table 3", name, vals[j]))
    return pd.DataFrame(rows, columns=["sample", "size_fraction_um", "source_table", "phase", "modal_pct"])


def chem_wide() -> pd.DataFrame:
    rows = []
    for soil in SOILS:
        for j, frac in enumerate(CHEM_FRACS):
            row = {"sample": soil, "size_fraction_um": frac}
            row.update({k: v[j] for k, v in zip(T4_ROWS, T4[soil])})
            rows.append(row)
    return pd.DataFrame(rows)


def check() -> pd.DataFrame:
    """Returns one row per check; raises if the transcription contradicts itself."""
    out = []
    for soil in SOILS:
        for j, frac in enumerate(MODAL_FRACS):
            s2 = sum(v[j] for v in T2[soil])
            out.append({"check": "Table 2 column sum = 100", "sample": soil, "fraction": frac,
                        "value": s2, "expected": 100.0, "ok": abs(s2 - 100.0) <= 0.15})
            s3 = sum(v[j] for v in T3[soil][:4])
            out.append({"check": "Table 3 sub-sets = printed total", "sample": soil, "fraction": frac,
                        "value": s3, "expected": T3[soil][4][j], "ok": abs(s3 - T3[soil][4][j]) <= 0.03})
        for j, frac in enumerate(CHEM_FRACS):
            s4 = sum(v[j] for v in T4[soil][:12])
            short = T4[soil][12][j] - s4
            out.append({"check": "Table 4 printed total - listed oxides in [-0.05, 0.20]", "sample": soil,
                        "fraction": frac, "value": s4, "expected": T4[soil][12][j],
                        "ok": -0.05 <= short <= 0.20})
    df = pd.DataFrame(out)
    bad = df[~df["ok"]]
    if len(bad):
        raise ValueError("Taylor 2010 transcription fails its own arithmetic:\n" + bad.to_string())
    return df


# Shkuratov, Kaydash & Pieters (2005a), Sol. Syst. Res. 39:255, Table 1 -- Al2O3, CaO, FeO, MgO
# of the LSCC highland fractions it lists.
# Its rows carry no size label; the fraction is the one whose Taylor et al. values they are,
# which is what the positional check below tests. 61220 in that table is a different soil.
SHK2005 = [("14141", "20-45", (15.00, 10.10, 11.60, 11.00)), ("14141", "10-20", (17.20, 10.70, 9.46, 9.08)),
           ("14163", "20-45", (15.40, 10.20, 11.50, 11.00)), ("14163", "10-20", (17.00, 10.80, 10.10, 9.57)),
           ("14259", "20-45", (15.80, 10.50, 11.00, 10.70)), ("14259", "10-20", (17.40, 11.00, 9.71, 9.44)),
           ("14260", "20-45", (16.30, 10.70, 10.70, 10.40)), ("14260", "10-20", (17.30, 11.00, 9.84, 9.53)),
           ("61141", "20-45", (26.10, 15.20, 5.15, 6.56)), ("62231", "20-45", (25.70, 15.30, 5.31, 6.59)),
           ("64801", "20-45", (26.50, 15.60, 4.82, 6.09)), ("67461", "20-45", (27.30, 16.10, 4.93, 5.11)),
           ("67481", "20-45", (26.70, 15.60, 5.19, 5.98))]
SHK2005_OXIDES = ["Al2O3", "CaO", "FeO", "MgO"]


def independent_check(seed: int = 0) -> tuple[pd.DataFrame, dict]:
    ours = chem_wide().set_index(["sample", "size_fraction_um"])
    rows, a, b = [], [], []
    for soil, frac, vals in SHK2005:
        mine = [float(ours.loc[(soil, frac), k]) for k in SHK2005_OXIDES]
        a.append(mine); b.append(vals)
        rows.append({"sample": soil, "fraction": frac,
                     **{"taylor_" + k: m for k, m in zip(SHK2005_OXIDES, mine)},
                     **{"shk2005_" + k: v for k, v in zip(SHK2005_OXIDES, vals)},
                     "agree": bool(np.all(np.abs(np.array(mine) - np.array(vals)) <= 0.051))})
    a, b = np.array(a), np.array(b)
    perm = np.random.default_rng(seed).permutation(len(a))
    shuffled = np.all(np.abs(a - b[perm]) <= 0.051, axis=1)
    df = pd.DataFrame(rows)
    return df, {"n": len(df), "agree_real": int(df["agree"].sum()), "agree_shuffled": int(shuffled.sum()),
                "n_moved": int((perm != np.arange(len(perm))).sum())}


def pyroxene_crosscheck(seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Table 2 pyroxene vs Table 3 total, positionally; plus a shuffled negative control."""
    keys = [(s, j) for s in SOILS for j in range(3)]
    t2 = np.array([T2[s][1][j] for s, j in keys])
    t3 = np.array([T3[s][4][j] for s, j in keys])
    agree = np.abs(t2 - t3) <= 0.15
    rows = pd.DataFrame({"sample": [k[0] for k in keys], "fraction": [MODAL_FRACS[k[1]] for k in keys],
                         "table2_pyroxene": t2, "table3_total": t3, "agree": agree})
    perm = np.random.default_rng(seed).permutation(len(keys))
    shuffled_agree = np.abs(t2 - t3[perm]) <= 0.15
    return rows, {"n": len(keys), "agree_real": int(agree.sum()), "agree_shuffled": int(shuffled_agree.sum())}
