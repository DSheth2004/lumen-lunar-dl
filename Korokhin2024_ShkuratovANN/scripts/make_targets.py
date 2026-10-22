"""
Writes paper_targets.csv -- the registry of every published number the audit scores.

Written 2026-09-19 BEFORE the M3 site fits (step06) and the ANN runs (step07) had
produced any result, so no tolerance below was chosen with those results in view.
Tables 1 and 7 and Eq. (2) had already been run; their tolerances follow the rules
fixed in the plan (Table 10 non-uniqueness for fitted concentrations), applied
literally. Re-running this script must not change a tolerance without a note in
PAPER_NOTES.md.

Amendment 1 (2026-09-19), after the Surkov thesis (the Shkuratov 2019 / Surkov 2020
recipe) and Shkuratov et al. (2005a) were obtained, and BEFORE steps 05-10 were re-run
with them: no tolerance changed; C_nFe_Is_mare, C_nFe_Is_high and C_PLG_mare became
gated (the reason they were reported-only -- unpublished MLR coefficients -- no longer
holds: Shkuratov 2005a Table 3 prints them); the Fig. 31 targets were added
(C_ILM_BD_mare, C_ILM_BD_claim_high_none, C_ILM_ANN_over_BD_mare). PAPER_NOTES D15, D20.

Amendment 2 (2026-09-21), after Pieters et al. (2006) and Korokhin et al. (2008) were
obtained and BEFORE steps 08-10 were re-run: C_nFe_Is_* now compare with Is itself
(Pieters 2006 Table 3, the Is coefficients of Eq. 3), tolerance unchanged; Fig. 30's
C_ILM_TiO2ANN_all added (gated, 0.91 +- 0.10). PAPER_NOTES D15, D22.

Tolerance classes:
  closed form     printed rounding
  lab fits        the optimizer's own demonstrated non-uniqueness (Table 10)
  M3 sites / ANN  the paper's qualitative claims are gated (1 = holds); per-site
                  numbers are reported only, because the sites and the preprocessing
                  are reconstructions (step04, D2)
"""
from __future__ import annotations

import collections
import csv

import korokhin_common as kc

R = []


def add(tid, src, q, pub, tol, basis, status, gated="yes"):
    R.append([tid, src, q, pub, tol, basis, status, gated])


HALF10 = {"S": 20.9, "nFe": 0.002, "Gr": 2.0, "CPX": 3.0, "OPX": 2.2, "OLV": 0.45, "PLG": 5.2, "ILM": 1.6}
B10 = ("half the spread of this parameter across the paper's own Table 10 (same spectrum, "
       "sigma_Rel within 0.03, different start): the optimizer's demonstrated non-uniqueness")
P = kc.PARAMS

# ---- closed form ------------------------------------------------------------
for a, t in ((0.05, 1.304), (0.10, 0.441), (0.20, 0.180), (0.30, 0.095), (0.40, 0.053), (0.50, 0.030)):
    add("T5_tau_%03d" % round(a * 100), "Table 5/6", "tau_Gr for A_Gr=%.2f (Eq. A2, n=2, q=0.5)" % a,
        t, 0.0005, "closed form; printed to 3 decimals", "step01_tau_gray.csv")

# ---- Table 1, as printed ------------------------------------------------------
T1 = {1: ([56, 44], 32.9, 1.6), 2: ([39, 34, 27], 43.3, 1.1), 3: ([40, 23, 37], 39.4, 1.0)}
NAMES = {1: ["OLV", "PYX"], 2: ["OLV", "PYX", "ANO"], 3: ["OLV", "PYX", "MET"]}
for m, (pct, S, sr) in T1.items():
    for j, v in enumerate(pct):
        add("T1_m%d_%s" % (m, NAMES[m][j]), "Table 1",
            "mixture %d restored %s proportion (%%, column order as printed)" % (m, NAMES[m][j]),
            v, 5.0, "optimizer / fit-range ambiguity; 5 points ~ the Table 10 PLG spread", "step02_table1.csv")
    add("T1_m%d_S" % m, "Table 1", "mixture %d restored grain size (um)" % m, S, round(0.1 * S, 2),
        "10 %: fit grid and range not stated (D4)", "step02_table1.csv")
    add("T1_m%d_sigrel" % m, "Table 1", "mixture %d sigma_Rel (%%)" % m, sr, 0.3,
        "printed to 0.1; fit grid/range (D4)", "step02_table1.csv")
# the same numbers read with the first two columns swapped -- evidence for F2, not a target
SW = {1: [44, 56], 2: [34, 39, 27], 3: [23, 40, 37]}
for m, pct in SW.items():
    for j, v in enumerate(pct):
        add("T1swap_m%d_%s" % (m, NAMES[m][j]), "Table 1 (OLV/PYX columns read swapped)",
            "mixture %d %s if the first two printed columns are PYX, OLV (F2)" % (m, NAMES[m][j]),
            v, 5.0, "as T1; reported only", "step02_table1.csv", "no")

# ---- Table 7 ------------------------------------------------------------------
T7 = {"S": 57.1, "nFe": 0.052, "Gr": 17.29, "CPX": 7.91, "OPX": 0.0, "OLV": 3.54, "PLG": 6.09, "ILM": 3.19}
for k, v in T7.items():
    add("T7_%s" % k, "Table 7", "agglutinate LS-CMP-045 fit: %s" % k, v, HALF10[k], B10, "step03_table7.csv")
add("T7_sigrel", "Table 7", "agglutinate LS-CMP-045 fit: sigma_Rel (%)", 1.55, 0.2,
    "fit grid/start (D4, D6)", "step03_table7.csv")

# ---- Eq. (2) ------------------------------------------------------------------
add("E2_slope", "Eq. (2)", "slope, M3 -> RELAB, Apollo 16 vs 62231", 1.685, 0.05,
    "pixel box and 62231 size fraction not stated (D7)", "step05_eq2_rederived.csv")
add("E2_intercept", "Eq. (2)", "intercept, M3 -> RELAB", 0.0355, 0.01,
    "~5 % of a mare albedo; pixel box / fraction not stated (D7)", "step05_eq2_rederived.csv")
add("E2_msd", "Eq. (2)", "relative rms of the calibration (%)", 1.4, 0.5,
    "pixel box not stated", "step05_eq2_rederived.csv")

# ---- site location check ------------------------------------------------------
for s, v in ((1, 0.2004), (2, 0.1732), (3, 0.3582), (4, 0.2632)):
    add("S_meanA_site%d" % s, "Tables 8-15 (sigma / sigma_Rel)",
        "site %d mean albedo <A> implied by the printed sigma and sigma_Rel" % s, v, 0.015,
        "~5 %; checks the recovered site pixel; refinement used Fig. 12, so not independent",
        "step05_site_check.csv", "no")

# ---- Table 9 --------------------------------------------------------------------
add("T9_claim_site1_largest_S", "Table 9 / Sec. 4.2", "site 1 (young crater) has the largest S (1 = yes)",
    1, 0, "claim", "step06_table9.csv")
add("T9_claim_olv_only_site1", "Table 9 / Sec. 4.2",
    "olivine <= 2 % at sites 2-4, 'not detected besides the young crater' (1 = yes)", 1, 0,
    "claim; 2 % is well below the olivine printed at site 1 (6.98 %)", "step06_table9.csv")
add("T9_claim_plg_highland", "Table 9 / Sec. 4.2",
    "PLG at both highland sites (3, 4) exceeds PLG at both mare sites (1 = yes)", 1, 0, "claim",
    "step06_table9.csv")
add("T9_claim_cpx_gt_opx", "Table 9 / Sec. 4.2", "CPX > 'OPX' at every site (1 = yes)", 1, 0,
    "claim: 'concentrations of clinopyroxene are found to be higher than those for orthopyroxene'",
    "step06_table9.csv")
for s, v in ((1, 0.77), (2, 0.70), (3, 0.43), (4, 0.64)):
    add("T9_sigrel_site%d" % s, "Table 9", "site %d sigma_Rel of the Nelder-Mead fit (%%)" % s, v, 0.3,
        "site pixel recovered from a figure (step04/05); preprocessing approximated (D2)", "step06_table9.csv")
T9 = {1: (138.7, 0.087, 54.39, 17.65, 6.54, 6.98, 25.78, 7.00),
      2: (88.4, 0.139, 43.53, 18.88, 0.01, 0.00, 30.61, 13.24),
      3: (8.0, 0.041, 16.42, 11.41, 0.00, 0.59, 68.21, 6.59),
      4: (38.1, 0.071, 37.89, 11.53, 1.48, 0.00, 48.40, 11.57)}
for s, vals in T9.items():
    for k, v in zip(P, vals):
        add("T9_site%d_%s" % (s, k), "Table 9", "site %d %s (Nelder-Mead)" % (s, k), v, HALF10[k],
            B10 + "; reported only (site recovered from a figure)", "step06_table9.csv", "no")

# ---- Tables 5, 6, 8, 10 -------------------------------------------------------------
add("T5_claim_weak_Agr", "Table 5 / Sec. 3.2",
    "site 1: max StdDev of a mineral concentration over A_Gr 0.05-0.50 (%)", 1.04, 1.0,
    "'depend weakly on A_Gr': the printed maximum (PLG) plus 1 point", "step06_table5.csv")
add("T5_sd_S", "Table 5", "site 1: StdDev of S over the A_Gr sweep (um)", 0.41, 1.5,
    "weak dependence", "step06_table5.csv")
add("T6_claim_weak_Agr", "Table 6 / Sec. 3.2",
    "site 4: max StdDev of a mineral concentration over A_Gr (%)", 1.27, 1.0, "as T5", "step06_table6.csv")
add("T6_sd_S", "Table 6", "site 4: StdDev of S over the A_Gr sweep (um)", 1.32, 1.5,
    "weak dependence", "step06_table6.csv")
add("T8_claim_site2_agg_over100", "Table 8 / Sec. 3.2",
    "agglutinate as a basis spectrum: c_AGG > 100 % at site 2 (1 = yes)", 1, 0, "claim", "step06_table8.csv")
add("T8_claim_sum_over_110", "Table 8",
    "agglutinate basis: sum of concentrations > 110 % at every site (1 = yes)", 1, 0,
    "claim ('distorts the results'); printed sums 110.7-147.5 %", "step06_table8.csv")
add("T10_claim_nonunique", "Table 10 / Sec. 4.3",
    "S spread >= 30 um across starting c_PLG while sigma_Rel spread <= 0.05 (1 = yes)", 1, 0,
    "claim; printed 41.8 um and 0.03", "step06_table10.csv")

# ---- ANN ----------------------------------------------------------------------------
for it, v, tol in ((1, 2.39, 0.5), (2, 1.26, 0.4), (3, 1.12, 0.4)):
    add("T12_sigrel_iter%d" % it, "Table 12",
        "scene-mean sigma_Rel after iteration %d (%%), median of 5 seeds" % it, v, tol,
        "preprocessing approximated (D2); about a third of the value", "step07_table12_full.csv")
add("T12_claim_order", "Table 12", "scene-mean sigma_Rel falls with each iteration (1 = yes, seed medians)",
    1, 0, "claim", "step07_table12_full.csv")
add("T12_claim_highland_worse_iter1", "Sec. 5.5", "iteration 1: highland sigma_Rel > mare (1 = yes)", 1, 0,
    "claim (3 % vs 2.2 %)", "step07_table12_full.csv")
add("T12_claim_highland_better_iter2", "Sec. 5.5", "iteration 2: highland sigma_Rel < mare (1 = yes)", 1, 0,
    "claim (1.0 % vs 1.5 %)", "step07_table12_full.csv")
add("T15_claim_site1_largest_S", "Table 15 / Sec. 5.4", "ANN iteration 3: site 1 has the largest S (1 = yes)",
    1, 0, "claim", "step07_sites_full.csv")
add("T15_claim_site1_low_nFe", "Table 15 / Sec. 5.4",
    "ANN iteration 3: c_nFe at site 1 < c_nFe at site 2, young vs mature mare (1 = yes)", 1, 0, "claim",
    "step07_sites_full.csv")
for it, v in ((1, 66), (2, 5336), (3, 4256)):
    add("T12_optepoch_iter%d" % it, "Table 12", "optimal epoch, iteration %d (seed median)" % it, v, 1e9,
        "initialisation and sample order not stated (D12); reported only", "step07_table12_full.csv", "no")
T15 = {1: (55.4, 0.041, 31.29, 31.71, 1.60, 1.58, 20.74, 13.76),
       2: (35.6, 0.095, 32.31, 23.29, 1.37, 1.48, 21.75, 15.97),
       3: (1.4, 0.046, 31.78, 38.29, 1.39, 1.22, 30.63, 6.82),
       4: (10.6, 0.063, 31.97, 29.85, 1.40, 1.32, 28.07, 13.46)}
for s, vals in T15.items():
    for k, v in zip(P, vals):
        add("T15_site%d_%s" % (s, k), "Table 15", "ANN iteration 3, site %d %s (seed median)" % (s, k), v,
            HALF10[k], B10 + "; reported only", "step07_sites_full.csv", "no")

# ---- Fig. 20 ------------------------------------------------------------------------
add("F20_claim_S_bimodal", "Fig. 20a", "histogram of S over the scene is bimodal (1 = yes)", 1, 0, "claim",
    "step09_histograms.csv")
add("F20_claim_PLG_bimodal", "Fig. 20g", "histogram of c_PLG is bimodal (1 = yes)", 1, 0, "claim",
    "step09_histograms.csv")
for k, v in (("S_lo", 9.5), ("S_hi", 34.0), ("PLG_lo", 23.0), ("PLG_hi", 28.5), ("OPX", 1.40),
             ("OLV_lo", 1.31), ("OLV_hi", 1.48), ("ILM", 16.0), ("Gr", 32.15)):
    add("F20_peak_%s" % k, "Fig. 20 (read by eye)", "histogram peak %s" % k, v, 1e9,
        "read by eye from the figure; reported only", "step09_histograms.csv", "no")

# ---- Sec. 5.6-5.7 correlations -------------------------------------------------------
add("C_ILM_TiO2_high", "Sec. 5.7 / Fig. 29b", "corr(c_ILM ANN, TiO2 Lucey Eq. 7), highlands", 0.87, 0.10,
    "closed-form TiO2; A(415) extrapolated as the paper does", "step09_correlations.csv")
add("C_CPX_A_high", "Sec. 5.6 / Fig. 21b", "corr(c_CPX ANN, <A>), highlands", 0.98, 0.05,
    "ANN against albedo only; printed to 2 d.p.", "step09_correlations.csv")
for tid, q, v, src in (("C_nFe_Is_mare", "corr(c_nFe ANN, Is MLR of Pieters 2006), mare", 0.85, "Pieters 2006"),
                       ("C_nFe_Is_high", "corr(c_nFe ANN, Is MLR of Pieters 2006), highlands", 0.83, "Pieters 2006"),
                       ("C_PLG_mare", "corr(c_PLG ANN, PLG MLR of Shkuratov 2005a), mare", 0.92, "Shkuratov 2005a")):
    add(tid, "Sec. 5.7", q, v, 0.10,
        "Eq. (3) with the published coefficients (%s Table 3) (D15)" % src, "step09_correlations.csv")
for tid, q, v in (("C_S_d", "corr(S ANN, d MLR), whole scene", 0.99),
                  ("C_OLV_high", "corr(c_OLV ANN, OLV MLR), highlands", 0.82)):
    add(tid, "Sec. 5.7", q, v, 0.10, "MLR coefficients not published, refitted on 19 LSCC soils (D15); reported only",
        "step09_correlations.csv", "no")
add("C_ILM_TiO2ANN_all", "Sec. 5.7 / Fig. 30b",
    "corr(c_ILM ANN, TiO2 by the moderately non-linear ANN of Korokhin 2008), whole scene", 0.91, 0.10,
    "that ANN retrained on Korokhin 2008 Table 1 (its published weights lack biases), gate k = 0.95 +- 0.03 (D22)",
    "step09_correlations.csv")
add("C_ILM_BD_mare", "Sec. 5.7 / Fig. 31c", "corr(c_ILM ANN, ilmenite by 1550-nm band depth, Surkov 2020), mare",
    0.54, 0.10, "closed form (Surkov 2020 Eqs. 4.10-4.12) on the recipe cube (D20)", "step09_correlations.csv")
add("C_ILM_BD_claim_high_none", "Sec. 5.7 / Fig. 31c",
    "highlands: no correlation between ANN and band-depth ilmenite, |r| < 0.3 (1 = yes)", 1, 0,
    "claim ('Highland areas have no correlation'); 0.3 = a weak correlation at most", "step09_correlations.csv")
add("C_ILM_ANN_over_BD_mare", "Sec. 5.7", "median ANN c_ILM / median band-depth ilmenite, mare ('twice')",
    2.0, 0.5, "'our method gives twice the ilmenite content'; reported only", "step09_correlations.csv", "no")

with open(kc.PAPER_ROOT / "paper_targets.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["target_id", "source", "quantity", "published_value", "tolerance", "tolerance_basis",
                "status_source", "gated"])
    w.writerows(R)
print(len(R), "targets:", dict(collections.Counter(r[7] for r in R)))
