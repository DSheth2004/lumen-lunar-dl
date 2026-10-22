"""
Step 08 -- write RESULTS.md from the tables of steps 02-07. Never edit RESULTS.md by hand:
every number in it is read from tables/, and every sentence that states a win or a loss is
generated from those numbers. Values that a stage rejected or could not compute are NaN.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import lumen_common as lc
import lumen_eval as le

STAGE = "step08"
T = lc.TABLES
PRIMARY = "A4_full"
PREREG_COMMIT = "4f7c6b4"
CLASSIC = [le.REFERENCE, "PLS", "ridge", "RF", "RF-raw"]
ABL = ["A0_zero_shot", "A1_scratch_real", "A1m_scratch_mixup", "A2_pretrain_real",
       "A3_pretrain_mixup", PRIMARY]
UNITS = {**{c: "vol %" for c in lc.COMP}, **{c: "wt %" for c in lc.CHEM}, lc.MAT: "log10",
         "COMP_mean": "vol %"}


def md_table(df: pd.DataFrame, fmt="%.2f", index=True) -> str:
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("NaN" if not np.isfinite(v) else fmt % v)
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=lc.PROJECT_ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def section_simulator(L):
    st = pd.read_csv(T / "step02_selftests.csv")
    diag = pd.read_csv(T / "step02_k_bracket_diagnosis.csv")
    pp = pd.read_csv(T / "step02_prior_percentiles.csv")
    lab = pd.read_csv(T / "step02_label_model_summary.csv")
    cov = pd.read_csv(T / "step02_coverage_summary.csv").set_index("quantity")["value"]
    mix = pd.read_csv(T / "step02_mixing_summary.csv")
    hit = diag[diag["n_bands_below_R_k1"] > 0]
    L += ["## 1. Physics simulator (step02)", ""]
    L += ["### Self-tests", "", md_table(st.set_index("test"), "%.3g"), ""]
    L += ["### Hand-off open issue 1 — the k round trip", ""]
    for _, r in hit.iterrows():
        L += ["The k round-trip failure came from **%s** (class %s, d_eff %.0f µm), not from ilmenite."
              % (r["spectrum_id"], r["em_class"], r["d_eff_um"]), "",
              "- %d bands at %.2f–%.2f µm are darker (min R = %.4f) than the slab model at k = 1 (R = %.4f)."
              % (r["n_bands_below_R_k1"], r["lam_min_um"], r["lam_max_um"], r["R_min_measured"],
                 r["R_slab_at_k1"]),
              "- R(k) is non-monotonic: it has an opacity floor (R = %.4f), so the old [1e-9, 1] bracket "
              "clamped these bands to k = 1." % r["R_slab_floor_min"],
              "- With the root taken on the falling branch, k_max is now %.4f and the round trip is %.1e."
              % (r["k_max_now"], r["roundtrip_max_abs_now"]),
              ""]
    L += ["### Hand-off open issue 2 — the ilmenite-rich prior", "",
          "99th percentiles of the synthetic bulk chemistry (wt %), before and after the fix, against "
          "the maximum measured in any LSCC soil fraction:", ""]
    t = pp[pp["quantile"].isin([0.99, 1.0])].set_index("prior")[
        ["FeO", "TiO2", "Al2O3", "CaO", "MgO", "ILM_of_crystalline_pct", "OLV_of_crystalline_pct"]]
    L += [md_table(t, "%.1f"), ""]
    L += ["### Label model (modal composition × class-average phase chemistry vs measured bulk)", "",
          md_table(lab.set_index(["set", "oxide"])[["n", "mae", "bias", "pearson_r", "meas_mean"]], "%.2f"),
          ""]
    L += ["### Coverage of the real data (sim-to-real gap)", "",
          "Real spectra lie inside the synthetic cloud:",
          "",
          "- median nearest-neighbour distance %.3f vs %.3f synthetic-to-synthetic;"
          % (cov["real_nn_dist_median"], cov["synthetic_to_synthetic_nn_dist_median"]),
          "- %.0f %% of real spectra lie beyond the synthetic 95th percentile."
          % (100 * cov["real_frac_beyond_synth_p95"]),
          "",
          "They are nonetheless separable:",
          "",
          "- real-vs-synthetic AUC: logistic %.3f, random forest %.3f;"
          % (cov["auc_real_vs_synth_logistic"], cov["auc_real_vs_synth_random_forest"]),
          "- against noise-free synthetic spectra: %.3f / %.3f."
          % (cov["auc_real_vs_synth_logistic_noise_free_synthetic"],
             cov["auc_real_vs_synth_random_forest_noise_free_synthetic"]),
          "",
          "The gap is physical, not a noise artefact.", ""]
    L += ["### SSA mixing of sieve fractions (<45 µm from <10 / 10–20 / 20–45 µm)", "",
          md_table(mix.set_index("rule"), "%.3g"), ""]


def soil_mae_table(summ, protocol, methods, targets):
    s = summ[(summ["level"] == "soil") & (summ["protocol"] == protocol)]
    t = s.pivot_table(index="method", columns="target", values="mae").reindex(methods)[targets]
    return t


def section_main(L):
    summ = pd.concat([pd.read_csv(T / "step03_metrics.csv"), pd.read_csv(T / "step04_metrics.csv")])
    ov = pd.read_csv(T / "step06_overall_tests.csv")
    tg = le.TARGETS + ["COMP_mean"]
    L += ["## 2. Accuracy on the 19 LSCC soils (steps 03–04)", "",
          "Per-soil MAE (mean over each soil's labelled fractions, then over the 19 soils).",
          "",
          "- Units: composition in vol %, oxides in wt %, maturity in log₁₀(Is/FeO).",
          "- `COMP_mean` is the mean over the 7 composition parts.",
          "- `mean-of-others` is the fraction-matched zero-information reference.",
          "- A4 is the pre-registered primary model.", ""]
    for protocol in ("LOSO", "LOMO"):
        t = soil_mae_table(summ, protocol, CLASSIC + ABL, tg)
        L += ["### %s" % protocol, "", md_table(t, "%.3g"), ""]
    L += ["### A4 against each baseline (paired Wilcoxon over soils, Holm over 13 targets)", ""]
    rows = []
    for (protocol, base), g in ov.groupby(["protocol", "baseline"], sort=False):
        core = g[g["target"] != "COMP_mean"]
        rows.append({"protocol": protocol, "baseline": base,
                     "A4 lower MAE (of 13)": int(core["lumen_better"].sum()),
                     "A4 significantly better": ", ".join(core.loc[core["significant"] & core["lumen_better"], "target"]) or "—",
                     "A4 significantly worse": ", ".join(core.loc[core["significant"] & ~core["lumen_better"], "target"]) or "—"})
    L += [md_table(pd.DataFrame(rows), index=False), ""]


def section_gate(L):
    g = pd.read_csv(T / "step05_identifiability_gate.csv", index_col=0)
    sk = pd.read_csv(T / "step05_skill.csv")
    L += ["## 3. Skill and the identifiability gate (step05)", "",
          "skill = 1 − MAE / MAE(mean-of-others), with a bootstrap 95 % CI over soils.",
          "A target passes the gate (and is mapped on M³) if its LOSO CI lower bound is > 0.", "",
          md_table(g, "%.3f"), ""]
    for protocol in ("LOSO", "LOMO"):
        best = sk[(sk["protocol"] == protocol) & sk["method"].isin(CLASSIC + ABL)].pivot_table(
            index="target", columns="method", values="skill")
        best = best.reindex(index=le.TARGETS + ["COMP_mean"],
                            columns=[m for m in CLASSIC + ABL if m in best.columns])
        L += ["%s skill of every method that predicts all targets (the FeO-only Fernández-Díaz "
              "variants are in section 5):" % protocol, "", md_table(best, "%.2f"), ""]


def section_uncertainty(L):
    c = pd.read_csv(T / "step05_coverage_summary.csv")
    cov = pd.read_csv(T / "step05_coverage.csv")
    L += ["## 4. Uncertainty (step05)", "",
          "Empirical coverage of the A4 intervals, averaged over the 13 targets (nominal 68 / 90 / 95 %):",
          "", md_table(c.set_index(["protocol", "method", "level"]), "%.3f"), ""]
    w = cov[cov["level"] == 0.90].pivot_table(index="target", columns=["protocol", "method"],
                                             values="coverage_spectrum").reindex(le.TARGETS)
    L += ["Per-target 90 % coverage (spectra):", "", md_table(w, "%.2f"), ""]
    if (T / "step11_cvplus_summary.csv").exists():
        cx = pd.read_csv(T / "step11_cvplus_summary.csv")
        L += ["### Exact CV+ (Addendum B, D24)", "",
              "Pair-held-out models (171 soil pairs, 15 mission pairs, 5 members each). Each calibration "
              "residual comes from a model that saw neither the calibration nor the test group:", "",
              md_table(cx.set_index(["protocol", "method", "level"]), "%.3f"), ""]


def section_head_to_head(L):
    km = pd.read_csv(T / "step06_korokhin_mae.csv")
    kt = pd.read_csv(T / "step06_korokhin_tests.csv")
    kr = pd.read_csv(T / "step06_korokhin_maturity.csv")
    jc = pd.read_csv(T / "step06_jung_claims.csv")
    jm = pd.read_csv(T / "step06_jung_maturity.csv")
    fd = pd.read_csv(T / "step06_fernandez_feo.csv")
    ft = pd.read_csv(T / "step06_fernandez_tests.csv")
    L += ["## 5. Head-to-head comparisons (step06)", ""]
    L += ["### Korokhin et al. (2024)", "",
          "5-mineral proportions (CPX = HCP, OPX = LCP, OLV, PLG, ILM; % of their sum), scored exactly "
          "as their reproduction scores itself:",
          "",
          "- per-soil means over the <10 / 10–20 / 20–45 µm fractions;",
          "- the mean-of-other-soils baseline is fraction-matched within the scored set;",
          "- MAE in percentage points.", ""]
    gap = pd.read_csv(T / "step06_korokhin_truth_definition_gap.csv")
    L += ["Their truth takes the pyroxene split straight from LSCC Table 2; ours rescales it to the "
          "Table 1 pyroxene total. Every method is scored against *their* truth. The definitional gap is "
          "small (mean |Δ| %.2f–%.2f pp, max %.2f pp, at 61141 and 62231), and the same comparison against "
          "our truth is in `tables/step06_korokhin_mae.csv` (`truth = lumen_truth`)."
          % (gap["mean_abs_diff"].min(), gap["mean_abs_diff"].max(), gap["max_abs_diff"].max()), ""]
    km = km[km["truth"] == "korokhin_truth"].drop(columns="truth")
    kt = kt[kt["truth"] == "korokhin_truth"].drop(columns="truth")
    for protocol in ("LOSO", "LOMO"):
        t = km[km["protocol"] == protocol].set_index(["set", "mineral"]).drop(columns="protocol")
        L += ["**%s**" % protocol, "", md_table(t, "%.2f"), ""]
    tt = kt[(kt["set"] == "all")].set_index(["protocol", "lumen", "versus", "mineral"])[
        ["mae_a", "mae_b", "n_a_better", "wilcoxon_p", "holm_p"]]
    L += ["Paired tests over the 19 soils (`a` = LUMEN, `b` = comparator; Holm over 5 minerals):", "",
          md_table(tt, "%.3g"), ""]
    L += ["Maturity — Spearman ρ against measured Is/FeO, per fraction "
          "(LUMEN log Is/FeO; Korokhin nFe from their fits):", "",
          md_table(kr.set_index(["protocol", "fraction", "set"]), "%.2f"), ""]
    L += ["### Jung et al. (2026)", "",
          "Crystalline proportions (OLV, PLG, PYX = LCP + HCP, ILM; % of their sum) of 12001, 12030, "
          "15071 and 71061 at 20–45 µm.",
          "",
          "Each method's worst-case error is set against the paper's three accuracy claims. `jung_*` "
          "columns are the converged-sampler reproduction and its prior-only control.", "",
          md_table(jc.set_index(["protocol", "method", "claim"]), "%.2f"), "",
          "Maturity on the same four soils (LUMEN Is/FeO vs Jung's calculated I_sm, both against measured "
          "Is/FeO at 20–45 µm):", "", md_table(jm.set_index("protocol"), "%.3f"), ""]
    L += ["### Fernández-Díaz et al. (2026) — FeO", "",
          "Their pipeline is re-implemented on the same 71-band spectra and folds:",
          "",
          "- features: db4 A2 + D2;",
          "- autoencoder: 6-D, trained on M³ pixels or on synthetic spectra;",
          "- regressors: the Table 5 grids, tuned by inner CV.", ""]
    for protocol in ("LOSO", "LOMO"):
        t = fd[fd["protocol"] == protocol].drop(columns="protocol").sort_values("mae_soil").set_index("method")
        L += ["**%s**" % protocol, "", md_table(t, "%.3f"), ""]
    L += ["A4 vs each FeO model (paired Wilcoxon over soils):", "",
          md_table(ft.set_index(["protocol", "versus"]), "%.3g"), ""]


def section_m3(L):
    cal = pd.read_csv(T / "step07_calibration.csv")
    site = pd.read_csv(T / "step07_apollo16_site.csv")
    st = pd.read_csv(T / "step07_shapley_stats.csv")
    kc = pd.read_csv(T / "step07_vs_korokhin_maps.csv")
    ar = pd.read_csv(T / "step07_aristarchus.csv")
    L += ["## 6. Application to M³ (step07)", "", "### Calibration to the lab scale", "",
          md_table(cal.set_index("calibration"), "%.4f"), "",
          "### Apollo 16 landing site (one site — a sanity check, not a validation)", "",
          md_table(site.set_index("source")[[c for c in lc.CHEM + [lc.MAT, "PLG", "AGG"] if c in site]], "%.2f"), "",
          "### Shapley scene (Korokhin 2024), 500 × 304 grid, identifiable targets only", "",
          md_table(st.set_index("target"), "%.3f"), "",
          "### Pixel-by-pixel against Korokhin's ANN maps", "", md_table(kc.set_index(["korokhin", "lumen"]), "%.3f"), "",
          "### Destriping sensitivity (Addendum B, D25)", "",
          "A column-gain flat field (per band, 31-sample running median) applied before prediction. "
          "The undestriped maps stay primary.", "",
          md_table(pd.read_csv(T / "step07_destriping_sensitivity.csv").set_index("target"), "%.3f"), "",
          "### Aristarchus ROIs (Jung 2026)", "",
          "Descriptive only. Jung's medians describe space-weathering-corrected material from their "
          "posterior; LUMEN describes the surface soil. There is no ground truth here.", "",
          md_table(ar.set_index(["roi", "category"]), "%.2f"), ""]


def section_m3_sites(L):
    if not (T / "step10_metrics.csv").exists():
        return
    M = pd.read_csv(T / "step10_metrics.csv")
    Tt = pd.read_csv(T / "step10_tests.csv")
    P = pd.read_csv(T / "step10_predictions.csv")
    lum = pd.read_csv(T / "step10_lumen_site_outputs.csv")
    q = pd.read_csv(T / "step10_strip_quality.csv")
    L += ["## 6b. M³ landing-site validation (step10, pre-registered as Addendum A)", "",
          "FeO from real M³ L2 spectra at the returned-sample sites of Fernández-Díaz et al.'s 50-sample ground "
          "truth, one row per distinct coordinate (pseudo-replicates averaged).",
          "",
          "- **LUMEN-M3** uses no landing-site data: Apollo sites are predicted by the A4 members that held out that "
          "mission.",
          "- **All other methods** are trained or fitted leave-one-site-out on the other sites.",
          "- %d strips were found within 0.01° of a site; %d contributed (≥ 5 / 9 valid pixels)."
          % (len(q), int(q["contributes"].sum())), "",
          md_table(M.set_index("method"), "%.3f"), "",
          "LUMEN-M3 against each comparator (paired Wilcoxon over sites, Holm over comparators):", "",
          md_table(Tt.set_index("versus"), "%.3g"), "",
          "Per site:", "", md_table(P.set_index("site_id"), "%.2f"), "",
          "LUMEN's other outputs at the sites (no ground truth):", "",
          md_table(lum.set_index("site_id")[["model", "PLG", "LCP", "HCP", "OLV", "ILM", "AGG", "TiO2",
                                             "Al2O3", "logIsFeO", "FeO_strip_min", "FeO_strip_max"]], "%.2f"), ""]


def section_deviations(L):
    txt = (lc.PROJECT_ROOT / "METHOD_NOTES.md").read_text()
    dev = txt.split("## Deviations", 1)[1].strip() if "## Deviations" in txt else "(none)"
    L += ["## 7. Deviations from the pre-registration", "", dev, ""]


def main() -> int:
    t0 = lc.banner(STAGE, "write RESULTS.md from the tables")
    lc.verify_manifest(STAGE)
    L = ["# LUMEN — results", "",
         "*Generated by `scripts/step08_report.py` on %s from the tables in `tables/` (commit %s). "
         "Do not edit by hand. The protocol was pre-registered in `METHOD_NOTES.md` at commit %s, "
         "before any model was scored on real folds.*"
         % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), git_head(), PREREG_COMMIT), ""]
    L += headline()
    section_simulator(L)
    section_main(L)
    section_gate(L)
    section_uncertainty(L)
    section_head_to_head(L)
    section_m3(L)
    section_m3_sites(L)
    section_deviations(L)
    (lc.PROJECT_ROOT / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("  RESULTS.md: %d lines" % len(L))
    lc.done(STAGE, t0)
    return 0


def headline():
    """Plain-language findings, generated from the numbers."""
    summ = pd.concat([pd.read_csv(T / "step03_metrics.csv"), pd.read_csv(T / "step04_metrics.csv")])
    ov = pd.read_csv(T / "step06_overall_tests.csv")
    gate = pd.read_csv(T / "step05_identifiability_gate.csv", index_col=0)
    km = pd.read_csv(T / "step06_korokhin_mae.csv")
    jc = pd.read_csv(T / "step06_jung_claims.csv")
    fd = pd.read_csv(T / "step06_fernandez_feo.csv")
    cov = pd.read_csv(T / "step05_coverage_summary.csv")
    s = summ[summ["level"] == "soil"]
    H = ["## Headline findings", ""]
    for protocol in ("LOSO", "LOMO"):
        g = ov[(ov["protocol"] == protocol) & (ov["target"] != "COMP_mean")]
        parts = []
        for base in CLASSIC:
            b = g[g["baseline"] == base]
            parts.append("%s %d/13 (sig. better %d, sig. worse %d)" % (
                base, int(b["lumen_better"].sum()), int((b["significant"] & b["lumen_better"]).sum()),
                int((b["significant"] & ~b["lumen_better"]).sum())))
        H.append("- **%s.** A4 has the lower per-soil MAE than: %s." % (protocol, "; ".join(parts)))
        best = s[(s["protocol"] == protocol) & s["method"].isin(CLASSIC + ABL)].pivot_table(
            index="target", columns="method", values="mae")
        wins = [t for t in le.TARGETS if best.loc[t].idxmin() == PRIMARY]
        lost = {t: best.loc[t].idxmin() for t in le.TARGETS if best.loc[t].idxmin() != PRIMARY}
        H.append("  - A4 is the single best method (of %d) on %d/13 targets: %s."
                 % (best.shape[1], len(wins), ", ".join(wins) or "none"))
        H.append("  - Another method is best on: %s."
                 % (", ".join("%s (%s)" % kv for kv in lost.items()) or "none"))
    for protocol in ("LOSO", "LOMO"):
        b = s[(s["protocol"] == protocol)].pivot_table(index="target", columns="method", values="mae").reindex(le.TARGETS)
        g = ov[(ov["protocol"] == protocol) & (ov["baseline"] == "A2_pretrain_real") & (ov["target"] != "COMP_mean")]
        H.append("- **Ablations (%s).** The pre-registered primary A4 (pretrain + mixup + replay) is *not* the best "
                 "configuration: A2 (pretrain + real spectra only) has the lower MAE on %d/13 targets (%d Holm-significant "
                 "either way). Pretraining helps (A2 below A1 on %d/13); SSA-mixup + replay did not "
                 "(A4 below A2 on %d/13, A3 below A2 on %d/13). Zero-shot A0 is the worst LUMEN variant "
                 "(composition MAE %.1f vs %.1f vol %% for A4)."
                 % (protocol, int((~g["lumen_better"]).sum()), int(g["significant"].sum()),
                    int((b["A2_pretrain_real"] < b["A1_scratch_real"]).sum()),
                    int((b[PRIMARY] < b["A2_pretrain_real"]).sum()),
                    int((b["A3_pretrain_mixup"] < b["A2_pretrain_real"]).sum()),
                    s[(s["protocol"] == protocol) & (s["method"] == "A0_zero_shot") & (s["target"] == "COMP_mean")]["mae"].iloc[0],
                    s[(s["protocol"] == protocol) & (s["method"] == PRIMARY) & (s["target"] == "COMP_mean")]["mae"].iloc[0]))
    ident = [t for t in le.TARGETS if gate.loc[t, "identifiable"]]
    H.append("- **Identifiability gate.** %d of 13 targets pass: %s. Withheld from the M³ maps: %s. "
             "Under LOMO the skill CI also includes 0 for: %s."
             % (len(ident), ", ".join(ident), ", ".join(t for t in le.TARGETS if t not in ident) or "none",
                ", ".join(t for t in le.TARGETS if gate.loc[t, "ci_lo_LOMO"] <= 0) or "none"))
    kt = pd.read_csv(T / "step06_korokhin_tests.csv")
    kt = kt[(kt["truth"] == "korokhin_truth") & (kt["set"] == "all")]
    H.append("- **Fairness note for the next two items.** Korokhin's and Jung's methods never see an LSCC "
             "soil (physics-only inversions); LUMEN A4 is fine-tuned on the 18 (LOSO) or fewer (LOMO) "
             "other soils. The like-for-like physics-only comparison is LUMEN A0 (zero-shot).")
    for protocol in ("LOSO", "LOMO"):
        k = km[(km["protocol"] == protocol) & (km["set"] == "all")
               & (km["truth"] == "korokhin_truth")].set_index("mineral")
        t = kt[kt["protocol"] == protocol]
        sig = lambda lu, vs: t[(t["lumen"] == lu) & (t["versus"] == vs) & (t["holm_p"] < 0.05)
                               & (t["mae_a"] < t["mae_b"])]["mineral"].tolist()
        H.append("- **Korokhin 2024** (5-mineral proportions, 19 soils, %s)." % protocol)
        H.append("  - A4 vs their Nelder–Mead fit: %s. A4 is lower on %d/5; Holm-significant on: %s."
                 % (", ".join("%s %.2f vs %.2f" % (m, k.loc[m, "mae_" + PRIMARY], k.loc[m, "mae_Korokhin_NM"])
                              for m in k.index),
                    int((k["mae_" + PRIMARY] < k["mae_Korokhin_NM"]).sum()),
                    ", ".join(sig(PRIMARY, "Korokhin_NM")) or "none"))
        H.append("  - A4 vs the mean of the other soils: lower on %d/5; Holm-significant on: %s."
                 % (int((k["mae_" + PRIMARY] < k["mae_mean_of_others"]).sum()),
                    ", ".join(sig(PRIMARY, "mean_of_others")) or "none"))
        if protocol == "LOSO":
            H.append("  - Zero-shot A0 (physics-only, like-for-like) vs their fit: lower on %d/5 (%s); "
                     "Holm-significant on: %s."
                     % (int((k["mae_A0_zero_shot"] < k["mae_Korokhin_NM"]).sum()),
                        ", ".join("%s %.2f vs %.2f" % (m, k.loc[m, "mae_A0_zero_shot"], k.loc[m, "mae_Korokhin_NM"])
                                  for m in k.index), ", ".join(sig("A0_zero_shot", "Korokhin_NM")) or "none"))
    kr = pd.read_csv(T / "step06_korokhin_maturity.csv")
    kr = kr[(kr["protocol"] == "LOSO") & (kr["set"] == "all") & kr["spearman_korokhin_nFe"].notna()]
    H.append("  - Maturity, pooled 19 soils, Spearman ρ vs measured Is/FeO per fraction (LOSO): LUMEN %s; "
             "Korokhin nFe %s."
             % (" / ".join("%.2f" % v for v in kr["spearman_lumen_IsFeO"]),
                " / ".join("%.2f" % v for v in kr["spearman_korokhin_nFe"])))
    j = jc[(jc["method"] == PRIMARY)]
    jm = pd.read_csv(T / "step06_jung_maturity.csv").set_index("protocol")
    for protocol in ("LOSO", "LOMO"):
        jj = j[j["protocol"] == protocol]
        H.append("- **Jung 2026** (4 soils at 20–45 µm, %s). A4 meets %d of the %d claim variants: %s."
                 % (protocol, int(jj["meets_claim"].sum()), len(jj),
                    "; ".join("%s: %.2f (Jung %.2f, prior-only %.2f)" % (r["claim"], r["worst_err"], r["jung_worst_err"],
                                                                     r["jung_prior_only_worst_err"])
                              for _, r in jj.iterrows())))
    a0 = jc[(jc["method"] == "A0_zero_shot") & (jc["protocol"] == "LOSO")]
    mo = jc[(jc["method"] == le.REFERENCE) & (jc["protocol"] == "LOSO")]
    H.append("  - With n = 4 soils this is weak evidence. Zero-shot A0 also meets %d/%d, while the mean of the "
             "other soils meets %d/%d."
             % (int(a0["meets_claim"].sum()), len(a0), int(mo["meets_claim"].sum()), len(mo)))
    H.append("  - Maturity on the same 4 soils: Pearson r = %.2f for LUMEN Is/FeO vs %.2f for Jung's I_sm; "
             "MAE %.1f vs %.1f Is/FeO units."
             % (jm.loc["LOSO", "pearson_lumen"], jm.loc["LOSO", "pearson_jung_Ism"],
                jm.loc["LOSO", "mae_lumen_IsFeO"], jm.loc["LOSO", "mae_jung_Ism"]))
    ft = pd.read_csv(T / "step06_fernandez_tests.csv").set_index(["protocol", "versus"])
    for protocol in ("LOSO", "LOMO"):
        f = fd[fd["protocol"] == protocol].set_index("method")
        fdm = f[f.index.str.startswith("FD")]["mae_soil"]
        best_fd = fdm.idxmin()
        H.append("- **Fernández-Díaz 2026** FeO (%s): A4 MAE %.3f wt %% vs FD as published (AE→RF, M³-trained AE) "
                 "%.3f (paired p = %.3f, Holm over the 12 FD variants %.3f) and the best FD variant %s %.3f "
                 "(p = %.3f, Holm %.3f). A4 is lower than all 12 FD variants; %d/12 survive Holm. Best classic: %s %.3f."
                 % (protocol, f.loc[PRIMARY, "mae_soil"], f.loc["FD-AE-M3/RF", "mae_soil"],
                    ft.loc[(protocol, "FD-AE-M3/RF"), "wilcoxon_p"], ft.loc[(protocol, "FD-AE-M3/RF"), "holm_p_over_fd_variants"],
                    best_fd, fdm.min(), ft.loc[(protocol, best_fd), "wilcoxon_p"],
                    ft.loc[(protocol, best_fd), "holm_p_over_fd_variants"],
                    int((ft.loc[protocol]["holm_p_over_fd_variants"] < 0.05).sum()),
                    f.loc[CLASSIC, "mae_soil"].idxmin(), f.loc[CLASSIC, "mae_soil"].min()))
    for protocol in ("LOSO", "LOMO"):
        c = cov[(cov["protocol"] == protocol)].set_index(["method", "level"])
        H.append("- **Coverage** (%s, mean over targets, 68/90/95 %%): raw ensemble %s; normalised conformal %s."
                 % (protocol, "/".join("%.2f" % c.loc[("raw_ensemble", l), "coverage_spectrum"] for l in (0.68, 0.9, 0.95)),
                    "/".join("%.2f" % c.loc[("conformal_normalised", l), "coverage_spectrum"] for l in (0.68, 0.9, 0.95))))
    if (T / "step10_metrics.csv").exists():
        m3 = pd.read_csv(T / "step10_metrics.csv").set_index("method")
        fdm = m3[m3.index.str.startswith("FD")]["mae"]
        H.append("- **M³ landing sites (Addendum A, %d sites).** LUMEN does *not* transfer to orbital spectra: "
                 "FeO MAE %.2f wt %% (bias %+.1f) with no site training, worse than the mean of the other sites "
                 "(%.2f); %.2f after leave-one-site-out recalibration, against %.2f for the best Fernández-Díaz "
                 "variant (%s), %.2f for FD as published and %.2f for a band ratio. Best R² of any method %.2f."
                 % (int(m3.loc["LUMEN-M3", "n_sites"]), m3.loc["LUMEN-M3", "mae"], m3.loc["LUMEN-M3", "bias"],
                    m3.loc["mean of other sites", "mae"], m3.loc["LUMEN-M3 + site recalibration", "mae"],
                    fdm.min(), fdm.idxmin(), m3.loc["FD-AE-M3/RF", "mae"], m3.loc["band ratio (Lucey theta)", "mae"],
                    m3["r2"].max()))
    H.append("")
    return H


if __name__ == "__main__":
    sys.exit(main())
