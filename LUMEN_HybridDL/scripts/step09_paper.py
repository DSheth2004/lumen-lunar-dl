"""
Step 09 -- assemble PAPER_DRAFT.md from the hand-written sections in paper/ and numbers read
from tables/ (the same tables as RESULTS.md). Every directional claim the prose makes is
asserted against the numbers first, so the draft cannot silently contradict the data.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import lumen_common as lc
import lumen_eval as le

STAGE = "step09"
T = lc.TABLES
P = lc.PROJECT_ROOT / "paper"
PRIMARY = "A4_full"
ABL = ["A0_zero_shot", "A1_scratch_real", "A1m_scratch_mixup", "A2_pretrain_real",
       "A3_pretrain_mixup", PRIMARY]


def fmt_cell(a, b, f="%.2f"):
    return (f % a if np.isfinite(a) else "NaN") + " / " + (f % b if np.isfinite(b) else "NaN")


def md(df):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [str(v) for v in r.values]) + " |")
    return "\n".join(out)


def main() -> int:
    t0 = lc.banner(STAGE, "assemble PAPER_DRAFT.md")
    lc.verify_manifest(STAGE)
    summ = pd.concat([pd.read_csv(T / "step03_metrics.csv"), pd.read_csv(T / "step04_metrics.csv")])
    soil = summ[summ["level"] == "soil"]
    mae = {p: soil[soil["protocol"] == p].pivot_table(index="target", columns="method", values="mae")
           for p in ("LOSO", "LOMO")}
    ov = pd.read_csv(T / "step06_overall_tests.csv")
    gate = pd.read_csv(T / "step05_identifiability_gate.csv", index_col=0)
    cov = pd.read_csv(T / "step05_coverage_summary.csv").set_index(["protocol", "method", "level"])
    km = pd.read_csv(T / "step06_korokhin_mae.csv")
    km = km[(km["truth"] == "korokhin_truth") & (km["set"] == "all")]
    kt = pd.read_csv(T / "step06_korokhin_tests.csv")
    kt = kt[(kt["truth"] == "korokhin_truth") & (kt["set"] == "all")]
    kr = pd.read_csv(T / "step06_korokhin_maturity.csv")
    jc = pd.read_csv(T / "step06_jung_claims.csv")
    jm = pd.read_csv(T / "step06_jung_maturity.csv").set_index("protocol")
    fd = pd.read_csv(T / "step06_fernandez_feo.csv").set_index(["protocol", "method"])
    ft = pd.read_csv(T / "step06_fernandez_tests.csv").set_index(["protocol", "versus"])
    st = pd.read_csv(T / "step02_selftests.csv").set_index("test")["value"]
    diag = pd.read_csv(T / "step02_k_bracket_diagnosis.csv")
    pp = pd.read_csv(T / "step02_prior_percentiles.csv")
    lab = pd.read_csv(T / "step02_label_model_summary.csv")
    c2 = pd.read_csv(T / "step02_coverage_summary.csv").set_index("quantity")["value"]
    mix = pd.read_csv(T / "step02_mixing_summary.csv").set_index("rule")
    cal = pd.read_csv(T / "step07_calibration.csv")
    site = pd.read_csv(T / "step07_apollo16_site.csv").set_index("source")
    sh = pd.read_csv(T / "step07_shapley_stats.csv").set_index("target")
    kmap = pd.read_csv(T / "step07_vs_korokhin_maps.csv")
    ar = pd.read_csv(T / "step07_aristarchus.csv")
    synp = pd.read_csv(T / "step02_synthetic_product.csv")

    V = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "n_synth": int(synp["n"].iloc[0])}
    # --- simulator
    p99 = pp[pp["quantile"] == 0.99].set_index("prior")
    V.update(rt_max=st["k_roundtrip_max_abs_all_endmembers"],
             k_issue_bands=int(diag["n_bands_below_R_k1"].sum()),
             tio2_p99_v0=p99.loc["v0", "TiO2"], tio2_p99_v1=p99.loc["v1", "TiO2"],
             feo_p99_v0=p99.loc["v0", "FeO"], feo_p99_v1=p99.loc["v1", "FeO"],
             tio2_lscc_max=pp[pp["prior"] == "LSCC_measured"]["TiO2"].iloc[0],
             lab_r_min=lab[lab["set"] == "all"]["pearson_r"].min(),
             lab_r_max=lab[lab["set"] == "all"]["pearson_r"].max(),
             lab_mare_tio2_bias=lab[(lab["set"] == "mare") & (lab["oxide"] == "TiO2")]["bias"].iloc[0],
             cov_real=c2["real_nn_dist_median"], cov_syn=c2["synthetic_to_synthetic_nn_dist_median"],
             auc_rf=c2["auc_real_vs_synth_random_forest"],
             auc_rf_nf=c2["auc_real_vs_synth_random_forest_noise_free_synthetic"],
             mix_oracle=mix.loc["oracle_ssa_fit", "fwd_rms_rel_err_median"])
    # --- main accuracy
    L, M = mae["LOSO"], mae["LOMO"]
    sk = pd.read_csv(T / "step05_skill.csv")
    skl = sk[(sk["protocol"] == "LOSO") & (sk["method"] == PRIMARY)].set_index("target")["skill"]

    def ovc(protocol, base, what):
        g = ov[(ov["protocol"] == protocol) & (ov["baseline"] == base) & (ov["target"] != "COMP_mean")]
        return int({"better": g["lumen_better"], "sig": g["significant"] & g["lumen_better"]}[what].sum())
    frf = ov[(ov["protocol"] == "LOSO") & (ov["baseline"] == "RF") & (ov["target"] == "FeO")].iloc[0]
    V.update(skill_comp_loso=skl["COMP_mean"], skill_feo_loso=skl["FeO"],
             n_better_rf_loso=ovc("LOSO", "RF", "better"), n_better_rf_lomo=ovc("LOMO", "RF", "better"),
             n_sig_pls_loso=ovc("LOSO", "PLS", "sig"), n_sig_mo_loso=ovc("LOSO", le.REFERENCE, "sig"),
             p_feo_rf_loso=frf["wilcoxon_p"], holm_feo_rf_loso=frf["holm_p"],
             comp_a4_loso=L.loc["COMP_mean", PRIMARY], comp_a4_lomo=M.loc["COMP_mean", PRIMARY],
             feo_a4_loso=L.loc["FeO", PRIMARY], feo_a4_lomo=M.loc["FeO", PRIMARY],
             tio2_a4_loso=L.loc["TiO2", PRIMARY], comp_a0=L.loc["COMP_mean", "A0_zero_shot"],
             lomo_factor=float(np.median((M.loc[le.TARGETS, PRIMARY] / L.loc[le.TARGETS, PRIMARY]).values)))
    for base in ("mean-of-others", "PLS", "ridge", "RF", "RF-raw"):
        g = ov[(ov["protocol"] == "LOSO") & (ov["baseline"] == base) & (ov["target"] != "COMP_mean")]
        assert g.loc[~g["lumen_better"], "target"].tolist() == ["OLV"], "only-olivine claim (%s)" % base
    rf_sig = ov[(ov["protocol"] == "LOSO") & (ov["baseline"] == "RF") & ov["significant"]]["target"].tolist()
    assert rf_sig == ["logIsFeO"], "prose says only maturity is significant vs RF (LOSO): %s" % rf_sig
    lomo_rf_loss = [t for t in le.TARGETS if M.loc[t, PRIMARY] >= M.loc[t, "RF"]]
    assert lomo_rf_loss == ["OLV", "OTH", "MgO"], "prose names OLV, OTH, MgO as LOMO losses vs RF: %s" % lomo_rf_loss
    others = [m for m in M.columns if m != le.REFERENCE and np.isfinite(M.loc["OTH", m])]
    assert all(M.loc["OTH", m] > M.loc["OTH", le.REFERENCE] for m in others), "OTH LOMO claim"
    mo_lomo = ov[(ov["protocol"] == "LOMO") & (ov["baseline"] == le.REFERENCE)].set_index("target")
    assert not mo_lomo.loc["AGG", "significant"] and not mo_lomo.loc["OTH", "significant"]
    # --- ablations
    cnt = lambda T_, a, b: int((T_.loc[le.TARGETS, a] < T_.loc[le.TARGETS, b]).sum())
    V.update(n_a2_vs_a1_loso=cnt(L, "A2_pretrain_real", "A1_scratch_real"),
             n_a2_vs_a1_lomo=cnt(M, "A2_pretrain_real", "A1_scratch_real"),
             n_a4_vs_a2_loso=cnt(L, PRIMARY, "A2_pretrain_real"),
             n_a1m_vs_a1_loso=cnt(L, "A1m_scratch_mixup", "A1_scratch_real"),
             n_a1m_vs_a1_lomo=cnt(M, "A1m_scratch_mixup", "A1_scratch_real"))
    assert V["n_a1m_vs_a1_loso"] > 6 and V["n_a1m_vs_a1_lomo"] <= 6, "mixup-from-scratch claim"
    assert not ov[(ov["baseline"] == "A2_pretrain_real")]["significant"].any(), "A4 vs A2 claim"
    assert L.loc["COMP_mean", "A0_zero_shot"] == L.loc["COMP_mean", ABL].max(), "A0 worst claim"
    # --- Korokhin
    kl = km[km["protocol"] == "LOSO"].set_index("mineral")
    kL = km[km["protocol"] == "LOMO"].set_index("mineral")
    sig = lambda prot, lu: kt[(kt["protocol"] == prot) & (kt["lumen"] == lu) & (kt["versus"] == "Korokhin_NM")
                              & (kt["holm_p"] < 0.05) & (kt["mae_a"] < kt["mae_b"])]["mineral"].tolist()
    red = 100 * (1 - kl["mae_" + PRIMARY] / kl["mae_Korokhin_NM"])
    rho = kr[(kr["protocol"] == "LOSO") & (kr["set"] == "all") & kr["spearman_korokhin_nFe"].notna()]
    V.update(kor_sig_loso=", ".join(sig("LOSO", PRIMARY)), kor_sig_lomo=", ".join(sig("LOMO", PRIMARY)) or "none",
             kor_olv_a4=kl.loc["OLV", "mae_" + PRIMARY], kor_olv_k=kl.loc["OLV", "mae_Korokhin_NM"],
             kor_plg_a4=kl.loc["PLG", "mae_" + PRIMARY], kor_plg_k=kl.loc["PLG", "mae_Korokhin_NM"],
             kor_plg_a4_lomo=kL.loc["PLG", "mae_" + PRIMARY],
             kor_n_lomo=int((kL["mae_" + PRIMARY] < kL["mae_Korokhin_NM"]).sum()),
             kor_a0_n=int((kl["mae_A0_zero_shot"] < kl["mae_Korokhin_NM"]).sum()),
             kor_a0_sig=", ".join(sig("LOSO", "A0_zero_shot")) or "none",
             kor_reduction_min=red.min(), kor_reduction_max=red.max(),
             kor_rho_l=" / ".join("%.2f" % v for v in rho["spearman_lumen_IsFeO"]),
             kor_rho_k=" / ".join("%.2f" % v for v in rho["spearman_korokhin_nFe"]))
    assert len(sig("LOSO", PRIMARY)) == 5 and sig("LOSO", "A0_zero_shot") == ["OLV"]
    assert kL.loc["PLG", "mae_" + PRIMARY] > kL.loc["PLG", "mae_Korokhin_NM"]
    # --- Jung
    jl = jc[(jc["protocol"] == "LOSO") & (jc["method"] == PRIMARY)].set_index("claim")
    V.update(jung_olv=jl.loc["olivine < 7", "worst_err"], jung_ilm=jl.loc["ilmenite < 10", "worst_err"],
             jung_plg=jl.loc["plagioclase < 30", "worst_err"], jung_ilm_j=jl.loc["ilmenite < 10 (15071 excepted)", "jung_worst_err"],
             jung_r_l=jm.loc["LOSO", "pearson_lumen"], jung_r_j=jm.loc["LOSO", "pearson_jung_Ism"])
    assert jc[jc["method"] == PRIMARY]["meets_claim"].all()
    assert jc[(jc["method"] == "A0_zero_shot")]["meets_claim"].all()
    mo = jc[(jc["method"] == le.REFERENCE) & (jc["protocol"] == "LOSO")].set_index("claim")
    assert not mo.loc["ilmenite < 10", "meets_claim"] and not mo.loc["plagioclase < 30", "meets_claim"]
    # --- Fernandez-Diaz
    fdm = {p: fd.loc[p]["mae_soil"] for p in ("LOSO", "LOMO")}
    fdv = {p: fdm[p][fdm[p].index.str.startswith("FD")] for p in fdm}
    best_lomo = fdv["LOMO"].idxmin()
    V.update(fd_a4_loso=fdm["LOSO"][PRIMARY], fd_pub_loso=fdm["LOSO"]["FD-AE-M3/RF"],
             fd_best_loso=fdv["LOSO"].min(), fd_a4_lomo=fdm["LOMO"][PRIMARY],
             fd_pub_lomo=fdm["LOMO"]["FD-AE-M3/RF"], fd_best_lomo=fdv["LOMO"].min(),
             fd_best_lomo_p=ft.loc[("LOMO", best_lomo), "wilcoxon_p"],
             fd_holm_loso=int((ft.loc["LOSO"]["holm_p_over_fd_variants"] < 0.05).sum()),
             fd_reduction=100 * (1 - fdm["LOSO"][PRIMARY] / fdm["LOSO"]["FD-AE-M3/RF"]))
    assert fdv["LOSO"].idxmin() == "FD-wavelet/SVR-RBF"
    assert (ft.loc["LOSO"].loc[fdv["LOSO"].index, "wilcoxon_p"] < 0.05).all()
    assert (fdv["LOSO"] > fdm["LOSO"][PRIMARY]).all() and (fdv["LOMO"] > fdm["LOMO"][PRIMARY]).all()
    regs = ["Lasso", "SVR-lin", "SVR-RBF", "RF"]
    V["fd_ae_worse_loso"] = int(sum(fdm["LOSO"]["FD-AE-M3/" + r] > fdm["LOSO"]["FD-wavelet/" + r] for r in regs))
    V["fd_ae_worse_lomo"] = int(sum(fdm["LOMO"]["FD-AE-M3/" + r] > fdm["LOMO"]["FD-wavelet/" + r] for r in regs))
    assert V["fd_ae_worse_loso"] >= 3
    assert all(fdm["LOMO"]["FD-AE-M3/" + r] < fdm["LOMO"]["FD-wavelet/" + r] for r in ("Lasso", "SVR-lin"))
    # --- head-to-head tally
    wins, excs = 0, []
    for p in ("LOSO", "LOMO"):
        k = km[km["protocol"] == p].set_index("mineral")
        for m in k.index:
            ok = k.loc[m, "mae_" + PRIMARY] < k.loc[m, "mae_Korokhin_NM"]
            wins += ok
            excs += [] if ok else ["%s vs Korokhin (%s)" % (m, p)]
        j = jc[(jc["protocol"] == p) & (jc["method"] == PRIMARY)]
        for _, r in j.iterrows():
            ok = r["worst_err"] < r["jung_worst_err"]
            wins += ok
            excs += [] if ok else ["%s vs Jung (%s)" % (r["claim"], p)]
        ok = fdm[p][PRIMARY] < fdm[p]["FD-AE-M3/RF"]
        wins += ok
        excs += [] if ok else ["FeO vs Fernández-Díaz (%s)" % p]
    V.update(n_pub_wins=int(wins), n_pub_comparisons=10 + int(len(jc[jc["method"] == PRIMARY])) + 2,
             pub_exceptions=", ".join(excs) or "none")
    # --- uncertainty / gate
    cv = lambda p, m: " / ".join("%.2f" % cov.loc[(p, m, l), "coverage_spectrum"] for l in (0.68, 0.9, 0.95))
    V.update(cov_raw_loso=cv("LOSO", "raw_ensemble"), cov_raw_lomo=cv("LOMO", "raw_ensemble"),
             cov_conf_loso=cv("LOSO", "conformal_normalised"), cov_conf_lomo=cv("LOMO", "conformal_normalised"),
             n_ident=int(gate.loc[le.TARGETS, "identifiable"].sum()))
    assert list(gate.index[~gate["identifiable"]]) == ["OLV", "OTH"] and gate.loc["OLV", "skill_LOSO"] < 0
    assert gate.loc["AGG", "ci_lo_LOMO"] <= 0
    # --- M3
    calx = cal.set_index("calibration")["rms_rel_err_vs_62231_pct"]
    eq2 = [s for s in site.index if s.startswith("LUMEN on M3 (Korokhin Eq. 2")][0]
    V.update(cal_eq2=calx.iloc[0], cal_ours=calx.iloc[2], a16_feo=site.loc[eq2, "FeO"],
             a16_tio2=site.loc[eq2, "TiO2"], a16_feo_meas=site.loc["measured 62231 <45 um", "FeO"],
             a16_tio2_meas=site.loc["measured 62231 <45 um", "TiO2"],
             a16_feo_mean=site.filter(like="measured A16", axis=0)["FeO"].iloc[0],
             sh_feo_m=sh.loc["FeO", "mean_mare"], sh_feo_h=sh.loc["FeO", "mean_highland"],
             sh_al_m=sh.loc["Al2O3", "mean_mare"], sh_al_h=sh.loc["Al2O3", "mean_highland"],
             kmap_cpx=kmap[(kmap["korokhin"] == "CPX")]["pearson"].iloc[0],
             kmap_nfe=kmap[(kmap["korokhin"] == "nFe")]["pearson"].iloc[0])
    ab = ar[(ar["roi"] == "roi_b") & (ar["category"] == "plagioclase")].iloc[0]
    V.update(ar_plg_b=ab["lumen_median_all"], ar_plg_b_j=ab["jung_median_wt"])
    cx = kmap[kmap["korokhin"] == "CPX"].iloc[0]
    assert cx["mare_minus_highland_korokhin"] < 0 < cx["mare_minus_highland_lumen"]
    assert V["sh_feo_m"] > V["sh_feo_h"] and V["sh_al_m"] < V["sh_al_h"]
    assert V["a16_tio2"] > V["a16_tio2_meas"]
    assert abs(V["kmap_nfe"]) < 0.4, "nFe vs Is/FeO 'nearly unrelated' claim"
    assert (rho["spearman_lumen_IsFeO"].values > rho["spearman_korokhin_nFe"].values).all()
    rw = kr[(kr["protocol"] == "LOSO") & (kr["set"] != "all") & kr["spearman_korokhin_nFe"].notna()]
    assert (rw["spearman_lumen_IsFeO"].values >= rw["spearman_korokhin_nFe"].values - 1e-9).all()

    cx = pd.read_csv(T / "step11_cvplus_summary.csv").set_index(["protocol", "method", "level"])
    V["cvx_loso"] = " / ".join("%.2f" % cx.loc[("LOSO", "cvplus_normalised", l), "coverage_spectrum"] for l in (0.68, 0.9, 0.95))
    V["cvx_lomo"] = " / ".join("%.2f" % cx.loc[("LOMO", "cvplus_normalised", l), "coverage_spectrum"] for l in (0.68, 0.9, 0.95))
    # --- M3 landing sites (Addendum A)
    m3 = pd.read_csv(T / "step10_metrics.csv").set_index("method")
    m3t = pd.read_csv(T / "step10_tests.csv")
    m3s = pd.read_csv(lc.DATA / "m3_sites" / "sites.csv")
    m3q = pd.read_csv(T / "step10_strip_quality.csv")
    fdm3 = m3[m3.index.str.startswith("FD")]
    V.update(m3_n_sites=int(m3.loc["LUMEN-M3", "n_sites"]), m3_n_strips=len(m3q),
             m3_lumen_mae=m3.loc["LUMEN-M3", "mae"], m3_lumen_bias=m3.loc["LUMEN-M3", "bias"],
             m3_lumen_bias_abs=abs(m3.loc["LUMEN-M3", "bias"]),
             m3_mo_mae=m3.loc["mean of other sites", "mae"],
             m3_recal_mae=m3.loc["LUMEN-M3 + site recalibration", "mae"],
             m3_fd_best=fdm3["mae"].idxmin(), m3_fd_best_mae=fdm3["mae"].min(),
             m3_fd_pub_mae=m3.loc["FD-AE-M3/RF", "mae"], m3_fd_pub_r2=m3.loc["FD-AE-M3/RF", "r2"],
             m3_ratio_mae=m3.loc["band ratio (Lucey theta)", "mae"], m3_best_r2=m3["r2"].max(),
             m3_n_comp=len(m3t), m3_n_sig_worse=int(((m3t["holm_p"] < 0.05) & (m3t["mae_lumen"] > m3t["mae_other"])).sum()),
             m3_a17_range=m3s["feo_range"].max())
    assert V["m3_lumen_mae"] > V["m3_mo_mae"] and V["m3_lumen_bias"] < 0
    assert V["m3_recal_mae"] > V["m3_fd_best_mae"] and V["m3_recal_mae"] > V["m3_ratio_mae"]
    assert V["m3_best_r2"] < 0.3

    ftm = pd.read_csv(T / "step12_metrics.csv").set_index("method")
    ftt = pd.read_csv(T / "step12_tests.csv")
    V.update(ft_mae=ftm.loc["LUMEN-M3-FT (orbital fine-tuning)", "mae"], ft_r2=ftm.loc["LUMEN-M3-FT (orbital fine-tuning)", "r2"],
             ft_n_sig=int((ftt["holm_p"] < 0.05).sum()), ft_n_comp=len(ftt), ft_n_pix=594)
    assert V["ft_mae"] < ftm[ftm.index.str.startswith("FD")]["mae"].min() and V["ft_mae"] == ftm["mae"].min()

    # --- tables
    rows = {}
    for t in le.TARGETS + ["COMP_mean"]:
        rows[t] = {m: fmt_cell(L.loc[t, m], M.loc[t, m]) for m in [le.REFERENCE, "PLS", "ridge", "RF", PRIMARY]}
    t1 = pd.DataFrame(rows).T
    t1.index.name = "target"
    V["table_main"] = ("**Table 1.** Per-soil MAE, leave-one-soil-out / leave-one-mission-out (composition "
                       "vol %, oxides wt %, maturity log₁₀ Is/FeO).\n\n" + md(t1))
    rows = {t: {m: fmt_cell(L.loc[t, m], M.loc[t, m]) for m in ABL} for t in le.TARGETS + ["COMP_mean"]}
    t2 = pd.DataFrame(rows).T
    t2.index.name = "target"
    V["table_ablation"] = ("**Table 2.** Ablations, per-soil MAE, leave-one-soil-out / leave-one-mission-out. "
                           "A4 (pretrain + mixup + replay) is the pre-registered primary model.\n\n" + md(t2))
    tk = pd.DataFrame({
        "mean of other soils": kl["mae_mean_of_others"].map("%.2f".__mod__),
        "Korokhin NM fit": kl["mae_Korokhin_NM"].map("%.2f".__mod__),
        "LUMEN A0 (zero-shot)": kl["mae_A0_zero_shot"].map("%.2f".__mod__),
        "LUMEN A4 (LOSO)": kl["mae_" + PRIMARY].map("%.2f".__mod__),
        "LUMEN A4 (LOMO)": kL["mae_" + PRIMARY].map("%.2f".__mod__),
        "Holm p, A4 vs NM (LOSO)": kt[(kt["protocol"] == "LOSO") & (kt["lumen"] == PRIMARY)
                                      & (kt["versus"] == "Korokhin_NM")].set_index("mineral")["holm_p"].map("%.3f".__mod__)})
    tk.index.name = "mineral"
    V["table_korokhin"] = ("**Table 3.** Korokhin et al. (2024) comparison: MAE of the 5-mineral proportions "
                           "(percentage points), 19 soils, scored as their reproduction scores itself.\n\n" + md(tk))
    meths = [PRIMARY, "A2_pretrain_real"] + sorted(fdv["LOSO"].index) + ["RF", "RF-raw", le.REFERENCE]
    tf = pd.DataFrame({
        "MAE LOSO": [("%.2f" % fd.loc[("LOSO", m), "mae_soil"]) for m in meths],
        "R² LOSO": [("%.2f" % fd.loc[("LOSO", m), "r2_spectrum"]) for m in meths],
        "MAE LOMO": [("%.2f" % fd.loc[("LOMO", m), "mae_soil"]) for m in meths],
        "R² LOMO": [("%.2f" % fd.loc[("LOMO", m), "r2_spectrum"]) for m in meths]}, index=meths)
    tf.index.name = "method"
    V["table_fd"] = ("**Table 4.** FeO (wt %) on identical spectra and folds: LUMEN against the Fernández-Díaz et al. "
                     "(2026) pipeline (FD-AE-M3/RF is the published configuration) and classic baselines.\n\n" + md(tf))

    results = (P / "results_template.md").read_text().format(**V)
    V["intro"] = (P / "intro_sections.md").read_text()
    V["methods"] = (P / "methods_sections.md").read_text()
    V["results"] = results
    out = (P / "frame_template.md").read_text().format(**V)
    (lc.PROJECT_ROOT / "PAPER_DRAFT.md").write_text(out)
    print("  PAPER_DRAFT.md: %d lines, %d words" % (out.count("\n"), len(out.split())))
    lc.done(STAGE, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
