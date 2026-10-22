"""
Step 06 -- head-to-head comparisons (METHOD_NOTES.md D15, D18), each scored the way the
compared paper scores itself, on identical data and folds.

  overall     A4 vs every baseline and ablation, per target, LOSO and LOMO: per-soil MAE,
              paired Wilcoxon over soils, Holm over the 13 targets of each comparison.
  Korokhin    5-mineral proportions (HCP=CPX, LCP="OPX", OLV, PLG, ILM)/sum, <10/10-20/20-45,
              per-soil means of the proportions, vs their Nelder-Mead fits and their
              fraction-matched mean-of-other-soils baseline; pooled and by set; maturity
              Spearman per fraction (LUMEN Is/FeO and their nFe vs measured Is/FeO).
  Jung        crystalline OLV / PLG / PYX / ILM proportions of 12001, 12030, 15071, 71061 at
              20-45 um vs their three accuracy claims, their inversion and their prior-only
              control; maturity over the same four soils.
  Fernandez   FeO MAE / RMSE / R2 of A4 vs every FD variant (and the classic baselines).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import lumen_common as lc
import lumen_eval as le
import lumen_plot as lp
from step04_lumen import load_lab

STAGE = "step06"
PRIMARY = "A4_full"
K_MAP = {"CPX": "HCP", "OPX": "LCP", "OLV": "OLV", "PLG": "PLG", "ILM": "ILM"}
JUNG_SOILS = ["12001", "12030", "15071", "71061"]
JUNG_CAT = {"olivine": "OLV", "plagioclase": "PLG", "pyroxene": "PYX", "ilmenite": "ILM"}
JUNG_CLAIMS = [("olivine", 7.0, False), ("ilmenite", 10.0, True), ("ilmenite", 10.0, False),
               ("plagioclase", 30.0, False)]          # (category, bound, exclude 15071)


def method_frame(pred, protocol, method):
    return pred[(pred["protocol"] == protocol) & (pred["method"] == method)].set_index("row").sort_index()


# --------------------------------------------------------------------------
# overall
# --------------------------------------------------------------------------
def overall(pred, meta):
    rows = []
    for protocol in ("LOSO", "LOMO"):
        a = pred[(pred["protocol"] == protocol) & (pred["method"] == PRIMARY)]
        others = [m for m in pred.loc[pred["protocol"] == protocol, "method"].unique()
                  if m != PRIMARY and not m.startswith("FD")]
        for m in others:
            b = pred[(pred["protocol"] == protocol) & (pred["method"] == m)]
            block = []
            for t in le.TARGETS + ["COMP_mean"]:
                if t == "COMP_mean":
                    ea, eb = le.per_soil_comp_mean(a, meta), le.per_soil_comp_mean(b, meta)
                else:
                    ea, eb = le.per_soil_errors(a, meta, t), le.per_soil_errors(b, meta, t)
                block.append({"protocol": protocol, "baseline": m, "target": t, **le.paired(ea, eb)})
            df = pd.DataFrame(block)
            core = df["target"] != "COMP_mean"
            df.loc[core, "holm_p"] = le.holm(df.loc[core, "wilcoxon_p"])
            df["lumen_better"] = df["mae_a"] < df["mae_b"]
            df["significant"] = df["holm_p"] < 0.05
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Korokhin et al. (2024)
# --------------------------------------------------------------------------
def five_mineral(comp_frame):
    v = np.column_stack([comp_frame[K_MAP[m]] for m in K_MAP])
    return 100 * v / v.sum(1, keepdims=True)


def korokhin_scores(df, fit_cols):
    """Their summarise(): per-soil means of the proportions; fraction-matched
    mean-of-other-soils baseline within the soils of df."""
    df = df.copy()
    for m in K_MAP:
        df["prop_base_" + m] = [df[(df["soil"] != r["soil"]) & (df["fraction"] == r["fraction"])]
                                ["prop_true_" + m].mean() for _, r in df.iterrows()]
    ps = df.groupby("soil").mean(numeric_only=True)
    err = {}
    for m in K_MAP:
        t = ps["prop_true_" + m]
        err[("base", m)] = (ps["prop_base_" + m] - t).abs()
        for name, col in fit_cols.items():
            err[(name, m)] = (ps[col % m] - t).abs()
    return err


def korokhin(pred, meta):
    k = pd.read_csv(lc.BASELINE_DIR / "korokhin2024" / "step10_lscc_fits.csv", dtype={"soil": str})
    key = meta.reset_index().rename(columns={"index": "row"})[["row", "soil", "fraction"]]
    k = k.merge(key, on=["soil", "fraction"], how="left")
    assert k["row"].notna().all() and len(k) == 56
    truth = five_mineral(meta.loc[k["row"].astype(int), lc.COMP].reset_index(drop=True))
    for j, m in enumerate(K_MAP):
        assert np.allclose(truth[:, j], k["prop_true_" + m], atol=1e-6), "5-mineral truth mismatch"
    rows_mae, rows_test, rows_rho = [], [], []
    for protocol in ("LOSO", "LOMO"):
        df = k.copy()
        fit_cols = {"Korokhin_NM": "prop_fit_%s"}
        for method in (PRIMARY, "A0_zero_shot"):
            g = method_frame(pred, protocol, method)
            p = five_mineral(g.loc[k["row"].astype(int), ["pred_" + c for c in lc.COMP]]
                             .rename(columns=lambda c: c[5:]).reset_index(drop=True))
            for j, m in enumerate(K_MAP):
                df["prop_%s_%s" % (method, m)] = p[:, j]
            fit_cols[method] = "prop_" + method + "_%s"
        for set_name, sub in [("all", df)] + list(df.groupby("set")):
            err = korokhin_scores(sub, fit_cols)
            for m in K_MAP:
                r = {"protocol": protocol, "set": set_name, "mineral": m,
                     "n_soils": len(err[("base", m)]), "mae_mean_of_others": err[("base", m)].mean()}
                for name in fit_cols:
                    r["mae_" + name] = err[(name, m)].mean()
                rows_mae.append(r)
                for name in (PRIMARY, "A0_zero_shot"):
                    for ref in ("Korokhin_NM", "base"):
                        t = le.paired(err[(name, m)], err[(ref, m)])
                        rows_test.append({"protocol": protocol, "set": set_name, "mineral": m,
                                          "lumen": name, "versus": "mean_of_others" if ref == "base" else ref,
                                          **t})
        # maturity: Spearman per fraction (LUMEN Is/FeO and Korokhin nFe vs measured Is/FeO)
        g = method_frame(pred, protocol, PRIMARY)
        for frac in lc.MODAL_FRACTIONS + ["<45"]:
            for set_name in ("mare", "highland", "all"):
                sel = (meta["fraction"] == frac) & ((meta["set"] == set_name) | (set_name == "all"))
                rows = np.flatnonzero(sel.to_numpy())
                rho_l = spearmanr(g.loc[rows, "pred_" + lc.MAT], meta.loc[rows, lc.MAT])[0]
                kk = k[(k["fraction"] == frac) & ((k["set"] == set_name) | (set_name == "all"))]
                rho_k = spearmanr(kk["fit_nFe"], kk["is_feo"])[0] if len(kk) > 2 else np.nan
                rows_rho.append({"protocol": protocol, "fraction": frac, "set": set_name,
                                 "n": len(rows), "spearman_lumen_IsFeO": rho_l,
                                 "spearman_korokhin_nFe": rho_k})
    tests = pd.DataFrame(rows_test)
    for (pr, st, lu, vs), idx in tests.groupby(["protocol", "set", "lumen", "versus"]).groups.items():
        tests.loc[idx, "holm_p"] = le.holm(tests.loc[idx, "wilcoxon_p"])
    return pd.DataFrame(rows_mae), tests, pd.DataFrame(rows_rho)


# --------------------------------------------------------------------------
# Jung et al. (2026)
# --------------------------------------------------------------------------
def crystalline(frame, prefix=""):
    plg, lcp, hcp = frame[prefix + "PLG"], frame[prefix + "LCP"], frame[prefix + "HCP"]
    olv, ilm = frame[prefix + "OLV"], frame[prefix + "ILM"]
    tot = plg + lcp + hcp + olv + ilm
    return pd.DataFrame({"OLV": 100 * olv / tot, "PLG": 100 * plg / tot,
                         "PYX": 100 * (lcp + hcp) / tot, "ILM": 100 * ilm / tot})


def jung(pred, meta):
    j = pd.read_csv(lc.BASELINE_DIR / "jung2026" / "step05_vs_ground_truth.csv", dtype={"sample": str})
    rows_idx = [int(np.flatnonzero((meta["soil"] == s) & (meta["fraction"] == "20-45"))[0]) for s in JUNG_SOILS]
    truth = crystalline(meta.loc[rows_idx]).set_index(pd.Index(JUNG_SOILS))
    per = []
    for protocol in ("LOSO", "LOMO"):
        for method in (PRIMARY, "A0_zero_shot", le.REFERENCE):
            g = method_frame(pred, protocol, method).loc[rows_idx]
            p = crystalline(g, "pred_").set_index(pd.Index(JUNG_SOILS))
            for s in JUNG_SOILS:
                for cat, c in JUNG_CAT.items():
                    jr = j[(j["sample"] == s) & (j["category"] == cat)].iloc[0]
                    assert abs(jr["lab_pct"] - truth.loc[s, c]) < 1e-6, "Jung truth mismatch"
                    per.append({"protocol": protocol, "method": method, "soil": s, "category": cat,
                                "truth": truth.loc[s, c], "pred": p.loc[s, c],
                                "abs_err": abs(p.loc[s, c] - truth.loc[s, c]),
                                "jung_abs_err": jr["abs_pct_diff"],
                                "jung_abs_err_vol_basis": jr["abs_pct_diff_vol_basis"],
                                "jung_prior_only_abs_err": jr["abs_pct_diff_prior"]})
    per = pd.DataFrame(per)
    claims = []
    for (protocol, method), g in per.groupby(["protocol", "method"]):
        for cat, bound, excl in JUNG_CLAIMS:
            h = g[(g["category"] == cat) & ~((g["soil"] == "15071") & excl)]
            claims.append({"protocol": protocol, "method": method,
                           "claim": "%s < %g%s" % (cat, bound, " (15071 excepted)" if excl else ""),
                           "bound": bound, "worst_err": h["abs_err"].max(),
                           "worst_soil": h.loc[h["abs_err"].idxmax(), "soil"],
                           "meets_claim": bool(h["abs_err"].max() < bound),
                           "jung_worst_err": h["jung_abs_err"].max(),
                           "jung_prior_only_worst_err": h["jung_prior_only_abs_err"].max(),
                           "mean_err": h["abs_err"].mean(), "jung_mean_err": h["jung_abs_err"].mean()})
    claims = pd.DataFrame(claims)
    # maturity on the same four soils
    mc = pd.read_csv(lc.BASELINE_DIR / "jung2026" / "step07_maturity_check.csv", dtype={"sample": str}).set_index("sample")
    mat = []
    for protocol in ("LOSO", "LOMO"):
        g = method_frame(pred, protocol, PRIMARY).loc[rows_idx]
        lum = 10 ** g["pred_" + lc.MAT].to_numpy()
        meas = mc.loc[JUNG_SOILS, "is_feo_measured_20_45um"].to_numpy(float)
        ism = mc.loc[JUNG_SOILS, "I_sm_calculated"].to_numpy(float)
        assert np.allclose(meas, 10 ** meta.loc[rows_idx, lc.MAT].to_numpy(), rtol=1e-6)
        mat.append({"protocol": protocol, "n_soils": 4,
                    "pearson_lumen": pearsonr(lum, meas)[0], "pearson_jung_Ism": pearsonr(ism, meas)[0],
                    "spearman_lumen": spearmanr(lum, meas)[0], "spearman_jung_Ism": spearmanr(ism, meas)[0],
                    "mae_lumen_IsFeO": np.abs(lum - meas).mean(), "mae_jung_Ism": np.abs(ism - meas).mean(),
                    "max_err_lumen_IsFeO": np.abs(lum - meas).max(), "max_err_jung_Ism": np.abs(ism - meas).max()})
    return per, claims, pd.DataFrame(mat)


# --------------------------------------------------------------------------
# Fernandez-Diaz et al. (2026)
# --------------------------------------------------------------------------
def fernandez(pred, meta, summ):
    fe = summ[summ["target"] == "FeO"].copy()
    rows = []
    for protocol in ("LOSO", "LOMO"):
        a = pred[(pred["protocol"] == protocol) & (pred["method"] == PRIMARY)]
        ea = le.per_soil_errors(a, meta, "FeO")
        for m in pred.loc[pred["protocol"] == protocol, "method"].unique():
            if m == PRIMARY:
                continue
            b = pred[(pred["protocol"] == protocol) & (pred["method"] == m)]
            if "pred_FeO" not in b or b["pred_FeO"].isna().all():
                continue
            rows.append({"protocol": protocol, "versus": m, **le.paired(ea, le.per_soil_errors(b, meta, "FeO"))})
    tests = pd.DataFrame(rows)
    fd = tests["versus"].str.startswith("FD")
    for protocol in ("LOSO", "LOMO"):
        sel = fd & (tests["protocol"] == protocol)
        tests.loc[sel, "holm_p_over_fd_variants"] = le.holm(tests.loc[sel, "wilcoxon_p"])
    wide = fe.pivot_table(index=["protocol", "method"], columns="level", values=["mae", "rmse", "r2"])
    wide.columns = ["%s_%s" % (a, b) for a, b in wide.columns]
    wide = wide.drop(columns=[c for c in wide.columns if c.endswith("_soil") and not c.startswith("mae")])
    return wide.reset_index(), tests


# --------------------------------------------------------------------------
def figures(summ, kor_mae, claims, fd_wide, pred, meta):
    import matplotlib.pyplot as plt
    methods = [le.REFERENCE, "PLS", "ridge", "RF", "RF-raw", "A0_zero_shot", "A1_scratch_real",
               "A2_pretrain_real", "A3_pretrain_mixup", PRIMARY]
    soil = summ[(summ["level"] == "soil")]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, protocol in zip(axs, ("LOSO", "LOMO")):
        s = soil[soil["protocol"] == protocol]
        ref = s[s["method"] == le.REFERENCE].set_index("target")["mae"]
        x = np.arange(len(le.TARGETS))
        for i, m in enumerate(methods):
            v = s[s["method"] == m].set_index("target")["mae"].reindex(le.TARGETS)
            col = lp.METHOD_COLOR.get(m, lp.C[i % 8] if not m.startswith("A") else None)
            if m.startswith("A"):
                col = {"A0_zero_shot": "#9ec5f4", "A1_scratch_real": "#86b6ef", "A2_pretrain_real": "#5598e7",
                       "A3_pretrain_mixup": "#256abf", PRIMARY: lp.C[0]}[m]
            ax.plot(x + (i - 4.5) * 0.07, (v / ref.reindex(le.TARGETS)).values, "o", color=col,
                    ms=5 if m == PRIMARY else 3.5, label=m)
        ax.axhline(1, color=lp.INK2, lw=0.8)
        ax.set_ylabel("MAE / MAE(mean-of-others)")
        ax.set_title(protocol)
        ax.set_ylim(0, 2.5)
    axs[1].set_xticks(np.arange(len(le.TARGETS)), le.TARGETS)
    axs[0].legend(ncol=5, fontsize=7, loc="upper left")
    lp.save(fig, lc.FIGURES / "step06_relative_mae.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, protocol in zip(axs, ("LOSO", "LOMO")):
        k = kor_mae[(kor_mae["protocol"] == protocol) & (kor_mae["set"] == "all")].set_index("mineral")
        x = np.arange(5)
        for i, (col, lab, c) in enumerate([("mae_mean_of_others", "mean of other soils", "#8a8984"),
                                           ("mae_Korokhin_NM", "Korokhin 2024 (NM fit)", lp.C[5]),
                                           ("mae_A0_zero_shot", "LUMEN zero-shot", "#86b6ef"),
                                           ("mae_" + PRIMARY, "LUMEN A4", lp.C[0])]):
            ax.bar(x + (i - 1.5) * 0.2, k[col].values, 0.18, color=c, label=lab)
        ax.set_xticks(x, list(K_MAP)); ax.set_title("5-mineral proportions, 19 soils (%s)" % protocol)
    axs[0].set_ylabel("MAE (percentage points)"); axs[0].legend(fontsize=7)
    lp.save(fig, lc.FIGURES / "step06_korokhin.png")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    c = claims[claims["method"] == PRIMARY]
    x = np.arange(len(JUNG_CLAIMS))
    for i, (protocol, col) in enumerate((("LOSO", lp.C[0]), ("LOMO", "#86b6ef"))):
        ax.bar(x + (i - 1) * 0.22, c[c["protocol"] == protocol]["worst_err"].values, 0.2, color=col,
               label="LUMEN A4 %s" % protocol)
    cc = c[c["protocol"] == "LOSO"]
    ax.bar(x + 0.22, cc["jung_worst_err"].values, 0.2, color=lp.C[3], label="Jung 2026 inversion")
    ax.scatter(x, cc["bound"].values, marker="_", s=600, color=lp.INK, label="claimed bound", zorder=3)
    ax.set_xticks(x, [s.replace(" (15071 excepted)", "\n(excl. 15071)") for s in cc["claim"]], fontsize=7)
    ax.set_ylabel("worst-case abs. error (pp)"); ax.legend(fontsize=7)
    lp.save(fig, lc.FIGURES / "step06_jung.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, protocol in zip(axs, ("LOSO", "LOMO")):
        f = fd_wide[fd_wide["protocol"] == protocol].sort_values("mae_soil")
        cols = [lp.C[0] if m == PRIMARY else ("#8a8984" if not m.startswith("FD") else lp.C[6])
                for m in f["method"]]
        ax.barh(np.arange(len(f)), f["mae_soil"], color=cols)
        ax.set_yticks(np.arange(len(f)), f["method"], fontsize=6)
        ax.set_xlabel("FeO MAE over soils (wt %)"); ax.set_title(protocol)
    lp.save(fig, lc.FIGURES / "step06_fernandez_feo.png")

    fig, axs = plt.subplots(2, 4, figsize=(12, 5.6))
    g = method_frame(pred, "LOSO", PRIMARY)
    for ax, t in zip(axs.ravel(), ["PLG", "AGG", "ILM", "FeO", "TiO2", "Al2O3", "CaO", lc.MAT]):
        y = meta[t].to_numpy(float)
        ok = np.isfinite(y)
        for s, col in (("mare", lp.C[1]), ("highland", lp.C[2])):
            sel = ok & (meta["set"] == s).to_numpy()
            ax.errorbar(y[sel], g["pred_" + t].to_numpy()[sel], yerr=g["sd_" + t].to_numpy()[sel],
                        fmt="o", ms=3.5, color=col, elinewidth=0.6, label=s)
        lo, hi = np.nanmin(y), np.nanmax(y)
        ax.plot([lo, hi], [lo, hi], color=lp.INK2, lw=0.8)
        ax.set_title(t); ax.set_xlabel("measured")
    axs[0, 0].set_ylabel("LUMEN A4 (LOSO)"); axs[0, 0].legend(fontsize=7)
    lp.save(fig, lc.FIGURES / "step06_pred_vs_true.png")


def main() -> int:
    t0 = lc.banner(STAGE, "head-to-head comparisons")
    lc.verify_manifest(STAGE)
    lp.setup()
    X, meta = load_lab()
    pred = pd.concat([pd.read_csv(lc.TABLES / "step03_predictions.csv"),
                      pd.read_csv(lc.TABLES / "step04_predictions.csv")], ignore_index=True)
    summ = pd.concat([pd.read_csv(lc.TABLES / "step03_metrics.csv"),
                      pd.read_csv(lc.TABLES / "step04_metrics.csv")], ignore_index=True)

    ov = overall(pred, meta)
    ov.to_csv(lc.TABLES / "step06_overall_tests.csv", index=False)
    print("  A4 vs baselines: targets where A4 has lower per-soil MAE (Holm-significant)")
    print(ov.groupby(["protocol", "baseline"]).agg(lumen_better=("lumen_better", "sum"),
                                                   significant_better=("significant", lambda s: int((s & ov.loc[s.index, "lumen_better"]).sum())),
                                                   significant_worse=("significant", lambda s: int((s & ~ov.loc[s.index, "lumen_better"]).sum())))
          .to_string())

    kor_mae, kor_tests, kor_rho = korokhin(pred, meta)
    kor_mae.to_csv(lc.TABLES / "step06_korokhin_mae.csv", index=False)
    kor_tests.to_csv(lc.TABLES / "step06_korokhin_tests.csv", index=False)
    kor_rho.to_csv(lc.TABLES / "step06_korokhin_maturity.csv", index=False)
    print("  Korokhin 2024: 5-mineral MAE, pooled 19 soils")
    print(kor_mae[kor_mae["set"] == "all"].round(2).to_string(index=False))

    jper, jclaims, jmat = jung(pred, meta)
    jper.to_csv(lc.TABLES / "step06_jung_per_soil.csv", index=False)
    jclaims.to_csv(lc.TABLES / "step06_jung_claims.csv", index=False)
    jmat.to_csv(lc.TABLES / "step06_jung_maturity.csv", index=False)
    print("  Jung 2026: worst-case crystalline errors vs the claims")
    print(jclaims.round(2).to_string(index=False))
    print(jmat.round(3).to_string(index=False))

    fd_wide, fd_tests = fernandez(pred, meta, summ)
    fd_wide.to_csv(lc.TABLES / "step06_fernandez_feo.csv", index=False)
    fd_tests.to_csv(lc.TABLES / "step06_fernandez_tests.csv", index=False)
    print("  Fernandez-Diaz 2026: FeO")
    print(fd_wide.sort_values(["protocol", "mae_soil"]).round(3).to_string(index=False))

    figures(summ, kor_mae, jclaims, fd_wide, pred, meta)
    lc.done(STAGE, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
