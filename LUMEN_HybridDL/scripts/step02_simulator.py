"""
Step 02 -- physics simulator: self-tests, label-model check, coverage of the real data,
the synthetic pretraining set, and a real-data test of the SSA-mixing rule.

  (a) self-tests: H-function / SSA inversion, the k(lambda) round trip of every
      end-member (< 1e-6 in reflectance), a random forward -> inverse -> forward test,
      and the sign of the npFe0 effect (darkens, reddens);
      + the diagnosis of the hand-off's open issue 1 (which spectra / bands broke the
      old [1e-9, 1] bracket) and open issue 2 (prior v0 vs v1 oxide percentiles);
  (b) label model: LSCC bulk FeO / TiO2 / Al2O3 / CaO / MgO computed from the measured
      modal composition x class-average phase chemistry (the simulator's expectation),
      compared with the measured bulk chemistry;
  (c) coverage: nearest-synthetic-neighbour distance of every LSCC spectrum, set against
      synthetic-to-synthetic distances, and the AUC of real-vs-synthetic classifiers
      (the sim-to-real gap);
  (d) N seeded synthetic spectra -> products/synthetic.npz (+ feature statistics);
  (e) SSA mixing on real data: each soil's measured <45 um spectrum reconstructed from
      its <10 / 10-20 / 20-45 um spectra, with mass proportions solved from the
      fraction chemistries and Is, under four mixing rules.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import pearsonr, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lumen_common as lc
import lumen_plot as lp
import physics as ph

STAGE = "step02"
RT_TOL = 1e-6


# --------------------------------------------------------------------------
# (a) self-tests
# --------------------------------------------------------------------------
def self_tests(lib: ph.EndmemberLibrary, sim: ph.Simulator, rng) -> pd.DataFrame:
    lam = lib.lam
    rows = []

    w = np.linspace(1e-4, 0.999, 20001)
    back = ph.reflectance_to_ssa(ph.ssa_to_reflectance(w))
    rows.append(("ssa_inverse_max_abs", float(np.nanmax(np.abs(back - w))), 1e-6))

    rows.append(("k_roundtrip_max_abs_all_endmembers", float(lib.table["roundtrip_max_abs"].max()),
                 RT_TOL))
    rows.append(("k_roundtrip_n_bands_clamped", float(lib.table["n_clamped_bright"].sum()
                                                       + lib.table["n_clamped_dark"].sum()), 0.0))

    # random k spectra on the physical branch -> R -> k -> R
    errs = []
    for _ in range(30):
        n = rng.choice(list(lc.EM_N.values()))
        D = float(np.exp(rng.uniform(np.log(5), np.log(150))))
        kf = ph.k_at_reflectance_floor(lam, n, D)
        k = np.exp(rng.uniform(np.log(1e-7), np.log(kf * 0.9)))
        R = ph.slab_reflectance(k, lam, n, D)
        k2 = ph.k_from_reflectance(R, lam, n, D)
        errs.append(np.max(np.abs(ph.slab_reflectance(k2, lam, n, D) - R)))
    rows.append(("random_forward_inverse_forward_max_abs", float(np.max(errs)), RT_TOL))

    # npFe0 darkens and reddens: R(1.5)/R(0.75) must increase, R(0.75) must fall
    i075, i15 = np.argmin(np.abs(lam - 0.75)), np.argmin(np.abs(lam - 1.5))
    k = lib.k[lib.index["PLG"][0]]
    a0 = 4 * np.pi * k / lam
    r0 = ph.ssa_to_reflectance(ph.slab_ssa(a0, 1.56, k, 30.0))
    r1 = ph.ssa_to_reflectance(ph.slab_ssa(a0 + 0.2 * sim.np_unit["PLG"], 1.56, k, 30.0))
    rows.append(("npFe0_darkening_R750_ratio", float(r1[i075] / r0[i075]), 1.0))
    rows.append(("npFe0_reddening_slope_ratio",
                 float((r1[i15] / r1[i075]) / (r0[i15] / r0[i075])), 1.0))

    df = pd.DataFrame(rows, columns=["test", "value", "threshold"])
    df["pass"] = [
        df.value[0] < 1e-6, df.value[1] < RT_TOL, df.value[2] == 0, df.value[3] < RT_TOL,
        df.value[4] < 1.0, df.value[5] > 1.0,
    ]
    return df


def k_bracket_diagnosis(lib: ph.EndmemberLibrary) -> pd.DataFrame:
    """Open issue 1: which spectra / bands lie below the slab model's R(k = 1)? Those are
    exactly the bands the hand-off's [1e-9, 1] bracket clamped to k = 1."""
    lam = lib.lam
    rows = []
    for i, r in lib.table.iterrows():
        n, D = lib.n[i], r["d_eff_um"]
        R_k1 = ph.slab_reflectance(1.0, lam, n, D)
        kf = ph.k_at_reflectance_floor(lam, n, D)
        R_floor = ph.slab_reflectance(kf, lam, n, D)
        bad = lib.R[i] < R_k1
        rows.append({
            "spectrum_id": r["spectrum_id"], "em_class": r["em_class"], "d_eff_um": D,
            "n_bands_below_R_k1": int(bad.sum()),
            "lam_min_um": float(lam[bad].min()) if bad.any() else np.nan,
            "lam_max_um": float(lam[bad].max()) if bad.any() else np.nan,
            "R_min_measured": float(lib.R[i].min()), "R_slab_at_k1": float(R_k1.min()),
            "R_slab_floor_min": float(R_floor.min()), "k_floor_median": float(np.median(kf)),
            "k_max_now": float(r["k_max"]), "roundtrip_max_abs_now": float(r["roundtrip_max_abs"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# (b) label model on the LSCC soils
# --------------------------------------------------------------------------
def class_average_chem(lib: ph.EndmemberLibrary) -> dict:
    avg = {c: lib.chem[lib.index[c]].mean(0) for c in ph.CRYST}
    # OTH host in the simulator: volcanic glass with p = 0.6, else a synthetic glass
    avg["OTH"] = 0.6 * lib.chem[lib.index["VGL"]].mean(0) + 0.4 * lib.chem[lib.index["HGL"]].mean(0)
    return avg


def label_model(meta: pd.DataFrame, lib: ph.EndmemberLibrary) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg = class_average_chem(lib)
    rho = {c: lc.EM_RHO[c] for c in lc.COMP}
    non_agg = ["PLG", "LCP", "HCP", "OLV", "ILM", "OTH"]
    m = meta[meta[lc.COMP].notna().all(axis=1)].copy()
    rows = []
    for _, r in m.iterrows():
        mass = np.array([r[c] * rho[c] for c in non_agg])
        chem = (mass[:, None] * np.array([avg[c] for c in non_agg])).sum(0) / mass.sum()
        rows.append({"soil": r["soil"], "set": r["set"], "fraction": r["fraction"],
                     **{"pred_" + o: chem[j] for j, o in enumerate(ph.OXIDES)},
                     **{"meas_" + o: r[o] for o in ph.OXIDES}})
    per = pd.DataFrame(rows)
    summ = []
    for grp, g in [("all", per)] + list(per.groupby("set")):
        for o in ph.OXIDES:
            e = g["pred_" + o] - g["meas_" + o]
            summ.append({"set": grp, "oxide": o, "n": len(g), "mae": e.abs().mean(),
                         "bias": e.mean(), "pearson_r": pearsonr(g["pred_" + o], g["meas_" + o])[0],
                         "meas_mean": g["meas_" + o].mean()})
    return per, pd.DataFrame(summ)


# --------------------------------------------------------------------------
# (c) coverage of the real data
# --------------------------------------------------------------------------
def coverage(X_real, meta, sim, rng, n_ref=100_000, n_query=2000, n_cls=1500):
    lam = sim.lam
    ref = sim.sample(n_ref, rng)["R"]
    qry = sim.sample(n_query, rng)["R"]
    F_ref = lc.spectral_features(ref, lam).reshape(n_ref, -1)
    mu, sd = F_ref.mean(0), F_ref.std(0) + 1e-6
    Z_ref = (F_ref - mu) / sd
    Z_q = (lc.spectral_features(qry, lam).reshape(n_query, -1) - mu) / sd
    F_real = lc.spectral_features(X_real, lam).reshape(len(X_real), -1)
    Z_real = (F_real - mu) / sd
    L = Z_ref.shape[1]

    ref_sq = (Z_ref ** 2).sum(1)

    def nn(Z, block=250):
        dist, idx = np.empty(len(Z)), np.empty(len(Z), int)
        for s in range(0, len(Z), block):
            z = Z[s:s + block]
            d2 = (z ** 2).sum(1)[:, None] + ref_sq[None, :] - 2 * z @ Z_ref.T
            j = np.argmin(d2, axis=1)
            idx[s:s + block] = j
            dist[s:s + block] = np.sqrt(np.maximum(d2[np.arange(len(z)), j], 0) / L)
        return dist, idx

    d_q, _ = nn(Z_q)
    d_real, j_real = nn(Z_real)
    lr_real = np.log(np.clip(X_real, 1e-4, None))
    lr_nn = np.log(np.clip(ref[j_real], 1e-4, None))
    rms_log = np.sqrt(((lr_real - lr_nn) ** 2).mean(1))
    per = meta[["soil", "set", "mission", "fraction"]].copy()
    per["nn_dist_z_rms"] = d_real
    per["nn_dist_percentile_of_synth"] = [100.0 * (d_q < d).mean() for d in d_real]
    per["nn_rms_log_reflectance"] = rms_log
    resid = (lr_real - lr_nn).mean(0)

    # real-vs-synthetic classifiers; real spectra grouped by soil, 5 folds
    # (the noise-free variant separates the physics gap from the added noise / tilt)
    soil_codes = pd.factorize(meta["soil"])[0]
    groups = np.r_[soil_codes, 1000 + rng.integers(0, 50, n_cls)]
    y = np.r_[np.ones(len(Z_real)), np.zeros(n_cls)]
    aucs = {}
    for variant, randomize in (("", True), ("_noise_free_synthetic", False)):
        syn = sim.sample(n_cls, rng, randomize_output=randomize)["R"]
        Z_syn = (lc.spectral_features(syn, lam).reshape(n_cls, -1) - mu) / sd
        Z = np.vstack([Z_real, Z_syn])
        for name, clf in [("logistic", make_pipeline(StandardScaler(),
                                                     LogisticRegression(C=1.0, max_iter=5000))),
                          ("random_forest", RandomForestClassifier(300, min_samples_leaf=2,
                                                                   n_jobs=-1, random_state=0))]:
            p = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(Z, y, groups):
                clf.fit(Z[tr], y[tr])
                p[te] = clf.predict_proba(Z[te])[:, 1]
            aucs[name + variant] = roc_auc_score(y, p)
    summ = pd.DataFrame([
        {"quantity": "synthetic_to_synthetic_nn_dist_median", "value": float(np.median(d_q))},
        {"quantity": "synthetic_to_synthetic_nn_dist_p95", "value": float(np.percentile(d_q, 95))},
        {"quantity": "real_nn_dist_median", "value": float(np.median(d_real))},
        {"quantity": "real_nn_dist_max", "value": float(np.max(d_real))},
        {"quantity": "real_nn_dist_percentile_median", "value": float(per["nn_dist_percentile_of_synth"].median())},
        {"quantity": "real_frac_beyond_synth_p95", "value": float((d_real > np.percentile(d_q, 95)).mean())},
        {"quantity": "real_nn_rms_log_reflectance_median", "value": float(np.median(rms_log))},
        *[{"quantity": "auc_real_vs_synth_" + k, "value": v} for k, v in aucs.items()],
        {"quantity": "n_reference_synthetic", "value": n_ref},
    ])
    return per, summ, resid, d_q


# --------------------------------------------------------------------------
# (e) SSA mixing of the size fractions on real data
# --------------------------------------------------------------------------
MIX_OXIDES = ["SiO2", "TiO2", "Al2O3", "MgO", "CaO", "FeO", "Na2O", "K2O"]


def fraction_vectors(chem_tab: pd.DataFrame, isfeo_tab: pd.DataFrame, soil: str):
    """Major oxides + Is (= Is/FeO x FeO, additive by mass) of the three sieve fractions
    (columns of A) and of the <45 um fraction (b), with analytical sds."""
    def vec(frac):
        c = chem_tab[(chem_tab["sample"] == soil) & (chem_tab["size_fraction_um"] == frac)]
        f = isfeo_tab[(isfeo_tab["sample"] == soil) & (isfeo_tab["size_fraction_um"] == frac)]
        if c.empty or f.empty:
            return None
        v = c[MIX_OXIDES].iloc[0].to_numpy(float)
        return np.r_[v, float(f["is_feo"].iloc[0]) * float(c["FeO"].iloc[0])]
    parts = [vec(f) for f in lc.MODAL_FRACTIONS]
    bulk = vec("<45")
    if bulk is None or any(p is None for p in parts):
        return None
    sd = np.r_[np.maximum(0.03 * np.abs(bulk[:-1]), 0.03), 0.10 * bulk[-1]]
    return np.column_stack(parts), bulk, sd


def simplex_lsq(A, b, sd):
    """min || (A m - b) / sd ||, m >= 0, sum m = 1 (heavy equality row)."""
    Aw = np.vstack([A / sd[:, None], 1e3 * np.ones((1, A.shape[1]))])
    bw = np.r_[b / sd, 1e3]
    m = lsq_linear(Aw, bw, bounds=(0, 1)).x
    return m / m.sum()


def chi2(A, b, sd, m):
    return float((((A @ m - b) / sd) ** 2).sum())


MIX_RULES = {  # name -> (space, weighting)
    "ssa_cross_section": ("ssa", "xs"),        # the LUMEN mixup rule (pre-specified)
    "ssa_mass": ("ssa", "mass"),
    "reflectance_mass": ("R", "mass"),
    "reflectance_cross_section": ("R", "xs"),
}


def mixing_validation(meta, X, rng, n_boot=500):
    """(e1) forward: masses solved from the fraction chemistries + Is, then each rule
    predicts the <45 um spectrum; (e2) inverse: the effective weights that best
    reproduce the measured <45 um spectrum (in SSA or R) are converted into the mass
    proportions each rule implies, and those are scored against the chemistry (chi2)."""
    chem = pd.concat([pd.read_csv(lc.GT_DIR / "lscc_bulk_chemistry.csv"),
                      pd.read_csv(lc.GT_DIR / "lscc_highland_bulk_chemistry.csv")])
    chem["sample"] = chem["sample"].astype(str)
    isfeo = pd.concat([pd.read_csv(lc.GT_DIR / "lscc_is_feo.csv"),
                       pd.read_csv(lc.GT_DIR / "lscc_highland_is_feo.csv")])
    isfeo["sample"] = isfeo["sample"].astype(str)
    D = np.array([lc.FRACTION_NOMINAL_D_UM[f] for f in lc.MODAL_FRACTIONS])
    fwd, inv, examples = [], [], {}
    for soil in lc.ALL_SOILS:
        idx = {f: np.flatnonzero((meta["soil"] == soil) & (meta["fraction"] == f)) for f in lc.FRACTIONS}
        if any(len(v) == 0 for v in idx.values()):
            continue
        fv = fraction_vectors(chem, isfeo, soil)
        if fv is None:
            continue
        A, b, sd = fv
        m = simplex_lsq(A, b, sd)
        boot = np.array([simplex_lsq(A + rng.normal(0, 1, A.shape) * sd[:, None],
                                     b + rng.normal(0, 1, b.shape) * sd, sd) for _ in range(n_boot)])
        Rf = np.vstack([X[idx[f][0]] for f in lc.MODAL_FRACTIONS])
        Rm = X[idx["<45"][0]]
        space = {"ssa": (ph.reflectance_to_ssa(Rf), ph.reflectance_to_ssa(Rm)), "R": (Rf, Rm)}
        preds = {}
        for rule, (sp, wt) in MIX_RULES.items():
            w = m if wt == "mass" else (m / D) / (m / D).sum()
            mix_ = w @ space[sp][0]
            preds[rule] = ph.ssa_to_reflectance(mix_) if sp == "ssa" else mix_
        preds["single_fraction_10_20"] = Rf[1]
        for rule, p in preds.items():
            rel = (p - Rm) / Rm
            fwd.append({"soil": soil, "set": lc.SOIL_SET[soil], "rule": rule,
                        "rms_rel_err_pct": 100 * np.sqrt(np.mean(rel ** 2)),
                        "mean_rel_err_pct": 100 * rel.mean(),
                        "m_lt10": m[0], "m_10_20": m[1], "m_20_45": m[2],
                        "m_lt10_boot_sd": boot[:, 0].std(), "m_10_20_boot_sd": boot[:, 1].std(),
                        "m_20_45_boot_sd": boot[:, 2].std(), "chem_chi2": chi2(A, b, sd, m)})
        examples[soil] = (Rm, preds)
        row = {"soil": soil, "set": lc.SOIL_SET[soil], "chi2_chem_optimum": chi2(A, b, sd, m),
               "chi2_uniform_mass": chi2(A, b, sd, np.ones(3) / 3), "n_constraints": len(b)}
        for sp, (Af, bm) in space.items():
            x = simplex_lsq(Af.T, bm, bm)                      # relative residuals
            p = x @ Af
            pr = ph.ssa_to_reflectance(p) if sp == "ssa" else p
            row["oracle_%s_rms_rel_err_pct" % sp] = 100 * np.sqrt(np.mean(((pr - Rm) / Rm) ** 2))
            for rule, (sp2, wt) in MIX_RULES.items():
                if sp2 != sp:
                    continue
                mi = x if wt == "mass" else x * D / (x * D).sum()
                row["implied_m_lt10_" + rule] = mi[0]
                row["chi2_implied_" + rule] = chi2(A, b, sd, mi)
        inv.append(row)
    return pd.DataFrame(fwd), pd.DataFrame(inv), examples


# --------------------------------------------------------------------------
# prior percentiles (open issue 2)
# --------------------------------------------------------------------------
def prior_percentiles(sim, meta, rng, n=200_000):
    rows = []
    for prior in ("v0", "v1"):
        out = sim.sample(n, rng, prior=prior)
        chem = pd.DataFrame(out["chem"], columns=ph.OXIDES)
        comp = pd.DataFrame(out["comp"], columns=lc.COMP)
        cr = comp[ph.CRYST].div(comp[ph.CRYST].sum(axis=1), axis=0)
        for q in (0.5, 0.95, 0.99, 0.999):
            rows.append({"prior": prior, "quantile": q, **chem.quantile(q).to_dict(),
                         "ILM_of_crystalline_pct": 100 * cr["ILM"].quantile(q),
                         "OLV_of_crystalline_pct": 100 * cr["OLV"].quantile(q)})
    rows.append({"prior": "LSCC_measured", "quantile": 1.0, **meta[ph.OXIDES].max().to_dict()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-synth", type=int, default=400_000)
    ap.add_argument("--chunk", type=int, default=20_000)
    args = ap.parse_args()

    t0 = lc.banner(STAGE, "physics simulator: self-tests, label model, coverage, synthetic set")
    lc.verify_manifest(STAGE)
    lp.setup()
    ss = np.random.SeedSequence(args.seed)
    rng_test, rng_prior, rng_cov, rng_mix = [np.random.default_rng(s) for s in ss.spawn(4)]

    lam = lc.input_wavelengths_um()
    tb = time.time()
    lib = ph.EndmemberLibrary(lam)
    t_lib = time.time() - tb
    sim = ph.Simulator(lib)
    lab = np.load(lc.PRODUCTS / "lab_dataset.npz")
    X = lab["X"].astype(float)
    meta = pd.read_csv(lc.TABLES / "step01_lab_dataset.csv", dtype={"soil": str})

    # ---- (a) --------------------------------------------------------------------
    st = self_tests(lib, sim, rng_test)
    st.to_csv(lc.TABLES / "step02_selftests.csv", index=False)
    lib.table.to_csv(lc.TABLES / "step02_k_roundtrip.csv", index=False)
    diag = k_bracket_diagnosis(lib)
    diag.to_csv(lc.TABLES / "step02_k_bracket_diagnosis.csv", index=False)
    print("  (a) library built in %.1f s" % t_lib)
    print(st.to_string(index=False))
    hit = diag[diag["n_bands_below_R_k1"] > 0]
    print("  open issue 1: spectra below the slab R(k=1) (clamped by the old bracket):")
    print(hit[["spectrum_id", "em_class", "d_eff_um", "n_bands_below_R_k1", "lam_min_um",
               "lam_max_um", "R_min_measured", "R_slab_at_k1", "R_slab_floor_min",
               "k_max_now"]].to_string(index=False))
    pp = prior_percentiles(sim, meta, rng_prior)
    pp.to_csv(lc.TABLES / "step02_prior_percentiles.csv", index=False)
    print("  open issue 2: oxide percentiles, prior v0 (hand-off) vs v1 (D2b + D3b):")
    print(pp[pp["quantile"].isin([0.99, 1.0])].round(1).to_string(index=False))
    if not st["pass"].all():
        print(st[~st["pass"]])
        sys.exit("[%s] FATAL: simulator self-test failed" % STAGE)

    # ---- (b) --------------------------------------------------------------------
    per_lab, summ_lab = label_model(meta, lib)
    per_lab.to_csv(lc.TABLES / "step02_label_model.csv", index=False)
    summ_lab.to_csv(lc.TABLES / "step02_label_model_summary.csv", index=False)
    print("  (b) label model: LSCC bulk oxides from modal x class-average phase chemistry")
    print(summ_lab[summ_lab["set"] == "all"].round(2).to_string(index=False))

    # ---- (c) --------------------------------------------------------------------
    per_cov, summ_cov, resid, d_q = coverage(X, meta, sim, rng_cov)
    per_cov.to_csv(lc.TABLES / "step02_coverage.csv", index=False)
    summ_cov.to_csv(lc.TABLES / "step02_coverage_summary.csv", index=False)
    pd.DataFrame({"lam_um": lam, "mean_log_residual_real_minus_nn": resid}).to_csv(
        lc.TABLES / "step02_coverage_residual.csv", index=False)
    print("  (c) coverage of the real data")
    print(summ_cov.round(3).to_string(index=False))

    # ---- (e) --------------------------------------------------------------------
    mix, inv, examples = mixing_validation(meta, X, rng_mix)
    mix.to_csv(lc.TABLES / "step02_mixing_validation.csv", index=False)
    inv.to_csv(lc.TABLES / "step02_mixing_inverse.csv", index=False)
    a = mix[mix.rule == "ssa_cross_section"].set_index("soil")["rms_rel_err_pct"]
    ai = inv.set_index("soil")["chi2_implied_ssa_cross_section"]
    rows = []
    for rule in mix["rule"].unique():
        e = mix[mix.rule == rule].set_index("soil")["rms_rel_err_pct"].loc[a.index]
        r = {"rule": rule, "n_soils": len(a), "fwd_rms_rel_err_median": e.median(),
             "fwd_rms_rel_err_max": e.max()}
        if rule != "ssa_cross_section":
            r["fwd_n_lumen_rule_better"] = int((a < e).sum())
            r["fwd_wilcoxon_p_vs_lumen_rule"] = wilcoxon(a, e).pvalue
        if rule in MIX_RULES:
            c = inv.set_index("soil")["chi2_implied_" + rule].loc[a.index]
            r["inv_chi2_implied_median"] = c.median()
            if rule != "ssa_cross_section":
                r["inv_n_lumen_rule_better"] = int((ai < c).sum())
                r["inv_wilcoxon_p_vs_lumen_rule"] = wilcoxon(ai, c).pvalue
        rows.append(r)
    rows.append({"rule": "chemistry_optimum", "inv_chi2_implied_median": inv["chi2_chem_optimum"].median()})
    rows.append({"rule": "uniform_mass", "inv_chi2_implied_median": inv["chi2_uniform_mass"].median()})
    rows.append({"rule": "oracle_ssa_fit", "fwd_rms_rel_err_median": inv["oracle_ssa_rms_rel_err_pct"].median(),
                 "fwd_rms_rel_err_max": inv["oracle_ssa_rms_rel_err_pct"].max()})
    rows.append({"rule": "oracle_reflectance_fit", "fwd_rms_rel_err_median": inv["oracle_R_rms_rel_err_pct"].median(),
                 "fwd_rms_rel_err_max": inv["oracle_R_rms_rel_err_pct"].max()})
    ms = pd.DataFrame(rows)
    ms.to_csv(lc.TABLES / "step02_mixing_summary.csv", index=False)
    print("  (e) <45 um spectrum vs its three sieve fractions (%d soils)" % len(a))
    print("      chemistry-solved masses: median bootstrap sd of m(<10) = %.2f"
          % mix.groupby("soil")["m_lt10_boot_sd"].first().median())
    print(ms.round(3).to_string(index=False))

    # ---- (d) --------------------------------------------------------------------
    tg = time.time()
    chunks = []
    for s in np.random.SeedSequence(args.seed + 1).spawn(int(np.ceil(args.n_synth / args.chunk))):
        chunks.append(sim.sample(args.chunk, np.random.default_rng(s)))
    syn = {k: np.concatenate([c[k] for c in chunks])[:args.n_synth] for k in chunks[0]}
    t_gen = time.time() - tg
    F = lc.spectral_features(syn["R"], lam)
    feat_mean, feat_std = F.mean(0), F.std(0) + 1e-6
    np.savez(lc.PRODUCTS / "synthetic.npz", **syn, lam_um=lam, feat_mean=feat_mean,
             feat_std=feat_std, seed=args.seed + 1)
    rows = []
    for k in ["comp", "chem", "logIsFeO", "logD", "npfe_wt"]:
        v = syn[k].reshape(len(syn[k]), -1)
        names = {"comp": lc.COMP, "chem": ph.OXIDES}.get(k, [k])
        for j, nm in enumerate(names):
            q = np.percentile(v[:, j], [1, 5, 50, 95, 99])
            rows.append({"label": nm, "p01": q[0], "p05": q[1], "p50": q[2], "p95": q[3],
                         "p99": q[4], "finite_frac": float(np.isfinite(v[:, j]).mean())})
    rows.append({"label": "R", "p01": np.percentile(syn["R"], 1), "p05": np.percentile(syn["R"], 5),
                 "p50": np.median(syn["R"]), "p95": np.percentile(syn["R"], 95),
                 "p99": np.percentile(syn["R"], 99), "finite_frac": float(np.isfinite(syn["R"]).mean())})
    gs = pd.DataFrame(rows)
    gs.to_csv(lc.TABLES / "step02_synthetic_summary.csv", index=False)
    comp_sum = syn["comp"].sum(1)
    print("  (d) %d synthetic spectra in %.1f s (seed %d); composition sums %.3f-%.3f %%"
          % (args.n_synth, t_gen, args.seed + 1, comp_sum.min(), comp_sum.max()))
    print(gs.round(3).to_string(index=False))
    pd.DataFrame([{"n": args.n_synth, "seed": args.seed + 1, "generation_s": t_gen,
                   "library_build_s": t_lib,
                   "sha256": lc.sha256_of(lc.PRODUCTS / "synthetic.npz")}]).to_csv(
        lc.TABLES / "step02_synthetic_product.csv", index=False)

    figures(lib, diag, pp, per_lab, per_cov, d_q, resid, mix, inv, examples, X, meta, syn)
    lc.done(STAGE, t0)
    return 0


def figures(lib, diag, pp, per_lab, per_cov, d_q, resid, mix, inv, examples, X, meta, syn):
    import matplotlib.pyplot as plt
    lam = lib.lam
    # k of c1sc50 and its reflectance vs the slab limits
    i = int(np.flatnonzero(lib.table["spectrum_id"] == "c1sc50")[0])
    n, D = lib.n[i], lib.table["d_eff_um"].iloc[i]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(lam, lib.R[i], color=lp.C[0], label="c1sc50 measured")
    ax[0].plot(lam, ph.slab_reflectance(1.0, lam, n, D), color=lp.C[1], label="slab R(k = 1)")
    ax[0].plot(lam, ph.slab_reflectance(ph.k_at_reflectance_floor(lam, n, D), lam, n, D),
               color=lp.INK2, ls="--", label="slab floor")
    ax[0].set_xlabel("wavelength (um)"); ax[0].set_ylabel("reflectance factor"); ax[0].legend()
    ax[0].set_title("Open issue 1: L19 glass is darker than R(k = 1)")
    ax[1].semilogy(lam, lib.k[i], color=lp.C[0], label="k, falling-branch root")
    ax[1].semilogy(lam, ph.k_at_reflectance_floor(lam, n, D), color=lp.INK2, ls="--", label="k at floor")
    ax[1].set_xlabel("wavelength (um)"); ax[1].set_ylabel("k"); ax[1].legend()
    lp.save(fig, lc.FIGURES / "step02_k_c1sc50.png")

    # synthetic envelope vs LSCC
    fig, ax = plt.subplots(figsize=(6, 3.4))
    q = np.percentile(syn["R"][:50000], [1, 25, 50, 75, 99], axis=0)
    ax.fill_between(lam, q[0], q[4], color=lp.C[0], alpha=0.12, lw=0, label="synthetic 1-99 %")
    ax.fill_between(lam, q[1], q[3], color=lp.C[0], alpha=0.25, lw=0, label="synthetic 25-75 %")
    for s, col in (("mare", lp.C[1]), ("highland", lp.C[2])):
        sel = (meta["set"] == s).to_numpy()
        ax.plot(lam, X[sel].T, color=col, lw=0.7, alpha=0.8)
        ax.plot([], [], color=col, label="LSCC %s" % s)
    ax.set_xlabel("wavelength (um)"); ax.set_ylabel("reflectance factor"); ax.legend(ncol=2)
    ax.set_title("Synthetic prior vs the 75 LSCC spectra")
    lp.save(fig, lc.FIGURES / "step02_coverage_spectra.png")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(d_q, bins=60, color=lp.C[0], alpha=0.6, density=True, label="synthetic query")
    for s, col in (("mare", lp.C[1]), ("highland", lp.C[2])):
        ax[0].hist(per_cov.loc[per_cov["set"] == s, "nn_dist_z_rms"], bins=20, color=col,
                   alpha=0.7, density=True, label="LSCC %s" % s)
    ax[0].set_xlabel("NN distance to synthetic set (RMS z)"); ax[0].legend()
    ax[1].plot(lam, resid, color=lp.C[0])
    ax[1].axhline(0, color=lp.INK2, lw=0.8)
    ax[1].set_xlabel("wavelength (um)"); ax[1].set_ylabel("mean log(R_real / R_nn)")
    ax[1].set_title("Systematic sim-to-real residual")
    lp.save(fig, lc.FIGURES / "step02_coverage_nn.png")

    fig, axs = plt.subplots(1, 5, figsize=(13, 2.8))
    for a, o in zip(axs, ph.OXIDES):
        for s, col in (("mare", lp.C[1]), ("highland", lp.C[2])):
            g = per_lab[per_lab["set"] == s]
            a.scatter(g["meas_" + o], g["pred_" + o], s=14, color=col, label=s)
        lo = min(per_lab["meas_" + o].min(), per_lab["pred_" + o].min())
        hi = max(per_lab["meas_" + o].max(), per_lab["pred_" + o].max())
        a.plot([lo, hi], [lo, hi], color=lp.INK2, lw=0.8)
        a.set_title(o); a.set_xlabel("measured (wt %)")
    axs[0].set_ylabel("from modes x phase chemistry"); axs[0].legend()
    lp.save(fig, lc.FIGURES / "step02_label_model.png")

    soil = sorted(examples)[0]
    Rm, preds = examples[soil]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(lam, Rm, color=lp.INK, label="%s <45 um measured" % soil)
    for (rule, p), col in zip(preds.items(), lp.C):
        ax[0].plot(lam, p, color=col, lw=1.2, label=rule)
    ax[0].set_xlabel("wavelength (um)"); ax[0].legend(fontsize=7)
    order = list(MIX_RULES)
    ax[1].boxplot([inv["chi2_implied_" + r] for r in order] + [inv["chi2_uniform_mass"]],
                  tick_labels=[r.replace("_", "\n") for r in order] + ["uniform\nmass"])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("chemistry chi2 of implied masses"); ax[1].tick_params(axis="x", labelsize=6)
    ax[1].set_title("Inverse test, %d soils (lower = consistent)" % len(inv))
    lp.save(fig, lc.FIGURES / "step02_mixing.png")


if __name__ == "__main__":
    sys.exit(main())
