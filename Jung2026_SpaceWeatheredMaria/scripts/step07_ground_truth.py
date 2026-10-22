"""
Step 07 -- compare the inverted modal mineralogy against the LSCC lab values.

This is the comparison behind the paper's Figure 4 / Figure F1 (the stars and
grey bars) and behind the three accuracy claims made in Section 3:

    "olivine ... differs from LSCC lab values by less than 7 %"
    "ilmenite deviations are typically less than 10 % (15071 excepted)"
    "plagioclase discrepancies of 20-30 % in some samples"

Both sides are renormalised the way Figure 4 does it -- to the crystalline
phases only, excluding agglutinitic glass, volcanic glass and the unspecified
"Others" -- because the inversion's glass and agglutinate abundances have no
counterpart in the lab point counts.

The four inverted spectra are all the 20-45 um sieve fraction (verified from the
RELAB PDS4 labels in step00), so the matching LSCC column is fixed, not chosen.

Two caveats are carried as columns rather than resolved silently:

  * The LSCC values are *modal* abundances from point counting, i.e. by volume,
    while the sampler's parameters are mass fractions.  The comparison the paper
    plots is against the published modal numbers, so that is what is gated; the
    density-converted weight-percent version is computed alongside as a
    sensitivity, using the paper's own Table C1 densities.
  * Lab "pyroxene" is a single category, so pigeonite and augite are summed
    before comparing -- the same grouping Figure 4 uses.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import jung_common as jc

STAGE = "step07"

SOILS = ["12001", "12030", "15071", "71061"]
FRACTION = "20-45"          # fixed by the RELAB labels, see step00 metadata

# lab category  ->  the model end-members that make it up
GROUPS = {
    "olivine":     ["olivine"],
    "plagioclase": ["plagioclase"],
    "pyroxene":    ["pigeonite", "augite"],
    "ilmenite":    ["ilmenite"],
}


def lab_table() -> pd.DataFrame:
    """Crystalline-only LSCC modal abundances, renormalised to 100 %."""
    src = jc.DATA / "ground_truth" / "lscc_modal_mineralogy.csv"
    if not src.exists():
        sys.exit("[%s] FATAL: %s missing. See data/ground_truth/provenance.md." % (STAGE, src))
    df = pd.read_csv(src)
    df["sample"] = df["sample"].astype(str)
    df = df[(df["size_fraction_um"] == FRACTION) & (df["source_table"] == "Table 1")]
    out = []
    for s in SOILS:
        sub = df[df["sample"] == s].set_index("phase")["modal_pct"]
        raw = {c: float(sub[c]) for c in GROUPS}
        tot = sum(raw.values())
        for c, v in raw.items():
            out.append(dict(sample=s, category=c, lab_modal_pct_raw=v,
                            lab_crystalline_total=tot, lab_pct=100.0 * v / tot))
    return pd.DataFrame(out)


PRIOR_SEED = 0
PRIOR_DRAWS = 200_000


def prior_baseline() -> dict:
    """The same categories drawn from the Dirichlet prior alone, no data used.

    This is the no-information control for the three accuracy claims.  A claim
    that the prior alone already satisfies cannot be evidence that the inversion
    recovered mineralogy from the spectrum, so it is reported next to every
    comparison rather than left implicit.
    """
    rng = np.random.default_rng(PRIOR_SEED)
    f = rng.dirichlet(jc.PRIOR_MEANS * jc.PRIOR_ALPHA0, size=PRIOR_DRAWS)
    names = list(jc.ENDMEMBERS["name"])
    cry = [names.index(n) for n in jc.CRYSTALLINE]
    m = f[:, cry]
    m = m / m.sum(axis=1, keepdims=True) * 100.0
    sub = {n: k for k, n in enumerate(jc.CRYSTALLINE)}
    return {c: float(np.median(m[:, [sub[n] for n in members]].sum(axis=1)))
            for c, members in GROUPS.items()}


def maturity_check() -> pd.DataFrame | None:
    """Eq. (3)'s calculated I_sm against the maturity it is meant to estimate.

    I_sm is calibrated against Morris (1978) Is/FeO, which the LSCC tables now
    supply for exactly these soils and sieve fractions.  Reported, not gated:
    the paper's Table 1 values are reproduced to 0.28 by step02, so this is a
    statement about the semiempirical relation itself, not about our arithmetic.
    Fig. 2 shows scatter about the fit, and part of this difference is that
    scatter seen four points at a time.
    """
    isf = jc.DATA / "ground_truth" / "lscc_is_feo.csv"
    t1 = jc.TABLES / "step02_table1_reproduction.csv"
    if not (isf.exists() and t1.exists()):
        return None
    m = pd.read_csv(isf)
    m["sample"] = m["sample"].astype(str)
    calc = pd.read_csv(t1)
    calc["sample"] = calc["sample"].astype(str)
    calc = calc.set_index("sample")

    rows, rows_c = [], []
    for s in SOILS:
        if s not in calc.index:
            continue
        sub = m[m["sample"] == s].set_index("size_fraction_um")["is_feo"]
        rows.append(dict(
            sample=s,
            I_sm_calculated=float(calc.loc[s, "I_sm_calc"]),
            is_feo_measured_20_45um=float(sub.get(FRACTION, np.nan)),
            is_feo_measured_lt250um=float(sub.get("<250", np.nan))))
    df = pd.DataFrame(rows)
    df["abs_diff_20_45um"] = (df["I_sm_calculated"] - df["is_feo_measured_20_45um"]).abs()
    df.to_csv(jc.TABLES / "step07_maturity_check.csv", index=False)

    print("\n  Eq. (3) calculated I_sm vs the MEASURED maturity it estimates")
    print("  (reported, not gated -- Table 1's own values are reproduced to 0.28):")
    print("    %-8s %12s %14s %14s" %
          ("sample", "I_sm calc", "Is/FeO 20-45um", "Is/FeO <250um"))
    for _, r in df.iterrows():
        print("    %-8s %12.1f %14.0f %14.0f"
              % (r["sample"], r["I_sm_calculated"],
                 r["is_feo_measured_20_45um"], r["is_feo_measured_lt250um"]))
    # Does the index at least rank the four soils the way the measurement does?
    a = df.sort_values("I_sm_calculated")["sample"].tolist()
    b = df.sort_values("is_feo_measured_20_45um")["sample"].tolist()
    print("    worst |calc - measured| = %.0f units" % df["abs_diff_20_45um"].max())
    print("    rank order by I_sm calc : %s" % " < ".join(a))
    print("    rank order by measured  : %s   -> %s"
          % (" < ".join(b), "same" if a == b else "DIFFERENT"))

    # Widen the test beyond the four inverted soils to every mare soil for which
    # both a Delta and a measured Is/FeO exist.  The three soils that DEFINE
    # d_sw are absent from step02's table, which is what keeps this from being
    # circular -- these are independent of the calibration.
    both = calc.reset_index()
    both["sample"] = both["sample"].astype(str)
    for frac in (FRACTION, "<250"):
        mm = m[m["size_fraction_um"] == frac].set_index("sample")["is_feo"]
        sub = both[both["sample"].isin(mm.index)].copy()
        if len(sub) < 3:
            continue
        sub["measured"] = sub["sample"].map(mm).astype(float)
        r = float(np.corrcoef(sub["Delta"], sub["measured"])[0, 1])
        try:
            from scipy.stats import pearsonr
            _, p = pearsonr(sub["Delta"], sub["measured"])
        except Exception:
            p = float("nan")
        print("    corr(Delta, measured Is/FeO) over %d independent mare soils"
              " [%s um] = %+.3f (p = %.2f)" % (len(sub), frac, r, p))
        rows_c.append(dict(size_fraction_um=frac, n_soils=int(len(sub)),
                           pearson_r=r, p_value=p))
    if rows_c:
        pd.DataFrame(rows_c).to_csv(jc.TABLES / "step07_maturity_correlation.csv",
                                    index=False)
        print("    (n is small -- this bounds the relation rather than refuting it)")
        print("  -> tables/step07_maturity_correlation.csv")
    print("  -> tables/step07_maturity_check.csv")
    return df


def model_table() -> pd.DataFrame:
    """Posterior abundances, crystalline-only and renormalised to 100 %."""
    z = np.load(jc.TABLES / "step05_posteriors.npz")
    names = list(jc.ENDMEMBERS["name"])
    idx = {n: names.index(n) for n in names}
    rho = jc.ENDMEMBERS["rho_paper"].to_numpy()

    out = []
    for s in SOILS:
        if s not in z.files:
            sys.exit("[%s] FATAL: no posterior for %s in step05_posteriors.npz" % (STAGE, s))
        # theta is [9 grain sizes | 9 mass fractions]; the abundances are the
        # SECOND half -- same layout as crystalline_groups() in step04.
        f = z[s][:, len(names):]
        cry = np.array([idx[n] for n in jc.CRYSTALLINE])
        m = f[:, cry]
        m = m / m.sum(axis=1, keepdims=True)          # renormalise to crystalline
        # volume fractions, for the modal-vs-mass sensitivity column
        v = m / rho[cry][None, :]
        v = v / v.sum(axis=1, keepdims=True)
        sub = {n: k for k, n in enumerate(jc.CRYSTALLINE)}
        for c, members in GROUPS.items():
            cols = [sub[n] for n in members]
            mm = 100.0 * m[:, cols].sum(axis=1)
            vv = 100.0 * v[:, cols].sum(axis=1)
            out.append(dict(sample=s, category=c,
                            our_median_wt_pct=float(np.median(mm)),
                            our_p5_wt_pct=float(np.percentile(mm, 2.5)),
                            our_p95_wt_pct=float(np.percentile(mm, 97.5)),
                            our_median_vol_pct=float(np.median(vv))))
    return pd.DataFrame(out)


def main() -> int:
    jc.verify_manifest(STAGE)
    jc.banner(STAGE, "inverted mineralogy vs LSCC lab values (Figs 4 / F1)")

    df = lab_table().merge(model_table(), on=["sample", "category"], how="inner")
    # The paper plots the published modal numbers, so the gated comparison is
    # against those; the volume-converted column is the sensitivity.
    df["lab_wt_pct"] = df["lab_pct"]
    df["abs_pct_diff"] = (df["our_median_wt_pct"] - df["lab_wt_pct"]).abs()
    df["abs_pct_diff_vol_basis"] = (df["our_median_vol_pct"] - df["lab_wt_pct"]).abs()
    prior = prior_baseline()
    df["prior_only_pct"] = df["category"].map(prior)
    df["abs_pct_diff_prior"] = (df["prior_only_pct"] - df["lab_wt_pct"]).abs()

    # The paper's own numbers come from the released sampler, so its result is
    # carried alongside the converged one.  Neither is hidden: the gated column
    # is the converged config, because the released ensemble is degenerate
    # (PAPER_NOTES.md D7) and its "median" is close to a single point rather
    # than an estimate -- but both are printed so the choice is visible.
    rel = pd.read_csv(jc.TABLES / "step05_lscc_abundances.csv")
    rel["sample"] = rel["sample"].astype(str)
    rel = rel[rel["config"] == "released"].copy()
    rel["category"] = rel["category"].replace({"pyroxene_total": "pyroxene"})
    rel = rel.set_index(["sample", "category"])["median_wt_pct"]
    key = pd.MultiIndex.from_frame(df[["sample", "category"]])
    df["released_median_wt_pct"] = rel.reindex(key).to_numpy()
    df["abs_pct_diff_released"] = (df["released_median_wt_pct"] - df["lab_wt_pct"]).abs()
    df["lab_inside_95ci"] = ((df["lab_wt_pct"] >= df["our_p5_wt_pct"]) &
                             (df["lab_wt_pct"] <= df["our_p95_wt_pct"]))
    df = df.sort_values(["sample", "category"]).reset_index(drop=True)

    # Independent-path check: step05 renormalised the same posteriors to the
    # same crystalline categories by its own code.  Recomputing here from the
    # raw ensemble and diffing is the repo's standard correctness test.
    cross = pd.read_csv(jc.TABLES / "step05_lscc_abundances.csv")
    cross["sample"] = cross["sample"].astype(str)
    cross = cross[cross["config"] == "converged"]
    cross["category"] = cross["category"].replace({"pyroxene_total": "pyroxene"})
    cross = cross.set_index(["sample", "category"])["median_wt_pct"]
    chk = df.set_index(["sample", "category"])
    common = chk.index.intersection(cross.index)
    delta = (chk.loc[common, "our_median_wt_pct"] - cross.loc[common]).abs()
    print("  cross-check vs step05_lscc_abundances (independent renormalisation):")
    print("    max |difference| over %d cells = %.2e %s"
          % (len(common), delta.max(), "OK" if delta.max() < 1e-6 else "*** MISMATCH ***"))
    if delta.max() >= 1e-6:
        jc.verify_manifest(STAGE)
        sys.exit("[%s] FATAL: recomputed abundances disagree with step05." % STAGE)
    print("")

    df.to_csv(jc.TABLES / "step05_vs_ground_truth.csv", index=False)

    print("  LSCC 20-45 um modal abundances, renormalised to the crystalline phases")
    print("  %-8s %-12s %8s %8s %8s   %s" %
          ("sample", "category", "lab %", "ours %", "|diff|", "lab in 95% CI"))
    print("  " + "-" * 68)
    for _, r in df.iterrows():
        print("  %-8s %-12s %8.2f %8.2f %8.2f   %s"
              % (r["sample"], r["category"], r["lab_wt_pct"],
                 r["our_median_wt_pct"], r["abs_pct_diff"],
                 "yes" if r["lab_inside_95ci"] else "no"))

    def worst(cat, exclude=()):
        sub = df[(df["category"] == cat) & (~df["sample"].isin(exclude))]
        i = sub["abs_pct_diff"].idxmax()
        return float(sub.loc[i, "abs_pct_diff"]), sub.loc[i, "sample"]

    ol, ol_s = worst("olivine")
    il, il_s = worst("ilmenite", exclude=("15071",))
    il_all, il_all_s = worst("ilmenite")
    pl, pl_s = worst("plagioclase")

    print("\n  Section 3 accuracy claims (worst case over the four soils,")
    print("  in percentage points of crystalline-renormalised abundance):")
    print("    olivine     < 7   ->  %5.2f  (%s)   %s" % (ol, ol_s, "PASS" if ol <= 7 else "FAIL"))
    print("    ilmenite    < 10  ->  %5.2f  (%s)   %s   [15071 excepted by the paper; "
          "including it: %.2f (%s)]" % (il, il_s, "PASS" if il <= 10 else "FAIL", il_all, il_all_s))
    print("    plagioclase < 30  ->  %5.2f  (%s)   %s" % (pl, pl_s, "PASS" if pl <= 30 else "FAIL"))
    print("\n  sensitivity: on a volume basis instead of mass the worst-case")
    print("  differences become olivine %.2f, ilmenite %.2f, plagioclase %.2f"
          % (df[df["category"] == "olivine"]["abs_pct_diff_vol_basis"].max(),
             df[(df["category"] == "ilmenite") & (df["sample"] != "15071")]["abs_pct_diff_vol_basis"].max(),
             df[df["category"] == "plagioclase"]["abs_pct_diff_vol_basis"].max()))

    # ---- the no-information control --------------------------------------
    def worst_prior(cat, exclude=()):
        sub = df[(df["category"] == cat) & (~df["sample"].isin(exclude))]
        return float(sub["abs_pct_diff_prior"].max())

    def worst_released(cat, exclude=()):
        sub = df[(df["category"] == cat) & (~df["sample"].isin(exclude))]
        return float(sub["abs_pct_diff_released"].max())

    p_ol = worst_prior("olivine")
    p_il = worst_prior("ilmenite", exclude=("15071",))
    p_pl = worst_prior("plagioclase")
    r_ol = worst_released("olivine")
    r_il = worst_released("ilmenite", exclude=("15071",))
    r_pl = worst_released("plagioclase")
    base = pd.DataFrame([
        dict(claim="olivine", bound=7.0, prior_only_pct=prior["olivine"],
             worst_err_inversion=ol, worst_err_released=r_ol,
             worst_err_prior_only=p_ol,
             released_satisfies_claim=bool(r_ol <= 7.0),
             prior_alone_satisfies_claim=bool(p_ol <= 7.0)),
        dict(claim="ilmenite (15071 excepted)", bound=10.0, prior_only_pct=prior["ilmenite"],
             worst_err_inversion=il, worst_err_released=r_il,
             worst_err_prior_only=p_il,
             released_satisfies_claim=bool(r_il <= 10.0),
             prior_alone_satisfies_claim=bool(p_il <= 10.0)),
        dict(claim="plagioclase", bound=30.0, prior_only_pct=prior["plagioclase"],
             worst_err_inversion=pl, worst_err_released=r_pl,
             worst_err_prior_only=p_pl,
             released_satisfies_claim=bool(r_pl <= 30.0),
             prior_alone_satisfies_claim=bool(p_pl <= 30.0)),
    ])
    base.to_csv(jc.TABLES / "step07_prior_baseline.csv", index=False)

    print("\n  Same three claims under the RELEASED sampler -- the configuration the")
    print("  paper's own numbers come from (degenerate ensemble, see D7):")
    print("    olivine     < 7   ->  %5.2f   %s" % (r_ol, "PASS" if r_ol <= 7 else "FAIL"))
    print("    ilmenite    < 10  ->  %5.2f   %s" % (r_il, "PASS" if r_il <= 10 else "FAIL"))
    print("    plagioclase < 30  ->  %5.2f   %s" % (r_pl, "PASS" if r_pl <= 30 else "FAIL"))

    print("\n  NO-INFORMATION CONTROL -- the same three claims scored against the")
    print("  Dirichlet prior alone, with the spectra never consulted:")
    print("    %-26s %9s %11s %11s   %s"
          % ("claim", "bound", "inversion", "prior only", "prior alone passes?"))
    for _, r in base.iterrows():
        print("    %-26s %9.1f %11.2f %11.2f   %s"
              % (r["claim"], r["bound"], r["worst_err_inversion"],
                 r["worst_err_prior_only"],
                 "YES" if r["prior_alone_satisfies_claim"] else "no"))
    npass = int(base["prior_alone_satisfies_claim"].sum())
    if npass:
        print("\n    %d of 3 claims are met by the prior mean alone, so meeting them is"
              % npass)
        print("    not by itself evidence that the inversion recovered mineralogy")
        print("    from the spectrum.  See PAPER_NOTES.md D12.")

    maturity_check()

    print("\n  -> tables/step05_vs_ground_truth.csv")
    print("  -> tables/step07_prior_baseline.csv")

    pd.DataFrame([
        dict(claim="olivine < 7 %", bound=7.0, observed=ol, worst_sample=ol_s),
        dict(claim="ilmenite < 10 % (15071 excepted)", bound=10.0, observed=il,
             worst_sample=il_s),
        dict(claim="ilmenite < 10 % (including 15071)", bound=10.0, observed=il_all,
             worst_sample=il_all_s),
        dict(claim="plagioclase < 30 %", bound=30.0, observed=pl, worst_sample=pl_s),
    ]).to_csv(jc.TABLES / "step07_accuracy_claims.csv", index=False)

    jc.verify_manifest(STAGE)
    # Scoring these three claims is audit_paper_targets.py's job -- they are
    # registered there as gated targets -- so this stage reports and does not
    # gate.  A genuine disagreement with the paper is a result, not a broken
    # stage, and must not block the figures or the audit that records it.
    ok = (ol <= 7) and (il <= 10) and (pl <= 30)
    print("\n  STEP 07 done (%s with the three published claims; scored by the audit)"
          % ("consistent" if ok else "NOT consistent"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
