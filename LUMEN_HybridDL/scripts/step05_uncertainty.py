"""
Step 05 -- uncertainty and identifiability (METHOD_NOTES.md D16, D17).

  coverage   for the primary model (A4), at 68 / 90 / 95 %, per spectrum and per soil,
             under LOSO and LOMO, of
               (i)   the raw ensemble interval mu +- z sigma;
               (ii)  CV+-style conformal with normalised scores |y - mu| / sigma, calibrated
                     on the out-of-fold predictions of the OTHER soils (LOSO) or missions
                     (LOMO);
               (iii) the same with absolute residuals (sigma = 1);
             plus mean interval widths.
  skill      1 - MAE / MAE(mean-of-others) with a bootstrap 95 % CI over soils, for every
             method and target; the identifiability gate (LOSO CI lower bound > 0).
  quantiles  conformal q per target from all LOSO scores, used for the M3 maps (step07).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import lumen_common as lc
import lumen_eval as le
import lumen_plot as lp
from step04_lumen import load_lab

STAGE = "step05"
PRIMARY = "A4_full"
LEVELS = [0.68, 0.90, 0.95]
SEED = 0


def conformal_q(scores: np.ndarray, level: float) -> float:
    s = np.sort(scores[np.isfinite(scores)])
    n = len(s)
    k = int(np.ceil(level * (n + 1)))
    return np.inf if k > n else float(s[k - 1])


def group_of(meta, protocol):
    return (meta["soil"] if protocol == "LOSO" else meta["mission"]).to_numpy()


def coverage_tables(pred, meta):
    rows, widths = [], []
    for protocol in ("LOSO", "LOMO"):
        g = pred[(pred["protocol"] == protocol) & (pred["method"] == PRIMARY)].set_index("row").sort_index()
        rows_idx = g.index.to_numpy()
        grp = group_of(meta, protocol)[rows_idx]
        soil = meta["soil"].to_numpy()[rows_idx]
        for t in le.TARGETS:
            y = meta[t].to_numpy(float)[rows_idx]
            mu = g["pred_" + t].to_numpy(float)
            sd = g["sd_" + t].to_numpy(float)
            ok = np.isfinite(y)
            s_norm = np.abs(y - mu) / sd
            s_abs = np.abs(y - mu)
            for level in LEVELS:
                z = norm.ppf(0.5 + level / 2)
                half = {"raw_ensemble": z * sd}
                hn, ha = np.full(len(y), np.nan), np.full(len(y), np.nan)
                for gg in np.unique(grp):
                    test = grp == gg
                    cal = ok & ~test
                    hn[test] = conformal_q(s_norm[cal], level) * sd[test]
                    ha[test] = conformal_q(s_abs[cal], level)
                half["conformal_normalised"] = hn
                half["conformal_absolute"] = ha
                for method, h in half.items():
                    lo, hi = mu - h, mu + h
                    if t in lc.COMP:
                        lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
                    cov = (y >= lo) & (y <= hi)
                    c = pd.DataFrame({"soil": soil[ok], "cov": cov[ok]})
                    rows.append({"protocol": protocol, "target": t, "method": method, "level": level,
                                 "n_spectra": int(ok.sum()),
                                 "coverage_spectrum": float(cov[ok].mean()),
                                 "coverage_soil": float(c.groupby("soil")["cov"].mean().mean()),
                                 "mean_width": float(np.mean(np.where(np.isfinite(hi - lo), hi - lo, np.nan)[ok])),
                                 "n_infinite": int((~np.isfinite(h[ok])).sum())})
    return pd.DataFrame(rows)


def skill_table(pred, meta, rng):
    rows = []
    for protocol in ("LOSO", "LOMO"):
        P = pred[pred["protocol"] == protocol]
        ref = P[P["method"] == le.REFERENCE]
        for method, g in P.groupby("method", sort=False):
            if method == le.REFERENCE:
                continue
            for t in le.TARGETS + ["COMP_mean"]:
                need = lc.COMP if t == "COMP_mean" else [t]
                if any("pred_" + c not in g or g["pred_" + c].isna().all() for c in need):
                    continue
                if t == "COMP_mean":
                    em, er = le.per_soil_comp_mean(g, meta), le.per_soil_comp_mean(ref, meta)
                else:
                    em, er = le.per_soil_errors(g, meta, t), le.per_soil_errors(ref, meta, t)
                sk, lo, hi = le.bootstrap_skill(em, er, rng)
                rows.append({"protocol": protocol, "method": method, "target": t,
                             "mae": em.mean(), "mae_reference": er.mean(),
                             "skill": sk, "skill_lo": lo, "skill_hi": hi,
                             "significant": bool(lo > 0)})
    return pd.DataFrame(rows)


def main() -> int:
    t0 = lc.banner(STAGE, "uncertainty (coverage) and identifiability (skill gate)")
    lc.verify_manifest(STAGE)
    lp.setup()
    rng = np.random.default_rng(SEED)
    X, meta = load_lab()
    p3 = pd.read_csv(lc.TABLES / "step03_predictions.csv")
    p4 = pd.read_csv(lc.TABLES / "step04_predictions.csv")
    pred = pd.concat([p3, p4], ignore_index=True)

    cov = coverage_tables(p4, meta)
    cov.to_csv(lc.TABLES / "step05_coverage.csv", index=False)
    summ = cov.groupby(["protocol", "method", "level"])[["coverage_spectrum", "coverage_soil"]].mean()
    summ.to_csv(lc.TABLES / "step05_coverage_summary.csv")
    print("  coverage of A4 intervals, averaged over the 13 targets:")
    print(summ.round(3).to_string())

    sk = skill_table(pred, meta, rng)
    sk.to_csv(lc.TABLES / "step05_skill.csv", index=False)
    gate = sk[(sk["method"] == PRIMARY)].pivot_table(index="target", columns="protocol",
                                                    values=["skill", "skill_lo", "skill_hi"])
    loso = sk[(sk["method"] == PRIMARY) & (sk["protocol"] == "LOSO")].set_index("target")
    lomo = sk[(sk["method"] == PRIMARY) & (sk["protocol"] == "LOMO")].set_index("target")
    g = pd.DataFrame({"skill_LOSO": loso["skill"], "ci_lo_LOSO": loso["skill_lo"],
                      "ci_hi_LOSO": loso["skill_hi"], "skill_LOMO": lomo["skill"],
                      "ci_lo_LOMO": lomo["skill_lo"], "ci_hi_LOMO": lomo["skill_hi"]})
    g["identifiable"] = g["ci_lo_LOSO"] > 0
    g = g.loc[[t for t in le.TARGETS + ["COMP_mean"] if t in g.index]]
    g.to_csv(lc.TABLES / "step05_identifiability_gate.csv")
    print("  identifiability gate (A4, skill vs fraction-matched mean-of-others):")
    print(g.round(3).to_string())

    # conformal quantiles for the M3 maps: all LOSO normalised scores of A4
    a4 = p4[(p4["protocol"] == "LOSO") & (p4["method"] == PRIMARY)].set_index("row").sort_index()
    qrows = []
    for t in le.TARGETS:
        y = meta[t].to_numpy(float)[a4.index.to_numpy()]
        s = np.abs(y - a4["pred_" + t].to_numpy()) / a4["sd_" + t].to_numpy()
        for level in LEVELS:
            qrows.append({"target": t, "level": level, "q_normalised": conformal_q(s, level)})
    pd.DataFrame(qrows).to_csv(lc.TABLES / "step05_conformal_quantiles.csv", index=False)

    # calibration figure
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, protocol in zip(axs, ("LOSO", "LOMO")):
        c = cov[cov["protocol"] == protocol].groupby(["method", "level"])["coverage_spectrum"]
        for (m, col) in zip(("raw_ensemble", "conformal_normalised", "conformal_absolute"), lp.C):
            v = c.mean().loc[m]
            lo, hi = c.min().loc[m], c.max().loc[m]
            ax.plot(v.index, v.values, "o-", color=col, label=m.replace("_", " "))
            ax.fill_between(v.index, lo.values, hi.values, color=col, alpha=0.12, lw=0)
        ax.plot([0.6, 1], [0.6, 1], color=lp.INK2, lw=0.8, ls="--")
        ax.set_xlabel("nominal coverage"); ax.set_title("%s (band: range over 13 targets)" % protocol)
    axs[0].set_ylabel("empirical coverage (spectra)"); axs[0].legend()
    lp.save(fig, lc.FIGURES / "step05_coverage.png")
    lc.done(STAGE, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
