"""
Shared scoring for every method (METHOD_NOTES.md D15-D16): per-soil errors, summary
metrics, skill against the mean-of-others reference with bootstrap CIs over soils, and
paired Wilcoxon tests with Holm correction.

Prediction tables are wide: one row per (protocol, method, spectrum `row`), with a
`pred_<target>` column per target (and `sd_<target>` where a method has uncertainty).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import lumen_common as lc

TARGETS = lc.COMP + lc.CHEM + [lc.MAT]
REFERENCE = "mean-of-others"


def per_soil_errors(pred: pd.DataFrame, meta: pd.DataFrame, target: str) -> pd.Series:
    """Mean absolute error over each soil's labelled fractions (index: soil)."""
    rows = pred["row"].to_numpy()
    y = meta[target].to_numpy(float)[rows]
    p = pred["pred_" + target].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    df = pd.DataFrame({"soil": meta["soil"].to_numpy()[rows][ok], "err": np.abs(p - y)[ok]})
    return df.groupby("soil")["err"].mean()


def per_soil_comp_mean(pred, meta) -> pd.Series:
    """Mean over the 7 composition parts of the per-soil MAE."""
    return pd.concat([per_soil_errors(pred, meta, t) for t in lc.COMP], axis=1).mean(axis=1)


def summarise(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, method), g in pred.groupby(["protocol", "method"], sort=False):
        for t in TARGETS + ["COMP_mean"]:
            if t == "COMP_mean":
                es = per_soil_comp_mean(g, meta)
                rows.append({"protocol": protocol, "method": method, "target": t, "level": "soil",
                             "n": len(es), "mae": es.mean()})
                continue
            if "pred_" + t not in g or g["pred_" + t].isna().all():
                continue
            es = per_soil_errors(g, meta, t)
            y = meta[t].to_numpy(float)[g["row"].to_numpy()]
            p = g["pred_" + t].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(p)
            y, p = y[ok], p[ok]
            rows.append({"protocol": protocol, "method": method, "target": t, "level": "soil",
                         "n": len(es), "mae": es.mean()})
            rows.append({"protocol": protocol, "method": method, "target": t, "level": "spectrum",
                         "n": len(y), "mae": np.abs(p - y).mean(),
                         "rmse": np.sqrt(np.mean((p - y) ** 2)),
                         "r2": 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2),
                         "spearman": spearmanr(p, y)[0], "bias": np.mean(p - y)})
    return pd.DataFrame(rows)


def bootstrap_skill(err_m: pd.Series, err_ref: pd.Series, rng, n_boot=2000):
    """skill = 1 - MAE_m / MAE_ref over soils, with a percentile bootstrap CI."""
    s = err_m.index.intersection(err_ref.index)
    a, b = err_m.loc[s].to_numpy(), err_ref.loc[s].to_numpy()
    skill = 1 - a.mean() / b.mean()
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    bs = 1 - a[idx].mean(1) / b[idx].mean(1)
    return skill, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def bootstrap_mae_ci(err: pd.Series, rng, n_boot=2000):
    e = err.to_numpy()
    idx = rng.integers(0, len(e), (n_boot, len(e)))
    m = e[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(err_a: pd.Series, err_b: pd.Series):
    """Wilcoxon signed-rank (two-sided) of per-soil errors a vs b."""
    s = err_a.index.intersection(err_b.index)
    a, b = err_a.loc[s].to_numpy(), err_b.loc[s].to_numpy()
    d = a - b
    p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return {"n": len(s), "mae_a": a.mean(), "mae_b": b.mean(), "n_a_better": int((a < b).sum()),
            "n_ties": int((a == b).sum()), "wilcoxon_p": p}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mean_of_others(meta: pd.DataFrame, train_idx, test_idx) -> pd.DataFrame:
    """Fraction-matched mean of the training spectra (D13)."""
    tr = meta.iloc[train_idx]
    out = {"row": np.asarray(test_idx)}
    for t in TARGETS:
        means = tr.groupby("fraction")[t].mean()
        out["pred_" + t] = meta.iloc[test_idx]["fraction"].map(means).to_numpy(float)
    return pd.DataFrame(out)
