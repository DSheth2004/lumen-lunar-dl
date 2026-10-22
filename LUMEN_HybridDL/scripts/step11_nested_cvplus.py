"""
Step 11 -- exact CV+ (jackknife+-style) intervals for the primary model A4 (METHOD_NOTES.md
Addendum B, D24).

For every unordered pair of held-out groups {i, j} (soils for LOSO, missions for LOMO) the
5 A4 members are fine-tuned without both, so model M_ij has seen neither. For test group
j, calibration group i contributes residuals of M_ij on i's spectra and M_ij's prediction
for j's spectra; the CV+ interval (Barber et al. 2021) is

  [ q_lo { mu_ij(x_j) - R_i } ,  q_hi { mu_ij(x_j) + R_i } ]   over all i != j spectra,

with R absolute (|y - mu|) or normalised (|y - mu| / sigma, scaled back by sigma_ij(x_j)).
Sub-commands: `run` (resumable, parallel) then `score`.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import zlib
from multiprocessing import get_context

import numpy as np
import pandas as pd

import lumen_common as lc
import lumen_eval as le
import lumen_plot as lp
import step04_lumen as s4

STAGE = "step11"
JOB_DIR = lc.PRODUCTS / "step11_jobs"
LEVELS = [0.68, 0.90, 0.95]
CONFIG = "A4_full"


def groups(meta, protocol):
    col = "soil" if protocol == "LOSO" else "mission"
    return col, sorted(meta[col].unique())


def jobs_for(meta):
    out = []
    for protocol in ("LOSO", "LOMO"):
        _, g = groups(meta, protocol)
        for a, b in itertools.combinations(g, 2):
            for k in range(s4.N_MEMBERS):
                out.append((protocol, a, b, k))
    return out


def path(job):
    return JOB_DIR / ("%s_%s_%s_m%d.npz" % job)


def _run(job):
    import lumen_model as lm
    protocol, a, b, k = job
    p = path(job)
    if p.exists():
        return 0.0
    t = time.time()
    X, meta = s4._W["X"], s4._W["meta"]
    col, _ = groups(meta, protocol)
    out = meta[col].isin([a, b]).to_numpy()
    seed = zlib.crc32(("pair|%s|%s|%s|%d" % (protocol, a, b, k)).encode()) % (2 ** 31)
    model = lm.finetune(CONFIG, s4._W["states"][k], s4.real_dict(X, meta, np.flatnonzero(~out)),
                        s4._W["feat"], s4._W["syn_targets"], s4._W["pool"], s4._W["scaler"],
                        seed=seed, lam=s4._W["lam"])
    o = lm.member_outputs(model, s4._W["scaler"](lc.spectral_features(X[out], s4._W["lam"])))
    np.savez(p, rows=np.flatnonzero(out), **o)
    return time.time() - t


def cmd_run(args):
    t0 = lc.banner(STAGE, "fine-tune A4 without every pair of groups")
    lc.verify_manifest(STAGE)
    JOB_DIR.mkdir(exist_ok=True)
    _, meta = s4.load_lab()
    jobs = jobs_for(meta)
    todo = [j for j in jobs if not path(j).exists()]
    print("  %d jobs, %d to run, %d workers" % (len(jobs), len(todo), args.workers), flush=True)
    n, tr = 0, time.time()
    with get_context("fork").Pool(args.workers, initializer=s4._worker_init) as pool:
        for _ in pool.imap_unordered(_run, todo):
            n += 1
            if n % 50 == 0 or n == len(todo):
                el = time.time() - tr
                print("  %4d / %d jobs (%.0f s, ~%.0f s left)" % (n, len(todo), el, el / n * (len(todo) - n)), flush=True)
    lc.done(STAGE, t0)


def pair_ensemble(protocol, a, b):
    import lumen_model as lm
    outs, rows = [], None
    for k in range(s4.N_MEMBERS):
        key = (protocol, a, b, k) if a < b else (protocol, b, a, k)
        d = np.load(path(key))
        rows = d["rows"]
        outs.append({kk: d[kk] for kk in ("comp_mean", "comp_var", "scal_mean", "scal_var")})
    e = lm.ensemble(outs)
    mu = np.hstack([e["comp_mean"], e["scal_mean"]])
    sd = np.sqrt(np.hstack([e["comp_var"], e["scal_var"]]))
    return rows, mu, sd


def cmd_score(args):
    t0 = lc.banner(STAGE, "exact CV+ coverage")
    lc.verify_manifest(STAGE)
    X, meta = s4.load_lab()
    Y = meta[le.TARGETS].to_numpy(float)
    rows_out = []
    for protocol in ("LOSO", "LOMO"):
        col, G = groups(meta, protocol)
        gid = meta[col].to_numpy()
        cache = {}
        for a, b in itertools.combinations(G, 2):
            r, mu, sd = pair_ensemble(protocol, a, b)
            cache[(a, b)] = cache[(b, a)] = (r, mu, sd)
        res = {lv: {m: [] for m in ("cvplus_absolute", "cvplus_normalised")} for lv in LEVELS}
        for j in G:
            test_rows = np.flatnonzero(gid == j)
            for t_idx, t in enumerate(le.TARGETS):
                lo_s = {m: [] for m in ("a", "n")}
                hi_s = {m: [] for m in ("a", "n")}
                for i in G:
                    if i == j:
                        continue
                    r, mu, sd = cache[(i, j)]
                    ri = np.isin(r, np.flatnonzero(gid == i))
                    rj = np.isin(r, test_rows)
                    yi = Y[r[ri], t_idx]
                    ok = np.isfinite(yi)
                    Ra = np.abs(yi - mu[ri, t_idx])[ok]
                    Rn = (np.abs(yi - mu[ri, t_idx]) / sd[ri, t_idx])[ok]
                    mj, sj = mu[rj, t_idx], sd[rj, t_idx]
                    lo_s["a"].append(mj[:, None] - Ra[None, :])
                    hi_s["a"].append(mj[:, None] + Ra[None, :])
                    lo_s["n"].append(mj[:, None] - sj[:, None] * Rn[None, :])
                    hi_s["n"].append(mj[:, None] + sj[:, None] * Rn[None, :])
                yj = Y[test_rows, t_idx]
                for m, key in (("cvplus_absolute", "a"), ("cvplus_normalised", "n")):
                    L = np.hstack(lo_s[key])
                    H = np.hstack(hi_s[key])
                    n = L.shape[1]
                    for lv in LEVELS:
                        alpha = 1 - lv
                        kq = int(np.floor(alpha * (n + 1)))
                        lo = np.sort(L, axis=1)[:, kq - 1] if kq >= 1 else np.full(len(yj), -np.inf)
                        kh = int(np.ceil((1 - alpha) * (n + 1)))
                        hi = np.sort(H, axis=1)[:, kh - 1] if kh <= n else np.full(len(yj), np.inf)
                        for y_, l_, h_ in zip(yj, lo, hi):
                            if np.isfinite(y_):
                                res[lv][m].append((t, j, (l_ <= y_) & (y_ <= h_), h_ - l_))
        for lv in LEVELS:
            for m, rec in res[lv].items():
                df = pd.DataFrame(rec, columns=["target", "group", "cov", "width"])
                for t, g in df.groupby("target"):
                    rows_out.append({"protocol": protocol, "method": m, "level": lv, "target": t,
                                     "coverage_spectrum": g["cov"].mean(),
                                     "coverage_group": g.groupby("group")["cov"].mean().mean(),
                                     "mean_width": g["width"].replace([np.inf, -np.inf], np.nan).mean()})
    out = pd.DataFrame(rows_out)
    out.to_csv(lc.TABLES / "step11_cvplus_coverage.csv", index=False)
    summ = out.groupby(["protocol", "method", "level"])[["coverage_spectrum", "coverage_group"]].mean()
    summ.to_csv(lc.TABLES / "step11_cvplus_summary.csv")
    approx = pd.read_csv(lc.TABLES / "step05_coverage_summary.csv")
    print("  exact CV+ (mean over 13 targets):")
    print(summ.round(3).to_string())
    print("  approximate (step05) for comparison:")
    print(approx.round(3).to_string(index=False))
    lc.done(STAGE, t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "score"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[a.command](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
