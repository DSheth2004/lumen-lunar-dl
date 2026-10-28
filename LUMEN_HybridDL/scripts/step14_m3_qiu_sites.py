"""
Step 14 -- orbital FeO on the 49 Qiu et al. (2025) sampling sites (METHOD_NOTES.md
Addendum D, D30 + D33: experiments E2, E3 and E6).

Site spectra are the D21 3 x 3-block medians of the M3 strips staged in data/m3_sites_qiu/
(primary) or the strips' centre pixels (sensitivity). Every method is refitted inside every
training fold, on the same site spectra:
  LUMEN-M3               zero-shot A4 ensemble (D22 rule: A4 LOMO members of an Apollo site's
                         mission, the final ensemble otherwise), Korokhin Eq. 2;
  LUMEN-M3 + recal       OLS y = a + b yhat on the training sites;
  LUMEN-M3-FT            D27 recipe (600 steps, 16 M3 pixels of the training sites + 16 lab
                         spectra per batch), one fine-tune per (fold, start model, member);
  LUMEN-embed + ridge    D28;
  FD-AE-M3 / FD-wavelet x {Lasso, SVR-lin, SVR-RBF, RF}   (D22; FD-AE-M3/RF = FD as published);
  E3 baselines           two M3 bands; Lucey theta on M3, refitted; Lucey et al. (2000) published
                         constants on the table's Clementine reflectances (in-sample, nothing
                         fitted); two Clementine bands, refitted; training-fold mean.
Protocols: P1 = 10-fold KFold(shuffle) over sites, seed 42 (+ 5 seeds as a sensitivity);
P2 = leave-one-landing-site-out (groups = mission).
Primary test: LUMEN-M3-FT vs FD-AE-M3/RF under P2, paired Wilcoxon over sites, Holm over the
comparators. E6: normalised conformal intervals for LUMEN-M3-FT under P2.

Sub-commands (run in order; each is resumable):
  comparators   FD and E3 baselines (no LUMEN weights needed)
  lumen         LUMEN-M3, recalibration, embed + ridge, LUMEN-M3-FT (needs the step04
                A4 LOMO and final members, `step04_lumen.py regen-m3-models`)
  report        tables, tests, conformal coverage, figures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
import zlib
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lumen_common as lc
import lumen_eval as le
import lumen_model as lm
import lumen_plot as lp
import step04_lumen as s4
import step10_m3_sites as s10
import step12_m3_finetune as s12
from step03_baselines import FD_GRIDS, dwt_a2d2
from step05_uncertainty import conformal_q

STAGE = "step14"
SITES = lc.DATA / "m3_sites_qiu"
QIU = lc.GT_DIR / "qiu2025_sampling_sites_feo.csv"
OUT = lc.PRODUCTS / "step14"
JOB_DIR = OUT / "ft_jobs"
P1_SEED = 42                                   # the FD reproduction's seed (config.yaml)
P1_SENS_SEEDS = [0, 1, 2, 3, 4]                # sensitivity (METHOD_NOTES Deviations)
N_FOLDS_P1 = 10
LAB_FOLDS = set(s10.MISSION_FOLD.values())     # A11 ... A17: missions with LSCC soils
LUCEY2000 = {"a": 17.427, "b": -7.565, "y0": 1.19, "x0": 0.08}   # Lucey, Blewett & Jolliff (2000)
FT = "LUMEN-M3-FT"
FD_PUB = "FD-AE-M3/RF"
LEVELS = (0.68, 0.90, 0.95)
SPECTRA = {"block3x3": False, "centre_pixel": True}   # primary, sensitivity
_G = {}


# --------------------------------------------------------------------------
# data and folds
# --------------------------------------------------------------------------
def start_model(mission):
    """D22: the A4 LOMO members of an Apollo site's mission, the final ensemble otherwise."""
    return mission if mission in LAB_FOLDS else "FINAL"


def load_sites(spectra):
    centre = SPECTRA[spectra]
    sites, specs, per_strip = s10.site_spectra(SITES, centre_only=centre)
    q = pd.read_csv(QIU).rename(columns={"site": "site_id"})
    sites = sites.merge(q, on="site_id", how="left", validate="1:1")
    assert sites["FeO_wt_pct"].notna().all(), "a staged site is missing from the Qiu table"
    assert np.allclose(sites["feo_mean"], sites["FeO_wt_pct"]), "staged truth differs from the Qiu table"
    excluded = sites[sites["n_strips_used"] == 0]["site_id"].tolist()
    sites = sites[sites["n_strips_used"] > 0].reset_index(drop=True)
    sites["start_model"] = sites["mission"].map(start_model)
    _, pix, pix_site = s12.pixel_spectra(SITES, centre_only=centre)
    keep = np.isin(pix_site, sites["site_id"])
    return {"spectra": spectra, "sites": sites, "excluded": excluded, "per_strip": per_strip,
            "sid": sites["site_id"].to_numpy(), "y": sites["feo_mean"].to_numpy(float),
            "R85": np.vstack([specs[s][0] for s in sites["site_id"]]),
            "pix": pix[keep], "pix_site": pix_site[keep]}


def runs():
    """(spectra, protocol, P1 seed) in reporting order. The centre-pixel sensitivity is run under
    the primary P1 seed and P2 only."""
    out = [("block3x3", "P2", -1), ("block3x3", "P1", P1_SEED)]
    out += [("block3x3", "P1", s) for s in P1_SENS_SEEDS]
    out += [("centre_pixel", "P2", -1), ("centre_pixel", "P1", P1_SEED)]
    return out


def run_name(protocol, seed):
    return protocol if protocol == "P2" else "P1s%d" % seed


def folds(D, protocol, seed):
    """[(fold name, train index, test index)] over sites."""
    n = len(D["sid"])
    if protocol == "P1":
        kf = KFold(n_splits=N_FOLDS_P1, shuffle=True, random_state=seed)
        return [("f%d" % i, tr, te) for i, (tr, te) in enumerate(kf.split(np.arange(n)))]
    mis = D["sites"]["mission"].to_numpy()
    return [(g, np.flatnonzero(mis != g), np.flatnonzero(mis == g)) for g in sorted(np.unique(mis))]


def m3_two_bands(R85):
    wl = lc.m3_wavelengths_nm()
    i750, i950 = np.argmin(np.abs(wl - 750)), np.argmin(np.abs(wl - 950))
    return R85[:, i950] / R85[:, i750], R85[:, i750]


def lucey2000_published(sites):
    r750, r950 = sites["clem_r750"].to_numpy(float), sites["clem_r950"].to_numpy(float)
    theta = -np.arctan((r950 / r750 - LUCEY2000["y0"]) / (r750 - LUCEY2000["x0"]))
    return LUCEY2000["a"] * theta + LUCEY2000["b"]


def ols(X, y, tr, te):
    X = np.column_stack([X]) if np.ndim(X) == 1 else np.asarray(X)
    return LinearRegression().fit(X[tr], y[tr]).predict(X[te])


# --------------------------------------------------------------------------
# comparators: FD (D22 settings) and the E3 baselines
# --------------------------------------------------------------------------
def cmd_comparators(args):
    t0 = lc.banner(STAGE, "comparators: Fernandez-Diaz variants and E3 baselines")
    lc.verify_manifest(STAGE)
    OUT.mkdir(parents=True, exist_ok=True)
    enc, hist = s10.fd_encoder(s10.SEED)                 # unsupervised, Shapley pixels, fold-independent
    hist.to_csv(lc.TABLES / "step14_fd_autoencoder_history.csv", index=False)
    for spectra in SPECTRA:
        path = OUT / ("comparators_%s.csv" % spectra)
        if path.exists():
            print("  %s exists -> skipped" % path.name)
            continue
        D = load_sites(spectra)
        y, sid = D["y"], D["sid"]
        W = dwt_a2d2(D["R85"][:, s10.FD_BANDS]).astype(np.float64)
        feats = {"FD-wavelet": W, "FD-AE-M3": enc(W.astype(np.float32))}
        ratio, r750 = m3_two_bands(D["R85"])
        theta = s10.lucey_theta(D["R85"], lc.m3_wavelengths_nm())
        S = D["sites"]
        c_ratio = S["clem_r950"].to_numpy(float) / S["clem_r750"].to_numpy(float)
        c750 = S["clem_r750"].to_numpy(float)
        lucey_pub = lucey2000_published(S)
        rows, choices = [], []
        for spectra_, protocol, seed in [r for r in runs() if r[0] == spectra]:
            tr0 = time.time()
            for fold, tr, te in folds(D, protocol, seed):
                pred = {}
                for fname, F in feats.items():
                    for rname, (est, grid) in FD_GRIDS.items():
                        name = "%s/%s" % (fname, rname)
                        groups = sid[tr]                     # inner GroupKFold by site (D22)
                        pipe = Pipeline([("x", StandardScaler()), ("m", est)])
                        gs = GridSearchCV(pipe, {"m__" + k: v for k, v in grid.items()},
                                          scoring="neg_mean_absolute_error",
                                          cv=GroupKFold(n_splits=min(5, len(np.unique(groups)))),
                                          n_jobs=args.workers)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            gs.fit(F[tr], y[tr], groups=groups)
                        pred[name] = gs.predict(F[te])
                        choices.append({"spectra": spectra, "protocol": protocol, "p1_seed": seed,
                                        "fold": fold, "method": name, "params": str(gs.best_params_),
                                        "inner_cv_mae": -gs.best_score_})
                pred["two M3 bands (R950/R750, R750)"] = ols(np.column_stack([ratio, r750]), y, tr, te)
                pred["Lucey theta (M3, refitted)"] = ols(theta, y, tr, te)
                pred["Lucey 2000 published (Clementine, in-sample)"] = lucey_pub[te]
                pred["two Clementine bands (refitted)"] = ols(np.column_stack([c_ratio, c750]), y, tr, te)
                pred["training-fold mean"] = np.full(len(te), y[tr].mean())
                for m, p in pred.items():
                    rows.append(pd.DataFrame({"spectra": spectra, "protocol": protocol, "p1_seed": seed,
                                              "fold": fold, "site_id": sid[te], "method": m, "pred": p}))
            print("  %s %s: comparators done (%.0f s)" % (spectra, run_name(protocol, seed),
                                                         time.time() - tr0), flush=True)
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
        pd.DataFrame(choices).to_csv(OUT / ("fd_hyperparameters_%s.csv" % spectra), index=False)
    lc.done(STAGE, t0)


# --------------------------------------------------------------------------
# LUMEN
# --------------------------------------------------------------------------
def _init():
    torch.set_num_threads(1)
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    s12._G["scaler"] = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    s12._G["lam"] = syn["lam_um"]
    _G["X"], _G["meta"] = s4.load_lab()
    _G["D"] = {sp: load_sites(sp) for sp in SPECTRA}


def ft_jobs():
    """One fine-tune per (spectra, protocol, P1 seed, fold, start model of a test site, member)."""
    jobs = []
    for spectra, protocol, seed in runs():
        D = _G["D"][spectra]
        st = D["sites"]["start_model"].to_numpy()
        for fold, tr, te in folds(D, protocol, seed):
            for start in sorted(set(st[te])):
                jobs += [(spectra, protocol, seed, fold, start, k) for k in range(s4.N_MEMBERS)]
    return jobs


def job_path(job):
    return JOB_DIR / ("%s_%s_%s_%s_m%d.json" % (job[0], run_name(job[1], job[2]), job[3], job[4], job[5]))


def _ft_job(job):
    spectra, protocol, seed, fold, start, k = job
    path = job_path(job)
    if path.exists():
        return 0.0
    t = time.time()
    D = _G["D"][spectra]
    fd = {f: (tr, te) for f, tr, te in folds(D, protocol, seed)}
    tr, te = fd[fold]
    te = te[D["sites"]["start_model"].to_numpy()[te] == start]
    y_site = dict(zip(D["sid"], D["y"]))
    on = np.isin(D["pix_site"], D["sid"][tr])            # M3 pixels of the training sites only
    y_pix = np.array([y_site[s] for s in D["pix_site"][on]])
    real = s4.real_dict(_G["X"], _G["meta"], s12.lab_training_idx(_G["meta"], start))
    # the seed omits the spectra, so the centre-pixel sensitivity differs only in its inputs
    jseed = zlib.crc32(("m3ft-qiu|%s|%d|%s|%s|%d" % (protocol, seed, fold, start, k)).encode()) % (2 ** 31)
    model = s12.finetune_m3(s12.state_path(start, k), D["pix"][on], y_pix, real, jseed)
    Xf = s12._G["scaler"](lc.spectral_features(s12.to_lab(D["R85"][te]), s12._G["lam"]))
    o = lm.member_outputs(model, Xf)
    path.write_text(json.dumps({"job": list(job), "seed": jseed, "n_train_pixels": int(on.sum()),
                                "site_id": D["sid"][te].tolist(), "FeO": o["scal_mean"][:, 0].tolist(),
                                "var": o["scal_var"][:, 0].tolist()}))
    return time.time() - t


def cmd_lumen(args):
    t0 = lc.banner(STAGE, "LUMEN on the Qiu sites: zero-shot, recalibration, embed + ridge, M3-FT")
    lc.verify_manifest(STAGE)
    need = [s12.state_path(f, k) for f in sorted(LAB_FOLDS) + ["FINAL"] for k in range(s4.N_MEMBERS)]
    miss = [p.name for p in need if not p.exists()]
    if miss:
        sys.exit("[%s] FATAL: missing A4 members (run step04_lumen.py regen-m3-models): %s"
                 % (STAGE, ", ".join(miss[:5])))
    OUT.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    _init()
    jobs = ft_jobs()
    todo = [j for j in jobs if not job_path(j).exists()]
    print("  LUMEN-M3-FT: %d fine-tunes, %d to run, %d workers" % (len(jobs), len(todo), args.workers),
          flush=True)
    if todo:
        n, tr0 = 0, time.time()
        with get_context(lc.POOL_START).Pool(args.workers, initializer=_init) as pool:
            for _ in pool.imap_unordered(_ft_job, todo):
                n += 1
                if n % 20 == 0 or n == len(todo):
                    el = time.time() - tr0
                    print("  %5d / %d  (%.0f s elapsed, ~%.0f s left)"
                          % (n, len(todo), el, el / n * (len(todo) - n)), flush=True)
    torch.set_num_threads(args.workers)
    models, scaler, lam = s10.lumen_models()
    rows, out_rows, member_rows = [], [], []
    for spectra in SPECTRA:
        D = _G["D"][spectra]
        S, sid, y = D["sites"], D["sid"], D["y"]
        st = S["start_model"].to_numpy()
        # zero-shot (fold-independent), with the other outputs reported without truth
        zs, zs_sd = np.empty(len(sid)), np.empty(len(sid))
        for i, s in enumerate(sid):
            o = s10.lumen_predict(models[st[i]], scaler, lam, D["R85"][i], s10.EQ2)
            zs[i], zs_sd[i] = o["FeO"][0], o["sd_FeO"][0]
            out_rows.append({"spectra": spectra, "site_id": s, "mission": S.loc[i, "mission"],
                             "model": st[i], **{k: float(v[0]) for k, v in o.items()}})
        # embeddings of every pixel and site spectrum, per start model (D28)
        E_pix, E_site = {}, {}
        for m in sorted(set(st)):
            E_pix[m] = s12.embeddings(models[m], D["pix"])
            E_site[m] = s12.embeddings(models[m], D["R85"])
        for spectra_, protocol, seed in [r for r in runs() if r[0] == spectra]:
            for fold, tr, te in folds(D, protocol, seed):
                recal = ols(zs, y, tr, te)
                on = np.isin(D["pix_site"], sid[tr])
                y_pix = pd.Series(y, index=sid).loc[D["pix_site"][on]].to_numpy(float)
                emb = np.empty(len(te))
                for m in sorted(set(st[te])):
                    r, alpha = s12.fit_embed_ridge(E_pix[m][on], y_pix, D["pix_site"][on])
                    sel = st[te] == m
                    emb[sel] = r.predict(E_site[m][te[sel]])
                    member_rows.append({"spectra": spectra, "protocol": protocol, "p1_seed": seed,
                                        "fold": fold, "start_model": m, "method": "LUMEN-embed + ridge",
                                        "alpha": alpha})
                # fine-tuned ensemble: mean of member means; variance = mean var + var of means
                ft = {}
                for m in sorted(set(st[te])):
                    J = [json.loads(job_path((spectra, protocol, seed, fold, m, k)).read_text())
                         for k in range(s4.N_MEMBERS)]
                    M, V = np.array([j["FeO"] for j in J]), np.array([j["var"] for j in J])
                    for c, s in enumerate(J[0]["site_id"]):
                        ft[s] = (M[:, c].mean(), np.sqrt(V[:, c].mean() + M[:, c].var()), M[:, c].std())
                        for k in range(s4.N_MEMBERS):
                            member_rows.append({"spectra": spectra, "protocol": protocol, "p1_seed": seed,
                                                "fold": fold, "start_model": m, "method": FT,
                                                "site_id": s, "member": k, "FeO": M[k, c], "var": V[k, c]})
                for j, i in enumerate(te):
                    base = {"spectra": spectra, "protocol": protocol, "p1_seed": seed, "fold": fold,
                            "site_id": sid[i]}
                    rows += [{**base, "method": "LUMEN-M3", "pred": zs[i], "sd": zs_sd[i]},
                             {**base, "method": "LUMEN-M3 + recal", "pred": recal[j]},
                             {**base, "method": "LUMEN-embed + ridge", "pred": emb[j]},
                             {**base, "method": FT, "pred": ft[sid[i]][0], "sd": ft[sid[i]][1],
                              "sd_members": ft[sid[i]][2]}]
    pd.DataFrame(rows).to_csv(OUT / "lumen.csv", index=False)
    pd.DataFrame(out_rows).to_csv(lc.TABLES / "step14_lumen_site_outputs.csv", index=False)
    pd.DataFrame(member_rows).to_csv(OUT / "lumen_members.csv", index=False)
    lc.done(STAGE, t0)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def metrics(g, truth_all):
    e = g["pred"].to_numpy() - g["truth"].to_numpy()
    t = g["truth"].to_numpy()
    return pd.Series({"n_sites": len(e), "mae": np.abs(e).mean(), "rmse": np.sqrt(np.mean(e ** 2)),
                      "r2": 1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2), "bias": e.mean(),
                      "max_abs_err": np.abs(e).max()})


def conformal(P, levels=LEVELS, sd_col="sd"):
    """E6: normalised scores from the out-of-fold predictions of the other landing sites (P2)."""
    y, mu, sd = P["truth"].to_numpy(), P["pred"].to_numpy(), P[sd_col].to_numpy()
    g = P["mission"].to_numpy()
    score = np.abs(y - mu) / sd
    rows = []
    for level in levels:
        cov, half, cov_raw = [], [], []
        for i in range(len(y)):
            q = conformal_q(score[g != g[i]], level)
            cov.append(abs(y[i] - mu[i]) <= q * sd[i])
            half.append(q * sd[i])
            cov_raw.append(abs(y[i] - mu[i]) <= norm.ppf(0.5 + level / 2) * sd[i])
        rows.append({"level": level, "n_sites": len(y), "coverage_conformal": np.mean(cov),
                     "mean_half_width_conformal": np.mean(half), "median_half_width_conformal":
                     np.median(half), "coverage_raw_gaussian": np.mean(cov_raw)})
    return pd.DataFrame(rows)


def cmd_report(args):
    t0 = lc.banner(STAGE, "report: metrics, tests, conformal coverage")
    lc.verify_manifest(STAGE)
    lp.setup()
    D = {sp: load_sites(sp) for sp in SPECTRA}
    site_rows = []
    for sp, d in D.items():
        S = d["sites"]
        for _, r in S.iterrows():
            site_rows.append({"spectra": sp, "site_id": r["site_id"], "mission": r["mission"],
                              "lat": r["lat"], "lon": r["lon"], "FeO": r["feo_mean"],
                              "n_strips_found": r["n_strips_found"], "n_strips_used": r["n_strips_used"],
                              "start_model": r["start_model"], "included": True})
        site_rows += [{"spectra": sp, "site_id": s, "included": False} for s in d["excluded"]]
        d["per_strip"].assign(spectra=sp).to_csv(lc.TABLES / ("step14_strip_quality_%s.csv" % sp), index=False)
        print("  %s: %d sites used, excluded (no usable strip): %s"
              % (sp, len(S), ", ".join(d["excluded"]) or "none"))
    pd.DataFrame(site_rows).to_csv(lc.TABLES / "step14_sites.csv", index=False)
    P = pd.concat([pd.read_csv(OUT / ("comparators_%s.csv" % sp)) for sp in SPECTRA]
                  + [pd.read_csv(OUT / "lumen.csv")], ignore_index=True)
    P = P.merge(pd.concat([d["sites"][["site_id", "mission", "feo_mean"]].assign(spectra=sp)
                           for sp, d in D.items()]).rename(columns={"feo_mean": "truth"}),
                on=["spectra", "site_id"], how="left", validate="m:1")
    P["run"] = [run_name(p, s) for p, s in zip(P["protocol"], P["p1_seed"])]
    key = ["spectra", "protocol", "p1_seed", "run", "method"]
    # every site predicted exactly once per run and method
    cnt = P.groupby(key + ["site_id"]).size()
    assert (cnt == 1).all(), "a site is predicted more than once in a run"
    P.to_csv(lc.TABLES / "step14_predictions.csv", index=False)
    pd.concat([pd.read_csv(OUT / ("fd_hyperparameters_%s.csv" % sp)) for sp in SPECTRA]).to_csv(
        lc.TABLES / "step14_fd_hyperparameters.csv", index=False)
    pd.read_csv(OUT / "lumen_members.csv").to_csv(lc.TABLES / "step14_lumen_members.csv", index=False)

    M = P.groupby(key).apply(lambda g: metrics(g, None), include_groups=False).reset_index()
    M["uses_landing_site_data"] = M["method"] != "LUMEN-M3"
    M["in_sample"] = M["method"].str.contains("in-sample")
    M = M.sort_values(["spectra", "protocol", "p1_seed", "mae"])
    M.to_csv(lc.TABLES / "step14_metrics.csv", index=False)
    L = P.assign(err=P["pred"] - P["truth"], abs_err=(P["pred"] - P["truth"]).abs()).groupby(
        key + ["mission"]).agg(n_sites=("site_id", "size"), truth_mean=("truth", "mean"),
                               pred_mean=("pred", "mean"), mean_error=("err", "mean"),
                               mae=("abs_err", "mean")).reset_index()
    L.to_csv(lc.TABLES / "step14_per_landing_site.csv", index=False)

    # tests: LUMEN-M3-FT against every comparator (not its own variants), Holm over comparators
    trows = []
    for (sp, prot, seed, run), g in P.groupby(["spectra", "protocol", "p1_seed", "run"]):
        W = g.pivot(index="site_id", columns="method", values="pred")
        truth = g.groupby("site_id")["truth"].first().loc[W.index]
        a = (W[FT] - truth).abs()
        sub = []
        for m in [m for m in W.columns if not m.startswith("LUMEN")]:
            b = (W[m] - truth).abs()
            sub.append({"spectra": sp, "protocol": prot, "p1_seed": seed, "run": run, "versus": m,
                        "n_sites": len(a), "mae_ft": a.mean(), "mae_other": b.mean(),
                        "n_ft_better": int((a < b).sum()), "n_ties": int((a == b).sum()),
                        "wilcoxon_p": wilcoxon(a, b).pvalue if np.any(a != b) else 1.0})
        sub = pd.DataFrame(sub)
        sub["holm_p"] = le.holm(sub["wilcoxon_p"])
        trows.append(sub)
    T = pd.concat(trows, ignore_index=True)
    T.to_csv(lc.TABLES / "step14_tests.csv", index=False)
    prim = T[(T["spectra"] == "block3x3") & (T["run"] == "P2") & (T["versus"] == FD_PUB)]
    prim.to_csv(lc.TABLES / "step14_primary_test.csv", index=False)

    # P1 seed sensitivity
    p1 = M[(M["spectra"] == "block3x3") & (M["protocol"] == "P1")]
    sens = p1.groupby("method").agg(mae_seed42=("mae", lambda s: s[p1.loc[s.index, "p1_seed"] == P1_SEED].iloc[0]),
                                    mae_mean_6seeds=("mae", "mean"), mae_sd_6seeds=("mae", "std"),
                                    mae_min=("mae", "min"), mae_max=("mae", "max"),
                                    r2_mean_6seeds=("r2", "mean")).reset_index().sort_values("mae_mean_6seeds")
    ranks = p1.assign(rank=p1.groupby("p1_seed")["mae"].rank()).groupby("method")["rank"].mean()
    sens["mean_rank"] = sens["method"].map(ranks)
    sens.to_csv(lc.TABLES / "step14_p1_seed_sensitivity.csv", index=False)

    # E6 conformal coverage (P2, primary spectra; centre pixel as sensitivity)
    crows = []
    for sp in SPECTRA:
        g = P[(P["spectra"] == sp) & (P["run"] == "P2") & (P["method"] == FT)]
        for sd_col, lab in (("sd", "predictive sd (mean member var + var of member means)"),
                            ("sd_members", "sd of member means")):
            crows.append(conformal(g, sd_col=sd_col).assign(spectra=sp, sigma=lab))
    C = pd.concat(crows, ignore_index=True)
    C.to_csv(lc.TABLES / "step14_conformal_coverage.csv", index=False)

    # ---- print the summary ---------------------------------------------------------
    pd.set_option("display.width", 200)
    cols = ["method", "n_sites", "mae", "rmse", "r2", "bias"]
    for sp, run in (("block3x3", "P1s%d" % P1_SEED), ("block3x3", "P2"),
                    ("centre_pixel", "P1s%d" % P1_SEED), ("centre_pixel", "P2")):
        print("\n  %s, %s" % (sp, run))
        print(M[(M["spectra"] == sp) & (M["run"] == run)][cols].round(3).to_string(index=False))
    print("\n  P1 seed sensitivity (block3x3, seeds 42 + %s)" % P1_SENS_SEEDS)
    print(sens.round(3).to_string(index=False))
    for sp, run in (("block3x3", "P2"), ("block3x3", "P1s%d" % P1_SEED)):
        print("\n  tests %s %s: %s vs comparators" % (sp, run, FT))
        print(T[(T["spectra"] == sp) & (T["run"] == run)].drop(columns=["spectra", "protocol", "p1_seed", "run"])
              .round(4).to_string(index=False))
    print("\n  PRIMARY (P2, block3x3): %s vs %s" % (FT, FD_PUB))
    print(prim.round(4).to_string(index=False))
    print("\n  E6 conformal coverage (P2)")
    print(C.round(3).to_string(index=False))
    for sp, run in (("block3x3", "P2"), ("block3x3", "P1s%d" % P1_SEED)):
        m = M[(M["spectra"] == sp) & (M["run"] == run)].set_index("method")["mae"]
        fd = m[m.index.str.startswith("FD")]
        e3 = m[[x for x in m.index if not x.startswith(("FD", "LUMEN"))]]
        print("  %s %s: %s MAE %.3f; best FD %s %.3f; FD as published %.3f; best E3 %s %.3f -> "
              "beats every FD variant: %s, beats FD as published: %s, beats every E3 baseline: %s"
              % (sp, run, FT, m[FT], fd.idxmin(), fd.min(), m[FD_PUB], e3.idxmin(), e3.min(),
                 m[FT] < fd.min(), m[FT] < m[FD_PUB], m[FT] < e3.min()))
    figures(P, M)
    lc.done(STAGE, t0)


def figures(P, M):
    import matplotlib.pyplot as plt
    for run in ("P2", "P1s%d" % P1_SEED):
        Mr = M[(M["spectra"] == "block3x3") & (M["run"] == run)].set_index("method")
        e3 = [m for m in Mr.index if not m.startswith(("FD", "LUMEN"))]
        show = list(dict.fromkeys([FT, FD_PUB, Mr.loc[[m for m in Mr.index if m.startswith("FD")], "mae"].idxmin(),
                                   Mr.loc[e3, "mae"].idxmin(), "Lucey 2000 published (Clementine, in-sample)"]))
        g = P[(P["spectra"] == "block3x3") & (P["run"] == run)]
        fig, axs = plt.subplots(1, len(show), figsize=(3.1 * len(show), 3.3), sharex=True, sharey=True)
        hi = max(g["truth"].max(), g[g["method"].isin(show)]["pred"].max()) + 1
        for ax, m in zip(axs, show):
            gm = g[g["method"] == m]
            for (mis, gg), mk in zip(gm.groupby("mission"), "osD^v<>p*hXP"):
                ax.scatter(gg["truth"], gg["pred"], s=16, marker=mk, label=mis,
                           color=lp.C[0] if m.startswith("LUMEN") else lp.C[6] if m.startswith("FD") else lp.C[3])
            ax.plot([0, hi], [0, hi], color=lp.INK2, lw=0.8)
            ax.set_title("%s\nMAE %.2f, R² %.2f" % (m, Mr.loc[m, "mae"], Mr.loc[m, "r2"]), fontsize=7)
            ax.set_xlabel("returned-sample FeO (wt %)")
        axs[0].set_ylabel("predicted (wt %)")
        axs[-1].legend(fontsize=5, loc="lower right", ncol=2)
        lp.save(fig, lc.FIGURES / ("step14_qiu_sites_%s.png" % run))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["comparators", "lumen", "report"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    {"comparators": cmd_comparators, "lumen": cmd_lumen, "report": cmd_report}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
