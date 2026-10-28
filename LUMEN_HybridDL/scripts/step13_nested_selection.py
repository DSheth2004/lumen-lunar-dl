"""
Step 13 -- E1: choosing between A2 and A4 inside the cross-validation (METHOD_NOTES.md
Addendum D, D32), with the regenerated checkpoints of D31.

For every outer fold (LOSO 19, LOMO 6) the outer training soils are split by
GroupKFold(5) (groups: soils, in sorted soil-id order). A2_pretrain_real and A4_full are
fine-tuned on each inner training split with members k = 0, 1; their 2-member ensemble
predicts the inner held-out soils. Selection score per candidate: the mean over the 13
targets of inner MAE_t / inner MAE_t(fraction-matched mean of the other inner-training
soils), where inner MAE_t is the mean over the outer training soils of the per-soil error,
each soil predicted by the inner fold that held it out. The lower score is selected; an
exact tie selects A4. Both A2 and A4 are re-run on every outer fold with all 5 members
(step04 job seeds), so the selected configuration's outer prediction is the 5-member
ensemble of that re-run.

  run     every inner and outer fine-tune job, in parallel, resumable
  score   selection, outer metrics, paired tests vs the step03 baselines,
          regeneration check vs the stored step04 predictions

Job seeds: inner crc32("E1|<protocol>|<fold>|inner<j>|<config>|<k>") % 2**31 (j = 0..4),
outer step04_lumen._job_seed(protocol, fold, config, k); both depend only on the job tuple.
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import lumen_common as lc
import lumen_eval as le
import step04_lumen as s4

STAGE = "step13"
CANDIDATES = ["A2_pretrain_real", "A4_full"]
TIE_CONFIG = "A4_full"                      # pre-registered primary wins an exact tie
INNER_SPLITS = 5
INNER_MEMBERS = [0, 1]
SELECTED = "LUMEN_selected"
PROTOCOLS = ("LOSO", "LOMO")
BASELINES = ["RF", "PLS", "ridge", "mean-of-others"]
OUT_KEYS = ("comp_mean", "comp_var", "scal_mean", "scal_var")


# --------------------------------------------------------------------------
# folds and jobs
# --------------------------------------------------------------------------
def outer_folds(meta, only=None):
    """[(protocol, fold, test mask)], optionally restricted to {"LOSO:10084", ...}."""
    out = []
    for protocol in PROTOCOLS:
        for fold, test in s4.folds(meta, protocol):
            if only is None or "%s:%s" % (protocol, fold) in only:
                out.append((protocol, fold, test))
    return out


def inner_splits(meta, test):
    """[(inner train rows, inner held-out rows)] over the outer training soils."""
    train_idx = np.flatnonzero(~test)
    soils = meta["soil"].to_numpy()[train_idx]
    order = np.argsort(soils, kind="stable")          # sorted soil-id order
    train_idx, soils = train_idx[order], soils[order]
    return [(train_idx[tr], train_idx[te])
            for tr, te in GroupKFold(n_splits=INNER_SPLITS).split(train_idx, groups=soils)]


def inner_seed(protocol, fold, j, config, k):
    return zlib.crc32(("E1|%s|%s|inner%d|%s|%d" % (protocol, fold, j, config, k)).encode()) % (2 ** 31)


def all_jobs(meta, only=None):
    jobs = []
    for protocol, fold, _ in outer_folds(meta, only):
        for j in range(INNER_SPLITS):
            for config in CANDIDATES:
                for k in INNER_MEMBERS:
                    jobs.append(("inner", protocol, fold, j, config, k))
    for protocol, fold, _ in outer_folds(meta, only):
        for config in CANDIDATES:
            for k in range(s4.N_MEMBERS):
                jobs.append(("outer", protocol, fold, config, k))
    return jobs


def job_path(job_dir, job):
    if job[0] == "inner":
        _, protocol, fold, j, config, k = job
        return job_dir / ("inner_%s_%s_inner%d_%s_m%d.npz" % (protocol, fold, j, config, k))
    _, protocol, fold, config, k = job
    return job_dir / ("outer_%s_%s_%s_m%d.npz" % (protocol, fold, config, k))


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
_CFG = {}


def ensure_features():
    """Synthetic input features, as built by step04 pretrain (never pretrains members)."""
    import lumen_model as lm
    if s4.FEAT_PATH.exists():
        return
    tf = time.time()
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    lam, R = syn["lam_um"], syn["R"]
    scaler = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    feat = np.lib.format.open_memmap(s4.FEAT_PATH, mode="w+", dtype=np.float32,
                                     shape=(len(R), 2, lam.size))
    for s in range(0, len(R), 50_000):
        feat[s:s + 50_000] = scaler(lc.spectral_features(R[s:s + 50_000], lam))
    feat.flush()
    del feat
    print("  synthetic features -> %s (%.0f s)" % (s4.FEAT_PATH.name, time.time() - tf), flush=True)


def _worker_init(member_dir, job_dir):
    # step04's initializer, with the member weights read from member_dir
    s4.member_path = lambda k: Path(member_dir) / ("pretrained_member%d.pt" % k)
    s4._worker_init()
    _CFG["job_dir"] = Path(job_dir)


def _run(job):
    import lumen_model as lm
    out_path = job_path(_CFG["job_dir"], job)
    if out_path.exists():
        return 0.0
    t = time.time()
    W = s4._W
    X, meta = W["X"], W["meta"]
    if job[0] == "inner":
        _, protocol, fold, j, config, k = job
        test = dict(s4.folds(meta, protocol))[fold]
        train_idx, test_idx = inner_splits(meta, test)[j]
        seed = inner_seed(protocol, fold, j, config, k)
    else:
        _, protocol, fold, config, k = job
        test = dict(s4.folds(meta, protocol))[fold]
        train_idx, test_idx = np.flatnonzero(~test), np.flatnonzero(test)
        seed = s4._job_seed(protocol, fold, config, k)
    model = lm.finetune(config, W["states"][k], s4.real_dict(X, meta, train_idx), W["feat"],
                        W["syn_targets"], W["pool"], W["scaler"], seed=seed, lam=W["lam"])
    o = lm.member_outputs(model, W["scaler"](lc.spectral_features(X[test_idx], W["lam"])))
    np.savez(out_path, test_idx=test_idx, train_idx=train_idx, seed=seed, **o)
    return time.time() - t


def cmd_run(args):
    t0 = lc.banner(STAGE, "E1: inner selection A2 vs A4 + outer re-runs")
    lc.verify_manifest(STAGE)
    for k in range(s4.N_MEMBERS):
        p = Path(args.member_dir) / ("pretrained_member%d.pt" % k)
        if not p.exists():
            raise SystemExit("missing %s -- copy the shared checkpoints (D31) first" % p)
    ensure_features()
    args.job_dir.mkdir(parents=True, exist_ok=True)
    _, meta = s4.load_lab()
    jobs = all_jobs(meta, args.only)
    todo = [j for j in jobs if not job_path(args.job_dir, j).exists()]
    print("  %d jobs, %d to run, %d workers, members from %s"
          % (len(jobs), len(todo), args.workers, args.member_dir), flush=True)
    n, tr = 0, time.time()
    with get_context(lc.POOL_START).Pool(args.workers, initializer=_worker_init,
                                         initargs=(str(args.member_dir), str(args.job_dir))) as pool:
        for _ in pool.imap_unordered(_run, todo):
            n += 1
            if n % 10 == 0 or n == len(todo):
                el = time.time() - tr
                print("  %4d / %d jobs (%.0f s elapsed, ~%.0f s left)"
                      % (n, len(todo), el, el / n * (len(todo) - n)), flush=True)
    lc.done(STAGE, t0)


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
def frame(ens, idx, protocol, method):
    """Wide prediction rows in the step04 layout."""
    import lumen_model as lm
    df = pd.DataFrame({"protocol": protocol, "method": method, "row": np.asarray(idx, int)})
    for c, nm in enumerate(lc.COMP):
        df["pred_" + nm] = ens["comp_mean"][:, c]
        df["sd_" + nm] = np.sqrt(ens["comp_var"][:, c])
    for c, nm in enumerate(lm.SCALARS):
        df["pred_" + nm] = ens["scal_mean"][:, c]
        df["sd_" + nm] = np.sqrt(ens["scal_var"][:, c])
    return df


def load_ensemble(job_dir, jobs):
    import lumen_model as lm
    outs, idx = [], None
    for job in jobs:
        d = np.load(job_path(job_dir, job))
        assert idx is None or np.array_equal(d["test_idx"], idx)
        idx = d["test_idx"]
        outs.append({kk: d[kk] for kk in OUT_KEYS})
    return lm.ensemble(outs), idx


def inner_scores(job_dir, meta, protocol, fold, test):
    """Per (config, target): inner MAE, inner mean-of-others MAE, ratio."""
    splits = inner_splits(meta, test)
    moo = pd.concat([le.mean_of_others(meta, tr, te) for tr, te in splits], ignore_index=True)
    rows = []
    for config in CANDIDATES:
        pred = []
        for j, (_, te) in enumerate(splits):
            ens, idx = load_ensemble(job_dir, [("inner", protocol, fold, j, config, k)
                                               for k in INNER_MEMBERS])
            assert np.array_equal(idx, te)
            pred.append(frame(ens, idx, protocol, config))
        pred = pd.concat(pred, ignore_index=True)
        for t in le.TARGETS:
            m = le.per_soil_errors(pred, meta, t).mean()
            r = le.per_soil_errors(moo, meta, t).mean()
            rows.append({"protocol": protocol, "fold": fold, "config": config, "target": t,
                         "inner_mae": m, "inner_mae_mean_of_others": r, "ratio": m / r})
    return rows


def tests(pred, meta, comparisons):
    rows = []
    for protocol, a_name, b_name in comparisons:
        a = pred[(pred["protocol"] == protocol) & (pred["method"] == a_name)]
        b = pred[(pred["protocol"] == protocol) & (pred["method"] == b_name)]
        if a.empty or b.empty:
            continue
        a = a[a["row"].isin(b["row"])]
        b = b[b["row"].isin(a["row"])]
        block = []
        for t in le.TARGETS + ["COMP_mean"]:
            if t == "COMP_mean":
                ea, eb = le.per_soil_comp_mean(a, meta), le.per_soil_comp_mean(b, meta)
            else:
                ea, eb = le.per_soil_errors(a, meta, t), le.per_soil_errors(b, meta, t)
            block.append({"protocol": protocol, "method": a_name, "baseline": b_name, "target": t,
                          **le.paired(ea, eb)})
        df = pd.DataFrame(block)
        core = df["target"] != "COMP_mean"
        df.loc[core, "holm_p"] = le.holm(df.loc[core, "wilcoxon_p"])
        df["method_better"] = df["mae_a"] < df["mae_b"]
        df["significant"] = df["holm_p"] < 0.05
        df["primary"] = (protocol == "LOSO") & (a_name == SELECTED) & (b_name == "RF")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def regeneration_check(pred, meta):
    stored = pd.read_csv(lc.TABLES / "step04_predictions.csv")
    rows = []
    for protocol in PROTOCOLS:
        for config in CANDIDATES:
            r = pred[(pred["protocol"] == protocol) & (pred["method"] == config)]
            s = stored[(stored["protocol"] == protocol) & (stored["method"] == config)]
            if r.empty:
                continue
            s = s.set_index("row").loc[r["row"]].reset_index()
            for t in le.TARGETS:
                d = np.abs(r["pred_" + t].to_numpy() - s["pred_" + t].to_numpy())
                er, es = le.per_soil_errors(r, meta, t), le.per_soil_errors(s, meta, t)
                rows.append({"protocol": protocol, "config": config, "target": t, "n_spectra": len(d),
                             "max_abs_diff": np.nanmax(d), "mean_abs_diff": np.nanmean(d),
                             "soil_mae_stored": es.mean(), "soil_mae_regenerated": er.mean(),
                             "soil_mae_diff": er.mean() - es.mean()})
    return pd.DataFrame(rows)


def cmd_score(args):
    t0 = lc.banner(STAGE, "E1: selection, outer metrics, tests, regeneration check")
    lc.verify_manifest(STAGE)
    X, meta = s4.load_lab()
    out = args.table_dir
    out.mkdir(parents=True, exist_ok=True)
    fold_rows, score_rows, sel_rows, preds = [], [], [], []
    for protocol, fold, test in outer_folds(meta, args.only):
        for j, (_, te) in enumerate(inner_splits(meta, test)):
            fold_rows.append({"protocol": protocol, "fold": fold, "inner": j,
                              "held_out_soils": " ".join(sorted(meta.loc[te, "soil"].unique()))})
        sc = inner_scores(args.job_dir, meta, protocol, fold, test)
        score_rows += sc
        score = pd.DataFrame(sc).groupby("config")["ratio"].mean()
        a2, a4 = score["A2_pretrain_real"], score["A4_full"]
        selected = "A2_pretrain_real" if a2 < a4 else TIE_CONFIG
        sel_rows.append({"protocol": protocol, "fold": fold, "score_A2_pretrain_real": a2,
                         "score_A4_full": a4, "selected": selected, "tie": bool(a2 == a4)})
        for config in CANDIDATES:
            ens, idx = load_ensemble(args.job_dir, [("outer", protocol, fold, config, k)
                                                    for k in range(s4.N_MEMBERS)])
            assert np.array_equal(idx, np.flatnonzero(test))
            preds.append(frame(ens, idx, protocol, config))
            if config == selected:
                preds.append(frame(ens, idx, protocol, SELECTED))
    pred = pd.concat(preds, ignore_index=True)
    sel = pd.DataFrame(sel_rows)
    counts = (sel.groupby(["protocol", "selected"]).size().rename("n_selected").reset_index()
              .merge(sel.groupby("protocol").size().rename("n_folds").reset_index(), on="protocol"))

    base = pd.read_csv(lc.TABLES / "step03_predictions.csv")
    base = base[base["method"].isin(BASELINES)]
    comparisons = [(p, SELECTED, b) for p in PROTOCOLS for b in BASELINES]
    comparisons += [(p, "A4_full", "A2_pretrain_real") for p in PROTOCOLS]
    tst = tests(pd.concat([pred, base], ignore_index=True), meta, comparisons)

    pd.DataFrame(fold_rows).to_csv(out / "step13_inner_folds.csv", index=False)
    pd.DataFrame(score_rows).to_csv(out / "step13_inner_scores.csv", index=False)
    sel.to_csv(out / "step13_selection.csv", index=False)
    counts.to_csv(out / "step13_selection_counts.csv", index=False)
    pred.to_csv(out / "step13_predictions.csv", index=False)
    le.summarise(pred, meta).to_csv(out / "step13_metrics.csv", index=False)
    tst.to_csv(out / "step13_tests.csv", index=False)
    regen = regeneration_check(pred, meta)
    regen.to_csv(out / ("stepD_regeneration_check.csv"), index=False)

    print("  selection per outer fold:")
    print(sel.round(4).to_string(index=False))
    print(counts.to_string(index=False))
    show = ["protocol", "baseline", "target", "n", "mae_a", "mae_b", "n_a_better", "wilcoxon_p", "holm_p",
            "significant"]
    for (protocol, a_name, b_name), g in tst.groupby(["protocol", "method", "baseline"], sort=False):
        print("  %s  %s vs %s:" % (protocol, a_name, b_name))
        print(g[show].round(4).to_string(index=False))
    print("  regeneration check (max |regenerated - stored step04|): %.3g"
          % regen["max_abs_diff"].max() if len(regen) else "  regeneration check: no rows")
    lc.done(STAGE, t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "score"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--member-dir", type=Path, default=lc.PRODUCTS,
                    help="directory holding pretrained_member{0..4}.pt")
    ap.add_argument("--job-dir", type=Path, default=lc.PRODUCTS / "step13_jobs")
    ap.add_argument("--table-dir", type=Path, default=lc.TABLES)
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to outer folds, e.g. LOSO:10084 (smoke tests)")
    a = ap.parse_args()
    if a.only is not None:
        a.only = set(a.only)
    {"run": cmd_run, "score": cmd_score}[a.command](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
