"""
Step 04 -- LUMEN: pretraining on the synthetic set, then per-fold fine-tuning with a
5-member ensemble under LOSO (19 folds) and LOMO (6 folds), plus the pre-registered
ablations (METHOD_NOTES.md D8-D12). Sub-commands, run in order:

  pretrain   5 members on the 400k synthetic spectra (synthetic data only)
  finetune   every (protocol, fold, configuration, member) job, in parallel, resumable;
             plus the final all-soil A4 ensemble used on M3 (step07)
  collect    ensemble the members -> tables/step04_predictions.csv, metrics, ablations

The held-out soil / mission is removed from fine-tuning, mixup and every statistic; the
input standardisation comes from the synthetic set only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

import lumen_common as lc

STAGE = "step04"
N_MEMBERS = 5
PRETRAIN_SEEDS = [100 + k for k in range(N_MEMBERS)]
PRETRAIN_EPOCHS = 15
N_VAL = 20_000
JOB_DIR = lc.PRODUCTS / "step04_jobs"
FEAT_PATH = lc.PRODUCTS / "synthetic_feat.npy"


def member_path(k):
    return lc.PRODUCTS / ("pretrained_member%d.pt" % k)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_lab():
    lab = np.load(lc.PRODUCTS / "lab_dataset.npz")
    meta = pd.read_csv(lc.TABLES / "step01_lab_dataset.csv", dtype={"soil": str})
    return lab["X"].astype(float), meta


def folds(meta, protocol):
    """[(fold name, boolean test mask over spectra)]"""
    if protocol == "LOSO":
        return [(s, (meta["soil"] == s).to_numpy()) for s in lc.ALL_SOILS]
    if protocol == "LOMO":
        return [(m, (meta["mission"] == m).to_numpy()) for m in sorted(meta["mission"].unique())]
    if protocol == "FINAL":
        return [("all", np.zeros(len(meta), bool))]
    raise ValueError(protocol)


def real_dict(X, meta, idx):
    return {"R": X[idx], "comp": meta.loc[idx, lc.COMP].to_numpy(float),
            "chem": meta.loc[idx, lc.CHEM].to_numpy(float),
            "logIsFeO": meta.loc[idx, lc.MAT].to_numpy(float),
            "d_nom": meta.loc[idx, "d_nominal_um"].to_numpy(float)}


# --------------------------------------------------------------------------
# pretrain
# --------------------------------------------------------------------------
def build_features(syn, scaler, lam):
    if not FEAT_PATH.exists():
        tf = time.time()
        feat = np.lib.format.open_memmap(FEAT_PATH, mode="w+", dtype=np.float32,
                                         shape=(len(syn["R"]), 2, lam.size))
        for s in range(0, len(syn["R"]), 50_000):
            feat[s:s + 50_000] = scaler(lc.spectral_features(syn["R"][s:s + 50_000], lam))
        feat.flush()
        del feat
        print("  synthetic features -> %s (%.0f s)" % (FEAT_PATH.name, time.time() - tf))


def cmd_pretrain(args):
    import torch
    import lumen_model as lm
    t0 = lc.banner(STAGE, "pretrain %d members on the synthetic set" % N_MEMBERS)
    lc.verify_manifest(STAGE)
    torch.set_num_threads(args.threads)
    syn = dict(np.load(lc.PRODUCTS / "synthetic.npz"))
    lam = syn["lam_um"]
    scaler = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    build_features(syn, scaler, lam)
    feat = np.load(FEAT_PATH)
    n = len(feat)
    tr, va = np.arange(0, n - N_VAL), np.arange(n - N_VAL, n)
    hist_all = []
    for k, seed in enumerate(PRETRAIN_SEEDS):
        if member_path(k).exists():
            print("  member %d exists -> skipped" % k)
            continue
        model, hist = lm.pretrain(feat, syn, tr, va, seed=seed, epochs=PRETRAIN_EPOCHS,
                                  comp_l1=lm.COMP_L1)
        torch.save(model.state_dict(), member_path(k))
        for h in hist:
            h["member"], h["seed"] = k, seed
        pd.DataFrame(hist).to_csv(lc.TABLES / ("step04_pretrain_history_m%d.csv" % k), index=False)
        hist_all += hist
    lc.done(STAGE, t0)


# --------------------------------------------------------------------------
# fine-tune
# --------------------------------------------------------------------------
_W = {}


def _worker_init():
    import torch
    import lumen_model as lm
    torch.set_num_threads(1)
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    _W["lam"] = syn["lam_um"]
    _W["scaler"] = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    _W["feat"] = np.load(FEAT_PATH, mmap_mode="r")
    sd = {k: syn[k] for k in ("comp", "chem", "logIsFeO", "logD", "npfe_wt")}
    n = len(sd["comp"])
    _W["syn_targets"] = lm.synthetic_targets(sd, np.arange(n))
    _W["pool"] = np.arange(0, n - N_VAL)
    _W["X"], _W["meta"] = load_lab()
    _W["states"] = {k: torch.load(member_path(k)) for k in range(N_MEMBERS)}


def _job_seed(protocol, fold, config, k):
    import zlib
    return zlib.crc32(("%s|%s|%s|%d" % (protocol, fold, config, k)).encode()) % (2 ** 31)


def _run_job(job):
    import torch
    import lumen_model as lm
    protocol, fold, config, k = job
    out_path = JOB_DIR / ("%s_%s_%s_m%d.npz" % (protocol, fold, config, k))
    if out_path.exists():
        return str(out_path), 0.0
    t = time.time()
    X, meta = _W["X"], _W["meta"]
    test = dict(folds(meta, protocol))[fold]
    train_idx = np.flatnonzero(~test)
    real = real_dict(X, meta, train_idx)
    model = lm.finetune(config, _W["states"][k], real, _W["feat"], _W["syn_targets"],
                        _W["pool"], _W["scaler"], seed=_job_seed(protocol, fold, config, k),
                        lam=_W["lam"])
    Xf = _W["scaler"](lc.spectral_features(X, _W["lam"]))
    o = lm.member_outputs(model, Xf)                    # predict every spectrum; keep test rows
    np.savez(out_path, test_idx=np.flatnonzero(test), **o)
    if protocol == "FINAL":
        torch.save(model.state_dict(), lc.PRODUCTS / ("final_%s_member%d.pt" % (config, k)))
    return str(out_path), time.time() - t


def _regen_job(job):
    """Re-run one stored LOMO A4 job, save its weights, report reproducibility."""
    import torch
    import lumen_model as lm
    protocol, fold, config, k = job
    X, meta = _W["X"], _W["meta"]
    test = dict(folds(meta, protocol))[fold]
    model = lm.finetune(config, _W["states"][k], real_dict(X, meta, np.flatnonzero(~test)), _W["feat"],
                        _W["syn_targets"], _W["pool"], _W["scaler"],
                        seed=_job_seed(protocol, fold, config, k), lam=_W["lam"])
    torch.save(model.state_dict(), lc.PRODUCTS / ("lomo_%s_%s_member%d.pt" % (config, fold, k)))
    o = lm.member_outputs(model, _W["scaler"](lc.spectral_features(X, _W["lam"])))
    d = np.load(JOB_DIR / ("%s_%s_%s_m%d.npz" % job))
    idx = d["test_idx"]
    return {"fold": fold, "member": k,
            "max_abs_diff_comp": float(np.abs(o["comp_mean"][idx] - d["comp_mean"][idx]).max()),
            "max_abs_diff_scalars": float(np.abs(o["scal_mean"][idx] - d["scal_mean"][idx]).max())}


def cmd_regen_lomo(args):
    """Weights of the A4 LOMO members, for the M3 landing-site test (Addendum A, D22)."""
    t0 = lc.banner(STAGE, "regenerate and save the A4 LOMO member weights")
    lc.verify_manifest(STAGE)
    _, meta = load_lab()
    jobs = [("LOMO", fold, "A4_full", k) for fold, _ in folds(meta, "LOMO") for k in range(N_MEMBERS)]
    with get_context(lc.POOL_START).Pool(args.workers, initializer=_worker_init) as pool:
        rows = pool.map(_regen_job, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(lc.TABLES / "step04_lomo_regeneration_check.csv", index=False)
    print(df.round(6).to_string(index=False))
    print("  max |difference| vs the stored step04 predictions: composition %.2e, scalars %.2e"
          % (df["max_abs_diff_comp"].max(), df["max_abs_diff_scalars"].max()))
    lc.done(STAGE, t0)


def _save_m3_job(job):
    """Fine-tune one A4 member (LOMO or FINAL), save its weights, return its test outputs."""
    import torch
    import lumen_model as lm
    protocol, fold, config, k = job
    X, meta = _W["X"], _W["meta"]
    test = dict(folds(meta, protocol))[fold]
    model = lm.finetune(config, _W["states"][k], real_dict(X, meta, np.flatnonzero(~test)), _W["feat"],
                        _W["syn_targets"], _W["pool"], _W["scaler"],
                        seed=_job_seed(protocol, fold, config, k), lam=_W["lam"])
    name = ("final_%s_member%d.pt" % (config, k)) if protocol == "FINAL" else \
        ("lomo_%s_%s_member%d.pt" % (config, fold, k))
    torch.save(model.state_dict(), lc.PRODUCTS / name)
    o = lm.member_outputs(model, _W["scaler"](lc.spectral_features(X, _W["lam"])))
    idx = np.flatnonzero(test)
    return job, idx, {kk: o[kk][idx] for kk in ("comp_mean", "comp_var", "scal_mean", "scal_var")}


def cmd_regen_m3_models(args):
    """The A4 LOMO members and the final all-soil A4 ensemble used on M3 (D22, D27), re-run from the
    pretrained members in products/ (Addendum D, D31). Checked against the stored LOMO predictions."""
    import lumen_model as lm
    t0 = lc.banner(STAGE, "regenerate the A4 LOMO and final members for M3")
    lc.verify_manifest(STAGE)
    missing = [member_path(k).name for k in range(N_MEMBERS) if not member_path(k).exists()]
    if missing:                                          # never pretrain here
        sys.exit("[%s] FATAL: pretrained members missing in products/: %s" % (STAGE, ", ".join(missing)))
    syn = dict(np.load(lc.PRODUCTS / "synthetic.npz"))
    build_features(syn, lm.FeatureScaler(syn["feat_mean"], syn["feat_std"]), syn["lam_um"])
    del syn
    _, meta = load_lab()
    jobs = [("LOMO", fold, "A4_full", k) for fold, _ in folds(meta, "LOMO") for k in range(N_MEMBERS)]
    jobs += [("FINAL", "all", "A4_full", k) for k in range(N_MEMBERS)]
    with get_context(lc.POOL_START).Pool(args.workers, initializer=_worker_init) as pool:
        res = pool.map(_save_m3_job, jobs)
    stored = pd.read_csv(lc.TABLES / "step04_predictions.csv")
    stored = stored[(stored["protocol"] == "LOMO") & (stored["method"] == "A4_full")].set_index("row")
    rows = []
    for fold, _ in folds(meta, "LOMO"):
        outs = [r for r in res if r[0][1] == fold and r[0][0] == "LOMO"]
        idx = outs[0][1]
        e = lm.ensemble([o for _, _, o in outs])
        for c, nm in enumerate(lc.COMP + lm.SCALARS):
            new = e["comp_mean"][:, c] if c < len(lc.COMP) else e["scal_mean"][:, c - len(lc.COMP)]
            d = new - stored.loc[idx, "pred_" + nm].to_numpy(float)
            rows.append({"fold": fold, "target": nm, "n_spectra": len(idx),
                         "max_abs_diff": float(np.abs(d).max()), "mean_abs_diff": float(np.abs(d).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(lc.TABLES / args.check_table, index=False)
    print(df.pivot_table(index="target", columns="fold", values="mean_abs_diff").round(4).to_string())
    print("  ensemble |regenerated - stored step04 LOMO A4| : max %.3g, mean %.3g"
          % (df["max_abs_diff"].max(), df["mean_abs_diff"].mean()))
    lc.done(STAGE, t0)


def cmd_finetune(args):
    import lumen_model as lm
    t0 = lc.banner(STAGE, "fine-tune: LOSO + LOMO x %d configurations x %d members"
                   % (len(lm.CONFIGS), N_MEMBERS))
    lc.verify_manifest(STAGE)
    JOB_DIR.mkdir(exist_ok=True)
    _, meta = load_lab()
    jobs = []
    for protocol in ("LOSO", "LOMO"):
        for fold, _ in folds(meta, protocol):
            for config in lm.CONFIGS:
                for k in range(N_MEMBERS):
                    jobs.append((protocol, fold, config, k))
    jobs += [("FINAL", "all", "A4_full", k) for k in range(N_MEMBERS)]
    todo = [j for j in jobs if not (JOB_DIR / ("%s_%s_%s_m%d.npz" % j)).exists()]
    print("  %d jobs, %d to run, %d workers" % (len(jobs), len(todo), args.workers))
    done_n, t_run = 0, time.time()
    with get_context(lc.POOL_START).Pool(args.workers, initializer=_worker_init) as pool:
        for path, dt in pool.imap_unordered(_run_job, todo):
            done_n += 1
            if done_n % 25 == 0 or done_n == len(todo):
                el = time.time() - t_run
                print("  %4d / %d jobs  (%.0f s elapsed, ~%.0f s left)"
                      % (done_n, len(todo), el, el / done_n * (len(todo) - done_n)), flush=True)
    lc.done(STAGE, t0)


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------
def zero_shot_outputs(X):
    import torch
    import lumen_model as lm
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    scaler = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    Xf = scaler(lc.spectral_features(X, syn["lam_um"]))
    outs = []
    for k in range(N_MEMBERS):
        m = lm.LumenNet()
        m.load_state_dict(torch.load(member_path(k)))
        outs.append(lm.member_outputs(m, Xf))
    return outs


def assemble(X, meta):
    """Wide prediction table: one row per (protocol, method, spectrum)."""
    import lumen_model as lm
    rows = []

    def emit(protocol, method, ens, idx):
        for j, i in enumerate(idx):
            r = {"protocol": protocol, "method": method, "row": int(i)}
            for c, nm in enumerate(lc.COMP):
                r["pred_" + nm] = ens["comp_mean"][j, c]
                r["sd_" + nm] = np.sqrt(ens["comp_var"][j, c])
            for c, nm in enumerate(lm.SCALARS):
                r["pred_" + nm] = ens["scal_mean"][j, c]
                r["sd_" + nm] = np.sqrt(ens["scal_var"][j, c])
            rows.append(r)

    zs = lm.ensemble(zero_shot_outputs(X))
    for protocol in ("LOSO", "LOMO"):
        emit(protocol, "A0_zero_shot", zs, np.arange(len(X)))
        for config in lm.CONFIGS:
            for fold, test in folds(meta, protocol):
                idx = np.flatnonzero(test)
                outs = []
                for k in range(N_MEMBERS):
                    d = np.load(JOB_DIR / ("%s_%s_%s_m%d.npz" % (protocol, fold, config, k)))
                    assert np.array_equal(d["test_idx"], idx)
                    outs.append({kk: d[kk][idx] for kk in ("comp_mean", "comp_var", "scal_mean", "scal_var")})
                emit(protocol, config, lm.ensemble(outs), idx)
    return pd.DataFrame(rows)


def cmd_collect(args):
    import lumen_eval as le
    t0 = lc.banner(STAGE, "collect: ensemble predictions, metrics, ablations")
    lc.verify_manifest(STAGE)
    X, meta = load_lab()
    pred = assemble(X, meta)
    pred.to_csv(lc.TABLES / "step04_predictions.csv", index=False)
    summ = le.summarise(pred, meta)
    summ.to_csv(lc.TABLES / "step04_metrics.csv", index=False)
    show = summ[summ["level"] == "soil"].pivot_table(index=["protocol", "method"], columns="target",
                                                     values="mae")[le.TARGETS]
    print(show.round(3).to_string())
    hist = pd.concat([pd.read_csv(lc.TABLES / ("step04_pretrain_history_m%d.csv" % k))
                      for k in range(N_MEMBERS)])
    hist.to_csv(lc.TABLES / "step04_pretrain_history.csv", index=False)
    lc.done(STAGE, t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["pretrain", "finetune", "collect", "regen-lomo", "regen-m3-models"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--check-table", default="step14_lab_model_regeneration_check.csv")
    args = ap.parse_args()
    {"pretrain": cmd_pretrain, "finetune": cmd_finetune, "collect": cmd_collect,
     "regen-lomo": cmd_regen_lomo, "regen-m3-models": cmd_regen_m3_models}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
