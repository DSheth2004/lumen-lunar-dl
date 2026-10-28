"""
Step 15 -- E4: design-choice ablations of the input channels and the architecture
(METHOD_NOTES.md D34). Sub-commands, run in order:

  pretrain   2 members (seeds 100, 101) of each variant on the synthetic set, with the
             step04 schedule; workers run in parallel, one thread each, resumable
  finetune   A4 recipe under LOSO (19 folds) and LOMO (6 folds) for every variant and for
             the main CNN (members 0-1 of the shared pretrained weights), resumable
  collect    2-member ensembles -> predictions, metrics, paired Wilcoxon + Holm vs the main CNN

Variants: (a) log-R channel only, (b) continuum-removed channel only, (c) MLP, (d) 1-D ResNet.
The main CNN reference uses products/pretrained_member{0,1}.pt, copied from
checkpoints/pretrained/ (D31); it is never pretrained here.
Every fine-tune uses the step04 job seed of A4_full for its (protocol, fold, member), so the
real / mixup / replay draws are identical across networks and only the network differs.
The comparison is a robustness check, not model selection: no variant replaces A4.
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
import step04_lumen as s4

STAGE = "step15"
CONFIG = "A4_full"
MEMBERS = [0, 1]
PRETRAIN_SEEDS = {0: 100, 1: 101}                  # = step04 seeds of members 0 and 1
MAIN = "A4_main_cnn"
VARIANTS = ["E4a_logR_only", "E4b_CR_only", "E4c_MLP", "E4d_ResNet1D"]
NETS = [MAIN] + VARIANTS
PROTOCOLS = ("LOSO", "LOMO")
JOB_DIR = lc.PRODUCTS / "step15_jobs"


def make_net(name):
    import lumen_model as lm
    return {MAIN: lm.LumenNet,
            "E4a_logR_only": lambda: lm.LumenNetChannel(0),
            "E4b_CR_only": lambda: lm.LumenNetChannel(1),
            "E4c_MLP": lm.LumenMLP,
            "E4d_ResNet1D": lm.LumenResNet1D}[name]


def pretrained_path(net, k):
    return s4.member_path(k) if net == MAIN else lc.PRODUCTS / ("step15_pretrained_%s_member%d.pt" % (net, k))


def job_path(job):
    return JOB_DIR / ("%s_%s_%s_m%d.npz" % job)


def ensure_features():
    """products/synthetic_feat.npy, built exactly as in step04 pretrain if absent."""
    import lumen_model as lm
    if s4.FEAT_PATH.exists():
        return
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    lam = syn["lam_um"]
    scaler = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    n = len(syn["R"])
    tf = time.time()
    tmp = s4.FEAT_PATH.with_suffix(".tmp.npy")
    feat = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, 2, lam.size))
    for s in range(0, n, 50_000):
        feat[s:s + 50_000] = scaler(lc.spectral_features(syn["R"][s:s + 50_000], lam))
    feat.flush()
    del feat
    os.replace(tmp, s4.FEAT_PATH)
    print("  synthetic features -> %s (%.0f s)" % (s4.FEAT_PATH.name, time.time() - tf), flush=True)


# --------------------------------------------------------------------------
# pretrain
# --------------------------------------------------------------------------
def _pretrain_job(job):
    import torch
    import lumen_model as lm
    net, k = job
    torch.set_num_threads(1)
    syn = dict(np.load(lc.PRODUCTS / "synthetic.npz"))
    feat = np.load(s4.FEAT_PATH, mmap_mode="r")
    n = len(feat)
    tr, va = np.arange(0, n - s4.N_VAL), np.arange(n - s4.N_VAL, n)
    t = time.time()

    def log(msg):
        print("  [%s m%d] %s" % (net, k, msg.strip()), flush=True)
    model, hist = lm.pretrain(feat, syn, tr, va, seed=PRETRAIN_SEEDS[k], epochs=s4.PRETRAIN_EPOCHS,
                              comp_l1=lm.COMP_L1, log=log, make_net=make_net(net))
    torch.save(model.state_dict(), pretrained_path(net, k))
    for h in hist:
        h["network"], h["member"], h["seed"] = net, k, PRETRAIN_SEEDS[k]
    pd.DataFrame(hist).to_csv(lc.TABLES / ("step15_pretrain_history_%s_m%d.csv" % (net, k)), index=False)
    return net, k, time.time() - t


def cmd_pretrain(args):
    import lumen_model as lm
    t0 = lc.banner(STAGE, "pretrain %d variants x %d members on the synthetic set"
                   % (len(VARIANTS), len(MEMBERS)))
    lc.verify_manifest(STAGE)
    ensure_features()
    rows = [{"network": n, "n_params": lm.n_params(make_net(n)())} for n in NETS]
    pd.DataFrame(rows).to_csv(lc.TABLES / "step15_params.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    jobs = [(v, k) for k in MEMBERS for v in VARIANTS if not pretrained_path(v, k).exists()]
    print("  %d pretraining jobs to run, %d workers" % (len(jobs), args.workers), flush=True)
    with get_context(lc.POOL_START).Pool(args.workers) as pool:
        for net, k, dt in pool.imap_unordered(_pretrain_job, jobs):
            print("  pretrained %s member %d (%.0f s)" % (net, k, dt), flush=True)
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
    _W["feat"] = np.load(s4.FEAT_PATH, mmap_mode="r")
    sd = {k: syn[k] for k in ("comp", "chem", "logIsFeO", "logD", "npfe_wt")}
    n = len(sd["comp"])
    _W["syn_targets"] = lm.synthetic_targets(sd, np.arange(n))
    _W["pool"] = np.arange(0, n - s4.N_VAL)
    _W["X"], _W["meta"] = s4.load_lab()
    _W["Xf"] = _W["scaler"](lc.spectral_features(_W["X"], _W["lam"]))


def _run_job(job):
    import torch
    import lumen_model as lm
    protocol, fold, net, k = job
    out = job_path(job)
    if out.exists():
        return str(out), 0.0
    t = time.time()
    X, meta = _W["X"], _W["meta"]
    test = dict(s4.folds(meta, protocol))[fold]
    real = s4.real_dict(X, meta, np.flatnonzero(~test))
    state = torch.load(pretrained_path(net, k))
    model = lm.finetune(CONFIG, state, real, _W["feat"], _W["syn_targets"], _W["pool"], _W["scaler"],
                        seed=s4._job_seed(protocol, fold, CONFIG, k), lam=_W["lam"],
                        make_net=make_net(net))
    o = lm.member_outputs(model, _W["Xf"])
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, test_idx=np.flatnonzero(test), **o)
    os.replace(tmp, out)
    return str(out), time.time() - t


def cmd_finetune(args):
    t0 = lc.banner(STAGE, "fine-tune (%s): %s x LOSO + LOMO x members %s" % (CONFIG, ", ".join(args.nets), MEMBERS))
    lc.verify_manifest(STAGE)
    for net in args.nets:
        for k in MEMBERS:
            if not pretrained_path(net, k).exists():
                raise SystemExit("missing pretrained weights: %s" % pretrained_path(net, k))
    JOB_DIR.mkdir(exist_ok=True)
    _, meta = s4.load_lab()
    jobs = [(p, fold, net, k) for p in PROTOCOLS for fold, _ in s4.folds(meta, p)
            for net in args.nets for k in MEMBERS]
    todo = [j for j in jobs if not job_path(j).exists()]
    print("  %d jobs, %d to run, %d workers" % (len(jobs), len(todo), args.workers), flush=True)
    done_n, t_run = 0, time.time()
    with get_context(lc.POOL_START).Pool(args.workers, initializer=_worker_init) as pool:
        for path, dt in pool.imap_unordered(_run_job, todo):
            done_n += 1
            if done_n % 10 == 0 or done_n == len(todo):
                el = time.time() - t_run
                print("  %4d / %d jobs  (%.0f s elapsed, ~%.0f s left)"
                      % (done_n, len(todo), el, el / done_n * (len(todo) - done_n)), flush=True)
    lc.done(STAGE, t0)


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------
def assemble(meta):
    import lumen_model as lm
    rows = []
    for protocol in PROTOCOLS:
        for net in NETS:
            for fold, test in s4.folds(meta, protocol):
                idx = np.flatnonzero(test)
                outs = []
                for k in MEMBERS:
                    d = np.load(job_path((protocol, fold, net, k)))
                    assert np.array_equal(d["test_idx"], idx)
                    outs.append({kk: d[kk][idx] for kk in ("comp_mean", "comp_var", "scal_mean", "scal_var")})
                ens = lm.ensemble(outs)
                for j, i in enumerate(idx):
                    r = {"protocol": protocol, "method": net, "row": int(i)}
                    for c, nm in enumerate(lc.COMP):
                        r["pred_" + nm] = ens["comp_mean"][j, c]
                        r["sd_" + nm] = np.sqrt(ens["comp_var"][j, c])
                    for c, nm in enumerate(lm.SCALARS):
                        r["pred_" + nm] = ens["scal_mean"][j, c]
                        r["sd_" + nm] = np.sqrt(ens["scal_var"][j, c])
                    rows.append(r)
    return pd.DataFrame(rows)


def cmd_collect(args):
    import lumen_eval as le
    t0 = lc.banner(STAGE, "collect: 2-member ensembles, metrics, paired tests vs the main CNN")
    lc.verify_manifest(STAGE)
    _, meta = s4.load_lab()
    pred = assemble(meta)
    pred.to_csv(lc.TABLES / "step15_predictions.csv", index=False)
    summ = le.summarise(pred, meta)
    summ.to_csv(lc.TABLES / "step15_metrics.csv", index=False)

    tests, counts = [], []
    for protocol in PROTOCOLS:
        g_main = pred[(pred["protocol"] == protocol) & (pred["method"] == MAIN)]
        for v in VARIANTS:
            g = pred[(pred["protocol"] == protocol) & (pred["method"] == v)]
            rs = []
            for t in le.TARGETS:
                r = le.paired(le.per_soil_errors(g, meta, t), le.per_soil_errors(g_main, meta, t))
                rs.append({"protocol": protocol, "variant": v, "target": t, "n_soils": r["n"],
                           "mae_variant": r["mae_a"], "mae_main": r["mae_b"],
                           "rel_change": r["mae_a"] / r["mae_b"] - 1,
                           "n_soils_variant_better": r["n_a_better"], "n_ties": r["n_ties"],
                           "wilcoxon_p": r["wilcoxon_p"]})
            adj = le.holm([r["wilcoxon_p"] for r in rs])
            for r, a in zip(rs, adj):
                r["p_holm"] = a
                r["verdict"] = ("variant better" if a < 0.05 and r["mae_variant"] < r["mae_main"] else
                                "variant worse" if a < 0.05 else "n.s.")
            tests += rs
            d = pd.DataFrame(rs)
            counts.append({"protocol": protocol, "variant": v, "n_targets": len(d),
                           "n_targets_lower_mae": int((d["mae_variant"] < d["mae_main"]).sum()),
                           "n_holm_sig_better": int((d["verdict"] == "variant better").sum()),
                           "n_holm_sig_worse": int((d["verdict"] == "variant worse").sum()),
                           "median_rel_change": float(d["rel_change"].median())})
    tests = pd.DataFrame(tests)
    counts = pd.DataFrame(counts)
    tests.to_csv(lc.TABLES / "step15_tests.csv", index=False)
    counts.to_csv(lc.TABLES / "step15_summary.csv", index=False)

    show = summ[summ["level"] == "soil"].pivot_table(index=["protocol", "method"], columns="target",
                                                     values="mae")[le.TARGETS + ["COMP_mean"]]
    print(show.round(3).to_string())
    print(counts.round(3).to_string(index=False))
    hist = pd.concat([pd.read_csv(lc.TABLES / ("step15_pretrain_history_%s_m%d.csv" % (v, k)))
                      for v in VARIANTS for k in MEMBERS])
    hist.to_csv(lc.TABLES / "step15_pretrain_history.csv", index=False)
    lc.done(STAGE, t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["pretrain", "finetune", "collect"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--nets", nargs="+", default=NETS, choices=NETS,
                    help="finetune only: networks to run (default: all, including the main CNN)")
    args = ap.parse_args()
    {"pretrain": cmd_pretrain, "finetune": cmd_finetune, "collect": cmd_collect}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
