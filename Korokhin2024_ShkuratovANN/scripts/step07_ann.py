"""
Step 07 -- the ANN mapping of Sec. 5: three training iterations, Tables 12-15,
Figs. 14-20.

  iteration 1: 20,000 spectra from uniform random parameters in the Table 11
               ranges; eta = 1.0 then k * sigma_Train (k = 0.01).
  iteration 2: the iteration-1 parameters of every scene pixel (152,000), each
               disturbed by less than +-0.5 % (D13: relative, uniform), turned back
               into spectra by the forward model; eta = 0.002.
  iteration 3: the same from iteration 2; eta = 0.010.
Stopping (Sec. 5.3): the epoch at which the ANN best reproduces the M3 spectra,
i.e. the minimum over epochs of the scene-mean sigma_Rel of forward(ANN(spectrum))
vs spectrum -- "Optimum" in Fig. 15b.

Budget choices (D11): iteration 1 evaluated every epoch up to 500 epochs (Fig. 15);
iterations 2-3 evaluated every 10 epochs, stopped 1,500 epochs after the best one
or at 10,000 (the paper's optima are 5,336 and 4,256). Every iteration starts from
fresh weights (D12). Five seeds run as parallel processes.

Usage:  python step07_ann.py            all seeds, then aggregate
        python step07_ann.py --fast     seed 0 only, short caps (smoke test)
        python step07_ann.py --seed S   one seed (used internally)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import ann
import korokhin_common as kc
import shkuratov as sk

STAGE = "step07"
ITER = {1: {"lr0": 1.0, "k": 0.01, "every": 1, "cap": 500, "patience": None},
        2: {"lr0": 0.002, "k": None, "every": 10, "cap": 10_000, "patience": 1_500},
        3: {"lr0": 0.010, "k": None, "every": 10, "cap": 10_000, "patience": 1_500}}
FAST = {1: 60, 2: 60, 3: 60}
JITTER = 0.005


def scene():
    cube = np.load(kc.PRODUCTS / "scene21_destriped.npy")           # (500, 21, 304)
    X = np.transpose(cube, (0, 2, 1)).reshape(-1, 21).astype(np.float64)
    return X, cube.shape[0], cube.shape[2]


def site_index(n_samples: int) -> dict:
    s = pd.read_csv(kc.PRODUCTS / "sites_rows.csv")
    return {int(r["site"]): int(r["row"]) * n_samples + int(r["sample"]) for _, r in s.iterrows()}


def table11_set(n: int, rng) -> np.ndarray:
    lo = np.array([kc.TABLE11_MIN[p] for p in kc.PARAMS])
    hi = np.array([kc.TABLE11_MAX[p] for p in kc.PARAMS])
    return lo + (hi - lo) * rng.random((n, len(kc.PARAMS)))


def sigma_rel_map(model, params, X) -> np.ndarray:
    out = np.empty(X.shape[0])
    for a in range(0, X.shape[0], 20_000):
        m = model.fast(params[a:a + 20_000])
        o = X[a:a + 20_000]
        out[a:a + 20_000] = 100 * np.sqrt(np.mean((m - o) ** 2, axis=1)) / o.mean(axis=1)
    return out


def train_iteration(it, seed, model, Ytrain, Xm3, fast=False, warm=None):
    """warm: the previous iteration's state -- fine-tune it (weights AND its input/output
    scaling, to which the weights are tied) instead of starting from fresh weights."""
    cfg = ITER[it]
    Xtrain = model.fast(Ytrain)
    ok = np.all(np.isfinite(Xtrain), axis=1)
    Xtrain, Ytrain = Xtrain[ok], Ytrain[ok]
    net = ann.MLP(seed * 100 + it)
    if warm is None:
        net.fit_scaling(Xtrain, Ytrain)
    else:
        net.load(warm)
    Xn, Tn = net.norm_x(Xtrain), net.norm_y(Ytrain)
    Xm3n = net.norm_x(Xm3)
    cap = FAST[it] if fast else cfg["cap"]
    every = 1 if fast else cfg["every"]
    lr = cfg["lr0"]
    hist, best = [], {"sigma_rel": np.inf}
    t_train = 0.0
    for ep in range(1, cap + 1):
        t = time.perf_counter()
        s_train = net.epoch(Xn, Tn, lr, kc.ANN_MOMENTUM)
        t_train += time.perf_counter() - t
        if cfg["k"] is not None:
            lr = cfg["k"] * s_train               # adaptive rate for the next epoch
        if ep % every == 0 or ep == 1:
            p = net.denorm_y(net.predict_norm(Xm3n))
            sr = sigma_rel_map(model, p, Xm3)
            m = float(np.nanmean(sr))
            hist.append({"epoch": ep, "sigma_train": s_train, "sigma_rel_mean": m, "lr": lr})
            if m < best["sigma_rel"]:
                best = {"sigma_rel": m, "epoch": ep, "state": net.state(), "t_train": t_train}
            if cfg["patience"] and not fast and ep - best["epoch"] >= cfg["patience"]:
                break
    net.load(best["state"])
    t = time.perf_counter()
    params = net.predict(Xm3)
    t_map = time.perf_counter() - t
    sr = sigma_rel_map(model, params, Xm3)
    info = {"iteration": it, "seed": seed, "pattern_size": int(Xtrain.shape[0]),
            "optimal_epoch": best["epoch"], "epochs_run": ep, "sigma_rel_mean": float(np.nanmean(sr)),
            "t_epoch_s": t_train / ep, "t_optim_min": best["t_train"] / 60.0, "t_map_s": t_map}
    return params, sr, pd.DataFrame(hist), info, best["state"]


def run_seed(seed: int, fast: bool, warm: bool = False) -> None:
    X, nr, ns = scene()
    model = sk.LunarModel(kc.M3_GRID_UM)
    rng = np.random.default_rng(seed)
    Y = table11_set(kc.N_TRAIN_ITER1, rng)
    tag = "fast" if fast else ("warm" if warm else "full")
    state = None
    for it in (1, 2, 3):
        if warm and it == 1:
            # iteration 1 is identical to the fresh-start run: reuse that seed's net
            base1 = kc.PRODUCTS / ("ann_full_seed%d_iter1" % seed)
            d = np.load(str(base1) + "_state.npz")
            state = {k: d[k] for k in d.files}
            params = np.load(str(base1) + "_params.npy").astype(np.float64)
            for suffix in ("_params.npy", "_sigma_rel.npy", "_state.npz"):
                src = kc.PRODUCTS / ("ann_full_seed%d_iter1%s" % (seed, suffix))
                (kc.PRODUCTS / ("ann_warm_seed%d_iter1%s" % (seed, suffix))).write_bytes(src.read_bytes())
            for f in ("step07_history_full_seed%d_iter1.csv", "step07_info_full_seed%d_iter1.json"):
                src = kc.TABLES / (f % seed)
                (kc.TABLES / (f.replace("full", "warm") % seed)).write_bytes(src.read_bytes())
            Y = params * (1.0 + rng.uniform(-JITTER, JITTER, params.shape))
            continue
        params, sr, hist, info, state = train_iteration(it, seed, model, Y, X, fast,
                                                        warm=state if warm else None)
        base = kc.PRODUCTS / ("ann_%s_seed%d_iter%d" % (tag, seed, it))
        np.save(str(base) + "_params.npy", params.astype(np.float32))
        np.save(str(base) + "_sigma_rel.npy", sr.astype(np.float32))
        np.savez(str(base) + "_state.npz", **state)
        hist.to_csv(kc.TABLES / ("step07_history_%s_seed%d_iter%d.csv" % (tag, seed, it)), index=False)
        (kc.TABLES / ("step07_info_%s_seed%d_iter%d.json" % (tag, seed, it))).write_text(json.dumps(info))
        print("  seed %d iteration %d: optimum epoch %d of %d, scene-mean sigma_Rel %.2f%%,"
              " %.3f s/epoch" % (seed, it, info["optimal_epoch"], info["epochs_run"],
                                 info["sigma_rel_mean"], info["t_epoch_s"]), flush=True)
        # next iteration's training set: this iteration's parameters, jittered by < +-0.5 %
        Y = params.astype(np.float64) * (1.0 + rng.uniform(-JITTER, JITTER, params.shape))


def torch_check() -> float:
    """numpy forward pass of ann.MLP vs an identical torch network."""
    import torch
    net = ann.MLP(123)
    X = np.random.default_rng(1).uniform(-1, 1, (1000, 21))
    ours = net.predict_norm(X)
    t = torch.nn.Sequential(torch.nn.Linear(21, 42), torch.nn.Sigmoid(), torch.nn.Linear(42, 12),
                            torch.nn.Sigmoid(), torch.nn.Linear(12, 8), torch.nn.Sigmoid()).double()
    with torch.no_grad():
        for layer, W, b in zip([t[0], t[2], t[4]], net.W, net.b):
            layer.weight.copy_(torch.from_numpy(W.T))
            layer.bias.copy_(torch.from_numpy(b))
        ref = t(torch.from_numpy(X)).numpy()
    return float(np.max(np.abs(ours - ref)))


def aggregate(tag: str, seeds) -> None:
    X, nr, ns = scene()
    idx = site_index(ns)
    high = np.load(kc.PRODUCTS / "highland_mask.npy").reshape(-1)
    infos, sites = [], []
    for s in seeds:
        for it in (1, 2, 3):
            f = kc.TABLES / ("step07_info_%s_seed%d_iter%d.json" % (tag, s, it))
            if not f.exists():
                continue
            info = json.loads(f.read_text())
            base = kc.PRODUCTS / ("ann_%s_seed%d_iter%d" % (tag, s, it))
            p = np.load(str(base) + "_params.npy").astype(float)
            sr = np.load(str(base) + "_sigma_rel.npy").astype(float)
            info["sigma_rel_mare"] = float(np.nanmean(sr[~high]))
            info["sigma_rel_highland"] = float(np.nanmean(sr[high]))
            infos.append(info)
            for site, k in idx.items():
                row = {"seed": s, "iteration": it, "site": site}
                row.update(dict(zip(kc.PARAMS, p[k])))
                row["sum_c"] = float(p[k, 2:].sum())
                row["mean_albedo"] = float(X[k].mean())
                row["sigma_rel"] = float(sr[k])
                row["sigma"] = row["sigma_rel"] * row["mean_albedo"] / 100.0
                sites.append(row)
    pd.DataFrame(infos).to_csv(kc.TABLES / ("step07_table12_%s.csv" % tag), index=False)
    pd.DataFrame(sites).to_csv(kc.TABLES / ("step07_sites_%s.csv" % tag), index=False)
    t = pd.DataFrame(infos)
    if len(t):
        print(t.groupby("iteration")[["optimal_epoch", "sigma_rel_mean", "sigma_rel_mare",
                                      "sigma_rel_highland", "t_epoch_s"]].agg(["mean", "std"]).round(3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--warm", action="store_true",
                    help="sensitivity D12: iterations 2-3 fine-tune the previous net (seeds 0-2)")
    a = ap.parse_args()
    if a.seed is not None:
        run_seed(a.seed, a.fast, a.warm)
        return 0
    kc.banner(STAGE, "ANN mapping, three iterations (Tables 12-15)")
    kc.verify_manifest(STAGE)
    d = torch_check()
    pd.DataFrame({"max_abs_diff_vs_torch": [d]}).to_csv(kc.TABLES / "step07_torch_check.csv", index=False)
    print("  numpy forward pass vs torch on identical weights: max |diff| = %.2e" % d)
    if d > 1e-10:
        sys.exit("[%s] FATAL: forward pass disagrees with torch" % STAGE)
    seeds = (0,) if a.fast else ((0, 1, 2) if a.warm else kc.SEEDS)
    tag = "fast" if a.fast else ("warm" if a.warm else "full")
    extra = (["--fast"] if a.fast else []) + (["--warm"] if a.warm else [])
    procs = [subprocess.Popen([sys.executable, "-u", __file__, "--seed", str(s)] + extra) for s in seeds]
    rc = [p.wait() for p in procs]
    if any(rc):
        sys.exit("[%s] FATAL: a seed process failed: %s" % (STAGE, rc))
    aggregate(tag, seeds)
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
