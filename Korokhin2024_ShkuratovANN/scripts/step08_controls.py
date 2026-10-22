"""
Step 08 -- controls the paper does not run.

The paper's only evidence that the ANN retrieves composition correctly is the
spectral-fit error sigma_Rel, which measures RECONSTRUCTION. These controls ask
whether the parameters themselves are recovered, and how much of each map comes
from the data rather than from the training ranges chosen in Table 11.

  C1  synthetic recovery. 5,000 fresh parameter sets drawn from the Table 11 ranges,
      turned into spectra, 1 % relative Gaussian noise added (about the paper's
      final sigma_Rel), fed to every trained net. Per parameter: R^2 and the RMSE
      relative to that of always answering the range midpoint (skill < 1 = better
      than guessing).
  C2a no-information net. Iteration 1 retrained with the spectra shuffled against the
      parameters: it can only learn the training distribution. Its map medians are
      what the Table 11 ranges alone produce.
  C2b range shift. Iteration 1 retrained with OPX and OLV ranges moved from -2...5 to
      0...10 and gray from 20...40 to 30...50. For each, (change in map median) /
      (change in range midpoint): ~1 means the map follows the range, ~0 the data.
  C2c inheritance. Iterations 2-3 train only on iteration-1 outputs +-0.5 %, so they
      can refine but not relocate them: map medians and pixel-wise correlation of
      iteration 3 vs iteration 1, from the step07 runs.
  C3  ANN vs Nelder-Mead on 1,000 pixels (500 mare + 500 highland, seeded), with
      timing extrapolated to the whole scene (the paper: ~35 h vs < 1 s mapping).

Usage: python step08_controls.py [--tag full|fast] [--workers 4] [--skip-nm]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import ann
import korokhin_common as kc
import shkuratov as sk
import step07_ann as s7

STAGE = "step08"
NOISE = 0.01
N_TEST = 5000
SHIFT_MIN = {"OPX": 0.0, "OLV": 0.0, "Gr": 30.0}
SHIFT_MAX = {"OPX": 10.0, "OLV": 10.0, "Gr": 50.0}
C2_SEEDS = (0, 1, 2)
N_NM = 500                        # per branch


def load_net(path) -> ann.MLP:
    net = ann.MLP(0)
    d = np.load(path)
    net.load({k: d[k] for k in d.files})
    return net


def recovery(tag: str) -> pd.DataFrame:
    model = sk.LunarModel(kc.M3_GRID_UM)
    rng = np.random.default_rng(12345)
    Y = s7.table11_set(N_TEST, rng)
    X = model.fast(Y)
    ok = np.all(np.isfinite(X), axis=1)
    Y, X = Y[ok], X[ok]
    Xn = X * (1.0 + NOISE * rng.standard_normal(X.shape))
    mid = np.array([0.5 * (kc.TABLE11_MIN[p] + kc.TABLE11_MAX[p]) for p in kc.PARAMS])
    rows = []
    for f in sorted(kc.PRODUCTS.glob("ann_%s_seed*_iter*_state.npz" % tag)):
        seed = int(f.name.split("seed")[1].split("_")[0])
        it = int(f.name.split("iter")[1].split("_")[0])
        pred = load_net(f).predict(Xn)
        for j, p in enumerate(kc.PARAMS):
            err = pred[:, j] - Y[:, j]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            rmse_mid = float(np.sqrt(np.mean((mid[j] - Y[:, j]) ** 2)))
            ss_tot = float(np.sum((Y[:, j] - Y[:, j].mean()) ** 2))
            rows.append({"seed": seed, "iteration": it, "param": p, "rmse": rmse,
                         "rmse_midpoint_guess": rmse_mid, "skill_ratio": rmse / rmse_mid,
                         "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot})
    return pd.DataFrame(rows)


def iter1_run(seed: int, lo: dict, hi: dict, shuffle: bool):
    """Iteration 1 exactly as step07, with altered ranges or shuffled spectra."""
    X3, nr, ns = s7.scene()
    model = sk.LunarModel(kc.M3_GRID_UM)
    rng = np.random.default_rng(seed)
    lo_v = np.array([lo.get(p, kc.TABLE11_MIN[p]) for p in kc.PARAMS])
    hi_v = np.array([hi.get(p, kc.TABLE11_MAX[p]) for p in kc.PARAMS])
    Y = lo_v + (hi_v - lo_v) * rng.random((kc.N_TRAIN_ITER1, 8))
    if shuffle:
        class Shuffled:                                  # forward model, rows permuted
            def __init__(self, m, perm):
                self.m, self.perm = m, perm

            def fast(self, P):
                out = self.m.fast(P)
                return out[self.perm] if out.shape[0] == self.perm.size else out
        perm = rng.permutation(Y.shape[0])
        params, sr, hist, info, _ = s7.train_iteration(1, seed + 1000, Shuffled(model, perm), Y, X3)
        # the stopping rule and sigma_Rel must use the TRUE forward model
        sr = s7.sigma_rel_map(model, params, X3)
    else:
        params, sr, hist, info, _ = s7.train_iteration(1, seed + 2000, model, Y, X3)
    return params, sr


def nm_pixel(args):
    import step03_basis_set as s3
    obs = args
    model = sk.LunarModel(kc.M3_GRID_UM)
    t = time.perf_counter()
    r = s3.fit_8param(model, obs)
    return r["theta"], 100 * r["sigma"] / obs.mean(), time.perf_counter() - t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-nm", action="store_true")
    a = ap.parse_args()
    kc.banner(STAGE, "controls: recovery, range-dependence, inheritance, Nelder-Mead (tag=%s)" % a.tag)
    kc.verify_manifest(STAGE)

    # C1
    rec = recovery(a.tag)
    rec.to_csv(kc.TABLES / "step08_c1_recovery.csv", index=False)
    s = rec.groupby(["iteration", "param"])[["r2", "skill_ratio"]].agg(["mean", "std"]).round(3)
    print("  C1 synthetic recovery (1 % noise), mean over seeds:")
    print(s.to_string())

    # C2a / C2b
    X3, nr, ns = s7.scene()
    ref_meds = {}
    real = sorted(kc.PRODUCTS.glob("ann_%s_seed*_iter1_params.npy" % a.tag))
    base = np.median(np.stack([np.median(np.load(f), axis=0) for f in real]), axis=0)
    rows = []
    with ProcessPoolExecutor(max_workers=min(a.workers, 6)) as ex:
        jobs = {}
        for sd in C2_SEEDS:
            jobs[("shuffled", sd)] = ex.submit(iter1_run, sd, {}, {}, True)
            jobs[("shifted", sd)] = ex.submit(iter1_run, sd, SHIFT_MIN, SHIFT_MAX, False)
        for (kind, sd), j in jobs.items():
            params, sr = j.result()
            med = np.median(params, axis=0)
            for i, p in enumerate(kc.PARAMS):
                rows.append({"control": kind, "seed": sd, "param": p, "map_median": float(med[i]),
                             "real_iter1_median": float(base[i]),
                             "sigma_rel_mean": float(np.nanmean(sr))})
    c2 = pd.DataFrame(rows)
    mid0 = {p: 0.5 * (kc.TABLE11_MIN[p] + kc.TABLE11_MAX[p]) for p in kc.PARAMS}
    mid1 = {p: 0.5 * (SHIFT_MIN.get(p, kc.TABLE11_MIN[p]) + SHIFT_MAX.get(p, kc.TABLE11_MAX[p]))
            for p in kc.PARAMS}
    summ = c2.groupby(["control", "param"]).agg(map_median=("map_median", "mean"),
                                                 real_iter1_median=("real_iter1_median", "first"),
                                                 sigma_rel_mean=("sigma_rel_mean", "mean")).reset_index()
    summ["range_midpoint"] = summ["param"].map(mid0)
    summ["shifted_midpoint"] = summ["param"].map(mid1)
    sh = summ["control"] == "shifted"
    summ["follows_range"] = np.where(sh & (summ["shifted_midpoint"] != summ["range_midpoint"]),
                                     (summ["map_median"] - summ["real_iter1_median"])
                                     / (summ["shifted_midpoint"] - summ["range_midpoint"]), np.nan)
    c2.to_csv(kc.TABLES / "step08_c2_runs.csv", index=False)
    summ.to_csv(kc.TABLES / "step08_c2_summary.csv", index=False)
    print("\n  C2 map medians: real iteration 1 vs no-information net vs shifted ranges")
    print(summ.round(3).to_string(index=False))

    # C2c inheritance
    rows = []
    for f1 in real:
        f3 = str(f1).replace("_iter1_", "_iter3_")
        p1, p3 = np.load(f1).astype(float), np.load(f3).astype(float)
        seed = int(f1.name.split("seed")[1].split("_")[0])
        for i, p in enumerate(kc.PARAMS):
            rows.append({"seed": seed, "param": p, "median_iter1": float(np.median(p1[:, i])),
                         "median_iter3": float(np.median(p3[:, i])),
                         "iqr_iter1": float(np.subtract(*np.percentile(p1[:, i], [75, 25]))),
                         "iqr_iter3": float(np.subtract(*np.percentile(p3[:, i], [75, 25]))),
                         "pixel_corr_iter1_iter3": float(np.corrcoef(p1[:, i], p3[:, i])[0, 1])})
    inh = pd.DataFrame(rows)
    inh.to_csv(kc.TABLES / "step08_c2c_inheritance.csv", index=False)
    print("\n  C2c iteration 3 vs iteration 1 (mean over seeds)")
    print(inh.groupby("param").mean(numeric_only=True).drop(columns="seed").round(3).to_string())

    # C3
    if a.skip_nm:
        print("  C3 skipped (--skip-nm)")
        kc.verify_manifest(STAGE)
        return 0
    high = np.load(kc.PRODUCTS / "highland_mask.npy").reshape(-1)
    rng = np.random.default_rng(777)
    pix = np.r_[rng.choice(np.flatnonzero(~high), N_NM, replace=False),
                rng.choice(np.flatnonzero(high), N_NM, replace=False)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(nm_pixel, [X3[k] for k in pix], chunksize=10))
    wall = time.perf_counter() - t0
    nm = np.array([r[0] for r in res])
    ann_med = np.load(kc.PRODUCTS / ("ann_%s_median_iter3.npy" % a.tag))[pix].astype(float) \
        if (kc.PRODUCTS / ("ann_%s_median_iter3.npy" % a.tag)).exists() else None
    rows = []
    for j, p in enumerate(kc.PARAMS):
        for branch, m in (("mare", ~high[pix]), ("highland", high[pix])):
            row = {"param": p, "branch": branch, "nm_median": float(np.median(nm[m, j]))}
            if ann_med is not None:
                row.update({"ann_median": float(np.median(ann_med[m, j])),
                            "corr_ann_nm": float(np.corrcoef(nm[m, j], ann_med[m, j])[0, 1]),
                            "mean_abs_diff": float(np.mean(np.abs(nm[m, j] - ann_med[m, j])))})
            rows.append(row)
    c3 = pd.DataFrame(rows)
    per_px = np.array([r[2] for r in res])
    info12 = pd.read_csv(kc.TABLES / ("step07_table12_%s.csv" % a.tag))
    ann_train_min = float(info12.groupby("seed")["t_optim_min"].sum().median())
    ann_map_s = float(info12["t_map_s"].median())
    timing = pd.DataFrame([{"nm_seconds_per_pixel_single_core": float(np.median(per_px)),
                            "nm_extrapolated_scene_hours_single_core": float(np.median(per_px)) * 152000 / 3600,
                            "nm_sigma_rel_mean": float(np.mean([r[1] for r in res])),
                            "ann_training_minutes_3_iterations": ann_train_min,
                            "ann_mapping_seconds": ann_map_s, "nm_wall_seconds": wall,
                            "paper_nm_scene_hours": 35.0}])
    c3.to_csv(kc.TABLES / "step08_c3_ann_vs_nm.csv", index=False)
    timing.to_csv(kc.TABLES / "step08_c3_timing.csv", index=False)
    pd.DataFrame(nm, columns=kc.PARAMS).assign(pixel=pix).to_csv(kc.TABLES / "step08_c3_nm_fits.csv", index=False)
    print("\n  C3 ANN vs Nelder-Mead on %d pixels" % len(pix))
    print(c3.round(3).to_string(index=False))
    print(timing.round(3).T.to_string())
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
