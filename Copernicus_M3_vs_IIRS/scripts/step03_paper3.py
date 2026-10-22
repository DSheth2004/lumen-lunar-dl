"""
Step 03 -- Paper 3 (Korokhin et al. 2024, Shkuratov model + ANN) on Copernicus.

The method, reused from ../Korokhin2024_ShkuratovANN/scripts without changes to its
physics or training rules:
  * spectra: the recipe's lambda-convolution (Gaussian, FWHM 3 band pixels, odd
    reflection at the ends), linear resampling to the 0.1-um grid points, Eq. (2)
    A_RELAB = 1.685 A + 0.0355;
  * the 8-parameter Shkuratov forward model (S, nFe, Gr, CPX, OPX, OLV, PLG, ILM),
    sk.LunarModel on the configuration's grid;
  * the 3-iteration MLP training of step07 (Table 11 set, then the scene's own
    parameters jittered by < +-0.5 %), stopping at the epoch that best reproduces the
    scene spectra.

What differs here, each forced by the data and applied identically to all configs:
  * grid.  M3_full uses the paper's 21 points (0.54-2.54 um). The common configs use
    the 14 points inside 0.85-2.50 um and outside the IIRS gaps: 0.94-1.14, 1.34-1.84,
    2.04-2.44 um. The MLP's input layer is 14 wide instead of 21; nothing else changes.
    The model is a physics forward model, so it is simply evaluated on those 14 points.
  * no Fourier destriping.  The recipe's stripe filter works on the raw along-track
    image; these cubes are already map-projected. Paper 3's own `no_destripe`
    sensitivity is the matching variant.
  * IIRS lambda-convolution runs within each detector segment, so no smoothing crosses
    a gap.
  * Eq. (2) is M3's published transfer to the RELAB system. It is applied unchanged to
    IIRS as well: there is no published IIRS equivalent, and deriving one from M3 would
    force the two sensors to agree. The raw brightness difference is reported in step01.
  * three seeds (0, 1, 2) instead of five, to fit four configurations on 8 cores.
  * scene = the 98,421 joint pixels (step01), the same for every configuration.

Also computed, needing no visible bands: the 1550-nm band-depth ilmenite of Surkov
et al. (2020) (paper 3 Fig. 31), with the hull over each configuration's own bands.
Not computable for IIRS (need 415 or 750 nm): the MLR maps (Eqs. 3-5), OMAT/TiO2
(Eqs. 6-7), and the Korokhin (2008) TiO2 ANN. They are listed, not approximated.

Usage:  python step03_paper3.py                  all configs x seeds, then aggregate
        python step03_paper3.py --fast           60-epoch smoke test, seed 0
        python step03_paper3.py --config C --seed S   one job (used internally)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import cc_common as cc

sys.path.insert(0, str(cc.KOROKHIN_SCRIPTS))
import ann                                   # noqa: E402
import korokhin_common as kc                 # noqa: E402
import shkuratov as sk                       # noqa: E402
from step05_m3_preprocess import lambda_kernel   # noqa: E402
from step07_ann import ITER, JITTER, sigma_rel_map, table11_set   # noqa: E402
from step09_maps_comparisons import BD_NM, BD_TO_ILM   # noqa: E402

STAGE = "step03"
SEEDS = (0, 1, 2)
FAST_EPOCHS = 60
RECIPE_MAX_UM = 2.6
NOT_COMPUTABLE_IIRS = [
    ("MLR CPX/PLG/Is/OPX/OLV/d (Eqs. 3-5)", "A(415), A(750)"),
    ("OMAT_Fe (Eq. 6)", "R(750)"),
    ("TiO2, Lucey (Eq. 7)", "R(415), R(750)"),
    ("TiO2, Korokhin 2008 ANN (Fig. 30)", "A(415), A(750)"),
]


def grid_for(config: str) -> np.ndarray:
    g = kc.M3_GRID_UM
    return g if config == "M3_full" else g[cc.in_common(g)]


def segments(wl_um: np.ndarray) -> list[np.ndarray]:
    """Index runs without a gap (> 60 nm; M3 spacing is at most 40 nm) between neighbouring bands."""
    cut = np.flatnonzero(np.diff(wl_um) > 0.06) + 1
    return np.split(np.arange(wl_um.size), cut)


def spectra(config: str) -> tuple[np.ndarray, np.ndarray]:
    """(grid_um, X[n_joint, n_grid]) in the RELAB system, per the recipe above."""
    sensor = "M3" if config.startswith("M3") else "IIRS"
    wl, cube = cc.wavelengths_um(sensor), cc.read_cube(sensor)
    joint = np.load(cc.PRODUCTS / "joint_mask.npy")
    S = cube[:, joint]                                       # (bands, n)
    use = np.isfinite(S).all(axis=1) & (wl <= RECIPE_MAX_UM)
    wl, S = wl[use], S[use]
    if config == "IIRS_scaled":
        S = S / cc.iirs_scale_factor()
    sm = np.empty_like(S)
    for seg in segments(wl):
        sm[seg] = lambda_kernel(seg.size) @ S[seg]
    grid = grid_for(config)
    X = np.empty((S.shape[1], grid.size))
    for g, lam in enumerate(grid):
        i = int(np.clip(np.searchsorted(wl, lam), 1, wl.size - 1))
        if not (wl[i - 1] <= lam <= wl[i]) or wl[i] - wl[i - 1] > 0.06:
            if not (config == "M3_full" and lam < wl[0]):
                sys.exit("[%s] FATAL: grid point %.2f um not bracketed by %s bands"
                         % (STAGE, lam, sensor))
        f = float(np.clip((lam - wl[i - 1]) / (wl[i] - wl[i - 1]), 0.0, 1.0))
        X[:, g] = (1 - f) * sm[i - 1] + f * sm[i]
    return grid, kc.EQ2_SLOPE * X + kc.EQ2_INTERCEPT


def train_iteration(it, seed, model, Ytrain, Xs, n_in, fast):
    """step07.train_iteration with the input width as a parameter (fresh weights)."""
    cfg = ITER[it]
    Xtrain = model.fast(Ytrain)
    ok = np.all(np.isfinite(Xtrain), axis=1)
    Xtrain, Ytrain = Xtrain[ok], Ytrain[ok]
    net = ann.MLP(seed * 100 + it, layers=(n_in,) + tuple(kc.ANN_LAYERS[1:]))
    net.fit_scaling(Xtrain, Ytrain)
    Xn, Tn, Xsn = net.norm_x(Xtrain), net.norm_y(Ytrain), net.norm_x(Xs)
    cap = FAST_EPOCHS if fast else cfg["cap"]
    every = 1 if fast else cfg["every"]
    lr = cfg["lr0"]
    hist, best = [], {"sigma_rel": np.inf}
    for ep in range(1, cap + 1):
        s_train = net.epoch(Xn, Tn, lr, kc.ANN_MOMENTUM)
        if cfg["k"] is not None:
            lr = cfg["k"] * s_train
        if ep % every == 0 or ep == 1:
            p = net.denorm_y(net.predict_norm(Xsn))
            m = float(np.nanmean(sigma_rel_map(model, p, Xs)))
            hist.append({"epoch": ep, "sigma_train": s_train, "sigma_rel_mean": m})
            if m < best["sigma_rel"]:
                best = {"sigma_rel": m, "epoch": ep, "state": net.state()}
            if cfg["patience"] and not fast and ep - best["epoch"] >= cfg["patience"]:
                break
    net.load(best["state"])
    params = net.predict(Xs)
    sr = sigma_rel_map(model, params, Xs)
    info = {"iteration": it, "seed": seed, "pattern_size": int(Xtrain.shape[0]),
            "optimal_epoch": best["epoch"], "epochs_run": ep,
            "sigma_rel_mean": float(np.nanmean(sr))}
    return params, sr, pd.DataFrame(hist), info


def run_job(config: str, seed: int, fast: bool) -> None:
    grid, X = spectra(config)
    model = sk.LunarModel(grid)
    rng = np.random.default_rng(seed)
    Y = table11_set(kc.N_TRAIN_ITER1, rng)
    tag = "fast" if fast else "full"
    for it in (1, 2, 3):
        t0 = time.time()
        params, sr, hist, info = train_iteration(it, seed, model, Y, X, grid.size, fast)
        info.update({"config": config, "n_inputs": int(grid.size), "minutes": (time.time() - t0) / 60})
        base = "p3_%s_%s_seed%d_iter%d" % (tag, config, seed, it)
        np.save(cc.PRODUCTS / (base + "_params.npy"), params.astype(np.float32))
        np.save(cc.PRODUCTS / (base + "_sigma_rel.npy"), sr.astype(np.float32))
        hist.to_csv(cc.TABLES / (base + "_history.csv"), index=False)
        (cc.TABLES / (base + "_info.json")).write_text(json.dumps(info))
        print("  %s seed %d iter %d: optimum epoch %d of %d, sigma_Rel %.2f%%, %.1f min"
              % (config, seed, it, info["optimal_epoch"], info["epochs_run"],
                 info["sigma_rel_mean"], info["minutes"]), flush=True)
        Y = params.astype(np.float64) * (1.0 + rng.uniform(-JITTER, JITTER, params.shape))


def ilmenite_1550(config: str) -> np.ndarray:
    """Surkov (2020) Eq. 4.11-4.12 per joint pixel: upper convex hull over the
    configuration's native bands (<= 2.6 um, or the common bands), S = 1 - A/A_CH,
    BD = mean S at 1489-1579 nm (interpolated), ilmenite wt% = 202.8 BD."""
    sensor = "M3" if config.startswith("M3") else "IIRS"
    wl, cube = cc.wavelengths_um(sensor), cc.read_cube(sensor)
    joint = np.load(cc.PRODUCTS / "joint_mask.npy")
    S = cube[:, joint]
    use = np.isfinite(S).all(axis=1) & (wl <= RECIPE_MAX_UM)
    if config != "M3_full":
        use &= cc.in_common(wl)
    wl, S = wl[use], S[use]
    if config == "IIRS_scaled":
        S = S / cc.iirs_scale_factor()
    target = np.asarray(BD_NM) * 1e-3
    out = np.empty(S.shape[1])
    for p in range(S.shape[1]):
        y = S[:, p]
        hull = []
        for i in range(wl.size):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (wl[b] - wl[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (wl[i] - wl[a]) >= 0:
                    hull.pop()
                else:
                    break
            hull.append(i)
        cr = y / np.interp(wl, wl[hull], y[hull])
        out[p] = BD_TO_ILM * float(np.mean(1.0 - np.interp(target, wl, cr)))
    return out


def aggregate(tag: str, seeds) -> None:
    joint = np.load(cc.PRODUCTS / "joint_mask.npy")
    infos = []
    for config in cc.CONFIGS:
        stack = []
        for s in seeds:
            base = cc.PRODUCTS / ("p3_%s_%s_seed%d_iter3_params.npy" % (tag, config, s))
            if base.exists():
                stack.append(np.load(base).astype(float))
            for it in (1, 2, 3):
                f = cc.TABLES / ("p3_%s_%s_seed%d_iter%d_info.json" % (tag, config, s, it))
                if f.exists():
                    infos.append(json.loads(f.read_text()))
        if not stack:
            continue
        med = np.median(np.stack(stack), axis=0)                # (n_joint, 8)
        np.save(cc.PRODUCTS / ("p3_%s_%s_median_params.npy" % (tag, config)), med.astype(np.float32))
        spread = np.std(np.stack(stack), axis=0).mean(axis=0) if len(stack) > 1 else np.full(8, np.nan)
        for j, name in enumerate(kc.PARAMS):
            img = np.full(joint.shape, np.nan)
            img[joint] = med[:, j]
            cc.save_map(cc.PRODUCTS / ("paper3_%s_%s.tif" % (config, name)), img)
        print("  %s: %d seeds; mean seed-to-seed SD per parameter: %s"
              % (config, len(stack), ", ".join("%s %.3g" % (n, v) for n, v in zip(kc.PARAMS, spread))))
    pd.DataFrame(infos).to_csv(cc.TABLES / ("step03_training_%s.csv" % tag), index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if a.config is not None:
        run_job(a.config, a.seed, a.fast)
        return 0

    cc.banner(STAGE, "Paper 3 (Korokhin 2024) on Copernicus, M3 and IIRS")
    cc.verify_manifest(STAGE)
    seeds = (0,) if a.fast else SEEDS
    tag = "fast" if a.fast else "full"

    if not a.aggregate_only:
        for config in cc.CONFIGS:
            grid, X = spectra(config)
            print("  %-12s %2d grid points %.2f-%.2f um, %d pixels, <A> median %.4f"
                  % (config, grid.size, grid[0], grid[-1], X.shape[0], float(np.median(X.mean(axis=1)))))
        env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        jobs = [(c, s) for c in cc.CONFIGS for s in seeds]
        procs = [subprocess.Popen([sys.executable, "-u", __file__, "--config", c, "--seed", str(s)]
                                  + (["--fast"] if a.fast else []), env=env) for c, s in jobs]
        rc = [p.wait() for p in procs]
        if any(rc):
            sys.exit("[%s] FATAL: a job failed: %s" % (STAGE, list(zip(jobs, rc))))

    aggregate(tag, seeds)

    # ---- 1550-nm ilmenite band depth, and what cannot be computed ----------
    joint = np.load(cc.PRODUCTS / "joint_mask.npy")
    for config in cc.CONFIGS:
        v = ilmenite_1550(config)
        np.save(cc.PRODUCTS / ("p3_ilm1550_%s.npy" % config), v.astype(np.float32))
        img = np.full(joint.shape, np.nan)
        img[joint] = v
        cc.save_map(cc.PRODUCTS / ("paper3_%s_ILM_bd1550.tif" % config), img)
        print("  1550-nm band-depth ilmenite, %-12s median %.2f wt%%" % (config, float(np.median(v))))
    pd.DataFrame(NOT_COMPUTABLE_IIRS, columns=["product", "needs"]).assign(
        IIRS="not computable (IIRS starts at 847 nm)").to_csv(
        cc.TABLES / "step03_not_computable_for_iirs.csv", index=False)

    cc.verify_manifest(STAGE)
    print("\n  STEP 03 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
