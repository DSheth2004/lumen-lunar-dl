"""
Step 03 -- baselines under the same LOSO (19 folds) and LOMO (6 folds) splits as LUMEN
(METHOD_NOTES.md D13, D14).

  classic   mean-of-others (fraction-matched), PLS, ridge, RF on the LUMEN input features
            (log R + continuum-removed R, 142 values) and RF on the 71 raw reflectances.
            Hyperparameters by inner GroupKFold over the training soils only.
  FD        Fernandez-Diaz et al. (2026) re-implemented on the same 71-band spectra:
            2-level db4 DWT (A2 + D2) -> 6-D autoencoder (trained unsupervised on the M3
            Shapley pixels, or on synthetic spectra) -> Lasso / SVR-lin / SVR-RBF / RF with
            the Table 5 grids; plus the no-AE variant (wavelet -> regressors). FeO only.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pywt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import lumen_common as lc
import lumen_eval as le
from step04_lumen import folds, load_lab

STAGE = "step03"
KOROKHIN_EQ2 = (1.685, 0.0355)          # A_lab = a * A_M3 + b (Korokhin et al. 2024, Eq. 2)


def inner_cv(groups):
    return GroupKFold(n_splits=min(5, len(np.unique(groups))))


# --------------------------------------------------------------------------
# classic baselines
# --------------------------------------------------------------------------
class YScaled:
    """Fit a multi-output regressor on training-standardised targets."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, Y, groups=None):
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-9
        Z = (Y - self.mu) / self.sd
        if groups is not None:
            self.est.fit(X, Z, groups=groups)
        else:
            self.est.fit(X, Z)
        return self

    def predict(self, X):
        p = self.est.predict(X)
        return np.asarray(p).reshape(len(X), -1) * self.sd + self.mu


def make_classic(name, n_train, seed):
    if name == "PLS":
        grid = {"pls__n_components": list(range(1, min(10, n_train - 2) + 1))}
        pipe = Pipeline([("x", StandardScaler()), ("pls", PLSRegression(scale=False))])
        return GridSearchCV(pipe, grid, scoring="neg_mean_squared_error", n_jobs=1), True
    if name == "ridge":
        grid = {"ridge__alpha": list(np.logspace(-3, 4, 15))}
        pipe = Pipeline([("x", StandardScaler()), ("ridge", Ridge())])
        return GridSearchCV(pipe, grid, scoring="neg_mean_squared_error", n_jobs=1), True
    if name in ("RF", "RF-raw"):
        return RandomForestRegressor(500, max_features=1 / 3, random_state=seed, n_jobs=1), False
    raise ValueError(name)


def fit_predict(name, Xtr, Ytr, gtr, Xte, seed):
    est, tuned = make_classic(name, len(Xtr), seed)
    if tuned:
        est.cv = inner_cv(gtr)
        m = YScaled(est).fit(Xtr, Ytr, groups=gtr)
        chosen = est.best_params_
    else:
        m = YScaled(est).fit(Xtr, Ytr)
        chosen = {}
    return m.predict(Xte), chosen


def run_classic(X, meta, F, seed):
    rows, choices = [], []
    comp_lab = meta[lc.COMP].notna().all(axis=1).to_numpy()
    feats = {"PLS": F, "ridge": F, "RF": F, "RF-raw": X}
    for protocol in ("LOSO", "LOMO"):
        for fold, test in folds(meta, protocol):
            tr, te = np.flatnonzero(~test), np.flatnonzero(test)
            g = meta["soil"].to_numpy()
            mo = le.mean_of_others(meta, tr, te)
            mo.insert(0, "method", le.REFERENCE)
            mo.insert(0, "protocol", protocol)
            rows.append(mo)
            for name, Fm in feats.items():
                out = {"protocol": protocol, "method": name, "row": te}
                trc = tr[comp_lab[tr]]
                pc, ch1 = fit_predict(name, Fm[trc], meta.loc[trc, lc.COMP].to_numpy(float),
                                      g[trc], Fm[te], seed)
                pc = np.clip(pc, 0, None)
                pc = 100 * pc / pc.sum(1, keepdims=True)
                for j, t in enumerate(lc.COMP):
                    out["pred_" + t] = pc[:, j]
                sc = lc.CHEM + [lc.MAT]
                ps, ch2 = fit_predict(name, Fm[tr], meta.loc[tr, sc].to_numpy(float), g[tr],
                                      Fm[te], seed)
                for j, t in enumerate(sc):
                    out["pred_" + t] = ps[:, j]
                rows.append(pd.DataFrame(out))
                choices.append({"protocol": protocol, "fold": fold, "method": name,
                                "comp_params": str(ch1), "scalar_params": str(ch2)})
        print("  classic baselines: %s done" % protocol, flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(choices)


# --------------------------------------------------------------------------
# Fernandez-Diaz et al. (2026)
# --------------------------------------------------------------------------
def dwt_a2d2(R):
    c = pywt.wavedec(np.asarray(R, float), "db4", mode="periodization", level=2, axis=1)
    return np.hstack([c[0], c[1]]).astype(np.float32)


def train_fd_autoencoder(W, seed, log):
    """FD's 7-layer AE (in -> 32 -> 16 -> 6 -> 16 -> 32 -> in), as specified in the hand-off."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)                 # a tiny MLP: more threads only add contention
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = W.shape[1]

    def block(i, o):
        return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.LeakyReLU(0.01), nn.Dropout(0.3)]
    enc = nn.Sequential(*block(d, 32), *block(32, 16), nn.Linear(16, 6))
    dec = nn.Sequential(*block(6, 16), *block(16, 32), nn.Linear(32, d))
    for m in list(enc.modules()) + list(dec.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    perm = rng.permutation(len(W))
    nv = len(W) // 10
    va, tr = perm[:nv], perm[nv:]
    Wt = torch.from_numpy(W)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(200):
        enc.train(); dec.train()
        p = rng.permutation(tr)
        for s in range(0, len(p) - 255, 256):
            xb = Wt[p[s:s + 256]]
            loss = ((dec(enc(xb)) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); dec.eval()
        with torch.no_grad():
            vl = float(((dec(enc(Wt[va])) - Wt[va]) ** 2).mean())
        hist.append({"epoch": ep + 1, "val_mse": vl})
        if vl < best - 1e-9:
            best, bad = vl, 0
            best_state = ({k: v.clone() for k, v in enc.state_dict().items()},
                          {k: v.clone() for k, v in dec.state_dict().items()})
        else:
            bad += 1
            if bad >= 12:
                break
    enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    enc.eval()
    log("    AE: %d epochs, best val MSE %.3e (n = %d)" % (len(hist), best, len(W)))

    def encode(Wq):
        with torch.no_grad():
            return enc(torch.from_numpy(Wq)).numpy().astype(np.float64)
    return encode, pd.DataFrame(hist)


def m3_shapley_lab_scale():
    cube = np.load(lc.M3_SHAPLEY_DIR / "scene_rfl.npy", mmap_mode="r")      # (lines, 85, samples)
    R = np.transpose(np.asarray(cube), (0, 2, 1)).reshape(-1, cube.shape[1])[:, lc.input_band_index()]
    R = KOROKHIN_EQ2[0] * R + KOROKHIN_EQ2[1]
    ok = np.isfinite(R).all(1) & (R > 0).all(1) & (R < 1.5).all(1)
    return R[ok], int(ok.sum()), len(ok)


FD_GRIDS = {   # Table 5 of the reproduction (FernandezDiaz2026_WaveletAE_FeO/src/model.py)
    "Lasso": (Lasso(max_iter=10000), {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}),
    "SVR-lin": (SVR(kernel="linear"), {"C": [0.01, 0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.5]}),
    "SVR-RBF": (SVR(kernel="rbf"), {"C": [0.1, 1.0, 10.0, 100.0], "epsilon": [0.01, 0.1, 0.5],
                                    "gamma": ["scale", "auto"]}),
    "RF": (RandomForestRegressor(random_state=42, n_jobs=1),
           {"n_estimators": [50, 100, 200, 500], "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]}),
}


def run_fd(X, meta, seed, n_jobs, log):
    W_lab = dwt_a2d2(X)
    feats = {"FD-wavelet": W_lab.astype(np.float64)}
    hist = []
    R_m3, n_ok, n_all = m3_shapley_lab_scale()
    log("  FD: M3 Shapley pixels kept %d / %d (lab scale, Korokhin Eq. 2)" % (n_ok, n_all))
    enc, h = train_fd_autoencoder(dwt_a2d2(R_m3), seed, log)
    feats["FD-AE-M3"] = enc(W_lab)
    hist.append(h.assign(ae="M3"))
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    Rs = syn["R"][np.random.default_rng(seed).choice(len(syn["R"]), n_ok, replace=False)]
    enc, h = train_fd_autoencoder(dwt_a2d2(Rs), seed, log)
    feats["FD-AE-synth"] = enc(W_lab)
    hist.append(h.assign(ae="synthetic"))

    y = meta["FeO"].to_numpy(float)
    g = meta["soil"].to_numpy()
    rows, choices = [], []
    for protocol in ("LOSO", "LOMO"):
        t0 = time.time()
        for fold, test in folds(meta, protocol):
            tr, te = np.flatnonzero(~test), np.flatnonzero(test)
            for fname, Fm in feats.items():
                for rname, (est, grid) in FD_GRIDS.items():
                    pipe = Pipeline([("x", StandardScaler()), ("m", est)])
                    gs = GridSearchCV(pipe, {"m__" + k: v for k, v in grid.items()},
                                      scoring="neg_mean_absolute_error", cv=inner_cv(g[tr]),
                                      n_jobs=n_jobs)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        gs.fit(Fm[tr], y[tr], groups=g[tr])
                    rows.append(pd.DataFrame({"protocol": protocol, "method": "%s/%s" % (fname, rname),
                                              "row": te, "pred_FeO": gs.predict(Fm[te])}))
                    choices.append({"protocol": protocol, "fold": fold, "features": fname,
                                    "regressor": rname, "params": str(gs.best_params_),
                                    "inner_cv_mae": -gs.best_score_})
        log("  FD %s done (%.0f s)" % (protocol, time.time() - t0))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(choices), pd.concat(hist)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    t0 = lc.banner(STAGE, "baselines under LOSO and LOMO")
    lc.verify_manifest(STAGE)
    X, meta = load_lab()
    F = lc.spectral_features(X).reshape(len(X), -1).astype(np.float64)

    pred_c, ch_c = run_classic(X, meta, F, args.seed)
    pred_f, ch_f, hist = run_fd(X, meta, args.seed, args.jobs, print)
    pred = pd.concat([pred_c, pred_f], ignore_index=True)
    pred.to_csv(lc.TABLES / "step03_predictions.csv", index=False)
    pd.concat([ch_c, ch_f], ignore_index=True).to_csv(lc.TABLES / "step03_hyperparameters.csv", index=False)
    hist.to_csv(lc.TABLES / "step03_fd_autoencoder_history.csv", index=False)

    summ = le.summarise(pred, meta)
    summ.to_csv(lc.TABLES / "step03_metrics.csv", index=False)
    soil = summ[summ["level"] == "soil"]
    show = soil[~soil["method"].str.startswith("FD")].pivot_table(
        index=["protocol", "method"], columns="target", values="mae")[le.TARGETS + ["COMP_mean"]]
    print(show.round(2).to_string())
    print(soil[soil["target"] == "FeO"].pivot_table(index="method", columns="protocol",
                                                    values="mae").round(3).to_string())
    lc.done(STAGE, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
