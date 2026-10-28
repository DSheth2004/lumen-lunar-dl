"""
Step 10 -- M3 landing-site validation (METHOD_NOTES.md Addendum A, D20-D23).

FeO at the returned-sample landing sites, from real M3 L2 spectra staged by step00b:
  LUMEN-M3          A4 ensemble, no M3 training data; for an Apollo site the members that
                    held out that mission (LOMO), otherwise the final all-soil ensemble;
  LUMEN-M3 + recal  leave-one-site-out OLS on LUMEN's site predictions;
  FD as published   83 bands -> db4 A2+D2 (42) -> 6-D AE (Shapley pixels) -> regressors,
                    leave-one-site-out, trained on the samples of the other sites;
  FD no-AE          wavelet coefficients -> regressors;
  band ratio        Lucey-type theta from R750 and R950, leave-one-site-out linear fit;
  mean of others    mean truth of the other sites.
Scored over sites (MAE, RMSE, R2); paired Wilcoxon, Holm over comparators.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lumen_common as lc
import lumen_eval as le
import lumen_model as lm
import lumen_plot as lp
from step03_baselines import FD_GRIDS, dwt_a2d2, train_fd_autoencoder
from step04_lumen import N_MEMBERS

STAGE = "step10"
SITES = lc.DATA / "m3_sites"
EQ2 = (1.685, 0.0355)
REDERIVED = (1.6857, 0.0136)                 # step07, A16 3x3 vs 62231 <45 um (sensitivity only)
FD_BANDS = np.arange(2, 85)                  # M3 bands 3-85 (FD's 83 bands, 475-3000 nm)
MISSION_FOLD = {"Apollo 11": "A11", "Apollo 12": "A12", "Apollo 14": "A14", "Apollo 15": "A15",
                "Apollo 16": "A16", "Apollo 17": "A17"}
SEED = 0


def block_pixels(block, centre_only=False):
    """(3, 85, 3) block -> its pixels (9 x 85, or the centre pixel only), the valid mask and
    the minimum number of valid pixels for the strip to contribute (D21)."""
    px = np.transpose(block, (0, 2, 1)).reshape(9, 85)
    if centre_only:
        px = px[[4]]
    sub = px[:, FD_BANDS]
    ok = np.isfinite(sub).all(1) & (sub > 0).all(1) & (sub < 1).all(1)
    return px, ok, 1 if centre_only else 5


def site_spectra(root=SITES, centre_only=False):
    sites = pd.read_csv(root / "sites.csv")
    strips = pd.read_csv(root / "site_strips.csv")
    blocks = np.load(root / "site_strip_blocks.npy")               # (k, 3, 85, 3)
    kept = strips[strips["kept"]].copy()
    spec_rows, per_strip = {}, []
    for sid, g in kept.groupby("site_id"):
        S = []
        for _, r in g.iterrows():
            px, ok, need = block_pixels(blocks[int(r["block"])], centre_only)
            per_strip.append({"site_id": sid, "product": r["product"], "n_valid_pixels": int(ok.sum()),
                              "contributes": bool(ok.sum() >= need)})
            if ok.sum() >= need:
                S.append(px[ok].mean(0))
        if S:
            spec_rows[sid] = (np.median(np.asarray(S), axis=0), np.asarray(S))
    ps = pd.DataFrame(per_strip)
    sites["n_strips_found"] = sites["site_id"].map(kept.groupby("site_id").size()).fillna(0).astype(int)
    sites["n_strips_used"] = sites["site_id"].map(ps[ps["contributes"]].groupby("site_id").size()).fillna(0).astype(int)
    return sites, spec_rows, ps


def lumen_models():
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    scaler = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    lam = syn["lam_um"]

    def load(pattern):
        out = []
        for k in range(N_MEMBERS):
            m = lm.LumenNet()
            m.load_state_dict(torch.load(lc.PRODUCTS / (pattern % k)))
            out.append(m)
        return out
    models = {"FINAL": load("final_A4_full_member%d.pt")}
    for fold in MISSION_FOLD.values():
        models[fold] = load("lomo_A4_full_" + fold + "_member%d.pt")
    return models, scaler, lam


def lumen_predict(models, scaler, lam, R85, cal):
    R = cal[0] * np.atleast_2d(R85)[:, lc.input_band_index()] + cal[1]
    e = lm.ensemble([lm.member_outputs(m, scaler(lc.spectral_features(R, lam))) for m in models])
    out = {}
    for j, t in enumerate(lc.COMP):
        out[t], out["sd_" + t] = e["comp_mean"][:, j], np.sqrt(e["comp_var"][:, j])
    for j, t in enumerate(lm.SCALARS):
        out[t], out["sd_" + t] = e["scal_mean"][:, j], np.sqrt(e["scal_var"][:, j])
    return out


def loso_linear(x, y):
    """Leave-one-site-out OLS y = a + b x."""
    p = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b, a = np.polyfit(x[m], y[m], 1)
        p[i] = a + b * x[i]
    return p


def lucey_theta(R85, wl_nm):
    i750, i950 = np.argmin(np.abs(wl_nm - 750)), np.argmin(np.abs(wl_nm - 950))
    r750, r950 = R85[:, i750], R85[:, i950]
    return np.arctan((r950 / r750 - 1.19) / (r750 - 0.08))


def fd_encoder(seed):
    """FD's autoencoder, trained unsupervised on the Shapley-scene pixels (unscaled L2, 83 bands)."""
    raw = np.load(lc.M3_SHAPLEY_DIR / "scene_rfl.npy")
    pix = np.transpose(raw, (0, 2, 1)).reshape(-1, raw.shape[1])[:, FD_BANDS].astype(float)
    pix = pix[np.isfinite(pix).all(1) & (pix > 0).all(1) & (pix < 1).all(1)]
    return train_fd_autoencoder(dwt_a2d2(pix), seed, print)


def fd_loso(sites, specs, gt, seed):
    """FD on its own task, leave-one-site-out; training rows are samples."""
    enc, hist = fd_encoder(seed)
    sid = sites["site_id"].to_numpy()
    W_site = dwt_a2d2(np.vstack([specs[s][0][FD_BANDS] for s in sid])).astype(np.float64)
    feats = {"FD-wavelet": W_site, "FD-AE-M3": enc(W_site.astype(np.float32))}
    # sample rows -> their site's spectrum
    gt = gt.copy()
    gt["site_id"] = gt["key"].map(dict(zip(zip(sites["lat"], sites["lon"]), sid)))
    gt = gt[gt["site_id"].isin(sid)]
    srow = gt["site_id"].map({s: i for i, s in enumerate(sid)}).to_numpy()
    y = gt["feo_wt_pct"].to_numpy(float)
    preds, choices = {}, []
    for fname, Fs in feats.items():
        for rname, (est, grid) in FD_GRIDS.items():
            name = "%s/%s" % (fname, rname)
            p = np.empty(len(sid))
            for i, s in enumerate(sid):
                tr = gt["site_id"].to_numpy() != s
                groups = gt["site_id"].to_numpy()[tr]
                pipe = Pipeline([("x", StandardScaler()), ("m", est)])
                gs = GridSearchCV(pipe, {"m__" + k: v for k, v in grid.items()},
                                  scoring="neg_mean_absolute_error",
                                  cv=GroupKFold(n_splits=min(5, len(np.unique(groups)))), n_jobs=-1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gs.fit(Fs[srow[tr]], y[tr], groups=groups)
                p[i] = gs.predict(Fs[[i]])[0]
                choices.append({"method": name, "held_out_site": s, "params": str(gs.best_params_)})
            preds[name] = p
    return preds, pd.DataFrame(choices), hist


def main() -> int:
    t0 = lc.banner(STAGE, "M3 landing-site validation (Addendum A)")
    lc.verify_manifest(STAGE)
    lp.setup()
    torch.set_num_threads(4)
    sites, specs, per_strip = site_spectra()
    per_strip.to_csv(lc.TABLES / "step10_strip_quality.csv", index=False)
    print(sites[["site_id", "mission", "lat", "lon", "n_samples", "feo_mean", "feo_range",
                 "n_strips_found", "n_strips_used"]].round(3).to_string(index=False))
    missing = sites[sites["n_strips_used"] == 0]["site_id"].tolist()
    print("  sites without a usable M3 strip (excluded): %s" % (", ".join(missing) or "none"))
    sites = sites[sites["n_strips_used"] > 0].reset_index(drop=True)
    sid = sites["site_id"].to_numpy()
    truth = sites["feo_mean"].to_numpy(float)
    R85 = np.vstack([specs[s][0] for s in sid])
    wl = lc.m3_wavelengths_nm()

    # ---- LUMEN ----------------------------------------------------------------
    models, scaler, lam = lumen_models()
    rows, strip_rows = [], []
    for i, s in sites.iterrows():
        fold = MISSION_FOLD.get(s["mission"], "FINAL")
        out = lumen_predict(models[fold], scaler, lam, specs[s["site_id"]][0], EQ2)
        out2 = lumen_predict(models[fold], scaler, lam, specs[s["site_id"]][0], REDERIVED)
        per = lumen_predict(models[fold], scaler, lam, specs[s["site_id"]][1], EQ2)
        rows.append({"site_id": s["site_id"], "model": fold, **{k: float(v[0]) for k, v in out.items()},
                     "FeO_rederived_cal": float(out2["FeO"][0]),
                     "FeO_strip_min": float(per["FeO"].min()), "FeO_strip_max": float(per["FeO"].max())})
    lum = pd.DataFrame(rows)
    lum.to_csv(lc.TABLES / "step10_lumen_site_outputs.csv", index=False)
    pred = {"LUMEN-M3": lum["FeO"].to_numpy(),
            "LUMEN-M3 (re-derived calibration)": lum["FeO_rederived_cal"].to_numpy()}
    pred["LUMEN-M3 + site recalibration"] = loso_linear(pred["LUMEN-M3"], truth)

    # ---- comparators ----------------------------------------------------------
    gt = pd.read_csv(lc.GT_DIR / "fernandez2026_ground_truth_50.csv")
    gt["key"] = list(zip(gt["latitude"].round(4), gt["longitude"].round(4)))
    fd_pred, fd_choice, fd_hist = fd_loso(sites, specs, gt, SEED)
    fd_choice.to_csv(lc.TABLES / "step10_fd_hyperparameters.csv", index=False)
    pred.update(fd_pred)
    pred["band ratio (Lucey theta)"] = loso_linear(lucey_theta(R85, wl), truth)
    pred["mean of other sites"] = np.array([truth[np.arange(len(truth)) != i].mean() for i in range(len(truth))])

    P = pd.DataFrame({"site_id": sid, "mission": sites["mission"], "truth_FeO": truth, **pred})
    P.to_csv(lc.TABLES / "step10_predictions.csv", index=False)
    # which methods use landing-site data for training or calibration
    uses_sites = {m: True for m in pred}
    uses_sites["LUMEN-M3"] = False
    # the re-derived calibration is fitted at the Apollo 16 landing pixel, itself a test site
    mrows = []
    for m, p in pred.items():
        e = p - truth
        mrows.append({"method": m, "n_sites": len(truth), "mae": np.abs(e).mean(),
                      "rmse": np.sqrt(np.mean(e ** 2)),
                      "r2": 1 - np.sum(e ** 2) / np.sum((truth - truth.mean()) ** 2),
                      "bias": e.mean(), "max_abs_err": np.abs(e).max(),
                      "uses_landing_site_data": uses_sites[m]})
    M = pd.DataFrame(mrows).sort_values("mae")
    M.to_csv(lc.TABLES / "step10_metrics.csv", index=False)
    # D23: LUMEN-M3 against every comparator (not its own variants), Holm over comparators
    a = np.abs(pred["LUMEN-M3"] - truth)
    trows = []
    for m in [m for m in pred if not m.startswith("LUMEN")]:
        b = np.abs(pred[m] - truth)
        trows.append({"versus": m, "mae_lumen": a.mean(), "mae_other": b.mean(),
                      "n_lumen_better": int((a < b).sum()), "n_sites": len(a),
                      "wilcoxon_p": wilcoxon(a, b).pvalue if np.any(a != b) else 1.0})
    Tt = pd.DataFrame(trows)
    Tt["holm_p"] = le.holm(Tt["wilcoxon_p"])
    Tt.to_csv(lc.TABLES / "step10_tests.csv", index=False)
    print(P.round(2).to_string(index=False))
    print(M.round(3).to_string(index=False))
    print(Tt.round(4).to_string(index=False))
    figures(P, M)
    lc.done(STAGE, t0)
    return 0


def figures(P, M):
    import matplotlib.pyplot as plt
    show = ["LUMEN-M3", "LUMEN-M3 + site recalibration", "FD-AE-M3/RF",
            M[M["method"].str.startswith("FD")].iloc[0]["method"]]
    show = list(dict.fromkeys(show))
    fig, axs = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.4), sharex=True, sharey=True)
    lo, hi = 0, max(P["truth_FeO"].max(), P[show].max().max()) + 1
    for ax, m in zip(axs, show):
        mm = M.set_index("method").loc[m]
        for (mis, g), mk in zip(P.groupby("mission"), "osD^v<>p*hXP"):
            ax.scatter(g["truth_FeO"], g[m], s=22, marker=mk, color=lp.C[0] if m.startswith("LUMEN") else lp.C[6],
                       label=mis)
        ax.plot([lo, hi], [lo, hi], color=lp.INK2, lw=0.8)
        ax.set_title("%s\nMAE %.2f, R² %.2f" % (m, mm["mae"], mm["r2"]), fontsize=8)
        ax.set_xlabel("returned-sample FeO (wt %)")
    axs[0].set_ylabel("predicted from M3 (wt %)")
    axs[-1].legend(fontsize=6, loc="lower right")
    lp.save(fig, lc.FIGURES / "step10_m3_sites.png")


if __name__ == "__main__":
    sys.exit(main())
