"""
Step 12 -- orbital fine-tuning on the M3 landing sites (METHOD_NOTES.md Addendum C,
D26-D29). FeO, leave-one-site-out over the 18 sites, against the step10 comparators.

  LUMEN-M3-FT      the step10 A4 member, fine-tuned 600 steps on M3 pixel spectra of the
                   other 17 sites (FeO only) mixed 1:1 with its own laboratory training
                   spectra (all targets);
  LUMEN-embed+ridge  ridge regression on the members' 128-d trunk embedding of the pixel
                   spectra, alpha by inner leave-one-site-out.
"""
from __future__ import annotations

import math
import sys
import zlib
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge

import lumen_common as lc
import lumen_eval as le
import lumen_model as lm
import lumen_plot as lp
import step04_lumen as s4
import step10_m3_sites as s10

STAGE = "step12"
STEPS, LR, WD, N_M3, N_LAB = 600, 1e-4, 1e-4, 16, 16
ALPHAS = [0.1, 1.0, 10.0, 100.0]
_G = {}


def pixel_spectra():
    sites = pd.read_csv(s10.SITES / "sites.csv")
    strips = pd.read_csv(s10.SITES / "site_strips.csv")
    blocks = np.load(s10.SITES / "site_strip_blocks.npy")
    R, sid = [], []
    for _, r in strips[strips["kept"]].iterrows():
        px = np.transpose(blocks[int(r["block"])], (0, 2, 1)).reshape(9, 85)
        sub = px[:, s10.FD_BANDS]
        ok = np.isfinite(sub).all(1) & (sub > 0).all(1) & (sub < 1).all(1)
        if ok.sum() >= 5:
            R.append(px[ok])
            sid += [r["site_id"]] * int(ok.sum())
    return sites, np.vstack(R), np.asarray(sid)


def model_fold(mission):
    return s10.MISSION_FOLD.get(mission, "FINAL")


def state_path(fold, k):
    return lc.PRODUCTS / (("final_A4_full_member%d.pt" % k) if fold == "FINAL"
                          else ("lomo_A4_full_%s_member%d.pt" % (fold, k)))


def _init():
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    _G["scaler"] = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    _G["lam"] = syn["lam_um"]
    _G["X"], _G["meta"] = s4.load_lab()
    _G["sites"], _G["pix"], _G["pix_site"] = pixel_spectra()
    sites, specs, _ = s10.site_spectra()
    _G["site_spec"] = {s: specs[s][0] for s in specs}
    torch.set_num_threads(1)


def to_lab(R85):
    return s10.EQ2[0] * np.atleast_2d(R85)[:, lc.input_band_index()] + s10.EQ2[1]


def _job(job):
    site, k = job
    S, meta, X = _G["sites"], _G["meta"], _G["X"]
    srow = S[S["site_id"] == site].iloc[0]
    fold = model_fold(srow["mission"])
    seed = zlib.crc32(("m3ft|%s|%d" % (site, k)).encode()) % (2 ** 31)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = lm.LumenNet()
    model.load_state_dict(torch.load(state_path(fold, k)))
    # data: M3 pixels of the other sites; lab spectra of this model's training soils
    tr = _G["pix_site"] != site
    Rm = to_lab(_G["pix"][tr])
    feo_site = S.set_index("site_id")["feo_mean"]
    ym = feo_site.loc[_G["pix_site"][tr]].to_numpy(float)
    lab_idx = np.flatnonzero(meta["mission"] != fold) if fold != "FINAL" else np.arange(len(meta))
    real = s4.real_dict(X, meta, lab_idx)
    jm = rng.integers(0, len(Rm), STEPS * N_M3)
    jl = rng.integers(0, len(lab_idx), STEPS * N_LAB)
    R_all = lm.output_randomisation(np.vstack([Rm[jm], real["R"][jl]]), rng, _G["lam"])
    Xf = _G["scaler"](lc.spectral_features(R_all, _G["lam"]))
    scal_m = np.full((len(jm), len(lm.SCALARS)), np.nan)
    scal_m[:, 0] = ym[jm]
    scal_l = np.column_stack([real["chem"][jl], real["logIsFeO"][jl]])
    scal = np.vstack([scal_m, scal_l]).astype(np.float32)
    mask = np.isfinite(scal)
    scal = lm.scale_scalars(np.nan_to_num(scal))
    comp = np.vstack([np.full((len(jm), lm.N_COMP), np.nan), real["comp"][jl]]).astype(np.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    model.train()
    nm = len(jm)
    for s in range(STEPS):
        idx = np.r_[np.arange(s * N_M3, (s + 1) * N_M3), nm + np.arange(s * N_LAB, (s + 1) * N_LAB)]
        b = lm.to_tensor_batch(Xf[idx], comp[idx], scal[idx], mask[idx])
        loss = lm.loss_fn(model(b["x"]), b, comp_l1=lm.COMP_L1, aux_weight=0.0)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
    o = lm.member_outputs(model, _G["scaler"](lc.spectral_features(to_lab(_G["site_spec"][site]), _G["lam"])))
    return site, k, float(o["scal_mean"][0, 0]), float(o["scal_var"][0, 0])


@torch.no_grad()
def embeddings(models, R):
    Xf = torch.from_numpy(_G["scaler"](lc.spectral_features(to_lab(R), _G["lam"])))
    out = []
    for m in models:
        m.eval()
        out.append(m.trunk(m.body(Xf)).numpy())
    return np.mean(out, axis=0)


def embed_ridge(sites):
    models, _, _ = s10.lumen_models()
    pred = {}
    for _, s in sites.iterrows():
        site = s["site_id"]
        ms = models[model_fold(s["mission"])]
        tr = _G["pix_site"] != site
        E = embeddings(ms, _G["pix"][tr])
        y = sites.set_index("site_id")["feo_mean"].loc[_G["pix_site"][tr]].to_numpy(float)
        g = _G["pix_site"][tr]
        best, ba = np.inf, ALPHAS[0]
        for a in ALPHAS:                                   # inner leave-one-site-out
            err = []
            for gs in np.unique(g):
                m = g != gs
                r = Ridge(alpha=a).fit(E[m], y[m])
                err.append(abs(r.predict(E[~m]).mean() - y[~m][0]))
            if np.mean(err) < best:
                best, ba = np.mean(err), a
        r = Ridge(alpha=ba).fit(E, y)
        pred[site] = float(r.predict(embeddings(ms, _G["site_spec"][site][None]))[0])
    return pred


def main() -> int:
    t0 = lc.banner(STAGE, "orbital fine-tuning on the M3 landing sites (Addendum C)")
    lc.verify_manifest(STAGE)
    lp.setup()
    _init()
    sites = _G["sites"]
    print("  M3 pixel spectra: %d over %d sites" % (len(_G["pix"]), len(np.unique(_G["pix_site"]))))
    jobs = [(s, k) for s in sites["site_id"] for k in range(s4.N_MEMBERS)]
    with get_context(lc.POOL_START).Pool(4, initializer=_init) as pool:
        res = pool.map(_job, jobs)
    r = pd.DataFrame(res, columns=["site_id", "member", "FeO", "var"])
    ft = r.groupby("site_id")["FeO"].mean()
    torch.set_num_threads(4)
    er = embed_ridge(sites)

    P = pd.read_csv(lc.TABLES / "step10_predictions.csv")
    P["LUMEN-M3-FT (orbital fine-tuning)"] = P["site_id"].map(ft)
    P["LUMEN-embed + ridge"] = P["site_id"].map(er)
    P.to_csv(lc.TABLES / "step12_predictions.csv", index=False)
    truth = P["truth_FeO"].to_numpy()
    methods = [c for c in P.columns if c not in ("site_id", "mission", "truth_FeO")]
    M = pd.DataFrame([{"method": m, "mae": np.abs(P[m] - truth).mean(),
                       "rmse": np.sqrt(np.mean((P[m] - truth) ** 2)),
                       "r2": 1 - np.sum((P[m] - truth) ** 2) / np.sum((truth - truth.mean()) ** 2),
                       "bias": (P[m] - truth).mean()} for m in methods]).sort_values("mae")
    M.to_csv(lc.TABLES / "step12_metrics.csv", index=False)
    a = np.abs(P["LUMEN-M3-FT (orbital fine-tuning)"] - truth)
    rows = []
    for m in [m for m in methods if not m.startswith("LUMEN")]:
        b = np.abs(P[m] - truth)
        rows.append({"versus": m, "mae_ft": a.mean(), "mae_other": b.mean(), "n_ft_better": int((a < b).sum()),
                     "wilcoxon_p": wilcoxon(a, b).pvalue})
    Tt = pd.DataFrame(rows)
    Tt["holm_p"] = le.holm(Tt["wilcoxon_p"])
    Tt.to_csv(lc.TABLES / "step12_tests.csv", index=False)
    fd = M[M["method"].str.startswith("FD")]
    beats = bool(M.set_index("method").loc["LUMEN-M3-FT (orbital fine-tuning)", "mae"] < fd["mae"].min())
    print(M.round(3).to_string(index=False))
    print(Tt.round(4).to_string(index=False))
    print("  LUMEN-M3-FT beats every FD variant on MAE: %s" % beats)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4), sharex=True, sharey=True)
    for ax, m in zip(axs, ["LUMEN-M3-FT (orbital fine-tuning)", fd.iloc[0]["method"]]):
        ax.scatter(truth, P[m], s=22, color=lp.C[0] if m.startswith("LUMEN") else lp.C[6])
        ax.plot([0, 24], [0, 24], color=lp.INK2, lw=0.8)
        mm = M.set_index("method").loc[m]
        ax.set_title("%s\nMAE %.2f, R² %.2f" % (m, mm["mae"], mm["r2"]), fontsize=8)
        ax.set_xlabel("returned-sample FeO (wt %)")
    axs[0].set_ylabel("predicted from M3 (wt %)")
    lp.save(fig, lc.FIGURES / "step12_m3_finetune.png")
    lc.done(STAGE, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
