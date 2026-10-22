"""
Step 07 -- LUMEN on Moon Mineralogy Mapper data (METHOD_NOTES.md D19).

  calibration  M3 L2 -> lab (RELAB) scale with Korokhin Eq. 2 as published (primary);
               Eq. 2 re-derived from the Apollo 16 block against LSCC 62231 <45 um as a
               sensitivity check.
  Apollo 16    LUMEN at the landing site vs the measured A16 soils (one site; a sanity
               check, not a validation).
  Shapley      the Korokhin (2024) scene, resampled along-track to their 500 x 304 grid,
               mapped with the final A4 ensemble + conformal 90 % half-widths, only for
               targets that pass the identifiability gate; compared pixel by pixel with
               their ANN maps.
  Aristarchus  the two Jung (2026) ROI blocks: crystalline proportions over all valid
               pixels and over the freshest (top 7 % 1-um band depth) pixels, beside their
               M3 posterior medians (which describe space-weathering-corrected material;
               the comparison is descriptive, there is no ground truth).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

import lumen_common as lc
import lumen_eval as le
import lumen_model as lm
import lumen_plot as lp
from step04_lumen import N_MEMBERS, load_lab

STAGE = "step07"
EQ2 = (1.685, 0.0355)
KOROKHIN_REDERIVED = (1.669, 0.016)
K_PARAMS = ["S", "nFe", "Gr", "CPX", "OPX", "OLV", "PLG", "ILM"]
N_ROWS = 500
A16_LINE, A16_SAMPLE = 30, 177


def load_models():
    syn = np.load(lc.PRODUCTS / "synthetic.npz")
    scaler = lm.FeatureScaler(syn["feat_mean"], syn["feat_std"])
    models = []
    for k in range(N_MEMBERS):
        m = lm.LumenNet()
        m.load_state_dict(torch.load(lc.PRODUCTS / ("final_A4_full_member%d.pt" % k)))
        models.append(m)
    return models, scaler, syn["lam_um"]


def predict(models, scaler, lam, R):
    Xf = scaler(lc.spectral_features(R, lam))
    return lm.ensemble([lm.member_outputs(m, Xf) for m in models])


def to_square_rows(cube, n_rows):
    """Linear along-track resampling (lines, k, samples) -> (n_rows, k, samples), as in the
    Korokhin reproduction (step05_m3_preprocess.to_square_rows)."""
    L = cube.shape[0]
    pos = (np.arange(n_rows) + 0.5) * L / n_rows - 0.5
    i0 = np.clip(np.floor(pos).astype(int), 0, L - 1)
    i1 = np.clip(i0 + 1, 0, L - 1)
    f = (pos - i0)[:, None, None]
    return (1 - f) * cube[i0] + f * cube[i1]


def pixels(cube):
    """(lines, 85, samples) -> (lines*samples, 71 input bands), invalid rows flagged."""
    R = np.transpose(np.asarray(cube, float), (0, 2, 1)).reshape(-1, cube.shape[1])[:, lc.input_band_index()]
    ok = np.isfinite(R).all(1) & (R > 0).all(1) & (R < 1.0).all(1)
    return R, ok


def flat_outputs(ens):
    out = {}
    for j, t in enumerate(lc.COMP):
        out[t] = ens["comp_mean"][:, j]
        out["sd_" + t] = np.sqrt(ens["comp_var"][:, j])
    for j, t in enumerate(lm.SCALARS):
        out[t] = ens["scal_mean"][:, j]
        out["sd_" + t] = np.sqrt(ens["scal_var"][:, j])
    return out


def main() -> int:
    t0 = lc.banner(STAGE, "LUMEN on M3: calibration, Apollo 16, Shapley maps, Aristarchus")
    lc.verify_manifest(STAGE)
    lp.setup()
    torch.set_num_threads(4)
    models, scaler, lam = load_models()
    X, meta = load_lab()
    gate = pd.read_csv(lc.TABLES / "step05_identifiability_gate.csv", index_col=0)
    q = pd.read_csv(lc.TABLES / "step05_conformal_quantiles.csv")
    q90 = q[q["level"] == 0.90].set_index("target")["q_normalised"]
    mapped = [t for t in le.TARGETS if bool(gate.loc[t, "identifiable"])]
    print("  identifiable (mapped): %s" % ", ".join(mapped))
    print("  withheld by the gate:  %s" % ", ".join(t for t in le.TARGETS if t not in mapped))

    # ---- calibration ------------------------------------------------------------
    a16 = np.load(lc.M3_SHAPLEY_DIR / "apollo16_rfl.npy")
    win = a16[A16_LINE - 1:A16_LINE + 2, :, A16_SAMPLE - 1:A16_SAMPLE + 2]
    m3_site = np.transpose(win, (0, 2, 1)).reshape(-1, a16.shape[1])[:, lc.input_band_index()].mean(0)
    i62231 = int(np.flatnonzero((meta["soil"] == "62231") & (meta["fraction"] == "<45"))[0])
    lab_site = X[i62231]
    slope, icpt = np.polyfit(m3_site, lab_site, 1)
    cal = pd.DataFrame([
        {"calibration": "Korokhin Eq. 2 (published, primary)", "slope": EQ2[0], "intercept": EQ2[1]},
        {"calibration": "Korokhin reproduction re-derivation", "slope": KOROKHIN_REDERIVED[0],
         "intercept": KOROKHIN_REDERIVED[1]},
        {"calibration": "re-derived here (A16 3x3 vs 62231 <45 um, 71 bands)", "slope": slope,
         "intercept": icpt},
    ])
    for i, r in cal.iterrows():
        cal.loc[i, "rms_rel_err_vs_62231_pct"] = 100 * np.sqrt(np.mean(((r["slope"] * m3_site + r["intercept"]) - lab_site) ** 2 / lab_site ** 2))
    cal.to_csv(lc.TABLES / "step07_calibration.csv", index=False)
    print(cal.round(4).to_string(index=False))

    # ---- Apollo 16 site ---------------------------------------------------------
    rows = []
    for _, r in cal.iterrows():
        e = flat_outputs(predict(models, scaler, lam, (r["slope"] * m3_site + r["intercept"])[None]))
        rows.append({"source": "LUMEN on M3 (%s)" % r["calibration"], **{t: float(e[t][0]) for t in le.TARGETS},
                     **{"sd_" + t: float(e["sd_" + t][0]) for t in le.TARGETS}})
    e = flat_outputs(predict(models, scaler, lam, lab_site[None]))
    rows.append({"source": "LUMEN on lab 62231 <45 um (in training set)", **{t: float(e[t][0]) for t in le.TARGETS}})
    rows.append({"source": "measured 62231 <45 um", **meta.loc[i62231, le.TARGETS].to_dict()})
    a16m = meta[(meta["mission"] == "A16") & (meta["fraction"] == "<45")]
    rows.append({"source": "measured A16 soils <45 um, mean of %d" % len(a16m), **a16m[le.TARGETS].mean().to_dict()})
    site = pd.DataFrame(rows)
    site.to_csv(lc.TABLES / "step07_apollo16_site.csv", index=False)
    print(site[["source"] + lc.CHEM + [lc.MAT, "PLG", "AGG"]].round(2).to_string(index=False))

    # ---- Shapley scene ----------------------------------------------------------
    raw = np.load(lc.M3_SHAPLEY_DIR / "scene_rfl.npy")
    sq = to_square_rows(raw, N_ROWS)                                       # (500, 85, 304)
    R, ok = pixels(sq)
    R = EQ2[0] * R + EQ2[1]
    e = flat_outputs(predict(models, scaler, lam, np.where(ok[:, None], R, 0.1)))
    shape = (N_ROWS, sq.shape[2])
    maps = {}
    for t in le.TARGETS:
        if t not in mapped:
            continue
        maps[t] = np.where(ok, e[t], np.nan).reshape(shape).astype(np.float32)
        maps["sd_" + t] = np.where(ok, e["sd_" + t], np.nan).reshape(shape).astype(np.float32)
        maps["hw90_" + t] = (q90[t] * maps["sd_" + t]).astype(np.float32)
    np.savez_compressed(lc.PRODUCTS / "step07_shapley_maps.npz", valid=ok.reshape(shape), **maps)
    stats = []
    high = np.load(lc.BASELINE_DIR / "korokhin2024" / "highland_mask.npy")
    for t in mapped:
        v = maps[t][ok.reshape(shape)]
        stats.append({"target": t, "median": np.median(v), "p05": np.percentile(v, 5), "p95": np.percentile(v, 95),
                      "median_hw90": np.nanmedian(maps["hw90_" + t]),
                      "mean_mare": np.nanmean(maps[t][~high]), "mean_highland": np.nanmean(maps[t][high])})
    stats = pd.DataFrame(stats)
    stats.to_csv(lc.TABLES / "step07_shapley_stats.csv", index=False)
    print("  Shapley scene: %d / %d valid pixels" % (ok.sum(), ok.size))
    print(stats.round(3).to_string(index=False))

    # ---- comparison with Korokhin's ANN maps ------------------------------------
    K = np.load(lc.BASELINE_DIR / "korokhin2024" / "ann_full_median_iter3.npy").reshape(N_ROWS, -1, 8)
    kp = {p: K[:, :, i] for i, p in enumerate(K_PARAMS)}
    ksum = sum(kp[m] for m in ["CPX", "OPX", "OLV", "PLG", "ILM"])
    comp_rows = []
    lum5 = None
    if all(c in maps for c in ["HCP", "LCP", "OLV", "PLG", "ILM"]):
        lsum = maps["HCP"] + maps["LCP"] + maps["OLV"] + maps["PLG"] + maps["ILM"]
        lum5 = {"CPX": maps["HCP"], "OPX": maps["LCP"], "OLV": maps["OLV"], "PLG": maps["PLG"], "ILM": maps["ILM"]}
        lum5 = {k: 100 * v / lsum for k, v in lum5.items()}
    pairs = [("CPX", "HCP"), ("OPX", "LCP"), ("OLV", "OLV"), ("PLG", "PLG"), ("ILM", "ILM"),
             ("nFe", lc.MAT), ("ILM", "TiO2"), ("PLG", "Al2O3"), ("S", lc.MAT)]
    valid = ok.reshape(shape) & np.isfinite(K).all(-1)
    for kname, lname in pairs:
        if lname not in maps:
            comp_rows.append({"korokhin": kname, "lumen": lname, "status": "LUMEN target withheld by the gate"})
            continue
        if kname in ["CPX", "OPX", "OLV", "PLG", "ILM"] and lname in lc.COMP and lum5 is not None:
            kimg, limg, unit = 100 * kp[kname] / ksum, lum5[kname], "5-mineral proportion (%)"
        else:
            kimg, limg, unit = kp[kname], maps[lname], "raw"
        a, b = kimg[valid], limg[valid]
        comp_rows.append({"korokhin": kname, "lumen": lname, "status": "compared", "unit": unit,
                          "pearson": pearsonr(a, b)[0], "spearman": spearmanr(a[::7], b[::7])[0],
                          "mean_korokhin": a.mean(), "mean_lumen": b.mean(),
                          "mare_minus_highland_korokhin": kimg[valid & ~high].mean() - kimg[valid & high].mean(),
                          "mare_minus_highland_lumen": limg[valid & ~high].mean() - limg[valid & high].mean()})
    kc = pd.DataFrame(comp_rows)
    kc.to_csv(lc.TABLES / "step07_vs_korokhin_maps.csv", index=False)
    print(kc.round(3).to_string(index=False))

    # ---- Aristarchus ROIs -------------------------------------------------------
    jm = pd.read_csv(lc.BASELINE_DIR / "jung2026" / "step06_m3_abundances.csv")
    arows = []
    i097 = int(np.argmin(np.abs(lam - 0.97)))
    for roi in ("roi_b", "roi_c"):
        blk = np.load(lc.M3_ARISTARCHUS_DIR / ("%s_rfl_block.npy" % roi))
        Rb, okb = pixels(blk)
        Rb = EQ2[0] * Rb + EQ2[1]
        Rv = Rb[okb]
        eb = flat_outputs(predict(models, scaler, lam, Rv))
        cr = lc.continuum_removed_fast(lam, Rv)
        depth = 1 - cr[:, i097]
        fresh = depth >= np.percentile(depth, 93)
        tot = eb["PLG"] + eb["LCP"] + eb["HCP"] + eb["OLV"] + eb["ILM"]
        cry = {"olivine": 100 * eb["OLV"] / tot, "plagioclase": 100 * eb["PLG"] / tot,
               "pyroxene_total": 100 * (eb["LCP"] + eb["HCP"]) / tot, "ilmenite": 100 * eb["ILM"] / tot}
        gmap = {"olivine": "OLV", "plagioclase": "PLG", "pyroxene_total": "HCP", "ilmenite": "ILM"}
        for cat, v in cry.items():
            jr = jm[(jm["roi"] == roi) & (jm["category"] == cat)].iloc[0]
            ident = bool(gate.loc[gmap[cat], "identifiable"]) and (cat != "pyroxene_total" or bool(gate.loc["LCP", "identifiable"]))
            arows.append({"roi": roi, "category": cat, "n_valid": int(okb.sum()), "n_fresh": int(fresh.sum()),
                          "lumen_median_all": np.median(v), "lumen_p05_all": np.percentile(v, 5),
                          "lumen_p95_all": np.percentile(v, 95), "lumen_median_fresh": np.median(v[fresh]),
                          "jung_median_wt": jr["median_wt_pct"], "jung_p2.5": jr["p2.5_wt_pct"],
                          "jung_p97.5": jr["p97.5_wt_pct"], "lumen_parts_identifiable": ident})
        for t in (lc.CHEM + [lc.MAT]):
            arows.append({"roi": roi, "category": t, "n_valid": int(okb.sum()), "n_fresh": int(fresh.sum()),
                          "lumen_median_all": np.median(eb[t]), "lumen_p05_all": np.percentile(eb[t], 5),
                          "lumen_p95_all": np.percentile(eb[t], 95), "lumen_median_fresh": np.median(eb[t][fresh]),
                          "lumen_parts_identifiable": bool(gate.loc[t, "identifiable"])})
    ar = pd.DataFrame(arows)
    ar.to_csv(lc.TABLES / "step07_aristarchus.csv", index=False)
    print(ar.round(2).to_string(index=False))

    figures(maps, mapped, kp, ksum, valid, lum5)
    lc.done(STAGE, t0)
    return 0


def figures(maps, mapped, kp, ksum, valid, lum5):
    import matplotlib.pyplot as plt
    show = [t for t in ["FeO", "TiO2", "Al2O3", lc.MAT, "PLG", "AGG"] if t in mapped][:4]
    if not show:
        return
    fig, axs = plt.subplots(2, len(show), figsize=(3.2 * len(show), 7.5))
    axs = np.atleast_2d(axs).reshape(2, -1)
    for j, t in enumerate(show):
        im = axs[0, j].imshow(maps[t], cmap="viridis", aspect="auto",
                              vmin=np.nanpercentile(maps[t], 1), vmax=np.nanpercentile(maps[t], 99))
        axs[0, j].set_title(t); plt.colorbar(im, ax=axs[0, j], fraction=0.046)
        im = axs[1, j].imshow(maps["hw90_" + t], cmap="magma", aspect="auto",
                              vmin=np.nanpercentile(maps["hw90_" + t], 1), vmax=np.nanpercentile(maps["hw90_" + t], 99))
        axs[1, j].set_title("90 % conformal half-width"); plt.colorbar(im, ax=axs[1, j], fraction=0.046)
        for a in axs[:, j]:
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
    lp.save(fig, lc.FIGURES / "step07_shapley_maps.png")
    if lc.MAT in maps:
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ax.hexbin(kp["nFe"][valid], maps[lc.MAT][valid], gridsize=60, cmap="Blues", bins="log")
        ax.set_xlabel("Korokhin nFe (ANN)"); ax.set_ylabel("LUMEN log10(Is/FeO)")
        lp.save(fig, lc.FIGURES / "step07_vs_korokhin_maturity.png")


if __name__ == "__main__":
    sys.exit(main())
