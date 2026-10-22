"""
Step 09 -- the maps (Figs. 19-21) and the comparisons with other methods (Sec. 5.7).

Maps: the pixel-wise median over the five seeds of the iteration-3 ANN outputs.

Histograms (Fig. 20): 60 bins between the 0.5th and 99.5th percentiles, 3-bin
running mean; peaks by scipy.signal.find_peaks with prominence >= 10 % of the
maximum. "Bimodal" = at least two such peaks (D16).

Comparisons (Sec. 5.7), on M3 reduced to the RELAB system (Eq. 2), from the
recipe-filtered native cube of step05:
  * MLR, Eqs. (3)-(5). The paper prints no coefficients (D15). Three variants, all
    reported (tables/step09_correlations_variants.csv):
      published -- Eq. (3) IS Eq. (1) of Shkuratov et al. (2005a) and Eq. (2) of
                   Pieters et al. (2006), the two sources the paper cites for it:
                   log10 P = a A(415) + b A(750) + c A(900) + d A(1000) + e, A in %.
                   Shkuratov 2005a Table 3: CPX, PLG, Is/FeO; Pieters 2006 Table 3:
                   Is, OPx, augite, Is/FeO. The reading (log10, coefficient order) is
                   confirmed on the LSCC spectra before use
                   (tables/step09_mlr_published_check.csv).
      refit_all -- least squares on all 19 LSCC soils, mare (Taylor et al. 2001) + highland
                   (Taylor et al. 2010): 57 fractions. The paper's MLRs are fitted on
                   "the LSCC data (Pieters et al., 2000, 2002, 2006; Taylor et al.,
                   2001a, 2001b)", which include the highland soils.
      refit_mare -- the same on the 9 mare soils only.
    Primary per target: published where it exists -- Is and OPX (Pieters 2006), CPX
    and PLG (Shkuratov 2005a) -- refit_all otherwise (OLV by Eq. 5, d by Eq. 4, for
    which neither source prints coefficients). Is/FeO (both sources) and augite are
    reported beside Is and CPX.
    LSCC bands: Gaussian filters at the Clementine centres and widths (415/40,
    750/10, 900/30, 950/30, 1000/30 nm). M3 bands: A(415) extrapolated linearly from
    the first six M3 channels, 540.84-730.48 nm, as the paper does; the others
    interpolated from the native channels.
  * OMAT_Fe (Eq. 6) and TiO2 (Eq. 7), Lucey et al., closed form.
  * Fig. 31: ilmenite by the 1550-nm band depth of Surkov et al. (2020), reprinted
    as ch. 4 of Surkov's thesis: continuum = the upper convex hull of the spectrum
    (native bands 541-2577 nm, M3 L2 reflectance as Surkov used it), S = 1 - A/A_CH,
    BD = mean S at 1489, 1509, 1529, 1549, 1579 nm (Eq. 4.11), ilmenite wt% =
    202.8 BD (Eq. 4.12).
Correlation coefficients are Pearson's, per mare / highland branch (step05 mask).
  * Fig. 30: TiO2 by the "moderately non-linear" ANN of Korokhin et al. (2008),
    retrained on that paper's own Table 1 (korokhin2008_tio2.py; its published weights
    lack bias terms and do not reproduce the published correlations -- reported, not
    used), applied to the M3 Clementine-like bands as fractions. Seed median of 5.

Usage: python step09_maps_comparisons.py [--tag fast|full]
"""
from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import korokhin_common as kc
import korokhin2008_tio2 as k08
from step05_m3_preprocess import to_square_rows

STAGE = "step09"
CLEM = {415: 40.0, 750: 10.0, 900: 30.0, 950: 30.0, 1000: 30.0}     # centre: FWHM, nm
SOILS = ["10084", "12001", "12030", "15041", "15071", "70181", "71061", "71501", "79221"]
HIGHLAND_SOILS = ["14141", "14163", "14259", "14260", "61141", "61221", "62231", "64801", "67461", "67481"]
FRACTIONS = {"<10": 5.0, "10-20": 15.0, "20-45": 32.5}
# Shkuratov, Kaydash & Pieters (2005a) Table 3: a, b, c, d, e of Eq. (1) for A(415), A(750),
# A(900), A(1000) in %, constant last.
SHK2005 = {"CPX": (-0.129, 0.244, 0.147, -0.321, 1.27),
           "PLG": (0.114, -0.126, -0.021, 0.093, 1.064),
           "IsFeO": (0.006, -0.224, 0.138, 0.06, 1.944)}
# Pieters et al. (2006) Table 3: b1..b5 of their Eq. (2) -- the paper's Eq. (3) -- same form.
PIETERS2006 = {"Is": (-0.032, -0.163, 0.204, -0.055, 3.240),
               "OPX": (-0.039, 0.320, -0.784, 0.469, 0.271),
               "Augite": (-0.191, 0.211, 0.616, -0.727, 1.206),
               "IsFeO_P06": (-0.0012, -0.219, 0.097, 0.099, 1.945)}
PUBLISHED = {**SHK2005, **PIETERS2006}
PUBLISHED_K = {"CPX": 0.93, "PLG": 0.93, "IsFeO": 0.88, "Is": 0.92, "OPX": 0.51, "Augite": 0.93, "IsFeO_P06": 0.92}
PUBLISHED_SRC = {"CPX": "Shkuratov 2005a", "PLG": "Shkuratov 2005a", "IsFeO": "Shkuratov 2005a",
                 "Is": "Pieters 2006", "OPX": "Pieters 2006", "Augite": "Pieters 2006", "IsFeO_P06": "Pieters 2006"}
BD_NM = (1489.0, 1509.0, 1529.0, 1549.0, 1579.0)                  # Surkov 2020, Eq. 4.11
BD_TO_ILM = 202.8                                                  # Surkov 2020, Eq. 4.12
RANGES = {"S": (0, 80), "nFe": (0.03, 0.10), "Gr": (31.4, 32.4), "CPX": (20, 40), "OPX": (1.35, 1.60),
          "OLV": (1.2, 1.7), "PLG": (18, 32), "ILM": (5, 20)}       # the paper's Fig. 19 scales


def ann_maps(tag: str):
    ps = sorted(kc.PRODUCTS.glob("ann_%s_seed*_iter3_params.npy" % tag))
    if not ps:
        sys.exit("[%s] FATAL: no iteration-3 ANN maps for tag %s -- run step07" % (STAGE, tag))
    return np.median(np.stack([np.load(p).astype(float) for p in ps]), axis=0), len(ps)


def histogram_peaks(v: np.ndarray):
    lo, hi = np.nanpercentile(v, [0.5, 99.5])
    h, e = np.histogram(v[np.isfinite(v)], bins=60, range=(lo, hi))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    pk, prop = find_peaks(np.r_[0, hs, 0], prominence=0.10 * hs.max())
    centres = 0.5 * (e[:-1] + e[1:])
    pk = pk - 1
    order = np.argsort(prop["prominences"])[::-1]
    peaks = sorted(centres[pk[order[:2]]]) if len(pk) else []
    return h, e, peaks, len(pk)


def native_recipe():
    """step05's recipe-filtered native cube (lines, bands <= 2.6 um, samples), L2 units."""
    return (np.load(kc.PRODUCTS / "scene_native_recipe.npy"),
            pd.read_csv(kc.PRODUCTS / "scene_native_recipe_wl.csv")["wavelength_nm"].to_numpy())


def m3_clementine(highland_shape):
    native, wl = native_recipe()
    use = np.flatnonzero((wl >= 540) & (wl <= 1030))
    bands = native[:, use, :].astype(float)
    bands = to_square_rows(kc.EQ2_SLOPE * bands + kc.EQ2_INTERCEPT, highland_shape[0]) * 100.0  # %
    w = wl[use]
    first6 = np.arange(6)                                       # 540.84 ... 730.48 nm
    x = w[first6]
    y = bands[:, first6, :]
    xm = x.mean()
    slope = ((x - xm)[None, :, None] * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / ((x - xm) ** 2).sum()
    a415 = y.mean(axis=1) + slope * (415.0 - xm)

    def at(nm):
        i = np.searchsorted(w, nm)
        if w[i] == nm:
            return bands[:, i, :]
        f = (nm - w[i - 1]) / (w[i] - w[i - 1])
        return (1 - f) * bands[:, i - 1, :] + f * bands[:, i, :]
    return {415: a415, 750: at(750.0), 900: at(900.0), 950: at(950.0), 1000: at(1000.0)}


def lscc_clementine(lam_um, refl):
    out = {}
    for c, fw in CLEM.items():
        sig = fw / 1000.0 / 2.3548
        wgt = np.exp(-0.5 * ((lam_um - c / 1000.0) / sig) ** 2)
        out[c] = 100.0 * np.nansum(wgt * refl) / np.sum(wgt[np.isfinite(refl)])
    return out


def design(A: dict, eq: int) -> np.ndarray:
    AR = A[750]
    CBR, CIR1, CIR2, CIR3 = A[415] / A[750], A[900] / A[750], A[950] / A[750], A[1000] / A[750]
    D = A[750] * A[950] / A[900] ** 2
    if eq == 3:
        cols = [A[415], A[750], A[900], A[1000]]
    elif eq == 4:
        cols = [AR, CBR, CIR1, CIR2, D]
    else:
        cols = [AR, CBR, CIR1, CIR2, CIR3, D]
    cols = [np.asarray(c, float) for c in cols]
    return np.stack([np.ones_like(cols[0])] + cols, axis=-1)


def lscc_training():
    gt = kc.GROUND_TRUTH
    modal = pd.concat([pd.read_csv(gt / "lscc_modal_mineralogy.csv"),
                       pd.read_csv(gt / "lscc_highland_modal_mineralogy.csv")])
    isfeo = pd.concat([pd.read_csv(gt / "lscc_is_feo.csv"), pd.read_csv(gt / "lscc_highland_is_feo.csv")])
    chem = pd.concat([pd.read_csv(gt / "lscc_bulk_chemistry.csv"),
                      pd.read_csv(gt / "lscc_highland_bulk_chemistry.csv")])
    rows = []
    for soil in SOILS + HIGHLAND_SOILS:
        kd = kc.read_kdata(soil)
        for frac, dmid in FRACTIONS.items():
            m = modal[(modal["sample"].astype(str) == soil) & (modal["size_fraction_um"] == frac)]
            if m.empty or frac not in kd:
                continue
            v = m.drop_duplicates("phase").set_index("phase")["modal_pct"]
            i = isfeo[(isfeo["sample"].astype(str) == soil) & (isfeo["size_fraction_um"] == frac)]
            c = chem[(chem["sample"].astype(str) == soil) & (chem["size_fraction_um"] == frac)]
            isf = float(i["is_feo"].iloc[0]) if len(i) else np.nan
            row = {"soil": soil, "set": "mare" if soil in SOILS else "highland", "fraction": frac, "d": dmid,
                   "CPX": v.get("mg_clinopyroxene", np.nan) + v.get("fe_clinopyroxene", np.nan),
                   "OPX": v.get("orthopyroxene", np.nan), "PLG": v.get("plagioclase", np.nan),
                   "OLV": v.get("olivine", np.nan), "IsFeO": isf, "Augite": v.get("mg_clinopyroxene", np.nan),
                   "Is": isf * float(c["FeO"].iloc[0]) if len(c) else np.nan}
            row.update({"A%d" % k: val for k, val in lscc_clementine(kd["lam_um"].to_numpy(),
                                                                       kd[frac].to_numpy()).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def published_mlr(A: dict, key: str, base: float = 10.0) -> np.ndarray:
    a, b, c, d, e = PUBLISHED[key]
    return base ** (a * A[415] + b * A[750] + c * A[900] + d * A[1000] + e)


def check_published(lscc: pd.DataFrame) -> pd.DataFrame:
    """Confirms the reading of the published coefficients on the LSCC spectra: log base
    (the one with the smaller median |log error| is used) and the correlation, against
    the k each source gives on its own sample set (PUBLISHED_K)."""
    A = {k: lscc["A%d" % k].to_numpy() for k in CLEM}
    rows = []
    truth = {"CPX": "CPX", "PLG": "PLG", "IsFeO": "IsFeO", "Is": "Is", "OPX": "OPX", "Augite": "Augite",
             "IsFeO_P06": "IsFeO"}
    for key, col in truth.items():
        r_pub = PUBLISHED_K[key]
        t = lscc[col].to_numpy()
        ok = np.isfinite(t) & (t > 0)
        for base in (10.0, np.e):
            pred = published_mlr(A, key, base)
            rows.append({"target": key, "source": PUBLISHED_SRC[key], "truth_column": col,
                         "log_base": "10" if base == 10.0 else "e", "n": int(ok.sum()),
                         "r_log_pred_vs_log_true": float(np.corrcoef(np.log(pred[ok]), np.log(t[ok]))[0, 1]),
                         "r_published": r_pub,
                         "median_abs_log10_error": float(np.median(np.abs(np.log10(pred[ok] / t[ok])))),
                         "median_pred": float(np.median(pred[ok])), "median_true": float(np.median(t[ok]))})
    return pd.DataFrame(rows)


def upper_hull_depth(spec: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(n_pix, n_band) -> S = 1 - A / A_CH with A_CH the upper convex hull (Eq. 4.10)."""
    out = np.empty_like(spec)
    for p in range(spec.shape[0]):
        y = spec[p]
        hull = [0]
        for i in range(1, len(x)):
            while len(hull) >= 2:
                i0, i1 = hull[-2], hull[-1]
                if (y[i1] - y[i0]) * (x[i] - x[i0]) <= (y[i] - y[i0]) * (x[i1] - x[i0]):
                    hull.pop()
                else:
                    break
            hull.append(i)
        out[p] = 1.0 - y / np.interp(x, x[hull], y[hull])
    return out


def ilmenite_band_depth(n_rows: int) -> np.ndarray:
    """Surkov et al. (2020) ilmenite (wt%) per scene pixel, (n_rows * 304,)."""
    native, wl = native_recipe()
    sq = to_square_rows(native.astype(float), n_rows)                 # (rows, bands, samples)
    spec = np.moveaxis(sq, 1, 2).reshape(-1, wl.size)
    S = upper_hull_depth(spec, wl)
    cols = [int(np.argmin(np.abs(wl - b))) for b in BD_NM]
    if np.max(np.abs(wl[cols] - np.array(BD_NM))) > 1.0:
        sys.exit("[%s] FATAL: M3 bands do not match the Surkov 2020 band-depth channels" % STAGE)
    return BD_TO_ILM * S[:, cols].mean(axis=1)


def korokhin2008_map(A: dict, lscc: pd.DataFrame):
    """Fig. 30: TiO2 map (seed median of the retrained moderate net) + the check table."""
    df = k08.table1()
    shape = k08.check_table1(df)
    if not (shape["rows"] == 76 and shape["soils"] == 19 and shape["size_order_ok"] and shape["albedo_in_0_1"]):
        sys.exit("[%s] FATAL: Korokhin 2008 Table 1 transcription malformed: %s" % (STAGE, shape))
    # the standardization reading is fixed by Table 2 itself: for AR and CBR, x0 is exactly
    # the mid-range and xk = 2 / range of the Table 1 values, i.e. X = (x - x0) * xk in
    # [-1, 1] with AR a fraction. (For CIR1-3 and D they are not -- those were set on some
    # other sample set; the printed values are used as printed, and the mismatch reported.)
    P1 = k08.spectral_params(*(df[c].to_numpy() for c in ("A415", "A750", "A900", "A950", "A1000")))
    mid = (P1.max(axis=0) + P1.min(axis=0)) / 2
    rng2 = 2 / (P1.max(axis=0) - P1.min(axis=0))
    x0 = np.array([k08.X0[n] for n in k08.SPEC]); xk = np.array([k08.XK[n] for n in k08.SPEC])
    reading = {"standardization": "multiply", "AR_percent": False}
    rows = [{"check": "Table 2 x0 = mid-range of Table 1 input %s" % n, "value": float(m), "expected": float(a),
             "ok": bool(abs(m - a) <= 0.002 * max(1, abs(a)))} for n, m, a in zip(k08.SPEC, mid, x0)]
    rows += [{"check": "Table 2 xk = 2 / range of Table 1 input %s" % n, "value": float(m), "expected": float(a),
              "ok": bool(abs(m - a) <= 0.01 * a)} for n, m, a in zip(k08.SPEC, rng2, xk)]
    _, g = k08.reading_test(df)
    for _, r in g.iterrows():
        rows.append({"check": "published %s net, %s, AR in %s: k on Table 1 (paper: %s)"
                     % (r["net"], r["standardization"], "%" if r["AR_percent"] else "fraction",
                        "0.99" if r["net"] == "significant" else "Fig. 22, high"),
                     "value": r["k_vs_measured"], "expected": np.nan, "ok": np.nan})
    Pm = k08.spectral_params(*(A[c] / 100.0 for c in (415, 750, 900, 950, 1000)))
    maps_, ks = [], []
    for seed in kc.SEEDS:
        w, kk = k08.retrain(df, reading, seed)
        ks.append(kk)
        maps_.append(k08.moderate(Pm, w, reading))
    rows.append({"check": "retrained moderate net: median k on Table 1 over %d seeds (paper 0.95, gate +-0.03)"
                 % len(ks), "value": float(np.median(ks)), "expected": 0.95,
                 "ok": bool(abs(np.median(ks) - 0.95) <= 0.03)})
    # our LSCC Clementine-band filters vs the albedos Korokhin 2008 printed (same soils, fractions)
    t1 = df.set_index(["sample", "size"])
    for c in (415, 750, 900, 950, 1000):
        a, b = [], []
        for _, r in lscc.iterrows():
            key = (str(r["soil"]), r["fraction"])
            if key in t1.index:
                a.append(r["A%d" % c] / 100.0); b.append(t1.loc[key, "A%d" % c])
        a, b = np.array(a), np.array(b)
        rows.append({"check": "our Clementine filter at %d nm vs Korokhin 2008 Table 1 (n = %d): median rel. diff %%"
                     % (c, len(a)), "value": float(100 * np.median(a / b - 1)), "expected": 0.0,
                     "ok": np.nan})
    chk = pd.DataFrame(rows)
    proof = chk[chk["check"].str.startswith("Table 2") & chk["check"].str.contains("input (AR|CBR)$", regex=True)]
    if len(proof) != 4 or not proof["ok"].all():
        sys.exit("[%s] FATAL: Korokhin 2008 Table 2 does not standardize Table 1 as read" % STAGE)
    if not chk.iloc[-6]["ok"]:
        sys.exit("[%s] FATAL: retrained moderate TiO2 net misses the published k = 0.95 +- 0.03" % STAGE)
    print(chk.round(3).to_string(index=False))
    return np.median(np.stack(maps_), axis=0), chk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    tag = ap.parse_args().tag
    kc.banner(STAGE, "maps, histograms and comparisons (Figs. 19-29), tag=%s" % tag)
    kc.verify_manifest(STAGE)
    P, n_seeds = ann_maps(tag)
    high2 = np.load(kc.PRODUCTS / "highland_mask.npy")
    nr, ns = high2.shape
    high = high2.reshape(-1)
    meanA = np.load(kc.PRODUCTS / "mean_albedo.npy").reshape(-1)
    maps = {k: P[:, j] for j, k in enumerate(kc.PARAMS)}
    np.save(kc.PRODUCTS / ("ann_%s_median_iter3.npy" % tag), P.astype(np.float32))

    # ---- Fig. 19 / 20 --------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(13, 9))
    hrows = []
    fig2, axes2 = plt.subplots(2, 4, figsize=(13, 6))
    for ax, ax2, k in zip(axes.flat, axes2.flat, kc.PARAMS):
        im = ax.imshow(maps[k].reshape(nr, ns), cmap="gray", vmin=RANGES[k][0], vmax=RANGES[k][1])
        ax.set_title("%s (paper's scale %s)" % (k, RANGES[k]), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04)
        h, e, peaks, npk = histogram_peaks(maps[k])
        ax2.bar(0.5 * (e[:-1] + e[1:]), h, width=np.diff(e), color="c", edgecolor="k", lw=0.3)
        ax2.set_title("%s: %d peak(s) %s" % (k, npk, ", ".join("%.3g" % p for p in peaks)), fontsize=8)
        hrows.append({"param": k, "n_peaks": npk, "bimodal": int(npk >= 2),
                      "peak_lo": peaks[0] if peaks else np.nan,
                      "peak_hi": peaks[-1] if len(peaks) > 1 else np.nan,
                      "p1": float(np.percentile(maps[k], 1)), "p99": float(np.percentile(maps[k], 99)),
                      "median": float(np.median(maps[k])),
                      "training_range_midpoint": 0.5 * (kc.TABLE11_MIN[k] + kc.TABLE11_MAX[k])})
    fig.suptitle("Fig. 19: iteration-3 ANN maps, median of %d seeds" % n_seeds)
    fig.tight_layout(); fig.savefig(kc.FIGURES / ("fig19_ann_maps_%s.png" % tag), dpi=110); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(kc.FIGURES / ("fig20_histograms_%s.png" % tag), dpi=110); plt.close(fig2)
    pd.DataFrame(hrows).to_csv(kc.TABLES / "step09_histograms.csv", index=False)
    print(pd.DataFrame(hrows).round(3).to_string(index=False))

    # ---- comparisons ---------------------------------------------------------
    A = {k: v.reshape(-1) for k, v in m3_clementine((nr, ns)).items()}
    omat = np.sqrt((A[750] / 100 - 0.08) ** 2 + (A[950] / A[750] - 1.19) ** 2)
    ratio = (A[415] / A[750] - 0.42) / (A[750] / 100.0)
    tio2 = 3.71 * np.clip(np.arctan(ratio), 0, None) ** 5.98
    lscc = lscc_training()
    lscc.to_csv(kc.TABLES / "step09_lscc_mlr_training.csv", index=False)
    pchk = check_published(lscc)
    pchk.to_csv(kc.TABLES / "step09_mlr_published_check.csv", index=False)
    print(pchk.round(3).to_string(index=False))
    base = {}
    for key in PUBLISHED:
        c = pchk[pchk["target"] == key].sort_values("median_abs_log10_error").iloc[0]
        base[key] = 10.0 if c["log_base"] == "10" else np.e
    mlr = {"published": {k: published_mlr(A, k, base[k]) for k in PUBLISHED}}
    coefs = []
    for variant, sub in (("refit_all", lscc), ("refit_mare", lscc[lscc["set"] == "mare"])):
        mlr[variant] = {}
        for target, eq in (("Is", 3), ("IsFeO", 3), ("CPX", 3), ("OPX", 3), ("PLG", 3), ("OLV", 5), ("d", 4)):
            t = sub.dropna(subset=[target])
            t = t[t[target] > 0]
            Xl = design({k: t["A%d" % k].to_numpy() for k in CLEM}, eq)
            beta, *_ = np.linalg.lstsq(Xl, np.log(t[target].to_numpy()), rcond=None)
            fit_r = np.corrcoef(Xl @ beta, np.log(t[target]))[0, 1]
            mlr[variant][target] = np.exp(design(A, eq) @ beta)
            coefs.append({"variant": variant, "target": target, "equation": eq, "n_spectra": len(t),
                          "n_soils": t["soil"].nunique(), "r_on_lscc": fit_r,
                          **{"b%d" % i: b for i, b in enumerate(beta)}})
    pd.DataFrame(coefs).to_csv(kc.TABLES / "step09_mlr_coefficients.csv", index=False)
    ilm_bd = ilmenite_band_depth(nr)
    np.save(kc.PRODUCTS / "ilmenite_band_depth_surkov2020.npy", ilm_bd.astype(np.float32))
    tio2_ann, k08 = korokhin2008_map(A, lscc)
    k08.to_csv(kc.TABLES / "step09_korokhin2008_checks.csv", index=False)
    np.save(kc.PRODUCTS / "tio2_korokhin2008_moderate.npy", tio2_ann.astype(np.float32))

    def corr(a, b, m):
        ok = m & np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    allpx = np.ones_like(high)
    # (id, ANN map, comparison key, mask); keys resolved per MLR variant below
    IS_KEYS = ("Is", "IsFeO", "IsFeO_P06")
    mlr_cor = [("C_nFe_Is_mare", "nFe", IS_KEYS, ~high), ("C_nFe_Is_high", "nFe", IS_KEYS, high),
               ("C_S_d", "S", ("d",), allpx),
               ("C_PLG_mare", "PLG", ("PLG",), ~high), ("C_PLG_high", "PLG", ("PLG",), high),
               ("C_OLV_high", "OLV", ("OLV",), high), ("C_OLV_mare", "OLV", ("OLV",), ~high),
               ("C_CPX_mare", "CPX", ("CPX", "Augite"), ~high), ("C_CPX_high", "CPX", ("CPX", "Augite"), high),
               ("C_OPX_all", "OPX", ("OPX",), allpx)]
    vrows, prim = [], []
    for cid, k, keys, m in mlr_cor:
        for variant in ("published", "refit_all", "refit_mare"):
            for key in keys:
                if key in mlr[variant]:
                    vrows.append({"id": cid, "variant": variant, "mlr_target": key, "r": corr(maps[k], mlr[variant][key], m)})
        pub = [key for key in keys if key in mlr["published"]]
        v, key = ("published", pub[0]) if pub else ("refit_all", keys[0])
        prim.append({"id": cid, "r": corr(maps[k], mlr[v][key], m), "comparison": "MLR %s (%s)" % (key, v)})
    pd.DataFrame(vrows).to_csv(kc.TABLES / "step09_correlations_variants.csv", index=False)
    other = [("C_ILM_TiO2_mare", maps["ILM"], tio2, ~high, "TiO2 Lucey Eq. 7"),
             ("C_ILM_TiO2_high", maps["ILM"], tio2, high, "TiO2 Lucey Eq. 7"),
             ("C_ILM_BD_mare", maps["ILM"], ilm_bd, ~high, "ilmenite, 1550-nm band depth (Surkov 2020)"),
             ("C_ILM_BD_high", maps["ILM"], ilm_bd, high, "ilmenite, 1550-nm band depth (Surkov 2020)"),
             ("C_ILM_TiO2ANN_all", maps["ILM"], tio2_ann, allpx, "TiO2, moderately non-linear ANN (Korokhin 2008), retrained"),
             ("C_ILM_TiO2ANN_mare", maps["ILM"], tio2_ann, ~high, "TiO2, moderately non-linear ANN (Korokhin 2008), retrained"),
             ("C_ILM_TiO2ANN_high", maps["ILM"], tio2_ann, high, "TiO2, moderately non-linear ANN (Korokhin 2008), retrained"),
             ("C_nFe_OMAT_mare", maps["nFe"], omat, ~high, "OMAT Lucey Eq. 6"),
             ("C_nFe_OMAT_high", maps["nFe"], omat, high, "OMAT Lucey Eq. 6"),
             ("C_CPX_A_high", maps["CPX"], meanA, high, "<A>"),
             ("C_PLG_A_high", maps["PLG"], meanA, high, "<A>"),
             ("C_Gr_A_all", maps["Gr"], meanA, allpx, "<A>")]
    prim += [{"id": c[0], "r": corr(c[1], c[2], c[3]), "comparison": c[4]} for c in other]
    mare_ratio = float(np.median(maps["ILM"][~high]) / np.median(ilm_bd[~high]))
    prim.append({"id": "C_ILM_ANN_over_BD_mare", "r": mare_ratio,
                 "comparison": "median ANN c_ILM / median band-depth ilmenite, mare (a ratio, not r)"})
    cdf = pd.DataFrame(prim)
    cdf.to_csv(kc.TABLES / "step09_correlations.csv", index=False)
    print(cdf.round(3).to_string(index=False))
    print("  MLR: published (Shkuratov 2005a, Pieters 2006) for %s; refits on %d LSCC spectra from %d soils"
          % (", ".join(PUBLISHED), len(lscc), lscc["soil"].nunique()))
    print("  band-depth ilmenite: median %.2f wt%% (mare) / %.2f wt%% (highland); 99th pct %.1f"
          % (np.median(ilm_bd[~high]), np.median(ilm_bd[high]), np.percentile(ilm_bd, 99)))

    # correlation diagrams (Figs. 21-29 analogues)
    pairs = [("CPX", "<A>", maps["CPX"], meanA), ("PLG", "<A>", maps["PLG"], meanA),
             ("nFe", "Is (MLR, Pieters 2006)", maps["nFe"], mlr["published"]["Is"]),
             ("S", "d (MLR refit)", maps["S"], mlr["refit_all"]["d"]),
             ("PLG", "PLG (MLR, Shkuratov 2005a)", maps["PLG"], mlr["published"]["PLG"]),
             ("OLV", "OLV (MLR refit)", maps["OLV"], mlr["refit_all"]["OLV"]),
             ("nFe", "OMAT (Lucey)", maps["nFe"], omat), ("ILM", "TiO2 (Lucey)", maps["ILM"], tio2)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (yk, xk, yv, xv) in zip(axes.flat, pairs):
        ok = np.isfinite(xv) & np.isfinite(yv)
        lo, hi = np.percentile(xv[ok], [0.5, 99.5]), np.percentile(yv[ok], [0.5, 99.5])
        ax.hist2d(xv[ok], yv[ok], bins=120, range=[lo, hi], cmap="gray", cmin=1)
        ax.set_xlabel(xk); ax.set_ylabel("%s (ANN)" % yk)
    fig.tight_layout(); fig.savefig(kc.FIGURES / ("fig21-29_correlations_%s.png" % tag), dpi=110)
    plt.close(fig)

    # Fig. 31 analogue: band-depth ilmenite vs the ANN
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, v, t in ((axes[0], ilm_bd, "(a) ilmenite, 1550-nm BD (Surkov 2020), wt%"),
                     (axes[1], maps["ILM"], "(b) c_ILM, ANN iteration 3, %")):
        lo, hi = np.percentile(v, [1, 99])
        im = ax.imshow(v.reshape(nr, ns), cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(t, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04)
    for m, col, lab in ((~high, "k", "mare"), (high, "r", "highland")):
        axes[2].plot(ilm_bd[m][::20], maps["ILM"][m][::20], ".", ms=1, color=col, label=lab)
    axes[2].set_xlabel("ILM, % (BD)"); axes[2].set_ylabel("ILM, % (ANN)"); axes[2].legend(markerscale=8)
    axes[2].set_title("(c) r mare %.2f, highland %.2f" % (cdf.set_index("id").loc["C_ILM_BD_mare", "r"],
                                                         cdf.set_index("id").loc["C_ILM_BD_high", "r"]), fontsize=8)
    fig.tight_layout(); fig.savefig(kc.FIGURES / ("fig31_ilmenite_band_depth_%s.png" % tag), dpi=110)
    plt.close(fig)

    # Fig. 30 analogue: TiO2 by the retrained Korokhin (2008) moderate ANN vs c_ILM
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    lo, hi = np.percentile(tio2_ann, [1, 99])
    im = axes[0].imshow(tio2_ann.reshape(nr, ns), cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("(a) TiO2, moderately non-linear ANN (Korokhin 2008), wt%", fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], orientation="horizontal", fraction=0.04)
    axes[1].plot(tio2_ann[::20], maps["ILM"][::20], "k.", ms=1)
    axes[1].set_xlabel("TiO2, % (ANN 2008)"); axes[1].set_ylabel("c_ILM, % (this paper's ANN)")
    axes[1].set_title("(b) r = %.2f (paper: 0.91)" % cdf.set_index("id").loc["C_ILM_TiO2ANN_all", "r"], fontsize=8)
    fig.tight_layout(); fig.savefig(kc.FIGURES / ("fig30_tio2_ann_%s.png" % tag), dpi=110)
    plt.close(fig)
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
