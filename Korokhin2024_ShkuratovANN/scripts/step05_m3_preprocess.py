"""
Step 05 -- M3 L2 -> the paper's 21-point spectra in the RELAB photometric system.

What the paper did (Sec. 4.2): M3 spectra "generated ... by applying the technique
developed by Shkuratov et al. (2019) and Surkov et al. (2020, 2021)" -- a Gaussian
lambda-convolution of the spectra and a Fourier filtration of the images against
the M3 stripe pattern -- then resampled to a 0.1-um step (21 points) and reduced to
the RELAB photometric system by Eq. (2).

The recipe (primary; PAPER_NOTES D2). Shkuratov et al. (2019) and Surkov
et al. (2020) are reprinted as chapters 2 and 4 of Surkov's PhD thesis (Goettingen),
Secs. 2.2 and 4.2:
  1. Gaussian lambda-convolution of each pixel's spectrum on the native M3 bands,
     "kernel width of about 3 spectral pixels on the level of the half-height"
     (Surkov 2020; Shkuratov 2019 used ~4), bands beyond 2.6 um not used, the ends of
     the spectrum extended by odd reflection (a one-sided kernel biased the 541-nm
     channel by 6 %). The thesis's acceptance rule -- every channel's image mean
     preserved within 0.2 % -- cannot be met on this scene by any kernel >= 1.5 pixels:
     the mean spectrum curves strongly at 580-750 nm. The stated width is kept and the
     per-channel change is reported (tables/step05_recipe_channel_means.csv).
  2. Fourier filtration of each channel's image, in reflectance (Eqs. 2.4-2.6): the
     2-D spectrum is set to C = 0 inside a domain Omega holding the comb of the
     vertical stripes -- here the band |k_line| <= w_line (cycles per line) at
     cross-track frequencies |k_sample| >= k_min. The thesis picks Omega by eye
     ("disappearance of the stripe pattern", "absence of features related to the Moon
     on the ratio image", scene mean within 0.2 %) after locating the comb in the 2-D
     power spectrum (thesis Figs. 2.6-2.7, 4.4). Here the comb is located numerically,
     with thresholds fixed before any downstream result was seen (locate_comb), on the
     950-nm channel the thesis illustrates, and the same Omega is applied to every
     channel (one rule per scene). Power excess is log10 power over the median of the
     |k_line| = 20-40 rows in the same k_sample band:
       k_min  = the lower edge of the run of k_sample bands (edges 1/150, 1/50, 1/25,
                1/12, 1/2 cycles per sample), counted from the highest down, in which the
                k_line = 0 row exceeds the |k_line| = 8-15 rows by >= 0.3 dex (x2) --
                the comb's wings, where the thesis locates the strip noise (Fig. 4.4);
                below the run the lunar scene, not the stripes, dominates;
       w_line = the last |k_line| such that every row 0..w_line in that band exceeds the
                |k_line| = 8-15 rows by >= 0.3 dex.
     Omega never contains k = 0, so the scene mean is unchanged by construction.
     Two numeric filter-choice rules tried first (a block-column-mean stripe power and a
     ratio-image leak test) were rejected because neither could tell stripes from lunar
     texture on this scene; see PAPER_NOTES D2.
  3. The paper: "resampled ... to a uniform wavelength step size of 0.1 um" -- linear
     interpolation of the filtered native spectra at 0.54, 0.64, ..., 2.54 um.
  4. Eq. (2): A_RELAB = 1.685 A_M3 + 0.0355, as published;
  5. along-track resampling to square ground pixels, 888 raw lines -> 500 rows,
     so the scene has the paper's 304 x 500 = 152,000 pixels (step04).
Sensitivity cubes, reported beside the primary one, never substituted for it:
  * no_destripe -- step 1 and 3 only (no Fourier filtration);
  * approx_v1   -- a simpler approximation of the recipe: one Gaussian
                   kernel (FWHM 0.1 um) smoothing and resampling at once, then a
                   log-domain notch on the k_line = 0 row for cross-track periods
                   < 25 samples.
The recipe-filtered native cube (bands <= 2.6 um, L2 units) is saved for step09
(Clementine bands; the 1550-nm band depth of Surkov 2020).

Also here:
  * Eq. (2) re-derived from the Apollo 16 image the paper names, against the
    LSCC spectrum of 62231: the lab spectrum is taken to the native M3 bands
    (Gaussian, FWHM = local band spacing), put through the same lambda-convolution
    and resampling, and compared on the 21-point grid. <45 um is primary (D7); the
    other three fractions are reported beside it;
  * <A> map (Fig. 3), mare/highland split (Otsu threshold on <A>, D8);
  * the four sites: our spectra vs the digitized Fig. 12 and vs the <A> implied
    by the paper's tables.
"""
from __future__ import annotations

import json
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import korokhin_common as kc

STAGE = "step05"
FWHM_UM = 0.1                   # approx_v1 only
STRIPE_MAX_PERIOD = 25          # samples, approx_v1 only
RECIPE_FWHM_PIX = 3.0           # Surkov et al. (2020): "about 3 spectral pixels"
RECIPE_MAX_NM = 2600.0          # "We did not use wavelengths larger than 2.6 um"
FILTER_BAND_NM = 950.0          # the channel the thesis illustrates the filtration on
KS_EDGES = (1 / 150, 1 / 50, 1 / 25, 1 / 12, 0.5)         # k_sample bands, cycles per sample
COMB_DEX = 0.3                  # x2 power excess that marks the stripe comb
BLOCK_LINES = 64
REF_LINES = 888                 # w_line is expressed in bins of the scene block
N_ROWS = 500
APOLLO_BOX = 2                  # (2*2+1)^2 = 25 pixels around the LM site
SITE_REFINE = 8                 # +-rows/samples searched around each Fig. 3 cross (D9)


def refine_site(cube, row, sample, ref):
    """Pixel within +-SITE_REFINE whose spectrum is closest (rel. rms) to the digitized
    Fig. 12 spectrum. The window was fixed by requiring the optimum to be identical for
    +-8 and +-12 at all four sites; at +-20 the featureless sites jump to distant
    look-alikes, which is why the window is not larger. Only the plotted spectrum is
    used -- never the fitted parameters the audit scores."""
    ok = np.isfinite(ref)
    best = (np.inf, 0, 0)
    n_close = 0
    errs = []
    for dr in range(-SITE_REFINE, SITE_REFINE + 1):
        for ds in range(-SITE_REFINE, SITE_REFINE + 1):
            sp = cube[row + dr, :, sample + ds].astype(float)
            e = np.sqrt(np.mean((sp[ok] - ref[ok]) ** 2)) / ref[ok].mean()
            errs.append(e)
            if e < best[0]:
                best = (e, dr, ds)
    errs = np.array(errs)
    return best[1], best[2], best[0], int((errs < best[0] + 0.01).sum())


def gaussian_matrix(wl_um: np.ndarray, valid: np.ndarray) -> np.ndarray:
    sig = FWHM_UM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    W = np.exp(-0.5 * ((kc.M3_GRID_UM[:, None] - wl_um[None, :]) / sig) ** 2)
    W[:, ~valid] = 0.0
    return W / W.sum(axis=1, keepdims=True)


def convolve(cube: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(lines, bands, samples) -> (lines, 21, samples)."""
    c = np.where(cube <= -990, 0.0, cube.astype(np.float64))
    return np.einsum("kb,lbs->lks", W, c)


# ---- the recipe (Shkuratov 2019 / Surkov 2020) --------------------------------------
def recipe_bands(wl_nm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Indices of the native bands the recipe uses: valid and <= 2.6 um."""
    return np.flatnonzero(valid & (wl_nm <= RECIPE_MAX_NM))


def lambda_kernel(n: int, fwhm_pix: float = RECIPE_FWHM_PIX) -> np.ndarray:
    """(n, n) matrix of the Gaussian lambda-convolution in band-index space, truncated at
    3 sigma, with the spectrum extended past both ends by odd reflection
    (x[-j] = 2 x[0] - x[j]), which keeps a linear trend unbiased at the edges."""
    sig = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    h = int(np.ceil(3 * sig))
    g = np.exp(-0.5 * (np.arange(-h, h + 1) / sig) ** 2)
    g /= g.sum()
    K = np.zeros((n, n))
    for i in range(n):
        for t, wgt in zip(range(i - h, i + h + 1), g):
            if t < 0:                                   # 2 x[0] - x[-t]
                K[i, 0] += 2 * wgt
                K[i, -t] -= wgt
            elif t >= n:                                # 2 x[n-1] - x[2(n-1) - t]
                K[i, n - 1] += 2 * wgt
                K[i, 2 * (n - 1) - t] -= wgt
            else:
                K[i, t] += wgt
    return K


def lambda_convolve(cube: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(lines, n, samples) -> same shape; the recipe's step 1."""
    return np.einsum("kb,lbs->lks", K, cube.astype(np.float64))


def fourier_filter(img: np.ndarray, w_line: float, k_min: float) -> np.ndarray:
    """The recipe's step 2 (C = 0 inside Omega), on reflectance. w_line in bins of a
    REF_LINES-line image, so the same Omega applies to blocks of other lengths."""
    F = np.fft.fft2(img)
    kl = np.fft.fftfreq(img.shape[0])                   # cycles per line
    ks = np.fft.fftfreq(img.shape[1])                   # cycles per sample
    omega = (np.abs(kl) <= (w_line + 1e-9) / REF_LINES)[:, None] & (np.abs(ks) >= k_min)[None, :]
    F[omega] = 0.0
    return np.real(np.fft.ifft2(F))


def stripe_power(img: np.ndarray) -> float:
    """Reported only: rms across-track high-pass of 64-line block column means / mean."""
    out = []
    for a in range(0, img.shape[0] - BLOCK_LINES + 1, BLOCK_LINES):
        cm = img[a:a + BLOCK_LINES].mean(axis=0)
        hp = cm - np.convolve(np.pad(cm, 7, mode="edge"), np.ones(15) / 15, mode="valid")
        out.append(np.sqrt(np.mean(hp ** 2)))
    return float(np.mean(out) / img.mean())


def comb_profile(img: np.ndarray) -> pd.DataFrame:
    """log10 power excess of each |k_line| row (0-15) over the |k_line| = 20-40 rows, per
    k_sample band; |k_line| counts bins of a REF_LINES-line image."""
    P = np.log10(np.abs(np.fft.fft2(img - img.mean())) ** 2 + 1e-30)
    kl = np.rint(np.abs(np.fft.fftfreq(img.shape[0])) * REF_LINES).astype(int)
    ks = np.abs(np.fft.fftfreq(img.shape[1]))
    rows = []
    for lo, hi in zip(KS_EDGES[:-1], KS_EDGES[1:]):
        sel = (ks >= lo) & (ks < hi + 1e-12)
        bg = np.median(P[(kl >= 20) & (kl < 40)][:, sel])
        for b in range(16):
            r = P[kl == b][:, sel]
            if r.size:
                rows.append({"ks_lo": lo, "ks_hi": hi, "k_line": b, "excess_dex": float(np.median(r) - bg)})
    return pd.DataFrame(rows)


def locate_comb(img: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    prof = comb_profile(img)
    picks = []
    for lo in KS_EDGES[:-1]:
        p = prof[prof["ks_lo"] == lo].set_index("k_line")["excess_dex"]
        ref = p[(p.index >= 8) & (p.index <= 15)].median()
        above = p - ref >= COMB_DEX
        w = -1
        while (w + 1) in above.index and above[w + 1]:
            w += 1
        picks.append({"ks_lo": lo, "offcomb_ref_dex": ref, "comb_row0_excess_dex": p.loc[0] - ref,
                      "w_line": w, "stripes_dominate": bool(above.get(0, False))})
    picks = pd.DataFrame(picks).sort_values("ks_lo", ascending=False).reset_index(drop=True)
    # the comb's wings: the run of bands, from the highest k_sample down, in which the
    # stripes dominate. (A first version took the lowest such band anywhere; the lowest
    # band, 1/150-1/50, also shows an along-track excess -- plausibly the lunar scene,
    # e.g. the Crisium margin -- across a gap with no comb at 1/25-1/12, and would have
    # zeroed every cross-track period < 150 samples.)
    run = 0
    while run < len(picks) and picks.loc[run, "stripes_dominate"]:
        run += 1
    if run == 0:
        sys.exit("[%s] FATAL: no stripe comb found in the 2-D power spectrum" % STAGE)
    k_min = float(picks.loc[run - 1, "ks_lo"])
    w_line = float(picks.loc[run - 1, "w_line"])
    picks["in_omega"] = picks.index < run
    return w_line, k_min, prof.merge(picks, on="ks_lo")


def recipe_native(raw: np.ndarray, idx: np.ndarray, K: np.ndarray, w: float, k: float,
                  fourier: bool = True) -> np.ndarray:
    """(lines, 85, samples) L2 -> (lines, len(idx), samples): smoothed (+ filtered)."""
    sm = lambda_convolve(raw[:, idx, :], K)
    if not fourier:
        return sm
    return np.stack([fourier_filter(sm[:, j, :], w, k) for j in range(sm.shape[1])], axis=1)


def to_grid(native: np.ndarray, wl_um: np.ndarray) -> np.ndarray:
    """Linear interpolation along the band axis at the 21 grid points, (lines, 21, samples)."""
    out = np.empty((native.shape[0], kc.M3_GRID_UM.size, native.shape[2]))
    for g, lam in enumerate(kc.M3_GRID_UM):
        i = int(np.clip(np.searchsorted(wl_um, lam), 1, wl_um.size - 1))
        f = float(np.clip((lam - wl_um[i - 1]) / (wl_um[i] - wl_um[i - 1]), 0.0, 1.0))
        out[:, g, :] = (1 - f) * native[:, i - 1, :] + f * native[:, i, :]
    return out


def lab_to_m3(lam_um: np.ndarray, refl: np.ndarray, wl_um: np.ndarray, K: np.ndarray) -> np.ndarray:
    """A lab spectrum as the recipe would see it: M3 band responses (Gaussian, FWHM =
    local band spacing), the same lambda-convolution, the same resampling."""
    fw = np.gradient(wl_um)
    sig = fw / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    W = np.exp(-0.5 * ((wl_um[:, None] - lam_um[None, :]) / sig[:, None]) ** 2)
    ok = np.isfinite(refl)
    native = (W[:, ok] @ refl[ok]) / W[:, ok].sum(axis=1)
    sm = K @ native
    return to_grid(sm[None, :, None], wl_um)[0, :, 0]


# ---- approx_v1 (kept as a sensitivity) ----------------------------------------------
def destripe(img: np.ndarray) -> np.ndarray:
    """Remove along-track-constant, short-period cross-track structure (one channel)."""
    lg = np.log(np.clip(img, 1e-4, None))
    F = np.fft.fft2(lg)
    ks = np.fft.fftfreq(img.shape[1])                      # cycles per sample
    notch = np.abs(ks) > 1.0 / STRIPE_MAX_PERIOD
    F[0, notch] = 0.0                                      # k_line = 0 row only
    return np.exp(np.real(np.fft.ifft2(F)))


def to_square_rows(cube: np.ndarray, n_rows: int) -> np.ndarray:
    """Linear along-track resampling, (lines, k, samples) -> (n_rows, k, samples)."""
    L = cube.shape[0]
    pos = (np.arange(n_rows) + 0.5) * L / n_rows - 0.5
    i0 = np.clip(np.floor(pos).astype(int), 0, L - 1)
    i1 = np.clip(i0 + 1, 0, L - 1)
    f = (pos - i0)[:, None, None]
    return (1 - f) * cube[i0] + f * cube[i1]


def otsu(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    u8 = np.clip((v - v.min()) / (v.max() - v.min()) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(v.min() + t / 255.0 * (v.max() - v.min()))


def main() -> int:
    kc.banner(STAGE, "M3 L2 -> 21-point RELAB-system spectra (Eq. 2), 304 x 500 scene")
    kc.verify_manifest(STAGE)
    wl_nm = pd.read_csv(kc.M3_DIR / "m3_wavelengths_nm.csv")["wavelength_nm"].to_numpy()
    wl = wl_nm / 1000.0
    raw = np.load(kc.M3_DIR / "scene_rfl.npy")
    valid_band = ~np.all(raw <= -990, axis=(0, 2))

    # ---- the recipe -------------------------------------------------------
    idx = recipe_bands(wl_nm, valid_band)
    K = lambda_kernel(idx.size)
    rawf = np.where(raw <= -990, np.nan, raw.astype(np.float64))
    sm = lambda_convolve(np.nan_to_num(rawf[:, idx, :]), K)
    mean_dev = 100 * (sm.mean(axis=(0, 2)) / np.nanmean(rawf[:, idx, :], axis=(0, 2)) - 1)
    pd.DataFrame({"wavelength_nm": wl_nm[idx], "mean_change_pct": mean_dev}).to_csv(
        kc.TABLES / "step05_recipe_channel_means.csv", index=False)
    print("  recipe step 1: %d native bands (%.0f-%.0f nm), Gaussian FWHM %.1f band pixels;"
          " largest channel-mean change %.3f %% (thesis criterion 0.2 %%; %d channels outside)"
          % (idx.size, wl_nm[idx[0]], wl_nm[idx[-1]], RECIPE_FWHM_PIX, np.abs(mean_dev).max(),
             int((np.abs(mean_dev) > 0.2).sum())))
    ref = int(np.argmin(np.abs(wl_nm[idx] - FILTER_BAND_NM)))
    w_line, k_min, comb = locate_comb(sm[:, ref, :])
    comb.to_csv(kc.TABLES / "step05_fourier_comb.csv", index=False)
    f_ref = fourier_filter(sm[:, ref, :], w_line, k_min)
    print("  recipe step 2: stripe comb located on %.0f nm -> Omega = |k_line| <= %.0f/888 cyc/line,"
          " |k_sample| >= 1/%.0f cyc/sample; scene mean change %.5f %%"
          % (wl_nm[idx[ref]], w_line, 1 / k_min, 100 * (f_ref.mean() / sm[:, ref, :].mean() - 1)))
    print(comb.drop_duplicates("ks_lo")[["ks_lo", "offcomb_ref_dex", "comb_row0_excess_dex", "w_line", "in_omega",
                                          "stripes_dominate"]].round(3).to_string(index=False))
    native = np.stack([fourier_filter(sm[:, j, :], w_line, k_min) for j in range(idx.size)], axis=1)
    np.save(kc.PRODUCTS / "scene_native_recipe.npy", native.astype(np.float32))
    pd.DataFrame({"band": idx, "wavelength_nm": wl_nm[idx]}).to_csv(
        kc.PRODUCTS / "scene_native_recipe_wl.csv", index=False)
    pd.DataFrame([{"fwhm_band_pixels": RECIPE_FWHM_PIX, "max_nm": RECIPE_MAX_NM, "w_line_bins": w_line,
                   "k_min_cyc_per_sample": k_min, "filter_band_nm": wl_nm[idx[ref]],
                   "max_abs_channel_mean_change_pct": float(np.abs(mean_dev).max())}]).to_csv(
        kc.TABLES / "step05_recipe_parameters.csv", index=False)

    # ---- approx_v1 (sensitivity) ----------------------------------------------
    W = gaussian_matrix(wl, valid_band)
    conv = convolve(raw, W)                                 # (888, 21, 304), L2 units
    v1 = np.stack([destripe(conv[:, k, :]) for k in range(conv.shape[1])], axis=1)

    variants = {"destriped": to_grid(native, wl[idx]),       # primary: the recipe
                "no_destripe": to_grid(sm, wl[idx]),
                "approx_v1": v1}
    srows = []
    for name, c in variants.items():
        k950 = int(np.argmin(np.abs(kc.M3_GRID_UM - 0.94)))
        srows.append({"cube": name, "stripe_power_940nm": stripe_power(c[:, k950, :]),
                      "col_step_median": float(np.nanmedian(np.abs(np.diff(c.mean(axis=0), axis=1))
                                                            / c.mean(axis=0)[:, 1:]))})
    srows = pd.DataFrame(srows)
    srows.to_csv(kc.TABLES / "step05_stripe_metrics.csv", index=False)
    print(srows.round(5).to_string(index=False))

    cubes = {}
    for name, c in variants.items():
        relab = kc.EQ2_SLOPE * c + kc.EQ2_INTERCEPT
        cubes[name] = to_square_rows(relab, N_ROWS).astype(np.float32)
        np.save(kc.PRODUCTS / ("scene21_%s.npy" % name), cubes[name])
    cube = cubes["destriped"]                               # (500, 21, 304)
    mean_a = cube.mean(axis=1)
    np.save(kc.PRODUCTS / "mean_albedo.npy", mean_a)
    thr = otsu(mean_a)
    highland = mean_a > thr
    np.save(kc.PRODUCTS / "highland_mask.npy", highland)
    print("  scene %d x %d = %d pixels; <A> %.3f-%.3f; Otsu mare/highland split at <A> = %.3f"
          " (%.1f%% highland)" % (cube.shape[0], cube.shape[2], cube.shape[0] * cube.shape[2],
                                  np.percentile(mean_a, 1), np.percentile(mean_a, 99), thr,
                                  100 * highland.mean()))

    # ---- sites ---------------------------------------------------------
    sites = pd.read_csv(kc.M3_DIR / "sites.csv")
    L = raw.shape[0]
    sites["row_fig3"] = np.round((sites["line_in_scene"] + 0.5) * N_ROWS / L - 0.5).astype(int)
    sites["sample_fig3"] = sites["sample"]
    implied = pd.read_csv(kc.PAPER_TABLES / "derived_site_mean_albedo.csv").set_index("site")["mean"]
    fig12 = pd.read_csv(kc.DIGITIZED / "fig12_site_spectra.csv")
    refined = []
    for _, s in sites.iterrows():
        ref = fig12[fig12["site"] == s["site"]]["refl"].to_numpy()
        dr, ds, e, n_near = refine_site(cube, int(s["row_fig3"]), int(s["sample_fig3"]), ref)
        refined.append((int(s["row_fig3"]) + dr, int(s["sample_fig3"]) + ds, dr, ds, n_near))
    sites["row"] = [r[0] for r in refined]
    sites["sample"] = [r[1] for r in refined]
    sites["refine_drow"] = [r[2] for r in refined]
    sites["refine_dsample"] = [r[3] for r in refined]
    sites["pixels_within_1pct_of_best"] = [r[4] for r in refined]
    rows, spectra = [], []
    for _, s in sites.iterrows():
        for name, c in cubes.items():
            for where, (ri, si) in (("fig3_cross", (s["row_fig3"], s["sample_fig3"])),
                                    ("refined", (s["row"], s["sample"]))):
                sp = c[int(ri), :, int(si)].astype(float)
                ref = fig12[fig12["site"] == s["site"]]["refl"].to_numpy()
                ok = np.isfinite(ref)
                rows.append({"site": s["site"], "cube": name, "position": where, "row": ri,
                             "sample": si, "mean_albedo": sp.mean(),
                             "implied_mean_albedo": implied.loc[s["site"]],
                             "rms_vs_fig12": float(np.sqrt(np.mean((sp[ok] - ref[ok]) ** 2))),
                             "rel_rms_vs_fig12_pct": float(100 * np.sqrt(np.mean((sp[ok] - ref[ok]) ** 2))
                                                           / ref[ok].mean())})
                if name == "destriped" and where == "refined":
                    spectra.append(sp)
    chk = pd.DataFrame(rows)
    chk.to_csv(kc.TABLES / "step05_site_check.csv", index=False)
    sites.to_csv(kc.PRODUCTS / "sites_rows.csv", index=False)
    np.save(kc.PRODUCTS / "site_spectra.npy", np.array(spectra))
    for _, s in sites.iterrows():
        print("  site %d: Fig. 3 cross -> row %d, sample %d; refined by (%+d, %+d) to row %d, sample %d"
              " (%d pixels within 1 point of the best match)"
              % (s["site"], s["row_fig3"], s["sample_fig3"], s["refine_drow"], s["refine_dsample"],
                 s["row"], s["sample"], s["pixels_within_1pct_of_best"]))
    for _, r in chk[(chk["cube"] == "destriped") & (chk["position"] == "refined")].iterrows():
        print("  site %d (row %d, sample %d): <A> %.4f vs %.4f implied by the tables;"
              " rms vs Fig. 12 %.4f (%.1f%%)" % (r["site"], r["row"], r["sample"], r["mean_albedo"],
                                                r["implied_mean_albedo"], r["rms_vs_fig12"],
                                                r["rel_rms_vs_fig12_pct"]))

    # ---- Eq. (2) re-derived from the Apollo 16 image ---------------------
    meta = json.loads((kc.M3_DIR / "scene_meta.json").read_text())["apollo16"]
    a16raw = np.load(kc.M3_DIR / "apollo16_rfl.npy")
    a16 = to_grid(recipe_native(np.where(a16raw <= -990, 0.0, a16raw.astype(np.float64)), idx, K,
                                w_line, k_min), wl[idx])
    li = meta["centre_line"] - meta["first_line"]
    si = meta["centre_sample"]
    box = a16[li - APOLLO_BOX: li + APOLLO_BOX + 1, :, si - APOLLO_BOX: si + APOLLO_BOX + 1]
    m3 = box.mean(axis=(0, 2))
    a16v1 = convolve(a16raw, W)[li - APOLLO_BOX: li + APOLLO_BOX + 1, :, si - APOLLO_BOX: si + APOLLO_BOX + 1]
    m3v1 = a16v1.mean(axis=(0, 2))
    kd = kc.read_kdata("62231")
    sig = FWHM_UM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    Wl = np.exp(-0.5 * ((kc.M3_GRID_UM[:, None] - kd["lam_um"].to_numpy()[None, :]) / sig) ** 2)
    eq_rows = []
    for frac in ("<45", "20-45", "10-20", "<10"):
        for proc, m3v, lab in (
                ("recipe", m3, lab_to_m3(kd["lam_um"].to_numpy(), kd[frac].to_numpy(), wl[idx], K)),
                ("approx_v1", m3v1, (Wl / Wl.sum(axis=1, keepdims=True)) @ kd[frac].to_numpy())):
            slope, icpt = np.polyfit(m3v, lab, 1)
            pred_pub = kc.EQ2_SLOPE * m3v + kc.EQ2_INTERCEPT
            eq_rows.append({"processing": proc, "lab_fraction_um": frac, "primary": proc == "recipe" and frac == "<45",
                            "slope": slope, "intercept": icpt,
                            "rel_rms_pct": 100 * np.sqrt(np.mean((slope * m3v + icpt - lab) ** 2)) / lab.mean(),
                            "rel_rms_with_published_pct": 100 * np.sqrt(np.mean((pred_pub - lab) ** 2)) / lab.mean(),
                            "published_slope": kc.EQ2_SLOPE, "published_intercept": kc.EQ2_INTERCEPT,
                            "published_rel_rms_pct": 1.4, "apollo16_pixels": box.shape[0] * box.shape[2],
                            "lab_spectrum": "62231 %s um (LSCC Kdata)" % frac})
    eq_all = pd.DataFrame(eq_rows)
    eq_all.to_csv(kc.TABLES / "step05_eq2_sensitivity.csv", index=False)
    eq2 = eq_all[eq_all["primary"]].drop(columns=["primary"])
    eq2.to_csv(kc.TABLES / "step05_eq2_rederived.csv", index=False)
    print("  Eq.(2) re-derived (recipe, 62231 <45 um): A_RELAB = %.3f A_M3 + %.4f (rel. rms %.2f%%);"
          " published 1.685 / 0.0355 gives %.2f%% (paper: 1.4%%)"
          % (eq2["slope"].iloc[0], eq2["intercept"].iloc[0], eq2["rel_rms_pct"].iloc[0],
             eq2["rel_rms_with_published_pct"].iloc[0]))
    print(eq_all[["processing", "lab_fraction_um", "slope", "intercept", "rel_rms_pct"]].round(4).to_string(index=False))

    # ---- figures --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 7))
    im = ax.imshow(mean_a, cmap="gray", vmin=0.1, vmax=0.4, aspect="equal")
    for _, s in sites.iterrows():
        ax.plot(s["sample"], s["row"], "+", color="r", ms=12, mew=1.5)
        ax.text(s["sample"] - 12, s["row"] - 8, str(int(s["site"])), color="r", fontsize=10)
    ax.set_title("Fig. 3: <A>, %s" % json.loads((kc.M3_DIR / "scene_meta.json").read_text())["scene"],
                 fontsize=8)
    plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, label="<A>")
    fig.tight_layout(); fig.savefig(kc.FIGURES / "fig03_mean_albedo.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (_, s), sp in zip(axes.flat, sites.iterrows(), spectra):
        ref = fig12[fig12["site"] == s["site"]]
        ax.plot(ref["lam_um"], ref["refl"], "o", ms=4, color="0.4", label="Fig. 12 (digitized)")
        ax.plot(kc.M3_GRID_UM, sp, "b-", label="ours")
        ax.set_title("site %d" % s["site"]); ax.set_ylim(0, 0.5); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(kc.FIGURES / "fig01_site_spectra_vs_paper.png", dpi=120)
    plt.close(fig)
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
