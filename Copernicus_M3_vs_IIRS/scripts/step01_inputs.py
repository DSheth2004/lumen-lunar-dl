"""
Step 01 -- inputs: grid identity, the shared pixel set, zones, and the raw
brightness comparison of the two sensors before either paper is applied.

Writes
  input_manifest.csv                     SHA-256 of every input (the guard)
  tables/step01_grid_check.csv           both cubes on the same grid?
  tables/step01_pixel_accounting.csv     valid / joint pixels per zone
  tables/step01_iirs_scale_factor.csv    the one number used by IIRS_scaled
  tables/step01_radiometry.csv           median M3 and IIRS reflectance per zone at
                                         common wavelengths, and their ratio
  products/joint_mask.npy, zone_map.npy
  figures/step01_coverage_zones.png, step01_radiometry.png
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

import cc_common as cc

STAGE = "step01"
CHECK_NM = [950, 1000, 1250, 1500, 1800, 2000, 2300, 2450]


def main() -> int:
    cc.banner(STAGE, "inputs, shared pixels, zones, raw radiometry")
    cc.write_manifest()
    cc.verify_manifest(STAGE)

    # ---- grid identity --------------------------------------------------
    rows = []
    with rasterio.open(cc.M3_CUBE) as a, rasterio.open(cc.IIRS_CUBE) as b:
        for k, fa, fb in (("crs", a.crs.to_wkt(), b.crs.to_wkt()),
                          ("transform", tuple(a.transform), tuple(b.transform)),
                          ("width", a.width, b.width), ("height", a.height, b.height)):
            rows.append({"property": k, "M3": str(fa), "IIRS": str(fb), "identical": fa == fb})
        nb = (a.count, b.count)
    grid = pd.DataFrame(rows)
    grid.to_csv(cc.TABLES / "step01_grid_check.csv", index=False)
    print(grid[["property", "identical"]].to_string(index=False))
    if not grid["identical"].all():
        sys.exit("[%s] FATAL: the two cubes are not on the same grid." % STAGE)
    print("  bands: M3 %d, IIRS %d" % nb)

    wl_m, wl_i = cc.wavelengths_um("M3"), cc.wavelengths_um("IIRS")
    if len(wl_m) != nb[0] or len(wl_i) != nb[1]:
        sys.exit("[%s] FATAL: wavelength table length != band count." % STAGE)
    print("  M3   %.1f-%.1f nm   IIRS %.1f-%.1f nm   common window %.0f-%.0f nm"
          % (1e3 * wl_m[0], 1e3 * wl_m[-1], 1e3 * wl_i[0], 1e3 * wl_i[-1],
             1e3 * cc.COMMON_LO_UM, 1e3 * cc.COMMON_HI_UM))

    # ---- pixels and zones ------------------------------------------------
    mm, mi = cc.read_mask("M3"), cc.read_mask("IIRS")
    joint = cc.joint_mask()
    zone = cc.zone_map()
    np.save(cc.PRODUCTS / "joint_mask.npy", joint)
    np.save(cc.PRODUCTS / "zone_map.npy", zone)
    acc = []
    for i, (name, lo, hi) in enumerate(cc.ZONES):
        z = zone == i
        acc.append({"zone": name, "r_lo_radii": lo, "r_hi_radii": hi,
                    "grid_px": int(z.sum()), "M3_valid": int((z & mm).sum()),
                    "IIRS_valid": int((z & mi).sum()), "both_masks": int((z & mm & mi).sum()),
                    "joint_used": int((z & joint).sum())})
    acc.append({"zone": "ALL", "r_lo_radii": np.nan, "r_hi_radii": np.nan,
                "grid_px": int(zone.size), "M3_valid": int(mm.sum()), "IIRS_valid": int(mi.sum()),
                "both_masks": int((mm & mi).sum()), "joint_used": int(joint.sum())})
    acc = pd.DataFrame(acc)
    acc.to_csv(cc.TABLES / "step01_pixel_accounting.csv", index=False)
    print(acc.to_string(index=False))

    # ---- radiometry: the sensors side by side, before any paper ---------
    cm, ci = cc.read_cube("M3"), cc.read_cube("IIRS")
    med_m = {i: np.nanmedian(cm[:, (zone == i) & joint], axis=1) for i in range(len(cc.ZONES))}
    med_i = {i: np.nanmedian(ci[:, (zone == i) & joint], axis=1) for i in range(len(cc.ZONES))}
    all_m = np.nanmedian(cm[:, joint], axis=1)
    all_i = np.nanmedian(ci[:, joint], axis=1)

    # the one scale number: median over shared pixels AND common-range IIRS bands of the
    # per-pixel IIRS/M3 ratio (M3 linearly interpolated to each IIRS band centre)
    sel_i = (wl_i >= cc.COMMON_LO_UM) & (wl_i <= cc.COMMON_HI_UM)
    ratios = []
    for b in np.flatnonzero(sel_i):
        j = int(np.searchsorted(wl_m, wl_i[b]))
        f = (wl_i[b] - wl_m[j - 1]) / (wl_m[j] - wl_m[j - 1])
        m_at = (1 - f) * cm[j - 1][joint] + f * cm[j][joint]
        ratios.append(ci[b][joint] / m_at)
    scale = float(np.nanmedian(np.concatenate(ratios)))
    pd.DataFrame({"scale": [scale], "definition": [
        "median over joint pixels and IIRS bands in 0.85-2.50 um of IIRS / M3 (M3 linearly "
        "interpolated to the IIRS band centre)"]}).to_csv(
        cc.TABLES / "step01_iirs_scale_factor.csv", index=False)
    print("  IIRS_scaled factor (median IIRS/M3): %.4f" % scale)

    rad = []
    for i, (name, _, _) in list(enumerate(cc.ZONES)) + [(-1, ("ALL", 0, 0))]:
        if i == -1:
            name = "ALL"
            a, b = all_m, all_i
        else:
            a, b = med_m[i], med_i[i]
        if not np.isfinite(b).any():
            continue
        for w in CHECK_NM:
            ra, rb = np.interp(w * 1e-3, wl_m, a), np.interp(w * 1e-3, wl_i, b)
            rad.append({"zone": name, "wavelength_nm": w, "M3_median": ra,
                        "IIRS_median": rb, "IIRS_over_M3": rb / ra})
    rad = pd.DataFrame(rad)
    rad.to_csv(cc.TABLES / "step01_radiometry.csv", index=False)
    print(rad[rad.zone == "ALL"].round(4).to_string(index=False))

    # per-pixel agreement at 1500 nm (outside both absorption bands)
    k_m = int(np.argmin(np.abs(wl_m - 1.5)))
    k_i = int(np.argmin(np.abs(wl_i - 1.5)))
    s = cc.r2_bias_rmse(cm[k_m][joint], ci[k_i][joint])
    pd.DataFrame([{"quantity": "reflectance near 1500 nm", "M3_band_nm": 1e3 * wl_m[k_m],
                   "IIRS_band_nm": 1e3 * wl_i[k_i], **s}]).to_csv(
        cc.TABLES / "step01_pixel_agreement_1500nm.csv", index=False)
    print("  per-pixel reflectance ~1500 nm, IIRS vs M3: r %.3f  bias %+.4f  RMSE %.4f  (n %d)"
          % (s["pearson_r"], s["bias"], s["rmse"], s["n"]))

    # ---- co-registration vs illumination ----------------------------------
    # Fine detail (high-pass, sigma 8 px) carries topographic shading; broad detail
    # (low-pass, sigma 6 px) carries albedo. A misregistration would show as a better
    # high-pass match at some non-zero shift; opposite illumination shows as a
    # NEGATIVE high-pass correlation at zero shift that no shift repairs.
    from scipy.ndimage import gaussian_filter

    def _fill(x):
        return np.where(np.isfinite(x), x, np.nanmedian(x))

    A = _fill(cm[k_m]) - gaussian_filter(_fill(cm[k_m]), 8)
    B = _fill(ci[k_i]) - gaussian_filter(_fill(ci[k_i]), 8)
    shifts = []
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            Bs = np.roll(np.roll(B, dy, 0), dx, 1)
            m = joint & np.roll(np.roll(joint, dy, 0), dx, 1)
            m[:10] = m[-10:] = False
            m[:, :10] = False
            shifts.append({"dy": dy, "dx": dx, "highpass_r": float(np.corrcoef(A[m], Bs[m])[0, 1])})
    shifts = pd.DataFrame(shifts)
    shifts.to_csv(cc.TABLES / "step01_shift_search.csv", index=False)
    r0 = float(shifts.query("dy == 0 and dx == 0")["highpass_r"].iloc[0])
    best = shifts.loc[shifts["highpass_r"].idxmax()]
    lp_r = float(np.corrcoef(gaussian_filter(_fill(cm[k_m]), 6)[joint],
                             gaussian_filter(_fill(ci[k_i]), 6)[joint])[0, 1])
    pd.DataFrame([{"highpass_r_zero_shift": r0, "best_highpass_r": best["highpass_r"],
                   "best_dy": int(best["dy"]), "best_dx": int(best["dx"]),
                   "lowpass_r_zero_shift": lp_r}]).to_csv(
        cc.TABLES / "step01_illumination_check.csv", index=False)
    print("  fine detail r at zero shift %.3f (best over +-8 px: %.3f at dy %d dx %d);"
          " broad detail r %.3f" % (r0, best["highpass_r"], best["dy"], best["dx"], lp_r))

    # the IIRS gaps the common range excludes must be the ones in the band list
    gaps = [(wl_i[k], wl_i[k + 1]) for k in np.flatnonzero(np.diff(wl_i) > 0.03)]
    if len(gaps) != len(cc.IIRS_GAPS_UM) or any(
            abs(a - c) > 1e-3 or abs(b - d) > 1e-3 for (a, b), (c, d) in zip(gaps, cc.IIRS_GAPS_UM)):
        sys.exit("[%s] FATAL: IIRS gaps %s differ from cc.IIRS_GAPS_UM" % (STAGE, gaps))
    print("  IIRS gaps excluded from the common range: %s"
          % ", ".join("%.0f-%.0f nm" % (1e3 * a, 1e3 * b) for a, b in gaps))

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 8), sharey=True)
    ax[0].imshow(cm[k_m], cmap="gray", vmin=np.nanpercentile(cm[k_m], 1),
                 vmax=np.nanpercentile(cm[k_m], 99))
    ax[0].set_title("M3 %.0f nm" % (1e3 * wl_m[k_m]))
    ax[1].imshow(np.where(mi, ci[k_i], np.nan), cmap="gray",
                 vmin=np.nanpercentile(ci[k_i][mi], 1), vmax=np.nanpercentile(ci[k_i][mi], 99))
    ax[1].set_title("IIRS %.0f nm" % (1e3 * wl_i[k_i]))
    zs = np.where(joint, zone, np.nan).astype(float)
    im = ax[2].imshow(zs, cmap="viridis", vmin=0, vmax=len(cc.ZONES) - 1)
    ax[2].set_title("zones on shared pixels")
    cb = fig.colorbar(im, ax=ax[2], ticks=range(len(cc.ZONES)), fraction=0.08)
    cb.ax.set_yticklabels([z[0] for z in cc.ZONES])
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(cc.FIGURES / "step01_coverage_zones.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (name, _, _) in enumerate(cc.ZONES):
        if np.isfinite(med_i[i]).any():
            l, = ax[0].plot(wl_m, med_m[i], "-", lw=1.2, label="M3 " + name)
            ax[0].plot(wl_i, med_i[i], "--", lw=1.2, color=l.get_color(), label="IIRS " + name)
    ax[0].set_xlim(0.5, 2.6)
    ax[0].set_xlabel("wavelength (um)")
    ax[0].set_ylabel("median reflectance, shared pixels")
    ax[0].legend(fontsize=7, ncol=2)
    ratio = all_i / np.interp(wl_i, wl_m, all_m)
    ax[1].plot(wl_i, ratio, "k.-", ms=3)
    ax[1].axhline(1, color="gray", lw=0.8)
    ax[1].axhline(scale, color="C3", lw=0.8, ls="--", label="one scale number %.3f" % scale)
    ax[1].set_xlabel("wavelength (um)")
    ax[1].set_ylabel("IIRS / M3 (scene median spectra)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(cc.FIGURES / "step01_radiometry.png", dpi=110)
    plt.close(fig)

    cc.verify_manifest(STAGE)
    print("\n  STEP 01 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
