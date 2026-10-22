"""
Step 04 -- compare the two sensors, for each paper.

Main comparison: IIRS_common vs M3_common (same pixels, same wavelengths).
Also reported: IIRS_scaled vs M3_common (brightness offset removed), and
M3_common vs M3_full (what cutting the range to 0.85-2.50 um costs on M3 alone).

Three levels, because step01 found the two images lit from opposite sides (the
fine-detail correlation is negative at zero shift and no shift repairs it):
  pixel     per pixel. Topographic shading differs between the images, and neither
            paper corrects for it, so this level mixes illumination with composition.
  smoothed  Gaussian-smoothed maps (sigma 6 px = 0.84 km, mask-normalised), which
            suppress slope shading and keep albedo/composition patterns.
  zone      medians over the four distance zones. The primary comparison.

Writes tables/step04_*.csv and figures/step04_*.png.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import cc_common as cc

sys.path.insert(0, str(cc.KOROKHIN_SCRIPTS))
import korokhin_common as kc                # noqa: E402

STAGE = "step04"
SMOOTH_PX = 6.0
PAIRS = [("IIRS_common", "M3_common", "main: IIRS vs M3, same wavelengths"),
         ("IIRS_scaled", "M3_common", "IIRS with the brightness offset removed"),
         ("M3_common", "M3_full", "M3 range cut vs M3 full range")]


def smooth(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.where(mask & np.isfinite(img), img, 0.0)
    w = (mask & np.isfinite(img)).astype(float)
    num, den = gaussian_filter(v, SMOOTH_PX), gaussian_filter(w, SMOOTH_PX)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(mask & (den > 0.2), out, np.nan)


def as_image(v: np.ndarray, joint: np.ndarray) -> np.ndarray:
    img = np.full(joint.shape, np.nan)
    img[joint] = v
    return img


def quantities(joint: np.ndarray) -> dict:
    """{quantity: {config: per-joint-pixel vector}} for every mapped output."""
    q = {}
    for c in cc.CONFIGS:
        f = cc.PRODUCTS / ("p1_%s_depth970.npy" % c)
        if f.exists():
            q.setdefault("P1 band depth 970 nm", {})[c] = np.load(f).astype(float)
        f = cc.PRODUCTS / ("p3_full_%s_median_params.npy" % c)
        if f.exists():
            p = np.load(f).astype(float)
            for j, name in enumerate(kc.PARAMS):
                q.setdefault("P3 ANN %s" % name, {})[c] = p[:, j]
        f = cc.PRODUCTS / ("p3_ilm1550_%s.npy" % c)
        if f.exists():
            q.setdefault("P3 ilmenite (1550-nm band depth)", {})[c] = np.load(f).astype(float)
    return q


def main() -> int:
    cc.banner(STAGE, "M3 vs IIRS, for each paper")
    cc.verify_manifest(STAGE)
    joint = np.load(cc.PRODUCTS / "joint_mask.npy")
    zone_img = np.load(cc.PRODUCTS / "zone_map.npy")
    zone = zone_img[joint]
    q = quantities(joint)

    stats, zrows = [], []
    for name, byc in q.items():
        for c, v in byc.items():
            for i, (zn, _, _) in enumerate(cc.ZONES):
                s = v[zone == i]
                zrows.append({"quantity": name, "config": c, "zone": zn,
                              "median": float(np.nanmedian(s)), "p25": float(np.nanpercentile(s, 25)),
                              "p75": float(np.nanpercentile(s, 75))})
        for test, ref, label in PAIRS:
            if test not in byc or ref not in byc:
                continue
            x, y = byc[ref], byc[test]
            px = cc.r2_bias_rmse(x, y)
            sx, sy = smooth(as_image(x, joint), joint), smooth(as_image(y, joint), joint)
            sm = cc.r2_bias_rmse(sx[joint], sy[joint])
            zx = np.array([np.nanmedian(x[zone == i]) for i in range(len(cc.ZONES))])
            zy = np.array([np.nanmedian(y[zone == i]) for i in range(len(cc.ZONES))])
            spread = float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
            stats.append({"quantity": name, "test": test, "reference": ref, "comparison": label,
                          "pixel_r": px["pearson_r"], "pixel_bias": px["bias"], "pixel_rmse": px["rmse"],
                          "smoothed_r": sm["pearson_r"], "smoothed_bias": sm["bias"],
                          "smoothed_rmse": sm["rmse"],
                          "zone_max_abs_diff": float(np.nanmax(np.abs(zy - zx))),
                          "zone_rank_agree": bool((np.argsort(zx) == np.argsort(zy)).all()),
                          "reference_IQR": spread,
                          "zone_max_abs_diff_over_IQR": float(np.nanmax(np.abs(zy - zx))) / spread
                          if spread > 0 else np.nan, "n": px["n"]})
    stats = pd.DataFrame(stats)
    zt = pd.DataFrame(zrows)
    stats.to_csv(cc.TABLES / "step04_agreement.csv", index=False)
    zt.to_csv(cc.TABLES / "step04_zone_medians.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(stats[stats.test == "IIRS_common"][["quantity", "pixel_r", "smoothed_r",
                                                  "smoothed_bias", "zone_max_abs_diff",
                                                  "zone_max_abs_diff_over_IQR",
                                                  "zone_rank_agree"]].round(3).to_string(index=False))

    # ---- Paper 1 ROI-level results side by side -----------------------------
    f = cc.TABLES / "step02_paper1_targets.csv"
    if f.exists():
        t = pd.read_csv(f)
        cols = [c for c in ("d_mix", "Delta", "I_sm", "r2_map") if c in t]
        wide = t.pivot(index="target", columns="config", values=cols)
        wide.to_csv(cc.TABLES / "step04_paper1_targets_side_by_side.csv")
        print("\n  Paper 1, per target:\n" + wide.round(3).to_string())
    f = cc.TABLES / "step02_paper1_abundances.csv"
    if f.exists():
        a = pd.read_csv(f)
        wide = a.pivot_table(index=["target", "category"], columns="config", values="median_wt_pct")
        wide.to_csv(cc.TABLES / "step04_paper1_abundances_side_by_side.csv")
        if {"IIRS_common", "M3_common"} <= set(wide.columns):
            d = (wide["IIRS_common"] - wide["M3_common"]).abs()
            print("\n  Paper 1 abundances: max |IIRS - M3| (common range) %.1f wt%%, median %.1f"
                  % (d.max(), d.median()))

    # ---- figures ------------------------------------------------------------
    show = [n for n in ("P1 band depth 970 nm", "P3 ANN S", "P3 ANN nFe", "P3 ANN CPX",
                        "P3 ANN PLG", "P3 ANN ILM", "P3 ilmenite (1550-nm band depth)") if n in q]
    if show:
        r0, r1 = 0, joint.shape[0]
        c1 = int(np.flatnonzero(joint.any(axis=0)).max()) + 1
        fig, ax = plt.subplots(2, len(show), figsize=(2.1 * len(show), 11))
        ax = np.atleast_2d(ax).reshape(2, -1)
        for k, n in enumerate(show):
            a_ = smooth(as_image(q[n]["M3_common"], joint), joint)[r0:r1, :c1]
            b_ = smooth(as_image(q[n]["IIRS_common"], joint), joint)[r0:r1, :c1]
            lo, hi = np.nanpercentile(np.r_[a_[np.isfinite(a_)], b_[np.isfinite(b_)]], [2, 98])
            for r, (img, who) in enumerate(((a_, "M3"), (b_, "IIRS"))):
                im = ax[r, k].imshow(img, vmin=lo, vmax=hi, cmap="viridis", aspect="auto")
                ax[r, k].set_title("%s\n%s" % (who, n.replace("P3 ", "").replace("P1 ", "")), fontsize=8)
                ax[r, k].set_xticks([])
                ax[r, k].set_yticks([])
            fig.colorbar(im, ax=ax[:, k], orientation="horizontal", fraction=0.03, pad=0.02)
        fig.suptitle("Common range 0.85-2.50 um, smoothed (sigma 0.84 km); same pixels in both rows")
        fig.savefig(cc.FIGURES / "step04_maps_M3_vs_IIRS.png", dpi=100)
        plt.close(fig)

        fig, ax = plt.subplots(1, len(show), figsize=(2.6 * len(show), 3))
        for k, n in enumerate(show):
            x = smooth(as_image(q[n]["M3_common"], joint), joint)[joint]
            y = smooth(as_image(q[n]["IIRS_common"], joint), joint)[joint]
            ok = np.isfinite(x) & np.isfinite(y)
            ax[k].hexbin(x[ok], y[ok], gridsize=45, bins="log", mincnt=1, cmap="Greys")
            lo, hi = np.nanpercentile(np.r_[x[ok], y[ok]], [1, 99])
            ax[k].plot([lo, hi], [lo, hi], "r-", lw=0.8)
            ax[k].set_xlim(lo, hi)
            ax[k].set_ylim(lo, hi)
            ax[k].set_title(n.replace("P3 ", "").replace("P1 ", ""), fontsize=8)
            ax[k].set_xlabel("M3", fontsize=8)
            ax[k].set_ylabel("IIRS", fontsize=8)
        fig.tight_layout()
        fig.savefig(cc.FIGURES / "step04_scatter_smoothed.png", dpi=100)
        plt.close(fig)

    # ---- the IIRS segment-start artefact ---------------------------------------
    wl_m, wl_i = cc.wavelengths_um("M3"), cc.wavelengths_um("IIRS")
    cm, ci = cc.read_cube("M3"), cc.read_cube("IIRS")
    sm_ = np.nanmedian(cm[:, joint], axis=1)
    si_ = np.nanmedian(ci[:, joint], axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for wl, s, lab in ((wl_m, sm_, "M3"), (wl_i, si_, "IIRS")):
        w = cc.in_common(wl) & np.isfinite(s)
        ax.plot(wl[w], s[w] / np.polyval(np.polyfit(wl[w], s[w], 1), wl[w]), ".-", ms=3, label=lab)
    for lo, hi in cc.IIRS_GAPS_UM:
        ax.axvspan(lo, hi, color="0.85")
        ax.axvline(hi, color="C3", lw=0.8, ls=":")
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("scene median / straight-line fit")
    ax.set_title("IIRS dips start at the first band of each detector segment (red dotted); grey = IIRS gaps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cc.FIGURES / "step04_iirs_segment_artefact.png", dpi=110)
    plt.close(fig)

    cc.verify_manifest(STAGE)
    print("\n  STEP 04 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
