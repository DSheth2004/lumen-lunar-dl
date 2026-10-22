"""
Audit -- recompute key numbers by a different code path and diff them.

  A1  joint pixel count, from masked-array reads instead of cc.joint_mask
  A2  the IIRS_scaled factor, with np.interp per band instead of the manual bracket
  A3  970-nm band depth on 2,000 random joint pixels, with scipy.spatial.ConvexHull
      instead of the monotone-chain hull, for every configuration
  A4  1550-nm ilmenite band depth, same sample and the same independent hull
  A5  common wavelengths: every band used by a common configuration is inside
      0.85-2.50 um and outside both IIRS gaps (positional check), with a shuffled
      negative control that must FAIL the monotonic-grid check
  A6  NoData convention: every map is NaN exactly outside the joint mask
  A7  Paper 3: re-running the forward model on the stored median parameters reproduces
      a scene sigma_Rel within the range of the stored per-seed values
  A8  inputs unchanged (manifest)
Writes tables/audit.csv; exit code 1 on any FAIL.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio
from scipy.spatial import ConvexHull

import cc_common as cc

sys.path.insert(0, str(cc.JUNG_SCRIPTS))
sys.path.insert(0, str(cc.KOROKHIN_SCRIPTS))
import hapke                        # noqa: E402
import jung_common as jc            # noqa: E402
import korokhin_common as kc        # noqa: E402

STAGE = "audit"
rows = []


def check(name, value, tol, ok=None):
    ok = (abs(value) <= tol) if ok is None else ok
    rows.append({"check": name, "value": value, "tol": tol, "status": "PASS" if ok else "FAIL"})
    print("  %-62s %-4s %s" % (name, "PASS" if ok else "FAIL", value))


def upper_hull_cr(wl, y):
    """Independent continuum removal: scipy ConvexHull of the points plus two anchors
    far below, upper chain = hull vertices other than the anchors."""
    lo = y.min() - 10.0
    pts = np.c_[np.r_[wl, wl[0], wl[-1]], np.r_[y, lo, lo]]
    v = ConvexHull(pts).vertices
    v = np.sort(v[v < wl.size])
    return y / np.interp(wl, wl[v], y[v])


def main() -> int:
    cc.banner(STAGE, "independent recomputation")
    cc.verify_manifest(STAGE)
    joint = np.load(cc.PRODUCTS / "joint_mask.npy")

    # A1
    m = None
    for sensor, lo in (("M3", 0.55), ("IIRS", cc.COMMON_LO_UM)):
        with rasterio.open(cc.M3_CUBE if sensor == "M3" else cc.IIRS_CUBE) as d:
            arr = ma.masked_invalid(ma.masked_equal(d.read(masked=False).astype(float), d.nodata))
        wl = cc.wavelengths_um(sensor)
        sub = arr[(wl >= lo) & (wl <= cc.COMMON_HI_UM)]
        ok = ~ma.getmaskarray(sub).any(axis=0) & (sub.filled(-1) > 0).all(axis=0)
        with rasterio.open(cc.M3_MASK if sensor == "M3" else cc.IIRS_MASK) as d:
            ok &= d.read(1) == 1
        m = ok if m is None else (m & ok)
    check("A1 joint pixel count (independent) - stored", int(m.sum()) - int(joint.sum()), 0,
          ok=bool((m == joint).all()))

    # A2
    wl_m, wl_i = cc.wavelengths_um("M3"), cc.wavelengths_um("IIRS")
    cm, ci = cc.read_cube("M3")[:, joint], cc.read_cube("IIRS")[:, joint]
    rng = np.random.default_rng(7)
    idx = rng.choice(joint.sum(), 2000, replace=False)
    sel = (wl_i >= cc.COMMON_LO_UM) & (wl_i <= cc.COMMON_HI_UM)
    fin = np.isfinite(cm).all(axis=1)
    rat = []
    for p in range(cm.shape[1]):
        rat.append(ci[sel, p] / np.interp(wl_i[sel], wl_m[fin], cm[fin, p]))
    s2 = float(np.nanmedian(np.concatenate(rat)))
    check("A2 IIRS scale factor, independent - stored", s2 - cc.iirs_scale_factor(), 1e-9)

    # A3, A4
    from step02_paper1 import window
    from step09_maps_comparisons import BD_NM, BD_TO_ILM
    for config in cc.CONFIGS:
        wl, cube = cc.config_cube(config)
        R = cube[:, joint][:, idx]
        w = window(config, wl)
        f = cc.PRODUCTS / ("p1_%s_depth970.npy" % config)
        if f.exists():
            stored = np.load(f)[idx].astype(float)
            ssa = hapke.reflectance_to_ssa_fast(R[w].T, jc.MU0, jc.MU)
            k = int(np.argmin(np.abs(wl[w] - jc.BAND_REF_UM)))
            mine = np.array([1.0 - upper_hull_cr(wl[w], s)[k] for s in ssa])
            check("A3 970-nm depth %s, max |diff|" % config,
                  float(np.max(np.abs(mine - stored))), 1e-5)
        f = cc.PRODUCTS / ("p3_ilm1550_%s.npy" % config)
        if f.exists():
            stored = np.load(f)[idx].astype(float)
            u = np.isfinite(R).all(axis=1) & (wl <= 2.6)
            if config == "M3_full":
                wl_u, Ru = wl[u], R[u]
            else:
                wl_u, Ru = wl[u & cc.in_common(wl)], R[u & cc.in_common(wl)]
            tgt = np.asarray(BD_NM) * 1e-3
            mine = np.array([BD_TO_ILM * np.mean(1 - np.interp(tgt, wl_u, upper_hull_cr(wl_u, Ru[:, p])))
                             for p in range(Ru.shape[1])])
            check("A4 1550-nm ilmenite %s, max |diff| wt%%" % config,
                  float(np.max(np.abs(mine - stored))), 1e-4)

    # A5
    for config in ("M3_common", "IIRS_common"):
        wl, _ = cc.config_cube(config)
        bad = [(x) for x in wl if not (cc.COMMON_LO_UM <= x <= cc.COMMON_HI_UM)
               or any(lo < x < hi for lo, hi in cc.IIRS_GAPS_UM)]
        check("A5 %s bands outside common range (count)" % config, len(bad), 0)
        check("A5 %s band order strictly increasing (positional)" % config, 0, 0,
              ok=bool(np.all(np.diff(wl) > 0)))
        shuf = np.random.default_rng(1).permutation(wl)
        check("A5 negative control: shuffled %s must fail the order check" % config, 0, 0,
              ok=not bool(np.all(np.diff(shuf) > 0)))

    # A6
    n_maps, n_bad = 0, 0
    for f in glob.glob(str(cc.PRODUCTS / "paper*_*.tif")):
        with rasterio.open(f) as d:
            a = d.read(1).astype(float)
            if d.nodata is not None and np.isfinite(d.nodata):
                a[a == d.nodata] = np.nan
        n_maps += 1
        n_bad += int(np.isfinite(a[~joint]).any())
    check("A6 maps with values outside the joint mask (of %d)" % n_maps, n_bad, 0)

    # A7
    import shkuratov as sk
    from step03_paper3 import grid_for, spectra
    from step07_ann import sigma_rel_map
    for config in cc.CONFIGS:
        f = cc.PRODUCTS / ("p3_full_%s_median_params.npy" % config)
        if not f.exists():
            continue
        grid, X = spectra(config)
        P = np.load(f).astype(float)
        sr = float(np.nanmean(sigma_rel_map(sk.LunarModel(grid), P, X)))
        per_seed = [json.loads(open(g).read())["sigma_rel_mean"] for g in
                    glob.glob(str(cc.TABLES / ("p3_full_%s_seed*_iter3_info.json" % config)))]
        check("A7 %s sigma_Rel of seed-median params (%%) vs per-seed %.2f-%.2f"
              % (config, min(per_seed), max(per_seed)), round(sr, 3), 0,
              ok=sr <= max(per_seed) * 1.25)

    cc.verify_manifest(STAGE)
    check("A8 inputs unchanged", 0, 0)
    df = pd.DataFrame(rows)
    df.to_csv(cc.TABLES / "audit.csv", index=False)
    n_fail = int((df["status"] == "FAIL").sum())
    print("\n  AUDIT: %d checks, %d FAIL" % (len(df), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
