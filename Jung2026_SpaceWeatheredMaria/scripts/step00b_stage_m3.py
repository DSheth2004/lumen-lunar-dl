"""
Step 00b -- stage the M3 Aristarchus regions of interest into data/.

Downloads only what is needed: the L2 label, the ENVI header (for band centres),
enough of the L1B LOC file to locate the ROI, and the RFL image lines that cover
it.  The full strips are 1.8-3.1 GB each; this pulls a few MB.

Re-runs the manifest afterwards so the guard covers the new inputs.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

import jung_common as jc
import m3_fetch

STAGE = "step00b"


def stage_wavelengths(hdr_text: str) -> np.ndarray:
    """Pull the 85 band centres out of the RFL ENVI header."""
    m = re.search(r"wavelength\s*=\s*\{(.*?)\}", hdr_text, re.S | re.I)
    if not m:
        raise RuntimeError("no wavelength list in the ENVI header")
    vals = [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    return np.asarray(vals)


def main() -> int:
    jc.banner(STAGE, "staging M3 Aristarchus ROIs (range requests only)")
    jc.M3_STAGED.mkdir(parents=True, exist_ok=True)

    prov = []
    wl = None

    for roi, cfg in m3_fetch.SCENES.items():
        urls = m3_fetch.scene_urls(cfg)
        print("\n  --- %s : %s ---" % (roi, cfg["product"]))

        # ---- L2 label -----------------------------------------------------
        lbl = m3_fetch.http_get(urls["label"]).decode("latin-1")
        (jc.M3_STAGED / ("%s_L2.LBL" % roi)).write_text(lbl)
        info = m3_fetch.parse_label(lbl)
        n_lines = info["LINES"]
        print("    label: %d lines x %d samples x %d bands"
              % (n_lines, info["LINE_SAMPLES"], info["BANDS"]))
        thermal = re.search(r"THERMAL_CORRECTION_FLAG\s*=\s*\"?(\w+)", lbl)
        print("    THERMAL_CORRECTION_FLAG = %s" % (thermal.group(1) if thermal else "?"))

        # ---- ENVI header for band centres ---------------------------------
        if wl is None:
            hdr_url = urls["rfl"].replace("_RFL.IMG", "_RFL.HDR")
            hdr = m3_fetch.http_get(hdr_url).decode("latin-1")
            (jc.M3_STAGED / "rfl.hdr").write_text(hdr)
            wl = stage_wavelengths(hdr)
            pd.DataFrame({"band": np.arange(1, len(wl) + 1),
                          "wavelength_nm": wl}).to_csv(
                jc.M3_STAGED / "m3_wavelengths_nm.csv", index=False)
            print("    band centres: %d, %.2f-%.2f nm -> m3_wavelengths_nm.csv"
                  % (len(wl), wl.min(), wl.max()))
            prov.append({"roi": roi, "kind": "rfl_hdr", "url": hdr_url,
                         "bytes": len(hdr)})

        # ---- locate the ROI in the LOC file -------------------------------
        print("    scanning LOC for (%.2fN, %.2fE) ..." % (cfg["lat"], cfg["lon"]))
        loc = m3_fetch.find_roi_lines(urls["loc"], n_lines, cfg["lat"], cfg["lon"])
        print("    nearest pixel: line %d, sample %d (%.4f deg away); "
              "taking lines %d-%d" % (loc["line"], loc["sample"], loc["dist_deg"],
                                      loc["line_start"], loc["line_stop"]))
        pd.DataFrame([{**loc, "product": cfg["product"],
                       "lat": cfg["lat"], "lon": cfg["lon"]}]).to_csv(
            jc.M3_STAGED / ("%s_location.csv" % roi), index=False)

        # ---- fetch just those RFL lines -----------------------------------
        block = m3_fetch.fetch_rfl_block(urls["rfl"], loc["line_start"], loc["line_stop"])
        np.save(jc.M3_STAGED / ("%s_rfl_block.npy" % roi), block)
        mb = block.nbytes / 1e6
        print("    fetched %s block (%.1f MB) instead of the full strip" % (block.shape, mb))
        finite = np.isfinite(block).all(axis=1).sum()
        print("    %d of %d pixels finite across all bands"
              % (finite, block.shape[0] * block.shape[2]))

        prov += [
            {"roi": roi, "kind": "l2_label", "url": urls["label"], "bytes": len(lbl)},
            {"roi": roi, "kind": "loc_scan", "url": urls["loc"], "bytes": np.nan},
            {"roi": roi, "kind": "rfl_block", "url": urls["rfl"],
             "bytes": int(block.nbytes),
             "line_start": loc["line_start"], "line_stop": loc["line_stop"]},
        ]

    pd.DataFrame(prov).to_csv(jc.TABLES / "step00b_m3_provenance.csv", index=False)
    jc.write_manifest()
    jc.verify_manifest(STAGE)
    print("\n  manifest updated to include the M3 inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
