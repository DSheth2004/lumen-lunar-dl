"""
Step 04 -- recover the paper's M3 scene and its four test sites, then stage them.

The paper names only the Apollo 16 calibration image (M3G20090607T110414). The
Shapley test area is shown as Fig. 3 (mean albedo) with four crosses; neither the
observation ID nor any coordinate is printed. They are recovered here.

Decision rule (fixed before looking at the result):
  1. candidates = every L2 strip whose own footprint AND geolocation contain
     Shapley's centre (9.4 N, 56.9 E) to within 0.05 deg;
  2. for each, a mean-albedo window around Shapley is correlated (normalised
     cross-correlation) with the Fig. 3 raster, the crosses and labels blanked,
     over vertical extent (the figure's line count is unknown) and the four flips;
  3. accept the best strip only if its NCC exceeds every other by >= 0.05.
Two independent confirmations are then reported, not tuned:
  * the implied pixel count (fitted line span resampled to square ground pixels,
    x 304 samples) against the paper's 152,000 pixels (Table 12, iterations 2-3);
  * the site albedos (step05, once in the RELAB photometric system) against
    <A> implied by the paper's own tables and the digitized Fig. 12 spectra.

Staged, never whole strips: the scene's full 85-band lines, its LOC backplane,
and a 61-line block of the Apollo 16 image for re-deriving Eq. (2).
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import pandas as pd

import korokhin_common as kc
import m3_io

STAGE = "step04"
ACCEPT_DEG = 0.05
WINDOW_HALF = 700          # raw lines either side of Shapley searched for the Fig. 3 frame
NCC_MARGIN = 0.05
PAPER_PIXELS = 152_000
N_ROWS_PAPER = PAPER_PIXELS // 304          # 500 square-pixel rows
APOLLO_HALF = 30
# centre of the large bright-rimmed crater at lower left of Fig. 3 (raster px, read by eye)
FIG3_LARGE_CRATER = (97, 650)


def mean_albedo(block: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    sel = (wl_nm >= 540) & (wl_nm <= 2600)
    b = block[:, sel, :].astype(float)
    b[b <= -990] = np.nan
    return np.nanmean(b, axis=1)


def fig3_template() -> np.ndarray:
    img = cv2.imread(str(kc.PDF_IMAGES / "fig03_raster.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    x0, y0, x1, y1 = (4, 4, 591, 858)
    t = img[y0:y1, x0:x1].copy()
    sites = pd.read_csv(kc.DIGITIZED / "fig03_sites.csv")
    for _, s in sites.iterrows():            # blank each cross and its number label
        cx, cy = s["raster_x"] - x0, s["raster_y"] - y0
        t[int(cy - 26): int(cy + 22), int(cx - 50): int(cx + 22)] = np.nan
    return t


def ncc_search(alb: np.ndarray, tmpl: np.ndarray, heights) -> dict:
    fill_t = np.nanmean(tmpl)
    best = {"ncc": -1.0}
    for flipud in (False, True):
        for fliplr in (False, True):
            a = alb[::-1] if flipud else alb
            a = a[:, ::-1] if fliplr else a
            a = np.nan_to_num(a, nan=float(np.nanmean(a))).astype(np.float32)
            for h in heights:
                if h > a.shape[0]:
                    continue
                t = cv2.resize(np.nan_to_num(tmpl, nan=fill_t), (304, int(h)),
                               interpolation=cv2.INTER_AREA)
                r = cv2.matchTemplate(a, t, cv2.TM_CCOEFF_NORMED)
                k = int(np.argmax(r))
                if r.flat[k] > best["ncc"]:
                    best = {"ncc": float(r.flat[k]), "flipud": flipud, "fliplr": fliplr,
                            "height": int(h), "offset": int(k // r.shape[1])}
    return best


def ground_spacing(loc_url: str, line: int) -> tuple[float, float]:
    lon, lat = m3_io.read_loc_lines(loc_url, 304, line - 5, line + 6)
    la, lo = np.radians(lat), np.radians(lon)
    R = 1737.4e3

    def d(a1, o1, a2, o2):
        c = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(o1 - o2)
        return R * np.arccos(np.clip(c, -1, 1))
    cross = float(np.nanmedian(d(la[:, 1:], lo[:, 1:], la[:, :-1], lo[:, :-1])))
    along = float(np.nanmedian(d(la[1:], lo[1:], la[:-1], lo[:-1])))
    return cross, along


def main() -> int:
    kc.banner(STAGE, "recover the Shapley scene and test sites from Fig. 3; stage M3 data")
    kc.verify_manifest(STAGE)
    kc.M3_DIR.mkdir(parents=True, exist_ok=True)
    lat, lon = kc.SHAPLEY_LATLON

    # ---- 1. candidates --------------------------------------------------
    cands = m3_io.ode_candidates(lat, lon)
    rows = []
    for c in cands:
        u = m3_io.paths_from_label_url(c["label_url"])
        lab = m3_io.read_label(u["l2_lbl"])
        loc = m3_io.locate_pixel(u["loc"], lab["LINES"], lab["LINE_SAMPLES"], lat, lon)
        rows.append({**c, **u, **lab, **{"shapley_" + k: v for k, v in loc.items()}})
    cand = pd.DataFrame(rows)
    cand["contains_shapley"] = cand["shapley_dist_deg"] <= ACCEPT_DEG
    print("  %d strips with a plausible footprint; %d contain Shapley's centre"
          % (len(cand), int(cand["contains_shapley"].sum())))

    # ---- 2. correlate each with Fig. 3 ---------------------------------
    tmpl = fig3_template()
    wl = m3_io.read_hdr_wavelengths(cand.iloc[0]["rfl_hdr"])
    results = []
    for _, c in cand[cand["contains_shapley"]].iterrows():
        i0 = max(0, int(c["shapley_line"]) - WINDOW_HALF)
        i1 = min(int(c["LINES"]), int(c["shapley_line"]) + WINDOW_HALF)
        alb = np.vstack([mean_albedo(m3_io.fetch_rfl_lines(c["rfl"], 304, 85, a, min(a + 200, i1)), wl)
                         for a in range(i0, i1, 200)])
        coarse = ncc_search(alb, tmpl, range(300, 1301, 10))
        fine = ncc_search(alb, tmpl, range(coarse["height"] - 12, coarse["height"] + 13))
        cross, along = ground_spacing(c["loc"], int(c["shapley_line"]))
        fine.update({"pdsid": c["pdsid"], "window_first_line": i0, "cross_m": cross,
                     "along_m": along, "shapley_sample": int(c["shapley_sample"]),
                     "implied_square_rows": fine["height"] * along / cross})
        fine["implied_pixels"] = 304 * fine["implied_square_rows"]
        # where does this strip's GEOLOCATED Shapley centre fall in the Fig. 3 raster?
        # (independent of the correlation: Shapley is the large crater at lower left)
        fine["shapley_fig3_x"] = 4 + (int(c["shapley_sample"]) + 0.5) / 304.0 * 587
        fine["shapley_fig3_y"] = 4 + (int(c["shapley_line"]) - i0 - fine["offset"] + 0.5) \
            / fine["height"] * 854
        results.append(fine)
        print("  %s  NCC %.3f  flips(ud,lr)=(%s,%s)  span %d lines from %d  pixel %.0f x %.0f m"
              "  -> %.0f square rows = %.0f px (paper: %d)"
              % (c["pdsid"], fine["ncc"], fine["flipud"], fine["fliplr"], fine["height"],
                 i0 + fine["offset"], cross, along, fine["implied_square_rows"],
                 fine["implied_pixels"], PAPER_PIXELS))
        print("      geolocated Shapley centre lands at Fig. 3 raster (%.0f, %.0f); the large"
              " crater there is centred near (%d, %d)"
              % (fine["shapley_fig3_x"], fine["shapley_fig3_y"], *FIG3_LARGE_CRATER))
    res = pd.DataFrame(results).sort_values("ncc", ascending=False).reset_index(drop=True)
    res.to_csv(kc.TABLES / "step04_scene_candidates.csv", index=False)
    cand.drop(columns=["label_url"]).to_csv(kc.TABLES / "step04_ode_candidates.csv", index=False)
    if len(res) == 0:
        sys.exit("[%s] STOP: no strip contains Shapley -- the scene cannot be identified from Fig. 3" % STAGE)
    margin = res["ncc"].iloc[0] - (res["ncc"].iloc[1] if len(res) > 1 else -1.0)
    win = res.iloc[0]
    print("  winner %s, NCC margin over runner-up %.3f (rule: >= %.2f)" % (win["pdsid"], margin, NCC_MARGIN))
    if margin < NCC_MARGIN:
        sys.exit("[%s] STOP: scene identification ambiguous -- the scene cannot be identified from Fig. 3" % STAGE)
    if win["flipud"] or win["fliplr"]:
        sys.exit("[%s] STOP: winning match needs a flip; site mapping below assumes none" % STAGE)

    # ---- 3. scene extent: 500 square-pixel rows centred on the fitted frame -----
    c = cand.set_index("pdsid").loc[win["pdsid"]]
    fit_first = int(win["window_first_line"] + win["offset"])
    fit_span = int(win["height"])
    raw_span = int(round(N_ROWS_PAPER * win["cross_m"] / win["along_m"]))
    centre = fit_first + fit_span / 2.0
    first = int(round(centre - raw_span / 2.0))
    last = first + raw_span                       # exclusive
    block = m3_io.fetch_rfl_lines(c["rfl"], 304, 85, first, last)
    lon_b, lat_b = m3_io.read_loc_lines(c["loc"], 304, first, last)
    np.save(kc.M3_DIR / "scene_rfl.npy", block)
    np.save(kc.M3_DIR / "scene_loc.npy", np.stack([lon_b, lat_b]).astype(np.float64))
    pd.DataFrame({"band": np.arange(1, 86), "wavelength_nm": wl}).to_csv(
        kc.M3_DIR / "m3_wavelengths_nm.csv", index=False)

    # ---- 4. the four sites ----------------------------------------------
    sites = pd.read_csv(kc.DIGITIZED / "fig03_sites.csv")
    out = []
    for _, s in sites.iterrows():
        sample = s["frac_x"] * 304.0 - 0.5
        line_abs = fit_first + s["frac_y"] * fit_span - 0.5
        li, si = int(round(line_abs)) - first, int(round(sample))
        out.append({"site": int(s["site"]), "line_in_strip": int(round(line_abs)),
                    "line_in_scene": li, "sample": si,
                    "lat": float(lat_b[li, si]), "lon": float(lon_b[li, si])})
    sites_df = pd.DataFrame(out)
    sites_df.to_csv(kc.M3_DIR / "sites.csv", index=False)
    for _, s in sites_df.iterrows():
        print("  site %d: strip line %d, sample %d -> %.4f N, %.4f E"
              % (s["site"], s["line_in_strip"], s["sample"], s["lat"], s["lon"]))

    # ---- 5. Apollo 16 calibration block (Eq. 2) -------------------------
    alat, alon = kc.APOLLO16_LATLON
    a16 = [x for x in m3_io.ode_candidates(alat, alon) if x["pdsid"].startswith(kc.EQ2_SCENE)]
    if not a16:
        # ODE's footprint for this strip may be one of the 0-360 artefacts: build URLs directly
        a16 = [{"label_url": m3_io.PDS + "CH1M3_0004/DATA/20090415_20090816/200906/L2/%s_V01_L2.LBL"
                % kc.EQ2_SCENE}]
    ua = m3_io.paths_from_label_url(a16[0]["label_url"])
    la = m3_io.read_label(ua["l2_lbl"])
    pa = m3_io.locate_pixel(ua["loc"], la["LINES"], la["LINE_SAMPLES"], alat, alon)
    a0 = max(0, pa["line"] - APOLLO_HALF)
    a1 = min(la["LINES"], pa["line"] + APOLLO_HALF + 1)
    np.save(kc.M3_DIR / "apollo16_rfl.npy", m3_io.fetch_rfl_lines(ua["rfl"], 304, 85, a0, a1))
    lon_a, lat_a = m3_io.read_loc_lines(ua["loc"], 304, a0, a1)
    np.save(kc.M3_DIR / "apollo16_loc.npy", np.stack([lon_a, lat_a]))
    print("  Apollo 16: %s line %d sample %d (%.4f deg from the LM site)"
          % (ua["base"], pa["line"], pa["sample"], pa["dist_deg"]))

    meta = {"scene": c["base"], "rfl_url": c["rfl"], "loc_url": c["loc"],
            "strip_lines": int(c["LINES"]), "first_line": first, "last_line_exclusive": last,
            "fig3_fit_first_line": fit_first, "fig3_fit_span_lines": fit_span,
            "fig3_ncc": float(win["ncc"]), "ncc_margin": float(margin),
            "cross_track_m": float(win["cross_m"]), "along_track_m": float(win["along_m"]),
            "rows_after_square_resampling": N_ROWS_PAPER,
            "implied_pixels_of_fig3_frame": float(win["implied_pixels"]),
            "apollo16": {"scene": ua["base"], "rfl_url": ua["rfl"], "first_line": int(a0),
                         "centre_line": int(pa["line"]), "centre_sample": int(pa["sample"]),
                         "dist_deg": float(pa["dist_deg"])}}
    (kc.M3_DIR / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    kc.write_manifest()
    kc.verify_manifest(STAGE)
    print("  staged %d raw lines (%d-%d) x 85 bands x 304 samples; manifest resealed"
          % (last - first, first, last - 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
