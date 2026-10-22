"""
Step 00c -- pull what the paper only shows as figures out of the PDF.

  * Fig. 3  -- the mean-albedo image of the test area and the four site crosses.
              The paper gives neither the M3 observation ID nor site coordinates;
              step04 recovers both by matching this raster to M3 data.
  * Fig. 12 -- the measured M3 spectrum (blue) at each test site, 21 points,
              0.54-2.54 um, in the RELAB photometric system. Used to confirm the
              recovered sites and our preprocessing.
  * Fig. A1 -- alpha_Fe(lambda), the paper's own tabulation of the nFe absorption.

Calibration is automatic: each plot frame is found as the longest dark lines, and
the x-axis tick pitch (ticks every 0.1 um in both figures) is measured from the
frame edge. The known first tick (0.54 um in Fig. 12, 0.3 um in Fig. A1) anchors
x; the frame top/bottom anchor y. An overlay PNG is written for every digitized
figure so the points can be inspected against the source.

Crosses in Fig. 3 were located by zooming on the raster (4x nearest-neighbour);
the centre of each "+" is recorded below to +-1 raster pixel.
"""
from __future__ import annotations

import io
import sys

import cv2
import numpy as np
import pandas as pd
from pypdf import PdfReader
from PIL import Image

import korokhin_common as kc

STAGE = "step00c"

# (0-based page, image index on that page) in the supplied PDF
FIG03 = (1, 1)
FIG12 = (9, 0)
FIGA1 = (27, 0)

# Fig. 3 raster: the image area excludes the 4-px frame and the colour bar below y=858
FIG03_IMAGE_BOX = (4, 4, 591, 858)                 # x0, y0, x1, y1 in raster pixels
FIG03_CROSSES = {1: (480.0, 185.5), 2: (534.5, 156.0), 3: (302.0, 637.0), 4: (236.75, 702.5)}


def extract(page: int, idx: int) -> np.ndarray:
    r = PdfReader(str(kc.PAPER_PDF))
    im = list(r.pages[page].images)[idx]
    return np.array(Image.open(io.BytesIO(im.data)).convert("RGB"))


def frame(gray: np.ndarray) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the plot frame: the longest dark lines."""
    dark = gray < 110
    rows = dark.sum(axis=1)
    cols = dark.sum(axis=0)
    h, w = gray.shape
    top = int(np.argmax(rows[: h // 2]))
    bottom = h // 2 + int(np.argmax(rows[h // 2:]))
    left = int(np.argmax(cols[: w // 2]))
    right = w // 2 + int(np.argmax(cols[w // 2:]))
    return left, top, right, bottom


def tick_pitch(gray: np.ndarray, left: int, right: int, edge_row: int) -> float:
    """Spacing of the tick marks on a horizontal frame edge.

    The frame line is ~3 px thick and ticks may point inward or outward, so both
    sides are searched (rows edge-8..edge-4 and edge+4..edge+8) and the side with
    the most regular set of short dark segments wins. Pitch = median gap.
    """
    best = None
    for rows in ([edge_row - k for k in range(4, 9)], [edge_row + k for k in range(4, 9)]):
        band = (gray[rows, :] < 110).sum(axis=0) >= 4
        band[: left + 4] = False
        band[right - 3:] = False
        xs = np.flatnonzero(band)
        if xs.size < 4:
            continue
        groups = np.split(xs, np.flatnonzero(np.diff(xs) > 2) + 1)
        centres = np.array([g.mean() for g in groups])
        gaps = np.diff(centres)
        if gaps.size < 3:
            continue
        pitch = float(np.median(gaps))
        regular = int((np.abs(gaps - pitch) < 1.5).sum())
        if best is None or regular > best[0]:
            best = (regular, pitch)
    if best is None:
        raise RuntimeError("no ticks found")
    return best[1]


def digitize_curve(rgb, mask_fn, x_of_lam, lam, y_of_px, halfwidth=2, rows=None):
    """Row of the masked colour at each wavelength column.

    Only rows inside the plot frame are considered (`rows` = (first, last)); within
    a column the masked pixels are split into contiguous runs and the largest run is
    taken -- the plotted line is several pixels thick, JPEG fringes of other curves
    and labels are one or two.
    """
    m = mask_fn(rgb)
    if rows is not None:
        keep = np.zeros(m.shape[0], dtype=bool)
        keep[rows[0]: rows[1] + 1] = True
        m = m & keep[:, None]
    out = []
    for l in lam:
        x = int(round(x_of_lam(l)))
        ys = np.flatnonzero(m[:, max(0, x - halfwidth): x + halfwidth + 1].any(axis=1))
        if ys.size == 0:
            out.append(np.nan)
            continue
        runs = np.split(ys, np.flatnonzero(np.diff(ys) > 3) + 1)
        best = max(runs, key=len)
        out.append(y_of_px(np.median(best)))
    return np.array(out)


def fig12() -> pd.DataFrame:
    rgb = extract(*FIG12)
    h, w = rgb.shape[:2]
    a_int = lambda a: a.astype(int)
    # blue-dominant pixels; the thinner right-hand panels render the line as dark blue (B~100)
    blue = lambda a: ((a_int(a)[..., 2] - np.maximum(a_int(a)[..., 0], a_int(a)[..., 1]) > 35) & (a_int(a)[..., 2] > 45)
                      & (a_int(a)[..., 0] < 70) & (a_int(a)[..., 1] < 70))   # not magenta edges
    panels = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
    rows = []
    over = rgb.copy()
    for site, (pr, pc) in panels.items():
        y0, y1 = pr * h // 2, (pr + 1) * h // 2
        x0, x1 = pc * w // 2, (pc + 1) * w // 2
        sub = rgb[y0:y1, x0:x1]
        g = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
        left, top, right, bottom = frame(g)
        pitch = tick_pitch(g, left, right, bottom)              # 0.1 um per tick
        x_of = lambda l: left + 1 + (l - 0.54) / 0.1 * pitch
        y_of = lambda py: 0.5 * (bottom + 1 - py) / (bottom - top)  # frame spans 0.0 ... 0.5
        refl = digitize_curve(sub, blue, x_of, kc.M3_GRID_UM, y_of, rows=(top + 3, bottom - 3))
        for l, r in zip(kc.M3_GRID_UM, refl):
            rows.append({"site": site, "lam_um": l, "refl": r})
            if np.isfinite(r):
                cv2.circle(over, (int(x0 + x_of(l)), int(y0 + bottom - r / 0.5 * (bottom - top))),
                           4, (0, 200, 0), 1)
        print("  Fig.12 site %d: frame x %d-%d y %d-%d, tick pitch %.2f px/0.1um, %d/21 points"
              % (site, left, right, top, bottom, pitch, int(np.isfinite(refl).sum())))
    cv2.imwrite(str(kc.PDF_IMAGES / "fig12_overlay.png"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    return pd.DataFrame(rows)


def figA1() -> pd.DataFrame:
    rgb = extract(*FIGA1)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    left, top, right, bottom = frame(g)
    pitch = tick_pitch(g, left, right, bottom)
    x_of = lambda l: left + 1 + (l - 0.3) / 0.1 * pitch
    y_of = lambda py: 90.0 * (bottom + 1 - py) / (bottom - top)  # frame spans 0 ... 90
    lam = np.round(np.arange(0.30, 2.601, 0.02), 2)
    inner = np.zeros_like(g, dtype=bool)
    inner[top + 3: bottom - 2, left + 3: right - 2] = True
    black = lambda a: (cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) < 90) & inner
    alpha = digitize_curve(rgb, black, x_of, lam, y_of, halfwidth=1)
    print("  Fig.A1: frame x %d-%d y %d-%d, tick pitch %.2f px/0.1um, %d/%d points"
          % (left, right, top, bottom, pitch, int(np.isfinite(alpha).sum()), lam.size))
    over = rgb.copy()
    for l, a in zip(lam, alpha):
        if np.isfinite(a):
            cv2.circle(over, (int(x_of(l)), int(bottom - a / 90.0 * (bottom - top))), 3, (220, 0, 0), 1)
    cv2.imwrite(str(kc.PDF_IMAGES / "figA1_overlay.png"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    return pd.DataFrame({"lam_um": lam, "alpha_fe": alpha}).dropna()


def fig03() -> pd.DataFrame:
    rgb = extract(*FIG03)
    cv2.imwrite(str(kc.PDF_IMAGES / "fig03_raster.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    x0, y0, x1, y1 = FIG03_IMAGE_BOX
    rows = [{"site": s, "raster_x": x, "raster_y": y,
             "frac_x": (x - x0) / (x1 - x0), "frac_y": (y - y0) / (y1 - y0)}
            for s, (x, y) in FIG03_CROSSES.items()]
    return pd.DataFrame(rows)


def main() -> int:
    kc.banner(STAGE, "digitize Figs. 3, 12 and A1 from the supplied PDF")
    s = fig03()
    s.to_csv(kc.DIGITIZED / "fig03_sites.csv", index=False)
    print("  Fig.3: %d site crosses recorded (image box %s)" % (len(s), FIG03_IMAGE_BOX))
    f12 = fig12()
    f12.to_csv(kc.DIGITIZED / "fig12_site_spectra.csv", index=False)
    a1 = figA1()
    a1.to_csv(kc.DIGITIZED / "figA1_alpha_fe.csv", index=False)
    # quick sanity: the paper's own site albedos (Table 9 sigma/sigma_rel) vs Fig. 12 means
    implied = pd.read_csv(kc.PAPER_TABLES / "derived_site_mean_albedo.csv").set_index("site")["mean"]
    for site, g in f12.groupby("site"):
        print("  site %d: mean of digitized Fig.12 spectrum %.4f   <A> implied by tables %.4f"
              % (site, g["refl"].mean(), implied.loc[site]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
