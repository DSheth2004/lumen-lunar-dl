"""
Fetch just the Aristarchus regions of interest out of the M3 L2 archive.

The M3 Level 2 reflectance strips are 1.8-3.1 GB each because they run pole to
pole, but the label declares

    RECORD_BYTES = 103360 = 304 samples x 85 bands x 4 bytes
    BAND_STORAGE_TYPE = LINE_INTERLEAVED

so one image line is exactly one record and any line can be addressed directly.
This module therefore pulls the (small) LOC file to locate the ROI, then uses
HTTP range requests to download only the handful of RFL lines that cover it --
a few MB instead of several GB.

Provenance for everything fetched is written to data/m3_aristarchus/.

M3 L2 is described by its own label as "pixel located, thermal corrected,
photometry corrected, reflectance data": the thermal correction is the standard
one applied during PDS L2 processing, NOT the Li & Milliken (2016) correction
the paper applies.  That difference is recorded as a deviation.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://pds-imaging.jpl.nasa.gov/data/m3/"

# Scenes containing the two ROIs of paper Figure 5, from an ODE footprint query.
SCENES = {
    "roi_b": {
        "product": "M3G20090208T160125_V01",
        "dir": "CH1M3_0004/DATA/20081118_20090214/200902/L2/",
        "loc_dir": "CH1M3_0003/DATA/20081118_20090214/200902/L1B/",
        "loc_product": "M3G20090208T160125_V03",
        "lat": 21.37, "lon": 320.54,
        "label": "Figure 5(b) fresh-crater ROI",
    },
    "roi_c": {
        "product": "M3G20090208T233940_V01",
        "dir": "CH1M3_0004/DATA/20081118_20090214/200902/L2/",
        "loc_dir": "CH1M3_0003/DATA/20081118_20090214/200902/L1B/",
        "loc_product": "M3G20090208T233940_V03",
        "lat": 31.76, "lon": 315.82,
        "label": "Figure 5(c) fresh-crater ROI",
    },
}

N_SAMPLES = 304
N_BANDS_RFL = 85
N_BANDS_LOC = 3
BYTES_PER = 4
RFL_RECORD = N_SAMPLES * N_BANDS_RFL * BYTES_PER      # 103360
LOC_RECORD = N_SAMPLES * N_BANDS_LOC * 8              # LOC is 64-bit


_SESSION = None


def _session():
    """One keep-alive session with retries.

    The archive drops connections (and the local resolver starts failing) if it
    is hit with hundreds of rapid independent range requests, so connections are
    pooled and retried with backoff.
    """
    global _SESSION
    if _SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        retry = Retry(total=6, backoff_factor=1.0,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SESSION = s
    return _SESSION


def http_get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = "bytes=%d-%d" % byte_range
    last = None
    for attempt in range(6):
        try:
            r = _session().get(url, headers=headers, timeout=300)
            r.raise_for_status()
            return r.content
        except Exception as exc:          # DNS blips included
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("GET failed after retries: %s (%s)" % (url, last))


def parse_label(text: str) -> dict:
    out = {}
    for key in ("LINES", "LINE_SAMPLES", "BANDS", "RECORD_BYTES", "FILE_RECORDS"):
        m = re.search(r"^\s*%s\s*=\s*(\d+)" % key, text, re.M)
        if m:
            out[key] = int(m.group(1))
    return out


def loc_record_bytes(sample_bits: int) -> int:
    return N_SAMPLES * N_BANDS_LOC * (sample_bits // 8)


def find_roi_lines(loc_url: str, n_lines: int, lat: float, lon: float,
                   sample_bits: int = 64, n_probe: int = 24,
                   half_window: int = 24, verbose: bool = True):
    """Locate the image line/sample nearest (lat, lon) in a LOC file.

    Along an M3 strip latitude is very nearly monotonic in line number, so
    instead of scanning every line we probe `n_probe` lines spread over the
    strip, fit line as a function of latitude, jump to the prediction, and then
    read one contiguous block around it.  That is ~30 requests rather than
    several hundred, which matters because the archive throttles.

    LOC bands are (longitude, latitude, radius), line-interleaved like the RFL.
    """
    rec = loc_record_bytes(sample_bits)
    dtype = np.dtype("<f8" if sample_bits == 64 else "<f4")

    def read_lines(i0, i1):
        """Read LOC lines [i0, i1) in ONE request. Returns (lons, lats) arrays."""
        i1 = min(i1, n_lines)
        raw = http_get(loc_url, (i0 * rec, i1 * rec - 1))
        a = np.frombuffer(raw, dtype=dtype)
        n = a.size // (N_SAMPLES * N_BANDS_LOC)
        if n == 0:
            return None, None
        a = a[:n * N_SAMPLES * N_BANDS_LOC].reshape(n, N_BANDS_LOC, N_SAMPLES)
        return a[:, 0, :], a[:, 1, :]

    def dist(lons, lats):
        dlon = ((lons - lon + 180.0) % 360.0) - 180.0
        return np.hypot(lats - lat, dlon * np.cos(np.radians(lat)))

    # ---- probe the strip to learn line <-> latitude ----------------------
    probes = np.unique(np.linspace(0, n_lines - 1, n_probe).astype(int))
    plines, plats = [], []
    for i in probes:
        lons, lats = read_lines(i, i + 1)
        if lons is None:
            continue
        mid = N_SAMPLES // 2
        if np.isfinite(lats[0, mid]):
            plines.append(i)
            plats.append(float(lats[0, mid]))
    if len(plines) < 3:
        raise RuntimeError("could not probe LOC file %s" % loc_url)

    plines = np.asarray(plines, dtype=float)
    plats = np.asarray(plats, dtype=float)
    order = np.argsort(plats)
    guess = int(np.clip(np.interp(lat, plats[order], plines[order]), 0, n_lines - 1))
    if verbose:
        print("      probed %d lines; latitude %.2f predicted at line %d"
              % (len(plines), lat, guess))

    # ---- read one contiguous block around the prediction and refine -------
    span = max(half_window * 4, 120)
    lo = max(0, guess - span)
    hi = min(n_lines, guess + span)
    lons, lats = read_lines(lo, hi)
    if lons is None:
        raise RuntimeError("could not read LOC block around line %d" % guess)
    d = dist(lons, lats)
    flat = int(np.nanargmin(d))
    line = lo + flat // N_SAMPLES
    sample = flat % N_SAMPLES
    dist_deg = float(np.nanmin(d))

    start = max(0, line - half_window)
    stop = min(n_lines, line + half_window + 1)
    return {"line": int(line), "sample": int(sample), "dist_deg": dist_deg,
            "line_start": int(start), "line_stop": int(stop),
            "n_probe_requests": int(len(plines)) + 1}


def fetch_rfl_block(rfl_url: str, line_start: int, line_stop: int) -> np.ndarray:
    """Download lines [line_start, line_stop) of an RFL file. Returns (L, B, S)."""
    raw = http_get(rfl_url, (line_start * RFL_RECORD, line_stop * RFL_RECORD - 1))
    a = np.frombuffer(raw, dtype="<f4")
    n = (line_stop - line_start)
    return a.reshape(n, N_BANDS_RFL, N_SAMPLES)


def scene_urls(cfg: dict) -> dict:
    return {
        "label": BASE + cfg["dir"] + cfg["product"] + "_L2.LBL",
        "rfl": BASE + cfg["dir"] + cfg["product"] + "_RFL.IMG",
        "loc": BASE + cfg["loc_dir"] + cfg["loc_product"] + "_LOC.IMG",
        "loc_label": BASE + cfg["loc_dir"] + cfg["loc_product"] + "_L1B.LBL",
        "obs": BASE + cfg["loc_dir"] + cfg["loc_product"] + "_OBS.IMG",
    }


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))
