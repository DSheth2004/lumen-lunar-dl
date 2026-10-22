"""
M3 L2 access by HTTP range request -- never download a whole strip.

Copied (with small generalisations) from
  FernandezDiaz2026_WaveletAE_FeO/scripts/m3_sites.py   (ODE search, LOC probing)
  Jung2026_SpaceWeatheredMaria/scripts/m3_fetch.py      (block fetch)
so this directory stays self-contained.

Facts it relies on (checked against the PDS archive):
  * an L2 RFL record is exactly one image line: samples x bands x float32,
    band-interleaved-by-line, so any run of lines is one range request;
  * geolocation lives in the L1B LOC backplane (lon, lat, radius; float64, BIL);
  * ODE footprint boxes are unreliable -- many products claim 0-360 deg of
    longitude -- so candidates are filtered on their own extents and then
    confirmed against LOC.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

import numpy as np

PDS = "https://planetarydata.jpl.nasa.gov/img/data/m3/"
ODE = "https://oderest.rsl.wustl.edu/live2/?"
MAX_LON_SPAN = 20.0        # a real M3 strip is a few degrees wide near the equator


def _get(url: str, byte_range=None, timeout=120, tries=4) -> bytes:
    last = None
    for k in range(tries):
        try:
            h = {"User-Agent": "Mozilla/5.0"}
            if byte_range is not None:
                h["Range"] = "bytes=%d-%d" % byte_range
            req = urllib.request.Request(url, headers=h)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as exc:          # network hiccups: back off and retry
            last = exc
            time.sleep(2.0 * (k + 1))
    raise RuntimeError("GET failed: %s (%s)" % (url, last))


def ode_candidates(lat: float, lon: float, half=0.1) -> list[dict]:
    """L2 products whose own lat/lon extents contain the point (all pages)."""
    out, off = [], 0
    while True:
        q = {"target": "moon", "query": "product", "results": "fmpc", "output": "JSON",
             "ihid": "CH1-ORB", "iid": "M3", "pt": "REFIMG",
             "minlatitude": lat - half, "maxlatitude": lat + half,
             "westernlongitude": lon - half, "easternlongitude": lon + half,
             "limit": 500, "offset": off}
        r = json.loads(_get(ODE + urllib.parse.urlencode(q), timeout=180).decode())
        prods = r.get("ODEResults", {}).get("Products", {}).get("Product", [])
        prods = [prods] if isinstance(prods, dict) else prods
        for p in prods:
            try:
                mn, mx = float(p["Minimum_latitude"]), float(p["Maximum_latitude"])
                we, ea = float(p["Westernmost_longitude"]), float(p["Easternmost_longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            span = ea - we
            if mn <= lat <= mx and we <= lon <= ea and 0 < span < MAX_LON_SPAN:
                out.append({"pdsid": p["pdsid"], "label_url": p.get("LabelURL", ""),
                            "min_lat": mn, "max_lat": mx, "west_lon": we,
                            "east_lon": ea, "lon_span": span})
        if len(prods) < 500:
            break
        off += 500
    return sorted(out, key=lambda d: d["pdsid"])


def paths_from_label_url(label_url: str) -> dict:
    """L2 RFL/HDR/label and the matching L1B LOC/label URLs from an ODE label URL."""
    m = re.search(r"(CH1M3_\d+)/DATA/([0-9_]+)/(\d{6})/L2/(M3G\d{8}T\d{6})_V\d+_L2\.LBL",
                  label_url)
    if not m:
        raise ValueError("unrecognised L2 label URL: %r" % label_url)
    _vol, rng, ym, base = m.groups()
    l1b = PDS + "CH1M3_0003/DATA/%s/%s/L1B/%s_V03" % (rng, ym, base)
    l2 = PDS + "CH1M3_0004/DATA/%s/%s/L2/%s_V01" % (rng, ym, base)
    return {"base": base, "rfl": l2 + "_RFL.IMG", "rfl_hdr": l2 + "_RFL.HDR",
            "l2_lbl": l2 + "_L2.LBL", "loc": l1b + "_LOC.IMG", "l1b_lbl": l1b + "_L1B.LBL"}


def read_label(url: str) -> dict:
    """LINES / LINE_SAMPLES / BANDS of the first image object in a PDS3 label."""
    txt = _get(url, timeout=90).decode("latin-1")
    out = {}
    for k in ("LINES", "LINE_SAMPLES", "BANDS"):
        m = re.search(r"^\s*%s\s*=\s*(\d+)" % k, txt, re.M)
        if m:
            out[k] = int(m.group(1))
    return out


def read_hdr_wavelengths(url: str) -> np.ndarray:
    """ENVI header 'wavelength = { ... }' in nm."""
    txt = _get(url, timeout=90).decode("latin-1")
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, re.I | re.S)
    return np.array([float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()])


def read_loc_lines(loc_url: str, n_samples: int, i0: int, i1: int):
    """(lon, lat) arrays for lines [i0, i1) of an L1B LOC backplane."""
    rec = n_samples * 3 * 8
    raw = _get(loc_url, (i0 * rec, i1 * rec - 1))
    a = np.frombuffer(raw, dtype="<f8")
    n = a.size // (n_samples * 3)
    a = a[:n * n_samples * 3].reshape(n, 3, n_samples)
    return a[:, 0, :], a[:, 1, :]


def locate_pixel(loc_url: str, n_lines: int, n_samples: int, lat: float, lon: float,
                 n_probe: int = 40, half: int = 400) -> dict:
    """Nearest (line, sample) to (lat, lon) from a few probe lines + one block."""
    probes = np.unique(np.linspace(0, n_lines - 1, n_probe).astype(int))
    pl, pa = [], []
    mid = n_samples // 2
    for i in probes:
        lo_, la_ = read_loc_lines(loc_url, n_samples, i, i + 1)
        if la_.size and np.isfinite(la_[0, mid]) and abs(la_[0, mid]) <= 90:
            pl.append(i)
            pa.append(float(la_[0, mid]))
    pl, pa = np.asarray(pl, float), np.asarray(pa, float)
    order = np.argsort(pa)
    guess = int(np.clip(np.interp(lat, pa[order], pl[order]), 0, n_lines - 1))
    lo, hi = max(0, guess - half), min(n_lines, guess + half)
    lons, lats = read_loc_lines(loc_url, n_samples, lo, hi)
    dlon = ((lons - lon + 180.0) % 360.0) - 180.0
    d = np.hypot(lats - lat, dlon * np.cos(np.radians(lat)))
    k = int(np.nanargmin(d))
    return {"line": lo + k // n_samples, "sample": k % n_samples, "dist_deg": float(np.nanmin(d)),
            "lat_increases_with_line": bool(pa[-1] > pa[0]) if len(pa) > 1 else None}


def fetch_rfl_lines(rfl_url: str, n_samples: int, n_bands: int, i0: int, i1: int) -> np.ndarray:
    """Lines [i0, i1) of an L2 RFL image as (lines, bands, samples) float32."""
    rec = n_samples * n_bands * 4
    raw = _get(rfl_url, (i0 * rec, i1 * rec - 1), timeout=600)
    return np.frombuffer(raw, dtype="<f4").reshape(-1, n_bands, n_samples).copy()
