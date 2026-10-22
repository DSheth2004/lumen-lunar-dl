"""
Step 00b -- stage M3 L2 spectra at the returned-sample landing sites (METHOD_NOTES.md
Addendum A, D20-D21). Run once with network access to planetarydata.jpl.nasa.gov; it
writes data/m3_sites/ and regenerates the SHA-256 manifest. Every later stage reads only
the staged copy.

  1. global-mode L2 strips from the archive's own index (CH1M3_0004/INDEX/L2_INDEX.TAB);
  2. footprints: 5 lines of every strip's L1B LOC file (one 7296-byte range request each);
  3. per site (distinct coordinate of FD's 50-sample ground truth): candidate strips from
     the footprints, the site line found by Newton steps on latitude, the nearest pixel
     from a +-25-line LOC block; a strip is kept if that pixel is within 0.01 deg and at
     least 5 samples from either swath edge;
  4. the 3 x 3-pixel RFL block around it (3 lines x 103360 bytes).

M3 L2 RFL records are one image line each (304 samples x 85 bands x 4 bytes, line
interleaved), so only the needed lines are downloaded.
"""
from __future__ import annotations

import csv
import io
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import lumen_common as lc

STAGE = "step00b"
BASE = "https://planetarydata.jpl.nasa.gov/img/data/m3/"
INDEX_FILES = ["CH1M3_0004/INDEX/L2_INDEX.TAB", "CH1M3_0004/INDEX/L2_INDEX.LBL"]
OUT = lc.DATA / "m3_sites"
STAGING = lc.PRODUCTS / "m3_sites_staging"          # written here first, moved into data/ at the end
CACHE = lc.PRODUCTS / "m3_footprint_cache.npz"
NS, NB = 304, 85
RFL_REC = NS * NB * 4
LOC_REC = NS * 3 * 8
MAX_DIST_DEG = 0.01
EDGE = 5
N_PROBE = 5


_S = None


def session():
    global _S
    if _S is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        s = requests.Session()
        s.headers.update({"User-Agent": "lumen-lunar-dl/1.0 (research; range requests)"})
        retry = Retry(total=6, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"])
        ad = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        s.mount("https://", ad)
        _S = s
    return _S


def get(url, rng=None):
    hdr = {"Range": "bytes=%d-%d" % rng} if rng else {}
    last = None
    for attempt in range(6):
        try:
            r = session().get(url, headers=hdr, timeout=120)
            r.raise_for_status()
            if rng and r.status_code != 206:
                raise RuntimeError("server ignored the range request for %s" % url)
            return r.content
        except Exception as e:                      # noqa: BLE001 - retried, then raised
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("GET failed: %s (%s)" % (url, last))


def loc_lines(url, i0, i1):
    raw = get(url, (i0 * LOC_REC, i1 * LOC_REC - 1))
    a = np.frombuffer(raw, "<f8").reshape(-1, 3, NS)
    return a[:, 0, :], a[:, 1, :]                    # lon, lat (lines, samples)


# --------------------------------------------------------------------------
def global_products(index_text):
    out = []
    for r in csv.reader(io.StringIO(index_text), skipinitialspace=True):
        if r[11].strip() != "GLOBAL" or int(r[14]) < 50:
            continue
        spec = r[4].strip()                          # DATA/.../L2/<id>_L2.LBL
        d = spec.rsplit("/", 1)[0]
        out.append({"product": r[2].strip().replace("_RFL", ""), "n_lines": int(r[14]),
                    "rfl_url": BASE + "CH1M3_0004/" + spec.replace("_L2.LBL", "_RFL.IMG"),
                    "loc_url": BASE + "CH1M3_0003/" + d.replace("/L2", "/L1B") + "/" + r[23].strip()})
    return pd.DataFrame(out)


def footprints(prod):
    """Probe N_PROBE lines per strip; cached so a re-run does not re-download."""
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=False)
        if list(c["product"]) == list(prod["product"]):
            return c["lines"], c["lon"], c["lat"]
    lines = np.stack([np.unique(np.linspace(0, n - 1, N_PROBE).astype(int)) if n >= N_PROBE
                      else np.arange(N_PROBE) % n for n in prod["n_lines"]])
    lon = np.full((len(prod), N_PROBE, NS), np.nan)
    lat = np.full((len(prod), N_PROBE, NS), np.nan)

    def one(i):
        url = prod["loc_url"].iloc[i]
        for j, ln in enumerate(lines[i]):
            lo, la = loc_lines(url, int(ln), int(ln) + 1)
            lon[i, j], lat[i, j] = lo[0], la[0]
        return i
    t = time.time()
    with ThreadPoolExecutor(6) as ex:
        for k, _ in enumerate(ex.map(one, range(len(prod)))):
            if (k + 1) % 100 == 0:
                print("    footprints %d / %d (%.0f s)" % (k + 1, len(prod), time.time() - t), flush=True)
    np.savez(CACHE, product=prod["product"].to_numpy(), lines=lines, lon=lon, lat=lat)
    return lines, lon, lat


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def candidates(site_lat, site_lon, lines, lon, lat):
    """(strip index, guessed line) for strips whose probed track brackets the site."""
    out = []
    mid = NS // 2
    for i in range(len(lines)):
        cl = lat[i, :, mid]
        for j in range(len(cl) - 1):
            a, b = cl[j], cl[j + 1]
            if not (np.isfinite(a) and np.isfinite(b)) or not (min(a, b) - 0.5 <= site_lat <= max(a, b) + 0.5):
                continue
            f = 0.5 if b == a else np.clip((site_lat - a) / (b - a), 0, 1)
            g = lines[i, j] + f * (lines[i, j + 1] - lines[i, j])
            lon_e = [wrap(lon[i, j, s] + f * wrap(lon[i, j + 1, s] - lon[i, j, s]) - site_lon) for s in (EDGE, NS - 1 - EDGE)]
            lon_c = wrap(lon[i, j, mid] + f * wrap(lon[i, j + 1, mid] - lon[i, j, mid]) - site_lon)
            half = max(abs(lon_e[0] - lon_c), abs(lon_e[1] - lon_c))
            if np.isfinite(lon_c) and abs(lon_c) <= half + 0.5:
                out.append((i, int(round(g)), (int(lines[i, j]), int(lines[i, j + 1]))))
    return out


def locate(url, n_lines, guess, seg, site_lat, site_lon):
    """Newton steps on the centre-sample latitude, then the nearest pixel in a block."""
    g = int(np.clip(guess, 0, n_lines - 1))
    lo_l, hi_l = seg
    for _ in range(4):
        _, la = loc_lines(url, g, g + 1)
        _, la2 = loc_lines(url, min(g + 50, n_lines - 1), min(g + 50, n_lines - 1) + 1)
        slope = (la2[0, NS // 2] - la[0, NS // 2]) / max(1, min(g + 50, n_lines - 1) - g)
        if not np.isfinite(slope) or abs(slope) < 1e-9:
            break
        step = (site_lat - la[0, NS // 2]) / slope
        g2 = int(np.clip(g + step, max(0, lo_l - 200), min(n_lines - 1, hi_l + 200)))
        if abs(g2 - g) < 3:
            g = g2
            break
        g = g2
    i0, i1 = max(0, g - 25), min(n_lines, g + 26)
    lon, lat = loc_lines(url, i0, i1)
    d = np.hypot(lat - site_lat, wrap(lon - site_lon) * np.cos(np.radians(site_lat)))
    k = int(np.nanargmin(d))
    return i0 + k // NS, k % NS, float(np.nanmin(d))


def sites_table():
    gt = pd.read_csv(lc.GT_DIR / "fernandez2026_ground_truth_50.csv")
    gt["key"] = list(zip(gt["latitude"].round(4), gt["longitude"].round(4)))
    rows = []
    short = {"Apollo 11": "A11", "Apollo 12": "A12", "Apollo 14": "A14", "Apollo 15": "A15",
             "Apollo 16": "A16", "Apollo 17": "A17", "Luna 16": "L16", "Luna 20": "L20",
             "Luna 24": "L24", "Chang'E-3": "CE3", "Chang'E-5": "CE5", "Chang'E-6": "CE6"}
    count = {}
    for key, g in gt.groupby("key", sort=False):
        m = g["mission"].iloc[0]
        count[m] = count.get(m, 0) + 1
        sid = short[m] + ("" if count[m] == 1 else "_%d" % count[m])
        rows.append({"site_id": sid, "mission": m, "lat": key[0], "lon": key[1], "n_samples": len(g),
                     "feo_mean": g["feo_wt_pct"].mean(), "feo_sd": g["feo_wt_pct"].std(ddof=0),
                     "feo_range": g["feo_wt_pct"].max() - g["feo_wt_pct"].min(),
                     "samples": ";".join(g["sample_id"]), "sources": ";".join(sorted(set(g["source"])))})
    return pd.DataFrame(rows)


def main() -> int:
    t0 = lc.banner(STAGE, "stage M3 L2 spectra at the landing sites (Addendum A)")
    lc.verify_manifest(STAGE)
    (STAGING / "index").mkdir(parents=True, exist_ok=True)
    for f in INDEX_FILES:
        p = STAGING / "index" / Path(f).name
        if not p.exists():
            p.write_bytes(get(BASE + f))
    prod = global_products((STAGING / "index" / "L2_INDEX.TAB").read_text())
    print("  global-mode L2 strips: %d" % len(prod))
    lines, lon, lat = footprints(prod)
    print("  footprints: %d strips probed" % len(prod))

    sites = sites_table()
    rows, blocks = [], []
    for _, s in sites.iterrows():
        cands = candidates(s["lat"], s["lon"], lines, lon, lat)
        kept = 0
        for i, guess, seg in cands:
            p = prod.iloc[i]
            try:
                ln, smp, dist = locate(p["loc_url"], p["n_lines"], guess, seg, s["lat"], s["lon"])
            except RuntimeError as e:
                print("    %s %s: LOC read failed (%s)" % (s["site_id"], p["product"], e))
                continue
            ok = dist <= MAX_DIST_DEG and EDGE <= smp <= NS - 1 - EDGE and 1 <= ln <= p["n_lines"] - 2
            rec = {"site_id": s["site_id"], "product": p["product"], "line": ln, "sample": smp,
                   "dist_deg": dist, "kept": ok, "rfl_url": p["rfl_url"], "loc_url": p["loc_url"]}
            if ok:
                raw = get(p["rfl_url"], ((ln - 1) * RFL_REC, (ln + 2) * RFL_REC - 1))
                blk = np.frombuffer(raw, "<f4").reshape(3, NB, NS)[:, :, smp - 1:smp + 2].copy()
                rec["block"] = len(blocks)
                blocks.append(blk)
                kept += 1
            rows.append(rec)
        print("  %-8s (%7.3f, %8.3f) FeO %.1f: %d candidate strips, %d within %.2f deg"
              % (s["site_id"], s["lat"], s["lon"], s["feo_mean"], len(cands), kept, MAX_DIST_DEG), flush=True)
    strips = pd.DataFrame(rows)
    sites.to_csv(STAGING / "sites.csv", index=False)
    strips.to_csv(STAGING / "site_strips.csv", index=False)
    np.save(STAGING / "site_strip_blocks.npy", np.asarray(blocks, np.float32))     # (k, 3 lines, 85, 3 samples)
    pd.DataFrame({"product": prod["product"], "n_lines": prod["n_lines"],
                  **{"probe_line_%d" % j: lines[:, j] for j in range(N_PROBE)},
                  **{"centre_lat_%d" % j: lat[:, j, NS // 2] for j in range(N_PROBE)},
                  **{"centre_lon_%d" % j: lon[:, j, NS // 2] for j in range(N_PROBE)}}).to_csv(
        STAGING / "footprints.csv", index=False)
    (STAGING / "provenance.md").write_text(
        "M3 L2 reflectance (CH1M3_0004) and L1B LOC (CH1M3_0003) from %s, fetched by "
        "scripts/step00b_fetch_m3_sites.py on %s with HTTP range requests (only the lines used). "
        "Sites: distinct coordinates of data/ground_truth/fernandez2026_ground_truth_50.csv. "
        "site_strip_blocks.npy[k] is the 3x3-pixel block (lines, 85 bands, samples) of row "
        "`block` = k of site_strips.csv, unscaled L2 reflectance.\n"
        % (BASE, time.strftime("%Y-%m-%d", time.gmtime())))
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(STAGING, OUT)
    df = lc.write_manifest()
    print("  manifest regenerated: %d files, %.1f MB" % (len(df), df["bytes"].sum() / 1e6))
    lc.done(STAGE, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
