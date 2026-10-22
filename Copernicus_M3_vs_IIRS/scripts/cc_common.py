"""
Shared paths, input guard, scene loading and zones for the Copernicus M3-vs-IIRS
comparison.

Inputs (read-only, never modified):
  M3   ../../STUDENT2_M3_INPUT/   M3G20090416T122951 L2 reflectance, EQC 140 m, 85 bands
  IIRS ../../iirs/                Chandrayaan-2 IIRS reflectance, EQC 140 m, 86 bands
Both cubes sit on the same 311 x 1192 grid (same projection, pixel size and bounds);
step01 checks this before anything else runs.

The paper pipelines are reused, not copied: Jung et al. (2026) from
../Jung2026_SpaceWeatheredMaria/scripts and Korokhin et al. (2024) from
../Korokhin2024_ShkuratovANN/scripts, each imported by path in its own step script.

Configurations (one rule per configuration, applied to every pixel):
  M3_full      M3 over each paper's own wavelength window
  M3_common    M3 cut to the range both sensors share, 0.85-2.50 um
  IIRS_common  IIRS as delivered, same 0.85-2.50 um range
  IIRS_scaled  IIRS divided by ONE scene-wide number (the median IIRS/M3 ratio over the
               shared pixels and common range): a sensitivity that separates "IIRS is
               brighter" from "IIRS has a different spectral shape". Never the primary.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMPL_ROOT = ROOT.parent
REPO_ROOT = IMPL_ROOT.parent
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
PRODUCTS = ROOT / "products"
MANIFEST = ROOT / "input_manifest.csv"

JUNG_SCRIPTS = IMPL_ROOT / "Jung2026_SpaceWeatheredMaria" / "scripts"
KOROKHIN_SCRIPTS = IMPL_ROOT / "Korokhin2024_ShkuratovANN" / "scripts"

M3_DIR = REPO_ROOT / "STUDENT2_M3_INPUT"
IIRS_DIR = REPO_ROOT / "iirs"
M3_CUBE = M3_DIR / "01_MASTER_DATA" / "M3G20090416T122951_V01_RFL_Copernicus_EQC_140m.tif"
M3_WL = M3_DIR / "02_WAVELENGTHS" / "m3_actual_wavelengths_nm.csv"
M3_MASK = M3_DIR / "03_VALID_MASK" / "M3G20090416T122951_Copernicus_valid_mask_B03_B73.tif"
IIRS_CUBE = IIRS_DIR / "IIRS_Copernicus_EQC_140m_86bands.tif"
IIRS_WL = IIRS_DIR / "IIRS_Copernicus_86band_wavelengths_nm.csv"
IIRS_MASK = IIRS_DIR / "IIRS_final_artifact_guarded_valid_mask.tif"
INPUTS = [M3_CUBE, M3_WL, M3_MASK, IIRS_CUBE, IIRS_WL, IIRS_MASK]

M3_ID = "M3G20090416T122951"
COMMON_LO_UM, COMMON_HI_UM = 0.85, 2.50
# The IIRS band list has two gaps between its detector segments (step01 prints them).
# "common" means inside 0.85-2.50 um AND outside both gaps, for BOTH sensors, so the two
# are judged on exactly the same wavelengths.
IIRS_GAPS_UM = [(1.1674, 1.2853), (1.8415, 1.9762)]
CONFIGS = ["M3_full", "M3_common", "IIRS_common", "IIRS_scaled"]

# Copernicus, IAU Gazetteer of Planetary Nomenclature: 9.62 N, 339.92 E, D = 96.07 km.
CENTER_LAT, CENTER_LON = 9.62, -20.08
CRATER_RADIUS_KM = 96.07 / 2.0
R_MOON_M = 1737400.0
# Zones by distance from the centre in crater radii. Pure geometry: they depend on
# neither sensor, so neither sensor's data decides which pixels are compared.
ZONES = [("central", 0.0, 0.25), ("floor", 0.25, 0.60), ("wall_rim", 0.60, 1.00),
         ("proximal_ejecta", 1.00, 2.00)]     # the grid ends inside 2 radii


def banner(stage: str, msg: str = "") -> None:
    print("\n" + "=" * 78 + "\n[%s] %s\n" % (stage, msg) + "=" * 78, flush=True)


# --------------------------------------------------------------------------
# input guard: SHA-256 of every input file, checked at the start and end of each step
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_rows() -> list[dict]:
    return [{"path": p.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_of(p),
             "bytes": p.stat().st_size} for p in INPUTS]


def write_manifest() -> pd.DataFrame:
    df = pd.DataFrame(_manifest_rows())
    df.to_csv(MANIFEST, index=False)
    return df


def verify_manifest(stage: str) -> None:
    if not MANIFEST.exists():
        sys.exit("[%s] FATAL: %s missing - run step01_inputs.py first." % (stage, MANIFEST))
    stored = pd.read_csv(MANIFEST)
    current = pd.DataFrame(_manifest_rows())
    if list(stored["path"]) != list(current["path"]):
        sys.exit("[%s] FATAL: input file list changed. Refusing to run." % stage)
    bad = stored["sha256"].to_numpy() != current["sha256"].to_numpy()
    if bad.any():
        sys.exit("[%s] FATAL: input checksum mismatch: %s. Refusing to run."
                 % (stage, ", ".join(current.loc[bad, "path"])))


# --------------------------------------------------------------------------
# scene loading
# --------------------------------------------------------------------------
def wavelengths_um(sensor: str) -> np.ndarray:
    f = M3_WL if sensor == "M3" else IIRS_WL
    return pd.read_csv(f)["wavelength_nm"].to_numpy(dtype=float) * 1e-3


def read_cube(sensor: str) -> np.ndarray:
    """(bands, rows, cols) float64, NoData and non-finite values as NaN."""
    f = M3_CUBE if sensor == "M3" else IIRS_CUBE
    with rasterio.open(f) as d:
        a = d.read().astype(np.float64)
        nd = d.nodata
    if nd is not None:
        a[a == nd] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a


def read_mask(sensor: str) -> np.ndarray:
    f = M3_MASK if sensor == "M3" else IIRS_MASK
    with rasterio.open(f) as d:
        return d.read(1) == 1


def grid_info() -> dict:
    with rasterio.open(M3_CUBE) as d:
        return {"transform": d.transform, "crs": d.crs, "shape": (d.height, d.width)}


def joint_mask() -> np.ndarray:
    """Pixels valid in BOTH delivered masks and physically plausible (finite, > 0) in
    both cubes over every band inside 0.55-2.50 um (M3) and 0.85-2.50 um (IIRS).
    The same pixels feed every configuration of both papers."""
    m = read_mask("M3") & read_mask("IIRS")
    for sensor, lo in (("M3", 0.55), ("IIRS", COMMON_LO_UM)):
        wl = wavelengths_um(sensor)
        sel = (wl >= lo) & (wl <= COMMON_HI_UM)
        c = read_cube(sensor)[sel]
        m &= np.isfinite(c).all(axis=0) & (c > 0).all(axis=0)
    return m


def latlon() -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre latitude/longitude (deg) of the equirectangular grid
    (standard parallel 0, central meridian 0, sphere R = 1737.4 km)."""
    g = grid_info()
    rows, cols = np.mgrid[0:g["shape"][0], 0:g["shape"][1]]
    t = g["transform"]
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    return np.degrees(y / R_MOON_M), np.degrees(x / R_MOON_M)


def radial_distance_radii() -> np.ndarray:
    """Great-circle distance from the crater centre, in crater radii."""
    lat, lon = latlon()
    p1, p2 = np.radians(CENTER_LAT), np.radians(lat)
    dl = np.radians(lon - CENTER_LON)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    km = 2 * R_MOON_M / 1e3 * np.arcsin(np.sqrt(a))
    return km / CRATER_RADIUS_KM


def zone_map() -> np.ndarray:
    """Integer zone index per pixel (index into ZONES)."""
    r = radial_distance_radii()
    z = np.full(r.shape, -1, dtype=np.int16)
    for i, (_, lo, hi) in enumerate(ZONES):
        z[(r >= lo) & (r < hi)] = i
    return z


def iirs_scale_factor() -> float:
    """The ONE number used by IIRS_scaled, written by step01."""
    return float(pd.read_csv(TABLES / "step01_iirs_scale_factor.csv")["scale"].iloc[0])


def in_common(wl_um: np.ndarray) -> np.ndarray:
    """True where a wavelength is inside the common range and outside both IIRS gaps."""
    wl_um = np.asarray(wl_um, dtype=float)
    ok = (wl_um >= COMMON_LO_UM) & (wl_um <= COMMON_HI_UM)
    for lo, hi in IIRS_GAPS_UM:
        ok &= ~((wl_um > lo) & (wl_um < hi))
    return ok


def config_cube(config: str) -> tuple[np.ndarray, np.ndarray]:
    """(wavelengths_um, cube (bands, rows, cols)) for a configuration, cut to its range.
    M3_full keeps every band; the paper steps apply their own windows on top."""
    if config == "M3_full":
        return wavelengths_um("M3"), read_cube("M3")
    sensor = "M3" if config == "M3_common" else "IIRS"
    wl, c = wavelengths_um(sensor), read_cube(sensor)
    sel = in_common(wl)
    wl, c = wl[sel], c[sel]
    if config == "IIRS_scaled":
        c = c / iirs_scale_factor()
    return wl, c


def save_map(path: Path, arr: np.ndarray, nodata: float = np.nan) -> None:
    """Single-band float32 GeoTIFF on the shared grid; rejected pixels are real NoData."""
    g = grid_info()
    a = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                       count=1, dtype="float32", crs=g["crs"], transform=g["transform"],
                       nodata=nodata, compress="deflate") as d:
        d.write(a, 1)


def r2_bias_rmse(x: np.ndarray, y: np.ndarray) -> dict:
    """Agreement of y (test) with x (reference): Pearson r, R2 of y = x, bias, RMSE."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return {"n": int(x.size), "pearson_r": np.nan, "r2_identity": np.nan,
                "bias": np.nan, "rmse": np.nan}
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    return {"n": int(x.size), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "r2_identity": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "bias": float(np.mean(y - x)), "rmse": float(np.sqrt(np.mean((y - x) ** 2)))}
