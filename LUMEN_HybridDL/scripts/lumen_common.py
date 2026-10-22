"""
Shared definitions for LUMEN -- a hybrid physics + deep-learning framework for lunar
mineralogy, chemistry and maturity from visible/near-infrared reflectance spectra.

LUMEN = LUnar Mineralogy Estimation Network.

This module holds paths, the read-only input guard, RELAB / LSCC I/O, the M3 band grid
and the target definitions that every stage is driven from. Nothing here writes into any
source folder: inputs are copied into data/ by step00 and checksummed there.

Conventions carried over from the rest of this repository (load-bearing, not stylistic):
  * data/ carries a SHA-256 manifest, re-verified at the START and END of every stage;
  * band/wavelength correspondence is positional and checked, never sorted silently;
  * every stage takes an explicit seed;
  * RESULTS.md is written by code, never by hand.
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPL_ROOT = PROJECT_ROOT.parent                      # paper-implementations/
REPO_ROOT = IMPL_ROOT.parent                         # RMS/
DATA = PROJECT_ROOT / "data"
TABLES = PROJECT_ROOT / "tables"
FIGURES = PROJECT_ROOT / "figures"
PRODUCTS = PROJECT_ROOT / "products"
MANIFEST = DATA / "manifest.csv"

LSCC_DIR = DATA / "lscc"
GT_DIR = DATA / "ground_truth"
RELAB_DIR = DATA / "relab"
OPTICAL_DIR = DATA / "optical_constants"
M3_SHAPLEY_DIR = DATA / "m3_shapley"
M3_ARISTARCHUS_DIR = DATA / "m3_aristarchus"
BASELINE_DIR = DATA / "baselines"

# read-only sources outside this directory (only step00 reads them)
SRC_KOROKHIN = IMPL_ROOT / "Korokhin2024_ShkuratovANN"
SRC_JUNG = IMPL_ROOT / "Jung2026_SpaceWeatheredMaria"
SRC_FERNANDEZ = IMPL_ROOT / "FernandezDiaz2026_WaveletAE_FeO"
SRC_RELAB_CART = REPO_ROOT / "RELAB_LUNAR_RAW" / "cartorder"

for _d in (TABLES, FIGURES, PRODUCTS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# measurement geometry: every RELAB spectrum used here is i = 30 deg, e = 0 deg
# (checked from the PDS4 labels in step00), and M3 L2 reflectance is photometrically
# normalised to the same geometry.
# --------------------------------------------------------------------------
INCIDENCE_DEG = 30.0
EMISSION_DEG = 0.0
MU0 = float(np.cos(np.radians(INCIDENCE_DEG)))
MU = float(np.cos(np.radians(EMISSION_DEG)))

# --------------------------------------------------------------------------
# M3 band grid. The 85 global-mode band centres are read from the staged ENVI header
# list (data/m3_shapley/m3_wavelengths_nm.csv). The model input is bands 3..73
# (1-based, 540.84-2497.11 nm, 71 bands): bands 1-2 are poorly calibrated and beyond
# 2.5 um the L2 thermal correction dominates -- the same window the repository's other
# M3 work uses. Global mode bins the 12.3-nm target channels 2x below 1.55 um and 4x
# above, so the spectral response is taken as a Gaussian of FWHM = local band spacing.
# --------------------------------------------------------------------------
M3_FIRST_BAND = 3
M3_LAST_BAND = 73


def m3_wavelengths_nm() -> np.ndarray:
    df = pd.read_csv(M3_SHAPLEY_DIR / "m3_wavelengths_nm.csv")
    assert list(df["band"]) == list(range(1, 86)), "M3 band numbering is not 1..85 in order"
    return df["wavelength_nm"].to_numpy(float)


def input_band_index() -> np.ndarray:
    """0-based indices of the model's input bands within the 85-band M3 cube."""
    return np.arange(M3_FIRST_BAND - 1, M3_LAST_BAND)


def input_wavelengths_um() -> np.ndarray:
    return m3_wavelengths_nm()[input_band_index()] / 1000.0


def m3_fwhm_um(centres_um: np.ndarray) -> np.ndarray:
    """FWHM of each band = local centre spacing (global-mode binning)."""
    c = np.asarray(centres_um, float)
    sp = np.gradient(c)
    return np.abs(sp)


def srf_resample(lam_src_um, y_src, centres_um, fwhm_um=None) -> np.ndarray:
    """Convolve a finely sampled spectrum with Gaussian band responses.

    Returns NaN for any band whose +-1.5 FWHM support is not covered by the source.
    Works on a single spectrum (1-D y) or a stack (2-D, spectra along axis 0).
    """
    lam_src_um = np.asarray(lam_src_um, float)
    y = np.asarray(y_src, float)
    one = y.ndim == 1
    y = np.atleast_2d(y)
    centres_um = np.asarray(centres_um, float)
    if fwhm_um is None:
        fwhm_um = m3_fwhm_um(centres_um)
    sig = np.asarray(fwhm_um, float) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    W = np.exp(-0.5 * ((centres_um[:, None] - lam_src_um[None, :]) / sig[:, None]) ** 2)
    W = W / W.sum(axis=1, keepdims=True)
    out = y @ W.T
    covered = (centres_um - 1.5 * np.asarray(fwhm_um) >= lam_src_um.min() - 1e-9) & \
              (centres_um + 1.5 * np.asarray(fwhm_um) <= lam_src_um.max() + 1e-9)
    out[:, ~covered] = np.nan
    return out[0] if one else out


# --------------------------------------------------------------------------
# LSCC soils (Taylor et al. 2001 mare; Taylor et al. 2010 highland)
# --------------------------------------------------------------------------
MARE_SOILS = ["10084", "12001", "12030", "15041", "15071", "70181", "71061", "71501", "79221"]
HIGHLAND_SOILS = ["14141", "14163", "14259", "14260", "61141", "61221", "62231", "64801",
                  "67461", "67481"]
ALL_SOILS = MARE_SOILS + HIGHLAND_SOILS
SOIL_SET = {s: "mare" for s in MARE_SOILS} | {s: "highland" for s in HIGHLAND_SOILS}
MISSION = {s: "A" + s[:2] if s[:2] in ("10", "12", "14", "15") else
           ("A16" if s[0] == "6" else "A17") for s in ALL_SOILS}
MISSION["10084"] = "A11"
FRACTIONS = ["<10", "10-20", "20-45", "<45"]
MODAL_FRACTIONS = ["<10", "10-20", "20-45"]          # modal mineralogy is measured on these
FRACTION_NOMINAL_D_UM = {"<10": 5.0, "10-20": 15.0, "20-45": 32.5, "<45": 17.0}

# --------------------------------------------------------------------------
# targets
#   COMP  closed 7-part modal composition (volume %, LSCC point/area counts):
#         PLG plagioclase; LCP low-Ca pyroxene (orthopyroxene + pigeonite);
#         HCP high-Ca pyroxene (Mg- + Fe-clinopyroxene); OLV olivine; ILM ilmenite;
#         AGG agglutinitic glass; OTH volcanic glass + "others" (+ K-phases, highland)
#   CHEM  bulk oxides, wt %
#   MAT   log10(Is/FeO), ferromagnetic-resonance maturity
# --------------------------------------------------------------------------
COMP = ["PLG", "LCP", "HCP", "OLV", "ILM", "AGG", "OTH"]
MINERALS5 = ["HCP", "LCP", "OLV", "PLG", "ILM"]          # Korokhin's 5-mineral subtotal
CHEM = ["FeO", "TiO2", "Al2O3", "CaO", "MgO"]
MAT = "logIsFeO"
SCALARS = CHEM + [MAT]

# --------------------------------------------------------------------------
# End-member library for the physics simulator.
#
# Rule (D1, METHOD_NOTES.md): no material derived from any of the 19 evaluation soils
# may enter the simulator, otherwise synthetic pretraining would leak information
# about a held-out soil. Every spectrum below is a mineral separate from a ROCK
# (LRMCC basalts 15058, 15555, 70017, 70035, 12063; highland rocks 15415, 76535,
# 78235, 72415, 60015/60025, 65315, 69955), from soil 62241 (not an LSCC soil), a
# volcanic glass, or a synthetic lunar glass. All RELAB agglutinate separates come
# from LSCC soils (15041, 62231, 79221, 14259, 61141, 67461) and are therefore NOT
# used; agglutinates are modelled physically instead (glass host + npFe0 + mpFe0).
#
# columns: spectrum id, class, text the PDS4 specimen name must contain, nominal
#          effective grain size of the separate (um) used to derive k(lambda)
# --------------------------------------------------------------------------
ENDMEMBER_LIBRARY = pd.DataFrame(
    [
        # plagioclase
        ("c1lr171", "PLG", "15555,965 plagioclase", 45.0),
        ("c1lr174", "PLG", "15058,276 plagioclase", 45.0),
        ("c1lr177", "PLG", "70017,535 plagioclase", 45.0),
        ("c1lr181", "PLG", "70035,188 plagioclase", 45.0),
        ("c1lr183", "PLG", "62241,21 plagioclase", 45.0),
        ("c1lr224", "PLG", "15415,60 plagioclase", 45.0),
        ("c1lr226", "PLG", "76535,93 plagioclase", 45.0),
        ("c1lr229", "PLG", "60015 plagioclase", 110.0),
        ("c1lr240c", "PLG", "60025,885", 60.0),
        ("c1lr241c", "PLG", "65315,153", 60.0),
        ("c1lr242c", "PLG", "60015,801", 60.0),
        ("c1lr243c", "PLG", "69955,65", 60.0),
        ("c1lr210", "PLG", "15058,276 plagioclase D", 20.0),
        ("c1lr214", "PLG", "15555,965 plagioclase D", 20.0),
        ("c1lr217", "PLG", "70017,535 plagioclase D", 20.0),
        ("c1lr221", "PLG", "70035,188 plagioclase D", 20.0),
        ("c1lr223", "PLG", "62241,21 plagioclase D", 20.0),
        # low-Ca pyroxene (orthopyroxene / pigeonite)
        ("c1lr173", "LCP", "15058,276 green pyroxene", 45.0),
        ("c1lr176", "LCP", "70017,535 light-brown pyroxene", 45.0),
        ("c1lr180", "LCP", "70035,188 light-brown pyroxene", 45.0),
        ("c1ls12", "LCP", "78235,9002A", 80.0),
        ("c1lr209", "LCP", "15058,276 green pyroxene D", 20.0),
        ("c1lr216", "LCP", "70017,535 light-brown pyroxene D", 20.0),
        ("c1lr220", "LCP", "70035,188 light-brown pyroxene D", 20.0),
        # high-Ca pyroxene (augite)
        ("c1lr168", "HCP", "15555,965 reddish-brown pyroxene", 45.0),
        ("c1lr170", "HCP", "15555,965 light-brown pyroxene", 45.0),
        ("c1lr172", "HCP", "15058,276 brown pyroxene", 45.0),
        ("c1lr175", "HCP", "70017,535 deep-brown pyroxene", 45.0),
        ("c1lr179", "HCP", "70035,188 deep-brown pyroxene", 45.0),
        ("c1ls09", "HCP", "12063,199", 80.0),
        ("c1lr208", "HCP", "15058,276 brown pyroxene D", 20.0),
        ("c1lr211", "HCP", "15555,965 reddish-brown pyroxene D", 20.0),
        ("c1lr213", "HCP", "15555,965 light-brown pyroxene D", 20.0),
        ("c1lr215", "HCP", "70017,535 deep-brown pyroxene D", 20.0),
        ("c1lr219", "HCP", "70035,188 dark-brown pyroxene D", 20.0),
        # olivine
        ("c1lr169", "OLV", "15555,965 olivine", 45.0),
        ("c1lr225", "OLV", "72415,64 olivine", 45.0),
        ("c1lr227", "OLV", "76535,94 olivine", 45.0),
        ("c1ls05", "OLV", "72415,64b", 80.0),
        ("c1lr212", "OLV", "15555,965 olivine D", 20.0),
        ("c1lr14", "OLV", "72415,64c", 20.0),
        # ilmenite
        ("c1lr178", "ILM", "70017,535 ilmenite", 45.0),
        ("c1lr182", "ILM", "70035,188 ilmenite", 45.0),
        ("c1lr218", "ILM", "70017,535 ilmenite D", 20.0),
        ("c1lr222", "ILM", "70035,188 ilmenite D", 20.0),
        # glasses: volcanic (orange 74220, green 15401) and synthetic lunar glasses used as
        # agglutinate hosts (SC-EAC-050 plagioclase-composition, SC-EAC-053 pyroxene-composition)
        ("c1lr51", "VGL", "74220", 60.0),
        ("c1lr52", "VGL", "15401", 60.0),
        ("c1sc50", "HGL", "L19", 50.0),
        ("c1sc53", "HGL", "L22", 50.0),
    ],
    columns=["spectrum_id", "em_class", "specimen_must_contain", "d_eff_um"],
)

# Real refractive index per class (Hapke slab model) and solid density (g/cm3).
EM_N = {"PLG": 1.56, "LCP": 1.73, "HCP": 1.73, "OLV": 1.75, "ILM": 2.13, "VGL": 1.64,
        "HGL": 1.64}
EM_RHO = {"PLG": 2.73, "LCP": 3.40, "HCP": 3.35, "OLV": 3.60, "ILM": 4.75, "AGG": 3.00,
          "OTH": 3.10}

# --------------------------------------------------------------------------
# manifest guard
# --------------------------------------------------------------------------
def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_rows(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != MANIFEST.name:
            rows.append({"relpath": p.relative_to(root).as_posix(),
                         "sha256": sha256_of(p), "bytes": p.stat().st_size})
    return rows


def write_manifest() -> pd.DataFrame:
    df = pd.DataFrame(_manifest_rows(DATA))
    df.to_csv(MANIFEST, index=False)
    return df


def verify_manifest(stage: str) -> None:
    """Abort if data/ has drifted from the stored manifest."""
    if not MANIFEST.exists():
        sys.exit("[%s] FATAL: %s missing - run step00_stage_data.py first." % (stage, MANIFEST))
    stored = pd.read_csv(MANIFEST).sort_values("relpath").reset_index(drop=True)
    current = pd.DataFrame(_manifest_rows(DATA)).sort_values("relpath").reset_index(drop=True)
    if len(stored) != len(current):
        sys.exit("[%s] FATAL: data/ file count changed (%d -> %d). Refusing to run."
                 % (stage, len(stored), len(current)))
    if not (stored["relpath"].to_numpy() == current["relpath"].to_numpy()).all():
        sys.exit("[%s] FATAL: data/ file names changed. Refusing to run." % stage)
    bad = stored["sha256"].to_numpy() != current["sha256"].to_numpy()
    if bad.any():
        names = ", ".join(current.loc[bad, "relpath"].head(5))
        sys.exit("[%s] FATAL: input checksum mismatch: %s. Refusing to run." % (stage, names))


# --------------------------------------------------------------------------
# RELAB I/O
# --------------------------------------------------------------------------
def read_relab_tab(path) -> tuple[np.ndarray, np.ndarray]:
    """RELAB PDS4 .tab: first line is the record count, then wavelength (nm) and
    reflectance. Returns (wavelength in microns, reflectance), fill rows dropped, sorted."""
    lines = Path(path).read_text(errors="replace").splitlines()
    total = int(float(lines[0].strip()))
    data = np.genfromtxt(lines[1:total + 1], dtype=float, usecols=(0, 1))
    lam, r = data[:, 0] * 1e-3, data[:, 1]
    ok = np.isfinite(lam) & np.isfinite(r) & (r > 0) & (r < 2)
    order = np.argsort(lam[ok])
    return lam[ok][order], r[ok][order]


_XML = {
    "specimen_id": r"<speclib:specimen_id>(.*?)</speclib:specimen_id>",
    "specimen_name": r"<speclib:specimen_name>(.*?)</speclib:specimen_name>",
    "description": r"<speclib:specimen_description>(.*?)</speclib:specimen_description>",
    "size_min_um": r"<speclib:specimen_min_size[^>]*>(.*?)</speclib:specimen_min_size>",
    "size_max_um": r"<speclib:specimen_max_size[^>]*>(.*?)</speclib:specimen_max_size>",
    "incidence_deg": r"<speclib:incidence_angle[^>]*>(.*?)</speclib:incidence_angle>",
    "emission_deg": r"<speclib:emission_angle[^>]*>(.*?)</speclib:emission_angle>",
}


def read_relab_xml(path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, pat in _XML.items():
        m = re.search(pat, text, re.S)
        out[key] = (m.group(1).strip() if m else "").replace("&lt;", "<").replace("&gt;", ">") \
            .replace("&quot;", '"').replace("&amp;", "&")
    return out


# --------------------------------------------------------------------------
# LSCC Kdata spectra (300-2600 nm, 5 nm): columns <10, 10-20, 20-45, <45 um
# --------------------------------------------------------------------------
def read_kdata(soil: str) -> pd.DataFrame:
    """LSCC Kdata file -> DataFrame with lam_um and one column per available fraction.

    The files mix CR / LF line endings and leave empty sd cells, so lines are split on
    any CR/LF and fields on tabs, keeping empties (the same reader as the Korokhin
    reproduction, which parsed these files first)."""
    raw = (LSCC_DIR / ("%sKdata.txt" % soil)).read_text(errors="replace")
    lines = [l for l in re.split(r"\r\n|\r|\n", raw) if l.strip()]
    head = lines[0].split("\t")
    rows = []
    for l in lines[1:]:
        f = l.split("\t")
        rows.append([float(v) if v.strip() not in ("", "-") else np.nan for v in f[:len(head)]]
                    + [np.nan] * (len(head) - len(f)))
    df = pd.DataFrame(rows, columns=head)
    out = {"lam_um": df.iloc[:, 0].to_numpy() / 1000.0}
    for c in head[1:]:
        c2 = c.replace(soil, "").replace("_", "").strip()
        if c2 in FRACTIONS:
            out[c2] = df[c].to_numpy()
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def banner(stage: str, title: str) -> float:
    print("\n" + "=" * 78)
    print("[%s] %s" % (stage, title))
    print("=" * 78, flush=True)
    return time.time()


def done(stage: str, t0: float) -> None:
    verify_manifest(stage)
    print("[%s] done in %.1f s (manifest re-verified)" % (stage, time.time() - t0), flush=True)


def continuum_removed(lam_um: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Divide by the upper convex hull, vectorised over a stack of spectra (rows).

    Monotone-chain upper hull per spectrum; spectra are short (71 bands) so a Python
    loop over spectra is fast enough for the lab sets, and `continuum_removed_fast`
    handles image cubes."""
    R = np.atleast_2d(np.asarray(R, float))
    out = np.empty_like(R)
    x = np.asarray(lam_um, float)
    for i, y in enumerate(R):
        hull = [0]
        for j in range(1, len(x)):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                # keep b only if it lies above the chord a -> j
                if (y[b] - y[a]) * (x[j] - x[a]) <= (y[j] - y[a]) * (x[b] - x[a]):
                    hull.pop()
                else:
                    break
            hull.append(j)
        cont = np.interp(x, x[hull], y[hull])
        out[i] = y / cont
    return out


def continuum_removed_fast(lam_um: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Same upper-hull continuum removal as continuum_removed, vectorised over rows.

    Monotone chain run on all spectra at once (one stack per row); the hull is then
    interpolated band by band from the previous and next hull vertex of each row."""
    R = np.atleast_2d(np.asarray(R, float))
    x = np.asarray(lam_um, float)
    n, L = R.shape
    rows = np.arange(n)
    stack = np.zeros((n, L), int)
    top = np.ones(n, int)
    for j in range(1, L):
        active = rows
        while active.size:
            t = top[active]
            can = t >= 2
            active = active[can]
            if not active.size:
                break
            t = top[active]
            a, b = stack[active, t - 2], stack[active, t - 1]
            ya, yb, yj = R[active, a], R[active, b], R[active, j]
            pop = (yb - ya) * (x[j] - x[a]) <= (yj - ya) * (x[b] - x[a])
            active = active[pop]
            top[active] -= 1
        stack[rows, top] = j
        top += 1
    on = np.zeros((n, L), bool)
    for t in range(L):
        sel = top > t
        on[rows[sel], stack[sel, t]] = True
    idx = np.broadcast_to(np.arange(L), (n, L))
    prev = np.maximum.accumulate(np.where(on, idx, 0), axis=1)
    nxt = np.minimum.accumulate(np.where(on, idx, L - 1)[:, ::-1], axis=1)[:, ::-1]
    yp, yn = np.take_along_axis(R, prev, 1), np.take_along_axis(R, nxt, 1)
    xp, xn = x[prev], x[nxt]
    span = np.where(nxt > prev, xn - xp, 1.0)
    cont = np.where(nxt > prev, yp + (yn - yp) * (x[None, :] - xp) / span, R)
    return R / cont


def spectral_features(R: np.ndarray, lam_um: np.ndarray | None = None) -> np.ndarray:
    """Model input: (n, 2, L) = [log reflectance, convex-hull continuum-removed R]."""
    lam_um = input_wavelengths_um() if lam_um is None else lam_um
    R = np.clip(np.atleast_2d(np.asarray(R, float)), 1e-4, None)
    return np.stack([np.log(R), continuum_removed_fast(lam_um, R)], axis=1).astype(np.float32)
