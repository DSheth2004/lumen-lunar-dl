"""
Shared definitions for the Korokhin et al. (2024) reproduction.

Paths, the constants the paper fixes, the SHA-256 manifest guard and RELAB I/O.

data/ is checksummed (data/manifest.csv) and re-verified at the START and END of every
stage, so an accidental modification aborts the run instead of quietly changing a
result.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
PAPER_ROOT = Path(__file__).resolve().parent.parent
IMPL_ROOT = PAPER_ROOT.parent                      # paper-implementations/
REPO_ROOT = IMPL_ROOT.parent                       # RMS/
DATA = PAPER_ROOT / "data"
RELAB_DIR = DATA / "relab"
GROUND_TRUTH = DATA / "ground_truth"
OPTICAL = DATA / "optical_constants"
M3_DIR = DATA / "m3"
TABLES = PAPER_ROOT / "tables"
PRODUCTS = PAPER_ROOT / "products"                 # derived arrays (processed cube, maps)
FIGURES = PAPER_ROOT / "figures"
REFERENCE = PAPER_ROOT / "reference"
PAPER_TABLES = REFERENCE / "paper_tables"
DIGITIZED = REFERENCE / "digitized"
PDF_IMAGES = REFERENCE / "pdf_images"
MANIFEST = DATA / "manifest.csv"
PAPER_PDF = IMPL_ROOT / "papers" / "3rd.pdf"

# read-only sources outside this directory
RELAB_RAW = REPO_ROOT / "RELAB"                    # cartOrder*/cartorder/*.tab|xml
MARE_GROUND_TRUTH = IMPL_ROOT / "Jung2026_SpaceWeatheredMaria" / "data" / "ground_truth"  # LSCC mare tables, transcribed and checked there

for _d in (TABLES, FIGURES, PAPER_TABLES, DIGITIZED, PDF_IMAGES, PRODUCTS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# constants the paper fixes (section and equation numbers are the paper's)
# --------------------------------------------------------------------------
Q = 0.5                    # packing density, Sec. 3.1 / 4.1(2)
N_TRANSPARENT = 1.6        # CPX, OPX, OLV, PLG, Sec. 3.1 / 4.1(5)
N_OPAQUE = 2.0             # ilmenite, gray component, metal, Sec. 3.1 / 4.1(6)
S_LAB_UM = 30.0            # grain size used to turn the basis spectra into k(lambda), Table 4
A_GRAY = 0.15              # gray-component albedo used for M3, Sec. 3.2
D_RIND_UM = 0.2            # d = 2h, h = 100 nm rind, Appendix A.4 / Eq. A14

# The M3 spectra are resampled to 0.1 um steps: 21 points, 0.54 ... 2.54 um (Sec. 5.3(1)).
M3_GRID_UM = np.round(np.arange(0.54, 2.5401, 0.1), 2)
assert M3_GRID_UM.size == 21

# Eq. (2): M3 albedo -> RELAB photometric system, from the Apollo 16 / 62231 comparison.
EQ2_SLOPE = 1.685
EQ2_INTERCEPT = 0.0355
EQ2_SCENE = "M3G20090607T110414"
APOLLO16_LATLON = (-8.973, 15.500)                 # Apollo 16 LM site
SHAPLEY_LATLON = (9.4, 56.9)                       # 9 deg 24' N, 56 deg 54' E

# Order of the 8 free parameters: the column order of Table 11.
PARAMS = ["S", "nFe", "Gr", "CPX", "OPX", "OLV", "PLG", "ILM"]
MINERALS = ["CPX", "OPX", "OLV", "PLG", "ILM"]
TRANSPARENT = ["CPX", "OPX", "OLV", "PLG"]

# Table 11: ranges for generating the iteration-1 training set. Concentrations in %.
TABLE11_MIN = {"S": 0.0, "nFe": 0.0, "Gr": 20.0, "CPX": 0.0, "OPX": -2.0, "OLV": -2.0,
               "PLG": 0.0, "ILM": 0.0}
TABLE11_MAX = {"S": 120.0, "nFe": 0.2, "Gr": 40.0, "CPX": 50.0, "OPX": 5.0, "OLV": 5.0,
               "PLG": 35.0, "ILM": 20.0}
N_TRAIN_ITER1 = 20_000

# Table 4: the basis spectra set actually used. RELAB spectrum IDs resolved from the
# sample IDs by reading the local PDS4 labels (all <45 um, "D" fraction).
BASIS = {
    "CPX": {"spectrum": "c1lr215", "relab_id": "LR-CMP-215", "sample": "70017", "n": N_TRANSPARENT},
    "OPX": {"spectrum": "c1lr220", "relab_id": "LR-CMP-220", "sample": "70035", "n": N_TRANSPARENT},
    "OLV": {"spectrum": "c1lr212", "relab_id": "LR-CMP-212", "sample": "15555", "n": N_TRANSPARENT},
    "PLG": {"spectrum": "c1lr217", "relab_id": "LR-CMP-217", "sample": "70017", "n": N_TRANSPARENT},
    "ILM": {"spectrum": "c1lr218", "relab_id": "LR-CMP-218", "sample": "70017", "n": N_OPAQUE},
}

# ANN, Sec. 5.3 and Table 12
ANN_LAYERS = (21, 42, 12, 8)
ANN_MOMENTUM = 0.9
ANN_OUT_LO, ANN_OUT_HI = 0.05, 0.95

SEED = 0
SEEDS = (0, 1, 2, 3, 4)


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
        sys.exit("[%s] FATAL: %s missing - run step00_stage_inputs.py first."
                 % (stage, MANIFEST))
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
    """Read a RELAB PDS4 .tab spectrum: first line is the record count, then
    wavelength (nm) and reflectance. Returns wavelength in MICRONS."""
    lines = Path(path).read_text().splitlines()
    total = int(float(lines[0].strip()))
    data = np.genfromtxt(lines[1:total + 1], dtype=float)
    return data[:, 0] * 1e-3, data[:, 1]


_XML_PATTERNS = {
    "relab_id": r"<name>((?:LR|LS|SC|MS|XT|PO|PE|PA|SB|SW|JB|PI|DD|CG)-[A-Z]+-\d+[A-Z-]*)</name>",
    "specimen": r"<speclib:specimen_name>(.*?)</speclib:specimen_name>",
    "size_min_um": r"<speclib:specimen_min_size[^>]*>(.*?)</speclib:specimen_min_size>",
    "size_max_um": r"<speclib:specimen_max_size[^>]*>(.*?)</speclib:specimen_max_size>",
    "incidence_deg": r"<speclib:incidence_angle[^>]*>(.*?)</speclib:incidence_angle>",
    "emission_deg": r"<speclib:emission_angle[^>]*>(.*?)</speclib:emission_angle>",
    "specimen_type": r"<speclib:specimen_type>(.*?)</speclib:specimen_type>",
}


def read_relab_xml(path) -> dict:
    """Identity and geometry fields from a RELAB PDS4 label."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, pat in _XML_PATTERNS.items():
        m = re.search(pat, text, re.S)
        out[key] = m.group(1).strip() if m else ""
    out["specimen"] = out["specimen"].replace("&lt;", "<").replace("&gt;", ">")
    return out


def relab_spectrum(spectrum_id: str) -> tuple[np.ndarray, np.ndarray]:
    """A staged RELAB spectrum (wavelength um, reflectance), NaN/fill rows dropped."""
    lam, r = read_relab_tab(RELAB_DIR / ("%s.tab" % spectrum_id))
    ok = np.isfinite(lam) & np.isfinite(r) & (r > 0) & (r < 2)
    order = np.argsort(lam[ok])
    return lam[ok][order], r[ok][order]


def read_kdata(sample: str) -> pd.DataFrame:
    """LSCC Kdata file -> columns lam_um, '<10', '10-20', '20-45', '<45' (reflectance).

    The files mix line-ending conventions (some are CR-only) and leave empty sd
    cells, so lines are split on any CR/LF and fields on tabs, keeping empties.
    """
    raw = (DATA / "lscc" / ("%sKdata.txt" % sample)).read_text(errors="replace")
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
        c2 = c.replace(sample, "").replace("_", "").strip()
        if c2 in ("<10", "10-20", "20-45", "<45"):
            out[c2] = df[c].to_numpy()
    return pd.DataFrame(out)


def gauss_to_grid(lam_src, y_src, grid=M3_GRID_UM, fwhm_um: float = 0.1) -> np.ndarray:
    """Gaussian convolution (FWHM 0.1 um) of a finely sampled spectrum onto `grid` --
    the same kernel step05 applies to M3, so lab and orbital spectra match."""
    lam_src, y_src = np.asarray(lam_src, float), np.asarray(y_src, float)
    ok = np.isfinite(y_src)
    sig = fwhm_um / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    W = np.exp(-0.5 * ((np.asarray(grid)[:, None] - lam_src[ok][None, :]) / sig) ** 2)
    return (W / W.sum(axis=1, keepdims=True)) @ y_src[ok]


def resample(lam_src, y_src, lam_dst) -> np.ndarray:
    """Linear interpolation; NaN outside the source range (never extrapolated)."""
    return np.interp(lam_dst, lam_src, y_src, left=np.nan, right=np.nan)


# --------------------------------------------------------------------------
# fit-quality measures (Eq. 1 and the sigma_rel used in every table)
# --------------------------------------------------------------------------
def sigma(model, observed) -> np.ndarray:
    """Eq. (1): RMS deviation over the spectral points (last axis)."""
    d = np.asarray(model) - np.asarray(observed)
    return np.sqrt(np.nanmean(d * d, axis=-1))


def sigma_rel(model, observed, mean_albedo=None) -> np.ndarray:
    """sigma_Rel = 100% sigma / <A>, <A> = albedo averaged over the fitted range."""
    obs = np.asarray(observed)
    a = np.nanmean(obs, axis=-1) if mean_albedo is None else mean_albedo
    return 100.0 * sigma(model, observed) / a


def banner(stage: str, msg: str = "") -> None:
    print("\n" + "=" * 78)
    print("[%s] %s" % (stage, msg))
    print("=" * 78)
