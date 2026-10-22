"""
Shared definitions for the Jung et al. (2026) reproduction.

Jung, J.-I., Lapotre, M. G. A., Milliken, R. E., & Minson, S. E. (2026),
"Remote Compositional Analyses of Space-weathered Lunar Maria",
The Planetary Science Journal 7(1) 18.  doi:10.3847/PSJ/ae2b57  (CC-BY-4.0)

Reference implementation by the authors (CC-BY-4.0):
    Jung, J.-I., Lapotre, M., & Minson, S. (2025), Zenodo, doi:10.5281/zenodo.17258920

This module holds paths, the read-only input guard, RELAB I/O, and the tables of
end-member / spectrum identifiers that the rest of the pipeline is driven from.
Nothing here writes into any source data folder.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent
REPO_ROOT = PAPER_ROOT.parent.parent

DATA = PAPER_ROOT / "data"
RELAB_STAGED = DATA / "relab"
GROUND_TRUTH = DATA / "ground_truth"
M3_STAGED = DATA / "m3_aristarchus"
REFERENCE = PAPER_ROOT / "reference"
ZENODO = REFERENCE / "space-weathering-main"

TABLES = PAPER_ROOT / "tables"
FIGURES = PAPER_ROOT / "figures"

MANIFEST = DATA / "manifest.csv"

# Read-only source folder. Never written to, and not required once data/ is
# staged -- step00 falls back to the staged copies when it is absent, so this
# folder runs standalone.
SRC_RELAB = REPO_ROOT / "RELAB"

for _d in (TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Measurement geometry
#
# Every RELAB spectrum used by the paper is labelled incidence 30 deg /
# emission 0 deg (verified from the PDS4 labels, see step01), and the authors'
# example notebook instantiates HapkeRTM with exactly that geometry.
# --------------------------------------------------------------------------
INCIDENCE_DEG = 30.0
EMERGENCE_DEG = 0.0
PHASE_DEG = 30.0
MU0 = float(np.cos(np.radians(INCIDENCE_DEG)))
MU = float(np.cos(np.radians(EMERGENCE_DEG)))

# Wavelength window the authors' load_data.m fits over.
FIT_LAM_MIN_UM = 0.55
FIT_LAM_MAX_UM = 2.50

# Band-depth reference wavelength (paper Appendix A).
BAND_REF_UM = 0.970

# --------------------------------------------------------------------------
# Nine compositional end-members (paper Table C1).
#
# rho_paper / n_paper are Table C1 as printed; rho_code / n_code are the values
# actually used by the authors' released code (load_data.m rho vector and column
# 2 of each "data-this study/*.txt").  Where they disagree both are carried so
# the audit can report which one reproduces the published result.
# --------------------------------------------------------------------------
ENDMEMBERS = pd.DataFrame(
    [
        ("olivine",           "c1lr169", "LR-CMP-169", "15555,965",    3320.0, 3320.0, 1.83, 1.83,  80.0),
        ("plagioclase",       "c1lr177", "LR-CMP-177", "70017,535",    2690.0, 2690.0, 1.56, 1.56,  80.0),
        ("pigeonite",         "c1lr176", "LR-CMP-176", "70017,535",    3400.0, 3400.0, 1.77, 1.77,  80.0),
        ("augite",            "c1lr179", "LR-CMP-179", "70035,188",    3380.0, 3380.0, 1.73, 1.73,  80.0),
        ("ilmenite",          "c1lr178", "LR-CMP-178", "70017,535",    4270.0, 4720.0, 2.13, 2.13,  80.0),
        ("pyroxene_glass",    "c1sc53",  "SC-EAC-053", "Glass of L22", 3300.0, 3300.0, 1.64, 1.64,  50.0),
        ("plagioclase_glass", "c1sc50",  "SC-EAC-050", "Glass of L19", 3000.0, 3000.0, 1.64, 1.64,  50.0),
        ("orange_glass",      "c1lr51",  "LR-CMP-051", "74220,806",    3000.0, 3300.0, 1.64, 1.64, 150.0),
        ("agglutinate",       "c0ls152", "LS-BWD-152", "79221",        3000.0, 3000.0, 1.49, 2.16, 150.0),
    ],
    columns=["name", "relab_id", "relab_code", "specimen",
             "rho_paper", "rho_code", "n_paper", "n_code", "D_used_um"],
)

# Order used by the authors' load_data.m / prior vector.
ENDMEMBER_ORDER = list(ENDMEMBERS["name"])

# Dirichlet prior means, load_data.m component_means, in ENDMEMBER_ORDER.
PRIOR_MEANS = np.array([0.07, 0.39, 0.16, 0.16, 0.10, 0.115, 0.005 / 3, 0.005 / 3, 0.005 / 3])
PRIOR_ALPHA0 = 30.0
PRIOR_SIZE_LOW_UM = 0.0
PRIOR_SIZE_HIGH_UM = 990.0

# Crystalline phases reported in Fig. 4 (glass + agglutinate excluded, renormalised).
CRYSTALLINE = ["olivine", "plagioclase", "pigeonite", "augite", "ilmenite"]

# --------------------------------------------------------------------------
# Apollo spectra behind Step 1 (paper Fig. 2 / Fig. A1).
#
# role: "fresh" -> pulverized fresh mare basalt rocks, define d_dc
#       "sw"    -> most space-weathered soils, define d_sw and omega_sw
#       "soil"  -> the remaining LSCC mare soil spectra
# is_target marks the four soils inverted in Table 1 / Figs 3, 4, F1.
# --------------------------------------------------------------------------
SPECTRA = pd.DataFrame(
    [
        ("c1lr152", "LR-CMP-152", "15058", "0-125", "fresh", False),
        ("c1lr154", "LR-CMP-154", "15555", "0-125", "fresh", False),
        ("c1lr156", "LR-CMP-156", "70017", "0-125", "fresh", False),
        ("c1lr158", "LR-CMP-158", "70035", "0-125", "fresh", False),
        ("c1lr15",  "LR-CMP-015", "10084", "20-40", "sw",    False),
        ("c1lr12",  "LR-CMP-012", "15041", "20-44", "sw",    False),
        ("c1lr17",  "LR-CMP-017", "79221", "20-40", "sw",    False),
        ("c1lr10",  "LR-CMP-010", "12001", "20-44", "soil",  True),
        ("c1lr32",  "LR-CMP-032", "12030", "20-44", "soil",  True),
        ("c1lr20",  "LR-CMP-020", "15071", "20-40", "soil",  True),
        ("c1lr57",  "LR-CMP-057", "71061", "20-45", "soil",  True),
        ("c1lr23",  "LR-CMP-023", "70181", "20-44", "soil",  False),
        ("c1lr28",  "LR-CMP-028", "71501", "20-44", "soil",  False),
    ],
    columns=["relab_id", "relab_code", "sample", "size_um", "role", "is_target"],
)

ALL_RELAB_IDS = sorted(set(SPECTRA["relab_id"]) | set(ENDMEMBERS["relab_id"]))


# --------------------------------------------------------------------------
# Hashing / manifest guard
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_rows(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != MANIFEST.name:
            rows.append({
                "relpath": p.relative_to(root).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            })
    return rows


def write_manifest() -> pd.DataFrame:
    """Compute and store the SHA-256 manifest of data/."""
    df = pd.DataFrame(_manifest_rows(DATA))
    df.to_csv(MANIFEST, index=False)
    return df


def verify_manifest(stage: str) -> None:
    """Abort if data/ has drifted from the stored manifest.

    Called at the start AND end of every stage, matching the guard used by
    M3_COPERNICUS/scripts/m3_common.py.
    """
    if not MANIFEST.exists():
        sys.exit("[%s] FATAL: %s missing - run step00_stage_data.py first." % (stage, MANIFEST))
    stored = pd.read_csv(MANIFEST).sort_values("relpath").reset_index(drop=True)
    current = pd.DataFrame(_manifest_rows(DATA)).sort_values("relpath").reset_index(drop=True)
    if len(stored) != len(current):
        sys.exit("[%s] FATAL: data/ file count changed (%d -> %d). Refusing to run."
                 % (stage, len(stored), len(current)))
    bad = stored["sha256"].to_numpy() != current["sha256"].to_numpy()
    if bad.any():
        names = ", ".join(current.loc[bad, "relpath"].head(5))
        sys.exit("[%s] FATAL: input checksum mismatch: %s. Refusing to run." % (stage, names))


# --------------------------------------------------------------------------
# RELAB I/O
# --------------------------------------------------------------------------
def read_relab_tab(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a RELAB PDS4 .tab spectrum.

    Mirrors loadData.importspeclib in the authors' release: the first line is a
    record count, column 0 is wavelength in nm, column 1 is reflectance.
    Returns wavelength in MICRONS and reflectance.
    """
    lines = Path(path).read_text().splitlines()
    total = int(float(lines[0].strip()))
    data = np.genfromtxt(lines[1:total + 1], dtype=float)
    return data[:, 0] * 1e-3, data[:, 1]


_XML_PATTERNS = {
    "relab_code": r"<name>((?:LR|LS|SC)-[A-Z]+-\d+)</name>",
    "specimen": r"<speclib:specimen_name>(.*?)</speclib:specimen_name>",
    "size_min_um": r"<speclib:specimen_min_size[^>]*>(.*?)</speclib:specimen_min_size>",
    "size_max_um": r"<speclib:specimen_max_size[^>]*>(.*?)</speclib:specimen_max_size>",
    "incidence_deg": r"<speclib:incidence_angle[^>]*>(.*?)</speclib:incidence_angle>",
    "emission_deg": r"<speclib:emission_angle[^>]*>(.*?)</speclib:emission_angle>",
    "specimen_type": r"<speclib:specimen_type>(.*?)</speclib:specimen_type>",
    "material_type": r"<speclib:material_type>(.*?)</speclib:material_type>",
}


def read_relab_xml(path) -> dict:
    """Pull identity and geometry fields out of a RELAB PDS4 label."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, pat in _XML_PATTERNS.items():
        m = re.search(pat, text, re.S)
        out[key] = m.group(1).strip().replace("&lt;", "<").replace("&gt;", ">") if m else ""
    return out


def staged_tab(relab_id: str) -> Path:
    return RELAB_STAGED / (relab_id + ".tab")


def staged_xml(relab_id: str) -> Path:
    return RELAB_STAGED / (relab_id + ".xml")


def load_spectrum(relab_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a staged RELAB spectrum as (wavelength_um, reflectance)."""
    return read_relab_tab(staged_tab(relab_id))


# --------------------------------------------------------------------------
# Spectral helpers
# --------------------------------------------------------------------------
def convex_hull_removal(wl: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Continuum-remove by dividing through the upper convex hull.

    Equivalent to pysptools.spectro.convex_hull_removal, which the authors' code
    calls. Returns the continuum-removed spectrum (1.0 on the hull).
    One hull per spectrum over the supplied window, never per-band.
    """
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)
    hull: list[int] = []
    for i in range(len(wl)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (wl[b] - wl[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (wl[i] - wl[a])
            if cross >= 0:          # b lies below segment a-i, so drop it
                hull.pop()
            else:
                break
        hull.append(i)
    cont = np.interp(wl, wl[hull], y[hull])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cont > 0, y / cont, np.nan)


def band_depth_at(wl: np.ndarray, cr: np.ndarray, target_um: float = BAND_REF_UM) -> float:
    """Band depth = 1 - continuum-removed value at the band closest to target."""
    idx = int(np.argmin(np.abs(np.asarray(wl) - target_um)))
    return float(1.0 - cr[idx])


def restrict(wl, y, lo=FIT_LAM_MIN_UM, hi=FIT_LAM_MAX_UM):
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (wl >= lo) & (wl <= hi)
    return wl[m], y[m]


def r_squared(obs, model) -> float:
    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(model)
    obs, model = obs[ok], model[ok]
    ss_res = float(np.sum((obs - model) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def banner(stage: str, msg: str = "") -> None:
    print("\n" + "=" * 78 + "\n[%s] %s\n" % (stage, msg) + "=" * 78)
