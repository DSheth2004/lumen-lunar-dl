"""
Step 01 -- the labelled laboratory dataset: 19 LSCC soils x 4 sieve fractions.

Each RELAB Kdata spectrum (300-2600 nm, 5 nm) is convolved onto the 71 M3 input bands
(540.84-2497.11 nm, Gaussian responses of FWHM = band spacing). Targets:

  COMP (7-part, vol %): PLG, LCP = opx + pigeonite, HCP = Mg- + Fe-cpx, OLV, ILM,
        AGG = agglutinitic glass, OTH = volcanic glass + others (+ K-phases).
        Table 1 (mare) / Table 2 (highland) give the phase totals; the pyroxene split
        comes from Table 2 / Table 3 and is rescaled to the Table 1/2 pyroxene total so
        the composition stays closed. Measured on <10, 10-20, 20-45 only.
  CHEM (wt %): FeO, TiO2, Al2O3, CaO, MgO -- all four fractions.
  MAT: log10(Is/FeO) -- all four fractions.

Grouping columns (soil, set, mission, fraction) drive every cross-validation split.
Nothing here is fitted; this stage only assembles and checks.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import lumen_common as lc

STAGE = "step01"


def load_modal() -> pd.DataFrame:
    m = pd.concat([pd.read_csv(lc.GT_DIR / "lscc_modal_mineralogy.csv"),
                   pd.read_csv(lc.GT_DIR / "lscc_highland_modal_mineralogy.csv")])
    m["sample"] = m["sample"].astype(str)
    return m


def composition(modal: pd.DataFrame, soil: str, frac: str) -> dict | None:
    g = modal[(modal["sample"] == soil) & (modal["size_fraction_um"] == frac)]
    if g.empty:
        return None
    v = g.groupby("phase")["modal_pct"].first()
    pyx = float(v.get("pyroxene", np.nan))
    lcp_raw = float(v.get("orthopyroxene", 0.0) + v.get("pigeonite", 0.0))
    hcp_raw = float(v.get("mg_clinopyroxene", 0.0) + v.get("fe_clinopyroxene", 0.0))
    if not np.isfinite(pyx) or lcp_raw + hcp_raw <= 0:
        return None
    comp = {
        "PLG": float(v.get("plagioclase", 0.0)),
        "LCP": pyx * lcp_raw / (lcp_raw + hcp_raw),
        "HCP": pyx * hcp_raw / (lcp_raw + hcp_raw),
        "OLV": float(v.get("olivine", 0.0)),
        "ILM": float(v.get("ilmenite", 0.0)),
        "AGG": float(v.get("agglutinitic_glass", 0.0)),
        "OTH": float(v.get("volcanic_glass", 0.0) + v.get("others", 0.0) + v.get("k_phases", 0.0)),
    }
    tot = sum(comp.values())
    out = {k: 100.0 * x / tot for k, x in comp.items()}
    out["printed_total"] = tot
    return out


def main() -> int:
    t0 = lc.banner(STAGE, "labelled LSCC dataset on the M3 band grid")
    lc.verify_manifest(STAGE)

    lam_in = lc.input_wavelengths_um()
    assert lam_in.size == 71 and abs(lam_in[0] - 0.54084) < 1e-6 and abs(lam_in[-1] - 2.49711) < 1e-6
    modal = load_modal()
    chem = pd.concat([pd.read_csv(lc.GT_DIR / "lscc_bulk_chemistry.csv"),
                      pd.read_csv(lc.GT_DIR / "lscc_highland_bulk_chemistry.csv")])
    chem["sample"] = chem["sample"].astype(str)
    isfeo = pd.concat([pd.read_csv(lc.GT_DIR / "lscc_is_feo.csv"),
                       pd.read_csv(lc.GT_DIR / "lscc_highland_is_feo.csv")])
    isfeo["sample"] = isfeo["sample"].astype(str)

    rows, X, Xfull = [], [], []
    lam_full = None
    for soil in lc.ALL_SOILS:
        kd = lc.read_kdata(soil)
        lam_src = kd["lam_um"].to_numpy()
        if lam_full is None:
            lam_full = lam_src
        assert np.allclose(lam_src, lam_full), "Kdata grids differ between soils"
        for frac in lc.FRACTIONS:
            if frac not in kd or not np.isfinite(kd[frac]).sum() > 400:
                print("  %s %-5s: no spectrum in Kdata -> skipped" % (soil, frac))
                continue
            y = kd[frac].to_numpy()
            if np.isnan(y).any():
                ok = np.isfinite(y)
                y = np.interp(lam_src, lam_src[ok], y[ok])
            r = lc.srf_resample(lam_src, y, lam_in)
            assert np.isfinite(r).all()
            row = {"soil": soil, "set": lc.SOIL_SET[soil], "mission": lc.MISSION[soil],
                   "fraction": frac, "d_nominal_um": lc.FRACTION_NOMINAL_D_UM[frac]}
            c = composition(modal, soil, frac) if frac in lc.MODAL_FRACTIONS else None
            for k in lc.COMP:
                row[k] = c[k] if c else np.nan
            row["modal_printed_total"] = c["printed_total"] if c else np.nan
            ch = chem[(chem["sample"] == soil) & (chem["size_fraction_um"] == frac)]
            for k in lc.CHEM:
                row[k] = float(ch[k].iloc[0]) if len(ch) else np.nan
            fi = isfeo[(isfeo["sample"] == soil) & (isfeo["size_fraction_um"] == frac)]
            row["IsFeO"] = float(fi["is_feo"].iloc[0]) if len(fi) else np.nan
            row[lc.MAT] = np.log10(row["IsFeO"]) if np.isfinite(row["IsFeO"]) else np.nan
            rows.append(row)
            X.append(r)
            Xfull.append(y)

    meta = pd.DataFrame(rows)
    X = np.asarray(X, np.float32)
    Xfull = np.asarray(Xfull, np.float32)

    # ---- checks -----------------------------------------------------------------
    n_comp = meta[lc.COMP].notna().all(axis=1).sum()
    n_chem = meta[lc.CHEM].notna().all(axis=1).sum()
    n_mat = meta[lc.MAT].notna().sum()
    print("  spectra: %d   (soils %d)   with composition: %d   chemistry: %d   Is/FeO: %d"
          % (len(meta), meta["soil"].nunique(), n_comp, n_chem, n_mat))
    tot = meta["modal_printed_total"].dropna()
    print("  printed modal totals: %.1f - %.1f %%" % (tot.min(), tot.max()))
    assert np.allclose(meta.loc[meta[lc.COMP].notna().all(axis=1), lc.COMP].sum(axis=1), 100.0)
    print("  missions: %s" % meta.groupby("mission")["soil"].nunique().to_dict())
    rng_tab = meta.groupby("set")[lc.CHEM + ["IsFeO"]].agg(["min", "max"]).round(2)
    print(rng_tab.to_string())

    meta.to_csv(lc.TABLES / "step01_lab_dataset.csv", index=False)
    np.savez_compressed(lc.PRODUCTS / "lab_dataset.npz", X=X, lam_um=lam_in, Xfull=Xfull,
                        lam_full_um=lam_full)
    lc.done(STAGE, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
