"""
Step 00 -- stage every input into data/ and write the SHA-256 manifest.

Sources are opened read-only. After this stage the folder is self-contained: every
later stage reads only data/, verifies the manifest at its start and end, and aborts
on any drift.

Staged:
  lscc/            19 LSCC Kdata spectra (RELAB, 300-2600 nm, four sieve fractions)
  ground_truth/    LSCC modal mineralogy, bulk chemistry and Is/FeO (mare: Taylor et al.
                   2001, transcribed and checked in the Jung/Korokhin reproductions;
                   highland: Taylor et al. 2010, transcribed in the Korokhin reproduction)
                   + the 50-sample FeO list of Fernandez-Diaz et al. (2026)
  relab/           the end-member library of lumen_common.ENDMEMBER_LIBRARY (.tab + .xml)
  optical_constants/  Fe n,k (Johnson & Christy 1974; Ordal et al. 1988)
  m3_shapley/      M3G20090728T214215 block (Korokhin 2024 scene) + Apollo 16 block
                   (M3G20090607T110414), with locations and band centres
  m3_aristarchus/  the two Jung (2026) Aristarchus ROI blocks
  baselines/       the published-method outputs this project is compared against, as
                   computed by the reproductions in this repository (read-only copies)

Identity check: every end-member's PDS4 label must name the expected specimen; a
shuffled assignment is run as a negative control and must be rejected.
"""
from __future__ import annotations

import shutil
import sys

import numpy as np
import pandas as pd

import lumen_common as lc

STAGE = "step00"


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        sys.exit("[%s] FATAL: missing source %s" % (STAGE, src))
    shutil.copy2(src, dst)


def main() -> int:
    t0 = lc.banner(STAGE, "stage inputs into data/ and write the SHA-256 manifest")

    # Self-contained mode: when the sibling source folders are absent (e.g. a fresh clone
    # of this project on another machine / in a cloud session), data/ is already staged
    # and shipped with its manifest -- verify it and stop, never re-write it.
    if not (lc.SRC_KOROKHIN.exists() and lc.SRC_JUNG.exists() and lc.SRC_RELAB_CART.exists()):
        print("  source folders not present -> verifying the shipped data/ against its manifest")
        lc.verify_manifest(STAGE)
        print("  data/ verified: %d files" % len(pd.read_csv(lc.MANIFEST)))
        return 0

    # --- LSCC spectra and ground truth -------------------------------------------
    for soil in lc.ALL_SOILS:
        copy(lc.SRC_KOROKHIN / "data" / "lscc" / ("%sKdata.txt" % soil),
             lc.LSCC_DIR / ("%sKdata.txt" % soil))
    for name in ["lscc_modal_mineralogy.csv", "lscc_bulk_chemistry.csv", "lscc_is_feo.csv",
                 "lscc_highland_modal_mineralogy.csv", "lscc_highland_bulk_chemistry.csv",
                 "lscc_highland_is_feo.csv", "README.md"]:
        copy(lc.SRC_KOROKHIN / "data" / "ground_truth" / name, lc.GT_DIR / name)
    copy(lc.SRC_FERNANDEZ / "data" / "processed" / "ground_truth" / "ground_truth_50.csv",
         lc.GT_DIR / "fernandez2026_ground_truth_50.csv")

    # --- RELAB end-members -------------------------------------------------------
    jung_relab = lc.SRC_JUNG / "data" / "relab"
    for sid in lc.ENDMEMBER_LIBRARY["spectrum_id"]:
        for ext in (".tab", ".xml"):
            src = lc.SRC_RELAB_CART / (sid + ext)
            if not src.exists():
                src = jung_relab / (sid + ext)
            copy(src, lc.RELAB_DIR / (sid + ext))

    # --- Fe optical constants ----------------------------------------------------
    for p in (lc.SRC_KOROKHIN / "data" / "optical_constants").iterdir():
        copy(p, lc.OPTICAL_DIR / p.name)

    # --- M3 blocks ---------------------------------------------------------------
    for name in ["scene_rfl.npy", "scene_loc.npy", "apollo16_rfl.npy", "apollo16_loc.npy",
                 "m3_wavelengths_nm.csv", "scene_meta.json", "sites.csv"]:
        copy(lc.SRC_KOROKHIN / "data" / "m3" / name, lc.M3_SHAPLEY_DIR / name)
    for p in (lc.SRC_JUNG / "data" / "m3_aristarchus").iterdir():
        copy(p, lc.M3_ARISTARCHUS_DIR / p.name)

    # --- published-method outputs (baselines) ------------------------------------
    kt, kp = lc.SRC_KOROKHIN / "tables", lc.SRC_KOROKHIN / "products"
    for name in ["step10_lscc_fits.csv", "step10_lscc_summary.csv",
                 "step10_lscc_summary_by_set.csv", "step08_c1_recovery.csv",
                 "step08_c3_timing.csv"]:
        copy(kt / name, lc.BASELINE_DIR / "korokhin2024" / name)
    for name in ["ann_full_median_iter3.npy", "highland_mask.npy", "mean_albedo.npy"]:
        copy(kp / name, lc.BASELINE_DIR / "korokhin2024" / name)
    jt = lc.SRC_JUNG / "tables"
    for name in ["step05_vs_ground_truth.csv", "step05_lscc_abundances.csv",
                 "step07_accuracy_claims.csv", "step07_prior_baseline.csv",
                 "step07_maturity_check.csv", "step06_m3_abundances.csv"]:
        copy(jt / name, lc.BASELINE_DIR / "jung2026" / name)

    # --- identity check of the end-member library --------------------------------
    rows = []
    for _, r in lc.ENDMEMBER_LIBRARY.iterrows():
        meta = lc.read_relab_xml(lc.RELAB_DIR / (r["spectrum_id"] + ".xml"))
        lam, refl = lc.read_relab_tab(lc.RELAB_DIR / (r["spectrum_id"] + ".tab"))
        name = meta["specimen_name"]
        ok = r["specimen_must_contain"].split(" D")[0] in name
        rows.append({**r.to_dict(), **meta, "lam_min_um": lam.min(), "lam_max_um": lam.max(),
                     "n_points": len(lam), "identity_ok": ok,
                     "geometry_ok": (meta["incidence_deg"].startswith("30")
                                     and meta["emission_deg"] in ("0", "0.0", "-0", "-0.0"))})
    ident = pd.DataFrame(rows)
    ident.to_csv(lc.TABLES / "step00_endmember_identity.csv", index=False)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ident))
    neg = [ident["specimen_must_contain"].iloc[j].split(" D")[0] in ident["specimen_name"].iloc[i]
           for i, j in enumerate(perm)]
    n_neg_pass = int(np.sum(neg))
    n_fixed = int(np.sum([ident["specimen_must_contain"].iloc[j].split(" D")[0] ==
                          ident["specimen_must_contain"].iloc[i].split(" D")[0]
                          for i, j in enumerate(perm)]))
    print("  end-member identity: %d / %d match their PDS4 labels"
          % (ident["identity_ok"].sum(), len(ident)))
    print("  geometry i=30/e=0:   %d / %d" % (ident["geometry_ok"].sum(), len(ident)))
    print("  shuffled control:    %d / %d accepted (%d are same-specimen fixed points)"
          % (n_neg_pass, len(ident), n_fixed))
    if not ident["identity_ok"].all():
        print(ident.loc[~ident["identity_ok"], ["spectrum_id", "specimen_must_contain",
                                                "specimen_name"]].to_string())
        sys.exit("[%s] FATAL: end-member identity check failed" % STAGE)
    if n_neg_pass > n_fixed:
        sys.exit("[%s] FATAL: negative control accepted non-matching specimens" % STAGE)

    # --- leakage rule: no end-member from an evaluation soil ----------------------
    leak = [s for s in ident["specimen_name"] if any(soil in s for soil in lc.ALL_SOILS)]
    if leak:
        sys.exit("[%s] FATAL: end-member(s) derived from an evaluation soil: %s" % (STAGE, leak))
    print("  leakage rule:        no end-member derived from the 19 evaluation soils")

    df = lc.write_manifest()
    print("  manifest: %d files, %.1f MB" % (len(df), df["bytes"].sum() / 1e6))
    lc.done(STAGE, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
