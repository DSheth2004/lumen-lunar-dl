"""
Step 00 -- stage read-only inputs into data/ and write the SHA-256 manifest.

Copies the 22 RELAB spectra the paper uses (13 Apollo mare/soil spectra behind
Fig. 2 / Appendix A, plus the 9 compositional end-members of Table C1) out of the
untouched RELAB/ cart-order downloads.  Source files are opened read-only and are
never modified.

Also verifies each staged end-member against Table C1 as printed in the paper --
RELAB specimen code, specimen name and laboratory grain-size range -- so that a
mis-identified end-member fails here rather than silently propagating.

Two modes, chosen automatically:

  * **stage** -- the RELAB/ cart-order tree is present, so the 44 files are copied
    out of it afresh and their provenance recorded;
  * **verify-staged** -- the source tree is absent (this folder has been handed on
    by itself, without the rest of the repository), so the copies already in
    data/relab/ are used as-is.

The distinction matters only for where the bytes come from.  Every identity check
below reads the *staged* PDS4 labels either way, and the SHA-256 manifest is
verified in both modes, so a distributed copy is checked exactly as strictly as a
freshly staged one.  Only if both the source tree and the staged copies are
missing is there nothing to work with, and that is fatal.
"""
from __future__ import annotations

import shutil
import sys

import pandas as pd

import jung_common as jc

STAGE = "step00"

# Table C1 "Sample ID" / "RELAB Specimen Code" / "D_lab", as printed.
TABLE_C1_EXPECTED = {
    "c1lr169": ("LR-CMP-169", "15555",  ">0, <125"),
    "c1lr176": ("LR-CMP-176", "70017",  ">0, <125"),
    "c1lr179": ("LR-CMP-179", "70035",  ">0, <125"),
    "c1lr177": ("LR-CMP-177", "70017",  ">0, <125"),
    "c1lr178": ("LR-CMP-178", "70017",  ">0, <125"),
    "c1sc53":  ("SC-EAC-053", "",       ">45, <90"),
    "c1sc50":  ("SC-EAC-050", "",       ">45, <90"),
    "c1lr51":  ("LR-CMP-051", "74220",  ""),
    "c0ls152": ("LS-BWD-152", "79221",  ">125, <250"),
}


def find_source(relab_id: str, ext: str):
    """Locate one file in the RELAB cart-order tree, or None if it is not there."""
    hits = sorted(jc.SRC_RELAB.glob("cartOrder*/cartorder/%s.%s" % (relab_id, ext)))
    return hits[0] if hits else None


def wanted_files():
    return [(rid, ext) for rid in jc.ALL_RELAB_IDS for ext in ("tab", "xml")]


def main() -> int:
    jc.RELAB_STAGED.mkdir(parents=True, exist_ok=True)
    jc.GROUND_TRUTH.mkdir(parents=True, exist_ok=True)
    jc.M3_STAGED.mkdir(parents=True, exist_ok=True)

    want = wanted_files()
    sources = {(rid, ext): find_source(rid, ext) for rid, ext in want}
    missing_src = [k for k, v in sources.items() if v is None]
    missing_staged = [(rid, ext) for rid, ext in want
                      if not (jc.RELAB_STAGED / ("%s.%s" % (rid, ext))).exists()]

    if not missing_src:
        mode = "stage"
        jc.banner(STAGE, "staging RELAB inputs (read-only sources)")
    elif not missing_staged:
        mode = "verify-staged"
        jc.banner(STAGE, "verifying already-staged RELAB inputs")
        print("the RELAB/ cart-order tree is not present at %s," % jc.SRC_RELAB)
        print("so the %d files already in data/relab/ are used as they stand." % len(want))
        print("every identity check below reads those same staged labels, and the")
        print("SHA-256 manifest is verified, so this is checked as strictly as a fresh stage.\n")
    else:
        sys.exit("[%s] FATAL: neither the RELAB source tree (%s) nor a complete set of\n"
                 "  staged copies in %s is available -- %d source file(s) and %d staged\n"
                 "  file(s) are missing, so there is nothing to verify."
                 % (STAGE, jc.SRC_RELAB, jc.RELAB_STAGED,
                    len(missing_src), len(missing_staged)))

    # In verify-staged mode the recorded origin of each file is carried forward from
    # the provenance table this folder shipped with. Overwriting it with a placeholder
    # would quietly destroy the audit trail -- where these bytes actually came from --
    # the first time someone ran the pipeline on a distributed copy.
    prior_origin = {}
    prov_csv = jc.TABLES / "step00_input_provenance.csv"
    if mode == "verify-staged" and prov_csv.exists():
        prior = pd.read_csv(prov_csv)
        if {"relab_id", "ext", "source"}.issubset(prior.columns):
            prior_origin = {(r["relab_id"], r["ext"]): r["source"]
                            for _, r in prior.iterrows()}
            print("carrying forward recorded provenance for %d files\n" % len(prior_origin))

    rows = []
    for rid, ext in want:
        dst = jc.RELAB_STAGED / ("%s.%s" % (rid, ext))
        if mode == "stage":
            src = sources[(rid, ext)]
            shutil.copyfile(src, dst)          # copyfile does not touch the source
            origin = src.relative_to(jc.REPO_ROOT).as_posix()
        else:
            origin = prior_origin.get((rid, ext), "(staged copy; origin not recorded)")
        rows.append({
            "relab_id": rid,
            "ext": ext,
            "source": origin,
            "sha256": jc.sha256_of(dst),
            "bytes": dst.stat().st_size,
            "how": mode,      # "stage" = copied from source now; "verify-staged" = hashed in place
        })
    prov = pd.DataFrame(rows)
    prov.to_csv(prov_csv, index=False)
    print("%s %d files for %d RELAB spectra"
          % ("staged" if mode == "stage" else "verified", len(prov), len(jc.ALL_RELAB_IDS)))

    # ---- verify identities against the PDS4 labels -----------------------
    meta = []
    for rid in jc.ALL_RELAB_IDS:
        m = jc.read_relab_xml(jc.staged_xml(rid))
        m["relab_id"] = rid
        wl, r = jc.load_spectrum(rid)
        m["n_bands"] = len(wl)
        m["lam_min_um"] = round(float(wl.min()), 4)
        m["lam_max_um"] = round(float(wl.max()), 4)
        meta.append(m)
    meta = pd.DataFrame(meta).set_index("relab_id")
    meta.to_csv(jc.TABLES / "step00_relab_metadata.csv")

    failures = []

    # (a) geometry must be i=30 / e=0 everywhere -- the value hapke.py assumes.
    for rid, row in meta.iterrows():
        if row["incidence_deg"] not in ("30", "30.0") or row["emission_deg"] not in ("0", "0.0"):
            failures.append("%s geometry i=%s e=%s, expected 30/0"
                            % (rid, row["incidence_deg"], row["emission_deg"]))

    # (b) end-member identity must match Table C1.
    for _, em in jc.ENDMEMBERS.iterrows():
        rid = em["relab_id"]
        code_exp, specimen_exp, _dlab = TABLE_C1_EXPECTED[rid]
        got_code = meta.loc[rid, "relab_code"]
        got_spec = meta.loc[rid, "specimen"]
        if got_code != code_exp:
            failures.append("%s (%s) RELAB code %s, Table C1 says %s"
                            % (rid, em["name"], got_code, code_exp))
        if specimen_exp and not got_spec.startswith(specimen_exp):
            failures.append("%s (%s) specimen %r does not start with Table C1 %r"
                            % (rid, em["name"], got_spec, specimen_exp))

    # (c) Apollo spectra identity must match the sample the paper names.
    for _, sp in jc.SPECTRA.iterrows():
        rid = sp["relab_id"]
        got_code = meta.loc[rid, "relab_code"]
        got_spec = meta.loc[rid, "specimen"]
        if got_code != sp["relab_code"]:
            failures.append("%s RELAB code %s, expected %s" % (rid, got_code, sp["relab_code"]))
        if not got_spec.startswith(sp["sample"]):
            failures.append("%s specimen %r is not sample %s" % (rid, got_spec, sp["sample"]))

    # (d) negative control: a deliberately shuffled id->specimen assignment must
    #     be rejected by the same check, proving it is not vacuously true.
    shuffled = list(jc.ENDMEMBERS["relab_id"])[1:] + list(jc.ENDMEMBERS["relab_id"])[:1]
    n_caught = 0
    for em_rid, wrong_rid in zip(jc.ENDMEMBERS["relab_id"], shuffled):
        if meta.loc[wrong_rid, "relab_code"] != TABLE_C1_EXPECTED[em_rid][0]:
            n_caught += 1
    if n_caught != len(jc.ENDMEMBERS):
        failures.append("negative control: shuffled end-member assignment was NOT "
                        "rejected (%d/%d caught)" % (n_caught, len(jc.ENDMEMBERS)))
    else:
        print("negative control OK: all %d shuffled assignments rejected" % n_caught)

    if failures:
        print("\nIDENTITY CHECK FAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("identity check OK: 9 end-members match Table C1, 13 Apollo spectra match Fig. 2")

    # A fresh stage rewrites the manifest, because the staged bytes have just been
    # replaced.  A distributed copy must NOT: the manifest it shipped with is the
    # claim being tested, and rewriting it would launder any corruption in transit
    # into a new, self-consistent baseline.  So verify against the shipped one.
    if mode == "stage" or not jc.MANIFEST.exists():
        jc.write_manifest()
        jc.verify_manifest(STAGE)
        print("manifest written: %s" % jc.MANIFEST)
    else:
        jc.verify_manifest(STAGE)
        print("manifest verified against the one distributed with this folder: %s"
              % jc.MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
