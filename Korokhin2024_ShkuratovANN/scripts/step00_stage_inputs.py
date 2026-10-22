"""
Step 00 -- stage every input this reproduction reads into data/, then seal it.

Sources (all opened read-only):
  * RELAB PDS4 spectra from RMS/RELAB/cartOrder*/ (downloaded earlier from
    pds-speclib.rsl.wustl.edu). One .tab + .xml per spectrum.
  * LSCC Kdata (Pieters/Taylor consortium spectra, 19 soils x 4 size fractions)
    from RMS/RELAB_LSCC_2day_pilot/01_native_reflectance/lscc/.
  * LSCC mare ground truth (modal mineralogy, Is/FeO, bulk chemistry; Taylor et al.
    2001), transcribed and checked in the companion reproduction
    ../Jung2026_SpaceWeatheredMaria/; LSCC highland ground truth (Taylor et al. 2010)
    from lscc_highland_taylor2010.py. Sources in data/ground_truth/README.md.
  * Metallic-iron optical constants from the refractiveindex.info database (CC0).

Identity check. Every staged RELAB spectrum's label is compared POSITIONALLY with
the identity the paper prints for it (RELAB sample ID; sample number; size range).
A deliberately shuffled copy of the expected list must FAIL the same check,
otherwise the check could not detect a mix-up (repository convention).
"""
from __future__ import annotations

import shutil
import sys
import urllib.request

import numpy as np
import pandas as pd

import korokhin_common as kc

STAGE = "step00"

# (spectrum, expected RELAB sample ID, expected text in specimen name, max size um,
#  role, source in the paper; max size None where the label carries none). Spectrum IDs are the paper's where it prints them
#  (Tables 1-3); for Table 4 they were resolved from the printed sample IDs.
EXPECTED = [
    # Table 4 -- the basis set used for M3
    ("c1lr215", "LR-CMP-215", "70017", 45, "basis CPX (deep-brown pyroxene)", "Table 4"),
    ("c1lr220", "LR-CMP-220", "70035", 45, "basis OPX (light-brown pyroxene)", "Table 4"),
    ("c1lr212", "LR-CMP-212", "15555", 45, "basis OLV", "Table 4"),
    ("c1lr217", "LR-CMP-217", "70017", 45, "basis PLG", "Table 4"),
    ("c1lr218", "LR-CMP-218", "70017", 45, "basis ILM", "Table 4"),
    # Table 1 / Figs. 4-6 -- man-made mixtures and their end-members
    ("c1ms16", "MS-CMP-016", "", None, "mixture 1: OLV+PYX 50:50", "Table 1, Fig. 4"),
    ("c1ms14", "MS-CMP-014", "", 75, "mixture 1 end-member OLV", "Fig. 4"),
    ("c1ms15", "MS-CMP-015", "", 75, "mixture 1 end-member PYX", "Fig. 4"),
    ("c1xt36", "XT-CMP-036", "", 75, "mixture 2: OLV+OPX+ANO 33:33:33", "Table 1, Fig. 5"),
    ("c2po81", "PO-CMP-081-C", "", 75, "mixture 2 end-member OLV", "Fig. 5"),
    ("c1pe41", "PE-CMP-041-C", "", 75, "mixture 2 end-member OPX", "Fig. 5"),
    ("c1pa60", "PA-CMP-060-C", "", 75, "mixture 2 end-member ANO", "Fig. 5"),
    ("c1sc07", "SC-EAC-007", "", 90, "mixture 3: OLV+PYX+MET 40:30:30", "Table 1, Fig. 6"),
    ("c1sc02", "SC-EAC-002", "", 90, "mixture 3 end-member OLV", "Fig. 6"),
    ("c1sc09", "SC-EAC-009", "", 90, "mixture 3 end-member low-Ca PYX", "Fig. 6"),
    ("c1sc06", "SC-EAC-006", "", 90, "mixture 3 end-member Odessa metal", "Fig. 6"),
    # Table 2 / Fig. 7 -- Cloutis & Gaffey (1993) set (CASC02 not available locally)
    ("c1sc37", "SC-EAC-037", "", 90, "C&G set PLG", "Table 2, Fig. 7"),
    ("c1sc36", "SC-EAC-036", "", 90, "C&G set OPX", "Table 2, Fig. 7"),
    ("c1sc35", "SC-EAC-035", "", 90, "C&G set CPX", "Table 2, Fig. 7"),
    ("c1sc34", "SC-EAC-034", "", 90, "C&G set ILM", "Table 2, Fig. 7"),
    # Table 3 / Figs. 8, 10 -- Noble et al. (2006) set (4 of 7 not available locally)
    ("c2ls45", "LS-CMP-045", "10084", 1000, "Noble set AGG; Table 7 / Table 8 agglutinate", "Table 3, 7, 8"),
    ("c1sb06", "SB-RGB-006", "", None, "Noble set OPX", "Table 3, Fig. 8"),
    ("c1sw12", "SW-CMP-012", "", 500, "Noble set PLG", "Table 3, Fig. 8"),
]
NOT_AVAILABLE = [("casc02", "Table 2 OLV"), ("c2ls09", "Table 3 CPX"), ("cals35", "Table 3 GLS"),
                 ("cjb308", "Table 3 ILM"), ("c2po26", "Table 3 OLV")]

LSCC_KDATA = kc.REPO_ROOT / "RELAB_LSCC_2day_pilot" / "01_native_reflectance" / "lscc"
FE_URLS = {
    "Fe_JohnsonChristy1974.yml": "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Fe/nk/Johnson.yml",
    "Fe_Ordal1988.yml": "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/Fe/nk/Ordal.yml",
}


def find_source(spectrum_id: str):
    for tab in sorted(kc.RELAB_RAW.glob("cartOrder*/**/%s.tab" % spectrum_id)):
        xml = tab.with_suffix(".xml")
        if xml.exists():
            return tab, xml
    return None, None


def identity_check(expected: list, labels: dict) -> pd.DataFrame:
    rows = []
    for spec, rid, sample, max_um, role, src in expected:
        lab = labels.get(spec, {})
        ok_id = lab.get("relab_id", "") == rid
        ok_sample = (sample == "") or lab.get("specimen", "").startswith(sample)
        try:
            got_max = float(lab.get("size_max_um", "nan"))
        except ValueError:
            got_max = np.nan
        ok_size = (max_um is None) or (np.isfinite(got_max) and got_max == max_um)
        rows.append({"spectrum": spec, "role": role, "paper_source": src,
                     "expected_relab_id": rid, "label_relab_id": lab.get("relab_id", ""),
                     "expected_sample": sample, "label_specimen": lab.get("specimen", ""),
                     "expected_max_um": max_um, "label_max_um": got_max,
                     "id_ok": ok_id, "sample_ok": ok_sample, "size_ok": ok_size,
                     "all_ok": ok_id and ok_sample and ok_size})
    return pd.DataFrame(rows)


GROUND_TRUTH_README = """Measured composition of the Lunar Soil Characterization Consortium (LSCC) soils.

Mare soils (9): lscc_modal_mineralogy.csv, lscc_is_feo.csv, lscc_bulk_chemistry.csv.
  Source: LSCC data Tables 1-3 (modal mineralogy, pyroxene sub-sets, bulk chemistry and
  Is/FeO), Planetary Geosciences Institute, University of Tennessee,
  http://pgi.utk.edu/wp-content/uploads/2016/04/Table{1,2,3}.pdf (retrieved through the
  Internet Archive), for Taylor, Pieters, Keller, Morris & McKay (2001), JGR 106(E11),
  27985-27999. Transcribed by hand; 81 checks against the tables' own column sums and
  totals. Modal abundances are point counts, i.e. by volume.

Highland soils (10, Apollo 14 and 16): lscc_highland_*.csv.
  Source: Taylor et al. (2010), JGR 115, E02002 (NASA NTRS preprint 20100006633), Tables
  2-4, transcribed in scripts/lscc_highland_taylor2010.py. Checked against the tables' own
  arithmetic and, for Al2O3/CaO/FeO/MgO, against Shkuratov et al. (2005a) Table 1
  (tables/step00_highland_checks.csv).
"""


def main() -> int:
    kc.banner(STAGE, "stage RELAB, LSCC, ground truth and Fe optical constants")
    for d in (kc.RELAB_DIR, kc.GROUND_TRUTH, kc.OPTICAL, kc.DATA / "lscc"):
        d.mkdir(parents=True, exist_ok=True)

    # ---- RELAB spectra ---------------------------------------------------
    labels, prov = {}, []
    for spec, *_ in EXPECTED:
        tab, xml = find_source(spec)
        if tab is None:
            sys.exit("[%s] FATAL: %s not found under %s" % (STAGE, spec, kc.RELAB_RAW))
        shutil.copy2(tab, kc.RELAB_DIR / tab.name)
        shutil.copy2(xml, kc.RELAB_DIR / xml.name)
        labels[spec] = kc.read_relab_xml(kc.RELAB_DIR / xml.name)
        prov.append({"spectrum": spec, "source": str(tab.relative_to(kc.REPO_ROOT)),
                     "sha256": kc.sha256_of(tab)})
    pd.DataFrame(prov).to_csv(kc.RELAB_DIR / "provenance.csv", index=False)
    for spec, why in NOT_AVAILABLE:
        tab, _ = find_source(spec)
        print("  %-7s %s -> %s" % (spec, why, "FOUND (unexpected)" if tab else "not available locally"))

    chk = identity_check(EXPECTED, labels)
    chk.to_csv(kc.TABLES / "step00_identity_check.csv", index=False)
    print("  identity check: %d / %d spectra match the paper's IDs, samples and sizes"
          % (int(chk["all_ok"].sum()), len(chk)))
    if not chk["all_ok"].all():
        print(chk.loc[~chk["all_ok"]].to_string())
        sys.exit("[%s] FATAL: staged spectra do not match the paper's identities" % STAGE)

    # negative control: shuffle which label is paired with which expectation
    rng = np.random.default_rng(kc.SEED)
    perm = rng.permutation(len(EXPECTED))
    shuffled = [(EXPECTED[i][0],) + tuple(EXPECTED[j][1:]) for i, j in enumerate(perm)]
    neg = identity_check(shuffled, labels)
    n_moved = int((perm != np.arange(len(perm))).sum())
    print("  shuffled negative control: %d of %d rows moved, %d of them rejected"
          % (n_moved, len(perm), int((~neg["all_ok"]).sum())))
    if int((~neg["all_ok"]).sum()) < n_moved:
        sys.exit("[%s] FATAL: the identity check cannot detect a shuffled pairing" % STAGE)
    pd.DataFrame({"n_rows": [len(perm)], "n_moved": [n_moved],
                  "n_rejected": [int((~neg["all_ok"]).sum())]}).to_csv(
        kc.TABLES / "step00_identity_negative_control.csv", index=False)

    # ---- LSCC Kdata ------------------------------------------------------
    n_k = 0
    for f in sorted(LSCC_KDATA.glob("*Kdata.txt")):
        shutil.copy2(f, kc.DATA / "lscc" / f.name)
        n_k += 1
    print("  LSCC Kdata: %d soils staged" % n_k)

    # ---- LSCC mare ground truth (Taylor et al. 2001) ---------------------
    for f in ("lscc_modal_mineralogy.csv", "lscc_is_feo.csv", "lscc_bulk_chemistry.csv"):
        shutil.copy2(kc.MARE_GROUND_TRUTH / f, kc.GROUND_TRUTH / f)
    (kc.GROUND_TRUTH / "README.md").write_text(GROUND_TRUTH_README, encoding="utf-8")

    # ---- LSCC highland ground truth (Taylor et al. 2010) -----------------
    import lscc_highland_taylor2010 as hl
    arith = hl.check()
    indep, indep_s = hl.independent_check(kc.SEED)
    pyx, pyx_s = hl.pyroxene_crosscheck(kc.SEED)
    print("  highland LSCC (Taylor 2010): %d arithmetic checks pass; Al2O3/CaO/FeO/MgO equal to"
          " Shkuratov 2005a Table 1 in %d/%d rows (shuffled control: %d rows agree, %d of %d unmoved);"
          " Table 2 vs Table 3 pyroxene agree in %d/%d (shuffled: %d)"
          % (len(arith), indep_s["agree_real"], indep_s["n"], indep_s["agree_shuffled"],
             indep_s["n"] - indep_s["n_moved"], indep_s["n"],
             pyx_s["agree_real"], pyx_s["n"], pyx_s["agree_shuffled"]))
    if indep_s["agree_real"] < indep_s["n"] or indep_s["agree_shuffled"] >= indep_s["agree_real"]:
        sys.exit("[%s] FATAL: highland transcription disagrees with Shkuratov 2005a" % STAGE)
    pd.concat([arith.assign(kind="arithmetic"),
               indep.assign(kind="vs Shkuratov 2005a Table 1"),
               pyx.assign(kind="Table 2 pyroxene vs Table 3 total (source's own)")]).to_csv(
        kc.TABLES / "step00_highland_checks.csv", index=False)
    hl.modal_long().to_csv(kc.GROUND_TRUTH / "lscc_highland_modal_mineralogy.csv", index=False)
    chem = hl.chem_wide()
    chem.drop(columns=["is_feo"]).to_csv(kc.GROUND_TRUTH / "lscc_highland_bulk_chemistry.csv", index=False)
    chem[["sample", "size_fraction_um", "is_feo"]].assign(source_table="Table 4").to_csv(
        kc.GROUND_TRUTH / "lscc_highland_is_feo.csv", index=False)

    # ---- Fe optical constants -------------------------------------------
    for name, url in FE_URLS.items():
        dst = kc.OPTICAL / name
        if not dst.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            dst.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    (kc.OPTICAL / "README.md").write_text(
        "Metallic iron n,k from the refractiveindex.info database (public domain, CC0):\n"
        "- Fe_JohnsonChristy1974.yml -- Johnson & Christy (1974) Phys. Rev. B 9, 5056; "
        "0.188-1.937 um. Cited by the paper.\n"
        "- Fe_Ordal1988.yml -- Ordal et al. (1988) Appl. Opt. 27, 1203; 0.667-286 um. Stands in\n"
        "  for Weaver et al. (1979), which the paper used beyond 1.94 um but which is not openly\n"
        "  available (deviation D3).\n", encoding="utf-8")

    man = kc.write_manifest()
    kc.verify_manifest(STAGE)
    print("  manifest: %d files sealed" % len(man))
    kc.verify_manifest(STAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
