"""
Ordered driver. Aborts on the first failing stage.

A non-zero exit from `audit` means a reproduction target does not match, not that
a stage crashed -- the two are reported differently.

  --fast     skip the slow stages (network downloads, 5-seed ANN, controls);
             the audit then scores whatever tables already exist
  --from X   start at stage X;  --only X   run stage X alone
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("step00", "step00_stage_inputs.py", [], "stage RELAB, LSCC, ground truth, Fe n,k; seal data/"),
    ("step00b", "step00b_paper_tables.py", [], "transcribe the paper's tables; check their arithmetic"),
    ("step00c", "step00c_digitize.py", [], "digitize Figs. 3, 12, A1 from the PDF"),
    ("step01", "step01_model_selftests.py", [], "Shkuratov model self-tests; Table 5 tau_Gr; alpha_Fe"),
    ("step02", "step02_lab_mixtures.py", [], "Table 1: three RELAB mixtures"),
    ("step03", "step03_basis_set.py", [], "basis set, end-member identities, Table 7"),
    ("step04", "step04_m3_scene.py", [], "recover the M3 scene and sites from Fig. 3; stage M3"),
    ("step05", "step05_m3_preprocess.py", [], "M3 -> 21-point RELAB-system spectra; Eq. 2"),
    ("step06", "step06_m3_site_fits.py", [], "Nelder-Mead at the sites: Tables 5, 6, 8, 9, 10"),
    ("step07", "step07_ann.py", [], "ANN, three iterations x 5 seeds: Tables 12-15"),
    ("step09", "step09_maps_comparisons.py", [], "maps, histograms, MLR / Lucey comparisons"),
    ("step08", "step08_controls.py", [], "controls: recovery, range dependence, Nelder-Mead"),
    ("step10", "step10_lscc_validation.py", [], "LSCC mare + highland soils: retrieved vs measured"),
    ("targets", "make_targets.py", [], "write paper_targets.csv"),
    ("audit", "audit_paper_targets.py", [], "score everything against the published values"),
]

SLOW = {"step04", "step07", "step08"}

# Stages that read sources outside this folder. Their outputs (data/, reference/) are kept
# in the folder, so without those sources the stages are skipped and the rest runs as is.
import korokhin_common as kc  # noqa: E402
EXTERNAL = {
    "step00": ([kc.RELAB_RAW, kc.MARE_GROUND_TRUTH,
                kc.REPO_ROOT / "RELAB_LSCC_2day_pilot" / "01_native_reflectance" / "lscc"], kc.MANIFEST),
    "step00c": ([kc.PAPER_PDF], kc.DIGITIZED),
}


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the slow stages")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    started = args.start is None
    ran = 0
    t0 = time.time()
    for name, script, extra, desc in STAGES:
        if args.only and name != args.only:
            continue
        if not started:
            if name == args.start:
                started = True
            else:
                continue
        if name in EXTERNAL:
            sources, output = EXTERNAL[name]
            missing = [s for s in sources if not s.exists()]
            if missing and output.exists():
                print("[run_all] SKIP %-8s (source not present: %s; using the staged copy)"
                      % (name, missing[0]))
                continue
        if args.fast and name in SLOW:
            print("[run_all] SKIP %-8s (--fast)  %s" % (name, desc))
            continue
        print("\n" + "#" * 78)
        print("# [run_all] %s  --  %s" % (name, desc))
        print("#" * 78)
        s = time.time()
        rc = subprocess.call([sys.executable, "-u", str(HERE / script)] + extra, cwd=str(HERE))
        dt = time.time() - s
        ran += 1
        if rc != 0:
            if name == "audit":
                print("\n[run_all] %d stages ran; the audit reports gated targets that do not "
                      "reproduce." % (ran - 1))
                print("[run_all] see RESULTS.md -- this is a result, not a crash.")
                return rc
            print("\n[run_all] %s FAILED (exit %d) after %.0f s -- stopping." % (name, rc, dt))
            return rc
        print("[run_all] %s ok (%.0f s)" % (name, dt))
    print("\n[run_all] all %d stages completed in %.0f s" % (ran, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
