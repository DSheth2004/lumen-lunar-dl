"""
Run the whole reproduction in order, aborting on the first failing stage.

    python run_all.py              # everything
    python run_all.py --fast       # skip the long MCMC stages
    python run_all.py --from step03

Stages that download from the PDS archive (step00b) are skipped automatically
once their outputs are already staged, unless --refetch is given.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("step00", "step00_stage_data.py", [], "stage RELAB inputs, verify Table C1 identities"),
    ("step00b", "step00b_stage_m3.py", [], "stage the M3 Aristarchus ROIs"),
    ("step01", "step01_hapke_selftest.py", [], "Hapke core self-tests (Appendices C/D/E)"),
    ("step02", "step02_band_depth.py", [], "Step 1: band-depth correction, Table 1"),
    ("step03", "step03_endmember_optical_constants.py", [], "end-member optical constants, Figure C1"),
    ("step04", "step04_catmip_synthetic.py", [], "CATMIP synthetic recovery gate"),
    ("step05", "step05_unmix_lscc.py", [], "Step 2: LSCC unmixing, Figure 3"),
    ("step06", "step06_m3_aristarchus.py", [], "M3 at Aristarchus, Figures 5 and 6"),
    ("step07", "step07_ground_truth.py", [], "inverted mineralogy vs LSCC lab values, Figs 4/F1"),
    ("step08", "step08_figures.py", [], "redraw Figs 2, 3, 4, 5, 6, C1, F1"),
    ("step09", "step09_paper_figures.py", [], "reproduce Figs 1, 3, 5, 6, A1 in the paper's layout"),
    ("audit", "audit_paper_targets.py", [], "score everything against the published values"),
]

SLOW = {"step04", "step05", "step06"}


def main() -> int:
    # Children are launched with -u; without this the parent's own banners
    # are block-buffered when stdout is a file and all flush at the end,
    # which makes the log read as though the stages ran out of order.
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the long MCMC stages")
    ap.add_argument("--from", dest="start", default=None, help="start at this stage")
    ap.add_argument("--only", default=None, help="run only this stage")
    ap.add_argument("--refetch", action="store_true", help="re-download the M3 ROIs")
    args = ap.parse_args()

    m3_ready = (HERE.parent / "data" / "m3_aristarchus" / "roi_c_rfl_block.npy").exists()

    started = args.start is None
    total0 = time.time()
    ran = 0
    for name, script, extra, desc in STAGES:
        if args.only and name != args.only:
            continue
        if not started:
            if name == args.start:
                started = True
            else:
                continue
        if args.fast and name in SLOW:
            print("[run_all] SKIP %-8s (--fast)  %s" % (name, desc))
            continue
        if name == "step00b" and m3_ready and not args.refetch:
            print("[run_all] SKIP %-8s (already staged; --refetch to force)" % name)
            continue

        print("\n" + "#" * 78)
        print("# [run_all] %s  --  %s" % (name, desc))
        print("#" * 78)
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", str(HERE / script)] + extra, cwd=str(HERE))
        dt = time.time() - t0
        ran += 1
        if rc != 0:
            if name == "audit":
                # The audit gates on published values, so a non-zero exit here
                # means "a reproduction target does not match", not "a stage
                # crashed".  Every stage before it has already run.
                print("\n[run_all] %d stages ran; the audit reports gated targets "
                      "that do not reproduce." % (ran - 1))
                print("[run_all] see RESULTS.md -- this is a result, not a crash.")
                return rc
            print("\n[run_all] %s FAILED (exit %d) after %.0f s -- stopping." % (name, rc, dt))
            return rc
        print("[run_all] %s ok (%.0f s)" % (name, dt))

    print("\n[run_all] all stages completed in %.0f s" % (time.time() - total0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
