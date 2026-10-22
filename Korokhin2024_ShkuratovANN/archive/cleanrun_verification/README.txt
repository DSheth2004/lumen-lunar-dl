Clean end-to-end verification, 2026-09-21.

`python scripts/run_all.py` was run in a separate folder holding only scripts/ and archive/,
from empty data/, tables/, figures/ and products/ (inputs re-staged,
M3 lines fetched again by step04). All 15 stages completed; the audit exited 1 because gated targets fail (a result,
not a crash). Total time about 4.3 h.

Against the main folder: all 156 audit values are identical, and every table in tables/
is identical except its run-time columns (seconds, epoch times, the C3 timing).

run_all_log.txt           console log of that run
audit_paper_targets.csv   its audit table (written before the NEAR MISS status was added)
