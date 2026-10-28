"""Check that every number in paper/MANUSCRIPT.md is traceable.

A number in the current manuscript passes if it appears in a reference version of the manuscript
(default: the last committed MANUSCRIPT.md, `git show HEAD:...`) or in a committed result table
(tables/*.csv, compared after rounding to the precision written in the text). Anything else is
listed and the script exits non-zero, so an edit cannot silently introduce or change a number.

Usage: python check_numbers.py [--ref GIT_REV]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NUM = re.compile(r"(?<![\w.])[−-]?\d+(?:[.,]\d+)*(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    text = re.sub(r"https?://\S+", " ", text)                  # URLs / DOIs
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)        # display maths
    text = text.split("## References")[0]                       # reference list is checked by build
    out = set()
    for m in NUM.finditer(text):
        s = m.group(0).replace("−", "-").replace(",", "")
        if s.lstrip("-").replace(".", "").isdigit():
            out.add(s.lstrip("-"))
    return out


def table_values() -> set[str]:
    vals = set()
    for p in (ROOT / "tables").glob("*.csv"):
        for tok in re.findall(r"-?\d+\.\d+|-?\d+", p.read_text(encoding="utf-8", errors="ignore")):
            try:
                x = abs(float(tok))
            except ValueError:
                continue
            vals.add(tok.lstrip("-"))
            for d in (0, 1, 2, 3):
                vals.add(("%." + str(d) + "f") % x)
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    args = ap.parse_args()
    cur = (HERE / "MANUSCRIPT.md").read_text(encoding="utf-8")
    ref = subprocess.run(["git", "show", "%s:LUMEN_HybridDL/paper/MANUSCRIPT.md" % args.ref],
                         cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8").stdout
    if not ref:
        print("ERROR: could not read the reference manuscript at %s" % args.ref)
        return 1
    known = numbers(ref) | table_values()
    new = sorted(numbers(cur) - known, key=lambda s: (len(s), s))
    print("numbers in manuscript: %d; traceable: %d" % (len(numbers(cur)), len(numbers(cur)) - len(new)))
    if new:
        print("UNTRACEABLE numbers (not in %s manuscript nor in tables/*.csv):" % args.ref)
        for s in new:
            ctx = re.search(r".{0,50}" + re.escape(s) + r".{0,30}", cur.replace("−", "-"))
            print("  %-8s %s" % (s, ctx.group(0).replace("\n", " ") if ctx else ""))
        return 1
    print("all numbers traceable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
