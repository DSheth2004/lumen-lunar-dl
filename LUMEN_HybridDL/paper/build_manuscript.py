"""Assemble paper/MANUSCRIPT.md from paper/manuscript_src/*.md.

Citations are written as [@key] or [@key1; @key2] in the sources. They are numbered in order of
first appearance, as MDPI requires, and the reference list is appended in that order from
manuscript_src/references.tsv (key<TAB>formatted entry; metadata checked against Crossref).
Unknown keys and unused references are reported; an unknown key is an error.
Also reports every [[PENDING-...]] placeholder still in the text.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "manuscript_src"
OUT = HERE / "MANUSCRIPT.md"
CITE = re.compile(r"\[(@[\w]+(?:;\s*@[\w]+)*)\]")


def compress(nums):
    """[3, 4, 5, 9] -> '3–5,9' (MDPI style)."""
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(str(nums[i]) if j == i else ("%d,%d" % (nums[i], nums[j]) if j == i + 1
                                                    else "%d–%d" % (nums[i], nums[j])))
        i = j + 1
    return ",".join(parts)


def main() -> int:
    refs = {}
    for line in (SRC / "references.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            k, entry = line.split("\t", 1)
            refs[k] = entry
    text = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in sorted(SRC.glob("[0-9][0-9]_*.md")))
    order: list[str] = []
    unknown = set()

    def number(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in refs:
                unknown.add(k)
                continue
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        return "[%s]" % compress(nums)

    body = CITE.sub(number, text)
    if unknown:
        print("ERROR: unknown citation keys:", ", ".join(sorted(unknown)))
        return 1
    ref_list = "\n".join("%d. %s" % (i, refs[k]) for i, k in enumerate(order, 1))
    OUT.write_text(body + "\n\n## References\n\n" + ref_list + "\n", encoding="utf-8", newline="\n")
    unused = [k for k in refs if k not in order]
    pending = re.findall(r"\[\[(PENDING-[^:\]]+)", body)
    words = len(re.sub(r"[#*|`$\\]", " ", body).split())
    print("MANUSCRIPT.md: %d words, %d references cited" % (words, len(order)))
    if unused:
        print("unused references:", ", ".join(unused))
    print("pending placeholders: %d (%s)" % (len(pending), ", ".join(sorted(set(pending))) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
