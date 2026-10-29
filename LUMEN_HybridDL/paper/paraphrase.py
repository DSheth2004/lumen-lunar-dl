"""Round-trip the manuscript prose through a plain-text file for rewriting by hand.

    python paraphrase.py export   -> paper/PARAPHRASE.txt (+ paper/paraphrase_map.json)
    python paraphrase.py import [file] [--accept-math]   -> writes the edited paragraphs (default PARAPHRASE.txt)
                                            back into manuscript_src/*.md

Only prose is exported: highlights, abstract, body paragraphs, list items, figure and table
captions and the filled back-matter statements. Headings appear as context lines. Equations, tables,
figure links, keywords and the title are not exported and stay as they are.

Citations appear as [1], [2, 5] etc., numbered in order of first appearance (the same numbers as
the PDF). They are turned back into [@key] on import, so they can be moved, dropped or reused.
Inline maths written as $...$ is exported as is and must be left unchanged.

After an import, rebuild with build_manuscript.py, check_numbers.py and the other build scripts.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "manuscript_src"
TXT = HERE / "PARAPHRASE.txt"
MAP = HERE / "paraphrase_map.json"
CITE = re.compile(r"\[(@[\w]+(?:;\s*@[\w]+)*)\]")
NUMCITE = re.compile(r"\[(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\]")
MARK = re.compile(r"^<<(P\d{3})>>\s*$")
# optional list bullet, optional bold label ("**Figure 1.**", "**Funding:**"), then the prose
LINE = re.compile(r"^((?:- )?(?:\*\*[^*]+\*\*\s*)?)(.*)$")

HEADER = """LUMEN manuscript: text for rewriting
=====================================

Rewrite the text under each <<P...>> marker in your own words. Rules:
  * Keep every <<P...>> marker line exactly as it is. Lines starting with === are section
    headings, shown for orientation only; edits to them are ignored.
  * A paragraph may run over several lines; line breaks inside it are joined into one paragraph.
  * Citations are [1], [2, 5] or [3-5], the same numbers as in the PDF. Keep, move or drop them,
    but only use numbers that already appear in this file.
  * Leave anything inside $...$ (maths) exactly as it is.
  * Keep the numbers and the claims; check_numbers.py will flag any number that no longer
    matches the results.
  * Equations, tables, figures, the title and the keywords are not in this file and stay as they are.
"""


def src_files():
    return sorted(SRC.glob("*.md"))


def export() -> int:
    order: list[str] = []
    for f in src_files():
        for m in CITE.finditer(f.read_text(encoding="utf-8")):
            for k in re.findall(r"@([\w]+)", m.group(1)):
                if k not in order:
                    order.append(k)
    num = {k: i + 1 for i, k in enumerate(order)}

    out, blocks, n = [HEADER], {}, 0
    for f in src_files():
        for line in f.read_text(encoding="utf-8").split("\n"):
            s = line.strip()
            if s.startswith("#"):
                out.append("\n=== %s ===\n" % s.lstrip("#").strip())
                continue
            if (not s or s.startswith(("|", "![", "$$", "**Keywords:**"))
                    or re.fullmatch(r"\*\*[^*]+\*\*", s)):
                continue
            prefix, body = LINE.match(line).groups()
            if not body.strip():
                continue
            n += 1
            pid = "P%03d" % n
            blocks[pid] = {"file": f.name, "line": line, "prefix": prefix}
            text = CITE.sub(lambda m: "[%s]" % ", ".join(
                str(num[k]) for k in re.findall(r"@([\w]+)", m.group(1))), body)
            out.append("<<%s>>\n%s\n" % (pid, text))
    TXT.write_text("\n".join(out), encoding="utf-8", newline="\n")
    MAP.write_text(json.dumps({"citations": order, "blocks": blocks}, ensure_ascii=False, indent=1),
                   encoding="utf-8", newline="\n")
    print("wrote %s: %d paragraphs, %d citation numbers" % (TXT.name, n, len(order)))
    return 0


def import_(txt: Path = TXT, check_math: bool = True) -> int:
    mp = json.loads(MAP.read_text(encoding="utf-8"))
    order, blocks = mp["citations"], mp["blocks"]

    new, cur, buf = {}, None, []
    for line in txt.read_text(encoding="utf-8").replace("\r", "").split("\n") + ["=== end ==="]:
        m = MARK.match(line.strip())
        if m or line.startswith("==="):
            if cur:
                new[cur] = " ".join(x.strip() for x in buf if x.strip())
            cur, buf = (m.group(1) if m else None), []
        elif cur:
            buf.append(line)

    errors = []
    missing = sorted(set(blocks) - set(new))
    unknown = sorted(set(new) - set(blocks))
    if missing:
        errors.append("markers missing from the text file: %s" % ", ".join(missing))
    if unknown:
        errors.append("unknown markers: %s" % ", ".join(unknown))

    def to_keys(pid, m):
        keys = []
        for part in re.split(r"\s*,\s*", m.group(1)):
            ends = re.split(r"\s*[-–]\s*", part)
            for i in range(int(ends[0]), int(ends[-1]) + 1):
                if not 1 <= i <= len(order):
                    errors.append("%s: citation [%d] does not exist" % (pid, i))
                    return m.group(0)
                keys.append(order[i - 1])
        return "[%s]" % "; ".join("@" + k for k in dict.fromkeys(keys))

    texts = {f.name: f.read_text(encoding="utf-8").split("\n") for f in src_files()}
    changed = 0
    for pid, b in blocks.items():
        if pid not in new:
            continue
        if not new[pid]:
            errors.append("%s is empty" % pid)
            continue
        old_math = re.findall(r"\$[^$]+\$", b["line"])
        body = NUMCITE.sub(lambda m: to_keys(pid, m), new[pid])
        if check_math and sorted(re.findall(r"\$[^$]+\$", body)) != sorted(old_math):
            errors.append("%s: the $...$ maths was changed" % pid)
            continue
        line = b["prefix"] + body
        if line == b["line"]:
            continue
        lines = texts[b["file"]]
        if lines.count(b["line"]) != 1:
            errors.append("%s: the original paragraph is no longer in %s (sources changed since export?)"
                          % (pid, b["file"]))
            continue
        lines[lines.index(b["line"])] = line
        b["line"] = line  # so the text file can be edited further and imported again
        changed += 1

    if errors:
        print("nothing written:\n  " + "\n  ".join(errors))
        return 1
    for f in src_files():
        f.write_text("\n".join(texts[f.name]), encoding="utf-8", newline="\n")
    MAP.write_text(json.dumps(mp, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("updated %d of %d paragraphs in manuscript_src/; now run build_manuscript.py and check_numbers.py"
          % (changed, len(blocks)))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        sys.exit(export())
    if cmd == "import":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        sys.exit(import_(HERE / args[0] if args else TXT, "--accept-math" not in sys.argv))
    print(__doc__)
    sys.exit(2)
