"""Minimal corrections to v2 sentences whose meaning contradicted the results.

Each (paragraph, old, new) pair must match exactly once, in manuscript_src/*.md and in
v2/UPDATED_restored.txt, or nothing is written. Prints the exact number of words changed.
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TXT = HERE / "UPDATED_restored.txt"
MDS = sorted((HERE.parent / "manuscript_src").glob("*.md"))

FIX = [
    ("P002", "The ranges between similarities are not very large,", "Conformal intervals reach nominal coverage,"),
    ("P005", "gate to with skillful outcomes", "gate to withhold unskillful outcomes"),
    ("P005", "Periodictures reach", "Intervals reach"),
    ("P005", "(3.58 wt%) or worse than", "(3.58 wt%) and significantly worse than"),
    ("P006", "is a trick without solid foundation", "is an inverse problem with little ground truth"),
    ("P007", "The method I build on", "The method we build on"),
    ("P011", "any target with apparent skill", "any target without apparent skill"),
    ("P012", "are reported as positive", "are reported alongside positive"),
    ("P013", "we show that without orbital labels, a small orbital fine-tuning set recovers, and",
     "we show that laboratory accuracy does not transfer without orbital labels, a small orbital fine-tuning set recovers part of it, and"),
    ("P015", "is contigated in", "is convolved in"),
    ("P015", "channels smaller than 2.5 µm", "channels beyond 2.5 µm"),
    ("P051", "inner 5-fold stratified cross-validation", "inner 5-fold grouped cross-validation"),
    ("P051", "the A2 performs worse than A4 on 12/13 targets but is not olivine",
     "the A2 performs better on 12/13 targets but not on olivine"),
    ("P056", "does not produce LSCC soils", "does not see LSCC soils"),
    ("P056", "for LUMEN without training on A0 (only physics)", "for LUMEN A0 (trained only on physics)"),
    ("P058", "arranged in descending order according to the amount of crystalline phases",
     "scored on the proportions of crystalline phases"),
    ("P058", "the LOSS error", "the LOSO error"),
    ("P058", "the inverted maturity index results in 17.56", "the inversion results in 17.56"),
    ("P062", "cross validation within each site of", "cross validation over the sites of"),
    ("P062", "spatial autocorrelation.", "spatial autocorrelation (Table 6)."),
    ("P065", "significantly differs from no-FD values and the two", "does not differ significantly from the FD values or the two"),
    ("P071", "inferior to the LCP design under LOSO, but significantly superior to it on",
     "inferior for LCP under LOSO, but significantly superior on"),
    ("P072", "target: thirteen sites,", "thirteen targets,"),
    ("P076", "an unseen target comprising four channels", "an unseen mission on four targets"),
    ("P076", "random forest, making configuration selection unnecessary.",
     "random forest, independent of post-hoc configuration selection."),
    ("P078", "the maximum decrease is", "the maximum increase is"),
    ("P078", "model has the biggest percentage increase", "model has a percentage increase"),
    ("P081", " Where olivine is plentiful, detection isn’t an issue.", ""),
]

txt = TXT.read_text(encoding="utf-8")
mds = {p: p.read_text(encoding="utf-8") for p in MDS}
errors, added, removed = [], 0, 0
for pid, old, new in FIX:
    if txt.count(old) != 1:
        errors.append("%s: %r found %d times in the text file" % (pid, old, txt.count(old)))
    hits = [p for p in MDS if old in mds[p]]
    if len(hits) != 1 or mds[hits[0]].count(old) != 1:
        errors.append("%s: %r not found exactly once in manuscript_src" % (pid, old))
        continue
    txt = txt.replace(old, new)
    mds[hits[0]] = mds[hits[0]].replace(old, new)
    a, b = old.split(), new.split()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag != "equal":
            removed += i2 - i1
            added += j2 - j1
if errors:
    raise SystemExit("nothing written:\n  " + "\n  ".join(errors))
TXT.write_text(txt, encoding="utf-8", newline="\n")
for p, s in mds.items():
    p.write_text(s, encoding="utf-8", newline="\n")
print("%d fixes in %d paragraphs: %d words removed, %d words added"
      % (len(FIX), len({f[0] for f in FIX}), removed, added))
