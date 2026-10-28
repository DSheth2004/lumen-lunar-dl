"""Write fig_evaluation.tex (Figure 4: evaluation design) from the real group sizes.

Laboratory: 19 LSCC soils grouped by mission (A11 1, A12 2, A14 4, A15 2, A16 6, A17 4; A14 and A16
are highland). Orbital: the 49 Qiu et al. (2025) sites grouped by landing site, as staged in
data/m3_sites_qiu/sites.csv. Each row shows one example fold of a protocol.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SITES = HERE.parent.parent / "data" / "m3_sites_qiu" / "sites.csv"

LAB = [("A11", 1, "mare"), ("A12", 2, "mare"), ("A14", 4, "highland"), ("A15", 2, "mare"),
       ("A16", 6, "highland"), ("A17", 4, "mare")]
ORBIT_ORDER = ["A11", "A12", "A14", "A15", "A16", "A17", "L16", "L20", "L24", "CE3", "CE5", "CE6"]
INNER = ["phyE!25", "phyE!45", "phyE!65", "phyE!85", "phyE"]
TRAIN = "draw=lrnE, fill=lrnF, line width=0.5pt"
TEST = "draw=outE, fill=outE!70, line width=0.5pt"
IDLE = "draw=grey, fill=white, line width=0.5pt"
X0 = 3.6                                                  # left column for row titles and notes
BS = "\\\\"                                               # TeX line break


def positions(groups, step, gap):
    xs, x = [], 0.0
    for g in groups:
        n = g[1]
        xs.append([x + i * step for i in range(n)])
        x += n * step + gap
    return xs


def dot(x, y, style, d):
    return r"\node[circle, minimum size=%.2fcm, inner sep=0pt, %s] at (%.3f,%.3f) {};" % (d, style, x, y)


def row_title(y, name, desc):
    return [r"\node[anchor=west, font=\footnotesize\bfseries] at (0,%.2f) {%s};" % (y + 0.1, name),
            r"\node[anchor=west, note, align=left] at (0,%.2f) {%s};" % (y - 0.2, desc)]


def main():
    out = []

    # ---------------- (a) laboratory
    lab_xs = positions(LAB, 0.42, 0.42)
    out.append(r"\node[title, anchor=west] at (0,0.75) {(a) Laboratory: 19 LSCC soils};")
    rows = [("LOSO", "19 folds, one soil out"), ("LOMO", "6 folds, one mission out"),
            ("E1 (nested)", "inner 5-fold by soil")]
    for r, (name, desc) in enumerate(rows):
        y = -0.75 * r
        out += row_title(y, name, desc)
        k = 0
        for g, (gname, n, kind) in enumerate(LAB):
            for i, x in enumerate(lab_xs[g]):
                if r == 0:
                    st = TEST if (gname == "A15" and i == 0) else TRAIN
                elif r == 1:
                    st = TEST if gname == "A16" else TRAIN
                elif gname == "A15" and i == 0:
                    st = IDLE
                else:
                    st = "draw=phyE, fill=%s, line width=0.5pt" % INNER[k % 5]
                    k += 1
                out.append(dot(X0 + x, y, st, 0.3))
    for g, (gname, n, kind) in enumerate(LAB):
        cx = X0 + (lab_xs[g][0] + lab_xs[g][-1]) / 2
        out.append(r"\node[note, text=ink] at (%.3f,%.2f) {%s};" % (cx, -1.92, gname))
        out.append(r"\node[note] at (%.3f,%.2f) {%s};" % (cx, -2.2, "highland" if kind == "highland" else "mare"))
    out.append(r"\node[note, anchor=west, align=left] at (%.2f,-1.5) {%s};" % (
        X0 + 10.0, BS.join(["In each outer training set, an inner", "5-fold split by soil chooses A2 or A4;",
                            "the outer test soil (white) is never seen."])))

    # ---------------- (b) orbital
    sites = pd.read_csv(SITES)
    counts = sites["mission"].value_counts()
    orb = [(m, int(counts[m])) for m in ORBIT_ORDER]
    assert sum(n for _, n in orb) == 49, orb
    orb_xs = positions(orb, 0.255, 0.30)
    yb = -3.55
    out.append(r"\node[title, anchor=west] at (0,%.2f) {(b) Orbital FeO: 49 returned-sample sites};" % (yb + 0.75))
    p1_test = {("A15", 3), ("A16", 7), ("A17", 2), ("A17", 13), ("L24", 0)}     # one example random fold
    for r, (name, desc) in enumerate([("P1", "10 random folds over sites"), ("P2", "12 folds, one landing site out")]):
        y = yb - 0.75 * r
        out += row_title(y, name, desc)
        for g, (gname, n) in enumerate(orb):
            for i, x in enumerate(orb_xs[g]):
                st = TEST if ((r == 0 and (gname, i) in p1_test) or (r == 1 and gname == "A17")) else TRAIN
                out.append(dot(X0 + x, y, st, 0.2))
    for g, (gname, n) in enumerate(orb):
        cx = X0 + (orb_xs[g][0] + orb_xs[g][-1]) / 2
        out.append(r"\node[note, text=ink] at (%.3f,%.2f) {%s};" % (cx, yb - 1.2, gname))
        out.append(r"\node[note] at (%.3f,%.2f) {%d};" % (cx, yb - 1.48, n))

    # ---------------- legend
    ly = yb - 2.05
    legend = [(0.0, TRAIN, "training"), (2.2, TEST, "held out (example fold)"),
              (6.4, "draw=phyE, fill=phyE!45, line width=0.5pt", "inner folds (E1)")]
    for dx, st, label in legend:
        out.append(dot(X0 + dx, ly, st, 0.25))
        out.append(r"\node[note, text=ink, anchor=west] at (%.2f,%.2f) {%s};" % (X0 + dx + 0.2, ly, label))

    tex = ("\\documentclass{article}\n\\input{lumenstyle}\n\\begin{document}\n\\begin{lumenfig}\n"
           "\\begin{tikzpicture}\n" + "\n".join(out) + "\n\\end{tikzpicture}\n\\end{lumenfig}\n\\end{document}\n")
    (HERE / "fig_evaluation.tex").write_text(tex, encoding="utf-8", newline="\n")
    print("wrote fig_evaluation.tex")


if __name__ == "__main__":
    main()
