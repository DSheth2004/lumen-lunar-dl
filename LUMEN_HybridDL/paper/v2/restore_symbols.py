"""Restore the characters lost when the v2 rewrite (paper/UPDATED.txt) was copied from the PDF.

Only symbols, units, signs, sub/superscripts and maths markup are put back; the wording is not
touched. Every replacement must match exactly the number of times given, or nothing is written.
Output: paper/v2/UPDATED_restored.txt (UPDATED.txt itself is left as delivered).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "UPDATED.txt"
OUT = HERE / "UPDATED_restored.txt"

FIX = {
    "P005": [("TiO2", "TiO₂")],
    "P006": [("(M3)", "(M³)"), ("Fe2+", "Fe²⁺"), ("1 and 2 m,", "1 and 2 µm,")],
    "P007": [("TiO2", "TiO₂"), ("1 and 2 m band", "1 and 2 µm band"), ("calibrated to M3 [", "calibrated to M³ ["),
             ("map of M3 FeO", "map of M³ FeO"), ("Fernández-Daz", "Fernández-Díaz")],
    "P008": [("on M3 by", "on M³ by"), ("only on M3 against", "only on M³ against")],
    "P009": [("only R2 of", "only R² of"), ("Fernández-Daz", "Fernández-Díaz")],
    "P013": [("Using M3 spectra", "Using M³ spectra")],
    "P015": [("analyzed as 10, 10–20, 20–45 and 45 m fractions", "analyzed as <10, 10–20, 20–45 and <45 µm fractions"),
             ("The 10–20 m spectrum", "The 10–20 µm spectrum"), ("TiO2, Al2O3", "TiO₂, Al₂O₃"),
             ("71 M3 global", "71 M³ global"), ("(540.8 – 2497.1 nm with", "(540.8–2497.1 nm) with"),
             ("than 2.5 m.", "than 2.5 µm.")],
    "P017": [("are M3 Level-2", "are M³ Level-2")],
    "P020": [("where w is", "where $w$ is"), ("(SSA), 0 and are the cosines", "(SSA), $\\mu_0$ and $\\mu$ are the cosines"),
             ("geometry. H is", "geometry. $H$ is")],
    "P024": [("α = 4πk/λ", "$\\alpha = 4\\pi k/\\lambda$"), ("for k(λ)", "for $k(\\lambda)$"),
             ("reflectance and k –", "reflectance and $k$ –"), ("Equations (1)-(4)", "Equations (1)–(4)")],
    "P032": [("concentrations α = softplus(·) + 0.05", "concentrations $\\boldsymbol\\alpha = \\mathrm{softplus}(\\cdot)+0.05$"),
             ("mean α/α0, with α0 = ∑j αj,", "mean $\\boldsymbol\\alpha/\\alpha_0$, with $\\alpha_0=\\sum_j\\alpha_j$,"),
             ("log10(", "log₁₀("), ("npFe0", "npFe⁰")],
    "P034": [("and =0.005", "and $\\epsilon = 0.005$"), ("by L1 regularizer", "by $L_1$ regularizer")],
    "P035": [("using λ1 = 10", "using $\\lambda_1 = 10$"), ("for β = 0.5", "for $\\beta=0.5$")],
    "P036": [("blocked by sg,", "blocked by $\\mathrm{sg}$,"), ("and m_t prevents", "and $m_t$ prevents"),
             ("includes L_comp and L_scal", "includes $\\mathcal L_{comp}$ and $\\mathcal L_{scal}$")],
    "P037": [("by mass * FeO", "by mass × FeO"), ("of 2-3 training", "of 2–3 training")],
    "P038": [("s_i = |y_i − μ_i|/σ_i", "$s_i = |y_i - \\mu_i|/\\sigma_i$"),
             ("the ⌈(1−a)(n+1)⌉ smallest value q_{1−a}", "the $\\lceil (1-a)(n+1)\\rceil$-th smallest value $q_{1-a}$"),
             ("interval, μ ± q_{1−a} σ.", "interval, $\\mu \\pm q_{1-a}\\,\\sigma$.")],
    "P039": [("MAE_ref", "$\\mathrm{MAE}_{ref}$")],
    "P041": [("1e-4", "10⁻⁴")],
    "P042": [(" -> ", " → ", 6), ("trained on M3 pixels", "trained on M³ pixels"), ("uses M3-trained", "uses M³-trained")],
    "P043": [("y = a + bȳ", "$y = a + b\\hat y$")],
    "P048": [("7.8 × 10 −11 .", "7.8 × 10⁻¹¹."), ("TiO 2", "TiO₂", 2), ("x class-average", "× class-average"),
             ("<45 m spectra", "<45 µm spectra")],
    "P049": [("TiO2", "TiO₂")],
    "P052": [("TiO ,", "TiO₂,")],
    "P062": [("2.29wt%", "2.29 wt%")],
    "P066": [("3x3", "3 × 3")],
    "P067": [("two M3 band", "two M³ band")],
    "P073": [("(r = 0.79)", "(r = −0.79)"), ("(r = 0.28)", "(r = −0.28)")],
    "P078": [("from -1% to 2%", "from −1% to +2%")],
    "P079": [("6.3-7.5", "6.3–7.5")],
}

text = SRC.read_text(encoding="utf-8").replace("\r", "")
parts = re.split(r"(?m)^(<<P\d{3}>>)\s*$", text)
out, errors = [parts[0]], []
for i in range(1, len(parts), 2):
    pid = parts[i][2:-2]
    body, tail = parts[i + 1], ""
    m = re.search(r"\n(?====)", body)  # keep any section heading that follows the paragraph
    if m:
        body, tail = body[:m.start()], body[m.start():]
    para = " ".join(body.split())
    for rule in FIX.get(pid, []):
        old, new, n = (rule + (1,))[:3]
        if para.count(old) != n:
            errors.append("%s: %r found %d times, expected %d" % (pid, old, para.count(old), n))
        para = para.replace(old, new)
    out.append("%s\n%s\n%s" % (parts[i], para, tail if tail else "\n"))
if errors:
    raise SystemExit("nothing written:\n  " + "\n  ".join(errors))
OUT.write_text("".join(out), encoding="utf-8", newline="\n")
print("wrote %s (%d paragraphs, %d replacements)" % (OUT.name, len(parts) // 2,
                                                    sum((r + (1,))[2] for v in FIX.values() for r in v)))
