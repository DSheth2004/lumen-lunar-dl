"""Build the LaTeX version of the manuscript in MDPI's official LaTeX class (Remote Sensing).

Input : paper/MANUSCRIPT.md (from build_manuscript.py; the same source as the Word build)
        paper/mdpi_latex/Definitions/ (MDPI LaTeX template, mdpi-res.com/data/MDPI_template.zip)
Output: paper/latex/LUMEN_RemoteSensing.tex, paper/latex/Definitions/, paper/latex/figures/,
        paper/LUMEN_latex_overleaf.zip (upload to Overleaf and compile with pdfLaTeX)

Template boilerplate is switched off from the document preamble, without editing MDPI's class:
the "Version ... submitted to" running header and DOI footer (page styles redefined to "N of M"),
the MDPI logo, the left-column box (editor, dates, citation, copyright) and the publisher's note.
Author, affiliation, contribution, funding and conflict fields are left empty.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

TPL = HERE / "mdpi_latex" / "Definitions"
OUT_DIR = HERE / "latex"
OUT_TEX = OUT_DIR / "LUMEN_RemoteSensing.tex"
ZIP = HERE / "LUMEN_latex_overleaf.zip"
FIG_DIR = HERE.parent / "figures"
VECTOR = {"fig01_framework.png": "fig_framework.pdf", "fig02_simulator.png": "fig_simulator.pdf",
          "fig03_architecture.png": "fig_architecture.pdf", "fig04_evaluation.png": "fig_evaluation.pdf"}
MAX_PX = 2400  # 300 dpi across the full 20 cm figure width; larger PNGs only slow pdfLaTeX down


def write_light_png(src: Path, dst: Path) -> None:
    """Copy a raster figure flattened onto white (no alpha channel) and capped at MAX_PX wide.
    pdfLaTeX has to split and recompress PNG alpha on every compile, which is what pushes the
    manuscript past Overleaf's free compile-time limit."""
    from PIL import Image

    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    if im.width > MAX_PX:
        im = im.resize((MAX_PX, round(im.height * MAX_PX / im.width)), Image.LANCZOS)
    im.save(dst, optimize=True)

SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ᵃ", "0123456789-+a"))
SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
CHARS = {"‐": "-", "‑": "-", "−": "$-$", "–": "--", "—": "---", "×": r"$\times$", "±": r"$\pm$", "≤": r"$\leq$",
         "≥": r"$\geq$", "≈": r"$\approx$", "→": r"$\rightarrow$", "⇒": r"$\Rightarrow$", "µ": r"$\upmu$",
         "μ": r"$\upmu$", "°": r"$^{\circ}$", "θ": r"$\theta$", "ρ": r"$\rho$", "β": r"$\beta$",
         "α": r"$\alpha$", "σ": r"$\sigma$", "λ": r"$\lambda$", "ε": r"$\epsilon$", "·": r"$\cdot$",
         "…": r"\ldots{}", "’": "'", "‘": "`", "“": "``", "”": "''", "\u00a0": "~", "\u2009": r"\,",
         "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
OK_LATIN = set("áéíóúàèìòùâêîôûäëïöüçñÁÉÍÓÚÖÜÇÑåøæßœ")
TOKEN = re.compile(r"(\$[^$\n]+?\$)|(\[@[\w]+(?:;\s*@[\w]+)*\])|(\*\*.+?\*\*)|(\*[^*\s][^*]*?\*)|(`[^`]+`)|(https?://[^\s)]+[^\s).,;])")


def text(s: str) -> str:
    """Plain text -> LaTeX (escapes, unicode super/subscripts, symbols)."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in SUP or c in SUB:
            table, mode = (SUP, "^") if c in SUP else (SUB, "_")
            j = i
            while j < len(s) and s[j] in table:
                j += 1
            out.append("$%s{%s}$" % (mode, "".join(table[x] for x in s[i:j])))
            i = j
            continue
        if c in CHARS:
            out.append(CHARS[c])
        elif ord(c) > 127 and c not in OK_LATIN:
            raise SystemExit("unmapped character %r in: %s" % (c, s[:80]))
        else:
            out.append(c)
        i += 1
    return "".join(out).replace("$$", "")            # merge adjacent math shifts, e.g. $^{3}$$-$


def inline(s: str) -> str:
    """Markdown inline -> LaTeX: maths kept, citations, bold, italics, code, URLs."""
    out, pos = [], 0
    for m in TOKEN.finditer(s):
        out.append(text(s[pos:m.start()]))
        maths, cite, bold, ital, code, url = m.groups()
        if maths:
            out.append(maths)
        elif cite:
            out.append(r"~\cite{%s}" % ",".join(k.strip().lstrip("@") for k in cite[1:-1].split(";")))
        elif bold:
            out.append(r"\textbf{%s}" % inline(bold[2:-2]))
        elif ital:
            out.append(r"\emph{%s}" % inline(ital[1:-1]))
        elif code:
            out.append(r"\texttt{%s}" % text(code[1:-1]))
        else:
            out.append(r"\url{%s}" % url)
        pos = m.end()
    out.append(text(s[pos:]))
    return "".join(out).replace(" ~\\cite", "~\\cite")


# --------------------------------------------------------------------------------------------
def parse(lines):
    """The same block grammar as build_mdpi_docx.parse_blocks."""
    sys.path.insert(0, str(HERE))
    from build_mdpi_docx import parse_blocks
    return parse_blocks(lines)


def heading(lvl, t):
    t = re.sub(r"^\d+(\.\d+)*\.\s+", "", t)
    return "\n\\%s{%s}\n" % ({2: "section", 3: "subsection", 4: "subsubsection"}[lvl], inline(t))


def table(rows, caption, label, footer, wide):
    ncol = max(len(r) for r in rows)
    head = " & ".join(r"\textbf{%s}" % inline(c.replace("**", "")) for c in rows[0]) + r" \\"
    body = "\n".join(" & ".join(inline(c) for c in (r + [""] * (ncol - len(r)))) + r" \\" for r in rows[1:])
    spec = "L" + "C" * (ncol - 1)
    width = r"\fulllength" if wide else r"\textwidth"
    tab = ("\\begin{tabularx}{%s}{%s}\n\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n\\end{tabularx}"
           % (width, spec, head, body))
    if wide:
        tab = "\\begin{adjustwidth}{-\\extralength}{0cm}\n\\small\n%s\n\\end{adjustwidth}" % tab
    else:
        tab = "\\small\n" + tab
    foot = ("\n\\noindent{\\footnotesize{%s}}" % inline(footer)) if footer else ""
    return ("\n\\begin{table}[H]\n\\caption{%s\\label{%s}}\n%s%s\n\\end{table}\n"
            % (inline(caption), label, tab, foot))


def figure(path, caption, label):
    name = Path(path).name
    target = VECTOR.get(name, name)
    return ("\n\\begin{figure}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\centering\n"
            "\\includegraphics[width=\\linewidth]{figures/%s}\n\\end{adjustwidth}\n"
            "\\caption{%s\\label{%s}}\n\\end{figure}\n" % (Path(target).stem, inline(caption), label))


def items(its):
    out, depth = [], -1
    for it in its:
        lvl = it["lvl"]
        env = "enumerate" if it["ordered"] else "itemize"
        while depth < lvl:
            out.append("\\begin{%s}[leftmargin=*,labelsep=5.8mm]" % env)
            depth += 1
        while depth > lvl:
            out.append("\\end{itemize}")
            depth -= 1
        out.append("\\item " + inline(it["text"]))
    env = "enumerate" if its[0]["ordered"] else "itemize"
    while depth > 0:
        out.append("\\end{itemize}")
        depth -= 1
    out.append("\\end{%s}" % env)
    return "\n" + "\n".join(out) + "\n"


def main() -> int:
    srcdir = HERE / "manuscript_src"
    md = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in sorted(srcdir.glob("[0-9][0-9]_*.md")))
    tsv = dict(line.split("\t", 1) for line in (srcdir / "references.tsv").read_text(encoding="utf-8").splitlines()
               if line.strip())
    title = re.search(r"^# (.+)$", md, re.M).group(1).strip()
    hl = md[md.index("## Highlights"):md.index("## Abstract")]
    find_b = [b.strip()[2:] for b in re.split(r"\n(?=- )", hl.split("**What are the implications")[0])[1:]]
    impl_b = [b.strip()[2:] for b in re.split(r"\n(?=- )", hl.split("**What are the implications of the main findings?**")[1])[1:]]
    abstract = " ".join(md[md.index("## Abstract") + 11:md.index("**Keywords:**")].split())
    keywords = re.search(r"\*\*Keywords:\*\*\s*(.+)", md).group(1).strip()
    main_md, back_md = md[md.index("## 1. Introduction"):].split("\n---\n")

    blocks = parse(main_md.splitlines())
    parts, n_tab, n_fig = [], 0, 0
    k = 0
    while k < len(blocks):
        b = blocks[k]
        t = b["t"]
        nxt = blocks[k + 1] if k + 1 < len(blocks) else None
        if t == "h":
            parts.append(heading(b["lvl"], b["text"]))
        elif t == "eq":
            parts.append("\n\\begin{equation}\n%s\n\\label{eq%s}\n\\end{equation}\n" % (b["tex"].strip(), b["num"]))
        elif t == "list":
            parts.append(items(b["items"]))
        elif t == "img":
            cap = nxt["text"] if nxt and nxt["t"] == "p" and nxt["text"].startswith("**Figure") else ""
            n_fig += 1
            parts.append(figure(b["path"], re.sub(r"^\*\*Figure \d+\.\*\*\s*", "", cap), "fig%d" % n_fig))
            k += 2 if cap else 1
            continue
        elif t == "p" and re.match(r"\*\*Table \d+\.\*\*", b["text"]) and nxt and nxt["t"] == "tbl":
            foot = None
            after = blocks[k + 2] if k + 2 < len(blocks) else None
            if after and after["t"] == "p" and after["text"].startswith("ᵃ"):
                foot = after["text"]
            n_tab += 1
            rows = nxt["rows"]
            parts.append(table(rows, re.sub(r"^\*\*Table \d+\.\*\*\s*", "", b["text"]), "tab%d" % n_tab,
                               foot, wide=max(len(r) for r in rows) >= 5))
            k += 3 if foot else 2
            continue
        elif t == "p":
            parts.append("\n" + inline(b["text"]) + "\n")
        k += 1

    # back matter
    def field(label):
        m = re.search(r"\*\*%s:\*\*(.*)" % re.escape(label), back_md)
        return inline(m.group(1).strip()) if m and m.group(1).strip() else ""
    abbrev = [r for r in re.findall(r"^\| (.+?) \| (.+?) \|$", back_md, re.M) if r[0] != "Abbreviation" and not r[0].startswith("---")]
    abbr_tab = "\n".join("%s & %s\\\\" % (inline(a), inline(b)) for a, b in abbrev)

    order = list(dict.fromkeys(k for grp in re.findall(r"\[(@[\w]+(?:;\s*@[\w]+)*)\]", md)
                               for k in [x.strip().lstrip("@") for x in grp.split(";")]))
    missing = [k for k in order if k not in tsv]
    if missing:
        raise SystemExit("unknown citation keys: %s" % missing)
    refs = [tsv[k] for k in order]                     # numbered in order of first citation, as MDPI requires
    keys = list(tsv)
    bib = "\n".join("\\bibitem[%s]{%s}\n%s" % (k, k, inline(r).replace("\\textbf{", "{\\bf ")) for k, r in zip(order, refs))

    hl_tex = ("\\noindent\\textbf{What are the main findings?}\n\\begin{itemize}[labelsep=2.5mm,topsep=-3pt]\n"
              + "\n".join("\\item " + inline(x) for x in find_b) + "\n\\end{itemize}\\vspace{3pt}\n"
              "\\noindent\\textbf{What are the implications of the main findings?}\n"
              "\\begin{itemize}[labelsep=2.5mm,topsep=-3pt]\n"
              + "\n".join("\\item " + inline(x) for x in impl_b) + "\n\\end{itemize}")

    doc = r"""%% LUMEN manuscript, MDPI Remote Sensing LaTeX class. Generated by paper/build_latex.py from
%% paper/MANUSCRIPT.md: edit paper/manuscript_src/*.md and rebuild rather than editing this file.
\documentclass[remotesensing,article,submit,pdftex,moreauthors]{Definitions/mdpi}

\firstpage{1}
\makeatletter
\setcounter{page}{\@firstpage}
\makeatother
\pubvolume{1}
\issuenum{1}
\articlenumber{0}
\pubyear{2026}
\copyrightyear{2026}
\datereceived{ }
\daterevised{ }
\dateaccepted{ }
\datepublished{ }
\hreflink{https://doi.org/}

%% ---- working draft: no journal boilerplate (MDPI's class file is left unchanged) ----
\makeatletter
\renewcommand{\contentleftcolumn}{}                           % left column: editor, dates, copyright
\let\LUMEN@includegraphics\includegraphics                    % MDPI and journal logos
\renewcommand{\includegraphics}[2][]{\IfSubStr{#2}{Definitions/logo}{}{\LUMEN@includegraphics[#1]{#2}}}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0pt}}
\fancyhf{}
\rhead{\fontsize{8}{8}\selectfont\thepage{} of \pageref*{LastPage}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\makeatother

\addhighlights{yes}
\renewcommand{\addhighlights}{%
@@HIGHLIGHTS@@
}

\Title{@@TITLE@@}
\TitleCitation{@@TITLE@@}
\Author{}
\AuthorNames{}
\isAPAStyle{\AuthorCitation{}}{\isChicagoStyle{\AuthorCitation{}}{\AuthorCitation{}}}
\address{}
\corres{}

\abstract{@@ABSTRACT@@}
\keyword{@@KEYWORDS@@}

\begin{document}
@@BODY@@

\vspace{6pt}

\authorcontributions{}
\funding{}
\dataavailability{@@DATAAVAIL@@}
\acknowledgments{@@ACK@@}
\conflictsofinterest{}

\abbreviations{Abbreviations}{
The following abbreviations are used in this manuscript:
\\

\noindent
\begin{tabular}{@{}ll}
@@ABBREV@@
\end{tabular}
}

\begin{adjustwidth}{-\extralength}{0cm}
\reftitle{References}
\begin{thebibliography}{999}
@@BIB@@
\end{thebibliography}
\end{adjustwidth}
\end{document}
"""
    rep = {"HIGHLIGHTS": hl_tex, "TITLE": inline(title), "ABSTRACT": inline(abstract),
           "KEYWORDS": inline(keywords), "BODY": "".join(parts), "DATAAVAIL": field("Data Availability Statement"),
           "ACK": field("Acknowledgments"), "ABBREV": abbr_tab, "BIB": bib}
    for key, val in rep.items():
        marker = "@@%s@@" % key
        if marker not in doc:
            raise SystemExit("template marker %s missing" % marker)
        doc = doc.replace(marker, val)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "figures").mkdir(parents=True)
    shutil.copytree(TPL, OUT_DIR / "Definitions")
    for png in sorted(set(re.findall(r"figures/([\w.-]+\.png)", main_md))):
        if png in VECTOR:
            shutil.copy(HERE / "figures_src" / VECTOR[png], OUT_DIR / "figures" / VECTOR[png])
        else:
            write_light_png(FIG_DIR / png, OUT_DIR / "figures" / png)
    OUT_TEX.write_text(doc, encoding="utf-8", newline="\n")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT_DIR.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT_DIR).as_posix())
    unused = set(keys) - set(order)
    print("wrote %s: %d figures, %d tables, %d references%s"
          % (OUT_TEX.relative_to(HERE), n_fig, n_tab, len(refs), "" if not unused else " (unused: %s)" % unused))
    print("wrote %s (upload to Overleaf; compiler: pdfLaTeX)" % ZIP.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
