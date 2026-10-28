"""Export paper/MANUSCRIPT_MDPI.docx to paper/MANUSCRIPT_MDPI.pdf with Microsoft Word.

Word itself does the layout, so the PDF matches what opens in Word (MDPI template, Office Math
equations, line numbers). Requires Windows, Microsoft Word and pywin32 (pip install pywin32).

Full rebuild after editing manuscript_src/*.md:
    python build_manuscript.py && python build_mdpi_docx.py && python build_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "MANUSCRIPT_MDPI.docx"
OUT = HERE / "MANUSCRIPT_MDPI.pdf"
WD_EXPORT_PDF = 17
WD_STAT_PAGES = 2


def main() -> int:
    import win32com.client
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(SRC), ReadOnly=True, AddToRecentFiles=False)
        doc.Fields.Update()
        pages = doc.ComputeStatistics(WD_STAT_PAGES)
        doc.ExportAsFixedFormat(str(OUT), WD_EXPORT_PDF)
        doc.Close(False)
    finally:
        word.Quit()
    print("wrote %s (%d pages)" % (OUT.name, pages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
