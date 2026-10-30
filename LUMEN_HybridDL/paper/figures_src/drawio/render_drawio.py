"""Render the draw.io diagrams (make_diagrams.py) to vector PDF and 3x PNG with headless Chrome.

    python render_drawio.py        -> rendered/<name>.pdf and rendered/<name>.png

make_diagrams.py writes the .drawio files and a self-contained HTML preview of each page; the
preview is printed to PDF (vector, used by the LaTeX build) and screenshotted at 3x (used by Word).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rendered"
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
          shutil.which("chrome") or "", shutil.which("google-chrome") or "", shutil.which("chromium") or ""]


def main() -> int:
    chrome = next((c for c in CHROME if c and Path(c).exists()), None)
    if not chrome:
        sys.exit("Google Chrome is required to render the diagrams")
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([sys.executable, "-X", "utf8", str(HERE / "make_diagrams.py"), str(tmp / "drawio"),
                        str(tmp / "prev")], check=True)
        for prev in sorted((tmp / "prev").glob("*.html")):
            html = prev.read_text(encoding="utf-8")
            w, h = re.search(r"width:(\d+)px;height:(\d+)px", html).groups()
            page = html.replace("<html>", '<html><head><meta charset="utf-8"><style>@page{size:%spx %spx;'
                                "margin:0}html,body{margin:0}</style></head>" % (w, h), 1)
            src = tmp / ("page_" + prev.name)
            src.write_text(page, encoding="utf-8")
            for kind, args in (("pdf", ["--no-pdf-header-footer", "--print-to-pdf=%s" % (OUT / (prev.stem + ".pdf"))]),
                               ("png", ["--hide-scrollbars", "--force-device-scale-factor=3", "--window-size=%s,%s" % (w, h),
                                        "--screenshot=%s" % (OUT / (prev.stem + ".png"))])):
                subprocess.run([chrome, "--headless=new", "--disable-gpu", "--user-data-dir=%s" % (tmp / ("profile_" + prev.stem + kind))]
                               + args + [src.as_uri()], check=True, capture_output=True, timeout=120)
            print("  %s (%s x %s)" % (prev.stem, w, h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
