"""Tiny draw.io (mxGraph) writer + approximate HTML preview renderer.

Every edge is a straight polyline (edgeStyle=none) with explicit exit/entry
ports and waypoints, so draw.io renders exactly what the preview shows.
"""
from xml.sax.saxutils import escape, quoteattr

FONT = "Helvetica"
INK = "#1F2937"
MUTED = "#6B7280"
LINE = "#374151"

# palette: (fill, stroke, title)
PAL = {
    "data":  ("#FFF6E5", "#D08C2B", "#8A5A12"),
    "phys":  ("#EAF2FC", "#2F6DB5", "#1D4A80"),
    "learn": ("#E9F6EE", "#2E8B57", "#1C5E39"),
    "rep":   ("#F2EEFB", "#6B4FA0", "#4A3278"),
    "neut":  ("#FFFFFF", "#9CA3AF", "#374151"),
    "gray":  ("#F3F4F6", "#9CA3AF", "#374151"),
}


def parse_style(s):
    d = {}
    for part in s.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
        else:
            d[part] = "1"
    return d


class Diagram:
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.cells = []
        self._n = 0

    def _id(self, p="c"):
        self._n += 1
        return f"{p}{self._n}"

    # ---------------------------------------------------------------- vertices
    def vertex(self, x, y, w, h, value="", style="", id=None):
        id = id or self._id("v")
        self.cells.append(dict(kind="v", id=id, x=x, y=y, w=w, h=h, value=value, style=style))
        return id

    def card(self, x, y, w, h, title, body="", small="", kind="phys", dashed=False,
             fs=13, tfs=15, align="center", sw=1.5, id=None):
        fill, stroke, tcol = PAL[kind]
        html = f'<b style="font-size:{tfs}px;color:{tcol}">{title}</b>'
        if body:
            html += f'<div style="margin-top:4px;font-size:{fs}px">{body}</div>'
        if small:
            html += f'<div style="margin-top:4px;font-size:{fs-1}px;color:{MUTED}">{small}</div>'
        st = (f"rounded=1;arcSize=7;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
              f"strokeWidth={sw};fontFamily={FONT};fontSize={fs};fontColor={INK};align={align};"
              f"verticalAlign=middle;spacing=8;")
        if dashed:
            st += "dashed=1;dashPattern=6 4;"
        return self.vertex(x, y, w, h, html, st, id)

    def group(self, x, y, w, h, title="", badge=None, kind="neut", id=None):
        stroke = PAL[kind][1] if kind != "neut" else "#C7CCD4"
        gid = self.vertex(x, y, w, h, "",
                          f"rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=#FAFBFC;"
                          f"strokeColor={stroke};dashed=1;dashPattern=5 4;strokeWidth=1.2;"
                          f"opacity=100;", id)
        tx = x + 12
        if badge is not None:
            col = PAL[kind][1] if kind != "neut" else "#6B7280"
            self.vertex(x + 12, y - 13, 26, 26, f"<b>{badge}</b>",
                        f"ellipse;whiteSpace=wrap;html=1;fillColor={col};strokeColor=#FFFFFF;"
                        f"strokeWidth=2;fontColor=#FFFFFF;fontFamily={FONT};fontSize=13;")
            tx = x + 44
        if title:
            col = PAL[kind][2] if kind != "neut" else "#4B5563"
            self.text(tx, y - 12, w - (tx - x) - 10, 24,
                      f'<b style="letter-spacing:1px">{title}</b>', fs=12, color=col,
                      align="left", bg="#FFFFFF")
        return gid

    def text(self, x, y, w, h, html, fs=12, color=INK, align="center", valign="middle",
             bg=None, id=None, rotation=0):
        st = (f"text;html=1;whiteSpace=wrap;fontFamily={FONT};fontSize={fs};fontColor={color};"
              f"align={align};verticalAlign={valign};spacing=2;")
        if bg:
            st += f"labelBackgroundColor={bg};"
        if rotation:
            st += f"rotation={rotation};"
        return self.vertex(x, y, w, h, html, st, id)

    def shape(self, x, y, w, h, kind="ellipse", fill="#FFFFFF", stroke=LINE, sw=1.2,
              value="", fs=12, extra="", id=None):
        st = (f"{kind};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
              f"strokeWidth={sw};fontFamily={FONT};fontSize={fs};fontColor={INK};{extra}")
        return self.vertex(x, y, w, h, value, st, id)

    # ------------------------------------------------------------------- edges
    def edge(self, src, tgt, exit=None, entry=None, pts=(), label="", dashed=False,
             color=LINE, sw=1.6, arrow=True, lfs=11, lpos=None, start_arrow=False):
        st = (f"edgeStyle=none;rounded=1;html=1;strokeColor={color};strokeWidth={sw};"
              f"fontFamily={FONT};fontSize={lfs};fontColor={MUTED};labelBackgroundColor=#FFFFFF;"
              f"endArrow={'blockThin' if arrow else 'none'};endFill=1;endSize=7;"
              f"startArrow={'blockThin' if start_arrow else 'none'};startFill=1;startSize=7;")
        if exit:
            st += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        if dashed:
            st += "dashed=1;dashPattern=5 4;"
        eid = self._id("e")
        self.cells.append(dict(kind="e", id=eid, src=src, tgt=tgt, style=st, value=label,
                               pts=list(pts), lpos=lpos))
        return eid

    # -------------------------------------------------------------------- XML
    def page_xml(self, pid):
        out = [f'<diagram id="{pid}" name={quoteattr(self.name)}>',
               f'<mxGraphModel dx="{self.w}" dy="{self.h}" grid="0" gridSize="10" guides="1" '
               f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
               f'pageWidth="{self.w}" pageHeight="{self.h}" background="#FFFFFF" math="0" shadow="0">',
               "<root>", '<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        for c in self.cells:
            v = quoteattr(c["value"])
            if c["kind"] == "v":
                out.append(f'<mxCell id="{c["id"]}" value={v} style={quoteattr(c["style"])} '
                           f'vertex="1" parent="1"><mxGeometry x="{c["x"]}" y="{c["y"]}" '
                           f'width="{c["w"]}" height="{c["h"]}" as="geometry"/></mxCell>')
            else:
                geo = '<mxGeometry relative="1" as="geometry">'
                if c["lpos"] is not None:
                    geo = f'<mxGeometry x="{c["lpos"]}" relative="1" as="geometry">'
                if c["pts"]:
                    geo += '<Array as="points">' + "".join(
                        f'<mxPoint x="{px}" y="{py}"/>' for px, py in c["pts"]) + "</Array>"
                geo += "</mxGeometry>"
                out.append(f'<mxCell id="{c["id"]}" value={v} style={quoteattr(c["style"])} '
                           f'edge="1" parent="1" source="{c["src"]}" target="{c["tgt"]}">'
                           f'{geo}</mxCell>')
        out += ["</root>", "</mxGraphModel>", "</diagram>"]
        return "\n".join(out)

    # ---------------------------------------------------------------- preview
    def preview_html(self):
        byid = {c["id"]: c for c in self.cells if c["kind"] == "v"}
        parts = [f'<div style="position:relative;width:{self.w}px;height:{self.h}px;'
                 f'background:#fff;font-family:Helvetica,Arial,sans-serif;overflow:hidden">']
        svg = [f'<svg width="{self.w}" height="{self.h}" style="position:absolute;left:0;top:0;'
               f'z-index:{{Z}}" xmlns="http://www.w3.org/2000/svg">'
               '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
               'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
               'fill="context-stroke"/></marker></defs>']
        z = 1
        for c in self.cells:
            s = parse_style(c["style"])
            if c["kind"] == "v":
                x, y, w, h = c["x"], c["y"], c["w"], c["h"]
                fill = s.get("fillColor", "none")
                stroke = s.get("strokeColor", "none")
                sw = s.get("strokeWidth", "1")
                is_text = "text" in s
                br = "50%" if "ellipse" in s else (f'{min(w, h) * float(s.get("arcSize", 10)) / 100}px' if s.get("rounded") == "1" else "0")
                border = "none" if is_text or stroke == "none" else f'{sw}px {"dashed" if s.get("dashed") == "1" else "solid"} {stroke}'
                if "partialRectangle" in s:
                    border = "none"
                    bl = f'{sw}px solid {stroke}'
                    extra = f"border-left:{bl};border-right:{bl};border-bottom:{bl};border-top:none;"
                else:
                    extra = ""
                bg = "transparent" if is_text else fill
                if "rhombus" in s:
                    parts.append(f'<svg style="position:absolute;left:{x}px;top:{y}px;z-index:{z}" width="{w}" height="{h}"><polygon points="{w/2},1 {w-1},{h/2} {w/2},{h-1} 1,{h/2}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/></svg>')
                    bg, border = "transparent", "none"
                al = s.get("align", "center")
                va = {"top": "flex-start", "bottom": "flex-end"}.get(s.get("verticalAlign", "middle"), "center")
                rot = s.get("rotation")
                horiz = s.get("horizontal", "1") == "0"
                lab = c["value"]
                lbg = s.get("labelBackgroundColor")
                inner = f'<span style="background:{lbg}">{lab}</span>' if lbg else lab
                tr = f"transform:rotate({rot}deg);" if rot else ""
                if horiz:
                    inner = f'<div style="transform:rotate(-90deg);white-space:nowrap">{inner}</div>'
                parts.append(
                    f'<div style="position:absolute;box-sizing:border-box;left:{x}px;top:{y}px;width:{w}px;'
                    f'height:{h}px;background:{bg};border:{border};{extra}border-radius:{br};display:flex;'
                    f'align-items:{va};justify-content:center;text-align:{al};padding:{s.get("spacing", "2")}px;'
                    f'font-size:{s.get("fontSize", "12")}px;color:{s.get("fontColor", INK)};z-index:{z};'
                    f'opacity:{int(s.get("opacity", "100")) / 100};{tr}line-height:1.25">'
                    f'<div style="width:100%">{inner}</div></div>')
                z += 1
        # edges on top of everything except label text, like draw.io ordering is close enough
        for c in self.cells:
            if c["kind"] != "e":
                continue
            s = parse_style(c["style"])
            a, b = byid[c["src"]], byid[c["tgt"]]

            def port(v, px, py):
                return v["x"] + float(px) * v["w"], v["y"] + float(py) * v["h"]

            def ctr(v):
                return v["x"] + v["w"] / 2, v["y"] + v["h"] / 2

            p0 = port(a, s["exitX"], s["exitY"]) if "exitX" in s else ctr(a)
            p1 = port(b, s["entryX"], s["entryY"]) if "entryX" in s else ctr(b)
            pts = [p0] + [tuple(p) for p in c["pts"]] + [p1]
            d = "M" + " L".join(f"{px},{py}" for px, py in pts)
            dash = 'stroke-dasharray="5 4"' if s.get("dashed") == "1" else ""
            me = 'marker-end="url(#a)"' if s.get("endArrow") != "none" else ""
            ms = 'marker-start="url(#a)"' if s.get("startArrow") not in (None, "none") else ""
            svg.append(f'<path d="{d}" fill="none" stroke="{s["strokeColor"]}" '
                       f'stroke-width="{s["strokeWidth"]}" {dash} {me} {ms}/>')
            if c["value"]:
                # label at midpoint of the polyline (by segment count; approximate)
                import math
                L = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
                t = (float(c["lpos"]) + 1) / 2 if c["lpos"] is not None else 0.5
                target, acc = sum(L) * t, 0
                for i, l in enumerate(L):
                    if acc + l >= target:
                        f = (target - acc) / l if l else 0
                        mx = pts[i][0] + f * (pts[i + 1][0] - pts[i][0])
                        my = pts[i][1] + f * (pts[i + 1][1] - pts[i][1])
                        break
                    acc += l
                svg.append(f'<foreignObject x="{mx-60}" y="{my-14}" width="120" height="28">'
                           f'<div xmlns="http://www.w3.org/1999/xhtml" style="font:{s.get("fontSize", 11)}px Helvetica;'
                           f'color:{MUTED};text-align:center"><span style="background:#fff">{c["value"]}</span></div></foreignObject>')
        svg.append("</svg>")
        parts.append("".join(svg).replace("{Z}", str(z)))
        parts.append("</div>")
        return "".join(parts)


def write_drawio(path, diagrams):
    body = "\n".join(d.page_xml(f"lumen-p{i}") for i, d in enumerate(diagrams))
    xml = (f'<mxfile host="app.diagrams.net" agent="LUMEN diagram generator" version="24.7.0" '
           f'type="device">\n{body}\n</mxfile>\n')
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
