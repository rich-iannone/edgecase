"""Shared design system for edgecase doc graphics.

Every figure is a `Scene` on one fixed banner canvas (`W` x `H`). A scene is
built from structured elements (nodes, edges, text, cards, hulls), never raw
SVG strings, so the same geometry can be (a) serialized for both themes and
(b) linted for collisions, overflow, and other layout defects.

Figures are composed from a small set of *templates* that fix where panels,
arrows, and captions go. Families of methods share a template and a visual
encoding, which is where the cohesion comes from. See DOC_GRAPHICS_PLAN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

# --- canvas & typography ---------------------------------------------------

W, H = 760, 220  # every figure; ~1:1 with the docs content column
MARGIN = 12

R = 16  # default node radius
R_SMALL = 11  # dense or miniature graphs (contrast pairs, multiples)

STROKE = 1.5
EMPH_STROKE = 2.25

FONT_SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# The only allowed text sizes. Linted.
SIZE_LABEL = 13  # node labels
SIZE_LABEL_SMALL = 10  # labels inside R_SMALL nodes
SIZE_CAPTION = 12  # panel captions (code)
SIZE_CARD = 13  # result card body
SIZE_NOTE = 11  # badges, weights, table cells, axis ticks
SIZE_SYMBOL = 22  # operator glyphs between panels (∪, =, ...)
TEXT_SIZES = {SIZE_LABEL, SIZE_LABEL_SMALL, SIZE_CAPTION, SIZE_CARD, SIZE_NOTE, SIZE_SYMBOL}

CAPTION_Y = 206  # caption baseline for every template
CONTENT_TOP = MARGIN
CONTENT_BOTTOM = 184  # graphs/cards must stay above this line

# --- theme -----------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    suffix: Literal["light", "dark"]
    ink: str
    paper: str
    accent: str
    g2: str  # teal
    g3: str  # violet
    g4: str  # gold


LIGHT = Theme("light", ink="#1f2328", paper="#ffffff", accent="#c2410c",
              g2="#0e7490", g3="#7c3aed", g4="#a16207")
DARK = Theme("dark", ink="#e6edf3", paper="#0d1117", accent="#f97316",
             g2="#22d3ee", g3="#a78bfa", g4="#facc15")
THEMES = (LIGHT, DARK)


def _mix(c1: str, c2: str, t: float) -> str:
    """Blend hex c1 toward c2 by t (0 = c1, 1 = c2)."""
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


Color = Literal["ink", "accent", "g2", "g3", "g4", "paper"]


def _color(theme: Theme, c: Color) -> str:
    return {"ink": theme.ink, "accent": theme.accent, "g2": theme.g2,
            "g3": theme.g3, "g4": theme.g4, "paper": theme.paper}[c]


def _tint(theme: Theme, c: Color, amount: float = 0.86) -> str:
    return _mix(_color(theme, c), theme.paper, amount)


# --- element styles --------------------------------------------------------
#
# The vocabulary (identical meaning in every figure):
#   default  plain ink              ordinary graph element
#   accent   accent stroke + tint   THE thing the method is about / returns
#   soft     accent stroke, no fill the input you named (start node, query
#                                   pair); soft edges are thin, lighter accent
#   muted    ink at 35%             present but irrelevant context
#   removed  dashed, ink at 35%     deleted / absent / filtered out
#   ghost    dashed accent          would-be or newly inferred element
#   g2/g3/g4 group colors + tint    partitions (communities, colors, sides);
#                                   group 1 is always `accent`

Style = Literal["default", "accent", "soft", "muted", "removed", "ghost", "g2", "g3", "g4"]
MUTED = 0.35


def _paint(theme: Theme, style: Style) -> tuple[str, float, str, float, str]:
    """(stroke, stroke_opacity, fill, stroke_width, dasharray) for a style."""
    p = theme.paper
    if style == "accent":
        return theme.accent, 1, _tint(theme, "accent"), EMPH_STROKE, ""
    if style == "soft":
        return theme.accent, 1, p, EMPH_STROKE, ""
    if style == "muted":
        return theme.ink, MUTED, p, STROKE, ""
    if style == "removed":
        return theme.ink, MUTED, p, STROKE, "4 3"
    if style == "ghost":
        return theme.accent, 1, p, STROKE, "4 3"
    if style in ("g2", "g3", "g4"):
        return _color(theme, style), 1, _tint(theme, style), EMPH_STROKE, ""
    return theme.ink, 1, p, STROKE, ""


# --- elements --------------------------------------------------------------

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class Node:
    id: str
    x: float
    y: float
    label: str = ""
    style: Style = "default"
    r: float = R
    heat: float | None = None  # 0..1 score shading (node metrics); overrides fill


@dataclass
class Edge:
    a: str
    b: str
    directed: bool = False
    style: Style = "default"
    weight: str | None = None
    bend: float = 0.0  # perpendicular bow in px (multi-edges, reciprocal pairs)
    weight_t: float = 0.5  # where along the edge the weight sits
    width: float | None = None  # override stroke width (e.g., flow volume)


@dataclass
class Text:
    x: float
    y: float
    text: str
    size: int = SIZE_CAPTION
    mono: bool = True
    anchor: Literal["start", "middle", "end"] = "middle"
    color: Color = "ink"
    opacity: float = 1.0
    weight: int = 400
    italic: bool = False
    baseline: Literal["central", "alphabetic"] = "central"
    role: str = "text"  # "caption", "label", "weight", "badge", "card", ...
    owner: str | None = None  # node/edge id this text belongs to (lint)
    boxed: bool = False  # paper background box (weights on edges)
    max_w: float | None = None  # captions: width of the panel they label (lint)

    def bbox(self) -> BBox:
        w = text_width(self.text, self.size, self.mono, self.weight)
        h = self.size * 1.0
        if self.anchor == "start":
            x0 = self.x
        elif self.anchor == "end":
            x0 = self.x - w
        else:
            x0 = self.x - w / 2
        if self.baseline == "central":
            y0 = self.y - h / 2
        else:
            y0 = self.y - h * 0.78
        return (x0, y0, x0 + w, y0 + h)


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    rx: float = 6
    stroke: Color | None = "ink"
    stroke_opacity: float = 0.35
    fill: Literal["paper", "tint", "none"] = "paper"
    fill_color: Color = "ink"
    dash: str = ""
    role: str = "rect"  # "card", "table", "bar", "frame"
    solid: bool = False  # filled with the color itself (bars)
    fill_opacity: float = 1.0

    def bbox(self) -> BBox:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    color: Color = "ink"
    opacity: float = 1.0
    width: float = STROKE
    dash: str = ""
    arrow: bool = False
    role: str = "line"  # "panel_arrow", "divider", "axis", "rule"


@dataclass
class Hull:
    """Soft rounded region behind a set of nodes (communities, components)."""

    nodes: list[str]
    color: Color = "accent"
    pad: float = 9
    opacity: float = 0.13


@dataclass
class Path:
    d: str
    color: Color = "ink"
    opacity: float = 1.0
    width: float = STROKE
    dash: str = ""
    fill: str = "none"
    role: str = "path"


# --- text metrics ----------------------------------------------------------


def text_width(s: str, size: float, mono: bool = True, weight: int = 400) -> float:
    if mono:
        return len(s) * size * 0.61
    # Rough proportional-font estimate, deliberately on the wide side.
    w = 0.0
    for ch in s:
        if ch in "il.,:;|!'":
            w += 0.3
        elif ch in "mwMW":
            w += 0.85
        elif ch.isupper() or ch.isdigit():
            w += 0.66
        else:
            w += 0.56
    return w * size * (1.05 if weight >= 600 else 1.0)


# --- scene -----------------------------------------------------------------


class Scene:
    """A figure under construction. Coordinates are absolute canvas px."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.texts: list[Text] = []
        self.rects: list[Rect] = []
        self.lines: list[Line] = []
        self.hulls: list[Hull] = []
        self.paths: list[Path] = []
        self._lint_ignore: set[str] = set()

    # -- graph ---------------------------------------------------------------

    def node(self, id: str, x: float, y: float, label: str | None = None,
             style: Style = "default", r: float = R, heat: float | None = None) -> Node:
        if id in self.nodes:
            raise ValueError(f"duplicate node id {id!r}")
        n = Node(id, x, y, id if label is None else label, style, r, heat)
        self.nodes[id] = n
        return n

    def edge(self, a: str, b: str, directed: bool = False, style: Style = "default",
             weight: str | float | None = None, bend: float = 0.0,
             weight_t: float = 0.5, width: float | None = None) -> Edge:
        e = Edge(a, b, directed, style, None if weight is None else str(weight),
                 bend, weight_t, width)
        self.edges.append(e)
        return e

    def graph(self, pos: dict[str, tuple[float, float]], edges: Iterable,
              *, directed: bool = False, styles: dict[str, Style] | None = None,
              labels: dict[str, str] | bool = True, r: float = R,
              prefix: str = "", heat: dict[str, float] | None = None) -> dict[str, Node]:
        """Add many nodes/edges at once.

        `edges` items are (a, b) or (a, b, dict(style=..., weight=..., ...)).
        `prefix` namespaces ids so two panels can reuse the same labels.
        """
        styles = styles or {}
        out = {}
        for k, (x, y) in pos.items():
            if labels is True:
                lab = k
            elif labels is False:
                lab = ""
            else:
                lab = labels.get(k, k)
            out[k] = self.node(prefix + k, x, y, lab, styles.get(k, "default"), r,
                               None if heat is None else heat.get(k))
        for e in edges:
            a, b = e[0], e[1]
            kw = dict(e[2]) if len(e) > 2 else {}
            kw.setdefault("directed", directed)
            self.edge(prefix + a, prefix + b, **kw)
        return out

    # -- text & furniture ----------------------------------------------------

    def text(self, x: float, y: float, s: str, **kw) -> Text:
        t = Text(x, y, s, **kw)
        self.texts.append(t)
        return t

    def caption(self, x: float, s: str, *, emphasis: bool = False,
                max_w: float | None = None) -> Text:
        """Code caption under a panel, on the shared caption baseline."""
        return self.text(x, CAPTION_Y, s, size=SIZE_CAPTION, role="caption",
                         opacity=1.0 if emphasis else 0.72,
                         color="accent" if emphasis else "ink", max_w=max_w)

    def badge(self, node_id: str, s: str, *, where: str = "auto", filled: bool = False,
              color: Color = "accent") -> Text:
        """Small pill attached to a node: visit order, metric value, ids.

        `where` is a compass direction (n, ne, e, se, s, sw, w, nw) or "auto",
        which picks the first direction (diagonals first) whose pill clears
        every edge, node, and existing text. Add edges before badges.
        """
        n = self.nodes[node_id]
        tw = text_width(s, SIZE_NOTE)
        pw, ph = max(tw + 9, 17), 16

        def spot(d: str) -> tuple[float, float]:
            ang = math.radians({"n": -90, "ne": -45, "e": 0, "se": 45, "s": 90, "sw": 135,
                                "w": 180, "nw": -135}[d])
            ux, uy = math.cos(ang), math.sin(ang)
            # push the pill out until its nearest edge clears the circle by ~1px
            k = n.r + 1
            return (n.x + ux * k + (pw / 2 - 3) * (1 if ux > 0.1 else -1 if ux < -0.1 else 0),
                    n.y + uy * k + (ph / 2 - 2) * (1 if uy > 0.1 else -1 if uy < -0.1 else 0)
                    + (0 if abs(uy) > 0.1 else 0))

        order = ["ne", "nw", "se", "sw", "n", "s", "e", "w"] if where == "auto" else [where]
        cx, cy = spot(order[0])
        for d in order:
            x, y = spot(d)
            if where != "auto" or self._clear((x - pw / 2, y - ph / 2, x + pw / 2, y + ph / 2), node_id):
                cx, cy = x, y
                break
        self.rects.append(Rect(cx - pw / 2, cy - ph / 2, pw, ph, rx=ph / 2,
                               stroke=color, stroke_opacity=1, fill="paper",
                               fill_color=color, role="badge", solid=filled))
        return self.text(cx, cy, s, size=SIZE_NOTE, role="badge", owner=node_id,
                         color="paper" if filled else "ink", weight=600 if filled else 400)

    def _clear(self, b: BBox, owner: str | None = None) -> bool:
        """True if box `b` is free of edges, nodes (other than owner), and text."""
        x0, y0, x1, y1 = b
        if x0 < MARGIN or y0 < MARGIN or x1 > W - MARGIN or y1 > CONTENT_BOTTOM:
            return False
        for n in self.nodes.values():
            nx, ny = min(max(n.x, x0), x1), min(max(n.y, y0), y1)
            lim = n.r * 0.6 if n.id == owner else n.r + 2
            if math.hypot(n.x - nx, n.y - ny) < lim:
                return False
        for e in self.edges:
            for px, py in self.edge_geometry(e)[1]:
                if x0 - 2 <= px <= x1 + 2 and y0 - 2 <= py <= y1 + 2:
                    return False
            pts = self.edge_geometry(e)[1]
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                for t in (0.25, 0.5, 0.75):
                    px, py = ax + (bx - ax) * t, ay + (by - ay) * t
                    if x0 - 2 <= px <= x1 + 2 and y0 - 2 <= py <= y1 + 2:
                        return False
        for t in self.texts:
            tb = t.bbox()
            if x0 < tb[2] + 1 and tb[0] < x1 + 1 and y0 < tb[3] + 1 and tb[1] < y1 + 1:
                return False
        for r in self.rects:
            if r.role in ("card", "table", "badge"):
                rb = r.bbox()
                if x0 < rb[2] + 1 and rb[0] < x1 + 1 and y0 < rb[3] + 1 and rb[1] < y1 + 1:
                    return False
        return True

    def card(self, x: float, y: float, w: float, lines: list[str | tuple[str, Color]],
             *, title: str | None = None, align: Literal["start", "middle"] = "start") -> Rect:
        """Result card: the 'output' of a compute-style figure.

        Lines are strings, or (string, color) to highlight a value.
        """
        lh = 20
        pad_y = 12
        n = len(lines) + (1 if title else 0)
        h = pad_y * 2 + lh * n - 4
        rect = Rect(x, y, w, h, rx=8, role="card")
        self.rects.append(rect)
        ty = y + pad_y + lh / 2 - 2
        tx = x + 14 if align == "start" else x + w / 2
        if title:
            self.text(tx, ty, title, size=SIZE_NOTE, anchor=align, opacity=0.6, role="card")
            ty += lh
        for ln in lines:
            s, c = (ln, "ink") if isinstance(ln, str) else ln
            self.text(tx, ty, s, size=SIZE_CARD, anchor=align, color=c, role="card",
                      weight=600 if c != "ink" else 400)
            ty += lh
        return rect

    def card_centered(self, box: "Box", lines: list, *, title: str | None = None,
                      w: float | None = None, align: Literal["start", "middle"] = "start") -> Rect:
        """A result card centered in `box` (vertically within the content area)."""
        longest = max([text_width(s if isinstance(s, str) else s[0], SIZE_CARD) for s in lines]
                      + ([text_width(title, SIZE_NOTE)] if title else []))
        cw = w if w is not None else min(box.w, longest + 30)
        n = len(lines) + (1 if title else 0)
        ch = 24 + 20 * n - 4
        return self.card(box.cx - cw / 2, box.cy - ch / 2, cw, lines, title=title, align=align)

    def table(self, x: float, y: float, rows: list[list[str]], col_w: list[float],
              *, row_h: float = 20, highlight: dict[tuple[int, int], Color] | None = None,
              header: bool = True) -> BBox:
        """Compact grid; row 0 is the header when header=True."""
        highlight = highlight or {}
        total = sum(col_w)
        self.rects.append(Rect(x, y, total, row_h * len(rows), rx=4, role="table"))
        for i, row in enumerate(rows):
            cx = x
            for j, (v, cw) in enumerate(zip(row, col_w)):
                hc = highlight.get((i, j))
                if hc:
                    self.rects.append(Rect(cx + 1, y + i * row_h + 1, cw - 2, row_h - 2, rx=2,
                                           stroke=None, fill="tint", fill_color=hc, role="cell"))
                self.text(cx + cw / 2, y + i * row_h + row_h / 2, v, size=SIZE_NOTE,
                          weight=600 if (header and i == 0) or hc else 400,
                          color=hc or "ink", opacity=0.7 if (header and i == 0) else 1,
                          role="table")
                cx += cw
        for i in range(1, len(rows)):
            self.lines.append(Line(x, y + i * row_h, x + total, y + i * row_h, opacity=0.2 if not (header and i == 1) else 0.5,
                                   width=1, role="rule"))
        cx = x
        for cw in col_w[:-1]:
            cx += cw
            self.lines.append(Line(cx, y, cx, y + row_h * len(rows), opacity=0.2, width=1, role="rule"))
        return (x, y, x + total, y + row_h * len(rows))

    def table_centered(self, box: "Box", rows: list[list[str]], col_w: list[float], **kw) -> BBox:
        tw = sum(col_w)
        th = kw.get("row_h", 20) * len(rows)
        return self.table(box.cx - tw / 2, box.cy - th / 2, rows, col_w, **kw)

    def hull(self, nodes: list[str], color: Color = "accent", pad: float = 9,
             opacity: float = 0.13) -> None:
        self.hulls.append(Hull(nodes, color, pad, opacity))

    def arrow(self, x1: float, y: float, x2: float) -> None:
        """The 'becomes' arrow between panels."""
        self.lines.append(Line(x1, y, x2, y, width=2.25, arrow=True, role="panel_arrow"))

    def divider(self, x: float) -> None:
        self.lines.append(Line(x, CONTENT_TOP + 6, x, CAPTION_Y + 4, opacity=0.2, width=1,
                               dash="3 4", role="divider"))

    def symbol(self, x: float, y: float, s: str) -> Text:
        return self.text(x, y, s, size=SIZE_SYMBOL, mono=False, weight=600, role="symbol")

    def line(self, *a, **kw) -> Line:
        ln = Line(*a, **kw)
        self.lines.append(ln)
        return ln

    def rect(self, *a, **kw) -> Rect:
        r = Rect(*a, **kw)
        self.rects.append(r)
        return r

    def path(self, d: str, **kw) -> Path:
        p = Path(d, **kw)
        self.paths.append(p)
        return p

    def lint_ignore(self, *rules: str) -> None:
        """Opt a figure out of specific lint rules (use sparingly; say why)."""
        self._lint_ignore.update(rules)

    # -- rendering -----------------------------------------------------------

    def render(self, theme: Theme, title: str = "") -> str:
        o: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" role="img" xml:space="preserve">',
        ]
        if title:
            o.append(f"<title>{_esc(title)}</title>")
        o.append(f'<rect width="{W}" height="{H}" fill="{theme.paper}"/>')
        for h in self.hulls:
            o.append(self._render_hull(h, theme))
        for r in self.rects:
            if r.role in ("card", "table", "frame", "bar", "cell", "region"):
                o.append(_render_rect(r, theme))
        for p in self.paths:
            o.append(_render_path(p, theme))
        for ln in self.lines:
            o.append(_render_line(ln, theme))
        for e in self.edges:
            o.append(self._render_edge(e, theme))
        for n in self.nodes.values():
            o.append(_render_node(n, theme))
        for r in self.rects:
            if r.role == "badge":
                o.append(_render_rect(r, theme))
        for t in self.texts:
            o.append(_render_text(t, theme))
        o.append("</svg>")
        return "\n".join(o) + "\n"

    def _render_hull(self, h: Hull, theme: Theme) -> str:
        pts = [(self.nodes[i].x, self.nodes[i].y) for i in h.nodes]
        rad = max(self.nodes[i].r for i in h.nodes) + h.pad
        hull = _convex_hull(pts)
        c = _color(theme, h.color)
        if len(hull) == 1:
            x, y = hull[0]
            shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{c}"/>'
        else:
            d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in hull) + " Z"
            shape = (f'<path d="{d}" fill="{c}" stroke="{c}" stroke-width="{rad * 2:.1f}" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        return f'<g opacity="{h.opacity}">{shape}</g>'

    def edge_geometry(self, e: Edge) -> tuple[str, list[tuple[float, float]], tuple[float, float, float, float]]:
        """SVG path `d`, sampled polyline (for lint), and end tangent (x, y, ux, uy)."""
        a, b = self.nodes[e.a], self.nodes[e.b]
        if e.a == e.b:
            # Self-loop: teardrop above the node.
            lr = a.r * 0.95
            x0, y0 = a.x - a.r * 0.55, a.y - a.r * 0.83
            x3, y3 = a.x + a.r * 0.55, a.y - a.r * 0.83
            c1 = (a.x - lr * 1.9, a.y - a.r - lr * 2.3)
            c2 = (a.x + lr * 1.9, a.y - a.r - lr * 2.3)
            d = f"M {x0:.2f} {y0:.2f} C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {x3:.2f} {y3:.2f}"
            pts = [_cubic((x0, y0), c1, c2, (x3, y3), t / 20) for t in range(21)]
            ux, uy = _unit(x3 - pts[-2][0], y3 - pts[-2][1])
            return d, pts, (x3, y3, ux, uy)
        dx, dy = b.x - a.x, b.y - a.y
        dist = math.hypot(dx, dy) or 1
        ux, uy = dx / dist, dy / dist
        if e.bend == 0:
            x1, y1 = a.x + ux * a.r, a.y + uy * a.r
            x2, y2 = b.x - ux * b.r, b.y - uy * b.r
            if e.directed:
                x2, y2 = x2 - ux * 1.5, y2 - uy * 1.5
            d = f"M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}"
            pts = [(x1 + (x2 - x1) * t / 10, y1 + (y2 - y1) * t / 10) for t in range(11)]
            return d, pts, (x2, y2, ux, uy)
        px, py = -uy, ux
        mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
        cx, cy = mx + px * e.bend * 2, my + py * e.bend * 2  # apex sits at `bend`
        full = [_quad((a.x, a.y), (cx, cy), (b.x, b.y), t / 200) for t in range(201)]
        # trim to node borders
        i0 = next(i for i, p in enumerate(full) if math.hypot(p[0] - a.x, p[1] - a.y) >= a.r)
        rb = b.r + (1.5 if e.directed else 0)
        i1 = max(i for i, p in enumerate(full) if math.hypot(p[0] - b.x, p[1] - b.y) >= rb)
        seg = full[i0 : i1 + 1]
        d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in seg[::4] + [seg[-1]])
        ex, ey = seg[-1]
        tx, ty = _unit(ex - seg[-3][0], ey - seg[-3][1])
        return d, seg, (ex, ey, tx, ty)

    def edge_point(self, e: Edge, t: float) -> tuple[float, float]:
        _, pts, _ = self.edge_geometry(e)
        i = min(len(pts) - 1, max(0, round(t * (len(pts) - 1))))
        return pts[i]

    def _render_edge(self, e: Edge, theme: Theme) -> str:
        stroke, op, _, sw, dash = _paint(theme, e.style)
        if e.style == "default":
            sw = STROKE
        elif e.style == "soft":
            # input edges: thinner, lighter accent so they never read as results
            sw, op = STROKE, 0.7
        if e.width is not None:
            sw = e.width
        d, pts, (ex, ey, ux, uy) = self.edge_geometry(e)
        if e.directed:
            # shorten the line so it ends under the arrowhead
            hl = 8 + sw
            d_line = _shorten_path(pts, hl * 0.7)
        else:
            d_line = d
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out = [f'<path d="{d_line}" fill="none" stroke="{stroke}" stroke-opacity="{op}" '
               f'stroke-width="{sw}" stroke-linecap="round"{dash_attr}/>']
        if e.directed:
            hl, hw = 8 + sw, 5.5 + sw
            bx, by = ex - ux * hl, ey - uy * hl
            px, py = -uy, ux
            out.append(f'<polygon points="{ex:.2f},{ey:.2f} {bx + px * hw / 2:.2f},{by + py * hw / 2:.2f} '
                       f'{bx - px * hw / 2:.2f},{by - py * hw / 2:.2f}" fill="{stroke}" fill-opacity="{op}"/>')
        if e.weight is not None:
            wx, wy = self.edge_point(e, e.weight_t)
            tw = text_width(e.weight, SIZE_NOTE) + 6
            c = stroke
            out.append(f'<rect x="{wx - tw / 2:.2f}" y="{wy - 8:.2f}" width="{tw:.2f}" height="16" rx="3" '
                       f'fill="{theme.paper}"/>')
            out.append(f'<text x="{wx:.2f}" y="{wy:.2f}" text-anchor="middle" dominant-baseline="central" '
                       f'font-family="{FONT_MONO}" font-size="{SIZE_NOTE}" fill="{c}" '
                       f'fill-opacity="{max(op, 0.55) if e.style in ("muted", "removed") else 1}">'
                       f"{_esc(e.weight)}</text>")
        return "\n".join(out)

    def weight_texts(self) -> list[Text]:
        out = []
        for i, e in enumerate(self.edges):
            if e.weight is not None:
                wx, wy = self.edge_point(e, e.weight_t)
                out.append(Text(wx, wy, e.weight, size=SIZE_NOTE, role="weight", owner=f"edge{i}"))
        return out


# --- rendering helpers -----------------------------------------------------


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


HEAT_MAX = 0.55  # strongest heat fill; keeps labels legible in both themes


def heat_fill(theme: Theme, heat: float) -> str:
    return _mix(theme.paper, theme.accent, HEAT_MAX * max(0.0, min(1.0, heat)))


def _render_node(n: Node, theme: Theme) -> str:
    stroke, op, fill, sw, dash = _paint(theme, n.style)
    if n.heat is not None:
        fill = heat_fill(theme, n.heat)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out = [f'<circle cx="{n.x:.1f}" cy="{n.y:.1f}" r="{n.r}" fill="{fill}" stroke="{stroke}" '
           f'stroke-opacity="{op}" stroke-width="{sw}"{dash_attr}/>']
    if n.label:
        size = SIZE_LABEL if n.r >= R else SIZE_LABEL_SMALL
        if n.heat is not None:
            col, lop = theme.ink, 1
        elif n.style in ("muted", "removed"):
            col, lop = theme.ink, 0.5
        elif n.style in ("accent", "soft", "ghost"):
            col, lop = theme.accent, 1
        elif n.style in ("g2", "g3", "g4"):
            col, lop = _color(theme, n.style), 1
        else:
            col, lop = theme.ink, 1
        out.append(f'<text x="{n.x:.1f}" y="{n.y:.1f}" text-anchor="middle" dominant-baseline="central" '
                   f'font-family="{FONT_SANS}" font-size="{size}" font-weight="600" fill="{col}" '
                   f'fill-opacity="{lop}">{_esc(n.label)}</text>')
    return "\n".join(out)


def _render_text(t: Text, theme: Theme) -> str:
    fam = FONT_MONO if t.mono else FONT_SANS
    col = _color(theme, t.color)
    style = ' font-style="italic"' if t.italic else ""
    db = ' dominant-baseline="central"' if t.baseline == "central" else ""
    out = ""
    if t.boxed:
        x0, y0, x1, y1 = t.bbox()
        out = (f'<rect x="{x0 - 3:.1f}" y="{y0 - 2:.1f}" width="{x1 - x0 + 6:.1f}" '
               f'height="{y1 - y0 + 4:.1f}" rx="3" fill="{theme.paper}"/>\n')
    return out + (f'<text x="{t.x:.1f}" y="{t.y:.1f}" text-anchor="{t.anchor}"{db} font-family="{fam}" '
                  f'font-size="{t.size}" font-weight="{t.weight}"{style} fill="{col}" '
                  f'fill-opacity="{t.opacity}">{_esc(t.text)}</text>')


def _render_rect(r: Rect, theme: Theme) -> str:
    if r.solid:
        fill = _color(theme, r.fill_color)
    elif r.fill == "paper":
        fill = theme.paper
    elif r.fill == "tint":
        fill = _tint(theme, r.fill_color, 0.8)
    else:
        fill = "none"
    stroke = (f' stroke="{_color(theme, r.stroke)}" stroke-opacity="{r.stroke_opacity}" stroke-width="1"'
              if r.stroke else "")
    dash = f' stroke-dasharray="{r.dash}"' if r.dash else ""
    fo = f' fill-opacity="{r.fill_opacity}"' if r.fill_opacity != 1 else ""
    return (f'<rect x="{r.x:.1f}" y="{r.y:.1f}" width="{r.w:.1f}" height="{r.h:.1f}" rx="{r.rx}" '
            f'fill="{fill}"{fo}{stroke}{dash}/>')


def _render_line(ln: Line, theme: Theme) -> str:
    c = _color(theme, ln.color)
    dash = f' stroke-dasharray="{ln.dash}"' if ln.dash else ""
    if not ln.arrow:
        return (f'<line x1="{ln.x1:.1f}" y1="{ln.y1:.1f}" x2="{ln.x2:.1f}" y2="{ln.y2:.1f}" stroke="{c}" '
                f'stroke-opacity="{ln.opacity}" stroke-width="{ln.width}" stroke-linecap="round"{dash}/>')
    ux, uy = _unit(ln.x2 - ln.x1, ln.y2 - ln.y1)
    hl, hw = 9, 9
    bx, by = ln.x2 - ux * hl, ln.y2 - uy * hl
    px, py = -uy, ux
    return (f'<line x1="{ln.x1:.1f}" y1="{ln.y1:.1f}" x2="{bx + ux:.1f}" y2="{by + uy:.1f}" stroke="{c}" '
            f'stroke-opacity="{ln.opacity}" stroke-width="{ln.width}" stroke-linecap="round"{dash}/>\n'
            f'<polygon points="{ln.x2:.1f},{ln.y2:.1f} {bx + px * hw / 2:.1f},{by + py * hw / 2:.1f} '
            f'{bx - px * hw / 2:.1f},{by - py * hw / 2:.1f}" fill="{c}" fill-opacity="{ln.opacity}"/>')


def _render_path(p: Path, theme: Theme) -> str:
    c = _color(theme, p.color)
    dash = f' stroke-dasharray="{p.dash}"' if p.dash else ""
    fill = c if p.fill == "color" else ("none" if p.fill == "none" else p.fill)
    return (f'<path d="{p.d}" fill="{fill}" stroke="{c}" stroke-opacity="{p.opacity}" '
            f'stroke-width="{p.width}" stroke-linecap="round" stroke-linejoin="round"{dash}/>')


# --- geometry --------------------------------------------------------------


def _unit(dx: float, dy: float) -> tuple[float, float]:
    d = math.hypot(dx, dy) or 1
    return dx / d, dy / d


def _quad(p0, p1, p2, t):
    u = 1 - t
    return (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])


def _cubic(p0, p1, p2, p3, t):
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1])


def _shorten_path(pts: list[tuple[float, float]], amount: float) -> str:
    """Polyline `d` with `amount` px removed from the end."""
    out = list(pts)
    remaining = amount
    while len(out) >= 2:
        (x0, y0), (x1, y1) = out[-2], out[-1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg > remaining:
            t = (seg - remaining) / seg
            out[-1] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            break
        remaining -= seg
        out.pop()
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in out)


def _convex_hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


# --- panel boxes & templates ----------------------------------------------


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


ARROW_Y = 98  # vertical center of the content area; panel arrows sit here
_TOP, _BOT = CONTENT_TOP, CONTENT_BOTTOM


class Layout:
    """Fixed panel geometry. Every figure uses exactly one of these.

    transform   graph  →  graph           mutations, transforms, extractions
    compute     graph  →  result card     metrics, queries, lists, tables
    contrast    graph  |  graph           predicates (True vs. False example)
    binary      A  ∘  B  =  C             set algebra on two graphs
    multiples   graph  graph  graph       generators: one shape at growing sizes
    single      one wide graph            rare; when one picture says it all
    """

    TRANSFORM = (Box(MARGIN, _TOP, 352, _BOT), Box(408, _TOP, W - MARGIN, _BOT))
    TRANSFORM_ARROW = (362, 398)

    COMPUTE = (Box(MARGIN, _TOP, 440, _BOT), Box(500, _TOP, W - MARGIN, _BOT))
    COMPUTE_ARROW = (452, 488)

    CONTRAST = (Box(MARGIN, _TOP, 372, _BOT), Box(388, _TOP, W - MARGIN, _BOT))
    CONTRAST_DIVIDER = 380

    BINARY = (Box(MARGIN, _TOP, 232, _BOT), Box(270, _TOP, 490, _BOT), Box(528, _TOP, W - MARGIN, _BOT))
    BINARY_SYMBOLS = (251, 509)

    MULTIPLES = (Box(MARGIN, _TOP, 252, _BOT), Box(260, _TOP, 500, _BOT), Box(508, _TOP, W - MARGIN, _BOT))

    SINGLE = Box(MARGIN, _TOP, W - MARGIN, _BOT)


def transform(s: Scene, left_caption: str, right_caption: str) -> tuple[Box, Box]:
    a, b = Layout.TRANSFORM
    s.arrow(Layout.TRANSFORM_ARROW[0], ARROW_Y, Layout.TRANSFORM_ARROW[1])
    s.caption(a.cx, left_caption, max_w=a.w)
    s.caption(b.cx, right_caption, max_w=b.w)
    return a, b


def compute(s: Scene, left_caption: str, right_caption: str = "") -> tuple[Box, Box]:
    a, b = Layout.COMPUTE
    s.arrow(Layout.COMPUTE_ARROW[0], ARROW_Y, Layout.COMPUTE_ARROW[1])
    s.caption(a.cx, left_caption, max_w=a.w)
    if right_caption:
        # Long calls (count_strongly_connected_components()...) may spill left
        # into the empty caption-row space under the arrow, right edge pinned
        # to the margin; short ones stay centered under the result.
        w = text_width(right_caption, SIZE_CAPTION)
        x = min(b.cx, W - MARGIN - w / 2)
        s.caption(x, right_caption, max_w=W - MARGIN - a.x1)
    return a, b


def contrast(s: Scene, left_caption: str, right_caption: str,
             left_result: str = "True", right_result: str = "False") -> tuple[Box, Box]:
    """Predicate pair. Captions read like `is_tree() → True`."""
    a, b = Layout.CONTRAST
    s.divider(Layout.CONTRAST_DIVIDER)
    _result_caption(s, a.cx, left_caption, left_result, True)
    _result_caption(s, b.cx, right_caption, right_result, False)
    return a, b


def _result_caption(s: Scene, cx: float, call: str, result: str, positive: bool) -> None:
    full = f"{call} → {result}" if call else result
    tw = text_width(full, SIZE_CAPTION)
    x0 = cx - tw / 2
    head = f"{call} → " if call else ""
    if head:
        s.text(x0, CAPTION_Y, head, size=SIZE_CAPTION, anchor="start", opacity=0.72, role="caption")
    s.text(x0 + text_width(head, SIZE_CAPTION), CAPTION_Y, result, size=SIZE_CAPTION, anchor="start",
           color="accent" if positive else "ink", weight=600, role="caption")


def binary(s: Scene, op: str, captions: tuple[str, str, str]) -> tuple[Box, Box, Box]:
    a, b, c = Layout.BINARY
    s.symbol(Layout.BINARY_SYMBOLS[0], ARROW_Y, op)
    s.symbol(Layout.BINARY_SYMBOLS[1], ARROW_Y, "=")
    for box, cap in zip((a, b, c), captions):
        s.caption(box.cx, cap, max_w=box.w)
    return a, b, c


def multiples(s: Scene, captions: tuple[str, str, str]) -> tuple[Box, Box, Box]:
    boxes = Layout.MULTIPLES
    for box, cap in zip(boxes, captions):
        s.caption(box.cx, cap, max_w=box.w)
    return boxes


def single(s: Scene, caption: str) -> Box:
    box = Layout.SINGLE
    s.caption(box.cx, caption, max_w=box.w)
    return box


# --- placement helpers -----------------------------------------------------

UNIT = 64  # preferred distance between adjacent grid positions


def place(coords: dict[str, tuple[float, float]], box: Box, *, unit: float = UNIT,
          r: float = R, pad: float = 6) -> dict[str, tuple[float, float]]:
    """Map abstract layout coords (any units) into `box`, centered.

    Abstract coords are multiplied by `unit` px; if the result would not fit
    (accounting for node radius + pad), the unit shrinks uniformly. Use the
    same unit across a family so graphs read at the same density.
    """
    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    sx, sy = (max(xs) - min(xs)) or 0, (max(ys) - min(ys)) or 0
    avail_w = box.w - 2 * (r + pad)
    avail_h = box.h - 2 * (r + pad)
    u = unit
    if sx * u > avail_w:
        u = avail_w / sx
    if sy * u > avail_h:
        u = avail_h / sy
    mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return {k: (round(box.cx + (x - mx) * u, 1), round(box.cy + (y - my) * u, 1))
            for k, (x, y) in coords.items()}


def ring(ids: list[str], radius: float = 1.0, start_deg: float = -90,
         center: tuple[float, float] = (0, 0)) -> dict[str, tuple[float, float]]:
    """Abstract coords for nodes evenly spaced on a circle (clockwise)."""
    n = len(ids)
    return {k: (center[0] + radius * math.cos(math.radians(start_deg + 360 * i / n)),
                center[1] + radius * math.sin(math.radians(start_deg + 360 * i / n)))
            for i, k in enumerate(ids)}


def shift(coords: dict[str, tuple[float, float]], dx: float, dy: float = 0) -> dict[str, tuple[float, float]]:
    return {k: (x + dx, y + dy) for k, (x, y) in coords.items()}


# --- registry --------------------------------------------------------------

Family = Literal[
    "mutation", "selection", "traversal", "paths", "node-metrics", "graph-metrics",
    "similarity", "communities", "predicates", "connectivity", "substructures",
    "generators", "data", "transforms", "set-ops", "diffusion", "enrichment",
]


@dataclass
class Figure:
    method: str  # Graph method name; also the file slug
    family: Family
    title: str
    alt: str
    build: Callable[[Scene], None]
    template: str = ""

    @property
    def slug(self) -> str:
        return self.method.replace("_", "-")

    def scene(self) -> Scene:
        s = Scene()
        self.build(s)
        return s


REGISTRY: list[Figure] = []


def figure(method: str, family: Family, title: str, alt: str):
    """Decorator registering a figure builder: `def build(s: Scene) -> None`."""

    def deco(fn: Callable[[Scene], None]) -> Callable[[Scene], None]:
        if any(f.method == method for f in REGISTRY):
            raise ValueError(f"duplicate figure for {method!r}")
        REGISTRY.append(Figure(method, family, title, alt, fn))
        return fn

    return deco
