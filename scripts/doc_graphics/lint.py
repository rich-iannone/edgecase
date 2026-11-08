"""Geometric lint for doc-graphics scenes.

Catches the defects that make a figure look careless: text colliding with
nodes, edges, or other text; elements leaving the canvas; nodes overlapping;
edges passing *through* unrelated nodes; off-palette font sizes; and stock
"LLM numbers" like 42.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from core import (
    CAPTION_Y,
    CONTENT_BOTTOM,
    MARGIN,
    TEXT_SIZES,
    BBox,
    H,
    Scene,
    Text,
    W,
    text_width,
)

BANNED_NUMBERS = re.compile(r"(?<![\d.])42(?![\d])")


@dataclass
class Issue:
    rule: str
    msg: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.msg}"


def _overlap(a: BBox, b: BBox, gap: float = 0) -> bool:
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def _circle_box(cx, cy, r, b: BBox, gap: float = 0) -> bool:
    nx = min(max(cx, b[0]), b[2])
    ny = min(max(cy, b[1]), b[3])
    return math.hypot(cx - nx, cy - ny) < r + gap


def _seg_box(p, q, b: BBox, gap: float = 0) -> bool:
    x0, y0, x1, y1 = b[0] - gap, b[1] - gap, b[2] + gap, b[3] + gap
    # sample; segments here are short and boxes are small
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 2))
    for i in range(n + 1):
        t = i / n
        x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _pt_seg_dist(px, py, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    if L == 0:
        return math.hypot(px - a[0], py - a[1])
    t = max(0, min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / L))
    return math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))


def lint(scene: Scene) -> list[Issue]:
    issues: list[Issue] = []
    ign = scene._lint_ignore

    def add(rule: str, msg: str) -> None:
        if rule not in ign:
            issues.append(Issue(rule, msg))

    texts: list[Text] = list(scene.texts) + scene.weight_texts()
    nodes = scene.nodes
    edge_polys = []
    for i, e in enumerate(scene.edges):
        if e.a not in nodes or e.b not in nodes:
            add("dangling-edge", f"edge {e.a}->{e.b} references a missing node")
            continue
        edge_polys.append((i, e, scene.edge_geometry(e)[1]))

    # banned numbers / font sizes / non-empty
    for t in texts:
        if BANNED_NUMBERS.search(t.text):
            add("no-42", f"text {t.text!r} uses 42")
        if t.size not in TEXT_SIZES:
            add("font-size", f"text {t.text!r} uses off-scale size {t.size}")
    for n in nodes.values():
        if BANNED_NUMBERS.search(n.label):
            add("no-42", f"node label {n.label!r} uses 42")
        if n.label:
            size = 13 if n.r >= 16 else 10
            if text_width(n.label, size, mono=False, weight=600) > n.r * 2 - 4:
                add("label-fit", f"node label {n.label!r} is too wide for r={n.r}")

    # canvas bounds
    def inside(b: BBox, what: str, m: float = MARGIN - 4) -> None:
        if b[0] < m or b[1] < m or b[2] > W - m or b[3] > H - m:
            add("bounds", f"{what} leaves the canvas: {tuple(round(v) for v in b)}")

    for n in nodes.values():
        inside((n.x - n.r, n.y - n.r, n.x + n.r, n.y + n.r), f"node {n.id}")
        if n.y + n.r > CONTENT_BOTTOM + 4:
            add("caption-zone", f"node {n.id} intrudes on the caption row")
    for t in texts:
        inside(t.bbox(), f"text {t.text!r}")
    for r in scene.rects:
        inside(r.bbox(), f"{r.role} rect")
        if r.role in ("card", "table") and r.y + r.h > CONTENT_BOTTOM + 4:
            add("caption-zone", f"{r.role} intrudes on the caption row")
    for _, e, pts in edge_polys:
        for x, y in pts:
            if x < MARGIN - 4 or x > W - MARGIN + 4 or y < MARGIN - 4 or y > H - MARGIN + 4:
                add("bounds", f"edge {e.a}->{e.b} leaves the canvas")
                break

    # node-node spacing
    nl = list(nodes.values())
    for i in range(len(nl)):
        for j in range(i + 1, len(nl)):
            a, b = nl[i], nl[j]
            if math.hypot(a.x - b.x, a.y - b.y) < a.r + b.r + 8:
                add("node-overlap", f"nodes {a.id} and {b.id} are too close")

    # edge passing through a non-endpoint node
    for _, e, pts in edge_polys:
        for n in nl:
            if n.id in (e.a, e.b):
                continue
            for p, q in zip(pts, pts[1:]):
                if _pt_seg_dist(n.x, n.y, p, q) < n.r + 3:
                    add("edge-through-node", f"edge {e.a}->{e.b} passes through node {n.id}")
                    break

    # text collisions
    for t in texts:
        tb = t.bbox()
        for n in nl:
            if t.role == "label":
                continue
            if t.owner == n.id and t.role == "badge":
                # badge may touch its own node but not cover its label
                if _circle_box(n.x, n.y, n.r * 0.55, tb):
                    add("text-node", f"badge {t.text!r} covers node {n.id}'s label")
                continue
            if _circle_box(n.x, n.y, n.r, tb, gap=2):
                add("text-node", f"text {t.text!r} collides with node {n.id}")
        for i, e, pts in edge_polys:
            if t.role == "weight" and t.owner == f"edge{i}":
                continue
            if t.boxed:
                continue
            for p, q in zip(pts, pts[1:]):
                if _seg_box(p, q, tb, gap=1):
                    add("text-edge", f"text {t.text!r} collides with edge {e.a}->{e.b}")
                    break
    # text vs containers
    containers = [r for r in scene.rects if r.role in ("card", "table", "bar", "frame", "region")]
    for t in texts:
        tb = t.bbox()
        for r in containers:
            rb = r.bbox()
            contained = tb[0] >= rb[0] + 2 and tb[2] <= rb[2] - 2 and tb[1] >= rb[1] and tb[3] <= rb[3]
            if t.role in ("card", "table") and r.role in ("card", "table"):
                if _overlap(tb, rb) and not contained:
                    add("text-overflow", f"text {t.text!r} overflows its {r.role}")
            elif r.role in ("frame", "region"):
                continue  # labels may sit inside decorative frames
            elif _overlap(tb, rb, gap=1):
                add("text-rect", f"text {t.text!r} collides with a {r.role}")

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            # caption fragments on one baseline are laid out to abut exactly
            gap = -1 if a.role == b.role == "caption" and a.y == b.y else 1
            if _overlap(a.bbox(), b.bbox(), gap=gap):
                add("text-text", f"texts {a.text!r} and {b.text!r} overlap")

    # cards/tables/bars must not sit on the graph
    for r in scene.rects:
        if r.role not in ("card", "table", "bar"):
            continue
        rb = r.bbox()
        for n in nl:
            if _circle_box(n.x, n.y, n.r, rb, gap=4):
                add("rect-node", f"{r.role} collides with node {n.id}")
        for _, e, pts in edge_polys:
            for p, q in zip(pts, pts[1:]):
                if _seg_box(p, q, rb, gap=2):
                    add("rect-edge", f"{r.role} collides with edge {e.a}->{e.b}")
                    break

    # panel arrows must be clear of everything
    for ln in scene.lines:
        if ln.role != "panel_arrow":
            continue
        ab = (min(ln.x1, ln.x2) - 2, ln.y1 - 6, max(ln.x1, ln.x2) + 2, ln.y1 + 6)
        for n in nl:
            if _circle_box(n.x, n.y, n.r, ab, gap=6):
                add("arrow-clear", f"panel arrow touches node {n.id}")
        for _, e, pts in edge_polys:
            for p, q in zip(pts, pts[1:]):
                if _seg_box(p, q, ab, gap=2):
                    add("arrow-clear", f"panel arrow touches edge {e.a}->{e.b}")
                    break
        for t in texts:
            if _overlap(t.bbox(), ab):
                add("arrow-clear", f"panel arrow touches text {t.text!r}")

    # captions only on the caption row, and no wider than their panel
    for t in scene.texts:
        if t.role == "caption" and abs(t.y - CAPTION_Y) > 0.5:
            add("caption-row", f"caption {t.text!r} is off the caption baseline")
        if t.role == "caption" and t.max_w is not None:
            if text_width(t.text, t.size) > t.max_w:
                add("caption-width", f"caption {t.text!r} is wider than its panel; shorten it")

    seen, out = set(), []
    for it in issues:
        k = str(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out
