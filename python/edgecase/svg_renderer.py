"""Static SVG graph renderer for publication-quality small graphs.

Produces clean SVG with optional hover interactivity, visually consistent with Cosmograph's
aesthetic: filled circles, thin semi-transparent edges, minimal chrome.
"""

from __future__ import annotations

import dataclasses
import hashlib
import html as html_mod
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    edge_color: str
    label_color: str
    tooltip_bg: str
    tooltip_fg: str


LIGHT = Theme(
    name="light",
    bg="#fafafa",
    edge_color="#94a3b8",
    label_color="#374151",
    tooltip_bg="rgba(15,23,42,0.88)",
    tooltip_fg="#f1f5f9",
)

DARK = Theme(
    name="dark",
    bg="#0d1117",
    edge_color="#4b5563",
    label_color="#d1d5db",
    tooltip_bg="rgba(241,245,249,0.92)",
    tooltip_fg="#0f172a",
)

THEMES = {"light": LIGHT, "dark": DARK}
THEME_NAMES = ("auto", "light", "dark")
LAYOUTS = ("force", "circular", "random")
LAYOUT_MODES = ("progressive", "stable")


def _check_style(
    layout: str, theme: Optional[str], layout_mode: Optional[str] = None, stored_ok: bool = False
) -> None:
    """Raise `ValueError` for an unknown `layout=`, `theme=` or `layout_mode=` (they used to
    fall back silently to "force", "light" and "progressive"-like behavior).

    With `stored_ok=True`, any other non-empty string is taken as the name of a node attribute
    holding positions; `_stored_layout()` checks it against the graph."""
    if layout not in LAYOUTS and not (stored_ok and isinstance(layout, str) and layout):
        raise ValueError(f"Unknown layout {layout!r}; use one of {', '.join(map(repr, LAYOUTS))}")
    if theme is not None and theme not in THEME_NAMES:
        raise ValueError(f"Unknown theme {theme!r}; use 'auto', 'light' or 'dark'")
    if layout_mode is not None and layout_mode not in LAYOUT_MODES:
        raise ValueError(f"Unknown layout_mode {layout_mode!r}; use 'progressive' or 'stable'")


# Pages that say which color scheme they show, which overrides the system setting: Quarto (and
# great-docs) and Bootstrap pages, JupyterLab and Notebook 7, VS Code notebooks, and Colab
_LIGHT_PAGES = (
    'html.quarto-light, body.quarto-light, html[data-bs-theme="light"], '
    '[data-jp-theme-light="true"], body.vscode-light, body.vscode-high-contrast-light, '
    'html[theme="light"]'
)
_DARK_PAGES = (
    'html.quarto-dark, body.quarto-dark, html[data-bs-theme="dark"], '
    '[data-jp-theme-light="false"], body.vscode-dark, '
    "body.vscode-high-contrast:not(.vscode-high-contrast-light), "
    'html[theme="dark"]'
)
_VAR_RE = re.compile(r"var\((--ect-[0-9a-f]{8})\)")


class _Colors:
    """The colors of one rendering in the light, dark or automatic theme.

    `pick(light, dark)` gives the color (or opacity) to use. For a fixed theme, that is one of the
    two values. For `"auto"`, it is a CSS variable that `finish()` defines twice, with the light
    value and with the dark one, choosing between them by the page's color scheme where the page
    sets one (see `_LIGHT_PAGES` and `_DARK_PAGES`), or else by the system's. One rendering then
    suits both, and follows the page when its reader switches between them.

    `theme=None` is `"auto"` on screen and `"light"` in saved files, since programs other than
    browsers don't apply CSS variables.
    """

    def __init__(self, theme: Optional[str]) -> None:
        self.auto = theme in (None, "auto")
        self.dark = theme == "dark"
        self.save_light = theme is None
        self._vars: Dict[Tuple[Any, Any], str] = {}
        if self.auto:
            self.theme = Theme(
                name="auto",
                **{
                    f.name: self.pick(getattr(LIGHT, f.name), getattr(DARK, f.name))
                    for f in dataclasses.fields(Theme)
                    if f.name != "name"
                },
            )
        else:
            self.theme = DARK if self.dark else LIGHT

    def pick(self, light, dark):
        if not self.auto:
            return dark if self.dark else light
        if light == dark:
            return light
        # Named after its values, so that the definitions of every rendering on a page agree
        name = self._vars.setdefault(
            (light, dark), "--ect-" + hashlib.sha1(f"{light}|{dark}".encode()).hexdigest()[:8]
        )
        return f"var({name})"

    def tint(self, color: str, alpha: str = "22") -> str:
        """`color` (a hex color or a variable from `pick()`) at the opacity of the hex `alpha`."""
        if color.startswith("#"):
            return color + alpha
        return f"color-mix(in srgb, {color} {int(alpha, 16) / 2.55:.0f}%, transparent)"

    def scheme(self) -> str:
        """The `color-scheme` for the browser-drawn parts (sliders, menus, scroll bars)."""
        return self.pick("light", "dark")

    def _css(self) -> str:
        light = ";".join(f"{name}:{lt}" for (lt, _), name in self._vars.items())
        dark = ";".join(f"{name}:{dk}" for (_, dk), name in self._vars.items())
        # Each rule outranks the ones before it whatever the order of the renderings on the page
        # (by its specificity), and the page's scheme outranks the system's. Dark comes last, so
        # that it wins should a page somehow carry both markers
        return (
            f":where(.ec-theme){{{light}}}"
            f"@media (prefers-color-scheme: dark){{.ec-theme{{{dark}}}}}"
            f":is({_LIGHT_PAGES}) .ec-theme{{{light}}}"
            f":is({_DARK_PAGES}) .ec-theme{{{dark}}}"
        )

    def finish(self, svg_str: str, html_str: str, output: Optional[str]) -> "StepPlayer":
        """The `StepPlayer` for the rendering, saved to `output=` when that is given."""
        for_files = None
        if self.save_light:
            values = {name: lt for (lt, _), name in self._vars.items()}
            for_files = tuple(
                _VAR_RE.sub(lambda m: str(values[m.group(1)]), markup)
                for markup in (svg_str, html_str)
            )
        if self.auto and self._vars:
            style = f'<style class="ec-theme-vars">{self._css()}</style>'
            # The SVG carries its own definitions, so that it works on its own, too
            head_end = svg_str.index(">", svg_str.index("<svg")) + 1
            head = svg_str[:head_end]
            if ' class="' in head:
                head = head.replace(' class="', ' class="ec-theme ', 1)
            else:
                head = head.replace("<svg", '<svg class="ec-theme"', 1)
            svg_str = head + style + svg_str[head_end:]
            html_str = f'<div class="ec-theme" style="display:contents">{style}{html_str}</div>'
        result = StepPlayer(svg_str, html_str, for_files)
        if output:
            result.save(output)
        return result


# Default node color when no color_by mapping
DEFAULT_NODE_COLOR = "#4e79a7"

# Font
FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
)


# ---------------------------------------------------------------------------
# StepPlayer result object
# ---------------------------------------------------------------------------


class StepPlayer:
    """Interactive graph visualization result.

    Returned by `Graph.play()`, `Graph.plot()`, and `Graph.plot_traversal()`. Renders inline in
    Jupyter via `_repr_html_` and can be saved with `.save()`.
    """

    def __init__(
        self, svg_str: str, html_str: str, for_files: Optional[Tuple[str, str]] = None
    ) -> None:
        self._svg = svg_str
        self._html = html_str
        # The SVG and HTML that `save()` writes instead, if they differ (for the default theme)
        self._for_files = for_files

    @property
    def svg(self) -> str:
        return self._svg

    def _repr_svg_(self) -> str:
        return self._svg

    def _repr_html_(self) -> str:
        return self._html

    def save(self, path: str) -> None:
        p = Path(path)
        svg_str, html_str = self._for_files or (self._svg, self._html)
        content = html_str if p.suffix == ".html" else svg_str
        p.write_text(content, encoding="utf-8")

    def __repr__(self) -> str:
        return f"StepPlayer({len(self._svg)} chars)"


# ---------------------------------------------------------------------------
# Layout algorithms
# ---------------------------------------------------------------------------


def _circle_order(nodes: list, edges) -> list:
    """The nodes in the order a depth-first search meets them, for going around a circle.

    `nodes` come in the preferred order: each search starts from the first node not yet met and
    moves on to neighbors in that order. Edge direction is ignored. Linked nodes then tend to sit
    next to each other: a tree has no crossing edges, and a ring numbered in order stays a ring.
    """
    rank = {v: i for i, v in enumerate(nodes)}
    adj: Dict[Any, set] = {v: set() for v in nodes}
    for s, t in edges:
        if s != t and s in adj and t in adj:
            adj[s].add(t)
            adj[t].add(s)
    seen: set = set()
    order = []
    for root in nodes:
        stack = [root]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            order.append(v)
            # pushed last-first, so the earliest neighbor is visited next
            stack.extend(sorted(adj[v] - seen, key=rank.__getitem__, reverse=True))
    return order


def _circular_layout(
    node_ids: List[str],
    width: float,
    height: float,
    padding: float = 60,
    edges: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Tuple[float, float]]:
    n = len(node_ids)
    if n == 0:
        return {}
    if n == 1:
        return {node_ids[0]: (width / 2, height / 2)}

    # Node IDs come in storage order, which isn't the order the nodes were added, so start from
    # ID order, and with the edges go around the circle in depth-first order from there, as
    # `layout_circular()` does, so that linked nodes sit near each other
    if all(nid.isdigit() for nid in node_ids):
        node_ids = sorted(node_ids, key=int)
    if edges is not None:
        node_ids = _circle_order(node_ids, edges)
    cx, cy = width / 2, height / 2
    r = min(width, height) / 2 - padding
    positions = {}
    for i, nid in enumerate(node_ids):
        angle = 2 * math.pi * i / n - math.pi / 2
        positions[nid] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
    return positions


def _random_layout(
    node_ids: List[str],
    width: float,
    height: float,
    padding: float = 60,
    seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    rng = random.Random(seed)
    positions = {}
    for nid in node_ids:
        x = rng.uniform(padding, width - padding)
        y = rng.uniform(padding, height - padding)
        positions[nid] = (x, y)
    return positions


def _stored_layout(
    graph,
    node_ids: list,
    attr: str,
    width: float,
    height: float,
    padding: float = 60,
) -> Dict[str, Tuple[float, float]]:
    """Canvas positions from a node attribute holding `(x, y)` coordinates.

    The coordinates are fitted to the canvas with their aspect ratio kept, and `y` points up
    (as in the `layout_*()` methods and NetworkX drawings). Raises `ValueError` if no node has
    the attribute, or if some node lacks it or holds something other than two finite numbers.
    """
    raw: Dict[str, Tuple[float, float]] = {}
    bad = []
    for nid in node_ids:
        obj = graph.get_node(nid)
        value = obj.attributes.get(attr) if obj is not None and obj.attributes else None
        try:
            x, y = (float(v) for v in value)
            if isinstance(value, (str, bytes)) or not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError
        except (TypeError, ValueError):
            bad.append((nid, value))
            continue
        raw[str(nid)] = (x, y)
    if not raw:
        raise ValueError(
            f"Unknown layout {attr!r}; use one of {', '.join(map(repr, LAYOUTS))}, or the name "
            "of a node attribute holding (x, y) positions (set by the layout_*() methods)"
        )
    if bad:
        nid, value = bad[0]
        raise ValueError(
            f"Layout attribute {attr!r} must hold an (x, y) pair of finite numbers on every "
            f"node; {len(bad)} of {len(node_ids)} nodes don't (node {nid!r} has {value!r})"
        )

    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    range_x = (max(xs) - min(xs)) or 1.0
    range_y = (max(ys) - min(ys)) or 1.0
    scale = min((width - 2 * padding) / range_x, (height - 2 * padding) / range_y)
    mid_x, mid_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return {
        nid: (width / 2 + (x - mid_x) * scale, height / 2 - (y - mid_y) * scale)
        for nid, (x, y) in raw.items()
    }


def _fruchterman_reingold_python(
    n: int,
    edges: List[Tuple[int, int]],
    iterations: int = 300,
    seed: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """Pure-Python Fruchterman-Reingold force-directed layout."""
    if n == 0:
        return {}
    if n == 1:
        return {0: (0.0, 0.0)}

    # Scale area and ideal distance with graph size for better cluster separation
    area = float(n) * 400.0
    k = math.sqrt(area / n)
    # Boost repulsion for larger graphs to push communities apart
    repulsion_k2 = k * k * (1.0 + max(0, n - 30) / 50.0)

    spread = area * 0.4
    rng = random.Random(seed)
    pos_x = [rng.uniform(-spread, spread) for _ in range(n)]
    pos_y = [rng.uniform(-spread, spread) for _ in range(n)]

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    temp = area * 0.15
    cool = temp / (iterations + 1)
    # Weak gravity pulls toward center, preventing drift while allowing separation
    gravity = 0.6

    for it in range(iterations):
        disp_x = [0.0] * n
        disp_y = [0.0] * n

        # Repulsive forces (all pairs — use grid optimization for large n)
        if n <= 500:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos_x[i] - pos_x[j]
                    dy = pos_y[i] - pos_y[j]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 0.01:
                        dist = 0.01
                    force = repulsion_k2 / dist
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    disp_x[i] += fx
                    disp_y[i] += fy
                    disp_x[j] -= fx
                    disp_y[j] -= fy
        else:
            # Grid-based approximation for larger graphs
            cell_size = k * 2.5
            grid: Dict[Tuple[int, int], List[int]] = {}
            for i in range(n):
                cx = int(pos_x[i] / cell_size)
                cy = int(pos_y[i] / cell_size)
                key = (cx, cy)
                if key not in grid:
                    grid[key] = []
                grid[key].append(i)

            for i in range(n):
                cx = int(pos_x[i] / cell_size)
                cy = int(pos_y[i] / cell_size)
                for dcx in range(-2, 3):
                    for dcy in range(-2, 3):
                        nkey = (cx + dcx, cy + dcy)
                        if nkey not in grid:
                            continue
                        for j in grid[nkey]:
                            if j <= i:
                                continue
                            dx = pos_x[i] - pos_x[j]
                            dy = pos_y[i] - pos_y[j]
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist < 0.01:
                                dist = 0.01
                            force = repulsion_k2 / dist
                            fx = (dx / dist) * force
                            fy = (dy / dist) * force
                            disp_x[i] += fx
                            disp_y[i] += fy
                            disp_x[j] -= fx
                            disp_y[j] -= fy

        # Attractive forces (edges only)
        for u, v in edges:
            dx = pos_x[u] - pos_x[v]
            dy = pos_y[u] - pos_y[v]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.01:
                dist = 0.01
            force = (dist * dist) / k
            fx = (dx / dist) * force
            fy = (dy / dist) * force
            disp_x[u] -= fx
            disp_y[u] -= fy
            disp_x[v] += fx
            disp_y[v] += fy

        # Gravity: gently pull toward origin to keep layout centered
        for i in range(n):
            disp_x[i] -= pos_x[i] * gravity
            disp_y[i] -= pos_y[i] * gravity

        # Apply displacements with temperature limit
        for i in range(n):
            d = math.sqrt(disp_x[i] ** 2 + disp_y[i] ** 2)
            if d > 0.01:
                scale_f = min(d, temp) / d
                pos_x[i] += disp_x[i] * scale_f
                pos_y[i] += disp_y[i] * scale_f

        temp -= cool

    return {i: (pos_x[i], pos_y[i]) for i in range(n)}


def _point_to_segment_dist(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> Tuple[float, float, float]:
    """Distance from point (px,py) to segment (ax,ay)-(bx,by).

    Returns (distance, t_param, side) where t_param is the projection parameter [0,1] along the
    segment and side is +1/-1 indicating which side of the segment the point lies on.
    """
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-12:
        d = math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        return (d, 0.0, 1.0)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    d = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    cross = dx * (py - ay) - dy * (px - ax)
    side = 1.0 if cross >= 0 else -1.0
    return (d, t, side)


def _route_edge(
    src_id: str,
    dst_id: str,
    positions: Dict[str, Tuple[float, float]],
    radii: Dict[str, float],
    all_ids: List[str],
    margin: float = 4.0,
) -> Optional[List[Tuple[float, float]]]:
    """Compute waypoints that route an edge around intermediate nodes.

    Returns None if the straight line is clear (no routing needed), or a list of waypoints
    [src, wp1, wp2, ..., dst] that avoids all intermediate node circles.
    """
    x1, y1 = positions[src_id]
    x2, y2 = positions[dst_id]
    r_src = radii.get(src_id, 5.0)
    r_dst = radii.get(dst_id, 5.0)

    obstacles = []
    for nid in all_ids:
        if nid == src_id or nid == dst_id:
            continue
        cx, cy = positions[nid]
        r = radii.get(nid, 5.0)
        dist, t, side = _point_to_segment_dist(cx, cy, x1, y1, x2, y2)
        clearance = r + margin
        if dist < clearance and 0.0 < t < 1.0:
            obstacles.append((t, nid, cx, cy, r, dist, side))

    if not obstacles:
        return None

    obstacles.sort(key=lambda o: o[0])

    seg_dx = x2 - x1
    seg_dy = y2 - y1
    seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
    if seg_len < 1e-6:
        return None
    perp_x = -seg_dy / seg_len
    perp_y = seg_dx / seg_len

    waypoints = [(x1, y1)]
    for _t, _nid, cx, cy, r, dist, side in obstacles:
        detour_dist = r + margin + 2.0
        if dist < 1e-6:
            wx = cx + perp_x * detour_dist * side
            wy = cy + perp_y * detour_dist * side
        else:
            dx_off = cx - (x1 + _t * seg_dx)
            dy_off = cy - (y1 + _t * seg_dy)
            off_len = math.sqrt(dx_off * dx_off + dy_off * dy_off)
            if off_len < 1e-6:
                wx = cx + perp_x * detour_dist * side
                wy = cy + perp_y * detour_dist * side
            else:
                push_x = -dx_off / off_len
                push_y = -dy_off / off_len
                wx = cx + push_x * detour_dist
                wy = cy + push_y * detour_dist
        waypoints.append((wx, wy))
    waypoints.append((x2, y2))
    return waypoints


def _waypoints_to_svg_path(
    waypoints: List[Tuple[float, float]],
    r_src: float,
    r_dst: float,
) -> str:
    """Convert waypoints to a smooth SVG cubic bezier path string, trimmed to stop at node circle
    edges."""
    if len(waypoints) < 2:
        return ""

    pts = list(waypoints)

    # Trim start: move first point along direction to second point
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    d = math.sqrt(dx * dx + dy * dy)
    if d > r_src + 1:
        pts[0] = (pts[0][0] + dx / d * r_src, pts[0][1] + dy / d * r_src)

    # Trim end: move last point along direction from second-to-last
    dx = pts[-1][0] - pts[-2][0]
    dy = pts[-1][1] - pts[-2][1]
    d = math.sqrt(dx * dx + dy * dy)
    if d > r_dst + 1:
        pts[-1] = (pts[-1][0] - dx / d * r_dst, pts[-1][1] - dy / d * r_dst)

    if len(pts) == 2:
        return f"M{pts[0][0]:.1f},{pts[0][1]:.1f} L{pts[1][0]:.1f},{pts[1][1]:.1f}"

    # Catmull-Rom to cubic bezier for smooth curves through waypoints
    path = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    n = len(pts)
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[min(i + 1, n - 1)]
        p3 = pts[min(i + 2, n - 1)]

        # Catmull-Rom to cubic bezier control points (alpha=0.5 centripetal)
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0

        path += f" C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {p2[0]:.1f},{p2[1]:.1f}"

    return path


def _remove_overlaps(
    positions: Dict[str, Tuple[float, float]],
    radii: Dict[str, float],
    padding: float = 2.0,
    max_passes: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Push overlapping nodes apart until no circles overlap.

    Uses iterative pairwise displacement with grid acceleration.
    Each pass finds overlapping pairs and pushes them apart by half
    the overlap distance each, along the line connecting their centers.
    """
    ids = list(positions.keys())
    rng = random.Random(seed)
    n = len(ids)
    if n < 2:
        return dict(positions)

    px = {nid: positions[nid][0] for nid in ids}
    py = {nid: positions[nid][1] for nid in ids}

    for _pass in range(max_passes):
        moved = False
        max_r = max(radii.get(nid, 5.0) for nid in ids)
        cell = (max_r + padding) * 2.2
        if cell < 1.0:
            cell = 1.0

        grid: Dict[Tuple[int, int], List[str]] = {}
        for nid in ids:
            cx = int(px[nid] / cell)
            cy = int(py[nid] / cell)
            key = (cx, cy)
            if key not in grid:
                grid[key] = []
            grid[key].append(nid)

        for nid in ids:
            ri = radii.get(nid, 5.0)
            cx = int(px[nid] / cell)
            cy = int(py[nid] / cell)
            for dcx in range(-1, 2):
                for dcy in range(-1, 2):
                    nkey = (cx + dcx, cy + dcy)
                    if nkey not in grid:
                        continue
                    for other in grid[nkey]:
                        if other <= nid:
                            continue
                        rj = radii.get(other, 5.0)
                        dx = px[nid] - px[other]
                        dy = py[nid] - py[other]
                        dist = math.sqrt(dx * dx + dy * dy)
                        min_dist = ri + rj + padding
                        if dist < min_dist:
                            if dist < 0.01:
                                angle = rng.uniform(0, 2 * math.pi)
                                dx, dy = math.cos(angle), math.sin(angle)
                                dist = 0.01
                                ux, uy = dx, dy
                            else:
                                ux, uy = dx / dist, dy / dist
                            overlap = min_dist - dist
                            push = overlap * 0.55
                            px[nid] += ux * push
                            py[nid] += uy * push
                            px[other] -= ux * push
                            py[other] -= uy * push
                            moved = True

        if not moved:
            break

    return {nid: (px[nid], py[nid]) for nid in ids}


def _turn_to_fill(
    raw: Dict[int, Tuple[float, float]], width: float, height: float
) -> Dict[int, Tuple[float, float]]:
    """Rotate layout coordinates to the angle at which they are drawn largest in a frame.

    A force layout's orientation is arbitrary, so without this a long graph can come out
    upright in a wide frame and be drawn small. Angles are tried in 5-degree steps, and the
    first of any that fit equally well wins, so the result is deterministic.
    """
    pts = list(raw.values())
    best_angle, best_scale = 0.0, 0.0
    for step in range(36):
        angle = math.radians(step * 5)
        c, s = math.cos(angle), math.sin(angle)
        us = [x * c - y * s for x, y in pts]
        vs = [x * s + y * c for x, y in pts]
        range_u = max(us) - min(us)
        range_v = max(vs) - min(vs)
        if range_u == 0 and range_v == 0:
            return raw
        scale = min(
            width / range_u if range_u > 0 else math.inf,
            height / range_v if range_v > 0 else math.inf,
        )
        if scale > best_scale * (1 + 1e-9):
            best_angle, best_scale = angle, scale
    if best_angle == 0.0:
        return raw
    c, s = math.cos(best_angle), math.sin(best_angle)
    return {idx: (x * c - y * s, x * s + y * c) for idx, (x, y) in raw.items()}


def _force_layout_raw(
    n: int,
    edges: List[Tuple[int, int]],
    iterations: int,
    seed: Optional[int],
) -> Dict[int, Tuple[float, float]]:
    """Force-directed positions keyed by node index, from the Rust engine or the Python one."""
    raw = None
    try:
        from edgecase._core import force_directed_layout_rust

        raw = force_directed_layout_rust(n, edges, min(iterations, 500), seed=seed)
    except Exception:
        pass
    if not raw:
        raw = _fruchterman_reingold_python(n, edges, iterations, seed=seed)
    return raw


def _index_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """The connected components of nodes `0..n`, largest first, ties by their first node."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s, t in edges:
        rs, rt = find(s), find(t)
        if rs != rt:
            parent[max(rs, rt)] = min(rs, rt)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda c: (-len(c), c[0]))


def _pack_components(
    components: List[List[int]],
    edges: List[Tuple[int, int]],
    width: float,
    height: float,
    iterations: int,
    seed: Optional[int],
) -> Dict[int, Tuple[float, float]]:
    """Lay out each component alone and pack them in rows to fill a `width` by `height` frame.

    Each piece is sized by the square root of its node count, so nodes are spaced alike in
    all of them. The pieces go into rows, largest first, and the row width that lets the
    whole be drawn largest in the frame is kept.
    """
    which = {}
    for ci, comp in enumerate(components):
        for j, i in enumerate(comp):
            which[i] = (ci, j)
    comp_edges: List[List[Tuple[int, int]]] = [[] for _ in components]
    for s, t in edges:
        ci, js = which[s]
        comp_edges[ci].append((js, which[t][1]))

    gap = 1.2
    pieces = []  # (positions relative to the piece's corner, piece width, piece height)
    for comp, cedges in zip(components, comp_edges):
        if len(comp) == 1:
            pieces.append(({comp[0]: (0.0, 0.0)}, 0.0, 0.0))
            continue
        raw = _force_layout_raw(len(comp), cedges, iterations, seed)
        raw = _turn_to_fill(raw, width, height)
        xs = [p[0] for p in raw.values()]
        ys = [p[1] for p in raw.values()]
        extent = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
        size = 1.2 * math.sqrt(len(comp))
        f = size / extent
        rel = {comp[j]: ((x - min(xs)) * f, (y - min(ys)) * f) for j, (x, y) in raw.items()}
        pieces.append((rel, (max(xs) - min(xs)) * f, (max(ys) - min(ys)) * f))

    def shelve(row_width: float):
        rows: List[List[int]] = [[]]
        used = 0.0
        for k, (_, w, _h) in enumerate(pieces):
            if rows[-1] and used + gap + w > row_width:
                rows.append([])
                used = 0.0
            used += (gap if rows[-1] else 0.0) + w
            rows[-1].append(k)
        return rows

    def extent_of(rows):
        total_w = max(sum(pieces[k][1] for k in row) + gap * (len(row) - 1) for row in rows)
        total_h = sum(max(pieces[k][2] for k in row) for row in rows) + gap * (len(rows) - 1)
        return total_w, total_h

    best_rows, best_scale = None, -1.0
    widths = sorted({sum(p[1] for p in pieces[: k + 1]) + gap * k for k in range(len(pieces))})
    for row_width in widths:
        rows = shelve(row_width)
        total_w, total_h = extent_of(rows)
        scale = min(width / max(total_w, 1e-9), height / max(total_h, 1e-9))
        if scale > best_scale * (1 + 1e-9):
            best_rows, best_scale = rows, scale

    total_w, _ = extent_of(best_rows)
    out: Dict[int, Tuple[float, float]] = {}
    y0 = 0.0
    for row in best_rows:
        row_h = max(pieces[k][2] for k in row)
        row_w = sum(pieces[k][1] for k in row) + gap * (len(row) - 1)
        x0 = (total_w - row_w) / 2
        for k in row:
            rel, w, h = pieces[k]
            dy = y0 + (row_h - h) / 2
            for i, (x, y) in rel.items():
                out[i] = (x0 + x, dy + y)
            x0 += w + gap
        y0 += row_h + gap
    return out


def _force_directed_layout(
    node_ids: List[str],
    edges: List[Tuple[str, str]],
    width: float,
    height: float,
    iterations: int = 300,
    seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute force-directed layout, using Rust engine with Python fallback."""
    if len(node_ids) == 0:
        return {}

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    indexed_edges = []
    for s, t in edges:
        if s in id_to_idx and t in id_to_idx:
            indexed_edges.append((id_to_idx[s], id_to_idx[t]))

    pad = 60.0
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad

    components = _index_components(len(node_ids), indexed_edges)
    if len(components) > 1:
        # Laid out together, the pieces push each other to the corners of the frame and each
        # is drawn small, so each piece is laid out alone and the pieces are packed
        raw = _pack_components(components, indexed_edges, usable_w, usable_h, iterations, seed)
    else:
        raw = _force_layout_raw(len(node_ids), indexed_edges, iterations, seed)
        if raw:
            raw = _turn_to_fill(raw, usable_w, usable_h)

    if not raw:
        return _circular_layout(node_ids, width, height)

    xs = [pos[0] for pos in raw.values()]
    ys = [pos[1] for pos in raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x != min_x else 1.0
    range_y = max_y - min_y if max_y != min_y else 1.0

    scale = min(usable_w / range_x, usable_h / range_y)

    cx_raw = (min_x + max_x) / 2
    cy_raw = (min_y + max_y) / 2
    cx_svg = width / 2
    cy_svg = height / 2

    # Key order follows node_ids (the Rust result is a hash map in arbitrary order), so later
    # order-dependent steps like overlap removal are reproducible
    positions = {}
    for idx, pos in sorted(raw.items()):
        x = cx_svg + (pos[0] - cx_raw) * scale
        y = cy_svg + (pos[1] - cy_raw) * scale
        positions[node_ids[idx]] = (x, y)
    return positions


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _darken(hex_color: str, factor: float = 0.25) -> str:
    """Darken a hex color by the given factor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _contrast_color(hex_color: str) -> str:
    """Return white or dark text for readability against hex_color."""
    h = hex_color.lstrip("#")
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
    except (ValueError, IndexError):
        return "#374151"

    def adj(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * adj(r) + 0.7152 * adj(g) + 0.0722 * adj(b)
    return "#ffffff" if lum < 0.45 else "#374151"


def _build_color_map(
    values: List[Any],
    palette_name: str = "tableau",
) -> Dict[Any, str]:
    """Map unique categorical values to palette colors."""
    try:
        from edgecase.colors import get_palette

        colors = get_palette(palette_name)
    except Exception:
        colors = [
            "#4e79a7",
            "#f28e2c",
            "#e15759",
            "#76b7b2",
            "#59a14f",
            "#edc949",
            "#af7aa1",
            "#ff9da7",
            "#9c755f",
            "#bab0ab",
        ]
    unique = list(dict.fromkeys(values))
    mapping = {}
    for i, val in enumerate(unique):
        mapping[val] = colors[i % len(colors)]
    return mapping


# ---------------------------------------------------------------------------
# Color scales and highlighting
# ---------------------------------------------------------------------------

COLOR_SCALES = ("sequential", "diverging", "categorical")

# Each stop of a numeric color scale is (position, light-theme color, dark-theme color). The
# sequential scale is one blue, from light to dark on the light surface and the other way on the
# dark one, so that high values stand out on both; its lightest (or darkest) step still stands
# clear of the surface. The diverging scale runs from blue through a neutral gray to red
SEQUENTIAL_STOPS = ((0.0, "#86b6ef", "#184f95"), (1.0, "#0d366b", "#cde2fb"))
DIVERGING_STOPS = (
    (-1.0, "#1c5cab", "#5598e7"),
    (0.0, "#d3d1cb", "#4a4944"),
    (1.0, "#b42318", "#f0706a"),
)
# Nodes and edges with no value on a numeric scale
MISSING_COLOR = ("#d1d5db", "#4b5563")

# highlight=: what it picks out is in the accent color (or keeps its mapped color), a "soft"
# element in a lighter version of it, and everything else is muted
HIGHLIGHT_LEVELS = ("accent", "soft")
ACCENT_COLOR = ("#c2410c", "#f97316")
MUTED_NODE_COLOR = ("#d4d4d8", "#3f3f46")
MUTED_LABEL_COLOR = ("#a1a1aa", "#71717a")


def _srgb_to_oklab(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    lin = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255.0
        lin.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin
    l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _oklab_to_srgb(lab: Tuple[float, float, float]) -> str:
    L, a, b = lab
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )
    out = []
    for c in rgb:
        c = min(max(c, 0.0), 1.0)
        c = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        out.append(round(c * 255))
    return "#" + "".join(f"{c:02x}" for c in out)


def _mix(c1: str, c2: str, t: float) -> str:
    """The color `t` of the way from hex `c1` to hex `c2`, mixed in OKLab."""
    a, b = _srgb_to_oklab(c1), _srgb_to_oklab(c2)
    return _oklab_to_srgb(tuple(x + (y - x) * t for x, y in zip(a, b)))


def _scale_color(stops, t: float) -> Tuple[str, str]:
    """The (light, dark) colors at position `t` of a scale's `stops`, in steps of a hundredth of
    the scale, so that a plot needs at most one color variable per step."""
    lo, hi = stops[0][0], stops[-1][0]
    t = round(min(max(t, lo), hi) * 100) / 100
    for (p0, l0, d0), (p1, l1, d1) in zip(stops, stops[1:]):
        if t <= p1:
            f = (t - p0) / (p1 - p0)
            return _mix(l0, l1, f), _mix(d0, d1, f)
    return stops[-1][1], stops[-1][2]


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _color_scale(values: List[Any], scale: Optional[str], param: str) -> str:
    """The scale for mapping `values` to colors: `scale` if given, else sequential for
    numbers among which there is a fraction (a float), and categorical for anything else."""
    if scale is not None:
        if scale not in COLOR_SCALES:
            raise ValueError(
                f"Unknown color_scale {scale!r}; use one of {', '.join(map(repr, COLOR_SCALES))}"
            )
        present = [v for v in values if v is not None]
        if scale != "categorical" and not all(_is_number(v) for v in present):
            raise ValueError(f"color_scale={scale!r} needs numeric values of {param}")
        return scale
    present = [v for v in values if v is not None]
    numeric = present and all(_is_number(v) for v in present)
    if numeric and any(isinstance(v, float) for v in present):
        return "sequential"
    return "categorical"


def _numeric_colors(
    values: List[Any], scale: str, midpoint: Optional[float]
) -> Tuple[List[Tuple[str, str]], Tuple[float, float, Optional[float]]]:
    """(light, dark) colors for `values` on the sequential or diverging `scale`, and the
    (low, high, midpoint) the scale's ends stand for."""
    nums = [v for v in values if _is_number(v)]
    if not nums:
        return [MISSING_COLOR] * len(values), (0.0, 0.0, None)
    lo, hi = min(nums), max(nums)
    if scale == "diverging":
        mid = 0.0 if midpoint is None else float(midpoint)
        extent = max(abs(hi - mid), abs(mid - lo)) or 1.0
        out = [
            _scale_color(DIVERGING_STOPS, (v - mid) / extent) if _is_number(v) else MISSING_COLOR
            for v in values
        ]
        return out, (mid - extent, mid + extent, mid)
    span = (hi - lo) or 1.0
    out = [
        _scale_color(SEQUENTIAL_STOPS, (v - lo) / span) if _is_number(v) else MISSING_COLOR
        for v in values
    ]
    return out, (lo, hi, None)


def _sorted_values(values: List[Any]) -> List[Any]:
    """The distinct `values` in order (`None` last), or in order of first appearance should they
    not be comparable."""
    unique = list(dict.fromkeys(v for v in values if v is not None))
    try:
        unique.sort()
    except TypeError:
        pass
    if any(v is None for v in values):
        unique.append(None)
    return unique


def _is_marked(v) -> bool:
    """Whether an attribute value picks its node or edge out for `highlight=`: any value but a
    missing one (`None` or NaN) or `False`. A 0 counts, being, e.g., the first node of a path."""
    return v is not None and v is not False and not (isinstance(v, float) and math.isnan(v))


def _graph_edges(graph) -> Dict[int, Tuple[int, int, Dict[str, Any], Optional[float]]]:
    """{edge ID: (source, target, attributes, weight)}."""
    out = {}
    for eid in graph.get_edge_ids():
        e = graph.get_edge(eid)
        out[eid] = (e.source, e.target, dict(e.attributes) if e.attributes else {}, e.weight)
    return out


def _resolve_highlight(
    graph, highlight, highlight_edges
) -> Dict[str, Tuple[set, set]]:
    """{level: (node IDs, edge IDs)} picked out by `highlight=` and `highlight_edges=`, for each
    of `HIGHLIGHT_LEVELS`.

    `highlight=` takes nodes: one node (an ID or a label), a list of them, `"selection"` (the
    selected nodes and edges), or the name of a node or edge attribute (the nodes or edges whose
    value `_is_marked()`). A list in which each node is joined to the next is a path, and its
    edges are picked out as well; so are those of an attribute whose marked values are distinct
    whole numbers, taken in their order (as `shortest_path_order` holds a path). A list can also
    hold `(u, v)` pairs, for the edges between them. `highlight_edges=` takes edges: an edge ID,
    a list of edge IDs or `(u, v)` pairs (or a dict keyed by edge IDs, as `maximum_flow()`
    returns), or the name of an edge attribute. Either can be a dict mapping `"accent"` and
    `"soft"` to such values. The ends of a picked-out edge are picked out at its level, too, and
    what is picked out at both levels is accent.
    """
    out = {level: (set(), set()) for level in HIGHLIGHT_LEVELS}
    graph_edges = None
    node_attrs = None

    def edges_info():
        nonlocal graph_edges
        if graph_edges is None:
            graph_edges = _graph_edges(graph)
        return graph_edges

    def nodes_info():
        nonlocal node_attrs
        if node_attrs is None:
            node_attrs = {}
            for nid in graph.get_node_ids(limit=None):
                obj = graph.get_node(nid)
                node_attrs[nid] = dict(obj.attributes) if obj and obj.attributes else {}
        return node_attrs

    def node_id(ref, param):
        if isinstance(ref, bool) or not isinstance(ref, (int, str)):
            raise TypeError(f"{param}= takes node IDs or labels, got {ref!r}")
        try:
            return graph._resolve_node_id(ref)
        except Exception:
            raise ValueError(f"{param}={ref!r}: no such node") from None

    def pair_edges(u, v, param):
        a, b = node_id(u, param), node_id(v, param)
        eids = list(graph.find_edge_ids(a, b))
        if not graph.is_directed() and not eids:
            eids = list(graph.find_edge_ids(b, a))
        if not eids:
            raise ValueError(f"{param}: no edge joins {u!r} and {v!r}")
        return eids

    def path_edges(ids):
        found = []
        for a, b in zip(ids, ids[1:]):
            eids = list(graph.find_edge_ids(a, b))
            if not graph.is_directed() and not eids:
                eids = list(graph.find_edge_ids(b, a))
            if not eids:
                return []
            found.extend(eids)
        return found

    def edge_attr_marked(name):
        info = edges_info()
        if not any(name in attrs for _, _, attrs, _ in info.values()):
            return None
        return {eid for eid, (_, _, attrs, _) in info.items() if _is_marked(attrs.get(name))}

    def nodes_spec(spec):
        param = "highlight"
        if isinstance(spec, str):
            if spec == "selection":
                return set(graph.get_selected_node_ids()), set(graph.get_selected_edge_ids())
            attrs = nodes_info()
            if any(spec in a for a in attrs.values()):
                marked = {nid: a[spec] for nid, a in attrs.items() if _is_marked(a.get(spec))}
                edges = set()
                vals = list(marked.values())
                if (
                    len(vals) >= 2
                    and all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
                    and len(set(vals)) == len(vals)
                ):
                    edges = set(path_edges(sorted(marked, key=marked.get)))
                return set(marked), edges
            edges = edge_attr_marked(spec)
            if edges is not None:
                return set(), edges
            try:
                return {graph._resolve_node_id(spec)}, set()
            except Exception:
                raise ValueError(
                    f"highlight={spec!r} is neither a node nor the name of a node or edge "
                    f"attribute (or 'selection')"
                ) from None
        if isinstance(spec, int) and not isinstance(spec, bool):
            return {node_id(spec, param)}, set()
        if isinstance(spec, dict):
            spec = list(spec)
        try:
            items = list(spec)
        except TypeError:
            raise TypeError(f"highlight= takes nodes, edges or an attribute name, got {spec!r}")
        nodes, edges, ordered = set(), set(), []
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                edges.update(pair_edges(item[0], item[1], param))
            else:
                ordered.append(node_id(item, param))
        nodes.update(ordered)
        if not isinstance(spec, (set, frozenset)) and len(ordered) == len(items):
            edges.update(path_edges(ordered))
        return nodes, edges

    def edges_spec(spec):
        param = "highlight_edges"
        if isinstance(spec, str):
            if spec == "selection":
                return set(), set(graph.get_selected_edge_ids())
            edges = edge_attr_marked(spec)
            if edges is None:
                raise ValueError(f"highlight_edges={spec!r}: no edge has an attribute {spec!r}")
            return set(), edges
        if isinstance(spec, int) and not isinstance(spec, bool):
            spec = [spec]
        if isinstance(spec, dict):
            spec = list(spec)
        try:
            items = list(spec)
        except TypeError:
            raise TypeError(f"highlight_edges= takes edges or an attribute name, got {spec!r}")
        edges = set()
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                edges.update(pair_edges(item[0], item[1], param))
            elif isinstance(item, int) and not isinstance(item, bool):
                if not graph.has_edge_id(item):
                    raise ValueError(f"highlight_edges: no edge has the ID {item!r}")
                edges.add(item)
            else:
                raise TypeError(
                    f"highlight_edges= takes edge IDs or (u, v) pairs, got {item!r}"
                )
        return set(), edges

    for value, resolve in ((highlight, nodes_spec), (highlight_edges, edges_spec)):
        if value is None:
            continue
        if isinstance(value, dict) and value and set(value) <= set(HIGHLIGHT_LEVELS):
            levels = list(value.items())
        else:
            levels = [("accent", value)]
        for level, spec in levels:
            if spec is None:
                continue
            nodes, edges = resolve(spec)
            out[level][0].update(nodes)
            out[level][1].update(edges)

    for level in HIGHLIGHT_LEVELS:
        for eid in out[level][1]:
            out[level][0].update(graph.get_node_ids_by_edge_id(eid))
    out["soft"][0].difference_update(out["accent"][0])
    out["soft"][1].difference_update(out["accent"][1])
    return out


# ---------------------------------------------------------------------------
# Multi-edge fan-out
# ---------------------------------------------------------------------------


def _compute_edge_fans(
    edges: List[Tuple[str, str]],
) -> Tuple[List[float], List[float]]:
    """Compute bow values and loop sizes for fanning out parallel edges.

    For each edge, returns:

    - bow: curve amount (0.0 = straight). For edges between the same unordered pair of nodes, bows
    are assigned so the curves visually fan out.
    - loop_size: scaling multiplier for self-loop arcs (1.0 = default).

    The key insight: for edge A→B, the perpendicular is in one direction; for B→A it's reversed. So
    to place them on OPPOSITE visual sides, both need the SAME bow value (the perpendicular reversal
    provides the mirror). To place them on the same visual side, they need opposite bow values.

    We assign visual offsets along a canonical perpendicular, then convert:
    canonical direction = src < tgt. For canonical-direction edges, bow = v. For reverse-direction
    edges, bow = -v.
    """
    from collections import defaultdict

    pair_indices: Dict[Tuple[str, str], List[Tuple[int, str, str]]] = defaultdict(list)
    for i, (src, tgt) in enumerate(edges):
        if src == tgt:
            pair_indices[("__loop__", src)].append((i, src, tgt))
        else:
            key = (min(src, tgt), max(src, tgt))
            pair_indices[key].append((i, src, tgt))

    bows = [0.0] * len(edges)
    loop_sizes = [1.0] * len(edges)

    for key, group in pair_indices.items():
        if key[0] == "__loop__":
            for k, (idx, _s, _t) in enumerate(group):
                loop_sizes[idx] = 1.0 + k * 0.6
            continue

        n = len(group)
        if n == 1:
            continue

        step = 0.2
        total = step * (n - 1)
        if total > 1.0:
            step = 1.0 / (n - 1)

        canonical_a, canonical_b = key
        for k, (idx, src, tgt) in enumerate(group):
            visual_offset = (k - (n - 1) / 2) * step
            is_canonical = src <= tgt
            bows[idx] = visual_offset if is_canonical else -visual_offset

    return bows, loop_sizes


# ---------------------------------------------------------------------------
# SVG drawing primitives
# ---------------------------------------------------------------------------


def _svg_edge_hit_area(geometry: str, width: float, edge_id: str, attrs_json: str) -> str:
    """An invisible, wider copy of an edge's stroke, which takes the hover for its tooltip, since
    the edge itself can be thinner than a pixel. `geometry` is the element's tag and shape, e.g.,
    `'path d="M 0 0 L 9 9"'`."""
    return (
        f'<{geometry} fill="none" stroke="transparent" stroke-width="{max(8.0, width + 4):.1f}" '
        f"pointer-events=\"stroke\" data-edge=\"{edge_id}\" data-attrs='{attrs_json}'/>"
    )


def _svg_edge_straight(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r_src: float,
    r_dst: float,
    color: str,
    width: float,
    opacity: float,
    directed: bool,
    edge_id: str,
    attrs_json: str,
) -> str:
    """Render a straight edge with endpoint trimming and optional arrowhead."""
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return ""

    ux, uy = dx / dist, dy / dist

    # Trim endpoints to node borders
    sx = x1 + ux * r_src
    sy = y1 + uy * r_src
    ex = x2 - ux * r_dst
    ey = y2 - uy * r_dst

    parts = []

    # Arrowhead scales with edge width
    arrow_scale = max(width / 1.2, 0.6)
    head_len = 8.0 * arrow_scale
    head_wid = 5.0 * arrow_scale
    if directed:
        ex_line = ex - ux * head_len
        ey_line = ey - uy * head_len
    else:
        ex_line = ex
        ey_line = ey

    if directed:
        bx = ex - ux * head_len
        by = ey - uy * head_len
        ppx, ppy = -uy, ux
        p1x = bx + ppx * head_wid / 2
        p1y = by + ppy * head_wid / 2
        p2x = bx - ppx * head_wid / 2
        p2y = by - ppy * head_wid / 2
        parts.append(
            f'<g opacity="{opacity}">'
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex_line:.1f}" y2="{ey_line:.1f}" '
            f'stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"/>'
            f'<polygon points="{ex:.1f},{ey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
            f'fill="{color}"/>'
            f"</g>"
        )
    else:
        parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex_line:.1f}" y2="{ey_line:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round"/>'
        )
    parts.append(
        _svg_edge_hit_area(
            f'line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}"',
            width,
            edge_id,
            attrs_json,
        )
    )

    return "\n".join(parts)


def _svg_edge_routed(
    waypoints: List[Tuple[float, float]],
    r_src: float,
    r_dst: float,
    color: str,
    width: float,
    opacity: float,
    directed: bool,
    edge_id: str,
    attrs_json: str,
) -> str:
    """Render a routed edge as a smooth bezier path through waypoints."""
    path_d = _waypoints_to_svg_path(waypoints, r_src, r_dst)
    if not path_d:
        return ""

    parts = []
    hit_area = _svg_edge_hit_area(f'path d="{path_d}"', width, edge_id, attrs_json)
    arrow_scale = max(width / 1.2, 0.6)
    head_len = 8.0 * arrow_scale
    head_wid = 5.0 * arrow_scale

    if directed:
        pts = list(waypoints)
        dx = pts[-1][0] - pts[-2][0]
        dy = pts[-1][1] - pts[-2][1]
        d = math.sqrt(dx * dx + dy * dy)
        if d > 1e-6:
            ux, uy = dx / d, dy / d
            ex = pts[-1][0] - ux * r_dst
            ey = pts[-1][1] - uy * r_dst
            bx = ex - ux * head_len
            by = ey - uy * head_len
            ppx, ppy = -uy, ux
            p1x = bx + ppx * head_wid / 2
            p1y = by + ppy * head_wid / 2
            p2x = bx - ppx * head_wid / 2
            p2y = by - ppy * head_wid / 2
            # Trim path to stop before arrowhead
            trim_pts = list(waypoints)
            trim_pts[-1] = (bx, by)
            path_d = _waypoints_to_svg_path(trim_pts, r_src, 0)
            parts.append(
                f'<g opacity="{opacity}">'
                f'<path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" stroke-linecap="round"/>'
                f'<polygon points="{ex:.1f},{ey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                f'fill="{color}"/>'
                f"</g>"
            )
        else:
            parts.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" stroke-opacity="{opacity}" '
                f'stroke-linecap="round"/>'
            )
    else:
        parts.append(
            f'<path d="{path_d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round"/>'
        )
    parts.append(hit_area)

    return "\n".join(parts)


def _svg_edge_curved(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    r_src: float,
    r_dst: float,
    color: str,
    width: float,
    opacity: float,
    directed: bool,
    edge_id: str,
    attrs_json: str,
    bow: float = 0.2,
) -> str:
    """Render a curved edge (quadratic Bezier) with optional arrowhead."""
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return ""

    ux, uy = dx / dist, dy / dist
    # Perpendicular
    px, py = -uy, ux

    # Control point bowed perpendicular to the edge midpoint
    mx = (x1 + x2) / 2 + px * dist * bow
    my = (y1 + y2) / 2 + py * dist * bow

    # Trim start: move along direction from src to control point
    cdx_s = mx - x1
    cdy_s = my - y1
    cd_s = math.sqrt(cdx_s * cdx_s + cdy_s * cdy_s)
    if cd_s > 1e-6:
        sx = x1 + (cdx_s / cd_s) * r_src
        sy = y1 + (cdy_s / cd_s) * r_src
    else:
        sx, sy = x1, y1

    # Trim end: use the tangent at t=1 (from control point toward dst)
    # so the trim aligns with the curve's actual approach angle
    tan_ex = x2 - mx
    tan_ey = y2 - my
    tan_e_len = math.sqrt(tan_ex * tan_ex + tan_ey * tan_ey)
    if tan_e_len > 1e-6:
        tex, tey = tan_ex / tan_e_len, tan_ey / tan_e_len
        ex = x2 - tex * r_dst
        ey = y2 - tey * r_dst
    else:
        ex, ey = x2, y2

    parts = []
    path_id = f"path-{edge_id}"

    if directed:
        # Arrowhead pointing along curve tangent at t=1
        tx = 2 * (ex - mx)
        ty = 2 * (ey - my)
        t_len = math.sqrt(tx * tx + ty * ty)
        arrow_svg = ""
        if t_len > 1e-6:
            tux, tuy = tx / t_len, ty / t_len
            arrow_scale = max(width / 1.2, 0.6)
            head_len, head_wid = 8.0 * arrow_scale, 5.0 * arrow_scale
            bx = ex - tux * head_len
            by = ey - tuy * head_len
            ppx, ppy = -tuy, tux
            p1x = bx + ppx * head_wid / 2
            p1y = by + ppy * head_wid / 2
            p2x = bx - ppx * head_wid / 2
            p2y = by - ppy * head_wid / 2
            arrow_svg = (
                f'<polygon points="{ex:.1f},{ey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                f'fill="{color}"/>'
            )
        parts.append(
            f'<g opacity="{opacity}">'
            f'<path id="{path_id}" d="M {sx:.1f} {sy:.1f} Q {mx:.1f} {my:.1f} {ex:.1f} {ey:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"/>'
            f"{arrow_svg}</g>"
        )
    else:
        parts.append(
            f'<path id="{path_id}" d="M {sx:.1f} {sy:.1f} Q {mx:.1f} {my:.1f} {ex:.1f} {ey:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round"/>'
        )
    parts.append(
        _svg_edge_hit_area(
            f'path d="M {sx:.1f} {sy:.1f} Q {mx:.1f} {my:.1f} {ex:.1f} {ey:.1f}"',
            width,
            edge_id,
            attrs_json,
        )
    )

    return "\n".join(parts)


def _svg_edge_loop(
    x: float,
    y: float,
    r: float,
    color: str,
    width: float,
    opacity: float,
    directed: bool,
    edge_id: str,
    attrs_json: str,
    loop_size: float = 1.0,
) -> str:
    """Render a self-loop as a circular arc above the node."""
    lr = r * 1.2 * loop_size
    # Arc center above the node
    cx_arc = x
    cy_arc = y - r - lr
    # Start and end points on the node border
    angle = 0.45
    sx = x + r * math.sin(-angle)
    sy = y - r * math.cos(-angle)
    ex = x + r * math.sin(angle)
    ey = y - r * math.cos(angle)

    # Use two cubic bezier curves for a smooth loop
    cp_h = lr * 1.6
    c1x = sx - cp_h * 0.3
    c1y = sy - cp_h
    c2x = cx_arc - lr
    c2y = cy_arc
    c3x = cx_arc + lr
    c3y = cy_arc
    c4x = ex + cp_h * 0.3
    c4y = ey - cp_h

    path_d = (
        f"M {sx:.1f} {sy:.1f} "
        f"C {c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {cx_arc:.1f} {cy_arc - lr * 0.2:.1f} "
        f"C {c3x:.1f} {c3y:.1f} {c4x:.1f} {c4y:.1f} {ex:.1f} {ey:.1f}"
    )

    parts = []
    path_id = f"path-{edge_id}"
    if directed:
        # Arrowhead tangent at the end of the curve
        tan_x = ex - c4x
        tan_y = ey - c4y
        t_len = math.sqrt(tan_x * tan_x + tan_y * tan_y)
        arrow_svg = ""
        if t_len > 1e-6:
            tux, tuy = tan_x / t_len, tan_y / t_len
            arrow_scale = max(width / 1.2, 0.6)
            head_len, head_wid = 8.0 * arrow_scale, 5.0 * arrow_scale
            bx = ex - tux * head_len
            by = ey - tuy * head_len
            ppx, ppy = -tuy, tux
            p1x = bx + ppx * head_wid / 2
            p1y = by + ppy * head_wid / 2
            p2x = bx - ppx * head_wid / 2
            p2y = by - ppy * head_wid / 2
            arrow_svg = (
                f'<polygon points="{ex:.1f},{ey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                f'fill="{color}"/>'
            )
        parts.append(
            f'<g opacity="{opacity}">'
            f'<path id="{path_id}" d="{path_d}" '
            f'fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"/>'
            f"{arrow_svg}</g>"
        )
    else:
        parts.append(
            f'<path id="{path_id}" d="{path_d}" '
            f'fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round"/>'
        )
    parts.append(_svg_edge_hit_area(f'path d="{path_d}"', width, edge_id, attrs_json))
    return "\n".join(parts)


def _svg_node(
    x: float,
    y: float,
    r: float,
    fill: str,
    node_id: str,
    attrs_json: str,
    border_opacity: float = 0.3,
    border: Optional[str] = None,
    border_width: float = 1.5,
) -> str:
    """A node's circle. `border` defaults to a darker `fill`, which must then be a hex color."""
    if border is None:
        border = _darken(fill, 0.3)
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
        f'fill="{fill}" stroke="{border}" stroke-width="{border_width:g}" '
        f'stroke-opacity="{border_opacity}" '
        f"data-node=\"{html_mod.escape(str(node_id))}\" data-attrs='{attrs_json}' "
        f'/>'
    )


def _svg_label(
    x: float,
    y: float,
    r: float,
    text: str,
    color: str,
    font_size: float,
) -> str:
    # Position label below and slightly right of node
    lx = x
    ly = y + r + font_size + 2
    escaped = html_mod.escape(str(text))
    return (
        f'<text x="{lx:.1f}" y="{ly:.1f}" '
        f'text-anchor="middle" '
        f'font-family="{FONT_FAMILY}" font-size="{font_size:.1f}" '
        f'fill="{color}" pointer-events="none">{escaped}</text>'
    )


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

LEGEND_MAX_ITEMS = 12
_LEGEND_FONT = 11.0
_LEGEND_ROW = 17.0
_LEGEND_PAD = 12.0
_LEGEND_RAMP = 90.0
_LEGEND_MAX_CHARS = 22


def _fmt_num(v: float) -> str:
    r = _round_sig(float(v), 3)
    return str(int(r)) if float(r).is_integer() and abs(r) < 1e15 else f"{r:g}"


def _legend_text(text: str) -> str:
    return text if len(text) <= _LEGEND_MAX_CHARS else text[: _LEGEND_MAX_CHARS - 1] + "…"


def _text_width(text: str, size: float = _LEGEND_FONT) -> float:
    return len(text) * size * 0.58


def _svg_legend(
    blocks: List[Dict[str, Any]], colors: "_Colors", th: Theme
) -> Tuple[str, float, float]:
    """The legend for `blocks` (as `render_plot()` gathers them), drawn from (0, 0) in pixels, and
    its width and height.

    A `"swatches"` block lists `(shape, (light, dark), text)` items, the shape being `"node"` (a
    dot) or `"edge"` (a line). A `"ramp"` block draws its `scale` as a bar, labelled with the
    values its `ends` stand for (and the midpoint of a diverging scale), and a gray dot for nodes
    without a value when it is `missing` one.
    """
    parts = []
    y = _LEGEND_PAD
    widest = 0.0
    text_x = 22.0
    font = f'font-family="{FONT_FAMILY}" font-size="{_LEGEND_FONT:g}"'

    def text(x, ty, s, weight=""):
        w = f' font-weight="{weight}"' if weight else ""
        return (
            f'<text x="{x:.1f}" y="{ty:.1f}" {font}{w} fill="{th.label_color}" '
            f'dominant-baseline="middle">{html_mod.escape(s)}</text>'
        )

    for block in blocks:
        title = _legend_text(str(block["title"]))
        parts.append(text(0, y, title, "600"))
        widest = max(widest, _text_width(title) * 1.05)
        y += _LEGEND_ROW
        if block["kind"] == "swatches":
            for shape, pair, label in block["items"]:
                c = colors.pick(*pair)
                if shape == "node":
                    border = colors.pick(_darken(pair[0], 0.3), _darken(pair[1], 0.3))
                    parts.append(
                        f'<circle cx="7" cy="{y:.1f}" r="5" fill="{c}" stroke="{border}" '
                        f'stroke-width="1" stroke-opacity="0.4"/>'
                    )
                else:
                    parts.append(
                        f'<line x1="0" y1="{y:.1f}" x2="14" y2="{y:.1f}" stroke="{c}" '
                        f'stroke-width="3" stroke-linecap="round"/>'
                    )
                label = _legend_text(label)
                parts.append(text(text_x, y, label))
                widest = max(widest, text_x + _text_width(label))
                y += _LEGEND_ROW
        else:
            lo, hi, mid = block["ends"]
            stops = DIVERGING_STOPS if block["scale"] == "diverging" else SEQUENTIAL_STOPS
            first = stops[0][0]
            stop_marks = []
            for k in range(11):
                light, dark = _scale_color(stops, first + (stops[-1][0] - first) * k / 10)
                stop_marks.append(
                    f'<stop offset="{k / 10:g}" stop-color="{colors.pick(light, dark)}"/>'
                )
            # Named after its colors, so that equal ramps on one page can share the name
            grad = "ec-ramp-" + hashlib.sha1("".join(stop_marks).encode()).hexdigest()[:8]
            top = y - _LEGEND_FONT / 2
            parts.append(
                f'<defs><linearGradient id="{grad}" x1="0" y1="1" x2="0" y2="0">'
                + "".join(stop_marks)
                + "</linearGradient></defs>"
                f'<rect x="2" y="{top:.1f}" width="10" height="{_LEGEND_RAMP:g}" rx="2" '
                f'fill="url(#{grad})"/>'
            )
            marks = [(top, _fmt_num(hi)), (top + _LEGEND_RAMP, _fmt_num(lo))]
            if mid is not None:
                marks.insert(1, (top + _LEGEND_RAMP / 2, _fmt_num(mid)))
            for ty, label in marks:
                parts.append(text(text_x, ty, label))
                widest = max(widest, text_x + _text_width(label))
            y = top + _LEGEND_RAMP + _LEGEND_ROW
            if block.get("missing"):
                pair = MISSING_COLOR
                parts.append(
                    f'<circle cx="7" cy="{y:.1f}" r="5" fill="{colors.pick(*pair)}"/>'
                )
                parts.append(text(text_x, y, "no value"))
                widest = max(widest, text_x + _text_width("no value"))
                y += _LEGEND_ROW
        y += _LEGEND_ROW / 2
    return "".join(parts), widest + 2 * _LEGEND_PAD, y


# ---------------------------------------------------------------------------
# Compute viewBox to fit all nodes
# ---------------------------------------------------------------------------


def _compute_viewbox(
    positions: Dict[str, Tuple[float, float]],
    radii: Dict[str, float],
    labels: Optional[Dict[str, str]] = None,
    label_space: float = 20.0,
    padding: float = 30.0,
    frame: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float]:
    """Return (min_x, min_y, width, height) enclosing all nodes and labels.

    With `frame=(width, height)`, the box is widened on its shorter side, around the drawing's
    center, to the frame's aspect ratio. The drawing then fills the frame instead of sitting in
    it with empty bands on two sides, and a background drawn over the box covers the whole frame.
    """
    if not positions:
        return (0, 0, 100, 100)

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for nid, (x, y) in positions.items():
        r = radii.get(nid, 16)
        # Estimate label width: ~7px per character
        label_w = 0.0
        if labels and nid in labels:
            label_w = len(str(labels[nid])) * 7.0 / 2.0
        horiz_extra = max(r, label_w) + padding
        vert_extra = r + padding
        min_x = min(min_x, x - horiz_extra)
        min_y = min(min_y, y - vert_extra)
        max_x = max(max_x, x + horiz_extra)
        max_y = max(max_y, y + vert_extra + label_space)

    w = max(max_x - min_x, 1)
    h = max(max_y - min_y, 1)
    if frame is not None:
        aspect = frame[0] / frame[1]
        if w / h < aspect:
            min_x -= (h * aspect - w) / 2
            w = h * aspect
        else:
            min_y -= (w / aspect - h) / 2
            h = w / aspect
    return (min_x, min_y, w, h)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def _plot_node_values(graph, node_ids, node_data, key, param):
    """Resolve `label_by=` / `size_by=` / `color_by=` to a value per node (keyed by str ID).

    `"id"` is always the node ID. Otherwise a stored node attribute with that name wins, and
    `"label"` (falling back to the ID) and `"degree"` are computed. Any other name that no node has raises, rather than
    silently plotting nothing. Nodes lacking a stored attribute get `None`.
    """
    if key is None:
        return {str(nid): None for nid in node_ids}
    if key == "id":
        return {str(nid): str(nid) for nid in node_ids}
    if any(key in attrs for attrs in node_data.values()):
        return {str(nid): node_data[nid].get(key) for nid in node_ids}
    if key == "label":
        return {
            str(nid): (lbl if (lbl := graph.get_label_from_id(nid)) is not None else str(nid))
            for nid in node_ids
        }
    if key == "degree":
        # get_degree() rather than degrees(), which also writes a "degree" column
        return {str(nid): graph.get_degree(nid) for nid in node_ids}
    raise ValueError(
        f"{param}={key!r}: no node has an attribute {key!r} (built-in options are 'id', "
        f"'label' and 'degree')"
    )


def _round_sig(v: float, digits: int = 4) -> float:
    """`v` to `digits` significant digits, but keeping all of its whole-number digits."""
    if v == 0 or not math.isfinite(v):
        return v
    return round(v, max(0, digits - 1 - math.floor(math.log10(abs(v)))))


def _is_num_vector(v) -> bool:
    return (
        isinstance(v, (list, tuple))
        and len(v) > 0
        and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v)
    )


def _tooltip_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Round the floats in the nodes' (or edges') tooltip values, which can run to 16 digits.

    A single number keeps 4 significant digits. The numbers in a vector (e.g., a stored `pos`) are
    rounded to the same decimal place in every row: a hundredth of the typical spacing of that
    many points spread over the vectors' range. Significant digits would instead show a
    coordinate that is nearly 0 as noise like 1.3e-17.
    """
    decimals = {}
    for key in dict.fromkeys(k for row in rows for k in row):
        vectors = [row[key] for row in rows if _is_num_vector(row.get(key))]
        comps = [c for v in vectors for c in v if math.isfinite(c)]
        if comps:
            spread = max(comps) - min(comps)
            decimals[key] = (
                max(0, math.ceil(math.log10(math.sqrt(len(vectors)) / spread)) + 2)
                if spread > 0
                else 2
            )

    def tidy(key, v):
        if isinstance(v, float):
            return _round_sig(v)
        if key in decimals and _is_num_vector(v):
            return [round(c, decimals[key]) for c in v]
        return v

    return [{k: tidy(k, v) for k, v in row.items()} for row in rows]


def _attrs_json(attrs: Dict[str, Any]) -> str:
    """`attrs` as JSON for a `data-attrs='...'` attribute."""
    return (
        json.dumps(attrs, default=str)
        .replace("&", "&amp;")
        .replace("'", "&#39;")
        .replace("<", "&lt;")
    )


def render_plot(
    graph,
    layout: str = "force",
    color_by: Optional[str] = None,
    edge_color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    label_by: str = "id",
    edge_label_by: Optional[str] = None,
    curved_edges: bool = False,
    theme: Optional[str] = None,
    palette: str = "tableau",
    width: int = 700,
    height: int = 500,
    node_size: float = 16.0,
    edge_width: float = 1.2,
    edge_opacity: float = 0.4,
    cross_community_color: Optional[str] = None,
    node_overlap: Union[str, float] = "remove",
    edge_routing: Union[str, float] = "around",
    show_labels: bool = True,
    output: Optional[str] = None,
    seed: Optional[int] = None,
    edge_width_by: Optional[str] = None,
    color_scale: Optional[str] = None,
    color_midpoint: Optional[float] = None,
    highlight=None,
    highlight_edges=None,
    legend: bool = True,
) -> "StepPlayer":
    """Render a graph as a static SVG wrapped in a StepPlayer."""

    _check_style(layout, theme, stored_ok=True)
    if color_scale is not None and color_scale not in COLOR_SCALES:
        raise ValueError(
            f"Unknown color_scale {color_scale!r}; use one of "
            f"{', '.join(map(repr, COLOR_SCALES))}"
        )
    colors = _Colors(theme)
    th = colors.theme
    if color_by or edge_color_by:
        from edgecase.colors import get_palette

        get_palette(palette)  # raises ValueError for an unknown name

    # --- Extract graph data ---
    node_ids = list(graph.get_node_ids(limit=None))
    if not node_ids:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            f'<rect width="100%" height="100%" fill="{th.bg}"/>'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            f'fill="{th.label_color}" font-family="{FONT_FAMILY}" '
            f'font-size="14">Empty graph</text></svg>'
        )
        return colors.finish(svg, svg, output)

    n = len(node_ids)
    if edge_width == 1.2:
        edge_width = max(0.5, 3.0 - n * 0.06) if n < 40 else max(0.5, 1.2 - (n - 40) * 0.002)
    edge_width = max(edge_width, 0.5)

    # Adaptive node size: radius ≈ 15% of avg node spacing so nodes don't overlap
    if node_size == 16.0:
        avg_spacing = math.sqrt(float(width * height) / max(n, 1))
        node_size = max(3.5, min(16.0, avg_spacing * 0.15))

    # Node data
    node_data = {}
    for nid in node_ids:
        obj = graph.get_node(nid)
        attrs = dict(obj.attributes) if obj and obj.attributes else {}
        node_data[nid] = attrs

    # Per-node values for label_by= / size_by= / color_by= (keyed by str ID)
    label_values = _plot_node_values(graph, node_ids, node_data, label_by, "label_by")
    size_by_values = _plot_node_values(graph, node_ids, node_data, size_by, "size_by")
    color_by_values = _plot_node_values(graph, node_ids, node_data, color_by, "color_by")

    # Edge data (supports multi-edges between same pair)
    is_directed = graph.is_directed()
    edges: List[Tuple[str, str]] = []
    edge_attrs_list: List[Dict] = []
    # Each edge's ID and weight (None for an edge drawn without its edge object)
    edge_id_list: List[Optional[int]] = []
    edge_weights: List[Optional[float]] = []
    # The start of each edge's tooltip: its ID, ends (named as their labels are drawn, and in the
    # order stored) and weight. Its attributes follow
    edge_tip_heads: List[Dict] = []
    arrow = "→" if is_directed else "—"

    def drawn_name(nid):
        value = label_values[str(nid)]
        return str(nid) if value is None else str(value)

    def tip_head(src, tgt, e_obj=None):
        if e_obj is not None:
            src, tgt = e_obj.source, e_obj.target
        head = {"id": e_obj.id} if e_obj is not None else {}
        head["edge"] = f"{drawn_name(src)} {arrow} {drawn_name(tgt)}"
        if e_obj is not None and e_obj.weight is not None:
            head["weight"] = e_obj.weight
        return head

    seen_pairs: set = set()
    for nid in node_ids:
        try:
            neighbors = graph.get_neighbors(nid)
        except Exception:
            continue
        for nb in neighbors:
            if is_directed:
                pair_key = (str(nid), str(nb))
            else:
                pair_key = tuple(sorted([str(nid), str(nb)]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                try:
                    eids = graph.find_edge_ids(nid, nb)
                except Exception:
                    eids = []
                if not eids:
                    edges.append((str(nid), str(nb)))
                    edge_attrs_list.append({})
                    edge_tip_heads.append(tip_head(nid, nb))
                    edge_id_list.append(None)
                    edge_weights.append(None)
                else:
                    for eid in eids:
                        edges.append((str(nid), str(nb)))
                        edge_id_list.append(eid)
                        try:
                            e_obj = graph.get_edge(eid)
                            edge_attrs_list.append(
                                dict(e_obj.attributes) if e_obj and e_obj.attributes else {}
                            )
                            edge_tip_heads.append(tip_head(nid, nb, e_obj))
                            edge_weights.append(e_obj.weight)
                        except Exception:
                            edge_attrs_list.append({})
                            edge_tip_heads.append(tip_head(nid, nb))
                            edge_weights.append(None)

    def edge_values(key):
        """Each edge's value of the edge attribute `key` (or its weight, for `"weight"`), or
        `None` if no edge has the attribute."""
        if any(key in a for a in edge_attrs_list):
            return [a.get(key) for a in edge_attrs_list]
        if key == "weight" and any(w is not None for w in edge_weights):
            return list(edge_weights)
        return None

    # edge_color_by= names an edge attribute, or else a node attribute whose groups the edges
    # take on
    edge_color_attr_values = edge_values(edge_color_by) if edge_color_by else None
    edge_color_values = (
        _plot_node_values(graph, node_ids, node_data, edge_color_by, "edge_color_by")
        if edge_color_by and edge_color_attr_values is None
        else {str(nid): None for nid in node_ids}
    )
    edge_width_values = None
    if edge_width_by:
        edge_width_values = edge_values(edge_width_by)
        if edge_width_values is None:
            raise ValueError(
                f"edge_width_by={edge_width_by!r}: no edge has an attribute {edge_width_by!r}"
            )


    # Convert node_ids to strings for consistency
    str_ids = [str(nid) for nid in node_ids]

    # --- Layout ---
    if layout == "circular":
        positions = _circular_layout(str_ids, float(width), float(height), edges=edges)
    elif layout == "random":
        positions = _random_layout(str_ids, float(width), float(height), seed=seed)
    elif layout not in LAYOUTS:
        positions = _stored_layout(graph, node_ids, layout, float(width), float(height))
    else:
        n = len(str_ids)
        fr_iters = 800 if n > 300 else (600 if n > 150 else 400)
        positions = _force_directed_layout(
            str_ids,
            edges,
            float(width),
            float(height),
            iterations=fr_iters,
            seed=seed,
        )

    # --- Aesthetic mappings ---

    # Color mapping. `node_colors` holds each node's (light, dark) colors; the legend's blocks
    # gather in `legend_blocks`
    legend_blocks: List[Dict[str, Any]] = []
    node_colors: Dict[str, Tuple[str, str]] = {}
    node_community = {}
    if color_by:
        cat_values = [color_by_values[nid] for nid in str_ids]
        scale = _color_scale(cat_values, color_scale, "color_by")
        if scale == "categorical":
            for nid, val in zip(str_ids, cat_values):
                node_community[nid] = val
            # Nodes without a value are gray, as on a numeric scale, rather than taking a hue
            order = _sorted_values(cat_values)
            present = [v for v in order if v is not None]
            palette_map = _build_color_map(present, palette)
            pairs = {v: (palette_map[v],) * 2 for v in present}
            pairs[None] = MISSING_COLOR
            for nid, val in zip(str_ids, cat_values):
                node_colors[nid] = pairs.get(val, (DEFAULT_NODE_COLOR,) * 2)
            legend_blocks.append(
                {
                    "title": color_by,
                    "kind": "swatches",
                    "items": [
                        ("node", pairs[v], "no value" if v is None else str(v)) for v in order
                    ],
                }
            )
        else:
            mapped, ends = _numeric_colors(cat_values, scale, color_midpoint)
            node_colors = dict(zip(str_ids, mapped))
            legend_blocks.append(
                {
                    "title": color_by,
                    "kind": "ramp",
                    "scale": scale,
                    "ends": ends,
                    "missing": any(not _is_number(v) for v in cat_values),
                }
            )
    else:
        for nid in str_ids:
            node_colors[nid] = (DEFAULT_NODE_COLOR, DEFAULT_NODE_COLOR)
    color_map = {nid: colors.pick(*pair) for nid, pair in node_colors.items()}

    # Edge color mapping: by an edge attribute's values, or by the groups of the endpoints
    edge_community = {}
    edge_attr_colors: Optional[List[Optional[Tuple[str, str]]]] = None
    if edge_color_attr_values is not None:
        scale = _color_scale(edge_color_attr_values, None, "edge_color_by")
        if scale == "categorical":
            order = _sorted_values(edge_color_attr_values)
            present = [v for v in order if v is not None]
            ecb_palette = _build_color_map(present, palette)
            edge_attr_colors = [
                (ecb_palette[v],) * 2 if v is not None else None for v in edge_color_attr_values
            ]
            legend_blocks.append(
                {
                    "title": edge_color_by,
                    "kind": "swatches",
                    "items": [("edge", (ecb_palette[v],) * 2, str(v)) for v in present],
                }
            )
        else:
            mapped, ends = _numeric_colors(edge_color_attr_values, scale, None)
            edge_attr_colors = [
                pair if _is_number(v) else None for pair, v in zip(mapped, edge_color_attr_values)
            ]
            legend_blocks.append(
                {"title": edge_color_by, "kind": "ramp", "scale": scale, "ends": ends}
            )
            # A color scale reads poorly through the default, faint edges
            if edge_opacity == 0.4:
                edge_opacity = 0.85
    elif edge_color_by:
        for nid in str_ids:
            edge_community[nid] = edge_color_values[nid]
        ecb_vals = [v for v in edge_community.values() if v is not None]
        order = _sorted_values(ecb_vals)
        ecb_palette = _build_color_map(order, palette) if ecb_vals else {}
        ecb_node_colors = {
            nid: ecb_palette.get(edge_community.get(nid), th.edge_color) for nid in str_ids
        }
        default_cross = ("#52525b", "#a1a1aa")
        ecb_cross_color = cross_community_color or colors.pick(*default_cross)
        cross_item = (
            "edge",
            (cross_community_color,) * 2 if cross_community_color else default_cross,
            "between groups",
        )
        if edge_color_by == color_by and legend_blocks and legend_blocks[0]["kind"] == "swatches":
            legend_blocks[0]["items"].append(cross_item)
        else:
            legend_blocks.append(
                {
                    "title": edge_color_by,
                    "kind": "swatches",
                    "items": [("edge", (ecb_palette[v],) * 2, str(v)) for v in order]
                    + [cross_item],
                }
            )

    # Edge widths from edge_width_by=, from half to two and a half times the base width
    edge_widths = [edge_width] * len(edges)
    if edge_width_values is not None:
        nums = [v for v in edge_width_values if _is_number(v)]
        if nums:
            lo, hi = min(nums), max(nums)
            span = (hi - lo) or 1.0
            edge_widths = [
                edge_width * (0.5 + 2.0 * (v - lo) / span) if _is_number(v) else edge_width
                for v in edge_width_values
            ]

    # Highlighting: each node's and edge's level ("accent", "soft", or None for muted)
    node_level: Dict[str, Optional[str]] = {}
    edge_level: List[Optional[str]] = []
    if highlight is not None or highlight_edges is not None:
        picked = _resolve_highlight(graph, highlight, highlight_edges)
        for nid in str_ids:
            node_level[nid] = None
        edge_level = [None] * len(edges)
        index_of_edge = {eid: i for i, eid in enumerate(edge_id_list) if eid is not None}
        for level in reversed(HIGHLIGHT_LEVELS):
            for nid in picked[level][0]:
                node_level[str(nid)] = level
            for eid in picked[level][1]:
                if eid in index_of_edge:
                    edge_level[index_of_edge[eid]] = level

    # Legends are shown only for what they can tell apart: a palette's colors repeat after
    # its length, so a categorical block with more values than that is left out
    if not legend:
        legend_blocks = []
    else:
        from edgecase.colors import get_palette

        n_colors = len(get_palette(palette)) if color_by or edge_color_by else 0
        legend_blocks = [
            b
            for b in legend_blocks
            if b["kind"] == "ramp"
            or sum(1 for item in b["items"] if item[2] != "between groups")
            <= min(n_colors, LEGEND_MAX_ITEMS)
        ]

    # Size mapping
    radii = {}
    if size_by:
        size_values = []
        for nid in str_ids:
            val = size_by_values[nid]
            try:
                size_values.append(float(val))
            except (TypeError, ValueError):
                size_values.append(None)

        valid = [v for v in size_values if v is not None]
        if valid:
            lo, hi = min(valid), max(valid)
            rng = hi - lo if hi != lo else 1.0
            for nid, val in zip(str_ids, size_values):
                if val is not None:
                    norm = (val - lo) / rng
                    radii[nid] = node_size * (0.5 + norm * 1.5)
                else:
                    radii[nid] = node_size
        else:
            radii = {nid: node_size for nid in str_ids}
    else:
        radii = {nid: node_size for nid in str_ids}

    # --- Overlap removal (force layout only) ---
    if node_overlap != "allow" and layout not in ("circular",) and n >= 2:
        overlap_pad = float(node_overlap) if isinstance(node_overlap, (int, float)) else 2.0
        positions = _remove_overlaps(positions, radii, padding=overlap_pad, seed=seed)

    # --- Build label map for viewBox computation ---
    label_map = {}
    if show_labels:
        for nid in str_ids:
            value = label_values[nid]
            label_map[nid] = nid if value is None else str(value)

    # --- Compute viewBox ---
    # The legend gets a column of its own at the right, so that it never covers the drawing
    legend_svg, legend_w = "", 0
    if legend_blocks:
        legend_svg, legend_w, _ = _svg_legend(legend_blocks, colors, th)
    legend_w = min(legend_w, width * 0.4)
    vb = _compute_viewbox(
        positions,
        radii,
        labels=label_map if show_labels else None,
        label_space=20 if show_labels else 0,
        frame=(width - legend_w, height),
    )
    if legend_blocks:
        # Viewbox units per pixel
        unit = vb[3] / height
        vb = (vb[0], vb[1], vb[2] + legend_w * unit, vb[3])
    vb_str = f"{vb[0]:.1f} {vb[1]:.1f} {vb[2]:.1f} {vb[3]:.1f}"

    # --- Build SVG ---
    svg_parts = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="{vb_str}" '
        f'style="font-family: {FONT_FAMILY};">'
    )
    svg_parts.append(
        f'<rect x="{vb[0]:.1f}" y="{vb[1]:.1f}" width="{vb[2]:.1f}" height="{vb[3]:.1f}" fill="{th.bg}"/>'
    )

    # Precompute routed edge paths (avoid nodes for non-curved edges)
    edge_routes: Dict[Tuple[str, str], Optional[List[Tuple[float, float]]]] = {}
    if edge_routing != "straight" and not curved_edges and len(str_ids) >= 3:
        route_margin = float(edge_routing) if isinstance(edge_routing, (int, float)) else 3.0
        for src, tgt in edges:
            if src in positions and tgt in positions:
                route = _route_edge(src, tgt, positions, radii, str_ids, margin=route_margin)
                if route is not None:
                    edge_routes[(src, tgt)] = route

    # Compute fan-out bows for parallel / reciprocal edges
    edge_bows, edge_loop_sizes = _compute_edge_fans(edges)

    edge_tips = _tooltip_rows(
        [{**head, **e_attrs} for head, e_attrs in zip(edge_tip_heads, edge_attrs_list)]
    )

    # Highlighted elements are drawn over the muted ones, and accent over soft
    level_rank = {None: 0, "soft": 1, "accent": 2}
    accent = colors.pick(*ACCENT_COLOR)

    # Edges first (painter's algorithm)
    edge_order = range(len(edges))
    if edge_level:
        edge_order = sorted(edge_order, key=lambda j: level_rank[edge_level[j]])
    for i in edge_order:
        src, tgt = edges[i]
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        r_src = radii.get(src, node_size)
        r_dst = radii.get(tgt, node_size)

        # Edge color
        e_color = th.edge_color
        if edge_attr_colors is not None:
            if edge_attr_colors[i] is not None:
                e_color = colors.pick(*edge_attr_colors[i])
        elif edge_color_by and edge_community:
            src_comm = edge_community.get(src)
            tgt_comm = edge_community.get(tgt)
            if src_comm is not None and src_comm == tgt_comm:
                e_color = ecb_node_colors.get(src, th.edge_color)
            elif src_comm is not None or tgt_comm is not None:
                e_color = ecb_cross_color
        elif (
            color_by
            and node_community.get(src) == node_community.get(tgt)
            and node_community.get(src) is not None
        ):
            e_color = color_map.get(src, th.edge_color)

        e_width = edge_widths[i]
        e_opacity = edge_opacity
        if edge_level:
            level = edge_level[i]
            if level is None:
                e_color = th.edge_color
                e_opacity = edge_opacity * 0.5
            else:
                # A highlighted edge keeps a color that edge_color_by= gives it, or else takes
                # the accent (rather than its group's color from color_by=, so that a path or
                # matching is all of one color)
                if not edge_color_by or e_color == th.edge_color:
                    e_color = accent
                if level == "accent":
                    e_width = max(e_width * 2.0, 1.5)
                    e_opacity = max(edge_opacity, 0.9)
                else:
                    e_width = max(e_width * 1.3, 1.0)
                    e_opacity = max(edge_opacity, 0.5)

        attrs_j = _attrs_json(edge_tips[i])
        eid = f"{src}-{tgt}-{i}"

        bow = edge_bows[i]

        # Self-loop
        if src == tgt:
            svg_parts.append(
                _svg_edge_loop(
                    x1,
                    y1,
                    r_src,
                    e_color,
                    e_width,
                    e_opacity,
                    is_directed,
                    eid,
                    attrs_j,
                    loop_size=edge_loop_sizes[i],
                )
            )
            continue

        route = edge_routes.get((src, tgt))
        if curved_edges:
            svg_parts.append(
                _svg_edge_curved(
                    x1,
                    y1,
                    x2,
                    y2,
                    r_src,
                    r_dst,
                    e_color,
                    e_width,
                    e_opacity,
                    is_directed,
                    eid,
                    attrs_j,
                    bow=0.2 + bow,
                )
            )
        elif abs(bow) > 1e-6:
            svg_parts.append(
                _svg_edge_curved(
                    x1,
                    y1,
                    x2,
                    y2,
                    r_src,
                    r_dst,
                    e_color,
                    e_width,
                    e_opacity,
                    is_directed,
                    eid,
                    attrs_j,
                    bow=bow,
                )
            )
        elif route is not None:
            svg_parts.append(
                _svg_edge_routed(
                    route,
                    r_src,
                    r_dst,
                    e_color,
                    e_width,
                    e_opacity,
                    is_directed,
                    eid,
                    attrs_j,
                )
            )
        else:
            svg_parts.append(
                _svg_edge_straight(
                    x1,
                    y1,
                    x2,
                    y2,
                    r_src,
                    r_dst,
                    e_color,
                    e_width,
                    e_opacity,
                    is_directed,
                    eid,
                    attrs_j,
                )
            )

    # Edge labels (stretch goal)
    if edge_label_by:
        edge_label_values = edge_values(edge_label_by) or []
        # Each edge's place among the edges between the same two nodes, numbered as in
        # `_compute_edge_fans()`
        fan_members: Dict[Tuple[str, str], List[int]] = {}
        for i, (src, tgt) in enumerate(edges):
            if src != tgt:
                fan_members.setdefault((min(src, tgt), max(src, tgt)), []).append(i)
        fan_place = {i: (k, m) for m in fan_members.values() for k, i in enumerate(m)}
        for i, (src, tgt) in enumerate(edges):
            if src not in positions or tgt not in positions:
                continue
            label_val = edge_label_values[i] if i < len(edge_label_values) else None
            if label_val is None:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[tgt]
            eid = f"{src}-{tgt}-{i}"
            k, fan = fan_place.get(i, (0, [i]))

            if curved_edges:
                # For textPath, text reads left-to-right along the path.
                # If the path runs right-to-left, add a reversed copy for the label.
                text_path_ref = f"path-{eid}"
                if x1 > x2:
                    # Emit a hidden reversed path so text reads correctly
                    dx_e = x2 - x1
                    dy_e = y2 - y1
                    d_e = math.sqrt(dx_e * dx_e + dy_e * dy_e)
                    if d_e > 1e-6:
                        ux_e, uy_e = dx_e / d_e, dy_e / d_e
                        px_e, py_e = -uy_e, ux_e
                        r_src = radii.get(src, node_size)
                        r_dst = radii.get(tgt, node_size)
                        bow_val = 0.2 + edge_bows[i]
                        cmx = (x1 + x2) / 2 + px_e * d_e * bow_val
                        cmy = (y1 + y2) / 2 + py_e * d_e * bow_val
                        # Reversed: swap start/end
                        cdx_s2 = cmx - x2
                        cdy_s2 = cmy - y2
                        cd_s2 = math.sqrt(cdx_s2**2 + cdy_s2**2)
                        sx2 = x2 + (cdx_s2 / cd_s2) * r_dst if cd_s2 > 1e-6 else x2
                        sy2 = y2 + (cdy_s2 / cd_s2) * r_dst if cd_s2 > 1e-6 else y2
                        cdx_e2 = cmx - x1
                        cdy_e2 = cmy - y1
                        cd_e2 = math.sqrt(cdx_e2**2 + cdy_e2**2)
                        ex2 = x1 + (cdx_e2 / cd_e2) * r_src if cd_e2 > 1e-6 else x1
                        ey2 = y1 + (cdy_e2 / cd_e2) * r_src if cd_e2 > 1e-6 else y1
                        rev_id = f"path-{eid}-rev"
                        svg_parts.append(
                            f'<path id="{rev_id}" '
                            f'd="M {sx2:.1f} {sy2:.1f} Q {cmx:.1f} {cmy:.1f} {ex2:.1f} {ey2:.1f}" '
                            f'fill="none" stroke="none"/>'
                        )
                        text_path_ref = rev_id
                svg_parts.append(
                    f'<text font-family="{FONT_FAMILY}" font-size="10" '
                    f'fill="{th.label_color}" fill-opacity="0.7" pointer-events="none">'
                    f'<textPath href="#{text_path_ref}" startOffset="50%" text-anchor="middle">'
                    f"{html_mod.escape(str(label_val))}</textPath></text>"
                )
            elif len(fan) >= 3 and math.dist((x1, y1), (x2, y2)) > 1e-6:
                # Three or more edges between two nodes leave no room beside the inner ones, so
                # each label sits on its own edge, ringed by the background. The labels step
                # along the edges too, in the order of the curves, each step the length of a
                # label's box measured along the edges, so that neighbors don't overlap however
                # the edges are angled (as far as the edges are long enough)
                a_x, a_y, b_x, b_y = (x1, y1, x2, y2) if src <= tgt else (x2, y2, x1, y1)
                dist = math.dist((a_x, a_y), (b_x, b_y))
                box_w = max(
                    _text_width(str(edge_label_values[j]), 10)
                    for j in fan
                    if j < len(edge_label_values) and edge_label_values[j] is not None
                )
                step = (box_w * abs(b_x - a_x) + 11 * abs(b_y - a_y)) / dist + 3
                rank = k - (len(fan) - 1) / 2
                t = 0.5 + rank * min(step / dist, 0.7 / (len(fan) - 1))
                # The edge's offset from the straight line, on the side of the A-to-B normal
                v = edge_bows[i] if src <= tgt else -edge_bows[i]
                bulge = dist * v * 2 * t * (1 - t)
                mx = a_x + (b_x - a_x) * t - (b_y - a_y) / dist * bulge
                my = a_y + (b_y - a_y) * t + (b_x - a_x) / dist * bulge
                svg_parts.append(
                    f'<text x="{mx:.1f}" y="{my + 3.5:.1f}" text-anchor="middle" '
                    f'font-family="{FONT_FAMILY}" font-size="10" '
                    f'fill="{th.label_color}" fill-opacity="0.7" stroke="{th.bg}" '
                    f'stroke-width="3" stroke-linejoin="round" paint-order="stroke" '
                    f'pointer-events="none">{html_mod.escape(str(label_val))}</text>'
                )
            else:
                # Midpoint label, offset perpendicular to the edge. A parallel edge is drawn as a
                # curve, so its label goes by the middle of the curve, on the outside of it. Beside
                # a steep edge, the text starts or ends at that point, so it stays clear of the line
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                anchor = "middle"
                dx, dy = x2 - x1, y2 - y1
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-6:
                    px, py = -dy / dist, dx / dist
                    bow = edge_bows[i]
                    offset = 8 + dist * abs(bow) / 2 if abs(bow) > 1e-6 else 8
                    if bow < -1e-6:
                        px, py = -px, -py
                    mx += px * offset
                    my += py * offset
                    if abs(px) > 0.7:
                        anchor = "start" if px > 0 else "end"
                        my += 3
                svg_parts.append(
                    f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="{anchor}" '
                    f'font-family="{FONT_FAMILY}" font-size="10" '
                    f'fill="{th.label_color}" fill-opacity="0.7" '
                    f'pointer-events="none">{html_mod.escape(str(label_val))}</text>'
                )

    # Node tooltips: the ID and the label as well, when it differs from the ID, since either may
    # be drawn under the node, then the node's attributes
    node_tips = []
    for raw_nid, nid in zip(node_ids, str_ids):
        tip = {"id": nid}
        n_label = graph.get_label_from_id(raw_nid)
        if n_label is not None and str(n_label) != nid:
            tip["label"] = n_label
        node_tips.append({**tip, **node_data.get(raw_nid, {})})
    node_tips = dict(zip(str_ids, _tooltip_rows(node_tips)))

    # Nodes on top
    node_order = list(str_ids)
    if node_level:
        node_order.sort(key=lambda v: level_rank[node_level[v]])
    surface = (LIGHT.bg, DARK.bg)
    for nid in node_order:
        if nid not in positions:
            continue
        x, y = positions[nid]
        r = radii.get(nid, node_size)
        light, dark = node_colors.get(nid, (DEFAULT_NODE_COLOR, DEFAULT_NODE_COLOR))
        border_opacity, border_width = 0.3, 1.5
        if node_level:
            level = node_level[nid]
            if not color_by:
                light, dark = ACCENT_COLOR
            if level is None:
                light, dark = MUTED_NODE_COLOR
            elif level == "soft":
                # A ring of the color around a pale version of it
                ring = colors.pick(light, dark)
                light, dark = _mix(light, surface[0], 0.7), _mix(dark, surface[1], 0.7)
                border_opacity, border_width = 1.0, 2.0
        fill = colors.pick(light, dark)
        border = colors.pick(_darken(light, 0.3), _darken(dark, 0.3))
        if node_level and node_level[nid] == "soft":
            border = ring
        svg_parts.append(
            _svg_node(
                x,
                y,
                r,
                fill,
                nid,
                _attrs_json(node_tips[nid]),
                border_opacity=border_opacity,
                border=border,
                border_width=border_width,
            )
        )

    # Labels
    if show_labels:
        font_size = max(8, min(13, node_size * 0.7))
        muted_label = colors.pick(*MUTED_LABEL_COLOR)
        for nid in str_ids:
            if nid not in positions:
                continue
            x, y = positions[nid]
            r = radii.get(nid, node_size)

            text = label_map[nid]
            color = muted_label if node_level and node_level[nid] is None else th.label_color

            svg_parts.append(_svg_label(x, y, r, text, color, font_size))

    if legend_blocks:
        svg_parts.append(
            f'<g class="ec-legend" transform="translate('
            f"{vb[0] + vb[2] - (legend_w - _LEGEND_PAD) * unit:.1f} "
            f'{vb[1]:.1f}) scale({unit:.4f})">{legend_svg}</g>'
        )

    svg_parts.append("</svg>")
    svg_str = "\n".join(svg_parts)

    # --- HTML wrapper with hover tooltips ---
    html_str = _generate_html_wrapper(svg_str, th, width, height)

    return colors.finish(svg_str, html_str, output)


# ---------------------------------------------------------------------------
# HTML wrapper with JS hover tooltips
# ---------------------------------------------------------------------------

_tooltip_counter = 0


def _generate_html_wrapper(
    svg_str: str,
    theme: Theme,
    width: int,
    height: int,
) -> str:
    global _tooltip_counter
    _tooltip_counter += 1
    tip_id = f"ec-tooltip-{_tooltip_counter}"
    # Centered in the page or cell, and scaled down (keeping its shape) where it is wider than that.
    # Quarto puts no space around cell outputs, so the margin keeps the plot off the code and text
    # around it (margin rather than padding, since the tooltip is placed from the SVG's corner)
    return f"""<div class="ec-plot" style="position:relative;width:fit-content;max-width:100%;margin:1em auto">
<style>.ec-plot > svg {{ display:block; max-width:100%; height:auto; }}</style>
{svg_str}
<div id="{tip_id}" style="
  display:none;position:absolute;padding:6px 10px;border-radius:6px;
  font-family:{FONT_FAMILY};font-size:12px;pointer-events:none;
  background:{theme.tooltip_bg};color:{theme.tooltip_fg};
  box-shadow:0 2px 8px rgba(0,0,0,0.18);max-width:min(260px,100%);box-sizing:border-box;
  z-index:999;line-height:1.4;overflow-wrap:anywhere;
"></div>
<script>
(function(){{
  var tip=document.currentScript.parentElement.querySelector('#{tip_id}');
  var svg=document.currentScript.parentElement.querySelector('svg');
  if(!svg||!tip) return;
  // Next to the pointer but inside the plot (flipped to the pointer's left near the right edge),
  // since anything past its edges is clipped, and scrolls, in a notebook or Quarto cell
  function place(ev){{
    var r=svg.getBoundingClientRect();
    var x=ev.clientX-r.left+12,y=ev.clientY-r.top-10;
    if(x+tip.offsetWidth>r.width) x=ev.clientX-r.left-12-tip.offsetWidth;
    tip.style.left=Math.max(0,Math.min(x,r.width-tip.offsetWidth))+'px';
    tip.style.top=Math.max(0,Math.min(y,r.height-tip.offsetHeight))+'px';
  }}
  function show(el,ev){{
    var raw=el.getAttribute('data-attrs');
    if(!raw) return;
    try{{var d=JSON.parse(raw)}}catch(e){{return}}
    var h='';
    for(var k in d){{
      var v=d[k];
      if(Array.isArray(v)) v=v.join(', ');
      h+='<div><b>'+k+'</b>: '+v+'</div>';
    }}
    if(!h) return;
    tip.innerHTML=h;tip.style.display='block';
    place(ev);
  }}
  svg.addEventListener('mouseover',function(ev){{
    var t=ev.target;
    if(t.hasAttribute('data-node')||t.hasAttribute('data-edge')) show(t,ev);
  }});
  svg.addEventListener('mousemove',function(ev){{
    if(tip.style.display==='block') place(ev);
  }});
  svg.addEventListener('mouseout',function(ev){{
    var t=ev.target;
    if(t.hasAttribute('data-node')||t.hasAttribute('data-edge'))
      tip.style.display='none';
  }});
}})();
</script>
</div>"""


# ---------------------------------------------------------------------------
# Interactive traversal visualization
# ---------------------------------------------------------------------------

_traversal_counter = 0


def _infer_tree_edges(
    visit_order: List[str],
    adjacency: Dict[str, List[str]],
    earliest: bool = False,
) -> List[Tuple[str, str]]:
    """Infer tree edges from a visit order.

    For each node after the first, the tree edge connects it to an already-visited neighbor:
    the one that appears latest in visit_order (the DFS parent), or with `earliest=True` the one
    that appears first (the BFS parent, since BFS discovers a node from the earliest-visited
    neighbor it has). `adjacency` maps each node to its possible parents (its neighbors, or its
    predecessors on a directed graph).
    """
    order_idx = {nid: i for i, nid in enumerate(visit_order)}
    tree_edges: List[Tuple[str, str]] = []
    visited_set: set = set()
    for i, nid in enumerate(visit_order):
        if i == 0:
            visited_set.add(nid)
            continue
        best_parent: Optional[str] = None
        best_pos = -1
        for nb in adjacency.get(nid, []):
            if nb in visited_set:
                pos = order_idx.get(nb, -1)
                if best_parent is None or (pos < best_pos if earliest else pos > best_pos):
                    best_pos = pos
                    best_parent = nb
        if best_parent is not None:
            tree_edges.append((best_parent, nid))
        visited_set.add(nid)
    return tree_edges


def _build_traversal_steps(
    visit_order: List[str],
    tree_edges: List[Tuple[str, str]],
    adjacency: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Precompute traversal step data for the JS state machine."""
    steps: List[Dict[str, Any]] = []
    visited_set: set = set()
    tree_edges_so_far: List[List[str]] = []
    step_labels: Dict[str, int] = {}

    tree_edge_for_node: Dict[str, Tuple[str, str]] = {}
    for src, tgt in tree_edges:
        tree_edge_for_node[tgt] = (src, tgt)

    # Step 0: all unvisited, start node is frontier
    steps.append(
        {
            "current": None,
            "current_edge": None,
            "visited": [],
            "frontier": [visit_order[0]] if visit_order else [],
            "tree_edges": [],
            "step_labels": {},
        }
    )

    for i, nid in enumerate(visit_order):
        visited_set.add(nid)
        step_labels[nid] = i + 1

        current_edge = None
        if nid in tree_edge_for_node:
            te = tree_edge_for_node[nid]
            current_edge = [te[0], te[1]]
            tree_edges_so_far.append(current_edge)

        frontier: set = set()
        for v in visited_set:
            for nb in adjacency.get(v, []):
                if nb not in visited_set:
                    frontier.add(nb)

        steps.append(
            {
                "current": nid,
                "current_edge": current_edge,
                "visited": list(step_labels),  # in visit order
                "frontier": sorted(frontier),
                "tree_edges": [list(e) for e in tree_edges_so_far],
                "step_labels": dict(step_labels),
            }
        )

    return steps


def _sanitize_svg_id(nid: str) -> str:
    """Make a node ID safe for use in SVG id attributes."""
    safe = ""
    for c in str(nid):
        safe += c if c.isalnum() or c in "-_" else "_"
    if safe and safe[0].isdigit():
        safe = "n" + safe
    return safe or "node"


def render_traversal(
    graph,
    visit_order: List[str],
    tree_edges: Optional[List[Tuple[str, str]]] = None,
    algorithm: str = "",
    layout: str = "force",
    theme: Optional[str] = None,
    width: int = 700,
    height: int = 500,
    node_size: float = 18.0,
    edge_width: float = 1.5,
    node_overlap: Union[str, float] = "remove",
    show_labels: bool = True,
    output: Optional[str] = None,
    seed: Optional[int] = None,
) -> "StepPlayer":
    """Render an interactive traversal StepPlayer with a step-through slider."""
    global _traversal_counter
    _traversal_counter += 1
    uid = f"tv{_traversal_counter}"

    _check_style(layout, theme, stored_ok=True)
    colors = _Colors(theme)
    th = colors.theme

    # --- Theme-dependent traversal colors ---
    unvisited_fill = colors.pick("#d1d5db", "#4b5563")
    frontier_fill = colors.pick("#93c5fd", "#60a5fa")
    current_fill = colors.pick("#2563eb", "#3b82f6")
    accent = colors.pick("#2563eb", "#60a5fa")
    tree_edge_color = colors.pick("#2563eb", "#60a5fa")
    current_edge_color = colors.pick("#1d4ed8", "#93c5fd")
    nontree_visited_edge = colors.pick("#cbd5e1", "#64748b")
    unvisited_edge_opacity = colors.pick(0.2, 0.35)

    # --- Extract graph data (same pattern as render_plot) ---
    node_ids = list(graph.get_node_ids(limit=None))
    str_ids = [str(nid) for nid in node_ids]

    n = len(node_ids)
    if edge_width == 1.5:
        edge_width = max(0.5, 3.0 - n * 0.06) if n < 40 else max(0.5, 1.5 - (n - 40) * 0.002)
    edge_width = max(edge_width, 0.5)

    node_data: Dict[str, Dict] = {}
    for nid in node_ids:
        obj = graph.get_node(nid)
        attrs = dict(obj.attributes) if obj and obj.attributes else {}
        node_data[str(nid)] = attrs

    is_directed = graph.is_directed()
    edges: List[Tuple[str, str]] = []
    seen: set = set()
    adjacency: Dict[str, List[str]] = {nid: [] for nid in str_ids}
    for nid in node_ids:
        try:
            neighbors = graph.get_neighbors(nid)
        except Exception:
            continue
        for nb in neighbors:
            if is_directed:
                key = (str(nid), str(nb))
            else:
                key = tuple(sorted([str(nid), str(nb)]))
            if key not in seen:
                seen.add(key)
                edges.append((str(nid), str(nb)))
            adjacency.setdefault(str(nid), []).append(str(nb))
            if not is_directed:
                adjacency.setdefault(str(nb), []).append(str(nid))

    # Deduplicate adjacency lists
    for k in adjacency:
        adjacency[k] = list(dict.fromkeys(adjacency[k]))

    visit_order_str = [str(v) for v in visit_order]

    # --- Infer tree edges if not provided ---
    if tree_edges is None:
        # A node's tree parent reaches it by an edge, so on directed graphs look at the node's
        # predecessors (the adjacency lists hold successors only)
        parent_candidates = adjacency
        if is_directed:
            parent_candidates = {nid: [] for nid in adjacency}
            for src, targets in adjacency.items():
                for tgt in targets:
                    parent_candidates.setdefault(tgt, []).append(src)
        is_bfs = algorithm.strip().upper().startswith("BFS")
        tree_edges_str = _infer_tree_edges(visit_order_str, parent_candidates, earliest=is_bfs)
    else:
        tree_edges_str = [(str(s), str(t)) for s, t in tree_edges]

    # --- Build step data ---
    steps = _build_traversal_steps(visit_order_str, tree_edges_str, adjacency)

    # --- Layout ---
    if layout == "circular":
        positions = _circular_layout(str_ids, float(width), float(height), edges=edges)
    elif layout == "random":
        positions = _random_layout(str_ids, float(width), float(height), seed=seed)
    elif layout not in LAYOUTS:
        positions = _stored_layout(graph, node_ids, layout, float(width), float(height))
    else:
        n = len(str_ids)
        fr_iters = 800 if n > 300 else (600 if n > 150 else 400)
        positions = _force_directed_layout(
            str_ids,
            edges,
            float(width),
            float(height),
            iterations=fr_iters,
            seed=seed,
        )

    # --- Color map (for visited state) ---
    color_map: Dict[str, str] = {}
    for nid in str_ids:
        color_map[nid] = DEFAULT_NODE_COLOR

    radii = {nid: node_size for nid in str_ids}

    # --- Overlap removal (force/random layout) ---
    if node_overlap != "allow" and layout not in ("circular",) and len(str_ids) >= 2:
        overlap_pad = float(node_overlap) if isinstance(node_overlap, (int, float)) else 2.0
        positions = _remove_overlaps(positions, radii, padding=overlap_pad, seed=seed)

    # --- Compute viewBox ---
    # Nodes show their labels, or their IDs if they have none
    label_map = {}
    if show_labels:
        for nid in node_ids:
            obj = graph.get_node(nid)
            label_map[str(nid)] = str(obj.label) if obj and obj.label else str(nid)
    vb = _compute_viewbox(
        positions,
        radii,
        labels=label_map if show_labels else None,
        label_space=20 if show_labels else 0,
        frame=(width, height),
    )
    vb_str = f"{vb[0]:.1f} {vb[1]:.1f} {vb[2]:.1f} {vb[3]:.1f}"

    # --- Build SVG ---
    svg_parts: List[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="{vb_str}" '
        f'id="{uid}-svg" '
        f'style="font-family: {FONT_FAMILY};">'
    )
    svg_parts.append(
        f'<rect x="{vb[0]:.1f}" y="{vb[1]:.1f}" '
        f'width="{vb[2]:.1f}" height="{vb[3]:.1f}" fill="{th.bg}"/>'
    )

    # Defs for glow filter
    svg_parts.append(
        '<defs><filter id="blur-glow"><feGaussianBlur stdDeviation="3" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter></defs>"
    )

    # Build edge-key to index mapping for JS
    edge_keys: List[str] = []

    # Compute fan-out bows for parallel / reciprocal edges
    trav_bows, trav_loop_sizes = _compute_edge_fans(edges)

    # Edges first (painter's algorithm)
    for ei, (src, tgt) in enumerate(edges):
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        r_src = radii.get(src, node_size)
        r_dst = radii.get(tgt, node_size)
        s_src = _sanitize_svg_id(src)
        s_tgt = _sanitize_svg_id(tgt)
        ek = f"{s_src}-{s_tgt}"
        edge_keys.append(f"{src}\t{tgt}")

        bow = trav_bows[ei]

        grp = (
            f'<g id="{uid}-eg-{ek}" opacity="{unvisited_edge_opacity}" '
            f'style="transition: opacity 0.3s ease">'
        )

        if src == tgt:
            lr = r_src * 1.2 * trav_loop_sizes[ei]
            cx_arc = x1
            cy_arc = y1 - r_src - lr
            angle = 0.45
            lsx = x1 + r_src * math.sin(-angle)
            lsy = y1 - r_src * math.cos(-angle)
            lex = x1 + r_src * math.sin(angle)
            ley = y1 - r_src * math.cos(angle)
            cp_h = lr * 1.6
            c1x = lsx - cp_h * 0.3
            c1y = lsy - cp_h
            c2x = cx_arc - lr
            c2y = cy_arc
            c3x = cx_arc + lr
            c3y = cy_arc
            c4x = lex + cp_h * 0.3
            c4y = ley - cp_h
            path_d = (
                f"M {lsx:.1f} {lsy:.1f} "
                f"C {c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {cx_arc:.1f} {cy_arc - lr * 0.2:.1f} "
                f"C {c3x:.1f} {c3y:.1f} {c4x:.1f} {c4y:.1f} {lex:.1f} {ley:.1f}"
            )
            grp += (
                f'<path id="{uid}-el-{ek}" d="{path_d}" '
                f'fill="none" stroke="{th.edge_color}" stroke-width="{edge_width}" '
                f'stroke-linecap="round" '
                f'style="transition: stroke 0.3s ease, stroke-width 0.3s ease, '
                f'stroke-dasharray 0.3s ease"/>'
            )
            if is_directed:
                tan_x = lex - c4x
                tan_y = ley - c4y
                t_len = math.sqrt(tan_x * tan_x + tan_y * tan_y)
                if t_len > 1e-6:
                    tux, tuy = tan_x / t_len, tan_y / t_len
                    arrow_scale = max(edge_width / 1.2, 0.6)
                    head_len = 8.0 * arrow_scale
                    head_wid = 5.0 * arrow_scale
                    abx = lex - tux * head_len
                    aby = ley - tuy * head_len
                    ppx, ppy = -tuy, tux
                    p1x = abx + ppx * head_wid / 2
                    p1y = aby + ppy * head_wid / 2
                    p2x = abx - ppx * head_wid / 2
                    p2y = aby - ppy * head_wid / 2
                    grp += (
                        f'<polygon id="{uid}-ea-{ek}" '
                        f'points="{lex:.1f},{ley:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                        f'fill="{th.edge_color}" '
                        f'style="transition: fill 0.3s ease"/>'
                    )

        elif abs(bow) > 1e-6:
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                grp += "</g>"
                svg_parts.append(grp)
                continue
            cux, cuy = dx / dist, dy / dist
            cpx, cpy = -cuy, cux
            mx = (x1 + x2) / 2 + cpx * dist * bow
            my = (y1 + y2) / 2 + cpy * dist * bow
            cdx_s = mx - x1
            cdy_s = my - y1
            cd_s = math.sqrt(cdx_s * cdx_s + cdy_s * cdy_s)
            csx = x1 + (cdx_s / cd_s) * r_src if cd_s > 1e-6 else x1
            csy = y1 + (cdy_s / cd_s) * r_src if cd_s > 1e-6 else y1
            tan_ex = x2 - mx
            tan_ey = y2 - my
            tan_e_len = math.sqrt(tan_ex * tan_ex + tan_ey * tan_ey)
            cex = x2 - (tan_ex / tan_e_len) * r_dst if tan_e_len > 1e-6 else x2
            cey = y2 - (tan_ey / tan_e_len) * r_dst if tan_e_len > 1e-6 else y2
            grp += (
                f'<path id="{uid}-el-{ek}" '
                f'd="M {csx:.1f} {csy:.1f} Q {mx:.1f} {my:.1f} {cex:.1f} {cey:.1f}" '
                f'fill="none" stroke="{th.edge_color}" stroke-width="{edge_width}" '
                f'stroke-linecap="round" '
                f'style="transition: stroke 0.3s ease, stroke-width 0.3s ease, '
                f'stroke-dasharray 0.3s ease"/>'
            )
            if is_directed:
                atx = 2 * (cex - mx)
                aty = 2 * (cey - my)
                at_len = math.sqrt(atx * atx + aty * aty)
                if at_len > 1e-6:
                    atux, atuy = atx / at_len, aty / at_len
                    arrow_scale = max(edge_width / 1.2, 0.6)
                    head_len = 8.0 * arrow_scale
                    head_wid = 5.0 * arrow_scale
                    abx = cex - atux * head_len
                    aby = cey - atuy * head_len
                    ppx, ppy = -atuy, atux
                    p1x = abx + ppx * head_wid / 2
                    p1y = aby + ppy * head_wid / 2
                    p2x = abx - ppx * head_wid / 2
                    p2y = aby - ppy * head_wid / 2
                    grp += (
                        f'<polygon id="{uid}-ea-{ek}" '
                        f'points="{cex:.1f},{cey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                        f'fill="{th.edge_color}" '
                        f'style="transition: fill 0.3s ease"/>'
                    )

        else:
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                grp += "</g>"
                svg_parts.append(grp)
                continue
            ux, uy = dx / dist, dy / dist
            sx = x1 + ux * r_src
            sy = y1 + uy * r_src
            ex = x2 - ux * r_dst
            ey = y2 - uy * r_dst
            arrow_scale = max(edge_width / 1.2, 0.6)
            head_len = 8.0 * arrow_scale
            head_wid = 5.0 * arrow_scale
            ex_line, ey_line = ex, ey
            if is_directed:
                ex_line = ex - ux * head_len
                ey_line = ey - uy * head_len
            grp += (
                f'<line id="{uid}-el-{ek}" '
                f'x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex_line:.1f}" y2="{ey_line:.1f}" '
                f'stroke="{th.edge_color}" stroke-width="{edge_width}" '
                f'stroke-linecap="round" '
                f'style="transition: stroke 0.3s ease, stroke-width 0.3s ease, '
                f'stroke-dasharray 0.3s ease"/>'
            )
            if is_directed:
                bx = ex - ux * head_len
                by = ey - uy * head_len
                px, py = -uy, ux
                p1x, p1y = bx + px * head_wid / 2, by + py * head_wid / 2
                p2x, p2y = bx - px * head_wid / 2, by - py * head_wid / 2
                grp += (
                    f'<polygon id="{uid}-ea-{ek}" '
                    f'points="{ex:.1f},{ey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                    f'fill="{th.edge_color}" '
                    f'style="transition: fill 0.3s ease"/>'
                )

        grp += "</g>"
        svg_parts.append(grp)

    # Node groups
    node_safe_ids: Dict[str, str] = {}
    for nid in str_ids:
        if nid not in positions:
            continue
        x, y = positions[nid]
        r = radii.get(nid, node_size)
        fill = color_map.get(nid, DEFAULT_NODE_COLOR)
        s_nid = _sanitize_svg_id(nid)
        node_safe_ids[nid] = s_nid
        border = _darken(fill, 0.3)

        svg_parts.append(f'<g id="{uid}-ng-{s_nid}">')

        # Glow circle (for current state)
        svg_parts.append(
            f'<circle id="{uid}-gw-{s_nid}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{r + 8:.1f}" fill="none" stroke="{accent}" stroke-width="4" '
            f'opacity="0" filter="url(#blur-glow)" '
            f'style="transition: opacity 0.3s ease"/>'
        )

        # Ring circle (for frontier state)
        svg_parts.append(
            f'<circle id="{uid}-rg-{s_nid}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{r + 4:.1f}" fill="none" stroke="{accent}" stroke-width="2" '
            f'stroke-dasharray="4 3" opacity="0" '
            f'style="transition: opacity 0.3s ease"/>'
        )

        # Main fill circle
        svg_parts.append(
            f'<circle id="{uid}-fl-{s_nid}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{r:.1f}" fill="{unvisited_fill}" stroke="{border}" '
            f'stroke-width="1.5" stroke-opacity="0.3" opacity="0.4" '
            f'style="transition: fill 0.3s ease, opacity 0.3s ease;"/>'
        )

        # Step number text (inside node)
        step_font = max(8, r * 0.7)
        svg_parts.append(
            f'<text id="{uid}-sn-{s_nid}" x="{x:.1f}" y="{y + step_font * 0.35:.1f}" '
            f'text-anchor="middle" font-family="{FONT_FAMILY}" '
            f'font-size="{step_font:.1f}" font-weight="bold" '
            f'fill="white" opacity="0" pointer-events="none" '
            f'style="transition: opacity 0.2s ease"></text>'
        )

        svg_parts.append("</g>")

    # Labels (outside node groups, on top)
    if show_labels:
        font_size = max(8, min(13, node_size * 0.7))
        for nid in str_ids:
            if nid not in positions:
                continue
            x, y = positions[nid]
            r = radii.get(nid, node_size)
            s_nid = _sanitize_svg_id(nid)
            svg_parts.append(
                f'<text id="{uid}-lb-{s_nid}" x="{x:.1f}" '
                f'y="{y + r + font_size + 2:.1f}" '
                f'text-anchor="middle" font-family="{FONT_FAMILY}" '
                f'font-size="{font_size:.1f}" fill="{th.label_color}" '
                f'opacity="0.3" pointer-events="none" '
                f'style="transition: opacity 0.3s ease">'
                f"{html_mod.escape(label_map[nid])}</text>"
            )

    svg_parts.append("</svg>")
    svg_str = "\n".join(svg_parts)

    # --- Build interactive HTML wrapper ---
    html_str = _generate_traversal_html_wrapper(
        svg_str,
        th,
        uid,
        steps,
        visit_order_str,
        str_ids,
        edges,
        edge_keys,
        node_safe_ids,
        color_map,
        algorithm,
        colors,
        is_directed,
        unvisited_fill,
        frontier_fill,
        current_fill,
        accent,
        tree_edge_color,
        current_edge_color,
        nontree_visited_edge,
        unvisited_edge_opacity,
        edge_width,
        width,
    )

    return colors.finish(svg_str, html_str, output)


def _panel_style(th: Theme, colors: _Colors) -> str:
    """CSS for a player's outer box: a card in the theme's colors, like `render_steps()`'s."""
    border = colors.pick("#d0d7de", "#30363d")
    return (
        f"background:{th.bg};color:{th.label_color};color-scheme:{colors.scheme()};"
        f"border:1px solid {border};border-radius:8px;overflow:hidden"
    )


def _generate_traversal_html_wrapper(
    svg_str: str,
    th: Theme,
    uid: str,
    steps: List[Dict[str, Any]],
    visit_order: List[str],
    str_ids: List[str],
    edges: List[Tuple[str, str]],
    edge_keys: List[str],
    node_safe_ids: Dict[str, str],
    color_map: Dict[str, str],
    algorithm: str,
    colors: _Colors,
    is_directed: bool,
    unvisited_fill: str,
    frontier_fill: str,
    current_fill: str,
    accent: str,
    tree_edge_color: str,
    current_edge_color: str,
    nontree_visited_edge: str,
    unvisited_edge_opacity: float,
    edge_width: float,
    svg_width: int,
) -> str:
    """Generate the full HTML with controls and JS for traversal."""
    n_steps = len(visit_order)
    algo_label = f"{algorithm} traversal — " if algorithm else "Traversal — "
    algo_label += f"{n_steps} nodes"

    # Serialize step data to JSON
    steps_json = json.dumps(steps)

    # Build node color map for JS (original colors for visited state)
    node_colors_json = json.dumps(color_map)

    # Build safe-ID mapping for JS
    safe_ids_json = json.dumps(node_safe_ids)

    # Build edge key list for JS: tab-separated "src\ttgt"
    edge_keys_json = json.dumps(edge_keys)

    # Sanitized edge key map for getElementById
    edge_safe_keys: Dict[str, str] = {}
    for ek in edge_keys:
        src, tgt = ek.split("\t")
        edge_safe_keys[ek] = f"{_sanitize_svg_id(src)}-{_sanitize_svg_id(tgt)}"
    edge_safe_json = json.dumps(edge_safe_keys)

    btn_bg = colors.pick("#e5e7eb", "#374151")
    btn_fg = colors.pick("#374151", "#e5e7eb")
    ctrl_accent = accent

    # The controls sit on the theme's background, not the page's, which can be dark under a light
    # theme (or vice versa) and leave the text unreadable. `color-scheme` does the same for the
    # browser-drawn parts (the slider's track and the speed menu)
    return f"""<div class="ec-plot" style="width:fit-content;max-width:min({svg_width}px,100%);margin:1em auto;font-family:{FONT_FAMILY};
  {_panel_style(th, colors)}">
<style>.ec-plot > svg {{ display:block; max-width:100%; height:auto; }}</style>
{svg_str}
<div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
  font-family:{FONT_FAMILY};font-size:13px;color:{th.label_color};">
  <button id="{uid}-play" title="Play / Pause" style="
    width:32px;height:32px;border:none;border-radius:6px;cursor:pointer;
    background:{btn_bg};color:{btn_fg};font-size:14px;
    display:flex;align-items:center;justify-content:center;
  ">&#9654;</button>
  <input id="{uid}-slider" type="range" min="0" max="{n_steps}"
    value="0" style="flex:1;accent-color:{ctrl_accent};"/>
  <span id="{uid}-counter" style="min-width:90px;text-align:center;
    font-variant-numeric:tabular-nums;">Step 0 / {n_steps}</span>
  <select id="{uid}-speed" style="
    border:1px solid {btn_bg};border-radius:4px;padding:2px 4px;
    background:transparent;color:{th.label_color};font-size:12px;">
    <option value="1200">Slow</option>
    <option value="600" selected>Normal</option>
    <option value="250">Fast</option>
    <option value="100">Very fast</option>
  </select>
</div>
<div style="font-size:11px;color:{th.label_color};opacity:0.7;padding:0 10px 8px;">
  {algo_label}
</div>
<script>
(function(){{
  var U='{uid}';
  var STEPS={steps_json};
  var NCOLORS={node_colors_json};
  var SIDS={safe_ids_json};
  var EKS={edge_keys_json};
  var ESAFE={edge_safe_json};
  var svg=document.getElementById(U+'-svg');
  var slider=document.getElementById(U+'-slider');
  var counter=document.getElementById(U+'-counter');
  var playBtn=document.getElementById(U+'-play');
  var speedSel=document.getElementById(U+'-speed');
  if(!svg||!slider) return;
  var playing=false,timer=null,curStep=0;
  var N={n_steps};
  var EW={edge_width};
  var DIR={"true" if is_directed else "false"};
  var UF='{unvisited_fill}',FF='{frontier_fill}',CF='{current_fill}';
  var ACC='{accent}',TEC='{tree_edge_color}',CEC='{current_edge_color}';
  var NVE='{nontree_visited_edge}',EC='{th.edge_color}';
  var UEO='{unvisited_edge_opacity}';

  function el(id){{ return document.getElementById(id); }}

  function applyStep(k){{
    var s=STEPS[k];
    var visited=new Set(s.visited||[]);
    var frontier=new Set(s.frontier||[]);
    var cur=s.current;
    var labels=s.step_labels||{{}};

    var teSet=new Set();
    (s.tree_edges||[]).forEach(function(e){{
      teSet.add(e[0]+'\\t'+e[1]);
      teSet.add(e[1]+'\\t'+e[0]);
    }});
    var ceKey=s.current_edge?s.current_edge[0]+'\\t'+s.current_edge[1]:null;
    var ceKeyR=s.current_edge?s.current_edge[1]+'\\t'+s.current_edge[0]:null;

    for(var nid in SIDS){{
      var sid=SIDS[nid];
      var fl=el(U+'-fl-'+sid);
      var gw=el(U+'-gw-'+sid);
      var rg=el(U+'-rg-'+sid);
      var sn=el(U+'-sn-'+sid);
      var lb=el(U+'-lb-'+sid);
      if(!fl) continue;

      if(nid===cur){{
        fl.setAttribute('fill',CF);
        fl.setAttribute('opacity','1');
        fl.setAttribute('stroke-opacity','0.5');
        if(gw)gw.setAttribute('opacity','0.5');
        if(rg)rg.setAttribute('opacity','0');
        if(sn){{ sn.textContent=labels[nid]||''; sn.setAttribute('opacity','1'); }}
        if(lb)lb.setAttribute('opacity','0.9');
      }} else if(visited.has(nid)){{
        fl.setAttribute('fill',NCOLORS[nid]||'{DEFAULT_NODE_COLOR}');
        fl.setAttribute('opacity','1');
        fl.setAttribute('stroke-opacity','0.3');
        if(gw)gw.setAttribute('opacity','0');
        if(rg)rg.setAttribute('opacity','0');
        if(sn){{ sn.textContent=labels[nid]||''; sn.setAttribute('opacity','0.7'); }}
        if(lb)lb.setAttribute('opacity','0.8');
      }} else if(frontier.has(nid)){{
        fl.setAttribute('fill',FF);
        fl.setAttribute('opacity','0.7');
        fl.setAttribute('stroke-opacity','0.2');
        if(gw)gw.setAttribute('opacity','0');
        if(rg)rg.setAttribute('opacity','0.7');
        if(sn)sn.setAttribute('opacity','0');
        if(lb)lb.setAttribute('opacity','0.6');
      }} else {{
        fl.setAttribute('fill',UF);
        fl.setAttribute('opacity','0.4');
        fl.setAttribute('stroke-opacity','0.15');
        if(gw)gw.setAttribute('opacity','0');
        if(rg)rg.setAttribute('opacity','0');
        if(sn)sn.setAttribute('opacity','0');
        if(lb)lb.setAttribute('opacity','0.3');
      }}
    }}

    EKS.forEach(function(ek){{
      var sk=ESAFE[ek];
      var grp=el(U+'-eg-'+sk);
      var line=el(U+'-el-'+sk);
      var arrow=DIR?el(U+'-ea-'+sk):null;
      if(!line) return;
      var parts=ek.split('\\t');
      var isCur=(ek===ceKey||ek===ceKeyR);
      var isTree=teSet.has(ek);
      var bothVisited=visited.has(parts[0])&&visited.has(parts[1]);

      if(isCur){{
        if(grp)grp.setAttribute('opacity','1');
        line.setAttribute('stroke',CEC);
        line.setAttribute('stroke-width',''+(EW*3.5));
        line.setAttribute('stroke-dasharray','none');
        if(arrow)arrow.setAttribute('fill',CEC);
      }} else if(isTree){{
        if(grp)grp.setAttribute('opacity','0.7');
        line.setAttribute('stroke',TEC);
        line.setAttribute('stroke-width',''+(EW*2.5));
        line.setAttribute('stroke-dasharray','none');
        if(arrow)arrow.setAttribute('fill',TEC);
      }} else if(bothVisited){{
        if(grp)grp.setAttribute('opacity','0.25');
        line.setAttribute('stroke',NVE);
        line.setAttribute('stroke-width',''+EW);
        line.setAttribute('stroke-dasharray','4 2');
        if(arrow)arrow.setAttribute('fill',NVE);
      }} else {{
        if(grp)grp.setAttribute('opacity',''+UEO);
        line.setAttribute('stroke',EC);
        line.setAttribute('stroke-width',''+Math.max(0.5,EW*0.4));
        line.setAttribute('stroke-dasharray','none');
        if(arrow)arrow.setAttribute('fill',EC);
      }}
    }});

    counter.textContent='Step '+k+' / '+N;
  }}

  function advance(){{
    if(curStep>=N){{ clearInterval(timer); playing=false; playBtn.innerHTML='&#9654;'; return; }}
    curStep++;
    slider.value=curStep;
    applyStep(curStep);
  }}

  slider.addEventListener('input',function(e){{
    curStep=parseInt(e.target.value);
    applyStep(curStep);
  }});

  playBtn.addEventListener('click',function(){{
    if(playing){{ clearInterval(timer); playing=false; playBtn.innerHTML='&#9654;'; }}
    else {{
      if(curStep>=N){{ curStep=0; slider.value=0; applyStep(0); }}
      var d=parseInt(speedSel.value);
      timer=setInterval(advance,d);
      playing=true;
      playBtn.innerHTML='&#9646;&#9646;';
    }}
  }});

  speedSel.addEventListener('change',function(){{
    if(playing){{ clearInterval(timer); timer=setInterval(advance,parseInt(speedSel.value)); }}
  }});

  applyStep(0);
}})();
</script>
</div>"""


# ---------------------------------------------------------------------------
# Traversal-chain counter (unique IDs across multiple renders)
# ---------------------------------------------------------------------------

_chain_counter = 0


# ---------------------------------------------------------------------------
# TraversalRecorder — captures selection state at each step
# ---------------------------------------------------------------------------


class TraversalRecorder:
    """Records a sequence of selection/traversal operations on a graph.

    Each call to a `trav_*()` or `select_*()` method is delegated to the underlying graph, then the
    resulting selection (node IDs or edge IDs) is captured as a step. The recorded chain can be
    rendered as an interactive visualization with `plot()`.
    """

    _EDGE_OPS = frozenset({"trav_out_edge", "trav_in_edge", "trav_reverse_edge"})
    _NODE_OPS = frozenset(
        {
            "trav_out",
            "trav_in",
            "trav_both",
            "trav_out_until",
            "trav_in_until",
            "select_nodes",
            "select_nodes_by_id",
            "select_nodes_by_degree",
            "select_nodes_by_centrality",
            "select_nodes_in_neighborhood",
            "select_last_nodes_created",
            "invert_node_selection",
        }
    )
    _EDGE_SELECT_OPS = frozenset(
        {
            "select_edges",
            "select_edges_by_id",
            "select_edges_by_node",
            "select_last_edges_created",
            "invert_edge_selection",
        }
    )

    def __init__(self, graph) -> None:
        self._graph = graph
        self._steps: List[Dict[str, Any]] = []

    def _snapshot(self, op_name: str, op_args_str: str) -> None:
        if op_name in self._EDGE_OPS or op_name in self._EDGE_SELECT_OPS:
            sel_type = "edges"
            ids = list(self._graph.get_selected_edge_ids())
        else:
            sel_type = "nodes"
            ids = list(self._graph.get_selected_node_ids())
        self._steps.append(
            {
                "op": op_name,
                "op_display": f"{op_name}({op_args_str})" if op_args_str else f"{op_name}()",
                "selection_type": sel_type,
                "ids": ids,
            }
        )

    def _delegate(self, method_name: str, args_str: str, *args, **kwargs):
        getattr(self._graph, method_name)(*args, **kwargs)
        self._snapshot(method_name, args_str)
        return self

    # --- Node traversals ---
    def trav_out(self):
        return self._delegate("trav_out", "")

    def trav_in(self):
        return self._delegate("trav_in", "")

    def trav_both(self):
        return self._delegate("trav_both", "")

    def trav_out_until(
        self, conditions, max_steps=30, exclude_unmatched=True, add_to_selection=False
    ):
        return self._delegate(
            "trav_out_until",
            "...",
            conditions,
            max_steps=max_steps,
            exclude_unmatched=exclude_unmatched,
            add_to_selection=add_to_selection,
        )

    def trav_in_until(
        self, conditions, max_steps=30, exclude_unmatched=True, add_to_selection=False
    ):
        return self._delegate(
            "trav_in_until",
            "...",
            conditions,
            max_steps=max_steps,
            exclude_unmatched=exclude_unmatched,
            add_to_selection=add_to_selection,
        )

    # --- Edge traversals ---
    def trav_out_edge(self):
        return self._delegate("trav_out_edge", "")

    def trav_in_edge(self):
        return self._delegate("trav_in_edge", "")

    def trav_reverse_edge(self, add_to_selection=False):
        args = "add_to_selection=True" if add_to_selection else ""
        return self._delegate("trav_reverse_edge", args, add_to_selection=add_to_selection)

    # --- Node selections ---
    def select_nodes(self, predicate):
        return self._delegate("select_nodes", "...", predicate)

    def select_nodes_by_id(self, node_ids):
        label = ", ".join(str(i) for i in node_ids[:5])
        if len(node_ids) > 5:
            label += ", ..."
        return self._delegate("select_nodes_by_id", label, node_ids)

    def select_nodes_by_degree(self, **kwargs):
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        return self._delegate("select_nodes_by_degree", ", ".join(parts), **kwargs)

    def select_nodes_by_centrality(self, **kwargs):
        parts = [f"{k}={v!r}" for k, v in kwargs.items()]
        return self._delegate("select_nodes_by_centrality", ", ".join(parts), **kwargs)

    def select_nodes_in_neighborhood(self, nodes, k=1, mode="out"):
        return self._delegate("select_nodes_in_neighborhood", f"..., k={k}", nodes, k=k, mode=mode)

    # --- Edge selections ---
    def select_edges(self, predicate):
        return self._delegate("select_edges", "...", predicate)

    def select_edges_by_id(self, edge_ids):
        label = ", ".join(str(i) for i in edge_ids[:5])
        if len(edge_ids) > 5:
            label += ", ..."
        return self._delegate("select_edges_by_id", label, edge_ids)

    def select_edges_by_node(self, nodes):
        return self._delegate("select_edges_by_node", "...", nodes)

    # --- Deselection ---
    def deselect_nodes(self, node_ids):
        return self._delegate("deselect_nodes", "...", node_ids)

    def deselect_edges(self, edge_ids):
        return self._delegate("deselect_edges", "...", edge_ids)

    def clear_selection(self):
        return self._delegate("clear_selection", "")

    def invert_node_selection(self):
        return self._delegate("invert_node_selection", "")

    def invert_edge_selection(self):
        return self._delegate("invert_edge_selection", "")

    @property
    def steps(self) -> List[Dict[str, Any]]:
        return list(self._steps)

    def plot(
        self,
        layout: str = "force",
        theme: Optional[str] = None,
        width: int = 700,
        height: int = 500,
        node_size: float = 18.0,
        edge_width: float = 1.5,
        node_overlap: Union[str, float] = "remove",
        show_labels: bool = True,
        output: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> StepPlayer:
        return render_traversal_chain(
            self._graph,
            self._steps,
            layout=layout,
            theme=theme,
            width=width,
            height=height,
            node_size=node_size,
            edge_width=edge_width,
            node_overlap=node_overlap,
            show_labels=show_labels,
            output=output,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# render_traversal_chain — interactive selection-chain visualization
# ---------------------------------------------------------------------------


def render_traversal_chain(
    graph,
    chain_steps: List[Dict[str, Any]],
    layout: str = "force",
    theme: Optional[str] = None,
    width: int = 700,
    height: int = 500,
    node_size: float = 18.0,
    edge_width: float = 1.5,
    node_overlap: Union[str, float] = "remove",
    show_labels: bool = True,
    output: Optional[str] = None,
    seed: Optional[int] = None,
) -> StepPlayer:
    """Render an interactive traversal-chain visualization.

    Each step in *chain_steps* is a dict with keys `op`, `op_display`, `selection_type` (`"nodes"`
    or `"edges"`), and `ids` (list of integer IDs).  The visualization shows the full graph and lets
    the user step through the chain, highlighting the selected nodes or edges at each step.
    """
    global _chain_counter
    _chain_counter += 1
    uid = f"tc{_chain_counter}"

    _check_style(layout, theme, stored_ok=True)
    colors = _Colors(theme)
    th = colors.theme

    # --- Theme colors ---
    unvisited_fill = colors.pick("#d1d5db", "#4b5563")
    accent = colors.pick("#2563eb", "#60a5fa")
    accent_light = colors.pick("#bfdbfe", "#93c5fd")
    prev_accent = colors.pick("#a5b4fc", "#818cf8")
    selected_edge_color = colors.pick("#2563eb", "#60a5fa")
    related_edge_color = colors.pick("#93c5fd", "#60a5fa")
    unvisited_edge_opacity = colors.pick(0.12, 0.25)

    # --- Extract graph data ---
    node_ids = list(graph.get_node_ids(limit=None))
    str_ids = [str(nid) for nid in node_ids]
    n = len(node_ids)

    if edge_width == 1.5:
        edge_width = max(0.5, 3.0 - n * 0.06) if n < 40 else max(0.5, 1.5 - (n - 40) * 0.002)
    edge_width = max(edge_width, 0.5)

    if node_size == 18.0:
        avg_spacing = math.sqrt(float(width * height) / max(n, 1))
        node_size = max(5.0, min(18.0, avg_spacing * 0.18))

    node_data: Dict[str, Dict] = {}
    for nid in node_ids:
        obj = graph.get_node(nid)
        attrs = dict(obj.attributes) if obj and obj.attributes else {}
        node_data[str(nid)] = attrs

    is_directed = graph.is_directed()
    edges: List[Tuple[str, str]] = []
    seen_edges: set = set()
    adjacency: Dict[str, List[str]] = {nid: [] for nid in str_ids}
    for nid in node_ids:
        try:
            neighbors = graph.get_neighbors(nid)
        except Exception:
            continue
        for nb in neighbors:
            if is_directed:
                key = (str(nid), str(nb))
            else:
                key = tuple(sorted([str(nid), str(nb)]))
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append((str(nid), str(nb)))
            adjacency.setdefault(str(nid), []).append(str(nb))
            if not is_directed:
                adjacency.setdefault(str(nb), []).append(str(nid))
    for k in adjacency:
        adjacency[k] = list(dict.fromkeys(adjacency[k]))

    # --- Build edge-ID → visual-edge mapping ---
    # For each graph edge, find the edge_ids that correspond to it
    edge_id_to_visual: Dict[int, Tuple[str, str]] = {}
    visual_to_edge_ids: Dict[Tuple[str, str], List[int]] = {}
    for src_s, tgt_s in edges:
        try:
            eids = graph.find_edge_ids(
                int(src_s) if src_s.isdigit() else src_s, int(tgt_s) if tgt_s.isdigit() else tgt_s
            )
        except Exception:
            eids = []
        visual_to_edge_ids[(src_s, tgt_s)] = list(eids)
        for eid in eids:
            edge_id_to_visual[eid] = (src_s, tgt_s)

    # Also try reverse lookup for directed graphs where edge IDs may be stored
    # under the actual source/target from get_edge()
    all_edge_ids_in_steps = set()
    for step in chain_steps:
        if step["selection_type"] == "edges":
            all_edge_ids_in_steps.update(step["ids"])
    for eid in all_edge_ids_in_steps:
        if eid not in edge_id_to_visual:
            try:
                e_obj = graph.get_edge(eid)
                if e_obj:
                    s, t = str(e_obj.source), str(e_obj.target)
                    edge_id_to_visual[eid] = (s, t)
                    visual_to_edge_ids.setdefault((s, t), []).append(eid)
            except Exception:
                pass

    # --- Precompute step data for JS ---
    # Convert each step's IDs to the string-based format used in rendering
    js_steps = []
    for i, step in enumerate(chain_steps):
        sel_type = step["selection_type"]
        if sel_type == "nodes":
            sel_node_strs = [str(sid) for sid in step["ids"]]
            # Incident edges get subtle highlight
            related_edge_keys = []
            for ns in sel_node_strs:
                for src_s, tgt_s in edges:
                    if src_s == ns or tgt_s == ns:
                        s_src = _sanitize_svg_id(src_s)
                        s_tgt = _sanitize_svg_id(tgt_s)
                        related_edge_keys.append(f"{s_src}-{s_tgt}")
            js_steps.append(
                {
                    "sel": "nodes",
                    "nodes": sel_node_strs,
                    "edges": [],
                    "related_edges": list(dict.fromkeys(related_edge_keys)),
                    "op": step["op_display"],
                }
            )
        else:
            sel_edge_visuals = []
            connected_nodes = []
            for eid in step["ids"]:
                vis = edge_id_to_visual.get(eid)
                if vis:
                    s_src = _sanitize_svg_id(vis[0])
                    s_tgt = _sanitize_svg_id(vis[1])
                    sel_edge_visuals.append(f"{s_src}-{s_tgt}")
                    connected_nodes.extend(vis)
            js_steps.append(
                {
                    "sel": "edges",
                    "nodes": [],
                    "edges": list(dict.fromkeys(sel_edge_visuals)),
                    "related_nodes": list(dict.fromkeys(connected_nodes)),
                    "op": step["op_display"],
                }
            )

    # Previous-step data for "prev" indicators
    for i in range(len(js_steps)):
        if i > 0:
            prev = js_steps[i - 1]
            if prev["sel"] == "nodes":
                js_steps[i]["prev_nodes"] = prev["nodes"]
            else:
                js_steps[i]["prev_edges"] = prev.get("edges", [])
        else:
            js_steps[i]["prev_nodes"] = []

    # --- Layout ---
    if layout == "circular":
        positions = _circular_layout(str_ids, float(width), float(height), edges=edges)
    elif layout == "random":
        positions = _random_layout(str_ids, float(width), float(height), seed=seed)
    elif layout not in LAYOUTS:
        positions = _stored_layout(graph, node_ids, layout, float(width), float(height))
    else:
        fr_iters = 800 if n > 300 else (600 if n > 150 else 400)
        positions = _force_directed_layout(
            str_ids,
            edges,
            float(width),
            float(height),
            iterations=fr_iters,
            seed=seed,
        )

    radii = {nid: node_size for nid in str_ids}

    # --- Overlap removal ---
    if node_overlap != "allow" and layout not in ("circular",) and n >= 2:
        overlap_pad = float(node_overlap) if isinstance(node_overlap, (int, float)) else 2.0
        positions = _remove_overlaps(positions, radii, padding=overlap_pad, seed=seed)

    # --- ViewBox ---
    label_map = {}
    if show_labels:
        for nid in str_ids:
            lbl = node_data.get(nid, {}).get("label", nid)
            label_map[nid] = str(lbl) if lbl else nid
    vb = _compute_viewbox(
        positions,
        radii,
        labels=label_map if show_labels else None,
        label_space=20 if show_labels else 0,
        frame=(width, height),
    )
    vb_str = f"{vb[0]:.1f} {vb[1]:.1f} {vb[2]:.1f} {vb[3]:.1f}"

    # --- Build SVG ---
    svg_parts: List[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="{vb_str}" '
        f'id="{uid}-svg" '
        f'style="font-family: {FONT_FAMILY};">'
    )
    svg_parts.append(
        f'<rect x="{vb[0]:.1f}" y="{vb[1]:.1f}" '
        f'width="{vb[2]:.1f}" height="{vb[3]:.1f}" fill="{th.bg}"/>'
    )

    # Defs — glow filter
    svg_parts.append(
        f"<defs>"
        f'<filter id="{uid}-glow"><feGaussianBlur stdDeviation="3" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f"</filter></defs>"
    )

    # --- Edges ---
    # Compute fan-out bows for parallel / reciprocal edges
    chain_bows, chain_loop_sizes = _compute_edge_fans(edges)

    edge_keys: List[str] = []
    for ei, (src, tgt) in enumerate(edges):
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        r_src = radii.get(src, node_size)
        r_dst = radii.get(tgt, node_size)
        s_src = _sanitize_svg_id(src)
        s_tgt = _sanitize_svg_id(tgt)
        ek = f"{s_src}-{s_tgt}"
        edge_keys.append(ek)

        bow = chain_bows[ei]

        grp = (
            f'<g id="{uid}-eg-{ek}" opacity="{unvisited_edge_opacity}" '
            f'style="transition: opacity 0.3s ease, filter 0.3s ease">'
        )

        if src == tgt:
            lr = r_src * 1.2 * chain_loop_sizes[ei]
            cx_arc = x1
            cy_arc = y1 - r_src - lr
            angle = 0.45
            lsx = x1 + r_src * math.sin(-angle)
            lsy = y1 - r_src * math.cos(-angle)
            lex = x1 + r_src * math.sin(angle)
            ley = y1 - r_src * math.cos(angle)
            cp_h = lr * 1.6
            c1x = lsx - cp_h * 0.3
            c1y = lsy - cp_h
            c2x = cx_arc - lr
            c2y = cy_arc
            c3x = cx_arc + lr
            c3y = cy_arc
            c4x = lex + cp_h * 0.3
            c4y = ley - cp_h
            path_d = (
                f"M {lsx:.1f} {lsy:.1f} "
                f"C {c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {cx_arc:.1f} {cy_arc - lr * 0.2:.1f} "
                f"C {c3x:.1f} {c3y:.1f} {c4x:.1f} {c4y:.1f} {lex:.1f} {ley:.1f}"
            )
            grp += (
                f'<path id="{uid}-el-{ek}" d="{path_d}" '
                f'fill="none" stroke="{th.edge_color}" stroke-width="{edge_width}" '
                f'stroke-linecap="round" '
                f'style="transition: stroke 0.3s ease, stroke-width 0.3s ease"/>'
            )
            if is_directed:
                tan_x = lex - c4x
                tan_y = ley - c4y
                t_len = math.sqrt(tan_x * tan_x + tan_y * tan_y)
                if t_len > 1e-6:
                    tux, tuy = tan_x / t_len, tan_y / t_len
                    arrow_scale = max(edge_width / 1.2, 0.6)
                    head_len = 8.0 * arrow_scale
                    head_wid = 5.0 * arrow_scale
                    abx = lex - tux * head_len
                    aby = ley - tuy * head_len
                    ppx, ppy = -tuy, tux
                    p1x = abx + ppx * head_wid / 2
                    p1y = aby + ppy * head_wid / 2
                    p2x = abx - ppx * head_wid / 2
                    p2y = aby - ppy * head_wid / 2
                    grp += (
                        f'<polygon id="{uid}-ea-{ek}" '
                        f'points="{lex:.1f},{ley:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                        f'fill="{th.edge_color}" '
                        f'style="transition: fill 0.3s ease"/>'
                    )

        elif abs(bow) > 1e-6:
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                grp += "</g>"
                svg_parts.append(grp)
                continue
            cux, cuy = dx / dist, dy / dist
            cpx, cpy = -cuy, cux
            mx = (x1 + x2) / 2 + cpx * dist * bow
            my = (y1 + y2) / 2 + cpy * dist * bow
            cdx_s = mx - x1
            cdy_s = my - y1
            cd_s = math.sqrt(cdx_s * cdx_s + cdy_s * cdy_s)
            csx = x1 + (cdx_s / cd_s) * r_src if cd_s > 1e-6 else x1
            csy = y1 + (cdy_s / cd_s) * r_src if cd_s > 1e-6 else y1
            tan_ex = x2 - mx
            tan_ey = y2 - my
            tan_e_len = math.sqrt(tan_ex * tan_ex + tan_ey * tan_ey)
            cex = x2 - (tan_ex / tan_e_len) * r_dst if tan_e_len > 1e-6 else x2
            cey = y2 - (tan_ey / tan_e_len) * r_dst if tan_e_len > 1e-6 else y2
            grp += (
                f'<path id="{uid}-el-{ek}" '
                f'd="M {csx:.1f} {csy:.1f} Q {mx:.1f} {my:.1f} {cex:.1f} {cey:.1f}" '
                f'fill="none" stroke="{th.edge_color}" stroke-width="{edge_width}" '
                f'stroke-linecap="round" '
                f'style="transition: stroke 0.3s ease, stroke-width 0.3s ease"/>'
            )
            if is_directed:
                atx = 2 * (cex - mx)
                aty = 2 * (cey - my)
                at_len = math.sqrt(atx * atx + aty * aty)
                if at_len > 1e-6:
                    atux, atuy = atx / at_len, aty / at_len
                    arrow_scale = max(edge_width / 1.2, 0.6)
                    head_len = 8.0 * arrow_scale
                    head_wid = 5.0 * arrow_scale
                    abx = cex - atux * head_len
                    aby = cey - atuy * head_len
                    ppx, ppy = -atuy, atux
                    p1x = abx + ppx * head_wid / 2
                    p1y = aby + ppy * head_wid / 2
                    p2x = abx - ppx * head_wid / 2
                    p2y = aby - ppy * head_wid / 2
                    grp += (
                        f'<polygon id="{uid}-ea-{ek}" '
                        f'points="{cex:.1f},{cey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                        f'fill="{th.edge_color}" '
                        f'style="transition: fill 0.3s ease"/>'
                    )

        else:
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                grp += "</g>"
                svg_parts.append(grp)
                continue
            ux, uy = dx / dist, dy / dist
            sx = x1 + ux * r_src
            sy = y1 + uy * r_src
            ex = x2 - ux * r_dst
            ey = y2 - uy * r_dst
            arrow_scale = max(edge_width / 1.2, 0.6)
            head_len = 8.0 * arrow_scale
            head_wid = 5.0 * arrow_scale
            ex_line, ey_line = ex, ey
            if is_directed:
                ex_line = ex - ux * head_len
                ey_line = ey - uy * head_len
            grp += (
                f'<line id="{uid}-el-{ek}" '
                f'x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex_line:.1f}" y2="{ey_line:.1f}" '
                f'stroke="{th.edge_color}" stroke-width="{edge_width}" '
                f'stroke-linecap="round" '
                f'style="transition: stroke 0.3s ease, stroke-width 0.3s ease"/>'
            )
            if is_directed:
                bx = ex - ux * head_len
                by = ey - uy * head_len
                px, py = -uy, ux
                p1x = bx + px * head_wid / 2
                p1y = by + py * head_wid / 2
                p2x = bx - px * head_wid / 2
                p2y = by - py * head_wid / 2
                grp += (
                    f'<polygon id="{uid}-ea-{ek}" '
                    f'points="{ex:.1f},{ey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                    f'fill="{th.edge_color}" '
                    f'style="transition: fill 0.3s ease"/>'
                )

        grp += "</g>"
        svg_parts.append(grp)

    # --- Nodes ---
    node_safe_ids: Dict[str, str] = {}
    for nid in str_ids:
        if nid not in positions:
            continue
        x, y = positions[nid]
        r = radii.get(nid, node_size)
        s_nid = _sanitize_svg_id(nid)
        node_safe_ids[nid] = s_nid
        border = _darken(DEFAULT_NODE_COLOR, 0.3)

        svg_parts.append(f'<g id="{uid}-ng-{s_nid}">')

        # Glow (selected state)
        svg_parts.append(
            f'<circle id="{uid}-gw-{s_nid}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{r + 8:.1f}" fill="none" stroke="{accent}" stroke-width="4" '
            f'opacity="0" filter="url(#{uid}-glow)" '
            f'style="transition: opacity 0.3s ease"/>'
        )
        # Ring (previous-step / related state)
        svg_parts.append(
            f'<circle id="{uid}-rg-{s_nid}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{r + 4:.1f}" fill="none" stroke="{prev_accent}" stroke-width="2" '
            f'stroke-dasharray="4 3" opacity="0" '
            f'style="transition: opacity 0.3s ease, stroke 0.3s ease"/>'
        )
        # Main fill
        svg_parts.append(
            f'<circle id="{uid}-fl-{s_nid}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{r:.1f}" fill="{unvisited_fill}" stroke="{border}" '
            f'stroke-width="1.5" stroke-opacity="0.3" opacity="0.4" '
            f'style="transition: fill 0.3s ease, opacity 0.3s ease"/>'
        )
        svg_parts.append("</g>")

    # --- Labels ---
    if show_labels:
        font_size = max(8, min(13, node_size * 0.7))
        for nid in str_ids:
            if nid not in positions:
                continue
            x, y = positions[nid]
            r = radii.get(nid, node_size)
            s_nid = _sanitize_svg_id(nid)
            lbl = label_map.get(nid, nid)
            svg_parts.append(
                f'<text id="{uid}-lb-{s_nid}" x="{x:.1f}" '
                f'y="{y + r + font_size + 2:.1f}" '
                f'text-anchor="middle" font-family="{FONT_FAMILY}" '
                f'font-size="{font_size:.1f}" fill="{th.label_color}" '
                f'opacity="0.3" pointer-events="none" '
                f'style="transition: opacity 0.3s ease">'
                f"{html_mod.escape(str(lbl))}</text>"
            )

    svg_parts.append("</svg>")
    svg_str = "\n".join(svg_parts)

    # --- Build HTML wrapper ---
    html_str = _generate_chain_html_wrapper(
        svg_str=svg_str,
        th=th,
        uid=uid,
        js_steps=js_steps,
        chain_steps=chain_steps,
        str_ids=str_ids,
        node_safe_ids=node_safe_ids,
        edge_keys=edge_keys,
        colors=colors,
        is_directed=is_directed,
        unvisited_fill=unvisited_fill,
        accent=accent,
        accent_light=accent_light,
        prev_accent=prev_accent,
        selected_edge_color=selected_edge_color,
        related_edge_color=related_edge_color,
        unvisited_edge_opacity=unvisited_edge_opacity,
        edge_width=edge_width,
        svg_width=width,
    )

    return colors.finish(svg_str, html_str, output)


def _generate_chain_html_wrapper(
    svg_str: str,
    th: Theme,
    uid: str,
    js_steps: List[Dict[str, Any]],
    chain_steps: List[Dict[str, Any]],
    str_ids: List[str],
    node_safe_ids: Dict[str, str],
    edge_keys: List[str],
    colors: _Colors,
    is_directed: bool,
    unvisited_fill: str,
    accent: str,
    accent_light: str,
    prev_accent: str,
    selected_edge_color: str,
    related_edge_color: str,
    unvisited_edge_opacity: float,
    edge_width: float,
    svg_width: int,
) -> str:
    """Generate the full HTML with controls and JS for traversal chain."""
    n_steps = len(chain_steps)
    steps_json = json.dumps(js_steps)
    safe_ids_json = json.dumps(node_safe_ids)

    btn_bg = colors.pick("#e5e7eb", "#374151")
    btn_fg = colors.pick("#374151", "#e5e7eb")

    # Build step log HTML
    step_log_lines = []
    for i, step in enumerate(chain_steps):
        sel_type = step["selection_type"]
        if sel_type == "nodes":
            badge_bg = colors.pick("#dbeafe", "#1e3a5f")
            badge_fg = colors.pick("#1e40af", "#93c5fd")
            badge_label = "nodes"
        else:
            badge_bg = colors.pick("#fef3c7", "#78350f")
            badge_fg = colors.pick("#92400e", "#fcd34d")
            badge_label = "edges"
        ids_display = ", ".join(str(x) for x in step["ids"][:8])
        if len(step["ids"]) > 8:
            ids_display += f", ... ({len(step['ids'])} total)"
        step_log_lines.append(
            f'<div id="{uid}-log-{i}" style="display:flex;gap:8px;align-items:baseline;'
            f"padding:3px 8px;border-radius:4px;opacity:0.5;"
            f'transition:opacity 0.2s ease,background 0.2s ease">'
            f'<span style="min-width:18px;font-size:12px">{i + 1}.</span>'
            f'<span style="background:{badge_bg};color:{badge_fg};'
            f'padding:1px 6px;border-radius:4px;font-size:11px;white-space:nowrap">'
            f"{badge_label}</span>"
            f'<span style="font-size:12px;font-family:monospace">'
            f"{html_mod.escape(step['op_display'])}</span>"
            f'<span style="color:{th.label_color};opacity:0.5;margin-left:auto;'
            f'font-size:11px;font-family:monospace;white-space:nowrap">'
            f"→ {{{html_mod.escape(ids_display)}}}</span>"
            f"</div>"
        )
    step_log_html = "\n".join(step_log_lines)

    # Step pill bar (SVG overlay approach is complex, use HTML pills below SVG)
    pill_items = []
    for i, step in enumerate(chain_steps):
        short_op = (
            step["op"]
            .replace("select_nodes_by_id", "select")
            .replace("select_edges_by_id", "select")
        )
        if len(short_op) > 16:
            short_op = short_op[:14] + ".."
        pill_items.append(
            f'<span id="{uid}-pill-{i}" style="'
            f"padding:3px 10px;border-radius:12px;font-size:11px;cursor:pointer;"
            f"white-space:nowrap;border:1px solid {btn_bg};color:{th.label_color};"
            f'transition:background 0.2s ease,color 0.2s ease,border-color 0.2s ease"'
            f' data-step="{i}">{html_mod.escape(short_op)}</span>'
        )
    pills_html = "\n".join(pill_items)

    # On the theme's background, as in `_generate_traversal_html_wrapper()`
    return f"""<div class="ec-plot" style="width:fit-content;max-width:min({svg_width}px,100%);margin:1em auto;font-family:{FONT_FAMILY};
  {_panel_style(th, colors)}">
<style>.ec-plot > svg {{ display:block; max-width:100%; height:auto; }}</style>
{svg_str}
<div style="display:flex;gap:6px;padding:8px 10px 2px;flex-wrap:wrap;align-items:center">
{pills_html}
</div>
<div style="display:flex;align-items:center;gap:8px;padding:8px 10px;font-size:13px;color:{th.label_color}">
  <button id="{uid}-play" title="Play / Pause" style="
    width:32px;height:32px;border:none;border-radius:6px;cursor:pointer;
    background:{btn_bg};color:{btn_fg};font-size:14px;
    display:flex;align-items:center;justify-content:center">&#9654;</button>
  <input id="{uid}-slider" type="range" min="0" max="{n_steps - 1}"
    value="0" style="flex:1;accent-color:{accent}"/>
  <span id="{uid}-counter" style="min-width:90px;text-align:center;
    font-variant-numeric:tabular-nums">Step 1 / {n_steps}</span>
  <select id="{uid}-speed" style="
    border:1px solid {btn_bg};border-radius:4px;padding:2px 4px;
    background:transparent;color:{th.label_color};font-size:12px">
    <option value="1500">Slow</option>
    <option value="800" selected>Normal</option>
    <option value="350">Fast</option>
    <option value="150">Very fast</option>
  </select>
</div>
<details style="border-top:1px solid {btn_bg};margin-top:4px">
  <summary style="padding:8px 10px;font-size:12px;color:{th.label_color};cursor:pointer;
    opacity:0.75;list-style:none;user-select:none">
    &#9662; Traversal steps ({n_steps} operations)
  </summary>
  <div id="{uid}-log" style="padding:4px 6px;line-height:1.9">
    {step_log_html}
  </div>
</details>
<script>
(function(){{
  var U='{uid}';
  var STEPS={steps_json};
  var SAFE={safe_ids_json};
  var N={n_steps};
  var curStep=0;
  var playing=false;
  var timer=null;

  var slider=document.getElementById(U+'-slider');
  var counter=document.getElementById(U+'-counter');
  var playBtn=document.getElementById(U+'-play');
  var speedSel=document.getElementById(U+'-speed');

  var COLORS={{
    unvisited:'{unvisited_fill}',
    accent:'{accent}',
    accentLight:'{accent_light}',
    prevAccent:'{prev_accent}',
    selEdge:'{selected_edge_color}',
    relEdge:'{related_edge_color}',
    edgeDefault:'{th.edge_color}',
    unvisitedEdgeOp:'{unvisited_edge_opacity}',
    edgeWidth:{edge_width}
  }};

  function el(id){{ return document.getElementById(id); }}

  function resetAll(){{
    // Reset all nodes to unvisited
    for(var nid in SAFE){{
      var s=SAFE[nid];
      var gw=el(U+'-gw-'+s); if(gw){{ gw.setAttribute('opacity','0'); }}
      var rg=el(U+'-rg-'+s); if(rg){{ rg.setAttribute('opacity','0'); }}
      var fl=el(U+'-fl-'+s); if(fl){{ fl.setAttribute('fill',COLORS.unvisited); fl.setAttribute('opacity','0.4'); }}
      var lb=el(U+'-lb-'+s); if(lb){{ lb.setAttribute('opacity','0.3'); }}
    }}
    // Reset all edges
    var egs=document.querySelectorAll('[id^="'+U+'-eg-"]');
    for(var i=0;i<egs.length;i++){{
      egs[i].setAttribute('opacity',String(COLORS.unvisitedEdgeOp));
      egs[i].removeAttribute('filter');
    }}
    var els=document.querySelectorAll('[id^="'+U+'-el-"]');
    for(var i=0;i<els.length;i++){{
      els[i].setAttribute('stroke',COLORS.edgeDefault);
      els[i].setAttribute('stroke-width',String(COLORS.edgeWidth));
    }}
    var eas=document.querySelectorAll('[id^="'+U+'-ea-"]');
    for(var i=0;i<eas.length;i++){{
      eas[i].setAttribute('fill',COLORS.edgeDefault);
    }}
    // Reset pills
    for(var j=0;j<N;j++){{
      var p=el(U+'-pill-'+j);
      if(p){{ p.style.background='transparent'; p.style.color='{th.label_color}'; p.style.borderColor='{btn_bg}'; }}
    }}
    // Reset log
    for(var j=0;j<N;j++){{
      var lg=el(U+'-log-'+j);
      if(lg){{ lg.style.opacity='0.5'; lg.style.background='transparent'; }}
    }}
  }}

  function applyStep(k){{
    resetAll();
    if(k<0||k>=N) return;
    var step=STEPS[k];

    // Highlight pill
    var pill=el(U+'-pill-'+k);
    if(pill){{ pill.style.background=COLORS.accent; pill.style.color='white'; pill.style.borderColor=COLORS.accent; }}
    // Highlight log line
    var lg=el(U+'-log-'+k);
    if(lg){{ lg.style.opacity='1'; lg.style.background='{colors.tint(accent_light)}'; }}

    if(step.sel==='nodes'){{
      // Show previous-step nodes with dashed ring
      var prevN=step.prev_nodes||[];
      for(var i=0;i<prevN.length;i++){{
        var s=SAFE[prevN[i]];
        if(!s) continue;
        var rg=el(U+'-rg-'+s); if(rg){{ rg.setAttribute('opacity','0.5'); rg.setAttribute('stroke',COLORS.prevAccent); }}
        var fl=el(U+'-fl-'+s); if(fl){{ fl.setAttribute('fill',COLORS.prevAccent); fl.setAttribute('opacity','0.45'); }}
        var lb=el(U+'-lb-'+s); if(lb){{ lb.setAttribute('opacity','0.6'); }}
      }}
      // Show selected nodes with glow
      for(var i=0;i<step.nodes.length;i++){{
        var s=SAFE[step.nodes[i]];
        if(!s) continue;
        var gw=el(U+'-gw-'+s); if(gw){{ gw.setAttribute('opacity','0.5'); }}
        var fl=el(U+'-fl-'+s); if(fl){{ fl.setAttribute('fill',COLORS.accent); fl.setAttribute('opacity','1'); }}
        var lb=el(U+'-lb-'+s); if(lb){{ lb.setAttribute('opacity','1'); }}
      }}
      // Related edges
      var re=step.related_edges||[];
      for(var i=0;i<re.length;i++){{
        var eg=el(U+'-eg-'+re[i]); if(eg){{ eg.setAttribute('opacity','0.4'); }}
        var eln=el(U+'-el-'+re[i]); if(eln){{ eln.setAttribute('stroke',COLORS.relEdge); }}
      }}
    }} else {{
      // Edge selection
      // Show related nodes with ring
      var relN=step.related_nodes||[];
      for(var i=0;i<relN.length;i++){{
        var s=SAFE[relN[i]];
        if(!s) continue;
        var rg=el(U+'-rg-'+s); if(rg){{ rg.setAttribute('opacity','0.4'); rg.setAttribute('stroke',COLORS.accent); }}
        var fl=el(U+'-fl-'+s); if(fl){{ fl.setAttribute('fill',COLORS.accentLight); fl.setAttribute('opacity','0.7'); }}
        var lb=el(U+'-lb-'+s); if(lb){{ lb.setAttribute('opacity','0.8'); }}
      }}
      // Previous edges (dashed)
      var prevE=step.prev_edges||[];
      for(var i=0;i<prevE.length;i++){{
        var eg=el(U+'-eg-'+prevE[i]); if(eg){{ eg.setAttribute('opacity','0.3'); }}
        var eln=el(U+'-el-'+prevE[i]); if(eln){{ eln.setAttribute('stroke',COLORS.prevAccent); }}
      }}
      // Selected edges — bright + thick
      for(var i=0;i<step.edges.length;i++){{
        var ek=step.edges[i];
        var eg=el(U+'-eg-'+ek); if(eg){{ eg.setAttribute('opacity','1'); }}
        var eln=el(U+'-el-'+ek);
        if(eln){{
          eln.setAttribute('stroke',COLORS.selEdge);
          eln.setAttribute('stroke-width',String(COLORS.edgeWidth*2.5));
        }}
        var ea=el(U+'-ea-'+ek); if(ea){{ ea.setAttribute('fill',COLORS.selEdge); }}
      }}
    }}

    counter.textContent='Step '+(k+1)+' / '+N;
    slider.value=k;
  }}

  function advance(){{
    curStep++;
    if(curStep>=N){{ clearInterval(timer); playing=false; playBtn.innerHTML='&#9654;'; curStep=N-1; return; }}
    applyStep(curStep);
  }}

  slider.addEventListener('input',function(){{
    curStep=parseInt(this.value);
    applyStep(curStep);
  }});

  playBtn.addEventListener('click',function(){{
    if(playing){{ clearInterval(timer); playing=false; playBtn.innerHTML='&#9654;'; }}
    else {{
      if(curStep>=N-1){{ curStep=-1; }}
      var d=parseInt(speedSel.value);
      timer=setInterval(advance,d);
      playing=true;
      playBtn.innerHTML='&#9646;&#9646;';
    }}
  }});

  speedSel.addEventListener('change',function(){{
    if(playing){{ clearInterval(timer); timer=setInterval(advance,parseInt(speedSel.value)); }}
  }});

  // Pill clicks
  var pills=document.querySelectorAll('[id^="'+U+'-pill-"]');
  for(var i=0;i<pills.length;i++){{
    pills[i].addEventListener('click',function(){{
      var s=parseInt(this.getAttribute('data-step'));
      curStep=s; applyStep(s); slider.value=s;
    }});
  }}

  applyStep(0);
}})();
</script>
</div>"""


# ---------------------------------------------------------------------------
# Step-through renderer
# ---------------------------------------------------------------------------

_step_counter = 0


def _delta_json(delta) -> dict:
    """A step delta in the compact form the step player reads."""
    delta_obj = {
        "na": list(delta.nodes_added),
        "nr": list(delta.nodes_removed),
        "ea": [list(e) for e in delta.edges_added],
        "er": [list(e) for e in delta.edges_removed],
    }
    if delta.attrs_changed:
        delta_obj["ac"] = {
            k: _delta_val_repr(v) for k, v in delta.attrs_changed.items()
        }
    if delta.nodes_mutated:
        delta_obj["nm"] = list(delta.nodes_mutated)
    if delta.edges_mutated:
        delta_obj["em"] = [list(e) for e in delta.edges_mutated]
    if delta.nodes_selected:
        delta_obj["sn"] = list(delta.nodes_selected)
    if delta.edges_selected:
        delta_obj["se"] = [list(e) for e in delta.edges_selected]
    if delta.metric_values:
        delta_obj["mv"] = {
            k: round(v, 4) if v is not None else 0.0
            for k, v in delta.metric_values.items()
        }
    if delta.node_groups:
        delta_obj["ng"] = dict(delta.node_groups)
    if delta.edge_metric_values:
        delta_obj["emv"] = [
            [src, tgt, round(v, 4) if v is not None else 0.0]
            for (src, tgt), v in delta.edge_metric_values.items()
        ]
    return delta_obj


def _delta_val_repr(v):
    """Format a delta attribute value for JSON display."""
    if isinstance(v, dict):
        return str(v)
    if isinstance(v, (list, tuple)):
        return str(v)
    return v


_PROGRESSIVE_MAX_NODES = 300
_PROGRESSIVE_PAIR_BUDGET = 2_500_000


def _relax_layout(
    labels: List[str],
    edges: List[Tuple[str, str]],
    pos: Dict[str, List[float]],
    k: float,
    iterations: int,
    temp_start: float,
    mobility: Dict[str, float],
) -> None:
    """Warm-started Fruchterman-Reingold relaxation, updating `pos` in place."""
    n = len(labels)
    if n < 2 or iterations <= 0:
        return
    idx = {lbl: i for i, lbl in enumerate(labels)}
    xs = [pos[lbl][0] for lbl in labels]
    ys = [pos[lbl][1] for lbl in labels]
    mob = [mobility.get(lbl, 1.0) for lbl in labels]
    pairs = [(idx[s], idx[t]) for s, t in edges if s in idx and t in idx and s != t]
    k2 = k * k
    gravity = 0.6
    temp_end = k * 0.01
    for it in range(iterations):
        cx = sum(xs) / n
        cy = sum(ys) / n
        temp = temp_start + (temp_end - temp_start) * it / max(iterations - 1, 1)
        dx_acc = [0.0] * n
        dy_acc = [0.0] * n
        for i in range(n):
            xi, yi = xs[i], ys[i]
            for j in range(i + 1, n):
                dx = xi - xs[j]
                dy = yi - ys[j]
                d2 = dx * dx + dy * dy
                if d2 < 0.01:
                    dx, dy, d2 = 0.1 * ((i - j) % 3 - 1) + 0.05, 0.1, 0.0225
                f = k2 / d2
                dx_acc[i] += dx * f
                dy_acc[i] += dy * f
                dx_acc[j] -= dx * f
                dy_acc[j] -= dy * f
        for a, b in pairs:
            dx = xs[a] - xs[b]
            dy = ys[a] - ys[b]
            d = math.sqrt(dx * dx + dy * dy) or 0.01
            f = d / k
            dx_acc[a] -= dx * f
            dy_acc[a] -= dy * f
            dx_acc[b] += dx * f
            dy_acc[b] += dy * f
        for i in range(n):
            dx = dx_acc[i] - (xs[i] - cx) * gravity
            dy = dy_acc[i] - (ys[i] - cy) * gravity
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0:
                step = min(d, temp * mob[i])
                xs[i] += dx / d * step
                ys[i] += dy / d * step
    for i, lbl in enumerate(labels):
        pos[lbl] = [xs[i], ys[i]]


def _align_to_previous(
    pos: Dict[str, List[float]],
    before: Dict[str, Tuple[float, float]],
) -> None:
    """Remove rigid rotation/translation of `pos` relative to the shared nodes in `before`."""
    shared = [lbl for lbl in before if lbl in pos]
    if not shared:
        return
    m = len(shared)
    ax = sum(pos[lbl][0] for lbl in shared) / m
    ay = sum(pos[lbl][1] for lbl in shared) / m
    bx = sum(before[lbl][0] for lbl in shared) / m
    by = sum(before[lbl][1] for lbl in shared) / m
    theta = 0.0
    if m >= 2:
        num = den = 0.0
        for lbl in shared:
            px, py = pos[lbl][0] - ax, pos[lbl][1] - ay
            qx, qy = before[lbl][0] - bx, before[lbl][1] - by
            num += px * qy - py * qx
            den += px * qx + py * qy
        theta = math.atan2(num, den)
    c, s = math.cos(theta), math.sin(theta)
    for lbl, p in pos.items():
        px, py = p[0] - ax, p[1] - ay
        pos[lbl] = [bx + c * px - s * py, by + s * px + c * py]


def _progressive_layouts(
    states: List[Dict[str, list]],
    final_order: List[str],
    node_size: float,
    layout: str,
    width: float,
    height: float,
    seed: Optional[int] = None,
) -> Optional[Tuple[List[Dict[str, Tuple[float, float]]], List[int], List[List[str]]]]:
    """
    Compute one layout frame per structurally distinct graph state.

    Returns `(frames, state_frame, present)` where each frame maps every label that ever appears
    to a position (nodes not yet added sit where they will first appear; removed nodes keep their
    last position), `state_frame[i]` is the frame index for `states[i]`, and `present[f]` lists
    the nodes actually in the graph for frame `f`. Returns `None` when the
    graph is too large to lay out progressively. Without a `seed`, the placement jitter uses a
    fixed seed, so the frames are reproducible either way.
    """
    if not final_order or len(final_order) > _PROGRESSIVE_MAX_NODES:
        return None

    # Collapse consecutive states with identical structure into one frame
    frame_states: List[Tuple[List[str], List[Tuple[str, str]]]] = []
    state_frame: List[int] = []
    prev_key = None
    for st in states:
        key = (tuple(st["nodes"]), tuple(st["edges"]))
        if key != prev_key:
            frame_states.append((list(st["nodes"]), list(st["edges"])))
            prev_key = key
        state_frame.append(max(len(frame_states) - 1, 0))
    if not frame_states:
        return None

    cx, cy = width / 2.0, height / 2.0
    raw_frames: List[Dict[str, Tuple[float, float]]] = []

    if layout == "circular":
        order_idx = {lbl: i for i, lbl in enumerate(final_order)}
        for nodes, _ in frame_states:
            present = sorted(nodes, key=lambda lbl: order_idx.get(lbl, 0))
            raw_frames.append(_circular_layout(present, width, height))
    else:
        pad = 60.0
        k = math.sqrt(max((width - 2 * pad) * (height - 2 * pad), 1.0) / len(final_order)) * 0.55
        iters_base = 50
        total_pairs = sum(len(nodes) ** 2 / 2 * iters_base for nodes, _ in frame_states)
        scale = min(1.0, _PROGRESSIVE_PAIR_BUDGET / max(total_pairs, 1.0))
        golden = math.pi * (3 - math.sqrt(5))
        rng = random.Random(7 if seed is None else seed)
        pos: Dict[str, List[float]] = {}
        placed_count = 0
        prev_edges: set = set()

        for nodes, edges in frame_states:
            node_set = set(nodes)
            edge_set = set(edges)
            # Nodes whose surroundings changed this frame are free to move
            touched: set = set()
            for s, t in edge_set ^ prev_edges:
                touched.add(s)
                touched.add(t)
            prev_edges = edge_set
            for lbl in list(pos):
                if lbl not in node_set:
                    del pos[lbl]
            adj: Dict[str, List[str]] = {lbl: [] for lbl in nodes}
            for s, t in edges:
                if s in adj and t in adj:
                    adj[s].append(t)
                    adj[t].append(s)

            new_nodes = [lbl for lbl in nodes if lbl not in pos]
            pending = list(new_nodes)
            while pending:
                # Prefer nodes that already have a placed neighbor, so new structure grows
                # outward from the existing graph
                pick = None
                for lbl in pending:
                    if any(nb in pos for nb in adj[lbl]):
                        pick = lbl
                        break
                if pick is None:
                    pick = pending[0]
                pending.remove(pick)

                if pos:
                    gx = sum(p[0] for p in pos.values()) / len(pos)
                    gy = sum(p[1] for p in pos.values()) / len(pos)
                else:
                    gx, gy = cx, cy
                placed_nbs = [pos[nb] for nb in adj[pick] if nb in pos]
                angle = golden * placed_count + rng.uniform(-0.4, 0.4)
                placed_count += 1
                if placed_nbs:
                    nx = sum(p[0] for p in placed_nbs) / len(placed_nbs)
                    ny = sum(p[1] for p in placed_nbs) / len(placed_nbs)
                    ox, oy = nx - gx, ny - gy
                    norm = math.hypot(ox, oy)
                    if norm > 1e-6:
                        base = math.atan2(oy, ox)
                        angle = base + rng.uniform(-0.6, 0.6)
                    pos[pick] = [nx + math.cos(angle) * k * 0.8, ny + math.sin(angle) * k * 0.8]
                elif not pos:
                    pos[pick] = [cx, cy]
                else:
                    # Unconnected node: next slot on a sunflower spiral around the centroid
                    radius = k * 0.9 * math.sqrt(len(pos) + 0.5)
                    pos[pick] = [gx + math.cos(angle) * radius, gy + math.sin(angle) * radius]

            new_set = set(new_nodes)
            frac_changed = len(new_set | (touched & node_set)) / max(len(nodes), 1)
            iters = max(int((15 + iters_base * min(1.0, frac_changed * 2 + 0.3)) * scale), 6)
            temp_start = k * (0.15 + 0.45 * min(1.0, frac_changed * 2))
            mobility = {
                lbl: 1.0 if lbl in new_set else (0.6 if lbl in touched else 0.15)
                for lbl in nodes
            }
            before = {lbl: (pos[lbl][0], pos[lbl][1]) for lbl in nodes if lbl not in new_set}
            _relax_layout(nodes, edges, pos, k, iters, temp_start, mobility)
            _align_to_previous(pos, before)
            raw_frames.append({lbl: (p[0], p[1]) for lbl, p in pos.items()})

        # One uniform transform for all frames. Scale so the final frame's median edge length is
        # a comfortable multiple of the node radius; with no edges, fit the canvas instead.
        all_x = [p[0] for fr in raw_frames for p in fr.values()]
        all_y = [p[1] for fr in raw_frames for p in fr.values()]
        if all_x:
            bw = max(max(all_x) - min(all_x), 1.0)
            bh = max(max(all_y) - min(all_y), 1.0)
            s = min((width - 2 * pad) / bw, (height - 2 * pad) / bh, 1.5)
            last_frame = raw_frames[-1]
            lens = sorted(
                math.dist(last_frame[a], last_frame[b])
                for a, b in frame_states[-1][1]
                if a in last_frame and b in last_frame and a != b
            )
            if lens and lens[len(lens) // 2] > 1e-6:
                s = node_size * 4.5 / lens[len(lens) // 2]
            mx = (max(all_x) + min(all_x)) / 2
            my = (max(all_y) + min(all_y)) / 2
            raw_frames = [
                {lbl: (cx + (x - mx) * s, cy + (y - my) * s) for lbl, (x, y) in fr.items()}
                for fr in raw_frames
            ]

    # Fill each frame so every label has a position: before first appearance use the first
    # position, after removal keep the last known position
    all_labels = list(final_order)
    first_pos: Dict[str, Tuple[float, float]] = {}
    for fr in raw_frames:
        for lbl, p in fr.items():
            first_pos.setdefault(lbl, p)
    frames: List[Dict[str, Tuple[float, float]]] = []
    last_pos: Dict[str, Tuple[float, float]] = {}
    for fr in raw_frames:
        last_pos.update(fr)
        full = {}
        for lbl in all_labels:
            if lbl in fr:
                full[lbl] = fr[lbl]
            elif lbl in last_pos:
                full[lbl] = last_pos[lbl]
            elif lbl in first_pos:
                full[lbl] = first_pos[lbl]
            else:
                full[lbl] = (cx, cy)
        frames.append(full)
    return frames, state_frame, [nodes for nodes, _ in frame_states]


def _speed_options_html(speed: float) -> str:
    """`<option>`s for the playback speed menu: 0.5x/1x/2x/4x, plus `speed` if it's another value.

    Each option's value is the delay between steps in ms (1200 ms at 1x), and `speed` is selected.
    """
    speeds = sorted({0.5, 1.0, 2.0, 4.0, float(speed)})
    return "\n".join(
        f'      <option value="{round(1200 / s)}"{" selected" if s == speed else ""}>'
        f"{s:g}&times;</option>"
        for s in speeds
    )


# The step player's button icons, drawn in the button's text color (the arrow glyphs for these
# differ by font, and make "play" and "next" look the same)
_ICON_PREV = (
    '<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">'
    '<rect x="1" y="1.5" width="2" height="9"/><path d="M11 1.5v9L4 6z"/></svg>'
)
_ICON_NEXT = (
    '<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">'
    '<path d="M1 1.5v9L8 6z"/><rect x="9" y="1.5" width="2" height="9"/></svg>'
)
_ICON_PLAY = (
    '<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">'
    '<path d="M2.5 1v10L11 6z"/></svg>'
)
_ICON_PAUSE = (
    '<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">'
    '<rect x="2" y="1.5" width="3" height="9"/><rect x="7" y="1.5" width="3" height="9"/></svg>'
)


def render_steps(
    graph,
    steps: list,
    full_steps: Optional[list] = None,
    baseline: Optional[tuple] = None,
    layout: str = "force",
    theme: Optional[str] = None,
    width: int = 700,
    height: int = 500,
    node_size: float = 16.0,
    edge_width: float = 1.2,
    show_labels: bool = True,
    auto_play: bool = False,
    speed: float = 1.0,
    layout_mode: str = "progressive",
    output: Optional[str] = None,
    seed: Optional[int] = None,
    highlight=None,
    highlight_edges=None,
) -> "StepPlayer":
    """Render an interactive step-through visualization of recorded graph steps."""
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not speed > 0:
        raise ValueError(f"speed must be a positive number, got {speed!r}")
    global _step_counter
    _step_counter += 1
    uid = f"stv{_step_counter}"

    _check_style(layout, theme, layout_mode, stored_ok=True)
    colors = _Colors(theme)
    th = colors.theme
    sel_color = colors.pick("#7c3aed", "#a78bfa")
    node_color = colors.pick(DEFAULT_NODE_COLOR, "#60a5fa")

    if not steps:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            f'<rect width="100%" height="100%" fill="{th.bg}"/>'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            f'fill="{th.label_color}" font-family="{FONT_FAMILY}" '
            f'font-size="14">No steps recorded</text></svg>'
        )
        return colors.finish(svg, svg, output)

    is_directed = graph.is_directed()

    # --- Build cumulative step states (using labels) ---
    # When full_steps is given (filtered playback), compute state from the full
    # history so the graph is correct at each displayed step, then extract only
    # the states for the steps we're actually showing.
    state_source = full_steps if full_steps is not None else steps
    visible_indices = {id(s) for s in steps} if full_steps is not None else None

    step_states = []
    all_states = []
    visible_state_idx = []
    # Start from what existed when the log was last cleared (empty if it never was)
    base_nodes, base_edges = baseline if baseline is not None else ([], [])
    cum_nodes: set = set(base_nodes)
    cum_edges: set = set(base_edges)
    all_nodes: set = set(cum_nodes)
    all_edges: set = set(cum_edges)

    for step in state_source:
        delta = step.delta
        for label in delta.nodes_added:
            cum_nodes.add(label)
        for src, tgt in delta.edges_added:
            cum_edges.add((src, tgt))
        for label in delta.nodes_removed:
            cum_nodes.discard(label)
            cum_edges = {(s, t) for s, t in cum_edges if s != label and t != label}
        for pair in delta.edges_removed:
            cum_edges.discard(pair)
        all_nodes.update(cum_nodes)
        all_edges.update(cum_edges)
        state = {
            "nodes": sorted(cum_nodes),
            "edges": [(s, t) for s, t in sorted(cum_edges)],
        }
        all_states.append(state)
        if visible_indices is None or id(step) in visible_indices:
            visible_state_idx.append(len(all_states) - 1)
            step_states.append(state)

    # --- Build label <-> id maps from graph (for nodes still in graph) ---
    node_ids = list(graph.get_node_ids(limit=None))
    id_to_label: Dict[str, str] = {}
    label_to_id: Dict[str, str] = {}
    for nid in node_ids:
        obj = graph.get_node(nid)
        label = str(obj.label) if obj and obj.label else str(nid)
        id_to_label[str(nid)] = label
        label_to_id[label] = str(nid)

    # For deleted nodes not in graph, use label as its own ID
    for label in all_nodes:
        if label not in label_to_id:
            label_to_id[label] = label
            id_to_label[label] = label

    # All label-IDs for layout (sorted for deterministic order)
    all_label_ids = sorted(label_to_id[lbl] for lbl in all_nodes)
    all_edge_ids = [(label_to_id.get(s, s), label_to_id.get(t, t)) for s, t in sorted(all_edges)]

    # --- Adaptive sizes ---
    n = len(all_label_ids)
    if node_size == 16.0:
        avg_spacing = math.sqrt(float(width * height) / max(n, 1))
        node_size = max(3.5, min(16.0, avg_spacing * 0.15))
    if edge_width == 1.2:
        edge_width = max(0.5, 3.0 - n * 0.06) if n < 40 else max(0.5, 1.2 - (n - 40) * 0.002)
    edge_width = max(edge_width, 0.5)

    # --- Layout (on maximal graph state — all nodes/edges that ever appeared) ---
    if layout == "circular":
        positions = _circular_layout(
            all_label_ids, float(width), float(height), edges=all_edge_ids
        )
    elif layout == "random":
        positions = _random_layout(all_label_ids, float(width), float(height), seed=seed)
    elif layout not in LAYOUTS:
        # Stored positions for the nodes the graph still has; ones deleted since are spread at
        # random, as they have no position of their own
        present = graph.get_node_ids(limit=None)
        positions = _stored_layout(graph, present, layout, float(width), float(height))
        gone = [nid for nid in all_label_ids if nid not in positions]
        positions.update(_random_layout(gone, float(width), float(height), seed=seed))
    else:
        fr_iters = 800 if n > 300 else (600 if n > 150 else 400)
        positions = _force_directed_layout(
            all_label_ids,
            all_edge_ids,
            float(width),
            float(height),
            iterations=fr_iters,
            seed=seed,
        )

    # Overlap removal (force and random layouts only)
    radii = {nid: node_size for nid in all_label_ids}
    if layout in ("force", "random") and n >= 2:
        positions = _remove_overlaps(positions, radii, padding=2.0, seed=seed)

    # Progressive layout: one frame per structurally distinct state, animated between steps.
    # `step_frames[i]` is the frame index for visible step i; frames are keyed by label.
    layout_frames: List[Dict[str, List[float]]] = []
    step_frames: List[int] = []
    if layout_mode == "progressive" and layout in ("force", "circular"):
        final_order = [id_to_label.get(nid, nid) for nid in all_label_ids]
        prog = _progressive_layouts(
            all_states, final_order, node_size, layout, float(width), float(height), seed=seed
        )
        if prog is not None:
            frames, state_frame, frame_present = prog
            used = sorted({state_frame[i] for i in visible_state_idx})
            used_present = [frame_present[f] for f in used]
            remap = {f: j for j, f in enumerate(used)}
            layout_frames = [
                {lbl: [round(x, 1), round(y, 1)] for lbl, (x, y) in frames[f].items()}
                for f in used
            ]
            step_frames = [remap[state_frame[i]] for i in visible_state_idx]
            positions = {
                label_to_id.get(lbl, lbl): (xy[0], xy[1])
                for lbl, xy in layout_frames[0].items()
            }

    # ViewBox. For progressive layouts each frame gets a camera: the final framing, widened only
    # when that frame's nodes extend past it. `bg_box` covers every camera position.
    label_map = {}
    if show_labels:
        for nid in all_label_ids:
            label_map[nid] = id_to_label.get(nid, nid)
    frame_viewboxes: List[List[float]] = []
    if layout_frames:
        fitted = []
        prev_present: List[str] = []
        for fr, present in zip(layout_frames, used_present):
            # Include the previous frame's nodes so just-removed nodes stay in view
            in_view = set(present) | set(prev_present)
            prev_present = present
            fr_pos = {label_to_id.get(lbl, lbl): (fr[lbl][0], fr[lbl][1]) for lbl in in_view}
            fitted.append(_compute_viewbox(fr_pos, radii, labels=label_map if show_labels else None))
        final_vb = fitted[-1]
        for fv in fitted:
            if fv == (0, 0, 100, 100):
                fv = final_vb
            x0 = min(fv[0], final_vb[0])
            y0 = min(fv[1], final_vb[1])
            x1 = max(fv[0] + fv[2], final_vb[0] + final_vb[2])
            y1 = max(fv[1] + fv[3], final_vb[1] + final_vb[3])
            frame_viewboxes.append([round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)])
        vb = tuple(frame_viewboxes[step_frames[0]])
        bx0 = min(v[0] for v in frame_viewboxes)
        by0 = min(v[1] for v in frame_viewboxes)
        bx1 = max(v[0] + v[2] for v in frame_viewboxes)
        by1 = max(v[1] + v[3] for v in frame_viewboxes)
        bg_box = (bx0, by0, bx1 - bx0, by1 - by0)
    else:
        vb = _compute_viewbox(positions, radii, labels=label_map if show_labels else None)
        bg_box = vb
    vb_str = f"{vb[0]:.1f} {vb[1]:.1f} {vb[2]:.1f} {vb[3]:.1f}"

    # --- Build SVG ---
    # Glows for added, removed, selected and mutated elements. Edges get their own copies over
    # the whole canvas: a filter region sized to the element's bounding box is empty for a
    # horizontal or vertical line, which hides the line altogether.
    canvas = 'filterUnits="userSpaceOnUse" x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}"'
    glow_defs = "".join(
        f'<filter id="{uid}-glow-{name}{suffix}" {region}>'
        f'<feGaussianBlur stdDeviation="{blur}" result="blur"/>'
        f'<feFlood flood-color="{color}" flood-opacity="{alpha}" result="color"/>'
        f'<feComposite in="color" in2="blur" operator="in" result="glow"/>'
        f'<feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f"</filter>"
        for name, color, alpha, blur in (
            ("add", "#22c55e", 0.6, 3),
            ("rm", "#ef4444", 0.6, 3),
            ("sel", sel_color, 0.5, 3),
            ("mut", "#f59e0b", 0.5, 2.5),
        )
        for suffix, region in (
            ("", 'x="-50%" y="-50%" width="200%" height="200%"'),
            ("-e", canvas.format(*bg_box)),
        )
    )
    svg_parts = [
        f'<svg id="{uid}-svg" xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="{vb_str}" '
        f'style="font-family:{FONT_FAMILY};">',
        f"<defs>{glow_defs}</defs>",
        f'<rect x="{bg_box[0]:.1f}" y="{bg_box[1]:.1f}" width="{bg_box[2]:.1f}" '
        f'height="{bg_box[3]:.1f}" fill="{th.bg}"/>',
    ]

    # Edges (all hidden initially; data-src/data-tgt use labels)
    hit_width = max(12, edge_width * 6)
    for src_label, tgt_label in sorted(all_edges):
        src_id = label_to_id.get(src_label, src_label)
        tgt_id = label_to_id.get(tgt_label, tgt_label)
        x1, y1 = positions.get(src_id, (0, 0))
        x2, y2 = positions.get(tgt_id, (0, 0))
        svg_parts.append(
            f'<line class="step-edge-hit" '
            f'data-src="{html_mod.escape(src_label)}" '
            f'data-tgt="{html_mod.escape(tgt_label)}" '
            f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="transparent" stroke-width="{hit_width:.1f}" '
            f'fill="none" pointer-events="stroke"/>'
        )
        svg_parts.append(
            f'<line class="step-edge" '
            f'data-src="{html_mod.escape(src_label)}" '
            f'data-tgt="{html_mod.escape(tgt_label)}" '
            f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{th.edge_color}" stroke-width="{edge_width:.1f}" '
            f'stroke-opacity="0.4" opacity="0" pointer-events="none" '
            f'style="transition:opacity 0.4s ease,stroke 0.3s ease,stroke-width 0.3s ease;"/>'
        )
        if is_directed:
            dx, dy = x2 - x1, y2 - y1
            ln = math.sqrt(dx * dx + dy * dy) or 1
            ux, uy = dx / ln, dy / ln
            ax = x2 - ux * (node_size + 2)
            ay = y2 - uy * (node_size + 2)
            sz = max(4, edge_width * 2.5)
            px, py = -uy, ux
            svg_parts.append(
                f'<polygon class="step-edge" '
                f'data-src="{html_mod.escape(src_label)}" '
                f'data-tgt="{html_mod.escape(tgt_label)}" '
                f'points="{ax + ux * sz:.1f},{ay + uy * sz:.1f} '
                f"{ax + px * sz * 0.5:.1f},{ay + py * sz * 0.5:.1f} "
                f'{ax - px * sz * 0.5:.1f},{ay - py * sz * 0.5:.1f}" '
                f'fill="{th.edge_color}" opacity="0" '
                f'style="transition:opacity 0.4s ease,fill 0.3s ease;"/>'
            )

    # Nodes (all hidden initially; data-label uses label)
    for nid in all_label_ids:
        x, y = positions.get(nid, (0, 0))
        label = id_to_label.get(nid, nid)
        svg_parts.append(
            f'<circle class="step-node" data-label="{html_mod.escape(label)}" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{node_size:.1f}" '
            f'fill="{node_color}" stroke="none" stroke-width="2" opacity="0" '
            f'style="transition:opacity 0.3s ease,fill 0.3s ease,r 0.3s ease,stroke 0.3s ease,filter 0.3s ease;"/>'
        )
        svg_parts.append(
            f'<text class="step-order" data-label="{html_mod.escape(label)}" '
            f'x="{x:.1f}" y="{y + 4:.1f}" '
            f'text-anchor="middle" font-size="{max(9, node_size * 0.7):.0f}" '
            f'font-weight="700" fill="#fff" opacity="0" '
            f'pointer-events="none" '
            f'style="transition:opacity 0.3s ease;"></text>'
        )
        if show_labels:
            svg_parts.append(
                f'<text class="step-label" data-label="{html_mod.escape(label)}" '
                f'x="{x:.1f}" y="{y + node_size + 12:.1f}" '
                f'text-anchor="middle" font-size="10" '
                f'fill="{th.label_color}" opacity="0" '
                f'style="transition:opacity 0.3s ease;">'
                f"{html_mod.escape(label)}</text>"
            )

    svg_parts.append("</svg>")
    svg_str = "\n".join(svg_parts)

    # --- Step data JSON (expand traverse sub-steps) ---
    # Prepend an initial step showing the empty graph state, or the graph as it was when the
    # log was cleared
    dir_label = "directed" if is_directed else "undirected"
    if base_nodes:
        counts = (
            f"{len(base_nodes)} node{'s' if len(base_nodes) != 1 else ''}, "
            f"{len(base_edges)} edge{'s' if len(base_edges) != 1 else ''}"
        )
        init_code = f"# Graph when the step log was cleared: {counts}"
        init_narration = f"The graph as it was when the step log was cleared: {counts}."
    else:
        init_code = f"Graph(directed={'True' if is_directed else 'False'})"
        init_narration = f"Empty {dir_label} graph."
    step_data = [
        {
            "kind": "init",
            "method": "",
            "code": init_code,
            "narration": init_narration,
            "nodes": sorted(base_nodes),
            "edges": [list(e) for e in sorted(base_edges)],
            "delta": {"na": [], "nr": [], "ea": [], "er": []},
        }
    ]
    group_counter = 0
    for i, step in enumerate(steps):
        state_nodes = step_states[i]["nodes"]
        state_edges = step_states[i]["edges"]
        step_data_mark = len(step_data)

        if step.kind == "traverse" and step.sub_steps:
            # Build adjacency from the graph for frontier computation
            adj: Dict[str, List[str]] = {}
            for nid in graph.get_node_ids(limit=None):
                obj = graph.get_node(nid)
                lbl = str(obj.label) if obj and obj.label else str(nid)
                try:
                    nbs = []
                    for nb in graph.get_neighbors(nid):
                        nb_obj = graph.get_node(nb)
                        nbs.append(str(nb_obj.label) if nb_obj and nb_obj.label else str(nb))
                    adj[lbl] = nbs
                    if not graph.is_directed():
                        for nb_lbl in nbs:
                            adj.setdefault(nb_lbl, []).append(lbl)
                except Exception:
                    pass
            for k in adj:
                adj[k] = list(dict.fromkeys(adj[k]))

            # Infer tree edges for this traversal
            visit_labels = [sub.delta.nodes_added[0] for sub in step.sub_steps]
            tree_edge_for: Dict[str, Tuple[str, str]] = {}
            for sub in step.sub_steps[1:]:
                lbl = sub.delta.nodes_added[0]
                parts = sub.code.split(" → ")
                if len(parts) == 2:
                    tree_edge_for[lbl] = (parts[0], parts[1])

            group_counter += 1
            gid = group_counter
            n_subs = len(step.sub_steps)
            parent_code = step.code
            # A note() on the traversal leads its first frame
            from edgecase.steps import auto_narrate

            own_note = step.narration if step.narration != auto_narrate(step) else None

            visited_so_far: set = set()
            visit_order: list = []
            tree_edges_so_far: list = []
            for j, sub in enumerate(step.sub_steps):
                lbl = sub.delta.nodes_added[0]
                visited_so_far.add(lbl)
                visit_order.append(lbl)

                current_edge = None
                if lbl in tree_edge_for:
                    te = tree_edge_for[lbl]
                    current_edge = [te[0], te[1]]
                    tree_edges_so_far.append(current_edge)

                frontier = set()
                for v in visited_so_far:
                    for nb in adj.get(v, []):
                        if nb not in visited_so_far:
                            frontier.add(nb)

                step_data.append(
                    {
                        "kind": "traverse",
                        "method": step.method,
                        "code": sub.code,
                        "narration": (
                            f"{own_note} {sub.narration}"
                            if j == 0 and own_note
                            else sub.narration
                        ),
                        "nodes": state_nodes,
                        "edges": state_edges,
                        "group": {
                            "id": gid,
                            "pos": j,
                            "total": n_subs,
                            "parent_code": parent_code,
                        },
                        "traverse": {
                            "current": lbl,
                            "visited": sorted(visited_so_far),
                            "frontier": sorted(frontier),
                            "tree_edges": [list(e) for e in tree_edges_so_far],
                            "current_edge": current_edge,
                            "step_num": j + 1,
                            "total": len(step.sub_steps),
                            "step_labels": {v: k + 1 for k, v in enumerate(visit_order)},
                        },
                    }
                )
        else:
            # Other steps with sub-steps (such as the rounds of `voterank()`) get a frame per
            # sub-step, bracketed as a group like a traversal
            frames = [(step.code, step.narration, step.delta, None)]
            if step.sub_steps:
                from edgecase.steps import auto_narrate

                group_counter += 1
                own_note = step.narration if step.narration != auto_narrate(step) else None
                frames = [
                    (
                        sub.code,
                        f"{own_note} {sub.narration}" if j == 0 and own_note else sub.narration,
                        sub.delta,
                        {
                            "id": group_counter,
                            "pos": j,
                            "total": len(step.sub_steps),
                            "parent_code": step.code,
                        },
                    )
                    for j, sub in enumerate(step.sub_steps)
                ]
            for code, narration, delta, group in frames:
                frame = {
                    "kind": step.kind,
                    "method": step.method,
                    "code": code,
                    "narration": narration,
                    "nodes": state_nodes,
                    "edges": state_edges,
                    "delta": _delta_json(delta),
                }
                if group is not None:
                    frame["group"] = group
                step_data.append(frame)
        if step_frames:
            for sd in step_data[step_data_mark:]:
                sd["lf"] = step_frames[i]

    # Append a final step showing the settled graph state
    final_nodes = step_states[-1]["nodes"] if step_states else []
    final_edges = step_states[-1]["edges"] if step_states else []
    n_nodes = len(final_nodes)
    n_edges = len(final_edges)
    step_data.append(
        {
            "kind": "done",
            "method": "",
            "code": f"# Final graph: {n_nodes} node{'s' if n_nodes != 1 else ''}, {n_edges} edge{'s' if n_edges != 1 else ''}",
            "narration": f"Graph ready — {n_nodes} node{'s' if n_nodes != 1 else ''}, {n_edges} edge{'s' if n_edges != 1 else ''}.",
            "nodes": final_nodes,
            "edges": final_edges,
            "delta": {"na": [], "nr": [], "ea": [], "er": []},
        }
    )
    if highlight is not None or highlight_edges is not None:
        # What highlight= picks out in the graph as it is now, by label, for the final frame
        picked = _resolve_highlight(graph, highlight, highlight_edges)

        def ends(eid):
            e = graph.get_edge(eid)
            return [id_to_label.get(str(e.source)), id_to_label.get(str(e.target))]

        step_data[-1]["hl"] = {
            level[0] + key: sorted(
                [id_to_label[str(nid)] for nid in picked[level][0]]
                if key == "n"
                else [ends(eid) for eid in picked[level][1]]
            )
            for level in HIGHLIGHT_LEVELS
            for key in ("n", "e")
        }
    if step_frames:
        step_data[0]["lf"] = step_frames[0]
        step_data[-1]["lf"] = step_frames[-1]

    steps_json = json.dumps(step_data)

    # --- Build HTML ---
    bg = th.bg
    fg = th.label_color
    accent = colors.pick("#4e79a7", "#60a5fa")
    border = colors.pick("#d0d7de", "#30363d")
    code_bg = colors.pick("#f6f8fa", "#161b22")
    narr_fg = colors.pick("#656d76", "#8b949e")
    ctrl_bg = colors.pick("#f0f3f6", "#0d1117")

    # Traversal colors
    trav_current = colors.pick("#2563eb", "#3b82f6")
    trav_frontier = colors.pick("#93c5fd", "#60a5fa")
    trav_unvisited = colors.pick("#d1d5db", "#4b5563")
    trav_tree_edge = colors.pick("#2563eb", "#60a5fa")

    # Highlight colors for the final frame, as in plot()
    hl_accent = colors.pick(*ACCENT_COLOR)
    hl_soft = colors.pick(_mix(ACCENT_COLOR[0], LIGHT.bg, 0.7), _mix(ACCENT_COLOR[1], DARK.bg, 0.7))
    hl_muted = colors.pick(*MUTED_NODE_COLOR)

    kw_color = colors.pick("#8839ef", "#cba6f7")
    fn_color = colors.pick("#1e66f5", "#89b4fa")
    str_color = colors.pick("#40a02b", "#a6e3a1")
    num_color = colors.pick("#fe640b", "#fab387")
    bool_color = colors.pick("#fe640b", "#fab387")
    cmt_color = colors.pick("#9ca0b0", "#6c7086")

    # Metric color ramp (low → high)
    # slate-300 to blue-600 (slate-700 to blue-400 in the dark theme)
    metric_ends = [((203, 213, 225), (51, 65, 85)), ((37, 99, 235), (96, 165, 250))]
    if colors.auto:
        # Mixed by the browser, since the two ends are variables
        lo, hi = (colors.pick(f"rgb{light}", f"rgb{dark}") for light, dark in metric_ends)
        metric_js = f"return 'color-mix(in srgb, {lo}, {hi} '+(t*100).toFixed(1)+'%)';"
    else:
        lo, hi = (list(colors.pick(light, dark)) for light, dark in metric_ends)
        metric_js = (
            f"var lo={lo},hi={hi};\n"
            "    var r=Math.round(lo[0]+(hi[0]-lo[0])*t);\n"
            "    var g=Math.round(lo[1]+(hi[1]-lo[1])*t);\n"
            "    var b=Math.round(lo[2]+(hi[2]-lo[2])*t);\n"
            "    return 'rgb('+r+','+g+','+b+')';"
        )

    # Timeline segment colors by kind
    tl_colors = {
        "init": colors.pick("#64748b", "#94a3b8"),
        "done": colors.pick("#64748b", "#94a3b8"),
        "build_nodes": "#22c55e",
        "build_edges": "#22c55e",
        "remove": "#ef4444",
        "mutate": "#f59e0b",
        "select": colors.pick("#7c3aed", "#a78bfa"),
        "traverse": "#3b82f6",
        "transform": "#06b6d4",
        "generate": "#06b6d4",
        "annotate": colors.pick("#9ca3af", "#6b7280"),
        "query": "#8b5cf6",
        "metric": "#8b5cf6",
    }
    tl_default = "#6b7280"
    tl_segs = []
    # Track groups for bracket rendering
    groups: dict = {}  # gid -> (start_idx, end_idx, color)
    for i, sd in enumerate(step_data):
        c = tl_colors.get(sd["kind"], tl_default)
        cls = "tl-seg active" if i == 0 else "tl-seg"
        grp = sd.get("group")
        grp_attr = ""
        if grp:
            gid = grp["id"]
            grp_attr = f' data-group="{gid}"'
            if gid not in groups:
                groups[gid] = {"start": i, "end": i, "color": c}
            else:
                groups[gid]["end"] = i
        tl_segs.append(
            f'<div class="{cls}" data-idx="{i}"{grp_attr} '
            f'style="background:{c}" title="{sd["kind"]}: {sd["method"]}()"></div>'
        )
    tl_html = "\n    ".join(tl_segs)

    # Build group bracket data for JS
    group_brackets = []
    for gid, ginfo in sorted(groups.items()):
        group_brackets.append(
            {"id": gid, "start": ginfo["start"], "end": ginfo["end"], "color": ginfo["color"]}
        )

    # --- Step list overlay items + notch indicators ---
    sl_items = []
    notch_items = []
    emitted_group_headers: set = set()
    for i, sd in enumerate(step_data):
        c = tl_colors.get(sd["kind"], tl_default)
        grp = sd.get("group")

        # Emit group header before first sub-step
        if grp and grp["id"] not in emitted_group_headers:
            emitted_group_headers.add(grp["id"])
            parent_code = grp["parent_code"]
            sl_items.append(
                f'<div class="sl-group-header" data-group="{grp["id"]}">'
                f'<span class="sl-badge" style="background:{colors.tint(c)};color:{c}">'
                f'{sd["kind"]}</span>'
                f'<span class="sl-method">{parent_code}</span>'
                f'<span class="sl-group-count">{grp["total"]} steps</span>'
                f"</div>"
            )

        cls = "sl-item active" if i == 0 else "sl-item"
        if grp:
            cls += " sl-sub"
        method = sd["method"] + "()" if sd["kind"] != "annotate" else ""
        narr = sd.get("narration", "")
        narr_esc = narr.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
        if len(narr_esc) > 50:
            narr_esc = narr_esc[:47] + "..."

        if grp:
            sub_label = sd.get("narration", sd["code"])
            if len(sub_label) > 40:
                sub_label = sub_label[:37] + "..."
            sub_label_esc = (
                sub_label.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
            )
            sl_items.append(
                f'<div class="{cls}" data-idx="{i}" data-group="{grp["id"]}" title="{narr_esc}">'
                f'<span class="sl-num">{grp["pos"] + 1}</span>'
                f'<span class="sl-sub-label">{sub_label_esc}</span>'
                f"</div>"
            )
        else:
            sl_items.append(
                f'<div class="{cls}" data-idx="{i}" title="{narr_esc}">'
                f'<span class="sl-num">{i + 1}</span>'
                f'<span class="sl-badge" style="background:{colors.tint(c)};color:{c}">'
                f"{sd['kind']}</span>"
                f'<span class="sl-method">{method}</span>'
                f"</div>"
            )
        ncls = "step-notch active" if i == 0 else "step-notch"
        notch_items.append(f'<div class="{ncls}" data-idx="{i}" style="background:{c}"></div>')
    sl_html = "\n      ".join(sl_items)
    notch_html = "\n      ".join(notch_items)

    # The font size and line height are set here so that a page's (e.g., a docs site's larger text)
    # don't spread out the panel below the drawing
    html = f"""<div id="{uid}" style="font-family:{FONT_FAMILY};font-size:13px;line-height:1.4;max-width:{width}px;margin:1em auto;overflow:hidden;color-scheme:{colors.scheme()};">
<style>
#{uid} .step-panel {{
  border:1px solid {border};
  border-radius:8px;
  overflow:hidden;
  background:{bg};
}}
#{uid} .step-svg-wrap {{
  border-bottom:1px solid {border};
  background:{bg};
}}
#{uid} .step-code-panel {{
  padding:12px 16px;
  border-bottom:1px solid {border};
  box-sizing:border-box;
  overflow:hidden;
  position:relative;
}}
#{uid} .step-code-panel pre {{
  margin:0;
  padding:8px 12px;
  background:{code_bg};
  border-radius:6px;
  font-size:12.5px;
  line-height:1.5;
  overflow:hidden;
  white-space:nowrap;
  color:{fg};
  position:relative;
  max-width:100%;
  box-sizing:border-box;
}}
#{uid} .step-code-panel pre code {{
  color:inherit;
  background:none;
  padding:0;
  font-size:inherit;
}}
#{uid} .step-modal-body pre {{
  color:inherit;
  background:none;
  border:none;
  padding:0;
}}
#{uid} .code-expand {{
  display:none;
  position:absolute;
  right:4px;
  top:50%;
  transform:translateY(-50%);
  background:{code_bg};
  padding:2px 6px 2px 12px;
  font-size:11px;
  font-weight:600;
  color:{accent};
  cursor:pointer;
  border-radius:3px;
  font-family:inherit;
  box-shadow:-8px 0 8px {code_bg};
  z-index:2;
}}
#{uid} .code-expand:hover {{
  color:{fg};
}}
#{uid} .kw {{ color:{kw_color}; }}
#{uid} .fn {{ color:{fn_color}; }}
#{uid} .str {{ color:{str_color}; }}
#{uid} .num {{ color:{num_color}; }}
#{uid} .bool {{ color:{bool_color}; }}
#{uid} .cmt {{ color:{cmt_color}; font-style:italic; }}
#{uid} .step-narration {{
  margin-top:8px;
  font-size:13px;
  color:{narr_fg};
  font-style:italic;
  overflow:hidden;
  display:-webkit-box;
  -webkit-line-clamp:2;
  -webkit-box-orient:vertical;
  cursor:default;
}}
#{uid} .step-narration.truncated {{
  cursor:pointer;
}}
#{uid} .step-note-text {{
  margin:0;
  font-size:14px;
  line-height:1.6;
  color:{fg};
  overflow:hidden;
  display:-webkit-box;
  -webkit-line-clamp:3;
  -webkit-box-orient:vertical;
}}
#{uid} .step-note-text.truncated {{
  cursor:pointer;
}}
#{uid} .step-more-link {{
  font-style:normal;
  font-size:11px;
  color:{accent};
  font-weight:500;
  cursor:pointer;
  margin-left:4px;
}}
#{uid} .step-modal-overlay {{
  display:none;
  position:fixed;
  top:0;left:0;right:0;bottom:0;
  background:rgba(0,0,0,{colors.pick(".4", ".6")});
  z-index:9999;
  align-items:center;
  justify-content:center;
}}
#{uid} .step-modal-overlay.open {{
  display:flex;
}}
#{uid} .step-modal {{
  background:{bg};
  border:1px solid {border};
  border-radius:10px;
  padding:20px 24px;
  max-width:520px;
  max-height:60vh;
  overflow-y:auto;
  width:90%;
  box-shadow:0 8px 32px rgba(0,0,0,.25);
}}
#{uid} .step-modal-close {{
  float:right;
  background:none;
  border:none;
  font-size:18px;
  color:{narr_fg};
  cursor:pointer;
  padding:0 4px;
  line-height:1;
}}
#{uid} .step-modal-close:hover {{ color:{fg}; }}
#{uid} .step-modal-body {{
  font-size:14px;
  line-height:1.7;
  color:{fg};
  white-space:pre-wrap;
  margin-top:4px;
}}
#{uid} .step-timeline {{
  display:flex;
  height:6px;
  gap:1px;
  padding:0 16px;
  background:{ctrl_bg};
  cursor:pointer;
  position:relative;
}}
#{uid} .step-timeline .tl-seg {{
  flex:1;
  border-radius:1px;
  opacity:0.45;
  transition:opacity 0.15s ease,height 0.15s ease,margin-top 0.15s ease;
  height:4px;
  margin-top:1px;
}}
#{uid} .step-timeline .tl-seg:hover {{
  opacity:0.75;
}}
#{uid} .step-timeline .tl-seg.active {{
  opacity:1;
  height:6px;
  margin-top:0;
}}
#{uid} .step-controls {{
  display:flex;
  align-items:center;
  gap:8px;
  padding:10px 16px;
  background:{ctrl_bg};
}}
#{uid} .step-controls button {{
  background:{accent};
  color:#fff;
  border:none;
  border-radius:4px;
  padding:0;
  width:30px;
  height:28px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  cursor:pointer;
  font-size:14px;
  font-family:inherit;
}}
#{uid} .step-controls button:hover {{ opacity:0.85; }}
#{uid} .step-controls .replay-btn {{
  display:none;
  background:{colors.pick("#9ca3af", "#6b7280")};
}}
#{uid} .step-controls input[type=range] {{
  flex:1;
  accent-color:{accent};
}}
#{uid} .step-counter {{
  font-size:12.5px;
  color:{fg};
  min-width:80px;
  text-align:right;
}}
#{uid} .step-speed {{
  border:1px solid {border};
  border-radius:4px;
  padding:2px 4px;
  background:transparent;
  color:{fg};
  font-size:11.5px;
  font-family:inherit;
  cursor:pointer;
}}
#{uid} .step-kind-badge {{
  display:block;
  width:fit-content;
  padding:2px 8px;
  border-radius:3px;
  font-size:11px;
  font-weight:600;
  margin-bottom:6px;
  background:{colors.tint(accent)};
  color:{accent};
}}
#{uid} .step-delta {{
  margin-top:6px;
  font-size:11.5px;
  line-height:1.6;
  color:{narr_fg};
}}
#{uid} .step-delta .delta-chip {{
  display:inline-block;
  padding:1px 7px;
  border-radius:3px;
  font-size:10.5px;
  font-weight:500;
  margin-right:4px;
  margin-bottom:2px;
}}
#{uid} .delta-add {{
  background:{colors.pick("#dcfce7", "#16533822")};
  color:{colors.pick("#15803d", "#4ade80")};
}}
#{uid} .delta-rm {{
  background:{colors.pick("#fee2e2", "#45181822")};
  color:{colors.pick("#b91c1c", "#f87171")};
}}
#{uid} .delta-attr {{
  background:{colors.pick("#fef3c7", "#42290622")};
  color:{colors.pick("#b45309", "#fbbf24")};
}}
#{uid} .delta-sel {{
  background:{colors.pick("#ede9fe", "#2e1a6522")};
  color:{colors.pick("#6d28d9", "#a78bfa")};
}}
#{uid} .step-tooltip {{
  display:none;
  position:absolute;
  pointer-events:none;
  z-index:100;
  background:{colors.pick("#fff", "#1e293b")};
  border:1px solid {border};
  border-radius:6px;
  padding:6px 10px;
  font-size:11.5px;
  line-height:1.5;
  color:{fg};
  box-shadow:0 2px 8px rgba(0,0,0,{colors.pick(".12", ".3")});
  max-width:220px;
  white-space:nowrap;
}}
#{uid} .step-tooltip.visible {{
  display:block;
}}
#{uid} .step-tooltip .tt-label {{
  font-weight:600;
  font-size:12px;
}}
#{uid} .step-tooltip .tt-state {{
  font-size:10.5px;
  margin-top:2px;
  opacity:0.75;
}}
#{uid} .step-notches {{
  position:absolute;
  left:0; top:0; bottom:0;
  width:20px;
  display:flex;
  flex-direction:column;
  justify-content:center;
  gap:1px;
  padding:8px 0;
  z-index:50;
  cursor:pointer;
}}
#{uid} .step-notch {{
  width:4px;
  height:4px;
  min-height:2px;
  border-radius:1px;
  margin-left:4px;
  opacity:0.35;
  transition:opacity 0.15s ease, background 0.15s ease;
}}
#{uid} .step-notch.active {{
  opacity:1;
  width:5px;
}}
#{uid} .step-overlay {{
  display:none;
  position:absolute;
  left:0; top:0;
  width:80%;
  max-height:80%;
  overflow-y:auto;
  background:{colors.pick("rgba(255,255,255,0.94)", "rgba(15,23,42,0.92)")};
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
  border-radius:0 8px 8px 0;
  border:1px solid {colors.pick("rgba(0,0,0,0.08)", "rgba(255,255,255,0.08)")};
  border-left:none;
  box-shadow:0 4px 24px rgba(0,0,0,{colors.pick(".12", ".4")});
  z-index:60;
  padding:6px 0;
  font-size:11.5px;
}}
#{uid} .step-overlay.visible {{
  display:block;
}}
#{uid} .sl-item {{
  display:flex;
  align-items:center;
  gap:6px;
  padding:5px 10px;
  cursor:pointer;
  transition:background 0.1s ease;
}}
#{uid} .sl-item:hover {{
  background:{colors.pick("rgba(0,0,0,0.04)", "rgba(255,255,255,0.06)")};
}}
#{uid} .sl-item.active {{
  background:{colors.tint(accent)};
  border-left:2px solid {accent};
}}
#{uid} .sl-num {{
  color:{narr_fg};
  font-size:10px;
  min-width:18px;
  text-align:right;
  flex-shrink:0;
}}
#{uid} .sl-badge {{
  font-size:9px;
  font-weight:600;
  padding:1px 5px;
  border-radius:3px;
  flex-shrink:0;
  white-space:nowrap;
}}
#{uid} .sl-method {{
  color:{fg};
  font-size:11px;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}}
#{uid} .sl-group-header {{
  display:flex;
  align-items:center;
  gap:6px;
  padding:6px 10px 2px 10px;
  font-size:11px;
  opacity:0.8;
  border-top:1px solid {colors.pick("rgba(0,0,0,0.06)", "rgba(255,255,255,0.06)")};
  margin-top:2px;
}}
#{uid} .sl-group-count {{
  color:{narr_fg};
  font-size:9px;
  margin-left:auto;
  flex-shrink:0;
}}
#{uid} .sl-item.sl-sub {{
  padding-left:28px;
}}
#{uid} .sl-sub-label {{
  color:{fg};
  font-size:10.5px;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}}
#{uid} .tl-bracket {{
  position:absolute;
  bottom:0;
  height:2px;
  border-radius:1px;
  opacity:0.6;
  pointer-events:none;
}}
</style>

<div class="step-panel">
  <div class="step-svg-wrap" style="position:relative;">
    {svg_str}
    <div class="step-tooltip" id="{uid}-tooltip"></div>
    <div class="step-notches" id="{uid}-notches">
      {notch_html}
    </div>
    <div class="step-overlay" id="{uid}-overlay">
      {sl_html}
    </div>
  </div>
  <div class="step-code-panel" id="{uid}-code">
    <span class="step-kind-badge" id="{uid}-badge"></span>
    <pre id="{uid}-codewrap"><code id="{uid}-codetext"></code><span class="code-expand" id="{uid}-codemore">[&hellip;]</span></pre>
    <p class="step-note-text" id="{uid}-notetext" style="display:none"></p>
    <div class="step-narration" id="{uid}-narration"></div>
    <div class="step-delta" id="{uid}-delta"></div>
  </div>
  <div class="step-timeline" id="{uid}-timeline">
    {tl_html}
  </div>
  <div class="step-controls">
    <button id="{uid}-prev" title="Previous step (&#8592;)">{_ICON_PREV}</button>
    <button id="{uid}-play" title="Play / Pause (Space)">{_ICON_PLAY}</button>
    <button id="{uid}-next" title="Next step (&#8594;)">{_ICON_NEXT}</button>
    <button id="{uid}-replay" class="replay-btn" title="Replay from start (Home)">&#8634;</button>
    <input type="range" id="{uid}-slider" min="0" max="{len(step_data) - 1}" value="0" step="1"/>
    <span class="step-counter" id="{uid}-counter">Step 1 / {len(step_data)}</span>
    <select id="{uid}-speed" class="step-speed" title="Playback speed">
{_speed_options_html(speed)}
    </select>
  </div>
</div>
<div class="step-modal-overlay" id="{uid}-modal">
  <div class="step-modal">
    <button class="step-modal-close" id="{uid}-modal-close">&times;</button>
    <div class="step-modal-body" id="{uid}-modal-body"></div>
  </div>
</div>

<script>
(function(){{
  var STEPS={steps_json};
  var cur=0, playing=false, timer=null;
  var directed={"true" if is_directed else "false"};
  var COL_DEFAULT='{node_color}';
  var COL_CURRENT='{trav_current}';
  var COL_FRONTIER='{trav_frontier}';
  var COL_UNVISITED='{trav_unvisited}';
  var COL_TREE_EDGE='{trav_tree_edge}';
  var COL_EDGE='{th.edge_color}';
  var COL_ACC='{hl_accent}',COL_ACC_SOFT='{hl_soft}',COL_MUTED='{hl_muted}';

  var svg=document.getElementById('{uid}-svg');
  var codeEl=document.getElementById('{uid}-codetext');
  var codeWrap=document.getElementById('{uid}-codewrap');
  var noteEl=document.getElementById('{uid}-notetext');
  var narrEl=document.getElementById('{uid}-narration');
  var codeMore=document.getElementById('{uid}-codemore');
  var badgeEl=document.getElementById('{uid}-badge');
  var deltaEl=document.getElementById('{uid}-delta');
  var slider=document.getElementById('{uid}-slider');
  var counter=document.getElementById('{uid}-counter');
  var prevBtn=document.getElementById('{uid}-prev');
  var playBtn=document.getElementById('{uid}-play');
  var nextBtn=document.getElementById('{uid}-next');
  var replayBtn=document.getElementById('{uid}-replay');
  var speedSel=document.getElementById('{uid}-speed');
  var timeline=document.getElementById('{uid}-timeline');
  var tlSegs=timeline.querySelectorAll('.tl-seg');
  var modalOverlay=document.getElementById('{uid}-modal');
  var modalBody=document.getElementById('{uid}-modal-body');
  var modalClose=document.getElementById('{uid}-modal-close');
  var notches=document.getElementById('{uid}-notches');
  var notchEls=notches.querySelectorAll('.step-notch');
  var overlay=document.getElementById('{uid}-overlay');
  var slItems=overlay.querySelectorAll('.sl-item');
  var wrapper=document.getElementById('{uid}');
  var tooltip=document.getElementById('{uid}-tooltip');
  var svgWrap=svg.parentElement;

  // Render group brackets in the timeline
  var GROUPS={json.dumps(group_brackets)};
  (function(){{
    if(!GROUPS.length) return;
    for(var g=0;g<GROUPS.length;g++){{
      var gi=GROUPS[g];
      var first=tlSegs[gi.start], last=tlSegs[gi.end];
      if(!first||!last) continue;
      var br=document.createElement('div');
      br.className='tl-bracket';
      br.setAttribute('data-group',gi.id);
      br.style.background=gi.color;
      timeline.appendChild(br);
      var posUpdate=function(bracket,s,e){{
        return function(){{
          var fs=tlSegs[s], fe=tlSegs[e];
          if(!fs||!fe) return;
          var tl=timeline.getBoundingClientRect();
          var fl=fs.getBoundingClientRect();
          var fr=fe.getBoundingClientRect();
          bracket.style.left=(fl.left-tl.left)+'px';
          bracket.style.width=(fr.right-fl.left)+'px';
        }};
      }}(br,gi.start,gi.end);
      posUpdate();
      window.addEventListener('resize',posUpdate);
    }}
  }})();

  // Progressive layout: per-step position frames, animated between steps
  var LAYOUTS={json.dumps(layout_frames)};
  var VIEWBOXES={json.dumps(frame_viewboxes)};
  var NODE_R={node_size:.2f}, ARROW_SZ={max(4, edge_width * 2.5):.2f};
  var curFrame=(STEPS[0].lf!==undefined)?STEPS[0].lf:-1;
  var curPos={{}}, posEls={{}}, edgeEls=[], animId=null;
  var reduceMotion=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(LAYOUTS.length){{
    svg.querySelectorAll('.step-node,.step-order,.step-label').forEach(function(el){{
      var l=el.getAttribute('data-label');
      (posEls[l]=posEls[l]||[]).push(el);
    }});
    svg.querySelectorAll('.step-edge,.step-edge-hit').forEach(function(el){{
      edgeEls.push({{el:el,s:el.getAttribute('data-src'),t:el.getAttribute('data-tgt')}});
    }});
    var f0=LAYOUTS[curFrame];
    for(var l0 in f0) curPos[l0]=[f0[l0][0],f0[l0][1]];
    var curVB=VIEWBOXES[curFrame].slice();
  }}
  function setViewBox(v){{
    svg.setAttribute('viewBox',v[0].toFixed(1)+' '+v[1].toFixed(1)+' '+v[2].toFixed(1)+' '+v[3].toFixed(1));
  }}
  function setPositions(P){{
    for(var l in posEls){{
      var p=P[l];
      if(!p) continue;
      var els=posEls[l];
      for(var i=0;i<els.length;i++){{
        var el=els[i];
        if(el.classList.contains('step-node')){{
          el.setAttribute('cx',p[0].toFixed(1));
          el.setAttribute('cy',p[1].toFixed(1));
        }} else {{
          var dy=el.classList.contains('step-order')?4:NODE_R+12;
          el.setAttribute('x',p[0].toFixed(1));
          el.setAttribute('y',(p[1]+dy).toFixed(1));
        }}
      }}
    }}
    for(var i=0;i<edgeEls.length;i++){{
      var e=edgeEls[i], a=P[e.s], b=P[e.t];
      if(!a||!b) continue;
      if(e.el.tagName==='polygon'){{
        var dx=b[0]-a[0], dy=b[1]-a[1];
        var ln=Math.sqrt(dx*dx+dy*dy)||1, ux=dx/ln, uy=dy/ln;
        var ax=b[0]-ux*(NODE_R+2), ay=b[1]-uy*(NODE_R+2);
        var px=-uy, py=ux, sz=ARROW_SZ;
        e.el.setAttribute('points',
          (ax+ux*sz).toFixed(1)+','+(ay+uy*sz).toFixed(1)+' '+
          (ax+px*sz*0.5).toFixed(1)+','+(ay+py*sz*0.5).toFixed(1)+' '+
          (ax-px*sz*0.5).toFixed(1)+','+(ay-py*sz*0.5).toFixed(1));
      }} else {{
        e.el.setAttribute('x1',a[0].toFixed(1));
        e.el.setAttribute('y1',a[1].toFixed(1));
        e.el.setAttribute('x2',b[0].toFixed(1));
        e.el.setAttribute('y2',b[1].toFixed(1));
      }}
    }}
  }}
  function animateTo(fi){{
    var target=LAYOUTS[fi];
    if(!target) return;
    curFrame=fi;
    if(animId){{ cancelAnimationFrame(animId); animId=null; }}
    var dur=reduceMotion?0:Math.min(550,getInterval()*0.6);
    var vbTo=VIEWBOXES[fi];
    if(dur<=0){{
      for(var l in target) curPos[l]=[target[l][0],target[l][1]];
      setPositions(curPos);
      curVB=vbTo.slice();
      setViewBox(curVB);
      return;
    }}
    var from={{}};
    for(var l in curPos) from[l]=[curPos[l][0],curPos[l][1]];
    var vbFrom=curVB.slice();
    var t0=null;
    function tick(ts){{
      if(t0===null) t0=ts;
      var u=Math.min(1,(ts-t0)/dur);
      var k=u<0.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
      for(var l in target){{
        var a=from[l]||target[l], b=target[l];
        curPos[l]=[a[0]+(b[0]-a[0])*k,a[1]+(b[1]-a[1])*k];
      }}
      setPositions(curPos);
      for(var q=0;q<4;q++) curVB[q]=vbFrom[q]+(vbTo[q]-vbFrom[q])*k;
      setViewBox(curVB);
      animId=u<1?requestAnimationFrame(tick):null;
    }}
    animId=requestAnimationFrame(tick);
  }}

  function nodeState(lbl){{
    var s=STEPS[cur],d=s.delta||{{}},tv=s.traverse||null;
    if(tv){{
      var sl=tv.step_labels||{{}};
      var ord=sl[lbl]?' (#'+sl[lbl]+')':'';
      if(lbl===tv.current) return 'Current node'+ord;
      var vis=new Set(tv.visited);
      if(vis.has(lbl)) return 'Visited'+ord;
      var fr=new Set(tv.frontier);
      if(fr.has(lbl)) return 'Frontier';
      return 'Unvisited';
    }}
    if((d.na||[]).indexOf(lbl)!==-1) return 'Added this step';
    if((d.nr||[]).indexOf(lbl)!==-1) return 'Removed this step';
    if((d.sn||[]).indexOf(lbl)!==-1) return 'Selected';
    if((d.nm||[]).indexOf(lbl)!==-1) return 'Mutated';
    if(d.ac && Object.keys(d.ac).length>0 && (d.nm||[]).length===0 && s.method.indexOf('node')!==-1) return 'Mutated';
    if(d.mv && d.mv.hasOwnProperty(lbl)){{
      var attrName=d.ac?Object.keys(d.ac)[0]:'value';
      return attrName+' = '+d.mv[lbl].toFixed(3);
    }}
    return '';
  }}
  function edgeState(src,tgt){{
    var s=STEPS[cur],d=s.delta||{{}},tv=s.traverse||null;
    if(tv){{
      if(tv.current_edge && ((tv.current_edge[0]===src&&tv.current_edge[1]===tgt)||(tv.current_edge[0]===tgt&&tv.current_edge[1]===src))) return 'Current edge';
      var te=new Set();
      tv.tree_edges.forEach(function(e){{ te.add(e[0]+'\\0'+e[1]); te.add(e[1]+'\\0'+e[0]); }});
      if(te.has(src+'\\0'+tgt)) return 'Tree edge';
      return '';
    }}
    var ea=new Set();
    (d.ea||[]).forEach(function(e){{ ea.add(e[0]+'\\0'+e[1]); ea.add(e[1]+'\\0'+e[0]); }});
    if(ea.has(src+'\\0'+tgt)) return 'Added this step';
    var er=new Set();
    (d.er||[]).forEach(function(e){{ er.add(e[0]+'\\0'+e[1]); er.add(e[1]+'\\0'+e[0]); }});
    if(er.has(src+'\\0'+tgt)) return 'Removed this step';
    var se=new Set();
    (d.se||[]).forEach(function(e){{ se.add(e[0]+'\\0'+e[1]); se.add(e[1]+'\\0'+e[0]); }});
    if(se.has(src+'\\0'+tgt)) return 'Selected';
    var em=new Set();
    (d.em||[]).forEach(function(e){{ em.add(e[0]+'\\0'+e[1]); em.add(e[1]+'\\0'+e[0]); }});
    if(em.has(src+'\\0'+tgt)) return 'Mutated';
    if(d.ac && Object.keys(d.ac).length>0 && (d.em||[]).length===0 && s.method.indexOf('edge')!==-1) return 'Mutated';
    return '';
  }}
  function showTooltip(html,e){{
    tooltip.innerHTML=html;
    tooltip.classList.add('visible');
    positionTooltip(e);
  }}
  function positionTooltip(e){{
    var r=svgWrap.getBoundingClientRect();
    var x=e.clientX-r.left+12;
    var y=e.clientY-r.top-10;
    if(x+tooltip.offsetWidth>r.width-8) x=e.clientX-r.left-tooltip.offsetWidth-8;
    if(y<4) y=4;
    tooltip.style.left=x+'px';
    tooltip.style.top=y+'px';
  }}
  function hideTooltip(){{
    tooltip.classList.remove('visible');
  }}
  svg.addEventListener('mouseover',function(e){{
    var node=e.target.closest('.step-node');
    if(node){{
      var lbl=node.getAttribute('data-label');
      if(parseFloat(node.getAttribute('opacity'))<0.01) return;
      var st=nodeState(lbl);
      var h='<div class="tt-label">'+lbl+'</div>';
      if(st) h+='<div class="tt-state">'+st+'</div>';
      showTooltip(h,e);
      return;
    }}
    var edge=e.target.closest('.step-edge,.step-edge-hit');
    if(edge){{
      var src=edge.getAttribute('data-src'), tgt=edge.getAttribute('data-tgt');
      var visEdge=svg.querySelector('.step-edge[data-src="'+src+'"][data-tgt="'+tgt+'"]');
      if(visEdge && parseFloat(visEdge.getAttribute('opacity'))<0.01) return;
      var st=edgeState(src,tgt);
      var h='<div class="tt-label">'+src+' \\u2192 '+tgt+'</div>';
      if(st) h+='<div class="tt-state">'+st+'</div>';
      showTooltip(h,e);
      return;
    }}
    hideTooltip();
  }});
  svg.addEventListener('mousemove',function(e){{
    if(tooltip.classList.contains('visible')) positionTooltip(e);
  }});
  svg.addEventListener('mouseout',function(e){{
    if(!svg.contains(e.relatedTarget)) hideTooltip();
  }});
  svg.addEventListener('mouseleave',hideTooltip);

  function isTruncated(el){{
    return el.scrollHeight>el.clientHeight+1;
  }}
  function openModal(text,asCode){{
    if(asCode){{
      modalBody.innerHTML='<pre style="margin:0;white-space:pre-wrap;word-break:break-all;font-size:13px;line-height:1.6">'+hlPy(text)+'</pre>';
    }} else {{
      modalBody.textContent=text;
    }}
    modalOverlay.classList.add('open');
  }}
  function closeModal(){{
    modalOverlay.classList.remove('open');
  }}
  var moreLink=document.createElement('span');
  moreLink.className='step-more-link';
  moreLink.textContent='\\u2026more';
  moreLink._text='';
  moreLink.addEventListener('click',function(){{ openModal(moreLink._text); }});

  function addMoreLink(el,text){{
    if(moreLink.parentNode) moreLink.parentNode.removeChild(moreLink);
    el.classList.remove('truncated');
    el.onclick=null;
    setTimeout(function(){{
      if(isTruncated(el)){{
        el.classList.add('truncated');
        moreLink._text=text;
        el.parentNode.insertBefore(moreLink,el.nextSibling);
        el.onclick=function(){{ openModal(text); }};
      }}
    }},0);
  }}
  var _codeRaw='';
  function checkCodeOverflow(){{
    setTimeout(function(){{
      if(codeWrap.scrollWidth>codeWrap.clientWidth+1){{
        codeMore.style.display='inline';
      }} else {{
        codeMore.style.display='none';
      }}
    }},0);
  }}
  codeMore.addEventListener('click',function(){{ openModal(_codeRaw,true); }});
  modalClose.addEventListener('click',closeModal);
  modalOverlay.addEventListener('click',function(e){{
    if(e.target===modalOverlay) closeModal();
  }});
  document.addEventListener('keydown',function(e){{
    if(e.key==='Escape') closeModal();
  }});

  var GLOW_ADD='url(#{uid}-glow-add)';
  var GLOW_RM='url(#{uid}-glow-rm)';
  var GLOW_MUT='url(#{uid}-glow-mut)';
  var GLOW_SEL='url(#{uid}-glow-sel)';
  var GLOW_ADD_E='url(#{uid}-glow-add-e)',GLOW_RM_E='url(#{uid}-glow-rm-e)',GLOW_SEL_E='url(#{uid}-glow-sel-e)';
  var COL_ADD='#22c55e';
  var COL_RM='#ef4444';
  var COL_MUT='#f59e0b';
  var COL_SEL='{sel_color}';

  // Categorical palette for node groups (communities, generations, components)
  var GROUP_COLS=['#2563eb','#f59e0b','#10b981','#ef4444','#8b5cf6','#ec4899','#14b8a6','#a16207','#64748b','#84cc16'];
  function groupColor(k){{ return GROUP_COLS[((k%GROUP_COLS.length)+GROUP_COLS.length)%GROUP_COLS.length]; }}

  function metricColor(t){{
    {metric_js}
  }}

  function hlPy(c){{
    var h=c.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return h.replace(/("(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.)*')|(#[^\\n]*)|(\\b(?:import|from|as|def|class|return|if|elif|else|for|in|while|with|try|except|raise|pass|break|continue|and|or|not|is|lambda|yield|global|nonlocal|assert|del)\\b)|(\\b(?:True|False|None)\\b)|(\\b\\d+\\.?\\d*\\b)|([a-zA-Z_]\\w*(?=\\s*\\())/g,
      function(m,s,c,k,b,n,f){{
        if(s) return '<span class="str">'+m+'</span>';
        if(c) return '<span class="cmt">'+m+'</span>';
        if(k) return '<span class="kw">'+m+'</span>';
        if(b) return '<span class="bool">'+m+'</span>';
        if(n) return '<span class="num">'+m+'</span>';
        if(f) return '<span class="fn">'+m+'</span>';
        return m;
      }});
  }}

  // The panel below the drawing, for step `s`
  function fillPanel(s){{
    var tv=s.traverse||null;
    var d=s.delta||{{}};
    badgeEl.textContent=tv?s.method.toUpperCase()+' ('+tv.step_num+'/'+tv.total+')':s.kind;
    if(s.kind==='annotate'){{
      codeWrap.style.display='none';
      noteEl.style.display='block';
      noteEl.textContent=s.narration;
      narrEl.style.display='none';
      codeMore.style.display='none';
    }} else {{
      codeWrap.style.display='';
      noteEl.style.display='none';
      narrEl.style.display='';
      _codeRaw=s.code;
      codeEl.innerHTML=hlPy(s.code);
      narrEl.textContent=s.narration;
    }}

    // Delta detail chips
    var dh='';
    if(!tv && d){{
      if(d.na && d.na.length){{
        var lbl=d.na.length===1?d.na[0]:d.na.length+' nodes';
        dh+='<span class="delta-chip delta-add">+ '+lbl+'</span> ';
      }}
      if(d.ea && d.ea.length){{
        var n=d.ea.length;
        var lbl=n===1?d.ea[0][0]+' \\u2192 '+d.ea[0][1]:n+' edges';
        dh+='<span class="delta-chip delta-add">+ '+lbl+'</span> ';
      }}
      if(d.nr && d.nr.length){{
        var lbl=d.nr.length===1?d.nr[0]:d.nr.length+' nodes';
        dh+='<span class="delta-chip delta-rm">&minus; '+lbl+'</span> ';
      }}
      if(d.er && d.er.length){{
        var n=d.er.length;
        var lbl=n===1?d.er[0][0]+' \\u2192 '+d.er[0][1]:n+' edges';
        dh+='<span class="delta-chip delta-rm">&minus; '+lbl+'</span> ';
      }}
      if(d.ac){{
        for(var k in d.ac){{
          var v=d.ac[k];
          var vs=typeof v==='string'?v:JSON.stringify(v);
          if(vs.length>30) vs=vs.substring(0,27)+'\\u2026';
          dh+='<span class="delta-chip delta-attr">'+k+' = '+vs+'</span> ';
        }}
      }}
      if(d.sn && d.sn.length){{
        var lbl=d.sn.length<=4?d.sn.join(', '):d.sn.length+' nodes';
        dh+='<span class="delta-chip delta-sel">\\u25c9 '+lbl+'</span> ';
      }}
      if(d.se && d.se.length){{
        var n=d.se.length;
        var lbl=n<=2?d.se.map(function(e){{return e[0]+' \\u2192 '+e[1];}}).join(', '):n+' edges';
        dh+='<span class="delta-chip delta-sel">\\u25c9 '+lbl+'</span> ';
      }}
    }}
    deltaEl.innerHTML=dh;
  }}
  function showPanel(s){{
    fillPanel(s);
    if(s.kind==='annotate'){{
      addMoreLink(noteEl,s.narration);
    }} else {{
      checkCodeOverflow();
      addMoreLink(narrEl,s.narration);
    }}
  }}

  // The panel is as tall as the tallest step needs (at the width its text wraps to), so that
  // nothing is cut off and the controls don't move as the steps change
  var codePanel=document.getElementById('{uid}-code');
  var fitWidth=-1;
  function fitPanel(){{
    if(!codePanel.clientWidth) return;
    if(moreLink.parentNode) moreLink.parentNode.removeChild(moreLink);
    codePanel.style.height='auto';
    var tallest=0;
    STEPS.forEach(function(s){{
      fillPanel(s);
      var el=s.kind==='annotate'?noteEl:narrEl;
      if(isTruncated(el)) el.parentNode.insertBefore(moreLink,el.nextSibling);
      tallest=Math.max(tallest,codePanel.offsetHeight);
      if(moreLink.parentNode) moreLink.parentNode.removeChild(moreLink);
    }});
    codePanel.style.height=tallest+'px';
    showPanel(STEPS[cur]);
  }}

  function applyStep(idx){{
    var s=STEPS[idx];
    if(LAYOUTS.length && s.lf!==undefined && s.lf!==curFrame) animateTo(s.lf);
    var vn=new Set(s.nodes);
    var ve=new Set();
    s.edges.forEach(function(e){{
      ve.add(e[0]+'\\0'+e[1]);
      if(!directed) ve.add(e[1]+'\\0'+e[0]);
    }});

    var tv=s.traverse||null;
    var tvVisited=tv?new Set(tv.visited):null;
    var tvFrontier=tv?new Set(tv.frontier):null;
    var tvTreeEdges=new Set();
    if(tv){{
      tv.tree_edges.forEach(function(e){{
        tvTreeEdges.add(e[0]+'\\0'+e[1]);
        if(!directed) tvTreeEdges.add(e[1]+'\\0'+e[0]);
      }});
    }}

    // Delta sets (for non-traverse steps)
    var d=s.delta||{{}};
    var addedN=new Set(d.na||[]);
    var removedN=new Set(d.nr||[]);
    var addedE=new Set();
    (d.ea||[]).forEach(function(e){{
      addedE.add(e[0]+'\\0'+e[1]);
      if(!directed) addedE.add(e[1]+'\\0'+e[0]);
    }});
    var removedE=new Set();
    (d.er||[]).forEach(function(e){{
      removedE.add(e[0]+'\\0'+e[1]);
      if(!directed) removedE.add(e[1]+'\\0'+e[0]);
    }});
    var hasMut=d.ac && Object.keys(d.ac).length>0;
    var mutN=new Set(d.nm||[]);
    var mutE=new Set();
    (d.em||[]).forEach(function(e){{
      mutE.add(e[0]+'\\0'+e[1]);
      if(!directed) mutE.add(e[1]+'\\0'+e[0]);
    }});
    var selN=new Set(d.sn||[]);
    var selE=new Set();
    (d.se||[]).forEach(function(e){{
      selE.add(e[0]+'\\0'+e[1]);
      if(!directed) selE.add(e[1]+'\\0'+e[0]);
    }});
    var hasSel=selN.size>0||selE.size>0;

    // What highlight= picks out (final frame only): accent and soft nodes and edges
    var hl=s.hl||null;
    function edgeSet(list){{
      var out=new Set();
      (list||[]).forEach(function(e){{
        out.add(e[0]+'\\0'+e[1]);
        if(!directed) out.add(e[1]+'\\0'+e[0]);
      }});
      return out;
    }}
    var hlAN=new Set(hl?hl.an:[]),hlSN=new Set(hl?hl.sn:[]);
    var hlAE=edgeSet(hl?hl.ae:null),hlSE=edgeSet(hl?hl.se:null);

    // Metric values — compute normalized [0,1] range
    var mv=d.mv||null;
    var mvMin=0,mvMax=1;
    if(mv){{
      var vals=Object.values(mv);
      mvMin=Math.min.apply(null,vals);
      mvMax=Math.max.apply(null,vals);
      if(mvMax===mvMin) mvMax=mvMin+1;
    }}

    // Node groups (e.g. communities): categorical colors
    var ng=(d.ng && Object.keys(d.ng).length)?d.ng:null;

    // Edge metric values (e.g. edge betweenness) — same [0,1] normalization
    var emv=null;
    var emvMin=0,emvMax=1;
    if(d.emv && d.emv.length){{
      emv=new Map();
      d.emv.forEach(function(e){{
        emv.set(e[0]+'\\0'+e[1],e[2]);
        if(!directed) emv.set(e[1]+'\\0'+e[0],e[2]);
      }});
      var evals=d.emv.map(function(e){{return e[2];}});
      // Scaled from zero, so the smallest value (such as the least flow) still shows
      emvMin=Math.min(0,Math.min.apply(null,evals));
      emvMax=Math.max.apply(null,evals);
      if(emvMax===emvMin) emvMax=emvMin+1;
    }}

    // Nodes
    var circles=svg.querySelectorAll('.step-node');
    for(var i=0;i<circles.length;i++){{
      var el=circles[i];
      var lbl=el.getAttribute('data-label');
      el.removeAttribute('filter');
      el.setAttribute('stroke','none');
      if(removedN.has(lbl)){{
        el.setAttribute('opacity','0.35');
        el.setAttribute('fill',COL_RM);
        el.setAttribute('filter',GLOW_RM);
        el.setAttribute('r','{node_size:.1f}');
      }} else if(!vn.has(lbl)){{
        el.setAttribute('opacity','0');
        el.setAttribute('fill',COL_DEFAULT);
        el.setAttribute('r','{node_size:.1f}');
      }} else if(tv){{
        el.setAttribute('opacity','1');
        if(lbl===tv.current){{
          el.setAttribute('fill',COL_CURRENT);
          el.setAttribute('r','{node_size * 1.4:.1f}');
        }} else if(tvVisited.has(lbl)){{
          el.setAttribute('fill',COL_DEFAULT);
          el.setAttribute('r','{node_size:.1f}');
        }} else if(tvFrontier.has(lbl)){{
          el.setAttribute('fill',COL_FRONTIER);
          el.setAttribute('r','{node_size:.1f}');
        }} else {{
          el.setAttribute('fill',COL_UNVISITED);
          el.setAttribute('opacity','0.4');
          el.setAttribute('r','{node_size:.1f}');
        }}
      }} else if(hl){{
        el.setAttribute('opacity','1');
        el.setAttribute('r','{node_size:.1f}');
        if(hlAN.has(lbl)){{
          el.setAttribute('fill',COL_ACC);
          el.setAttribute('r','{node_size * 1.15:.1f}');
        }} else if(hlSN.has(lbl)){{
          el.setAttribute('fill',COL_ACC_SOFT);
          el.setAttribute('stroke',COL_ACC);
        }} else {{
          el.setAttribute('fill',COL_MUTED);
        }}
      }} else if(addedN.has(lbl)){{
        el.setAttribute('opacity','1');
        el.setAttribute('fill',COL_ADD);
        el.setAttribute('filter',GLOW_ADD);
        el.setAttribute('r','{node_size:.1f}');
      }} else if(selN.has(lbl)){{
        el.setAttribute('opacity','1');
        el.setAttribute('fill',COL_SEL);
        el.setAttribute('filter',GLOW_SEL);
        el.setAttribute('r','{node_size * 1.15:.1f}');
      }} else if(hasSel && selN.size>0){{
        el.setAttribute('opacity','0.35');
        el.setAttribute('fill',COL_DEFAULT);
        el.setAttribute('r','{node_size:.1f}');
      }} else if(mutN.has(lbl)){{
        el.setAttribute('opacity','1');
        el.setAttribute('fill',COL_DEFAULT);
        el.setAttribute('stroke',COL_MUT);
        el.setAttribute('r','{node_size:.1f}');
      }} else if(hasMut && mutN.size===0 && s.method.indexOf('node')!==-1){{
        el.setAttribute('opacity','1');
        el.setAttribute('fill',COL_DEFAULT);
        el.setAttribute('stroke',COL_MUT);
        el.setAttribute('r','{node_size:.1f}');
      }} else if(ng && ng.hasOwnProperty(lbl)){{
        el.setAttribute('opacity','1');
        el.setAttribute('fill',groupColor(ng[lbl]));
        el.setAttribute('r','{node_size:.1f}');
      }} else if(ng){{
        el.setAttribute('opacity','0.3');
        el.setAttribute('fill',COL_DEFAULT);
        el.setAttribute('r','{node_size:.1f}');
      }} else if(mv && mv.hasOwnProperty(lbl)){{
        var t=(mv[lbl]-mvMin)/(mvMax-mvMin);
        el.setAttribute('opacity','1');
        el.setAttribute('fill',metricColor(t));
        el.setAttribute('r',''+(({node_size:.1f})*(0.8+0.5*t)).toFixed(1));
      }} else if(mv){{
        el.setAttribute('opacity','0.3');
        el.setAttribute('fill',COL_DEFAULT);
        el.setAttribute('r','{node_size:.1f}');
      }} else {{
        el.setAttribute('opacity','1');
        el.setAttribute('fill',COL_DEFAULT);
        el.setAttribute('r','{node_size:.1f}');
      }}
    }}

    // Labels
    var labels=svg.querySelectorAll('.step-label');
    for(var i=0;i<labels.length;i++){{
      var el=labels[i];
      var lbl=el.getAttribute('data-label');
      if(removedN.has(lbl)){{
        el.setAttribute('opacity','0.4');
        el.setAttribute('fill',COL_RM);
      }} else if(!vn.has(lbl)){{
        el.setAttribute('opacity','0');
        el.setAttribute('fill','{th.label_color}');
      }} else if(tv && !tvVisited.has(lbl) && !tvFrontier.has(lbl) && lbl!==tv.current){{
        el.setAttribute('opacity','0.3');
        el.setAttribute('fill','{th.label_color}');
      }} else if(hl){{
        el.setAttribute('opacity',hlAN.has(lbl)||hlSN.has(lbl)?'1':'0.45');
        el.setAttribute('fill','{th.label_color}');
      }} else if(addedN.has(lbl)){{
        el.setAttribute('opacity','1');
        el.setAttribute('fill',COL_ADD);
      }} else if(selN.has(lbl)){{
        el.setAttribute('opacity','1');
        el.setAttribute('fill',COL_SEL);
      }} else if(hasSel && selN.size>0){{
        el.setAttribute('opacity','0.3');
        el.setAttribute('fill','{th.label_color}');
      }} else if(mv && mv.hasOwnProperty(lbl)){{
        el.setAttribute('opacity','1');
        el.setAttribute('fill','{th.label_color}');
      }} else if(mv){{
        el.setAttribute('opacity','0.3');
        el.setAttribute('fill','{th.label_color}');
      }} else {{
        el.setAttribute('opacity','1');
        el.setAttribute('fill','{th.label_color}');
      }}
    }}

    // Step-order numbers (traversal only)
    var orders=svg.querySelectorAll('.step-order');
    var sl=tv?tv.step_labels:{{}};
    for(var i=0;i<orders.length;i++){{
      var el=orders[i];
      var lbl=el.getAttribute('data-label');
      if(tv && sl[lbl]){{
        el.textContent=sl[lbl];
        el.setAttribute('opacity', lbl===tv.current?'1':'0.85');
      }} else {{
        el.textContent='';
        el.setAttribute('opacity','0');
      }}
    }}

    // Edges
    var edges=svg.querySelectorAll('.step-edge');
    for(var i=0;i<edges.length;i++){{
      var el=edges[i];
      var key=el.getAttribute('data-src')+'\\0'+el.getAttribute('data-tgt');
      el.removeAttribute('filter');
      if(el.tagName!=='polygon') el.setAttribute('stroke-opacity','0.4');
      if(removedE.has(key)){{
        el.setAttribute('opacity','0.4');
        el.setAttribute('stroke',COL_RM);
        el.setAttribute('stroke-width','{edge_width * 1.5:.1f}');
        el.setAttribute('stroke-dasharray','4 3');
        el.setAttribute('filter',GLOW_RM_E);
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_RM);
      }} else if(!ve.has(key)){{
        el.setAttribute('opacity','0');
        el.setAttribute('stroke',COL_EDGE);
        el.setAttribute('stroke-width','{edge_width:.1f}');
        el.removeAttribute('stroke-dasharray');
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_EDGE);
      }} else if(tv){{
        el.removeAttribute('stroke-dasharray');
        if(tv.current_edge && (key===tv.current_edge[0]+'\\0'+tv.current_edge[1]||(!directed && key===tv.current_edge[1]+'\\0'+tv.current_edge[0]))){{
          el.setAttribute('opacity','1');
          el.setAttribute('stroke',COL_CURRENT);
          el.setAttribute('stroke-width','{edge_width * 3:.1f}');
          if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_CURRENT);
        }} else if(tvTreeEdges.has(key)){{
          el.setAttribute('opacity','0.7');
          el.setAttribute('stroke',COL_TREE_EDGE);
          el.setAttribute('stroke-width','{edge_width * 2:.1f}');
          if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_TREE_EDGE);
        }} else {{
          el.setAttribute('opacity','0.15');
          el.setAttribute('stroke',COL_EDGE);
          el.setAttribute('stroke-width','{edge_width:.1f}');
          if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_EDGE);
        }}
      }} else if(hl){{
        el.removeAttribute('stroke-dasharray');
        var lvl=hlAE.has(key)?2:(hlSE.has(key)?1:0);
        var hc=lvl?COL_ACC:COL_EDGE;
        el.setAttribute('opacity',['0.2','0.55','1'][lvl]);
        if(lvl===2&&el.tagName!=='polygon') el.setAttribute('stroke-opacity','0.9');
        el.setAttribute('stroke',hc);
        var hw=[{edge_width:.1f},{edge_width * 1.6:.1f},{edge_width * 2.5:.1f}];
        el.setAttribute('stroke-width',''+hw[lvl]);
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',hc);
      }} else if(addedE.has(key)){{
        el.setAttribute('opacity','1');
        el.setAttribute('stroke',COL_ADD);
        el.setAttribute('stroke-width','{edge_width * 2:.1f}');
        el.removeAttribute('stroke-dasharray');
        el.setAttribute('filter',GLOW_ADD_E);
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_ADD);
      }} else if(selE.has(key)){{
        el.setAttribute('opacity','1');
        el.setAttribute('stroke',COL_SEL);
        el.setAttribute('stroke-width','{edge_width * 2.5:.1f}');
        el.removeAttribute('stroke-dasharray');
        el.setAttribute('filter',GLOW_SEL_E);
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_SEL);
      }} else if(hasSel){{
        var dimEdge=true;
        if(selN.size>0 && selE.size===0){{
          var eSrc=el.getAttribute('data-src');
          var eTgt=el.getAttribute('data-tgt');
          if(selN.has(eSrc)&&selN.has(eTgt)) dimEdge=false;
        }}
        if(dimEdge){{
          el.setAttribute('opacity','0.15');
          el.setAttribute('stroke',COL_EDGE);
          el.setAttribute('stroke-width','{edge_width:.1f}');
          el.removeAttribute('stroke-dasharray');
          if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_EDGE);
        }} else {{
          el.setAttribute('opacity','1');
          el.setAttribute('stroke',COL_SEL);
          el.setAttribute('stroke-width','{edge_width * 1.8:.1f}');
          el.removeAttribute('stroke-dasharray');
          if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_SEL);
        }}
      }} else if(mutE.has(key)){{
        el.setAttribute('opacity','1');
        el.setAttribute('stroke',COL_MUT);
        el.setAttribute('stroke-width','{edge_width * 2:.1f}');
        el.removeAttribute('stroke-dasharray');
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_MUT);
      }} else if(emv && emv.has(key)){{
        var et=(emv.get(key)-emvMin)/(emvMax-emvMin);
        var ec=metricColor(et);
        el.setAttribute('opacity','1');
        el.setAttribute('stroke',ec);
        el.setAttribute('stroke-width',''+(({edge_width:.1f})*(1+3*et)).toFixed(1));
        el.removeAttribute('stroke-dasharray');
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',ec);
      }} else if(emv){{
        el.setAttribute('opacity','0.3');
        el.setAttribute('stroke',COL_EDGE);
        el.setAttribute('stroke-width','{edge_width:.1f}');
        el.removeAttribute('stroke-dasharray');
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_EDGE);
      }} else if(hasMut && mutE.size===0 && s.method.indexOf('edge')!==-1){{
        el.setAttribute('opacity','1');
        el.setAttribute('stroke',COL_MUT);
        el.setAttribute('stroke-width','{edge_width * 1.8:.1f}');
        el.removeAttribute('stroke-dasharray');
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_MUT);
      }} else {{
        el.setAttribute('opacity','1');
        el.setAttribute('stroke',COL_EDGE);
        el.setAttribute('stroke-width','{edge_width:.1f}');
        el.removeAttribute('stroke-dasharray');
        if(el.hasAttribute('fill')&&el.tagName==='polygon') el.setAttribute('fill',COL_EDGE);
      }}
    }}

    showPanel(s);

    slider.value=idx;
    counter.textContent='Step '+(idx+1)+' / '+STEPS.length;
    for(var t=0;t<tlSegs.length;t++){{
      tlSegs[t].classList.toggle('active',t===idx);
    }}
    for(var t=0;t<slItems.length;t++){{
      slItems[t].classList.toggle('active',t===idx);
    }}
    for(var t=0;t<notchEls.length;t++){{
      notchEls[t].classList.toggle('active',t===idx);
    }}
    // Highlight group bracket when a sub-step is active
    var activeGroup=s.group?s.group.id:null;
    var brackets=timeline.querySelectorAll('.tl-bracket');
    for(var t=0;t<brackets.length;t++){{
      var bg=parseInt(brackets[t].getAttribute('data-group'));
      brackets[t].style.opacity=bg===activeGroup?'1':'0.4';
    }}
    // Highlight group header in step list
    var grpHeaders=overlay.querySelectorAll('.sl-group-header');
    for(var t=0;t<grpHeaders.length;t++){{
      var hg=parseInt(grpHeaders[t].getAttribute('data-group'));
      grpHeaders[t].style.opacity=hg===activeGroup?'1':'0.6';
    }}
    if(overlay.classList.contains('visible')&&slItems[idx]){{
      slItems[idx].scrollIntoView({{block:'nearest',behavior:'smooth'}});
    }}
    var atEnd=idx>=STEPS.length-1;
    replayBtn.style.display=atEnd&&!playing?'inline-flex':'none';
    nextBtn.style.display=atEnd?'none':'inline-flex';
  }}

  function stopPlay(){{
    playing=false;
    playBtn.innerHTML='{_ICON_PLAY}';
    if(timer){{ clearInterval(timer); timer=null; }}
    var atEnd=cur>=STEPS.length-1;
    replayBtn.style.display=atEnd?'inline-flex':'none';
    nextBtn.style.display=atEnd?'none':'inline-flex';
  }}

  function getInterval(){{
    return parseInt(speedSel.value)||1200;
  }}
  function startPlay(){{
    playing=true;
    playBtn.innerHTML='{_ICON_PAUSE}';
    replayBtn.style.display='none';
    nextBtn.style.display='inline-flex';
    timer=setInterval(function(){{
      if(cur<STEPS.length-1){{ cur++; applyStep(cur); }}
      else {{ stopPlay(); }}
    }}, getInterval());
  }}

  prevBtn.addEventListener('click',function(){{
    stopPlay();
    if(cur>0){{ cur--; applyStep(cur); }}
  }});
  nextBtn.addEventListener('click',function(){{
    stopPlay();
    if(cur<STEPS.length-1){{ cur++; applyStep(cur); }}
  }});
  playBtn.addEventListener('click',function(){{
    if(playing){{ stopPlay(); }}
    else {{ if(cur>=STEPS.length-1){{ cur=0; }} startPlay(); }}
  }});
  replayBtn.addEventListener('click',function(){{
    stopPlay();
    cur=0;
    applyStep(cur);
  }});
  slider.addEventListener('input',function(){{
    stopPlay();
    cur=parseInt(this.value);
    applyStep(cur);
  }});
  speedSel.addEventListener('change',function(){{
    if(playing){{ clearInterval(timer); timer=setInterval(function(){{
      if(cur<STEPS.length-1){{ cur++; applyStep(cur); }}
      else {{ stopPlay(); }}
    }}, getInterval()); }}
  }});
  timeline.addEventListener('click',function(e){{
    var seg=e.target.closest('.tl-seg');
    if(!seg) return;
    var idx=parseInt(seg.getAttribute('data-idx'));
    if(isNaN(idx)) return;
    stopPlay();
    cur=idx;
    applyStep(cur);
  }});
  var overlayTimer=null;
  function showOverlay(){{
    clearTimeout(overlayTimer);
    overlay.classList.add('visible');
    if(slItems[cur]) slItems[cur].scrollIntoView({{block:'nearest',behavior:'instant'}});
  }}
  function scheduleHide(){{
    overlayTimer=setTimeout(function(){{ overlay.classList.remove('visible'); }},300);
  }}
  notches.addEventListener('mouseenter',showOverlay);
  notches.addEventListener('mouseleave',scheduleHide);
  overlay.addEventListener('mouseenter',function(){{ clearTimeout(overlayTimer); }});
  overlay.addEventListener('mouseleave',function(){{ overlay.classList.remove('visible'); }});
  notches.addEventListener('click',function(e){{
    var n=e.target.closest('.step-notch');
    if(!n) return;
    var idx=parseInt(n.getAttribute('data-idx'));
    if(isNaN(idx)) return;
    stopPlay(); cur=idx; applyStep(cur);
  }});
  overlay.addEventListener('click',function(e){{
    var item=e.target.closest('.sl-item');
    if(!item) return;
    var idx=parseInt(item.getAttribute('data-idx'));
    if(isNaN(idx)) return;
    stopPlay(); cur=idx; applyStep(cur);
  }});

  document.addEventListener('keydown',function(e){{
    if(!document.getElementById('{uid}')) return;
    if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT') return;
    if(e.key==='ArrowRight'){{ nextBtn.click(); e.preventDefault(); }}
    else if(e.key==='ArrowLeft'){{ prevBtn.click(); e.preventDefault(); }}
    else if(e.key===' '){{ playBtn.click(); e.preventDefault(); }}
    else if(e.key==='Home'){{ stopPlay(); cur=0; applyStep(cur); e.preventDefault(); }}
    else if(e.key==='End'){{ stopPlay(); cur=STEPS.length-1; applyStep(cur); e.preventDefault(); }}
    else if(e.key==='l'||e.key==='L'){{ overlay.classList.toggle('visible'); e.preventDefault(); }}
  }});

  applyStep(0);
  new ResizeObserver(function(){{
    if(codePanel.clientWidth!==fitWidth){{ fitWidth=codePanel.clientWidth; fitPanel(); }}
  }}).observe(wrapper);
  {"startPlay();" if auto_play else ""}
}})();
</script>
</div>"""

    return colors.finish(svg_str, html, output)
