"""Static SVG graph renderer for publication-quality small graphs.

Produces clean SVG with optional hover interactivity, visually consistent
with Cosmograph's aesthetic: filled circles, thin semi-transparent edges,
minimal chrome.
"""

from __future__ import annotations

import html as html_mod
import json
import math
import random
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

# Default node color when no color_by mapping
DEFAULT_NODE_COLOR = "#4e79a7"

# Font
FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
)


# ---------------------------------------------------------------------------
# PlotSVG result object
# ---------------------------------------------------------------------------


class PlotSVG:
    """SVG visualization result.

    Renders inline in Jupyter via ``_repr_html_`` (with hover tooltips) or
    ``_repr_svg_`` (pure SVG). Can be saved to file with ``.save()``.
    """

    def __init__(self, svg_str: str, html_str: str) -> None:
        self._svg = svg_str
        self._html = html_str

    @property
    def svg(self) -> str:
        return self._svg

    def _repr_svg_(self) -> str:
        return self._svg

    def _repr_html_(self) -> str:
        return self._html

    def save(self, path: str) -> None:
        p = Path(path)
        content = self._html if p.suffix == ".html" else self._svg
        p.write_text(content, encoding="utf-8")

    def __repr__(self) -> str:
        return f"PlotSVG({len(self._svg)} chars)"


# ---------------------------------------------------------------------------
# Layout algorithms
# ---------------------------------------------------------------------------


def _circular_layout(
    node_ids: List[str],
    width: float,
    height: float,
    padding: float = 60,
) -> Dict[str, Tuple[float, float]]:
    n = len(node_ids)
    if n == 0:
        return {}
    if n == 1:
        return {node_ids[0]: (width / 2, height / 2)}

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


def _fruchterman_reingold_python(
    n: int,
    edges: List[Tuple[int, int]],
    iterations: int = 300,
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
    pos_x = [random.uniform(-spread, spread) for _ in range(n)]
    pos_y = [random.uniform(-spread, spread) for _ in range(n)]

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    temp = area * 0.15
    cool = temp / (iterations + 1)
    # Weak gravity pulls toward center, preventing drift while allowing separation
    gravity = 0.3

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

    Returns (distance, t_param, side) where t_param is the projection
    parameter [0,1] along the segment and side is +1/-1 indicating which
    side of the segment the point lies on.
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

    Returns None if the straight line is clear (no routing needed),
    or a list of waypoints [src, wp1, wp2, ..., dst] that avoids all
    intermediate node circles.
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
    """Convert waypoints to a smooth SVG cubic bezier path string,
    trimmed to stop at node circle edges."""
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
) -> Dict[str, Tuple[float, float]]:
    """Push overlapping nodes apart until no circles overlap.

    Uses iterative pairwise displacement with grid acceleration.
    Each pass finds overlapping pairs and pushes them apart by half
    the overlap distance each, along the line connecting their centers.
    """
    ids = list(positions.keys())
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
                                angle = random.uniform(0, 2 * math.pi)
                                dx, dy = math.cos(angle), math.sin(angle)
                                dist = 0.01
                            overlap = min_dist - dist
                            push = overlap * 0.55
                            ux, uy = dx / dist, dy / dist
                            px[nid] += ux * push
                            py[nid] += uy * push
                            px[other] -= ux * push
                            py[other] -= uy * push
                            moved = True

        if not moved:
            break

    return {nid: (px[nid], py[nid]) for nid in ids}


def _force_directed_layout(
    node_ids: List[str],
    edges: List[Tuple[str, str]],
    width: float,
    height: float,
    iterations: int = 300,
) -> Dict[str, Tuple[float, float]]:
    """Compute force-directed layout, using Rust engine with Python fallback."""
    if len(node_ids) == 0:
        return {}

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    indexed_edges = []
    for s, t in edges:
        if s in id_to_idx and t in id_to_idx:
            indexed_edges.append((id_to_idx[s], id_to_idx[t]))

    raw = None
    try:
        from edgecase._core import force_directed_layout_rust

        raw = force_directed_layout_rust(len(node_ids), indexed_edges, min(iterations, 500))
    except Exception:
        pass

    if not raw:
        raw = _fruchterman_reingold_python(len(node_ids), indexed_edges, iterations)

    if not raw:
        return _circular_layout(node_ids, width, height)

    xs = [pos[0] for pos in raw.values()]
    ys = [pos[1] for pos in raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x != min_x else 1.0
    range_y = max_y - min_y if max_y != min_y else 1.0

    pad = 60.0
    usable_w = width - 2 * pad
    usable_h = height - 2 * pad
    scale = min(usable_w / range_x, usable_h / range_y)

    cx_raw = (min_x + max_x) / 2
    cy_raw = (min_y + max_y) / 2
    cx_svg = width / 2
    cy_svg = height / 2

    positions = {}
    for idx, pos in raw.items():
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
# Multi-edge fan-out
# ---------------------------------------------------------------------------


def _compute_edge_fans(
    edges: List[Tuple[str, str]],
) -> Tuple[List[float], List[float]]:
    """Compute bow values and loop sizes for fanning out parallel edges.

    For each edge, returns:
    - bow: curve amount (0.0 = straight). For edges between the same unordered
      pair of nodes, bows are assigned so the curves visually fan out.
    - loop_size: scaling multiplier for self-loop arcs (1.0 = default).

    The key insight: for edge A→B, the perpendicular is in one direction;
    for B→A it's reversed. So to place them on OPPOSITE visual sides, both
    need the SAME bow value (the perpendicular reversal provides the mirror).
    To place them on the same visual side, they need opposite bow values.

    We assign visual offsets along a canonical perpendicular, then convert:
    canonical direction = src < tgt. For canonical-direction edges, bow = v.
    For reverse-direction edges, bow = -v.
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
            is_canonical = (src <= tgt)
            bows[idx] = visual_offset if is_canonical else -visual_offset

    return bows, loop_sizes


# ---------------------------------------------------------------------------
# SVG drawing primitives
# ---------------------------------------------------------------------------


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
            f'stroke-linecap="round" data-edge="{edge_id}" data-attrs=\'{attrs_json}\'/>'
            f'<polygon points="{ex:.1f},{ey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
            f'fill="{color}"/>'
            f"</g>"
        )
    else:
        parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex_line:.1f}" y2="{ey_line:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round" data-edge="{edge_id}" data-attrs=\'{attrs_json}\'/>'
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
                f'stroke-width="{width}" stroke-linecap="round" '
                f"data-edge=\"{edge_id}\" data-attrs='{attrs_json}'/>"
                f'<polygon points="{ex:.1f},{ey:.1f} {p1x:.1f},{p1y:.1f} {p2x:.1f},{p2y:.1f}" '
                f'fill="{color}"/>'
                f"</g>"
            )
        else:
            parts.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" stroke-opacity="{opacity}" '
                f'stroke-linecap="round" data-edge="{edge_id}" data-attrs=\'{attrs_json}\'/>'
            )
    else:
        parts.append(
            f'<path d="{path_d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round" data-edge="{edge_id}" data-attrs=\'{attrs_json}\'/>'
        )

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
            f'stroke-linecap="round" data-edge="{edge_id}" data-attrs=\'{attrs_json}\'/>'
            f"{arrow_svg}</g>"
        )
    else:
        parts.append(
            f'<path id="{path_id}" d="M {sx:.1f} {sy:.1f} Q {mx:.1f} {my:.1f} {ex:.1f} {ey:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round" data-edge="{edge_id}" data-attrs=\'{attrs_json}\'/>'
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
            f'stroke-linecap="round" data-edge="{edge_id}" data-attrs=\'{attrs_json}\'/>'
            f"{arrow_svg}</g>"
        )
    else:
        parts.append(
            f'<path id="{path_id}" d="{path_d}" '
            f'fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-linecap="round" data-edge="{edge_id}" data-attrs=\'{attrs_json}\'/>'
        )
    return "\n".join(parts)


def _svg_node(
    x: float,
    y: float,
    r: float,
    fill: str,
    node_id: str,
    attrs_json: str,
    border_opacity: float = 0.3,
) -> str:
    border = _darken(fill, 0.3)
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
        f'fill="{fill}" stroke="{border}" stroke-width="1.5" '
        f'stroke-opacity="{border_opacity}" '
        f"data-node=\"{html_mod.escape(str(node_id))}\" data-attrs='{attrs_json}' "
        f'style="cursor:pointer"/>'
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
# Compute viewBox to fit all nodes
# ---------------------------------------------------------------------------


def _compute_viewbox(
    positions: Dict[str, Tuple[float, float]],
    radii: Dict[str, float],
    labels: Optional[Dict[str, str]] = None,
    label_space: float = 20.0,
    padding: float = 30.0,
) -> Tuple[float, float, float, float]:
    """Return (min_x, min_y, width, height) enclosing all nodes and labels."""
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

    w = max_x - min_x
    h = max_y - min_y
    return (min_x, min_y, max(w, 1), max(h, 1))


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_plot(
    graph,
    layout: str = "force",
    color_by: Optional[str] = None,
    edge_color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    label_by: str = "id",
    edge_label_by: Optional[str] = None,
    curved_edges: bool = False,
    theme: str = "light",
    palette: str = "tableau",
    width: int = 600,
    height: int = 400,
    node_size: float = 16.0,
    edge_width: float = 1.2,
    edge_opacity: float = 0.4,
    cross_community_color: Optional[str] = None,
    node_overlap: Union[str, float] = "remove",
    edge_routing: Union[str, float] = "around",
    show_labels: bool = True,
    output: Optional[str] = None,
) -> "PlotSVG":
    """Render a graph as static SVG. Returns a PlotSVG object."""

    th = THEMES.get(theme, LIGHT)

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
        result = PlotSVG(svg, svg)
        if output:
            result.save(output)
        return result

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

    # Edge data (supports multi-edges between same pair)
    is_directed = graph.is_directed()
    edges: List[Tuple[str, str]] = []
    edge_attrs_list: List[Dict] = []
    seen_pairs: set = set()
    for nid in node_ids:
        try:
            neighbors = graph.neighbors(nid)
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
                else:
                    for eid in eids:
                        edges.append((str(nid), str(nb)))
                        try:
                            e_obj = graph.get_edge(eid)
                            edge_attrs_list.append(
                                dict(e_obj.attributes) if e_obj and e_obj.attributes else {}
                            )
                        except Exception:
                            edge_attrs_list.append({})

    # Convert node_ids to strings for consistency
    str_ids = [str(nid) for nid in node_ids]

    # --- Layout ---
    if layout == "circular":
        positions = _circular_layout(str_ids, float(width), float(height))
    elif layout == "random":
        positions = _random_layout(str_ids, float(width), float(height))
    else:
        n = len(str_ids)
        fr_iters = 800 if n > 300 else (600 if n > 150 else 400)
        positions = _force_directed_layout(
            str_ids,
            edges,
            float(width),
            float(height),
            iterations=fr_iters,
        )

    # --- Aesthetic mappings ---

    # Color mapping
    color_map = {}
    node_community = {}
    if color_by:
        cat_values = []
        for nid in str_ids:
            val = node_data.get(int(nid) if nid.isdigit() else nid, {}).get(color_by)
            cat_values.append(val)
            node_community[nid] = val
        palette_map = _build_color_map(cat_values, palette)
        for nid, val in zip(str_ids, cat_values):
            color_map[nid] = palette_map.get(val, DEFAULT_NODE_COLOR)
    else:
        for nid in str_ids:
            color_map[nid] = DEFAULT_NODE_COLOR

    # Edge color mapping (community-aware)
    edge_community = {}
    if edge_color_by:
        ecb_attr = edge_color_by
        for nid in str_ids:
            val = node_data.get(int(nid) if nid.isdigit() else nid, {}).get(ecb_attr)
            edge_community[nid] = val
        if not any(v is not None for v in edge_community.values()):
            edge_community = {}
        ecb_vals = [v for v in edge_community.values() if v is not None]
        ecb_palette = _build_color_map(ecb_vals, palette) if ecb_vals else {}
        ecb_node_colors = {
            nid: ecb_palette.get(edge_community.get(nid), th.edge_color) for nid in str_ids
        }
        default_cross = "#e07a7a" if theme != "dark" else "#b45555"
        ecb_cross_color = cross_community_color or default_cross

    # Size mapping
    radii = {}
    if size_by:
        size_values = []
        for nid in str_ids:
            val = node_data.get(int(nid) if nid.isdigit() else nid, {}).get(size_by)
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
        positions = _remove_overlaps(positions, radii, padding=overlap_pad)

    # --- Build label map for viewBox computation ---
    label_map = {}
    if show_labels:
        for nid in str_ids:
            raw_nid = int(nid) if nid.isdigit() else nid
            n_attrs = node_data.get(raw_nid, {})
            if label_by == "id":
                label_map[nid] = str(nid)
            else:
                label_map[nid] = str(n_attrs.get(label_by, nid))

    # --- Compute viewBox ---
    vb = _compute_viewbox(
        positions,
        radii,
        labels=label_map if show_labels else None,
        label_space=20 if show_labels else 0,
    )
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

    # Edges first (painter's algorithm)
    for i, (src, tgt) in enumerate(edges):
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        r_src = radii.get(src, node_size)
        r_dst = radii.get(tgt, node_size)

        # Edge color
        e_color = th.edge_color
        if edge_color_by and edge_community:
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

        e_attrs = edge_attrs_list[i] if i < len(edge_attrs_list) else {}
        attrs_j = (
            json.dumps(e_attrs, default=str)
            .replace("&", "&amp;")
            .replace("'", "&#39;")
            .replace("<", "&lt;")
        )
        eid = f"{src}-{tgt}-{i}"

        bow = edge_bows[i]

        # Self-loop
        if src == tgt:
            svg_parts.append(
                _svg_edge_loop(
                    x1, y1, r_src, e_color, edge_width,
                    edge_opacity, is_directed, eid, attrs_j,
                    loop_size=edge_loop_sizes[i],
                )
            )
            continue

        route = edge_routes.get((src, tgt))
        if curved_edges:
            svg_parts.append(
                _svg_edge_curved(
                    x1, y1, x2, y2, r_src, r_dst,
                    e_color, edge_width, edge_opacity,
                    is_directed, eid, attrs_j,
                )
            )
        elif abs(bow) > 1e-6:
            svg_parts.append(
                _svg_edge_curved(
                    x1, y1, x2, y2, r_src, r_dst,
                    e_color, edge_width, edge_opacity,
                    is_directed, eid, attrs_j, bow=bow,
                )
            )
        elif route is not None:
            svg_parts.append(
                _svg_edge_routed(
                    route, r_src, r_dst,
                    e_color, edge_width, edge_opacity,
                    is_directed, eid, attrs_j,
                )
            )
        else:
            svg_parts.append(
                _svg_edge_straight(
                    x1, y1, x2, y2, r_src, r_dst,
                    e_color, edge_width, edge_opacity,
                    is_directed, eid, attrs_j,
                )
            )

    # Edge labels (stretch goal)
    if edge_label_by:
        for i, (src, tgt) in enumerate(edges):
            if src not in positions or tgt not in positions:
                continue
            e_attrs = edge_attrs_list[i] if i < len(edge_attrs_list) else {}
            label_val = e_attrs.get(edge_label_by)
            if label_val is None:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[tgt]
            eid = f"{src}-{tgt}-{i}"

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
                        bow_val = 0.2
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
            else:
                # Midpoint label for straight edges
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                # Offset perpendicular to edge
                dx, dy = x2 - x1, y2 - y1
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-6:
                    px, py = -dy / dist, dx / dist
                    mx += px * 8
                    my += py * 8
                svg_parts.append(
                    f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" '
                    f'font-family="{FONT_FAMILY}" font-size="10" '
                    f'fill="{th.label_color}" fill-opacity="0.7" '
                    f'pointer-events="none">{html_mod.escape(str(label_val))}</text>'
                )

    # Nodes on top
    for nid in str_ids:
        if nid not in positions:
            continue
        x, y = positions[nid]
        r = radii.get(nid, node_size)
        fill = color_map.get(nid, DEFAULT_NODE_COLOR)

        raw_nid = int(nid) if nid.isdigit() else nid
        n_attrs = node_data.get(raw_nid, {})
        attrs_j = (
            json.dumps({"id": nid, **n_attrs}, default=str)
            .replace("&", "&amp;")
            .replace("'", "&#39;")
            .replace("<", "&lt;")
        )

        svg_parts.append(_svg_node(x, y, r, fill, nid, attrs_j))

    # Labels
    if show_labels:
        font_size = max(8, min(13, node_size * 0.7))
        for nid in str_ids:
            if nid not in positions:
                continue
            x, y = positions[nid]
            r = radii.get(nid, node_size)

            raw_nid = int(nid) if nid.isdigit() else nid
            n_attrs = node_data.get(raw_nid, {})
            if label_by == "id":
                text = str(nid)
            else:
                text = str(n_attrs.get(label_by, nid))

            svg_parts.append(_svg_label(x, y, r, text, th.label_color, font_size))

    svg_parts.append("</svg>")
    svg_str = "\n".join(svg_parts)

    # --- HTML wrapper with hover tooltips ---
    html_str = _generate_html_wrapper(svg_str, th, width, height)

    result = PlotSVG(svg_str, html_str)
    if output:
        result.save(output)
    return result


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
    return f"""<div style="position:relative;display:inline-block">
{svg_str}
<div id="{tip_id}" style="
  display:none;position:absolute;padding:6px 10px;border-radius:6px;
  font-family:{FONT_FAMILY};font-size:12px;pointer-events:none;
  background:{theme.tooltip_bg};color:{theme.tooltip_fg};
  box-shadow:0 2px 8px rgba(0,0,0,0.18);max-width:260px;z-index:999;
  line-height:1.4;
"></div>
<script>
(function(){{
  var tip=document.currentScript.parentElement.querySelector('#{tip_id}');
  var svg=document.currentScript.parentElement.querySelector('svg');
  if(!svg||!tip) return;
  function show(el,ev){{
    var raw=el.getAttribute('data-attrs');
    if(!raw) return;
    try{{var d=JSON.parse(raw)}}catch(e){{return}}
    var h='';
    for(var k in d){{
      var v=d[k];
      if(typeof v==='number'&&!Number.isInteger(v)) v=v.toFixed(4);
      h+='<div><b>'+k+'</b>: '+v+'</div>';
    }}
    if(!h) return;
    tip.innerHTML=h;tip.style.display='block';
    var r=svg.getBoundingClientRect();
    tip.style.left=(ev.clientX-r.left+12)+'px';
    tip.style.top=(ev.clientY-r.top-10)+'px';
  }}
  svg.addEventListener('mouseover',function(ev){{
    var t=ev.target;
    if(t.hasAttribute('data-node')||t.hasAttribute('data-edge')) show(t,ev);
  }});
  svg.addEventListener('mousemove',function(ev){{
    var r=svg.getBoundingClientRect();
    tip.style.left=(ev.clientX-r.left+12)+'px';
    tip.style.top=(ev.clientY-r.top-10)+'px';
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
) -> List[Tuple[str, str]]:
    """Infer tree edges from a visit order.

    For each node after the first, the tree edge connects it to its
    already-visited neighbor that appears latest in visit_order.
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
                if pos > best_pos:
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
                "visited": list(visited_set),
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
    theme: str = "light",
    width: int = 700,
    height: int = 500,
    node_size: float = 18.0,
    edge_width: float = 1.5,
    node_overlap: Union[str, float] = "remove",
    show_labels: bool = True,
    output: Optional[str] = None,
) -> "PlotSVG":
    """Render an interactive graph traversal visualization.

    Returns a PlotSVG with an HTML slider that steps through the traversal.
    """
    global _traversal_counter
    _traversal_counter += 1
    uid = f"tv{_traversal_counter}"

    th = THEMES.get(theme, LIGHT)

    # --- Theme-dependent traversal colors ---
    is_dark = theme == "dark"
    unvisited_fill = "#4b5563" if is_dark else "#d1d5db"
    frontier_fill = "#60a5fa" if is_dark else "#93c5fd"
    current_fill = "#3b82f6" if is_dark else "#2563eb"
    accent = "#60a5fa" if is_dark else "#2563eb"
    tree_edge_color = "#60a5fa" if is_dark else "#2563eb"
    current_edge_color = "#93c5fd" if is_dark else "#1d4ed8"
    nontree_visited_edge = "#64748b" if is_dark else "#cbd5e1"
    unvisited_edge_opacity = 0.35 if is_dark else 0.2

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
            neighbors = graph.neighbors(nid)
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
        tree_edges_str = _infer_tree_edges(visit_order_str, adjacency)
    else:
        tree_edges_str = [(str(s), str(t)) for s, t in tree_edges]

    # --- Build step data ---
    steps = _build_traversal_steps(visit_order_str, tree_edges_str, adjacency)

    # --- Layout ---
    if layout == "circular":
        positions = _circular_layout(str_ids, float(width), float(height))
    elif layout == "random":
        positions = _random_layout(str_ids, float(width), float(height))
    else:
        n = len(str_ids)
        fr_iters = 800 if n > 300 else (600 if n > 150 else 400)
        positions = _force_directed_layout(
            str_ids,
            edges,
            float(width),
            float(height),
            iterations=fr_iters,
        )

    # --- Color map (for visited state) ---
    color_map: Dict[str, str] = {}
    for nid in str_ids:
        color_map[nid] = DEFAULT_NODE_COLOR

    radii = {nid: node_size for nid in str_ids}

    # --- Overlap removal (force/random layout) ---
    if node_overlap != "allow" and layout not in ("circular",) and len(str_ids) >= 2:
        overlap_pad = float(node_overlap) if isinstance(node_overlap, (int, float)) else 2.0
        positions = _remove_overlaps(positions, radii, padding=overlap_pad)

    # --- Compute viewBox ---
    label_map = {}
    if show_labels:
        for nid in str_ids:
            label_map[nid] = str(nid)
    vb = _compute_viewbox(
        positions,
        radii,
        labels=label_map if show_labels else None,
        label_space=20 if show_labels else 0,
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
            f'style="transition: fill 0.3s ease, opacity 0.3s ease; cursor:pointer"/>'
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
                f"{html_mod.escape(str(nid))}</text>"
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
        is_dark,
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

    result = PlotSVG(svg_str, html_str)
    if output:
        result.save(output)
    return result


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
    is_dark: bool,
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

    btn_bg = "#374151" if is_dark else "#e5e7eb"
    btn_fg = "#e5e7eb" if is_dark else "#374151"
    ctrl_accent = accent

    return f"""<div style="display:inline-block;max-width:{svg_width}px">
{svg_str}
<div style="display:flex;align-items:center;gap:8px;padding:8px 4px;
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
<div style="font-size:11px;color:{th.label_color};opacity:0.5;padding:0 4px 4px;">
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
  var UEO={unvisited_edge_opacity};

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

    Each call to a ``trav_*`` or ``select_*`` method is delegated to the
    underlying graph, then the resulting selection (node IDs or edge IDs)
    is captured as a step.  The recorded chain can be rendered as an
    interactive visualization with :meth:`plot`.
    """

    _EDGE_OPS = frozenset({"trav_out_edge", "trav_in_edge", "trav_reverse_edge"})
    _NODE_OPS = frozenset({
        "trav_out", "trav_in", "trav_both",
        "trav_out_until", "trav_in_until",
        "select_nodes", "select_nodes_by_id", "select_nodes_by_degree",
        "select_nodes_by_centrality", "select_nodes_in_neighborhood",
        "select_last_nodes_created", "invert_node_selection",
    })
    _EDGE_SELECT_OPS = frozenset({
        "select_edges", "select_edges_by_id", "select_edges_by_node",
        "select_last_edges_created", "invert_edge_selection",
    })

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
        self._steps.append({
            "op": op_name,
            "op_display": f"{op_name}({op_args_str})" if op_args_str else f"{op_name}()",
            "selection_type": sel_type,
            "ids": ids,
        })

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

    def trav_out_until(self, conditions, max_steps=30, exclude_unmatched=True, add_to_selection=False):
        return self._delegate("trav_out_until", "...", conditions, max_steps=max_steps,
                              exclude_unmatched=exclude_unmatched, add_to_selection=add_to_selection)

    def trav_in_until(self, conditions, max_steps=30, exclude_unmatched=True, add_to_selection=False):
        return self._delegate("trav_in_until", "...", conditions, max_steps=max_steps,
                              exclude_unmatched=exclude_unmatched, add_to_selection=add_to_selection)

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
        theme: str = "light",
        width: int = 700,
        height: int = 500,
        node_size: float = 18.0,
        edge_width: float = 1.5,
        node_overlap: Union[str, float] = "remove",
        show_labels: bool = True,
        output: Optional[str] = None,
    ) -> PlotSVG:
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
        )


# ---------------------------------------------------------------------------
# render_traversal_chain — interactive selection-chain visualization
# ---------------------------------------------------------------------------


def render_traversal_chain(
    graph,
    chain_steps: List[Dict[str, Any]],
    layout: str = "force",
    theme: str = "light",
    width: int = 700,
    height: int = 500,
    node_size: float = 18.0,
    edge_width: float = 1.5,
    node_overlap: Union[str, float] = "remove",
    show_labels: bool = True,
    output: Optional[str] = None,
) -> PlotSVG:
    """Render an interactive traversal-chain visualization.

    Each step in *chain_steps* is a dict with keys ``op``, ``op_display``,
    ``selection_type`` (``"nodes"`` or ``"edges"``), and ``ids`` (list of
    integer IDs).  The visualization shows the full graph and lets the user
    step through the chain, highlighting the selected nodes or edges at each
    step.
    """
    global _chain_counter
    _chain_counter += 1
    uid = f"tc{_chain_counter}"

    th = THEMES.get(theme, LIGHT)
    is_dark = theme == "dark"

    # --- Theme colors ---
    unvisited_fill = "#4b5563" if is_dark else "#d1d5db"
    accent = "#60a5fa" if is_dark else "#2563eb"
    accent_light = "#93c5fd" if is_dark else "#bfdbfe"
    prev_accent = "#818cf8" if is_dark else "#a5b4fc"
    selected_edge_color = "#60a5fa" if is_dark else "#2563eb"
    related_edge_color = "#60a5fa" if is_dark else "#93c5fd"
    unvisited_edge_opacity = 0.25 if is_dark else 0.12

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
            neighbors = graph.neighbors(nid)
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
            eids = graph.find_edge_ids(int(src_s) if src_s.isdigit() else src_s,
                                       int(tgt_s) if tgt_s.isdigit() else tgt_s)
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
            js_steps.append({
                "sel": "nodes",
                "nodes": sel_node_strs,
                "edges": [],
                "related_edges": list(dict.fromkeys(related_edge_keys)),
                "op": step["op_display"],
            })
        else:
            sel_edge_visuals = []
            connected_nodes = set()
            for eid in step["ids"]:
                vis = edge_id_to_visual.get(eid)
                if vis:
                    s_src = _sanitize_svg_id(vis[0])
                    s_tgt = _sanitize_svg_id(vis[1])
                    sel_edge_visuals.append(f"{s_src}-{s_tgt}")
                    connected_nodes.add(vis[0])
                    connected_nodes.add(vis[1])
            js_steps.append({
                "sel": "edges",
                "nodes": [],
                "edges": list(dict.fromkeys(sel_edge_visuals)),
                "related_nodes": list(connected_nodes),
                "op": step["op_display"],
            })

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
        positions = _circular_layout(str_ids, float(width), float(height))
    elif layout == "random":
        positions = _random_layout(str_ids, float(width), float(height))
    else:
        fr_iters = 800 if n > 300 else (600 if n > 150 else 400)
        positions = _force_directed_layout(
            str_ids, edges, float(width), float(height), iterations=fr_iters,
        )

    radii = {nid: node_size for nid in str_ids}

    # --- Overlap removal ---
    if node_overlap != "allow" and layout not in ("circular",) and n >= 2:
        overlap_pad = float(node_overlap) if isinstance(node_overlap, (int, float)) else 2.0
        positions = _remove_overlaps(positions, radii, padding=overlap_pad)

    # --- ViewBox ---
    label_map = {}
    if show_labels:
        for nid in str_ids:
            lbl = node_data.get(nid, {}).get("label", nid)
            label_map[nid] = str(lbl) if lbl else nid
    vb = _compute_viewbox(
        positions, radii,
        labels=label_map if show_labels else None,
        label_space=20 if show_labels else 0,
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
        f'<defs>'
        f'<filter id="{uid}-glow"><feGaussianBlur stdDeviation="3" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter></defs>'
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
        is_dark=is_dark,
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

    result = PlotSVG(svg_str, html_str)
    if output:
        result.save(output)
    return result


def _generate_chain_html_wrapper(
    svg_str: str,
    th: Theme,
    uid: str,
    js_steps: List[Dict[str, Any]],
    chain_steps: List[Dict[str, Any]],
    str_ids: List[str],
    node_safe_ids: Dict[str, str],
    edge_keys: List[str],
    is_dark: bool,
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

    btn_bg = "#374151" if is_dark else "#e5e7eb"
    btn_fg = "#e5e7eb" if is_dark else "#374151"

    # Build step log HTML
    step_log_lines = []
    for i, step in enumerate(chain_steps):
        sel_type = step["selection_type"]
        if sel_type == "nodes":
            badge_bg = "#dbeafe" if not is_dark else "#1e3a5f"
            badge_fg = "#1e40af" if not is_dark else "#93c5fd"
            badge_label = "nodes"
        else:
            badge_bg = "#fef3c7" if not is_dark else "#78350f"
            badge_fg = "#92400e" if not is_dark else "#fcd34d"
            badge_label = "edges"
        ids_display = ", ".join(str(x) for x in step["ids"][:8])
        if len(step["ids"]) > 8:
            ids_display += f", ... ({len(step['ids'])} total)"
        step_log_lines.append(
            f'<div id="{uid}-log-{i}" style="display:flex;gap:8px;align-items:baseline;'
            f'padding:3px 8px;border-radius:4px;opacity:0.5;'
            f'transition:opacity 0.2s ease,background 0.2s ease">'
            f'<span style="min-width:18px;font-size:12px">{i + 1}.</span>'
            f'<span style="background:{badge_bg};color:{badge_fg};'
            f'padding:1px 6px;border-radius:4px;font-size:11px;white-space:nowrap">'
            f'{badge_label}</span>'
            f'<span style="font-size:12px;font-family:monospace">'
            f'{html_mod.escape(step["op_display"])}</span>'
            f'<span style="color:{th.label_color};opacity:0.5;margin-left:auto;'
            f'font-size:11px;font-family:monospace;white-space:nowrap">'
            f'→ {{{html_mod.escape(ids_display)}}}</span>'
            f'</div>'
        )
    step_log_html = "\n".join(step_log_lines)

    # Step pill bar (SVG overlay approach is complex, use HTML pills below SVG)
    pill_items = []
    for i, step in enumerate(chain_steps):
        short_op = step["op"].replace("select_nodes_by_id", "select").replace("select_edges_by_id", "select")
        if len(short_op) > 16:
            short_op = short_op[:14] + ".."
        pill_items.append(
            f'<span id="{uid}-pill-{i}" style="'
            f'padding:3px 10px;border-radius:12px;font-size:11px;cursor:pointer;'
            f'white-space:nowrap;border:1px solid {btn_bg};color:{th.label_color};'
            f'transition:background 0.2s ease,color 0.2s ease,border-color 0.2s ease"'
            f' data-step="{i}">{html_mod.escape(short_op)}</span>'
        )
    pills_html = "\n".join(pill_items)

    return f"""<div style="display:inline-block;max-width:{svg_width}px;font-family:{FONT_FAMILY}">
{svg_str}
<div style="display:flex;gap:6px;padding:6px 4px;flex-wrap:wrap;align-items:center">
{pills_html}
</div>
<div style="display:flex;align-items:center;gap:8px;padding:8px 4px;font-size:13px;color:{th.label_color}">
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
  <summary style="padding:8px 4px;font-size:12px;color:{th.label_color};cursor:pointer;
    opacity:0.6;list-style:none;user-select:none">
    &#9662; Traversal steps ({n_steps} operations)
  </summary>
  <div id="{uid}-log" style="padding:4px;line-height:1.9">
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
    unvisitedEdgeOp:{unvisited_edge_opacity},
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
    if(lg){{ lg.style.opacity='1'; lg.style.background='{accent_light}22'; }}

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
