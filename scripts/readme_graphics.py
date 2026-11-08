#!/usr/bin/env python3
"""Draw the README figures into `assets/readme/`, each as a light and a dark SVG.

The network pictures come straight from `Graph.plot()`, with seeds picked so that each layout
fills its canvas. The robustness and benchmark charts are drawn here in the colors of the doc
diagrams (`scripts/doc_graphics/core.py`); the benchmark chart reads the timings that
`scripts/readme_benchmarks.py` writes to `assets/readme/benchmarks.json`.

Usage:
    python scripts/readme_graphics.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import edgecase as ec

OUT = Path(__file__).resolve().parent.parent / "assets" / "readme"

FONT_SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# The doc diagram colors, on the backgrounds that `plot()` uses
THEMES = {
    "light": dict(ink="#1f2328", bg="#fafafa", accent="#c2410c", g2="#0e7490", g3="#7c3aed"),
    "dark": dict(ink="#e6edf3", bg="#0d1117", accent="#f97316", g2="#22d3ee", g3="#a78bfa"),
}


def mix(c1: str, c2: str, t: float) -> str:
    """Blend hex c1 toward c2 by t (0 = c1, 1 = c2)."""
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def write(name: str, theme: str, svg: str) -> None:
    path = OUT / f"{name}.{theme}.svg"
    path.write_text(svg)
    print(f"wrote {path.relative_to(OUT.parent.parent)}")


# --- network pictures ------------------------------------------------------


def fit_to_viewbox(svg: str, width: int) -> str:
    """Size the SVG to its cropped viewBox so no empty band is left around the drawing."""
    _, _, vw, vh = (float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split())
    height = round(width * vh / vw)
    return re.sub(r'width="\d+" height="\d+"', f'width="{width}" height="{height}"', svg, count=1)


def plot_pair(name: str, graph: ec.Graph, width: int, height: int, **kwargs) -> None:
    for theme in THEMES:
        svg = graph.plot(width=width, height=height, theme=theme, **kwargs).svg
        write(name, theme, fit_to_viewbox(svg, width))


def network_pictures() -> None:
    # Les Misérables: Leiden communities, nodes sized by betweenness
    mis = ec.les_miserables().cmty_leiden().betweenness_centrality()
    plot_pair(
        "hero",
        mis,
        880,
        530,
        color_by="cmty_leiden",
        size_by="betweenness_centrality",
        palette="Set2",
        show_labels=False,
        seed=6,
    )

    # The tour: the karate club's communities, nodes sized by PageRank
    karate = ec.karate_club().cmty_louvain().pagerank()
    plot_pair(
        "karate",
        karate,
        800,
        480,
        color_by="cmty_louvain",
        size_by="pagerank",
        label_by="label",
        seed=9,
    )

    # Brokers: the Florentine families sized by betweenness
    florentine = ec.florentine_families().betweenness_centrality()
    plot_pair(
        "florentine",
        florentine,
        640,
        400,
        size_by="betweenness_centrality",
        label_by="label",
        seed=2,
    )

    # Payment rings found with a Cypher pattern
    payments = payments_graph()
    (
        payments.select_match(
            "MATCH (a)-[x]->(b)-[y]->(c)-[z]->(a) "
            "WHERE x.weight >= 1000 AND y.weight >= 1000 AND z.weight >= 1000"
        )
        .set_node_attrs_ws(flag="ring")
        .invert_node_selection()
        .set_node_attrs_ws(flag="ok")
        .clear_selection()
    )
    plot_pair(
        "payments", payments, 600, 360, color_by="flag", label_by="label", palette="Set1", seed=38
    )


def payments_graph() -> ec.Graph:
    return ec.Graph.from_edges(
        [
            ("Dee", "Eve", 80),
            ("Eve", "Dee", 25),
            ("Ana", "Dee", 120),
            ("Dee", "Ben", 60),
            ("Ben", "Eve", 300),
            ("Gus", "Dee", 45),
            ("Eve", "Finn", 2_000),
            ("Ana", "Ben", 9_500),
            ("Ben", "Cruz", 9_400),
            ("Cruz", "Ana", 9_300),
            ("Finn", "Gus", 4_950),
            ("Gus", "Hal", 4_900),
            ("Hal", "Finn", 4_850),
        ],
        directed=True,
    )


# --- robustness chart ------------------------------------------------------


def robustness_chart() -> None:
    grid = ec.power_grid()
    upto = 0.30
    steps = math.ceil(upto * grid.count_nodes())
    series = [
        ("Random failure", "g2", "random"),
        ("Attack by degree", "accent", "degree"),
        ("Attack by betweenness", "g3", "betweenness"),
    ]
    curves = {
        strategy: grid.network_robustness(strategy=strategy, steps=steps, seed=1, return_dict=True)
        for _, _, strategy in series
    }

    width, height = 760, 320
    left, right, top, bottom = 64, 728, 24, 264

    def px(f: float) -> float:
        return left + f / upto * (right - left)

    def py(f: float) -> float:
        return bottom - f * (bottom - top)

    for theme, c in THEMES.items():
        grid_line = mix(c["ink"], c["bg"], 0.86)
        muted = mix(c["ink"], c["bg"], 0.35)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{FONT_SANS}">',
            f'<rect width="{width}" height="{height}" fill="{c["bg"]}"/>',
        ]
        for i in range(0, 7):
            x = px(i * 0.05)
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{grid_line}"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{bottom + 18}" font-size="11" fill="{muted}" '
                f'text-anchor="middle">{i * 5}%</text>'
            )
        for i in range(0, 5):
            y = py(i * 0.25)
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{grid_line}"/>'
            )
            parts.append(
                f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="{muted}" '
                f'text-anchor="end">{i * 25}%</text>'
            )
        parts.append(
            f'<text x="{(left + right) / 2}" y="{height - 10}" font-size="12" fill="{c["ink"]}" '
            f'text-anchor="middle">Share of substations removed</text>'
        )
        parts.append(
            f'<text transform="translate(16 {(top + bottom) / 2}) rotate(-90)" font-size="12" '
            f'fill="{c["ink"]}" text-anchor="middle">Share still in the largest component</text>'
        )

        # The 5% mark, where the targeted attacks have already halved the grid
        x5 = px(0.05)
        parts.append(
            f'<line x1="{x5:.1f}" y1="{top}" x2="{x5:.1f}" y2="{bottom}" stroke="{muted}" '
            f'stroke-dasharray="3 3"/>'
        )

        for name, color, strategy in series:
            points = " ".join(
                f"{px(f):.1f},{py(s):.1f}" for f, s in [(0.0, 1.0), *curves[strategy]]
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{c[color]}" stroke-width="2.25" '
                f'stroke-linejoin="round"/>'
            )

        def at(strategy: str, f: float) -> float:
            return min(curves[strategy], key=lambda p: abs(p[0] - f))[1]

        # The two targeted attacks meet at 5%, so one label serves both
        for strategy, color in (("random", "g2"), ("degree", "accent")):
            s = at(strategy, 0.05)
            parts.append(f'<circle cx="{x5:.1f}" cy="{py(s):.1f}" r="4" fill="{c[color]}"/>')
            parts.append(
                f'<text x="{x5 + 8:.1f}" y="{py(s) - 8:.1f}" font-size="12" font-weight="600" '
                f'fill="{c[color]}">{s:.0%}</text>'
            )

        # Legend in the empty upper right
        lx, ly = px(0.19), top + 26
        for k, (name, color, _) in enumerate(series):
            y = ly + k * 22
            parts.append(
                f'<line x1="{lx:.1f}" y1="{y}" x2="{lx + 24:.1f}" y2="{y}" stroke="{c[color]}" '
                f'stroke-width="2.25"/>'
            )
            parts.append(
                f'<text x="{lx + 32:.1f}" y="{y + 4}" font-size="12" fill="{c["ink"]}">{name}</text>'
            )
        parts.append("</svg>")
        write("robustness", theme, "\n".join(parts) + "\n")


# --- benchmark chart -------------------------------------------------------


def benchmark_chart() -> None:
    data = json.loads((OUT / "benchmarks.json").read_text())
    rows = data["rows"]

    width = 760
    row_h, bar_h = 34, 11
    left, right, top = 208, 700, 46
    height = top + row_h * len(rows) + 40
    lo, hi = 0.1, 300.0

    def px(v: float) -> float:
        return left + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (
            right - left
        )

    def fmt(v: float) -> str:
        return f"{v:.0f}×" if v >= 10 else f"{v:.1f}×"

    for theme, c in THEMES.items():
        grid_line = mix(c["ink"], c["bg"], 0.86)
        muted = mix(c["ink"], c["bg"], 0.35)
        other = mix(c["ink"], c["bg"], 0.55)
        bottom = top + row_h * len(rows)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{FONT_SANS}">',
            f'<rect width="{width}" height="{height}" fill="{c["bg"]}"/>',
        ]

        # Legend
        parts.append(
            f'<rect x="{left}" y="14" width="12" height="12" rx="2" fill="{c["accent"]}"/>'
        )
        parts.append(
            f'<text x="{left + 18}" y="24" font-size="12" fill="{c["ink"]}">EdgeCase</text>'
        )
        parts.append(f'<rect x="{left + 96}" y="14" width="12" height="12" rx="2" fill="{other}"/>')
        parts.append(
            f'<text x="{left + 114}" y="24" font-size="12" fill="{c["ink"]}">igraph</text>'
        )
        parts.append(
            f'<text x="{right}" y="24" font-size="12" fill="{muted}" text-anchor="end">'
            f"speed-up over NetworkX (log scale)</text>"
        )

        for v in (0.1, 1, 10, 100):
            x = px(v)
            stroke = muted if v == 1 else grid_line
            parts.append(
                f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}" stroke="{stroke}"/>'
            )
            label = "NetworkX" if v == 1 else f"{v:g}×"
            parts.append(
                f'<text x="{x:.1f}" y="{bottom + 18}" font-size="11" fill="{muted}" '
                f'text-anchor="middle">{label}</text>'
            )

        for k, row in enumerate(rows):
            y0 = top + k * row_h
            task = row["task"]
            if row["graph"].startswith("power_grid"):
                task += " (power grid)"
            parts.append(
                f'<text x="{left - 12}" y="{y0 + row_h / 2 + 4:.1f}" font-size="12" '
                f'fill="{c["ink"]}" text-anchor="end">{task}</text>'
            )
            for j, (lib, color) in enumerate((("edgecase", c["accent"]), ("igraph", other))):
                v = row["networkx"] / row[lib]
                y = y0 + 5 + j * (bar_h + 2)
                x1, x2 = sorted((px(1), px(v)))
                parts.append(
                    f'<rect x="{x1:.1f}" y="{y}" width="{max(x2 - x1, 1):.1f}" height="{bar_h}" '
                    f'rx="2" fill="{color}"/>'
                )
                tx, anchor = (x2 + 5, "start") if v >= 1 else (x1 - 5, "end")
                weight = ' font-weight="600"' if lib == "edgecase" else ""
                parts.append(
                    f'<text x="{tx:.1f}" y="{y + bar_h - 2}" font-size="11" fill="{color}"'
                    f'{weight} text-anchor="{anchor}" font-family="{FONT_MONO}">{fmt(v)}</text>'
                )
        parts.append("</svg>")
        write("benchmarks", theme, "\n".join(parts) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    network_pictures()
    robustness_chart()
    benchmark_chart()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
