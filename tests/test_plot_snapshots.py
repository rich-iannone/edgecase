"""Snapshot tests for `Graph.plot()`.

Each case renders a small graph and compares the HTML (which holds the SVG) with a file in
`tests/snapshots/plot/`, byte for byte. They pin down everything between the node positions and
the markup: sizes, colors, scales, highlighting, legends, edge shapes, routing, labels, tooltips,
the themes and lean output. Any change to the output shows up here as a diff, so a refactor of
the renderer can prove that it changes nothing.

The positions are fixed, so that the force layout (whose floating point can differ in the last
digit between platforms) has no part in the comparisons: the graphs take them from
`positions.json` as a `"pos"` attribute, or from the circular and random layouts.

After a change to the output that is meant, rewrite the snapshots with

    EDGECASE_UPDATE_SNAPSHOTS=1 python -m pytest tests/test_plot_snapshots.py

and review the diff of `tests/snapshots/plot/` before committing it.
"""

import difflib
import json
import math
import os
import re
from pathlib import Path

import edgecase as ec
import edgecase.svg_renderer as svg_renderer
import pytest

SNAPSHOTS = Path(__file__).parent / "snapshots" / "plot"
UPDATE = os.environ.get("EDGECASE_UPDATE_SNAPSHOTS") == "1"
POSITIONS = json.loads((SNAPSHOTS / "positions.json").read_text(encoding="utf-8"))


def _with_positions(g, name):
    g.set_node_attrs("pos", {int(k): tuple(v) for k, v in POSITIONS[name].items()})
    return g


def _florentine():
    return _with_positions(ec.florentine_families(), "florentine_families")


def _karate():
    g = _with_positions(ec.karate_club(), "karate_club")
    # A score with a sign, for the diverging scale
    g.set_node_attrs("score", {nid: (nid * 37 % 100) / 53 - 1 for nid in g.get_node_ids()})
    return g


# A small multigraph with what the datasets lack: parallel and reciprocal edges, self-loops,
# weights, edge attributes with gaps, a node in the way of an edge, node attributes with gaps, a
# NaN, a vector attribute and an isolated node
_MULTI_NODES = {
    "a": {"pos": (0, 0), "group": "x", "score": 0.5, "emb": [0.123456789, 2.0], "size": 3.5},
    "b": {"pos": (2, 1.5), "group": "x", "score": 1.2345678, "size": 1.0},
    "c": {"pos": (4, 0.2), "group": "x", "score": -0.75, "size": 2.25},
    "d": {"pos": (1.5, -1.0), "group": "y", "score": float("nan")},
    "e": {"pos": (3, -1.75), "group": "y", "score": 2.0, "size": 4.0},
    "f": {"pos": (4.5, -1.2), "group": "y", "score": 0.0, "size": 0.5},
    "g": {"pos": (0.3, -2.4), "score": 1.0},
    "h": {"pos": (5.5, 1.4), "group": "z", "score": 1.5, "size": 2.0},
    "i": {"pos": (4.9, -1.6), "group": "z"},
}
_MULTI_EDGES = [
    ("a", "b", 1.0, {"kind": "road", "flow": 3}),
    ("a", "b", 2.0, {"kind": "rail", "flow": 1}),
    ("a", "b", 0.5, {"kind": "road", "flow": 2}),
    ("b", "a", 1.5, {"kind": "air", "flow": 5}),
    ("c", "c", 1.0, {"kind": "road", "flow": 1}),
    ("c", "c", 2.0, {"kind": "rail"}),
    # Passes by e, so it's routed around it
    ("g", "f", 3.0, {"kind": "road", "flow": 4}),
    ("a", "d", None, {"flow": float("nan")}),
    ("d", "e", 1.0, {"kind": "air", "flow": 2}),
    ("e", "d", 2.0, {"kind": "rail", "flow": 1}),
    ("e", "f", 2.5, {"kind": "road", "flow": 6}),
    ("f", "c", 1.0, {}),
    ("c", "h", 0.25, {"kind": "rail", "flow": 0}),
    ("h", "b", 1.0, {"kind": "road", "flow": 2}),
    ("b", "c", 4.0, {"kind": "air", "flow": 3}),
]


def _multi(directed=True):
    g = ec.Graph(directed=directed)
    g.add_nodes(list(_MULTI_NODES))
    for label, attrs in _MULTI_NODES.items():
        for key, value in attrs.items():
            g.set_node_attrs(key, {label: value})
    for src, tgt, weight, attrs in _MULTI_EDGES:
        g.add_edge(src, tgt, weight=weight, **attrs)
    return g


_PATH = ["0", "1", "2", "3"]

# Each case: (graph builder, plot() arguments, lean). Lean cases lower the lean threshold so
# that a small graph is drawn as a large one is
CASES = {
    "empty": (ec.Graph, {}, False),
    "florentine": (_florentine, {"layout": "pos"}, False),
    "florentine_light": (_florentine, {"layout": "pos", "theme": "light"}, False),
    "florentine_dark": (_florentine, {"layout": "pos", "theme": "dark"}, False),
    "florentine_circular": (_florentine, {"layout": "circular", "theme": "light"}, False),
    "florentine_random": (_florentine, {"layout": "random", "seed": 3, "theme": "light"}, False),
    "florentine_raw": (
        _florentine,
        {
            "layout": "pos",
            "node_overlap": "allow",
            "edge_routing": "straight",
            "show_labels": False,
        },
        False,
    ),
    "florentine_degree": (
        _florentine,
        {"layout": "pos", "color_by": "degree", "size_by": "degree", "label_by": "label"},
        False,
    ),
    "florentine_rel": (
        _florentine,
        {"layout": "pos", "edge_color_by": "rel", "edge_label_by": "rel", "theme": "light"},
        False,
    ),
    "florentine_curved": (
        _florentine,
        {"layout": "pos", "curved_edges": True, "edge_label_by": "rel", "theme": "dark"},
        False,
    ),
    "karate_faction": (
        _karate,
        {"layout": "pos", "color_by": "faction", "show_labels": False},
        False,
    ),
    "karate_faction_edges": (
        _karate,
        {
            "layout": "pos",
            "color_by": "faction",
            "edge_color_by": "faction",
            "cross_community_color": "#7c3aed",
            "palette": "vibrant",
            "theme": "light",
        },
        False,
    ),
    "karate_role_by_faction": (
        _karate,
        {"layout": "pos", "color_by": "role", "edge_color_by": "faction", "theme": "dark"},
        False,
    ),
    "karate_diverging": (
        _karate,
        {
            "layout": "pos",
            "color_by": "score",
            "color_scale": "diverging",
            "color_midpoint": 0.2,
            "size_by": "score",
        },
        False,
    ),
    "karate_sequential_small": (
        _karate,
        {
            "layout": "pos",
            "color_by": "score",
            "color_scale": "sequential",
            "legend": False,
            "width": 400,
            "height": 300,
            "node_size": 6,
            "edge_width": 2.0,
            "edge_opacity": 0.6,
            "theme": "light",
        },
        False,
    ),
    "karate_path": (_karate, {"layout": "pos", "highlight": _PATH, "theme": "dark"}, False),
    "karate_two_levels": (
        _karate,
        {
            "layout": "pos",
            "color_by": "faction",
            "highlight": {"accent": ["0", "33"], "soft": ["1", "2", "32"]},
            "highlight_edges": {"soft": [("0", "1")]},
        },
        False,
    ),
    "karate_highlight_edges": (
        _karate,
        {"layout": "pos", "highlight_edges": [("0", "1"), ("1", "2")], "theme": "light"},
        False,
    ),
    "karate_lean": (
        _karate,
        {"layout": "pos", "color_by": "faction", "highlight": ["0", "33"]},
        True,
    ),
    "multi": (_multi, {"layout": "pos", "theme": "light"}, False),
    "multi_clearance": (
        _multi,
        {"layout": "pos", "node_overlap": 30, "edge_routing": 25, "theme": "light"},
        False,
    ),
    "multi_undirected": (lambda: _multi(directed=False), {"layout": "pos"}, False),
    "multi_weights": (
        _multi,
        {
            "layout": "pos",
            "color_by": "group",
            "edge_width_by": "flow",
            "edge_color_by": "weight",
            "theme": "light",
        },
        False,
    ),
    "multi_kinds": (
        _multi,
        {
            "layout": "pos",
            "color_by": "score",
            "edge_color_by": "kind",
            "edge_label_by": "kind",
            "label_by": "group",
            "theme": "dark",
        },
        False,
    ),
    "multi_sized": (_multi, {"layout": "pos", "color_by": "size", "size_by": "size"}, False),
    "multi_curved": (
        _multi,
        {"layout": "pos", "curved_edges": True, "edge_label_by": "flow", "theme": "light"},
        False,
    ),
    "multi_lean": (
        _multi,
        {"layout": "pos", "edge_width_by": "flow", "highlight_edges": [("a", "b")]},
        True,
    ),
    "multi_lean_curved": (
        _multi,
        {"layout": "pos", "curved_edges": True, "edge_label_by": "kind", "theme": "light"},
        True,
    ),
}


def _render(case, monkeypatch):
    build, kwargs, lean = CASES[case]
    if lean:
        monkeypatch.setattr(svg_renderer, "LEAN_MIN", 3)
    return build().plot(**kwargs)


def _normalized(markup):
    # The tooltip's element ID counts the renderings made so far
    return re.sub(r"ec-tooltip-[0-9]+", "ec-tooltip-N", markup)


@pytest.mark.parametrize("case", sorted(CASES))
def test_plot_matches_snapshot(case, monkeypatch):
    html = _normalized(_render(case, monkeypatch)._repr_html_())
    path = SNAPSHOTS / f"{case}.html"
    if UPDATE:
        path.write_text(html, encoding="utf-8", newline="\n")
        return
    if not path.exists():
        pytest.fail(f"{case!r} has no snapshot yet; set EDGECASE_UPDATE_SNAPSHOTS=1 to write it")
    expected = path.read_text(encoding="utf-8")
    if html != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                html.splitlines(keepends=True),
                f"snapshots/plot/{path.name}",
                "now",
                n=1,
            )
        )
        pytest.fail(
            f"plot() output for {case!r} differs from its snapshot (set "
            f"EDGECASE_UPDATE_SNAPSHOTS=1 to rewrite it if the change is meant):\n{diff[:6000]}"
        )


@pytest.mark.parametrize("case", sorted(CASES))
def test_svg_is_the_html_drawing(case, monkeypatch):
    # The snapshots hold the HTML; the SVG is the drawing inside it, with the theme's
    # definitions for the automatic theme
    p = _render(case, monkeypatch)
    svg = p.svg.replace('<svg class="ec-theme" ', "<svg ", 1)
    svg = re.sub(r'<style class="ec-theme-vars">.*?</style>', "", svg, count=1)
    assert svg in p._repr_html_()


@pytest.mark.parametrize("case", ["karate_two_levels", "multi_kinds", "multi_lean_curved"])
def test_one_scene_draws_every_theme(case, monkeypatch):
    # The scene doesn't depend on the theme: drawn in each one, it's what plot() draws in it
    build, kwargs, lean = CASES[case]
    if lean:
        monkeypatch.setattr(svg_renderer, "LEAN_MIN", 3)
    options = {k: v for k, v in kwargs.items() if k != "theme"}
    scene = svg_renderer.build_scene(build(), **options)
    for theme in (None, "light", "dark"):
        colors = svg_renderer._Colors(theme)
        drawn = colors.finish(*svg_renderer.emit_svg(scene, colors), None)
        p = build().plot(**options, theme=theme)
        assert drawn.svg == p.svg
        assert _normalized(drawn._repr_html_()) == _normalized(p._repr_html_())


def test_scene_checks_its_columns():
    from edgecase._core import Scene

    scene = Scene(200, 100, colors=[("#111111", "#eeeeee")])
    nodes = (["a", "b"], [0.0, 1.0], [0.0, 1.0], [5.0, 5.0], [0, 0], [0, 0], [1.5, 1.5])
    with pytest.raises(ValueError, match="x has 1 entries for 2 nodes"):
        scene.set_nodes(["a", "b"], [0.0], *nodes[2:], [0.3, 0.3], [0, 1], ["{}", "{}"])
    with pytest.raises(ValueError, match="fill has index 1, past the 1 colors"):
        scene.set_nodes(*nodes[:4], [0, 1], *nodes[5:], [0.3, 0.3], [0, 1], ["{}", "{}"])
    scene.set_nodes(*nodes, [0.3, 0.3], [1, 0], ["{}", "{}"])
    assert scene.node_count == 2
    edge = ([0], [1], [0], [1.0], [0.6], [0.0], [1.0], [None], [0])
    with pytest.raises(ValueError, match="tgt has index 2, past the 2 nodes"):
        scene.set_edges([0], [2], *edge[2:], ["{}"])
    # Lean scenes hold the edges' tooltips in one table, and others hold them edge by edge
    with pytest.raises(ValueError, match="unless the scene is lean"):
        scene.set_edges(*edge)
    scene.set_edges(*edge, ["{}"])
    assert scene.edge_count == 1
    with pytest.raises(ValueError, match="colors has 0 entries"):
        scene.svg([], "#fff", "#000", "sans-serif")
    assert scene.svg(["#123456"], "#fff", "#000", "sans-serif").endswith("</svg>")


def test_saved_default_theme_is_the_light_one(tmp_path):
    # Saved files take the light colors of the automatic theme, so they match the light theme
    p = _florentine().plot(layout="pos")
    p.save(tmp_path / "auto.svg")
    p.save(tmp_path / "auto.html")
    light = _florentine().plot(layout="pos", theme="light")
    assert (tmp_path / "auto.svg").read_text(encoding="utf-8") == light.svg
    auto_html = _normalized((tmp_path / "auto.html").read_text(encoding="utf-8"))
    assert auto_html == _normalized(light._repr_html_())


def test_every_snapshot_has_a_case():
    stale = {p.stem for p in SNAPSHOTS.glob("*.html")} - set(CASES)
    assert not stale, f"snapshots without a case: {sorted(stale)}"


def test_positions_cover_the_graphs():
    for name, build in (
        ("florentine_families", ec.florentine_families),
        ("karate_club", ec.karate_club),
    ):
        g = build()
        assert set(POSITIONS[name]) == {str(nid) for nid in g.get_node_ids()}
        assert all(math.isfinite(c) for xy in POSITIONS[name].values() for c in xy)
