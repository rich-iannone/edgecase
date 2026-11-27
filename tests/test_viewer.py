"""Tests for the viewer: `Scene.payload()`, the scene as the bytes that `static/viewer.js` draws
with WebGL, and `plot(backend="viewer")`, the HTML that holds both.

The viewer itself runs in a browser, so these tests read what it's given, and not what it draws.
"""

import base64
import gzip
import html as html_mod
import json
import re
import struct
from pathlib import Path

import edgecase as ec
import edgecase.svg_renderer as svg_renderer
import pytest
from edgecase._core import Scene

PAIRS = [("#3366cc", "#88aaee"), ("#cc3333", "#cc3333"), ("#22aa55", "#66dd99")]


def _payload(data):
    """The header and columns of a payload, each column as a tuple of its entries."""
    assert data[:4] == b"ECV1"
    (length,) = struct.unpack("<I", data[4:8])
    assert length % 4 == 0
    header = json.loads(data[8 : 8 + length])
    columns = {}
    for name, column in header["columns"].items():
        offset, count = column["offset"], column["length"]
        assert offset % 4 == 0 and offset >= 8 + length
        kind = {"f32": "f", "u32": "I"}[column["type"]]
        columns[name] = struct.unpack(f"<{count}{kind}", data[offset : offset + 4 * count])
    return header, columns


def _scene(curved=False, node_order=(0, 1, 2), edge_order=(0, 1, 2), bow=0.0):
    scene = Scene(200, 100, curved=curved, viewbox=(0.0, 0.0, 200.0, 100.0), colors=PAIRS)
    scene.set_nodes(
        ["a", "b", "c"], [20.0, 180.0, 100.0], [50.0, 50.0, 85.0], [10.0, 10.0, 5.0], [1] * 3,
        [1] * 3, [0.0] * 3, [1.0] * 3, list(node_order), ["{}"] * 3,
    )  # fmt: skip
    scene.set_edges(
        [0, 2, 2], [1, 0, 1], [0, 2, 0], [4.0, 1.0, 2.0], [0.6] * 3, [bow] * 3, [1.0] * 3,
        [None] * 3, list(edge_order), ["{}"] * 3,
    )  # fmt: skip
    return scene


def test_the_payload_holds_the_scenes_columns():
    data = _scene().payload()
    header, columns = _payload(data)
    assert (header["version"], header["nodes"], header["edges"]) == (1, 3, 3)
    assert (header["width"], header["height"], header["viewbox"]) == (200, 100, [0, 0, 200, 100])
    assert header["directed"] is False
    assert columns["x"] == (20.0, 180.0, 100.0) and columns["y"] == (50.0, 50.0, 85.0)
    assert columns["r"] == (10.0, 10.0, 5.0)
    assert columns["src"] == (0, 2, 2) and columns["tgt"] == (1, 0, 1)
    assert columns["color"] == (0, 2, 0) and columns["width"] == (4.0, 1.0, 2.0)
    # The columns whose entries are all the same are constants, and the order is left out when
    # it's the nodes' own
    assert header["constants"] == {
        "fill": 1, "border": 1, "border_width": 0.0, "border_opacity": 1.0,
        "opacity": pytest.approx(0.6), "bow": 0.0, "loop_size": 1.0,
    }  # fmt: skip
    assert set(columns) == {"x", "y", "r", "src", "tgt", "color", "width"}
    # The columns follow one another, from the end of the header to the end of the payload
    ends = sorted((c["offset"], c["offset"] + 4 * c["length"]) for c in header["columns"].values())
    assert ends[0][0] == 8 + struct.unpack("<I", data[4:8])[0] and ends[-1][1] == len(data)
    assert all(a[1] == b[0] for a, b in zip(ends, ends[1:]))
    # The colors in each theme, from 0 to 1
    assert header["colors"]["light"][0] == pytest.approx([0x33 / 255, 0x66 / 255, 0xCC / 255, 1])
    assert header["colors"]["dark"][0] == pytest.approx([0x88 / 255, 0xAA / 255, 0xEE / 255, 1])


def test_the_payload_is_in_drawing_order():
    header, columns = _payload(_scene(node_order=(2, 0, 1), edge_order=(1, 2, 0)).payload())
    # The nodes' order, which the viewer draws them in
    assert columns["order"] == (2, 0, 1)
    # The edges as they are drawn, highlighted ones last
    assert columns["src"] == (2, 2, 0) and columns["tgt"] == (0, 1, 1)
    assert columns["width"] == (1.0, 2.0, 4.0) and columns["color"] == (2, 0, 0)


def test_curved_edges_bow_out():
    header, _ = _payload(_scene(curved=True).payload())
    assert header["constants"]["bow"] == pytest.approx(0.2)
    # Parallel edges bow out further, from the curve's own
    header, _ = _payload(_scene(curved=True, bow=0.1).payload())
    assert header["constants"]["bow"] == pytest.approx(0.3)
    scene = _scene()
    scene.set_edges(
        [0, 2, 2], [1, 0, 1], [0] * 3, [1.0] * 3, [1.0] * 3, [0.0, 0.1, -0.1], [1.0] * 3,
        [None] * 3, [0, 1, 2], ["{}"] * 3,
    )  # fmt: skip
    _, columns = _payload(scene.payload())
    assert columns["bow"] == pytest.approx((0.0, 0.1, -0.1))


def _html_payload(html):
    encoded = re.search(
        r'<script type="application/octet-stream" class="ec-payload">([^<]+)<', html
    )
    return _payload(base64.b64decode(encoded.group(1)))


def _options(html):
    return json.loads(re.search(r"mount\(root, bytes\.buffer, (\{.*\})\);", html).group(1))


def test_the_viewer_backend_holds_the_viewer_and_the_payload():
    g = ec.karate_club().cmty_louvain(attr_name="c")
    html = g.plot(backend="viewer", seed=1, color_by="c")._repr_html_()
    header, columns = _html_payload(html)
    assert (header["nodes"], header["edges"]) == (34, 78) and len(columns["x"]) == 34
    # The viewer's code, and the legend over it, as SVG
    assert '<script type="module">' in html and "export function mount(" in html
    assert 'class="ec-legend-layer"' in html and 'class="ec-legend"' in html
    # Not the SVG's nodes and edges, whose tooltips are in a table of their own
    drawing = html.split('class="ec-legend"')[0]
    assert "<circle" not in drawing and "<line" not in drawing and "data-attrs" not in html
    assert 'class="ec-tips"' in html
    # Each plot's root is its own
    ids = re.findall(r'<div class="ec-viewer" id="([^"]+)"', html)
    again = g.plot(backend="viewer", seed=1, color_by="c")._repr_html_()
    assert len(ids) == 1 and ids != re.findall(r'<div class="ec-viewer" id="([^"]+)"', again)
    with pytest.raises(ValueError, match="use one of 'auto', 'svg', 'png', 'viewer'"):
        g.plot(backend="webgl")


def _tips(html):
    """The viewer's tooltips: a dict for each node, in the nodes' order, and for each edge, in the
    order drawn."""
    packed = re.search(r'<script type="application/octet-stream" class="ec-tips">([^<]+)<', html)
    tables = json.loads(gzip.decompress(base64.b64decode(packed.group(1))))

    def rows(table):
        count = max((len(c) for c in table["c"] if isinstance(c, list)), default=0)
        return [
            {k: c[i] if isinstance(c, list) else c[str(i)] for k, c in zip(table["k"], table["c"])
             if isinstance(c, list) or str(i) in c}
            for i in range(count)
        ]  # fmt: skip

    return rows(tables["nodes"]), rows(tables["edges"])


def _svg_tips(svg, kind):
    found = re.findall(rf" data-{kind}=\"[^\"]*\" data-attrs='([^']*)'", svg)
    return [json.loads(html_mod.unescape(attrs)) for attrs in found]


def test_the_viewer_has_the_svgs_tooltips(monkeypatch):
    g = ec.karate_club().cmty_louvain(attr_name="c")
    g.set_edge_attrs("kind", "tie", edge_ids=[1, 2])
    opts = {"seed": 1, "theme": "light", "edge_routing": "straight", "color_by": "c"}
    html = g.plot(backend="viewer", **opts)._repr_html_()
    nodes, edges = _tips(html)
    svg = g.plot(backend="svg", **opts).svg
    by_id = {tip["id"]: tip for tip in nodes}
    assert len(by_id) == 34 and by_id == {tip["id"]: tip for tip in _svg_tips(svg, "node")}
    in_svg = _svg_tips(svg, "edge")
    assert sorted(edges, key=lambda tip: tip["id"]) == sorted(in_svg, key=lambda tip: tip["id"])
    assert by_id["1"] == {"id": "1", "label": "0", "faction": 1, "role": "Instructor", "c": 0}
    assert {"id": 2, "edge": "1 — 3", "kind": "tie"} in edges
    # The edges' rows are in the order they're drawn, which puts highlighted edges last
    html = g.plot(backend="viewer", seed=1, highlight=[30, 34])._repr_html_()
    nodes, edges = _tips(html)
    _, columns = _html_payload(html)
    # (whose ends the tooltip names in the order stored)
    ends = [{nodes[a]["id"], nodes[b]["id"]} for a, b in zip(columns["src"], columns["tgt"])]
    assert [set(tip["edge"].split(" — ")) for tip in edges] == ends
    assert ends[-1] == {"30", "34"}
    # A lean plot's too, whose SVG keeps the edges' tooltips in a table of its own
    full = _tips(g.plot(backend="viewer", **opts)._repr_html_())
    monkeypatch.setattr(svg_renderer, "LEAN_MIN", 10)
    plot = g.plot(backend="viewer", **opts)
    assert _tips(plot._repr_html_()) == full
    assert plot.svg == g.plot(backend="svg", **opts).svg and ' data-e="' in plot.svg


def test_the_viewers_tooltips_are_text():
    g = ec.Graph().add_nodes(["a", "b"]).add_edges([("a", "b")])
    g.set_node_attrs("note", {"a": "</script><img src=x onerror=alert(1)>"})
    html = g.plot(backend="viewer", seed=1)._repr_html_()
    # Gzipped in the page, so no markup of the graph's can end the script
    notes = [tip.get("note") for tip in _tips(html)[0]]
    assert "<img" not in html and "</script><img src=x onerror=alert(1)>" in notes
    # And shown as text
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    hover = js[js.index("function hover()") : js.index("// Pointers")]
    assert "name.textContent = k" in hover and "innerHTML" not in hover


def test_the_viewer_follows_the_theme(tmp_path):
    g = ec.karate_club()
    # In the automatic theme, the viewer reads it from a variable that the page's scheme sets
    plot = g.plot(backend="viewer", seed=1)
    theme = _options(plot._repr_html_())["theme"]
    assert theme.startswith("var(--ect-")
    assert f"{theme[4:-1]}:light" in plot._repr_html_()
    # A saved file is in the light theme
    plot.save(tmp_path / "plot.html")
    assert _options((tmp_path / "plot.html").read_text())["theme"] == "light"
    for given in ("light", "dark"):
        html = g.plot(backend="viewer", seed=1, theme=given)._repr_html_()
        assert _options(html)["theme"] == given
        # The box's colors are variables of its own, in the plot's theme unless the theme control
        # gives it another
        bg = svg_renderer.LIGHT.bg if given == "light" else svg_renderer.DARK.bg
        root = re.search(r'<div class="ec-viewer" id="([^"]+)"', html).group(1)
        base = re.search(rf"<style>#{root}\{{([^}}]+)\}}", html).group(1)
        assert re.search(r"background:var\(--ecv-\w+\);", _box(html))
        assert f":{bg};" in base and "var(--ect-" not in base


def test_the_viewer_has_the_svg_and_png_of_its_plot(tmp_path):
    g = ec.karate_club()
    # Edges aren't routed around nodes in the viewer, so its SVG is the straight-edged one
    opts = {"seed": 1, "theme": "light", "edge_routing": "straight"}
    viewer, vector = g.plot(backend="viewer", **opts), g.plot(backend="svg", **opts)
    assert viewer._svg.startswith("<svg") and "<circle" not in viewer._svg
    assert viewer.svg == vector.svg
    viewer.save(tmp_path / "plot.svg")
    assert (tmp_path / "plot.svg").read_text() == vector.svg
    assert viewer._repr_png_() == vector._repr_png_()
    routed = g.plot(backend="viewer", seed=1, theme="light").svg
    assert routed == viewer.svg and "<path" not in routed.split('class="ec-legend"')[0]


def test_an_empty_viewer_plot_is_an_svg():
    html = ec.Graph().plot(backend="viewer")._repr_html_()
    assert "ec-viewer" not in html and "<svg" in html


def test_the_viewer_has_smaller_nodes(monkeypatch):
    g = ec.Graph().add_nodes(list(range(3000)))
    scene = svg_renderer.build_scene(g, layout="random", seed=1, backend="viewer")
    header, _ = _payload(scene.payload())
    assert header["constants"]["r"] < 2.5
    # And no limit to its size, but a warning past what a notebook should hold
    monkeypatch.setattr(svg_renderer, "PLOT_BYTES_MAX", 1000)
    monkeypatch.setattr(svg_renderer, "NOTEBOOK_BYTES_MAX", 1000)
    with pytest.warns(UserWarning, match="backend='png' makes it smaller") as caught:
        plot = ec.karate_club().plot(seed=1, backend="viewer")
    assert caught[0].filename == __file__
    # Its SVG is too large for it
    with pytest.raises(ValueError, match=r"The plot's SVG would take at least \d MB"):
        plot.svg


def test_the_svg_of_a_large_viewer_plot_warns(monkeypatch):
    plot = ec.karate_club().plot(seed=1, backend="viewer")
    monkeypatch.setattr(svg_renderer, "NOTEBOOK_BYTES_MAX", 1000)
    with pytest.warns(UserWarning, match="which the viewer shows more readily"):
        plot.svg


def _box(html):
    return re.search(r'<div class="ec-viewer" id="[^"]+" style="([^"]+)"', html).group(1)


def test_the_viewer_can_fill_its_container():
    g = ec.karate_club()
    # By default it's the plot's size, or scaled down to fit, keeping its shape
    box = _box(g.plot(backend="viewer", seed=1)._repr_html_())
    assert "width:700px;aspect-ratio:700/500;max-width:100%" in box
    # A CSS width keeps the height, and the layout keeps the default canvas's shape
    plot = g.plot(backend="viewer", seed=1, width="100%", height=400)
    box = _box(plot._repr_html_())
    assert "width:100%;height:400px;max-width:100%" in box and "aspect-ratio" not in box
    header, _ = _html_payload(plot._repr_html_())
    assert (header["width"], header["height"]) == (560, 400)
    assert 'width="560" height="400"' in plot.svg
    assert "width:40em;" in _box(g.plot(backend="viewer", seed=1, width="40em")._repr_html_())
    with pytest.raises(ValueError, match="width='100%' needs backend='viewer'"):
        g.plot(width="100%")
    for bad in ("wide", "100", "-5px", "100%; color:red"):
        with pytest.raises(ValueError, match="isn't a CSS length"):
            g.plot(backend="viewer", width=bad)


def test_the_viewers_legend_keeps_its_size():
    g = ec.karate_club().cmty_louvain(attr_name="c")
    for width in (700, "100%"):
        html = g.plot(backend="viewer", seed=1, color_by="c", width=width)._repr_html_()
        layer = re.search(r'<div class="ec-legend-layer"[^>]*><svg [^>]*>', html).group(0)
        # In pixels, the same whatever the plot's size, for the viewer to fold when it's small
        size = re.search(r'width="(\d+)" height="(\d+)" viewBox="0 0 (\d+) (\d+)"', layer)
        lw, lh = int(size.group(1)), int(size.group(2))
        assert (size.group(3), size.group(4)) == (str(lw), str(lh))
        # A title and a row for each community
        rows = 1 + html.split("ec-legend-layer")[1].count("<circle")
        assert 40 < lw < 200 and lh > rows * svg_renderer._LEGEND_ROW
    # Without a legend there's no layer
    html = ec.karate_club().plot(backend="viewer", seed=1)._repr_html_()
    assert 'class="ec-legend-layer"' not in html


def test_the_viewer_shows_full_screen():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    # The plot's box, or an overlay over the window where the page doesn't allow that
    assert "requestFullscreen" in js and "ec-expanded" in js and "ec-legend-folded" in js


def test_the_viewers_code_is_ascii():
    # It is inlined in pages that may not say how they are encoded
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_bytes()
    assert js.isascii() and b"export function mount(" in js and b"export function parse(" in js


def test_the_payload_has_what_the_controls_start_from():
    g = ec.karate_club()
    header, _ = _html_payload(g.plot(backend="viewer", seed=1)._repr_html_())
    style = header["style"]
    assert set(style) == {"node_size", "edge_width", "edge_opacity", "curved"}
    assert style["curved"] is False and style["edge_opacity"] == pytest.approx(0.4)
    # The nodes' radii and the edges' widths are the style's, where nothing encodes them
    assert header["constants"]["r"] == pytest.approx(style["node_size"])
    assert header["constants"]["width"] == pytest.approx(style["edge_width"])
    html = g.plot(backend="viewer", seed=1, curved_edges=True, node_size=9, edge_width=2)
    style = _html_payload(html._repr_html_())[0]["style"]
    assert style["curved"] is True and (style["node_size"], style["edge_width"]) == (9, 2)
    # A scene that isn't a plot's has none
    assert _payload(_scene().payload())[0]["style"] is None


def test_the_payload_has_the_highlighted_edges_levels():
    g = ec.karate_club()
    header, columns = _html_payload(
        g.plot(backend="viewer", seed=1, highlight=[1, 2, 3])._repr_html_()
    )
    level = columns.get("level") or (header["constants"]["level"],) * header["edges"]
    assert {0, 2} <= set(level) <= {0, 1, 2}
    base = columns.get("base_width") or (header["constants"]["base_width"],) * header["edges"]
    width = columns["width"]
    w = header["style"]["edge_width"]
    for lv, b, wd in zip(level, base, width):
        want = b if lv == 0 else max(1.3 * b, 1.0) if lv == 1 else max(2 * b, 1.5)
        assert wd == pytest.approx(want, rel=1e-6)
    assert all(b == pytest.approx(w) for b in base)
    # Without highlighting there are no levels
    header, columns = _html_payload(g.plot(backend="viewer", seed=1)._repr_html_())
    assert "level" not in columns and "level" not in header["constants"]


def test_the_viewer_has_the_call_for_its_code():
    g = ec.karate_club().cmty_louvain(attr_name="c")
    html = g.plot(backend="viewer", seed=1, color_by="c", width=700, highlight=[1, 2])._repr_html_()
    call = _options(html)["call"]
    # The parameters in order, and the arguments that aren't the defaults, as Python source
    assert call["params"][:2] == ["layout", "color_by"] and "self" not in call["params"]
    assert dict(call["args"]) == {
        "color_by": '"c"', "seed": "1", "highlight": "[1, 2]", "backend": '"viewer"',
    }  # fmt: skip
    # Not the file it was written to
    plot = g.plot(backend="viewer", seed=1, output=None)
    assert "output" not in dict(_options(plot._repr_html_())["call"]["args"])
    assert svg_renderer._source(("a", 1)) == '("a", 1)' and svg_renderer._source(("a",)) == '("a",)'
    assert svg_renderer._source({"x": [1.5, None]}) == '{"x": [1.5, None]}'


def test_the_viewer_knows_whether_the_nodes_size_moves_them():
    g = ec.karate_club()
    assert _options(g.plot(backend="viewer", seed=1)._repr_html_())["moves"] is True
    for opts in ({"node_overlap": "allow"}, {"layout": "circular"}):
        assert _options(g.plot(backend="viewer", seed=1, **opts)._repr_html_())["moves"] is False


def test_the_viewer_can_change_its_theme():
    html = ec.karate_club().plot(backend="viewer", seed=1, theme="light")._repr_html_()
    root = re.search(r'<div class="ec-viewer" id="([^"]+)"', html).group(1)
    # Its variables in each theme the control can choose, the automatic one following the page
    for name in ("ec-light", "ec-dark", "ec-auto"):
        assert f"#{root}.{name}{{" in html
    assert f"@media (prefers-color-scheme: dark){{#{root}.ec-auto{{" in html
    assert svg_renderer.DARK.bg in html


def test_the_viewer_has_its_controls():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    # A panel behind a button, of tabs, that docks in a wide box, and writes the call
    for part in ('"Settings"', '"tablist"', '"Copy as code"', "aria-expanded", "ec-docked"):
        assert part in js


def _label_texts(html):
    packed = re.search(r'<script type="application/octet-stream" class="ec-labels">([^<]+)<', html)
    return packed and json.loads(gzip.decompress(base64.b64decode(packed.group(1))))


def test_the_viewer_has_the_labels():
    g = ec.les_miserables()
    html = g.plot(backend="viewer", seed=1, label_by="label")._repr_html_()
    header, columns = _html_payload(html)
    # What the plot draws at most, and the order in which the viewer chooses again for the view
    labels = header["labels"]
    assert (labels["show"], labels["limit"], labels["auto"]) == (True, 30, 30)
    assert labels["font_size"] > 0
    assert sorted(columns["label_order"]) == list(range(77))
    color = columns.get("label_color") or (header["constants"]["label_color"],) * 77
    assert len(color) == 77
    # The texts, out of the payload, as the tooltips are
    texts = _label_texts(html)
    assert len(texts["nodes"]) == 77 and "Valjean" in texts["nodes"] and texts["edges"] is None
    # Which the viewer has without show_labels= too, for its control to show them
    html = g.plot(backend="viewer", seed=1, show_labels=False, max_labels=5)._repr_html_()
    labels = _html_payload(html)[0]["labels"]
    assert (labels["show"], labels["limit"], labels["auto"]) == (False, 5, 30)
    assert len(_label_texts(html)["nodes"]) == 77
    assert dict(_options(html)["call"]["args"])["max_labels"] == "5"
    # A small graph's are all drawn
    labels = _html_payload(ec.karate_club().plot(backend="viewer", seed=1)._repr_html_())[0]
    assert (labels["labels"]["limit"], labels["labels"]["auto"]) == (None, None)
    # The scene that isn't a plot's has none
    assert _payload(_scene().payload())[0]["labels"] is None


def test_the_viewer_has_the_edges_labels_in_drawing_order():
    g = ec.karate_club()
    g.set_edge_attrs("tag", {i: f"e{i}" for i in g.get_edge_ids()})
    html = g.plot(backend="viewer", seed=1, edge_label_by="tag", highlight=[30, 34])._repr_html_()
    # As the tooltips are, highlighted edges last
    edges = _label_texts(html)["edges"]
    assert edges == [f"e{tip['id']}" for tip in _tips(html)[1]]
    assert len(edges) == 78 and len(set(edges)) == 78


def test_the_viewer_has_a_labels_tab():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    assert 'tab: "Labels"' in js and "function labelLayer(" in js and 'name: "max_labels"' in js


def test_the_payload_has_the_self_loops_sizes():
    scene = _scene()
    scene.set_edges(
        [0, 0, 1], [0, 0, 2], [0] * 3, [1.0] * 3, [1.0] * 3, [0.0] * 3, [1.0, 1.6, 1.0],
        [None] * 3, [0, 1, 2], ["{}"] * 3,
    )  # fmt: skip
    _, columns = _payload(scene.payload())
    assert columns["loop_size"] == pytest.approx((1.0, 1.6, 1.0))
    # A multigraph's, as the SVG draws them
    g = ec.Graph(directed=True).add_nodes(["a", "b"])
    g.add_edges([("a", "a"), ("a", "a"), ("a", "b")])
    header, columns = _html_payload(g.plot(backend="viewer", seed=1)._repr_html_())
    assert header["directed"] is True

    def col(name):
        return columns.get(name) or (header["constants"][name],) * header["edges"]

    loops = sorted(k for s, t, k in zip(col("src"), col("tgt"), col("loop_size")) if s == t)
    assert len(loops) == 2 and loops[0] == 1.0 and loops[1] > 1.0


def test_the_viewer_has_the_edges_tone_and_the_shading(monkeypatch):
    g = ec.karate_club()
    options = _options(g.plot(backend="viewer", seed=1)._repr_html_())
    # "auto" is what backend="auto" would draw: SVG here, so "over"
    assert options["tone"] == {"value": "auto", "auto": "over"} and options["shade"] is False
    html = g.plot(backend="viewer", seed=1, edge_tone="log", shade=True)._repr_html_()
    options = _options(html)
    assert options["tone"] == {"value": "log", "auto": "over"} and options["shade"] is True
    assert dict(options["call"]["args"])["edge_tone"] == '"log"'
    # And "log" where it would draw images
    monkeypatch.setattr(svg_renderer, "SVG_ELEMENTS_MAX", 50)
    assert _options(g.plot(backend="viewer", seed=1)._repr_html_())["tone"]["auto"] == "log"


def test_the_viewer_draws_arrowheads_self_loops_and_the_log_tone():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    shader = js[js.index("const LINK_VS") : js.index("const LINK_FS")]
    # The arrowhead's corners after the strip, a self-loop's two cubic curves, and the edges
    # trimmed at the nodes' borders
    assert "u_directed" in shader and "a_loop_size" in shader and "cubic(" in shader
    assert "Self-loops aren't drawn" not in js and "[1, 1, 2, 0, 2, 0, 2, 1, 2, 2]" in js
    # The log tone, opaque at the view's densest pixel, and the spheres' gradient
    assert "u_lone" in js and "PEAK_FS" in js and "function loneAt(" in js
    assert "vec2(-0.35, 0.35)" in js and "1.5 * v_r" in js
    # Their controls, and the picker takes self-loops
    assert 'name: "edge_tone"' in js and 'name: "shade"' in js and "function loopPoints(" in js


def test_the_payload_has_what_the_selection_draws():
    g = ec.karate_club()
    html = g.plot(backend="viewer", seed=1)._repr_html_()
    header, columns = _html_payload(html)
    select = header["select"]
    assert set(select) == {
        "accent", "accent_border", "muted", "muted_border", "label", "muted_label", "edge",
        "node_colors", "edge_colors",
    }  # fmt: skip
    assert (select["node_colors"], select["edge_colors"]) == (False, False)
    # Its colors are in the table, after the scene's, which the SVG has
    light = header["colors"]["light"]
    assert all(select[k] < len(light) for k in select if k not in ("node_colors", "edge_colors"))
    assert len(header["colors"]["dark"]) == len(light)
    # The nodes' IDs, in two halves
    hi = columns.get("node_id_hi") or (header["constants"]["node_id_hi"],) * 34
    assert sorted(columns["node_id"]) == list(range(1, 35)) and set(hi) == {0}
    assert [int(tip["id"]) for tip in _tips(html)[0]] == list(columns["node_id"])
    # The columns without highlighting are the plot's own where it has none
    assert not any(name.startswith("base_") and name != "base_width" for name in columns)
    html = g.plot(backend="viewer", seed=1, color_by="faction", highlight=[1, 2, 3])._repr_html_()
    header, columns = _html_payload(html)
    assert header["select"]["node_colors"] is True
    assert sorted(columns["base_label_order"]) == list(range(34))
    fill = columns.get("base_fill") or (header["constants"]["base_fill"],) * 34
    assert len(set(fill)) == 2 and set(fill) != set(columns.get("fill") or ())
    assert len(columns.get("base_color") or (header["constants"]["base_color"],) * 78) == 78
    # A scene that isn't a plot's has no selection
    assert _payload(_scene().payload())[0]["select"] is None


def test_the_payload_has_the_legends_keys():
    g = ec.karate_club()
    for e in g.get_edge_ids(limit=None):
        g.set_edge_attrs("kind", "strong" if e % 3 else "weak", edge_ids=[e])
    html = g.plot(backend="viewer", seed=1, color_by="faction", edge_color_by="kind")._repr_html_()
    header, columns = _html_payload(html)
    # The swatches' places, plus 1, in the order the legend draws them
    assert set(columns["node_key"]) == {1, 2} and set(columns["edge_key"]) == {3, 4}
    key = r'<g class="ec-key" data-key="(\d+)" data-block="(\d+)"[^>]*aria-label="([^"]+)"'
    keys = re.findall(key, html)
    assert keys == [("0", "0", "1"), ("1", "0", "2"), ("2", "1", "strong"), ("3", "1", "weak")]
    # The edges by their kind, in drawing order, as the tooltips have them
    kinds = [tip["kind"] for tip in _tips(html)[1]]
    assert [{"strong": 3, "weak": 4}[k] for k in kinds] == list(columns["edge_key"])
    # Edges colored by their nodes' groups: those within each, and those between
    html = g.plot(backend="viewer", seed=1, edge_color_by="faction")._repr_html_()
    header, columns = _html_payload(html)
    assert set(columns["edge_key"]) == {1, 2, 3} and header["constants"]["node_key"] == 0
    assert html.count('class="ec-key"') == 3
    # Not in the SVG, nor without a legend
    svg = g.plot(backend="svg", seed=1, color_by="faction", edge_color_by="kind").svg
    assert "ec-key" not in svg
    html = g.plot(backend="viewer", seed=1, color_by="faction", legend=False)._repr_html_()
    header, columns = _html_payload(html)
    assert "node_key" not in columns and "node_key" not in header["constants"]


def test_the_viewer_selects_searches_and_filters():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    # A click or a lasso selects, drawn as highlight= draws it, and written as it
    assert "function clickAt(" in js and "function lassoEnd(" in js and "ec-lasso" in js
    assert "function relook(" in js and 'args.set("highlight_edges"' in js
    # The search pill, and the legend's swatches as checkboxes
    assert "function searchPill(" in js and '"combobox"' in js and "function flyTo(" in js
    assert ".ec-key" in js and "The legend's filters aren't in the code" in js
