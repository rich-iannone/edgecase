"""Tests for the viewer: `Scene.payload()`, the scene as the bytes that `static/viewer.js` draws
with WebGL, and `plot(backend="viewer")`, the HTML that holds both.

The viewer itself runs in a browser, so these tests read what it's given, and not what it draws.
"""

import base64
import gzip
import hashlib
import html as html_mod
import json
import re
import shutil
import struct
import subprocess
import sys
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
    return json.loads(
        re.search(r'<script type="application/json" class="ec-options">([^<]+)<', html).group(1)
    )


def test_the_viewer_backend_holds_the_viewer_and_the_payload():
    g = ec.karate_club().cmty_louvain(attr_name="c")
    html = g.plot(backend="viewer", seed=1, color_by="c")._repr_html_()
    header, columns = _html_payload(html)
    assert (header["nodes"], header["edges"]) == (34, 78) and len(columns["x"]) == 34
    # The viewer's code, the runtime that defines the element, and the legend over it, as SVG
    assert html.count('<script type="module">') == 1 and svg_renderer._runtime_js() in html
    assert 'customElements.define("ec-graph"' in html
    assert 'class="ec-legend-layer"' in html and 'class="ec-legend"' in html
    # Not the SVG's nodes and edges, whose tooltips are in a table of their own
    drawing = html.split('class="ec-legend"')[0]
    assert "<circle" not in drawing and "<line" not in drawing and "data-attrs" not in html
    assert 'class="ec-tips"' in html
    # Each plot's root is its own
    ids = re.findall(r'<ec-graph id="([^"]+)"', html)
    again = g.plot(backend="viewer", seed=1, color_by="c")._repr_html_()
    assert len(ids) == 1 and ids != re.findall(r'<ec-graph id="([^"]+)"', again)
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
    assert _tips(plot._repr_html_()) == full and ' data-e="' in plot.svg
    # (and is the static SVG, but for the legend's place)
    plain = {**opts, "color_by": None}
    assert g.plot(backend="viewer", **plain).svg == g.plot(backend="svg", **plain).svg


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
        root = re.search(r'<ec-graph id="([^"]+)"', html).group(1)
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
    assert "ec-graph" not in html and "<svg" in html


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
    return re.search(r'<ec-graph id="[^"]+" style="([^"]+)"', html).group(1)


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
    pad = svg_renderer._LEGEND_PAD

    def layer_size(html):
        layer = re.search(r'<div class="ec-legend-layer"([^>]*)><svg [^>]*>', html)
        # At the bottom left
        assert "left:0;bottom:0" in layer.group(1)
        # In pixels, the same whatever the plot's size, for the viewer to fold when it's small
        size = re.search(r'width="(\d+)" height="(\d+)" viewBox="0 0 (\d+) (\d+)"', layer.group(0))
        lw, lh = int(size.group(1)), int(size.group(2))
        assert (size.group(3), size.group(4)) == (str(lw), str(lh))
        return lw, lh

    for width in (700, "100%"):
        html = g.plot(backend="viewer", seed=1, color_by="c", width=width)._repr_html_()
        # The title and the communities after it, in a row
        lw, lh = layer_size(html)
        swatches = html.split("ec-legend-layer")[1].count("<circle")
        assert swatches > 2 and swatches * 40 < lw < 700 and lh == 2 * pad
    # A narrow plot's wraps
    lw, lh = layer_size(g.plot(backend="viewer", seed=1, color_by="c", width=120)._repr_html_())
    assert lw <= 120 and lh >= 2 * pad + svg_renderer._LEGEND_ROW
    # Without a legend there's no layer
    html = ec.karate_club().plot(backend="viewer", seed=1)._repr_html_()
    assert 'class="ec-legend-layer"' not in html


def _legend_place(svg):
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    move = re.search(r'class="ec-legend" transform="translate\(([-\d.]+) ([-\d.]+)\) scale', svg)
    drawing = svg.split('class="ec-legend"')[0]
    bottom = max(
        float(cy) + float(r)
        for cy, r in re.findall(r'<circle cx="[-\d.]+" cy="([-\d.]+)" r="([\d.]+)"', drawing)
    )
    return vb, float(move.group(1)), float(move.group(2)), bottom


def test_the_viewers_legend_is_a_band_below_the_drawing():
    g = ec.karate_club()
    viewer = g.plot(backend="viewer", seed=1, color_by="faction")
    vb, x, y, bottom = _legend_place(viewer.svg)
    unit = vb[3] / 500
    pad = svg_renderer._LEGEND_PAD
    # At the bottom left of the viewer's SVG, as the viewer has it, below the nodes
    assert x == pytest.approx(vb[0] + pad * unit, abs=0.1)
    assert y == pytest.approx(vb[1] + vb[3] - 2 * pad * unit, abs=0.1)
    assert bottom < y
    # and the home view frames the drawing above it
    assert viewer.view == viewer._home and viewer.view["zoom"] == 1
    # The static SVG keeps it at the top right
    vb, x, y, _ = _legend_place(g.plot(backend="svg", seed=1, color_by="faction").svg)
    assert y == pytest.approx(vb[1], abs=0.1) and x > vb[0] + vb[2] / 2


def test_the_legend_row_wraps():
    items = [
        ("node", ("#111111", "#eeeeee"), "business"),
        ("node", ("#222222", "#dddddd"), "private person"),
    ]
    swatches = {"kind": "swatches", "title": "kind", "items": items}
    ramp = {"kind": "ramp", "title": "degree", "scale": "diverging", "ends": (1, 17, 8)}
    ramp["missing"] = True
    pad = svg_renderer._LEGEND_PAD
    marks, w, h = svg_renderer._legend_row([swatches, ramp], 1000)
    # A row, the ramp's values in order along it: low, its two halves split by the midpoint, high
    assert h == 2 * pad and w < 1000 and {m[2] for m in marks if m[0] == "text"} == {pad}
    texts = [m[3] for m in marks if m[0] == "text"]
    assert texts == ["kind", "business", "private person", "degree", "1", "8", "17", "no value"]
    bars = [m for m in marks if m[0] == "bar"]
    x = {m[3]: m[1] for m in marks if m[0] == "text"}
    assert x["1"] < bars[0][1] < bars[0][2] < x["8"] < bars[1][1] < bars[1][2] < x["17"]
    # (which share one gradient from the first's start to the second's end)
    assert bars[0][5:] == bars[1][5:] == (bars[0][1], bars[1][2])
    colors = svg_renderer._Colors("light")
    svg = svg_renderer._svg_legend(marks, colors, colors.theme)
    assert svg.count("<linearGradient") == 1 and svg.count('fill="url(#ec-ramp-') == 2
    # Narrow, a block's items wrap to go on under its first
    marks, w, h = svg_renderer._legend_row([swatches, ramp], 230)
    assert w <= 230 and h > 2 * pad + svg_renderer._LEGEND_ROW
    business, person = [m for m in marks if m[0] == "node"]
    assert business[3] == person[3] > 0 and person[1] == business[1] + svg_renderer._LEGEND_ROW
    # As keys, each swatch's box is around it
    keys = svg_renderer._svg_legend(marks, colors, colors.theme, keys=True)
    assert f'<rect x="{business[3] - 4:g}"' in keys and keys.count('class="ec-key"') == 2


def test_the_viewers_look():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    style = js[js.index("const STYLE") :]
    # A light border, controls of glass over the drawing until the plot is hovered, and the
    # panel's tabs marked as changed without growing
    assert ".ec-viewer::after" in style and "--ec-glass" in style and "backdrop-filter" in style
    assert ":is(:hover, :focus-within) .ec-controls button" in style
    assert "[role=tab].ec-changed::after" in style and "position: absolute" in style
    # The legend's folding and the hint over it
    assert "LEGEND_SHARE * canvas.clientHeight" in js and "--ec-legend-h" in js
    # Zoomed in, the arrowheads grow and darken, no longer than their nodes are across, and a
    # border is at most half its node's radius
    shader = js[js.index("const LINK_VS") : js.index("const LINK_FS")]
    assert "HEAD_GROW" in shader and "HEAD_GAIN" in shader and "rb / (4.0 * base)" in shader
    points = js[js.index("const POINT_VS") : js.index("const POINT_FS")]
    assert "min(st.y, 0.5 * nd.z)" in points


def test_the_viewer_shows_full_screen():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    # The plot's box, or an overlay over the window where the page doesn't allow that
    assert "requestFullscreen" in js and "ec-expanded" in js and "ec-legend-folded" in js


def test_the_viewers_code_is_ascii():
    # It is inlined in pages that may not say how they are encoded
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_bytes()
    assert js.isascii() and b"export function mount(" in js and b"export function parse(" in js
    small = (Path(svg_renderer.__file__).parent / "static" / "viewer.min.js").read_bytes()
    assert small.isascii() and b" as mount," in small and b" as parse}" in small


def test_the_minified_runtime_is_up_to_date():
    # `scripts/minify_viewer.py` writes it, after changing `viewer.js`, with the hash of what it
    # minified on its first line
    static = Path(svg_renderer.__file__).parent / "static"
    digest = hashlib.sha256((static / "viewer.js").read_bytes()).hexdigest()
    head = (static / "viewer.min.js").read_text().split("\n", 1)[0]
    assert head == f"// edgecase's viewer.js, minified by scripts/minify_viewer.py, sha256 {digest}"
    assert svg_renderer._runtime_file() == static / "viewer.min.js"
    assert len(svg_renderer._runtime_js()) < 0.6 * (static / "viewer.js").stat().st_size


def test_the_runtime_is_the_viewers_code_while_the_minified_one_is_out_of_date(
    tmp_path, monkeypatch
):
    (tmp_path / "static").mkdir()
    source, small = tmp_path / "static" / "viewer.js", tmp_path / "static" / "viewer.min.js"
    source.write_text("export function mount() {}\n")
    monkeypatch.setattr(svg_renderer, "__file__", str(tmp_path / "svg_renderer.py"))
    # None, one of another `viewer.js`, and one of this one
    assert svg_renderer._runtime_file() == source
    small.write_text(f"// sha256 {hashlib.sha256(b'other').hexdigest()}\nexport{{}};\n")
    assert svg_renderer._runtime_file() == source
    small.write_text(f"// sha256 {hashlib.sha256(source.read_bytes()).hexdigest()}\nexport{{}};\n")
    assert svg_renderer._runtime_file() == small


def test_a_page_can_carry_the_runtime_once(tmp_path, monkeypatch):
    monkeypatch.setattr(svg_renderer, "_runtime", "inline")
    js = svg_renderer._runtime_js()
    plot = ec.karate_club().plot(backend="viewer", seed=1)
    # The element alone, which holds no code, and with the runtime after it, by default
    alone = plot.as_raw_html(runtime="none")
    assert alone.endswith("</ec-graph></div>") and '<script type="module"' not in alone
    assert plot.as_raw_html() == plot._repr_html_() == f"{alone}\n{ec.runtime_tags()}"
    assert ec.runtime_tags() == f'<script type="module">\n{js}</script>'
    # With a nonce for the page's policy, or from a file
    assert ec.runtime_tags(nonce="r4nd0m").startswith('<script type="module" nonce="r4nd0m">\n')
    tag = ec.runtime_tags(nonce='a"b', src="/assets/ec-graph.js?v=1&x=2")
    assert tag == (
        '<script type="module" src="/assets/ec-graph.js?v=1&amp;x=2" nonce="a&quot;b"></script>'
    )
    ec.write_runtime(tmp_path / "ec-graph.js")
    assert (tmp_path / "ec-graph.js").read_text() == js
    # For the plots of a notebook, or of a page rendered from one, until set again; a file
    # always has it
    ec.set_runtime("none")
    assert plot._repr_html_() == plot.as_raw_html() == alone
    assert plot.as_raw_html(runtime="inline").endswith(f"{js}</script>")
    plot.save(tmp_path / "plot.html")
    assert (tmp_path / "plot.html").read_text().count(js) == 1
    ec.set_runtime("inline")
    assert plot._repr_html_().endswith(f"{js}</script>")
    # The other renderings hold their own code
    svg = ec.karate_club().plot(seed=1)
    assert svg.as_raw_html(runtime="none") == svg.as_raw_html() == svg._repr_html_()
    for bad in (lambda: ec.set_runtime("cdn"), lambda: plot.as_raw_html(runtime="once")):
        with pytest.raises(ValueError, match="use one of 'inline', 'none'"):
            bad()


def test_the_viewers_options_stay_in_their_script():
    # An argument's source can hold what would end the script
    name = "</script><b>"
    g = ec.karate_club()
    ids = g.get_node_ids(limit=None)
    g.set_node_attrs(name, {v: str(v % 3) for v in ids}, nodes=ids)
    html = g.plot(backend="viewer", seed=1, color_by=name)._repr_html_()
    raw = re.search(r'class="ec-options">([^<]+)<', html).group(1)
    assert "\\u003c/script>" in raw
    assert dict(_options(html)["call"]["args"])["color_by"] == json.dumps(name)


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
    root = re.search(r'<ec-graph id="([^"]+)"', html).group(1)
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


def _filters(header, columns):
    """The payload's filters by name, each with its values, NaN for none."""
    out = {}
    for f in header["filters"]:
        count = header[f["of"]]
        values = columns.get(f["column"]) or (header["constants"][f["column"]],) * count
        out[f["name"]] = {**f, "values": [v + f["min"] for v in values]}
    return out


def test_the_payload_has_the_numeric_attributes_to_filter_by():
    g = ec.karate_club()
    ids = g.get_node_ids(limit=None)
    for i in ids:
        if i % 7:
            g.set_node_attrs("age", 18 + i, nodes=[i])
        g.set_node_attrs("score", 0.5 * i, nodes=[i])
        g.set_node_attrs("vip", i % 2 == 0, nodes=[i])
        g.set_node_attrs("group", "a" if i % 2 else "b", nodes=[i])
        g.set_node_attrs("mixed", i if i > 1 else "one", nodes=[i])
        g.set_node_attrs("huge", 10**400 if i == 1 else i, nodes=[i])
        g.set_node_attrs(
            "odd", float("nan") if i == 1 else float("inf") if i == 2 else 1.0 * i, nodes=[i]
        )
    for e in g.get_edge_ids(limit=None):
        g.set_edge_attrs("eid", e, edge_ids=[e])
        g.set_edge_attrs("same", 3, edge_ids=[e])
    header, columns = _html_payload(g.plot(backend="viewer", seed=1)._repr_html_())
    found = _filters(header, columns)
    # The nodes' degree, then their attributes and the edges' in the order they first appear,
    # those of ints or floats (not bools) with two values or more
    assert list(found) == ["degree", "faction", "age", "score", "odd", "eid"]
    assert [found[k]["of"] for k in found] == ["nodes"] * 5 + ["edges"]
    assert [f["column"] for f in header["filters"]] == [f"filter_{k}" for k in range(6)]
    assert [found[k]["integer"] for k in found] == [True, True, True, False, False, True]
    # Each node's value, in the nodes' order, none for a value that is missing or not finite
    order = columns["node_id"]
    age = found["age"]
    assert (age["min"], age["max"], age["missing"]) == (19, 52, 4)
    assert [None if v != v else v for v in age["values"]] == [
        None if i % 7 == 0 else 18 + i for i in order
    ]
    assert found["score"]["values"] == [0.5 * i for i in order]
    assert found["odd"]["missing"] == 2 and found["odd"]["min"] == 3
    assert found["degree"]["values"] == [g.get_degree(i) for i in order]
    # The edges' in the order drawn, as their tooltips are
    assert found["eid"]["values"] == [
        tip["id"] for tip in _tips(g.plot(backend="viewer", seed=1)._repr_html_())[1]
    ]
    # A float far from 0 keeps its precision, as the column holds it less the least
    g = ec.Graph().add_nodes(["a", "b", "c"]).add_edges([("a", "b"), ("b", "c")])
    g.set_node_attrs("t", [1e9 + 0.25, 1e9 + 0.5, 1e9 + 0.75], nodes=["a", "b", "c"])
    found = _filters(*_html_payload(g.plot(backend="viewer", seed=1)._repr_html_()))
    assert sorted(found["t"]["values"]) == [1e9 + 0.25, 1e9 + 0.5, 1e9 + 0.75]
    # A directed graph's degree counts the edges in and out
    g = ec.Graph(directed=True).add_nodes(["a", "b", "c"])
    g.add_edges([("a", "b"), ("a", "c"), ("b", "a")])
    header, columns = _html_payload(g.plot(backend="viewer", seed=1)._repr_html_())
    degree = dict(zip(columns["node_id"], _filters(header, columns)["degree"]["values"]))
    assert sorted(degree.values()) == [1, 2, 3]
    assert degree == {
        i: g.get_in_degree(i) + g.get_out_degree(i) for i in g.get_node_ids(limit=None)
    }


def test_the_payload_filters_by_degree_and_weight_unless_attributes_have_their_names():
    g = ec.les_miserables()
    found = _filters(*_html_payload(g.plot(backend="viewer", seed=1)._repr_html_()))
    assert list(found) == ["degree", "weight"]
    # Whole weights are ints
    assert (found["weight"]["of"], found["weight"]["integer"]) == ("edges", True)
    assert (found["weight"]["min"], found["weight"]["max"]) == (1, 31)
    g = ec.Graph().add_nodes(["a", "b", "c"]).add_edges([("a", "b"), ("b", "c"), ("c", "a")])
    g.set_edge_attrs("weight", [0.5, 1.5, 2.0], edge_ids=g.get_edge_ids(limit=None))
    found = _filters(*_html_payload(g.plot(backend="viewer", seed=1)._repr_html_()))
    # Every node's degree is 2, which leaves nothing to filter by
    assert list(found) == ["weight"] and found["weight"]["integer"] is False
    # An attribute named for them takes their place, here as one that isn't a number
    g.add_nodes(["d"]).add_edges([("a", "d")])
    g.set_node_attrs("degree", ["low", "high", "low", "low"], nodes=["a", "b", "c", "d"])
    found = _filters(*_html_payload(g.plot(backend="viewer", seed=1)._repr_html_()))
    assert "degree" not in found
    # None for a scene that isn't a plot's, and none at all for a graph without numbers
    assert _payload(_scene().payload())[0]["filters"] is None
    g = ec.Graph().add_nodes(["a", "b", "c", "d"])
    g.add_edges([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    assert _html_payload(g.plot(backend="viewer", seed=1)._repr_html_())[0]["filters"] == []


def test_the_viewer_has_a_filter_tab():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    # Each filter is a column, a range and a histogram that brushes it, and hides what is out
    # of it, as the legend's swatches do; both are in the copied code, as filters=
    assert "function filterTab(" in js and "ec-hist" in js and "ec-brush" in js
    assert "Add a filter" in js and "hidden while filtered" in js
    assert 'args.set("filters"' in js and "The filters aren't in the code" not in js


def test_the_viewer_selects_searches_and_filters():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    # A click or a lasso selects, drawn as highlight= draws it, and written as it
    assert "function clickAt(" in js and "function lassoEnd(" in js and "ec-lasso" in js
    assert "function relook(" in js and 'args.set("highlight_edges"' in js
    # The search pill, and the legend's swatches as checkboxes
    assert "function searchPill(" in js and '"combobox"' in js and "function flyTo(" in js
    assert ".ec-key" in js and "function keyFilters(" in js


def test_the_payload_has_the_groups_of_the_cluster_force():
    g = ec.karate_club()
    html = g.plot(backend="viewer", seed=1, color_by="faction")._repr_html_()
    header, columns = _html_payload(html)
    # Each node's group of a categorical `color_by=`: its value's place among them, plus 1
    assert header["cluster"] == {"name": "faction", "count": 2}
    faction = g.get_node_attrs("faction")
    places = {v: k + 1 for k, v in enumerate(sorted(set(faction.values())))}
    assert list(columns["cluster"]) == [places[faction[i]] for i in columns["node_id"]]
    # And the legend's band, which the viewer fits a moved layout above
    assert header["legend_h"] > 0
    html = g.plot(backend="viewer", seed=1, color_by="faction", legend=False)._repr_html_()
    header, _ = _html_payload(html)
    assert header["legend_h"] == 0 and header["cluster"]["count"] == 2
    # None for a ramp, nor without `color_by=`, nor for a scene that isn't a plot's
    ids = g.get_node_ids(limit=None)
    g.set_node_attrs("score", [0.37 * i for i in ids], nodes=ids)
    for kwargs in [{"color_by": "score"}, {}]:
        header, columns = _html_payload(g.plot(backend="viewer", seed=1, **kwargs)._repr_html_())
        assert header["cluster"] is None and "cluster" not in columns
        assert "cluster" not in header["constants"]
    assert _payload(_scene().payload())[0]["cluster"] is None


def test_the_viewer_has_a_layout_tab():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    # The forces by the names that the Rust layout is to take, which the copied code leaves out
    for name in ["gravity", "repulsion", "link_strength", "link_distance", "friction", "collision"]:
        assert f'force("{name}"' in js
    assert 'name: "cluster_by"' in js and 'force("cluster_strength"' in js
    assert "if (c.value === c.start || c.view) continue;" in js
    assert "The simulation's positions aren't in the code" in js
    # Run, pause and restore; a node dragged while it runs; a worker, or the page where none
    assert "function runBar(" in js and "Restore positions" in js and "function grab(" in js
    assert "function simulation(init)" in js and "function engine(init, got)" in js
    assert "new Worker(" in js and "function workerMain(" in js


NODE = shutil.which("node")
# A ring of 40 with chords, in two groups, from a circle, with nodes on one point, run as it is,
# again, with collisions and the cluster force, and with a node pinned and held
_RING = """
const n = 40, src = [], tgt = [];
for (let i = 0; i < n; i++) { src.push(i); tgt.push((i + 1) % n); if (i % 5 === 0) { src.push(i); tgt.push((i + 13) % n); } }
src.push(3); tgt.push(3);
const x = Float32Array.from({ length: n }, (_, i) => 300 + 200 * Math.cos(i / 6));
const y = Float32Array.from({ length: n }, (_, i) => 200 + 150 * Math.sin(i / 6));
x[7] = x[8]; y[7] = y[8];
const cluster = Uint32Array.from({ length: n }, (_, i) => 1 + (i % 2));
const init = (forces) => ({ n, m: src.length, frame: [0, 0, 600, 400], x, y, r: 6, src: Uint32Array.from(src),
  tgt: Uint32Array.from(tgt), cluster, groups: 2, forces, alpha: 0.1 });
function run(forces, pin) {
  const sim = simulation(init(forces));
  if (pin) { sim.pin(0, 100, 50); sim.hold(true); }
  let out, ticks = 0;
  do { out = sim.step(1); ticks++; } while (!out.settled && ticks < 400);
  if (pin) { sim.hold(false); out = sim.step(1); }
  return { ticks, settled: out.settled, x: Array.from(out.x), y: Array.from(out.y) };
}
const spread = (r) => { const m = (k) => r.x.filter((_, i) => i % 2 === k); return Math.max(...m(0)) - Math.min(...m(0)); };
const a = run({}), b = run({}), c = run({ collision: 1, cluster: true, cluster_strength: 0.5 }), p = run({}, true);
console.log(JSON.stringify({ a: [a.ticks, a.settled], same: JSON.stringify(a) === JSON.stringify(b),
  finite: [a, c, p].every((r) => r.x.concat(r.y).every(Number.isFinite)), clustered: spread(c) < spread(a),
  pinned: [p.x[0], p.y[0], p.ticks, p.settled] }));
"""  # noqa: E501


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_the_simulation_settles_alike_each_time(tmp_path):
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    start = js.index("function simulation(init) {")
    body = js[start : js.index("\n}\n", start) + 2]
    (tmp_path / "sim.mjs").write_text(body + _RING)
    run = subprocess.run([NODE, tmp_path / "sim.mjs"], capture_output=True, text=True, check=True)
    out = json.loads(run.stdout)
    # About 200 ticks from the warm start, the same each time
    assert out["a"] == [200, True] and out["same"] and out["finite"]
    # The cluster force draws each group together
    assert out["clustered"]
    # A node pinned stays put, and the drag's heat keeps it running until let go
    assert out["pinned"] == [pytest.approx(100, abs=1e-3), pytest.approx(50, abs=1e-3), 400, False]


# A GPU that draws `fit` edges in each of the display's frames: frames that take as long as their
# edges take, or as many whole frames, from a budget of 150,000 between 20,000 and 600,000
_FRAMES = """
const TICK = 1000 / 60;
function run(fit, frames, whole) {
  const b = frameBudget(150000, 20000, 600000), seen = [];
  for (let k = 0; k < frames; k++) {
    const t = TICK * Math.max(1, b.value / fit);
    seen.push([b.value, whole ? TICK * Math.ceil(t / TICK - 1e-9) : t]);
    b.frame(seen[k][1]);
  }
  const late = seen.slice(frames / 2);
  return { value: b.value, values: new Set(late.map((s) => s[0])).size,
    mean: late.reduce((a, s) => a + s[1], 0) / late.length, least: Math.min(...late.map((s) => s[0])) };
}
const slow = frameBudget(150000, 20000, 600000), steps = [];
for (let k = 0; k < 3; k++) steps.push(slow.frame(1000));
const few = frameBudget(150000, 20000, 5000), budgets = [few.value];
for (let k = 0; k < 50; k++) budgets.push(few.frame(k < 25 ? 5 : 1000));
console.log(JSON.stringify({ fast: run(1e9, 400, false), even: run(90000, 4000, false),
  whole: run(100000, 4000, true), slow: steps, few: Array.from(new Set(budgets)) }));
"""  # noqa: E501


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_the_viewers_budget_follows_its_frames_times(tmp_path):
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    start = js.index("function frameBudget(start, least, most) {")
    body = js[start : js.index("\n}\n", start) + 2]
    (script := tmp_path / "budget.mjs").write_text(body + _FRAMES)
    run = subprocess.run([NODE, script], capture_output=True, text=True, check=True)
    out = json.loads(run.stdout)
    tick = 1000 / 60
    # Frames in time bring it up to all the edges
    assert out["fast"]["value"] == 600_000
    # Frames that take as long as their edges do bring it down to about what fits, and it stays
    assert out["even"]["values"] == 1 and 90_000 <= out["even"]["value"] < 1.5 * 90_000
    assert out["even"]["mean"] < 1.5 * tick
    # Frames of whole frames, which a step up past what fits doubles, seldom miss one
    assert out["whole"]["mean"] < 1.02 * tick and out["whole"]["least"] > 40_000
    # Frames of a second each bring it down to the least at once, and fewer edges than that are
    # all drawn
    assert out["slow"] == [37_500, 20_000, 20_000] and out["few"] == [5_000]
    # It is used: the frames are timed, and the shaders compile without waiting where they can
    assert "frameBudget(BUDGET, BUDGET_MIN, header.edges)" in js and "budget.frame(" in js
    assert '"KHR_parallel_shader_compile"' in js and "COMPLETION_STATUS_KHR" in js


def _kernel(monkeypatch, kernel=True):
    """IPython, as a notebook's kernel has it, or as a terminal's (without a kernel)."""
    shell = type("Shell", (), {"kernel": object()})() if kernel else object()
    fake = type(sys)("IPython")
    fake.get_ipython = lambda: shell
    monkeypatch.setitem(sys.modules, "IPython", fake)
    monkeypatch.delenv("QUARTO_FIG_FORMAT", raising=False)


def test_a_viewer_plot_is_a_widget_in_a_notebooks_kernel(monkeypatch):
    js = svg_renderer._runtime_js()
    plot = ec.karate_club().plot(backend="viewer", seed=1)
    svg = ec.karate_club().plot(seed=1)
    # A kernel, but not one that Quarto runs to render a page, nor IPython in a terminal
    with monkeypatch.context() as m:
        _kernel(m)
        assert svg_renderer._widget_host()
        m.setenv("QUARTO_FIG_FORMAT", "png")
        assert not svg_renderer._widget_host()
        _kernel(m, kernel=False)
        assert not svg_renderer._widget_host()
        m.delitem(sys.modules, "IPython")
        assert not svg_renderer._widget_host()
    monkeypatch.setattr(svg_renderer, "_widget_host", lambda: False)
    assert plot._repr_mimebundle_() is None
    monkeypatch.setattr(svg_renderer, "_widget_host", lambda: True)
    data, metadata = plot._repr_mimebundle_()
    widget = plot.widget
    assert widget is plot.widget and data["text/plain"] == repr(plot)
    assert data["application/vnd.jupyter.widget-view+json"]["model_id"] == widget.model_id
    # The widget holds the plot and its state, and its module is the runtime
    assert widget._esm == js and widget.html == plot.as_raw_html(runtime="none")
    assert widget.selection == plot._raw("selection") and widget.view == plot.view
    assert widget.filters == []
    # Not the other renderings
    assert svg._repr_mimebundle_() is None and svg.widget is None
    # marimo shows the widget, and the other renderings as before
    assert plot._display_() is widget
    shown = svg._display_()
    assert shown._repr_html_() == svg._repr_html_() and shown._repr_png_() == svg._repr_png_()


def test_the_viewers_state_is_what_the_viewer_has():
    plot = ec.karate_club().plot(backend="viewer", seed=1)
    home = plot.view
    assert plot.selection == {"nodes": [], "edges": []} and plot.filters == []
    assert set(home) == {"x", "y", "zoom"} and home["zoom"] == 1
    # Each ID once and in order, and an undirected edge's ends in order
    plot.selection = {"nodes": [3, 1, 3, "2"], "edges": [(2, 1), [1, 2], (4, 3)]}
    assert plot.selection == {"nodes": [1, 2, 3], "edges": [(1, 2), (3, 4)]}
    plot.selection = {"edges": [(2, 1)]}
    assert plot.selection == {"nodes": [], "edges": [(1, 2)]}
    plot.selection = None
    assert plot.selection == {"nodes": [], "edges": []}
    g = ec.Graph(directed=True)
    g.add_edges([("a", "b"), ("b", "a")], auto_create_nodes=True)
    directed = g.plot(backend="viewer", seed=1)
    directed.selection = {"edges": [(2, 1), (1, 2)]}
    assert directed.selection["edges"] == [(1, 2), (2, 1)]
    # The view keeps what is left out, and the viewer's zooms; None is the view that fits
    plot.view = {"zoom": 5000}
    assert plot.view == {**home, "zoom": 1000.0}
    plot.view = {"x": 10, "zoom": 0.1}
    assert plot.view == {"x": 10.0, "y": home["y"], "zoom": 0.25}
    plot.view = None
    assert plot.view == home
    # A filter's "of" can be left out, and its range is from low to high
    plot.filters = [
        {"attr": "age", "range": [30, 40]},
        {"attr": "w", "of": "edges", "range": (1, 1)},
    ]
    assert plot.filters == [
        {"attr": "age", "of": None, "range": (30.0, 40.0)},
        {"attr": "w", "of": "edges", "range": (1.0, 1.0)},
    ]
    plot.filters = None
    assert plot.filters == []
    bad = [
        ("selection", [1, 2], TypeError, "takes a dict"),
        ("selection", {"node": [1]}, ValueError, "not 'node'"),
        ("selection", {"nodes": [True]}, TypeError, "takes node IDs"),
        ("selection", {"nodes": ["a"]}, TypeError, "takes node IDs"),
        ("selection", {"edges": [(1, 2, 3)]}, TypeError, r"takes \(u, v\) pairs"),
        ("view", {"zoom": 0}, ValueError, "more than 0"),
        ("view", {"x": float("nan")}, TypeError, "finite number"),
        ("view", {"cx": 1}, ValueError, "not 'cx'"),
        ("filters", {"attr": "age"}, TypeError, "list of dicts"),
        ("filters", [{"attr": 1, "range": (0, 1)}], TypeError, "attribute's name"),
        ("filters", [{"attr": "a", "of": "graph", "range": (0, 1)}], ValueError, "'nodes' or"),
        ("filters", [{"attr": "a", "range": 1}], TypeError, r"\(low, high\) pair"),
        ("filters", [{"attr": "a", "range": (2, 1)}], ValueError, "from low to high"),
    ]
    for name, value, error, match in bad:
        with pytest.raises(error, match=match):
            setattr(plot, name, value)
    assert plot.selection == {"nodes": [], "edges": []} and plot.filters == []
    # The other renderings have none of it
    svg = ec.karate_club().plot(seed=1)
    assert svg.selection is svg.view is svg.filters is None
    for name in ("selection", "view", "filters"):
        with pytest.raises(ValueError, match="Only a plot with backend='viewer'"):
            setattr(svg, name, None)
    with pytest.raises(ValueError, match="Only a plot with backend='viewer'"):
        svg.on_change(print)


def test_the_viewers_selection_is_the_graphs():
    g = ec.karate_club()
    plot = g.plot(backend="viewer", seed=1)
    plot.selection = {"nodes": [1, 34, 99]}
    assert sorted(g.get_selected_node_ids()) == [1, 34]
    assert g.plot(highlight="selection", seed=1).svg == g.plot(highlight=[1, 34], seed=1).svg
    # A pair selects every edge between its ends
    plot.selection = {"edges": [(2, 1), (1, 3), (5, 99)]}
    edges = sorted(g.find_edge_ids(1, 2) + g.find_edge_ids(1, 3))
    assert g.get_selected_node_ids() == [] and sorted(g.get_selected_edge_ids()) == edges
    assert (
        g.plot(highlight_edges="selection", seed=1).svg == g.plot(highlight_edges=edges, seed=1).svg
    )
    # A graph selects nodes or edges, so it has the nodes of both
    plot.selection = {"nodes": [5], "edges": [(1, 2)]}
    assert g.get_selected_node_ids() == [5] and g.get_selected_edge_ids() == []
    plot.selection = None
    assert g.get_selected_node_ids() == [] and g.get_selected_edge_ids() == []
    # Selecting in the graph leaves the viewer's selection
    g.select_nodes_by_id([7])
    assert plot.selection == {"nodes": [], "edges": []}
    g2 = ec.Graph(directed=True)
    g2.add_edges([("a", "b"), ("a", "b"), ("b", "a")], auto_create_nodes=True)
    directed = g2.plot(backend="viewer", seed=1)
    directed.selection = {"edges": [(1, 2)]}
    assert sorted(g2.get_selected_edge_ids()) == sorted(g2.find_edge_ids(1, 2)) != []
    assert len(g2.get_selected_edge_ids()) == 2


def test_on_change_follows_the_viewer_and_python():
    g = ec.karate_club()
    plot = g.plot(backend="viewer", seed=1)
    seen = []

    @plot.on_change
    def changed(name, value):
        seen.append((name, value, sorted(g.get_selected_node_ids())))

    assert callable(changed)
    plot.selection = {"nodes": [2, 1]}
    plot.selection = {"nodes": [1, 2]}
    plot.view = {"zoom": 2}
    plot.filters = []
    assert seen == [
        ("selection", {"nodes": [1, 2], "edges": []}, [1, 2]),
        ("view", {**plot.view, "zoom": 2.0}, [1, 2]),
    ]
    # Once there is a widget, it holds the state, which the viewer changes through it
    widget = plot.widget
    assert widget.selection == {"nodes": [1, 2], "edges": []} and widget.view["zoom"] == 2
    seen.clear()
    widget.selection = {"nodes": [34], "edges": []}
    widget.filters = [{"attr": "faction", "of": "nodes", "range": [1, 1]}]
    assert seen == [
        ("selection", {"nodes": [34], "edges": []}, [34]),
        ("filters", [{"attr": "faction", "of": "nodes", "range": (1.0, 1.0)}], [34]),
    ]
    assert plot.selection["nodes"] == [34]
    plot.view = None
    assert widget.view == plot.view and seen[-1][0] == "view"
    with pytest.raises(TypeError, match="takes a function"):
        plot.on_change("print")


def test_the_widgets_module_renders_the_plot():
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_text()
    # anywidget's render(): the element, once upgraded, with its state both ways, and drawn
    # again for new HTML. The default export gives it to anywidget, Panel and marimo, and is
    # Streamlit's component
    assert "export default function host(component) {" in js
    assert "component?.parentElement ? streamlit(component) : { render }" in js
    assert "function render({ model, el })" in js and 'model.on("change:html", draw)' in js
    assert "customElements.upgrade(plot)" in js and "model.save_changes()" in js
    assert '"ec-change"' in js and 'const STATES = ["selection", "view", "filters"];' in js
    assert tuple(svg_renderer.STATES) == ("selection", "view", "filters")
    # A shadow root, as marimo shows a widget in, has the style, the focus and the clicks
    assert "root.getRootNode()" in js and "e.composedPath()" in js
    assert "document.activeElement" not in js


def _aged():
    g = ec.karate_club()
    g.set_node_attrs("age", [20 + i for i in range(34)], nodes=g.get_node_ids(limit=None))
    return g


def _drawn(svg):
    """The nodes that an SVG draws, by ID: (x, y, r)."""
    found = re.findall(
        r'<circle cx="([-\d.]+)" cy="([-\d.]+)" r="([\d.]+)"[^>]*data-node="(\d+)"', svg
    )
    return {int(i): (float(x), float(y), float(r)) for x, y, r, i in found}


def test_plot_filters_leave_out_what_they_hide():
    g = _aged()
    age = g.get_node_attrs("age")
    shown = {i for i, a in age.items() if 30 <= a <= 45}
    p = g.plot(
        seed=1, backend="svg", edge_routing="straight", filters=[{"attr": "age", "range": (30, 45)}]
    )
    assert set(_drawn(p.svg)) == shown
    ends = [(u, v) for _, u, v in g.get_edges_as_tuples() if u in shown and v in shown]
    assert p.svg.count("data-edge=") == len(ends)
    # The images draw them alike, and an edge's attribute filters edges
    g.plot(seed=1, backend="png", filters=[{"attr": "age", "range": (30, 45)}])._png_bytes()
    edge_ids = [e for e, _, _ in g.get_edges_as_tuples()]
    g.set_edge_attrs("w", {e: float(k) for k, e in enumerate(edge_ids)})
    p = g.plot(
        seed=1,
        backend="svg",
        edge_routing="straight",
        filters=[{"attr": "w", "of": "edges", "range": (0, 9)}],
    )
    assert len(_drawn(p.svg)) == 34 and p.svg.count("data-edge=") == 10


def test_plot_filters_of_values_turn_the_legends_swatches_off():
    g = ec.karate_club()
    faction = g.get_node_attrs("faction")
    p = g.plot(
        seed=1, backend="svg", color_by="faction", filters=[{"attr": "faction", "values": [2]}]
    )
    assert set(_drawn(p.svg)) == {i for i, f in faction.items() if f == 2}
    # The swatch off is faint, as the viewer draws it
    faint = re.findall(r'<g opacity="0\.35">(.*?)</g>', p.svg)
    assert len(faint) == 1 and ">1</text>" in faint[0]
    assert '<g opacity="0.35">' not in g.plot(seed=1, backend="svg", color_by="faction").svg


@pytest.mark.parametrize(
    "filters, error",
    [
        ([{"attr": "nope", "range": (1, 2)}], ValueError),
        ([{"attr": "age", "values": [1]}], ValueError),
        ([{"attr": "faction"}], ValueError),
        ([{"attr": "faction", "range": (1, 2), "values": [1]}], ValueError),
        ([{"attr": "faction", "values": [[1]]}], TypeError),
        ([{"attr": "faction", "values": 1}], TypeError),
        ([{"attr": "faction", "values": [float("nan")]}], TypeError),
        ([{"attr": "faction", "of": "edges", "values": [1]}], ValueError),
        ({"attr": "faction", "values": [1]}, TypeError),
    ],
)
def test_plot_filters_are_checked(filters, error):
    with pytest.raises(error):
        _aged().plot(seed=1, backend="svg", color_by="faction", filters=filters)


def test_plot_filters_of_values_need_the_legend():
    with pytest.raises(ValueError, match="swatches in the legend"):
        ec.karate_club().plot(
            backend="svg",
            color_by="faction",
            legend=False,
            filters=[{"attr": "faction", "values": [1]}],
        )


def test_filters_of_values_are_normalized():
    norm = svg_renderer._norm_filters
    assert norm([{"attr": "a", "values": {"b", "a"}}]) == [
        {"attr": "a", "of": None, "values": ["a", "b"]}
    ]
    # Each once, 1 and True alike, as the legend's groups take them
    assert norm([{"attr": "a", "of": "nodes", "values": [1, True, 1.0, None, 2]}])[0]["values"] == [
        1,
        None,
        2,
    ]
    # As the viewer gives them back
    given = [
        {"attr": "a", "of": "edges", "values": ["x"]},
        {"attr": "b", "of": "nodes", "range": [1, 2]},
    ]
    assert norm(given) == [given[0], {"attr": "b", "of": "nodes", "range": [1.0, 2.0]}]


def test_plot_view_draws_what_the_viewer_shows_there():
    g = ec.karate_club()
    home = g.plot(seed=1, backend="svg", view={})
    at = g.plot(seed=1, backend="svg", view={"zoom": 2.0})
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', home.svg).group(1).split()]
    assert vb == [float(v) for v in re.search(r'viewBox="([^"]+)"', at.svg).group(1).split()]
    hx, hy = vb[0] + vb[2] / 2, vb[1] + vb[3] / 2
    before, after = _drawn(home.svg), _drawn(at.svg)
    assert len(before) == 34 and 0 < len(after) < 34
    for i, (x, y, r) in after.items():
        x0, y0, r0 = before[i]
        assert x == pytest.approx(hx + (x0 - hx) * 2, abs=0.2)
        assert y == pytest.approx(hy + (y0 - hy) * 2, abs=0.2)
        assert r == pytest.approx(r0 * 2**0.5, abs=0.1)
    # Framed as the viewer frames it: the legend along the bottom, the edges unrouted
    shown = g.plot(seed=1, backend="viewer", color_by="faction")
    framed = g.plot(seed=1, backend="svg", color_by="faction", view={})
    assert _drawn(framed.svg) == _drawn(shown.svg)
    # The images too, and a point named
    g.plot(seed=1, backend="png", view={"x": 300, "y": 200, "zoom": 3})._png_bytes()


@pytest.mark.parametrize(
    "view, error",
    [
        ({"zoom": 0}, ValueError),
        ({"x": float("nan")}, TypeError),
        ({"z": 1}, ValueError),
        (2, TypeError),
    ],
)
def test_plot_view_is_checked(view, error):
    with pytest.raises(error):
        ec.karate_club().plot(backend="svg", view=view)


def test_the_viewer_starts_at_the_view_and_the_filters_given():
    g = _aged()
    filters = [
        {"attr": "age", "range": (25, 40)},
        {"attr": "faction", "of": "nodes", "values": [2]},
    ]
    p = g.plot(seed=1, backend="viewer", color_by="faction", view={"zoom": 30000}, filters=filters)
    options = _options(p._repr_html_())
    home = g.plot(seed=1, backend="viewer", color_by="faction").view
    assert options["state"]["view"] == {**home, "zoom": 1000.0} == p.view
    values = {"attr": "faction", "of": "nodes", "values": [2]}
    assert options["state"]["filters"] == [
        {"attr": "age", "of": None, "range": [25.0, 40.0]},
        values,
    ]
    assert p.filters == [{"attr": "age", "of": None, "range": (25.0, 40.0)}, values]
    # The legend's swatches, as filters of values name them
    assert options["keys"] == [
        {"attr": "faction", "of": "nodes", "value": 1, "source": "1"},
        {"attr": "faction", "of": "nodes", "value": 2, "source": "2"},
    ]
    # Without them, none
    plain = _options(g.plot(seed=1, backend="viewer")._repr_html_())
    assert plain["state"] == {"view": None, "filters": []} and plain["keys"] == []
