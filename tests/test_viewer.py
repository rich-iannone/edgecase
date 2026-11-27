"""Tests for the viewer: `Scene.payload()`, the scene as the bytes that `static/viewer.js` draws
with WebGL, and `plot(backend="viewer")`, the HTML that holds both.

The viewer itself runs in a browser, so these tests read what it's given, and not what it draws.
"""

import base64
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
        "opacity": pytest.approx(0.6), "bow": 0.0,
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
    # Not the SVG's nodes and edges, or their tooltips
    drawing = html.split('class="ec-legend"')[0]
    assert "<circle" not in drawing and "<line" not in drawing and "data-attrs" not in html
    # Each plot's root is its own
    ids = re.findall(r'<div class="ec-viewer" id="([^"]+)"', html)
    again = g.plot(backend="viewer", seed=1, color_by="c")._repr_html_()
    assert len(ids) == 1 and ids != re.findall(r'<div class="ec-viewer" id="([^"]+)"', again)
    with pytest.raises(ValueError, match="use one of 'auto', 'svg', 'png', 'viewer'"):
        g.plot(backend="webgl")


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
        bg = svg_renderer.LIGHT.bg if given == "light" else svg_renderer.DARK.bg
        assert f"background:{bg};" in html


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


def test_the_viewers_code_is_ascii():
    # It is inlined in pages that may not say how they are encoded
    js = (Path(svg_renderer.__file__).parent / "static" / "viewer.js").read_bytes()
    assert js.isascii() and b"export function mount(" in js and b"export function parse(" in js
