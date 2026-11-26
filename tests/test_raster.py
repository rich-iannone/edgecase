"""Tests for `Scene.png()`, the scene drawn as a raster image, and for the layers that
`_raster_layers()` makes of it for each theme.

The images are read back with a small PNG decoder of the standard library's, so that the tests
need nothing else installed.
"""

import base64
import re
import struct
import zlib

import edgecase as ec
import edgecase.svg_renderer as svg_renderer
import pytest
from edgecase._core import Scene

BG = "#fafafa"
EDGE, EDGE_DARK = "#3366cc", "#88aaee"
FILL = "#cc3333"
OTHER, OTHER_DARK = "#22aa55", "#66dd99"
# The scene's color table, and its colors in the light theme
PAIRS = [(EDGE, EDGE_DARK), (FILL, FILL), (OTHER, OTHER_DARK)]
LIGHTS = [light for light, _ in PAIRS]


def _decode(png):
    """The width, height, channels and rows of pixels of an 8-bit PNG without a palette."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, data = 8, b""
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind, body = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and color in (0, 2, 4, 6)
        elif kind == b"IDAT":
            data += body
        pos += 12 + length
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw, stride = zlib.decompress(data), width * channels
    rows, prev = [], bytearray(stride)
    for y in range(height):
        kind = raw[y * (stride + 1)]
        row = bytearray(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
        for i in range(stride):
            a = row[i - channels] if i >= channels else 0
            b, c = prev[i], prev[i - channels] if i >= channels else 0
            if kind == 1:
                row[i] = (row[i] + a) & 0xFF
            elif kind == 2:
                row[i] = (row[i] + b) & 0xFF
            elif kind == 3:
                row[i] = (row[i] + (a + b) // 2) & 0xFF
            elif kind == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        rows.append(row)
        prev = row
    return width, height, channels, rows


def _pixel(image, x, y):
    _, _, channels, rows = image
    return tuple(rows[y][x * channels : (x + 1) * channels])


def _pixels(image):
    width, height, _, _ = image
    return [_pixel(image, x, y) for y in range(height) for x in range(width)]


def _rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _scene(edge_colors=(0,), opacity=0.6, viewbox=(0.0, 0.0, 200.0, 100.0)):
    # Nodes at the ends of a frame of 200 by 100 and one under its middle, all in FILL, with an
    # edge 4 wide from end to end, and from the middle node to each end for more colors
    scene = Scene(200, 100, viewbox=viewbox, colors=PAIRS)
    scene.set_nodes(
        ["a", "b", "c"], [20.0, 180.0, 100.0], [50.0, 50.0, 85.0], [10.0] * 3, [1] * 3, [1] * 3,
        [0.0] * 3, [1.0] * 3, [0, 1, 2], ["{}"] * 3,
    )  # fmt: skip
    ends = [(0, 1), (2, 0), (2, 1)][: len(edge_colors)]
    m = len(ends)
    scene.set_edges(
        [s for s, _ in ends], [t for _, t in ends], list(edge_colors), [4.0] * m, [opacity] * m,
        [0.0] * m, [1.0] * m, [None] * m, list(range(m)), ["{}"] * m,
    )  # fmt: skip
    return scene


def test_the_image_is_the_frame_at_scale():
    for scale, size in ((1.0, (200, 100)), (2.0, (400, 200)), (0.5, (100, 50))):
        width, height, channels, _ = _decode(_scene().png(LIGHTS, bg=BG, scale=scale))
        assert (width, height, channels) == (*size, 3)


def test_the_edges_and_nodes_have_their_colors():
    image = _decode(_scene().png(LIGHTS, bg=BG, scale=1.0, tone="over"))
    assert _pixel(image, 2, 2) == _rgb(BG)
    assert _pixel(image, 20, 50) == _rgb(FILL)
    # The middle of the edge, at its opacity over the background, as the SVG draws it
    mixed = tuple(round(0.6 * e + 0.4 * b) for e, b in zip(_rgb(EDGE), _rgb(BG)))
    assert all(abs(p - m) <= 1 for p, m in zip(_pixel(image, 100, 50), mixed))
    # Off the edge, nothing
    assert _pixel(image, 100, 40) == _rgb(BG)


def test_no_background_is_transparent():
    image = _decode(_scene().png(LIGHTS, scale=1.0))
    assert image[2] == 4
    assert _pixel(image, 2, 2)[3] == 0
    assert _pixel(image, 20, 50) == (*_rgb(FILL), 255)
    r, g, b, a = _pixel(image, 100, 50)
    assert 0 < a < 255 and (r, g, b) == _rgb(EDGE)


def test_the_layers_draw_apart():
    edges = _decode(_scene().png(LIGHTS, scale=1.0, nodes=False))
    nodes = _decode(_scene().png(LIGHTS, scale=1.0, edges=False))
    assert _pixel(edges, 20, 50)[3] < 255 and _pixel(edges, 100, 50)[3] > 0
    assert _pixel(nodes, 20, 50) == (*_rgb(FILL), 255) and _pixel(nodes, 100, 50)[3] == 0
    # The nodes over the edges over the background are the whole drawn at once, but for the
    # rounding of each layer to bytes
    whole = _decode(_scene().png(LIGHTS, bg=BG, scale=1.0))
    for e, n, w in zip(_pixels(edges), _pixels(nodes), _pixels(whole)):
        under = [(c * e[3] + b * (255 - e[3])) / 255 for c, b in zip(e, _rgb(BG))]
        over = [(c * n[3] + u * (255 - n[3])) / 255 for c, u in zip(n, under)]
        assert all(abs(o - p) <= 1.5 for o, p in zip(over, w))


def test_the_mask_is_the_edges_opacity():
    scene = _scene()
    mask = _decode(scene.edge_mask(scale=1.0))
    assert mask[2] == 2
    edges = _decode(scene.png(LIGHTS, scale=1.0, nodes=False))
    assert [px[1] for px in _pixels(mask)] == [px[3] for px in _pixels(edges)]
    assert {px[0] for px in _pixels(mask)} == {255}
    # The same, whatever the colors
    other = _decode(scene.png([OTHER, OTHER, OTHER], scale=1.0, nodes=False))
    assert [px[3] for px in _pixels(other)] == [px[3] for px in _pixels(edges)]


def test_the_tint_is_the_one_color_of_every_edge():
    assert _scene((0,)).edge_tint == 0
    assert _scene((0, 0, 0)).edge_tint == 0
    assert _scene((0, 2)).edge_tint is None
    assert _scene(()).edge_tint is None
    # Two entries of the table with the same colors are one color
    scene = Scene(200, 100, colors=[*PAIRS, PAIRS[0]])
    scene.set_nodes(["a", "b"], [0.0, 1.0], [0.0, 1.0], [5.0] * 2, [1] * 2, [1] * 2, [0.0] * 2,
                    [1.0] * 2, [0, 1], ["{}"] * 2)  # fmt: skip
    scene.set_edges([0, 1], [1, 0], [0, 3], [1.0] * 2, [0.5] * 2, [0.0] * 2, [1.0] * 2,
                    [None] * 2, [0, 1], ["{}"] * 2)  # fmt: skip
    assert scene.edge_tint == 0
    # A color with an alpha of its own isn't a tint, since a mask would put its alpha on the
    # density rather than on each edge
    scene = Scene(200, 100, colors=[("#3366cc80", EDGE_DARK), (FILL, FILL)])
    scene.set_nodes(["a", "b"], [0.0, 1.0], [0.0, 1.0], [5.0] * 2, [1] * 2, [1] * 2, [0.0] * 2,
                    [1.0] * 2, [0, 1], ["{}"] * 2)  # fmt: skip
    scene.set_edges([0], [1], [0], [1.0], [0.5], [0.0], [1.0], [None], [0], ["{}"])
    assert scene.edge_tint is None


def test_the_box_is_the_frame_in_the_viewbox():
    assert _scene().raster_box(1.0) == (0.0, 0.0, 200.0, 100.0)
    # A viewBox of another shape is fitted in the frame and centered, as SVG places it, so the
    # frame reaches past it on its shorter side
    x, y, w, h = _scene(viewbox=(10.0, 20.0, 100.0, 100.0)).raster_box(2.0)
    assert (x, y, w, h) == pytest.approx((-40.0, 20.0, 200.0, 100.0))


def test_colors_are_read_as_css_reads_them():
    scene = _scene((0, 2))
    want = scene.png(LIGHTS, bg=BG)
    assert scene.png(["#36c", "rgb(204, 51, 51)", OTHER], bg="rgb(250 250 250)") == want
    assert scene.png([EDGE, FILL, "black"], bg=BG) == scene.png([EDGE, FILL, "#000"], bg=BG)
    # A color's alpha goes on the opacity of what it colors, as in SVG
    half = _scene(opacity=1.0).png(["#3366cc80", FILL, OTHER], bg=BG, tone="over")
    assert half == _scene(opacity=128 / 255).png(LIGHTS, bg=BG, tone="over")


def test_a_lone_edge_looks_the_same_in_every_tone():
    # The tones differ only where edges pile up
    images = [_scene().png(LIGHTS, bg=BG, tone=tone) for tone in ("over", "log", "eq_hist")]
    assert images[0] == images[1] == images[2]


def test_the_image_is_the_same_every_time():
    g = ec.karate_club()
    scene = svg_renderer.build_scene(g, layout="circular", show_labels=False)
    colors = svg_renderer._Colors("light")
    hexes = [colors.pick(*pair) for pair in scene.colors]
    for tone in ("over", "log", "eq_hist"):
        png = scene.png(hexes, bg=colors.theme.bg, tone=tone)
        assert png == scene.png(hexes, bg=colors.theme.bg, tone=tone)
        # The size, from the header
        assert png[12:16] == b"IHDR"
        size = struct.unpack(">II", png[16:24])
        assert size == (round(scene.width * 2), round(scene.height * 2))


def test_png_checks_its_arguments():
    scene = _scene()
    with pytest.raises(ValueError, match="colors has 1 entries for 3 colors in the table"):
        scene.png([EDGE])
    with pytest.raises(ValueError, match="can't read the color 'var.--ect-1.' for a raster"):
        scene.png([EDGE, "var(--ect-1)", OTHER])
    with pytest.raises(ValueError, match="can't read the color '#12345'"):
        scene.png(LIGHTS, bg="#12345")
    with pytest.raises(ValueError, match="bg must be opaque, not '#fafafa80'"):
        scene.png(LIGHTS, bg="#fafafa80")
    with pytest.raises(ValueError, match="tone must be 'over', 'log' or 'eq_hist', not 'linear'"):
        scene.png(LIGHTS, tone="linear")
    with pytest.raises(ValueError, match="tone must be"):
        scene.edge_mask(tone="linear")
    for scale in (0.0, -1.0, float("nan")):
        for call in (lambda: scene.png(LIGHTS, scale=scale), lambda: scene.edge_mask(scale)):
            with pytest.raises(ValueError, match="scale must be greater than 0"):
                call()
        with pytest.raises(ValueError, match="scale must be greater than 0"):
            scene.raster_box(scale)
    with pytest.raises(ValueError, match="makes an image of 200000×100000 pixels; the most is"):
        scene.png(LIGHTS, scale=1000.0)


def _images(markup):
    """The images in `markup`, decoded, with what is in their tags apart from them."""
    found = []
    for tag, href in re.findall(r'(<image [^>]*?)href="data:image/png;base64,([^"]*)"', markup):
        found.append((tag, _decode(base64.b64decode(href))))
    return found


def test_edges_of_one_color_are_a_mask_in_it():
    scene = _scene((0, 0))
    for theme, fill in (("light", EDGE), ("dark", EDGE_DARK), ("auto", "var(--ect-")):
        colors = svg_renderer._Colors(theme)
        markup = svg_renderer._raster_layers(scene, colors, scale=1.0)
        mask_id = re.search(r'<mask id="(ec-edges-[0-9a-f]{10})">', markup).group(1)
        assert f'fill="{fill}' in markup and f'mask="url(#{mask_id})"' in markup
        # The mask, then the nodes (which look the same in either theme)
        (mask_tag, mask), (_, nodes) = _images(markup)
        assert mask[2] == 2 and nodes[2] == 4
        assert 'x="0.000" y="0.000" width="200.000" height="100.000"' in mask_tag
        assert "display" not in markup
    assert "var(" not in svg_renderer._raster_layers(scene, svg_renderer._Colors("dark"))


def test_edges_of_more_colors_are_an_image_for_each_theme():
    scene = _scene((0, 2))
    colors = svg_renderer._Colors("auto")
    markup = svg_renderer._raster_layers(scene, colors, scale=1.0)
    assert "<mask" not in markup
    (light_tag, light), (dark_tag, dark), (nodes_tag, _) = _images(markup)
    # Each shows in its own theme
    shown = colors.pick("inline", "none")
    assert f'style="display:{shown}"' in light_tag
    assert f'style="display:{colors.pick("none", "inline")}"' in dark_tag
    assert "style" not in nodes_tag
    assert _pixel(light, 100, 50)[:3] == _rgb(EDGE)
    assert _pixel(dark, 100, 50)[:3] == _rgb(EDGE_DARK)
    # In a theme of its own, one image
    for theme, want in (("light", OTHER), ("dark", OTHER_DARK)):
        markup = svg_renderer._raster_layers(scene, svg_renderer._Colors(theme), scale=1.0)
        (_, edges), _ = _images(markup)
        assert _pixel(edges, 60, 67)[:3] == _rgb(want)
        assert "display" not in markup
