"""Tests for `Scene.png()`, the scene drawn as a raster image.

The images are read back with a small PNG decoder of the standard library's, so that the tests
need nothing else installed.
"""

import struct
import zlib

import edgecase as ec
import edgecase.svg_renderer as svg_renderer
import pytest
from edgecase._core import Scene

BG = "#fafafa"
EDGE = "#3366cc"
FILL = "#cc3333"


def _decode(png):
    """The width, height, channels and rows of pixels of a PNG of 8-bit RGB or RGBA."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, data = 8, b""
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind, body = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and color in (2, 6)
        elif kind == b"IDAT":
            data += body
        pos += 12 + length
    channels = 3 if color == 2 else 4
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


def _rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _scene(opacity=0.6):
    # Two nodes at the ends of a frame of 200 by 100, with an edge 4 wide between them
    scene = Scene(200, 100, viewbox=(0.0, 0.0, 200.0, 100.0), colors=[("", "")] * 2)
    scene.set_nodes(
        ["a", "b"], [20.0, 180.0], [50.0, 50.0], [10.0, 10.0], [1, 1], [1, 1], [0.0, 0.0],
        [1.0, 1.0], [0, 1], ["{}", "{}"],
    )  # fmt: skip
    scene.set_edges([0], [1], [0], [4.0], [opacity], [0.0], [1.0], [None], [0], ["{}"])
    return scene


def test_the_image_is_the_frame_at_scale():
    for scale, size in ((1.0, (200, 100)), (2.0, (400, 200)), (0.5, (100, 50))):
        width, height, channels, _ = _decode(_scene().png([EDGE, FILL], bg=BG, scale=scale))
        assert (width, height, channels) == (*size, 3)


def test_the_edges_and_nodes_have_their_colors():
    image = _decode(_scene().png([EDGE, FILL], bg=BG, scale=1.0, tone="over"))
    assert _pixel(image, 2, 2) == _rgb(BG)
    assert _pixel(image, 20, 50) == _rgb(FILL)
    # The middle of the edge, at its opacity over the background, as the SVG draws it
    mixed = tuple(round(0.6 * e + 0.4 * b) for e, b in zip(_rgb(EDGE), _rgb(BG)))
    assert all(abs(p - m) <= 1 for p, m in zip(_pixel(image, 100, 50), mixed))
    # Off the edge, nothing
    assert _pixel(image, 100, 40) == _rgb(BG)


def test_no_background_is_transparent():
    image = _decode(_scene().png([EDGE, FILL], scale=1.0))
    assert image[2] == 4
    assert _pixel(image, 2, 2)[3] == 0
    assert _pixel(image, 20, 50) == (*_rgb(FILL), 255)
    r, g, b, a = _pixel(image, 100, 50)
    assert 0 < a < 255 and (r, g, b) == _rgb(EDGE)


def test_a_lone_edge_looks_the_same_in_every_tone():
    # The tones differ only where edges pile up
    images = [_scene().png([EDGE, FILL], bg=BG, tone=tone) for tone in ("over", "log", "eq_hist")]
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
    with pytest.raises(ValueError, match="colors has 1 entries for 2 colors in the table"):
        scene.png([EDGE])
    with pytest.raises(ValueError, match="not a hex color"):
        scene.png([EDGE, "red"])
    with pytest.raises(ValueError, match="not a hex color"):
        scene.png([EDGE, FILL], bg="#12345")
    with pytest.raises(ValueError, match="tone must be 'over', 'log' or 'eq_hist', not 'linear'"):
        scene.png([EDGE, FILL], tone="linear")
    for scale in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="scale must be greater than 0"):
            scene.png([EDGE, FILL], scale=scale)
    with pytest.raises(ValueError, match="makes an image of 200000×100000 pixels; the most is"):
        scene.png([EDGE, FILL], scale=1000.0)
    # Short hex colors are read as CSS reads them
    assert scene.png(["#36c", "#c33"], bg="#fafafa") == scene.png([EDGE, FILL], bg=BG)
