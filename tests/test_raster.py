"""Tests for `Scene.png()`, the scene drawn as a raster image, in full color or in a palette, with
an overlay drawn from SVG, for the layers that `_raster_layers()` makes of it for each theme, and
for what `plot()` makes of them: `backend="png"`, and a plot's PNG image.

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
    """The width, height, channels and rows of pixels of an 8-bit PNG. An indexed one comes out as
    RGBA, its entries looked up."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, data, palette, alphas = 8, b"", [], b""
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind, body = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and color in (0, 2, 3, 4, 6)
        elif kind == b"PLTE":
            palette = [tuple(body[i : i + 3]) for i in range(0, len(body), 3)]
        elif kind == b"tRNS":
            alphas = body
        elif kind == b"IDAT":
            data += body
        pos += 12 + length
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
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
    if color == 3:
        entries = [(*rgb, alphas[i] if i < len(alphas) else 255) for i, rgb in enumerate(palette)]
        rows = [bytearray(b"".join(bytes(entries[i]) for i in row)) for row in rows]
        channels = 4
    return width, height, channels, rows


def _color_type(png):
    return png[25]


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
    png = scene.edge_mask(scale=1.0)
    assert _color_type(png) == 3
    mask = _decode(png)
    edges = _decode(scene.png(LIGHTS, scale=1.0, nodes=False))
    assert [px[3] for px in _pixels(mask)] == [px[3] for px in _pixels(edges)]
    assert {px[:3] for px in _pixels(mask) if px[3]} == {(255, 255, 255)}
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


def test_a_palette_keeps_few_colors():
    # The small scene has fewer than 256 colors, which the palette keeps as they are
    for kwargs in ({}, {"bg": BG}, {"nodes": False}):
        png = _scene((0, 2)).png(LIGHTS, scale=1.0, palette=True, **kwargs)
        assert _color_type(png) == 3
        full = _decode(_scene((0, 2)).png(LIGHTS, scale=1.0, **kwargs))
        want = [px if len(px) == 4 else (*px, 255) for px in _pixels(full)]
        assert _pixels(_decode(png)) == [px if px[3] else (0, 0, 0, 0) for px in want]


def test_a_palette_comes_near_more_colors():
    g = ec.karate_club()
    scene = svg_renderer.build_scene(g, layout="circular", show_labels=False, color_by="degree")
    hexes = [light for light, _ in scene.colors]
    full = _decode(scene.png(hexes, scale=1.0, edges=False))
    assert len(set(_pixels(full))) > 256
    png = scene.png(hexes, scale=1.0, edges=False, palette=True)
    assert png == scene.png(hexes, scale=1.0, edges=False, palette=True)
    (length,) = struct.unpack(">I", png[33:37])
    assert png[37:41] == b"PLTE" and length <= 3 * 256
    # Over the light background, the mean difference is a fraction of a level
    bg = _rgb(BG)
    diffs = []
    for p, q in zip(_pixels(full), _pixels(_decode(png))):
        for c in range(3):
            seen = [(px[c] * px[3] + bg[c] * (255 - px[3])) / 255 for px in (p, q)]
            diffs.append(abs(seen[0] - seen[1]))
    assert sum(diffs) / len(diffs) < 0.05 and max(diffs) < 24


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
        png = base64.b64decode(href)
        assert _color_type(png) == 3
        found.append((tag, _decode(png)))
    return found


def test_edges_of_one_color_are_a_mask_in_it():
    scene = _scene((0, 0))
    for theme, fill in (("light", EDGE), ("dark", EDGE_DARK), ("auto", "var(--ect-")):
        colors = svg_renderer._Colors(theme)
        edges, nodes = svg_renderer._raster_layers(scene, colors, scale=1.0)
        mask_id = re.search(r'<mask id="(ec-edges-[0-9a-f]{10})">', edges).group(1)
        assert f'fill="{fill}' in edges and f'mask="url(#{mask_id})"' in edges
        # The mask, then the nodes (which look the same in either theme)
        ((mask_tag, mask),), ((_, nodes_image),) = _images(edges), _images(nodes)
        assert {px[:3] for px in _pixels(mask) if px[3]} == {(255, 255, 255)}
        assert _pixel(nodes_image, 20, 50) == (*_rgb(FILL), 255)
        assert 'x="0.000" y="0.000" width="200.000" height="100.000"' in mask_tag
        assert "display" not in edges + nodes
    layers = svg_renderer._raster_layers(scene, svg_renderer._Colors("dark"))
    assert "var(" not in "".join(layers)


def test_edges_of_more_colors_are_an_image_for_each_theme():
    scene = _scene((0, 2))
    colors = svg_renderer._Colors("auto")
    edges, nodes = svg_renderer._raster_layers(scene, colors, scale=1.0)
    assert "<mask" not in edges
    (light_tag, light), (dark_tag, dark) = _images(edges)
    # Each shows in its own theme
    shown = colors.pick("inline", "none")
    assert f'style="display:{shown}"' in light_tag
    assert f'style="display:{colors.pick("none", "inline")}"' in dark_tag
    ((nodes_tag, _),) = _images(nodes)
    assert "style" not in nodes_tag
    assert _pixel(light, 100, 50)[:3] == _rgb(EDGE)
    assert _pixel(dark, 100, 50)[:3] == _rgb(EDGE_DARK)
    # In a theme of its own, one image
    for theme, want in (("light", OTHER), ("dark", OTHER_DARK)):
        edges, nodes = svg_renderer._raster_layers(scene, svg_renderer._Colors(theme), scale=1.0)
        ((_, image),) = _images(edges)
        assert _pixel(image, 60, 67)[:3] == _rgb(want)
        assert "display" not in edges + nodes


def test_no_edges_is_an_empty_layer():
    scene = _scene(())
    edges, nodes = svg_renderer._raster_layers(scene, svg_renderer._Colors("light"), scale=1.0)
    assert edges == "" and len(_images(nodes)) == 1


# An SVG of the scene's frame for an overlay: a rectangle over its left half, and nothing else
def _overlay(fill="#000000", opacity=1.0, body=""):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 200 100">'
        f'<rect width="100" height="100" fill="{fill}" fill-opacity="{opacity}"/>{body}</svg>'
    )


def test_the_overlay_goes_over_the_rest():
    scene = _scene()
    plain = _decode(scene.png(LIGHTS, bg=BG, scale=1.0))
    covered = _decode(scene.png(LIGHTS, bg=BG, scale=1.0, overlay=_overlay("#00ff00")))
    # Over the node at the left, and nothing at the right
    assert _pixel(covered, 20, 50) == (0, 255, 0)
    assert _pixels(covered)[0] == (0, 255, 0)
    assert [_pixel(covered, x, y) for x in range(101, 200) for y in range(100)] == [
        _pixel(plain, x, y) for x in range(101, 200) for y in range(100)
    ]
    # At its opacity, over the background
    half = _decode(scene.png(LIGHTS, bg=BG, scale=1.0, overlay=_overlay("#000000", 0.5)))
    assert all(abs(c - b / 2) <= 1 for c, b in zip(_pixel(half, 5, 5), _rgb(BG)))
    # Over nothing, its color as it is
    clear = _decode(scene.png(LIGHTS, scale=1.0, overlay=_overlay("#3366cc", 0.5)))
    r, g, b, a = _pixel(clear, 5, 5)
    assert abs(a - 128) <= 1 and all(abs(c - w) <= 2 for c, w in zip((r, g, b), _rgb(EDGE)))


def test_the_overlay_is_stretched_to_the_image():
    scene = _scene()
    image = _decode(scene.png(LIGHTS, bg=BG, scale=2.0, overlay=_overlay("#00ff00")))
    assert _pixel(image, 199, 150) == (0, 255, 0) and _pixel(image, 201, 150) != (0, 255, 0)


def test_the_overlay_draws_text():
    scene = _scene(())
    text = '<text x="110" y="40" font-size="20" fill="#000">Hi</text>'
    plain = _decode(scene.png(LIGHTS, bg=BG, scale=1.0, overlay=_overlay(opacity=0)))
    image = _decode(scene.png(LIGHTS, bg=BG, scale=1.0, overlay=_overlay(opacity=0, body=text)))
    dark = [px for px in _pixels(image) if sum(px) < 200]
    assert len(dark) > 20 and not any(sum(px) < 200 for px in _pixels(plain))


def test_the_overlay_must_be_svg():
    with pytest.raises(ValueError, match="can't read the overlay's SVG"):
        _scene().png(LIGHTS, overlay="<svg")


def _size(png):
    return struct.unpack(">II", png[16:24])


def test_the_png_backend_draws_images_under_text():
    g = ec.les_miserables().k_core(k=6).cmty_louvain(attr_name="c")
    opts = {"seed": 1, "color_by": "c", "edge_label_by": "weight"}
    vector, raster = g.plot(**opts), g.plot(backend="png", **opts)
    assert "<circle" in vector.svg and "<image" not in vector.svg
    drawing = raster.svg.split('class="ec-legend"')[0]
    assert "<circle" not in drawing and "<line" not in drawing
    assert len(_images(raster.svg)) >= 2
    # The labels, edge labels and legend stay text
    texts = re.findall(r"<text[^>]*>([^<]*)<", raster.svg)
    assert texts == re.findall(r"<text[^>]*>([^<]*)<", vector.svg)
    assert 'class="ec-legend"' in raster.svg
    # No tooltips
    assert "data-attrs" not in raster.svg and "<script" not in raster._repr_html_()
    with pytest.raises(ValueError, match="Unknown backend 'jpeg'; use one of 'auto', 'svg', 'png'"):
        g.plot(backend="jpeg")


def test_curved_edge_labels_have_their_paths():
    # The edges are in the image, so each label's path is drawn on its own
    g = ec.les_miserables().k_core(k=6)
    svg = g.plot(backend="png", curved_edges=True, edge_label_by="weight", layout="circular").svg
    refs = re.findall(r'<textPath href="#([^"]+)"', svg)
    assert refs and set(refs) <= set(re.findall(r'<path id="([^"]+)"', svg))


def test_the_png_backend_has_smaller_nodes_and_no_size_limit(monkeypatch):
    g = ec.Graph().add_nodes(list(range(3000)))
    radii = {}
    for backend in ("svg", "png"):
        scene = svg_renderer.build_scene(g, layout="random", seed=1, backend=backend)
        svg, _ = svg_renderer.emit_svg(scene, svg_renderer._Colors("light"))
        radii[backend] = {float(r) for r in re.findall(r'<circle [^>]*r="([0-9.]+)"', svg)}
    assert radii["svg"] == {3.5} and max(radii["png"]) < 2.5
    monkeypatch.setattr(svg_renderer, "PLOT_BYTES_MAX", 1000)
    with pytest.raises(ValueError, match=r"past its limit of 0 MB\. Plot it with backend='png'"):
        ec.karate_club().plot(backend="svg")
    ec.karate_club().plot(backend="png")


def test_the_warning_suggests_the_png_backend(monkeypatch):
    monkeypatch.setattr(svg_renderer, "NOTEBOOK_BYTES_MAX", 1000)
    with pytest.warns(UserWarning, match="backend='png' makes it smaller"):
        ec.karate_club().plot(seed=1, backend="svg")
    with pytest.warns(UserWarning, match=r"MB a notebook output .*\. Plotting a part"):
        ec.karate_club().plot(seed=1, backend="png")


def _radii(markup):
    return {float(r) for r in re.findall(r'<circle [^>]*r="([0-9.]+)"', markup)}


def test_auto_picks_the_scene_by_its_svg(monkeypatch):
    g = ec.Graph().add_nodes(list(range(3000)))
    given, radii = [], {}

    def draw(scene):
        return svg_renderer.emit_svg(scene, svg_renderer._Colors("light"))[0]

    for answer in (True, False):

        def fits(scene):
            given.append(_radii(draw(scene)))
            return answer

        scene = svg_renderer.build_scene(g, layout="random", seed=1, backend="auto", fits=fits)
        radii[answer] = _radii(draw(scene))
    # It is asked about the SVG's scene, and gives that or the images' (with smaller nodes)
    assert given == [{3.5}, {3.5}]
    assert radii[True] == {3.5} and max(radii[False]) < 2.5
    # It isn't asked when the SVG would surely be too large
    monkeypatch.setattr(svg_renderer, "NOTEBOOK_BYTES_MAX", 1000)
    scene = svg_renderer.build_scene(g, layout="random", seed=1, backend="auto", fits=pytest.fail)
    assert _radii(draw(scene)) == radii[False]


def _unnumbered(html):
    return re.sub(r"ec-tooltip-\d+", "ec-tooltip", html)


def _parallel_edges():
    # 20,000 edges between 50 nodes, whose SVG takes at least 0.8 MB, and is about 1.8 MB
    pairs = [(k % 50, (k * 7 + 1) % 50) for k in range(20000)]
    return ec.Graph.from_edges([(a, b) for a, b in pairs if a != b])


@pytest.mark.parametrize("budget", [None, 1_000_000])
def test_auto_is_svg_within_the_budget_and_images_past_it(monkeypatch, budget):
    g = _parallel_edges()
    if budget:
        monkeypatch.setattr(svg_renderer, "NOTEBOOK_BYTES_MAX", budget)
    opts = {"seed": 1, "show_labels": False}
    auto = g.plot(**opts)
    picked = g.plot(backend="png" if budget else "svg", **opts)
    assert ("<image" in auto._repr_html_()) == bool(budget)
    # As the backend picked draws it, but for the numbering of the tooltips on the page
    assert _unnumbered(auto._repr_html_()) == _unnumbered(picked._repr_html_())
    assert auto._repr_png_()[0] == picked._repr_png_()[0]


def test_the_svg_of_images_is_drawn_when_wanted(monkeypatch, tmp_path):
    monkeypatch.setattr(svg_renderer, "NOTEBOOK_BYTES_MAX", 1_000_000)
    g = _parallel_edges()
    for theme in ("light", None):
        auto = g.plot(seed=1, show_labels=False, theme=theme)
        vector = g.plot(seed=1, show_labels=False, theme=theme, backend="svg")
        html = auto._repr_html_()
        # The SVG that the images stand in for, with a note on its size
        with pytest.warns(UserWarning, match=r"The SVG is \d MB, which is why") as caught:
            auto.save(tmp_path / "auto.svg")
        assert caught[0].filename == __file__
        vector.save(tmp_path / "vector.svg")
        saved = (tmp_path / "auto.svg").read_text()
        assert saved == (tmp_path / "vector.svg").read_text() and "<image" not in saved
        assert auto._repr_html_() == html
    assert auto.svg.startswith('<svg class="ec-theme"') and "<image" not in auto.svg
    # It is too large
    auto = g.plot(seed=1, show_labels=False)
    monkeypatch.setattr(svg_renderer, "PLOT_BYTES_MAX", 1000)
    with pytest.raises(ValueError, match=r"The plot's SVG would take at least 1 MB .*\.png"):
        auto.svg
    # Asked for, the images stay in the SVG
    assert "<image" in g.plot(seed=1, show_labels=False, backend="png").svg


@pytest.mark.parametrize("backend", ["svg", "png"])
def test_a_plot_is_also_a_png(backend):
    plot = ec.karate_club().plot(seed=1, width=300, height=200, backend=backend)
    png, meta = plot._repr_png_()
    # At twice the size, shown at the plot's
    assert meta == {"width": 300, "height": 200} and _size(png) == (600, 400)
    assert _color_type(png) == 3
    assert plot._repr_png_()[0] is png
    image = _decode(png)
    # The background, a node, and a label in the label color
    assert _pixel(image, 1, 1)[:3] == _rgb(svg_renderer.LIGHT.bg)
    label = _rgb(svg_renderer.LIGHT.label_color)
    near = [px for px in _pixels(image) if sum(abs(a - b) for a, b in zip(px, label)) < 40]
    assert len(near) > 50
    # The notebook keeps the HTML and the PNG, not the SVG
    assert not hasattr(plot, "_repr_svg_")


def test_the_png_is_in_the_theme():
    for theme, bg in ((None, "light"), ("auto", "light"), ("light", "light"), ("dark", "dark")):
        png = ec.karate_club().plot(seed=1, theme=theme)._repr_png_()[0]
        want = getattr(svg_renderer, bg.upper()).bg
        assert _pixel(_decode(png), 0, 0)[:3] == _rgb(want), theme
    # The legend, too
    light = ec.karate_club().plot(seed=1, color_by="faction", theme="light")._repr_png_()[0]
    plain = ec.karate_club().plot(seed=1, color_by="faction", legend=False, theme="light")
    assert light != plain._repr_png_()[0]


def test_save_follows_the_extension(tmp_path):
    plot = ec.karate_club().plot(seed=1, backend="png")
    for name in ("a.svg", "a.html", "a.htm", "a.png", "A.PNG"):
        plot.save(tmp_path / name)
    assert (tmp_path / "a.svg").read_text().startswith("<svg")
    assert (tmp_path / "a.html").read_text() == (tmp_path / "a.htm").read_text()
    assert (tmp_path / "a.html").read_text().startswith("<div")
    assert (tmp_path / "a.png").read_bytes() == plot._repr_png_()[0]
    assert (tmp_path / "A.PNG").read_bytes() == plot._repr_png_()[0]
    ec.karate_club().plot(seed=1, output=str(tmp_path / "b.png"))
    assert (tmp_path / "b.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for name in ("a.pdf", "a"):
        with pytest.raises(ValueError, match="the extension must be .html, .svg or .png"):
            plot.save(tmp_path / name)
    # A player has no image
    player = ec.karate_club().plot_traversal(source=1)
    assert player._repr_png_() is None
    with pytest.raises(ValueError, match="Only a plot.. can be saved as .png"):
        player.save(tmp_path / "p.png")


def test_a_saved_file_leaves_out_the_other_theme():
    # Edges of more than one color are an image for each theme, and a file is in the light one
    g = ec.karate_club().cmty_louvain(attr_name="c")
    plot = g.plot(seed=1, color_by="c", edge_color_by="c", backend="png")
    hidden = re.findall(r'<image [^>]*style="display:', plot.svg)
    assert len(hidden) == 2
    svg, html = plot._for_files
    for markup in (svg, html):
        assert "display:none" not in markup and len(_images(markup)) == 2


def test_an_empty_graph_is_a_png():
    png, meta = ec.Graph().plot()._repr_png_()
    assert meta == {"width": 700, "height": 500} and _size(png) == (1400, 1000)
    # "Empty graph", in the middle
    image = _decode(png)
    bg = _rgb(svg_renderer.LIGHT.bg)
    assert any(_pixel(image, x, 490)[:3] != bg for x in range(600, 800))
