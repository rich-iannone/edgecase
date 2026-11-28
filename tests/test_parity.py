"""Parity tests: fixed scenes and views drawn by the viewer, by the PNG backend and as SVG, compared
pixel by pixel, so that the three keep to the visual model of `src/scene/VISUAL_MODEL.md`.

The viewer is drawn in headless Chromium on SwiftShader, the software GPU, so that it draws the
same anywhere; the SVG is rasterized by the same browser. Each pair of images must be within a
tolerance of each other: a mean difference (of 255, over the color channels) over the pixels
that either draws, and a share of all the pixels that differ by more than `OFF`. The labels the
viewer chooses must be those of the SVG.

The tests need playwright and its Chromium (`pip install edgecase[parity]`, then
`playwright install chromium`) and are skipped without them, unless `EDGECASE_PARITY_REQUIRED`
is set, as it is in CI. On a failure the images and a diff of each pair that failed (red where
they differ) are written to `EDGECASE_PARITY_OUT`, or to the test's temporary directory.
"""

import base64
import os
import re
from pathlib import Path

import edgecase as ec
import pytest

# Without playwright or its Chromium, fail rather than skip
REQUIRED = bool(os.environ.get("EDGECASE_PARITY_REQUIRED"))
if REQUIRED:
    from playwright import sync_api
else:
    sync_api = pytest.importorskip("playwright.sync_api")

# The plot's size, and the pixels per unit: the PNG backend's `RASTER_SCALE`, and the browser's
# device pixel ratio
WIDTH, HEIGHT, SCALE = 700, 500, 2
FLAGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
# A pixel is off where it differs by more than this, of 255
OFF = 32
# The most mean difference and share of pixels off: for the viewer against the PNG, which draw
# by the model's coverage, against the SVG, which the browser antialiases its own way, and for
# scenes with labels, whose glyphs come from different font engines
TOLERANCE = {"raster": (1.0, 1e-4), "svg": (2.0, 1e-4), "labels": (8.0, 0.01)}

# The viewer's page: without its toolbar and corner, so that only the canvases show
PAGE = """<!doctype html><meta charset="utf-8">
<style>
body { margin: 0; }
.ec-viewer { border-radius: 0 !important; }
.ec-viewer::after { display: none !important; }
.ec-viewer > :not(canvas) { visibility: hidden !important; }
</style>
"""

# Once the viewer has drawn, and its labels if it has them
DRAWN = """(labels) => {
  const v = document.querySelector("ec-graph")?.viewer;
  return !!v && !v.renderer.compiling && (!labels || v.labels.drawn.nodes.length > 0);
}"""
SETTLE = """async () => {
  for (let i = 0; i < 5; i++) await new Promise(requestAnimationFrame);
}"""
# The IDs of the nodes whose labels the viewer drew
DRAWN_LABELS = """() => {
  const v = document.querySelector("ec-graph").viewer;
  return v.labels.drawn.nodes.map((k) => String(v.data.node_id[k]));
}"""
# Two images compared: their sizes, the mean difference over the pixels either draws (that aren't
# the background of its corner), the share of pixels off, and a diff image, red where they differ
# over the first one faded
COMPARE = """async ([a, b, off]) => {
  const load = (src) => new Promise((ok, no) => {
    const img = new Image();
    img.onload = () => ok(img);
    img.onerror = no;
    img.src = src;
  });
  const [ia, ib] = await Promise.all([load(a), load(b)]);
  const sizes = [[ia.naturalWidth, ia.naturalHeight], [ib.naturalWidth, ib.naturalHeight]];
  const [w, h] = sizes[0];
  if (sizes[1][0] !== w || sizes[1][1] !== h) return { sizes };
  const pixels = (img) => {
    const ctx = new OffscreenCanvas(w, h).getContext("2d");
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, w, h).data;
  };
  const pa = pixels(ia), pb = pixels(ib);
  const diff = new ImageData(w, h), out = diff.data;
  const drawn = (p, i) => Math.abs(p[i] - p[0]) + Math.abs(p[i + 1] - p[1]) + Math.abs(p[i + 2] - p[2]) > 6;
  let sum = 0, ink = 0, offs = 0;
  for (let i = 0; i < pa.length; i += 4) {
    const d = (Math.abs(pa[i] - pb[i]) + Math.abs(pa[i + 1] - pb[i + 1]) + Math.abs(pa[i + 2] - pb[i + 2])) / 3;
    if (drawn(pa, i) || drawn(pb, i)) {
      sum += d;
      ink++;
    }
    if (d > off) offs++;
    const grey = 255 - (255 - (pa[i] + pa[i + 1] + pa[i + 2]) / 3) * 0.25, t = Math.min(d / off, 1);
    out[i] = grey + (255 - grey) * t;
    out[i + 1] = out[i + 2] = grey * (1 - t);
    out[i + 3] = 255;
  }
  const canvas = new OffscreenCanvas(w, h);
  canvas.getContext("2d").putImageData(diff, 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  const url = await new Promise((ok) => {
    const reader = new FileReader();
    reader.onload = () => ok(reader.result);
    reader.readAsDataURL(blob);
  });
  return { sizes, mean: sum / Math.max(ink, 1), share: offs / (w * h), diff: url };
}"""


def _looped():
    """A directed graph with reciprocal edges, a fan and self-loops."""
    edges = [(i, (i * 7 + 3) % 30) for i in range(30)] + [(i, i + 1) for i in range(29)]
    return ec.Graph.from_edges(edges + [(3, 3), (8, 8), (8, 8)], directed=True)


GRAPHS = {
    "karate": ec.karate_club,
    "les_miserables": ec.les_miserables,
    "looped": _looped,
    "jazz": ec.jazz,
}

# By name: the graph, the plot's arguments, and whether there is an SVG of it to compare
KARATE = {"color_by": "faction"}
CASES = {
    "plain": ("karate", {**KARATE, "show_labels": False}, True),
    "shaded": ("karate", {**KARATE, "show_labels": False, "shade": True}, True),
    "curved": ("karate", {**KARATE, "show_labels": False, "curved_edges": True}, True),
    "dark": ("karate", {**KARATE, "show_labels": False, "theme": "dark"}, True),
    "thin": (
        "karate",
        {"show_labels": False, "node_size": 1.5, "edge_width": 0.3, "view": {"zoom": 1}},
        True,
    ),
    "directed": ("looped", {"show_labels": False}, True),
    "directed_curved": ("looped", {"show_labels": False, "curved_edges": True}, True),
    # The arrowheads grow, as far as their nodes allow, at an opacity over the viewer's `HEAD_GAIN`
    # so that they don't come forward
    "directed_zoom": (
        "looped",
        {"show_labels": False, "node_size": 4, "edge_opacity": 0.9, "view": {"zoom": 6}},
        True,
    ),
    "log": ("jazz", {"show_labels": False, "edge_tone": "log"}, False),
    "labels": ("karate", KARATE, True),
    "labels_zoom": ("karate", {**KARATE, "view": {"zoom": 2}}, True),
    "highlight": ("karate", {**KARATE, "highlight": [1, 2, 3, 34]}, True),
    "filters": ("karate", {**KARATE, "filters": [{"attr": "degree", "range": (2, 10)}]}, True),
    "many_labels": ("les_miserables", {}, True),
    "many_labels_zoom": ("les_miserables", {"view": {"zoom": 2.5}}, True),
    "many_labels_filtered": (
        "les_miserables",
        {"filters": [{"attr": "degree", "range": (3, 40)}]},
        True,
    ),
}


def _url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


@pytest.fixture(scope="module")
def page():
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=FLAGS)
        except sync_api.Error as e:
            if REQUIRED:
                raise
            pytest.skip(f"Chromium won't start (run `playwright install chromium`): {e}")
        page = browser.new_page(
            viewport={"width": WIDTH + 60, "height": HEIGHT + 60}, device_scale_factor=SCALE
        )
        webgl = page.evaluate("() => !!document.createElement('canvas').getContext('webgl2')")
        assert webgl, "Chromium has no WebGL 2 on SwiftShader"
        yield page
        browser.close()


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    where = os.environ.get("EDGECASE_PARITY_OUT")
    if where:
        Path(where).mkdir(parents=True, exist_ok=True)
        return Path(where)
    return tmp_path_factory.mktemp("parity")


def draw(page, graph, kw, folder, svg=True):
    """The plot of `graph` with `kw` drawn by the viewer, by the PNG backend and (with `svg`) as
    SVG, as PNG images by name, and the IDs of the nodes whose labels the viewer drew."""
    kw = {"seed": 1, "theme": "light", "edge_routing": "straight", "legend": False, **kw}
    graph.plot(backend="viewer", **kw).save(str(folder / "viewer.html"))
    html = folder / "page.html"
    html.write_text(PAGE + (folder / "viewer.html").read_text(encoding="utf-8"), encoding="utf-8")
    page.goto(html.as_uri())
    labels = kw.get("show_labels", True)
    page.wait_for_function(DRAWN, arg=labels, timeout=30_000)
    page.evaluate(SETTLE)
    images = {"viewer": page.locator("ec-graph").screenshot()}
    drawn = page.evaluate(DRAWN_LABELS)
    graph.plot(backend="png", **kw).save(str(folder / "plot.png"))
    images["png"] = (folder / "plot.png").read_bytes()
    if svg:
        graph.plot(backend="svg", **kw).save(str(folder / "plot.svg"))
        page.goto((folder / "plot.svg").as_uri())
        images["svg"] = page.locator("svg").first.screenshot()
    return images, drawn


def compare(page, a, b):
    """The two PNG images compared, as `COMPARE` gives it."""
    page.goto("about:blank")
    return page.evaluate(COMPARE, [_url(a), _url(b), OFF])


def tolerance(a, b, labels):
    """The most mean difference and share of pixels off between the images `a` and `b`."""
    return TOLERANCE["labels" if labels else "svg" if "svg" in (a, b) else "raster"]


def failures(page, images, labels):
    """The pairs of `images` that differ by more than their tolerance, as `(a, b, comparison)`."""
    names, failed = list(images), []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = compare(page, images[a], images[b])
            assert r["sizes"] == [[WIDTH * SCALE, HEIGHT * SCALE]] * 2, f"{a}, {b}: {r['sizes']}"
            most_mean, most_share = tolerance(a, b, labels)
            if r["mean"] > most_mean or r["share"] > most_share:
                failed.append((a, b, r))
    return failed


@pytest.mark.parametrize("name", list(CASES))
def test_the_viewer_the_png_and_the_svg_draw_alike(name, page, out, tmp_path):
    graph_name, kw, svg = CASES[name]
    labels = kw.get("show_labels", True)
    images, drawn = draw(page, GRAPHS[graph_name](), kw, tmp_path, svg)
    failed = failures(page, images, labels)
    if failed:
        for which, png in images.items():
            (out / f"{name}-{which}.png").write_bytes(png)
        for a, b, r in failed:
            diff = base64.b64decode(r["diff"].partition(",")[2])
            (out / f"{name}-{a}-{b}-diff.png").write_bytes(diff)
    assert not failed, f"{name} (images in {out}): " + "; ".join(
        f"{a} and {b}, mean {r['mean']:.3f} and share {r['share']:.5f}" for a, b, r in failed
    )
    if labels and svg:
        texts = re.findall(r"<text[^>]*>([^<]*)</text>", (tmp_path / "plot.svg").read_text())
        assert texts, name
        assert sorted(drawn) == sorted(texts), f"{name}: the viewer's labels aren't the SVG's"


@pytest.mark.parametrize(
    ("name", "old", "new"),
    [
        # The shading's highlight reaches 1.6 of the radius, for 1.5
        ("shaded", "/ (1.5 * v_r), 1.0)", "/ (1.6 * v_r), 1.0)"),
        # Arrowheads 9 long, for 8
        ("directed", "float head_len = 8.0 * grow", "float head_len = 9.0 * grow"),
        # Zoomed in, arrowheads up to a fifth of their node across, for a quarter
        ("directed_zoom", "rb / (4.0 * base)", "rb / (5.0 * base)"),
    ],
)
def test_a_small_change_to_the_viewer_fails(name, old, new, page, tmp_path):
    graph_name, kw, _ = CASES[name]
    images, _ = draw(page, GRAPHS[graph_name](), kw, tmp_path, svg=False)
    assert not failures(page, images, False)
    html = tmp_path / "page.html"
    text = html.read_text(encoding="utf-8")
    assert text.count(old) == 1
    html.write_text(text.replace(old, new), encoding="utf-8")
    page.goto(html.as_uri())
    page.wait_for_function(DRAWN, arg=False, timeout=30_000)
    page.evaluate(SETTLE)
    images["viewer"] = page.locator("ec-graph").screenshot()
    assert failures(page, images, False)
