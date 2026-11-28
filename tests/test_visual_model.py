"""Tests that the visual model's shared constants, as `src/scene/VISUAL_MODEL.md` lists them,
have the same values where the Rust sources and the viewer write them.

Each constant is found by snippets of the sources, with `#` where the number is. A snippet must be
found at least once, and each number it finds must be the constant's value, so that a change on
one side alone fails here.
"""

import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "scene" / "VISUAL_MODEL.md"

SHADE, GEOMETRY, BUILDER = "src/scene/shade.rs", "src/scene/geometry.rs", "src/scene/builder.rs"
RASTER, SVG, LABELS = "src/scene/raster.rs", "src/scene/svg.rs", "src/scene/labels.rs"
PAYLOAD, VIEWER = "src/scene/payload.rs", "python/edgecase/static/viewer.js"


def _neg(v):
    return -v


def _rest(v):
    return 1 - v


# By name, the snippets that hold each constant: (file, snippet) or (file, snippet, f), where f
# turns the constant into the number written there
SITES = {
    "HIGHLIGHT": [
        (SHADE, "pub const HIGHLIGHT: f64 = -#;", _neg),
        # The viewer's y runs up
        (VIEWER, "length(v_uv - vec2(-#, 0.35) * v_r)", _neg),
        (VIEWER, "length(v_uv - vec2(-0.35, #) * v_r)", _neg),
    ],
    "REACH": [(SHADE, "pub const REACH: f64 = #;"), (VIEWER, "/ (# * v_r), 1.0)")],
    "MID": [
        (SHADE, "pub const MID: f64 = #;"),
        (VIEWER, "fill = t < #"),
        (VIEWER, "v_fill, t / #)"),
        (VIEWER, "(t - #) / 0.55"),
        (VIEWER, "(t - 0.45) / #)", _rest),
    ],
    "LIGHTEN": [(SHADE, "const LIGHTEN: f32 = #;"), (VIEWER, "mix(v_fill, vec3(1.0), #)")],
    "DARKEN": [(SHADE, "const DARKEN: f32 = #;"), (VIEWER, "mix(v_fill, # * v_fill,", _rest)],
    "CURVE": [
        (SVG, "if scene.curved { # + bow } else { bow }"),
        (SVG, "= # + edges.bow[i];"),
        (RASTER, "if scene.curved { # + bow } else { bow }"),
        (PAYLOAD, "if scene.curved { # } else { 0.0 }"),
        (BUILDER, "if curved { # + bows[i] } else { bows[i] }"),
        (VIEWER, "orig.bow + (curve ? # : -#)"),
    ],
    "ARROW_WIDTH": [
        (GEOMETRY, "let arrow_scale = py_max(width / #, 0.6);"),
        (BUILDER, "let base = py_max(w / #, 0.6);"),
        (BUILDER, "# * base * zoomed"),
        (VIEWER, "float base = max(a_width / #, 0.6)"),
    ],
    "ARROW_MIN": [
        (GEOMETRY, "let arrow_scale = py_max(width / 1.2, #);"),
        (BUILDER, "let base = py_max(w / 1.2, #);"),
        (VIEWER, "float base = max(a_width / 1.2, #)"),
    ],
    "HEAD_LENGTH": [
        (GEOMETRY, "let head_len = # * arrow_scale;"),
        (VIEWER, "float head_len = # * grow"),
    ],
    "HEAD_WIDTH": [
        (GEOMETRY, "let head_wid = # * arrow_scale;"),
        (VIEWER, "head_wid = # * grow;"),
    ],
    "HEAD_GROW": [(BUILDER, ".clamp(1.0, #);"), (VIEWER, "const float HEAD_GROW = #,")],
    "HEAD_NODE": [
        (BUILDER, ".min(radii[self.tgt[i]] / (# * base))"),
        (VIEWER, "rb / (# * base)"),
    ],
    "LOOP_SIN": [(GEOMETRY, "const LOOP_SIN: f64 = #;"), (VIEWER, "LOOP_SIN = #,")],
    "LOOP_COS": [(GEOMETRY, "const LOOP_COS: f64 = #;"), (VIEWER, "LOOP_COS = #;")],
    "LOOP_RADIUS": [
        (GEOMETRY, "let lr = r * # * loop_size;"),
        (VIEWER, "float lr = # * ra * a_loop_size"),
        (VIEWER, "const lr = # * r * size"),
    ],
    "LOOP_CONTROL": [(GEOMETRY, "let cp_h = lr * #;"), (VIEWER, "cp = # * lr")],
    "LOOP_LEAN": [(GEOMETRY, "cp_h * #;"), (VIEWER, "# * cp,")],
    "LOOP_TOP": [
        (GEOMETRY, "cy_arc - lr * #)"),
        (VIEWER, "vec2(0.0, # * lr)"),
        (VIEWER, "top - # * lr"),
    ],
    "THIN": [
        (RASTER, "if disc.r < # {"),
        (RASTER, "(disc.r / #).powi(2)"),
        (RASTER, "} else if disc.bw < # {"),
        (RASTER, "disc.bw / # * disc.border_opacity"),
        (VIEWER, "(r < # ? pow(r / #, 2.0) : 1.0)"),
        (VIEWER, "v_r = max(r, #);"),
        (VIEWER, "(bw < # ? bw / # : 1.0)"),
        (VIEWER, "v_bw = max(bw, #);"),
    ],
    "BORDER_MAX": [
        (BUILDER, "py_min(bw, # * enc.radii[k])"),
        (VIEWER, "min(st.y, # * nd.z)"),
    ],
    "DENSITY_MAX": [
        (RASTER, "-(-gain.min(#)).ln_1p()"),
        (RASTER, "-(-a.min(#)).ln_1p()"),
        (VIEWER, "-log(1.0 - min(a, #))"),
        (VIEWER, "-Math.log(1 - Math.min(gain, #))"),
    ],
    "SOLID": [(RASTER, "const SOLID: f32 = #;"), (VIEWER, "texelFetch(u_peak, ivec2(0), 0).r, #)")],
    "LONE_STEP": [(RASTER, ".max(lone * #);"), (VIEWER, "u_lone * #);")],
    "MUTED_OPACITY": [(BUILDER, "opacity = enc.opacity * #;"), (VIEWER, "l === 0 ? # * o")],
    "SOFT_WIDTH": [(BUILDER, "w = py_max(w * #, 1.0);"), (VIEWER, "Math.max(# * w, 1)")],
    "SOFT_WIDTH_MIN": [(BUILDER, "w = py_max(w * 1.3, #);"), (VIEWER, "Math.max(1.3 * w, #)")],
    "SOFT_OPACITY": [
        (BUILDER, "w = py_max(w * 1.3, 1.0); opacity = py_max(enc.opacity, #);"),
        (VIEWER, "Math.max(o, l === 1 ? # : 0.9)"),
    ],
    "ACCENT_WIDTH": [(BUILDER, "w = py_max(w * #, 1.5);"), (VIEWER, "Math.max(# * w, 1.5)")],
    "ACCENT_WIDTH_MIN": [(BUILDER, "w = py_max(w * 2.0, #);"), (VIEWER, "Math.max(2 * w, #)")],
    "ACCENT_OPACITY": [
        (BUILDER, "w = py_max(w * 2.0, 1.5); opacity = py_max(enc.opacity, #);"),
        (VIEWER, "Math.max(o, l === 1 ? 0.5 : #)"),
    ],
    "LABEL_FONT_MIN": [
        (BUILDER, "py_max(#, py_min(13.0, enc.node_size * 0.7))"),
        (VIEWER, 'Math.max(#, Math.min(13, 0.7 * v("node_size")))'),
    ],
    "LABEL_FONT_MAX": [
        (BUILDER, "py_max(8.0, py_min(#, enc.node_size * 0.7))"),
        (VIEWER, 'Math.max(8, Math.min(#, 0.7 * v("node_size")))'),
    ],
    "LABEL_FONT_SCALE": [
        (BUILDER, "py_max(8.0, py_min(13.0, enc.node_size * #))"),
        (VIEWER, 'Math.max(8, Math.min(13, # * v("node_size")))'),
    ],
    "LABEL_BELOW": [
        (LABELS, "let top = y + r + #;"),
        (SVG, "let ly = nodes.y[k] + nodes.r[k] + size + #;"),
        (VIEWER, "const top = Y(k) + at(data.r, k) * v.grow + #;"),
    ],
    "LABEL_LINE": [(LABELS, "top + font * #]"), (VIEWER, "top + # * size")],
    "LABEL_GAP": [(LABELS, "const GAP: f64 = #;"), (VIEWER, "const LABEL_GAP = #;")],
    "EDGE_FONT": [
        (SVG, 'font-size=\\"#\\" fill=\\"{label_color}\\"'),
        (VIEWER, "const EDGE_FONT = #;"),
    ],
    "EDGE_LABEL_OPACITY": [
        (SVG, 'fill-opacity=\\"#\\"'),
        (VIEWER, "ctx.globalAlpha = #; ctx.fillText(text, 0, 0);"),
    ],
    "HALO": [
        (SVG, 'stroke-width=\\"#\\" stroke-linejoin=\\"round\\" paint-order=\\"stroke\\"'),
        (VIEWER, 'lineWidth: #, strokeStyle: v.bg, textAlign: "center"'),
    ],
}


def _table():
    """The constants that VISUAL_MODEL.md lists, by name, as numbers."""
    rows = re.findall(r"^\| `(\w+)` \| (\S+) \|", MODEL.read_text(encoding="utf-8"), re.M)
    assert rows, "VISUAL_MODEL.md has no table of shared constants"
    table = {}
    for name, value in rows:
        assert name not in table, f"{name} is listed twice"
        table[name] = float(value)
    return table


def _pattern(snippet):
    """A regular expression for `snippet`, with a group for each `#` and any run of white space
    where it has a space."""
    parts = [re.escape(part).replace(r"\ ", r"\s+") for part in snippet.split("#")]
    return re.compile(r"(\d+(?:\.\d*)?)".join(parts))


@pytest.fixture(scope="module")
def table():
    return _table()


@pytest.fixture(scope="module")
def sources():
    return {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in {site[0] for sites in SITES.values() for site in sites}
    }


def test_the_table_lists_exactly_the_constants_checked(table):
    assert sorted(table) == sorted(SITES)


@pytest.mark.parametrize("name", sorted(SITES))
def test_a_constant_has_its_value_at_each_site(name, table, sources):
    value = table[name]
    for path, snippet, *f in SITES[name]:
        want = f[0](value) if f else value
        found = _pattern(snippet).findall(sources[path])
        assert found, f"{name}: {snippet!r} isn't in {path}"
        for match in found:
            for number in match if isinstance(match, tuple) else (match,):
                # The viewer's GLSL writes the self-loop's sine and cosine to 8 places
                assert math.isclose(float(number), want, rel_tol=1e-7), (
                    f"{name} is {want} but {path} has {number} at {snippet!r}"
                )


def test_the_self_loop_meets_its_node_at_0_45_radians(table):
    assert table["LOOP_SIN"] == math.sin(0.45)
    assert table["LOOP_COS"] == math.cos(0.45)


def test_the_table_names_sites_that_exist(table):
    # Files of `src/scene/`, with their functions and structs, and the viewer's shaders and
    # functions
    text = MODEL.read_text(encoding="utf-8")
    rows = re.findall(r"^\| `\w+` \| \S+ \| (.+) \| (.+) \|$", text, re.M)
    assert len(rows) == len(table)
    viewer = (ROOT / VIEWER).read_text(encoding="utf-8")
    for rust, js in rows:
        for site in re.findall(r"`([^`]+)`", rust):
            file, _, fn = site.partition("::")
            file = file if file.endswith(".rs") else f"{file}.rs"
            path = ROOT / "src" / "scene" / file
            assert path.exists(), site
            name = fn.removesuffix("()")
            if name:
                assert re.search(rf"\b(fn|struct) {name}\b", path.read_text(encoding="utf-8")), site
        for site in re.findall(r"`([^`]+)`", js):
            name = site.removesuffix("()")
            assert re.search(rf"\b(const {name} =|function {name}\()", viewer), site
