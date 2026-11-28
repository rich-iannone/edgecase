# The visual model

A plot is drawn three ways from one scene (`builder.rs`): as SVG (`svg.rs`), as PNG (`raster.rs`,
with the labels and the legend drawn over it from the SVG by `overlay.rs`), and by the viewer
(`python/edgecase/static/viewer.js`, WebGL 2 for the marks and a 2D canvas for the labels). The
shapes come from `geometry.rs`, the shading from `shade.rs` and the label choice from `labels.rs`,
and the viewer has each rule again in JavaScript or GLSL. This file is those rules, written down
once, with where each is implemented.

`tests/test_visual_model.py` checks that the constants in the table at the end have the same
values in the Rust sources and in the viewer. `tests/test_parity.py` draws fixed scenes and views
with the three and compares the images; it needs `pip install edgecase[parity]` and
`playwright install chromium`, and runs in CI's parity job.

## Units and framing

- Scene units are the plot's pixels at zoom 1: `width` by `height` (700 by 500 by default).
- The viewBox (`geometry::fit_viewbox()`) holds the nodes with 30 of padding, 20 more below for
  labels when any are shown, and room for each label's half width (7 per character over 2). It is
  then widened to the aspect of the box left after the legend (a column at most 0.4 of the width,
  or a band at most 0.4 of the height).
- The viewBox is fitted into the box, centred, at `s = min(w / vw, h / vh)`, as SVG's
  `preserveAspectRatio` does (`raster::View::new()`, the viewer's `fit()`). The PNG is drawn at
  `RASTER_SCALE` (2) pixels per unit, the viewer at the device's pixel ratio.

## Camera and zoom

A view (`plot(view=...)`, the viewer's camera) is a centre `c` and a zoom.

- Positions: `home + (p − c)·zoom`, where `home` is the viewBox's centre (`builder.rs`, the camera
  in `scene()`; the viewer's `view()`).
- Node radii grow by `√zoom`; edge widths, fonts and label gaps stay in pixels.
- Node borders are at most half the radius, `min(bw, 0.5·r)`, then grow by `√zoom`.
- Arrowheads grow with the node they point at, by
  `zoomed = clamp(min(√zoom, r_tgt / (4·base)), 1, 3)` with `base = max(w / 1.2, 0.6)`: the head
  is drawn as for an edge of width `1.2·base·zoomed` (`builder.rs` head widths; `LINK_VS`).
- Items wholly out of sight are left out, and the labels are chosen again among the nodes in
  sight, at their size in pixels.
- With a view the static backends take the viewer's framing: its node-size minimum, the legend in
  a row and no edge routing.

## Painter's order

The background, then the edges, the edge labels, the nodes, the node labels and the legend
(`svg::draw()`). Edges and nodes are each drawn in order of highlight level, muted first. The
raster backend draws the edges and nodes as one image and the SVG the rest over it. The viewer
draws the edges into a density buffer that `TONE_FS` turns into color, then the nodes, then the
labels on a canvas over the marks.

## Nodes

- A disc of radius `r` with a border centred on its edge, `bw` wide, of opacity `bo`. By default
  the border is a darker version of the fill, 1.5 wide at opacity 0.3.
- Coverage is `clamp(r + 0.5 − d, 0, 1)` at distance `d` from the centre, and the border's
  `clamp(bw/2 + 0.5 − |d − r|, 0, 1)` (`raster::draw_discs()`, `POINT_FS`).
- A node smaller than a pixel (`r < 0.5`) is drawn at `r = 0.5` with its fill times `(r / 0.5)²`;
  a border thinner than a pixel (`bw/2 < 0.5`) is drawn at 0.5 with its opacity times
  `(bw/2) / 0.5` (`raster::draw_discs()`, `POINT_VS`).
- Shading (`shade.rs`; the SVG's `radialGradient`; `POINT_FS`): a highlight at
  `(HIGHLIGHT·r, HIGHLIGHT·r)` from the centre (up and to the left), and `t = min(|p − h| /
  (REACH·r), 1)`. Up to `t = MID` the color runs from the fill mixed `LIGHTEN` toward white to
  the fill, and from there to the fill mixed `DARKEN` toward black. The SVG gradient puts its
  centre at `(1 + HIGHLIGHT) / 2` of the box with radius `REACH / 2`, and its stops at 0, `MID`
  and 1.

## Edges

- **Straight** (`geometry::straight()`; `LINK_VS`): from border to border of the two nodes.
  Directed, the line stops at the arrowhead's base.
- **Curved** (`geometry::curved_points()`; `LINK_VS`): a quadratic Bézier with its control point
  at `mid + perp·dist·bow`, where `perp` is the unit normal and `dist` the length between the
  centres. The start is `r_src` from the source toward the control point, the end `r_dst` from
  the target back along the tangent there; the arrowhead points along that tangent.
- **Bow:** each edge's own bow (parallel edges fan out by steps of 0.2, `builder::fans()`), plus
  `CURVE` (0.2) when the plot's edges are curved (`svg.rs`, `raster.rs`, `payload.rs`; the
  viewer's Curved control adds or takes away 0.2 in `restyle()`).
- **Self-loops** (`geometry::self_loop_points()`; `LINK_VS`, `loopPoints()`): two cubics from
  the node's border at `±LOOP_SIN·r` across and `LOOP_COS·r` up, with the loop's radius
  `lr = 1.2·r·loop_size`, control height `cp = 1.6·lr`, the first and last controls leaning out by
  `0.3·cp` and the top `0.2·lr` above the arcs' meeting.
- **Arrowheads** (`geometry::arrow()`; `LINK_VS`): `scale = max(width / 1.2, 0.6)`, a triangle
  `8·scale` long and `5·scale` wide with its tip on the target's border.
- **Stroke:** round caps and joins, `width` wide. A stroke thinner than a pixel is drawn a pixel
  wide with its opacity times its width in pixels (`raster::shapes()`, `LINK_VS`, `loneAt()`).
  Coverage is `clamp(0.5 + distance inside, 0, 1)`.
- **Routed** edges (Catmull-Rom cubics through the router's points, `geometry::catmull_rom()`)
  are static only.

## Highlight levels

With `highlight=`, each node and edge has a level: 0 (muted), 1 (soft) or 2 (accent).
(`builder.rs`, the levels in `scene()`; the viewer's `restyle()`.)

- Edges at level 0 take the default edge color at half their opacity. At level 1 and 2 they take
  the accent (unless `edge_color_by=` colors them): width `max(1.3·w, 1)` and opacity
  `max(o, 0.5)` at level 1, `max(2·w, 1.5)` and `max(o, 0.9)` at level 2.
- Nodes at level 0 take the muted color, and their labels the muted label color. At level 1 the
  fill is the color mixed 0.7 toward the surface, ringed by the color at opacity 1, 2 wide.

## Blending and tone

- Each edge's coverage `a` (times its opacity) adds `−ln(1 − min(a, 0.999))` to its pixels'
  density, and its color weighted by that (`raster::add_ink()`, `LINK_FS`).
- **Over** (the default): opacity `1 − e^(−D)`, the same as SVG strokes of one color stacked
  (`raster::ToneMap`, `TONE_FS`). The color is the density-weighted mean of the edges' colors.
- **Log** (`edge_tone="log"`): `lone` is the median edge's density (at least 1e-3), `top` is
  `max(densest, SOLID, 1.01·lone)` with `SOLID = 3.912` (98% opaque). Above `lone`, the opacity is
  `lone_a + (1 − lone_a)·min(ln(D / lone) / ln(top / lone), 1)` with `lone_a = 1 − e^(−lone)`.
- Nodes are drawn over the edges with ordinary over compositing, in sRGB.

## Node labels

- Font size `max(8, min(13, 0.7·node_size))` (`builder.rs`; `restyle()`).
- Each label is centred under its node; its box runs from `y + r + 2` down `1.25·font`, and its
  baseline is `font` below the box's top (`labels::label_box()`, `svg::draw_labels()`,
  `labelLayer()`).
- Choice (`labels.rs`; `labelLayer()`): every label of a plot of up to `ALL_MAX` (50) nodes, and
  otherwise up to `AUTO_MAX` (30), or `max_labels=`. Those are taken in order of level (accent
  first), radius, degree and index, each unless its box comes within `GAP` (2) of one taken
  before.
- The viewer draws the first taken over the rest, with a halo of the background 3 wide.

## Edge labels

- 10 pixels, the label color at opacity 0.7.
- The SVG places them for print: along a curved edge's path, on a fan of 3 or more edges at its
  middle with a halo of the background 3 wide, and otherwise beside the edge's middle. The
  viewer draws them along the edge, turned with it, and leaves out those that would overlap.

## Differences by design

- The publisher routes edges around nodes (`edge_routing=`); the viewer doesn't, nor does a
  static plot with a view.
- The publisher has a fixed legend (a column or a band) and places edge labels for print; the
  viewer has its legend in a row at the bottom, which folds.
- The viewer has hover, the controls and the simulation, and its arrowheads grow toward
  `HEAD_GAIN` (0.85) opacity as they grow with the zoom.
- The raster's `Tone::EqHist` has no counterpart in the viewer (nor in `plot()`'s `edge_tone=`).

## Known divergences

The parity tests measure these.

- The SVG has no halo under node labels; the viewer does.
- Label widths: the Rust choice estimates `CHAR_W` (0.58) of the font per character, while the
  viewer measures the text, so near a collision the two can take different labels. The glyphs
  come from different font engines (Chrome; fontdb and resvg in `overlay.rs`).
- Without a view, the static backends don't clamp a node's border to half its radius.
- The raster's curved edges and self-loops run on under the arrowhead, which adds ink twice
  there; the SVG draws line and head as one group, and the viewer's stroke gives way under the
  head.
- Where edges of different colors cross, the SVG shows the last drawn on top, and the raster and
  the viewer the density-weighted mean.
- The viewer's `lone` is the median of up to 4096 edges, evenly spread.
- The viewer draws edge labels over the nodes; the SVG under them.

## Shared constants

| Name | Value | Rust | Viewer |
|---|---|---|---|
| `HIGHLIGHT` | -0.35 | `shade.rs` | `POINT_FS` |
| `REACH` | 1.5 | `shade.rs` | `POINT_FS` |
| `MID` | 0.45 | `shade.rs` | `POINT_FS` |
| `LIGHTEN` | 0.5 | `shade.rs` | `POINT_FS` |
| `DARKEN` | 0.35 | `shade.rs` | `POINT_FS` |
| `CURVE` | 0.2 | `svg.rs`, `raster.rs`, `payload.rs`, `builder.rs` | `restyle()` |
| `ARROW_WIDTH` | 1.2 | `geometry::arrow()`, `builder.rs` | `LINK_VS` |
| `ARROW_MIN` | 0.6 | `geometry::arrow()`, `builder.rs` | `LINK_VS` |
| `HEAD_LENGTH` | 8 | `geometry::arrow()` | `LINK_VS` |
| `HEAD_WIDTH` | 5 | `geometry::arrow()` | `LINK_VS` |
| `HEAD_GROW` | 3 | `builder.rs` | `LINK_VS` |
| `HEAD_NODE` | 4 | `builder.rs` | `LINK_VS` |
| `LOOP_SIN` | 0.43496553411123023 | `geometry.rs` | `LINK_VS`, `loopPoints()` |
| `LOOP_COS` | 0.9004471023526769 | `geometry.rs` | `LINK_VS`, `loopPoints()` |
| `LOOP_RADIUS` | 1.2 | `geometry::self_loop_points()` | `LINK_VS`, `loopPoints()` |
| `LOOP_CONTROL` | 1.6 | `geometry::self_loop_points()` | `LINK_VS`, `loopPoints()` |
| `LOOP_LEAN` | 0.3 | `geometry::self_loop_points()` | `LINK_VS`, `loopPoints()` |
| `LOOP_TOP` | 0.2 | `geometry::self_loop_points()` | `LINK_VS`, `loopPoints()` |
| `THIN` | 0.5 | `raster::draw_discs()` | `POINT_VS` |
| `BORDER_MAX` | 0.5 | `builder.rs` | `POINT_VS` |
| `DENSITY_MAX` | 0.999 | `raster.rs` | `LINK_FS`, `loneAt()` |
| `SOLID` | 3.912 | `raster::ToneMap` | `TONE_FS` |
| `LONE_STEP` | 1.01 | `raster::ToneMap` | `TONE_FS` |
| `MUTED_OPACITY` | 0.5 | `builder.rs` | `restyle()` |
| `SOFT_WIDTH` | 1.3 | `builder.rs` | `restyle()` |
| `SOFT_WIDTH_MIN` | 1 | `builder.rs` | `restyle()` |
| `SOFT_OPACITY` | 0.5 | `builder.rs` | `restyle()` |
| `ACCENT_WIDTH` | 2 | `builder.rs` | `restyle()` |
| `ACCENT_WIDTH_MIN` | 1.5 | `builder.rs` | `restyle()` |
| `ACCENT_OPACITY` | 0.9 | `builder.rs` | `restyle()` |
| `LABEL_FONT_MIN` | 8 | `builder.rs` | `restyle()` |
| `LABEL_FONT_MAX` | 13 | `builder.rs` | `restyle()` |
| `LABEL_FONT_SCALE` | 0.7 | `builder.rs` | `restyle()` |
| `LABEL_BELOW` | 2 | `labels::label_box()`, `svg::draw_labels()` | `labelLayer()` |
| `LABEL_LINE` | 1.25 | `labels::label_box()` | `labelLayer()` |
| `LABEL_GAP` | 2 | `labels.rs` | `labelLayer()` |
| `EDGE_FONT` | 10 | `svg::draw_edge_labels()` | `labelLayer()` |
| `EDGE_LABEL_OPACITY` | 0.7 | `svg::draw_edge_labels()` | `labelLayer()` |
| `HALO` | 3 | `svg::draw_edge_labels()` | `labelLayer()` |
