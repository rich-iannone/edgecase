// The scene that `plot()` draws: its nodes, edges and labels as typed columns, filled from
// Python once it has read the attributes and worked out the encodings, and the SVG they make
mod builder;
mod color;
mod css;
mod geometry;
mod highlight;
mod labels;
mod overlay;
mod palette;
mod payload;
mod pyfmt;
mod raster;
mod shade;
mod svg;
mod tips;

pub use builder::SceneBuilder;
pub use highlight::{edge_ends, marked_edges, marked_nodes};
pub use shade::shade_stops;
pub use tips::{tooltip_attrs, tooltip_table};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[derive(Default)]
struct Nodes {
    ids: Vec<String>,
    x: Vec<f64>,
    y: Vec<f64>,
    r: Vec<f64>,
    // Indexes into the scene's colors
    fill: Vec<usize>,
    border: Vec<usize>,
    border_width: Vec<f64>,
    border_opacity: Vec<f64>,
    // The nodes as drawn, the highlighted ones last
    order: Vec<usize>,
    // The tooltips, as JSON for a `data-attrs='...'` attribute
    tips: Vec<String>,
}

/// The nodes' labels: each one's text and color, and which of them are drawn (see `labels.rs`).
/// `show` is `show_labels=`, without which the viewer has them for its controls but draws none;
/// `order` is the nodes in order of priority, and `limit` how many labels are drawn at most,
/// None for all of them, which is `auto` when `max_labels=` doesn't say
struct Labels {
    texts: Vec<String>,
    colors: Vec<usize>,
    font_size: f64,
    show: bool,
    order: Vec<usize>,
    limit: Option<usize>,
    auto: Option<usize>,
    shown: Vec<bool>,
}

#[derive(Default)]
struct Edges {
    // The ends, as node indexes
    src: Vec<usize>,
    tgt: Vec<usize>,
    color: Vec<usize>,
    width: Vec<f64>,
    opacity: Vec<f64>,
    // How far parallel and reciprocal edges bow out, and the size of self-loops
    bow: Vec<f64>,
    loop_size: Vec<f64>,
    // The waypoints of an edge routed around nodes, or None for a direct one
    route: Vec<Option<Vec<(f64, f64)>>>,
    order: Vec<usize>,
    // The tooltips as for the nodes, or none in lean SVG, which holds them in one table
    tips: Vec<String>,
}

/// For the viewer's controls, which redraw the scene as `plot()` would with other values: the
/// node size, edge width and edge opacity that the encodings started from (after their automatic
/// values), and with highlighting, each edge's level (0 muted, 1 soft, 2 accent) and its width
/// before highlighting widened it
struct Controls {
    node_size: f64,
    edge_width: f64,
    edge_opacity: f64,
    highlight: Option<(Vec<u8>, Vec<f64>)>,
}

fn check_len(column: &str, len: usize, n: usize, of: &str) -> PyResult<()> {
    if len == n {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{column} has {len} entries for {n} {of}"
        )))
    }
}

fn check_index(column: &str, values: &[usize], bound: usize, of: &str) -> PyResult<()> {
    match values.iter().find(|&&v| v >= bound) {
        None => Ok(()),
        Some(v) => Err(PyValueError::new_err(format!(
            "{column} has index {v}, past the {bound} {of}"
        ))),
    }
}

/// What `plot()` draws, apart from the theme: the frame, the color table, and the nodes, edges
/// and labels as columns, an entry for each node or edge. Python fills it with `set_nodes()`,
/// `set_labels()`, `set_edges()` and `set_edge_labels()`, and `svg()` draws it.
///
/// The color columns index `colors`, a table of (light, dark) pairs, which `svg()` takes in the
/// theme's colors. A scene without nodes is an empty graph.
#[pyclass(module = "edgecase._core")]
pub struct Scene {
    /// The size in pixels
    #[pyo3(get)]
    width: f64,
    #[pyo3(get)]
    height: f64,
    #[pyo3(get)]
    directed: bool,
    /// Every edge drawn as a curve
    #[pyo3(get)]
    curved: bool,
    /// Drawn as lean SVG, as large graphs are
    #[pyo3(get)]
    lean: bool,
    /// (min x, min y, width, height), legend included
    #[pyo3(get)]
    viewbox: (f64, f64, f64, f64),
    #[pyo3(get)]
    colors: Vec<(String, String)>,
    /// The legend's marks, which Python draws, and its width in pixels, at the right
    #[pyo3(get)]
    legend: PyObject,
    #[pyo3(get)]
    legend_w: f64,
    /// In lean SVG, the edges' tooltips as one JSON table
    #[pyo3(get)]
    edge_tip_table: Option<String>,
    /// For the viewer, the nodes' tooltips and the edges', in the order they are drawn, as
    /// JSON tables
    #[pyo3(get)]
    tip_tables: Option<(String, String)>,
    controls: Option<Controls>,
    nodes: Nodes,
    labels: Option<Labels>,
    edges: Edges,
    edge_labels: Option<Vec<Option<String>>>,
}

#[pymethods]
impl Scene {
    #[new]
    #[pyo3(signature = (
        width,
        height,
        directed=false,
        curved=false,
        lean=false,
        viewbox=(0.0, 0.0, 100.0, 100.0),
        colors=Vec::new(),
        legend=None,
        legend_w=0.0,
        edge_tip_table=None,
        tip_tables=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        width: f64,
        height: f64,
        directed: bool,
        curved: bool,
        lean: bool,
        viewbox: (f64, f64, f64, f64),
        colors: Vec<(String, String)>,
        legend: Option<PyObject>,
        legend_w: f64,
        edge_tip_table: Option<String>,
        tip_tables: Option<(String, String)>,
    ) -> Self {
        Scene {
            width,
            height,
            directed,
            curved,
            lean,
            viewbox,
            colors,
            legend: legend.unwrap_or_else(|| py.None()),
            legend_w,
            edge_tip_table,
            tip_tables,
            controls: None,
            nodes: Nodes::default(),
            labels: None,
            edges: Edges::default(),
            edge_labels: None,
        }
    }

    /// The nodes, in the order of their IDs.
    #[allow(clippy::too_many_arguments)]
    fn set_nodes(
        &mut self,
        ids: Vec<String>,
        x: Vec<f64>,
        y: Vec<f64>,
        r: Vec<f64>,
        fill: Vec<usize>,
        border: Vec<usize>,
        border_width: Vec<f64>,
        border_opacity: Vec<f64>,
        order: Vec<usize>,
        tips: Vec<String>,
    ) -> PyResult<()> {
        let n = ids.len();
        for (column, len) in [
            ("x", x.len()),
            ("y", y.len()),
            ("r", r.len()),
            ("fill", fill.len()),
            ("border", border.len()),
            ("border_width", border_width.len()),
            ("border_opacity", border_opacity.len()),
            ("order", order.len()),
            ("tips", tips.len()),
        ] {
            check_len(column, len, n, "nodes")?;
        }
        check_index("fill", &fill, self.colors.len(), "colors")?;
        check_index("border", &border, self.colors.len(), "colors")?;
        check_index("order", &order, n, "nodes")?;
        self.nodes = Nodes {
            ids,
            x,
            y,
            r,
            fill,
            border,
            border_width,
            border_opacity,
            order,
            tips,
        };
        Ok(())
    }

    /// The text under each node, in `font_size`, all of them drawn.
    fn set_labels(
        &mut self,
        texts: Vec<String>,
        colors: Vec<usize>,
        font_size: f64,
    ) -> PyResult<()> {
        let n = self.nodes.ids.len();
        check_len("texts", texts.len(), n, "nodes")?;
        check_len("colors", colors.len(), n, "nodes")?;
        check_index("colors", &colors, self.colors.len(), "colors")?;
        self.labels = Some(Labels {
            shown: vec![true; n],
            texts,
            colors,
            font_size,
            show: true,
            order: (0..n).collect(),
            limit: None,
            auto: None,
        });
        Ok(())
    }

    /// The edges, with their ends as node indexes. `tips` is None in lean SVG.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (src, tgt, color, width, opacity, bow, loop_size, route, order, tips=None))]
    fn set_edges(
        &mut self,
        src: Vec<usize>,
        tgt: Vec<usize>,
        color: Vec<usize>,
        width: Vec<f64>,
        opacity: Vec<f64>,
        bow: Vec<f64>,
        loop_size: Vec<f64>,
        route: Vec<Option<Vec<(f64, f64)>>>,
        order: Vec<usize>,
        tips: Option<Vec<String>>,
    ) -> PyResult<()> {
        let m = src.len();
        if tips.is_some() == self.lean {
            return Err(PyValueError::new_err(
                "edges have tips of their own unless the scene is lean",
            ));
        }
        let tips = tips.unwrap_or_default();
        for (column, len) in [
            ("tgt", tgt.len()),
            ("color", color.len()),
            ("width", width.len()),
            ("opacity", opacity.len()),
            ("bow", bow.len()),
            ("loop_size", loop_size.len()),
            ("route", route.len()),
            ("order", order.len()),
        ] {
            check_len(column, len, m, "edges")?;
        }
        if !self.lean {
            check_len("tips", tips.len(), m, "edges")?;
        }
        let n = self.nodes.ids.len();
        check_index("src", &src, n, "nodes")?;
        check_index("tgt", &tgt, n, "nodes")?;
        check_index("color", &color, self.colors.len(), "colors")?;
        check_index("order", &order, m, "edges")?;
        self.edges = Edges {
            src,
            tgt,
            color,
            width,
            opacity,
            bow,
            loop_size,
            route,
            order,
            tips,
        };
        Ok(())
    }

    /// The text along each edge, None for none.
    fn set_edge_labels(&mut self, labels: Vec<Option<String>>) -> PyResult<()> {
        check_len("labels", labels.len(), self.edges.src.len(), "edges")?;
        self.edge_labels = Some(labels);
        Ok(())
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.nodes.ids.len()
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.edges.src.len()
    }

    /// The scene as SVG, in a theme: `colors` are the color table's colors in it, `bg`,
    /// `label_color` its background and text colors, and `legend` the legend's markup. `raster`
    /// is the markup of the edges' layer and the nodes' as images (see `png()`), to draw in
    /// place of the edges and the nodes, between which the edge labels go. `shade` shades the
    /// nodes as spheres (see `shade.rs`), unless they are images: for each color of the table,
    /// the id of its gradient and its highlight's and far side's colors (from `shade_stops()`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        colors, bg, label_color, font_family, legend=None, raster=None, shade=None
    ))]
    fn svg(
        &self,
        py: Python<'_>,
        colors: Vec<String>,
        bg: String,
        label_color: String,
        font_family: String,
        legend: Option<String>,
        raster: Option<(String, String)>,
        shade: Option<Vec<(String, String, String)>>,
    ) -> PyResult<String> {
        check_len(
            "colors",
            colors.len(),
            self.colors.len(),
            "colors in the table",
        )?;
        if let Some(shade) = &shade {
            check_len(
                "shade",
                shade.len(),
                self.colors.len(),
                "colors in the table",
            )?;
        }
        let style = svg::Style {
            colors: &colors,
            bg: &bg,
            label_color: &label_color,
            font_family: &font_family,
            legend: legend.as_deref(),
            raster: raster.as_ref().map(|(e, n)| (e.as_str(), n.as_str())),
            shade: shade.as_deref(),
        };
        Ok(py.allow_threads(|| svg::draw(self, &style)))
    }

    /// The scene as a PNG image, with the edges drawn as density, in a theme: `colors` are the
    /// color table's colors in it, as hex, and `bg` its background, or None for a transparent
    /// one. The image is `scale` pixels to a pixel of the frame. `tone` is how the edges'
    /// density shows: `"over"` as the SVG draws it, `"log"` by its logarithm and `"eq_hist"` by
    /// its rank. `edges` and `nodes` say which are drawn, so that each can be a layer of its
    /// own. With `palette`, the image is indexed, in at most 256 colors: its own if it has no
    /// more, otherwise the nearest that the quantizer finds, which makes it two to three times
    /// smaller. Labels and the legend aren't drawn, but `overlay` is an SVG of the frame's size
    /// drawn over the rest, such as `svg()` makes of them, its text in the system's fonts.
    /// `shade` shades the nodes as spheres, as `svg()` does.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        colors, bg=None, scale=2.0, tone="log", edges=true, nodes=true, palette=false,
        overlay=None, shade=false
    ))]
    fn png<'py>(
        &self,
        py: Python<'py>,
        colors: Vec<String>,
        bg: Option<String>,
        scale: f64,
        tone: &str,
        edges: bool,
        nodes: bool,
        palette: bool,
        overlay: Option<String>,
        shade: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        check_len(
            "colors",
            colors.len(),
            self.colors.len(),
            "colors in the table",
        )?;
        let colors = colors
            .iter()
            .map(|c| css::parse(c))
            .collect::<PyResult<Vec<_>>>()?;
        let output = match bg {
            Some(bg) => match css::parse(&bg)? {
                [r, g, b, a] if a >= 1.0 => raster::Output::Over([r, g, b]),
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "bg must be opaque, not '{bg}'"
                    )))
                }
            },
            None => raster::Output::Rgba,
        };
        let style = raster::Style {
            colors: colors.iter().map(|&[r, g, b, _]| [r, g, b]).collect(),
            alphas: colors.iter().map(|c| c[3]).collect(),
            output,
            scale,
            tone: raster::Tone::parse(tone)?,
            edges,
            nodes,
            shade,
            overlay: match overlay {
                Some(svg) => {
                    let (width, height) = self.raster_size(scale)?;
                    Some(py.allow_threads(|| overlay::draw(&svg, width, height))?)
                }
                None => None,
            },
        };
        self.raster(py, &style, palette)
    }

    /// The scene as the viewer's payload (see `payload.rs`): its frame, color table, nodes and
    /// edges, as columns of 32-bit numbers after a JSON header. The labels' texts and the
    /// tooltips aren't in it (see `label_texts`).
    fn payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.allow_threads(|| payload::payload(self))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// For the viewer, the labels' texts as JSON, `{"nodes": [...], "edges": [...]}`: the
    /// nodes' in their order and the edges' in the order drawn, None where an edge has none,
    /// and either list null when there are no such labels. None when there are neither
    #[getter]
    fn label_texts(&self) -> Option<String> {
        if self.labels.is_none() && self.edge_labels.is_none() {
            return None;
        }
        let nodes = self.labels.as_ref().map(|l| &l.texts);
        let edges = self
            .edge_labels
            .as_ref()
            .map(|l| self.edges.order.iter().map(|&i| &l[i]).collect::<Vec<_>>());
        Some(serde_json::json!({"nodes": nodes, "edges": edges}).to_string())
    }

    /// The edges as a PNG mask: white, with the opacity at which `png()` draws them, to be
    /// tinted with any color, the edges' own in either theme when `edge_tint` is one. It is
    /// indexed, a palette of white at each opacity, which loses nothing.
    #[pyo3(signature = (scale=2.0, tone="log"))]
    fn edge_mask<'py>(
        &self,
        py: Python<'py>,
        scale: f64,
        tone: &str,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let style = raster::Style {
            colors: vec![[1.0; 3]; self.colors.len()],
            alphas: vec![1.0; self.colors.len()],
            output: raster::Output::Mask,
            scale,
            tone: raster::Tone::parse(tone)?,
            edges: true,
            nodes: false,
            shade: false,
            overlay: None,
        };
        self.raster(py, &style, true)
    }

    /// The index in the color table of the color that every edge has, opaque in both themes,
    /// or None if their colors differ (or there are no edges). A color with an alpha of its own
    /// is left out, since its alpha would go on the edges' density rather than on each edge
    #[getter]
    fn edge_tint(&self) -> Option<usize> {
        let first = *self.edges.color.first()?;
        let tint = &self.colors[first];
        let opaque = |c: &str| css::parse(c).is_ok_and(|c| c[3] >= 1.0);
        (opaque(&tint.0)
            && opaque(&tint.1)
            && self.edges.color.iter().all(|&c| self.colors[c] == *tint))
        .then_some(first)
    }

    /// The box `(x, y, width, height)` that the images of `png()` and `edge_mask()` at `scale`
    /// cover, in the viewBox's units: the frame, give or take the rounding of the pixels.
    #[pyo3(signature = (scale=2.0))]
    fn raster_box(&self, scale: f64) -> PyResult<(f64, f64, f64, f64)> {
        let (width, height) = self.raster_size(scale)?;
        Ok(raster::View::new(self.viewbox, width, height).frame(width, height))
    }
}

impl Scene {
    /// The size in pixels of an image of the frame at `scale`
    fn raster_size(&self, scale: f64) -> PyResult<(u32, u32)> {
        if scale.is_nan() || scale <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "scale must be greater than 0, not {scale}"
            )));
        }
        let (width, height) = (
            (self.width * scale).round().max(1.0),
            (self.height * scale).round().max(1.0),
        );
        if width * height > PIXELS_MAX {
            return Err(PyValueError::new_err(format!(
                "scale {scale} makes an image of {width}×{height} pixels; the most is \
                 {PIXELS_MAX} pixels"
            )));
        }
        Ok((width as u32, height as u32))
    }

    /// The scene drawn in `style`, as PNG, indexed with `palette`
    fn raster<'py>(
        &self,
        py: Python<'py>,
        style: &raster::Style,
        palette: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.raster_size(style.scale)?;
        let png = py.allow_threads(|| {
            let image = raster::draw(self, style);
            if palette { image.indexed() } else { image }.png()
        });
        Ok(PyBytes::new(py, &png))
    }
}

/// The most pixels a raster image may have, 16k × 16k
const PIXELS_MAX: f64 = 268_435_456.0;

/// The box `(min x, min y, width, height)` around nodes at `xs`, `ys` of radii `rs`, with
/// `padding` around them and room under them for labels of `label_chars` characters. With
/// `frame=(width, height)`, it's widened on its shorter side, around the drawing's center, to
/// the frame's aspect ratio.
#[pyfunction]
#[pyo3(signature = (xs, ys, rs, label_chars=None, label_space=20.0, padding=30.0, frame=None))]
pub fn fit_viewbox(
    xs: Vec<f64>,
    ys: Vec<f64>,
    rs: Vec<f64>,
    label_chars: Option<Vec<usize>>,
    label_space: f64,
    padding: f64,
    frame: Option<(f64, f64)>,
) -> PyResult<(f64, f64, f64, f64)> {
    check_len("ys", ys.len(), xs.len(), "nodes")?;
    check_len("rs", rs.len(), xs.len(), "nodes")?;
    if let Some(chars) = &label_chars {
        check_len("label_chars", chars.len(), xs.len(), "nodes")?;
    }
    Ok(geometry::fit_viewbox(
        &xs,
        &ys,
        &rs,
        label_chars.as_deref(),
        label_space,
        padding,
        frame,
    ))
}
