// The scene that `plot()` draws: its nodes, edges and labels as typed columns, filled from
// Python once it has read the attributes and worked out the encodings, and the SVG they make
mod geometry;
mod pyfmt;
mod svg;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

struct Labels {
    texts: Vec<String>,
    colors: Vec<usize>,
    font_size: f64,
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

    /// The text under each node, in `font_size`.
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
            texts,
            colors,
            font_size,
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
    /// `label_color` its background and text colors, and `legend` the legend's markup.
    #[pyo3(signature = (colors, bg, label_color, font_family, legend=None))]
    fn svg(
        &self,
        py: Python<'_>,
        colors: Vec<String>,
        bg: String,
        label_color: String,
        font_family: String,
        legend: Option<String>,
    ) -> PyResult<String> {
        check_len(
            "colors",
            colors.len(),
            self.colors.len(),
            "colors in the table",
        )?;
        let style = svg::Style {
            colors: &colors,
            bg: &bg,
            label_color: &label_color,
            font_family: &font_family,
            legend: legend.as_deref(),
        };
        Ok(py.allow_threads(|| svg::draw(self, &style)))
    }
}

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
