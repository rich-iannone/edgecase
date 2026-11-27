// The scene builder: it reads the nodes, edges and attribute values from the graph, works out
// the encodings (the colors of the groups and numeric scales, the sizes, widths and highlight
// levels), the fans of parallel edges and the tooltips, and fills a `Scene`, so that drawing a
// graph runs no Python for each node or edge. Each step follows what the Python renderer did, to
// the order of the colors in the table
use super::color::{darken, mix, Pair, Scale, Stop};
use super::geometry::{fit_viewbox, py_max, py_min};
use super::labels;
use super::tips::{Cell, Tips};
use super::{Base, Controls, Edges, Filter, Labels, Nodes, Scene, Selection};
use crate::graph::Graph;
use crate::layout::routing::route_edges;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};

/// An attribute's value for each node or edge, None where it has none (or holds None)
type Values = Vec<Option<PyObject>>;

/// The colors the encodings use, as (light, dark) pairs, from the Python renderer
#[derive(FromPyObject)]
pub struct SceneColors {
    #[pyo3(item)]
    node: Pair,
    #[pyo3(item)]
    edge: Pair,
    #[pyo3(item)]
    label: Pair,
    /// The background, which a soft node's fill is mixed with
    #[pyo3(item)]
    surface: Pair,
    /// No value on a scale
    #[pyo3(item)]
    missing: Pair,
    #[pyo3(item)]
    accent: Pair,
    #[pyo3(item)]
    muted_node: Pair,
    #[pyo3(item)]
    muted_label: Pair,
    /// Edges between groups, without `cross_community_color=`
    #[pyo3(item)]
    cross: Pair,
    #[pyo3(item)]
    sequential: Vec<Stop>,
    #[pyo3(item)]
    diverging: Vec<Stop>,
}

/// Color pairs, each held once, by index
#[derive(Clone, Default)]
struct Pairs {
    list: Vec<Pair>,
    ids: HashMap<Pair, usize>,
}

impl Pairs {
    fn id(&mut self, pair: Pair) -> usize {
        if let Some(&id) = self.ids.get(&pair) {
            return id;
        }
        self.ids.insert(pair.clone(), self.list.len());
        self.list.push(pair);
        self.list.len() - 1
    }

    fn same(&mut self, color: &str) -> usize {
        self.id((color.to_string(), color.to_string()))
    }
}

/// The scene's table of colors: the pairs in the order the scene first uses them
struct Table {
    colors: Vec<Pair>,
    of: Vec<Option<usize>>,
}

impl Table {
    fn index(&mut self, pairs: &Pairs, id: usize) -> usize {
        if id >= self.of.len() {
            self.of.resize(id + 1, None);
        }
        *self.of[id].get_or_insert_with(|| {
            self.colors.push(pairs.list[id].clone());
            self.colors.len() - 1
        })
    }
}

/// The distinct values of an attribute (as a dict holds keys, so `1` and `1.0` are one), and
/// which of them each node or edge has
struct Groups {
    of: Vec<Option<usize>>,
    firsts: Vec<PyObject>,
    // Whether a group's value equals itself, which NaN doesn't
    self_equal: Vec<bool>,
}

impl Groups {
    fn new(py: Python<'_>, values: &Values) -> PyResult<Self> {
        let index = PyDict::new(py);
        let mut groups = Groups {
            of: Vec::with_capacity(values.len()),
            firsts: Vec::new(),
            self_equal: Vec::new(),
        };
        for v in values {
            let Some(v) = v else {
                groups.of.push(None);
                continue;
            };
            let v = v.bind(py);
            let g = match index.get_item(v)? {
                Some(g) => g.extract()?,
                None => {
                    let g = groups.firsts.len();
                    index.set_item(v, g)?;
                    groups.firsts.push(v.clone().unbind());
                    groups.self_equal.push(v.eq(v)?);
                    g
                }
            };
            groups.of.push(Some(g));
        }
        Ok(groups)
    }

    /// Each group's place among them once sorted. Should they not all compare, the places are
    /// those that `list.sort()` leaves them in when it gives up
    fn ranks(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        let list = PyList::new(py, &self.firsts)?;
        if let Err(e) = list.sort() {
            if !e.is_instance_of::<PyTypeError>(py) {
                return Err(e);
            }
        }
        let place: FxHashMap<usize, usize> = list
            .iter()
            .enumerate()
            .map(|(k, v)| (v.as_ptr() as usize, k))
            .collect();
        Ok(self
            .firsts
            .iter()
            .map(|v| place[&(v.as_ptr() as usize)])
            .collect())
    }

    /// The groups in sorted order
    fn sorted(&self, ranks: &[usize]) -> Vec<usize> {
        let mut order = vec![0; ranks.len()];
        for (g, &r) in ranks.iter().enumerate() {
            order[r] = g;
        }
        order
    }

    /// Whether elements `i` and `j` have a value, and the same one
    fn same(&self, i: usize, j: usize) -> bool {
        match (self.of[i], self.of[j]) {
            (Some(a), Some(b)) => a == b && self.self_equal[a],
            _ => false,
        }
    }
}

/// `v` as a number on a numeric scale: an int or a float, not a bool, and finite
fn number(v: &Bound<PyAny>) -> PyResult<Option<f64>> {
    if v.is_instance_of::<PyBool>()
        || !(v.is_instance_of::<PyInt>() || v.is_instance_of::<PyFloat>())
    {
        return Ok(None);
    }
    let x: f64 = v.extract()?;
    Ok(x.is_finite().then_some(x))
}

fn numbers(py: Python<'_>, values: &Values) -> PyResult<Vec<Option<f64>>> {
    values
        .iter()
        .map(|v| match v {
            Some(v) => number(v.bind(py)),
            None => Ok(None),
        })
        .collect()
}

/// The scale for mapping `values` to colors: `forced` if given, else sequential for numbers
/// among which there is a fraction (a float), and categorical for anything else
fn pick_scale(
    py: Python<'_>,
    values: &Values,
    nums: &[Option<f64>],
    forced: Option<&str>,
    param: &str,
) -> PyResult<String> {
    let mut present = values.iter().zip(nums).filter(|(v, _)| v.is_some());
    if let Some(scale) = forced {
        if scale != "categorical" && present.any(|(_, n)| n.is_none()) {
            return Err(PyValueError::new_err(format!(
                "color_scale='{scale}' needs numeric values of {param}"
            )));
        }
        return Ok(scale.to_string());
    }
    let mut any = false;
    let mut fraction = false;
    for (v, n) in present {
        if n.is_none() {
            return Ok("categorical".into());
        }
        any = true;
        fraction |= v
            .as_ref()
            .is_some_and(|v| v.bind(py).is_instance_of::<PyFloat>());
    }
    Ok(if any && fraction {
        "sequential"
    } else {
        "categorical"
    }
    .into())
}

/// Python's `float(v)`
fn py_float(v: &Bound<PyAny>) -> PyResult<f64> {
    v.py().get_type::<PyFloat>().call1((v,))?.extract()
}

/// What a swatch of the legend stands for, which the viewer's legend can hide: the nodes of a
/// group of `color_by=` (None for those without a value), the edges of a color that
/// `edge_color_by=` gives an edge attribute's group, the edges within a group of a node
/// attribute's, or those between groups
#[derive(Clone, Copy)]
enum Key {
    Nodes(Option<usize>),
    EdgeColor(usize),
    Within(usize),
    Cross,
}

/// A block of the legend
enum Block {
    /// `(shape, color, text, key)` items
    Swatches {
        title: PyObject,
        items: Vec<(&'static str, usize, PyObject, Key)>,
    },
    /// A numeric scale, labelled with the values its `ends` (low, high, midpoint) stand for
    Ramp {
        title: PyObject,
        scale: String,
        ends: (f64, f64, Option<f64>),
        missing: Option<bool>,
    },
}

/// How the edges are colored
enum EdgeColors {
    /// By color_by='s groups, an edge within one taking its color
    Plain,
    /// By an edge attribute: each edge's color, or None for none
    Each(Vec<Option<usize>>),
    /// By the groups of the ends' values of a node attribute, an edge between groups taking
    /// `cross`
    Groups {
        groups: Groups,
        colors: Vec<usize>,
        cross: usize,
    },
}

/// The encodings, from `encode()`
struct Encoded {
    pairs: Pairs,
    node_color: Vec<usize>,
    node_groups: Option<Groups>,
    edge_colors: EdgeColors,
    widths: Vec<f64>,
    opacity: f64,
    radii: Vec<f64>,
    node_size: f64,
    edge_width: f64,
    // What the legend's swatches stand for, in the order it draws them
    keys: Vec<Key>,
}

/// How the edges are colored by `edge_color_by=`
enum EdgeColorBy {
    Off,
    /// By an edge attribute (or the weights)
    Edges(Values),
    /// By a node attribute
    Nodes(Values),
}

/// What `plot()` draws of a graph, read from it once: the nodes and edges, and the values of
/// the attributes that the arguments name. `encode()` maps the values to colors and sizes, and
/// `highlight()` takes what `highlight=` picks out; `scene()` then makes the `Scene` at the
/// layout's positions.
#[pyclass(module = "edgecase._core")]
pub struct SceneBuilder {
    colors: SceneColors,
    directed: bool,
    ids: Vec<u64>,
    // Each node's degree, its in- and out-degree together in a directed graph
    degrees: Vec<usize>,
    names: Vec<String>,
    index: FxHashMap<u64, usize>,
    node_labels: Vec<Option<String>>,
    node_attrs: Vec<Option<Py<PyDict>>>,
    // The text drawn under each node
    labels: Vec<String>,
    // The edges' ends as node indexes, their IDs (None for an edge drawn without its object),
    // their ends as stored, weights and attributes
    src: Vec<usize>,
    tgt: Vec<usize>,
    edge_ids: Vec<Option<u64>>,
    tip_ends: Vec<(usize, usize)>,
    weights: Vec<Option<f64>>,
    edge_attrs: Vec<Option<Py<PyDict>>>,
    color_by: Option<PyObject>,
    color_values: Values,
    size_values: Option<Values>,
    edge_color_by: Option<PyObject>,
    edge_color: EdgeColorBy,
    edge_width_values: Option<Values>,
    edge_labels: Option<Vec<Option<String>>>,
    enc: Option<Encoded>,
    // The highlight level of each node and edge: 0 muted, 1 soft, 2 accent
    levels: Option<(Vec<u8>, Vec<u8>)>,
}

/// The border of a node of color `c`, the color darkened, made once for each color
fn darker(pairs: &mut Pairs, made: &mut FxHashMap<usize, usize>, c: usize) -> PyResult<usize> {
    if let Some(&dk) = made.get(&c) {
        return Ok(dk);
    }
    let (l, d) = pairs.list[c].clone();
    let dk = pairs.id((darken(&l, 0.3)?, darken(&d, 0.3)?));
    made.insert(c, dk);
    Ok(dk)
}

fn non_empty(py: Python<'_>, attrs: Option<&Py<PyDict>>) -> Option<Py<PyDict>> {
    attrs
        .filter(|a| !a.bind(py).is_empty())
        .map(|a| a.clone_ref(py))
}

/// The numeric attributes in `attrs` (each node's or edge's), in the order they first appear,
/// as filters for the viewer: those whose values are all ints or floats (not bools), and that
/// have two values or more. A value that isn't finite, or None, counts as none
fn numeric_attrs(
    py: Python<'_>,
    attrs: &[Option<Py<PyDict>>],
    edges: bool,
) -> PyResult<Vec<Filter>> {
    // Each name's place among them, and whether its values are numbers so far
    let mut place: FxHashMap<String, usize> = FxHashMap::default();
    let mut found: Vec<(Filter, bool)> = Vec::new();
    for (i, a) in attrs.iter().enumerate() {
        let Some(a) = a else { continue };
        for (key, v) in a.bind(py).iter() {
            let Ok(key) = key.downcast::<PyString>() else {
                continue;
            };
            let name = key.to_str()?;
            let p = match place.get(name) {
                Some(&p) => p,
                None => {
                    place.insert(name.to_string(), found.len());
                    let values = vec![f64::NAN; attrs.len()];
                    found.push((filter(name.to_string(), edges, values), true));
                    found.len() - 1
                }
            };
            let (f, numeric) = &mut found[p];
            if !*numeric || v.is_none() {
                continue;
            }
            if v.is_instance_of::<PyBool>() {
                *numeric = false;
            } else if v.is_instance_of::<PyInt>() {
                // An int too large for a float isn't one to filter by
                match v.extract::<f64>() {
                    Ok(x) => f.values[i] = x,
                    Err(_) => *numeric = false,
                }
            } else if v.is_instance_of::<PyFloat>() {
                f.integer = false;
                let x: f64 = v.extract()?;
                if x.is_finite() {
                    f.values[i] = x;
                }
            } else {
                *numeric = false;
            }
            if !*numeric {
                f.values = Vec::new();
            }
        }
    }
    Ok(found
        .into_iter()
        .filter(|(_, numeric)| *numeric)
        .filter_map(|(f, _)| ranged(f))
        .collect())
}

fn filter(name: String, edges: bool, values: Vec<f64>) -> Filter {
    Filter {
        name,
        edges,
        min: 0.0,
        max: 0.0,
        integer: true,
        values,
    }
}

/// `f` with the least and the most of its values, if they differ
fn ranged(mut f: Filter) -> Option<Filter> {
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in f.values.iter().filter(|v| !v.is_nan()) {
        min = min.min(v);
        max = max.max(v);
    }
    (min < max).then(|| {
        (f.min, f.max) = (min, max);
        f
    })
}

fn truthy(key: &Option<Bound<PyAny>>) -> PyResult<bool> {
    match key {
        Some(k) => k.is_truthy(),
        None => Ok(false),
    }
}

/// The bows of parallel and reciprocal edges, fanned out to either side, and the sizes of
/// self-loops, each one at a node larger than the last
fn fans(names: &[String], src: &[usize], tgt: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let m = src.len();
    let mut groups: FxHashMap<(usize, usize), Vec<usize>> = FxHashMap::default();
    let mut keys = Vec::new();
    for i in 0..m {
        let (a, b) = (src[i], tgt[i]);
        let key = if a == b || names[a] <= names[b] {
            (a, b)
        } else {
            (b, a)
        };
        groups
            .entry(key)
            .or_insert_with(|| {
                keys.push(key);
                Vec::new()
            })
            .push(i);
    }
    let mut bows = vec![0.0; m];
    let mut loop_sizes = vec![1.0; m];
    for key in keys {
        let group = &groups[&key];
        if key.0 == key.1 {
            for (k, &i) in group.iter().enumerate() {
                loop_sizes[i] = 1.0 + k as f64 * 0.6;
            }
            continue;
        }
        let n = group.len();
        if n == 1 {
            continue;
        }
        let mut step = 0.2;
        if step * (n - 1) as f64 > 1.0 {
            step = 1.0 / (n - 1) as f64;
        }
        for (k, &i) in group.iter().enumerate() {
            let offset = (k as f64 - (n - 1) as f64 / 2.0) * step;
            bows[i] = if names[src[i]] <= names[tgt[i]] {
                offset
            } else {
                -offset
            };
        }
    }
    (bows, loop_sizes)
}

impl SceneBuilder {
    /// Each node's value for `label_by=`, `size_by=` or `color_by=` `key`: the ID for `"id"`, a
    /// stored attribute of that name, or else the label (falling back to the ID) for
    /// `"label"` and the degree for `"degree"`
    fn node_values(
        &self,
        py: Python<'_>,
        key: Option<&Bound<PyAny>>,
        param: &str,
    ) -> PyResult<Values> {
        let n = self.ids.len();
        let Some(key) = key else {
            return Ok((0..n).map(|_| None).collect());
        };
        if key.eq("id")? {
            return Ok(self
                .names
                .iter()
                .map(|s| Some(PyString::new(py, s).into_any().unbind()))
                .collect());
        }
        let mut has = false;
        for attrs in self.node_attrs.iter().flatten() {
            if attrs.bind(py).contains(key)? {
                has = true;
                break;
            }
        }
        if has {
            return self
                .node_attrs
                .iter()
                .map(|a| match a {
                    Some(a) => Ok(a
                        .bind(py)
                        .get_item(key)?
                        .filter(|v| !v.is_none())
                        .map(Bound::unbind)),
                    None => Ok(None),
                })
                .collect();
        }
        if key.eq("label")? {
            return Ok((0..n)
                .map(|k| {
                    let text = self.node_labels[k].as_deref().unwrap_or(&self.names[k]);
                    Some(PyString::new(py, text).into_any().unbind())
                })
                .collect());
        }
        if key.eq("degree")? {
            return self
                .degrees
                .iter()
                .map(|&degree| Ok(Some(degree.into_pyobject(py)?.into_any().unbind())))
                .collect();
        }
        let key = key.repr()?;
        Err(PyValueError::new_err(format!(
            "{param}={key}: no node has an attribute {key} (built-in options are 'id', 'label' \
             and 'degree')"
        )))
    }

    /// Each edge's value of the edge attribute `key` (or its weight, for `"weight"`), or None
    /// if no edge has the attribute
    fn edge_values(&self, py: Python<'_>, key: &Bound<PyAny>) -> PyResult<Option<Values>> {
        let mut has = false;
        for attrs in self.edge_attrs.iter().flatten() {
            if attrs.bind(py).contains(key)? {
                has = true;
                break;
            }
        }
        if has {
            return self
                .edge_attrs
                .iter()
                .map(|a| match a {
                    Some(a) => Ok(a
                        .bind(py)
                        .get_item(key)?
                        .filter(|v| !v.is_none())
                        .map(Bound::unbind)),
                    None => Ok(None),
                })
                .collect::<PyResult<_>>()
                .map(Some);
        }
        if key.eq("weight")? && self.weights.iter().any(Option::is_some) {
            return Ok(Some(
                self.weights
                    .iter()
                    .map(|w| w.map(|w| PyFloat::new(py, w).into_any().unbind()))
                    .collect(),
            ));
        }
        Ok(None)
    }

    /// The edges, found from each node's neighbors, a pair of nodes once (either way round in
    /// an undirected graph), with all the edges that join them
    fn read_edges(&mut self, py: Python<'_>, graph: &Graph) {
        let mut seen: FxHashSet<(u64, u64)> = FxHashSet::default();
        for i in 0..self.ids.len() {
            let nid = self.ids[i];
            let Ok(neighbors) = graph.neighbors(nid) else {
                continue;
            };
            for nb in neighbors {
                let pair = if self.directed || nid <= nb {
                    (nid, nb)
                } else {
                    (nb, nid)
                };
                if !seen.insert(pair) {
                    continue;
                }
                let Some(&j) = self.index.get(&nb) else {
                    continue;
                };
                let eids = graph.find_edge_ids(nid, nb);
                if eids.is_empty() {
                    self.push_edge(i, j, None, (i, j), None, None);
                }
                for eid in eids {
                    let edge = graph.edge_by_id(eid);
                    let ends = edge.and_then(|e| {
                        Some((*self.index.get(&e.source)?, *self.index.get(&e.target)?))
                    });
                    match (edge, ends) {
                        (Some(e), Some(ends)) => {
                            let attrs = non_empty(py, e.attributes.as_ref());
                            self.push_edge(i, j, Some(eid), ends, e.weight, attrs);
                        }
                        _ => self.push_edge(i, j, Some(eid), (i, j), None, None),
                    }
                }
            }
        }
    }

    /// Which of the legend's swatches (`keys`, as it draws them) each node and edge falls under:
    /// 0 for none, else the swatch's place plus 1
    fn legend_keys(&self, keys: &[Key]) -> (Vec<u32>, Vec<u32>) {
        let enc = self.enc.as_ref().expect("encoded");
        let mut node_key = vec![0u32; self.ids.len()];
        let mut edge_key = vec![0u32; self.src.len()];
        for (place, key) in keys.iter().enumerate() {
            let place = place as u32 + 1;
            match (*key, &enc.edge_colors) {
                (Key::Nodes(g), _) => {
                    let Some(groups) = &enc.node_groups else {
                        continue;
                    };
                    for (k, &of) in groups.of.iter().enumerate() {
                        if of == g && node_key[k] == 0 {
                            node_key[k] = place;
                        }
                    }
                }
                (Key::EdgeColor(c), EdgeColors::Each(colors)) => {
                    for (i, &own) in colors.iter().enumerate() {
                        if own == Some(c) && edge_key[i] == 0 {
                            edge_key[i] = place;
                        }
                    }
                }
                (Key::Within(_) | Key::Cross, EdgeColors::Groups { groups, .. }) => {
                    for i in 0..self.src.len() {
                        let (a, b) = (self.src[i], self.tgt[i]);
                        let falls = match *key {
                            Key::Within(g) => groups.same(a, b) && groups.of[a] == Some(g),
                            _ => {
                                !groups.same(a, b)
                                    && (groups.of[a].is_some() || groups.of[b].is_some())
                            }
                        };
                        if falls && edge_key[i] == 0 {
                            edge_key[i] = place;
                        }
                    }
                }
                _ => {}
            }
        }
        (node_key, edge_key)
    }

    fn push_edge(
        &mut self,
        src: usize,
        tgt: usize,
        id: Option<u64>,
        ends: (usize, usize),
        weight: Option<f64>,
        attrs: Option<Py<PyDict>>,
    ) {
        self.src.push(src);
        self.tgt.push(tgt);
        self.edge_ids.push(id);
        self.tip_ends.push(ends);
        self.weights.push(weight);
        self.edge_attrs.push(attrs);
    }

    /// Colors for numbers `nums` on the sequential or diverging `scale` (missing ones in
    /// `missing`), and the (low, high, midpoint) the scale's ends stand for
    fn numeric_colors(
        &self,
        pairs: &mut Pairs,
        nums: &[Option<f64>],
        scale: &str,
        midpoint: Option<&Bound<PyAny>>,
        missing: usize,
    ) -> PyResult<(Vec<usize>, (f64, f64, Option<f64>))> {
        let mut present = nums.iter().flatten().copied();
        let Some(first) = present.next() else {
            return Ok((vec![missing; nums.len()], (0.0, 0.0, None)));
        };
        let (lo, hi) = present.fold((first, first), |(lo, hi), v| (py_min(lo, v), py_max(hi, v)));
        let (stops, base, extent, ends) = if scale == "diverging" {
            let mid = match midpoint {
                Some(m) if !m.is_none() => py_float(m)?,
                _ => 0.0,
            };
            let mut extent = py_max((hi - mid).abs(), (mid - lo).abs());
            if extent == 0.0 {
                extent = 1.0;
            }
            let ends = (mid - extent, mid + extent, Some(mid));
            (&self.colors.diverging, mid, extent, ends)
        } else {
            let mut span = hi - lo;
            if span == 0.0 {
                span = 1.0;
            }
            (&self.colors.sequential, lo, span, (lo, hi, None))
        };
        let mut scale = Scale::new(stops.clone());
        let mut ids: HashMap<Pair, usize> = HashMap::new();
        let mut out = Vec::with_capacity(nums.len());
        for v in nums {
            out.push(match v {
                Some(v) => {
                    let pair = scale.at((v - base) / extent)?;
                    *ids.entry(pair).or_insert_with_key(|p| pairs.id(p.clone()))
                }
                None => missing,
            });
        }
        Ok((out, ends))
    }
}

#[pymethods]
impl SceneBuilder {
    #[new]
    #[pyo3(signature = (
        graph,
        colors,
        label_by=None,
        size_by=None,
        color_by=None,
        edge_color_by=None,
        edge_width_by=None,
        edge_label_by=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        graph: PyRef<'_, Graph>,
        colors: SceneColors,
        label_by: Option<Bound<'_, PyAny>>,
        size_by: Option<Bound<'_, PyAny>>,
        color_by: Option<Bound<'_, PyAny>>,
        edge_color_by: Option<Bound<'_, PyAny>>,
        edge_width_by: Option<Bound<'_, PyAny>>,
        edge_label_by: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let ids = graph.get_node_ids();
        let names: Vec<String> = ids.iter().map(u64::to_string).collect();
        let index = ids.iter().enumerate().map(|(k, &id)| (id, k)).collect();
        let directed = graph.is_directed();
        let degrees = ids
            .iter()
            .map(|&id| {
                if directed {
                    graph.get_in_degree(id) + graph.get_out_degree(id)
                } else {
                    graph.degree_rust(id)
                }
            })
            .collect();
        let mut node_labels = Vec::with_capacity(ids.len());
        let mut node_attrs = Vec::with_capacity(ids.len());
        for &id in &ids {
            let node = graph.node_by_id(id);
            node_labels.push(node.and_then(|n| n.label.clone()));
            node_attrs.push(non_empty(py, node.and_then(|n| n.attributes.as_ref())));
        }
        let mut b = SceneBuilder {
            colors,
            directed,
            ids,
            degrees,
            names,
            index,
            node_labels,
            node_attrs,
            labels: Vec::new(),
            src: Vec::new(),
            tgt: Vec::new(),
            edge_ids: Vec::new(),
            tip_ends: Vec::new(),
            weights: Vec::new(),
            edge_attrs: Vec::new(),
            color_by: color_by.as_ref().map(|k| k.clone().unbind()),
            color_values: Vec::new(),
            size_values: None,
            edge_color_by: edge_color_by.as_ref().map(|k| k.clone().unbind()),
            edge_color: EdgeColorBy::Off,
            edge_width_values: None,
            edge_labels: None,
            enc: None,
            levels: None,
        };
        let label_values = b.node_values(py, label_by.as_ref(), "label_by")?;
        let size_values = b.node_values(py, size_by.as_ref(), "size_by")?;
        b.color_values = b.node_values(py, color_by.as_ref(), "color_by")?;
        let mut labels = Vec::with_capacity(b.ids.len());
        for (name, v) in b.names.iter().zip(&label_values) {
            labels.push(match v {
                Some(v) => v.bind(py).str()?.to_str()?.to_string(),
                None => name.clone(),
            });
        }
        b.labels = labels;
        if truthy(&size_by)? {
            b.size_values = Some(size_values);
        }
        b.read_edges(py, &graph);
        // edge_color_by= names an edge attribute, or else a node attribute whose groups the
        // edges take on
        if let Some(key) = edge_color_by
            .as_ref()
            .filter(|k| k.is_truthy().unwrap_or(false))
        {
            b.edge_color = match b.edge_values(py, key)? {
                Some(values) => EdgeColorBy::Edges(values),
                None => EdgeColorBy::Nodes(b.node_values(py, Some(key), "edge_color_by")?),
            };
        }
        if let Some(key) = edge_width_by
            .as_ref()
            .filter(|k| k.is_truthy().unwrap_or(false))
        {
            match b.edge_values(py, key)? {
                Some(values) => b.edge_width_values = Some(values),
                None => {
                    let key = key.repr()?;
                    return Err(PyValueError::new_err(format!(
                        "edge_width_by={key}: no edge has an attribute {key}"
                    )));
                }
            }
        }
        if let Some(key) = edge_label_by
            .as_ref()
            .filter(|k| k.is_truthy().unwrap_or(false))
        {
            let values = b
                .edge_values(py, key)?
                .unwrap_or_else(|| (0..b.src.len()).map(|_| None).collect());
            let mut texts = Vec::with_capacity(values.len());
            for v in values {
                texts.push(match v {
                    Some(v) => Some(v.bind(py).str()?.to_str()?.to_string()),
                    None => None,
                });
            }
            b.edge_labels = Some(texts);
        }
        Ok(b)
    }

    /// The nodes' IDs as text, in the order of the graph's node IDs
    #[getter]
    fn node_names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// The edges' ends as node IDs in text
    #[getter]
    fn edge_names(&self) -> Vec<(String, String)> {
        self.src
            .iter()
            .zip(&self.tgt)
            .map(|(&a, &b)| (self.names[a].clone(), self.names[b].clone()))
            .collect()
    }

    /// The nodes' radii, once encoded
    #[getter]
    fn radii(&self) -> PyResult<Vec<f64>> {
        match &self.enc {
            Some(enc) => Ok(enc.radii.clone()),
            None => Err(PyValueError::new_err("the radii come from encode()")),
        }
    }

    /// Map the values to colors (from `palette`, or on a numeric scale), sizes and widths, and
    /// return the legend's blocks: those that can tell their values apart with no more than
    /// `legend_max` items, or none for None
    #[pyo3(signature = (
        palette,
        color_scale,
        color_midpoint,
        cross_community_color,
        node_size,
        edge_width,
        edge_opacity,
        legend_max,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        palette: Vec<String>,
        color_scale: Option<String>,
        color_midpoint: Option<Bound<'py, PyAny>>,
        cross_community_color: Option<String>,
        node_size: f64,
        edge_width: f64,
        mut edge_opacity: f64,
        legend_max: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let n = self.ids.len();
        let m = self.src.len();
        let color_by_on = match &self.color_by {
            Some(k) => k.bind(py).is_truthy()?,
            None => false,
        };
        let edge_color_by_on = !matches!(self.edge_color, EdgeColorBy::Off);
        if (color_by_on || edge_color_by_on) && palette.is_empty() {
            return Err(PyValueError::new_err("the palette has no colors"));
        }
        let mut pairs = Pairs::default();
        let mut blocks: Vec<Block> = Vec::new();
        let missing = pairs.id(self.colors.missing.clone());
        let text =
            |v: &PyObject| -> PyResult<PyObject> { Ok(v.bind(py).str()?.into_any().unbind()) };
        let no_value = || PyString::new(py, "no value").into_any().unbind();
        let title =
            |k: &Option<PyObject>| k.as_ref().map_or_else(|| py.None(), |k| k.clone_ref(py));

        // The nodes' colors
        let mut node_groups = None;
        let node_color = if color_by_on {
            let nums = numbers(py, &self.color_values)?;
            let scale = pick_scale(
                py,
                &self.color_values,
                &nums,
                color_scale.as_deref(),
                "color_by",
            )?;
            if scale == "categorical" {
                // Nodes without a value are gray, as on a numeric scale, rather than taking a hue
                let groups = Groups::new(py, &self.color_values)?;
                let ranks = groups.ranks(py)?;
                let colors: Vec<usize> = ranks
                    .iter()
                    .map(|&r| pairs.same(&palette[r % palette.len()]))
                    .collect();
                let mut items = Vec::new();
                for g in groups.sorted(&ranks) {
                    items.push((
                        "node",
                        colors[g],
                        text(&groups.firsts[g])?,
                        Key::Nodes(Some(g)),
                    ));
                }
                if groups.of.iter().any(Option::is_none) {
                    items.push(("node", missing, no_value(), Key::Nodes(None)));
                }
                blocks.push(Block::Swatches {
                    title: title(&self.color_by),
                    items,
                });
                let node_color = groups
                    .of
                    .iter()
                    .map(|g| g.map_or(missing, |g| colors[g]))
                    .collect();
                node_groups = Some(groups);
                node_color
            } else {
                let (colors, ends) = self.numeric_colors(
                    &mut pairs,
                    &nums,
                    &scale,
                    color_midpoint.as_ref(),
                    missing,
                )?;
                blocks.push(Block::Ramp {
                    title: title(&self.color_by),
                    scale,
                    ends,
                    missing: Some(nums.iter().any(Option::is_none)),
                });
                colors
            }
        } else {
            vec![pairs.id(self.colors.node.clone()); n]
        };

        // The edges' colors: by an edge attribute's values, or by the groups of the ends
        let edge_colors = match &self.edge_color {
            EdgeColorBy::Off => EdgeColors::Plain,
            EdgeColorBy::Edges(values) => {
                let nums = numbers(py, values)?;
                let scale = pick_scale(py, values, &nums, None, "edge_color_by")?;
                if scale == "categorical" {
                    let groups = Groups::new(py, values)?;
                    let ranks = groups.ranks(py)?;
                    let colors: Vec<usize> = ranks
                        .iter()
                        .map(|&r| pairs.same(&palette[r % palette.len()]))
                        .collect();
                    let mut items = Vec::new();
                    for g in groups.sorted(&ranks) {
                        let key = Key::EdgeColor(colors[g]);
                        items.push(("edge", colors[g], text(&groups.firsts[g])?, key));
                    }
                    blocks.push(Block::Swatches {
                        title: title(&self.edge_color_by),
                        items,
                    });
                    EdgeColors::Each(groups.of.iter().map(|g| g.map(|g| colors[g])).collect())
                } else {
                    let (colors, ends) =
                        self.numeric_colors(&mut pairs, &nums, &scale, None, missing)?;
                    blocks.push(Block::Ramp {
                        title: title(&self.edge_color_by),
                        scale,
                        ends,
                        missing: None,
                    });
                    // A color scale reads poorly through the default, faint edges
                    if edge_opacity == 0.4 {
                        edge_opacity = 0.85;
                    }
                    EdgeColors::Each(
                        colors
                            .into_iter()
                            .zip(&nums)
                            .map(|(c, v)| v.map(|_| c))
                            .collect(),
                    )
                }
            }
            EdgeColorBy::Nodes(values) => {
                let groups = Groups::new(py, values)?;
                let ranks = groups.ranks(py)?;
                let colors: Vec<usize> = ranks
                    .iter()
                    .map(|&r| pairs.same(&palette[r % palette.len()]))
                    .collect();
                let cross = match cross_community_color.as_deref().filter(|c| !c.is_empty()) {
                    Some(c) => pairs.same(c),
                    None => pairs.id(self.colors.cross.clone()),
                };
                let cross_item = || {
                    (
                        "edge",
                        cross,
                        PyString::new(py, "between groups").into_any().unbind(),
                        Key::Cross,
                    )
                };
                let same_key = match (&self.edge_color_by, &self.color_by) {
                    (Some(a), Some(b)) => a.bind(py).eq(b.bind(py))?,
                    _ => false,
                };
                match blocks.first_mut() {
                    Some(Block::Swatches { items, .. }) if same_key => items.push(cross_item()),
                    _ => {
                        let mut items = Vec::new();
                        for g in groups.sorted(&ranks) {
                            items.push((
                                "edge",
                                colors[g],
                                text(&groups.firsts[g])?,
                                Key::Within(g),
                            ));
                        }
                        items.push(cross_item());
                        blocks.push(Block::Swatches {
                            title: title(&self.edge_color_by),
                            items,
                        });
                    }
                }
                EdgeColors::Groups {
                    groups,
                    colors,
                    cross,
                }
            }
        };

        // Edge widths from edge_width_by=, from half to two and a half times the base width
        let mut widths = vec![edge_width; m];
        if let Some(values) = &self.edge_width_values {
            let nums = numbers(py, values)?;
            let mut present = nums.iter().flatten().copied();
            if let Some(first) = present.next() {
                let (lo, hi) =
                    present.fold((first, first), |(lo, hi), v| (py_min(lo, v), py_max(hi, v)));
                let mut span = hi - lo;
                if span == 0.0 {
                    span = 1.0;
                }
                for (w, v) in widths.iter_mut().zip(&nums) {
                    if let Some(v) = v {
                        *w = edge_width * (0.5 + 2.0 * (v - lo) / span);
                    }
                }
            }
        }

        // Node sizes from size_by=, from half to twice the base size
        let mut radii = vec![node_size; n];
        if let Some(values) = &self.size_values {
            let mut sizes = Vec::with_capacity(n);
            for v in values {
                sizes.push(match v {
                    Some(v) => match py_float(v.bind(py)) {
                        Ok(x) => Some(x),
                        Err(e)
                            if e.is_instance_of::<PyTypeError>(py)
                                || e.is_instance_of::<PyValueError>(py) =>
                        {
                            None
                        }
                        Err(e) => return Err(e),
                    },
                    // float(None) is a TypeError
                    None => None,
                });
            }
            let mut valid = sizes.iter().flatten().copied();
            if let Some(first) = valid.next() {
                let (lo, hi) =
                    valid.fold((first, first), |(lo, hi), v| (py_min(lo, v), py_max(hi, v)));
                let range = if hi != lo { hi - lo } else { 1.0 };
                for (r, v) in radii.iter_mut().zip(&sizes) {
                    if let Some(v) = v {
                        *r = node_size * (0.5 + (v - lo) / range * 1.5);
                    }
                }
            }
        }

        // Legends are shown only for what they can tell apart: a palette's colors repeat after
        // its length, so a categorical block with more values than that is left out
        let out = PyList::empty(py);
        let mut keys = Vec::new();
        if let Some(max) = legend_max {
            let n_colors = if color_by_on || edge_color_by_on {
                palette.len()
            } else {
                0
            };
            for block in &blocks {
                let dict = PyDict::new(py);
                match block {
                    Block::Swatches { title, items } => {
                        let count = items
                            .iter()
                            .filter(|(_, _, t, _)| {
                                !t.bind(py).eq("between groups").unwrap_or(false)
                            })
                            .count();
                        if count > n_colors.min(max) {
                            continue;
                        }
                        dict.set_item("title", title)?;
                        dict.set_item("kind", "swatches")?;
                        let list = PyList::empty(py);
                        for (shape, color, text, key) in items {
                            keys.push(*key);
                            let pair = &pairs.list[*color];
                            list.append(PyTuple::new(
                                py,
                                [
                                    PyString::new(py, shape).into_any(),
                                    (pair.0.as_str(), pair.1.as_str())
                                        .into_pyobject(py)?
                                        .into_any(),
                                    text.bind(py).clone(),
                                ],
                            )?)?;
                        }
                        dict.set_item("items", list)?;
                    }
                    Block::Ramp {
                        title,
                        scale,
                        ends,
                        missing,
                    } => {
                        dict.set_item("title", title)?;
                        dict.set_item("kind", "ramp")?;
                        dict.set_item("scale", scale)?;
                        dict.set_item("ends", *ends)?;
                        if let Some(missing) = missing {
                            dict.set_item("missing", missing)?;
                        }
                    }
                }
                out.append(dict)?;
            }
        }

        self.enc = Some(Encoded {
            pairs,
            node_color,
            node_groups,
            edge_colors,
            widths,
            opacity: edge_opacity,
            radii,
            node_size,
            edge_width,
            keys,
        });
        Ok(out)
    }

    /// Highlight what `picked` holds: `{level: (node IDs, edge IDs)}` for `"accent"` and
    /// `"soft"`, everything else being muted
    fn highlight(&mut self, picked: HashMap<String, (HashSet<u64>, HashSet<u64>)>) {
        let mut node_level = vec![0u8; self.ids.len()];
        let mut edge_level = vec![0u8; self.src.len()];
        let edge_index: FxHashMap<u64, usize> = self
            .edge_ids
            .iter()
            .enumerate()
            .filter_map(|(i, id)| id.map(|id| (id, i)))
            .collect();
        for (level, rank) in [("soft", 1), ("accent", 2)] {
            let Some((nodes, edges)) = picked.get(level) else {
                continue;
            };
            for id in nodes {
                if let Some(&k) = self.index.get(id) {
                    node_level[k] = rank;
                }
            }
            for id in edges {
                if let Some(&i) = edge_index.get(id) {
                    edge_level[i] = rank;
                }
            }
        }
        self.levels = Some((node_level, edge_level));
    }

    /// The scene, with the nodes at `positions` (`{node ID as text: (x, y)}`) in a frame of
    /// `width` by `height` pixels. `legend` holds the legend's marks, `legend_w` pixels wide, in
    /// a column at the right, or with a `legend_h`, that high, in a band along the bottom, and
    /// `route` the margin and greatest number of waypoints for routing edges around nodes,
    /// None for straight edges. With `show_labels`, at most `max_labels` labels are drawn (see
    /// `labels.rs`), or with None, every one up to `labels::ALL_MAX` nodes and otherwise
    /// `labels::AUTO_MAX`. More than `lean_min` nodes or edges make it lean, and `viewer` keeps
    /// what the viewer needs: the tooltips as tables, and what its controls change
    #[pyo3(signature = (
        positions,
        width,
        height,
        curved,
        show_labels,
        legend,
        legend_w,
        route=None,
        max_labels=None,
        lean_min=2000,
        viewer=false,
        legend_h=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn scene(
        &self,
        py: Python<'_>,
        positions: &Bound<'_, PyDict>,
        width: f64,
        height: f64,
        curved: bool,
        show_labels: bool,
        legend: Bound<'_, PyAny>,
        legend_w: f64,
        route: Option<(f64, Option<usize>)>,
        max_labels: Option<usize>,
        lean_min: usize,
        viewer: bool,
        legend_h: f64,
    ) -> PyResult<Scene> {
        let Some(enc) = &self.enc else {
            return Err(PyValueError::new_err("encode() comes before scene()"));
        };
        let n = self.ids.len();
        let m = self.src.len();

        // The positions, and the nodes in the order the layout gave them, the order the
        // viewbox takes them in
        let by_name: FxHashMap<&str, usize> = self
            .names
            .iter()
            .enumerate()
            .map(|(k, s)| (s.as_str(), k))
            .collect();
        let (mut xs, mut ys) = (vec![0.0; n], vec![0.0; n]);
        let mut placed = vec![false; n];
        let mut order = Vec::with_capacity(n);
        for (key, p) in positions.iter() {
            let Ok(name) = key.downcast::<PyString>() else {
                continue;
            };
            let Some(&k) = by_name.get(name.to_str()?) else {
                continue;
            };
            xs[k] = p.get_item(0)?.extract()?;
            ys[k] = p.get_item(1)?.extract()?;
            if !placed[k] {
                placed[k] = true;
                order.push(k);
            }
        }
        if let Some(k) = placed.iter().position(|p| !p) {
            return Err(PyKeyError::new_err(self.names[k].clone()));
        }

        // The legend gets a column of its own at the right, or a band along the bottom, so that
        // it never covers the drawing
        let band = py_min(legend_h, height * 0.4);
        let column = if legend_h > 0.0 {
            0.0
        } else {
            py_min(legend_w, width * 0.4)
        };
        let pick = |v: &[f64]| order.iter().map(|&k| v[k]).collect::<Vec<_>>();
        // The labels drawn, which the frame makes room for. Their font is in the frame's units,
        // as the nodes are, so which of them overlap doesn't depend on the frame
        let levels = self.levels.as_ref();
        let font_size = py_max(8.0, py_min(13.0, enc.node_size * 0.7));
        let auto_limit = labels::auto_limit(n);
        let label_limit = max_labels.or(auto_limit);
        let mut degree = vec![0usize; n];
        for (&s, &t) in self.src.iter().zip(&self.tgt) {
            degree[s] += 1;
            degree[t] += 1;
        }
        let label_order = labels::priority(
            levels.map(|(node_level, _)| node_level.as_slice()),
            &enc.radii,
            &degree,
        );
        let shown = if show_labels {
            let (radii, texts) = (&enc.radii, &self.labels);
            labels::choose(texts, &xs, &ys, radii, font_size, &label_order, label_limit)
        } else {
            vec![false; n]
        };
        let any_shown = shown.iter().any(|&s| s);
        let chars: Option<Vec<usize>> = any_shown.then(|| {
            order
                .iter()
                .map(|&k| {
                    if shown[k] {
                        self.labels[k].chars().count()
                    } else {
                        0
                    }
                })
                .collect()
        });
        let mut viewbox = fit_viewbox(
            &pick(&xs),
            &pick(&ys),
            &pick(&enc.radii),
            chars.as_deref(),
            if any_shown { 20.0 } else { 0.0 },
            30.0,
            Some((width - column, height - band)),
        );
        if legend.is_truthy()? {
            // Viewbox units per pixel
            if legend_h > 0.0 {
                let unit = viewbox.2 / width;
                viewbox.3 += band * unit;
            } else {
                let unit = viewbox.3 / height;
                viewbox.2 += column * unit;
            }
        }
        let legend_w = if legend_h > 0.0 { legend_w } else { column };

        // Edges routed around the nodes in their way, once for each pair of ends however many
        // edges join them
        let mut routes: Vec<Option<Vec<(f64, f64)>>> = vec![None; m];
        if let Some((margin, max_waypoints)) = route {
            let mut pair_of: FxHashMap<(usize, usize), usize> = FxHashMap::default();
            let mut pairs = Vec::new();
            let mut edge_pair = Vec::with_capacity(m);
            for i in 0..m {
                let pair = (self.src[i], self.tgt[i]);
                let k = *pair_of.entry(pair).or_insert_with(|| {
                    pairs.push(pair);
                    pairs.len() - 1
                });
                edge_pair.push(k);
            }
            let found = route_edges(
                py,
                xs.clone(),
                ys.clone(),
                enc.radii.clone(),
                pairs,
                margin,
                max_waypoints,
            )?;
            // None when the routes would bend too much
            if let Some(found) = found {
                for (route, &k) in routes.iter_mut().zip(&edge_pair) {
                    route.clone_from(&found[k]);
                }
            }
        }
        let (bows, loop_sizes) = fans(&self.names, &self.src, &self.tgt);

        // The colors, as indexes into the scene's table of those it uses
        let mut pairs = enc.pairs.clone();
        let mut table = Table {
            colors: Vec::new(),
            of: Vec::new(),
        };
        let edge_default = pairs.id(self.colors.edge.clone());
        let accent = pairs.id(self.colors.accent.clone());
        let muted_node = pairs.id(self.colors.muted_node.clone());
        let muted_label = pairs.id(self.colors.muted_label.clone());
        let label = pairs.id(self.colors.label.clone());
        let edge_color_by_on = !matches!(self.edge_color, EdgeColorBy::Off);
        let color_by_on = match &self.color_by {
            Some(k) => k.bind(py).is_truthy()?,
            None => false,
        };

        // Edges
        let mut edge_color = Vec::with_capacity(m);
        let mut edge_width = Vec::with_capacity(m);
        let mut edge_opacity = Vec::with_capacity(m);
        // For the viewer, each edge's color before highlighting changed it
        let mut edge_base = Vec::new();
        for i in 0..m {
            let (a, b) = (self.src[i], self.tgt[i]);
            let mut c = edge_default;
            match &enc.edge_colors {
                EdgeColors::Each(colors) => {
                    if let Some(own) = colors[i] {
                        c = own;
                    }
                }
                EdgeColors::Groups {
                    groups,
                    colors,
                    cross,
                } => {
                    if groups.same(a, b) {
                        c = colors[groups.of[a].expect("a node in a group")];
                    } else if groups.of[a].is_some() || groups.of[b].is_some() {
                        c = *cross;
                    }
                }
                EdgeColors::Plain => {
                    if color_by_on && enc.node_groups.as_ref().is_some_and(|g| g.same(a, b)) {
                        c = enc.node_color[a];
                    }
                }
            }
            let mut w = enc.widths[i];
            let mut opacity = enc.opacity;
            if viewer && levels.is_some() {
                edge_base.push(c);
            }
            if let Some((_, edge_level)) = levels {
                match edge_level[i] {
                    0 => {
                        c = edge_default;
                        opacity = enc.opacity * 0.5;
                    }
                    level => {
                        // A highlighted edge keeps a color that edge_color_by= gives it, or
                        // else takes the accent (rather than its group's color from
                        // color_by=, so that a path or matching is all of one color)
                        if !edge_color_by_on || c == edge_default {
                            c = accent;
                        }
                        if level == 2 {
                            w = py_max(w * 2.0, 1.5);
                            opacity = py_max(enc.opacity, 0.9);
                        } else {
                            w = py_max(w * 1.3, 1.0);
                            opacity = py_max(enc.opacity, 0.5);
                        }
                    }
                }
            }
            edge_color.push(table.index(&pairs, c));
            edge_width.push(w);
            edge_opacity.push(opacity);
        }
        // Highlighted elements are drawn over the muted ones, and accent over soft
        let mut edge_order: Vec<usize> = (0..m).collect();
        if let Some((_, edge_level)) = levels {
            edge_order.sort_by_key(|&i| edge_level[i]);
        }

        // Nodes
        let surface = &self.colors.surface;
        let mut softened: FxHashMap<usize, usize> = FxHashMap::default();
        let mut darkened: FxHashMap<usize, usize> = FxHashMap::default();
        let mut fill = Vec::with_capacity(n);
        let mut border = Vec::with_capacity(n);
        let mut border_width = Vec::with_capacity(n);
        let mut border_opacity = Vec::with_capacity(n);
        for k in 0..n {
            let mut c = enc.node_color[k];
            let (mut bo, mut bw) = (0.3, 1.5);
            let mut ring = None;
            if let Some((node_level, _)) = levels {
                if !color_by_on {
                    c = accent;
                }
                match node_level[k] {
                    0 => c = muted_node,
                    1 => {
                        // A ring of the color around a pale version of it
                        ring = Some(c);
                        c = match softened.get(&c) {
                            Some(&s) => s,
                            None => {
                                let (l, d) = pairs.list[c].clone();
                                let s = pairs
                                    .id((mix(&l, &surface.0, 0.7)?, mix(&d, &surface.1, 0.7)?));
                                softened.insert(c, s);
                                s
                            }
                        };
                        (bo, bw) = (1.0, 2.0);
                    }
                    _ => {}
                }
            }
            fill.push(table.index(&pairs, c));
            let edge = match ring {
                Some(r) => r,
                None => darker(&mut pairs, &mut darkened, c)?,
            };
            border.push(table.index(&pairs, edge));
            border_width.push(bw);
            border_opacity.push(bo);
        }
        let mut node_order: Vec<usize> = (0..n).collect();
        if let Some((node_level, _)) = levels {
            node_order.sort_by_key(|&k| node_level[k]);
        }
        let mut label_colors = Vec::with_capacity(n);
        for k in 0..n {
            let muted = levels.is_some_and(|(node_level, _)| node_level[k] == 0);
            label_colors.push(table.index(&pairs, if muted { muted_label } else { label }));
        }

        // For the viewer's selection, the colors it draws with, after those the scene uses
        let main = table.colors.len();
        let selection = if viewer {
            let accent_border = darker(&mut pairs, &mut darkened, accent)?;
            let muted_border = darker(&mut pairs, &mut darkened, muted_node)?;
            let base = match levels {
                Some(_) => {
                    let mut base_fill = Vec::with_capacity(n);
                    let mut base_border = Vec::with_capacity(n);
                    for &c in &enc.node_color {
                        base_fill.push(table.index(&pairs, c));
                        let dk = darker(&mut pairs, &mut darkened, c)?;
                        base_border.push(table.index(&pairs, dk));
                    }
                    Some(Base {
                        fill: base_fill,
                        border: base_border,
                        color: edge_base.iter().map(|&c| table.index(&pairs, c)).collect(),
                        label_order: labels::priority(None, &enc.radii, &degree),
                    })
                }
                None => None,
            };
            let keys =
                (legend.is_truthy()? && !enc.keys.is_empty()).then(|| self.legend_keys(&enc.keys));
            Some(Selection {
                accent: table.index(&pairs, accent),
                accent_border: table.index(&pairs, accent_border),
                muted: table.index(&pairs, muted_node),
                muted_border: table.index(&pairs, muted_border),
                label: table.index(&pairs, label),
                muted_label: table.index(&pairs, muted_label),
                edge: table.index(&pairs, edge_default),
                node_colors: color_by_on,
                edge_colors: edge_color_by_on,
                extra: table.colors.split_off(main),
                base,
                keys,
            })
        } else {
            None
        };

        // For the viewer's filters, the numeric attributes, with the nodes' degree unless an
        // attribute has its name, as `color_by=` takes it, and the edges' weight likewise
        let mut filters = Vec::new();
        if viewer {
            let has = |attrs: &[Option<Py<PyDict>>], key: &str| -> PyResult<bool> {
                for a in attrs.iter().flatten() {
                    if a.bind(py).contains(key)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            };
            if !has(&self.node_attrs, "degree")? {
                let values = self.degrees.iter().map(|&d| d as f64).collect();
                filters.extend(ranged(filter("degree".into(), false, values)));
            }
            filters.extend(numeric_attrs(py, &self.node_attrs, false)?);
            if !has(&self.edge_attrs, "weight")? {
                let values = self.weights.iter().map(|w| w.unwrap_or(f64::NAN)).collect();
                // The weights are floats, whatever they were given as, so whole ones are ints
                let mut weight = filter("weight".into(), true, values);
                weight.integer = weight.values.iter().all(|w| w.is_nan() || w.fract() == 0.0);
                filters.extend(ranged(weight));
            }
            filters.extend(numeric_attrs(py, &self.edge_attrs, true)?);
        }

        // Tooltips: a node's ID, and its label as well when it differs from the ID, since
        // either may be drawn under the node, then its attributes; an edge's ID, ends (named as
        // their labels are drawn, and in the order stored) and weight, then its attributes
        let lean = n > lean_min || m > lean_min;
        let mut node_tips = Tips::new(py);
        let (id_key, label_key) = (intern!(py, "id").as_any(), intern!(py, "label").as_any());
        for k in 0..n {
            let mut head = vec![(node_tips.key(id_key)?, Cell::Text(self.names[k].clone()))];
            if let Some(l) = self.node_labels[k]
                .as_ref()
                .filter(|l| **l != self.names[k])
            {
                head.push((node_tips.key(label_key)?, Cell::Text(l.clone())));
            }
            node_tips.push(head, self.node_attrs[k].as_ref().map(|a| a.bind(py)))?;
        }
        node_tips.round()?;
        let mut edge_tips = Tips::new(py);
        let (edge_key, weight_key) = (intern!(py, "edge").as_any(), intern!(py, "weight").as_any());
        let arrow = if self.directed { "→" } else { "—" };
        for i in 0..m {
            let mut head = Vec::with_capacity(3);
            if let Some(id) = self.edge_ids[i] {
                head.push((edge_tips.key(id_key)?, Cell::Int(id)));
            }
            let (a, b) = self.tip_ends[i];
            let ends = format!("{} {arrow} {}", self.labels[a], self.labels[b]);
            head.push((edge_tips.key(edge_key)?, Cell::Text(ends)));
            if let Some(w) = self.weights[i] {
                head.push((edge_tips.key(weight_key)?, Cell::Float(w)));
            }
            edge_tips.push(head, self.edge_attrs[i].as_ref().map(|a| a.bind(py)))?;
        }
        edge_tips.round()?;
        // In lean SVG, an edge's tooltip is its row in one table of them
        let (edge_tip_table, edge_tip_json) = if lean {
            (Some(edge_tips.table(None)?), Vec::new())
        } else {
            (None, edge_tips.attrs_json()?)
        };
        let tip_tables = if viewer {
            Some((node_tips.table(None)?, edge_tips.table(Some(&edge_order))?))
        } else {
            None
        };
        // For the viewer's cluster force, the groups of a categorical `color_by=`
        let cluster = match (&enc.node_groups, &self.color_by) {
            (Some(groups), Some(name)) if viewer => {
                let ranks = groups.ranks(py)?;
                let of = groups
                    .of
                    .iter()
                    .map(|g| g.map_or(0, |g| ranks[g] as u32 + 1));
                Some((name.bind(py).str()?.to_string(), of.collect()))
            }
            _ => None,
        };
        let controls = selection.map(|selection| Controls {
            node_size: enc.node_size,
            edge_width: enc.edge_width,
            edge_opacity: enc.opacity,
            highlight: levels.map(|(_, edge_level)| (edge_level.clone(), enc.widths.clone())),
            selection,
            filters,
            cluster,
        });

        Ok(Scene {
            width,
            height,
            directed: self.directed,
            curved,
            lean,
            viewbox,
            colors: table.colors,
            legend: legend.unbind(),
            legend_w,
            legend_h,
            edge_tip_table,
            tip_tables,
            controls,
            nodes: Nodes {
                ids: self.names.clone(),
                x: xs,
                y: ys,
                r: enc.radii.clone(),
                fill,
                border,
                border_width,
                border_opacity,
                order: node_order,
                tips: node_tips.attrs_json()?,
            },
            // The viewer has them without `show_labels`, for its controls
            labels: (show_labels || viewer).then(|| Labels {
                texts: self.labels.clone(),
                colors: label_colors,
                font_size,
                show: show_labels,
                order: label_order,
                limit: label_limit,
                auto: auto_limit,
                shown,
            }),
            edges: Edges {
                src: self.src.clone(),
                tgt: self.tgt.clone(),
                color: edge_color,
                width: edge_width,
                opacity: edge_opacity,
                bow: bows,
                loop_size: loop_sizes,
                route: routes,
                order: edge_order,
                tips: edge_tip_json,
            },
            edge_labels: self.edge_labels.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parallel_edges_fan_out_and_loops_grow() {
        // From the Python renderer's _compute_edge_fans()
        let names = names(&["a", "b", "c"]);
        let (bows, loops) = fans(&names, &[0, 0, 1, 2, 2, 0], &[1, 1, 0, 2, 2, 2]);
        assert_eq!(bows, [-0.2, 0.0, -0.2, 0.0, 0.0, 0.0]);
        assert_eq!(loops, [1.0, 1.0, 1.0, 1.0, 1.6, 1.0]);
        let (bows, _) = fans(&names, &[0; 7], &[1; 7]);
        let step = 1.0 / 6.0;
        let want: Vec<f64> = (0..7).map(|k| (k as f64 - 3.0) * step).collect();
        assert_eq!(bows, want);
    }
}
