//! Centrality and link-prediction kernels with NetworkX's arithmetic, over a `PathAdjacency`.
//!
//! These follow NetworkX's pure-Python implementations operation by operation: nodes in the
//! order of the Python dict being iterated, neighbors in adjacency order, sums in the order
//! Python adds them (Python's compensated `sum()`, CPython's `math.hypot`). So their results
//! equal NetworkX's to the last bit, which the backend's tests check. Work is spread over
//! threads only where the order of the floating-point operations stays the same (each
//! source's search runs on its own; their contributions are then added in source order).
//!
//! Nodes are 0-based positions in the adjacency.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use crate::algorithms::path_search::{CompensatedSum, PathAdjacency};

// ============================================================================
// math.hypot
// ============================================================================

/// `x * y` as an unevaluated sum `hi + lo` (exact, using a fused multiply-add)
#[inline]
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

/// `a + b` as an unevaluated sum `hi + lo`, exact for `|a| >= |b|`
#[inline]
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let z = x - a;
    (x, b - z)
}

/// CPython's `vector_norm()`, the core of `math.hypot()`, for the absolute values `vec`
/// whose largest is `max`
fn vector_norm(vec: &mut [f64], max: f64, found_nan: bool) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if found_nan {
        return f64::NAN;
    }
    if max == 0.0 || vec.len() <= 1 {
        return max;
    }
    let max_e = frexp_exponent(max);
    if max_e < -1023 {
        // ldexp(1.0, -max_e) would overflow: scale subnormals up first
        for x in vec.iter_mut() {
            *x /= f64::MIN_POSITIVE;
        }
        return f64::MIN_POSITIVE * vector_norm(vec, max / f64::MIN_POSITIVE, found_nan);
    }
    let scale = ldexp(1.0, -max_e);
    let mut csum = 1.0f64;
    let mut frac1 = 0.0f64;
    let mut frac2 = 0.0f64;
    for &x in vec.iter() {
        let x = x * scale; // lossless scaling
        let (hi, lo) = dl_mul(x, x); // lossless squaring
        let (shi, slo) = dl_fast_sum(csum, hi); // lossless addition
        csum = shi;
        frac1 += lo;
        frac2 += slo;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (hi, lo) = dl_mul(-h, h);
    let (shi, slo) = dl_fast_sum(csum, hi);
    csum = shi;
    frac1 += lo;
    frac2 += slo;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h); // differential correction
    h / scale
}

/// The exponent `e` of `frexp(x) = (m, e)`, with `0.5 <= |m| < 1` (for finite nonzero `x`)
fn frexp_exponent(x: f64) -> i32 {
    let bits = x.abs().to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        // subnormal: normalize first
        let scaled = x.abs() * 2f64.powi(64);
        return frexp_exponent(scaled) - 64;
    }
    exponent - 1022
}

/// `x * 2**e`
fn ldexp(x: f64, e: i32) -> f64 {
    // (2**e may not be representable in one step for large |e|)
    if e > 1023 {
        x * 2f64.powi(1023) * 2f64.powi(e - 1023)
    } else if e < -1022 {
        x * 2f64.powi(-1022) * 2f64.powi(e + 1022)
    } else {
        x * 2f64.powi(e)
    }
}

/// `math.hypot(*values)` exactly as CPython computes it
pub(crate) fn python_hypot(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut vec: Vec<f64> = Vec::new();
    let mut max = 0.0f64;
    let mut found_nan = false;
    for x in values {
        let x = x.abs();
        found_nan |= x.is_nan();
        if x > max {
            max = x;
        }
        vec.push(x);
    }
    vector_norm(&mut vec, max, found_nan)
}

/// Python's `math.hypot(*values)`, bit for bit (as the power iterations here use it)
#[pyfunction]
pub fn nx_hypot(values: Vec<f64>) -> f64 {
    python_hypot(values)
}

// ============================================================================
// Power iterations
// ============================================================================

fn check_positions(n: usize, order: &[usize], vectors: &[(&str, usize)]) -> PyResult<()> {
    if order.len() != n || order.iter().any(|&v| v >= n) {
        return Err(PyValueError::new_err(
            "order must list every node position once",
        ));
    }
    let mut seen = vec![false; n];
    for &v in order {
        if std::mem::replace(&mut seen[v], true) {
            return Err(PyValueError::new_err("order lists a node twice"));
        }
    }
    for &(name, len) in vectors {
        if len != n {
            return Err(PyValueError::new_err(format!(
                "{} must have one value per node",
                name
            )));
        }
    }
    Ok(())
}

/// Add `x_last[v] * weight` to `x[w]` for every arc `v -> w`, visiting `v` in `order` and its
/// arcs in adjacency order (NetworkX's `for n in x: for nbr in G[n]: x[nbr] += ...`)
fn spread(adjacency: &PathAdjacency, order: &[usize], x_last: &[f64], x: &mut [f64]) {
    let (indptr, indices, weights) = adjacency.rows();
    for &v in order {
        let xv = x_last[v];
        for k in indptr[v]..indptr[v + 1] {
            x[indices[k]] += xv * weights[k];
        }
    }
}

/// Python's `sum(abs(x[n] - x_last[n]) for n in x)`
fn change(order: &[usize], x: &[f64], x_last: &[f64]) -> f64 {
    let mut sum = CompensatedSum::default();
    for &v in order {
        sum.add((x[v] - x_last[v]).abs());
    }
    sum.total()
}

/// `networkx.eigenvector_centrality`'s power iteration.
///
/// `order` lists the node positions in the order of NetworkX's dict `x` (the order of
/// `nstart`, or node order); `x0` is the normalized start vector by position. Each step
/// costs the adjacency's arc weights. Returns the centralities by position, or None if
/// `max_iter` steps do not converge.
#[pyfunction]
#[pyo3(signature = (adjacency, order, x0, max_iter=100, tol=1e-6))]
pub fn nx_eigenvector_centrality(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    order: Vec<usize>,
    x0: Vec<f64>,
    max_iter: i64,
    tol: f64,
) -> PyResult<Option<Vec<f64>>> {
    let n = adjacency.len();
    check_positions(n, &order, &[("x0", x0.len())])?;
    Ok(py.allow_threads(|| {
        let mut x = x0;
        let threshold = n as f64 * tol;
        for _ in 0..max_iter.max(0) {
            let x_last = x.clone();
            spread(adjacency, &order, &x_last, &mut x);
            let mut norm = python_hypot(order.iter().map(|&v| x[v]));
            if norm == 0.0 {
                norm = 1.0;
            }
            for value in x.iter_mut() {
                *value /= norm;
            }
            if change(&order, &x, &x_last) < threshold {
                return Some(x);
            }
        }
        None
    }))
}

/// `networkx.katz_centrality`'s iteration (`x = alpha * A^T x + beta`), as
/// `nx_eigenvector_centrality` with `beta` per position; `normalized` scales the result to
/// unit length.
#[pyfunction]
#[pyo3(signature = (adjacency, order, x0, beta, alpha=0.1, max_iter=1000, tol=1e-6,
                    normalized=true))]
#[allow(clippy::too_many_arguments)]
pub fn nx_katz_centrality(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    order: Vec<usize>,
    x0: Vec<f64>,
    beta: Vec<f64>,
    alpha: f64,
    max_iter: i64,
    tol: f64,
    normalized: bool,
) -> PyResult<Option<Vec<f64>>> {
    let n = adjacency.len();
    check_positions(n, &order, &[("x0", x0.len()), ("beta", beta.len())])?;
    Ok(py.allow_threads(|| {
        let mut x = x0;
        let threshold = n as f64 * tol;
        for _ in 0..max_iter.max(0) {
            let x_last = std::mem::replace(&mut x, vec![0.0; n]);
            spread(adjacency, &order, &x_last, &mut x);
            for &v in &order {
                x[v] = alpha * x[v] + beta[v];
            }
            if change(&order, &x, &x_last) < threshold {
                if normalized {
                    let norm = python_hypot(order.iter().map(|&v| x[v]));
                    // (NetworkX catches the ZeroDivisionError of 1.0 / 0.0)
                    let s = if norm == 0.0 { 1.0 } else { 1.0 / norm };
                    for value in x.iter_mut() {
                        *value *= s;
                    }
                }
                return Some(x);
            }
        }
        None
    }))
}

// ============================================================================
// Brandes' shortest-path counting (edge betweenness and subsets)
// ============================================================================

/// The edges of the graph in NetworkX's `G.edges()` order (one per pair of nodes, oriented
/// from the node that comes first), and the edge behind each arc of the adjacency
struct Edges {
    pairs: Vec<(usize, usize)>,
    of_arc: Vec<usize>,
}

impl Edges {
    fn new(adjacency: &PathAdjacency) -> Self {
        let (indptr, indices, _) = adjacency.rows();
        let n = adjacency.len();
        let directed = adjacency.is_directed();
        let mut pairs = Vec::new();
        let mut of_arc = vec![0usize; indices.len()];
        let mut index: FxHashMap<(usize, usize), usize> = FxHashMap::default();
        for v in 0..n {
            for k in indptr[v]..indptr[v + 1] {
                let w = indices[k];
                if directed {
                    of_arc[k] = pairs.len();
                    pairs.push((v, w));
                } else if v <= w {
                    of_arc[k] = pairs.len();
                    index.insert((v, w), pairs.len());
                    pairs.push((v, w));
                } else {
                    // (w's row came first, and lists v)
                    of_arc[k] = index[&(w, v)];
                }
            }
        }
        Edges { pairs, of_arc }
    }
}

/// One source's shortest paths: the nodes in the order they were finished (`S`), each
/// node's predecessors as `(predecessor, arc)` (`P`) and path counts (`sigma`)
struct Paths {
    order: Vec<usize>,
    preds: Vec<Vec<(usize, usize)>>,
    sigma: Vec<f64>,
}

/// NetworkX's `_single_source_shortest_path_basic` (BFS)
fn bfs_paths(adjacency: &PathAdjacency, s: usize) -> Paths {
    let (indptr, indices, _) = adjacency.rows();
    let n = adjacency.len();
    let mut preds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut sigma = vec![0.0f64; n];
    let mut dist = vec![usize::MAX; n];
    let mut order = Vec::new();
    sigma[s] = 1.0;
    dist[s] = 0;
    let mut queue = VecDeque::from([s]);
    while let Some(v) = queue.pop_front() {
        order.push(v);
        let dv = dist[v];
        let sigma_v = sigma[v];
        for k in indptr[v]..indptr[v + 1] {
            let w = indices[k];
            if dist[w] == usize::MAX {
                queue.push_back(w);
                dist[w] = dv + 1;
            }
            if dist[w] == dv + 1 {
                sigma[w] += sigma_v;
                preds[w].push((v, k));
            }
        }
    }
    Paths {
        order,
        preds,
        sigma,
    }
}

#[derive(Clone, Copy)]
struct Entry {
    dist: f64,
    count: u64,
    pred: usize,
    node: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // reversed: BinaryHeap pops the largest
        match other.dist.partial_cmp(&self.dist) {
            Some(Ordering::Equal) | None => other.count.cmp(&self.count),
            Some(ordering) => ordering,
        }
    }
}

/// NetworkX's `_single_source_dijkstra_path_basic`
fn dijkstra_paths(adjacency: &PathAdjacency, s: usize) -> Paths {
    let (indptr, indices, weights) = adjacency.rows();
    let n = adjacency.len();
    let mut preds: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut sigma = vec![0.0f64; n];
    let mut done = vec![false; n];
    let mut seen: Vec<Option<f64>> = vec![None; n];
    let mut order = Vec::new();
    sigma[s] = 1.0;
    seen[s] = Some(0.0);
    let mut count = 0u64;
    let mut heap = BinaryHeap::from([Entry {
        dist: 0.0,
        count,
        pred: s,
        node: s,
    }]);
    while let Some(Entry {
        dist,
        pred,
        node: v,
        ..
    }) = heap.pop()
    {
        if done[v] {
            continue;
        }
        sigma[v] += sigma[pred];
        order.push(v);
        done[v] = true;
        for k in indptr[v]..indptr[v + 1] {
            let w = indices[k];
            let vw_dist = dist + weights[k];
            if !done[w] && seen[w].is_none_or(|d| vw_dist < d) {
                seen[w] = Some(vw_dist);
                count += 1;
                heap.push(Entry {
                    dist: vw_dist,
                    count,
                    pred: v,
                    node: w,
                });
                sigma[w] = 0.0;
                preds[w].clear();
                preds[w].push((v, k));
            } else if seen[w] == Some(vw_dist) {
                sigma[w] += sigma[v];
                preds[w].push((v, k));
            }
        }
    }
    Paths {
        order,
        preds,
        sigma,
    }
}

/// How a source's shortest paths add to betweenness
#[derive(Clone, Copy, PartialEq, Eq)]
enum Accumulate {
    /// `_accumulate_edges`
    Edges,
    /// `_accumulate_edges_subset`
    EdgesSubset,
    /// `_accumulate_subset` (nodes)
    NodesSubset,
}

/// One source's contributions, in the order NetworkX adds them: `(edge, value)` and
/// `(node, value)`
type Contributions = (Vec<(usize, f64)>, Vec<(usize, f64)>);

fn accumulate(
    paths: &Paths,
    s: usize,
    edges: &Edges,
    mode: Accumulate,
    targets: Option<&[bool]>,
) -> Contributions {
    let n = paths.sigma.len();
    let sigma = &paths.sigma;
    let mut delta = vec![0.0f64; n];
    let mut edge_terms = Vec::new();
    let mut node_terms = Vec::new();
    let is_target = |w: usize| targets.is_some_and(|t| t[w]);
    for &w in paths.order.iter().rev() {
        match mode {
            Accumulate::Edges => {
                let coeff = (1.0 + delta[w]) / sigma[w];
                for &(v, k) in &paths.preds[w] {
                    let c = sigma[v] * coeff;
                    edge_terms.push((edges.of_arc[k], c));
                    delta[v] += c;
                }
            }
            Accumulate::EdgesSubset => {
                let count = paths.preds[w].len() as f64;
                for &(v, k) in &paths.preds[w] {
                    let c = if is_target(w) {
                        sigma[v] / sigma[w] * (1.0 + delta[w])
                    } else {
                        delta[w] / count
                    };
                    edge_terms.push((edges.of_arc[k], c));
                    delta[v] += c;
                }
            }
            Accumulate::NodesSubset => {
                // (the source is never a target of its own paths)
                let coeff = if w != s && is_target(w) {
                    (delta[w] + 1.0) / sigma[w]
                } else {
                    delta[w] / sigma[w]
                };
                for &(v, _) in &paths.preds[w] {
                    delta[v] += sigma[v] * coeff;
                }
            }
        }
        if w != s {
            node_terms.push((w, delta[w]));
        }
    }
    (edge_terms, node_terms)
}

/// Brandes' accumulation from every source in `sources` (node positions, in order), with
/// NetworkX's exact arithmetic: BFS paths, or Dijkstra over the arc weights with `weighted`.
///
/// `mode` is "edges" (`edge_betweenness_centrality`), "edges_subset" or "nodes_subset" (the
/// `*_betweenness_centrality_subset` functions, with `targets` a mask per node). Returns
/// `(edges, edge_values, node_values)`: the edges as position pairs in NetworkX's
/// `G.edges()` order with their sums, and the node sums, all before NetworkX's rescaling.
#[pyfunction]
#[pyo3(signature = (adjacency, sources, weighted=false, mode="edges", targets=None))]
#[allow(clippy::type_complexity)]
pub fn nx_brandes(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    sources: Vec<usize>,
    weighted: bool,
    mode: &str,
    targets: Option<Vec<bool>>,
) -> PyResult<(Vec<(usize, usize)>, Vec<f64>, Vec<f64>)> {
    let n = adjacency.len();
    let mode = match mode {
        "edges" => Accumulate::Edges,
        "edges_subset" => Accumulate::EdgesSubset,
        "nodes_subset" => Accumulate::NodesSubset,
        _ => {
            return Err(PyValueError::new_err(format!(
                "mode must be 'edges', 'edges_subset' or 'nodes_subset', not {:?}",
                mode
            )))
        }
    };
    if sources.iter().any(|&s| s >= n) {
        return Err(PyValueError::new_err("source position out of range"));
    }
    if targets.as_ref().is_some_and(|t| t.len() != n) {
        return Err(PyValueError::new_err(
            "targets must have one entry per node",
        ));
    }
    Ok(py.allow_threads(|| {
        let edges = Edges::new(adjacency);
        let mut edge_values = vec![0.0f64; edges.pairs.len()];
        let mut node_values = vec![0.0f64; n];
        let run = |&s: &usize| {
            let paths = if weighted {
                dijkstra_paths(adjacency, s)
            } else {
                bfs_paths(adjacency, s)
            };
            accumulate(&paths, s, &edges, mode, targets.as_deref())
        };
        // Search in parallel batches, then add each source's terms in source order
        let arcs = adjacency.rows().1.len();
        let batch = (8_000_000 / (n + arcs).max(1)).clamp(1, 256);
        for chunk in sources.chunks(batch) {
            let terms: Vec<Contributions> = if chunk.len() > 1 && chunk.len() * (n + arcs) > 200_000
            {
                chunk.par_iter().map(run).collect()
            } else {
                chunk.iter().map(run).collect()
            };
            for (edge_terms, node_terms) in terms {
                for (e, c) in edge_terms {
                    edge_values[e] += c;
                }
                for (v, d) in node_terms {
                    node_values[v] += d;
                }
            }
        }
        (edges.pairs, edge_values, node_values)
    }))
}

// ============================================================================
// Load centrality
// ============================================================================

/// One source's load (NetworkX's `_node_betweenness(G, source, cutoff, False, weight)`):
/// `(node, value)` for each reached node, or None on a negative-weight contradiction
fn source_load(
    adjacency: &PathAdjacency,
    source: usize,
    rank: &[usize],
    weighted: bool,
    cutoff: Option<f64>,
) -> Option<Vec<(usize, f64)>> {
    let (nodes, lengths, preds) = adjacency.predecessors_from(source, cutoff, weighted)?;
    // `onodes.sort()` of (length, node) pairs, ties by node (key order), positive lengths
    let mut onodes: Vec<(f64, usize)> = nodes
        .iter()
        .zip(&lengths)
        .filter(|&(_, &l)| l > 0.0)
        .map(|(&v, &l)| (l, v))
        .collect();
    onodes.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| rank[a.1].cmp(&rank[b.1]))
    });
    let mut between = vec![0.0f64; adjacency.len()];
    for &v in &nodes {
        between[v] = 1.0;
    }
    while let Some((_, v)) = onodes.pop() {
        let pred = &preds[v];
        let num_paths = pred.len() as f64;
        for &x in pred {
            if x == source {
                break;
            }
            between[x] += between[v] / num_paths;
        }
    }
    Some(nodes.iter().map(|&v| (v, between[v] - 1.0)).collect())
}

/// Load centrality sums as `networkx.load_centrality` (`newman_betweenness_centrality`)
/// accumulates them before normalizing: per node, the sum over every source (in node order)
/// of its load from that source. BFS predecessors, or Dijkstra's with `weighted`, within
/// `cutoff` (for BFS, None means no cutoff, as NetworkX treats a falsy one). `rank` orders
/// the nodes as NetworkX compares their keys (to break ties among equal distances).
/// Raises ValueError on a negative-weight contradiction.
#[pyfunction]
#[pyo3(signature = (adjacency, rank, weighted=false, cutoff=None))]
pub fn nx_load_centrality(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    rank: Vec<usize>,
    weighted: bool,
    cutoff: Option<f64>,
) -> PyResult<Vec<f64>> {
    let n = adjacency.len();
    if rank.len() != n {
        return Err(PyValueError::new_err("rank must have one entry per node"));
    }
    let totals = py.allow_threads(|| {
        let mut totals = vec![0.0f64; n];
        let arcs = adjacency.rows().1.len();
        let batch = (8_000_000 / (n + arcs).max(1)).clamp(1, 256);
        let sources: Vec<usize> = (0..n).collect();
        let run = |&s: &usize| source_load(adjacency, s, &rank, weighted, cutoff);
        for chunk in sources.chunks(batch) {
            let loads: Vec<Option<Vec<(usize, f64)>>> =
                if chunk.len() > 1 && chunk.len() * (n + arcs) > 200_000 {
                    chunk.par_iter().map(run).collect()
                } else {
                    chunk.iter().map(run).collect()
                };
            for load in loads {
                for (v, value) in load? {
                    totals[v] += value;
                }
            }
        }
        Some(totals)
    });
    totals.ok_or_else(|| PyValueError::new_err(("Contradictory paths found:", "negative weights?")))
}

// ============================================================================
// Link prediction
// ============================================================================

/// The score a link-prediction function gives a pair of nodes
#[derive(Clone, Copy, PartialEq, Eq)]
enum LinkScore {
    Jaccard,
    AdamicAdar,
    ResourceAllocation,
    PreferentialAttachment,
    CommonNeighbors,
}

/// Scores of the node pairs `(us[i], vs[i])` (node positions) for NetworkX's link-prediction
/// functions on an undirected graph without parallel edges, where `G.degree(w)` counts a
/// self-loop twice and common neighbors exclude the pair itself (`nx.common_neighbors`).
///
/// `kind` is "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment" or
/// "common_neighbors" (their number, for `common_neighbor_centrality`). Returns
/// `(values, counts)`: the score as a float and an integer per pair, for Python to turn into
/// NetworkX's types: the size of the union of the two neighborhoods (jaccard), the number of
/// common neighbors (Adamic-Adar, resource allocation, common neighbors) or the product of
/// the degrees (preferential attachment).
///
/// Adamic-Adar and resource allocation add their terms over the common neighbors in the
/// first node's neighbor order; NetworkX adds them in the order of a Python set, so with
/// three or more terms the last bit can differ.
#[pyfunction]
pub fn nx_link_scores(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    us: Vec<usize>,
    vs: Vec<usize>,
    kind: &str,
) -> PyResult<(Vec<f64>, Vec<i64>)> {
    let kind = match kind {
        "jaccard" => LinkScore::Jaccard,
        "adamic_adar" => LinkScore::AdamicAdar,
        "resource_allocation" => LinkScore::ResourceAllocation,
        "preferential_attachment" => LinkScore::PreferentialAttachment,
        "common_neighbors" => LinkScore::CommonNeighbors,
        _ => return Err(PyValueError::new_err(format!("unknown score {:?}", kind))),
    };
    let n = adjacency.len();
    if us.len() != vs.len() || us.iter().chain(&vs).any(|&x| x >= n) {
        return Err(PyValueError::new_err(
            "us and vs must be node positions of the same length",
        ));
    }
    let (indptr, indices, _) = adjacency.rows();
    let neighbors = |v: usize| &indices[indptr[v]..indptr[v + 1]];
    let degree = |v: usize| {
        let row = neighbors(v);
        row.len() + usize::from(row.contains(&v))
    };
    let score = |marks: &mut Vec<usize>, i: usize| -> (f64, i64) {
        let (u, v) = (us[i], vs[i]);
        if kind == LinkScore::PreferentialAttachment {
            let product = (degree(u) * degree(v)) as i64;
            return (product as f64, product);
        }
        // Mark v's neighbors with this pair's stamp (i + 1)
        let stamp = i + 1;
        for &w in neighbors(v) {
            marks[w] = stamp;
        }
        let mut common = 0usize;
        let mut both = 0usize; // |N(u) & N(v)|, the pair's own nodes included
        let mut sum = 0.0f64;
        for &w in neighbors(u) {
            if marks[w] == stamp {
                both += 1;
                if w != u && w != v {
                    common += 1;
                    match kind {
                        LinkScore::AdamicAdar => sum += 1.0 / (degree(w) as f64).ln(),
                        LinkScore::ResourceAllocation => sum += 1.0 / degree(w) as f64,
                        _ => {}
                    }
                }
            }
        }
        match kind {
            LinkScore::Jaccard => {
                let union = neighbors(u).len() + neighbors(v).len() - both;
                let value = if union == 0 {
                    0.0
                } else {
                    common as f64 / union as f64
                };
                (value, union as i64)
            }
            LinkScore::CommonNeighbors => (common as f64, common as i64),
            _ => (sum, common as i64),
        }
    };
    let pairs = us.len();
    let results: Vec<(f64, i64)> = py.allow_threads(|| {
        if pairs > 1 && pairs * 16 > 200_000 {
            (0..pairs)
                .into_par_iter()
                .map_init(|| vec![0usize; n], |marks, i| score(marks, i))
                .collect()
        } else {
            let mut marks = vec![0usize; n];
            (0..pairs).map(|i| score(&mut marks, i)).collect()
        }
    });
    Ok(results.into_iter().unzip())
}

/// Common-neighbor counts by community for NetworkX's Soundarajan-Hopcroft scores and
/// `within_inter_cluster`, for the node pairs `(us[i], vs[i])`. `communities` holds a code
/// per node (equal codes for equal communities, negative for a node without one).
///
/// Returns `(sums, common, same)` per pair: the number of common neighbors; when the two
/// nodes share a community, the number of common neighbors in it and the sum of `1 / degree`
/// over those (in the first node's neighbor order); and `same` = -1 or -2 if the first or
/// second node has no community, -3 if a common neighbor has none (when they share one).
#[pyfunction]
pub fn nx_community_link_scores(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    us: Vec<usize>,
    vs: Vec<usize>,
    communities: Vec<i64>,
) -> PyResult<(Vec<f64>, Vec<i64>, Vec<i64>)> {
    let n = adjacency.len();
    if us.len() != vs.len() || us.iter().chain(&vs).any(|&x| x >= n) || communities.len() != n {
        return Err(PyValueError::new_err(
            "us and vs must be node positions of the same length, with a community per node",
        ));
    }
    let (indptr, indices, _) = adjacency.rows();
    let neighbors = |v: usize| &indices[indptr[v]..indptr[v + 1]];
    let degree = |v: usize| {
        let row = neighbors(v);
        row.len() + usize::from(row.contains(&v))
    };
    let score = |marks: &mut Vec<usize>, i: usize| -> (f64, i64, i64) {
        let (u, v) = (us[i], vs[i]);
        let (cu, cv) = (communities[u], communities[v]);
        if cu < 0 {
            return (0.0, 0, -1);
        }
        if cv < 0 {
            return (0.0, 0, -2);
        }
        let stamp = i + 1;
        for &w in neighbors(v) {
            marks[w] = stamp;
        }
        let shared = cu == cv;
        let (mut common, mut same, mut sum, mut missing) = (0i64, 0i64, 0.0f64, false);
        for &w in neighbors(u) {
            if marks[w] == stamp && w != u && w != v {
                common += 1;
                if shared {
                    let cw = communities[w];
                    if cw < 0 {
                        missing = true;
                    } else if cw == cu {
                        same += 1;
                        sum += 1.0 / degree(w) as f64;
                    }
                }
            }
        }
        (sum, common, if missing { -3 } else { same })
    };
    let pairs = us.len();
    let results: Vec<(f64, i64, i64)> = py.allow_threads(|| {
        if pairs > 1 && pairs * 16 > 200_000 {
            (0..pairs)
                .into_par_iter()
                .map_init(|| vec![0usize; n], |marks, i| score(marks, i))
                .collect()
        } else {
            let mut marks = vec![0usize; n];
            (0..pairs).map(|i| score(&mut marks, i)).collect()
        }
    });
    let mut sums = Vec::with_capacity(pairs);
    let mut common = Vec::with_capacity(pairs);
    let mut same = Vec::with_capacity(pairs);
    for (s, c, k) in results {
        sums.push(s);
        common.push(c);
        same.push(k);
    }
    Ok((sums, common, same))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_matches_the_c_library() {
        assert_eq!(frexp_exponent(1.0), 1);
        assert_eq!(frexp_exponent(0.5), 0);
        assert_eq!(frexp_exponent(0.75), 0);
        assert_eq!(frexp_exponent(3.0), 2);
        assert_eq!(frexp_exponent(f64::MIN_POSITIVE), -1021);
        assert_eq!(frexp_exponent(f64::MIN_POSITIVE / 4.0), -1023);
    }

    #[test]
    fn hypot_is_exact_on_easy_cases() {
        assert_eq!(python_hypot([3.0, 4.0]), 5.0);
        assert_eq!(python_hypot([]), 0.0);
        assert_eq!(python_hypot([-2.5]), 2.5);
        assert!(python_hypot([1.0, f64::NAN]).is_nan());
        assert_eq!(python_hypot([f64::NAN, f64::INFINITY]), f64::INFINITY);
    }
}
