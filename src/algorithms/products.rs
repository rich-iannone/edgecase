//! Graph products (Cartesian, tensor, strong, lexicographic, rooted, corona, modular), the
//! graph power and random k-lifts.
//!
//! Product nodes are the pairs `(g, h)` of a node of the first graph and a node of the second,
//! labeled `"(g_label, h_label)"` and numbered row by row (the pair of the i-th node of the first
//! graph and the j-th node of the second, both in ID order, gets ID `i * m + j + 1`). Each node
//! attribute becomes a tuple `(value in first graph, value in second graph)`, as in NetworkX.
//! An edge that comes from a single factor edge keeps that edge's weight and attributes; an
//! edge that comes from a pair of edges (tensor-type) gets the product of their weights.

use crate::graph::{Edge, Graph, Node};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

struct Factor<'a> {
    ids: Vec<u64>,
    idx: FxHashMap<u64, usize>,
    nodes: Vec<&'a Node>,
    edges: Vec<(usize, usize, &'a Edge)>,
}

impl<'a> Factor<'a> {
    fn is_simple(&self, directed: bool) -> bool {
        let mut seen = FxHashSet::default();
        self.edges.iter().all(|&(a, b, _)| seen.insert(if directed { (a, b) } else { (a.min(b), a.max(b)) }))
    }
    fn new(graph: &'a Graph) -> Self {
        let mut ids = graph.get_node_ids();
        ids.sort_unstable();
        let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let all = graph.get_nodes();
        let nodes = ids.iter().map(|id| &all[id]).collect();
        let mut edges: Vec<(usize, usize, &Edge)> =
            graph.get_edges_internal().iter().map(|e| (idx[&e.source], idx[&e.target], e)).collect();
        edges.sort_unstable_by_key(|x| x.2.id);
        Factor { ids, idx, nodes, edges }
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
    /// Simple adjacency sets (self-loops kept, direction ignored when `undirected`).
    fn adjacency(&self, undirected: bool) -> Vec<FxHashSet<usize>> {
        let mut adj = vec![FxHashSet::default(); self.len()];
        for &(a, b, _) in &self.edges {
            adj[a].insert(b);
            if undirected {
                adj[b].insert(a);
            }
        }
        adj
    }
}

fn copy_attrs(py: Python, attrs: &Option<Py<PyDict>>) -> PyResult<Option<Py<PyDict>>> {
    Ok(match attrs {
        Some(d) => Some(d.bind(py).copy()?.unbind()),
        None => None,
    })
}

/// NetworkX's `_dict_product`: every key of either dict maps to `(value_1, value_2)`.
fn dict_product(py: Python, a: &Option<Py<PyDict>>, b: &Option<Py<PyDict>>) -> PyResult<Option<Py<PyDict>>> {
    if a.is_none() && b.is_none() {
        return Ok(None);
    }
    let out = PyDict::new(py);
    let none = py.None();
    let da = a.as_ref().map(|d| d.bind(py).clone());
    let db = b.as_ref().map(|d| d.bind(py).clone());
    for side in [&da, &db] {
        if let Some(d) = side {
            for key in d.keys() {
                if out.contains(&key)? {
                    continue;
                }
                let va = match &da {
                    Some(x) => x.get_item(&key)?.map(|v| v.unbind()).unwrap_or_else(|| none.clone_ref(py)),
                    None => none.clone_ref(py),
                };
                let vb = match &db {
                    Some(x) => x.get_item(&key)?.map(|v| v.unbind()).unwrap_or_else(|| none.clone_ref(py)),
                    None => none.clone_ref(py),
                };
                out.set_item(key, PyTuple::new(py, [va, vb])?)?;
            }
        }
    }
    Ok(if out.is_empty() { None } else { Some(out.unbind()) })
}

struct Builder {
    result: Graph,
    weighted: bool,
    directed: bool,
    /// when both factors are simple, the product is too: repeated pairs are skipped
    seen: Option<FxHashSet<(u64, u64)>>,
}

impl Builder {
    fn new(directed: bool, weighted: bool, simple: bool) -> Self {
        let seen = if simple { Some(FxHashSet::default()) } else { None };
        Builder { result: Graph::new_rust(directed, weighted), weighted, directed, seen }
    }
    fn fresh(&mut self, a: u64, b: u64) -> bool {
        let key = if self.directed { (a, b) } else { (a.min(b), a.max(b)) };
        match &mut self.seen {
            Some(seen) => seen.insert(key),
            None => true,
        }
    }
    fn node(&mut self, id: u64, label: Option<String>, attributes: Option<Py<PyDict>>) -> PyResult<()> {
        self.result.add_node_rust(Node { id, label, attributes })?;
        Ok(())
    }
    /// An edge copied from a single factor edge.
    fn copy_edge(&mut self, py: Python, a: u64, b: u64, from: &Edge) -> PyResult<()> {
        if !self.fresh(a, b) {
            return Ok(());
        }
        let attributes = copy_attrs(py, &from.attributes)?;
        let weight = if self.weighted { from.weight.or(Some(1.0)) } else { from.weight };
        self.result.add_edge_rust(Edge { id: 0, source: a, target: b, weight, attributes })?;
        Ok(())
    }
    /// An edge from a pair of factor edges: product weight, attribute tuples.
    fn pair_edge(&mut self, py: Python, a: u64, b: u64, e: &Edge, f: &Edge) -> PyResult<()> {
        if !self.fresh(a, b) {
            return Ok(());
        }
        let weight = if self.weighted { Some(e.weight.unwrap_or(1.0) * f.weight.unwrap_or(1.0)) } else { None };
        let attributes = dict_product(py, &e.attributes, &f.attributes)?;
        self.result.add_edge_rust(Edge { id: 0, source: a, target: b, weight, attributes })?;
        Ok(())
    }
    /// A structural edge with no factor data.
    fn plain_edge(&mut self, a: u64, b: u64) -> PyResult<()> {
        if !self.fresh(a, b) {
            return Ok(());
        }
        let weight = if self.weighted { Some(1.0) } else { None };
        self.result.add_edge_rust(Edge { id: 0, source: a, target: b, weight, attributes: None })?;
        Ok(())
    }
}

/// A graph product of `graph` and `other`: `kind` is `"cartesian"`, `"tensor"`, `"strong"`,
/// `"lexicographic"`, `"rooted"` (with `root` a node ID of `other`), `"corona"` or `"modular"`.
#[pyfunction]
#[pyo3(signature = (graph, other, kind, root=None))]
pub fn graph_product(py: Python, graph: &Graph, other: &Graph, kind: &str, root: Option<u64>) -> PyResult<Graph> {
    if graph.is_directed() != other.is_directed() {
        return value_err("both graphs must be directed, or both undirected");
    }
    let directed = graph.is_directed();
    if directed && matches!(kind, "corona" | "modular") {
        return value_err(format!("the {kind} product requires undirected graphs"));
    }
    let g = Factor::new(graph);
    let h = Factor::new(other);
    let (n, m) = (g.len(), h.len());
    let simple = g.is_simple(directed) && h.is_simple(directed);
    let mut b = Builder::new(directed, graph.is_weighted() || other.is_weighted(), simple);
    let pid = |i: usize, j: usize| (i * m + j + 1) as u64;
    let name = |f: &Factor, i: usize| f.nodes[i].label.clone().unwrap_or_else(|| f.ids[i].to_string());
    let pair_label = |i: usize, j: usize| Some(format!("({}, {})", name(&g, i), name(&h, j)));

    if kind == "corona" {
        // the first graph keeps its nodes (IDs 1..n), then one copy of `other` per node
        for i in 0..n {
            b.node((i + 1) as u64, g.nodes[i].label.clone(), copy_attrs(py, &g.nodes[i].attributes)?)?;
        }
        let cid = |i: usize, j: usize| (n + i * m + j + 1) as u64;
        for i in 0..n {
            for j in 0..m {
                b.node(cid(i, j), pair_label(i, j), copy_attrs(py, &h.nodes[j].attributes)?)?;
            }
        }
        for &(x, y, e) in &g.edges {
            b.copy_edge(py, (x + 1) as u64, (y + 1) as u64, e)?;
        }
        for i in 0..n {
            for &(x, y, e) in &h.edges {
                b.copy_edge(py, cid(i, x), cid(i, y), e)?;
            }
            for j in 0..m {
                b.plain_edge((i + 1) as u64, cid(i, j))?;
            }
        }
        return Ok(b.result);
    }

    for i in 0..n {
        for j in 0..m {
            let attrs = dict_product(py, &g.nodes[i].attributes, &h.nodes[j].attributes)?;
            b.node(pid(i, j), pair_label(i, j), attrs)?;
        }
    }
    // (u, x)-(v, x) for every edge u-v of the first graph and node x of the second
    let edges_cross_nodes = |b: &mut Builder| -> PyResult<()> {
        for &(u, v, e) in &g.edges {
            for x in 0..m {
                b.copy_edge(py, pid(u, x), pid(v, x), e)?;
            }
        }
        Ok(())
    };
    // (x, u)-(x, v) for every node x of the first graph and edge u-v of the second
    let nodes_cross_edges = |b: &mut Builder| -> PyResult<()> {
        for x in 0..n {
            for &(u, v, e) in &h.edges {
                b.copy_edge(py, pid(x, u), pid(x, v), e)?;
            }
        }
        Ok(())
    };
    // (u, x)-(v, y) for every pair of edges, plus (v, x)-(u, y) when undirected
    let edges_cross_edges = |b: &mut Builder| -> PyResult<()> {
        for &(u, v, e) in &g.edges {
            for &(x, y, f) in &h.edges {
                b.pair_edge(py, pid(u, x), pid(v, y), e, f)?;
                if !directed && u != v && x != y {
                    b.pair_edge(py, pid(v, x), pid(u, y), e, f)?;
                }
            }
        }
        Ok(())
    };
    match kind {
        "cartesian" => {
            edges_cross_nodes(&mut b)?;
            nodes_cross_edges(&mut b)?;
        }
        "tensor" => edges_cross_edges(&mut b)?,
        "strong" => {
            nodes_cross_edges(&mut b)?;
            edges_cross_nodes(&mut b)?;
            edges_cross_edges(&mut b)?;
        }
        "lexicographic" => {
            for &(u, v, e) in &g.edges {
                for x in 0..m {
                    for y in 0..m {
                        b.copy_edge(py, pid(u, x), pid(v, y), e)?;
                    }
                }
            }
            nodes_cross_edges(&mut b)?;
        }
        "rooted" => {
            let Some(r) = root.and_then(|id| h.idx.get(&id).copied()) else {
                return value_err("root must be a node of the second graph");
            };
            for &(u, v, e) in &g.edges {
                b.copy_edge(py, pid(u, r), pid(v, r), e)?;
            }
            nodes_cross_edges(&mut b)?;
        }
        "modular" => {
            // simple graphs: adjacent pairs in both, or non-adjacent pairs in both
            let ga = g.adjacency(true);
            let ha = h.adjacency(true);
            let mut seen: FxHashSet<(u64, u64)> = FxHashSet::default();
            let mut add = |b: &mut Builder, p: u64, q: u64| -> PyResult<()> {
                if seen.insert((p.min(q), p.max(q))) {
                    b.plain_edge(p, q)?;
                }
                Ok(())
            };
            for u in 0..n {
                for v in u..n {
                    let g_edge = ga[u].contains(&v);
                    for x in 0..m {
                        for y in x..m {
                            let h_edge = ha[x].contains(&y);
                            let both = g_edge && h_edge;
                            let neither = u != v && x != y && !g_edge && !h_edge;
                            if both || neither {
                                add(&mut b, pid(u, x), pid(v, y))?;
                                add(&mut b, pid(v, x), pid(u, y))?;
                            }
                        }
                    }
                }
            }
        }
        _ => return value_err(format!("unknown product {kind:?}")),
    }
    Ok(b.result)
}

/// The k-th power of an undirected graph: the same nodes (IDs, labels, attributes kept), joined
/// when their distance is at most `k`. Self-loops and parallel edges are not created.
#[pyfunction]
pub fn graph_power(py: Python, graph: &Graph, k: usize) -> PyResult<Graph> {
    if graph.is_directed() {
        return value_err("power() requires an undirected graph");
    }
    if k == 0 {
        return value_err("k must be a positive integer");
    }
    let g = Factor::new(graph);
    let n = g.len();
    let adj = g.adjacency(true);
    let mut result = Graph::new_rust(false, false);
    for i in 0..n {
        result.add_node_rust(Node { id: g.ids[i], label: g.nodes[i].label.clone(), attributes: copy_attrs(py, &g.nodes[i].attributes)? })?;
    }
    for s in 0..n {
        let mut dist = vec![usize::MAX; n];
        dist[s] = 0;
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            if dist[u] == k {
                continue;
            }
            for &w in &adj[u] {
                if dist[w] == usize::MAX {
                    dist[w] = dist[u] + 1;
                    q.push_back(w);
                }
            }
        }
        for t in s + 1..n {
            if dist[t] != usize::MAX {
                result.add_edge_rust(Edge { id: 0, source: g.ids[s], target: g.ids[t], weight: None, attributes: None })?;
            }
        }
    }
    Ok(result)
}

/// A random k-lift: `k` copies `(v, 0), ..., (v, k - 1)` of every node (the copy `i` of the j-th
/// node in ID order gets ID `j * k + i + 1`, the label `"(v_label, i)"` and `v`'s attributes),
/// and for every edge `u - v`, in edge ID order, a fresh uniform permutation `s` of `0..k` joining
/// `(u, i)` to `(v, s(i))` with the edge's weight and attributes. Every base edge gives exactly
/// `k` edges, so a self-loop lifts to the cycles of its permutation (as in NetworkX's multigraph
/// lifts).
#[pyfunction]
#[pyo3(signature = (graph, k, seed=None))]
pub fn random_k_lift(py: Python, graph: &Graph, k: usize, seed: Option<u64>) -> PyResult<Graph> {
    if k == 0 {
        return value_err("k must be a positive integer");
    }
    let g = Factor::new(graph);
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let mut b = Builder::new(graph.is_directed(), graph.is_weighted(), false);
    let lid = |j: usize, i: usize| (j * k + i + 1) as u64;
    for j in 0..g.len() {
        let name = g.nodes[j].label.clone().unwrap_or_else(|| g.ids[j].to_string());
        for i in 0..k {
            b.node(lid(j, i), Some(format!("({name}, {i})")), copy_attrs(py, &g.nodes[j].attributes)?)?;
        }
    }
    let mut perm: Vec<usize> = (0..k).collect();
    for &(u, v, e) in &g.edges {
        perm.shuffle(&mut rng);
        for (i, &p) in perm.iter().enumerate() {
            b.copy_edge(py, lid(u, i), lid(v, p), e)?;
        }
    }
    Ok(b.result)
}
