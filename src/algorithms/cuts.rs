//! Cuts and disjoint paths: s-t minimum cut partitions, Stoer-Wagner global minimum cut,
//! Gomory-Hu cut trees, edge- and node-disjoint paths, and the directed triadic census.
//!
//! Capacities follow `maximum_flow()`: the edge attribute named by `capacity`, else a `capacity`
//! attribute, else the edge weight on a weighted graph, else 1.0. Self-loops are ignored.

use crate::algorithms::flow::{edmonds_karp, get_edge_capacity};
use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BinaryHeap, VecDeque};

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn check_pair(graph: &Graph, s: u64, t: u64) -> PyResult<()> {
    for (name, v) in [("Source", s), ("Target", t)] {
        if !graph.has_node_id(v) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} node {} does not exist",
                name, v
            )));
        }
    }
    if s == t {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Source and target must be different nodes",
        ));
    }
    Ok(())
}

fn require_undirected(graph: &Graph, what: &str) -> PyResult<()> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} requires an undirected graph",
            what
        )));
    }
    Ok(())
}

/// Minimum s-t cut: `(value, source_side, sink_side)`, both sides sorted.
///
/// The source side is everything still reachable from `source` in the residual network of a
/// maximum flow, so it is the minimum cut closest to the source.
#[pyfunction]
#[pyo3(signature = (graph, source, sink, capacity=None))]
pub fn minimum_cut(
    graph: &Graph,
    source: u64,
    sink: u64,
    capacity: Option<&str>,
) -> PyResult<(f64, Vec<u64>, Vec<u64>)> {
    check_pair(graph, source, sink)?;
    let (value, _, side) = edmonds_karp(graph, source, sink, capacity)?;
    let set: FxHashSet<u64> = side.iter().copied().collect();
    let (ids, _) = sorted_ids(graph);
    let rest = ids.into_iter().filter(|id| !set.contains(id)).collect();
    Ok((value, side, rest))
}

/// Merged undirected capacity map: `w[i][j]` = total capacity between i and j (no self-loops).
fn capacity_maps(
    graph: &Graph,
    idx: &FxHashMap<u64, usize>,
    capacity: Option<&str>,
) -> PyResult<Vec<FxHashMap<usize, f64>>> {
    let mut w: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); idx.len()];
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let c = get_edge_capacity(e, capacity, graph.is_weighted())?;
        let (a, b) = (idx[&e.source], idx[&e.target]);
        *w[a].entry(b).or_insert(0.0) += c;
        *w[b].entry(a).or_insert(0.0) += c;
    }
    Ok(w)
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct OrdF(f64);
impl Eq for OrdF {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrdF {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&o.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Global minimum cut of an undirected graph (Stoer & Wagner 1997).
///
/// Returns `(value, side_a, side_b)`, both sorted, with the side holding the smallest node ID
/// first. A disconnected graph has value 0. Raises for directed graphs or fewer than 2 nodes.
#[pyfunction]
#[pyo3(signature = (graph, capacity=None))]
pub fn stoer_wagner(graph: &Graph, capacity: Option<&str>) -> PyResult<(f64, Vec<u64>, Vec<u64>)> {
    require_undirected(graph, "stoer_wagner()")?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "stoer_wagner() needs at least two nodes",
        ));
    }
    let mut w = capacity_maps(graph, &idx, capacity)?;
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut alive = vec![true; n];
    let mut best = f64::INFINITY;
    let mut best_side: Vec<usize> = Vec::new();

    for _ in 0..n - 1 {
        // Maximum adjacency ordering, from the first alive super-node
        let start = (0..n).find(|&i| alive[i]).unwrap();
        let mut key = vec![0.0f64; n];
        let mut added = vec![false; n];
        let mut heap: BinaryHeap<(OrdF, std::cmp::Reverse<usize>)> = BinaryHeap::new();
        for v in 0..n {
            if alive[v] {
                heap.push((OrdF(0.0), std::cmp::Reverse(v)));
            }
        }
        key[start] = f64::INFINITY;
        heap.push((OrdF(f64::INFINITY), std::cmp::Reverse(start)));
        let mut prev = usize::MAX;
        let mut last = usize::MAX;
        while let Some((OrdF(k), std::cmp::Reverse(v))) = heap.pop() {
            if added[v] || k < key[v] {
                continue;
            }
            added[v] = true;
            prev = last;
            last = v;
            for (&x, &c) in &w[v] {
                if alive[x] && !added[x] {
                    key[x] += c;
                    heap.push((OrdF(key[x]), std::cmp::Reverse(x)));
                }
            }
        }
        let cut = if last == start { 0.0 } else { key[last] };
        if cut < best {
            best = cut;
            best_side = members[last].clone();
        }
        // Merge `last` into `prev`
        let (s, t) = (prev, last);
        let moved = std::mem::take(&mut w[t]);
        for (x, c) in moved {
            if x == s {
                continue;
            }
            w[x].remove(&t);
            *w[x].entry(s).or_insert(0.0) += c;
            *w[s].entry(x).or_insert(0.0) += c;
        }
        w[s].remove(&t);
        let mv = std::mem::take(&mut members[t]);
        members[s].extend(mv);
        alive[t] = false;
    }
    let side: FxHashSet<usize> = best_side.into_iter().collect();
    let mut a: Vec<u64> = (0..n)
        .filter(|i| side.contains(i))
        .map(|i| ids[i])
        .collect();
    let mut b: Vec<u64> = (0..n)
        .filter(|i| !side.contains(i))
        .map(|i| ids[i])
        .collect();
    if b.first() < a.first() || a.is_empty() {
        std::mem::swap(&mut a, &mut b);
    }
    Ok((best, a, b))
}

/// Gomory-Hu cut tree (Gusfield's algorithm): `(node, parent, cut_value)` for every node but
/// the root (the smallest node ID). The minimum cut between any two nodes equals the smallest
/// cut value on the tree path between them. Undirected graphs only.
#[pyfunction]
#[pyo3(signature = (graph, capacity=None))]
pub fn gomory_hu(graph: &Graph, capacity: Option<&str>) -> PyResult<Vec<(u64, u64, f64)>> {
    require_undirected(graph, "gomory_hu_tree()")?;
    let (ids, _) = sorted_ids(graph);
    let n = ids.len();
    let mut parent = vec![0usize; n];
    let mut fl = vec![0.0f64; n];
    for s in 1..n {
        let t = parent[s];
        let (value, _, side) = edmonds_karp(graph, ids[s], ids[t], capacity)?;
        let x: FxHashSet<u64> = side.into_iter().collect();
        fl[s] = value;
        for i in 0..n {
            if i != s && x.contains(&ids[i]) && parent[i] == t {
                parent[i] = s;
            }
        }
        if x.contains(&ids[parent[t]]) && t != 0 {
            parent[s] = parent[t];
            parent[t] = s;
            fl[s] = fl[t];
            fl[t] = value;
        }
    }
    Ok((1..n).map(|i| (ids[i], ids[parent[i]], fl[i])).collect())
}

/// An undirected weighted tree on the given nodes (IDs, labels and attributes copied), with
/// the given weighted edges.
#[pyfunction]
pub fn weighted_tree_graph(
    py: Python,
    graph: &Graph,
    edges: Vec<(u64, u64, f64)>,
) -> PyResult<Graph> {
    use crate::graph::{Edge, Node};
    let mut result = Graph::new_rust(false, true);
    let (ids, _) = sorted_ids(graph);
    let nodes = graph.get_nodes();
    for id in ids {
        let node = &nodes[&id];
        let attributes = match &node.attributes {
            Some(d) => Some(d.bind(py).copy()?.unbind()),
            None => None,
        };
        result.add_node_rust(Node {
            id,
            label: node.label.clone(),
            attributes,
        })?;
    }
    for (a, b, w) in edges {
        result.add_edge_rust(Edge {
            id: 0,
            source: a,
            target: b,
            weight: Some(w),
            attributes: None,
        })?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------------------------
// Disjoint paths (unit-capacity max flow + path decomposition)
// ---------------------------------------------------------------------------------------------

struct Arc {
    to: usize,
    res: f64,
    rev: usize,
    /// (from, to) in the original graph this arc stands for (None for node-split arcs)
    step: Option<(usize, usize)>,
}

fn add_arc(
    arcs: &mut Vec<Arc>,
    out: &mut [Vec<usize>],
    u: usize,
    v: usize,
    cap: f64,
    back: f64,
    step: Option<(usize, usize)>,
) {
    let f = arcs.len();
    arcs.push(Arc {
        to: v,
        res: cap,
        rev: f + 1,
        step,
    });
    arcs.push(Arc {
        to: u,
        res: back,
        rev: f,
        step: step.map(|(a, b)| (b, a)),
    });
    out[u].push(f);
    out[v].push(f + 1);
}

/// Edmonds-Karp on unit arcs until `limit` augmentations or no path; returns the flow value.
fn unit_max_flow(arcs: &mut [Arc], out: &[Vec<usize>], s: usize, t: usize, limit: usize) -> usize {
    let mut flow = 0;
    while flow < limit {
        let mut parent: Vec<Option<usize>> = vec![None; out.len()];
        let mut seen = vec![false; out.len()];
        seen[s] = true;
        let mut queue = VecDeque::from([s]);
        while let Some(v) = queue.pop_front() {
            if v == t {
                break;
            }
            for &a in &out[v] {
                if arcs[a].res > 0.5 && !seen[arcs[a].to] {
                    seen[arcs[a].to] = true;
                    parent[arcs[a].to] = Some(a);
                    queue.push_back(arcs[a].to);
                }
            }
        }
        if !seen[t] {
            break;
        }
        let mut v = t;
        while v != s {
            let a = parent[v].unwrap();
            arcs[a].res -= 1.0;
            let r = arcs[a].rev;
            arcs[r].res += 1.0;
            v = arcs[r].to;
        }
        flow += 1;
    }
    flow
}

/// Split a unit flow into `count` paths from s to t, given the net flow on each original step.
fn decompose(
    units: Vec<(usize, usize)>,
    n: usize,
    s: usize,
    t: usize,
    count: usize,
    ids: &[u64],
) -> Vec<Vec<u64>> {
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n]; // (to, unit index)
    for (k, &(a, b)) in units.iter().enumerate() {
        adj[a].push((b, k));
    }
    let mut used = vec![false; units.len()];
    let mut paths = Vec::new();
    for _ in 0..count {
        // BFS over unused flow units
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut seen = vec![false; n];
        seen[s] = true;
        let mut queue = VecDeque::from([s]);
        while let Some(v) = queue.pop_front() {
            if v == t {
                break;
            }
            for &(w, k) in &adj[v] {
                if !used[k] && !seen[w] {
                    seen[w] = true;
                    parent[w] = Some((v, k));
                    queue.push_back(w);
                }
            }
        }
        if !seen[t] {
            break;
        }
        let mut path = vec![ids[t]];
        let mut v = t;
        while let Some((p, k)) = parent[v] {
            used[k] = true;
            path.push(ids[p]);
            v = p;
        }
        path.reverse();
        paths.push(path);
    }
    paths.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    paths
}

/// A maximum set of edge-disjoint paths from `source` to `target` (node-ID lists), at most
/// `max_paths` of them, shortest first. Capacities are ignored: each edge is used at most once.
#[pyfunction]
#[pyo3(signature = (graph, source, target, max_paths=None))]
pub fn edge_disjoint_paths(
    graph: &Graph,
    source: u64,
    target: u64,
    max_paths: Option<usize>,
) -> PyResult<Vec<Vec<u64>>> {
    check_pair(graph, source, target)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let directed = graph.is_directed();
    let mut arcs: Vec<Arc> = Vec::new();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fwd: Vec<usize> = Vec::new();
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let (a, b) = (idx[&e.source], idx[&e.target]);
        fwd.push(arcs.len());
        add_arc(
            &mut arcs,
            &mut out,
            a,
            b,
            1.0,
            if directed { 0.0 } else { 1.0 },
            Some((a, b)),
        );
    }
    let (s, t) = (idx[&source], idx[&target]);
    let flow = unit_max_flow(&mut arcs, &out, s, t, max_paths.unwrap_or(usize::MAX));
    let mut units = Vec::new();
    for f in fwd {
        let (a, b) = arcs[f].step.unwrap();
        let back = arcs[arcs[f].rev].res;
        // net flow a -> b: directed 1 - res; undirected (res_back - res_fwd) / 2
        let net = if directed {
            1.0 - arcs[f].res
        } else {
            (back - arcs[f].res) / 2.0
        };
        if net > 0.5 {
            units.push((a, b));
        } else if net < -0.5 {
            units.push((b, a));
        }
    }
    Ok(decompose(units, n, s, t, flow, &ids))
}

/// A maximum set of internally node-disjoint paths from `source` to `target`, at most
/// `max_paths` of them, shortest first. Each node other than the endpoints is used at most once.
#[pyfunction]
#[pyo3(signature = (graph, source, target, max_paths=None))]
pub fn node_disjoint_paths(
    graph: &Graph,
    source: u64,
    target: u64,
    max_paths: Option<usize>,
) -> PyResult<Vec<Vec<u64>>> {
    check_pair(graph, source, target)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let (s, t) = (idx[&source], idx[&target]);
    // node v -> in = 2v, out = 2v + 1; inner nodes carry one unit
    let mut arcs: Vec<Arc> = Vec::new();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); 2 * n];
    for v in 0..n {
        let cap = if v == s || v == t { n as f64 } else { 1.0 };
        add_arc(&mut arcs, &mut out, 2 * v, 2 * v + 1, cap, 0.0, None);
    }
    let mut steps: Vec<usize> = Vec::new();
    let directed = graph.is_directed();
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let (a, b) = (idx[&e.source], idx[&e.target]);
        steps.push(arcs.len());
        add_arc(
            &mut arcs,
            &mut out,
            2 * a + 1,
            2 * b,
            1.0,
            0.0,
            Some((a, b)),
        );
        if !directed {
            steps.push(arcs.len());
            add_arc(
                &mut arcs,
                &mut out,
                2 * b + 1,
                2 * a,
                1.0,
                0.0,
                Some((b, a)),
            );
        }
    }
    let flow = unit_max_flow(
        &mut arcs,
        &out,
        2 * s + 1,
        2 * t,
        max_paths.unwrap_or(usize::MAX),
    );
    let units: Vec<(usize, usize)> = steps
        .into_iter()
        .filter(|&f| arcs[f].res < 0.5)
        .map(|f| arcs[f].step.unwrap())
        .collect();
    Ok(decompose(units, n, s, t, flow, &ids))
}

// ---------------------------------------------------------------------------------------------
// Triadic census
// ---------------------------------------------------------------------------------------------

/// The 16 triad types in MAN notation (mutual, asymmetric, null dyad counts plus a letter).
pub const TRIAD_NAMES: [&str; 16] = [
    "003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D",
    "120U", "120C", "210", "300",
];

/// Batagelj-Mrvar triad code (0..63) to 1-based index into `TRIAD_NAMES` (as in NetworkX).
const TRICODES: [usize; 64] = [
    1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8, 5, 9, 9, 13, 6, 10, 9, 14, 7, 14,
    12, 15, 2, 5, 6, 7, 6, 9, 10, 14, 4, 9, 9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14, 15, 8,
    14, 13, 15, 11, 15, 15, 16,
];

/// Count every triad (unordered node triple) of a directed graph by type, in O(m * max degree)
/// (Batagelj & Mrvar 2001). Parallel edges count once; self-loops are ignored.
#[pyfunction]
pub fn triadic_census(graph: &Graph) -> PyResult<Vec<(String, u64)>> {
    if !graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "triadic_census() requires a directed graph",
        ));
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut succ: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    let mut pred: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let (a, b) = (idx[&e.source], idx[&e.target]);
        succ[a].insert(b);
        pred[b].insert(a);
    }
    let nbrs: Vec<FxHashSet<usize>> = (0..n)
        .map(|v| succ[v].union(&pred[v]).copied().collect())
        .collect();
    let code = |v: usize, u: usize, w: usize| -> usize {
        let mut c = 0;
        if succ[v].contains(&u) {
            c |= 1;
        }
        if succ[u].contains(&v) {
            c |= 2;
        }
        if succ[v].contains(&w) {
            c |= 4;
        }
        if succ[w].contains(&v) {
            c |= 8;
        }
        if succ[u].contains(&w) {
            c |= 16;
        }
        if succ[w].contains(&u) {
            c |= 32;
        }
        c
    };
    let mut census = [0u64; 16];
    for v in 0..n {
        for &u in &nbrs[v] {
            if u <= v {
                continue;
            }
            let mut s: FxHashSet<usize> = nbrs[v].union(&nbrs[u]).copied().collect();
            s.remove(&u);
            s.remove(&v);
            for &w in &s {
                if u < w || (v < w && w < u && !nbrs[v].contains(&w)) {
                    census[TRICODES[code(v, u, w)] - 1] += 1;
                }
            }
            let rest = (n - s.len() - 2) as u64;
            if succ[v].contains(&u) && succ[u].contains(&v) {
                census[2] += rest; // 102
            } else {
                census[1] += rest; // 012
            }
        }
    }
    let nf = n as u64;
    let total = if n >= 3 {
        nf * (nf - 1) * (nf - 2) / 6
    } else {
        0
    };
    census[0] = total - census[1..].iter().sum::<u64>();
    Ok(TRIAD_NAMES
        .iter()
        .zip(census)
        .map(|(k, c)| (k.to_string(), c))
        .collect())
}
