//! Basic structural queries: biconnected components, local bridges, isolated nodes,
//! barycenters, and the node set of an ego graph.

use crate::algorithms::path_queries::IndexedGraph;
use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BinaryHeap, VecDeque};

fn require_undirected(graph: &Graph, what: &str) -> PyResult<()> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} requires an undirected graph",
            what
        )));
    }
    Ok(())
}

/// Sorted node IDs and an undirected edge-labelled adjacency `adj[i] = [(j, edge_id)]`,
/// without self-loops.
fn edge_adjacency(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>, Vec<Vec<(usize, u64)>>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); ids.len()];
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let (Some(&s), Some(&t)) = (idx.get(&e.source), idx.get(&e.target)) else {
            continue;
        };
        adj[s].push((t, e.id));
        adj[t].push((s, e.id));
    }
    (ids, idx, adj)
}

/// Biconnected components (Hopcroft-Tarjan, iterative, on edges).
///
/// Each component is `(node_ids, edge_ids)`, both sorted. A bridge forms a two-node component;
/// isolated nodes and self-loops belong to no component; parallel edges stay together.
/// Components are ordered by their smallest node ID.
#[pyfunction]
pub fn biconnected_components(graph: &Graph) -> PyResult<Vec<(Vec<u64>, Vec<u64>)>> {
    require_undirected(graph, "biconnected_components()")?;
    let (ids, idx, adj) = edge_adjacency(graph);
    let ends: FxHashMap<u64, (usize, usize)> = graph
        .get_edges_internal()
        .iter()
        .filter(|e| e.source != e.target)
        .map(|e| (e.id, (idx[&e.source], idx[&e.target])))
        .collect();
    let n = ids.len();
    let mut disc = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut timer = 0usize;
    let mut edge_stack: Vec<u64> = Vec::new();
    let mut out: Vec<(Vec<u64>, Vec<u64>)> = Vec::new();

    for root in 0..n {
        if disc[root] != usize::MAX || adj[root].is_empty() {
            continue;
        }
        disc[root] = timer;
        low[root] = timer;
        timer += 1;
        // (node, edge used to reach it, next neighbor index)
        let mut frames: Vec<(usize, u64, usize)> = vec![(root, u64::MAX, 0)];
        while let Some(&mut (v, pe, ref mut i)) = frames.last_mut() {
            if *i < adj[v].len() {
                let (w, eid) = adj[v][*i];
                *i += 1;
                if eid == pe {
                    continue;
                }
                if disc[w] == usize::MAX {
                    edge_stack.push(eid);
                    disc[w] = timer;
                    low[w] = timer;
                    timer += 1;
                    frames.push((w, eid, 0));
                } else if disc[w] < disc[v] {
                    // back edge (or a parallel edge to the parent), recorded once
                    edge_stack.push(eid);
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                frames.pop();
                if let Some(&(u, _, _)) = frames.last() {
                    low[u] = low[u].min(low[v]);
                    if low[v] >= disc[u] {
                        // u separates v's subtree: everything since the tree edge u-v
                        let mut edges = Vec::new();
                        let mut nodes: FxHashSet<usize> = FxHashSet::default();
                        while let Some(e) = edge_stack.pop() {
                            let (a, b) = ends[&e];
                            nodes.insert(a);
                            nodes.insert(b);
                            edges.push(e);
                            if e == pe {
                                break;
                            }
                        }
                        let mut node_ids: Vec<u64> = nodes.into_iter().map(|k| ids[k]).collect();
                        node_ids.sort_unstable();
                        edges.sort_unstable();
                        out.push((node_ids, edges));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.0[0].cmp(&b.0[0]).then_with(|| a.0.cmp(&b.0)));
    Ok(out)
}

/// Local bridges: edges whose endpoints share no neighbor (so the edge is not in any triangle).
///
/// Returns `(edge_id, span)` sorted by edge ID, where span is the shortest-path distance between
/// the endpoints once that edge is removed (hops, or total weight when `weighted`), or infinity
/// if the edge is a bridge. Self-loops are never local bridges.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn local_bridges(graph: &Graph, weighted: bool) -> PyResult<Vec<(u64, f64)>> {
    require_undirected(graph, "local_bridges()")?;
    let (ids, idx, adj) = edge_adjacency(graph);
    let n = ids.len();
    let weight: FxHashMap<u64, f64> = graph
        .get_edges_internal()
        .iter()
        .map(|e| {
            (
                e.id,
                if weighted {
                    e.weight.unwrap_or(1.0)
                } else {
                    1.0
                },
            )
        })
        .collect();
    if weighted && weight.values().any(|&w| w < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "local_bridges() spans require non-negative edge weights",
        ));
    }
    let nbrs: Vec<FxHashSet<usize>> = adj
        .iter()
        .enumerate()
        .map(|(v, a)| a.iter().map(|&(w, _)| w).filter(|&w| w != v).collect())
        .collect();
    let mut out = Vec::new();
    let mut edges: Vec<_> = graph
        .get_edges_internal()
        .iter()
        .filter(|e| e.source != e.target)
        .map(|e| (e.id, idx[&e.source], idx[&e.target]))
        .collect();
    edges.sort_unstable_by_key(|e| e.0);
    for (eid, u, v) in edges {
        let (small, big) = if nbrs[u].len() <= nbrs[v].len() {
            (u, v)
        } else {
            (v, u)
        };
        if nbrs[small].iter().any(|w| nbrs[big].contains(w)) {
            continue;
        }
        // Distance u -> v avoiding this one edge
        let mut dist = vec![f64::INFINITY; n];
        dist[u] = 0.0;
        let span = if weighted {
            let mut heap = BinaryHeap::new();
            heap.push((std::cmp::Reverse(OrdF64(0.0)), u));
            let mut found = f64::INFINITY;
            while let Some((std::cmp::Reverse(OrdF64(d)), x)) = heap.pop() {
                if d > dist[x] {
                    continue;
                }
                if x == v {
                    found = d;
                    break;
                }
                for &(y, e) in &adj[x] {
                    if e == eid {
                        continue;
                    }
                    let nd = d + weight[&e];
                    if nd < dist[y] {
                        dist[y] = nd;
                        heap.push((std::cmp::Reverse(OrdF64(nd)), y));
                    }
                }
            }
            found
        } else {
            let mut queue = VecDeque::from([u]);
            let mut found = f64::INFINITY;
            'bfs: while let Some(x) = queue.pop_front() {
                for &(y, e) in &adj[x] {
                    if e == eid || dist[y].is_finite() {
                        continue;
                    }
                    dist[y] = dist[x] + 1.0;
                    if y == v {
                        found = dist[y];
                        break 'bfs;
                    }
                    queue.push_back(y);
                }
            }
            found
        };
        out.push((eid, span));
    }
    Ok(out)
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Nodes with no incident edges (in either direction), sorted. A self-loop makes a node
/// non-isolated.
#[pyfunction]
pub fn isolates(graph: &Graph) -> Vec<u64> {
    let touched: FxHashSet<u64> = graph
        .get_edges_internal()
        .iter()
        .flat_map(|e| [e.source, e.target])
        .collect();
    let mut out: Vec<u64> = graph
        .get_node_ids()
        .into_iter()
        .filter(|id| !touched.contains(id))
        .collect();
    out.sort_unstable();
    out
}

/// Barycenter: the nodes minimizing the total shortest-path distance to all other nodes.
///
/// Returns `(node_ids, min_total)`. Distances follow edge direction and use weights when
/// `weighted`. Raises `ValueError` unless every node can reach every other node.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn barycenter(graph: &Graph, weighted: bool) -> PyResult<(Vec<u64>, f64)> {
    use rayon::prelude::*;
    let g = IndexedGraph::new(graph, weighted);
    let n = g.len();
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }
    if weighted
        && graph
            .get_edges_internal()
            .iter()
            .any(|e| e.weight.is_some_and(|w| w < 0.0))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Barycenter requires non-negative edge weights",
        ));
    }
    let totals: Vec<Option<f64>> = (0..n)
        .into_par_iter()
        .map(|s| {
            let mut dist = vec![f64::INFINITY; n];
            dist[s] = 0.0;
            let mut heap = BinaryHeap::new();
            heap.push((std::cmp::Reverse(OrdF64(0.0)), s));
            let mut reached = 0usize;
            let mut total = 0.0;
            while let Some((std::cmp::Reverse(OrdF64(d)), x)) = heap.pop() {
                if d > dist[x] {
                    continue;
                }
                reached += 1;
                total += d;
                for &(y, c) in &g.out[x] {
                    let nd = d + c;
                    if nd < dist[y] {
                        dist[y] = nd;
                        heap.push((std::cmp::Reverse(OrdF64(nd)), y));
                    }
                }
            }
            (reached == n).then_some(total)
        })
        .collect();
    if totals.iter().any(|t| t.is_none()) {
        let what = if graph.is_directed() {
            "strongly connected"
        } else {
            "connected"
        };
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "The barycenter is only defined for a {} graph",
            what
        )));
    }
    let totals: Vec<f64> = totals.into_iter().map(|t| t.unwrap()).collect();
    let best = totals.iter().copied().fold(f64::INFINITY, f64::min);
    let tol = 1e-12 * best.abs().max(1.0);
    let nodes = (0..n)
        .filter(|&i| (totals[i] - best).abs() <= tol)
        .map(|i| g.ids[i])
        .collect();
    Ok((nodes, best))
}

/// Nodes within `radius` of `center` (inclusive), sorted, with their distances.
///
/// `mode` is "out" (follow edges), "in" (follow edges backward) or "all" (ignore direction);
/// it only matters on directed graphs. Distances are hops, or total weight when `weighted`.
#[pyfunction]
#[pyo3(signature = (graph, center, radius, mode="out", weighted=false))]
pub fn nodes_within(
    graph: &Graph,
    center: u64,
    radius: f64,
    mode: &str,
    weighted: bool,
) -> PyResult<Vec<(u64, f64)>> {
    let (use_out, use_in) = match mode {
        "out" => (true, false),
        "in" => (false, true),
        "all" | "total" => (true, true),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid mode '{}'. Must be 'out', 'in', or 'all'",
                mode
            )))
        }
    };
    if weighted
        && graph
            .get_edges_internal()
            .iter()
            .any(|e| e.weight.is_some_and(|w| w < 0.0))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Distances require non-negative edge weights",
        ));
    }
    let g = IndexedGraph::new(graph, weighted);
    let s = g.index_of(center)?;
    let n = g.len();
    let mut dist = vec![f64::INFINITY; n];
    dist[s] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push((std::cmp::Reverse(OrdF64(0.0)), s));
    let tol = 1e-12 * radius.abs().max(1.0);
    while let Some((std::cmp::Reverse(OrdF64(d)), x)) = heap.pop() {
        if d > dist[x] {
            continue;
        }
        let lists = [
            if use_out { Some(&g.out[x]) } else { None },
            if use_in { Some(&g.inn[x]) } else { None },
        ];
        for list in lists.into_iter().flatten() {
            for &(y, c) in list {
                let nd = d + c;
                if nd <= radius + tol && nd < dist[y] {
                    dist[y] = nd;
                    heap.push((std::cmp::Reverse(OrdF64(nd)), y));
                }
            }
        }
    }
    Ok((0..n)
        .filter(|&i| dist[i].is_finite())
        .map(|i| (g.ids[i], dist[i]))
        .collect())
}
