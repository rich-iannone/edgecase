//! Shortest-path queries: single/multi-source distances and paths, path lengths, reachability,
//! and enumeration of every shortest path between two nodes.
//!
//! All functions follow the convention of `shortest_path()`: on a weighted graph
//! (`Graph(weighted=True)`) each step costs the edge's weight (the cheapest of any parallel
//! edges; a missing weight counts as 1.0) and Dijkstra's algorithm is used; on an unweighted
//! graph every step costs one hop. Directed graphs are followed in edge direction.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Relative tolerance for treating two floating-point path costs as equal (ties).
const TIE_EPS: f64 = 1e-12;

/// Index-based view of a graph for path algorithms.
///
/// Nodes are sorted by ID so results (and tie-breaking) are deterministic. `out[i]` lists
/// `(neighbor, cost)`; `inn[i]` lists predecessors. Undirected edges appear in both directions
/// and `inn == out`. Self-loops are dropped.
pub(crate) struct IndexedGraph {
    pub ids: Vec<u64>,
    pub idx: FxHashMap<u64, usize>,
    pub out: Vec<Vec<(usize, f64)>>,
    pub inn: Vec<Vec<(usize, f64)>>,
}

impl IndexedGraph {
    /// Build the view. With `use_weights`, costs come from edge weights (default 1.0).
    pub fn new(graph: &Graph, use_weights: bool) -> Self {
        let mut ids = graph.get_node_ids();
        ids.sort_unstable();
        let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n = ids.len();
        let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut inn: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let undirected = !graph.is_directed();
        for e in graph.get_edges_internal() {
            if e.source == e.target {
                continue;
            }
            let (Some(&s), Some(&t)) = (idx.get(&e.source), idx.get(&e.target)) else {
                continue;
            };
            let w = if use_weights {
                e.weight.unwrap_or(1.0)
            } else {
                1.0
            };
            out[s].push((t, w));
            inn[t].push((s, w));
            if undirected {
                out[t].push((s, w));
                inn[s].push((t, w));
            }
        }
        IndexedGraph { ids, idx, out, inn }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Resolve a node ID, raising `KeyError` if it is not in the graph.
    pub fn index_of(&self, id: u64) -> PyResult<usize> {
        self.idx.get(&id).copied().ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Node {} does not exist", id))
        })
    }
}

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on cost, ties broken by node index for determinism
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn same_cost(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return a == b;
    }
    (a - b).abs() <= TIE_EPS * a.abs().max(b.abs()).max(1.0)
}

fn reject_negative_weights(graph: &Graph) -> PyResult<()> {
    if !graph.is_weighted() {
        return Ok(());
    }
    if let Some(e) = graph
        .get_edges_internal()
        .iter()
        .find(|e| e.weight.is_some_and(|w| w < 0.0))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Shortest paths require non-negative edge weights, but edge {} ({} -> {}) has \
             weight {}. Use bellman_ford_shortest_path() for graphs with negative weights.",
            e.id,
            e.source,
            e.target,
            e.weight.unwrap_or_default()
        )));
    }
    Ok(())
}

/// Result of a (multi-)source shortest-path search.
struct Search {
    dist: Vec<f64>,
    /// First predecessor found on a shortest path (None for sources and unreached nodes)
    pred: Vec<Option<usize>>,
    /// Every predecessor on some shortest path (only filled when `all_preds`)
    preds: Vec<Vec<usize>>,
}

/// Dijkstra from a set of sources (all at distance 0).
///
/// Stops early once `target` is settled. Nodes farther than `cutoff` are not reached.
fn dijkstra(
    g: &IndexedGraph,
    sources: &[usize],
    target: Option<usize>,
    cutoff: Option<f64>,
    all_preds: bool,
) -> Search {
    let n = g.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    let mut preds: Vec<Vec<usize>> = if all_preds {
        vec![Vec::new(); n]
    } else {
        Vec::new()
    };
    let mut done = vec![false; n];
    let mut heap = BinaryHeap::new();
    for &s in sources {
        if dist[s] != 0.0 {
            dist[s] = 0.0;
            heap.push(State { cost: 0.0, node: s });
        }
    }
    while let Some(State { cost, node }) = heap.pop() {
        if done[node] || cost > dist[node] {
            continue;
        }
        done[node] = true;
        if Some(node) == target {
            break;
        }
        for &(w, c) in &g.out[node] {
            let nc = cost + c;
            if let Some(cut) = cutoff {
                if nc > cut && !same_cost(nc, cut) {
                    continue;
                }
            }
            if done[w] {
                continue;
            }
            if nc < dist[w] && !same_cost(nc, dist[w]) {
                dist[w] = nc;
                pred[w] = Some(node);
                if all_preds {
                    preds[w].clear();
                    preds[w].push(node);
                }
                heap.push(State { cost: nc, node: w });
            } else if all_preds && same_cost(nc, dist[w]) && !preds[w].contains(&node) {
                preds[w].push(node);
            }
        }
    }
    Search { dist, pred, preds }
}

fn path_to(search: &Search, g: &IndexedGraph, t: usize) -> Vec<u64> {
    let mut path = vec![g.ids[t]];
    let mut cur = t;
    while let Some(p) = search.pred[cur] {
        path.push(g.ids[p]);
        cur = p;
    }
    path.reverse();
    path
}

fn resolve_sources(g: &IndexedGraph, sources: &[u64]) -> PyResult<Vec<usize>> {
    if sources.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "At least one source node is required",
        ));
    }
    sources.iter().map(|&s| g.index_of(s)).collect()
}

/// Shortest-path distance from the nearest of `sources` to every reachable node.
///
/// Nodes beyond `cutoff` (or unreachable) are omitted. Sources map to 0.
#[pyfunction]
#[pyo3(signature = (graph, sources, cutoff=None))]
pub fn multi_source_distances(
    graph: &Graph,
    sources: Vec<u64>,
    cutoff: Option<f64>,
) -> PyResult<FxHashMap<u64, f64>> {
    reject_negative_weights(graph)?;
    let g = IndexedGraph::new(graph, graph.is_weighted());
    let src = resolve_sources(&g, &sources)?;
    let s = dijkstra(&g, &src, None, cutoff, false);
    Ok(s.dist
        .iter()
        .enumerate()
        .filter(|(_, d)| d.is_finite())
        .map(|(i, &d)| (g.ids[i], d))
        .collect())
}

/// One shortest path from the nearest of `sources` to every reachable node.
///
/// Each path starts at the source it was reached from and ends at the key node.
#[pyfunction]
#[pyo3(signature = (graph, sources, cutoff=None))]
pub fn multi_source_paths(
    graph: &Graph,
    sources: Vec<u64>,
    cutoff: Option<f64>,
) -> PyResult<FxHashMap<u64, Vec<u64>>> {
    reject_negative_weights(graph)?;
    let g = IndexedGraph::new(graph, graph.is_weighted());
    let src = resolve_sources(&g, &sources)?;
    let s = dijkstra(&g, &src, None, cutoff, false);
    Ok((0..g.len())
        .filter(|&i| s.dist[i].is_finite())
        .map(|i| (g.ids[i], path_to(&s, &g, i)))
        .collect())
}

/// Shortest-path distance from `source` to `target`, or None if there is no path.
#[pyfunction]
pub fn shortest_path_length(graph: &Graph, source: u64, target: u64) -> PyResult<Option<f64>> {
    reject_negative_weights(graph)?;
    let g = IndexedGraph::new(graph, graph.is_weighted());
    let s = g.index_of(source)?;
    let t = g.index_of(target)?;
    let res = dijkstra(&g, &[s], Some(t), None, false);
    Ok(res.dist[t].is_finite().then_some(res.dist[t]))
}

/// Whether `target` can be reached from `source` (following edge directions).
#[pyfunction]
pub fn has_path(graph: &Graph, source: u64, target: u64) -> PyResult<bool> {
    let g = IndexedGraph::new(graph, false);
    let s = g.index_of(source)?;
    let t = g.index_of(target)?;
    if s == t {
        return Ok(true);
    }
    let mut seen = vec![false; g.len()];
    let mut queue = VecDeque::from([s]);
    seen[s] = true;
    while let Some(v) = queue.pop_front() {
        for &(w, _) in &g.out[v] {
            if w == t {
                return Ok(true);
            }
            if !seen[w] {
                seen[w] = true;
                queue.push_back(w);
            }
        }
    }
    Ok(false)
}

/// Every shortest path from `source` to `target`, in lexicographic order of node IDs.
///
/// Returns an empty list if there is no path. `max_paths` caps the enumeration (the number of
/// shortest paths can grow exponentially, e.g. on grids).
#[pyfunction]
#[pyo3(signature = (graph, source, target, max_paths=None))]
pub fn all_shortest_paths(
    graph: &Graph,
    source: u64,
    target: u64,
    max_paths: Option<usize>,
) -> PyResult<Vec<Vec<u64>>> {
    reject_negative_weights(graph)?;
    let g = IndexedGraph::new(graph, graph.is_weighted());
    let s = g.index_of(source)?;
    let t = g.index_of(target)?;
    let mut res = dijkstra(&g, &[s], Some(t), None, true);
    if !res.dist[t].is_finite() {
        return Ok(Vec::new());
    }
    for p in res.preds.iter_mut() {
        p.sort_unstable();
    }
    // Walk the predecessor DAG backward from target with an explicit stack
    let limit = max_paths.unwrap_or(usize::MAX);
    let mut out: Vec<Vec<u64>> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(t, 0)]; // (node, next predecessor to try)
    while let Some(&mut (node, ref mut k)) = stack.last_mut() {
        if node == s {
            let mut path: Vec<u64> = stack.iter().rev().map(|&(v, _)| g.ids[v]).collect();
            path.shrink_to_fit();
            out.push(path);
            stack.pop();
            if out.len() >= limit {
                break;
            }
            continue;
        }
        if *k < res.preds[node].len() {
            let p = res.preds[node][*k];
            *k += 1;
            stack.push((p, 0));
        } else {
            stack.pop();
        }
    }
    out.sort();
    Ok(out)
}

/// Bidirectional Dijkstra: search forward from `source` and backward from `target` at the same
/// time, stopping once the two frontiers prove no shorter connection exists. Returns
/// `(length, path)`, or None if `target` is unreachable. Uses weights on weighted graphs and
/// hop counts otherwise, like `shortest_path()`.
#[pyfunction]
pub fn bidirectional_dijkstra(
    graph: &Graph,
    source: u64,
    target: u64,
) -> PyResult<Option<(f64, Vec<u64>)>> {
    reject_negative_weights(graph)?;
    let g = IndexedGraph::new(graph, graph.is_weighted());
    let s = g.index_of(source)?;
    let t = g.index_of(target)?;
    if s == t {
        return Ok(Some((0.0, vec![source])));
    }
    let n = g.len();
    // index 0 = forward (from source along out-edges), 1 = backward (from target along in-edges)
    let mut dist = [vec![f64::INFINITY; n], vec![f64::INFINITY; n]];
    let mut pred: [Vec<Option<usize>>; 2] = [vec![None; n], vec![None; n]];
    let mut done = [vec![false; n], vec![false; n]];
    let mut heaps = [BinaryHeap::new(), BinaryHeap::new()];
    dist[0][s] = 0.0;
    dist[1][t] = 0.0;
    heaps[0].push(State { cost: 0.0, node: s });
    heaps[1].push(State { cost: 0.0, node: t });
    let mut best = f64::INFINITY;
    let mut meet: Option<usize> = None;

    loop {
        // Drop stale entries so each heap's top is a real tentative distance
        for k in 0..2 {
            while let Some(top) = heaps[k].peek() {
                if done[k][top.node] || top.cost > dist[k][top.node] {
                    heaps[k].pop();
                } else {
                    break;
                }
            }
        }
        let top = [
            heaps[0].peek().map(|x| x.cost),
            heaps[1].peek().map(|x| x.cost),
        ];
        let (Some(tf), Some(tb)) = (top[0], top[1]) else {
            break;
        };
        if tf + tb >= best {
            break;
        }
        let k = if tf <= tb { 0 } else { 1 };
        let State { cost, node: v } = heaps[k].pop().unwrap();
        done[k][v] = true;
        let edges = if k == 0 { &g.out[v] } else { &g.inn[v] };
        for &(w, c) in edges {
            let nd = cost + c;
            if nd < dist[k][w] && !same_cost(nd, dist[k][w]) {
                dist[k][w] = nd;
                pred[k][w] = Some(v);
                heaps[k].push(State { cost: nd, node: w });
            }
            let other = dist[1 - k][w];
            if other.is_finite() && dist[k][w] + other < best {
                best = dist[k][w] + other;
                meet = Some(w);
            }
        }
    }

    let Some(mid) = meet else {
        return Ok(None);
    };
    let mut path = vec![g.ids[mid]];
    let mut cur = mid;
    while let Some(p) = pred[0][cur] {
        path.push(g.ids[p]);
        cur = p;
    }
    path.reverse();
    cur = mid;
    while let Some(p) = pred[1][cur] {
        path.push(g.ids[p]);
        cur = p;
    }
    Ok(Some((best, path)))
}
