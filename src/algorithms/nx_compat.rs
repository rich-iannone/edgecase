//! Algorithm kernels with NetworkX's exact semantics, for edgecase's NetworkX backend.
//!
//! edgecase's own functions sometimes define things differently from NetworkX (PageRank's
//! handling of dangling nodes, for example), and the backend must return what NetworkX
//! returns. These kernels follow the NetworkX implementations step by step, including the
//! order in which they visit nodes, so that results match NetworkX's (up to floating-point
//! rounding).
//!
//! Nodes are visited in increasing ID order (the NetworkX backend numbers nodes `1..=n` in
//! NetworkX node order) and neighbors in the order of the graph's neighbor lists (which
//! `Graph.from_edge_arrays()` takes from NetworkX's adjacency). Results are vectors indexed by
//! the position of each node in increasing ID order.

use crate::graph::Graph;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Node IDs in increasing order and the position of each
fn node_positions(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let ids = graph.sorted_node_ids();
    let positions = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, positions)
}

/// Compressed adjacency over node positions: the distinct out-neighbors of node `i` (both
/// ends of an undirected edge) are `targets[offsets[i]..offsets[i + 1]]`, in the order of the
/// graph's neighbor lists (NetworkX's order for graphs built by `Graph.from_edge_arrays()`
/// from NetworkX adjacency), and `weights` holds the smallest weight among the edges to each
/// (as NetworkX's shortest-path searches see a multigraph; a missing weight counts as 1.0).
struct Adjacency {
    offsets: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<f64>,
}

impl Adjacency {
    fn build(
        graph: &Graph,
        ids: &[u64],
        positions: &FxHashMap<u64, usize>,
        weighted: bool,
    ) -> Self {
        let n = ids.len();
        let lists = graph.get_adjacency_list();
        let mut owner = vec![usize::MAX; n]; // neighbor -> last node that listed it
        let mut offsets = Vec::with_capacity(n + 1);
        let mut targets = Vec::with_capacity(graph.num_edges() * 2);
        let mut weights = Vec::new();
        offsets.push(0);
        for (v, &id) in ids.iter().enumerate() {
            if let Some(neighbors) = lists.get(&id) {
                for &t_id in neighbors {
                    let Some(&t) = positions.get(&t_id) else {
                        continue;
                    };
                    if owner[t] != v {
                        owner[t] = v;
                        targets.push(t);
                        if weighted {
                            weights.push(graph.get_edge_weight(id, t_id));
                        }
                    }
                }
            }
            offsets.push(targets.len());
        }
        Adjacency {
            offsets,
            targets,
            weights,
        }
    }

    #[inline]
    fn range(&self, v: usize) -> std::ops::Range<usize> {
        self.offsets[v]..self.offsets[v + 1]
    }
}

// ============================================================================
// PageRank
// ============================================================================

/// PageRank as `networkx.pagerank` computes it.
///
/// Every edge is an arc (both ways for an undirected edge, once for a self-loop) of weight
/// `weight` (1.0 if missing, or for every edge without `weighted`); parallel arcs add up.
/// Rank of a node whose out-weight is 0 is spread by `dangling` (default: the
/// personalization vector). Iterates until the L1 change is below `n * tol`.
///
/// `personalization`, `nstart` and `dangling` are per node in increasing ID order and are
/// normalized here; they must have one entry per node. Returns the ranks in the same order,
/// or None if `max_iter` iterations do not converge.
#[pyfunction]
#[pyo3(signature = (graph, alpha=0.85, max_iter=100, tol=1e-6, weighted=false,
                    personalization=None, nstart=None, dangling=None))]
#[allow(clippy::too_many_arguments)]
pub fn nx_pagerank(
    graph: &Graph,
    alpha: f64,
    max_iter: usize,
    tol: f64,
    weighted: bool,
    personalization: Option<Vec<f64>>,
    nstart: Option<Vec<f64>>,
    dangling: Option<Vec<f64>>,
) -> PyResult<Option<Vec<f64>>> {
    let (_, positions) = node_positions(graph);
    let n = positions.len();
    if n == 0 {
        return Ok(Some(Vec::new()));
    }
    for (name, vector) in [
        ("personalization", &personalization),
        ("nstart", &nstart),
        ("dangling", &dangling),
    ] {
        if let Some(v) = vector {
            if v.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{} must have one value per node",
                    name
                )));
            }
        }
    }

    // Incoming arcs per target, with the source's transition probability w / out-weight
    let directed = graph.is_directed();
    let mut arcs: Vec<(usize, usize, f64)> = Vec::with_capacity(graph.num_edges() * 2);
    for edge in graph.get_edges_internal() {
        let (Some(&s), Some(&t)) = (positions.get(&edge.source), positions.get(&edge.target))
        else {
            continue;
        };
        let w = if weighted {
            edge.weight.unwrap_or(1.0)
        } else {
            1.0
        };
        arcs.push((s, t, w));
        if !directed && s != t {
            arcs.push((t, s, w));
        }
    }
    let mut out_weight = vec![0.0f64; n];
    let mut in_offsets = vec![0usize; n + 1];
    for &(s, t, w) in &arcs {
        out_weight[s] += w;
        in_offsets[t + 1] += 1;
    }
    for i in 0..n {
        in_offsets[i + 1] += in_offsets[i];
    }
    let is_dangling: Vec<bool> = out_weight.iter().map(|&s| s == 0.0).collect();
    let mut fill = in_offsets.clone();
    let mut incoming = vec![(0usize, 0.0f64); arcs.len()];
    for &(s, t, w) in &arcs {
        let coefficient = if is_dangling[s] {
            0.0
        } else {
            w * (1.0 / out_weight[s])
        };
        incoming[fill[t]] = (s, coefficient);
        fill[t] += 1;
    }
    drop(arcs);

    let normalized = |v: Vec<f64>| {
        let total: f64 = v.iter().sum();
        v.into_iter().map(|x| x / total).collect::<Vec<f64>>()
    };
    let mut x = match nstart {
        Some(v) => normalized(v),
        None => vec![1.0 / n as f64; n],
    };
    let p = match personalization {
        Some(v) => normalized(v),
        None => vec![1.0 / n as f64; n],
    };
    let dangling_weights = match dangling {
        Some(v) => normalized(v),
        None => p.clone(),
    };

    let mut new_x = vec![0.0f64; n];
    for _ in 0..max_iter {
        let dangling_sum: f64 = x
            .iter()
            .zip(&is_dangling)
            .filter(|(_, &d)| d)
            .map(|(v, _)| v)
            .sum();
        let step = |(t, value): (usize, &mut f64)| {
            let mut sum = 0.0;
            for &(s, c) in &incoming[in_offsets[t]..in_offsets[t + 1]] {
                sum += x[s] * c;
            }
            *value = alpha * (sum + dangling_sum * dangling_weights[t]) + (1.0 - alpha) * p[t];
        };
        if n >= 1000 {
            new_x.par_iter_mut().enumerate().for_each(step);
        } else {
            new_x.iter_mut().enumerate().for_each(step);
        }
        let err: f64 = x.iter().zip(&new_x).map(|(a, b)| (b - a).abs()).sum();
        std::mem::swap(&mut x, &mut new_x);
        if err < n as f64 * tol {
            return Ok(Some(x));
        }
    }
    Ok(None)
}

// ============================================================================
// Betweenness centrality
// ============================================================================

/// One source's shortest-path DAG: nodes in order of discovery, predecessors, path counts
struct ShortestPaths {
    order: Vec<usize>,
    preds: Vec<Vec<usize>>,
    sigma: Vec<f64>,
}

impl ShortestPaths {
    fn new(n: usize) -> Self {
        ShortestPaths {
            order: Vec::new(),
            preds: vec![Vec::new(); n],
            sigma: vec![0.0; n],
        }
    }

    /// Reset the entries `order` touched (so one buffer serves many sources)
    fn clear(&mut self) {
        for &v in &self.order {
            self.preds[v].clear();
            self.sigma[v] = 0.0;
        }
        self.order.clear();
    }
}

/// `_single_source_shortest_path_basic` (BFS)
fn bfs_paths(adj: &Adjacency, s: usize, dist: &mut [usize], paths: &mut ShortestPaths) {
    const UNSEEN: usize = usize::MAX;
    paths.sigma[s] = 1.0;
    dist[s] = 0;
    let mut queue = VecDeque::from([s]);
    let mut seen = vec![s];
    while let Some(v) = queue.pop_front() {
        paths.order.push(v);
        let dv = dist[v];
        let sigma_v = paths.sigma[v];
        for k in adj.range(v) {
            let w = adj.targets[k];
            if dist[w] == UNSEEN {
                queue.push_back(w);
                dist[w] = dv + 1;
                seen.push(w);
            }
            if dist[w] == dv + 1 {
                paths.sigma[w] += sigma_v;
                paths.preds[w].push(v);
            }
        }
    }
    for v in seen {
        dist[v] = UNSEEN;
    }
}

/// Heap entry for Dijkstra, ordered like NetworkX's `(distance, counter, pred, node)` tuples
struct HeapEntry {
    dist: f64,
    count: u64,
    pred: usize,
    node: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // reversed: BinaryHeap is a max-heap
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.count.cmp(&self.count))
    }
}

/// `_single_source_dijkstra_path_basic`
fn dijkstra_paths(
    adj: &Adjacency,
    s: usize,
    seen: &mut [f64],
    done: &mut [bool],
    paths: &mut ShortestPaths,
) {
    let mut touched = vec![s];
    paths.sigma[s] = 1.0;
    seen[s] = 0.0;
    let mut count = 0u64;
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        dist: 0.0,
        count,
        pred: s,
        node: s,
    });
    while let Some(HeapEntry {
        dist,
        pred,
        node: v,
        ..
    }) = heap.pop()
    {
        if done[v] {
            continue;
        }
        paths.sigma[v] += paths.sigma[pred];
        paths.order.push(v);
        done[v] = true;
        for k in adj.range(v) {
            let w = adj.targets[k];
            let vw_dist = dist + adj.weights[k];
            if !done[w] && (seen[w].is_nan() || vw_dist < seen[w]) {
                if seen[w].is_nan() {
                    touched.push(w);
                }
                seen[w] = vw_dist;
                count += 1;
                heap.push(HeapEntry {
                    dist: vw_dist,
                    count,
                    pred: v,
                    node: w,
                });
                paths.sigma[w] = 0.0;
                paths.preds[w].clear();
                paths.preds[w].push(v);
            } else if vw_dist == seen[w] {
                paths.sigma[w] += paths.sigma[v];
                paths.preds[w].push(v);
            }
        }
    }
    // Every node pushed onto the heap is popped and finalized, so `paths.clear()` resets the
    // entries of all touched nodes
    for v in touched {
        seen[v] = f64::NAN;
        done[v] = false;
    }
}

/// `_accumulate_basic` / `_accumulate_endpoints`
fn accumulate(
    betweenness: &mut [f64],
    delta: &mut [f64],
    paths: &ShortestPaths,
    s: usize,
    endpoints: bool,
) {
    if endpoints {
        betweenness[s] += (paths.order.len() - 1) as f64;
    }
    for &w in paths.order.iter().rev() {
        let coefficient = (1.0 + delta[w]) / paths.sigma[w];
        for &v in &paths.preds[w] {
            delta[v] += paths.sigma[v] * coefficient;
        }
        if w != s {
            betweenness[w] += delta[w] + if endpoints { 1.0 } else { 0.0 };
        }
    }
    for &w in &paths.order {
        delta[w] = 0.0;
    }
}

/// Unnormalized node betweenness as `networkx.betweenness_centrality` accumulates it, before
/// its final rescaling.
///
/// `sources` (node IDs; default every node) are the sources of the shortest paths. With
/// `weighted`, path lengths are sums of edge weights (the smallest among parallel edges,
/// 1.0 if missing), else edge counts. Returns the sums per node in increasing ID order.
#[pyfunction]
#[pyo3(signature = (graph, sources=None, weighted=false, endpoints=false))]
pub fn nx_betweenness_centrality(
    graph: &Graph,
    sources: Option<Vec<u64>>,
    weighted: bool,
    endpoints: bool,
) -> PyResult<Vec<f64>> {
    let (ids, positions) = node_positions(graph);
    let n = ids.len();
    let sources: Vec<usize> = match sources {
        Some(sources) => sources
            .iter()
            .map(|id| {
                positions.get(id).copied().ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!("Node {} does not exist", id))
                })
            })
            .collect::<PyResult<_>>()?,
        None => (0..n).collect(),
    };
    let adj = Adjacency::build(graph, &ids, &positions, weighted);

    // Fixed chunks of sources, each summed in order and then added up in order, so the result
    // does not depend on thread scheduling
    let chunk_size = sources.len().div_ceil(64).max(16);
    let partials: Vec<Vec<f64>> = sources
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut betweenness = vec![0.0f64; n];
            let mut delta = vec![0.0f64; n];
            let mut paths = ShortestPaths::new(n);
            let mut dist = vec![usize::MAX; n];
            let mut seen = vec![f64::NAN; n];
            let mut done = vec![false; n];
            for &s in chunk {
                if weighted {
                    dijkstra_paths(&adj, s, &mut seen, &mut done, &mut paths);
                } else {
                    bfs_paths(&adj, s, &mut dist, &mut paths);
                }
                accumulate(&mut betweenness, &mut delta, &paths, s, endpoints);
                paths.clear();
            }
            betweenness
        })
        .collect();
    let mut betweenness = vec![0.0f64; n];
    for partial in partials {
        for (total, value) in betweenness.iter_mut().zip(partial) {
            *total += value;
        }
    }
    Ok(betweenness)
}

// ============================================================================
// Strongly connected components
// ============================================================================

/// Strongly connected components in the order `networkx.strongly_connected_components`
/// yields them (its non-recursive Tarjan/Pearce search), as lists of node IDs.
#[pyfunction]
pub fn nx_strongly_connected_components(graph: &Graph) -> Vec<Vec<u64>> {
    let (ids, positions) = node_positions(graph);
    let n = ids.len();
    let adj = Adjacency::build(graph, &ids, &positions, false);
    const UNSET: usize = 0;
    let mut lowlink = vec![UNSET; n]; // preorder indices start at 1
    let mut found = vec![false; n];
    let mut components: Vec<Vec<u64>> = Vec::new();
    // DFS path: (node, next position in its neighbor list, lead flag)
    let mut dfs: Vec<(usize, usize, bool)> = Vec::new();
    let mut comp_stack: Vec<usize> = Vec::new();
    let mut index = 0usize;

    for source in 0..n {
        if lowlink[source] != UNSET {
            continue;
        }
        index += 1;
        let root_low = index;
        lowlink[source] = index;
        dfs.push((source, adj.offsets[source], true));
        'path: while let Some(&(v, _, _)) = dfs.last() {
            let end = adj.offsets[v + 1];
            loop {
                let top = dfs.len() - 1;
                let k = dfs[top].1;
                if k >= end {
                    break;
                }
                dfs[top].1 += 1;
                let w = adj.targets[k];
                if lowlink[w] == UNSET {
                    // Tree arc: previsit w and descend
                    index += 1;
                    lowlink[w] = index;
                    dfs.push((w, adj.offsets[w], true));
                    continue 'path;
                }
                // Back or cross arc
                if !found[w] && lowlink[v] > lowlink[w] {
                    dfs[top].2 = false;
                    lowlink[v] = lowlink[w];
                    if lowlink[v] == root_low && index == n {
                        // Every node is in this DFS tree and v links back to the root: the
                        // rest of the graph is one component
                        let mut component: Vec<u64> = comp_stack.iter().map(|&u| ids[u]).collect();
                        component.extend(dfs.iter().map(|&(u, _, _)| ids[u]));
                        components.push(component);
                        return components;
                    }
                }
            }
            // All successors of v processed: postvisit
            let Some((v, _, lead)) = dfs.pop() else {
                break;
            };
            if lead {
                let v_low = lowlink[v];
                let mut component = vec![ids[v]];
                found[v] = true;
                while let Some(&u) = comp_stack.last() {
                    if lowlink[u] < v_low {
                        break;
                    }
                    comp_stack.pop();
                    found[u] = true;
                    component.push(ids[u]);
                }
                components.push(component);
            } else {
                // (a DFS root always leads, so v has a parent)
                comp_stack.push(v);
                if let Some(parent) = dfs.last_mut() {
                    if lowlink[parent.0] > lowlink[v] {
                        parent.2 = false;
                        lowlink[parent.0] = lowlink[v];
                    }
                }
            }
        }
    }
    components
}
