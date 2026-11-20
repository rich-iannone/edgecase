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

// ============================================================================
// Python's random number generator
// ============================================================================

/// CPython's `random.Random` generator (MT19937) and the parts of its API NetworkX's
/// randomized algorithms use (`shuffle`, `choice`), drawing exactly the numbers Python would.
///
/// The state comes from, and goes back to, `Random.getstate()`/`setstate()`: 624 words and
/// the position of the next one.
pub struct PyRandom {
    mt: [u32; 624],
    index: usize,
}

impl PyRandom {
    const N: usize = 624;
    const M: usize = 397;

    pub fn from_state(words: &[u32], index: usize) -> PyResult<Self> {
        if words.len() != Self::N || index > Self::N {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a Mersenne Twister state is 624 words and an index of at most 624",
            ));
        }
        let mut mt = [0u32; 624];
        mt.copy_from_slice(words);
        Ok(PyRandom { mt, index })
    }

    pub fn state(&self) -> (Vec<u32>, usize) {
        (self.mt.to_vec(), self.index)
    }

    /// `genrand_uint32` in CPython's `_randommodule.c`
    fn next_u32(&mut self) -> u32 {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        let mag01 = |y: u32| if y & 1 == 1 { MATRIX_A } else { 0 };
        if self.index >= Self::N {
            let mt = &mut self.mt;
            for kk in 0..Self::N - Self::M {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + Self::M] ^ (y >> 1) ^ mag01(y);
            }
            for kk in Self::N - Self::M..Self::N - 1 {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + Self::M - Self::N] ^ (y >> 1) ^ mag01(y);
            }
            let y = (mt[Self::N - 1] & UPPER) | (mt[0] & LOWER);
            mt[Self::N - 1] = mt[Self::M - 1] ^ (y >> 1) ^ mag01(y);
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` for `0 < k <= 32`
    fn getrandbits(&mut self, k: u32) -> u32 {
        self.next_u32() >> (32 - k)
    }

    /// `_randbelow_with_getrandbits(n)`, for `0 < n < 2**32`
    pub fn randbelow(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        let mut r = self.getrandbits(k) as usize;
        while r >= n {
            r = self.getrandbits(k) as usize;
        }
        r
    }

    /// `shuffle(x)`
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }
}

/// Words and index of a Python `Random.getstate()` after drawing `shuffle` permutations of
/// `0..n` for each n in `sizes` (for tests: compare with Python's own `shuffle`)
#[pyfunction]
pub fn nx_random_shuffles(
    words: Vec<u32>,
    index: usize,
    sizes: Vec<usize>,
) -> PyResult<(Vec<Vec<usize>>, Vec<u32>, usize)> {
    if sizes.iter().any(|&n| n as u64 > u32::MAX as u64) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sizes must fit in 32 bits",
        ));
    }
    let mut rng = PyRandom::from_state(&words, index)?;
    let perms = sizes
        .iter()
        .map(|&n| {
            let mut x: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut x);
            x
        })
        .collect();
    let (words, index) = rng.state();
    Ok((perms, words, index))
}

/// Neumaier-compensated running sum, the algorithm (and so the rounding) of Python's `sum()`
/// over floats. (The same as the shortest-path module's; kept separate so the two modules
/// can change independently.)
#[derive(Default, Clone, Copy)]
struct PySum {
    sum: f64,
    compensation: f64,
}

impl PySum {
    #[inline]
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.compensation += (self.sum - t) + x;
        } else {
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn total(&self) -> f64 {
        if self.compensation != 0.0 && self.compensation.is_finite() {
            self.sum + self.compensation
        } else {
            self.sum
        }
    }

    fn of(values: impl IntoIterator<Item = f64>) -> f64 {
        let mut s = PySum::default();
        for x in values {
            s.add(x);
        }
        s.total()
    }
}

/// Arcs `src[i] -> dst[i]` (positions, in NetworkX's adjacency order, both ways for an
/// undirected edge) as per-node ranges: arcs of node `u` are `offsets[u]..offsets[u + 1]`.
/// Arcs must come node by node in node order (as an adjacency dict lists them).
fn arc_offsets(n: usize, src: &[usize]) -> PyResult<Vec<usize>> {
    let mut offsets = vec![0usize; n + 1];
    for (i, &s) in src.iter().enumerate() {
        if s >= n || (i > 0 && s < src[i - 1]) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "arcs must be in node order",
            ));
        }
        offsets[s + 1] += 1;
    }
    for u in 0..n {
        offsets[u + 1] += offsets[u];
    }
    Ok(offsets)
}

/// Positions from 1-based node IDs, checking the range
fn positions_of(ids: Vec<u64>, n: usize) -> PyResult<Vec<usize>> {
    ids.into_iter()
        .map(|id| {
            if id == 0 || id as usize > n {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "node ID {} outside 1..={}",
                    id, n
                )))
            } else {
                Ok(id as usize - 1)
            }
        })
        .collect()
}

// ============================================================================
// Asynchronous label propagation
// ============================================================================

/// `networkx.community.asyn_lpa_communities`, drawing the same random numbers as NetworkX
/// from the given `random.Random` state.
///
/// Arcs are NetworkX's adjacency in order (`src`, `dst` as node IDs `1..=n`, grouped by
/// source; successors only for a directed graph). Without `weights` each distinct neighbor
/// counts once; with them each arc adds its weight. Returns each node's final label (labels
/// start as node positions) and the generator's state afterwards.
#[pyfunction]
#[pyo3(signature = (n, src, dst, weights, words, index))]
pub fn nx_asyn_lpa(
    n: usize,
    src: Vec<u64>,
    dst: Vec<u64>,
    weights: Option<Vec<f64>>,
    words: Vec<u32>,
    index: usize,
) -> PyResult<(Vec<usize>, Vec<u32>, usize)> {
    if src.len() != dst.len() || weights.as_ref().is_some_and(|w| w.len() != src.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "src, dst and weights must have the same length",
        ));
    }
    if n as u64 > u32::MAX as u64 {
        return Err(pyo3::exceptions::PyValueError::new_err("too many nodes"));
    }
    let src = positions_of(src, n)?;
    let dst = positions_of(dst, n)?;
    let offsets = arc_offsets(n, &src)?;
    let mut rng = PyRandom::from_state(&words, index)?;
    let mut labels: Vec<usize> = (0..n).collect();
    // label -> slot in the current node's frequency list (`stamp` marks this node's slots)
    let mut slot = vec![0usize; n];
    let mut stamp = vec![usize::MAX; n];
    let mut freq: Vec<(usize, f64)> = Vec::new();
    let mut nodes: Vec<usize> = Vec::with_capacity(n);
    let mut cont = true;
    let mut visit = 0usize;
    while cont {
        cont = false;
        nodes.clear();
        nodes.extend(0..n);
        rng.shuffle(&mut nodes);
        for &node in &nodes {
            let arcs = offsets[node]..offsets[node + 1];
            if arcs.is_empty() {
                continue;
            }
            visit += 1;
            freq.clear();
            let mut previous = usize::MAX;
            for k in arcs {
                let v = dst[k];
                let w = match &weights {
                    Some(weights) => weights[k],
                    None => {
                        // each distinct neighbor once (a multigraph's parallel arcs are
                        // consecutive)
                        if v == previous {
                            continue;
                        }
                        previous = v;
                        1.0
                    }
                };
                let label = labels[v];
                if stamp[label] == visit {
                    freq[slot[label]].1 += w;
                } else {
                    stamp[label] = visit;
                    slot[label] = freq.len();
                    freq.push((label, w));
                }
            }
            let max_freq = freq
                .iter()
                .map(|&(_, f)| f)
                .fold(f64::NEG_INFINITY, f64::max);
            let best: Vec<usize> = freq
                .iter()
                .filter(|&&(_, f)| f == max_freq)
                .map(|&(l, _)| l)
                .collect();
            if !best.contains(&labels[node]) {
                labels[node] = best[rng.randbelow(best.len())];
                cont = true;
            }
        }
    }
    let (words, index) = rng.state();
    Ok((labels, words, index))
}

// ============================================================================
// Louvain
// ============================================================================

/// A weighted graph whose neighbor lists keep insertion order, with NetworkX's
/// `add_edge(u, v, weight=...)` semantics: a new edge is appended at both ends (a self-loop
/// once); for an existing one only the weight changes.
struct OrderedGraph {
    directed: bool,
    succ: Vec<Vec<(usize, f64)>>,
    /// Predecessors (directed only), in the order the edges were added
    pred: Vec<Vec<(usize, f64)>>,
    /// (u, v) -> index of v in `succ[u]` (both orientations for undirected graphs)
    slot: FxHashMap<(usize, usize), usize>,
    /// (v, u) -> index of u in `pred[v]`
    pred_slot: FxHashMap<(usize, usize), usize>,
}

impl OrderedGraph {
    fn new(n: usize, directed: bool) -> Self {
        OrderedGraph {
            directed,
            succ: vec![Vec::new(); n],
            pred: if directed {
                vec![Vec::new(); n]
            } else {
                Vec::new()
            },
            slot: FxHashMap::default(),
            pred_slot: FxHashMap::default(),
        }
    }

    fn len(&self) -> usize {
        self.succ.len()
    }

    /// Add `w` to edge (u, v)'s weight, creating it (with weight `w`) if needed
    fn add_weight(&mut self, u: usize, v: usize, w: f64) {
        if let Some(&i) = self.slot.get(&(u, v)) {
            let total = w + self.succ[u][i].1;
            self.succ[u][i].1 = total;
            if self.directed {
                let j = self.pred_slot[&(v, u)];
                self.pred[v][j].1 = total;
            } else if u != v {
                let j = self.slot[&(v, u)];
                self.succ[v][j].1 = total;
            }
            return;
        }
        self.slot.insert((u, v), self.succ[u].len());
        self.succ[u].push((v, w));
        if self.directed {
            self.pred_slot.insert((v, u), self.pred[v].len());
            self.pred[v].push((u, w));
        } else if u != v {
            self.slot.insert((v, u), self.succ[v].len());
            self.succ[v].push((u, w));
        }
    }

    /// The edges as `G.edges(data="weight")` lists them
    fn edges(&self) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        for (u, nbrs) in self.succ.iter().enumerate() {
            for &(v, w) in nbrs {
                if self.directed || v >= u {
                    out.push((u, v, w));
                }
            }
        }
        out
    }

    fn self_loop(&self, u: usize) -> Option<f64> {
        self.slot.get(&(u, u)).map(|&i| self.succ[u][i].1)
    }

    /// `G.degree(weight="weight")` (undirected) or `G.out_degree(...)`
    fn out_degrees(&self) -> Vec<f64> {
        (0..self.len())
            .map(|u| {
                let d = PySum::of(self.succ[u].iter().map(|&(_, w)| w));
                if self.directed {
                    d
                } else {
                    d + self.self_loop(u).unwrap_or(0.0)
                }
            })
            .collect()
    }

    /// `G.in_degree(weight="weight")` (directed)
    fn in_degrees(&self) -> Vec<f64> {
        self.pred
            .iter()
            .map(|p| PySum::of(p.iter().map(|&(_, w)| w)))
            .collect()
    }

    /// `networkx.community.modularity(G, communities, weight="weight", resolution=...)`,
    /// summing in community order and, within a community, in node order
    fn modularity(&self, communities: &[Vec<usize>], resolution: f64) -> f64 {
        let n = self.len();
        let out_degree = self.out_degrees();
        let in_degree = if self.directed {
            self.in_degrees()
        } else {
            out_degree.clone()
        };
        let (m, norm) = if self.directed {
            let m = PySum::of(out_degree.iter().copied());
            (m, 1.0 / (m * m))
        } else {
            let deg_sum = PySum::of(out_degree.iter().copied());
            (deg_sum / 2.0, 1.0 / (deg_sum * deg_sum))
        };
        if m == 0.0 {
            return 0.0;
        }
        let mut member = vec![usize::MAX; n];
        let mut total = PySum::default();
        for (c, community) in communities.iter().enumerate() {
            let mut sorted = community.clone();
            sorted.sort_unstable();
            for &u in &sorted {
                member[u] = c;
            }
            // each internal edge once: from its first endpoint in node order
            let mut inside = PySum::default();
            for &u in &sorted {
                for &(v, w) in &self.succ[u] {
                    if member[v] == c && (self.directed || v >= u) {
                        inside.add(w);
                    }
                }
            }
            let out_sum = PySum::of(sorted.iter().map(|&u| out_degree[u]));
            let in_sum = if self.directed {
                PySum::of(sorted.iter().map(|&u| in_degree[u]))
            } else {
                out_sum
            };
            total.add(inside.total() / m - resolution * out_sum * in_sum * norm);
        }
        total.total()
    }
}

/// Louvain community detection as `networkx.community.louvain_partitions` runs it, level by
/// level, with the random visiting order of each level drawn from a Python `random.Random`
/// state as NetworkX draws it, so the partitions are NetworkX's.
///
/// Python drives the levels (yielding partitions and checking the modularity threshold) and
/// calls `one_level()` and `aggregate()`.
#[pyclass]
pub struct NxLouvain {
    graph: OrderedGraph,
    /// Original nodes (positions) inside each node of `graph`
    members: Vec<Vec<usize>>,
    /// This level's communities: nodes of `graph`, and the original nodes in each
    inner: Vec<Vec<usize>>,
    partition: Vec<Vec<usize>>,
    /// Size (total edge weight) of the original graph
    m: f64,
    resolution: f64,
}

#[pymethods]
impl NxLouvain {
    /// The level-0 graph: `G.edges(data=weight, default=1)` of the original graph (node IDs
    /// `1..=n`), merged into a simple graph by adding the weights of parallel edges; `m` is
    /// the original graph's `size(weight)`.
    #[new]
    #[pyo3(signature = (n, sources, targets, weights, directed, m, resolution))]
    fn new(
        n: usize,
        sources: Vec<u64>,
        targets: Vec<u64>,
        weights: Vec<f64>,
        directed: bool,
        m: f64,
        resolution: f64,
    ) -> PyResult<Self> {
        if sources.len() != targets.len() || weights.len() != sources.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sources, targets and weights must have the same length",
            ));
        }
        let sources = positions_of(sources, n)?;
        let targets = positions_of(targets, n)?;
        let mut graph = OrderedGraph::new(n, directed);
        for ((&u, &v), &w) in sources.iter().zip(&targets).zip(&weights) {
            graph.add_weight(u, v, w);
        }
        let singletons: Vec<Vec<usize>> = (0..n).map(|u| vec![u]).collect();
        Ok(NxLouvain {
            graph,
            members: singletons.clone(),
            inner: singletons.clone(),
            partition: singletons,
            m,
            resolution,
        })
    }

    /// Number of nodes of the current level's graph
    fn num_nodes(&self) -> usize {
        self.graph.len()
    }

    /// The modularity of the singleton partition of the current graph
    fn singleton_modularity(&self) -> f64 {
        let singletons: Vec<Vec<usize>> = (0..self.graph.len()).map(|u| vec![u]).collect();
        self.graph.modularity(&singletons, self.resolution)
    }

    /// `_one_level()`: move nodes between communities in the visiting order NetworkX draws
    /// (`random.Random.shuffle` from state `words`/`index`) until no move helps. Returns
    /// whether any node moved, and the generator's state afterwards.
    fn one_level(&mut self, words: Vec<u32>, index: usize) -> PyResult<(bool, Vec<u32>, usize)> {
        let mut rng = PyRandom::from_state(&words, index)?;
        let g = &self.graph;
        let n = g.len();
        let directed = g.directed;
        let m = self.m;
        let mut node2com: Vec<usize> = (0..n).collect();
        let out_degree = g.out_degrees();
        let in_degree = if directed { g.in_degrees() } else { Vec::new() };
        let gamma = if directed {
            self.resolution
        } else {
            self.resolution / 2.0
        };
        let mut stot = out_degree.clone(); // Stot (undirected) or Stot_out
        let mut stot_in = in_degree.clone();
        // Neighbors (no self-loops) with weights, in NetworkX's order
        let nbrs: Vec<Vec<(usize, f64)>> = (0..n)
            .map(|u| {
                if !directed {
                    return g.succ[u].iter().copied().filter(|&(v, _)| v != u).collect();
                }
                let mut list: Vec<(usize, f64)> = Vec::new();
                let mut at: FxHashMap<usize, usize> = FxHashMap::default();
                for &(v, w) in g.succ[u].iter().chain(g.pred[u].iter()) {
                    if v == u {
                        continue;
                    }
                    match at.get(&v) {
                        Some(&i) => list[i].1 += w,
                        None => {
                            at.insert(v, list.len());
                            list.push((v, 0.0 + w));
                        }
                    }
                }
                list
            })
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);

        // community -> slot in `by_com` for the node being moved (`stamp` marks this visit)
        let mut slot = vec![0usize; n];
        let mut stamp = vec![usize::MAX; n];
        let mut by_com: Vec<(usize, f64)> = Vec::new();
        let mut visit = 0usize;
        let mut improvement = false;
        let mut moves = 1;
        while moves > 0 {
            moves = 0;
            for &u in &order {
                visit += 1;
                by_com.clear();
                let u_com = node2com[u];
                for &(v, w) in &nbrs[u] {
                    let c = node2com[v];
                    if stamp[c] == visit {
                        by_com[slot[c]].1 += w;
                    } else {
                        stamp[c] = visit;
                        slot[c] = by_com.len();
                        by_com.push((c, 0.0 + w));
                    }
                }
                let u_in_com = if stamp[u_com] == visit {
                    by_com[slot[u_com]].1
                } else {
                    0.0
                };
                let x = if directed {
                    stot_in[u_com] -= in_degree[u];
                    stot[u_com] -= out_degree[u];
                    out_degree[u] * stot_in[u_com] + in_degree[u] * stot[u_com]
                } else {
                    stot[u_com] -= out_degree[u];
                    stot[u_com] * out_degree[u]
                };
                let remove_cost = u_in_com * m - gamma * x;
                let mut best_gain = remove_cost;
                let mut best_com = u_com;
                for &(c, w) in &by_com {
                    let x = if directed {
                        out_degree[u] * stot_in[c] + in_degree[u] * stot[c]
                    } else {
                        stot[c] * out_degree[u]
                    };
                    let gain = w * m - gamma * x;
                    if gain > best_gain {
                        best_gain = gain;
                        best_com = c;
                    }
                }
                if directed {
                    stot_in[best_com] += in_degree[u];
                }
                stot[best_com] += out_degree[u];
                if best_com != u_com {
                    node2com[u] = best_com;
                    improvement = true;
                    moves += 1;
                }
            }
        }

        // Communities in index order, empty ones dropped
        let mut inner: Vec<Vec<usize>> = vec![Vec::new(); n];
        for u in 0..n {
            inner[node2com[u]].push(u);
        }
        let inner: Vec<Vec<usize>> = inner.into_iter().filter(|c| !c.is_empty()).collect();
        self.partition = inner
            .iter()
            .map(|c| {
                let mut nodes: Vec<usize> = c
                    .iter()
                    .flat_map(|&u| self.members[u].iter().copied())
                    .collect();
                nodes.sort_unstable();
                nodes
            })
            .collect();
        self.inner = inner;
        let (words, index) = rng.state();
        Ok((improvement, words, index))
    }

    /// The communities found by the last `one_level()`, as lists of original node IDs
    fn partition(&self) -> Vec<Vec<u64>> {
        self.partition
            .iter()
            .map(|c| c.iter().map(|&o| o as u64 + 1).collect())
            .collect()
    }

    /// Modularity of the last level's communities on the current graph
    fn inner_modularity(&self) -> f64 {
        self.graph.modularity(&self.inner, self.resolution)
    }

    /// `_aggregate_graph()`: one node per community, edges summed between communities
    fn aggregate(&mut self) {
        let k = self.inner.len();
        let mut com = vec![0usize; self.graph.len()];
        for (i, c) in self.inner.iter().enumerate() {
            for &u in c {
                com[u] = i;
            }
        }
        let mut graph = OrderedGraph::new(k, self.graph.directed);
        for (u, v, w) in self.graph.edges() {
            graph.add_weight(com[u], com[v], w);
        }
        self.graph = graph;
        self.members = self.partition.clone();
        self.inner = (0..k).map(|u| vec![u]).collect();
    }
}

// ============================================================================
// Greedy modularity (Clauset-Newman-Moore)
// ============================================================================

/// A heap priority compared like a Python float: -0.0 and 0.0 are equal (they tie, and the
/// element breaks the tie), which `f64::total_cmp` alone would not do
#[derive(Clone, Copy, PartialEq)]
struct Priority(f64);

impl Priority {
    fn new(x: f64) -> Self {
        Priority(x + 0.0) // -0.0 + 0.0 == 0.0
    }
}

impl Eq for Priority {}
impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// `networkx.community.greedy_modularity_communities` up to its final sort: the merges
/// (absorbed community, absorbing community) it makes, as positions, and whether the merging
/// ran out (NetworkX's `StopIteration` branch) rather than stopped.
///
/// `edges` are `G.edges(data=weight, default=1)` (node IDs `1..=n`, in order); `a` and `b`
/// the per-node `a`/`b` vectors NetworkX computes (equal for undirected graphs); `rank` each
/// node's position in sorted node order, which breaks ties between equal gains as NetworkX's
/// `(row, col)` heap elements do. Merging stops as NetworkX's loop does for `cutoff` and
/// `best_n`. The heaps are ordered sets: popping the least `(priority, element)` is what
/// NetworkX's `MappedQueue` does, so the merges are the same.
#[pyfunction]
#[pyo3(signature = (n, sources, targets, weights, a, b, rank, q0, resolution, directed, cutoff, best_n))]
#[allow(clippy::too_many_arguments)]
pub fn nx_greedy_modularity(
    n: usize,
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    rank: Vec<usize>,
    q0: f64,
    resolution: f64,
    directed: bool,
    cutoff: usize,
    best_n: usize,
) -> PyResult<(Vec<(usize, usize)>, bool)> {
    use std::collections::BTreeSet;
    if sources.len() != targets.len()
        || weights.len() != sources.len()
        || a.len() != n
        || b.len() != n
        || rank.len() != n
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent input lengths",
        ));
    }
    let mut node_at = vec![usize::MAX; n]; // rank -> node
    for (v, &r) in rank.iter().enumerate() {
        if r >= n || node_at[r] != usize::MAX {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rank must be a permutation",
            ));
        }
        node_at[r] = v;
    }
    let sources = positions_of(sources, n)?;
    let targets = positions_of(targets, n)?;
    let mut a = a;
    let mut b = if directed { b } else { Vec::new() };

    // dq_dict
    let mut dq: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    for ((&u, &v), &w) in sources.iter().zip(&targets).zip(&weights) {
        if u == v {
            continue;
        }
        *dq[u].entry(v).or_insert(0.0) += w;
        *dq[v].entry(u).or_insert(0.0) += w;
    }
    for u in 0..n {
        let keys: Vec<usize> = dq[u].keys().copied().collect();
        for v in keys {
            let (bu, bv) = if directed { (b[u], b[v]) } else { (a[u], a[v]) };
            let wt = dq[u][&v];
            dq[u].insert(v, q0 * wt - resolution * (a[u] * bv + bu * a[v]));
        }
    }
    // Row heaps: (priority = -dq, rank of col); H: (priority, rank of row, rank of col)
    let mut rows: Vec<BTreeSet<(Priority, usize)>> = vec![BTreeSet::new(); n];
    for u in 0..n {
        for (&v, &d) in &dq[u] {
            rows[u].insert((Priority::new(-d), rank[v]));
        }
    }
    let mut heap: BTreeSet<(Priority, usize, usize)> = BTreeSet::new();
    let mut in_heap: Vec<Option<(Priority, usize)>> = vec![None; n];
    let top = |rows: &Vec<BTreeSet<(Priority, usize)>>, u: usize| rows[u].first().copied();
    macro_rules! heap_push {
        ($row:expr, $entry:expr) => {{
            let row: usize = $row;
            if in_heap[row].is_none() {
                let (p, c): (Priority, usize) = $entry;
                heap.insert((p, rank[row], c));
                in_heap[row] = Some((p, c));
            }
        }};
    }
    macro_rules! heap_remove {
        ($row:expr) => {{
            let row: usize = $row;
            if let Some((p, c)) = in_heap[row].take() {
                heap.remove(&(p, rank[row], c));
            }
        }};
    }
    for u in 0..n {
        if let Some(t) = top(&rows, u) {
            heap_push!(u, t);
        }
    }

    let mut merges: Vec<(usize, usize)> = Vec::new();
    let mut count = n;
    let mut exhausted = false;
    while count > cutoff {
        if heap.len() <= 1 {
            exhausted = true;
            break;
        }
        let Some((p, ru, rv)) = heap.pop_first() else {
            exhausted = true;
            break;
        };
        let (u, v) = (node_at[ru], node_at[rv]);
        in_heap[u] = None;
        let gain = -p.0;
        if gain < 0.0 && count <= best_n {
            break;
        }
        // dq_heap[u].pop(): (u, v) is row u's top
        rows[u].pop_first();
        if let Some(t) = top(&rows, u) {
            heap_push!(u, t);
        }
        let vu = (Priority::new(-dq[v][&u]), rank[u]);
        if top(&rows, v) == Some(vu) {
            heap_remove!(v);
            rows[v].remove(&vu);
            if let Some(t) = top(&rows, v) {
                heap_push!(v, t);
            }
        } else {
            rows[v].remove(&vu);
        }
        merges.push((u, v));
        count -= 1;

        let u_nbrs: Vec<usize> = dq[u].keys().copied().collect();
        let v_has: FxHashMap<usize, ()> = dq[v].keys().map(|&k| (k, ())).collect();
        let mut all: Vec<usize> = u_nbrs
            .iter()
            .copied()
            .chain(dq[v].keys().copied())
            .filter(|&w| w != u && w != v)
            .collect();
        all.sort_unstable();
        all.dedup();
        let (bu, bv) = if directed { (b[u], b[v]) } else { (a[u], a[v]) };
        for w in all {
            let in_v = v_has.contains_key(&w);
            let in_u = dq[u].contains_key(&w);
            let bw = if directed { b[w] } else { a[w] };
            let dq_vw = if in_u && in_v {
                dq[v][&w] + dq[u][&w]
            } else if in_v {
                dq[v][&w] - resolution * (a[u] * bw + a[w] * bu)
            } else {
                dq[u][&w] - resolution * (a[v] * bw + a[w] * bv)
            };
            for (row, col) in [(v, w), (w, v)] {
                let old = dq[row].insert(col, dq_vw);
                let old_top = top(&rows, row);
                let entry = (Priority::new(-dq_vw), rank[col]);
                if in_v {
                    if let Some(old) = old {
                        rows[row].remove(&(Priority::new(-old), rank[col]));
                    }
                }
                rows[row].insert(entry);
                match old_top {
                    None => heap_push!(row, entry),
                    Some(old_top) => {
                        let new_top = top(&rows, row).expect("row is not empty");
                        if old_top != new_top {
                            heap_remove!(row);
                            heap_push!(row, new_top);
                        }
                    }
                }
            }
        }
        for w in u_nbrs {
            let Some(old) = dq[w].remove(&u) else {
                continue;
            };
            if w == v {
                continue;
            }
            let wu = (Priority::new(-old), rank[u]);
            let uw = (Priority::new(-dq[u][&w]), rank[w]);
            for (row, entry) in [(w, wu), (u, uw)] {
                if top(&rows, row) == Some(entry) {
                    rows[row].remove(&entry);
                    heap_remove!(row);
                    if let Some(t) = top(&rows, row) {
                        heap_push!(row, t);
                    }
                } else {
                    rows[row].remove(&entry);
                }
            }
        }
        dq[u].clear();
        rows[u].clear();
        heap_remove!(u);
        a[v] += a[u];
        a[u] = 0.0;
        if directed {
            b[v] += b[u];
            b[u] = 0.0;
        }
    }
    Ok((merges, exhausted))
}

// ============================================================================
// Efficiency
// ============================================================================

/// Distinct neighbors of each node (positions), in the order of the graph's neighbor lists
fn distinct_neighbors(
    graph: &Graph,
    ids: &[u64],
    positions: &FxHashMap<u64, usize>,
) -> Vec<Vec<usize>> {
    let lists = graph.get_adjacency_list();
    let n = ids.len();
    let mut owner = vec![usize::MAX; n];
    ids.iter()
        .enumerate()
        .map(|(v, id)| {
            let mut out = Vec::new();
            if let Some(nbrs) = lists.get(id) {
                for w in nbrs {
                    if let Some(&t) = positions.get(w) {
                        if owner[t] != v {
                            owner[t] = v;
                            out.push(t);
                        }
                    }
                }
            }
            out
        })
        .collect()
}

/// The sum `networkx.global_efficiency` accumulates over a graph given by its nodes in
/// iteration order and a neighbor function: `1 / d` for every reachable pair at distance
/// `d > 0`, added one by one, by source and then in breadth-first discovery order.
fn efficiency_sum(
    order: &[usize],
    neighbors: impl Fn(usize, &mut Vec<usize>),
    dist: &mut [usize],
) -> f64 {
    const UNSEEN: usize = usize::MAX;
    let mut total = 0.0f64;
    let mut level: Vec<usize> = Vec::new();
    let mut next: Vec<usize> = Vec::new();
    let mut seen: Vec<usize> = Vec::new();
    let mut nbrs: Vec<usize> = Vec::new();
    for &s in order {
        dist[s] = 0;
        seen.push(s);
        level.clear();
        level.push(s);
        let mut d = 0usize;
        while !level.is_empty() {
            d += 1;
            next.clear();
            for &v in &level {
                neighbors(v, &mut nbrs);
                for &w in &nbrs {
                    if dist[w] == UNSEEN {
                        dist[w] = d;
                        seen.push(w);
                        next.push(w);
                        total += 1.0 / d as f64;
                    }
                }
            }
            std::mem::swap(&mut level, &mut next);
        }
        for &v in &seen {
            dist[v] = UNSEEN;
        }
        seen.clear();
    }
    total
}

/// `networkx.global_efficiency(G)` before its final division: the sum of `1 / d` over every
/// ordered pair at distance `d > 0`, added in NetworkX's order (sources in node order,
/// targets in breadth-first discovery order over the graph's neighbor lists)
#[pyfunction]
pub fn nx_global_efficiency_sum(graph: &Graph) -> f64 {
    let (ids, positions) = node_positions(graph);
    let n = ids.len();
    let nbrs = distinct_neighbors(graph, &ids, &positions);
    let order: Vec<usize> = (0..n).collect();
    let mut dist = vec![usize::MAX; n];
    efficiency_sum(
        &order,
        |v, out: &mut Vec<usize>| {
            out.clear();
            out.extend_from_slice(&nbrs[v]);
        },
        &mut dist,
    )
}

/// The sums `networkx.local_efficiency` takes `global_efficiency` of: for each node `v` (in
/// node order), the graph its neighbors induce, as NetworkX's subgraph view iterates it.
///
/// `members` lists each subgraph's nodes (as node IDs) in the order the view iterates them
/// (Python's set order, which the caller reads from the same set NetworkX builds, or node
/// order), split by `offsets`. A node's neighbors inside the subgraph come in its own
/// neighbor order. Returns each subgraph's efficiency sum.
#[pyfunction]
pub fn nx_local_efficiency_sums(
    graph: &Graph,
    offsets: Vec<usize>,
    members: Vec<u64>,
) -> PyResult<Vec<f64>> {
    let (ids, positions) = node_positions(graph);
    let n = ids.len();
    if offsets.len() != n + 1 || offsets.last().copied() != Some(members.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "offsets must split members into one list per node",
        ));
    }
    let members: Vec<usize> = members
        .iter()
        .map(|id| {
            positions.get(id).copied().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("unknown node ID {}", id))
            })
        })
        .collect::<PyResult<_>>()?;
    let nbrs = distinct_neighbors(graph, &ids, &positions);
    let mut mark = vec![usize::MAX; n]; // node -> subgraph (of node v) it belongs to
    let mut dist = vec![usize::MAX; n];
    let mut out = Vec::with_capacity(n);
    for v in 0..n {
        let (a, b) = (offsets[v], offsets[v + 1]);
        if a > b || b > members.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("bad offsets"));
        }
        let sub = &members[a..b];
        for &u in sub {
            mark[u] = v;
        }
        let total = efficiency_sum(
            sub,
            |u, list: &mut Vec<usize>| {
                list.clear();
                list.extend(nbrs[u].iter().copied().filter(|&w| mark[w] == v));
            },
            &mut dist,
        );
        for &u in sub {
            mark[u] = usize::MAX;
        }
        out.push(total);
    }
    Ok(out)
}
