//! Shortest-path extras: predecessor lists with ties (BFS, Dijkstra or Bellman-Ford), single-
//! target searches, Floyd-Warshall and Johnson all-pairs paths with negative weights, A* with a
//! Python heuristic, Yen's lazy enumeration of shortest simple paths, and simple paths as edge
//! sequences.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, VecDeque};

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

/// Indexed arcs: `out[u]` / `inn[u]` hold `(neighbor, weight, edge id)`; undirected edges are
/// stored both ways.
pub(crate) struct Net {
    ids: Vec<u64>,
    idx: FxHashMap<u64, usize>,
    out: Vec<Vec<(usize, f64, u64)>>,
    inn: Vec<Vec<(usize, f64, u64)>>,
    directed: bool,
    negative: bool,
}

impl Net {
    fn new(graph: &Graph, weighted: bool) -> PyResult<Self> {
        let mut ids = graph.get_node_ids();
        ids.sort_unstable();
        let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n = ids.len();
        let directed = graph.is_directed();
        let mut out = vec![Vec::new(); n];
        let mut inn = vec![Vec::new(); n];
        let mut negative = false;
        let mut edges: Vec<_> = graph.get_edges_internal().iter().collect();
        edges.sort_unstable_by_key(|e| e.id);
        for e in edges {
            let w = if weighted {
                e.weight.unwrap_or(1.0)
            } else {
                1.0
            };
            if !w.is_finite() {
                return value_err("edge weights must be finite");
            }
            negative |= w < 0.0;
            let (a, b) = (idx[&e.source], idx[&e.target]);
            out[a].push((b, w, e.id));
            inn[b].push((a, w, e.id));
            if !directed && a != b {
                out[b].push((a, w, e.id));
                inn[a].push((b, w, e.id));
            }
        }
        if negative && !directed {
            return value_err("an undirected graph with a negative edge has a negative cycle");
        }
        Ok(Net {
            ids,
            idx,
            out,
            inn,
            directed,
            negative,
        })
    }
    fn node(&self, id: u64) -> PyResult<usize> {
        match self.idx.get(&id) {
            Some(&i) => Ok(i),
            None => value_err(format!("Node {id} does not exist")),
        }
    }
    fn arcs(&self, reverse: bool) -> &Vec<Vec<(usize, f64, u64)>> {
        if reverse {
            &self.inn
        } else {
            &self.out
        }
    }
}

#[derive(PartialEq)]
struct Item(f64, usize);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
            .then(o.1.cmp(&self.1))
    }
}

/// Distances and every equal-cost predecessor from `sources` (along `reverse` arcs for
/// single-target searches). BFS when unweighted, Dijkstra for non-negative weights, and
/// Bellman-Ford with negative weights. `cutoff` bounds the distance.
fn shortest_preds(
    net: &Net,
    sources: &[usize],
    weighted: bool,
    reverse: bool,
    cutoff: Option<f64>,
) -> PyResult<(Vec<f64>, Vec<Vec<usize>>)> {
    let n = net.ids.len();
    let arcs = net.arcs(reverse);
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    let within = |d: f64| cutoff.map_or(true, |c| d <= c);
    for &s in sources {
        dist[s] = 0.0;
    }
    if !weighted {
        let mut q: VecDeque<usize> = sources.iter().copied().collect();
        while let Some(u) = q.pop_front() {
            for &(v, _, _) in &arcs[u] {
                let d = dist[u] + 1.0;
                if !within(d) {
                    continue;
                }
                if dist[v].is_infinite() {
                    dist[v] = d;
                    q.push_back(v);
                }
                if dist[v] == d && !pred[v].contains(&u) {
                    pred[v].push(u);
                }
            }
        }
    } else if !net.negative {
        let mut heap: BinaryHeap<Item> = sources.iter().map(|&s| Item(0.0, s)).collect();
        let mut done = vec![false; n];
        while let Some(Item(d, u)) = heap.pop() {
            if done[u] || d > dist[u] {
                continue;
            }
            done[u] = true;
            for &(v, w, _) in &arcs[u] {
                let nd = d + w;
                if !within(nd) || done[v] && nd > dist[v] {
                    continue;
                }
                if nd < dist[v] {
                    dist[v] = nd;
                    pred[v].clear();
                    pred[v].push(u);
                    heap.push(Item(nd, v));
                } else if nd == dist[v] && !pred[v].contains(&u) && v != u {
                    pred[v].push(u);
                }
            }
        }
    } else {
        for round in 0..=n {
            let mut changed = false;
            for u in 0..n {
                if dist[u].is_infinite() {
                    continue;
                }
                for &(v, w, _) in &arcs[u] {
                    let nd = dist[u] + w;
                    if nd < dist[v] {
                        dist[v] = nd;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
            if round == n {
                return value_err("the graph has a negative cycle reachable from the source");
            }
        }
        for u in 0..n {
            if dist[u].is_infinite() {
                continue;
            }
            for &(v, w, _) in &arcs[u] {
                if dist[u] + w == dist[v] && !sources.contains(&v) && !pred[v].contains(&u) {
                    pred[v].push(u);
                }
            }
        }
    }
    if net.negative && weighted {
        for v in 0..n {
            if !within(dist[v]) {
                dist[v] = f64::INFINITY;
            }
        }
        for v in 0..n {
            let keep: Vec<usize> = pred[v]
                .iter()
                .copied()
                .filter(|&u| dist[u].is_finite())
                .collect();
            pred[v] = keep;
        }
    }
    for &s in sources {
        pred[s].clear();
    }
    Ok((dist, pred))
}

/// Predecessor lists and distances: `(node, predecessors)` and `(node, distance)` pairs for
/// every node reached from `sources` (or reaching them, with `reverse`).
#[pyfunction]
#[pyo3(signature = (graph, sources, weighted=false, reverse=false, cutoff=None))]
pub fn shortest_path_predecessors(
    graph: &Graph,
    sources: Vec<u64>,
    weighted: bool,
    reverse: bool,
    cutoff: Option<f64>,
) -> PyResult<(Vec<(u64, Vec<u64>)>, Vec<(u64, f64)>)> {
    let net = Net::new(graph, weighted)?;
    let srcs: Vec<usize> = sources
        .iter()
        .map(|&s| net.node(s))
        .collect::<PyResult<_>>()?;
    let (dist, pred) = shortest_preds(&net, &srcs, weighted || net.negative, reverse, cutoff)?;
    let reached: Vec<usize> = (0..net.ids.len())
        .filter(|&v| dist[v].is_finite())
        .collect();
    Ok((
        reached
            .iter()
            .map(|&v| (net.ids[v], pred[v].iter().map(|&p| net.ids[p]).collect()))
            .collect(),
        reached.iter().map(|&v| (net.ids[v], dist[v])).collect(),
    ))
}

/// Floyd-Warshall all-pairs distances (negative weights allowed): `(ids, dist, pred)` where
/// `pred[s][t]` is the node before `t` on a shortest `s`-`t` path (`-1` if none).
#[pyfunction]
#[pyo3(signature = (graph, weighted=true))]
pub fn floyd_warshall(
    graph: &Graph,
    weighted: bool,
) -> PyResult<(Vec<u64>, Vec<Vec<f64>>, Vec<Vec<i64>>)> {
    let net = Net::new(graph, weighted)?;
    let n = net.ids.len();
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    let mut pred = vec![vec![-1i64; n]; n];
    for u in 0..n {
        dist[u][u] = 0.0;
        for &(v, w, _) in &net.out[u] {
            if w < dist[u][v] && !(u == v && w >= 0.0) {
                dist[u][v] = w;
                pred[u][v] = u as i64;
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            if dist[i][k].is_infinite() {
                continue;
            }
            for j in 0..n {
                let d = dist[i][k] + dist[k][j];
                if d < dist[i][j] {
                    dist[i][j] = d;
                    pred[i][j] = pred[k][j];
                }
            }
        }
    }
    if (0..n).any(|i| dist[i][i] < 0.0) {
        return value_err("the graph has a negative cycle");
    }
    Ok((net.ids.clone(), dist, pred))
}

/// Johnson's all-pairs shortest paths (negative weights allowed): for every source, a shortest
/// path to every reachable node, as node ID lists.
#[pyfunction]
#[pyo3(signature = (graph, weighted=true))]
pub fn johnson(graph: &Graph, weighted: bool) -> PyResult<Vec<(u64, Vec<(u64, Vec<u64>)>)>> {
    let net = Net::new(graph, weighted)?;
    let n = net.ids.len();
    // potentials from a virtual source (Bellman-Ford)
    let mut h = vec![0.0f64; n];
    for round in 0..=n {
        let mut changed = false;
        for u in 0..n {
            for &(v, w, _) in &net.out[u] {
                if h[u] + w < h[v] {
                    h[v] = h[u] + w;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        if round == n {
            return value_err("the graph has a negative cycle");
        }
    }
    let mut result = Vec::with_capacity(n);
    for s in 0..n {
        let mut dist = vec![f64::INFINITY; n];
        let mut parent = vec![usize::MAX; n];
        let mut done = vec![false; n];
        dist[s] = 0.0;
        let mut heap = BinaryHeap::from([Item(0.0, s)]);
        while let Some(Item(d, u)) = heap.pop() {
            if done[u] {
                continue;
            }
            done[u] = true;
            for &(v, w, _) in &net.out[u] {
                let nd = d + (w + h[u] - h[v]).max(0.0);
                if nd < dist[v] {
                    dist[v] = nd;
                    parent[v] = u;
                    heap.push(Item(nd, v));
                }
            }
        }
        let paths = (0..n)
            .filter(|&t| dist[t].is_finite())
            .map(|t| {
                let mut p = vec![t];
                let mut x = t;
                while x != s {
                    x = parent[x];
                    p.push(x);
                }
                p.reverse();
                (net.ids[t], p.into_iter().map(|v| net.ids[v]).collect())
            })
            .collect();
        result.push((net.ids[s], paths));
    }
    Ok(result)
}

/// A* search with an optional heuristic `heuristic(node_id, target_id) -> float` (it must not
/// overestimate the remaining distance). Returns `(length, path)` or None.
#[pyfunction]
#[pyo3(signature = (graph, source, target, heuristic=None, weighted=true))]
pub fn astar(
    py: Python,
    graph: &Graph,
    source: u64,
    target: u64,
    heuristic: Option<PyObject>,
    weighted: bool,
) -> PyResult<Option<(f64, Vec<u64>)>> {
    let net = Net::new(graph, weighted)?;
    if net.negative {
        return value_err("A* requires non-negative edge weights");
    }
    let (s, t) = (net.node(source)?, net.node(target)?);
    let n = net.ids.len();
    let mut cache: FxHashMap<usize, f64> = FxHashMap::default();
    let mut estimate = |v: usize| -> PyResult<f64> {
        if let Some(&h) = cache.get(&v) {
            return Ok(h);
        }
        let h = match &heuristic {
            Some(f) => f.call1(py, (net.ids[v], target))?.extract::<f64>(py)?,
            None => 0.0,
        };
        cache.insert(v, h);
        Ok(h)
    };
    let mut g = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n];
    let mut closed = vec![false; n];
    g[s] = 0.0;
    let mut heap = BinaryHeap::from([Item(estimate(s)?, s)]);
    while let Some(Item(_, u)) = heap.pop() {
        if closed[u] {
            continue;
        }
        if u == t {
            let mut path = vec![t];
            let mut x = t;
            while x != s {
                x = parent[x];
                path.push(x);
            }
            path.reverse();
            return Ok(Some((g[t], path.into_iter().map(|v| net.ids[v]).collect())));
        }
        closed[u] = true;
        for &(v, w, _) in &net.out[u] {
            let nd = g[u] + w;
            if nd < g[v] {
                g[v] = nd;
                parent[v] = u;
                heap.push(Item(nd + estimate(v)?, v));
            }
        }
    }
    Ok(None)
}

/// Simple paths from `source` to `target` as sequences of edge IDs (parallel edges give
/// distinct paths), with at most `cutoff` edges.
#[pyfunction]
#[pyo3(signature = (graph, source, target, cutoff=None))]
pub fn simple_edge_paths(
    graph: &Graph,
    source: u64,
    target: u64,
    cutoff: Option<usize>,
) -> PyResult<Vec<Vec<u64>>> {
    let net = Net::new(graph, false)?;
    let (s, t) = (net.node(source)?, net.node(target)?);
    let limit = cutoff.unwrap_or(net.ids.len().saturating_sub(1));
    let mut out = Vec::new();
    if s == t || limit == 0 {
        return Ok(out);
    }
    let mut on_path = vec![false; net.ids.len()];
    on_path[s] = true;
    let mut edges: Vec<u64> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(s, 0)];
    while let Some(&mut (u, ref mut k)) = stack.last_mut() {
        if *k < net.out[u].len() && edges.len() < limit {
            let (v, _, e) = net.out[u][*k];
            *k += 1;
            if v == t {
                let mut p = edges.clone();
                p.push(e);
                out.push(p);
            } else if !on_path[v] {
                on_path[v] = true;
                edges.push(e);
                stack.push((v, 0));
            }
        } else {
            stack.pop();
            on_path[u] = false;
            edges.pop();
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Yen's algorithm, one path per call
// ---------------------------------------------------------------------------------------------

/// Lazy iterator over simple paths from source to target in order of increasing length.
#[pyclass(module = "edgecase._core")]
pub struct ShortestSimplePaths {
    ids: Vec<u64>,
    /// simple adjacency: cheapest arc weight per ordered pair
    adj: Vec<FxHashMap<usize, f64>>,
    source: usize,
    target: usize,
    found: Vec<(f64, Vec<usize>)>,
    candidates: BTreeSet<(OrdF64, Vec<usize>)>,
    seen: FxHashSet<Vec<usize>>,
    exhausted: bool,
}

#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal)
    }
}

impl ShortestSimplePaths {
    fn cost(&self, path: &[usize]) -> f64 {
        path.windows(2).map(|w| self.adj[w[0]][&w[1]]).sum()
    }
    fn spur(
        &self,
        from: usize,
        banned_nodes: &[bool],
        banned_arcs: &FxHashSet<(usize, usize)>,
    ) -> Option<(f64, Vec<usize>)> {
        let n = self.ids.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut parent = vec![usize::MAX; n];
        let mut done = vec![false; n];
        dist[from] = 0.0;
        let mut heap = BinaryHeap::from([Item(0.0, from)]);
        while let Some(Item(d, u)) = heap.pop() {
            if done[u] {
                continue;
            }
            done[u] = true;
            if u == self.target {
                break;
            }
            let mut nbrs: Vec<(&usize, &f64)> = self.adj[u].iter().collect();
            nbrs.sort_unstable_by_key(|x| *x.0);
            for (&v, &w) in nbrs {
                if banned_nodes[v] || banned_arcs.contains(&(u, v)) {
                    continue;
                }
                let nd = d + w;
                if nd < dist[v] {
                    dist[v] = nd;
                    parent[v] = u;
                    heap.push(Item(nd, v));
                }
            }
        }
        if dist[self.target].is_infinite() {
            return None;
        }
        let mut p = vec![self.target];
        let mut x = self.target;
        while x != from {
            x = parent[x];
            p.push(x);
        }
        p.reverse();
        Some((dist[self.target], p))
    }
}

#[pymethods]
impl ShortestSimplePaths {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Vec<u64>> {
        if self.exhausted {
            return None;
        }
        let n = self.ids.len();
        if self.found.is_empty() {
            match self.spur(self.source, &vec![false; n], &FxHashSet::default()) {
                Some(first) => {
                    self.seen.insert(first.1.clone());
                    self.found.push(first);
                }
                None => {
                    self.exhausted = true;
                    return None;
                }
            }
        } else {
            let last = self.found.last().unwrap().1.clone();
            for i in 0..last.len() - 1 {
                let root = &last[..=i];
                let spur_node = last[i];
                let mut banned_arcs: FxHashSet<(usize, usize)> = FxHashSet::default();
                for (_, p) in &self.found {
                    if p.len() > i + 1 && &p[..=i] == root {
                        banned_arcs.insert((p[i], p[i + 1]));
                    }
                }
                let mut banned_nodes = vec![false; n];
                for &v in &root[..i] {
                    banned_nodes[v] = true;
                }
                if let Some((_, spur_path)) = self.spur(spur_node, &banned_nodes, &banned_arcs) {
                    let mut path = root[..i].to_vec();
                    path.extend(spur_path);
                    if self.seen.insert(path.clone()) {
                        let c = self.cost(&path);
                        self.candidates.insert((OrdF64(c), path));
                    }
                }
            }
            match self.candidates.pop_first() {
                Some((c, p)) => self.found.push((c.0, p)),
                None => {
                    self.exhausted = true;
                    return None;
                }
            }
        }
        Some(
            self.found
                .last()
                .unwrap()
                .1
                .iter()
                .map(|&v| self.ids[v])
                .collect(),
        )
    }
}

/// Start a lazy enumeration of simple paths in increasing order of length (Yen's algorithm).
#[pyfunction]
#[pyo3(signature = (graph, source, target, weighted=true))]
pub fn shortest_simple_paths(
    graph: &Graph,
    source: u64,
    target: u64,
    weighted: bool,
) -> PyResult<ShortestSimplePaths> {
    let net = Net::new(graph, weighted)?;
    if net.negative {
        return value_err("shortest_simple_paths() requires non-negative edge weights");
    }
    let (s, t) = (net.node(source)?, net.node(target)?);
    if s == t {
        return value_err("source and target must be different nodes");
    }
    let n = net.ids.len();
    let mut adj: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    for u in 0..n {
        for &(v, w, _) in &net.out[u] {
            if u == v {
                continue;
            }
            let entry = adj[u].entry(v).or_insert(w);
            if w < *entry {
                *entry = w;
            }
        }
    }
    let _ = net.directed;
    Ok(ShortestSimplePaths {
        ids: net.ids,
        adj,
        source: s,
        target: t,
        found: Vec::new(),
        candidates: BTreeSet::new(),
        seen: FxHashSet::default(),
        exhausted: false,
    })
}

// ---------------------------------------------------------------------------------------------
// ALT (A*, landmarks, triangle inequality) heuristic
// ---------------------------------------------------------------------------------------------

/// Precomputed landmark distances: calling it with `(u, v)` node IDs gives a lower bound on
/// the distance from `u` to `v` (`0` for nodes it does not know).
#[pyclass(module = "edgecase._core")]
pub struct AltTable {
    idx: FxHashMap<u64, usize>,
    n: usize,
    landmarks: Vec<u64>,
    /// `from[l * n + v]`: distance from landmark `l` to `v`
    from: Vec<f64>,
    /// `to[l * n + v]`: distance from `v` to landmark `l` (None when undirected: same as `from`)
    to: Option<Vec<f64>>,
}

#[pymethods]
impl AltTable {
    fn __call__(&self, u: u64, v: u64) -> f64 {
        let (Some(&a), Some(&b)) = (self.idx.get(&u), self.idx.get(&v)) else {
            return 0.0;
        };
        let to = self.to.as_ref().unwrap_or(&self.from);
        let mut best = 0.0f64;
        for l in 0..self.landmarks.len() {
            let (ua, vb) = (to[l * self.n + a], to[l * self.n + b]);
            if ua.is_finite() && vb.is_finite() {
                best = best.max(ua - vb);
            }
            let (ua, vb) = (self.from[l * self.n + a], self.from[l * self.n + b]);
            if ua.is_finite() && vb.is_finite() {
                best = best.max(vb - ua);
            }
        }
        best
    }

    #[getter]
    fn landmarks(&self) -> Vec<u64> {
        self.landmarks.clone()
    }
}

/// Build an ALT table. The landmarks are `start`, extended to `k` by greedy farthest-point
/// selection (each new landmark is the node farthest from its nearest landmark, unreachable
/// nodes first; ties go to the smallest ID).
#[pyfunction]
#[pyo3(signature = (graph, start, k, weighted=true))]
pub fn alt_heuristic(
    graph: &Graph,
    start: Vec<u64>,
    k: usize,
    weighted: bool,
) -> PyResult<AltTable> {
    let net = Net::new(graph, weighted)?;
    if net.negative {
        return value_err("alt_heuristic() requires non-negative edge weights");
    }
    let n = net.ids.len();
    let sweep = |s: usize, reverse: bool| -> PyResult<Vec<f64>> {
        Ok(shortest_preds(&net, &[s], weighted, reverse, None)?.0)
    };
    let mut chosen: Vec<usize> = start
        .iter()
        .map(|&s| net.node(s))
        .collect::<PyResult<_>>()?;
    let mut from = Vec::with_capacity(k.max(chosen.len()) * n);
    let mut nearest = vec![f64::INFINITY; n];
    let mut is_chosen = vec![false; n];
    for &s in &chosen {
        let d = sweep(s, false)?;
        for v in 0..n {
            nearest[v] = nearest[v].min(d[v]);
        }
        is_chosen[s] = true;
        from.extend(d);
    }
    while chosen.len() < k.min(n) {
        let mut pick = usize::MAX;
        for v in (0..n).filter(|&v| !is_chosen[v]) {
            if pick == usize::MAX || nearest[v] > nearest[pick] {
                pick = v;
            }
        }
        let d = sweep(pick, false)?;
        for v in 0..n {
            nearest[v] = nearest[v].min(d[v]);
        }
        is_chosen[pick] = true;
        chosen.push(pick);
        from.extend(d);
    }
    let to = if net.directed {
        let mut to = Vec::with_capacity(chosen.len() * n);
        for &s in &chosen {
            to.extend(sweep(s, true)?);
        }
        Some(to)
    } else {
        None
    };
    Ok(AltTable {
        landmarks: chosen.iter().map(|&s| net.ids[s]).collect(),
        idx: net.idx,
        n,
        from,
        to,
    })
}
