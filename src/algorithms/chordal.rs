//! Chordal graphs (maximum cardinality search, clique trees, minimal triangulation by MCS-M,
//! induced nodes), perfect graph recognition, all cliques enumeration and maximum-weight
//! cliques.
//!
//! All functions work on the underlying simple undirected graph: parallel edges count once and
//! self-loops are ignored. Ties are broken by node ID, which reproduces NetworkX's results for
//! graphs whose nodes were added in ID order.

use crate::graph::{Edge, Graph, Node};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeSet, VecDeque};

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

struct Simple {
    ids: Vec<u64>,
    idx: FxHashMap<u64, usize>,
    adj: Vec<BTreeSet<usize>>,
}

impl Simple {
    fn new(graph: &Graph) -> PyResult<Self> {
        if graph.is_directed() {
            return value_err("this method requires an undirected graph");
        }
        let mut ids = graph.get_node_ids();
        ids.sort_unstable();
        let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut adj = vec![BTreeSet::new(); ids.len()];
        for e in graph.get_edges_internal() {
            let (a, b) = (idx[&e.source], idx[&e.target]);
            if a != b {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        Ok(Simple { ids, idx, adj })
    }
    fn node(&self, id: u64) -> PyResult<usize> {
        match self.idx.get(&id) {
            Some(&i) => Ok(i),
            None => value_err(format!("Node {id} does not exist")),
        }
    }
}

/// Maximum cardinality search from `start` over `nodes` (all nodes when None). Returns the
/// first triple `(u, v, w)` where `v`'s numbered neighbors are not a clique (`u`-`w` missing),
/// or None if the visit order is a perfect elimination order (reversed).
fn chordality_breaker(
    adj: &[BTreeSet<usize>],
    nodes: &[usize],
    start: usize,
) -> Option<(usize, usize, usize)> {
    let mut numbered: FxHashSet<usize> = FxHashSet::default();
    numbered.insert(start);
    let mut count: FxHashMap<usize, usize> = nodes.iter().map(|&v| (v, 0)).collect();
    for &w in &adj[start] {
        if let Some(c) = count.get_mut(&w) {
            *c += 1;
        }
    }
    let mut remaining: BTreeSet<usize> = nodes.iter().copied().filter(|&v| v != start).collect();
    while !remaining.is_empty() {
        let v = *remaining
            .iter()
            .max_by_key(|&&v| (count[&v], std::cmp::Reverse(v)))
            .unwrap();
        remaining.remove(&v);
        let clique: Vec<usize> = adj[v]
            .iter()
            .copied()
            .filter(|w| numbered.contains(w))
            .collect();
        for (i, &a) in clique.iter().enumerate() {
            for &b in &clique[i + 1..] {
                if !adj[a].contains(&b) {
                    return Some((a, v, b));
                }
            }
        }
        numbered.insert(v);
        for &w in &adj[v] {
            if let Some(c) = count.get_mut(&w) {
                *c += 1;
            }
        }
    }
    None
}

fn components(adj: &[BTreeSet<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for s in 0..n {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut comp = vec![s];
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            for &w in &adj[u] {
                if !seen[w] {
                    seen[w] = true;
                    comp.push(w);
                    q.push_back(w);
                }
            }
        }
        comp.sort_unstable();
        out.push(comp);
    }
    out
}

/// True if every cycle of four or more nodes has a chord.
#[pyfunction]
pub fn is_chordal(graph: &Graph) -> PyResult<bool> {
    let g = Simple::new(graph)?;
    if g.ids.len() <= 3 {
        return Ok(true);
    }
    let all: Vec<usize> = (0..g.ids.len()).collect();
    // MCS restarted per component is equivalent to the NetworkX single search with zero counts
    Ok(chordality_breaker(&g.adj, &all, 0).is_none())
}

/// True if the graph on `0..n` with bitset adjacency `bits` has an odd hole: an induced cycle
/// of odd length five or more. Grows induced paths from each node `s` through larger nodes only,
/// so every hole is found from its smallest node; `block[x]` counts the path's interior nodes
/// adjacent to `x`, which would be chords.
fn has_odd_hole(bits: &[Vec<u64>]) -> bool {
    let n = bits.len();
    let adjacent = |u: usize, v: usize| bits[u][v / 64] >> (v % 64) & 1 == 1;
    let nbrs: Vec<Vec<usize>> = (0..n)
        .map(|u| (0..n).filter(|&v| adjacent(u, v)).collect())
        .collect();
    let mut block = vec![0u32; n];
    let mut on_path = vec![false; n];
    for s in 0..n {
        for &p1 in nbrs[s].iter().filter(|&&p| p > s) {
            let mut path = vec![p1];
            let mut it = vec![0usize];
            on_path[p1] = true;
            while let Some(&last) = path.last() {
                let i = it[it.len() - 1];
                if i == nbrs[last].len() {
                    path.pop();
                    it.pop();
                    on_path[last] = false;
                    if let Some(&prev) = path.last() {
                        for &y in &nbrs[prev] {
                            block[y] -= 1;
                        }
                    }
                    continue;
                }
                *it.last_mut().unwrap() += 1;
                let x = nbrs[last][i];
                if x <= s || on_path[x] || block[x] > 0 {
                    continue;
                }
                if adjacent(x, s) {
                    // Closes the hole s, p1, ..., last, x (a triangle if the path is just p1)
                    let len = path.len() + 2;
                    if len >= 5 && len % 2 == 1 {
                        return true;
                    }
                    continue;
                }
                for &y in &nbrs[last] {
                    block[y] += 1;
                }
                path.push(x);
                it.push(0);
                on_path[x] = true;
            }
        }
    }
    false
}

fn is_bipartite_adj(adj: &[BTreeSet<usize>]) -> bool {
    let mut side = vec![u8::MAX; adj.len()];
    for s in 0..adj.len() {
        if side[s] != u8::MAX {
            continue;
        }
        side[s] = 0;
        let mut queue = VecDeque::from([s]);
        while let Some(u) = queue.pop_front() {
            for &v in &adj[u] {
                if side[v] == u8::MAX {
                    side[v] = 1 - side[u];
                    queue.push_back(v);
                } else if side[v] == side[u] {
                    return false;
                }
            }
        }
    }
    true
}

/// True if the graph is perfect: by the Strong Perfect Graph Theorem (Chudnovsky, Robertson,
/// Seymour & Thomas 2006), if neither it nor its complement has an odd hole. Chordal and
/// bipartite graphs are accepted at once; otherwise holes are searched for directly, which takes
/// exponential time in the worst case (as in NetworkX).
#[pyfunction]
pub fn is_perfect_graph(graph: &Graph) -> PyResult<bool> {
    let g = Simple::new(graph)?;
    let n = g.ids.len();
    if n < 5 {
        return Ok(true);
    }
    let all: Vec<usize> = (0..n).collect();
    if chordality_breaker(&g.adj, &all, 0).is_none() || is_bipartite_adj(&g.adj) {
        return Ok(true);
    }
    let words = n / 64 + 1;
    let mut bits = vec![vec![0u64; words]; n];
    for (u, nb) in g.adj.iter().enumerate() {
        for &v in nb {
            bits[u][v / 64] |= 1 << (v % 64);
        }
    }
    if has_odd_hole(&bits) {
        return Ok(false);
    }
    let mut co = vec![vec![0u64; words]; n];
    for u in 0..n {
        for v in 0..n {
            if u != v && bits[u][v / 64] >> (v % 64) & 1 == 0 {
                co[u][v / 64] |= 1 << (v % 64);
            }
        }
    }
    Ok(!has_odd_hole(&co))
}

/// Maximal cliques of a chordal graph, from its maximum cardinality search (as in NetworkX).
#[pyfunction]
pub fn chordal_graph_cliques(graph: &Graph) -> PyResult<Vec<Vec<u64>>> {
    let g = Simple::new(graph)?;
    let all: Vec<usize> = (0..g.ids.len()).collect();
    if !all.is_empty() && chordality_breaker(&g.adj, &all, 0).is_some() {
        return value_err("the graph is not chordal");
    }
    let mut out = Vec::new();
    for comp in components(&g.adj) {
        if comp.len() == 1 {
            out.push(vec![g.ids[comp[0]]]);
            continue;
        }
        let mut numbered: FxHashSet<usize> = FxHashSet::default();
        let first = comp[0];
        numbered.insert(first);
        let mut count: FxHashMap<usize, usize> = comp.iter().map(|&v| (v, 0)).collect();
        for &w in &g.adj[first] {
            *count.get_mut(&w).unwrap() += 1;
        }
        let mut remaining: BTreeSet<usize> = comp.iter().copied().filter(|&v| v != first).collect();
        let mut current: BTreeSet<usize> = BTreeSet::from([first]);
        while !remaining.is_empty() {
            let v = *remaining
                .iter()
                .max_by_key(|&&v| (count[&v], std::cmp::Reverse(v)))
                .unwrap();
            remaining.remove(&v);
            let mut next: BTreeSet<usize> = g.adj[v]
                .iter()
                .copied()
                .filter(|w| numbered.contains(w))
                .collect();
            next.insert(v);
            if !next.is_superset(&current) {
                out.push(current.iter().map(|&x| g.ids[x]).collect());
            }
            current = next;
            numbered.insert(v);
            for &w in &g.adj[v] {
                *count.get_mut(&w).unwrap() += 1;
            }
        }
        out.push(current.iter().map(|&x| g.ids[x]).collect());
    }
    Ok(out)
}

/// Minimal triangulation by MCS-M (Berry, Blair, Heggernes and Peyton): the fill edges to add,
/// and the elimination numbering `alpha` (n down to 1, in visit order).
#[pyfunction]
pub fn complete_to_chordal(graph: &Graph) -> PyResult<(Vec<(u64, u64)>, Vec<(u64, usize)>)> {
    let g = Simple::new(graph)?;
    let n = g.ids.len();
    let mut weight = vec![0usize; n];
    let mut unnumbered: Vec<usize> = (0..n).collect();
    let mut alpha = vec![0usize; n];
    let mut chords: BTreeSet<(usize, usize)> = BTreeSet::new();
    for i in (1..=n).rev() {
        // first node of maximal weight in the remaining list
        let (pos, &z) = unnumbered
            .iter()
            .enumerate()
            .max_by_key(|(k, &v)| (weight[v], std::cmp::Reverse(*k)))
            .unwrap();
        unnumbered.remove(pos);
        alpha[z] = i;
        // bottleneck search: least possible maximum weight of internal nodes on a z-y path
        // through unnumbered nodes
        let mut best = vec![usize::MAX; n];
        let mut heap = std::collections::BinaryHeap::new();
        let open: FxHashSet<usize> = unnumbered.iter().copied().collect();
        for &y in &g.adj[z] {
            if open.contains(&y) && best[y] != 0 {
                best[y] = 0; // direct neighbor: no internal node
                heap.push(std::cmp::Reverse((0usize, y)));
            }
        }
        while let Some(std::cmp::Reverse((b, u))) = heap.pop() {
            if b > best[u] {
                continue;
            }
            // u may serve as an internal node for its neighbors
            let through = b.max(weight[u]);
            for &y in &g.adj[u] {
                if open.contains(&y) && through < best[y] {
                    best[y] = through;
                    heap.push(std::cmp::Reverse((through, y)));
                }
            }
        }
        let mut update = Vec::new();
        for &y in &unnumbered {
            if g.adj[z].contains(&y) {
                update.push(y);
            } else if best[y] != usize::MAX && best[y] < weight[y] {
                update.push(y);
                chords.insert((z.min(y), z.max(y)));
            }
        }
        for y in update {
            weight[y] += 1;
        }
    }
    Ok((
        chords
            .into_iter()
            .map(|(a, b)| (g.ids[a], g.ids[b]))
            .collect(),
        (0..n).map(|v| (g.ids[v], alpha[v])).collect(),
    ))
}

/// NetworkX's `find_induced_nodes`: the nodes of the induced paths between `s` and `t` in a
/// chordal graph (empty when `s` and `t` are adjacent or no such path exists).
#[pyfunction]
pub fn find_induced_nodes(graph: &Graph, s: u64, t: u64) -> PyResult<Vec<u64>> {
    let g = Simple::new(graph)?;
    let (si, ti) = (g.node(s)?, g.node(t)?);
    let all: Vec<usize> = (0..g.ids.len()).collect();
    if g.ids.len() > 3 && chordality_breaker(&g.adj, &all, 0).is_some() {
        return value_err("the graph is not chordal");
    }
    let mut h = g.adj.clone();
    h[si].insert(ti);
    h[ti].insert(si);
    let mut induced: BTreeSet<usize> = BTreeSet::new();
    while let Some((u, v, w)) = chordality_breaker(&h, &all, si) {
        for x in [u, v, w] {
            induced.insert(x);
            if x != si {
                h[si].insert(x);
                h[x].insert(si);
            }
        }
    }
    if !induced.is_empty() {
        induced.insert(ti);
        for &u in &g.adj[si] {
            if g.adj[u].iter().filter(|x| induced.contains(x)).count() == 2 {
                induced.insert(u);
                break;
            }
        }
    }
    Ok(induced.into_iter().map(|v| g.ids[v]).collect())
}

/// Every clique (not only maximal ones), in order of size, as in NetworkX's
/// `enumerate_all_cliques`.
#[pyfunction]
pub fn enumerate_all_cliques(graph: &Graph) -> PyResult<Vec<Vec<u64>>> {
    let g = Simple::new(graph)?;
    let n = g.ids.len();
    let later: Vec<Vec<usize>> = (0..n)
        .map(|u| g.adj[u].iter().copied().filter(|&v| v > u).collect())
        .collect();
    let mut queue: VecDeque<(Vec<usize>, Vec<usize>)> =
        (0..n).map(|u| (vec![u], later[u].clone())).collect();
    let mut out = Vec::new();
    while let Some((base, cands)) = queue.pop_front() {
        out.push(base.iter().map(|&v| g.ids[v]).collect());
        for (i, &u) in cands.iter().enumerate() {
            let mut next = base.clone();
            next.push(u);
            let rest: Vec<usize> = cands[i + 1..]
                .iter()
                .copied()
                .filter(|v| g.adj[u].contains(v))
                .collect();
            queue.push_back((next, rest));
        }
    }
    Ok(out)
}

/// Maximum-weight clique by branch and bound with a weighted coloring bound. Node weights come
/// from the node attribute `weight_attr` (1 for every node when None). Returns
/// `(node IDs, total weight)`.
#[pyfunction]
#[pyo3(signature = (graph, weight_attr=None))]
pub fn max_weight_clique(
    py: Python,
    graph: &Graph,
    weight_attr: Option<&str>,
) -> PyResult<(Vec<u64>, f64)> {
    let g = Simple::new(graph)?;
    let n = g.ids.len();
    let nodes = graph.get_nodes();
    let mut w = vec![1.0f64; n];
    if let Some(attr) = weight_attr {
        for (i, id) in g.ids.iter().enumerate() {
            let value = match &nodes[id].attributes {
                Some(d) => d
                    .bind(py)
                    .downcast::<PyDict>()?
                    .get_item(attr)?
                    .map(|v| v.extract::<f64>())
                    .transpose()?,
                None => None,
            };
            match value {
                Some(x) if x >= 0.0 && x.is_finite() => w[i] = x,
                Some(_) => return value_err("node weights must be non-negative numbers"),
                None => return value_err(format!("node {id} has no attribute {attr:?}")),
            }
        }
    }
    let mut best: (f64, Vec<usize>) = (0.0, Vec::new());
    fn expand(
        adj: &[BTreeSet<usize>],
        w: &[f64],
        current: &mut Vec<usize>,
        cur_w: f64,
        cands: Vec<usize>,
        best: &mut (f64, Vec<usize>),
    ) {
        if cur_w > best.0 {
            *best = (cur_w, current.clone());
        }
        if cands.is_empty() {
            return;
        }
        // greedy coloring: each color class is an independent set, so a clique takes at most
        // one node per class; the bound sums the class maxima
        let mut order = cands.clone();
        order.sort_by(|&a, &b| w[b].total_cmp(&w[a]).then(a.cmp(&b)));
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for &v in &order {
            match classes
                .iter_mut()
                .find(|c| c.iter().all(|&u| !adj[u].contains(&v)))
            {
                Some(c) => c.push(v),
                None => classes.push(vec![v]),
            }
        }
        let mut seq: Vec<(usize, f64)> = Vec::new(); // (node, bound of classes up to its own)
        let mut acc = 0.0;
        for class in &classes {
            acc += class.iter().map(|&v| w[v]).fold(0.0, f64::max);
            for &v in class {
                seq.push((v, acc));
            }
        }
        let mut remaining: FxHashSet<usize> = cands.iter().copied().collect();
        for &(v, bound) in seq.iter().rev() {
            if cur_w + bound <= best.0 {
                return;
            }
            remaining.remove(&v);
            let next: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|u| adj[v].contains(u))
                .collect();
            current.push(v);
            expand(adj, w, current, cur_w + w[v], next, best);
            current.pop();
        }
    }
    let mut current = Vec::new();
    expand(&g.adj, &w, &mut current, 0.0, (0..n).collect(), &mut best);
    let mut nodes_out: Vec<u64> = best.1.iter().map(|&v| g.ids[v]).collect();
    nodes_out.sort_unstable();
    Ok((nodes_out, best.0))
}

/// The graph with fill edges added (a copy of every node and edge, plus the chords).
#[pyfunction]
pub fn with_extra_edges(py: Python, graph: &Graph, edges: Vec<(u64, u64)>) -> PyResult<Graph> {
    let mut result = Graph::new_rust(graph.is_directed(), graph.is_weighted());
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
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
    let mut all: Vec<&Edge> = graph.get_edges_internal().iter().collect();
    all.sort_unstable_by_key(|e| e.id);
    for e in all {
        let attributes = match &e.attributes {
            Some(d) => Some(d.bind(py).copy()?.unbind()),
            None => None,
        };
        result.add_edge_rust(Edge {
            id: 0,
            source: e.source,
            target: e.target,
            weight: e.weight,
            attributes,
        })?;
    }
    for (a, b) in edges {
        let weight = if graph.is_weighted() { Some(1.0) } else { None };
        result.add_edge_rust(Edge {
            id: 0,
            source: a,
            target: b,
            weight,
            attributes: None,
        })?;
    }
    Ok(result)
}
