//! Connectivity and structure extras: chain decompositions, dominator trees and dominance
//! frontiers, k-edge-connected components and subgraphs, average node connectivity,
//! small-world reference graphs and coefficients, negative cycle detection, and the
//! community-aware link prediction scores.

use crate::algorithms::approximation::Dinic;
use crate::graph::{Edge, Graph, Node};
use crate::utils::Seed;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

fn edges_by_id(graph: &Graph) -> Vec<&Edge> {
    let mut edges: Vec<&Edge> = graph.get_edges_internal().iter().collect();
    edges.sort_unstable_by_key(|e| e.id);
    edges
}

/// Simple undirected adjacency in first-seen (edge ID) order, like a NetworkX `Graph` built by
/// adding the edges in ID order. Self-loops are kept once when `loops`.
fn simple_adjacency(graph: &Graph, idx: &FxHashMap<u64, usize>, loops: bool) -> Vec<Vec<usize>> {
    let n = idx.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut seen: FxHashSet<(usize, usize)> = FxHashSet::default();
    for e in edges_by_id(graph) {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        if a == b && !loops {
            continue;
        }
        if !seen.insert((a.min(b), a.max(b))) {
            continue;
        }
        adj[a].push(b);
        if a != b {
            adj[b].push(a);
        }
    }
    adj
}

// ---------------------------------------------------------------------------------------------
// Chain decomposition
// ---------------------------------------------------------------------------------------------

/// Schmidt's chain decomposition, following NetworkX's DFS conventions. Returns chains as
/// lists of `(u, v)` node ID pairs.
#[pyfunction]
#[pyo3(signature = (graph, root=None))]
pub fn chain_decomposition(graph: &Graph, root: Option<u64>) -> PyResult<Vec<Vec<(u64, u64)>>> {
    if graph.is_directed() {
        return value_err("chain_decomposition() requires an undirected graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let adj = simple_adjacency(graph, &idx, true);
    let starts: Vec<usize> = match root {
        Some(r) => match idx.get(&r) {
            Some(&i) => vec![i],
            None => return value_err(format!("Root node {r} does not exist")),
        },
        None => (0..n).collect(),
    };
    // DFS cycle forest: parent pointers and nontree out-edges (ancestor -> descendant)
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut in_h = vec![false; n];
    let mut h_out: Vec<Vec<usize>> = vec![Vec::new(); n]; // nontree successors, in insertion order
    let mut h_has: FxHashSet<(usize, usize)> = FxHashSet::default(); // all H edges (tree + nontree)
    let mut order = Vec::new();
    let mut visited = vec![false; n];
    for &s in &starts {
        if visited[s] {
            continue;
        }
        visited[s] = true;
        in_h[s] = true;
        order.push(s);
        let mut stack: Vec<(usize, usize)> = vec![(s, 0)];
        while let Some(&mut (u, ref mut k)) = stack.last_mut() {
            if *k < adj[u].len() {
                let v = adj[u][*k];
                *k += 1;
                if visited[v] {
                    // nontree edge seen from u: H gets v -> u unless u -> v... mirrors NetworkX's
                    // `elif d == "nontree" and v not in H[u]`
                    if !h_has.contains(&(u, v)) {
                        h_has.insert((v, u));
                        h_out[v].push(u);
                    }
                } else {
                    visited[v] = true;
                    in_h[v] = true;
                    parent[v] = Some(u);
                    h_has.insert((v, u));
                    order.push(v);
                    stack.push((v, 0));
                }
            } else {
                stack.pop();
            }
        }
    }
    let mut seen = vec![false; n];
    let mut chains = Vec::new();
    for &u in &order {
        seen[u] = true;
        for &v0 in &h_out[u] {
            let mut chain = Vec::new();
            let (mut a, mut b) = (u, v0);
            while !seen[b] {
                chain.push((ids[a], ids[b]));
                seen[b] = true;
                a = b;
                b = parent[b].expect("chain reached a root");
            }
            chain.push((ids[a], ids[b]));
            chains.push(chain);
        }
    }
    let _ = in_h;
    Ok(chains)
}

// ---------------------------------------------------------------------------------------------
// Dominators
// ---------------------------------------------------------------------------------------------

/// Immediate dominators from `start` (Cooper, Harvey and Kennedy), as `(node, idom)` pairs for
/// every node reachable from `start` (the start maps to itself), plus dominance frontiers.
#[pyfunction]
pub fn dominators(graph: &Graph, start: u64) -> PyResult<(Vec<(u64, u64)>, Vec<(u64, Vec<u64>)>)> {
    if !graph.is_directed() {
        return value_err("dominators require a directed graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let Some(&s) = idx.get(&start) else {
        return value_err(format!("Start node {start} does not exist"));
    };
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in edges_by_id(graph) {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        if !succ[a].contains(&b) {
            succ[a].push(b);
            pred[b].push(a);
        }
    }
    // DFS postorder from s
    let mut post = Vec::new();
    let mut seen = vec![false; n];
    seen[s] = true;
    let mut stack = vec![(s, 0usize)];
    while let Some(&mut (u, ref mut k)) = stack.last_mut() {
        if *k < succ[u].len() {
            let v = succ[u][*k];
            *k += 1;
            if !seen[v] {
                seen[v] = true;
                stack.push((v, 0));
            }
        } else {
            post.push(u);
            stack.pop();
        }
    }
    let mut dfn = vec![usize::MAX; n];
    for (i, &u) in post.iter().enumerate() {
        dfn[u] = i;
    }
    let mut idom = vec![usize::MAX; n];
    idom[s] = s;
    let order: Vec<usize> = post.iter().rev().skip(1).copied().collect();
    let intersect = |mut a: usize, mut b: usize, idom: &[usize]| {
        while a != b {
            while dfn[a] < dfn[b] {
                a = idom[a];
            }
            while dfn[a] > dfn[b] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &u in &order {
            let mut new = usize::MAX;
            for &p in &pred[u] {
                if idom[p] == usize::MAX {
                    continue;
                }
                new = if new == usize::MAX {
                    p
                } else {
                    intersect(p, new, &idom)
                };
            }
            if new != usize::MAX && idom[u] != new {
                idom[u] = new;
                changed = true;
            }
        }
    }
    let reach: Vec<usize> = (0..n).filter(|&u| idom[u] != usize::MAX).collect();
    let mut df: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &u in &reach {
        let preds: Vec<usize> = pred[u]
            .iter()
            .copied()
            .filter(|&p| idom[p] != usize::MAX)
            .collect();
        // The start has no immediate dominator of its own, so an edge back into it is a join:
        // its walk climbs up to and including the start (the start is in its own frontier)
        if u == s || pred[u].len() >= 2 {
            for mut v in preds {
                while u == s || v != idom[u] {
                    if !df[v].contains(&u) {
                        df[v].push(u);
                    }
                    if v == s {
                        break;
                    }
                    v = idom[v];
                }
            }
        }
    }
    let idoms = reach.iter().map(|&u| (ids[u], ids[idom[u]])).collect();
    let fronts = reach
        .iter()
        .map(|&u| {
            let mut f: Vec<u64> = df[u].iter().map(|&v| ids[v]).collect();
            f.sort_unstable();
            (ids[u], f)
        })
        .collect();
    Ok((idoms, fronts))
}

// ---------------------------------------------------------------------------------------------
// k-edge-connectivity
// ---------------------------------------------------------------------------------------------

/// Gomory-Hu tree (Gusfield) of the unit-capacity undirected multigraph on `nodes` (indices into
/// `adj`, where parallel edges repeat): `(node, parent, cut value)` for every node but the first.
fn unit_gomory_hu(nodes: &[usize], adj: &[Vec<usize>]) -> Vec<(usize, usize, i64)> {
    let k = nodes.len();
    let pos: FxHashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut parent = vec![0usize; k];
    let mut fl = vec![0i64; k];
    for s in 1..k {
        let t = parent[s];
        let mut net = Dinic::new(k);
        for (i, &u) in nodes.iter().enumerate() {
            for &w in &adj[u] {
                if let Some(&j) = pos.get(&w) {
                    if i < j {
                        net.add(i, j, 1, 1);
                    }
                }
            }
        }
        let value = net.max_flow(s, t);
        let side = net.source_side(s);
        fl[s] = value;
        for i in 0..k {
            if i != s && side[i] && parent[i] == t {
                parent[i] = s;
            }
        }
        if t != 0 && side[parent[t]] {
            parent[s] = parent[t];
            parent[t] = s;
            fl[s] = fl[t];
            fl[t] = value;
        }
    }
    (1..k)
        .map(|i| (nodes[i], nodes[parent[i]], fl[i]))
        .collect()
}

/// Undirected adjacency with parallel edges repeated and self-loops dropped.
fn multi_adjacency(graph: &Graph, idx: &FxHashMap<u64, usize>) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); idx.len()];
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        if a != b {
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    adj
}

/// Groups of `nodes` whose pairwise local edge connectivity (within the subgraph induced by
/// `nodes`) is at least `k`.
fn components_at(nodes: &[usize], adj: &[Vec<usize>], k: i64) -> Vec<Vec<usize>> {
    let mut root: FxHashMap<usize, usize> = nodes.iter().map(|&v| (v, v)).collect();
    fn find(root: &mut FxHashMap<usize, usize>, x: usize) -> usize {
        let mut r = x;
        while root[&r] != r {
            r = root[&r];
        }
        let mut y = x;
        while root[&y] != r {
            let next = root[&y];
            root.insert(y, r);
            y = next;
        }
        r
    }
    for (a, b, w) in unit_gomory_hu(nodes, adj) {
        if w >= k {
            let (ra, rb) = (find(&mut root, a), find(&mut root, b));
            if ra != rb {
                root.insert(ra, rb);
            }
        }
    }
    let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for &v in nodes {
        let r = find(&mut root, v);
        groups.entry(r).or_default().push(v);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    for g in out.iter_mut() {
        g.sort_unstable();
    }
    out.sort();
    out
}

/// Directed arcs with multiplicity (parallel arcs repeat), self-loops dropped.
fn arc_list(graph: &Graph, idx: &FxHashMap<u64, usize>) -> Vec<(usize, usize)> {
    graph
        .get_edges_internal()
        .iter()
        .map(|e| (idx[&e.source], idx[&e.target]))
        .filter(|(a, b)| a != b)
        .collect()
}

/// Groups of `nodes` with pairwise local edge connectivity at least `k` in both directions,
/// counting only arcs inside `nodes`. The relation is transitive, so each class is the set of
/// nodes mutually k-edge-connected to its first member.
fn directed_components_at(nodes: &[usize], arcs: &[(usize, usize)], k: i64) -> Vec<Vec<usize>> {
    let pos: FxHashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let local: Vec<(usize, usize)> = arcs
        .iter()
        .filter_map(|(a, b)| Some((*pos.get(a)?, *pos.get(b)?)))
        .collect();
    let lambda = |s: usize, t: usize| -> i64 {
        let mut net = Dinic::new(nodes.len());
        for &(a, b) in &local {
            net.add(a, b, 1, 0);
        }
        net.max_flow(s, t)
    };
    let mut remaining: Vec<usize> = (0..nodes.len()).collect();
    let mut out = Vec::new();
    while let Some(&r) = remaining.first() {
        let mut class = vec![r];
        let mut rest = Vec::new();
        for &v in &remaining[1..] {
            if lambda(r, v) >= k && lambda(v, r) >= k {
                class.push(v);
            } else {
                rest.push(v);
            }
        }
        let mut group: Vec<usize> = class.into_iter().map(|i| nodes[i]).collect();
        group.sort_unstable();
        out.push(group);
        remaining = rest;
    }
    out.sort();
    out
}

/// Splits a node set into its k-edge-connected classes.
type Splitter = Box<dyn Fn(&[usize]) -> Vec<Vec<usize>>>;

/// k-edge-connected components (`subgraphs=false`: pairwise local edge connectivity in the
/// whole graph at least `k`, both ways on directed graphs) or k-edge-connected subgraphs
/// (`subgraphs=true`: maximal induced subgraphs whose own edge connectivity is at least `k`).
#[pyfunction]
pub fn k_edge_components(graph: &Graph, k: i64, subgraphs: bool) -> PyResult<Vec<Vec<u64>>> {
    if k < 1 {
        return value_err("k must be at least 1");
    }
    let (ids, idx) = sorted_ids(graph);
    let all: Vec<usize> = (0..ids.len()).collect();
    let split: Splitter = if graph.is_directed() {
        let arcs = arc_list(graph, &idx);
        Box::new(move |nodes: &[usize]| directed_components_at(nodes, &arcs, k))
    } else {
        let adj = multi_adjacency(graph, &idx);
        Box::new(move |nodes: &[usize]| components_at(nodes, &adj, k))
    };
    let mut result = Vec::new();
    if !subgraphs {
        result = split(&all);
    } else {
        let mut todo = vec![all];
        while let Some(group) = todo.pop() {
            let parts = split(&group);
            if parts.len() == 1 {
                result.push(parts.into_iter().next().unwrap());
            } else {
                todo.extend(parts);
            }
        }
        result.sort();
    }
    Ok(result
        .into_iter()
        .map(|g| g.into_iter().map(|v| ids[v]).collect())
        .collect())
}

/// Distinct arcs between different nodes (both orientations if undirected).
fn node_flow_arcs(graph: &Graph, idx: &FxHashMap<u64, usize>) -> Vec<(usize, usize)> {
    let directed = graph.is_directed();
    let mut arcs: FxHashSet<(usize, usize)> = FxHashSet::default();
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        if a != b {
            arcs.insert((a, b));
            if !directed {
                arcs.insert((b, a));
            }
        }
    }
    let mut arcs: Vec<(usize, usize)> = arcs.into_iter().collect();
    arcs.sort_unstable();
    arcs
}

/// Local node connectivity of `s` and `t` in the usual auxiliary network (node v split into
/// 2v in and 2v + 1 out with a unit arc; flow from s_out to t_in), as NetworkX: an arc between
/// them counts as one path.
fn local_node_connectivity(n: usize, arcs: &[(usize, usize)], s: usize, t: usize) -> i64 {
    let mut net = Dinic::new(2 * n);
    for v in 0..n {
        net.add(2 * v, 2 * v + 1, 1, 0);
    }
    for &(a, b) in arcs {
        net.add(2 * a + 1, 2 * b, 1, 0);
    }
    net.max_flow(2 * s + 1, 2 * t)
}

/// Average local node connectivity over all pairs (ordered pairs if directed), with the same
/// unit-capacity auxiliary network as NetworkX (adjacent pairs count their direct edge).
#[pyfunction]
pub fn average_node_connectivity(graph: &Graph) -> PyResult<f64> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let directed = graph.is_directed();
    let arcs = node_flow_arcs(graph, &idx);
    let (mut num, mut den) = (0i64, 0i64);
    for s in 0..n {
        for t in 0..n {
            if s == t || (!directed && t < s) {
                continue;
            }
            num += local_node_connectivity(n, &arcs, s, t);
            den += 1;
        }
    }
    Ok(if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    })
}

/// Local node connectivity of every pair of `nodes` (all nodes if None): `(u, v, value)` for
/// unordered pairs if undirected, ordered pairs if directed.
#[pyfunction]
#[pyo3(signature = (graph, nodes=None))]
pub fn all_pairs_node_connectivity(
    graph: &Graph,
    nodes: Option<Vec<u64>>,
) -> PyResult<Vec<(u64, u64, i64)>> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let chosen: Vec<usize> = match nodes {
        None => (0..n).collect(),
        Some(list) => {
            let mut seen = FxHashSet::default();
            let mut out = Vec::new();
            for v in list {
                let i = *idx.get(&v).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!("Node {} does not exist", v))
                })?;
                if seen.insert(i) {
                    out.push(i);
                }
            }
            out
        }
    };
    let directed = graph.is_directed();
    let arcs = node_flow_arcs(graph, &idx);
    let mut out = Vec::new();
    for (i, &s) in chosen.iter().enumerate() {
        for (j, &t) in chosen.iter().enumerate() {
            if i == j || (!directed && j < i) {
                continue;
            }
            out.push((ids[s], ids[t], local_node_connectivity(n, &arcs, s, t)));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// All minimum node cuts
// ---------------------------------------------------------------------------------------------

/// The minimum x-v node separator closest to x with every node in `forced` on x's side, if its
/// size is `k`: `(separator, x's side)` as node indices.
fn closest_separator(
    adj: &[Vec<usize>],
    x: usize,
    v: usize,
    forced: &[bool],
    k: i64,
) -> Option<(Vec<usize>, Vec<bool>)> {
    let n = adj.len();
    let inf = n as i64 + 1;
    let mut net = Dinic::new(2 * n);
    for u in 0..n {
        let cap = if u == x || u == v || forced[u] {
            inf
        } else {
            1
        };
        net.add(2 * u, 2 * u + 1, cap, 0);
        if forced[u] {
            net.add(2 * x + 1, 2 * u, inf, 0);
        }
        for &w in &adj[u] {
            net.add(2 * u + 1, 2 * w, inf, 0);
        }
    }
    if net.max_flow(2 * x + 1, 2 * v) != k {
        return None;
    }
    let side = net.source_side(2 * x + 1);
    let sep: Vec<usize> = (0..n)
        .filter(|&u| side[2 * u] && !side[2 * u + 1])
        .collect();
    let inside: Vec<bool> = (0..n).map(|u| side[2 * u + 1]).collect();
    Some((sep, inside))
}

/// Every minimum node cut (separator of size equal to the node connectivity) of a connected
/// undirected graph, as sorted lists of node IDs. Complete graphs have none.
///
/// With X the k highest-degree nodes (Kanevsky 1993), every minimum cut is X itself or separates
/// some x in X from a node v not adjacent to x. The minimum x-v separators form a lattice: from
/// the one closest to x, forcing any of its nodes onto x's side and taking the closest
/// separator again reaches every other one, each found once per pair.
#[pyfunction]
pub fn all_node_cuts(graph: &Graph) -> PyResult<Vec<Vec<u64>>> {
    if graph.is_directed() {
        return value_err("all_node_cuts() requires an undirected graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let adj = simple_adjacency(graph, &idx, false);
    let adj: Vec<Vec<usize>> = adj
        .into_iter()
        .map(|mut a| {
            a.sort_unstable();
            a.dedup();
            a
        })
        .collect();
    if n <= 1 {
        return Ok(Vec::new());
    }
    if !connected_without(&adj, &vec![false; n]) {
        return value_err("all_node_cuts() requires a connected graph");
    }
    let adjacent = |a: usize, b: usize| adj[a].binary_search(&b).is_ok();
    if (0..n).all(|u| adj[u].len() == n - 1) {
        return Ok(Vec::new());
    }
    // node connectivity (Even 1975): min over non-adjacent pairs (v_i, v_j), i <= k
    let arcs: Vec<(usize, usize)> = (0..n)
        .flat_map(|u| adj[u].iter().map(move |&w| (u, w)))
        .collect();
    let mut k = adj.iter().map(|a| a.len()).min().unwrap() as i64;
    let mut i = 0;
    while (i as i64) <= k && i < n {
        for j in i + 1..n {
            if !adjacent(i, j) {
                k = k.min(local_node_connectivity(n, &arcs, i, j));
            }
        }
        i += 1;
    }
    let mut by_degree: Vec<usize> = (0..n).collect();
    by_degree.sort_by(|&a, &b| adj[b].len().cmp(&adj[a].len()).then(a.cmp(&b)));
    let x_set: Vec<usize> = by_degree[..k as usize].to_vec();
    let mut found: FxHashSet<Vec<usize>> = FxHashSet::default();
    let mut removed = vec![false; n];
    for &u in &x_set {
        removed[u] = true;
    }
    if !connected_without(&adj, &removed) {
        let mut sep = x_set.clone();
        sep.sort_unstable();
        found.insert(sep);
    }
    for &x in &x_set {
        for v in 0..n {
            if v == x || adjacent(x, v) {
                continue;
            }
            let mut visited: FxHashSet<Vec<usize>> = FxHashSet::default();
            let mut stack = Vec::new();
            if let Some(first) = closest_separator(&adj, x, v, &vec![false; n], k) {
                stack.push(first);
            }
            while let Some((sep, inside)) = stack.pop() {
                if !visited.insert(sep.clone()) {
                    continue;
                }
                for &z in &sep {
                    let mut forced = inside.clone();
                    forced[z] = true;
                    if let Some(next) = closest_separator(&adj, x, v, &forced, k) {
                        if !visited.contains(&next.0) {
                            stack.push(next);
                        }
                    }
                }
                found.insert(sep);
            }
        }
    }
    let mut cuts: Vec<Vec<usize>> = found.into_iter().collect();
    cuts.sort();
    Ok(cuts
        .into_iter()
        .map(|c| c.into_iter().map(|u| ids[u]).collect())
        .collect())
}

/// Whether the nodes not in `removed` induce a connected graph (true when at most one is left).
fn connected_without(adj: &[Vec<usize>], removed: &[bool]) -> bool {
    let n = adj.len();
    let Some(start) = (0..n).find(|&u| !removed[u]) else {
        return true;
    };
    let mut seen = removed.to_vec();
    seen[start] = true;
    let mut queue = VecDeque::from([start]);
    while let Some(u) = queue.pop_front() {
        for &w in &adj[u] {
            if !seen[w] {
                seen[w] = true;
                queue.push_back(w);
            }
        }
    }
    seen.iter().all(|&s| s)
}

// ---------------------------------------------------------------------------------------------
// Small-world reference graphs and coefficients
// ---------------------------------------------------------------------------------------------

struct Simple {
    adj: Vec<FxHashSet<usize>>,
}

impl Simple {
    fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }
    fn edges(&self) -> usize {
        self.adj.iter().map(|s| s.len()).sum::<usize>() / 2
    }
    fn connected(&self, a: usize, b: usize) -> bool {
        let mut seen = vec![false; self.adj.len()];
        seen[a] = true;
        let mut q = VecDeque::from([a]);
        while let Some(u) = q.pop_front() {
            if u == b {
                return true;
            }
            for &w in &self.adj[u] {
                if !seen[w] {
                    seen[w] = true;
                    q.push_back(w);
                }
            }
        }
        false
    }
    fn transitivity(&self) -> f64 {
        let (mut tri, mut triples) = (0usize, 0usize);
        for (u, nb) in self.adj.iter().enumerate() {
            let d = nb.len();
            triples += d * d.saturating_sub(1);
            for &v in nb {
                if v > u {
                    tri += nb
                        .iter()
                        .filter(|&&w| w > v && self.adj[v].contains(&w))
                        .count();
                }
            }
        }
        if triples == 0 {
            0.0
        } else {
            6.0 * tri as f64 / triples as f64
        }
    }
    fn average_clustering(&self) -> f64 {
        let n = self.adj.len();
        let mut total = 0.0;
        for nb in &self.adj {
            let d = nb.len();
            if d < 2 {
                continue;
            }
            let links = nb
                .iter()
                .map(|&v| {
                    nb.iter()
                        .filter(|&&w| w != v && self.adj[v].contains(&w))
                        .count()
                })
                .sum::<usize>();
            total += links as f64 / (d * (d - 1)) as f64;
        }
        total / n as f64
    }
    fn average_path_length(&self) -> PyResult<f64> {
        let n = self.adj.len();
        let mut total = 0usize;
        for s in 0..n {
            let mut dist = vec![usize::MAX; n];
            dist[s] = 0;
            let mut q = VecDeque::from([s]);
            while let Some(u) = q.pop_front() {
                for &w in &self.adj[u] {
                    if dist[w] == usize::MAX {
                        dist[w] = dist[u] + 1;
                        q.push_back(w);
                    }
                }
            }
            if dist.contains(&usize::MAX) {
                return value_err("the graph must be connected");
            }
            total += dist.iter().sum::<usize>();
        }
        Ok(total as f64 / (n * (n - 1)) as f64)
    }
}

fn simple_from(graph: &Graph) -> PyResult<(Vec<u64>, Simple)> {
    if graph.is_directed() {
        return value_err("small-world measures require an undirected graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let mut adj = vec![FxHashSet::default(); ids.len()];
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        if a != b {
            adj[a].insert(b);
            adj[b].insert(a);
        }
    }
    let g = Simple { adj };
    if ids.len() < 4 {
        return value_err("the graph needs at least four nodes");
    }
    if g.edges() < 2 {
        return value_err("the graph needs at least two edges");
    }
    Ok((ids, g))
}

/// Degree-preserving rewiring by double edge swaps, as in NetworkX's `random_reference`
/// (`lattice=false`) and `lattice_reference` (`lattice=true`, only swaps that do not lengthen
/// the edges on a ring of the nodes in ID order).
fn rewire(
    g: &Simple,
    niter: usize,
    lattice: bool,
    connectivity: bool,
    rng: &mut ChaCha8Rng,
) -> Simple {
    let mut adj = g.adj.clone();
    let n = adj.len();
    let degrees: Vec<usize> = (0..n).map(|v| g.degree(v)).collect();
    let total: usize = degrees.iter().sum();
    let mut cdf = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &d in &degrees {
        acc += d as f64 / total as f64;
        cdf.push(acc);
    }
    let pick = |rng: &mut ChaCha8Rng| -> usize {
        let r = rng.gen::<f64>();
        cdf.partition_point(|&c| c <= r).min(n - 1)
    };
    let ring = |a: usize, b: usize| -> usize {
        let d = a.abs_diff(b);
        d.min(n - d)
    };
    let m = g.edges();
    let tries = (n as f64 * m as f64 / (n as f64 * (n - 1) as f64 / 2.0)) as usize;
    let rounds = niter * m;
    let mut work = Simple { adj: Vec::new() };
    for _ in 0..rounds {
        let mut attempts = 0usize;
        let mut guard = 0usize;
        while attempts < tries && guard < 100 * tries.max(1) {
            guard += 1;
            let (a, c) = (pick(rng), pick(rng));
            if a == c || adj[a].is_empty() || adj[c].is_empty() {
                continue;
            }
            let b = *adj[a].iter().nth(rng.gen_range(0..adj[a].len())).unwrap();
            let d = *adj[c].iter().nth(rng.gen_range(0..adj[c].len())).unwrap();
            if b == a || b == c || b == d || d == a || d == c {
                continue;
            }
            if !adj[a].contains(&d)
                && !adj[c].contains(&b)
                && (!lattice || ring(a, d) + ring(c, b) <= ring(a, b) + ring(c, d))
            {
                adj[a].remove(&b);
                adj[b].remove(&a);
                adj[c].remove(&d);
                adj[d].remove(&c);
                adj[a].insert(d);
                adj[d].insert(a);
                adj[c].insert(b);
                adj[b].insert(c);
                work.adj = std::mem::take(&mut adj);
                let ok = !connectivity || work.connected(a, b);
                adj = std::mem::take(&mut work.adj);
                if ok {
                    break;
                }
                adj[a].remove(&d);
                adj[d].remove(&a);
                adj[c].remove(&b);
                adj[b].remove(&c);
                adj[a].insert(b);
                adj[b].insert(a);
                adj[c].insert(d);
                adj[d].insert(c);
            }
            attempts += 1;
        }
    }
    Simple { adj }
}

fn rng_from(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

/// A random (`lattice=false`) or lattice (`lattice=true`) reference graph with the same degree
/// sequence: every node of `graph` (IDs, labels, attributes kept) and the rewired edges.
#[pyfunction]
#[pyo3(signature = (graph, niter, lattice, connectivity=true, seed=None))]
pub fn reference_graph(
    py: Python,
    graph: &Graph,
    niter: usize,
    lattice: bool,
    connectivity: bool,
    seed: Option<Seed>,
) -> PyResult<Graph> {
    let seed = seed.map(u64::from);
    let (ids, g) = simple_from(graph)?;
    let mut rng = rng_from(seed);
    let r = rewire(&g, niter, lattice, connectivity, &mut rng);
    let mut result = Graph::new_rust(false, false);
    let nodes = graph.get_nodes();
    for &id in &ids {
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
    let mut pairs: Vec<(usize, usize)> = (0..ids.len())
        .flat_map(|u| {
            r.adj[u]
                .iter()
                .filter(move |&&v| u < v)
                .map(move |&v| (u, v))
        })
        .collect();
    pairs.sort_unstable();
    for (a, b) in pairs {
        result.add_edge_rust(Edge {
            id: 0,
            source: ids[a],
            target: ids[b],
            weight: None,
            attributes: None,
        })?;
    }
    Ok(result)
}

/// Small-world coefficients: `sigma` (Humphries and Gurney) or `omega` (Telesford et al.),
/// computed as in NetworkX. Returns `(value, (C, L, C_ref, L_ref))`.
#[pyfunction]
#[pyo3(signature = (graph, kind, niter, nrand, seed=None))]
pub fn small_world(
    graph: &Graph,
    kind: &str,
    niter: usize,
    nrand: usize,
    seed: Option<Seed>,
) -> PyResult<(f64, (f64, f64, f64, f64))> {
    let seed = seed.map(u64::from);
    let (_, g) = simple_from(graph)?;
    if nrand < 1 {
        return value_err("nrand must be at least 1");
    }
    let mut rng = rng_from(seed);
    let l = g.average_path_length()?;
    match kind {
        "sigma" => {
            let c = g.transitivity();
            let (mut cr, mut lr) = (0.0, 0.0);
            for _ in 0..nrand {
                let r = rewire(&g, niter, false, true, &mut rng);
                cr += r.transitivity();
                lr += r.average_path_length()?;
            }
            cr /= nrand as f64;
            lr /= nrand as f64;
            Ok(((c / cr) / (l / lr), (c, l, cr, lr)))
        }
        "omega" => {
            let c = g.average_clustering();
            let mut cl = c;
            let mut lr = 0.0;
            for _ in 0..nrand {
                let r = rewire(&g, 2 * niter, false, true, &mut rng);
                lr += r.average_path_length()?;
                let lat = rewire(&g, niter, true, true, &mut rng);
                cl = cl.max(lat.average_clustering());
            }
            lr /= nrand as f64;
            Ok((lr / l - c / cl, (c, l, cl, lr)))
        }
        _ => value_err(format!("unknown coefficient {kind:?}")),
    }
}

// ---------------------------------------------------------------------------------------------
// Negative cycles
// ---------------------------------------------------------------------------------------------

/// Bellman-Ford arcs (both directions for undirected graphs).
fn bf_arcs(graph: &Graph, idx: &FxHashMap<u64, usize>, weighted: bool) -> Vec<(usize, usize, f64)> {
    let mut arcs = Vec::new();
    for e in edges_by_id(graph) {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        let w = if weighted {
            e.weight.unwrap_or(1.0)
        } else {
            1.0
        };
        arcs.push((a, b, w));
        if !graph.is_directed() && a != b {
            arcs.push((b, a, w));
        }
    }
    arcs
}

/// Bellman-Ford from the given sources (distance 0); returns a node relaxed in round `n`, if
/// any, with the predecessor array.
fn bellman_ford(
    n: usize,
    arcs: &[(usize, usize, f64)],
    sources: &[usize],
) -> (Option<usize>, Vec<usize>) {
    let mut dist = vec![f64::INFINITY; n];
    let mut pred = vec![usize::MAX; n];
    for &s in sources {
        dist[s] = 0.0;
    }
    let mut last = None;
    for _ in 0..n {
        last = None;
        for &(a, b, w) in arcs {
            if dist[a].is_finite() && dist[a] + w < dist[b] {
                dist[b] = dist[a] + w;
                pred[b] = a;
                last = Some(b);
            }
        }
        if last.is_none() {
            break;
        }
    }
    (last, pred)
}

/// Whether any negative cycle exists (reachable from anywhere).
#[pyfunction]
#[pyo3(signature = (graph, weighted=true))]
pub fn negative_edge_cycle(graph: &Graph, weighted: bool) -> bool {
    let (ids, idx) = sorted_ids(graph);
    let arcs = bf_arcs(graph, &idx, weighted);
    let all: Vec<usize> = (0..ids.len()).collect();
    bellman_ford(ids.len(), &arcs, &all).0.is_some()
}

/// A negative cycle reachable from `source`, as node IDs with the first node repeated at the end.
#[pyfunction]
#[pyo3(signature = (graph, source, weighted=true))]
pub fn find_negative_cycle(graph: &Graph, source: u64, weighted: bool) -> PyResult<Vec<u64>> {
    let (ids, idx) = sorted_ids(graph);
    let Some(&s) = idx.get(&source) else {
        return value_err(format!("Source node {source} does not exist"));
    };
    let n = ids.len();
    let arcs = bf_arcs(graph, &idx, weighted);
    let (last, pred) = bellman_ford(n, &arcs, &[s]);
    let Some(mut v) = last else {
        return value_err("no negative cycle is reachable from the source");
    };
    for _ in 0..n {
        v = pred[v];
    }
    let mut cycle = vec![v];
    let mut u = pred[v];
    while u != v {
        cycle.push(u);
        u = pred[u];
    }
    cycle.push(v);
    cycle.reverse();
    Ok(cycle.into_iter().map(|i| ids[i]).collect())
}

/// Goldberg-Radzik single-source shortest paths (negative weights allowed). Returns
/// `(node, predecessor, distance)` for each node reachable from `source`, in node ID order (the
/// source has no predecessor). Each round topologically sorts the nodes reachable through
/// negative reduced-cost arcs from those relabeled in the previous round (finding a negative
/// cycle as a back arc), then scans them in that order, as NetworkX's `goldberg_radzik()` does.
#[pyfunction]
#[pyo3(signature = (graph, source, weighted=true))]
#[allow(clippy::type_complexity)]
pub fn goldberg_radzik(
    graph: &Graph,
    source: u64,
    weighted: bool,
) -> PyResult<Vec<(u64, Option<u64>, f64)>> {
    let (ids, idx) = sorted_ids(graph);
    let Some(&s) = idx.get(&source) else {
        return value_err(format!("Source node {source} does not exist"));
    };
    let n = ids.len();
    let mut succ: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (a, b, w) in bf_arcs(graph, &idx, weighted) {
        if a == b && w < 0.0 {
            return value_err("Negative cycle detected");
        }
        succ[a].push((b, w));
    }

    let mut d = vec![f64::INFINITY; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    let mut reached = vec![false; n];
    d[s] = 0.0;
    reached[s] = true;

    // Visit marker doubling as the count of negative reduced-cost arcs on the DFS path
    let mut neg_count: Vec<Option<usize>> = vec![None; n];
    let mut in_stack = vec![false; n];
    let mut relabeled = vec![s];
    let mut is_relabeled = vec![false; n];
    while !relabeled.is_empty() {
        // Topological sort of the nodes reachable through arcs of negative reduced cost
        let mut to_scan: Vec<usize> = Vec::new();
        let mut visited: Vec<usize> = Vec::new();
        for &root in &relabeled {
            if neg_count[root].is_some() || succ[root].iter().all(|&(v, w)| d[root] + w >= d[v]) {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
            in_stack[root] = true;
            neg_count[root] = Some(0);
            visited.push(root);
            while let Some(top) = stack.last_mut() {
                let u = top.0;
                if top.1 == succ[u].len() {
                    to_scan.push(u);
                    stack.pop();
                    in_stack[u] = false;
                    continue;
                }
                let (v, w) = succ[u][top.1];
                top.1 += 1;
                let t = d[u] + w;
                if t < d[v] {
                    d[v] = t;
                    pred[v] = Some(u);
                    reached[v] = true;
                    let count = neg_count[u].unwrap_or(0) + 1;
                    match neg_count[v] {
                        None => {
                            neg_count[v] = Some(count);
                            visited.push(v);
                            stack.push((v, 0));
                            in_stack[v] = true;
                        }
                        Some(c) if in_stack[v] && count > c => {
                            return value_err("Negative cycle detected");
                        }
                        _ => {}
                    }
                }
            }
        }
        for v in visited {
            neg_count[v] = None;
        }
        to_scan.reverse();

        // Relax the out-arcs in topological order
        for &v in &relabeled {
            is_relabeled[v] = false;
        }
        relabeled.clear();
        for &u in &to_scan {
            let d_u = d[u];
            for &(v, w) in &succ[u] {
                if d_u + w < d[v] {
                    d[v] = d_u + w;
                    pred[v] = Some(u);
                    reached[v] = true;
                    if !is_relabeled[v] {
                        is_relabeled[v] = true;
                        relabeled.push(v);
                    }
                }
            }
        }
    }
    Ok((0..n)
        .filter(|&i| reached[i])
        .map(|i| {
            (
                ids[i],
                if i == s {
                    None
                } else {
                    pred[i].map(|p| ids[p])
                },
                d[i],
            )
        })
        .collect())
}

// ---------------------------------------------------------------------------------------------
// Link prediction
// ---------------------------------------------------------------------------------------------

/// Link prediction scores for node pairs (all non-adjacent pairs if `pairs` is None), on the
/// underlying simple undirected graph: `"ccpa"` (common neighbor and centrality based
/// parameterized algorithm), `"cn_sh"` and `"ra_sh"` (Soundarajan-Hopcroft) and
/// `"within_inter_cluster"`. Community-aware scores take a community value per node.
#[pyfunction]
#[pyo3(signature = (graph, kind, pairs=None, community=None, alpha=0.8, delta=0.001))]
pub fn link_scores(
    graph: &Graph,
    kind: &str,
    pairs: Option<Vec<(u64, u64)>>,
    community: Option<FxHashMap<u64, i64>>,
    alpha: f64,
    delta: f64,
) -> PyResult<Vec<(u64, u64, f64)>> {
    if graph.is_directed() {
        return value_err("link prediction scores require an undirected graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        adj[a].insert(b);
        adj[b].insert(a);
    }
    let degree = |v: usize| adj[v].len() + adj[v].contains(&v) as usize;
    let pair_list: Vec<(usize, usize)> = match pairs {
        Some(list) => {
            let mut out = Vec::with_capacity(list.len());
            for (u, v) in list {
                match (idx.get(&u), idx.get(&v)) {
                    (Some(&a), Some(&b)) => out.push((a, b)),
                    _ => {
                        return value_err(format!("pair ({u}, {v}) has a node that does not exist"))
                    }
                }
            }
            out
        }
        None => (0..n)
            .flat_map(|u| (u + 1..n).map(move |v| (u, v)))
            .filter(|&(u, v)| !adj[u].contains(&v))
            .collect(),
    };
    let comm: Option<Vec<i64>> = match (&community, kind) {
        (_, "ccpa") => None,
        (Some(map), _) => {
            let mut c = Vec::with_capacity(n);
            for &id in &ids {
                match map.get(&id) {
                    Some(&x) => c.push(x),
                    None => return value_err(format!("node {id} has no community value")),
                }
            }
            Some(c)
        }
        (None, _) => return value_err("this score needs community information"),
    };
    let common = |u: usize, v: usize| -> Vec<usize> {
        adj[u]
            .iter()
            .copied()
            .filter(|&w| w != u && w != v && adj[v].contains(&w))
            .collect()
    };
    // BFS distances on demand for CCPA
    let mut dist_cache: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut out = Vec::with_capacity(pair_list.len());
    for (u, v) in pair_list {
        if u == v {
            return value_err("self-pairs are not supported");
        }
        let cn = common(u, v);
        let score = match kind {
            "ccpa" => {
                if alpha == 1.0 {
                    cn.len() as f64
                } else {
                    let dist = dist_cache.entry(u).or_insert_with(|| {
                        let mut d = vec![usize::MAX; n];
                        d[u] = 0;
                        let mut q = VecDeque::from([u]);
                        while let Some(x) = q.pop_front() {
                            for &y in &adj[x] {
                                if d[y] == usize::MAX {
                                    d[y] = d[x] + 1;
                                    q.push_back(y);
                                }
                            }
                        }
                        d
                    });
                    let path = if dist[v] == usize::MAX {
                        f64::INFINITY
                    } else {
                        dist[v] as f64
                    };
                    alpha * cn.len() as f64 + (1.0 - alpha) * n as f64 / path
                }
            }
            "cn_sh" => {
                let c = comm.as_ref().unwrap();
                let bonus = if c[u] == c[v] {
                    cn.iter().filter(|&&w| c[w] == c[u]).count()
                } else {
                    0
                };
                (cn.len() + bonus) as f64
            }
            "ra_sh" => {
                let c = comm.as_ref().unwrap();
                if c[u] != c[v] {
                    0.0
                } else {
                    cn.iter()
                        .filter(|&&w| c[w] == c[u])
                        .map(|&w| 1.0 / degree(w) as f64)
                        .sum()
                }
            }
            "within_inter_cluster" => {
                if delta <= 0.0 {
                    return value_err("delta must be greater than zero");
                }
                let c = comm.as_ref().unwrap();
                if c[u] != c[v] {
                    0.0
                } else {
                    let within = cn.iter().filter(|&&w| c[w] == c[u]).count();
                    within as f64 / ((cn.len() - within) as f64 + delta)
                }
            }
            _ => return value_err(format!("unknown score {kind:?}")),
        };
        out.push((ids[u], ids[v], score));
    }
    Ok(out)
}
