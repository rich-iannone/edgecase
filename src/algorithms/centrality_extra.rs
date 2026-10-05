//! Additional centralities: subgraph centrality, current-flow (electrical) betweenness and
//! closeness, group centralities, VoteRank, trophic levels, reaching centrality, subset
//! betweenness, and Burt's effective size / local constraint.
//!
//! Where a result depends on edge weights, `weighted` selects whether each edge's `weight`
//! (1.0 if missing) is used; otherwise every edge counts 1.0. Node orderings are by ID.

use crate::graph::Graph;
use nalgebra as na;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BinaryHeap, VecDeque};

fn err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn weight_of(e: &crate::graph::Edge, weighted: bool) -> f64 {
    if weighted {
        e.weight.unwrap_or(1.0)
    } else {
        1.0
    }
}

fn check_nonnegative(graph: &Graph, weighted: bool) -> PyResult<()> {
    if weighted && graph.get_edges_internal().iter().any(|e| e.weight.is_some_and(|w| w < 0.0)) {
        return Err(err("Edge weights must be non-negative"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Subgraph centrality
// ---------------------------------------------------------------------------------------------

/// Subgraph centrality (Estrada & Rodríguez-Velázquez 2005): the diagonal of exp(A), i.e. a
/// weighted count of closed walks through each node, with walks of length k weighted 1/k!.
/// The adjacency matrix is binary (parallel edges count once; a self-loop puts 1 on the
/// diagonal), as in NetworkX. Undirected graphs only.
#[pyfunction]
pub fn subgraph_centrality(graph: &Graph) -> PyResult<FxHashMap<u64, f64>> {
    if graph.is_directed() {
        return Err(err("subgraph_centrality() requires an undirected graph"));
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n == 0 {
        return Ok(FxHashMap::default());
    }
    let mut a = na::DMatrix::<f64>::zeros(n, n);
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        a[(s, t)] = 1.0;
        a[(t, s)] = 1.0;
    }
    let eig = na::SymmetricEigen::new(a);
    let expw: Vec<f64> = eig.eigenvalues.iter().map(|w| w.exp()).collect();
    Ok((0..n)
        .map(|i| {
            let sc: f64 = (0..n).map(|j| eig.eigenvectors[(i, j)].powi(2) * expw[j]).sum();
            (ids[i], sc)
        })
        .collect())
}

// ---------------------------------------------------------------------------------------------
// Current flow (random-walk / electrical) centralities
// ---------------------------------------------------------------------------------------------

/// Edges as `(u, v, conductance)` index triples (no self-loops), checking the graph is an
/// undirected connected network with positive total conductance at every node.
fn electrical_network(
    graph: &Graph,
    weighted: bool,
    what: &str,
) -> PyResult<(Vec<u64>, Vec<(usize, usize, f64)>)> {
    if graph.is_directed() {
        return Err(err(format!("{} requires an undirected graph", what)));
    }
    if weighted && graph.get_edges_internal().iter().any(|e| e.weight.is_some_and(|w| w <= 0.0)) {
        return Err(err(format!("{} requires positive edge weights (conductances)", what)));
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let edges: Vec<(usize, usize, f64)> = graph
        .get_edges_internal()
        .iter()
        .filter(|e| e.source != e.target)
        .map(|e| (idx[&e.source], idx[&e.target], weight_of(e, weighted)))
        .collect();
    // connectivity check
    if n > 0 {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v, _) in &edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut seen = vec![false; n];
        let mut q = VecDeque::from([0usize]);
        seen[0] = true;
        let mut count = 1;
        while let Some(x) = q.pop_front() {
            for &y in &adj[x] {
                if !seen[y] {
                    seen[y] = true;
                    count += 1;
                    q.push_back(y);
                }
            }
        }
        if count != n {
            return Err(err(format!("{} requires a connected graph", what)));
        }
    }
    Ok((ids, edges))
}

/// Inverse of the Laplacian with node 0 grounded (row and column 0 are zero). Potential
/// differences, currents and effective resistances computed from it are exact.
fn grounded_laplacian_inverse(n: usize, edges: &[(usize, usize, f64)]) -> PyResult<na::DMatrix<f64>> {
    let mut c = na::DMatrix::<f64>::zeros(n, n);
    if n <= 1 {
        return Ok(c);
    }
    let mut l = na::DMatrix::<f64>::zeros(n - 1, n - 1);
    for &(u, v, w) in edges {
        for (a, b) in [(u, v), (v, u)] {
            if a > 0 {
                l[(a - 1, a - 1)] += w;
                if b > 0 {
                    l[(a - 1, b - 1)] -= w;
                }
            }
        }
    }
    let inv = l
        .cholesky()
        .ok_or_else(|| err("The Laplacian is singular; is the graph connected?"))?
        .inverse();
    for i in 1..n {
        for j in 1..n {
            c[(i, j)] = inv[(i - 1, j - 1)];
        }
    }
    Ok(c)
}

/// Current-flow betweenness (Brandes & Fleischer 2005; Newman's random-walk betweenness):
/// for each node, the total current through it over all source-target pairs when one unit
/// is injected at the source and extracted at the target. Raw values sum over unordered
/// pairs not involving the node; `normalized` divides by (n-1)(n-2)/2. Matches NetworkX.
#[pyfunction]
#[pyo3(signature = (graph, normalized=true, weighted=false))]
pub fn current_flow_betweenness_centrality(
    graph: &Graph,
    normalized: bool,
    weighted: bool,
) -> PyResult<FxHashMap<u64, f64>> {
    let (ids, edges) = electrical_network(graph, weighted, "current_flow_betweenness_centrality()")?;
    let n = ids.len();
    if n < 3 {
        return Ok(ids.iter().map(|&id| (id, 0.0)).collect());
    }
    let c = grounded_laplacian_inverse(n, &edges)?;
    let mut total = vec![0.0f64; n];
    let mut row = vec![0.0f64; n];
    for &(u, v, w) in &edges {
        for k in 0..n {
            row[k] = w * (c[(u, k)] - c[(v, k)]);
        }
        row.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Σ_{s<t} |row_s - row_t| from sorted values
        let s_e: f64 = row
            .iter()
            .enumerate()
            .map(|(i, &x)| (2.0 * i as f64 - (n as f64 - 1.0)) * x)
            .sum();
        total[u] += 0.5 * s_e;
        total[v] += 0.5 * s_e;
    }
    // Each of the n-1 pairs with the node as an endpoint carried exactly half a unit through it
    let scale = if normalized { 2.0 / ((n - 1) as f64 * (n - 2) as f64) } else { 1.0 };
    Ok(ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, (total[i] - (n as f64 - 1.0) / 2.0).max(0.0) * scale))
        .collect())
}

/// Current-flow closeness (information centrality, Stephenson & Zelen 1989): the reciprocal of
/// the sum of effective resistances from a node to all others. Matches NetworkX.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn current_flow_closeness_centrality(
    graph: &Graph,
    weighted: bool,
) -> PyResult<FxHashMap<u64, f64>> {
    let (ids, edges) = electrical_network(graph, weighted, "current_flow_closeness_centrality()")?;
    let n = ids.len();
    let c = grounded_laplacian_inverse(n, &edges)?;
    Ok((0..n)
        .map(|v| {
            let total: f64 = (0..n).map(|w| c[(v, v)] + c[(w, w)] - 2.0 * c[(v, w)]).sum();
            (ids[v], if total > 0.0 { 1.0 / total } else { 0.0 })
        })
        .collect())
}

// ---------------------------------------------------------------------------------------------
// Shortest-path DAGs (BFS or Dijkstra) shared by group and subset betweenness
// ---------------------------------------------------------------------------------------------

struct SpDag {
    order: Vec<usize>,
    preds: Vec<Vec<usize>>,
    sigma: Vec<f64>,
    dist: Vec<f64>,
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct OrdF(f64);
impl Eq for OrdF {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrdF {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn same(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return a == b;
    }
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

/// adj[i] = [(j, w)] following edge direction (both ways if undirected), no self-loops.
fn adjacency(graph: &Graph, idx: &FxHashMap<u64, usize>, weighted: bool, reverse: bool) -> Vec<Vec<(usize, f64)>> {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); idx.len()];
    let undirected = !graph.is_directed();
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let (s, t) = (idx[&e.source], idx[&e.target]);
        let w = weight_of(e, weighted);
        if undirected {
            adj[s].push((t, w));
            adj[t].push((s, w));
        } else if reverse {
            adj[t].push((s, w));
        } else {
            adj[s].push((t, w));
        }
    }
    adj
}

fn sp_dag(adj: &[Vec<(usize, f64)>], s: usize) -> SpDag {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut sigma = vec![0.0; n];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut done = vec![false; n];
    let mut order = Vec::new();
    let mut heap = BinaryHeap::new();
    dist[s] = 0.0;
    sigma[s] = 1.0;
    heap.push((std::cmp::Reverse(OrdF(0.0)), s));
    while let Some((std::cmp::Reverse(OrdF(d)), v)) = heap.pop() {
        if done[v] || d > dist[v] {
            continue;
        }
        done[v] = true;
        order.push(v);
        for &(w, c) in &adj[v] {
            if done[w] {
                continue;
            }
            let nd = d + c;
            if nd < dist[w] && !same(nd, dist[w]) {
                dist[w] = nd;
                sigma[w] = sigma[v];
                preds[w] = vec![v];
                heap.push((std::cmp::Reverse(OrdF(nd)), w));
            } else if same(nd, dist[w]) {
                sigma[w] += sigma[v];
                preds[w].push(v);
            }
        }
    }
    SpDag { order, preds, sigma, dist }
}

/// Group betweenness (Everett & Borgatti 1999): the share of shortest paths between pairs of
/// nodes outside the group that pass through at least one group member. With `endpoints`,
/// pairs with an endpoint in the group count too. Normalized by (n-c)(n-c-1); unnormalized
/// undirected values count each pair once. Matches NetworkX.
#[pyfunction]
#[pyo3(signature = (graph, group, normalized=true, endpoints=false, weighted=false))]
pub fn group_betweenness_centrality(
    graph: &Graph,
    group: Vec<u64>,
    normalized: bool,
    endpoints: bool,
    weighted: bool,
) -> PyResult<f64> {
    check_nonnegative(graph, weighted)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut in_group = vec![false; n];
    for g in &group {
        in_group[*idx.get(g).ok_or_else(|| err(format!("Node {} does not exist", g)))?] = true;
    }
    let c = in_group.iter().filter(|&&x| x).count();
    let adj = adjacency(graph, &idx, weighted, false);
    let mut total = 0.0;
    for s in 0..n {
        if in_group[s] && !endpoints {
            continue;
        }
        let dag = sp_dag(&adj, s);
        // shortest paths from s avoiding every group node (s itself may be in the group only
        // when endpoints are counted, and then every path "touches" the group)
        let mut avoid = vec![0.0f64; n];
        if !in_group[s] {
            avoid[s] = 1.0;
        }
        for &v in &dag.order {
            if v == s || in_group[v] {
                continue;
            }
            avoid[v] = dag.preds[v].iter().map(|&p| avoid[p]).sum();
        }
        for t in 0..n {
            if t == s || !dag.dist[t].is_finite() {
                continue;
            }
            if in_group[t] && !endpoints {
                continue;
            }
            let through = if in_group[t] { dag.sigma[t] } else { dag.sigma[t] - avoid[t] };
            total += through / dag.sigma[t];
        }
    }
    if normalized {
        let denom = ((n - c) * (n.saturating_sub(c + 1))) as f64;
        if denom > 0.0 {
            total /= denom;
        }
    } else if !graph.is_directed() {
        total /= 2.0;
    }
    Ok(total)
}

/// Group closeness: |V - S| divided by the sum, over nodes outside the group, of their
/// distance to the nearest group member (distance *to* the group on directed graphs).
/// Unreachable nodes add 0. Matches NetworkX.
#[pyfunction]
#[pyo3(signature = (graph, group, weighted=false))]
pub fn group_closeness_centrality(graph: &Graph, group: Vec<u64>, weighted: bool) -> PyResult<f64> {
    check_nonnegative(graph, weighted)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let adj = adjacency(graph, &idx, weighted, true);
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();
    let mut in_group = vec![false; n];
    for g in &group {
        let i = *idx.get(g).ok_or_else(|| err(format!("Node {} does not exist", g)))?;
        in_group[i] = true;
        dist[i] = 0.0;
        heap.push((std::cmp::Reverse(OrdF(0.0)), i));
    }
    while let Some((std::cmp::Reverse(OrdF(d)), v)) = heap.pop() {
        if d > dist[v] {
            continue;
        }
        for &(w, c) in &adj[v] {
            if d + c < dist[w] {
                dist[w] = d + c;
                heap.push((std::cmp::Reverse(OrdF(d + c)), w));
            }
        }
    }
    let outside = in_group.iter().filter(|&&x| !x).count();
    let total: f64 = (0..n).filter(|&v| !in_group[v] && dist[v].is_finite()).map(|v| dist[v]).sum();
    Ok(if total > 0.0 { outside as f64 / total } else { 0.0 })
}

/// Group degree: the share of nodes outside the group adjacent to at least one member.
/// `mode` picks out-, in- or any neighbors on directed graphs.
#[pyfunction]
#[pyo3(signature = (graph, group, mode="all"))]
pub fn group_degree_centrality(graph: &Graph, group: Vec<u64>, mode: &str) -> PyResult<f64> {
    let (use_out, use_in) = match mode {
        "all" | "total" => (true, true),
        "out" => (true, false),
        "in" => (false, true),
        _ => return Err(err(format!("Invalid mode '{}'. Must be 'all', 'in', or 'out'", mode))),
    };
    let (ids, idx) = sorted_ids(graph);
    let members: FxHashSet<u64> = group.iter().copied().collect();
    for g in &members {
        if !idx.contains_key(g) {
            return Err(err(format!("Node {} does not exist", g)));
        }
    }
    let undirected = !graph.is_directed();
    let mut reached: FxHashSet<u64> = FxHashSet::default();
    for e in graph.get_edges_internal() {
        if members.contains(&e.source) && (use_out || undirected) {
            reached.insert(e.target);
        }
        if members.contains(&e.target) && (use_in || undirected) {
            reached.insert(e.source);
        }
    }
    let outside = ids.len() - members.len();
    let count = reached.iter().filter(|v| !members.contains(v)).count();
    Ok(if outside > 0 { count as f64 / outside as f64 } else { 0.0 })
}

// ---------------------------------------------------------------------------------------------
// VoteRank
// ---------------------------------------------------------------------------------------------

/// VoteRank (Zhang et al. 2016): pick spreaders one at a time; each round every node's score is
/// the voting ability of its neighbors (out-neighbors on directed graphs), the top scorer is
/// elected, loses its vote, and its neighbors' ability drops by 1 / average degree. Stops at
/// `count` nodes or when no one has a positive score. Matches NetworkX, including ties
/// (lowest node ID wins).
#[pyfunction]
#[pyo3(signature = (graph, count=None))]
pub fn voterank(graph: &Graph, count: Option<usize>) -> PyResult<Vec<u64>> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let k = count.unwrap_or(n).min(n);
    let directed = graph.is_directed();
    let edges: Vec<(usize, usize)> = graph
        .get_edges_internal()
        .iter()
        .map(|e| (idx[&e.source], idx[&e.target]))
        .collect();
    let avg = if directed {
        edges.len() as f64 / n as f64
    } else {
        2.0 * edges.len() as f64 / n as f64
    };
    let mut ability = vec![1.0f64; n];
    let mut elected = vec![false; n];
    let mut out = Vec::new();
    for _ in 0..k {
        let mut score = vec![0.0f64; n];
        for &(a, b) in &edges {
            score[a] += ability[b];
            if !directed {
                score[b] += ability[a];
            }
        }
        for v in 0..n {
            if elected[v] {
                score[v] = 0.0;
            }
        }
        let mut best = 0;
        for v in 1..n {
            if score[v] > score[best] {
                best = v;
            }
        }
        if score[best] == 0.0 {
            break;
        }
        out.push(ids[best]);
        elected[best] = true;
        ability[best] = 0.0;
        for &(a, b) in &edges {
            let nbr = if a == best {
                b
            } else if !directed && b == best {
                a
            } else {
                continue;
            };
            ability[nbr] = (ability[nbr] - 1.0 / avg).max(0.0);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Trophic levels
// ---------------------------------------------------------------------------------------------

/// Trophic levels (Levine 1980): basal nodes (no incoming edges) have level 1; every other
/// node's level is 1 plus the in-weight-averaged level of its predecessors. Requires every
/// node to be reachable from a basal node. Directed graphs only.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false, skip_self_loops=false))]
pub fn trophic_levels(
    graph: &Graph,
    weighted: bool,
    skip_self_loops: bool,
) -> PyResult<FxHashMap<u64, f64>> {
    if !graph.is_directed() {
        return Err(err("trophic_levels() requires a directed graph"));
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut in_w = vec![0.0f64; n];
    let mut has_in = vec![false; n];
    let mut w_in: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut out_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in graph.get_edges_internal() {
        if skip_self_loops && e.source == e.target {
            continue;
        }
        let (s, t) = (idx[&e.source], idx[&e.target]);
        let w = weight_of(e, weighted);
        in_w[t] += w;
        has_in[t] = true;
        w_in[t].push((s, w));
        out_adj[s].push(t);
    }
    let basal: Vec<usize> = (0..n).filter(|&v| !has_in[v]).collect();
    let unreachable_msg = "Trophic levels are only defined for graphs where every node has a path \
                           from a basal node (a node with no incoming edges)";
    if basal.is_empty() && n > 0 {
        return Err(err("The graph has no basal nodes (nodes with no incoming edges), so trophic levels are not defined"));
    }
    let mut seen = vec![false; n];
    let mut q: VecDeque<usize> = basal.iter().copied().collect();
    for &b in &basal {
        seen[b] = true;
    }
    while let Some(v) = q.pop_front() {
        for &w in &out_adj[v] {
            if !seen[w] {
                seen[w] = true;
                q.push_back(w);
            }
        }
    }
    if seen.iter().any(|&s| !s) {
        return Err(err(unreachable_msg));
    }
    // Solve (I - P) y = 1 over non-basal nodes, P[i][j] = w_ji / in_w_i (j non-basal);
    // basal predecessors contribute their level 1 to the right-hand side
    let rest: Vec<usize> = (0..n).filter(|&v| has_in[v]).collect();
    let pos: FxHashMap<usize, usize> = rest.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let m = rest.len();
    let mut a = na::DMatrix::<f64>::identity(m, m);
    let mut b = na::DVector::<f64>::from_element(m, 1.0);
    for (i, &v) in rest.iter().enumerate() {
        if in_w[v] == 0.0 {
            return Err(err(unreachable_msg));
        }
        for &(u, w) in &w_in[v] {
            let p = w / in_w[v];
            match pos.get(&u) {
                Some(&j) => a[(i, j)] -= p,
                None => b[i] += p, // basal predecessor, level 1
            }
        }
    }
    let y = a.lu().solve(&b).ok_or_else(|| err(unreachable_msg))?;
    let mut levels: FxHashMap<u64, f64> = FxHashMap::default();
    for v in 0..n {
        levels.insert(ids[v], 1.0);
    }
    for (i, &v) in rest.iter().enumerate() {
        levels.insert(ids[v], y[i]);
    }
    Ok(levels)
}

// ---------------------------------------------------------------------------------------------
// Reaching centrality
// ---------------------------------------------------------------------------------------------

/// Local reaching centrality of every node (Mones, Vicsek & Vicsek 2012), as in NetworkX:
/// unweighted directed graphs use the share of nodes a node can reach; unweighted undirected
/// graphs average 1/d over reachable nodes; weighted graphs average the mean edge weight along
/// the strongest paths (distance = total weight / edge weight), optionally normalized by the
/// mean edge weight.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false, normalized=true))]
pub fn local_reaching_centrality(
    graph: &Graph,
    weighted: bool,
    normalized: bool,
) -> PyResult<FxHashMap<u64, f64>> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n < 2 {
        return Err(err("Reaching centrality needs at least two nodes"));
    }
    let edges = graph.get_edges_internal();
    let total_w: f64 = edges.iter().map(|e| weight_of(e, weighted)).sum();
    if weighted && edges.iter().any(|e| e.weight.is_some_and(|w| w <= 0.0)) {
        return Err(err("Reaching centrality requires positive edge weights"));
    }
    if total_w <= 0.0 {
        return Err(err("Reaching centrality requires at least one edge"));
    }
    let directed = graph.is_directed();
    let norm = if weighted && normalized { total_w / edges.len() as f64 } else { 1.0 };
    // adjacency with (neighbor, weight); for paths, distance = total_w / weight when weighted
    let adj = adjacency(graph, &idx, weighted, false);
    let mut out = FxHashMap::default();
    for s in 0..n {
        // Dijkstra on the transformed distances, tracking path length and summed weight
        let mut dist = vec![f64::INFINITY; n];
        let mut hops = vec![0usize; n];
        let mut wsum = vec![0.0f64; n];
        let mut done = vec![false; n];
        let mut heap = BinaryHeap::new();
        dist[s] = 0.0;
        heap.push((std::cmp::Reverse(OrdF(0.0)), s));
        while let Some((std::cmp::Reverse(OrdF(d)), v)) = heap.pop() {
            if done[v] || d > dist[v] {
                continue;
            }
            done[v] = true;
            for &(w, c) in &adj[v] {
                let step = if weighted { total_w / c } else { 1.0 };
                if !done[w] && d + step < dist[w] {
                    dist[w] = d + step;
                    hops[w] = hops[v] + 1;
                    wsum[w] = wsum[v] + c;
                    heap.push((std::cmp::Reverse(OrdF(d + step)), w));
                }
            }
        }
        let reached: Vec<usize> = (0..n).filter(|&v| v != s && dist[v].is_finite()).collect();
        let value = if !weighted && directed {
            reached.len() as f64 / (n - 1) as f64
        } else if !weighted {
            reached.iter().map(|&v| 1.0 / hops[v] as f64).sum::<f64>() / (n - 1) as f64
        } else {
            reached.iter().map(|&v| wsum[v] / hops[v] as f64).sum::<f64>() / norm / (n - 1) as f64
        };
        out.insert(ids[s], value);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Subset betweenness
// ---------------------------------------------------------------------------------------------

/// Betweenness counting only shortest paths from `sources` to `targets` (Brandes 2008).
/// Scaling as NetworkX: normalized divides by (n-1)(n-2); otherwise undirected values are
/// halved.
#[pyfunction]
#[pyo3(signature = (graph, sources, targets, normalized=false, weighted=false))]
pub fn betweenness_centrality_subset(
    graph: &Graph,
    sources: Vec<u64>,
    targets: Vec<u64>,
    normalized: bool,
    weighted: bool,
) -> PyResult<FxHashMap<u64, f64>> {
    check_nonnegative(graph, weighted)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let lookup = |v: &u64| idx.get(v).copied().ok_or_else(|| err(format!("Node {} does not exist", v)));
    let srcs: Vec<usize> = sources.iter().map(lookup).collect::<PyResult<_>>()?;
    let mut is_target = vec![false; n];
    for t in &targets {
        is_target[lookup(t)?] = true;
    }
    let adj = adjacency(graph, &idx, weighted, false);
    let mut bc = vec![0.0f64; n];
    for &s in &srcs {
        let dag = sp_dag(&adj, s);
        let mut delta = vec![0.0f64; n];
        for &w in dag.order.iter().rev() {
            let coeff = if is_target[w] && w != s {
                (delta[w] + 1.0) / dag.sigma[w]
            } else {
                delta[w] / dag.sigma[w]
            };
            for &v in &dag.preds[w] {
                delta[v] += dag.sigma[v] * coeff;
            }
            if w != s {
                bc[w] += delta[w];
            }
        }
    }
    let scale = if normalized {
        if n > 2 { 1.0 / ((n - 1) * (n - 2)) as f64 } else { 1.0 }
    } else if graph.is_directed() {
        1.0
    } else {
        0.5
    };
    Ok(ids.iter().enumerate().map(|(i, &id)| (id, bc[i] * scale)).collect())
}

// ---------------------------------------------------------------------------------------------
// Structural holes: effective size and local constraint
// ---------------------------------------------------------------------------------------------

/// Mutual weights `mw[u][v] = a_uv + a_vu` (no self-loops).
fn mutual_weights(graph: &Graph, idx: &FxHashMap<u64, usize>, weighted: bool) -> Vec<FxHashMap<usize, f64>> {
    let mut mw: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); idx.len()];
    let directed = graph.is_directed();
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let (s, t) = (idx[&e.source], idx[&e.target]);
        let w = weight_of(e, weighted);
        // undirected edges count in both directions (a_uv = a_vu = w)
        let add = if directed { w } else { 2.0 * w };
        *mw[s].entry(t).or_insert(0.0) += add;
        *mw[t].entry(s).or_insert(0.0) += add;
    }
    mw
}

/// Burt's effective size of every node's ego network: the number of contacts minus their
/// redundancy. Nodes without contacts get None. Matches NetworkX (self-loops ignored).
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn effective_size(graph: &Graph, weighted: bool) -> PyResult<FxHashMap<u64, Option<f64>>> {
    let (ids, idx) = sorted_ids(graph);
    let mw = mutual_weights(graph, &idx, weighted);
    let sums: Vec<f64> = mw.iter().map(|m| m.values().sum()).collect();
    let maxs: Vec<f64> = mw.iter().map(|m| m.values().copied().fold(0.0, f64::max)).collect();
    let mut out = FxHashMap::default();
    for v in 0..ids.len() {
        if mw[v].is_empty() {
            out.insert(ids[v], None);
            continue;
        }
        let mut size = 0.0;
        for &u in mw[v].keys() {
            // redundancy of contact u: Σ_w p_vw * m_uw over v's contacts w
            let r: f64 = mw[v]
                .iter()
                .map(|(&w, &vw)| {
                    let p_vw = vw / sums[v];
                    let m_uw = mw[u].get(&w).map_or(0.0, |&uw| if maxs[u] > 0.0 { uw / maxs[u] } else { 0.0 });
                    p_vw * m_uw
                })
                .sum();
            size += 1.0 - r;
        }
        out.insert(ids[v], Some(size));
    }
    Ok(out)
}

/// Burt's local constraint of `u` with respect to `v`: (p_uv + Σ_w p_uw p_wv)^2, where p is
/// the proportional (mutual) tie strength. Matches NetworkX.
#[pyfunction]
#[pyo3(signature = (graph, u, v, weighted=false))]
pub fn local_constraint(graph: &Graph, u: u64, v: u64, weighted: bool) -> PyResult<f64> {
    let (_, idx) = sorted_ids(graph);
    let a = *idx.get(&u).ok_or_else(|| err(format!("Node {} does not exist", u)))?;
    let b = *idx.get(&v).ok_or_else(|| err(format!("Node {} does not exist", v)))?;
    let mw = mutual_weights(graph, &idx, weighted);
    let p = |x: usize, y: usize| -> f64 {
        let total: f64 = mw[x].values().sum();
        if total == 0.0 { 0.0 } else { mw[x].get(&y).copied().unwrap_or(0.0) / total }
    };
    let direct = p(a, b);
    let indirect: f64 = mw[a].keys().map(|&w| p(a, w) * p(w, b)).sum();
    Ok((direct + indirect).powi(2))
}
