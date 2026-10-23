//! Bipartite graph algorithms.
//!
//! - `bipartite_sets`: Return the two node partitions of a bipartite graph
//! - `bipartite_projection`: Project onto one partition
//! - `bipartite_degree_centrality`: Degree centrality within a partition
//! - `bipartite_closeness_centrality`: Closeness centrality for bipartite graphs
//! - `bipartite_betweenness_centrality`: Betweenness centrality for bipartite graphs
//! - `maximal_extendability`: The largest k for which every k independent edges extend to a
//!   perfect matching

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Neighbors of every node with edge direction ignored, one entry per edge end.
///
/// Bipartiteness is a property of the underlying undirected graph, so the partitions and
/// everything built on them follow edges both ways. On undirected graphs this is the usual
/// adjacency: parallel edges repeat a neighbor and a self-loop lists the node twice (which
/// makes the graph non-bipartite, as it should). `Graph::neighbors()` alone would only follow
/// out-edges on directed graphs.
fn undirected_adjacency(graph: &Graph) -> FxHashMap<u64, Vec<u64>> {
    let mut adj: FxHashMap<u64, Vec<u64>> =
        graph.get_node_ids().into_iter().map(|id| (id, Vec::new())).collect();
    for edge in graph.get_edges_internal() {
        adj.entry(edge.source).or_default().push(edge.target);
        adj.entry(edge.target).or_default().push(edge.source);
    }
    adj
}

/// Return the two partitions of a bipartite graph using BFS 2-coloring.
///
/// Returns (set_0, set_1) where set_0 contains nodes colored 0 and
/// set_1 contains nodes colored 1. Raises an error if the graph is not bipartite.
#[pyfunction]
#[pyo3(signature = (graph,))]
pub fn bipartite_sets(graph: &Graph) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Color in ascending ID order so the numbering is deterministic: in every
    // connected component, partition 0 is the side holding its lowest node ID.
    // (get_node_ids() is hash-ordered, which made `which=0` effectively arbitrary.)
    let mut node_ids = graph.get_node_ids();
    node_ids.sort_unstable();
    let adj = undirected_adjacency(graph);
    let mut color: FxHashMap<u64, u8> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    let mut set_0: Vec<u64> = Vec::new();
    let mut set_1: Vec<u64> = Vec::new();

    for &start_node in &node_ids {
        if color.contains_key(&start_node) {
            continue;
        }

        let mut queue: Vec<u64> = vec![start_node];
        color.insert(start_node, 0);

        while let Some(current) = queue.pop() {
            let current_color = *color.get(&current).unwrap();
            let next_color = 1 - current_color;

            if let Some(neighbors) = adj.get(&current) {
                for &neighbor in neighbors {
                    if let Some(&neighbor_color) = color.get(&neighbor) {
                        if neighbor_color == current_color {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "Graph is not bipartite",
                            ));
                        }
                    } else {
                        color.insert(neighbor, next_color);
                        queue.push(neighbor);
                    }
                }
            }
        }
    }

    for (&node_id, &c) in &color {
        if c == 0 {
            set_0.push(node_id);
        } else {
            set_1.push(node_id);
        }
    }

    set_0.sort_unstable();
    set_1.sort_unstable();

    Ok((set_0, set_1))
}

/// Project a bipartite graph onto one partition.
///
/// Two nodes in the chosen partition become connected in the projection if they
/// share at least one neighbor in the other partition. Edge weights in the
/// projection equal the number of shared neighbors.
///
/// `which` selects which partition to project onto: 0 or 1 (as returned by
/// `bipartite_sets`).
#[pyfunction]
#[pyo3(signature = (graph, which=0))]
pub fn bipartite_projection(
    graph: &Graph,
    which: u8,
) -> PyResult<Vec<(u64, u64, f64)>> {
    if which > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "which must be 0 or 1",
        ));
    }

    let (set_0, set_1) = bipartite_sets(graph)?;

    let (project_set, other_set) = if which == 0 {
        (set_0, set_1)
    } else {
        (set_1, set_0)
    };

    let project_members: FxHashSet<u64> = project_set.iter().copied().collect();

    // For each node in the other partition, collect its neighbors in the project partition
    // then create edges between all pairs of those neighbors
    let mut edge_weights: FxHashMap<(u64, u64), f64> = FxHashMap::default();
    let adj = undirected_adjacency(graph);

    for &other_node in &other_set {
        if let Some(neighbors) = adj.get(&other_node) {
            let project_neighbors: Vec<u64> = neighbors
                .iter()
                .filter(|n| project_members.contains(n))
                .copied()
                .collect();

            for (i, &a) in project_neighbors.iter().enumerate() {
                for &b in project_neighbors.iter().skip(i + 1) {
                    let edge = if a < b { (a, b) } else { (b, a) };
                    *edge_weights.entry(edge).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    let mut result: Vec<(u64, u64, f64)> = edge_weights
        .into_iter()
        .map(|((a, b), w)| (a, b, w))
        .collect();
    result.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    Ok(result)
}

/// Degree centrality for nodes in one partition of a bipartite graph.
///
/// Normalized by the size of the opposite partition (the maximum possible degree).
#[pyfunction]
#[pyo3(signature = (graph, which=0))]
pub fn bipartite_degree_centrality(
    graph: &Graph,
    which: u8,
) -> PyResult<FxHashMap<u64, f64>> {
    if which > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "which must be 0 or 1",
        ));
    }

    let (set_0, set_1) = bipartite_sets(graph)?;

    let (target_set, other_size) = if which == 0 {
        let other_len = set_1.len();
        (set_0, other_len)
    } else {
        let other_len = set_0.len();
        (set_1, other_len)
    };

    if other_size == 0 {
        return Ok(target_set.iter().map(|&id| (id, 0.0)).collect());
    }

    let norm = 1.0 / other_size as f64;
    let mut result = FxHashMap::default();
    let adj = undirected_adjacency(graph);

    for &node_id in &target_set {
        let degree = adj.get(&node_id).map(|n| n.len()).unwrap_or(0);
        result.insert(node_id, degree as f64 * norm);
    }

    Ok(result)
}

/// Hop distances from `source` to every reachable node.
///
/// Must be a FIFO queue: each node's distance is fixed when it is first discovered, which
/// is only the shortest distance if nodes are expanded in breadth-first order. (A LIFO
/// stack explores depth-first and can overstate distances on graphs with cycles.)
fn bfs_distances(adj: &FxHashMap<u64, Vec<u64>>, source: u64) -> FxHashMap<u64, f64> {
    let mut dist: FxHashMap<u64, f64> = FxHashMap::default();
    dist.insert(source, 0.0);
    let mut queue: std::collections::VecDeque<u64> = std::collections::VecDeque::from([source]);

    while let Some(current) = queue.pop_front() {
        let current_dist = dist[&current];
        if let Some(neighbors) = adj.get(&current) {
            for &neighbor in neighbors {
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(neighbor) {
                    e.insert(current_dist + 1.0);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    dist
}

/// Closeness centrality for nodes in one partition of a bipartite graph.
///
/// Closeness is the reciprocal of the average shortest path distance to all
/// nodes in the opposite partition.
#[pyfunction]
#[pyo3(signature = (graph, which=0))]
pub fn bipartite_closeness_centrality(
    graph: &Graph,
    which: u8,
) -> PyResult<FxHashMap<u64, f64>> {
    if which > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "which must be 0 or 1",
        ));
    }

    let (set_0, set_1) = bipartite_sets(graph)?;

    let (target_set, other_set) = if which == 0 {
        (set_0, set_1)
    } else {
        (set_1, set_0)
    };

    let other_members: FxHashSet<u64> = other_set.iter().copied().collect();
    let other_size = other_set.len();

    if other_size == 0 {
        return Ok(target_set.iter().map(|&id| (id, 0.0)).collect());
    }

    let mut result = FxHashMap::default();
    let adj = undirected_adjacency(graph);

    for &node_id in &target_set {
        let distances = bfs_distances(&adj, node_id);
        let total_dist: f64 = other_members
            .iter()
            .filter_map(|&other| distances.get(&other))
            .sum();

        let reachable: usize = other_members
            .iter()
            .filter(|&&other| distances.contains_key(&other))
            .count();

        let centrality = if reachable > 0 && total_dist > 0.0 {
            reachable as f64 / total_dist
        } else {
            0.0
        };

        result.insert(node_id, centrality);
    }

    Ok(result)
}

/// Betweenness centrality for nodes in one partition of a bipartite graph.
///
/// Measures how often a node lies on shortest paths between pairs of nodes
/// in the opposite partition.
#[pyfunction]
#[pyo3(signature = (graph, which=0))]
pub fn bipartite_betweenness_centrality(
    graph: &Graph,
    which: u8,
) -> PyResult<FxHashMap<u64, f64>> {
    if which > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "which must be 0 or 1",
        ));
    }

    let (set_0, set_1) = bipartite_sets(graph)?;

    let (target_set, other_set) = if which == 0 {
        (set_0, set_1)
    } else {
        (set_1, set_0)
    };

    let target_members: FxHashSet<u64> = target_set.iter().copied().collect();

    let mut betweenness: FxHashMap<u64, f64> = target_set.iter().map(|&id| (id, 0.0)).collect();
    let adj = undirected_adjacency(graph);

    // Brandes-like algorithm: BFS from each node in the other partition
    for &source in &other_set {
        // BFS
        let mut dist: FxHashMap<u64, i64> = FxHashMap::default();
        let mut sigma: FxHashMap<u64, f64> = FxHashMap::default();
        let mut predecessors: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        let mut stack: Vec<u64> = Vec::new();

        dist.insert(source, 0);
        sigma.insert(source, 1.0);

        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let v_dist = dist[&v];

            if let Some(neighbors) = adj.get(&v) {
                for &w in neighbors {
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                        e.insert(v_dist + 1);
                        queue.push_back(w);
                    }
                    if dist[&w] == v_dist + 1 {
                        *sigma.entry(w).or_insert(0.0) += sigma.get(&v).copied().unwrap_or(0.0);
                        predecessors.entry(w).or_default().push(v);
                    }
                }
            }
        }

        // Back-propagation
        let mut delta: FxHashMap<u64, f64> = FxHashMap::default();

        while let Some(w) = stack.pop() {
            let w_delta = delta.get(&w).copied().unwrap_or(0.0);
            let w_sigma = sigma.get(&w).copied().unwrap_or(1.0);

            if let Some(preds) = predecessors.get(&w) {
                for &v in preds {
                    let v_sigma = sigma.get(&v).copied().unwrap_or(1.0);
                    let contribution = (v_sigma / w_sigma) * (1.0 + w_delta);
                    *delta.entry(v).or_insert(0.0) += contribution;
                }
            }

            if w != source {
                if let Some(b) = betweenness.get_mut(&w) {
                    if target_members.contains(&w) {
                        *b += w_delta;
                    }
                }
            }
        }
    }

    // Normalize: for undirected graphs, divide by 2
    let norm = if other_set.len() >= 2 {
        2.0 / ((other_set.len() as f64) * (other_set.len() as f64 - 1.0))
    } else {
        1.0
    };

    for val in betweenness.values_mut() {
        *val *= norm;
    }

    Ok(betweenness)
}


fn value_err<T>(msg: &str) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.to_string()))
}

/// Unit-capacity flow network for counting internally node-disjoint paths: node `x` is split
/// into `2x` (in) and `2x + 1` (out) joined by an arc of capacity 1, and each arc `x -> y`
/// becomes `out(x) -> in(y)` with capacity 1. Arc `k`'s reverse is `k ^ 1`.
struct SplitNetwork {
    head: Vec<Vec<usize>>,
    to: Vec<usize>,
    cap: Vec<u8>,
}

impl SplitNetwork {
    fn new(n: usize, arcs: &[(usize, usize)]) -> Self {
        let mut net = SplitNetwork { head: vec![Vec::new(); 2 * n], to: Vec::new(), cap: Vec::new() };
        for x in 0..n {
            net.add(2 * x, 2 * x + 1);
        }
        for &(x, y) in arcs {
            net.add(2 * x + 1, 2 * y);
        }
        net
    }

    fn add(&mut self, a: usize, b: usize) {
        self.head[a].push(self.to.len());
        self.to.push(b);
        self.cap.push(1);
        self.head[b].push(self.to.len());
        self.to.push(a);
        self.cap.push(0);
    }

    /// Up to `limit` internally node-disjoint paths from `s` to `t` (augmenting along BFS paths).
    fn disjoint_paths(&self, s: usize, t: usize, limit: usize) -> usize {
        let (source, sink) = (2 * s + 1, 2 * t);
        let mut cap = self.cap.clone();
        let mut via = vec![usize::MAX; self.head.len()];
        let mut flow = 0;
        while flow < limit {
            via.fill(usize::MAX);
            via[source] = 0;
            let mut queue = std::collections::VecDeque::from([source]);
            'search: while let Some(x) = queue.pop_front() {
                for &k in &self.head[x] {
                    let y = self.to[k];
                    if cap[k] > 0 && via[y] == usize::MAX {
                        via[y] = k;
                        if y == sink {
                            break 'search;
                        }
                        queue.push_back(y);
                    }
                }
            }
            if via[sink] == usize::MAX {
                break;
            }
            let mut y = sink;
            while y != source {
                let k = via[y];
                cap[k] -= 1;
                cap[k ^ 1] += 1;
                y = self.to[k ^ 1];
            }
            flow += 1;
        }
        flow
    }
}

/// Extendability of a connected bipartite graph with a perfect matching: the largest `k` such
/// that every set of `k` independent edges extends to a perfect matching (NetworkX's
/// `bipartite.maximal_extendability`).
///
/// Following Lakhal and Litzler (1998): take a perfect matching M of the sides U and V, and
/// direct the edges of M from V to U and the others from U to V. The answer is 0 if that
/// residual graph is not strongly connected (NetworkX raises an error there instead), and
/// otherwise the fewest internally node-disjoint directed paths from a node of U to a node of
/// V. Direction, parallel edges and edge weights are ignored.
#[pyfunction]
pub fn maximal_extendability(graph: &Graph) -> PyResult<usize> {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let n = ids.len();
    if n == 0 {
        return value_err("The graph has no nodes");
    }
    let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut pairs: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut has_loop = false;
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        has_loop |= a == b;
        if a != b && pairs.insert((a.min(b), a.max(b))) {
            adj[a].push(b);
            adj[b].push(a);
        }
    }

    // connected, then two-colorable
    let mut side = vec![u8::MAX; n];
    side[0] = 0;
    let mut queue = std::collections::VecDeque::from([0usize]);
    let mut odd = has_loop;
    let mut reached = 1;
    while let Some(x) = queue.pop_front() {
        for &y in &adj[x] {
            if side[y] == u8::MAX {
                side[y] = 1 - side[x];
                reached += 1;
                queue.push_back(y);
            } else if side[y] == side[x] {
                odd = true;
            }
        }
    }
    if reached < n {
        return value_err("The graph is not connected");
    }
    if odd {
        return value_err("The graph is not bipartite");
    }

    // a maximum matching, by augmenting paths from each node of U (Kuhn's algorithm)
    let u_side: Vec<usize> = (0..n).filter(|&x| side[x] == 0).collect();
    let v_side: Vec<usize> = (0..n).filter(|&x| side[x] == 1).collect();
    let mut mate = vec![usize::MAX; n];
    fn augment(x: usize, adj: &[Vec<usize>], mate: &mut [usize], used: &mut [bool]) -> bool {
        for &y in &adj[x] {
            if !used[y] {
                used[y] = true;
                if mate[y] == usize::MAX || augment(mate[y], adj, mate, used) {
                    mate[y] = x;
                    mate[x] = y;
                    return true;
                }
            }
        }
        false
    }
    let mut matched = 0;
    for &x in &u_side {
        let mut used = vec![false; n];
        if augment(x, &adj, &mut mate, &mut used) {
            matched += 1;
        }
    }
    if u_side.len() != v_side.len() || matched != u_side.len() {
        return value_err("The graph has no perfect matching");
    }

    // the residual graph: matching edges V -> U, the others U -> V
    let mut arcs = Vec::new();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut inn: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &x in &u_side {
        for &y in &adj[x] {
            let (p, q) = if mate[x] == y { (y, x) } else { (x, y) };
            arcs.push((p, q));
            out[p].push(q);
            inn[q].push(p);
        }
    }
    let reaches_all = |nbrs: &[Vec<usize>]| {
        let mut seen = vec![false; n];
        seen[0] = true;
        let mut stack = vec![0usize];
        let mut count = 1;
        while let Some(x) = stack.pop() {
            for &y in &nbrs[x] {
                if !seen[y] {
                    seen[y] = true;
                    count += 1;
                    stack.push(y);
                }
            }
        }
        count == n
    };
    if !reaches_all(&out) || !reaches_all(&inn) {
        return Ok(0);
    }

    let net = SplitNetwork::new(n, &arcs);
    let mut k = u_side.iter().map(|&x| out[x].len()).min().unwrap_or(0);
    for &s in &u_side {
        for &t in &v_side {
            if k == 0 {
                return Ok(0);
            }
            k = k.min(net.disjoint_paths(s, t, k));
        }
    }
    Ok(k)
}
