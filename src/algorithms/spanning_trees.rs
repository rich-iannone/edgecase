//! Spanning tree algorithms for graphs
//!
//! This module provides algorithms for finding spanning trees:
//!
//! - `steiner_tree`: Find an approximate minimum Steiner tree connecting terminal nodes
//! - `all_spanning_trees`: Generate all spanning trees of a connected graph

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Entry for priority queue in Dijkstra's algorithm
#[derive(Clone)]
struct DijkstraEntry {
    distance: f64,
    node: u64,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node == other.node
    }
}

impl Eq for DijkstraEntry {}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (smaller distances have higher priority)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Union-Find data structure for cycle detection
struct UnionFind {
    parent: FxHashMap<u64, u64>,
    rank: FxHashMap<u64, usize>,
}

impl UnionFind {
    fn new() -> Self {
        UnionFind {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
        }
    }

    fn find(&mut self, mut x: u64) -> u64 {
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(x) {
            e.insert(x);
            self.rank.insert(x, 0);
            return x;
        }

        // Path compression
        while self.parent[&x] != x {
            let parent = self.parent[&x];
            let grandparent = *self.parent.get(&parent).unwrap_or(&parent);
            self.parent.insert(x, grandparent);
            x = parent;
        }
        x
    }

    fn union(&mut self, a: u64, b: u64) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);

        if ra == rb {
            return false;
        }

        let rank_a = self.rank[&ra];
        let rank_b = self.rank[&rb];

        if rank_a < rank_b {
            self.parent.insert(ra, rb);
        } else if rank_a > rank_b {
            self.parent.insert(rb, ra);
        } else {
            self.parent.insert(rb, ra);
            self.rank.insert(ra, rank_a + 1);
        }

        true
    }
}

/// Get edge weight from graph, optionally using custom attribute
fn get_edge_weight(graph: &Graph, edge_id: u64, weight_attr: Option<&str>) -> f64 {
    if let Some(edge) = graph.get_edge(edge_id) {
        if let Some(attr_name) = weight_attr {
            // Try to get weight from attributes with GIL
            Python::with_gil(|py| {
                if let Some(attrs) = &edge.attributes {
                    let attrs_bound = attrs.bind(py);
                    // First try direct access
                    if let Ok(Some(val)) = attrs_bound.get_item(attr_name) {
                        if let Ok(f) = val.extract::<f64>() {
                            return f;
                        }
                    }
                    // Then try nested under "attributes" key (EdgeCase convention)
                    if let Ok(Some(nested)) = attrs_bound.get_item("attributes") {
                        // Use downcast_bound to get it as a dict and call get_item on that
                        if let Ok(nested_dict) = nested.downcast::<pyo3::types::PyDict>() {
                            if let Ok(Some(val)) = nested_dict.get_item(attr_name) {
                                if let Ok(f) = val.extract::<f64>() {
                                    return f;
                                }
                            }
                        }
                    }
                }
                1.0
            })
        } else {
            // Use default weight field
            edge.weight.unwrap_or(1.0)
        }
    } else {
        1.0
    }
}

/// Run Dijkstra's algorithm from a source node to find shortest paths.
/// The lowest-cost edge joining `a` and `b` (either orientation on undirected graphs), with
/// its cost under `cost`. Ties go to the earliest edge. With parallel edges this picks the
/// cheapest one; the first one found used to be taken regardless of its weight.
fn cheapest_edge(graph: &Graph, a: u64, b: u64, cost: impl Fn(u64) -> f64) -> Option<(u64, f64)> {
    let mut best: Option<(u64, f64)> = None;
    for edge_id in graph.find_edge_ids(a, b) {
        let c = cost(edge_id);
        if best.is_none_or(|(_, best_cost)| c < best_cost) {
            best = Some((edge_id, c));
        }
    }
    best
}

/// Uses the specified weight attribute or default weights.
fn dijkstra_single_source(
    graph: &Graph,
    source: u64,
    weight_attr: Option<&str>,
) -> PyResult<(FxHashMap<u64, f64>, FxHashMap<u64, u64>)> {
    let mut distances: FxHashMap<u64, f64> = FxHashMap::default();
    let mut predecessors: FxHashMap<u64, u64> = FxHashMap::default();
    let mut heap = BinaryHeap::new();

    distances.insert(source, 0.0);
    heap.push(DijkstraEntry {
        distance: 0.0,
        node: source,
    });

    while let Some(DijkstraEntry { distance, node }) = heap.pop() {
        // Skip if we've found a better path
        if let Some(&best) = distances.get(&node) {
            if distance > best {
                continue;
            }
        }

        // Get neighbors using public API
        let neighbors = graph.neighbors(node)?;

        for neighbor in neighbors {
            // Cheapest edge between node and neighbor
            let Some((_, edge_weight)) =
                cheapest_edge(graph, node, neighbor, |eid| get_edge_weight(graph, eid, weight_attr))
            else {
                continue;
            };

            let new_dist = distance + edge_weight;

            let current_dist = distances.get(&neighbor).copied().unwrap_or(f64::INFINITY);

            if new_dist < current_dist {
                distances.insert(neighbor, new_dist);
                predecessors.insert(neighbor, node);
                heap.push(DijkstraEntry {
                    distance: new_dist,
                    node: neighbor,
                });
            }
        }
    }

    Ok((distances, predecessors))
}

/// Reconstruct path from source to target using predecessor map
fn reconstruct_path(predecessors: &FxHashMap<u64, u64>, source: u64, target: u64) -> Vec<u64> {
    let mut path = Vec::new();
    let mut current = target;

    while current != source {
        path.push(current);
        if let Some(&prev) = predecessors.get(&current) {
            current = prev;
        } else {
            return Vec::new(); // No path
        }
    }
    path.push(source);
    path.reverse();
    path
}

/// Helper to get label from node ID, with fallback to ID string
fn get_label_or_id(graph: &Graph, node_id: u64) -> String {
    graph
        .label_from_id(node_id)
        .ok()
        .flatten()
        .unwrap_or_else(|| node_id.to_string())
}

/// Find an approximate Steiner tree connecting terminal nodes.
///
/// Uses Mehlhorn's 2-approximation algorithm:
/// 1. Compute shortest paths between all pairs of terminal nodes
/// 2. Build metric closure (complete graph on terminals with shortest path weights)
/// 3. Find MST of metric closure
/// 4. Expand MST edges to actual paths in original graph
/// 5. Find MST of resulting subgraph to remove redundant edges
///
/// # Arguments
/// * `graph` - The input graph
/// * `terminal_nodes` - Labels of nodes that must be connected
/// * `weight_attr` - Optional name of edge attribute to use for weights
///
/// # Returns
/// List of (source_label, target_label, weight) tuples forming the Steiner tree
///
/// # Raises
/// * `PyValueError` if fewer than 2 terminal nodes provided
/// * `PyValueError` if any terminal node doesn't exist
/// * `PyValueError` if terminal nodes are not connected
#[pyfunction]
#[pyo3(signature = (graph, terminal_nodes, weight_attr=None))]
pub fn steiner_tree(
    graph: &Graph,
    terminal_nodes: Vec<String>,
    weight_attr: Option<String>,
) -> PyResult<Vec<(String, String, f64)>> {
    // Validate inputs
    if terminal_nodes.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "At least 2 terminal nodes are required",
        ));
    }

    // Convert labels to node IDs
    let mut terminal_ids: Vec<u64> = Vec::new();
    for label in &terminal_nodes {
        match graph.id_from_label(label) {
            Ok(node_id) => terminal_ids.push(node_id),
            Err(_) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Terminal node '{}' does not exist in the graph",
                    label
                )));
            }
        }
    }

    let weight_attr_ref = weight_attr.as_deref();

    // Special case: if only 2 terminals, just find shortest path
    if terminal_ids.len() == 2 {
        let (distances, predecessors) =
            dijkstra_single_source(graph, terminal_ids[0], weight_attr_ref)?;

        if !distances.contains_key(&terminal_ids[1]) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Terminal nodes are not connected",
            ));
        }

        let path = reconstruct_path(&predecessors, terminal_ids[0], terminal_ids[1]);
        return path_to_edge_tuples(graph, &path, weight_attr_ref);
    }

    // Step 1: Compute shortest paths from each terminal to all other nodes
    let mut terminal_distances: FxHashMap<u64, FxHashMap<u64, f64>> = FxHashMap::default();
    let mut terminal_predecessors: FxHashMap<u64, FxHashMap<u64, u64>> = FxHashMap::default();

    for &terminal in &terminal_ids {
        let (distances, predecessors) = dijkstra_single_source(graph, terminal, weight_attr_ref)?;
        terminal_distances.insert(terminal, distances);
        terminal_predecessors.insert(terminal, predecessors);
    }

    // Step 2: Build complete graph on terminals (metric closure)
    let mut metric_edges: Vec<(f64, u64, u64)> = Vec::new();

    for (i, &t1) in terminal_ids.iter().enumerate() {
        for &t2 in terminal_ids.iter().skip(i + 1) {
            if let Some(dist) = terminal_distances.get(&t1).and_then(|d| d.get(&t2)) {
                metric_edges.push((*dist, t1, t2));
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Terminal nodes are not connected",
                ));
            }
        }
    }

    // Step 3: Find MST of metric closure using Kruskal's algorithm
    metric_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut uf = UnionFind::new();
    let mut mst_edges: Vec<(u64, u64)> = Vec::new();
    let edges_needed = terminal_ids.len() - 1;

    for (_, t1, t2) in metric_edges {
        if uf.union(t1, t2) {
            mst_edges.push((t1, t2));
            if mst_edges.len() == edges_needed {
                break;
            }
        }
    }

    // Step 4: Expand MST edges to actual paths in original graph
    let mut steiner_edges: FxHashSet<(u64, u64)> = FxHashSet::default();

    for (t1, t2) in mst_edges {
        if let Some(predecessors) = terminal_predecessors.get(&t1) {
            let path = reconstruct_path(predecessors, t1, t2);
            for window in path.windows(2) {
                let (a, b) = (window[0], window[1]);
                let edge = if a < b { (a, b) } else { (b, a) };
                steiner_edges.insert(edge);
            }
        }
    }

    // Step 5: Build subgraph and find its MST to remove redundant edges
    let steiner_nodes: FxHashSet<u64> = steiner_edges
        .iter()
        .flat_map(|&(a, b)| vec![a, b])
        .collect();

    // Collect edges with weights for final MST
    let mut final_edges: Vec<(f64, u64, u64, u64)> = Vec::new(); // (weight, edge_id, source, target)

    for &(a, b) in &steiner_edges {
        let cheapest =
            cheapest_edge(graph, a, b, |eid| get_edge_weight(graph, eid, weight_attr_ref));
        if let Some((edge_id, weight)) = cheapest {
            if let Some(edge) = graph.get_edge(edge_id) {
                final_edges.push((weight, edge_id, edge.source, edge.target));
            }
        }
    }

    // Run Kruskal on the subgraph
    final_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut final_uf = UnionFind::new();
    let mut result_edges: Vec<(String, String, f64)> = Vec::new();
    let final_edges_needed = steiner_nodes.len().saturating_sub(1);

    for (weight, _edge_id, source, target) in final_edges {
        if final_uf.union(source, target) {
            let source_label = get_label_or_id(graph, source);
            let target_label = get_label_or_id(graph, target);
            result_edges.push((source_label, target_label, weight));
            if result_edges.len() == final_edges_needed {
                break;
            }
        }
    }

    Ok(result_edges)
}

/// Convert a path (sequence of node IDs) to edge tuples with labels
fn path_to_edge_tuples(
    graph: &Graph,
    path: &[u64],
    weight_attr: Option<&str>,
) -> PyResult<Vec<(String, String, f64)>> {
    let mut edges = Vec::new();

    for window in path.windows(2) {
        let (a, b) = (window[0], window[1]);
        let source_label = get_label_or_id(graph, a);
        let target_label = get_label_or_id(graph, b);

        // Cheapest edge's weight
        let weight = cheapest_edge(graph, a, b, |eid| get_edge_weight(graph, eid, weight_attr))
            .map_or(1.0, |(_, w)| w);

        edges.push((source_label, target_label, weight));
    }

    Ok(edges)
}

/// Generate all spanning trees of a connected graph.
///
/// Uses a recursive enumeration algorithm that systematically generates
/// all spanning trees by edge selection.
///
/// # Arguments
/// * `graph` - The input graph (must be connected)
/// * `max_trees` - Maximum number of spanning trees to generate
/// * `weight_attr` - Optional name of edge attribute to use for weights
///
/// # Returns
/// List of spanning trees, where each tree is a list of (source_label, target_label, weight) tuples
///
/// # Raises
/// * `PyValueError` if graph is not connected
#[pyfunction]
#[pyo3(signature = (graph, max_trees, weight_attr=None))]
pub fn all_spanning_trees(
    graph: &Graph,
    max_trees: usize,
    weight_attr: Option<String>,
) -> PyResult<Vec<Vec<(String, String, f64)>>> {
    let num_nodes = graph.num_nodes();

    // Handle edge cases
    if num_nodes == 0 {
        return Ok(vec![Vec::new()]);
    }

    if num_nodes == 1 {
        return Ok(vec![Vec::new()]);
    }

    // Check connectivity
    let node_ids: Vec<u64> = graph.get_node_ids();
    let start_node = node_ids[0];
    let mut visited: FxHashSet<u64> = FxHashSet::default();
    let mut stack = vec![start_node];

    while let Some(node) = stack.pop() {
        if visited.insert(node) {
            if let Ok(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }

    if visited.len() != num_nodes {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Graph is not connected; no spanning tree exists",
        ));
    }

    let weight_attr_ref = weight_attr.as_deref();

    // Get edges using public API
    let all_edges: Vec<(u64, u64, u64)> = graph.edges();

    // For undirected graphs, remove duplicate edges
    let is_directed = graph.is_directed();
    let mut seen_edges: FxHashSet<(u64, u64)> = FxHashSet::default();
    let mut unique_edges: Vec<(u64, u64, u64)> = Vec::new();

    for (edge_id, source, target) in all_edges {
        if is_directed {
            unique_edges.push((edge_id, source, target));
        } else {
            let (a, b) = if source < target {
                (source, target)
            } else {
                (target, source)
            };
            if seen_edges.insert((a, b)) {
                unique_edges.push((edge_id, a, b));
            }
        }
    }

    let edges_needed = num_nodes - 1;

    if unique_edges.len() < edges_needed {
        return Ok(Vec::new());
    }

    // Use iterative approach to enumerate spanning trees
    let mut results: Vec<Vec<(String, String, f64)>> = Vec::new();

    // State: (current tree edge indices, index to start from)
    let mut stack_states: Vec<(Vec<usize>, usize)> = vec![(Vec::new(), 0)];

    while let Some((current_tree, start_idx)) = stack_states.pop() {
        if results.len() >= max_trees {
            break;
        }

        if current_tree.len() == edges_needed {
            // Found a spanning tree - convert to output format
            let mut tree_edges: Vec<(String, String, f64)> = Vec::new();
            for &idx in &current_tree {
                let (edge_id, source, target) = unique_edges[idx];
                let source_label = get_label_or_id(graph, source);
                let target_label = get_label_or_id(graph, target);
                let weight = get_edge_weight(graph, edge_id, weight_attr_ref);
                tree_edges.push((source_label, target_label, weight));
            }
            results.push(tree_edges);
            continue;
        }

        // Try adding each remaining edge
        for i in start_idx..unique_edges.len() {
            // Check if adding this edge would create a cycle
            let mut test_tree = current_tree.clone();
            test_tree.push(i);

            if !would_create_cycle(&test_tree, &unique_edges) {
                stack_states.push((test_tree, i + 1));
            }
        }
    }

    Ok(results)
}

/// Check if adding edges would create a cycle using union-find
fn would_create_cycle(edge_indices: &[usize], edges: &[(u64, u64, u64)]) -> bool {
    let mut uf = UnionFind::new();

    for &idx in edge_indices {
        let (_, a, b) = edges[idx];
        if !uf.union(a, b) {
            return true; // Would create cycle
        }
    }

    false
}

/// Prize-collecting Steiner tree.
///
/// A variant of the Steiner tree where each node has a *prize* (benefit of
/// inclusion) and each edge has a *cost* (penalty for inclusion). The goal is
/// to find a connected subtree maximizing total prize minus total edge cost.
/// Not all prize nodes need to be included if the connecting cost is too high.
///
/// Uses a greedy heuristic:
/// 1. Start from the highest-prize node.
/// 2. Repeatedly attach the node with the best prize/cost ratio via its
///    shortest path to the current tree.
/// 3. Stop when no remaining node has a positive net benefit.
///
/// Returns edge tuples (source_label, target_label, weight) of the result tree.
#[pyfunction]
#[pyo3(signature = (graph, prizes, edge_penalty=1.0, weight_attr=None))]
pub fn prize_collecting_steiner_tree(
    graph: &Graph,
    prizes: FxHashMap<String, f64>,
    edge_penalty: f64,
    weight_attr: Option<String>,
) -> PyResult<Vec<(String, String, f64)>> {
    if prizes.is_empty() {
        return Ok(Vec::new());
    }

    let weight_attr_ref = weight_attr.as_deref();

    // Convert label-keyed prizes to id-keyed
    let mut prize_by_id: FxHashMap<u64, f64> = FxHashMap::default();
    for (label, &prize) in &prizes {
        match graph.id_from_label(label) {
            Ok(node_id) => {
                prize_by_id.insert(node_id, prize);
            }
            Err(_) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node '{}' does not exist in the graph",
                    label
                )));
            }
        }
    }

    // Start from the highest-prize node
    let &start_node = prize_by_id
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(id, _)| id)
        .unwrap();

    let mut tree_nodes: FxHashSet<u64> = FxHashSet::default();
    tree_nodes.insert(start_node);

    let mut tree_edges: Vec<(u64, u64, f64)> = Vec::new();
    let mut remaining: FxHashSet<u64> = prize_by_id
        .keys()
        .filter(|&&id| id != start_node)
        .copied()
        .collect();

    // Greedy expansion
    loop {
        if remaining.is_empty() {
            break;
        }

        // Find shortest path from any tree node to any remaining prize node
        let mut best_target: Option<u64> = None;
        let mut best_path: Option<Vec<u64>> = None;
        let mut best_cost: f64 = f64::INFINITY;
        let mut best_benefit: f64 = f64::NEG_INFINITY;

        for &tree_node in &tree_nodes {
            let (distances, predecessors) =
                dijkstra_single_source(graph, tree_node, weight_attr_ref)?;

            for &candidate in &remaining {
                if let Some(&dist) = distances.get(&candidate) {
                    let cost = dist * edge_penalty;
                    let prize = prize_by_id.get(&candidate).copied().unwrap_or(0.0);
                    let benefit = prize - cost;

                    if benefit > best_benefit
                        || (benefit == best_benefit && cost < best_cost)
                    {
                        best_benefit = benefit;
                        best_cost = cost;
                        best_target = Some(candidate);
                        best_path =
                            Some(reconstruct_path(&predecessors, tree_node, candidate));
                    }
                }
            }
        }

        // Stop if no remaining node has positive benefit
        if best_benefit <= 0.0 || best_target.is_none() || best_path.is_none() {
            break;
        }

        let path = best_path.unwrap();
        let target = best_target.unwrap();

        // Add path edges and intermediate nodes to the tree
        for window in path.windows(2) {
            let (a, b) = (window[0], window[1]);
            tree_nodes.insert(a);
            tree_nodes.insert(b);

            let weight =
                cheapest_edge(graph, a, b, |eid| get_edge_weight(graph, eid, weight_attr_ref))
                    .map_or(1.0, |(_, w)| w);
            tree_edges.push((a, b, weight));
        }

        remaining.remove(&target);
    }

    // Deduplicate edges and convert to labels
    let mut seen: FxHashSet<(u64, u64)> = FxHashSet::default();
    let mut result: Vec<(String, String, f64)> = Vec::new();

    for (a, b, w) in tree_edges {
        let edge = if a < b { (a, b) } else { (b, a) };
        if seen.insert(edge) {
            let source_label = get_label_or_id(graph, a);
            let target_label = get_label_or_id(graph, b);
            result.push((source_label, target_label, w));
        }
    }

    Ok(result)
}

/// Expand a seed set to a connected module using neighborhood scoring.
///
/// Starting from seed nodes, greedily adds neighbors that are well-connected
/// to the current module. The expansion stops when no neighbor exceeds the
/// score threshold or the module reaches `max_size`.
///
/// Score for a candidate node = (edges to current module) / (total degree).
/// This favors nodes that are densely connected to the module rather than
/// high-degree hub nodes.
#[pyfunction]
#[pyo3(signature = (graph, seeds, max_size=None, min_score=0.0))]
pub fn expand_seed_set(
    graph: &Graph,
    seeds: Vec<String>,
    max_size: Option<usize>,
    min_score: f64,
) -> PyResult<Vec<u64>> {
    if seeds.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seeds must contain at least one node",
        ));
    }

    let mut module: FxHashSet<u64> = FxHashSet::default();

    for label in &seeds {
        match graph.id_from_label(label) {
            Ok(node_id) => {
                module.insert(node_id);
            }
            Err(_) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Seed node '{}' does not exist in the graph",
                    label
                )));
            }
        }
    }

    let max = max_size.unwrap_or(module.len() * 3);

    loop {
        if module.len() >= max {
            break;
        }

        // Score all candidate neighbors
        let mut candidates: FxHashMap<u64, (usize, usize)> = FxHashMap::default();

        for &node_id in &module {
            if let Ok(neighbors) = graph.neighbors(node_id) {
                for &neighbor in &neighbors {
                    if !module.contains(&neighbor) {
                        let entry = candidates.entry(neighbor).or_insert((0, 0));
                        entry.0 += 1; // edges to module
                    }
                }
            }
        }

        // Compute total degree for each candidate
        for (&candidate, entry) in candidates.iter_mut() {
            entry.1 = graph.neighbors(candidate).map(|n| n.len()).unwrap_or(0);
        }

        // Find best candidate
        let best = candidates
            .iter()
            .map(|(&node, &(module_edges, total_degree))| {
                let score = if total_degree > 0 {
                    module_edges as f64 / total_degree as f64
                } else {
                    0.0
                };
                (node, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        match best {
            Some((node, score)) if score > min_score => {
                module.insert(node);
            }
            _ => break,
        }
    }

    let mut result: Vec<u64> = module.into_iter().collect();
    result.sort_unstable();
    Ok(result)
}

/// Find an approximate directed Steiner tree (Steiner arborescence).
///
/// Given a directed graph, a root node, and a set of terminal nodes, finds an
/// approximate minimum-weight arborescence rooted at `root` that contains a
/// directed path from `root` to every terminal.
///
/// Uses the Shortest-Path Heuristic (SPH) with edge-cost reuse: greedily adds
/// cheapest paths from root to uncovered terminals, then zeroes the cost of
/// used edges so subsequent paths can reuse them for free. This encourages
/// path sharing and produces compact arborescences.
///
/// Reference: Charikar, Chekuri, Cheung, Dai, Goel, Guha, Li (1999).
/// "Approximation Algorithms for Directed Steiner Problems."
/// Journal of Algorithms, 33(1):73-91. (Level-1 variant)
///
/// # Arguments
/// * `graph` - A directed graph
/// * `root` - Label of the root node
/// * `terminal_nodes` - Labels of target nodes to reach from root
/// * `weight_attr` - Optional edge attribute for weights (default: edge.weight or 1.0)
///
/// # Returns
/// List of (source_label, target_label, weight) tuples forming the arborescence
#[pyfunction]
#[pyo3(signature = (graph, root, terminal_nodes, weight_attr=None))]
pub fn directed_steiner_tree(
    graph: &Graph,
    root: String,
    terminal_nodes: Vec<String>,
    weight_attr: Option<String>,
) -> PyResult<Vec<(String, String, f64)>> {
    if !graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "directed_steiner_tree requires a directed graph; use steiner_tree for undirected",
        ));
    }

    if terminal_nodes.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "At least 1 terminal node is required",
        ));
    }

    let root_id = graph.id_from_label(&root).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("Root node '{}' not found", root))
    })?;

    let mut terminal_ids: Vec<u64> = Vec::with_capacity(terminal_nodes.len());
    for label in &terminal_nodes {
        let nid = graph.id_from_label(label).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!("Terminal node '{}' not found", label))
        })?;
        if nid == root_id {
            continue; // root reaching itself is trivial
        }
        terminal_ids.push(nid);
    }

    if terminal_ids.is_empty() {
        return Ok(Vec::new());
    }

    terminal_ids.sort_unstable();
    terminal_ids.dedup();

    let weight_attr_ref = weight_attr.as_deref();

    // Edge cost overrides: edge_id -> 0.0 (for already-used edges)
    let mut cost_overrides: FxHashMap<u64, f64> = FxHashMap::default();

    // Collected arborescence edges: (source_id, target_id, edge_id)
    let mut tree_edges: FxHashSet<u64> = FxHashSet::default();
    // Parent map for the arborescence
    let mut tree_parent: FxHashMap<u64, u64> = FxHashMap::default();

    let mut uncovered: FxHashSet<u64> = terminal_ids.iter().copied().collect();

    while !uncovered.is_empty() {
        // Dijkstra from root with cost overrides
        let (distances, predecessors) =
            dijkstra_with_overrides(graph, root_id, weight_attr_ref, &cost_overrides)?;

        // Find cheapest uncovered terminal
        let mut best_terminal: Option<u64> = None;
        let mut best_dist = f64::INFINITY;

        for &t in &uncovered {
            if let Some(&d) = distances.get(&t) {
                if d < best_dist {
                    best_dist = d;
                    best_terminal = Some(t);
                }
            }
        }

        let target = best_terminal.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "Some terminal nodes are not reachable from root",
            )
        })?;

        // Trace path from root to target, collect edges, zero their costs
        let path = reconstruct_path(&predecessors, root_id, target);
        if path.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "No directed path from root to terminal '{}'",
                get_label_or_id(graph, target)
            )));
        }

        for w in path.windows(2) {
            let (u, v) = (w[0], w[1]);
            // The edge the search actually used: cheapest under the current overrides
            let used = cheapest_edge(graph, u, v, |eid| {
                cost_overrides
                    .get(&eid)
                    .copied()
                    .unwrap_or_else(|| get_edge_weight(graph, eid, weight_attr_ref))
            });
            if let Some((eid, _)) = used {
                tree_edges.insert(eid);
                tree_parent.insert(v, u);
                cost_overrides.insert(eid, 0.0);
            }
        }

        // Mark covered terminals (any terminal on this path)
        for &node in &path {
            uncovered.remove(&node);
        }
    }

    // Build result edge list
    let mut result: Vec<(String, String, f64)> = Vec::with_capacity(tree_edges.len());
    for &eid in &tree_edges {
        if let Some(edge) = graph.get_edge(eid) {
            let src_label = get_label_or_id(graph, edge.source);
            let tgt_label = get_label_or_id(graph, edge.target);
            let weight = get_edge_weight(graph, eid, weight_attr_ref);
            result.push((src_label, tgt_label, weight));
        }
    }

    result.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    Ok(result)
}

/// Dijkstra with edge-cost overrides (used for edge reuse in SPH).
fn dijkstra_with_overrides(
    graph: &Graph,
    source: u64,
    weight_attr: Option<&str>,
    cost_overrides: &FxHashMap<u64, f64>,
) -> PyResult<(FxHashMap<u64, f64>, FxHashMap<u64, u64>)> {
    let mut distances: FxHashMap<u64, f64> = FxHashMap::default();
    let mut predecessors: FxHashMap<u64, u64> = FxHashMap::default();
    let mut heap = BinaryHeap::new();

    distances.insert(source, 0.0);
    heap.push(DijkstraEntry {
        distance: 0.0,
        node: source,
    });

    while let Some(DijkstraEntry { distance, node }) = heap.pop() {
        if let Some(&best) = distances.get(&node) {
            if distance > best {
                continue;
            }
        }

        let neighbors = graph.neighbors(node)?;

        for neighbor in neighbors {
            let Some((_, edge_weight)) = cheapest_edge(graph, node, neighbor, |eid| {
                cost_overrides
                    .get(&eid)
                    .copied()
                    .unwrap_or_else(|| get_edge_weight(graph, eid, weight_attr))
            }) else {
                continue;
            };

            let new_dist = distance + edge_weight;
            let current_dist = distances.get(&neighbor).copied().unwrap_or(f64::INFINITY);

            if new_dist < current_dist {
                distances.insert(neighbor, new_dist);
                predecessors.insert(neighbor, node);
                heap.push(DijkstraEntry {
                    distance: new_dist,
                    node: neighbor,
                });
            }
        }
    }

    Ok((distances, predecessors))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new();

        // Initially separate
        assert!(uf.union(1, 2));
        assert!(uf.union(3, 4));

        // Same component
        assert!(!uf.union(1, 2));

        // Connect components
        assert!(uf.union(2, 3));

        // Now all in same component
        assert!(!uf.union(1, 4));
    }
}
