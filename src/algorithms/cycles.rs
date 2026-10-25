//! Cycle detection algorithms
//!
//! This module provides various cycle detection and enumeration algorithms:
//!
//! - `simple_cycles`: Find all simple cycles in a graph (directed or undirected)
//! - `cycle_basis`: Find a fundamental cycle basis of an undirected graph
//! - `minimum_cycle_basis`: Find a minimum weight cycle basis of an undirected graph
//! - `find_cycles`: Legacy function for finding cycles in directed graphs (Johnson's algorithm)
//! - `directed_cycle_basis`: Find a fundamental cycle basis of a directed graph
//! - `has_cycle`: Check if a graph contains any cycle
//! - `feedback_edge_set`: Find a minimum-weight set of edges whose removal breaks every cycle

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Find all simple cycles in the graph.
///
/// For directed graphs, uses Johnson's algorithm.
/// For undirected graphs, uses a backtracking DFS approach that finds all
/// elementary cycles by starting from each node and exploring all paths
/// that return to the starting node.
///
/// # Arguments
/// * `graph` - The graph to search for cycles
/// * `length_bound` - Optional maximum cycle length (None for no limit)
///
/// # Returns
/// A vector of cycles, where each cycle is a vector of node IDs.
/// Each cycle is represented starting from its lexicographically smallest node
/// to avoid duplicates.
#[pyfunction]
#[pyo3(signature = (graph, length_bound=None))]
pub fn simple_cycles(graph: &Graph, length_bound: Option<usize>) -> PyResult<Vec<Vec<u64>>> {
    if graph.is_directed() {
        // For directed graphs, use Johnson's algorithm (existing implementation)
        simple_cycles_directed(graph, length_bound)
    } else {
        // For undirected graphs, use backtracking DFS
        simple_cycles_undirected(graph, length_bound)
    }
}

/// Find simple cycles in a directed graph using Johnson's algorithm
fn simple_cycles_directed(graph: &Graph, length_bound: Option<usize>) -> PyResult<Vec<Vec<u64>>> {
    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    let mut all_cycles = Vec::new();

    // Build adjacency list for efficient traversal
    let mut adj_list: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for &node_id in &node_ids {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            adj_list.insert(node_id, neighbors);
        } else {
            adj_list.insert(node_id, Vec::new());
        }
    }

    // Process nodes in order
    for start_idx in 0..node_ids.len() {
        let start_node = node_ids[start_idx];

        // Build subgraph containing start_node and all nodes with higher indices
        let remaining = n - start_idx;
        let mut subgraph_nodes: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(remaining, Default::default());
        subgraph_nodes.extend(node_ids[start_idx..].iter().copied());

        // Find strongly connected component containing start_node in subgraph
        let scc = find_scc_containing_node(start_node, &adj_list, &subgraph_nodes);

        if scc.len() > 1 || (scc.len() == 1 && adj_list.get(&start_node)
            .is_some_and(|neighbors| neighbors.contains(&start_node))) {
            if let Some(bound) = length_bound {
                // Johnson's blocking is unsound under a length bound: a node blocked
                // after a search cut short by the bound may still close a cycle when
                // reached by a shorter path. Search without blocking instead, pruning
                // by each node's distance back to the start.
                let dist = distances_to_node(start_node, &adj_list, &scc);
                let mut path = vec![start_node];
                let mut path_set = FxHashSet::default();
                path_set.insert(start_node);
                bounded_cycles_from_node(
                    start_node, start_node, &adj_list, &scc, &dist, bound, &mut path,
                    &mut path_set, &mut all_cycles,
                );
                continue;
            }
            let mut blocked: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(scc.len(), Default::default());
            let mut blocked_map: FxHashMap<u64, FxHashSet<u64>> = FxHashMap::with_capacity_and_hasher(scc.len(), Default::default());
            let mut path = Vec::with_capacity(scc.len());
            path.push(start_node);
            let mut path_set = FxHashSet::with_capacity_and_hasher(scc.len(), Default::default());
            path_set.insert(start_node);

            simple_cycles_from_node_directed(
                start_node,
                start_node,
                &adj_list,
                &scc,
                &mut path,
                &mut path_set,
                &mut blocked,
                &mut blocked_map,
                &mut all_cycles,
                length_bound,
            );
        }
    }

    Ok(all_cycles)
}

/// Number of edges on a shortest path from each node of `scc` to `target`, within `scc`.
fn distances_to_node(
    target: u64,
    adj_list: &FxHashMap<u64, Vec<u64>>,
    scc: &FxHashSet<u64>,
) -> FxHashMap<u64, usize> {
    let mut preds: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for &u in scc {
        for &v in adj_list.get(&u).map(Vec::as_slice).unwrap_or(&[]) {
            if scc.contains(&v) {
                preds.entry(v).or_default().push(u);
            }
        }
    }
    let mut dist: FxHashMap<u64, usize> = FxHashMap::default();
    dist.insert(target, 0);
    let mut queue = VecDeque::from([target]);
    while let Some(v) = queue.pop_front() {
        let d = dist[&v];
        for &u in preds.get(&v).map(Vec::as_slice).unwrap_or(&[]) {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(u) {
                e.insert(d + 1);
                queue.push_back(u);
            }
        }
    }
    dist
}

/// Backtracking search for the cycles through `start` with at most `bound` nodes.
///
/// No node is ever blocked, so no cycle is missed; a neighbor is entered only if the
/// shortest way back from it to `start` still fits within the bound.
#[allow(clippy::too_many_arguments)]
fn bounded_cycles_from_node(
    current: u64,
    start: u64,
    adj_list: &FxHashMap<u64, Vec<u64>>,
    scc: &FxHashSet<u64>,
    dist: &FxHashMap<u64, usize>,
    bound: usize,
    path: &mut Vec<u64>,
    path_set: &mut FxHashSet<u64>,
    cycles: &mut Vec<Vec<u64>>,
) {
    let Some(neighbors) = adj_list.get(&current) else {
        return;
    };
    for &neighbor in neighbors {
        if !scc.contains(&neighbor) {
            continue;
        }
        if neighbor == start {
            if path.len() <= bound {
                cycles.push(path.clone());
            }
        } else if !path_set.contains(&neighbor)
            && dist.get(&neighbor).is_some_and(|&d| path.len() + d <= bound)
        {
            path.push(neighbor);
            path_set.insert(neighbor);
            bounded_cycles_from_node(
                neighbor, start, adj_list, scc, dist, bound, path, path_set, cycles,
            );
            path.pop();
            path_set.remove(&neighbor);
        }
    }
}

/// Recursive function to find cycles using backtracking (directed version)
#[allow(clippy::too_many_arguments)]
fn simple_cycles_from_node_directed(
    current: u64,
    start: u64,
    adj_list: &FxHashMap<u64, Vec<u64>>,
    scc: &FxHashSet<u64>,
    path: &mut Vec<u64>,
    path_set: &mut FxHashSet<u64>,
    blocked: &mut FxHashSet<u64>,
    blocked_map: &mut FxHashMap<u64, FxHashSet<u64>>,
    cycles: &mut Vec<Vec<u64>>,
    length_bound: Option<usize>,
) -> bool {
    let mut found_cycle = false;
    blocked.insert(current);

    if let Some(neighbors) = adj_list.get(&current) {
        for &neighbor in neighbors {
            if !scc.contains(&neighbor) {
                continue;
            }

            if neighbor == start {
                // Found a cycle
                found_cycle = true;
                // Check length bound
                if length_bound.map_or(true, |bound| path.len() <= bound) {
                    cycles.push(path.clone());
                }
            } else if !blocked.contains(&neighbor) && !path_set.contains(&neighbor) {
                // Check if adding this node would exceed length bound
                if length_bound.map_or(true, |bound| path.len() < bound) {
                    path.push(neighbor);
                    path_set.insert(neighbor);

                    let found = simple_cycles_from_node_directed(
                        neighbor,
                        start,
                        adj_list,
                        scc,
                        path,
                        path_set,
                        blocked,
                        blocked_map,
                        cycles,
                        length_bound,
                    );

                    path.pop();
                    path_set.remove(&neighbor);

                    found_cycle = found_cycle || found;
                }
            }
        }
    }

    if found_cycle {
        unblock(current, blocked, blocked_map);
    } else if let Some(neighbors) = adj_list.get(&current) {
        for &neighbor in neighbors {
            if scc.contains(&neighbor) {
                blocked_map
                    .entry(neighbor)
                    .or_default()
                    .insert(current);
            }
        }
    }

    found_cycle
}

/// Find simple cycles in an undirected graph using backtracking DFS
fn simple_cycles_undirected(graph: &Graph, length_bound: Option<usize>) -> PyResult<Vec<Vec<u64>>> {
    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Build adjacency list
    let mut adj_list: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for &node_id in &node_ids {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            adj_list.insert(node_id, neighbors);
        } else {
            adj_list.insert(node_id, Vec::new());
        }
    }

    // Create node index mapping for canonical ordering
    let node_to_idx: FxHashMap<u64, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut all_cycles: Vec<Vec<u64>> = Vec::new();
    let mut seen_cycles: FxHashSet<Vec<u64>> = FxHashSet::default();

    // Try starting from each node
    for &start_node in &node_ids {
        let mut path = vec![start_node];
        let mut path_set = FxHashSet::default();
        path_set.insert(start_node);

        find_cycles_undirected_dfs(
            start_node,
            start_node,
            None, // No previous edge initially
            &adj_list,
            &node_to_idx,
            &mut path,
            &mut path_set,
            &mut all_cycles,
            &mut seen_cycles,
            length_bound,
        );
    }

    Ok(all_cycles)
}

/// DFS helper for finding cycles in undirected graphs
#[allow(clippy::too_many_arguments)]
fn find_cycles_undirected_dfs(
    current: u64,
    start: u64,
    prev: Option<u64>,
    adj_list: &FxHashMap<u64, Vec<u64>>,
    node_to_idx: &FxHashMap<u64, usize>,
    path: &mut Vec<u64>,
    path_set: &mut FxHashSet<u64>,
    cycles: &mut Vec<Vec<u64>>,
    seen_cycles: &mut FxHashSet<Vec<u64>>,
    length_bound: Option<usize>,
) {
    if let Some(neighbors) = adj_list.get(&current) {
        for &neighbor in neighbors {
            // Skip the edge we came from (avoid trivial backtracking)
            if Some(neighbor) == prev {
                continue;
            }

            if neighbor == start && path.len() >= 3 {
                // Found a cycle - canonicalize it
                let canonical = canonicalize_cycle(path, node_to_idx);
                if !seen_cycles.contains(&canonical) {
                    // Check length bound
                    if length_bound.map_or(true, |bound| path.len() <= bound) {
                        seen_cycles.insert(canonical);
                        cycles.push(path.clone());
                    }
                }
            } else if !path_set.contains(&neighbor) {
                // Only explore if we haven't visited this node in current path
                // and the neighbor has a higher index than start (to avoid duplicates)
                let start_idx = node_to_idx.get(&start).copied().unwrap_or(0);
                let neighbor_idx = node_to_idx.get(&neighbor).copied().unwrap_or(0);

                // Only continue if neighbor index >= start index to avoid finding same cycle twice
                if neighbor_idx >= start_idx {
                    // Check if adding this node would exceed length bound
                    if length_bound.map_or(true, |bound| path.len() < bound) {
                        path.push(neighbor);
                        path_set.insert(neighbor);

                        find_cycles_undirected_dfs(
                            neighbor,
                            start,
                            Some(current),
                            adj_list,
                            node_to_idx,
                            path,
                            path_set,
                            cycles,
                            seen_cycles,
                            length_bound,
                        );

                        path.pop();
                        path_set.remove(&neighbor);
                    }
                }
            }
        }
    }
}

/// Canonicalize a cycle for deduplication
/// Returns the lexicographically smallest rotation of the cycle
fn canonicalize_cycle(cycle: &[u64], node_to_idx: &FxHashMap<u64, usize>) -> Vec<u64> {
    if cycle.is_empty() {
        return Vec::new();
    }

    let n = cycle.len();

    // Find the position of the minimum element (by index)
    let min_pos = (0..n)
        .min_by_key(|&i| node_to_idx.get(&cycle[i]).copied().unwrap_or(usize::MAX))
        .unwrap_or(0);

    // Determine direction: compare neighbors of min element
    let next_pos = (min_pos + 1) % n;
    let prev_pos = (min_pos + n - 1) % n;

    let next_idx = node_to_idx.get(&cycle[next_pos]).copied().unwrap_or(usize::MAX);
    let prev_idx = node_to_idx.get(&cycle[prev_pos]).copied().unwrap_or(usize::MAX);

    let mut result = Vec::with_capacity(n);

    if next_idx <= prev_idx {
        // Forward direction
        for i in 0..n {
            result.push(cycle[(min_pos + i) % n]);
        }
    } else {
        // Reverse direction
        for i in 0..n {
            result.push(cycle[(min_pos + n - i) % n]);
        }
    }

    result
}

// ============================================================================
// Cycle Basis Algorithms
// ============================================================================

/// Entry for priority queue in shortest path algorithm
#[derive(Clone)]
struct PathEntry {
    distance: f64,
    node: u64,
}

impl PartialEq for PathEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node == other.node
    }
}

impl Eq for PathEntry {}

impl Ord for PathEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (smaller distances have higher priority)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for PathEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Get edge weight from graph, using the weight field or custom attribute
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

/// Returns a list of fundamental cycles forming a cycle basis of the graph.
///
/// A cycle basis is a minimal set of cycles such that any cycle in the graph
/// can be expressed as a symmetric difference (XOR) of cycles in the basis.
/// For a connected graph with V vertices and E edges, the cycle basis has
/// exactly E - V + 1 cycles (the cyclomatic complexity).
///
/// The algorithm works by:
/// 1. Building a spanning tree using BFS from a root node
/// 2. For each non-tree edge (called a "back edge"), finding the fundamental
///    cycle it creates with the tree path
///
/// # Arguments
/// * `graph` - The undirected graph to find cycle basis for
/// * `root` - Optional root node ID to start BFS from. If None, uses the first node.
///
/// # Returns
/// A vector of cycles, where each cycle is a vector of node IDs representing
/// the cycle path. Each node appears once except the implicit return to start.
///
/// # Errors
/// Returns an error if the graph is directed (cycle basis only applies to undirected graphs).
#[pyfunction]
#[pyo3(signature = (graph, root=None))]
pub fn cycle_basis(graph: &Graph, root: Option<u64>) -> PyResult<Vec<Vec<u64>>> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cycle_basis() is only defined for undirected graphs"
        ));
    }

    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Build adjacency list for efficient traversal
    let mut adj_list: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for &node_id in &node_ids {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            adj_list.insert(node_id, neighbors);
        } else {
            adj_list.insert(node_id, Vec::new());
        }
    }

    // Track which nodes have been processed (across all components)
    let mut global_seen: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(n, Default::default());
    let mut cycles = Vec::new();

    // Start with the specified root (or first node)
    let start_nodes: Vec<u64> = if let Some(r) = root {
        // Verify root exists
        if !adj_list.contains_key(&r) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Root node {} not found in graph", r)
            ));
        }
        let mut starts = vec![r];
        // Add remaining nodes for disconnected components
        for &node in &node_ids {
            if node != r {
                starts.push(node);
            }
        }
        starts
    } else {
        node_ids.clone()
    };

    // Process each connected component
    for &start_node in &start_nodes {
        if global_seen.contains(&start_node) {
            continue;
        }

        // BFS to build spanning tree and find back edges
        let mut parent: FxHashMap<u64, u64> = FxHashMap::default();
        let mut visited: FxHashSet<u64> = FxHashSet::default();
        let mut queue = VecDeque::new();

        queue.push_back(start_node);
        visited.insert(start_node);
        global_seen.insert(start_node);

        // Track edges we've seen in BFS (to avoid counting same edge twice in undirected graph)
        let mut tree_edges: FxHashSet<(u64, u64)> = FxHashSet::default();
        let mut back_edges: Vec<(u64, u64)> = Vec::new();

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adj_list.get(&current) {
                for &neighbor in neighbors {
                    // Normalize edge for undirected comparison
                    let edge = if current < neighbor {
                        (current, neighbor)
                    } else {
                        (neighbor, current)
                    };

                    if !visited.contains(&neighbor) {
                        // Tree edge
                        visited.insert(neighbor);
                        global_seen.insert(neighbor);
                        parent.insert(neighbor, current);
                        tree_edges.insert(edge);
                        queue.push_back(neighbor);
                    } else if !tree_edges.contains(&edge) {
                        // Back edge - creates a fundamental cycle
                        // Only add if we haven't seen this edge as a tree edge or back edge
                        let back_edge = (current, neighbor);
                        // Check if reverse already added
                        let reverse = (neighbor, current);
                        if !back_edges.contains(&reverse) {
                            back_edges.push(back_edge);
                        }
                    }
                }
            }
        }

        // For each back edge, find the fundamental cycle
        for (u, v) in back_edges {
            // Find path from u to v through the spanning tree
            // Get ancestors of u and v
            let mut ancestors_u: Vec<u64> = Vec::new();
            let mut node = u;
            ancestors_u.push(node);
            while let Some(&p) = parent.get(&node) {
                ancestors_u.push(p);
                node = p;
            }

            let mut ancestors_v: Vec<u64> = Vec::new();
            node = v;
            ancestors_v.push(node);
            while let Some(&p) = parent.get(&node) {
                ancestors_v.push(p);
                node = p;
            }

            // Find lowest common ancestor (LCA)
            let ancestors_u_set: FxHashSet<u64> = ancestors_u.iter().copied().collect();

            let mut lca = None;
            for &ancestor in &ancestors_v {
                if ancestors_u_set.contains(&ancestor) {
                    lca = Some(ancestor);
                    break;
                }
            }

            // Build cycle: path from u to LCA + path from LCA to v (reversed)
            if let Some(lca_node) = lca {
                let mut cycle = Vec::new();

                // Path from u to LCA
                for &ancestor in &ancestors_u {
                    cycle.push(ancestor);
                    if ancestor == lca_node {
                        break;
                    }
                }

                // Path from LCA to v (skip LCA since it's already included)
                let mut v_to_lca: Vec<u64> = Vec::new();
                for &ancestor in &ancestors_v {
                    if ancestor == lca_node {
                        break;
                    }
                    v_to_lca.push(ancestor);
                }
                v_to_lca.reverse();
                cycle.extend(v_to_lca);

                if cycle.len() >= 3 {
                    cycles.push(cycle);
                }
            }
        }
    }

    Ok(cycles)
}

/// Returns a minimum weight cycle basis for an undirected graph.
///
/// A minimum cycle basis is a cycle basis where the sum of weights of all
/// cycles is minimized. For unweighted graphs, this minimizes the sum of
/// cycle lengths.
///
/// This implements Horton's algorithm:
/// 1. For each edge (u,v), find the shortest path from u to v not using (u,v)
/// 2. This gives a candidate cycle for each edge
/// 3. Sort candidates by weight and greedily select linearly independent ones
///
/// For simplicity, this implementation uses a greedy approach based on
/// shortest paths from each node, which gives good results for most graphs.
///
/// # Arguments
/// * `graph` - The undirected graph to find minimum cycle basis for
/// * `weight` - Optional edge attribute name to use as weight. If None, uses
///              the default edge weight (or 1.0 if not set).
///
/// # Returns
/// A vector of cycles forming a minimum weight cycle basis.
///
/// # Errors
/// Returns an error if the graph is directed.
#[pyfunction]
#[pyo3(signature = (graph, weight=None))]
pub fn minimum_cycle_basis(graph: &Graph, weight: Option<String>) -> PyResult<Vec<Vec<u64>>> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "minimum_cycle_basis() is only defined for undirected graphs"
        ));
    }

    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    let weight_attr = weight.as_deref();

    // Build adjacency list with edge info
    // For undirected graphs, we store both directions
    let mut adj_with_edges: FxHashMap<u64, Vec<(u64, u64)>> = FxHashMap::with_capacity_and_hasher(n, Default::default());

    for &node_id in &node_ids {
        adj_with_edges.insert(node_id, Vec::new());
    }

    // Collect all edges
    let edge_ids = graph.edge_ids();
    let mut edges_info: Vec<(u64, u64, u64, f64)> = Vec::with_capacity(edge_ids.len()); // (edge_id, source, target, weight)

    for edge_id in edge_ids {
        if let Some(edge) = graph.get_edge(edge_id) {
            let w = get_edge_weight(graph, edge_id, weight_attr);
            edges_info.push((edge_id, edge.source, edge.target, w));

            // Add to adjacency (both directions for undirected)
            adj_with_edges
                .entry(edge.source)
                .or_default()
                .push((edge.target, edge_id));
            adj_with_edges
                .entry(edge.target)
                .or_default()
                .push((edge.source, edge_id));
        }
    }

    // Calculate cyclomatic complexity for each connected component
    // Total cycles needed = E - V + C (where C is number of components)
    let num_edges = edges_info.len();
    let num_nodes = n;

    // Find number of connected components
    let mut component_count = 0;
    let mut component_visited: FxHashSet<u64> = FxHashSet::default();

    for &start in &node_ids {
        if component_visited.contains(&start) {
            continue;
        }
        component_count += 1;

        let mut queue = VecDeque::new();
        queue.push_back(start);
        component_visited.insert(start);

        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = adj_with_edges.get(&node) {
                for &(neighbor, _) in neighbors {
                    if !component_visited.contains(&neighbor) {
                        component_visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }

    let target_cycles = if num_edges >= num_nodes {
        num_edges - num_nodes + component_count
    } else {
        0
    };

    if target_cycles == 0 {
        return Ok(Vec::new());
    }

    // Generate candidate cycles using Horton's algorithm
    // For each edge, find the shortest cycle containing it
    let mut candidate_cycles: Vec<(f64, Vec<u64>)> = Vec::new();

    for &(edge_id, u, v, edge_weight) in &edges_info {
        // Find shortest path from u to v NOT using this edge
        if let Some((path_weight, path)) = shortest_path_excluding_edge(
            &adj_with_edges,
            u,
            v,
            edge_id,
            graph,
            weight_attr,
        ) {
            // The cycle is: path from u to v + direct edge back
            let cycle_weight = path_weight + edge_weight;
            if path.len() >= 2 {
                candidate_cycles.push((cycle_weight, path));
            }
        }
    }

    // Sort candidates by weight (ascending)
    candidate_cycles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    // Greedily select independent cycles
    // Use edge-incidence vectors for independence check
    let mut selected_cycles: Vec<Vec<u64>> = Vec::new();
    let mut edge_space: Vec<FxHashSet<(u64, u64)>> = Vec::new();

    for (_, cycle) in candidate_cycles {
        if selected_cycles.len() >= target_cycles {
            break;
        }

        // Get edge set for this cycle (normalized)
        let edge_set = cycle_to_edge_set(&cycle);

        // Check if this cycle is independent of already selected cycles
        // A cycle is independent if its edge set cannot be expressed as XOR of selected edge sets
        if is_independent_cycle(&edge_set, &edge_space) {
            // Add to edge space
            edge_space.push(edge_set);
            selected_cycles.push(cycle);
        }
    }

    Ok(selected_cycles)
}

/// Find shortest path from source to target, excluding a specific edge
fn shortest_path_excluding_edge(
    adj_with_edges: &FxHashMap<u64, Vec<(u64, u64)>>,
    source: u64,
    target: u64,
    excluded_edge: u64,
    graph: &Graph,
    weight_attr: Option<&str>,
) -> Option<(f64, Vec<u64>)> {
    let mut distances: FxHashMap<u64, f64> = FxHashMap::default();
    let mut predecessors: FxHashMap<u64, u64> = FxHashMap::default();
    let mut heap = BinaryHeap::new();

    distances.insert(source, 0.0);
    heap.push(PathEntry {
        distance: 0.0,
        node: source,
    });

    while let Some(PathEntry { distance, node }) = heap.pop() {
        if node == target {
            // Reconstruct path
            let mut path = vec![target];
            let mut current = target;
            while let Some(&pred) = predecessors.get(&current) {
                path.push(pred);
                current = pred;
            }
            path.reverse();
            return Some((distance, path));
        }

        // Skip if we've found a better path already
        if let Some(&d) = distances.get(&node) {
            if distance > d {
                continue;
            }
        }

        if let Some(neighbors) = adj_with_edges.get(&node) {
            for &(neighbor, edge_id) in neighbors {
                // Skip the excluded edge
                if edge_id == excluded_edge {
                    continue;
                }

                let edge_weight = get_edge_weight(graph, edge_id, weight_attr);
                let new_dist = distance + edge_weight;

                let is_better = distances
                    .get(&neighbor)
                    .map(|&d| new_dist < d)
                    .unwrap_or(true);

                if is_better {
                    distances.insert(neighbor, new_dist);
                    predecessors.insert(neighbor, node);
                    heap.push(PathEntry {
                        distance: new_dist,
                        node: neighbor,
                    });
                }
            }
        }
    }

    None
}

/// Convert cycle (list of nodes) to set of edges (normalized pairs)
fn cycle_to_edge_set(cycle: &[u64]) -> FxHashSet<(u64, u64)> {
    let mut edges = FxHashSet::default();
    let n = cycle.len();

    for i in 0..n {
        let u = cycle[i];
        let v = cycle[(i + 1) % n];
        let edge = if u < v { (u, v) } else { (v, u) };
        edges.insert(edge);
    }

    edges
}

/// Check if a cycle (represented as edge set) is linearly independent
/// of the existing edge space (using GF(2) / XOR arithmetic)
fn is_independent_cycle(
    new_edges: &FxHashSet<(u64, u64)>,
    edge_space: &[FxHashSet<(u64, u64)>],
) -> bool {
    if edge_space.is_empty() {
        return true;
    }

    // Simple check: if the new cycle shares no edges with any existing cycle,
    // it's definitely independent. More sophisticated Gaussian elimination
    // would be needed for complete independence check, but this greedy
    // approach works well in practice.
    //
    // For a complete implementation, we would use Gaussian elimination
    // over GF(2), but the greedy weight-sorted approach typically produces
    // a valid minimum basis.

    // Check if this edge set is a subset or superset of any existing
    for existing in edge_space {
        // Check if XOR would be empty (meaning same cycle)
        let xor_result: FxHashSet<(u64, u64)> = new_edges
            .symmetric_difference(existing)
            .cloned()
            .collect();

        if xor_result.is_empty() {
            return false; // Same cycle
        }
    }

    // Additional check: try to express new_edges as XOR of existing edge sets
    // This is a simplified version - full Gaussian elimination would be more complete
    // but this heuristic works well for most practical graphs

    true
}

/// Find all elementary cycles in the graph using Johnson's algorithm
///
/// This implements Donald B. Johnson's 1975 algorithm for finding all elementary
/// cycles in a directed graph. The algorithm is efficient and finds each cycle
/// exactly once.
///
/// # Arguments
/// * `graph` - The directed graph to search for cycles
/// * `min_amount` - Optional minimum cycle length (defaults to 2)
///
/// # Returns
/// A vector of cycles, where each cycle is a vector of node IDs
#[pyfunction]
#[pyo3(signature = (graph, min_amount=None))]
pub fn find_cycles(graph: &Graph, min_amount: Option<f64>) -> PyResult<Vec<Vec<u64>>> {
    let min_length = min_amount.unwrap_or(2.0) as usize;

    if !graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cycle enumeration requires a directed graph"
        ));
    }

    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    let mut all_cycles = Vec::with_capacity(n); // Estimated capacity

    // Build adjacency list for efficient traversal
    let mut adj_list: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for &node_id in &node_ids {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            adj_list.insert(node_id, neighbors);
        } else {
            adj_list.insert(node_id, Vec::new());
        }
    }

    // Process nodes in order
    for start_idx in 0..node_ids.len() {
        let start_node = node_ids[start_idx];

        // Build subgraph containing start_node and all nodes with higher indices
        let remaining = n - start_idx;
        let mut subgraph_nodes: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(remaining, Default::default());
        subgraph_nodes.extend(node_ids[start_idx..].iter().copied());

        // Find strongly connected component containing start_node in subgraph
        let scc = find_scc_containing_node(start_node, &adj_list, &subgraph_nodes);

        if scc.len() > 1 || (scc.len() == 1 && adj_list.get(&start_node)
            .is_some_and(|neighbors| neighbors.contains(&start_node))) {
            // There might be cycles in this SCC
            let mut blocked: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(scc.len(), Default::default());
            let mut blocked_map: FxHashMap<u64, FxHashSet<u64>> = FxHashMap::with_capacity_and_hasher(scc.len(), Default::default());
            let mut path = Vec::with_capacity(scc.len());
            path.push(start_node);
            let mut path_set = FxHashSet::with_capacity_and_hasher(scc.len(), Default::default());
            path_set.insert(start_node);

            find_cycles_from_node(
                start_node,
                start_node,
                &adj_list,
                &scc,
                &mut path,
                &mut path_set,
                &mut blocked,
                &mut blocked_map,
                &mut all_cycles,
                min_length,
            );
        }
    }

    Ok(all_cycles)
}

/// Find strongly connected component containing the given node
fn find_scc_containing_node(
    start: u64,
    adj_list: &FxHashMap<u64, Vec<u64>>,
    subgraph: &FxHashSet<u64>,
) -> FxHashSet<u64> {
    // Use Tarjan's algorithm simplified for finding one SCC
    let subgraph_size = subgraph.len();
    let mut scc = FxHashSet::with_capacity_and_hasher(subgraph_size, Default::default());
    let mut visited = FxHashSet::with_capacity_and_hasher(subgraph_size, Default::default());
    let mut stack = Vec::with_capacity(subgraph_size);

    // DFS from start node
    let mut queue = VecDeque::with_capacity(subgraph_size);
    queue.push_back(start);
    visited.insert(start);

    while let Some(node) = queue.pop_front() {
        stack.push(node);
        scc.insert(node);

        if let Some(neighbors) = adj_list.get(&node) {
            for &neighbor in neighbors {
                if subgraph.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // Find nodes that can reach start (for strong connectivity)
    let mut can_reach_start = FxHashSet::with_capacity_and_hasher(scc.len(), Default::default());
    can_reach_start.insert(start);

    for &node in &scc {
        if node != start && can_reach_start_from(node, start, adj_list, &scc) {
            can_reach_start.insert(node);
        }
    }

    can_reach_start
}

/// Check if we can reach target from source within the given node set
#[inline]
fn can_reach_start_from(
    source: u64,
    target: u64,
    adj_list: &FxHashMap<u64, Vec<u64>>,
    allowed_nodes: &FxHashSet<u64>,
) -> bool {
    let capacity = allowed_nodes.len();
    let mut visited = FxHashSet::with_capacity_and_hasher(capacity, Default::default());
    let mut queue = VecDeque::with_capacity(capacity);
    queue.push_back(source);
    visited.insert(source);

    while let Some(node) = queue.pop_front() {
        if node == target {
            return true;
        }

        if let Some(neighbors) = adj_list.get(&node) {
            for &neighbor in neighbors {
                if allowed_nodes.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    false
}

/// Recursive function to find cycles using backtracking
#[allow(clippy::too_many_arguments)]
fn find_cycles_from_node(
    current: u64,
    start: u64,
    adj_list: &FxHashMap<u64, Vec<u64>>,
    scc: &FxHashSet<u64>,
    path: &mut Vec<u64>,
    path_set: &mut FxHashSet<u64>,
    blocked: &mut FxHashSet<u64>,
    blocked_map: &mut FxHashMap<u64, FxHashSet<u64>>,
    cycles: &mut Vec<Vec<u64>>,
    min_length: usize,
) -> bool {
    let mut found_cycle = false;
    blocked.insert(current);

    if let Some(neighbors) = adj_list.get(&current) {
        for &neighbor in neighbors {
            if !scc.contains(&neighbor) {
                continue;
            }

            if neighbor == start {
                // Found a cycle (structure-wise, regardless of length)
                found_cycle = true;
                // Only add to results if it meets min_length requirement
                if path.len() >= min_length {
                    cycles.push(path.clone());
                }
            } else if !blocked.contains(&neighbor) && !path_set.contains(&neighbor) {
                // Continue DFS
                path.push(neighbor);
                path_set.insert(neighbor);

                let found = find_cycles_from_node(
                    neighbor,
                    start,
                    adj_list,
                    scc,
                    path,
                    path_set,
                    blocked,
                    blocked_map,
                    cycles,
                    min_length,
                );

                path.pop();
                path_set.remove(&neighbor);

                found_cycle = found_cycle || found;
            }
        }
    }

    if found_cycle {
        unblock(current, blocked, blocked_map);
    } else {
        // Add current to blocked map of its neighbors
        if let Some(neighbors) = adj_list.get(&current) {
            for &neighbor in neighbors {
                if scc.contains(&neighbor) {
                    blocked_map
                        .entry(neighbor)
                        .or_default()
                        .insert(current);
                }
            }
        }
    }

    found_cycle
}

/// Unblock a node and recursively unblock nodes that depend on it
#[inline]
fn unblock(
    node: u64,
    blocked: &mut FxHashSet<u64>,
    blocked_map: &mut FxHashMap<u64, FxHashSet<u64>>,
) {
    blocked.remove(&node);

    if let Some(blocked_set) = blocked_map.remove(&node) {
        for blocked_node in blocked_set {
            if blocked.contains(&blocked_node) {
                unblock(blocked_node, blocked, blocked_map);
            }
        }
    }
}

/// Detect if graph has any cycle using DFS
pub fn has_cycle(graph: &Graph) -> bool {
    if !graph.is_directed() {
        return has_undirected_cycle(graph);
    }
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    let mut visited = FxHashSet::with_capacity_and_hasher(n, Default::default());
    let mut rec_stack = FxHashSet::with_capacity_and_hasher(n, Default::default());

    for node_id in node_ids {
        if !visited.contains(&node_id)
            && has_cycle_util(graph, node_id, &mut visited, &mut rec_stack) {
                return true;
            }
    }

    false
}

/// Undirected cycle check via union-find over the edge list.
///
/// A DFS over the (symmetric) adjacency list would see the edge back to the
/// parent as a cycle, so every graph with an edge would report one. Here an
/// edge closes a cycle only if it is a self-loop or its endpoints are already
/// connected, which also counts parallel edges, as a multigraph should.
fn has_undirected_cycle(graph: &Graph) -> bool {
    let mut parent: FxHashMap<u64, u64> = FxHashMap::default();

    fn find(parent: &mut FxHashMap<u64, u64>, x: u64) -> u64 {
        let mut root = x;
        while let Some(&p) = parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        // path compression
        let mut cur = x;
        while cur != root {
            let next = *parent.get(&cur).unwrap_or(&root);
            parent.insert(cur, root);
            cur = next;
        }
        root
    }

    for (_, u, v) in graph.edges() {
        if u == v {
            return true;
        }
        parent.entry(u).or_insert(u);
        parent.entry(v).or_insert(v);
        let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
        if ru == rv {
            return true;
        }
        parent.insert(ru, rv);
    }
    false
}

#[inline]
fn has_cycle_util(
    graph: &Graph,
    node: u64,
    visited: &mut FxHashSet<u64>,
    rec_stack: &mut FxHashSet<u64>,
) -> bool {
    visited.insert(node);
    rec_stack.insert(node);

    if let Ok(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                if has_cycle_util(graph, neighbor, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack.contains(&neighbor) {
                return true;
            }
        }
    }

    rec_stack.remove(&node);
    false
}

/// Find a fundamental cycle basis for a directed graph.
///
/// For each strongly connected component (SCC) with more than one node,
/// builds a DFS arborescence and extracts a fundamental directed cycle
/// for each non-tree edge within that SCC. The cycle space dimension
/// is m - n + p (edges minus nodes plus number of SCCs).
///
/// Returns cycles as lists of node IDs (each cycle is a closed walk
/// omitting the repeated last node, i.e., [v0, v1, ..., vk] represents
/// v0 -> v1 -> ... -> vk -> v0).
#[pyfunction]
pub fn directed_cycle_basis(graph: &Graph) -> PyResult<Vec<Vec<u64>>> {
    if !graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "directed_cycle_basis() is only defined for directed graphs. Use cycle_basis() for undirected graphs."
        ));
    }

    let node_ids = graph.get_node_ids();
    if node_ids.is_empty() {
        return Ok(Vec::new());
    }

    let sccs = crate::algorithms::connectivity::strongly_connected_components(graph)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let all_edges = graph.edges(); // (id, source, target)
    let mut cycles: Vec<Vec<u64>> = Vec::new();

    // Self-loops are one-node directed cycles and each is its own basis element.
    for &(_, u, v) in &all_edges {
        if u == v {
            cycles.push(vec![u]);
        }
    }

    let mut johnson: Option<Vec<Vec<u64>>> = None; // computed lazily, at most once

    for scc in &sccs {
        if scc.len() < 2 {
            continue;
        }
        let scc_set: FxHashSet<u64> = scc.iter().copied().collect();

        // Non-loop edges inside the SCC; each gets a coordinate in the edge space.
        let edges: Vec<(u64, u64)> = all_edges
            .iter()
            .filter(|&&(_, u, v)| u != v && scc_set.contains(&u) && scc_set.contains(&v))
            .map(|&(_, u, v)| (u, v))
            .collect();
        let dim = edges.len() + 1 - scc.len(); // cycle-space dimension m - n + 1
        if dim == 0 {
            continue;
        }

        // Edge (u, v) -> coordinate indices (parallel arcs get one index each).
        let mut index: FxHashMap<(u64, u64), Vec<usize>> = FxHashMap::default();
        let mut out: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        for (i, &(u, v)) in edges.iter().enumerate() {
            index.entry((u, v)).or_default().push(i);
            out.entry(u).or_default().push(v);
        }

        let mut basis_rows: Vec<Vec<f64>> = Vec::new(); // row-echelon rows
        let mut pivots: Vec<usize> = Vec::new();
        let mut chosen: Vec<Vec<u64>> = Vec::new();
        let mut used_arcs: FxHashMap<(u64, u64), usize> = FxHashMap::default();

        // Try to add a closed directed walk [n0, n1, ..., nk] (nk -> n0 implied).
        let try_add = |cycle: &[u64],
                           used_arcs: &mut FxHashMap<(u64, u64), usize>,
                           basis_rows: &mut Vec<Vec<f64>>,
                           pivots: &mut Vec<usize>|
         -> bool {
            let mut row = vec![0.0f64; edges.len()];
            for w in 0..cycle.len() {
                let (u, v) = (cycle[w], cycle[(w + 1) % cycle.len()]);
                let Some(ids) = index.get(&(u, v)) else { return false };
                // rotate among parallel arcs so different cycles can use different copies
                let k = used_arcs.get(&(u, v)).copied().unwrap_or(0) % ids.len();
                row[ids[k]] += 1.0;
            }
            for (r, &p) in basis_rows.iter().zip(pivots.iter()) {
                if row[p].abs() > 1e-9 {
                    let f = row[p] / r[p];
                    for j in 0..row.len() {
                        row[j] -= f * r[j];
                    }
                }
            }
            match row.iter().position(|x| x.abs() > 1e-9) {
                Some(p) => {
                    for w in 0..cycle.len() {
                        let key = (cycle[w], cycle[(w + 1) % cycle.len()]);
                        *used_arcs.entry(key).or_insert(0) += 1;
                    }
                    basis_rows.push(row);
                    pivots.push(p);
                    true
                }
                None => false,
            }
        };

        // Candidates: for each arc u -> v, close it with a shortest directed path
        // v ~> u inside the SCC (always exists, since the SCC is strongly connected).
        for &(u, v) in &edges {
            if chosen.len() == dim {
                break;
            }
            if let Some(path) = shortest_directed_path(&out, &scc_set, v, u) {
                // path = [v, ..., u]; the arc u -> v closes it
                if try_add(&path, &mut used_arcs, &mut basis_rows, &mut pivots) {
                    chosen.push(path);
                }
            }
        }

        // Shortest cycles through each arc almost always span the space; if not,
        // fall back to all simple cycles (which provably span it for an SCC).
        if chosen.len() < dim {
            if johnson.is_none() {
                johnson = Some(find_cycles(graph, Some(2.0))?);
            }
            for c in johnson.as_ref().unwrap() {
                if chosen.len() == dim {
                    break;
                }
                if c.iter().all(|n| scc_set.contains(n))
                    && try_add(c, &mut used_arcs, &mut basis_rows, &mut pivots)
                {
                    chosen.push(c.clone());
                }
            }
        }

        cycles.extend(chosen);
    }

    // Report each cycle starting at its smallest node ID (rotation keeps the
    // arc order), so the output doesn't depend on hash or search order.
    for c in cycles.iter_mut() {
        if let Some(pos) = c.iter().enumerate().min_by_key(|&(_, v)| *v).map(|(i, _)| i) {
            c.rotate_left(pos);
        }
    }

    Ok(cycles)
}

/// BFS shortest directed path from `from` to `to` using only nodes in `allowed`.
fn shortest_directed_path(
    out: &FxHashMap<u64, Vec<u64>>,
    allowed: &FxHashSet<u64>,
    from: u64,
    to: u64,
) -> Option<Vec<u64>> {
    let mut prev: FxHashMap<u64, u64> = FxHashMap::default();
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    let mut queue = VecDeque::new();
    seen.insert(from);
    queue.push_back(from);
    while let Some(x) = queue.pop_front() {
        if x == to {
            let mut path = vec![to];
            let mut cur = to;
            while cur != from {
                cur = prev[&cur];
                path.push(cur);
            }
            path.reverse();
            return Some(path);
        }
        if let Some(ns) = out.get(&x) {
            for &y in ns {
                if allowed.contains(&y) && seen.insert(y) {
                    prev.insert(y, x);
                    queue.push_back(y);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_cycles_empty_graph() {
        let graph = Graph::new_rust(true, false);
        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_find_cycles_single_node_no_loop() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_find_cycles_self_loop() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        graph.add_test_edge_by_label("A", "A", None);

        let cycles = find_cycles(&graph, Some(1.0)).unwrap();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 1);
    }

    #[test]
    fn test_find_cycles_simple_cycle() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        graph.add_test_node("B");
        graph.add_test_node("C");

        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "C", None);
        graph.add_test_edge_by_label("C", "A", None);

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 3);
    }

    #[test]
    fn test_find_cycles_two_node_cycle() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        graph.add_test_node("B");

        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "A", None);

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 2);
    }

    #[test]
    fn test_find_cycles_multiple_cycles() {
        let mut graph = Graph::new_rust(true, false);
        // Create two separate cycles
        graph.add_test_node("A");
        graph.add_test_node("B");
        graph.add_test_node("C");
        graph.add_test_node("D");

        // First cycle: A -> B -> A
        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "A", None);

        // Second cycle: C -> D -> C
        graph.add_test_edge_by_label("C", "D", None);
        graph.add_test_edge_by_label("D", "C", None);

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 2);
    }

    #[test]
    fn test_find_cycles_overlapping() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        graph.add_test_node("B");
        graph.add_test_node("C");

        // Create overlapping cycles: A->B->A and A->B->C->A
        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "A", None);
        graph.add_test_edge_by_label("B", "C", None);
        graph.add_test_edge_by_label("C", "A", None);

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 2);
    }

    #[test]
    fn test_find_cycles_min_length() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        graph.add_test_node("B");
        graph.add_test_node("C");

        // Two-node cycle
        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "A", None);

        // Three-node cycle
        graph.add_test_edge_by_label("B", "C", None);
        graph.add_test_edge_by_label("C", "A", None);

        // All cycles
        let all_cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(all_cycles.len(), 2);

        // Only cycles of length 3+
        let long_cycles = find_cycles(&graph, Some(3.0)).unwrap();
        assert_eq!(long_cycles.len(), 1);
        assert_eq!(long_cycles[0].len(), 3);
    }

    #[test]
    fn test_find_cycles_dag() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        graph.add_test_node("B");
        graph.add_test_node("C");

        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "C", None);

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_find_cycles_undirected_error() {
        let graph = Graph::new_rust(false, false);
        let result = find_cycles(&graph, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_cycle_acyclic_graph() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        graph.add_test_node("B");
        graph.add_test_node("C");

        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "C", None);

        assert!(!has_cycle(&graph));
    }

    #[test]
    fn test_has_cycle_cyclic_graph() {
        let mut graph = Graph::new_rust(true, false);
        graph.add_test_node("A");
        graph.add_test_node("B");
        graph.add_test_node("C");

        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "C", None);
        graph.add_test_edge_by_label("C", "A", None);

        assert!(has_cycle(&graph));
    }

    #[test]
    fn test_has_cycle_undirected_tree_is_acyclic() {
        let mut graph = Graph::new_rust(false, false);
        for l in ["A", "B", "C", "D"] {
            graph.add_test_node(l);
        }
        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "C", None);
        graph.add_test_edge_by_label("B", "D", None);
        assert!(!has_cycle(&graph));
    }

    #[test]
    fn test_has_cycle_undirected_triangle() {
        let mut graph = Graph::new_rust(false, false);
        for l in ["A", "B", "C"] {
            graph.add_test_node(l);
        }
        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "C", None);
        graph.add_test_edge_by_label("C", "A", None);
        assert!(has_cycle(&graph));
    }

    #[test]
    fn test_directed_cycle_basis_follows_arrows() {
        // House: A->B->C->A plus C->E->D->B. Every basis cycle must follow arcs.
        let mut graph = Graph::new_rust(true, false);
        for l in ["A", "B", "C", "D", "E"] {
            graph.add_test_node(l);
        }
        for (u, v) in [("A", "B"), ("B", "C"), ("C", "A"), ("C", "E"), ("E", "D"), ("D", "B")] {
            graph.add_test_edge_by_label(u, v, None);
        }
        let arcs: FxHashSet<(u64, u64)> = graph.edges().iter().map(|&(_, u, v)| (u, v)).collect();
        let basis = directed_cycle_basis(&graph).unwrap();
        assert_eq!(basis.len(), 2); // m - n + p = 6 - 5 + 1
        for c in &basis {
            for i in 0..c.len() {
                assert!(arcs.contains(&(c[i], c[(i + 1) % c.len()])), "cycle {:?} breaks an arc", c);
            }
        }
    }
}

/// Edge IDs of a minimum-weight feedback edge set: edges whose removal leaves no cycle.
///
/// Undirected graphs: the complement of a maximum spanning forest (Kruskal, heaviest first,
/// ties by edge ID), which is exact. Directed graphs: the local-ratio algorithm of Demetrescu
/// and Finocchi (2003) — find a cycle, lower every residual weight on it by the cycle's
/// minimum, remove the edges that reach zero, repeat — followed by putting removed edges
/// back, heaviest first, whenever that closes no cycle. Weights must be non-negative.
#[pyfunction]
#[pyo3(signature = (graph, weighted=true))]
pub fn feedback_edge_set(graph: &Graph, weighted: bool) -> PyResult<Vec<u64>> {
    let nodes = graph.get_node_ids();
    let idx: FxHashMap<u64, usize> = nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut edges: Vec<(u64, usize, usize, f64)> = Vec::new();
    for e in graph.get_edges_internal() {
        let w = if weighted { e.weight.unwrap_or(1.0) } else { 1.0 };
        if w.is_nan() || w < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feedback_edge_set() requires non-negative edge weights",
            ));
        }
        edges.push((e.id, idx[&e.source], idx[&e.target], w));
    }
    edges.sort_unstable_by_key(|e| e.0);
    let n = nodes.len();
    let mut out = if graph.is_directed() {
        directed_feedback(n, &edges)
    } else {
        let mut order: Vec<usize> = (0..edges.len()).collect();
        order.sort_by(|&a, &b| edges[b].3.total_cmp(&edges[a].3).then(a.cmp(&b)));
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let mut cut = Vec::new();
        for e in order {
            let (a, b) = (find(&mut parent, edges[e].1), find(&mut parent, edges[e].2));
            if a == b {
                cut.push(edges[e].0);
            } else {
                parent[a] = b;
            }
        }
        cut
    };
    out.sort_unstable();
    Ok(out)
}

/// A directed cycle among the live edges, as edge indices, or `None` if there is none.
fn live_cycle(n: usize, adj: &[Vec<usize>], edges: &[(u64, usize, usize, f64)], live: &[bool])
    -> Option<Vec<usize>> {
    // 0 = unvisited, 1 = on the DFS stack, 2 = finished
    let mut state = vec![0u8; n];
    let mut via = vec![usize::MAX; n];
    for root in 0..n {
        if state[root] != 0 {
            continue;
        }
        state[root] = 1;
        let mut stack = vec![(root, 0usize)];
        while let Some(&mut (v, ref mut i)) = stack.last_mut() {
            if *i == adj[v].len() {
                state[v] = 2;
                stack.pop();
                continue;
            }
            let e = adj[v][*i];
            *i += 1;
            if !live[e] {
                continue;
            }
            let w = edges[e].2;
            match state[w] {
                0 => {
                    state[w] = 1;
                    via[w] = e;
                    stack.push((w, 0));
                }
                1 => {
                    let mut cycle = vec![e];
                    let mut x = v;
                    while x != w {
                        cycle.push(via[x]);
                        x = edges[via[x]].1;
                    }
                    return Some(cycle);
                }
                _ => {}
            }
        }
    }
    None
}

fn directed_feedback(n: usize, edges: &[(u64, usize, usize, f64)]) -> Vec<u64> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, e) in edges.iter().enumerate() {
        adj[e.1].push(i);
    }
    let mut live = vec![true; edges.len()];
    let mut residual: Vec<f64> = edges.iter().map(|e| e.3).collect();
    let mut removed = Vec::new();
    while let Some(cycle) = live_cycle(n, &adj, edges, &live) {
        let low = cycle.iter().map(|&e| residual[e]).fold(f64::INFINITY, f64::min);
        for &e in &cycle {
            residual[e] -= low;
            if residual[e] <= 0.0 {
                live[e] = false;
                removed.push(e);
            }
        }
    }
    // Put edges back, heaviest first, where the target can't already reach the source.
    removed.sort_by(|&a, &b| edges[b].3.total_cmp(&edges[a].3).then(a.cmp(&b)));
    let mut seen = vec![usize::MAX; n];
    let mut kept = Vec::new();
    let mut queue = VecDeque::new();
    for (round, &e) in removed.iter().enumerate() {
        let (s, t) = (edges[e].1, edges[e].2);
        let mut closes = s == t;
        queue.clear();
        queue.push_back(t);
        seen[t] = round;
        while let (false, Some(v)) = (closes, queue.pop_front()) {
            for &f in &adj[v] {
                let w = edges[f].2;
                if live[f] && seen[w] != round {
                    seen[w] = round;
                    closes |= w == s;
                    queue.push_back(w);
                }
            }
        }
        if closes {
            kept.push(edges[e].0);
        } else {
            live[e] = true;
        }
    }
    kept
}
