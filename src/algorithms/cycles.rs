//! Cycle detection algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

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
            .map_or(false, |neighbors| neighbors.contains(&start_node))) {
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
    } else {
        if let Some(neighbors) = adj_list.get(&current) {
            for &neighbor in neighbors {
                if scc.contains(&neighbor) {
                    blocked_map
                        .entry(neighbor)
                        .or_insert_with(FxHashSet::default)
                        .insert(current);
                }
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
            .map_or(false, |neighbors| neighbors.contains(&start_node))) {
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
                        .or_insert_with(FxHashSet::default)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    #[test]
    fn test_find_cycles_empty_graph() {
        let graph = Graph::new(true, false);
        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_find_cycles_single_node_no_loop() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_find_cycles_self_loop() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph
            .add_edge(Edge::new("A".to_string(), "A".to_string(), None, None))
            .unwrap();

        let cycles = find_cycles(&graph, Some(1.0)).unwrap();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 1);
    }

    #[test]
    fn test_find_cycles_simple_cycle() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();

        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("C".to_string(), "A".to_string(), None, None))
            .unwrap();

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 3);
    }

    #[test]
    fn test_find_cycles_two_node_cycle() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();

        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "A".to_string(), None, None))
            .unwrap();

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].len(), 2);
    }

    #[test]
    fn test_find_cycles_multiple_cycles() {
        let mut graph = Graph::new(true, false);
        // Create two separate cycles
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();
        graph.add_node(Node::new("D".to_string(), None)).unwrap();

        // First cycle: A -> B -> A
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "A".to_string(), None, None))
            .unwrap();

        // Second cycle: C -> D -> C
        graph
            .add_edge(Edge::new("C".to_string(), "D".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("D".to_string(), "C".to_string(), None, None))
            .unwrap();

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 2);
    }

    #[test]
    fn test_find_cycles_overlapping() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();

        // Create overlapping cycles: A->B->A and A->B->C->A
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "A".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("C".to_string(), "A".to_string(), None, None))
            .unwrap();

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 2);
    }

    #[test]
    fn test_find_cycles_min_length() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();

        // Two-node cycle
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "A".to_string(), None, None))
            .unwrap();

        // Three-node cycle
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("C".to_string(), "A".to_string(), None, None))
            .unwrap();

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
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();

        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();

        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_find_cycles_undirected_error() {
        let graph = Graph::new(false, false);
        let result = find_cycles(&graph, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_cycle_acyclic_graph() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();

        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();

        assert!(!has_cycle(&graph));
    }

    #[test]
    fn test_has_cycle_cyclic_graph() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();

        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("C".to_string(), "A".to_string(), None, None))
            .unwrap();

        assert!(has_cycle(&graph));
    }
}
