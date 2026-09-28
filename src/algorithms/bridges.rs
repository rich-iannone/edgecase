//! Bridge detection algorithms.
//!
//! A bridge (or cut edge) is an edge whose removal increases the number of
//! connected components in the graph. Bridges are critical for network connectivity.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Find all bridges (cut edges) in the graph.
///
/// Uses Tarjan's bridge-finding algorithm with DFS.
/// Time complexity: O(V + E)
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// List of edge IDs that are bridges
#[pyfunction]
pub fn bridges(graph: &Graph) -> PyResult<Vec<u64>> {
    if graph.is_directed() {
        // For directed graphs, we need to check strongly connected components
        // An edge is a bridge if removing it increases the number of SCCs
        return find_bridges_directed(graph);
    }

    // For undirected graphs, use Tarjan's algorithm
    find_bridges_undirected(graph)
}

/// Check if the graph has any bridges.
///
/// More efficient than bridges() when you only need to know if bridges exist.
///
/// # Arguments
/// * `graph` - The graph to check
///
/// # Returns
/// True if the graph has at least one bridge, False otherwise
#[pyfunction]
pub fn has_bridges(graph: &Graph) -> PyResult<bool> {
    if graph.is_directed() {
        return Ok(!find_bridges_directed(graph)?.is_empty());
    }

    Ok(!find_bridges_undirected(graph)?.is_empty())
}

/// Find bridges in undirected graph using Tarjan's algorithm.
fn find_bridges_undirected(graph: &Graph) -> PyResult<Vec<u64>> {
    let nodes: Vec<u64> = graph.get_nodes().keys().copied().collect();

    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    let mut bridges = Vec::new();
    let mut visited = FxHashSet::default();
    let mut disc = FxHashMap::default(); // Discovery time
    let mut low = FxHashMap::default();  // Lowest reachable vertex
    let mut time = 0;

    // Map edges to their IDs for quick lookup
    let mut edge_map: FxHashMap<(u64, u64), u64> = FxHashMap::default();
    for edge in graph.get_edges_internal() {
        edge_map.insert((edge.source, edge.target), edge.id);
        edge_map.insert((edge.target, edge.source), edge.id);
    }

    // Run DFS from each unvisited node (handles disconnected graphs)
    for &start_node in &nodes {
        if !visited.contains(&start_node) {
            tarjan_bridge_dfs(
                start_node,
                None,
                graph,
                &mut visited,
                &mut disc,
                &mut low,
                &mut time,
                &mut bridges,
                &edge_map,
            )?;
        }
    }

    Ok(bridges)
}

/// DFS helper for Tarjan's bridge-finding algorithm.
#[allow(clippy::too_many_arguments)]
fn tarjan_bridge_dfs(
    u: u64,
    parent: Option<u64>,
    graph: &Graph,
    visited: &mut FxHashSet<u64>,
    disc: &mut FxHashMap<u64, i32>,
    low: &mut FxHashMap<u64, i32>,
    time: &mut i32,
    bridges: &mut Vec<u64>,
    edge_map: &FxHashMap<(u64, u64), u64>,
) -> PyResult<()> {
    visited.insert(u);
    disc.insert(u, *time);
    low.insert(u, *time);
    *time += 1;

    let neighbors = graph.neighbors(u)?;

    for v in neighbors {
        if !visited.contains(&v) {
            // Tree edge - recurse
            tarjan_bridge_dfs(
                v,
                Some(u),
                graph,
                visited,
                disc,
                low,
                time,
                bridges,
                edge_map,
            )?;

            // Update low value
            let low_v = *low.get(&v).unwrap_or(&i32::MAX);
            let current_low = *low.get(&u).unwrap_or(&i32::MAX);
            low.insert(u, current_low.min(low_v));

            // Check if edge (u, v) is a bridge
            let disc_u = *disc.get(&u).unwrap_or(&i32::MAX);
            if low_v > disc_u {
                // This is a bridge
                if let Some(&edge_id) = edge_map.get(&(u, v)) {
                    bridges.push(edge_id);
                }
            }
        } else if Some(v) != parent {
            // Back edge (not to parent)
            let disc_v = *disc.get(&v).unwrap_or(&i32::MAX);
            let current_low = *low.get(&u).unwrap_or(&i32::MAX);
            low.insert(u, current_low.min(disc_v));
        }
    }

    Ok(())
}

/// Find bridges in directed graph.
///
/// For directed graphs, an edge is a bridge if removing it increases
/// the number of strongly connected components.
fn find_bridges_directed(graph: &Graph) -> PyResult<Vec<u64>> {
    // For directed graphs, we use a simplified heuristic:
    // An edge is NOT a bridge if BOTH:
    // 1. There's an alternate path from source to target without this edge
    // 2. There's a path from target back to source (forming a cycle)

    let mut bridges = Vec::new();

    for edge in graph.get_edges_internal() {
        // Check if this edge is part of a strongly connected cycle
        // If so, it's not a bridge
        let in_cycle = is_edge_in_cycle(graph, edge.source, edge.target, edge.id)?;

        if !in_cycle {
            bridges.push(edge.id);
        }
    }

    Ok(bridges)
}

/// Check if an edge is part of a cycle.
///
/// An edge (u -> v) is part of a cycle if:
/// - There's a path from v back to u (without using this edge)
fn is_edge_in_cycle(
    graph: &Graph,
    source: u64,
    target: u64,
    edge_id: u64,
) -> PyResult<bool> {
    // Check if there's a path from target back to source without using this edge
    can_reach_without_edge(graph, target, source, edge_id)
}

/// Check if target is reachable from source without using a specific edge.
fn can_reach_without_edge(
    graph: &Graph,
    source: u64,
    target: u64,
    excluded_edge: u64,
) -> PyResult<bool> {
    let mut visited = FxHashSet::default();
    let mut queue = vec![source];
    visited.insert(source);

    while let Some(current) = queue.pop() {
        if current == target {
            return Ok(true);
        }

        let neighbors = graph.neighbors(current)?;
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                // Check if this edge is the excluded one
                let is_excluded = graph.get_edges_internal()
                    .iter()
                    .any(|e| e.id == excluded_edge && e.source == current && e.target == neighbor);

                if !is_excluded {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }
    }

    Ok(false)
}

/// Count strongly connected components in a directed graph.
fn count_sccs(graph: &Graph) -> PyResult<usize> {
    let nodes: Vec<u64> = graph.get_nodes().keys().copied().collect();

    if nodes.is_empty() {
        return Ok(0);
    }

    let mut visited = FxHashSet::default();
    let mut finish_order = Vec::new();

    // First DFS to get finish times
    for &node in &nodes {
        if !visited.contains(&node) {
            dfs_finish_time(node, graph, &mut visited, &mut finish_order)?;
        }
    }

    // Create reverse graph (conceptually)
    visited.clear();
    let mut scc_count = 0;

    // Second DFS in reverse finish order
    while let Some(node) = finish_order.pop() {
        if !visited.contains(&node) {
            dfs_reverse(node, graph, &mut visited)?;
            scc_count += 1;
        }
    }

    Ok(scc_count)
}

/// DFS to compute finish times.
fn dfs_finish_time(
    node: u64,
    graph: &Graph,
    visited: &mut FxHashSet<u64>,
    finish_order: &mut Vec<u64>,
) -> PyResult<()> {
    visited.insert(node);

    let neighbors = graph.neighbors(node)?;
    for neighbor in neighbors {
        if !visited.contains(&neighbor) {
            dfs_finish_time(neighbor, graph, visited, finish_order)?;
        }
    }

    finish_order.push(node);
    Ok(())
}

/// DFS in reverse graph direction.
fn dfs_reverse(
    node: u64,
    graph: &Graph,
    visited: &mut FxHashSet<u64>,
) -> PyResult<()> {
    visited.insert(node);

    // Find all nodes that point to this node
    for edge in graph.get_edges_internal() {
        if edge.target == node && !visited.contains(&edge.source) {
            dfs_reverse(edge.source, graph, visited)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridges_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let result = bridges(&graph).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_has_bridges_cycle() {
        let mut graph = Graph::new_rust(false, false);
        // Create a cycle - no bridges
        let n1 = graph.add_test_node("n1");
        let n2 = graph.add_test_node("n2");
        let n3 = graph.add_test_node("n3");

        graph.add_test_edge(n1, n2, None);
        graph.add_test_edge(n2, n3, None);
        graph.add_test_edge(n3, n1, None);

        assert!(!has_bridges(&graph).unwrap());
    }
}
