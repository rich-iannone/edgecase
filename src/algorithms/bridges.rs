//! Bridge detection algorithms.
//!
//! A bridge (or cut edge) is an edge whose removal increases the number of
//! connected components in the graph. Bridges are critical for network connectivity.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

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

/// Find bridges in an undirected graph.
///
/// A bridge is exactly a biconnected block made of a single edge, so this reuses the iterative,
/// edge-based block decomposition. Working on edges (not node pairs) means a doubled edge is
/// never a bridge, since removing one copy leaves the other; self-loops are never bridges.
/// Returns edge IDs sorted ascending.
fn find_bridges_undirected(graph: &Graph) -> PyResult<Vec<u64>> {
    let blocks = crate::algorithms::structure_basics::biconnected_components(graph)?;
    let mut bridges: Vec<u64> = blocks
        .into_iter()
        .filter(|(_, edges)| edges.len() == 1)
        .map(|(_, edges)| edges[0])
        .collect();
    bridges.sort_unstable();
    Ok(bridges)
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
fn is_edge_in_cycle(graph: &Graph, source: u64, target: u64, edge_id: u64) -> PyResult<bool> {
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
                let is_excluded = graph
                    .get_edges_internal()
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
