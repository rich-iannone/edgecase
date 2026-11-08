//! Graph matching algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

/// Check if a set of edges forms a valid matching.
///
/// A matching is a set of edges where no two edges share a common endpoint.
///
/// # Arguments
/// * `graph` - The input graph
/// * `edges` - List of edge IDs to check
///
/// # Returns
/// True if the edges form a valid matching, False otherwise
#[pyfunction]
pub fn is_matching(graph: &Graph, edges: Vec<u64>) -> PyResult<bool> {
    let mut endpoints = FxHashSet::default();

    for edge_id in edges {
        if let Some(edge) = graph.get_edge(edge_id) {
            // Check if either endpoint has been seen before
            if endpoints.contains(&edge.source) || endpoints.contains(&edge.target) {
                return Ok(false);
            }
            endpoints.insert(edge.source);
            endpoints.insert(edge.target);
        } else {
            // Edge doesn't exist
            return Ok(false);
        }
    }

    Ok(true)
}

/// Find a maximal matching in the graph.
///
/// A maximal matching is a matching that cannot be extended by adding another edge.
/// Uses a greedy algorithm that processes edges in creation (edge ID) order, so the
/// result is deterministic for a given graph but depends on the order edges were added.
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// List of edge IDs forming a maximal matching
#[pyfunction]
pub fn maximal_matching(graph: &Graph) -> PyResult<Vec<u64>> {
    let mut matched_nodes = FxHashSet::default();
    let mut matching = Vec::new();

    // Process edges in order
    for edge in graph.get_edges_internal() {
        // Skip if either endpoint is already matched
        if matched_nodes.contains(&edge.source) || matched_nodes.contains(&edge.target) {
            continue;
        }

        // Add edge to matching
        matching.push(edge.id);
        matched_nodes.insert(edge.source);
        matched_nodes.insert(edge.target);
    }

    Ok(matching)
}

/// Find a maximum weight matching in the graph.
///
/// Uses a greedy algorithm that selects edges by weight in descending order.
/// For unweighted graphs, all edges have weight 1.0.
///
/// Note: This is a greedy approximation, not guaranteed to be optimal.
/// For bipartite graphs, use specialized bipartite matching algorithms.
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// List of edge IDs forming a maximum weight matching
#[pyfunction]
pub fn max_weight_matching(graph: &Graph) -> PyResult<Vec<u64>> {
    let mut matched_nodes = FxHashSet::default();
    let mut matching = Vec::new();

    // Collect edges with weights
    let mut weighted_edges: Vec<(u64, f64)> = graph
        .get_edges_internal()
        .iter()
        .map(|e| {
            let weight = if graph.is_weighted() {
                e.weight.unwrap_or(1.0)
            } else {
                1.0
            };
            (e.id, weight)
        })
        .collect();

    // Sort by weight descending
    weighted_edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Greedily select edges
    for (edge_id, _weight) in weighted_edges {
        if let Some(edge) = graph.get_edge(edge_id) {
            // Skip if either endpoint is already matched
            if matched_nodes.contains(&edge.source) || matched_nodes.contains(&edge.target) {
                continue;
            }

            // Add edge to matching
            matching.push(edge.id);
            matched_nodes.insert(edge.source);
            matched_nodes.insert(edge.target);
        }
    }

    Ok(matching)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_matching_valid() {
        let mut graph = Graph::new_rust(false, false);

        // Create a simple graph: 1-2, 3-4, 5-6
        let n1 = graph.add_test_node("1");
        let n2 = graph.add_test_node("2");
        let n3 = graph.add_test_node("3");
        let n4 = graph.add_test_node("4");

        let e1 = graph.add_test_edge(n1, n2, Some(1.0));
        let e2 = graph.add_test_edge(n3, n4, Some(1.0));

        // These edges form a valid matching
        assert!(is_matching(&graph, vec![e1, e2]).unwrap());
    }

    #[test]
    fn test_is_matching_invalid() {
        let mut graph = Graph::new_rust(false, false);

        // Create a simple graph with shared endpoint
        let n1 = graph.add_test_node("1");
        let n2 = graph.add_test_node("2");
        let n3 = graph.add_test_node("3");

        let e1 = graph.add_test_edge(n1, n2, Some(1.0));
        let e2 = graph.add_test_edge(n2, n3, Some(1.0));

        // These edges share node n2, not a valid matching
        assert!(!is_matching(&graph, vec![e1, e2]).unwrap());
    }

    #[test]
    fn test_maximal_matching() {
        let mut graph = Graph::new_rust(false, false);

        // Create a path: 1-2-3-4
        let n1 = graph.add_test_node("1");
        let n2 = graph.add_test_node("2");
        let n3 = graph.add_test_node("3");
        let n4 = graph.add_test_node("4");

        graph.add_test_edge(n1, n2, Some(1.0));
        graph.add_test_edge(n2, n3, Some(1.0));
        graph.add_test_edge(n3, n4, Some(1.0));

        let matching = maximal_matching(&graph).unwrap();

        // Should get 2 edges (maximal but not maximum)
        assert!(!matching.is_empty());
        assert!(is_matching(&graph, matching).unwrap());
    }

    #[test]
    fn test_max_weight_matching() {
        let mut graph = Graph::new_rust(false, true);

        // Create a simple weighted graph
        let n1 = graph.add_test_node("1");
        let n2 = graph.add_test_node("2");
        let n3 = graph.add_test_node("3");
        let n4 = graph.add_test_node("4");

        // Heavy edge
        graph.add_test_edge(n1, n2, Some(10.0));
        // Light edges
        graph.add_test_edge(n2, n3, Some(1.0));
        graph.add_test_edge(n3, n4, Some(1.0));

        let matching = max_weight_matching(&graph).unwrap();

        // Should select heavy edge first
        assert!(!matching.is_empty());
        assert!(is_matching(&graph, matching).unwrap());
    }
}
