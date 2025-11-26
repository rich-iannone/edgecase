//! Graph regularity testing - checking if all nodes have the same degree
//!
//! A regular graph is one where all nodes have the same degree.
//! A k-regular graph is one where all nodes have degree exactly k.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Check if graph is regular (all nodes have same degree)
///
/// A graph is regular if all nodes have the same degree. For directed graphs,
/// this checks if all nodes have the same in-degree AND the same out-degree
/// (but in-degree and out-degree don't need to be equal to each other).
///
/// # Arguments
///
/// * `graph` - The graph to check
///
/// # Returns
///
/// True if the graph is regular, False otherwise
///
/// # Examples
///
/// ```
/// // Complete graph is regular
/// // Cycle is regular
/// // Star graph is NOT regular (hub has different degree than leaves)
/// ```
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn is_regular(graph: &Graph) -> PyResult<bool> {
    let node_ids = graph.get_node_ids();

    // Empty graph or single node is trivially regular
    if node_ids.len() <= 1 {
        return Ok(true);
    }

    if graph.is_directed() {
        // For directed graphs, check in-degree and out-degree separately
        let (in_degrees, out_degrees) = compute_directed_degrees(graph);

        // Get first node's degrees as reference
        let first_node = node_ids[0];
        let reference_in_degree = in_degrees.get(&first_node).copied().unwrap_or(0);
        let reference_out_degree = out_degrees.get(&first_node).copied().unwrap_or(0);

        // Check if all nodes have same in-degree and same out-degree
        for &node in &node_ids {
            let in_deg = in_degrees.get(&node).copied().unwrap_or(0);
            let out_deg = out_degrees.get(&node).copied().unwrap_or(0);

            if in_deg != reference_in_degree || out_deg != reference_out_degree {
                return Ok(false);
            }
        }

        Ok(true)
    } else {
        // For undirected graphs, just check degree
        let degrees = compute_undirected_degrees(graph);

        // Get first node's degree as reference
        let first_node = node_ids[0];
        let reference_degree = degrees.get(&first_node).copied().unwrap_or(0);

        // Check if all nodes have same degree
        for &node in &node_ids {
            let degree = degrees.get(&node).copied().unwrap_or(0);
            if degree != reference_degree {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// Check if graph is k-regular (all nodes have degree k)
///
/// A graph is k-regular if all nodes have degree exactly k. For directed graphs,
/// this checks if all nodes have in-degree k AND out-degree k.
///
/// # Arguments
///
/// * `graph` - The graph to check
/// * `k` - The degree value to check for
///
/// # Returns
///
/// True if the graph is k-regular, False otherwise
///
/// # Examples
///
/// ```
/// // Complete graph K_n is (n-1)-regular
/// // Cycle is 2-regular
/// // Empty graph is 0-regular
/// ```
#[pyfunction]
#[pyo3(signature = (graph, k))]
pub fn is_k_regular(graph: &Graph, k: usize) -> PyResult<bool> {
    let node_ids = graph.get_node_ids();

    // Empty graph is 0-regular
    if node_ids.is_empty() {
        return Ok(k == 0);
    }

    if graph.is_directed() {
        // For directed graphs, check in-degree and out-degree both equal k
        let (in_degrees, out_degrees) = compute_directed_degrees(graph);

        for &node in &node_ids {
            let in_deg = in_degrees.get(&node).copied().unwrap_or(0);
            let out_deg = out_degrees.get(&node).copied().unwrap_or(0);

            if in_deg != k || out_deg != k {
                return Ok(false);
            }
        }

        Ok(true)
    } else {
        // For undirected graphs, check degree equals k
        let degrees = compute_undirected_degrees(graph);

        for &node in &node_ids {
            let degree = degrees.get(&node).copied().unwrap_or(0);
            if degree != k {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// Compute undirected degrees for all nodes
fn compute_undirected_degrees(graph: &Graph) -> FxHashMap<u64, usize> {
    let mut degrees: FxHashMap<u64, usize> = FxHashMap::default();

    // Initialize all nodes with degree 0
    for &node in &graph.get_node_ids() {
        degrees.insert(node, 0);
    }

    // Count edges
    for edge in graph.get_edges_internal() {
        let source = edge.source;
        let target = edge.target;

        // Self-loop counts as degree 2 in undirected graphs
        if source == target {
            *degrees.entry(source).or_insert(0) += 2;
        } else {
            *degrees.entry(source).or_insert(0) += 1;
            *degrees.entry(target).or_insert(0) += 1;
        }
    }

    degrees
}

/// Compute directed in-degrees and out-degrees for all nodes
fn compute_directed_degrees(graph: &Graph) -> (FxHashMap<u64, usize>, FxHashMap<u64, usize>) {
    let mut in_degrees: FxHashMap<u64, usize> = FxHashMap::default();
    let mut out_degrees: FxHashMap<u64, usize> = FxHashMap::default();

    // Initialize all nodes with degree 0
    for &node in &graph.get_node_ids() {
        in_degrees.insert(node, 0);
        out_degrees.insert(node, 0);
    }

    // Count edges
    for edge in graph.get_edges_internal() {
        let source = edge.source;
        let target = edge.target;

        // Self-loop counts as both in-degree and out-degree
        if source == target {
            *in_degrees.entry(source).or_insert(0) += 1;
            *out_degrees.entry(source).or_insert(0) += 1;
        } else {
            *out_degrees.entry(source).or_insert(0) += 1;
            *in_degrees.entry(target).or_insert(0) += 1;
        }
    }

    (in_degrees, out_degrees)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph_is_regular() {
        let graph = Graph::new();
        assert!(is_regular(&graph).unwrap());
    }

    #[test]
    fn test_single_node_is_regular() {
        let mut graph = Graph::new();
        graph.add_node(Some("A".to_string()), None);
        assert!(is_regular(&graph).unwrap());
    }

    #[test]
    fn test_cycle_is_regular() {
        let mut graph = Graph::new();
        graph.add_cycle_graph(5, None, None, None);
        assert!(is_regular(&graph).unwrap());
    }

    #[test]
    fn test_star_not_regular() {
        let mut graph = Graph::new();
        graph.add_star_graph(4, None, None, None);
        assert!(!is_regular(&graph).unwrap());
    }

    #[test]
    fn test_empty_graph_is_0_regular() {
        let graph = Graph::new();
        assert!(is_k_regular(&graph, 0).unwrap());
        assert!(!is_k_regular(&graph, 1).unwrap());
    }

    #[test]
    fn test_cycle_is_2_regular() {
        let mut graph = Graph::new();
        graph.add_cycle_graph(5, None, None, None);
        assert!(is_k_regular(&graph, 2).unwrap());
        assert!(!is_k_regular(&graph, 3).unwrap());
    }

    #[test]
    fn test_isolated_nodes_are_0_regular() {
        let mut graph = Graph::new();
        graph.add_node(Some("A".to_string()), None);
        graph.add_node(Some("B".to_string()), None);
        graph.add_node(Some("C".to_string()), None);

        assert!(is_regular(&graph).unwrap());
        assert!(is_k_regular(&graph, 0).unwrap());
    }
}
