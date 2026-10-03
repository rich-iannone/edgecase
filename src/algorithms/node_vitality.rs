//! Node vitality computation - measuring the impact of removing a node
//!
//! Node vitality measures the importance of a node by quantifying how much
//! the graph's total distance increases when the node is removed.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Compute vitality for a single node
///
/// Node vitality is defined as:
/// vitality(v) = sum_distances(G) - sum_distances(G - v)
///
/// This measures how much removing the node increases total path lengths.
/// Higher values indicate more important nodes whose removal would significantly
/// impact graph connectivity.
///
/// # Arguments
///
/// * `graph` - The graph to analyze
/// * `node` - The node ID to compute vitality for
///
/// # Returns
///
/// The vitality score for the node (change in sum of distances)
///
/// # Complexity
///
/// O(V * E) where V is number of vertices and E is number of edges
#[pyfunction]
#[pyo3(signature = (graph, node))]
pub fn node_vitality(graph: &Graph, node: u64) -> PyResult<f64> {
    // Verify node exists
    let node_ids = graph.get_node_ids();
    if !node_ids.contains(&node) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Node {} does not exist in graph",
            node
        )));
    }

    // If graph has only one node, vitality is 0
    if node_ids.len() == 1 {
        return Ok(0.0);
    }

    // Compute sum of distances in original graph
    let original_sum = compute_sum_of_distances(graph, None);

    // Compute sum of distances with node removed
    let without_node_sum = compute_sum_of_distances(graph, Some(node));

    // Vitality is the difference
    Ok(original_sum - without_node_sum)
}

/// Compute sum of all pairwise shortest path distances
///
/// # Arguments
///
/// * `graph` - The graph to analyze
/// * `exclude_node` - Optional node to exclude (simulate its removal)
///
/// # Returns
///
/// Sum of all finite shortest path distances
fn compute_sum_of_distances(graph: &Graph, exclude_node: Option<u64>) -> f64 {
    let node_ids = graph.get_node_ids();
    let mut total_distance = 0.0;

    // For each source node
    for &source in &node_ids {
        // Skip if this is the excluded node
        if Some(source) == exclude_node {
            continue;
        }

        // BFS from this source to compute distances
        let distances = bfs_distances(graph, source, exclude_node);

        // Sum up all finite distances
        for (&target, &dist) in &distances {
            // Skip self-distances and excluded node
            if target != source && Some(target) != exclude_node
                && dist < f64::INFINITY {
                    total_distance += dist;
                }
        }
    }

    total_distance
}

/// BFS to compute shortest path distances from a source node
///
/// # Arguments
///
/// * `graph` - The graph to traverse
/// * `source` - Source node for BFS
/// * `exclude_node` - Optional node to exclude from traversal
///
/// # Returns
///
/// Map of node IDs to their shortest distances from source
fn bfs_distances(
    graph: &Graph,
    source: u64,
    exclude_node: Option<u64>,
) -> FxHashMap<u64, f64> {
    let mut distances: FxHashMap<u64, f64> = FxHashMap::default();
    let mut queue = VecDeque::new();

    // Initialize source
    distances.insert(source, 0.0);
    queue.push_back(source);

    while let Some(current) = queue.pop_front() {
        let current_dist = distances[&current];

        // Get neighbors based on graph directionality
        let neighbors = if graph.is_directed() {
            // For directed graphs, only follow outgoing edges
            get_out_neighbors(graph, current, exclude_node)
        } else {
            // For undirected graphs, get all neighbors
            get_neighbors(graph, current, exclude_node)
        };

        for neighbor in neighbors {
            // Skip if already visited
            if let std::collections::hash_map::Entry::Vacant(e) = distances.entry(neighbor) {
                e.insert(current_dist + 1.0);
                queue.push_back(neighbor);
            }
        }
    }

    // Set unreachable nodes to infinity
    for &node in &graph.get_node_ids() {
        if node != source && Some(node) != exclude_node && !distances.contains_key(&node) {
            distances.insert(node, f64::INFINITY);
        }
    }

    distances
}

/// Get neighbors of a node (undirected graph)
///
/// # Arguments
///
/// * `graph` - The graph
/// * `node` - Node to get neighbors for
/// * `exclude_node` - Optional node to exclude
///
/// # Returns
///
/// Vector of neighbor node IDs
fn get_neighbors(graph: &Graph, node: u64, exclude_node: Option<u64>) -> Vec<u64> {
    let mut neighbors = Vec::new();

    // Get all edges
    let edges = graph.get_edges_internal();

    for edge in edges {
        let source = edge.source;
        let target = edge.target;

        // Skip edges involving excluded node
        if Some(source) == exclude_node || Some(target) == exclude_node {
            continue;
        }

        // Add neighbor if this edge connects to our node
        if source == node && target != node {
            neighbors.push(target);
        } else if target == node && source != node {
            neighbors.push(source);
        }
    }

    neighbors
}

/// Get outgoing neighbors of a node (directed graph)
///
/// # Arguments
///
/// * `graph` - The graph
/// * `node` - Node to get outgoing neighbors for
/// * `exclude_node` - Optional node to exclude
///
/// # Returns
///
/// Vector of outgoing neighbor node IDs
fn get_out_neighbors(graph: &Graph, node: u64, exclude_node: Option<u64>) -> Vec<u64> {
    let mut neighbors = Vec::new();

    // Get all edges
    let edges = graph.get_edges_internal();

    for edge in edges {
        let source = edge.source;
        let target = edge.target;

        // Skip edges involving excluded node
        if Some(source) == exclude_node || Some(target) == exclude_node {
            continue;
        }

        // Only add targets where source is our node
        if source == node && target != node {
            neighbors.push(target);
        }
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_vitality_single_node() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_test_node("A");
        let node_id = graph.id_from_label("A").unwrap();

        let vitality = node_vitality(&graph, node_id).unwrap();
        assert_eq!(vitality, 0.0);
    }

    #[test]
    fn test_node_vitality_path() {
        let mut graph = Graph::new_rust(false, false);
        let nodes = graph.add_test_path(3);

        // Middle node is most vital (connects the path)
        let middle_vitality = node_vitality(&graph, nodes[1]).unwrap();
        let end_vitality = node_vitality(&graph, nodes[0]).unwrap();

        // Removing middle node disconnects graph (infinite distances)
        // Removing end node just removes 2 paths
        assert!(middle_vitality.is_infinite() || middle_vitality > end_vitality);
    }

    #[test]
    fn test_node_vitality_star() {
        let mut graph = Graph::new_rust(false, false);
        let nodes = graph.add_test_star(4);

        // First node is hub in star graph
        let hub_vitality = node_vitality(&graph, nodes[0]).unwrap();
        let leaf_vitality = node_vitality(&graph, nodes[1]).unwrap();

        // Hub is most vital (removal disconnects all leaves)
        assert!(hub_vitality.is_infinite() || hub_vitality > leaf_vitality);
    }
}
