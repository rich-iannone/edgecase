//! Rich Club Coefficient algorithm
//!
//! The rich-club coefficient measures the connectivity among high-degree nodes.
//! For each degree k, it computes the ratio of actual to potential edges
//! among nodes with degree > k.

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Compute the rich-club coefficient for all degree values.
///
/// The rich-club coefficient φ(k) is defined as:
/// φ(k) = 2 * E_k / (N_k * (N_k - 1))
///
/// where:
/// - N_k is the number of nodes with degree > k
/// - E_k is the number of edges among those nodes
///
/// # Arguments
/// * `graph` - The input graph (should be undirected without self-loops)
///
/// # Returns
/// A HashMap mapping degree k to rich-club coefficient φ(k)
///
/// # Notes
/// - This algorithm ignores edge weights
/// - For directed graphs, uses total degree
/// - Self-loops are ignored in edge counting
#[pyfunction]
pub fn rich_club_coefficient(graph: &Graph) -> PyResult<FxHashMap<usize, f64>> {
    let mut result: FxHashMap<usize, f64> = FxHashMap::default();

    // Compute degree for each node
    // For directed graphs, use total degree (in + out)
    let nodes: Vec<u64> = graph.get_nodes().keys().copied().collect();
    let mut degrees: FxHashMap<u64, usize> = FxHashMap::default();

    for &node_id in &nodes {
        let degree = if graph.is_directed() {
            // Total degree = in-degree + out-degree
            graph.get_in_degree(node_id) + graph.get_out_degree(node_id)
        } else {
            graph.degree_rust(node_id)
        };
        degrees.insert(node_id, degree);
    }

    if nodes.is_empty() {
        return Ok(result);
    }

    // Find the maximum degree
    let max_degree = *degrees.values().max().unwrap_or(&0);

    if max_degree == 0 {
        return Ok(result);
    }

    // For each k from 0 to max_degree - 1, compute the rich club coefficient
    for k in 0..max_degree {
        // Find nodes with degree > k
        let rich_nodes: FxHashSet<u64> = degrees
            .iter()
            .filter(|(_, &deg)| deg > k)
            .map(|(&node_id, _)| node_id)
            .collect();

        let n_k = rich_nodes.len();

        // Need at least 2 nodes to have edges between them
        if n_k < 2 {
            continue;
        }

        // Count edges among rich nodes
        let mut e_k: usize = 0;
        for edge in graph.get_edges_internal() {
            // Skip self-loops
            if edge.source == edge.target {
                continue;
            }

            if rich_nodes.contains(&edge.source) && rich_nodes.contains(&edge.target) {
                e_k += 1;
            }
        }

        // For undirected graphs, each edge is counted once in get_edges_internal
        // For directed graphs, we count each directed edge
        // The formula uses 2*E_k in the numerator for undirected graphs

        // Potential edges = N_k * (N_k - 1) for directed, N_k * (N_k - 1) / 2 for undirected
        // So φ(k) = E_k / (N_k * (N_k - 1) / 2) = 2 * E_k / (N_k * (N_k - 1)) for undirected
        // For directed: φ(k) = E_k / (N_k * (N_k - 1))

        let potential = n_k * (n_k - 1);

        let phi_k = if graph.is_directed() {
            e_k as f64 / potential as f64
        } else {
            // For undirected, each edge appears once, so multiply by 2
            (2.0 * e_k as f64) / potential as f64
        };

        result.insert(k, phi_k);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    fn create_test_graph() -> Graph {
        // Create a simple undirected graph:
        // 1 -- 2 -- 3
        // |    |
        // 4 -- 5
        // Node degrees: 1->2, 2->3, 3->1, 4->2, 5->2
        let mut graph = Graph::new_rust(false, false);

        for i in 1..=5 {
            graph
                .add_node_rust(Node {
                    id: i,
                    label: Some(i.to_string()),
                    attributes: None,
                })
                .unwrap();
        }

        // Edges
        let edges = vec![(1, 2), (2, 3), (1, 4), (4, 5), (2, 5)];
        for (s, t) in edges {
            graph
                .add_edge_rust(Edge {
                    id: 0,
                    source: s,
                    target: t,
                    weight: Some(1.0),
                    attributes: None,
                })
                .unwrap();
        }

        graph
    }

    #[test]
    fn test_rich_club_basic() {
        let graph = create_test_graph();
        let rc = rich_club_coefficient(&graph).unwrap();

        // With degrees [2, 3, 1, 2, 2]:
        // k=0: all 5 nodes (degree > 0), potential = 5*4 = 20, edges among them = 5
        //      φ(0) = 2*5/20 = 0.5
        // k=1: nodes with degree > 1 are [1, 2, 4, 5] (4 nodes), edges among them = 4 (1-2, 1-4, 4-5, 2-5)
        //      potential = 4*3 = 12, φ(1) = 2*4/12 = 0.667
        // k=2: nodes with degree > 2 are [2] (1 node), can't compute

        assert!(rc.contains_key(&0));
        assert!(rc.contains_key(&1));
        assert!(!rc.contains_key(&2)); // Only 1 node with degree > 2
    }

    #[test]
    fn test_rich_club_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let rc = rich_club_coefficient(&graph).unwrap();
        assert!(rc.is_empty());
    }

    #[test]
    fn test_rich_club_complete_graph() {
        // K4: 4 nodes, all connected
        // Each node has degree 3
        let mut graph = Graph::new_rust(false, false);

        for i in 1..=4 {
            graph
                .add_node_rust(Node {
                    id: i,
                    label: Some(i.to_string()),
                    attributes: None,
                })
                .unwrap();
        }

        // All pairs
        for i in 1..=4 {
            for j in (i + 1)..=4 {
                graph
                    .add_edge_rust(Edge {
                        id: 0,
                        source: i,
                        target: j,
                        weight: Some(1.0),
                        attributes: None,
                    })
                    .unwrap();
            }
        }

        let rc = rich_club_coefficient(&graph).unwrap();

        // All nodes have degree 3
        // k=0, 1, 2: all 4 nodes, 6 edges, potential = 12, φ = 1.0
        for k in 0..3 {
            assert!((rc.get(&k).unwrap() - 1.0).abs() < 1e-10);
        }
    }
}
