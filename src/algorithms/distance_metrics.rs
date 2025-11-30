//! Distance and spectral metrics for graphs.
//!
//! This module provides:
//! - `wiener_index`: Sum of all pairwise shortest path distances
//! - `estrada_index`: Sum of exponentials of adjacency matrix eigenvalues
//!
//! References:
//! - Wiener, H. (1947). Structural determination of paraffin boiling points.
//!   Journal of the American Chemical Society, 69(1), 17-20.
//! - Estrada, E. (2000). Characterization of 3D molecular structure.
//!   Chemical Physics Letters, 319(5-6), 713-718.

use crate::graph::Graph;
use nalgebra::{DMatrix, SymmetricEigen};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Compute the Wiener index of a graph.
///
/// The Wiener index is the sum of all pairwise shortest path distances:
/// W(G) = Σ_{u < v} d(u, v)
///
/// For disconnected graphs, returns infinity (f64::INFINITY) since some
/// pairs have infinite distance.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `weight` - Optional edge weight attribute. If None, uses unweighted distances.
///
/// # Returns
/// The Wiener index as a float. Returns 0.0 for empty graphs or single-node graphs.
/// Returns infinity for disconnected graphs.
///
/// # Notes
/// - For undirected graphs, each pair is counted once (u < v)
/// - For directed graphs, considers all ordered pairs (u, v) where u != v
/// - Time complexity: O(V * (V + E)) for unweighted, O(V * (V + E) log V) for weighted
#[pyfunction]
#[pyo3(signature = (graph, weight=None))]
pub fn wiener_index(graph: &Graph, weight: Option<&str>) -> PyResult<f64> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n <= 1 {
        return Ok(0.0);
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    let use_weights = weight.is_some() && graph.is_weighted();
    let is_directed = graph.is_directed();

    let mut total_distance = 0.0;

    // For each source node, compute distances to all other nodes
    for (i, &source) in node_ids.iter().enumerate() {
        let distances = if use_weights {
            single_source_dijkstra(graph, source, &node_ids, &node_to_idx)
        } else {
            single_source_bfs(graph, source, &node_ids, &node_to_idx)
        };

        // Sum distances
        if is_directed {
            // For directed graphs, sum all distances to other nodes
            for (j, &dist) in distances.iter().enumerate() {
                if i != j {
                    if dist.is_infinite() {
                        return Ok(f64::INFINITY);
                    }
                    total_distance += dist;
                }
            }
        } else {
            // For undirected graphs, only count pairs where i < j to avoid double counting
            for j in (i + 1)..n {
                let dist = distances[j];
                if dist.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total_distance += dist;
            }
        }
    }

    Ok(total_distance)
}

/// Helper: BFS for unweighted single-source shortest paths
fn single_source_bfs(
    graph: &Graph,
    source: u64,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
) -> Vec<f64> {
    let n = node_ids.len();
    let mut distances = vec![f64::INFINITY; n];
    let source_idx = node_to_idx[&source];
    distances[source_idx] = 0.0;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(current) = queue.pop_front() {
        let current_idx = node_to_idx[&current];
        let current_dist = distances[current_idx];

        if let Ok(neighbors) = graph.neighbors(current) {
            for neighbor in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                    if distances[neighbor_idx].is_infinite() {
                        distances[neighbor_idx] = current_dist + 1.0;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }

    distances
}

/// Get edge weight, checking both directions for undirected graphs
#[inline]
fn get_edge_weight_bidirectional(graph: &Graph, from: u64, to: u64) -> f64 {
    // Try forward direction first
    let weight = graph.get_edge_weight(from, to);
    if weight != 1.0 || graph.is_directed() {
        // Found a weight or it's directed (default 1.0 is correct)
        return weight;
    }
    // For undirected graphs with default weight, try reverse direction
    graph.get_edge_weight(to, from)
}

/// Helper: Dijkstra for weighted single-source shortest paths
fn single_source_dijkstra(
    graph: &Graph,
    source: u64,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
) -> Vec<f64> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct State {
        cost: f64,
        node_idx: usize,
    }

    impl Eq for State {}

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .cost
                .partial_cmp(&self.cost)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = node_ids.len();
    let mut distances = vec![f64::INFINITY; n];
    let source_idx = node_to_idx[&source];
    distances[source_idx] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0.0,
        node_idx: source_idx,
    });

    while let Some(State { cost, node_idx }) = heap.pop() {
        if cost > distances[node_idx] {
            continue;
        }

        let node = node_ids[node_idx];
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                    let edge_weight = get_edge_weight_bidirectional(graph, node, neighbor);
                    let new_cost = cost + edge_weight;

                    if new_cost < distances[neighbor_idx] {
                        distances[neighbor_idx] = new_cost;
                        heap.push(State {
                            cost: new_cost,
                            node_idx: neighbor_idx,
                        });
                    }
                }
            }
        }
    }

    distances
}

/// Compute the Estrada index of a graph.
///
/// The Estrada index is defined as the sum of exponentials of the eigenvalues
/// of the adjacency matrix:
/// EE(G) = Σ_i exp(λ_i)
///
/// where λ_i are the eigenvalues of the adjacency matrix.
///
/// The Estrada index is related to the subgraph centrality and counts closed
/// walks of all lengths, with shorter walks weighted more heavily.
///
/// # Arguments
/// * `graph` - The graph to analyze (must be undirected)
///
/// # Returns
/// The Estrada index as a float. Returns 0.0 for empty graphs.
///
/// # Errors
/// Returns an error if the graph is directed.
///
/// # Notes
/// - The Estrada index is always >= n (number of nodes) for any graph
/// - For a complete graph Kn: EE = (n-1)*exp(-1) + exp(n-1)
/// - For an empty graph (no edges): EE = n (all eigenvalues are 0)
/// - Time complexity: O(n³) for eigendecomposition
///
/// # References
/// - Estrada, E. (2000). Characterization of 3D molecular structure.
///   Chemical Physics Letters, 319(5-6), 713-718.
/// - Estrada, E., & Rodríguez-Velázquez, J. A. (2005). Subgraph centrality
///   in complex networks. Physical Review E, 71(5), 056103.
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn estrada_index(graph: &Graph) -> PyResult<f64> {
    // Validate graph type
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Estrada index is only defined for undirected graphs",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(0.0);
    }

    // Single node with no edges: eigenvalue is 0, exp(0) = 1
    if n == 1 {
        return Ok(1.0);
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Build adjacency matrix
    let mut adj_matrix = DMatrix::<f64>::zeros(n, n);

    for edge in graph.get_edges_internal() {
        if let (Some(&i), Some(&j)) = (
            node_to_idx.get(&edge.source),
            node_to_idx.get(&edge.target),
        ) {
            // Skip self-loops for Estrada index computation
            if i != j {
                // Undirected: symmetric
                adj_matrix[(i, j)] = 1.0;
                adj_matrix[(j, i)] = 1.0;
            }
        }
    }

    // Compute eigendecomposition
    let eigen = SymmetricEigen::new(adj_matrix);
    let eigenvalues = &eigen.eigenvalues;

    // Sum exp(λ_i) for all eigenvalues
    let estrada: f64 = eigenvalues.iter().map(|&lambda| lambda.exp()).sum();

    Ok(estrada)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    fn create_path_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        for i in 0..n {
            let node = Node {
                id: i as u64,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        for i in 0..(n - 1) {
            let edge = Edge {
                id: 0,
                source: i as u64,
                target: (i + 1) as u64,
                weight: None,
                attributes: None,
            };
            graph.add_edge_rust(edge).unwrap();
        }
        graph
    }

    fn create_complete_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        for i in 0..n {
            let node = Node {
                id: i as u64,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let edge = Edge {
                    id: 0,
                    source: i as u64,
                    target: j as u64,
                    weight: None,
                    attributes: None,
                };
                graph.add_edge_rust(edge).unwrap();
            }
        }
        graph
    }

    fn create_cycle_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        for i in 0..n {
            let node = Node {
                id: i as u64,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        for i in 0..n {
            let edge = Edge {
                id: 0,
                source: i as u64,
                target: ((i + 1) % n) as u64,
                weight: None,
                attributes: None,
            };
            graph.add_edge_rust(edge).unwrap();
        }
        graph
    }

    #[test]
    fn test_wiener_index_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_wiener_index_single_node() {
        let mut graph = Graph::new_rust(false, false);
        let node = Node {
            id: 1,
            label: None,
            attributes: None,
        };
        graph.add_node_rust(node).unwrap();
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_wiener_index_path_graph() {
        // Path P4: 0-1-2-3
        // Distances: d(0,1)=1, d(0,2)=2, d(0,3)=3, d(1,2)=1, d(1,3)=2, d(2,3)=1
        // Wiener index = 1+2+3+1+2+1 = 10
        let graph = create_path_graph(4);
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_wiener_index_complete_graph() {
        // Complete K4: all pairs have distance 1
        // Number of pairs = 4*3/2 = 6
        // Wiener index = 6
        let graph = create_complete_graph(4);
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_wiener_index_cycle_graph() {
        // Cycle C4: 0-1-2-3-0
        // Distances: d(0,1)=1, d(0,2)=2, d(0,3)=1, d(1,2)=1, d(1,3)=2, d(2,3)=1
        // Wiener index = 1+2+1+1+2+1 = 8
        let graph = create_cycle_graph(4);
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 8.0);
    }

    #[test]
    fn test_wiener_index_disconnected() {
        // Two isolated nodes
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(Node {
                id: 1,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_node_rust(Node {
                id: 2,
                label: None,
                attributes: None,
            })
            .unwrap();
        let result = wiener_index(&graph, None).unwrap();
        assert!(result.is_infinite());
    }

    #[test]
    fn test_estrada_index_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let result = estrada_index(&graph).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_estrada_index_single_node() {
        let mut graph = Graph::new_rust(false, false);
        let node = Node {
            id: 1,
            label: None,
            attributes: None,
        };
        graph.add_node_rust(node).unwrap();
        let result = estrada_index(&graph).unwrap();
        assert_eq!(result, 1.0); // exp(0) = 1
    }

    #[test]
    fn test_estrada_index_two_nodes_no_edge() {
        // Two isolated nodes: both eigenvalues are 0
        // EE = exp(0) + exp(0) = 2
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(Node {
                id: 1,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_node_rust(Node {
                id: 2,
                label: None,
                attributes: None,
            })
            .unwrap();
        let result = estrada_index(&graph).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_estrada_index_two_nodes_with_edge() {
        // Edge graph K2: eigenvalues are 1 and -1
        // EE = exp(1) + exp(-1) ≈ 2.718 + 0.368 ≈ 3.086
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(Node {
                id: 1,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_node_rust(Node {
                id: 2,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_edge_rust(Edge {
                id: 0,
                source: 1,
                target: 2,
                weight: None,
                attributes: None,
            })
            .unwrap();
        let result = estrada_index(&graph).unwrap();
        let expected = 1.0_f64.exp() + (-1.0_f64).exp();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_estrada_index_path_graph() {
        // Path P3: eigenvalues are -√2, 0, √2
        // EE = exp(-√2) + exp(0) + exp(√2)
        let graph = create_path_graph(3);
        let result = estrada_index(&graph).unwrap();
        let sqrt2 = 2.0_f64.sqrt();
        let expected = (-sqrt2).exp() + 1.0 + sqrt2.exp();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_estrada_index_complete_graph() {
        // Complete K3: eigenvalues are 2, -1, -1
        // EE = exp(2) + exp(-1) + exp(-1)
        let graph = create_complete_graph(3);
        let result = estrada_index(&graph).unwrap();
        let expected = 2.0_f64.exp() + 2.0 * (-1.0_f64).exp();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_estrada_index_directed_error() {
        let graph = Graph::new_rust(true, false);
        let result = estrada_index(&graph);
        assert!(result.is_err());
    }
}
