//! Communicability algorithms for graphs.
//!
//! Communicability measures the "amount of communication" between pairs of nodes
//! based on all walks between them, with shorter walks weighted more heavily.
//! It uses the matrix exponential of the adjacency matrix.
//!
//! References:
//! - Estrada, E., & Hatano, N. (2008). Communicability in complex networks.
//!   Physical Review E, 77(3), 036111.
//! - Estrada, E., Higham, D. J., & Hatano, N. (2009). Communicability betweenness
//!   in complex networks. Physica A, 388(5), 764-774.

use crate::graph::Graph;
use nalgebra::{DMatrix, SymmetricEigen};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Compute communicability between all pairs of nodes.
///
/// The communicability between nodes u and v is defined as the (u,v) entry
/// of the matrix exponential of the adjacency matrix: exp(A)_{uv}.
///
/// Using spectral decomposition: C(u,v) = Σ_j φ_j(u) * φ_j(v) * exp(λ_j)
/// where φ_j are eigenvectors and λ_j are eigenvalues.
///
/// # Arguments
/// * `graph` - The graph to analyze (must be undirected and simple)
///
/// # Returns
/// A dictionary of dictionaries: {node_u: {node_v: communicability_value}}
///
/// # Errors
/// Returns an error if the graph is directed or has self-loops/multi-edges.
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn communicability(graph: &Graph) -> PyResult<FxHashMap<u64, FxHashMap<u64, f64>>> {
    // Validate graph type
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Communicability is only defined for undirected graphs",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
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
            // Check for self-loops
            if i == j {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Communicability is not defined for graphs with self-loops",
                ));
            }
            // Undirected: symmetric
            adj_matrix[(i, j)] = 1.0;
            adj_matrix[(j, i)] = 1.0;
        }
    }

    // Compute eigendecomposition of adjacency matrix
    let eigen = SymmetricEigen::new(adj_matrix);
    let eigenvalues = &eigen.eigenvalues;
    let eigenvectors = &eigen.eigenvectors;

    // Compute matrix exponential using spectral decomposition:
    // exp(A) = V * diag(exp(λ)) * V^T
    // For entry (u, v): exp(A)_{uv} = Σ_j φ_j(u) * φ_j(v) * exp(λ_j)

    let mut result: FxHashMap<u64, FxHashMap<u64, f64>> = FxHashMap::default();

    for i in 0..n {
        let node_u = node_ids[i];
        let mut inner_map: FxHashMap<u64, f64> = FxHashMap::default();

        for j in 0..n {
            let node_v = node_ids[j];

            // Compute communicability between i and j
            let mut comm_value = 0.0;
            for k in 0..n {
                let phi_u = eigenvectors[(i, k)];
                let phi_v = eigenvectors[(j, k)];
                let exp_lambda = eigenvalues[k].exp();
                comm_value += phi_u * phi_v * exp_lambda;
            }

            inner_map.insert(node_v, comm_value);
        }

        result.insert(node_u, inner_map);
    }

    Ok(result)
}

/// Compute communicability betweenness centrality for all nodes.
///
/// This centrality measure quantifies the contribution of each node to the
/// communicability between all other pairs of nodes. It's based on the formula:
///
/// ω_r = (1/C) * Σ_{p≠q, q≠r} (G_prq / G_pq)
///
/// where:
/// - G_pq = exp(A)_{pq} is the communicability between p and q
/// - G_prq = exp(A)_{pq} - exp(A+E_r)_{pq} is the decrease in communicability
///   when node r is "removed" (edges zeroed)
/// - C = (n-1)² - (n-1) is a normalization factor
///
/// # Arguments
/// * `graph` - The graph to analyze (must be undirected and simple)
///
/// # Returns
/// A dictionary mapping node IDs to their communicability betweenness centrality.
///
/// # Errors
/// Returns an error if the graph is directed or has self-loops.
///
/// # References
/// Estrada, E., Higham, D. J., & Hatano, N. (2009). "Communicability betweenness
/// in complex networks". Physica A, 388(5), 764-774.
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn communicability_betweenness_centrality(graph: &Graph) -> PyResult<FxHashMap<u64, f64>> {
    // Validate graph type
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Communicability betweenness centrality is only defined for undirected graphs",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    if n == 1 {
        let mut result = FxHashMap::default();
        result.insert(node_ids[0], 0.0);
        return Ok(result);
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
            // Check for self-loops
            if i == j {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Communicability betweenness is not defined for graphs with self-loops",
                ));
            }
            // Undirected: symmetric
            adj_matrix[(i, j)] = 1.0;
            adj_matrix[(j, i)] = 1.0;
        }
    }

    // Compute exp(A) - the communicability matrix for the full graph
    let exp_a = matrix_exp_symmetric(&adj_matrix);

    // Normalization factor C = (n-1)^2 - (n-1)
    let n_f64 = n as f64;
    let normalization = (n_f64 - 1.0).powi(2) - (n_f64 - 1.0);

    let mut result = FxHashMap::default();

    // For each node r, compute its communicability betweenness
    for r in 0..n {
        let node_r = node_ids[r];

        // Create modified adjacency matrix A + E(r)
        // E(r) zeros out all edges connected to node r
        let mut adj_modified = adj_matrix.clone();
        for i in 0..n {
            adj_modified[(r, i)] = 0.0;
            adj_modified[(i, r)] = 0.0;
        }

        // Compute exp(A + E(r))
        let exp_a_modified = matrix_exp_symmetric(&adj_modified);

        // Compute betweenness: sum over all pairs (p, q) where p ≠ q and q ≠ r and p ≠ r
        let mut omega = 0.0;

        for p in 0..n {
            if p == r {
                continue;
            }
            for q in 0..n {
                if q == r || p == q {
                    continue;
                }

                let g_pq = exp_a[(p, q)];
                let g_prq = g_pq - exp_a_modified[(p, q)];

                if g_pq.abs() > 1e-15 {
                    omega += g_prq / g_pq;
                }
            }
        }

        // Normalize
        let centrality = if normalization > 0.0 {
            omega / normalization
        } else {
            0.0
        };

        result.insert(node_r, centrality);
    }

    Ok(result)
}

/// Compute the matrix exponential of a symmetric matrix using eigendecomposition.
///
/// For a symmetric matrix A with eigendecomposition A = V * D * V^T,
/// exp(A) = V * diag(exp(λ_i)) * V^T
fn matrix_exp_symmetric(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let n = matrix.nrows();

    if n == 0 {
        return DMatrix::<f64>::zeros(0, 0);
    }

    // Compute eigendecomposition
    let eigen = SymmetricEigen::new(matrix.clone());
    let eigenvalues = &eigen.eigenvalues;
    let eigenvectors = &eigen.eigenvectors;

    // Compute exp(A) = V * diag(exp(λ)) * V^T
    let mut result = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let mut value = 0.0;
            for k in 0..n {
                value += eigenvectors[(i, k)] * eigenvectors[(j, k)] * eigenvalues[k].exp();
            }
            result[(i, j)] = value;
        }
    }

    result
}

/// Compute the matrix exponential of the adjacency matrix.
///
/// This is primarily used for computing communicability but can be useful
/// for other applications involving the exponential of adjacency matrices.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// A dictionary mapping (node_u, node_v) tuples to exp(A)_{uv} values.
/// Only returns entries for existing node pairs (all pairs for dense representation).
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn communicability_exp(graph: &Graph) -> PyResult<FxHashMap<u64, FxHashMap<u64, f64>>> {
    // This is equivalent to communicability() but explicitly named for clarity
    communicability(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_path_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        for i in 0..n as u64 {
            graph
                .add_node_rust(crate::graph::Node {
                    id: i + 1,
                    label: Some((i + 1).to_string()),
                    attributes: None,
                })
                .unwrap();
        }
        for i in 1..n as u64 {
            graph
                .add_edge_rust(crate::graph::Edge {
                    id: 0,
                    source: i,
                    target: i + 1,
                    weight: None,
                    attributes: None,
                })
                .unwrap();
        }
        graph
    }

    fn create_star_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        // Center node is 1
        for i in 0..=n as u64 {
            graph
                .add_node_rust(crate::graph::Node {
                    id: i + 1,
                    label: Some((i + 1).to_string()),
                    attributes: None,
                })
                .unwrap();
        }
        // Connect all leaves to center (node 1)
        for i in 2..=(n + 1) as u64 {
            graph
                .add_edge_rust(crate::graph::Edge {
                    id: 0,
                    source: 1,
                    target: i,
                    weight: None,
                    attributes: None,
                })
                .unwrap();
        }
        graph
    }

    #[test]
    fn test_communicability_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let result = communicability(&graph).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_communicability_single_node() {
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(crate::graph::Node {
                id: 1,
                label: Some("1".to_string()),
                attributes: None,
            })
            .unwrap();

        let result = communicability(&graph).unwrap();
        assert_eq!(result.len(), 1);
        // Self-communicability of isolated node is 1 (exp(0) = 1)
        let comm_11 = result[&1][&1];
        assert!((comm_11 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_communicability_two_nodes() {
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(crate::graph::Node {
                id: 1,
                label: Some("1".to_string()),
                attributes: None,
            })
            .unwrap();
        graph
            .add_node_rust(crate::graph::Node {
                id: 2,
                label: Some("2".to_string()),
                attributes: None,
            })
            .unwrap();
        graph
            .add_edge_rust(crate::graph::Edge {
                id: 0,
                source: 1,
                target: 2,
                weight: None,
                attributes: None,
            })
            .unwrap();

        let result = communicability(&graph).unwrap();

        // For a two-node graph with edge, the adjacency matrix is [[0,1],[1,0]]
        // Eigenvalues: 1, -1. Eigenvectors: [1/√2, 1/√2], [1/√2, -1/√2]
        // exp(A)_11 = (1/2)(e^1 + e^{-1}) = cosh(1) ≈ 1.543
        // exp(A)_12 = (1/2)(e^1 - e^{-1}) = sinh(1) ≈ 1.175
        let comm_11 = result[&1][&1];
        let comm_12 = result[&1][&2];
        let comm_22 = result[&2][&2];

        assert!((comm_11 - 1.0_f64.cosh()).abs() < 1e-10);
        assert!((comm_22 - 1.0_f64.cosh()).abs() < 1e-10);
        assert!((comm_12 - 1.0_f64.sinh()).abs() < 1e-10);
    }

    #[test]
    fn test_communicability_symmetric() {
        let graph = create_path_graph(4);
        let result = communicability(&graph).unwrap();

        // Communicability should be symmetric: C(u,v) = C(v,u)
        for (&u, inner) in &result {
            for (&v, &comm_uv) in inner {
                let comm_vu = result[&v][&u];
                assert!(
                    (comm_uv - comm_vu).abs() < 1e-10,
                    "Communicability not symmetric for ({}, {})",
                    u,
                    v
                );
            }
        }
    }

    #[test]
    fn test_communicability_betweenness_empty() {
        let graph = Graph::new_rust(false, false);
        let result = communicability_betweenness_centrality(&graph).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_communicability_betweenness_single_node() {
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(crate::graph::Node {
                id: 1,
                label: Some("1".to_string()),
                attributes: None,
            })
            .unwrap();

        let result = communicability_betweenness_centrality(&graph).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[&1], 0.0);
    }

    #[test]
    fn test_communicability_betweenness_star() {
        let graph = create_star_graph(4); // Star with 5 nodes (1 center + 4 leaves)
        let result = communicability_betweenness_centrality(&graph).unwrap();

        // Center node (1) should have highest betweenness
        let center_betweenness = result[&1];
        for i in 2..=5u64 {
            assert!(
                center_betweenness >= result[&i],
                "Center should have highest betweenness"
            );
        }

        // Betweenness should be between 0 and 1
        for &bc in result.values() {
            assert!(bc >= 0.0 && bc <= 1.0, "Betweenness should be in [0, 1]");
        }
    }

    #[test]
    fn test_communicability_betweenness_path() {
        let graph = create_path_graph(5);
        let result = communicability_betweenness_centrality(&graph).unwrap();

        // Middle node (3) should have highest betweenness in a path
        let middle_betweenness = result[&3];
        let end_betweenness_1 = result[&1];
        let end_betweenness_5 = result[&5];

        assert!(
            middle_betweenness > end_betweenness_1,
            "Middle should have higher betweenness than end"
        );
        assert!(
            middle_betweenness > end_betweenness_5,
            "Middle should have higher betweenness than end"
        );
    }

    #[test]
    fn test_communicability_directed_error() {
        let graph = Graph::new_rust(true, false);
        let result = communicability(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_communicability_betweenness_directed_error() {
        let graph = Graph::new_rust(true, false);
        let result = communicability_betweenness_centrality(&graph);
        assert!(result.is_err());
    }
}
