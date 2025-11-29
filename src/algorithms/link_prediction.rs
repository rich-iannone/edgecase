//! Link prediction algorithms
//!
//! This module implements common link prediction methods for predicting
//! the likelihood of future connections between nodes in a graph.
//!
//! Implemented methods:
//! - Common neighbors
//! - Jaccard coefficient
//! - Adamic-Adar index
//! - Resource allocation index
//! - Preferential attachment

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

/// Get common neighbors of two nodes as a set.
///
/// Helper function used by multiple link prediction methods.
fn get_common_neighbors_set(graph: &Graph, u: u64, v: u64) -> FxHashSet<u64> {
    let neighbors_u: FxHashSet<u64> = graph
        .neighbors(u)
        .unwrap_or_default()
        .into_iter()
        .collect();
    let neighbors_v: FxHashSet<u64> = graph
        .neighbors(v)
        .unwrap_or_default()
        .into_iter()
        .collect();

    neighbors_u.intersection(&neighbors_v).copied().collect()
}

/// Get neighbors as a set for a node.
fn get_neighbors_set(graph: &Graph, node: u64) -> FxHashSet<u64> {
    graph
        .neighbors(node)
        .unwrap_or_default()
        .into_iter()
        .collect()
}

/// Compute the common neighbors count for pairs of nodes.
///
/// The common neighbors of two nodes is the number of nodes that are
/// neighbors of both. This is a simple link prediction heuristic based
/// on the intuition that two nodes are more likely to be connected if
/// they share many common neighbors.
///
/// # Arguments
/// * `graph` - The input graph (should be undirected for meaningful results)
/// * `ebunch` - List of (u, v) node pairs to compute scores for.
///              If None, computes for all non-adjacent node pairs.
///
/// # Returns
/// List of (u, v, score) tuples where score is the number of common neighbors.
///
/// # References
/// D. Liben-Nowell, J. Kleinberg. "The Link Prediction Problem for Social Networks"
/// CIKM, 2003.
#[pyfunction]
#[pyo3(signature = (graph, ebunch=None))]
pub fn common_neighbors(
    graph: &Graph,
    ebunch: Option<Vec<(u64, u64)>>,
) -> PyResult<Vec<(u64, u64, usize)>> {
    let pairs = get_node_pairs(graph, ebunch)?;
    let mut results = Vec::with_capacity(pairs.len());

    for (u, v) in pairs {
        let common = get_common_neighbors_set(graph, u, v);
        results.push((u, v, common.len()));
    }

    Ok(results)
}

/// Compute the Jaccard coefficient for pairs of nodes.
///
/// The Jaccard coefficient is the ratio of common neighbors to the union
/// of neighbors: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
///
/// Values range from 0 (no common neighbors) to 1 (identical neighborhoods).
///
/// # Arguments
/// * `graph` - The input graph (should be undirected for meaningful results)
/// * `ebunch` - List of (u, v) node pairs to compute scores for.
///              If None, computes for all non-adjacent node pairs.
///
/// # Returns
/// List of (u, v, score) tuples where score is the Jaccard coefficient.
///
/// # References
/// D. Liben-Nowell, J. Kleinberg. "The Link Prediction Problem for Social Networks"
/// CIKM, 2003.
#[pyfunction]
#[pyo3(signature = (graph, ebunch=None))]
pub fn jaccard_coefficient(
    graph: &Graph,
    ebunch: Option<Vec<(u64, u64)>>,
) -> PyResult<Vec<(u64, u64, f64)>> {
    let pairs = get_node_pairs(graph, ebunch)?;
    let mut results = Vec::with_capacity(pairs.len());

    for (u, v) in pairs {
        let neighbors_u = get_neighbors_set(graph, u);
        let neighbors_v = get_neighbors_set(graph, v);

        let intersection_size = neighbors_u.intersection(&neighbors_v).count();
        let union_size = neighbors_u.union(&neighbors_v).count();

        let score = if union_size == 0 {
            0.0
        } else {
            intersection_size as f64 / union_size as f64
        };

        results.push((u, v, score));
    }

    Ok(results)
}

/// Compute the Adamic-Adar index for pairs of nodes.
///
/// The Adamic-Adar index is defined as:
/// sum(1 / log(degree(w))) for w in common_neighbors(u, v)
///
/// It gives more weight to common neighbors with lower degrees, based on
/// the intuition that rare features are more informative than common ones.
///
/// # Arguments
/// * `graph` - The input graph (should be undirected for meaningful results)
/// * `ebunch` - List of (u, v) node pairs to compute scores for.
///              If None, computes for all non-adjacent node pairs.
///
/// # Returns
/// List of (u, v, score) tuples where score is the Adamic-Adar index.
///
/// # References
/// L. Adamic and E. Adar. "Friends and neighbors on the Web"
/// Social Networks, 2003.
#[pyfunction]
#[pyo3(signature = (graph, ebunch=None))]
pub fn adamic_adar_index(
    graph: &Graph,
    ebunch: Option<Vec<(u64, u64)>>,
) -> PyResult<Vec<(u64, u64, f64)>> {
    let pairs = get_node_pairs(graph, ebunch)?;
    let mut results = Vec::with_capacity(pairs.len());

    for (u, v) in pairs {
        let common = get_common_neighbors_set(graph, u, v);
        let mut score = 0.0;

        for w in common {
            let degree = get_neighbors_set(graph, w).len();
            if degree > 1 {
                score += 1.0 / (degree as f64).ln();
            }
            // If degree <= 1, log would be <= 0, which is undefined/problematic
            // We skip such neighbors (they don't contribute to the score)
        }

        results.push((u, v, score));
    }

    Ok(results)
}

/// Compute the resource allocation index for pairs of nodes.
///
/// The resource allocation index is defined as:
/// sum(1 / degree(w)) for w in common_neighbors(u, v)
///
/// It can be seen as a diffusion process where resources flow from u to v
/// through common neighbors. Each common neighbor w transmits 1/degree(w)
/// of the resources it receives.
///
/// # Arguments
/// * `graph` - The input graph (should be undirected for meaningful results)
/// * `ebunch` - List of (u, v) node pairs to compute scores for.
///              If None, computes for all non-adjacent node pairs.
///
/// # Returns
/// List of (u, v, score) tuples where score is the resource allocation index.
///
/// # References
/// T. Zhou, L. Lu, Y.-C. Zhang. "Network-based resource allocation"
/// Physical Review E, 2009.
#[pyfunction]
#[pyo3(signature = (graph, ebunch=None))]
pub fn resource_allocation_index(
    graph: &Graph,
    ebunch: Option<Vec<(u64, u64)>>,
) -> PyResult<Vec<(u64, u64, f64)>> {
    let pairs = get_node_pairs(graph, ebunch)?;
    let mut results = Vec::with_capacity(pairs.len());

    for (u, v) in pairs {
        let common = get_common_neighbors_set(graph, u, v);
        let mut score = 0.0;

        for w in common {
            let degree = get_neighbors_set(graph, w).len();
            if degree > 0 {
                score += 1.0 / degree as f64;
            }
        }

        results.push((u, v, score));
    }

    Ok(results)
}

/// Compute the preferential attachment score for pairs of nodes.
///
/// The preferential attachment score is simply the product of the degrees
/// of the two nodes: degree(u) * degree(v)
///
/// Based on the preferential attachment model where high-degree nodes
/// are more likely to attract new connections.
///
/// # Arguments
/// * `graph` - The input graph (should be undirected for meaningful results)
/// * `ebunch` - List of (u, v) node pairs to compute scores for.
///              If None, computes for all non-adjacent node pairs.
///
/// # Returns
/// List of (u, v, score) tuples where score is the preferential attachment value.
///
/// # References
/// A.-L. Barabasi and R. Albert. "Emergence of scaling in random networks"
/// Science, 1999.
#[pyfunction]
#[pyo3(signature = (graph, ebunch=None))]
pub fn preferential_attachment(
    graph: &Graph,
    ebunch: Option<Vec<(u64, u64)>>,
) -> PyResult<Vec<(u64, u64, usize)>> {
    let pairs = get_node_pairs(graph, ebunch)?;
    let mut results = Vec::with_capacity(pairs.len());

    for (u, v) in pairs {
        let degree_u = get_neighbors_set(graph, u).len();
        let degree_v = get_neighbors_set(graph, v).len();
        let score = degree_u * degree_v;
        results.push((u, v, score));
    }

    Ok(results)
}

/// Get node pairs to compute link prediction scores for.
///
/// If ebunch is provided, validates and returns those pairs.
/// Otherwise, returns all non-adjacent node pairs.
fn get_node_pairs(
    graph: &Graph,
    ebunch: Option<Vec<(u64, u64)>>,
) -> PyResult<Vec<(u64, u64)>> {
    match ebunch {
        Some(pairs) => {
            // Validate that all nodes exist
            let node_ids: FxHashSet<u64> = graph.get_node_ids().into_iter().collect();
            for (u, v) in &pairs {
                if !node_ids.contains(u) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Node {} does not exist in the graph",
                        u
                    )));
                }
                if !node_ids.contains(v) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Node {} does not exist in the graph",
                        v
                    )));
                }
            }
            Ok(pairs)
        }
        None => {
            // Generate all non-adjacent pairs
            let node_ids: Vec<u64> = graph.get_node_ids();
            let n = node_ids.len();
            let mut pairs = Vec::new();

            // Build adjacency set for quick lookup
            let mut adj: FxHashSet<(u64, u64)> = FxHashSet::default();
            for edge in graph.get_edges_internal() {
                adj.insert((edge.source, edge.target));
                if !graph.is_directed() {
                    adj.insert((edge.target, edge.source));
                }
            }

            for i in 0..n {
                for j in (i + 1)..n {
                    let u = node_ids[i];
                    let v = node_ids[j];
                    // Only include non-adjacent pairs
                    if !adj.contains(&(u, v)) && !adj.contains(&(v, u)) {
                        pairs.push((u, v));
                    }
                }
            }

            Ok(pairs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        // Create a simple test graph:
        //   0 -- 1 -- 2
        //   |    |
        //   3 -- 4
        let mut g = Graph::new(false, false); // undirected, unweighted

        // Add nodes
        for i in 0..5 {
            g.add_node_internal(i, None, None).unwrap();
        }

        // Add edges
        g.add_edge_internal(0, 1, None, None).unwrap();
        g.add_edge_internal(1, 2, None, None).unwrap();
        g.add_edge_internal(0, 3, None, None).unwrap();
        g.add_edge_internal(1, 4, None, None).unwrap();
        g.add_edge_internal(3, 4, None, None).unwrap();

        g
    }

    #[test]
    fn test_common_neighbors() {
        let g = create_test_graph();
        // Nodes 0 and 4 have common neighbor 1 and 3
        let result = common_neighbors(&g, Some(vec![(0, 4)])).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].2, 2); // 2 common neighbors
    }

    #[test]
    fn test_jaccard_coefficient() {
        let g = create_test_graph();
        let result = jaccard_coefficient(&g, Some(vec![(0, 4)])).unwrap();
        assert_eq!(result.len(), 1);
        // N(0) = {1, 3}, N(4) = {1, 3}
        // Intersection = {1, 3}, Union = {1, 3}
        // Jaccard = 2/2 = 1.0
        assert!((result[0].2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_preferential_attachment() {
        let g = create_test_graph();
        let result = preferential_attachment(&g, Some(vec![(0, 2)])).unwrap();
        assert_eq!(result.len(), 1);
        // degree(0) = 2, degree(2) = 1
        assert_eq!(result[0].2, 2);
    }

    #[test]
    fn test_resource_allocation() {
        let g = create_test_graph();
        let result = resource_allocation_index(&g, Some(vec![(0, 2)])).unwrap();
        assert_eq!(result.len(), 1);
        // Common neighbor is 1, degree(1) = 3
        // RA = 1/3
        assert!((result[0].2 - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_adamic_adar() {
        let g = create_test_graph();
        let result = adamic_adar_index(&g, Some(vec![(0, 2)])).unwrap();
        assert_eq!(result.len(), 1);
        // Common neighbor is 1, degree(1) = 3
        // AA = 1/ln(3)
        assert!((result[0].2 - 1.0 / 3.0_f64.ln()).abs() < 1e-10);
    }
}
