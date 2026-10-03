//! Dominating set algorithms for graphs
//!
//! A dominating set D of a graph G = (V, E) is a subset of V such that every
//! vertex not in D is adjacent to at least one member of D. This module provides:
//!
//! - `is_dominating_set`: Check if a set of nodes is a dominating set
//! - `dominating_set`: Find a dominating set (greedy algorithm)
//! - `min_weighted_dominating_set`: Find an approximate minimum weighted dominating set

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

/// Check if a set of nodes forms a dominating set.
///
/// A dominating set is a subset D of vertices such that every vertex not in D
/// is adjacent to at least one member of D.
///
/// # Arguments
/// * `graph` - The input graph
/// * `nodes` - List of node IDs to check
///
/// # Returns
/// True if the nodes form a dominating set, False otherwise
#[pyfunction]
pub fn is_dominating_set(graph: &Graph, nodes: Vec<u64>) -> PyResult<bool> {
    // Convert to hash set for O(1) lookups
    let dominating: FxHashSet<u64> = nodes.iter().copied().collect();

    // Check each node in the graph
    for node_id in graph.get_node_ids() {
        // If node is in the dominating set, it's covered
        if dominating.contains(&node_id) {
            continue;
        }

        // Otherwise, check if any neighbor is in the dominating set
        let neighbors = graph.neighbors(node_id)?;
        let mut is_dominated = false;
        for neighbor in neighbors {
            if dominating.contains(&neighbor) {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Find a dominating set for the graph.
///
/// This implements a greedy algorithm (Algorithm 7 from Esfahanian's
/// "Connectivity Algorithms" paper). It finds some dominating set,
/// not necessarily the smallest one.
///
/// # Algorithm
/// 1. Start with the smallest node ID (or specified start_with)
/// 2. Repeatedly add the smallest undominated node and mark its neighbors as dominated
/// 3. Repeat until all nodes are dominated
///
/// # Arguments
/// * `graph` - The input graph
/// * `start_with` - Optional starting node ID. If None, the smallest node ID is used.
///
/// # Returns
/// Set of node IDs forming a dominating set
///
/// # Raises
/// * `PyValueError` if start_with node doesn't exist in the graph
#[pyfunction]
#[pyo3(signature = (graph, start_with=None))]
pub fn dominating_set(graph: &Graph, start_with: Option<u64>) -> PyResult<Vec<u64>> {
    let all_nodes: FxHashSet<u64> = graph.get_node_ids().into_iter().collect();

    // Handle empty graph
    if all_nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Get starting node
    let start = match start_with {
        Some(node) => {
            if !graph.has_node_id(node) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node {} is not in the graph",
                    node
                )));
            }
            node
        }
        // Smallest ID, not hash order, so the result is the same on every run
        None => *all_nodes.iter().min().unwrap(),
    };

    let mut dom_set: FxHashSet<u64> = FxHashSet::default();
    dom_set.insert(start);

    // Track dominated nodes (nodes adjacent to someone in dom_set)
    let mut dominated: FxHashSet<u64> = graph.neighbors(start)?.into_iter().collect();

    // Track remaining undominated nodes not in dom_set (ordered, so the greedy
    // pass always takes the smallest undominated ID next)
    let mut remaining: std::collections::BTreeSet<u64> = all_nodes
        .iter()
        .filter(|n| !dominated.contains(n) && !dom_set.contains(n))
        .copied()
        .collect();

    while !remaining.is_empty() {
        // Pick the smallest undominated node
        let v = *remaining.iter().next().unwrap();
        remaining.remove(&v);

        // Get undominated neighbors
        let neighbors = graph.neighbors(v)?;
        let undominated_nbrs: Vec<u64> = neighbors
            .into_iter()
            .filter(|n| !dom_set.contains(n))
            .collect();

        // Add v to dominating set
        dom_set.insert(v);

        // Mark neighbors as dominated and remove from remaining
        for nbr in undominated_nbrs {
            dominated.insert(nbr);
            remaining.remove(&nbr);
        }
    }

    // Return sorted for deterministic output
    let mut result: Vec<u64> = dom_set.into_iter().collect();
    result.sort();
    Ok(result)
}

/// Find an approximate minimum weighted dominating set.
///
/// This uses a greedy algorithm that iteratively adds the most cost-effective
/// node (lowest weight / number of newly dominated nodes) to the dominating set.
///
/// The approximation guarantee is O(log(sum of weights)) * optimal.
///
/// # Arguments
/// * `graph` - The input graph
/// * `weights` - Optional dictionary mapping node ID to weight. If None or missing,
///               nodes have weight 1.
///
/// # Returns
/// Set of node IDs forming an approximate minimum weighted dominating set
#[pyfunction]
#[pyo3(signature = (graph, weights=None))]
pub fn min_weighted_dominating_set(
    graph: &Graph,
    weights: Option<std::collections::HashMap<u64, f64>>,
) -> PyResult<Vec<u64>> {
    let all_nodes: Vec<u64> = graph.get_node_ids();

    // Handle empty graph
    if all_nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Get weights for all nodes (default to 1.0)
    let node_weights: std::collections::HashMap<u64, f64> = match weights {
        Some(w) => {
            all_nodes
                .iter()
                .map(|&node| (node, *w.get(&node).unwrap_or(&1.0)))
                .collect()
        }
        None => all_nodes.iter().map(|&node| (node, 1.0)).collect(),
    };

    let mut dom_set: FxHashSet<u64> = FxHashSet::default();
    let mut uncovered: FxHashSet<u64> = all_nodes.iter().copied().collect();

    // Build closed neighborhoods (node + all neighbors)
    let mut neighborhoods: std::collections::HashMap<u64, FxHashSet<u64>> = all_nodes
        .iter()
        .map(|&node| {
            let mut neighborhood: FxHashSet<u64> = graph
                .neighbors(node)
                .unwrap_or_default()
                .into_iter()
                .collect();
            neighborhood.insert(node);
            (node, neighborhood)
        })
        .collect();

    while !uncovered.is_empty() {
        // Find the most cost-effective node to add
        // Cost = weight / |neighborhood ∩ uncovered|
        let best_node = neighborhoods
            .iter()
            .filter(|(_, nbhood)| !nbhood.is_disjoint(&uncovered))
            .min_by(|(node_a, nbhood_a), (node_b, nbhood_b)| {
                let count_a = nbhood_a.intersection(&uncovered).count();
                let count_b = nbhood_b.intersection(&uncovered).count();

                // Avoid division by zero
                let cost_a = if count_a > 0 {
                    node_weights[node_a] / count_a as f64
                } else {
                    f64::INFINITY
                };
                let cost_b = if count_b > 0 {
                    node_weights[node_b] / count_b as f64
                } else {
                    f64::INFINITY
                };

                // ties go to the smaller node ID (not hash order)
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(node_a.cmp(node_b))
            })
            .map(|(node, _)| *node);

        match best_node {
            Some(node) => {
                // Add node to dominating set
                dom_set.insert(node);

                // Remove covered nodes from uncovered
                if let Some(neighborhood) = neighborhoods.get(&node) {
                    for &covered in neighborhood {
                        uncovered.remove(&covered);
                    }
                }

                // Remove node from consideration
                neighborhoods.remove(&node);
            }
            None => {
                // No more nodes can cover anything - this shouldn't happen in a connected graph
                // but could happen if uncovered only has isolated nodes already in dom_set
                break;
            }
        }
    }

    // Return sorted for deterministic output
    let mut result: Vec<u64> = dom_set.into_iter().collect();
    result.sort();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_dominating_set_empty() {
        let graph = Graph::new_rust(false, false);
        assert!(is_dominating_set(&graph, vec![]).unwrap());
    }

    #[test]
    fn test_is_dominating_set_path() {
        let mut graph = Graph::new_rust(false, false);
        // Create path: 1-2-3-4 (node id 0 is reserved for auto-assignment)
        for i in 1..=4 {
            graph.add_test_node_with_id(i);
        }
        graph.add_test_edge(1, 2, None);
        graph.add_test_edge(2, 3, None);
        graph.add_test_edge(3, 4, None);

        // {2, 4} dominates all
        assert!(is_dominating_set(&graph, vec![2, 4]).unwrap());
        // {1, 3} dominates all
        assert!(is_dominating_set(&graph, vec![1, 3]).unwrap());
        // {2} doesn't dominate 4
        assert!(!is_dominating_set(&graph, vec![2]).unwrap());
    }

    #[test]
    fn test_dominating_set_complete() {
        let mut graph = Graph::new_rust(false, false);
        // Complete graph K4
        for i in 1..=4 {
            graph.add_test_node_with_id(i);
        }
        for i in 1..=4 {
            for j in (i + 1)..=4 {
                graph.add_test_edge(i, j, None);
            }
        }

        let dom = dominating_set(&graph, None).unwrap();
        // In complete graph, any single node dominates
        assert_eq!(dom.len(), 1);
        assert!(is_dominating_set(&graph, dom).unwrap());
    }

    #[test]
    fn test_dominating_set_path() {
        let mut graph = Graph::new_rust(false, false);
        // Path: 1-2-3-4
        for i in 1..=4 {
            graph.add_test_node_with_id(i);
        }
        graph.add_test_edge(1, 2, None);
        graph.add_test_edge(2, 3, None);
        graph.add_test_edge(3, 4, None);

        let dom = dominating_set(&graph, None).unwrap();
        assert!(is_dominating_set(&graph, dom).unwrap());
    }

    #[test]
    fn test_dominating_set_star() {
        let mut graph = Graph::new_rust(false, false);
        // Star with center 1
        graph.add_test_node_with_id(1);
        for i in 2..7 {
            graph.add_test_node_with_id(i);
            graph.add_test_edge(1, i, None);
        }

        let dom = dominating_set(&graph, Some(1)).unwrap();
        // Center alone dominates entire star
        assert_eq!(dom.len(), 1);
        assert!(dom.contains(&1));
    }
}
