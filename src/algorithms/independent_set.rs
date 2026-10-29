//! Independent set algorithms for graphs
//!
//! An independent set (or stable set) is a set of vertices in a graph where no two
//! vertices are adjacent. This module provides algorithms for finding:
//!
//! - Maximal independent sets: sets that cannot be extended by adding more vertices
//! - Maximum independent sets: largest possible independent sets (NP-hard, uses heuristics)

use crate::graph::Graph;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};

/// Neighbors of every node with edge direction ignored (in- and out-neighbors), deduplicated
/// and without self-loops.
///
/// Independence is a property of the underlying undirected graph: an edge `A -> B` means `A`
/// and `B` can't both be chosen, which is how `is_independent_set()` checks it. Using
/// `Graph::neighbors()` directly would only rule out out-neighbors on directed graphs.
fn undirected_neighbors(graph: &Graph) -> FxHashMap<u64, Vec<u64>> {
    let mut adj: FxHashMap<u64, Vec<u64>> = graph
        .get_node_ids()
        .into_iter()
        .map(|id| (id, Vec::new()))
        .collect();
    for edge in graph.get_edges_internal() {
        if edge.source != edge.target {
            adj.entry(edge.source).or_default().push(edge.target);
            adj.entry(edge.target).or_default().push(edge.source);
        }
    }
    for neighbors in adj.values_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adj
}

/// Find a maximal independent set in the graph.
///
/// A maximal independent set is an independent set that cannot be extended by adding
/// any other vertex (all non-members are adjacent to at least one member).
///
/// This uses a randomized greedy algorithm:
/// 1. Start with an empty set
/// 2. Randomly pick a vertex not adjacent to any in the set
/// 3. Add it to the set
/// 4. Repeat until no more vertices can be added
///
/// # Arguments
/// * `graph` - The input graph
/// * `nodes` - Optional starting set of nodes (must be independent). If None, starts empty.
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// Set of node IDs forming a maximal independent set
///
/// # Raises
/// * `PyValueError` if the starting nodes don't form an independent set
#[pyfunction]
#[pyo3(signature = (graph, nodes=None, seed=None))]
pub fn maximal_independent_set(
    graph: &Graph,
    nodes: Option<Vec<u64>>,
    seed: Option<u64>,
) -> PyResult<Vec<u64>> {
    let all_nodes: Vec<u64> = graph.get_node_ids();

    if all_nodes.is_empty() {
        return Ok(Vec::new());
    }

    let adj = undirected_neighbors(graph);

    // Initialize independent set
    let mut independent_set: FxHashSet<u64> = FxHashSet::default();
    let mut excluded: FxHashSet<u64> = FxHashSet::default();

    // If starting nodes provided, validate and use them
    if let Some(start_nodes) = nodes {
        // Validate starting nodes exist
        for &node in &start_nodes {
            if !graph.has_node_id(node) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Node {} does not exist in graph",
                    node
                )));
            }
        }

        // Validate starting nodes form an independent set
        for &node in &start_nodes {
            independent_set.insert(node);
        }

        // Check for adjacency between starting nodes
        for &node in &start_nodes {
            for neighbor in &adj[&node] {
                if independent_set.contains(neighbor) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Starting nodes do not form an independent set (adjacent nodes found)",
                    ));
                }
            }
        }

        // Mark starting nodes and their neighbors as excluded from further consideration
        for &node in &start_nodes {
            excluded.insert(node);
            excluded.extend(adj[&node].iter().copied());
        }
    }

    // Get available nodes (not in excluded set)
    let mut available: Vec<u64> = all_nodes
        .iter()
        .copied()
        .filter(|n| !excluded.contains(n))
        .collect();

    // Create RNG for randomization
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    // Shuffle available nodes for randomness
    available.shuffle(&mut rng);

    // Greedy algorithm: add nodes that don't conflict
    for node in available {
        if excluded.contains(&node) {
            continue;
        }

        // Add node to independent set
        independent_set.insert(node);
        excluded.insert(node);

        // Exclude all neighbors, whichever way the edge points
        excluded.extend(adj[&node].iter().copied());
    }

    // Return sorted for deterministic output order
    let mut result: Vec<u64> = independent_set.into_iter().collect();
    result.sort();
    Ok(result)
}

/// Check if a set of nodes forms an independent set.
///
/// An independent set is a set of vertices where no two vertices are adjacent.
///
/// # Arguments
/// * `graph` - The input graph
/// * `nodes` - List of node IDs to check
///
/// # Returns
/// True if the nodes form an independent set, False otherwise
#[pyfunction]
pub fn is_independent_set(graph: &Graph, nodes: Vec<u64>) -> PyResult<bool> {
    // Empty set is trivially independent
    if nodes.is_empty() {
        return Ok(true);
    }

    let node_set: FxHashSet<u64> = nodes.iter().copied().collect();

    // Check each node
    for &node in &nodes {
        // Verify node exists
        if !graph.has_node_id(node) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Node {} does not exist in graph",
                node
            )));
        }

        // Check if any neighbor is in the set
        let neighbors = graph.neighbors(node)?;
        for neighbor in neighbors {
            if node_set.contains(&neighbor) && neighbor != node {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Find a maximum independent set using a greedy heuristic.
///
/// The maximum independent set problem is NP-hard, so this uses a greedy
/// approximation that prioritizes nodes with minimum degree (fewer constraints).
///
/// Algorithm:
/// 1. Calculate degree for all nodes
/// 2. Pick the node with minimum degree
/// 3. Add it to the set and remove it and its neighbors from consideration
/// 4. Repeat until no nodes remain
///
/// This provides a good heuristic but is not guaranteed to find the true maximum.
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// Set of node IDs forming an approximate maximum independent set
#[pyfunction]
pub fn maximum_independent_set(graph: &Graph) -> PyResult<Vec<u64>> {
    let all_nodes: Vec<u64> = graph.get_node_ids();

    if all_nodes.is_empty() {
        return Ok(Vec::new());
    }

    let adj = undirected_neighbors(graph);

    // Track which nodes are still available
    let mut available: FxHashSet<u64> = all_nodes.iter().copied().collect();
    let mut independent_set: Vec<u64> = Vec::new();

    while !available.is_empty() {
        // Find node with minimum degree among available nodes
        let mut min_deg = usize::MAX;
        let mut min_node = 0u64;

        for &node in &all_nodes {
            if available.contains(&node) {
                // Recalculate effective degree (neighbors still available)
                let effective_deg = adj[&node].iter().filter(|n| available.contains(n)).count();

                if effective_deg < min_deg || (effective_deg == min_deg && node < min_node) {
                    min_deg = effective_deg;
                    min_node = node;
                }
            }
        }

        // Add minimum degree node to independent set
        independent_set.push(min_node);
        available.remove(&min_node);

        // Remove all neighbors from available set, whichever way the edge points
        for neighbor in &adj[&min_node] {
            available.remove(neighbor);
        }
    }

    // Sort for deterministic output
    independent_set.sort();
    Ok(independent_set)
}

/// Find multiple maximal independent sets using randomized search.
///
/// Runs the maximal independent set algorithm multiple times with different
/// random seeds to find diverse solutions.
///
/// # Arguments
/// * `graph` - The input graph
/// * `count` - Number of maximal independent sets to find
/// * `seed` - Optional base seed for reproducibility
///
/// # Returns
/// Vector of independent sets (each is a vector of node IDs)
#[pyfunction]
#[pyo3(signature = (graph, count=10, seed=None))]
pub fn find_maximal_independent_sets(
    graph: &Graph,
    count: usize,
    seed: Option<u64>,
) -> PyResult<Vec<Vec<u64>>> {
    let mut results: Vec<Vec<u64>> = Vec::with_capacity(count);
    let mut seen: FxHashSet<Vec<u64>> = FxHashSet::default();

    let base_seed = seed.unwrap_or(0);

    for i in 0..count.max(count * 3) {
        // Try more iterations to find unique sets
        if results.len() >= count {
            break;
        }

        let current_seed = if seed.is_some() {
            Some(base_seed.wrapping_add(i as u64))
        } else {
            None
        };

        let mis = maximal_independent_set(graph, None, current_seed)?;

        if !seen.contains(&mis) {
            seen.insert(mis.clone());
            results.push(mis);
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        // Create a simple graph: 1-2-3-4
        let mut graph = Graph::new_rust(false, false);
        for i in 1..=4 {
            graph.add_test_node_with_id(i);
        }
        graph.add_test_edge(1, 2, None);
        graph.add_test_edge(2, 3, None);
        graph.add_test_edge(3, 4, None);
        graph
    }

    #[test]
    fn test_is_independent_set_empty() {
        let graph = create_test_graph();
        assert!(is_independent_set(&graph, vec![]).unwrap());
    }

    #[test]
    fn test_is_independent_set_single() {
        let graph = create_test_graph();
        assert!(is_independent_set(&graph, vec![1]).unwrap());
    }

    #[test]
    fn test_is_independent_set_valid() {
        let graph = create_test_graph();
        // 1 and 3 are not adjacent
        assert!(is_independent_set(&graph, vec![1, 3]).unwrap());
        // 1 and 4 are not adjacent
        assert!(is_independent_set(&graph, vec![1, 4]).unwrap());
        // 2 and 4 are not adjacent
        assert!(is_independent_set(&graph, vec![2, 4]).unwrap());
    }

    #[test]
    fn test_is_independent_set_invalid() {
        let graph = create_test_graph();
        // 1 and 2 are adjacent
        assert!(!is_independent_set(&graph, vec![1, 2]).unwrap());
        // 2 and 3 are adjacent
        assert!(!is_independent_set(&graph, vec![2, 3]).unwrap());
    }

    #[test]
    fn test_maximal_independent_set() {
        let graph = create_test_graph();
        let mis = maximal_independent_set(&graph, None, Some(42)).unwrap();

        // Should be an independent set
        assert!(is_independent_set(&graph, mis.clone()).unwrap());

        // Should be maximal (can't add any more nodes)
        // For path 1-2-3-4, maximal sets are {1,3}, {1,4}, {2,4}
        assert!(mis.len() == 2);
    }

    #[test]
    fn test_maximum_independent_set() {
        let graph = create_test_graph();
        let mis = maximum_independent_set(&graph).unwrap();

        // Should be an independent set
        assert!(is_independent_set(&graph, mis.clone()).unwrap());

        // For path 1-2-3-4, maximum independent set has size 2
        assert!(mis.len() == 2);
    }

    #[test]
    fn test_maximal_with_seed_nodes() {
        let graph = create_test_graph();

        // Start with node 1
        let mis = maximal_independent_set(&graph, Some(vec![1]), Some(42)).unwrap();

        assert!(mis.contains(&1));
        assert!(is_independent_set(&graph, mis).unwrap());
    }

    #[test]
    fn test_maximal_with_invalid_seed() {
        let graph = create_test_graph();

        // Starting with adjacent nodes should fail
        let result = maximal_independent_set(&graph, Some(vec![1, 2]), Some(42));
        assert!(result.is_err());
    }
}
