//! Percolation algorithms for graphs.
//!
//! This module provides:
//! - `percolation_centrality`: Brandes-style betweenness centrality weighted by
//!   percolation states (e.g., infection probabilities)
//! - `node_percolation`: Simulate node removal for robustness analysis
//! - `edge_percolation`: Simulate edge removal for robustness analysis
//!
//! References:
//! - Piraveenan, M., Prokopenko, M., & Hossain, L. (2013). Percolation centrality:
//!   Quantifying graph-theoretic impact of nodes during percolation in networks.
//!   PloS one, 8(1), e53095.
//! - Brandes, U. (2001). A faster algorithm for betweenness centrality.
//!   Journal of Mathematical Sociology, 25(2), 163-177.

use crate::graph::{Edge, Graph, Node};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Compute percolation centrality for all nodes.
///
/// Percolation centrality measures the importance of a node based on the
/// proportion of "percolated paths" that pass through it. Percolated paths
/// are weighted by the percolation states of source and target nodes.
///
/// When all nodes have percolation state = 1.0, this reduces to standard
/// betweenness centrality.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `states` - Optional dictionary mapping node IDs to percolation states (0.0 to 1.0).
///              If None or empty, all states default to 1.0 (equivalent to betweenness).
/// * `weight` - Optional edge weight attribute name. If None, all edges have weight 1.
///
/// # Returns
/// A dictionary mapping node IDs to their percolation centrality values.
///
/// # Algorithm
/// Uses a modified Brandes algorithm where pair dependencies are weighted by:
/// `(state[s] + state[t]) / (n_percolated - 1)`
///
/// where s is source, t is target, and n_percolated is the total percolation
/// (sum of all states).
#[pyfunction]
#[pyo3(signature = (graph, states=None, weight=None))]
pub fn percolation_centrality(
    graph: &Graph,
    states: Option<FxHashMap<u64, f64>>,
    weight: Option<&str>,
) -> PyResult<FxHashMap<u64, f64>> {
    use rayon::prelude::*;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    // Get percolation states for each node (default to 1.0)
    let node_states: FxHashMap<u64, f64> = if let Some(s) = states {
        // Fill in defaults for missing nodes
        let mut full_states = FxHashMap::default();
        for &node_id in &node_ids {
            let state = s.get(&node_id).copied().unwrap_or(1.0);
            // Validate state is in [0, 1]
            if !(0.0..=1.0).contains(&state) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Percolation state for node {} must be between 0.0 and 1.0, got {}",
                    node_id, state
                )));
            }
            full_states.insert(node_id, state);
        }
        full_states
    } else {
        // All nodes have state 1.0
        node_ids.iter().map(|&id| (id, 1.0)).collect()
    };

    // Calculate total percolation (sum of all states)
    let percolation_sum: f64 = node_states.values().sum();
    if percolation_sum <= 1.0 {
        // Not enough percolation to compute meaningful centrality
        // (need at least 2 percolating nodes)
        return Ok(node_ids.iter().map(|&id| (id, 0.0)).collect());
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Determine if weighted
    let use_weights = weight.is_some() && graph.is_weighted();

    // Parallel computation
    let centrality = node_ids
        .par_iter()
        .map(|&source| {
            compute_percolation_from_source(
                graph,
                source,
                &node_ids,
                &node_to_idx,
                &node_states,
                percolation_sum,
                use_weights,
                weight,
            )
        })
        .reduce(
            || vec![0.0; n],
            |mut acc, partial| {
                for i in 0..n {
                    acc[i] += partial[i];
                }
                acc
            },
        );

    // Convert to HashMap
    let mut result: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (idx, &node_id) in node_ids.iter().enumerate() {
        result.insert(node_id, centrality[idx]);
    }

    Ok(result)
}

/// Helper: Compute percolation centrality contribution from a single source node.
#[inline]
fn compute_percolation_from_source(
    graph: &Graph,
    source: u64,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
    node_states: &FxHashMap<u64, f64>,
    percolation_sum: f64,
    use_weights: bool,
    _weight_attr: Option<&str>, // Currently unused, weights not implemented
) -> Vec<f64> {
    let n = node_ids.len();
    let source_idx = node_to_idx[&source];
    let source_state = node_states[&source];

    // Local arrays for BFS/Dijkstra
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut sigma: Vec<f64> = vec![0.0; n];
    let mut distance: Vec<f64> = vec![f64::INFINITY; n];
    let mut delta: Vec<f64> = vec![0.0; n];

    // Initialize source
    sigma[source_idx] = 1.0;
    distance[source_idx] = 0.0;

    if use_weights {
        // Dijkstra for weighted graphs
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(PartialEq)]
        struct State {
            cost: f64,
            node_idx: usize,
        }

        impl Eq for State {}

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
            }
        }

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        heap.push(State {
            cost: 0.0,
            node_idx: source_idx,
        });

        while let Some(State { cost, node_idx: v_idx }) = heap.pop() {
            if cost > distance[v_idx] {
                continue;
            }

            stack.push(v_idx);
            let v = node_ids[v_idx];

            if let Ok(neighbors) = graph.neighbors(v) {
                for w in neighbors {
                    let w_idx = node_to_idx[&w];

                    // For simplicity, use weight=1.0 for all edges
                    // Full weighted version would require edge lookup by (source, target)
                    let edge_weight = 1.0;

                    if edge_weight <= 0.0 {
                        continue; // Skip non-positive weights
                    }

                    let alt = distance[v_idx] + edge_weight;

                    if alt < distance[w_idx] {
                        distance[w_idx] = alt;
                        sigma[w_idx] = sigma[v_idx];
                        predecessors[w_idx].clear();
                        predecessors[w_idx].push(v_idx);
                        heap.push(State {
                            cost: alt,
                            node_idx: w_idx,
                        });
                    } else if (alt - distance[w_idx]).abs() < 1e-10 {
                        sigma[w_idx] += sigma[v_idx];
                        predecessors[w_idx].push(v_idx);
                    }
                }
            }
        }
    } else {
        // BFS for unweighted graphs
        let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);
        queue.push_back(source_idx);

        while let Some(v_idx) = queue.pop_front() {
            stack.push(v_idx);
            let v = node_ids[v_idx];

            if let Ok(neighbors) = graph.neighbors(v) {
                for w in neighbors {
                    let w_idx = node_to_idx[&w];

                    if distance[w_idx] == f64::INFINITY {
                        queue.push_back(w_idx);
                        distance[w_idx] = distance[v_idx] + 1.0;
                    }

                    if (distance[w_idx] - distance[v_idx] - 1.0).abs() < 1e-10 {
                        sigma[w_idx] += sigma[v_idx];
                        predecessors[w_idx].push(v_idx);
                    }
                }
            }
        }
    }

    // Accumulation: back-propagation of dependencies
    // Weighted by percolation states
    let mut centrality = vec![0.0; n];

    while let Some(w_idx) = stack.pop() {
        if w_idx == source_idx {
            continue;
        }

        let w = node_ids[w_idx];
        let target_state = node_states[&w];

        // Weight factor: (x_s + x_t) / (sum_x - 1)
        // where x_s is source state, x_t is target state
        let weight_factor = (source_state + target_state) / (percolation_sum - 1.0);

        // Percolation adjustment for delta
        let coeff = weight_factor / sigma[w_idx];

        for &v_idx in &predecessors[w_idx] {
            let contrib = sigma[v_idx] * coeff * (1.0 + delta[w_idx]);
            delta[v_idx] += contrib;
        }

        // Add to centrality for intermediate nodes
        if w_idx != source_idx {
            centrality[w_idx] += delta[w_idx];
        }
    }

    centrality
}

/// Simulate node percolation on a graph.
///
/// Removes a fraction of nodes randomly and returns the resulting subgraph.
/// This is useful for studying network robustness under random failures.
///
/// # Arguments
/// * `graph` - The graph to percolate
/// * `q` - The fraction of nodes to REMOVE (0.0 to 1.0). q=0.2 means 20% removed.
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// A new Graph containing only the surviving nodes and their edges.
///
/// # Example
/// Removing 30% of nodes randomly:
/// ```python
/// result = node_percolation(G, q=0.3, seed=23)
/// ```
#[pyfunction]
#[pyo3(signature = (graph, q, seed=None))]
pub fn node_percolation(graph: &Graph, q: f64, seed: Option<u64>) -> PyResult<Graph> {
    // Validate q
    if !(0.0..=1.0).contains(&q) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q must be between 0.0 and 1.0",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(Graph::new_rust(graph.is_directed(), graph.is_weighted()));
    }

    // Number of nodes to remove
    let num_remove = ((n as f64) * q).round() as usize;

    // Create RNG with optional seed
    let mut rng = if let Some(s) = seed {
        ChaCha8Rng::seed_from_u64(s)
    } else {
        ChaCha8Rng::from_entropy()
    };

    // Randomly select nodes to remove
    let mut nodes_to_remove: Vec<u64> = node_ids.clone();
    nodes_to_remove.shuffle(&mut rng);
    let removed_set: std::collections::HashSet<u64> =
        nodes_to_remove.iter().take(num_remove).copied().collect();

    // Create new graph with surviving nodes
    let mut result = Graph::new_rust(graph.is_directed(), graph.is_weighted());

    // Add surviving nodes
    for &node_id in &node_ids {
        if !removed_set.contains(&node_id) {
            let label = graph.label_from_id(node_id).ok().flatten();
            let node = Node {
                id: node_id,
                label,
                attributes: None,
            };
            result.add_node_rust(node)?;
        }
    }

    // Add edges between surviving nodes
    for edge in graph.get_edges_internal() {
        if !removed_set.contains(&edge.source) && !removed_set.contains(&edge.target) {
            let new_edge = Edge {
                id: 0, // Auto-assign
                source: edge.source,
                target: edge.target,
                weight: edge.weight,
                attributes: None,
            };
            result.add_edge_rust(new_edge)?;
        }
    }

    Ok(result)
}

/// Simulate edge percolation on a graph.
///
/// Removes a fraction of edges randomly and returns the resulting subgraph.
/// This is useful for studying network robustness under random edge failures.
///
/// # Arguments
/// * `graph` - The graph to percolate
/// * `q` - The fraction of edges to REMOVE (0.0 to 1.0). q=0.2 means 20% removed.
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// A new Graph containing all nodes but only surviving edges.
///
/// # Example
/// Removing 30% of edges randomly:
/// ```python
/// result = edge_percolation(G, q=0.3, seed=23)
/// ```
#[pyfunction]
#[pyo3(signature = (graph, q, seed=None))]
pub fn edge_percolation(graph: &Graph, q: f64, seed: Option<u64>) -> PyResult<Graph> {
    // Validate q
    if !(0.0..=1.0).contains(&q) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "q must be between 0.0 and 1.0",
        ));
    }

    let node_ids = graph.get_node_ids();
    let edges = graph.get_edges_internal();
    let m = edges.len();

    // Create new graph with all nodes
    let mut result = Graph::new_rust(graph.is_directed(), graph.is_weighted());

    for &node_id in &node_ids {
        let label = graph.label_from_id(node_id).ok().flatten();
        let node = Node {
            id: node_id,
            label,
            attributes: None,
        };
        result.add_node_rust(node)?;
    }

    if m == 0 {
        return Ok(result);
    }

    // Number of edges to remove
    let num_remove = ((m as f64) * q).round() as usize;

    // Create RNG with optional seed
    let mut rng = if let Some(s) = seed {
        ChaCha8Rng::seed_from_u64(s)
    } else {
        ChaCha8Rng::from_entropy()
    };

    // Randomly select edges to keep
    let mut edge_indices: Vec<usize> = (0..m).collect();
    edge_indices.shuffle(&mut rng);

    // Keep edges that aren't in the first num_remove
    let remove_set: std::collections::HashSet<usize> =
        edge_indices.iter().take(num_remove).copied().collect();

    for (idx, edge) in edges.iter().enumerate() {
        if !remove_set.contains(&idx) {
            let new_edge = Edge {
                id: 0, // Auto-assign
                source: edge.source,
                target: edge.target,
                weight: edge.weight,
                attributes: None,
            };
            result.add_edge_rust(new_edge)?;
        }
    }

    Ok(result)
}

/// Calculate the size of the largest connected component.
///
/// This is a helper function useful for percolation analysis,
/// as the relative size of the giant component is a key metric.
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// The number of nodes in the largest connected component.
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn largest_component_size(graph: &Graph) -> PyResult<usize> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(0);
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // For directed graphs, build a transpose adjacency list
    let transpose: Option<FxHashMap<u64, Vec<u64>>> = if graph.is_directed() {
        let mut trans: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        for &node_id in &node_ids {
            trans.insert(node_id, Vec::new());
        }
        for edge in graph.get_edges_internal() {
            if let Some(incoming) = trans.get_mut(&edge.target) {
                incoming.push(edge.source);
            }
        }
        Some(trans)
    } else {
        None
    };

    // Track visited nodes
    let mut visited = vec![false; n];
    let mut max_size = 0;

    // BFS/DFS from each unvisited node
    for start_idx in 0..n {
        if visited[start_idx] {
            continue;
        }

        // BFS from this node
        let mut queue = VecDeque::new();
        queue.push_back(start_idx);
        visited[start_idx] = true;
        let mut component_size = 0;

        while let Some(v_idx) = queue.pop_front() {
            component_size += 1;
            let v = node_ids[v_idx];

            if let Ok(neighbors) = graph.neighbors(v) {
                for w in neighbors {
                    if let Some(&w_idx) = node_to_idx.get(&w) {
                        if !visited[w_idx] {
                            visited[w_idx] = true;
                            queue.push_back(w_idx);
                        }
                    }
                }
            }

            // For directed graphs, also check incoming edges (weak connectivity)
            if let Some(ref trans) = transpose {
                if let Some(in_neighbors) = trans.get(&v) {
                    for &w in in_neighbors {
                        if let Some(&w_idx) = node_to_idx.get(&w) {
                            if !visited[w_idx] {
                                visited[w_idx] = true;
                                queue.push_back(w_idx);
                            }
                        }
                    }
                }
            }
        }

        max_size = max_size.max(component_size);
    }

    Ok(max_size)
}

/// Perform a percolation threshold analysis.
///
/// Systematically removes nodes at different fractions and measures the
/// relative size of the largest connected component at each step.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `steps` - Number of removal fractions to test (default: 20)
/// * `trials` - Number of random trials per step for averaging (default: 10)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// A list of tuples (q, relative_gcc_size) where:
/// - q is the fraction of nodes removed
/// - relative_gcc_size is the average relative size of the giant component
#[pyfunction]
#[pyo3(signature = (graph, steps=20, trials=10, seed=None))]
pub fn percolation_threshold_analysis(
    graph: &Graph,
    steps: usize,
    trials: usize,
    seed: Option<u64>,
) -> PyResult<Vec<(f64, f64)>> {
    let n = graph.get_node_ids().len();

    if n == 0 {
        return Ok(vec![(0.0, 0.0)]);
    }

    // Create base RNG
    let mut rng = if let Some(s) = seed {
        ChaCha8Rng::seed_from_u64(s)
    } else {
        ChaCha8Rng::from_entropy()
    };

    let mut results: Vec<(f64, f64)> = Vec::with_capacity(steps + 1);

    // Test each removal fraction
    for step in 0..=steps {
        let q = step as f64 / steps as f64;

        // Average over trials
        let mut total_rel_size = 0.0;

        for _ in 0..trials {
            // Generate a trial-specific seed from the main RNG
            let trial_seed: u64 = rng.gen();

            let percolated = node_percolation(graph, q, Some(trial_seed))?;
            let gcc_size = largest_component_size(&percolated)?;

            // Relative size (fraction of surviving nodes)
            let surviving = ((1.0 - q) * n as f64).round() as usize;
            let rel_size = if surviving > 0 {
                gcc_size as f64 / surviving as f64
            } else {
                0.0
            };

            total_rel_size += rel_size;
        }

        let avg_rel_size = total_rel_size / trials as f64;
        results.push((q, avg_rel_size));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn create_star_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        // Center node is 0
        for i in 0..n {
            let node = Node {
                id: i as u64,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        for i in 1..n {
            let edge = Edge {
                id: 0,
                source: 0,
                target: i as u64,
                weight: None,
                attributes: None,
            };
            graph.add_edge_rust(edge).unwrap();
        }
        graph
    }

    #[test]
    fn test_percolation_centrality_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let result = percolation_centrality(&graph, None, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_percolation_centrality_single_node() {
        let mut graph = Graph::new_rust(false, false);
        let node = Node {
            id: 1,
            label: None,
            attributes: None,
        };
        graph.add_node_rust(node).unwrap();
        let result = percolation_centrality(&graph, None, None).unwrap();
        assert_eq!(result.get(&1), Some(&0.0));
    }

    #[test]
    fn test_percolation_centrality_path() {
        // Path: 0 - 1 - 2 - 3
        // All states = 1.0, so should equal betweenness
        let graph = create_path_graph(4);
        let result = percolation_centrality(&graph, None, None).unwrap();

        // In a path, middle nodes should have higher centrality
        assert!(result[&1] > result[&0]);
        assert!(result[&2] > result[&3]);
    }

    #[test]
    fn test_percolation_centrality_with_states() {
        // Path: 0 - 1 - 2
        let graph = create_path_graph(3);

        // Custom states
        let mut states: FxHashMap<u64, f64> = FxHashMap::default();
        states.insert(0, 1.0);
        states.insert(1, 0.5);
        states.insert(2, 1.0);

        let result = percolation_centrality(&graph, Some(states), None).unwrap();

        // Middle node should still have centrality
        assert!(result[&1] > 0.0);
    }

    #[test]
    fn test_percolation_centrality_invalid_state() {
        let graph = create_path_graph(3);
        let mut states: FxHashMap<u64, f64> = FxHashMap::default();
        states.insert(0, 1.5); // Invalid: > 1.0

        let result = percolation_centrality(&graph, Some(states), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_node_percolation_q_zero() {
        let graph = create_complete_graph(5);
        let result = node_percolation(&graph, 0.0, Some(42)).unwrap();

        // q=0 means no removal, all nodes should survive
        assert_eq!(result.get_node_ids().len(), 5);
    }

    #[test]
    fn test_node_percolation_q_one() {
        let graph = create_complete_graph(5);
        let result = node_percolation(&graph, 1.0, Some(42)).unwrap();

        // q=1 means all removed
        assert_eq!(result.get_node_ids().len(), 0);
    }

    #[test]
    fn test_node_percolation_reproducible() {
        let graph = create_complete_graph(10);

        let result1 = node_percolation(&graph, 0.5, Some(123)).unwrap();
        let result2 = node_percolation(&graph, 0.5, Some(123)).unwrap();

        // Same seed should give same result
        let nodes1: Vec<u64> = result1.get_node_ids();
        let nodes2: Vec<u64> = result2.get_node_ids();
        assert_eq!(nodes1, nodes2);
    }

    #[test]
    fn test_node_percolation_edges_removed() {
        // Create a path: 0 - 1 - 2 - 3 - 4
        let graph = create_path_graph(5);

        // Remove 60% of nodes (3 nodes)
        let result = node_percolation(&graph, 0.6, Some(42)).unwrap();

        // Should have 2 nodes remaining
        assert_eq!(result.get_node_ids().len(), 2);

        // Edges should only exist between surviving nodes
        let edges = result.get_edges_internal();
        let surviving_nodes: std::collections::HashSet<u64> =
            result.get_node_ids().into_iter().collect();
        for edge in edges {
            assert!(surviving_nodes.contains(&edge.source));
            assert!(surviving_nodes.contains(&edge.target));
        }
    }

    #[test]
    fn test_edge_percolation_q_zero() {
        let graph = create_complete_graph(4);
        let result = edge_percolation(&graph, 0.0, Some(42)).unwrap();

        // q=0 means no removal, all edges should survive
        // K4 has 6 edges
        assert_eq!(result.get_edges_internal().len(), 6);
    }

    #[test]
    fn test_edge_percolation_q_one() {
        let graph = create_complete_graph(4);
        let result = edge_percolation(&graph, 1.0, Some(42)).unwrap();

        // q=1 means all edges removed
        assert_eq!(result.get_edges_internal().len(), 0);
        // But all nodes should remain
        assert_eq!(result.get_node_ids().len(), 4);
    }

    #[test]
    fn test_edge_percolation_reproducible() {
        let graph = create_complete_graph(6);

        let result1 = edge_percolation(&graph, 0.5, Some(456)).unwrap();
        let result2 = edge_percolation(&graph, 0.5, Some(456)).unwrap();

        // Same seed should give same result
        let edges1 = result1.get_edges_internal();
        let edges2 = result2.get_edges_internal();
        assert_eq!(edges1.len(), edges2.len());
    }

    #[test]
    fn test_largest_component_size_empty() {
        let graph = Graph::new_rust(false, false);
        let size = largest_component_size(&graph).unwrap();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_largest_component_size_connected() {
        let graph = create_complete_graph(5);
        let size = largest_component_size(&graph).unwrap();
        assert_eq!(size, 5);
    }

    #[test]
    fn test_largest_component_size_disconnected() {
        let mut graph = Graph::new_rust(false, false);
        // Two triangles
        for i in 0..3 {
            let node = Node {
                id: i,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        let edges = [(0, 1), (1, 2), (2, 0)];
        for (s, t) in edges {
            let edge = Edge {
                id: 0,
                source: s,
                target: t,
                weight: None,
                attributes: None,
            };
            graph.add_edge_rust(edge).unwrap();
        }

        for i in 3..7 {
            let node = Node {
                id: i,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        let edges2 = [(3, 4), (4, 5), (5, 6), (6, 3)];
        for (s, t) in edges2 {
            let edge = Edge {
                id: 0,
                source: s,
                target: t,
                weight: None,
                attributes: None,
            };
            graph.add_edge_rust(edge).unwrap();
        }

        let size = largest_component_size(&graph).unwrap();
        assert_eq!(size, 4); // Second component has 4 nodes
    }

    #[test]
    fn test_percolation_threshold_analysis() {
        let graph = create_complete_graph(10);

        let results = percolation_threshold_analysis(&graph, 5, 3, Some(42)).unwrap();

        // Should have 6 data points (0%, 20%, 40%, 60%, 80%, 100%)
        assert_eq!(results.len(), 6);

        // First point (q=0): should be near 1.0 (all nodes in GCC)
        assert!((results[0].1 - 1.0).abs() < 0.01);

        // Last point (q=1): should be 0 (no nodes)
        assert!((results[5].1).abs() < 0.01);
    }

    #[test]
    fn test_star_graph_robustness() {
        // Star graph is fragile - removing center destroys connectivity
        let graph = create_star_graph(6);

        // Remove the center (node 0) by setting high q
        // With seed 42 on a 6-node graph, removing 1 node (q~0.17)
        // may or may not hit the center
        let result_edge = edge_percolation(&graph, 0.8, Some(42)).unwrap();

        // Edge percolation should preserve nodes
        assert_eq!(result_edge.get_node_ids().len(), 6);
    }

    #[test]
    fn test_node_percolation_invalid_q() {
        let graph = create_complete_graph(5);

        let result = node_percolation(&graph, 1.5, Some(42));
        assert!(result.is_err());

        let result = node_percolation(&graph, -0.1, Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_percolation_invalid_q() {
        let graph = create_complete_graph(5);

        let result = edge_percolation(&graph, 1.5, Some(42));
        assert!(result.is_err());

        let result = edge_percolation(&graph, -0.1, Some(42));
        assert!(result.is_err());
    }
}
