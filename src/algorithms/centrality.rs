//! Centrality algorithms (PageRank, betweenness, etc.)

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Calculate betweenness centrality for all nodes using Brandes' algorithm
///
/// Brandes, U. (2001). A faster algorithm for betweenness centrality.
/// Journal of Mathematical Sociology, 25(2), 163-177.
///
/// Optimized with array-based storage for better cache locality
#[pyfunction]
#[inline]
pub fn betweenness_centrality(graph: &Graph) -> PyResult<FxHashMap<u64, f64>> {
    use std::collections::VecDeque;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    // Create node_id to index mapping for array-based storage
    let mut node_to_idx: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (idx, &node_id) in node_ids.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
    }

    // Initialize betweenness centrality (use vec for better cache locality)
    let mut betweenness: Vec<f64> = vec![0.0; n];

    // Pre-allocate reusable arrays
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut sigma: Vec<f64> = vec![0.0; n];
    let mut distance: Vec<i32> = vec![-1; n];
    let mut delta: Vec<f64> = vec![0.0; n];
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

    // For each node as source
    for &source in &node_ids {
        let source_idx = node_to_idx[&source];

        // Clear reusable data structures
        stack.clear();
        for pred in predecessors.iter_mut() {
            pred.clear();
        }
        sigma.fill(0.0);
        distance.fill(-1);
        delta.fill(0.0);
        queue.clear();

        // Initialize source
        sigma[source_idx] = 1.0;
        distance[source_idx] = 0;
        queue.push_back(source_idx);

        // BFS to find shortest paths
        while let Some(v_idx) = queue.pop_front() {
            stack.push(v_idx);
            let v = node_ids[v_idx];

            // Get neighbors of v
            if let Ok(neighbors) = graph.neighbors(v) {
                let dist_v = distance[v_idx];
                let sigma_v = sigma[v_idx];

                for w in neighbors {
                    let w_idx = node_to_idx[&w];

                    // First time we see w?
                    if distance[w_idx] < 0 {
                        queue.push_back(w_idx);
                        distance[w_idx] = dist_v + 1;
                    }

                    // Shortest path to w via v?
                    if distance[w_idx] == dist_v + 1 {
                        sigma[w_idx] += sigma_v;
                        predecessors[w_idx].push(v_idx);
                    }
                }
            }
        }

        // Accumulation: back-propagation of dependencies
        while let Some(w_idx) = stack.pop() {
            let delta_w = delta[w_idx];
            let sigma_w = sigma[w_idx];

            for &v_idx in &predecessors[w_idx] {
                delta[v_idx] += (sigma[v_idx] / sigma_w) * (1.0 + delta_w);
            }

            if w_idx != source_idx {
                betweenness[w_idx] += delta_w;
            }
        }
    }

    // Normalization
    // For undirected graphs, divide by 2 since each shortest path is counted twice
    if !graph.is_directed() {
        let half = 0.5;
        for value in betweenness.iter_mut() {
            *value *= half;
        }
    }

    // Normalization to match NetworkX:
    // - For undirected graphs: multiply by 2.0 / ((n-1)*(n-2))
    // - For directed graphs: multiply by 1.0 / (n-1)*(n-2)
    if n > 2 {
        let normalization = if graph.is_directed() {
            1.0 / ((n - 1) * (n - 2)) as f64
        } else {
            2.0 / ((n - 1) * (n - 2)) as f64
        };

        for value in betweenness.iter_mut() {
            *value *= normalization;
        }
    }

    // Convert back to HashMap with node IDs
    let mut result: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (idx, &node_id) in node_ids.iter().enumerate() {
        result.insert(node_id, betweenness[idx]);
    }

    Ok(result)
}

/// Calculate PageRank for all nodes
///
/// Optimized implementation using incoming edge lists instead of O(n²) neighbor checks.
/// Time complexity: O(iterations × edges) instead of O(iterations × nodes²)
#[pyfunction]
#[pyo3(signature = (graph, damping=0.85, max_iter=100, tol=1e-6))]
#[inline]
pub fn pagerank(
    graph: &Graph,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<FxHashMap<u64, f64>> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(FxHashMap::default());
    }

    let node_ids: Vec<u64> = graph.get_node_ids();

    // Build node_id -> index mapping for dense array access (cache-friendly)
    let mut node_to_idx: FxHashMap<u64, usize> = FxHashMap::default();
    node_to_idx.reserve(n);
    for (idx, &node_id) in node_ids.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
    }

    // Build incoming edges map: target -> [(source, weight)]
    // This is the transpose of the adjacency list
    let mut incoming_edges: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    incoming_edges.reserve(n);

    // Use Vec for out_degree (dense array is faster than HashMap for hot path)
    let mut out_degree: Vec<usize> = vec![0; n];

    // Initialize all nodes in incoming_edges with capacity hints
    for &node_id in &node_ids {
        incoming_edges.insert(node_id, Vec::with_capacity(8)); // Preallocate typical size
    }

    // Build transpose: for each edge u->v, add u to v's incoming list
    for &source_node in &node_ids {
        if let Ok(neighbors) = graph.neighbors(source_node) {
            let degree = neighbors.len();
            if let Some(&source_idx) = node_to_idx.get(&source_node) {
                out_degree[source_idx] = degree;
            }

            for &target_node in &neighbors {
                if let Some(incoming) = incoming_edges.get_mut(&target_node) {
                    incoming.push(source_node);
                }
            }
        }
    }

    // Use dense Vec arrays for ranks (better cache locality than HashMap)
    let mut ranks: Vec<f64> = vec![1.0 / n as f64; n];
    let mut new_ranks: Vec<f64> = vec![0.0; n];

    // Hoist invariant computation outside loop
    let base_rank = (1.0 - damping) / n as f64;
    for _iter in 0..max_iter {
        // Reset new_ranks using fast slice fill
        new_ranks.fill(base_rank);

        // Sum contributions from incoming edges
        for (idx, &node_id) in node_ids.iter().enumerate() {
            if let Some(incoming) = incoming_edges.get(&node_id) {
                let mut rank_sum = 0.0;

                // Hot loop: optimized for cache locality and branch prediction
                for &source_node in incoming {
                    if let Some(&source_idx) = node_to_idx.get(&source_node) {
                        let out_deg = out_degree[source_idx];
                        if out_deg > 0 {
                            rank_sum += ranks[source_idx] / out_deg as f64;
                        }
                    }
                }

                new_ranks[idx] += damping * rank_sum;
            }
        }

        // Check convergence using L1 norm - vectorized iteration
        let delta: f64 = ranks.iter()
            .zip(new_ranks.iter())
            .map(|(old, new)| (new - old).abs())
            .sum();

        // Swap ranks (zero-cost with move semantics)
        std::mem::swap(&mut ranks, &mut new_ranks);

        // Early exit if converged
        if delta < tol {
            break;
        }
    }

    // Normalize ranks to sum to 1.0
    let sum: f64 = ranks.iter().sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum; // Multiplication is faster than division
        for rank in ranks.iter_mut() {
            *rank *= inv_sum;
        }
    }

    // Convert Vec back to HashMap with node_id keys
    let mut result = FxHashMap::default();
    result.reserve(n);
    for (idx, &node_id) in node_ids.iter().enumerate() {
        result.insert(node_id, ranks[idx]);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    #[test]
    fn test_betweenness_centrality_empty_graph() {
        let graph = Graph::new(false, false);
        let centrality = betweenness_centrality(&graph).unwrap();
        assert_eq!(centrality.len(), 0);
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let graph = Graph::new(false, false);
        let ranks = pagerank(&graph, 0.85, 100, 1e-6).unwrap();
        assert_eq!(ranks.len(), 0);
    }

    #[test]
    fn test_pagerank_single_node() {
        let mut graph = Graph::new(false, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();

        let ranks = pagerank(&graph, 0.85, 100, 1e-6).unwrap();
        assert_eq!(ranks.len(), 1);
        assert!((ranks["A"] - 1.0).abs() < 1e-6);
    }
}
