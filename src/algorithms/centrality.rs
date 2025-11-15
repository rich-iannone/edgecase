//! Centrality algorithms (PageRank, betweenness, etc.)

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Calculate betweenness centrality for all nodes using Brandes' algorithm
///
/// Brandes, U. (2001). A faster algorithm for betweenness centrality.
/// Journal of Mathematical Sociology, 25(2), 163-177.
#[pyfunction]
pub fn betweenness_centrality(graph: &Graph) -> PyResult<FxHashMap<u64, f64>> {
    use std::collections::VecDeque;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    // Initialize betweenness centrality for all nodes
    let mut betweenness: FxHashMap<u64, f64> = FxHashMap::default();
    for &node_id in &node_ids {
        betweenness.insert(node_id, 0.0);
    }

    // For each node as source
    for &source in &node_ids {
        // Stack to store nodes in order of non-increasing distance from source
        let mut stack: Vec<u64> = Vec::new();

        // Predecessors on shortest paths from source
        let mut predecessors: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        for &node_id in &node_ids {
            predecessors.insert(node_id, Vec::new());
        }

        // Number of shortest paths from source to each node
        let mut sigma: FxHashMap<u64, f64> = FxHashMap::default();
        for &node_id in &node_ids {
            sigma.insert(node_id, 0.0);
        }
        sigma.insert(source, 1.0);

        // Distance from source
        let mut distance: FxHashMap<u64, i32> = FxHashMap::default();
        for &node_id in &node_ids {
            distance.insert(node_id, -1);
        }
        distance.insert(source, 0);

        // BFS to find shortest paths
        let mut queue: VecDeque<u64> = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);

            // Get neighbors of v
            if let Ok(neighbors) = graph.neighbors(v) {
                for w in neighbors {
                    // First time we see w?
                    if distance[&w] < 0 {
                        queue.push_back(w);
                        distance.insert(w, distance[&v] + 1);
                    }

                    // Shortest path to w via v?
                    if distance[&w] == distance[&v] + 1 {
                        let sigma_w = sigma[&w] + sigma[&v];
                        sigma.insert(w, sigma_w);
                        predecessors.get_mut(&w).unwrap().push(v);
                    }
                }
            }
        }

        // Accumulation: back-propagation of dependencies
        let mut delta: FxHashMap<u64, f64> = FxHashMap::default();
        for &node_id in &node_ids {
            delta.insert(node_id, 0.0);
        }

        // S returns vertices in order of non-increasing distance from source
        while let Some(w) = stack.pop() {
            for &v in &predecessors[&w] {
                let contribution = (sigma[&v] / sigma[&w]) * (1.0 + delta[&w]);
                let delta_v = delta[&v] + contribution;
                delta.insert(v, delta_v);
            }

            if w != source {
                let betweenness_w = betweenness[&w] + delta[&w];
                betweenness.insert(w, betweenness_w);
            }
        }
    }

    // Normalization
    // For undirected graphs, divide by 2 since each shortest path is counted twice
    // For directed graphs, no division needed
    if !graph.is_directed() {
        for value in betweenness.values_mut() {
            *value /= 2.0;
        }
    }

    // Normalization to match NetworkX:
    // - For undirected graphs: divide by ((n-1)*(n-2))/2
    // - For directed graphs: divide by (n-1)*(n-2)
    if n > 2 {
        let normalization = if graph.is_directed() {
            1.0 / ((n - 1) * (n - 2)) as f64
        } else {
            2.0 / ((n - 1) * (n - 2)) as f64
        };

        for value in betweenness.values_mut() {
            *value *= normalization;
        }
    }

    Ok(betweenness)
}

/// Calculate PageRank for all nodes
///
/// Optimized implementation using incoming edge lists instead of O(n²) neighbor checks.
/// Time complexity: O(iterations × edges) instead of O(iterations × nodes²)
#[pyfunction]
#[pyo3(signature = (graph, damping=0.85, max_iter=100, tol=1e-6))]
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

    // Build incoming edges map: target -> [(source, weight)]
    // This is the transpose of the adjacency list
    let mut incoming_edges: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    let mut out_degree: FxHashMap<u64, usize> = FxHashMap::default();

    // Initialize all nodes in incoming_edges
    for &node_id in &node_ids {
        incoming_edges.insert(node_id, Vec::new());
        out_degree.insert(node_id, 0);
    }

    // Build transpose: for each edge u->v, add u to v's incoming list
    for &source_node in &node_ids {
        if let Ok(neighbors) = graph.neighbors(source_node) {
            let degree = neighbors.len();
            out_degree.insert(source_node, degree);

            for &target_node in &neighbors {
                if let Some(incoming) = incoming_edges.get_mut(&target_node) {
                    incoming.push(source_node);
                }
            }
        }
    }

    // Initialize ranks
    let mut ranks = FxHashMap::default();
    let mut new_ranks = FxHashMap::default();
    let initial_rank = 1.0 / n as f64;

    for &node_id in &node_ids {
        ranks.insert(node_id, initial_rank);
        new_ranks.insert(node_id, 0.0);
    }

    // Power iteration
    let base_rank = (1.0 - damping) / n as f64;

    for _iter in 0..max_iter {
        // Reset new_ranks
        for value in new_ranks.values_mut() {
            *value = base_rank;
        }

        // Sum contributions from incoming edges
        for &node_id in &node_ids {
            if let Some(incoming) = incoming_edges.get(&node_id) {
                let mut rank_sum = 0.0;

                for &source_node in incoming {
                    if let Some(&source_rank) = ranks.get(&source_node) {
                        if let Some(&out_deg) = out_degree.get(&source_node) {
                            if out_deg > 0 {
                                rank_sum += source_rank / out_deg as f64;
                            }
                        }
                    }
                }

                if let Some(new_rank) = new_ranks.get_mut(&node_id) {
                    *new_rank += damping * rank_sum;
                }
            }
        }

        // Check convergence using L1 norm
        let mut delta = 0.0;
        for &node_id in &node_ids {
            let old = ranks.get(&node_id).unwrap_or(&0.0);
            let new = new_ranks.get(&node_id).unwrap_or(&0.0);
            delta += (new - old).abs();
        }

        // Swap ranks
        std::mem::swap(&mut ranks, &mut new_ranks);

        // Early exit if converged
        if delta < tol {
            break;
        }
    }

    // Normalize ranks to sum to 1.0
    let sum: f64 = ranks.values().sum();
    if sum > 0.0 {
        for rank in ranks.values_mut() {
            *rank /= sum;
        }
    }

    Ok(ranks)
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
