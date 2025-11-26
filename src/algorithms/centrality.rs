//! Centrality algorithms (PageRank, betweenness, etc.)

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Calculate betweenness centrality for all nodes using Brandes' algorithm
///
/// Brandes, U. (2001). A faster algorithm for betweenness centrality.
/// Journal of Mathematical Sociology, 25(2), 163-177.
///
/// Parallelized with Rayon for near-linear speedup with core count.
/// Each source node's BFS traversal is computed independently in parallel.
///
/// Performance: O(V·E) time, ~6-8x speedup on 8-core CPUs
#[pyfunction]
#[inline]
pub fn betweenness_centrality(graph: &Graph) -> PyResult<FxHashMap<u64, f64>> {
    use rayon::prelude::*;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    // Create node_id to index mapping for array-based storage
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Parallel computation: each source node processes independently
    let betweenness = node_ids
        .par_iter()
        .map(|&source| {
            compute_betweenness_from_source(graph, source, &node_ids, &node_to_idx)
        })
        .reduce(
            || vec![0.0; n],
            |mut acc, partial| {
                for i in 0..n {
                    acc[i] += partial[i];
                }
                acc
            }
        );

    // Normalization
    let mut normalized = betweenness;

    // For undirected graphs, divide by 2 since each shortest path is counted twice
    if !graph.is_directed() {
        let half = 0.5;
        for value in normalized.iter_mut() {
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

        for value in normalized.iter_mut() {
            *value *= normalization;
        }
    }

    // Convert back to HashMap with node IDs
    let mut result: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (idx, &node_id) in node_ids.iter().enumerate() {
        result.insert(node_id, normalized[idx]);
    }

    Ok(result)
}

/// Helper: Compute betweenness contribution from a single source node
///
/// This function is called in parallel for each source node.
/// Returns a Vec of partial betweenness values (indexed by node index).
#[inline]
fn compute_betweenness_from_source(
    graph: &Graph,
    source: u64,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
) -> Vec<f64> {
    use std::collections::VecDeque;

    let n = node_ids.len();
    let source_idx = node_to_idx[&source];

    // Local arrays for this source's BFS
    let mut stack: Vec<usize> = Vec::with_capacity(n);
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut sigma: Vec<f64> = vec![0.0; n];
    let mut distance: Vec<i32> = vec![-1; n];
    let mut delta: Vec<f64> = vec![0.0; n];
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

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
    let mut betweenness = vec![0.0; n];
    while let Some(w_idx) = stack.pop() {
        let delta_w = delta[w_idx];
        let sigma_w = sigma[w_idx];

        for &v_idx in &predecessors[w_idx] {
            delta[v_idx] += (sigma[v_idx] / sigma_w) * (1.0 + delta_w);
        }

        if w_idx != source_idx {
            betweenness[w_idx] = delta_w;
        }
    }

    betweenness
}

/// Calculate load centrality for all nodes
///
/// Load centrality is similar to betweenness centrality but uses a different
/// accumulation formula. It measures the fraction of shortest paths that pass
/// through each node, where the load on each node is accumulated differently
/// than in betweenness centrality.
///
/// The key difference from betweenness:
/// - Betweenness: dependency accumulation with sigma ratios
/// - Load: simpler load accumulation counting predecessor contributions
///
/// References:
/// - Newman, M. E. J. (2001). Scientific collaboration networks. II.
///   Shortest paths, weighted networks, and centrality.
///   Physical Review E, 64(016132).
///
/// Performance: O(V·E) time with BFS from each node, parallelized with Rayon
#[pyfunction]
#[pyo3(signature = (graph, normalized=true, cutoff=None))]
#[inline]
pub fn load_centrality(
    graph: &Graph,
    normalized: bool,
    cutoff: Option<i32>,
) -> PyResult<FxHashMap<u64, f64>> {
    use rayon::prelude::*;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    // Create node_id to index mapping for array-based storage
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Parallel computation: each source node processes independently
    let load = node_ids
        .par_iter()
        .map(|&source| {
            compute_load_from_source(graph, source, &node_ids, &node_to_idx, cutoff)
        })
        .reduce(
            || vec![0.0; n],
            |mut acc, partial| {
                for i in 0..n {
                    acc[i] += partial[i];
                }
                acc
            }
        );

    let mut result_vec = load;

    // Normalization if requested
    if normalized && n > 2 {
        let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
        for value in result_vec.iter_mut() {
            *value *= scale;
        }
    }

    // Convert back to HashMap with node IDs
    let mut result: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (idx, &node_id) in node_ids.iter().enumerate() {
        result.insert(node_id, result_vec[idx]);
    }

    Ok(result)
}

/// Helper: Compute load contribution from a single source node
///
/// This implements the Newman load centrality algorithm which differs
/// from betweenness in the accumulation step.
#[inline]
fn compute_load_from_source(
    graph: &Graph,
    source: u64,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
    cutoff: Option<i32>,
) -> Vec<f64> {
    use std::collections::VecDeque;

    let n = node_ids.len();
    let source_idx = node_to_idx[&source];

    // Local arrays for this source's BFS
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut distance: Vec<i32> = vec![-1; n];
    let mut load: Vec<f64> = vec![1.0; n];  // Initialize with 1.0
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);
    let mut ordered_nodes: Vec<usize> = Vec::with_capacity(n);

    // Initialize source
    distance[source_idx] = 0;
    queue.push_back(source_idx);

    // BFS to find shortest paths
    while let Some(v_idx) = queue.pop_front() {
        ordered_nodes.push(v_idx);
        let v = node_ids[v_idx];
        let dist_v = distance[v_idx];

        // Check cutoff
        if let Some(max_dist) = cutoff {
            if dist_v >= max_dist {
                continue;
            }
        }

        // Get neighbors of v
        if let Ok(neighbors) = graph.neighbors(v) {
            for w in neighbors {
                let w_idx = node_to_idx[&w];

                // First time we see w?
                if distance[w_idx] < 0 {
                    queue.push_back(w_idx);
                    distance[w_idx] = dist_v + 1;
                }

                // Shortest path to w via v?
                if distance[w_idx] == dist_v + 1 {
                    predecessors[w_idx].push(v_idx);
                }
            }
        }
    }

    // Accumulation: process nodes in reverse order (by distance)
    // Start from nodes furthest from source
    while let Some(v_idx) = ordered_nodes.pop() {
        if v_idx == source_idx {
            continue;  // Skip source
        }

        let num_preds = predecessors[v_idx].len();
        if num_preds > 0 {
            let load_v = load[v_idx];
            let contribution = load_v / num_preds as f64;

            for &pred_idx in &predecessors[v_idx] {
                if pred_idx == source_idx {
                    break;  // Stop if we hit source
                }
                load[pred_idx] += contribution;
            }
        }
    }

    // Subtract 1 from each (initial value) and zero out source
    for i in 0..n {
        if i != source_idx && distance[i] >= 0 {
            load[i] -= 1.0;
        } else if i == source_idx {
            load[i] = 0.0;
        } else {
            // Unreachable nodes get 0
            load[i] = 0.0;
        }
    }

    load
}

/// Calculate harmonic centrality for all nodes
///
/// Harmonic centrality is the sum of the reciprocal of the shortest path distances
/// from a node to all other reachable nodes. For unreachable nodes, the distance is
/// considered infinite, and the reciprocal is 0.
///
/// Formula: H(v) = sum_{u != v} 1/d(v,u)
///
/// When normalized=true, divides by (n-1) to get the mean inverse distance.
///
/// References:
/// - Marchiori, M., & Latora, V. (2000). Harmony in the small-world.
///   Physica A: Statistical Mechanics and its Applications, 285(3-4), 539-546.
///
/// Performance: O(V·E) time with BFS from each node, parallelized with Rayon
#[pyfunction]
#[pyo3(signature = (graph, mode="out", weights=None, normalized=false, cutoff=-1))]
#[inline]
pub fn harmonic_centrality(
    graph: &Graph,
    mode: &str,
    weights: Option<&str>,
    normalized: bool,
    cutoff: i32,
) -> PyResult<FxHashMap<u64, f64>> {
    use rayon::prelude::*;

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    // Validate mode parameter
    let reverse_edges = match mode {
        "out" => false,
        "in" => true,
        "all" | "total" => {
            // For undirected mode, we'll handle it by treating the graph as undirected
            if graph.is_directed() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "mode='all' requires treating directed graph as undirected"
                ));
            }
            false
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid mode '{}'. Must be 'out', 'in', 'all', or 'total'", mode)
            ));
        }
    };

    // Currently only supporting unweighted version
    if weights.is_some() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Weighted harmonic centrality not yet implemented"
        ));
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Parallel computation: compute harmonic centrality for each node
    let harmonic: Vec<f64> = node_ids
        .par_iter()
        .map(|&source| {
            compute_harmonic_from_node(
                graph,
                source,
                &node_ids,
                &node_to_idx,
                reverse_edges,
                cutoff,
            )
        })
        .collect();

    // Normalize if requested
    let mut result: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(n, Default::default());

    if normalized && n > 1 {
        let normalization = 1.0 / (n - 1) as f64;
        for (idx, &node_id) in node_ids.iter().enumerate() {
            result.insert(node_id, harmonic[idx] * normalization);
        }
    } else {
        for (idx, &node_id) in node_ids.iter().enumerate() {
            result.insert(node_id, harmonic[idx]);
        }
    }

    Ok(result)
}

/// Helper: Compute harmonic centrality for a single node
///
/// Uses BFS to find shortest path distances to all reachable nodes,
/// then sums 1/distance for each reachable node.
#[inline]
fn compute_harmonic_from_node(
    graph: &Graph,
    source: u64,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
    reverse_edges: bool,
    cutoff: i32,
) -> f64 {
    use std::collections::VecDeque;

    let n = node_ids.len();
    let source_idx = node_to_idx[&source];

    let mut distance: Vec<i32> = vec![-1; n];
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

    // Initialize source
    distance[source_idx] = 0;
    queue.push_back(source_idx);

    let mut harmonic_sum = 0.0;

    // BFS to find shortest path distances
    while let Some(v_idx) = queue.pop_front() {
        let v = node_ids[v_idx];
        let dist_v = distance[v_idx];

        // Check cutoff
        if cutoff >= 0 && dist_v >= cutoff {
            continue;
        }

        // Get neighbors based on edge direction
        let neighbors = if reverse_edges {
            // For 'in' mode, we need predecessors (nodes that point to v)
            // This requires iterating through all nodes - less efficient
            let mut preds = Vec::new();
            for &u in node_ids {
                if u != v {
                    if let Ok(u_neighbors) = graph.neighbors(u) {
                        if u_neighbors.contains(&v) {
                            preds.push(u);
                        }
                    }
                }
            }
            Ok(preds)
        } else {
            // For 'out' mode, use normal successors
            graph.neighbors(v)
        };

        if let Ok(neighbors) = neighbors {
            for w in neighbors {
                if let Some(&w_idx) = node_to_idx.get(&w) {
                    // First time we see w?
                    if distance[w_idx] < 0 {
                        let new_dist = dist_v + 1;

                        // Check if we should explore this node (cutoff check)
                        if cutoff < 0 || new_dist <= cutoff {
                            distance[w_idx] = new_dist;
                            queue.push_back(w_idx);

                            // Add to harmonic sum: 1/distance
                            harmonic_sum += 1.0 / new_dist as f64;
                        }
                    }
                }
            }
        }
    }

    harmonic_sum
}

/// Calculate PageRank for all nodes
///
/// Highly optimized implementation for large graphs using:
/// - CSR-like format with index-based storage (no HashMap lookups in hot loop)
/// - Precomputed inverse out-degrees (multiplication instead of division)
/// - Parallel computation with Rayon for large graphs
/// - Contiguous memory layout for cache efficiency
///
/// Time complexity: O(iterations × edges)
/// Space complexity: O(nodes + edges)
#[pyfunction]
#[pyo3(signature = (graph, damping=0.85, max_iter=100, tol=1e-6))]
#[inline]
pub fn pagerank(
    graph: &Graph,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<FxHashMap<u64, f64>> {
    use rayon::prelude::*;

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(FxHashMap::default());
    }

    let node_ids: Vec<u64> = graph.get_node_ids();

    // Build node_id -> index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // ============================================================
    // BUILD CSR-LIKE INCOMING EDGES STRUCTURE
    // Store indices instead of node IDs to eliminate HashMap lookups
    // ============================================================

    // First pass: count incoming edges per node
    let mut in_degree: Vec<usize> = vec![0; n];
    let mut out_degree: Vec<usize> = vec![0; n];

    for (source_idx, &source_node) in node_ids.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(source_node) {
            let degree = neighbors.len();
            out_degree[source_idx] = degree;

            for &target_node in &neighbors {
                if let Some(&target_idx) = node_to_idx.get(&target_node) {
                    in_degree[target_idx] += 1;
                }
            }
        }
    }

    // Build CSR structure: incoming_data stores all source indices contiguously
    // incoming_offsets[i] = start position of node i's incoming edges
    let total_edges: usize = in_degree.iter().sum();
    let mut incoming_data: Vec<usize> = vec![0; total_edges];
    let mut incoming_offsets: Vec<usize> = Vec::with_capacity(n + 1);
    incoming_offsets.push(0);

    for &count in &in_degree {
        incoming_offsets.push(incoming_offsets.last().unwrap() + count);
    }

    // Temporary counter for filling incoming_data (tracks current write position per node)
    let mut write_pos: Vec<usize> = incoming_offsets[..n].to_vec();

    // Second pass: fill incoming edges
    for (source_idx, &source_node) in node_ids.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(source_node) {
            for &target_node in &neighbors {
                if let Some(&target_idx) = node_to_idx.get(&target_node) {
                    let pos = write_pos[target_idx];
                    incoming_data[pos] = source_idx;
                    write_pos[target_idx] += 1;
                }
            }
        }
    }

    // ============================================================
    // PRECOMPUTE INVERSE OUT-DEGREES (avoid division in hot loop)
    // ============================================================
    let inv_out_degree: Vec<f64> = out_degree
        .iter()
        .map(|&deg| if deg > 0 { 1.0 / deg as f64 } else { 0.0 })
        .collect();

    // ============================================================
    // POWER ITERATION WITH PARALLELIZATION
    // ============================================================
    let mut ranks: Vec<f64> = vec![1.0 / n as f64; n];
    let mut new_ranks: Vec<f64> = vec![0.0; n];
    let base_rank = (1.0 - damping) / n as f64;

    for _iter in 0..max_iter {
        // Parallel computation for large graphs (threshold: 1000 nodes)
        if n >= 1000 {
            // Parallel version
            new_ranks.par_iter_mut().enumerate().for_each(|(target_idx, new_rank)| {
                *new_rank = base_rank;

                // Get incoming edges for this target node
                let start = incoming_offsets[target_idx];
                let end = incoming_offsets[target_idx + 1];

                // Sum contributions from all incoming edges
                let mut rank_sum = 0.0;
                for i in start..end {
                    let source_idx = incoming_data[i];
                    rank_sum += ranks[source_idx] * inv_out_degree[source_idx];
                }

                *new_rank += damping * rank_sum;
            });
        } else {
            // Sequential version for small graphs
            for target_idx in 0..n {
                new_ranks[target_idx] = base_rank;

                let start = incoming_offsets[target_idx];
                let end = incoming_offsets[target_idx + 1];

                let mut rank_sum = 0.0;
                for i in start..end {
                    let source_idx = incoming_data[i];
                    rank_sum += ranks[source_idx] * inv_out_degree[source_idx];
                }

                new_ranks[target_idx] += damping * rank_sum;
            }
        }

        // Check convergence using L1 norm
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
        let inv_sum = 1.0 / sum;
        for rank in ranks.iter_mut() {
            *rank *= inv_sum;
        }
    }

    // Convert Vec back to HashMap with node_id keys
    let mut result = FxHashMap::with_capacity_and_hasher(n, Default::default());
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

/// Calculate second order centrality for all nodes in an undirected graph
///
/// Second order centrality measures the standard deviation of return times
/// for a perpetual random walk on the graph. Lower values indicate higher centrality.
///
/// Based on: Kermarrec, Le Merrer, Sericola, Trédan (2011)
/// "Second order centrality: Distributed assessment of nodes criticity in complex networks"
/// Computer Communications 34(5):619-628
///
/// Algorithm:
/// 1. Balance graph for Metropolis-Hastings random walk (add self-loops)
/// 2. Create transition probability matrix P
/// 3. For each node i, compute M[:, i] by solving: (I - Q_i) * M[:, i] = 1
///    where Q_i is P with column i zeroed
/// 4. Centrality = sqrt(2 * sum(M[:, i]) - n * (n + 1))
///
/// Complexity: O(n^3) due to solving n linear systems of size n
/// This makes it viable only for small to medium graphs (< 1000 nodes)
///
/// Note: Currently only supports unweighted graphs
///
/// Returns:
/// - HashMap mapping node_label to second order centrality value
///
/// Errors:
/// - Returns error if graph is directed (not supported)
/// - Returns error if graph is empty
/// - Returns error if graph is not connected
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn second_order_centrality(
    graph: &Graph,
) -> PyResult<FxHashMap<String, f64>> {
    use nalgebra::{DMatrix, DVector};

    // Check if graph is directed
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Second order centrality is not defined for directed graphs",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Empty graph"));
    }

    // Check connectivity - use BFS from first node
    {
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0);
        visited[0] = true;
        let mut count = 1;

        while let Some(idx) = queue.pop_front() {
            let node_id = node_ids[idx];
            if let Ok(neighbors) = graph.neighbors(node_id) {
                for neighbor_id in neighbors {
                    if let Some(neighbor_idx) = node_ids.iter().position(|&id| id == neighbor_id) {
                        if !visited[neighbor_idx] {
                            visited[neighbor_idx] = true;
                            queue.push_back(neighbor_idx);
                            count += 1;
                        }
                    }
                }
            }
        }

        if count != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Graph is not connected",
            ));
        }
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Build degree vector (unweighted)
    let mut in_degree = vec![0; n];
    for i in 0..n {
        let node_id = node_ids[i];
        if let Ok(neighbors) = graph.neighbors(node_id) {
            in_degree[i] = neighbors.len();
        }
    }

    let d_max = *in_degree.iter().max().unwrap_or(&0);

    // Build transition probability matrix P
    // Balance graph by adding self-loops to achieve equal in-degree
    let mut p_matrix = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        let node_id = node_ids[i];
        let self_loop_weight = d_max - in_degree[i];

        // Add self-loop if needed
        if self_loop_weight > 0 {
            p_matrix[(i, i)] = self_loop_weight as f64;
        }

        // Add edges to neighbors (weight 1.0 each)
        if let Ok(neighbors) = graph.neighbors(node_id) {
            for neighbor_id in neighbors {
                if let Some(&j) = node_to_idx.get(&neighbor_id) {
                    p_matrix[(i, j)] += 1.0;
                }
            }
        }

        // Normalize row to get transition probabilities
        let row_sum = p_matrix.row(i).sum();
        if row_sum > 0.0 {
            for j in 0..n {
                p_matrix[(i, j)] /= row_sum;
            }
        }
    }

    // Compute M matrix: M[:, i] = solve((I - Q_i), ones)
    // where Q_i is P with column i zeroed
    let identity = DMatrix::<f64>::identity(n, n);
    let ones = DVector::<f64>::from_element(n, 1.0);
    let mut m_matrix = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        // Create Q_i: P with column i zeroed
        let mut q_i = p_matrix.clone();
        for j in 0..n {
            q_i[(j, i)] = 0.0;
        }

        // Solve (I - Q_i) * m_col = ones
        let system = &identity - &q_i;

        // Use LU decomposition to solve the system
        match system.lu().solve(&ones) {
            Some(m_col) => {
                for j in 0..n {
                    m_matrix[(j, i)] = m_col[j];
                }
            }
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Failed to solve linear system (matrix may be singular)",
                ));
            }
        }
    }

    // Compute second order centrality: sqrt(2 * sum(M[:, i]) - n * (n + 1))
    let mut result = FxHashMap::default();
    let n_f64 = n as f64;

    for i in 0..n {
        let node_id = node_ids[i];
        let node = graph
            .get_node(node_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Node not found")
            })?;

        let label = node.label.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Node has no label")
        })?;

        let m_sum: f64 = m_matrix.column(i).iter().sum();
        let centrality = (2.0 * m_sum - n_f64 * (n_f64 + 1.0)).sqrt();

        result.insert(label, centrality);
    }

    Ok(result)
}

/// Compute average neighbor degree for each node.
///
/// For each node, calculates the average degree of its neighbors. This metric is used in
/// degree correlation analysis and can indicate whether nodes tend to connect to similar-degree
/// nodes (assortative mixing) or different-degree nodes (disassortative mixing).
///
/// For directed graphs:
/// - `source="out"`: Use out-neighbors (successors) of each node
/// - `source="in"`: Use in-neighbors (predecessors) of each node
/// - `source="in+out"`: Use both in-neighbors and out-neighbors
/// - `target="out"`: Use out-degree of neighbors
/// - `target="in"`: Use in-degree of neighbors
/// - `target="in+out"`: Use total degree of neighbors
///
/// For weighted graphs, the average is weighted by edge weights:
/// k_nn,i = (1 / s_i) * Σ_j w_ij * k_j
/// where s_i is the weighted degree of node i.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `source` - For directed graphs: "in", "out", or "in+out" (default: "out")
/// * `target` - For directed graphs: "in", "out", or "in+out" (default: "out")
/// * `weight` - Optional edge attribute name for weights
///
/// # Returns
/// * `HashMap<String, f64>` - Map from node label to average neighbor degree
///
/// # Errors
/// * Returns error if graph is empty
/// * Returns error if weight attribute is specified but not found or not numeric
/// * Returns error if source/target parameters are invalid for directed/undirected graphs
///
/// # References
/// * Barrat, A., Barthélemy, M., Pastor-Satorras, R., & Vespignani, A. (2004).
///   "The architecture of complex weighted networks". PNAS 101(11): 3747-3752.
#[pyfunction]
#[pyo3(signature = (graph, source=None, target=None, weight=None))]
pub fn average_neighbor_degree(
    graph: &Graph,
    source: Option<&str>,
    target: Option<&str>,
    weight: Option<&str>,
) -> PyResult<FxHashMap<String, f64>> {
    let n = graph.num_nodes();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute average neighbor degree for empty graph",
        ));
    }

    // Validate source/target parameters
    let source_dir = if graph.is_directed() {
        match source.unwrap_or("out") {
            "in" => "in",
            "out" => "out",
            "in+out" => "in+out",
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "source must be 'in', 'out', or 'in+out' for directed graphs",
                ))
            }
        }
    } else {
        if source.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "source parameter not allowed for undirected graphs",
            ));
        }
        "out" // unused for undirected
    };

    let target_dir = if graph.is_directed() {
        match target.unwrap_or("out") {
            "in" => "in",
            "out" => "out",
            "in+out" => "in+out",
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "target must be 'in', 'out', or 'in+out' for directed graphs",
                ))
            }
        }
    } else {
        if target.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "target parameter not allowed for undirected graphs",
            ));
        }
        "out" // unused for undirected
    };

    // Compute degrees for all nodes using public methods
    let node_ids = graph.get_node_ids();
    let all_node_ids = node_ids.clone(); // Clone for use in closures

    // Get target degree for a node based on target_dir setting
    let get_target_degree = |node_id: u64| -> f64 {
        if !graph.is_directed() {
            graph.degree_rust(node_id) as f64
        } else {
            match target_dir {
                "in" => graph.get_in_degree(node_id) as f64,
                "out" => graph.get_out_degree(node_id) as f64,
                "in+out" => (graph.get_in_degree(node_id) + graph.get_out_degree(node_id)) as f64,
                _ => unreachable!(),
            }
        }
    };

    // Build adjacency structure using the neighbors method
    let get_neighbors = |node_id: u64| -> PyResult<Vec<u64>> {
        if !graph.is_directed() {
            // For undirected graphs, get neighbors
            graph.neighbors(node_id)
        } else {
            // For directed graphs, handle source direction
            match source_dir {
                "out" => graph.neighbors(node_id), // successors
                "in" => {
                    // Get predecessors by finding all edges pointing to this node
                    let mut preds = Vec::new();
                    for other_id in &all_node_ids {
                        if let Ok(neighbors) = graph.neighbors(*other_id) {
                            if neighbors.contains(&node_id) {
                                preds.push(*other_id);
                            }
                        }
                    }
                    Ok(preds)
                }
                "in+out" => {
                    // Get both successors and predecessors
                    let mut all = graph.neighbors(node_id)?;
                    for other_id in &all_node_ids {
                        if let Ok(neighbors) = graph.neighbors(*other_id) {
                            if neighbors.contains(&node_id) && !all.contains(other_id) {
                                all.push(*other_id);
                            }
                        }
                    }
                    Ok(all)
                }
                _ => unreachable!(),
            }
        }
    };

    // Get edge weights if specified
    if weight.is_some() {
        // For now, weighted version is not implemented
        // TODO: Access edge DataFrame properly to get weights
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Weighted average_neighbor_degree not yet implemented",
        ));
    }

    // Compute average neighbor degree for each node
    let mut result = FxHashMap::default();

    for node_id in node_ids {
        let node = graph
            .get_node(node_id)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Node not found")
            })?;

        let label = node.label.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Node has no label")
        })?;

        let neighbors = get_neighbors(node_id)?;

        if neighbors.is_empty() {
            result.insert(label, 0.0);
            continue;
        }

        // Unweighted version: simple average
        let sum: f64 = neighbors
            .iter()
            .map(|&neighbor_id| get_target_degree(neighbor_id))
            .sum();
        let avg_degree = sum / neighbors.len() as f64;

        result.insert(label, avg_degree);
    }

    Ok(result)
}
