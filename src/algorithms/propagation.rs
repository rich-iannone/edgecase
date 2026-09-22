//! Network propagation algorithms for gene prioritization and signal diffusion.
//!
//! - `personalized_pagerank`: PageRank with per-node restart probabilities
//! - `random_walk_with_restart`: Iterative RWR from a seed set
//! - `heat_diffusion`: Heat kernel diffusion from seed nodes

use crate::graph::Graph;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Personalized PageRank with a custom restart distribution.
///
/// Unlike standard PageRank (uniform restart), the random surfer teleports
/// back to seed nodes with probabilities given by `personalization`.
#[pyfunction]
#[pyo3(signature = (graph, personalization, damping=0.85, max_iter=100, tol=1e-6))]
pub fn personalized_pagerank(
    graph: &Graph,
    personalization: FxHashMap<u64, f64>,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<FxHashMap<u64, f64>> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(FxHashMap::default());
    }

    if personalization.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "personalization must contain at least one node",
        ));
    }

    let node_ids: Vec<u64> = graph.get_node_ids();

    // Normalize personalization vector
    let p_sum: f64 = personalization.values().sum();
    if p_sum <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "personalization values must sum to a positive number",
        ));
    }
    let inv_p_sum = 1.0 / p_sum;

    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Build personalization vector indexed by position
    let mut p_vec: Vec<f64> = vec![0.0; n];
    for (&node_id, &value) in &personalization {
        if let Some(&idx) = node_to_idx.get(&node_id) {
            p_vec[idx] = value * inv_p_sum;
        }
    }

    // Build CSR incoming edges
    let mut in_degree: Vec<usize> = vec![0; n];
    let mut out_degree: Vec<usize> = vec![0; n];

    for (source_idx, &source_node) in node_ids.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(source_node) {
            out_degree[source_idx] = neighbors.len();
            for &target_node in &neighbors {
                if let Some(&target_idx) = node_to_idx.get(&target_node) {
                    in_degree[target_idx] += 1;
                }
            }
        }
    }

    let total_edges: usize = in_degree.iter().sum();
    let mut incoming_data: Vec<usize> = vec![0; total_edges];
    let mut incoming_offsets: Vec<usize> = Vec::with_capacity(n + 1);
    incoming_offsets.push(0);
    for &count in &in_degree {
        incoming_offsets.push(incoming_offsets.last().unwrap() + count);
    }

    let mut write_pos: Vec<usize> = incoming_offsets[..n].to_vec();
    for (source_idx, &source_node) in node_ids.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(source_node) {
            for &target_node in &neighbors {
                if let Some(&target_idx) = node_to_idx.get(&target_node) {
                    incoming_data[write_pos[target_idx]] = source_idx;
                    write_pos[target_idx] += 1;
                }
            }
        }
    }

    let inv_out_degree: Vec<f64> = out_degree
        .iter()
        .map(|&deg| if deg > 0 { 1.0 / deg as f64 } else { 0.0 })
        .collect();

    // Power iteration with personalized restart
    let mut ranks: Vec<f64> = p_vec.clone();
    let mut new_ranks: Vec<f64> = vec![0.0; n];

    for _iter in 0..max_iter {
        // Dangling node mass (nodes with no outgoing edges)
        let dangling_sum: f64 = node_ids
            .iter()
            .enumerate()
            .filter(|(idx, _)| out_degree[*idx] == 0)
            .map(|(idx, _)| ranks[idx])
            .sum();

        if n >= 1000 {
            let ranks_ref = &ranks;
            let incoming_data_ref = &incoming_data;
            let incoming_offsets_ref = &incoming_offsets;
            let inv_out_degree_ref = &inv_out_degree;
            let p_vec_ref = &p_vec;

            new_ranks
                .par_iter_mut()
                .enumerate()
                .for_each(|(target_idx, new_rank)| {
                    let start = incoming_offsets_ref[target_idx];
                    let end = incoming_offsets_ref[target_idx + 1];

                    let mut rank_sum = 0.0;
                    for i in start..end {
                        let source_idx = incoming_data_ref[i];
                        rank_sum += ranks_ref[source_idx] * inv_out_degree_ref[source_idx];
                    }

                    *new_rank = (1.0 - damping) * p_vec_ref[target_idx]
                        + damping * (rank_sum + dangling_sum * p_vec_ref[target_idx]);
                });
        } else {
            for target_idx in 0..n {
                let start = incoming_offsets[target_idx];
                let end = incoming_offsets[target_idx + 1];

                let mut rank_sum = 0.0;
                for i in start..end {
                    let source_idx = incoming_data[i];
                    rank_sum += ranks[source_idx] * inv_out_degree[source_idx];
                }

                new_ranks[target_idx] = (1.0 - damping) * p_vec[target_idx]
                    + damping * (rank_sum + dangling_sum * p_vec[target_idx]);
            }
        }

        let delta: f64 = ranks
            .iter()
            .zip(new_ranks.iter())
            .map(|(old, new)| (new - old).abs())
            .sum();

        std::mem::swap(&mut ranks, &mut new_ranks);

        if delta < tol {
            break;
        }
    }

    // Normalize
    let sum: f64 = ranks.iter().sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for rank in ranks.iter_mut() {
            *rank *= inv_sum;
        }
    }

    Ok(node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, ranks[idx]))
        .collect())
}

/// Random walk with restart from a set of seed nodes.
///
/// At each step, with probability `restart_prob` the walker teleports back
/// to one of the seed nodes (uniformly), otherwise it follows a random edge.
/// Returns the steady-state visitation probability for every node.
#[pyfunction]
#[pyo3(signature = (graph, seeds, restart_prob=0.15, max_iter=100, tol=1e-6))]
pub fn random_walk_with_restart(
    graph: &Graph,
    seeds: FxHashMap<u64, f64>,
    restart_prob: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<FxHashMap<u64, f64>> {
    // RWR is equivalent to personalized PageRank with damping = 1 - restart_prob
    personalized_pagerank(graph, seeds, 1.0 - restart_prob, max_iter, tol)
}

/// Heat diffusion from seed nodes using the heat kernel.
///
/// Propagates "heat" from seed nodes through the network. The parameter `t`
/// controls how far the heat spreads: small `t` stays local, large `t` reaches
/// the whole graph.
///
/// Uses iterative approximation: e^{-tL} * h0 ≈ (I - tL/k)^k * h0
/// where L is the graph Laplacian and k is the number of steps.
#[pyfunction]
#[pyo3(signature = (graph, seeds, t=1.0, max_iter=50))]
pub fn heat_diffusion(
    graph: &Graph,
    seeds: FxHashMap<u64, f64>,
    t: f64,
    max_iter: usize,
) -> PyResult<FxHashMap<u64, f64>> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(FxHashMap::default());
    }

    if seeds.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seeds must contain at least one node",
        ));
    }

    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "diffusion time t must be positive",
        ));
    }

    let node_ids: Vec<u64> = graph.get_node_ids();
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Build degree vector and adjacency
    let mut degree: Vec<f64> = vec![0.0; n];
    let mut adj_offsets: Vec<usize> = Vec::with_capacity(n + 1);
    let mut adj_data: Vec<usize> = Vec::new();
    adj_offsets.push(0);

    for (idx, &node_id) in node_ids.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node_id) {
            degree[idx] = neighbors.len() as f64;
            for &neighbor in &neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                    adj_data.push(neighbor_idx);
                }
            }
        }
        adj_offsets.push(adj_data.len());
    }

    // Initialize heat vector from seeds
    let mut heat: Vec<f64> = vec![0.0; n];
    for (&node_id, &value) in &seeds {
        if let Some(&idx) = node_to_idx.get(&node_id) {
            heat[idx] = value;
        }
    }

    // Iterative approximation of e^{-tL} * h0
    // Using the relation: h(t+dt) = h(t) + dt * (-L * h(t))
    // where -L*h means: for each node, sum of (neighbor_heat - own_heat * degree)
    let dt = t / max_iter as f64;
    let mut new_heat: Vec<f64> = vec![0.0; n];

    for _step in 0..max_iter {
        for i in 0..n {
            let start = adj_offsets[i];
            let end = adj_offsets[i + 1];

            let mut neighbor_sum = 0.0;
            for j in start..end {
                neighbor_sum += heat[adj_data[j]];
            }

            // -L * h = A*h - D*h, so update is h + dt*(A*h - D*h)
            new_heat[i] = heat[i] + dt * (neighbor_sum - degree[i] * heat[i]);
        }

        std::mem::swap(&mut heat, &mut new_heat);
    }

    // Clamp negative values to zero (numerical artifacts)
    for h in heat.iter_mut() {
        if *h < 0.0 {
            *h = 0.0;
        }
    }

    Ok(node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, heat[idx]))
        .collect())
}

