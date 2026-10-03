//! Null model graph generators for statistical testing.
//!
//! - `configuration_model`: random graph matching a given degree sequence
//! - `degree_preserving_rewire`: edge rewiring preserving all degrees
//! - `expected_degree_graph`: Chung-Lu model with expected degree sequence
//! - `generate_stochastic_block_model`: SBM graph from block sizes and probability matrix

use crate::graph::{Edge, Graph, Node};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashSet;

/// Generate a random graph matching a given degree sequence (configuration model).
///
/// Creates stubs for each node (one per degree), then randomly pairs them.
/// May produce self-loops and multi-edges; call `simplify()` afterward if needed.
#[pyfunction]
#[pyo3(signature = (degree_sequence, directed=false, seed=None))]
pub fn configuration_model(
    degree_sequence: Vec<usize>,
    directed: bool,
    seed: Option<u64>,
) -> PyResult<Graph> {
    let n = degree_sequence.len();
    if n == 0 {
        return Ok(Graph::new_rust(directed, false));
    }

    let total_stubs: usize = degree_sequence.iter().sum();
    if !directed && total_stubs % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Degree sequence must have even sum for undirected graphs",
        ));
    }

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut result = Graph::new_rust(directed, false);

    // Add nodes
    let mut node_ids: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        let node = Node { id: 0, label: Some(format!("{}", i)), attributes: None };
        let id = result.add_node_rust(node)?;
        node_ids.push(id);
    }

    if directed {
        let mut targets: Vec<u64> = Vec::with_capacity(total_stubs);
        for (i, &deg) in degree_sequence.iter().enumerate() {
            for _ in 0..deg {
                targets.push(node_ids[i]);
            }
        }
        targets.shuffle(&mut rng);

        let mut sources: Vec<u64> = Vec::with_capacity(total_stubs);
        for (i, &deg) in degree_sequence.iter().enumerate() {
            for _ in 0..deg {
                sources.push(node_ids[i]);
            }
        }
        sources.shuffle(&mut rng);

        for (s, t) in sources.iter().zip(targets.iter()) {
            let edge = Edge { id: 0, source: *s, target: *t, weight: None, attributes: None };
            let _ = result.add_edge_rust(edge);
        }
    } else {
        let mut stubs: Vec<u64> = Vec::with_capacity(total_stubs);
        for (i, &deg) in degree_sequence.iter().enumerate() {
            for _ in 0..deg {
                stubs.push(node_ids[i]);
            }
        }
        stubs.shuffle(&mut rng);

        for pair in stubs.chunks_exact(2) {
            let edge = Edge { id: 0, source: pair[0], target: pair[1], weight: None, attributes: None };
            let _ = result.add_edge_rust(edge);
        }
    }

    Ok(result)
}

/// Randomly rewire edges while preserving the degree of every node.
///
/// Performs `n_swaps` edge swap operations. Each swap picks two edges
/// (a-b, c-d) and replaces them with (a-d, c-b), checking that the new
/// edges don't already exist.
#[pyfunction]
#[pyo3(signature = (graph, n_swaps=None, seed=None))]
pub fn degree_preserving_rewire(
    graph: &Graph,
    n_swaps: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Graph> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let directed = graph.is_directed();
    let weighted = graph.is_weighted();

    // Collect edges as (source, target) pairs
    let raw_edges = graph.edges();
    let mut edge_list: Vec<(u64, u64)> = raw_edges.iter().map(|&(_, s, t)| (s, t)).collect();

    if edge_list.len() < 2 {
        // Not enough edges to swap; return a copy
        let mut result = Graph::new_rust(directed, weighted);
        for &node_id in graph.get_node_ids().iter() {
            let label = graph.label_from_id(node_id).ok().flatten();
            let node = Node { id: 0, label, attributes: None };
            result.add_node_rust(node)?;
        }
        for &(s, t) in &edge_list {
            let edge = Edge { id: 0, source: s, target: t, weight: None, attributes: None };
            let _ = result.add_edge_rust(edge);
        }
        return Ok(result);
    }

    let swaps = n_swaps.unwrap_or(edge_list.len() * 10);

    let mut edge_set: FxHashSet<(u64, u64)> = FxHashSet::default();
    for &(s, t) in &edge_list {
        if directed {
            edge_set.insert((s, t));
        } else {
            let key = if s <= t { (s, t) } else { (t, s) };
            edge_set.insert(key);
        }
    }

    let m = edge_list.len();
    let mut attempts = 0;
    let mut successful = 0;
    let max_attempts = swaps * 10;

    while successful < swaps && attempts < max_attempts {
        attempts += 1;

        let i = rng.gen_range(0..m);
        let j = rng.gen_range(0..m);
        if i == j {
            continue;
        }

        let (a, b) = edge_list[i];
        let (c, d) = edge_list[j];

        if a == d || c == b {
            continue;
        }

        // Undirected edges are stored with the smaller endpoint first
        let key = |x: u64, y: u64| if directed || x <= y { (x, y) } else { (y, x) };
        let new_e1 = key(a, d);
        let new_e2 = key(c, b);

        if edge_set.contains(&new_e1) || edge_set.contains(&new_e2) {
            continue;
        }

        let old_e1 = key(a, b);
        let old_e2 = key(c, d);

        edge_set.remove(&old_e1);
        edge_set.remove(&old_e2);
        edge_set.insert(new_e1);
        edge_set.insert(new_e2);

        edge_list[i] = if directed { (a, d) } else { new_e1 };
        edge_list[j] = if directed { (c, b) } else { new_e2 };

        successful += 1;
    }

    let mut result = Graph::new_rust(directed, weighted);

    for &node_id in graph.get_node_ids().iter() {
        let label = graph.label_from_id(node_id).ok().flatten();
        let node = Node { id: node_id, label, attributes: None };
        result.add_node_rust(node)?;
    }

    for &(s, t) in &edge_list {
        let edge = Edge { id: 0, source: s, target: t, weight: None, attributes: None };
        let _ = result.add_edge_rust(edge);
    }

    Ok(result)
}

/// Generate a random graph with a given expected degree sequence (Chung-Lu model).
///
/// Each edge (i, j) is included independently with probability
/// `w_i * w_j / sum(w)`, where `w` is the weight vector.
#[pyfunction]
#[pyo3(signature = (weights, seed=None))]
pub fn expected_degree_graph(
    weights: Vec<f64>,
    seed: Option<u64>,
) -> PyResult<Graph> {
    let n = weights.len();
    if n == 0 {
        return Ok(Graph::new_rust(false, false));
    }

    for (i, &w) in weights.iter().enumerate() {
        if w < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Weight at index {} is negative: {}",
                i, w
            )));
        }
    }

    let w_sum: f64 = weights.iter().sum();
    if w_sum <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sum of weights must be positive",
        ));
    }

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut result = Graph::new_rust(false, false);

    let mut node_ids: Vec<u64> = Vec::with_capacity(n);
    for i in 0..n {
        let node = Node { id: 0, label: Some(format!("{}", i)), attributes: None };
        let id = result.add_node_rust(node)?;
        node_ids.push(id);
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let p = (weights[i] * weights[j]) / w_sum;
            let p_clamped = p.min(1.0);
            if rng.gen::<f64>() < p_clamped {
                let edge = Edge { id: 0, source: node_ids[i], target: node_ids[j], weight: None, attributes: None };
                let _ = result.add_edge_rust(edge);
            }
        }
    }

    Ok(result)
}

/// Generate a graph from a stochastic block model.
///
/// `sizes` gives the number of nodes in each block. `p_matrix` is a flat
/// row-major k×k probability matrix where `p_matrix[i*k + j]` is the edge
/// probability between blocks `i` and `j`.
#[pyfunction]
#[pyo3(signature = (sizes, p_matrix, seed=None))]
pub fn generate_stochastic_block_model(
    sizes: Vec<usize>,
    p_matrix: Vec<f64>,
    seed: Option<u64>,
) -> PyResult<Graph> {
    let k = sizes.len();
    if k == 0 {
        return Ok(Graph::new_rust(false, false));
    }

    if p_matrix.len() != k * k {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "p_matrix must have {} entries for {} blocks, got {}",
            k * k,
            k,
            p_matrix.len()
        )));
    }

    for &p in &p_matrix {
        if !(0.0..=1.0).contains(&p) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All probabilities in p_matrix must be between 0.0 and 1.0",
            ));
        }
    }

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut result = Graph::new_rust(false, false);

    // Create nodes grouped by block
    let mut blocks: Vec<Vec<u64>> = Vec::with_capacity(k);
    let mut node_idx = 0;

    for &size in sizes.iter() {
        let mut block_nodes = Vec::with_capacity(size);
        for _ in 0..size {
            let node = Node { id: 0, label: Some(format!("{}", node_idx)), attributes: None };
            let id = result.add_node_rust(node)?;
            block_nodes.push(id);
            node_idx += 1;
        }
        blocks.push(block_nodes);
    }

    // Add edges based on block probability matrix
    for i in 0..k {
        for j in i..k {
            let p = p_matrix[i * k + j];
            if p <= 0.0 {
                continue;
            }

            if i == j {
                // Within-block edges
                let block = &blocks[i];
                for a in 0..block.len() {
                    for b in (a + 1)..block.len() {
                        if rng.gen::<f64>() < p {
                            let edge = Edge { id: 0, source: block[a], target: block[b], weight: None, attributes: None };
                            let _ = result.add_edge_rust(edge);
                        }
                    }
                }
            } else {
                // Between-block edges
                for &node_a in &blocks[i] {
                    for &node_b in &blocks[j] {
                        if rng.gen::<f64>() < p {
                            let edge = Edge { id: 0, source: node_a, target: node_b, weight: None, attributes: None };
                            let _ = result.add_edge_rust(edge);
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}
