use crate::graph::Graph;
use crate::utils::Seed;
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

fn simulate_spread(
    adj: &FxHashMap<u64, Vec<u64>>,
    seeds: &FxHashSet<u64>,
    probability: f64,
    rng: &mut ChaCha8Rng,
) -> usize {
    let mut activated = seeds.clone();
    let mut frontier: VecDeque<u64> = seeds.iter().copied().collect();

    while let Some(node) = frontier.pop_front() {
        if let Some(neighbors) = adj.get(&node) {
            for &nbr in neighbors {
                if !activated.contains(&nbr) && rng.gen::<f64>() < probability {
                    activated.insert(nbr);
                    frontier.push_back(nbr);
                }
            }
        }
    }
    activated.len()
}

#[pyfunction]
#[pyo3(signature = (graph, k, model="independent_cascade", probability=0.1, n_simulations=100, seed=None))]
pub fn influence_maximization(
    graph: &Graph,
    k: usize,
    model: &str,
    probability: f64,
    n_simulations: usize,
    seed: Option<Seed>,
) -> PyResult<Vec<(u64, f64)>> {
    let seed = seed.map(u64::from);
    if model != "independent_cascade" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown model '{}'. Use 'independent_cascade'.",
            model
        )));
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Probability must be between 0.0 and 1.0.",
        ));
    }
    if n_simulations == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_simulations must be > 0.",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut adj: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for &nid in &node_ids {
        adj.insert(nid, Vec::new());
    }
    for edge in graph.get_edges_internal() {
        adj.entry(edge.source).or_default().push(edge.target);
        if !graph.is_directed() {
            adj.entry(edge.target).or_default().push(edge.source);
        }
    }

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let actual_k = k.min(n);
    let mut selected: Vec<(u64, f64)> = Vec::with_capacity(actual_k);
    let mut seed_set: FxHashSet<u64> = FxHashSet::default();
    let mut candidates: FxHashSet<u64> = node_ids.iter().copied().collect();

    // CELF-style greedy: first pass computes marginal gain for all candidates
    let mut marginal_gains: Vec<(u64, f64)> = Vec::with_capacity(n);
    for &cand in &candidates {
        let mut test_seeds = seed_set.clone();
        test_seeds.insert(cand);
        let mut total_spread = 0.0;
        for _ in 0..n_simulations {
            total_spread += simulate_spread(&adj, &test_seeds, probability, &mut rng) as f64;
        }
        let avg = total_spread / n_simulations as f64;
        marginal_gains.push((cand, avg));
    }
    marginal_gains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Select first node
    if let Some(&(best_id, best_spread)) = marginal_gains.first() {
        seed_set.insert(best_id);
        candidates.remove(&best_id);
        selected.push((best_id, best_spread));
    }

    // Greedy selection for remaining k-1 nodes
    for _ in 1..actual_k {
        let mut best_id = 0u64;
        let mut best_marginal = f64::NEG_INFINITY;
        let current_spread = {
            let mut total = 0.0;
            for _ in 0..n_simulations {
                total += simulate_spread(&adj, &seed_set, probability, &mut rng) as f64;
            }
            total / n_simulations as f64
        };

        for &cand in &candidates {
            let mut test_seeds = seed_set.clone();
            test_seeds.insert(cand);
            let mut total = 0.0;
            for _ in 0..n_simulations {
                total += simulate_spread(&adj, &test_seeds, probability, &mut rng) as f64;
            }
            let marginal = total / n_simulations as f64 - current_spread;
            if marginal > best_marginal {
                best_marginal = marginal;
                best_id = cand;
            }
        }

        seed_set.insert(best_id);
        candidates.remove(&best_id);
        selected.push((best_id, best_marginal));
    }

    Ok(selected)
}
