use crate::graph::Graph;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

fn largest_component_fraction(
    adj: &FxHashMap<u64, Vec<u64>>,
    alive: &FxHashSet<u64>,
    n_original: usize,
) -> f64 {
    if alive.is_empty() || n_original == 0 {
        return 0.0;
    }
    let mut visited = FxHashSet::default();
    let mut max_size: usize = 0;
    for &start in alive {
        if visited.contains(&start) {
            continue;
        }
        let mut size = 0usize;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(node) = queue.pop_front() {
            size += 1;
            if let Some(neighbors) = adj.get(&node) {
                for &nbr in neighbors {
                    if alive.contains(&nbr) && !visited.contains(&nbr) {
                        visited.insert(nbr);
                        queue.push_back(nbr);
                    }
                }
            }
        }
        if size > max_size {
            max_size = size;
        }
    }
    max_size as f64 / n_original as f64
}

fn build_undirected_adj(graph: &Graph) -> FxHashMap<u64, Vec<u64>> {
    let mut adj: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for &nid in &graph.get_node_ids() {
        adj.insert(nid, Vec::new());
    }
    for edge in graph.get_edges_internal() {
        adj.entry(edge.source).or_default().push(edge.target);
        adj.entry(edge.target).or_default().push(edge.source);
    }
    adj
}

#[pyfunction]
#[pyo3(signature = (graph, strategy="degree", steps=None, seed=None))]
pub fn network_robustness(
    graph: &Graph,
    strategy: &str,
    steps: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Vec<(f64, f64)>> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    if n == 0 {
        return Ok(vec![(0.0, 0.0)]);
    }

    let adj = build_undirected_adj(graph);
    let mut alive: FxHashSet<u64> = node_ids.iter().copied().collect();
    let num_steps = steps.unwrap_or(n);

    let removal_order: Vec<u64> = match strategy {
        "random" => {
            let mut rng = match seed {
                Some(s) => ChaCha8Rng::seed_from_u64(s),
                None => ChaCha8Rng::from_entropy(),
            };
            let mut order = node_ids.clone();
            order.shuffle(&mut rng);
            order
        }
        "degree" => {
            let mut degrees: Vec<(u64, usize)> = node_ids
                .iter()
                .map(|&nid| (nid, adj.get(&nid).map_or(0, |v| v.len())))
                .collect();
            degrees.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            degrees.into_iter().map(|(nid, _)| nid).collect()
        }
        "betweenness" => {
            let bc = crate::algorithms::centrality::betweenness_centrality(graph)?;
            let mut scored: Vec<(u64, f64)> = bc.into_iter().collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
            scored.into_iter().map(|(nid, _)| nid).collect()
        }
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown strategy '{}'. Use 'random', 'degree', or 'betweenness'.", other),
            ));
        }
    };

    let mut curve = Vec::with_capacity(num_steps + 1);
    let initial_frac = largest_component_fraction(&adj, &alive, n);
    curve.push((0.0, initial_frac));

    for (i, &node_id) in removal_order.iter().take(num_steps).enumerate() {
        alive.remove(&node_id);
        let frac_removed = (i + 1) as f64 / n as f64;
        let lc_frac = largest_component_fraction(&adj, &alive, n);
        curve.push((frac_removed, lc_frac));
    }

    Ok(curve)
}
