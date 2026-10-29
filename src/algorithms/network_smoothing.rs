use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[pyfunction]
#[pyo3(signature = (graph, scores, alpha=0.5, max_iter=100, tol=1e-6))]
pub fn network_smooth(
    graph: &Graph,
    scores: FxHashMap<u64, f64>,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<FxHashMap<u64, f64>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "alpha must be between 0.0 and 1.0.",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    if n == 0 {
        return Ok(FxHashMap::default());
    }

    let adj = graph.get_adjacency_list();

    let mut reverse_adj: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    if graph.is_directed() {
        for &nid in &node_ids {
            reverse_adj.insert(nid, Vec::new());
        }
        for edge in graph.get_edges_internal() {
            reverse_adj
                .entry(edge.target)
                .or_default()
                .push(edge.source);
        }
    }

    let edge_weights: FxHashMap<(u64, u64), f64> = if graph.is_weighted() {
        graph
            .get_edges_internal()
            .iter()
            .map(|e| ((e.source, e.target), e.weight.unwrap_or(1.0)))
            .collect()
    } else {
        FxHashMap::default()
    };

    // Compute weighted out-degrees for normalization
    let mut out_strength: FxHashMap<u64, f64> = FxHashMap::default();
    for &nid in &node_ids {
        let neighbors = adj.get(&nid).map(|v| v.as_slice()).unwrap_or(&[]);
        let strength: f64 = if graph.is_weighted() {
            neighbors
                .iter()
                .map(|&nbr| edge_weights.get(&(nid, nbr)).copied().unwrap_or(1.0))
                .sum()
        } else {
            neighbors.len() as f64
        };
        out_strength.insert(nid, strength);
    }

    // Initial scores: use provided or 0.0
    let initial: FxHashMap<u64, f64> = node_ids
        .iter()
        .map(|&nid| (nid, scores.get(&nid).copied().unwrap_or(0.0)))
        .collect();

    // Iterative smoothing: f(t+1) = alpha * W_norm * f(t) + (1 - alpha) * f(0)
    let mut current = initial.clone();

    for _ in 0..max_iter {
        let mut next: FxHashMap<u64, f64> = FxHashMap::default();

        for &nid in &node_ids {
            let in_neighbors = if graph.is_directed() {
                reverse_adj.get(&nid).map(|v| v.as_slice()).unwrap_or(&[])
            } else {
                adj.get(&nid).map(|v| v.as_slice()).unwrap_or(&[])
            };

            let mut neighbor_sum = 0.0;
            for &nbr in in_neighbors {
                let w = if graph.is_weighted() {
                    if graph.is_directed() {
                        edge_weights.get(&(nbr, nid)).copied().unwrap_or(1.0)
                    } else {
                        edge_weights
                            .get(&(nbr, nid))
                            .or_else(|| edge_weights.get(&(nid, nbr)))
                            .copied()
                            .unwrap_or(1.0)
                    }
                } else {
                    1.0
                };
                let strength = out_strength.get(&nbr).copied().unwrap_or(1.0);
                if strength > 0.0 {
                    neighbor_sum += w / strength * current.get(&nbr).copied().unwrap_or(0.0);
                }
            }

            let smoothed =
                alpha * neighbor_sum + (1.0 - alpha) * initial.get(&nid).copied().unwrap_or(0.0);
            next.insert(nid, smoothed);
        }

        let max_diff: f64 = node_ids
            .iter()
            .map(|&nid| (next.get(&nid).unwrap_or(&0.0) - current.get(&nid).unwrap_or(&0.0)).abs())
            .fold(0.0f64, f64::max);

        current = next;
        if max_diff < tol {
            break;
        }
    }

    Ok(current)
}
