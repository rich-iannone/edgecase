use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;
use std::collections::HashMap;

/// Compute the topological overlap matrix (TOM).
///
/// For each pair of nodes (i, j), TOM measures how much they share the same
/// neighbors, normalized by connectivity:
///
///   TOM(i, j) = (|N(i) ∩ N(j)| + a_ij) / (min(k_i, k_j) + 1 - a_ij)
///
/// where a_ij = 1 if (i,j) is an edge and k_i is the degree of node i.
///
/// Returns a nested dict: result[node_i][node_j] = TOM value.
#[pyfunction]
#[pyo3(signature = (graph,))]
pub fn topological_overlap_matrix(
    graph: &Graph,
) -> PyResult<HashMap<u64, HashMap<u64, f64>>> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    // Build adjacency sets for fast intersection
    let adj = graph.get_adjacency_list();
    let mut neighbor_sets: HashMap<u64, FxHashSet<u64>> = HashMap::new();

    for &nid in &node_ids {
        let mut s = FxHashSet::default();
        if let Some(nbrs) = adj.get(&nid) {
            for &nbr in nbrs {
                s.insert(nbr);
            }
        }
        // For directed graphs, also include incoming edges as neighbors
        if graph.is_directed() {
            for &(_, src, tgt) in graph.edges().iter() {
                if tgt == nid {
                    s.insert(src);
                }
            }
        }
        neighbor_sets.insert(nid, s);
    }

    let mut result: HashMap<u64, HashMap<u64, f64>> = HashMap::new();

    for i in 0..n {
        let ni = node_ids[i];
        let set_i = &neighbor_sets[&ni];
        let ki = set_i.len();
        let mut row: HashMap<u64, f64> = HashMap::new();

        for &nj in &node_ids[..n] {
            if ni == nj {
                row.insert(nj, 1.0);
                continue;
            }

            let set_j = &neighbor_sets[&nj];
            let kj = set_j.len();

            // Shared neighbors
            let shared = set_i.intersection(set_j).count();

            // a_ij: is there a direct edge between i and j?
            let a_ij = if set_i.contains(&nj) { 1usize } else { 0usize };

            let min_k = ki.min(kj);
            let denom = min_k + 1 - a_ij;

            let tom = if denom > 0 {
                (shared + a_ij) as f64 / denom as f64
            } else {
                0.0
            };

            row.insert(nj, tom);
        }
        result.insert(ni, row);
    }

    Ok(result)
}
