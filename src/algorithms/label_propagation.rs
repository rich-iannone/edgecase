use crate::graph::Graph;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

/// Semi-supervised label propagation for node classification.
///
/// Spreads known labels through the network: each unlabeled node adopts the
/// most common label among its neighbors (weighted by edge weight), iterating
/// until convergence or `max_iter` is reached. Returns a dict sorted by node ID.
#[pyfunction]
#[pyo3(signature = (graph, known_labels, max_iter=100))]
pub fn label_propagation<'py>(
    py: Python<'py>,
    graph: &Graph,
    known_labels: HashMap<u64, i64>,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let node_ids = graph.get_node_ids();
    if node_ids.is_empty() {
        return Ok(PyDict::new(py));
    }

    // Validate known labels, naming the smallest unknown node (not the first in hash order)
    let valid: FxHashSet<u64> = node_ids.iter().copied().collect();
    if let Some(nid) = known_labels.keys().filter(|nid| !valid.contains(nid)).min() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Node {} in known_labels not found in graph",
            nid
        )));
    }

    if known_labels.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "known_labels must contain at least one labeled node",
        ));
    }

    let directed = graph.is_directed();
    let adj = graph.get_adjacency_list();

    // Build reverse adjacency for directed graphs
    let rev_adj: FxHashMap<u64, Vec<u64>> = if directed {
        let mut ra: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        for &nid in &node_ids {
            ra.insert(nid, Vec::new());
        }
        for &(_, s, t) in graph.edges().iter() {
            ra.entry(t).or_default().push(s);
        }
        ra
    } else {
        FxHashMap::default()
    };

    // Initialize: known labels are fixed; unlabeled nodes get label -1
    let mut labels: FxHashMap<u64, i64> = FxHashMap::default();
    let fixed: FxHashSet<u64> = known_labels.keys().copied().collect();

    for &nid in &node_ids {
        if let Some(&lbl) = known_labels.get(&nid) {
            labels.insert(nid, lbl);
        } else {
            labels.insert(nid, -1);
        }
    }

    for _ in 0..max_iter {
        let mut changed = false;

        for &nid in &node_ids {
            if fixed.contains(&nid) {
                continue;
            }

            // Count weighted votes from neighbors
            let mut votes: FxHashMap<i64, f64> = FxHashMap::default();

            // Outgoing neighbors (or all neighbors for undirected). Each distinct neighbor
            // votes once with the total weight of the edges joining them, so every parallel
            // edge counts with its own weight (adjacency lists repeat a neighbor per edge)
            if let Some(nbrs) = adj.get(&nid) {
                let mut seen: FxHashSet<u64> = FxHashSet::default();
                for &nbr in nbrs {
                    if !seen.insert(nbr) {
                        continue;
                    }
                    let lbl = *labels.get(&nbr).unwrap_or(&-1);
                    if lbl != -1 {
                        let w = graph.get_total_edge_weight(nid, nbr);
                        *votes.entry(lbl).or_insert(0.0) += w;
                    }
                }
            }

            // For directed graphs, also consider incoming edges
            if directed {
                if let Some(preds) = rev_adj.get(&nid) {
                    let mut seen: FxHashSet<u64> = FxHashSet::default();
                    for &pred in preds {
                        if !seen.insert(pred) {
                            continue;
                        }
                        let lbl = *labels.get(&pred).unwrap_or(&-1);
                        if lbl != -1 {
                            let w = graph.get_total_edge_weight(pred, nid);
                            *votes.entry(lbl).or_insert(0.0) += w;
                        }
                    }
                }
            }

            if let Some((&best_label, _)) = votes
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                let old = *labels.get(&nid).unwrap_or(&-1);
                if best_label != old {
                    labels.insert(nid, best_label);
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    let mut sorted_ids = node_ids;
    sorted_ids.sort_unstable();
    sorted_ids
        .iter()
        .map(|nid| (*nid, labels[nid]))
        .into_py_dict(py)
}
