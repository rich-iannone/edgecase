use pyo3::prelude::*;
use crate::graph::Graph;
use rustc_hash::FxHashSet;

/// Compute dispersion between two nodes u and v.
///
/// Dispersion measures embeddedness in a network. A link between two actors
/// (u and v) has high dispersion when their mutual ties (s and t) are not
/// well connected with each other.
///
/// # Arguments
///
/// * `graph` - The input graph
/// * `u` - Source node ID
/// * `v` - Target node ID
/// * `normalized` - Whether to normalize by embeddedness
///
/// # Returns
///
/// The dispersion score between u and v
#[pyfunction]
#[pyo3(signature = (graph, u, v, normalized=true))]
pub fn dispersion(
    graph: &Graph,
    u: u64,
    v: u64,
    normalized: bool,
) -> PyResult<f64> {
    // Verify nodes exist
    if !graph.has_node_id(u) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Node {} does not exist",
            u
        )));
    }
    if !graph.has_node_id(v) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Node {} does not exist",
            v
        )));
    }

    // Get neighbors of u and v
    let u_nbrs: FxHashSet<u64> = graph.neighbors(u)?.into_iter().collect();
    let v_nbrs: FxHashSet<u64> = graph.neighbors(v)?.into_iter().collect();

    // Find common neighbors (excluding u and v themselves)
    let common: Vec<u64> = u_nbrs
        .intersection(&v_nbrs)
        .copied()
        .filter(|&n| n != u && n != v)
        .collect();

    if common.is_empty() {
        return Ok(0.0);
    }

    // Count pairs of common neighbors that are NOT connected and do NOT share neighbors
    let mut total = 0;
    let n = common.len();

    for i in 0..n {
        let s = common[i];
        let s_nbrs: FxHashSet<u64> = graph
            .neighbors(s)?
            .into_iter()
            .filter(|&n| n != u && n != v)
            .collect();

        for j in (i + 1)..n {
            let t = common[j];

            // Check if s and t are directly connected
            if !s_nbrs.contains(&t) {
                // Check if s and t share any common neighbors (excluding u and v)
                let t_nbrs: FxHashSet<u64> = graph
                    .neighbors(t)?
                    .into_iter()
                    .filter(|&n| n != u && n != v)
                    .collect();

                if s_nbrs.is_disjoint(&t_nbrs) {
                    total += 1;
                }
            }
        }
    }

    let embeddedness = common.len();
    let dispersion_val = if normalized {
        if embeddedness > 0 {
            (total as f64) / (embeddedness as f64)
        } else {
            0.0
        }
    } else {
        total as f64
    };

    Ok(dispersion_val)
}
