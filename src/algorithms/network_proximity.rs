use crate::algorithms::paths::single_source_dijkstra_distances;
use crate::graph::Graph;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Compute pairwise shortest path distances between two node sets.
/// Returns a map from (source, target) to distance for all reachable pairs.
fn pairwise_distances(
    graph: &Graph,
    set_a: &[u64],
    set_b: &[u64],
) -> PyResult<Vec<(u64, u64, f64)>> {
    let mut pairs = Vec::new();

    for &a in set_a {
        let dists = single_source_dijkstra_distances(graph, a)?;
        for &b in set_b {
            if a == b {
                pairs.push((a, b, 0.0));
            } else if let Some(&d) = dists.get(&b) {
                pairs.push((a, b, d));
            }
        }
    }
    Ok(pairs)
}

/// Compute network proximity between two sets of nodes.
///
/// Methods:
/// - "closest": minimum shortest-path distance across all (a, b) pairs
/// - "shortest": average of shortest-path distances across all (a, b) pairs
/// - "kernel": average of exp(-d) for all (a, b) pairs (diffusion kernel similarity)
/// - "separation": network separation (s_AB - (s_AA + s_BB) / 2) from Menche et al.
#[pyfunction]
#[pyo3(signature = (graph, set_a, set_b, method="closest"))]
pub fn network_proximity(
    graph: &Graph,
    set_a: Vec<u64>,
    set_b: Vec<u64>,
    method: &str,
) -> PyResult<HashMap<String, f64>> {
    if set_a.is_empty() || set_b.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both node sets must be non-empty",
        ));
    }

    let node_ids = graph.get_node_ids();
    let valid: rustc_hash::FxHashSet<u64> = node_ids.iter().copied().collect();
    for &a in &set_a {
        if !valid.contains(&a) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Node {} not found in graph", a),
            ));
        }
    }
    for &b in &set_b {
        if !valid.contains(&b) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Node {} not found in graph", b),
            ));
        }
    }

    match method {
        "closest" => compute_closest(graph, &set_a, &set_b),
        "shortest" => compute_shortest(graph, &set_a, &set_b),
        "kernel" => compute_kernel(graph, &set_a, &set_b),
        "separation" => compute_separation(graph, &set_a, &set_b),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown method '{}'; use 'closest', 'shortest', 'kernel', or 'separation'", method),
        )),
    }
}

fn compute_closest(
    graph: &Graph,
    set_a: &[u64],
    set_b: &[u64],
) -> PyResult<HashMap<String, f64>> {
    let pairs = pairwise_distances(graph, set_a, set_b)?;

    let mut min_dist = f64::INFINITY;
    let mut reachable = 0u64;
    let total = (set_a.len() * set_b.len()) as u64;

    for &(_, _, d) in &pairs {
        if d < min_dist {
            min_dist = d;
        }
        reachable += 1;
    }

    if reachable == 0 {
        min_dist = f64::INFINITY;
    }

    let mut result = HashMap::new();
    result.insert("proximity".to_string(), min_dist);
    result.insert("reachable_pairs".to_string(), reachable as f64);
    result.insert("total_pairs".to_string(), total as f64);
    Ok(result)
}

fn compute_shortest(
    graph: &Graph,
    set_a: &[u64],
    set_b: &[u64],
) -> PyResult<HashMap<String, f64>> {
    let pairs = pairwise_distances(graph, set_a, set_b)?;

    let mut sum = 0.0;
    let mut reachable = 0u64;
    let total = (set_a.len() * set_b.len()) as u64;

    for &(_, _, d) in &pairs {
        sum += d;
        reachable += 1;
    }

    let avg = if reachable > 0 {
        sum / reachable as f64
    } else {
        f64::INFINITY
    };

    let mut result = HashMap::new();
    result.insert("proximity".to_string(), avg);
    result.insert("reachable_pairs".to_string(), reachable as f64);
    result.insert("total_pairs".to_string(), total as f64);
    Ok(result)
}

fn compute_kernel(
    graph: &Graph,
    set_a: &[u64],
    set_b: &[u64],
) -> PyResult<HashMap<String, f64>> {
    let pairs = pairwise_distances(graph, set_a, set_b)?;

    let mut sum = 0.0;
    let mut reachable = 0u64;
    let total = (set_a.len() * set_b.len()) as u64;

    for &(_, _, d) in &pairs {
        sum += (-d).exp();
        reachable += 1;
    }

    let avg = if reachable > 0 {
        sum / reachable as f64
    } else {
        0.0
    };

    let mut result = HashMap::new();
    result.insert("proximity".to_string(), avg);
    result.insert("reachable_pairs".to_string(), reachable as f64);
    result.insert("total_pairs".to_string(), total as f64);
    Ok(result)
}

fn mean_internal_distance(
    graph: &Graph,
    nodes: &[u64],
) -> PyResult<f64> {
    if nodes.len() <= 1 {
        return Ok(0.0);
    }

    let mut sum = 0.0;
    let mut count = 0u64;

    for &a in nodes {
        let dists = single_source_dijkstra_distances(graph, a)?;
        for &b in nodes {
            if a != b {
                if let Some(&d) = dists.get(&b) {
                    // Use minimum distance from a to any node in the set
                    // (Menche et al. definition: d_A = avg over a of min_b d(a,b))
                    // But for internal distance, we use all-pairs average
                    sum += d;
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        Ok(sum / count as f64)
    } else {
        Ok(f64::INFINITY)
    }
}

fn compute_separation(
    graph: &Graph,
    set_a: &[u64],
    set_b: &[u64],
) -> PyResult<HashMap<String, f64>> {
    // Menche et al. (Science 2015) separation:
    // s_AB = <d_AB> - (<d_AA> + <d_BB>) / 2
    // where <d_XY> is the average shortest path between sets X and Y
    // Negative separation means the sets overlap in network space

    let pairs_ab = pairwise_distances(graph, set_a, set_b)?;
    let d_aa = mean_internal_distance(graph, set_a)?;
    let d_bb = mean_internal_distance(graph, set_b)?;

    let mut sum_ab = 0.0;
    let mut count_ab = 0u64;
    for &(_, _, d) in &pairs_ab {
        sum_ab += d;
        count_ab += 1;
    }

    let d_ab = if count_ab > 0 {
        sum_ab / count_ab as f64
    } else {
        f64::INFINITY
    };

    let separation = d_ab - (d_aa + d_bb) / 2.0;

    let mut result = HashMap::new();
    result.insert("separation".to_string(), separation);
    result.insert("d_ab".to_string(), d_ab);
    result.insert("d_aa".to_string(), d_aa);
    result.insert("d_bb".to_string(), d_bb);
    Ok(result)
}
