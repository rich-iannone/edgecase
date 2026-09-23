use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

fn compute_weighted_degrees(graph: &Graph, power: f64) -> Vec<f64> {
    let node_ids = graph.get_node_ids();
    let adj = graph.get_adjacency_list();
    let edges = graph.get_edges_internal();

    let mut edge_weights: FxHashMap<(u64, u64), f64> = FxHashMap::default();
    for e in edges {
        let w = e.weight.unwrap_or(1.0);
        edge_weights.insert((e.source, e.target), w);
        if !graph.is_directed() {
            edge_weights.insert((e.target, e.source), w);
        }
    }

    node_ids
        .iter()
        .map(|&nid| {
            let neighbors = adj.get(&nid).map(|v| v.as_slice()).unwrap_or(&[]);
            neighbors
                .iter()
                .map(|&nbr| {
                    let w = edge_weights.get(&(nid, nbr)).copied().unwrap_or(1.0).abs();
                    w.powf(power)
                })
                .sum()
        })
        .collect()
}

fn scale_free_fit(degrees: &[f64]) -> (f64, f64, f64) {
    if degrees.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let max_k = degrees.iter().cloned().fold(0.0f64, f64::max);
    if max_k <= 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let n = degrees.len() as f64;
    let num_bins = ((max_k.sqrt()).ceil() as usize).max(10).min(50);
    let bin_width = max_k / num_bins as f64;

    let mut counts = vec![0usize; num_bins];
    for &d in degrees {
        if d <= 0.0 {
            continue;
        }
        let bin = ((d / bin_width).floor() as usize).min(num_bins - 1);
        counts[bin] += 1;
    }

    let mut log_k = Vec::new();
    let mut log_p = Vec::new();
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let k_center = bin_width * (i as f64 + 0.5);
        let p = count as f64 / n;
        log_k.push(k_center.ln());
        log_p.push(p.ln());
    }

    if log_k.len() < 2 {
        return (0.0, 0.0, 0.0);
    }

    let n_pts = log_k.len() as f64;
    let sum_x: f64 = log_k.iter().sum();
    let sum_y: f64 = log_p.iter().sum();
    let sum_xy: f64 = log_k.iter().zip(log_p.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_k.iter().map(|x| x * x).sum();

    let denom = n_pts * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, 0.0, 0.0);
    }

    let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n_pts;

    let mean_y = sum_y / n_pts;
    let ss_tot: f64 = log_p.iter().map(|y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = log_k
        .iter()
        .zip(log_p.iter())
        .map(|(x, y)| {
            let predicted = slope * x + intercept;
            (y - predicted).powi(2)
        })
        .sum();

    let r_squared = if ss_tot > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    // For scale-free, slope should be negative; use signed R²
    let signed_r2 = if slope < 0.0 { r_squared } else { -r_squared };

    (signed_r2, slope, r_squared)
}

#[pyfunction]
#[pyo3(signature = (graph, powers=None, r_squared_cutoff=0.8))]
pub fn soft_threshold_power(
    graph: &Graph,
    powers: Option<Vec<f64>>,
    r_squared_cutoff: f64,
) -> PyResult<FxHashMap<String, PyObject>> {
    Python::with_gil(|py| {
        let candidate_powers = powers.unwrap_or_else(|| {
            let mut p: Vec<f64> = (1..=10).map(|i| i as f64).collect();
            p.extend([12.0, 14.0, 16.0, 18.0, 20.0]);
            p
        });

        let mut results: Vec<(f64, f64, f64, f64)> = Vec::new(); // (power, signed_r2, slope, mean_k)

        for &power in &candidate_powers {
            let degrees = compute_weighted_degrees(graph, power);
            let mean_k: f64 = if degrees.is_empty() {
                0.0
            } else {
                degrees.iter().sum::<f64>() / degrees.len() as f64
            };
            let (signed_r2, slope, _) = scale_free_fit(&degrees);
            results.push((power, signed_r2, slope, mean_k));
        }

        let recommended = results
            .iter()
            .find(|(_, sr2, _, _)| *sr2 >= r_squared_cutoff)
            .map(|(p, _, _, _)| *p);

        let powers_list: Vec<PyObject> = results.iter().map(|(p, _, _, _)| p.into_pyobject(py).unwrap().into_any().unbind()).collect();
        let r2_list: Vec<PyObject> = results.iter().map(|(_, sr2, _, _)| sr2.into_pyobject(py).unwrap().into_any().unbind()).collect();
        let slope_list: Vec<PyObject> = results.iter().map(|(_, _, s, _)| s.into_pyobject(py).unwrap().into_any().unbind()).collect();
        let mean_k_list: Vec<PyObject> = results.iter().map(|(_, _, _, mk)| mk.into_pyobject(py).unwrap().into_any().unbind()).collect();

        let mut out: FxHashMap<String, PyObject> = FxHashMap::default();
        out.insert("powers".to_string(), powers_list.into_pyobject(py).unwrap().into_any().unbind());
        out.insert("signed_r_squared".to_string(), r2_list.into_pyobject(py).unwrap().into_any().unbind());
        out.insert("slope".to_string(), slope_list.into_pyobject(py).unwrap().into_any().unbind());
        out.insert("mean_connectivity".to_string(), mean_k_list.into_pyobject(py).unwrap().into_any().unbind());
        out.insert("r_squared_cutoff".to_string(), r_squared_cutoff.into_pyobject(py).unwrap().into_any().unbind());

        let rec_obj: PyObject = match recommended {
            Some(p) => p.into_pyobject(py).unwrap().into_any().unbind(),
            None => py.None(),
        };
        out.insert("recommended_power".to_string(), rec_obj);

        Ok(out)
    })
}
