//! DomiRank centrality algorithm
//!
//! Implements the DomiRank centrality measure from:
//! Engsig, M., Tejedor, A., Moreno, Y. et al. "DomiRank Centrality reveals
//! structural fragility of complex networks via node dominance."
//! Nat Commun 15, 56 (2024). https://doi.org/10.1038/s41467-023-44257-0

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Find all elementary cycles in the graph using DomiRank's iterative method
///
/// DomiRank centrality computes node dominance in their neighborhoods.
/// The competition parameter sigma tunes the balance between local and global
/// topological information. For sigma -> 0, converges to degree centrality.
/// For large sigma, identifies globally dominant nodes.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `sigma` - Competition parameter (if None, auto-computed from smallest eigenvalue and alpha)
/// * `alpha` - Alpha parameter for sigma computation (default 0.95)
/// * `dt` - Time step for Newton iteration (default 0.1)
/// * `epsilon` - Convergence tolerance (default 1e-5)
/// * `max_iter` - Maximum iterations (default 1000)
/// * `patience` - Steps between convergence checks (default 10)
///
/// # Returns
/// HashMap of node_id -> dominance score, along with sigma value and convergence status
#[pyfunction]
#[pyo3(signature = (graph, sigma=None, alpha=0.95, dt=0.1, epsilon=1e-5, max_iter=1000, patience=10, max_depth=50))]
#[inline]
pub fn domirank(
    graph: &Graph,
    sigma: Option<f64>,
    alpha: f64,
    dt: f64,
    epsilon: f64,
    max_iter: usize,
    patience: usize,
    max_depth: usize,
) -> PyResult<(FxHashMap<u64, f64>, f64, bool)> {
    // Validate parameters
    if alpha <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "alpha must be positive (alpha > 0)"
        ));
    }
    if dt <= 0.0 || dt >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dt must be in range (0, 1)"
        ));
    }
    if epsilon <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "epsilon must be positive (epsilon > 0)"
        ));
    }
    if patience * 3 > max_iter {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_iter must be > 3 * patience"
        ));
    }

    let node_ids: Vec<u64> = graph.get_node_ids();
    let num_nodes = node_ids.len();

    if num_nodes == 0 {
        return Ok((FxHashMap::default(), 0.0, true));
    }

    // Build adjacency structure with degrees
    let mut adj_list: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(num_nodes, Default::default());
    let mut degrees: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(num_nodes, Default::default());

    for &node_id in &node_ids {
        let neighbors = graph.neighbors(node_id).unwrap_or_default();
        let degree = neighbors.len() as f64;
        adj_list.insert(node_id, neighbors);
        degrees.insert(node_id, degree);
    }

    // Determine sigma
    let final_sigma = if let Some(s) = sigma {
        s
    } else {
        // Find smallest eigenvalue (most negative) using bisection
        let smallest_eigenvalue = find_smallest_eigenvalue(
            &adj_list,
            &degrees,
            max_depth,
            dt,
            epsilon,
            max_iter,
            patience,
        )?;
        alpha / smallest_eigenvalue.abs()
    };

    // Run iterative DomiRank
    let (psi, converged) = domirank_iterative(
        &adj_list,
        &degrees,
        final_sigma,
        dt,
        epsilon,
        max_iter,
        patience,
    )?;

    Ok((psi, final_sigma, converged))
}

/// Iterative DomiRank computation using Newton's method
///
/// Solves: d𝚪ᵢ(t)/dt = σ(dᵢ - Σⱼ Aᵢⱼ 𝚪ⱼ(t)) - 𝚪ᵢ(t)
///
/// # Arguments
/// * `adj_list` - Adjacency list representation
/// * `degrees` - Node degrees (weighted if graph is weighted)
/// * `sigma` - Competition parameter
/// * `dt` - Time step
/// * `epsilon` - Convergence tolerance
/// * `max_iter` - Maximum iterations
/// * `patience` - Steps between convergence checks
///
/// # Returns
/// (dominance scores, converged flag)
#[inline]
fn domirank_iterative(
    adj_list: &FxHashMap<u64, Vec<u64>>,
    degrees: &FxHashMap<u64, f64>,
    sigma: f64,
    dt: f64,
    epsilon: f64,
    max_iter: usize,
    patience: usize,
) -> PyResult<(FxHashMap<u64, f64>, bool)> {
    let num_nodes = degrees.len();

    // Initialize psi with small uniform values
    let mut psi: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(num_nodes, Default::default());
    let initial_val = 1.0 / (num_nodes as f64);
    for &node_id in degrees.keys() {
        psi.insert(node_id, initial_val);
    }

    // Track convergence
    let mut max_vals: Vec<f64> = Vec::with_capacity(max_iter / patience + 1);
    let boundary = epsilon * (num_nodes as f64) * dt;

    // Preallocate new_psi outside loop for swap optimization
    let mut new_psi: FxHashMap<u64, f64> = FxHashMap::with_capacity_and_hasher(num_nodes, Default::default());

    for i in 0..max_iter {
        // Compute update: temp_val = σ(d - A·psi) - psi
        let mut max_temp_val: f64 = 0.0;
        let mut sum_abs_temp_val = 0.0;
        new_psi.clear();

        for (&node_id, &degree) in degrees.iter() {
            let current_psi = *psi.get(&node_id).unwrap_or(&0.0);

            // Compute Σⱼ Aᵢⱼ psiⱼ (sum of neighbor psi values)
            let neighbor_sum: f64 = adj_list
                .get(&node_id)
                .map(|neighbors| {
                    neighbors
                        .iter()
                        .map(|&neighbor_id| *psi.get(&neighbor_id).unwrap_or(&0.0))
                        .sum()
                })
                .unwrap_or(0.0);

            // Update: d𝚪/dt = σ(d - Σpsi) - 𝚪
            let temp_val = sigma * (degree - neighbor_sum) - current_psi;
            let new_val = current_psi + temp_val * dt;

            new_psi.insert(node_id, new_val);

            max_temp_val = max_temp_val.max(temp_val.abs());
            sum_abs_temp_val += temp_val.abs();
        }

        // Swap instead of reassignment to avoid allocation
        std::mem::swap(&mut psi, &mut new_psi);

        // Check convergence periodically
        if i % patience == 0 {
            if sum_abs_temp_val < boundary {
                // Converged
                return Ok((psi, true));
            }

            max_vals.push(max_temp_val);

            // Check for divergence (3 consecutive increases)
            if max_vals.len() >= 3 {
                let len = max_vals.len();
                if max_vals[len - 1] > max_vals[len - 2]
                    && max_vals[len - 2] > max_vals[len - 3] {
                    // Diverging
                    return Ok((psi, false));
                }
            }
        }
    }

    // Max iterations reached
    Ok((psi, false))
}

/// Find the smallest eigenvalue (most negative) using bisection search
///
/// Uses the convergence behavior of domirank_iterative to identify
/// when the system becomes unstable, which indicates we've crossed
/// the smallest eigenvalue threshold.
///
/// # Arguments
/// * `adj_list` - Graph adjacency structure
/// * `degrees` - Node degrees
/// * `max_depth` - Maximum bisection iterations
/// * `dt` - Time step for iterative solver
/// * `epsilon` - Convergence tolerance
/// * `max_iter` - Max iterations for each test
/// * `patience` - Patience for convergence checks
///
/// # Returns
/// Smallest (most negative) eigenvalue
#[inline]
fn find_smallest_eigenvalue(
    adj_list: &FxHashMap<u64, Vec<u64>>,
    degrees: &FxHashMap<u64, f64>,
    max_depth: usize,
    dt: f64,
    epsilon: f64,
    max_iter: usize,
    patience: usize,
) -> PyResult<f64> {
    // Find maximum degree for initial bounds
    let max_degree = degrees.values().cloned().fold(0.0, f64::max);

    if max_degree == 0.0 {
        // Isolated nodes only
        return Ok(-1.0);
    }

    let mut min_val = 0.0;
    let mut max_val = 1.0;
    let mut x = (min_val + max_val) / max_degree;

    for _ in 0..max_depth {
        if max_val - min_val < epsilon {
            break;
        }

        // Test if this sigma causes convergence
        let (_, converged) = domirank_iterative(
            adj_list,
            degrees,
            x,
            dt,
            epsilon,
            max_iter,
            patience,
        )?;

        if converged {
            min_val = x;
        } else {
            max_val = (x + max_val) / 2.0;
        }

        x = (min_val + max_val) / 2.0;
    }

    let final_val = (max_val + min_val) / 2.0;
    Ok(-1.0 / final_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domirank_empty_graph() {
        let graph = Graph::new(false, false);
        let result = domirank(&graph, None, 0.95, 0.1, 1e-5, 1000, 10, 50);
        assert!(result.is_ok());
        let (scores, sigma, converged) = result.unwrap();
        assert_eq!(scores.len(), 0);
        assert!(converged);
    }

    #[test]
    fn test_domirank_single_node() {
        let mut graph = Graph::new(false, false);
        graph.add_node("A", None, None).unwrap();

        let result = domirank(&graph, Some(0.5), 0.95, 0.1, 1e-5, 1000, 10, 50);
        assert!(result.is_ok());
        let (scores, _, converged) = result.unwrap();
        assert_eq!(scores.len(), 1);
        assert!(converged);
    }

    #[test]
    fn test_domirank_parameter_validation() {
        let mut graph = Graph::new(false, false);
        graph.add_node("A", None, None).unwrap();

        // Negative alpha
        assert!(domirank(&graph, None, -0.5, 0.1, 1e-5, 1000, 10, 50).is_err());

        // Invalid dt
        assert!(domirank(&graph, None, 0.95, 1.5, 1e-5, 1000, 10, 50).is_err());
        assert!(domirank(&graph, None, 0.95, -0.1, 1e-5, 1000, 10, 50).is_err());

        // Negative epsilon
        assert!(domirank(&graph, None, 0.95, 0.1, -1e-5, 1000, 10, 50).is_err());

        // Invalid patience/max_iter ratio
        assert!(domirank(&graph, None, 0.95, 0.1, 1e-5, 10, 10, 50).is_err());
    }
}
