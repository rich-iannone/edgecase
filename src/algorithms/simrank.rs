//! SimRank* similarity with PJsim closed-form solver.
//!
//! SimRank* (Yu et al., VLDBJ 2019) fixes the "zero-similarity" problem of the
//! original SimRank by considering asymmetric in-link paths. PJsim (Nijhara
//! et al., SIGMOD 2026) reformulates SimRank* as a Lyapunov equation and
//! solves it exactly via Bartels-Stewart (real Schur decomposition), avoiding
//! the iteration/tolerance trade-off of iterative solvers.

use crate::graph::Graph;
use nalgebra::DMatrix;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Compute SimRank* similarity between all node pairs.
///
/// Returns a nested dict: `{node_a: {node_b: similarity}}`.
///
/// Two solution methods are available:
/// - `"closed_form"` (default): exact PJsim solver via the Bartels-Stewart
///   algorithm (real Schur decomposition of the Lyapunov equation).
///   O(n³) time, O(n²) space. No tolerance parameter needed.
/// - `"iterative"`: fixed-point iteration of the SimRank* bilinear
///   recursion. O(K·n³) time where K = max_iterations.
///
/// # The SimRank* equation
///
/// S = (C/2) · (Q·S + S·Qᵀ) + (1-C)·I
///
/// where Q is the column-stochastic transition matrix built from in-neighbors
/// and C is the importance/decay factor.
///
/// # The PJsim reformulation
///
/// Rearranging gives the continuous Lyapunov equation:
///   X·S + S·Xᵀ = 2(1-C)·I     where X = I - C·Q   …(*)
///
/// (*) is solved exactly by the Bartels-Stewart algorithm.
/// The raw solution is then normalized: S_norm[i,j] = S[i,j] / sqrt(S[i,i]*S[j,j]).
#[pyfunction]
#[pyo3(signature = (graph, importance_factor=0.8, max_iterations=100, tolerance=1e-6, method="closed_form"))]
pub fn simrank_star_similarity(
    graph: &Graph,
    importance_factor: f64,
    max_iterations: usize,
    tolerance: f64,
    method: &str,
) -> PyResult<FxHashMap<u64, FxHashMap<u64, f64>>> {
    let c = importance_factor;
    if !(0.0..1.0).contains(&c) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "importance_factor must be in [0, 1)",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(FxHashMap::default());
    }

    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    // Build column-stochastic transition matrix Q from in-neighbors.
    // Q[i][j] = 1/|In(j)| if i is an in-neighbor of j, else 0.
    // For undirected graphs, in-neighbors = neighbors.
    let q = build_transition_matrix(graph, &node_ids, &node_to_idx);

    let sim = match method {
        "closed_form" => solve_pjsim(&q, c, n),
        "iterative" => solve_iterative(&q, c, n, max_iterations, tolerance),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'closed_form' or 'iterative'",
            ))
        }
    };

    // Convert DMatrix to nested HashMap
    let mut result: FxHashMap<u64, FxHashMap<u64, f64>> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    for (i, &id_a) in node_ids.iter().enumerate() {
        let mut inner: FxHashMap<u64, f64> =
            FxHashMap::with_capacity_and_hasher(n, Default::default());
        for (j, &id_b) in node_ids.iter().enumerate() {
            let val = sim[(i, j)];
            if val.abs() > 1e-15 {
                inner.insert(id_b, val);
            }
        }
        result.insert(id_a, inner);
    }

    Ok(result)
}

/// Compute SimRank* similarity between a specific pair of nodes.
#[pyfunction]
#[pyo3(signature = (graph, source, target, importance_factor=0.8, max_iterations=100, tolerance=1e-6, method="closed_form"))]
pub fn simrank_star_similarity_pair(
    graph: &Graph,
    source: u64,
    target: u64,
    importance_factor: f64,
    max_iterations: usize,
    tolerance: f64,
    method: &str,
) -> PyResult<f64> {
    let node_ids = graph.get_node_ids();
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    let src_idx = node_to_idx.get(&source).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Node {} not found", source))
    })?;
    let tgt_idx = node_to_idx.get(&target).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Node {} not found", target))
    })?;

    if source == target {
        return Ok(1.0);
    }

    let c = importance_factor;
    if !(0.0..1.0).contains(&c) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "importance_factor must be in [0, 1)",
        ));
    }

    let n = node_ids.len();
    let q = build_transition_matrix(graph, &node_ids, &node_to_idx);

    let sim = match method {
        "closed_form" => solve_pjsim(&q, c, n),
        "iterative" => solve_iterative(&q, c, n, max_iterations, tolerance),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'closed_form' or 'iterative'",
            ))
        }
    };

    Ok(sim[(*src_idx, *tgt_idx)])
}

/// Build column-stochastic transition matrix Q.
///
/// Q[i][j] = 1/|In(j)| if node i is an in-neighbor of node j, else 0.
fn build_transition_matrix(
    graph: &Graph,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
) -> DMatrix<f64> {
    let n = node_ids.len();
    let mut q = DMatrix::<f64>::zeros(n, n);

    if graph.is_directed() {
        // For directed graphs: in-neighbors of j are nodes with edges -> j
        // Build in-neighbor lists from the edge list
        let mut in_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for edge in graph.get_edges_internal() {
            if let (Some(&src_idx), Some(&tgt_idx)) = (
                node_to_idx.get(&edge.source),
                node_to_idx.get(&edge.target),
            ) {
                in_neighbors[tgt_idx].push(src_idx);
            }
        }

        for j in 0..n {
            let in_deg = in_neighbors[j].len();
            if in_deg > 0 {
                let weight = 1.0 / in_deg as f64;
                for &i in &in_neighbors[j] {
                    q[(i, j)] = weight;
                }
            }
        }
    } else {
        // For undirected: in-neighbors = neighbors
        for (j, &node_j) in node_ids.iter().enumerate() {
            if let Ok(neighbors) = graph.neighbors(node_j) {
                let deg = neighbors.len();
                if deg > 0 {
                    let weight = 1.0 / deg as f64;
                    for nbr in neighbors {
                        if let Some(&i) = node_to_idx.get(&nbr) {
                            q[(i, j)] = weight;
                        }
                    }
                }
            }
        }
    }

    q
}

/// PJsim: solve the Lyapunov equation X·S + S·Xᵀ = 2(1-C)·I exactly
/// using the Bartels-Stewart algorithm (real Schur decomposition).
///
/// X = I - C·Q
///
/// Steps:
/// 1. Compute real Schur decomposition: X = U·T·Uᵀ
/// 2. Transform RHS: F = Uᵀ · [2(1-C)·I] · U
/// 3. Solve T·Y + Y·Tᵀ = F  (quasi-triangular Lyapunov) by back-substitution
/// 4. Recover S = U·Y·Uᵀ
/// 5. Normalize: S_norm[i,j] = S[i,j] / sqrt(S[i,i] * S[j,j])
fn solve_pjsim(q: &DMatrix<f64>, c: f64, n: usize) -> DMatrix<f64> {
    if n == 0 {
        return DMatrix::<f64>::zeros(0, 0);
    }
    if n == 1 {
        return DMatrix::from_element(1, 1, 1.0);
    }

    let identity = DMatrix::<f64>::identity(n, n);

    // X = I - C * Q
    let x = &identity - q * c;

    // Real Schur decomposition: X = U * T * U^T
    let schur = nalgebra::Schur::new(x);
    let (u, t) = schur.unpack();

    // RHS in Schur basis: F = U^T * [2(1-C)*I] * U
    let rhs_coeff = 2.0 * (1.0 - c);
    let f = &u.transpose() * (rhs_coeff * &identity) * &u;

    // Solve the quasi-triangular Lyapunov equation: T*Y + Y*T^T = F
    let y = solve_quasi_triangular_lyapunov(&t, &f);

    // S = U * Y * U^T
    let s = &u * y * u.transpose();

    normalize_similarity(s, n)
}

/// Solve T*Y + Y*T^T = F where T is quasi-upper-triangular (real Schur form).
///
/// Column j equation: (T + T[j,j]*I) * Y[:,j] = F[:,j] - sum_{k>j} Y[:,k] * T[j,k]
/// Since T is upper-triangular, T[j,k]=0 for k<j, so corrections come only from
/// columns k>j. We solve from the last column backwards.
fn solve_quasi_triangular_lyapunov(t: &DMatrix<f64>, f: &DMatrix<f64>) -> DMatrix<f64> {
    let n = t.nrows();
    let mut y = DMatrix::<f64>::zeros(n, n);

    // Identify block structure (scan forward, then process backward)
    let mut block_starts: Vec<(usize, usize)> = Vec::new(); // (start_col, block_size)
    let mut j = 0;
    while j < n {
        if j + 1 < n && t[(j + 1, j)].abs() > 1e-12 {
            block_starts.push((j, 2));
            j += 2;
        } else {
            block_starts.push((j, 1));
            j += 1;
        }
    }

    // Solve backwards (last block first)
    for &(col, bs) in block_starts.iter().rev() {
        if bs == 1 {
            let t_jj = t[(col, col)];

            let mut rhs = f.column(col).clone_owned();
            // Subtract contributions from already-solved columns k > col
            for k in (col + 1)..n {
                let t_jk = t[(col, k)];
                if t_jk.abs() > 1e-15 {
                    for i in 0..n {
                        rhs[i] -= y[(i, k)] * t_jk;
                    }
                }
            }

            let a = t + DMatrix::<f64>::identity(n, n) * t_jj;
            if let Some(sol) = nalgebra::LU::new(a).solve(&rhs) {
                for i in 0..n {
                    y[(i, col)] = sol[i];
                }
            }
        } else {
            // 2x2 block at (col, col+1)
            let mut rhs0 = f.column(col).clone_owned();
            let mut rhs1 = f.column(col + 1).clone_owned();

            // Corrections from already-solved columns k > col+1
            for k in (col + 2)..n {
                let t_j0_k = t[(col, k)];
                let t_j1_k = t[(col + 1, k)];
                if t_j0_k.abs() > 1e-15 {
                    for i in 0..n {
                        rhs0[i] -= y[(i, k)] * t_j0_k;
                    }
                }
                if t_j1_k.abs() > 1e-15 {
                    for i in 0..n {
                        rhs1[i] -= y[(i, k)] * t_j1_k;
                    }
                }
            }

            let t00 = t[(col, col)];
            let t01 = t[(col, col + 1)];
            let t10 = t[(col + 1, col)];
            let t11 = t[(col + 1, col + 1)];

            let mut a_big = DMatrix::<f64>::zeros(2 * n, 2 * n);
            for i in 0..n {
                for jj in 0..n {
                    a_big[(i, jj)] = t[(i, jj)] + if i == jj { t00 } else { 0.0 };
                }
            }
            for i in 0..n {
                a_big[(i, n + i)] = t10;
            }
            for i in 0..n {
                a_big[(n + i, i)] = t01;
            }
            for i in 0..n {
                for jj in 0..n {
                    a_big[(n + i, n + jj)] =
                        t[(i, jj)] + if i == jj { t11 } else { 0.0 };
                }
            }

            let mut rhs_big = nalgebra::DVector::<f64>::zeros(2 * n);
            for i in 0..n {
                rhs_big[i] = rhs0[i];
                rhs_big[n + i] = rhs1[i];
            }

            if let Some(sol) = nalgebra::LU::new(a_big).solve(&rhs_big) {
                for i in 0..n {
                    y[(i, col)] = sol[i];
                    y[(i, col + 1)] = sol[n + i];
                }
            }
        }
    }

    y
}

/// Iterative SimRank* solver: fixed-point iteration of
/// S_{k+1} = (C/2) * (Q * S_k + S_k * Q^T) + (1-C) * I
/// then normalize: S_norm[i,j] = S[i,j] / sqrt(S[i,i] * S[j,j])
fn solve_iterative(
    q: &DMatrix<f64>,
    c: f64,
    n: usize,
    max_iter: usize,
    tol: f64,
) -> DMatrix<f64> {
    if n == 0 {
        return DMatrix::<f64>::zeros(0, 0);
    }
    if n == 1 {
        return DMatrix::from_element(1, 1, 1.0);
    }

    let identity = DMatrix::<f64>::identity(n, n);
    let qt = q.transpose();
    let half_c = c / 2.0;
    let one_minus_c = 1.0 - c;

    let mut s = identity.clone();

    for _ in 0..max_iter {
        let s_new = (q * &s + &s * &qt) * half_c + &identity * one_minus_c;

        let diff = &s_new - &s;
        let norm: f64 = diff.iter().map(|x| x * x).sum::<f64>().sqrt();

        s = s_new;

        if norm < tol {
            break;
        }
    }

    normalize_similarity(s, n)
}

/// Normalize raw similarity matrix: S_norm[i,j] = S[i,j] / sqrt(S[i,i] * S[j,j])
fn normalize_similarity(s: DMatrix<f64>, n: usize) -> DMatrix<f64> {
    let diag: Vec<f64> = (0..n).map(|i| s[(i, i)].sqrt()).collect();

    let mut result = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        result[(i, i)] = 1.0;
        for j in (i + 1)..n {
            let denom = diag[i] * diag[j];
            let val = if denom > 1e-15 {
                ((s[(i, j)] + s[(j, i)]) / 2.0 / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            result[(i, j)] = val;
            result[(j, i)] = val;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    fn node(id: u64) -> Node {
        Node {
            id,
            label: None,
            attributes: None,
        }
    }

    fn edge(src: u64, tgt: u64) -> Edge {
        Edge {
            id: 0,
            source: src,
            target: tgt,
            weight: None,
            attributes: None,
        }
    }

    fn make_simple_directed() -> Graph {
        // Classic SimRank example: university/univ, ProfA/ProfB -> univ
        //   1 -> 3
        //   2 -> 3
        let mut g = Graph::new_rust(true, false);
        g.add_node_rust(node(1)).unwrap();
        g.add_node_rust(node(2)).unwrap();
        g.add_node_rust(node(3)).unwrap();
        g.add_edge_rust(edge(1, 3)).unwrap();
        g.add_edge_rust(edge(2, 3)).unwrap();
        g
    }

    fn make_diamond_directed() -> Graph {
        //   1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
        let mut g = Graph::new_rust(true, false);
        for i in 1..=4 {
            g.add_node_rust(node(i)).unwrap();
        }
        g.add_edge_rust(edge(1, 2)).unwrap();
        g.add_edge_rust(edge(1, 3)).unwrap();
        g.add_edge_rust(edge(2, 4)).unwrap();
        g.add_edge_rust(edge(3, 4)).unwrap();
        g
    }

    fn make_triangle_undirected() -> Graph {
        let mut g = Graph::new_rust(false, false);
        for i in 1..=3 {
            g.add_node_rust(node(i)).unwrap();
        }
        g.add_edge_rust(edge(1, 2)).unwrap();
        g.add_edge_rust(edge(2, 3)).unwrap();
        g.add_edge_rust(edge(3, 1)).unwrap();
        g
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::new_rust(false, false);
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut g = Graph::new_rust(false, false);
        g.add_node_rust(node(1)).unwrap();
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        assert_eq!(result.len(), 1);
        let inner = result.get(&1).unwrap();
        assert!((inner.get(&1).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_self_similarity_is_one() {
        let g = make_diamond_directed();
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        for (&nid, inner) in &result {
            let self_sim = inner.get(&nid).copied().unwrap_or(0.0);
            assert!(
                (self_sim - 1.0).abs() < 1e-10,
                "Self-similarity of node {} is {}, expected 1.0",
                nid,
                self_sim
            );
        }
    }

    #[test]
    fn test_symmetric_nodes_have_positive_similarity() {
        // Nodes 1 and 2 both point to node 3 -- they should have positive similarity
        let g = make_simple_directed();
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        let sim_12 = result
            .get(&1)
            .and_then(|m| m.get(&2))
            .copied()
            .unwrap_or(0.0);
        assert!(
            sim_12 > 0.0,
            "Nodes 1,2 that share an out-neighbor should have positive similarity, got {}",
            sim_12
        );
    }

    #[test]
    fn test_diamond_simrank_star_nonzero() {
        // In diamond graph, nodes 2 and 3 share both in-neighbor (1) and
        // out-neighbor (4). SimRank* should give them positive similarity.
        // Original SimRank would also catch this, but SimRank* should be
        // positive for more pairs.
        let g = make_diamond_directed();
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        let sim_23 = result
            .get(&2)
            .and_then(|m| m.get(&3))
            .copied()
            .unwrap_or(0.0);
        assert!(
            sim_23 > 0.0,
            "Nodes 2,3 in diamond should have positive SimRank*, got {}",
            sim_23
        );
    }

    #[test]
    fn test_methods_agree() {
        let g = make_diamond_directed();
        let closed = simrank_star_similarity(&g, 0.8, 500, 1e-8, "closed_form").unwrap();
        let iterative = simrank_star_similarity(&g, 0.8, 500, 1e-8, "iterative").unwrap();

        for (&nid_a, inner_c) in &closed {
            for (&nid_b, &val_c) in inner_c {
                let val_i = iterative
                    .get(&nid_a)
                    .and_then(|m| m.get(&nid_b))
                    .copied()
                    .unwrap_or(0.0);
                assert!(
                    (val_c - val_i).abs() < 1e-4,
                    "closed_form vs iterative disagree for ({},{}): {} vs {}",
                    nid_a,
                    nid_b,
                    val_c,
                    val_i
                );
            }
        }
    }

    #[test]
    fn test_undirected_triangle() {
        let g = make_triangle_undirected();
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        // All pairs should have equal similarity in a triangle
        let sim_12 = result
            .get(&1)
            .and_then(|m| m.get(&2))
            .copied()
            .unwrap_or(0.0);
        let sim_23 = result
            .get(&2)
            .and_then(|m| m.get(&3))
            .copied()
            .unwrap_or(0.0);
        let sim_13 = result
            .get(&1)
            .and_then(|m| m.get(&3))
            .copied()
            .unwrap_or(0.0);
        assert!(
            (sim_12 - sim_23).abs() < 1e-6 && (sim_23 - sim_13).abs() < 1e-6,
            "Triangle nodes should have equal pairwise similarity: {} {} {}",
            sim_12,
            sim_23,
            sim_13
        );
    }

    #[test]
    fn test_pair_function() {
        let g = make_diamond_directed();
        let full = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        let pair = simrank_star_similarity_pair(&g, 2, 3, 0.8, 100, 1e-6, "closed_form").unwrap();
        let expected = full.get(&2).and_then(|m| m.get(&3)).copied().unwrap_or(0.0);
        assert!(
            (pair - expected).abs() < 1e-10,
            "Pair function should match full matrix: {} vs {}",
            pair,
            expected
        );
    }

    #[test]
    fn test_similarity_in_unit_range() {
        let g = make_diamond_directed();
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        for inner in result.values() {
            for &val in inner.values() {
                assert!(
                    (0.0..=1.0 + 1e-10).contains(&val),
                    "Similarity should be in [0,1], got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_invalid_importance_factor() {
        let g = Graph::new_rust(false, false);
        assert!(simrank_star_similarity(&g, 1.0, 100, 1e-6, "closed_form").is_err());
        assert!(simrank_star_similarity(&g, -0.1, 100, 1e-6, "closed_form").is_err());
    }

    #[test]
    fn test_invalid_method() {
        let g = make_simple_directed();
        assert!(simrank_star_similarity(&g, 0.8, 100, 1e-6, "invalid").is_err());
    }
}
