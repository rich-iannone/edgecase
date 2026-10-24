//! SimRank* similarity with PJsim closed-form solver.
//!
//! SimRank* (Yu et al., VLDBJ 2019) fixes the "zero-similarity" problem of the
//! original SimRank by considering asymmetric in-link paths. PJsim (Nijhara
//! et al., SIGMOD 2026) reformulates SimRank* as a Lyapunov equation and
//! solves it exactly via Bartels-Stewart (real Schur decomposition), avoiding
//! the iteration/tolerance trade-off of iterative solvers.
//!
//! Also Panther (Zhang et al., KDD 2015): similarity as the share of random paths that two
//! nodes both appear on, with the weighted random walks it samples.

use crate::graph::Graph;
use nalgebra::DMatrix;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
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
/// where Q is the row-stochastic backward transition matrix (row a averages over the
/// in-neighbors of a) and C is the importance/decay factor.
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

    // Build row-stochastic backward transition matrix Q from in-neighbors.
    // Q[j][i] = 1/|In(j)| if i is an in-neighbor of j, else 0.
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

/// Build the row-stochastic backward transition matrix Q of Yu et al.
///
/// Q[j][i] = 1/|In(j)| if node i is an in-neighbor of node j, else 0, so row j averages
/// over j's in-neighbors and (Q·S)[a][b] = mean over x in In(a) of S[x][b]. In-neighbors are
/// distinct (parallel edges count once); for undirected graphs they are the neighbors.
fn build_transition_matrix(
    graph: &Graph,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
) -> DMatrix<f64> {
    let n = node_ids.len();
    let mut in_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

    if graph.is_directed() {
        // For directed graphs: in-neighbors of j are nodes with edges -> j
        for edge in graph.get_edges_internal() {
            if let (Some(&src_idx), Some(&tgt_idx)) = (
                node_to_idx.get(&edge.source),
                node_to_idx.get(&edge.target),
            ) {
                in_neighbors[tgt_idx].push(src_idx);
            }
        }
    } else {
        // For undirected: in-neighbors = neighbors
        for (j, &node_j) in node_ids.iter().enumerate() {
            if let Ok(neighbors) = graph.neighbors(node_j) {
                in_neighbors[j].extend(neighbors.iter().filter_map(|nbr| node_to_idx.get(nbr)));
            }
        }
    }

    let mut q = DMatrix::<f64>::zeros(n, n);
    for (j, nbrs) in in_neighbors.iter_mut().enumerate() {
        nbrs.sort_unstable();
        nbrs.dedup();
        if !nbrs.is_empty() {
            let weight = 1.0 / nbrs.len() as f64;
            for &i in nbrs.iter() {
                q[(j, i)] = weight;
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
            // Column col of Y*T^T picks up Y[:,col+1]*T[col,col+1] (t01), and column col+1
            // picks up Y[:,col]*T[col+1,col] (t10)
            for i in 0..n {
                a_big[(i, n + i)] = t01;
            }
            for i in 0..n {
                a_big[(n + i, i)] = t10;
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

// ---------------------------------------------------------------------------------------------
// Panther (random-path similarity)
// ---------------------------------------------------------------------------------------------

/// Sorted node IDs and, per node, its out-arcs with cumulative transition weights (an arc per
/// edge, both ways in an undirected graph, a self-loop once), as NetworkX's
/// `to_numpy_array()` rows.
#[allow(clippy::type_complexity)]
fn walk_table(
    graph: &Graph,
    weighted: bool,
) -> PyResult<(Vec<u64>, FxHashMap<u64, usize>, Vec<Vec<(usize, f64)>>)> {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx: FxHashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut edges: Vec<_> = graph.get_edges_internal().iter().collect();
    edges.sort_unstable_by_key(|e| e.id);
    let mut out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ids.len()];
    for e in edges {
        let w = if weighted {
            e.weight.unwrap_or(1.0)
        } else {
            1.0
        };
        if !w.is_finite() || w < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Random paths need non-negative finite weights; edge {} has weight {w}",
                e.id
            )));
        }
        let (a, b) = (idx[&e.source], idx[&e.target]);
        out[a].push((b, w));
        if !graph.is_directed() && a != b {
            out[b].push((a, w));
        }
    }
    for arcs in &mut out {
        let mut total = 0.0;
        for arc in arcs.iter_mut() {
            total += arc.1;
            arc.1 = total;
        }
    }
    Ok((ids, idx, out))
}

/// `sample_size` random walks of `path_length` steps (so `path_length + 1` nodes each), as
/// node indices. Each starts at `start`, or at a uniformly random node with out-edges of
/// positive weight (as in NetworkX #8578), and steps to an out-neighbor with probability
/// proportional to the edge weight.
fn walks(
    ids: &[u64],
    out: &[Vec<(usize, f64)>],
    sample_size: usize,
    path_length: usize,
    start: Option<usize>,
    rng: &mut ChaCha8Rng,
) -> PyResult<Vec<Vec<usize>>> {
    let starts: Vec<usize> = if start.is_some() {
        Vec::new()
    } else {
        (0..ids.len()).filter(|&i| out[i].last().is_some_and(|a| a.1 > 0.0)).collect()
    };
    if start.is_none() && starts.is_empty() && sample_size > 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No node has out-edges of positive weight, so no random path can start",
        ));
    }
    let mut paths = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        let mut u = start.unwrap_or_else(|| starts[rng.gen_range(0..starts.len())]);
        let mut path = Vec::with_capacity(path_length + 1);
        path.push(u);
        for _ in 0..path_length {
            let arcs = &out[u];
            let total = arcs.last().map_or(0.0, |a| a.1);
            if total <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "A random path reached node {}, which has no out-edges of positive weight",
                    ids[u]
                )));
            }
            let r = rng.gen::<f64>() * total;
            let k = arcs.partition_point(|a| a.1 <= r).min(arcs.len() - 1);
            u = arcs[k].0;
            path.push(u);
        }
        paths.push(path);
    }
    Ok(paths)
}

fn rng_from(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

/// Random walks for Panther, as lists of node IDs (see `walks`).
#[pyfunction]
#[pyo3(signature = (graph, sample_size, path_length=5, weighted=true, seed=None, source=None))]
pub fn generate_random_paths(
    graph: &Graph,
    sample_size: usize,
    path_length: usize,
    weighted: bool,
    seed: Option<u64>,
    source: Option<u64>,
) -> PyResult<Vec<Vec<u64>>> {
    let (ids, idx, out) = walk_table(graph, weighted)?;
    if ids.is_empty() {
        return if sample_size == 0 {
            Ok(Vec::new())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "The graph has no nodes",
            ))
        };
    }
    let start = match source {
        Some(s) => Some(*idx.get(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Source node {s} does not exist"))
        })?),
        None => None,
    };
    let mut rng = rng_from(seed);
    let paths = walks(&ids, &out, sample_size, path_length, start, &mut rng)?;
    Ok(paths
        .into_iter()
        .map(|p| p.into_iter().map(|i| ids[i]).collect())
        .collect())
}

/// The random paths shared by both Panther measures, on the graph without its isolated nodes
/// (which no walk can reach anyway): node IDs, the source's index and the paths.
struct PantherSample {
    ids: Vec<u64>,
    src: usize,
    paths: Vec<Vec<usize>>,
}

#[allow(clippy::too_many_arguments)]
fn panther_sample(
    graph: &Graph,
    source: u64,
    path_length: usize,
    c: f64,
    delta: f64,
    eps: Option<f64>,
    weighted: bool,
    seed: Option<u64>,
) -> PyResult<PantherSample> {
    let err = |msg: String| Err(pyo3::exceptions::PyValueError::new_err(msg));
    if path_length < 2 {
        return err(format!("path_length must be at least 2, got {path_length}"));
    }
    if c.is_nan() || c <= 0.0 {
        return err(format!("c must be positive, got {c}"));
    }
    if delta.is_nan() || delta <= 0.0 || delta >= 1.0 {
        return err(format!("delta must be between 0 and 1, got {delta}"));
    }
    if matches!(eps, Some(e) if e.is_nan() || e <= 0.0) {
        return err("eps must be positive".to_string());
    }
    let (ids, idx, out) = walk_table(graph, weighted)?;
    let Some(&s) = idx.get(&source) else {
        return err(format!("Source node {source} does not exist"));
    };
    let n = ids.len();
    let mut isolated = vec![true; n];
    for e in graph.get_edges_internal() {
        isolated[idx[&e.source]] = false;
        isolated[idx[&e.target]] = false;
    }
    if isolated[s] {
        return err(format!(
            "Panther similarity isn't defined for the isolated source node {source}"
        ));
    }

    let kept: Vec<usize> = (0..n).filter(|&i| !isolated[i]).collect();
    let mut new_index = vec![usize::MAX; n];
    for (j, &i) in kept.iter().enumerate() {
        new_index[i] = j;
    }
    let kept_ids: Vec<u64> = kept.iter().map(|&i| ids[i]).collect();
    let kept_out: Vec<Vec<(usize, f64)>> = kept
        .iter()
        .map(|&i| out[i].iter().map(|&(v, w)| (new_index[v], w)).collect())
        .collect();

    let m = graph.get_edges_internal().len() as f64;
    let eps = eps.unwrap_or_else(|| (1.0 / m).sqrt());
    // C(T + 1, 2) pairs of positions on a path of T + 1 nodes (NetworkX #8578)
    let pairs = ((path_length + 1) * path_length / 2) as f64;
    let size = (c / (eps * eps)) * (pairs.log2() + 1.0 + (1.0 / delta).ln());
    let sample_size = size.ceil().max(0.0) as usize;
    if sample_size == 0 {
        return err("The sample size works out to 0 paths; lower eps or raise c".to_string());
    }

    let mut rng = rng_from(seed);
    let paths = walks(
        &kept_ids,
        &kept_out,
        sample_size,
        path_length,
        None,
        &mut rng,
    )?;
    Ok(PantherSample { ids: kept_ids, src: new_index[s], paths })
}

/// Panther similarity (Zhang et al., KDD 2015) of every non-isolated node to `source`: the
/// share of `R` random paths that contain both. Isolated nodes are dropped first, as in
/// NetworkX. `R = ceil((c / eps^2) (log2 C(T + 1, 2) + 1 + ln(1 / delta)))` with `T` the path
/// length and `eps` defaulting to `sqrt(1 / |E|)`. Returns `R` and `(node, score)` for every
/// non-isolated node other than the source, in node ID order.
#[pyfunction]
#[pyo3(signature = (graph, source, path_length=5, c=0.5, delta=0.1, eps=None, weighted=true, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn panther_similarity(
    graph: &Graph,
    source: u64,
    path_length: usize,
    c: f64,
    delta: f64,
    eps: Option<f64>,
    weighted: bool,
    seed: Option<u64>,
) -> PyResult<(usize, Vec<(u64, f64)>)> {
    let sample = panther_sample(graph, source, path_length, c, delta, eps, weighted, seed)?;
    let (n, src, r_total) = (sample.ids.len(), sample.src, sample.paths.len());
    let mut common = vec![0usize; n];
    let mut seen = vec![usize::MAX; n];
    for (r, path) in sample.paths.iter().enumerate() {
        if !path.contains(&src) {
            continue;
        }
        for &v in path {
            if seen[v] != r {
                seen[v] = r;
                common[v] += 1;
            }
        }
    }
    let scores = (0..n)
        .filter(|&j| j != src)
        .map(|j| (sample.ids[j], common[j] as f64 / r_total as f64))
        .collect();
    Ok((r_total, scores))
}

/// Panther++ (Zhang et al., KDD 2015), as NetworkX's `panther_vector_similarity`: each
/// non-isolated node gets a feature vector of its `dimensions` largest Panther scores
/// (against every node, itself included, in descending order), and a node's similarity to
/// the source is `1 / d` for the Euclidean distance `d` between their vectors, with `d`
/// clamped to machine epsilon (NetworkX #8578).
/// Returns `(node, score)` for every non-isolated node other than the source, nearest
/// first (ties by node ID).
#[pyfunction]
#[pyo3(signature = (graph, source, dimensions=10, path_length=5, c=0.5, delta=0.1, eps=None, weighted=true, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn panther_vector_similarity(
    graph: &Graph,
    source: u64,
    dimensions: usize,
    path_length: usize,
    c: f64,
    delta: f64,
    eps: Option<f64>,
    weighted: bool,
    seed: Option<u64>,
) -> PyResult<Vec<(u64, f64)>> {
    let err = |msg: String| Err(pyo3::exceptions::PyValueError::new_err(msg));
    if dimensions == 0 {
        return err("D must be a positive integer".to_string());
    }
    let sample = panther_sample(graph, source, path_length, c, delta, eps, weighted, seed)?;
    let n = sample.ids.len();
    if dimensions > n {
        return err(format!(
            "D={dimensions} asks for more similarity scores than the {n} non-isolated nodes"
        ));
    }
    // Co-occurrence counts of every pair of nodes on a path (a node with itself counts the
    // paths through it), kept sparse
    let mut co: Vec<FxHashMap<usize, u32>> = vec![FxHashMap::default(); n];
    let mut distinct: Vec<usize> = Vec::new();
    for path in &sample.paths {
        distinct.clear();
        distinct.extend(path.iter().copied());
        distinct.sort_unstable();
        distinct.dedup();
        for &a in &distinct {
            for &b in &distinct {
                *co[a].entry(b).or_insert(0) += 1;
            }
        }
    }
    let r_total = sample.paths.len() as f64;
    let theta: Vec<Vec<f64>> = co
        .iter()
        .map(|row| {
            let mut top: Vec<u32> = row.values().copied().collect();
            top.sort_unstable_by(|a, b| b.cmp(a));
            top.resize(dimensions, 0);
            top.into_iter().map(|x| x as f64 / r_total).collect()
        })
        .collect();
    let base = &theta[sample.src];
    let mut out: Vec<(u64, f64, f64)> = (0..n)
        .filter(|&j| j != sample.src)
        .map(|j| {
            let d = theta[j].iter().zip(base).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
            (sample.ids[j], d, 1.0 / d.max(f64::EPSILON))
        })
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
    Ok(out.into_iter().map(|(id, _, score)| (id, score)).collect())
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
        // Nodes 1 and 2 are both linked from node 3 -- they should have positive similarity
        let mut g = Graph::new_rust(true, false);
        g.add_node_rust(node(1)).unwrap();
        g.add_node_rust(node(2)).unwrap();
        g.add_node_rust(node(3)).unwrap();
        g.add_edge_rust(edge(3, 1)).unwrap();
        g.add_edge_rust(edge(3, 2)).unwrap();
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        let sim_12 = result
            .get(&1)
            .and_then(|m| m.get(&2))
            .copied()
            .unwrap_or(0.0);
        assert!(
            sim_12 > 0.0,
            "Nodes 1,2 that share an in-neighbor should have positive similarity, got {}",
            sim_12
        );

        // Sharing only an out-neighbor (1 -> 3, 2 -> 3) gives no in-link evidence
        let g = make_simple_directed();
        let result = simrank_star_similarity(&g, 0.8, 100, 1e-6, "closed_form").unwrap();
        assert_eq!(result.get(&1).and_then(|m| m.get(&2)).copied().unwrap_or(0.0), 0.0);
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
