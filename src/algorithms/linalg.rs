//! Graph matrices (adjacency, incidence, Laplacian, normalized and directed Laplacian,
//! modularity, Bethe Hessian), their spectra, spectral partitioning (algebraic connectivity,
//! Fiedler vector, spectral ordering and bisection), and the Kemeny constant.
//!
//! Matrices are dense, with rows and columns in node-ID order. Semantics follow NetworkX: an
//! undirected self-loop puts its weight once on the adjacency diagonal (so it cancels out of the
//! Laplacian), parallel edges add up, and directed Laplacians use out-degrees. `weighted`
//! selects each edge's `weight` (1.0 if missing) instead of 1.

use crate::algorithms::sparse_eigen::CsrMatrix;
use crate::graph::{normalize_eigenvector_sign, spectral_laplacian, spectral_symmetric, Graph};
use nalgebra as na;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

fn err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn weight_of(e: &crate::graph::Edge, weighted: bool) -> f64 {
    if weighted {
        e.weight.unwrap_or(1.0)
    } else {
        1.0
    }
}

fn to_rows(m: &na::DMatrix<f64>) -> Vec<Vec<f64>> {
    // `+ 0.0` turns negative zeros (from negating empty entries) into plain zeros
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)] + 0.0).collect())
        .collect()
}

fn dense_adjacency(graph: &Graph, idx: &FxHashMap<u64, usize>, weighted: bool) -> na::DMatrix<f64> {
    let n = idx.len();
    let mut a = na::DMatrix::<f64>::zeros(n, n);
    let undirected = !graph.is_directed();
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        let w = weight_of(e, weighted);
        a[(s, t)] += w;
        if undirected && s != t {
            a[(t, s)] += w;
        }
    }
    a
}

fn laplacian_from(a: &na::DMatrix<f64>, normalized: bool) -> na::DMatrix<f64> {
    let n = a.nrows();
    let deg: Vec<f64> = (0..n).map(|i| a.row(i).sum()).collect();
    let mut l = -a.clone();
    for i in 0..n {
        l[(i, i)] += deg[i];
    }
    if normalized {
        let s: Vec<f64> = deg
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();
        for i in 0..n {
            for j in 0..n {
                l[(i, j)] *= s[i] * s[j];
            }
        }
    }
    l
}

/// Adjacency matrix and its node order.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn adjacency_matrix(graph: &Graph, weighted: bool) -> PyResult<(Vec<Vec<f64>>, Vec<u64>)> {
    let (ids, idx) = sorted_ids(graph);
    Ok((to_rows(&dense_adjacency(graph, &idx, weighted)), ids))
}

/// Incidence matrix (nodes x edges), with node and edge orders. Unoriented: both endpoints get
/// the weight; oriented: -w at the source, +w at the target. Self-loops give zero columns.
#[pyfunction]
#[pyo3(signature = (graph, oriented=false, weighted=false))]
pub fn incidence_matrix(
    graph: &Graph,
    oriented: bool,
    weighted: bool,
) -> PyResult<(Vec<Vec<f64>>, Vec<u64>, Vec<u64>)> {
    let (ids, idx) = sorted_ids(graph);
    let mut edges: Vec<&crate::graph::Edge> = graph.get_edges_internal().iter().collect();
    edges.sort_unstable_by_key(|e| e.id);
    let mut m = vec![vec![0.0; edges.len()]; ids.len()];
    for (j, e) in edges.iter().enumerate() {
        if e.source == e.target {
            continue;
        }
        let w = weight_of(e, weighted);
        let (s, t) = (idx[&e.source], idx[&e.target]);
        m[s][j] = if oriented { -w } else { w };
        m[t][j] = w;
    }
    Ok((m, ids, edges.iter().map(|e| e.id).collect()))
}

/// Laplacian D - A (out-degrees on directed graphs), or the symmetric normalized Laplacian
/// D^-1/2 (D - A) D^-1/2.
#[pyfunction]
#[pyo3(signature = (graph, normalized=false, weighted=false))]
pub fn laplacian_matrix(
    graph: &Graph,
    normalized: bool,
    weighted: bool,
) -> PyResult<(Vec<Vec<f64>>, Vec<u64>)> {
    let (ids, idx) = sorted_ids(graph);
    let a = dense_adjacency(graph, &idx, weighted);
    Ok((to_rows(&laplacian_from(&a, normalized)), ids))
}

/// Strongly connected?
fn strongly_connected(adj: &[Vec<usize>], radj: &[Vec<usize>]) -> bool {
    let n = adj.len();
    if n == 0 {
        return true;
    }
    for lists in [adj, radj] {
        let mut seen = vec![false; n];
        let mut q = VecDeque::from([0usize]);
        seen[0] = true;
        let mut count = 1;
        while let Some(v) = q.pop_front() {
            for &w in &lists[v] {
                if !seen[w] {
                    seen[w] = true;
                    count += 1;
                    q.push_back(w);
                }
            }
        }
        if count != n {
            return false;
        }
    }
    true
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// Period of a strongly connected digraph (gcd of level differences along edges).
fn is_aperiodic(adj: &[Vec<usize>]) -> bool {
    let n = adj.len();
    let mut level = vec![-1i64; n];
    level[0] = 0;
    let mut q = VecDeque::from([0usize]);
    let mut g = 0i64;
    while let Some(v) = q.pop_front() {
        for &w in &adj[v] {
            if level[w] < 0 {
                level[w] = level[v] + 1;
                q.push_back(w);
            } else {
                g = gcd(g, level[v] + 1 - level[w]);
            }
        }
    }
    g == 1
}

/// Chung's directed Laplacian: I - (Φ^½ P Φ^-½ + Φ^-½ Pᵀ Φ^½) / 2, where P is the transition
/// matrix of a random walk ("random"), a lazy walk ("lazy", half the time staying put) or a
/// PageRank walk with teleport ("pagerank", damping `alpha`), and Φ its stationary
/// distribution. With `walk_type=None` the walk is chosen as in NetworkX: random if the graph
/// is strongly connected and aperiodic, lazy if strongly connected but periodic, else PageRank.
#[pyfunction]
#[pyo3(signature = (graph, walk_type=None, alpha=0.95, weighted=false))]
pub fn directed_laplacian_matrix(
    graph: &Graph,
    walk_type: Option<&str>,
    alpha: f64,
    weighted: bool,
) -> PyResult<(Vec<Vec<f64>>, Vec<u64>, String)> {
    if !graph.is_directed() {
        return Err(err("directed_laplacian_matrix() requires a directed graph"));
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n == 0 {
        return Ok((Vec::new(), ids, "random".to_string()));
    }
    let a = dense_adjacency(graph, &idx, weighted);
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut radj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if a[(i, j)] != 0.0 {
                adj[i].push(j);
                radj[j].push(i);
            }
        }
    }
    let walk = match walk_type {
        Some(w @ ("random" | "lazy" | "pagerank")) => w.to_string(),
        Some(other) => {
            return Err(err(format!(
                "walk_type must be 'random', 'lazy' or 'pagerank', got '{}'",
                other
            )))
        }
        None => {
            if strongly_connected(&adj, &radj) {
                if is_aperiodic(&adj) { "random" } else { "lazy" }.to_string()
            } else {
                "pagerank".to_string()
            }
        }
    };
    let rowsum: Vec<f64> = (0..n).map(|i| a.row(i).sum()).collect();
    let mut p = na::DMatrix::<f64>::zeros(n, n);
    match walk.as_str() {
        "random" | "lazy" => {
            if rowsum.contains(&0.0) {
                return Err(err(
                    "The random and lazy walks need every node to have an out-edge",
                ));
            }
            for i in 0..n {
                for j in 0..n {
                    p[(i, j)] = a[(i, j)] / rowsum[i];
                }
            }
            if walk == "lazy" {
                p = (p + na::DMatrix::<f64>::identity(n, n)) / 2.0;
            }
        }
        _ => {
            if !(0.0 < alpha && alpha < 1.0) {
                return Err(err("alpha must be between 0 and 1"));
            }
            for i in 0..n {
                for j in 0..n {
                    let row = if rowsum[i] == 0.0 {
                        1.0 / n as f64
                    } else {
                        a[(i, j)] / rowsum[i]
                    };
                    p[(i, j)] = alpha * row + (1.0 - alpha) / n as f64;
                }
            }
        }
    }
    // Stationary distribution by power iteration on the lazy chain (I + P) / 2, which has the
    // same stationary vector and converges even when P is periodic
    let lazy = (p.clone() + na::DMatrix::<f64>::identity(n, n)) / 2.0;
    let mut phi = na::DVector::<f64>::from_element(n, 1.0 / n as f64);
    for _ in 0..100_000 {
        let next = lazy.transpose() * &phi;
        let next = &next / next.sum();
        let delta = (&next - &phi).abs().max();
        phi = next;
        if delta < 1e-15 {
            break;
        }
    }
    let sq: Vec<f64> = phi.iter().map(|v| v.abs().sqrt()).collect();
    let mut q = na::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            q[(i, j)] = if sq[j] > 0.0 {
                sq[i] * p[(i, j)] / sq[j]
            } else {
                0.0
            };
        }
    }
    let l = na::DMatrix::<f64>::identity(n, n) - (&q + q.transpose()) / 2.0;
    Ok((to_rows(&l), ids, walk))
}

fn modularity_dense(
    graph: &Graph,
    idx: &FxHashMap<u64, usize>,
    weighted: bool,
) -> na::DMatrix<f64> {
    let a = dense_adjacency(graph, idx, weighted);
    let n = a.nrows();
    let mut b = a.clone();
    if graph.is_directed() {
        let kout: Vec<f64> = (0..n).map(|i| a.row(i).sum()).collect();
        let kin: Vec<f64> = (0..n).map(|j| a.column(j).sum()).collect();
        let m: f64 = kin.iter().sum();
        if m > 0.0 {
            for i in 0..n {
                for j in 0..n {
                    b[(i, j)] -= kout[i] * kin[j] / m;
                }
            }
        }
    } else {
        let k: Vec<f64> = (0..n).map(|i| a.row(i).sum()).collect();
        let two_m: f64 = k.iter().sum();
        if two_m > 0.0 {
            for i in 0..n {
                for j in 0..n {
                    b[(i, j)] -= k[i] * k[j] / two_m;
                }
            }
        }
    }
    b
}

/// Modularity matrix B = A - k kᵀ / 2m (undirected) or A - k_out k_inᵀ / m (directed).
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn modularity_matrix(graph: &Graph, weighted: bool) -> PyResult<(Vec<Vec<f64>>, Vec<u64>)> {
    let (ids, idx) = sorted_ids(graph);
    Ok((to_rows(&modularity_dense(graph, &idx, weighted)), ids))
}

fn bethe_dense(
    graph: &Graph,
    idx: &FxHashMap<u64, usize>,
    r: Option<f64>,
    weighted: bool,
) -> PyResult<(na::DMatrix<f64>, f64)> {
    if graph.is_directed() {
        return Err(err("bethe_hessian_matrix() requires an undirected graph"));
    }
    let n = idx.len();
    let r = match r {
        Some(r) => r,
        None => {
            // NetworkX: mean of squared (unweighted) degrees over mean degree, minus 1
            let mut deg = vec![0.0f64; n];
            for e in graph.get_edges_internal() {
                deg[idx[&e.source]] += 1.0;
                deg[idx[&e.target]] += 1.0;
            }
            let s: f64 = deg.iter().sum();
            if s == 0.0 {
                return Err(err("The default r needs at least one edge; pass r="));
            }
            deg.iter().map(|d| d * d).sum::<f64>() / s - 1.0
        }
    };
    let a = dense_adjacency(graph, idx, weighted);
    let mut h = -a.clone() * r;
    for i in 0..n {
        h[(i, i)] += r * r - 1.0 + a.row(i).sum();
    }
    Ok((h, r))
}

/// Bethe Hessian H(r) = (r² - 1) I - r A + D (Saade, Krzakala & Zdeborová 2014).
#[pyfunction]
#[pyo3(signature = (graph, r=None, weighted=false))]
pub fn bethe_hessian_matrix(
    graph: &Graph,
    r: Option<f64>,
    weighted: bool,
) -> PyResult<(Vec<Vec<f64>>, Vec<u64>, f64)> {
    let (ids, idx) = sorted_ids(graph);
    let (h, r) = bethe_dense(graph, &idx, r, weighted)?;
    Ok((to_rows(&h), ids, r))
}

/// Eigenvalues of the modularity matrix (ascending); complex on directed graphs.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn modularity_spectrum(py: Python, graph: &Graph, weighted: bool) -> PyResult<PyObject> {
    let (ids, idx) = sorted_ids(graph);
    if ids.is_empty() {
        // nalgebra's decompositions abort on a 0 × 0 matrix
        return Ok(pyo3::types::PyList::empty(py).into());
    }
    let b = modularity_dense(graph, &idx, weighted);
    if graph.is_directed() {
        // nalgebra's `complex_eigenvalues()` never converges on some of these matrices
        // (an edgeless digraph's zero matrix among them), so use the EISPACK-style solver
        let mut vals = crate::algorithms::eigen::general_eigenvalues(&b)
            .ok_or_else(|| err("Eigenvalue computation failed to converge for directed graph"))?;
        vals.sort_by(|x, y| x.re.total_cmp(&y.re).then(x.im.total_cmp(&y.im)));
        let list = pyo3::types::PyList::empty(py);
        for v in vals {
            list.append(pyo3::types::PyComplex::from_doubles(py, v.re, v.im))?;
        }
        Ok(list.into())
    } else {
        let mut vals: Vec<f64> = na::SymmetricEigen::new(b)
            .eigenvalues
            .iter()
            .copied()
            .collect();
        vals.sort_by(|x, y| x.total_cmp(y));
        Ok(vals.into_pyobject(py)?.into())
    }
}

/// Eigenvalues of the Bethe Hessian (ascending).
#[pyfunction]
#[pyo3(signature = (graph, r=None, weighted=false))]
pub fn bethe_hessian_spectrum(graph: &Graph, r: Option<f64>, weighted: bool) -> PyResult<Vec<f64>> {
    let (ids, idx) = sorted_ids(graph);
    let (h, _) = bethe_dense(graph, &idx, r, weighted)?;
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut vals: Vec<f64> = na::SymmetricEigen::new(h)
        .eigenvalues
        .iter()
        .copied()
        .collect();
    vals.sort_by(|x, y| x.total_cmp(y));
    Ok(vals)
}

/// Leading-eigenvector modularity bipartition (Newman 2006), as NetworkX's
/// `spectral_modularity_bipartition`: nodes split by the sign of their entry in the
/// eigenvector of the modularity matrix's largest eigenvalue. The sign is fixed so the
/// lowest-ID node with a nonzero entry is negative; negative entries form the first side and
/// the rest (zeros included) the second. Returns `(first, second, largest eigenvalue)`.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn spectral_modularity_bipartition(
    graph: &Graph,
    weighted: bool,
) -> PyResult<(Vec<u64>, Vec<u64>, f64)> {
    if graph.is_directed() {
        return Err(err(
            "spectral_modularity_bipartition() requires an undirected graph",
        ));
    }
    let (ids, idx) = sorted_ids(graph);
    if ids.is_empty() {
        return Ok((Vec::new(), Vec::new(), 0.0));
    }
    let b = modularity_dense(graph, &idx, weighted);
    if b.iter().all(|&x| x == 0.0) {
        // No edges: every eigenvector is as good as any other, so nothing is split off
        return Ok((Vec::new(), ids, 0.0));
    }
    let eig = na::SymmetricEigen::new(b);
    let lead = (0..ids.len()).fold(0, |best, j| {
        if eig.eigenvalues[j] > eig.eigenvalues[best] {
            j
        } else {
            best
        }
    });
    let v = eig.eigenvectors.column(lead);
    let tol = 1e-12;
    let sign = match v.iter().find(|x| x.abs() > tol) {
        Some(&x) if x > 0.0 => -1.0,
        _ => 1.0,
    };
    let (mut first, mut second) = (Vec::new(), Vec::new());
    for (i, &id) in ids.iter().enumerate() {
        if sign * v[i] < -tol {
            first.push(id);
        } else {
            second.push(id);
        }
    }
    Ok((first, second, eig.eigenvalues[lead]))
}

/// The magnetic Laplacian (Fanuel et al. 2018) as NetworkX builds it: with W' = (W + Wᵀ) / 2
/// for the summed edge weights W (self-loops left out), H_jk = W'_jk e^{2πiqδ_jk}, where δ_jk
/// is +1 if only j → k is an edge, -1 if only k → j is, and 0 if both are; then L = D - H
/// for the row sums D of |H|, or I - D^{-1/2} H D^{-1/2} (zero rows for isolated nodes, with
/// 1 on the diagonal) when normalized. Hermitian, so its eigenvalues are real.
fn magnetic_dense(
    graph: &Graph,
    idx: &FxHashMap<u64, usize>,
    q: f64,
    normalized: bool,
    weighted: bool,
) -> PyResult<na::DMatrix<na::Complex<f64>>> {
    if !graph.is_directed() {
        return Err(err("The magnetic Laplacian requires a directed graph"));
    }
    if !(0.0..=0.5).contains(&q) {
        return Err(err(format!("q must be between 0 and 0.5, got {q}")));
    }
    let n = idx.len();
    let mut w: FxHashMap<(usize, usize), f64> = FxHashMap::default();
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        if s != t {
            *w.entry((s, t)).or_insert(0.0) += weight_of(e, weighted);
        }
    }
    let phase = 2.0 * std::f64::consts::PI * q;
    let mut h = na::DMatrix::<na::Complex<f64>>::zeros(n, n);
    for (&(s, t), &wst) in &w {
        let back = w.get(&(t, s));
        let sym = 0.5 * (wst + back.copied().unwrap_or(0.0));
        let delta = if back.is_some() { 0.0 } else { 1.0 };
        h[(s, t)] = na::Complex::from_polar(sym, phase * delta);
        if back.is_none() {
            h[(t, s)] = na::Complex::from_polar(sym, -phase);
        }
    }
    let deg: Vec<f64> = (0..n)
        .map(|i| h.row(i).iter().map(|z| z.norm()).sum())
        .collect();
    if normalized {
        let inv: Vec<f64> = deg
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] *= -(inv[i] * inv[j]);
            }
            h[(i, i)] += 1.0;
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                h[(i, j)] = -h[(i, j)];
            }
            h[(i, i)] += deg[i];
        }
    }
    Ok(h)
}

/// The magnetic Laplacian as rows of Python complex numbers, with the node IDs.
#[pyfunction]
#[pyo3(signature = (graph, q=0.25, normalized=false, weighted=false))]
pub fn magnetic_laplacian_matrix(
    py: Python,
    graph: &Graph,
    q: f64,
    normalized: bool,
    weighted: bool,
) -> PyResult<(PyObject, Vec<u64>)> {
    let (ids, idx) = sorted_ids(graph);
    let l = magnetic_dense(graph, &idx, q, normalized, weighted)?;
    let rows = pyo3::types::PyList::empty(py);
    for i in 0..ids.len() {
        let row = pyo3::types::PyList::empty(py);
        for j in 0..ids.len() {
            let z = l[(i, j)];
            row.append(pyo3::types::PyComplex::from_doubles(py, z.re, z.im))?;
        }
        rows.append(row)?;
    }
    Ok((rows.into(), ids))
}

/// Eigenvalues of the magnetic Laplacian (real, as it is Hermitian), in increasing order.
#[pyfunction]
#[pyo3(signature = (graph, q=0.25, normalized=false, weighted=false))]
pub fn magnetic_spectrum(
    graph: &Graph,
    q: f64,
    normalized: bool,
    weighted: bool,
) -> PyResult<Vec<f64>> {
    let (ids, idx) = sorted_ids(graph);
    let l = magnetic_dense(graph, &idx, q, normalized, weighted)?;
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut vals: Vec<f64> = na::SymmetricEigen::new(l)
        .eigenvalues
        .iter()
        .copied()
        .collect();
    vals.sort_by(|x, y| x.total_cmp(y));
    Ok(vals)
}

/// Connected components of an undirected graph as index lists (in order of smallest member).
fn components(graph: &Graph, idx: &FxHashMap<u64, usize>) -> Vec<Vec<usize>> {
    let n = idx.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        adj[s].push(t);
        adj[t].push(s);
    }
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut comp = vec![root];
        let mut q = VecDeque::from([root]);
        while let Some(v) = q.pop_front() {
            for &w in &adj[v] {
                if !seen[w] {
                    seen[w] = true;
                    comp.push(w);
                    q.push_back(w);
                }
            }
        }
        comp.sort_unstable();
        out.push(comp);
    }
    out
}

/// (second smallest eigenvalue, its eigenvector) of the (normalized) Laplacian restricted to
/// the given nodes, using the shared sparse/dense symmetric solver.
fn fiedler_pair(
    graph: &Graph,
    idx: &FxHashMap<u64, usize>,
    nodes: &[usize],
    weighted: bool,
    normalized: bool,
) -> PyResult<(f64, Vec<f64>)> {
    let local: FxHashMap<usize, usize> = nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut triplets = Vec::new();
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        if let (Some(&i), Some(&j)) = (local.get(&s), local.get(&t)) {
            let w = weight_of(e, weighted);
            triplets.push((i, j, w));
            if i != j {
                triplets.push((j, i, w));
            }
        }
    }
    let adj = CsrMatrix::from_triplets(nodes.len(), triplets);
    let lap = spectral_laplacian(&adj, normalized);
    let (pairs, _) = spectral_symmetric(&lap, "smallest", 2, false)?;
    let (value, mut vector) = pairs
        .into_iter()
        .nth(1)
        .ok_or_else(|| err("Need at least two nodes for a Fiedler vector"))?;
    normalize_eigenvector_sign(&mut vector);
    Ok((value, vector))
}

fn require_undirected(graph: &Graph, what: &str) -> PyResult<()> {
    if graph.is_directed() {
        return Err(err(format!("{} requires an undirected graph", what)));
    }
    Ok(())
}

fn check_weights(graph: &Graph, weighted: bool) -> PyResult<()> {
    if weighted
        && graph
            .get_edges_internal()
            .iter()
            .any(|e| e.weight.is_some_and(|w| w < 0.0))
    {
        return Err(err("Edge weights must be non-negative"));
    }
    Ok(())
}

/// Algebraic connectivity: the second smallest Laplacian eigenvalue (0 if disconnected).
#[pyfunction]
#[pyo3(signature = (graph, normalized=false, weighted=false))]
pub fn algebraic_connectivity(graph: &Graph, normalized: bool, weighted: bool) -> PyResult<f64> {
    require_undirected(graph, "algebraic_connectivity()")?;
    check_weights(graph, weighted)?;
    let (ids, idx) = sorted_ids(graph);
    if ids.len() < 2 {
        return Err(err("algebraic_connectivity() needs at least two nodes"));
    }
    if components(graph, &idx).len() > 1 {
        return Ok(0.0);
    }
    let all: Vec<usize> = (0..ids.len()).collect();
    Ok(fiedler_pair(graph, &idx, &all, weighted, normalized)?
        .0
        .max(0.0))
}

/// Fiedler vector (eigenvector of the second smallest Laplacian eigenvalue), unit length, sign
/// fixed so the first non-negligible entry (in node-ID order) is positive.
#[pyfunction]
#[pyo3(signature = (graph, normalized=false, weighted=false))]
pub fn fiedler_vector(
    graph: &Graph,
    normalized: bool,
    weighted: bool,
) -> PyResult<FxHashMap<u64, f64>> {
    require_undirected(graph, "fiedler_vector()")?;
    check_weights(graph, weighted)?;
    let (ids, idx) = sorted_ids(graph);
    if ids.len() < 2 {
        return Err(err("fiedler_vector() needs at least two nodes"));
    }
    if components(graph, &idx).len() > 1 {
        return Err(err("fiedler_vector() requires a connected graph"));
    }
    let all: Vec<usize> = (0..ids.len()).collect();
    let (_, v) = fiedler_pair(graph, &idx, &all, weighted, normalized)?;
    Ok(ids.into_iter().zip(v).collect())
}

/// Spectral ordering: each connected component (in order of its smallest node ID) sorted by its
/// Fiedler vector; components of one or two nodes keep ID order.
#[pyfunction]
#[pyo3(signature = (graph, normalized=false, weighted=false))]
pub fn spectral_ordering(graph: &Graph, normalized: bool, weighted: bool) -> PyResult<Vec<u64>> {
    require_undirected(graph, "spectral_ordering()")?;
    check_weights(graph, weighted)?;
    let (ids, idx) = sorted_ids(graph);
    if ids.is_empty() {
        return Err(err("spectral_ordering() needs a non-empty graph"));
    }
    let mut order = Vec::with_capacity(ids.len());
    for comp in components(graph, &idx) {
        if comp.len() <= 2 {
            order.extend(comp.iter().map(|&v| ids[v]));
            continue;
        }
        let (_, f) = fiedler_pair(graph, &idx, &comp, weighted, normalized)?;
        let mut keyed: Vec<(f64, usize)> = f.into_iter().zip(comp.iter().copied()).collect();
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        order.extend(keyed.into_iter().map(|(_, v)| ids[v]));
    }
    Ok(order)
}

/// Kemeny constant of a connected undirected graph: the expected number of steps a random walk
/// takes to reach a target drawn from the stationary distribution, Σ_{i≥2} 1 / (1 - λ_i) over
/// the eigenvalues of the normalized adjacency D^-½ A D^-½.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn kemeny_constant(graph: &Graph, weighted: bool) -> PyResult<f64> {
    require_undirected(graph, "kemeny_constant()")?;
    check_weights(graph, weighted)?;
    let (ids, idx) = sorted_ids(graph);
    if ids.is_empty() {
        return Err(err("kemeny_constant() needs at least one node"));
    }
    if components(graph, &idx).len() > 1 {
        return Err(err("kemeny_constant() requires a connected graph"));
    }
    let a = dense_adjacency(graph, &idx, weighted);
    let n = a.nrows();
    let s: Vec<f64> = (0..n)
        .map(|i| {
            let d = a.row(i).sum();
            if d > 0.0 {
                1.0 / d.sqrt()
            } else {
                0.0
            }
        })
        .collect();
    let mut h = a.clone();
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] *= s[i] * s[j];
        }
    }
    let mut vals: Vec<f64> = na::SymmetricEigen::new(h)
        .eigenvalues
        .iter()
        .copied()
        .collect();
    vals.sort_by(|x, y| x.total_cmp(y));
    vals.pop();
    Ok(vals.iter().map(|l| 1.0 / (1.0 - l)).sum())
}

/// Diagonals of `cosh(A)` and `exp(A)` for the (symmetric) adjacency matrix, from its
/// eigendecomposition: `(node IDs, cosh diagonal, exp diagonal)`. Used for spectral
/// bipartivity (Estrada and Rodríguez-Velázquez, 2005).
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn exp_cosh_diagonals(
    graph: &Graph,
    weighted: bool,
) -> PyResult<(Vec<u64>, Vec<f64>, Vec<f64>)> {
    require_undirected(graph, "spectral_bipartivity()")?;
    let (ids, idx) = sorted_ids(graph);
    let a = dense_adjacency(graph, &idx, weighted);
    let n = ids.len();
    if n == 0 {
        return Ok((ids, Vec::new(), Vec::new()));
    }
    let eig = na::SymmetricEigen::new(a);
    let mut cosh = vec![0.0; n];
    let mut exp = vec![0.0; n];
    for k in 0..n {
        let lam = eig.eigenvalues[k];
        let (e, c) = (lam.exp(), lam.cosh());
        for i in 0..n {
            let v = eig.eigenvectors[(i, k)];
            exp[i] += v * v * e;
            cosh[i] += v * v * c;
        }
    }
    Ok((ids, cosh, exp))
}
