//! Numerical node layouts: Fruchterman-Reingold (spring), Kamada-Kawai, spectral, ForceAtlas2
//! and ARF.
//!
//! Each function returns raw `(id, x, y)` rows in node-ID order; the Python `layout_*`
//! methods rescale and center them. Edge direction is ignored: an edge `u -> v` pulls `u`
//! and `v` together like an undirected edge, parallel edges (and reciprocal arcs) add their
//! weights, and self-loops are skipped. `weighted` selects each edge's `weight` (1.0 if
//! missing) instead of 1.

use crate::algorithms::generators::make_rng;
use crate::algorithms::sparse_eigen::CsrMatrix;
use crate::graph::{spectral_laplacian, spectral_symmetric, Graph};
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{BinaryHeap, HashMap, VecDeque};

/// Node count from which per-iteration force sums run in parallel
const PARALLEL_MIN_NODES: usize = 128;

type Positions = Vec<(u64, f64, f64)>;

fn err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

/// Undirected neighbor lists with summed edge weights, self-loops dropped
fn neighbor_weights(
    graph: &Graph,
    idx: &FxHashMap<u64, usize>,
    weighted: bool,
) -> Vec<Vec<(usize, f64)>> {
    let mut maps: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); idx.len()];
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        if s == t {
            continue;
        }
        let w = if weighted {
            e.weight.unwrap_or(1.0)
        } else {
            1.0
        };
        *maps[s].entry(t).or_insert(0.0) += w;
        *maps[t].entry(s).or_insert(0.0) += w;
    }
    maps.into_iter()
        .map(|m| {
            let mut v: Vec<(usize, f64)> = m.into_iter().collect();
            v.sort_unstable_by_key(|&(j, _)| j);
            v
        })
        .collect()
}

/// Evaluate `f` for every node index, in parallel on larger graphs
fn per_node<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if n >= PARALLEL_MIN_NODES {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// Starting positions: given ones where present, uniform in `[0, extent)^2` elsewhere
fn initial_positions(
    ids: &[u64],
    initial: &Option<HashMap<u64, (f64, f64)>>,
    seed: Option<u64>,
) -> Vec<[f64; 2]> {
    let mut rng = make_rng(seed);
    let extent = initial
        .as_ref()
        .map(|m| {
            m.values()
                .fold(0.0_f64, |acc, &(x, y)| acc.max(x.abs()).max(y.abs()))
        })
        .filter(|&e| e > 0.0)
        .unwrap_or(1.0);
    ids.iter()
        .map(|id| {
            let random = [rng.gen::<f64>() * extent, rng.gen::<f64>() * extent];
            match initial.as_ref().and_then(|m| m.get(id)) {
                Some(&(x, y)) => [x, y],
                None => random,
            }
        })
        .collect()
}

fn check_initial(initial: &Option<HashMap<u64, (f64, f64)>>) -> PyResult<()> {
    if let Some(m) = initial {
        if m.values().any(|&(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err(err("Initial positions must be finite numbers"));
        }
    }
    Ok(())
}

fn rows(ids: &[u64], pos: &[[f64; 2]]) -> Positions {
    ids.iter()
        .zip(pos)
        .map(|(&id, p)| (id, p[0], p[1]))
        .collect()
}

fn check_weights(graph: &Graph, weighted: bool, what: &str) -> PyResult<()> {
    if weighted
        && graph
            .get_edges_internal()
            .iter()
            .any(|e| e.weight.is_some_and(|w| !w.is_finite() || w < 0.0))
    {
        return Err(err(format!(
            "{} requires finite, non-negative edge weights",
            what
        )));
    }
    Ok(())
}

/// Fruchterman-Reingold force-directed positions (NetworkX's dense `spring_layout()` scheme).
///
/// Every pair repels with `k^2 / d` and adjacent nodes attract with `w * d^2 / k`; each node
/// moves at most the current temperature, which cools linearly from a tenth of the starting
/// domain. Stops early once the mean step falls below `threshold`. Nodes in `fixed` keep
/// their starting positions.
#[pyfunction]
#[pyo3(signature = (graph, k=None, iterations=50, threshold=1e-4, weighted=false, initial=None, fixed=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn spring_layout(
    graph: &Graph,
    k: Option<f64>,
    iterations: usize,
    threshold: f64,
    weighted: bool,
    initial: Option<HashMap<u64, (f64, f64)>>,
    fixed: Option<Vec<u64>>,
    seed: Option<u64>,
) -> PyResult<Positions> {
    if let Some(k) = k {
        if !(k.is_finite() && k > 0.0) {
            return Err(err("k must be a positive number"));
        }
    }
    if !(threshold.is_finite() && threshold >= 0.0) {
        return Err(err("threshold must be a non-negative number"));
    }
    check_weights(graph, weighted, "layout_spring()")?;
    check_initial(&initial)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut is_fixed = vec![false; n];
    if let Some(fixed) = &fixed {
        for id in fixed {
            let i = *idx
                .get(id)
                .ok_or_else(|| err(format!("Fixed node {} not found", id)))?;
            if !initial.as_ref().is_some_and(|m| m.contains_key(id)) {
                return Err(err(format!("Fixed node {} has no initial position", id)));
            }
            is_fixed[i] = true;
        }
    }
    let mut pos = initial_positions(&ids, &initial, seed);
    if n <= 1 {
        return Ok(rows(&ids, &pos));
    }

    let k = k.unwrap_or_else(|| {
        let extent = match (&fixed, &initial) {
            (Some(_), Some(m)) => m
                .values()
                .fold(0.0_f64, |acc, &(x, y)| acc.max(x.abs()).max(y.abs())),
            _ => 1.0,
        };
        let extent = if extent > 0.0 { extent } else { 1.0 };
        extent / (n as f64).sqrt()
    });
    let adj = neighbor_weights(graph, &idx, weighted);

    let span = |axis: usize| {
        let (lo, hi) = pos
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p[axis]), hi.max(p[axis]))
            });
        hi - lo
    };
    let mut t = 0.1 * span(0).max(span(1));
    let dt = t / (iterations as f64 + 1.0);

    for _ in 0..iterations {
        let current = &pos;
        let adj = &adj;
        let steps: Vec<[f64; 2]> = per_node(n, |i| {
            let [xi, yi] = current[i];
            let mut disp = [0.0, 0.0];
            for (j, pj) in current.iter().enumerate() {
                if j == i {
                    continue;
                }
                let (dx, dy) = (xi - pj[0], yi - pj[1]);
                let d = (dx * dx + dy * dy).sqrt().max(0.01);
                let f = k * k / (d * d);
                disp[0] += dx * f;
                disp[1] += dy * f;
            }
            for &(j, w) in &adj[i] {
                let (dx, dy) = (xi - current[j][0], yi - current[j][1]);
                let d = (dx * dx + dy * dy).sqrt().max(0.01);
                let f = w * d / k;
                disp[0] -= dx * f;
                disp[1] -= dy * f;
            }
            let length = (disp[0] * disp[0] + disp[1] * disp[1]).sqrt();
            let length = if length < 0.01 { 0.1 } else { length };
            [disp[0] * t / length, disp[1] * t / length]
        });

        let mut moved = 0.0;
        for (i, step) in steps.iter().enumerate() {
            if is_fixed[i] {
                continue;
            }
            pos[i][0] += step[0];
            pos[i][1] += step[1];
            moved += step[0] * step[0] + step[1] * step[1];
        }
        t -= dt;
        if moved.sqrt() / (n as f64) < threshold {
            break;
        }
    }
    Ok(rows(&ids, &pos))
}

/// Shortest-path distances on the undirected view (BFS hops, or Dijkstra over weights)
fn all_pairs_distances(adj: &[Vec<(usize, f64)>], weighted: bool) -> Vec<Vec<f64>> {
    let n = adj.len();
    per_node(n, |src| {
        let mut dist = vec![f64::INFINITY; n];
        dist[src] = 0.0;
        if weighted {
            #[derive(PartialEq)]
            struct Item(f64, usize);
            impl Eq for Item {}
            impl PartialOrd for Item {
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(other))
                }
            }
            impl Ord for Item {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
                }
            }
            let mut heap = BinaryHeap::from([Item(0.0, src)]);
            while let Some(Item(d, u)) = heap.pop() {
                if d > dist[u] {
                    continue;
                }
                for &(v, w) in &adj[u] {
                    let nd = d + w;
                    if nd < dist[v] {
                        dist[v] = nd;
                        heap.push(Item(nd, v));
                    }
                }
            }
        } else {
            let mut queue = VecDeque::from([src]);
            while let Some(u) = queue.pop_front() {
                for &(v, _) in &adj[u] {
                    if dist[v].is_infinite() {
                        dist[v] = dist[u] + 1.0;
                        queue.push_back(v);
                    }
                }
            }
        }
        dist
    })
}

/// Kamada-Kawai stress `sum_{i<j} (|x_i - x_j| - d_ij)^2 / d_ij^2` and its gradient, for
/// positions flattened as `[x0, y0, x1, y1, ...]`
fn stress_and_gradient(x: &[f64], dist: &[Vec<f64>]) -> (f64, Vec<f64>) {
    let n = dist.len();
    let rows = per_node(n, |i| {
        let (mut e, mut gx, mut gy) = (0.0, 0.0, 0.0);
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = dist[i][j];
            let (dx, dy) = (x[2 * i] - x[2 * j], x[2 * i + 1] - x[2 * j + 1]);
            let r = (dx * dx + dy * dy).sqrt();
            let gap = r - d;
            e += gap * gap / (d * d);
            if r > 1e-12 {
                let f = 2.0 * gap / (d * d * r);
                gx += f * dx;
                gy += f * dy;
            }
        }
        (e, gx, gy)
    });
    let mut grad = Vec::with_capacity(2 * n);
    let mut energy = 0.0;
    for (e, gx, gy) in rows {
        // Each pair was counted from both ends
        energy += 0.5 * e;
        grad.push(gx);
        grad.push(gy);
    }
    (energy, grad)
}

/// One stress-majorization sweep (Gansner, Koren & North 2004): each node in turn moves to the
/// minimizer of the stress majorant with the others held fixed, which never raises the stress
fn majorize_sweep(x: &mut [f64], dist: &[Vec<f64>], rng: &mut impl Rng) {
    let n = dist.len();
    for i in 0..n {
        let (mut sx, mut sy, mut total) = (0.0, 0.0, 0.0);
        for (j, &d) in dist[i].iter().enumerate() {
            if j == i {
                continue;
            }
            let w = 1.0 / (d * d);
            let (mut dx, mut dy) = (x[2 * i] - x[2 * j], x[2 * i + 1] - x[2 * j + 1]);
            let mut norm = (dx * dx + dy * dy).sqrt();
            if norm < 1e-12 {
                dx = rng.gen::<f64>() - 0.5;
                dy = rng.gen::<f64>() - 0.5;
                norm = (dx * dx + dy * dy).sqrt();
            }
            sx += w * (x[2 * j] + d * dx / norm);
            sy += w * (x[2 * j + 1] + d * dy / norm);
            total += w;
        }
        x[2 * i] = sx / total;
        x[2 * i + 1] = sy / total;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// Minimize `f` from `x` with limited-memory BFGS (10 correction pairs) and a backtracking
/// line search. Stops after `max_iter` iterations, once an iteration lowers `f` by at most
/// the fraction `tol`, or once the gradient vanishes.
fn lbfgs(
    x: &mut Vec<f64>,
    f: &mut f64,
    g: &mut Vec<f64>,
    max_iter: usize,
    tol: f64,
    eval: impl Fn(&[f64]) -> (f64, Vec<f64>),
) {
    const MEMORY: usize = 10;
    let mut history: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::new();
    for _ in 0..max_iter {
        if g.iter().all(|v| v.abs() <= 1e-12) {
            break;
        }
        // Two-loop recursion for the search direction -H g
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let a = rho * dot(s, &q);
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
            alphas.push(a);
        }
        let h0 = history
            .back()
            .map_or(1.0 / dot(g, g).sqrt().max(1.0), |(s, y, _)| {
                dot(s, y) / dot(y, y)
            });
        q.iter_mut().for_each(|qi| *qi *= h0);
        for ((s, y, rho), a) in history.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(y, &q);
            q.iter_mut().zip(s).for_each(|(qi, si)| *qi += (a - b) * si);
        }
        let mut dir: Vec<f64> = q.iter().map(|v| -v).collect();
        let mut slope = dot(g, &dir);
        if slope >= 0.0 {
            history.clear();
            dir = g.iter().map(|v| -v / dot(g, g).sqrt().max(1.0)).collect();
            slope = dot(g, &dir);
        }

        // Backtrack until the Armijo condition holds
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..60 {
            let trial: Vec<f64> = x.iter().zip(&dir).map(|(xi, di)| xi + step * di).collect();
            let (ft, gt) = eval(&trial);
            if ft.is_finite() && ft <= *f + 1e-4 * step * slope {
                accepted = Some((trial, ft, gt));
                break;
            }
            step *= 0.5;
        }
        let Some((trial, ft, gt)) = accepted else {
            break;
        };
        let s: Vec<f64> = trial.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = gt.iter().zip(g.iter()).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 * dot(&y, &y).sqrt() * dot(&s, &s).sqrt() {
            if history.len() == MEMORY {
                history.pop_front();
            }
            history.push_back((s, y, 1.0 / sy));
        }
        let decrease = *f - ft;
        *x = trial;
        *g = gt;
        *f = ft;
        if decrease <= tol * f.max(f64::MIN_POSITIVE) {
            break;
        }
    }
}

/// Kamada-Kawai positions, found by stress majorization followed by L-BFGS.
///
/// Minimizes `sum_{i<j} (|x_i - x_j| - d_ij)^2 / d_ij^2`, where `d_ij` is the shortest-path
/// distance, the objective NetworkX's `kamada_kawai_layout()` hands to L-BFGS. Starts on a
/// circle unless `initial` positions are given. Pairs in different components are placed one unit
/// further apart than the largest finite distance.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false, initial=None, max_iter=500, tol=1e-6))]
pub fn kamada_kawai_layout(
    graph: &Graph,
    weighted: bool,
    initial: Option<HashMap<u64, (f64, f64)>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Positions> {
    if weighted
        && graph
            .get_edges_internal()
            .iter()
            .any(|e| e.source != e.target && e.weight.is_some_and(|w| !w.is_finite() || w <= 0.0))
    {
        return Err(err(
            "layout_kamada_kawai() requires finite, positive edge weights",
        ));
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(err("tol must be a non-negative number"));
    }
    check_initial(&initial)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut pos: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            match initial.as_ref().and_then(|m| m.get(&ids[i])) {
                Some(&(x, y)) => [x, y],
                None => [theta.cos(), theta.sin()],
            }
        })
        .collect();
    if n <= 1 {
        return Ok(rows(&ids, &pos));
    }

    // Parallel edges keep their lightest weight as the distance
    let mut adj: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    for e in graph.get_edges_internal() {
        let (s, t) = (idx[&e.source], idx[&e.target]);
        if s == t {
            continue;
        }
        let w = if weighted {
            e.weight.unwrap_or(1.0)
        } else {
            1.0
        };
        for (a, b) in [(s, t), (t, s)] {
            let slot = adj[a].entry(b).or_insert(f64::INFINITY);
            *slot = slot.min(w);
        }
    }
    let adj: Vec<Vec<(usize, f64)>> = adj.into_iter().map(|m| m.into_iter().collect()).collect();
    let mut dist = all_pairs_distances(&adj, weighted);
    let far = dist
        .iter()
        .flatten()
        .filter(|d| d.is_finite())
        .fold(0.0_f64, |a, &b| a.max(b))
        + 1.0;
    for row in dist.iter_mut() {
        for d in row.iter_mut() {
            if d.is_infinite() {
                *d = far;
            }
        }
    }

    // Two nodes in the same spot have no direction to move apart along; nudge them
    let mut rng = make_rng(Some(0));
    let mut seen: FxHashMap<(u64, u64), ()> = FxHashMap::default();
    for p in pos.iter_mut() {
        while seen.insert((p[0].to_bits(), p[1].to_bits()), ()).is_some() {
            p[0] += (rng.gen::<f64>() - 0.5) * 1e-3;
            p[1] += (rng.gen::<f64>() - 0.5) * 1e-3;
        }
    }

    // Majorization sweeps get the coarse shape reliably; L-BFGS then converges much faster
    // through the long, flat valleys (a path straightening out) where sweeps crawl
    let mut x: Vec<f64> = pos.iter().flatten().copied().collect();
    let mut energy = stress_and_gradient(&x, &dist).0;
    for _ in 0..max_iter {
        majorize_sweep(&mut x, &dist, &mut rng);
        let current = stress_and_gradient(&x, &dist).0;
        let decrease = energy - current;
        energy = current;
        if decrease <= 1e-3 * energy.max(f64::MIN_POSITIVE) {
            break;
        }
    }
    let mut grad;
    (energy, grad) = stress_and_gradient(&x, &dist);
    lbfgs(&mut x, &mut energy, &mut grad, max_iter, tol, |x| {
        stress_and_gradient(x, &dist)
    });
    let pos: Vec<[f64; 2]> = x.chunks(2).map(|c| [c[0], c[1]]).collect();
    Ok(rows(&ids, &pos))
}

/// Spectral positions: the Laplacian eigenvectors of the second and third smallest
/// eigenvalues, as coordinates (NetworkX's `spectral_layout()`).
///
/// Directed graphs use `A + A^T`. Graphs with fewer than three nodes have no third
/// eigenvector and are placed by the Python wrapper instead.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false))]
pub fn spectral_layout(graph: &Graph, weighted: bool) -> PyResult<Positions> {
    check_weights(graph, weighted, "layout_spectral()")?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n < 3 {
        return Err(err("layout_spectral() needs at least three nodes"));
    }
    let adj = neighbor_weights(graph, &idx, weighted);
    let triplets: Vec<(usize, usize, f64)> = adj
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().map(move |&(j, w)| (i, j, w)))
        .collect();
    let lap = spectral_laplacian(&CsrMatrix::from_triplets(n, triplets), false);
    let (pairs, _) = spectral_symmetric(&lap, "smallest", 3, false)?;
    if pairs.len() < 3 {
        return Err(err("Eigenvalue computation returned too few eigenvectors"));
    }
    let (xs, ys) = (&pairs[1].1, &pairs[2].1);
    Ok(ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, xs[i], ys[i]))
        .collect())
}

/// ForceAtlas2 positions (Jacomy, Venturini, Heymann & Bastian 2014).
///
/// Nodes of mass `m` (degree + 1 unless given) repel with `scaling_ratio * m_i * m_j / d`;
/// edges attract with `w * d` (`w * log(1 + d)` in LinLog mode, divided by the source mass
/// when dissuading hubs); gravity pulls each node toward the centroid with `gravity * m`
/// (`gravity * m * d` when strong). Steps use the paper's adaptive global and per-node speeds,
/// driven by how much each node's force swings between iterations. Giving node sizes turns
/// on overlap prevention: distances are measured between node borders, overlapping nodes
/// repel strongly and stop attracting.
#[pyfunction]
#[pyo3(signature = (graph, max_iter=100, jitter_tolerance=1.0, scaling_ratio=2.0, gravity=1.0, strong_gravity=false, linlog=false, dissuade_hubs=false, weighted=false, mass=None, size=None, initial=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn forceatlas2_layout(
    graph: &Graph,
    max_iter: usize,
    jitter_tolerance: f64,
    scaling_ratio: f64,
    gravity: f64,
    strong_gravity: bool,
    linlog: bool,
    dissuade_hubs: bool,
    weighted: bool,
    mass: Option<HashMap<u64, f64>>,
    size: Option<HashMap<u64, f64>>,
    initial: Option<HashMap<u64, (f64, f64)>>,
    seed: Option<u64>,
) -> PyResult<Positions> {
    for (name, value) in [
        ("jitter_tolerance", jitter_tolerance),
        ("scaling_ratio", scaling_ratio),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(err(format!("{} must be a positive number", name)));
        }
    }
    if !(gravity.is_finite() && gravity >= 0.0) {
        return Err(err("gravity must be a non-negative number"));
    }
    check_weights(graph, weighted, "layout_forceatlas2()")?;
    check_initial(&initial)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut pos = initial_positions(&ids, &initial, seed);
    if n <= 1 {
        return Ok(rows(&ids, &pos));
    }

    let mut degree = vec![0.0; n];
    for e in graph.get_edges_internal() {
        degree[idx[&e.source]] += 1.0;
        degree[idx[&e.target]] += 1.0;
    }
    let masses: Vec<f64> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            mass.as_ref()
                .and_then(|m| m.get(id).copied())
                .unwrap_or(degree[i] + 1.0)
        })
        .collect();
    if masses.iter().any(|m| !(m.is_finite() && *m > 0.0)) {
        return Err(err("Node masses must be positive numbers"));
    }
    let adjust_sizes = size.is_some();
    let sizes: Vec<f64> = ids
        .iter()
        .map(|id| {
            size.as_ref()
                .and_then(|m| m.get(id).copied())
                .unwrap_or(1.0)
        })
        .collect();
    if sizes.iter().any(|s| !(s.is_finite() && *s >= 0.0)) {
        return Err(err("Node sizes must be non-negative numbers"));
    }
    let adj = neighbor_weights(graph, &idx, weighted);

    let mut old_force = vec![[0.0, 0.0]; n];
    let (mut speed, mut speed_efficiency) = (1.0_f64, 1.0_f64);
    for _ in 0..max_iter {
        let (cx, cy) = pos
            .iter()
            .fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
        let (cx, cy) = (cx / n as f64, cy / n as f64);
        let current = &pos;
        let (adj, masses, sizes) = (&adj, &masses, &sizes);
        let force: Vec<[f64; 2]> = per_node(n, |i| {
            let [xi, yi] = current[i];
            let mut f = [0.0, 0.0];
            for (j, pj) in current.iter().enumerate() {
                if j == i {
                    continue;
                }
                let (dx, dy) = (xi - pj[0], yi - pj[1]);
                let mut d = (dx * dx + dy * dy).sqrt();
                let factor = if adjust_sizes {
                    d -= sizes[i] + sizes[j];
                    if d > 0.0 {
                        scaling_ratio * masses[i] * masses[j] / (d * d)
                    } else if d < 0.0 {
                        100.0 * scaling_ratio * masses[i] * masses[j]
                    } else {
                        0.0
                    }
                } else if d > 0.0 {
                    scaling_ratio * masses[i] * masses[j] / (d * d)
                } else {
                    0.0
                };
                f[0] += dx * factor;
                f[1] += dy * factor;
            }
            for &(j, w) in &adj[i] {
                let (dx, dy) = (xi - current[j][0], yi - current[j][1]);
                let mut d = (dx * dx + dy * dy).sqrt();
                if adjust_sizes {
                    d -= sizes[i] + sizes[j];
                    if d <= 0.0 {
                        continue;
                    }
                }
                let mut factor = if linlog {
                    if d > 0.0 {
                        w * (1.0 + d).ln() / d
                    } else {
                        0.0
                    }
                } else {
                    w
                };
                if dissuade_hubs {
                    factor /= masses[i];
                }
                f[0] -= dx * factor;
                f[1] -= dy * factor;
            }
            let (gx, gy) = (xi - cx, yi - cy);
            let gd = (gx * gx + gy * gy).sqrt();
            let g = if strong_gravity {
                gravity * masses[i]
            } else if gd > 0.0 {
                gravity * masses[i] / gd
            } else {
                0.0
            };
            f[0] -= gx * g;
            f[1] -= gy * g;
            f
        });

        // Adaptive speed (the paper's Algorithm 1, as in Gephi)
        let norm = |v: [f64; 2]| (v[0] * v[0] + v[1] * v[1]).sqrt();
        let swinging: Vec<f64> = (0..n)
            .map(|i| {
                masses[i] * norm([force[i][0] - old_force[i][0], force[i][1] - old_force[i][1]])
            })
            .collect();
        let total_swinging: f64 = swinging.iter().sum();
        let total_traction: f64 = (0..n)
            .map(|i| {
                0.5 * masses[i]
                    * norm([force[i][0] + old_force[i][0], force[i][1] + old_force[i][1]])
            })
            .sum();
        let optimal_jitter = 0.05 * (n as f64).sqrt();
        let min_jitter = optimal_jitter.sqrt();
        let max_jitter = 10.0_f64;
        let mut jitter = jitter_tolerance
            * min_jitter.max(max_jitter.min(optimal_jitter * total_traction / (n * n) as f64));
        let min_speed_efficiency = 0.05;
        if total_traction > 0.0 && total_swinging / total_traction > 2.0 {
            if speed_efficiency > min_speed_efficiency {
                speed_efficiency *= 0.5;
            }
            jitter = jitter.max(jitter_tolerance);
        }
        let target_speed = if total_swinging > 0.0 {
            jitter * speed_efficiency * total_traction / total_swinging
        } else {
            f64::INFINITY
        };
        if total_swinging > jitter * total_traction {
            if speed_efficiency > min_speed_efficiency {
                speed_efficiency *= 0.7;
            }
        } else if speed < 1000.0 {
            speed_efficiency *= 1.3;
        }
        speed += (target_speed - speed).min(0.5 * speed);

        let mut moved = 0.0;
        for i in 0..n {
            let factor = if adjust_sizes {
                let df = norm(force[i]);
                let f = 0.1 * speed / (1.0 + (speed * swinging[i]).sqrt());
                if df > 0.0 {
                    (f * df).min(10.0) / df
                } else {
                    0.0
                }
            } else {
                speed / (1.0 + (speed * swinging[i]).sqrt())
            };
            let step = [force[i][0] * factor, force[i][1] * factor];
            pos[i][0] += step[0];
            pos[i][1] += step[1];
            moved += step[0].abs() + step[1].abs();
        }
        old_force = force;
        if moved < 1e-10 {
            break;
        }
    }
    Ok(rows(&ids, &pos))
}

/// Attractive and repulsive forces (ARF) positions (Geipel 2007, NetworkX's `arf_layout()`).
///
/// Every pair is joined by a spring of strength 1, raised to `a` for adjacent nodes, and
/// repels with `rho / d`, where `rho = scaling * sqrt(n)`. Positions follow the force field
/// in steps of `dt` until the summed force falls to `etol` or `max_iter` steps have run.
#[pyfunction]
#[pyo3(signature = (graph, scaling=1.0, a=1.1, etol=1e-6, dt=1e-3, max_iter=1000, initial=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn arf_layout(
    graph: &Graph,
    scaling: f64,
    a: f64,
    etol: f64,
    dt: f64,
    max_iter: usize,
    initial: Option<HashMap<u64, (f64, f64)>>,
    seed: Option<u64>,
) -> PyResult<Positions> {
    if !(a.is_finite() && a > 1.0) {
        return Err(err("a must be larger than 1"));
    }
    for (name, value) in [("scaling", scaling), ("dt", dt)] {
        if !(value.is_finite() && value > 0.0) {
            return Err(err(format!("{} must be a positive number", name)));
        }
    }
    if !(etol.is_finite() && etol >= 0.0) {
        return Err(err("etol must be a non-negative number"));
    }
    check_initial(&initial)?;
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut pos = initial_positions(&ids, &initial, seed);
    if n <= 1 {
        return Ok(rows(&ids, &pos));
    }
    let adj = neighbor_weights(graph, &idx, false);
    let rho = scaling * (n as f64).sqrt();

    for _ in 0..max_iter {
        let (sx, sy) = pos
            .iter()
            .fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
        let current = &pos;
        let adj = &adj;
        let change: Vec<[f64; 2]> = per_node(n, |j| {
            let [xj, yj] = current[j];
            // Unit springs to every node, plus (a - 1) more to each neighbor
            let mut c = [sx - n as f64 * xj, sy - n as f64 * yj];
            for &(i, _) in &adj[j] {
                c[0] += (a - 1.0) * (current[i][0] - xj);
                c[1] += (a - 1.0) * (current[i][1] - yj);
            }
            for (i, pi) in current.iter().enumerate() {
                let (dx, dy) = (pi[0] - xj, pi[1] - yj);
                let d = (dx * dx + dy * dy).sqrt();
                if i != j && d > 0.0 {
                    c[0] -= rho * dx / d;
                    c[1] -= rho * dy / d;
                }
            }
            c
        });
        let mut error = 0.0;
        for (p, c) in pos.iter_mut().zip(&change) {
            p[0] += c[0] * dt;
            p[1] += c[1] * dt;
            error += (c[0] * c[0] + c[1] * c[1]).sqrt();
        }
        if error <= etol {
            break;
        }
    }
    Ok(rows(&ids, &pos))
}
