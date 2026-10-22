//! Exact graph edit distance by depth-first branch and bound.
//!
//! A port of the search behind NetworkX's `optimize_edit_paths()` (Abu-Aisheh et al., 2015).
//! Vertex and edge edit costs arrive as dense matrices built on the Python side, so the search
//! here only sees indices: nodes are numbered `0..n1` and `0..n2`, edges `0..e1` and `0..e2`,
//! and each edge carries the node indices of its endpoints. Lower bounds come from linear sum
//! assignments over the pending vertex and edge mappings, solved with the Hungarian algorithm.
//!
//! One deliberate difference from NetworkX concerns self-loops: here a loop may only be
//! substituted by a loop on the corresponding node, and loops are settled together with their
//! node. NetworkX lets a loop stand in for any edge at the matched nodes, and leaves loops on the
//! first mapped node pending, so it can report a cost below the true distance.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;

type Pair = (Option<usize>, Option<usize>);
type EditPath = (Vec<Pair>, Vec<Pair>, f64);

/// Square cost matrix with an optimal assignment `row[p] <-> col[p]` of total cost `ls`.
#[derive(Clone)]
struct CostMatrix {
    c: Vec<f64>,
    dim: usize,
    row: Vec<usize>,
    col: Vec<usize>,
    ls: f64,
}

impl CostMatrix {
    fn empty() -> Self {
        CostMatrix {
            c: Vec::new(),
            dim: 0,
            row: Vec::new(),
            col: Vec::new(),
            ls: 0.0,
        }
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.c[i * self.dim + j]
    }
}

/// Minimum-cost perfect assignment of a square matrix: the column given to each row.
fn lsap(c: &[f64], dim: usize) -> Vec<usize> {
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; dim + 1];
    let mut v = vec![0.0f64; dim + 1];
    let mut p = vec![0usize; dim + 1];
    let mut way = vec![0usize; dim + 1];
    for i in 1..=dim {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; dim + 1];
        let mut used = vec![false; dim + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=dim {
                if used[j] {
                    continue;
                }
                let cur = c[(i0 - 1) * dim + (j - 1)] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=dim {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut col_of = vec![0usize; dim];
    for j in 1..=dim {
        if p[j] != 0 {
            col_of[p[j] - 1] = j - 1;
        }
    }
    col_of
}

/// Solve the assignment of an `(m + n)`-square matrix, then pair the dummy assignments with
/// the substitutions (each substitution `i <-> j` gets the dummy `m + j <-> n + i`), which the
/// cheap reduction in `get_edit_ops` relies on.
fn make_cost_matrix(c: Vec<f64>, m: usize, n: usize) -> CostMatrix {
    let dim = m + n;
    debug_assert_eq!(c.len(), dim * dim);
    let mut row: Vec<usize> = (0..dim).collect();
    let mut col = lsap(&c, dim);
    let subst: Vec<usize> = (0..dim).filter(|&p| row[p] < m && col[p] < n).collect();
    let dummy: Vec<usize> = (0..dim).filter(|&p| row[p] >= m && col[p] >= n).collect();
    for (&d, &s) in dummy.iter().zip(&subst) {
        row[d] = col[s] + m;
        col[d] = row[s] + n;
    }
    let ls = (0..dim).map(|p| c[row[p] * dim + col[p]]).sum();
    CostMatrix {
        c,
        dim,
        row,
        col,
        ls,
    }
}

/// Submatrix of `c` with the rows and columns selected by `keep_row` and `keep_col`.
fn submatrix(c: &[f64], dim: usize, keep_row: &[bool], keep_col: &[bool]) -> Vec<f64> {
    let cols: Vec<usize> = (0..dim).filter(|&k| keep_col[k]).collect();
    let mut out = Vec::with_capacity(cols.len() * cols.len());
    for r in (0..dim).filter(|&k| keep_row[k]) {
        out.extend(cols.iter().map(|&k| c[r * dim + k]));
    }
    out
}

/// Rows `i` (and `m + j`) and columns `j` (and `n + i`) of `c`: the cross of the mappings.
fn extract_c(c: &[f64], dim: usize, i: &[usize], j: &[usize], m: usize, n: usize) -> Vec<f64> {
    let keep_row: Vec<bool> = (0..dim)
        .map(|k| i.contains(&k) || (k >= m && j.contains(&(k - m))))
        .collect();
    let keep_col: Vec<bool> = (0..dim)
        .map(|k| j.contains(&k) || (k >= n && i.contains(&(k - n))))
        .collect();
    submatrix(c, dim, &keep_row, &keep_col)
}

/// `c` without the rows and columns that `extract_c` would select.
fn reduce_c(c: &[f64], dim: usize, i: &[usize], j: &[usize], m: usize, n: usize) -> Vec<f64> {
    let keep_row: Vec<bool> = (0..dim)
        .map(|k| !(i.contains(&k) || (k >= m && j.contains(&(k - m)))))
        .collect();
    let keep_col: Vec<bool> = (0..dim)
        .map(|k| !(j.contains(&k) || (k >= n && i.contains(&(k - n)))))
        .collect();
    submatrix(c, dim, &keep_row, &keep_col)
}

/// Drop the values in `removed` from an index permutation and renumber the rest.
fn reduce_ind(ind: &[usize], removed: &[usize]) -> Vec<usize> {
    ind.iter()
        .filter(|k| !removed.contains(k))
        .map(|&k| {
            let mut seen: Vec<usize> = Vec::new();
            for &r in removed {
                if r < k && !seen.contains(&r) {
                    seen.push(r);
                }
            }
            k - seen.len()
        })
        .collect()
}

/// One candidate vertex edit: the mapping `i <-> j`, the reduced vertex matrix, the edge
/// mappings it settles, the reduced edge matrix and the cost of the operation.
struct EditOp {
    i: usize,
    j: usize,
    cv: CostMatrix,
    xy: Vec<(usize, usize)>,
    ce: CostMatrix,
    cost: f64,
}

struct Search {
    g_ends: Vec<(usize, usize)>,
    h_ends: Vec<(usize, usize)>,
    directed: bool,
    edge_inf: f64,
    upper_bound: Option<f64>,
    strictly_decreasing: bool,
    deadline: Option<(Instant, f64)>,
    best_only: bool,
    maxcost: f64,
    pending_u: Vec<usize>,
    pending_v: Vec<usize>,
    pending_g: Vec<usize>,
    pending_h: Vec<usize>,
    matched_uv: Vec<Pair>,
    matched_gh: Vec<Pair>,
    results: Vec<EditPath>,
}

impl Search {
    fn prune(&self, cost: f64) -> bool {
        if let Some((start, limit)) = self.deadline {
            if start.elapsed().as_secs_f64() > limit {
                return true;
            }
        }
        if let Some(ub) = self.upper_bound {
            if cost > ub {
                return true;
            }
        }
        cost > self.maxcost || (self.strictly_decreasing && cost >= self.maxcost)
    }

    /// Whether pending edge `g` (in G1) may be substituted by pending edge `h` (in G2) once
    /// `u <-> v` joins the matched vertices.
    fn compatible(
        &self,
        g: (usize, usize),
        h: (usize, usize),
        u: Option<usize>,
        v: Option<usize>,
    ) -> bool {
        let (ga, gb) = (Some(g.0), Some(g.1));
        let (ha, hb) = (Some(h.0), Some(h.1));
        if ga == gb || ha == hb {
            // Self-loops correspond only to self-loops on the matched partner.
            if ga != gb || ha != hb {
                return false;
            }
            return (ga == u && ha == v)
                || self.matched_uv.iter().any(|&(p, q)| ga == p && ha == q);
        }
        self.matched_uv.iter().any(|&(p, q)| {
            if self.directed {
                (ga == p && gb == u && ha == q && hb == v)
                    || (ga == u && gb == p && ha == v && hb == q)
            } else {
                ((ga == p && gb == u) || (ga == u && gb == p))
                    && ((ha == q && hb == v) || (ha == v && hb == q))
            }
        })
    }

    /// The pending edges settled by the vertex mapping `u <-> v`, and their optimal mapping.
    fn match_edges(
        &self,
        u: Option<usize>,
        v: Option<usize>,
        ce: &CostMatrix,
    ) -> (Vec<(usize, usize)>, CostMatrix) {
        let big_m = self.pending_g.len();
        let big_n = self.pending_h.len();
        let touches = |a: usize, b: usize, w: Option<usize>, side: usize| {
            let (a, b) = (Some(a), Some(b));
            (a == w && b == w)
                || self.matched_uv.iter().any(|pq| {
                    let p = if side == 0 { pq.0 } else { pq.1 };
                    (a == p && b == w) || (a == w && b == p) || (a == p && b == p)
                })
        };
        let g_ind: Vec<usize> = (0..big_m)
            .filter(|&i| {
                let (a, b) = self.g_ends[self.pending_g[i]];
                touches(a, b, u, 0)
            })
            .collect();
        let h_ind: Vec<usize> = (0..big_n)
            .filter(|&j| {
                let (a, b) = self.h_ends[self.pending_h[j]];
                touches(a, b, v, 1)
            })
            .collect();
        let (m, n) = (g_ind.len(), h_ind.len());
        if m == 0 && n == 0 {
            return (Vec::new(), CostMatrix::empty());
        }
        let mut c = extract_c(&ce.c, ce.dim, &g_ind, &h_ind, big_m, big_n);
        let dim = m + n;
        for (k, &i) in g_ind.iter().enumerate() {
            let g = self.g_ends[self.pending_g[i]];
            for (l, &j) in h_ind.iter().enumerate() {
                let h = self.h_ends[self.pending_h[j]];
                if !self.compatible(g, h, u, v) {
                    c[k * dim + l] = self.edge_inf;
                }
            }
        }
        let local = make_cost_matrix(c, m, n);
        let ij = (0..dim)
            .filter_map(|p| {
                let (k, l) = (local.row[p], local.col[p]);
                if k < m || l < n {
                    Some((
                        if k < m { g_ind[k] } else { big_m + h_ind[l] },
                        if l < n { h_ind[l] } else { big_n + g_ind[k] },
                    ))
                } else {
                    None
                }
            })
            .collect();
        (ij, local)
    }

    fn reduce_ce(&self, ce: &CostMatrix, ij: &[(usize, usize)]) -> CostMatrix {
        if ij.is_empty() {
            return ce.clone();
        }
        let (big_m, big_n) = (self.pending_g.len(), self.pending_h.len());
        let i: Vec<usize> = ij.iter().map(|t| t.0).collect();
        let j: Vec<usize> = ij.iter().map(|t| t.1).collect();
        let m_i = big_m - i.iter().filter(|&&t| t < big_m).count();
        let n_j = big_n - j.iter().filter(|&&t| t < big_n).count();
        make_cost_matrix(reduce_c(&ce.c, ce.dim, &i, &j, big_m, big_n), m_i, n_j)
    }

    fn vertex(&self, i: usize, j: usize) -> (Option<usize>, Option<usize>) {
        (
            self.pending_u.get(i).copied(),
            self.pending_v.get(j).copied(),
        )
    }

    fn get_edit_paths(&mut self, cv: &CostMatrix, ce: &CostMatrix, matched_cost: f64) {
        if self.prune(matched_cost + cv.ls + ce.ls) {
            return;
        }
        let (m, n) = (self.pending_u.len(), self.pending_v.len());
        if m == 0 && n == 0 {
            self.maxcost = self.maxcost.min(matched_cost);
            if self.best_only {
                if let Some(last) = self.results.last() {
                    if matched_cost < last.2 {
                        self.results.clear();
                    }
                }
            }
            self.results.push((
                self.matched_uv.clone(),
                self.matched_gh.clone(),
                matched_cost,
            ));
            return;
        }

        // 1) the vertex mapping from the optimal linear sum assignment
        let (fixed_i, fixed_j) = (0..cv.dim)
            .map(|p| (cv.row[p], cv.col[p]))
            .filter(|&(k, l)| k < m || l < n)
            .min()
            .expect("a pending vertex has a mapping");
        let (u, v) = self.vertex(fixed_i, fixed_j);
        let (xy, local) = self.match_edges(u, v, ce);
        let ce_xy = self.reduce_ce(ce, &xy);
        if !self.prune(matched_cost + cv.ls + local.ls + ce_xy.ls) {
            let (i, j) = (fixed_i, fixed_j);
            let cv_ij = CostMatrix {
                c: reduce_c(&cv.c, cv.dim, &[i], &[j], m, n),
                dim: cv.dim - if i < m && j < n { 2 } else { 1 },
                row: reduce_ind(&cv.row, &[i, m + j]),
                col: reduce_ind(&cv.col, &[j, n + i]),
                ls: cv.ls - cv.at(i, j),
            };
            let cost = cv.at(i, j) + local.ls;
            self.dive(
                EditOp {
                    i,
                    j,
                    cv: cv_ij,
                    xy,
                    ce: ce_xy,
                    cost,
                },
                matched_cost,
            );
        }

        // 2) the other candidates for the same column (or row), best lower bound first
        let candidates: Vec<(usize, usize)> = if m <= n {
            (0..m + n)
                .filter(|&t| t != fixed_i && (t < m || t == m + fixed_j))
                .map(|t| (t, fixed_j))
                .collect()
        } else {
            (0..m + n)
                .filter(|&t| t != fixed_j && (t < n || t == n + fixed_i))
                .map(|t| (fixed_i, t))
                .collect()
        };
        let mut other: Vec<EditOp> = Vec::new();
        for (i, j) in candidates {
            if self.prune(matched_cost + cv.at(i, j) + ce.ls) {
                continue;
            }
            let cv_ij = make_cost_matrix(
                reduce_c(&cv.c, cv.dim, &[i], &[j], m, n),
                if i < m { m - 1 } else { m },
                if j < n { n - 1 } else { n },
            );
            if self.prune(matched_cost + cv.at(i, j) + cv_ij.ls + ce.ls) {
                continue;
            }
            let (u, v) = self.vertex(i, j);
            let (xy, local) = self.match_edges(u, v, ce);
            if self.prune(matched_cost + cv.at(i, j) + cv_ij.ls + local.ls) {
                continue;
            }
            let ce_xy = self.reduce_ce(ce, &xy);
            if self.prune(matched_cost + cv.at(i, j) + cv_ij.ls + local.ls + ce_xy.ls) {
                continue;
            }
            let cost = cv.at(i, j) + local.ls;
            other.push(EditOp {
                i,
                j,
                cv: cv_ij,
                xy,
                ce: ce_xy,
                cost,
            });
        }
        other.sort_by(|a, b| (a.cost + a.cv.ls + a.ce.ls).total_cmp(&(b.cost + b.cv.ls + b.ce.ls)));
        for op in other {
            self.dive(op, matched_cost);
        }
    }

    /// Apply an edit operation, search below it, and undo it.
    fn dive(&mut self, op: EditOp, matched_cost: f64) {
        if self.prune(matched_cost + op.cost + op.cv.ls + op.ce.ls) {
            return;
        }
        let EditOp {
            i,
            j,
            cv,
            xy,
            ce,
            cost,
        } = op;
        let u = if i < self.pending_u.len() {
            Some(self.pending_u.remove(i))
        } else {
            None
        };
        let v = if j < self.pending_v.len() {
            Some(self.pending_v.remove(j))
        } else {
            None
        };
        self.matched_uv.push((u, v));
        let (len_g, len_h) = (self.pending_g.len(), self.pending_h.len());
        for &(x, y) in &xy {
            let g = if x < len_g {
                Some(self.pending_g[x])
            } else {
                None
            };
            let h = if y < len_h {
                Some(self.pending_h[y])
            } else {
                None
            };
            self.matched_gh.push((g, h));
        }
        let mut sorted_x: Vec<usize> = xy.iter().map(|t| t.0).filter(|&x| x < len_g).collect();
        let mut sorted_y: Vec<usize> = xy.iter().map(|t| t.1).filter(|&y| y < len_h).collect();
        sorted_x.sort_unstable();
        sorted_y.sort_unstable();
        let removed_g: Vec<usize> = sorted_x
            .iter()
            .rev()
            .map(|&x| self.pending_g.remove(x))
            .collect();
        let removed_h: Vec<usize> = sorted_y
            .iter()
            .rev()
            .map(|&y| self.pending_h.remove(y))
            .collect();

        self.get_edit_paths(&cv, &ce, matched_cost + cost);

        // backtrack
        if let Some(u) = u {
            self.pending_u.insert(i, u);
        }
        if let Some(v) = v {
            self.pending_v.insert(j, v);
        }
        self.matched_uv.pop();
        for (&x, &g) in sorted_x.iter().zip(removed_g.iter().rev()) {
            self.pending_g.insert(x, g);
        }
        for (&y, &h) in sorted_y.iter().zip(removed_h.iter().rev()) {
            self.pending_h.insert(y, h);
        }
        for _ in &xy {
            self.matched_gh.pop();
        }
    }
}

/// Square `(m + n)` edit cost matrix: substitutions top left, deletions on the diagonal of the
/// top right block, insertions on the diagonal of the bottom left block, zeros bottom right,
/// and a finite "infinity" (larger than any complete edit path) everywhere else.
fn edit_matrix(subst: &[f64], del: &[f64], ins: &[f64]) -> (Vec<f64>, f64) {
    let (m, n) = (del.len(), ins.len());
    let dim = m + n;
    let inf = subst.iter().sum::<f64>() + del.iter().sum::<f64>() + ins.iter().sum::<f64>() + 1.0;
    let mut c = vec![0.0; dim * dim];
    for i in 0..m {
        for j in 0..n {
            c[i * dim + j] = subst[i * n + j];
        }
        for k in 0..m {
            c[i * dim + n + k] = if i == k { del[i] } else { inf };
        }
    }
    for i in 0..n {
        for j in 0..n {
            c[(m + i) * dim + j] = if i == j { ins[i] } else { inf };
        }
    }
    (c, inf)
}

/// Branch-and-bound search for graph edit paths.
///
/// `pending_u` and `pending_v` are the node indices to map (all nodes except the roots), in
/// the order of the rows and columns of `node_subst`. Edges are given by endpoint node indices.
/// Returns the edit paths found, each as `(node_pairs, edge_pairs, cost)`; with
/// `best_only=True` only the paths of the final, lowest cost are kept.
#[pyfunction]
#[pyo3(signature = (
    pending_u, pending_v, node_subst, node_del, node_ins,
    g_ends, h_ends, edge_subst, edge_del, edge_ins,
    directed, roots, initial_cost, upper_bound, strictly_decreasing, timeout, best_only,
))]
#[allow(clippy::too_many_arguments)]
pub fn optimize_edit_paths(
    pending_u: Vec<usize>,
    pending_v: Vec<usize>,
    node_subst: Vec<f64>,
    node_del: Vec<f64>,
    node_ins: Vec<f64>,
    g_ends: Vec<(usize, usize)>,
    h_ends: Vec<(usize, usize)>,
    edge_subst: Vec<f64>,
    edge_del: Vec<f64>,
    edge_ins: Vec<f64>,
    directed: bool,
    roots: Option<(usize, usize)>,
    initial_cost: f64,
    upper_bound: Option<f64>,
    strictly_decreasing: bool,
    timeout: Option<f64>,
    best_only: bool,
) -> PyResult<Vec<EditPath>> {
    let (m, n) = (pending_u.len(), pending_v.len());
    let (e1, e2) = (g_ends.len(), h_ends.len());
    if node_subst.len() != m * n || node_del.len() != m || node_ins.len() != n {
        return Err(PyValueError::new_err(
            "node cost arrays do not match the pending nodes",
        ));
    }
    if edge_subst.len() != e1 * e2 || edge_del.len() != e1 || edge_ins.len() != e2 {
        return Err(PyValueError::new_err(
            "edge cost arrays do not match the edges",
        ));
    }
    let (cv_c, _) = edit_matrix(&node_subst, &node_del, &node_ins);
    let (ce_c, edge_inf) = edit_matrix(&edge_subst, &edge_del, &edge_ins);
    let cv = make_cost_matrix(cv_c, m, n);
    let ce = make_cost_matrix(ce_c, e1, e2);
    let maxcost = cv.c.iter().sum::<f64>() + ce.c.iter().sum::<f64>() + 1.0;
    let mut search = Search {
        g_ends,
        h_ends,
        directed,
        edge_inf,
        upper_bound,
        strictly_decreasing,
        deadline: timeout.map(|t| (Instant::now(), t)),
        best_only,
        maxcost,
        pending_u,
        pending_v,
        pending_g: (0..e1).collect(),
        pending_h: (0..e2).collect(),
        matched_uv: roots
            .map(|(a, b)| vec![(Some(a), Some(b))])
            .unwrap_or_default(),
        matched_gh: Vec::new(),
        results: Vec::new(),
    };
    search.get_edit_paths(&cv, &ce, initial_cost);
    Ok(search.results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsap_finds_minimum() {
        let c = vec![4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let col = lsap(&c, 3);
        let total: f64 = (0..3).map(|i| c[i * 3 + col[i]]).sum();
        assert_eq!(total, 5.0);
    }

    #[test]
    fn reduce_ind_renumbers() {
        assert_eq!(reduce_ind(&[3, 0, 5, 2, 1, 4], &[2, 5]), vec![2, 0, 1, 3]);
        assert_eq!(reduce_ind(&[1, 0, 2], &[2, 2]), vec![1, 0]);
    }

    fn path(n: usize) -> Vec<(usize, usize)> {
        (1..n).map(|i| (i - 1, i)).collect()
    }

    fn ged(n1: usize, g: Vec<(usize, usize)>, n2: usize, h: Vec<(usize, usize)>) -> f64 {
        let (e1, e2) = (g.len(), h.len());
        let res = optimize_edit_paths(
            (0..n1).collect(),
            (0..n2).collect(),
            vec![0.0; n1 * n2],
            vec![1.0; n1],
            vec![1.0; n2],
            g,
            h,
            vec![0.0; e1 * e2],
            vec![1.0; e1],
            vec![1.0; e2],
            false,
            None,
            0.0,
            None,
            true,
            None,
            true,
        )
        .unwrap();
        res.last().unwrap().2
    }

    #[test]
    fn path_versus_cycle() {
        let mut cycle = path(4);
        cycle.push((3, 0));
        assert_eq!(ged(4, path(4), 4, cycle), 1.0);
        assert_eq!(ged(3, path(3), 3, path(3)), 0.0);
        assert_eq!(ged(2, path(2), 4, path(4)), 4.0);
    }

    #[test]
    fn loops_only_match_loops() {
        assert_eq!(ged(2, vec![(0, 1), (0, 0)], 3, vec![(0, 1), (1, 2)]), 3.0);
        assert_eq!(ged(1, vec![(0, 0)], 1, vec![]), 1.0);
        assert_eq!(ged(1, vec![(0, 0)], 1, vec![(0, 0)]), 0.0);
    }
}
