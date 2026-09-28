//! Extremal eigenpairs of large sparse symmetric matrices
//!
//! Dense eigendecomposition costs O(n³) time and O(n²) memory even when only a
//! handful of eigenvalues are wanted. This module computes the `k` largest or
//! smallest eigenpairs of a symmetric CSR matrix with Chebyshev-filtered subspace
//! iteration (Zhou & Saad, 2007):
//!
//! 1. Keep an orthonormal block `X` of `k + guard` vectors.
//! 2. Rayleigh-Ritz: diagonalize `Xᵀ A X` and rotate `X` onto the Ritz vectors.
//! 3. Stop once the `k` wanted Ritz pairs have residual `‖A x − θ x‖ ≤ tol · ‖A‖`.
//! 4. Otherwise apply a Chebyshev polynomial of `A` that damps the unwanted interval
//!    `[cut, upper]` and amplifies the wanted end, then re-orthonormalize. `cut` sits
//!    between the largest Ritz value and the lowest eigenvalue the residuals see;
//!    `upper` is a Lanczos estimate capped by the Gershgorin bound. The polynomial
//!    degree grows as iterations accumulate, so hard problems get stronger filters.
//!
//! Being a block method, it resolves repeated eigenvalues (up to the block size),
//! which are common in graph spectra (components, cliques, symmetric graphs). A
//! single-vector Lanczos process can silently miss those copies.

use nalgebra::{DMatrix, SymmetricEigen};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Symmetric sparse matrix in compressed sparse row format
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    n: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Build an `n × n` matrix from `(row, col, value)` triplets, summing duplicates
    pub fn from_triplets(n: usize, mut triplets: Vec<(usize, usize, f64)>) -> Self {
        triplets.sort_unstable_by_key(|&(i, j, _)| (i, j));

        let mut indptr = vec![0usize; n + 1];
        let mut indices = Vec::with_capacity(triplets.len());
        let mut values: Vec<f64> = Vec::with_capacity(triplets.len());

        let mut last: Option<(usize, usize)> = None;
        for (i, j, v) in triplets {
            if last == Some((i, j)) {
                *values.last_mut().unwrap() += v;
            } else {
                indices.push(j);
                values.push(v);
                indptr[i + 1] += 1;
                last = Some((i, j));
            }
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }

        CsrMatrix {
            n,
            indptr,
            indices,
            values,
        }
    }

    /// Matrix dimension
    pub fn n(&self) -> usize {
        self.n
    }

    /// Iterate over `(col, value)` entries of row `i`
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.indptr[i]..self.indptr[i + 1];
        self.indices[range.clone()]
            .iter()
            .copied()
            .zip(self.values[range].iter().copied())
    }

    /// Sum of each row
    pub fn row_sums(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| self.row(i).map(|(_, v)| v).sum())
            .collect()
    }

    /// Diagonal entry `(i, i)`
    pub fn diagonal(&self, i: usize) -> f64 {
        self.row(i).filter(|&(j, _)| j == i).map(|(_, v)| v).sum()
    }

    /// Expand to a dense matrix
    pub fn to_dense(&self) -> DMatrix<f64> {
        let mut dense = DMatrix::<f64>::zeros(self.n, self.n);
        for i in 0..self.n {
            for (j, v) in self.row(i) {
                dense[(i, j)] += v;
            }
        }
        dense
    }

    /// Gershgorin interval containing every eigenvalue
    pub fn gershgorin_bounds(&self) -> (f64, f64) {
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for i in 0..self.n {
            let mut center = 0.0;
            let mut radius = 0.0;
            for (j, v) in self.row(i) {
                if j == i {
                    center += v;
                } else {
                    radius += v.abs();
                }
            }
            lower = lower.min(center - radius);
            upper = upper.max(center + radius);
        }
        (lower, upper)
    }

    /// `y = sign · A x` for a single vector
    fn mul_vec(&self, sign: f64, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in self.indptr[i]..self.indptr[i + 1] {
                acc += self.values[k] * x[self.indices[k]];
            }
            *yi = sign * acc;
        }
    }

    /// `Y = sign · A X` for a block of column vectors
    fn mul_block(&self, sign: f64, x: &DMatrix<f64>) -> DMatrix<f64> {
        let n = self.n;
        let mut y = DMatrix::<f64>::zeros(n, x.ncols());
        y.as_mut_slice()
            .par_chunks_mut(n)
            .zip(x.as_slice().par_chunks(n))
            .for_each(|(yc, xc)| self.mul_vec(sign, xc, yc));
        y
    }
}

/// Which end of the spectrum to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extremal {
    Largest,
    Smallest,
}

/// Solver settings
#[derive(Debug, Clone)]
pub struct SubspaceOptions {
    /// Residual tolerance relative to the spectral radius bound
    pub tol: f64,
    /// Maximum number of filter + Rayleigh-Ritz iterations
    pub max_iter: usize,
    /// Initial Chebyshev polynomial degree per iteration
    pub degree: usize,
    /// Largest Chebyshev degree; the degree doubles every `degree_period` iterations
    pub max_degree: usize,
    /// Iterations between degree increases
    pub degree_period: usize,
    /// Extra block vectors beyond `k` (minimum)
    pub min_guard: usize,
    /// Seed for the random starting block
    pub seed: u64,
}

impl Default for SubspaceOptions {
    fn default() -> Self {
        SubspaceOptions {
            tol: 1e-11,
            max_iter: 5000,
            degree: 10,
            max_degree: 40,
            degree_period: 30,
            min_guard: 8,
            seed: 0x5eed,
        }
    }
}

/// Iteration cap that keeps the sparse solver cheaper than a dense solve
///
/// Used when a dense fallback is available: a stalled sparse solve should give up
/// after spending roughly what the dense eigendecomposition would cost.
pub fn iteration_budget(a: &CsrMatrix, k: usize, options: &SubspaceOptions) -> usize {
    let n = a.n() as f64;
    let b = block_size(k, options) as f64;
    let nnz = a.values.len().max(a.n()) as f64;
    let dense_flops = 10.0 * n * n * n;
    let per_iter_flops = (options.max_degree as f64 + 2.0) * 2.0 * nnz * b + 4.0 * n * b * b;
    ((dense_flops / per_iter_flops) as usize).clamp(50, options.max_iter)
}

/// Block size used for `k` wanted eigenpairs
pub fn block_size(k: usize, options: &SubspaceOptions) -> usize {
    k + options.min_guard.max(k / 2)
}

/// Compute the `k` extremal eigenpairs of a symmetric sparse matrix
///
/// Returns pairs sorted from the wanted end inwards (descending for `Largest`,
/// ascending for `Smallest`), with unit eigenvectors. Returns `None` if the
/// iteration does not converge within `options.max_iter`. Callers should use a
/// dense solver when `block_size(k)` is not much smaller than `n`.
pub fn symmetric_extremal_eigenpairs(
    a: &CsrMatrix,
    k: usize,
    which: Extremal,
    options: &SubspaceOptions,
) -> Option<Vec<(f64, Vec<f64>)>> {
    let n = a.n();
    if k == 0 {
        return Some(Vec::new());
    }
    let b = block_size(k, options).min(n);
    assert!(k <= b, "k must not exceed the matrix dimension");

    // Work with B = sign · A and always look for B's smallest eigenvalues
    let sign = if which == Extremal::Largest {
        -1.0
    } else {
        1.0
    };
    let (g_lower, g_upper) = a.gershgorin_bounds();
    let (lower, g_upper) = if sign > 0.0 {
        (g_lower, g_upper)
    } else {
        (-g_upper, -g_lower)
    };
    let scale = lower.abs().max(g_upper.abs()).max(f64::MIN_POSITIVE);
    let threshold = options.tol * scale;

    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    let mut x = orthonormalize(DMatrix::from_fn(n, b, |_, _| rng.gen_range(-1.0..1.0)));

    // A tighter window [cut, upper] makes the filter far more effective on graphs
    // whose Gershgorin bound is loose (e.g. hubs in heavy-tailed degree distributions)
    let start: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let upper = lanczos_upper_bound(a, sign, &start, 20).min(g_upper);

    for iter in 0..options.max_iter {
        // Rayleigh-Ritz on span(X)
        let bx = a.mul_block(sign, &x);
        let h = x.transpose() * &bx;
        let h = (&h + h.transpose()) * 0.5;
        let eig = SymmetricEigen::new(h);

        let mut order: Vec<usize> = (0..b).collect();
        order.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
        let theta: Vec<f64> = order.iter().map(|&i| eig.eigenvalues[i]).collect();
        let w = DMatrix::from_fn(b, b, |r, c| eig.eigenvectors[(r, order[c])]);

        x = &x * &w;
        let bx = bx * &w;

        let residuals =
            &bx - &x * DMatrix::from_diagonal(&nalgebra::DVector::from_vec(theta.clone()));
        let worst = (0..k)
            .map(|c| residuals.column(c).norm())
            .fold(0.0, f64::max);
        if worst <= threshold {
            return Some(
                (0..k)
                    .map(|c| (sign * theta[c], x.column(c).iter().copied().collect()))
                    .collect(),
            );
        }

        // Damp [cut, upper]; everything below `cut` is amplified. The largest Ritz
        // value alone is a poor cut when a repeated eigenvalue has more copies than
        // the block holds: every Ritz value (guards included) drifts onto it, the cut
        // closes in on the wanted eigenvalue, and the filter stops separating it from
        // the rest. So move the cut halfway towards the lowest eigenvalue seen by the
        // residuals, which are orthogonal to the block and so estimate what it misses.
        let mut cut = theta[b - 1];
        if let Some(missing) = residual_spectrum_min(a, sign, &residuals) {
            if missing > cut {
                cut += 0.5 * (missing - cut);
            }
        }
        let half_width = (upper - cut) / 2.0;
        let y = if half_width > 1e-12 * scale {
            // Easy problems finish at low degree; slow ones get stronger filters
            let doublings = (iter / options.degree_period.max(1)).min(8) as u32;
            let degree = (options.degree << doublings).min(options.max_degree.max(options.degree));
            chebyshev_filter(a, sign, &x, degree, lower, cut, upper)
        } else {
            // Ritz values already reach the top of the spectrum: shifted power step
            x.clone() * upper - a.mul_block(sign, &x)
        };
        x = orthonormalize(y);
    }

    None
}

/// Apply the scaled Chebyshev filter `p_m(B) X` that damps `[cut, upper]`
///
/// Uses the scaled three-term recurrence so values stay O(1) near `lower`, a lower
/// bound of B's spectrum.
fn chebyshev_filter(
    a: &CsrMatrix,
    sign: f64,
    x: &DMatrix<f64>,
    degree: usize,
    lower: f64,
    cut: f64,
    upper: f64,
) -> DMatrix<f64> {
    let e = (upper - cut) / 2.0;
    let c = (upper + cut) / 2.0;
    let mut sigma = e / (lower.min(cut) - c);
    let tau = 2.0 / sigma;

    let mut prev = x.clone();
    let mut cur = (a.mul_block(sign, x) - x * c) * (sigma / e);
    for _ in 1..degree {
        let sigma_next = 1.0 / (tau - sigma);
        let next = (a.mul_block(sign, &cur) - &cur * c) * (2.0 * sigma_next / e)
            - &prev * (sigma * sigma_next);
        prev = cur;
        cur = next;
        sigma = sigma_next;
    }
    cur
}

/// Lowest eigenvalue estimate for the part of the spectrum the block is missing
///
/// Rayleigh-Ritz residuals are orthogonal to the block, so Rayleigh-Ritz on the span
/// of the significant residual vectors estimates the smallest eigenvalue among the
/// contaminating components. Negligible residuals are dropped first: orthonormalizing
/// them would inject rounding noise that carries components of every eigenvalue.
fn residual_spectrum_min(a: &CsrMatrix, sign: f64, residuals: &DMatrix<f64>) -> Option<f64> {
    let norms: Vec<f64> = residuals.column_iter().map(|c| c.norm()).collect();
    let largest = norms.iter().copied().fold(0.0, f64::max);
    let keep: Vec<usize> = (0..norms.len())
        .filter(|&c| norms[c] > 1e-3 * largest)
        .collect();
    if keep.is_empty() || largest <= f64::MIN_POSITIVE {
        return None;
    }

    let r = DMatrix::from_fn(residuals.nrows(), keep.len(), |i, j| {
        residuals[(i, keep[j])]
    });
    let q = orthonormalize(r);
    let h = q.transpose() * a.mul_block(sign, &q);
    let h = (&h + h.transpose()) * 0.5;
    SymmetricEigen::new(h)
        .eigenvalues
        .iter()
        .copied()
        .min_by(|x, y| x.total_cmp(y))
}

/// Upper bound on the spectrum of `sign · A` from a short Lanczos run
///
/// Returns `θ_max + |β_m|`, the largest Ritz value plus the norm of the final
/// Lanczos residual (Zhou & Li, 2011). Uses full reorthogonalization; a breakdown
/// (invariant subspace) ends the run early.
fn lanczos_upper_bound(a: &CsrMatrix, sign: f64, start: &[f64], steps: usize) -> f64 {
    let n = a.n();
    let steps = steps.min(n).max(1);

    let norm = start.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut basis: Vec<Vec<f64>> = vec![start.iter().map(|v| v / norm).collect()];
    let mut alphas = Vec::with_capacity(steps);
    let mut betas: Vec<f64> = Vec::with_capacity(steps);
    let mut w = vec![0.0; n];

    for j in 0..steps {
        a.mul_vec(sign, &basis[j], &mut w);
        let alpha: f64 = w.iter().zip(&basis[j]).map(|(x, y)| x * y).sum();
        alphas.push(alpha);
        for q in &basis {
            let proj: f64 = w.iter().zip(q).map(|(x, y)| x * y).sum();
            w.iter_mut().zip(q).for_each(|(x, y)| *x -= proj * y);
        }
        let beta = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        betas.push(beta);
        if beta <= 1e-12 * alpha.abs().max(1.0) || j + 1 == steps {
            break;
        }
        basis.push(w.iter().map(|v| v / beta).collect());
    }

    let m = alphas.len();
    let t = DMatrix::from_fn(m, m, |i, j| {
        if i == j {
            alphas[i]
        } else if i + 1 == j {
            betas[i]
        } else if j + 1 == i {
            betas[j]
        } else {
            0.0
        }
    });
    let theta_max = SymmetricEigen::new(t)
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::MIN, f64::max);
    theta_max + betas[m - 1].abs()
}

/// Orthonormal basis for the column space (thin Householder QR)
fn orthonormalize(y: DMatrix<f64>) -> DMatrix<f64> {
    nalgebra::linalg::QR::new(y).q()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_laplacian(n: usize) -> CsrMatrix {
        let mut t = Vec::new();
        for i in 0..n {
            let deg = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
            t.push((i, i, deg));
            if i + 1 < n {
                t.push((i, i + 1, -1.0));
                t.push((i + 1, i, -1.0));
            }
        }
        CsrMatrix::from_triplets(n, t)
    }

    fn check_pairs(a: &CsrMatrix, pairs: &[(f64, Vec<f64>)], tol: f64) {
        let dense = a.to_dense();
        let v = DMatrix::from_fn(a.n(), pairs.len(), |r, c| pairs[c].1[r]);
        let gram = v.transpose() * &v;
        assert!((gram - DMatrix::identity(pairs.len(), pairs.len())).norm() < 1e-8);
        for (lambda, vec) in pairs {
            let x = nalgebra::DVector::from_column_slice(vec);
            assert!((&dense * &x - &x * *lambda).norm() < tol);
        }
    }

    #[test]
    fn test_from_triplets_sums_duplicates() {
        let m = CsrMatrix::from_triplets(2, vec![(0, 1, 1.0), (1, 0, 1.0), (0, 1, 2.0)]);
        let d = m.to_dense();
        assert_eq!(d[(0, 1)], 3.0);
        assert_eq!(d[(1, 0)], 1.0);
        assert_eq!(m.row_sums(), vec![3.0, 1.0]);
    }

    #[test]
    fn test_path_laplacian_smallest_matches_formula() {
        // P_n Laplacian eigenvalues: 2 - 2 cos(pi j / n)
        let n = 400;
        let a = path_laplacian(n);
        let pairs =
            symmetric_extremal_eigenpairs(&a, 5, Extremal::Smallest, &SubspaceOptions::default())
                .unwrap();
        for (j, (lambda, _)) in pairs.iter().enumerate() {
            let expected = 2.0 - 2.0 * (std::f64::consts::PI * j as f64 / n as f64).cos();
            assert!(
                (lambda - expected).abs() < 1e-9,
                "{} vs {}",
                lambda,
                expected
            );
        }
        check_pairs(&a, &pairs, 1e-8);
    }

    #[test]
    fn test_path_laplacian_largest_matches_formula() {
        let n = 300;
        let a = path_laplacian(n);
        let pairs =
            symmetric_extremal_eigenpairs(&a, 4, Extremal::Largest, &SubspaceOptions::default())
                .unwrap();
        for (j, (lambda, _)) in pairs.iter().enumerate() {
            let idx = (n - 1 - j) as f64;
            let expected = 2.0 - 2.0 * (std::f64::consts::PI * idx / n as f64).cos();
            assert!((lambda - expected).abs() < 1e-9);
        }
        check_pairs(&a, &pairs, 1e-8);
    }

    #[test]
    fn test_repeated_eigenvalues_from_components() {
        // Five disjoint triangles: Laplacian eigenvalue 0 with multiplicity 5
        let mut t = Vec::new();
        for c in 0..5 {
            let base = 3 * c;
            for i in 0..3 {
                t.push((base + i, base + i, 2.0));
                for j in 0..3 {
                    if i != j {
                        t.push((base + i, base + j, -1.0));
                    }
                }
            }
        }
        // Pad with a long path so the matrix is large and sparse
        let offset = 15;
        let path = path_laplacian(200);
        for i in 0..200 {
            for (j, v) in path.row(i) {
                t.push((offset + i, offset + j, v));
            }
        }
        let a = CsrMatrix::from_triplets(215, t);
        let pairs =
            symmetric_extremal_eigenpairs(&a, 7, Extremal::Smallest, &SubspaceOptions::default())
                .unwrap();
        // 5 triangles + 1 path component give six zeros; then the path's λ₂
        for (lambda, _) in &pairs[..6] {
            assert!(lambda.abs() < 1e-9);
        }
        let expected = 2.0 - 2.0 * (std::f64::consts::PI / 200.0).cos();
        assert!((pairs[6].0 - expected).abs() < 1e-9);
        check_pairs(&a, &pairs, 1e-8);
    }

    #[test]
    fn test_matches_dense_solver() {
        // Deterministic pseudo-random sparse symmetric matrix
        let n = 300;
        let mut state: u64 = 42;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        let mut t = Vec::new();
        for i in 0..n {
            for _ in 0..3 {
                let j = (next() * n as f64) as usize % n;
                let v = next() - 0.5;
                t.push((i, j, v));
                t.push((j, i, v));
            }
        }
        let a = CsrMatrix::from_triplets(n, t);
        let mut dense: Vec<f64> = SymmetricEigen::new(a.to_dense())
            .eigenvalues
            .iter()
            .copied()
            .collect();
        dense.sort_by(|x, y| x.total_cmp(y));

        let opts = SubspaceOptions::default();
        let small = symmetric_extremal_eigenpairs(&a, 6, Extremal::Smallest, &opts).unwrap();
        let large = symmetric_extremal_eigenpairs(&a, 6, Extremal::Largest, &opts).unwrap();
        for i in 0..6 {
            assert!((small[i].0 - dense[i]).abs() < 1e-9);
            assert!((large[i].0 - dense[n - 1 - i]).abs() < 1e-9);
        }
        check_pairs(&a, &small, 1e-8);
        check_pairs(&a, &large, 1e-8);
    }

    #[test]
    fn test_block_inside_highly_repeated_eigenvalue() {
        // 50 disjoint K_10: Laplacian eigenvalues 0 (x50) and 10 (x450); both ends
        // are repeated far beyond the block size
        let mut t = Vec::new();
        for c in 0..50 {
            for i in 0..10 {
                t.push((10 * c + i, 10 * c + i, 9.0));
                for j in 0..10 {
                    if i != j {
                        t.push((10 * c + i, 10 * c + j, -1.0));
                    }
                }
            }
        }
        let a = CsrMatrix::from_triplets(500, t);
        let opts = SubspaceOptions {
            max_iter: 200,
            ..SubspaceOptions::default()
        };
        let large = symmetric_extremal_eigenpairs(&a, 6, Extremal::Largest, &opts).unwrap();
        let small = symmetric_extremal_eigenpairs(&a, 6, Extremal::Smallest, &opts).unwrap();
        assert!(large.iter().all(|(v, _)| (v - 10.0).abs() < 1e-9));
        assert!(small.iter().all(|(v, _)| v.abs() < 1e-9));
        check_pairs(&a, &large, 1e-8);
        check_pairs(&a, &small, 1e-8);
    }

    #[test]
    fn test_clique_adjacency_repeated_minus_one() {
        // 50 disjoint K_10 adjacency: eigenvalues 9 (x50) and -1 (x450)
        let mut t = Vec::new();
        for c in 0..50 {
            for i in 0..10 {
                for j in 0..10 {
                    if i != j {
                        t.push((10 * c + i, 10 * c + j, 1.0));
                    }
                }
            }
        }
        let a = CsrMatrix::from_triplets(500, t);
        let opts = SubspaceOptions {
            max_iter: 200,
            ..SubspaceOptions::default()
        };
        let small = symmetric_extremal_eigenpairs(&a, 6, Extremal::Smallest, &opts).unwrap();
        assert!(small.iter().all(|(v, _)| (v + 1.0).abs() < 1e-9));
        check_pairs(&a, &small, 1e-8);
    }

    #[test]
    fn test_null_space_larger_than_block() {
        // 300 isolated nodes plus a 200-node path: λ = 0 has multiplicity 301, and the
        // path's λ₂ ≈ 2.5e-4 sits just above it
        let path = path_laplacian(200);
        let mut t = Vec::new();
        for i in 0..200 {
            for (j, v) in path.row(i) {
                t.push((300 + i, 300 + j, v));
            }
        }
        let a = CsrMatrix::from_triplets(500, t);
        let opts = SubspaceOptions {
            max_iter: 1000,
            ..SubspaceOptions::default()
        };
        let small = symmetric_extremal_eigenpairs(&a, 2, Extremal::Smallest, &opts).unwrap();
        assert!(small.iter().all(|(v, _)| v.abs() < 1e-9));
        check_pairs(&a, &small, 1e-8);
    }

    #[test]
    fn test_lanczos_upper_bound_is_an_upper_bound() {
        let a = path_laplacian(500);
        let start: Vec<f64> = (0..500).map(|i| ((i * 37 % 101) as f64) - 50.0).collect();
        let bound = lanczos_upper_bound(&a, 1.0, &start, 20);
        let lambda_max = 2.0 - 2.0 * (std::f64::consts::PI * 499.0 / 500.0).cos();
        assert!(bound >= lambda_max - 1e-12);
    }

    #[test]
    fn test_zero_matrix_and_k_zero() {
        let a = CsrMatrix::from_triplets(50, Vec::new());
        let opts = SubspaceOptions::default();
        let pairs = symmetric_extremal_eigenpairs(&a, 3, Extremal::Largest, &opts).unwrap();
        assert!(pairs.iter().all(|(v, _)| *v == 0.0));
        assert!(
            symmetric_extremal_eigenpairs(&a, 0, Extremal::Smallest, &opts)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_deterministic() {
        let a = path_laplacian(250);
        let opts = SubspaceOptions::default();
        let p1 = symmetric_extremal_eigenpairs(&a, 3, Extremal::Smallest, &opts).unwrap();
        let p2 = symmetric_extremal_eigenpairs(&a, 3, Extremal::Smallest, &opts).unwrap();
        assert_eq!(p1, p2);
    }
}
