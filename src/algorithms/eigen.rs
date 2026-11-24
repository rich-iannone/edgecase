//! Eigenvalues of general (non-symmetric) real matrices
//!
//! nalgebra's `Schur` decomposition uses a purely relative deflation test and no
//! exceptional shifts, so it never converges on matrices that are common in graph
//! analysis: the adjacency matrix of a directed cycle (a permutation matrix) or of a
//! DAG (nilpotent, zero diagonal). This module follows the classic EISPACK pipeline
//! instead:
//!
//! 1. Isolate eigenvalues exposed by zero rows/columns (sources and sinks). For DAGs
//!    this peels away the whole matrix and yields exact zeros.
//! 2. Balance the remaining block with power-of-two diagonal scaling (`balanc`).
//! 3. Reduce to upper Hessenberg form (Householder, via nalgebra).
//! 4. Run the Francis double-shift QR iteration with exceptional shifts (`hqr`).
//!
//! It also has `symmetric_eigen()`, a replacement for nalgebra's symmetric
//! eigendecomposition, which can mismatch eigenvalues and eigenvectors.

use nalgebra::linalg::SymmetricTridiagonal;
use nalgebra::{Complex, DMatrix, Dyn, SymmetricEigen};

/// Maximum QR iterations spent on a single eigenvalue before giving up
const MAX_ITERATIONS_PER_EIGENVALUE: usize = 60;

/// Compute all eigenvalues of a square real matrix
///
/// Returns `None` if the QR iteration fails to converge. The eigenvalues are returned
/// in no particular order; complex eigenvalues appear in conjugate pairs.
pub fn general_eigenvalues(matrix: &DMatrix<f64>) -> Option<Vec<Complex<f64>>> {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "matrix must be square");

    let mut eigenvalues = Vec::with_capacity(n);
    let active = isolate_eigenvalues(matrix, &mut eigenvalues);
    if active.is_empty() {
        return Some(eigenvalues);
    }

    let m = active.len();
    let mut block = DMatrix::<f64>::from_fn(m, m, |i, j| matrix[(active[i], active[j])]);
    balance(&mut block);
    let hessenberg = nalgebra::linalg::Hessenberg::new(block).h();

    eigenvalues.extend(hessenberg_eigenvalues(&hessenberg)?);
    Some(eigenvalues)
}

/// Peel off indices whose row or column is zero off the diagonal
///
/// Removing such an index from the principal submatrix leaves a block-triangular
/// structure, so its diagonal entry is an exact eigenvalue. Returns the indices of
/// the block that still needs an iterative solve.
fn isolate_eigenvalues(matrix: &DMatrix<f64>, eigenvalues: &mut Vec<Complex<f64>>) -> Vec<usize> {
    let mut active: Vec<usize> = (0..matrix.nrows()).collect();

    loop {
        let isolated = active.iter().position(|&i| {
            let row_zero = active.iter().all(|&j| j == i || matrix[(i, j)] == 0.0);
            let col_zero = active.iter().all(|&j| j == i || matrix[(j, i)] == 0.0);
            row_zero || col_zero
        });

        match isolated {
            Some(pos) => {
                let i = active.remove(pos);
                eigenvalues.push(Complex::new(matrix[(i, i)], 0.0));
            }
            None => return active,
        }
    }
}

/// Balance a matrix in place with power-of-two diagonal similarity scaling
///
/// Port of EISPACK `balanc` (without the permutation step, handled by
/// `isolate_eigenvalues`). Scaling by powers of two introduces no rounding error and
/// improves the accuracy of the subsequent QR iteration.
fn balance(a: &mut DMatrix<f64>) {
    const RADIX: f64 = 2.0;
    let n = a.nrows();
    let radix_sq = RADIX * RADIX;

    let mut done = false;
    while !done {
        done = true;
        for i in 0..n {
            let mut c = 0.0;
            let mut r = 0.0;
            for j in 0..n {
                if j != i {
                    c += a[(j, i)].abs();
                    r += a[(i, j)].abs();
                }
            }
            if c == 0.0 || r == 0.0 {
                continue;
            }

            let s = c + r;
            let mut f = 1.0;
            let mut g = r / RADIX;
            while c < g {
                f *= RADIX;
                c *= radix_sq;
            }
            g = r * RADIX;
            while c > g {
                f /= RADIX;
                c /= radix_sq;
            }

            if (c + r) / f < 0.95 * s {
                done = false;
                let g = 1.0 / f;
                for j in 0..n {
                    a[(i, j)] *= g;
                }
                for j in 0..n {
                    a[(j, i)] *= f;
                }
            }
        }
    }
}

/// Eigenvalues of an upper Hessenberg matrix via Francis double-shift QR
///
/// Port of EISPACK `hqr`. Indices are 1-based internally to mirror the reference
/// algorithm exactly.
// Index loops mirror the published 1-based hqr algorithm, which addresses several
// rows at once; iterator rewrites would obscure it.
#[allow(clippy::needless_range_loop)]
fn hessenberg_eigenvalues(h: &DMatrix<f64>) -> Option<Vec<Complex<f64>>> {
    let n = h.nrows();
    let mut a = vec![vec![0.0_f64; n + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=n {
            a[i][j] = h[(i - 1, j - 1)];
        }
    }
    let mut wr = vec![0.0_f64; n + 1];
    let mut wi = vec![0.0_f64; n + 1];

    let mut anorm = 0.0;
    for i in 1..=n {
        for j in i.saturating_sub(1).max(1)..=n {
            anorm += a[i][j].abs();
        }
    }

    let mut nn = n as isize;
    let mut t = 0.0;
    while nn >= 1 {
        let mut its = 0;
        loop {
            let nu = nn as usize;

            // Look for a single small subdiagonal element
            let mut l = nu;
            while l >= 2 {
                let mut s = a[l - 1][l - 1].abs() + a[l][l].abs();
                if s == 0.0 {
                    s = anorm;
                }
                if a[l][l - 1].abs() + s == s {
                    a[l][l - 1] = 0.0;
                    break;
                }
                l -= 1;
            }

            let mut x = a[nu][nu];
            if l == nu {
                // One root found
                wr[nu] = x + t;
                wi[nu] = 0.0;
                nn -= 1;
            } else {
                let mut y = a[nu - 1][nu - 1];
                let mut w = a[nu][nu - 1] * a[nu - 1][nu];
                if l == nu - 1 {
                    // Two roots found
                    let p = 0.5 * (y - x);
                    let q = p * p + w;
                    let mut z = q.abs().sqrt();
                    x += t;
                    if q >= 0.0 {
                        z = p + z.copysign(p);
                        wr[nu - 1] = x + z;
                        wr[nu] = x + z;
                        if z != 0.0 {
                            wr[nu] = x - w / z;
                        }
                        wi[nu - 1] = 0.0;
                        wi[nu] = 0.0;
                    } else {
                        wr[nu - 1] = x + p;
                        wr[nu] = x + p;
                        wi[nu - 1] = -z;
                        wi[nu] = z;
                    }
                    nn -= 2;
                } else {
                    // No roots found yet; continue iterating
                    if its == MAX_ITERATIONS_PER_EIGENVALUE {
                        return None;
                    }
                    if its > 0 && its % 10 == 0 {
                        // Exceptional shift to break cycles (e.g. permutation matrices)
                        t += x;
                        for i in 1..=nu {
                            a[i][i] -= x;
                        }
                        let s = a[nu][nu - 1].abs() + a[nu - 1][nu - 2].abs();
                        x = 0.75 * s;
                        y = x;
                        w = -0.4375 * s * s;
                    }
                    its += 1;

                    // Form shift and look for two consecutive small subdiagonal elements
                    let (mut p, mut q, mut r);
                    let mut m = nu - 2;
                    loop {
                        let z = a[m][m];
                        r = x - z;
                        let s = y - z;
                        p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
                        q = a[m + 1][m + 1] - z - r - s;
                        r = a[m + 2][m + 1];
                        let s = p.abs() + q.abs() + r.abs();
                        p /= s;
                        q /= s;
                        r /= s;
                        if m == l {
                            break;
                        }
                        let u = a[m][m - 1].abs() * (q.abs() + r.abs());
                        let v = p.abs() * (a[m - 1][m - 1].abs() + z.abs() + a[m + 1][m + 1].abs());
                        if u + v == v {
                            break;
                        }
                        m -= 1;
                    }

                    for i in (m + 2)..=nu {
                        a[i][i - 2] = 0.0;
                        if i != m + 2 {
                            a[i][i - 3] = 0.0;
                        }
                    }

                    // Double QR step on rows l..nn and columns m..nn
                    for k in m..nu {
                        if k != m {
                            p = a[k][k - 1];
                            q = a[k + 1][k - 1];
                            r = 0.0;
                            if k != nu - 1 {
                                r = a[k + 2][k - 1];
                            }
                            x = p.abs() + q.abs() + r.abs();
                            if x != 0.0 {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        let s = (p * p + q * q + r * r).sqrt().copysign(p);
                        if s != 0.0 {
                            if k == m {
                                if l != m {
                                    a[k][k - 1] = -a[k][k - 1];
                                }
                            } else {
                                a[k][k - 1] = -s * x;
                            }
                            p += s;
                            x = p / s;
                            y = q / s;
                            let z = r / s;
                            q /= p;
                            r /= p;
                            for j in k..=nu {
                                let mut p = a[k][j] + q * a[k + 1][j];
                                if k != nu - 1 {
                                    p += r * a[k + 2][j];
                                    a[k + 2][j] -= p * z;
                                }
                                a[k + 1][j] -= p * y;
                                a[k][j] -= p * x;
                            }
                            let mmin = nu.min(k + 3);
                            for i in l..=mmin {
                                let mut p = x * a[i][k] + y * a[i][k + 1];
                                if k != nu - 1 {
                                    p += z * a[i][k + 2];
                                    a[i][k + 2] -= p * r;
                                }
                                a[i][k + 1] -= p * q;
                                a[i][k] -= p;
                            }
                        }
                    }
                }
            }

            if nn < 1 || (l as isize) >= nn - 1 {
                break;
            }
        }
    }

    Some((1..=n).map(|i| Complex::new(wr[i], wi[i])).collect())
}

/// Most QL sweeps spent on a single eigenvalue before falling back to nalgebra
const MAX_QL_SWEEPS_PER_EIGENVALUE: usize = 60;

/// Eigenvalues and eigenvectors of a real symmetric matrix
///
/// A drop-in for `nalgebra::SymmetricEigen::new` wherever the eigenvectors are used.
/// nalgebra 0.33 can pair a 2x2 block's eigenvalues with each other's eigenvectors
/// (its issue #1109, fixed in 0.34, which needs a newer Rust than our MSRV): on the
/// football Laplacian, `λ₂` came back with the constant eigenvector. This keeps
/// nalgebra's Householder tridiagonalization and diagonalizes the tridiagonal matrix
/// with the implicit QL iteration (EISPACK's `tql2`), accumulating the rotations into
/// the eigenvectors. Eigenvalues are in no particular order, column `j` of
/// `eigenvectors` belonging to `eigenvalues[j]`.
pub fn symmetric_eigen(matrix: DMatrix<f64>) -> SymmetricEigen<f64, Dyn> {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "matrix must be square");
    if n <= 1 {
        return SymmetricEigen {
            eigenvectors: DMatrix::identity(n, n),
            eigenvalues: matrix.diagonal(),
        };
    }
    let (mut z, mut d, off) = SymmetricTridiagonal::new(matrix.clone()).unpack();
    // e[i] couples d[i] and d[i + 1]; the trailing zero simplifies the sweeps
    let mut e: Vec<f64> = off.iter().copied().chain(std::iter::once(0.0)).collect();
    let norm = (0..n).fold(0.0f64, |acc, i| acc.max(d[i].abs() + e[i].abs()));
    let tol = f64::EPSILON * norm;

    for l in 0..n {
        let mut sweeps = 0;
        loop {
            // Find the first negligible coupling at or after l
            let mut m = l;
            while m + 1 < n && e[m].abs() > tol {
                m += 1;
            }
            if m == l {
                break;
            }
            sweeps += 1;
            if sweeps > MAX_QL_SWEEPS_PER_EIGENVALUE {
                return SymmetricEigen::new(matrix);
            }
            // Wilkinson-style shift from the leading 2x2 block
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut split = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    // The block split during the sweep; start over on the smaller one
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    split = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                // Rotate columns i and i + 1 of the eigenvectors (contiguous, column-major)
                let (left, right) = z.as_mut_slice().split_at_mut((i + 1) * n);
                let (zi, zj) = (&mut left[i * n..], &mut right[..n]);
                for k in 0..n {
                    let f = zj[k];
                    zj[k] = s * zi[k] + c * f;
                    zi[k] = c * zi[k] - s * f;
                }
            }
            if split {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    SymmetricEigen {
        eigenvectors: z,
        eigenvalues: d,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Match computed eigenvalues to expected ones as multisets
    fn assert_spectrum(actual: &[Complex<f64>], expected: &[Complex<f64>], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        let mut remaining: Vec<Complex<f64>> = expected.to_vec();
        for value in actual {
            let (idx, dist) = remaining
                .iter()
                .enumerate()
                .map(|(i, e)| (i, (value - e).norm()))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert!(
                dist < tol,
                "{} not in expected spectrum {:?}",
                value,
                expected
            );
            remaining.remove(idx);
        }
    }

    fn real(values: &[f64]) -> Vec<Complex<f64>> {
        values.iter().map(|&v| Complex::new(v, 0.0)).collect()
    }

    fn cycle(n: usize) -> DMatrix<f64> {
        DMatrix::from_fn(n, n, |i, j| if j == (i + 1) % n { 1.0 } else { 0.0 })
    }

    #[test]
    fn test_one_by_one() {
        let m = DMatrix::from_row_slice(1, 1, &[4.5]);
        assert_spectrum(&general_eigenvalues(&m).unwrap(), &real(&[4.5]), 1e-12);
    }

    #[test]
    fn test_zero_matrix() {
        let m = DMatrix::<f64>::zeros(4, 4);
        assert_spectrum(&general_eigenvalues(&m).unwrap(), &real(&[0.0; 4]), 1e-12);
    }

    #[test]
    fn test_rotation_has_complex_pair() {
        let m = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let expected = vec![Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)];
        assert_spectrum(&general_eigenvalues(&m).unwrap(), &expected, 1e-12);
    }

    #[test]
    fn test_directed_cycles_are_roots_of_unity() {
        for n in 2..=12 {
            let expected: Vec<Complex<f64>> = (0..n)
                .map(|k| Complex::from_polar(1.0, 2.0 * std::f64::consts::PI * k as f64 / n as f64))
                .collect();
            let values = general_eigenvalues(&cycle(n)).expect("cycle should converge");
            assert_spectrum(&values, &expected, 1e-9);
        }
    }

    #[test]
    fn test_nilpotent_jordan_block_is_exact() {
        // Shift matrix: isolated by zero rows/columns, so eigenvalues are exact zeros
        let n = 20;
        let m = DMatrix::from_fn(n, n, |i, j| if j == i + 1 { 1.0 } else { 0.0 });
        for v in general_eigenvalues(&m).unwrap() {
            assert_eq!(v, Complex::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_triangular_reads_diagonal() {
        let m = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 0.0, -4.0, 5.0, 0.0, 0.0, 6.0]);
        assert_spectrum(
            &general_eigenvalues(&m).unwrap(),
            &real(&[1.0, -4.0, 6.0]),
            1e-12,
        );
    }

    #[test]
    fn test_companion_matrix_roots() {
        // (x - 1)(x + 2)(x - 3)(x^2 + 1) = x^5 - 2x^4 - 4x^3 + 4x^2 - 5x + 6
        let coeffs = [-2.0, -4.0, 4.0, -5.0, 6.0];
        let n = coeffs.len();
        let m = DMatrix::from_fn(n, n, |i, j| {
            if i == 0 {
                -coeffs[j]
            } else if j == i - 1 {
                1.0
            } else {
                0.0
            }
        });
        let expected = vec![
            Complex::new(1.0, 0.0),
            Complex::new(-2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, -1.0),
        ];
        assert_spectrum(&general_eigenvalues(&m).unwrap(), &expected, 1e-9);
    }

    #[test]
    fn test_symmetric_matches_symmetric_solver() {
        let n = 12;
        let m = DMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 7 + i * j) % 5) as f64 - 2.0);
        let expected: Vec<f64> = nalgebra::SymmetricEigen::new(m.clone())
            .eigenvalues
            .iter()
            .copied()
            .collect();
        assert_spectrum(&general_eigenvalues(&m).unwrap(), &real(&expected), 1e-8);
    }

    #[test]
    fn test_badly_scaled_matrix() {
        // Similar to diag(1, 2, 3) under a wildly scaled diagonal similarity
        let d = [1e-6, 1.0, 1e6];
        let base = DMatrix::from_row_slice(3, 3, &[1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1e-3, 3.0]);
        let expected: Vec<Complex<f64>> = general_eigenvalues(&base).unwrap();
        let scaled = DMatrix::from_fn(3, 3, |i, j| base[(i, j)] * d[i] / d[j]);
        assert_spectrum(&general_eigenvalues(&scaled).unwrap(), &expected, 1e-8);
    }

    #[test]
    fn test_trace_and_determinant_preserved() {
        // Pseudo-random dense matrix: sum of eigenvalues = trace, product = determinant
        let n = 15;
        let mut state: u64 = 12345;
        let m = DMatrix::from_fn(n, n, |_, _| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        });
        let values = general_eigenvalues(&m).unwrap();
        let sum: Complex<f64> = values.iter().sum();
        let product: Complex<f64> = values.iter().product();
        assert!((sum.re - m.trace()).abs() < 1e-9 && sum.im.abs() < 1e-9);
        let det = m.clone().determinant();
        assert!((product.re - det).abs() < 1e-8 * det.abs().max(1.0) && product.im.abs() < 1e-8);
    }

    /// Largest entries of `M V - V Λ` and `VᵀV - I`
    fn symmetric_errors(m: &DMatrix<f64>) -> (f64, f64) {
        let n = m.nrows();
        let eig = symmetric_eigen(m.clone());
        let v = &eig.eigenvectors;
        let residual = (m * v - v * DMatrix::from_diagonal(&eig.eigenvalues)).amax();
        let orthogonality = (v.transpose() * v - DMatrix::<f64>::identity(n, n)).amax();
        (residual, orthogonality)
    }

    #[test]
    fn test_symmetric_eigen_pairs_values_with_vectors() {
        // nalgebra issue #1109: SymmetricEigen mismatches eigenvalues and eigenvectors
        let m = DMatrix::from_row_slice(
            4,
            4,
            &[
                -19884.07, -10.07188, 11.277279, -188560.63, -10.07188, 12.518197, 1.3770627,
                -102.97504, 11.277279, 1.3770627, 14.587362, 113.26099, -188560.63, -102.97504,
                113.26099, -1788112.3,
            ],
        );
        let (residual, orthogonality) = symmetric_errors(&m);
        assert!(residual < 1e-8 * 1788112.3, "residual {}", residual);
        assert!(orthogonality < 1e-12);

        for n in [1, 2, 3, 8, 40] {
            let a = DMatrix::from_fn(n, n, |i, j| ((i * 7 + j * 3 + i * j) % 11) as f64 - 5.0);
            let (residual, orthogonality) = symmetric_errors(&(&a + a.transpose()));
            assert!(residual < 1e-11 && orthogonality < 1e-12, "n = {}", n);
        }
        // Repeated eigenvalues: the complete graph K6 (5, then -1 five times)
        let k = DMatrix::from_fn(6, 6, |i, j| if i == j { 0.0 } else { 1.0 });
        let (residual, orthogonality) = symmetric_errors(&k);
        assert!(residual < 1e-12 && orthogonality < 1e-12);
        assert_eq!(symmetric_eigen(DMatrix::zeros(0, 0)).eigenvalues.len(), 0);
    }
}
