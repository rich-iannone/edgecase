//! Optimized set intersection for triangle counting
//!
//! This module provides optimized set intersection operations using:
//! - Galloping search for skewed size differences
//! - Optimized scalar merge for balanced sizes
//! - SIMD hints for auto-vectorization

/// Count common elements in two sorted slices where elements > threshold
/// Returns the count of elements that appear in both slices and are > threshold
///
/// Uses adaptive algorithm:
/// - Galloping search when size ratio > 4:1
/// - Optimized scalar merge otherwise
#[inline]
pub fn count_intersection_sorted_above(a: &[u64], b: &[u64], threshold: u64) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    // Use galloping search if one array is much larger
    let size_ratio = if a.len() > b.len() {
        a.len() / b.len().max(1)
    } else {
        b.len() / a.len().max(1)
    };

    if size_ratio >= 4 {
        // Galloping search: binary search small list in large list
        if a.len() < b.len() {
            count_intersection_galloping(a, b, threshold)
        } else {
            count_intersection_galloping(b, a, threshold)
        }
    } else {
        // Balanced sizes: use optimized merge
        count_intersection_merge(a, b, threshold)
    }
}

/// Galloping search: for each element in small list, binary search in large list
#[inline]
fn count_intersection_galloping(small: &[u64], large: &[u64], threshold: u64) -> usize {
    let mut count = 0;
    let mut large_start = 0;

    for &elem in small {
        if elem <= threshold {
            continue;
        }

        // Binary search in remaining portion of large array
        if large_start < large.len() {
            match large[large_start..].binary_search(&elem) {
                Ok(pos) => {
                    count += 1;
                    large_start += pos + 1;
                }
                Err(pos) => {
                    large_start += pos;
                }
            }
        }
    }

    count
}

/// Optimized merge-based intersection
/// This version is optimized for auto-vectorization by the compiler
#[inline]
fn count_intersection_merge(a: &[u64], b: &[u64], threshold: u64) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;

    // Main merge loop - optimized for branch prediction and vectorization
    while i < a.len() && j < b.len() {
        let av = unsafe { *a.get_unchecked(i) };
        let bv = unsafe { *b.get_unchecked(j) };

        // Branchless comparison for better performance
        let equal = av == bv;
        let a_less = av < bv;

        // Count if equal and above threshold
        count += (equal && av > threshold) as usize;

        // Advance indices branchlessly
        i += (equal || a_less) as usize;
        j += (equal || !a_less) as usize;
    }

    count
}

/// Scalar merge-based intersection (safe fallback)
#[allow(dead_code)]
#[inline(always)]
fn count_intersection_scalar(a: &[u64], b: &[u64], threshold: u64) -> usize {
    let mut count = 0;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        let av = a[i];
        let bv = b[j];

        if av == bv && av > threshold {
            count += 1;
            i += 1;
            j += 1;
        } else if av < bv {
            i += 1;
        } else {
            j += 1;
        }
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_intersection() {
        let a = vec![1, 3, 5, 7, 9, 11];
        let b = vec![2, 3, 5, 8, 9, 12];

        // Count elements in both that are > 4
        let result = count_intersection_scalar(&a, &b, 4);
        assert_eq!(result, 2); // 5 and 9 are common and > 4
    }

    #[test]
    fn test_intersection_empty() {
        let a = vec![1, 2, 3];
        let b: Vec<u64> = vec![];

        let result = count_intersection_sorted_above(&a, &b, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_intersection_no_matches() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];

        let result = count_intersection_sorted_above(&a, &b, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_intersection_all_match() {
        let a = vec![5, 10, 15, 20];
        let b = vec![5, 10, 15, 20];

        let result = count_intersection_sorted_above(&a, &b, 0);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_intersection_with_threshold() {
        let a = vec![1, 5, 10, 15, 20];
        let b = vec![1, 5, 10, 15, 20];

        // Only count matches > 7
        let result = count_intersection_sorted_above(&a, &b, 7);
        assert_eq!(result, 3); // 10, 15, 20
    }

    #[test]
    fn test_large_intersection() {
        let a: Vec<u64> = (0..100).map(|x| x * 2).collect();
        let b: Vec<u64> = (0..100).map(|x| x * 3).collect();

        // Find common multiples of 6
        let result = count_intersection_sorted_above(&a, &b, 0);
        assert_eq!(result, 33); // 0, 6, 12, ..., 192
    }
}
