// Packing the laid-out pieces of a graph (its components) in rows to fill a frame
use pyo3::prelude::*;

/// A running total that comes out as Python's `sum()` of the same floats does: Neumaier's
/// compensated sum from Python 3.12, and a plain one before. The packing turns on comparing
/// such totals with each other, so they have to agree to the last bit
struct PySum {
    total: f64,
    compensation: f64,
    compensated: bool,
}

impl PySum {
    fn new(compensated: bool) -> Self {
        PySum {
            total: 0.0,
            compensation: 0.0,
            compensated,
        }
    }

    fn add(&mut self, x: f64) {
        let t = self.total + x;
        if self.compensated {
            self.compensation += if self.total.abs() >= x.abs() {
                (self.total - t) + x
            } else {
                (x - t) + self.total
            };
        }
        self.total = t;
    }

    fn value(&self) -> f64 {
        if self.compensation != 0.0 && self.compensation.is_finite() {
            self.total + self.compensation
        } else {
            self.total
        }
    }
}

fn py_sum(values: &[f64], compensated: bool) -> f64 {
    let mut sum = PySum::new(compensated);
    for &v in values {
        sum.add(v);
    }
    sum.value()
}

/// Python's `max()`: the first of the greatest
fn py_max(values: &[f64]) -> f64 {
    values[1..]
        .iter()
        .fold(values[0], |m, &v| if v > m { v } else { m })
}

/// How rows of pieces fit a frame: the scale they can be drawn at, or `None` once it's clear
/// they can't be drawn larger than `to_beat`
struct Shelving {
    widths: Vec<f64>,
    heights: Vec<f64>,
    gap: f64,
    frame: (f64, f64),
    compensated: bool,
}

/// A bound on a scale, from totals that can only grow, needs this much slack to hold when the
/// height total is compensated and so isn't exactly monotone
const SLACK: f64 = 1e-6;

impl Shelving {
    /// Shelve the pieces in order into rows no wider than `row_width` (a row always takes one
    /// piece), returning the row lengths and the scale
    fn shelve(&self, row_width: f64, to_beat: f64) -> Option<(Vec<usize>, f64)> {
        let (frame_w, frame_h) = self.frame;
        let gap = self.gap;
        let mut rows = Vec::new();
        let mut total_w = f64::NEG_INFINITY;
        let mut heights = PySum::new(self.compensated);

        // Adds the row `start..end` to the totals, and tells whether the scale might still beat
        // `to_beat`
        let mut close = |start: usize, end: usize, rows: &mut Vec<usize>| {
            let row_w =
                py_sum(&self.widths[start..end], self.compensated) + gap * (end - start - 1) as f64;
            if row_w > total_w {
                total_w = row_w;
            }
            heights.add(py_max(&self.heights[start..end]));
            rows.push(end - start);
            let total_h = heights.value() + gap * (rows.len() - 1) as f64;
            frame_w / total_w.max(1e-9) > to_beat
                && frame_h / total_h.max(1e-9) * (1.0 + SLACK) > to_beat
        };

        let mut start = 0;
        let mut used = 0.0;
        for k in 0..self.widths.len() {
            let w = self.widths[k];
            if k > start && used + gap + w > row_width {
                if !close(start, k, &mut rows) {
                    return None;
                }
                start = k;
                used = 0.0;
            }
            used += (if k > start { gap } else { 0.0 }) + w;
        }
        if !close(start, self.widths.len(), &mut rows) {
            return None;
        }

        let total_h = heights.value() + gap * (rows.len() - 1) as f64;
        let scale = (frame_w / total_w.max(1e-9)).min(frame_h / total_h.max(1e-9));
        Some((rows, scale))
    }
}

/// Shelve pieces in rows to fill a frame, trying each row width that ends a row after some
/// piece and keeping the one that draws the whole largest
///
/// # Arguments
/// * `widths`, `heights` - The pieces' sizes, in the order they're shelved
/// * `gap` - Space between pieces, and between rows
/// * `frame_w`, `frame_h` - The frame to fill
/// * `compensated` - Whether to add as Python 3.12 and later do (see `PySum`), so that the
///   rows are those of the Python version of this
///
/// # Returns
/// The number of pieces in each row, top to bottom
#[pyfunction]
#[pyo3(signature = (widths, heights, gap, frame_w, frame_h, compensated))]
pub fn shelve_pieces(
    widths: Vec<f64>,
    heights: Vec<f64>,
    gap: f64,
    frame_w: f64,
    frame_h: f64,
    compensated: bool,
) -> PyResult<Vec<usize>> {
    if widths.len() != heights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "widths and heights must have the same length",
        ));
    }
    if widths
        .iter()
        .chain(&heights)
        .any(|v| !(v.is_finite() && *v >= 0.0))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "piece sizes must be finite and not negative",
        ));
    }
    if widths.is_empty() {
        return Ok(Vec::new());
    }

    // The width of the first k + 1 pieces in a row, for each k
    let mut candidates = Vec::with_capacity(widths.len());
    let mut sum = PySum::new(compensated);
    for (k, &w) in widths.iter().enumerate() {
        sum.add(w);
        candidates.push(sum.value() + gap * k as f64);
    }
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();

    let shelving = Shelving {
        widths,
        heights,
        gap,
        frame: (frame_w, frame_h),
        compensated,
    };
    let mut best: Option<Vec<usize>> = None;
    let mut best_scale = -1.0;
    for (c, &row_width) in candidates.iter().enumerate() {
        let to_beat = best_scale * (1.0 + 1e-9);
        // The first row holds at least the pieces that the narrower width before this one
        // fits (they fall short of this width by a whole piece and gap), so the whole is at
        // least that wide
        if c > 0
            && best.is_some()
            && frame_w / candidates[c - 1].max(1e-9) * (1.0 + SLACK) <= to_beat
        {
            continue;
        }
        if let Some((rows, scale)) = shelving.shelve(row_width, to_beat) {
            if scale > to_beat {
                best = Some(rows);
                best_scale = scale;
            }
        }
    }
    Ok(best.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compensated_sum_matches_python() {
        // Python 3.12 and later: sum([1e16, 1.0, -1e16]) == 1.0, where a plain sum gives 0.0
        assert_eq!(py_sum(&[1e16, 1.0, -1e16], true), 1.0);
        assert_eq!(py_sum(&[1e16, 1.0, -1e16], false), 0.0);
        assert_eq!(py_sum(&[0.1; 10], true), 1.0);
        assert_eq!(py_sum(&[0.1; 10], false), 0.9999999999999999);
        assert_eq!(py_sum(&[], true), 0.0);
    }
}
