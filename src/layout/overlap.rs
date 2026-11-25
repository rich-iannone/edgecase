// Pushing drawn nodes apart until no two circles overlap
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::f64::consts::PI;

/// The nodes in each cell of a grid, in node order: cell `key` holds
/// `nodes[range.0..range.1]` for `cells[key] == range`
struct Cells {
    cells: FxHashMap<(i64, i64), (usize, usize)>,
    nodes: Vec<usize>,
}

/// A node's cell. This truncates toward zero rather than flooring, so the cells either side of
/// each axis are twice as wide
fn cell_of(x: f64, y: f64, cell: f64) -> (i64, i64) {
    ((x / cell) as i64, (y / cell) as i64)
}

impl Cells {
    fn new(xs: &[f64], ys: &[f64], cell: f64) -> Self {
        let keys: Vec<(i64, i64)> = (0..xs.len()).map(|i| cell_of(xs[i], ys[i], cell)).collect();
        let mut nodes: Vec<usize> = (0..xs.len()).collect();
        // Stable, so each cell keeps its nodes in node order
        nodes.sort_by_key(|&i| keys[i]);
        let mut cells = FxHashMap::default();
        let mut start = 0;
        for end in 1..=nodes.len() {
            if end == nodes.len() || keys[nodes[end]] != keys[nodes[start]] {
                cells.insert(keys[nodes[start]], (start, end));
                start = end;
            }
        }
        Cells { cells, nodes }
    }

    fn get(&self, key: (i64, i64)) -> &[usize] {
        match self.cells.get(&key) {
            Some(&(start, end)) => &self.nodes[start..end],
            None => &[],
        }
    }
}

/// Push overlapping nodes apart until no circles overlap
///
/// Each pass visits the nodes in order and pushes each overlapping pair apart along the line
/// between their centers, by a little more than half the overlap each, moving them as it goes.
/// It stops after a pass that moves nothing, or after `max_passes`.
///
/// # Arguments
/// * `xs`, `ys` - Node centers, by node index
/// * `radii` - Node radii, by node index
/// * `order` - A rank for each node: a pair is pushed apart when the node of lower rank is
///   visited
/// * `rng` - An object with a `uniform(a, b)` method (such as `random.Random`), which picks the
///   direction to push two nodes at the same point
/// * `padding` - Space to leave between circles
/// * `max_passes` - The most passes to make
///
/// # Returns
/// The new `(xs, ys)`
#[pyfunction]
#[pyo3(signature = (xs, ys, radii, order, rng, padding=2.0, max_passes=50))]
pub fn remove_overlaps(
    mut xs: Vec<f64>,
    mut ys: Vec<f64>,
    radii: Vec<f64>,
    order: Vec<usize>,
    rng: &Bound<'_, PyAny>,
    padding: f64,
    max_passes: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n = xs.len();
    if ys.len() != n || radii.len() != n || order.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xs, ys, radii and order must have the same length",
        ));
    }
    if xs.iter().chain(&ys).any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "node positions must be finite",
        ));
    }
    if n < 2 {
        return Ok((xs, ys));
    }

    // Wider than any pair's least distance, so overlapping nodes are in neighboring cells
    let max_r = radii.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let cell = ((max_r + padding) * 2.2).max(1.0);

    for _ in 0..max_passes {
        let mut moved = false;
        // Where the nodes were at the start of the pass
        let cells = Cells::new(&xs, &ys, cell);

        for i in 0..n {
            let (cx, cy) = cell_of(xs[i], ys[i], cell);
            for dcx in -1..=1 {
                for dcy in -1..=1 {
                    for &j in cells.get((cx + dcx, cy + dcy)) {
                        if order[j] <= order[i] {
                            continue;
                        }
                        let (dx, dy) = (xs[i] - xs[j], ys[i] - ys[j]);
                        let mut dist = (dx * dx + dy * dy).sqrt();
                        let min_dist = radii[i] + radii[j] + padding;
                        if dist >= min_dist {
                            continue;
                        }
                        let (ux, uy) = if dist < 0.01 {
                            let angle: f64 =
                                rng.call_method1("uniform", (0.0, 2.0 * PI))?.extract()?;
                            dist = 0.01;
                            (angle.cos(), angle.sin())
                        } else {
                            (dx / dist, dy / dist)
                        };
                        let push = (min_dist - dist) * 0.55;
                        xs[i] += ux * push;
                        ys[i] += uy * push;
                        xs[j] -= ux * push;
                        ys[j] -= uy * push;
                        moved = true;
                    }
                }
            }
        }

        if !moved {
            break;
        }
    }
    Ok((xs, ys))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cells_hold_their_nodes_in_order() {
        let xs = vec![5.0, -5.0, 25.0, 6.0, -0.5];
        let ys = vec![5.0, 5.0, 5.0, 9.0, 0.5];
        let cells = Cells::new(&xs, &ys, 10.0);
        // -5 and -0.5 truncate to cell 0 along with 5 and 6
        assert_eq!(cells.get((0, 0)), &[0, 1, 3, 4]);
        assert_eq!(cells.get((2, 0)), &[2]);
        assert!(cells.get((1, 0)).is_empty());
    }
}
