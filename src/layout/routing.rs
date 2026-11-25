// Routing drawn edges around the nodes they would pass through
use pyo3::prelude::*;
use rayon::prelude::*;

/// A grid of square cells over the node centers, holding each cell's nodes in index order
struct Grid {
    x0: f64,
    y0: f64,
    cell: f64,
    cols: usize,
    rows: usize,
    // nodes of cell `c` are `nodes[starts[c]..starts[c + 1]]`
    starts: Vec<usize>,
    nodes: Vec<usize>,
}

impl Grid {
    /// `reach` is the farthest any node's clearance extends from its center
    fn new(xs: &[f64], ys: &[f64], reach: f64) -> Self {
        let n = xs.len();
        let fold = |v: &[f64]| {
            v.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &a| {
                    (lo.min(a), hi.max(a))
                })
        };
        let (x0, x1) = fold(xs);
        let (y0, y1) = fold(ys);
        let (w, h) = ((x1 - x0).max(0.0), (y1 - y0).max(0.0));
        // About one node per cell, no cell narrower than the reach (so a node's clearance
        // stays within its cell's neighbors) and at most 2048 cells across
        let mut cell = reach.max((w * h / n as f64).sqrt()).max(w.max(h) / 2048.0);
        if !(cell.is_finite() && cell > 0.0) {
            cell = 1.0;
        }
        let cols = (w / cell) as usize + 1;
        let rows = (h / cell) as usize + 1;

        let mut grid = Grid {
            x0,
            y0,
            cell,
            cols,
            rows,
            starts: vec![0; cols * rows + 1],
            nodes: vec![0; n],
        };
        let cells: Vec<usize> = (0..n)
            .map(|i| grid.row(ys[i]) * cols + grid.col(xs[i]))
            .collect();
        for &c in &cells {
            grid.starts[c + 1] += 1;
        }
        for c in 0..cols * rows {
            grid.starts[c + 1] += grid.starts[c];
        }
        let mut next = grid.starts.clone();
        for (i, &c) in cells.iter().enumerate() {
            grid.nodes[next[c]] = i;
            next[c] += 1;
        }
        grid
    }

    fn col(&self, x: f64) -> usize {
        (((x - self.x0) / self.cell).max(0.0) as usize).min(self.cols - 1)
    }

    fn row(&self, y: f64) -> usize {
        (((y - self.y0) / self.cell).max(0.0) as usize).min(self.rows - 1)
    }

    /// Calls `visit` once for every node whose center is within one cell of the segment
    fn near_segment(
        &self,
        (x1, y1): (f64, f64),
        (x2, y2): (f64, f64),
        mut visit: impl FnMut(usize),
    ) {
        let s = self.cell;
        let (x_lo, x_hi) = (x1.min(x2), x1.max(x2));
        let (y_lo, y_hi) = (y1.min(y2), y1.max(y2));
        let dx = x2 - x1;
        // The segment's y at `x`, kept within its ends
        let y_at = |x: f64| (y1 + (x - x1) * (y2 - y1) / dx).clamp(y_lo, y_hi);

        // One cell beyond what the reach needs on every side, for rounding
        for c in self.col(x_lo - 2.0 * s)..=self.col(x_hi + 2.0 * s) {
            let strip_lo = self.x0 + c as f64 * s - 2.0 * s;
            let strip_hi = self.x0 + (c + 1) as f64 * s + 2.0 * s;
            let (a, b) = (strip_lo.max(x_lo), strip_hi.min(x_hi));
            if a > b {
                // The segment ends more than a cell short of this column
                continue;
            }
            let (lo, hi) = if dx.abs() < 1e-12 {
                (y_lo, y_hi)
            } else {
                let (ya, yb) = (y_at(a), y_at(b));
                (ya.min(yb), ya.max(yb))
            };
            for r in self.row(lo - 2.0 * s)..=self.row(hi + 2.0 * s) {
                let cell = r * self.cols + c;
                for &i in &self.nodes[self.starts[cell]..self.starts[cell + 1]] {
                    visit(i);
                }
            }
        }
    }
}

/// Distance from a point to a segment, with the projection parameter (in [0, 1]) and the side
/// of the segment the point is on (+1 or -1)
fn point_to_segment_dist(
    (px, py): (f64, f64),
    (ax, ay): (f64, f64),
    (bx, by): (f64, f64),
) -> (f64, f64, f64) {
    let (dx, dy) = (bx - ax, by - ay);
    let seg_len2 = dx * dx + dy * dy;
    if seg_len2 < 1e-12 {
        return (((px - ax).powi(2) + (py - ay).powi(2)).sqrt(), 0.0, 1.0);
    }
    let t = (((px - ax) * dx + (py - ay) * dy) / seg_len2).clamp(0.0, 1.0);
    let (proj_x, proj_y) = (ax + t * dx, ay + t * dy);
    let d = ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt();
    let cross = dx * (py - ay) - dy * (px - ax);
    (d, t, if cross >= 0.0 { 1.0 } else { -1.0 })
}

/// The waypoints `[src, wp1, ..., dst]` that take one edge around the nodes in its way, or
/// `None` when the straight line is clear
fn route_edge(
    grid: &Grid,
    xs: &[f64],
    ys: &[f64],
    radii: &[f64],
    (src, dst): (usize, usize),
    margin: f64,
) -> Option<Vec<(f64, f64)>> {
    let (a, b) = ((xs[src], ys[src]), (xs[dst], ys[dst]));

    // (t, node, distance, side), for each node the line passes too close to
    let mut obstacles = Vec::new();
    grid.near_segment(a, b, |i| {
        if i == src || i == dst {
            return;
        }
        let (dist, t, side) = point_to_segment_dist((xs[i], ys[i]), a, b);
        if dist < radii[i] + margin && 0.0 < t && t < 1.0 {
            obstacles.push((t, i, dist, side));
        }
    });
    if obstacles.is_empty() {
        return None;
    }
    // In order along the edge, and in node order where two are level
    obstacles.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));

    let (seg_dx, seg_dy) = (b.0 - a.0, b.1 - a.1);
    let seg_len = (seg_dx * seg_dx + seg_dy * seg_dy).sqrt();
    if seg_len < 1e-6 {
        return None;
    }
    let (perp_x, perp_y) = (-seg_dy / seg_len, seg_dx / seg_len);

    let mut waypoints = Vec::with_capacity(obstacles.len() + 2);
    waypoints.push(a);
    for (t, i, dist, side) in obstacles {
        let (cx, cy) = (xs[i], ys[i]);
        let detour = radii[i] + margin + 2.0;
        let sideways = (cx + perp_x * detour * side, cy + perp_y * detour * side);
        if dist < 1e-6 {
            waypoints.push(sideways);
            continue;
        }
        // Push the waypoint from the node's center through the line, out past its clearance
        let dx_off = cx - (a.0 + t * seg_dx);
        let dy_off = cy - (a.1 + t * seg_dy);
        let off_len = (dx_off * dx_off + dy_off * dy_off).sqrt();
        if off_len < 1e-6 {
            waypoints.push(sideways);
        } else {
            waypoints.push((
                cx - dx_off / off_len * detour,
                cy - dy_off / off_len * detour,
            ));
        }
    }
    waypoints.push(b);
    Some(waypoints)
}

/// Route drawn edges around the nodes they would otherwise pass through
///
/// # Arguments
/// * `xs`, `ys` - Node centers, by node index
/// * `radii` - Node radii, by node index
/// * `edges` - Edges as (source, target) node indices
/// * `margin` - Space to keep between an edge and a node's circle
///
/// # Returns
/// For each edge, its waypoints `[source, wp1, ..., target]`, or `None` when the straight line
/// between its ends doesn't pass within `margin` of another node's circle
#[pyfunction]
#[pyo3(signature = (xs, ys, radii, edges, margin=4.0))]
pub fn route_edges(
    py: Python<'_>,
    xs: Vec<f64>,
    ys: Vec<f64>,
    radii: Vec<f64>,
    edges: Vec<(usize, usize)>,
    margin: f64,
) -> PyResult<Vec<Option<Vec<(f64, f64)>>>> {
    let n = xs.len();
    if ys.len() != n || radii.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xs, ys and radii must have the same length",
        ));
    }
    if xs.iter().chain(&ys).any(|v| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "node positions must be finite",
        ));
    }
    if let Some(&(s, t)) = edges.iter().find(|&&(s, t)| s >= n || t >= n) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "edge ({}, {}) names a node past the last of {}",
            s, t, n
        )));
    }
    if n == 0 {
        return Ok(vec![None; edges.len()]);
    }
    Ok(py.allow_threads(|| {
        let reach = radii.iter().fold(0.0f64, |acc, &r| acc.max(r + margin));
        let grid = Grid::new(&xs, &ys, reach);
        edges
            .par_iter()
            .map(|&edge| route_edge(&grid, &xs, &ys, &radii, edge, margin))
            .collect()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every node close to the line, found by checking them all
    fn brute_force(xs: &[f64], ys: &[f64], reach: f64, a: (f64, f64), b: (f64, f64)) -> Vec<usize> {
        (0..xs.len())
            .filter(|&i| point_to_segment_dist((xs[i], ys[i]), a, b).0 <= reach)
            .collect()
    }

    #[test]
    fn test_grid_finds_every_node_near_a_segment() {
        // A spread of points (with a dense clump, so cells hold several) and segments of every
        // slope, including vertical, horizontal and zero-length ones
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for i in 0..2000 {
            let spread = if i % 4 == 0 { 20.0 } else { 1000.0 };
            xs.push(next() * spread);
            ys.push(next() * spread * 0.6);
        }
        for reach in [0.5, 8.0, 60.0] {
            let grid = Grid::new(&xs, &ys, reach);
            for k in 0..300 {
                let a = (xs[k], ys[k]);
                let b = match k % 5 {
                    0 => (xs[k], ys[k + 1]),
                    1 => (xs[k + 1], ys[k]),
                    2 => a,
                    _ => (xs[k + 7], ys[k + 7]),
                };
                let mut found = Vec::new();
                grid.near_segment(a, b, |i| found.push(i));
                let mut sorted = found.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), found.len(), "a node was visited twice");
                for i in brute_force(&xs, &ys, reach, a, b) {
                    assert!(
                        sorted.binary_search(&i).is_ok(),
                        "missed node {} (reach {})",
                        i,
                        reach
                    );
                }
            }
        }
    }

    #[test]
    fn test_route_edge_goes_around_a_node_in_the_way() {
        let xs = vec![0.0, 100.0, 50.0, 50.0];
        let ys = vec![0.0, 0.0, 1.0, 80.0];
        let radii = vec![5.0; 4];
        let grid = Grid::new(&xs, &ys, 9.0);
        let route = route_edge(&grid, &xs, &ys, &radii, (0, 1), 4.0).unwrap();
        // Only node 2 is in the way, and the detour passes on the side away from its center
        assert_eq!(route.len(), 3);
        assert_eq!(route[0], (0.0, 0.0));
        assert_eq!(route[2], (100.0, 0.0));
        assert!((route[1].0 - 50.0).abs() < 1e-9);
        assert!((route[1].1 - (1.0 - 11.0)).abs() < 1e-9);
        // A clear line needs no route
        assert!(route_edge(&grid, &xs, &ys, &radii, (2, 3), 4.0).is_none());
    }
}
