// The shapes of drawn edges: straight, routed through waypoints, curved and looped. The
// arithmetic follows the Python renderer's operation for operation, so that the coordinates
// round to the same digits
use super::pyfmt::F1;
use std::fmt::Write;

/// An arrowhead: its tip, where the edge meets the target's border, and its other two corners
#[derive(Clone, Copy)]
pub struct Head {
    pub ex: f64,
    pub ey: f64,
    pub p1x: f64,
    pub p1y: f64,
    pub p2x: f64,
    pub p2y: f64,
}

/// Python's `max(a, b)`: `a` unless `b` is greater
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `min(a, b)`: `a` unless `b` is less
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// The middle of an arrowhead's base and its two other corners, for a head of an edge of
/// `width` with its tip at `(ex, ey)`, pointing along `(ux, uy)`
fn arrow(ex: f64, ey: f64, ux: f64, uy: f64, width: f64) -> (f64, f64, f64, f64, f64, f64) {
    // The arrowhead scales with the edge's width
    let arrow_scale = py_max(width / 1.2, 0.6);
    let head_len = 8.0 * arrow_scale;
    let head_wid = 5.0 * arrow_scale;
    let bx = ex - ux * head_len;
    let by = ey - uy * head_len;
    let (ppx, ppy) = (-uy, ux);
    let p1x = bx + ppx * head_wid / 2.0;
    let p1y = by + ppy * head_wid / 2.0;
    let p2x = bx - ppx * head_wid / 2.0;
    let p2y = by - ppy * head_wid / 2.0;
    (bx, by, p1x, p1y, p2x, p2y)
}

/// A straight edge's line, trimmed to the node borders and short of its arrowhead
pub struct Straight {
    pub sx: f64,
    pub sy: f64,
    /// Where the line stops, at the arrowhead's base (or at `(ex, ey)` without one)
    pub lx: f64,
    pub ly: f64,
    /// Where the edge meets the target
    pub ex: f64,
    pub ey: f64,
    pub head: Option<Head>,
}

/// A straight edge, or `None` for one whose ends are at one point
#[allow(clippy::too_many_arguments)]
pub fn straight(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    r_src: f64,
    r_dst: f64,
    width: f64,
    directed: bool,
) -> Option<Straight> {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 1e-6 {
        return None;
    }
    let (ux, uy) = (dx / dist, dy / dist);
    let sx = x1 + ux * r_src;
    let sy = y1 + uy * r_src;
    let ex = x2 - ux * r_dst;
    let ey = y2 - uy * r_dst;
    if !directed {
        return Some(Straight {
            sx,
            sy,
            lx: ex,
            ly: ey,
            ex,
            ey,
            head: None,
        });
    }
    let (bx, by, p1x, p1y, p2x, p2y) = arrow(ex, ey, ux, uy, width);
    Some(Straight {
        sx,
        sy,
        lx: bx,
        ly: by,
        ex,
        ey,
        head: Some(Head {
            ex,
            ey,
            p1x,
            p1y,
            p2x,
            p2y,
        }),
    })
}

/// `waypoints` trimmed to stop at the borders of the end nodes, or `None` for fewer than two
pub fn trimmed_waypoints(
    waypoints: &[(f64, f64)],
    r_src: f64,
    r_dst: f64,
) -> Option<Vec<(f64, f64)>> {
    if waypoints.len() < 2 {
        return None;
    }
    let mut pts = waypoints.to_vec();
    let n = pts.len();

    let dx = pts[1].0 - pts[0].0;
    let dy = pts[1].1 - pts[0].1;
    let d = (dx * dx + dy * dy).sqrt();
    if d > r_src + 1.0 {
        pts[0] = (pts[0].0 + dx / d * r_src, pts[0].1 + dy / d * r_src);
    }
    let dx = pts[n - 1].0 - pts[n - 2].0;
    let dy = pts[n - 1].1 - pts[n - 2].1;
    let d = (dx * dx + dy * dy).sqrt();
    if d > r_dst + 1.0 {
        pts[n - 1] = (pts[n - 1].0 - dx / d * r_dst, pts[n - 1].1 - dy / d * r_dst);
    }
    Some(pts)
}

/// The cubic Bezier (Catmull-Rom) from point `i` of `pts` to the next: its two control points
/// and its end
pub fn catmull_rom(pts: &[(f64, f64)], i: usize) -> [(f64, f64); 3] {
    let n = pts.len();
    let p0 = pts[i.saturating_sub(1)];
    let p1 = pts[i];
    let p2 = pts[(i + 1).min(n - 1)];
    let p3 = pts[(i + 2).min(n - 1)];
    let c1x = p1.0 + (p2.0 - p0.0) / 6.0;
    let c1y = p1.1 + (p2.1 - p0.1) / 6.0;
    let c2x = p2.0 - (p3.0 - p1.0) / 6.0;
    let c2y = p2.1 - (p3.1 - p1.1) / 6.0;
    [(c1x, c1y), (c2x, c2y), p2]
}

/// The `d` of a path through points already trimmed: a line for two, and a smooth run of cubic
/// Beziers (Catmull-Rom) for more
fn waypoints_d(pts: &[(f64, f64)]) -> String {
    let n = pts.len();
    let mut path = format!("M{},{}", F1(pts[0].0), F1(pts[0].1));
    if n == 2 {
        let _ = write!(path, " L{},{}", F1(pts[1].0), F1(pts[1].1));
        return path;
    }
    for i in 0..n - 1 {
        let [c1, c2, p2] = catmull_rom(pts, i);
        let _ = write!(
            path,
            " C{},{} {},{} {},{}",
            F1(c1.0),
            F1(c1.1),
            F1(c2.0),
            F1(c2.1),
            F1(p2.0),
            F1(p2.1)
        );
    }
    path
}

/// A path through `waypoints`, trimmed to stop at the borders of the end nodes: a line for two
/// points, and a smooth run of cubic Beziers (Catmull-Rom) for more. Empty for fewer than two
pub fn waypoints_path(waypoints: &[(f64, f64)], r_src: f64, r_dst: f64) -> String {
    trimmed_waypoints(waypoints, r_src, r_dst).map_or_else(String::new, |pts| waypoints_d(&pts))
}

/// A routed edge's arrowhead, and the middle of its base, where the path stops; `None` when the
/// route's last leg has no length
pub fn routed_head(waypoints: &[(f64, f64)], r_dst: f64, width: f64) -> Option<(Head, f64, f64)> {
    let n = waypoints.len();
    if n < 2 {
        return None;
    }
    let dx = waypoints[n - 1].0 - waypoints[n - 2].0;
    let dy = waypoints[n - 1].1 - waypoints[n - 2].1;
    let d = (dx * dx + dy * dy).sqrt();
    if d <= 1e-6 {
        return None;
    }
    let (ux, uy) = (dx / d, dy / d);
    let ex = waypoints[n - 1].0 - ux * r_dst;
    let ey = waypoints[n - 1].1 - uy * r_dst;
    let (bx, by, p1x, p1y, p2x, p2y) = arrow(ex, ey, ux, uy, width);
    let head = Head {
        ex,
        ey,
        p1x,
        p1y,
        p2x,
        p2y,
    };
    Some((head, bx, by))
}

/// The waypoints of a routed edge's path, as `routed()` draws it through them, stopping short
/// of its arrowhead (`None` for fewer than two), and the arrowhead (`None` for none)
pub fn routed_points(
    waypoints: &[(f64, f64)],
    r_src: f64,
    r_dst: f64,
    width: f64,
    directed: bool,
) -> (Option<Vec<(f64, f64)>>, Option<Head>) {
    match routed_head(waypoints, r_dst, width).filter(|_| directed) {
        None => (trimmed_waypoints(waypoints, r_src, r_dst), None),
        Some((head, bx, by)) => {
            let mut trimmed = waypoints.to_vec();
            *trimmed.last_mut().expect("a head has a route") = (bx, by);
            (trimmed_waypoints(&trimmed, r_src, 0.0), Some(head))
        }
    }
}

/// A routed edge's path, stopping short of its arrowhead, and the arrowhead (`None` for none)
pub fn routed(
    waypoints: &[(f64, f64)],
    r_src: f64,
    r_dst: f64,
    width: f64,
    directed: bool,
) -> (String, Option<Head>) {
    let (pts, head) = routed_points(waypoints, r_src, r_dst, width, directed);
    (pts.map_or_else(String::new, |pts| waypoints_d(&pts)), head)
}

/// A curved edge, a quadratic Bezier bowed out by `bow` of its length, trimmed to the node
/// borders: its start, control point and end, and its arrowhead (`None` for none). `None` for an
/// edge whose ends are at one point
#[allow(clippy::too_many_arguments)]
pub fn curved_points(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    r_src: f64,
    r_dst: f64,
    width: f64,
    directed: bool,
    bow: f64,
) -> Option<([(f64, f64); 3], Option<Head>)> {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 1e-6 {
        return None;
    }
    let (ux, uy) = (dx / dist, dy / dist);
    let (px, py) = (-uy, ux);

    // The control point, out from the middle of the edge
    let mx = (x1 + x2) / 2.0 + px * dist * bow;
    let my = (y1 + y2) / 2.0 + py * dist * bow;

    // The start, toward the control point
    let cdx_s = mx - x1;
    let cdy_s = my - y1;
    let cd_s = (cdx_s * cdx_s + cdy_s * cdy_s).sqrt();
    let (sx, sy) = if cd_s > 1e-6 {
        (x1 + (cdx_s / cd_s) * r_src, y1 + (cdy_s / cd_s) * r_src)
    } else {
        (x1, y1)
    };
    // The end, along the curve's tangent there, so that it meets the border as the curve does
    let tan_ex = x2 - mx;
    let tan_ey = y2 - my;
    let tan_e_len = (tan_ex * tan_ex + tan_ey * tan_ey).sqrt();
    let (ex, ey) = if tan_e_len > 1e-6 {
        let (tex, tey) = (tan_ex / tan_e_len, tan_ey / tan_e_len);
        (x2 - tex * r_dst, y2 - tey * r_dst)
    } else {
        (x2, y2)
    };
    let points = [(sx, sy), (mx, my), (ex, ey)];
    if !directed {
        return Some((points, None));
    }
    let tx = 2.0 * (ex - mx);
    let ty = 2.0 * (ey - my);
    let t_len = (tx * tx + ty * ty).sqrt();
    if t_len <= 1e-6 {
        return Some((points, None));
    }
    let (_, _, p1x, p1y, p2x, p2y) = arrow(ex, ey, tx / t_len, ty / t_len, width);
    let head = Head {
        ex,
        ey,
        p1x,
        p1y,
        p2x,
        p2y,
    };
    Some((points, Some(head)))
}

/// A curved edge, as `curved_points()` gives it: its path's `d` and its arrowhead
#[allow(clippy::too_many_arguments)]
pub fn curved(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    r_src: f64,
    r_dst: f64,
    width: f64,
    directed: bool,
    bow: f64,
) -> Option<(String, Option<Head>)> {
    let ([(sx, sy), (mx, my), (ex, ey)], head) =
        curved_points(x1, y1, x2, y2, r_src, r_dst, width, directed, bow)?;
    let d = format!(
        "M {} {} Q {} {} {} {}",
        F1(sx),
        F1(sy),
        F1(mx),
        F1(my),
        F1(ex),
        F1(ey)
    );
    Some((d, head))
}

// sin(0.45) and cos(0.45), as Python's math module gives them, for the ends of a self-loop
const LOOP_SIN: f64 = 0.43496553411123023;
const LOOP_COS: f64 = 0.9004471023526769;

/// A self-loop above its node, of `loop_size` times the usual size, as `curved()` gives it: its
/// start, then two cubic Beziers as their two control points and their end each, and its
/// arrowhead (`None` for none)
pub fn self_loop_points(
    x: f64,
    y: f64,
    r: f64,
    width: f64,
    directed: bool,
    loop_size: f64,
) -> ([(f64, f64); 7], Option<Head>) {
    let lr = r * 1.2 * loop_size;
    // The arc's center, above the node
    let cx_arc = x;
    let cy_arc = y - r - lr;
    // The ends, on the node's border
    let sx = x + r * -LOOP_SIN;
    let sy = y - r * LOOP_COS;
    let ex = x + r * LOOP_SIN;
    let ey = y - r * LOOP_COS;

    // Two cubic Beziers
    let cp_h = lr * 1.6;
    let c1x = sx - cp_h * 0.3;
    let c1y = sy - cp_h;
    let c2x = cx_arc - lr;
    let c2y = cy_arc;
    let c3x = cx_arc + lr;
    let c3y = cy_arc;
    let c4x = ex + cp_h * 0.3;
    let c4y = ey - cp_h;
    let points = [
        (sx, sy),
        (c1x, c1y),
        (c2x, c2y),
        (cx_arc, cy_arc - lr * 0.2),
        (c3x, c3y),
        (c4x, c4y),
        (ex, ey),
    ];
    if !directed {
        return (points, None);
    }
    let tan_x = ex - c4x;
    let tan_y = ey - c4y;
    let t_len = (tan_x * tan_x + tan_y * tan_y).sqrt();
    if t_len <= 1e-6 {
        return (points, None);
    }
    let (_, _, p1x, p1y, p2x, p2y) = arrow(ex, ey, tan_x / t_len, tan_y / t_len, width);
    let head = Head {
        ex,
        ey,
        p1x,
        p1y,
        p2x,
        p2y,
    };
    (points, Some(head))
}

/// A self-loop, as `self_loop_points()` gives it: its path's `d` and its arrowhead
pub fn self_loop(
    x: f64,
    y: f64,
    r: f64,
    width: f64,
    directed: bool,
    loop_size: f64,
) -> (String, Option<Head>) {
    let (points, head) = self_loop_points(x, y, r, width, directed, loop_size);
    let mut d = format!("M {} {}", F1(points[0].0), F1(points[0].1));
    for (k, (px, py)) in points[1..].iter().enumerate() {
        let _ = write!(
            d,
            " {}{} {}",
            if k % 3 == 0 { "C " } else { "" },
            F1(*px),
            F1(*py)
        );
    }
    (d, head)
}

/// The box `(min x, min y, width, height)` around nodes at `xs`, `ys` of radii `rs`, with
/// `padding` around them, room under them for labels of `label_chars` characters, and widened
/// on its shorter side to the aspect ratio of `frame`
pub fn fit_viewbox(
    xs: &[f64],
    ys: &[f64],
    rs: &[f64],
    label_chars: Option<&[usize]>,
    label_space: f64,
    padding: f64,
    frame: Option<(f64, f64)>,
) -> (f64, f64, f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0, 100.0, 100.0);
    }
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for k in 0..xs.len() {
        let r = rs[k];
        // About 7 pixels a character
        let label_w = label_chars.map_or(0.0, |chars| chars[k] as f64 * 7.0 / 2.0);
        let horiz_extra = py_max(r, label_w) + padding;
        let vert_extra = r + padding;
        min_x = py_min(min_x, xs[k] - horiz_extra);
        min_y = py_min(min_y, ys[k] - vert_extra);
        max_x = py_max(max_x, xs[k] + horiz_extra);
        max_y = py_max(max_y, ys[k] + vert_extra + label_space);
    }
    let mut w = py_max(max_x - min_x, 1.0);
    let mut h = py_max(max_y - min_y, 1.0);
    if let Some((frame_w, frame_h)) = frame {
        let aspect = frame_w / frame_h;
        if w / h < aspect {
            min_x -= (h * aspect - w) / 2.0;
            w = h * aspect;
        } else {
            min_y -= (w / aspect - h) / 2.0;
            h = w / aspect;
        }
    }
    (min_x, min_y, w, h)
}
