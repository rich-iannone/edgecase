// Which node labels a plot draws: up to a limit, in order of priority (highlighted nodes, then
// larger ones, then those of higher degree), leaving out any that would overlap one drawn before
// it, so that the labels of a large graph can be read. The viewer chooses again, by the same
// rule, for what is in view (see `drawLabels()` in `static/viewer.js`)
use rustc_hash::FxHashMap;

/// With no more nodes than this, a plot draws every label, and otherwise `AUTO_MAX` of them
pub const ALL_MAX: usize = 50;
pub const AUTO_MAX: usize = 30;
/// A character's width, as a share of the font size: about that of the plot's fonts
pub const CHAR_W: f64 = 0.58;
/// The room left between labels
const GAP: f64 = 2.0;

/// How many labels a plot of `n` nodes draws when `max_labels=` doesn't say: None for all
pub fn auto_limit(n: usize) -> Option<usize> {
    (n > ALL_MAX).then_some(AUTO_MAX)
}

/// The nodes in order of priority: by highlight level (`level`, accent first), then radius,
/// then degree, and then in their own order
pub fn priority(level: Option<&[u8]>, r: &[f64], degree: &[usize]) -> Vec<usize> {
    let lv = |k: usize| level.map_or(0, |l| l[k]);
    let mut order: Vec<usize> = (0..r.len()).collect();
    order.sort_by(|&a, &b| {
        lv(b)
            .cmp(&lv(a))
            .then(r[b].total_cmp(&r[a]))
            .then(degree[b].cmp(&degree[a]))
            .then(a.cmp(&b))
    });
    order
}

/// The box (left, top, right, bottom) of a label of `chars` characters in `font`, centered under
/// a node at `x`, `y` of radius `r`, as `draw_labels()` draws it
fn label_box(chars: usize, x: f64, y: f64, r: f64, font: f64) -> [f64; 4] {
    let half = chars as f64 * font * CHAR_W / 2.0;
    let top = y + r + 2.0;
    [x - half, top, x + half, top + font * 1.25]
}

/// Which labels are drawn: all of them when `limit` is None, and otherwise up to `limit` of
/// them, taken in `order`, each one that doesn't overlap one taken before it. An empty label
/// isn't counted
pub fn choose(
    texts: &[String],
    x: &[f64],
    y: &[f64],
    r: &[f64],
    font: f64,
    order: &[usize],
    limit: Option<usize>,
) -> Vec<bool> {
    let Some(limit) = limit else {
        return vec![true; texts.len()];
    };
    let mut shown = vec![false; texts.len()];
    // The boxes taken, by the cells of a grid that they cover
    let cell = (font * 8.0).max(1.0);
    let mut grid: FxHashMap<(i64, i64), Vec<[f64; 4]>> = FxHashMap::default();
    let mut count = 0;
    for &k in order {
        if count == limit {
            break;
        }
        let chars = texts[k].chars().count();
        if chars == 0 {
            continue;
        }
        let b = label_box(chars, x[k], y[k], r[k], font);
        let cells = |b: &[f64; 4]| {
            let (c0, c1) = ((b[0] / cell).floor() as i64, (b[2] / cell).floor() as i64);
            let (r0, r1) = ((b[1] / cell).floor() as i64, (b[3] / cell).floor() as i64);
            (c0..=c1).flat_map(move |c| (r0..=r1).map(move |r| (c, r)))
        };
        let overlaps = |o: &[f64; 4]| {
            b[0] < o[2] + GAP && o[0] < b[2] + GAP && b[1] < o[3] + GAP && o[1] < b[3] + GAP
        };
        if cells(&b).any(|c| grid.get(&c).is_some_and(|boxes| boxes.iter().any(overlaps))) {
            continue;
        }
        for c in cells(&b) {
            grid.entry(c).or_default().push(b);
        }
        shown[k] = true;
        count += 1;
    }
    shown
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn a_plot_draws_every_label_of_a_small_graph() {
        assert_eq!(auto_limit(50), None);
        assert_eq!(auto_limit(51), Some(AUTO_MAX));
    }

    #[test]
    fn highlighted_larger_and_better_joined_nodes_come_first() {
        let r = [5.0, 5.0, 9.0, 5.0];
        let degree = [1, 3, 0, 3];
        assert_eq!(priority(None, &r, &degree), [2, 1, 3, 0]);
        assert_eq!(priority(Some(&[0, 0, 0, 2]), &r, &degree), [3, 2, 1, 0]);
    }

    #[test]
    fn labels_that_would_overlap_are_left_out() {
        // Nodes 0 and 1 are too close for both labels, and 2 is far from both
        let t = texts(&["alpha", "beta", "gamma", ""]);
        let (x, y, r) = ([0.0, 10.0, 200.0, 400.0], [0.0, 0.0, 0.0, 0.0], [5.0; 4]);
        let order = [3, 1, 0, 2];
        assert_eq!(
            choose(&t, &x, &y, &r, 10.0, &order, Some(10)),
            [false, true, true, false]
        );
        // Up to the limit
        assert_eq!(
            choose(&t, &x, &y, &r, 10.0, &order, Some(1)),
            [false, true, false, false]
        );
        assert_eq!(choose(&t, &x, &y, &r, 10.0, &order, Some(0)), [false; 4]);
        // Without a limit, all of them, however close
        assert_eq!(choose(&t, &x, &y, &r, 10.0, &order, None), [true; 4]);
        // One under the other overlaps if they're close enough
        let (x, y) = ([0.0, 0.0, 0.0, 0.0], [0.0, 12.0, 40.0, 80.0]);
        assert_eq!(
            choose(&t, &x, &y, &r, 10.0, &[0, 1, 2], Some(10)),
            [true, false, true, false]
        );
    }
}
