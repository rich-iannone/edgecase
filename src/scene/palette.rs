// A raster reduced to at most 256 colors, so that it can be written as an indexed PNG: a byte a
// pixel rather than four. An image of no more colors keeps them all. Otherwise its colors,
// weighted by how many pixels have them, are split into the boxes of least variance (as median
// cut does, but at the best point rather than the median), and the boxes' means are then refined
// as k-means. All in premultiplied color, where the error is the error that shows over any
// background. There's no dithering, which would cost compression and add noise.
use rayon::prelude::*;

/// The most colors in a palette
pub const COLORS_MAX: usize = 256;

/// Rounds of k-means after the boxes
const ROUNDS: usize = 8;

/// An image as indexes into a palette of straight (not premultiplied) RGBA
pub struct Indexed {
    pub palette: Vec<[u8; 4]>,
    pub indexes: Vec<u8>,
}

/// A distinct color of the image: premultiplied, from 0 to 255, with how many pixels have it and
/// its place among the distinct colors
#[derive(Clone, Copy)]
struct Point {
    c: [f32; 4],
    w: f64,
    id: usize,
}

/// A color as a key that sorts, every fully transparent color the same
fn key(px: &[u8]) -> u32 {
    if px[3] == 0 {
        0
    } else {
        u32::from_be_bytes([px[0], px[1], px[2], px[3]])
    }
}

fn premultiplied(key: u32) -> [f32; 4] {
    let [r, g, b, a] = key.to_be_bytes().map(f32::from);
    let s = a / 255.0;
    [r * s, g * s, b * s, a]
}

/// A premultiplied color as straight bytes
fn straight(c: [f32; 4]) -> [u8; 4] {
    let a = c[3].round().clamp(0.0, 255.0);
    if a == 0.0 {
        return [0; 4];
    }
    let byte = |v: f32| (v * 255.0 / a).round().clamp(0.0, 255.0) as u8;
    [byte(c[0]), byte(c[1]), byte(c[2]), a as u8]
}

/// Sums of weights, of weighted colors and of weighted squares, from which the mean and the error
/// (the weighted sum of squared distances from the mean) of a set of colors follow
#[derive(Clone, Copy, Default)]
struct Sums {
    w: f64,
    c: [f64; 4],
    sq: f64,
}

impl Sums {
    fn of(points: &[Point]) -> Sums {
        let mut s = Sums::default();
        points.iter().for_each(|p| s.add(p));
        s
    }

    fn add(&mut self, p: &Point) {
        self.w += p.w;
        for i in 0..4 {
            self.c[i] += p.w * p.c[i] as f64;
            self.sq += p.w * (p.c[i] as f64).powi(2);
        }
    }

    fn minus(&self, o: &Sums) -> Sums {
        Sums {
            w: self.w - o.w,
            c: [0, 1, 2, 3].map(|i| self.c[i] - o.c[i]),
            sq: self.sq - o.sq,
        }
    }

    fn error(&self) -> f64 {
        if self.w <= 0.0 {
            return 0.0;
        }
        (self.sq - self.c.iter().map(|v| v * v).sum::<f64>() / self.w).max(0.0)
    }

    fn mean(&self) -> [f32; 4] {
        self.c.map(|v| (v / self.w) as f32)
    }
}

/// A run of the points, and its error
struct Cell {
    start: usize,
    end: usize,
    error: f64,
}

/// The points split into at most `k` runs: each time the run of the most error is sorted along its
/// axis of most variance and cut where the two halves have the least error
fn split(points: &mut [Point], k: usize) -> Vec<Cell> {
    let mut cells = vec![Cell {
        start: 0,
        end: points.len(),
        error: Sums::of(points).error(),
    }];
    while cells.len() < k {
        let Some(at) = (0..cells.len())
            .filter(|&i| cells[i].error > 0.0)
            .max_by(|&a, &b| cells[a].error.total_cmp(&cells[b].error))
        else {
            break;
        };
        let (start, end) = (cells[at].start, cells[at].end);
        let run = &mut points[start..end];
        let all = Sums::of(run);
        let mean = all.mean();
        let mut spread = [0.0f64; 4];
        for p in run.iter() {
            for (i, s) in spread.iter_mut().enumerate() {
                *s += p.w * ((p.c[i] - mean[i]) as f64).powi(2);
            }
        }
        let axis = (0..4)
            .max_by(|&a, &b| spread[a].total_cmp(&spread[b]))
            .unwrap_or(0);
        // Ties by the place, so that the order is the same every time
        run.sort_unstable_by(|a, b| a.c[axis].total_cmp(&b.c[axis]).then(a.id.cmp(&b.id)));
        let mut left = Sums::default();
        let mut best = (f64::INFINITY, 0, 0.0, 0.0);
        for i in 0..run.len() - 1 {
            left.add(&run[i]);
            // Only between different values on the axis, so that equal ones stay together
            if run[i].c[axis] == run[i + 1].c[axis] {
                continue;
            }
            let (le, re) = (left.error(), all.minus(&left).error());
            if le + re < best.0 {
                best = (le + re, i + 1, le, re);
            }
        }
        let (_, cut, le, re) = best;
        if cut == 0 {
            // The run's colors are the same on its axis of most variance, so on every axis, but
            // for rounding
            cells[at].error = 0.0;
            continue;
        }
        cells[at] = Cell {
            start,
            end: start + cut,
            error: le,
        };
        cells.push(Cell {
            start: start + cut,
            end,
            error: re,
        });
    }
    cells
}

/// The nearest of `centers` to each point
fn nearest(points: &[Point], centers: &[[f32; 4]]) -> Vec<u16> {
    points
        .par_iter()
        .map(|p| {
            let mut best = (f32::INFINITY, 0);
            for (k, m) in centers.iter().enumerate() {
                let d: f32 = (0..4).map(|i| (p.c[i] - m[i]).powi(2)).sum();
                if d < best.0 {
                    best = (d, k as u16);
                }
            }
            best.1
        })
        .collect()
}

/// A palette for more distinct colors than it can hold, and each color's entry in it
fn reduce(distinct: &[u32], counts: &[u32]) -> (Vec<[u8; 4]>, Vec<u16>) {
    // Transparency has an entry of its own, so that nothing comes out faintly over it
    let clear = distinct[0] == 0;
    let mut points: Vec<Point> = (clear as usize..distinct.len())
        .map(|id| Point {
            c: premultiplied(distinct[id]),
            w: counts[id] as f64,
            id,
        })
        .collect();
    let cells = split(&mut points, COLORS_MAX - clear as usize);
    let mut centers: Vec<[f32; 4]> = cells
        .iter()
        .map(|c| Sums::of(&points[c.start..c.end]).mean())
        .collect();

    let mut near = nearest(&points, &centers);
    for _ in 0..ROUNDS {
        let mut cluster = vec![Sums::default(); centers.len()];
        for (p, &k) in points.iter().zip(&near) {
            cluster[k as usize].add(p);
        }
        for (m, s) in centers.iter_mut().zip(&cluster) {
            if s.w > 0.0 {
                *m = s.mean();
            }
        }
        let next = nearest(&points, &centers);
        let settled = next == near;
        near = next;
        if settled {
            break;
        }
    }

    // A color with most of its cluster's pixels, as a node's fill has, is kept as it is
    let mut heaviest = vec![None::<&Point>; centers.len()];
    let mut total = vec![0.0f64; centers.len()];
    for (p, &k) in points.iter().zip(&near) {
        let k = k as usize;
        total[k] += p.w;
        if heaviest[k].is_none_or(|h| p.w > h.w) {
            heaviest[k] = Some(p);
        }
    }
    for (k, m) in centers.iter_mut().enumerate() {
        if let Some(h) = heaviest[k].filter(|h| h.w * 2.0 >= total[k]) {
            *m = h.c;
        }
    }

    let mut palette: Vec<[u8; 4]> = centers.iter().map(|&m| straight(m)).collect();
    let mut entry = vec![0u16; distinct.len()];
    for (p, &k) in points.iter().zip(&near) {
        entry[p.id] = k;
    }
    if clear {
        entry[0] = palette.len() as u16;
        palette.push([0; 4]);
    }
    (palette, entry)
}

/// The RGBA pixels as at most 256 colors
pub fn quantize(rgba: &[u8]) -> Indexed {
    let keys: Vec<u32> = rgba.par_chunks_exact(4).map(key).collect();
    let mut sorted = keys.clone();
    sorted.par_sort_unstable();
    let (mut distinct, mut counts) = (Vec::new(), Vec::new());
    for &k in &sorted {
        if distinct.last() == Some(&k) {
            *counts.last_mut().unwrap() += 1u32;
        } else {
            distinct.push(k);
            counts.push(1);
        }
    }

    let (palette, entry) = if distinct.len() <= COLORS_MAX {
        let palette = distinct.iter().map(|&k| k.to_be_bytes()).collect();
        (palette, (0..distinct.len() as u16).collect())
    } else {
        reduce(&distinct, &counts)
    };

    // In order of opacity, so that the translucent entries come first and the tRNS chunk can stop
    // at the last of them, then of lightness, so that entries alike sit together
    let luma = |c: &[u8; 4]| 2 * c[0] as u32 + 5 * c[1] as u32 + c[2] as u32;
    let mut order: Vec<usize> = (0..palette.len()).collect();
    order.sort_by_key(|&i| (palette[i][3], luma(&palette[i]), palette[i]));
    let mut rank = vec![0u8; palette.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r as u8;
    }
    let indexes = keys
        .par_iter()
        .map(|k| {
            let at = distinct
                .binary_search(k)
                .expect("every key is a distinct color");
            rank[entry[at] as usize]
        })
        .collect();
    Indexed {
        palette: order.iter().map(|&i| palette[i]).collect(),
        indexes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn colors_of(image: &Indexed) -> Vec<[u8; 4]> {
        image
            .indexes
            .iter()
            .map(|&i| image.palette[i as usize])
            .collect()
    }

    #[test]
    fn few_colors_are_kept() {
        let pixels: Vec<[u8; 4]> = (0..600u32)
            .map(|i| {
                [
                    (i % 7 * 30) as u8,
                    200,
                    (i % 5 * 50) as u8,
                    [0, 80, 255][i as usize % 3],
                ]
            })
            .collect();
        let image = quantize(pixels.concat().as_slice());
        // Every transparent pixel is the one transparent entry
        let want: Vec<[u8; 4]> = pixels
            .iter()
            .map(|&p| if p[3] == 0 { [0; 4] } else { p })
            .collect();
        assert_eq!(colors_of(&image), want);
        assert!(image.palette.windows(2).all(|w| w[0][3] <= w[1][3]));
    }

    #[test]
    fn many_colors_come_near() {
        // A gradient of 4096 colors at 64 opacities, over a large field of one color, and
        // transparency
        let mut pixels: Vec<[u8; 4]> = Vec::new();
        for i in 0..4096u32 {
            pixels.push([
                (i % 64 * 4) as u8,
                (i / 64 * 4) as u8,
                128,
                (i % 61 * 4 + 15) as u8,
            ]);
        }
        pixels.extend(std::iter::repeat_n([40, 90, 200, 255], 20_000));
        pixels.extend(std::iter::repeat_n([7, 7, 7, 0], 1000));
        let image = quantize(pixels.concat().as_slice());
        assert!(image.palette.len() <= COLORS_MAX);
        let got = colors_of(&image);
        // The field and the transparency come out as they are
        assert!(got[4096..24_096].iter().all(|&c| c == [40, 90, 200, 255]));
        assert!(got[24_096..].iter().all(|&c| c[3] == 0));
        // The rest within a few levels, premultiplied
        let mut worst = 0.0f32;
        let mut total = 0.0f32;
        for (p, q) in pixels[..4096].iter().zip(&got) {
            let (p, q) = (
                premultiplied(u32::from_be_bytes(*p)),
                premultiplied(u32::from_be_bytes(*q)),
            );
            let d = (0..4).map(|i| (p[i] - q[i]).abs()).fold(0.0, f32::max);
            worst = worst.max(d);
            total += d;
        }
        assert!(
            total / 4096.0 < 8.0 && worst < 24.0,
            "mean {} worst {worst}",
            total / 4096.0
        );
        // And the same every time
        let again = quantize(pixels.concat().as_slice());
        assert_eq!(
            (again.palette, again.indexes),
            (image.palette, image.indexes)
        );
    }
}
