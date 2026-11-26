// The scene drawn as pixels, for graphs too large to draw as SVG. The edges are drawn as density:
// each pixel sums how much of what is behind it every edge over it would hide, so that where
// thousands of edges cross reads darker than where ten do, rather than as one solid block. The
// nodes go over them as the SVG draws them. The shapes are those of the SVG, from the same
// geometry
use super::geometry::{self, Head};
use super::Scene;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// How the density of the edges shows as their opacity
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tone {
    /// As the SVG draws them: each edge over what is behind it, which turns solid where more than
    /// a few cross
    Over,
    /// As the SVG draws them up to the density of a lone edge, and from there the density's
    /// logarithm, up to opaque at the densest pixel
    Log,
    /// As `Log`, but by the density's rank among the pixels' (histogram equalization), which
    /// spreads the opacities evenly whatever the range of densities
    EqHist,
}

impl Tone {
    pub fn parse(name: &str) -> PyResult<Tone> {
        match name {
            "over" => Ok(Tone::Over),
            "log" => Ok(Tone::Log),
            "eq_hist" => Ok(Tone::EqHist),
            _ => Err(PyValueError::new_err(format!(
                "tone must be 'over', 'log' or 'eq_hist', not '{name}'"
            ))),
        }
    }
}

/// A color as red, green and blue from 0 to 1
pub type Rgb = [f32; 3];

/// The pixels a raster is written as
#[derive(Clone, Copy)]
pub enum Output {
    /// RGB, over a background
    Over(Rgb),
    /// RGBA, over nothing
    Rgba,
    /// Gray and alpha, white everywhere, its alpha that of the RGBA: a mask that shows the
    /// drawing in any one color, read by its alpha or by its luminance
    Mask,
}

/// What the raster is drawn in: the color table's colors in the theme, and what of the scene
pub struct Style {
    pub colors: Vec<Rgb>,
    /// The colors' alphas, which multiply the opacities of what they color
    pub alphas: Vec<f32>,
    pub output: Output,
    /// Pixels to a pixel of the frame
    pub scale: f64,
    pub tone: Tone,
    /// Whether the edges are drawn, and the nodes
    pub edges: bool,
    pub nodes: bool,
}

/// A drawn raster: its size, and its pixels, row by row
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub color: png::ColorType,
}

impl Image {
    /// The image as PNG
    pub fn png(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut encoder = png::Encoder::new(&mut out, self.width, self.height);
        encoder.set_color(self.color);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Balanced);
        let mut writer = encoder
            .write_header()
            .expect("a PNG header writes to memory");
        writer
            .write_image_data(&self.pixels)
            .expect("PNG data writes to memory");
        writer.finish().expect("a PNG finishes in memory");
        out
    }
}

/// How far a curve flattened into straight pieces may stray from it, in pixels
const FLATNESS: f32 = 0.2;

/// At most this many straight pieces to a curve
const PIECES_MAX: f32 = 256.0;

/// Rows of pixels drawn together, in parallel with the other bands
const BAND: usize = 16;

/// A straight piece of an edge's stroke, in pixels
struct Seg {
    a: [f32; 2],
    b: [f32; 2],
    /// Half the stroke's width, but at least half a pixel
    hw: f32,
    /// How much of a pixel that the stroke covers it hides: the edge's opacity, times its width
    /// for one thinner than a pixel
    gain: f32,
    color: u32,
    /// Whether the start and the end are round, as the stroke's ends are, rather than cut square
    /// where pieces meet
    round_a: bool,
    round_b: bool,
}

/// An arrowhead, in pixels
struct Tri {
    p: [[f32; 2]; 3],
    gain: f32,
    color: u32,
}

/// A node, in pixels
struct Disc {
    x: f32,
    y: f32,
    r: f32,
    fill: u32,
    fill_opacity: f32,
    /// Half the border's width, centered on the node's edge as the SVG's stroke is
    bw: f32,
    border: u32,
    border_opacity: f32,
}

/// From the scene's frame to pixels: the viewBox fitted into the image and centered, as the
/// SVG's default `preserveAspectRatio` places it
pub struct View {
    s: f64,
    ox: f64,
    oy: f64,
}

impl View {
    pub fn new(viewbox: (f64, f64, f64, f64), width: u32, height: u32) -> View {
        let (vx, vy, vw, vh) = viewbox;
        let (w, h) = (width as f64, height as f64);
        let s = (w / vw).min(h / vh);
        View {
            s,
            ox: (w - vw * s) / 2.0 - vx * s,
            oy: (h - vh * s) / 2.0 - vy * s,
        }
    }

    fn pt(&self, (x, y): (f64, f64)) -> [f32; 2] {
        [(x * self.s + self.ox) as f32, (y * self.s + self.oy) as f32]
    }

    /// The box `(x, y, width, height)` that an image of `width` × `height` pixels covers, in the
    /// viewBox's units (with 0 rather than -0 at an origin, for the markup)
    pub fn frame(&self, width: u32, height: u32) -> (f64, f64, f64, f64) {
        let s = self.s;
        (
            -self.ox / s + 0.0,
            -self.oy / s + 0.0,
            width as f64 / s,
            height as f64 / s,
        )
    }
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Points along a quadratic Bezier from `p0` (not included) to `p2`
fn flatten_quad(out: &mut Vec<[f32; 2]>, p0: [f32; 2], c: [f32; 2], p2: [f32; 2]) {
    // A quadratic strays from its chords by at most |p0 - 2c + p2| / (4 n²)
    let dd = norm([p0[0] - 2.0 * c[0] + p2[0], p0[1] - 2.0 * c[1] + p2[1]]);
    let n = (dd / (4.0 * FLATNESS)).sqrt().ceil().clamp(1.0, PIECES_MAX) as usize;
    for k in 1..=n {
        let t = k as f32 / n as f32;
        let u = 1.0 - t;
        out.push([0, 1].map(|d| u * u * p0[d] + 2.0 * u * t * c[d] + t * t * p2[d]));
    }
}

/// Points along a cubic Bezier from `p0` (not included) to `p3`
fn flatten_cubic(out: &mut Vec<[f32; 2]>, p0: [f32; 2], c1: [f32; 2], c2: [f32; 2], p3: [f32; 2]) {
    // A cubic strays from its chords by at most 3/4 of its control polygon's greatest second
    // difference over n²
    let d1 = norm([p0[0] - 2.0 * c1[0] + c2[0], p0[1] - 2.0 * c1[1] + c2[1]]);
    let d2 = norm([c1[0] - 2.0 * c2[0] + p3[0], c1[1] - 2.0 * c2[1] + p3[1]]);
    let n = (0.75 * d1.max(d2) / FLATNESS)
        .sqrt()
        .ceil()
        .clamp(1.0, PIECES_MAX) as usize;
    for k in 1..=n {
        let t = k as f32 / n as f32;
        let u = 1.0 - t;
        out.push([0, 1].map(|d| {
            u * u * u * p0[d]
                + 3.0 * u * u * t * c1[d]
                + 3.0 * u * t * t * c2[d]
                + t * t * t * p3[d]
        }));
    }
}

/// What the raster draws of the scene, in pixels: the edges' pieces and arrowheads, and the
/// nodes, each in the order the SVG draws them
struct Shapes {
    segs: Vec<Seg>,
    tris: Vec<Tri>,
    discs: Vec<Disc>,
    /// How much of a pixel that it covers a typical edge hides, as a density
    lone: f32,
}

/// An edge's path through `pts`, with its arrowhead
#[allow(clippy::too_many_arguments)]
fn push_edge(
    shapes: &mut Shapes,
    pts: &[[f32; 2]],
    head: Option<Head>,
    view: &View,
    hw: f32,
    gain: f32,
    head_gain: f32,
    color: u32,
) {
    let n = pts.len();
    for k in 1..n {
        shapes.segs.push(Seg {
            a: pts[k - 1],
            b: pts[k],
            hw,
            gain,
            color,
            round_a: k == 1,
            round_b: k == n - 1,
        });
    }
    if let Some(h) = head {
        shapes.tris.push(Tri {
            p: [(h.ex, h.ey), (h.p1x, h.p1y), (h.p2x, h.p2y)].map(|p| view.pt(p)),
            gain: head_gain,
            color,
        });
    }
}

/// The shapes of the edges and of the nodes that `style` draws
fn shapes(scene: &Scene, view: &View, style: &Style) -> Shapes {
    let (nodes, edges) = (&scene.nodes, &scene.edges);
    let edge_order: &[usize] = if style.edges { &edges.order } else { &[] };
    let node_order: &[usize] = if style.nodes { &nodes.order } else { &[] };
    let directed = scene.directed;
    let s = view.s as f32;
    let mut shapes = Shapes {
        segs: Vec::with_capacity(edge_order.len()),
        tris: Vec::with_capacity(if directed { edge_order.len() } else { 0 }),
        discs: Vec::with_capacity(node_order.len()),
        lone: 0.0,
    };
    let mut lone = Vec::with_capacity(edge_order.len());
    let mut pts: Vec<[f32; 2]> = Vec::new();
    for &i in edge_order {
        let (a, b) = (edges.src[i], edges.tgt[i]);
        let (x1, y1, x2, y2) = (nodes.x[a], nodes.y[a], nodes.x[b], nodes.y[b]);
        let (r_src, r_dst) = (nodes.r[a], nodes.r[b]);
        let color = edges.color[i];
        let (width, opacity) = (
            edges.width[i],
            edges.opacity[i] as f32 * style.alphas[color],
        );
        let bow = edges.bow[i];
        pts.clear();
        // As `svg::draw_edges()` picks the shape
        let head = if a == b {
            let (p, head) =
                geometry::self_loop_points(x1, y1, r_src, width, directed, edges.loop_size[i]);
            let p = p.map(|p| view.pt(p));
            pts.push(p[0]);
            flatten_cubic(&mut pts, p[0], p[1], p[2], p[3]);
            flatten_cubic(&mut pts, p[3], p[4], p[5], p[6]);
            head
        } else if scene.curved || bow.abs() > 1e-6 {
            let bow = if scene.curved { 0.2 + bow } else { bow };
            let Some((p, head)) =
                geometry::curved_points(x1, y1, x2, y2, r_src, r_dst, width, directed, bow)
            else {
                continue;
            };
            let p = p.map(|p| view.pt(p));
            pts.push(p[0]);
            flatten_quad(&mut pts, p[0], p[1], p[2]);
            head
        } else if let Some(route) = edges.route[i].as_deref() {
            let (route, head) = geometry::routed_points(route, r_src, r_dst, width, directed);
            let Some(route) = route else { continue };
            pts.push(view.pt(route[0]));
            if route.len() == 2 {
                pts.push(view.pt(route[1]));
            } else {
                for k in 0..route.len() - 1 {
                    let [c1, c2, p2] = geometry::catmull_rom(&route, k);
                    let p0 = view.pt(route[k]);
                    flatten_cubic(&mut pts, p0, view.pt(c1), view.pt(c2), view.pt(p2));
                }
            }
            head
        } else {
            let Some(line) = geometry::straight(x1, y1, x2, y2, r_src, r_dst, width, directed)
            else {
                continue;
            };
            pts.push(view.pt((line.sx, line.sy)));
            pts.push(view.pt((line.lx, line.ly)));
            line.head
        };
        if !pts.iter().flatten().all(|v| v.is_finite()) {
            continue;
        }
        // A stroke thinner than a pixel is drawn a pixel wide, as faint as it is thin
        let w = width as f32 * s;
        let (hw, gain) = if w < 1.0 {
            (0.5, opacity * w.max(0.0))
        } else {
            (w / 2.0, opacity)
        };
        lone.push(-(-gain.min(0.999)).ln_1p());
        push_edge(
            &mut shapes,
            &pts,
            head,
            view,
            hw,
            gain,
            opacity,
            color as u32,
        );
    }
    // The median edge's density
    if !lone.is_empty() {
        let mid = lone.len() / 2;
        let (_, m, _) = lone.select_nth_unstable_by(mid, f32::total_cmp);
        shapes.lone = *m;
    }
    for &k in node_order {
        let [x, y] = view.pt((nodes.x[k], nodes.y[k]));
        let r = nodes.r[k] as f32 * s;
        if !(x.is_finite() && y.is_finite() && r.is_finite()) {
            continue;
        }
        shapes.discs.push(Disc {
            x,
            y,
            r,
            fill: nodes.fill[k] as u32,
            fill_opacity: style.alphas[nodes.fill[k]],
            bw: (nodes.border_width[k] as f32 * s / 2.0).max(0.0),
            border: nodes.border[k] as u32,
            border_opacity: nodes.border_opacity[k] as f32 * style.alphas[nodes.border[k]],
        });
    }
    shapes
}

/// A pixel's density, and its edges' colors weighted by their share of it
type Ink = [f32; 4];

/// Add `a`, the share of a pixel an edge of `color` hides, to the pixel's ink
fn add_ink(ink: &mut Ink, a: f32, color: &Rgb) {
    // Densities add where the shares they hide multiply, as layers of SVG strokes do
    let d = -(-a.min(0.999)).ln_1p();
    ink[0] += d;
    ink[1] += d * color[0];
    ink[2] += d * color[1];
    ink[3] += d * color[2];
}

/// The edge pieces' ink in a band of rows, from row `y0`
fn draw_segs(band: &mut [Ink], y0: usize, width: usize, segs: &[Seg], colors: &[Rgb]) {
    let rows = band.len() / width;
    let (band_top, band_bottom) = (y0 as f32, (y0 + rows) as f32);
    for seg in segs {
        let reach = seg.hw + 0.5;
        let [ax, ay] = seg.a;
        let [bx, by] = seg.b;
        let (top, bottom) = (ay.min(by) - reach, ay.max(by) + reach);
        if bottom < band_top || top > band_bottom {
            continue;
        }
        let (dx, dy) = (bx - ax, by - ay);
        let len2 = dx * dx + dy * dy;
        let inv = if len2 > 1e-12 { 1.0 / len2 } else { 0.0 };
        let color = &colors[seg.color as usize];
        // The rows whose pixel centers are within reach
        let j0 = ((top - 0.5).ceil() as isize).max(y0 as isize) as usize;
        let j1 = ((bottom - 0.5).floor() as isize).min((y0 + rows) as isize - 1);
        if j1 < j0 as isize {
            continue;
        }
        for j in j0..=j1 as usize {
            let py = j as f32 + 0.5;
            // The span of the piece within reach of this row, and so the pixels within reach
            let (mut t0, mut t1) = if dy.abs() > 1e-6 {
                let (u, v) = ((py - reach - ay) / dy, (py + reach - ay) / dy);
                (u.min(v), u.max(v))
            } else {
                (0.0, 1.0)
            };
            t0 = t0.max(0.0);
            t1 = t1.min(1.0);
            if t1 < t0 {
                continue;
            }
            let (xa, xb) = (ax + t0 * dx, ax + t1 * dx);
            let i0 = ((xa.min(xb) - reach - 0.5).ceil() as isize).max(0) as usize;
            let i1 = ((xa.max(xb) + reach - 0.5).floor() as isize).min(width as isize - 1);
            if i1 < i0 as isize {
                continue;
            }
            let row = &mut band[(j - y0) * width..(j - y0 + 1) * width];
            for (i, ink) in row.iter_mut().enumerate().take(i1 as usize + 1).skip(i0) {
                let px = i as f32 + 0.5;
                let t = ((px - ax) * dx + (py - ay) * dy) * inv;
                if (t < 0.0 && !seg.round_a) || (t >= 1.0 && !seg.round_b) {
                    continue;
                }
                let t = t.clamp(0.0, 1.0);
                let (ex, ey) = (px - ax - t * dx, py - ay - t * dy);
                let c = (reach - (ex * ex + ey * ey).sqrt()).min(1.0);
                if c > 0.0 {
                    add_ink(ink, c * seg.gain, color);
                }
            }
        }
    }
}

/// The arrowheads' ink in a band of rows, from row `y0`
fn draw_tris(band: &mut [Ink], y0: usize, width: usize, tris: &[Tri], colors: &[Rgb]) {
    let rows = band.len() / width;
    for tri in tris {
        let [p0, p1, p2] = tri.p;
        let top = p0[1].min(p1[1]).min(p2[1]) - 0.5;
        let bottom = p0[1].max(p1[1]).max(p2[1]) + 0.5;
        if bottom < y0 as f32 || top > (y0 + rows) as f32 {
            continue;
        }
        // Each side as a line, its normal pointing out: a pixel's distance outside the
        // triangle is the greatest of its distances past the sides
        let area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        if area.abs() < 1e-9 {
            continue;
        }
        let sign = area.signum();
        let sides = [(p0, p1), (p1, p2), (p2, p0)].map(|(a, b)| {
            let e = sub(b, a);
            let len = norm(e).max(1e-9);
            let (nx, ny) = (sign * e[1] / len, -sign * e[0] / len);
            (nx, ny, nx * a[0] + ny * a[1])
        });
        let left = p0[0].min(p1[0]).min(p2[0]) - 0.5;
        let right = p0[0].max(p1[0]).max(p2[0]) + 0.5;
        let j0 = ((top - 0.5).ceil() as isize).max(y0 as isize) as usize;
        let j1 = ((bottom - 0.5).floor() as isize).min((y0 + rows) as isize - 1);
        let i0 = ((left - 0.5).ceil() as isize).max(0) as usize;
        let i1 = ((right - 0.5).floor() as isize).min(width as isize - 1);
        if j1 < j0 as isize || i1 < i0 as isize {
            continue;
        }
        let color = &colors[tri.color as usize];
        for j in j0..=j1 as usize {
            let py = j as f32 + 0.5;
            let row = &mut band[(j - y0) * width..(j - y0 + 1) * width];
            for (i, ink) in row.iter_mut().enumerate().take(i1 as usize + 1).skip(i0) {
                let px = i as f32 + 0.5;
                let out = sides
                    .iter()
                    .map(|&(nx, ny, c)| nx * px + ny * py - c)
                    .fold(f32::NEG_INFINITY, f32::max);
                let c = (0.5 - out).min(1.0);
                if c > 0.0 {
                    add_ink(ink, c * tri.gain, color);
                }
            }
        }
    }
}

/// Paint `color` over a premultiplied pixel, covering `a` of it
fn over(px: &mut [f32; 4], color: &Rgb, a: f32) {
    let keep = 1.0 - a;
    px[0] = color[0] * a + px[0] * keep;
    px[1] = color[1] * a + px[1] * keep;
    px[2] = color[2] * a + px[2] * keep;
    px[3] = a + px[3] * keep;
}

/// The nodes over a band of premultiplied pixels, from row `y0`
fn draw_discs(band: &mut [[f32; 4]], y0: usize, width: usize, discs: &[Disc], colors: &[Rgb]) {
    let rows = band.len() / width;
    for disc in discs {
        // A node or border thinner than a pixel is drawn a pixel across, as faint as it is small
        let (r, fill_gain) = if disc.r < 0.5 {
            (0.5, (disc.r / 0.5).powi(2) * disc.fill_opacity)
        } else {
            (disc.r, disc.fill_opacity)
        };
        let (bw, border_gain) = if disc.bw <= 0.0 || disc.border_opacity <= 0.0 {
            (0.0, 0.0)
        } else if disc.bw < 0.5 {
            (0.5, disc.bw / 0.5 * disc.border_opacity)
        } else {
            (disc.bw, disc.border_opacity)
        };
        let reach = r.max(disc.r + bw) + 0.5;
        let (top, bottom) = (disc.y - reach, disc.y + reach);
        if bottom < y0 as f32 || top > (y0 + rows) as f32 {
            continue;
        }
        let j0 = ((top - 0.5).ceil() as isize).max(y0 as isize) as usize;
        let j1 = ((bottom - 0.5).floor() as isize).min((y0 + rows) as isize - 1);
        let i0 = ((disc.x - reach - 0.5).ceil() as isize).max(0) as usize;
        let i1 = ((disc.x + reach - 0.5).floor() as isize).min(width as isize - 1);
        if j1 < j0 as isize || i1 < i0 as isize {
            continue;
        }
        let (fill, border) = (&colors[disc.fill as usize], &colors[disc.border as usize]);
        for j in j0..=j1 as usize {
            let dy = j as f32 + 0.5 - disc.y;
            let row = &mut band[(j - y0) * width..(j - y0 + 1) * width];
            for (i, px) in row.iter_mut().enumerate().take(i1 as usize + 1).skip(i0) {
                let dx = i as f32 + 0.5 - disc.x;
                let d = (dx * dx + dy * dy).sqrt();
                let a = (r + 0.5 - d).clamp(0.0, 1.0) * fill_gain;
                if a > 0.0 {
                    over(px, fill, a);
                }
                if border_gain > 0.0 {
                    let a = (bw + 0.5 - (d - disc.r).abs()).clamp(0.0, 1.0) * border_gain;
                    if a > 0.0 {
                        over(px, border, a);
                    }
                }
            }
        }
    }
}

/// Bins of the logarithm of the density, for `Tone::EqHist`
const BINS: usize = 4096;

/// From a pixel's density to its opacity
struct ToneMap {
    tone: Tone,
    /// A lone edge's density, and the opacity it has drawn over what's behind it
    lone: f32,
    lone_a: f32,
    /// The density at which the logarithm reaches opaque
    top: f32,
    /// For `Tone::EqHist`, the share of the pixels denser than a lone edge at or below each bin
    cdf: Vec<f32>,
}

/// The density at which an edge drawn over what's behind it is 98% opaque
const SOLID: f32 = 3.912;

impl ToneMap {
    fn new(tone: Tone, lone: f32, ink: &[Ink]) -> ToneMap {
        let lone = lone.max(1e-3);
        let densest = ink.par_iter().map(|p| p[0]).reduce(|| 0.0, f32::max);
        let top = densest.max(SOLID).max(lone * 1.01);
        let mut map = ToneMap {
            tone,
            lone,
            lone_a: -(-lone).exp_m1(),
            top,
            cdf: Vec::new(),
        };
        if tone == Tone::EqHist {
            let counts = ink
                .par_chunks(1 << 16)
                .map(|chunk| {
                    let mut counts = vec![0u64; BINS];
                    for p in chunk {
                        if p[0] > lone {
                            counts[map.bin(p[0])] += 1;
                        }
                    }
                    counts
                })
                .reduce(
                    || vec![0u64; BINS],
                    |mut a, b| {
                        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                        a
                    },
                );
            let total = counts.iter().sum::<u64>().max(1) as f64;
            let mut seen = 0u64;
            map.cdf = counts
                .iter()
                .map(|&c| {
                    seen += c;
                    (seen as f64 / total) as f32
                })
                .collect();
        }
        map
    }

    fn bin(&self, d: f32) -> usize {
        let t = (d / self.lone).ln() / (self.top / self.lone).ln();
        ((t * BINS as f32) as usize).min(BINS - 1)
    }

    fn alpha(&self, d: f32) -> f32 {
        if self.tone == Tone::Over || d <= self.lone {
            return -(-d).exp_m1();
        }
        let t = match self.tone {
            Tone::Log => (d / self.lone).ln() / (self.top / self.lone).ln(),
            _ => self.cdf[self.bin(d)],
        };
        self.lone_a + (1.0 - self.lone_a) * t.min(1.0)
    }
}

/// The scene drawn in `style`
pub fn draw(scene: &Scene, style: &Style) -> Image {
    let width = (scene.width * style.scale).round().max(1.0) as u32;
    let height = (scene.height * style.scale).round().max(1.0) as u32;
    let (w, h) = (width as usize, height as usize);
    let view = View::new(scene.viewbox, width, height);
    let shapes = shapes(scene, &view, style);
    let colors = &style.colors;

    let mut ink: Vec<Ink> = vec![[0.0; 4]; w * h];
    ink.par_chunks_mut(w * BAND)
        .enumerate()
        .for_each(|(k, band)| {
            draw_segs(band, k * BAND, w, &shapes.segs, colors);
            draw_tris(band, k * BAND, w, &shapes.tris, colors);
        });
    let tone = ToneMap::new(style.tone, shapes.lone, &ink);

    // The ink becomes premultiplied color, then the nodes go over it
    let mut pixels = ink;
    pixels
        .par_chunks_mut(w * BAND)
        .enumerate()
        .for_each(|(k, band)| {
            for px in band.iter_mut() {
                let d = px[0];
                if d > 0.0 {
                    let a = tone.alpha(d);
                    *px = [px[1] / d * a, px[2] / d * a, px[3] / d * a, a];
                }
            }
            draw_discs(band, k * BAND, w, &shapes.discs, colors);
        });

    let to_byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let (pixels, color): (Vec<u8>, _) = match style.output {
        Output::Over(bg) => (
            pixels
                .par_iter()
                .flat_map_iter(|px| {
                    let keep = 1.0 - px[3];
                    [0, 1, 2].map(|c| to_byte(px[c] + bg[c] * keep))
                })
                .collect(),
            png::ColorType::Rgb,
        ),
        Output::Rgba => (
            pixels
                .par_iter()
                .flat_map_iter(|px| {
                    let a = px[3];
                    if a <= 0.0 {
                        [0; 4]
                    } else {
                        let [r, g, b] = [0, 1, 2].map(|c| to_byte(px[c] / a));
                        [r, g, b, to_byte(a)]
                    }
                })
                .collect(),
            png::ColorType::Rgba,
        ),
        Output::Mask => (
            pixels
                .par_iter()
                .flat_map_iter(|px| [255, to_byte(px[3])])
                .collect(),
            png::ColorType::GrayscaleAlpha,
        ),
    };
    Image {
        width,
        height,
        pixels,
        color,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band(w: usize, h: usize) -> Vec<Ink> {
        vec![[0.0; 4]; w * h]
    }

    fn seg(a: [f32; 2], b: [f32; 2], hw: f32, gain: f32) -> Seg {
        Seg {
            a,
            b,
            hw,
            gain,
            color: 0,
            round_a: true,
            round_b: true,
        }
    }

    #[test]
    fn a_stroke_covers_its_area() {
        // A faint stroke, so that the density is close to its coverage: 40 pixels long and 3
        // wide, with round ends, covers 40 × 3 + π × 1.5²
        let colors = [[1.0, 0.0, 0.0]];
        let mut ink = band(64, 16);
        let s = seg([10.0, 8.0], [50.0, 8.0], 1.5, 0.01);
        draw_segs(&mut ink, 0, 64, &[s], &colors);
        let area: f32 = ink.iter().map(|p| p[0]).sum::<f32>() / 0.01;
        let want = 40.0 * 3.0 + std::f32::consts::PI * 1.5 * 1.5;
        assert!((area - want).abs() / want < 0.01, "{area} vs {want}");
        // In its color only
        assert!(ink
            .iter()
            .all(|p| p[2] == 0.0 && p[3] == 0.0 && (p[1] - p[0]).abs() < 1e-6));
        // A slanted one, the same
        let mut ink = band(64, 64);
        let s = seg([10.0, 10.0], [38.284, 38.284], 1.5, 0.01);
        draw_segs(&mut ink, 0, 64, &[s], &colors);
        let area: f32 = ink.iter().map(|p| p[0]).sum::<f32>() / 0.01;
        assert!((area - want).abs() / want < 0.01, "{area} vs {want}");
    }

    #[test]
    fn bands_draw_what_the_whole_draws() {
        let colors = [[0.0, 0.0, 1.0]];
        let segs = [
            seg([3.0, 1.0], [60.0, 47.0], 0.5, 0.3),
            seg([60.0, 2.0], [5.0, 40.0], 2.0, 0.5),
        ];
        let mut whole = band(64, 48);
        draw_segs(&mut whole, 0, 64, &segs, &colors);
        let mut banded = band(64, 48);
        for (k, b) in banded.chunks_mut(64 * 5).enumerate() {
            draw_segs(b, k * 5, 64, &segs, &colors);
        }
        assert!(whole == banded);
    }

    #[test]
    fn pieces_join_without_doubling() {
        // Two pieces of one straight stroke draw as the stroke in one piece
        let colors = [[1.0, 1.0, 1.0]];
        let mut one = band(64, 16);
        draw_segs(
            &mut one,
            0,
            64,
            &[seg([4.0, 8.2], [60.0, 8.2], 1.0, 0.4)],
            &colors,
        );
        let mut two = band(64, 16);
        let mut first = seg([4.0, 8.2], [30.5, 8.2], 1.0, 0.4);
        let mut second = seg([30.5, 8.2], [60.0, 8.2], 1.0, 0.4);
        first.round_b = false;
        second.round_a = false;
        draw_segs(&mut two, 0, 64, &[first, second], &colors);
        for (p, q) in one.iter().zip(&two) {
            assert!((p[0] - q[0]).abs() < 1e-5);
        }
    }

    #[test]
    fn densities_add_as_layers_hide() {
        // Two edges at 0.4 over one pixel hide 1 - 0.6² of it
        let mut ink = [0.0; 4];
        add_ink(&mut ink, 0.4, &[1.0, 0.0, 0.0]);
        add_ink(&mut ink, 0.4, &[0.0, 1.0, 0.0]);
        assert!((-(-ink[0]).exp_m1() - 0.64).abs() < 1e-6);
        assert!((ink[1] - ink[2]).abs() < 1e-6);
    }

    #[test]
    fn tones_rise_with_density() {
        let lone = -(-0.4f32).ln_1p();
        let ink: Vec<Ink> = (0..1000)
            .map(|k| [k as f32 * 0.05, 0.0, 0.0, 0.0])
            .collect();
        for tone in [Tone::Over, Tone::Log, Tone::EqHist] {
            let map = ToneMap::new(tone, lone, &ink);
            // A lone edge as the SVG draws it
            assert!((map.alpha(lone) - 0.4).abs() < 1e-5, "{tone:?}");
            let mut last = 0.0;
            for p in &ink {
                let a = map.alpha(p[0]);
                assert!(a >= last && a <= 1.0, "{tone:?}");
                last = a;
            }
            if tone != Tone::Over {
                assert!((map.alpha(49.95) - 1.0).abs() < 1e-3, "{tone:?}");
                assert!(map.alpha(4.0) < 0.9, "{tone:?}");
            }
        }
    }
}
