// A scene drawn as SVG
use super::geometry::{self, py_min, Head};
use super::pyfmt::{Escaped, General, Px, Repr, F1};
use super::{shade, Scene};
use std::collections::HashMap;
use std::fmt::Write;

/// What a scene is drawn in: its color table's colors in the theme, and the theme's own
pub struct Style<'a> {
    pub colors: &'a [String],
    pub bg: &'a str,
    pub label_color: &'a str,
    pub font_family: &'a str,
    /// The legend's markup, drawn over the rest
    pub legend: Option<&'a str>,
    /// The markup of the edges' layer and of the nodes' as images, drawn in place of the edges
    /// and of the nodes
    pub raster: Option<(&'a str, &'a str)>,
    /// To shade the nodes as spheres (see `shade.rs`): for each color of the table, the id of
    /// its gradient and the colors of its highlight and its far side
    pub shade: Option<&'a [(String, String, String)]>,
}

/// The SVG's parts, a line each
struct Parts {
    buf: String,
    started: bool,
}

impl Parts {
    /// The buffer, to write the next part into
    fn part(&mut self) -> &mut String {
        if self.started {
            self.buf.push('\n');
        }
        self.started = true;
        &mut self.buf
    }

    fn push(&mut self, part: &str) {
        self.part().push_str(part);
    }
}

/// Elements of lean SVG, gathered into groups that carry their style. Each run of consecutive
/// elements with the same style is one group, or one group per layer (an edge's line and its
/// arrowhead), and its elements inherit the style from it
#[derive(Default)]
struct Runs {
    /// The layers' opening tags
    groups: Vec<String>,
    /// Each layer's elements so far, a line each
    layers: Vec<Option<String>>,
}

impl Runs {
    /// Add an element to each layer (`None` for none)
    fn add(&mut self, out: &mut Parts, groups: Vec<String>, elements: &[Option<&str>]) {
        if groups != self.groups {
            self.flush(out);
            self.layers = vec![None; groups.len()];
            self.groups = groups;
        }
        for (layer, element) in self.layers.iter_mut().zip(elements) {
            if let Some(element) = element {
                match layer {
                    Some(text) => {
                        text.push('\n');
                        text.push_str(element);
                    }
                    None => *layer = Some(element.to_string()),
                }
            }
        }
    }

    /// Write out the run so far, so that what follows is drawn over it
    fn flush(&mut self, out: &mut Parts) {
        for (group, layer) in self.groups.iter().zip(&self.layers) {
            if let Some(text) = layer {
                out.push(group);
                out.push(text);
                out.push("</g>");
            }
        }
        self.groups.clear();
        self.layers.clear();
    }
}

/// The SVG of `scene` in `style`
pub fn draw(scene: &Scene, style: &Style<'_>) -> String {
    let (width, height) = (scene.width, scene.height);
    if scene.nodes.ids.is_empty() {
        return format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
             viewBox=\"0 0 {w} {h}\"><rect width=\"100%\" height=\"100%\" fill=\"{bg}\"/>\
             <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"{}\" font-family=\"{}\" \
             font-size=\"14\">Empty graph</text></svg>",
            Repr(width / 2.0),
            Repr(height / 2.0),
            style.label_color,
            style.font_family,
            w = Px(width),
            h = Px(height),
            bg = style.bg,
        );
    }
    let mut out = Parts {
        buf: String::with_capacity(
            4096 + 160 * scene.nodes.ids.len() + 120 * scene.edges.src.len(),
        ),
        started: false,
    };
    let (x, y, w, h) = scene.viewbox;
    let _ = write!(
        out.part(),
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"{} {} {} {}\" style=\"font-family: {};\">",
        Px(width),
        Px(height),
        F1(x),
        F1(y),
        F1(w),
        F1(h),
        style.font_family
    );
    let _ = write!(
        out.part(),
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
        F1(x),
        F1(y),
        F1(w),
        F1(h),
        style.bg
    );
    if style.raster.is_none() {
        draw_gradients(scene, style, &mut out);
    }
    // The painter's algorithm: edges first, nodes over them and labels over those
    match style.raster {
        Some((edges, _)) => out.push(edges),
        None => draw_edges(scene, style, &mut out),
    }
    draw_edge_labels(scene, style, &mut out);
    match style.raster {
        Some((_, nodes)) => out.push(nodes),
        None => draw_nodes(scene, style, &mut out),
    }
    draw_labels(scene, style, &mut out);
    if let Some(legend) = style.legend {
        out.push(legend);
    }
    out.push("</svg>");
    out.buf
}

/// An arrowhead's polygon
fn polygon(p: &mut String, head: &Head, color: &str) {
    let _ = write!(
        p,
        "<polygon points=\"{},{} {},{} {},{}\" fill=\"{}\"/>",
        F1(head.ex),
        F1(head.ey),
        F1(head.p1x),
        F1(head.p1y),
        F1(head.p2x),
        F1(head.p2y),
        color
    );
}

/// An invisible, wider copy of an edge's stroke, which takes the hover for its tooltip, since
/// the edge itself can be thinner than a pixel. `shape` is the element's tag and shape, e.g.
/// `path d="M 0 0 L 9 9"`
fn hit_area(out: &mut Parts, shape: &str, width: f64, edge_id: &str, tip: &str) {
    let _ = write!(
        out.part(),
        "<{shape} fill=\"none\" stroke=\"transparent\" stroke-width=\"{}\" \
         pointer-events=\"stroke\" data-edge=\"{edge_id}\" data-attrs='{tip}'/>",
        F1(geometry::py_max(8.0, width + 4.0))
    );
}

/// An edge's drawing outside lean SVG: its style, its ID and its tooltip
struct Stroke<'a> {
    color: &'a str,
    width: f64,
    opacity: f64,
    edge_id: &'a str,
    tip: &'a str,
}

/// A curved edge or a self-loop, with its ID for a label to follow, then its hit area
fn draw_curve(out: &mut Parts, d: &str, head: Option<Head>, directed: bool, s: &Stroke<'_>) {
    let (color, width, opacity) = (s.color, Repr(s.width), Repr(s.opacity));
    let p = out.part();
    if directed {
        let _ = write!(
            p,
            "<g opacity=\"{opacity}\"><path id=\"path-{}\" d=\"{d}\" fill=\"none\" \
             stroke=\"{color}\" stroke-width=\"{width}\" stroke-linecap=\"round\"/>",
            s.edge_id
        );
        if let Some(head) = head {
            polygon(p, &head, color);
        }
        p.push_str("</g>");
    } else {
        let _ = write!(
            p,
            "<path id=\"path-{}\" d=\"{d}\" fill=\"none\" stroke=\"{color}\" \
             stroke-width=\"{width}\" stroke-opacity=\"{opacity}\" stroke-linecap=\"round\"/>",
            s.edge_id
        );
    }
    hit_area(out, &format!("path d=\"{d}\""), s.width, s.edge_id, s.tip);
}

/// An edge routed around nodes, then its hit area. Nothing (an empty line) for a route of
/// fewer than two points
fn draw_routed(
    out: &mut Parts,
    route: &[(f64, f64)],
    ends: (f64, f64),
    directed: bool,
    s: &Stroke<'_>,
) {
    let full = geometry::waypoints_path(route, ends.0, ends.1);
    if full.is_empty() {
        out.part();
        return;
    }
    let (color, width, opacity) = (s.color, Repr(s.width), Repr(s.opacity));
    let (d, head) = geometry::routed(route, ends.0, ends.1, s.width, directed);
    let p = out.part();
    match head {
        Some(head) => {
            let _ = write!(
                p,
                "<g opacity=\"{opacity}\"><path d=\"{d}\" fill=\"none\" stroke=\"{color}\" \
                 stroke-width=\"{width}\" stroke-linecap=\"round\"/>"
            );
            polygon(p, &head, color);
            p.push_str("</g>");
        }
        None => {
            let _ = write!(
                p,
                "<path d=\"{d}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"{width}\" \
                 stroke-opacity=\"{opacity}\" stroke-linecap=\"round\"/>"
            );
        }
    }
    hit_area(
        out,
        &format!("path d=\"{full}\""),
        s.width,
        s.edge_id,
        s.tip,
    );
}

/// A straight edge, then its hit area. Nothing (an empty line) for one whose ends are at one
/// point
fn draw_straight(out: &mut Parts, line: Option<geometry::Straight>, s: &Stroke<'_>) {
    let Some(line) = line else {
        out.part();
        return;
    };
    let (color, width, opacity) = (s.color, Repr(s.width), Repr(s.opacity));
    let (sx, sy, lx, ly) = (F1(line.sx), F1(line.sy), F1(line.lx), F1(line.ly));
    let p = out.part();
    match line.head {
        Some(head) => {
            let _ = write!(
                p,
                "<g opacity=\"{opacity}\"><line x1=\"{sx}\" y1=\"{sy}\" x2=\"{lx}\" y2=\"{ly}\" \
                 stroke=\"{color}\" stroke-width=\"{width}\" stroke-linecap=\"round\"/>"
            );
            polygon(p, &head, color);
            p.push_str("</g>");
        }
        None => {
            let _ = write!(
                p,
                "<line x1=\"{sx}\" y1=\"{sy}\" x2=\"{lx}\" y2=\"{ly}\" stroke=\"{color}\" \
                 stroke-width=\"{width}\" stroke-opacity=\"{opacity}\" stroke-linecap=\"round\"/>"
            );
        }
    }
    let shape = format!(
        "line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"",
        F1(line.sx),
        F1(line.sy),
        F1(line.ex),
        F1(line.ey)
    );
    hit_area(out, &shape, s.width, s.edge_id, s.tip);
}

/// A straight or routed edge in lean SVG, as `draw_straight()` and `draw_routed()` draw it: its
/// path's `d` and its arrowhead
#[allow(clippy::too_many_arguments)]
fn lean_line(
    route: Option<&[(f64, f64)]>,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    r_src: f64,
    r_dst: f64,
    width: f64,
    directed: bool,
) -> Option<(String, Option<Head>)> {
    if let Some(route) = route {
        return Some(geometry::routed(route, r_src, r_dst, width, directed));
    }
    let line = geometry::straight(x1, y1, x2, y2, r_src, r_dst, width, directed)?;
    let d = format!(
        "M{} {}L{} {}",
        F1(line.sx),
        F1(line.sy),
        F1(line.lx),
        F1(line.ly)
    );
    Some((d, line.head))
}

fn draw_edges(scene: &Scene, style: &Style<'_>, out: &mut Parts) {
    let (nodes, edges) = (&scene.nodes, &scene.edges);
    let directed = scene.directed;
    let mut runs = Runs::default();
    for &i in &edges.order {
        let (s, t) = (edges.src[i], edges.tgt[i]);
        let (x1, y1, x2, y2) = (nodes.x[s], nodes.y[s], nodes.x[t], nodes.y[t]);
        let (r_src, r_dst) = (nodes.r[s], nodes.r[t]);
        let color = style.colors[edges.color[i]].as_str();
        let (width, opacity) = (edges.width[i], edges.opacity[i]);
        let bow = edges.bow[i];
        let route = edges.route[i].as_deref();
        let edge_id = || format!("{}-{}-{}", nodes.ids[s], nodes.ids[t], i);

        if scene.lean {
            // The edge joins the run of edges drawn in its style, and its tooltip is its row
            // (`data-e`) in the table of them
            let shape = if s == t {
                Some(geometry::self_loop(
                    x1,
                    y1,
                    r_src,
                    width,
                    directed,
                    edges.loop_size[i],
                ))
            } else if scene.curved || bow.abs() > 1e-6 {
                let bow = if scene.curved { 0.2 + bow } else { bow };
                geometry::curved(x1, y1, x2, y2, r_src, r_dst, width, directed, bow)
            } else {
                lean_line(route, x1, y1, x2, y2, r_src, r_dst, width, directed)
            };
            let Some((d, head)) = shape else { continue };
            let (w, o) = (Repr(width), Repr(opacity));
            let groups = vec![
                format!(
                    "<g fill=\"none\" stroke=\"{color}\" stroke-width=\"{w}\" \
                     stroke-opacity=\"{o}\" stroke-linecap=\"round\">"
                ),
                format!("<g fill=\"{color}\" fill-opacity=\"{o}\">"),
            ];
            // Curved edges keep their IDs for their labels, which follow them
            let path_id = if scene.curved && scene.edge_labels.is_some() {
                format!(" id=\"path-{}\"", edge_id())
            } else {
                String::new()
            };
            let line = format!("<path data-e=\"{i}\"{path_id} d=\"{d}\"/>");
            let head = head.map(|h| {
                format!(
                    "<path data-e=\"{i}\" d=\"M{} {}L{} {}L{} {}Z\"/>",
                    F1(h.ex),
                    F1(h.ey),
                    F1(h.p1x),
                    F1(h.p1y),
                    F1(h.p2x),
                    F1(h.p2y)
                )
            });
            runs.add(out, groups, &[Some(&line), head.as_deref()]);
            continue;
        }

        let edge_id = edge_id();
        let stroke = Stroke {
            color,
            width,
            opacity,
            edge_id: &edge_id,
            tip: &edges.tips[i],
        };
        if s == t {
            let (d, head) = geometry::self_loop(x1, y1, r_src, width, directed, edges.loop_size[i]);
            draw_curve(out, &d, head, directed, &stroke);
        } else if scene.curved || bow.abs() > 1e-6 {
            let bow = if scene.curved { 0.2 + bow } else { bow };
            match geometry::curved(x1, y1, x2, y2, r_src, r_dst, width, directed, bow) {
                Some((d, head)) => draw_curve(out, &d, head, directed, &stroke),
                None => {
                    out.part();
                }
            }
        } else if let Some(route) = route {
            draw_routed(out, route, (r_src, r_dst), directed, &stroke);
        } else {
            let line = geometry::straight(x1, y1, x2, y2, r_src, r_dst, width, directed);
            draw_straight(out, line, &stroke);
        }
    }
    runs.flush(out);
}

fn draw_edge_labels(scene: &Scene, style: &Style<'_>, out: &mut Parts) {
    let Some(labels) = &scene.edge_labels else {
        return;
    };
    let (nodes, edges) = (&scene.nodes, &scene.edges);
    let (font, label_color, bg) = (style.font_family, style.label_color, style.bg);

    // Each edge's place among the edges between the same two nodes, numbered as the fans of
    // parallel edges are
    let mut fans: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
    for i in 0..edges.src.len() {
        let (s, t) = (edges.src[i], edges.tgt[i]);
        if s != t {
            let (a, b) = (nodes.ids[s].as_str(), nodes.ids[t].as_str());
            fans.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    let mut fan_place: Vec<Option<(usize, &[usize])>> = vec![None; edges.src.len()];
    for fan in fans.values() {
        for (k, &i) in fan.iter().enumerate() {
            fan_place[i] = Some((k, fan));
        }
    }

    for (i, label) in labels.iter().enumerate() {
        let Some(label) = label else { continue };
        let (s, t) = (edges.src[i], edges.tgt[i]);
        let (src, tgt) = (nodes.ids[s].as_str(), nodes.ids[t].as_str());
        let (x1, y1, x2, y2) = (nodes.x[s], nodes.y[s], nodes.x[t], nodes.y[t]);
        let edge_id = format!("{src}-{tgt}-{i}");
        let alone = [i];
        let (k, fan) = fan_place[i].unwrap_or((0, &alone[..]));
        let text = Escaped(label);

        if scene.curved {
            // A label follows its edge's path, and reads from left to right along it, so an
            // edge that runs from right to left gets a reversed copy of its path to follow
            let mut path_ref = format!("path-{edge_id}");
            let mut reversed = false;
            if x1 > x2 {
                let dx_e = x2 - x1;
                let dy_e = y2 - y1;
                let d_e = (dx_e * dx_e + dy_e * dy_e).sqrt();
                if d_e > 1e-6 {
                    let (ux_e, uy_e) = (dx_e / d_e, dy_e / d_e);
                    let (px_e, py_e) = (-uy_e, ux_e);
                    let (r_src, r_dst) = (nodes.r[s], nodes.r[t]);
                    let bow_val = 0.2 + edges.bow[i];
                    let cmx = (x1 + x2) / 2.0 + px_e * d_e * bow_val;
                    let cmy = (y1 + y2) / 2.0 + py_e * d_e * bow_val;
                    let cdx_s2 = cmx - x2;
                    let cdy_s2 = cmy - y2;
                    let cd_s2 = (cdx_s2 * cdx_s2 + cdy_s2 * cdy_s2).sqrt();
                    let (sx2, sy2) = if cd_s2 > 1e-6 {
                        (x2 + (cdx_s2 / cd_s2) * r_dst, y2 + (cdy_s2 / cd_s2) * r_dst)
                    } else {
                        (x2, y2)
                    };
                    let cdx_e2 = cmx - x1;
                    let cdy_e2 = cmy - y1;
                    let cd_e2 = (cdx_e2 * cdx_e2 + cdy_e2 * cdy_e2).sqrt();
                    let (ex2, ey2) = if cd_e2 > 1e-6 {
                        (x1 + (cdx_e2 / cd_e2) * r_src, y1 + (cdy_e2 / cd_e2) * r_src)
                    } else {
                        (x1, y1)
                    };
                    path_ref = format!("path-{edge_id}-rev");
                    reversed = true;
                    let _ = write!(
                        out.part(),
                        "<path id=\"{path_ref}\" d=\"M {} {} Q {} {} {} {}\" fill=\"none\" \
                         stroke=\"none\"/>",
                        F1(sx2),
                        F1(sy2),
                        F1(cmx),
                        F1(cmy),
                        F1(ex2),
                        F1(ey2)
                    );
                }
            }
            if style.raster.is_some() && !reversed {
                // The edge is in the image, so the label has a path of its own, drawn as
                // `draw_edges()` draws the edge's
                let (r_src, r_dst, width) = (nodes.r[s], nodes.r[t], edges.width[i]);
                let directed = scene.directed;
                let shape = if s == t {
                    let size = edges.loop_size[i];
                    Some(geometry::self_loop(x1, y1, r_src, width, directed, size))
                } else {
                    let bow = 0.2 + edges.bow[i];
                    geometry::curved(x1, y1, x2, y2, r_src, r_dst, width, directed, bow)
                };
                if let Some((d, _)) = shape {
                    let _ = write!(
                        out.part(),
                        "<path id=\"{path_ref}\" d=\"{d}\" fill=\"none\" stroke=\"none\"/>"
                    );
                }
            }
            let _ = write!(
                out.part(),
                "<text font-family=\"{font}\" font-size=\"10\" fill=\"{label_color}\" \
                 fill-opacity=\"0.7\" pointer-events=\"none\"><textPath href=\"#{path_ref}\" \
                 startOffset=\"50%\" text-anchor=\"middle\">{text}</textPath></text>"
            );
        } else if fan.len() >= 3 && (x1 - x2).hypot(y1 - y2) > 1e-6 {
            // Three or more edges between two nodes leave no room beside the inner ones, so
            // each label sits on its own edge, ringed by the background. The labels step along
            // the edges too, in the order of the curves, each step the length of a label's box
            // measured along the edges, so that neighbors don't overlap however the edges are
            // angled (as far as the edges are long enough)
            let (a_x, a_y, b_x, b_y) = if src <= tgt {
                (x1, y1, x2, y2)
            } else {
                (x2, y2, x1, y1)
            };
            let dist = (a_x - b_x).hypot(a_y - b_y);
            let box_w = fan
                .iter()
                .filter_map(|&j| labels[j].as_deref())
                .map(|text| text.chars().count() as f64 * 10.0 * 0.58)
                .fold(f64::NEG_INFINITY, geometry::py_max);
            let step = (box_w * (b_x - a_x).abs() + 11.0 * (b_y - a_y).abs()) / dist + 3.0;
            let rank = k as f64 - (fan.len() - 1) as f64 / 2.0;
            let t_ = 0.5 + rank * py_min(step / dist, 0.7 / (fan.len() - 1) as f64);
            // The edge's offset from the straight line, on the side of the A-to-B normal
            let v = if src <= tgt {
                edges.bow[i]
            } else {
                -edges.bow[i]
            };
            let bulge = dist * v * 2.0 * t_ * (1.0 - t_);
            let mx = a_x + (b_x - a_x) * t_ - (b_y - a_y) / dist * bulge;
            let my = a_y + (b_y - a_y) * t_ + (b_x - a_x) / dist * bulge;
            let _ = write!(
                out.part(),
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"{font}\" \
                 font-size=\"10\" fill=\"{label_color}\" fill-opacity=\"0.7\" stroke=\"{bg}\" \
                 stroke-width=\"3\" stroke-linejoin=\"round\" paint-order=\"stroke\" \
                 pointer-events=\"none\">{text}</text>",
                F1(mx),
                F1(my + 3.5)
            );
        } else {
            // At the middle, out to the side of the edge. A parallel edge is drawn as a curve,
            // so its label goes by the middle of the curve, on the outside of it. Beside a
            // steep edge, the text starts or ends at that point, so it stays clear of the line
            let mut mx = (x1 + x2) / 2.0;
            let mut my = (y1 + y2) / 2.0;
            let mut anchor = "middle";
            let (dx, dy) = (x2 - x1, y2 - y1);
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 1e-6 {
                let (mut px, mut py) = (-dy / dist, dx / dist);
                let bow = edges.bow[i];
                let offset = if bow.abs() > 1e-6 {
                    8.0 + dist * bow.abs() / 2.0
                } else {
                    8.0
                };
                if bow < -1e-6 {
                    (px, py) = (-px, -py);
                }
                mx += px * offset;
                my += py * offset;
                if px.abs() > 0.7 {
                    anchor = if px > 0.0 { "start" } else { "end" };
                    my += 3.0;
                }
            }
            let _ = write!(
                out.part(),
                "<text x=\"{}\" y=\"{}\" text-anchor=\"{anchor}\" font-family=\"{font}\" \
                 font-size=\"10\" fill=\"{label_color}\" fill-opacity=\"0.7\" \
                 pointer-events=\"none\">{text}</text>",
                F1(mx),
                F1(my)
            );
        }
    }
}

/// The gradients of the nodes' fills, shaded, in the order of the color table
fn draw_gradients(scene: &Scene, style: &Style<'_>, out: &mut Parts) {
    let Some(shade) = style.shade else {
        return;
    };
    let mut used = vec![false; style.colors.len()];
    scene.nodes.fill.iter().for_each(|&c| used[c] = true);
    let buf = out.part();
    buf.push_str("<defs>");
    // Its box is the node's, so that one gradient serves every node of the color
    let center = General((1.0 + shade::HIGHLIGHT) / 2.0);
    let reach = General(shade::REACH / 2.0);
    for (k, (id, hi, lo)) in shade.iter().enumerate().filter(|&(k, _)| used[k]) {
        let _ = write!(
            buf,
            "<radialGradient id=\"{id}\" cx=\"{center}\" cy=\"{center}\" r=\"{reach}\">\
             <stop offset=\"0\" stop-color=\"{hi}\"/><stop offset=\"{}\" stop-color=\"{}\"/>\
             <stop offset=\"1\" stop-color=\"{lo}\"/></radialGradient>",
            General(shade::MID),
            style.colors[k]
        );
    }
    buf.push_str("</defs>");
}

fn draw_nodes(scene: &Scene, style: &Style<'_>, out: &mut Parts) {
    let nodes = &scene.nodes;
    let mut runs = Runs::default();
    // Shaded, each fill is its gradient
    let fills: Vec<String> = match style.shade {
        Some(shade) => shade.iter().map(|(id, ..)| format!("url(#{id})")).collect(),
        None => style.colors.to_vec(),
    };
    for &k in &nodes.order {
        let (x, y, r) = (F1(nodes.x[k]), F1(nodes.y[k]), F1(nodes.r[k]));
        let fill = &fills[nodes.fill[k]];
        let border = &style.colors[nodes.border[k]];
        let border_width = General(nodes.border_width[k]);
        let border_opacity = Repr(nodes.border_opacity[k]);
        let (id, tip) = (Escaped(&nodes.ids[k]), &nodes.tips[k]);
        if scene.lean {
            let group = format!(
                "<g fill=\"{fill}\" stroke=\"{border}\" stroke-width=\"{border_width}\" \
                 stroke-opacity=\"{border_opacity}\">"
            );
            let circle = format!(
                "<circle cx=\"{x}\" cy=\"{y}\" r=\"{r}\" data-node=\"{id}\" data-attrs='{tip}'/>"
            );
            runs.add(out, vec![group], &[Some(&circle)]);
            continue;
        }
        let _ = write!(
            out.part(),
            "<circle cx=\"{x}\" cy=\"{y}\" r=\"{r}\" fill=\"{fill}\" stroke=\"{border}\" \
             stroke-width=\"{border_width}\" stroke-opacity=\"{border_opacity}\" \
             data-node=\"{id}\" data-attrs='{tip}' />"
        );
    }
    runs.flush(out);
}

fn draw_labels(scene: &Scene, style: &Style<'_>, out: &mut Parts) {
    let Some(labels) = &scene.labels else {
        return;
    };
    let nodes = &scene.nodes;
    let size = labels.font_size;
    for (k, text) in labels.texts.iter().enumerate() {
        if !labels.shown[k] {
            continue;
        }
        // Under the node
        let ly = nodes.y[k] + nodes.r[k] + size + 2.0;
        let _ = write!(
            out.part(),
            "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"{}\" \
             font-size=\"{}\" fill=\"{}\" pointer-events=\"none\">{}</text>",
            F1(nodes.x[k]),
            F1(ly),
            style.font_family,
            F1(size),
            style.colors[labels.colors[k]],
            Escaped(text)
        );
    }
}
