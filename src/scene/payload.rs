// The scene as the viewer's payload: a JSON header, then the columns that the viewer uploads to
// the GPU as they are, little-endian
//
//   "ECV1", the header's length in bytes (u32), the header, padded with spaces to 4 bytes,
//   then each column at its offset, 4-byte aligned
//
// The header has the counts, the frame, the color table in each theme (RGBA from 0 to 1) and,
// under "columns", each column's type, offset in bytes and length. A column whose entries are
// all the same is left out, and its value is under "constants" instead. The edges are in the
// order in which they are drawn.
//
// For the viewer's controls, "style" has the node size, edge width and edge opacity that the
// encodings started from and whether the edges are curved (by 0.2 of their length), and with
// highlighting, the edges' "level" and "base_width" columns have each one's level (0 muted,
// 1 soft, 2 accent) and its width before highlighting.
//
// For its selection, "select" has the indexes of the colors it draws with, which may run past
// the scene's own (the SVG's) into those after them, and whether color_by= and edge_color_by=
// color the nodes and edges; with highlighting, the columns "base_fill", "base_border",
// "base_color" and "base_label_order" have what highlighting changed as it was without it; and
// with a legend, "node_key" and "edge_key" have the swatch each node and edge falls under (0 for
// none, else the swatch's place plus 1). The nodes' IDs are the columns "node_id" and
// "node_id_hi", their low and high 32 bits.
//
// For its filters, "filters" has the numeric attributes of the nodes and of the edges: each one's
// name, whose it is ("nodes" or "edges"), its column, the least and the most of its values,
// whether they are all ints, and how many have none. The column "filter_<k>" has each node's or
// edge's value less the least, which keeps a float's precision for values far from 0, and NaN
// for none.
//
// With node labels, "labels" has their font size, whether they are shown, the most of them that
// are drawn ("limit", null for all) and what that is when `max_labels=` doesn't say ("auto"), and
// the columns "label_order", the nodes in order of priority (see `labels.rs`), and "label_color".
// Their texts aren't in the payload (see `Scene.label_texts`).
use super::{css, Scene};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Map, Value};

pub const MAGIC: &[u8; 4] = b"ECV1";

enum Column {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

impl Column {
    /// The value of every entry, if they are all the same
    fn constant(&self) -> Option<Value> {
        match self {
            Column::F32(v) => {
                let first = *v.first()?;
                v.iter()
                    .all(|&x| x.to_bits() == first.to_bits())
                    .then(|| json!(first))
            }
            Column::U32(v) => {
                let first = *v.first()?;
                v.iter().all(|&x| x == first).then(|| json!(first))
            }
        }
    }
}

fn f32s(values: &[f64]) -> Column {
    Column::F32(values.iter().map(|&v| v as f32).collect())
}

fn u32s(values: &[usize]) -> Column {
    Column::U32(values.iter().map(|&v| v as u32).collect())
}

fn in_order<T: Copy>(values: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|&i| values[i]).collect()
}

/// The bytes of the payload for `scene`
pub fn payload(scene: &Scene) -> PyResult<Vec<u8>> {
    let (nodes, edges) = (&scene.nodes, &scene.edges);
    let n = nodes.ids.len();
    let m = edges.src.len();
    let mut columns: Vec<(&str, Column)> = vec![
        ("x", f32s(&nodes.x)),
        ("y", f32s(&nodes.y)),
        ("r", f32s(&nodes.r)),
        ("fill", u32s(&nodes.fill)),
        ("border", u32s(&nodes.border)),
        ("border_width", f32s(&nodes.border_width)),
        ("border_opacity", f32s(&nodes.border_opacity)),
    ];
    // The order in which the nodes are drawn, the highlighted ones last, unless it's theirs
    if nodes.order.iter().enumerate().any(|(i, &k)| i != k) {
        columns.push(("order", u32s(&nodes.order)));
    }
    // The edges in the order in which they are drawn, the highlighted ones last. How far each
    // bows out is a share of its length, by `curved_points()`'s rule
    let order = &edges.order;
    let base = if scene.curved { 0.2 } else { 0.0 };
    let bow: Vec<f64> = order.iter().map(|&i| base + edges.bow[i]).collect();
    columns.extend([
        ("src", u32s(&in_order(&edges.src, order))),
        ("tgt", u32s(&in_order(&edges.tgt, order))),
        ("color", u32s(&in_order(&edges.color, order))),
        ("width", f32s(&in_order(&edges.width, order))),
        ("opacity", f32s(&in_order(&edges.opacity, order))),
        ("bow", f32s(&bow)),
        ("loop_size", f32s(&in_order(&edges.loop_size, order))),
    ]);
    if let Some(labels) = &scene.labels {
        columns.push(("label_order", u32s(&labels.order)));
        columns.push(("label_color", u32s(&labels.colors)));
    }
    let controls = scene.controls.as_ref();
    if let Some((level, base_width)) = controls.and_then(|c| c.highlight.as_ref()) {
        let level: Vec<usize> = order.iter().map(|&i| level[i] as usize).collect();
        columns.push(("level", u32s(&level)));
        columns.push(("base_width", f32s(&in_order(base_width, order))));
    }
    let filters = controls.map_or(&[][..], |c| c.filters.as_slice());
    let filter_names: Vec<String> = (0..filters.len()).map(|k| format!("filter_{k}")).collect();
    for (f, name) in filters.iter().zip(&filter_names) {
        let values = if f.edges {
            in_order(&f.values, order)
        } else {
            f.values.clone()
        };
        let column = values.iter().map(|&v| (v - f.min) as f32).collect();
        columns.push((name, Column::F32(column)));
    }
    let selection = controls.map(|c| &c.selection);
    if let Some(sel) = selection {
        // The nodes' IDs, which the viewer's selection writes out
        let mut ids = Vec::with_capacity(n);
        for id in &nodes.ids {
            ids.push(
                id.parse::<u64>()
                    .map_err(|_| PyValueError::new_err(format!("node ID {id} isn't an integer")))?,
            );
        }
        columns.push((
            "node_id",
            Column::U32(ids.iter().map(|&v| v as u32).collect()),
        ));
        columns.push((
            "node_id_hi",
            Column::U32(ids.iter().map(|&v| (v >> 32) as u32).collect()),
        ));
        if let Some(base) = &sel.base {
            columns.push(("base_fill", u32s(&base.fill)));
            columns.push(("base_border", u32s(&base.border)));
            columns.push(("base_color", u32s(&in_order(&base.color, order))));
            columns.push(("base_label_order", u32s(&base.label_order)));
        }
        if let Some((node_key, edge_key)) = &sel.keys {
            columns.push(("node_key", Column::U32(node_key.clone())));
            columns.push(("edge_key", Column::U32(in_order(edge_key, order))));
        }
    }

    let extra = selection.map_or(&[][..], |s| s.extra.as_slice());
    let mut light = Vec::with_capacity(scene.colors.len() + extra.len());
    let mut dark = Vec::with_capacity(scene.colors.len() + extra.len());
    for (lt, dk) in scene.colors.iter().chain(extra) {
        light.push(css::parse(lt)?.to_vec());
        dark.push(css::parse(dk)?.to_vec());
    }

    // The columns' places, after the header, which their places are in: the header's length
    // depends on the offsets, so they are counted from the start of the columns and moved
    // along by the header's length once that is known
    let mut constants = Map::new();
    let mut kept = Vec::new();
    let mut place = 0usize;
    let mut placed = Vec::new();
    for (name, column) in columns {
        if let Some(value) = column.constant() {
            constants.insert(name.to_string(), value);
            continue;
        }
        let (kind, len) = match &column {
            Column::F32(v) => ("f32", v.len()),
            Column::U32(v) => ("u32", v.len()),
        };
        placed.push((name, kind, place, len));
        place += 4 * len;
        kept.push(column);
    }
    let header_for = |start: usize| {
        let mut cols = Map::new();
        for &(name, kind, at, len) in &placed {
            cols.insert(
                name.to_string(),
                json!({"type": kind, "offset": start + at, "length": len}),
            );
        }
        json!({
            "version": 1,
            "nodes": n,
            "edges": m,
            "directed": scene.directed,
            "width": scene.width,
            "height": scene.height,
            "viewbox": [scene.viewbox.0, scene.viewbox.1, scene.viewbox.2, scene.viewbox.3],
            "legend_w": scene.legend_w,
            "colors": {"light": light, "dark": dark},
            "style": controls.map(|c| json!({
                "node_size": c.node_size,
                "edge_width": c.edge_width,
                "edge_opacity": c.edge_opacity,
                "curved": scene.curved,
            })),
            "labels": scene.labels.as_ref().map(|l| json!({
                "font_size": l.font_size,
                "show": l.show,
                "limit": l.limit,
                "auto": l.auto,
            })),
            "select": selection.map(|s| json!({
                "accent": s.accent,
                "accent_border": s.accent_border,
                "muted": s.muted,
                "muted_border": s.muted_border,
                "label": s.label,
                "muted_label": s.muted_label,
                "edge": s.edge,
                "node_colors": s.node_colors,
                "edge_colors": s.edge_colors,
            })),
            "filters": controls.map(|_| filters.iter().zip(&filter_names).map(|(f, name)| json!({
                "name": f.name,
                "of": if f.edges { "edges" } else { "nodes" },
                "column": name,
                "min": f.min,
                "max": f.max,
                "integer": f.integer,
                "missing": f.values.iter().filter(|v| v.is_nan()).count(),
            })).collect::<Vec<_>>()),
            "constants": constants,
            "columns": cols,
        })
        .to_string()
    };
    // The offsets grow the header by a digit now and then, so this settles in a few rounds
    let mut header = header_for(0);
    loop {
        let len = header.len().div_ceil(4) * 4;
        let next = header_for(8 + len);
        if next.len().div_ceil(4) * 4 == len {
            header = next;
            break;
        }
        header = next;
    }
    let padded = header.len().div_ceil(4) * 4;

    let mut out = Vec::with_capacity(8 + padded + place);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(padded as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.resize(8 + padded, b' ');
    for column in kept {
        match column {
            Column::F32(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
            Column::U32(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        }
    }
    Ok(out)
}
