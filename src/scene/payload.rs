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
use super::{css, Scene};
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
    ]);
    let controls = scene.controls.as_ref();
    if let Some((level, base_width)) = controls.and_then(|c| c.highlight.as_ref()) {
        let level: Vec<usize> = order.iter().map(|&i| level[i] as usize).collect();
        columns.push(("level", u32s(&level)));
        columns.push(("base_width", f32s(&in_order(base_width, order))));
    }

    let mut light = Vec::with_capacity(scene.colors.len());
    let mut dark = Vec::with_capacity(scene.colors.len());
    for (lt, dk) in &scene.colors {
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
