// Layout algorithms for graph visualization
pub mod force_directed;

pub use force_directed::{ForceDirectedLayout, LayoutConfig};

use pyo3::prelude::*;
use std::collections::HashMap;

/// Compute force-directed layout for a graph (Python interface)
///
/// # Arguments
/// * `node_count` - Number of nodes in the graph
/// * `edges` - List of edges as (source, target) tuples
/// * `iterations` - Number of layout iterations (default: 50)
/// * `use_barnes_hut` - Whether to use Barnes-Hut optimization (default: true for >100 nodes)
///
/// # Returns
/// Dictionary mapping node indices to (x, y) positions
#[pyfunction]
#[pyo3(signature = (node_count, edges, iterations=50, use_barnes_hut=None))]
pub fn force_directed_layout_rust(
    node_count: usize,
    edges: Vec<(usize, usize)>,
    iterations: usize,
    use_barnes_hut: Option<bool>,
) -> PyResult<HashMap<usize, (f64, f64)>> {
    let use_bh = use_barnes_hut.unwrap_or(node_count > 100);

    let config = LayoutConfig {
        iterations,
        k: None,
        area: None,
        use_barnes_hut: use_bh,
        theta: 0.5,
        damping: 0.9,
    };

    let layout = ForceDirectedLayout::new(config);
    let positions = layout.compute(node_count, &edges);

    Ok(positions)
}

