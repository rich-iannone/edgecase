//! Graph coloring algorithms
//!
//! Graph coloring assigns labels (colors) to graph elements subject to constraints.
//! Vertex coloring assigns colors to nodes such that no two adjacent nodes share
//! the same color.
//!
//! This module provides:
//!
//! - `greedy_color`: Greedy vertex coloring with multiple ordering strategies
//! - `is_valid_coloring`: Verify a coloring is valid

use crate::graph::Graph;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;

/// Ordering strategies for greedy coloring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColoringStrategy {
    /// Process nodes in arbitrary order (order they appear in graph)
    Arbitrary,
    /// Process nodes in order of decreasing degree (largest degree first)
    LargestFirst,
    /// Process nodes in order of increasing degree (smallest degree first)
    SmallestLast,
    /// Random order
    Random,
    /// Saturation Largest First (DSATUR) - prioritize nodes with most colored neighbors
    SaturationLargestFirst,
    /// Independent set based coloring
    IndependentSet,
}

impl ColoringStrategy {
    fn from_str(s: &str) -> PyResult<Self> {
        match s.to_lowercase().as_str() {
            "arbitrary" | "random_sequential" => Ok(ColoringStrategy::Arbitrary),
            "largest_first" => Ok(ColoringStrategy::LargestFirst),
            "smallest_last" => Ok(ColoringStrategy::SmallestLast),
            "random" => Ok(ColoringStrategy::Random),
            "saturation_largest_first" | "dsatur" => Ok(ColoringStrategy::SaturationLargestFirst),
            "independent_set" => Ok(ColoringStrategy::IndependentSet),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown coloring strategy: '{}'. Valid options: 'arbitrary', 'largest_first', \
                 'smallest_last', 'random', 'saturation_largest_first', 'independent_set'",
                s
            ))),
        }
    }
}

/// Color a graph using a greedy algorithm with the specified strategy.
///
/// The greedy coloring algorithm assigns colors to nodes one at a time,
/// always using the smallest available color that doesn't conflict with
/// already-colored neighbors. The order in which nodes are processed
/// affects the number of colors used.
///
/// # Arguments
/// * `graph` - The input graph (directed or undirected)
/// * `strategy` - Ordering strategy for processing nodes:
///   - "arbitrary": Process in graph order (default)
///   - "largest_first": Process high-degree nodes first
///   - "smallest_last": Process low-degree nodes first (often produces better results)
///   - "random": Random ordering
///   - "saturation_largest_first" / "dsatur": Prioritize nodes with most colored neighbors
///   - "independent_set": Color independent sets iteratively
/// * `interchange` - Whether to use color interchange optimization (not yet implemented)
///
/// # Returns
/// A dictionary mapping node IDs to their assigned color (0-indexed integers)
///
/// # Notes
/// - Greedy coloring doesn't guarantee optimal (minimum) colors
/// - For undirected graphs, ensures no adjacent nodes share a color
/// - For directed graphs, treats edges as undirected for coloring purposes
/// - The number of colors used is at most max_degree + 1
#[pyfunction]
#[pyo3(signature = (graph, strategy=None, interchange=None))]
pub fn greedy_color(
    graph: &Graph,
    strategy: Option<String>,
    interchange: Option<bool>,
) -> PyResult<Py<PyDict>> {
    let strat = match strategy {
        Some(s) => ColoringStrategy::from_str(&s)?,
        None => ColoringStrategy::Arbitrary,
    };

    let _use_interchange = interchange.unwrap_or(false);
    // Note: interchange optimization not yet implemented

    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Python::with_gil(|py| Ok(PyDict::new(py).into()));
    }

    // Build adjacency list (treating directed as undirected for coloring)
    let mut adj_list: FxHashMap<u64, FxHashSet<u64>> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());

    // Initialize empty sets for all nodes
    for &node_id in &node_ids {
        adj_list.insert(node_id, FxHashSet::default());
    }

    // Add outgoing edges
    for &node_id in &node_ids {
        if let Ok(out_neighbors) = graph.neighbors(node_id) {
            for neighbor in out_neighbors {
                if neighbor != node_id {
                    // Skip self-loops
                    adj_list.entry(node_id).or_default().insert(neighbor);
                    // For undirected graphs or treating directed as undirected,
                    // add reverse edge as well
                    adj_list.entry(neighbor).or_default().insert(node_id);
                }
            }
        }
    }

    // Get ordered list of nodes based on strategy
    let ordered_nodes = order_nodes_by_strategy(&node_ids, &adj_list, strat)?;

    // Perform greedy coloring
    let coloring = match strat {
        ColoringStrategy::SaturationLargestFirst => {
            dsatur_coloring(&node_ids, &adj_list)?
        }
        ColoringStrategy::IndependentSet => {
            independent_set_coloring(&node_ids, &adj_list)?
        }
        _ => {
            greedy_sequential_coloring(&ordered_nodes, &adj_list)?
        }
    };

    // Convert to Python dict
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        for (node, color) in coloring {
            dict.set_item(node, color)?;
        }
        Ok(dict.into())
    })
}

/// Order nodes according to the specified strategy
fn order_nodes_by_strategy(
    node_ids: &[u64],
    adj_list: &FxHashMap<u64, FxHashSet<u64>>,
    strategy: ColoringStrategy,
) -> PyResult<Vec<u64>> {
    let mut ordered = node_ids.to_vec();

    match strategy {
        ColoringStrategy::Arbitrary => {
            // Keep original order
        }
        ColoringStrategy::LargestFirst => {
            // Sort by degree descending
            ordered.sort_by_key(|&node| {
                Reverse(adj_list.get(&node).map(|s| s.len()).unwrap_or(0))
            });
        }
        ColoringStrategy::SmallestLast => {
            // Smallest-last ordering: repeatedly remove min-degree node
            ordered = smallest_last_ordering(node_ids, adj_list);
        }
        ColoringStrategy::Random => {
            // Shuffle randomly
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
            ordered.shuffle(&mut rng);
        }
        ColoringStrategy::SaturationLargestFirst | ColoringStrategy::IndependentSet => {
            // These use their own ordering internally
        }
    }

    Ok(ordered)
}

/// Compute smallest-last ordering
/// Repeatedly removes the node with minimum degree from the remaining graph
fn smallest_last_ordering(
    node_ids: &[u64],
    adj_list: &FxHashMap<u64, FxHashSet<u64>>,
) -> Vec<u64> {
    let n = node_ids.len();
    let mut result = Vec::with_capacity(n);
    let mut remaining: FxHashSet<u64> = node_ids.iter().copied().collect();

    // Track current degrees in remaining subgraph
    let mut degrees: FxHashMap<u64, usize> = FxHashMap::default();
    for &node in node_ids {
        let deg = adj_list
            .get(&node)
            .map(|neighbors| neighbors.iter().filter(|n| remaining.contains(n)).count())
            .unwrap_or(0);
        degrees.insert(node, deg);
    }

    while !remaining.is_empty() {
        // Find node with minimum degree in remaining graph
        let min_node = *remaining
            .iter()
            .min_by_key(|&&node| degrees.get(&node).unwrap_or(&usize::MAX))
            .unwrap();

        // Remove this node and update neighbor degrees
        remaining.remove(&min_node);
        if let Some(neighbors) = adj_list.get(&min_node) {
            for &neighbor in neighbors {
                if remaining.contains(&neighbor) {
                    if let Some(deg) = degrees.get_mut(&neighbor) {
                        *deg = deg.saturating_sub(1);
                    }
                }
            }
        }

        result.push(min_node);
    }

    // Reverse to get smallest-last order (process in reverse removal order)
    result.reverse();
    result
}

/// Basic greedy sequential coloring
fn greedy_sequential_coloring(
    ordered_nodes: &[u64],
    adj_list: &FxHashMap<u64, FxHashSet<u64>>,
) -> PyResult<FxHashMap<u64, usize>> {
    let mut coloring: FxHashMap<u64, usize> = FxHashMap::default();

    for &node in ordered_nodes {
        // Find colors used by neighbors
        let mut neighbor_colors: FxHashSet<usize> = FxHashSet::default();
        if let Some(neighbors) = adj_list.get(&node) {
            for &neighbor in neighbors {
                if let Some(&color) = coloring.get(&neighbor) {
                    neighbor_colors.insert(color);
                }
            }
        }

        // Find smallest available color
        let mut color = 0;
        while neighbor_colors.contains(&color) {
            color += 1;
        }

        coloring.insert(node, color);
    }

    Ok(coloring)
}

/// DSATUR (Degree of Saturation) algorithm
/// Prioritizes nodes with the most differently-colored neighbors
fn dsatur_coloring(
    node_ids: &[u64],
    adj_list: &FxHashMap<u64, FxHashSet<u64>>,
) -> PyResult<FxHashMap<u64, usize>> {
    let n = node_ids.len();
    let mut coloring: FxHashMap<u64, usize> = FxHashMap::with_capacity_and_hasher(n, Default::default());
    let mut uncolored: FxHashSet<u64> = node_ids.iter().copied().collect();

    // Track saturation (number of different colors in neighborhood)
    let mut saturation: FxHashMap<u64, FxHashSet<usize>> = FxHashMap::default();
    for &node in node_ids {
        saturation.insert(node, FxHashSet::default());
    }

    while !uncolored.is_empty() {
        // Select node with maximum saturation, breaking ties by degree
        let next_node = *uncolored
            .iter()
            .max_by(|&&a, &&b| {
                let sat_a = saturation.get(&a).map(|s| s.len()).unwrap_or(0);
                let sat_b = saturation.get(&b).map(|s| s.len()).unwrap_or(0);
                let deg_a = adj_list.get(&a).map(|s| s.len()).unwrap_or(0);
                let deg_b = adj_list.get(&b).map(|s| s.len()).unwrap_or(0);
                sat_a.cmp(&sat_b).then(deg_a.cmp(&deg_b))
            })
            .unwrap();

        // Find smallest available color
        let neighbor_colors = saturation.get(&next_node).cloned().unwrap_or_default();
        let mut color = 0;
        while neighbor_colors.contains(&color) {
            color += 1;
        }

        // Assign color
        coloring.insert(next_node, color);
        uncolored.remove(&next_node);

        // Update saturation of uncolored neighbors
        if let Some(neighbors) = adj_list.get(&next_node) {
            for &neighbor in neighbors {
                if uncolored.contains(&neighbor) {
                    if let Some(sat) = saturation.get_mut(&neighbor) {
                        sat.insert(color);
                    }
                }
            }
        }
    }

    Ok(coloring)
}

/// Independent set based coloring
/// Finds independent sets and assigns each set a unique color
fn independent_set_coloring(
    node_ids: &[u64],
    adj_list: &FxHashMap<u64, FxHashSet<u64>>,
) -> PyResult<FxHashMap<u64, usize>> {
    let mut coloring: FxHashMap<u64, usize> = FxHashMap::default();
    let mut uncolored: FxHashSet<u64> = node_ids.iter().copied().collect();
    let mut current_color = 0;

    while !uncolored.is_empty() {
        // Find a maximal independent set among uncolored nodes
        let mut independent_set: FxHashSet<u64> = FxHashSet::default();
        let mut excluded: FxHashSet<u64> = FxHashSet::default();

        // Sort uncolored by degree (prefer high-degree nodes first)
        let mut candidates: Vec<u64> = uncolored.iter().copied().collect();
        candidates.sort_by_key(|&node| {
            Reverse(
                adj_list
                    .get(&node)
                    .map(|neighbors| neighbors.iter().filter(|n| uncolored.contains(n)).count())
                    .unwrap_or(0),
            )
        });

        for node in candidates {
            if !excluded.contains(&node) {
                independent_set.insert(node);
                excluded.insert(node);

                // Exclude all neighbors
                if let Some(neighbors) = adj_list.get(&node) {
                    for &neighbor in neighbors {
                        excluded.insert(neighbor);
                    }
                }
            }
        }

        // Assign current color to all nodes in independent set
        for node in independent_set {
            coloring.insert(node, current_color);
            uncolored.remove(&node);
        }

        current_color += 1;
    }

    Ok(coloring)
}

/// Check if a coloring is valid (no adjacent nodes share a color)
///
/// # Arguments
/// * `graph` - The input graph
/// * `coloring` - HashMap mapping node IDs to colors
///
/// # Returns
/// True if the coloring is valid, False otherwise
#[pyfunction]
pub fn is_valid_coloring(
    graph: &Graph,
    coloring: std::collections::HashMap<u64, i64>,
) -> PyResult<bool> {
    // Convert to FxHashMap for faster lookups
    let color_map: FxHashMap<u64, i64> = coloring.into_iter().collect();

    // Build symmetric adjacency for checking (treat directed as undirected)
    let node_ids = graph.get_node_ids();

    for &node in &node_ids {
        let node_color = match color_map.get(&node) {
            Some(&c) => c,
            None => continue, // Uncolored node is okay for validity check
        };

        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if neighbor == node {
                    continue; // Skip self-loops
                }
                if let Some(&neighbor_color) = color_map.get(&neighbor) {
                    if node_color == neighbor_color {
                        return Ok(false);
                    }
                }
            }
        }
    }

    Ok(true)
}

/// Get the chromatic number upper bound from a coloring
/// (The number of distinct colors used)
#[pyfunction]
pub fn chromatic_number_upper_bound(coloring: std::collections::HashMap<u64, i64>) -> PyResult<usize> {
    let colors: FxHashSet<i64> = coloring.values().copied().collect();
    Ok(colors.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new_rust(false, false);
        // Create a simple triangle
        graph.add_test_node("A");
        graph.add_test_node("B");
        graph.add_test_node("C");
        graph.add_test_edge_by_label("A", "B", None);
        graph.add_test_edge_by_label("B", "C", None);
        graph.add_test_edge_by_label("C", "A", None);
        graph
    }

    #[test]
    fn test_greedy_coloring_triangle() {
        pyo3::prepare_freethreaded_python();

        let graph = create_test_graph();
        let result = greedy_color(&graph, None, None);
        assert!(result.is_ok());

        Python::with_gil(|py| {
            let dict = result.unwrap();
            let bound = dict.bind(py);
            // Triangle needs exactly 3 colors
            let mut colors: FxHashSet<i64> = FxHashSet::default();
            for value in bound.values() {
                colors.insert(value.extract().unwrap());
            }
            assert_eq!(colors.len(), 3);
        });
    }

    #[test]
    fn test_smallest_last_ordering() {
        let mut adj_list: FxHashMap<u64, FxHashSet<u64>> = FxHashMap::default();
        adj_list.insert(1, [2, 3].iter().copied().collect());
        adj_list.insert(2, [1, 3].iter().copied().collect());
        adj_list.insert(3, [1, 2].iter().copied().collect());

        let nodes = vec![1, 2, 3];
        let ordering = smallest_last_ordering(&nodes, &adj_list);
        assert_eq!(ordering.len(), 3);
    }
}
