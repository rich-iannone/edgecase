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
            "arbitrary" => Ok(ColoringStrategy::Arbitrary),
            "largest_first" => Ok(ColoringStrategy::LargestFirst),
            "smallest_last" => Ok(ColoringStrategy::SmallestLast),
            "random" | "random_sequential" => Ok(ColoringStrategy::Random),
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
/// * `interchange` - Whether to use the color interchange optimization: before opening a new
///   color for a node, try to free an existing one by swapping two colors along Kempe chains.
///   Not available with "saturation_largest_first"/"dsatur" or "independent_set".
/// * `seed` - Seed for the "random" strategy. If None, the order is drawn from system entropy.
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
#[pyo3(signature = (graph, strategy=None, interchange=None, seed=None))]
pub fn greedy_color(
    graph: &Graph,
    strategy: Option<String>,
    interchange: Option<bool>,
    seed: Option<u64>,
) -> PyResult<Py<PyDict>> {
    let strat = match strategy {
        Some(s) => ColoringStrategy::from_str(&s)?,
        None => ColoringStrategy::Arbitrary,
    };

    let use_interchange = interchange.unwrap_or(false);
    if use_interchange
        && matches!(
            strat,
            ColoringStrategy::SaturationLargestFirst | ColoringStrategy::IndependentSet
        )
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "interchange=True cannot be used with the 'saturation_largest_first' (DSATUR) or \
             'independent_set' strategies, which choose their own coloring order",
        ));
    }

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
    let ordered_nodes = order_nodes_by_strategy(&node_ids, &adj_list, strat, seed)?;

    // Perform greedy coloring
    let coloring = match strat {
        ColoringStrategy::SaturationLargestFirst => dsatur_coloring(&node_ids, &adj_list)?,
        ColoringStrategy::IndependentSet => independent_set_coloring(&node_ids, &adj_list)?,
        _ if use_interchange => interchange_coloring(&ordered_nodes, &adj_list),
        _ => greedy_sequential_coloring(&ordered_nodes, &adj_list)?,
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
    seed: Option<u64>,
) -> PyResult<Vec<u64>> {
    let mut ordered = node_ids.to_vec();

    match strategy {
        ColoringStrategy::Arbitrary => {
            // Keep original order
        }
        ColoringStrategy::LargestFirst => {
            // Sort by degree descending
            ordered.sort_by_key(|&node| Reverse(adj_list.get(&node).map(|s| s.len()).unwrap_or(0)));
        }
        ColoringStrategy::SmallestLast => {
            // Smallest-last ordering: repeatedly remove min-degree node
            ordered = smallest_last_ordering(node_ids, adj_list);
        }
        ColoringStrategy::Random => {
            // Shuffle randomly; sort first so a given seed gives the same order regardless of
            // the graph's internal (hash) storage order
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            ordered.sort_unstable();
            let mut rng = match seed {
                Some(s) => rand_chacha::ChaCha8Rng::seed_from_u64(s),
                None => rand_chacha::ChaCha8Rng::from_entropy(),
            };
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
fn smallest_last_ordering(node_ids: &[u64], adj_list: &FxHashMap<u64, FxHashSet<u64>>) -> Vec<u64> {
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

/// Greedy sequential coloring with color interchange (Kempe chain swaps)
///
/// Nodes are colored in `ordered_nodes` order with the smallest free color. When a node would
/// need a brand-new color, try each pair of existing colors (a, b): take the Kempe chains, the
/// connected components of the subgraph induced by nodes colored a or b, that contain the
/// node's a-colored neighbors. If none of those chains contains a b-colored neighbor of the
/// node, swapping a and b inside them keeps the coloring valid and leaves no neighbor with
/// color a, so the node takes a and no new color is opened.
fn interchange_coloring(
    ordered_nodes: &[u64],
    adj_list: &FxHashMap<u64, FxHashSet<u64>>,
) -> FxHashMap<u64, usize> {
    let mut coloring: FxHashMap<u64, usize> = FxHashMap::default();
    let mut num_colors = 0;

    for &node in ordered_nodes {
        let neighbors = adj_list.get(&node);
        let neighbor_colors: FxHashSet<usize> = neighbors
            .into_iter()
            .flatten()
            .filter_map(|nb| coloring.get(nb).copied())
            .collect();

        let mut color = 0;
        while neighbor_colors.contains(&color) {
            color += 1;
        }

        if color == num_colors && num_colors >= 2 {
            if let Some(freed) = try_interchange(node, adj_list, &mut coloring, num_colors) {
                color = freed;
            }
        }

        if color == num_colors {
            num_colors += 1;
        }
        coloring.insert(node, color);
    }

    coloring
}

/// Try to free a color for `node` by swapping two colors along Kempe chains.
/// Returns the freed color (already vacated around `node`), or None if no swap works.
fn try_interchange(
    node: u64,
    adj_list: &FxHashMap<u64, FxHashSet<u64>>,
    coloring: &mut FxHashMap<u64, usize>,
    num_colors: usize,
) -> Option<usize> {
    let neighbors: Vec<u64> = adj_list.get(&node)?.iter().copied().collect();

    for a in 0..num_colors {
        for b in 0..num_colors {
            if a == b {
                continue;
            }

            // Kempe chains through the node's a-colored neighbors, in the {a, b} subgraph
            let mut chain: FxHashSet<u64> = FxHashSet::default();
            let mut stack: Vec<u64> = neighbors
                .iter()
                .copied()
                .filter(|nb| coloring.get(nb) == Some(&a))
                .collect();
            chain.extend(stack.iter().copied());
            while let Some(current) = stack.pop() {
                for &next in adj_list.get(&current).into_iter().flatten() {
                    let next_color = coloring.get(&next).copied();
                    if (next_color == Some(a) || next_color == Some(b)) && chain.insert(next) {
                        stack.push(next);
                    }
                }
            }

            // Swapping is only useful if no b-colored neighbor would turn into a
            let blocked = neighbors
                .iter()
                .any(|nb| coloring.get(nb) == Some(&b) && chain.contains(nb));
            if blocked {
                continue;
            }

            for member in chain {
                let c = coloring
                    .get_mut(&member)
                    .expect("chain members are colored");
                *c = if *c == a { b } else { a };
            }
            return Some(a);
        }
    }

    None
}

/// DSATUR (Degree of Saturation) algorithm
/// Prioritizes nodes with the most differently-colored neighbors
fn dsatur_coloring(
    node_ids: &[u64],
    adj_list: &FxHashMap<u64, FxHashSet<u64>>,
) -> PyResult<FxHashMap<u64, usize>> {
    let n = node_ids.len();
    let mut coloring: FxHashMap<u64, usize> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
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
pub fn chromatic_number_upper_bound(
    coloring: std::collections::HashMap<u64, i64>,
) -> PyResult<usize> {
    let colors: FxHashSet<i64> = coloring.values().copied().collect();
    Ok(colors.len())
}

/// Color the edges with at most Δ + 1 colors so that edges sharing an endpoint differ
/// (Misra and Gries, 1992, a constructive proof of Vizing's theorem).
///
/// Edges are colored in ID order, direction ignored. Each edge (x, f) takes a color free at
/// both ends if one exists; otherwise a maximal fan of x starting at f is built, a c/d
/// alternating path from x is inverted (c free at x, d free at the fan's last node), and a
/// prefix of the fan is rotated so the edge can take d. O(m · n) time.
///
/// Errors on self-loops and on parallel edges (including a directed pair u → v, v → u),
/// for which Δ + 1 colors may not suffice.
///
/// Returns (edge ID, color) pairs in edge ID order, colors 0..=Δ.
#[pyfunction]
pub fn edge_coloring(graph: &Graph) -> PyResult<Vec<(u64, usize)>> {
    const NONE: usize = usize::MAX;
    let mut raw = graph.edges();
    raw.sort_unstable();
    let mut node_ids = graph.get_node_ids();
    node_ids.sort_unstable();
    let index: FxHashMap<u64, usize> = node_ids.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    let mut ends: Vec<(usize, usize)> = Vec::with_capacity(raw.len());
    let mut pairs: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut degree = vec![0usize; node_ids.len()];
    for &(_, s, t) in &raw {
        let (a, b) = (index[&s], index[&t]);
        if a == b {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "edge_coloring() needs a graph without self-loops",
            ));
        }
        if !pairs.insert((a.min(b), a.max(b))) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "edge_coloring() needs a graph without parallel edges (on a directed graph, \
                 without both u -> v and v -> u)",
            ));
        }
        degree[a] += 1;
        degree[b] += 1;
        ends.push((a, b));
    }
    let k = match degree.iter().max() {
        Some(&d) if d > 0 => d + 1,
        _ => return Ok(Vec::new()),
    };

    // at[v * k + c]: the edge of color c at node v, or NONE when c is free at v
    let mut at = vec![NONE; node_ids.len() * k];
    let mut color = vec![NONE; ends.len()];
    let mut mark = vec![NONE; node_ids.len()];
    let other = |e: usize, v: usize, ends: &[(usize, usize)]| {
        if ends[e].0 == v {
            ends[e].1
        } else {
            ends[e].0
        }
    };

    fn paint(
        e: usize,
        c: usize,
        ends: &[(usize, usize)],
        at: &mut [usize],
        color: &mut [usize],
        k: usize,
    ) {
        let (a, b) = ends[e];
        at[a * k + c] = e;
        at[b * k + c] = e;
        color[e] = c;
    }
    fn unpaint(e: usize, ends: &[(usize, usize)], at: &mut [usize], color: &mut [usize], k: usize) {
        let (a, b) = ends[e];
        let c = color[e];
        at[a * k + c] = usize::MAX;
        at[b * k + c] = usize::MAX;
        color[e] = usize::MAX;
    }

    for e in 0..ends.len() {
        let (x, f) = ends[e];
        if let Some(c) = (0..k).find(|&c| at[x * k + c] == NONE && at[f * k + c] == NONE) {
            paint(e, c, &ends, &mut at, &mut color, k);
            continue;
        }

        // Maximal fan of x from f: each next edge (x, y) has a color free at the previous node
        let mut fan_v = vec![f];
        let mut fan_e = vec![e];
        mark[f] = e;
        loop {
            let last = *fan_v.last().unwrap();
            let next = (0..k).find_map(|c| {
                let ex = at[x * k + c];
                if at[last * k + c] != NONE || ex == NONE {
                    return None;
                }
                let y = other(ex, x, &ends);
                (mark[y] != e).then_some((y, ex))
            });
            match next {
                Some((y, ex)) => {
                    mark[y] = e;
                    fan_v.push(y);
                    fan_e.push(ex);
                }
                None => break,
            }
        }

        let last = *fan_v.last().unwrap();
        let c = (0..k).find(|&c| at[x * k + c] == NONE).unwrap();
        let d = (0..k).find(|&c| at[last * k + c] == NONE).unwrap();

        // Invert the c/d path from x, so that d becomes free at x
        if c != d {
            let mut path = Vec::new();
            let (mut cur, mut want) = (x, d);
            while at[cur * k + want] != NONE {
                let ed = at[cur * k + want];
                path.push(ed);
                cur = other(ed, cur, &ends);
                want = if want == d { c } else { d };
            }
            let old: Vec<usize> = path.iter().map(|&ed| color[ed]).collect();
            for &ed in &path {
                unpaint(ed, &ends, &mut at, &mut color, k);
            }
            for (&ed, &was) in path.iter().zip(&old) {
                paint(
                    ed,
                    if was == c { d } else { c },
                    &ends,
                    &mut at,
                    &mut color,
                    k,
                );
            }
        }

        // The first node w with d free on a prefix of the fan that is still a fan
        let mut w = None;
        for i in 0..fan_v.len() {
            if i > 0 && at[fan_v[i - 1] * k + color[fan_e[i]]] != NONE {
                break;
            }
            if at[fan_v[i] * k + d] == NONE {
                w = Some(i);
                break;
            }
        }
        let w = w.expect("Misra-Gries: the inverted path always leaves a usable fan prefix");

        // Rotate the fan up to w, then color (x, w) with d
        for i in 0..w {
            let moved = color[fan_e[i + 1]];
            unpaint(fan_e[i + 1], &ends, &mut at, &mut color, k);
            paint(fan_e[i], moved, &ends, &mut at, &mut color, k);
        }
        paint(fan_e[w], d, &ends, &mut at, &mut color, k);
    }

    Ok(raw
        .iter()
        .zip(&color)
        .map(|(&(id, _, _), &c)| (id, c))
        .collect())
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
        let result = greedy_color(&graph, None, None, None);
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
