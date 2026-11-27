//! Greedy coloring, maximal independent sets and dominating sets with NetworkX's exact
//! results.
//!
//! Several of NetworkX's strategies pick nodes in the iteration or `pop()` order of Python
//! sets (smallest-last's degree buckets, the independent-set strategy, the components of
//! connected-sequential, `dominating_set`, `maximal_independent_set`); those sets are
//! modeled with `PySetModel` (CPython's set tables, from the node keys' hashes), and random
//! choices are drawn with CPython's Mersenne Twister (`PyRandom`), so the colorings and sets
//! are NetworkX's. Graphs are undirected without parallel edges or self-loops (the Python
//! side leaves other graphs to NetworkX). Nodes are 0-based positions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, VecDeque};

use crate::algorithms::nx_cliques::PySetModel;
use crate::algorithms::nx_compat::PyRandom;
use crate::algorithms::path_search::PathAdjacency;

const NONE: usize = usize::MAX;

/// Each node's neighbors
fn neighbor_lists(adjacency: &PathAdjacency) -> Vec<Vec<usize>> {
    let (indptr, indices, _) = adjacency.rows();
    (0..adjacency.len())
        .map(|u| indices[indptr[u]..indptr[u + 1]].to_vec())
        .collect()
}

/// A set model holding `members` (added in that order)
fn model_of(members: impl IntoIterator<Item = usize>, hashes: &[i64]) -> PySetModel {
    let mut s = PySetModel::new();
    for v in members {
        s.add(v, hashes[v]);
    }
    s
}

/// `strategy_smallest_last`: on NetworkX's `G.copy()` (an undirected copy lists each node's
/// earlier neighbors first), repeatedly take a node of least remaining degree from its
/// degree bucket (`set.pop()`), and return them last to first
fn smallest_last(nbrs: &[Vec<usize>], hashes: &[i64]) -> Vec<usize> {
    let n = nbrs.len();
    let copy: Vec<Vec<usize>> = (0..n)
        .map(|u| {
            let mut earlier: Vec<usize> = nbrs[u].iter().copied().filter(|&v| v < u).collect();
            earlier.sort_unstable();
            earlier.extend(nbrs[u].iter().copied().filter(|&v| v >= u));
            earlier
        })
        .collect();
    let mut removed = vec![false; n];
    let mut degree: Vec<i64> = copy.iter().map(|l| l.len() as i64).collect();
    let mut buckets: BTreeMap<i64, PySetModel> = BTreeMap::new();
    let mut lbound = i64::MAX;
    for u in 0..n {
        buckets
            .entry(degree[u])
            .or_insert_with(PySetModel::new)
            .add(u, hashes[u]);
        lbound = lbound.min(degree[u]);
    }
    let mut result = VecDeque::with_capacity(n);
    for _ in 0..n {
        let min_degree = *buckets.range(lbound..).next().expect("a bucket").0;
        let bucket = buckets.get_mut(&min_degree).expect("present");
        let u = bucket.pop().expect("not empty");
        if bucket.len() == 0 {
            buckets.remove(&min_degree);
        }
        result.push_front(u);
        for &v in &copy[u] {
            if removed[v] {
                continue;
            }
            let d = degree[v];
            let bucket = buckets.get_mut(&d).expect("present");
            bucket.discard(v, hashes[v]);
            if bucket.len() == 0 {
                buckets.remove(&d);
            }
            buckets
                .entry(d - 1)
                .or_insert_with(PySetModel::new)
                .add(v, hashes[v]);
            degree[v] = d - 1;
        }
        removed[u] = true;
        lbound = min_degree - 1;
    }
    result.into_iter().collect()
}

/// `strategy_independent_set`: repeatedly a greedy maximal independent set of the remaining
/// nodes (least remaining degree first, in set order), in set order
fn independent_sets(nbrs: &[Vec<usize>], hashes: &[i64]) -> Vec<usize> {
    let n = nbrs.len();
    let mut order = Vec::with_capacity(n);
    let mut remaining_nodes = model_of(0..n, hashes);
    let mut in_remaining_nodes = vec![true; n];
    while remaining_nodes.len() > 0 {
        // _maximal_independent_set(G.subgraph(remaining_nodes)): the view's node set is
        // built from remaining_nodes' iteration order, and `set(view)` iterates that set
        // when it is under half the graph (NetworkX's FilterAtlas), else the graph's nodes
        let mut result = PySetModel::new();
        let shown = model_of(remaining_nodes.iter().map(|(k, _)| k), hashes);
        let mut remaining = if 2 * shown.len() < n {
            model_of(shown.iter().map(|(k, _)| k), hashes)
        } else {
            model_of((0..n).filter(|&v| in_remaining_nodes[v]), hashes)
        };
        let mut in_remaining: Vec<bool> = in_remaining_nodes.clone();
        while remaining.len() > 0 {
            let mut best = NONE;
            let mut best_degree = usize::MAX;
            for (u, _) in remaining.iter() {
                let d = nbrs[u].iter().filter(|&&w| in_remaining[w]).count();
                if d < best_degree {
                    best = u;
                    best_degree = d;
                }
            }
            let v = best;
            result.add(v, hashes[v]);
            // remaining -= set(G[v]) | {v}
            let gone = model_of(
                nbrs[v]
                    .iter()
                    .copied()
                    .filter(|&w| in_remaining[w])
                    .chain([v]),
                hashes,
            );
            for (w, _) in gone.iter() {
                in_remaining[w] = false;
            }
            remaining.difference_update(&gone);
        }
        remaining_nodes.difference_update(&result);
        for (v, _) in result.iter() {
            in_remaining_nodes[v] = false;
            order.push(v);
        }
    }
    order
}

/// `strategy_connected_sequential`: per connected component (NetworkX's `_plain_bfs` set,
/// whose first member in set order is the start), the start and then the nodes in BFS or
/// DFS order from it
fn connected_sequential(nbrs: &[Vec<usize>], hashes: &[i64], dfs: bool) -> Vec<usize> {
    let n = nbrs.len();
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for v in 0..n {
        if seen[v] {
            continue;
        }
        // _plain_bfs: {v} plus the nodes in breadth-first order
        let mut component = PySetModel::new();
        component.add(v, hashes[v]);
        seen[v] = true;
        let mut level = vec![v];
        while !level.is_empty() {
            let mut next = Vec::new();
            for &u in &level {
                for &w in &nbrs[u] {
                    if !seen[w] {
                        seen[w] = true;
                        component.add(w, hashes[w]);
                        next.push(w);
                    }
                }
            }
            level = next;
        }
        let source = component.iter().next().expect("not empty").0;
        order.push(source);
        let mut visited = vec![false; n];
        visited[source] = true;
        if dfs {
            // nx.dfs_edges(G, source)
            let mut stack = vec![(source, 0usize)];
            while let Some(&(u, k)) = stack.last() {
                if k < nbrs[u].len() {
                    stack.last_mut().expect("not empty").1 = k + 1;
                    let w = nbrs[u][k];
                    if !visited[w] {
                        visited[w] = true;
                        order.push(w);
                        stack.push((w, 0));
                    }
                } else {
                    stack.pop();
                }
            }
        } else {
            // nx.bfs_edges(G, source)
            let mut queue = VecDeque::from([source]);
            while let Some(u) = queue.pop_front() {
                for &w in &nbrs[u] {
                    if !visited[w] {
                        visited[w] = true;
                        order.push(w);
                        queue.push_back(w);
                    }
                }
            }
        }
    }
    order
}

/// The smallest color no colored neighbor has
fn first_free_color(nbrs: &[usize], colors: &[usize]) -> usize {
    let mut used: Vec<usize> = nbrs
        .iter()
        .map(|&w| colors[w])
        .filter(|&c| c != NONE)
        .collect();
    used.sort_unstable();
    used.dedup();
    let mut color = 0;
    for c in used {
        if c == color {
            color += 1;
        } else if c > color {
            break;
        }
    }
    color
}

/// NetworkX's `greedy_color` (without interchange) for `strategy` ("largest_first",
/// "smallest_last", "independent_set", "connected_sequential_bfs",
/// "connected_sequential_dfs", "saturation_largest_first", or "given" with `order`).
/// Returns the nodes in the order colored and their colors.
#[pyfunction]
#[pyo3(signature = (adjacency, hashes, strategy, order=None))]
pub fn nx_greedy_color(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    hashes: Vec<i64>,
    strategy: &str,
    order: Option<Vec<usize>>,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let n = adjacency.len();
    if hashes.len() != n {
        return Err(PyValueError::new_err("hashes must have one entry per node"));
    }
    let nbrs = neighbor_lists(adjacency);
    let degree: Vec<usize> = nbrs.iter().map(|l| l.len()).collect();
    let strategy = strategy.to_string();
    py.allow_threads(move || {
        let mut colors = vec![NONE; n];
        let mut colored = Vec::with_capacity(n);
        if strategy == "saturation_largest_first" {
            if n == 0 {
                return Ok((colored, Vec::new()));
            }
            let mut saturation: Vec<std::collections::BTreeSet<usize>> =
                vec![Default::default(); n];
            let mut node = 0;
            for v in 1..n {
                if degree[v] > degree[node] {
                    node = v;
                }
            }
            loop {
                let color = first_free_color(&nbrs[node], &colors);
                colors[node] = color;
                colored.push(node);
                for &w in &nbrs[node] {
                    saturation[w].insert(color);
                }
                if colored.len() == n {
                    break;
                }
                let mut best = NONE;
                for v in 0..n {
                    if colors[v] != NONE {
                        continue;
                    }
                    if best == NONE
                        || (saturation[v].len(), degree[v]) > (saturation[best].len(), degree[best])
                    {
                        best = v;
                    }
                }
                node = best;
            }
            let result = colored.iter().map(|&v| colors[v]).collect();
            return Ok((colored, result));
        }
        let sequence: Vec<usize> = match strategy.as_str() {
            "largest_first" => {
                let mut nodes: Vec<usize> = (0..n).collect();
                // (sorted(..., reverse=True) keeps equal degrees in node order)
                nodes.sort_by(|&a, &b| degree[b].cmp(&degree[a]));
                nodes
            }
            "smallest_last" => smallest_last(&nbrs, &hashes),
            "independent_set" => independent_sets(&nbrs, &hashes),
            "connected_sequential_bfs" => connected_sequential(&nbrs, &hashes, false),
            "connected_sequential_dfs" => connected_sequential(&nbrs, &hashes, true),
            "given" => match order {
                Some(order) if order.iter().all(|&v| v < n) => order,
                _ => {
                    return Err(PyValueError::new_err(
                        "strategy 'given' needs node positions",
                    ))
                }
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown strategy {:?}",
                    other
                )));
            }
        };
        for u in sequence {
            let color = first_free_color(&nbrs[u], &colors);
            colors[u] = color;
            colored.push(u);
        }
        let result = colored.iter().map(|&v| colors[v]).collect();
        Ok((colored, result))
    })
}

/// NetworkX's `maximal_independent_set` after its checks: from the independent `start`
/// nodes (positions, in the iteration order of NetworkX's set of them), repeatedly
/// `seed.choice(list(available_nodes))`, with the generator state `(words, index)`. Returns
/// the chosen nodes in order and the generator's new state.
#[pyfunction]
pub fn nx_maximal_independent_set(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    hashes: Vec<i64>,
    start: Vec<usize>,
    words: Vec<u32>,
    index: usize,
) -> PyResult<(Vec<usize>, Vec<u32>, usize)> {
    let n = adjacency.len();
    if hashes.len() != n || start.iter().any(|&v| v >= n) {
        return Err(PyValueError::new_err(
            "one hash per node; start nodes are positions",
        ));
    }
    let mut rng = PyRandom::from_state(&words, index)?;
    let nbrs = neighbor_lists(adjacency);
    let chosen = py.allow_threads(|| {
        // available_nodes = set(G.nodes()).difference(neighbors.union(nodes))
        let mut excluded = vec![false; n];
        for &v in &start {
            excluded[v] = true;
            for &w in &nbrs[v] {
                excluded[w] = true;
            }
        }
        let all = model_of(0..n, &hashes);
        let excluded_set = model_of((0..n).filter(|&v| excluded[v]), &hashes);
        let mut available = PySetModel::difference(&all, &excluded_set);
        let mut chosen = Vec::new();
        while available.len() > 0 {
            let members: Vec<usize> = available.iter().map(|(k, _)| k).collect();
            let node = members[rng.randbelow(members.len())];
            chosen.push(node);
            // available_nodes.difference_update(list(G.adj[node]) + [node])
            let gone = model_of(nbrs[node].iter().copied().chain([node]), &hashes);
            available.difference_update_iterable(gone.iter());
        }
        chosen
    });
    let (words, index) = rng.state();
    Ok((chosen, words, index))
}

/// NetworkX's `dominating_set(G, start_with)` (the start a position): the nodes in the order
/// NetworkX adds them to the dominating set (the start, then the nodes it pops from its set
/// of remaining nodes)
#[pyfunction]
pub fn nx_dominating_set(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    hashes: Vec<i64>,
    start: usize,
) -> PyResult<Vec<usize>> {
    let n = adjacency.len();
    if hashes.len() != n || start >= n {
        return Err(PyValueError::new_err(
            "one hash per node; start is a position",
        ));
    }
    let nbrs = neighbor_lists(adjacency);
    Ok(py.allow_threads(|| {
        let all_nodes = model_of(0..n, &hashes);
        let mut in_dominating = vec![false; n];
        in_dominating[start] = true;
        let mut added = vec![start];
        let dominated = model_of(nbrs[start].iter().copied(), &hashes);
        let start_set = model_of([start], &hashes);
        let mut remaining = PySetModel::difference(&all_nodes, &dominated);
        remaining = PySetModel::difference(&remaining, &start_set);
        while let Some(v) = remaining.pop() {
            // undominated_nbrs = set(G[v]) - dominating_set
            let undominated = model_of(
                nbrs[v].iter().copied().filter(|&w| !in_dominating[w]),
                &hashes,
            );
            in_dominating[v] = true;
            added.push(v);
            remaining.difference_update(&undominated);
        }
        added
    }))
}
