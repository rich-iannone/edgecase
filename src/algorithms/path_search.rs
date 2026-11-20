//! Shortest-path searches that follow a given neighbor order.
//!
//! `PathAdjacency` is a compressed sparse row (CSR) adjacency built from parallel arrays of
//! arcs `(source, target, weight)`. Its searches visit each node's neighbors in the order the
//! arcs were given and break ties the way NetworkX's pure-Python implementations do: BFS level
//! by level in discovery order, and Dijkstra with a heap of `(distance, insertion counter,
//! node)`. A converter that passes NetworkX's adjacency order (the NetworkX backend does) gets
//! NetworkX's exact answers (the same paths, predecessor lists and result order), not merely
//! equally short ones.
//!
//! Nodes are given as IDs `1..=n` (edgecase's convention; ID 0 is reserved); results are
//! 0-based node positions (`ID - 1`).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::OnceLock;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList};
use rayon::prelude::*;

/// "No node" (an unset parent)
const NONE: usize = usize::MAX;
/// The root of a search tree (a node that has no parent but is reached)
const ROOT: usize = usize::MAX - 1;

/// Searches from more sources than this per call may run in parallel, when the graph is big
/// enough to be worth it (see `PathAdjacency::parallel`)
const PARALLEL_WORK: usize = 200_000;

/// Adjacency rows: the neighbors of node `v` are `indices[indptr[v]..indptr[v + 1]]`, with the
/// cost of each step in the same range of `weights`.
struct Csr {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights: Vec<f64>,
}

impl Csr {
    #[inline]
    fn targets(&self, v: usize) -> &[usize] {
        &self.indices[self.indptr[v]..self.indptr[v + 1]]
    }

    #[inline]
    fn arcs(&self, v: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.indptr[v]..self.indptr[v + 1];
        self.indices[range.clone()]
            .iter()
            .copied()
            .zip(self.weights[range].iter().copied())
    }

    /// The reverse adjacency: `u`'s row lists `v` for each arc `v -> u`, in order of `v` and,
    /// for one `v`, in the order of `v`'s row.
    fn reversed(&self) -> Csr {
        let n = self.indptr.len() - 1;
        let m = self.indices.len();
        let mut indptr = vec![0usize; n + 1];
        for &u in &self.indices {
            indptr[u + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let mut next = indptr.clone();
        let mut indices = vec![0usize; m];
        let mut weights = vec![0.0f64; m];
        for v in 0..n {
            for k in self.indptr[v]..self.indptr[v + 1] {
                let u = self.indices[k];
                let p = next[u];
                next[u] += 1;
                indices[p] = v;
                weights[p] = self.weights[k];
            }
        }
        Csr {
            indptr,
            indices,
            weights,
        }
    }
}

/// What a Dijkstra search records besides distances
#[derive(Clone, Copy, PartialEq, Eq)]
enum Track {
    /// Nothing
    Lengths,
    /// The node that last lowered each node's tentative distance (NetworkX's `pred[v][0]`)
    Parent,
    /// Every predecessor on a shortest path (NetworkX's `pred` lists)
    AllPreds,
}

/// A heap entry, ordered so that `BinaryHeap` pops the smallest `(dist, count)` first, the
/// order Python's `heapq` gives `(dist, count, node)` tuples
#[derive(Clone, Copy)]
struct Entry {
    dist: f64,
    count: u64,
    node: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.dist.partial_cmp(&self.dist) {
            Some(Ordering::Equal) | None => other.count.cmp(&self.count),
            Some(ordering) => ordering,
        }
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A negative edge made a finished node's distance shorter (Dijkstra cannot handle it)
struct Contradiction;

/// The result of a Dijkstra search
struct Dijkstra {
    /// Finished nodes, in the order they were finished (NetworkX's `dist` order)
    order: Vec<usize>,
    /// Distance of each node in `order`
    dist: Vec<f64>,
    /// Per node: the node that last lowered its tentative distance (`Track::Parent`)
    parent: Vec<usize>,
    /// Nodes in the order they were first reached (NetworkX's `pred` order; `AllPreds`)
    seen_order: Vec<usize>,
    /// Per node: its predecessors on shortest paths (`AllPreds`)
    preds: Vec<Vec<usize>>,
}

/// Neumaier-compensated running sum, the algorithm (and so the rounding) of Python's `sum()`
/// over floats
#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    compensation: f64,
}

impl CompensatedSum {
    #[inline]
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.compensation += (self.sum - t) + x;
        } else {
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    fn total(&self) -> f64 {
        if self.compensation != 0.0 && self.compensation.is_finite() {
            self.sum + self.compensation
        } else {
            self.sum
        }
    }
}

/// A graph's adjacency for shortest-path searches, in a given neighbor order.
///
/// Built from parallel arrays of arcs: `sources[i] -> targets[i]` with step cost
/// `weights[i]` (node IDs `1..=num_nodes`). An undirected graph lists each edge in both
/// directions (a self-loop once), the way an adjacency dict does. Each node's neighbors are
/// kept in the order of its arcs; consecutive arcs between the same two nodes (a multigraph's
/// parallel edges) become one step that costs the least of them. A weight of `None` hides the
/// arc, and without `weights` every step costs 1.
///
/// The search methods take and return 0-based node positions (ID - 1).
#[pyclass(frozen, module = "edgecase._core")]
pub struct PathAdjacency {
    n: usize,
    directed: bool,
    /// Every weight is a Python int (or bool), so distances are integers
    integral: bool,
    /// Some arc was hidden (a `None` weight)
    hidden: bool,
    out: Csr,
    /// The reverse adjacency (predecessors), built on first use for directed graphs
    rev: OnceLock<Csr>,
}

impl PathAdjacency {
    fn forward(&self) -> &Csr {
        &self.out
    }

    /// The adjacency to search: predecessors (`reverse`) or successors
    fn csr(&self, reverse: bool) -> &Csr {
        if reverse && self.directed {
            self.rev.get_or_init(|| self.out.reversed())
        } else {
            &self.out
        }
    }

    fn check_node(&self, v: usize) -> PyResult<usize> {
        if v < self.n {
            Ok(v)
        } else {
            Err(PyIndexError::new_err(format!(
                "node position {} out of range for {} nodes",
                v, self.n
            )))
        }
    }

    /// Whether a call with `searches` searches should spread them over threads
    fn parallel(&self, searches: usize) -> bool {
        searches > 1 && searches * (self.n + self.out.indices.len()) > PARALLEL_WORK
    }

    /// Breadth-first search from `source`, expanding nodes whose level is below `cutoff`.
    /// Returns the reached nodes in discovery order, their levels and (with `parents`) the
    /// position in that order of the node that discovered each (the source: 0).
    fn bfs(
        &self,
        csr: &Csr,
        source: usize,
        cutoff: f64,
        parents: bool,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut seen = vec![false; self.n];
        seen[source] = true;
        let mut order = vec![source];
        let mut levels = vec![0usize];
        let mut parent_positions = Vec::new();
        if parents {
            parent_positions.push(0);
        }
        let mut head = 0;
        // (once every node is reached there is nothing left to discover)
        while head < order.len() && order.len() < self.n {
            let v = order[head];
            let level = levels[head];
            // (as Python's `cutoff > level`, false for a NaN cutoff)
            if cutoff.partial_cmp(&(level as f64)) != Some(Ordering::Greater) {
                break;
            }
            for &w in csr.targets(v) {
                if !seen[w] {
                    seen[w] = true;
                    order.push(w);
                    levels.push(level + 1);
                    if parents {
                        parent_positions.push(head);
                    }
                }
            }
            head += 1;
        }
        (order, levels, parent_positions)
    }

    /// Dijkstra's algorithm from `sources`, as NetworkX's `_dijkstra_multisource`.
    fn run_dijkstra(
        &self,
        csr: &Csr,
        sources: &[usize],
        cutoff: Option<f64>,
        target: Option<usize>,
        track: Track,
    ) -> Result<Dijkstra, Contradiction> {
        let n = self.n;
        let mut finished = vec![false; n];
        let mut dist_of = vec![0.0f64; n];
        let mut seen = vec![false; n];
        let mut seen_dist = vec![f64::INFINITY; n];
        let mut parent = if track == Track::Lengths {
            Vec::new()
        } else {
            vec![NONE; n]
        };
        let mut preds: Vec<Vec<usize>> = if track == Track::AllPreds {
            vec![Vec::new(); n]
        } else {
            Vec::new()
        };
        let mut seen_order = Vec::new();
        let mut order = Vec::new();
        let mut dist = Vec::new();
        let mut heap = BinaryHeap::new();
        let mut count = 0u64;
        for &s in sources {
            if !seen[s] && track == Track::AllPreds {
                seen_order.push(s);
            }
            seen[s] = true;
            seen_dist[s] = 0.0;
            heap.push(Entry {
                dist: 0.0,
                count,
                node: s,
            });
            count += 1;
        }
        while let Some(Entry {
            dist: d, node: v, ..
        }) = heap.pop()
        {
            if finished[v] {
                continue;
            }
            finished[v] = true;
            dist_of[v] = d;
            order.push(v);
            dist.push(d);
            if target == Some(v) {
                break;
            }
            for (u, cost) in csr.arcs(v) {
                let vu = d + cost;
                if cutoff.is_some_and(|c| vu > c) {
                    continue;
                }
                if finished[u] {
                    let ud = dist_of[u];
                    if vu < ud {
                        return Err(Contradiction);
                    }
                    if track == Track::AllPreds && vu == ud {
                        preds[u].push(v);
                    }
                } else if !seen[u] || vu < seen_dist[u] {
                    if !seen[u] && track == Track::AllPreds {
                        seen_order.push(u);
                    }
                    seen[u] = true;
                    seen_dist[u] = vu;
                    heap.push(Entry {
                        dist: vu,
                        count,
                        node: u,
                    });
                    count += 1;
                    match track {
                        Track::Lengths => {}
                        Track::Parent => parent[u] = v,
                        Track::AllPreds => {
                            parent[u] = v;
                            preds[u].clear();
                            preds[u].push(v);
                        }
                    }
                } else if track == Track::AllPreds && vu == seen_dist[u] {
                    preds[u].push(v);
                }
            }
        }
        Ok(Dijkstra {
            order,
            dist,
            parent,
            seen_order,
            preds,
        })
    }

    /// For each node in `order`, the position in `order` of its parent (sources: their own)
    fn parent_positions(&self, order: &[usize], parent: &[usize]) -> Vec<usize> {
        let mut position = vec![NONE; self.n];
        for (i, &v) in order.iter().enumerate() {
            position[v] = i;
        }
        order
            .iter()
            .enumerate()
            .map(|(i, &v)| match parent[v] {
                NONE => i,
                p if position[p] == NONE => i,
                p => position[p],
            })
            .collect()
    }

    /// Distances as Python numbers: ints if every weight is an int
    fn numbers(&self, py: Python<'_>, values: &[f64]) -> PyResult<PyObject> {
        if self.integral && values.iter().all(|x| x.is_finite()) {
            let ints: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            Ok(ints.into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(values.into_pyobject(py)?.into_any().unbind())
        }
    }

    fn number(&self, py: Python<'_>, value: f64) -> PyResult<PyObject> {
        if self.integral && value.is_finite() {
            Ok((value as i64).into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(value.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// `{nodes[order[i]]: levels[i]}`
    fn level_dict<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        order: &[usize],
        levels: &[usize],
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(nodes.py());
        for (&v, &level) in order.iter().zip(levels) {
            dict.set_item(nodes.get_item(v)?, level)?;
        }
        Ok(dict)
    }

    /// `{nodes[order[i]]: dist[i]}`: ints if every weight is an int, else floats except for
    /// the first `roots` nodes (the sources), whose distance is the int 0 as in NetworkX
    fn distance_dict<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        order: &[usize],
        dist: &[f64],
        roots: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(nodes.py());
        let integral = self.integral && dist.iter().all(|x| x.is_finite());
        for (i, (&v, &d)) in order.iter().zip(dist).enumerate() {
            let key = nodes.get_item(v)?;
            if integral {
                dict.set_item(key, d as i64)?;
            } else if i < roots {
                dict.set_item(key, 0)?;
            } else {
                dict.set_item(key, d)?;
            }
        }
        Ok(dict)
    }

    /// The distances from `source` in search order: BFS levels (`weighted=false`) or
    /// Dijkstra distances, over predecessors with `reverse`
    fn ordered_distances(
        &self,
        source: usize,
        weighted: bool,
        reverse: bool,
    ) -> Result<(Vec<usize>, Vec<f64>), Contradiction> {
        let csr = self.csr(reverse);
        if weighted {
            let r = self.run_dijkstra(csr, &[source], None, None, Track::Lengths)?;
            Ok((r.order, r.dist))
        } else {
            let (order, levels, _) = self.bfs(csr, source, f64::INFINITY, false);
            Ok((order, levels.into_iter().map(|l| l as f64).collect()))
        }
    }

    /// Search from each of `sources` (in parallel batches when big) and hand each search's
    /// `(position in sources, order, distances)` to `visit` in the order of `sources`.
    /// Stops at the first search that hits a contradiction and returns its position.
    fn visit_searches<F>(
        &self,
        sources: &[usize],
        weighted: bool,
        reverse: bool,
        mut visit: F,
    ) -> Option<usize>
    where
        F: FnMut(usize, &[usize], &[f64]),
    {
        let chunk = (4_000_000 / self.n.max(1)).clamp(1, 1024);
        let run = |&s: &usize| self.ordered_distances(s, weighted, reverse);
        let mut start = 0;
        while start < sources.len() {
            let end = (start + chunk).min(sources.len());
            let batch = &sources[start..end];
            let results: Vec<Result<(Vec<usize>, Vec<f64>), Contradiction>> =
                if self.parallel(batch.len()) {
                    batch.par_iter().map(run).collect()
                } else {
                    batch.iter().map(run).collect()
                };
            for (i, result) in results.into_iter().enumerate() {
                match result {
                    Ok((order, dist)) => visit(start + i, &order, &dist),
                    Err(Contradiction) => return Some(start + i),
                }
            }
            start = end;
        }
        None
    }

    fn sources(&self, sources: Vec<usize>) -> PyResult<Vec<usize>> {
        for &s in &sources {
            self.check_node(s)?;
        }
        Ok(sources)
    }
}

fn contradiction_error() -> PyErr {
    // NetworkX raises `ValueError("Contradictory paths found:", "negative weights?")`
    PyValueError::new_err(("Contradictory paths found:", "negative weights?"))
}

#[pymethods]
impl PathAdjacency {
    #[new]
    #[pyo3(signature = (num_nodes, sources, targets, weights=None, directed=false))]
    fn new(
        num_nodes: usize,
        sources: Vec<u64>,
        targets: Vec<u64>,
        weights: Option<&Bound<'_, PyAny>>,
        directed: bool,
    ) -> PyResult<Self> {
        let m = sources.len();
        if targets.len() != m {
            return Err(PyValueError::new_err(
                "sources and targets must have the same length",
            ));
        }
        let mut integral = true;
        let mut arc_weights: Vec<Option<f64>> = Vec::new();
        if let Some(weights) = weights {
            arc_weights.reserve(m);
            for item in weights.try_iter()? {
                let item = item?;
                if item.is_none() {
                    arc_weights.push(None);
                    continue;
                }
                if !item.is_instance_of::<PyInt>() {
                    integral = false;
                }
                arc_weights.push(Some(item.extract::<f64>()?));
            }
            if arc_weights.len() != m {
                return Err(PyValueError::new_err(
                    "weights must have the same length as sources",
                ));
            }
        }
        let n = num_nodes;
        let mut counts = vec![0usize; n + 1];
        for (&s, &t) in sources.iter().zip(&targets) {
            if s == 0 || s as usize > n || t == 0 || t as usize > n {
                return Err(PyValueError::new_err(format!(
                    "Arc ({}, {}) refers to a node outside 1..={}",
                    s, t, n
                )));
            }
            counts[s as usize] += 1;
        }
        for i in 0..n {
            counts[i + 1] += counts[i];
        }
        // Stable counting sort of the arcs by source
        let mut next = counts.clone();
        let mut sorted = vec![0usize; m];
        for (i, &s) in sources.iter().enumerate() {
            let s = s as usize - 1;
            sorted[next[s]] = i;
            next[s] += 1;
        }
        // Rows, merging runs of arcs to the same neighbor and dropping hidden arcs
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(m);
        let mut costs = Vec::with_capacity(m);
        let mut hidden = false;
        indptr.push(0);
        for v in 0..n {
            let row_start = indices.len();
            let mut previous = NONE;
            for &i in &sorted[counts[v]..counts[v + 1]] {
                let t = targets[i] as usize - 1;
                let weight = if weights.is_some() {
                    arc_weights[i]
                } else {
                    Some(1.0)
                };
                let same = t == previous;
                previous = t;
                match weight {
                    None => hidden = true,
                    Some(w) => {
                        let last = indices.len();
                        if same && last > row_start && indices[last - 1] == t {
                            let c: &mut f64 = &mut costs[last - 1];
                            if w < *c {
                                *c = w;
                            }
                        } else {
                            indices.push(t);
                            costs.push(w);
                        }
                    }
                }
            }
            indptr.push(indices.len());
        }
        Ok(PathAdjacency {
            n,
            directed,
            integral,
            hidden,
            out: Csr {
                indptr,
                indices,
                weights: costs,
            },
            rev: OnceLock::new(),
        })
    }

    /// The number of nodes
    #[getter]
    fn num_nodes(&self) -> usize {
        self.n
    }

    /// Whether every weight is a Python int (so distances are ints)
    #[getter]
    fn integral(&self) -> bool {
        self.integral
    }

    /// Whether some arc was hidden by a `None` weight
    #[getter]
    fn hidden(&self) -> bool {
        self.hidden
    }

    /// Breadth-first search from `source` (NetworkX's `single_source_shortest_path_length`
    /// and `single_source_shortest_path` order). Expands nodes whose level is below `cutoff`
    /// and follows predecessors instead of successors with `reverse`.
    ///
    /// Returns `(order, levels, parents)`: the reached nodes in discovery order, their levels,
    /// and (with `parents=True`, else an empty list) the position in `order` of the node that
    /// discovered each.
    #[pyo3(signature = (source, cutoff=f64::INFINITY, reverse=false, parents=false))]
    fn bfs_levels(
        &self,
        py: Python<'_>,
        source: usize,
        cutoff: f64,
        reverse: bool,
        parents: bool,
    ) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        self.check_node(source)?;
        let csr = self.csr(reverse);
        Ok(py.allow_threads(|| self.bfs(csr, source, cutoff, parents)))
    }

    /// `bfs_levels()` from each of `sources`, as a list of results (in parallel when big)
    #[pyo3(signature = (sources, cutoff=f64::INFINITY, reverse=false, parents=false))]
    fn bfs_levels_many(
        &self,
        py: Python<'_>,
        sources: Vec<usize>,
        cutoff: f64,
        reverse: bool,
        parents: bool,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>> {
        let sources = self.sources(sources)?;
        let csr = self.csr(reverse);
        Ok(py.allow_threads(|| {
            if self.parallel(sources.len()) {
                sources
                    .par_iter()
                    .map(|&s| self.bfs(csr, s, cutoff, parents))
                    .collect()
            } else {
                sources
                    .iter()
                    .map(|&s| self.bfs(csr, s, cutoff, parents))
                    .collect()
            }
        }))
    }

    /// `{node: level}` from `bfs_levels()`, keyed by `nodes[position]` (NetworkX's
    /// `single_source_shortest_path_length`)
    #[pyo3(signature = (nodes, source, cutoff=f64::INFINITY, reverse=false))]
    fn bfs_lengths<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        source: usize,
        cutoff: f64,
        reverse: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (order, levels, _) = self.bfs_levels(nodes.py(), source, cutoff, reverse, false)?;
        self.level_dict(nodes, &order, &levels)
    }

    /// `bfs_lengths()` from each of `sources` (searches in parallel when big)
    #[pyo3(signature = (nodes, sources, cutoff=f64::INFINITY, reverse=false))]
    fn bfs_lengths_many<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        sources: Vec<usize>,
        cutoff: f64,
        reverse: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let results = self.bfs_levels_many(nodes.py(), sources, cutoff, reverse, false)?;
        results
            .iter()
            .map(|(order, levels, _)| self.level_dict(nodes, order, levels))
            .collect()
    }

    /// `{node: distance}` from Dijkstra's algorithm from `sources`, keyed by
    /// `nodes[position]` (NetworkX's `multi_source_dijkstra_path_length`)
    #[pyo3(signature = (nodes, sources, cutoff=None, reverse=false))]
    fn dijkstra_lengths<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        sources: Vec<usize>,
        cutoff: Option<f64>,
        reverse: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sources = self.sources(sources)?;
        let csr = self.csr(reverse);
        let result = nodes
            .py()
            .allow_threads(|| self.run_dijkstra(csr, &sources, cutoff, None, Track::Lengths))
            .map_err(|_| contradiction_error())?;
        let mut distinct = sources.clone();
        distinct.sort_unstable();
        distinct.dedup();
        self.distance_dict(nodes, &result.order, &result.dist, distinct.len())
    }

    /// `dijkstra_lengths()` from each of `sources` separately (in parallel when big), cut
    /// short before the first source whose search raises; the second item is True then.
    #[pyo3(signature = (nodes, sources, cutoff=None))]
    fn dijkstra_lengths_many<'py>(
        &self,
        nodes: &Bound<'py, PyList>,
        sources: Vec<usize>,
        cutoff: Option<f64>,
    ) -> PyResult<(Vec<Bound<'py, PyDict>>, bool)> {
        let sources = self.sources(sources)?;
        let csr = self.forward();
        let run = |s: &usize| {
            self.run_dijkstra(csr, std::slice::from_ref(s), cutoff, None, Track::Lengths)
        };
        let results: Vec<Result<Dijkstra, Contradiction>> = nodes.py().allow_threads(|| {
            if self.parallel(sources.len()) {
                sources.par_iter().map(run).collect()
            } else {
                sources.iter().map(run).collect()
            }
        });
        let mut dicts = Vec::with_capacity(results.len());
        for result in results {
            match result {
                Ok(r) => dicts.push(self.distance_dict(nodes, &r.order, &r.dist, 1)?),
                Err(Contradiction) => return Ok((dicts, true)),
            }
        }
        Ok((dicts, false))
    }

    /// NetworkX's `predecessor`: breadth-first from `source`, recording every neighbor one
    /// level up. Stops after the level that reaches `cutoff` (None: no cutoff).
    ///
    /// Returns `(order, levels, preds)` with the reached nodes in discovery order.
    #[pyo3(signature = (source, cutoff=None))]
    fn bfs_predecessors(
        &self,
        source: usize,
        cutoff: Option<f64>,
    ) -> PyResult<(Vec<usize>, Vec<usize>, Vec<Vec<usize>>)> {
        self.check_node(source)?;
        let csr = self.forward();
        let mut level_of = vec![usize::MAX; self.n];
        let mut position = vec![NONE; self.n];
        level_of[source] = 0;
        position[source] = 0;
        let mut order = vec![source];
        let mut levels = vec![0usize];
        let mut preds: Vec<Vec<usize>> = vec![Vec::new()];
        let mut level = 0usize;
        let mut start = 0;
        while start < order.len() {
            level += 1;
            let end = order.len();
            for i in start..end {
                let v = order[i];
                for &w in csr.targets(v) {
                    if level_of[w] == usize::MAX {
                        level_of[w] = level;
                        position[w] = order.len();
                        order.push(w);
                        levels.push(level);
                        preds.push(vec![v]);
                    } else if level_of[w] == level {
                        preds[position[w]].push(v);
                    }
                }
            }
            start = end;
            if cutoff.is_some_and(|c| c <= level as f64) {
                break;
            }
        }
        Ok((order, levels, preds))
    }

    /// NetworkX's `bidirectional_shortest_path`: a path from `source` to `target` with the
    /// fewest edges, or None. Grows the smaller of the two search frontiers each round, along
    /// successors from `source` and predecessors from `target`.
    fn bidirectional_bfs(
        &self,
        py: Python<'_>,
        source: usize,
        target: usize,
    ) -> PyResult<Option<Vec<usize>>> {
        self.check_node(source)?;
        self.check_node(target)?;
        if source == target {
            return Ok(Some(vec![source]));
        }
        let forward_csr = self.forward();
        let reverse_csr = self.csr(true);
        let n = self.n;
        Ok(py.allow_threads(|| {
            let mut pred = vec![NONE; n];
            let mut succ = vec![NONE; n];
            pred[source] = ROOT;
            succ[target] = ROOT;
            let mut forward = vec![source];
            let mut reverse = vec![target];
            let mut meet = NONE;
            'search: while !forward.is_empty() && !reverse.is_empty() {
                if forward.len() <= reverse.len() {
                    let this_level = std::mem::take(&mut forward);
                    for v in this_level {
                        for &w in forward_csr.targets(v) {
                            if pred[w] == NONE {
                                forward.push(w);
                                pred[w] = v;
                            }
                            if succ[w] != NONE {
                                meet = w;
                                break 'search;
                            }
                        }
                    }
                } else {
                    let this_level = std::mem::take(&mut reverse);
                    for v in this_level {
                        for &w in reverse_csr.targets(v) {
                            if succ[w] == NONE {
                                succ[w] = v;
                                reverse.push(w);
                            }
                            if pred[w] != NONE {
                                meet = w;
                                break 'search;
                            }
                        }
                    }
                }
            }
            if meet == NONE {
                return None;
            }
            let mut path = Vec::new();
            let mut w = meet;
            while w != ROOT {
                path.push(w);
                w = pred[w];
            }
            path.reverse();
            let mut w = succ[meet];
            while w != ROOT {
                path.push(w);
                w = succ[w];
            }
            Some(path)
        }))
    }

    /// Dijkstra's algorithm from `sources` (NetworkX's `_dijkstra_multisource`): ignores steps
    /// that would exceed `cutoff`, stops once `target` is finished, and follows predecessors
    /// instead of successors with `reverse`.
    ///
    /// `track` is "lengths", "paths" or "preds". Returns `(order, dist, extra)`: the finished
    /// nodes in the order they were finished and their distances, plus
    /// - "lengths": None
    /// - "paths" without a target: the position in `order` of each node's parent (on the
    ///   path NetworkX reports; a source: its own position)
    /// - "paths" with a target: the path to `target` (None if not reached)
    /// - "preds": `(seen_order, preds)`, the reached nodes in the order they were first
    ///   reached and their predecessor lists
    ///
    /// Raises ValueError like NetworkX when a negative weight contradicts a finished distance.
    #[pyo3(signature = (sources, cutoff=None, target=None, track="lengths", reverse=false))]
    fn dijkstra(
        &self,
        py: Python<'_>,
        sources: Vec<usize>,
        cutoff: Option<f64>,
        target: Option<usize>,
        track: &str,
        reverse: bool,
    ) -> PyResult<(Vec<usize>, PyObject, PyObject)> {
        let sources = self.sources(sources)?;
        if let Some(t) = target {
            self.check_node(t)?;
        }
        let mode = match track {
            "lengths" => Track::Lengths,
            "paths" => Track::Parent,
            "preds" => Track::AllPreds,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "track must be 'lengths', 'paths' or 'preds', not {:?}",
                    track
                )))
            }
        };
        let csr = self.csr(reverse);
        let result = py
            .allow_threads(|| self.run_dijkstra(csr, &sources, cutoff, target, mode))
            .map_err(|_| contradiction_error())?;
        let dist = self.numbers(py, &result.dist)?;
        let extra = match mode {
            Track::Lengths => py.None(),
            Track::Parent => match target {
                None => self
                    .parent_positions(&result.order, &result.parent)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
                Some(t) => {
                    if result.order.last() != Some(&t) {
                        py.None()
                    } else {
                        let mut path = vec![t];
                        let mut v = result.parent[t];
                        while v != NONE && path.len() <= self.n {
                            path.push(v);
                            v = result.parent[v];
                        }
                        path.reverse();
                        path.into_pyobject(py)?.into_any().unbind()
                    }
                }
            },
            Track::AllPreds => {
                let mut all_preds = result.preds;
                let preds: Vec<Vec<usize>> = result
                    .seen_order
                    .iter()
                    .map(|&v| std::mem::take(&mut all_preds[v]))
                    .collect();
                (result.seen_order, preds)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        };
        Ok((result.order, dist, extra))
    }

    /// `dijkstra()` from each of `sources` separately, without a target (in parallel when
    /// big). Returns a list of `(order, dist, parents)` (parents as with "paths", else None),
    /// cut short before the first source whose search raises; the second item is True then.
    #[pyo3(signature = (sources, cutoff=None, paths=false))]
    #[allow(clippy::type_complexity)]
    fn dijkstra_many(
        &self,
        py: Python<'_>,
        sources: Vec<usize>,
        cutoff: Option<f64>,
        paths: bool,
    ) -> PyResult<(Vec<(Vec<usize>, PyObject, PyObject)>, bool)> {
        let sources = self.sources(sources)?;
        let csr = self.forward();
        let mode = if paths { Track::Parent } else { Track::Lengths };
        let run = |s: &usize| self.run_dijkstra(csr, std::slice::from_ref(s), cutoff, None, mode);
        let results: Vec<Result<Dijkstra, Contradiction>> = py.allow_threads(|| {
            if self.parallel(sources.len()) {
                sources.par_iter().map(run).collect()
            } else {
                sources.iter().map(run).collect()
            }
        });
        let mut out = Vec::with_capacity(results.len());
        for result in results {
            match result {
                Ok(r) => {
                    let dist = self.numbers(py, &r.dist)?;
                    let parents = if paths {
                        self.parent_positions(&r.order, &r.parent)
                            .into_pyobject(py)?
                            .into_any()
                            .unbind()
                    } else {
                        py.None()
                    };
                    out.push((r.order, dist, parents));
                }
                Err(Contradiction) => return Ok((out, true)),
            }
        }
        Ok((out, false))
    }

    /// NetworkX's `bidirectional_dijkstra`: `(length, path)` from `source` to `target`, or
    /// None if there is no path. Alternates between a search along successors from `source`
    /// and one along predecessors from `target`, with one insertion counter for both heaps.
    fn bidirectional_dijkstra(
        &self,
        py: Python<'_>,
        source: usize,
        target: usize,
    ) -> PyResult<Option<(PyObject, Vec<usize>)>> {
        self.check_node(source)?;
        self.check_node(target)?;
        if source == target {
            return Ok(Some((
                0i64.into_pyobject(py)?.into_any().unbind(),
                vec![source],
            )));
        }
        let csrs = [self.forward(), self.csr(true)];
        let n = self.n;
        let found = py.allow_threads(|| -> Result<Option<(f64, Vec<usize>)>, PyErr> {
            let mut finished = [vec![false; n], vec![false; n]];
            let mut dists = [vec![0.0f64; n], vec![0.0f64; n]];
            let mut seen = [vec![false; n], vec![false; n]];
            let mut seen_dist = [vec![f64::INFINITY; n], vec![f64::INFINITY; n]];
            let mut preds = [vec![NONE; n], vec![NONE; n]];
            preds[0][source] = ROOT;
            preds[1][target] = ROOT;
            seen[0][source] = true;
            seen_dist[0][source] = 0.0;
            seen[1][target] = true;
            seen_dist[1][target] = 0.0;
            let mut fringe = [BinaryHeap::new(), BinaryHeap::new()];
            fringe[0].push(Entry {
                dist: 0.0,
                count: 0,
                node: source,
            });
            fringe[1].push(Entry {
                dist: 0.0,
                count: 1,
                node: target,
            });
            let mut count = 2u64;
            let mut final_dist: Option<f64> = None;
            let mut meet = NONE;
            let mut direction = 1;
            while !fringe[0].is_empty() && !fringe[1].is_empty() {
                direction = 1 - direction;
                let other = 1 - direction;
                let Entry {
                    dist: d, node: v, ..
                } = fringe[direction].pop().expect("not empty");
                if finished[direction][v] {
                    continue;
                }
                finished[direction][v] = true;
                dists[direction][v] = d;
                if finished[other][v] {
                    let length = final_dist.unwrap_or(d + dists[other][v]);
                    let mut path = Vec::new();
                    let mut w = meet;
                    while w != ROOT && w != NONE {
                        path.push(w);
                        w = preds[0][w];
                    }
                    path.reverse();
                    if meet != NONE {
                        let mut w = preds[1][meet];
                        while w != ROOT && w != NONE {
                            path.push(w);
                            w = preds[1][w];
                        }
                    }
                    return Ok(Some((length, path)));
                }
                for (w, cost) in csrs[direction].arcs(v) {
                    let vw = d + cost;
                    if finished[direction][w] {
                        if vw < dists[direction][w] {
                            return Err(PyValueError::new_err(
                                "Contradictory paths found: negative weights?",
                            ));
                        }
                    } else if !seen[direction][w] || vw < seen_dist[direction][w] {
                        seen[direction][w] = true;
                        seen_dist[direction][w] = vw;
                        fringe[direction].push(Entry {
                            dist: vw,
                            count,
                            node: w,
                        });
                        count += 1;
                        preds[direction][w] = v;
                        if seen[other][w] {
                            let total = vw + seen_dist[other][w];
                            if final_dist.is_none_or(|f| f > total) {
                                final_dist = Some(total);
                                meet = w;
                            }
                        }
                    }
                }
            }
            Ok(None)
        })?;
        match found {
            None => Ok(None),
            Some((length, path)) => Ok(Some((self.number(py, length)?, path))),
        }
    }

    /// Whether every node can reach every other one (strongly connected, for a directed
    /// graph). Hidden arcs count as missing.
    fn is_strongly_connected(&self, py: Python<'_>) -> bool {
        if self.n == 0 {
            return false;
        }
        let reaches_all = |csr: &Csr| self.bfs(csr, 0, f64::INFINITY, false).0.len() == self.n;
        py.allow_threads(|| {
            reaches_all(self.forward()) && (!self.directed || reaches_all(self.csr(true)))
        })
    }

    /// The sum of the shortest-path distances from every node to every node it reaches, in
    /// node order and then in each search's order (NetworkX's `average_shortest_path_length`
    /// numerator). Counts edges if not `weighted`; with weights, floats are summed with
    /// Python's `sum()` algorithm in NetworkX's order, so the result is the same to the bit.
    fn distance_sum(&self, py: Python<'_>, weighted: bool) -> PyResult<PyObject> {
        let n = self.n;
        let csr = self.forward();
        if !weighted {
            let total: u128 = py.allow_threads(|| {
                let levels = |s: usize| -> u128 {
                    self.bfs(csr, s, f64::INFINITY, false)
                        .1
                        .iter()
                        .map(|&l| l as u128)
                        .sum()
                };
                if self.parallel(n) {
                    (0..n).into_par_iter().map(levels).sum()
                } else {
                    (0..n).map(levels).sum()
                }
            });
            return Ok(total.into_pyobject(py)?.into_any().unbind());
        }
        let chunk = (4_000_000 / n.max(1)).clamp(1, 1024);
        let integral = self.integral;
        let result = py.allow_threads(|| -> Result<(i128, f64), Contradiction> {
            let mut int_total: i128 = 0;
            let mut float_total = CompensatedSum::default();
            let mut start = 0;
            while start < n {
                let end = (start + chunk).min(n);
                let run = |s: usize| {
                    self.run_dijkstra(csr, &[s], None, None, Track::Lengths)
                        .map(|r| r.dist)
                };
                let dists: Vec<Result<Vec<f64>, Contradiction>> = if self.parallel(end - start) {
                    (start..end).into_par_iter().map(run).collect()
                } else {
                    (start..end).map(run).collect()
                };
                for d in dists {
                    let d = d?;
                    if integral {
                        int_total += d.iter().map(|&x| x as i128).sum::<i128>();
                    } else {
                        for x in d {
                            float_total.add(x);
                        }
                    }
                }
                start = end;
            }
            Ok((int_total, float_total.total()))
        });
        match result {
            Err(Contradiction) => Err(contradiction_error()),
            Ok((int_total, _)) if integral => Ok(int_total.into_pyobject(py)?.into_any().unbind()),
            Ok((_, float_total)) => Ok(float_total.into_pyobject(py)?.into_any().unbind()),
        }
    }

    /// Per source: how many nodes it reaches, the sum of their distances (as Python's
    /// `sum()` over NetworkX's distance dict computes it: exact for ints, compensated in
    /// search order for floats) and the largest distance. BFS levels unless `weighted`;
    /// over predecessors with `reverse`.
    ///
    /// Returns `(reached, sums, maxima, failed)`; `failed` is the position in `sources` of
    /// the first search that hit a negative-weight contradiction (results stop there), else
    /// None.
    #[pyo3(signature = (sources, weighted=false, reverse=false))]
    #[allow(clippy::type_complexity)]
    fn source_stats(
        &self,
        py: Python<'_>,
        sources: Vec<usize>,
        weighted: bool,
        reverse: bool,
    ) -> PyResult<(Vec<usize>, Vec<f64>, Vec<f64>, Option<usize>)> {
        let sources = self.sources(sources)?;
        let integral = !weighted || self.integral;
        Ok(py.allow_threads(|| {
            let mut reached = Vec::with_capacity(sources.len());
            let mut sums = Vec::with_capacity(sources.len());
            let mut maxima = Vec::with_capacity(sources.len());
            let failed = self.visit_searches(&sources, weighted, reverse, |_, _, dist| {
                reached.push(dist.len());
                if integral {
                    sums.push(dist.iter().map(|&x| x as i128).sum::<i128>() as f64);
                } else {
                    let mut sum = CompensatedSum::default();
                    for &x in dist {
                        sum.add(x);
                    }
                    sums.push(sum.total());
                }
                // (Python's max() keeps the first of equal values: the source's 0)
                maxima.push(dist.iter().copied().fold(f64::NEG_INFINITY, |m, x| {
                    if x > m {
                        x
                    } else {
                        m
                    }
                }));
            });
            (reached, sums, maxima, failed)
        }))
    }

    /// Sums of inverse distances `1 / d` over the pairs (source, reached node) with `d != 0`,
    /// added in the order NetworkX's `harmonic_centrality` and `harmonic_diameter` add them:
    /// sources in the given order, then each search's order. Only reached nodes with
    /// `members[node]` count (all if None). Each term goes to the reached node, or to the
    /// source with `at_source`. BFS levels unless `weighted`; over predecessors with
    /// `reverse`.
    ///
    /// Returns `(per_node, touched, total, failed)`: the per-node sums, which nodes got a
    /// term, the sum of all terms in order, and the position in `sources` of the first
    /// search that hit a negative-weight contradiction (None if none did).
    #[pyo3(signature = (sources, weighted=false, reverse=false, members=None, at_source=false))]
    #[allow(clippy::type_complexity)]
    fn inverse_distance_sums(
        &self,
        py: Python<'_>,
        sources: Vec<usize>,
        weighted: bool,
        reverse: bool,
        members: Option<Vec<bool>>,
        at_source: bool,
    ) -> PyResult<(Vec<f64>, Vec<bool>, f64, Option<usize>)> {
        let sources = self.sources(sources)?;
        if members.as_ref().is_some_and(|m| m.len() != self.n) {
            return Err(PyValueError::new_err(
                "members must have one entry per node",
            ));
        }
        let n = self.n;
        Ok(py.allow_threads(|| {
            let mut per_node = vec![0.0f64; n];
            let mut touched = vec![false; n];
            let mut total = 0.0f64;
            let failed = self.visit_searches(&sources, weighted, reverse, |i, order, dist| {
                let source = sources[i];
                for (&u, &d) in order.iter().zip(dist) {
                    if d != 0.0 && members.as_ref().is_none_or(|m| m[u]) {
                        let x = 1.0 / d;
                        let at = if at_source { source } else { u };
                        per_node[at] = if touched[at] { per_node[at] + x } else { x };
                        touched[at] = true;
                        total += x;
                    }
                }
            });
            (per_node, touched, total, failed)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adjacency(n: usize, arcs: &[(u64, u64, f64)], directed: bool) -> PathAdjacency {
        Python::with_gil(|py| {
            let sources: Vec<u64> = arcs.iter().map(|a| a.0).collect();
            let targets: Vec<u64> = arcs.iter().map(|a| a.1).collect();
            let weights: Vec<f64> = arcs.iter().map(|a| a.2).collect();
            let weights = weights.into_pyobject(py).unwrap();
            PathAdjacency::new(n, sources, targets, Some(weights.as_any()), directed).unwrap()
        })
    }

    #[test]
    fn neighbors_keep_the_given_order_and_parallel_arcs_merge() {
        // 1 -> 3 (twice, costs 5 and 2), then 1 -> 2
        let adj = adjacency(3, &[(1, 3, 5.0), (1, 3, 2.0), (1, 2, 1.0)], true);
        assert_eq!(adj.out.targets(0), &[2, 1]);
        assert_eq!(adj.out.weights[..2], [2.0, 1.0]);
        let rev = adj.csr(true);
        assert_eq!(rev.targets(2), &[0]);
        assert_eq!(rev.targets(1), &[0]);
    }

    #[test]
    fn dijkstra_breaks_ties_by_insertion_order() {
        // Undirected square 1-2-4, 1-3-4 with unit weights; 1 lists 3 before 2
        let arcs = [
            (1, 3, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 4, 1.0),
            (3, 1, 1.0),
            (3, 4, 1.0),
            (4, 2, 1.0),
            (4, 3, 1.0),
        ];
        let adj = adjacency(4, &arcs, false);
        let r = adj
            .run_dijkstra(adj.forward(), &[0], None, None, Track::Parent)
            .ok()
            .unwrap();
        assert_eq!(r.order, vec![0, 2, 1, 3]);
        assert_eq!(r.parent[3], 2);
    }

    #[test]
    fn compensated_sum_matches_python() {
        let mut s = CompensatedSum::default();
        for x in [1e16, 1.0, -1e16] {
            s.add(x);
        }
        assert_eq!(s.total(), 1.0);
    }
}
