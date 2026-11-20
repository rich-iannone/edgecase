//! Depth-first structure kernels with NetworkX's exact order, over a `PathAdjacency`.
//!
//! Biconnected components and articulation points, chain decomposition (and so bridges),
//! cycle bases, local bridges and girth, each following NetworkX's implementation step by
//! step (neighbors in adjacency order, the same stacks and tie rules), so the components,
//! chains, cycles and their order are NetworkX's. Nodes are 0-based positions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::algorithms::path_search::PathAdjacency;

const NONE: usize = usize::MAX;

/// NetworkX's `_biconnected_dfs`: with `components`, the edge lists of the biconnected
/// components (as `biconnected_component_edges` yields them); otherwise the articulation
/// points in the order found, with repeats (`articulation_points` drops those).
#[pyfunction]
#[pyo3(signature = (adjacency, components=true))]
pub fn nx_biconnected_dfs(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    components: bool,
) -> (Vec<Vec<(usize, usize)>>, Vec<usize>) {
    py.allow_threads(|| {
        let (indptr, indices, _) = adjacency.rows();
        let n = adjacency.len();
        let mut visited = vec![false; n];
        let mut discovery = vec![0usize; n];
        let mut low = vec![0usize; n];
        let mut tree_index = vec![0usize; n];
        let mut comps = Vec::new();
        let mut articulation = Vec::new();
        let mut edge_stack: Vec<(usize, usize)> = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            discovery[start] = 0;
            low[start] = 0;
            let mut count = 1;
            let mut root_children = 0;
            visited[start] = true;
            edge_stack.clear();
            // (grandparent, parent, next arc of parent)
            let mut stack = vec![(start, start, indptr[start])];
            while let Some(&(grandparent, parent, k)) = stack.last() {
                if k < indptr[parent + 1] {
                    stack.last_mut().expect("not empty").2 = k + 1;
                    let child = indices[k];
                    if grandparent == child {
                        continue;
                    }
                    if visited[child] {
                        if discovery[child] <= discovery[parent] {
                            low[parent] = low[parent].min(discovery[child]);
                            if components {
                                edge_stack.push((parent, child));
                            }
                        }
                    } else {
                        discovery[child] = count;
                        low[child] = count;
                        count += 1;
                        visited[child] = true;
                        stack.push((parent, child, indptr[child]));
                        if components {
                            tree_index[child] = edge_stack.len();
                            edge_stack.push((parent, child));
                        }
                    }
                } else {
                    stack.pop();
                    if stack.len() > 1 {
                        if low[parent] >= discovery[grandparent] {
                            if components {
                                comps.push(edge_stack.split_off(tree_index[parent]));
                            } else {
                                articulation.push(grandparent);
                            }
                        }
                        low[grandparent] = low[parent].min(low[grandparent]);
                    } else if !stack.is_empty() {
                        root_children += 1;
                        if components {
                            comps.push(edge_stack.split_off(tree_index[parent]));
                        }
                    }
                }
            }
            if !components && root_children > 1 {
                articulation.push(start);
            }
        }
        (comps, articulation)
    })
}

/// NetworkX's `chain_decomposition(G, root)`: the chains (lists of edges) in order, from the
/// depth-first search of every component, or only `root`'s.
#[pyfunction]
#[pyo3(signature = (adjacency, root=None))]
pub fn nx_chain_decomposition(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    root: Option<usize>,
) -> PyResult<Vec<Vec<(usize, usize)>>> {
    let n = adjacency.len();
    if root.is_some_and(|r| r >= n) {
        return Err(PyValueError::new_err("root position out of range"));
    }
    Ok(py.allow_threads(|| {
        let (indptr, indices, _) = adjacency.rows();
        let mut has_parent = vec![false; n];
        let mut parent = vec![NONE; n];
        let mut nontree: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut nontree_keys: FxHashSet<(usize, usize)> = FxHashSet::default();
        let mut nodes = Vec::new();
        let starts: Vec<usize> = match root {
            Some(r) => vec![r],
            None => (0..n).collect(),
        };
        for start in starts {
            if has_parent[start] {
                continue;
            }
            has_parent[start] = true;
            nodes.push(start);
            let mut stack = vec![(start, indptr[start])];
            while let Some(&(node, k)) = stack.last() {
                if k == indptr[node + 1] {
                    stack.pop();
                    continue;
                }
                stack.last_mut().expect("not empty").1 = k + 1;
                let nbr = indices[k];
                if has_parent[nbr] {
                    if nbr != parent[node] && !nontree_keys.contains(&(node, nbr)) {
                        // nontree[nbr][node] = None
                        if nontree_keys.insert((nbr, node)) {
                            nontree[nbr].push(node);
                        }
                    }
                } else {
                    parent[nbr] = node;
                    has_parent[nbr] = true;
                    nodes.push(nbr);
                    stack.push((nbr, indptr[nbr]));
                }
            }
        }
        let mut visited = vec![false; n];
        let mut chains = Vec::new();
        for &u in &nodes {
            visited[u] = true;
            for &v0 in &nontree[u] {
                let mut chain = vec![(u, v0)];
                let mut v = v0;
                while !visited[v] {
                    visited[v] = true;
                    let p = parent[v];
                    chain.push((v, p));
                    v = p;
                }
                chains.push(chain);
            }
        }
        chains
    }))
}

/// NetworkX's `cycle_basis(G, root)`: the cycles (lists of nodes) in the order found. The
/// search trees start from `root` and then from the last node not yet reached, as NetworkX's
/// `dict.popitem()` picks them. The cycles are lists of `nodes[position]` (the node keys).
#[pyfunction]
#[pyo3(signature = (adjacency, nodes, root=None))]
pub fn nx_cycle_basis<'py>(
    adjacency: &PathAdjacency,
    nodes: &Bound<'py, PyList>,
    root: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let n = adjacency.len();
    if root.is_some_and(|r| r >= n) || nodes.len() != n {
        return Err(PyValueError::new_err(
            "root position out of range, or not one key per node",
        ));
    }
    let cycles = nodes.py().allow_threads(|| cycle_basis(adjacency, root));
    let result = PyList::empty(nodes.py());
    for cycle in cycles {
        let keys: Vec<Bound<'py, PyAny>> = cycle
            .into_iter()
            .map(|v| nodes.get_item(v))
            .collect::<PyResult<_>>()?;
        result.append(PyList::new(nodes.py(), keys)?)?;
    }
    Ok(result)
}

#[allow(clippy::map_entry)]
fn cycle_basis(adjacency: &PathAdjacency, root: Option<usize>) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    {
        let (indptr, indices, _) = adjacency.rows();
        let mut remaining = vec![true; n];
        let mut last = n; // remaining nodes are among 0..last
        let mut cycles = Vec::new();
        let mut pred = vec![NONE; n];
        let mut used: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
        let mut root = root;
        loop {
            let r = match root.take() {
                Some(r) => r,
                None => {
                    while last > 0 && !remaining[last - 1] {
                        last -= 1;
                    }
                    if last == 0 {
                        break;
                    }
                    last -= 1;
                    remaining[last] = false;
                    last
                }
            };
            used.clear();
            let mut reached = vec![r];
            pred[r] = r;
            used.insert(r, FxHashSet::default());
            let mut stack = vec![r];
            while let Some(z) = stack.pop() {
                for k in indptr[z]..indptr[z + 1] {
                    let nbr = indices[k];
                    if !used.contains_key(&nbr) {
                        pred[nbr] = z;
                        reached.push(nbr);
                        stack.push(nbr);
                        used.insert(nbr, FxHashSet::from_iter([z]));
                    } else if nbr == z {
                        cycles.push(vec![z]);
                    } else if !used[&z].contains(&nbr) {
                        let pn = &used[&nbr];
                        let mut cycle = vec![nbr, z];
                        let mut p = pred[z];
                        while !pn.contains(&p) {
                            cycle.push(p);
                            p = pred[p];
                        }
                        cycle.push(p);
                        cycles.push(cycle);
                        used.get_mut(&nbr).expect("used").insert(z);
                    }
                }
            }
            for v in reached {
                remaining[v] = false;
            }
            if !remaining.iter().take(last).any(|&r| r) {
                break;
            }
        }
        cycles
    }
}

/// For each edge `(us[i], vs[i])` of an undirected graph: whether it is a local bridge (its
/// ends share no neighbor) and, with `with_span`, its span: the shortest distance between its
/// ends without it (inf if none). Returns `(is_local_bridge, spans)`; Err on a
/// negative-weight contradiction.
#[pyfunction]
#[pyo3(signature = (adjacency, us, vs, with_span=true))]
pub fn nx_local_bridges(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    us: Vec<usize>,
    vs: Vec<usize>,
    with_span: bool,
) -> PyResult<(Vec<bool>, Vec<f64>)> {
    let n = adjacency.len();
    if us.len() != vs.len() || us.iter().chain(&vs).any(|&x| x >= n) {
        return Err(PyValueError::new_err(
            "us and vs must be node positions of the same length",
        ));
    }
    let (indptr, indices, _) = adjacency.rows();
    let check = |marks: &mut Vec<usize>, i: usize| -> Result<(bool, f64), ()> {
        let (u, v) = (us[i], vs[i]);
        let stamp = i + 1;
        for &w in &indices[indptr[v]..indptr[v + 1]] {
            marks[w] = stamp;
        }
        let shared = indices[indptr[u]..indptr[u + 1]]
            .iter()
            .any(|&w| marks[w] == stamp);
        if shared || !with_span {
            return Ok((!shared, f64::NAN));
        }
        let span = adjacency.distance_avoiding(u, v)?;
        Ok((true, span.unwrap_or(f64::INFINITY)))
    };
    let pairs = us.len();
    let results: Vec<Result<(bool, f64), ()>> = py.allow_threads(|| {
        if pairs > 1 && pairs * (n + indices.len()) > 2_000_000 {
            (0..pairs)
                .into_par_iter()
                .map_init(|| vec![0usize; n], |marks, i| check(marks, i))
                .collect()
        } else {
            let mut marks = vec![0usize; n];
            (0..pairs).map(|i| check(&mut marks, i)).collect()
        }
    });
    let mut flags = Vec::with_capacity(pairs);
    let mut spans = Vec::with_capacity(pairs);
    for result in results {
        let (flag, span) = result.map_err(|_| {
            PyValueError::new_err(("Contradictory paths found:", "negative weights?"))
        })?;
        flags.push(flag);
        spans.push(span);
    }
    Ok((flags, spans))
}

/// The length of the shortest cycle from a breadth-first search from `source` (NetworkX's
/// `girth` loop over `bfs_labeled_edges` for one node), given the best length so far
fn girth_from(adjacency: &PathAdjacency, source: usize) -> Option<usize> {
    let (indptr, indices, _) = adjacency.rows();
    let n = adjacency.len();
    let mut depth = vec![usize::MAX; n];
    let mut done = vec![false; n]; // (bfs_labeled_edges' `visited`)
    let mut best: Option<usize> = None;
    let mut limit: Option<usize> = None; // depth_limit (inf while None)
    depth[source] = 0;
    let mut queue = VecDeque::from([source]);
    while let Some(u) = queue.pop_front() {
        let du = depth[u];
        for &v in &indices[indptr[u]..indptr[u + 1]] {
            if limit.is_some_and(|l| du > l) {
                return best;
            }
            if depth[v] == usize::MAX {
                depth[v] = du + 1;
                queue.push_back(v);
            } else {
                let dv = depth[v];
                let level = du == dv;
                if level && done[v] {
                    continue; // (not yielded)
                }
                if level || du < dv {
                    let length = du + du + 2 - usize::from(level);
                    if best.is_none_or(|b| length < b) {
                        best = Some(length);
                        limit = Some(du - usize::from(level));
                    }
                }
            }
        }
        done[u] = true;
    }
    best
}

/// NetworkX's `girth` of an undirected graph: the length of a shortest cycle (None if the
/// graph has none)
#[pyfunction]
pub fn nx_girth(py: Python<'_>, adjacency: &PathAdjacency) -> Option<usize> {
    let n = adjacency.len();
    py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .filter_map(|s| girth_from(adjacency, s))
            .min()
    })
}

/// A graph's arcs one by one (parallel edges each on their own, unlike `PathAdjacency`), for
/// NetworkX's edge-by-edge depth-first search (`edge_dfs`, and `find_cycle` on it).
///
/// Built from the arcs `sources[i] -> targets[i]` (node IDs `1..=n`) in adjacency order (an
/// undirected edge from both ends, a self-loop once) and, for a directed graph, optionally
/// the predecessor order (`pred_nodes[i]` has the `pred_counts[i]` predecessors that follow
/// in `preds`), which orders `in_edges`. The results are arc indices.
#[pyclass(frozen, module = "edgecase._core")]
pub struct ArcAdjacency {
    directed: bool,
    /// Out-arcs of node `v`: arcs `out_start[v]..out_start[v + 1]` (arcs are grouped by
    /// source, in order)
    out_start: Vec<usize>,
    out_arcs: Vec<usize>,
    /// In-arcs of node `v` (directed graphs), in predecessor order
    in_start: Vec<usize>,
    in_arcs: Vec<usize>,
    arc_source: Vec<usize>,
    arc_target: Vec<usize>,
    /// The edge behind each arc (an undirected edge's two arcs share one)
    edge_of_arc: Vec<usize>,
}

#[pymethods]
impl ArcAdjacency {
    #[new]
    #[pyo3(signature = (num_nodes, sources, targets, directed, pred_nodes=None, pred_counts=None,
                        preds=None))]
    fn new(
        num_nodes: usize,
        sources: Vec<u64>,
        targets: Vec<u64>,
        directed: bool,
        pred_nodes: Option<Vec<u64>>,
        pred_counts: Option<Vec<usize>>,
        preds: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        let n = num_nodes;
        let m = sources.len();
        if targets.len() != m
            || sources
                .iter()
                .chain(&targets)
                .any(|&x| x == 0 || x as usize > n)
        {
            return Err(PyValueError::new_err(
                "sources and targets must be node IDs 1..=num_nodes of the same length",
            ));
        }
        let arc_source: Vec<usize> = sources.iter().map(|&s| s as usize - 1).collect();
        let arc_target: Vec<usize> = targets.iter().map(|&t| t as usize - 1).collect();
        // Out-arcs by source (stable)
        let mut out_start = vec![0usize; n + 1];
        for &s in &arc_source {
            out_start[s + 1] += 1;
        }
        for i in 0..n {
            out_start[i + 1] += out_start[i];
        }
        let mut next = out_start.clone();
        let mut out_arcs = vec![0usize; m];
        for (a, &s) in arc_source.iter().enumerate() {
            out_arcs[next[s]] = a;
            next[s] += 1;
        }
        // Runs of parallel arcs: (source, target) -> (first position in out_arcs, count)
        let mut runs: FxHashMap<(usize, usize), (usize, usize)> = FxHashMap::default();
        let mut rank = vec![0usize; m]; // position of each arc within its run
        for v in 0..n {
            let mut k = out_start[v];
            while k < out_start[v + 1] {
                let t = arc_target[out_arcs[k]];
                let start = k;
                while k < out_start[v + 1] && arc_target[out_arcs[k]] == t {
                    rank[out_arcs[k]] = k - start;
                    k += 1;
                }
                runs.entry((v, t)).or_insert((start, k - start));
            }
        }
        // Edge identities
        let mut edge_of_arc = vec![0usize; m];
        if directed {
            edge_of_arc = (0..m).collect();
        } else {
            let mut ids: FxHashMap<(usize, usize, usize), usize> = FxHashMap::default();
            for a in 0..m {
                let (s, t) = (arc_source[a], arc_target[a]);
                let key = (s.min(t), s.max(t), rank[a]);
                let next_id = ids.len();
                edge_of_arc[a] = *ids.entry(key).or_insert(next_id);
            }
        }
        // In-arcs (directed): per node, its predecessors' runs of arcs into it
        let mut in_start = vec![0usize; n + 1];
        let mut in_arcs = Vec::new();
        if directed {
            let mut lists: Vec<Vec<usize>> = vec![Vec::new(); n];
            match (pred_nodes, pred_counts, preds) {
                (Some(pred_nodes), Some(pred_counts), Some(preds)) => {
                    let mut i = 0;
                    for (&v, &count) in pred_nodes.iter().zip(&pred_counts) {
                        let v = v as usize - 1;
                        for &u in &preds[i..i + count] {
                            let u = u as usize - 1;
                            if let Some(&(start, len)) = runs.get(&(u, v)) {
                                lists[v].extend_from_slice(&out_arcs[start..start + len]);
                            }
                        }
                        i += count;
                    }
                }
                _ => {
                    for &a in &out_arcs {
                        lists[arc_target[a]].push(a);
                    }
                }
            }
            for v in 0..n {
                in_arcs.extend_from_slice(&lists[v]);
                in_start[v + 1] = in_arcs.len();
            }
        }
        Ok(ArcAdjacency {
            directed,
            out_start,
            out_arcs,
            in_start,
            in_arcs,
            arc_source,
            arc_target,
            edge_of_arc,
        })
    }

    /// NetworkX's `edge_dfs(G, start, orientation)` from each of `starts` (positions) in
    /// turn, sharing visited edges and nodes: the edges as `(arc, reverse)`, `reverse` for an
    /// in-edge. `orientation` is None, "original", "reverse" or "ignore" (directed graphs;
    /// an undirected graph only ever follows its edges forward).
    #[pyo3(signature = (starts, orientation=None))]
    fn edge_dfs(
        &self,
        py: Python<'_>,
        starts: Vec<usize>,
        orientation: Option<&str>,
    ) -> PyResult<Vec<(usize, bool)>> {
        let n = self.out_start.len() - 1;
        if starts.iter().any(|&s| s >= n) {
            return Err(PyValueError::new_err("start position out of range"));
        }
        let (outward, inward) = match (self.directed, orientation) {
            (false, _) | (true, None) | (true, Some("original")) => (true, false),
            (true, Some("reverse")) => (false, true),
            (true, Some("ignore")) => (true, true),
            _ => return Err(PyValueError::new_err("invalid orientation argument.")),
        };
        Ok(py.allow_threads(|| {
            let mut visited_edges = vec![false; self.edge_of_arc.len()];
            let mut visited_nodes = vec![false; n];
            // Per node: (next out position, next in position)
            let mut cursor = vec![(0usize, 0usize); n];
            let mut result = Vec::new();
            for &start in &starts {
                let mut stack = vec![start];
                while let Some(&node) = stack.last() {
                    if !visited_nodes[node] {
                        visited_nodes[node] = true;
                        cursor[node] = (self.out_start[node], self.in_start[node]);
                    }
                    let (out_k, in_k) = cursor[node];
                    let edge = if outward && out_k < self.out_start[node + 1] {
                        cursor[node].0 += 1;
                        Some((self.out_arcs[out_k], false))
                    } else if inward && in_k < self.in_start[node + 1] {
                        cursor[node].1 += 1;
                        Some((self.in_arcs[in_k], true))
                    } else {
                        None
                    };
                    match edge {
                        None => {
                            stack.pop();
                        }
                        Some((arc, reverse)) => {
                            let id = self.edge_of_arc[arc];
                            if !visited_edges[id] {
                                visited_edges[id] = true;
                                // (an in-edge leads back to its tail; edges from the
                                // node lead to their head)
                                if reverse {
                                    stack.push(self.arc_source[arc]);
                                } else {
                                    stack.push(self.arc_target[arc]);
                                }
                                result.push((arc, reverse));
                            }
                        }
                    }
                }
            }
            result
        }))
    }

    /// NetworkX's `edge_bfs(G, starts, orientation)`: the edges as `(arc, reverse)` in the
    /// order NetworkX yields them (orientation as for `edge_dfs()`).
    #[pyo3(signature = (starts, orientation=None))]
    fn edge_bfs(
        &self,
        py: Python<'_>,
        starts: Vec<usize>,
        orientation: Option<&str>,
    ) -> PyResult<Vec<(usize, bool)>> {
        let n = self.out_start.len() - 1;
        if starts.iter().any(|&s| s >= n) {
            return Err(PyValueError::new_err("start position out of range"));
        }
        let (outward, inward) = match (self.directed, orientation) {
            (false, _) | (true, None) | (true, Some("original")) => (true, false),
            (true, Some("reverse")) => (false, true),
            (true, Some("ignore")) => (true, true),
            _ => return Err(PyValueError::new_err("invalid orientation argument.")),
        };
        Ok(py.allow_threads(|| {
            let mut visited_nodes = vec![false; n];
            for &s in &starts {
                visited_nodes[s] = true;
            }
            let mut visited_edges = vec![false; self.edge_of_arc.len()];
            let mut queue: VecDeque<usize> = starts.iter().copied().collect();
            let mut result = Vec::new();
            while let Some(parent) = queue.pop_front() {
                let outs = if outward {
                    &self.out_arcs[self.out_start[parent]..self.out_start[parent + 1]]
                } else {
                    &[][..]
                };
                let ins = if inward {
                    &self.in_arcs[self.in_start[parent]..self.in_start[parent + 1]]
                } else {
                    &[][..]
                };
                let edges = outs
                    .iter()
                    .map(|&a| (a, false))
                    .chain(ins.iter().map(|&a| (a, true)));
                for (arc, reverse) in edges {
                    let child = if reverse {
                        self.arc_source[arc]
                    } else {
                        self.arc_target[arc]
                    };
                    if !visited_nodes[child] {
                        visited_nodes[child] = true;
                        queue.push_back(child);
                    }
                    let id = self.edge_of_arc[arc];
                    if !visited_edges[id] {
                        visited_edges[id] = true;
                        result.push((arc, reverse));
                    }
                }
            }
            result
        }))
    }

    /// NetworkX's `lexicographical_topological_sort`: nodes with no remaining in-edges are
    /// taken smallest `rank` first (the caller ranks the nodes by NetworkX's sort key).
    /// Returns the order and whether nodes on a cycle were left over.
    fn lexicographical_topological_sort(
        &self,
        py: Python<'_>,
        rank: Vec<usize>,
    ) -> PyResult<(Vec<usize>, bool)> {
        let n = self.out_start.len() - 1;
        if rank.len() != n {
            return Err(PyValueError::new_err("rank must have one entry per node"));
        }
        Ok(py.allow_threads(|| {
            let mut indegree = vec![0usize; n];
            for &t in &self.arc_target {
                indegree[t] += 1;
            }
            let mut heap: BinaryHeap<Reverse<(usize, usize)>> = (0..n)
                .filter(|&v| indegree[v] == 0)
                .map(|v| Reverse((rank[v], v)))
                .collect();
            let mut order = Vec::with_capacity(n);
            while let Some(Reverse((_, node))) = heap.pop() {
                for &a in &self.out_arcs[self.out_start[node]..self.out_start[node + 1]] {
                    let child = self.arc_target[a];
                    indegree[child] -= 1;
                    if indegree[child] == 0 {
                        heap.push(Reverse((rank[child], child)));
                    }
                }
                order.push(node);
            }
            let left_over = order.len() < n;
            (order, left_over)
        }))
    }

    /// NetworkX's `dag_longest_path` over `topo_order` (positions), with `weights` per arc
    /// (in arc order): each node's best predecessor in predecessor order (the heaviest of
    /// parallel edges), a node starting afresh when every way in is negative. Returns the
    /// path and its length as `dag_longest_path_length` adds it up; None if `topo_order`
    /// reaches a node before one of its predecessors.
    fn dag_longest_path(
        &self,
        py: Python<'_>,
        topo_order: Vec<usize>,
        weights: Vec<f64>,
    ) -> PyResult<Option<(Vec<usize>, f64)>> {
        let n = self.out_start.len() - 1;
        if weights.len() != self.arc_source.len() || topo_order.iter().any(|&v| v >= n) {
            return Err(PyValueError::new_err(
                "weights must have one value per arc, topo_order node positions",
            ));
        }
        // The heaviest of a run of parallel arcs (the first of the heaviest)
        let heaviest = |arcs: &[usize]| -> f64 {
            let mut best = weights[arcs[0]];
            for &a in &arcs[1..] {
                if weights[a] > best {
                    best = weights[a];
                }
            }
            best
        };
        Ok(py.allow_threads(|| {
            let mut dist: Vec<Option<(f64, usize)>> = vec![None; n];
            let mut order = Vec::new();
            for &v in &topo_order {
                let ins = &self.in_arcs[self.in_start[v]..self.in_start[v + 1]];
                let mut best: Option<(f64, usize)> = None;
                let mut k = 0;
                while k < ins.len() {
                    let u = self.arc_source[ins[k]];
                    let start = k;
                    while k < ins.len() && self.arc_source[ins[k]] == u {
                        k += 1;
                    }
                    let (du, _) = dist[u]?;
                    let candidate = du + heaviest(&ins[start..k]);
                    if best.is_none_or(|(b, _)| candidate > b) {
                        best = Some((candidate, u));
                    }
                }
                let maxu = best.unwrap_or((0.0, v));
                if dist[v].is_none() {
                    order.push(v);
                }
                dist[v] = Some(if maxu.0 >= 0.0 { maxu } else { (0.0, v) });
            }
            // The end: the first node with the greatest distance
            let mut end = *order.first()?;
            for &v in &order {
                if dist[v]?.0 > dist[end]?.0 {
                    end = v;
                }
            }
            let mut path = Vec::new();
            let mut u = NONE;
            let mut v = end;
            while u != v {
                path.push(v);
                u = v;
                v = dist[v]?.1;
            }
            path.reverse();
            // The length as NetworkX adds it up along the path
            let mut length = 0.0;
            for pair in path.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let run: Vec<usize> = self.out_arcs[self.out_start[a]..self.out_start[a + 1]]
                    .iter()
                    .copied()
                    .filter(|&x| self.arc_target[x] == b)
                    .collect();
                length += heaviest(&run);
            }
            Some((path, length))
        }))
    }

    /// NetworkX's Hierholzer walk (`_simplegraph_eulerian_circuit` /
    /// `_multigraph_eulerian_circuit`) from `source`, always taking the first remaining edge
    /// of the current node in adjacency order. NetworkX walks a copy: `G.reverse()` with
    /// `reverse` (directed graphs), whose edges out of a node are its in-edges in arc order,
    /// or `G.copy()`, whose edges are in the order the copy re-adds them. Returns `(last_vertex, current_vertex, arc)` per step as NetworkX yields them,
    /// `arc` being the edge that led to `last_vertex`.
    #[pyo3(signature = (source, reverse=false))]
    fn eulerian_circuit(
        &self,
        py: Python<'_>,
        source: usize,
        reverse: bool,
    ) -> PyResult<Vec<(usize, usize, usize)>> {
        let n = self.out_start.len() - 1;
        if source >= n {
            return Err(PyValueError::new_err("source position out of range"));
        }
        Ok(py.allow_threads(|| {
            // Each node's edges in the walked graph: (start, arcs)
            let (start, arcs): (Vec<usize>, Vec<usize>) = if self.directed && reverse {
                let mut start = vec![0usize; n + 1];
                for &a in &self.out_arcs {
                    start[self.arc_target[a] + 1] += 1;
                }
                for i in 0..n {
                    start[i + 1] += start[i];
                }
                let mut next = start.clone();
                let mut arcs = vec![0usize; self.out_arcs.len()];
                for &a in &self.out_arcs {
                    let t = self.arc_target[a];
                    arcs[next[t]] = a;
                    next[t] += 1;
                }
                (start, arcs)
            } else if self.directed {
                (self.out_start.clone(), self.out_arcs.clone())
            } else {
                // NetworkX walks `G.copy()`, which re-adds the edges node by node, so each
                // node lists first its neighbors that come before it (in node order) and then
                // the others in its own order
                let mut arcs = Vec::with_capacity(self.out_arcs.len());
                for v in 0..n {
                    let row = &self.out_arcs[self.out_start[v]..self.out_start[v + 1]];
                    let mut earlier: Vec<usize> = row
                        .iter()
                        .copied()
                        .filter(|&a| self.arc_target[a] < v)
                        .collect();
                    earlier.sort_by_key(|&a| self.arc_target[a]);
                    arcs.extend(earlier);
                    arcs.extend(row.iter().copied().filter(|&a| self.arc_target[a] >= v));
                }
                (self.out_start.clone(), arcs)
            };
            let mut removed = vec![false; self.edge_of_arc.len()];
            let mut ptr: Vec<usize> = start[..n].to_vec();
            let mut stack: Vec<(usize, usize)> = vec![(source, NONE)];
            let mut last: Option<(usize, usize)> = None;
            let mut steps = Vec::new();
            while let Some(&(current, current_arc)) = stack.last() {
                while ptr[current] < start[current + 1]
                    && removed[self.edge_of_arc[arcs[ptr[current]]]]
                {
                    ptr[current] += 1;
                }
                if ptr[current] == start[current + 1] {
                    if let Some((last_vertex, last_arc)) = last {
                        steps.push((last_vertex, current, last_arc));
                    }
                    last = Some((current, current_arc));
                    stack.pop();
                } else {
                    let a = arcs[ptr[current]];
                    removed[self.edge_of_arc[a]] = true;
                    let next = if self.directed && reverse {
                        self.arc_source[a]
                    } else {
                        self.arc_target[a]
                    };
                    stack.push((next, a));
                }
            }
            steps
        }))
    }
}

/// NetworkX's `bipartite.color`: a depth-first two-coloring from each node with neighbors
/// (successors, for a directed graph), over predecessors then successors for a directed
/// graph. Returns the colored nodes in the order colored with their colors (1 for the start
/// of each search), or None if the graph is not bipartite. (NetworkX then colors the
/// isolated nodes 0.)
#[pyfunction]
pub fn nx_bipartite_color(
    py: Python<'_>,
    adjacency: &PathAdjacency,
) -> Option<(Vec<usize>, Vec<u8>)> {
    let n = adjacency.len();
    let directed = adjacency.is_directed();
    py.allow_threads(|| {
        let (indptr, indices, _) = adjacency.rows();
        let (rindptr, rindices) = if directed {
            let (p, i, _) = adjacency.reverse_rows();
            (p, i)
        } else {
            (indptr, indices)
        };
        let mut color = vec![u8::MAX; n];
        let mut order = Vec::new();
        let mut colors = Vec::new();
        for start in 0..n {
            if color[start] != u8::MAX || indptr[start] == indptr[start + 1] {
                continue;
            }
            color[start] = 1;
            order.push(start);
            colors.push(1);
            let mut queue = vec![start];
            while let Some(v) = queue.pop() {
                let c = 1 - color[v];
                let preds: &[usize] = if directed {
                    &rindices[rindptr[v]..rindptr[v + 1]]
                } else {
                    &[]
                };
                for &w in preds.iter().chain(&indices[indptr[v]..indptr[v + 1]]) {
                    if color[w] != u8::MAX {
                        if color[w] == color[v] {
                            return None;
                        }
                    } else {
                        color[w] = c;
                        order.push(w);
                        colors.push(c);
                        queue.push(w);
                    }
                }
            }
        }
        Some((order, colors))
    })
}

/// NetworkX's `immediate_dominators(G, start)` (Cooper, Harvey and Kennedy's iteration over
/// the reverse postorder of a depth-first search, predecessors in predecessor order).
/// Returns the nodes in the order they got a dominator and their immediate dominators.
#[pyfunction]
pub fn nx_immediate_dominators(
    py: Python<'_>,
    adjacency: &PathAdjacency,
    start: usize,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let n = adjacency.len();
    if start >= n {
        return Err(PyValueError::new_err("start position out of range"));
    }
    Ok(py.allow_threads(|| {
        let (indptr, indices, _) = adjacency.rows();
        let (rindptr, rindices, _) = adjacency.reverse_rows();
        // Depth-first postorder from start (nx.dfs_postorder_nodes)
        let mut seen = vec![false; n];
        let mut postorder = Vec::new();
        let mut stack = vec![(start, indptr[start])];
        seen[start] = true;
        while let Some(&(v, k)) = stack.last() {
            if k < indptr[v + 1] {
                stack.last_mut().expect("not empty").1 = k + 1;
                let w = indices[k];
                if !seen[w] {
                    seen[w] = true;
                    stack.push((w, indptr[w]));
                }
            } else {
                postorder.push(v);
                stack.pop();
            }
        }
        let mut dfn = vec![NONE; n];
        for (i, &u) in postorder.iter().enumerate() {
            dfn[u] = i;
        }
        let mut order = postorder;
        order.pop();
        order.reverse();
        let mut idom = vec![NONE; n];
        idom[start] = start;
        let mut assigned = Vec::new();
        let intersect = |idom: &[usize], mut u: usize, mut v: usize| {
            while u != v {
                while dfn[u] < dfn[v] {
                    u = idom[u];
                }
                while dfn[u] > dfn[v] {
                    v = idom[v];
                }
            }
            u
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &u in &order {
                let mut new_idom = NONE;
                for &v in &rindices[rindptr[u]..rindptr[u + 1]] {
                    if idom[v] != NONE {
                        new_idom = if new_idom == NONE {
                            v
                        } else {
                            intersect(&idom, new_idom, v)
                        };
                    }
                }
                if idom[u] != new_idom {
                    if idom[u] == NONE {
                        assigned.push(u);
                    }
                    idom[u] = new_idom;
                    changed = true;
                }
            }
        }
        let dominators = assigned.iter().map(|&u| idom[u]).collect();
        (assigned, dominators)
    }))
}

/// Whether a strongly connected graph is aperiodic (NetworkX's `is_aperiodic` after its
/// checks): the gcd of `level(u) - level(v) + 1` over its edges, with levels from a search
/// from `start`, is 1.
#[pyfunction]
pub fn nx_is_aperiodic(py: Python<'_>, adjacency: &PathAdjacency, start: usize) -> PyResult<bool> {
    let n = adjacency.len();
    if start >= n {
        return Err(PyValueError::new_err("start position out of range"));
    }
    Ok(py.allow_threads(|| {
        let (indptr, indices, _) = adjacency.rows();
        let mut level = vec![i64::MIN; n];
        level[start] = 0;
        let mut queue = VecDeque::from([start]);
        while let Some(u) = queue.pop_front() {
            for &v in &indices[indptr[u]..indptr[u + 1]] {
                if level[v] == i64::MIN {
                    level[v] = level[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        let gcd = |mut a: i64, mut b: i64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a.abs()
        };
        let mut g = 0i64;
        for u in 0..n {
            if level[u] == i64::MIN {
                continue;
            }
            for &v in &indices[indptr[u]..indptr[u + 1]] {
                g = gcd(g, level[u] - level[v] + 1);
                if g == 1 {
                    return true;
                }
            }
        }
        false
    }))
}
