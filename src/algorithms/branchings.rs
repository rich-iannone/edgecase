//! Trees and branchings: Edmonds' optimum branchings and spanning arborescences, the
//! matrix-tree theorem count of spanning trees, weighted uniform spanning tree sampling
//! (Wilson's algorithm), branching recognition and Prüfer sequences.
//!
//! Edge values come from the edge attribute named by `weight_attr`, else the edge weight
//! (1.0 when unset). An edge without a numeric value under `weight_attr` raises `ValueError`,
//! except that `"weight"` falls back to the weight field, as in `minimum_spanning_tree()`.

use crate::graph::{Edge, Graph, Node};
use crate::utils::Seed;
use nalgebra as na;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

fn sorted_ids(graph: &Graph) -> (Vec<u64>, FxHashMap<u64, usize>) {
    let mut ids = graph.get_node_ids();
    ids.sort_unstable();
    let idx = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    (ids, idx)
}

fn value_err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

fn edge_value(py: Python, e: &Edge, weight_attr: Option<&str>) -> PyResult<f64> {
    match weight_attr {
        None => Ok(e.weight.unwrap_or(1.0)),
        Some(attr) => crate::utils::require_named_edge_weight(py, e, attr, "weight_attr"),
    }
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// True when the graph's edges (direction ignored) connect all `n` indexed nodes.
fn spans(n: usize, pairs: impl Iterator<Item = (usize, usize)>) -> bool {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut parts = n;
    for (a, b) in pairs {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra] = rb;
            parts -= 1;
        }
    }
    parts <= 1
}

// ---------------------------------------------------------------------------------------------
// Edmonds' algorithm
// ---------------------------------------------------------------------------------------------

/// A branching weight ordered lexicographically: first the number `f` of forced (included)
/// edges, then the edge count `k`, then the total `w`. Optimum branchings use `k = 0` (pure
/// weight); spanning arborescences use `k = 1` per edge, so the maximum first takes as many
/// edges as possible (`n - 1` when a spanning arborescence exists) and only then the best
/// weight. Included edges carry `f = 1`, so they are all taken whenever some branching holds
/// them. Karp's contraction only needs an ordered group, so the triple works exactly, without
/// large additive constants.
#[derive(Clone, Copy, Debug)]
struct W {
    f: i64,
    k: i64,
    w: f64,
}

impl W {
    fn cmp(&self, other: &W) -> Ordering {
        self.f
            .cmp(&other.f)
            .then(self.k.cmp(&other.k))
            .then(self.w.partial_cmp(&other.w).unwrap_or(Ordering::Equal))
    }
    fn positive(&self) -> bool {
        self.cmp(&W { f: 0, k: 0, w: 0.0 }) == Ordering::Greater
    }
    fn shift(&self, minus: W, plus: W) -> W {
        W {
            f: self.f - minus.f + plus.f,
            k: self.k - minus.k + plus.k,
            w: self.w - minus.w + plus.w,
        }
    }
}

/// Maximum-weight branching (Edmonds 1967, with Karp's 1971 cycle contraction) on nodes
/// `0..n`. Returns indices into `edges`. Only positive-weight edges can be chosen; ties go
/// to the earliest edge.
fn max_branching(n: usize, edges: &[(usize, usize, W)]) -> Vec<usize> {
    // Heaviest positive in-edge of every node
    let mut best: Vec<Option<usize>> = vec![None; n];
    for (i, &(u, v, w)) in edges.iter().enumerate() {
        if u == v || !w.positive() {
            continue;
        }
        match best[v] {
            Some(j) if edges[j].2.cmp(&w) != Ordering::Less => {}
            _ => best[v] = Some(i),
        }
    }

    // Cycles of the chosen in-edges (each node has at most one, so each is a simple cycle)
    let mut state = vec![0u8; n]; // 0 new, 1 on the current walk, 2 done
    let mut cycle_of = vec![usize::MAX; n];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        let mut walk = Vec::new();
        let mut x = start;
        while state[x] == 0 {
            state[x] = 1;
            walk.push(x);
            match best[x] {
                Some(e) => x = edges[e].0,
                None => break,
            }
        }
        if state[x] == 1 && best[x].is_some() && walk.contains(&x) {
            let pos = walk.iter().position(|&y| y == x).unwrap();
            let cycle: Vec<usize> = walk[pos..].to_vec();
            for &y in &cycle {
                cycle_of[y] = cycles.len();
            }
            cycles.push(cycle);
        }
        for y in walk {
            state[y] = 2;
        }
    }
    if cycles.is_empty() {
        return best.into_iter().flatten().collect();
    }

    // Contract every cycle into one node
    let mut comp = vec![usize::MAX; n];
    let mut m = cycles.len();
    for (c, cycle) in cycles.iter().enumerate() {
        for &y in cycle {
            comp[y] = c;
        }
    }
    for x in 0..n {
        if comp[x] == usize::MAX {
            comp[x] = m;
            m += 1;
        }
    }
    let min_edge: Vec<usize> = cycles
        .iter()
        .map(|cycle| {
            let mut pick = best[cycle[0]].unwrap();
            for &y in &cycle[1..] {
                let e = best[y].unwrap();
                if edges[e].2.cmp(&edges[pick].2) == Ordering::Less {
                    pick = e;
                }
            }
            pick
        })
        .collect();
    let mut sub_edges = Vec::new();
    let mut origin = Vec::new();
    for (i, &(u, v, w)) in edges.iter().enumerate() {
        let (cu, cv) = (comp[u], comp[v]);
        if cu == cv {
            continue;
        }
        let w2 = match cycle_of[v] {
            usize::MAX => w,
            c => w.shift(edges[best[v].unwrap()].2, edges[min_edge[c]].2),
        };
        sub_edges.push((cu, cv, w2));
        origin.push(i);
    }

    let chosen: Vec<usize> = max_branching(m, &sub_edges)
        .into_iter()
        .map(|j| origin[j])
        .collect();
    let mut entered: Vec<Option<usize>> = vec![None; cycles.len()];
    for &e in &chosen {
        let v = edges[e].1;
        if cycle_of[v] != usize::MAX {
            entered[cycle_of[v]] = Some(v);
        }
    }
    let mut result = chosen;
    for (c, cycle) in cycles.iter().enumerate() {
        let drop = match entered[c] {
            Some(v) => best[v].unwrap(),
            None => min_edge[c],
        };
        result.extend(
            cycle
                .iter()
                .map(|&y| best[y].unwrap())
                .filter(|&e| e != drop),
        );
    }
    result
}

// ---------------------------------------------------------------------------------------------
// Edge partitions: included, excluded and open edges
// ---------------------------------------------------------------------------------------------

const OPEN: u8 = 0;
const INCLUDED: u8 = 1;
const EXCLUDED: u8 = 2;

/// The state of every edge (indexed like `edge_ids`) for the given included and excluded IDs.
fn partition_state(edge_ids: &[u64], include: &[u64], exclude: &[u64]) -> PyResult<Vec<u8>> {
    let pos: FxHashMap<u64, usize> = edge_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut state = vec![OPEN; edge_ids.len()];
    for (list, mark) in [(include, INCLUDED), (exclude, EXCLUDED)] {
        for id in list {
            let Some(&i) = pos.get(id) else {
                return value_err(format!("Edge with ID {id} not found in graph"));
            };
            if state[i] != OPEN && state[i] != mark {
                return value_err(format!("edge {id} is both included and excluded"));
            }
            state[i] = mark;
        }
    }
    Ok(state)
}

/// A pending partition in the Sörensen-Janssens enumeration: its optimum tree (edge indices)
/// with `key` (smaller is better) and `seq` (insertion order, to break ties).
struct Pending {
    key: f64,
    seq: u64,
    tree: Vec<usize>,
    state: Vec<u8>,
}

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pending {}
impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pending {
    // Reversed, so the max-heap `BinaryHeap` pops the smallest (key, seq) first
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .total_cmp(&self.key)
            .then(other.seq.cmp(&self.seq))
    }
}

/// Sörensen and Janssens (2005) partition scheme: after yielding the optimum tree of a
/// partition, split the rest of the partition on each of the tree's open edges in turn (the
/// edge excluded, then included for the following splits), solving each part with `solve`.
/// The best-first heap then yields every feasible tree exactly once, best first.
fn split_partition(
    heap: &mut BinaryHeap<Pending>,
    seq: &mut u64,
    entry: &Pending,
    mut solve: impl FnMut(&[u8]) -> Option<(f64, Vec<usize>)>,
) {
    let mut state = entry.state.clone();
    for &e in &entry.tree {
        if state[e] != OPEN {
            continue;
        }
        state[e] = EXCLUDED;
        if let Some((key, tree)) = solve(&state) {
            *seq += 1;
            heap.push(Pending {
                key,
                seq: *seq,
                tree,
                state: state.clone(),
            });
        }
        state[e] = INCLUDED;
    }
}

// ---------------------------------------------------------------------------------------------
// Optimum branchings and arborescences with constraints
// ---------------------------------------------------------------------------------------------

/// Node count, directed edges `(source index, target index, signed value)`, and edge IDs.
type BranchingEdges = (usize, Vec<(usize, usize, f64)>, Vec<u64>);

/// Directed edges `(source index, target index, signed value)` and their IDs, in storage
/// order. The value is negated for minimum problems, so every problem is a maximization.
fn branching_edges(
    py: Python,
    graph: &Graph,
    sign: f64,
    weight_attr: Option<&str>,
) -> PyResult<BranchingEdges> {
    if !graph.is_directed() {
        return value_err("branchings and arborescences require a directed graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let mut edges = Vec::new();
    let mut edge_ids = Vec::new();
    for e in graph.get_edges_internal() {
        let w = edge_value(py, e, weight_attr)?;
        if w.is_nan() {
            return value_err("edge values must not be NaN");
        }
        edges.push((idx[&e.source], idx[&e.target], sign * w));
        edge_ids.push(e.id);
    }
    Ok((ids.len(), edges, edge_ids))
}

/// The optimum branching (or spanning arborescence) that uses every included edge and no
/// excluded one, as sorted edge indices, or None when no such branching exists.
fn constrained_branching(
    n: usize,
    edges: &[(usize, usize, f64)],
    state: &[u8],
    spanning: bool,
) -> Option<Vec<usize>> {
    let mut sub = Vec::with_capacity(edges.len());
    let mut origin = Vec::with_capacity(edges.len());
    let mut forced = 0;
    for (i, &(u, v, w)) in edges.iter().enumerate() {
        if state[i] == EXCLUDED {
            continue;
        }
        let f = (state[i] == INCLUDED) as i64;
        forced += f as usize;
        sub.push((
            u,
            v,
            W {
                f,
                k: spanning as i64,
                w,
            },
        ));
        origin.push(i);
    }
    let mut chosen: Vec<usize> = max_branching(n, &sub)
        .into_iter()
        .map(|j| origin[j])
        .collect();
    if chosen.iter().filter(|&&i| state[i] == INCLUDED).count() != forced
        || (spanning && n > 0 && chosen.len() != n - 1)
    {
        return None;
    }
    chosen.sort_unstable();
    Some(chosen)
}

/// Optimum branching or spanning arborescence: the chosen edge IDs, in source edge order.
///
/// `kind` is one of `"max_branching"`, `"min_branching"`, `"max_arborescence"` and
/// `"min_arborescence"`. A minimum branching maximizes the negated weights, so it only uses
/// negative-weight edges (as in NetworkX); spanning arborescences raise when none exists.
/// The result uses every `include` edge and no `exclude` edge, or the call raises.
#[pyfunction]
#[pyo3(signature = (graph, kind, weight_attr=None, include=Vec::new(), exclude=Vec::new()))]
pub fn optimum_branching(
    py: Python,
    graph: &Graph,
    kind: &str,
    weight_attr: Option<&str>,
    include: Vec<u64>,
    exclude: Vec<u64>,
) -> PyResult<Vec<u64>> {
    if !graph.is_directed() {
        return value_err("branchings and arborescences require a directed graph");
    }
    let (spanning, sign) = match kind {
        "max_branching" => (false, 1.0),
        "min_branching" => (false, -1.0),
        "max_arborescence" => (true, 1.0),
        "min_arborescence" => (true, -1.0),
        _ => return value_err(format!("unknown branching kind {kind:?}")),
    };
    let (n, edges, edge_ids) = branching_edges(py, graph, sign, weight_attr)?;
    let state = partition_state(&edge_ids, &include, &exclude)?;
    match constrained_branching(n, &edges, &state, spanning) {
        Some(chosen) => Ok(chosen.into_iter().map(|i| edge_ids[i]).collect()),
        None if include.is_empty() && exclude.is_empty() => value_err(
            "no spanning arborescence exists: not every node can be reached from a single root",
        ),
        None => value_err(format!(
            "no {} uses every included edge and avoids every excluded one",
            if spanning {
                "spanning arborescence"
            } else {
                "branching"
            }
        )),
    }
}

/// Lazy enumeration of the spanning arborescences of a directed graph from best to worst
/// (Sörensen and Janssens' partitions, solved with Edmonds' algorithm), restricted to those
/// that use every included edge and no excluded one. Each item is a list of edge IDs.
#[pyclass(module = "edgecase._core")]
pub struct ArborescenceIter {
    n: usize,
    edges: Vec<(usize, usize, f64)>,
    edge_ids: Vec<u64>,
    heap: BinaryHeap<Pending>,
    seq: u64,
}

impl ArborescenceIter {
    fn solve(n: usize, edges: &[(usize, usize, f64)], state: &[u8]) -> Option<(f64, Vec<usize>)> {
        let tree = constrained_branching(n, edges, state, true)?;
        let total: f64 = tree.iter().map(|&i| edges[i].2).sum();
        Some((-total, tree))
    }
}

#[pymethods]
impl ArborescenceIter {
    #[new]
    #[pyo3(signature = (graph, minimum=true, weight_attr=None, include=Vec::new(), exclude=Vec::new()))]
    fn new(
        py: Python,
        graph: &Graph,
        minimum: bool,
        weight_attr: Option<&str>,
        include: Vec<u64>,
        exclude: Vec<u64>,
    ) -> PyResult<Self> {
        let sign = if minimum { -1.0 } else { 1.0 };
        let (n, edges, edge_ids) = branching_edges(py, graph, sign, weight_attr)?;
        let state = partition_state(&edge_ids, &include, &exclude)?;
        if constrained_branching(n, &edges, &vec![OPEN; edges.len()], true).is_none() {
            return value_err(
                "no spanning arborescence exists: not every node can be reached from a single root",
            );
        }
        let mut heap = BinaryHeap::new();
        if let Some((key, tree)) = Self::solve(n, &edges, &state) {
            heap.push(Pending {
                key,
                seq: 0,
                tree,
                state,
            });
        }
        Ok(ArborescenceIter {
            n,
            edges,
            edge_ids,
            heap,
            seq: 0,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Vec<u64>> {
        let entry = self.heap.pop()?;
        let (n, edges) = (self.n, &self.edges);
        split_partition(&mut self.heap, &mut self.seq, &entry, |state| {
            Self::solve(n, edges, state)
        });
        Some(entry.tree.iter().map(|&i| self.edge_ids[i]).collect())
    }
}

// ---------------------------------------------------------------------------------------------
// Spanning trees with constraints (Kruskal)
// ---------------------------------------------------------------------------------------------

/// Undirected view of the edges for Kruskal: index pairs, values, IDs (storage order), and the
/// edge indices sorted best first (stable, so ties keep storage order).
struct KruskalData {
    n: usize,
    pairs: Vec<(usize, usize)>,
    values: Vec<f64>,
    edge_ids: Vec<u64>,
    order: Vec<usize>,
}

fn kruskal_data(
    py: Python,
    graph: &Graph,
    minimum: bool,
    weight_attr: Option<&str>,
) -> PyResult<KruskalData> {
    let (ids, idx) = sorted_ids(graph);
    let mut pairs = Vec::new();
    let mut values = Vec::new();
    let mut edge_ids = Vec::new();
    for e in graph.get_edges_internal() {
        let w = edge_value(py, e, weight_attr)?;
        if w.is_nan() {
            return value_err("edge values must not be NaN");
        }
        pairs.push((idx[&e.source], idx[&e.target]));
        values.push(w);
        edge_ids.push(e.id);
    }
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    if minimum {
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    } else {
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    }
    Ok(KruskalData {
        n: ids.len(),
        pairs,
        values,
        edge_ids,
        order,
    })
}

/// Kruskal's algorithm on a partition: every included edge first (in storage order), then the
/// open edges best first, never an excluded one. Returns the chosen edge indices (a spanning
/// forest), or None when the included edges contain a cycle.
fn partition_kruskal(data: &KruskalData, state: &[u8]) -> Option<Vec<usize>> {
    let need = data.n.saturating_sub(1);
    let mut parent: Vec<usize> = (0..data.n).collect();
    let mut tree = Vec::with_capacity(need);
    let included = (0..state.len()).filter(|&i| state[i] == INCLUDED);
    let open = data.order.iter().copied().filter(|&i| state[i] == OPEN);
    for (i, forced) in included.map(|i| (i, true)).chain(open.map(|i| (i, false))) {
        if !forced && tree.len() == need {
            break;
        }
        let (a, b) = data.pairs[i];
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra == rb {
            if forced {
                return None;
            }
            continue;
        }
        parent[ra] = rb;
        tree.push(i);
    }
    Some(tree)
}

/// Minimum (or maximum) spanning forest that uses every `include` edge and no `exclude` edge
/// (NetworkX's `partition_spanning_tree`), direction ignored. Returns edge IDs, included edges
/// first, then in the order Kruskal picked them.
#[pyfunction]
#[pyo3(signature = (graph, minimum=true, weight_attr=None, include=Vec::new(), exclude=Vec::new()))]
pub fn partition_spanning_tree(
    py: Python,
    graph: &Graph,
    minimum: bool,
    weight_attr: Option<&str>,
    include: Vec<u64>,
    exclude: Vec<u64>,
) -> PyResult<Vec<u64>> {
    let data = kruskal_data(py, graph, minimum, weight_attr)?;
    let state = partition_state(&data.edge_ids, &include, &exclude)?;
    match partition_kruskal(&data, &state) {
        Some(tree) => Ok(tree.into_iter().map(|i| data.edge_ids[i]).collect()),
        None => value_err("the included edges contain a cycle"),
    }
}

/// Lazy enumeration of the spanning trees of a connected graph (direction ignored) from
/// lightest to heaviest, or heaviest to lightest (Sörensen and Janssens' partitions, solved
/// with Kruskal's algorithm), restricted to those that use every included edge and no excluded
/// one. Each item is a list of edge IDs.
#[pyclass(module = "edgecase._core")]
pub struct SpanningTreeIter {
    data: KruskalData,
    minimum: bool,
    heap: BinaryHeap<Pending>,
    seq: u64,
}

impl SpanningTreeIter {
    fn solve(data: &KruskalData, minimum: bool, state: &[u8]) -> Option<(f64, Vec<usize>)> {
        let tree = partition_kruskal(data, state)?;
        if tree.len() != data.n.saturating_sub(1) {
            return None;
        }
        let total: f64 = tree.iter().map(|&i| data.values[i]).sum();
        Some((if minimum { total } else { -total }, tree))
    }
}

#[pymethods]
impl SpanningTreeIter {
    #[new]
    #[pyo3(signature = (graph, minimum=true, weight_attr=None, include=Vec::new(), exclude=Vec::new()))]
    fn new(
        py: Python,
        graph: &Graph,
        minimum: bool,
        weight_attr: Option<&str>,
        include: Vec<u64>,
        exclude: Vec<u64>,
    ) -> PyResult<Self> {
        let data = kruskal_data(py, graph, minimum, weight_attr)?;
        let state = partition_state(&data.edge_ids, &include, &exclude)?;
        if !spans(data.n, data.pairs.iter().copied()) {
            return value_err("the graph is disconnected, so it has no spanning tree");
        }
        let mut heap = BinaryHeap::new();
        if let Some((key, tree)) = Self::solve(&data, minimum, &state) {
            heap.push(Pending {
                key,
                seq: 0,
                tree,
                state,
            });
        }
        Ok(SpanningTreeIter {
            data,
            minimum,
            heap,
            seq: 0,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<Vec<u64>> {
        let entry = self.heap.pop()?;
        let (data, minimum) = (&self.data, self.minimum);
        split_partition(&mut self.heap, &mut self.seq, &entry, |state| {
            Self::solve(data, minimum, state)
        });
        Some(entry.tree.iter().map(|&i| self.data.edge_ids[i]).collect())
    }
}

/// The values of the given edges (see the module notes), in the given order.
#[pyfunction]
#[pyo3(signature = (graph, edge_ids, weight_attr=None))]
pub fn edge_values(
    py: Python,
    graph: &Graph,
    edge_ids: Vec<u64>,
    weight_attr: Option<&str>,
) -> PyResult<Vec<f64>> {
    let by_id: FxHashMap<u64, &Edge> = graph
        .get_edges_internal()
        .iter()
        .map(|e| (e.id, e))
        .collect();
    edge_ids
        .iter()
        .map(|eid| match by_id.get(eid) {
            Some(e) => edge_value(py, e, weight_attr),
            None => value_err(format!("Edge with ID {eid} not found in graph")),
        })
        .collect()
}

/// A graph with every node of `graph` (or only `node_ids`, if given) and only the given edges,
/// with the same directedness and weighting. Node and edge IDs, labels, weights and (copied)
/// attributes are all kept.
#[pyfunction]
#[pyo3(signature = (graph, edge_ids, node_ids=None))]
pub fn spanning_subgraph(
    py: Python,
    graph: &Graph,
    edge_ids: Vec<u64>,
    node_ids: Option<Vec<u64>>,
) -> PyResult<Graph> {
    let mut result = Graph::new_rust(graph.is_directed(), graph.is_weighted());
    let ids = match node_ids {
        Some(mut ids) => {
            ids.sort_unstable();
            ids.dedup();
            ids
        }
        None => sorted_ids(graph).0,
    };
    let nodes = graph.get_nodes();
    for id in ids {
        let Some(node) = nodes.get(&id) else {
            return value_err(format!("Node with ID {id} not found in graph"));
        };
        let attributes = match &node.attributes {
            Some(d) => Some(d.bind(py).copy()?.unbind()),
            None => None,
        };
        result.add_node_rust(Node {
            id,
            label: node.label.clone(),
            attributes,
        })?;
    }
    let by_id: FxHashMap<u64, &Edge> = graph
        .get_edges_internal()
        .iter()
        .map(|e| (e.id, e))
        .collect();
    for eid in edge_ids {
        let Some(e) = by_id.get(&eid) else {
            return value_err(format!("Edge with ID {eid} not found in graph"));
        };
        let attributes = match &e.attributes {
            Some(d) => Some(d.bind(py).copy()?.unbind()),
            None => None,
        };
        result.add_edge_rust(Edge {
            id: eid,
            source: e.source,
            target: e.target,
            weight: e.weight,
            attributes,
        })?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------------------------
// Counting and sampling spanning trees
// ---------------------------------------------------------------------------------------------

/// Exact determinant of an integer matrix by Bareiss' fraction-free elimination, or None on
/// i128 overflow.
fn bareiss(mut a: Vec<Vec<i128>>) -> Option<i128> {
    let n = a.len();
    let mut sign = 1i128;
    let mut prev = 1i128;
    for k in 0..n {
        if a[k][k] == 0 {
            let Some(swap) = (k + 1..n).find(|&r| a[r][k] != 0) else {
                return Some(0);
            };
            a.swap(k, swap);
            sign = -sign;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                let x = a[i][j].checked_mul(a[k][k])?;
                let y = a[i][k].checked_mul(a[k][j])?;
                a[i][j] = x.checked_sub(y)? / prev;
            }
        }
        prev = a[k][k];
    }
    if n == 0 {
        return Some(1);
    }
    Some(sign * a[n - 1][n - 1])
}

/// Number of spanning trees (undirected) or of spanning arborescences rooted at `root` and
/// directed away from it, by Kirchhoff's matrix-tree theorem. With `weighted`, each tree
/// counts as the product of its edge weights. Unweighted counts are exact integers.
#[pyfunction]
#[pyo3(signature = (graph, root=None, weighted=false))]
pub fn number_of_spanning_trees(
    py: Python,
    graph: &Graph,
    root: Option<u64>,
    weighted: bool,
) -> PyResult<PyObject> {
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n == 0 {
        return value_err("the graph must contain at least one node");
    }
    let directed = graph.is_directed();
    let skip = match (directed, root) {
        (true, None) => return value_err("root= must be given for a directed graph"),
        (true, Some(r)) => match idx.get(&r) {
            Some(&i) => i,
            None => return value_err(format!("Root node {r} does not exist")),
        },
        (false, _) => 0,
    };
    let edges: Vec<(usize, usize, f64)> = graph
        .get_edges_internal()
        .iter()
        .filter(|e| e.source != e.target)
        .map(|e| {
            (
                idx[&e.source],
                idx[&e.target],
                if weighted {
                    e.weight.unwrap_or(1.0)
                } else {
                    1.0
                },
            )
        })
        .collect();
    if !spans(n, edges.iter().map(|&(a, b, _)| (a, b))) {
        return Ok(if weighted {
            0.0f64.into_pyobject(py)?.into_any().unbind()
        } else {
            0i64.into_pyobject(py)?.into_any().unbind()
        });
    }
    // Laplacian (out-degree convention for undirected, in-degree for arborescences)
    let mut lap = vec![vec![0.0f64; n]; n];
    for &(a, b, w) in &edges {
        lap[a][b] -= w;
        lap[b][b] += w;
        if !directed {
            lap[b][a] -= w;
            lap[a][a] += w;
        }
    }
    let keep: Vec<usize> = (0..n).filter(|&i| i != skip).collect();
    let m = keep.len();
    if !weighted {
        let minor: Vec<Vec<i128>> = keep
            .iter()
            .map(|&i| keep.iter().map(|&j| lap[i][j] as i128).collect())
            .collect();
        if let Some(det) = bareiss(minor.clone()) {
            return Ok(det.into_pyobject(py)?.into_any().unbind());
        }
        // Too large for 128 bits: residues modulo enough 61-bit primes to pin the value down
        // (Hadamard's bound), combined into an exact integer by the Python wrapper (CRT)
        let bits: f64 = minor
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| (x as f64) * (x as f64))
                    .sum::<f64>()
                    .sqrt()
                    .max(1.0)
                    .log2()
            })
            .sum();
        let count = (bits / 60.0).ceil() as usize + 2;
        let mut residues = Vec::with_capacity(count);
        let mut p = (1u64 << 61) - 1;
        while residues.len() < count {
            if is_prime(p) {
                residues.push((p, det_mod(&minor, p)));
            }
            p -= 2;
        }
        return Ok(residues.into_pyobject(py)?.into_any().unbind());
    }
    let det = na::DMatrix::from_fn(m, m, |i, j| lap[keep[i]][keep[j]]).determinant();
    Ok(det.into_pyobject(py)?.into_any().unbind())
}

fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn pow_mod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = mul_mod(r, a, p);
        }
        a = mul_mod(a, a, p);
        e >>= 1;
    }
    r
}

/// Deterministic Miller-Rabin for 64-bit integers.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let bases = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &b in &bases {
        if n % b == 0 {
            return n == b;
        }
    }
    let (mut d, mut r) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    'outer: for &a in &bases {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..r {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Determinant of an integer matrix modulo the prime `p` (Gaussian elimination).
fn det_mod(a: &[Vec<i128>], p: u64) -> u64 {
    let n = a.len();
    let mut m: Vec<Vec<u64>> = a
        .iter()
        .map(|row| {
            row.iter()
                .map(|&x| x.rem_euclid(p as i128) as u64)
                .collect()
        })
        .collect();
    let mut det = 1u64;
    for k in 0..n {
        let Some(piv) = (k..n).find(|&r| m[r][k] != 0) else {
            return 0;
        };
        if piv != k {
            m.swap(piv, k);
            det = p - det;
        }
        det = mul_mod(det, m[k][k], p);
        let inv = pow_mod(m[k][k], p - 2, p);
        for i in k + 1..n {
            if m[i][k] == 0 {
                continue;
            }
            let f = mul_mod(m[i][k], inv, p);
            for j in k..n {
                let sub = mul_mod(f, m[k][j], p);
                m[i][j] = (m[i][j] + p - sub) % p;
            }
        }
    }
    det % p
}

/// A random spanning tree of a connected undirected graph, chosen with probability
/// proportional to the product of its edge weights (uniformly when unweighted), by
/// Wilson's loop-erased random walk algorithm. Returns edge IDs.
#[pyfunction]
#[pyo3(signature = (graph, weighted=false, seed=None))]
pub fn random_spanning_tree(
    graph: &Graph,
    weighted: bool,
    seed: Option<Seed>,
) -> PyResult<Vec<u64>> {
    let seed = seed.map(u64::from);
    if graph.is_directed() {
        return value_err("random_spanning_tree() requires an undirected graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    // adjacency: (neighbor, edge id, cumulative weight)
    let mut adj: Vec<Vec<(usize, u64, f64)>> = vec![Vec::new(); n];
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let w = if weighted {
            e.weight.unwrap_or(1.0)
        } else {
            1.0
        };
        if !(0.0..f64::INFINITY).contains(&w) {
            // An infinite weight swamps the walk's cumulative sampling, so it never ends
            return value_err("edge weights must be finite and non-negative");
        }
        if w == 0.0 {
            continue;
        }
        let (a, b) = (idx[&e.source], idx[&e.target]);
        adj[a].push((b, e.id, w));
        adj[b].push((a, e.id, w));
    }
    if n == 0
        || !spans(
            n,
            adj.iter()
                .enumerate()
                .flat_map(|(a, l)| l.iter().map(move |&(b, _, _)| (a, b))),
        )
    {
        return value_err("the graph must be non-empty and connected (positive-weight edges)");
    }
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    Ok(wilson(adj, &mut rng))
}

/// Wilson's algorithm on an adjacency list of `(neighbor, edge id, weight)` entries (the
/// graph must be connected): a spanning tree drawn with probability proportional to the
/// product of its edge weights, as edge IDs.
fn wilson(mut adj: Vec<Vec<(usize, u64, f64)>>, rng: &mut ChaCha8Rng) -> Vec<u64> {
    let n = adj.len();
    for list in adj.iter_mut() {
        let mut total = 0.0;
        for item in list.iter_mut() {
            total += item.2;
            item.2 = total;
        }
    }
    let mut in_tree = vec![false; n];
    let mut next: Vec<(usize, u64)> = vec![(0, 0); n];
    in_tree[rng.gen_range(0..n)] = true;
    let mut tree = Vec::with_capacity(n.saturating_sub(1));
    for start in 0..n {
        let mut u = start;
        while !in_tree[u] {
            let list = &adj[u];
            let r = rng.gen::<f64>() * list[list.len() - 1].2;
            let k = list.partition_point(|item| item.2 <= r).min(list.len() - 1);
            next[u] = (list[k].0, list[k].1);
            u = list[k].0;
        }
        let mut u = start;
        while !in_tree[u] {
            in_tree[u] = true;
            tree.push(next[u].1);
            u = next[u].0;
        }
    }
    tree
}

/// A random spanning tree of a connected undirected graph, drawn with probability
/// proportional to the *sum* of its edge weights (NetworkX's additive distribution).
///
/// The distribution is an exact mixture: since `P(T) ∝ Σ_{e ∈ T} w(e)`, first pick one edge
/// `e` with probability proportional to `w(e) τ(G / e)`, the weight of the trees through `e`,
/// then a uniform spanning tree of `G` with `e` contracted, plus `e`. By Kirchhoff,
/// `τ(G / e) / τ(G)` is the effective resistance `R(e)` between the ends of `e` with unit
/// resistors, read off the inverse of the grounded Laplacian. Returns edge IDs.
#[pyfunction]
#[pyo3(signature = (graph, seed=None))]
pub fn random_spanning_tree_additive(graph: &Graph, seed: Option<Seed>) -> PyResult<Vec<u64>> {
    let seed = seed.map(u64::from);
    if graph.is_directed() {
        return value_err("random_spanning_tree() requires an undirected graph");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    let mut edges: Vec<(usize, usize, u64, f64)> = Vec::new();
    for e in graph.get_edges_internal() {
        if e.source == e.target {
            continue;
        }
        let w = e.weight.unwrap_or(1.0);
        if w.is_nan() || w < 0.0 {
            return value_err("edge weights must be non-negative");
        }
        edges.push((idx[&e.source], idx[&e.target], e.id, w));
    }
    if n == 0 || !spans(n, edges.iter().map(|&(a, b, _, _)| (a, b))) {
        return value_err("the graph must be non-empty and connected");
    }
    if n == 1 {
        return Ok(Vec::new());
    }

    // Effective resistances from the Laplacian grounded at the last node
    let m = n - 1;
    let mut lap = na::DMatrix::<f64>::zeros(m, m);
    for &(a, b, _, _) in &edges {
        if a < m {
            lap[(a, a)] += 1.0;
        }
        if b < m {
            lap[(b, b)] += 1.0;
        }
        if a < m && b < m {
            lap[(a, b)] -= 1.0;
            lap[(b, a)] -= 1.0;
        }
    }
    let Some(chol) = lap.cholesky() else {
        return value_err("the Laplacian of the graph could not be factorized");
    };
    let inv = chol.inverse();
    let at = |a: usize, b: usize| if a < m && b < m { inv[(a, b)] } else { 0.0 };
    let mut cumulative = Vec::with_capacity(edges.len());
    let mut total = 0.0;
    for &(a, b, _, w) in &edges {
        let r = (at(a, a) + at(b, b) - 2.0 * at(a, b)).max(0.0);
        total += w * r;
        cumulative.push(total);
    }
    if total.is_nan() || total <= 0.0 {
        return value_err("every spanning tree has total weight zero");
    }

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let r = rng.gen::<f64>() * total;
    let pick = cumulative.partition_point(|&c| c <= r).min(edges.len() - 1);
    let (pa, pb, pid, _) = edges[pick];

    // Uniform tree of the graph with the picked edge contracted (pb merged into pa); copies
    // of the picked edge become self-loops and drop out
    let merged = |x: usize| {
        let x = if x == pb { pa } else { x };
        if x > pb {
            x - 1
        } else {
            x
        }
    };
    let mut adj: Vec<Vec<(usize, u64, f64)>> = vec![Vec::new(); n - 1];
    for &(a, b, id, _) in &edges {
        let (a, b) = (merged(a), merged(b));
        if a != b {
            adj[a].push((b, id, 1.0));
            adj[b].push((a, id, 1.0));
        }
    }
    let mut tree = vec![pid];
    tree.extend(wilson(adj, &mut rng));
    Ok(tree)
}

// ---------------------------------------------------------------------------------------------
// Joining rooted trees
// ---------------------------------------------------------------------------------------------

/// A new rooted tree: a fresh root joined to the root of every given tree (NetworkX's
/// `join_trees`). The root gets ID 1 and the copied nodes the following IDs, tree by tree in
/// ascending ID order; node `i` is labeled `first_label + i - 1`. Attributes and weights are
/// copied, and with `label_attribute` the old label (or ID, when unlabeled) of every copied
/// node is stored under that attribute. Edges to the subtree roots point away from the root.
#[pyfunction]
#[pyo3(signature = (trees, roots, label_attribute=None, first_label=1))]
pub fn join_trees(
    py: Python,
    trees: Vec<PyRef<Graph>>,
    roots: Vec<u64>,
    label_attribute: Option<&str>,
    first_label: i64,
) -> PyResult<Graph> {
    let (directed, weighted) = match trees.first() {
        Some(t) => (t.is_directed(), t.is_weighted()),
        None => (false, false),
    };
    if trees.len() != roots.len() {
        return value_err("join_trees() needs one root per tree");
    }
    let mut result = Graph::new_rust(directed, weighted);
    let label = |id: u64| Some((first_label + id as i64 - 1).to_string());
    result.add_node_rust(Node {
        id: 1,
        label: label(1),
        attributes: None,
    })?;
    let mut next = 2u64;
    let mut new_roots = Vec::with_capacity(trees.len());
    for (tree, &root) in trees.iter().zip(&roots) {
        if tree.is_directed() != directed {
            return value_err("the trees must all be directed or all undirected");
        }
        let (ids, idx) = sorted_ids(tree);
        let Some(&root_pos) = idx.get(&root) else {
            return value_err(format!("Node with ID {root} not found in graph"));
        };
        new_roots.push(next + root_pos as u64);
        let nodes = tree.get_nodes();
        for &old in &ids {
            let node = &nodes[&old];
            let id = next + idx[&old] as u64;
            let attributes = match (&node.attributes, label_attribute) {
                (attrs, None) => crate::graph::copy_attrs(py, attrs),
                (attrs, Some(name)) => {
                    let dict = match attrs {
                        Some(d) => d.bind(py).copy()?,
                        None => PyDict::new(py),
                    };
                    match &node.label {
                        Some(l) => dict.set_item(name, l)?,
                        None => dict.set_item(name, old)?,
                    }
                    Some(dict.unbind())
                }
            };
            result.add_node_rust(Node {
                id,
                label: label(id),
                attributes,
            })?;
        }
        for e in tree.get_edges_internal() {
            result.add_edge_rust(Edge {
                id: 0,
                source: next + idx[&e.source] as u64,
                target: next + idx[&e.target] as u64,
                weight: e.weight,
                attributes: crate::graph::copy_attrs(py, &e.attributes),
            })?;
        }
        next += ids.len() as u64;
    }
    for target in new_roots {
        let weight = if weighted { Some(1.0) } else { None };
        result.add_edge_rust(Edge {
            id: 0,
            source: 1,
            target,
            weight,
            attributes: None,
        })?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------------------------
// Recognition and Prüfer sequences
// ---------------------------------------------------------------------------------------------

/// True for a directed forest in which every node has at most one incoming edge.
#[pyfunction]
pub fn is_branching(graph: &Graph) -> bool {
    if !graph.is_directed() || graph.get_node_ids().is_empty() {
        return false;
    }
    let (_, idx) = sorted_ids(graph);
    let n = idx.len();
    let mut indeg = vec![0usize; n];
    let mut parent: Vec<usize> = (0..n).collect();
    for e in graph.get_edges_internal() {
        let (a, b) = (idx[&e.source], idx[&e.target]);
        indeg[b] += 1;
        if indeg[b] > 1 {
            return false;
        }
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra == rb {
            return false;
        }
        parent[ra] = rb;
    }
    true
}

/// Prüfer sequence of an undirected tree, ranking nodes by ascending ID. Returns node IDs.
#[pyfunction]
pub fn to_prufer_sequence(graph: &Graph) -> PyResult<Vec<u64>> {
    if graph.is_directed() {
        return value_err("to_prufer_sequence() requires an undirected tree");
    }
    let (ids, idx) = sorted_ids(graph);
    let n = ids.len();
    if n < 2 {
        return value_err("the Prüfer sequence is undefined for trees with fewer than two nodes");
    }
    let edges = graph.get_edges_internal();
    let pairs: Vec<(usize, usize)> = edges
        .iter()
        .map(|e| (idx[&e.source], idx[&e.target]))
        .collect();
    if pairs.len() != n - 1
        || pairs.iter().any(|&(a, b)| a == b)
        || !spans(n, pairs.iter().copied())
    {
        return value_err("the graph is not a tree");
    }
    let mut adj = vec![Vec::new(); n];
    let mut degree = vec![0usize; n];
    for &(a, b) in &pairs {
        adj[a].push(b);
        adj[b].push(a);
        degree[a] += 1;
        degree[b] += 1;
    }
    let mut removed = vec![false; n];
    let mut index = (0..n).find(|&k| degree[k] == 1).unwrap();
    let mut u = index;
    let mut result = Vec::with_capacity(n - 2);
    for _ in 0..n - 2 {
        let v = *adj[u].iter().find(|&&v| !removed[v]).unwrap();
        removed[u] = true;
        result.push(ids[v]);
        degree[v] -= 1;
        if v < index && degree[v] == 1 {
            u = v;
        } else {
            index = (index + 1..n)
                .find(|&k| degree[k] == 1 && !removed[k])
                .unwrap();
            u = index;
        }
    }
    Ok(result)
}

/// The tree on nodes `0..len + 2` encoded by a Prüfer sequence, as edge pairs.
#[pyfunction]
pub fn from_prufer_sequence(sequence: Vec<i64>) -> PyResult<Vec<(usize, usize)>> {
    let n = sequence.len() + 2;
    let mut degree = vec![1usize; n];
    for &v in &sequence {
        if v < 0 || v as usize >= n {
            return value_err(format!(
                "invalid Prüfer sequence: values must be between 0 and {}, got {v}",
                n - 1
            ));
        }
        degree[v as usize] += 1;
    }
    let mut edges = Vec::with_capacity(n - 1);
    let mut index = (0..n).find(|&k| degree[k] == 1).unwrap();
    let mut u = index;
    for &v in &sequence {
        let v = v as usize;
        edges.push((u, v));
        degree[u] = 0;
        degree[v] -= 1;
        if v < index && degree[v] == 1 {
            u = v;
        } else {
            index = (index + 1..n).find(|&k| degree[k] == 1).unwrap();
            u = index;
        }
    }
    degree[u] = 0;
    let last: Vec<usize> = (0..n).filter(|&k| degree[k] == 1).collect();
    edges.push((u, last[0]));
    Ok(edges)
}
