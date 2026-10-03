//! Community detection algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Detect communities using Louvain algorithm (placeholder)
#[pyfunction]
pub fn detect_communities(graph: &Graph) -> PyResult<FxHashMap<u64, usize>> {
    let mut communities = FxHashMap::default();

    for (idx, &node_id) in graph.get_node_ids().iter().enumerate() {
        communities.insert(node_id, idx);
    }

    Ok(communities)
}

/// Girvan-Newman community detection with per-component optimization.
///
/// Returns a list of community partitions (each partition is a list of
/// sorted node-ID vectors). The first entry is the partition that achieved
/// the highest modularity; subsequent entries are the remaining levels of
/// the dendrogram in removal order.
///
/// Optimizations over the naive O(m^2 n) approach:
/// - Edge betweenness recomputed only for the affected component
/// - Parallel BFS sources via Rayon within each betweenness pass
/// - Incremental component tracking avoids full-graph BFS each iteration
#[pyfunction]
#[pyo3(signature = (graph, most_valuable_edge=None))]
pub fn girvan_newman(
    graph: &Graph,
    most_valuable_edge: Option<String>,
) -> PyResult<Vec<Vec<Vec<u64>>>> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Girvan-Newman requires an undirected graph",
        ));
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(vec![]);
    }

    let node_ids = graph.get_node_ids();

    if n == 1 {
        return Ok(vec![vec![node_ids]]);
    }

    if graph.num_edges() == 0 {
        let singletons: Vec<Vec<u64>> = node_ids.iter().map(|&id| vec![id]).collect();
        return Ok(vec![singletons]);
    }

    let use_weight = most_valuable_edge.as_deref() == Some("weight");

    // Snapshot of the original graph for modularity evaluation
    let orig_adj: FxHashMap<u64, FxHashSet<u64>> = graph
        .get_adjacency_list()
        .iter()
        .map(|(&k, v)| (k, v.iter().copied().collect()))
        .collect();
    let orig_degrees: FxHashMap<u64, f64> = node_ids
        .iter()
        .map(|&id| {
            let deg = orig_adj.get(&id).map(|v| v.len()).unwrap_or(0) as f64;
            (id, deg)
        })
        .collect();
    let m_orig = graph.num_edges() as f64;

    // Working adjacency (edges get removed from this)
    let mut adj: FxHashMap<u64, FxHashSet<u64>> = orig_adj.clone();

    // Weight lookup for weighted variant
    let mut weights: FxHashMap<(u64, u64), f64> = FxHashMap::default();
    if use_weight {
        for edge in graph.get_edges_internal() {
            let key = canonical(edge.source, edge.target);
            let w = edge.weight.unwrap_or(1.0);
            weights.insert(key, w);
        }
    }

    // Component tracking
    let mut node_component: FxHashMap<u64, usize> = FxHashMap::default();
    let mut components: FxHashMap<usize, FxHashSet<u64>> = FxHashMap::default();
    {
        let comps = find_components_from_adj(&adj);
        for (comp_id, members) in comps.into_iter().enumerate() {
            for &nid in &members {
                node_component.insert(nid, comp_id);
            }
            components.insert(comp_id, members);
        }
    }
    let mut next_comp_id = components.len();

    let mut edge_count: usize = adj.values().map(|ns| ns.len()).sum::<usize>() / 2;

    let mut best_modularity =
        compute_modularity(&orig_adj, &orig_degrees, m_orig, &components);
    let mut best_partition = partition_from_components(&components);

    // Per-component edge betweenness cache
    let mut comp_betweenness: FxHashMap<usize, FxHashMap<(u64, u64), f64>> =
        FxHashMap::default();
    for (&comp_id, members) in &components {
        if members.len() > 1 {
            let eb = edge_betweenness_for_component(members, &adj, use_weight, &weights);
            comp_betweenness.insert(comp_id, eb);
        }
    }

    let mut dendrogram: Vec<Vec<Vec<u64>>> = Vec::new();
    dendrogram.push(best_partition.clone());

    let mut prev_num_components = components.len();

    for _iter in 0..graph.num_edges() {
        if edge_count == 0 {
            break;
        }

        // Find edge with max betweenness across all components. Ties (within a
        // relative 1e-9, since the parallel sums carry rounding noise) go to the
        // smallest canonical edge, so the dendrogram doesn't depend on hash-map
        // iteration order.
        let mut max_val = f64::NEG_INFINITY;
        let mut max_edge: Option<(u64, u64)> = None;

        for eb in comp_betweenness.values() {
            for (&edge, &val) in eb {
                let tol = 1e-9 * max_val.abs().max(1.0);
                let better = val > max_val + tol
                    || ((val - max_val).abs() <= tol && Some(edge) < max_edge);
                if max_edge.is_none() || better {
                    if val > max_val {
                        max_val = val;
                    }
                    max_edge = Some(edge);
                }
            }
        }

        let (u, v) = match max_edge {
            Some(e) => e,
            None => break,
        };

        // Remove edge from working graph
        if let Some(ns) = adj.get_mut(&u) {
            ns.remove(&v);
        }
        if let Some(ns) = adj.get_mut(&v) {
            ns.remove(&u);
        }
        edge_count -= 1;

        // Which component was affected?
        let affected_comp = node_component[&u];
        let affected_members = match components.get(&affected_comp) {
            Some(m) => m.clone(),
            None => continue,
        };

        // Did the removal split the component?
        let sub_components = find_components_in_set(&affected_members, &adj);

        if sub_components.len() > 1 {
            components.remove(&affected_comp);
            comp_betweenness.remove(&affected_comp);

            for new_members in sub_components {
                let cid = next_comp_id;
                next_comp_id += 1;
                for &nid in &new_members {
                    node_component.insert(nid, cid);
                }
                if new_members.len() > 1 {
                    let eb = edge_betweenness_for_component(
                        &new_members,
                        &adj,
                        use_weight,
                        &weights,
                    );
                    comp_betweenness.insert(cid, eb);
                }
                components.insert(cid, new_members);
            }
        } else {
            // Same component, recompute its betweenness
            if affected_members.len() > 1 {
                let eb = edge_betweenness_for_component(
                    &affected_members,
                    &adj,
                    use_weight,
                    &weights,
                );
                comp_betweenness.insert(affected_comp, eb);
            } else {
                comp_betweenness.remove(&affected_comp);
            }
        }

        let current_num_components = components.len();
        if current_num_components != prev_num_components {
            prev_num_components = current_num_components;

            let partition = partition_from_components(&components);
            let modularity =
                compute_modularity(&orig_adj, &orig_degrees, m_orig, &components);

            if modularity > best_modularity {
                best_modularity = modularity;
                best_partition = partition.clone();
            }

            dendrogram.push(partition);
        }
    }

    // Best-modularity partition goes first
    if let Some(pos) = dendrogram.iter().position(|p| p == &best_partition) {
        if pos != 0 {
            dendrogram.swap(0, pos);
        }
    }

    Ok(dendrogram)
}

#[inline]
fn canonical(a: u64, b: u64) -> (u64, u64) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn find_components_from_adj(adj: &FxHashMap<u64, FxHashSet<u64>>) -> Vec<FxHashSet<u64>> {
    let mut visited = FxHashSet::default();
    let mut result = Vec::new();
    let mut queue = VecDeque::new();

    for &start in adj.keys() {
        if visited.contains(&start) {
            continue;
        }
        let mut component = FxHashSet::default();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            component.insert(node);
            if let Some(ns) = adj.get(&node) {
                for &nbr in ns {
                    if visited.insert(nbr) {
                        queue.push_back(nbr);
                    }
                }
            }
        }
        result.push(component);
    }
    result
}

fn find_components_in_set(
    members: &FxHashSet<u64>,
    adj: &FxHashMap<u64, FxHashSet<u64>>,
) -> Vec<FxHashSet<u64>> {
    let mut visited = FxHashSet::default();
    let mut result = Vec::new();
    let mut queue = VecDeque::new();

    for &start in members {
        if visited.contains(&start) {
            continue;
        }
        let mut component = FxHashSet::default();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            component.insert(node);
            if let Some(ns) = adj.get(&node) {
                for &nbr in ns {
                    if members.contains(&nbr) && visited.insert(nbr) {
                        queue.push_back(nbr);
                    }
                }
            }
        }
        result.push(component);
    }
    result
}

/// Brandes' algorithm for edge betweenness within a single component, parallelized over sources.
fn edge_betweenness_for_component(
    members: &FxHashSet<u64>,
    adj: &FxHashMap<u64, FxHashSet<u64>>,
    use_weight: bool,
    weights: &FxHashMap<(u64, u64), f64>,
) -> FxHashMap<(u64, u64), f64> {
    use rayon::prelude::*;

    let member_vec: Vec<u64> = members.iter().copied().collect();
    if member_vec.len() <= 1 {
        return FxHashMap::default();
    }

    let partial_results: Vec<FxHashMap<(u64, u64), f64>> = member_vec
        .par_iter()
        .map(|&source| {
            brandes_single_source(source, members, adj, use_weight, weights)
        })
        .collect();

    let mut betweenness: FxHashMap<(u64, u64), f64> = FxHashMap::default();
    for partial in partial_results {
        for (edge, val) in partial {
            *betweenness.entry(edge).or_insert(0.0) += val;
        }
    }

    // Undirected: each pair counted from both endpoints
    for val in betweenness.values_mut() {
        *val /= 2.0;
    }

    betweenness
}

fn brandes_single_source(
    source: u64,
    members: &FxHashSet<u64>,
    adj: &FxHashMap<u64, FxHashSet<u64>>,
    use_weight: bool,
    weights: &FxHashMap<(u64, u64), f64>,
) -> FxHashMap<(u64, u64), f64> {
    let n = members.len();
    let mut stack: Vec<u64> = Vec::with_capacity(n);
    let mut predecessors: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    let mut num_paths: FxHashMap<u64, f64> = FxHashMap::default();
    let mut dist: FxHashMap<u64, f64> = FxHashMap::default();

    num_paths.insert(source, 1.0);
    dist.insert(source, 0.0);

    if use_weight {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(std::cmp::Reverse(OrdF64Pair(0.0, source)));

        while let Some(std::cmp::Reverse(OrdF64Pair(d, node))) = heap.pop() {
            if d > *dist.get(&node).unwrap_or(&f64::INFINITY) {
                continue;
            }
            stack.push(node);

            if let Some(ns) = adj.get(&node) {
                for &nbr in ns {
                    if !members.contains(&nbr) {
                        continue;
                    }
                    let w = weights.get(&canonical(node, nbr)).copied().unwrap_or(1.0);
                    let new_dist = d + w;
                    let old_dist = *dist.get(&nbr).unwrap_or(&f64::INFINITY);

                    if new_dist < old_dist {
                        dist.insert(nbr, new_dist);
                        num_paths.insert(nbr, *num_paths.get(&node).unwrap_or(&0.0));
                        predecessors.insert(nbr, vec![node]);
                        heap.push(std::cmp::Reverse(OrdF64Pair(new_dist, nbr)));
                    } else if (new_dist - old_dist).abs() < 1e-12 {
                        *num_paths.entry(nbr).or_insert(0.0) +=
                            num_paths.get(&node).copied().unwrap_or(0.0);
                        predecessors.entry(nbr).or_default().push(node);
                    }
                }
            }
        }
    } else {
        let mut queue = VecDeque::with_capacity(n);
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            stack.push(node);
            let node_dist = dist[&node];

            if let Some(ns) = adj.get(&node) {
                for &nbr in ns {
                    if !members.contains(&nbr) {
                        continue;
                    }
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(nbr) {
                        e.insert(node_dist + 1.0);
                        queue.push_back(nbr);
                    }
                    if (dist[&nbr] - (node_dist + 1.0)).abs() < 1e-12 {
                        *num_paths.entry(nbr).or_insert(0.0) +=
                            num_paths.get(&node).copied().unwrap_or(0.0);
                        predecessors.entry(nbr).or_default().push(node);
                    }
                }
            }
        }
    }

    let mut dependency: FxHashMap<u64, f64> = FxHashMap::default();
    let mut edge_btwns: FxHashMap<(u64, u64), f64> = FxHashMap::default();

    while let Some(w) = stack.pop() {
        if let Some(preds) = predecessors.get(&w) {
            for &v in preds {
                let coeff = (num_paths.get(&v).unwrap_or(&1.0)
                    / num_paths.get(&w).unwrap_or(&1.0))
                    * (1.0 + dependency.get(&w).unwrap_or(&0.0));
                *dependency.entry(v).or_insert(0.0) += coeff;
                *edge_btwns.entry(canonical(v, w)).or_insert(0.0) += coeff;
            }
        }
    }

    edge_btwns
}

#[derive(Clone, Copy)]
struct OrdF64Pair(f64, u64);

impl PartialEq for OrdF64Pair {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Eq for OrdF64Pair {}
impl PartialOrd for OrdF64Pair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF64Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.1.cmp(&other.1))
    }
}

/// Newman-Girvan modularity: Q = (1/2m) * sum_{ij} [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
/// Evaluated against the *original* graph adjacency (not the working copy).
fn compute_modularity(
    orig_adj: &FxHashMap<u64, FxHashSet<u64>>,
    orig_degrees: &FxHashMap<u64, f64>,
    m: f64,
    components: &FxHashMap<usize, FxHashSet<u64>>,
) -> f64 {
    if m == 0.0 {
        return 0.0;
    }
    let inv_2m = 1.0 / (2.0 * m);
    let mut q = 0.0;

    for members in components.values() {
        let mut l_c: f64 = 0.0;
        let mut d_c: f64 = 0.0;

        for &u in members {
            d_c += orig_degrees.get(&u).copied().unwrap_or(0.0);
            if let Some(ns) = orig_adj.get(&u) {
                for &v in ns {
                    if members.contains(&v) {
                        l_c += 1.0;
                    }
                }
            }
        }
        // Each undirected edge counted twice in the adjacency scan
        l_c /= 2.0;

        q += l_c / m - (d_c * inv_2m) * (d_c * inv_2m);
    }

    q
}

fn partition_from_components(components: &FxHashMap<usize, FxHashSet<u64>>) -> Vec<Vec<u64>> {
    let mut partition: Vec<Vec<u64>> = components
        .values()
        .map(|members| {
            let mut v: Vec<u64> = members.iter().copied().collect();
            v.sort_unstable();
            v
        })
        .collect();
    partition.sort_unstable();
    partition
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    fn node(id: u64) -> Node {
        Node {
            id,
            label: None,
            attributes: None,
        }
    }

    fn edge(src: u64, tgt: u64) -> Edge {
        Edge {
            id: 0,
            source: src,
            target: tgt,
            weight: None,
            attributes: None,
        }
    }

    fn make_two_triangles() -> Graph {
        let mut g = Graph::new_rust(false, false);
        for i in 1..=6 {
            g.add_node_rust(node(i)).unwrap();
        }
        // Triangle 1: 1-2-3
        g.add_edge_rust(edge(1, 2)).unwrap();
        g.add_edge_rust(edge(2, 3)).unwrap();
        g.add_edge_rust(edge(3, 1)).unwrap();
        // Bridge: 3-4
        g.add_edge_rust(edge(3, 4)).unwrap();
        // Triangle 2: 4-5-6
        g.add_edge_rust(edge(4, 5)).unwrap();
        g.add_edge_rust(edge(5, 6)).unwrap();
        g.add_edge_rust(edge(6, 4)).unwrap();
        g
    }

    fn make_karate_like() -> Graph {
        let mut g = Graph::new_rust(false, false);
        for i in 1..=12 {
            g.add_node_rust(node(i)).unwrap();
        }
        // Group A: clique on {1..6}
        for i in 1..=6u64 {
            for j in (i + 1)..=6 {
                g.add_edge_rust(edge(i, j)).unwrap();
            }
        }
        // Group B: clique on {7..12}
        for i in 7..=12u64 {
            for j in (i + 1)..=12 {
                g.add_edge_rust(edge(i, j)).unwrap();
            }
        }
        // Two bridge edges
        g.add_edge_rust(edge(3, 8)).unwrap();
        g.add_edge_rust(edge(5, 10)).unwrap();
        g
    }

    #[test]
    fn test_detect_communities_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let communities = detect_communities(&graph).unwrap();
        assert_eq!(communities.len(), 0);
    }

    #[test]
    fn test_detect_communities_assigns_communities() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_node_rust(node(1)).unwrap();
        graph.add_node_rust(node(2)).unwrap();
        let communities = detect_communities(&graph).unwrap();
        assert_eq!(communities.len(), 2);
    }

    #[test]
    fn test_girvan_newman_empty() {
        let graph = Graph::new_rust(false, false);
        let result = girvan_newman(&graph, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_girvan_newman_single_node() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_node_rust(node(1)).unwrap();
        let result = girvan_newman(&graph, None).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![vec![1]]);
    }

    #[test]
    fn test_girvan_newman_no_edges() {
        let mut graph = Graph::new_rust(false, false);
        graph.add_node_rust(node(1)).unwrap();
        graph.add_node_rust(node(2)).unwrap();
        let result = girvan_newman(&graph, None).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_girvan_newman_two_triangles() {
        let graph = make_two_triangles();
        let result = girvan_newman(&graph, None).unwrap();

        assert!(!result.is_empty());
        let best = &result[0];
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].len(), 3);
        assert_eq!(best[1].len(), 3);

        let c1: FxHashSet<u64> = best[0].iter().copied().collect();
        let c2: FxHashSet<u64> = best[1].iter().copied().collect();
        let expected1: FxHashSet<u64> = [1, 2, 3].iter().copied().collect();
        let expected2: FxHashSet<u64> = [4, 5, 6].iter().copied().collect();
        assert!(
            (c1 == expected1 && c2 == expected2)
                || (c1 == expected2 && c2 == expected1)
        );
    }

    #[test]
    fn test_girvan_newman_two_cliques() {
        let graph = make_karate_like();
        let result = girvan_newman(&graph, None).unwrap();

        assert!(!result.is_empty());
        let best = &result[0];
        assert_eq!(best.len(), 2);

        let group_a: FxHashSet<u64> = (1..=6).collect();
        let group_b: FxHashSet<u64> = (7..=12).collect();
        let c1: FxHashSet<u64> = best[0].iter().copied().collect();
        let c2: FxHashSet<u64> = best[1].iter().copied().collect();
        assert!(
            (c1 == group_a && c2 == group_b) || (c1 == group_b && c2 == group_a)
        );
    }

    #[test]
    fn test_girvan_newman_directed_errors() {
        let graph = Graph::new_rust(true, false);
        assert!(girvan_newman(&graph, None).is_err());
    }

    #[test]
    fn test_girvan_newman_dendrogram_has_multiple_levels() {
        let graph = make_two_triangles();
        let result = girvan_newman(&graph, None).unwrap();
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_girvan_newman_all_nodes_present() {
        let graph = make_two_triangles();
        let result = girvan_newman(&graph, None).unwrap();
        for partition in &result {
            let all_nodes: FxHashSet<u64> =
                partition.iter().flat_map(|c| c.iter().copied()).collect();
            let expected: FxHashSet<u64> = (1..=6).collect();
            assert_eq!(all_nodes, expected);
        }
    }

    #[test]
    fn test_edge_betweenness_bridge_highest() {
        let graph = make_two_triangles();
        let adj: FxHashMap<u64, FxHashSet<u64>> = graph
            .get_adjacency_list()
            .iter()
            .map(|(&k, v)| (k, v.iter().copied().collect()))
            .collect();
        let all_nodes: FxHashSet<u64> = adj.keys().copied().collect();
        let weights = FxHashMap::default();
        let eb = edge_betweenness_for_component(&all_nodes, &adj, false, &weights);

        let bridge = canonical(3, 4);
        let bridge_val = eb.get(&bridge).copied().unwrap_or(0.0);

        for (&e, &val) in &eb {
            if e != bridge {
                assert!(
                    bridge_val >= val,
                    "Bridge {:?} betweenness {} should be >= edge {:?} betweenness {}",
                    bridge, bridge_val, e, val
                );
            }
        }
    }
}
