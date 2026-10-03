use crate::graph::Graph;
use rustc_hash::{FxHashMap, FxHashSet};

/// Count weakly connected components in a graph.
/// For undirected graphs, this is the same as regular connected components.
/// For directed graphs, ignores edge direction (treats as undirected).
#[inline]
pub fn count_weakly_connected_components(graph: &Graph) -> usize {
    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return 0;
    }

    let mut visited: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(n, Default::default());
    let mut component_count = 0;

    for &node_id in &node_ids {
        if !visited.contains(&node_id) {
            component_count += 1;
            // BFS to find all nodes in this component
            let mut queue: Vec<u64> = Vec::with_capacity(n);
            queue.push(node_id);
            visited.insert(node_id);

            while let Some(current) = queue.pop() {
                // Check forward edges (successors)
                if let Some(neighbors) = graph.get_adjacency_list().get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push(neighbor);
                        }
                    }
                }

                // For directed graphs, also check reverse edges (predecessors)
                if graph.is_directed() {
                    for edge in graph.get_edges_internal() {
                        if edge.target == current && !visited.contains(&edge.source) {
                            visited.insert(edge.source);
                            queue.push(edge.source);
                        }
                    }
                }
            }
        }
    }

    component_count
}

/// Count strongly connected components.
///
/// For directed graphs: Uses Kosaraju's algorithm to find maximal sets where
/// every node is reachable from every other node following directed edges.
///
/// For undirected graphs: Returns the same result as weakly connected components,
/// since strong connectivity equals regular connectivity in undirected graphs.
/// This matches igraph's behavior.
///
/// Uses Kosaraju's algorithm for directed graphs:
/// 1. Fill order by finishing times (DFS)
/// 2. Transpose the graph
/// 3. DFS on transposed graph in reverse finish order
#[inline]
pub fn count_strongly_connected_components(graph: &Graph) -> Result<usize, String> {
    // For undirected graphs, strong connectivity = regular connectivity
    if !graph.is_directed() {
        return Ok(count_weakly_connected_components(graph));
    }

    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(0);
    }

    // Step 1: Fill vertices in stack according to their finishing times
    let mut visited: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(n, Default::default());
    let mut stack: Vec<u64> = Vec::with_capacity(n);

    fn fill_order(
        node: u64,
        visited: &mut FxHashSet<u64>,
        stack: &mut Vec<u64>,
        adj_list: &FxHashMap<u64, Vec<u64>>,
    ) {
        visited.insert(node);

        if let Some(neighbors) = adj_list.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    fill_order(neighbor, visited, stack, adj_list);
                }
            }
        }

        stack.push(node);
    }

    for &node_id in &node_ids {
        if !visited.contains(&node_id) {
            fill_order(node_id, &mut visited, &mut stack, graph.get_adjacency_list());
        }
    }

    // Step 2: Create transposed graph (reverse all edges)
    let mut transposed: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());

    for edge in graph.get_edges_internal() {
        transposed
            .entry(edge.target)
            .or_insert_with(Vec::new)
            .push(edge.source);
    }

    // Step 3: Do DFS on transposed graph in reverse finish order
    visited.clear();
    let mut component_count = 0;

    fn dfs(
        node: u64,
        visited: &mut FxHashSet<u64>,
        adj_list: &FxHashMap<u64, Vec<u64>>,
    ) {
        visited.insert(node);

        if let Some(neighbors) = adj_list.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    dfs(neighbor, visited, adj_list);
                }
            }
        }
    }

    while let Some(node) = stack.pop() {
        if !visited.contains(&node) {
            dfs(node, &mut visited, &transposed);
            component_count += 1;
        }
    }

    Ok(component_count)
}

/// Get all weakly connected components as sets of node IDs.
/// For undirected graphs, this is the same as regular connected components.
/// For directed graphs, ignores edge direction (treats as undirected).
///
/// Returns a vector of vectors, where each inner vector contains the node IDs
/// in a single component. Components are returned in no particular order.
#[inline]
pub fn weakly_connected_components(graph: &Graph) -> Vec<Vec<u64>> {
    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Vec::new();
    }

    let mut visited: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(n, Default::default());
    let mut components: Vec<Vec<u64>> = Vec::new();

    for &node_id in &node_ids {
        if !visited.contains(&node_id) {
            let mut component: Vec<u64> = Vec::new();
            // BFS to find all nodes in this component
            let mut queue: Vec<u64> = Vec::with_capacity(n);
            queue.push(node_id);
            visited.insert(node_id);

            while let Some(current) = queue.pop() {
                component.push(current);

                // Check forward edges (successors)
                if let Some(neighbors) = graph.get_adjacency_list().get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push(neighbor);
                        }
                    }
                }

                // For directed graphs, also check reverse edges (predecessors)
                if graph.is_directed() {
                    for edge in graph.get_edges_internal() {
                        if edge.target == current && !visited.contains(&edge.source) {
                            visited.insert(edge.source);
                            queue.push(edge.source);
                        }
                    }
                }
            }

            components.push(component);
        }
    }

    components
}

/// Get all strongly connected components as sets of node IDs.
///
/// For directed graphs: Uses Kosaraju's algorithm to find maximal sets where
/// every node is reachable from every other node following directed edges.
///
/// For undirected graphs: Returns the same result as weakly connected components,
/// since strong connectivity equals regular connectivity in undirected graphs.
///
/// Returns a vector of vectors, where each inner vector contains the node IDs
/// in a single strongly connected component.
#[inline]
pub fn strongly_connected_components(graph: &Graph) -> Result<Vec<Vec<u64>>, String> {
    // For undirected graphs, strong connectivity = regular connectivity
    if !graph.is_directed() {
        return Ok(weakly_connected_components(graph));
    }

    let node_ids: Vec<u64> = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Step 1: Fill vertices in stack according to their finishing times
    let mut visited: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(n, Default::default());
    let mut stack: Vec<u64> = Vec::with_capacity(n);

    fn fill_order(
        node: u64,
        visited: &mut FxHashSet<u64>,
        stack: &mut Vec<u64>,
        adj_list: &FxHashMap<u64, Vec<u64>>,
    ) {
        visited.insert(node);

        if let Some(neighbors) = adj_list.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    fill_order(neighbor, visited, stack, adj_list);
                }
            }
        }

        stack.push(node);
    }

    for &node_id in &node_ids {
        if !visited.contains(&node_id) {
            fill_order(node_id, &mut visited, &mut stack, graph.get_adjacency_list());
        }
    }

    // Step 2: Create transposed graph (reverse all edges)
    let mut transposed: FxHashMap<u64, Vec<u64>> = FxHashMap::with_capacity_and_hasher(n, Default::default());

    for edge in graph.get_edges_internal() {
        transposed
            .entry(edge.target)
            .or_insert_with(Vec::new)
            .push(edge.source);
    }

    // Step 3: Do DFS on transposed graph in reverse finish order, collecting components
    visited.clear();
    let mut components: Vec<Vec<u64>> = Vec::new();

    fn dfs_collect(
        node: u64,
        visited: &mut FxHashSet<u64>,
        adj_list: &FxHashMap<u64, Vec<u64>>,
        component: &mut Vec<u64>,
    ) {
        visited.insert(node);
        component.push(node);

        if let Some(neighbors) = adj_list.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    dfs_collect(neighbor, visited, adj_list, component);
                }
            }
        }
    }

    while let Some(node) = stack.pop() {
        if !visited.contains(&node) {
            let mut component: Vec<u64> = Vec::new();
            dfs_collect(node, &mut visited, &transposed, &mut component);
            components.push(component);
        }
    }

    Ok(components)
}

/// Check if a directed graph is strongly connected.
///
/// A directed graph is strongly connected if there is a path from every vertex
/// to every other vertex following directed edges.
///
/// For undirected graphs, this is the same as checking regular connectivity.
#[inline]
pub fn is_strongly_connected(graph: &Graph) -> Result<bool, String> {
    if graph.get_node_ids().is_empty() {
        // Empty graph is considered connected (vacuously true)
        return Ok(true);
    }

    // A graph is strongly connected iff it has exactly one SCC
    Ok(count_strongly_connected_components(graph)? == 1)
}

/// Check if a directed graph is weakly connected.
///
/// A directed graph is weakly connected if the underlying undirected graph
/// (ignoring edge directions) is connected.
///
/// For undirected graphs, this is the same as checking regular connectivity.
#[inline]
pub fn is_weakly_connected(graph: &Graph) -> bool {
    if graph.get_node_ids().is_empty() {
        // Empty graph is considered connected (vacuously true)
        return true;
    }

    // A graph is weakly connected iff it has exactly one WCC
    count_weakly_connected_components(graph) == 1
}

/// Find k-components of a graph (directed or undirected).
///
/// A k-component is a maximal subset of nodes such that every pair of nodes
/// in the subset has node connectivity at least k. The result is a mapping
/// from k to the list of k-components for that level.
///
/// For directed graphs, k-connectivity means that removing any fewer than k
/// nodes keeps the subgraph strongly connected.
///
/// Uses the Moody-White cohesive blocking approach:
/// 1. Find base components (SCCs for directed, CCs for undirected)
/// 2. Recursively find higher-connectivity subsets using minimum separators
#[pyo3::pyfunction]
pub fn k_components(graph: &Graph) -> pyo3::PyResult<FxHashMap<usize, Vec<Vec<u64>>>> {
    let mut result: FxHashMap<usize, Vec<Vec<u64>>> = FxHashMap::default();

    let node_ids = graph.get_node_ids();
    if node_ids.is_empty() {
        return Ok(result);
    }

    // k=1: SCCs for directed, CCs for undirected
    let base_components = if graph.is_directed() {
        strongly_connected_components(graph)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?
    } else {
        weakly_connected_components_list(graph)
    };

    // Filter to components with >= 2 nodes
    let base_nontrivial: Vec<Vec<u64>> = base_components
        .into_iter()
        .filter(|c| c.len() >= 2)
        .collect();

    if base_nontrivial.is_empty() {
        return Ok(result);
    }

    result.entry(1).or_default().extend(base_nontrivial.iter().cloned());

    // For each base component, recursively find higher-k components
    for component in &base_nontrivial {
        find_k_components_recursive(graph, component, 1, &mut result)?;
    }

    // Different separator choices can reach the same node set, or a subset of
    // another set at the same level; k-components are the *maximal* sets.
    for comps in result.values_mut() {
        for c in comps.iter_mut() {
            c.sort_unstable();
        }
        comps.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        comps.dedup();
        let mut kept: Vec<Vec<u64>> = Vec::with_capacity(comps.len());
        for c in comps.drain(..) {
            let is_subset = kept.iter().any(|k: &Vec<u64>| {
                c.iter().all(|x| k.binary_search(x).is_ok())
            });
            if !is_subset {
                kept.push(c);
            }
        }
        *comps = kept;
    }

    Ok(result)
}

fn find_k_components_recursive(
    graph: &Graph,
    nodes: &[u64],
    current_k: usize,
    result: &mut FxHashMap<usize, Vec<Vec<u64>>>,
) -> pyo3::PyResult<()> {
    if nodes.len() <= current_k {
        return Ok(());
    }

    // Compute the connectivity of this subgraph
    let connectivity = if graph.is_directed() {
        directed_subgraph_connectivity(graph, nodes)?
    } else {
        graph.induced_subgraph_connectivity(nodes)?
    };

    // Register this component at all levels from current_k+1 up to its connectivity
    for k in (current_k + 1)..=connectivity {
        result.entry(k).or_default().push(nodes.to_vec());
    }

    // A candidate (component + separator) can be disconnected; look inside each piece.
    if connectivity == 0 {
        let pieces = if graph.is_directed() {
            find_directed_components_in_subgraph(graph, nodes)
        } else {
            graph.find_components_in_subgraph(nodes)
        };
        if pieces.len() > 1 {
            for piece in pieces {
                if piece.len() > current_k {
                    find_k_components_recursive(graph, &piece, current_k, result)?;
                }
            }
        }
        return Ok(());
    }

    // Even when this set is *only* `connectivity`-connected (no better than the
    // level we arrived at), denser subsets may hide behind its minimum
    // separators, so always split there and recurse (Moody-White). Stopping at
    // `connectivity <= current_k` meant a graph with a cut vertex never got past k=1.
    let level = current_k.max(connectivity);
    let separators = graph.find_minimal_separators(nodes, connectivity)?;

    if separators.is_empty() {
        return Ok(());
    }

    let mut processed = FxHashSet::default();

    for separator in &separators {
        let sep_set: FxHashSet<u64> = separator.iter().copied().collect();

        let remaining: Vec<u64> = nodes
            .iter()
            .filter(|&&n| !sep_set.contains(&n))
            .copied()
            .collect();

        if remaining.is_empty() {
            continue;
        }

        let components = if graph.is_directed() {
            find_directed_components_in_subgraph(graph, &remaining)
        } else {
            graph.find_components_in_subgraph(&remaining)
        };

        for comp in components {
            if comp.len() <= 1 {
                continue;
            }

            // Form candidate: component + separator
            let mut candidate: Vec<u64> = comp;
            candidate.extend(separator.iter().copied());
            candidate.sort_unstable();
            candidate.dedup();

            let sig = format!("{:?}", candidate);
            if processed.contains(&sig) {
                continue;
            }
            processed.insert(sig);

            // candidate is a strict subset of `nodes` (the separator splits it),
            // so the recursion terminates
            if candidate.len() <= level || candidate.len() >= nodes.len() {
                continue;
            }

            find_k_components_recursive(graph, &candidate, level, result)?;
        }
    }

    Ok(())
}

/// Compute directed node connectivity of a subgraph.
///
/// For directed graphs, connectivity κ means the graph remains strongly
/// connected after removing any fewer than κ vertices. We compute this as
/// the minimum over all ordered pairs (u,v) of the directed node connectivity.
fn directed_subgraph_connectivity(graph: &Graph, nodes: &[u64]) -> pyo3::PyResult<usize> {
    let n = nodes.len();

    if n <= 1 {
        return Ok(0);
    }

    if n == 2 {
        let has_fwd = graph.has_edge(nodes[0], nodes[1]);
        let has_bwd = graph.has_edge(nodes[1], nodes[0]);
        return Ok(if has_fwd && has_bwd { 1 } else { 0 });
    }

    let node_set: FxHashSet<u64> = nodes.iter().copied().collect();

    // Check if the subgraph is strongly connected
    if !is_subgraph_strongly_connected(graph, &node_set) {
        return Ok(0);
    }

    // Compute min pairwise directed connectivity (both directions)
    let mut min_conn = usize::MAX;

    for i in 0..n {
        for j in (i + 1)..n {
            let u = nodes[i];
            let v = nodes[j];

            // Check u -> v connectivity
            let conn_uv = graph.node_connectivity_between_in_subgraph(u, v, &node_set)?;
            min_conn = min_conn.min(conn_uv);

            if min_conn == 0 {
                return Ok(0);
            }

            // Check v -> u connectivity
            let conn_vu = graph.node_connectivity_between_in_subgraph(v, u, &node_set)?;
            min_conn = min_conn.min(conn_vu);

            if min_conn <= 1 {
                return Ok(min_conn);
            }
        }
    }

    if min_conn == usize::MAX {
        Ok(n - 1)
    } else {
        Ok(min_conn)
    }
}

/// Check if a subgraph is strongly connected (all nodes can reach all others via directed edges).
fn is_subgraph_strongly_connected(graph: &Graph, node_set: &FxHashSet<u64>) -> bool {
    if node_set.len() <= 1 {
        return true;
    }

    let start = *node_set.iter().next().unwrap();
    let adj = graph.get_adjacency_list();

    // Forward reachability
    let mut visited = FxHashSet::default();
    let mut queue = std::collections::VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        if let Some(neighbors) = adj.get(&u) {
            for &v in neighbors {
                if node_set.contains(&v) && !visited.contains(&v) {
                    visited.insert(v);
                    queue.push_back(v);
                }
            }
        }
    }

    if visited.len() != node_set.len() {
        return false;
    }

    // Backward reachability (reverse edges)
    let mut visited = FxHashSet::default();
    let mut queue = std::collections::VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    // Build reverse adjacency restricted to node_set
    let edges = graph.get_edges_internal();
    let mut reverse_adj: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for edge in edges {
        if node_set.contains(&edge.source) && node_set.contains(&edge.target) {
            reverse_adj.entry(edge.target).or_default().push(edge.source);
        }
    }

    while let Some(u) = queue.pop_front() {
        if let Some(preds) = reverse_adj.get(&u) {
            for &v in preds {
                if !visited.contains(&v) {
                    visited.insert(v);
                    queue.push_back(v);
                }
            }
        }
    }

    visited.len() == node_set.len()
}

/// Find strongly connected components within a subgraph (restricted to node_set).
fn find_directed_components_in_subgraph(graph: &Graph, nodes: &[u64]) -> Vec<Vec<u64>> {
    let node_set: FxHashSet<u64> = nodes.iter().copied().collect();
    let adj = graph.get_adjacency_list();

    // Kosaraju's within the subgraph
    // Step 1: forward DFS to get finish order
    let mut visited = FxHashSet::default();
    let mut stack = Vec::new();

    fn fill_order(
        node: u64,
        visited: &mut FxHashSet<u64>,
        stack: &mut Vec<u64>,
        adj: &FxHashMap<u64, Vec<u64>>,
        node_set: &FxHashSet<u64>,
    ) {
        visited.insert(node);
        if let Some(neighbors) = adj.get(&node) {
            for &v in neighbors {
                if node_set.contains(&v) && !visited.contains(&v) {
                    fill_order(v, visited, stack, adj, node_set);
                }
            }
        }
        stack.push(node);
    }

    for &node in nodes {
        if !visited.contains(&node) {
            fill_order(node, &mut visited, &mut stack, adj, &node_set);
        }
    }

    // Step 2: Build reverse adjacency
    let edges = graph.get_edges_internal();
    let mut reverse_adj: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    for edge in edges {
        if node_set.contains(&edge.source) && node_set.contains(&edge.target) {
            reverse_adj.entry(edge.target).or_default().push(edge.source);
        }
    }

    // Step 3: DFS on transposed in reverse finish order
    visited.clear();
    let mut components = Vec::new();

    fn dfs_collect(
        node: u64,
        visited: &mut FxHashSet<u64>,
        adj: &FxHashMap<u64, Vec<u64>>,
        component: &mut Vec<u64>,
    ) {
        visited.insert(node);
        component.push(node);
        if let Some(neighbors) = adj.get(&node) {
            for &v in neighbors {
                if !visited.contains(&v) {
                    dfs_collect(v, visited, adj, component);
                }
            }
        }
    }

    while let Some(node) = stack.pop() {
        if !visited.contains(&node) {
            let mut component = Vec::new();
            dfs_collect(node, &mut visited, &reverse_adj, &mut component);
            components.push(component);
        }
    }

    components
}

/// Get weakly connected components as a list of node ID vectors.
fn weakly_connected_components_list(graph: &Graph) -> Vec<Vec<u64>> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Vec::new();
    }

    let adj = graph.get_adjacency_list();
    let mut visited = FxHashSet::with_capacity_and_hasher(n, Default::default());
    let mut components = Vec::new();

    // For undirected graphs, adjacency_list has both directions
    // For directed, we need to ignore direction
    let mut undirected_adj: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    if graph.is_directed() {
        for (&node, neighbors) in adj.iter() {
            for &nbr in neighbors {
                undirected_adj.entry(node).or_default().push(nbr);
                undirected_adj.entry(nbr).or_default().push(node);
            }
        }
    }

    let effective_adj = if graph.is_directed() { &undirected_adj } else { adj };

    for &node in &node_ids {
        if visited.contains(&node) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        visited.insert(node);
        queue.push_back(node);

        while let Some(current) = queue.pop_front() {
            component.push(current);
            if let Some(neighbors) = effective_adj.get(&current) {
                for &nbr in neighbors {
                    if !visited.contains(&nbr) {
                        visited.insert(nbr);
                        queue.push_back(nbr);
                    }
                }
            }
        }

        components.push(component);
    }

    components
}
