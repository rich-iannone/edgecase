use crate::graph::Graph;
use rustc_hash::{FxHashMap, FxHashSet};

/// Count weakly connected components in a graph.
/// For undirected graphs, this is the same as regular connected components.
/// For directed graphs, ignores edge direction (treats as undirected).
pub fn count_weakly_connected_components(graph: &Graph) -> usize {
    let node_ids: Vec<u64> = graph.get_node_ids();
    
    if node_ids.is_empty() {
        return 0;
    }

    let mut visited: FxHashSet<u64> = FxHashSet::default();
    let mut component_count = 0;

    for &node_id in &node_ids {
        if !visited.contains(&node_id) {
            component_count += 1;
            // BFS to find all nodes in this component
            let mut queue: Vec<u64> = vec![node_id];
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

/// Count strongly connected components in a directed graph.
/// Uses Kosaraju's algorithm:
/// 1. Fill order by finishing times (DFS)
/// 2. Transpose the graph
/// 3. DFS on transposed graph in reverse finish order
pub fn count_strongly_connected_components(graph: &Graph) -> Result<usize, String> {
    if !graph.is_directed() {
        return Err("Strongly connected components only defined for directed graphs".to_string());
    }

    let node_ids: Vec<u64> = graph.get_node_ids();
    
    if node_ids.is_empty() {
        return Ok(0);
    }

    // Step 1: Fill vertices in stack according to their finishing times
    let mut visited: FxHashSet<u64> = FxHashSet::default();
    let mut stack: Vec<u64> = Vec::new();

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
    let mut transposed: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
    
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
