//! Shortest path algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Ordering;

/// Priority queue item for Dijkstra's algorithm
#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse the ordering
        other.cost.partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Find shortest path between two nodes
#[pyfunction]
pub fn shortest_path(graph: &Graph, source: u64, target: u64) -> PyResult<Option<Vec<u64>>> {
    // Check if nodes exist
    if !graph.has_node_id(source) || !graph.has_node_id(target) {
        return Ok(None);
    }
    
    // Check if graph is weighted - use Dijkstra if so, BFS otherwise
    if graph.is_weighted() {
        dijkstra_shortest_path(graph, source, target)
    } else {
        bfs_shortest_path(graph, source, target)
    }
}

/// Dijkstra's algorithm for weighted graphs
fn dijkstra_shortest_path(graph: &Graph, source: u64, target: u64) -> PyResult<Option<Vec<u64>>> {
    let node_ids = graph.get_node_ids();
    let mut node_to_idx: FxHashMap<u64, usize> = FxHashMap::default();
    let mut idx_to_node: Vec<u64> = Vec::new();
    
    // Build node index mappings
    for (idx, &node_id) in node_ids.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
        idx_to_node.push(node_id);
    }
    
    let source_idx = *node_to_idx.get(&source).unwrap();
    let target_idx = *node_to_idx.get(&target).unwrap();
    
    // Initialize distances and predecessors
    let n = node_ids.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    let mut heap = BinaryHeap::new();
    
    dist[source_idx] = 0.0;
    heap.push(State { cost: 0.0, node: source_idx });
    
    while let Some(State { cost, node }) = heap.pop() {
        // Skip if we've found a better path already
        if cost > dist[node] {
            continue;
        }
        
        // Found target
        if node == target_idx {
            break;
        }
        
        // Explore neighbors
        let node_id = idx_to_node[node];
        if let Ok(neighbors) = graph.neighbors(node_id) {
            for neighbor_id in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor_id) {
                    // Get edge weight
                    let weight = graph.get_edge_weight(node_id, neighbor_id);
                    let new_cost = cost + weight;
                    
                    // Relaxation step
                    if new_cost < dist[neighbor_idx] {
                        dist[neighbor_idx] = new_cost;
                        prev[neighbor_idx] = Some(node);
                        heap.push(State { cost: new_cost, node: neighbor_idx });
                    }
                }
            }
        }
    }
    
    // Reconstruct path
    if dist[target_idx].is_infinite() {
        return Ok(None);
    }
    
    let mut path = Vec::new();
    let mut current = target_idx;
    
    while let Some(p) = prev[current] {
        path.push(idx_to_node[current]);
        current = p;
    }
    path.push(idx_to_node[current]);
    path.reverse();
    
    Ok(Some(path))
}

/// BFS for unweighted graphs
fn bfs_shortest_path(graph: &Graph, source: u64, target: u64) -> PyResult<Option<Vec<u64>>> {
    // Simple BFS for unweighted graphs
    let mut queue = VecDeque::new();
    let mut visited = FxHashMap::default();
    let mut parent: FxHashMap<u64, u64> = FxHashMap::default();
    
    queue.push_back(source);
    visited.insert(source, true);
    
    while let Some(current) = queue.pop_front() {
        if current == target {
            // Reconstruct path
            let mut path = vec![current];
            let mut node = current;
            
            while let Some(&p) = parent.get(&node) {
                path.push(p);
                node = p;
            }
            
            path.reverse();
            return Ok(Some(path));
        }
        
        if let Ok(neighbors) = graph.neighbors(current) {
            for neighbor in neighbors {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, true);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    
    // No path found
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    #[test]
    fn test_shortest_path_direct_connection() {
        let mut graph = Graph::new(false, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        
        let path = shortest_path(&graph, "A", "B").unwrap();
        assert_eq!(path, vec!["A", "B"]);
    }

    #[test]
    fn test_shortest_path_multiple_hops() {
        let mut graph = Graph::new(false, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();
        
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();
        
        let path = shortest_path(&graph, "A", "C").unwrap();
        assert_eq!(path, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_shortest_path_no_connection() {
        let mut graph = Graph::new(false, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        
        let result = shortest_path(&graph, "A", "B");
        assert!(result.is_err());
    }
}
