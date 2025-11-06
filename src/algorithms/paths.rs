//! Shortest path algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Find shortest path between two nodes
#[pyfunction]
pub fn shortest_path(graph: &Graph, source: &str, target: &str) -> PyResult<Option<Vec<String>>> {
    // Check if nodes exist
    if !graph.has_node(source) || !graph.has_node(target) {
        return Ok(None);
    }
    
    // Simple BFS for unweighted graphs
    let mut queue = VecDeque::new();
    let mut visited = FxHashMap::default();
    let mut parent: FxHashMap<String, String> = FxHashMap::default();
    
    queue.push_back(source.to_string());
    visited.insert(source.to_string(), true);
    
    while let Some(current) = queue.pop_front() {
        if current == target {
            // Reconstruct path
            let mut path = vec![current.clone()];
            let mut node = current;
            
            while let Some(p) = parent.get(&node) {
                path.push(p.clone());
                node = p.clone();
            }
            
            path.reverse();
            return Ok(Some(path));
        }
        
        if let Ok(neighbors) = graph.neighbors(&current) {
            for neighbor in neighbors {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor.clone(), true);
                    parent.insert(neighbor.clone(), current.clone());
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
