//! Cycle detection algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

/// Find all cycles in the graph
#[pyfunction]
#[pyo3(signature = (_graph, _min_amount=None))]
pub fn find_cycles(_graph: &Graph, _min_amount: Option<f64>) -> PyResult<Vec<Vec<u64>>> {
    // Placeholder implementation
    // TODO: Implement Johnson's algorithm for finding all cycles
    let cycles: Vec<Vec<u64>> = Vec::new();
    Ok(cycles)
}

/// Detect if graph has any cycle using DFS
pub fn has_cycle(graph: &Graph) -> bool {
    let mut visited = FxHashSet::default();
    let mut rec_stack = FxHashSet::default();
    
    for node_id in graph.get_node_ids() {
        if !visited.contains(&node_id)
            && has_cycle_util(graph, node_id, &mut visited, &mut rec_stack) {
                return true;
            }
    }
    
    false
}

fn has_cycle_util(
    graph: &Graph,
    node: u64,
    visited: &mut FxHashSet<u64>,
    rec_stack: &mut FxHashSet<u64>,
) -> bool {
    visited.insert(node);
    rec_stack.insert(node);
    
    if let Ok(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                if has_cycle_util(graph, neighbor, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack.contains(&neighbor) {
                return true;
            }
        }
    }
    
    rec_stack.remove(&node);
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    #[test]
    fn test_find_cycles_empty_graph() {
        let graph = Graph::new(false, false);
        let cycles = find_cycles(&graph, None).unwrap();
        assert_eq!(cycles.len(), 0);
    }

    #[test]
    fn test_has_cycle_acyclic_graph() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();
        
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();
        
        assert!(!has_cycle(&graph));
    }

    #[test]
    fn test_has_cycle_cyclic_graph() {
        let mut graph = Graph::new(true, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        graph.add_node(Node::new("B".to_string(), None)).unwrap();
        graph.add_node(Node::new("C".to_string(), None)).unwrap();
        
        graph
            .add_edge(Edge::new("A".to_string(), "B".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("B".to_string(), "C".to_string(), None, None))
            .unwrap();
        graph
            .add_edge(Edge::new("C".to_string(), "A".to_string(), None, None))
            .unwrap();
        
        assert!(has_cycle(&graph));
    }
}
