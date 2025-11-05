//! Cycle detection algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

/// Find all cycles in the graph
#[pyfunction]
#[pyo3(signature = (graph, min_amount=None))]
pub fn find_cycles(graph: &Graph, min_amount: Option<f64>) -> PyResult<Vec<Vec<String>>> {
    // Placeholder implementation
    // TODO: Implement Johnson's algorithm for finding all cycles
    let cycles: Vec<Vec<String>> = Vec::new();
    Ok(cycles)
}

/// Detect if graph has any cycle using DFS
pub fn has_cycle(graph: &Graph) -> bool {
    let mut visited = FxHashSet::default();
    let mut rec_stack = FxHashSet::default();
    
    for node_id in graph.node_ids() {
        if !visited.contains(&node_id) {
            if has_cycle_util(graph, &node_id, &mut visited, &mut rec_stack) {
                return true;
            }
        }
    }
    
    false
}

fn has_cycle_util(
    graph: &Graph,
    node: &str,
    visited: &mut FxHashSet<String>,
    rec_stack: &mut FxHashSet<String>,
) -> bool {
    visited.insert(node.to_string());
    rec_stack.insert(node.to_string());
    
    if let Ok(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                if has_cycle_util(graph, &neighbor, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack.contains(&neighbor) {
                return true;
            }
        }
    }
    
    rec_stack.remove(node);
    false
}

