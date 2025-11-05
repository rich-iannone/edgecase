//! Shortest path algorithms

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Ordering;

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    position: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Find shortest path between two nodes
#[pyfunction]
pub fn shortest_path(graph: &Graph, source: &str, target: &str) -> PyResult<Vec<String>> {
    // Simple BFS for unweighted graphs
    let mut queue = VecDeque::new();
    let mut visited = FxHashMap::default();
    let mut parent = FxHashMap::default();
    
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
            return Ok(path);
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
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "No path found from '{}' to '{}'",
        source, target
    )))
}

