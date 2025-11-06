//! Centrality algorithms (PageRank, betweenness, etc.)

use crate::graph::Graph;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

/// Calculate betweenness centrality for all nodes
#[pyfunction]
pub fn betweenness_centrality(graph: &Graph) -> PyResult<FxHashMap<String, f64>> {
    // Placeholder implementation
    // TODO: Implement Brandes' algorithm for betweenness centrality
    let mut centrality = FxHashMap::default();
    
    for node_id in graph.node_ids() {
        centrality.insert(node_id, 0.0);
    }
    
    Ok(centrality)
}

/// Calculate PageRank for all nodes
#[pyfunction]
#[pyo3(signature = (graph, damping=0.85, max_iter=100, tol=1e-6))]
pub fn pagerank(
    graph: &Graph,
    damping: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<FxHashMap<String, f64>> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(FxHashMap::default());
    }
    
    let mut ranks = FxHashMap::default();
    let initial_rank = 1.0 / n as f64;
    
    // Initialize all nodes with equal rank
    for node_id in graph.node_ids() {
        ranks.insert(node_id.clone(), initial_rank);
    }
    
    // Power iteration
    for _ in 0..max_iter {
        let mut new_ranks = FxHashMap::default();
        let mut converged = true;
        
        for node_id in graph.node_ids() {
            let mut rank_sum = 0.0;
            
            // Sum contributions from incoming neighbors
            // Note: This is simplified and needs proper implementation
            // for directed graphs with in-degree consideration
            
            let new_rank = (1.0 - damping) / n as f64 + damping * rank_sum;
            new_ranks.insert(node_id.clone(), new_rank);
            
            if let Some(old_rank) = ranks.get(&node_id) {
                if (new_rank - old_rank).abs() > tol {
                    converged = false;
                }
            }
        }
        
        ranks = new_ranks;
        
        if converged {
            break;
        }
    }
    
    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    #[test]
    fn test_betweenness_centrality_empty_graph() {
        let graph = Graph::new(false, false);
        let centrality = betweenness_centrality(&graph).unwrap();
        assert_eq!(centrality.len(), 0);
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let graph = Graph::new(false, false);
        let ranks = pagerank(&graph, 0.85, 100, 1e-6).unwrap();
        assert_eq!(ranks.len(), 0);
    }

    #[test]
    fn test_pagerank_single_node() {
        let mut graph = Graph::new(false, false);
        graph.add_node(Node::new("A".to_string(), None)).unwrap();
        
        let ranks = pagerank(&graph, 0.85, 100, 1e-6).unwrap();
        assert_eq!(ranks.len(), 1);
        assert!((ranks["A"] - 1.0).abs() < 1e-6);
    }
}
