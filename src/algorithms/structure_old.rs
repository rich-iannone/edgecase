//! Graph structure and decomposition algorithms

use crate::graph::{Edge, Graph, Node};
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

/// Extract k-core subgraph
///
/// The k-core is the maximal subgraph where all nodes have degree >= k.
/// Returns a new Graph containing only the nodes in the k-core.
#[pyfunction]
#[inline]
pub fn k_core(graph: &Graph, k: usize) -> PyResult<Graph> {
    // Get coreness for all nodes
    let coreness = graph.get_coreness();
    
    // Find nodes with coreness >= k
    let core_nodes: FxHashSet<u64> = coreness
        .iter()
        .filter(|(_, &c)| c >= k)
        .map(|(&node, _)| node)
        .collect();
    
    if core_nodes.is_empty() {
        // Return empty graph with same properties
        return Ok(Graph::new_py(graph.is_directed(), graph.is_weighted()));
    }
    
    // Create subgraph with only core nodes
    let mut subgraph = Graph::new_py(graph.is_directed(), graph.is_weighted());
    
    // Add nodes
    for &node_id in &core_nodes {
        if let Some(node) = graph.get_node(node_id) {
            let new_node = Node::new_py(node_id, node.label.clone(), node.attributes.clone());
            subgraph.add_node_py(new_node)?;
        }
    }
    
    // Add edges between core nodes using internal edge iterator
    let edges = graph.get_edges_internal();
    for edge in edges {
        let source = edge.source;
        let target = edge.target;
        
        if core_nodes.contains(&source) && core_nodes.contains(&target) {
            let new_edge = Edge::new_py(
                edge.id,
                edge.source,
                edge.target,
                edge.weight,
                edge.attributes.clone(),
            );
            // For undirected graphs, check if already added
            if subgraph.get_edge(edge.id).is_none() {
                let _ = subgraph.add_edge_py(new_edge);
            }
        }
    }
    
    Ok(subgraph)
}

/// Extract k-shell subgraph
///
/// The k-shell is the set of nodes with coreness exactly equal to k.
/// Returns a new Graph containing only nodes in the k-shell.
#[pyfunction]
#[inline]
pub fn k_shell(graph: &Graph, k: usize) -> PyResult<Graph> {
    // Get coreness for all nodes
    let coreness = graph.get_coreness();
    
    // Find nodes with coreness exactly equal to k
    let shell_nodes: FxHashSet<u64> = coreness
        .iter()
        .filter(|(_, &c)| c == k)
        .map(|(&node, _)| node)
        .collect();
    
    if shell_nodes.is_empty() {
        // Return empty graph with same properties
        return Ok(Graph::new_py(graph.is_directed(), graph.is_weighted()));
    }
    
    // Create subgraph with only shell nodes
    let mut subgraph = Graph::new_py(graph.is_directed(), graph.is_weighted());
    
    // Add nodes
    for &node_id in &shell_nodes {
        if let Some(node) = graph.get_node(node_id) {
            let new_node = Node::new_py(node_id, node.label.clone(), node.attributes.clone());
            subgraph.add_node_py(new_node)?;
        }
    }
    
    // Add edges between shell nodes
    let edges = graph.get_edges_internal();
    for edge in edges {
        let source = edge.source;
        let target = edge.target;
        
        if shell_nodes.contains(&source) && shell_nodes.contains(&target) {
            let new_edge = Edge::new_py(
                edge.id,
                edge.source,
                edge.target,
                edge.weight,
                edge.attributes.clone(),
            );
            if subgraph.get_edge(edge.id).is_none() {
                let _ = subgraph.add_edge_py(new_edge);
            }
        }
    }
    
    Ok(subgraph)
}

/// Extract k-crust subgraph
///
/// The k-crust is the complement of the (k+1)-core, i.e., nodes with coreness <= k.
/// Returns a new Graph containing only nodes in the k-crust.
#[pyfunction]
#[inline]
pub fn k_crust(graph: &Graph, k: usize) -> PyResult<Graph> {
    // Get coreness for all nodes
    let coreness = graph.get_coreness();
    
    // Find nodes with coreness <= k
    let crust_nodes: FxHashSet<u64> = coreness
        .iter()
        .filter(|(_, &c)| c <= k)
        .map(|(&node, _)| node)
        .collect();
    
    if crust_nodes.is_empty() {
        // Return empty graph with same properties
        return Ok(Graph::new_py(graph.is_directed(), graph.is_weighted()));
    }
    
    // Create subgraph with only crust nodes
    let mut subgraph = Graph::new_py(graph.is_directed(), graph.is_weighted());
    
    // Add nodes
    for &node_id in &crust_nodes {
        if let Some(node) = graph.get_node(node_id) {
            let new_node = Node::new_py(node_id, node.label.clone(), node.attributes.clone());
            subgraph.add_node_py(new_node)?;
        }
    }
    
    // Add edges between crust nodes
    let edges = graph.get_edges_internal();
    for edge in edges {
        let source = edge.source;
        let target = edge.target;
        
        if crust_nodes.contains(&source) && crust_nodes.contains(&target) {
            let new_edge = Edge::new_py(
                edge.id,
                edge.source,
                edge.target,
                edge.weight,
                edge.attributes.clone(),
            );
            if subgraph.get_edge(edge.id).is_none() {
                let _ = subgraph.add_edge_py(new_edge);
            }
        }
    }
    
    Ok(subgraph)
}

/// Extract k-truss subgraph
///
/// The k-truss is the maximal subgraph where every edge is part of at least (k-2) triangles.
/// Uses iterative edge pruning: repeatedly removes edges with fewer than (k-2) triangles.
/// More restrictive than k-core: k-truss ⊆ k-core for the same k.
#[pyfunction]
#[inline]
pub fn k_truss(graph: &Graph, k: usize) -> PyResult<Graph> {
    if k < 3 {
        // k-truss is only defined for k >= 3
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k must be >= 3 for k-truss"
        ));
    }
    
    let min_triangles = k.saturating_sub(2);
    
    // Start with a copy - create new graph with all nodes and edges
    let mut working_graph = Graph::new_py(graph.is_directed(), graph.is_weighted());
    
    // Copy all nodes
    for node_id in graph.get_node_ids() {
        if let Some(node) = graph.get_node(node_id) {
            let new_node = Node::new_py(node_id, node.label.clone(), node.attributes.clone());
            working_graph.add_node_py(new_node)?;
        }
    }
    
    // Copy all edges
    for edge in graph.get_edges_internal() {
        let new_edge = Edge::new_py(
            edge.id,
            edge.source,
            edge.target,
            edge.weight,
            edge.attributes.clone(),
        );
        working_graph.add_edge_py(new_edge)?;
    }
    
    // Iteratively remove edges with insufficient triangle support
    let mut changed = true;
    while changed {
        changed = false;
        
        // Calculate triangle count for each edge
        let mut edges_to_remove: Vec<u64> = Vec::new();
        
        // Get all edges
        let edges = working_graph.get_edges_internal().clone();
        for edge in edges {
            let node_id = edge.source;
            let neighbor_id = edge.target;
            
            // For undirected graphs, avoid double processing
            if !working_graph.is_directed() && node_id > neighbor_id {
                continue;
            }
            
            // Count triangles containing this edge
            let triangle_count = count_triangles_containing_edge(
                &working_graph,
                node_id,
                neighbor_id,
            )?;
            
            if triangle_count < min_triangles {
                edges_to_remove.push(edge.id);
                changed = true;
            }
        }
        
        // Remove edges with insufficient support
        for edge_id in edges_to_remove {
            working_graph.remove_edge_py(edge_id)?;
        }
    }
    
    // Remove isolated nodes (nodes with no edges)
    let node_ids = working_graph.get_node_ids();
    let mut isolated_nodes = Vec::new();
    
    for &node_id in &node_ids {
        if working_graph.degree_py(node_id) == 0 {
            isolated_nodes.push(node_id);
        }
    }
    
    for node_id in isolated_nodes {
        working_graph.remove_node_py(node_id)?;
    }
    
    Ok(working_graph)
}

/// Helper function: Count triangles containing a specific edge
#[inline]
fn count_triangles_containing_edge(graph: &Graph, u: u64, v: u64) -> PyResult<usize> {
    // A triangle exists when u-v is connected and both u and v share a common neighbor w
    let mut count = 0;
    
    // Get neighbors of u and v
    let neighbors_u = graph.neighbors(u)?;
    let neighbors_v = graph.neighbors(v)?;
    
    let neighbors_u_set: FxHashSet<u64> = neighbors_u.into_iter().collect();
    let neighbors_v_set: FxHashSet<u64> = neighbors_v.into_iter().collect();
    
    // Count common neighbors (excluding u and v themselves)
    for &neighbor in &neighbors_u_set {
        if neighbor != v && neighbor != u && neighbors_v_set.contains(&neighbor) {
            count += 1;
        }
    }
    
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_core_simple() {
        let mut graph = Graph::new_py(false, false);
        
        // Create a graph with clear k-core structure
        for i in 1..=6 {
            let node = Node::new_py(i, format!("{}", i), None);
            graph.add_node_py(node).unwrap();
        }
        
        // Core nodes 1,2,3 form a triangle (3-core)
        graph.add_edge_py(Edge::new_py(0, 1, 2, None, None)).unwrap();
        graph.add_edge_py(Edge::new_py(0, 1, 3, None, None)).unwrap();
        graph.add_edge_py(Edge::new_py(0, 2, 3, None, None)).unwrap();
        
        // Peripheral nodes
        graph.add_edge_py(Edge::new_py(0, 1, 4, None, None)).unwrap();
        graph.add_edge_py(Edge::new_py(0, 2, 5, None, None)).unwrap();
        graph.add_edge_py(Edge::new_py(0, 3, 6, None, None)).unwrap();
        
        let core2 = k_core(&graph, 2).unwrap();
        assert!(core2.has_node_id(1));
        assert!(core2.has_node_id(2));
        assert!(core2.has_node_id(3));
        
        let core3 = k_core(&graph, 3).unwrap();
        assert_eq!(core3.get_node_ids().len(), 3);
    }

    #[test]
    fn test_k_truss_simple() {
        let mut graph = Graph::new_py(false, false);
        
        // Create a graph with triangles
        for i in 1..=4 {
            let node = Node::new_py(i, format!("{}", i), None);
            graph.add_node_py(node).unwrap();
        }
        
        // Triangle 1-2-3
        graph.add_edge_py(Edge::new_py(0, 1, 2, None, None)).unwrap();
        graph.add_edge_py(Edge::new_py(0, 1, 3, None, None)).unwrap();
        graph.add_edge_py(Edge::new_py(0, 2, 3, None, None)).unwrap();
        
        // Extra edge to node 4 (not in triangle)
        graph.add_edge_py(Edge::new_py(0, 1, 4, None, None)).unwrap();
        
        let truss3 = k_truss(&graph, 3).unwrap();
        
        // 3-truss should contain the triangle
        assert!(truss3.has_node_id(1));
        assert!(truss3.has_node_id(2));
        assert!(truss3.has_node_id(3));
        
        // Node 4 should not be in 3-truss (edge 1-4 not in any triangle)
        assert!(!truss3.has_node_id(4));
    }
}


/// Extract k-core subgraph
///
/// The k-core is the maximal subgraph where all nodes have degree >= k.
/// Returns a new Graph containing only the nodes in the k-core.
#[pyfunction]
#[inline]
pub fn k_core(graph: &Graph, k: usize) -> PyResult<Graph> {
    // Get coreness for all nodes
    let coreness = graph.get_coreness();
    
    // Find nodes with coreness >= k
    let core_nodes: FxHashSet<u64> = coreness
        .iter()
        .filter(|(_, &c)| c >= k)
        .map(|(&node, _)| node)
        .collect();
    
    if core_nodes.is_empty() {
        // Return empty graph with same properties
        return Ok(Graph::new(graph.is_directed(), graph.is_weighted()));
    }
    
    // Create subgraph with only core nodes
    let mut subgraph = Graph::new(graph.is_directed(), graph.is_weighted());
    
    // Add nodes
    for &node_id in &core_nodes {
        if let Some(node) = graph.get_node(node_id) {
            let new_node = Node::new(node.id, node.label.clone(), node.attributes.clone());
            subgraph.add_node(new_node)?;
        }
    }
    
    // Add edges between core nodes
    let all_edges = graph.get_edges();
    for edge in all_edges {
        let source = edge.source;
        let target = edge.target;
        
        if core_nodes.contains(&source) && core_nodes.contains(&target) {
            let new_edge = Edge::new(
                edge.id,
                edge.source,
                edge.target,
                edge.weight,
                edge.attributes.clone(),
            );
            // Handle potential duplicate edges for undirected graphs
            if subgraph.get_edge(edge.id).is_none() {
                subgraph.add_edge(new_edge)?;
            }
        }
    }
    
    Ok(subgraph)
}

/// Extract k-shell subgraph
///
/// The k-shell is the set of nodes with coreness exactly equal to k.
/// Returns a new Graph containing only nodes in the k-shell.
#[pyfunction]
#[inline]
pub fn k_shell(graph: &Graph, k: usize) -> PyResult<Graph> {
    // Get coreness for all nodes
    let coreness = graph.get_coreness();
    
    // Find nodes with coreness exactly equal to k
    let shell_nodes: FxHashSet<u64> = coreness
        .iter()
        .filter(|(_, &c)| c == k)
        .map(|(&node, _)| node)
        .collect();
    
    if shell_nodes.is_empty() {
        // Return empty graph with same properties
        return Ok(Graph::new(graph.is_directed(), graph.is_weighted()));
    }
    
    // Create subgraph with only shell nodes
    let mut subgraph = Graph::new(graph.is_directed(), graph.is_weighted());
    
    // Add nodes
    for &node_id in &shell_nodes {
        if let Some(node) = graph.get_node(node_id) {
            let new_node = Node::new(node.id, node.label.clone(), node.attributes.clone());
            subgraph.add_node(new_node)?;
        }
    }
    
    // Add edges between shell nodes
    let all_edges = graph.get_edges();
    for edge in all_edges {
        let source = edge.source;
        let target = edge.target;
        
        if shell_nodes.contains(&source) && shell_nodes.contains(&target) {
            let new_edge = Edge::new(
                edge.id,
                edge.source,
                edge.target,
                edge.weight,
                edge.attributes.clone(),
            );
            if subgraph.get_edge(edge.id).is_none() {
                subgraph.add_edge(new_edge)?;
            }
        }
    }
    
    Ok(subgraph)
}

/// Extract k-crust subgraph
///
/// The k-crust is the complement of the (k+1)-core, i.e., nodes with coreness <= k.
/// Returns a new Graph containing only nodes in the k-crust.
#[pyfunction]
#[inline]
pub fn k_crust(graph: &Graph, k: usize) -> PyResult<Graph> {
    // Get coreness for all nodes
    let coreness = graph.get_coreness();
    
    // Find nodes with coreness <= k
    let crust_nodes: FxHashSet<u64> = coreness
        .iter()
        .filter(|(_, &c)| c <= k)
        .map(|(&node, _)| node)
        .collect();
    
    if crust_nodes.is_empty() {
        // Return empty graph with same properties
        return Ok(Graph::new(graph.is_directed(), graph.is_weighted()));
    }
    
    // Create subgraph with only crust nodes
    let mut subgraph = Graph::new(graph.is_directed(), graph.is_weighted());
    
    // Add nodes
    for &node_id in &crust_nodes {
        if let Some(node) = graph.get_node(node_id) {
            let new_node = Node::new(node.id, node.label.clone(), node.attributes.clone());
            subgraph.add_node(new_node)?;
        }
    }
    
    // Add edges between crust nodes
    let all_edges = graph.get_edges();
    for edge in all_edges {
        let source = edge.source;
        let target = edge.target;
        
        if crust_nodes.contains(&source) && crust_nodes.contains(&target) {
            let new_edge = Edge::new(
                edge.id,
                edge.source,
                edge.target,
                edge.weight,
                edge.attributes.clone(),
            );
            if subgraph.get_edge(edge.id).is_none() {
                subgraph.add_edge(new_edge)?;
            }
        }
    }
    
    Ok(subgraph)
}

/// Extract k-truss subgraph
///
/// The k-truss is the maximal subgraph where every edge is part of at least (k-2) triangles.
/// Uses iterative edge pruning: repeatedly removes edges with fewer than (k-2) triangles.
/// More restrictive than k-core: k-truss ⊆ k-core for the same k.
#[pyfunction]
#[inline]
pub fn k_truss(graph: &Graph, k: usize) -> PyResult<Graph> {
    if k < 3 {
        // k-truss is only defined for k >= 3
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k must be >= 3 for k-truss"
        ));
    }
    
    let min_triangles = k.saturating_sub(2);
    
    // Start with a copy - create new graph with all nodes and edges
    let mut working_graph = Graph::new(graph.is_directed(), graph.is_weighted());
    
    // Copy all nodes
    for node in graph.get_nodes() {
        let new_node = Node::new(node.id, node.label.clone(), node.attributes.clone());
        working_graph.add_node(new_node)?;
    }
    
    // Copy all edges
    for edge in graph.get_edges() {
        let new_edge = Edge::new(
            edge.id,
            edge.source,
            edge.target,
            edge.weight,
            edge.attributes.clone(),
        );
        working_graph.add_edge(new_edge)?;
    }
    
    // Iteratively remove edges with insufficient triangle support
    let mut changed = true;
    while changed {
        changed = false;
        
        // Calculate triangle count for each edge
        let mut edges_to_remove: Vec<u64> = Vec::new();
        
        // Get all edges
        let edges = working_graph.get_edges();
        for edge in edges {
            let node_id = edge.source;
            let neighbor_id = edge.target;
            
            // For undirected graphs, avoid double processing
            if !working_graph.is_directed() && node_id > neighbor_id {
                continue;
            }
            
            // Count triangles containing this edge
            let triangle_count = count_triangles_containing_edge(
                &working_graph,
                node_id,
                neighbor_id,
            )?;
            
            if triangle_count < min_triangles {
                edges_to_remove.push(edge.id);
                changed = true;
            }
        }
        
        // Remove edges with insufficient support
        for edge_id in edges_to_remove {
            working_graph.remove_edge(edge_id)?;
        }
    }
    
    // Remove isolated nodes (nodes with no edges)
    let node_ids = working_graph.get_node_ids();
    let mut isolated_nodes = Vec::new();
    
    for &node_id in &node_ids {
        if working_graph.degree(node_id) == 0 {
            isolated_nodes.push(node_id);
        }
    }
    
    for node_id in isolated_nodes {
        working_graph.remove_node(node_id)?;
    }
    
    Ok(working_graph)
}

/// Helper function: Count triangles containing a specific edge
#[inline]
fn count_triangles_containing_edge(graph: &Graph, u: u64, v: u64) -> PyResult<usize> {
    // A triangle exists when u-v is connected and both u and v share a common neighbor w
    let mut count = 0;
    
    // Get neighbors of u and v
    let neighbors_u = graph.neighbors(u)?;
    let neighbors_v = graph.neighbors(v)?;
    
    let neighbors_u_set: FxHashSet<u64> = neighbors_u.into_iter().collect();
    let neighbors_v_set: FxHashSet<u64> = neighbors_v.into_iter().collect();
    
    // Count common neighbors (excluding u and v themselves)
    for &neighbor in &neighbors_u_set {
        if neighbor != v && neighbor != u && neighbors_v_set.contains(&neighbor) {
            count += 1;
        }
    }
    
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_core_simple() {
        let mut graph = Graph::new(false, false);
        
        // Create a graph with clear k-core structure
        for i in 1..=6 {
            let node = Node::new(i, format!("{}", i), None);
            graph.add_node(node).unwrap();
        }
        
        // Core nodes 1,2,3 form a triangle (3-core)
        graph.add_edge(Edge::new(0, 1, 2, None, None)).unwrap();
        graph.add_edge(Edge::new(0, 1, 3, None, None)).unwrap();
        graph.add_edge(Edge::new(0, 2, 3, None, None)).unwrap();
        
        // Peripheral nodes
        graph.add_edge(Edge::new(0, 1, 4, None, None)).unwrap();
        graph.add_edge(Edge::new(0, 2, 5, None, None)).unwrap();
        graph.add_edge(Edge::new(0, 3, 6, None, None)).unwrap();
        
        let core2 = k_core(&graph, 2).unwrap();
        assert!(core2.has_node_id(1));
        assert!(core2.has_node_id(2));
        assert!(core2.has_node_id(3));
        
        let core3 = k_core(&graph, 3).unwrap();
        assert_eq!(core3.get_node_ids().len(), 3);
    }

    #[test]
    fn test_k_truss_simple() {
        let mut graph = Graph::new(false, false);
        
        // Create a graph with triangles
        for i in 1..=4 {
            let node = Node::new(i, format!("{}", i), None);
            graph.add_node(node).unwrap();
        }
        
        // Triangle 1-2-3
        graph.add_edge(Edge::new(0, 1, 2, None, None)).unwrap();
        graph.add_edge(Edge::new(0, 1, 3, None, None)).unwrap();
        graph.add_edge(Edge::new(0, 2, 3, None, None)).unwrap();
        
        // Extra edge to node 4 (not in triangle)
        graph.add_edge(Edge::new(0, 1, 4, None, None)).unwrap();
        
        let truss3 = k_truss(&graph, 3).unwrap();
        
        // 3-truss should contain the triangle
        assert!(truss3.has_node_id(1));
        assert!(truss3.has_node_id(2));
        assert!(truss3.has_node_id(3));
        
        // Node 4 should not be in 3-truss (edge 1-4 not in any triangle)
        assert!(!truss3.has_node_id(4));
    }
}
