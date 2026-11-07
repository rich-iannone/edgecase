//! Distance and spectral metrics for graphs.
//!
//! This module provides:
//! - `wiener_index`: Sum of all pairwise shortest path distances
//! - `hyper_wiener_index`: Half the sum of d + d² over all node pairs
//! - `estrada_index`: Sum of exponentials of adjacency matrix eigenvalues
//! - `resistance_distance`: Effective resistance between node pairs
//!
//! References:
//! - Wiener, H. (1947). Structural determination of paraffin boiling points.
//!   Journal of the American Chemical Society, 69(1), 17-20.
//! - Estrada, E. (2000). Characterization of 3D molecular structure.
//!   Chemical Physics Letters, 319(5-6), 713-718.
//! - Klein, D. J. & Randić, M. (1993). Resistance Distance.
//!   J. Math. Chem. 12: 81-95.

use crate::graph::Graph;
use nalgebra::{DMatrix, SymmetricEigen};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Compute the Wiener index of a graph.
///
/// The Wiener index is the sum of all pairwise shortest path distances:
/// W(G) = Σ_{u < v} d(u, v)
///
/// For disconnected graphs, returns infinity (f64::INFINITY) since some
/// pairs have infinite distance.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `weight` - Optional edge weight attribute. If None, uses unweighted distances.
///
/// # Returns
/// The Wiener index as a float. Returns 0.0 for empty graphs or single-node graphs.
/// Returns infinity for disconnected graphs.
///
/// # Notes
/// - For undirected graphs, each pair is counted once (u < v)
/// - For directed graphs, considers all ordered pairs (u, v) where u != v
/// - Time complexity: O(V * (V + E)) for unweighted, O(V * (V + E) log V) for weighted
#[pyfunction]
#[pyo3(signature = (graph, weight=None))]
pub fn wiener_index(graph: &Graph, weight: Option<&str>) -> PyResult<f64> {
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n <= 1 {
        return Ok(0.0);
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    let use_weights = weight.is_some() && graph.is_weighted();
    let is_directed = graph.is_directed();

    let mut total_distance = 0.0;

    // For each source node, compute distances to all other nodes
    for (i, &source) in node_ids.iter().enumerate() {
        let distances = if use_weights {
            single_source_dijkstra(graph, source, &node_ids, &node_to_idx)
        } else {
            single_source_bfs(graph, source, &node_ids, &node_to_idx)
        };

        // Sum distances
        if is_directed {
            // For directed graphs, sum all distances to other nodes
            for (j, &dist) in distances.iter().enumerate() {
                if i != j {
                    if dist.is_infinite() {
                        return Ok(f64::INFINITY);
                    }
                    total_distance += dist;
                }
            }
        } else {
            // For undirected graphs, only count pairs where i < j to avoid double counting
            for &dist in &distances[i + 1..n] {
                if dist.is_infinite() {
                    return Ok(f64::INFINITY);
                }
                total_distance += dist;
            }
        }
    }

    Ok(total_distance)
}

/// Compute the hyper-Wiener index of an undirected graph (Randić 1993; Klein, Lukovits &
/// Gutman 1995): WW(G) = ½ Σ_{u < v} (d(u, v) + d(u, v)²).
///
/// Returns 0.0 for graphs with fewer than two nodes and infinity for disconnected graphs.
/// `weight` switches to weighted distances on a weighted graph, as in `wiener_index`.
#[pyfunction]
#[pyo3(signature = (graph, weight=None))]
pub fn hyper_wiener_index(graph: &Graph, weight: Option<&str>) -> PyResult<f64> {
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the hyper-Wiener index is defined for undirected graphs",
        ));
    }
    let node_ids = graph.get_node_ids();
    let n = node_ids.len();
    if n <= 1 {
        return Ok(0.0);
    }
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();
    let use_weights = weight.is_some() && graph.is_weighted();
    let mut total = 0.0;
    for (i, &source) in node_ids.iter().enumerate() {
        let distances = if use_weights {
            single_source_dijkstra(graph, source, &node_ids, &node_to_idx)
        } else {
            single_source_bfs(graph, source, &node_ids, &node_to_idx)
        };
        for &d in &distances[i + 1..n] {
            if d.is_infinite() {
                return Ok(f64::INFINITY);
            }
            total += d + d * d;
        }
    }
    Ok(total / 2.0)
}

/// Helper: BFS for unweighted single-source shortest paths
fn single_source_bfs(
    graph: &Graph,
    source: u64,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
) -> Vec<f64> {
    let n = node_ids.len();
    let mut distances = vec![f64::INFINITY; n];
    let source_idx = node_to_idx[&source];
    distances[source_idx] = 0.0;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(current) = queue.pop_front() {
        let current_idx = node_to_idx[&current];
        let current_dist = distances[current_idx];

        if let Ok(neighbors) = graph.neighbors(current) {
            for neighbor in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                    if distances[neighbor_idx].is_infinite() {
                        distances[neighbor_idx] = current_dist + 1.0;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }

    distances
}

/// Get edge weight; `Graph::get_edge_weight()` already considers both orientations of an
/// undirected edge and takes the cheapest parallel edge
#[inline]
fn get_edge_weight_bidirectional(graph: &Graph, from: u64, to: u64) -> f64 {
    graph.get_edge_weight(from, to)
}

/// Helper: Dijkstra for weighted single-source shortest paths
fn single_source_dijkstra(
    graph: &Graph,
    source: u64,
    node_ids: &[u64],
    node_to_idx: &FxHashMap<u64, usize>,
) -> Vec<f64> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct State {
        cost: f64,
        node_idx: usize,
    }

    impl Eq for State {}

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .cost
                .partial_cmp(&self.cost)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = node_ids.len();
    let mut distances = vec![f64::INFINITY; n];
    let source_idx = node_to_idx[&source];
    distances[source_idx] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(State {
        cost: 0.0,
        node_idx: source_idx,
    });

    while let Some(State { cost, node_idx }) = heap.pop() {
        if cost > distances[node_idx] {
            continue;
        }

        let node = node_ids[node_idx];
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                    let edge_weight = get_edge_weight_bidirectional(graph, node, neighbor);
                    let new_cost = cost + edge_weight;

                    if new_cost < distances[neighbor_idx] {
                        distances[neighbor_idx] = new_cost;
                        heap.push(State {
                            cost: new_cost,
                            node_idx: neighbor_idx,
                        });
                    }
                }
            }
        }
    }

    distances
}

/// Compute the Estrada index of a graph.
///
/// The Estrada index is defined as the sum of exponentials of the eigenvalues
/// of the adjacency matrix:
/// EE(G) = Σ_i exp(λ_i)
///
/// where λ_i are the eigenvalues of the adjacency matrix.
///
/// The Estrada index is related to the subgraph centrality and counts closed
/// walks of all lengths, with shorter walks weighted more heavily.
///
/// # Arguments
/// * `graph` - The graph to analyze (must be undirected)
///
/// # Returns
/// The Estrada index as a float. Returns 0.0 for empty graphs.
///
/// # Errors
/// Returns an error if the graph is directed.
///
/// # Notes
/// - The Estrada index is always >= n (number of nodes) for any graph
/// - For a complete graph Kn: EE = (n-1)*exp(-1) + exp(n-1)
/// - For an empty graph (no edges): EE = n (all eigenvalues are 0)
/// - Time complexity: O(n³) for eigendecomposition
///
/// # References
/// - Estrada, E. (2000). Characterization of 3D molecular structure.
///   Chemical Physics Letters, 319(5-6), 713-718.
/// - Estrada, E., & Rodríguez-Velázquez, J. A. (2005). Subgraph centrality
///   in complex networks. Physical Review E, 71(5), 056103.
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn estrada_index(graph: &Graph) -> PyResult<f64> {
    // Validate graph type
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Estrada index is only defined for undirected graphs",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n == 0 {
        return Ok(0.0);
    }

    // Single node with no edges: eigenvalue is 0, exp(0) = 1
    if n == 1 {
        return Ok(1.0);
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Build adjacency matrix
    let mut adj_matrix = DMatrix::<f64>::zeros(n, n);

    for edge in graph.get_edges_internal() {
        if let (Some(&i), Some(&j)) = (node_to_idx.get(&edge.source), node_to_idx.get(&edge.target))
        {
            // Skip self-loops for Estrada index computation
            if i != j {
                // Undirected: symmetric
                adj_matrix[(i, j)] = 1.0;
                adj_matrix[(j, i)] = 1.0;
            }
        }
    }

    // Compute eigendecomposition
    let eigen = SymmetricEigen::new(adj_matrix);
    let eigenvalues = &eigen.eigenvalues;

    // Sum exp(λ_i) for all eigenvalues
    let estrada: f64 = eigenvalues.iter().map(|&lambda| lambda.exp()).sum();

    Ok(estrada)
}

/// Laplacian structure shared by the resistance-distance functions.
struct ResistanceSetup {
    node_ids: Vec<u64>,
    node_to_idx: FxHashMap<u64, usize>,
    /// Component label of each node index
    component: Vec<usize>,
    /// Edge multiplicities between distinct node indices (parallel edges are parallel resistors)
    edges: Vec<(usize, usize)>,
}

impl ResistanceSetup {
    fn new(graph: &Graph) -> Self {
        let node_ids = graph.get_node_ids();
        let n = node_ids.len();
        let node_to_idx: FxHashMap<u64, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx))
            .collect();

        // Self-loops carry no current, so they're skipped
        let edges: Vec<(usize, usize)> = graph
            .get_edges_internal()
            .iter()
            .filter_map(|edge| {
                match (node_to_idx.get(&edge.source), node_to_idx.get(&edge.target)) {
                    (Some(&i), Some(&j)) if i != j => Some((i, j)),
                    _ => None,
                }
            })
            .collect();

        // Union-find for connected components
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for &(i, j) in &edges {
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri != rj {
                parent[ri] = rj;
            }
        }
        let component = (0..n).map(|i| find(&mut parent, i)).collect();

        ResistanceSetup {
            node_ids,
            node_to_idx,
            component,
            edges,
        }
    }

    /// Indices of the nodes in the component with label `label`
    fn members(&self, label: usize) -> Vec<usize> {
        (0..self.node_ids.len())
            .filter(|&i| self.component[i] == label)
            .collect()
    }

    /// Γ = (L + J/m)⁻¹ for the component made of `members` (m nodes), indexed by position in
    /// `members`. L + J/m is symmetric positive definite for a connected component, so a
    /// Cholesky inverse is exact; a pseudoinverse via eigendecomposition loses accuracy when
    /// the Laplacian has repeated eigenvalues (e.g. ~1e-4 error on a 5-cycle).
    fn component_gamma(&self, members: &[usize]) -> PyResult<DMatrix<f64>> {
        let m = members.len();
        let local: FxHashMap<usize, usize> =
            members.iter().enumerate().map(|(k, &i)| (i, k)).collect();

        let mut laplacian = DMatrix::<f64>::from_element(m, m, 1.0 / m as f64);
        for &(i, j) in &self.edges {
            if let (Some(&a), Some(&b)) = (local.get(&i), local.get(&j)) {
                laplacian[(a, b)] -= 1.0;
                laplacian[(b, a)] -= 1.0;
                laplacian[(a, a)] += 1.0;
                laplacian[(b, b)] += 1.0;
            }
        }

        laplacian
            .cholesky()
            .map(|chol| chol.inverse())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "Resistance distance computation failed: Laplacian system is not positive definite",
                )
            })
    }
}

/// Compute the resistance distance between two nodes in a graph.
///
/// The resistance distance (also called effective resistance) is the electrical
/// resistance between two nodes if each edge is replaced by a 1-ohm resistor:
///
/// Ω(i,j) = Γ(i,i) + Γ(j,j) - 2*Γ(i,j)
///
/// where Γ = (L + (1/m)J)⁻¹ for the m-node connected component holding i and j, L is its
/// Laplacian matrix, and J is the all-ones matrix.
///
/// # Arguments
/// * `graph` - The graph to analyze (must be undirected)
/// * `source` - Source node ID
/// * `target` - Target node ID
///
/// # Returns
/// The resistance distance as a float; infinity if the nodes are in different components.
///
/// # Errors
/// Returns an error if:
/// - The graph is directed
/// - Either source or target node doesn't exist
///
/// # Notes
/// - For adjacent nodes connected by a single edge: Ω(i,j) ≤ 1
/// - Resistance distance satisfies the triangle inequality
/// - The sum of all resistance distances is the Kirchhoff index
/// - Time complexity: O(m³) for the component containing the two nodes
///
/// # References
/// - Klein, D. J. & Randić, M. (1993). Resistance Distance. J. Math. Chem. 12: 81-95.
#[pyfunction]
#[pyo3(signature = (graph, source, target))]
pub fn resistance_distance(graph: &Graph, source: u64, target: u64) -> PyResult<f64> {
    // Validate graph type
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Resistance distance is only defined for undirected graphs",
        ));
    }

    // Validate nodes exist
    if !graph.has_node_id(source) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Source node {} does not exist in the graph",
            source
        )));
    }
    if !graph.has_node_id(target) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Target node {} does not exist in the graph",
            target
        )));
    }

    // Same node has zero resistance distance
    if source == target {
        return Ok(0.0);
    }

    let setup = ResistanceSetup::new(graph);
    let source_idx = setup.node_to_idx[&source];
    let target_idx = setup.node_to_idx[&target];

    // No path means no current can flow: the resistance is infinite
    if setup.component[source_idx] != setup.component[target_idx] {
        return Ok(f64::INFINITY);
    }

    let members = setup.members(setup.component[source_idx]);
    let gamma = setup.component_gamma(&members)?;
    let s = members.iter().position(|&i| i == source_idx).unwrap();
    let t = members.iter().position(|&i| i == target_idx).unwrap();

    let resistance = gamma[(s, s)] + gamma[(t, t)] - 2.0 * gamma[(s, t)];
    Ok(resistance.max(0.0))
}

/// Compute the resistance distance matrix for all node pairs.
///
/// Returns an n×n matrix where entry (i,j) is the resistance distance between
/// nodes i and j. The matrix is symmetric with zeros on the diagonal.
///
/// # Arguments
/// * `graph` - The graph to analyze (must be undirected)
///
/// # Returns
/// A dictionary mapping (source_id, target_id) tuples to resistance distances.
///
/// # Errors
/// Returns an error if the graph is directed.
///
/// # Notes
/// - Time complexity: O(Σ m³) over the connected components (sizes m)
/// - For disconnected graphs, distances between components are infinite
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn resistance_distance_matrix<'py>(
    py: Python<'py>,
    graph: &Graph,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    // Validate graph type
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Resistance distance is only defined for undirected graphs",
        ));
    }

    let setup = ResistanceSetup::new(graph);
    let n = setup.node_ids.len();

    // Every pair starts infinite (different components) and same-node pairs are zero;
    // pairs inside a component are filled in from that component's Γ
    let mut matrix = vec![f64::INFINITY; n * n];
    for i in 0..n {
        matrix[i * n + i] = 0.0;
    }

    let mut labels: Vec<usize> = setup.component.clone();
    labels.sort_unstable();
    labels.dedup();

    for label in labels {
        let members = setup.members(label);
        if members.len() < 2 {
            continue;
        }
        let gamma = setup.component_gamma(&members)?;
        for (a, &i) in members.iter().enumerate() {
            for (b, &j) in members.iter().enumerate() {
                if a != b {
                    let resistance = (gamma[(a, a)] + gamma[(b, b)] - 2.0 * gamma[(a, b)]).max(0.0);
                    matrix[i * n + j] = resistance;
                }
            }
        }
    }

    // Built in node order rather than from a HashMap, so the key order is the same every call
    let result = pyo3::types::PyDict::new(py);
    for (i, &node_i) in setup.node_ids.iter().enumerate() {
        for (j, &node_j) in setup.node_ids.iter().enumerate() {
            result.set_item((node_i, node_j), matrix[i * n + j])?;
        }
    }
    Ok(result)
}

/// Compute the Kirchhoff index (total resistance) of a graph.
///
/// The Kirchhoff index is the sum of all pairwise resistance distances:
/// Kf(G) = Σ_{i < j} Ω(i,j)
///
/// It can also be computed from the Laplacian eigenvalues:
/// Kf(G) = n * Σ_{k=1}^{n-1} (1/λ_k)
///
/// where λ_k are the non-zero eigenvalues of the Laplacian.
///
/// # Arguments
/// * `graph` - The graph to analyze (must be undirected and connected)
///
/// # Returns
/// The Kirchhoff index as a float. Returns 0.0 for single-node graphs.
/// Returns infinity for disconnected graphs.
///
/// # Errors
/// Returns an error if the graph is directed.
///
/// # Notes
/// - For a path graph Pn: Kf = n(n²-1)/6
/// - For a complete graph Kn: Kf = n-1
/// - For a cycle graph Cn: Kf = n(n²-1)/12
/// - Time complexity: O(n³) for eigendecomposition
///
/// # References
/// - Gutman, I. & Mohar, B. (1996). The quasi-Wiener and the Kirchhoff indices coincide.
///   J. Chem. Inf. Comput. Sci. 36(5): 982-985.
#[pyfunction]
#[pyo3(signature = (graph))]
pub fn kirchhoff_index(graph: &Graph) -> PyResult<f64> {
    // Validate graph type
    if graph.is_directed() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kirchhoff index is only defined for undirected graphs",
        ));
    }

    let node_ids = graph.get_node_ids();
    let n = node_ids.len();

    if n <= 1 {
        return Ok(0.0);
    }

    // Create node_id to index mapping
    let node_to_idx: FxHashMap<u64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(idx, &node_id)| (node_id, idx))
        .collect();

    // Build Laplacian matrix
    let mut laplacian = DMatrix::<f64>::zeros(n, n);

    for edge in graph.get_edges_internal() {
        if let (Some(&i), Some(&j)) = (node_to_idx.get(&edge.source), node_to_idx.get(&edge.target))
        {
            if i != j {
                laplacian[(i, j)] -= 1.0;
                laplacian[(j, i)] -= 1.0;
                laplacian[(i, i)] += 1.0;
                laplacian[(j, j)] += 1.0;
            }
        }
    }

    // Compute eigenvalues of Laplacian
    let eigen = SymmetricEigen::new(laplacian);
    let eigenvalues = &eigen.eigenvalues;

    // Threshold for considering an eigenvalue as zero
    let threshold = 1e-10;

    // Kirchhoff index = n * Σ(1/λ_k) for non-zero eigenvalues
    let mut sum_inv_eigenvalues = 0.0;
    let mut zero_eigenvalue_count = 0;

    for &lambda in eigenvalues.iter() {
        if lambda.abs() < threshold {
            zero_eigenvalue_count += 1;
        } else {
            sum_inv_eigenvalues += 1.0 / lambda;
        }
    }

    // For a connected graph, there should be exactly one zero eigenvalue
    // More than one indicates disconnected components
    if zero_eigenvalue_count > 1 {
        return Ok(f64::INFINITY);
    }

    Ok((n as f64) * sum_inv_eigenvalues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Node};

    // Helper graphs use node ids 1..=n (node id 0 is reserved for auto-assignment),
    // so the conceptual node k in the comments below has id k + 1.
    fn create_path_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        for i in 0..n {
            let node = Node {
                id: i as u64 + 1,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        for i in 0..(n - 1) {
            let edge = Edge {
                id: 0,
                source: i as u64 + 1,
                target: (i + 2) as u64,
                weight: None,
                attributes: None,
            };
            graph.add_edge_rust(edge).unwrap();
        }
        graph
    }

    fn create_complete_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        for i in 0..n {
            let node = Node {
                id: i as u64 + 1,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let edge = Edge {
                    id: 0,
                    source: i as u64 + 1,
                    target: j as u64 + 1,
                    weight: None,
                    attributes: None,
                };
                graph.add_edge_rust(edge).unwrap();
            }
        }
        graph
    }

    fn create_cycle_graph(n: usize) -> Graph {
        let mut graph = Graph::new_rust(false, false);
        for i in 0..n {
            let node = Node {
                id: i as u64 + 1,
                label: None,
                attributes: None,
            };
            graph.add_node_rust(node).unwrap();
        }
        for i in 0..n {
            let edge = Edge {
                id: 0,
                source: i as u64 + 1,
                target: ((i + 1) % n) as u64 + 1,
                weight: None,
                attributes: None,
            };
            graph.add_edge_rust(edge).unwrap();
        }
        graph
    }

    #[test]
    fn test_wiener_index_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_wiener_index_single_node() {
        let mut graph = Graph::new_rust(false, false);
        let node = Node {
            id: 1,
            label: None,
            attributes: None,
        };
        graph.add_node_rust(node).unwrap();
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_wiener_index_path_graph() {
        // Path P4: 0-1-2-3
        // Distances: d(0,1)=1, d(0,2)=2, d(0,3)=3, d(1,2)=1, d(1,3)=2, d(2,3)=1
        // Wiener index = 1+2+3+1+2+1 = 10
        let graph = create_path_graph(4);
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_wiener_index_complete_graph() {
        // Complete K4: all pairs have distance 1
        // Number of pairs = 4*3/2 = 6
        // Wiener index = 6
        let graph = create_complete_graph(4);
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_wiener_index_cycle_graph() {
        // Cycle C4: 0-1-2-3-0
        // Distances: d(0,1)=1, d(0,2)=2, d(0,3)=1, d(1,2)=1, d(1,3)=2, d(2,3)=1
        // Wiener index = 1+2+1+1+2+1 = 8
        let graph = create_cycle_graph(4);
        let result = wiener_index(&graph, None).unwrap();
        assert_eq!(result, 8.0);
    }

    #[test]
    fn test_wiener_index_disconnected() {
        // Two isolated nodes
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(Node {
                id: 1,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_node_rust(Node {
                id: 2,
                label: None,
                attributes: None,
            })
            .unwrap();
        let result = wiener_index(&graph, None).unwrap();
        assert!(result.is_infinite());
    }

    #[test]
    fn test_estrada_index_empty_graph() {
        let graph = Graph::new_rust(false, false);
        let result = estrada_index(&graph).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_estrada_index_single_node() {
        let mut graph = Graph::new_rust(false, false);
        let node = Node {
            id: 1,
            label: None,
            attributes: None,
        };
        graph.add_node_rust(node).unwrap();
        let result = estrada_index(&graph).unwrap();
        assert_eq!(result, 1.0); // exp(0) = 1
    }

    #[test]
    fn test_estrada_index_two_nodes_no_edge() {
        // Two isolated nodes: both eigenvalues are 0
        // EE = exp(0) + exp(0) = 2
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(Node {
                id: 1,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_node_rust(Node {
                id: 2,
                label: None,
                attributes: None,
            })
            .unwrap();
        let result = estrada_index(&graph).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_estrada_index_two_nodes_with_edge() {
        // Edge graph K2: eigenvalues are 1 and -1
        // EE = exp(1) + exp(-1) ≈ 2.718 + 0.368 ≈ 3.086
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(Node {
                id: 1,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_node_rust(Node {
                id: 2,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_edge_rust(Edge {
                id: 0,
                source: 1,
                target: 2,
                weight: None,
                attributes: None,
            })
            .unwrap();
        let result = estrada_index(&graph).unwrap();
        let expected = 1.0_f64.exp() + (-1.0_f64).exp();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_estrada_index_path_graph() {
        // Path P3: eigenvalues are -√2, 0, √2
        // EE = exp(-√2) + exp(0) + exp(√2)
        let graph = create_path_graph(3);
        let result = estrada_index(&graph).unwrap();
        let sqrt2 = 2.0_f64.sqrt();
        let expected = (-sqrt2).exp() + 1.0 + sqrt2.exp();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_estrada_index_complete_graph() {
        // Complete K3: eigenvalues are 2, -1, -1
        // EE = exp(2) + exp(-1) + exp(-1)
        let graph = create_complete_graph(3);
        let result = estrada_index(&graph).unwrap();
        let expected = 2.0_f64.exp() + 2.0 * (-1.0_f64).exp();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_estrada_index_directed_error() {
        let graph = Graph::new_rust(true, false);
        let result = estrada_index(&graph);
        assert!(result.is_err());
    }

    // Resistance distance tests

    #[test]
    fn test_resistance_distance_same_node() {
        let graph = create_path_graph(3);
        let result = resistance_distance(&graph, 1, 1).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_resistance_distance_adjacent_nodes() {
        // Two adjacent nodes: resistance = 1.0
        let graph = create_path_graph(2);
        let result = resistance_distance(&graph, 1, 2).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_resistance_distance_path_endpoints() {
        // Path P3: 0-1-2
        // Ω(0,2) = 2.0 (two resistors in series)
        let graph = create_path_graph(3);
        let result = resistance_distance(&graph, 1, 3).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_resistance_distance_complete_graph() {
        // Complete K4: all pairs have resistance 1/2
        // (n-1 parallel paths, each with resistance 2)
        // For Kn: Ω(i,j) = 2/n for all i≠j
        let graph = create_complete_graph(4);
        let result = resistance_distance(&graph, 1, 2).unwrap();
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_resistance_distance_triangle() {
        // Triangle K3: Ω(i,j) = 2/3 for all pairs
        let graph = create_complete_graph(3);
        let result = resistance_distance(&graph, 1, 2).unwrap();
        assert!((result - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_resistance_distance_cycle() {
        // Cycle C4: 0-1-2-3-0
        // Ω(0,1) = 1 * 3 / (1 + 3) = 3/4 (parallel paths of length 1 and 3)
        let graph = create_cycle_graph(4);
        let result = resistance_distance(&graph, 1, 2).unwrap();
        assert!((result - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_resistance_distance_symmetric() {
        let graph = create_path_graph(4);
        let r01 = resistance_distance(&graph, 1, 2).unwrap();
        let r10 = resistance_distance(&graph, 2, 1).unwrap();
        assert!((r01 - r10).abs() < 1e-10);
    }

    #[test]
    fn test_resistance_distance_directed_error() {
        let graph = Graph::new_rust(true, false);
        let result = resistance_distance(&graph, 1, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_resistance_distance_node_not_found() {
        let graph = create_path_graph(3);
        let result = resistance_distance(&graph, 1, 99);
        assert!(result.is_err());
    }

    // Kirchhoff index tests

    #[test]
    fn test_kirchhoff_index_empty() {
        let graph = Graph::new_rust(false, false);
        let result = kirchhoff_index(&graph).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_kirchhoff_index_single_node() {
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(Node {
                id: 1,
                label: None,
                attributes: None,
            })
            .unwrap();
        let result = kirchhoff_index(&graph).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_kirchhoff_index_path() {
        // Path Pn: Kf = n(n²-1)/6
        // P4: Kf = 4*(16-1)/6 = 60/6 = 10
        let graph = create_path_graph(4);
        let result = kirchhoff_index(&graph).unwrap();
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_kirchhoff_index_complete() {
        // Complete Kn: Kf = n-1
        // K4: Kf = 3
        let graph = create_complete_graph(4);
        let result = kirchhoff_index(&graph).unwrap();
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_kirchhoff_index_cycle() {
        // Cycle Cn: Kf = n(n²-1)/12
        // C4: Kf = 4*(16-1)/12 = 60/12 = 5
        let graph = create_cycle_graph(4);
        let result = kirchhoff_index(&graph).unwrap();
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_kirchhoff_index_disconnected() {
        // Disconnected graph should return infinity
        let mut graph = Graph::new_rust(false, false);
        graph
            .add_node_rust(Node {
                id: 1,
                label: None,
                attributes: None,
            })
            .unwrap();
        graph
            .add_node_rust(Node {
                id: 2,
                label: None,
                attributes: None,
            })
            .unwrap();
        let result = kirchhoff_index(&graph).unwrap();
        assert!(result.is_infinite());
    }

    #[test]
    fn test_kirchhoff_index_directed_error() {
        let graph = Graph::new_rust(true, false);
        let result = kirchhoff_index(&graph);
        assert!(result.is_err());
    }
}
