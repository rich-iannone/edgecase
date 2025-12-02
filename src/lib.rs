//! EdgeCase: High-performance graph analysis library
//!
//! This is the Rust core that provides performance-critical graph algorithms
//! with Python bindings via PyO3.

use pyo3::prelude::*;

pub mod algorithms;
pub mod graph;
pub mod layout;
pub mod temporal;
pub mod utils;

/// Initialize the Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<graph::Graph>()?;
    m.add_class::<graph::Node>()?;
    m.add_class::<graph::Edge>()?;
    m.add_class::<graph::AdaStar>()?;

    // Algorithm functions
    m.add_function(wrap_pyfunction!(algorithms::centrality::betweenness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::load_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::second_order_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::average_neighbor_degree, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::harmonic_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community::detect_communities, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::find_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::simple_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::cycle_basis, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::minimum_cycle_basis, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::domirank::domirank, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::paths::shortest_path, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::paths::bellman_ford_shortest_path, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::paths::all_pairs_shortest_path_length, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::paths::all_pairs_shortest_path, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::paths::k_shortest_paths, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure::k_core, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure::k_shell, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure::k_crust, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure::k_truss, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure::k_corona, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching::is_matching, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching::maximal_matching, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching::max_weight_matching, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::eulerian::is_eulerian, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::eulerian::is_semieulerian, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::eulerian::eulerian_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::bridges::bridges, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::bridges::has_bridges, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::node_vitality::node_vitality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::regularity::is_regular, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::regularity::is_k_regular, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dispersion::dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::efficiency::global_efficiency, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::efficiency::local_efficiency, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::independent_set::maximal_independent_set, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::independent_set::maximum_independent_set, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::independent_set::is_independent_set, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::independent_set::find_maximal_independent_sets, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dominating_set::is_dominating_set, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dominating_set::dominating_set, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dominating_set::min_weighted_dominating_set, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::spanning_trees::steiner_tree, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::spanning_trees::all_spanning_trees, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::coloring::greedy_color, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::coloring::is_valid_coloring, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::coloring::chromatic_number_upper_bound, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::line_graph::line_graph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::line_graph::line_graph_node_mapping, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::flow::max_flow, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::flow::max_flow_value, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::flow::min_cost_flow, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::link_prediction::common_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::link_prediction::jaccard_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::link_prediction::adamic_adar_index, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::link_prediction::resource_allocation_index, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::link_prediction::preferential_attachment, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::contraction::contract_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::contraction::contracted_edge, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::contraction::contract_nodes_multi, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::contraction::quotient_graph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::graph_union, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::graph_intersection, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::graph_difference, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::graph_symmetric_difference, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::graph_compose, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::graph_disjoint_union, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::transitive_closure, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::transitive_reduction, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::edge_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::to_adjacency_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::graph_ops::from_adjacency_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::communicability::communicability, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::communicability::communicability_exp, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::communicability::communicability_betweenness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::percolation::percolation_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::percolation::node_percolation, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::percolation::edge_percolation, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::percolation::largest_component_size, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::percolation::percolation_threshold_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::distance_metrics::wiener_index, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::distance_metrics::estrada_index, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::distance_metrics::resistance_distance, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::distance_metrics::resistance_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::distance_metrics::kirchhoff_index, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::rich_club::rich_club_coefficient, m)?)?;

    // Layout functions
    m.add_function(wrap_pyfunction!(layout::force_directed_layout_rust, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_initialization() {
        // Test that the module can be initialized
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = PyModule::new_bound(py, "_core").unwrap();
            assert!(_core(&module).is_ok());
        });
    }
}
