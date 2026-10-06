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
    m.add_function(wrap_pyfunction!(algorithms::centrality::knotty_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::bridging_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::laplacian_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::closeness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::degree_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::edge_betweenness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_queries::multi_source_distances, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_queries::multi_source_paths, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_queries::shortest_path_length, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_queries::has_path, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_queries::all_shortest_paths, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dag::ancestors, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dag::descendants, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dag::topological_generations, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dag::dag_longest_path, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dag::condensation, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dag::lowest_common_ancestor, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community_detection::modularity, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community_detection::leiden, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community_detection::label_propagation_communities, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community_detection::k_clique_communities, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::betweenness_centrality_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_queries::bidirectional_dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::dag::all_topological_sorts, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community_detection::partition_quality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community_detection::greedy_modularity, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure_basics::biconnected_components, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure_basics::local_bridges, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure_basics::isolates, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure_basics::barycenter, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::structure_basics::nodes_within, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::traversal::bfs_search, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::traversal::dfs_search, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::traversal::tree_graph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::eulerian::eulerian_path, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::eulerian::has_eulerian_path, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::eulerian::eulerize, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cuts::minimum_cut, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cuts::stoer_wagner, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cuts::gomory_hu, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cuts::weighted_tree_graph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cuts::edge_disjoint_paths, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cuts::node_disjoint_paths, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cuts::triadic_census, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::subgraph_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::current_flow_betweenness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::current_flow_closeness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::group_betweenness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::group_closeness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::group_degree_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::voterank, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::trophic_levels, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::local_reaching_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::betweenness_centrality_subset, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::effective_size, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality_extra::local_constraint, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::mixing_cuts::square_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::mixing_cuts::generalized_degree, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::mixing_cuts::average_degree_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::mixing_cuts::degree_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::mixing_cuts::cut_profile, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::branchings::optimum_branching, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::branchings::spanning_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::branchings::edge_values, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::branchings::number_of_spanning_trees, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::branchings::random_spanning_tree, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::branchings::is_branching, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::branchings::to_prufer_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::branchings::from_prufer_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching_cover::blossom_max_weight_matching, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching_cover::hopcroft_karp_matching, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching_cover::bipartite_vertex_cover, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching_cover::min_weight_full_matching, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching_cover::min_edge_cover, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::matching_cover::min_weighted_vertex_cover, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::approximation::traveling_salesman, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::approximation::metric_closure, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::approximation::tree_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::approximation::densest_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::approximation::max_cut, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::isomorphism::match_graphs, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::isomorphism::simrank, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::chain_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::dominators, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::k_edge_components, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::average_node_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::reference_graph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::small_world, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::negative_edge_cycle, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::find_negative_cycle, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity_extras::link_scores, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::products::graph_product, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::products::graph_power, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_extras::shortest_path_predecessors, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_extras::floyd_warshall, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_extras::johnson, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_extras::astar, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_extras::simple_edge_paths, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::path_extras::shortest_simple_paths, m)?)?;
    m.add_class::<algorithms::path_extras::ShortestSimplePaths>()?;
    m.add_function(wrap_pyfunction!(algorithms::chordal::is_chordal, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::chordal::chordal_graph_cliques, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::chordal::complete_to_chordal, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::chordal::find_induced_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::chordal::enumerate_all_cliques, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::chordal::max_weight_clique, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::chordal::with_extra_edges, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::adjacency_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::incidence_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::laplacian_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::directed_laplacian_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::modularity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::bethe_hessian_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::modularity_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::bethe_hessian_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::algebraic_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::fiedler_vector, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::spectral_ordering, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::linalg::kemeny_constant, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community::detect_communities, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community::girvan_newman, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::find_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::simple_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::cycle_basis, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::minimum_cycle_basis, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::directed_cycle_basis, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::connectivity::k_components, m)?)?;
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
    m.add_function(wrap_pyfunction!(algorithms::node_vitality::closeness_vitality_weighted, m)?)?;
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
    m.add_function(wrap_pyfunction!(algorithms::spanning_trees::directed_steiner_tree, m)?)?;
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

    // SimRank* similarity
    m.add_function(wrap_pyfunction!(algorithms::simrank::simrank_star_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::simrank::simrank_star_similarity_pair, m)?)?;

    // Propagation algorithms
    m.add_function(wrap_pyfunction!(algorithms::propagation::personalized_pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::propagation::random_walk_with_restart, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::propagation::heat_diffusion, m)?)?;

    // Bipartite algorithms
    m.add_function(wrap_pyfunction!(algorithms::bipartite::bipartite_sets, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::bipartite::bipartite_projection, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::bipartite::bipartite_degree_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::bipartite::bipartite_closeness_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::bipartite::bipartite_betweenness_centrality, m)?)?;

    // Seed module discovery
    m.add_function(wrap_pyfunction!(algorithms::spanning_trees::prize_collecting_steiner_tree, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::spanning_trees::expand_seed_set, m)?)?;

    // Null models and randomization
    m.add_function(wrap_pyfunction!(algorithms::null_models::configuration_model, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::null_models::degree_preserving_rewire, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::null_models::expected_degree_graph, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::null_models::generate_stochastic_block_model, m)?)?;

    // Network motif census
    m.add_function(wrap_pyfunction!(algorithms::motifs::motif_census, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::motifs::motif_significance, m)?)?;

    // Network proximity
    m.add_function(wrap_pyfunction!(algorithms::network_proximity::network_proximity, m)?)?;

    // Label propagation
    m.add_function(wrap_pyfunction!(algorithms::label_propagation::label_propagation, m)?)?;

    // Topological overlap
    m.add_function(wrap_pyfunction!(algorithms::topological_overlap::topological_overlap_matrix, m)?)?;

    // Network robustness
    m.add_function(wrap_pyfunction!(algorithms::robustness::network_robustness, m)?)?;

    // Influence maximization
    m.add_function(wrap_pyfunction!(algorithms::influence_maximization::influence_maximization, m)?)?;

    // Network smoothing
    m.add_function(wrap_pyfunction!(algorithms::network_smoothing::network_smooth, m)?)?;

    // WGCNA
    m.add_function(wrap_pyfunction!(algorithms::wgcna::soft_threshold_power, m)?)?;

    // Network comparison
    m.add_function(wrap_pyfunction!(algorithms::network_comparison::edge_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::network_comparison::differential_edges, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::network_comparison::graph_edit_distance, m)?)?;

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
            let module = PyModule::new(py, "_core").unwrap();
            assert!(_core(&module).is_ok());
        });
    }
}
