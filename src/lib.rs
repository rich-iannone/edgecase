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
    m.add_function(wrap_pyfunction!(algorithms::centrality::harmonic_centrality, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::centrality::pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::community::detect_communities, m)?)?;
    m.add_function(wrap_pyfunction!(algorithms::cycles::find_cycles, m)?)?;
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
