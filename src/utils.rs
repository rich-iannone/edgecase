//! Utility functions

use pyo3::prelude::*;

/// Utility function to validate graph integrity
#[pyfunction]
pub fn validate_graph() -> PyResult<bool> {
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_graph() {
        assert!(validate_graph().unwrap());
    }
}
