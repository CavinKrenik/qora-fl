//! # Qora-FL: Quorum-Oriented Robust Aggregation for Federated Learning
//!
//! Qora (pronounced "KOR-ah") provides Byzantine-tolerant aggregation
//! for federated learning through quorum consensus.
//!
//! ## Aggregation Methods
//!
//! - [`trimmed_mean()`] - Coordinate-wise trimmed mean (~30% Byzantine tolerance)
//! - [`median()`] - Coordinate-wise median (~50% Byzantine tolerance)
//! - [`aggregate_krum()`] - Krum selection with fixed-point math (n >= 2f+3)
//! - [`fedavg()`] - Standard FedAvg baseline (no Byzantine tolerance)
//!
//! ## High-Level API
//!
//! Use [`ByzantineAggregator`] for a convenient interface with built-in
//! reputation tracking.

#![deny(missing_docs)]

pub mod aggregators;
pub mod error;
pub mod math;
pub mod reputation;
pub mod verification;

// Re-exports
pub use aggregators::aggregate_krum;
pub use aggregators::fedavg;
pub use aggregators::median;
pub use aggregators::trimmed_mean;
pub use aggregators::{AggregationMethod, ByzantineAggregator};
pub use error::QoraError;
pub use reputation::ReputationTracker;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Python bindings via PyO3
#[cfg(feature = "python")]
mod python {
    use ndarray::Array2;
    use numpy::{IntoPyArray, PyArray2};
    use pyo3::prelude::*;

    use crate::{AggregationMethod, QoraError};

    fn parse_method(method: &str) -> PyResult<AggregationMethod> {
        match method {
            "trimmed_mean" => Ok(AggregationMethod::TrimmedMean),
            "median" => Ok(AggregationMethod::Median),
            "fedavg" => Ok(AggregationMethod::FedAvg),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown method '{}'. Use 'trimmed_mean', 'median', or 'fedavg'",
                method
            ))),
        }
    }

    fn qora_err(e: QoraError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    }

    #[pyclass(name = "ByzantineAggregator")]
    struct PyByzantineAggregator {
        inner: crate::ByzantineAggregator,
    }

    #[pymethods]
    impl PyByzantineAggregator {
        #[new]
        fn new(method: String, trim_fraction: f32) -> PyResult<Self> {
            let agg_method = parse_method(&method)?;
            Ok(Self {
                inner: crate::ByzantineAggregator::new(agg_method, trim_fraction),
            })
        }

        fn aggregate<'py>(
            &mut self,
            py: Python<'py>,
            updates: Vec<numpy::PyReadonlyArray2<'py, f32>>,
        ) -> PyResult<&'py PyArray2<f32>> {
            let rust_updates: Vec<Array2<f32>> = updates
                .iter()
                .map(|arr| arr.as_array().to_owned())
                .collect();

            let result = self
                .inner
                .aggregate(&rust_updates, None)
                .map_err(qora_err)?;
            Ok(result.into_pyarray(py))
        }

        fn get_reputation(&self, client_id: &str) -> f32 {
            self.inner.get_reputation(client_id)
        }

        fn reset_reputation(&mut self) {
            self.inner.reset_reputation();
        }
    }

    #[pymodule]
    fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyByzantineAggregator>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
