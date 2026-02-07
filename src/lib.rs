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
    use std::collections::HashMap;

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

    fn json_err(e: serde_json::Error) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e))
    }

    /// Byzantine-tolerant aggregator for federated learning model updates.
    ///
    /// Wraps Rust-backed aggregation algorithms (trimmed mean, median, fedavg)
    /// with optional client reputation tracking.
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

        /// Aggregate client model updates using the configured method.
        ///
        /// Args:
        ///     updates: List of 2D numpy arrays (one per client).
        ///     client_ids: Optional list of client identifier strings for reputation tracking.
        ///
        /// Returns:
        ///     Aggregated 2D numpy array.
        #[pyo3(signature = (updates, client_ids=None))]
        fn aggregate<'py>(
            &mut self,
            py: Python<'py>,
            updates: Vec<numpy::PyReadonlyArray2<'py, f32>>,
            client_ids: Option<Vec<String>>,
        ) -> PyResult<&'py PyArray2<f32>> {
            let rust_updates: Vec<Array2<f32>> = updates
                .iter()
                .map(|arr| arr.as_array().to_owned())
                .collect();

            let ids_ref = client_ids.as_deref();
            let result = self
                .inner
                .aggregate(&rust_updates, ids_ref)
                .map_err(qora_err)?;
            Ok(result.into_pyarray(py))
        }

        /// Get the reputation score for a client (default 0.5 for unknown clients).
        fn get_reputation(&self, client_id: &str) -> f32 {
            self.inner.get_reputation(client_id)
        }

        /// Reset all reputation scores.
        fn reset_reputation(&mut self) {
            self.inner.reset_reputation();
        }

        /// Serialize aggregator state to JSON for persistence.
        fn to_json(&self) -> PyResult<String> {
            serde_json::to_string(&self.inner).map_err(json_err)
        }

        /// Restore aggregator from a JSON string.
        #[staticmethod]
        fn from_json(json_str: &str) -> PyResult<Self> {
            let inner: crate::ByzantineAggregator =
                serde_json::from_str(json_str).map_err(json_err)?;
            Ok(Self { inner })
        }
    }

    /// Standalone reputation manager with string-based client IDs.
    ///
    /// Provides persistent trust scores for federated learning clients.
    /// Scores range from 0.0 (fully distrusted) to 1.0 (fully trusted),
    /// with new clients starting at 0.5. Clients below the ban threshold
    /// are excluded from aggregation.
    #[pyclass(name = "ReputationManager")]
    struct PyReputationManager {
        scores: HashMap<String, f32>,
        ban_threshold: f32,
    }

    #[pymethods]
    impl PyReputationManager {
        #[new]
        #[pyo3(signature = (ban_threshold=0.2))]
        fn new(ban_threshold: f32) -> Self {
            Self {
                scores: HashMap::new(),
                ban_threshold,
            }
        }

        /// Get the trust score for a client (default 0.5 for unknown).
        fn get_score(&self, client_id: &str) -> f32 {
            self.scores.get(client_id).copied().unwrap_or(0.5)
        }

        /// Set the trust score for a client (clamped to [0.0, 1.0]).
        fn set_score(&mut self, client_id: String, score: f32) {
            self.scores.insert(client_id, score.clamp(0.0, 1.0));
        }

        /// Increase a client's reputation by the given amount.
        fn reward(&mut self, client_id: String, amount: f32) {
            let score = self.scores.entry(client_id).or_insert(0.5);
            *score = (*score + amount).min(1.0);
        }

        /// Decrease a client's reputation by the given amount.
        fn penalize(&mut self, client_id: String, amount: f32) {
            let score = self.scores.entry(client_id).or_insert(0.5);
            *score = (*score - amount).max(0.0);
        }

        /// Check if a client is banned (score below ban threshold).
        fn is_banned(&self, client_id: &str) -> bool {
            self.get_score(client_id) < self.ban_threshold
        }

        /// Get all non-banned clients and their scores.
        fn active_clients(&self) -> Vec<(String, f32)> {
            self.scores
                .iter()
                .filter(|(_, &score)| score >= self.ban_threshold)
                .map(|(id, &score)| (id.clone(), score))
                .collect()
        }

        /// Get all client scores as a dictionary.
        fn all_scores(&self) -> HashMap<String, f32> {
            self.scores.clone()
        }

        /// Reset all reputation scores.
        fn reset(&mut self) {
            self.scores.clear();
        }

        /// Serialize reputation state to JSON for persistence between restarts.
        fn to_json(&self) -> PyResult<String> {
            serde_json::to_string(&self.scores).map_err(json_err)
        }

        /// Restore reputation state from a JSON string.
        #[staticmethod]
        #[pyo3(signature = (json_str, ban_threshold=0.2))]
        fn from_json(json_str: &str, ban_threshold: f32) -> PyResult<Self> {
            let scores: HashMap<String, f32> = serde_json::from_str(json_str).map_err(json_err)?;
            Ok(Self {
                scores,
                ban_threshold,
            })
        }
    }

    #[pymodule]
    fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyByzantineAggregator>()?;
        m.add_class::<PyReputationManager>()?;
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
