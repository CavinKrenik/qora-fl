//! # Qora-FL: Quorum-Oriented Robust Aggregation for Federated Learning
//!
//! Qora (pronounced "KOR-ah") provides Byzantine-tolerant aggregation
//! for federated learning through quorum consensus.
//!
//! ## Aggregation Methods
//!
//! - [`trimmed_mean()`] - Coordinate-wise trimmed mean (~30% Byzantine tolerance)
//! - [`median()`] - Coordinate-wise median (~50% Byzantine tolerance)
//! - [`aggregate_krum()`] - Krum selection with I16F16 fixed-point (n >= 2f+3)
//! - [`aggregate_krum_bfp16()`] - Krum selection with BFP-16 block floating-point (n >= 2f+3)
//! - [`aggregate_multi_krum_bfp16()`] - Multi-Krum: top-m selection + averaging (n >= 2f+3)
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
pub use aggregators::aggregate_krum_bfp16;
pub use aggregators::aggregate_multi_krum_bfp16;
pub use aggregators::fedavg;
pub use aggregators::median;
pub use aggregators::trimmed_mean;
pub use aggregators::{AggregationMethod, ByzantineAggregator};
pub use error::QoraError;
pub use reputation::ReputationStore;
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

    use crate::reputation::ReputationStore;
    use crate::{AggregationMethod, QoraError};

    fn parse_method(method: &str) -> PyResult<AggregationMethod> {
        let err_msg = "Use 'trimmed_mean', 'median', 'fedavg', 'krum[:N]', or 'multi_krum[:f:m]'";
        match method {
            "trimmed_mean" => Ok(AggregationMethod::TrimmedMean),
            "median" => Ok(AggregationMethod::Median),
            "fedavg" => Ok(AggregationMethod::FedAvg),
            s if s.starts_with("multi_krum") => {
                // Accept "multi_krum" (f=1, m=3) or "multi_krum:f:m"
                let (f, m) = if s == "multi_krum" {
                    (1, 3)
                } else if let Some(params) = s.strip_prefix("multi_krum:") {
                    let parts: Vec<&str> = params.split(':').collect();
                    if parts.len() != 2 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid multi_krum parameters '{}'. Use 'multi_krum' or 'multi_krum:f:m'",
                            s
                        )));
                    }
                    let f = parts[0].parse::<usize>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid multi_krum f parameter '{}'. {}",
                            parts[0], err_msg
                        ))
                    })?;
                    let m = parts[1].parse::<usize>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid multi_krum m parameter '{}'. {}",
                            parts[1], err_msg
                        ))
                    })?;
                    (f, m)
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown method '{}'. {}",
                        method, err_msg
                    )));
                };
                Ok(AggregationMethod::MultiKrum(f, m))
            }
            s if s.starts_with("krum") => {
                let f = if s == "krum" {
                    1
                } else if let Some(n) = s.strip_prefix("krum:") {
                    n.parse::<usize>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid krum parameter '{}'. {}",
                            s, err_msg
                        ))
                    })?
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown method '{}'. {}",
                        method, err_msg
                    )));
                };
                Ok(AggregationMethod::Krum(f))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown method '{}'. {}",
                method, err_msg
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
        #[pyo3(signature = (method, trim_fraction, ban_threshold=0.0, adaptive_trim=false))]
        fn new(
            method: String,
            trim_fraction: f32,
            ban_threshold: f32,
            adaptive_trim: bool,
        ) -> PyResult<Self> {
            let agg_method = parse_method(&method)?;
            let mut inner = if ban_threshold > 0.0 {
                crate::ByzantineAggregator::with_ban_threshold(
                    agg_method,
                    trim_fraction,
                    ban_threshold,
                )
            } else {
                crate::ByzantineAggregator::new(agg_method, trim_fraction)
            };
            inner.set_adaptive_trim(adaptive_trim);
            Ok(Self { inner })
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

        /// Decay all reputation scores toward 0.5 (default).
        ///
        /// Call once per round to allow penalized clients to recover.
        ///
        /// Args:
        ///     rate: Decay rate in (0.0, 1.0). Typical: 0.01-0.05.
        #[pyo3(signature = (rate=0.02))]
        fn decay_reputations(&mut self, rate: f32) {
            self.inner.decay_reputations(rate);
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
        inner: ReputationStore<String>,
        ban_threshold: f32,
    }

    #[pymethods]
    impl PyReputationManager {
        #[new]
        #[pyo3(signature = (ban_threshold=0.2))]
        fn new(ban_threshold: f32) -> Self {
            Self {
                inner: ReputationStore::new(),
                ban_threshold,
            }
        }

        /// Get the trust score for a client (default 0.5 for unknown).
        fn get_score(&self, client_id: &str) -> f32 {
            self.inner.get_score(client_id)
        }

        /// Set the trust score for a client (clamped to [0.0, 1.0]).
        fn set_score(&mut self, client_id: String, score: f32) {
            self.inner.set_score(client_id, score);
        }

        /// Increase a client's reputation by the given amount.
        fn reward(&mut self, client_id: String, amount: f32) {
            self.inner.reward(client_id, amount);
        }

        /// Decrease a client's reputation by the given amount.
        fn penalize(&mut self, client_id: String, amount: f32) {
            self.inner.penalize(client_id, amount);
        }

        /// Check if a client is banned (score below ban threshold).
        fn is_banned(&self, client_id: &str) -> bool {
            self.inner.is_banned(client_id, self.ban_threshold)
        }

        /// Get all non-banned clients and their scores.
        fn active_clients(&self) -> Vec<(String, f32)> {
            self.inner
                .iter()
                .filter(|(_, &score)| score >= self.ban_threshold)
                .map(|(id, &score)| (id.clone(), score))
                .collect()
        }

        /// Get all client scores as a dictionary.
        fn all_scores(&self) -> HashMap<String, f32> {
            self.inner.iter().map(|(k, &v)| (k.clone(), v)).collect()
        }

        /// Reset all reputation scores.
        fn reset(&mut self) {
            self.inner.clear();
        }

        /// Decay all reputation scores toward 0.5 (default).
        ///
        /// Args:
        ///     rate: Decay rate in (0.0, 1.0). Typical: 0.01-0.05.
        #[pyo3(signature = (rate=0.02))]
        fn decay(&mut self, rate: f32) {
            self.inner.decay_toward_default(rate);
        }

        /// Compute the influence weight for a client: min(rep^3, 0.8).
        fn influence_weight(&self, client_id: &str) -> f32 {
            self.inner.influence_weight(client_id, 0.8)
        }

        /// Serialize reputation state to JSON for persistence between restarts.
        fn to_json(&self) -> PyResult<String> {
            serde_json::to_string(&self.inner).map_err(json_err)
        }

        /// Restore reputation state from a JSON string.
        #[staticmethod]
        #[pyo3(signature = (json_str, ban_threshold=0.2))]
        fn from_json(json_str: &str, ban_threshold: f32) -> PyResult<Self> {
            let inner: ReputationStore<String> =
                serde_json::from_str(json_str).map_err(json_err)?;
            Ok(Self {
                inner,
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
