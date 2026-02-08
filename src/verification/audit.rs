//! Aggregation audit log for post-hoc analysis.
//!
//! Records metadata about each aggregation round, enabling
//! reproducibility analysis and anomaly detection.

use serde::{Deserialize, Serialize};

/// Metadata for a single aggregation round.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregationAuditEntry {
    /// Round number (0-indexed).
    pub round: usize,
    /// Number of clients that participated.
    pub n_clients: usize,
    /// Number of clients excluded (e.g., by ban gating).
    pub n_excluded: usize,
    /// Aggregation method used (as string).
    pub method: String,
    /// Effective trim fraction used (for TrimmedMean).
    pub trim_fraction: Option<f32>,
}

/// Append-only audit log of aggregation rounds.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AuditLog {
    entries: Vec<AggregationAuditEntry>,
}

impl AuditLog {
    /// Create a new, empty audit log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append an entry to the log.
    pub fn push(&mut self, entry: AggregationAuditEntry) {
        self.entries.push(entry);
    }

    /// Get all entries.
    pub fn entries(&self) -> &[AggregationAuditEntry] {
        &self.entries
    }

    /// Number of recorded rounds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize the audit log to JSON.
    #[cfg(feature = "python")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_push_and_len() {
        let mut log = AuditLog::new();
        assert!(log.is_empty());

        log.push(AggregationAuditEntry {
            round: 0,
            n_clients: 10,
            n_excluded: 2,
            method: "trimmed_mean".to_string(),
            trim_fraction: Some(0.2),
        });

        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
        assert_eq!(log.entries()[0].round, 0);
        assert_eq!(log.entries()[0].n_clients, 10);
    }

    #[test]
    fn test_audit_log_multiple_entries() {
        let mut log = AuditLog::new();
        for i in 0..5 {
            log.push(AggregationAuditEntry {
                round: i,
                n_clients: 10,
                n_excluded: 0,
                method: "krum".to_string(),
                trim_fraction: None,
            });
        }
        assert_eq!(log.len(), 5);
        assert_eq!(log.entries()[4].round, 4);
    }

    #[test]
    fn test_audit_entry_serde() {
        let entry = AggregationAuditEntry {
            round: 3,
            n_clients: 20,
            n_excluded: 5,
            method: "multi_krum".to_string(),
            trim_fraction: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let restored: AggregationAuditEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.round, 3);
        assert_eq!(restored.n_clients, 20);
    }
}
