//! Legacy aggregation audit log.
//!
//! **Superseded by [`crate::audit`]**, which carries a per-update disposition,
//! typed rejection reasons, and full method parameters. The types here predate
//! that schema, are not produced or consumed by anything, and are retained only
//! because they were published in 0.3.1. Prefer
//! [`crate::audit::AggregationAuditEntry`] -- note that it is a *different*
//! type with the same name, reached through `crate::audit` rather than
//! `crate::verification::audit`.

use serde::{Deserialize, Serialize};

/// Metadata for a single aggregation round.
///
/// Superseded by [`crate::AggregationAuditEntry`], which records a per-update
/// disposition, typed rejection reasons, and full method parameters. The two
/// are different types with the same name and **do not share a serialized
/// shape**; records written with this type cannot be read as the new one.
#[deprecated(
    note = "use qora_fl::AggregationAuditEntry; the legacy verification audit schema will be removed in a future breaking release"
)]
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
///
/// No replacement storage type is provided: persistence is now caller-owned.
/// Serialize [`crate::AggregationAuditEntry`] and store the records with
/// whatever the application already uses.
#[deprecated(
    note = "audit persistence is now caller-owned; serialize and store qora_fl::AggregationAuditEntry using application-defined storage"
)]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
// The field type is itself deprecated; scoped to this definition only.
#[allow(deprecated)]
pub struct AuditLog {
    entries: Vec<AggregationAuditEntry>,
}

#[allow(deprecated)]
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
#[allow(deprecated)] // the deprecated types still ship, so keep testing them
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
