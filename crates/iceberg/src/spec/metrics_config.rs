// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Column metrics modes and per-column metrics configuration.
//!
//! Ports the self-contained core of Java's `org.apache.iceberg.MetricsModes` and
//! `org.apache.iceberg.MetricsConfig`: the [`MetricsMode`] value type (with its
//! `truncate(N)` parsing) and the table-property resolution of a per-column mode
//! ([`MetricsConfig`]). These govern WHICH column statistics a writer persists
//! (`value_counts`, bounds, …) — they do not change the on-disk format itself.
//!
//! # Java parity
//!
//! [`MetricsMode::parse`] mirrors `MetricsModes.fromString` (case-insensitive
//! `none`/`counts`/`full` and `truncate(N)` with `N > 0`). [`MetricsConfig::from_properties`]
//! mirrors `MetricsConfig.from(Map)`: the table default comes from
//! `write.metadata.metrics.default` (falling back to `truncate(16)`), per-column overrides
//! come from `write.metadata.metrics.column.<name>`, and an unparsable mode is logged and
//! falls back (to the table default for a column, or to `truncate(16)` for the default
//! itself) rather than failing — matching Java's warn-and-continue behavior.
//!
//! NAMED residue (deferred — both need a schema / sort-order the config alone does not
//! carry): the wide-schema `write.metadata.metrics.max-inferred-column-defaults` inference
//! (Java caps how many columns receive the default, the rest becoming [`MetricsMode::None`])
//! and the sorted-column auto-promotion (a `none`/`counts` default is upgraded to
//! `truncate(16)` for sort-key columns). Wiring the resolved mode into the Parquet writer's
//! bound collection is a separate change (it alters the bounds written to every table) and
//! is intentionally not done here.

use std::collections::HashMap;

use crate::{Error, ErrorKind, Result};

/// Table property naming the default metrics mode for all columns.
///
/// Java: `TableProperties.DEFAULT_WRITE_METRICS_MODE`.
pub const METRICS_MODE_DEFAULT_KEY: &str = "write.metadata.metrics.default";

/// Prefix for per-column metrics-mode overrides; the column name follows the dot.
///
/// Java: `TableProperties.METRICS_MODE_COLUMN_CONF_PREFIX`.
pub const METRICS_MODE_COLUMN_CONF_PREFIX: &str = "write.metadata.metrics.column.";

/// The default metrics mode when `write.metadata.metrics.default` is unset.
///
/// Java: `TableProperties.DEFAULT_WRITE_METRICS_MODE_DEFAULT` (`"truncate(16)"`).
pub const DEFAULT_WRITE_METRICS_MODE_DEFAULT: &str = "truncate(16)";

/// The metrics collected and persisted for a column.
///
/// Mirrors the four `org.apache.iceberg.MetricsModes.MetricsMode` implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricsMode {
    /// Persist no metrics for the column.
    None,
    /// Persist only `value_counts`, `null_value_counts`, and `nan_value_counts`.
    Counts,
    /// Counts plus lower/upper bounds truncated to the given length (in characters for
    /// strings, bytes for binary). The length is always positive.
    Truncate(u32),
    /// Counts plus full (untruncated) lower/upper bounds.
    Full,
}

impl MetricsMode {
    /// Parse a metrics-mode string, mirroring Java `MetricsModes.fromString`.
    ///
    /// Accepts (case-insensitive) `none`, `counts`, `full`, and `truncate(N)` where `N` is a
    /// positive integer. Any other input — including `truncate(0)`, a non-numeric or
    /// signed length, or stray whitespace inside the parentheses — is rejected with a
    /// [`ErrorKind::DataInvalid`] error, matching Java's `IllegalArgumentException`.
    pub fn parse(mode: &str) -> Result<Self> {
        let lowered = mode.to_lowercase();
        match lowered.as_str() {
            "none" => Ok(MetricsMode::None),
            "counts" => Ok(MetricsMode::Counts),
            "full" => Ok(MetricsMode::Full),
            other => {
                // Mirror Java's `Pattern.compile("truncate\\((\\d+)\\)")` exactly: the whole
                // string must be `truncate(` + one-or-more ASCII digits + `)`. The digit guard
                // also stops `i32::parse` from accepting a leading `+`.
                if let Some(inner) = other
                    .strip_prefix("truncate(")
                    .and_then(|rest| rest.strip_suffix(')'))
                    && !inner.is_empty()
                    && inner.bytes().all(|b| b.is_ascii_digit())
                {
                    // Java parses the length with `Integer.parseInt`, so its domain is the
                    // signed 32-bit range: a value above `i32::MAX` overflows and is rejected
                    // (caught as `IllegalArgumentException`). Parse as `i32` to match that
                    // domain exactly — `u32` would over-liberally accept `(i32::MAX, u32::MAX]`.
                    let length: i32 = inner.parse().map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Invalid truncate length in metrics mode {mode:?}"),
                        )
                        .with_source(e)
                    })?;
                    // Java `Truncate.withLength` requires a positive length.
                    if length <= 0 {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            "Truncate length should be positive",
                        ));
                    }
                    // `length` is now in `1..=i32::MAX`, so the cast is exact (no truncation).
                    Ok(MetricsMode::Truncate(length as u32))
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid metrics mode: {mode:?}"),
                    ))
                }
            }
        }
    }
}

impl std::fmt::Display for MetricsMode {
    /// Render the canonical string form, matching each Java `MetricsMode.toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricsMode::None => write!(f, "none"),
            MetricsMode::Counts => write!(f, "counts"),
            MetricsMode::Truncate(length) => write!(f, "truncate({length})"),
            MetricsMode::Full => write!(f, "full"),
        }
    }
}

/// Per-column metrics configuration resolved from table properties.
///
/// Mirrors the property-driven core of Java `org.apache.iceberg.MetricsConfig`: a table-wide
/// default mode plus per-column overrides. See the module docs for the deferred
/// schema-dependent behavior (wide-schema inference and sorted-column promotion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricsConfig {
    default_mode: MetricsMode,
    column_modes: HashMap<String, MetricsMode>,
}

impl MetricsConfig {
    /// Resolve a [`MetricsConfig`] from raw table properties, mirroring
    /// `MetricsConfig.from(Map<String, String>)`.
    ///
    /// The default mode is read from `write.metadata.metrics.default` (falling back to
    /// `truncate(16)`); per-column overrides are read from every
    /// `write.metadata.metrics.column.<name>` key. An unparsable mode is logged via
    /// `tracing::warn!` and falls back — to the resolved default for a column, or to
    /// `truncate(16)` for the default itself — rather than erroring, matching Java.
    pub fn from_properties(properties: &HashMap<String, String>) -> Self {
        let default_mode = match properties.get(METRICS_MODE_DEFAULT_KEY) {
            Some(raw) => MetricsMode::parse(raw).unwrap_or_else(|error| {
                tracing::warn!(
                    ?error,
                    key = METRICS_MODE_DEFAULT_KEY,
                    value = raw,
                    "invalid default metrics mode; falling back to {DEFAULT_WRITE_METRICS_MODE_DEFAULT}"
                );
                Self::default_mode()
            }),
            None => Self::default_mode(),
        };

        let mut column_modes = HashMap::new();
        for (key, raw) in properties {
            let Some(column) = key.strip_prefix(METRICS_MODE_COLUMN_CONF_PREFIX) else {
                continue;
            };
            // Java does not special-case the bare prefix: `key.replaceFirst(PREFIX, "")` yields
            // an empty alias and it is stored verbatim. We mirror that (it is harmless — an
            // empty column name cannot match any real schema field).
            let mode = MetricsMode::parse(raw).unwrap_or_else(|error| {
                tracing::warn!(
                    ?error,
                    key,
                    value = raw,
                    "invalid column metrics mode; falling back to the table default"
                );
                default_mode
            });
            column_modes.insert(column.to_string(), mode);
        }

        MetricsConfig {
            default_mode,
            column_modes,
        }
    }

    /// The resolved metrics mode for a column: its explicit override if present, else the
    /// table default. Mirrors `MetricsConfig.columnMode(String)`.
    pub fn column_mode(&self, column: &str) -> MetricsMode {
        self.column_modes
            .get(column)
            .copied()
            .unwrap_or(self.default_mode)
    }

    /// The table-wide default mode.
    pub fn default_mode_of(&self) -> MetricsMode {
        self.default_mode
    }

    /// The built-in default (`truncate(16)`) used when no property overrides it.
    fn default_mode() -> MetricsMode {
        // The constant is statically valid; parse is the single source of truth for the form.
        MetricsMode::parse(DEFAULT_WRITE_METRICS_MODE_DEFAULT)
            .expect("DEFAULT_WRITE_METRICS_MODE_DEFAULT must be a valid metrics mode")
    }
}

impl Default for MetricsConfig {
    /// An empty config whose default mode is `truncate(16)` and which carries no per-column
    /// overrides — i.e. `MetricsConfig.from(emptyMap())`.
    fn default() -> Self {
        MetricsConfig {
            default_mode: Self::default_mode(),
            column_modes: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parse_accepts_keyword_modes() {
        assert_eq!(MetricsMode::parse("none").unwrap(), MetricsMode::None);
        assert_eq!(MetricsMode::parse("counts").unwrap(), MetricsMode::Counts);
        assert_eq!(MetricsMode::parse("full").unwrap(), MetricsMode::Full);
    }

    #[test]
    fn parse_is_case_insensitive() {
        // Java uses equalsIgnoreCase / toLowerCase(Locale.ROOT).
        assert_eq!(MetricsMode::parse("NONE").unwrap(), MetricsMode::None);
        assert_eq!(MetricsMode::parse("Counts").unwrap(), MetricsMode::Counts);
        assert_eq!(MetricsMode::parse("FULL").unwrap(), MetricsMode::Full);
        assert_eq!(
            MetricsMode::parse("TRUNCATE(10)").unwrap(),
            MetricsMode::Truncate(10)
        );
    }

    #[test]
    fn parse_accepts_truncate_with_positive_length() {
        assert_eq!(
            MetricsMode::parse("truncate(1)").unwrap(),
            MetricsMode::Truncate(1)
        );
        assert_eq!(
            MetricsMode::parse("truncate(16)").unwrap(),
            MetricsMode::Truncate(16)
        );
    }

    #[test]
    fn parse_truncate_length_matches_java_int_domain() {
        // Java parses the length with `Integer.parseInt`, so `i32::MAX` is the largest accepted
        // value and anything above it overflows → reject. We parse as `i32` to match exactly.
        assert_eq!(
            MetricsMode::parse("truncate(2147483647)").unwrap(),
            MetricsMode::Truncate(2147483647) // i32::MAX
        );
        // i32::MAX + 1 — within u32 but rejected by Java's int parse; must reject here too.
        assert!(MetricsMode::parse("truncate(2147483648)").is_err());
    }

    #[test]
    fn parse_rejects_zero_truncate_length() {
        // Java `Truncate.withLength` requires length > 0.
        let err = MetricsMode::parse("truncate(0)").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn parse_rejects_malformed_truncate() {
        // Each of these fails Java's `truncate\((\d+)\)` whole-string regex (sign, whitespace,
        // empty, non-numeric, unterminated) or the `i32` parse range.
        for bad in [
            "truncate(-1)",
            "truncate(+5)",
            "truncate()",
            "truncate( 16)",
            "truncate(16 )",
            "truncate(abc)",
            "truncate(16",
            "truncate16)",
            "truncate(99999999999999999999)",
        ] {
            assert!(
                MetricsMode::parse(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn parse_rejects_unknown_mode() {
        assert!(MetricsMode::parse("bogus").is_err());
        assert!(MetricsMode::parse("").is_err());
        assert!(MetricsMode::parse("count").is_err());
    }

    #[test]
    fn display_round_trips_through_parse() {
        for mode in [
            MetricsMode::None,
            MetricsMode::Counts,
            MetricsMode::Truncate(16),
            MetricsMode::Truncate(1),
            MetricsMode::Full,
        ] {
            let rendered = mode.to_string();
            assert_eq!(MetricsMode::parse(&rendered).unwrap(), mode);
        }
        // Pin the exact canonical strings (matches each Java MetricsMode.toString()).
        assert_eq!(MetricsMode::None.to_string(), "none");
        assert_eq!(MetricsMode::Counts.to_string(), "counts");
        assert_eq!(MetricsMode::Truncate(16).to_string(), "truncate(16)");
        assert_eq!(MetricsMode::Full.to_string(), "full");
    }

    #[test]
    fn config_default_is_truncate_16() {
        let config = MetricsConfig::from_properties(&HashMap::new());
        assert_eq!(config.default_mode_of(), MetricsMode::Truncate(16));
        assert_eq!(config.column_mode("any"), MetricsMode::Truncate(16));
        assert_eq!(MetricsConfig::default(), config);
    }

    #[test]
    fn config_reads_explicit_default() {
        let config =
            MetricsConfig::from_properties(&props(&[("write.metadata.metrics.default", "full")]));
        assert_eq!(config.default_mode_of(), MetricsMode::Full);
        assert_eq!(config.column_mode("unspecified"), MetricsMode::Full);
    }

    #[test]
    fn config_column_override_takes_precedence_over_default() {
        let config = MetricsConfig::from_properties(&props(&[
            ("write.metadata.metrics.default", "counts"),
            ("write.metadata.metrics.column.id", "full"),
            ("write.metadata.metrics.column.name", "none"),
        ]));
        assert_eq!(config.column_mode("id"), MetricsMode::Full);
        assert_eq!(config.column_mode("name"), MetricsMode::None);
        // A column without an override falls back to the default.
        assert_eq!(config.column_mode("other"), MetricsMode::Counts);
    }

    #[test]
    fn config_invalid_default_falls_back_to_truncate_16() {
        // Java warns and falls back to DEFAULT_WRITE_METRICS_MODE_DEFAULT rather than throwing.
        let config = MetricsConfig::from_properties(&props(&[(
            "write.metadata.metrics.default",
            "truncate(0)",
        )]));
        assert_eq!(config.default_mode_of(), MetricsMode::Truncate(16));
    }

    #[test]
    fn config_invalid_column_falls_back_to_default() {
        let config = MetricsConfig::from_properties(&props(&[
            ("write.metadata.metrics.default", "counts"),
            ("write.metadata.metrics.column.id", "bogus"),
        ]));
        // The bad column mode falls back to the resolved default, not to truncate(16).
        assert_eq!(config.column_mode("id"), MetricsMode::Counts);
    }

    #[test]
    fn config_stores_bare_prefix_as_empty_column_alias() {
        // Java stores the bare-prefix key under the empty alias (`replaceFirst` → ""), so
        // `columnMode("")` returns that override. We mirror that exactly.
        let config = MetricsConfig::from_properties(&props(&[
            ("write.metadata.metrics.column.", "full"),
            ("write.metadata.metrics.default", "counts"),
        ]));
        assert_eq!(config.column_mode(""), MetricsMode::Full);
        // A real column name still falls back to the table default.
        assert_eq!(config.column_mode("id"), MetricsMode::Counts);
    }
}
