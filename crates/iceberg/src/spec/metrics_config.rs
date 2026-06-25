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
//! The bound-truncation primitives ([`MetricsMode::truncate_lower_bound`] /
//! [`MetricsMode::truncate_upper_bound`]) port Java `ParquetMetrics.truncateLowerBound` /
//! `truncateUpperBound` together with `UnicodeUtil`/`BinaryUtil`'s string/binary truncation;
//! the Parquet writer applies them per column (see `writer::file_writer::parquet_writer`).
//!
//! NAMED residue (deferred — both need a schema / sort-order the config alone does not
//! carry): the wide-schema `write.metadata.metrics.max-inferred-column-defaults` inference
//! (Java caps how many columns receive the default, the rest becoming [`MetricsMode::None`])
//! and the sorted-column auto-promotion (a `none`/`counts` default is upgraded to
//! `truncate(16)` for sort-key columns). Threading a table's *resolved* config through the
//! data-file writer construction (so non-default `write.metadata.metrics.*` is honored on
//! data files, not just the built-in `truncate(16)` default) is a further plumbing pass.

use std::collections::HashMap;

use super::{Datum, PrimitiveLiteral, PrimitiveType};
use crate::metadata_columns::{
    RESERVED_COL_NAME_DELETE_FILE_PATH, RESERVED_COL_NAME_DELETE_FILE_POS,
};
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

    /// Whether this mode persists lower/upper bounds at all.
    ///
    /// Bounds are kept for [`MetricsMode::Truncate`] and [`MetricsMode::Full`]; [`MetricsMode::None`]
    /// and [`MetricsMode::Counts`] persist no bounds. Mirrors Java
    /// `ParquetMetrics.truncateLength(mode) > 0` (`None`/`Counts` → 0).
    pub fn collects_bounds(&self) -> bool {
        matches!(self, MetricsMode::Truncate(_) | MetricsMode::Full)
    }

    /// Whether this mode persists counts (`value_counts`, `null_value_counts`, `nan_value_counts`)
    /// and column sizes. Everything except [`MetricsMode::None`] does; `None` persists nothing for
    /// the column (Java skips the column entirely when its mode is `None`).
    pub fn collects_counts(&self) -> bool {
        !matches!(self, MetricsMode::None)
    }

    /// Truncate a column's lower bound for this mode, mirroring Java
    /// `ParquetMetrics.truncateLowerBound`.
    ///
    /// Returns `None` when this mode persists no bounds ([`MetricsMode::None`]/[`MetricsMode::Counts`]).
    /// For [`MetricsMode::Full`] the bound is returned unchanged. For [`MetricsMode::Truncate`] only
    /// `string` and `binary` values are truncated (to the first `N` Unicode code points / bytes via
    /// `UnicodeUtil.truncateStringMin` / `BinaryUtil.truncateBinaryMin`); every other type — including
    /// `fixed`, `uuid`, and the numeric/decimal types — is returned unchanged, matching Java's
    /// `switch` whose `default` returns the value as-is.
    pub fn truncate_lower_bound(&self, datum: &Datum) -> Option<Datum> {
        let length = match self {
            MetricsMode::None | MetricsMode::Counts => return None,
            MetricsMode::Full => return Some(datum.clone()),
            MetricsMode::Truncate(length) => *length,
        };
        match (datum.data_type(), datum.literal()) {
            (PrimitiveType::String, PrimitiveLiteral::String(value)) => {
                Some(Datum::string(truncate_string_min(value, length)))
            }
            (PrimitiveType::Binary, PrimitiveLiteral::Binary(value)) => {
                Some(Datum::binary(truncate_binary_min(value, length)))
            }
            _ => Some(datum.clone()),
        }
    }

    /// Truncate a column's upper bound for this mode, mirroring Java
    /// `ParquetMetrics.truncateUpperBound`.
    ///
    /// Like [`MetricsMode::truncate_lower_bound`], but for `string`/`binary` under
    /// [`MetricsMode::Truncate`] it truncates *up* (`UnicodeUtil.truncateStringMax` /
    /// `BinaryUtil.truncateBinaryMax`): the prefix is incremented so it remains a valid upper bound.
    /// When no valid upper bound exists (every truncated code point / byte is at its maximum) Java
    /// returns `null` and drops the bound — here that is `None` even though the mode collects bounds.
    pub fn truncate_upper_bound(&self, datum: &Datum) -> Option<Datum> {
        let length = match self {
            MetricsMode::None | MetricsMode::Counts => return None,
            MetricsMode::Full => return Some(datum.clone()),
            MetricsMode::Truncate(length) => *length,
        };
        match (datum.data_type(), datum.literal()) {
            (PrimitiveType::String, PrimitiveLiteral::String(value)) => {
                truncate_string_max(value, length).map(Datum::string)
            }
            (PrimitiveType::Binary, PrimitiveLiteral::Binary(value)) => {
                truncate_binary_max(value, length).map(Datum::binary)
            }
            _ => Some(datum.clone()),
        }
    }
}

/// Java `BinaryUtil.truncateBinaryMin`: the first `length` bytes, or the input unchanged when it is
/// already no longer than `length`.
fn truncate_binary_min(input: &[u8], length: u32) -> Vec<u8> {
    let length = length as usize;
    if length >= input.len() {
        input.to_vec()
    } else {
        input[..length].to_vec()
    }
}

/// Java `BinaryUtil.truncateBinaryMax`: truncate to `length` bytes, then increment from the last
/// byte backward; the first byte that does not wrap past `0xFF` is incremented and the result cut
/// there. Returns `None` when every truncated byte is `0xFF` (no valid upper bound — Java returns
/// `null`).
fn truncate_binary_max(input: &[u8], length: u32) -> Option<Vec<u8>> {
    let length = length as usize;
    if length >= input.len() {
        return Some(input.to_vec());
    }
    let mut truncated = input[..length].to_vec();
    for i in (0..length).rev() {
        if truncated[i] != 0xFF {
            truncated[i] += 1;
            truncated.truncate(i + 1);
            return Some(truncated);
        }
    }
    None
}

/// Java `UnicodeUtil.truncateString`: the first `length` Unicode code points, or the input unchanged
/// when it has no more than `length`. A Rust `&str` is well-formed UTF-8 with no surrogates, so its
/// `char`s are exactly Java's code points.
fn truncate_string_min(input: &str, length: u32) -> String {
    let length = length as usize;
    match input.char_indices().nth(length) {
        Some((byte_idx, _)) => input[..byte_idx].to_string(),
        // Fewer than `length + 1` code points → no truncation.
        None => input.to_string(),
    }
}

/// Java `UnicodeUtil.truncateStringMax` / `internalTruncateMax`: truncate to `length` code points,
/// then increment the last code point that can be incremented. Returns `None` when no code point can
/// be incremented (Java returns `null`).
fn truncate_string_max(input: &str, length: u32) -> Option<String> {
    let truncated = truncate_string_min(input, length);
    // No truncation happened → the exact value is already a valid upper bound (Java compares the
    // pre/post lengths; an untruncated prefix has the same byte length as the input).
    if truncated.len() == input.len() {
        return Some(truncated);
    }
    let mut chars: Vec<char> = truncated.chars().collect();
    for i in (0..chars.len()).rev() {
        if let Some(next) = increment_code_point(chars[i]) {
            chars.truncate(i);
            chars.push(next);
            return Some(chars.into_iter().collect());
        }
    }
    None
}

/// Java `UnicodeUtil.incrementCodePoint`: the next code point, skipping the UTF-16 surrogate gap, or
/// `None` when incrementing the maximum code point would overflow (Java returns `0` to signal this).
fn increment_code_point(c: char) -> Option<char> {
    let code_point = c as u32;
    if code_point == char::MAX as u32 {
        // Java: `Character.MAX_CODE_POINT` → 0 (overflow).
        return None;
    }
    // Java jumps from `MIN_SURROGATE - 1` (0xD7FF) straight to `MAX_SURROGATE + 1` (0xE000); a Rust
    // `char` is never a surrogate so 0xD7FF is the only value whose successor lands in the gap.
    let next = if code_point == 0xD7FF {
        0xE000
    } else {
        code_point + 1
    };
    char::from_u32(next)
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

    /// The metrics config for a position-delete file, mirroring `MetricsConfig.forPositionDelete`.
    ///
    /// Starts from the built-in default (`truncate(16)`) but forces the reserved position-delete
    /// columns `file_path` and `pos` to [`MetricsMode::Full`], so a delete file's path/position
    /// bounds are kept exact — delete-file pruning relies on them. (Java overlays this on the
    /// *table's* resolved config; threading the table config in is part of the writer-plumbing
    /// residue noted in the module docs.)
    pub fn for_position_delete() -> Self {
        let mut config = Self::default();
        config.column_modes.insert(
            RESERVED_COL_NAME_DELETE_FILE_PATH.to_string(),
            MetricsMode::Full,
        );
        config.column_modes.insert(
            RESERVED_COL_NAME_DELETE_FILE_POS.to_string(),
            MetricsMode::Full,
        );
        config
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

    // ---- bound truncation (Java UnicodeUtil / BinaryUtil / ParquetMetrics parity) -------------

    #[test]
    fn truncate_binary_min_takes_prefix_or_input() {
        // Longer than the limit → first `length` bytes.
        assert_eq!(truncate_binary_min(&[1, 2, 3, 4, 5], 3), vec![1, 2, 3]);
        // length >= len → unchanged (no copy of a longer prefix).
        assert_eq!(truncate_binary_min(&[1, 2], 5), vec![1, 2]);
        assert_eq!(truncate_binary_min(&[1, 2, 3], 3), vec![1, 2, 3]);
    }

    #[test]
    fn truncate_binary_max_increments_last_non_ff_byte() {
        // Truncate to 2 bytes then increment the last byte.
        assert_eq!(
            truncate_binary_max(&[1, 2, 3, 4], 2),
            Some(vec![1, 3]),
            "last truncated byte 0x02 → 0x03"
        );
        // Trailing 0xFF bytes wrap; carry to the first byte that can increment, then cut there.
        assert_eq!(
            truncate_binary_max(&[1, 0xFF, 0xFF, 9], 3),
            Some(vec![2]),
            "0xFF,0xFF wrap → byte 0 (0x01) increments to 0x02 and the result is cut to length 1"
        );
        // length >= len → unchanged, never None.
        assert_eq!(truncate_binary_max(&[1, 2], 5), Some(vec![1, 2]));
        // Every truncated byte is 0xFF → no valid upper bound.
        assert_eq!(truncate_binary_max(&[0xFF, 0xFF, 0x00], 2), None);
    }

    #[test]
    fn truncate_string_min_counts_code_points_not_bytes() {
        assert_eq!(truncate_string_min("hello world", 5), "hello");
        // Shorter than the limit → unchanged.
        assert_eq!(truncate_string_min("hi", 5), "hi");
        assert_eq!(truncate_string_min("hello", 5), "hello");
        // Multi-byte code points: "é" (2 bytes) + "猫" (3 bytes) each count as ONE code point, so a
        // byte-based truncation would slice mid-character; the code-point count must not.
        assert_eq!(truncate_string_min("é猫x", 2), "é猫");
        assert_eq!(truncate_string_min("😀😀😀", 2), "😀😀");
    }

    #[test]
    fn truncate_string_max_increments_last_code_point() {
        assert_eq!(
            truncate_string_max("hello world", 5),
            Some("hellp".to_string())
        );
        // No truncation → unchanged, never incremented.
        assert_eq!(truncate_string_max("hi", 5), Some("hi".to_string()));
        // Multi-byte: increment the last retained code point.
        assert_eq!(truncate_string_max("aé猫", 2), Some("aê".to_string()));
        // Max code point at the tail wraps; carry to the previous code point.
        let max = char::MAX; // U+10FFFF
        let input: String = ['a', max, max].iter().collect();
        assert_eq!(truncate_string_max(&input, 2), Some("b".to_string()));
        // Every retained code point is the max → no valid upper bound.
        let all_max: String = [max, max, 'z'].iter().collect();
        assert_eq!(truncate_string_max(&all_max, 2), None);
    }

    #[test]
    fn increment_code_point_skips_surrogate_gap_and_overflows_at_max() {
        // 0xD7FF jumps the UTF-16 surrogate gap to 0xE000.
        assert_eq!(
            increment_code_point(char::from_u32(0xD7FF).unwrap()),
            char::from_u32(0xE000)
        );
        assert_eq!(increment_code_point('a'), Some('b'));
        // The maximum code point overflows.
        assert_eq!(increment_code_point(char::MAX), None);
    }

    #[test]
    fn mode_truncate_only_touches_string_and_binary() {
        let mode = MetricsMode::Truncate(2);
        // String/binary truncate.
        assert_eq!(
            mode.truncate_lower_bound(&Datum::string("hello")),
            Some(Datum::string("he"))
        );
        assert_eq!(
            mode.truncate_upper_bound(&Datum::string("hello")),
            Some(Datum::string("hf"))
        );
        assert_eq!(
            mode.truncate_lower_bound(&Datum::binary(vec![1, 2, 3, 4])),
            Some(Datum::binary(vec![1, 2]))
        );
        // Non-string/binary (the Java `default` arm) is returned unchanged even under Truncate.
        let long = Datum::long(1234567890);
        assert_eq!(mode.truncate_lower_bound(&long), Some(long.clone()));
        assert_eq!(mode.truncate_upper_bound(&long), Some(long));
    }

    #[test]
    fn mode_full_never_truncates_none_and_counts_drop_bounds() {
        let s = Datum::string("a very long string value");
        // Full keeps the bound exactly.
        assert_eq!(MetricsMode::Full.truncate_lower_bound(&s), Some(s.clone()));
        assert_eq!(MetricsMode::Full.truncate_upper_bound(&s), Some(s.clone()));
        // None / Counts persist no bounds.
        assert_eq!(MetricsMode::None.truncate_lower_bound(&s), None);
        assert_eq!(MetricsMode::None.truncate_upper_bound(&s), None);
        assert_eq!(MetricsMode::Counts.truncate_lower_bound(&s), None);
        assert_eq!(MetricsMode::Counts.truncate_upper_bound(&s), None);
    }

    #[test]
    fn collects_bounds_and_counts_match_mode_semantics() {
        assert!(!MetricsMode::None.collects_bounds());
        assert!(!MetricsMode::None.collects_counts());
        assert!(!MetricsMode::Counts.collects_bounds());
        assert!(MetricsMode::Counts.collects_counts());
        assert!(MetricsMode::Truncate(16).collects_bounds());
        assert!(MetricsMode::Truncate(16).collects_counts());
        assert!(MetricsMode::Full.collects_bounds());
        assert!(MetricsMode::Full.collects_counts());
    }

    #[test]
    fn for_position_delete_forces_path_and_pos_to_full() {
        let config = MetricsConfig::for_position_delete();
        assert_eq!(config.column_mode("file_path"), MetricsMode::Full);
        assert_eq!(config.column_mode("pos"), MetricsMode::Full);
        // Every other column keeps the default truncate(16).
        assert_eq!(
            config.column_mode("anything_else"),
            MetricsMode::Truncate(16)
        );
    }
}
