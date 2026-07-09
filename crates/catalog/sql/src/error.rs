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

use iceberg::{Error, ErrorKind, NamespaceIdent, Result, TableIdent};

/// Format an sqlx error into iceberg error.
pub fn from_sqlx_error(error: sqlx::Error) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "operation failed for hitting sqlx error".to_string(),
    )
    .with_source(error)
}

/// Format an sqlx error from the COMMIT statement (the `UPDATE ... WHERE metadata_location = ?`
/// CAS, or the wrapping SQL transaction's `COMMIT`) into an iceberg error, classifying
/// sent-vs-unsent (GAP_MATRIX row R157).
///
/// Once the DML has been handed to the database, a connection-level failure
/// ([`sqlx::Error::Io`], [`sqlx::Error::Protocol`], [`sqlx::Error::WorkerCrashed`]) leaves the
/// outcome AMBIGUOUS: the statement (or the transaction `COMMIT`) may have been applied
/// server-side even though the acknowledgement never arrived — mapped to
/// [`ErrorKind::CommitStateUnknown`], which `Transaction::commit` never retries (a retry could
/// apply the same update twice). Every other sqlx error is a definite local/definite-response
/// failure and keeps the [`from_sqlx_error`] mapping.
///
/// Java analogue: `JdbcTableOperations.doCommit` (iceberg-core 1.10.0, L131-145) wraps
/// connection-level `SQLException`s in `UncheckedSQLException`, which — not being a
/// `CommitFailedException` and not `CleanableFailure` — `SnapshotProducer.commit()` neither
/// retries (`onlyRetryOn(CommitFailedException.class)`) nor cleans up after (the strict-cleanup
/// gate, `SnapshotProducer.java` L472). The observable no-retry / no-cleanup / surfaced
/// semantics match; this fork additionally NAMES the ambiguity so a caller can distinguish
/// may-have-landed from safe-to-rerun.
pub fn from_sqlx_commit_error(error: sqlx::Error) -> Error {
    match &error {
        sqlx::Error::Io(_) | sqlx::Error::Protocol(_) | sqlx::Error::WorkerCrashed => Error::new(
            ErrorKind::CommitStateUnknown,
            "connection-level failure while committing; the commit may have been applied \
                 — the commit state is unknown. Verify before retrying: retrying an \
                 already-applied commit duplicates its changes.",
        )
        .with_source(error),
        _ => from_sqlx_error(error),
    }
}

pub fn no_such_namespace_err<T>(namespace: &NamespaceIdent) -> Result<T> {
    Err(Error::new(
        ErrorKind::Unexpected,
        format!("No such namespace: {namespace:?}"),
    ))
}

pub fn no_such_table_err<T>(table_ident: &TableIdent) -> Result<T> {
    Err(Error::new(
        ErrorKind::Unexpected,
        format!("No such table: {table_ident:?}"),
    ))
}

pub fn table_already_exists_err<T>(table_ident: &TableIdent) -> Result<T> {
    Err(Error::new(
        ErrorKind::Unexpected,
        format!("Table {table_ident:?} already exists."),
    ))
}

pub fn no_such_view_err<T>(view_ident: &TableIdent) -> Result<T> {
    Err(Error::new(
        ErrorKind::ViewNotFound,
        format!("No such view: {view_ident:?}"),
    ))
}

pub fn view_already_exists_err<T>(view_ident: &TableIdent) -> Result<T> {
    Err(Error::new(
        ErrorKind::ViewAlreadyExists,
        format!("View {view_ident:?} already exists."),
    ))
}

/// A table already occupies the name a view is trying to take. Mirrors Java
/// `JdbcViewOperations.doCommit` (offset 89): `AlreadyExistsException("Table with same name already
/// exists: %s")` — tables and views share one name space in a JDBC catalog.
pub fn table_with_same_name_err<T>(view_ident: &TableIdent) -> Result<T> {
    Err(Error::new(
        ErrorKind::TableAlreadyExists,
        format!("Table with same name already exists: {view_ident}"),
    ))
}

/// A view already occupies the name a table is trying to take. Mirrors Java
/// `JdbcCatalog$ViewAwareTableBuilder` (offset 27): `AlreadyExistsException("View with same name
/// already exists: %s")`.
pub fn view_with_same_name_err<T>(table_ident: &TableIdent) -> Result<T> {
    Err(Error::new(
        ErrorKind::ViewAlreadyExists,
        format!("View with same name already exists: {table_ident}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Risk (GAP_MATRIX row R157): a connection-level failure AFTER the commit CAS was handed
    /// to the database (the UPDATE, or the wrapping SQL `COMMIT`) is mapped like any other sqlx
    /// error — a retryable outer loop (or a caller) then re-runs a commit that may already be
    /// applied server-side, duplicating it. Pins that the POST-SEND connection failure shapes
    /// (`Io` / `Protocol` / `WorkerCrashed`) classify to `CommitStateUnknown`, non-retryable,
    /// with the source chain intact.
    #[test]
    fn test_post_send_connection_failure_maps_to_commit_state_unknown() {
        let io_error = sqlx::Error::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "connection reset by peer awaiting the CAS acknowledgement",
        ));
        let protocol_error = sqlx::Error::Protocol("truncated response frame".to_string());
        let worker_crashed = sqlx::Error::WorkerCrashed;

        for (label, sqlx_error) in [
            ("Io", io_error),
            ("Protocol", protocol_error),
            ("WorkerCrashed", worker_crashed),
        ] {
            let error = from_sqlx_commit_error(sqlx_error);
            assert_eq!(
                error.kind(),
                ErrorKind::CommitStateUnknown,
                "sqlx::Error::{label} after send must classify as unknown outcome"
            );
            assert!(
                !error.retryable(),
                "an unknown-outcome commit error must not advertise retryability ({label})"
            );
        }
        // The source chain survives for the Io shape (checked once; the mapping is shared).
        let error = from_sqlx_commit_error(sqlx::Error::Io(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "connection reset by peer",
        )));
        assert!(
            std::error::Error::source(&error)
                .expect("the sqlx cause must survive as the source")
                .to_string()
                .contains("connection reset by peer")
        );
    }

    /// Risk (GAP_MATRIX row R157, the over-broadening direction): a failure that PROVABLY did
    /// not apply the commit — the pool never yielded a connection (`PoolTimedOut`, nothing was
    /// sent) or the database definitively answered (`RowNotFound`, a decode error) — is
    /// classified as unknown, sending callers into needless reconciliation. Pins that definite
    /// failures keep the plain terminal mapping.
    #[test]
    fn test_definite_commit_failures_stay_terminal_not_unknown() {
        for (label, sqlx_error) in [
            ("PoolTimedOut", sqlx::Error::PoolTimedOut),
            ("RowNotFound", sqlx::Error::RowNotFound),
        ] {
            let error = from_sqlx_commit_error(sqlx_error);
            assert_eq!(
                error.kind(),
                ErrorKind::Unexpected,
                "sqlx::Error::{label} is a definite (never-sent / definite-response) failure \
                 and must keep the terminal mapping, not classify as unknown"
            );
        }
    }
}
