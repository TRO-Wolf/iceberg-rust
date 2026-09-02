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

#[cfg(test)]
use std::collections::VecDeque;
use std::fmt::Debug;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(test)]
use std::sync::{Arc, Mutex};

use anyhow::anyhow;
use async_trait::async_trait;
use aws_sdk_glue::error::ProvideErrorMetadata;
use aws_sdk_glue::operation::update_table::UpdateTableError;
use aws_sdk_glue::types::TableInput;
#[cfg(test)]
use iceberg::table::Table;
use iceberg::{Error, ErrorKind, Result, TableIdent};

use crate::error::{CommitSendDisposition, classify_commit_send_disposition};

pub(crate) struct GlueUpdateTableCall {
    pub database_name: String,
    pub table_input: TableInput,
    pub version_id: Option<String>,
    pub catalog_id: Option<String>,
}

pub(crate) enum GlueCommitSend {
    Success,
    #[allow(dead_code)]
    AcceptedResponseLost,
    Transport(Box<aws_sdk_glue::error::SdkError<UpdateTableError>>),
    #[allow(dead_code)]
    ModeledService(UpdateTableError),
}

#[async_trait]
pub(crate) trait GlueCommitTransport: Send + Sync + Debug {
    async fn send_update_table(&self, call: GlueUpdateTableCall) -> GlueCommitSend;
    #[cfg(test)]
    fn catalog_commit_attempts(&self) -> u64;
}

pub(crate) struct LiveGlueCommitTransport {
    client: aws_sdk_glue::Client,
    catalog_id: Option<String>,
    attempts: AtomicU64,
}

impl Debug for LiveGlueCommitTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveGlueCommitTransport")
            .field("catalog_id", &self.catalog_id)
            .field("attempts", &self.attempts.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl LiveGlueCommitTransport {
    pub(crate) fn new(client: aws_sdk_glue::Client, catalog_id: Option<String>) -> Self {
        Self {
            client,
            catalog_id,
            attempts: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl GlueCommitTransport for LiveGlueCommitTransport {
    async fn send_update_table(&self, call: GlueUpdateTableCall) -> GlueCommitSend {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let mut builder = self
            .client
            .update_table()
            .database_name(call.database_name)
            .set_skip_archive(Some(true))
            .table_input(call.table_input);
        if let Some(version_id) = call.version_id {
            builder = builder.version_id(version_id);
        }
        let catalog_id = call.catalog_id.or_else(|| self.catalog_id.clone());
        if let Some(catalog_id) = catalog_id {
            builder = builder.catalog_id(catalog_id);
        }
        match builder.send().await {
            Ok(_) => GlueCommitSend::Success,
            Err(error) => GlueCommitSend::Transport(Box::new(error)),
        }
    }

    #[cfg(test)]
    fn catalog_commit_attempts(&self) -> u64 {
        self.attempts.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
pub(crate) struct DiscardingGlueCommitTransport {
    inner: Arc<dyn GlueCommitTransport>,
    attempts: AtomicU64,
    observed_accepted_response_lost: AtomicBool,
}

#[cfg(test)]
impl Debug for DiscardingGlueCommitTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscardingGlueCommitTransport")
            .field("inner", &self.inner)
            .field("attempts", &self.attempts.load(Ordering::SeqCst))
            .finish()
    }
}

#[cfg(test)]
impl DiscardingGlueCommitTransport {
    pub(crate) fn new(inner: Arc<dyn GlueCommitTransport>) -> Self {
        Self {
            inner,
            attempts: AtomicU64::new(0),
            observed_accepted_response_lost: AtomicBool::new(false),
        }
    }

    pub(crate) fn observed_accepted_response_lost(&self) -> bool {
        self.observed_accepted_response_lost.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
#[async_trait]
impl GlueCommitTransport for DiscardingGlueCommitTransport {
    async fn send_update_table(&self, call: GlueUpdateTableCall) -> GlueCommitSend {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        match self.inner.send_update_table(call).await {
            GlueCommitSend::Success => {
                self.observed_accepted_response_lost
                    .store(true, Ordering::SeqCst);
                GlueCommitSend::AcceptedResponseLost
            }
            other => other,
        }
    }

    fn catalog_commit_attempts(&self) -> u64 {
        self.attempts.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) enum GlueCommitScript {
    StopBeforeSend,
    MaybeSentLost,
    AcceptThenLose,
    Success,
    ConcurrentModification,
    AccessDenied,
}

#[cfg(test)]
pub(crate) struct ScriptedGlueCommitTransport {
    scripts: Mutex<VecDeque<GlueCommitScript>>,
    attempts: AtomicU64,
    observed_accepted_response_lost: AtomicBool,
}

#[cfg(test)]
impl Debug for ScriptedGlueCommitTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedGlueCommitTransport")
            .field("attempts", &self.attempts.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl ScriptedGlueCommitTransport {
    pub(crate) fn new(scripts: impl IntoIterator<Item = GlueCommitScript>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(scripts.into_iter().collect()),
            attempts: AtomicU64::new(0),
            observed_accepted_response_lost: AtomicBool::new(false),
        })
    }

    pub(crate) fn observed_accepted_response_lost(&self) -> bool {
        self.observed_accepted_response_lost.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
#[async_trait]
impl GlueCommitTransport for ScriptedGlueCommitTransport {
    async fn send_update_table(&self, _call: GlueUpdateTableCall) -> GlueCommitSend {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let script = {
            let mut queue = match self.scripts.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            queue.pop_front().unwrap_or(GlueCommitScript::Success)
        };
        match script {
            GlueCommitScript::StopBeforeSend => GlueCommitSend::Transport(Box::new(
                aws_sdk_glue::error::SdkError::construction_failure(boxed_err(
                    "stopped before send",
                )),
            )),
            GlueCommitScript::MaybeSentLost => {
                GlueCommitSend::Transport(Box::new(aws_sdk_glue::error::SdkError::timeout_error(
                    boxed_err("lost after the request may have reached Glue"),
                )))
            }
            GlueCommitScript::AcceptThenLose => {
                self.observed_accepted_response_lost
                    .store(true, Ordering::SeqCst);
                GlueCommitSend::AcceptedResponseLost
            }
            GlueCommitScript::Success => GlueCommitSend::Success,
            GlueCommitScript::ConcurrentModification => {
                GlueCommitSend::ModeledService(UpdateTableError::ConcurrentModificationException(
                    aws_sdk_glue::types::error::ConcurrentModificationException::builder().build(),
                ))
            }
            GlueCommitScript::AccessDenied => {
                GlueCommitSend::ModeledService(UpdateTableError::generic(
                    aws_sdk_glue::error::ErrorMetadata::builder()
                        .code("AccessDeniedException")
                        .message("not authorized to update table")
                        .build(),
                ))
            }
        }
    }

    fn catalog_commit_attempts(&self) -> u64 {
        self.attempts.load(Ordering::SeqCst)
    }
}

fn boxed_err(message: &str) -> Box<dyn std::error::Error + Send + Sync> {
    message.to_string().into()
}

pub(crate) fn map_glue_commit_send(send: GlueCommitSend, table_ident: &TableIdent) -> Result<()> {
    match send {
        GlueCommitSend::Success => Ok(()),
        GlueCommitSend::AcceptedResponseLost => {
            let timeout = aws_sdk_glue::error::SdkError::timeout_error(boxed_err(
                "Glue accepted the update; the response was lost",
            ));
            Err(map_glue_commit_sdk_error(timeout, table_ident))
        }
        GlueCommitSend::Transport(error) => Err(map_glue_commit_sdk_error(*error, table_ident)),
        GlueCommitSend::ModeledService(error) => {
            Err(map_update_table_service_error(error, table_ident))
        }
    }
}

#[cfg(test)]
pub(crate) fn glue_commit_send_landed(send: &GlueCommitSend) -> bool {
    matches!(
        send,
        GlueCommitSend::Success | GlueCommitSend::AcceptedResponseLost
    )
}

pub(crate) fn map_glue_commit_sdk_error(
    error: aws_sdk_glue::error::SdkError<UpdateTableError>,
    table_ident: &TableIdent,
) -> Error {
    match classify_commit_send_disposition(&error) {
        CommitSendDisposition::MaybeSent => Error::new(
            ErrorKind::CommitStateUnknown,
            format!(
                "Commit outcome unknown for table {table_ident}: the update request \
                 may have reached Glue before the failure. Verify whether the commit \
                 landed before retrying: retrying an already-applied commit \
                 duplicates its changes."
            ),
        )
        .with_source(anyhow!("aws sdk error: {error:?}")),
        CommitSendDisposition::NeverSent => Error::new(
            ErrorKind::Unexpected,
            format!(
                "Operation failed for table: {table_ident} before the update request \
                 was sent"
            ),
        )
        .with_source(anyhow!("aws sdk error: {error:?}")),
        CommitSendDisposition::ResponseReceived => {
            map_update_table_service_error(error.into_service_error(), table_ident)
        }
    }
}

pub(crate) fn map_update_table_service_error(
    error: UpdateTableError,
    table_ident: &TableIdent,
) -> Error {
    if error.code() == Some("AccessDeniedException") {
        return authorization_denied(table_ident, &error);
    }
    match error {
        UpdateTableError::EntityNotFoundException(_) => Error::new(
            ErrorKind::TableNotFound,
            format!("Table {table_ident} is not found"),
        ),
        UpdateTableError::ConcurrentModificationException(_) => Error::new(
            ErrorKind::CatalogCommitConflicts,
            format!("Commit failed for table: {table_ident}"),
        )
        .with_retryable(true),
        UpdateTableError::InternalServiceException(_)
        | UpdateTableError::OperationTimeoutException(_) => Error::new(
            ErrorKind::CommitStateUnknown,
            format!(
                "Commit outcome unknown for table {table_ident}: Glue failed (or timed out) \
                 while processing the update — it may have been applied. Verify before \
                 retrying: retrying an already-applied commit duplicates its changes."
            ),
        ),
        _ => Error::new(
            ErrorKind::Unexpected,
            format!("Operation failed for table: {table_ident} for hitting aws sdk error"),
        ),
    }
    .with_source(anyhow!("aws sdk error: {error:?}"))
}

fn authorization_denied(table_ident: &TableIdent, error: &UpdateTableError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        format!(
            "Authorization denied for table {table_ident}: Glue refused the update and it was not sent as a retryable commit"
        ),
    )
    .with_source(anyhow!("aws sdk error: {error:?}"))
}

#[cfg(test)]
pub(crate) struct GlueCommitHarness {
    inner: Mutex<HarnessState>,
}

#[cfg(test)]
struct HarnessState {
    table: Table,
    version_id: Option<String>,
}

#[cfg(test)]
impl GlueCommitHarness {
    pub(crate) fn new(table: Table, version_id: Option<String>) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(HarnessState { table, version_id }),
        })
    }

    pub(crate) fn pointer(&self) -> (String, Option<String>) {
        let guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        (
            guard.table.metadata_location().unwrap_or("").to_string(),
            guard.version_id.clone(),
        )
    }

    pub(crate) fn table(&self) -> Table {
        let guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.table.clone()
    }

    pub(crate) fn publish(&self, table: Table) {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let next = match guard.version_id.as_deref() {
            Some(current) => {
                let numeric = current.trim_start_matches('v').parse::<u64>().unwrap_or(0);
                format!("v{}", numeric.saturating_add(1))
            }
            None => "v1".to_string(),
        };
        guard.version_id = Some(next);
        guard.table = table;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_table_ident() -> TableIdent {
        TableIdent::from_strs(["ns1", "test1"]).expect("build test table ident")
    }

    #[test]
    fn test_concurrent_modification_stays_retryable_conflict() {
        let error = map_update_table_service_error(
            UpdateTableError::ConcurrentModificationException(
                aws_sdk_glue::types::error::ConcurrentModificationException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(error.kind(), iceberg::ErrorKind::CatalogCommitConflicts);
        assert!(error.retryable(), "a CAS conflict is safely retryable");
    }

    #[test]
    fn test_internal_service_and_operation_timeout_map_to_unknown_outcome() {
        for error in [
            UpdateTableError::InternalServiceException(
                aws_sdk_glue::types::error::InternalServiceException::builder().build(),
            ),
            UpdateTableError::OperationTimeoutException(
                aws_sdk_glue::types::error::OperationTimeoutException::builder().build(),
            ),
        ] {
            let mapped = map_update_table_service_error(error, &test_table_ident());
            assert_eq!(mapped.kind(), iceberg::ErrorKind::CommitStateUnknown);
            assert!(
                !mapped.retryable(),
                "an unknown-outcome commit error must not advertise retryability"
            );
        }
    }

    #[test]
    fn test_definite_service_rejections_stay_terminal() {
        let not_found = map_update_table_service_error(
            UpdateTableError::EntityNotFoundException(
                aws_sdk_glue::types::error::EntityNotFoundException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(not_found.kind(), iceberg::ErrorKind::TableNotFound);

        let invalid_input = map_update_table_service_error(
            UpdateTableError::InvalidInputException(
                aws_sdk_glue::types::error::InvalidInputException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(invalid_input.kind(), iceberg::ErrorKind::Unexpected);
        assert!(!invalid_input.retryable());
    }

    #[test]
    fn test_access_denied_is_terminal_not_unknown() {
        let mapped = map_update_table_service_error(
            UpdateTableError::generic(
                aws_sdk_glue::error::ErrorMetadata::builder()
                    .code("AccessDeniedException")
                    .message("not authorized")
                    .build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(mapped.kind(), iceberg::ErrorKind::Unexpected);
        assert!(!mapped.retryable());
        assert_ne!(mapped.kind(), iceberg::ErrorKind::CommitStateUnknown);
        assert!(mapped.message().contains("Authorization denied"));
    }

    #[test]
    fn test_never_sent_sdk_error_maps_terminal_unexpected() {
        let error = map_glue_commit_sdk_error(
            aws_sdk_glue::error::SdkError::construction_failure(boxed_err("invalid request")),
            &test_table_ident(),
        );
        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(!error.retryable());
        assert!(error.message().contains("before the update request"));
    }
}
