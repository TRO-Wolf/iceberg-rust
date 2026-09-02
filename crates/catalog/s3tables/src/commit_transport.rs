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

use async_trait::async_trait;
use aws_sdk_s3tables::operation::update_table_metadata_location::UpdateTableMetadataLocationError;
#[cfg(test)]
use iceberg::table::Table;
use iceberg::{Error, ErrorKind, Result, TableIdent};

pub(crate) struct S3TablesUpdateCall {
    pub table_bucket_arn: String,
    pub namespace: String,
    pub name: String,
    pub version_token: String,
    pub metadata_location: String,
}

pub(crate) enum S3TablesCommitSend {
    Success,
    #[allow(dead_code)]
    AcceptedResponseLost,
    Transport(Box<aws_sdk_s3tables::error::SdkError<UpdateTableMetadataLocationError>>),
    #[allow(dead_code)]
    ModeledService(UpdateTableMetadataLocationError),
}

#[async_trait]
pub(crate) trait S3TablesCommitTransport: Send + Sync + Debug {
    async fn send_update_metadata_location(&self, call: S3TablesUpdateCall) -> S3TablesCommitSend;
    #[cfg(test)]
    fn catalog_commit_attempts(&self) -> u64;
}

pub(crate) enum CommitSendDisposition {
    NeverSent,
    MaybeSent,
    ResponseReceived,
}

pub(crate) fn classify_commit_send_disposition<E, R>(
    error: &aws_sdk_s3tables::error::SdkError<E, R>,
) -> CommitSendDisposition {
    use aws_sdk_s3tables::error::SdkError;
    match error {
        SdkError::ConstructionFailure(_) => CommitSendDisposition::NeverSent,
        SdkError::DispatchFailure(dispatch) if dispatch.is_user() || dispatch.is_other() => {
            CommitSendDisposition::NeverSent
        }
        SdkError::ServiceError(_) => CommitSendDisposition::ResponseReceived,
        _ => CommitSendDisposition::MaybeSent,
    }
}

pub(crate) struct LiveS3TablesCommitTransport {
    client: aws_sdk_s3tables::Client,
    attempts: AtomicU64,
}

impl Debug for LiveS3TablesCommitTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveS3TablesCommitTransport")
            .field("attempts", &self.attempts.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl LiveS3TablesCommitTransport {
    pub(crate) fn new(client: aws_sdk_s3tables::Client) -> Self {
        Self {
            client,
            attempts: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl S3TablesCommitTransport for LiveS3TablesCommitTransport {
    async fn send_update_metadata_location(&self, call: S3TablesUpdateCall) -> S3TablesCommitSend {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let builder = self
            .client
            .update_table_metadata_location()
            .table_bucket_arn(call.table_bucket_arn)
            .namespace(call.namespace)
            .name(call.name)
            .version_token(call.version_token)
            .metadata_location(call.metadata_location);
        match builder.send().await {
            Ok(_) => S3TablesCommitSend::Success,
            Err(error) => S3TablesCommitSend::Transport(Box::new(error)),
        }
    }

    #[cfg(test)]
    fn catalog_commit_attempts(&self) -> u64 {
        self.attempts.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
pub(crate) struct DiscardingS3TablesCommitTransport {
    inner: Arc<dyn S3TablesCommitTransport>,
    attempts: AtomicU64,
    observed_accepted_response_lost: AtomicBool,
}

#[cfg(test)]
impl Debug for DiscardingS3TablesCommitTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscardingS3TablesCommitTransport")
            .field("inner", &self.inner)
            .field("attempts", &self.attempts.load(Ordering::SeqCst))
            .finish()
    }
}

#[cfg(test)]
impl DiscardingS3TablesCommitTransport {
    pub(crate) fn new(inner: Arc<dyn S3TablesCommitTransport>) -> Self {
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
impl S3TablesCommitTransport for DiscardingS3TablesCommitTransport {
    async fn send_update_metadata_location(&self, call: S3TablesUpdateCall) -> S3TablesCommitSend {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        match self.inner.send_update_metadata_location(call).await {
            S3TablesCommitSend::Success => {
                self.observed_accepted_response_lost
                    .store(true, Ordering::SeqCst);
                S3TablesCommitSend::AcceptedResponseLost
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
pub(crate) enum S3TablesCommitScript {
    StopBeforeSend,
    MaybeSentLost,
    AcceptThenLose,
    Success,
    Conflict,
    Forbidden,
}

#[cfg(test)]
pub(crate) struct ScriptedS3TablesCommitTransport {
    scripts: Mutex<VecDeque<S3TablesCommitScript>>,
    attempts: AtomicU64,
    observed_accepted_response_lost: AtomicBool,
}

#[cfg(test)]
impl Debug for ScriptedS3TablesCommitTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedS3TablesCommitTransport")
            .field("attempts", &self.attempts.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl ScriptedS3TablesCommitTransport {
    pub(crate) fn new(scripts: impl IntoIterator<Item = S3TablesCommitScript>) -> Arc<Self> {
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
impl S3TablesCommitTransport for ScriptedS3TablesCommitTransport {
    async fn send_update_metadata_location(&self, _call: S3TablesUpdateCall) -> S3TablesCommitSend {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let script = {
            let mut queue = match self.scripts.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            queue.pop_front().unwrap_or(S3TablesCommitScript::Success)
        };
        match script {
            S3TablesCommitScript::StopBeforeSend => S3TablesCommitSend::Transport(Box::new(
                aws_sdk_s3tables::error::SdkError::construction_failure(boxed_err(
                    "stopped before send",
                )),
            )),
            S3TablesCommitScript::MaybeSentLost => S3TablesCommitSend::Transport(Box::new(
                aws_sdk_s3tables::error::SdkError::timeout_error(boxed_err(
                    "lost after the request may have reached S3 Tables",
                )),
            )),
            S3TablesCommitScript::AcceptThenLose => {
                self.observed_accepted_response_lost
                    .store(true, Ordering::SeqCst);
                S3TablesCommitSend::AcceptedResponseLost
            }
            S3TablesCommitScript::Success => S3TablesCommitSend::Success,
            S3TablesCommitScript::Conflict => S3TablesCommitSend::ModeledService(
                UpdateTableMetadataLocationError::ConflictException(
                    aws_sdk_s3tables::types::error::ConflictException::builder().build(),
                ),
            ),
            S3TablesCommitScript::Forbidden => S3TablesCommitSend::ModeledService(
                UpdateTableMetadataLocationError::ForbiddenException(
                    aws_sdk_s3tables::types::error::ForbiddenException::builder().build(),
                ),
            ),
        }
    }

    fn catalog_commit_attempts(&self) -> u64 {
        self.attempts.load(Ordering::SeqCst)
    }
}

fn boxed_err(message: &str) -> Box<dyn std::error::Error + Send + Sync> {
    message.to_string().into()
}

#[cfg(test)]
pub(crate) fn s3tables_commit_send_landed(send: &S3TablesCommitSend) -> bool {
    matches!(
        send,
        S3TablesCommitSend::Success | S3TablesCommitSend::AcceptedResponseLost
    )
}

pub(crate) fn map_s3tables_commit_send(
    send: S3TablesCommitSend,
    table_ident: &TableIdent,
) -> Result<()> {
    match send {
        S3TablesCommitSend::Success => Ok(()),
        S3TablesCommitSend::AcceptedResponseLost => {
            let timeout = aws_sdk_s3tables::error::SdkError::timeout_error(boxed_err(
                "S3 Tables accepted the update; the response was lost",
            ));
            Err(map_s3tables_commit_sdk_error(timeout, table_ident))
        }
        S3TablesCommitSend::Transport(error) => {
            Err(map_s3tables_commit_sdk_error(*error, table_ident))
        }
        S3TablesCommitSend::ModeledService(error) => Err(
            map_update_table_metadata_location_service_error(error, table_ident),
        ),
    }
}

pub(crate) fn map_s3tables_commit_sdk_error(
    error: aws_sdk_s3tables::error::SdkError<UpdateTableMetadataLocationError>,
    table_ident: &TableIdent,
) -> Error {
    match classify_commit_send_disposition(&error) {
        CommitSendDisposition::MaybeSent => Error::new(
            ErrorKind::CommitStateUnknown,
            format!(
                "Commit outcome unknown for table {table_ident}: the update request \
                 may have reached S3 Tables before the failure. Verify whether the \
                 commit landed before retrying: retrying an already-applied commit \
                 duplicates its changes."
            ),
        )
        .with_source(anyhow::Error::msg(format!("aws sdk error: {error:?}"))),
        CommitSendDisposition::NeverSent => Error::new(
            ErrorKind::Unexpected,
            format!(
                "Operation failed for table: {table_ident} before the update request \
                 was sent"
            ),
        )
        .with_source(anyhow::Error::msg(format!("aws sdk error: {error:?}"))),
        CommitSendDisposition::ResponseReceived => {
            map_update_table_metadata_location_service_error(
                error.into_service_error(),
                table_ident,
            )
        }
    }
}

pub(crate) fn map_update_table_metadata_location_service_error(
    error: UpdateTableMetadataLocationError,
    table_ident: &TableIdent,
) -> Error {
    match error {
        UpdateTableMetadataLocationError::ConflictException(_) => Error::new(
            ErrorKind::CatalogCommitConflicts,
            format!("Commit conflicted for table: {table_ident}"),
        )
        .with_retryable(true),
        UpdateTableMetadataLocationError::NotFoundException(_) => Error::new(
            ErrorKind::TableNotFound,
            format!("Table {table_ident} is not found"),
        ),
        UpdateTableMetadataLocationError::InternalServerErrorException(_) => Error::new(
            ErrorKind::CommitStateUnknown,
            format!(
                "Commit outcome unknown for table {table_ident}: S3 Tables failed while \
                 processing the update — it may have been applied. Verify before retrying: \
                 retrying an already-applied commit duplicates its changes."
            ),
        ),
        UpdateTableMetadataLocationError::ForbiddenException(_) => Error::new(
            ErrorKind::Unexpected,
            format!(
                "Authorization denied for table {table_ident}: S3 Tables refused the update and it was not sent as a retryable commit"
            ),
        ),
        _ => Error::new(
            ErrorKind::Unexpected,
            "Operation failed for hitting aws sdk error",
        ),
    }
    .with_source(anyhow::Error::msg(format!("aws sdk error: {error:?}")))
}

#[cfg(test)]
pub(crate) struct S3TablesCommitHarness {
    inner: Mutex<HarnessState>,
}

#[cfg(test)]
struct HarnessState {
    table: Table,
    version_token: String,
}

#[cfg(test)]
impl S3TablesCommitHarness {
    pub(crate) fn new(table: Table, version_token: String) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(HarnessState {
                table,
                version_token,
            }),
        })
    }

    pub(crate) fn pointer(&self) -> (String, String) {
        let guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        (
            guard.table.metadata_location().unwrap_or("").to_string(),
            guard.version_token.clone(),
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
        let numeric = guard
            .version_token
            .trim_start_matches('v')
            .parse::<u64>()
            .unwrap_or(0);
        guard.version_token = format!("v{}", numeric.saturating_add(1));
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
    fn test_conflict_exception_stays_retryable_conflict() {
        let error = map_update_table_metadata_location_service_error(
            UpdateTableMetadataLocationError::ConflictException(
                aws_sdk_s3tables::types::error::ConflictException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(error.kind(), iceberg::ErrorKind::CatalogCommitConflicts);
        assert!(error.retryable(), "a CAS conflict is safely retryable");
    }

    #[test]
    fn test_internal_server_error_maps_to_unknown_outcome_but_not_found_stays_terminal() {
        let unknown = map_update_table_metadata_location_service_error(
            UpdateTableMetadataLocationError::InternalServerErrorException(
                aws_sdk_s3tables::types::error::InternalServerErrorException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(unknown.kind(), iceberg::ErrorKind::CommitStateUnknown);
        assert!(
            !unknown.retryable(),
            "an unknown-outcome commit error must not advertise retryability"
        );

        let not_found = map_update_table_metadata_location_service_error(
            UpdateTableMetadataLocationError::NotFoundException(
                aws_sdk_s3tables::types::error::NotFoundException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(not_found.kind(), iceberg::ErrorKind::TableNotFound);
    }

    #[test]
    fn test_forbidden_is_terminal_not_unknown() {
        let mapped = map_update_table_metadata_location_service_error(
            UpdateTableMetadataLocationError::ForbiddenException(
                aws_sdk_s3tables::types::error::ForbiddenException::builder().build(),
            ),
            &test_table_ident(),
        );
        assert_eq!(mapped.kind(), iceberg::ErrorKind::Unexpected);
        assert!(!mapped.retryable());
        assert_ne!(mapped.kind(), iceberg::ErrorKind::CommitStateUnknown);
        assert!(mapped.message().contains("Authorization denied"));
    }

    #[test]
    fn test_commit_send_disposition_split() {
        use aws_sdk_s3tables::error::ConnectorError;
        type TestSdkError = aws_sdk_s3tables::error::SdkError<(), ()>;
        fn boxed(msg: &str) -> Box<dyn std::error::Error + Send + Sync> {
            msg.to_string().into()
        }

        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::timeout_error(boxed("timed out"))),
            CommitSendDisposition::MaybeSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::dispatch_failure(ConnectorError::io(
                boxed("reset mid-exchange")
            ))),
            CommitSendDisposition::MaybeSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::response_error(
                boxed("unparsable response"),
                ()
            )),
            CommitSendDisposition::MaybeSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::construction_failure(boxed(
                "invalid request"
            ))),
            CommitSendDisposition::NeverSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::dispatch_failure(
                ConnectorError::user(boxed("client-side setup failure"))
            )),
            CommitSendDisposition::NeverSent
        ));
        assert!(matches!(
            classify_commit_send_disposition(&TestSdkError::service_error((), ())),
            CommitSendDisposition::ResponseReceived
        ));
    }

    #[test]
    fn test_never_sent_sdk_error_maps_terminal_unexpected() {
        let error = map_s3tables_commit_sdk_error(
            aws_sdk_s3tables::error::SdkError::construction_failure(boxed_err("invalid request")),
            &test_table_ident(),
        );
        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(!error.retryable());
        assert!(error.message().contains("before the update request"));
    }
}
