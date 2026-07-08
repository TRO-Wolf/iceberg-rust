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

use std::fmt::Debug;

use anyhow::anyhow;
use iceberg::{Error, ErrorKind};

/// Format AWS SDK error into iceberg error
pub(crate) fn from_aws_sdk_error<T>(error: aws_sdk_glue::error::SdkError<T>) -> Error
where T: Debug {
    Error::new(
        ErrorKind::Unexpected,
        "Operation failed for hitting aws sdk error".to_string(),
    )
    .with_source(anyhow!("aws sdk error: {error:?}"))
}

/// Format AWS Build error into iceberg error
pub(crate) fn from_aws_build_error(error: aws_sdk_glue::error::BuildError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Operation failed for hitting aws build error".to_string(),
    )
    .with_source(anyhow!("aws build error: {error:?}"))
}

/// Where a failed AWS SDK COMMIT call stopped, classified sent-vs-unsent (GAP_MATRIX row R157).
pub(crate) enum CommitSendDisposition {
    /// The request provably never left the client — the failure keeps its terminal mapping.
    NeverSent,
    /// The request MAY have reached the service: the commit outcome is ambiguous and must map
    /// to `ErrorKind::CommitStateUnknown` (retrying could apply the same update twice).
    MaybeSent,
    /// The service definitively responded with a modeled error — classify by the service error.
    ResponseReceived,
}

/// Classify the transport layer of a failed SDK call on the COMMIT path (`update_table`).
///
/// Java analogue: `GlueTableOperations.doCommit` (iceberg-aws 1.10.0, L162-191) rethrows
/// `CommitFailedException`, reconciles every other `RuntimeException` via `checkCommitStatus`
/// (`BaseMetastoreOperations.CommitStatus {SUCCESS, FAILURE, UNKNOWN}`), and surfaces
/// `CommitStateUnknownException(persistFailure)` when the outcome cannot be confirmed. This
/// offline slice ports the CLASSIFICATION (which failures are ambiguous); the
/// reconciliation-by-refresh step needs a live catalog and stays with the credentialed slice.
///
/// The `ConnectorError` split: an `is_user()` / `is_other()` dispatch failure is a client-side
/// setup problem (the request was never written); an `is_io()` / `is_timeout()` dispatch
/// failure can occur mid-exchange, after the request bytes reached the service — the AWS SDK
/// does not distinguish connect-refused from reset-after-send, so the ambiguous side is chosen
/// (needless reconciliation is safe; a duplicate commit is not). Unknown future variants of the
/// `#[non_exhaustive]` `SdkError` also classify ambiguous for the same reason.
pub(crate) fn classify_commit_send_disposition<E, R>(
    error: &aws_sdk_glue::error::SdkError<E, R>,
) -> CommitSendDisposition {
    use aws_sdk_glue::error::SdkError;
    match error {
        SdkError::ConstructionFailure(_) => CommitSendDisposition::NeverSent,
        SdkError::DispatchFailure(dispatch) if dispatch.is_user() || dispatch.is_other() => {
            CommitSendDisposition::NeverSent
        }
        SdkError::ServiceError(_) => CommitSendDisposition::ResponseReceived,
        // TimeoutError (the operation timed out awaiting completion), io/timeout dispatch
        // failures, ResponseError (a response arrived but could not be understood), and any
        // future variant: the request may have reached the service.
        _ => CommitSendDisposition::MaybeSent,
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_glue::error::ConnectorError;

    use super::*;

    type TestSdkError = aws_sdk_glue::error::SdkError<(), ()>;

    fn boxed(msg: &str) -> Box<dyn std::error::Error + Send + Sync> {
        msg.to_string().into()
    }

    /// Risk (GAP_MATRIX row R157): a POST-SEND ambiguous failure (operation timeout, io/timeout
    /// dispatch failure, unparsable response) is classified NeverSent/terminal — an outer loop
    /// or caller then re-runs a commit Glue may already have applied (duplicate rows). Pins the
    /// ambiguous side of the split.
    #[test]
    fn test_post_send_ambiguous_sdk_failures_classify_maybe_sent() {
        let ambiguous: Vec<(&str, TestSdkError)> = vec![
            ("timeout", TestSdkError::timeout_error(boxed("timed out"))),
            (
                "dispatch-io",
                TestSdkError::dispatch_failure(ConnectorError::io(boxed("reset mid-exchange"))),
            ),
            (
                "dispatch-timeout",
                TestSdkError::dispatch_failure(ConnectorError::timeout(boxed("stalled"))),
            ),
            (
                "response-error",
                TestSdkError::response_error(boxed("unparsable response"), ()),
            ),
        ];
        for (label, sdk_error) in ambiguous {
            assert!(
                matches!(
                    classify_commit_send_disposition(&sdk_error),
                    CommitSendDisposition::MaybeSent
                ),
                "{label}: a post-send failure must classify as MaybeSent (unknown outcome)"
            );
        }
    }

    /// Risk (GAP_MATRIX row R157, the over-broadening direction): a failure that provably never
    /// sent the request (request construction, client-side dispatch setup) — or a definitive
    /// service response — is classified ambiguous, sending callers into needless commit
    /// reconciliation. Pins the never-sent and response-received sides of the split.
    #[test]
    fn test_never_sent_and_service_response_do_not_classify_maybe_sent() {
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
}
