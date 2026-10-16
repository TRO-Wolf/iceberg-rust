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
pub(crate) fn classify_commit_send_disposition<E, R>(
    error: &aws_sdk_glue::error::SdkError<E, R>,
) -> CommitSendDisposition {
    use aws_sdk_glue::error::SdkError;
    match error {
        SdkError::ConstructionFailure(_) => CommitSendDisposition::NeverSent,
        SdkError::DispatchFailure(dispatch)
            if dispatch.as_other() == Some(aws_config::retry::ErrorKind::TransientError) =>
        {
            CommitSendDisposition::MaybeSent
        }
        SdkError::DispatchFailure(dispatch)
            if dispatch.is_user()
                || dispatch.is_other()
                || connection_refused(dispatch.as_connector_error()) =>
        {
            CommitSendDisposition::NeverSent
        }
        SdkError::ServiceError(_) => CommitSendDisposition::ResponseReceived,
        _ => CommitSendDisposition::MaybeSent,
    }
}

const MAX_SOURCE_DEPTH: usize = 16;

fn connection_refused(error: Option<&aws_sdk_glue::error::ConnectorError>) -> bool {
    let first = error.and_then(std::error::Error::source);
    std::iter::successors(first, |error| error.source())
        .take(MAX_SOURCE_DEPTH)
        .filter_map(|error| error.downcast_ref::<std::io::Error>())
        .any(|error| error.kind() == std::io::ErrorKind::ConnectionRefused)
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
    fn test_incomplete_message_classifies_maybe_sent_and_unclassified_other_stays_never_sent() {
        let closed = TestSdkError::dispatch_failure(ConnectorError::other(
            boxed("connection closed before message completed"),
            Some(aws_config::retry::ErrorKind::TransientError),
        ));
        assert!(matches!(
            classify_commit_send_disposition(&closed),
            CommitSendDisposition::MaybeSent
        ));
        let unclassified =
            TestSdkError::dispatch_failure(ConnectorError::other(boxed("unrecognized"), None));
        assert!(matches!(
            classify_commit_send_disposition(&unclassified),
            CommitSendDisposition::NeverSent
        ));
    }

    #[test]
    fn test_refused_connection_classifies_never_sent_and_reset_stays_maybe_sent() {
        let refused = TestSdkError::dispatch_failure(ConnectorError::io(Box::new(
            std::io::Error::from(std::io::ErrorKind::ConnectionRefused),
        )));
        assert!(matches!(
            classify_commit_send_disposition(&refused),
            CommitSendDisposition::NeverSent
        ));
        let reset = TestSdkError::dispatch_failure(ConnectorError::io(Box::new(
            std::io::Error::from(std::io::ErrorKind::ConnectionReset),
        )));
        assert!(matches!(
            classify_commit_send_disposition(&reset),
            CommitSendDisposition::MaybeSent
        ));
    }

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
