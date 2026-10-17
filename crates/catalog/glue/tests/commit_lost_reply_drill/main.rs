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

mod endpoint;

use endpoint::*;
use iceberg::{Catalog, ErrorKind};

#[tokio::test]
async fn glue_lost_reply_is_not_resent_and_reconciles_as_landed() {
    let measured = drill(&[CommitScript::ApplyThenReset], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenReset
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_lost_unapplied_commit_is_unknown_and_keeps_the_pre_state() {
    let measured = drill(&[CommitScript::DropThenReset], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenReset
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_landed_commit_answered_conflict_lands_once() {
    let measured = drill(&[CommitScript::ApplyThenConflict], false).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenConflict
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_reads_keep_their_sdk_retry() {
    let endpoint = FakeGlue::start(Connections::KeepAlive).await;
    let catalog = glue_catalog(&endpoint.url).await;
    let table = seeded_table(&catalog).await;
    endpoint.reset_next_reads(1);
    let reloaded = catalog
        .load_table(table.identifier())
        .await
        .expect("a reset read is retried by the SDK");
    assert_eq!(reloaded.metadata_location(), table.metadata_location());
    let reads: Vec<Answer> = endpoint
        .log()
        .iter()
        .filter(|logged| logged.operation == "GetTable")
        .map(|logged| logged.answer)
        .collect();
    assert_eq!(reads, vec![Answer::Reset, Answer::Answered]);
}

#[tokio::test]
async fn glue_lost_reply_on_a_closed_connection_reconciles_as_landed() {
    let measured = drill(&[CommitScript::ApplyThenClose], true).await;
    let &[Some(sent)] = measured.sent_snapshot_ids.as_slice() else {
        panic!("one commit request naming one snapshot");
    };
    assert_eq!(measured.marked_snapshot_ids, vec![sent]);
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenClosed
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_unapplied_commit_on_a_closed_connection_is_unknown_and_keeps_the_pre_state() {
    let measured = drill(&[CommitScript::DropThenClose], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenClosed
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_refused_commit_connection_is_never_sent_and_not_reconciled() {
    let measured = drill_refused_commit_connection().await;
    assert!(measured.commits().is_empty());
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Err(ErrorKind::Unexpected));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_server_error_with_an_unmodeled_code_after_the_commit_reconciles_as_landed() {
    let measured = drill(
        &[CommitScript::ApplyThenError(500, "InternalFailure")],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenError(500)
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_server_error_with_an_unmodeled_code_before_the_commit_is_unknown_and_keeps_the_pre_state()
 {
    let measured = drill(&[CommitScript::DropThenError(500, "InternalFailure")], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenError(500)
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_modeled_server_error_after_the_commit_reconciles_as_landed() {
    let measured = drill(
        &[CommitScript::ApplyThenError(
            500,
            "InternalServiceException",
        )],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenError(500)
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_unparsable_reply_after_the_commit_reconciles_as_landed() {
    let measured = drill(&[CommitScript::ApplyThenGarbage], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenGarbage
    )]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_unparsable_reply_before_the_commit_is_unknown_and_keeps_the_pre_state() {
    let measured = drill(&[CommitScript::DropThenGarbage], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenGarbage
    )]);
    assert_eq!(measured.outcome, Err(ErrorKind::CommitStateUnknown));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_client_error_with_an_unmodeled_code_is_definitive_and_not_reconciled() {
    let measured = drill(
        &[CommitScript::DropThenError(400, "ValidationException")],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::DroppedThenError(400)
    )]);
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Err(ErrorKind::Unexpected));
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_an_earlier_attempt_that_lands_late_is_found_before_the_rebase() {
    let measured = drill(
        &[
            CommitScript::DeferThenConflict,
            CommitScript::LandDeferredThenConflict,
        ],
        false,
    )
    .await;
    assert_eq!(measured.commits(), vec![
        (Some("1"), Answer::DeferredThenConflict),
        (Some("1"), Answer::LandedDeferredThenConflict),
    ]);
    let &[Some(first), Some(second)] = measured.sent_snapshot_ids.as_slice() else {
        panic!("two commit requests, each naming one snapshot");
    };
    assert_ne!(first, second);
    assert_eq!(measured.marked_snapshot_ids, vec![first]);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_throttled_commit_is_resent_unchanged_and_lands_once() {
    let measured = drill(
        &[
            CommitScript::DropThenError(400, "ThrottlingException"),
            CommitScript::DropThenError(400, "ThrottlingException"),
        ],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![
        (Some("1"), Answer::DroppedThenError(400)),
        (Some("1"), Answer::DroppedThenError(400)),
        (Some("1"), Answer::Answered),
    ]);
    let sent = measured.metadata_sent();
    assert!(sent.iter().all(|location| *location == sent[0]));
    let &[Some(first), Some(second), Some(third)] = measured.sent_snapshot_ids.as_slice() else {
        panic!("three commit requests, each naming one snapshot");
    };
    assert_eq!((first, second), (third, third));
    assert_eq!(measured.reads_between_commits(), 0);
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshot_ids, vec![first]);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_throttled_on_every_send_fails_terminally_after_three_sends() {
    let measured = drill(
        &[
            CommitScript::DropThenError(400, "ThrottlingException"),
            CommitScript::DropThenError(400, "ThrottlingException"),
            CommitScript::DropThenError(400, "ThrottlingException"),
            CommitScript::DropThenError(400, "ThrottlingException"),
        ],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![
        (Some("1"), Answer::DroppedThenError(400)),
        (Some("1"), Answer::DroppedThenError(400)),
        (Some("1"), Answer::DroppedThenError(400)),
    ]);
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Err(ErrorKind::Unexpected));
    let message = measured.message.as_deref().unwrap_or_default();
    assert!(message.contains("nothing was applied"), "{message}");
    assert!(message.contains("on each of 3 sends"), "{message}");
    assert_eq!(measured.marked_snapshots, 0);
    assert_eq!(measured.rows, 0);
}

#[tokio::test]
async fn glue_a_late_landing_attempt_is_found_by_the_unknown_outcome_read_back() {
    let measured = drill(
        &[
            CommitScript::DropThenError(400, "ConcurrentModificationException"),
            CommitScript::DeferThenConflict,
            CommitScript::LandDeferredThenReset,
        ],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![
        (Some("1"), Answer::DroppedThenError(400)),
        (Some("1"), Answer::DeferredThenConflict),
        (Some("1"), Answer::LandedDeferredThenReset),
    ]);
    let &[Some(_), Some(second), Some(_)] = measured.sent_snapshot_ids.as_slice() else {
        panic!("three commit requests, each naming one snapshot");
    };
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshot_ids, vec![second]);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_unreadable_success_reply_after_the_commit_is_success_without_a_read_back() {
    let measured = drill(&[CommitScript::ApplyThenUnreadable], true).await;
    assert_eq!(measured.commits(), vec![(
        Some("1"),
        Answer::AppliedThenUnreadable
    )]);
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshots, 1);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}

#[tokio::test]
async fn glue_slow_down_on_a_503_is_resent_unchanged_and_lands_once() {
    let measured = drill(
        &[
            CommitScript::DropThenError(503, "SlowDown"),
            CommitScript::DropThenError(503, "SlowDown"),
        ],
        true,
    )
    .await;
    assert_eq!(measured.commits(), vec![
        (Some("1"), Answer::DroppedThenError(503)),
        (Some("1"), Answer::DroppedThenError(503)),
        (Some("1"), Answer::Answered),
    ]);
    let sent = measured.metadata_sent();
    assert!(sent.iter().all(|location| *location == sent[0]));
    let &[Some(first), Some(second), Some(third)] = measured.sent_snapshot_ids.as_slice() else {
        panic!("three commit requests, each naming one snapshot");
    };
    assert_eq!((first, second), (third, third));
    assert_eq!(measured.reads_between_commits(), 0);
    assert_eq!(measured.reads(), 2);
    assert_eq!(measured.outcome, Ok(()));
    assert_eq!(measured.marked_snapshot_ids, vec![first]);
    assert_eq!(measured.rows, ROWS_PER_APPEND);
}
