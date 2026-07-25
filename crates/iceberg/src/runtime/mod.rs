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

// This module contains the async runtime abstraction for iceberg.
//
// There is exactly ONE runtime arm here (tokio); the crate carries no `wasm`/`async-std`/blocking
// alternative, so the join-failure policy below is the whole story for every `spawn` in the crate.

use std::any::Any;
use std::future::Future;
use std::panic::resume_unwind;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::task::{self, JoinError};

use crate::{Error, ErrorKind, Result};

pub struct JoinHandle<T>(task::JoinHandle<T>);

impl<T> Unpin for JoinHandle<T> {}

/// Awaiting a [`JoinHandle`] yields the task's value directly, so a task that produced NO value
/// (it panicked, or it was cancelled by a runtime shutdown) has no error channel to travel
/// through and must unwind into the waiter.
///
/// * **Task panicked.** The original panic payload is re-raised verbatim with
///   [`std::panic::resume_unwind`], which preserves the payload a downstream
///   [`std::panic::catch_unwind`] boundary (or a panic hook) would see. This mirrors Java, where
///   `Future.get()` rethrows the task's own exception at the caller (`ExecutionException::getCause`
///   is the original throwable; Iceberg's own `Tasks`/`ExceptionUtil.castAndThrow` goes further and
///   rethrows the unchecked exception unwrapped). The previous `.expect("tokio spawned task
///   failed")` DESTROYED the payload — every task panic surfaced as the same opaque string — which
///   is neither Java parity nor debuggable.
/// * **Task cancelled.** Nothing failed; there is simply no value. With `Output = T` the only
///   honest option is a panic that says so. Callers whose signature can carry an error should use
///   [`JoinHandle::try_join`] instead, which turns BOTH cases into a typed [`Error`].
///
/// Converting `Output` itself to `Result<T, Error>` would touch every awaited *and* detached
/// `spawn` site in the crate; [`JoinHandle::try_join`] gets the typed error to the call sites whose
/// signatures already return [`Result`] without that churn.
impl<T: Send + 'static> Future for JoinHandle<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let JoinHandle(handle) = self.get_mut();
        Pin::new(handle).poll(cx).map(|result| match result {
            Ok(value) => value,
            Err(join_error) if join_error.is_panic() => resume_unwind(join_error.into_panic()),
            Err(join_error) => panic!(
                "iceberg spawned task was cancelled before it produced a value: {join_error}"
            ),
        })
    }
}

impl<T: Send + 'static> JoinHandle<T> {
    /// Await the task, mapping a join failure to a typed [`Error`] instead of unwinding into the
    /// caller.
    ///
    /// Use this wherever the call site already returns [`Result`]: a spawned task that panics or
    /// is cancelled then fails that one operation instead of tearing down whatever is awaiting it.
    /// A panic is reported with its original payload rendered into the message (and logged at
    /// `error` level), so the panic is never silently swallowed.
    ///
    /// The panic is NOT re-raised, so a task that leaves shared state half-updated would have that
    /// state observed by the caller. Every current caller runs a self-contained CPU-bound decode
    /// over owned bytes, which holds no shared invariants; tasks that mutate shared state should
    /// keep awaiting the handle directly (see the [`Future`] impl above).
    pub(crate) async fn try_join(self) -> Result<T> {
        match self.0.await {
            Ok(value) => Ok(value),
            Err(join_error) => Err(join_failure_to_error(join_error)),
        }
    }
}

/// Render a [`JoinError`] as a typed [`Error`], keeping the panic payload (or the join error
/// itself, for a cancellation) reachable rather than collapsing both into one opaque string.
fn join_failure_to_error(join_error: JoinError) -> Error {
    if join_error.is_panic() {
        let payload = panic_payload_message(&*join_error.into_panic());
        tracing::error!(
            panic.payload = %payload,
            "a spawned iceberg task panicked; reporting it as a typed error"
        );
        return Error::new(
            ErrorKind::Unexpected,
            format!("spawned task panicked: {payload}"),
        );
    }

    Error::new(
        ErrorKind::Unexpected,
        "spawned task was cancelled before it produced a value",
    )
    .with_source(join_error)
}

/// Best-effort rendering of a panic payload. `panic!`/`assert!` produce either a `&'static str`
/// (literal, no formatting) or a `String` (formatted); anything else came from
/// `panic_any` and can only be described by its type.
fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[allow(dead_code)]
pub fn spawn<F>(f: F) -> JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    JoinHandle(task::spawn(f))
}

#[allow(dead_code)]
pub fn spawn_blocking<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    JoinHandle(task::spawn_blocking(f))
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::time::Duration;

    use futures::FutureExt;

    use super::*;

    #[tokio::test]
    async fn test_tokio_spawn() {
        let handle = spawn(async { 1 + 1 });
        assert_eq!(handle.await, 2);
    }

    #[tokio::test]
    async fn test_tokio_spawn_blocking() {
        let handle = spawn_blocking(|| 1 + 1);
        assert_eq!(handle.await, 2);
    }

    /// Risk pinned (audit SAF-006): awaiting a handle whose task PANICKED must re-raise the
    /// ORIGINAL panic payload in the waiter (Java `Future.get` rethrows the task's own exception),
    /// not a fixed string that erases which task failed and why. MUTATION: restoring
    /// `.map(|r| r.expect("tokio spawned task failed"))` makes the observed payload the `expect`
    /// message instead of the task's, so the payload assertion below fails (RED).
    #[tokio::test]
    async fn test_await_re_raises_the_original_panic_payload() {
        // The panicking task also runs the process panic hook, so a backtrace line for
        // `saf006-original-payload` in the test output is expected, not a failure.
        let outcome = AssertUnwindSafe(async {
            spawn(async {
                panic!("saf006-original-payload");
            })
            .await
        })
        .catch_unwind()
        .await;

        let payload = outcome.expect_err("the waiter must observe the task's panic");
        assert_eq!(
            payload
                .downcast_ref::<&'static str>()
                .copied()
                .unwrap_or("<not a &str payload>"),
            "saf006-original-payload",
            "the waiter must see the task's own panic payload, not a substitute message"
        );
    }

    /// Risk pinned (audit SAF-006): `try_join` must convert a task panic into a typed error that
    /// still names the original payload, instead of unwinding into the caller. MUTATION: dropping
    /// the `is_panic` arm of `join_failure_to_error` (reporting every join failure as a
    /// cancellation) fails both assertions below (RED).
    ///
    /// The assertions read [`Error::message`], NOT `to_string()`: `Display for Error` appends the
    /// source chain, and a `JoinError` renders the panic payload itself — so a `to_string()`
    /// `contains(payload)` check passes even when this conversion is removed entirely. Only the
    /// message proves the panic was classified as a panic.
    #[tokio::test]
    async fn test_try_join_maps_a_task_panic_to_a_typed_error() {
        let error = spawn(async {
            panic!("saf006-typed-payload");
        })
        .try_join()
        .await
        .expect_err("a panicking task must surface a typed error through try_join");

        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert_eq!(
            error.message(),
            "spawned task panicked: saf006-typed-payload",
            "the typed error must classify the failure as a panic and carry the original payload"
        );
    }

    /// Risk pinned (audit SAF-006): a CANCELLED task produced no value but nothing failed — it
    /// must not be reported as a panic, and `try_join` must not unwind. MUTATION: routing the
    /// cancelled arm of `join_failure_to_error` through `into_panic()` panics this test (tokio
    /// `JoinError::into_panic` panics when the error is a cancellation).
    #[tokio::test]
    async fn test_try_join_maps_a_cancelled_task_to_a_typed_error() {
        let handle = spawn(async {
            std::future::pending::<()>().await;
        });
        handle.0.abort();

        let error = tokio::time::timeout(Duration::from_secs(5), handle.try_join())
            .await
            .expect("try_join on an aborted task must resolve, not hang")
            .expect_err("a cancelled task must surface a typed error");

        assert_eq!(error.kind(), ErrorKind::Unexpected);
        // `message()`, not `to_string()`: the `JoinError` in the source chain also says
        // "cancelled", so a `to_string()` check would pass without this classification.
        assert_eq!(
            error.message(),
            "spawned task was cancelled before it produced a value",
            "a cancellation must be reported as a cancellation, not as a panic"
        );
        assert!(
            std::error::Error::source(&error).is_some(),
            "the JoinError must stay reachable through the error chain"
        );
    }

    /// The happy path must be untouched by the join-failure policy: `try_join` yields the task's
    /// value, wrapped in `Ok`.
    #[tokio::test]
    async fn test_try_join_returns_the_task_value() {
        let value = spawn(async { 40 + 2 })
            .try_join()
            .await
            .expect("a task that completes must yield its value");
        assert_eq!(value, 42);
    }

    /// A panic payload that is neither `&str` nor `String` must still produce a typed error rather
    /// than an unwind or an empty message.
    #[tokio::test]
    async fn test_try_join_describes_a_non_string_panic_payload() {
        let error = spawn(async {
            std::panic::panic_any(7u32);
        })
        .try_join()
        .await
        .expect_err("a panicking task must surface a typed error through try_join");

        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert_eq!(
            error.message(),
            "spawned task panicked: <non-string panic payload>",
            "an opaque payload must still be described, not dropped"
        );
    }
}
