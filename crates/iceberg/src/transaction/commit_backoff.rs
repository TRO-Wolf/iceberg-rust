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

use std::time::{Duration, Instant};

use rand::Rng;

use crate::spec::TableProperties;

type ElapsedSource = Box<dyn FnMut() -> Duration + Send + Sync>;
type JitterSource = Box<dyn FnMut(u64) -> u64 + Send + Sync>;

pub(crate) struct CommitRetryBackoff {
    min_sleep_ms: u64,
    max_sleep_ms: u64,
    max_duration: Duration,
    max_attempts: usize,
    attempt: usize,
    elapsed: ElapsedSource,
    jitter: JitterSource,
}

impl CommitRetryBackoff {
    pub(crate) fn new(props: TableProperties) -> Self {
        let start = Instant::now();
        Self::with_sources(
            props,
            move || start.elapsed(),
            |bound| rand::rng().random_range(0..bound),
        )
    }

    fn with_sources(
        props: TableProperties,
        elapsed: impl FnMut() -> Duration + Send + Sync + 'static,
        jitter: impl FnMut(u64) -> u64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            min_sleep_ms: props.commit_min_retry_wait_ms,
            max_sleep_ms: props.commit_max_retry_wait_ms,
            max_duration: Duration::from_millis(props.commit_total_retry_timeout_ms),
            max_attempts: props.commit_num_retries.saturating_add(1),
            attempt: 0,
            elapsed: Box::new(elapsed),
            jitter: Box::new(jitter),
        }
    }
}

impl Iterator for CommitRetryBackoff {
    type Item = Duration;

    fn next(&mut self) -> Option<Duration> {
        self.attempt = self.attempt.saturating_add(1);
        if self.attempt >= self.max_attempts {
            return None;
        }
        if self.attempt > 1 && (self.elapsed)() > self.max_duration {
            return None;
        }
        let delay_ms = (self.min_sleep_ms as f64 * 2.0f64.powf((self.attempt - 1) as f64))
            .min(self.max_sleep_ms as f64)
            .clamp(0.0, i32::MAX as f64) as u64;
        let jitter_ms = (self.jitter)((delay_ms / 10).max(1));
        Some(Duration::from_millis(delay_ms.saturating_add(jitter_ms)))
    }
}

#[cfg(test)]
#[path = "commit_backoff_tests.rs"]
mod tests;
