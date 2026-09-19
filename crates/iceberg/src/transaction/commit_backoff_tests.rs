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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::CommitRetryBackoff;
use crate::spec::TableProperties;

fn props(min_ms: u64, max_ms: u64, total_ms: u64, retries: usize) -> TableProperties {
    TableProperties::try_from(&HashMap::from([
        (
            TableProperties::PROPERTY_COMMIT_MIN_RETRY_WAIT_MS.to_string(),
            min_ms.to_string(),
        ),
        (
            TableProperties::PROPERTY_COMMIT_MAX_RETRY_WAIT_MS.to_string(),
            max_ms.to_string(),
        ),
        (
            TableProperties::PROPERTY_COMMIT_TOTAL_RETRY_TIME_MS.to_string(),
            total_ms.to_string(),
        ),
        (
            TableProperties::PROPERTY_COMMIT_NUM_RETRIES.to_string(),
            retries.to_string(),
        ),
    ]))
    .expect("test table properties must parse")
}

fn backoff_with(
    props: TableProperties,
    elapsed_ms: Vec<u64>,
    jitter_ms: Vec<u64>,
) -> CommitRetryBackoff {
    let mut elapsed = elapsed_ms.into_iter();
    let mut jitter = jitter_ms.into_iter();
    CommitRetryBackoff::with_sources(
        props,
        move || Duration::from_millis(elapsed.next().unwrap_or(0)),
        move |_| jitter.next().unwrap_or(0),
    )
}

#[test]
fn default_schedule_matches_java_formula_with_injected_jitter() {
    let mut backoff = backoff_with(props(100, 60_000, 1_800_000, 4), vec![], vec![3, 7, 0, 79]);
    let sleeps: Vec<Duration> = backoff.by_ref().collect();
    assert_eq!(
        vec![
            Duration::from_millis(103),
            Duration::from_millis(207),
            Duration::from_millis(400),
            Duration::from_millis(879),
        ],
        sleeps
    );
}

#[test]
fn small_config_schedule_matches_java_formula_with_injected_jitter() {
    let mut backoff = backoff_with(props(10, 50, 1_800_000, 4), vec![], vec![0, 1, 3, 4]);
    let sleeps: Vec<Duration> = backoff.by_ref().collect();
    assert_eq!(
        vec![
            Duration::from_millis(10),
            Duration::from_millis(21),
            Duration::from_millis(43),
            Duration::from_millis(54),
        ],
        sleeps
    );
}

#[test]
fn jitter_bound_is_java_next_int_bound() {
    let bounds = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&bounds);
    let mut backoff = CommitRetryBackoff::with_sources(
        props(100, 60_000, 1_800_000, 4),
        || Duration::ZERO,
        move |bound| {
            captured.lock().expect("bounds lock").push(bound);
            0
        },
    );
    let _: Vec<Duration> = backoff.by_ref().collect();
    assert_eq!(vec![10, 20, 40, 80], *bounds.lock().expect("bounds lock"));
}

#[test]
fn jittered_sleep_stays_inside_java_bound() {
    let mut backoff = CommitRetryBackoff::with_sources(
        props(100, 60_000, 1_800_000, 4),
        || Duration::ZERO,
        |bound| bound - 1,
    );
    let sleeps: Vec<Duration> = backoff.by_ref().collect();
    assert_eq!(
        vec![
            Duration::from_millis(109),
            Duration::from_millis(219),
            Duration::from_millis(439),
            Duration::from_millis(879),
        ],
        sleeps
    );
}

#[test]
fn sub_ten_ms_delay_has_unit_jitter_bound() {
    let bounds = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&bounds);
    let mut backoff = CommitRetryBackoff::with_sources(
        props(5, 50, 1_800_000, 2),
        || Duration::ZERO,
        move |bound| {
            captured.lock().expect("bounds lock").push(bound);
            bound - 1
        },
    );
    let sleeps: Vec<Duration> = backoff.by_ref().collect();
    assert_eq!(
        vec![Duration::from_millis(5), Duration::from_millis(10)],
        sleeps
    );
    assert_eq!(vec![1, 1], *bounds.lock().expect("bounds lock"));
}

#[test]
fn delay_caps_at_max_wait() {
    let mut backoff = backoff_with(props(100, 150, 1_800_000, 3), vec![], vec![0, 0, 0]);
    let sleeps: Vec<Duration> = backoff.by_ref().collect();
    assert_eq!(
        vec![
            Duration::from_millis(100),
            Duration::from_millis(150),
            Duration::from_millis(150),
        ],
        sleeps
    );
}

#[test]
fn first_failure_is_exempt_from_total_timeout() {
    let mut backoff = backoff_with(props(100, 60_000, 1_000, 4), vec![u64::MAX], vec![0]);
    assert_eq!(Some(Duration::from_millis(100)), backoff.next());
    assert_eq!(None, backoff.next());
}

#[test]
fn total_timeout_stops_retries_after_first_attempt() {
    let mut backoff = backoff_with(props(100, 60_000, 1_000, 4), vec![0, 0, 1_001], vec![
        0, 0, 0,
    ]);
    let sleeps: Vec<Duration> = backoff.by_ref().collect();
    assert_eq!(
        vec![Duration::from_millis(100), Duration::from_millis(200)],
        sleeps
    );
}

#[test]
fn elapsed_equal_to_timeout_still_retries() {
    let mut backoff = backoff_with(props(100, 60_000, 1_000, 4), vec![1_000], vec![0, 0, 0, 0]);
    let sleeps: Vec<Duration> = backoff.by_ref().collect();
    assert_eq!(4, sleeps.len());
}

#[test]
fn num_retries_bounds_the_number_of_sleeps() {
    let mut backoff = backoff_with(props(100, 60_000, 1_800_000, 4), vec![], vec![]);
    assert_eq!(4, backoff.by_ref().count());
}

#[test]
fn zero_retries_never_sleeps() {
    let mut backoff = backoff_with(props(100, 60_000, 1_800_000, 0), vec![], vec![]);
    assert_eq!(None, backoff.next());
}
