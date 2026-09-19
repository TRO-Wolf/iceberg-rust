<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# F-COMMIT-JITTER-1 — Java's commit-retry delay, jitter included

**Date:** 2026-09-19. **Base:** `origin/main` `d09afd3d` (fork tip at branch
`fix/f-commit-jitter-1`). **Model:** swe-2-high. **Path:** step 1 MEASURE (first
commit), step 2 RED-FIRST pins, step 3 IMPLEMENT, step 4 MUTATION + gates (later
commits).

This ledger retires when the unit lands or the owner removes it.

## Defect

Barrier-released concurrent appends: RePark commits 5–7 of 16 where Spark
4.1.2's InMemoryCatalog commits 7–9 (RePark run 23b). The fork reads Java's
retry budget (`commit.retry.*` → backon `ExponentialBuilder` in
`Transaction::build_backoff`, `crates/iceberg/src/transaction/mod.rs`) but ships
NO jitter: retries that collide all sleep the identical deterministic delay and
collide again at the same instant.

## Java 1.11.0, verified against bytecode

Jar: `iceberg-spark-runtime-4.1_2.13-1.11.0.jar`, `javap -c -p`.

### `Tasks$Builder` — builder fields (javap, constructor + setters)

- `retry(int numRetries)` → `maxAttempts = numRetries + 1` (offsets 209-217:
  `iload_1, iconst_1, iadd, putfield maxAttempts`). `noRetry()` → `maxAttempts=1`.
- `exponentialBackoff(minSleep, maxSleep, maxDuration, factor)` stores the four
  fields in arg order (offsets 245-260).
- Constructor defaults: `maxAttempts=1`, `minSleepTimeMs=1000`,
  `maxSleepTimeMs=600000`, `maxDurationMs=600000`, `scaleFactor=2.0` — all
  overridden by the commit call site.

### `SnapshotProducer.commit()` — the call site (javap offsets 660-713)

`Tasks.foreach(ops)` → `retry(base.propertyAsInt("commit.retry.num-retries", 4))`
→ `exponentialBackoff(propertyAsInt("commit.retry.min-wait-ms", 100),
propertyAsInt("commit.retry.max-wait-ms", 60000),
propertyAsInt("commit.retry.total-timeout-ms", 1800000), 2.0)` →
`onlyRetryOn(CommitFailedException.class)` → `countAttempts(commitMetrics.attempts())`
→ `run(...)`. A `catch (CommitStateUnknownException) { throw }` sits ahead of the
cleanup catch (offsets 129+), so unknown outcomes are never retried — the fork's
`.when()` kind check already mirrors that.

### `Tasks$Builder.runTaskWithRetry` — the loop (javap offsets 0-371)

- Offset 0: `startTimeMs = System.currentTimeMillis()` — wall clock starts at
  method entry, BEFORE attempt 1 runs.
- Offset 4-7: local `attempt` starts 0; `iinc 5,1` at loop head → the task runs
  on attempts 1, 2, 3, … — **`attempt` starts at 1** on the first failure.
- Offset 39-44 (catch): `durationMs = currentTimeMillis() - startTimeMs`.
- Offsets 46-68, the stop test, decoded exactly:
  `throw` iff `attempt >= maxAttempts || (durationMs > maxDurationMs && attempt > 1)`.
  - The **total-timeout check is wall-clock elapsed since attempt 1**, measured
    at the failure — BEFORE the sleep is computed or taken. The upcoming sleep is
    NOT counted, so the sleep itself can overshoot the deadline.
  - **The first failure is exempt**: `attempt > 1` is required for the timeout
    stop. With `maxAttempts > 1`, Java always retries at least once even if
    attempt 1 already exceeded `maxDurationMs`.
  - Strict `>`: `durationMs == maxDurationMs` still retries.
  - `"Stopping retries after {} ms"` logs only when `durationMs > maxDurationMs`
    (offsets 71-90), then `athrow`.
- Offsets 100-245: retryability filter — `shouldRetryPredicate` else
  `onlyRetryExceptions` class-membership else `stopRetryExceptions` — AFTER the
  stop test. The fork's `.when()` runs before `backoff.next()`, which swaps the
  order, but both paths throw the same error either way: unobservable.
- Offsets 246-273, the delay:
  `delayMs = (int) Math.min(minSleepTimeMs * Math.pow(scaleFactor, attempt - 1),
  (double) maxSleepTimeMs)`. `pow` exponent is `attempt - 1` → the FIRST sleep is
  `minSleepTimeMs` exactly.
- Offsets 275-293, the jitter:
  `ThreadLocalRandom.current().nextInt(Math.max(1, (int)(delayMs * 0.1)))` →
  uniform `[0, max(1, (int)(delayMs * 0.1)))`. For `delayMs < 10` the bound is 1,
  so jitter is 0. `(int)(delayMs * 0.1)` equals integer `delayMs / 10` for every
  `delayMs` in `[0, 2^31)`: the double `0.1` is `> 1/10` by 5.5e-18, and for
  `delayMs < ~1e17` the product can never round DOWN across an integer boundary
  nor UP to `floor(delayMs/10)+1`.
- Offsets 295-347: `sleepTimeMs = delayMs + jitter` (int add — can overflow to a
  negative and throw `IllegalArgumentException` out of `Thread.sleep` only when
  `delayMs` is within ~10% of `Integer.MAX_VALUE`; unreachable via the min cap
  unless `max-wait-ms` is set near `2^31`), warn log, then
  `TimeUnit.MILLISECONDS.sleep(sleepTimeMs)`.
- `InterruptedException` during the sleep → re-interrupt + `RuntimeException`
  (350-367). Tokio's async sleep has no interrupt analogue; not ported.

## The fork's current schedule — backon 1.6.0 `ExponentialBackoff`

`build_backoff` (`mod.rs:498`): `with_min_delay(min)`, `with_max_delay(max)`,
`with_total_delay(total)`, `with_max_times(num-retries)`, `with_factor(2.0)`,
NO `.with_jitter()`. Yields (`backoff/exponential.rs:204-253`):

- Defaults (min 100ms, max 60s, retries 4, total 30min):
  sleeps `[100ms, 200ms, 400ms, 800ms]` — deterministic, identical for every
  barrier-released committer.
- Small config (min 10ms, max 50ms, retries 4): `[10ms, 20ms, 40ms, 50ms]` — the
  4th clamps at `max_delay`.
- backon's own jitter (not enabled): `delay *= (1 + fastrand.f32())` =
  `+[0, 100%)` of the delay — the wrong shape regardless (Java is `+[0, 10%)`).
- backon's `with_total_delay` sums SCHEDULED sleeps only and counts the upcoming
  sleep (`cumulative + next <= total`); Java measures wall-clock elapsed
  INCLUDING task execution, checks before the sleep, and exempts attempt 1.
  Neither subsumes the other — a faithful Java port needs the wall-clock check.
- `with_max_times(4)` yields ≤ 4 sleeps = Java's `maxAttempts = 4 + 1`. Equal.

## Divergence the port must fix

1. Jitter `+[0, max(1, delayMs/10))` added to every sleep — the actual fix for
   the lockstep wake-up.
2. Total-timeout stop on WALL-CLOCK elapsed (`Instant::elapsed() > total`),
   checked per failure, attempt-1 exempt, strict `>`.
3. `d2i` saturation: Java casts `min(...)` to `int` — clamp to
   `[0, i32::MAX]` before `as u64`.

## Status

PROVEN so far: the bytecode facts above and the backon schedule. OPEN: the pins,
the implementation, the mutation evidence — recorded after steps 2-4.
