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

//! Temporal value conversions for dates, times, and timestamps

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, TimeZone, Utc};

pub(crate) mod date {
    use super::*;

    pub(crate) fn date_to_days(date: &NaiveDate) -> i32 {
        date.signed_duration_since(
            // This is always the same and shouldn't fail
            NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
        )
        .num_days() as i32
    }

    pub(crate) fn days_to_date(days: i32) -> Option<NaiveDate> {
        // A `days`-since-epoch value near the `i32` extremes (e.g. from corrupt/hostile
        // on-disk bytes) pushes the date past chrono's representable range, which made the
        // former `+ TimeDelta::try_days(..).unwrap()` panic. Return `None` instead: callers
        // render a placeholder (`Display`) or a typed `DataInvalid` error (fallible JSON),
        // mirroring the sibling temporal converters (`microseconds_to_time`, etc.).
        Some(
            chrono::DateTime::UNIX_EPOCH
                .checked_add_signed(TimeDelta::try_days(i64::from(days))?)?
                .naive_utc()
                .date(),
        )
    }

    /// Returns unix epoch.
    pub(crate) fn unix_epoch() -> DateTime<Utc> {
        Utc.timestamp_nanos(0)
    }

    /// Creates date literal from `NaiveDate`, assuming it's utc timezone.
    pub(crate) fn date_from_naive_date(date: NaiveDate) -> i32 {
        (date - unix_epoch().date_naive()).num_days() as i32
    }
}

pub(crate) mod time {
    use super::*;

    pub(crate) fn time_to_microseconds(time: &NaiveTime) -> i64 {
        time.signed_duration_since(
            // This is always the same and shouldn't fail
            NaiveTime::from_num_seconds_from_midnight_opt(0, 0).unwrap(),
        )
        .num_microseconds()
        .unwrap()
    }

    /// Converts a microsecond-of-day `Time` value into a [`NaiveTime`].
    ///
    /// Returns `None` for an out-of-range stored value (negative, or `>= 86_400_000_000`, i.e. at
    /// or past 24h). Such a value can reach us from on-disk bytes (min/max stats, partition values,
    /// manifest entries) on a corrupt or hostile file, so this must not panic — callers render a
    /// placeholder rather than unwrapping. The previous `secs as u32` / `rem as u32` casts wrapped
    /// silently on a negative input, which is corrected here by rejecting it via checked conversion.
    pub(crate) fn microseconds_to_time(micros: i64) -> Option<NaiveTime> {
        let (secs, rem) = (micros / 1_000_000, micros % 1_000_000);

        // A negative `micros` yields a negative `secs`/`rem`; `u32::try_from` rejects it instead of
        // wrapping. `rem` is in `0..1_000_000`, so `* 1_000` (nanoseconds) cannot overflow `u32`.
        let secs = u32::try_from(secs).ok()?;
        let nanos = u32::try_from(rem).ok()? * 1_000;
        NaiveTime::from_num_seconds_from_midnight_opt(secs, nanos)
    }
}

pub(crate) mod timestamp {
    use super::*;

    pub(crate) fn datetime_to_microseconds(time: &NaiveDateTime) -> i64 {
        time.and_utc().timestamp_micros()
    }

    /// Converts a microsecond `Timestamp` value into a [`NaiveDateTime`].
    ///
    /// Returns `None` for a value outside chrono's representable range. This shouldn't happen until
    /// roughly the year 262000, but the value originates from on-disk bytes, so a corrupt/hostile
    /// stored value must produce a placeholder via the caller rather than a panic.
    pub(crate) fn microseconds_to_datetime(micros: i64) -> Option<NaiveDateTime> {
        Some(DateTime::from_timestamp_micros(micros)?.naive_utc())
    }

    pub(crate) fn nanoseconds_to_datetime(nanos: i64) -> NaiveDateTime {
        DateTime::from_timestamp_nanos(nanos).naive_utc()
    }
}

pub(crate) mod timestamptz {
    use super::*;

    pub(crate) fn datetimetz_to_microseconds(time: &DateTime<Utc>) -> i64 {
        time.timestamp_micros()
    }

    /// Converts a microsecond `Timestamptz` value into a UTC [`DateTime`].
    ///
    /// Returns `None` for a value outside chrono's representable range. The previous `rem as u32`
    /// cast wrapped a negative sub-second remainder into a giant nanosecond count (which then
    /// failed the unwrap and panicked); here `from_timestamp` is fed a normalized non-negative
    /// remainder and the `None` is propagated instead of unwrapped.
    pub(crate) fn microseconds_to_datetimetz(micros: i64) -> Option<DateTime<Utc>> {
        // Euclidean division keeps the remainder in `0..1_000_000` even for negative inputs, so the
        // `secs`/`nanos` pair is the correct (floored) split and the `u32` nanos cannot overflow.
        let secs = micros.div_euclid(1_000_000);
        let nanos = micros.rem_euclid(1_000_000) as u32 * 1_000;
        DateTime::from_timestamp(secs, nanos)
    }

    /// Converts a nanosecond `TimestamptzNs` value into a UTC [`DateTime`].
    ///
    /// Returns `None` for a value outside chrono's representable range. As with the microsecond
    /// converter, euclidean division yields a non-negative sub-second remainder so the `u32` nanos
    /// cast is safe and a negative input no longer wraps into a panic.
    pub(crate) fn nanoseconds_to_datetimetz(nanos: i64) -> Option<DateTime<Utc>> {
        let secs = nanos.div_euclid(1_000_000_000);
        let sub_nanos = nanos.rem_euclid(1_000_000_000) as u32;
        DateTime::from_timestamp(secs, sub_nanos)
    }
}

#[cfg(test)]
mod tests {
    use super::date::{date_to_days, days_to_date};

    /// SAF-003: an out-of-range `days`-since-epoch value (corrupt/hostile on-disk bytes) must
    /// yield `None`, not panic. For any `i32` the former `TimeDelta::try_days(days as i64)` was
    /// always `Some`, so the panic was in the subsequent `DateTime + TimeDelta` (`Add`), which
    /// overflowed chrono's representable range near the `i32` extremes.
    ///
    /// MUTATION (restore `pub(crate) fn days_to_date(days: i32) -> NaiveDate` built with
    /// `(chrono::DateTime::UNIX_EPOCH + TimeDelta::try_days(days as i64).unwrap()).naive_utc().date()`):
    /// `days_to_date(i32::MAX)` panics in chrono's `DateTime + TimeDelta`.
    #[test]
    fn test_days_to_date_out_of_range_is_none_not_panic() {
        assert!(days_to_date(i32::MAX).is_none());
        assert!(days_to_date(i32::MIN).is_none());
    }

    #[test]
    fn test_days_to_date_roundtrip_valid() {
        for days in [0, 1, -1, 18_000, -18_000, 100_000, -100_000] {
            let date = days_to_date(days).expect("in-range days-since-epoch must convert");
            assert_eq!(date_to_days(&date), days);
        }
    }
}
