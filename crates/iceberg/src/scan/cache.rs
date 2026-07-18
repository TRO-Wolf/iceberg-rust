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
use std::sync::{Arc, RwLock};

use crate::expr::visitors::expression_evaluator::ExpressionEvaluator;
use crate::expr::visitors::inclusive_projection::InclusiveProjection;
use crate::expr::visitors::manifest_evaluator::ManifestEvaluator;
use crate::expr::{Bind, BoundPredicate};
use crate::spec::{Schema, TableMetadataRef};
use crate::{Error, ErrorKind, Result};

/// Manages the caching of [`BoundPredicate`] objects
/// for [`PartitionSpec`]s based on partition spec id.
#[derive(Debug)]
pub(crate) struct PartitionFilterCache(RwLock<HashMap<i32, Arc<BoundPredicate>>>);

impl PartitionFilterCache {
    /// Creates a new [`PartitionFilterCache`]
    /// with an empty internal HashMap.
    pub(crate) fn new() -> Self {
        Self(RwLock::new(HashMap::new()))
    }

    /// Retrieves a [`BoundPredicate`] from the cache
    /// or computes it if not present.
    pub(crate) fn get(
        &self,
        spec_id: i32,
        table_metadata: &TableMetadataRef,
        schema: &Schema,
        case_sensitive: bool,
        filter: BoundPredicate,
    ) -> Result<Arc<BoundPredicate>> {
        // we need a block here to ensure that the `read()` gets dropped before we hit the `write()`
        // below, otherwise we hit deadlock
        {
            let read = self.0.read().map_err(|_| {
                Error::new(
                    ErrorKind::Unexpected,
                    "PartitionFilterCache RwLock was poisoned",
                )
            })?;

            if read.contains_key(&spec_id) {
                return Ok(read.get(&spec_id).unwrap().clone());
            }
        }

        let partition_spec = table_metadata
            .partition_spec_by_id(spec_id)
            .ok_or(Error::new(
                ErrorKind::Unexpected,
                format!("Could not find partition spec for id {spec_id}"),
            ))?;

        let partition_type = partition_spec.partition_type(schema)?;
        let partition_fields = partition_type.fields().to_owned();
        let partition_schema = Arc::new(
            Schema::builder()
                .with_schema_id(partition_spec.spec_id())
                .with_fields(partition_fields)
                .build()?,
        );

        let mut inclusive_projection = InclusiveProjection::new(partition_spec.clone());

        let partition_filter = inclusive_projection
            .project(&filter)?
            .rewrite_not()
            .bind(partition_schema.clone(), case_sensitive)?;

        self.0
            .write()
            .map_err(|_| {
                Error::new(
                    ErrorKind::Unexpected,
                    "PartitionFilterCache RwLock was poisoned",
                )
            })?
            .insert(spec_id, Arc::new(partition_filter));

        let read = self.0.read().map_err(|_| {
            Error::new(
                ErrorKind::Unexpected,
                "PartitionFilterCache RwLock was poisoned",
            )
        })?;

        Ok(read.get(&spec_id).unwrap().clone())
    }
}

/// Manages the caching of [`ManifestEvaluator`] objects
/// for [`PartitionSpec`]s based on partition spec id.
#[derive(Debug)]
pub(crate) struct ManifestEvaluatorCache(RwLock<HashMap<i32, Arc<ManifestEvaluator>>>);

impl ManifestEvaluatorCache {
    /// Creates a new [`ManifestEvaluatorCache`]
    /// with an empty internal HashMap.
    pub(crate) fn new() -> Self {
        Self(RwLock::new(HashMap::new()))
    }

    /// Retrieves a [`ManifestEvaluator`] from the cache
    /// or computes it if not present.
    pub(crate) fn get(
        &self,
        spec_id: i32,
        partition_filter: Arc<BoundPredicate>,
    ) -> Arc<ManifestEvaluator> {
        // This accessor is infallible by signature (its sole caller consumes the evaluator
        // inline during planning), so a poisoned lock is RECOVERED via `into_inner` rather than
        // mapped-to-error-then-`unwrap`-panicked as before. The guarded `HashMap` critical
        // sections do only `contains_key`/`get`/`insert`/`clone` — no re-entrant user code that
        // could tear the map — so a guard left by a panicked holder still wraps a coherent map;
        // recovering it keeps planning alive instead of cascading the panic (the crate's
        // `into_inner` policy, see `arrow/reader.rs`). `PartitionFilterCache`/
        // `ExpressionEvaluatorCache` map poison to a typed error instead because their signatures
        // return `Result`; this one cannot without rippling a signature change into `scan/context.rs`.
        //
        // we need a block here to ensure that the `read()` gets dropped before we hit the `write()`
        // below, otherwise we hit deadlock
        {
            let read = self
                .0
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            if read.contains_key(&spec_id) {
                return read.get(&spec_id).unwrap().clone();
            }
        }

        self.0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(
                spec_id,
                Arc::new(ManifestEvaluator::builder(partition_filter.as_ref().clone()).build()),
            );

        let read = self
            .0
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        read.get(&spec_id).unwrap().clone()
    }
}

/// Manages the caching of [`ExpressionEvaluator`] objects
/// for [`PartitionSpec`]s based on partition spec id.
#[derive(Debug)]
pub(crate) struct ExpressionEvaluatorCache(RwLock<HashMap<i32, Arc<ExpressionEvaluator>>>);

impl ExpressionEvaluatorCache {
    /// Creates a new [`ExpressionEvaluatorCache`]
    /// with an empty internal HashMap.
    pub(crate) fn new() -> Self {
        Self(RwLock::new(HashMap::new()))
    }

    /// Retrieves a [`ExpressionEvaluator`] from the cache
    /// or computes it if not present.
    pub(crate) fn get(
        &self,
        spec_id: i32,
        partition_filter: &BoundPredicate,
    ) -> Result<Arc<ExpressionEvaluator>> {
        // we need a block here to ensure that the `read()` gets dropped before we hit the `write()`
        // below, otherwise we hit deadlock
        {
            let read = self.0.read().map_err(|_| {
                Error::new(
                    ErrorKind::Unexpected,
                    "ExpressionEvaluatorCache RwLock was poisoned",
                )
            })?;

            if read.contains_key(&spec_id) {
                return Ok(read.get(&spec_id).unwrap().clone());
            }
        }

        self.0
            .write()
            .map_err(|_| {
                Error::new(
                    ErrorKind::Unexpected,
                    "ExpressionEvaluatorCache RwLock was poisoned",
                )
            })?
            .insert(
                spec_id,
                Arc::new(ExpressionEvaluator::new(partition_filter.clone())),
            );

        let read = self.0.read().map_err(|_| {
            Error::new(
                ErrorKind::Unexpected,
                "ExpressionEvaluatorCache RwLock was poisoned",
            )
        })?;

        Ok(read.get(&spec_id).unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::expr::Predicate;

    /// A trivially-bound predicate for exercising the caches; the survival predicate itself is
    /// irrelevant to the lock-recovery behavior under test.
    fn bound_always_true() -> BoundPredicate {
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));
        Predicate::AlwaysTrue
            .bind(schema, true)
            .expect("bind AlwaysTrue")
    }

    /// Poison `lock` by panicking while holding its write guard (the guard poisons the lock as it
    /// drops during the unwind). Caught in-thread so the test proceeds against the poisoned lock.
    fn poison_lock<T>(lock: &RwLock<T>) {
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = lock.write().expect("acquire write guard to poison");
            panic!("intentionally poison the cache lock");
        }));
        assert!(
            poisoned.is_err(),
            "the poisoning closure must have panicked while holding the guard"
        );
    }

    /// Risk pinned (audit SAF-006): `ManifestEvaluatorCache::get` is infallible by signature, so a
    /// poisoned lock must be RECOVERED (`into_inner`), never mapped-to-error-then-`unwrap`-panicked.
    /// MUTATION: restoring `.map_err(|_| Error::new(...)).unwrap()` panics this test.
    #[test]
    fn test_manifest_evaluator_cache_recovers_poisoned_lock() {
        let cache = ManifestEvaluatorCache::new();
        poison_lock(&cache.0);

        // Must return an evaluator without panicking, and cache it under spec 0.
        let evaluator = cache.get(0, Arc::new(bound_always_true()));
        assert!(
            cache
                .0
                .read()
                .unwrap_or_else(|p| p.into_inner())
                .contains_key(&0),
            "the evaluator must be cached after a recovered get"
        );
        // A second get on the recovered lock hits the cache (same Arc).
        let evaluator2 = cache.get(0, Arc::new(bound_always_true()));
        assert!(
            Arc::ptr_eq(&evaluator, &evaluator2),
            "the second get must reuse the cached evaluator"
        );
    }

    /// Risk pinned (audit SAF-006): `ExpressionEvaluatorCache::get` returns `Result`, so a poisoned
    /// lock must surface a TYPED error naming THIS cache — not panic, and not the copy-pasted wrong
    /// lock name. MUTATION: restoring the poison `.unwrap()` panics; leaving the old
    /// "PartitionFilterCache" / "ManifestEvaluatorCache" message fails the name assertion.
    #[test]
    fn test_expression_evaluator_cache_poison_yields_typed_error_named_correctly() {
        let cache = ExpressionEvaluatorCache::new();
        let filter = bound_always_true();
        poison_lock(&cache.0);

        let error = cache
            .get(0, &filter)
            .expect_err("a poisoned lock must surface a typed error, not panic");
        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error.to_string().contains("ExpressionEvaluatorCache"),
            "the error must name the ExpressionEvaluatorCache, not a copy-pasted cache name: {error}"
        );
    }
}
