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

//! Per-partition-spec evaluator caches shared across one scan.
//!
//! Lock policy: a poisoned `RwLock` is RECOVERED (`PoisonError::into_inner`, the crate-wide
//! pattern — see e.g. `metrics::mod`, `events::mod`, `arrow::reader`) rather than unwrapped or
//! propagated. The cached values are pure derived data — a deterministic function of
//! (partition spec, schema, filter) — and each insert completes atomically under the write
//! guard, so a panic in another task cannot leave a logically torn map. Recovering keeps the
//! shared cache usable instead of failing (or aborting) every subsequent scan that shares it.

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
        // Scope the read guard so it is dropped before the `write()` below (deadlock avoidance).
        {
            let read = self
                .0
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(cached) = read.get(&spec_id) {
                return Ok(cached.clone());
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

        Ok(self
            .0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(spec_id)
            .or_insert_with(|| Arc::new(partition_filter))
            .clone())
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
        // Scope the read guard so it is dropped before the `write()` below (deadlock avoidance).
        {
            let read = self
                .0
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(cached) = read.get(&spec_id) {
                return cached.clone();
            }
        }

        self.0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(spec_id)
            .or_insert_with(|| {
                Arc::new(ManifestEvaluator::builder(partition_filter.as_ref().clone()).build())
            })
            .clone()
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
    ) -> Arc<ExpressionEvaluator> {
        // Scope the read guard so it is dropped before the `write()` below (deadlock avoidance).
        {
            let read = self
                .0
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(cached) = read.get(&spec_id) {
                return cached.clone();
            }
        }

        self.0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(spec_id)
            .or_insert_with(|| Arc::new(ExpressionEvaluator::new(partition_filter.clone())))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Reference;
    use crate::spec::{
        Datum, FormatVersion, NestedField, PartitionSpec, PrimitiveType, SortOrder,
        TableMetadataBuilder, Transform, Type,
    };

    /// Panics while holding `lock`'s write guard so the lock is left POISONED — the setup
    /// for every recovery pin below. Asserts the poisoning actually took.
    fn poison<T>(lock: &RwLock<T>) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = lock
                .write()
                .expect("test setup: the lock must not already be poisoned");
            panic!("deliberately poison the scan-cache lock");
        }));
        assert!(result.is_err(), "the poisoning closure must panic");
        assert!(lock.is_poisoned(), "test setup: the lock must be poisoned");
    }

    fn table_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![Arc::new(NestedField::optional(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("test schema must build"),
        )
    }

    /// A predicate bound against the TABLE schema (the input `PartitionFilterCache` projects).
    fn bound_table_filter(schema: &Arc<Schema>) -> BoundPredicate {
        Reference::new("id")
            .less_than(Datum::int(10))
            .bind(schema.clone(), true)
            .expect("binding id < 10 must succeed")
    }

    /// Table metadata with one identity-partitioned spec (id 0) over `table_schema`.
    fn table_metadata_with_identity_spec(schema: &Arc<Schema>) -> TableMetadataRef {
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_partition_field("id", "id_part", Transform::Identity)
            .expect("adding the identity partition field must succeed")
            .build()
            .expect("partition spec must build");
        let sort_order = SortOrder::builder()
            .build(schema)
            .expect("sort order must build");
        let metadata = TableMetadataBuilder::new(
            schema.as_ref().clone(),
            spec,
            sort_order,
            "memory://test/table".to_string(),
            FormatVersion::V2,
            HashMap::new(),
        )
        .expect("metadata builder must construct")
        .build()
        .expect("table metadata must build");
        Arc::new(metadata.metadata)
    }

    /// SAF-003 pin (P1c): `PartitionFilterCache` must RECOVER from a poisoned lock — hit,
    /// miss-compute, and the not-found error path must all work after a poisoning panic,
    /// never panic and never fail with a lock error.
    #[test]
    fn test_partition_filter_cache_recovers_from_poisoned_lock() {
        let schema = table_schema();
        let metadata = table_metadata_with_identity_spec(&schema);

        // Hit path on a previously poisoned lock: serve the seeded entry.
        let cache = PartitionFilterCache::new();
        let seeded = cache
            .get(0, &metadata, &schema, true, bound_table_filter(&schema))
            .expect("seeding the cache must succeed");
        poison(&cache.0);
        let hit = cache
            .get(0, &metadata, &schema, true, bound_table_filter(&schema))
            .expect("a poisoned lock must be recovered, not surfaced");
        assert!(
            Arc::ptr_eq(&seeded, &hit),
            "recovery must serve the entry cached before the panic"
        );

        // Unknown spec id after poisoning: the ordinary lookup error, not a lock error.
        let err = cache
            .get(42, &metadata, &schema, true, bound_table_filter(&schema))
            .expect_err("an unknown spec id must still error");
        assert!(
            err.to_string().contains("partition spec"),
            "expected the spec-lookup error, got: {err}"
        );

        // Miss path (compute + insert) on a poisoned lock that was never seeded.
        let cold_cache = PartitionFilterCache::new();
        poison(&cold_cache.0);
        cold_cache
            .get(0, &metadata, &schema, true, bound_table_filter(&schema))
            .expect("compute + insert must succeed on a recovered lock");
    }

    /// SAF-003 pin (P1a): `ManifestEvaluatorCache` must RECOVER from a poisoned lock on both
    /// the hit and the miss path (previously `map_err(..).unwrap()` — a guaranteed panic).
    #[test]
    fn test_manifest_evaluator_cache_recovers_from_poisoned_lock() {
        let schema = table_schema();
        let filter = Arc::new(bound_table_filter(&schema));

        let cache = ManifestEvaluatorCache::new();
        let seeded = cache.get(1, filter.clone());
        poison(&cache.0);

        let hit = cache.get(1, filter.clone());
        assert!(
            Arc::ptr_eq(&seeded, &hit),
            "recovery must serve the entry cached before the panic"
        );

        let miss = cache.get(2, filter);
        assert!(
            !Arc::ptr_eq(&seeded, &miss),
            "a different spec id must compute (and cache) its own evaluator"
        );
    }

    /// SAF-003 pin (P1b): `ExpressionEvaluatorCache` must RECOVER from a poisoned lock on both
    /// the hit and the miss path (previously `map_err(..).unwrap()` — a guaranteed panic).
    #[test]
    fn test_expression_evaluator_cache_recovers_from_poisoned_lock() {
        let schema = table_schema();
        let filter = bound_table_filter(&schema);

        let cache = ExpressionEvaluatorCache::new();
        let seeded = cache.get(1, &filter);
        poison(&cache.0);

        let hit = cache.get(1, &filter);
        assert!(
            Arc::ptr_eq(&seeded, &hit),
            "recovery must serve the entry cached before the panic"
        );

        let miss = cache.get(2, &filter);
        assert!(
            !Arc::ptr_eq(&seeded, &miss),
            "a different spec id must compute (and cache) its own evaluator"
        );
    }
}
