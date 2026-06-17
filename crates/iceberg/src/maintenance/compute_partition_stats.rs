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

//! `ComputePartitionStats` — the engine-agnostic action wrapper over the partition-statistics
//! compute/write/register core, the Rust port of Java 1.10.0
//! `org.apache.iceberg.actions.ComputePartitionStats` (interface) +
//! `org.apache.iceberg.actions.BaseComputePartitionStats` (the `iceberg-core` impl, NOT Spark).
//!
//! # The Java contract this mirrors
//!
//! `ComputePartitionStats extends Action<ComputePartitionStats, ComputePartitionStats.Result>` with one
//! own method `snapshot(long)` plus the inherited `execute() -> Result`; `Result` exposes only
//! `statisticsFile() -> PartitionStatisticsFile`. The core impl `BaseComputePartitionStats.execute()`
//! calls `PartitionStatsHandler.computeAndWriteStatsFile(table[, snapshotId])` then
//! `table.updatePartitionStatistics().setPartitionStatistics(file).commit()` and returns the registered
//! file. This wrapper is a THIN composition over the already-complete core:
//! [`compute_and_write_stats_file`](crate::maintenance::compute_and_write_stats_file) (the full
//! compute+write) and [`register_partition_stats_file`](crate::maintenance::register_partition_stats_file)
//! (the commit, which routes through the
//! [`UpdatePartitionStatisticsAction`](crate::transaction::Transaction::update_partition_statistics)
//! seam — Java's `updatePartitionStatistics().setPartitionStatistics(file).commit()`).
//!
//! # Empty / unpartitioned result (the `Ok(None)` case)
//!
//! [`compute_and_write_stats_file`] returns `Ok(None)` when there is nothing to write — an empty
//! computed collection (a partitioned table with no rows in the snapshot). Java `Action.execute()`
//! returns a NON-null `Result`, but the partition-stats action path assumes a partitioned table with
//! content (`computeAndWriteStatsFile` itself errors on an UNPARTITIONED table via the
//! [`unified_partition_type`](crate::maintenance::unified_partition_type) precondition). Because there
//! is no `PartitionStatisticsFile` to put in a non-null `Result`, this action surfaces the `None` case
//! as a typed [`DataInvalid`](crate::ErrorKind::DataInvalid) error rather than fabricating an empty
//! result — the table had no partition statistics to compute.

use crate::maintenance::{compute_and_write_stats_file, register_partition_stats_file};
use crate::spec::{PartitionStatisticsFile, Snapshot};
use crate::table::Table;
use crate::{Catalog, Error, ErrorKind, Result};

/// Computes per-partition statistics over a table snapshot, writes them to an on-disk partition-stats
/// file, and registers the resulting [`PartitionStatisticsFile`] into the table metadata — the Rust
/// port of Java's `ComputePartitionStats` action (`BaseComputePartitionStats`).
///
/// Builder semantics mirror Java's `ComputePartitionStats` interface:
///
/// - [`ComputePartitionStats::snapshot_id`] — the snapshot to compute over (Java `snapshot(long)`).
///   **Default: the current snapshot.**
/// - [`ComputePartitionStats::execute`] — resolves the snapshot, runs the compute/write core
///   ([`compute_and_write_stats_file`]), then registers the file through the
///   [`UpdatePartitionStatisticsAction`](crate::transaction::Transaction::update_partition_statistics)
///   seam ([`register_partition_stats_file`]), returning the refreshed [`Table`] and the registered
///   [`PartitionStatisticsFile`].
pub struct ComputePartitionStats {
    table: Table,
    snapshot_id: Option<i64>,
}

impl ComputePartitionStats {
    /// Creates a new action for `table`. With no further configuration, [`Self::execute`] computes
    /// partition statistics over the current snapshot.
    pub fn new(table: Table) -> Self {
        Self {
            table,
            snapshot_id: None,
        }
    }

    /// Computes stats over the given snapshot instead of the current one (Java `snapshot(long)`).
    pub fn snapshot_id(mut self, snapshot_id: i64) -> Self {
        self.snapshot_id = Some(snapshot_id);
        self
    }

    /// Resolves the snapshot to compute over — the configured one, else the current snapshot.
    fn resolve_snapshot(&self) -> Result<Snapshot> {
        let metadata = self.table.metadata();
        let snapshot = match self.snapshot_id {
            Some(snapshot_id) => metadata.snapshot_by_id(snapshot_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Snapshot id {snapshot_id} does not exist in the table"),
                )
            })?,
            None => metadata.current_snapshot().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot compute partition stats: the table has no current snapshot",
                )
            })?,
        };
        Ok(snapshot.as_ref().clone())
    }

    /// Runs the action: resolve the snapshot → compute + write the partition-stats file → register it
    /// through the partition-statistics transaction seam. Returns the refreshed [`Table`] and the
    /// registered [`PartitionStatisticsFile`].
    ///
    /// # Errors
    ///
    /// - `DataInvalid` for an unknown configured snapshot id, a missing current snapshot, or when the
    ///   compute yields no statistics file (an empty computed collection — see the module docs on the
    ///   `Ok(None)` case; Java's `execute()` returns a non-null `Result`, so this action has no file to
    ///   return).
    /// - Propagates `DataInvalid` "Table must be partitioned" for an unpartitioned table and
    ///   `FeatureUnsupported` for a non-parquet `write.format.default` (from
    ///   [`compute_and_write_stats_file`]), plus manifest-read / IO / parquet-encode / catalog-commit
    ///   errors.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<ComputePartitionStatsResult> {
        let snapshot = self.resolve_snapshot()?;

        let statistics_file = compute_and_write_stats_file(&self.table, &snapshot)
            .await?
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "No partition statistics to compute for snapshot {}: the computed collection \
                         is empty",
                        snapshot.snapshot_id()
                    ),
                )
            })?;

        let table =
            register_partition_stats_file(catalog, &self.table, statistics_file.clone()).await?;

        Ok(ComputePartitionStatsResult {
            table,
            statistics_file,
        })
    }
}

/// The result of [`ComputePartitionStats::execute`]: the refreshed table and the registered partition
/// statistics file — the Rust port of Java `ComputePartitionStats.Result` (whose only member is
/// `statisticsFile()`). The refreshed [`Table`] is carried for caller convenience (the Rust register
/// path returns it), beyond the Java `Result` surface.
#[derive(Debug)]
pub struct ComputePartitionStatsResult {
    /// The table refreshed after the partition-statistics file was committed.
    pub table: Table,
    /// The partition-statistics file that was written and registered (Java `Result.statisticsFile()`).
    pub statistics_file: PartitionStatisticsFile,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::io::{FileIO, FileIOBuilder, LocalFsStorageFactory};
    use crate::memory::MemoryCatalogBuilder;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, NestedField,
        PrimitiveType, Schema, Struct, Transform, Type,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

    const DATA_FILE_SIZE: u64 = 100;

    async fn e2e_catalog() -> (impl Catalog, FileIO, TempDir) {
        let temp_dir = TempDir::new().expect("temp dir");
        let warehouse = temp_dir
            .path()
            .to_str()
            .expect("utf8 temp path")
            .to_string();
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                HashMap::from([("warehouse".to_string(), warehouse)]),
            )
            .await
            .expect("load local-fs memory catalog");
        let file_io = FileIOBuilder::new(Arc::new(LocalFsStorageFactory)).build();
        (catalog, file_io, temp_dir)
    }

    fn x_long_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            ))])
            .build()
            .expect("build schema")
    }

    /// A table partitioned by `identity(x)` (spec 0, field id 1000).
    async fn create_x_partitioned_table(catalog: &impl Catalog) -> Table {
        let spec = crate::spec::PartitionSpec::builder(x_long_schema())
            .with_spec_id(0)
            .add_partition_field("x", "x", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(x_long_schema())
            .partition_spec(spec.into_unbound())
            .build();
        catalog.create_table(&namespace, creation).await.unwrap()
    }

    /// An UNPARTITIONED table (empty spec) — `compute_and_write_stats_file` errors on this.
    async fn create_unpartitioned_table(catalog: &impl Catalog) -> Table {
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(x_long_schema())
            .partition_spec(crate::spec::PartitionSpec::unpartition_spec().into_unbound())
            .build();
        catalog.create_table(&namespace, creation).await.unwrap()
    }

    async fn write_file(file_io: &FileIO, path: &str, content: &[u8]) {
        file_io
            .new_output(path)
            .unwrap()
            .write(bytes::Bytes::copy_from_slice(content))
            .await
            .unwrap();
    }

    async fn data_file(file_io: &FileIO, path: &str, partition: Struct, records: u64) -> DataFile {
        write_file(file_io, path, &vec![0u8; DATA_FILE_SIZE as usize]).await;
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(DATA_FILE_SIZE)
            .record_count(records)
            .partition_spec_id(0)
            .partition(partition)
            .build()
            .unwrap()
    }

    fn x_struct(value: i64) -> Struct {
        Struct::from_iter([Some(Literal::long(value))])
    }

    async fn append(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(table);
        let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Smoke test (THROWAWAY local-fs MemoryCatalog): the action computes + writes + registers a
    /// partition-stats file end-to-end over a partitioned table. The registered file is keyed by the
    /// computed snapshot id, and the refreshed metadata carries it (proving the action routed through
    /// the `UpdatePartitionStatisticsAction` seam, not a stub).
    #[tokio::test]
    async fn test_compute_partition_stats_executes_live() {
        let (catalog, file_io, _tmp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d1.parquet"),
                x_struct(1),
                3,
            )
            .await,
            data_file(
                &file_io,
                &format!("{location}/data/x=2/d2.parquet"),
                x_struct(2),
                5,
            )
            .await,
        ])
        .await;
        let snapshot_id = table.metadata().current_snapshot_id().unwrap();

        let result = ComputePartitionStats::new(table)
            .execute(&catalog)
            .await
            .expect("compute partition stats");

        // The returned file is keyed to the computed snapshot.
        assert_eq!(result.statistics_file.snapshot_id, snapshot_id);
        assert!(result.statistics_file.file_size_in_bytes > 0);

        // The refreshed metadata carries the registered file (seam commit landed).
        let registered = result
            .table
            .metadata()
            .partition_statistics_for_snapshot(snapshot_id)
            .expect("registered partition statistics");
        assert_eq!(registered.snapshot_id, snapshot_id);
        assert_eq!(
            registered.statistics_path,
            result.statistics_file.statistics_path
        );

        // The file physically exists on disk.
        assert!(
            file_io
                .exists(&result.statistics_file.statistics_path)
                .await
                .expect("exists check"),
            "the partition-stats file must be written to disk"
        );
    }

    /// The explicit-`snapshot_id` path computes over the requested snapshot, not the current one.
    #[tokio::test]
    async fn test_compute_partition_stats_over_requested_snapshot() {
        let (catalog, file_io, _tmp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/s1.parquet"),
                x_struct(1),
                3,
            )
            .await,
        ])
        .await;
        let snapshot_one = table.metadata().current_snapshot_id().unwrap();

        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=2/s2.parquet"),
                x_struct(2),
                5,
            )
            .await,
        ])
        .await;
        let snapshot_two = table.metadata().current_snapshot_id().unwrap();
        assert_ne!(snapshot_one, snapshot_two);

        let result = ComputePartitionStats::new(table)
            .snapshot_id(snapshot_one)
            .execute(&catalog)
            .await
            .expect("compute over snapshot one");
        assert_eq!(result.statistics_file.snapshot_id, snapshot_one);
    }

    /// An unknown configured snapshot id errors (DataInvalid) before any compute.
    #[tokio::test]
    async fn test_unknown_snapshot_id_errors() {
        let (catalog, file_io, _tmp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/x=1/d.parquet"),
                x_struct(1),
                3,
            )
            .await,
        ])
        .await;

        let error = ComputePartitionStats::new(table)
            .snapshot_id(999_999)
            .execute(&catalog)
            .await
            .expect_err("unknown snapshot must error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(error.to_string().contains("999999"));
    }

    /// An unpartitioned table errors (DataInvalid "Table must be partitioned") — propagated from the
    /// compute core, NOT a fabricated empty result (snag 2).
    #[tokio::test]
    async fn test_unpartitioned_table_errors() {
        let (catalog, file_io, _tmp) = e2e_catalog().await;
        let table = create_unpartitioned_table(&catalog).await;
        let location = table.metadata().location().to_string();
        let table = append(&catalog, &table, vec![
            data_file(
                &file_io,
                &format!("{location}/data/d.parquet"),
                Struct::empty(),
                3,
            )
            .await,
        ])
        .await;

        let error = ComputePartitionStats::new(table)
            .execute(&catalog)
            .await
            .expect_err("unpartitioned table must error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
    }

    /// A table with no current snapshot errors (DataInvalid) — the default-snapshot resolution fails
    /// before any compute.
    #[tokio::test]
    async fn test_no_current_snapshot_errors() {
        let (catalog, _file_io, _tmp) = e2e_catalog().await;
        let table = create_x_partitioned_table(&catalog).await;

        let error = ComputePartitionStats::new(table)
            .execute(&catalog)
            .await
            .expect_err("a table with no snapshot must error");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(error.to_string().contains("no current snapshot"));
    }
}
