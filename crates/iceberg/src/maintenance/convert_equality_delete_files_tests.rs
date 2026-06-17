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

//! Tests for [`ConvertEqualityDeleteFiles`]. Each is a READ-IDENTITY proof: the live row set is asserted
//! IDENTICAL before (eq-deletes) and after (pos-deletes) conversion, plus the Result counts. Each test
//! pins one of the four corruption-stallers and FAILS if the corresponding logic is mutated.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use futures::TryStreamExt;
use tempfile::TempDir;

use super::*;
use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType,
    NestedField, PartitionSpec, PrimitiveType, Schema as IcebergSchema, Struct, Transform, Type,
};
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::file_writer::{FileWriter, FileWriterBuilder};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

// =================================================================================================
// Helpers (table build / writers / scan) — adapted from `remove_dangling_delete_files::tests`.
// =================================================================================================

async fn local_fs_catalog() -> (impl Catalog, TempDir) {
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
            std::collections::HashMap::from([("warehouse".to_string(), warehouse)]),
        )
        .await
        .expect("load local-fs memory catalog");
    (catalog, temp_dir)
}

fn three_long_schema() -> IcebergSchema {
    IcebergSchema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "x",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "y",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                3,
                "z",
                Type::Primitive(PrimitiveType::Long),
            )),
        ])
        .build()
        .expect("build schema")
}

async fn create_partitioned_table(catalog: &impl Catalog, format_version: FormatVersion) -> Table {
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .expect("add partition field")
        .build()
        .expect("build spec");
    create_table_with_spec(catalog, schema, spec, format_version).await
}

async fn create_unpartitioned_table(
    catalog: &impl Catalog,
    format_version: FormatVersion,
) -> Table {
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .build()
        .expect("build spec");
    create_table_with_spec(catalog, schema, spec, format_version).await
}

async fn create_table_with_spec(
    catalog: &impl Catalog,
    schema: IcebergSchema,
    spec: PartitionSpec,
    format_version: FormatVersion,
) -> Table {
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, std::collections::HashMap::new())
        .await
        .expect("create namespace");
    let table_ident = TableIdent::new(namespace.clone(), "t".to_string());
    let creation = TableCreation::builder()
        .name(table_ident.name().to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(format_version)
        .build();
    catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table")
}

/// Write a DATA file with one row group holding all rows.
async fn write_data_file(
    table: &Table,
    file_name: &str,
    part_value: i64,
    rows: &[(i64, i64, i64)],
) -> DataFile {
    write_data_file_rg(table, file_name, part_value, rows, None).await
}

/// Write a DATA file, optionally forcing `max_row_group_size` so a file spans MULTIPLE row groups
/// (the absolute-`_pos` staller).
async fn write_data_file_rg(
    table: &Table,
    file_name: &str,
    part_value: i64,
    rows: &[(i64, i64, i64)],
    max_row_group_size: Option<usize>,
) -> DataFile {
    use crate::arrow::schema_to_arrow_schema;

    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());

    let xs: Vec<i64> = rows.iter().map(|(x, _, _)| *x).collect();
    let ys: Vec<i64> = rows.iter().map(|(_, y, _)| *y).collect();
    let zs: Vec<i64> = rows.iter().map(|(_, _, z)| *z).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .unwrap();

    let file_path = format!("{}/data/{}", table.metadata().location(), file_name);
    let output = table.file_io().new_output(file_path).unwrap();
    let mut props = parquet::file::properties::WriterProperties::builder();
    if let Some(rg) = max_row_group_size {
        props = props.set_max_row_group_size(rg);
    }
    let parquet_builder = ParquetWriterBuilder::new(props.build(), schema.clone());
    let mut writer = parquet_builder.build(output).await.unwrap();
    writer.write(&batch).await.unwrap();
    let data_file_builders = writer.close().await.unwrap();

    let mut builder = data_file_builders.into_iter().next().unwrap();
    // Use the spec's partition shape: an empty struct for an unpartitioned spec, else the identity
    // partition value.
    let partition = if table.metadata().default_partition_spec().is_unpartitioned() {
        Struct::empty()
    } else {
        Struct::from_iter([Some(Literal::long(part_value))])
    };
    builder
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(partition)
        .build()
        .unwrap()
}

/// Write an equality-delete file in partition `part_value`, deleting rows with the given `y` values.
async fn write_equality_delete_file(table: &Table, part_value: i64, delete_ys: &[i64]) -> DataFile {
    write_eq_delete_inner(table, Some(part_value), delete_ys).await
}

/// Write an UNPARTITIONED (global) equality-delete file deleting the given `y` values.
async fn write_unpartitioned_equality_delete_file(table: &Table, delete_ys: &[i64]) -> DataFile {
    write_eq_delete_inner(table, None, delete_ys).await
}

async fn write_eq_delete_inner(
    table: &Table,
    part_value: Option<i64>,
    delete_ys: &[i64],
) -> DataFile {
    use crate::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};

    let schema = table.metadata().current_schema().clone();
    let config = EqualityDeleteWriterConfig::new(vec![2], schema.clone()).unwrap();
    let delete_schema =
        Arc::new(arrow_schema_to_schema(config.projected_arrow_schema_ref()).unwrap());

    let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
    let file_name_gen = DefaultFileNameGenerator::new(
        "eq-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        delete_schema,
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let partition_key = part_value.map(|pv| {
        crate::spec::PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            schema.clone(),
            Struct::from_iter([Some(Literal::long(pv))]),
        )
    });
    let mut writer = EqualityDeleteFileWriterBuilder::new(rolling, config)
        .build(partition_key)
        .await
        .unwrap();

    let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
    // x is the partition value where partitioned, else 0 (the eq-delete only constrains y=field 2).
    let xs: Vec<i64> = delete_ys.iter().map(|_| part_value.unwrap_or(0)).collect();
    let ys: Vec<i64> = delete_ys.to_vec();
    let zs: Vec<i64> = delete_ys.iter().map(|_| 0).collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(xs)) as ArrayRef,
        Arc::new(Int64Array::from(ys)) as ArrayRef,
        Arc::new(Int64Array::from(zs)) as ArrayRef,
    ])
    .unwrap();
    writer.write(batch).await.unwrap();
    writer.close().await.unwrap().into_iter().next().unwrap()
}

async fn append_files(catalog: &impl Catalog, table: &Table, files: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.row_delta().add_deletes(deletes);
    let tx = action.apply(tx).unwrap();
    tx.commit(catalog).await.unwrap()
}

/// Scan the table and collect the `y` column values (merge-on-read deletes applied) — the read signal.
async fn scan_y_values(table: &Table) -> HashSet<i64> {
    let stream = table
        .scan()
        .select(["y"])
        .build()
        .unwrap()
        .to_arrow()
        .await
        .unwrap();
    let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
    let mut values = HashSet::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for index in 0..col.len() {
            values.insert(col.value(index));
        }
    }
    values
}

/// The set of (content_type, format) of every live DELETE file in the current snapshot.
async fn live_delete_files(table: &Table) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut files = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                files.push(entry.data_file().clone());
            }
        }
    }
    files
}

/// The (data_file, sequence_number) of every live DELETE entry — for the seq-stamp staller.
async fn live_delete_entries_with_seq(table: &Table) -> Vec<(DataFile, Option<i64>)> {
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .unwrap();
    let mut out = Vec::new();
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await.unwrap();
        for entry in manifest.entries() {
            if entry.is_alive() {
                out.push((entry.data_file().clone(), entry.sequence_number()));
            }
        }
    }
    out
}

fn count_eq(files: &[DataFile]) -> usize {
    files
        .iter()
        .filter(|f| f.content_type() == DataContentType::EqualityDeletes)
        .count()
}

fn count_pos(files: &[DataFile]) -> usize {
    files
        .iter()
        .filter(|f| f.content_type() == DataContentType::PositionDeletes)
        .count()
}

// =================================================================================================
// CROWN JEWEL — read-identity over a known masked subset.
// =================================================================================================

/// THE CROWN JEWEL (read-identity). A single equality delete masks a known subset (y=20). Convert it;
/// the post-conversion MoR scan must return the SAME live rows ({10,30}), the eq-delete file must be
/// GONE, exactly one pos-delete file added, and the Result counts must be (1, 1).
///
/// MUTATION COVERAGE: staller (4) inversion — if the pipeline collected SURVIVING rows instead of
/// MATCHING ones, the pos-delete would delete {10,30} and the scan would return {20}, failing the
/// read-identity assertion. Staller (1)/(2) are pinned by dedicated tests below.
#[tokio::test]
async fn test_crown_jewel_read_identity_known_masked_subset() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[
        (0, 10, 100),
        (0, 20, 200),
        (0, 30, 300),
    ])
    .await;
    let table = append_files(&catalog, &table, vec![x]).await;

    let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
    let eq_path = eq_delete.file_path().to_string();
    let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30]),
        "before: eq-delete masks y=20"
    );

    let result = ConvertEqualityDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result, ConvertEqualityDeleteFilesResult {
        converted_equality_delete_files_count: 1,
        added_position_delete_files_count: 1,
    });

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();

    // READ IDENTITY: the live row set is unchanged.
    let after = scan_y_values(&reloaded).await;
    assert_eq!(
        after, before,
        "read identity: live rows IDENTICAL before vs after"
    );

    // The eq-delete is gone; exactly one pos-delete is live.
    let deletes = live_delete_files(&reloaded).await;
    assert_eq!(
        count_eq(&deletes),
        0,
        "the equality delete must be converted away"
    );
    assert_eq!(count_pos(&deletes), 1, "exactly one position delete added");
    assert!(
        !deletes.iter().any(|f| f.file_path() == eq_path),
        "the original eq-delete path must no longer be live"
    );
}

// =================================================================================================
// STALLER (2) — data-seq stamping. The new pos-delete must carry the eq-delete's data seq.
// =================================================================================================

/// The new position delete must be stamped with the SOURCE equality delete's DATA sequence number, NOT
/// the (higher) inherited rewrite-snapshot seq. Data X is at seq 1; the eq-delete (deleting y=20) is at
/// seq 2. After conversion the pos-delete must carry seq 2 (so it still applies to X at seq 1). If it
/// inherited the rewrite snapshot's higher seq it would STILL apply here (3 > 1), so this test ALSO
/// proves the seq is exactly 2 (the eq-delete's), not the rewrite seq — the precise stamp.
///
/// MUTATION COVERAGE: staller (2) — change `add_delete_file_with_sequence_number(.., seq)` to
/// `add_delete_file(..)` (inherit) and the live pos-delete seq becomes the rewrite snapshot's seq (3),
/// not 2; this assertion fails.
#[tokio::test]
async fn test_converted_pos_delete_carries_eq_delete_data_seq() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
    let table = append_files(&catalog, &table, vec![x]).await; // X: data seq 1

    let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
    let table = add_deletes(&catalog, &table, vec![eq_delete]).await; // eq-delete: data seq 2

    // Record the eq-delete's data seq.
    let eq_seq = live_delete_entries_with_seq(&table)
        .await
        .into_iter()
        .find(|(f, _)| f.content_type() == DataContentType::EqualityDeletes)
        .and_then(|(_, seq)| seq)
        .expect("eq-delete has a seq");
    assert_eq!(eq_seq, 2, "fixture: the eq-delete is at data seq 2");

    ConvertEqualityDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let pos_entries: Vec<(DataFile, Option<i64>)> = live_delete_entries_with_seq(&reloaded)
        .await
        .into_iter()
        .filter(|(f, _)| f.content_type() == DataContentType::PositionDeletes)
        .collect();
    assert_eq!(
        pos_entries.len(),
        1,
        "exactly one pos-delete after conversion"
    );
    assert_eq!(
        pos_entries[0].1,
        Some(eq_seq),
        "the converted pos-delete MUST carry the eq-delete's data seq (2), not the inherited rewrite seq"
    );

    // And the read identity still holds (the stamped delete still masks y=20).
    assert_eq!(scan_y_values(&reloaded).await, HashSet::from([10]));
}

// =================================================================================================
// STALLER (3) — applicability scope (partitioned; per-partition; lower-seq only).
// =================================================================================================

/// PARTITION ISOLATION. An equality delete in partition x=0 (deleting y=20) must convert ONLY positions
/// in partition-0 data files; a partition-1 data file with the SAME y=20 value must be untouched. The
/// read identity must hold per-partition: {10,30} in part 0 and {20,40} in part 1 both survive exactly.
///
/// MUTATION COVERAGE: staller (3) — drop the `partition == partition` clause in `is_applicable` and the
/// eq-delete would mask y=20 in partition 1 too, changing the after-set; this assertion fails.
#[tokio::test]
async fn test_partition_isolation_applies_only_to_own_partition() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let p0 = write_data_file(&table, "p0.parquet", 0, &[
        (0, 10, 1),
        (0, 20, 2),
        (0, 30, 3),
    ])
    .await;
    let p1 = write_data_file(&table, "p1.parquet", 1, &[(1, 20, 4), (1, 40, 5)]).await;
    let table = append_files(&catalog, &table, vec![p0, p1]).await;

    // Eq-delete in partition 0 deletes y=20 (applies only to partition 0).
    let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
    let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30, 20, 40]),
        "before: only partition-0 y=20 is dropped; partition-1 y=20 survives"
    );

    let result = ConvertEqualityDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.converted_equality_delete_files_count, 1);
    assert_eq!(result.added_position_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity per-partition: partition-1 y=20 must NOT be masked by a partition-0 eq-delete"
    );
}

/// APPLICABILITY BOUNDARY (equal-seq must NOT be masked). An equality delete applies STRICTLY to
/// lower-seq data (`data_seq < eq_seq`). Append data X (seq 1); in the SAME snapshot as the eq-delete
/// append data W via row_delta so W and the eq-delete share data seq 2. The eq-delete (deleting y=20)
/// must mask X's y=20 (seq 1 < 2) but NOT W's y=20 (seq 2 == 2). Read identity: W's y=20 survives.
///
/// MUTATION COVERAGE: staller (3) — change the strict `data_seq >= eq.sequence_number` cutoff to
/// `data_seq > eq.sequence_number` (i.e. allow equal-seq) and W's y=20 would be position-deleted,
/// changing the after-set; this assertion fails.
#[tokio::test]
async fn test_applicability_boundary_equal_seq_not_masked() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let table = append_files(&catalog, &table, vec![x]).await; // X: seq 1

    // W (data) AND the eq-delete committed in ONE row_delta → both at data seq 2.
    let w = write_data_file(&table, "w.parquet", 0, &[(0, 20, 9), (0, 50, 10)]).await;
    let eq_delete = write_equality_delete_file(&table, 0, &[20]).await;
    let tx = Transaction::new(&table);
    let action = tx
        .row_delta()
        .add_data_files(vec![w])
        .add_deletes(vec![eq_delete]);
    let tx = action.apply(tx).unwrap();
    let table = tx.commit(&catalog).await.unwrap();

    let before = scan_y_values(&table).await;
    // X.y=20 dropped (seq 1 < 2); W.y=20 survives (seq 2 == 2, not strictly lower).
    assert_eq!(
        before,
        HashSet::from([10, 20, 50]),
        "before: X.y=20 dropped, W.y=20 (equal-seq) survives"
    );

    ConvertEqualityDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the equal-seq data file W must NOT be position-deleted (only X's y=20 stays masked)"
    );
}

// =================================================================================================
// STALLER (1) — absolute _pos across MULTIPLE row groups.
// =================================================================================================

/// ABSOLUTE _pos ACROSS ROW GROUPS. A single data file with 7 rows written at `max_row_group_size = 2`
/// (so 4 row groups: rows 0-1, 2-3, 4-5, 6) carries the deleted value y=60 at ABSOLUTE position 5 (the
/// 6th row, in the 3rd row group). If positions were batch/row-group-relative, the pos-delete would
/// carry pos 1 (relative within the 3rd row group) and either delete the WRONG row or nothing, breaking
/// read identity. The conversion must produce a pos-delete with the file-absolute position so the scan
/// returns the same live rows.
///
/// MUTATION COVERAGE: staller (1) — reset `absolute_pos` to 0 each batch (batch-relative) and the
/// computed position for y=60 becomes 1 instead of 5; the post-conversion scan then drops the wrong row
/// and the read-identity assertion fails.
#[tokio::test]
async fn test_absolute_pos_across_multiple_row_groups() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    // 7 rows, y = 0,10,20,30,40,50,60; delete y=60 at absolute pos 6 and y=20 at absolute pos 2.
    let rows: Vec<(i64, i64, i64)> = (0..7).map(|i| (0i64, i * 10, i)).collect();
    let x = write_data_file_rg(&table, "multi-rg.parquet", 0, &rows, Some(2)).await;
    assert_eq!(x.record_count, 7);
    let table = append_files(&catalog, &table, vec![x]).await;

    let eq_delete = write_equality_delete_file(&table, 0, &[20, 60]).await;
    let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([0, 10, 30, 40, 50]),
        "before: y=20 (pos 2) and y=60 (pos 6) are dropped"
    );

    ConvertEqualityDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: absolute positions across 4 row groups keep exactly y=20 and y=60 masked"
    );
    // The new pos-delete must encode the FILE-ABSOLUTE positions 2 and 6.
    let positions = read_position_delete_positions(&reloaded).await;
    assert_eq!(
        positions,
        vec![2i64, 6],
        "the pos-delete must carry FILE-ABSOLUTE positions (2, 6), not row-group-relative ones"
    );
}

/// Read every live position-delete file in the table and return the sorted `pos` values it carries.
async fn read_position_delete_positions(table: &Table) -> Vec<i64> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let mut positions = Vec::new();
    for file in live_delete_files(table).await {
        if file.content_type() != DataContentType::PositionDeletes {
            continue;
        }
        let input = table.file_io().new_input(file.file_path()).unwrap();
        let bytes = input.read().await.unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .unwrap()
            .build()
            .unwrap();
        for batch in reader {
            let batch = batch.unwrap();
            let pos = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..pos.len() {
                positions.push(pos.value(i));
            }
        }
    }
    positions.sort();
    positions
}

// =================================================================================================
// UNPARTITIONED (global) equality delete.
// =================================================================================================

/// An UNPARTITIONED equality delete is a GLOBAL delete — it applies to every lower-seq data file. On an
/// unpartitioned table, deleting y=20 masks y=20 wherever it appears. Read identity must hold after
/// conversion.
#[tokio::test]
async fn test_unpartitioned_global_equality_delete_read_identity() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[(1, 10, 1), (2, 20, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 0, &[(3, 20, 3), (4, 30, 4)]).await;
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let eq_delete = write_unpartitioned_equality_delete_file(&table, &[20]).await;
    let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30]),
        "before: global eq-delete drops every y=20"
    );

    let result = ConvertEqualityDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.converted_equality_delete_files_count, 1);
    assert_eq!(result.added_position_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the global eq-delete's masks (both y=20 rows) stay masked as positions"
    );
    let deletes = live_delete_files(&reloaded).await;
    assert_eq!(count_eq(&deletes), 0);
    assert_eq!(count_pos(&deletes), 1);
}

// =================================================================================================
// FILTER + no-op edges.
// =================================================================================================

/// The `filter(Expression)` restricts conversion to matching partitions. With two eq-deletes in
/// partitions 0 and 1, `filter(x == 0)` converts ONLY the partition-0 eq-delete; the partition-1
/// eq-delete is left as an equality delete. Read identity holds throughout.
#[tokio::test]
async fn test_filter_restricts_converted_partitions() {
    use crate::expr::Reference;

    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let p0 = write_data_file(&table, "p0.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let p1 = write_data_file(&table, "p1.parquet", 1, &[(1, 30, 3), (1, 40, 4)]).await;
    let table = append_files(&catalog, &table, vec![p0, p1]).await;

    let eq0 = write_equality_delete_file(&table, 0, &[20]).await;
    let eq1 = write_equality_delete_file(&table, 1, &[40]).await;
    let table = add_deletes(&catalog, &table, vec![eq0, eq1]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([10, 30]));

    let result = ConvertEqualityDeleteFiles::new(table.clone())
        .filter(Reference::new("x").equal_to(Datum::long(0)))
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.converted_equality_delete_files_count, 1,
        "only the partition-0 eq-delete is converted"
    );
    assert_eq!(result.added_position_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity preserved under filter"
    );
    let deletes = live_delete_files(&reloaded).await;
    assert_eq!(
        count_eq(&deletes),
        1,
        "the partition-1 eq-delete remains an equality delete"
    );
    assert_eq!(
        count_pos(&deletes),
        1,
        "the partition-0 eq-delete became a position delete"
    );
}

/// No current snapshot → no-op, zero counts, no commit.
#[tokio::test]
async fn test_no_current_snapshot_is_a_no_op() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let result = ConvertEqualityDeleteFiles::new(table)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result, ConvertEqualityDeleteFilesResult::default());
}

/// No equality-delete files → no-op, zero counts, no commit.
#[tokio::test]
async fn test_no_equality_deletes_is_a_no_op() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 1)]).await;
    let table = append_files(&catalog, &table, vec![x]).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = ConvertEqualityDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result, ConvertEqualityDeleteFilesResult::default());

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a no-op must NOT commit a new snapshot"
    );
}
