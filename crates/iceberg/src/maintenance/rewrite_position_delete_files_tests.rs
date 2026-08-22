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

//! Tests for [`RewritePositionDeleteFiles`]. Each is a corruption-class READ-IDENTITY proof: the
//! merge-on-read live row set is asserted IDENTICAL before (many parquet pos-deletes) and after (fewer,
//! compacted pos-deletes), plus the four `Result` counts. The crown jewel + the seq-stamp test pin the
//! silent-corruption staller (the compacted file must carry the group MAX rewritten data seq); the
//! grouping + partition-isolation tests pin the `(spec, partition)` planning; the DV test pins the
//! V2-parquet-only scope (a Puffin deletion vector is NOT compacted).

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use tempfile::TempDir;

use super::*;
use crate::io::LocalFsStorageFactory;
use crate::memory::MemoryCatalogBuilder;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, Literal, ManifestContentType,
    NestedField, PartitionKey, PartitionSpec, PrimitiveType, Schema as IcebergSchema, Struct,
    Transform, Type,
};
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::writer::base_writer::position_delete_writer::{
    PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
};
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::file_writer::{FileWriter, FileWriterBuilder};
use crate::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};

// =================================================================================================
// Helpers (table build / data + position-delete writers / scan) — same shape as the convert tests.
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

/// Write a DATA file in partition `part_value` holding `rows` (the file path is returned for use as the
/// position-delete `file_path` target).
async fn write_data_file(
    table: &Table,
    file_name: &str,
    part_value: i64,
    rows: &[(i64, i64, i64)],
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
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = parquet_builder.build(output).await.unwrap();
    writer.write(&batch).await.unwrap();
    let data_file_builders = writer.close().await.unwrap();

    let mut builder = data_file_builders.into_iter().next().unwrap();
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

/// Write a real PARQUET position-delete file masking the given `(target_path, pos)` pairs in partition
/// `part_value`. Returns the resulting position-delete [`DataFile`] (so it can be committed via
/// `add_deletes`). This is the multi-pos-delete fixture the action compacts.
async fn write_position_delete_file(
    table: &Table,
    part_value: Option<i64>,
    pairs: &[(&str, i64)],
) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let config = PositionDeleteWriterConfig::new().unwrap();

    let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        config.schema().clone(),
    );
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );

    let partition_key = part_value.map(|pv| {
        PartitionKey::new(
            table.metadata().default_partition_spec().as_ref().clone(),
            schema.clone(),
            Struct::from_iter([Some(Literal::long(pv))]),
        )
        .expect("PartitionKey::new: valid partition tuple")
    });
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(partition_key)
        .await
        .unwrap();

    let paths: Vec<&str> = pairs.iter().map(|(p, _)| *p).collect();
    let positions: Vec<i64> = pairs.iter().map(|(_, p)| *p).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
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

/// Every live DELETE file in the current snapshot.
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

/// The `(data_file, sequence_number)` of every live DELETE entry — for the seq-stamp staller.
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

fn count_pos(files: &[DataFile]) -> usize {
    files
        .iter()
        .filter(|f| f.content_type() == DataContentType::PositionDeletes)
        .count()
}

// =================================================================================================
// CROWN JEWEL — read-identity over a data file masked by 2+ parquet position-delete files.
// =================================================================================================

/// THE CROWN JEWEL (read-identity). A single data file is masked by TWO separate parquet position-delete
/// files (one masking pos 1 = y=20, one masking pos 3 = y=40). Compact them; the post-compaction MoR
/// scan must return the SAME live rows ({10,30,50}), the two old pos-delete files must be GONE, exactly
/// ONE compacted pos-delete added, and the Result counts must be (2 rewritten, 1 added).
///
/// MUTATION COVERAGE: grouping — if compaction collected positions from only one of the two files (e.g.
/// a `break` after the first), one masked row would resurrect and the after-set would differ from
/// before, failing the read-identity assertion.
#[tokio::test]
async fn test_crown_jewel_read_identity_data_file_masked_by_two_pos_deletes() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    // Data file: y = 10,20,30,40,50 at positions 0..5.
    let x = write_data_file(&table, "x.parquet", 0, &[
        (0, 10, 100),
        (0, 20, 200),
        (0, 30, 300),
        (0, 40, 400),
        (0, 50, 500),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await; // X: data seq 1

    // Two SEPARATE parquet pos-delete files, each masking a distinct position of X.
    let pd1 = write_position_delete_file(&table, Some(0), &[(&x_path, 1)]).await; // y=20
    let table = add_deletes(&catalog, &table, vec![pd1]).await; // seq 2
    let pd2 = write_position_delete_file(&table, Some(0), &[(&x_path, 3)]).await; // y=40
    let table = add_deletes(&catalog, &table, vec![pd2]).await; // seq 3

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30, 50]),
        "before: two pos-deletes mask y=20 (pos 1) and y=40 (pos 3)"
    );
    assert_eq!(
        count_pos(&live_delete_files(&table).await),
        2,
        "before: two live position-delete files"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "two files rewritten"
    );
    assert_eq!(
        result.added_delete_files_count, 1,
        "one compacted file added"
    );
    assert!(
        result.rewritten_bytes_count > 0,
        "rewritten bytes must be non-zero"
    );
    assert!(result.added_bytes_count > 0, "added bytes must be non-zero");

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();

    // READ IDENTITY: the live row set is unchanged.
    let after = scan_y_values(&reloaded).await;
    assert_eq!(
        after, before,
        "read identity: live rows IDENTICAL before vs after compaction"
    );

    // Exactly one compacted pos-delete is live (fewer files).
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        1,
        "the two pos-deletes are compacted into exactly one"
    );
}

// =================================================================================================
// STALLER (seq stamping) — the compacted file must carry the group MAX rewritten data seq.
// =================================================================================================

/// SEQ STAMPING (the silent-corruption staller). Data X is at seq 1; two pos-deletes mask it at seqs 2
/// and 3. The compacted file MUST carry the group MAX rewritten data seq (3) — NOT the inherited
/// (higher) rewrite-snapshot seq, NOT the min (2). If it carried the inherited seq it would still apply
/// here (4 > 1) so the read would look fine — this test therefore asserts the EXACT stamped seq is 3,
/// pinning the precise stamp.
///
/// MUTATION COVERAGE: change `add_delete_file_with_sequence_number(.., max_seq)` to
/// `add_delete_file(..)` (inherit) and the live compacted pos-delete seq becomes the rewrite snapshot's
/// seq (4), not 3; the seq assertion fails. Change `.max()` to `.min()` and the stamp becomes 2; fails.
#[tokio::test]
async fn test_compacted_file_carries_group_max_rewritten_seq() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[
        (0, 10, 1),
        (0, 20, 2),
        (0, 30, 3),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await; // X: data seq 1

    let pd1 = write_position_delete_file(&table, Some(0), &[(&x_path, 1)]).await; // y=20
    let table = add_deletes(&catalog, &table, vec![pd1]).await; // seq 2
    let pd2 = write_position_delete_file(&table, Some(0), &[(&x_path, 0)]).await; // y=10
    let table = add_deletes(&catalog, &table, vec![pd2]).await; // seq 3

    // Confirm the fixture seqs: the two pos-deletes are at seq 2 and 3 (max = 3).
    let seqs: Vec<i64> = live_delete_entries_with_seq(&table)
        .await
        .into_iter()
        .filter(|(f, _)| f.content_type() == DataContentType::PositionDeletes)
        .filter_map(|(_, seq)| seq)
        .collect();
    assert_eq!(
        seqs.iter().copied().max(),
        Some(3),
        "fixture: the group MAX rewritten pos-delete seq is 3"
    );

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([30]), "before: y=10 and y=20 masked");

    RewritePositionDeleteFiles::new(table.clone())
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
        "exactly one compacted pos-delete after compaction"
    );
    assert_eq!(
        pos_entries[0].1,
        Some(3),
        "the compacted pos-delete MUST carry the group MAX rewritten data seq (3), \
         not the inherited rewrite seq and not the min"
    );

    // And read identity still holds (the stamped delete still masks y=10 and y=20).
    assert_eq!(scan_y_values(&reloaded).await, HashSet::from([30]));
}

/// SEQ STAMPING — the resurrection guard. Data X at seq 1 is masked by two pos-deletes; a SECOND data
/// file W at seq 4 (committed AFTER the deletes) also lives. The compacted file must be stamped seq 3
/// (the group max of the rewritten deletes), which is `< 4` so it never touches W, and `> 1` so it still
/// masks X. If the stamp were inherited (seq 5 from the rewrite), it would `> 1` so X stays masked
/// (looks fine) — but the resurrection failure mode is the INVERSE: an OVER-low stamp. We pin the read
/// identity across BOTH data files so any wrong stamp that changes the masked set fails.
#[tokio::test]
async fn test_seq_stamp_does_not_resurrect_or_over_apply() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await; // X: seq 1

    let pd1 = write_position_delete_file(&table, Some(0), &[(&x_path, 0)]).await; // X.y=10
    let table = add_deletes(&catalog, &table, vec![pd1]).await; // seq 2
    let pd2 = write_position_delete_file(&table, Some(0), &[(&x_path, 1)]).await; // X.y=20
    let table = add_deletes(&catalog, &table, vec![pd2]).await; // seq 3

    // A NEW data file W (seq 4) with the same y values lives — the deletes must NOT touch it.
    let w = write_data_file(&table, "w.parquet", 0, &[(0, 10, 9), (0, 20, 10)]).await;
    let table = append_files(&catalog, &table, vec![w]).await; // W: seq 4

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 20]),
        "before: X fully masked; W (seq 4 > delete seqs) survives with y=10,20"
    );

    RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the compacted delete still masks X (seq 3 > 1) and never touches W (seq 4)"
    );
}

// =================================================================================================
// GROUPING + PARTITION ISOLATION.
// =================================================================================================

/// MULTI-FILE GROUPING across DATA files in one partition. Two data files in partition 0, each masked by
/// its own pos-delete file. Both pos-deletes share `(spec 0, partition 0)`, so they compact into ONE
/// file carrying both data files' positions. Read identity must hold.
#[tokio::test]
async fn test_multi_file_grouping_one_partition() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 0, &[(0, 30, 3), (0, 40, 4)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let pda = write_position_delete_file(&table, Some(0), &[(&a_path, 1)]).await; // a.y=20
    let table = add_deletes(&catalog, &table, vec![pda]).await;
    let pdb = write_position_delete_file(&table, Some(0), &[(&b_path, 0)]).await; // b.y=30
    let table = add_deletes(&catalog, &table, vec![pdb]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 40]),
        "before: y=20 and y=30 masked"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 2);
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the compacted file carries BOTH data files' positions"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
}

/// PARTITION ISOLATION + MULTI-GROUP TABLE ADVANCE. Two partitions, each with two pos-delete
/// files. The action compacts EACH partition's group SEPARATELY (one compacted file per
/// partition, never merging across partitions). Read identity per-partition must hold.
///
/// Each group commits its own `Replace` snapshot; the action advances the base table after each
/// group commit (mirrors `RewriteDataFiles`) so the next `Transaction` is built on the prior
/// group's tip. Without advance, `do_commit` still refreshes + re-applies (correctness holds;
/// cost is extra re-apply work — not forced CAS conflict retries).
///
/// MUTATION COVERAGE: collapse the `(spec, partition)` group key to spec-only and both partitions' files
/// would merge into one group; the compacted file's partition would be wrong and the per-partition read
/// identity would break (or the writer/commit would error on a partition mismatch).
#[tokio::test]
async fn test_partition_isolation_compacts_each_group_separately() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let p0 = write_data_file(&table, "p0.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let p1 = write_data_file(&table, "p1.parquet", 1, &[(1, 30, 3), (1, 40, 4)]).await;
    let p0_path = p0.file_path().to_string();
    let p1_path = p1.file_path().to_string();
    let table = append_files(&catalog, &table, vec![p0, p1]).await;

    // Partition 0: two pos-deletes (mask y=20 in two parts). Partition 1: two pos-deletes (mask y=40).
    let p0d1 = write_position_delete_file(&table, Some(0), &[(&p0_path, 1)]).await; // p0.y=20
    let p1d1 = write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await; // p1.y=40
    let table = add_deletes(&catalog, &table, vec![p0d1, p1d1]).await;
    let p0d2 = write_position_delete_file(&table, Some(0), &[(&p0_path, 1)]).await; // dup p0.y=20
    let p1d2 = write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await; // dup p1.y=40
    let table = add_deletes(&catalog, &table, vec![p0d2, p1d2]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30]),
        "before: y=20 (part 0) and y=40 (part 1) masked"
    );
    let history_before = table.metadata().history().len();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    // Two groups (one per partition), each compacting two files into one.
    assert_eq!(result.rewritten_delete_files_count, 4, "4 files rewritten");
    assert_eq!(
        result.added_delete_files_count, 2,
        "one compacted file per partition group"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity per-partition after independent group compaction"
    );
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        2,
        "exactly two compacted files (one per partition)"
    );
    // Two sequential group commits each append a snapshot log entry. (Advance is a re-apply
    // avoidance / RewriteDataFiles parity seam; history +2 pins multi-group commits, not that
    // CAS would fail without advance — `do_commit` refreshes a stale base on first attempt.)
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 2,
        "two group commits must each produce a Replace snapshot"
    );
}

/// FILTER restriction. Two partitions, each with two pos-deletes. `filter(x == 0)` compacts ONLY the
/// partition-0 group; partition 1's pos-deletes are left untouched. Read identity holds throughout.
#[tokio::test]
async fn test_filter_restricts_compacted_partitions() {
    use crate::expr::Reference;
    use crate::spec::Datum;

    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let p0 = write_data_file(&table, "p0.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let p1 = write_data_file(&table, "p1.parquet", 1, &[(1, 30, 3), (1, 40, 4)]).await;
    let p0_path = p0.file_path().to_string();
    let p1_path = p1.file_path().to_string();
    let table = append_files(&catalog, &table, vec![p0, p1]).await;

    let p0d1 = write_position_delete_file(&table, Some(0), &[(&p0_path, 1)]).await;
    let p1d1 = write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![p0d1, p1d1]).await;
    let p0d2 = write_position_delete_file(&table, Some(0), &[(&p0_path, 1)]).await;
    let p1d2 = write_position_delete_file(&table, Some(1), &[(&p1_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![p0d2, p1d2]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([10, 30]));

    let result = RewritePositionDeleteFiles::new(table.clone())
        .filter(Reference::new("x").equal_to(Datum::long(0)))
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "only the partition-0 group is compacted"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity preserved under filter"
    );
    // Partition 0 compacted (2 -> 1); partition 1 untouched (still 2). Total 3 live pos-deletes.
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        3,
        "partition 0 compacted to 1; partition 1's two files remain"
    );
}

// =================================================================================================
// V3 DELETION-VECTOR SCOPE — a DV is NOT compacted by this action.
// =================================================================================================

/// V2-PARQUET-ONLY SCOPE. On a V3 table, TWO data files in partition 0 are each masked by a Puffin
/// DELETION VECTOR. A DV is file-scoped and never bin-packed, so this action must SKIP both — even
/// though they share `(spec 0, partition 0)` and would otherwise form a compactable 2-file group. The
/// action is a no-op: both DVs stay live, the read set is unchanged.
///
/// MUTATION COVERAGE: drop the `file_format() != Parquet` skip and the two DVs would be enumerated as a
/// 2-file `(spec 0, partition 0)` "position delete" group — passing the `entries.len() < 2` guard — and
/// the action would try to read each Puffin DV as a parquet file (failing the read, or wrongly handling it).
/// This test (zero counts, both DVs intact, read identity) fails. The 2-DV group is what makes the skip
/// load-bearing (a single DV would be dropped by the single-file-group guard regardless).
#[tokio::test]
async fn test_v3_deletion_vectors_are_not_compacted() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V3).await;

    let a = write_data_file(&table, "a.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 0, &[(0, 30, 3), (0, 40, 4)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a, b]).await;

    // Two Puffin DVs in the SAME partition 0: one masks a.y=20 (pos 1), one masks b.y=30 (pos 0).
    let dva = write_deletion_vector(&table, &a_path, &[1]).await;
    let table = add_deletes(&catalog, &table, vec![dva]).await;
    let dvb = write_deletion_vector(&table, &b_path, &[0]).await;
    let table = add_deletes(&catalog, &table, vec![dvb]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 40]),
        "before: the two DVs mask a.y=20 and b.y=30"
    );

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "DVs are NOT compacted by this action — zero counts, no commit (even a 2-DV same-partition group)"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity: the DVs are untouched, the live set unchanged"
    );
    // Both Puffin DVs are still live, and none became a parquet pos-delete.
    let deletes = live_delete_files(&reloaded).await;
    assert_eq!(deletes.len(), 2, "both DVs remain live");
    assert!(
        deletes
            .iter()
            .all(|f| f.file_format() == DataFileFormat::Puffin),
        "every surviving delete is a Puffin DV (none was compacted into a parquet pos-delete)"
    );
}

/// Write a single-data-file Puffin DELETION VECTOR masking the given absolute positions of `target_path`,
/// in partition x=0. Uses the [`DVFileWriter`] (the same writer the DV write path uses), so the produced
/// `DeleteFile` is a faithful Puffin DV the scan applies.
async fn write_deletion_vector(table: &Table, target_path: &str, positions: &[u64]) -> DataFile {
    use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;

    let dv_path = format!(
        "{}/data/dv-{}.puffin",
        table.metadata().location(),
        uuid::Uuid::now_v7()
    );
    let output = table.file_io().new_output(&dv_path).unwrap();
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(0))]),
    )
    .expect("PartitionKey::new: valid partition tuple");
    let mut writer = DVFileWriter::new(output);
    for &pos in positions {
        writer
            .delete(target_path, pos, Some(&partition_key))
            .expect("record DV position");
    }
    writer
        .close()
        .await
        .expect("close DV writer")
        .into_iter()
        .next()
        .expect("one DV delete file")
}

// =================================================================================================
// NO-OP edges.
// =================================================================================================

/// No current snapshot → no-op, zero counts, no commit.
#[tokio::test]
async fn test_no_current_snapshot_is_a_no_op() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let result = RewritePositionDeleteFiles::new(table)
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result, RewritePositionDeleteFilesResult::default());
}

/// A group of ONLY ONE position-delete file → nothing to compact (Java's planner drops single-file
/// groups). No-op, zero counts, no new snapshot.
#[tokio::test]
async fn test_single_file_group_is_a_no_op() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;

    let x = write_data_file(&table, "x.parquet", 0, &[(0, 10, 1), (0, 20, 2)]).await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;

    let pd = write_position_delete_file(&table, Some(0), &[(&x_path, 1)]).await;
    let table = add_deletes(&catalog, &table, vec![pd]).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a single-file group is not compacted"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a no-op must NOT commit a new snapshot"
    );
    // Read identity trivially holds (the single delete is unchanged).
    assert_eq!(scan_y_values(&reloaded).await, HashSet::from([10]));
}

/// Unpartitioned table: two pos-delete files in the single unpartitioned group compact into one. Read
/// identity holds.
#[tokio::test]
async fn test_unpartitioned_group_compacts() {
    let (catalog, _temp) = local_fs_catalog().await;
    let table = create_unpartitioned_table(&catalog, FormatVersion::V2).await;

    let a = write_data_file(&table, "a.parquet", 0, &[(1, 10, 1), (2, 20, 2)]).await;
    let b = write_data_file(&table, "b.parquet", 0, &[(3, 30, 3), (4, 40, 4)]).await;
    let a_path = a.file_path().to_string();
    let b_path = b.file_path().to_string();
    let table = append_files(&catalog, &table, vec![a, b]).await;

    let pda = write_position_delete_file(&table, None, &[(&a_path, 1)]).await; // a.y=20
    let table = add_deletes(&catalog, &table, vec![pda]).await;
    let pdb = write_position_delete_file(&table, None, &[(&b_path, 0)]).await; // b.y=30
    let table = add_deletes(&catalog, &table, vec![pdb]).await;

    let before = scan_y_values(&table).await;
    assert_eq!(before, HashSet::from([10, 40]));

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(result.rewritten_delete_files_count, 2);
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        scan_y_values(&reloaded).await,
        before,
        "read identity (unpartitioned)"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
}

// =================================================================================================
// CONFIG SEAM — the size/count thresholds, their Java defaults, the delete-specific property, the
// `Long.parseLong`-exact parse, and Java's `sizeThresholds` preconditions.
//
// Every default here is asserted WHITE-BOX through `resolve_config()` (private, but reachable — this
// is a child module), never end to end: observing 50331648 / 120795955 end to end would need ~48 MiB
// and ~115 MiB fixtures. These pins are therefore named `test_config_*` / `test_resolve_config_*` /
// `test_parse_delete_target_*` and NEVER `test_admission_*`, so no later change lowers the target to
// make them end-to-end and thereby unpins the ratio constants.
// =================================================================================================

/// A minimal unpartitioned table carrying `properties` — the fixture for the white-box config pins.
/// The config seam reads only `metadata().properties()`, so the table needs no snapshot and no files.
async fn config_table(properties: &[(&str, &str)]) -> (TempDir, Table) {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = three_long_schema();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, std::collections::HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .properties(
            properties
                .iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect::<Vec<_>>(),
        )
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");
    (temp_dir, table)
}

/// `write.delete.target-file-size-bytes` as a `HashMap`, for the pins that exercise the parse function
/// alone (no table needed).
fn delete_target_property(value: &str) -> std::collections::HashMap<String, String> {
    std::collections::HashMap::from([(
        TableProperties::PROPERTY_WRITE_DELETE_TARGET_FILE_SIZE_BYTES.to_string(),
        value.to_string(),
    )])
}

// ------------------------------------------------------------------------------------------------
// C-035 — the parse: one function whose accept/reject domain is `Long.parseLong`'s, EXACTLY.
// ------------------------------------------------------------------------------------------------

/// C-035 element 1 (ABSENT). No property ⇒ the 64 MiB delete default.
#[test]
fn test_parse_delete_target_absent_is_64_mib() {
    let properties = std::collections::HashMap::new();
    assert_eq!(
        parse_delete_target_file_size(&properties).expect("an absent property takes the default"),
        67108864
    );
}

/// C-035 element 2 (a well-formed decimal in `[2, i64::MAX - 1]`) — the only class that can also
/// SURVIVE the preconditions. A leading `+` is accepted, matching `Long.parseLong`.
#[test]
fn test_parse_delete_target_mid_band_is_accepted() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("12345678")).expect("mid band"),
        12345678
    );
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("+12345678"))
            .expect("Long.parseLong accepts a leading '+', and so must this"),
        12345678
    );
}

/// C-035 element 3, upper endpoint. `i64::MAX` PARSES (this is a parse-function assertion only —
/// whether it then survives the preconditions is `resolve_config`'s business, pinned separately by
/// `test_resolve_config_rejects_target_at_i64_max_on_the_max_precondition`).
#[test]
fn test_parse_delete_target_at_i64_max_is_accepted_by_the_parse() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("9223372036854775807"))
            .expect("Long.parseLong accepts Long.MAX_VALUE"),
        i64::MAX
    );
}

/// C-035 element 3, lower endpoint. `"1"` PARSES (parse-function assertion only).
#[test]
fn test_parse_delete_target_at_one_is_accepted_by_the_parse() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("1")).expect("'1' parses"),
        1
    );
}

/// C-035 element 6 (unparsable / empty). Every one of these throws `NumberFormatException` in Java,
/// so every one must be a `DataInvalid` here — including the forms Rust's parser could plausibly have
/// accepted (`1_000` is a Rust literal, not a `Long.parseLong` input).
#[test]
fn test_parse_delete_target_unparsable_is_data_invalid() {
    for value in ["", "abc", "1_000", " 12", "12 ", "0x10", "1.0", "12,3", "+"] {
        let error = parse_delete_target_file_size(&delete_target_property(value))
            .expect_err("an unparsable delete target must be rejected loudly");
        assert_eq!(error.kind(), ErrorKind::DataInvalid, "value {value:?}");
        assert_eq!(
            error.message(),
            format!(
                "Invalid value '{value}' for table property 'write.delete.target-file-size-bytes'"
            ),
            "value {value:?}"
        );
    }
}

/// C-035 element 6 (magnitude outside the `long` range) — THE anti-`u64` pin. `Long.parseLong` throws
/// on anything above `Long.MAX_VALUE`, so these must be rejected AT THE PARSE. A `u64` parse would
/// accept every one of them and hand the action a threshold Java cannot express.
#[test]
fn test_parse_delete_target_above_i64_max_is_rejected() {
    for value in [
        "9223372036854775808",  // i64::MAX + 1
        "18446744073709551615", // u64::MAX — accepted by a u64 parse, rejected by Long.parseLong
        "99999999999999999999999",
    ] {
        let error = parse_delete_target_file_size(&delete_target_property(value))
            .expect_err("a value outside Java's long range must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid, "value {value:?}");
    }
    // ... and the boundary directly below it still parses, so the rejection is two-sided.
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("9223372036854775807"))
            .expect("i64::MAX itself parses"),
        i64::MAX
    );
}

/// C-035 element 4 (`"0"`). It PARSES — and is then rejected downstream by precondition (1) carrying
/// Java's verbatim `'%s' is set to %s but must be > 0`. A `u64` parse would also parse it, so the
/// discriminating half is the message.
#[tokio::test]
async fn test_parse_delete_target_zero_is_rejected_by_the_target_precondition() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("0")).expect("'0' parses"),
        0
    );

    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "0")]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect_err("a zero target must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' is set to 0 but must be > 0"
    );
}

/// C-035 element 5 (a NEGATIVE decimal). `"-1"` PARSES as `i64` exactly as `Long.parseLong` parses it
/// — a `u64` parse would reject it at the parse with a fork-only message — and is then rejected by
/// precondition (1).
///
/// This is also the ORDER pin for C-006: at `target = -1` this port's `d2l` saturates BOTH ratio
/// products to `0`, so `min = 0` and precondition (3) `target > min` (`-1 > 0`) is INDEPENDENTLY
/// false. Hoisting (3) above (1) therefore reports the min-threshold message instead, and this
/// assertion reds.
#[tokio::test]
async fn test_parse_delete_target_negative_is_rejected_by_the_target_precondition() {
    assert_eq!(
        parse_delete_target_file_size(&delete_target_property("-1"))
            .expect("'-1' parses as i64, exactly as Long.parseLong parses it"),
        -1
    );

    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "-1")]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect_err("a negative target must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' is set to -1 but must be > 0"
    );
}

// ------------------------------------------------------------------------------------------------
// C-002 / C-032 — the target resolves from the DELETE property, never from the data-file one.
// ------------------------------------------------------------------------------------------------

/// C-002. No property ⇒ 64 MiB (Java `defaultTargetFileSize`'s `ldc2_w 67108864L`).
#[tokio::test]
async fn test_config_target_file_size_default_is_64_mib() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.target_file_size_bytes, 67108864);
}

/// C-002. The DELETE-specific property moves the target.
#[tokio::test]
async fn test_config_target_file_size_resolves_delete_property() {
    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "12345678")]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("12345678 satisfies every precondition");
    assert_eq!(config.target_file_size_bytes, 12345678);
    // The ratios follow the resolved target, not the default.
    assert_eq!(config.min_file_size_bytes, 9259258);
    assert_eq!(config.max_file_size_bytes, 22222220);
}

/// C-002, the NEGATIVE CONTROL. `write.target-file-size-bytes` is the DATA-file target (512 MiB) and
/// must not move the delete target by so much as a byte — Java's
/// `BinPackRewritePositionDeletePlanner.defaultTargetFileSize` never reads it. Reds if the port
/// resolves from `TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES`.
#[tokio::test]
async fn test_config_write_target_file_size_does_not_move_delete_target() {
    let (_temp, table) = config_table(&[("write.target-file-size-bytes", "536870912")]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the delete defaults still hold");
    assert_eq!(
        config.target_file_size_bytes, 67108864,
        "the data-file target must not leak into the delete target"
    );
    assert_ne!(config.target_file_size_bytes, 536870912);
}

/// C-001. The builder wins over the property.
#[tokio::test]
async fn test_config_target_file_size_builder_overrides_property() {
    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "12345678")]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(99_999_999)
        .resolve_config()
        .expect("99999999 satisfies every precondition");
    assert_eq!(config.target_file_size_bytes, 99_999_999);
}

// ------------------------------------------------------------------------------------------------
// C-007 — the ratio defaults through the shared `d2l` helper.
// ------------------------------------------------------------------------------------------------

/// C-007. `min = d2l(target * 0.75)`; at the 64 MiB delete default that is EXACTLY 50331648 (dyadic,
/// so no rounding is involved). Asserted as a literal so the ratio constant cannot drift.
#[tokio::test]
async fn test_config_min_file_size_default_is_three_quarters_target() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.min_file_size_bytes, 50331648);
}

/// C-007. `max = d2l(target * 1.8)`; at the 64 MiB delete default `1.8 * 2^26 = 120795955.2`, which
/// Java's `d2l` TRUNCATES to 120795955. Asserted as a literal, so both the ratio and the truncation
/// (vs rounding) are pinned.
#[tokio::test]
async fn test_config_max_file_size_default_is_one_point_eight_target() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.max_file_size_bytes, 120795955);
}

/// C-007, THE CLAMP. Java's `d2l` saturates at `Long.MAX_VALUE`; Rust's `as u64` saturates at
/// `u64::MAX`, so `.min(i64::MAX as u64)` is the parity act.
///
/// The window in which it is OBSERVABLE is `target ∈ (2^63 / 1.8, i64::MAX - 1]`: there the clamped
/// max IS `i64::MAX` while the unclamped one is larger, and BOTH are above the target, so
/// `resolve_config` returns `Ok` either way and the equality is the discriminator rather than the
/// Ok/Err split. `6e18` sits inside it.
#[tokio::test]
async fn test_config_max_file_size_clamps_to_java_long_max() {
    const TARGET: u64 = 6_000_000_000_000_000_000;
    // Fixture preconditions: inside the observable window, and the unclamped product really does
    // exceed Java's ceiling (otherwise this test would pass with the clamp deleted).
    assert!(
        (TARGET as f64) > (i64::MAX as f64) / 1.8,
        "fixture: the target must sit above 2^63 / 1.8"
    );
    let unclamped = (TARGET as f64 * 1.8) as u64;
    assert!(
        unclamped > i64::MAX as u64,
        "fixture: the UNCLAMPED product must exceed Long.MAX_VALUE, else the clamp is unobservable"
    );

    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(TARGET)
        .resolve_config()
        .expect("target < clamped max, so this resolves");
    assert_eq!(
        config.max_file_size_bytes,
        i64::MAX as u64,
        "d2l must saturate at Long.MAX_VALUE, not at u64::MAX"
    );
    assert_ne!(config.max_file_size_bytes, unclamped);
}

// ------------------------------------------------------------------------------------------------
// C-001 — the remaining two ported options and their defaults.
// ------------------------------------------------------------------------------------------------

/// C-001. `min_input_files` defaults to Java's `MIN_INPUT_FILES_DEFAULT = 5`. Asserted with
/// `assert_eq!` so it reds on a raise AND on a lower.
///
/// TODO(tests): the two-sided ADMISSION pin — four files declined, five admitted, at the DEFAULT
/// config — belongs to the gate, not to this config seam, and is owed by the test increment.
#[tokio::test]
async fn test_config_min_input_files_default_is_five() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.min_input_files, 5);
}

/// C-001. `max_file_group_size_bytes` defaults to Java's
/// `MAX_FILE_GROUP_SIZE_BYTES_DEFAULT = 107374182400` (100 GiB).
#[tokio::test]
async fn test_config_max_file_group_size_default_is_100_gib() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect("the defaults satisfy every precondition");
    assert_eq!(config.max_file_group_size_bytes, 107374182400);
    assert_eq!(config.max_file_group_size_bytes, 100 * 1024 * 1024 * 1024);
}

/// C-001. All five ported builders land on the resolved config — one assertion per option, so a
/// builder wired to the wrong field reds.
#[tokio::test]
async fn test_config_explicit_overrides_are_returned_unchanged() {
    let (_temp, table) = config_table(&[]).await;
    let config = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(200_000)
        .min_file_size_bytes(100_000)
        .max_file_size_bytes(400_000)
        .min_input_files(2)
        .max_file_group_size_bytes(1_000_000)
        .resolve_config()
        .expect("a legal knob combination");
    assert_eq!(config.target_file_size_bytes, 200_000);
    assert_eq!(config.min_file_size_bytes, 100_000);
    assert_eq!(config.max_file_size_bytes, 400_000);
    assert_eq!(config.min_input_files, 2);
    assert_eq!(config.max_file_group_size_bytes, 1_000_000);
}

// ------------------------------------------------------------------------------------------------
// C-006 — Java's `sizeThresholds` preconditions, each with Java's verbatim message.
// ------------------------------------------------------------------------------------------------

/// C-006 precondition (1), via the builder path.
#[tokio::test]
async fn test_resolve_config_rejects_target_zero() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(0)
        .resolve_config()
        .expect_err("a zero target must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' is set to 0 but must be > 0"
    );
}

/// C-006 precondition (1) at a NEGATIVE target — the ORDER-discriminating pin. See
/// `test_parse_delete_target_negative_is_rejected_by_the_target_precondition` for why `min` is `0`
/// here and why that makes precondition (3) independently false.
#[tokio::test]
async fn test_resolve_config_rejects_target_negative_with_the_must_be_positive_message() {
    let (_temp, table) = config_table(&[("write.delete.target-file-size-bytes", "-7")]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect_err("a negative target must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' is set to -7 but must be > 0",
        "precondition (1) must report first: at a negative target the saturating d2l makes min = 0, \
         so (3) `target > min` is independently false and a hoisted (3) would report instead"
    );
}

/// C-006 precondition (3), STRICT: `target == min` is rejected, `target == min + 1` is not.
#[tokio::test]
async fn test_resolve_config_rejects_target_le_min() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .target_file_size_bytes(200_000)
        .min_file_size_bytes(200_000)
        .max_file_size_bytes(400_000)
        .resolve_config()
        .expect_err("target == min must be rejected (STRICT >)");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' (200000) must be > 'min-file-size-bytes' (200000), all new files \
         will be smaller than the min threshold"
    );

    // Two-sided: one byte above the min is legal, so the pin is not vacuous.
    RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(200_001)
        .min_file_size_bytes(200_000)
        .max_file_size_bytes(400_000)
        .resolve_config()
        .expect("target == min + 1 is legal");
}

/// C-006 precondition (4), STRICT: `target == max` is rejected, `target == max - 1` is not.
#[tokio::test]
async fn test_resolve_config_rejects_target_ge_max() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .target_file_size_bytes(400_000)
        .min_file_size_bytes(100_000)
        .max_file_size_bytes(400_000)
        .resolve_config()
        .expect_err("target == max must be rejected (STRICT <)");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' (400000) must be < 'max-file-size-bytes' (400000), all new files \
         will be larger than the max threshold"
    );

    // Two-sided: one byte below the max is legal.
    RewritePositionDeleteFiles::new(table)
        .target_file_size_bytes(399_999)
        .min_file_size_bytes(100_000)
        .max_file_size_bytes(400_000)
        .resolve_config()
        .expect("target == max - 1 is legal");
}

/// C-006 precondition (5).
#[tokio::test]
async fn test_resolve_config_rejects_min_input_files_zero() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(0)
        .resolve_config()
        .expect_err("min_input_files = 0 must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'min-input-files' is set to 0 but must be > 0"
    );

    RewritePositionDeleteFiles::new(table)
        .min_input_files(1)
        .resolve_config()
        .expect("min_input_files = 1 is legal");
}

/// C-006 precondition (6).
#[tokio::test]
async fn test_resolve_config_rejects_max_file_group_size_zero() {
    let (_temp, table) = config_table(&[]).await;
    let error = RewritePositionDeleteFiles::new(table.clone())
        .max_file_group_size_bytes(0)
        .resolve_config()
        .expect_err("max_file_group_size_bytes = 0 must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'max-file-group-size-bytes' is set to 0 but must be > 0"
    );

    RewritePositionDeleteFiles::new(table)
        .max_file_group_size_bytes(1)
        .resolve_config()
        .expect("max_file_group_size_bytes = 1 is legal");
}

/// C-006 precondition (7) — ONE LEG PER KNOB. Java's thresholds are `long`s fed by `Long.parseLong`,
/// so an override above `i64::MAX` is a config Java cannot express; admitting it would open a state
/// in which `too_much_content` (`input_size > max`) is unreachable for every possible input.
///
/// Each leg sets exactly ONE over-range knob and asserts that knob's own message, which is what
/// forces (7) to run at override-READ time: checked after (3)/(4) instead, the `min` leg would be
/// caught by (3) and the `target` leg by (4), both with the wrong message.
#[tokio::test]
async fn test_resolve_config_rejects_size_override_above_i64_max() {
    const ABOVE: u64 = i64::MAX as u64 + 1;
    let (_temp, table) = config_table(&[]).await;

    for (option, action) in [
        (
            "target-file-size-bytes",
            RewritePositionDeleteFiles::new(table.clone()).target_file_size_bytes(ABOVE),
        ),
        (
            "min-file-size-bytes",
            RewritePositionDeleteFiles::new(table.clone()).min_file_size_bytes(ABOVE),
        ),
        (
            "max-file-size-bytes",
            RewritePositionDeleteFiles::new(table.clone()).max_file_size_bytes(ABOVE),
        ),
    ] {
        let error = action
            .resolve_config()
            .expect_err("an override above i64::MAX must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid, "option {option}");
        assert_eq!(
            error.message(),
            format!(
                "Invalid value '{ABOVE}' for '{option}': it must be <= 9223372036854775807 — \
                 Java's option is a `long`, so a larger threshold has no Java analogue"
            ),
            "option {option}"
        );
    }

    // Two-sided: `i64::MAX` itself is accepted BY (7) on every knob (the `max` leg then resolves
    // cleanly; the other two are rejected later, by (3)/(4), which is a different message).
    RewritePositionDeleteFiles::new(table)
        .max_file_size_bytes(i64::MAX as u64)
        .resolve_config()
        .expect("i64::MAX is inside Java's long domain");
}

/// C-035 element 3 downstream. `i64::MAX` PARSES, then falls to precondition (4): its defaulted max
/// CLAMPS to `i64::MAX`, so `target < max` is false. Asserts `Err` + the message, never a resolved
/// value.
#[tokio::test]
async fn test_resolve_config_rejects_target_at_i64_max_on_the_max_precondition() {
    let (_temp, table) =
        config_table(&[("write.delete.target-file-size-bytes", "9223372036854775807")]).await;
    let error = RewritePositionDeleteFiles::new(table)
        .resolve_config()
        .expect_err("target == the clamped max must be rejected");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        error.message(),
        "'target-file-size-bytes' (9223372036854775807) must be < 'max-file-size-bytes' \
         (9223372036854775807), all new files will be larger than the max threshold"
    );
}
