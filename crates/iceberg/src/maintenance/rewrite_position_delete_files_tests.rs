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

/// PARTITION ISOLATION. Two partitions, each with two pos-delete files. The action compacts EACH
/// partition's group SEPARATELY (one compacted file per partition, never merging across partitions).
/// Read identity per-partition must hold.
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
    );
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
