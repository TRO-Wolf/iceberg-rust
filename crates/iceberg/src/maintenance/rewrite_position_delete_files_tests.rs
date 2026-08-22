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
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is READ IDENTITY across a compaction — not
/// admission. It therefore sets `.min_input_files(2)` explicitly, which is also acceptance item
/// 2: the floor is configuration, not a hard-coded constant. Removing that knob must RED this
/// test — it asserts post-`execute` SHAPE, never read identity alone.
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
        .min_input_files(2)
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
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the sequence-number STAMP the compacted
/// file carries — not admission. It therefore sets `.min_input_files(2)` explicitly, which is
/// also acceptance item 2: the floor is configuration, not a hard-coded constant. Removing that
/// knob must RED this test — it asserts post-`execute` SHAPE, never read identity alone.
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
        .min_input_files(2)
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
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the sequence-number STAMP (resurrection
/// / over-apply) — not admission. It therefore sets `.min_input_files(2)` explicitly, which is
/// also acceptance item 2: the floor is configuration, not a hard-coded constant. Removing that
/// knob must RED this test — it asserts post-`execute` SHAPE, never read identity alone.
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

    let result = RewritePositionDeleteFiles::new(table.clone())
        .min_input_files(2)
        .execute(&catalog)
        .await
        .unwrap();

    // SHAPE FIRST (acceptance criterion 4 / design call 2). Read identity ALONE is satisfied by a
    // DECLINED bin doing nothing, so this test asserts the bin was actually ADMITTED before it
    // asserts anything about what the admitted run produced. Remove the `.min_input_files(2)` knob
    // above and these two assertions red.
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "the two pos-deletes must actually be rewritten — a declined bin rewrites nothing"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let pos_entries: Vec<(DataFile, Option<i64>)> = live_delete_entries_with_seq(&reloaded)
        .await
        .into_iter()
        .filter(|(f, _)| f.content_type() == DataContentType::PositionDeletes)
        .collect();
    assert_eq!(
        pos_entries.len(),
        1,
        "the two pos-deletes compacted into ONE — a declined bin would leave two"
    );
    // POWER ON ITS OWN SUBJECT: pin the EXACT stamp, not just the row set. Read identity cannot
    // separate `max` from `inherit` here (both are > X's seq 1, so X stays masked either way), and
    // it cannot separate `max` from `min` either (2 and 3 both mask X and both miss W). The stamp
    // assertion can, and does.
    assert_eq!(
        pos_entries[0].1,
        Some(3),
        "the compacted pos-delete MUST carry the group MAX rewritten data seq (3): `< 4` so it \
         never touches W, `> 1` so it still masks X"
    );
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
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is GROUPING across data files in one
/// partition — not admission. It therefore sets `.min_input_files(2)` explicitly, which is also
/// acceptance item 2: the floor is configuration, not a hard-coded constant. Removing that knob
/// must RED this test — it asserts post-`execute` SHAPE, never read identity alone.
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
        .min_input_files(2)
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
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is PARTITION ISOLATION and the per-group
/// commit — not admission. It therefore sets `.min_input_files(2)` explicitly, which is also
/// acceptance item 2: the floor is configuration, not a hard-coded constant. Removing that knob
/// must RED this test — it asserts post-`execute` SHAPE, never read identity alone.
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
        .min_input_files(2)
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
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the user FILTER stage (C-005 S2) — not
/// admission. It therefore sets `.min_input_files(2)` explicitly, which is also acceptance item
/// 2: the floor is configuration, not a hard-coded constant. Removing that knob must RED this
/// test — it asserts post-`execute` SHAPE, never read identity alone.
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
        .min_input_files(2)
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
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the V2-PARQUET-ONLY scope — not
/// admission. It therefore sets `.min_input_files(2)` explicitly, which is also acceptance item
/// 2: the floor is configuration, not a hard-coded constant. Its PIN FORM is the APPLIED
/// mutation above, NOT knob removal: with the Parquet skip in place both DVs are dropped at S1,
/// so the action is a no-op with or without the knob, and removing it leaves this test GREEN.
/// The knob is load-bearing only for the MUTATED build, where the two DVs form an admissible
/// 2-file bin. The stronger rework — five Puffin DVs at the default config, no knob — is ruled
/// to the test increment.
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
        .min_input_files(2)
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
/// MIGRATED (size gate): this fixture is a TWO-file group, deliberately BELOW Java's
/// `MIN_INPUT_FILES_DEFAULT` of 5, because its subject is the UNPARTITIONED group key — not
/// admission. It therefore sets `.min_input_files(2)` explicitly, which is also acceptance item
/// 2: the floor is configuration, not a hard-coded constant. Removing that knob must RED this
/// test — it asserts post-`execute` SHAPE, never read identity alone.
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
        .min_input_files(2)
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

// =================================================================================================
// THE ADMISSION GATE — the candidate filter (C-004), the six-stage pipeline order (C-005), the
// three-clause group filter (C-003), the shared bin packer (C-027) and the saturating input sum
// (C-041).
//
// FIXTURE DISCIPLINE (C-036), applied without exception below: a size-class fixture is NEVER sized
// to fixed knobs. The position-delete file is written first, its `file_size_in_bytes` is MEASURED
// off the returned `DataFile`, and the knobs are then set AROUND that measurement. Every size-class
// test asserts its measured size against the RESOLVED thresholds BEFORE `execute`, so a fixture that
// drifts out of its size class fails loudly instead of passing vacuously. Where a knob derivation
// makes a window assert an algebraic identity in the fixture size, that is RECORDED as true by
// construction rather than dressed up as a check.
// =================================================================================================

/// Write a position-delete file masking `count` consecutive positions of `target_path` starting at
/// `first_pos`, in partition 0. This is the SIZE DIAL for the size-class fixtures: more masked
/// positions ⇒ a strictly larger file (the `file_path` column dictionary-encodes to one value, so the
/// growth is the `pos` column). The size is never predicted — it is measured off the return value.
async fn write_sized_pos_delete(
    table: &Table,
    target_path: &str,
    first_pos: i64,
    count: i64,
) -> DataFile {
    let pairs: Vec<(&str, i64)> = (first_pos..first_pos + count)
        .map(|pos| (target_path, pos))
        .collect();
    write_position_delete_file(table, Some(0), &pairs).await
}

/// A partitioned table with ONE five-row data file (y = 10,20,30,40,50 at positions 0..5) already
/// committed — the base every gate fixture below builds its position-delete files on. Returns the
/// catalog, the temp dir (kept alive), the table and the data file's path.
async fn gate_table() -> (impl Catalog, TempDir, Table, String) {
    let (catalog, temp) = local_fs_catalog().await;
    let table = create_partitioned_table(&catalog, FormatVersion::V2).await;
    let x = write_data_file(&table, "x.parquet", 0, &[
        (0, 10, 100),
        (0, 20, 200),
        (0, 30, 300),
        (0, 40, 400),
        (0, 50, 500),
    ])
    .await;
    let x_path = x.file_path().to_string();
    let table = append_files(&catalog, &table, vec![x]).await;
    (catalog, temp, table, x_path)
}

/// The live position-delete files in MANIFEST ORDER — the same order
/// `collect_position_delete_groups` walks, so a fixture whose packing depends on order can assert
/// the order it actually gets instead of assuming it.
async fn live_pos_delete_paths(table: &Table) -> Vec<String> {
    live_delete_files(table)
        .await
        .iter()
        .filter(|f| f.content_type() == DataContentType::PositionDeletes)
        .map(|f| f.file_path().to_string())
        .collect()
}

// ------------------------------------------------------------------------------------------------
// C-003 element 1 — `enough_input_files`'s `size > 1` conjunct.
// ------------------------------------------------------------------------------------------------

/// C-003 element 1. `.min_input_files(1)` with a SINGLE sub-min position-delete file. The file IS a
/// candidate (sub-min) and forms a bin of one, so the gate is reached; `enough_input_files` is
/// `size > 1 && size >= min_input_files`, and only the `size > 1` conjunct declines it — `1 >= 1`
/// holds. `enough_content` is false for the same reason, and the file is far below `max`, so
/// `too_much_content` is false too.
///
/// MUTATION COVERAGE: this is the ONLY killer of `enough_input_files`'s `size > 1` conjunct. Drop it
/// and the bin is admitted, producing a Replace snapshot and non-zero counts.
#[tokio::test]
async fn test_admission_min_input_files_one_still_declines_lone_sub_min_file() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size = pd.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd]).await;

    let action = || RewritePositionDeleteFiles::new(table.clone()).min_input_files(1);
    let config = action()
        .resolve_config()
        .expect("the defaults plus min_input_files(1) are legal");

    // PRECONDITIONS (measured against the RESOLVED thresholds, before execute).
    assert!(
        size < config.min_file_size_bytes,
        "fixture: the lone file must be SUB-MIN so it is a candidate and reaches the gate \
         (measured {size}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        size <= config.max_file_size_bytes,
        "fixture: too_much_content must NOT fire, or the pin would pass for the wrong reason"
    );
    assert_eq!(
        config.min_input_files, 1,
        "fixture: the floor is lowered to 1"
    );

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a bin of ONE is declined by `size > 1` even at min_input_files(1)"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined bin must NOT commit"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-003 element 2 (+ C-001, and the execute-altitude proof that the resolved config is CONSULTED).
// ------------------------------------------------------------------------------------------------

/// C-003 element 2, TWO-SIDED at the DEFAULT config — the acceptance-criterion-1 pin and THE parity
/// number this whole change is about. Four small position-delete files in one partition are DECLINED
/// under Java's `MIN_INPUT_FILES_DEFAULT = 5`; a fifth flips the same partition to ADMITTED.
///
/// NON-VACUITY: the fixture asserts that the four/five-file input size is at or below BOTH the
/// resolved target and the resolved max, so neither `enough_content` nor `too_much_content` can be
/// the admitter — the fifth file admits through `enough_input_files` and nothing else.
///
/// MUTATION COVERAGE, two-sided and both applied: `MIN_INPUT_FILES_DEFAULT` 5 → 4 admits the
/// four-file case (first half reds); 5 → 6 declines the five-file case (second half reds). It is
/// also the EXECUTE-ALTITUDE pin that the resolved config is actually consumed by the planner: the
/// action sets no builder overrides at all, so a planner that ignored `config.min_input_files` (say,
/// by keeping a hard-coded floor of 2) admits the four-file case and reds here.
#[tokio::test]
async fn test_admission_min_input_files_default_five_declines_four_admits_five() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // FOUR position-delete files, each masking one position of X.
    let mut sizes: Vec<u64> = Vec::new();
    let mut table = table;
    for pos in 0..4i64 {
        let pd = write_sized_pos_delete(&table, &x_path, pos, 1).await;
        sizes.push(pd.file_size_in_bytes);
        table = add_deletes(&catalog, &table, vec![pd]).await;
    }

    let config = RewritePositionDeleteFiles::new(table.clone())
        .resolve_config()
        .expect("the pure defaults are legal");
    assert_eq!(
        config.min_input_files, 5,
        "fixture: this pin runs at Java's DEFAULT floor, with no builder override"
    );
    let four_input: u64 = sizes.iter().sum();
    for size in &sizes {
        assert!(
            *size < config.min_file_size_bytes,
            "fixture: every file must be SUB-MIN so all four are candidates \
             (measured {size}, resolved min {})",
            config.min_file_size_bytes
        );
    }
    assert!(
        four_input <= config.target_file_size_bytes && four_input <= config.max_file_size_bytes,
        "fixture NON-VACUITY: the four-file input ({four_input}) must clear NEITHER the target nor \
         the max, so only the COUNT clause can ever admit this partition"
    );

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([50]),
        "before: positions 0..4 are masked"
    );
    let snapshot_before = table.metadata().current_snapshot_id();

    // (a) FOUR files at the default floor of five ⇒ DECLINED.
    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "FOUR small position-delete files are BELOW Java's floor of five — not rewritten"
    );
    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined partition must NOT commit"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 4);

    // (b) A FIFTH file in the same partition ⇒ ADMITTED.
    let pd5 = write_sized_pos_delete(&reloaded, &x_path, 4, 1).await;
    let five_input = four_input + pd5.file_size_in_bytes;
    assert!(
        pd5.file_size_in_bytes < config.min_file_size_bytes,
        "fixture: the fifth file is SUB-MIN too"
    );
    assert!(
        five_input <= config.target_file_size_bytes && five_input <= config.max_file_size_bytes,
        "fixture NON-VACUITY: the five-file input ({five_input}) still clears neither size clause"
    );
    let table = add_deletes(&catalog, &reloaded, vec![pd5]).await;
    let before_five = scan_y_values(&table).await;

    let result = RewritePositionDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 5,
        "FIVE files meet Java's floor — the whole bin is rewritten"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        1,
        "the five position-delete files compact into one"
    );
    assert_eq!(
        scan_y_values(&reloaded).await,
        before_five,
        "read identity across the admitted compaction"
    );
}

// ------------------------------------------------------------------------------------------------
// C-003 element 4 — `enough_content` and its STRICTNESS at the target.
// ------------------------------------------------------------------------------------------------

/// C-003 element 4. TWO sub-min position-delete files whose MEASURED sizes sum to just over the
/// target: below the count floor of five, so only `enough_content` can admit them.
///
/// KNOBS FROM THE MEASUREMENTS: `target := S_A + S_B - 1`, with min/max defaulted (0.75x / 1.8x), so
/// both files are sub-min candidates and the bin's input is one byte over the target.
///
/// RECORDED AS TRUE BY CONSTRUCTION, not asserted as if it were falsifiable: `input_size > target`
/// is an identity under `target := S_A + S_B - 1`. The falsifiable content is on the MEASURED
/// OUTPUT — one Replace snapshot, two files rewritten, one added — plus the preconditions that make
/// the OTHER two clauses provably false.
///
/// MUTATION COVERAGE: delete the `enough_content` disjunct and the bin is declined (2 < 5, and the
/// input is far below max) — zero counts, no snapshot.
#[tokio::test]
async fn test_admission_enough_content_admits_two_files_over_target() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd_a = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size_a = pd_a.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_a]).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 3, 1).await;
    let size_b = pd_b.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_b]).await;

    let target = size_a + size_b - 1;
    let action = || RewritePositionDeleteFiles::new(table.clone()).target_file_size_bytes(target);
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS.
    assert!(
        size_a < config.min_file_size_bytes && size_b < config.min_file_size_bytes,
        "fixture: both files must be SUB-MIN candidates (measured {size_a}/{size_b}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        2 < config.min_input_files,
        "fixture NON-VACUITY: two files are BELOW the count floor, so enough_input_files is false"
    );
    assert!(
        size_a + size_b <= config.max_file_size_bytes,
        "fixture NON-VACUITY: the input must NOT exceed max, so too_much_content is false too"
    );

    let before = scan_y_values(&table).await;
    let history_before = table.metadata().history().len();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "enough_content admits a two-file bin whose bytes exceed the target"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 1,
        "exactly one Replace snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 1);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-003 element 4, THE BOUNDARY. The same two sub-min files, but `target := S_A + S_B` exactly.
/// `enough_content` is `input_size > target`, STRICT, so an input EQUAL to the target is declined.
///
/// RECORDED AS TRUE BY CONSTRUCTION: `input_size == target` is an identity under the knob choice.
/// The falsifiable content is the MEASURED OUTCOME — zero counts and no new snapshot — plus the
/// preconditions that rule out the other two clauses.
///
/// MUTATION COVERAGE: relax `input_size > target` to `>=` and this bin is admitted.
#[tokio::test]
async fn test_admission_input_size_exactly_target_is_declined() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd_a = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size_a = pd_a.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_a]).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 3, 1).await;
    let size_b = pd_b.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_b]).await;

    let target = size_a + size_b;
    let action = || RewritePositionDeleteFiles::new(table.clone()).target_file_size_bytes(target);
    let config = action().resolve_config().expect("legal knobs");

    assert!(
        size_a < config.min_file_size_bytes && size_b < config.min_file_size_bytes,
        "fixture: both files must be SUB-MIN candidates, so the bin REACHES the gate"
    );
    assert!(2 < config.min_input_files, "fixture: below the count floor");
    assert!(
        size_a + size_b <= config.max_file_size_bytes,
        "fixture: too_much_content must be false"
    );

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "an input EXACTLY at the target is declined — the comparison is STRICT"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "a declined bin must NOT commit"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 2);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-003 element 5 — `too_much_content` EXISTS and carries NO `size > 1` guard.
// ------------------------------------------------------------------------------------------------

/// C-003 element 5, and acceptance criterion 3's ADMISSION leg. A LONE position-delete file above
/// `max_file_size_bytes` IS admitted: `too_much_content` is `input_size > max_file_size_bytes` with
/// no `size > 1` guard, unlike the other two clauses.
///
/// KNOBS FROM THE MEASUREMENT: `max := S - 1`, `target := S - 2`, `min := S - 3`, so the single file
/// is oversized (hence a candidate) and forms a bin of one.
///
/// This test pins ADMISSION only. What the writer then does with an oversized input — the roll bound
/// and the fixed point — is the writer increment's subject, not this clause's.
///
/// MUTATION COVERAGE: delete the `too_much_content` disjunct and this bin is declined by both
/// `size > 1` guards — zero counts, no snapshot.
#[tokio::test]
async fn test_admission_too_much_content_admits_lone_oversized_file() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size = pd.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size - 3)
            .target_file_size_bytes(size - 2)
            .max_file_size_bytes(size - 1)
    };
    let config = action().resolve_config().expect("legal knobs");

    assert!(
        size > config.max_file_size_bytes,
        "fixture: the lone file must be OVERSIZED (measured {size}, resolved max {})",
        config.max_file_size_bytes
    );
    assert!(
        1 < config.min_input_files,
        "fixture NON-VACUITY: one file is below the count floor, so enough_input_files is false"
    );

    let before = scan_y_values(&table).await;
    let history_before = table.metadata().history().len();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 1,
        "too_much_content admits a LONE oversized file — it has no `size > 1` guard"
    );
    assert!(result.added_delete_files_count >= 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 1,
        "exactly one Replace snapshot"
    );
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-004 — the size-only candidate filter, and its two STRICT boundaries.
// ------------------------------------------------------------------------------------------------

/// C-004 element 1 (C-036 recipe 8), and C-042's OUTCOME pin. FIVE position-delete files in one
/// partition, every one of them strictly IN RANGE, so `outsideDesiredFileSizeRange` rejects all
/// five: zero candidates, zero bins, zero commits — even though five files WOULD satisfy
/// `enough_input_files` had they reached the gate.
///
/// KNOBS FROM THE MEASUREMENTS: `min := min_i(S_i)`, `max := max_i(S_i) + 1`, `target := min + 1`
/// (the five files are written with different masked-position counts so their sizes differ).
///
/// MUTATION COVERAGE: delete the candidate filter and all five files reach the packer as one bin,
/// which `enough_input_files` (5 >= 5) then admits — non-zero counts and a Replace snapshot.
///
/// EXPLICITLY NOT A MUTANT-KILLER for C-042's `if candidates.is_empty() { continue; }`
/// short-circuit: deleting that branch is observationally a no-op, because `pack_bins` on an empty
/// input returns an empty `Vec` and the per-bin loop then never runs. That line is recorded as
/// UNKILLABLE, not as covered.
#[tokio::test]
async fn test_candidate_filter_drops_in_range_files_before_packing() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let mut sizes: Vec<u64> = Vec::new();
    let mut table = table;
    for count in 1..=5i64 {
        let pd = write_sized_pos_delete(&table, &x_path, 100, count).await;
        sizes.push(pd.file_size_in_bytes);
        table = add_deletes(&catalog, &table, vec![pd]).await;
    }
    let smallest = *sizes.iter().min().expect("five files");
    let largest = *sizes.iter().max().expect("five files");
    assert!(
        largest >= smallest + 2,
        "fixture: the five measured sizes must span at least 2 bytes so `target = min + 1` is \
         strictly below `max` (measured {smallest}..{largest})"
    );

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(smallest)
            .target_file_size_bytes(smallest + 1)
            .max_file_size_bytes(largest + 1)
    };
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS. The two window legs are true by construction (min := min_i, max := max_i + 1);
    // they are asserted anyway because they also check that the builders reach the resolved config.
    for size in &sizes {
        assert!(
            *size >= config.min_file_size_bytes && *size <= config.max_file_size_bytes,
            "fixture: every file must be IN RANGE (measured {size}, resolved [{}, {}])",
            config.min_file_size_bytes,
            config.max_file_size_bytes
        );
    }
    assert!(
        5 >= config.min_input_files,
        "fixture NON-VACUITY: five files WOULD satisfy enough_input_files if the candidate filter \
         did not drop them first"
    );

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a partition of well-sized files yields no candidate, no bin and no commit"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "no candidate ⇒ no snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 5);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-004 element 2 (C-036 recipe 4). `length < min_file_size_bytes` is STRICT: a file whose MEASURED
/// size is EXACTLY the resolved min is NOT a candidate.
///
/// A lone boundary file cannot discriminate this (both the mutated and the unmutated planner leave a
/// declined, zero-count table), so the fixture adds a strictly smaller COMPANION that makes
/// candidacy outcome-bearing: unmutated, only the companion is a candidate and its bin of one is
/// declined; mutated to `<=`, the boundary file joins it and the two-file bin clears the target.
///
/// MUTATION COVERAGE: `length < min` → `length <= min` admits the bin and reds this test.
#[tokio::test]
async fn test_candidate_filter_keeps_file_at_exactly_min_file_size() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // A is the BOUNDARY file (more masked positions ⇒ strictly larger); B is the sub-min companion.
    let pd_a = write_sized_pos_delete(&table, &x_path, 100, 5).await;
    let size_a = pd_a.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_a]).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size_b = pd_b.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_b]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size_a)
            .target_file_size_bytes(size_a + 1)
            .max_file_size_bytes(size_a + 2)
    };
    let config = action().resolve_config().expect("legal knobs");

    assert_eq!(
        size_a, config.min_file_size_bytes,
        "fixture: A sits EXACTLY on the resolved min"
    );
    assert!(
        size_b < config.min_file_size_bytes,
        "fixture: B is STRICTLY sub-min (measured {size_b}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        size_a + size_b > config.target_file_size_bytes,
        "fixture NON-VACUITY: were A wrongly a candidate, the two-file bin WOULD clear the target"
    );
    assert!(2 < config.min_input_files, "fixture: below the count floor");

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a file EXACTLY at min is not a candidate, so only B is packed — and a bin of one is declined"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "declined ⇒ no snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 2);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-004 element 3 (C-036 recipe 4, mirrored). `length > max_file_size_bytes` is STRICT: a file whose
/// MEASURED size is EXACTLY the resolved max is NOT a candidate. Same companion construction as the
/// min boundary, for the same reason.
///
/// MUTATION COVERAGE: `length > max` → `length >= max` admits the bin and reds this test.
#[tokio::test]
async fn test_candidate_filter_keeps_file_at_exactly_max_file_size() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    let pd_a = write_sized_pos_delete(&table, &x_path, 100, 5).await;
    let size_a = pd_a.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_a]).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    let size_b = pd_b.file_size_in_bytes;
    let table = add_deletes(&catalog, &table, vec![pd_b]).await;

    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size_a - 2)
            .target_file_size_bytes(size_a - 1)
            .max_file_size_bytes(size_a)
    };
    let config = action().resolve_config().expect("legal knobs");

    assert_eq!(
        size_a, config.max_file_size_bytes,
        "fixture: A sits EXACTLY on the resolved max"
    );
    assert!(
        size_b < config.min_file_size_bytes,
        "fixture: B is STRICTLY sub-min (measured {size_b}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        size_a + size_b > config.target_file_size_bytes,
        "fixture NON-VACUITY: were A wrongly a candidate, the two-file bin WOULD clear the target"
    );
    assert!(
        size_b <= config.max_file_size_bytes,
        "fixture: B's own bin must not clear too_much_content"
    );
    assert!(2 < config.min_input_files, "fixture: below the count floor");

    let before = scan_y_values(&table).await;
    let snapshot_before = table.metadata().current_snapshot_id();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result,
        RewritePositionDeleteFilesResult::default(),
        "a file EXACTLY at max is not a candidate, so only B is packed — and a bin of one is declined"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().current_snapshot_id(),
        snapshot_before,
        "declined ⇒ no snapshot"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 2);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-005 — the pipeline ORDER (S4 strictly before S5) and the filter STAGE.
// ------------------------------------------------------------------------------------------------

/// C-005's ORDER pin (C-036 recipe 5), discriminating BY BIN COUNT. One partition, THREE files in
/// manifest order A, X, B: A and B sub-min candidates, X strictly in range. The knobs come from the
/// measurements — `min := S_X - 1`, `target := S_X`, `max := S_X + 1`, and a group size
/// `W := S_A + S_B` that A and B exactly fill.
///
/// FILTER-THEN-PACK (Java's order, and this port's): X is dropped at S4, so A and B pack into ONE
/// bin whose input clears the target ⇒ one Replace snapshot, two files rewritten.
///
/// PACK-THEN-FILTER (the mutant): X is still present at packing, and since `S_A + S_X > W` and
/// `S_X + S_B > W`, the greedy lookback-1 packer emits the three SINGLETONS [A], [X], [B]. Dropping X
/// afterwards leaves two bins of one, both declined by the `size > 1` guards ⇒ ZERO snapshots.
///
/// RECORDED AS TRUE BY CONSTRUCTION, not asserted as falsifiable: `min < S_X < max` and
/// `S_A + S_B <= W` are identities under the knob choice. The falsifiable preconditions are the
/// ones that make the mutant emit singletons (`S_X > S_A`, `S_X > S_B`), the ones that make A and B
/// candidates while X is not, the MANIFEST ORDER, and `S_A + S_B > target`.
#[tokio::test]
async fn test_candidate_filter_runs_before_packing() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // ONE commit, so the three entries land in ONE manifest in this order: A, X, B.
    let pd_a = write_sized_pos_delete(&table, &x_path, 1, 1).await; // masks y=20
    let pd_x = write_sized_pos_delete(&table, &x_path, 100, 40).await; // masks nothing live
    let pd_b = write_sized_pos_delete(&table, &x_path, 3, 1).await; // masks y=40
    let (size_a, size_x, size_b) = (
        pd_a.file_size_in_bytes,
        pd_x.file_size_in_bytes,
        pd_b.file_size_in_bytes,
    );
    let (path_a, path_x, path_b) = (
        pd_a.file_path().to_string(),
        pd_x.file_path().to_string(),
        pd_b.file_path().to_string(),
    );
    let table = add_deletes(&catalog, &table, vec![pd_a, pd_x, pd_b]).await;

    assert_eq!(
        live_pos_delete_paths(&table).await,
        vec![path_a, path_x, path_b],
        "fixture: the manifest order must be A, X, B — the packing the mutant produces depends on it"
    );

    let group_size = size_a + size_b;
    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_file_size_bytes(size_x - 1)
            .target_file_size_bytes(size_x)
            .max_file_size_bytes(size_x + 1)
            .max_file_group_size_bytes(group_size)
    };
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS.
    assert!(
        size_a < config.min_file_size_bytes && size_b < config.min_file_size_bytes,
        "fixture: A and B are SUB-MIN candidates (measured {size_a}/{size_b}, resolved min {})",
        config.min_file_size_bytes
    );
    assert!(
        size_x >= config.min_file_size_bytes && size_x <= config.max_file_size_bytes,
        "fixture: X is strictly IN RANGE, so S4 must drop it"
    );
    assert!(
        size_x > size_a && size_x > size_b,
        "fixture: X must be larger than each of A and B, or the pack-then-filter mutant would NOT \
         emit singletons and the pin would not discriminate (measured A={size_a}, X={size_x}, B={size_b})"
    );
    assert!(
        size_a + size_b > config.target_file_size_bytes,
        "fixture: the A+B bin must clear the target so the correct order ADMITS it"
    );
    assert!(
        2 < config.min_input_files,
        "fixture NON-VACUITY: two files are below the count floor, so enough_content is the only \
         admitting clause"
    );

    let before = scan_y_values(&table).await;
    assert_eq!(
        before,
        HashSet::from([10, 30, 50]),
        "before: A masks y=20, B masks y=40"
    );
    let history_before = table.metadata().history().len();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 2,
        "filter-then-pack: A and B form ONE admitted bin; the in-range X never reaches the packer"
    );
    assert_eq!(result.added_delete_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 1,
        "exactly ONE Replace snapshot — the pack-then-filter mutant commits none"
    );
    assert_eq!(
        count_pos(&live_delete_files(&reloaded).await),
        2,
        "the compacted A+B file plus the untouched X"
    );
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

/// C-005's FILTER-STAGE pin for the hoist's NAMED CONSEQUENCE (behaviour flip 2). The user `filter`
/// is now bound ONCE, right after the no-snapshot early return and BEFORE the manifest walk, exactly
/// where Java binds it (at the `PositionDeletes` scan). So an UNBINDABLE filter fails on ANY table
/// with a current snapshot — here one that has no position-delete files at all, and therefore no
/// group that could ever be admitted.
///
/// Under the pre-hoist, per-group binding this same call returned `Ok` with zero counts, because the
/// bind only ran inside a group that already had two files. The flip is deliberate: a filter the
/// engine cannot bind is a caller error, and failing loudly beats silently compacting nothing.
#[tokio::test]
async fn test_unbindable_filter_errors_even_when_no_group_is_admissible() {
    use crate::expr::Reference;
    use crate::spec::Datum;

    let (catalog, _temp, table, _x_path) = gate_table().await;
    assert!(
        table.metadata().current_snapshot().is_some(),
        "fixture: the table HAS a current snapshot (the data-file append)"
    );
    assert!(
        live_delete_files(&table).await.is_empty(),
        "fixture: and NO delete files, so no group is admissible under any gate"
    );

    let error = RewritePositionDeleteFiles::new(table.clone())
        .filter(Reference::new("no_such_column").equal_to(Datum::long(0)))
        .execute(&catalog)
        .await
        .expect_err("an unbindable filter must fail at the bind, not be silently ignored");
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error
            .message()
            .contains("filter could not be bound to the table schema"),
        "unexpected message: {}",
        error.message()
    );
}

// ------------------------------------------------------------------------------------------------
// C-027 — the SHARED bin packer, reused through a weight closure.
// ------------------------------------------------------------------------------------------------

/// C-027 (C-036 recipe 7's fixture shape). `max_file_group_size_bytes` splits ONE partition into TWO
/// bins, and each admitted bin commits its own Replace snapshot — which is only possible if the
/// position-delete planner really goes through the shared `pack_bins` with the delete file's size as
/// its weight.
///
/// KNOBS FROM THE MEASUREMENTS: four files A, B, C, D in manifest order, all sub-min (so all four are
/// candidates), `.min_input_files(2)`, and `W := max(S_A + S_B, S_C + S_D)` with `W < S_A + S_B + S_C`
/// asserted — under the greedy forward first-fit that yields exactly {A, B} and {C, D}.
///
/// MUTATION COVERAGE: pass a constant weight (or `u64::MAX` as the group size) and the four files
/// pack into ONE bin — one snapshot, one added file, and both assertions red. Reverse the packer's
/// emission order and the manifest-order precondition reds.
#[tokio::test]
async fn test_admission_max_file_group_size_splits_partition_into_bins() {
    let (catalog, _temp, table, x_path) = gate_table().await;

    // Distinct sizes, so the packing genuinely depends on the weights and on the manifest order.
    let pd_a = write_sized_pos_delete(&table, &x_path, 100, 1).await;
    let pd_b = write_sized_pos_delete(&table, &x_path, 200, 2).await;
    let pd_c = write_sized_pos_delete(&table, &x_path, 300, 3).await;
    let pd_d = write_sized_pos_delete(&table, &x_path, 400, 4).await;
    let sizes = [
        pd_a.file_size_in_bytes,
        pd_b.file_size_in_bytes,
        pd_c.file_size_in_bytes,
        pd_d.file_size_in_bytes,
    ];
    let paths: Vec<String> = [&pd_a, &pd_b, &pd_c, &pd_d]
        .iter()
        .map(|f| f.file_path().to_string())
        .collect();
    let table = add_deletes(&catalog, &table, vec![pd_a, pd_b, pd_c, pd_d]).await;

    assert_eq!(
        live_pos_delete_paths(&table).await,
        paths,
        "fixture: the manifest order must be A, B, C, D — the bin boundaries depend on it"
    );

    let group_size = (sizes[0] + sizes[1]).max(sizes[2] + sizes[3]);
    let action = || {
        RewritePositionDeleteFiles::new(table.clone())
            .min_input_files(2)
            .min_file_size_bytes(100_000)
            .target_file_size_bytes(200_000)
            .max_file_size_bytes(400_000)
            .max_file_group_size_bytes(group_size)
    };
    let config = action().resolve_config().expect("legal knobs");

    // PRECONDITIONS, all over MEASURED sizes.
    for size in &sizes {
        assert!(
            *size < config.min_file_size_bytes,
            "fixture: every file must be SUB-MIN so all four are candidates \
             (measured {size}, resolved min {})",
            config.min_file_size_bytes
        );
    }
    assert!(
        2 >= config.min_input_files,
        "fixture: a bin of two must clear the count floor"
    );
    assert!(
        sizes[0] + sizes[1] <= config.max_file_group_size_bytes,
        "fixture: A and B must fit ONE bin"
    );
    assert!(
        config.max_file_group_size_bytes < sizes[0] + sizes[1] + sizes[2],
        "fixture: C must NOT fit alongside A and B — this is what forces the split"
    );
    assert!(
        sizes[2] + sizes[3] <= config.max_file_group_size_bytes,
        "fixture: C and D must fit ONE bin"
    );

    let before = scan_y_values(&table).await;
    let history_before = table.metadata().history().len();

    let result = action().execute(&catalog).await.unwrap();
    assert_eq!(
        result.rewritten_delete_files_count, 4,
        "all four candidates are rewritten, but across TWO bins"
    );
    assert_eq!(
        result.added_delete_files_count, 2,
        "one compacted file per BIN — a single bin would add one"
    );

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    assert_eq!(
        reloaded.metadata().history().len(),
        history_before + 2,
        "two admitted bins ⇒ two Replace snapshots"
    );
    assert_eq!(count_pos(&live_delete_files(&reloaded).await), 2);
    assert_eq!(scan_y_values(&reloaded).await, before, "read identity");
}

// ------------------------------------------------------------------------------------------------
// C-041 — the group input-size sum SATURATES where Java's `long` sum wraps.
// ------------------------------------------------------------------------------------------------

/// C-041. The bin input-size sum is `saturating_add`, so an overflowing bin resolves to `u64::MAX`
/// and stays ADMITTED. Java's `inputSize` is `stream().mapToLong(..).sum()`, which WRAPS to a small
/// (or negative) value and would DECLINE — a named, recorded divergence in the safe direction.
///
/// WHITE-BOX on `group_qualifies` with two fabricated entries, because an end-to-end fixture would
/// then try to READ 16 EiB of position deletes. The thresholds are chosen so the two arithmetics
/// disagree on the OUTCOME: saturating gives `u64::MAX > target`, while wrapping gives
/// `u64::MAX + 10 = 9`, which clears neither the target nor the max, and the count floor is above
/// the bin size either way.
///
/// MUTATION COVERAGE: change `saturating_add` to `wrapping_add` in `group_qualifies` and the bin is
/// declined.
#[tokio::test]
async fn test_group_input_size_saturates_not_wraps() {
    let (_catalog, _temp, table, x_path) = gate_table().await;

    let mut huge = write_sized_pos_delete(&table, &x_path, 1, 1).await;
    huge.file_size_in_bytes = u64::MAX;
    let mut small = write_sized_pos_delete(&table, &x_path, 3, 1).await;
    small.file_size_in_bytes = 10;

    let bin = vec![
        LiveDeleteEntry {
            data_file: huge,
            sequence_number: 1,
        },
        LiveDeleteEntry {
            data_file: small,
            sequence_number: 1,
        },
    ];
    let config = ResolvedConfig {
        target_file_size_bytes: 100,
        min_file_size_bytes: 50,
        max_file_size_bytes: 1_000,
        // Above the bin size, so `enough_input_files` cannot be the admitter under EITHER arithmetic.
        min_input_files: 10,
        max_file_group_size_bytes: 1_000_000,
    };

    assert!(
        group_qualifies(&bin, &config),
        "u64::MAX + 10 must SATURATE to u64::MAX (admitted via enough_content), not WRAP to 9 \
         (which clears neither the target 100 nor the max 1000, and would be declined)"
    );
}

// ------------------------------------------------------------------------------------------------
// C-003 elements 3 and 5b — the two gate leaves that are unreachable END TO END, pinned WHITE BOX.
//
// C-003 recorded both as "UNKILLABLE, never claimed pinned". That scoping is now SUPERSEDED, in the
// fork's favour: both proofs rest on candidate-filter REACHABILITY (a bin in those states cannot be
// produced by `plan_bins`, because its member would not have been a candidate), not on the config
// space, and this module already opens a white-box seam on `group_qualifies`. Through that seam the
// states ARE constructible and both mutants DO die. The end-to-end unreachability argument stays
// true and stays recorded on `group_qualifies`; what changes is that neither leaf is now unpinned.
// ------------------------------------------------------------------------------------------------

/// A [`LiveDeleteEntry`] whose `file_size_in_bytes` is SET to `size` — the same white-box seam
/// `test_group_input_size_saturates_not_wraps` uses. The underlying file is real but is never read:
/// `group_qualifies` reads nothing off the entry except its size.
async fn entry_of_size(table: &Table, target_path: &str, size: u64) -> LiveDeleteEntry {
    let mut data_file = write_sized_pos_delete(table, target_path, 1, 1).await;
    data_file.file_size_in_bytes = size;
    LiveDeleteEntry {
        data_file,
        sequence_number: 1,
    }
}

/// The thresholds both white-box gate pins run against: `min < target < max` (C-006 preconditions
/// (3) and (4)), with `min_input_files` well above the bin size so `enough_input_files` cannot be
/// the admitter under any of the mutants below.
fn white_box_gate_config() -> ResolvedConfig {
    ResolvedConfig {
        target_file_size_bytes: 100,
        min_file_size_bytes: 50,
        max_file_size_bytes: 1_000,
        min_input_files: 10,
        max_file_group_size_bytes: 1_000_000,
    }
}

/// C-003 element 5b — `too_much_content`'s BOUNDARY STRICTNESS. A bin whose input size is EXACTLY
/// `max_file_size_bytes` is DECLINED, because the comparison is `input_size > max`, strict.
///
/// END-TO-END UNREACHABLE, and that is why C-003 recorded it as unkillable: such a bin is either of
/// size 1 — and then its file's length equals max, so `outsideDesiredFileSizeRange` never made it a
/// candidate and it never reaches the gate — or of size >= 2, and then `enough_content` is already
/// true because `max > target`. WHITE BOX the state is trivially constructible, so the mutant dies.
///
/// MUTATION COVERAGE: `input_size > max` → `>=` admits this bin.
#[tokio::test]
async fn test_gate_input_size_exactly_max_is_declined_white_box() {
    let (_catalog, _temp, table, x_path) = gate_table().await;
    let config = white_box_gate_config();

    let bin = vec![entry_of_size(&table, &x_path, config.max_file_size_bytes).await];

    assert!(
        !group_qualifies(&bin, &config),
        "an input size EXACTLY at max is declined — too_much_content is `input_size > max`, STRICT"
    );
}

/// C-003 element 3 — `enough_content`'s `size > 1` conjunct. A LONE file whose length is above the
/// target but at or below the max is DECLINED: `enough_content` is `size > 1 && input_size > target`,
/// and the `size > 1` conjunct is the only thing that declines it.
///
/// END-TO-END UNREACHABLE, and that is why C-003 recorded it as unkillable: a bin of one that
/// reaches the gate must be a candidate, so its length is either below min — and `min < target`, so
/// `enough_content` is false either way — or above max, in which case `too_much_content` is already
/// true and carries the admission. WHITE BOX the in-band lone file is constructible, so the mutant
/// dies.
///
/// MUTATION COVERAGE: drop the `size > 1` conjunct from `enough_content` and this bin is admitted.
#[tokio::test]
async fn test_gate_enough_content_size_guard_declines_lone_over_target_file_white_box() {
    let (_catalog, _temp, table, x_path) = gate_table().await;
    let config = white_box_gate_config();

    let size = 500;
    assert!(
        size > config.target_file_size_bytes && size < config.max_file_size_bytes,
        "fixture: the lone file is strictly BETWEEN target and max, so enough_content's size guard \
         is the ONLY clause declining it"
    );
    let bin = vec![entry_of_size(&table, &x_path, size).await];

    assert!(
        !group_qualifies(&bin, &config),
        "a LONE file above the target is declined by enough_content's `size > 1` conjunct"
    );
}
