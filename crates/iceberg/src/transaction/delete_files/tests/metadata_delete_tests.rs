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

use super::{append_files, live_file_paths};
use crate::expr::{Predicate, Reference};
use crate::memory::tests::new_memory_catalog;
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, FormatVersion, Literal,
    NestedField, Operation, PrimitiveType, Schema, Struct, Transform, Type, UnboundPartitionSpec,
};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::transform::create_transform_function;
use crate::{Catalog, TableCreation, TableIdent};

fn oracle_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::optional(2, "cat", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "v", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("the oracle schema builds")
}

async fn make_oracle_table_in_catalog(
    catalog: &impl Catalog,
    format_version: FormatVersion,
    spec: Option<UnboundPartitionSpec>,
) -> Table {
    let table_ident =
        TableIdent::from_strs([format!("ns1-{}", uuid::Uuid::new_v4()), "test1".to_string()])
            .expect("table ident parses");
    catalog
        .create_namespace(table_ident.namespace(), HashMap::new())
        .await
        .expect("namespace creates");

    let builder = TableCreation::builder()
        .name(table_ident.name().to_string())
        .schema(oracle_schema())
        .format_version(format_version);
    let creation = match spec {
        Some(spec) => builder.partition_spec(spec).build(),
        None => builder.build(),
    };
    catalog
        .create_table(table_ident.namespace(), creation)
        .await
        .expect("table creates")
}

#[allow(clippy::too_many_arguments)]
fn oracle_data_file(
    path: &str,
    spec_id: i32,
    partition: Struct,
    record_count: u64,
    value_counts: HashMap<i32, u64>,
    null_counts: HashMap<i32, u64>,
    lower_bounds: HashMap<i32, Datum>,
    upper_bounds: HashMap<i32, Datum>,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(record_count)
        .partition_spec_id(spec_id)
        .partition(partition)
        .value_counts(value_counts)
        .null_value_counts(null_counts)
        .nan_value_counts(HashMap::new())
        .lower_bounds(lower_bounds)
        .upper_bounds(upper_bounds)
        .build()
        .expect("data file builds")
}

fn counted_file(
    path: &str,
    spec_id: i32,
    partition: Struct,
    record_count: u64,
    null_counts: &[(i32, u64)],
    bounds: &[(i32, Datum, Datum)],
) -> DataFile {
    let mut value_counts = HashMap::new();
    for (field_id, _) in null_counts {
        value_counts.insert(*field_id, record_count);
    }
    for (field_id, _, _) in bounds {
        value_counts.insert(*field_id, record_count);
    }
    let lower_bounds = bounds
        .iter()
        .map(|(field_id, lower, _)| (*field_id, lower.clone()))
        .collect();
    let upper_bounds = bounds
        .iter()
        .map(|(field_id, _, upper)| (*field_id, upper.clone()))
        .collect();
    oracle_data_file(
        path,
        spec_id,
        partition,
        record_count,
        value_counts,
        null_counts.iter().copied().collect(),
        lower_bounds,
        upper_bounds,
    )
}

fn oracle_file1() -> DataFile {
    counted_file(
        "test/file1.parquet",
        0,
        Struct::empty(),
        3,
        &[(1, 0), (2, 0), (3, 0)],
        &[
            (1, Datum::int(1), Datum::int(3)),
            (2, Datum::string("x"), Datum::string("y")),
            (3, Datum::string("a"), Datum::string("c")),
        ],
    )
}

fn oracle_file2() -> DataFile {
    counted_file(
        "test/file2.parquet",
        0,
        Struct::empty(),
        1,
        &[(1, 0), (2, 0), (3, 0)],
        &[
            (1, Datum::int(7), Datum::int(7)),
            (2, Datum::string("g"), Datum::string("g")),
            (3, Datum::string("x"), Datum::string("x")),
        ],
    )
}

fn oracle_file3() -> DataFile {
    counted_file(
        "test/file3.parquet",
        0,
        Struct::empty(),
        2,
        &[(1, 0), (2, 0), (3, 0)],
        &[
            (1, Datum::int(8), Datum::int(9)),
            (2, Datum::string("g"), Datum::string("g")),
            (3, Datum::string("y"), Datum::string("z")),
        ],
    )
}

fn partitioned_oracle_file(
    path: &str,
    spec_id: i32,
    partition_literals: Vec<Option<Literal>>,
    record_count: u64,
    bounds: &[(i32, Datum, Datum)],
) -> DataFile {
    counted_file(
        path,
        spec_id,
        Struct::from_iter(partition_literals),
        record_count,
        &[(1, 0), (2, 0), (3, 0)],
        bounds,
    )
}

fn synthetic_position_delete(path: &str, spec_id: i32, partition: Struct) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(10)
        .record_count(1)
        .partition_spec_id(spec_id)
        .partition(partition)
        .build()
        .expect("position delete file builds")
}

fn synthetic_deletion_vector(
    path: &str,
    spec_id: i32,
    partition: Struct,
    referenced_data_file: &str,
) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Puffin)
        .file_size_in_bytes(10)
        .record_count(1)
        .partition_spec_id(spec_id)
        .partition(partition)
        .referenced_data_file(Some(referenced_data_file.to_string()))
        .content_offset(Some(4))
        .content_size_in_bytes(Some(40))
        .build()
        .expect("deletion vector builds")
}

async fn add_deletes(catalog: &impl Catalog, table: &Table, deletes: Vec<DataFile>) -> Table {
    let tx = Transaction::new(table);
    let action = tx.row_delta().add_deletes(deletes);
    let tx = action.apply(tx).expect("row_delta applies");
    tx.commit(catalog).await.expect("row_delta commits")
}

async fn assert_decision(
    table: &Table,
    predicate: &Predicate,
    branch: Option<&str>,
    expected: bool,
    why: &str,
) {
    let decision = table
        .can_delete_using_metadata(predicate, branch, true)
        .await
        .expect("can_delete_using_metadata resolves");
    assert_eq!(decision, expected, "{why}");
}

async fn delete_summary(
    catalog: &impl Catalog,
    table: &Table,
    predicate: Predicate,
) -> (Table, crate::spec::Summary) {
    let tx = Transaction::new(table);
    let action = tx.delete_files().delete_from_row_filter(predicate);
    let tx = action.apply(tx).expect("delete_from_row_filter applies");
    let table = tx.commit(catalog).await.expect("delete commits");
    let summary = table
        .metadata()
        .current_snapshot()
        .expect("delete snapshot exists")
        .summary()
        .clone();
    (table, summary)
}

fn summary_prop(summary: &crate::spec::Summary, key: &str) -> Option<String> {
    summary.additional_properties.get(key).cloned()
}

fn assert_delete_summary(
    summary: &crate::spec::Summary,
    deleted_data_files: &str,
    deleted_records: &str,
) {
    assert_eq!(
        summary.operation,
        Operation::Delete,
        "a metadata-only delete commits op `delete`, got {:?}",
        summary.operation
    );
    assert_eq!(
        summary_prop(summary, "deleted-data-files").as_deref(),
        Some(deleted_data_files),
        "deleted-data-files"
    );
    assert_eq!(
        summary_prop(summary, "deleted-records").as_deref(),
        Some(deleted_records),
        "deleted-records"
    );
}

#[tokio::test]
async fn can_delete_using_metadata_whole_one_file() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    let predicate = Reference::new("id").equal_to(Datum::int(7));
    assert_decision(&table, &predicate, None, true,
        "id=7: file1 [1,3] is pruned, file2 [7,7] is wholly matched — the oracle commits a metadata delete")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "1");
}

#[tokio::test]
async fn can_delete_using_metadata_whole_two_files() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![
        oracle_file1(),
        oracle_file2(),
        oracle_file3(),
    ])
    .await;

    let predicate = Reference::new("id").greater_than_or_equal_to(Datum::int(7));
    assert_decision(&table, &predicate, None, true,
        "id>=7: file1 is pruned, file2 and file3 are wholly matched — the oracle commits a metadata delete")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "2", "3");
}

#[tokio::test]
async fn can_delete_using_metadata_mixed_partial_and_whole() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    assert_decision(
        &table,
        &Reference::new("id").greater_than_or_equal_to(Datum::int(3)),
        None,
        false,
        "id>=3: file1 [1,3] stays a candidate but is only partially matched — the oracle is NOT metadata",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_partial_only() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    assert_decision(
        &table,
        &Reference::new("id").equal_to(Datum::int(2)),
        None,
        false,
        "id=2: file1 [1,3] is a partial candidate — the oracle is NOT metadata",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_all_rows_true_and_no_predicate() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    assert_decision(&table, &Predicate::AlwaysTrue, None, true,
        "always-true (a DELETE with no WHERE reaches the same predicate): every file is wholly matched")
        .await;

    let (table, summary) = delete_summary(&catalog, &table, Predicate::AlwaysTrue).await;
    assert_delete_summary(&summary, "2", "4");
    assert!(
        live_file_paths(&table).await.is_empty(),
        "the whole-table metadata delete removed every file"
    );
}

#[tokio::test]
async fn can_delete_using_metadata_no_match_is_vacuously_true() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    assert_decision(
        &table,
        &Reference::new("id").equal_to(Datum::int(99)),
        None,
        true,
        "id=99 plans zero files; Java's all(tasks) is vacuously true — the oracle commits an empty delete snapshot",
    )
    .await;

    let tx = Transaction::new(&table);
    let action = tx
        .delete_files()
        .delete_from_row_filter(Reference::new("id").equal_to(Datum::int(99)));
    let tx = action.apply(tx).expect("delete_from_row_filter applies");
    let result = tx.commit(&catalog).await;
    assert!(
        result.is_err(),
        "the fork's DeleteFiles today rejects the empty by-filter commit; the oracle's Spark commits an empty `delete` snapshot — a named divergence"
    );
}

#[tokio::test]
async fn can_delete_using_metadata_string_eq_whole() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    let predicate = Reference::new("cat").equal_to(Datum::string("g"));
    assert_decision(&table, &predicate, None, true,
        "cat='g': file1 [x,y] is pruned, file2 [g,g] is wholly matched — the oracle commits a metadata delete")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "1");
}

#[tokio::test]
async fn can_delete_using_metadata_partition_select_identity() {
    let catalog = new_memory_catalog().await;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "cat", Transform::Identity)
        .expect("spec builds")
        .build();
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, Some(spec)).await;
    let table = append_files(&catalog, &table, vec![
        partitioned_oracle_file(
            "test/file1.parquet",
            0,
            vec![Some(Literal::string("x"))],
            3,
            &[
                (1, Datum::int(1), Datum::int(3)),
                (2, Datum::string("x"), Datum::string("x")),
                (3, Datum::string("a"), Datum::string("c")),
            ],
        ),
        partitioned_oracle_file(
            "test/file2.parquet",
            0,
            vec![Some(Literal::string("g"))],
            1,
            &[
                (1, Datum::int(7), Datum::int(7)),
                (2, Datum::string("g"), Datum::string("g")),
                (3, Datum::string("x"), Datum::string("x")),
            ],
        ),
    ])
    .await;

    let predicate = Reference::new("cat").equal_to(Datum::string("x"));
    assert_decision(&table, &predicate, None, true,
        "cat='x' selects whole partitions under identity(cat) — selectsPartitions short-circuits true")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "3");
}

#[tokio::test]
async fn can_delete_using_metadata_partition_select_bucket() {
    let catalog = new_memory_catalog().await;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(1, "id_bucket", Transform::Bucket(4))
        .expect("spec builds")
        .build();
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, Some(spec)).await;

    let bucket = |id: i32| {
        let transform =
            create_transform_function(&Transform::Bucket(4)).expect("bucket transform resolves");
        let datum = transform
            .transform_literal_result(&Datum::int(id))
            .expect("bucket of a literal resolves");
        Literal::from(datum)
    };

    let table = append_files(&catalog, &table, vec![
        partitioned_oracle_file("test/file1.parquet", 0, vec![Some(bucket(3))], 1, &[
            (1, Datum::int(3), Datum::int(3)),
            (2, Datum::string("x"), Datum::string("x")),
            (3, Datum::string("a"), Datum::string("a")),
        ]),
        partitioned_oracle_file("test/file2.parquet", 0, vec![Some(bucket(7))], 1, &[
            (1, Datum::int(7), Datum::int(7)),
            (2, Datum::string("g"), Datum::string("g")),
            (3, Datum::string("x"), Datum::string("x")),
        ]),
    ])
    .await;

    let predicate = Reference::new("id").equal_to(Datum::int(7));
    assert_decision(&table, &predicate, None, true,
        "id=7 under bucket(4,id): file1's bucket is pruned, file2 [7,7] is wholly matched — the oracle commits a metadata delete")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "1");
}

#[tokio::test]
async fn can_delete_using_metadata_partition_plus_metrics() {
    let catalog = new_memory_catalog().await;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "cat", Transform::Identity)
        .expect("spec builds")
        .build();
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, Some(spec)).await;
    let table = append_files(&catalog, &table, vec![
        partitioned_oracle_file(
            "test/file1.parquet",
            0,
            vec![Some(Literal::string("x"))],
            2,
            &[
                (1, Datum::int(1), Datum::int(2)),
                (2, Datum::string("x"), Datum::string("x")),
                (3, Datum::string("a"), Datum::string("b")),
            ],
        ),
        partitioned_oracle_file(
            "test/file2.parquet",
            0,
            vec![Some(Literal::string("g"))],
            1,
            &[
                (1, Datum::int(7), Datum::int(7)),
                (2, Datum::string("g"), Datum::string("g")),
                (3, Datum::string("x"), Datum::string("x")),
            ],
        ),
    ])
    .await;

    let predicate = Reference::new("cat")
        .equal_to(Datum::string("x"))
        .and(Reference::new("id").less_than_or_equal_to(Datum::int(2)));
    assert_decision(&table, &predicate, None, true,
        "cat='x' AND id<=2: file2's partition is pruned; file1 proves by strict metrics on both conjuncts")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "2");
}

#[tokio::test]
async fn can_delete_using_metadata_partition_partial() {
    let catalog = new_memory_catalog().await;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "cat", Transform::Identity)
        .expect("spec builds")
        .build();
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, Some(spec)).await;
    let table = append_files(&catalog, &table, vec![
        partitioned_oracle_file(
            "test/file1.parquet",
            0,
            vec![Some(Literal::string("x"))],
            2,
            &[
                (1, Datum::int(1), Datum::int(2)),
                (2, Datum::string("x"), Datum::string("x")),
                (3, Datum::string("a"), Datum::string("b")),
            ],
        ),
        partitioned_oracle_file(
            "test/file2.parquet",
            0,
            vec![Some(Literal::string("g"))],
            1,
            &[
                (1, Datum::int(7), Datum::int(7)),
                (2, Datum::string("g"), Datum::string("g")),
                (3, Datum::string("x"), Datum::string("x")),
            ],
        ),
    ])
    .await;

    assert_decision(
        &table,
        &Reference::new("cat")
            .equal_to(Datum::string("x"))
            .and(Reference::new("id").equal_to(Datum::int(1))),
        None,
        false,
        "cat='x' AND id=1: file1's id [1,2] is only partially covered — the oracle is NOT metadata",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_partition_arm_only_proof() {
    let catalog = new_memory_catalog().await;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "cat", Transform::Identity)
        .expect("spec builds")
        .build();
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, Some(spec)).await;

    let file = oracle_data_file(
        "test/file1.parquet",
        0,
        Struct::from_iter([Some(Literal::string("x"))]),
        3,
        HashMap::from([(1, 3), (3, 3)]),
        HashMap::from([(1, 0), (3, 0)]),
        HashMap::from([(1, Datum::int(1)), (3, Datum::string("a"))]),
        HashMap::from([(1, Datum::int(3)), (3, Datum::string("c"))]),
    );
    let table = append_files(&catalog, &table, vec![file]).await;

    let predicate = Reference::new("cat")
        .equal_to(Datum::string("x"))
        .or(Reference::new("id").equal_to(Datum::int(99)));
    assert_decision(&table, &predicate, None, true,
        "cat='x' OR id=99: strict projection proves the partition arm while the metrics arm cannot (cat bounds absent) — dropping the partition arm turns this pin red")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "3");
}

#[tokio::test]
async fn can_delete_using_metadata_prior_deletes_then_whole_v2() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;
    let table = add_deletes(&catalog, &table, vec![synthetic_position_delete(
        "test/pos-del-1.parquet",
        0,
        Struct::empty(),
    )])
    .await;

    let predicate = Reference::new("id").less_than_or_equal_to(Datum::int(3));
    assert_decision(&table, &predicate, None, true,
        "id<=3: file1 is wholly matched despite its attached position delete — delete files do not enter the decision")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "3");
    assert_eq!(
        summary_prop(&summary, "total-delete-files").as_deref(),
        Some("1"),
        "v2 keeps file1's position delete file, matching the oracle"
    );
}

#[tokio::test]
async fn can_delete_using_metadata_prior_deletes_then_whole_v3() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V3, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;
    let table = add_deletes(&catalog, &table, vec![synthetic_deletion_vector(
        "test/dv-1.puffin",
        0,
        Struct::empty(),
        "test/file1.parquet",
    )])
    .await;

    let predicate = Reference::new("id").less_than_or_equal_to(Datum::int(3));
    assert_decision(
        &table,
        &predicate,
        None,
        true,
        "id<=3 on a v3 table: file1 is wholly matched despite its attached deletion vector",
    )
    .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "3");
    assert_eq!(
        summary_prop(&summary, "removed-dvs").as_deref(),
        Some("1"),
        "v3 drops file1's deletion vector with the file, matching the oracle"
    );
}

#[tokio::test]
async fn can_delete_using_metadata_prior_deletes_then_rest() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;
    let table = add_deletes(&catalog, &table, vec![synthetic_position_delete(
        "test/pos-del-1.parquet",
        0,
        Struct::empty(),
    )])
    .await;

    assert_decision(
        &table,
        &Reference::new("id").is_in(vec![Datum::int(2), Datum::int(3)]),
        None,
        false,
        "id IN (2,3): file1 [1,3] is a candidate but strict IN requires a single-valued bound — the oracle is NOT metadata",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_prior_deletes_then_rest_cow_shape() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let rewritten_file1 = counted_file(
        "test/file1.parquet",
        0,
        Struct::empty(),
        2,
        &[(1, 0), (2, 0), (3, 0)],
        &[
            (1, Datum::int(2), Datum::int(3)),
            (2, Datum::string("x"), Datum::string("y")),
            (3, Datum::string("b"), Datum::string("c")),
        ],
    );
    let table = append_files(&catalog, &table, vec![rewritten_file1, oracle_file2()]).await;

    assert_decision(
        &table,
        &Reference::new("id").is_in(vec![Datum::int(2), Datum::int(3)]),
        None,
        false,
        "id IN (2,3) on the rewritten file [2,3]: strict IN still requires lower==upper — the decision is false (the oracle's CoW `delete` op is a delete-only overwrite, not metadata)",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_is_null_whole() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;

    let all_null_v_file2 = oracle_data_file(
        "test/file2.parquet",
        0,
        Struct::empty(),
        1,
        HashMap::from([(1, 1), (2, 1), (3, 1)]),
        HashMap::from([(1, 0), (2, 0), (3, 1)]),
        HashMap::from([(1, Datum::int(7)), (2, Datum::string("g"))]),
        HashMap::from([(1, Datum::int(7)), (2, Datum::string("g"))]),
    );
    let table = append_files(&catalog, &table, vec![oracle_file1(), all_null_v_file2]).await;

    let predicate = Reference::new("v").is_null();
    assert_decision(&table, &predicate, None, true,
        "v IS NULL: file1 (v null-count 0) is pruned, file2 (all-null v) is wholly matched — the oracle commits a metadata delete")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "1");
}

#[tokio::test]
async fn can_delete_using_metadata_or_whole() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    let predicate = Reference::new("id")
        .equal_to(Datum::int(7))
        .or(Reference::new("id").equal_to(Datum::int(99)));
    assert_decision(&table, &predicate, None, true,
        "id=7 OR id=99: file1 is pruned, file2 proves via the id=7 disjunct — the oracle commits a metadata delete")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "1");
}

#[tokio::test]
async fn can_delete_using_metadata_not_in_whole() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    assert_decision(
        &table,
        &Reference::new("id").is_not_in(vec![Datum::int(1), Datum::int(2), Datum::int(3)]),
        None,
        false,
        "id NOT IN (1,2,3): file1 [1,3] is a candidate but strict NOT IN fails — the oracle is NOT metadata",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_nondeterministic_like() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    let predicate = Reference::new("v").starts_with(Datum::string("x"));
    assert_decision(&table, &predicate, None, true,
        "v LIKE 'x%' (startsWith): file1 [a,c] is pruned, file2 [x,x] wholly matches — the oracle commits a metadata delete")
        .await;

    let (_, summary) = delete_summary(&catalog, &table, predicate).await;
    assert_delete_summary(&summary, "1", "1");
}

#[tokio::test]
async fn can_delete_using_metadata_uses_the_named_ref() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1()]).await;

    let snapshot_id = table
        .metadata()
        .current_snapshot()
        .expect("seed snapshot")
        .snapshot_id();
    let tx = Transaction::new(&table);
    let action = tx.manage_snapshots().create_branch("b1", snapshot_id);
    let tx = action.apply(tx).expect("create_branch applies");
    let table = tx.commit(&catalog).await.expect("branch commits");

    let partial_file2 = counted_file(
        "test/file2.parquet",
        0,
        Struct::empty(),
        1,
        &[(1, 0), (2, 0), (3, 0)],
        &[
            (1, Datum::int(2), Datum::int(9)),
            (2, Datum::string("g"), Datum::string("g")),
            (3, Datum::string("x"), Datum::string("x")),
        ],
    );
    let table = append_files(&catalog, &table, vec![partial_file2]).await;

    let predicate = Reference::new("id").less_than_or_equal_to(Datum::int(3));
    assert_decision(
        &table,
        &predicate,
        Some("b1"),
        true,
        "id<=3 against branch b1 (file1 only): wholly matched — skipping use_ref turns this pin red",
    )
    .await;
    assert_decision(
        &table,
        &predicate,
        None,
        false,
        "id<=3 against main (file1 + partial file2 [2,9]): not provable — the ref changes the answer",
    )
    .await;
}

#[tokio::test]
async fn can_delete_using_metadata_case_sensitivity_binds_the_filter() {
    let catalog = new_memory_catalog().await;
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, None).await;
    let table = append_files(&catalog, &table, vec![oracle_file1(), oracle_file2()]).await;

    let predicate = Reference::new("id").equal_to(Datum::int(7));
    assert!(
        table
            .can_delete_using_metadata(&predicate, None, false)
            .await
            .expect("case-insensitive filter binding resolves `id`"),
        "id=7 with case_sensitive=false still proves file2"
    );

    let miscased = Reference::new("ID").equal_to(Datum::int(7));
    assert!(
        table
            .can_delete_using_metadata(&miscased, None, false)
            .await
            .is_err(),
        "ID=7 with case_sensitive=false: the scan filter binds, but Java's strict metrics and partition arms bind case-sensitively and reject — the fork mirrors the loud failure"
    );
    assert!(
        table
            .can_delete_using_metadata(&miscased, None, true)
            .await
            .is_err(),
        "ID=7 with case_sensitive=true rejects at the filter bind"
    );
}

#[tokio::test]
async fn can_delete_using_metadata_selects_partitions_requires_every_spec() {
    let catalog = new_memory_catalog().await;
    let spec = UnboundPartitionSpec::builder()
        .with_spec_id(0)
        .add_partition_field(2, "cat", Transform::Identity)
        .expect("spec builds")
        .build();
    let table = make_oracle_table_in_catalog(&catalog, FormatVersion::V2, Some(spec)).await;
    let table = append_files(&catalog, &table, vec![partitioned_oracle_file(
        "test/file1.parquet",
        0,
        vec![Some(Literal::string("x"))],
        2,
        &[
            (1, Datum::int(3), Datum::int(5)),
            (2, Datum::string("x"), Datum::string("x")),
            (3, Datum::string("a"), Datum::string("b")),
        ],
    )])
    .await;

    let tx = Transaction::new(&table);
    let action = tx.update_partition_spec().add_field("id");
    let tx = action.apply(tx).expect("update_partition_spec applies");
    let table = tx.commit(&catalog).await.expect("spec evolution commits");
    let spec1 = table.metadata().default_partition_spec_id();
    assert_ne!(spec1, 0, "the spec evolved to a second spec id");

    let table = append_files(&catalog, &table, vec![partitioned_oracle_file(
        "test/file2.parquet",
        spec1,
        vec![Some(Literal::string("g")), Some(Literal::int(7))],
        1,
        &[
            (1, Datum::int(7), Datum::int(7)),
            (2, Datum::string("g"), Datum::string("g")),
            (3, Datum::string("x"), Datum::string("x")),
        ],
    )])
    .await;

    assert_decision(
        &table,
        &Reference::new("id").equal_to(Datum::int(3)),
        None,
        false,
        "id=3 selects whole partitions under the DEFAULT spec ({cat,id}) but not under spec 0 ({cat}); checking every spec forces the scan path, where file1 [3,5] is an unproven candidate — a default-spec-only check wrongly answers true",
    )
    .await;
}
