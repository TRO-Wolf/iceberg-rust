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

use std::cmp::Ordering;

use crate::Catalog;
use crate::maintenance::RewriteStrategy;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::local_fs_catalog;
use crate::maintenance::rewrite_data_files_sort_harness::{
    OracleRow, OutputFile, concatenated_rows, indexes, oracle_batches, oracle_row, oracle_table,
    output_files,
};
use crate::maintenance::rewrite_data_files_sort_vectors::{
    SPARK_SORT_EXPLICIT, SPARK_SORT_EXPLICIT_MULTI, SPARK_SORT_EXPLICIT_OVER_TABLE_ORDER,
    SPARK_SORT_PARTITIONED, SPARK_SORT_TABLE_ORDER, SPARK_SORT_TABLE_ORDER_PARTITIONED,
    SPARK_SORT_TARGET_SMALL_FILES, SPARK_SORT_TRANSFORM_BUCKET, SPARK_SORT_TRANSFORM_DAYS,
    SPARK_SORT_TRANSFORM_TRUNC,
};
use crate::spec::{FormatVersion, NullOrder, SortDirection, SortField, SortOrder, Transform};
use crate::transaction::{ApplyTransactionAction, Transaction};

fn order_of(fields: Vec<SortField>) -> SortOrder {
    SortOrder {
        order_id: 1,
        fields,
    }
}

fn field(source_id: i32, transform: Transform, ascending: bool, nulls_first: bool) -> SortField {
    SortField {
        source_id,
        transform,
        direction: if ascending {
            SortDirection::Ascending
        } else {
            SortDirection::Descending
        },
        null_order: if nulls_first {
            NullOrder::First
        } else {
            NullOrder::Last
        },
    }
}

fn id_desc_nulls_last() -> SortOrder {
    order_of(vec![field(1, Transform::Identity, false, false)])
}

fn compare_id_desc_nulls_last(left: &OracleRow, right: &OracleRow) -> Ordering {
    match (left.id, right.id) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(left), Some(right)) => right.cmp(&left),
    }
}

fn compare_cat_asc_nulls_first_v_desc_nulls_last(left: &OracleRow, right: &OracleRow) -> Ordering {
    let cats = match (left.cat.as_ref(), right.cat.as_ref()) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => left.cmp(right),
    };
    cats.then_with(|| match (left.v, right.v) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(left), Some(right)) => right.total_cmp(&left),
    })
}

fn compare_s_asc_nulls_last_then_id(left: &OracleRow, right: &OracleRow) -> Ordering {
    let strings = match (left.s.as_ref(), right.s.as_ref()) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(left), Some(right)) => left.cmp(right),
    };
    strings.then_with(|| match (left.id, right.id) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => left.cmp(&right),
    })
}

fn compare_id_asc_nulls_first(left: &OracleRow, right: &OracleRow) -> Ordering {
    match (left.id, right.id) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => left.cmp(&right),
    }
}

fn rows_of(spark_order: &[i64]) -> Vec<OracleRow> {
    spark_order.iter().map(|index| oracle_row(*index)).collect()
}

fn assert_same_key_order(
    cell: &str,
    spark_order: &[i64],
    rewritten: &[OracleRow],
    compare: impl Fn(&OracleRow, &OracleRow) -> Ordering,
) {
    let expected = rows_of(spark_order);
    assert_eq!(
        expected.len(),
        rewritten.len(),
        "{cell}: the rewrite kept {} rows, Spark kept {}",
        rewritten.len(),
        expected.len()
    );
    for (position, (spark, fork)) in expected.iter().zip(rewritten.iter()).enumerate() {
        assert_eq!(
            compare(spark, fork),
            Ordering::Equal,
            "{cell}: row {position} sorts differently: Spark has {spark:?}, the rewrite has {fork:?}"
        );
    }
    let mut spark_rows = indexes(&expected);
    let mut fork_rows = indexes(rewritten);
    spark_rows.sort_unstable();
    fork_rows.sort_unstable();
    assert_eq!(spark_rows, fork_rows, "{cell}: the row multiset changed");
}

fn assert_non_decreasing(
    cell: &str,
    rows: &[OracleRow],
    compare: impl Fn(&OracleRow, &OracleRow) -> Ordering,
) {
    for window in rows.windows(2) {
        assert_ne!(
            compare(&window[0], &window[1]),
            Ordering::Greater,
            "{cell}: {:?} precedes {:?} but sorts after it",
            window[0],
            window[1]
        );
    }
}

fn stamps(files: &[OutputFile]) -> Vec<Option<i32>> {
    files.iter().map(|file| file.sort_order_id).collect()
}

#[tokio::test]
async fn sort_explicit_order_matches_the_spark_row_order_and_stamps_zero() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(&catalog, FormatVersion::V2, false, None).await;

    let result = RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(id_desc_nulls_last()))
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("sort rewrite");
    assert_eq!(result.rewritten_data_files_count, 4);

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    assert_eq!(stamps(&files), vec![Some(0); files.len()]);
    assert_same_key_order(
        "SORT-EXPLICIT",
        SPARK_SORT_EXPLICIT,
        &concatenated_rows(&files),
        compare_id_desc_nulls_last,
    );
}

#[tokio::test]
async fn sort_explicit_multi_key_matches_the_spark_row_order() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(&catalog, FormatVersion::V2, false, None).await;
    let order = order_of(vec![
        field(2, Transform::Identity, true, true),
        field(4, Transform::Identity, false, false),
    ]);

    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(order))
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("sort rewrite");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    assert_same_key_order(
        "SORT-EXPLICIT-MULTI",
        SPARK_SORT_EXPLICIT_MULTI,
        &concatenated_rows(&files),
        compare_cat_asc_nulls_first_v_desc_nulls_last,
    );
}

#[tokio::test]
async fn sort_by_table_order_uses_the_default_order_and_stamps_its_id() {
    let (catalog, _guard) = local_fs_catalog().await;
    let order = order_of(vec![
        field(5, Transform::Identity, true, false),
        field(1, Transform::Identity, true, true),
    ]);
    let table = oracle_table(&catalog, FormatVersion::V2, false, Some(order)).await;

    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::SortByTableOrder)
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("sort rewrite");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    assert_eq!(stamps(&files), vec![Some(1); files.len()]);
    assert_same_key_order(
        "SORT-TABLE-ORDER",
        SPARK_SORT_TABLE_ORDER,
        &concatenated_rows(&files),
        compare_s_asc_nulls_last_then_id,
    );
}

#[tokio::test]
async fn sort_explicit_order_overrides_the_table_order_and_stamps_zero() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table_order = order_of(vec![field(5, Transform::Identity, true, true)]);
    let table = oracle_table(&catalog, FormatVersion::V2, false, Some(table_order)).await;
    let explicit = order_of(vec![field(1, Transform::Identity, true, true)]);

    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(explicit))
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("sort rewrite");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    assert_eq!(stamps(&files), vec![Some(0); files.len()]);
    assert_same_key_order(
        "SORT-EXPLICIT-OVER-TABLE-ORDER",
        SPARK_SORT_EXPLICIT_OVER_TABLE_ORDER,
        &concatenated_rows(&files),
        compare_id_asc_nulls_first,
    );
}

#[tokio::test]
async fn sort_explicit_order_equal_to_the_table_order_stamps_the_table_order_id() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(
        &catalog,
        FormatVersion::V2,
        false,
        Some(id_desc_nulls_last()),
    )
    .await;

    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(id_desc_nulls_last()))
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("sort rewrite");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    assert_eq!(stamps(&files), vec![Some(1); files.len()]);
}

#[tokio::test]
async fn sort_explicit_order_equal_to_a_historical_table_order_stamps_that_order_id() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(
        &catalog,
        FormatVersion::V2,
        false,
        Some(id_desc_nulls_last()),
    )
    .await;
    let transaction = Transaction::new(&table);
    let action = transaction
        .replace_sort_order()
        .asc("s", NullOrder::First)
        .apply(transaction)
        .expect("replace sort order");
    let table = action.commit(&catalog).await.expect("commit sort order");
    assert_ne!(
        table.metadata().default_sort_order().order_id,
        1,
        "the default order must have moved on for this cell to mean anything"
    );

    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(id_desc_nulls_last()))
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("sort rewrite");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    assert_eq!(stamps(&files), vec![Some(1); files.len()]);
}

#[tokio::test]
async fn sort_by_table_order_on_an_unsorted_table_is_refused() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(&catalog, FormatVersion::V2, false, None).await;

    let error = RewriteDataFiles::new(table)
        .strategy(RewriteStrategy::SortByTableOrder)
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect_err("an unsorted table cannot sort");
    assert!(
        error
            .message()
            .contains("Cannot sort data without a valid sort order, table 'ns-")
            && error
                .message()
                .contains("is unsorted and no sort order is provided"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn sort_by_transformed_keys_matches_the_spark_row_order() {
    for (cell, order, expected) in [
        (
            "SORT-TRANSFORM-BUCKET",
            order_of(vec![
                field(1, Transform::Bucket(4), true, true),
                field(1, Transform::Identity, true, true),
            ]),
            SPARK_SORT_TRANSFORM_BUCKET,
        ),
        (
            "SORT-TRANSFORM-TRUNC",
            order_of(vec![
                field(5, Transform::Truncate(2), false, false),
                field(1, Transform::Identity, true, true),
            ]),
            SPARK_SORT_TRANSFORM_TRUNC,
        ),
        (
            "SORT-TRANSFORM-DAYS",
            order_of(vec![
                field(3, Transform::Day, true, true),
                field(1, Transform::Identity, true, true),
            ]),
            SPARK_SORT_TRANSFORM_DAYS,
        ),
    ] {
        let (catalog, _guard) = local_fs_catalog().await;
        let table = oracle_table(&catalog, FormatVersion::V2, false, None).await;
        RewriteDataFiles::new(table.clone())
            .strategy(RewriteStrategy::Sort(order))
            .rewrite_all(true)
            .execute(&catalog)
            .await
            .expect("sort rewrite");
        let table = catalog
            .load_table(table.identifier())
            .await
            .expect("reload");
        let rows = concatenated_rows(&output_files(&table).await);
        assert_eq!(
            indexes(&rows),
            expected.to_vec(),
            "{cell}: the rewritten row order is not Spark's"
        );
    }
}

#[tokio::test]
async fn sort_partitioned_orders_every_partition_group() {
    for (cell, table_order, strategy, expected, stamp) in [
        (
            "SORT-PARTITIONED",
            None,
            RewriteStrategy::Sort(id_desc_nulls_last()),
            SPARK_SORT_PARTITIONED,
            Some(0),
        ),
        (
            "SORT-TABLE-ORDER-PARTITIONED",
            Some(id_desc_nulls_last()),
            RewriteStrategy::SortByTableOrder,
            SPARK_SORT_TABLE_ORDER_PARTITIONED,
            Some(1),
        ),
    ] {
        let (catalog, _guard) = local_fs_catalog().await;
        let table = oracle_table(&catalog, FormatVersion::V2, true, table_order).await;
        RewriteDataFiles::new(table.clone())
            .strategy(strategy)
            .rewrite_all(true)
            .execute(&catalog)
            .await
            .expect("sort rewrite");
        let table = catalog
            .load_table(table.identifier())
            .await
            .expect("reload");
        let files = output_files(&table).await;
        assert_eq!(files.len(), 4, "{cell}: one output file per partition");
        assert_eq!(stamps(&files), vec![stamp; files.len()], "{cell}");
        for (partition, spark_rows) in expected {
            let file = files
                .iter()
                .find(|file| {
                    let rows_cat = file.rows.first().and_then(|row| row.cat.clone());
                    rows_cat.as_deref().unwrap_or("") == *partition
                })
                .unwrap_or_else(|| panic!("{cell}: no output file for partition '{partition}'"));
            assert_same_key_order(cell, spark_rows, &file.rows, compare_id_desc_nulls_last);
        }
    }
}

#[tokio::test]
async fn sort_with_a_small_target_keeps_one_global_order_in_one_rolled_file() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(&catalog, FormatVersion::V2, false, None).await;
    let order = order_of(vec![field(1, Transform::Identity, true, true)]);

    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(order))
        .target_file_size_bytes(1500)
        .min_file_size_bytes(1)
        .max_file_size_bytes(2700)
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("sort rewrite");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    for file in &files {
        assert_non_decreasing("SORT-TARGET-SMALL", &file.rows, compare_id_asc_nulls_first);
    }
    let spark_order: Vec<i64> = SPARK_SORT_TARGET_SMALL_FILES
        .iter()
        .flat_map(|file| file.iter().copied())
        .collect();
    assert_same_key_order(
        "SORT-TARGET-SMALL",
        &spark_order,
        &concatenated_rows(&files),
        compare_id_asc_nulls_first,
    );
}

#[tokio::test]
async fn bin_pack_stays_the_default_strategy() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = oracle_table(&catalog, FormatVersion::V2, false, None).await;

    RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("bin-pack rewrite");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = output_files(&table).await;
    assert_eq!(stamps(&files), vec![Some(0); files.len()]);
    let mut rewritten = indexes(&concatenated_rows(&files));
    rewritten.sort_unstable();
    let mut inputs: Vec<i64> = oracle_batches()
        .into_iter()
        .flatten()
        .map(|row| row.index)
        .collect();
    inputs.sort_unstable();
    assert_eq!(rewritten, inputs);
}
