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
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;

use crate::arrow::{ArrowFileReader, schema_to_arrow_schema};
use crate::io::FileMetadata;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{append_files, local_fs_catalog};
use crate::maintenance::rewrite_data_files_sort_harness::{live_output_paths, spill_files};
use crate::maintenance::{RewriteStrategy, ZOrderSpec};
use crate::scan::FileScanTask;
use crate::spec::{
    DataContentType, DataFile, FormatVersion, NestedField, NullOrder, PrimitiveType, Schema,
    SortDirection, SortField, SortOrder, Struct, Transform, Type,
};
use crate::table::Table;
use crate::writer::file_writer::{FileWriter, FileWriterBuilder, ParquetWriterBuilder};
use crate::{Catalog, NamespaceIdent, TableCreation};

const ROWS_PER_FILE: i64 = 6000;
const INPUT_FILES: i64 = 4;

fn wide_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "s",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build the wide schema")
}

fn id_asc() -> SortOrder {
    SortOrder {
        order_id: 1,
        fields: vec![SortField {
            source_id: 1,
            transform: Transform::Identity,
            direction: SortDirection::Ascending,
            null_order: NullOrder::First,
        }],
    }
}

async fn scattered_table(catalog: &impl Catalog) -> Table {
    let schema = wide_schema();
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation {
        name: "t".to_string(),
        location: None,
        schema: schema.clone(),
        partition_spec: None,
        sort_order: None,
        properties: HashMap::new(),
        format_version: FormatVersion::V2,
    };
    let mut table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create the scattered table");
    for file in 0..INPUT_FILES {
        let rows: Vec<i64> = (0..ROWS_PER_FILE)
            .map(|row| ((row * INPUT_FILES + file) * 7919) % (ROWS_PER_FILE * INPUT_FILES))
            .collect();
        let data_file = write_scattered_file(&table, &format!("in-{file}.parquet"), &rows).await;
        table = append_files(catalog, &table, vec![data_file]).await;
    }
    table
}

async fn write_scattered_file(table: &Table, file_name: &str, ids: &[i64]) -> DataFile {
    let schema = table.metadata().current_schema();
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).expect("arrow schema"));
    let strings: Vec<String> = ids
        .iter()
        .map(|id| format!("row-{id:08}-padding"))
        .collect();
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
        Arc::new(StringArray::from(strings)) as ArrayRef,
    ])
    .expect("build the scattered batch");

    let file_path = format!("{}/data/{file_name}", table.metadata().location());
    let output = table.file_io().new_output(file_path).expect("output file");
    let builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    let mut writer = builder.build(output).await.expect("parquet writer");
    writer.write(&batch).await.expect("write scattered rows");
    let mut data_file = writer
        .close()
        .await
        .expect("close scattered writer")
        .into_iter()
        .next()
        .expect("one data file builder");
    data_file
        .content(DataContentType::Data)
        .partition_spec_id(0)
        .partition(Struct::empty());
    data_file.build().expect("build the scattered data file")
}

async fn planned_group(table: &Table) -> Vec<FileScanTask> {
    table
        .scan()
        .build()
        .expect("scan builder")
        .plan_files()
        .await
        .expect("plan files")
        .try_collect()
        .await
        .expect("collect tasks")
}

async fn file_ids(table: &Table, path: &str) -> Vec<Option<i64>> {
    let input = table.file_io().new_input(path).expect("input file");
    let size = input.metadata().await.expect("metadata").size;
    let reader = ArrowFileReader::new(FileMetadata { size }, input.reader().await.expect("reader"));
    let stream = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .expect("stream builder")
        .build()
        .expect("stream");
    let batches: Vec<RecordBatch> = stream.try_collect().await.expect("read batches");
    let mut ids = Vec::new();
    for batch in batches {
        let column = batch.column_by_name("id").expect("id column").clone();
        for row in 0..batch.num_rows() {
            ids.push((!column.is_null(row)).then(|| column.as_primitive::<Int64Type>().value(row)));
        }
    }
    ids
}

fn assert_ascending(context: &str, ids: &[Option<i64>]) {
    for window in ids.windows(2) {
        assert!(
            window[0] <= window[1],
            "{context}: {:?} precedes {:?} but sorts after it",
            window[0],
            window[1]
        );
    }
}

#[tokio::test]
async fn sorted_output_files_hold_one_global_order_in_disjoint_ranges() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = scattered_table(&catalog).await;

    RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(id_asc()))
        .target_file_size_bytes(16 * 1024)
        .min_file_size_bytes(1)
        .max_file_size_bytes(64 * 1024)
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("sort rewrite");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload");
    let files = live_output_paths(&table).await;
    assert!(
        files.len() > 1,
        "the rewrite wrote {} file(s); the pin needs a group that rolls",
        files.len()
    );

    let mut all = Vec::new();
    for (path, _) in &files {
        let ids = file_ids(&table, path).await;
        assert_ascending("one output file", &ids);
        all.extend(ids);
    }
    assert_ascending("across output files in name order", &all);
    assert_eq!(all.len() as i64, ROWS_PER_FILE * INPUT_FILES);
}

#[tokio::test]
async fn a_small_sort_budget_spills_runs_and_still_writes_one_global_order() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = scattered_table(&catalog).await;
    let budget = 64 * 1024u64;

    let action = RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(id_asc()))
        .sort_memory_budget_bytes(budget)
        .rewrite_all(true);
    let group = planned_group(&table).await;
    let written = action
        .write_group_for_test(&table, &group)
        .await
        .expect("sorted write");

    assert!(
        written.sort_stats.spilled_runs > 1,
        "the budget must force several runs, got {}",
        written.sort_stats.spilled_runs
    );
    assert!(
        written.sort_stats.peak_sort_bytes <= budget * 2,
        "buffered {} bytes against a {budget}-byte budget",
        written.sort_stats.peak_sort_bytes
    );
    assert_eq!(
        spill_files(&table),
        Vec::<String>::new(),
        "a spill file survived the rewrite"
    );

    let mut ids = Vec::new();
    for file in &written.files {
        ids.extend(file_ids(&table, file.file_path()).await);
    }
    assert_ascending("the merged output", &ids);
    assert_eq!(ids.len() as i64, ROWS_PER_FILE * INPUT_FILES);
    let mut sorted_ids = ids.clone();
    sorted_ids.sort_unstable();
    assert_eq!(ids, sorted_ids);
}

#[tokio::test]
async fn more_runs_than_the_merge_fan_in_merge_in_passes() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = scattered_table(&catalog).await;

    let action = RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::Sort(id_asc()))
        .sort_memory_budget_bytes(1)
        .rewrite_all(true);
    let group = planned_group(&table).await;
    let written = action
        .write_group_for_test(&table, &group)
        .await
        .expect("sorted write");

    assert!(
        written.sort_stats.spilled_runs
            > crate::maintenance::rewrite_data_files_sort_run::SORT_MERGE_FAN_IN,
        "the pin needs more runs than the fan-in, got {}",
        written.sort_stats.spilled_runs
    );
    assert!(
        written.sort_stats.merge_passes > 1,
        "more runs than the fan-in must merge in passes, got {}",
        written.sort_stats.merge_passes
    );
    assert_eq!(spill_files(&table), Vec::<String>::new());

    let mut ids = Vec::new();
    for file in &written.files {
        ids.extend(file_ids(&table, file.file_path()).await);
    }
    assert_ascending("the multi-pass merged output", &ids);
    assert_eq!(ids.len() as i64, ROWS_PER_FILE * INPUT_FILES);
}

#[tokio::test]
async fn a_small_sort_budget_spills_a_zorder_rewrite_too() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = scattered_table(&catalog).await;

    let action = RewriteDataFiles::new(table.clone())
        .strategy(RewriteStrategy::ZOrder(ZOrderSpec::new(["id", "s"])))
        .sort_memory_budget_bytes(64 * 1024)
        .rewrite_all(true);
    let group = planned_group(&table).await;
    let written = action
        .write_group_for_test(&table, &group)
        .await
        .expect("z-order write");

    assert!(written.sort_stats.spilled_runs > 1);
    assert_eq!(spill_files(&table), Vec::<String>::new());
    let mut rows = 0i64;
    for file in &written.files {
        rows += file_ids(&table, file.file_path()).await.len() as i64;
    }
    assert_eq!(rows, ROWS_PER_FILE * INPUT_FILES);
}

#[tokio::test]
async fn strategy_option_preconditions_are_refused_with_javas_messages() {
    let (catalog, _guard) = local_fs_catalog().await;
    let table = scattered_table(&catalog).await;

    let cases: Vec<(RewriteDataFiles, &str)> = vec![
        (
            RewriteDataFiles::new(table.clone())
                .strategy(RewriteStrategy::Sort(id_asc()))
                .shuffle_partitions_per_file(0),
            "'shuffle-partitions-per-file' is set to 0 but must be > 0",
        ),
        (
            RewriteDataFiles::new(table.clone())
                .strategy(RewriteStrategy::Sort(id_asc()))
                .compression_factor(0.0),
            "'compression-factor' is set to 0.0 but must be > 0",
        ),
        (
            RewriteDataFiles::new(table.clone()).compression_factor(2.0),
            "Cannot use options [compression-factor], they are not supported by the action or the rewriter BIN-PACK",
        ),
        (
            RewriteDataFiles::new(table.clone()).shuffle_partitions_per_file(2),
            "Cannot use options [shuffle-partitions-per-file], they are not supported by the action or the rewriter BIN-PACK",
        ),
        (
            RewriteDataFiles::new(table.clone())
                .strategy(RewriteStrategy::Sort(id_asc()))
                .sort_memory_budget_bytes(0),
            "'sort-memory-budget-bytes' is set to 0 but must be > 0",
        ),
    ];
    for (action, expected) in cases {
        let error = action
            .rewrite_all(true)
            .execute(&catalog)
            .await
            .expect_err("the option must be refused");
        assert_eq!(error.message(), expected);
    }
}

#[tokio::test]
async fn layout_only_options_are_accepted_and_change_nothing() {
    let (catalog, _guard) = local_fs_catalog().await;
    let mut outputs = Vec::new();
    for knob in 0..3 {
        let table = scattered_table(&catalog).await;
        let action = RewriteDataFiles::new(table.clone())
            .strategy(RewriteStrategy::Sort(id_asc()))
            .rewrite_all(true);
        let action = match knob {
            1 => action.shuffle_partitions_per_file(4),
            2 => action.compression_factor(2.5),
            _ => action,
        };
        action.execute(&catalog).await.expect("sort rewrite");
        let table = catalog
            .load_table(table.identifier())
            .await
            .expect("reload");
        let mut ids = Vec::new();
        for (path, _) in live_output_paths(&table).await {
            ids.extend(file_ids(&table, &path).await);
        }
        outputs.push(ids);
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0], outputs[2]);
}
