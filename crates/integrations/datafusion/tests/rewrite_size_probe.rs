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

mod rewrite_size_shared;

use iceberg::Catalog;
use iceberg::spec::TableProperties;
use rewrite_size_shared::{
    ProbeFixture, SetTotals, build_bed, create_fixture, flip_table, live_file_paths, measure_set,
    print_first_rows, print_writer_properties_table, remove_table_property, scan_tasks,
};

async fn run_repark_scenario(fixture: &ProbeFixture) -> (SetTotals, SetTotals) {
    build_bed(fixture).await;
    let input_paths = live_file_paths(&fixture.catalog, &fixture.table_ident).await;
    let input_totals = measure_set(
        "INSERT files (zstd level 3, Spark-equivalent bed)",
        &input_paths,
        2,
    );
    if let Some(first) = input_paths.first() {
        print_first_rows(first, "INSERT");
    }
    let input_tasks = {
        let table = fixture
            .catalog
            .load_table(&fixture.table_ident)
            .await
            .expect("load table for tasks");
        scan_tasks(&table).await
    };
    println!(
        "== input scan task order (first 10): {:?}",
        input_tasks
            .iter()
            .take(10)
            .map(|task| task.data_file_path().to_string())
            .collect::<Vec<_>>()
    );
    let table = remove_table_property(
        &fixture.catalog,
        &fixture.table_ident,
        TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL,
    )
    .await;
    let result = iceberg::maintenance::RewriteDataFiles::new(table.clone())
        .min_input_files(2)
        .execute(fixture.catalog.as_ref())
        .await
        .expect("run rewrite_data_files");
    println!(
        "== rewrite_data_files at fork defaults (zstd level 3): rewritten={} added={} groups={}",
        result.rewritten_data_files_count,
        result.added_data_files_count,
        result.file_groups.len(),
    );
    let output_paths = live_file_paths(&fixture.catalog, &fixture.table_ident).await;
    let output_totals = measure_set("rewrite_data_files output", &output_paths, 2);
    if let Some(first) = output_paths.first() {
        print_first_rows(first, "REWRITE");
    }
    print_writer_properties_table(&table);
    flip_table(
        &table,
        &input_tasks,
        input_totals.compressed,
        fixture.scratch.path(),
    )
    .await;
    (input_totals, output_totals)
}

#[tokio::test]
#[ignore = "measurement probe, not a CI pin"]
async fn rewrite_size_probe() {
    let fixture = create_fixture(Some("3")).await;
    let (input, output) = run_repark_scenario(&fixture).await;
    let ratio = output.compressed as f64 / input.compressed as f64;
    println!(
        "== totals: input_compressed={} output_compressed={} out/in={ratio:.6}",
        input.compressed, output.compressed,
    );

    assert!(
        input.files >= 150,
        "bed must land >= 150 insert files, got {}",
        input.files
    );
    assert!(input.compressed > 0 && input.uncompressed > 0);
    let input_ratio = input.compressed as f64 / input.uncompressed as f64;
    assert!(
        (0.30..=0.50).contains(&input_ratio),
        "bed must reproduce the RePark zstd ratio (~0.38), got {input_ratio:.4}"
    );
    assert!(
        ratio <= 1.05,
        "post-fix check: rewrite output must stay within 5% of input compressed bytes, got {ratio:.4}"
    );
}
