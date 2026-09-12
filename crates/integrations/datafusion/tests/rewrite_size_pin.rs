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

use std::collections::HashMap;
use std::time::Instant;

use iceberg::Catalog;
use iceberg::spec::TableProperties;
use rewrite_size_shared::counting::create_counting_fixture;
use rewrite_size_shared::{
    build_bed, build_bed_n, build_low_cardinality_bed, column_dictionary_evidence, create_fixture,
    create_low_cardinality_fixture, live_file_paths, measure_set, remove_table_property,
};

const LEVEL_PIN_BATCHES: usize = 40;

#[tokio::test]
async fn rewrite_output_within_five_percent_of_input() {
    let fixture = create_fixture(Some("3")).await;
    build_bed(&fixture).await;
    let input_paths = live_file_paths(&fixture.catalog, &fixture.table_ident).await;
    let input = measure_set("pin input", &input_paths, 0);

    let table = remove_table_property(
        &fixture.catalog,
        &fixture.table_ident,
        TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL,
    )
    .await;
    iceberg::maintenance::RewriteDataFiles::new(table)
        .min_input_files(2)
        .execute(fixture.catalog.as_ref())
        .await
        .expect("run rewrite_data_files");

    let output_paths = live_file_paths(&fixture.catalog, &fixture.table_ident).await;
    let output = measure_set("pin output", &output_paths, 0);
    let ratio = output.compressed as f64 / input.compressed as f64;
    assert!(
        ratio <= 1.05,
        "rewrite output must stay within 5% of input compressed bytes, got {ratio:.4} ({} vs {})",
        output.compressed,
        input.compressed,
    );
}

#[tokio::test]
async fn rewrite_keeps_dictionary_on_low_cardinality_columns() {
    let fixture = create_low_cardinality_fixture(None).await;
    build_low_cardinality_bed(&fixture).await;
    let input_paths = live_file_paths(&fixture.catalog, &fixture.table_ident).await;
    let input = measure_set("low-cardinality pin input", &input_paths, 0);

    let table = fixture
        .catalog
        .load_table(&fixture.table_ident)
        .await
        .expect("load table for rewrite");
    iceberg::maintenance::RewriteDataFiles::new(table)
        .min_input_files(2)
        .execute(fixture.catalog.as_ref())
        .await
        .expect("run rewrite_data_files");

    let output_paths = live_file_paths(&fixture.catalog, &fixture.table_ident).await;
    let output = measure_set("low-cardinality pin output", &output_paths, 0);
    let ratio = output.compressed as f64 / input.compressed as f64;
    assert!(
        ratio <= 1.05,
        "low-cardinality rewrite output must stay within 5% of input compressed bytes, got {ratio:.4} ({} vs {})",
        output.compressed,
        input.compressed,
    );

    let (grp_chunks, grp_dict_pages, grp_dict_data) =
        column_dictionary_evidence(&output_paths, "grp");
    assert!(grp_chunks > 0, "output must carry grp column chunks");
    assert_eq!(
        grp_dict_pages, grp_chunks,
        "every output grp chunk must keep its dictionary page ({grp_dict_pages}/{grp_chunks})"
    );
    assert_eq!(
        grp_dict_data, grp_chunks,
        "every output grp chunk's data pages must stay dictionary-encoded ({grp_dict_data}/{grp_chunks})"
    );

    let (id_chunks, id_dict_pages, _) = column_dictionary_evidence(&output_paths, "id");
    assert!(id_chunks > 0, "output must carry id column chunks");
    assert_eq!(
        id_dict_pages, 0,
        "the near-unique id column must not write a dictionary page ({id_dict_pages}/{id_chunks})"
    );
}

#[tokio::test]
async fn insert_without_level_property_writes_level_three_bytes() {
    let default_fixture = create_fixture(None).await;
    build_bed_n(&default_fixture, LEVEL_PIN_BATCHES).await;
    let default_paths =
        live_file_paths(&default_fixture.catalog, &default_fixture.table_ident).await;
    let default_totals = measure_set("no-level-property insert", &default_paths, 0);

    let level3_fixture = create_fixture(Some("3")).await;
    build_bed_n(&level3_fixture, LEVEL_PIN_BATCHES).await;
    let level3_paths = live_file_paths(&level3_fixture.catalog, &level3_fixture.table_ident).await;
    let level3_totals = measure_set("level-3 insert", &level3_paths, 0);

    let level1_fixture = create_fixture(Some("1")).await;
    build_bed_n(&level1_fixture, LEVEL_PIN_BATCHES).await;
    let level1_paths = live_file_paths(&level1_fixture.catalog, &level1_fixture.table_ident).await;
    let level1_totals = measure_set("level-1 insert", &level1_paths, 0);

    assert_eq!(
        default_totals.compressed, level3_totals.compressed,
        "an INSERT without write.parquet.compression-level must write the same compressed bytes as an explicit level-3 write ({} vs {})",
        default_totals.compressed, level3_totals.compressed,
    );
    assert_ne!(
        default_totals.compressed, level1_totals.compressed,
        "the default-level INSERT must differ from a level-1 write or the pin proves nothing ({} vs {})",
        default_totals.compressed, level1_totals.compressed,
    );
}

#[tokio::test]
async fn rewrite_fetches_each_input_footer_once() {
    let (fixture, reads) = create_counting_fixture(Some("3")).await;
    build_bed(&fixture).await;
    let input_paths = live_file_paths(&fixture.catalog, &fixture.table_ident).await;
    let sizes: HashMap<String, u64> = input_paths
        .iter()
        .map(|path| {
            let local = path.strip_prefix("file://").unwrap_or(path);
            let size = std::fs::metadata(local)
                .unwrap_or_else(|error| panic!("stat {local}: {error}"))
                .len();
            (path.clone(), size)
        })
        .collect();

    let table = remove_table_property(
        &fixture.catalog,
        &fixture.table_ident,
        TableProperties::PROPERTY_PARQUET_COMPRESSION_LEVEL,
    )
    .await;
    reads.lock().expect("read log").clear();
    let started = Instant::now();
    iceberg::maintenance::RewriteDataFiles::new(table)
        .min_input_files(2)
        .execute(fixture.catalog.as_ref())
        .await
        .expect("run rewrite_data_files");
    let wall = started.elapsed();

    let log = reads.lock().expect("read log");
    let mut total_tail_fetches = 0usize;
    let mut total_fetched = 0u64;
    for path in &input_paths {
        let size = sizes[path];
        let ranges = log
            .get(path)
            .unwrap_or_else(|| panic!("no reads recorded for {path}"));
        let fetched: u64 = ranges.iter().map(|range| range.end - range.start).sum();
        total_tail_fetches += ranges.iter().filter(|range| range.end == size).count();
        total_fetched += fetched;
        let mut sorted = ranges.clone();
        sorted.sort_by_key(|range| range.start);
        for pair in sorted.windows(2) {
            assert!(
                pair[0].end <= pair[1].start,
                "{path}: overlapping reads fetch bytes twice: {:?} then {:?}",
                pair[0],
                pair[1],
            );
        }
        let tail_fetches = ranges.iter().filter(|range| range.end == size).count();
        assert_eq!(
            tail_fetches, 1,
            "{path}: the footer tail must be fetched exactly once across the whole rewrite, got {tail_fetches} tail-reaching reads (ranges: {ranges:?})",
        );
    }
    println!(
        "footer-fuse pin: files={} tail_fetches={} fetched_bytes={} fetched_per_file={:.0} wall={:?}",
        input_paths.len(),
        total_tail_fetches,
        total_fetched,
        total_fetched as f64 / input_paths.len() as f64,
        wall,
    );
}
