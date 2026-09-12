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

use iceberg::spec::TableProperties;
use rewrite_size_shared::{
    build_bed, create_fixture, live_file_paths, measure_set, remove_table_property,
};

#[tokio::test]
#[ignore = "forward pin for F-REWRITE-SIZE-1 step 2; red until the fix lands"]
async fn rewrite_output_within_ten_percent_of_input() {
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
        ratio <= 1.1,
        "rewrite output must stay within 10% of input compressed bytes, got {ratio:.4} ({} vs {})",
        output.compressed,
        input.compressed,
    );
}
