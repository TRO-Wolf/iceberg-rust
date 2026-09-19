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

use super::*;

#[tokio::test]
async fn execute_fails_loud_when_write_metadata_path_is_outside_the_source_prefix() {
    let (catalog, file_io, tmp) = local_fs_catalog().await;
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let warehouse = tmp.path().to_str().expect("utf8 temp path").to_string();
    let table = catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .schema(two_long_schema())
                .partition_spec(PartitionSpec::unpartition_spec())
                .properties([(
                    "write.metadata.path".to_string(),
                    format!("{warehouse}/external-meta"),
                )])
                .build(),
        )
        .await
        .expect("create table");
    let location = table.metadata().location().to_string();
    let d1 = real_data_file(&file_io, &format!("{location}/data/d1.parquet"), b"d1").await;
    let table = append(&catalog, &table, vec![d1]).await;
    assert!(
        table
            .metadata_location_result()
            .expect("metadata location")
            .starts_with(&format!("{warehouse}/external-meta/")),
        "the fixture must place metadata outside the table location"
    );

    let err = super::RewriteTablePath::new(table)
        .rewrite_location_prefix(&location, "s3://bucket/relocated")
        .staging_location(format!("{location}-staging"))
        .execute(&file_io)
        .await
        .expect_err("metadata outside the source prefix must fail loudly");
    assert!(
        err.to_string().contains("does not start with"),
        "expected a source-prefix relativize error, got: {err}"
    );
}
