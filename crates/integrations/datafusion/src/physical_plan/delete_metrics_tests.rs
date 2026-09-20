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

use iceberg::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use iceberg::{CatalogBuilder, NamespaceIdent, TableCreation};

use super::*;

async fn metrics_none_table() -> (iceberg::table::Table, TempDir) {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "metrics_none",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().expect("utf8").to_string(),
            )]),
        )
        .await
        .expect("catalog");
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "val", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .location(format!("{}/t", warehouse.path().to_str().expect("utf8")))
        .schema(schema)
        .format_version(FormatVersion::V2)
        .properties(HashMap::from([(
            "write.metadata.metrics.default".to_string(),
            "none".to_string(),
        )]))
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("table");
    (table, warehouse)
}

#[tokio::test]
async fn test_streaming_data_file_writer_honors_metrics_default_none() {
    let (table, _warehouse) = metrics_none_table().await;
    let arrow_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let batch = RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
    ])
    .expect("batch");

    let mut writer = crate::physical_plan::row_lineage::StreamingDataFileWriter::try_new(
        &table,
        table.metadata().default_partition_spec().clone(),
    )
    .expect("streaming writer");
    writer.write_batch(batch).await.expect("write batch");
    let files = writer.finish().await.expect("finish");
    assert_eq!(files.len(), 1);
    let file = &files[0];

    assert_eq!(file.record_count(), 3);
    for (name, len) in [
        ("column_sizes", file.column_sizes().len()),
        ("value_counts", file.value_counts().len()),
        ("null_value_counts", file.null_value_counts().len()),
        ("nan_value_counts", file.nan_value_counts().len()),
        ("lower_bounds", file.lower_bounds().len()),
        ("upper_bounds", file.upper_bounds().len()),
    ] {
        assert_eq!(len, 0, "metrics.default=none must strip {name}");
    }
}

#[tokio::test]
async fn test_write_position_deletes_honors_metrics_default_none() {
    let (table, _warehouse) = metrics_none_table().await;
    let deleted_path = format!(
        "{}/data/a-very-long-referenced-data-file-name.parquet",
        table.metadata().location()
    );
    let pairs = vec![(deleted_path.clone(), 0), (deleted_path.clone(), 7)];
    let files = crate::physical_plan::delete::delete_position_deletes::write_position_deletes(
        &table, &pairs, None,
    )
    .await
    .expect("write position deletes");
    assert_eq!(files.len(), 1);
    let file = &files[0];

    let reserved: HashSet<i32> = HashSet::from([
        RESERVED_FIELD_ID_DELETE_FILE_PATH,
        RESERVED_FIELD_ID_DELETE_FILE_POS,
    ]);
    for keys in [
        file.column_sizes()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
        file.value_counts()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
        file.null_value_counts()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
        file.lower_bounds()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
        file.upper_bounds()
            .keys()
            .copied()
            .collect::<HashSet<i32>>(),
    ] {
        assert_eq!(keys, reserved);
    }
    assert!(file.nan_value_counts().is_empty());
    let bound = file
        .lower_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        .expect("file_path lower bound")
        .to_bytes()
        .unwrap();
    let path_str = String::from_utf8(bound.as_ref().to_vec()).unwrap();
    assert_eq!(path_str, deleted_path, "file_path bound must be FULL");
}
