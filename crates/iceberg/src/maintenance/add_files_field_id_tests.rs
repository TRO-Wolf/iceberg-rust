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

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema as ArrowSchema};
use bytes::Bytes;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};

use super::add_files::{AddFiles, AddFilesSource};
use super::add_files_tests::{
    create_table, create_table_with_properties, flat_source, id_v_schema, live_data_files,
    local_fs_catalog, long_column, scan_id_v, source_root, string_column, write_source_file,
};
use crate::spec::{
    Datum, FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, Transform, Type,
};
use crate::{Catalog, NamespaceIdent, TableCreation};

#[tokio::test]
async fn the_table_metrics_config_decides_the_adopted_bounds() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let counted = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let counted_root = source_root(&temp_dir, "metrics-default");
    flat_source(&counted, &counted_root).await;
    AddFiles::new(counted.clone(), AddFilesSource::Directory(counted_root))
        .execute(&catalog)
        .await
        .expect("adopt under the default metrics config");
    let counted = catalog
        .load_table(counted.identifier())
        .await
        .expect("reload table");
    let adopted = live_data_files(&counted).await;
    assert!(
        !adopted[0].lower_bounds().is_empty(),
        "the default truncate(16) mode keeps bounds"
    );
    assert!(!adopted[0].column_sizes().is_empty());

    let none = create_table_with_properties(
        &catalog,
        id_v_schema(),
        None,
        FormatVersion::V2,
        HashMap::from([(
            "write.metadata.metrics.default".to_string(),
            "none".to_string(),
        )]),
    )
    .await;
    let none_root = source_root(&temp_dir, "metrics-none");
    flat_source(&none, &none_root).await;
    AddFiles::new(none.clone(), AddFilesSource::Directory(none_root))
        .execute(&catalog)
        .await
        .expect("adopt under metrics mode none");
    let none = catalog
        .load_table(none.identifier())
        .await
        .expect("reload table");
    let adopted = live_data_files(&none).await;
    assert!(
        adopted[0].lower_bounds().is_empty() && adopted[0].upper_bounds().is_empty(),
        "MetricsConfig::for_table reads write.metadata.metrics.default"
    );
    assert!(
        adopted[0].column_sizes().is_empty() && adopted[0].value_counts().is_empty(),
        "mode none persists nothing for the column"
    );
    assert_eq!(
        adopted[0].record_count(),
        2,
        "the record count comes from the footer, not the metrics config"
    );
}

pub(super) fn field_id_parquet_bytes(columns: &[(&str, i32, ArrayRef)]) -> Bytes {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, field_id, array)| {
            Field::new(*name, array.data_type().clone(), true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                field_id.to_string(),
            )]))
        })
        .collect();
    let arrow_schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        columns.iter().map(|(_, _, array)| array.clone()).collect(),
    )
    .expect("source record batch");
    let mut buffer: Vec<u8> = Vec::new();
    let mut writer =
        ArrowWriter::try_new(&mut buffer, arrow_schema, None).expect("source parquet writer");
    writer.write(&batch).expect("write source batch");
    writer.close().expect("close source parquet writer");
    Bytes::from(buffer)
}

#[tokio::test]
async fn an_id_less_source_resolves_its_columns_by_name_not_by_position() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "reordered");
    write_source_file(&table, &format!("{root}/part-00000.parquet"), &[
        ("v", string_column(&["p", "q"])),
        ("id", long_column(&[10, 11])),
    ])
    .await;

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("a source whose column order differs from the table's field ids");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    assert_eq!(
        files[0].lower_bounds().get(&1),
        Some(&Datum::long(10)),
        "field 1 is id because the name mapping says so, not because it is the first column"
    );
    assert_eq!(
        files[0].lower_bounds().get(&2),
        Some(&Datum::string("p")),
        "field 2 is v"
    );
    assert_eq!(scan_id_v(&table).await, vec![
        (Some(10), Some("p".to_string())),
        (Some(11), Some("q".to_string())),
    ]);
}

#[tokio::test]
async fn a_source_that_carries_field_ids_is_resolved_by_those_ids() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "embedded-ids");
    table
        .file_io()
        .new_output(format!("{root}/part-00000.parquet"))
        .expect("source output file")
        .write(field_id_parquet_bytes(&[
            ("x", 1, long_column(&[10, 11])),
            ("y", 2, string_column(&["p", "q"])),
        ]))
        .await
        .expect("write a source that carries field ids");

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("a source whose columns carry field ids the table's names would not find");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    assert_eq!(
        files[0].lower_bounds().get(&1),
        Some(&Datum::long(10)),
        "Java ParquetUtil uses the file's own ids when it has any, so column x is field 1"
    );
    assert_eq!(
        files[0].lower_bounds().get(&2),
        Some(&Datum::string("p")),
        "column y is field 2"
    );
}

#[tokio::test]
async fn a_field_id_the_table_schema_lacks_is_dropped_from_the_adopted_file() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_schema(), None, FormatVersion::V2).await;
    let root = source_root(&temp_dir, "ghost-id");
    table
        .file_io()
        .new_output(format!("{root}/part-00000.parquet"))
        .expect("source output file")
        .write(field_id_parquet_bytes(&[
            ("id", 1, long_column(&[10, 11])),
            ("ghost", 9, string_column(&["p", "q"])),
        ]))
        .await
        .expect("write a source carrying an id the table does not have");

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("a source carrying a foreign field id");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    assert_eq!(files[0].lower_bounds().get(&1), Some(&Datum::long(10)));
    assert!(
        !files[0].lower_bounds().contains_key(&9)
            && !files[0].column_sizes().contains_key(&9)
            && !files[0].value_counts().contains_key(&9),
        "ledger D-3a: an id the table schema lacks carries no metrics into the manifest"
    );
    assert_eq!(scan_id_v(&table).await, vec![
        (Some(10), None),
        (Some(11), None)
    ]);
}

#[tokio::test]
async fn a_two_column_partition_tuple_follows_the_spec_field_order() {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let schema = Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "cat",
                Type::Primitive(PrimitiveType::String),
            )),
            Arc::new(NestedField::optional(
                3,
                "dept",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("build id/cat/dept schema");
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("cat", "cat", Transform::Identity)
        .expect("add cat")
        .add_partition_field("dept", "dept", Transform::Identity)
        .expect("add dept")
        .build()
        .expect("build spec");
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("create namespace");
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .expect("create table");

    let root = source_root(&temp_dir, "two-column");
    write_source_file(
        &table,
        &format!("{root}/cat=x/dept=hr/part-00000.parquet"),
        &[("id", long_column(&[1]))],
    )
    .await;

    AddFiles::new(table.clone(), AddFilesSource::Directory(root))
        .execute(&catalog)
        .await
        .expect("adopt a two-column hive layout");

    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    let files = live_data_files(&table).await;
    let partition: Vec<Option<&crate::spec::Literal>> = files[0].partition().iter().collect();
    assert_eq!(partition.len(), 2);
    assert_eq!(
        partition[0],
        Some(&crate::spec::Literal::string("x")),
        "Java maps each SPEC FIELD NAME through the partition map, in spec order"
    );
    assert_eq!(partition[1], Some(&crate::spec::Literal::string("hr")));
}
