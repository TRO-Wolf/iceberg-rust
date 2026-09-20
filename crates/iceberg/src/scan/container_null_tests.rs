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

#![allow(missing_docs)]

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use futures::TryStreamExt;
use minijinja::context;
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use super::partitioning_fixtures::render_template;
use super::tests::{TableTestFixture, decode_int64_column};
use crate::TableIdent;
use crate::expr::{Predicate, Reference};
use crate::io::FileIO;
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Datum, ManifestEntry, ManifestListWriter,
    ManifestStatus, ManifestWriterBuilder, NestedField, PartitionSpec, PrimitiveType, Schema,
    Struct, StructType, TableMetadata, TableProperties, Type,
};
use crate::table::Table;

const CONTAINER_NAME_MAPPING: &str = r#"[{"field-id":9,"names":["id"]},{"field-id":10,"names":["st"],"fields":[{"field-id":11,"names":["a"]},{"field-id":12,"names":["b"]}]},{"field-id":13,"names":["xs"],"fields":[{"field-id":14,"names":["element"]}]},{"field-id":15,"names":["mp"],"fields":[{"field-id":16,"names":["key"]},{"field-id":17,"names":["value"]}]},{"field-id":18,"names":["deep"],"fields":[{"field-id":19,"names":["inner"],"fields":[{"field-id":20,"names":["x"]},{"field-id":21,"names":["ys"],"fields":[{"field-id":22,"names":["element"]}]}]}]}]"#;

impl TableTestFixture {
    pub fn new_container_columns() -> Self {
        Self::new_container_columns_inner(None)
    }

    pub fn new_container_columns_with_name_mapping(name_mapping_json: &str) -> Self {
        Self::new_container_columns_inner(Some(name_mapping_json))
    }

    fn new_container_columns_inner(name_mapping_json: Option<&str>) -> Self {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().join("table1");
        let manifest_list1_location = table_location.join("metadata/manifests_list_1.avro");
        let manifest_list2_location = table_location.join("metadata/manifests_list_2.avro");
        let table_metadata1_location = table_location.join("metadata/v1.json");

        let file_io = FileIO::new_with_fs();

        let mut table_metadata = {
            let template_json_str = fs::read_to_string(format!(
                "{}/testdata/example_table_metadata_v2.json",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let metadata_json = render_template(&template_json_str, context! {
                table_location => &table_location,
                manifest_list_1_location => &manifest_list1_location,
                manifest_list_2_location => &manifest_list2_location,
                table_metadata_1_location => &table_metadata1_location,
            });
            serde_json::from_str::<TableMetadata>(&metadata_json).unwrap()
        };

        table_metadata.default_spec = Arc::new(PartitionSpec::unpartition_spec());
        table_metadata.partition_specs.clear();
        table_metadata.default_partition_type = StructType::new(vec![]);
        table_metadata
            .partition_specs
            .insert(0, table_metadata.default_spec.clone());

        let extended_fields: Vec<_> = table_metadata
            .schemas
            .get(&1)
            .expect("template carries schema id 1")
            .as_struct()
            .fields()
            .iter()
            .cloned()
            .chain([
                NestedField::required(9, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(
                    10,
                    "st",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(11, "a", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::optional(12, "b", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
                NestedField::optional(
                    13,
                    "xs",
                    Type::List(crate::spec::ListType::new(
                        NestedField::optional(14, "element", Type::Primitive(PrimitiveType::Int))
                            .into(),
                    )),
                )
                .into(),
                NestedField::optional(
                    15,
                    "mp",
                    Type::Map(crate::spec::MapType::new(
                        NestedField::required(16, "key", Type::Primitive(PrimitiveType::String))
                            .into(),
                        NestedField::optional(17, "value", Type::Primitive(PrimitiveType::Int))
                            .into(),
                    )),
                )
                .into(),
                NestedField::optional(
                    18,
                    "deep",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(
                            19,
                            "inner",
                            Type::Struct(StructType::new(vec![
                                NestedField::optional(
                                    20,
                                    "x",
                                    Type::Primitive(PrimitiveType::String),
                                )
                                .into(),
                                NestedField::optional(
                                    21,
                                    "ys",
                                    Type::List(crate::spec::ListType::new(
                                        NestedField::optional(
                                            22,
                                            "element",
                                            Type::Primitive(PrimitiveType::Int),
                                        )
                                        .into(),
                                    )),
                                )
                                .into(),
                            ])),
                        )
                        .into(),
                    ])),
                )
                .into(),
            ])
            .collect();
        let extended_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(extended_fields)
                .build()
                .expect("extend fixture schema with container columns"),
        );
        table_metadata.schemas.insert(1, extended_schema);
        table_metadata.last_column_id = 22;
        if let Some(mapping) = name_mapping_json {
            table_metadata.properties.insert(
                TableProperties::PROPERTY_DEFAULT_NAME_MAPPING.to_string(),
                mapping.to_string(),
            );
        }

        let table = Table::builder()
            .metadata(table_metadata)
            .identifier(TableIdent::from_strs(["db", "table1"]).unwrap())
            .file_io(file_io.clone())
            .metadata_location(table_metadata1_location.to_str().unwrap())
            .build()
            .unwrap();

        Self {
            table_location: table_location.to_str().unwrap().to_string(),
            table,
        }
    }

    fn write_container_parquet_file(&self) -> u64 {
        Self::write_container_columns_to_parquet(
            &self.table_location,
            "containers.parquet",
            container_test_columns(true),
        )
    }

    fn write_id_less_container_parquet_file(&self) -> u64 {
        let mut columns = template_test_columns();
        columns.extend(container_test_columns(false));
        Self::write_container_columns_to_parquet(
            &self.table_location,
            "containers_idless.parquet",
            columns,
        )
    }

    fn write_reordered_id_less_container_parquet_file(&self) -> u64 {
        let mut columns = container_test_columns(false);
        columns.reverse();
        Self::write_container_columns_to_parquet(
            &self.table_location,
            "containers_reordered.parquet",
            columns,
        )
    }

    fn write_container_columns_to_parquet(
        table_location: &str,
        file_name: &str,
        columns: Vec<(arrow_schema::Field, ArrayRef)>,
    ) -> u64 {
        std::fs::create_dir_all(table_location).unwrap();
        let (fields, arrays): (Vec<_>, Vec<_>) = columns.into_iter().unzip();
        let arrow_schema = Arc::new(arrow_schema::Schema::new(fields));
        let batch = RecordBatch::try_new(arrow_schema.clone(), arrays).unwrap();
        let file = File::create(format!("{table_location}/{file_name}")).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            arrow_schema,
            Some(WriterProperties::builder().build()),
        )
        .unwrap();
        writer.write(&batch).expect("Writing batch");
        writer.close().unwrap();

        std::fs::metadata(format!("{table_location}/{file_name}"))
            .unwrap()
            .len()
    }

    pub async fn setup_container_manifest_files(&mut self) {
        let size = self.write_container_parquet_file();
        self.register_container_manifest_file("containers.parquet", size)
            .await;
    }

    pub async fn setup_id_less_container_manifest_files(&mut self) {
        let size = self.write_id_less_container_parquet_file();
        self.register_container_manifest_file("containers_idless.parquet", size)
            .await;
    }

    pub async fn setup_reordered_id_less_container_manifest_files(&mut self) {
        let size = self.write_reordered_id_less_container_parquet_file();
        self.register_container_manifest_file("containers_reordered.parquet", size)
            .await;
    }
}

fn container_test_columns(with_ids: bool) -> Vec<(arrow_schema::Field, ArrayRef)> {
    let field_id_metadata =
        |id: &str| HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]);
    let fid = |id: i32| {
        if with_ids {
            field_id_metadata(&id.to_string())
        } else {
            HashMap::new()
        }
    };

    let id_field =
        arrow_schema::Field::new("id", arrow_schema::DataType::Int64, false).with_metadata(fid(9));

    let a_field = Arc::new(
        arrow_schema::Field::new("a", arrow_schema::DataType::Utf8, true).with_metadata(fid(11)),
    );
    let b_field = Arc::new(
        arrow_schema::Field::new("b", arrow_schema::DataType::Int32, true).with_metadata(fid(12)),
    );
    let st_field = arrow_schema::Field::new(
        "st",
        arrow_schema::DataType::Struct(arrow_schema::Fields::from([
            a_field.clone(),
            b_field.clone(),
        ])),
        true,
    )
    .with_metadata(fid(10));

    let xs_element_field = Arc::new(
        arrow_schema::Field::new("element", arrow_schema::DataType::Int32, true)
            .with_metadata(fid(14)),
    );
    let xs_field = arrow_schema::Field::new(
        "xs",
        arrow_schema::DataType::List(xs_element_field.clone()),
        true,
    )
    .with_metadata(fid(13));

    let key_field = Arc::new(
        arrow_schema::Field::new("key", arrow_schema::DataType::Utf8, false).with_metadata(fid(16)),
    );
    let value_field = Arc::new(
        arrow_schema::Field::new("value", arrow_schema::DataType::Int32, true)
            .with_metadata(fid(17)),
    );
    let entries_field = Arc::new(arrow_schema::Field::new(
        "entries",
        arrow_schema::DataType::Struct(arrow_schema::Fields::from([
            key_field.clone(),
            value_field.clone(),
        ])),
        false,
    ));
    let mp_field = arrow_schema::Field::new(
        "mp",
        arrow_schema::DataType::Map(entries_field.clone(), false),
        true,
    )
    .with_metadata(fid(15));

    let x_field = Arc::new(
        arrow_schema::Field::new("x", arrow_schema::DataType::Utf8, true).with_metadata(fid(20)),
    );
    let ys_element_field = Arc::new(
        arrow_schema::Field::new("element", arrow_schema::DataType::Int32, true)
            .with_metadata(fid(22)),
    );
    let ys_field = Arc::new(
        arrow_schema::Field::new(
            "ys",
            arrow_schema::DataType::List(ys_element_field.clone()),
            true,
        )
        .with_metadata(fid(21)),
    );
    let inner_field = Arc::new(
        arrow_schema::Field::new(
            "inner",
            arrow_schema::DataType::Struct(arrow_schema::Fields::from([
                x_field.clone(),
                ys_field.clone(),
            ])),
            true,
        )
        .with_metadata(fid(19)),
    );
    let deep_field = arrow_schema::Field::new(
        "deep",
        arrow_schema::DataType::Struct(arrow_schema::Fields::from([inner_field.clone()])),
        true,
    )
    .with_metadata(fid(18));

    let id_col = Arc::new(Int64Array::from_iter_values([1, 2, 3, 4])) as ArrayRef;

    let st_col = Arc::new(arrow_array::StructArray::new(
        arrow_schema::Fields::from([a_field, b_field]),
        vec![
            Arc::new(StringArray::from(vec![Some("a1"), None, None, Some("a4")])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(4)])) as ArrayRef,
        ],
        Some(arrow_buffer::NullBuffer::from(vec![
            true, false, true, true,
        ])),
    )) as ArrayRef;

    let xs_col = Arc::new(arrow_array::ListArray::new(
        xs_element_field,
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0i32, 2, 2, 2, 4])),
        Arc::new(Int32Array::from(vec![1, 2, 4, 5])),
        Some(arrow_buffer::NullBuffer::from(vec![
            true, false, true, true,
        ])),
    )) as ArrayRef;

    let mp_entries = arrow_array::StructArray::new(
        arrow_schema::Fields::from([key_field, value_field]),
        vec![
            Arc::new(StringArray::from(vec!["k", "k2"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 4])) as ArrayRef,
        ],
        None,
    );
    let mp_col = Arc::new(arrow_array::MapArray::new(
        entries_field,
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0i32, 1, 1, 1, 2])),
        mp_entries,
        Some(arrow_buffer::NullBuffer::from(vec![
            true, false, true, true,
        ])),
        false,
    )) as ArrayRef;

    let ys_col = Arc::new(arrow_array::ListArray::new(
        ys_element_field,
        arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0i32, 1, 1, 1, 1])),
        Arc::new(Int32Array::from(vec![1])),
        Some(arrow_buffer::NullBuffer::from(vec![
            true, true, false, true,
        ])),
    )) as ArrayRef;
    let inner_col = Arc::new(arrow_array::StructArray::new(
        arrow_schema::Fields::from([x_field, ys_field]),
        vec![
            Arc::new(StringArray::from(vec![Some("x1"), None, None, None])) as ArrayRef,
            ys_col,
        ],
        Some(arrow_buffer::NullBuffer::from(vec![
            true, false, true, false,
        ])),
    )) as ArrayRef;
    let deep_col = Arc::new(arrow_array::StructArray::new(
        arrow_schema::Fields::from([inner_field]),
        vec![inner_col],
        Some(arrow_buffer::NullBuffer::from(vec![
            true, false, true, true,
        ])),
    )) as ArrayRef;

    vec![
        (id_field, id_col),
        (st_field, st_col),
        (xs_field, xs_col),
        (mp_field, mp_col),
        (deep_field, deep_col),
    ]
}

fn template_test_columns() -> Vec<(arrow_schema::Field, ArrayRef)> {
    let column = |name: &str, data_type: arrow_schema::DataType, array: ArrayRef| {
        (arrow_schema::Field::new(name, data_type, false), array)
    };
    vec![
        column(
            "x",
            arrow_schema::DataType::Int64,
            Arc::new(Int64Array::from_iter_values([10, 20, 30, 40])) as ArrayRef,
        ),
        column(
            "y",
            arrow_schema::DataType::Int64,
            Arc::new(Int64Array::from_iter_values([1, 2, 3, 4])) as ArrayRef,
        ),
        column(
            "z",
            arrow_schema::DataType::Int64,
            Arc::new(Int64Array::from_iter_values([5, 6, 7, 8])) as ArrayRef,
        ),
        column(
            "a",
            arrow_schema::DataType::Utf8,
            Arc::new(StringArray::from(vec!["w", "x", "y", "z"])) as ArrayRef,
        ),
        column(
            "dbl",
            arrow_schema::DataType::Float64,
            Arc::new(Float64Array::from_iter_values([1.0, 2.0, 3.0, 4.0])) as ArrayRef,
        ),
        column(
            "i32",
            arrow_schema::DataType::Int32,
            Arc::new(Int32Array::from_iter_values([100, 200, 300, 400])) as ArrayRef,
        ),
        column(
            "i64",
            arrow_schema::DataType::Int64,
            Arc::new(Int64Array::from_iter_values([1000, 2000, 3000, 4000])) as ArrayRef,
        ),
        column(
            "bool",
            arrow_schema::DataType::Boolean,
            Arc::new(BooleanArray::from(vec![true, false, true, false])) as ArrayRef,
        ),
    ]
}

impl TableTestFixture {
    async fn register_container_manifest_file(&mut self, file_name: &str, size: u64) {
        let current_snapshot = self.table.metadata().current_snapshot().unwrap();
        let current_schema = current_snapshot.schema(self.table.metadata()).unwrap();
        let current_partition_spec = Arc::new(PartitionSpec::unpartition_spec());

        let mut writer = ManifestWriterBuilder::new(
            self.next_manifest_file(),
            Some(current_snapshot.snapshot_id()),
            None,
            current_schema.clone(),
            current_partition_spec.as_ref().clone(),
        )
        .build_v2_data();

        writer
            .add_entry(
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(format!("{}/{file_name}", &self.table_location))
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(size)
                            .record_count(4)
                            .partition(Struct::empty())
                            .key_metadata(None)
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .unwrap();

        let data_file_manifest = writer.write_manifest_file().await.unwrap();

        let mut manifest_list_write = ManifestListWriter::v2(
            self.table
                .file_io()
                .new_output(current_snapshot.manifest_list())
                .unwrap(),
            current_snapshot.snapshot_id(),
            current_snapshot.parent_snapshot_id(),
            current_snapshot.sequence_number(),
        );
        manifest_list_write
            .add_manifests(vec![data_file_manifest].into_iter())
            .unwrap();
        manifest_list_write.close().await.unwrap();
    }
}

fn container_oracle_cases() -> Vec<(&'static str, Predicate, Vec<i64>)> {
    vec![
        ("st IS NULL", Reference::new("st").is_null(), vec![2]),
        ("st IS NOT NULL", Reference::new("st").is_not_null(), vec![
            1, 3, 4,
        ]),
        ("xs IS NULL", Reference::new("xs").is_null(), vec![2]),
        ("xs IS NOT NULL", Reference::new("xs").is_not_null(), vec![
            1, 3, 4,
        ]),
        ("mp IS NULL", Reference::new("mp").is_null(), vec![2]),
        ("mp IS NOT NULL", Reference::new("mp").is_not_null(), vec![
            1, 3, 4,
        ]),
        ("st.a IS NULL", Reference::new("st.a").is_null(), vec![2, 3]),
        (
            "deep.inner IS NULL",
            Reference::new("deep.inner").is_null(),
            vec![2, 4],
        ),
        (
            "deep.inner.x IS NULL",
            Reference::new("deep.inner.x").is_null(),
            vec![2, 3, 4],
        ),
        (
            "deep.inner.ys IS NULL",
            Reference::new("deep.inner.ys").is_null(),
            vec![2, 3, 4],
        ),
        (
            "st IS NULL OR id = 1",
            Reference::new("st")
                .is_null()
                .or(Reference::new("id").equal_to(Datum::long(1))),
            vec![1, 2],
        ),
        (
            "st IS NOT NULL AND id > 2",
            Reference::new("st")
                .is_not_null()
                .and(Reference::new("id").greater_than(Datum::long(2))),
            vec![3, 4],
        ),
    ]
}

async fn scanned_container_ids(
    fixture: &TableTestFixture,
    predicate: Predicate,
    display: &str,
    row_selection_enabled: bool,
) -> Vec<i64> {
    let table_scan = fixture
        .table
        .scan()
        .select(["id"])
        .with_row_selection_enabled(row_selection_enabled)
        .with_filter(predicate)
        .build()
        .unwrap_or_else(|e| panic!("build the scan for `{display}`: {e}"));
    let batches: Vec<_> = table_scan
        .to_arrow()
        .await
        .expect("open the arrow stream")
        .try_collect()
        .await
        .unwrap_or_else(|e| panic!("collect the filtered batches for `{display}`: {e}"));

    let mut ids: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            let col = batch
                .column_by_name("id")
                .expect("scan output carries the id column");
            decode_int64_column(col)
                .iter()
                .map(|value| value.expect("id is a required column"))
                .collect::<Vec<_>>()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn test_filter_on_arrow_container_null_predicates_match_spark_oracle() {
    let mut fixture = TableTestFixture::new_container_columns();
    fixture.setup_container_manifest_files().await;

    for (display, predicate, expected_ids) in container_oracle_cases() {
        let ids = scanned_container_ids(&fixture, predicate, display, false).await;
        assert_eq!(ids, expected_ids, "filter `{display}`");
    }
}

#[tokio::test]
async fn test_container_null_predicates_match_spark_oracle_under_page_index_row_selection() {
    let mut fixture = TableTestFixture::new_container_columns();
    fixture.setup_container_manifest_files().await;

    for (display, predicate, expected_ids) in container_oracle_cases() {
        let ids = scanned_container_ids(&fixture, predicate, display, true).await;
        assert_eq!(
            ids, expected_ids,
            "filter `{display}` under page-index row selection"
        );
    }
}

#[tokio::test]
async fn test_container_null_predicates_match_spark_oracle_without_field_ids() {
    let mut fixture = TableTestFixture::new_container_columns();
    fixture.setup_id_less_container_manifest_files().await;

    for (display, predicate, expected_ids) in container_oracle_cases() {
        let ids = scanned_container_ids(&fixture, predicate, display, false).await;
        assert_eq!(ids, expected_ids, "filter `{display}` without field ids");
    }
}

#[tokio::test]
async fn test_container_null_predicates_match_spark_oracle_with_name_mapping() {
    let mut fixture =
        TableTestFixture::new_container_columns_with_name_mapping(CONTAINER_NAME_MAPPING);
    fixture
        .setup_reordered_id_less_container_manifest_files()
        .await;

    for (display, predicate, expected_ids) in container_oracle_cases() {
        let ids = scanned_container_ids(&fixture, predicate, display, false).await;
        assert_eq!(
            ids, expected_ids,
            "filter `{display}` with name mapping over reordered columns"
        );
    }
}
