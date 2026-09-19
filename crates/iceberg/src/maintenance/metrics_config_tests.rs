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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, ListArray, RecordBatch, StringArray,
    StructArray,
};
use arrow_buffer::OffsetBuffer;

use crate::arrow::schema_to_arrow_schema;
use crate::maintenance::convert_equality_delete_files::ConvertEqualityDeleteFiles;
use crate::maintenance::rewrite_data_files::RewriteDataFiles;
use crate::maintenance::rewrite_data_files::tests::{
    add_deletes, append_files, local_fs_catalog, write_data_file, write_equality_delete_file,
};
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, ListType, MetricsConfig,
    NestedField, NullOrder, PartitionSpec, PrimitiveType, Schema, SortDirection, SortField,
    SortOrder, StructType, Transform, Type,
};
use crate::table::Table;
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Catalog, NamespaceIdent, TableCreation};

const METRICS_DEFAULT_KEY: &str = "write.metadata.metrics.default";
const METRICS_MAX_INFERRED_KEY: &str = "write.metadata.metrics.max-inferred-column-defaults";

fn oracle_schema() -> Schema {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "s", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(3, "d", Type::Primitive(PrimitiveType::Double)).into(),
            NestedField::required(
                4,
                "st",
                Type::Struct(StructType::new(vec![
                    NestedField::required(6, "a", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(7, "b", Type::Primitive(PrimitiveType::Int)).into(),
                ])),
            )
            .into(),
            NestedField::required(
                5,
                "xs",
                Type::List(ListType::new(
                    NestedField::required(8, "element", Type::Primitive(PrimitiveType::Int)).into(),
                )),
            )
            .into(),
        ])
        .build()
        .unwrap()
}

fn oracle_batch(schema: &Schema) -> RecordBatch {
    let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
    let st_children = match arrow_schema.field(3).data_type() {
        arrow_schema::DataType::Struct(fields) => fields.clone(),
        other => panic!("st field is not struct: {other:?}"),
    };
    let st = StructArray::from(vec![
        (
            st_children[0].clone(),
            Arc::new(StringArray::from(vec!["aa", "zz"])) as ArrayRef,
        ),
        (
            st_children[1].clone(),
            Arc::new(Int32Array::from(vec![1, 3])) as ArrayRef,
        ),
    ]);
    let xs_child = match arrow_schema.field(4).data_type() {
        arrow_schema::DataType::List(field) => field.clone(),
        other => panic!("xs field is not list: {other:?}"),
    };
    let xs = ListArray::new(
        xs_child,
        OffsetBuffer::new(vec![0, 2, 3].into()),
        Arc::new(Int32Array::from(vec![1, 2, 3])),
        None,
    );
    RecordBatch::try_new(arrow_schema, vec![
        Arc::new(Int64Array::from(vec![1, 3])) as ArrayRef,
        Arc::new(StringArray::from(vec![
            "alpha-long-string-value-0001",
            "zulu-long-string-value-0003",
        ])) as ArrayRef,
        Arc::new(Float64Array::from(vec![1.5, 2.5])) as ArrayRef,
        Arc::new(st) as ArrayRef,
        Arc::new(xs) as ArrayRef,
    ])
    .unwrap()
}

async fn create_oracle_table(
    catalog: &impl Catalog,
    properties: &[(&str, &str)],
    sort_order: Option<SortOrder>,
) -> Table {
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let builder = TableCreation::builder()
        .name("t".to_string())
        .schema(oracle_schema())
        .properties(
            properties
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string())),
        )
        .format_version(FormatVersion::V2);
    let creation = match sort_order {
        Some(order) => builder.sort_order(order).build(),
        None => builder.build(),
    };
    catalog
        .create_table(&namespace, creation)
        .await
        .unwrap()
}

fn sort_by(source_id: i32) -> SortOrder {
    SortOrder::builder()
        .with_sort_field(
            SortField::builder()
                .source_id(source_id)
                .transform(Transform::Identity)
                .direction(SortDirection::Ascending)
                .null_order(NullOrder::First)
                .build(),
        )
        .build_unbound()
        .unwrap()
}

async fn write_oracle_file(table: &Table, config: Option<MetricsConfig>) -> DataFile {
    let schema = table.metadata().current_schema().clone();
    let batch = oracle_batch(&schema);
    let location_gen = DefaultLocationGenerator::new(table.metadata().clone()).unwrap();
    let file_name_gen = DefaultFileNameGenerator::new(
        "oracle".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let mut parquet_builder = ParquetWriterBuilder::new(
        parquet::file::properties::WriterProperties::builder().build(),
        schema.clone(),
    );
    if let Some(config) = config {
        parquet_builder = parquet_builder.with_metrics_config(config);
    }
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let mut writer = DataFileWriterBuilder::new(rolling)
        .unpartitioned()
        .build(None)
        .await
        .unwrap();
    writer.write(batch).await.unwrap();
    writer
        .close()
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn bound_hex(map: &HashMap<i32, Datum>) -> Vec<(i32, String)> {
    let mut out: Vec<(i32, String)> = map
        .iter()
        .map(|(id, datum)| (*id, hex(datum.to_bytes().unwrap().as_ref())))
        .collect();
    out.sort();
    out
}

fn assert_maps(file: &DataFile, expected: &CellExpect) {
    let size_keys: HashSet<i32> = file.column_sizes().keys().copied().collect();
    assert_eq!(
        size_keys,
        expected.column_size_keys.iter().copied().collect(),
        "column_sizes keys"
    );
    let expected_vc: HashMap<i32, u64> = expected.value_counts.iter().copied().collect();
    assert_eq!(*file.value_counts(), expected_vc, "value_counts");
    let expected_nvc: HashMap<i32, u64> = expected.null_value_counts.iter().copied().collect();
    assert_eq!(*file.null_value_counts(), expected_nvc, "null_value_counts");
    let expected_nan: HashMap<i32, u64> = expected.nan_value_counts.iter().copied().collect();
    assert_eq!(*file.nan_value_counts(), expected_nan, "nan_value_counts");
    let expected_lower: Vec<(i32, String)> = expected
        .lower_bounds
        .iter()
        .map(|(id, h)| (*id, (*h).to_string()))
        .collect();
    assert_eq!(bound_hex(file.lower_bounds()), expected_lower, "lower_bounds");
    let expected_upper: Vec<(i32, String)> = expected
        .upper_bounds
        .iter()
        .map(|(id, h)| (*id, (*h).to_string()))
        .collect();
    assert_eq!(bound_hex(file.upper_bounds()), expected_upper, "upper_bounds");
}

struct CellExpect {
    column_size_keys: &'static [i32],
    value_counts: &'static [(i32, u64)],
    null_value_counts: &'static [(i32, u64)],
    nan_value_counts: &'static [(i32, u64)],
    lower_bounds: &'static [(i32, &'static str)],
    upper_bounds: &'static [(i32, &'static str)],
}

const SIZES_ALL: &[i32] = &[1, 2, 3, 6, 7, 8];
const VC_ALL: &[(i32, u64)] = &[(1, 2), (2, 2), (3, 2), (6, 2), (7, 2)];
const NVC_ALL: &[(i32, u64)] = &[(1, 0), (2, 0), (3, 0), (6, 0), (7, 0)];
const NAN_D: &[(i32, u64)] = &[(3, 0)];
const EMPTY: &[(i32, u64)] = &[];
const NO_SIZES: &[i32] = &[];
const LOWER_DEFAULT: &[(i32, &str)] = &[
    (1, "0100000000000000"),
    (2, "616c7068612d6c6f6e672d737472696e"),
    (3, "000000000000f83f"),
    (6, "6161"),
    (7, "01000000"),
];
const UPPER_DEFAULT: &[(i32, &str)] = &[
    (1, "0300000000000000"),
    (2, "7a756c752d6c6f6e672d737472696e68"),
    (3, "0000000000000440"),
    (6, "7a7a"),
    (7, "03000000"),
];
const NO_BOUNDS: &[(i32, &str)] = &[];

async fn run_oracle_cell(
    properties: &[(&str, &str)],
    sort_order: Option<SortOrder>,
    expected: &CellExpect,
) {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table = create_oracle_table(&catalog, properties, sort_order).await;
    let file = write_oracle_file(&table, Some(MetricsConfig::for_table(table.metadata()))).await;
    assert_eq!(file.record_count(), 2);
    assert_maps(&file, expected);
}

#[tokio::test]
async fn oracle_cell_default() {
    run_oracle_cell(&[], None, &CellExpect {
        column_size_keys: SIZES_ALL,
        value_counts: VC_ALL,
        null_value_counts: NVC_ALL,
        nan_value_counts: NAN_D,
        lower_bounds: LOWER_DEFAULT,
        upper_bounds: UPPER_DEFAULT,
    })
    .await;
}

#[tokio::test]
async fn oracle_cell_none() {
    run_oracle_cell(&[(METRICS_DEFAULT_KEY, "none")], None, &CellExpect {
        column_size_keys: NO_SIZES,
        value_counts: EMPTY,
        null_value_counts: EMPTY,
        nan_value_counts: EMPTY,
        lower_bounds: NO_BOUNDS,
        upper_bounds: NO_BOUNDS,
    })
    .await;
}

#[tokio::test]
async fn oracle_cell_counts() {
    run_oracle_cell(&[(METRICS_DEFAULT_KEY, "counts")], None, &CellExpect {
        column_size_keys: SIZES_ALL,
        value_counts: VC_ALL,
        null_value_counts: NVC_ALL,
        nan_value_counts: NAN_D,
        lower_bounds: NO_BOUNDS,
        upper_bounds: NO_BOUNDS,
    })
    .await;
}

#[tokio::test]
async fn oracle_cell_truncate4() {
    run_oracle_cell(&[(METRICS_DEFAULT_KEY, "truncate(4)")], None, &CellExpect {
        column_size_keys: SIZES_ALL,
        value_counts: VC_ALL,
        null_value_counts: NVC_ALL,
        nan_value_counts: NAN_D,
        lower_bounds: &[
            (1, "0100000000000000"),
            (2, "616c7068"),
            (3, "000000000000f83f"),
            (6, "6161"),
            (7, "01000000"),
        ],
        upper_bounds: &[
            (1, "0300000000000000"),
            (2, "7a756c76"),
            (3, "0000000000000440"),
            (6, "7a7a"),
            (7, "03000000"),
        ],
    })
    .await;
}

#[tokio::test]
async fn oracle_cell_full() {
    run_oracle_cell(&[(METRICS_DEFAULT_KEY, "full")], None, &CellExpect {
        column_size_keys: SIZES_ALL,
        value_counts: VC_ALL,
        null_value_counts: NVC_ALL,
        nan_value_counts: NAN_D,
        lower_bounds: &[
            (1, "0100000000000000"),
            (2, "616c7068612d6c6f6e672d737472696e672d76616c75652d30303031"),
            (3, "000000000000f83f"),
            (6, "6161"),
            (7, "01000000"),
        ],
        upper_bounds: &[
            (1, "0300000000000000"),
            (2, "7a756c752d6c6f6e672d737472696e672d76616c75652d30303033"),
            (3, "0000000000000440"),
            (6, "7a7a"),
            (7, "03000000"),
        ],
    })
    .await;
}

#[tokio::test]
async fn oracle_cell_column_s_none() {
    run_oracle_cell(
        &[("write.metadata.metrics.column.s", "none")],
        None,
        &CellExpect {
            column_size_keys: &[1, 3, 6, 7, 8],
            value_counts: &[(1, 2), (3, 2), (6, 2), (7, 2)],
            null_value_counts: &[(1, 0), (3, 0), (6, 0), (7, 0)],
            nan_value_counts: NAN_D,
            lower_bounds: &[
                (1, "0100000000000000"),
                (3, "000000000000f83f"),
                (6, "6161"),
                (7, "01000000"),
            ],
            upper_bounds: &[
                (1, "0300000000000000"),
                (3, "0000000000000440"),
                (6, "7a7a"),
                (7, "03000000"),
            ],
        },
    )
    .await;
}

#[tokio::test]
async fn oracle_cell_nested_override() {
    run_oracle_cell(
        &[
            (METRICS_DEFAULT_KEY, "none"),
            ("write.metadata.metrics.column.st.a", "full"),
        ],
        None,
        &CellExpect {
            column_size_keys: &[6],
            value_counts: &[(6, 2)],
            null_value_counts: &[(6, 0)],
            nan_value_counts: EMPTY,
            lower_bounds: &[(6, "6161")],
            upper_bounds: &[(6, "7a7a")],
        },
    )
    .await;
}

#[tokio::test]
async fn oracle_cell_max_inferred_2() {
    run_oracle_cell(&[(METRICS_MAX_INFERRED_KEY, "2")], None, &CellExpect {
        column_size_keys: &[1, 2],
        value_counts: &[(1, 2), (2, 2)],
        null_value_counts: &[(1, 0), (2, 0)],
        nan_value_counts: EMPTY,
        lower_bounds: &[
            (1, "0100000000000000"),
            (2, "616c7068612d6c6f6e672d737472696e"),
        ],
        upper_bounds: &[
            (1, "0300000000000000"),
            (2, "7a756c752d6c6f6e672d737472696e68"),
        ],
    })
    .await;
}

#[tokio::test]
async fn oracle_cell_max_inferred_2_default_set() {
    run_oracle_cell(
        &[
            (METRICS_MAX_INFERRED_KEY, "2"),
            (METRICS_DEFAULT_KEY, "counts"),
        ],
        None,
        &CellExpect {
            column_size_keys: SIZES_ALL,
            value_counts: VC_ALL,
            null_value_counts: NVC_ALL,
            nan_value_counts: NAN_D,
            lower_bounds: NO_BOUNDS,
            upper_bounds: NO_BOUNDS,
        },
    )
    .await;
}

#[tokio::test]
async fn oracle_cell_sorted_none() {
    run_oracle_cell(
        &[(METRICS_DEFAULT_KEY, "none")],
        Some(sort_by(2)),
        &CellExpect {
            column_size_keys: &[2],
            value_counts: &[(2, 2)],
            null_value_counts: &[(2, 0)],
            nan_value_counts: EMPTY,
            lower_bounds: &[(2, "616c7068612d6c6f6e672d737472696e")],
            upper_bounds: &[(2, "7a756c752d6c6f6e672d737472696e68")],
        },
    )
    .await;
}

#[tokio::test]
async fn oracle_cell_sorted_counts() {
    run_oracle_cell(
        &[(METRICS_DEFAULT_KEY, "counts")],
        Some(sort_by(3)),
        &CellExpect {
            column_size_keys: SIZES_ALL,
            value_counts: VC_ALL,
            null_value_counts: NVC_ALL,
            nan_value_counts: NAN_D,
            lower_bounds: &[(3, "000000000000f83f")],
            upper_bounds: &[(3, "0000000000000440")],
        },
    )
    .await;
}

#[tokio::test]
async fn oracle_cell_bad_mode() {
    run_oracle_cell(&[(METRICS_DEFAULT_KEY, "bogus")], None, &CellExpect {
        column_size_keys: SIZES_ALL,
        value_counts: VC_ALL,
        null_value_counts: NVC_ALL,
        nan_value_counts: NAN_D,
        lower_bounds: LOWER_DEFAULT,
        upper_bounds: UPPER_DEFAULT,
    })
    .await;
}

async fn live_files(table: &Table, content: DataContentType) -> Vec<DataFile> {
    let snapshot = table.metadata().current_snapshot().expect("snapshot");
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let mut out = Vec::new();
    for manifest_file in manifest_list.entries() {
        let manifest = manifest_file
            .load_manifest(table.file_io())
            .await
            .expect("manifest");
        for entry in manifest.entries() {
            if entry.is_alive() && entry.content_type() == content {
                out.push(entry.data_file().clone());
            }
        }
    }
    out
}

#[tokio::test]
async fn rewrite_data_files_honors_metrics_default_none() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let table =
        create_oracle_table(&catalog, &[(METRICS_DEFAULT_KEY, "none")], None).await;
    let input = write_oracle_file(&table, None).await;
    assert!(
        !input.lower_bounds().is_empty(),
        "input file must carry bounds so the rewrite proves table-config behavior"
    );
    let table = append_files(&catalog, &table, vec![input]).await;

    let result = RewriteDataFiles::new(table.clone())
        .rewrite_all(true)
        .execute(&catalog)
        .await
        .expect("rewrite must succeed");
    assert_eq!(result.rewritten_data_files_count, 1);
    assert_eq!(result.added_data_files_count, 1);

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let files = live_files(&reloaded, DataContentType::Data).await;
    assert_eq!(files.len(), 1);
    assert_maps(&files[0], &CellExpect {
        column_size_keys: NO_SIZES,
        value_counts: EMPTY,
        null_value_counts: EMPTY,
        nan_value_counts: EMPTY,
        lower_bounds: NO_BOUNDS,
        upper_bounds: NO_BOUNDS,
    });
}

#[tokio::test]
async fn position_delete_keeps_full_bounds_under_none_default() {
    let (catalog, _tmp) = local_fs_catalog().await;
    let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::new_v4()));
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()
        .unwrap();
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("x", "x", Transform::Identity)
        .unwrap()
        .build()
        .unwrap();
    let creation = TableCreation::builder()
        .name("t".to_string())
        .schema(schema)
        .partition_spec(spec)
        .properties([(METRICS_DEFAULT_KEY.to_string(), "none".to_string())])
        .format_version(FormatVersion::V2)
        .build();
    let table = catalog
        .create_table(&namespace, creation)
        .await
        .unwrap();

    let data = write_data_file(&table, "d.parquet", 0, &[(0, 10, 100), (0, 20, 200)]).await;
    let data_path = data.file_path().to_string();
    let table = append_files(&catalog, &table, vec![data]).await;
    let eq_delete = write_equality_delete_file(&table, 0, &[10]).await;
    let table = add_deletes(&catalog, &table, vec![eq_delete]).await;

    ConvertEqualityDeleteFiles::new(table.clone())
        .execute(&catalog)
        .await
        .expect("convert must succeed");

    let reloaded = catalog.load_table(table.identifier()).await.unwrap();
    let deletes = live_files(&reloaded, DataContentType::PositionDeletes).await;
    assert_eq!(deletes.len(), 1);
    let file = &deletes[0];
    let reserved: HashSet<i32> = HashSet::from([
        RESERVED_FIELD_ID_DELETE_FILE_PATH,
        RESERVED_FIELD_ID_DELETE_FILE_POS,
    ]);
    assert_eq!(
        file.column_sizes().keys().copied().collect::<HashSet<_>>(),
        reserved
    );
    assert_eq!(
        file.value_counts().keys().copied().collect::<HashSet<_>>(),
        reserved
    );
    assert_eq!(
        file.lower_bounds().keys().copied().collect::<HashSet<_>>(),
        reserved
    );
    assert_eq!(
        file.upper_bounds().keys().copied().collect::<HashSet<_>>(),
        reserved
    );
    assert_eq!(*file.value_counts().get(&RESERVED_FIELD_ID_DELETE_FILE_PATH).unwrap(), 1);
    let path_bound = file
        .lower_bounds()
        .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        .expect("file_path lower bound")
        .to_bytes()
        .unwrap();
    assert_eq!(
        path_bound.as_ref(),
        data_path.as_bytes(),
        "file_path bound must be the FULL untruncated path"
    );
    assert!(file.nan_value_counts().is_empty());
}
