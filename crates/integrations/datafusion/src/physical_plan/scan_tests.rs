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

use datafusion::arrow::array::{Array, ArrayRef};
use datafusion::arrow::datatypes::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use iceberg::TableIdent;
use iceberg::io::FileIO;
use iceberg::spec::{
    FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder,
    TableMetadataBuilder, Type,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::*;

pub(crate) fn create_test_table() -> Table {
    let schema = Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::required(
                2,
                "data",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .expect("test schema must build");

    let partition_spec = PartitionSpec::builder(schema.clone())
        .build()
        .expect("partition spec must build");
    let sort_order = SortOrder::builder()
        .build(&schema)
        .expect("sort order must build");
    let table_metadata = TableMetadataBuilder::new(
        schema,
        partition_spec,
        sort_order,
        "memory://test/table".to_string(),
        FormatVersion::V2,
        HashMap::new(),
    )
    .expect("metadata builder must construct")
    .build()
    .expect("table metadata must build");

    Table::builder()
        .metadata(table_metadata.metadata)
        .identifier(TableIdent::from_strs(["test", "table"]).expect("ident must parse"))
        .file_io(FileIO::new_with_memory())
        .metadata_location("memory://test/metadata.json".to_string())
        .build()
        .expect("table must build")
}

pub(crate) fn test_arrow_schema() -> ArrowSchemaRef {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int64, false),
        ArrowField::new("data", ArrowDataType::Utf8, false),
    ]))
}

#[test]
fn test_scan_out_of_bounds_projection_is_error_not_panic() {
    let err = IcebergTableScan::new(
        create_test_table(),
        None,
        false,
        test_arrow_schema(),
        Some(&vec![0, 99]),
        &[],
        None,
    )
    .expect_err("projection index 99 on a 2-column schema must be a planning error");
    assert!(
        err.to_string().contains("99"),
        "the error should name the offending index: {err}"
    );
}

#[test]
fn test_conform_batch_null_fills_and_reorders() {
    let scanned = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("data", ArrowDataType::Utf8, false),
            ArrowField::new("id", ArrowDataType::Int64, false),
        ])),
        vec![
            Arc::new(datafusion::arrow::array::StringArray::from(vec!["a", "b"])),
            Arc::new(datafusion::arrow::array::Int64Array::from(vec![1, 2])),
        ],
    )
    .expect("the scanned batch must build");

    let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int64, false),
        ArrowField::new("data", ArrowDataType::Utf8, false),
        ArrowField::new("added_later", ArrowDataType::Int32, true),
    ]));

    let sources = vec![
        ColumnSource::Scanned("id".to_string()),
        ColumnSource::Scanned("data".to_string()),
        ColumnSource::Absent,
    ];
    let conformed = conform_batch(scanned, &advertised, &sources).expect("the batch must conform");
    assert_eq!(conformed.schema(), advertised);
    assert_eq!(conformed.num_rows(), 2);
    assert_eq!(
        conformed
            .column_by_name("added_later")
            .expect("the added column must be present")
            .null_count(),
        2,
        "a column the scan could not read must be all-NULL"
    );
    assert_eq!(
        conformed
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("column 0 must be the advertised `id`")
            .values(),
        &[1, 2]
    );
}

#[test]
fn test_conform_batch_preserves_row_count_with_no_columns() {
    let scanned = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int64,
            false,
        )])),
        vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
            1, 2, 3,
        ]))],
    )
    .expect("the scanned batch must build");

    let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(Vec::<ArrowField>::new()));
    let conformed = conform_batch(scanned, &advertised, &[]).expect("the batch must conform");
    assert_eq!(conformed.num_columns(), 0);
    assert_eq!(
        conformed.num_rows(),
        3,
        "the row count must survive a zero-column projection"
    );
}

#[test]
fn test_conform_batch_rejects_a_changed_type() {
    let scanned = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int64,
            false,
        )])),
        vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
            1,
        ]))],
    )
    .expect("the scanned batch must build");
    let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        ArrowDataType::Int32,
        false,
    )]));
    let err = conform_batch(scanned, &advertised, &[ColumnSource::Scanned(
        "id".to_string(),
    )])
    .expect_err("an illegal type change must not be silently coerced");
    assert!(
        err.to_string().contains("id") && err.to_string().contains("Int32"),
        "the error must name the column and the expected type: {err}"
    );
}

#[test]
fn test_conform_batch_rejects_absent_required_column() {
    let scanned = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int64,
            false,
        )])),
        vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
            1,
        ]))],
    )
    .expect("the scanned batch must build");

    let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int64, false),
        ArrowField::new("required_new", ArrowDataType::Utf8, false),
    ]));
    let sources = vec![
        ColumnSource::Scanned("id".to_string()),
        ColumnSource::Absent,
    ];
    let err = conform_batch(scanned, &advertised, &sources)
        .expect_err("an absent required column must not be null-filled");
    assert!(
        err.to_string().contains("required_new"),
        "the error must name the column: {err}"
    );
}

#[test]
fn test_arrow_promotion_mirror_agrees_with_iceberg_rule() {
    use iceberg::spec::{PrimitiveType as P, Type as T, is_promotion_allowed};

    let primitives = [
        P::Boolean,
        P::Int,
        P::Long,
        P::Float,
        P::Double,
        P::Decimal {
            precision: 9,
            scale: 2,
        },
        P::Decimal {
            precision: 18,
            scale: 2,
        },
        P::Decimal {
            precision: 18,
            scale: 3,
        },
        P::Date,
        P::Time,
        P::Timestamp,
        P::Timestamptz,
        P::TimestampNs,
        P::TimestamptzNs,
        P::String,
        P::Uuid,
        P::Fixed(16),
        P::Binary,
        P::Unknown,
    ];
    assert_eq!(
        primitives.len(),
        17 + 2,
        "17 variants, with two extra decimals for the precision/scale arm"
    );

    let arrow_of = |primitive: &P| -> ArrowDataType {
        let schema = Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                1,
                "c",
                Type::Primitive(primitive.clone()),
            ))])
            .build()
            .expect("one-field schema must build");
        iceberg::arrow::schema_to_arrow_schema(&schema)
            .expect("schema must convert to arrow")
            .field(0)
            .data_type()
            .clone()
    };

    let mut checked = 0;
    let mut collisions = 0;
    for from in &primitives {
        for to in &primitives {
            let (arrow_from, arrow_to) = (arrow_of(from), arrow_of(to));
            let expected = is_promotion_allowed(&T::Primitive(from.clone()), to);
            let actual = is_arrow_promotion_allowed(&arrow_from, &arrow_to);
            if from != to && arrow_from == arrow_to {
                assert!(
                    actual,
                    "the mirror's identity arm must accept {from} -> {to} (both {arrow_from:?})"
                );
                collisions += 1;
            } else {
                assert_eq!(
                    actual, expected,
                    "mirror disagrees for {from} -> {to} (arrow {arrow_from:?} -> {arrow_to:?})"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, primitives.len() * primitives.len());
    assert!(is_arrow_promotion_allowed(
        &ArrowDataType::Int32,
        &ArrowDataType::Int64
    ));
    assert_eq!(
        collisions, 2,
        "the uuid / fixed[16] collision must be the only one this matrix hits"
    );
}

#[test]
fn test_conform_column_null_fills_a_nested_field() {
    use datafusion::arrow::array::{Int32Array, StructArray};
    use datafusion::arrow::buffer::NullBuffer;
    use datafusion::arrow::datatypes::Fields;

    let scanned_children = Fields::from(vec![field_with_id("a", ArrowDataType::Int32, true, 3)]);
    let scanned: ArrayRef = Arc::new(
        StructArray::try_new(
            scanned_children,
            vec![Arc::new(Int32Array::from(vec![Some(5), Some(6)]))],
            Some(NullBuffer::from(vec![true, false])),
        )
        .expect("the scanned struct must build"),
    );

    let target_children = Fields::from(vec![
        field_with_id("a", ArrowDataType::Int32, true, 3),
        field_with_id("b", ArrowDataType::Int32, true, 4),
    ]);
    let target = ArrowField::new("s", ArrowDataType::Struct(target_children.clone()), true);
    let conformed = conform_column(&scanned, &target, "s").expect("the struct must conform");
    assert_eq!(conformed.data_type(), target.data_type());
    let conformed = conformed
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("still a struct");
    assert_eq!(conformed.len(), 2);
    assert_eq!(
        conformed.column(1).null_count(),
        2,
        "the added child must be all-NULL"
    );
    assert!(
        conformed.is_null(1),
        "the struct's own null buffer must survive"
    );
}

#[test]
fn test_conform_column_recurses_into_a_list_element() {
    use datafusion::arrow::array::{Int32Array, ListArray, StructArray};
    use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
    use datafusion::arrow::datatypes::Fields;

    let scanned_children = Fields::from(vec![field_with_id("a", ArrowDataType::Int32, true, 3)]);
    let scanned_element = Arc::new(ArrowField::new(
        "element",
        ArrowDataType::Struct(scanned_children.clone()),
        true,
    ));
    let scanned_values: ArrayRef = Arc::new(
        StructArray::try_new(
            scanned_children,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
            None,
        )
        .expect("element struct"),
    );
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 2, 3]));
    let scanned: ArrayRef = Arc::new(
        ListArray::try_new(scanned_element, offsets, scanned_values, None)
            .expect("the scanned list must build"),
    );

    let target_children = Fields::from(vec![
        field_with_id("a", ArrowDataType::Int32, true, 3),
        field_with_id("b", ArrowDataType::Int32, true, 4),
    ]);
    let target_element = Arc::new(ArrowField::new(
        "element",
        ArrowDataType::Struct(target_children),
        true,
    ));
    let target = ArrowField::new("l", ArrowDataType::List(target_element), true);

    let conformed = conform_column(&scanned, &target, "l").expect("the list must conform");
    assert_eq!(conformed.data_type(), target.data_type());
    let conformed = conformed
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("still a list");
    assert_eq!(conformed.len(), 2, "the list offsets must be preserved");
    assert_eq!(conformed.value(0).len(), 2);
    assert_eq!(conformed.value(1).len(), 1);
    let values = conformed
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("element struct");
    assert_eq!(
        values.column(1).null_count(),
        3,
        "the added element field must be all-NULL"
    );
}

#[test]
fn test_conform_column_recurses_into_a_map_value() {
    use datafusion::arrow::array::{Int32Array, MapArray, StringArray, StructArray};
    use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
    use datafusion::arrow::datatypes::Fields;

    let value_children = Fields::from(vec![field_with_id("a", ArrowDataType::Int32, true, 3)]);
    let scanned_entry_fields = Fields::from(vec![
        field_with_id("key", ArrowDataType::Utf8, false, 5),
        field_with_id(
            "value",
            ArrowDataType::Struct(value_children.clone()),
            true,
            6,
        ),
    ]);
    let scanned_entries = StructArray::try_new(
        scanned_entry_fields.clone(),
        vec![
            Arc::new(StringArray::from(vec!["k1", "k2"])),
            Arc::new(
                StructArray::try_new(
                    value_children,
                    vec![Arc::new(Int32Array::from(vec![7, 8]))],
                    None,
                )
                .expect("value struct"),
            ),
        ],
        None,
    )
    .expect("entries struct");
    let scanned_entries_field = Arc::new(ArrowField::new(
        "entries",
        ArrowDataType::Struct(scanned_entry_fields),
        false,
    ));
    let scanned: ArrayRef = Arc::new(
        MapArray::try_new(
            scanned_entries_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 2])),
            scanned_entries,
            None,
            false,
        )
        .expect("the scanned map must build"),
    );

    let target_value_children = Fields::from(vec![
        field_with_id("a", ArrowDataType::Int32, true, 3),
        field_with_id("b", ArrowDataType::Int32, true, 4),
    ]);
    let target_entry_fields = Fields::from(vec![
        field_with_id("key", ArrowDataType::Utf8, false, 5),
        field_with_id(
            "value",
            ArrowDataType::Struct(target_value_children),
            true,
            6,
        ),
    ]);
    let target = ArrowField::new(
        "m",
        ArrowDataType::Map(
            Arc::new(ArrowField::new(
                "entries",
                ArrowDataType::Struct(target_entry_fields),
                false,
            )),
            false,
        ),
        true,
    );

    let conformed = conform_column(&scanned, &target, "m").expect("the map must conform");
    assert_eq!(conformed.data_type(), target.data_type());
    let conformed = conformed
        .as_any()
        .downcast_ref::<MapArray>()
        .expect("still a map");
    assert_eq!(conformed.len(), 1);
    assert_eq!(conformed.value_length(0), 2, "the map offsets must survive");
    let values = conformed
        .entries()
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("value struct")
        .clone();
    assert_eq!(
        values.column(1).null_count(),
        2,
        "the added value field must be all-NULL"
    );
}

#[test]
fn test_conform_column_names_the_nested_path_on_an_illegal_change() {
    use datafusion::arrow::array::{Int64Array, StructArray};
    use datafusion::arrow::datatypes::Fields;

    let scanned_children = Fields::from(vec![field_with_id("a", ArrowDataType::Int64, true, 3)]);
    let scanned: ArrayRef = Arc::new(
        StructArray::try_new(
            scanned_children,
            vec![Arc::new(Int64Array::from(vec![5]))],
            None,
        )
        .expect("the scanned struct must build"),
    );
    let target_children = Fields::from(vec![field_with_id("a", ArrowDataType::Int32, true, 3)]);
    let target = ArrowField::new("s", ArrowDataType::Struct(target_children), true);

    let err = conform_column(&scanned, &target, "s")
        .expect_err("a narrowing nested change must not be coerced");
    assert!(
        err.to_string().contains("s.a"),
        "the error must name the nested path: {err}"
    );
}

#[test]
fn test_conform_batch_promotes_a_legal_type_change() {
    let scanned = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int32,
            false,
        )])),
        vec![Arc::new(datafusion::arrow::array::Int32Array::from(vec![
            7, -3,
        ]))],
    )
    .expect("the scanned batch must build");

    let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        ArrowDataType::Int64,
        false,
    )]));

    let conformed = conform_batch(scanned, &advertised, &[ColumnSource::Scanned(
        "id".to_string(),
    )])
    .expect("int -> long is a legal Iceberg promotion");
    assert_eq!(conformed.schema(), advertised);
    assert_eq!(
        conformed
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("the promoted column must be Int64")
            .values(),
        &[7, -3],
        "the values must survive the widening"
    );
}

async fn table_with_snapshot() -> Table {
    let metadata_file_path = format!(
        "{}/tests/test_data/TableMetadataV2Valid.json",
        env!("CARGO_MANIFEST_DIR")
    );
    iceberg::table::StaticTable::from_metadata_file(
        &metadata_file_path,
        iceberg::TableIdent::from_strs(["ns", "t"]).expect("ident must parse"),
        iceberg::io::FileIO::new_with_fs(),
    )
    .await
    .expect("the fixture metadata must load")
    .into_table()
}

fn field_with_id(name: &str, data_type: ArrowDataType, nullable: bool, id: i32) -> ArrowField {
    ArrowField::new(name, data_type, nullable).with_metadata(HashMap::from([(
        PARQUET_FIELD_ID_META_KEY.to_string(),
        id.to_string(),
    )]))
}

#[tokio::test]
async fn test_resolve_projection_requires_a_field_id() {
    let table = table_with_snapshot().await;
    let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        ArrowDataType::Int64,
        false,
    )]));

    let err = resolve_bindings(&table, None, &advertised, false)
        .expect_err("a field without an id must not be bound by name");
    assert!(
        err.to_string().contains(PARQUET_FIELD_ID_META_KEY) && err.to_string().contains('x'),
        "the error must name the column and the missing metadata: {err}"
    );
}

#[tokio::test]
async fn test_resolve_projection_binds_by_field_id_not_by_name() {
    let table = table_with_snapshot().await;
    let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
        field_with_id("renamed_y", ArrowDataType::Int64, false, 2),
        field_with_id("added_later", ArrowDataType::Int32, true, 99),
    ]));

    let bindings =
        resolve_bindings(&table, None, &advertised, false).expect("the bindings must resolve");
    let (scan_columns, sources) =
        project_bindings(&advertised, &bindings).expect("the projection must resolve");
    assert_eq!(
        scan_columns,
        vec!["y".to_string()],
        "the SNAPSHOT schema's name for field 2 is what gets selected"
    );
    assert_eq!(sources, vec![
        ColumnSource::Scanned("y".to_string()),
        ColumnSource::Absent,
    ]);
}

#[test]
fn test_scan_valid_projection_schema_and_names() {
    let projected = IcebergTableScan::new(
        create_test_table(),
        None,
        false,
        test_arrow_schema(),
        Some(&vec![1]),
        &[],
        None,
    )
    .expect("a valid projection must plan");
    assert_eq!(projected.projection(), Some(&["data".to_string()][..]));
    let output_schema = projected.schema();
    assert_eq!(output_schema.fields().len(), 1);
    assert_eq!(output_schema.field(0).name(), "data");
    assert_eq!(output_schema.field(0).data_type(), &ArrowDataType::Utf8);

    let unprojected = IcebergTableScan::new(
        create_test_table(),
        None,
        false,
        test_arrow_schema(),
        None,
        &[],
        None,
    )
    .expect("a scan without projection must plan");
    assert_eq!(unprojected.projection(), None);
    assert_eq!(unprojected.schema(), test_arrow_schema());
}

#[test]
fn test_scan_knobs_from_context_wires_batch_size_and_concurrency() {
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;

    let config = SessionConfig::new()
        .set_usize("datafusion.execution.batch_size", 17)
        .set_usize("datafusion.execution.target_partitions", 5);
    let state = SessionStateBuilder::new().with_config(config).build();
    let context = state.task_ctx();

    let knobs = scan_knobs_from_context(&context);
    assert_eq!(knobs.batch_size, Some(17));
    assert_eq!(knobs.data_file_concurrency, Some(5));
}

#[test]
fn test_clamp_scan_knob_floors_zero_to_one() {
    assert_eq!(clamp_scan_knob(0), 1);
    assert_eq!(clamp_scan_knob(1), 1);
    assert_eq!(clamp_scan_knob(8), 8);
}

#[test]
fn test_get_batch_stream_clamps_zero_knobs_at_apply() {
    let knobs = ScanKnobs {
        batch_size: Some(0),
        data_file_concurrency: Some(0),
        target_partitions: 1,
        multi_partition_scan: true,
        row_selection_enabled: true,
    };
    let effective_batch = knobs.batch_size.map(clamp_scan_knob);
    let effective_conc = knobs.data_file_concurrency.map(clamp_scan_knob);
    assert_eq!(effective_batch, Some(1));
    assert_eq!(effective_conc, Some(1));
}

#[test]
fn test_scan_knobs_clamps_zero_batch_size() {
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;

    let config = SessionConfig::new().set_usize("datafusion.execution.batch_size", 0);
    let state = SessionStateBuilder::new().with_config(config).build();
    let knobs = scan_knobs_from_context(&state.task_ctx());
    assert_eq!(
        knobs.batch_size,
        Some(1),
        "batch_size 0 must clamp to 1 (Parquet empty-stream hazard)"
    );
    assert!(
        knobs.data_file_concurrency.is_some_and(|c| c >= 1),
        "data-file concurrency must stay ≥ 1"
    );
}

#[test]
fn test_scan_knobs_target_partitions_zero_still_at_least_one() {
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;

    let config = SessionConfig::new().set_usize("datafusion.execution.target_partitions", 0);
    let state = SessionStateBuilder::new().with_config(config).build();
    let knobs = scan_knobs_from_context(&state.task_ctx());
    assert!(
        knobs.data_file_concurrency.is_some_and(|c| c >= 1),
        "target_partitions 0 must not yield concurrency 0 (hang hazard)"
    );
}

#[test]
fn test_pin13_off_switch_independent_of_target_partitions() {
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;

    let mut config = SessionConfig::new().set_usize("datafusion.execution.target_partitions", 8);
    ensure_iceberg_scan_options(&mut config);
    config.options_mut().extensions.insert(IcebergScanOptions {
        multi_partition_scan: false,
        data_file_concurrency: 0,
        row_selection_enabled: true,
    });
    let state = SessionStateBuilder::new().with_config(config).build();
    let knobs = scan_knobs_from_context(&state.task_ctx());
    assert!(!knobs.multi_partition_scan);
    assert_eq!(knobs.target_partitions, 8);
    let t = if knobs.multi_partition_scan {
        knobs.target_partitions
    } else {
        1
    };
    assert_eq!(t, 1);
}

#[test]
fn test_pin14_distinct_l_surface() {
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;

    let mut config = SessionConfig::new().set_usize("datafusion.execution.target_partitions", 4);
    ensure_iceberg_scan_options(&mut config);
    config.options_mut().extensions.insert(IcebergScanOptions {
        multi_partition_scan: true,
        data_file_concurrency: 16,
        row_selection_enabled: true,
    });
    let state = SessionStateBuilder::new().with_config(config).build();
    let knobs = scan_knobs_from_context(&state.task_ctx());
    assert_eq!(knobs.target_partitions, 4);
    assert_eq!(knobs.data_file_concurrency, Some(16));

    let mut config2 = SessionConfig::new().set_usize("datafusion.execution.target_partitions", 4);
    config2.options_mut().extensions.insert(IcebergScanOptions {
        multi_partition_scan: true,
        data_file_concurrency: 2,
        row_selection_enabled: true,
    });
    let state2 = SessionStateBuilder::new().with_config(config2).build();
    let knobs2 = scan_knobs_from_context(&state2.task_ctx());
    assert_eq!(knobs2.target_partitions, 4);
    assert_eq!(knobs2.data_file_concurrency, Some(2));
    assert_ne!(knobs.data_file_concurrency, knobs2.data_file_concurrency);
}

#[tokio::test]
async fn test_execute_out_of_range_errors() {
    use datafusion::execution::TaskContext;

    let scan = IcebergTableScan::new(
        create_test_table(),
        None,
        false,
        test_arrow_schema(),
        None,
        &[],
        None,
    )
    .expect("scan");
    let ctx = Arc::new(TaskContext::default());
    match scan.execute(1, ctx) {
        Ok(_) => panic!("i≥N must error, got Ok stream"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("out of range"),
                "expected out-of-range typed error, got: {msg}"
            );
        }
    }
}

#[tokio::test]
async fn test_pin2_execute_out_of_range_multipath() {
    use datafusion::execution::TaskContext;

    let table = create_test_table();
    let knobs = ScanKnobs {
        batch_size: Some(1024),
        data_file_concurrency: Some(4),
        target_partitions: 4,
        multi_partition_scan: true,
        row_selection_enabled: true,
    };
    let scan = IcebergTableScan::plan(
        table,
        None,
        false,
        test_arrow_schema_with_field_ids(),
        None,
        &[],
        Some(10),
        knobs,
    )
    .await
    .expect("empty table plan");
    assert_eq!(scan.partition_work().len(), 1);
    assert_eq!(scan.properties().output_partitioning().partition_count(), 1);
    assert_eq!(scan.limit(), Some(10), "pin 5: limit retained when N=1");
    let ctx = Arc::new(TaskContext::default());
    match scan.execute(1, ctx) {
        Ok(_) => panic!("i≥N multi-path must error"),
        Err(err) => {
            assert!(err.to_string().contains("out of range"), "got: {err}");
        }
    }
}

pub(crate) fn test_arrow_schema_with_field_ids() -> ArrowSchemaRef {
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        ArrowField::new("data", ArrowDataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]))
}
