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

#[test]
fn unknown_column_readable_metrics_are_all_null_while_known_columns_keep_theirs() {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;

    use crate::arrow::schema_to_arrow_schema;
    use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Datum, Struct};

    let schema = Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "c",
                Type::Primitive(PrimitiveType::Unknown),
            )),
        ])
        .build()
        .expect("schema with unknown column");

    let field = readable_metrics_field(&schema, 10);
    let host_schema = Schema::builder()
        .with_fields(vec![field])
        .build()
        .expect("host schema");
    let arrow = schema_to_arrow_schema(&host_schema).expect("arrow schema");
    let arrow_fields = readable_metrics_struct_fields(&arrow).expect("struct fields");

    let data_file = DataFileBuilder::default()
        .partition_spec_id(0)
        .content(DataContentType::Data)
        .file_path("x".to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(1)
        .record_count(2)
        .partition(Struct::empty())
        .column_sizes(std::collections::HashMap::from([(1, 39u64)]))
        .value_counts(std::collections::HashMap::from([(1, 2u64)]))
        .null_value_counts(std::collections::HashMap::from([(1, 0u64)]))
        .lower_bounds(std::collections::HashMap::from([(1, Datum::long(0))]))
        .upper_bounds(std::collections::HashMap::from([(1, Datum::long(1))]))
        .build()
        .expect("data file");

    let mut builder = ReadableMetricsBuilder::try_new(&arrow_fields, &schema).expect("builder");
    builder.append(&data_file).expect("append metrics");
    let array = builder.finish();

    let unknown_struct = array.column_by_name("c").expect("c metrics").as_struct();
    for metric in [
        "column_size",
        "value_count",
        "null_value_count",
        "nan_value_count",
        "lower_bound",
        "upper_bound",
    ] {
        let child = unknown_struct
            .column_by_name(metric)
            .expect("metric child");
        assert_eq!(
            child.logical_null_count(),
            1,
            "unknown column metric {metric} must be null"
        );
    }
    assert_eq!(
        unknown_struct
            .column_by_name("lower_bound")
            .expect("lower bound")
            .data_type(),
        &arrow_schema::DataType::Null,
        "unknown lower bound carries the Null type"
    );
    assert_eq!(
        unknown_struct
            .column_by_name("upper_bound")
            .expect("upper bound")
            .data_type(),
        &arrow_schema::DataType::Null,
        "unknown upper bound carries the Null type"
    );

    let id_struct = array.column_by_name("id").expect("id metrics").as_struct();
    let count = |name: &str| {
        id_struct
            .column_by_name(name)
            .expect("count child")
            .as_primitive::<Int64Type>()
            .value(0)
    };
    assert_eq!(count("column_size"), 39);
    assert_eq!(count("value_count"), 2);
    assert_eq!(count("null_value_count"), 0);
    assert!(
        id_struct
            .column_by_name("nan_value_count")
            .expect("nan child")
            .is_null(0)
    );
    assert_eq!(
        id_struct
            .column_by_name("lower_bound")
            .expect("lower child")
            .as_primitive::<Int64Type>()
            .value(0),
        0
    );
    assert_eq!(
        id_struct
            .column_by_name("upper_bound")
            .expect("upper child")
            .as_primitive::<Int64Type>()
            .value(0),
        1
    );
}

#[test]
fn nested_unknown_leaf_readable_metrics_are_all_null_while_its_sibling_keeps_its_own() {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;

    use crate::arrow::schema_to_arrow_schema;
    use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Datum, Struct, StructType};

    let schema = Schema::builder()
        .with_schema_id(1)
        .with_fields(vec![
            Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(
                2,
                "s",
                Type::Struct(StructType::new(vec![
                    Arc::new(NestedField::optional(
                        3,
                        "a",
                        Type::Primitive(PrimitiveType::Unknown),
                    )),
                    Arc::new(NestedField::optional(
                        4,
                        "b",
                        Type::Primitive(PrimitiveType::Long),
                    )),
                ])),
            )),
        ])
        .build()
        .expect("schema with nested unknown leaf");

    let field = readable_metrics_field(&schema, 10);
    let host_schema = Schema::builder()
        .with_fields(vec![field])
        .build()
        .expect("host schema");
    let arrow = schema_to_arrow_schema(&host_schema).expect("arrow schema");
    let arrow_fields = readable_metrics_struct_fields(&arrow).expect("struct fields");

    let data_file = DataFileBuilder::default()
        .partition_spec_id(0)
        .content(DataContentType::Data)
        .file_path("x".to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(1)
        .record_count(2)
        .partition(Struct::empty())
        .column_sizes(std::collections::HashMap::from([(1, 39u64), (4, 41u64)]))
        .value_counts(std::collections::HashMap::from([(1, 2u64), (4, 2u64)]))
        .null_value_counts(std::collections::HashMap::from([(1, 0u64), (4, 0u64)]))
        .lower_bounds(std::collections::HashMap::from([
            (1, Datum::long(0)),
            (4, Datum::long(10)),
        ]))
        .upper_bounds(std::collections::HashMap::from([
            (1, Datum::long(1)),
            (4, Datum::long(20)),
        ]))
        .build()
        .expect("data file");

    let mut builder = ReadableMetricsBuilder::try_new(&arrow_fields, &schema).expect("builder");
    builder.append(&data_file).expect("append metrics");
    let array = builder.finish();

    let unknown_leaf = array.column_by_name("s.a").expect("s.a metrics").as_struct();
    for metric in [
        "column_size",
        "value_count",
        "null_value_count",
        "nan_value_count",
        "lower_bound",
        "upper_bound",
    ] {
        assert_eq!(
            unknown_leaf
                .column_by_name(metric)
                .expect("metric child")
                .logical_null_count(),
            1,
            "nested unknown metric {metric} must be null"
        );
    }
    for bound in ["lower_bound", "upper_bound"] {
        assert_eq!(
            unknown_leaf
                .column_by_name(bound)
                .expect("bound child")
                .data_type(),
            &arrow_schema::DataType::Null,
            "nested unknown {bound} carries the Null type"
        );
    }

    let sibling = array.column_by_name("s.b").expect("s.b metrics").as_struct();
    let value = |name: &str| {
        sibling
            .column_by_name(name)
            .expect("sibling child")
            .as_primitive::<Int64Type>()
            .value(0)
    };
    assert_eq!(value("column_size"), 41);
    assert_eq!(value("value_count"), 2);
    assert_eq!(value("null_value_count"), 0);
    assert_eq!(value("lower_bound"), 10);
    assert_eq!(value("upper_bound"), 20);
    assert!(
        sibling
            .column_by_name("nan_value_count")
            .expect("nan child")
            .is_null(0)
    );
}
