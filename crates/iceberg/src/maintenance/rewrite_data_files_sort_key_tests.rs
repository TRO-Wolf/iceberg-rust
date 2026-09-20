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

use std::sync::Arc;

use arrow_array::{
    ArrayRef, Decimal128Array, Decimal256Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_buffer::i256;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

use crate::maintenance::rewrite_data_files_sort::ResolvedStrategy;
use crate::maintenance::rewrite_data_files_sort_key::KeyPlan;
use crate::maintenance::rewrite_data_files_zorder::ZOrderEncoder;
use crate::spec::{
    NestedField, NullOrder, PrimitiveType, Schema, SortDirection, SortField, SortOrder, Transform,
    Type,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn schema_of(fields: Vec<(i32, &str, PrimitiveType)>) -> (Schema, Arc<ArrowSchema>) {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(
            fields
                .into_iter()
                .map(|(id, name, primitive)| {
                    Arc::new(NestedField::optional(id, name, Type::Primitive(primitive)))
                })
                .collect::<Vec<_>>(),
        )
        .build()
        .expect("build the key schema");
    let arrow_schema = Arc::new(crate::arrow::schema_to_arrow_schema(&schema).expect("arrow"));
    (schema, arrow_schema)
}

fn ascending(source_id: i32) -> SortField {
    SortField {
        source_id,
        transform: Transform::Identity,
        direction: SortDirection::Ascending,
        null_order: NullOrder::Last,
    }
}

fn descending(source_id: i32) -> SortField {
    SortField {
        source_id,
        transform: Transform::Identity,
        direction: SortDirection::Descending,
        null_order: NullOrder::Last,
    }
}

fn sort_keys(
    schema: &Schema,
    arrow_schema: &Arc<ArrowSchema>,
    fields: Vec<SortField>,
    batch: &RecordBatch,
) -> Vec<Vec<u8>> {
    let plan = KeyPlan::build(
        &ResolvedStrategy::Sort {
            order: SortOrder {
                order_id: 1,
                fields,
            },
            stamp: 0,
        },
        schema,
        arrow_schema,
    )
    .expect("build the sort key plan")
    .expect("a sort strategy always produces a key plan");
    plan.encode(batch).expect("encode the sort keys")
}

fn row_order(keys: &[Vec<u8>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|left, right| keys[*left].cmp(&keys[*right]));
    order
}

#[test]
fn whole_number_keys_place_every_negative_below_every_positive() {
    let (schema, arrow_schema) = schema_of(vec![
        (1, "l", PrimitiveType::Long),
        (2, "i", PrimitiveType::Int),
    ]);
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(vec![
            Some(-1i64),
            Some(0),
            Some(i64::MIN),
            Some(i64::MAX),
            None,
        ])) as ArrayRef,
        Arc::new(Int32Array::from(vec![
            Some(-1i32),
            Some(0),
            Some(i32::MIN),
            Some(i32::MAX),
            None,
        ])) as ArrayRef,
    ])
    .expect("whole-number batch");

    let longs = sort_keys(&schema, &arrow_schema, vec![ascending(1)], &batch);
    assert_eq!(hex(&longs[0]), "007fffffffffffffff");
    assert_eq!(hex(&longs[1]), "008000000000000000");
    assert_eq!(hex(&longs[2]), "000000000000000000");
    assert_eq!(hex(&longs[3]), "00ffffffffffffffff");
    assert_eq!(hex(&longs[4]), "01");
    assert_eq!(row_order(&longs), vec![2, 0, 1, 3, 4]);

    let ints = sort_keys(&schema, &arrow_schema, vec![ascending(2)], &batch);
    assert_eq!(hex(&ints[0]), "007fffffffffffffff");
    assert_eq!(hex(&ints[1]), "008000000000000000");
    assert_eq!(row_order(&ints), vec![2, 0, 1, 3, 4]);

    let descending_longs = sort_keys(&schema, &arrow_schema, vec![descending(1)], &batch);
    assert_eq!(row_order(&descending_longs), vec![3, 1, 0, 2, 4]);
}

#[test]
fn variable_length_keys_escape_their_zero_bytes_and_terminate() {
    let (schema, arrow_schema) = schema_of(vec![
        (1, "s", PrimitiveType::String),
        (2, "id", PrimitiveType::Long),
    ]);
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(StringArray::from(vec![
            Some("a"),
            Some("a\u{0}b"),
            Some("ab"),
            Some("a\u{0}"),
        ])) as ArrayRef,
        Arc::new(Int64Array::from(vec![
            Some(1i64),
            Some(2),
            Some(3),
            Some(4),
        ])) as ArrayRef,
    ])
    .expect("string batch");

    let keys = sort_keys(&schema, &arrow_schema, vec![ascending(1)], &batch);
    assert_eq!(hex(&keys[0]), "00610000");
    assert_eq!(hex(&keys[1]), "006100ff620000");
    assert_eq!(hex(&keys[2]), "0061620000");
    assert_eq!(hex(&keys[3]), "006100ff0000");
    assert_eq!(row_order(&keys), vec![0, 3, 1, 2]);

    let two_fields = sort_keys(
        &schema,
        &arrow_schema,
        vec![ascending(1), ascending(2)],
        &batch,
    );
    assert_eq!(
        row_order(&two_fields),
        vec![0, 3, 1, 2],
        "a terminated, escaped key orders 'a' < 'a\\0' < 'a\\0b' < 'ab' whatever follows it"
    );

    let descending_first = sort_keys(
        &schema,
        &arrow_schema,
        vec![descending(1), ascending(2)],
        &batch,
    );
    assert_eq!(
        row_order(&descending_first),
        vec![2, 1, 3, 0],
        "a descending variable-length key needs its terminator to keep 'ab' above 'a'"
    );
}

#[test]
fn the_max_output_size_cap_bounds_the_interleaved_z_value() {
    let (schema, arrow_schema) = schema_of(vec![
        (1, "a", PrimitiveType::Long),
        (2, "b", PrimitiveType::Long),
    ]);
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int64Array::from(vec![Some(0i64), Some(0), Some(1)])) as ArrayRef,
        Arc::new(Int64Array::from(vec![Some(0i64), Some(1), Some(0)])) as ArrayRef,
    ])
    .expect("z batch");

    let names = ["a".to_string(), "b".to_string()];
    let mut capped = vec![Vec::new(); batch.num_rows()];
    ZOrderEncoder::build(&names, &schema, &arrow_schema, 8, 4)
        .expect("capped encoder")
        .encode(&batch, &mut capped)
        .expect("encode the capped z values");
    for (row, key) in capped.iter().enumerate() {
        assert_eq!(
            key.len(),
            4,
            "row {row}: 'max-output-size' 4 caps the interleaved value at 4 bytes"
        );
    }
    assert_eq!(capped[0], capped[1]);
    assert_eq!(capped[0], capped[2]);

    let mut uncapped = vec![Vec::new(); batch.num_rows()];
    ZOrderEncoder::build(&names, &schema, &arrow_schema, 8, i32::MAX as usize)
        .expect("uncapped encoder")
        .encode(&batch, &mut uncapped)
        .expect("encode the uncapped z values");
    for (row, key) in uncapped.iter().enumerate() {
        assert_eq!(
            key.len(),
            16,
            "row {row}: two long columns interleave to 16"
        );
    }
    assert!(uncapped[0] < uncapped[1] && uncapped[1] < uncapped[2]);
    assert_eq!(hex(&capped[0]), hex(&uncapped[0][..4]));
}

#[test]
fn wide_decimal_keys_sort_big_endian_with_a_flipped_sign() {
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![Arc::new(NestedField::optional(
            1,
            "d",
            Type::Primitive(PrimitiveType::Decimal {
                precision: 38,
                scale: 0,
            }),
        ))])
        .build()
        .expect("build the decimal schema");
    let values: [i128; 5] = [i128::from(i64::MIN), -1, 0, 1, i128::from(i64::MAX)];

    let wide_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "d",
        DataType::Decimal256(38, 0),
        true,
    )]));
    let wide = RecordBatch::try_new(wide_schema.clone(), vec![Arc::new(
        Decimal256Array::from(
            values
                .iter()
                .copied()
                .map(i256::from_i128)
                .collect::<Vec<i256>>(),
        )
        .with_precision_and_scale(38, 0)
        .expect("decimal256 array"),
    ) as ArrayRef])
    .expect("decimal256 batch");
    let wide_keys = sort_keys(&schema, &wide_schema, vec![ascending(1)], &wide);
    assert_eq!(row_order(&wide_keys), vec![0, 1, 2, 3, 4]);
    assert_eq!(
        hex(&wide_keys[1]),
        "007fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    );
    assert_eq!(
        hex(&wide_keys[2]),
        "008000000000000000000000000000000000000000000000000000000000000000"
    );
    assert_eq!(
        hex(&wide_keys[3]),
        "008000000000000000000000000000000000000000000000000000000000000001"
    );

    let narrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "d",
        DataType::Decimal128(38, 0),
        true,
    )]));
    let narrow = RecordBatch::try_new(narrow_schema.clone(), vec![Arc::new(
        Decimal128Array::from(values.to_vec())
            .with_precision_and_scale(38, 0)
            .expect("decimal128 array"),
    ) as ArrayRef])
    .expect("decimal128 batch");
    let narrow_keys = sort_keys(&schema, &narrow_schema, vec![ascending(1)], &narrow);
    assert_eq!(row_order(&narrow_keys), vec![0, 1, 2, 3, 4]);
    assert_eq!(hex(&narrow_keys[2]), "0080000000000000000000000000000000");
}
