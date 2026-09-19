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
    ArrayRef, BinaryArray, Decimal128Array, Float32Array, Float64Array, Int32Array, RecordBatch,
};
use arrow_schema::{DataType, Schema as ArrowSchema};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::page_prune_fixture::*;
use crate::expr::Reference;
use crate::spec::{Datum, NestedField, PrimitiveLiteral, PrimitiveType, Type};

#[tokio::test]
async fn s_added_column_predicates_match_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = iceberg_schema(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "added", Type::Primitive(PrimitiveType::String)),
    ]);
    let is_null = bound(&schema, Reference::new("added").is_null());
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2], Some(is_null))).await;
    assert_eq!(rows.len(), ROWS);
    let eq = bound(
        &schema,
        Reference::new("added").equal_to(Datum::string("x")),
    );
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2], Some(eq))).await;
    assert_eq!(rows.len(), 0);
    let not_null = bound(&schema, Reference::new("added").is_not_null());
    let rows = on_off(task(&data_path, schema, &[1, 2], Some(not_null))).await;
    assert_eq!(rows.len(), 0);
}

#[tokio::test]
async fn s_renamed_column_predicate_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = iceberg_schema(vec![NestedField::required(
        1,
        "renamed_id",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let predicate = bound(
        &schema,
        Reference::new("renamed_id").greater_than_or_equal_to(Datum::int(256)),
    );
    let rows = on_off(task(&data_path, schema, &[1], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 256);
}

#[tokio::test]
async fn s_readded_name_with_new_field_id_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = iceberg_schema(vec![NestedField::optional(
        9,
        "id",
        Type::Primitive(PrimitiveType::Int),
    )]);
    let is_null = bound(&schema, Reference::new("id").is_null());
    let rows = on_off(task(&data_path, schema.clone(), &[9], Some(is_null))).await;
    assert_eq!(rows.len(), ROWS);
    let eq = bound(&schema, Reference::new("id").equal_to(Datum::int(300)));
    let rows = on_off(task(&data_path, schema, &[9], Some(eq))).await;
    assert_eq!(rows.len(), 0);
}

#[tokio::test]
async fn s_int_to_long_promotion_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_pages(&data_path, &ids);
    let schema = iceberg_schema(vec![NestedField::required(
        1,
        "id",
        Type::Primitive(PrimitiveType::Long),
    )]);
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::long(256)),
    );
    let rows = on_off(task(&data_path, schema, &[1], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 256);
}

#[tokio::test]
async fn s_float_to_double_promotion_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "f",
        DataType::Float32,
        false,
        1,
    )]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Float32Array::from(
        (0..ROWS).map(|i| i as f32).collect::<Vec<f32>>(),
    )) as ArrayRef])
    .expect("batch");
    write_parquet(&data_path, arrow_schema, &[batch], page_props());
    let schema = iceberg_schema(vec![NestedField::required(
        1,
        "f",
        Type::Primitive(PrimitiveType::Double),
    )]);
    let predicate = bound(
        &schema,
        Reference::new("f").greater_than_or_equal_to(Datum::double(256.0)),
    );
    let rows = on_off(task(&data_path, schema, &[1], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 256);
}

#[tokio::test]
async fn s_decimal_on_fixed_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "d",
        DataType::Decimal128(38, 2),
        false,
        1,
    )]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
        Decimal128Array::from((0..ROWS as i128).collect::<Vec<i128>>())
            .with_precision_and_scale(38, 2)
            .expect("decimal"),
    ) as ArrayRef])
    .expect("batch");
    write_parquet(&data_path, arrow_schema, &[batch], page_props());
    let schema = iceberg_schema(vec![NestedField::required(
        1,
        "d",
        Type::Primitive(PrimitiveType::Decimal {
            precision: 38,
            scale: 2,
        }),
    )]);
    let predicate = bound(
        &schema,
        Reference::new("d").greater_than_or_equal_to(Datum::new(
            PrimitiveType::Decimal {
                precision: 38,
                scale: 2,
            },
            PrimitiveLiteral::Int128(256),
        )),
    );
    let selection =
        page_selection(&file_metadata(&data_path), &schema, &predicate, &None).expect("selection");
    assert_prunes(&selection);
    let rows = on_off(task(&data_path, schema, &[1], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 256);
}

#[tokio::test]
async fn s_decimal_widening_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "d",
        DataType::Decimal128(10, 2),
        false,
        1,
    )]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(
        Decimal128Array::from((0..ROWS as i128).collect::<Vec<i128>>())
            .with_precision_and_scale(10, 2)
            .expect("decimal"),
    ) as ArrayRef])
    .expect("batch");
    write_parquet(&data_path, arrow_schema, &[batch], page_props());
    let schema = iceberg_schema(vec![NestedField::required(
        1,
        "d",
        Type::Primitive(PrimitiveType::Decimal {
            precision: 12,
            scale: 2,
        }),
    )]);
    let predicate = bound(
        &schema,
        Reference::new("d").greater_than_or_equal_to(Datum::new(
            PrimitiveType::Decimal {
                precision: 12,
                scale: 2,
            },
            PrimitiveLiteral::Int128(256),
        )),
    );
    let selection =
        page_selection(&file_metadata(&data_path), &schema, &predicate, &None).expect("selection");
    assert_prunes(&selection);
    let rows = on_off(task(&data_path, schema, &[1], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 256);
}

fn write_null_pages(path: &str) {
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let strings: Vec<Option<String>> = (0..ROWS)
        .map(|i| {
            if i < 128 {
                None
            } else if i < 256 {
                (i % 2 == 0).then(|| format!("v{i}"))
            } else {
                Some(format!("v{i}"))
            }
        })
        .collect();
    write_id_s_pages(path, &ids, &strings);
}

#[tokio::test]
async fn n_is_null_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    write_null_pages(&data_path);
    let schema = id_s_schema();
    let predicate = bound(&schema, Reference::new("s").is_null());
    let metadata = file_metadata(&data_path);
    let selection = page_selection(&metadata, &schema, &predicate, &None).expect("selection");
    assert_prunes(&selection);
    let rows = on_off(task(&data_path, schema, &[1, 2], Some(predicate))).await;
    assert_eq!(rows.len(), 128 + 64);
}

#[tokio::test]
async fn n_is_not_null_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    write_null_pages(&data_path);
    let schema = id_s_schema();
    let predicate = bound(&schema, Reference::new("s").is_not_null());
    let selection =
        page_selection(&file_metadata(&data_path), &schema, &predicate, &None).expect("selection");
    assert_prunes(&selection);
    let rows = on_off(task(&data_path, schema, &[1, 2], Some(predicate))).await;
    assert_eq!(rows.len(), 64 + 256);
}

#[tokio::test]
async fn n_all_null_pages_skipped_for_lt_declared_divergence() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    write_null_pages(&data_path);
    let schema = id_s_schema();
    let predicate = bound(
        &schema,
        Reference::new("s").less_than(Datum::string("v999")),
    );
    let metadata = file_metadata(&data_path);
    let ci = metadata.column_index().expect("column index");
    if let parquet::file::page_index::column_index::ColumnIndexMetaData::BYTE_ARRAY(idx) = &ci[0][1]
    {
        for p in 0..2 {
            assert!(
                idx.is_null_page(p),
                "fixture pages 0-1 must be all-null in the column index"
            );
        }
    }
    let selection =
        page_selection(&file_metadata(&data_path), &schema, &predicate, &None).expect("selection");
    let selectors: Vec<_> = selection.iter().collect();
    assert!(
        selectors[0].skip && selectors[0].row_count == 128,
        "all-null pages must be skipped for < (column-index semantics)"
    );
    assert!(
        selectors.iter().skip(1).all(|s| !s.skip),
        "mixed and non-null pages must be kept for <"
    );
    let on = collect(
        task(&data_path, schema.clone(), &[1, 2], Some(predicate.clone())),
        true,
    )
    .await;
    let off = collect(task(&data_path, schema, &[1, 2], Some(predicate)), false).await;
    assert_eq!(dump(&on).len(), ROWS - 128);
    assert_eq!(dump(&off).len(), ROWS);
}

#[tokio::test]
async fn n_eq_not_eq_not_in_match_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    write_null_pages(&data_path);
    let schema = id_s_schema();
    let eq = bound(&schema, Reference::new("s").equal_to(Datum::string("v300")));
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2], Some(eq))).await;
    assert_eq!(rows.len(), 1);
    let not_eq = bound(
        &schema,
        Reference::new("s").not_equal_to(Datum::string("v300")),
    );
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2], Some(not_eq))).await;
    assert_eq!(rows.len(), ROWS - 1);
    let not_in = bound(
        &schema,
        Reference::new("s").is_not_in([Datum::string("v300"), Datum::string("v301")]),
    );
    let rows = on_off(task(&data_path, schema, &[1, 2], Some(not_in))).await;
    assert_eq!(rows.len(), ROWS - 2);
}

fn write_nan_pages(path: &str) {
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let floats: Vec<f32> = (0..ROWS)
        .map(|i| {
            if (256..320).contains(&i) {
                f32::NAN
            } else if i < 256 {
                i as f32
            } else {
                (i - 320) as f32
            }
        })
        .collect();
    let doubles: Vec<f64> = floats.iter().map(|f| f64::from(*f)).collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, false, 1),
        field("f", DataType::Float32, false, 2),
        field("d", DataType::Float64, false, 3),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(ids)) as ArrayRef,
        Arc::new(Float32Array::from(floats)) as ArrayRef,
        Arc::new(Float64Array::from(doubles)) as ArrayRef,
    ])
    .expect("batch");
    write_parquet(path, arrow_schema, &[batch], page_props());
}

fn nan_schema() -> crate::spec::SchemaRef {
    iceberg_schema(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(2, "f", Type::Primitive(PrimitiveType::Float)),
        NestedField::required(3, "d", Type::Primitive(PrimitiveType::Double)),
    ])
}

#[tokio::test]
async fn f_is_nan_and_not_nan_match_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    write_nan_pages(&data_path);
    let schema = nan_schema();
    let predicate = bound(&schema, Reference::new("f").is_nan());
    let rows = on_off(task(
        &data_path,
        schema.clone(),
        &[1, 2, 3],
        Some(predicate),
    ))
    .await;
    assert_eq!(rows.len(), 64);
    let predicate = bound(&schema, Reference::new("f").is_not_nan());
    let rows = on_off(task(
        &data_path,
        schema.clone(),
        &[1, 2, 3],
        Some(predicate),
    ))
    .await;
    assert_eq!(rows.len(), ROWS - 64);
    let predicate = bound(&schema, Reference::new("d").is_nan());
    let rows = on_off(task(
        &data_path,
        schema.clone(),
        &[1, 2, 3],
        Some(predicate),
    ))
    .await;
    assert_eq!(rows.len(), 64);
    let predicate = bound(&schema, Reference::new("d").is_not_nan());
    let rows = on_off(task(&data_path, schema, &[1, 2, 3], Some(predicate))).await;
    assert_eq!(rows.len(), ROWS - 64);
}

#[tokio::test]
async fn f_lt_gt_not_match_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    write_nan_pages(&data_path);
    let schema = nan_schema();
    let lt = bound(&schema, Reference::new("f").less_than(Datum::float(100.0)));
    let id_pred = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(256)),
    );
    let selection =
        page_selection(&file_metadata(&data_path), &schema, &id_pred, &None).expect("selection");
    assert_prunes(&selection);
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2, 3], Some(lt))).await;
    assert_eq!(rows.len(), 200);
    let gt = bound(
        &schema,
        Reference::new("f").greater_than(Datum::float(400.0)),
    );
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2, 3], Some(gt))).await;
    assert_eq!(rows.len(), 0);
    let not_lt = bound(
        &schema,
        Reference::new("f").less_than(Datum::float(100.0)).negate(),
    );
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2, 3], Some(not_lt))).await;
    assert_eq!(rows.len(), ROWS - 200);
    let lt_double = bound(&schema, Reference::new("d").less_than(Datum::double(100.0)));
    let rows = on_off(task(&data_path, schema, &[1, 2, 3], Some(lt_double))).await;
    assert_eq!(rows.len(), 200);
}

#[tokio::test]
async fn f_eq_nan_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    write_nan_pages(&data_path);
    let schema = nan_schema();
    let predicate = bound(
        &schema,
        Reference::new("f").equal_to(Datum::float(f32::NAN)),
    );
    let rows = on_off(task(&data_path, schema, &[1, 2, 3], Some(predicate))).await;
    assert_eq!(rows.len(), 0);
}

#[test]
fn f_eq_in_nan_bound_keeps_page() {
    use fnv::FnvHashSet;

    use crate::expr::visitors::page_index_evaluator::{PageIndexEvaluator, PageNullCount};

    let datum = Datum::float(100.0);
    assert!(
        PageIndexEvaluator::eq_keeps_page(
            Some(Datum::float(f32::NAN)),
            Some(Datum::float(5.0)),
            PageNullCount::NoneNull,
            &datum,
        ),
        "a NaN min bound is unreliable and must keep the page"
    );
    assert!(
        PageIndexEvaluator::eq_keeps_page(
            Some(Datum::float(0.0)),
            Some(Datum::float(f32::NAN)),
            PageNullCount::NoneNull,
            &datum,
        ),
        "a NaN max bound is unreliable and must keep the page"
    );
    assert!(
        PageIndexEvaluator::eq_keeps_page(
            Some(Datum::float(0.0)),
            Some(Datum::float(5.0)),
            PageNullCount::NoneNull,
            &Datum::float(f32::NAN),
        ),
        "a NaN literal is unreliable and must keep the page"
    );
    assert!(
        !PageIndexEvaluator::eq_keeps_page(
            Some(Datum::float(0.0)),
            Some(Datum::float(5.0)),
            PageNullCount::NoneNull,
            &datum,
        ),
        "finite bounds that exclude the literal still skip the page"
    );

    let literals: FnvHashSet<Datum> = [Datum::float(100.0)].into_iter().collect();
    assert!(
        PageIndexEvaluator::in_keeps_page(
            Some(Datum::float(f32::NAN)),
            Some(Datum::float(5.0)),
            PageNullCount::NoneNull,
            &literals,
        ),
        "a NaN min bound is unreliable and must keep the page for IN"
    );
    assert!(
        PageIndexEvaluator::in_keeps_page(
            Some(Datum::float(0.0)),
            Some(Datum::float(f32::NAN)),
            PageNullCount::NoneNull,
            &literals,
        ),
        "a NaN max bound is unreliable and must keep the page for IN"
    );
    let nan_literals: FnvHashSet<Datum> = [Datum::float(f32::NAN)].into_iter().collect();
    assert!(
        PageIndexEvaluator::in_keeps_page(
            Some(Datum::float(0.0)),
            Some(Datum::float(5.0)),
            PageNullCount::NoneNull,
            &nan_literals,
        ),
        "a NaN literal is unreliable and must keep the page for IN"
    );
    assert!(
        !PageIndexEvaluator::in_keeps_page(
            Some(Datum::float(0.0)),
            Some(Datum::float(5.0)),
            PageNullCount::NoneNull,
            &literals,
        ),
        "finite bounds that exclude every literal still skip the page"
    );
}

#[tokio::test]
async fn t_truncated_string_bounds_match_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let prefix = "x".repeat(80);
    let values: Vec<Option<String>> = (0..ROWS)
        .map(|i| Some(format!("{prefix}{i:05}suffix")))
        .collect();
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    write_id_s_pages(&data_path, &ids, &values);
    let schema = id_s_schema();
    let target = format!("{prefix}00300suffix");
    let eq = bound(
        &schema,
        Reference::new("s").equal_to(Datum::string(&target)),
    );
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2], Some(eq))).await;
    assert_eq!(rows.len(), 1);
    let lt = bound(
        &schema,
        Reference::new("s").less_than(Datum::string(&target)),
    );
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2], Some(lt))).await;
    assert_eq!(rows.len(), 300);
    let ge = bound(
        &schema,
        Reference::new("s").greater_than_or_equal_to(Datum::string(&target)),
    );
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2], Some(ge))).await;
    assert_eq!(rows.len(), ROWS - 300);
    let starts = bound(
        &schema,
        Reference::new("s").starts_with(Datum::string(format!("{prefix}003"))),
    );
    let rows = on_off(task(&data_path, schema.clone(), &[1, 2], Some(starts))).await;
    assert_eq!(rows.len(), 100);
    let not_starts = bound(
        &schema,
        Reference::new("s").not_starts_with(Datum::string(&prefix)),
    );
    let rows = on_off(task(&data_path, schema, &[1, 2], Some(not_starts))).await;
    assert_eq!(rows.len(), 0);
}

#[tokio::test]
async fn t_binary_column_matches_unfiltered() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let values: Vec<Vec<u8>> = (0..ROWS)
        .map(|i| (i as u64).to_be_bytes().to_vec())
        .collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        field("id", DataType::Int32, false, 1),
        field("b", DataType::Binary, false, 2),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from((0..ROWS as i32).collect::<Vec<i32>>())) as ArrayRef,
        Arc::new(BinaryArray::from(
            values.iter().map(|v| v.as_slice()).collect::<Vec<&[u8]>>(),
        )) as ArrayRef,
    ])
    .expect("batch");
    write_parquet(&data_path, arrow_schema, &[batch], page_props());
    let schema = iceberg_schema(vec![
        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)),
        NestedField::required(2, "b", Type::Primitive(PrimitiveType::Binary)),
    ]);
    let eq = bound(
        &schema,
        Reference::new("b").equal_to(Datum::binary(values[300].clone())),
    );
    let selection =
        page_selection(&file_metadata(&data_path), &schema, &eq, &None).expect("selection");
    assert_prunes(&selection);
    let rows = on_off(task(&data_path, schema, &[1, 2], Some(eq))).await;
    assert_eq!(rows.len(), 1);
}

#[tokio::test]
async fn r_ranged_task_intersects_row_group_and_page_selection() {
    let tmp = tmpdir();
    let data_path = path(&tmp, "data.parquet");
    let ids: Vec<i32> = (0..ROWS as i32).collect();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field(
        "id",
        DataType::Int32,
        false,
        1,
    )]));
    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(ids.clone())) as ArrayRef,
        ])
        .expect("batch");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_data_page_row_count_limit(32)
        .set_write_batch_size(32)
        .set_max_row_group_row_count(Some(128))
        .build();
    write_parquet(&data_path, arrow_schema, &[batch], props);
    let metadata = file_metadata(&data_path);
    assert_eq!(metadata.num_row_groups(), 4);
    assert_page_count(&metadata, 3, 0, 3);
    let rg2_start = {
        let first_column = metadata.row_group(2).columns().first().expect("column");
        let data_offset = first_column.data_page_offset();
        match first_column.dictionary_page_offset() {
            Some(dict) if data_offset > dict => dict,
            _ => data_offset,
        }
    } as u64;
    let file_size = std::fs::metadata(&data_path).expect("stat").len();
    let schema = id_schema();
    let predicate = bound(
        &schema,
        Reference::new("id").greater_than_or_equal_to(Datum::int(384)),
    );
    let selection =
        page_selection(&metadata, &schema, &predicate, &Some(vec![2, 3])).expect("selection");
    assert_eq!(selected_rows(&selection), 128);
    let mut t = task(&data_path, schema, &[1], Some(predicate));
    t.start = rg2_start;
    t.length = file_size - rg2_start;
    let rows = on_off(t).await;
    assert_eq!(rows.len(), 128);
}
