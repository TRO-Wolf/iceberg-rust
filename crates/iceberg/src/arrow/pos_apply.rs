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

use arrow_arith::boolean::{and, not};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_select::filter::{filter, filter_record_batch};
use parquet::file::metadata::RowGroupMetaData;

use crate::arrow::delete_filter::positional_delete_keep_mask;
use crate::arrow::equality_delete_set::EqDeleteKeySet;
use crate::arrow::record_batch_predicate::evaluate_predicate_to_mask;
use crate::arrow::record_batch_transformer::RecordBatchTransformer;
use crate::delete_vector::DeleteVector;
use crate::error::Result;
use crate::expr::BoundPredicate;
use crate::metadata_columns::RESERVED_COL_NAME_DELETED;
use crate::{Error, ErrorKind};

pub(super) struct BatchDeleteInputs<'a> {
    pub(super) positional_deletes: Option<&'a Arc<DeleteVector>>,
    pub(super) eq_delete_predicate: Option<&'a BoundPredicate>,
    pub(super) eq_delete_sets: Option<&'a [EqDeleteKeySet]>,
}

pub(super) fn apply_pos_aware_batch(
    batch: RecordBatch,
    transformer: &mut RecordBatchTransformer,
    absolute_pos: &mut u64,
    deletes: BatchDeleteInputs<'_>,
    residual_predicate: Option<&BoundPredicate>,
    include_deleted: bool,
) -> Result<RecordBatch> {
    let row_count = batch.num_rows();
    let batch_base = *absolute_pos;
    let transformed = transformer.process_record_batch(batch.clone())?;
    debug_assert!(
        {
            use arrow_array::Int64Array;

            use crate::metadata_columns::RESERVED_COL_NAME_POS;
            match transformed.column_by_name(RESERVED_COL_NAME_POS) {
                Some(col) if row_count > 0 => col
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .is_some_and(|a| a.value(0) as u64 == batch_base),
                _ => true,
            }
        },
        "absolute_pos desynced from transformer _pos (batch_base={batch_base}, rows={row_count})"
    );
    *absolute_pos = absolute_pos.saturating_add(row_count as u64);
    if !include_deleted {
        let mask = survival_mask(
            &batch,
            row_count,
            batch_base,
            deletes.positional_deletes,
            residual_predicate,
            deletes.eq_delete_predicate,
            deletes.eq_delete_sets,
        )?;
        return match mask {
            None => Ok(transformed),
            Some(mask) => filter_record_batch(&transformed, &mask).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to apply merge-on-read deletes to a data batch under _pos / whole-file scan",
                )
                .with_source(e)
            }),
        };
    }
    let delete_keep = survival_mask(
        &batch,
        row_count,
        batch_base,
        deletes.positional_deletes,
        None,
        deletes.eq_delete_predicate,
        deletes.eq_delete_sets,
    )?;
    let residual_mask = match residual_predicate {
        None => None,
        Some(predicate) => Some(coerce_nulls_to_false(&evaluate_predicate_to_mask(
            predicate, &batch,
        )?)),
    };
    mark_deleted_and_filter(transformed, delete_keep, residual_mask)
}

pub(super) fn apply_pushdown_path_batch(
    batch: RecordBatch,
    absolute_pos: &mut u64,
    positional_deletes: Option<&Arc<DeleteVector>>,
    eq_delete_predicate: Option<&BoundPredicate>,
    eq_delete_sets: Option<&[EqDeleteKeySet]>,
    residual_predicate: Option<&BoundPredicate>,
    include_deleted: bool,
) -> Result<RecordBatch> {
    let row_count = batch.num_rows();
    if !include_deleted {
        let mut batch = batch;
        if let Some(mask) = eq_delete_keep_mask(
            &batch,
            batch.num_rows(),
            eq_delete_predicate,
            eq_delete_sets,
        )? {
            batch = filter_record_batch(&batch, &mask).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to apply equality-delete keyset keep-mask to a Parquet data batch",
                )
                .with_source(e)
            })?;
        }
        if let Some(residual) = residual_predicate {
            let mask = coerce_nulls_to_false(&evaluate_predicate_to_mask(residual, &batch)?);
            batch = filter_record_batch(&batch, &mask).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to apply post-decode residual predicate to a Parquet data batch",
                )
                .with_source(e)
            })?;
        }
        return Ok(batch);
    }
    let batch_base = *absolute_pos;
    *absolute_pos = absolute_pos.saturating_add(row_count as u64);
    let delete_keep = survival_mask(
        &batch,
        row_count,
        batch_base,
        positional_deletes,
        None,
        eq_delete_predicate,
        eq_delete_sets,
    )?;
    let residual_mask = match residual_predicate {
        None => None,
        Some(predicate) => Some(coerce_nulls_to_false(&evaluate_predicate_to_mask(
            predicate, &batch,
        )?)),
    };
    mark_deleted_and_filter(batch, delete_keep, residual_mask)
}

fn mark_deleted_and_filter(
    batch: RecordBatch,
    delete_keep: Option<BooleanArray>,
    residual_mask: Option<BooleanArray>,
) -> Result<RecordBatch> {
    let row_count = batch.num_rows();
    let deleted = match delete_keep {
        None => BooleanArray::from(vec![false; row_count]),
        Some(keep) => not(&keep).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to invert the merge-on-read keep mask for the _deleted column",
            )
            .with_source(e)
        })?,
    };
    let (batch, deleted) = match residual_mask {
        None => (batch, Arc::new(deleted) as ArrayRef),
        Some(mask) => (
            filter_record_batch(&batch, &mask).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to apply the scan residual under a _deleted projection",
                )
                .with_source(e)
            })?,
            filter(&deleted, &mask).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to restrict the _deleted verdict to the residual survivors",
                )
                .with_source(e)
            })?,
        ),
    };
    let deleted = deleted
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Filtering the BooleanArray _deleted verdict did not return a BooleanArray",
            )
        })?;
    overwrite_deleted_column(batch, deleted)
}

fn overwrite_deleted_column(batch: RecordBatch, deleted: &BooleanArray) -> Result<RecordBatch> {
    let (index, _) = batch
        .schema()
        .column_with_name(RESERVED_COL_NAME_DELETED)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "A _deleted scan reached the delete verdict with no _deleted column in the batch",
            )
        })?;
    let mut columns = batch.columns().to_vec();
    columns[index] = Arc::new(deleted.clone());
    RecordBatch::try_new(batch.schema(), columns).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to attach the _deleted verdict to a data batch",
        )
        .with_source(e)
    })
}

pub(super) fn selected_groups_start_ordinal(
    row_groups: &[RowGroupMetaData],
    selected: Option<&[usize]>,
) -> u64 {
    let first = selected
        .and_then(|indices| indices.iter().min())
        .copied()
        .unwrap_or(0);
    row_groups
        .iter()
        .take(first)
        .map(|group| group.num_rows().max(0) as u64)
        .sum()
}

pub(super) fn survival_mask(
    batch: &RecordBatch,
    num_rows: usize,
    batch_base: u64,
    positional_deletes: Option<&Arc<DeleteVector>>,
    residual_predicate: Option<&BoundPredicate>,
    eq_delete_predicate: Option<&BoundPredicate>,
    eq_delete_sets: Option<&[EqDeleteKeySet]>,
) -> Result<Option<BooleanArray>> {
    let positional_mask: Option<BooleanArray> = match positional_deletes {
        Some(deletes) => {
            if deletes.is_empty() {
                None
            } else {
                Some(positional_delete_keep_mask(
                    deletes.as_ref(),
                    batch_base,
                    num_rows,
                ))
            }
        }
        None => None,
    };

    let predicate_keep = |predicate: &BoundPredicate| -> Result<BooleanArray> {
        Ok(coerce_nulls_to_false(&evaluate_predicate_to_mask(
            predicate, batch,
        )?))
    };

    let residual_mask: Option<BooleanArray> = match residual_predicate {
        Some(predicate) => Some(predicate_keep(predicate)?),
        None => None,
    };

    let eq_delete_mask = eq_delete_keep_mask(batch, num_rows, eq_delete_predicate, eq_delete_sets)?;

    let combine =
        |a: Option<BooleanArray>, b: Option<BooleanArray>| -> Result<Option<BooleanArray>> {
            match (a, b) {
                (None, None) => Ok(None),
                (Some(m), None) | (None, Some(m)) => Ok(Some(m)),
                (Some(x), Some(y)) => Ok(Some(and(&x, &y).map_err(|e| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to combine merge-on-read delete masks for a data batch",
                    )
                    .with_source(e)
                })?)),
            }
        };
    let combined = combine(positional_mask, residual_mask)?;
    combine(combined, eq_delete_mask)
}

pub(super) fn eq_delete_keep_mask(
    batch: &RecordBatch,
    num_rows: usize,
    eq_delete_predicate: Option<&BoundPredicate>,
    eq_delete_sets: Option<&[EqDeleteKeySet]>,
) -> Result<Option<BooleanArray>> {
    let mut from_sets: Option<BooleanArray> = None;
    if let Some(sets) = eq_delete_sets.filter(|s| !s.is_empty()) {
        let mut keep = vec![true; num_rows];
        let mut all_sets_safe = true;
        for set in sets {
            match set.delete_mask(batch)? {
                Some(deleted) => {
                    for (k, d) in keep.iter_mut().zip(deleted.iter()) {
                        *k &= !*d;
                    }
                }
                None => {
                    all_sets_safe = false;
                    break;
                }
            }
        }
        if all_sets_safe {
            from_sets = Some(BooleanArray::from(keep));
        }
    }
    match from_sets {
        Some(mask) => Ok(Some(mask)),
        None => match eq_delete_predicate {
            Some(predicate) => Ok(Some(coerce_nulls_to_false(&evaluate_predicate_to_mask(
                predicate, batch,
            )?))),
            None => Ok(None),
        },
    }
}

pub(super) fn coerce_nulls_to_false(mask: &BooleanArray) -> BooleanArray {
    if mask.null_count() == 0 {
        return mask.clone();
    }
    BooleanArray::from_iter((0..mask.len()).map(|i| Some(mask.is_valid(i) && mask.value(i))))
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::file::metadata::{ColumnChunkMetaData, RowGroupMetaData};
    use parquet::schema::types::SchemaDescriptor;
    use roaring::RoaringTreemap;

    use super::{
        BatchDeleteInputs, apply_pos_aware_batch, apply_pushdown_path_batch,
        selected_groups_start_ordinal,
    };
    use crate::arrow::equality_delete_set::EqDeleteKeySet;
    use crate::arrow::record_batch_transformer::RecordBatchTransformerBuilder;
    use crate::delete_vector::DeleteVector;
    use crate::expr::{Bind, Reference};
    use crate::metadata_columns::{RESERVED_COL_NAME_DELETED, RESERVED_FIELD_ID_DELETED};
    use crate::spec::{Datum, NestedField, PrimitiveType, Schema, Type};

    fn id_field_with_id() -> Field {
        Field::new("id", DataType::Int64, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )]))
    }

    fn batch_with_deleted(ids: Vec<i64>, flags: Vec<bool>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            id_field_with_id(),
            Field::new(RESERVED_COL_NAME_DELETED, DataType::Boolean, false),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(BooleanArray::from(flags)) as ArrayRef,
        ])
        .expect("test batch with a _deleted column builds")
    }

    fn batch_without_deleted(ids: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![id_field_with_id()]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(ids)) as ArrayRef])
            .expect("test batch without a _deleted column builds")
    }

    fn id_flag_pairs(batch: &RecordBatch) -> Vec<(i64, bool)> {
        let ids = batch
            .column_by_name("id")
            .expect("test batch carries id")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("test id column is Int64");
        let flags = batch
            .column_by_name(RESERVED_COL_NAME_DELETED)
            .expect("test batch carries _deleted")
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("test _deleted column is Boolean");
        (0..batch.num_rows())
            .map(|i| (ids.value(i), flags.value(i)))
            .collect()
    }

    fn id_list(batch: &RecordBatch) -> Vec<i64> {
        let ids = batch
            .column_by_name("id")
            .expect("test batch carries id")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("test id column is Int64");
        (0..batch.num_rows()).map(|i| ids.value(i)).collect()
    }

    fn snapshot_schema() -> Arc<Schema> {
        Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .expect("single-column test schema builds"),
        )
    }

    fn row_group_with_rows(num_rows: i64, ordinal: i16) -> RowGroupMetaData {
        use parquet::schema::types::Type as SchemaType;

        let schema = SchemaType::group_type_builder("schema")
            .with_fields(vec![Arc::new(
                SchemaType::primitive_type_builder("a", parquet::basic::Type::INT32)
                    .build()
                    .expect("test primitive type builds"),
            )])
            .build()
            .expect("test group type builds");
        let descr = Arc::new(SchemaDescriptor::new(Arc::new(schema)));
        let columns: Vec<ColumnChunkMetaData> = descr
            .columns()
            .iter()
            .map(|ptr| {
                ColumnChunkMetaData::builder(ptr.clone())
                    .build()
                    .expect("test column chunk builds")
            })
            .collect();
        RowGroupMetaData::builder(descr)
            .set_num_rows(num_rows)
            .set_total_byte_size(2000)
            .set_column_metadata(columns)
            .set_ordinal(ordinal)
            .build()
            .expect("test row group builds")
    }

    #[test]
    fn selected_groups_start_ordinal_counts_only_leading_groups() {
        let groups = vec![
            row_group_with_rows(100, 0),
            row_group_with_rows(50, 1),
            row_group_with_rows(200, 2),
        ];
        assert_eq!(selected_groups_start_ordinal(&groups, None), 0);
        assert_eq!(selected_groups_start_ordinal(&groups, Some(&[])), 0);
        assert_eq!(selected_groups_start_ordinal(&groups, Some(&[0])), 0);
        assert_eq!(selected_groups_start_ordinal(&groups, Some(&[1, 2])), 100);
        assert_eq!(selected_groups_start_ordinal(&groups, Some(&[2])), 150);
        assert_eq!(selected_groups_start_ordinal(&groups, Some(&[2, 1])), 100);
    }

    #[test]
    fn pushdown_include_deleted_marks_positional_victims_and_keeps_rows() {
        let batch = batch_with_deleted(vec![10, 20, 30], vec![false, false, false]);
        let deletes = Arc::new(DeleteVector::new(RoaringTreemap::from_iter([1])));
        let mut absolute_pos = 0;
        let out = apply_pushdown_path_batch(
            batch,
            &mut absolute_pos,
            Some(&deletes),
            None,
            None,
            None,
            true,
        )
        .expect("include_deleted apply succeeds");
        assert_eq!(out.num_rows(), 3);
        assert_eq!(id_flag_pairs(&out), vec![
            (10, false),
            (20, true),
            (30, false)
        ]);
        assert_eq!(absolute_pos, 3);
    }

    #[test]
    fn pushdown_include_deleted_without_any_delete_keeps_all_false() {
        let batch = batch_with_deleted(vec![10, 20, 30], vec![false, false, false]);
        let mut absolute_pos = 7;
        let out = apply_pushdown_path_batch(batch, &mut absolute_pos, None, None, None, None, true)
            .expect("include_deleted apply succeeds");
        assert_eq!(id_flag_pairs(&out), vec![
            (10, false),
            (20, false),
            (30, false)
        ]);
        assert_eq!(absolute_pos, 10);
    }

    #[test]
    fn pushdown_include_deleted_applies_residual_but_keeps_verdict() {
        let batch = batch_with_deleted(vec![10, 20, 30], vec![false, false, false]);
        let deletes = Arc::new(DeleteVector::new(RoaringTreemap::from_iter([1])));
        let residual = Reference::new("id")
            .greater_than(Datum::long(10))
            .bind(snapshot_schema(), false)
            .expect("residual binds");
        let mut absolute_pos = 0;
        let out = apply_pushdown_path_batch(
            batch,
            &mut absolute_pos,
            Some(&deletes),
            None,
            None,
            Some(&residual),
            true,
        )
        .expect("include_deleted apply succeeds");
        assert_eq!(id_flag_pairs(&out), vec![(20, true), (30, false)]);
        assert_eq!(absolute_pos, 3);
    }

    #[test]
    fn pushdown_exclude_deleted_still_filters_eq_deletes() {
        let batch = batch_with_deleted(vec![10, 20, 30], vec![false, false, false]);
        let set =
            EqDeleteKeySet::try_build(vec![(1, "id".to_string(), PrimitiveType::Long)], vec![
                vec![Some(Datum::long(20))],
            ])
            .expect("eq-delete key set builds");
        let mut absolute_pos = 0;
        let out = apply_pushdown_path_batch(
            batch,
            &mut absolute_pos,
            None,
            None,
            Some(std::slice::from_ref(&set)),
            None,
            false,
        )
        .expect("exclude_deleted apply succeeds");
        assert_eq!(id_list(&out), vec![10, 30]);
    }

    #[test]
    fn pushdown_include_deleted_without_column_fails() {
        let batch = batch_without_deleted(vec![10, 20, 30]);
        let mut absolute_pos = 0;
        let err = apply_pushdown_path_batch(batch, &mut absolute_pos, None, None, None, None, true)
            .expect_err("a _deleted scan with no _deleted column must fail");
        assert!(
            err.to_string().contains("no _deleted column"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn pos_aware_include_deleted_marks_victims_through_transformer() {
        let batch = batch_without_deleted(vec![10, 20, 30]);
        let mut transformer =
            RecordBatchTransformerBuilder::new(snapshot_schema(), &[1, RESERVED_FIELD_ID_DELETED])
                .build();
        let deletes = Arc::new(DeleteVector::new(RoaringTreemap::from_iter([0])));
        let mut absolute_pos = 0;
        let out = apply_pos_aware_batch(
            batch,
            &mut transformer,
            &mut absolute_pos,
            BatchDeleteInputs {
                positional_deletes: Some(&deletes),
                eq_delete_predicate: None,
                eq_delete_sets: None,
            },
            None,
            true,
        )
        .expect("include_deleted pos-aware apply succeeds");
        assert_eq!(out.num_rows(), 3);
        assert_eq!(id_flag_pairs(&out), vec![
            (10, true),
            (20, false),
            (30, false)
        ]);
        assert_eq!(absolute_pos, 3);
    }
}
