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

use arrow_arith::boolean::and;
use arrow_array::{Array, BooleanArray, RecordBatch};
use arrow_select::filter::filter_record_batch;

use crate::arrow::delete_filter::positional_delete_keep_mask;
use crate::arrow::equality_delete_set::EqDeleteKeySet;
use crate::arrow::record_batch_predicate::evaluate_predicate_to_mask;
use crate::arrow::record_batch_transformer::RecordBatchTransformer;
use crate::delete_vector::DeleteVector;
use crate::error::Result;
use crate::expr::BoundPredicate;
use crate::{Error, ErrorKind};

pub(super) fn apply_pos_aware_batch(
    batch: RecordBatch,
    transformer: &mut RecordBatchTransformer,
    absolute_pos: &mut u64,
    positional_deletes: Option<&Arc<DeleteVector>>,
    residual_predicate: Option<&BoundPredicate>,
    eq_delete_predicate: Option<&BoundPredicate>,
    eq_delete_sets: Option<&[EqDeleteKeySet]>,
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
    let mask = survival_mask(
        &batch,
        row_count,
        batch_base,
        positional_deletes,
        residual_predicate,
        eq_delete_predicate,
        eq_delete_sets,
    )?;
    *absolute_pos = absolute_pos.saturating_add(row_count as u64);
    match mask {
        None => Ok(transformed),
        Some(mask) => filter_record_batch(&transformed, &mask).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to apply merge-on-read deletes to a data batch under _pos / whole-file scan",
            )
            .with_source(e)
        }),
    }
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
