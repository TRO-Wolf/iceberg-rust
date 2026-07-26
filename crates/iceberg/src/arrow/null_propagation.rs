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

//! Null-bit propagation across Arrow struct boundaries.
//!
//! Arrow does **not** require a null struct slot to mask its children: a `StructArray` whose row
//! `i` is NULL may hold any bytes in each child at row `i` (`StructArray::try_new` only requires
//! the *reverse* containment — a non-nullable field's own nulls must be masked by the parent).
//! Consequently, the moment a nested child array is handed to a consumer **on its own** — detached
//! from the parent that made it unreachable — that consumer sees live-looking garbage where the
//! logical value is NULL.
//!
//! Java never has this problem because its writers/readers descend the value tree: a null struct
//! short-circuits at the union writer (`ValueWriters$OptionWriter.write`, iceberg-core 1.10.0 —
//! `if (value == null) { encoder.writeIndex(nullIndex); }`), so a child under a null parent is
//! never visited at all. The Rust equivalent, wherever a subtree is flattened into a standalone
//! array, is to union each ancestor's validity into the child before handing it over.
//!
//! This module is the one home for that walk:
//!
//! * [`array_with_parent_validity`] — the single step (union one parent's nulls into one child).
//! * [`propagate_struct_validity`] — the whole tree (push every struct's validity down into its
//!   fields, recursively).
//!
//! Propagation stops at a collection boundary (list / large list / fixed-size list / map /
//! dictionary / run-end): those children are *not* row-aligned with the parent, so a parent null
//! bit has no per-child-row meaning there. The collection array itself still receives its own
//! parent's nulls, which is row-aligned and correct.

use arrow_array::{Array, ArrayRef, StructArray, make_array};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;

use crate::{Error, ErrorKind, Result};

/// Maximum struct nesting depth [`propagate_struct_validity`] will descend before failing.
///
/// Mirrors the schema visitor's own bound (`spec::schema::visitor`'s `MAX_SCHEMA_NESTING_DEPTH`):
/// the array shape here comes from a data/delete file, i.e. from outside the process, so a
/// malicious or corrupt file must not be able to overflow the thread stack.
const MAX_NULL_PROPAGATION_DEPTH: usize = 128;

/// Union `parent_nulls` into `child`'s own validity, returning an array whose logical nulls are
/// `child`'s nulls ∪ `parent_nulls`.
///
/// Returns `child` unchanged (a cheap `Arc` clone) whenever the union would add nothing — no
/// parent nulls at all, or every parent null already present in the child. That keeps the common
/// case allocation-free and byte-identical.
///
/// # Errors
///
/// * `DataInvalid` if `parent_nulls` and `child` disagree on length — they are supposed to be
///   row-aligned siblings of the same struct.
/// * `FeatureUnsupported` if `child`'s encoding cannot carry a top-level validity mask at all
///   (`RunEndEncoded`, `Union`). Failing loudly here is deliberate: silently returning the
///   un-masked child is exactly the corruption this module exists to prevent.
pub(crate) fn array_with_parent_validity(
    child: &ArrayRef,
    parent_nulls: Option<&NullBuffer>,
) -> Result<ArrayRef> {
    Ok(with_parent_validity_opt(child, parent_nulls)?.unwrap_or_else(|| child.clone()))
}

/// [`array_with_parent_validity`], but reports "nothing to do" as `None` so callers that rebuild a
/// parent can tell whether any descendant actually changed.
fn with_parent_validity_opt(
    child: &ArrayRef,
    parent_nulls: Option<&NullBuffer>,
) -> Result<Option<ArrayRef>> {
    let Some(parent_nulls) = parent_nulls else {
        return Ok(None);
    };
    if parent_nulls.null_count() == 0 {
        return Ok(None);
    }
    if parent_nulls.len() != child.len() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "Parent validity is not row-aligned with its struct child",
        )
        .with_context("parent_rows", parent_nulls.len().to_string())
        .with_context("child_rows", child.len().to_string()));
    }

    let own_nulls = child.logical_nulls();
    if own_nulls
        .as_ref()
        .is_some_and(|own| own.contains(parent_nulls))
    {
        // Every parent null is already a child null — the union is the child's own buffer.
        return Ok(None);
    }

    if !matches!(
        child.data_type(),
        DataType::RunEndEncoded(_, _) | DataType::Union(_, _)
    ) {
        let unioned =
            NullBuffer::union(Some(parent_nulls), own_nulls.as_ref()).ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Null buffer union produced no buffer despite a non-empty parent validity",
                )
            })?;
        let data = child
            .to_data()
            .into_builder()
            .nulls(Some(unioned))
            .build()
            .map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Failed to apply the parent's validity to a struct child",
                )
                .with_context("child_type", format!("{:?}", child.data_type()))
                .with_source(err)
            })?;
        return Ok(Some(make_array(data)));
    }

    Err(Error::new(
        ErrorKind::FeatureUnsupported,
        "Struct child encoding cannot carry a validity mask, so the parent's NULLs cannot be \
         propagated into it",
    )
    .with_context("child_type", format!("{:?}", child.data_type())))
}

/// Push every struct's validity down into its fields, recursively, so that any subtree extracted
/// from the result carries the NULLs of every struct that encloses it.
///
/// Returns `array` unchanged (a cheap `Arc` clone) when no struct in the tree has NULLs to
/// contribute — the overwhelmingly common shape, and the reason this is safe to call on every
/// batch.
pub(crate) fn propagate_struct_validity(array: &ArrayRef) -> Result<ArrayRef> {
    Ok(propagate(array, None, 0)?.unwrap_or_else(|| array.clone()))
}

/// Depth-bounded body of [`propagate_struct_validity`]; `None` means "unchanged".
fn propagate(
    array: &ArrayRef,
    parent_nulls: Option<&NullBuffer>,
    depth: usize,
) -> Result<Option<ArrayRef>> {
    if depth > MAX_NULL_PROPAGATION_DEPTH {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Struct nesting exceeds maximum depth {MAX_NULL_PROPAGATION_DEPTH}"),
        ));
    }

    // Step 1: this node absorbs its own parent's validity.
    let masked = with_parent_validity_opt(array, parent_nulls)?;
    let node = masked.as_ref().unwrap_or(array);

    // Step 2: only a struct is row-aligned with its children, so only a struct propagates further.
    if !matches!(node.data_type(), DataType::Struct(_)) {
        return Ok(masked);
    }
    let struct_array = node.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
        Error::new(
            ErrorKind::Unexpected,
            "Array reports a struct data type but is not a StructArray",
        )
    })?;

    // `StructArray::logical_nulls` is exactly its own validity, which after step 1 already
    // includes every ancestor's.
    let effective = struct_array.nulls().cloned();
    let mut children = Vec::with_capacity(struct_array.num_columns());
    let mut any_child_changed = false;
    for column in struct_array.columns() {
        match propagate(column, effective.as_ref(), depth + 1)? {
            Some(new_child) => {
                any_child_changed = true;
                children.push(new_child);
            }
            None => children.push(column.clone()),
        }
    }

    if !any_child_changed {
        return Ok(masked);
    }

    let DataType::Struct(fields) = struct_array.data_type() else {
        return Err(Error::new(
            ErrorKind::Unexpected,
            "StructArray data type is not a struct",
        ));
    };
    let rebuilt =
        StructArray::try_new_with_length(fields.clone(), children, effective, struct_array.len())
            .map_err(|err| {
            Error::new(
                ErrorKind::DataInvalid,
                "Failed to rebuild a struct array after propagating its validity",
            )
            .with_source(err)
        })?;
    Ok(Some(std::sync::Arc::new(rebuilt) as ArrayRef))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int32Type;
    use arrow_array::{Array, ArrayRef, Int32Array, RunArray, StringArray, StructArray};
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Fields};

    use super::*;
    use crate::ErrorKind;

    fn int_child(values: Vec<Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from(values)) as ArrayRef
    }

    fn struct_of(name: &str, child: ArrayRef, valid: Vec<bool>) -> ArrayRef {
        let field = Arc::new(Field::new(name, child.data_type().clone(), true));
        Arc::new(
            StructArray::try_new(
                Fields::from(vec![field]),
                vec![child],
                Some(NullBuffer::from(valid)),
            )
            .expect("struct array"),
        ) as ArrayRef
    }

    #[test]
    fn test_no_parent_nulls_returns_child_untouched() {
        let child = int_child(vec![Some(1), Some(2)]);
        let out = array_with_parent_validity(&child, None).expect("no parent validity");
        assert_eq!(out.logical_null_count(), 0);
        assert!(
            Arc::ptr_eq(&child, &out),
            "an absent parent validity must not reallocate"
        );
    }

    #[test]
    fn test_parent_null_masks_live_child_value() {
        // The core defect: the child looks live at row 0, the parent says the row is NULL.
        let child = int_child(vec![Some(7), Some(42)]);
        let parent = NullBuffer::from(vec![false, true]);
        let out = array_with_parent_validity(&child, Some(&parent)).expect("union");
        assert!(out.is_null(0), "row 0 must become NULL from the parent");
        assert!(out.is_valid(1), "row 1 must stay live");
        assert_eq!(out.logical_null_count(), 1);
    }

    #[test]
    fn test_union_is_a_union_not_a_replacement() {
        // Child NULL at row 1, parent NULL at row 0 — both must survive.
        let child = int_child(vec![Some(7), None, Some(9)]);
        let parent = NullBuffer::from(vec![false, true, true]);
        let out = array_with_parent_validity(&child, Some(&parent)).expect("union");
        assert!(out.is_null(0), "parent null must survive");
        assert!(out.is_null(1), "child null must survive");
        assert!(out.is_valid(2), "live row must survive");
    }

    #[test]
    fn test_row_count_mismatch_is_a_typed_error() {
        let child = int_child(vec![Some(1), Some(2)]);
        let parent = NullBuffer::from(vec![false, true, true]);
        let err = array_with_parent_validity(&child, Some(&parent))
            .expect_err("misaligned validity must not be applied");
        assert_eq!(err.kind(), ErrorKind::DataInvalid);
    }

    #[test]
    fn test_run_end_encoded_child_fails_loudly() {
        // REE cannot carry a top-level null mask; the parent's NULLs would be silently dropped.
        let run_ends = Int32Array::from(vec![1, 2]);
        let values = StringArray::from(vec!["a", "b"]);
        let ree: ArrayRef =
            Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &values).expect("run array"));
        let parent = NullBuffer::from(vec![false, true]);
        let err = array_with_parent_validity(&ree, Some(&parent))
            .expect_err("REE child must fail loudly, not silently keep the stale value");
        assert_eq!(err.kind(), ErrorKind::FeatureUnsupported);
    }

    #[test]
    fn test_propagate_reaches_a_grandchild() {
        // outer{ mid{ leaf } } with outer NULL at row 0 and every inner slot live.
        let leaf = int_child(vec![Some(7), Some(42)]);
        let mid = struct_of("leaf", leaf, vec![true, true]);
        let outer = struct_of("mid", mid, vec![false, true]);

        let out = propagate_struct_validity(&outer).expect("propagate");
        let out_struct = out
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("outer struct");
        let mid_out = out_struct.column(0);
        assert!(mid_out.is_null(0), "mid must inherit the outer NULL");
        let leaf_out = mid_out
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("mid struct")
            .column(0);
        assert!(
            leaf_out.is_null(0),
            "the grandchild leaf must inherit the outer NULL — this is the whole point"
        );
        assert!(leaf_out.is_valid(1), "the live row must stay live");
    }

    #[test]
    fn test_propagate_is_a_noop_without_nulls() {
        let leaf = int_child(vec![Some(7), Some(42)]);
        let mid = struct_of("leaf", leaf, vec![true, true]);
        let outer = struct_of("mid", mid, vec![true, true]);
        let out = propagate_struct_validity(&outer).expect("propagate");
        assert!(
            Arc::ptr_eq(&outer, &out),
            "a null-free tree must be returned untouched, not rebuilt"
        );
    }

    #[test]
    fn test_propagate_preserves_data_type_and_length() {
        let leaf = int_child(vec![Some(7), Some(42)]);
        let mid = struct_of("leaf", leaf, vec![true, true]);
        let outer = struct_of("mid", mid, vec![false, true]);
        let out = propagate_struct_validity(&outer).expect("propagate");
        assert_eq!(
            out.data_type(),
            outer.data_type(),
            "propagation must not change the arrow type"
        );
        assert_eq!(out.len(), outer.len(), "propagation must not change length");
    }

    #[test]
    fn test_propagate_stops_at_a_dictionary_boundary() {
        // A dictionary child is row-aligned with the parent, so the parent's NULL lands on the
        // KEY (making the row null) — the dictionary VALUES must not be touched.
        use arrow_array::DictionaryArray;
        let keys = Int32Array::from(vec![0, 1]);
        let values = StringArray::from(vec!["x", "y"]);
        let dict: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).expect("dictionary"),
        );
        let outer = struct_of("d", dict, vec![false, true]);
        let out = propagate_struct_validity(&outer).expect("propagate");
        let child = out
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("outer struct")
            .column(0);
        assert!(child.is_null(0), "dictionary row 0 must become NULL");
        assert!(child.is_valid(1), "dictionary row 1 must stay live");
        assert_eq!(
            child.data_type(),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            "the dictionary encoding must be preserved"
        );
    }
}
