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

use datafusion::arrow::array::{Array, ArrayRef, Int64Array, RunArray, StringArray};
use datafusion::arrow::datatypes::Int32Type;
use datafusion::common::{DataFusionError, Result as DFResult};

pub(crate) fn decode_file_path(col: &ArrayRef, row: usize) -> DFResult<String> {
    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        if plain.is_null(row) {
            return Err(null_file_path_error(row));
        }
        return Ok(plain.value(row).to_string());
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let physical = run.get_physical_index(row);
        let values = run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                DataFusionError::Internal("_file REE values are not Utf8".to_string())
            })?;
        if values.is_null(physical) {
            return Err(null_file_path_error(row));
        }
        return Ok(values.value(physical).to_string());
    }
    Err(DataFusionError::Internal(format!(
        "unexpected _file column type: {:?}",
        col.data_type()
    )))
}

fn null_file_path_error(row: usize) -> DataFusionError {
    DataFusionError::Internal(format!(
        "reserved _file column is NULL at row {row}; a position delete cannot be keyed by an \
         unknown data file"
    ))
}

pub(crate) fn decode_position(col: &Int64Array, row: usize) -> DFResult<i64> {
    if col.is_null(row) {
        return Err(DataFusionError::Internal(format!(
            "reserved _pos column is NULL at row {row}; a position delete cannot be keyed by an \
             unknown row position"
        )));
    }
    Ok(col.value(row))
}

pub(crate) fn decode_file_paths_batch(col: &ArrayRef) -> DFResult<Vec<&str>> {
    if let Some(plain) = col.as_any().downcast_ref::<StringArray>() {
        return (0..plain.len())
            .map(|row| {
                if plain.is_null(row) {
                    return Err(null_file_path_error(row));
                }
                Ok(plain.value(row))
            })
            .collect();
    }
    if let Some(run) = col.as_any().downcast_ref::<RunArray<Int32Type>>() {
        let values = run
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                DataFusionError::Internal("_file REE values are not Utf8".to_string())
            })?;
        let mut out = Vec::with_capacity(run.len());
        if run.offset() == 0 {
            let run_ends = run.run_ends().values();
            let mut start = 0usize;
            for (physical, &end) in run_ends.iter().enumerate() {
                let end = usize::try_from(end).map_err(|_| {
                    DataFusionError::Internal("_file REE run-end is negative".to_string())
                })?;
                if start < end && values.is_null(physical) {
                    return Err(null_file_path_error(start));
                }
                let value = values.value(physical);
                for _ in start..end {
                    out.push(value);
                }
                start = end;
            }
        } else {
            for row in 0..run.len() {
                let physical = run.get_physical_index(row);
                if values.is_null(physical) {
                    return Err(null_file_path_error(row));
                }
                out.push(values.value(physical));
            }
        }
        return Ok(out);
    }
    Err(DataFusionError::Internal(format!(
        "unexpected _file column type: {:?}",
        col.data_type()
    )))
}
