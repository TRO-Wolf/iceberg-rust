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

use arrow_array::builder::{Int32Builder, Int64Builder};

use super::data_file::{DynMapBuilder, dyn_child};
use crate::Result;

pub(super) fn append_count_map(builder: &mut DynMapBuilder, map: &HashMap<i32, u64>) -> Result<()> {
    let mut keys: Vec<&i32> = map.keys().collect();
    keys.sort_unstable();
    for key in keys {
        dyn_child::<Int32Builder>(builder.keys(), "count map key")?.append_value(*key);
        dyn_child::<Int64Builder>(builder.values(), "count map value")?
            .append_value(map[key] as i64);
    }
    builder.append(true)?;
    Ok(())
}

pub(super) fn append_count_map_or_null_for_pos_delete(
    builder: &mut DynMapBuilder,
    map: &HashMap<i32, u64>,
    is_pos_delete: bool,
) -> Result<()> {
    if is_pos_delete && map.is_empty() {
        builder.append(false)?;
        Ok(())
    } else {
        append_count_map(builder, map)
    }
}
