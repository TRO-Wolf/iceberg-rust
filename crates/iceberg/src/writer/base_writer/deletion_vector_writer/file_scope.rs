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

use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
use crate::spec::{DataContentType, DataFile};

/// Whether a previous delete file is file-scoped, so the merge may discard it from the table
/// state. Mirrors Java `ContentFileUtil.isFileScoped`, which is
/// `referencedDataFile(df) != null`. The predicate is broader than `is_deletion_vector`. A DV
/// qualifies because it carries `referenced_data_file`, not because it is a DV.
///
/// | Delete file | File-scoped |
/// |---|---|
/// | equality delete | no |
/// | non-null `referenced_data_file` | yes |
/// | position delete whose `_file_path` lower and upper bounds are present and equal | yes |
/// | any other position delete, which spans many data files | no |
pub(super) fn is_file_scoped(delete_file: &DataFile) -> bool {
    if delete_file.content_type() == DataContentType::EqualityDeletes {
        return false;
    }
    if delete_file.referenced_data_file().is_some() {
        return true;
    }
    // The Java `referencedDataFile` fallback: a position delete whose `_file_path` bounds pin a
    // single data file is file-scoped even without the explicit field. `lower_bounds`/`upper_bounds`
    // are `HashMap<i32, Datum>`; equal Datums under the reserved path id mean a one-data-file delete.
    match (
        delete_file
            .lower_bounds()
            .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
        delete_file
            .upper_bounds()
            .get(&RESERVED_FIELD_ID_DELETE_FILE_PATH),
    ) {
        (Some(lower), Some(upper)) => lower == upper,
        _ => false,
    }
}
