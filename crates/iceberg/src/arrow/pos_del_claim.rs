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

use crate::scan::FileScanTaskDeleteFile;
use crate::spec::{DataContentType, DataFileFormat};
use crate::{Error, ErrorKind};

pub(crate) fn pos_del_claim_key(delete: &FileScanTaskDeleteFile) -> Option<String> {
    if delete.file_type != DataContentType::PositionDeletes {
        return None;
    }
    if delete.file_format == DataFileFormat::Puffin {
        let offset = delete.content_offset.filter(|offset| *offset >= 0)?;
        Some(format!("{}@{offset}", delete.file_path))
    } else {
        Some(delete.file_path.clone())
    }
}

pub(crate) fn pos_del_load_failed_error(file_path: &str, reason: &str) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        format!(
            "the loader for positional delete file '{file_path}' terminated without publishing \
             its deletes: {reason}"
        ),
    )
}
