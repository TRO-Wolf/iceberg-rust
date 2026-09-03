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

use datafusion::common::{DataFusionError, Result as DFResult};
use iceberg::delete_vector_container::{
    DvContainerClose, close_touched_dv_containers_with_partitions,
};
use iceberg::table::Table;

use crate::to_datafusion_error;

pub(crate) async fn write_deletion_vectors(
    table: &Table,
    pairs: &[(String, i64)],
    scan_snapshot_id: Option<i64>,
) -> DFResult<DvContainerClose> {
    let mut new_positions: HashMap<String, Vec<u64>> = HashMap::new();
    for (path, position) in pairs {
        let position = u64::try_from(*position).map_err(|_| {
            DataFusionError::Internal(format!(
                "deletion-vector: negative row position {position} for data file `{path}`"
            ))
        })?;
        new_positions
            .entry(path.clone())
            .or_default()
            .push(position);
    }
    close_touched_dv_containers_with_partitions(
        table,
        &new_positions,
        scan_snapshot_id,
        &HashMap::new(),
        None,
    )
    .await
    .map_err(to_datafusion_error)
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal,
        Struct as IcebergStruct, referenced_data_file_location,
    };

    fn delete_file_of(content: DataContentType, file_format: DataFileFormat) -> DataFile {
        let mut builder = DataFileBuilder::default();
        builder
            .content(content)
            .file_path("s3://b/d".to_string())
            .file_format(file_format)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0);
        if content == DataContentType::EqualityDeletes {
            builder.equality_ids(Some(vec![1]));
        }
        if file_format == DataFileFormat::Puffin {
            builder
                .content_offset(Some(4))
                .content_size_in_bytes(Some(40))
                .referenced_data_file(Some("s3://b/a.parquet".to_string()));
        }
        builder.build().expect("build the delete file")
    }

    #[test]
    fn test_legacy_delete_entry_derives_the_name_from_equal_file_path_bounds() {
        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("s3://b/pos-del.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(7)
            .partition(IcebergStruct::from_iter([Some(Literal::long(999))]))
            .lower_bounds(std::collections::HashMap::from([(
                iceberg::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("s3://b/a.parquet"),
            )]))
            .upper_bounds(std::collections::HashMap::from([(
                iceberg::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("s3://b/a.parquet"),
            )]))
            .build()
            .expect("build a bounds-scoped position delete");
        assert_eq!(
            referenced_data_file_location(&delete_file).as_deref(),
            Some("s3://b/a.parquet"),
            "equal file_path bounds name the data file"
        );
        assert_eq!(
            delete_file.partition_spec_id(),
            7,
            "bounds-scoped delete keeps its spec id"
        );
        assert_eq!(
            delete_file.partition(),
            &IcebergStruct::from_iter([Some(Literal::long(999))]),
            "bounds-scoped delete keeps its partition"
        );
    }

    #[test]
    fn test_puffin_position_delete_is_a_deletion_vector() {
        let file = delete_file_of(DataContentType::PositionDeletes, DataFileFormat::Puffin);
        assert!(
            iceberg::spec::is_deletion_vector(&file),
            "a puffin position delete is a deletion vector"
        );
        let parquet = delete_file_of(DataContentType::PositionDeletes, DataFileFormat::Parquet);
        assert!(
            !iceberg::spec::is_deletion_vector(&parquet),
            "a parquet position delete is a live legacy delete"
        );
    }
}
