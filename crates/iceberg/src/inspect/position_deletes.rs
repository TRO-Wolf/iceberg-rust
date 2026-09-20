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

//! `position_deletes` metadata table — schema and scan.
//!
//! Mirrors Java `PositionDeletesTable` (`core/.../PositionDeletesTable.java`): the schema is
//! `calculateSchema` — the fixed metadata columns (`MetadataColumns.DELETE_FILE_PATH` /
//! `DELETE_FILE_POS` / `DELETE_FILE_ROW_*` / `PARTITION_COLUMN_ID` / `SPEC_ID_COLUMN_ID` /
//! `FILE_PATH_COLUMN_ID`, plus the v3 DV columns `CONTENT_OFFSET_COLUMN_ID` /
//! `CONTENT_SIZE_IN_BYTES_COLUMN_ID`), the partition-field id reassignment (smallest positive
//! ids not used by ANY table schema nor the metadata columns), and the empty-partition
//! `TypeUtil.selectNot(PARTITION_COLUMN_ID)` drop. The scan is `PositionDeletesBatchScan`:
//! the current snapshot's DELETE manifests filtered by both manifest evaluators
//! (transformed-spec + own-spec, keyed on `manifest.partitionSpecId()`), live
//! `POSITION_DELETES` entries only, one task per delete file carrying its spec and residual;
//! Puffin delete files read as deletion vectors, Parquet files as positional-delete rows.
//!
//! One deliberate bound, tracked in GAP_MATRIX R142:
//! - **Partition type** is [`TableMetadata::unified_partition_type`] (Java
//!   `Partitioning.partitionType`). That automatically corrects the remapped-child-id set
//!   and the empty-partition drop predicate.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::StructBuilder;
use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray, StructArray};
use arrow_schema::{DataType, Fields};
use futures::{StreamExt, TryStreamExt, stream};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::data_file::partition_field_ids_by_spec;
use super::manifest_source::{MetadataScope, collect_manifest_files};
use super::partition_values::append_partition;
use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::arrow::schema_to_arrow_schema;
use crate::delete_vector::load_delete_vector;
use crate::expr::visitors::expression_evaluator::ExpressionEvaluator;
use crate::expr::visitors::inclusive_projection::InclusiveProjection;
use crate::expr::visitors::manifest_evaluator::ManifestEvaluator;
use crate::expr::visitors::residual_evaluator::ResidualEvaluator;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::metadata_columns::{
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::scan::ArrowRecordBatchStream;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, ManifestContentType, NestedField, NestedFieldRef,
    PartitionField, PartitionSpec, PartitionSpecRef, PrimitiveType, Schema, StructType, Transform,
    Type,
};
use crate::table::Table;
use crate::{Error, ErrorKind, Result};

/// Java `Integer.MAX_VALUE` — the anchor for `MetadataColumns` reserved field ids.
const JAVA_INT_MAX: i32 = i32::MAX;

/// `MetadataColumns.DELETE_FILE_PATH` (`file_path`, required string).
const DELETE_FILE_PATH_ID: i32 = JAVA_INT_MAX - 101;
/// `MetadataColumns.DELETE_FILE_POS` (`pos`, required long).
const DELETE_FILE_POS_ID: i32 = JAVA_INT_MAX - 102;
/// `MetadataColumns.DELETE_FILE_ROW_FIELD_ID` (`row`, optional struct of the table schema).
const DELETE_FILE_ROW_FIELD_ID: i32 = JAVA_INT_MAX - 103;
/// `MetadataColumns.PARTITION_COLUMN_ID` (`partition`, required struct; dropped when empty).
const PARTITION_COLUMN_ID: i32 = JAVA_INT_MAX - 5;
/// `MetadataColumns.SPEC_ID_COLUMN_ID` (`spec_id`, required int).
const SPEC_ID_COLUMN_ID: i32 = JAVA_INT_MAX - 4;
/// `MetadataColumns.FILE_PATH_COLUMN_ID` (`delete_file_path`, required string).
const FILE_PATH_COLUMN_ID: i32 = JAVA_INT_MAX - 1;
/// `MetadataColumns.CONTENT_OFFSET_COLUMN_ID` (`content_offset`, optional long, v3+).
const CONTENT_OFFSET_COLUMN_ID: i32 = JAVA_INT_MAX - 6;
/// `MetadataColumns.CONTENT_SIZE_IN_BYTES_COLUMN_ID` (`content_size_in_bytes`, optional long, v3+).
const CONTENT_SIZE_IN_BYTES_COLUMN_ID: i32 = JAVA_INT_MAX - 7;

/// PositionDeletes table (schema only — see the module doc for the scan bound).
pub struct PositionDeletesTable<'a> {
    table: &'a Table,
    /// Java `Partitioning.partitionType(table)` — stored so [`Self::schema`] stays infallible.
    unified_partition_type: StructType,
}

impl<'a> PositionDeletesTable<'a> {
    /// Fallible constructor: resolves the unified partition type up front.
    ///
    /// The DataFusion `IcebergMetadataTableProvider::try_new` is the public
    /// fallible seam (A5).
    ///
    /// # Errors
    ///
    /// Propagates [`crate::spec::TableMetadata::unified_partition_type`].
    pub fn try_new(table: &'a Table) -> Result<Self> {
        let unified_partition_type = table.metadata().unified_partition_type()?;
        Ok(Self {
            table,
            unified_partition_type,
        })
    }

    /// Create a new PositionDeletes table instance.
    ///
    /// Signature stays infallible (A5). On a G1/G2 table this falls back to
    /// [`TableMetadata::default_partition_type`] so `inspect().position_deletes().schema()`
    /// cannot panic; [`Self::try_new`] is the loud refuse path.
    pub fn new(table: &'a Table) -> Self {
        match Self::try_new(table) {
            Ok(this) => this,
            Err(_) => Self {
                table,
                unified_partition_type: table.metadata().default_partition_type().clone(),
            },
        }
    }

    /// Returns the iceberg schema of the `position_deletes` table.
    ///
    /// Transcribed from Java `PositionDeletesTable.calculateSchema`: column list order is
    /// `file_path`, `pos`, `row`, `partition`, `spec_id`, `delete_file_path` (+ `content_offset`,
    /// `content_size_in_bytes` on format v3+); partition child ids are reassigned to the smallest
    /// positive ids unused by any table schema or metadata column; an EMPTY *unified* partition
    /// type drops the `partition` column entirely (`TypeUtil.selectNot(PARTITION_COLUMN_ID)`).
    pub fn schema(&self) -> Schema {
        let metadata = self.table.metadata();
        let partition_type = &self.unified_partition_type;
        let table_struct = metadata.current_schema().as_struct().clone();
        let format_version = metadata.format_version() as u8;

        let mut fields: Vec<NestedFieldRef> = vec![
            NestedField::required(
                DELETE_FILE_PATH_ID,
                "file_path",
                Type::Primitive(PrimitiveType::String),
            )
            .into(),
            NestedField::required(
                DELETE_FILE_POS_ID,
                "pos",
                Type::Primitive(PrimitiveType::Long),
            )
            .into(),
            NestedField::optional(DELETE_FILE_ROW_FIELD_ID, "row", Type::Struct(table_struct))
                .into(),
        ];

        // Java: partition child ids are reassigned before the schema is built (the remap
        // callback passed to `new Schema(...)` touches ONLY `idsToReassign`).
        if !partition_type.fields().is_empty() {
            let remapped = remap_partition_field_ids(self.table, partition_type);
            fields.push(
                NestedField::required(PARTITION_COLUMN_ID, "partition", Type::Struct(remapped))
                    .into(),
            );
        }

        fields.push(
            NestedField::required(
                SPEC_ID_COLUMN_ID,
                "spec_id",
                Type::Primitive(PrimitiveType::Int),
            )
            .into(),
        );
        fields.push(
            NestedField::required(
                FILE_PATH_COLUMN_ID,
                "delete_file_path",
                Type::Primitive(PrimitiveType::String),
            )
            .into(),
        );

        if format_version >= 3 {
            fields.push(
                NestedField::optional(
                    CONTENT_OFFSET_COLUMN_ID,
                    "content_offset",
                    Type::Primitive(PrimitiveType::Long),
                )
                .into(),
            );
            fields.push(
                NestedField::optional(
                    CONTENT_SIZE_IN_BYTES_COLUMN_ID,
                    "content_size_in_bytes",
                    Type::Primitive(PrimitiveType::Long),
                )
                .into(),
            );
        }

        Schema::builder()
            .with_fields(fields)
            .build()
            .expect("position_deletes metadata table schema is structurally valid")
    }

    /// Scans the `position_deletes` metadata table (Java `PositionDeletesBatchScan`).
    pub async fn scan(&self) -> Result<ArrowRecordBatchStream> {
        let metadata_schema = self.schema();
        let arrow_schema = Arc::new(schema_to_arrow_schema(&metadata_schema)?);
        let tasks = self.plan_position_delete_tasks(&metadata_schema).await?;
        let columns = self.read_planned_tasks(&tasks, &arrow_schema).await?;
        let batch = RecordBatch::try_new(arrow_schema, columns)?;
        Ok(stream::iter(vec![Ok(batch)]).boxed())
    }

    async fn plan_position_delete_tasks(
        &self,
        metadata_schema: &Schema,
    ) -> Result<Vec<PlannedPositionDelete>> {
        const CASE_SENSITIVE: bool = true;
        let metadata = self.table.metadata();
        let table_schema = metadata.current_schema().clone();

        let scan_filter =
            Predicate::AlwaysTrue.bind(Arc::new(metadata_schema.clone()), CASE_SENSITIVE)?;
        let base_filter = Predicate::AlwaysTrue.bind(table_schema.clone(), CASE_SENSITIVE)?;

        let reassigned = partition_id_reassignment(self.table, &self.unified_partition_type);
        let transformed_specs: HashMap<i32, PartitionSpecRef> = metadata
            .partition_specs_iter()
            .map(|spec| (spec.spec_id(), Arc::new(transform_spec(spec, &reassigned))))
            .collect();

        let mut transformed_evaluators: HashMap<i32, Arc<ManifestEvaluator>> = HashMap::new();
        let mut base_evaluators: HashMap<i32, Arc<ManifestEvaluator>> = HashMap::new();
        let mut partition_evaluators: HashMap<i32, Arc<ExpressionEvaluator>> = HashMap::new();
        let mut residual_evaluators: HashMap<i32, Arc<ResidualEvaluator>> = HashMap::new();

        let manifest_files =
            collect_manifest_files(self.table, MetadataScope::CurrentSnapshot).await?;
        let schema_fallback = Some(table_schema.clone());
        let mut tasks = Vec::new();
        for manifest_file in &manifest_files {
            if manifest_file.content != ManifestContentType::Deletes {
                continue;
            }
            let spec_id = manifest_file.partition_spec_id;
            let transformed_spec = transformed_specs.get(&spec_id).cloned().ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("no transformed partition spec for spec id {spec_id}"),
                )
            })?;
            let own_spec = metadata
                .partition_spec_by_id(spec_id)
                .cloned()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        format!("no partition spec for spec id {spec_id}"),
                    )
                })?;

            if let std::collections::hash_map::Entry::Vacant(entry) =
                transformed_evaluators.entry(spec_id)
            {
                let bound = project_partition_filter(
                    &transformed_spec,
                    metadata_schema,
                    &scan_filter,
                    CASE_SENSITIVE,
                )?;
                entry.insert(Arc::new(ManifestEvaluator::builder(bound).build()));
            }
            if let std::collections::hash_map::Entry::Vacant(entry) = base_evaluators.entry(spec_id)
            {
                let bound = project_partition_filter(
                    &own_spec,
                    &table_schema,
                    &base_filter,
                    CASE_SENSITIVE,
                )?;
                entry.insert(Arc::new(ManifestEvaluator::builder(bound.clone()).build()));
                partition_evaluators.insert(spec_id, Arc::new(ExpressionEvaluator::new(bound)));
            }
            if !transformed_evaluators[&spec_id].eval(manifest_file)? {
                continue;
            }
            if !base_evaluators[&spec_id].eval(manifest_file)? {
                continue;
            }

            let manifest = manifest_file
                .load_manifest_with_schema_fallback(self.table.file_io(), schema_fallback.clone())
                .await?;
            let partition_evaluator = partition_evaluators[&spec_id].clone();
            for entry in manifest.entries() {
                if !entry.is_alive() {
                    continue;
                }
                let data_file = entry.data_file();
                if data_file.content_type() != DataContentType::PositionDeletes {
                    continue;
                }
                if !partition_evaluator.eval(data_file)? {
                    continue;
                }
                let file_spec_id = data_file.partition_spec_id();
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    residual_evaluators.entry(file_spec_id)
                {
                    let file_transformed_spec = transformed_specs
                        .get(&file_spec_id)
                        .cloned()
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::Unexpected,
                                format!("no transformed partition spec for spec id {file_spec_id}"),
                            )
                        })?;
                    entry.insert(Arc::new(ResidualEvaluator::of(
                        file_transformed_spec,
                        metadata_schema,
                        scan_filter.clone(),
                        CASE_SENSITIVE,
                    )?));
                }
                let residual =
                    residual_evaluators[&file_spec_id].residual_for(data_file.partition())?;
                tasks.push(PlannedPositionDelete {
                    data_file: data_file.clone(),
                    spec_id: file_spec_id,
                    residual,
                });
            }
        }
        Ok(tasks)
    }

    async fn read_planned_tasks(
        &self,
        tasks: &[PlannedPositionDelete],
        arrow_schema: &arrow_schema::Schema,
    ) -> Result<Vec<ArrayRef>> {
        let metadata = self.table.metadata();
        let is_v3 = metadata.format_version() as u8 >= 3;
        let has_partition = !self.unified_partition_type.fields().is_empty();
        let row_fields = match arrow_schema.field_with_name("row")?.data_type() {
            DataType::Struct(fields) => fields.clone(),
            other => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("position_deletes row column is not a struct: {other}"),
                ));
            }
        };
        let partition_arrow_fields = if has_partition {
            match arrow_schema.field_with_name("partition")?.data_type() {
                DataType::Struct(fields) => Some(fields.clone()),
                other => {
                    return Err(Error::new(
                        ErrorKind::Unexpected,
                        format!("position_deletes partition column is not a struct: {other}"),
                    ));
                }
            }
        } else {
            None
        };
        let partition_field_ids = partition_field_ids_by_spec(metadata);

        let column_index = |name: &str| {
            arrow_schema
                .fields()
                .iter()
                .position(|field| field.name() == name)
                .expect("position_deletes arrow schema is built from the metadata schema")
        };
        let file_path_idx = column_index("file_path");
        let pos_idx = column_index("pos");
        let row_idx = column_index("row");
        let partition_idx = has_partition.then(|| column_index("partition"));
        let spec_idx = column_index("spec_id");
        let delete_file_path_idx = column_index("delete_file_path");
        let offset_idx = is_v3.then(|| column_index("content_offset"));
        let size_idx = is_v3.then(|| column_index("content_size_in_bytes"));

        let mut parts: Vec<Vec<ArrayRef>> = vec![Vec::new(); arrow_schema.fields().len()];
        for task in tasks {
            if !matches!(task.residual, Predicate::AlwaysTrue) {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    "position_deletes scan produced a non-vacuous residual: pushing a residual \
                     into the delete-file read requires the metadata-scan filter plumbing, \
                     which is a separate unit",
                ));
            }
            let data_file = &task.data_file;
            let (row_count, file_paths, positions, rows) =
                self.read_delete_file_rows(data_file, &row_fields).await?;
            parts[file_path_idx].push(file_paths);
            parts[pos_idx].push(positions);
            parts[row_idx].push(Arc::new(rows));
            if let Some(partition_idx) = partition_idx {
                let arrow_fields = partition_arrow_fields
                    .as_ref()
                    .expect("partition arrow fields resolved when the column exists");
                let source_field_ids = partition_field_ids
                    .get(&data_file.partition_spec_id())
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::Unexpected,
                            format!(
                                "no partition field ids for spec id {}",
                                data_file.partition_spec_id()
                            ),
                        )
                    })?;
                let mut builder = StructBuilder::from_fields(arrow_fields.clone(), row_count);
                for _ in 0..row_count {
                    append_partition(
                        &mut builder,
                        &self.unified_partition_type,
                        source_field_ids,
                        data_file.partition(),
                    )?;
                }
                parts[partition_idx].push(Arc::new(builder.finish()));
            }
            parts[spec_idx].push(Arc::new(Int32Array::from(vec![task.spec_id; row_count])));
            parts[delete_file_path_idx].push(Arc::new(StringArray::from(vec![
                data_file
                    .file_path();
                row_count
            ])));
            if let Some(offset_idx) = offset_idx {
                parts[offset_idx].push(Arc::new(Int64Array::from(vec![
                    data_file.content_offset();
                    row_count
                ])));
                parts[size_idx.expect("v3 size index resolved with the offset index")].push(
                    Arc::new(Int64Array::from(vec![
                        data_file.content_size_in_bytes();
                        row_count
                    ])),
                );
            }
        }

        arrow_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| match parts[index].len() {
                0 => Ok(arrow_array::new_empty_array(field.data_type())),
                1 => Ok(parts[index].pop().expect("single part")),
                _ => arrow_select::concat::concat(
                    &parts[index]
                        .iter()
                        .map(|array| array.as_ref() as &dyn Array)
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| Error::new(ErrorKind::Unexpected, format!("{error}"))),
            })
            .collect()
    }

    async fn read_delete_file_rows(
        &self,
        data_file: &DataFile,
        row_fields: &Fields,
    ) -> Result<(usize, ArrayRef, ArrayRef, StructArray)> {
        if data_file.file_format() == DataFileFormat::Puffin {
            self.read_deletion_vector_rows(data_file, row_fields).await
        } else {
            self.read_parquet_delete_rows(data_file, row_fields).await
        }
    }

    async fn read_deletion_vector_rows(
        &self,
        data_file: &DataFile,
        row_fields: &Fields,
    ) -> Result<(usize, ArrayRef, ArrayRef, StructArray)> {
        let referenced = data_file.referenced_data_file().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "deletion vector '{}' carries no referenced_data_file",
                    data_file.file_path()
                ),
            )
        })?;
        let delete_vector = load_delete_vector(self.table.file_io(), data_file).await?;
        let positions = delete_vector
            .iter()
            .map(|pos| {
                i64::try_from(pos).map_err(|_| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "deletion vector '{}' has position {pos} beyond the i64 range",
                            data_file.file_path()
                        ),
                    )
                })
            })
            .collect::<Result<Vec<i64>>>()?;
        let row_count = positions.len();
        let file_paths: ArrayRef =
            Arc::new(StringArray::from(vec![referenced.as_str(); row_count]));
        let positions_array: ArrayRef = Arc::new(Int64Array::from(positions));
        let rows = StructArray::new_null(row_fields.clone(), row_count);
        Ok((row_count, file_paths, positions_array, rows))
    }

    async fn read_parquet_delete_rows(
        &self,
        data_file: &DataFile,
        row_fields: &Fields,
    ) -> Result<(usize, ArrayRef, ArrayRef, StructArray)> {
        if data_file.file_format() != DataFileFormat::Parquet {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "position_deletes metadata table scan reads Parquet delete files and Puffin \
                     deletion vectors; '{}' is {:?}",
                    data_file.file_path(),
                    data_file.file_format()
                ),
            ));
        }
        let loader = BasicDeleteFileLoader::new(self.table.file_io().clone());
        let stream = loader
            .parquet_to_batch_stream_with_projection(
                data_file.file_path(),
                data_file.file_size_in_bytes(),
                None,
            )
            .await?;
        let batches = stream.try_collect::<Vec<RecordBatch>>().await?;
        let file_paths = concat_reserved_column(
            &batches,
            "file_path",
            RESERVED_FIELD_ID_DELETE_FILE_PATH,
            &DataType::Utf8,
            data_file.file_path(),
        )?;
        let positions = concat_reserved_column(
            &batches,
            "pos",
            RESERVED_FIELD_ID_DELETE_FILE_POS,
            &DataType::Int64,
            data_file.file_path(),
        )?;
        let row_count = positions.len();
        let rows = extract_row_column(&batches, row_fields, data_file.file_path())?;
        Ok((row_count, file_paths, positions, rows))
    }
}

struct PlannedPositionDelete {
    data_file: DataFile,
    spec_id: i32,
    residual: Predicate,
}

fn project_partition_filter(
    spec: &PartitionSpec,
    schema: &Schema,
    filter: &BoundPredicate,
    case_sensitive: bool,
) -> Result<BoundPredicate> {
    let projected = InclusiveProjection::new(Arc::new(spec.clone())).project(filter)?;
    let partition_type = spec.partition_type(schema)?;
    let partition_schema = Schema::builder()
        .with_schema_id(spec.spec_id())
        .with_fields(partition_type.fields().to_vec())
        .build()?;
    projected
        .rewrite_not()
        .bind(Arc::new(partition_schema), case_sensitive)
}

fn transform_spec(spec: &PartitionSpec, reassigned: &HashMap<i32, i32>) -> PartitionSpec {
    let fields = spec
        .fields()
        .iter()
        .map(|field| {
            let new_id = reassigned
                .get(&field.field_id)
                .copied()
                .unwrap_or(field.field_id);
            PartitionField {
                source_id: new_id,
                field_id: new_id,
                name: field.name.clone(),
                transform: Transform::Identity,
            }
        })
        .collect();
    PartitionSpec::from_fields_unchecked(spec.spec_id(), fields)
}

fn partition_id_reassignment(table: &Table, partition_type: &StructType) -> HashMap<i32, i32> {
    let mut used: std::collections::HashSet<i32> = [
        DELETE_FILE_PATH_ID,
        DELETE_FILE_POS_ID,
        DELETE_FILE_ROW_FIELD_ID,
        PARTITION_COLUMN_ID,
        SPEC_ID_COLUMN_ID,
        FILE_PATH_COLUMN_ID,
        CONTENT_OFFSET_COLUMN_ID,
        CONTENT_SIZE_IN_BYTES_COLUMN_ID,
    ]
    .into();
    for schema in table.metadata().schemas_iter() {
        collect_struct_field_ids(schema.as_struct(), &mut used);
    }

    let mut next_id = 0_i32;
    partition_type
        .fields()
        .iter()
        .map(|field| {
            loop {
                next_id += 1;
                if !used.contains(&next_id) {
                    break;
                }
            }
            (field.id, next_id)
        })
        .collect()
}

fn concat_reserved_column(
    batches: &[RecordBatch],
    name: &str,
    field_id: i32,
    expected: &DataType,
    delete_file_path: &str,
) -> Result<ArrayRef> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batches.len());
    for batch in batches {
        let index = batch
            .schema()
            .fields()
            .iter()
            .position(|field| {
                field
                    .metadata()
                    .get(PARQUET_FIELD_ID_META_KEY)
                    .and_then(|value| value.parse::<i32>().ok())
                    == Some(field_id)
                    || field.name() == name
            })
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("positional delete file '{delete_file_path}' has no '{name}' column"),
                )
            })?;
        let column = batch.column(index);
        if column.data_type() != expected {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "positional delete file '{delete_file_path}' column '{name}' is {}, expected \
                     {expected}",
                    column.data_type()
                ),
            ));
        }
        columns.push(column.clone());
    }
    match columns.len() {
        0 => Ok(arrow_array::new_empty_array(expected)),
        1 => Ok(columns.pop().expect("single column")),
        _ => arrow_select::concat::concat(
            &columns
                .iter()
                .map(|array| array.as_ref() as &dyn Array)
                .collect::<Vec<_>>(),
        )
        .map_err(|error| Error::new(ErrorKind::Unexpected, format!("{error}"))),
    }
}

fn extract_row_column(
    batches: &[RecordBatch],
    row_fields: &Fields,
    delete_file_path: &str,
) -> Result<StructArray> {
    let expected = DataType::Struct(row_fields.clone());
    let mut parts: Vec<StructArray> = Vec::with_capacity(batches.len());
    for batch in batches {
        match batch.schema().index_of("row") {
            Ok(index) => {
                let column = batch.column(index);
                if column.data_type() != &expected {
                    return Err(Error::new(
                        ErrorKind::FeatureUnsupported,
                        format!(
                            "positional delete file '{delete_file_path}' row column is {}, \
                             expected {expected}",
                            column.data_type()
                        ),
                    ));
                }
                let rows = column
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .expect("struct-typed row column downcasts to StructArray");
                parts.push(StructArray::new(
                    row_fields.clone(),
                    rows.columns().to_vec(),
                    rows.nulls().cloned(),
                ));
            }
            Err(_) => parts.push(StructArray::new_null(row_fields.clone(), batch.num_rows())),
        }
    }
    match parts.len() {
        0 => Ok(StructArray::new_null(row_fields.clone(), 0)),
        1 => Ok(parts.pop().expect("single part")),
        _ => {
            let concatenated = arrow_select::concat::concat(
                &parts
                    .iter()
                    .map(|array| array as &dyn Array)
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| Error::new(ErrorKind::Unexpected, format!("{error}")))?;
            concatenated
                .as_any()
                .downcast_ref::<StructArray>()
                .cloned()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "concatenated row column is not a struct",
                    )
                })
        }
    }
}

/// Java `calculateSchema` id reassignment: partition child ids move to the smallest positive
/// ids not used by any table schema (all schema versions) nor by the metadata columns / the
/// embedded `row` struct.
fn remap_partition_field_ids(table: &Table, partition_type: &StructType) -> StructType {
    let reassigned = partition_id_reassignment(table, partition_type);
    StructType::new(
        partition_type
            .fields()
            .iter()
            .map(|field| {
                NestedField::new(
                    reassigned[&field.id],
                    field.name.clone(),
                    (*field.field_type).clone(),
                    field.required,
                )
                .into()
            })
            .collect(),
    )
}

/// Recursively collect every field id reachable in `struct_type` (Java `TypeUtil.indexById`).
fn collect_struct_field_ids(struct_type: &StructType, used: &mut std::collections::HashSet<i32>) {
    for field in struct_type.fields() {
        used.insert(field.id);
        if let Type::Struct(nested) = field.field_type.as_ref() {
            collect_struct_field_ids(nested, used);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inspect::MetadataTableType;
    use crate::scan::tests::TableTestFixture;

    fn assert_field(schema: &Schema, index: usize, id: i32, name: &str, required: bool) {
        let field = &schema.as_struct().fields()[index];
        assert_eq!(field.id, id, "field[{index}] id ({name})");
        assert_eq!(field.name, name, "field[{index}] name");
        assert_eq!(field.required, required, "field[{index}] ({name}) required");
    }

    /// Cite: Java `PositionDeletesTable.calculateSchema` — fixed columns in builder order,
    /// v2 table (no DV columns), partition child ids reassigned off the table-schema id space.
    #[test]
    fn partitioned_schema_matches_java_calculate_schema() {
        let fixture = TableTestFixture::new();
        let schema = fixture.table.inspect().position_deletes().schema();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 6, "v2 partitioned column count");
        assert_field(&schema, 0, JAVA_INT_MAX - 101, "file_path", true);
        assert_field(&schema, 1, JAVA_INT_MAX - 102, "pos", true);
        assert_field(&schema, 2, JAVA_INT_MAX - 103, "row", false);
        assert_field(&schema, 3, JAVA_INT_MAX - 5, "partition", true);
        assert_field(&schema, 4, JAVA_INT_MAX - 4, "spec_id", true);
        assert_field(&schema, 5, JAVA_INT_MAX - 1, "delete_file_path", true);

        // Cite: `row` embeds the table schema struct as-is.
        let row = &fields[2];
        let Type::Struct(row_struct) = row.field_type.as_ref() else {
            panic!("row must be a struct");
        };
        assert_eq!(
            row_struct,
            fixture.table.metadata().current_schema().as_struct(),
            "row struct is the current table schema"
        );

        // Cite: the id-reassignment lambda — partition child ids move to the smallest positive
        // ids not used by ANY table schema nor the metadata columns.
        let Type::Struct(partition_struct) = fields[3].field_type.as_ref() else {
            panic!("partition must be a struct");
        };
        let mut used = std::collections::HashSet::new();
        for s in fixture.table.metadata().schemas_iter() {
            collect_struct_field_ids(s.as_struct(), &mut used);
        }
        for child in partition_struct.fields() {
            assert!(child.id > 0, "reassigned partition child id is positive");
            assert!(
                !used.contains(&child.id),
                "reassigned partition child id {} must not collide with any table schema id",
                child.id
            );
        }
    }

    /// Cite: Java `calculateSchema` tail — empty partition type returns
    /// `TypeUtil.selectNot(result, PARTITION_COLUMN_ID)` (drop, not empty struct).
    #[test]
    fn unpartitioned_schema_drops_partition_column() {
        let fixture = TableTestFixture::new_unpartitioned();
        let schema = fixture.table.inspect().position_deletes().schema();
        let fields = schema.as_struct().fields();
        assert_eq!(fields.len(), 5, "v2 unpartitioned column count");
        assert!(
            schema.field_by_id(PARTITION_COLUMN_ID).is_none(),
            "unpartitioned position_deletes must drop field {PARTITION_COLUMN_ID}"
        );
        assert_field(&schema, 0, JAVA_INT_MAX - 101, "file_path", true);
        assert_field(&schema, 3, JAVA_INT_MAX - 4, "spec_id", true);
        assert_field(&schema, 4, JAVA_INT_MAX - 1, "delete_file_path", true);
    }

    /// Cite: Java `MetadataTableType.from` — vocabulary + `$`-suffix resolution key.
    #[test]
    fn metadata_table_type_round_trips() {
        let ty = MetadataTableType::try_from("position_deletes").expect("vocabulary");
        assert_eq!(ty.as_str(), "position_deletes");
        assert!(
            MetadataTableType::all_types().any(|t| t.as_str() == "position_deletes"),
            "all_types must include position_deletes"
        );
    }

    /// Increment D: unified type has two children under widening evolution,
    /// so the remapped `partition` struct has two fields (default spec would
    /// also be two here; the next test is the discriminating one).
    #[test]
    fn widening_schema_partition_has_two_remapped_children() {
        let fixture = TableTestFixture::new_with_widening_spec_evolution();
        let schema = fixture.table.inspect().position_deletes().schema();
        let partition = schema
            .field_by_id(PARTITION_COLUMN_ID)
            .expect("widening table keeps partition");
        let Type::Struct(partition_struct) = partition.field_type.as_ref() else {
            panic!("partition must be a struct");
        };
        assert_eq!(
            partition_struct.fields().len(),
            2,
            "unified type {{x, y_bucket_8}} remaps two children"
        );
        assert_eq!(partition_struct.fields()[0].name, "x");
        assert_eq!(partition_struct.fields()[1].name, "y_bucket_8");
    }

    /// Discriminator vs default_partition_type: current spec is unpartitioned
    /// but unified type is `{x}`, so `partition` stays (Java empty-drop fires
    /// on `Partitioning.partitionType`, not the default spec).
    #[test]
    fn evolved_to_unpartitioned_keeps_partition_column() {
        let fixture = TableTestFixture::new_evolved_to_unpartitioned();
        let schema = fixture.table.inspect().position_deletes().schema();
        assert!(
            schema.field_by_id(PARTITION_COLUMN_ID).is_some(),
            "historical spec 0 keeps the partition column"
        );
        let partition = schema.field_by_id(PARTITION_COLUMN_ID).unwrap();
        let Type::Struct(partition_struct) = partition.field_type.as_ref() else {
            panic!("partition must be a struct");
        };
        assert_eq!(partition_struct.fields().len(), 1);
        assert_eq!(partition_struct.fields()[0].name, "x");
    }

    /// `try_new` is the G2 refuse path. `new_with_two_identity_specs` is a
    /// Java-invalid unifier input (field id 1000 reused for two sources) and
    /// is used here ONLY as a refusal pin, never as a successful unifier input.
    #[test]
    fn try_new_refuses_conflicting_field_ids() {
        let fixture = TableTestFixture::new_with_two_identity_specs();
        let error = match PositionDeletesTable::try_new(&fixture.table) {
            Ok(_) => panic!("G2: conflicting field ids must refuse try_new"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().starts_with("Conflicting partition fields"),
            "message was: {}",
            error.message()
        );
    }
}

#[cfg(test)]
mod scan_tests;
