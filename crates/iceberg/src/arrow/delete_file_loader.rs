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

use std::collections::HashSet;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::schema::types::SchemaDescriptor;

use crate::arrow::ArrowReader;
use crate::arrow::reader::ParquetReadOptions;
use crate::arrow::record_batch_transformer::RecordBatchTransformerBuilder;
use crate::io::FileIO;
use crate::metadata_columns::{
    RESERVED_COL_NAME_DELETE_FILE_PATH, RESERVED_COL_NAME_DELETE_FILE_POS,
    RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
};
use crate::scan::{ArrowRecordBatchStream, FileScanTaskDeleteFile};
use crate::spec::{Schema, SchemaRef};
use crate::{Error, ErrorKind, Result};

/// Delete File Loader
#[allow(unused)]
#[async_trait::async_trait]
pub trait DeleteFileLoader {
    /// Read the delete file referred to in the task
    ///
    /// Returns the contents of the delete file as a RecordBatch stream. Applies schema evolution.
    async fn read_delete_file(
        &self,
        task: &FileScanTaskDeleteFile,
        schema: SchemaRef,
    ) -> Result<ArrowRecordBatchStream>;
}

#[derive(Clone, Debug)]
pub(crate) struct BasicDeleteFileLoader {
    file_io: FileIO,
}

#[allow(unused_variables)]
impl BasicDeleteFileLoader {
    pub fn new(file_io: FileIO) -> Self {
        BasicDeleteFileLoader { file_io }
    }

    /// Loads a RecordBatchStream for a given datafile (full column projection).
    ///
    /// Prefer [`Self::parquet_to_batch_stream_with_projection`] when only a known subset of
    /// columns is required (positional deletes: `file_path` + `pos`; equality deletes: the
    /// `equality_ids` key columns).
    pub(crate) async fn parquet_to_batch_stream(
        &self,
        data_file_path: &str,
        file_size_in_bytes: u64,
    ) -> Result<ArrowRecordBatchStream> {
        self.parquet_to_batch_stream_with_projection(data_file_path, file_size_in_bytes, None)
            .await
    }

    /// Loads a RecordBatchStream, optionally projecting only `project_field_ids` leaf columns.
    ///
    /// Projection is best-effort (Wave B): when field ids are present in the Parquet/Arrow
    /// metadata we build a [`ProjectionMask`]; if the mask cannot be built safely (missing
    /// field-id metadata, incomplete match), we fall back to a full read rather than risk
    /// misreading columns. Name-based fallback for positional-delete reserved columns (`file_path` /
    /// `pos`) is attempted when field ids are absent but writers emitted the standard names.
    pub(crate) async fn parquet_to_batch_stream_with_projection(
        &self,
        data_file_path: &str,
        file_size_in_bytes: u64,
        project_field_ids: Option<&[i32]>,
    ) -> Result<ArrowRecordBatchStream> {
        /*
           Essentially a super-cut-down ArrowReader. We can't use ArrowReader directly
           as that introduces a circular dependency.
        */
        let parquet_read_options = ParquetReadOptions::builder().build();

        let (parquet_file_reader, arrow_metadata) = ArrowReader::open_parquet_file(
            data_file_path,
            &self.file_io,
            file_size_in_bytes,
            parquet_read_options,
        )
        .await?;

        let mut builder =
            ParquetRecordBatchStreamBuilder::new_with_metadata(parquet_file_reader, arrow_metadata);

        // Best-effort projection: if the mask cannot be built, fall back to a full read.
        if let Some(field_ids) = project_field_ids
            && let Some(mask) = try_build_delete_projection_mask(
                field_ids,
                builder.parquet_schema(),
                builder.schema(),
            )
        {
            builder = builder.with_projection(mask);
        }

        let record_batch_stream = builder
            .build()?
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}")));

        Ok(Box::pin(record_batch_stream) as ArrowRecordBatchStream)
    }

    /// Project only the reserved positional-delete columns (`file_path` + `pos`).
    ///
    /// Used by the MoR pos-delete load path so the optional third `row` column (and any other
    /// extras) are not decoded. Falls back to a full read when projection cannot be built.
    pub(crate) async fn parquet_positional_delete_batch_stream(
        &self,
        data_file_path: &str,
        file_size_in_bytes: u64,
    ) -> Result<ArrowRecordBatchStream> {
        self.parquet_to_batch_stream_with_projection(
            data_file_path,
            file_size_in_bytes,
            Some(&[
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                RESERVED_FIELD_ID_DELETE_FILE_POS,
            ]),
        )
        .await
    }

    /// Reads `length` bytes at `offset` from `file_path`.
    ///
    /// This is the deletion-vector read primitive: Java reads a DV blob with a single ranged
    /// read at the `DeleteFile`'s `content_offset` / `content_size_in_bytes`
    /// (`BaseDeleteLoader.readDV`, data/.../BaseDeleteLoader.java L171-183) rather than through
    /// the Puffin footer — the footer route would take 3+ requests per file (its own doc
    /// comment, L143-147) and the manifest already carries the exact blob range.
    pub(crate) async fn read_bytes_range(
        &self,
        file_path: &str,
        offset: u64,
        length: u64,
    ) -> Result<bytes::Bytes> {
        let end = offset.checked_add(length).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid byte range for delete file '{file_path}': offset {offset} + length \
                     {length} overflows"
                ),
            )
        })?;
        let input_file = self.file_io.new_input(file_path)?;
        let reader = input_file.reader().await?;
        reader.read(offset..end).await
    }

    /// Evolves the schema of the RecordBatches from an equality delete file.
    ///
    /// Per the [Iceberg spec](https://iceberg.apache.org/spec/#equality-delete-files),
    /// only evolves the specified `equality_ids` columns, not all table columns.
    pub(crate) async fn evolve_schema(
        record_batch_stream: ArrowRecordBatchStream,
        target_schema: Arc<Schema>,
        equality_ids: &[i32],
    ) -> Result<ArrowRecordBatchStream> {
        let mut record_batch_transformer =
            RecordBatchTransformerBuilder::new(target_schema.clone(), equality_ids).build();

        let record_batch_stream = record_batch_stream.map(move |record_batch| {
            record_batch.and_then(|record_batch| {
                record_batch_transformer.process_record_batch(record_batch)
            })
        });

        Ok(Box::pin(record_batch_stream) as ArrowRecordBatchStream)
    }
}

/// Best-effort projection mask for delete-file columns.
///
/// Returns `Some(mask)` only when every requested field id (or, for the positional-delete
/// reserved pair, the standard `file_path` / `pos` names) resolves to a Parquet leaf index.
/// Returns `None` when projection cannot be built safely — callers must fall back to a full
/// read so parsing never breaks on exotic layouts.
fn try_build_delete_projection_mask(
    field_ids: &[i32],
    parquet_schema: &SchemaDescriptor,
    arrow_schema: &arrow_schema::SchemaRef,
) -> Option<ProjectionMask> {
    if field_ids.is_empty() {
        return None;
    }

    let wanted: HashSet<i32> = field_ids.iter().copied().collect();
    let mut found: HashSet<i32> = HashSet::new();
    let mut indices: Vec<usize> = Vec::new();

    // Prefer field-id metadata (Iceberg writers stamp PARQUET_FIELD_ID_META_KEY).
    arrow_schema.fields().filter_leaves(|idx, field| {
        if let Some(id_str) = field.metadata().get(PARQUET_FIELD_ID_META_KEY)
            && let Ok(id) = id_str.parse::<i32>()
            && wanted.contains(&id)
        {
            indices.push(idx);
            found.insert(id);
            return true;
        }
        false
    });

    if found.len() == wanted.len() && !indices.is_empty() {
        return Some(ProjectionMask::leaves(parquet_schema, indices));
    }

    // Name-based fallback for the positional-delete reserved pair when field ids are missing
    // (some writers emit `file_path` / `pos` by name only). Only safe when the request is
    // exactly those two reserved ids — equality-delete keys must not be guessed by name.
    let is_pos_delete_projection = wanted.len() == 2
        && wanted.contains(&RESERVED_FIELD_ID_DELETE_FILE_PATH)
        && wanted.contains(&RESERVED_FIELD_ID_DELETE_FILE_POS);
    if !is_pos_delete_projection {
        // Incomplete field-id match or non-pos projection: refuse rather than guess.
        return None;
    }

    let mut name_indices: Vec<usize> = Vec::new();
    let mut saw_path = false;
    let mut saw_pos = false;
    arrow_schema
        .fields()
        .filter_leaves(|idx, field| match field.name().as_str() {
            RESERVED_COL_NAME_DELETE_FILE_PATH if !saw_path => {
                name_indices.push(idx);
                saw_path = true;
                true
            }
            RESERVED_COL_NAME_DELETE_FILE_POS if !saw_pos => {
                name_indices.push(idx);
                saw_pos = true;
                true
            }
            _ => false,
        });
    if saw_path && saw_pos && !name_indices.is_empty() {
        Some(ProjectionMask::leaves(parquet_schema, name_indices))
    } else {
        None
    }
}

#[async_trait::async_trait]
impl DeleteFileLoader for BasicDeleteFileLoader {
    async fn read_delete_file(
        &self,
        task: &FileScanTaskDeleteFile,
        schema: SchemaRef,
    ) -> Result<ArrowRecordBatchStream> {
        // Equality deletes: project only equality_ids, then evolve those columns.
        // Positional deletes: project only file_path + pos (Wave B); fall back to full read if
        // the projection mask cannot be built. Evolve field selection is unchanged.
        let raw_batch_stream = match &task.equality_ids {
            Some(ids) => {
                self.parquet_to_batch_stream_with_projection(
                    &task.file_path,
                    task.file_size_in_bytes,
                    Some(ids.as_slice()),
                )
                .await?
            }
            None => {
                self.parquet_positional_delete_batch_stream(
                    &task.file_path,
                    task.file_size_in_bytes,
                )
                .await?
            }
        };

        // For equality deletes, only evolve the equality_ids columns.
        // For positional deletes (equality_ids is None), use all field IDs (historical contract).
        let field_ids = match &task.equality_ids {
            Some(ids) => ids.clone(),
            None => schema.field_id_to_name_map().keys().cloned().collect(),
        };

        Self::evolve_schema(raw_batch_stream, schema, &field_ids).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::delete_filter::tests::setup;

    #[tokio::test]
    async fn test_basic_delete_file_loader_read_delete_file() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let delete_file_loader = BasicDeleteFileLoader::new(file_io.clone());

        let file_scan_tasks = setup(table_location);

        let result = delete_file_loader
            .read_delete_file(
                &file_scan_tasks[0].deletes[0],
                file_scan_tasks[0].schema_ref(),
            )
            .await
            .expect("read delete file");

        let result = result
            .try_collect::<Vec<_>>()
            .await
            .expect("collect delete batches");

        assert_eq!(result.len(), 1);
    }

    /// Positional-delete load with projection: a file carrying the optional third `row` column
    /// still yields path+pos when we project by reserved field id (Wave B).
    #[tokio::test]
    async fn test_positional_delete_projection_drops_row_column() {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp
            .path()
            .join("pos-del-with-row.parquet")
            .to_string_lossy()
            .to_string();

        // Three columns: reserved file_path + pos + optional row payload.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                RESERVED_FIELD_ID_DELETE_FILE_PATH.to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                RESERVED_FIELD_ID_DELETE_FILE_POS.to_string(),
            )])),
            Field::new("row", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "999".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(vec!["data.parquet", "data.parquet"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1i64, 3i64])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])) as ArrayRef,
        ])
        .expect("build three-column pos-delete batch");
        let file = File::create(&path).expect("create file");
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");

        let file_io = FileIO::new_with_fs();
        let loader = BasicDeleteFileLoader::new(file_io);
        let stream = loader
            .parquet_positional_delete_batch_stream(
                &path,
                std::fs::metadata(&path).expect("stat").len(),
            )
            .await
            .expect("projected pos-delete stream");
        let batches = stream
            .try_collect::<Vec<_>>()
            .await
            .expect("collect projected batches");

        assert_eq!(batches.len(), 1, "one batch");
        let batch = &batches[0];
        // Projection should keep only the two reserved columns (not `row`).
        assert_eq!(
            batch.num_columns(),
            2,
            "projected pos-delete batch must carry only file_path + pos, got {:?}",
            batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>()
        );
        assert!(batch.column_by_name("file_path").is_some());
        assert!(batch.column_by_name("pos").is_some());
        assert!(
            batch.column_by_name("row").is_none(),
            "optional row column must be projected out"
        );
        assert_eq!(batch.num_rows(), 2);
        let pos = batch
            .column_by_name("pos")
            .expect("pos")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 pos");
        assert_eq!(pos.values(), &[1, 3]);
    }

    /// Name-based fallback: a pos-delete file WITHOUT field-id metadata still projects when
    /// columns are named `file_path` / `pos`.
    #[tokio::test]
    async fn test_positional_delete_projection_name_fallback() {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp
            .path()
            .join("pos-del-no-ids.parquet")
            .to_string_lossy()
            .to_string();

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false),
            Field::new("pos", DataType::Int64, false),
            Field::new("extra", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(vec!["d.parquet"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![7i64])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("x")])) as ArrayRef,
        ])
        .expect("batch");
        let file = File::create(&path).expect("create");
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");

        let loader = BasicDeleteFileLoader::new(FileIO::new_with_fs());
        let batches = loader
            .parquet_positional_delete_batch_stream(
                &path,
                std::fs::metadata(&path).expect("stat").len(),
            )
            .await
            .expect("stream")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect");

        assert_eq!(batches[0].num_columns(), 2);
        assert!(batches[0].column_by_name("extra").is_none());
        assert_eq!(
            batches[0]
                .column_by_name("pos")
                .expect("pos")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("i64")
                .values(),
            &[7]
        );
    }

    /// Equality-delete path still projects `equality_ids` (and still evolves those columns).
    #[tokio::test]
    async fn test_equality_delete_projection_keeps_key_columns() {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp
            .path()
            .join("eq-del.parquet")
            .to_string_lossy()
            .to_string();

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("payload", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(Int64Array::from(vec![10i64, 20i64])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])) as ArrayRef,
        ])
        .expect("batch");
        let file = File::create(&path).expect("create");
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                .expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");

        let loader = BasicDeleteFileLoader::new(FileIO::new_with_fs());
        let batches = loader
            .parquet_to_batch_stream_with_projection(
                &path,
                std::fs::metadata(&path).expect("stat").len(),
                Some(&[1]),
            )
            .await
            .expect("stream")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect");

        assert_eq!(batches[0].num_columns(), 1);
        assert!(batches[0].column_by_name("id").is_some());
        assert!(batches[0].column_by_name("payload").is_none());
    }

    /// End-to-end: `read_delete_file` on a positional-delete task from the shared fixture still
    /// succeeds under the Wave B projection path (setup's empty table schema means evolve keeps
    /// zero table columns — raw projection of path+pos is covered by the dedicated projection
    /// tests above).
    #[tokio::test]
    async fn test_read_delete_file_pos_delete_with_projection() {
        let tmp = TempDir::new().expect("tempdir");
        let table_location = tmp.path();
        let file_io = FileIO::new_with_fs();
        let file_scan_tasks = setup(table_location);
        let loader = BasicDeleteFileLoader::new(file_io);

        let stream = loader
            .read_delete_file(
                &file_scan_tasks[0].deletes[0],
                file_scan_tasks[0].schema_ref(),
            )
            .await
            .expect("read pos delete");
        let batches = stream.try_collect::<Vec<_>>().await.expect("collect");
        assert_eq!(
            batches.len(),
            1,
            "one batch from the fixture pos-delete file"
        );
        // setup() writes 8 rows per pos-delete file — projection must not drop rows.
        assert_eq!(batches[0].num_rows(), 8);
    }

    /// Critic-octo C4-Q-001: pos-delete projection with only ONE reserved field id present
    /// still falls back to name-based selection of `file_path` + `pos` (not a partial mask).
    #[tokio::test]
    async fn test_positional_delete_projection_partial_field_ids_name_fallback() {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp
            .path()
            .join("pos-partial-ids.parquet")
            .to_string_lossy()
            .to_string();
        // file_path has reserved field id; pos has name only (no field id).
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                RESERVED_FIELD_ID_DELETE_FILE_PATH.to_string(),
            )])),
            Field::new("pos", DataType::Int64, false),
            Field::new("extra", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(vec!["d.parquet"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![11i64])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("x")])) as ArrayRef,
        ])
        .expect("batch");
        {
            let file = File::create(&path).expect("create");
            let mut writer =
                ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))
                    .expect("writer");
            writer.write(&batch).expect("write");
            writer.close().expect("close");
        }
        let loader = BasicDeleteFileLoader::new(FileIO::new_with_fs());
        let batches = loader
            .parquet_positional_delete_batch_stream(
                &path,
                std::fs::metadata(&path).expect("stat").len(),
            )
            .await
            .expect("stream")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect");
        assert_eq!(
            batches[0].num_columns(),
            2,
            "name fallback must project path+pos only"
        );
        assert!(batches[0].column_by_name("extra").is_none());
        assert_eq!(
            batches[0]
                .column_by_name("pos")
                .expect("pos")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("i64")
                .values(),
            &[11]
        );
    }

    /// Critic-octo C1-Q-003: incomplete field-id match must refuse the mask (→ full read),
    /// never return a partial ProjectionMask that could drop a requested leaf silently.
    #[test]
    fn test_try_build_delete_projection_mask_incomplete_field_ids_returns_none() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            // payload has NO field-id metadata — wanted id 2 is unresolvable.
            Field::new("payload", DataType::Utf8, true),
        ]));
        // Build a minimal parquet schema descriptor by writing a throwaway file — the mask
        // builder only needs leaf count alignment with the Arrow schema.
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("incomplete-ids.parquet");
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int64Array::from(vec![1i64])) as ArrayRef,
            Arc::new(StringArray::from(vec![Some("x")])) as ArrayRef,
        ])
        .expect("batch");
        {
            let file = File::create(&path).expect("create");
            let mut writer = ArrowWriter::try_new(
                file,
                arrow_schema.clone(),
                Some(WriterProperties::builder().build()),
            )
            .expect("writer");
            writer.write(&batch).expect("write");
            writer.close().expect("close");
        }
        let file = File::open(&path).expect("open");
        let meta =
            parquet::arrow::arrow_reader::ArrowReaderMetadata::load(&file, Default::default())
                .expect("meta");
        let mask = try_build_delete_projection_mask(&[1, 2], meta.parquet_schema(), meta.schema());
        assert!(
            mask.is_none(),
            "incomplete field-id match must return None (full-read fallback), got {mask:?}"
        );
        // Control: single resolvable id still builds a mask.
        let mask_ok = try_build_delete_projection_mask(&[1], meta.parquet_schema(), meta.schema());
        assert!(
            mask_ok.is_some(),
            "complete single-id match must build a mask"
        );
    }
}
