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

//! Deletion-vector (V3 Puffin DV) file writer, [`DVFileWriter`]. Java `BaseDVFileWriter`.
//! One Puffin, one `deletion-vector-v1` blob per referenced data file, sorted by path.
//! [`DVFileWriter::with_previous_deletes`] unions prior positions and returns file-scoped sources.

use std::collections::{BTreeMap, HashMap, HashSet};

mod file_scope;

use file_scope::is_file_scoped;

use crate::delete_vector::DeleteVector;
use crate::io::OutputFile;
use crate::metadata_columns::RESERVED_FIELD_ID_POS;
use crate::puffin::{Blob, CompressionCodec, DELETION_VECTOR_V1, PuffinWriter};
use crate::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, PartitionKey, PartitionSpec,
};
use crate::writer::base_writer::data_file_writer::resolve_partition_spec_id;
use crate::{Error, ErrorKind, Result};

/// The largest position a deletion vector may record. Mirrors Java
/// `RoaringPositionBitmap.MAX_POSITION`. The bitmap key holds at most `i32::MAX - 1`. The low
/// word is `Integer.MIN_VALUE` read unsigned, so it is `0x8000_0000`, not `0xFFFF_FFFF`. That
/// quirk is Java's, and the fork mirrors it.
pub const DV_MAX_POSITION: u64 = ((i32::MAX as u64 - 1) << 32) | 0x8000_0000;

/// Snapshot id and sequence number on a DV blob. −1 means the commit inherits them.
const INHERITED: i64 = -1;

/// Puffin blob property naming the data file a deletion vector applies to.
const REFERENCED_DATA_FILE_PROPERTY: &str = "referenced-data-file";

/// Puffin blob property carrying the number of deleted positions.
const CARDINALITY_PROPERTY: &str = "cardinality";

/// Per-data-file accumulation state: the position set and the partition context. The first
/// `delete` call for a path captures the partition.
#[derive(Debug)]
struct DeletesForDataFile {
    positions: DeleteVector,
    partition_key: Option<PartitionKey>,
}

/// A data file's previous deletes, for [`DVFileWriter::with_previous_deletes`].
/// Java `loadPreviousDeletes` / `PositionDeleteIndex`.
#[derive(Debug, Clone)]
pub struct PreviousDeletes {
    /// The data file's existing deleted positions. Load them through the production read path,
    /// not by hand.
    positions: DeleteVector,
    /// The delete files those positions came from. Each file-scoped entry becomes a rewritten
    /// delete file after the merge.
    source_delete_files: Vec<DataFile>,
}

impl PreviousDeletes {
    /// Build a `PreviousDeletes` from a data file's existing positions and the delete files they
    /// came from. The merge may mark those delete files as superseded.
    pub fn new(positions: DeleteVector, source_delete_files: Vec<DataFile>) -> Self {
        Self {
            positions,
            source_delete_files,
        }
    }
}

/// The result of [`DVFileWriter::close_with_result`]. Mirrors Java `DeleteWriteResult`.
///
/// Java also carries `referencedDataFiles` for conflict validation. `delete_files` already
/// determines it, so accessors derive it. A stored field could drift.
#[derive(Debug)]
pub struct DVWriteResult {
    /// One DV `DeleteFile` per referenced data file (Java `DeleteWriteResult.deleteFiles()`); the
    /// same value [`DVFileWriter::close`] returns. Feed these to `RowDelta.add_deletes`.
    pub delete_files: Vec<DataFile>,
    /// The FILE-SCOPED previous delete files that the merged DVs supersede (Java
    /// `DeleteWriteResult.rewrittenDeleteFiles()`). Feed these to `RowDelta.remove_deletes_many`.
    /// Non-file-scoped previous deletes (partition-scoped parquet position deletes) are NOT included
    /// — Java leaves them in the table (`BaseDVFileWriter` L121-124).
    pub rewritten_delete_files: Vec<DataFile>,
}

impl DVWriteResult {
    /// The set of data file paths the written DVs reference — Java
    /// `DeleteWriteResult.referencedDataFiles()`, which `RowDelta.validateDataFilesExist` consumes.
    /// Derived from [`delete_files`](Self::delete_files); `DataFileBuilder::build` rejects a DV
    /// with no `referenced_data_file`, so the `filter_map` is total.
    pub fn referenced_data_files(&self) -> HashSet<String> {
        self.delete_files
            .iter()
            .filter_map(|delete_file| delete_file.referenced_data_file())
            .collect()
    }

    /// Whether any data file is referenced — Java `DeleteWriteResult.referencesDataFiles()`. Avoids
    /// allocating the set.
    pub fn references_data_files(&self) -> bool {
        self.delete_files
            .iter()
            .any(|delete_file| delete_file.referenced_data_file().is_some())
    }
}

/// Writer for deletion vectors (V3 Puffin DVs), mirroring Java `BaseDVFileWriter`. Accumulate
/// deleted positions with [`delete`](Self::delete), then call [`close`](Self::close) for the DVs
/// alone, or [`close_with_result`](Self::close_with_result) for the DVs plus the superseded delete
/// files. Either writes one Puffin file and returns the per-data-file `DeleteFile` metadata for a
/// row-delta commit.
#[derive(Debug)]
pub struct DVFileWriter {
    /// Where the Puffin file goes. The underlying file is only created when `close()` actually
    /// has deletes to write (Java defers via a `Supplier<OutputFile>`; an [`OutputFile`] is
    /// equally lazy — no bytes hit storage until a writer is opened on it).
    output_file: OutputFile,
    /// Per referenced data file path, in sorted order (see the module docs on determinism).
    deletes_by_path: BTreeMap<String, DeletesForDataFile>,
    /// Per referenced data file path, the PREVIOUS deletes to merge in at close time (Java's
    /// `loadPreviousDeletes`). Empty unless [`with_previous_deletes`](Self::with_previous_deletes)
    /// was called.
    previous_deletes_by_path: HashMap<String, PreviousDeletes>,
    /// The spec to stamp on a DV whose [`delete`](Self::delete) calls carried no [`PartitionKey`].
    /// `None` errors at close. See [`unpartitioned`](Self::unpartitioned).
    partition_spec: Option<PartitionSpec>,
}

impl DVFileWriter {
    /// Creates a new `DVFileWriter` that will write its single Puffin file to `output_file`
    /// (only if at least one position is deleted before `close`).
    pub fn new(output_file: OutputFile) -> Self {
        Self {
            output_file,
            deletes_by_path: BTreeMap::new(),
            previous_deletes_by_path: HashMap::new(),
            partition_spec: None,
        }
    }

    /// Stamp [`PartitionSpec::unpartition_spec`] (spec id 0, no fields).
    pub fn unpartitioned(self) -> Self {
        self.with_partition_spec(PartitionSpec::unpartition_spec())
    }

    /// Spec to stamp when [`delete`](Self::delete) carries no [`PartitionKey`]. Java
    /// `BaseDVFileWriter.delete` takes the spec per call.
    pub fn with_partition_spec(mut self, partition_spec: PartitionSpec) -> Self {
        self.partition_spec = Some(partition_spec);
        self
    }

    /// Supply each data file's previous deletes to merge at close time. Mirrors the Java
    /// `loadPreviousDeletes` constructor argument. # Notes The merge visits only a path that also
    /// has new positions from [`delete`](Self::delete).
    pub fn with_previous_deletes(
        mut self,
        previous_deletes_by_path: HashMap<String, PreviousDeletes>,
    ) -> Self {
        self.previous_deletes_by_path = previous_deletes_by_path;
        self
    }

    /// Marks `position` of the data file at `data_file_path` as deleted, in the partition context
    /// `partition_key` (`None` for an unpartitioned table). Mirrors Java `BaseDVFileWriter.delete`.
    /// The first call for a path captures the partition context.
    pub fn delete(
        &mut self,
        data_file_path: &str,
        position: u64,
        partition_key: Option<&PartitionKey>,
    ) -> Result<()> {
        if position > DV_MAX_POSITION {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Deletion vector supports positions that are >= 0 and <= {DV_MAX_POSITION}: \
                     {position} (Java RoaringPositionBitmap.MAX_POSITION)"
                ),
            ));
        }

        if let Some(deletes) = self.deletes_by_path.get_mut(data_file_path) {
            deletes.positions.insert(position);
            return Ok(());
        }
        let mut deletes = DeletesForDataFile {
            positions: DeleteVector::default(),
            partition_key: partition_key.cloned(),
        };
        deletes.positions.insert(position);
        self.deletes_by_path
            .insert(data_file_path.to_string(), deletes);
        Ok(())
    }

    /// Write every accumulated deletion vector into one Puffin file. Return one `DeleteFile` per
    /// referenced data file, and discard the rewritten delete files.
    ///
    /// With no recorded deletes this writes no file and returns an empty vec. Use
    /// [`close_with_result`](Self::close_with_result) when the commit must also remove the
    /// superseded delete files.
    pub async fn close(self) -> Result<Vec<DataFile>> {
        Ok(self.close_with_result().await?.delete_files)
    }

    /// Write every accumulated deletion vector into one Puffin file. Return the DV `DeleteFile`s
    /// and the superseded file-scoped delete files. Mirrors Java `BaseDVFileWriter.close`.
    pub async fn close_with_result(mut self) -> Result<DVWriteResult> {
        // Merge each file's previous deletes into its new DV. Only a path with new positions is
        // visited, as in Java. Collect the file-scoped superseded sources.
        let mut rewritten_delete_files: Vec<DataFile> = Vec::new();
        for (data_file_path, deletes) in &mut self.deletes_by_path {
            let Some(previous) = self.previous_deletes_by_path.get(data_file_path) else {
                continue;
            };
            deletes.positions.merge(&previous.positions);
            for source_file in &previous.source_delete_files {
                // "only DVs and file-scoped deletes can be discarded from the table state"
                // (BaseDVFileWriter L121-124).
                if is_file_scoped(source_file) {
                    rewritten_delete_files.push(source_file.clone());
                }
            }
        }

        if self.deletes_by_path.is_empty() {
            return Ok(DVWriteResult {
                delete_files: Vec::new(),
                rewritten_delete_files,
            });
        }

        // Resolve every spec id before the Puffin file opens. A partitioned spec with no
        // PartitionKey fails here. A resolve after the write would leave a written, unreferenced
        // Puffin file on storage. This writer takes its key per `delete` call, so it cannot
        // resolve at build time like the other base writers.
        let spec_ids = self
            .deletes_by_path
            .values()
            .map(|deletes| {
                resolve_partition_spec_id(
                    self.partition_spec.as_ref(),
                    deletes.partition_key.as_ref(),
                )
            })
            .collect::<Result<Vec<i32>>>()?;

        // One Puffin file for ALL the vectors. The footer is uncompressed; `created-by`
        // identifies this writer (Java sets `IcebergBuild.fullVersion()` — the value differs,
        // which is footer-cosmetic and reader-irrelevant).
        let mut puffin_writer = PuffinWriter::new(
            &self.output_file,
            std::collections::HashMap::from([(
                crate::puffin::CREATED_BY_PROPERTY.to_string(),
                format!("iceberg-rust {}", env!("CARGO_PKG_VERSION")),
            )]),
            false,
        )
        .await?;

        // One uncompressed blob per referenced data file, in sorted path order for determinism.
        // The returned BlobMetadata carries the offset and length the DeleteFile references. The
        // positions already include the merged previous deletes.
        let mut blob_coordinates: Vec<(u64, u64)> = Vec::with_capacity(self.deletes_by_path.len());
        for (data_file_path, deletes) in &self.deletes_by_path {
            let blob_data = deletes.positions.serialize_deletion_vector_v1()?;
            let blob = Blob::builder()
                .r#type(DELETION_VECTOR_V1.to_string())
                .fields(vec![RESERVED_FIELD_ID_POS])
                .snapshot_id(INHERITED)
                .sequence_number(INHERITED)
                .data(blob_data)
                .properties(std::collections::HashMap::from([
                    (
                        REFERENCED_DATA_FILE_PROPERTY.to_string(),
                        data_file_path.clone(),
                    ),
                    (
                        CARDINALITY_PROPERTY.to_string(),
                        deletes.positions.len().to_string(),
                    ),
                ]))
                .build();
            let blob_metadata = puffin_writer.add(blob, CompressionCodec::None).await?;
            blob_coordinates.push((blob_metadata.offset(), blob_metadata.length()));
        }

        // "DVs share the Puffin path and file size but have different offsets" (Java L132-134).
        let puffin_file_size = puffin_writer.close().await?;
        let puffin_path = self.output_file.location().to_string();

        let delete_files = self
            .deletes_by_path
            .iter()
            .zip(blob_coordinates)
            .zip(spec_ids)
            .map(
                |(
                    ((data_file_path, deletes), (content_offset, content_size_in_bytes)),
                    spec_id,
                )| {
                    Self::create_dv_metadata(
                        &puffin_path,
                        puffin_file_size,
                        data_file_path,
                        deletes,
                        content_offset,
                        content_size_in_bytes,
                        spec_id,
                    )
                },
            )
            .collect::<Result<Vec<DataFile>>>()?;

        Ok(DVWriteResult {
            delete_files,
            rewritten_delete_files,
        })
    }

    /// Build the `DeleteFile` metadata for one deletion vector. Mirrors Java
    /// `BaseDVFileWriter.createDV`.
    fn create_dv_metadata(
        puffin_path: &str,
        puffin_file_size: u64,
        data_file_path: &str,
        deletes: &DeletesForDataFile,
        content_offset: u64,
        content_size_in_bytes: u64,
        partition_spec_id: i32,
    ) -> Result<DataFile> {
        let to_signed = |value: u64, what: &str| -> Result<i64> {
            i64::try_from(value).map_err(|_| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Deletion vector {what} {value} does not fit in i64"),
                )
            })
        };

        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::PositionDeletes)
            .file_format(DataFileFormat::Puffin)
            .file_path(puffin_path.to_string())
            .file_size_in_bytes(puffin_file_size)
            .record_count(deletes.positions.len())
            .referenced_data_file(Some(data_file_path.to_string()))
            .content_offset(Some(to_signed(content_offset, "content_offset")?))
            .content_size_in_bytes(Some(to_signed(
                content_size_in_bytes,
                "content_size_in_bytes",
            )?));
        if let Some(partition_key) = &deletes.partition_key {
            builder.partition(partition_key.data().clone());
        }
        builder.partition_spec_id(partition_spec_id);
        builder.build().map_err(|error| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to build deletion vector DeleteFile metadata: {error}"),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::arrow::caching_delete_file_loader::CachingDeleteFileLoader;
    use crate::io::FileIO;
    use crate::metadata_columns::RESERVED_FIELD_ID_DELETE_FILE_PATH;
    use crate::scan::FileScanTaskDeleteFile;
    use crate::spec::{Literal, NestedField, PartitionSpec, PrimitiveType, Schema, Struct, Type};

    fn output_file(file_io: &FileIO, dir: &TempDir, name: &str) -> OutputFile {
        let path = dir.path().join(name);
        file_io
            .new_output(path.to_str().expect("utf-8 temp path"))
            .expect("create output file")
    }

    /// Slice the written Puffin file at the DeleteFile's blob coordinates and decode the
    /// positions — exactly the ranged read the (D1) scan-side loader performs.
    fn decode_blob_at(puffin_bytes: &[u8], delete_file: &DataFile) -> Vec<u64> {
        let offset = usize::try_from(delete_file.content_offset().expect("offset present"))
            .expect("offset fits usize");
        let size = usize::try_from(
            delete_file
                .content_size_in_bytes()
                .expect("content size present"),
        )
        .expect("size fits usize");
        let vector =
            DeleteVector::deserialize_deletion_vector_v1(&puffin_bytes[offset..offset + size])
                .expect("blob at the recorded coordinates must decode");
        vector.iter().collect()
    }

    /// Slice the RAW blob bytes at the DeleteFile's coordinates (for byte-level comparison, e.g. the
    /// no-previous byte-identical floor).
    fn decode_region(puffin_bytes: &[u8], delete_file: &DataFile) -> Vec<u8> {
        let offset = usize::try_from(delete_file.content_offset().expect("offset present"))
            .expect("offset fits usize");
        let size = usize::try_from(
            delete_file
                .content_size_in_bytes()
                .expect("content size present"),
        )
        .expect("size fits usize");
        puffin_bytes[offset..offset + size].to_vec()
    }

    /// Risk pinned: the FULL per-DeleteFile metadata contract of Java `createDV` (L145-159) for
    /// a MULTI-file Puffin — wrong content/format/path/size resurrects rows or breaks the read;
    /// overlapping or swapped blob coordinates silently apply the WRONG vector to a data file.
    #[tokio::test]
    async fn test_dv_writer_multi_file_delete_files_carry_blob_coordinates() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let mut writer =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin")).unpartitioned();

        // Insertion order is deliberately NOT sorted; the output must be (sorted by path).
        writer
            .delete("s3://b/data/b.parquet", 7, None)
            .expect("delete");
        writer
            .delete("s3://b/data/a.parquet", 0, None)
            .expect("delete");
        writer
            .delete("s3://b/data/a.parquet", 3, None)
            .expect("delete");
        writer
            .delete("s3://b/data/a.parquet", (1u64 << 32) + 1, None)
            .expect("delete");
        let delete_files = writer.close().await.expect("close");

        assert_eq!(
            delete_files.len(),
            2,
            "one DeleteFile per referenced data file"
        );
        let a = &delete_files[0];
        let b = &delete_files[1];
        assert_eq!(
            a.referenced_data_file().as_deref(),
            Some("s3://b/data/a.parquet")
        );
        assert_eq!(
            b.referenced_data_file().as_deref(),
            Some("s3://b/data/b.parquet")
        );

        let puffin_path = a.file_path().to_string();
        let puffin_bytes = std::fs::read(&puffin_path).expect("read puffin file");
        for delete_file in [a, b] {
            assert_eq!(delete_file.content_type(), DataContentType::PositionDeletes);
            assert_eq!(delete_file.file_format(), DataFileFormat::Puffin);
            assert_eq!(delete_file.file_path(), puffin_path, "shared Puffin path");
            assert_eq!(
                delete_file.file_size_in_bytes(),
                puffin_bytes.len() as u64,
                "file_size_in_bytes must be the REAL on-disk Puffin size (footer included)"
            );
        }
        assert_eq!(a.record_count(), 3, "record_count == cardinality");
        assert_eq!(b.record_count(), 1);
        assert_ne!(
            a.content_offset(),
            b.content_offset(),
            "blobs must have distinct offsets"
        );

        // The blob at each DeleteFile's coordinates decodes to exactly that file's positions.
        assert_eq!(decode_blob_at(&puffin_bytes, a), vec![
            0,
            3,
            (1u64 << 32) + 1
        ]);
        assert_eq!(decode_blob_at(&puffin_bytes, b), vec![7]);
    }

    /// Risk pinned: "no deletes ⇒ NO Puffin file" (Java L106-109) — writing an empty Puffin
    /// would litter the table location with orphan files.
    #[tokio::test]
    async fn test_dv_writer_no_deletes_writes_no_file() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let out = output_file(&file_io, &temp_dir, "empty.puffin");
        let path = out.location().to_string();

        let writer = DVFileWriter::new(out).unpartitioned();
        let delete_files = writer.close().await.expect("close with no deletes");

        assert!(delete_files.is_empty());
        assert!(
            !std::path::Path::new(&path).exists(),
            "no Puffin file may be created when there are no deletes"
        );
    }

    /// Risk pinned: determinism. Two runs over the same logical deletes, in different insertion
    /// order, must produce an identical blob region and a structurally identical footer. Java gives
    /// no such guarantee, so the sorted blob order is this fork's contract.
    #[tokio::test]
    async fn test_dv_writer_deterministic_output_across_runs() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();

        let mut first =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "first.puffin")).unpartitioned();
        for (path, pos) in [
            ("p/x.parquet", 5u64),
            ("p/y.parquet", 9),
            ("p/x.parquet", 2),
        ] {
            first.delete(path, pos, None).expect("delete");
        }
        let first_files = first.close().await.expect("close first");

        let mut second =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "second.puffin")).unpartitioned();
        for (path, pos) in [
            ("p/y.parquet", 9u64),
            ("p/x.parquet", 2),
            ("p/x.parquet", 5),
        ] {
            second.delete(path, pos, None).expect("delete");
        }
        let second_files = second.close().await.expect("close second");

        // Identical blob coordinates per referenced data file...
        let coordinates = |files: &[DataFile]| -> Vec<(String, Option<i64>, Option<i64>)> {
            files
                .iter()
                .map(|f| {
                    (
                        f.referenced_data_file().expect("referenced path"),
                        f.content_offset(),
                        f.content_size_in_bytes(),
                    )
                })
                .collect()
        };
        assert_eq!(coordinates(&first_files), coordinates(&second_files));

        // ...an identical blob REGION (header magic through the last blob byte)...
        let blob_region_end = usize::try_from(
            first_files
                .iter()
                .map(|f| {
                    f.content_offset().expect("offset") + f.content_size_in_bytes().expect("size")
                })
                .max()
                .expect("at least one blob"),
        )
        .expect("fits usize");
        let first_bytes = std::fs::read(first_files[0].file_path()).expect("read first");
        let second_bytes = std::fs::read(second_files[0].file_path()).expect("read second");
        assert_eq!(
            first_bytes[..blob_region_end],
            second_bytes[..blob_region_end],
            "the same logical deletes must produce a byte-identical blob region"
        );

        // ...and a structurally identical footer (same blobs, offsets, properties).
        let first_footer = crate::puffin::FileMetadata::read(
            &file_io
                .new_input(first_files[0].file_path())
                .expect("input first"),
        )
        .await
        .expect("read first footer");
        let second_footer = crate::puffin::FileMetadata::read(
            &file_io
                .new_input(second_files[0].file_path())
                .expect("input second"),
        )
        .await
        .expect("read second footer");
        assert_eq!(first_footer, second_footer);
    }

    /// Risk pinned: the partition context is captured at the FIRST delete per path (Java
    /// `computeIfAbsent`, L74-79) and lands on the DeleteFile (`withPartition`, L152) — losing
    /// it would let partition pruning skip the DV's data file while keeping its deletes.
    #[tokio::test]
    async fn test_dv_writer_partition_captured_at_first_delete_per_path() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();

        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "category", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .expect("schema"),
        );
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(3)
            .add_partition_field("category", "category", crate::spec::Transform::Identity)
            .expect("partition field")
            .build()
            .expect("spec");
        let partition_a = PartitionKey::new(
            spec.clone(),
            schema.clone(),
            Struct::from_iter([Some(Literal::string("a"))]),
        )
        .expect("PartitionKey::new: valid partition tuple");
        let partition_b = PartitionKey::new(
            spec,
            schema,
            Struct::from_iter([Some(Literal::string("b"))]),
        )
        .expect("PartitionKey::new: valid partition tuple");

        let mut writer =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin")).unpartitioned();
        writer
            .delete("p/x.parquet", 1, Some(&partition_a))
            .expect("delete");
        // A later, DIFFERENT partition for the same path must be ignored (first capture wins).
        writer
            .delete("p/x.parquet", 2, Some(&partition_b))
            .expect("delete");
        let delete_files = writer.close().await.expect("close");

        assert_eq!(delete_files.len(), 1);
        assert_eq!(
            delete_files[0].partition(),
            &Struct::from_iter([Some(Literal::string("a"))]),
            "the FIRST delete's partition must be captured"
        );
        assert_eq!(delete_files[0].partition_spec_id, 3);
    }

    /// Risk pinned: deleting the same position twice must merge. `record_count` counts distinct
    /// positions. Double-counting corrupts the stats every planner trusts.
    #[tokio::test]
    async fn test_dv_writer_duplicate_position_counted_once() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let mut writer =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin")).unpartitioned();

        writer.delete("p/x.parquet", 7, None).expect("delete");
        writer
            .delete("p/x.parquet", 7, None)
            .expect("duplicate delete is a no-op, not an error");
        let delete_files = writer.close().await.expect("close");

        assert_eq!(delete_files.len(), 1);
        assert_eq!(
            delete_files[0].record_count(),
            1,
            "record_count must be the DISTINCT-position cardinality"
        );
        let puffin_bytes = std::fs::read(delete_files[0].file_path()).expect("read puffin");
        assert_eq!(decode_blob_at(&puffin_bytes, &delete_files[0]), vec![7]);
    }

    /// Risk pinned: the position door — Java `RoaringPositionBitmap.validatePosition` rejects
    /// positions above MAX_POSITION at set() time; accepting one here would fail later (or write
    /// a key Java's reader rejects). Pins the boundary EXACTLY: MAX accepted, MAX+1 rejected.
    #[tokio::test]
    async fn test_dv_writer_rejects_position_above_java_max() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let mut writer =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin")).unpartitioned();

        writer
            .delete("p/x.parquet", DV_MAX_POSITION, None)
            .expect("MAX_POSITION itself is legal");
        let error = writer
            .delete("p/x.parquet", DV_MAX_POSITION + 1, None)
            .expect_err("MAX_POSITION + 1 must be rejected");
        assert!(
            error.to_string().contains("positions that are >= 0 and <="),
            "error must name the bound, got: {error}"
        );
    }

    /// Risk pinned: a Puffin file this writer wrote, read back through the real caching loader
    /// from the returned metadata alone, must yield exactly the deleted positions. An offset off
    /// by the 4-byte header magic fails the framing check here.
    #[tokio::test]
    async fn test_dv_writer_round_trips_through_d1_loader() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let mut writer =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin")).unpartitioned();

        let data_file_x = "mem://data/x.parquet";
        let data_file_y = "mem://data/y.parquet";
        for pos in [0u64, 5, (1u64 << 32) + 7] {
            writer.delete(data_file_x, pos, None).expect("delete x");
        }
        for pos in 100u64..200 {
            writer.delete(data_file_y, pos, None).expect("delete y");
        }
        let delete_files = writer.close().await.expect("close");

        let tasks: Vec<FileScanTaskDeleteFile> = delete_files
            .iter()
            .map(|delete_file| FileScanTaskDeleteFile {
                file_path: delete_file.file_path().to_string(),
                file_size_in_bytes: delete_file.file_size_in_bytes(),
                file_type: delete_file.content_type(),
                partition_spec_id: delete_file.partition_spec_id,
                equality_ids: None,
                file_format: delete_file.file_format(),
                referenced_data_file: delete_file.referenced_data_file(),
                content_offset: delete_file.content_offset(),
                content_size_in_bytes: delete_file.content_size_in_bytes(),
                record_count: Some(delete_file.record_count()),
            })
            .collect();

        let loader = CachingDeleteFileLoader::new(file_io.clone(), 4);
        let delete_filter = loader
            .load_deletes(
                &tasks,
                Arc::new(Schema::builder().build().expect("empty schema")),
            )
            .await
            .expect("loader future")
            .expect("the D1 loader must load what the D2 writer wrote");

        let vector_x = delete_filter
            .resolve_delete_vector(&tasks, data_file_x)
            .expect("vector for data file x");
        let positions_x: Vec<u64> = vector_x.iter().collect();
        assert_eq!(positions_x, vec![0, 5, (1u64 << 32) + 7]);

        let vector_y = delete_filter
            .resolve_delete_vector(&tasks, data_file_y)
            .expect("vector for data file y");
        let positions_y: Vec<u64> = vector_y.iter().collect();
        assert_eq!(positions_y, (100u64..200).collect::<Vec<_>>());
    }

    // Previous-deletes MERGE hook (Java `BaseDVFileWriter.loadPreviousDeletes` + `isFileScoped`).

    /// A synthetic DV `DeleteFile` for `referenced_data_file` (file-scoped: carries the referenced
    /// field) — the shape a previous DV's source file has.
    fn synthetic_dv_delete_file(path: &str, referenced_data_file: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .referenced_data_file(Some(referenced_data_file.to_string()))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .build()
            .expect("build synthetic DV delete file")
    }

    /// A synthetic PARTITION-scoped parquet position delete (no `referenced_data_file`, no equal
    /// `_file_path` bounds) — NOT file-scoped, so the merge must NOT rewrite it.
    fn synthetic_partition_scoped_pos_delete(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(2)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .expect("build synthetic partition-scoped pos delete")
    }

    /// Risk pinned: the previous-deletes merge. The existing positions must union into the new
    /// DV, so the blob deletes old and new, and `record_count` is the merged count. A broken
    /// merge writes only the new positions, and the previous deletes resurrect.
    #[tokio::test]
    async fn test_dv_writer_merges_previous_positions_into_new_dv() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let data_file = "mem://data/x.parquet";

        // Previous deletes: positions {1} (loaded, e.g., from a prior DV), sourced from a DV file.
        let previous = PreviousDeletes::new(DeleteVector::new([1u64].into_iter().collect()), vec![
            synthetic_dv_delete_file("mem://data/dv1.puffin", data_file),
        ]);
        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "dv2.puffin"))
            .unpartitioned()
            .with_previous_deletes(HashMap::from([(data_file.to_string(), previous)]));
        // New delete: position {3}.
        writer
            .delete(data_file, 3, None)
            .expect("record new delete");

        let result = writer.close_with_result().await.expect("close with result");
        assert_eq!(result.delete_files.len(), 1, "one merged DV for the file");
        assert_eq!(
            result.delete_files[0].record_count(),
            2,
            "record_count must be the MERGED cardinality {{1,3}} = 2, not just the new {{3}}"
        );

        // The blob at the merged DV's coordinates decodes to the UNION {1, 3}.
        let puffin_bytes = std::fs::read(result.delete_files[0].file_path()).expect("read puffin");
        assert_eq!(
            decode_blob_at(&puffin_bytes, &result.delete_files[0]),
            vec![1, 3],
            "the merged DV must delete the UNION of previous {{1}} and new {{3}}"
        );

        // The file-scoped previous DV is returned as a rewritten (to-be-removed) delete file.
        assert_eq!(
            result.rewritten_delete_files.len(),
            1,
            "the superseded file-scoped DV must be returned for removal"
        );
        assert_eq!(
            result.rewritten_delete_files[0].file_path(),
            "mem://data/dv1.puffin"
        );
    }

    /// Risk pinned: `is_file_scoped` selectivity. A partition-scoped position delete spans many
    /// data files, so the merge must not rewrite it. Rewriting it drops deletes that still apply
    /// to the other data files, and rows resurrect there. The new positions still merge in.
    #[tokio::test]
    async fn test_dv_writer_does_not_rewrite_partition_scoped_previous_delete() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let data_file = "mem://data/x.parquet";

        let previous = PreviousDeletes::new(DeleteVector::new([1u64].into_iter().collect()), vec![
            synthetic_partition_scoped_pos_delete("mem://data/partition-deletes.parquet"),
        ]);
        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "dv.puffin"))
            .unpartitioned()
            .with_previous_deletes(HashMap::from([(data_file.to_string(), previous)]));
        writer
            .delete(data_file, 3, None)
            .expect("record new delete");

        let result = writer.close_with_result().await.expect("close with result");
        // The previous positions STILL merge (positions are unioned regardless of scope).
        assert_eq!(result.delete_files[0].record_count(), 2);
        let puffin_bytes = std::fs::read(result.delete_files[0].file_path()).expect("read puffin");
        assert_eq!(
            decode_blob_at(&puffin_bytes, &result.delete_files[0]),
            vec![1, 3]
        );
        // But the partition-scoped parquet delete is NOT a rewritten file (it may apply elsewhere).
        assert!(
            result.rewritten_delete_files.is_empty(),
            "a partition-scoped (non-file-scoped) previous delete must NOT be rewritten"
        );
    }

    /// Risk pinned: an EQUALITY-delete source file is never file-scoped (Java
    /// `ContentFileUtil.referencedDataFile` returns null for EQUALITY_DELETES) — and a DV does not
    /// supersede equality deletes anyway, so it must NOT appear in rewritten files even if supplied.
    #[tokio::test]
    async fn test_dv_writer_does_not_rewrite_equality_delete_previous_source() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let data_file = "mem://data/x.parquet";

        let eq_delete = DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path("mem://data/eq.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .equality_ids(Some(vec![1]))
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .expect("build eq delete");
        let previous = PreviousDeletes::new(DeleteVector::new([1u64].into_iter().collect()), vec![
            eq_delete,
        ]);
        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "dv.puffin"))
            .unpartitioned()
            .with_previous_deletes(HashMap::from([(data_file.to_string(), previous)]));
        writer
            .delete(data_file, 3, None)
            .expect("record new delete");

        let result = writer.close_with_result().await.expect("close with result");
        assert!(
            result.rewritten_delete_files.is_empty(),
            "an equality delete is never file-scoped — it must not be rewritten"
        );
    }

    /// Risk pinned: previous deletes for a path with NO new positions are IGNORED — Java iterates
    /// `deletesByPath.values()` and calls `loadPreviousDeletes` per entry, so a path never written
    /// to is never visited. Supplying previous deletes for an unwritten path must produce NO blob
    /// for it and NO rewritten file (the engine only loads previous deletes for files it rewrites).
    #[tokio::test]
    async fn test_dv_writer_ignores_previous_deletes_for_unwritten_path() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let written = "mem://data/written.parquet";
        let unwritten = "mem://data/unwritten.parquet";

        let previous_for_unwritten =
            PreviousDeletes::new(DeleteVector::new([9u64].into_iter().collect()), vec![
                synthetic_dv_delete_file("mem://data/old-dv.puffin", unwritten),
            ]);
        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "dv.puffin"))
            .unpartitioned()
            .with_previous_deletes(HashMap::from([(
                unwritten.to_string(),
                previous_for_unwritten,
            )]));
        // Only `written` gets a new position; `unwritten` is never written.
        writer.delete(written, 0, None).expect("record new delete");

        let result = writer.close_with_result().await.expect("close with result");
        assert_eq!(
            result.delete_files.len(),
            1,
            "only the written path produces a DV"
        );
        assert_eq!(
            result.delete_files[0].referenced_data_file().as_deref(),
            Some(written)
        );
        assert!(
            result.rewritten_delete_files.is_empty(),
            "previous deletes for an unwritten path must be ignored (Java visits deletesByPath only)"
        );
    }

    /// Risk pinned: BYTE-IDENTICAL no-previous floor — `with_previous_deletes(empty)` (and not
    /// calling it at all) must produce the EXACT same blob bytes as a fresh-only writer, so the
    /// D2/D4 byte-parity pins stay green. A merge step that touches the bytes even when there is
    /// nothing to merge would silently break Java byte-parity.
    #[tokio::test]
    async fn test_dv_writer_no_previous_deletes_is_byte_identical_to_fresh() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let data_file = "mem://data/x.parquet";

        let mut fresh =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "fresh.puffin")).unpartitioned();
        for pos in [0u64, 3, 7] {
            fresh.delete(data_file, pos, None).expect("fresh delete");
        }
        let fresh_files = fresh.close().await.expect("close fresh");

        let mut empty_prev = DVFileWriter::new(output_file(&file_io, &temp_dir, "empty.puffin"))
            .unpartitioned()
            .with_previous_deletes(HashMap::new());
        for pos in [0u64, 3, 7] {
            empty_prev.delete(data_file, pos, None).expect("delete");
        }
        let empty_prev_files = empty_prev.close().await.expect("close empty-prev");

        let fresh_blob = {
            let bytes = std::fs::read(fresh_files[0].file_path()).expect("read fresh");
            decode_region(&bytes, &fresh_files[0])
        };
        let empty_blob = {
            let bytes = std::fs::read(empty_prev_files[0].file_path()).expect("read empty");
            decode_region(&bytes, &empty_prev_files[0])
        };
        assert_eq!(
            fresh_blob, empty_blob,
            "an empty previous-deletes map must leave the blob bytes identical to fresh-only"
        );
    }

    /// Risk pinned: `is_file_scoped` predicate — the three Java `ContentFileUtil.referencedDataFile`
    /// branches (equality → false; explicit referenced field → true; equal `_file_path` bounds →
    /// true; absent/unequal bounds → false). A drift here misclassifies what gets rewritten,
    /// either dropping a still-applying delete (resurrection) or failing to remove a superseded one.
    #[test]
    fn test_is_file_scoped_mirrors_java_referenced_data_file() {
        use crate::spec::Datum;

        // (1) DV with explicit referenced_data_file → file-scoped.
        assert!(is_file_scoped(&synthetic_dv_delete_file(
            "dv.puffin",
            "data/x.parquet"
        )));

        // (2) equality delete → never file-scoped.
        let eq = DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path("eq.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(1)
            .equality_ids(Some(vec![1]))
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .expect("eq delete");
        assert!(!is_file_scoped(&eq));

        // (3) partition-scoped parquet pos delete, no bounds → NOT file-scoped.
        assert!(!is_file_scoped(&synthetic_partition_scoped_pos_delete(
            "part.parquet"
        )));

        // (4) position delete whose `_file_path` lower==upper bound pins ONE data file → file-scoped
        //     (the Java fallback when `referenced_data_file` is unset).
        let path_bound = Datum::string("data/x.parquet");
        let path_scoped = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("scoped.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .lower_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                path_bound.clone(),
            )]))
            .upper_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                path_bound,
            )]))
            .build()
            .expect("path-scoped pos delete");
        assert!(
            is_file_scoped(&path_scoped),
            "equal _file_path bounds pin one data file → file-scoped (Java fallback)"
        );

        // (5) position delete whose `_file_path` bounds DIFFER (spans many data files) → NOT
        //     file-scoped.
        let unequal = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("spanning.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .lower_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("data/a.parquet"),
            )]))
            .upper_bounds(HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_PATH,
                Datum::string("data/z.parquet"),
            )]))
            .build()
            .expect("spanning pos delete");
        assert!(
            !is_file_scoped(&unequal),
            "unequal _file_path bounds span many data files → NOT file-scoped"
        );
    }

    /// Build an unpartitioned spec carrying `spec_id`, and a one-field partitioned spec.
    fn specs_for_stamping(spec_id: i32) -> (PartitionSpec, PartitionSpec, Arc<Schema>) {
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "category", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .expect("schema"),
        );
        let unpartitioned = PartitionSpec::builder(schema.clone())
            .with_spec_id(spec_id)
            .build()
            .expect("unpartitioned spec");
        let partitioned = PartitionSpec::builder(schema.clone())
            .with_spec_id(spec_id)
            .add_partition_field("category", "category", crate::spec::Transform::Identity)
            .expect("partition field")
            .build()
            .expect("partitioned spec");
        (unpartitioned, partitioned, schema)
    }

    /// A DV written with no `PartitionKey` claims the CONFIGURED spec id, not the fabricated
    /// `DEFAULT_PARTITION_SPEC_ID` (0). Java stamps `spec.specId()` unconditionally
    /// (`FileMetadata.deleteFileBuilder(spec)`), so a keyless DV must not claim spec 0.
    #[tokio::test]
    async fn dv_with_partition_spec_stamps_configured_spec_without_a_key() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let (unpartitioned, _, _) = specs_for_stamping(7);

        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin"))
            .with_partition_spec(unpartitioned);
        writer.delete("p/x.parquet", 1, None).expect("delete");
        let delete_files = writer.close().await.expect("close");

        assert_eq!(delete_files.len(), 1, "one DV per referenced data file");
        assert_eq!(
            delete_files[0].partition_spec_id, 7,
            "a keyless DV must claim the configured spec, not DEFAULT_PARTITION_SPEC_ID"
        );
        assert_eq!(
            delete_files[0].partition(),
            &Struct::empty(),
            "an unpartitioned spec leaves the tuple empty"
        );
    }

    /// A `PartitionKey` on the `delete` call wins over a configured spec. The key carries the tuple
    /// AND the spec that tuple came from, so it is authoritative — a delete file claims the spec of
    /// the DATA FILES it deletes from, which is not always the table's current spec.
    #[tokio::test]
    async fn dv_partition_key_wins_over_configured_partition_spec() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let (configured, _, schema) = specs_for_stamping(7);
        let keyed_spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(9)
            .add_partition_field("category", "category", crate::spec::Transform::Identity)
            .expect("partition field")
            .build()
            .expect("keyed spec");
        let key = PartitionKey::new(
            keyed_spec,
            schema,
            Struct::from_iter([Some(Literal::string("a"))]),
        )
        .expect("PartitionKey::new: valid partition tuple");

        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin"))
            .with_partition_spec(configured);
        writer.delete("p/x.parquet", 1, Some(&key)).expect("delete");
        let delete_files = writer.close().await.expect("close");

        assert_eq!(
            delete_files[0].partition_spec_id, 9,
            "the key's spec wins over the configured spec"
        );
        assert_eq!(
            delete_files[0].partition(),
            &Struct::from_iter([Some(Literal::string("a"))]),
            "the key's tuple is stamped"
        );
    }

    /// A partitioned configured spec with no `PartitionKey` is rejected at close. The file would
    /// carry an EMPTY tuple while claiming a spec whose partition type has arity 1, which
    /// `SnapshotProducer::validate_partition_value` rejects at commit anyway.
    #[tokio::test]
    async fn dv_partitioned_spec_without_a_key_is_rejected() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let (_, partitioned, _) = specs_for_stamping(7);

        let puffin_path = temp_dir.path().join("deletes.puffin");
        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin"))
            .with_partition_spec(partitioned);
        writer.delete("p/x.parquet", 1, None).expect("delete");
        let error = writer
            .close()
            .await
            .expect_err("a partitioned spec with no key must not produce a DV");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("partition field(s)"),
            "the error names the arity mismatch, got: {}",
            error.message()
        );
        assert!(
            !puffin_path.exists(),
            "the rejection must fire before any byte reaches storage: an unreferenced Puffin \
             file is an orphan no commit will ever clean up"
        );
    }

    /// Each DV carries ITS OWN referenced file's spec id. `close_with_result` zips three iterators
    /// derived from the same `BTreeMap` — the resolved spec ids, the blob coordinates, and the
    /// entries. A misalignment stamps path A's DV with path B's spec, which is silently wrong; a
    /// length mismatch truncates the zip and DROPS a delete file, so rows resurrect.
    #[tokio::test]
    async fn dv_spec_ids_follow_their_own_referenced_file() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let (configured, _, schema) = specs_for_stamping(7);
        let keyed_spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(9)
            .add_partition_field("category", "category", crate::spec::Transform::Identity)
            .expect("partition field")
            .build()
            .expect("keyed spec");
        let key = PartitionKey::new(
            keyed_spec,
            schema,
            Struct::from_iter([Some(Literal::string("a"))]),
        )
        .expect("PartitionKey::new: valid partition tuple");

        // Sorted path order puts "p/a.parquet" first, so the two stampings are distinguishable.
        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin"))
            .with_partition_spec(configured);
        writer.delete("p/a.parquet", 1, Some(&key)).expect("delete");
        writer.delete("p/b.parquet", 2, None).expect("delete");
        let delete_files = writer.close().await.expect("close");

        assert_eq!(delete_files.len(), 2, "one DV per referenced data file");
        for delete_file in &delete_files {
            let referenced = delete_file
                .referenced_data_file()
                .expect("a DV always names its referenced data file");
            let expected = match referenced.as_str() {
                "p/a.parquet" => 9,
                "p/b.parquet" => 7,
                other => panic!("unexpected referenced data file: {other}"),
            };
            assert_eq!(
                delete_file.partition_spec_id, expected,
                "{referenced} must carry its own spec id, not another entry's"
            );
        }
    }

    /// The rejection keys on partition-field ARITY, not [`PartitionSpec::is_unpartitioned`]. An
    /// ALL-VOID spec is the discriminator: `is_unpartitioned()` is true for it, yet its partition
    /// type still has arity 1, so a file under it still needs a tuple of nulls. Keying on
    /// `is_unpartitioned` would wave this through into a commit-time arity failure instead.
    #[tokio::test]
    async fn dv_all_void_spec_without_a_key_is_rejected() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let (_, _, schema) = specs_for_stamping(7);
        let all_void = PartitionSpec::builder(schema)
            .with_spec_id(7)
            .add_partition_field("category", "category", crate::spec::Transform::Void)
            .expect("void partition field")
            .build()
            .expect("all-void spec");
        assert!(
            all_void.is_unpartitioned(),
            "fixture precondition: an all-void spec reports itself unpartitioned"
        );
        assert_eq!(
            all_void.fields().len(),
            1,
            "fixture precondition: it nonetheless has one partition field"
        );

        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin"))
            .with_partition_spec(all_void);
        writer.delete("p/x.parquet", 1, None).expect("delete");
        let error = writer
            .close()
            .await
            .expect_err("an all-void spec with no key must be rejected on arity");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.message().contains("partition field(s)"),
            "the error names the arity mismatch, got: {}",
            error.message()
        );
    }

    // ---- `DVWriteResult::referenced_data_files` / `references_data_files` (F-13 U3b) ------------
    //
    // The accessors are derived, so what is pinned is the projection: cardinality, that it reads
    // `delete_files` not `rewritten_delete_files`, and `referenced_data_file` not the DV's own path.

    /// Many DVs: the derived set is exactly the referenced data files, one per written blob.
    #[tokio::test]
    async fn dv_result_referenced_data_files_names_every_referenced_file() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let mut writer =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin")).unpartitioned();
        writer.delete("mem://data/a.parquet", 0, None).expect("a");
        writer.delete("mem://data/b.parquet", 4, None).expect("b");
        writer
            .delete("mem://data/b.parquet", 9, None)
            .expect("b again");
        writer.delete("mem://data/c.parquet", 1, None).expect("c");

        let result = writer.close_with_result().await.expect("close with result");
        assert_eq!(result.delete_files.len(), 3, "one DV per referenced file");
        assert_eq!(
            result.referenced_data_files(),
            HashSet::from([
                "mem://data/a.parquet".to_string(),
                "mem://data/b.parquet".to_string(),
                "mem://data/c.parquet".to_string(),
            ]),
            "every referenced file appears exactly once, regardless of how many positions it had"
        );
        assert!(result.references_data_files());
    }

    /// ZERO DVs: Java's `close` early-returns `CharSequenceSet.empty()`, so `referencesDataFiles()`
    /// is false.
    #[tokio::test]
    async fn dv_result_with_no_deletes_references_nothing() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let writer =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin")).unpartitioned();

        let result = writer.close_with_result().await.expect("close with result");
        assert!(result.delete_files.is_empty(), "fixture precondition");
        assert!(
            result.referenced_data_files().is_empty(),
            "no DV written means no data file referenced"
        );
        assert!(
            !result.references_data_files(),
            "`references_data_files` must be false on the empty result, not vacuously true"
        );
    }

    /// The SOURCE-MEMBER cell. Both members are non-empty and name DISJOINT data files, so reading
    /// `rewritten_delete_files` produces the wrong set rather than the same one.
    #[tokio::test]
    async fn dv_result_referenced_data_files_reads_the_dvs_not_the_rewritten_files() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let written = "mem://data/written.parquet";
        let superseded_target = "mem://data/superseded-target.parquet";

        // A file-scoped previous DV over `written`, whose own `referenced_data_file` names a
        // DIFFERENT path — so the two members cannot be confused for each other.
        let previous = PreviousDeletes::new(DeleteVector::new([2u64].into_iter().collect()), vec![
            synthetic_dv_delete_file("mem://data/old-dv.puffin", superseded_target),
        ]);
        let mut writer = DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin"))
            .unpartitioned()
            .with_previous_deletes(HashMap::from([(written.to_string(), previous)]));
        writer.delete(written, 7, None).expect("record new delete");

        let result = writer.close_with_result().await.expect("close with result");
        assert_eq!(
            result.rewritten_delete_files.len(),
            1,
            "fixture precondition: the previous file-scoped DV is superseded"
        );
        assert_eq!(
            result.referenced_data_files(),
            HashSet::from([written.to_string()]),
            "the set names what the NEW DVs reference; the superseded file's own target is absent"
        );
    }

    /// The FIELD cell: the set is the `referenced_data_file`, never the DV's own `file_path`. The
    /// Puffin path is shared by every blob, so reading `file_path` collapses the set.
    #[tokio::test]
    async fn dv_result_referenced_data_files_is_not_the_puffin_path() {
        let temp_dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let mut writer =
            DVFileWriter::new(output_file(&file_io, &temp_dir, "deletes.puffin")).unpartitioned();
        writer
            .delete("mem://data/only.parquet", 0, None)
            .expect("d");

        let result = writer.close_with_result().await.expect("close with result");
        let puffin_path = result.delete_files[0].file_path().to_string();
        assert!(
            puffin_path.ends_with("deletes.puffin"),
            "fixture precondition: the DV lives in the Puffin file, got {puffin_path}"
        );
        assert_eq!(
            result.referenced_data_files(),
            HashSet::from(["mem://data/only.parquet".to_string()]),
            "the referenced DATA file, not the Puffin container the DV was written into"
        );
    }
}
