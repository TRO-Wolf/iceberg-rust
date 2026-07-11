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

use std::collections::{HashMap, HashSet};
use std::ops::Not;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, StringArray, StructArray};
use futures::{StreamExt, TryStreamExt};
use tokio::sync::oneshot::{Receiver, channel};

use super::delete_filter::{DeleteFilter, PosDelLoadAction};
use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::arrow::equality_delete_set::EqDeleteKeySet;
use crate::arrow::{arrow_primitive_to_literal, arrow_schema_to_schema};
use crate::delete_vector::DeleteVector;
use crate::expr::Predicate::AlwaysTrue;
use crate::expr::{Predicate, Reference};
use crate::io::FileIO;
use crate::scan::{ArrowRecordBatchStream, FileScanTaskDeleteFile};
use crate::spec::{
    DataContentType, DataFileFormat, Datum, ListType, MapType, NestedField, NestedFieldRef,
    PartnerAccessor, PrimitiveType, Schema, SchemaRef, SchemaWithPartnerVisitor, StructType, Type,
    visit_schema_with_partner,
};
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Debug)]
pub(crate) struct CachingDeleteFileLoader {
    basic_delete_file_loader: BasicDeleteFileLoader,
    concurrency_limit_data_files: usize,
    /// Shared filter state to allow caching loaded deletes across multiple
    /// calls to `load_deletes` (e.g., across multiple file scan tasks).
    delete_filter: DeleteFilter,
}

// Intermediate context during processing of a delete file task.
enum DeleteFileContext {
    ExistingEqDel,
    ExistingPosDel,
    PosDels {
        file_path: String,
        stream: ArrowRecordBatchStream,
    },
    /// A freshly loaded + decoded Puffin deletion vector. `cache_key` is the loader's
    /// dedup/notify key (`{puffin path}@{blob offset}` — one Puffin file holds many DV blobs, so
    /// the bare file path would wrongly mark every later blob "already loaded");
    /// `referenced_data_file` is the data file the vector applies to and the key it is installed
    /// under in the [`DeleteFilter`].
    FreshDeletionVector {
        cache_key: String,
        referenced_data_file: String,
        delete_vector: DeleteVector,
    },
    FreshEqDel {
        batch_stream: ArrowRecordBatchStream,
        equality_ids: HashSet<i32>,
        sender: tokio::sync::oneshot::Sender<(Predicate, Option<EqDeleteKeySet>)>,
    },
}

// Final result of the processing of a delete file task before
// results are fully merged into the DeleteFileManager's state
enum ParsedDeleteFileContext {
    DelVecs {
        file_path: String,
        results: HashMap<String, DeleteVector>,
    },
    EqDel,
    ExistingPosDel,
}

#[allow(unused_variables)]
impl CachingDeleteFileLoader {
    pub(crate) fn new(file_io: FileIO, concurrency_limit_data_files: usize) -> Self {
        CachingDeleteFileLoader {
            basic_delete_file_loader: BasicDeleteFileLoader::new(file_io),
            concurrency_limit_data_files,
            delete_filter: DeleteFilter::default(),
        }
    }

    /// Initiates loading of all deletes for all the specified tasks
    ///
    /// Returned future completes once all positional deletes and delete vectors
    /// have loaded. EQ deletes are not waited for in this method but the returned
    /// DeleteFilter will await their loading when queried for them.
    ///
    ///  * Create a single stream of all delete file tasks irrespective of type,
    ///    so that we can respect the combined concurrency limit
    ///  * We then process each in two phases: load and parse.
    ///  * for positional deletes the load phase instantiates an ArrowRecordBatchStream to
    ///    stream the file contents out
    ///  * for eq deletes, we first check if the EQ delete is already loaded or being loaded by
    ///    another concurrently processing data file scan task. If it is, we skip it.
    ///    If not, the DeleteFilter is updated to contain a notifier to prevent other data file
    ///    tasks from starting to load the same equality delete file. We spawn a task to load
    ///    the EQ delete's record batch stream, convert it to a predicate, update the delete filter,
    ///    and notify any task that was waiting for it.
    ///  * a positional delete in PUFFIN format is a DELETION VECTOR: the load phase does one
    ///    ranged read of the `deletion-vector-v1` blob (at the manifest's content_offset /
    ///    content_size_in_bytes) and decodes it; the same notify machinery dedups concurrent
    ///    loads of one blob under the key `{path}@{offset}`.
    ///  * The parse phase parses each record batch stream according to its associated data type.
    ///    The result of this is a map of data file paths to delete vectors for the positional
    ///    delete tasks (a decoded deletion vector contributes a single entry keyed by its
    ///    referenced data file). For equality delete file tasks, this results in an unbound
    ///    Predicate.
    ///  * The unbound Predicates resulting from equality deletes are sent to their associated oneshot
    ///    channel to store them in the right place in the delete file managers state.
    ///  * The results of all of these futures are awaited on in parallel with the specified
    ///    level of concurrency and collected into a vec. We then combine all the delete
    ///    vector maps that resulted from any positional delete or delete vector files into a
    ///    single map and persist it in the state.
    ///
    ///
    ///  Conceptually, the data flow is like this:
    /// ```none
    ///                                          FileScanTaskDeleteFile
    ///                                                     |
    ///                                             Skip Started EQ Deletes
    ///                                                     |
    ///                                                     |
    ///                                       [load recordbatch stream / puffin]
    ///                                             DeleteFileContext
    ///                                                     |
    ///                                                     |
    ///                       +-----------------------------+--------------------------+
    ///                     Pos Del                      Del Vec                     EQ Del
    ///                       |                             |                          |
    ///              [parse pos del stream]         [parse del vec puffin]       [parse eq del]
    ///          HashMap<String, RoaringTreeMap> HashMap<String, RoaringTreeMap>   (Predicate, Sender)
    ///                       |                             |                          |
    ///                       |                             |                 [persist to state]
    ///                       |                             |                          ()
    ///                       |                             |                          |
    ///                       +-----------------------------+--------------------------+
    ///                                                     |
    ///                                             [buffer unordered]
    ///                                                     |
    ///                                            [combine del vectors]
    ///                                        HashMap<String, RoaringTreeMap>
    ///                                                     |
    ///                                        [persist del vectors to state]
    ///                                                    ()
    ///                                                    |
    ///                                                    |
    ///                                                 [join!]
    /// ```
    pub(crate) fn load_deletes(
        &self,
        delete_file_entries: &[FileScanTaskDeleteFile],
        schema: SchemaRef,
    ) -> Receiver<Result<DeleteFilter>> {
        let (tx, rx) = channel();

        // A data file must carry AT MOST ONE deletion vector. Java rejects the duplicate at
        // index-build time (`DeleteFileIndex.Builder.add`, DeleteFileIndex.java L528-535:
        // "Can't index multiple DVs for %s"); the Rust index lookup is infallible by signature,
        // so the same invalid state is rejected fail-loud HERE, at the load door, before any
        // vector is installed (silently unioning two DVs would over-delete; keeping one would
        // resurrect rows).
        let mut deletion_vector_targets = HashSet::new();
        for entry in delete_file_entries {
            if entry.file_type == DataContentType::PositionDeletes
                && entry.file_format == DataFileFormat::Puffin
                && let Some(referenced_data_file) = &entry.referenced_data_file
                && !deletion_vector_targets.insert(referenced_data_file.clone())
            {
                let _ = tx.send(Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Found multiple deletion vectors for data file '{referenced_data_file}'"
                    ),
                )));
                return rx;
            }
        }

        let stream_items = delete_file_entries
            .iter()
            .map(|t| {
                (
                    t.clone(),
                    self.basic_delete_file_loader.clone(),
                    self.delete_filter.clone(),
                    schema.clone(),
                )
            })
            .collect::<Vec<_>>();
        let task_stream = futures::stream::iter(stream_items);

        let del_filter = self.delete_filter.clone();
        let concurrency_limit_data_files = self.concurrency_limit_data_files;
        let basic_delete_file_loader = self.basic_delete_file_loader.clone();
        crate::runtime::spawn(async move {
            let result = async move {
                let mut del_filter = del_filter;
                let basic_delete_file_loader = basic_delete_file_loader.clone();

                let mut results_stream = task_stream
                    .map(move |(task, file_io, del_filter, schema)| {
                        let basic_delete_file_loader = basic_delete_file_loader.clone();
                        async move {
                            Self::load_file_for_task(
                                &task,
                                basic_delete_file_loader.clone(),
                                del_filter,
                                schema,
                            )
                            .await
                        }
                    })
                    .map(move |ctx| {
                        Ok(async { Self::parse_file_content_for_task(ctx.await?).await })
                    })
                    .try_buffer_unordered(concurrency_limit_data_files);

                while let Some(item) = results_stream.next().await {
                    let item = item?;
                    if let ParsedDeleteFileContext::DelVecs { file_path, results } = item {
                        for (data_file_path, delete_vector) in results.into_iter() {
                            del_filter.upsert_delete_vector(data_file_path, delete_vector);
                        }
                        // Mark the positional delete file as fully loaded so waiters can proceed
                        del_filter.finish_pos_del_load(&file_path);
                    }
                }

                Ok(del_filter)
            }
            .await;

            let _ = tx.send(result);
        });

        rx
    }

    async fn load_file_for_task(
        task: &FileScanTaskDeleteFile,
        basic_delete_file_loader: BasicDeleteFileLoader,
        del_filter: DeleteFilter,
        schema: SchemaRef,
    ) -> Result<DeleteFileContext> {
        match task.file_type {
            // A position delete in PUFFIN format is a DELETION VECTOR (Java
            // `ContentFileUtil.isDV`: `format() == FileFormat.PUFFIN`) — it must be routed to
            // the DV blob loader; handing it to the parquet reader misparses it.
            DataContentType::PositionDeletes if task.file_format == DataFileFormat::Puffin => {
                Self::load_deletion_vector_for_task(task, &basic_delete_file_loader, &del_filter)
                    .await
            }

            DataContentType::PositionDeletes => {
                match del_filter.try_start_pos_del_load(&task.file_path) {
                    PosDelLoadAction::AlreadyLoaded => Ok(DeleteFileContext::ExistingPosDel),
                    PosDelLoadAction::WaitFor(notify) => {
                        // Positional deletes are accessed synchronously by ArrowReader.
                        // We must wait here to ensure the data is ready before returning,
                        // otherwise ArrowReader might get an empty/partial result.
                        notify.notified().await;
                        Ok(DeleteFileContext::ExistingPosDel)
                    }
                    PosDelLoadAction::Load => Ok(DeleteFileContext::PosDels {
                        file_path: task.file_path.clone(),
                        stream: basic_delete_file_loader
                            .parquet_to_batch_stream(&task.file_path, task.file_size_in_bytes)
                            .await?,
                    }),
                }
            }

            DataContentType::EqualityDeletes => {
                let Some(notify) = del_filter.try_start_eq_del_load(&task.file_path) else {
                    return Ok(DeleteFileContext::ExistingEqDel);
                };

                let (sender, receiver) = channel();
                del_filter.insert_equality_delete(&task.file_path, receiver);

                // Per the Iceberg spec, evolve schema for equality deletes but only for the
                // equality_ids columns, not all table columns.
                let equality_ids_vec = task.equality_ids.clone().unwrap();
                let evolved_stream = BasicDeleteFileLoader::evolve_schema(
                    basic_delete_file_loader
                        .parquet_to_batch_stream(&task.file_path, task.file_size_in_bytes)
                        .await?,
                    schema,
                    &equality_ids_vec,
                )
                .await?;

                Ok(DeleteFileContext::FreshEqDel {
                    batch_stream: evolved_stream,
                    sender,
                    equality_ids: HashSet::from_iter(equality_ids_vec),
                })
            }

            DataContentType::Data => Err(Error::new(
                ErrorKind::Unexpected,
                "tasks with files of type Data not expected here",
            )),
        }
    }

    /// Loads + decodes one deletion vector blob, deduplicating concurrent loads of the SAME blob
    /// through the positional-delete notify machinery under the key `{puffin path}@{offset}`.
    ///
    /// Mirrors Java's scan-time DV read (`BaseDeleteLoader.readDV`, BaseDeleteLoader.java
    /// L171-183): ONE ranged read at `content_offset` of `content_size_in_bytes` bytes — not a
    /// Puffin footer round-trip (the footer route costs 3+ requests; the manifest already names
    /// the exact blob range, see the doc comment at L143-147) — then the `deletion-vector-v1`
    /// deserialization. Metadata validations mirror `BaseDeleteLoader.validateDV` (L266-283) and
    /// the cardinality check mirrors `BitmapPositionDeleteIndex.deserializeBitmap` (L203-209).
    async fn load_deletion_vector_for_task(
        task: &FileScanTaskDeleteFile,
        basic_delete_file_loader: &BasicDeleteFileLoader,
        del_filter: &DeleteFilter,
    ) -> Result<DeleteFileContext> {
        let (referenced_data_file, content_offset, content_size_in_bytes) =
            Self::validate_deletion_vector_task(task)?;

        let cache_key = format!("{}@{content_offset}", task.file_path);
        match del_filter.try_start_pos_del_load(&cache_key) {
            PosDelLoadAction::AlreadyLoaded => Ok(DeleteFileContext::ExistingPosDel),
            PosDelLoadAction::WaitFor(notify) => {
                // Like parquet positional deletes, the decoded vector must be fully available
                // before ArrowReader proceeds (retrieval is synchronous).
                notify.notified().await;
                Ok(DeleteFileContext::ExistingPosDel)
            }
            PosDelLoadAction::Load => {
                let blob = basic_delete_file_loader
                    .read_bytes_range(&task.file_path, content_offset, content_size_in_bytes)
                    .await?;
                let delete_vector = DeleteVector::deserialize_deletion_vector_v1(&blob)?;

                // Java validates the decoded cardinality against the DeleteFile's recordCount
                // (`deserializeBitmap`: "Invalid cardinality: %s, expected %s") — a mismatch
                // means the manifest and the blob disagree about how many rows are deleted.
                if let Some(expected_cardinality) = task.record_count
                    && delete_vector.len() != expected_cardinality
                {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Invalid deletion vector cardinality for '{}': decoded {} positions, \
                             manifest record_count expects {expected_cardinality}",
                            task.file_path,
                            delete_vector.len(),
                        ),
                    ));
                }

                Ok(DeleteFileContext::FreshDeletionVector {
                    cache_key,
                    referenced_data_file,
                    delete_vector,
                })
            }
        }
    }

    /// Validates the deletion-vector metadata on a delete-file task, mirroring Java
    /// `BaseDeleteLoader.validateDV` (offset non-null, length non-null, length <= 2GB) plus the
    /// keying prerequisite (`referenced_data_file` present — the Puffin spec makes
    /// `referenced-data-file` mandatory for `deletion-vector-v1`, and the loaded vector is keyed
    /// by it). Returns `(referenced_data_file, content_offset, content_size_in_bytes)` with the
    /// untrusted i64 ranges checked into u64.
    fn validate_deletion_vector_task(task: &FileScanTaskDeleteFile) -> Result<(String, u64, u64)> {
        let referenced_data_file = task.referenced_data_file.clone().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid deletion vector '{}': missing referenced_data_file",
                    task.file_path
                ),
            )
        })?;

        let content_offset = task
            .content_offset
            .and_then(|offset| u64::try_from(offset).ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Invalid deletion vector '{}': content_offset must be a non-negative \
                         integer, got {:?}",
                        task.file_path, task.content_offset
                    ),
                )
            })?;

        // Java: "Can't read DV larger than 2GB" (contentSizeInBytes <= Integer.MAX_VALUE);
        // negative sizes are equally invalid.
        let content_size_in_bytes = task
            .content_size_in_bytes
            .filter(|size| (0..=i64::from(i32::MAX)).contains(size))
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Invalid deletion vector '{}': content_size_in_bytes must be between 0 \
                         and {} (2GB), got {:?}",
                        task.file_path,
                        i32::MAX,
                        task.content_size_in_bytes
                    ),
                )
            })?;

        Ok((referenced_data_file, content_offset, content_size_in_bytes))
    }

    async fn parse_file_content_for_task(
        ctx: DeleteFileContext,
    ) -> Result<ParsedDeleteFileContext> {
        match ctx {
            DeleteFileContext::ExistingEqDel => Ok(ParsedDeleteFileContext::EqDel),
            DeleteFileContext::ExistingPosDel => Ok(ParsedDeleteFileContext::ExistingPosDel),
            DeleteFileContext::PosDels { file_path, stream } => {
                let del_vecs =
                    Self::parse_positional_deletes_record_batch_stream(&file_path, stream).await?;
                Ok(ParsedDeleteFileContext::DelVecs {
                    file_path,
                    results: del_vecs,
                })
            }
            // The decoded deletion vector is installed under the DATA FILE it references (the
            // DV's referenced_data_file) — NOT under the Puffin file's own path: the DeleteFilter
            // hands a scan task its delete vector by data-file-path lookup, so keying by the
            // Puffin path would orphan the vector and silently resurrect every deleted row.
            // `file_path` carries the loader's `{path}@{offset}` cache key so the notify
            // machinery marks the right blob loaded.
            DeleteFileContext::FreshDeletionVector {
                cache_key,
                referenced_data_file,
                delete_vector,
            } => Ok(ParsedDeleteFileContext::DelVecs {
                file_path: cache_key,
                results: HashMap::from([(referenced_data_file, delete_vector)]),
            }),
            DeleteFileContext::FreshEqDel {
                sender,
                batch_stream,
                equality_ids,
            } => {
                let predicate_and_set =
                    Self::parse_equality_deletes_with_keyset(batch_stream, equality_ids).await?;

                sender
                    .send(predicate_and_set)
                    .map_err(|_| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "Could not send eq delete predicate to state",
                        )
                    })
                    .map(|_| ParsedDeleteFileContext::EqDel)
            }
        }
    }

    /// Checked conversion of one position-delete row's `pos` value (untrusted i64 from the
    /// delete file) into a bitmap position.
    ///
    /// A corrupt delete file can carry a negative position; the old `pos as u64` wrapped it to
    /// a huge position that matches no row, so the delete silently failed OPEN (deleted rows
    /// resurrect) — the highest-severity silent-corruption class. Java fails loud on the same
    /// input: `BitmapPositionDeleteIndex.delete(long)` (BitmapPositionDeleteIndex.java L66-68)
    /// → `RoaringPositionBitmap.set(long)` (L73-74) → `validatePosition`
    /// (RoaringPositionBitmap.java L311-316), which throws `IllegalArgumentException`
    /// ("Bitmap supports positions that are >= 0 and <= %s: %s"). Parity nuance: Java's upper
    /// bound `MAX_POSITION` (0x7FFF_FFFE_8000_0000, a roaring 32-bit key-space limit below
    /// `i64::MAX`) is NOT mirrored — Rust's `RoaringTreemap` supports the full u64 position
    /// range, so only the negative bound applies here.
    ///
    /// `delete_file_path` is the position-delete file being parsed; `data_file_path` is the
    /// data file the row points at — both are named in the error so the corrupt file is
    /// identifiable from logs alone.
    fn checked_delete_position(
        delete_file_path: &str,
        data_file_path: &str,
        pos: i64,
    ) -> Result<u64> {
        u64::try_from(pos).map_err(|_| {
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Invalid position delete file '{delete_file_path}': negative position \
                     {pos} for data file '{data_file_path}'"
                ),
            )
        })
    }

    /// Parses a record batch stream coming from the positional delete file at
    /// `delete_file_path` (named in errors so corrupt input is identifiable).
    ///
    /// Returns a map of data file path to a delete vector
    async fn parse_positional_deletes_record_batch_stream(
        delete_file_path: &str,
        mut stream: ArrowRecordBatchStream,
    ) -> Result<HashMap<String, DeleteVector>> {
        let mut result: HashMap<String, DeleteVector> = HashMap::default();

        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let schema = batch.schema();
            let columns = batch.columns();

            let Some(file_paths) = columns[0].as_any().downcast_ref::<StringArray>() else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Could not downcast file paths array to StringArray",
                ));
            };
            let Some(positions) = columns[1].as_any().downcast_ref::<Int64Array>() else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Could not downcast positions array to Int64Array",
                ));
            };

            // Position-delete files are sorted by (path, pos), so equal paths arrive in CONTIGUOUS
            // runs. Cache the delete vector for the LAST-SEEN path and only re-resolve the map entry
            // (allocating an owned `String` key) when the path changes — instead of allocating a
            // `String` and hashing the map for EVERY row. The resulting map is identical to the
            // per-row form: same keys, same positions inserted in the same order (a sorted file has
            // one run per path; an unsorted file still lands every position in the right entry, it
            // just re-resolves on each path change). `current` holds the path string and its vector;
            // we splice the vector back into the map on each change and at end-of-batch.
            let mut current: Option<(&str, DeleteVector)> = None;
            for (file_path, pos) in file_paths.iter().zip(positions.iter()) {
                // Both columns are REQUIRED by the spec (Java `MetadataColumns.DELETE_FILE_POS`,
                // MetadataColumns.java L70-74, is `NestedField.required`; Java's read path NPEs
                // unboxing a null — `Deletes.toPositionIndexes`, Deletes.java L146). A null in
                // either column is corrupt input: fail closed with a typed error naming the
                // delete file, never panic and never skip the row.
                let Some(file_path) = file_path else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Invalid position delete file '{delete_file_path}': null file_path \
                             value (the file_path column is required)"
                        ),
                    ));
                };
                let Some(pos) = pos else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Invalid position delete file '{delete_file_path}': null position \
                             value for data file '{file_path}' (the pos column is required)"
                        ),
                    ));
                };

                match &mut current {
                    Some((path, vector)) if *path == file_path => {
                        vector.insert(Self::checked_delete_position(
                            delete_file_path,
                            file_path,
                            pos,
                        )?);
                    }
                    _ => {
                        // Flush the previous run's vector back into the map (merging if the path
                        // recurs in a later, non-contiguous run), then start the new path's run from
                        // whatever positions the map already holds for it.
                        if let Some((path, vector)) = current.take() {
                            *result.entry(path.to_string()).or_default() |= vector;
                        }
                        let mut vector =
                            std::mem::take(result.entry(file_path.to_string()).or_default());
                        vector.insert(Self::checked_delete_position(
                            delete_file_path,
                            file_path,
                            pos,
                        )?);
                        current = Some((file_path, vector));
                    }
                }
            }
            if let Some((path, vector)) = current.take() {
                *result.entry(path.to_string()).or_default() |= vector;
            }
        }

        Ok(result)
    }

    /// Parse an equality-delete file's record-batch stream into its SURVIVAL [`Predicate`] — a row that
    /// does NOT match any of the file's delete tuples (so a row the eq-delete DELETES makes this
    /// predicate false). `pub(crate)` so the `ConvertEqualityDeleteFiles` maintenance action can reuse
    /// the exact read-side parse to build the same predicate it inverts to find matching positions.
    pub(crate) async fn parse_equality_deletes_record_batch_stream(
        stream: ArrowRecordBatchStream,
        equality_ids: HashSet<i32>,
    ) -> Result<Predicate> {
        Ok(
            Self::parse_equality_deletes_with_keyset(stream, equality_ids)
                .await?
                .0,
        )
    }

    /// Like [`parse_equality_deletes_record_batch_stream`], but ALSO returns the hashed
    /// [`EqDeleteKeySet`] accelerator when (and only when) every key column's type is eligible for
    /// the O(R) set fast path (`EqDeleteKeySet::is_eligible_type`). The predicate is built EXACTLY as
    /// before — it remains the authoritative oracle and the fallback — so a `None` set simply means
    /// "apply via the predicate path." The set's delete tuples and the predicate's per-row leaves are
    /// produced from the SAME decoded [`Datum`]s, so they encode the identical delete condition.
    ///
    /// [`parse_equality_deletes_record_batch_stream`]: Self::parse_equality_deletes_record_batch_stream
    #[allow(clippy::type_complexity)]
    pub(crate) async fn parse_equality_deletes_with_keyset(
        mut stream: ArrowRecordBatchStream,
        equality_ids: HashSet<i32>,
    ) -> Result<(Predicate, Option<EqDeleteKeySet>)> {
        let mut row_predicates = Vec::new();
        let mut batch_schema_iceberg: Option<Schema> = None;
        let accessor = EqDelRecordBatchPartnerAccessor;

        // Parallel set-path collection: the ordered key columns (captured once, from the first batch
        // that yields columns) and every delete-key tuple. `set_eligible` latches false the moment a
        // key column type is ineligible, so a float (etc.) key disables the fast path for this file.
        let mut key_columns: Option<Vec<(i32, String, PrimitiveType)>> = None;
        let mut delete_tuples: Vec<Vec<Option<Datum>>> = Vec::new();
        let mut set_eligible = true;

        while let Some(record_batch) = stream.next().await {
            let record_batch = record_batch?;

            if record_batch.num_columns() == 0 {
                return Ok((AlwaysTrue, None));
            }

            let schema = match &batch_schema_iceberg {
                Some(schema) => schema,
                None => {
                    let schema = arrow_schema_to_schema(record_batch.schema().as_ref())?;
                    batch_schema_iceberg = Some(schema);
                    batch_schema_iceberg.as_ref().unwrap()
                }
            };

            let root_array: ArrayRef = Arc::new(StructArray::from(record_batch));

            let mut processor = EqDelColumnProcessor::new(&equality_ids);
            visit_schema_with_partner(schema, &root_array, &mut processor, &accessor)?;

            let mut datum_columns_with_names = processor.finish()?;
            if datum_columns_with_names.is_empty() {
                continue;
            }

            // Capture the ordered key columns once, and check eligibility for the fast path.
            if key_columns.is_none() {
                let columns: Vec<(i32, String, PrimitiveType)> = datum_columns_with_names
                    .iter()
                    .map(|(_, field_id, field_name, primitive_type)| {
                        (*field_id, field_name.clone(), primitive_type.clone())
                    })
                    .collect();
                set_eligible &= columns
                    .iter()
                    .all(|(_, _, ty)| EqDeleteKeySet::is_eligible_type(ty));
                key_columns = Some(columns);
            }

            // Process the collected columns in lockstep
            #[allow(clippy::len_zero)]
            while datum_columns_with_names[0].0.len() > 0 {
                let mut row_predicate = AlwaysTrue;
                let mut tuple: Vec<Option<Datum>> =
                    Vec::with_capacity(datum_columns_with_names.len());
                for &mut (ref mut column, _, ref field_name, _) in &mut datum_columns_with_names {
                    if let Some(item) = column.next() {
                        let cell = item?;
                        let cell_predicate = if let Some(datum) = &cell {
                            Reference::new(field_name.clone()).equal_to(datum.clone())
                        } else {
                            Reference::new(field_name.clone()).is_null()
                        };
                        row_predicate = row_predicate.and(cell_predicate);
                        tuple.push(cell);
                    }
                }
                row_predicates.push(row_predicate.not().rewrite_not());
                delete_tuples.push(tuple);
            }
        }

        // Build the set accelerator iff every key column was eligible. `try_build` re-checks the
        // gate (defence in depth) and returns `None` for an empty / ineligible key schema.
        let key_set = if set_eligible {
            key_columns.and_then(|columns| EqDeleteKeySet::try_build(columns, delete_tuples))
        } else {
            None
        };

        // All row predicates are combined to a single predicate by creating a balanced binary tree.
        // Using a simple fold would result in a deeply nested predicate that can cause a stack overflow.
        while row_predicates.len() > 1 {
            let mut next_level = Vec::with_capacity(row_predicates.len().div_ceil(2));
            let mut iter = row_predicates.into_iter();
            while let Some(p1) = iter.next() {
                if let Some(p2) = iter.next() {
                    next_level.push(p1.and(p2));
                } else {
                    next_level.push(p1);
                }
            }
            row_predicates = next_level;
        }

        let predicate = match row_predicates.pop() {
            Some(p) => p,
            None => AlwaysTrue,
        };
        Ok((predicate, key_set))
    }
}

struct EqDelColumnProcessor<'a> {
    equality_ids: &'a HashSet<i32>,
    collected_columns: Vec<(ArrayRef, i32, String, Type)>,
}

impl<'a> EqDelColumnProcessor<'a> {
    fn new(equality_ids: &'a HashSet<i32>) -> Self {
        Self {
            equality_ids,
            collected_columns: Vec::with_capacity(equality_ids.len()),
        }
    }

    #[allow(clippy::type_complexity)]
    fn finish(
        self,
    ) -> Result<
        Vec<(
            Box<dyn ExactSizeIterator<Item = Result<Option<Datum>>>>,
            i32,
            String,
            PrimitiveType,
        )>,
    > {
        self.collected_columns
            .into_iter()
            .map(|(array, field_id, field_name, field_type)| {
                let primitive_type = field_type
                    .as_primitive_type()
                    .ok_or_else(|| {
                        Error::new(ErrorKind::Unexpected, "field is not a primitive type")
                    })?
                    .clone();

                let lit_vec = arrow_primitive_to_literal(&array, &field_type)?;
                let datum_primitive_type = primitive_type.clone();
                let datum_iterator: Box<dyn ExactSizeIterator<Item = Result<Option<Datum>>>> =
                    Box::new(lit_vec.into_iter().map(move |c| {
                        c.map(|literal| {
                            literal
                                .as_primitive_literal()
                                .map(|primitive_literal| {
                                    Datum::new(datum_primitive_type.clone(), primitive_literal)
                                })
                                .ok_or(Error::new(
                                    ErrorKind::Unexpected,
                                    "failed to convert to primitive literal",
                                ))
                        })
                        .transpose()
                    }));

                Ok((datum_iterator, field_id, field_name, primitive_type))
            })
            .collect::<Result<Vec<_>>>()
    }
}

impl SchemaWithPartnerVisitor<ArrayRef> for EqDelColumnProcessor<'_> {
    type T = ();

    fn schema(&mut self, _schema: &Schema, _partner: &ArrayRef, _value: ()) -> Result<()> {
        Ok(())
    }

    fn field(&mut self, field: &NestedFieldRef, partner: &ArrayRef, _value: ()) -> Result<()> {
        if self.equality_ids.contains(&field.id) && field.field_type.as_primitive_type().is_some() {
            self.collected_columns.push((
                partner.clone(),
                field.id,
                field.name.clone(),
                field.field_type.as_ref().clone(),
            ));
        }
        Ok(())
    }

    fn r#struct(
        &mut self,
        _struct: &StructType,
        _partner: &ArrayRef,
        _results: Vec<()>,
    ) -> Result<()> {
        Ok(())
    }

    fn list(&mut self, _list: &ListType, _partner: &ArrayRef, _value: ()) -> Result<()> {
        Ok(())
    }

    fn map(
        &mut self,
        _map: &MapType,
        _partner: &ArrayRef,
        _key_value: (),
        _value: (),
    ) -> Result<()> {
        Ok(())
    }

    fn primitive(&mut self, _primitive: &PrimitiveType, _partner: &ArrayRef) -> Result<()> {
        Ok(())
    }
}

struct EqDelRecordBatchPartnerAccessor;

impl PartnerAccessor<ArrayRef> for EqDelRecordBatchPartnerAccessor {
    fn struct_partner<'a>(&self, schema_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        Ok(schema_partner)
    }

    fn field_partner<'a>(
        &self,
        struct_partner: &'a ArrayRef,
        field: &NestedField,
    ) -> Result<&'a ArrayRef> {
        let Some(struct_array) = struct_partner.as_any().downcast_ref::<StructArray>() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Expected struct array for field extraction",
            ));
        };

        // Find the field by name within the struct
        for (i, field_def) in struct_array.fields().iter().enumerate() {
            if field_def.name() == &field.name {
                return Ok(struct_array.column(i));
            }
        }

        Err(Error::new(
            ErrorKind::Unexpected,
            format!("Field {} not found in parent struct", field.name),
        ))
    }

    fn list_element_partner<'a>(&self, _list_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "List columns are unsupported in equality deletes",
        ))
    }

    fn map_key_partner<'a>(&self, _map_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Map columns are unsupported in equality deletes",
        ))
    }

    fn map_value_partner<'a>(&self, _map_partner: &'a ArrayRef) -> Result<&'a ArrayRef> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "Map columns are unsupported in equality deletes",
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        ArrayRef, BinaryArray, Int32Array, Int64Array, RecordBatch, StringArray, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields};
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::delete_filter::tests::setup;
    use crate::scan::FileScanTaskDeleteFile;
    use crate::spec::{DataContentType, Schema};

    #[tokio::test]
    async fn test_delete_file_loader_parse_equality_deletes() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().as_os_str().to_str().unwrap();
        let file_io = FileIO::new_with_fs();

        let eq_delete_file_path = setup_write_equality_delete_file_1(table_location);

        let basic_delete_file_loader = BasicDeleteFileLoader::new(file_io.clone());
        let record_batch_stream = basic_delete_file_loader
            .parquet_to_batch_stream(
                &eq_delete_file_path,
                std::fs::metadata(&eq_delete_file_path).unwrap().len(),
            )
            .await
            .expect("could not get batch stream");

        let eq_ids = HashSet::from_iter(vec![2, 3, 4, 6, 8]);

        let parsed_eq_delete = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            record_batch_stream,
            eq_ids,
        )
        .await
        .expect("error parsing batch stream");
        println!("{parsed_eq_delete}");

        let expected = "(((((y != 1) OR (z != 100)) OR (a != \"HELP\")) OR (sa != 4)) OR (b != 62696E6172795F64617461)) AND (((((y != 2) OR (z IS NOT NULL)) OR (a IS NOT NULL)) OR (sa != 5)) OR (b IS NOT NULL))".to_string();

        assert_eq!(parsed_eq_delete.to_string(), expected);
    }

    /// M5 equivalence: the interned `parse_positional_deletes_record_batch_stream` must build the
    /// EXACT same `HashMap<path, DeleteVector>` as a straightforward per-row reference, including the
    /// edge cases the run-cache must get right: contiguous runs of one path, MULTIPLE positions for a
    /// path, a path that RECURS in a later non-contiguous run (forcing the merge-back branch), and
    /// positions split across two batches. A `DeleteVector` is a set, so duplicate positions collapse
    /// — both paths must agree on the final position set per file.
    #[tokio::test]
    async fn test_parse_positional_deletes_interning_matches_per_row_reference() {
        use futures::stream;

        // Reference per-row implementation (the pre-M5 form): one map entry resolved per row.
        fn reference(batches: &[RecordBatch]) -> HashMap<String, Vec<u64>> {
            let mut out: HashMap<String, DeleteVector> = HashMap::default();
            for batch in batches {
                let paths = batch.column(0).as_string::<i32>();
                let positions = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                for (p, pos) in paths.iter().zip(positions.iter()) {
                    out.entry(p.expect("fixture file_path is non-null").to_string())
                        .or_default()
                        .insert(
                            u64::try_from(pos.expect("fixture pos is non-null"))
                                .expect("fixture positions are non-negative"),
                        );
                }
            }
            out.into_iter()
                .map(|(k, v)| (k, v.iter().collect()))
                .collect()
        }

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("file_path", DataType::Utf8, false),
            Field::new("pos", DataType::Int64, false),
        ]));

        let mk = |paths: Vec<&str>, pos: Vec<i64>| {
            RecordBatch::try_new(schema.clone(), vec![
                Arc::new(StringArray::from(paths)) as ArrayRef,
                Arc::new(Int64Array::from(pos)) as ArrayRef,
            ])
            .unwrap()
        };

        // Batch 1: a, b, a, b, a — "a" and "b" each RECUR in non-contiguous runs, so the SECOND
        // flush of each merges onto a map entry that is already non-empty (exercises the `|=`
        // merge-back, not just insert-into-empty). Includes a duplicate position for "a".
        let b1 = mk(
            vec![
                "a.parquet",
                "a.parquet",
                "b.parquet",
                "a.parquet",
                "b.parquet",
                "a.parquet",
            ],
            vec![10, 20, 5, 10, 6, 30],
        );
        // Batch 2: b again (split across batches) and a fresh c.
        let b2 = mk(vec!["b.parquet", "c.parquet", "c.parquet"], vec![5, 1, 2]);
        let batches = vec![b1, b2];

        let expected = reference(&batches);

        let stream_batches: Vec<crate::Result<RecordBatch>> =
            batches.iter().cloned().map(Ok).collect();
        let stream = Box::pin(stream::iter(stream_batches)) as ArrowRecordBatchStream;
        let actual_map = CachingDeleteFileLoader::parse_positional_deletes_record_batch_stream(
            "pos-dels.parquet",
            stream,
        )
        .await
        .expect("parse positional deletes");

        let actual: HashMap<String, Vec<u64>> = actual_map
            .into_iter()
            .map(|(k, v)| (k, v.iter().collect()))
            .collect();

        assert_eq!(
            actual, expected,
            "interned positional-delete parse must match the per-row reference map exactly"
        );
        // Pin the exact sets so a silent regression in either path is caught.
        assert_eq!(expected.get("a.parquet").unwrap().as_slice(), &[10, 20, 30]);
        assert_eq!(expected.get("b.parquet").unwrap().as_slice(), &[5, 6]);
        assert_eq!(expected.get("c.parquet").unwrap().as_slice(), &[1, 2]);
    }

    /// Write a REAL positional-delete parquet file with the spec's `file_path`/`pos` columns
    /// (reserved field ids 2147483546 / 2147483545). `pos` is written as a NULLABLE Int64 so
    /// corrupt fixtures (null positions) are expressible; conforming writers never emit nulls
    /// there (the column is required by the spec).
    fn write_pos_del_parquet(
        dir: &std::path::Path,
        file_name: &str,
        rows: &[(&str, Option<i64>)],
    ) -> String {
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            simple_field("file_path", DataType::Utf8, false, "2147483546"),
            simple_field("pos", DataType::Int64, true, "2147483545"),
        ]));
        let paths: Vec<&str> = rows.iter().map(|(path, _)| *path).collect();
        let positions: Vec<Option<i64>> = rows.iter().map(|(_, pos)| *pos).collect();
        let batch = RecordBatch::try_new(schema.clone(), vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions)) as ArrayRef,
        ])
        .expect("build positional-delete batch");

        let path = dir
            .join(file_name)
            .to_str()
            .expect("utf-8 path")
            .to_string();
        let file = File::create(&path).expect("create positional-delete parquet");
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(props)).expect("create parquet writer");
        writer.write(&batch).expect("write positional-delete batch");
        writer.close().expect("close parquet writer");
        path
    }

    /// Build the delete-file task entry for a parquet positional-delete file.
    fn parquet_pos_del_task(pos_del_path: &str) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile {
            file_path: pos_del_path.to_string(),
            file_size_in_bytes: std::fs::metadata(pos_del_path)
                .map(|m| m.len())
                .unwrap_or(0),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: crate::spec::DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }
    }

    /// Risk pinned (audit BUG-005, run-continuation insert site): a NEGATIVE position in a
    /// position-delete file must fail CLOSED with a typed `DataInvalid` error naming the
    /// delete file and the offending position — the pre-change `pos as u64` wrapped -1 to
    /// u64::MAX, which matches no row, so the delete silently failed OPEN and the deleted row
    /// RESURRECTED. The negative row is the SECOND row of a same-path run, so it is converted
    /// by the run-continuation branch (restoring `pos as u64` at that site turns exactly this
    /// test RED via a successful load). Java parity: `BitmapPositionDeleteIndex.delete(long)`
    /// → `RoaringPositionBitmap.set` → `validatePosition` (RoaringPositionBitmap.java
    /// L311-316, 1.10.0) throws IllegalArgumentException for pos < 0 — fail-loud in both
    /// implementations. Named divergence: Java's upper bound MAX_POSITION
    /// (0x7FFF_FFFE_8000_0000, a roaring key-space limit) is NOT mirrored; Rust's
    /// RoaringTreemap supports the full u64 position range.
    #[tokio::test]
    async fn test_negative_position_in_run_fails_closed_with_data_invalid() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let file_io = FileIO::new_with_fs();

        let data_file = format!("{}/data-1.parquet", tmp_dir.path().display());
        let pos_del_path = write_pos_del_parquet(tmp_dir.path(), "neg-pos-run.parquet", &[
            (&data_file, Some(0)),
            (&data_file, Some(-1)),
        ]);

        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let error = loader
            .load_deletes(
                &[parquet_pos_del_task(&pos_del_path)],
                Arc::new(Schema::builder().build().expect("empty schema")),
            )
            .await
            .expect("loader channel")
            .expect_err("a negative position must fail the load closed, not wrap to a huge u64");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.to_string().contains(&pos_del_path),
            "error must name the delete file: {error}"
        );
        assert!(
            error.to_string().contains("negative position -1"),
            "error must name the offending position: {error}"
        );
    }

    /// Risk pinned (audit BUG-005, new-path-run insert site): the same fail-closed bar when
    /// the negative position is the FIRST row of a path's run, which is converted by the
    /// new-path branch of the run cache (restoring `pos as u64` at that site turns exactly
    /// this test RED, independently of the run-continuation site).
    #[tokio::test]
    async fn test_negative_first_position_of_path_run_fails_closed() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let file_io = FileIO::new_with_fs();

        let data_file = format!("{}/data-1.parquet", tmp_dir.path().display());
        let pos_del_path = write_pos_del_parquet(tmp_dir.path(), "neg-pos-first.parquet", &[(
            &data_file,
            Some(-5),
        )]);

        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let error = loader
            .load_deletes(
                &[parquet_pos_del_task(&pos_del_path)],
                Arc::new(Schema::builder().build().expect("empty schema")),
            )
            .await
            .expect("loader channel")
            .expect_err("a negative first-of-run position must fail the load closed");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.to_string().contains(&pos_del_path),
            "error must name the delete file: {error}"
        );
        assert!(
            error.to_string().contains("negative position -5"),
            "error must name the offending position: {error}"
        );
    }

    /// Risk pinned (audit BUG-005): a NULL position reaching the production loader path must
    /// surface as a typed `DataInvalid` error naming the delete file — never a panic and
    /// never a silently skipped row. The `pos` column is REQUIRED by the spec (Java
    /// `MetadataColumns.DELETE_FILE_POS`, MetadataColumns.java L70-74 is
    /// `NestedField.required`); Java's reader fails loud unboxing the null
    /// (`Deletes.toPositionIndexes`, Deletes.java L146 — NPE), typed here.
    #[tokio::test]
    async fn test_null_position_yields_typed_error_not_panic() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let file_io = FileIO::new_with_fs();

        let data_file = format!("{}/data-1.parquet", tmp_dir.path().display());
        let pos_del_path =
            write_pos_del_parquet(tmp_dir.path(), "null-pos.parquet", &[(&data_file, None)]);

        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let error = loader
            .load_deletes(
                &[parquet_pos_del_task(&pos_del_path)],
                Arc::new(Schema::builder().build().expect("empty schema")),
            )
            .await
            .expect("loader channel")
            .expect_err("a null position must fail the load closed with a typed error");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.to_string().contains(&pos_del_path),
            "error must name the delete file: {error}"
        );
        assert!(
            error.to_string().contains("null position"),
            "error must name the null position column: {error}"
        );
    }

    /// Risk pinned: a NULL file_path row fails closed with a typed error naming the delete
    /// file — the sibling required column of the null-position case (replacing the guard
    /// with an unwrap panics this test). Built as an in-memory batch because the parquet
    /// fixture writer declares file_path non-nullable.
    #[tokio::test]
    async fn test_null_file_path_yields_typed_error_not_panic() {
        use futures::stream;

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("file_path", DataType::Utf8, true),
            Field::new("pos", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef,
            Arc::new(Int64Array::from(vec![7i64])) as ArrayRef,
        ])
        .expect("build batch with null file_path");
        let stream = Box::pin(stream::iter(vec![Ok(batch)])) as ArrowRecordBatchStream;

        let error = CachingDeleteFileLoader::parse_positional_deletes_record_batch_stream(
            "corrupt-pos-dels.parquet",
            stream,
        )
        .await
        .expect_err("a null file_path must fail closed with a typed error");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.to_string().contains("corrupt-pos-dels.parquet"),
            "error must name the delete file: {error}"
        );
        assert!(
            error.to_string().contains("null file_path"),
            "error must name the null file_path column: {error}"
        );
    }

    /// Happy-path CONTROL for the fail-closed guards (over-broaden direction): the SAME
    /// fixture shape with valid positions — including the BOUNDARY pos = 0, the smallest
    /// legal position — must load and apply the delete correctly. An over-broadened guard
    /// (e.g. rejecting `pos <= 0` instead of `pos < 0`) turns this test RED; the negative
    /// tests alone cannot catch an over-firing guard.
    #[tokio::test]
    async fn test_valid_positions_including_zero_boundary_still_apply() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let file_io = FileIO::new_with_fs();

        let data_file = format!("{}/data-1.parquet", tmp_dir.path().display());
        let pos_del_path = write_pos_del_parquet(tmp_dir.path(), "valid-pos.parquet", &[
            (&data_file, Some(0)),
            (&data_file, Some(3)),
        ]);

        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let delete_filter = loader
            .load_deletes(
                &[parquet_pos_del_task(&pos_del_path)],
                Arc::new(Schema::builder().build().expect("empty schema")),
            )
            .await
            .expect("loader channel")
            .expect("valid positions (including the 0 boundary) must load cleanly");

        let vector = delete_filter
            .get_delete_vector_for_path(&data_file)
            .expect("delete vector installed under the data file");
        let positions: Vec<u64> = vector.lock().expect("vector lock").iter().collect();
        assert_eq!(
            positions,
            vec![0, 3],
            "the valid delete positions must apply exactly (0 is the smallest legal position)"
        );
    }

    /// Create a simple field with metadata.
    fn simple_field(name: &str, ty: DataType, nullable: bool, value: &str) -> Field {
        arrow_schema::Field::new(name, ty, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            value.to_string(),
        )]))
    }

    fn setup_write_equality_delete_file_1(table_location: &str) -> String {
        let col_y_vals = vec![1, 2];
        let col_y = Arc::new(Int64Array::from(col_y_vals)) as ArrayRef;

        let col_z_vals = vec![Some(100), None];
        let col_z = Arc::new(Int64Array::from(col_z_vals)) as ArrayRef;

        let col_a_vals = vec![Some("HELP"), None];
        let col_a = Arc::new(StringArray::from(col_a_vals)) as ArrayRef;

        let col_s = Arc::new(StructArray::from(vec![
            (
                Arc::new(simple_field("sa", DataType::Int32, false, "6")),
                Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef,
            ),
            (
                Arc::new(simple_field("sb", DataType::Utf8, true, "7")),
                Arc::new(StringArray::from(vec![Some("x"), None])) as ArrayRef,
            ),
        ]));

        let col_b_vals = vec![Some(&b"binary_data"[..]), None];
        let col_b = Arc::new(BinaryArray::from(col_b_vals)) as ArrayRef;

        let equality_delete_schema = {
            let struct_field = DataType::Struct(Fields::from(vec![
                simple_field("sa", DataType::Int32, false, "6"),
                simple_field("sb", DataType::Utf8, true, "7"),
            ]));

            let fields = vec![
                Field::new("y", arrow_schema::DataType::Int64, true).with_metadata(HashMap::from(
                    [(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())],
                )),
                Field::new("z", arrow_schema::DataType::Int64, true).with_metadata(HashMap::from(
                    [(PARQUET_FIELD_ID_META_KEY.to_string(), "3".to_string())],
                )),
                Field::new("a", arrow_schema::DataType::Utf8, true).with_metadata(HashMap::from([
                    (PARQUET_FIELD_ID_META_KEY.to_string(), "4".to_string()),
                ])),
                simple_field("s", struct_field, false, "5"),
                simple_field("b", DataType::Binary, true, "8"),
            ];
            Arc::new(arrow_schema::Schema::new(fields))
        };

        let equality_deletes_to_write = RecordBatch::try_new(equality_delete_schema.clone(), vec![
            col_y, col_z, col_a, col_s, col_b,
        ])
        .unwrap();

        let path = format!("{}/equality-deletes-1.parquet", &table_location);

        let file = File::create(&path).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(
            file,
            equality_deletes_to_write.schema(),
            Some(props.clone()),
        )
        .unwrap();

        writer
            .write(&equality_deletes_to_write)
            .expect("Writing batch");

        // writer must be closed to write footer
        writer.close().unwrap();

        path
    }

    #[tokio::test]
    async fn test_caching_delete_file_loader_load_deletes() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let delete_file_loader = CachingDeleteFileLoader::new(file_io.clone(), 10);

        let file_scan_tasks = setup(table_location);

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();

        // union of pos dels from pos del file 1 and 2, ie
        // [0, 1, 3, 5, 6, 8, 1022, 1023] | [0, 1, 3, 5, 20, 21, 22, 23]
        // = [0, 1, 3, 5, 6, 8, 20, 21, 22, 23, 1022, 1023]
        assert_eq!(result.lock().unwrap().len(), 12);

        let result = delete_filter.get_delete_vector(&file_scan_tasks[1]);
        assert!(result.is_none()); // no pos dels for file 3
    }

    /// Verifies that evolve_schema on partial-schema equality deletes works correctly
    /// when only equality_ids columns are evolved, not all table columns.
    ///
    /// Per the [Iceberg spec](https://iceberg.apache.org/spec/#equality-delete-files),
    /// equality delete files can contain only a subset of columns.
    #[tokio::test]
    async fn test_partial_schema_equality_deletes_evolve_succeeds() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().as_os_str().to_str().unwrap();

        // Create table schema with REQUIRED fields
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    crate::spec::NestedField::required(
                        1,
                        "id",
                        crate::spec::Type::Primitive(crate::spec::PrimitiveType::Int),
                    )
                    .into(),
                    crate::spec::NestedField::required(
                        2,
                        "data",
                        crate::spec::Type::Primitive(crate::spec::PrimitiveType::String),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        // Write equality delete file with PARTIAL schema (only 'data' column)
        let delete_file_path = {
            let data_vals = vec!["a", "d", "g"];
            let data_col = Arc::new(StringArray::from(data_vals)) as ArrayRef;

            let delete_schema = Arc::new(arrow_schema::Schema::new(vec![simple_field(
                "data",
                DataType::Utf8,
                false,
                "2", // field ID
            )]));

            let delete_batch = RecordBatch::try_new(delete_schema.clone(), vec![data_col]).unwrap();

            let path = format!("{}/partial-eq-deletes.parquet", &table_location);
            let file = File::create(&path).unwrap();
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .build();
            let mut writer =
                ArrowWriter::try_new(file, delete_batch.schema(), Some(props)).unwrap();
            writer.write(&delete_batch).expect("Writing batch");
            writer.close().unwrap();
            path
        };

        let file_io = FileIO::new_with_fs();
        let basic_delete_file_loader = BasicDeleteFileLoader::new(file_io.clone());

        let batch_stream = basic_delete_file_loader
            .parquet_to_batch_stream(
                &delete_file_path,
                std::fs::metadata(&delete_file_path).unwrap().len(),
            )
            .await
            .unwrap();

        // Only evolve the equality_ids columns (field 2), not all table columns
        let equality_ids = vec![2];
        let evolved_stream =
            BasicDeleteFileLoader::evolve_schema(batch_stream, table_schema, &equality_ids)
                .await
                .unwrap();

        let result = evolved_stream.try_collect::<Vec<_>>().await;

        assert!(
            result.is_ok(),
            "Expected success when evolving only equality_ids columns, got error: {:?}",
            result.err()
        );

        let batches = result.unwrap();
        assert_eq!(batches.len(), 1);

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 1); // Only 'data' column

        // Verify the actual values are preserved after schema evolution
        let data_col = batch.column(0).as_string::<i32>();
        assert_eq!(data_col.value(0), "a");
        assert_eq!(data_col.value(1), "d");
        assert_eq!(data_col.value(2), "g");
    }

    /// Test loading a FileScanTask with BOTH positional and equality deletes.
    /// Verifies the fix for the inverted condition that caused "Missing predicate for equality delete file" errors.
    #[tokio::test]
    async fn test_load_deletes_with_mixed_types() {
        use crate::scan::FileScanTask;
        use crate::spec::{DataFileFormat, Schema};

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        // Create the data file schema
        let data_file_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    crate::spec::NestedField::optional(
                        2,
                        "y",
                        crate::spec::Type::Primitive(crate::spec::PrimitiveType::Long),
                    )
                    .into(),
                    crate::spec::NestedField::optional(
                        3,
                        "z",
                        crate::spec::Type::Primitive(crate::spec::PrimitiveType::Long),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        // Write positional delete file
        let positional_delete_schema = crate::arrow::delete_filter::tests::create_pos_del_schema();
        let file_path_values =
            vec![format!("{}/data-1.parquet", table_location.to_str().unwrap()); 4];
        let file_path_col = Arc::new(StringArray::from_iter_values(&file_path_values));
        let pos_col = Arc::new(Int64Array::from_iter_values(vec![0i64, 1, 2, 3]));

        let positional_deletes_to_write =
            RecordBatch::try_new(positional_delete_schema.clone(), vec![
                file_path_col,
                pos_col,
            ])
            .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let pos_del_path = format!("{}/pos-del-mixed.parquet", table_location.to_str().unwrap());
        let file = File::create(&pos_del_path).unwrap();
        let mut writer = ArrowWriter::try_new(
            file,
            positional_deletes_to_write.schema(),
            Some(props.clone()),
        )
        .unwrap();
        writer.write(&positional_deletes_to_write).unwrap();
        writer.close().unwrap();

        // Write equality delete file
        let eq_delete_path = setup_write_equality_delete_file_1(table_location.to_str().unwrap());

        // Create FileScanTask with BOTH positional and equality deletes
        let pos_del = FileScanTaskDeleteFile {
            file_path: pos_del_path.clone(),
            file_size_in_bytes: std::fs::metadata(&pos_del_path).unwrap().len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let eq_del = FileScanTaskDeleteFile {
            file_path: eq_delete_path.clone(),
            file_size_in_bytes: std::fs::metadata(&eq_delete_path).unwrap().len(),
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![2, 3]), // Only use field IDs that exist in both schemas
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let file_scan_task = FileScanTask {
            file_size_in_bytes: 0,
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: format!("{}/data-1.parquet", table_location.to_str().unwrap()),
            data_file_format: DataFileFormat::Parquet,
            schema: data_file_schema.clone(),
            project_field_ids: vec![2, 3],
            predicate: None,
            deletes: vec![pos_del, eq_del],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
        };

        // Load the deletes - should handle both types without error
        let delete_file_loader = CachingDeleteFileLoader::new(file_io.clone(), 10);
        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_task.deletes, file_scan_task.schema_ref())
            .await
            .unwrap()
            .unwrap();

        // Verify both delete types can be processed together
        let result = delete_filter
            .build_equality_delete_predicate(&file_scan_task)
            .await;
        assert!(
            result.is_ok(),
            "Failed to build equality delete predicate: {:?}",
            result.err()
        );
    }

    #[tokio::test]
    async fn test_large_equality_delete_batch_stack_overflow() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().as_os_str().to_str().unwrap();
        let file_io = FileIO::new_with_fs();

        // Create a large batch of equality deletes
        let num_rows = 20_000;
        let col_y_vals: Vec<i64> = (0..num_rows).collect();
        let col_y = Arc::new(Int64Array::from(col_y_vals)) as ArrayRef;

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("y", arrow_schema::DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));

        let record_batch = RecordBatch::try_new(schema.clone(), vec![col_y]).unwrap();

        // Write to file
        let path = format!("{}/large-eq-deletes.parquet", &table_location);
        let file = File::create(&path).unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&record_batch).unwrap();
        writer.close().unwrap();

        let basic_delete_file_loader = BasicDeleteFileLoader::new(file_io.clone());
        let record_batch_stream = basic_delete_file_loader
            .parquet_to_batch_stream(&path, std::fs::metadata(&path).unwrap().len())
            .await
            .expect("could not get batch stream");

        let eq_ids = HashSet::from_iter(vec![2]);

        let result = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            record_batch_stream,
            eq_ids,
        )
        .await;

        assert!(result.is_ok());
    }

    /// Write a REAL Puffin file containing one `deletion-vector-v1` blob for
    /// `referenced_data_file` with the given deleted positions, and return
    /// `(puffin_path, content_offset, content_size_in_bytes)` read back from the Puffin footer
    /// (the same coordinates a manifest's `DeleteFile` would carry).
    async fn write_dv_puffin_file(
        file_io: &FileIO,
        dir: &std::path::Path,
        file_name: &str,
        referenced_data_file: &str,
        positions: &[u64],
    ) -> (String, i64, i64) {
        use crate::puffin::{Blob, CompressionCodec, PuffinReader, PuffinWriter};

        let blob_bytes = crate::delete_vector::tests::encode_deletion_vector_v1(positions);
        let puffin_path = dir
            .join(file_name)
            .to_str()
            .expect("utf-8 path")
            .to_string();

        let output_file = file_io.new_output(&puffin_path).expect("new output");
        let mut writer = PuffinWriter::new(&output_file, HashMap::new(), false)
            .await
            .expect("create puffin writer");
        writer
            .add(
                Blob::builder()
                    .r#type(crate::puffin::DELETION_VECTOR_V1.to_string())
                    .fields(vec![])
                    .snapshot_id(-1)
                    .sequence_number(-1)
                    .data(blob_bytes)
                    .properties(HashMap::from([
                        (
                            "referenced-data-file".to_string(),
                            referenced_data_file.to_string(),
                        ),
                        ("cardinality".to_string(), positions.len().to_string()),
                    ]))
                    .build(),
                CompressionCodec::None,
            )
            .await
            .expect("add DV blob");
        writer.close().await.expect("close puffin writer");

        // Read the blob coordinates back from the footer — exactly what BaseDVFileWriter records
        // into the DeleteFile's content_offset / content_size_in_bytes.
        let input_file = file_io.new_input(&puffin_path).expect("new input");
        let puffin_reader = PuffinReader::new(input_file);
        let footer = puffin_reader.file_metadata().await.expect("read footer");
        let blob_metadata = footer.blobs().first().expect("one blob");
        (
            puffin_path,
            i64::try_from(blob_metadata.offset()).expect("offset fits i64"),
            i64::try_from(blob_metadata.length()).expect("length fits i64"),
        )
    }

    /// Write a REAL Puffin file containing MULTIPLE `deletion-vector-v1` blobs (one per
    /// `(referenced_data_file, positions)` pair, in order) and return
    /// `(puffin_path, vec![(content_offset, content_size_in_bytes)])` read back from the footer.
    async fn write_multi_dv_puffin_file(
        file_io: &FileIO,
        dir: &std::path::Path,
        file_name: &str,
        vectors: &[(&str, &[u64])],
    ) -> (String, Vec<(i64, i64)>) {
        use crate::puffin::{Blob, CompressionCodec, PuffinReader, PuffinWriter};

        let puffin_path = dir
            .join(file_name)
            .to_str()
            .expect("utf-8 path")
            .to_string();

        let output_file = file_io.new_output(&puffin_path).expect("new output");
        let mut writer = PuffinWriter::new(&output_file, HashMap::new(), false)
            .await
            .expect("create puffin writer");
        for (referenced_data_file, positions) in vectors {
            let blob_bytes = crate::delete_vector::tests::encode_deletion_vector_v1(positions);
            writer
                .add(
                    Blob::builder()
                        .r#type(crate::puffin::DELETION_VECTOR_V1.to_string())
                        .fields(vec![])
                        .snapshot_id(-1)
                        .sequence_number(-1)
                        .data(blob_bytes)
                        .properties(HashMap::from([
                            (
                                "referenced-data-file".to_string(),
                                referenced_data_file.to_string(),
                            ),
                            ("cardinality".to_string(), positions.len().to_string()),
                        ]))
                        .build(),
                    CompressionCodec::None,
                )
                .await
                .expect("add DV blob");
        }
        writer.close().await.expect("close puffin writer");

        let input_file = file_io.new_input(&puffin_path).expect("new input");
        let puffin_reader = PuffinReader::new(input_file);
        let footer = puffin_reader.file_metadata().await.expect("read footer");
        let coordinates = footer
            .blobs()
            .iter()
            .map(|blob_metadata| {
                (
                    i64::try_from(blob_metadata.offset()).expect("offset fits i64"),
                    i64::try_from(blob_metadata.length()).expect("length fits i64"),
                )
            })
            .collect();
        (puffin_path, coordinates)
    }

    /// Build the delete-file task entry for a deletion vector.
    fn dv_task(
        puffin_path: &str,
        referenced_data_file: &str,
        content_offset: i64,
        content_size_in_bytes: i64,
        record_count: u64,
    ) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile {
            file_path: puffin_path.to_string(),
            file_size_in_bytes: std::fs::metadata(puffin_path).map(|m| m.len()).unwrap_or(0),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: crate::spec::DataFileFormat::Puffin,
            referenced_data_file: Some(referenced_data_file.to_string()),
            content_offset: Some(content_offset),
            content_size_in_bytes: Some(content_size_in_bytes),
            record_count: Some(record_count),
        }
    }

    /// Risk pinned: the loader DISPATCH — a position delete in PUFFIN format must be routed to
    /// the DV blob decoder, not `parquet_to_batch_stream` (which would fail on Puffin bytes —
    /// the pre-change behavior). The decoded positions must be installed under the REFERENCED
    /// data file and ONLY there: not under the Puffin file's own path (the mutation-(b)
    /// sentinel) and not under a sibling data file.
    #[tokio::test]
    async fn test_dv_routes_to_dv_loader_and_keys_by_referenced_data_file() {
        let tmp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        let data_file_a = format!("{}/data-a.parquet", tmp_dir.path().display());
        let data_file_b = format!("{}/data-b.parquet", tmp_dir.path().display());
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file_a, &[
                1, 3,
            ])
            .await;

        let loader = CachingDeleteFileLoader::new(file_io.clone(), 10);
        let delete_filter = loader
            .load_deletes(
                &[dv_task(&puffin_path, &data_file_a, offset, length, 2)],
                Arc::new(Schema::builder().build().unwrap()),
            )
            .await
            .unwrap()
            .expect("DV load must succeed (parquet routing would fail here)");

        let vector = delete_filter
            .get_delete_vector_for_path(&data_file_a)
            .expect("vector must be keyed by the referenced data file");
        let positions: Vec<u64> = vector.lock().unwrap().iter().collect();
        assert_eq!(positions, vec![1, 3]);

        assert!(
            delete_filter
                .get_delete_vector_for_path(&puffin_path)
                .is_none(),
            "the vector must NOT be keyed by the Puffin file's own path"
        );
        assert!(
            delete_filter
                .get_delete_vector_for_path(&data_file_b)
                .is_none(),
            "a DV for data file A must not leak onto sibling data file B"
        );
    }

    /// Risk pinned: cache-hit semantics — loading the same DV blob twice through one loader must
    /// reuse the first decoded vector (`{path}@{offset}` dedup), not decode + union a second copy.
    #[tokio::test]
    async fn test_dv_second_load_reuses_cached_vector() {
        let tmp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        let data_file_a = format!("{}/data-a.parquet", tmp_dir.path().display());
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file_a, &[
                0, 2, 4,
            ])
            .await;

        let loader = CachingDeleteFileLoader::new(file_io.clone(), 10);
        let tasks = [dv_task(&puffin_path, &data_file_a, offset, length, 3)];
        let schema = Arc::new(Schema::builder().build().unwrap());

        let filter_1 = loader
            .load_deletes(&tasks, schema.clone())
            .await
            .unwrap()
            .expect("first DV load");
        let filter_2 = loader
            .load_deletes(&tasks, schema)
            .await
            .unwrap()
            .expect("second DV load");

        let vector_1 = filter_1.get_delete_vector_for_path(&data_file_a).unwrap();
        let vector_2 = filter_2.get_delete_vector_for_path(&data_file_a).unwrap();
        assert!(
            Arc::ptr_eq(&vector_1, &vector_2),
            "the second load must reuse the cached vector"
        );
        assert_eq!(
            vector_1.lock().unwrap().len(),
            3,
            "re-loading must not union a second copy into the vector"
        );
    }

    /// Risk pinned (reviewer, 2026-06-10): TWO deletion vectors in ONE Puffin file (different
    /// offsets, different referenced data files) — the exact case the `{path}@{offset}` cache
    /// key exists for. A bare-file-path key would mark blob 2 "already loaded" when blob 1
    /// finishes, silently dropping B's vector and resurrecting its deleted rows. Both blobs must
    /// load, and each must land under its own referenced data file.
    #[tokio::test]
    async fn test_two_dvs_in_one_puffin_file_both_load_under_own_data_file() {
        let tmp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        let data_file_a = format!("{}/data-a.parquet", tmp_dir.path().display());
        let data_file_b = format!("{}/data-b.parquet", tmp_dir.path().display());
        let (puffin_path, coordinates) =
            write_multi_dv_puffin_file(&file_io, tmp_dir.path(), "two-blobs.puffin", &[
                (&data_file_a, &[1, 3]),
                (&data_file_b, &[0, 2, 4]),
            ])
            .await;
        assert_eq!(coordinates.len(), 2, "fixture must hold two blobs");
        assert_ne!(
            coordinates[0].0, coordinates[1].0,
            "the two blobs must sit at distinct offsets"
        );

        let loader = CachingDeleteFileLoader::new(file_io.clone(), 10);
        let delete_filter = loader
            .load_deletes(
                &[
                    dv_task(
                        &puffin_path,
                        &data_file_a,
                        coordinates[0].0,
                        coordinates[0].1,
                        2,
                    ),
                    dv_task(
                        &puffin_path,
                        &data_file_b,
                        coordinates[1].0,
                        coordinates[1].1,
                        3,
                    ),
                ],
                Arc::new(Schema::builder().build().unwrap()),
            )
            .await
            .unwrap()
            .expect("both DV blobs in one Puffin file must load");

        let vector_a = delete_filter
            .get_delete_vector_for_path(&data_file_a)
            .expect("blob 1 must land under data file A");
        let positions_a: Vec<u64> = vector_a.lock().unwrap().iter().collect();
        assert_eq!(positions_a, vec![1, 3]);

        let vector_b = delete_filter
            .get_delete_vector_for_path(&data_file_b)
            .expect("blob 2 must land under data file B (not be marked already-loaded)");
        let positions_b: Vec<u64> = vector_b.lock().unwrap().iter().collect();
        assert_eq!(positions_b, vec![0, 2, 4]);
    }

    /// Risk pinned: TWO deletion vectors claiming the same data file is an invalid table state
    /// Java rejects at index-build ("Can't index multiple DVs for %s", DeleteFileIndex.java
    /// L528-535); the Rust loader rejects it at the load door — silently unioning would
    /// over-delete, keeping one would resurrect rows.
    #[tokio::test]
    async fn test_multiple_dvs_for_one_data_file_rejected() {
        let tmp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        let data_file_a = format!("{}/data-a.parquet", tmp_dir.path().display());
        let (puffin_1, offset_1, length_1) = write_dv_puffin_file(
            &file_io,
            tmp_dir.path(),
            "deletes-1.puffin",
            &data_file_a,
            &[1],
        )
        .await;
        let (puffin_2, offset_2, length_2) = write_dv_puffin_file(
            &file_io,
            tmp_dir.path(),
            "deletes-2.puffin",
            &data_file_a,
            &[3],
        )
        .await;

        let loader = CachingDeleteFileLoader::new(file_io.clone(), 10);
        let result = loader
            .load_deletes(
                &[
                    dv_task(&puffin_1, &data_file_a, offset_1, length_1, 1),
                    dv_task(&puffin_2, &data_file_a, offset_2, length_2, 1),
                ],
                Arc::new(Schema::builder().build().unwrap()),
            )
            .await
            .unwrap();

        let error = result.expect_err("duplicate DVs for one data file must be rejected");
        assert!(
            error
                .to_string()
                .contains("multiple deletion vectors for data file"),
            "error must name the duplicate-DV failure: {error}"
        );
    }

    /// Risk pinned: the metadata validations at the DV load door (Java
    /// `BaseDeleteLoader.validateDV`) — missing offset, out-of-range size, and a missing
    /// referenced data file each reject cleanly BY NAME, never panic or fall through to the
    /// parquet reader.
    #[tokio::test]
    async fn test_dv_invalid_metadata_rejected_cleanly() {
        let tmp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        let data_file_a = format!("{}/data-a.parquet", tmp_dir.path().display());
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file_a, &[
                1,
            ])
            .await;
        let schema = Arc::new(Schema::builder().build().unwrap());
        let loader = CachingDeleteFileLoader::new(file_io.clone(), 10);

        // Missing content_offset (Java: "Invalid DV, offset cannot be null").
        let mut missing_offset = dv_task(&puffin_path, &data_file_a, offset, length, 1);
        missing_offset.content_offset = None;
        let error = loader
            .load_deletes(&[missing_offset], schema.clone())
            .await
            .unwrap()
            .expect_err("missing content_offset must reject");
        assert!(error.to_string().contains("content_offset"), "{error}");

        // content_size_in_bytes above 2GB (Java: "Can't read DV larger than 2GB").
        let mut oversize = dv_task(&puffin_path, &data_file_a, offset, length, 1);
        oversize.content_size_in_bytes = Some(i64::from(i32::MAX) + 1);
        let error = loader
            .load_deletes(&[oversize], schema.clone())
            .await
            .unwrap()
            .expect_err("oversize content_size_in_bytes must reject");
        assert!(
            error.to_string().contains("content_size_in_bytes"),
            "{error}"
        );

        // Missing referenced_data_file (the keying prerequisite; mandatory per the Puffin spec).
        let mut missing_referenced = dv_task(&puffin_path, &data_file_a, offset, length, 1);
        missing_referenced.referenced_data_file = None;
        let error = loader
            .load_deletes(&[missing_referenced], schema.clone())
            .await
            .unwrap()
            .expect_err("missing referenced_data_file must reject");
        assert!(
            error.to_string().contains("referenced_data_file"),
            "{error}"
        );
    }

    /// Risk pinned: the manifest's record_count is the DV's cardinality; a decoded bitmap whose
    /// cardinality disagrees means the manifest and the blob diverge (Java `deserializeBitmap`:
    /// "Invalid cardinality: %s, expected %s") — silent acceptance would hide corruption.
    #[tokio::test]
    async fn test_dv_cardinality_mismatch_rejected() {
        let tmp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        let data_file_a = format!("{}/data-a.parquet", tmp_dir.path().display());
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file_a, &[
                1, 3,
            ])
            .await;

        let loader = CachingDeleteFileLoader::new(file_io.clone(), 10);
        let error = loader
            .load_deletes(
                // record_count says 5, the blob holds 2 positions.
                &[dv_task(&puffin_path, &data_file_a, offset, length, 5)],
                Arc::new(Schema::builder().build().unwrap()),
            )
            .await
            .unwrap()
            .expect_err("cardinality mismatch must reject");
        assert!(error.to_string().contains("cardinality"), "{error}");
    }

    /// Write a REAL parquet data file of one Int64 `id` column (field id 1) and return its path.
    fn write_data_parquet(dir: &std::path::Path, file_name: &str, ids: &[i64]) -> String {
        let schema = Arc::new(arrow_schema::Schema::new(vec![simple_field(
            "id",
            DataType::Int64,
            false,
            "1",
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![
                Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
            ])
            .unwrap();

        let path = dir
            .join(file_name)
            .to_str()
            .expect("utf-8 path")
            .to_string();
        let file = File::create(&path).unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    /// Build a [`crate::scan::FileScanTask`] over a real parquet data file with the given deletes.
    fn data_scan_task(
        data_file_path: &str,
        schema: SchemaRef,
        deletes: Vec<FileScanTaskDeleteFile>,
    ) -> crate::scan::FileScanTask {
        crate::scan::FileScanTask {
            file_size_in_bytes: std::fs::metadata(data_file_path)
                .map(|m| m.len())
                .unwrap_or(0),
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: data_file_path.to_string(),
            data_file_format: crate::spec::DataFileFormat::Parquet,
            schema,
            project_field_ids: vec![1],
            predicate: None,
            deletes,
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
        }
    }

    /// Risk pinned (scan-level): a deletion vector applied during a REAL Arrow read — the rows
    /// at the DV's positions are ABSENT from the data file it references while a SIBLING data
    /// file in the same scan is untouched. This is the read-machinery proof that the decoded
    /// vector flows loader → DeleteFilter → ArrowReader row selection; under the
    /// key-by-DV-file-path mutation the deleted rows resurrect and this test fails.
    #[tokio::test]
    async fn test_scan_with_dv_masks_positions_and_spares_sibling_file() {
        use futures::TryStreamExt;

        use crate::arrow::ArrowReaderBuilder;
        use crate::scan::FileScanTaskStream;

        let tmp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();

        let data_file_a =
            write_data_parquet(tmp_dir.path(), "data-a.parquet", &[10, 20, 30, 40, 50]);
        let data_file_b = write_data_parquet(tmp_dir.path(), "data-b.parquet", &[60, 70, 80]);

        // The DV deletes positions {1, 3} of data file A (ids 20 and 40).
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file_a, &[
                1, 3,
            ])
            .await;

        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    crate::spec::NestedField::required(
                        1,
                        "id",
                        crate::spec::Type::Primitive(crate::spec::PrimitiveType::Long),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let tasks: Vec<crate::Result<crate::scan::FileScanTask>> = vec![
            Ok(data_scan_task(&data_file_a, table_schema.clone(), vec![
                dv_task(&puffin_path, &data_file_a, offset, length, 2),
            ])),
            Ok(data_scan_task(&data_file_b, table_schema.clone(), vec![])),
        ];

        let reader = ArrowReaderBuilder::new(file_io).build();
        let batches: Vec<RecordBatch> = reader
            .read(Box::pin(futures::stream::iter(tasks)) as FileScanTaskStream)
            .expect("build record batch stream")
            .try_collect()
            .await
            .expect("read scan tasks");

        let mut ids: Vec<i64> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("id")
                    .expect("id column")
                    .as_primitive::<arrow_array::types::Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();

        assert_eq!(
            ids,
            vec![10, 30, 50, 60, 70, 80],
            "ids 20/40 (DV positions 1 and 3 of file A) must be absent; file B must be intact"
        );
    }

    #[tokio::test]
    async fn test_caching_delete_file_loader_caches_results() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let delete_file_loader = CachingDeleteFileLoader::new(file_io.clone(), 10);

        let file_scan_tasks = setup(table_location);

        // Load deletes for the first time
        let delete_filter_1 = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        // Load deletes for the second time (same task/files)
        let delete_filter_2 = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let dv1 = delete_filter_1
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();
        let dv2 = delete_filter_2
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();

        // Verify that the delete vectors point to the same memory location,
        // confirming that the second load reused the result from the first.
        assert!(Arc::ptr_eq(&dv1, &dv2));
    }
}
