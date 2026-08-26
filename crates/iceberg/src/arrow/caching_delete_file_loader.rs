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

use super::delete_filter::{DeleteFilter, PosDelLoadAction, PosDelLoadGuard, pos_del_claim_key};
use crate::arrow::delete_file_loader::BasicDeleteFileLoader;
use crate::arrow::equality_delete_set::EqDeleteKeySet;
use crate::arrow::null_propagation::propagate_struct_validity;
use crate::arrow::{arrow_primitive_to_literal, arrow_schema_to_schema};
use crate::delete_vector::{DeleteVector, validate_delete_vector_coordinates};
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
    /// A positional delete file this task claimed. `guard` is that claim. It publishes the terminal
    /// load state and wakes the waiters: on success through `publish_loaded`, and on every failure
    /// path through its `Drop`. It travels with the context, so the claim never outlives its task.
    PosDels {
        guard: PosDelLoadGuard,
        file_path: String,
        stream: ArrowRecordBatchStream,
    },
    /// A freshly loaded and decoded Puffin deletion vector. `guard` carries the dedup key
    /// `{puffin path}@{blob offset}`. One Puffin file holds many blobs, so a bare file-path key would
    /// wrongly mark every later blob loaded. `referenced_data_file` is the data file the vector applies
    /// to, and the key it is installed under in the [`DeleteFilter`].
    FreshDeletionVector {
        guard: PosDelLoadGuard,
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
        guard: PosDelLoadGuard,
        results: HashMap<String, DeleteVector>,
    },
    EqDel,
    ExistingPosDel,
}

impl CachingDeleteFileLoader {
    pub(crate) fn new(file_io: FileIO, concurrency_limit_data_files: usize) -> Self {
        CachingDeleteFileLoader {
            basic_delete_file_loader: BasicDeleteFileLoader::new(file_io),
            concurrency_limit_data_files,
            delete_filter: DeleteFilter::default(),
        }
    }

    /// Start loading every delete for the given tasks.
    ///
    /// # Notes
    ///
    /// The returned future completes once all positional deletes and deletion vectors have loaded.
    /// This method does not wait for equality deletes. The returned [`DeleteFilter`] awaits those when
    /// a caller queries them.
    ///
    /// One stream carries every delete file task, whatever its type, so one concurrency limit covers
    /// them all. A notifier stops two data-file tasks from loading the same equality delete. A
    /// positional delete in PUFFIN format is a deletion vector, and its `{path}@{offset}` key dedups
    /// concurrent loads of one blob.
    ///
    /// Each positional source's parsed map installs per source, under its claim key. It never merges
    /// into one shared data-file-keyed map, so `DeleteFilter::resolve_delete_vector` scopes application
    /// to each task's own delete files. Java builds one `DeleteFilter` per task.
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
    ///                                 [install each source's map under its claim key]
    ///                              HashMap<claim key, HashMap<String, RoaringTreeMap>>
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

        // A data file must carry at most one deletion vector. Java rejects a duplicate in
        // `DeleteFileIndex.Builder.add`. The Rust index lookup is infallible by signature, so this
        // load door rejects the same invalid state before any vector is installed. Unioning two
        // vectors would over-delete, and keeping one would resurrect rows.
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
            .map(|t| (t.clone(), self.delete_filter.clone(), schema.clone()))
            .collect::<Vec<_>>();
        let task_stream = futures::stream::iter(stream_items);

        let del_filter = self.delete_filter.clone();
        let concurrency_limit_data_files = self.concurrency_limit_data_files;
        let basic_delete_file_loader = self.basic_delete_file_loader.clone();
        crate::runtime::spawn(async move {
            let result = async move {
                let del_filter = del_filter;
                let basic_delete_file_loader = basic_delete_file_loader.clone();

                let mut results_stream = task_stream
                    .map(move |(task, del_filter, schema)| {
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
                    if let ParsedDeleteFileContext::DelVecs { guard, results } = item {
                        // Install this source's map under its claim key, not merged into shared
                        // per-data-file state. Delete application then scopes to each task's own
                        // delete files. Merging here lets a source loaded for one task delete rows
                        // from another task's file.
                        del_filter.install_pos_del_contribution(&guard, results);
                        // Mark the file loaded so waiters proceed. This must come after the install,
                        // in the same await-free block. A woken waiter resolves the contribution
                        // synchronously, so publishing first would hand it an absent result.
                        guard.publish_loaded();
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
            // A position delete in PUFFIN format is a deletion vector. It must reach the blob loader.
            // The parquet reader misparses it.
            DataContentType::PositionDeletes if task.file_format == DataFileFormat::Puffin => {
                Self::load_deletion_vector_for_task(task, &basic_delete_file_loader, &del_filter)
                    .await
            }

            DataContentType::PositionDeletes => {
                match del_filter.try_start_pos_del_load(&task.file_path)? {
                    PosDelLoadAction::AlreadyLoaded => Ok(DeleteFileContext::ExistingPosDel),
                    PosDelLoadAction::WaitFor(notified) => {
                        // Positional deletes are accessed synchronously by ArrowReader.
                        // We must wait here to ensure the data is ready before returning,
                        // otherwise ArrowReader might get an empty/partial result. A loader that
                        // died without publishing surfaces as a typed error here, never a hang.
                        del_filter
                            .wait_for_pos_del_load(&task.file_path, notified)
                            .await?;
                        Ok(DeleteFileContext::ExistingPosDel)
                    }
                    PosDelLoadAction::Load(mut guard) => {
                        // `guard` is a local. An `Err` from the stream open below returns early and
                        // drops it, publishing the terminal failed state to every waiter.
                        // `note_failure` hands the error to the guard first, so the waiters' typed
                        // error names the cause and not just the file.
                        //
                        // Project only the reserved `file_path` and `pos` columns. The read falls
                        // back to a full read when the ProjectionMask cannot be built safely.
                        let stream = basic_delete_file_loader
                            .parquet_positional_delete_batch_stream(
                                &task.file_path,
                                task.file_size_in_bytes,
                            )
                            .await
                            .map_err(|error| guard.note_failure(error))?;
                        Ok(DeleteFileContext::PosDels {
                            guard,
                            file_path: task.file_path.clone(),
                            stream,
                        })
                    }
                }
            }

            DataContentType::EqualityDeletes => {
                let Some(guard) = del_filter.try_start_eq_del_load(&task.file_path) else {
                    return Ok(DeleteFileContext::ExistingEqDel);
                };

                let (sender, receiver) = channel();
                guard.spawn_publisher(receiver);

                // Per the Iceberg spec, evolve schema for equality deletes but only for the
                // equality_ids columns, not all table columns.
                //
                // A malformed or foreign eq-delete task can arrive with `equality_ids: None`. Fail
                // loud with a typed error naming the file, rather than panic the scan on `unwrap`.
                // Java's `DeleteLoader` also throws on malformed delete metadata. The early return
                // drops `sender`, and the receiver task turns that into a terminal
                // `EqDelState::Failed`, so no waiter is stranded.
                let Some(equality_ids_vec) = task.equality_ids.clone() else {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Invalid equality delete file '{}': missing equality_ids (an \
                             equality-delete task must carry the field ids its rows delete on)",
                            task.file_path
                        ),
                    ));
                };
                // Project only the equality_ids key columns. The read falls back to a full read when
                // the ProjectionMask cannot be built safely.
                let evolved_stream = BasicDeleteFileLoader::evolve_schema(
                    basic_delete_file_loader
                        .parquet_to_batch_stream_with_projection(
                            &task.file_path,
                            task.file_size_in_bytes,
                            Some(equality_ids_vec.as_slice()),
                        )
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
    /// This mirrors Java's `BaseDeleteLoader.readDV`. It does one ranged read at `content_offset` of
    /// `content_size_in_bytes` bytes, then the `deletion-vector-v1` deserialization. A Puffin footer
    /// round-trip costs three or more requests, and the manifest already names the exact blob range.
    async fn load_deletion_vector_for_task(
        task: &FileScanTaskDeleteFile,
        basic_delete_file_loader: &BasicDeleteFileLoader,
        del_filter: &DeleteFilter,
    ) -> Result<DeleteFileContext> {
        let (referenced_data_file, content_offset, content_size_in_bytes) =
            Self::validate_deletion_vector_task(task)?;

        // Claim under the shared `pos_del_claim_key` derivation, so the load key is byte-identical to
        // the key `resolve_delete_vector` looks up. Key drift silently drops the vector. The
        // validation above guarantees a present, non-negative offset, so the error arm is defensive.
        let cache_key = pos_del_claim_key(task).ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "no claim key derivable for validated deletion vector '{}'",
                    task.file_path
                ),
            )
        })?;
        match del_filter.try_start_pos_del_load(&cache_key)? {
            PosDelLoadAction::AlreadyLoaded => Ok(DeleteFileContext::ExistingPosDel),
            PosDelLoadAction::WaitFor(notified) => {
                // Like parquet positional deletes, the decoded vector must be fully available
                // before ArrowReader proceeds (retrieval is synchronous). A loader that died
                // without publishing surfaces as a typed error here, never a hang.
                del_filter
                    .wait_for_pos_del_load(&cache_key, notified)
                    .await?;
                Ok(DeleteFileContext::ExistingPosDel)
            }
            // `guard` is a local: every `?` below returns early and drops it, publishing the
            // terminal failed state to every waiter on this blob — carrying the cause, which each
            // failure path hands over with `note_failure` before propagating it.
            PosDelLoadAction::Load(mut guard) => {
                let blob = basic_delete_file_loader
                    .read_bytes_range(&task.file_path, content_offset, content_size_in_bytes)
                    .await
                    .map_err(|error| guard.note_failure(error))?;
                let delete_vector = DeleteVector::deserialize_deletion_vector_v1(&blob)
                    .map_err(|error| guard.note_failure(error))?;

                // Java validates the decoded cardinality against the DeleteFile's recordCount. A
                // mismatch means the manifest and the blob disagree on how many rows are deleted.
                if let Some(expected_cardinality) = task.record_count
                    && delete_vector.len() != expected_cardinality
                {
                    return Err(guard.note_failure(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Invalid deletion vector cardinality for '{}': decoded {} positions, \
                             manifest record_count expects {expected_cardinality}",
                            task.file_path,
                            delete_vector.len(),
                        ),
                    )));
                }

                Ok(DeleteFileContext::FreshDeletionVector {
                    guard,
                    referenced_data_file,
                    delete_vector,
                })
            }
        }
    }

    /// Validates a delete-file task's DV metadata, returning
    /// `(referenced_data_file, content_offset, content_size_in_bytes)`.
    ///
    /// # Notes
    ///
    /// The rules live in [`validate_delete_vector_coordinates`] so this path and the public
    /// writer-side loader cannot drift apart.
    fn validate_deletion_vector_task(task: &FileScanTaskDeleteFile) -> Result<(String, u64, u64)> {
        let coordinates = validate_delete_vector_coordinates(
            &task.file_path,
            task.referenced_data_file.clone(),
            task.content_offset,
            task.content_size_in_bytes,
        )?;
        Ok((
            coordinates.referenced_data_file,
            coordinates.content_offset,
            coordinates.content_size_in_bytes,
        ))
    }

    async fn parse_file_content_for_task(
        ctx: DeleteFileContext,
    ) -> Result<ParsedDeleteFileContext> {
        match ctx {
            DeleteFileContext::ExistingEqDel => Ok(ParsedDeleteFileContext::EqDel),
            DeleteFileContext::ExistingPosDel => Ok(ParsedDeleteFileContext::ExistingPosDel),
            // `guard` is a local: a parse error returns early and drops it, publishing the terminal
            // failed state — with the parse error as its cause — to every waiter on this file.
            DeleteFileContext::PosDels {
                mut guard,
                file_path,
                stream,
            } => {
                let del_vecs =
                    Self::parse_positional_deletes_record_batch_stream(&file_path, stream)
                        .await
                        .map_err(|error| guard.note_failure(error))?;
                Ok(ParsedDeleteFileContext::DelVecs {
                    guard,
                    results: del_vecs,
                })
            }
            // Install the decoded vector under the data file it references, never under the Puffin
            // file's own path. The DeleteFilter hands a scan task its vector by data-file-path
            // lookup, so a Puffin-path key orphans the vector and resurrects every deleted row.
            // `guard` carries the `{path}@{offset}` cache key, so the notify machinery marks the
            // right blob loaded.
            DeleteFileContext::FreshDeletionVector {
                guard,
                referenced_data_file,
                delete_vector,
            } => Ok(ParsedDeleteFileContext::DelVecs {
                guard,
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

    /// Convert one position-delete row's untrusted `pos` value into a bitmap position.
    ///
    /// # Errors
    ///
    /// Fails on a negative position. A `pos as u64` cast wraps it to a huge position that matches no
    /// row, so the delete fails open and the deleted rows resurrect. Java fails loud on the same input
    /// through `RoaringPositionBitmap.validatePosition`. The fork does not mirror Java's upper bound
    /// `MAX_POSITION`, because `RoaringTreemap` supports the full u64 position range.
    ///
    /// # Notes
    ///
    /// The error names both the delete file and the data file, so logs alone identify the corrupt file.
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
            let columns = batch.columns();

            // This reader takes the two spec-required columns positionally: `file_path`, then `pos`. A
            // delete file is untrusted input, so a batch with fewer than two columns fails closed with
            // a typed error naming the file. An index panic here aborts the scan task's process.
            let [file_paths, positions, ..] = columns else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Invalid position delete file '{delete_file_path}': expected at least 2 \
                         columns (file_path, pos), found {}",
                        columns.len()
                    ),
                ));
            };

            let Some(file_paths) = file_paths.as_any().downcast_ref::<StringArray>() else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Could not downcast file paths array to StringArray",
                ));
            };
            let Some(positions) = positions.as_any().downcast_ref::<Int64Array>() else {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Could not downcast positions array to Int64Array",
                ));
            };

            // Position-delete files sort by (path, pos), so equal paths arrive in contiguous runs.
            // `current` caches the last-seen path and its vector, and the code re-resolves the map
            // entry only when the path changes. The resulting map matches the per-row form exactly.
            // An unsorted file still lands every position in the right entry; it only re-resolves more
            // often. The vector splices back into the map on each change and at end-of-batch.
            let mut current: Option<(&str, DeleteVector)> = None;
            for (file_path, pos) in file_paths.iter().zip(positions.iter()) {
                // The spec requires both columns. A null in either is corrupt input. Fail closed with
                // a typed error naming the delete file. Never panic, and never skip the row.
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
                        // Flush the previous run's vector into the map, merging when the path recurs in
                        // a later run. Then start the new run from the positions the map already holds.
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

    /// Parse an equality-delete file's record-batch stream into its survival [`Predicate`]. The
    /// predicate is false for a row the equality delete removes.
    ///
    /// # Notes
    ///
    /// It is `pub(crate)` so `ConvertEqualityDeleteFiles` reuses the exact read-side parse.
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

    /// Like [`parse_equality_deletes_record_batch_stream`], but it also returns the hashed
    /// [`EqDeleteKeySet`] accelerator when every key column type is eligible for the set fast path.
    /// The predicate stays the authoritative oracle and the fallback, so a `None` set means "apply
    /// through the predicate path". The set's tuples and the predicate's leaves come from the same
    /// decoded [`Datum`]s, so they encode the identical delete condition.
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

            // Push every struct's validity down into its fields before the visit. The processor below
            // collects each key column as a standalone array. Arrow does not require a null struct to
            // mask its children, so a key under a null struct would otherwise decode to stale child
            // bytes and produce `= <stale value>` instead of `IS NULL`.
            let root_array: ArrayRef = propagate_struct_validity(
                &(Arc::new(StructArray::from(record_batch)) as ArrayRef),
            )?;

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

            // Process the collected columns in lockstep. When every key column is set-eligible, skip
            // per-row predicate construction during the stream. The tree is built once afterwards,
            // only because the null-batch fallback needs it. When ineligible, collect predicates only.
            #[allow(clippy::len_zero)]
            while datum_columns_with_names[0].0.len() > 0 {
                if set_eligible {
                    let mut tuple: Vec<Option<Datum>> =
                        Vec::with_capacity(datum_columns_with_names.len());
                    for &mut (ref mut column, _, _, _) in &mut datum_columns_with_names {
                        if let Some(item) = column.next() {
                            tuple.push(item?);
                        }
                    }
                    delete_tuples.push(tuple);
                } else {
                    let mut row_predicate = AlwaysTrue;
                    for &mut (ref mut column, _, ref field_name, _) in &mut datum_columns_with_names
                    {
                        if let Some(item) = column.next() {
                            let cell = item?;
                            let cell_predicate = if let Some(datum) = &cell {
                                Reference::new(field_name.clone()).equal_to(datum.clone())
                            } else {
                                Reference::new(field_name.clone()).is_null()
                            };
                            row_predicate = row_predicate.and(cell_predicate);
                        }
                    }
                    row_predicates.push(row_predicate.not().rewrite_not());
                }
            }
        }

        // Build the set accelerator only when every key column was eligible. `try_build` re-checks the
        // gate and returns `None` for an empty or ineligible key schema.
        let key_set = if set_eligible {
            // Rebuild the survival predicate from the same tuples the set encodes. The null-batch
            // fallback needs it.
            if let Some(columns) = key_columns.as_ref() {
                for tuple in &delete_tuples {
                    let mut row_predicate = AlwaysTrue;
                    for (cell, (_, field_name, _)) in tuple.iter().zip(columns.iter()) {
                        let cell_predicate = if let Some(datum) = cell {
                            Reference::new(field_name.clone()).equal_to(datum.clone())
                        } else {
                            Reference::new(field_name.clone()).is_null()
                        };
                        row_predicate = row_predicate.and(cell_predicate);
                    }
                    row_predicates.push(row_predicate.not().rewrite_not());
                }
            }
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
    /// The struct field names currently being descended through, outermost first. A collected key's
    /// [`Reference`] name must be the full dotted path, because `Schema::name_to_id` indexes that form
    /// and [`crate::expr::Bind`] resolves only that form. A leaf-only name either fails to bind, or
    /// binds to the wrong column when a top-level column shares the leaf name.
    field_path: Vec<String>,
}

impl<'a> EqDelColumnProcessor<'a> {
    fn new(equality_ids: &'a HashSet<i32>) -> Self {
        Self {
            equality_ids,
            collected_columns: Vec::with_capacity(equality_ids.len()),
            field_path: Vec::new(),
        }
    }

    /// The full dotted name of `field` given the struct path currently being descended.
    fn full_field_name(&self, field: &NestedFieldRef) -> String {
        if self.field_path.is_empty() {
            field.name.clone()
        } else {
            format!("{}.{}", self.field_path.join("."), field.name)
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

    /// `before_struct_field` and `after_struct_field` bracket the visit of one struct field. Pushing
    /// here and popping there leaves [`Self::field_path`] holding exactly that field's ancestors.
    fn before_struct_field(&mut self, field: &NestedFieldRef, _partner: &ArrayRef) -> Result<()> {
        self.field_path.push(field.name.clone());
        Ok(())
    }

    fn after_struct_field(&mut self, field: &NestedFieldRef, _partner: &ArrayRef) -> Result<()> {
        self.field_path.pop().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "Unbalanced struct-field walk while collecting equality-delete key columns",
            )
            .with_context("field_name", &field.name)
        })?;
        Ok(())
    }

    fn field(&mut self, field: &NestedFieldRef, partner: &ArrayRef, _value: ()) -> Result<()> {
        if self.equality_ids.contains(&field.id) && field.field_type.as_primitive_type().is_some() {
            self.collected_columns.push((
                partner.clone(),
                field.id,
                self.full_field_name(field),
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
    use crate::expr::Bind;
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

        // `sa` lives inside the struct column `s`, so its reference is `s.sa`. That is the form
        // `Schema::name_to_id` indexes, and the only one `Bind` resolves.
        let expected = "(((((y != 1) OR (z != 100)) OR (a != \"HELP\")) OR (s.sa != 4)) OR (b != 62696E6172795F64617461)) AND (((((y != 2) OR (z IS NOT NULL)) OR (a IS NOT NULL)) OR (s.sa != 5)) OR (b IS NOT NULL))".to_string();

        assert_eq!(parsed_eq_delete.to_string(), expected);
    }

    /// Build an in-memory equality-delete batch whose key column is nested. Row 0's `nested` struct is
    /// null while the `k` slot underneath still holds a live `7`, because Arrow does not require a null
    /// struct to mask its children. Row 1 is fully live with `k = 9`.
    fn nested_key_equality_delete_batch() -> RecordBatch {
        let k_field = Arc::new(simple_field("k", DataType::Int32, true, "3"));
        let k_values = Arc::new(Int32Array::from(vec![Some(7), Some(9)])) as ArrayRef;
        let nested = StructArray::try_new(
            Fields::from(vec![k_field]),
            vec![k_values],
            Some(arrow_buffer::NullBuffer::from(vec![false, true])),
        )
        .expect("nested struct array");
        let nested_field = simple_field("nested", nested.data_type().clone(), true, "2");
        let id_field = simple_field("id", DataType::Int32, false, "1");
        let schema = Arc::new(arrow_schema::Schema::new(vec![id_field, nested_field]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(nested) as ArrayRef,
        ])
        .expect("nested-key equality delete batch")
    }

    /// The iceberg table schema `nested_key_equality_delete_batch` is a delete against.
    fn nested_key_table_schema() -> Schema {
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "nested",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "k", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .expect("nested key table schema")
    }

    /// A nested equality-delete key must decode as the value the row logically holds. Row 0's `nested`
    /// struct is null, so its key is null, and a null key matches only other nulls. Handing the `k`
    /// child back detached from its parent turns that into an equality against the stale `7` in the
    /// child buffer. The delete then removes rows holding 7 and misses the rows it was written for.
    /// This also pins the reference name: a leaf-only `k` cannot bind.
    #[tokio::test]
    async fn test_nested_equality_delete_key_uses_parent_validity_and_full_name() {
        use futures::stream;

        let batch = nested_key_equality_delete_batch();
        let stream: ArrowRecordBatchStream = stream::iter(vec![Ok(batch)]).boxed();

        let predicate = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            stream,
            HashSet::from_iter(vec![3]),
        )
        .await
        .expect("nested-key equality deletes must parse");

        assert_eq!(
            predicate.to_string(),
            "(nested.k IS NOT NULL) AND (nested.k != 9)",
            "row 0's key is NULL (its parent struct is NULL), and the reference must be the full \
             dotted path"
        );
    }

    /// The name half of the test above. `DeleteFilter::build_equality_delete_predicate` binds the
    /// parsed predicate against the table schema. A leaf-only reference either fails to bind, or binds
    /// to the wrong column when a top-level column shares the leaf name, and deletes wrong rows.
    #[tokio::test]
    async fn test_nested_equality_delete_predicate_binds_to_table_schema() {
        use futures::stream;

        let batch = nested_key_equality_delete_batch();
        let stream: ArrowRecordBatchStream = stream::iter(vec![Ok(batch)]).boxed();

        let predicate = CachingDeleteFileLoader::parse_equality_deletes_record_batch_stream(
            stream,
            HashSet::from_iter(vec![3]),
        )
        .await
        .expect("nested-key equality deletes must parse");

        let bound = predicate
            .bind(Arc::new(nested_key_table_schema()), true)
            .expect("the nested-key predicate must bind against the table schema");
        assert!(
            bound.to_string().contains("nested.k"),
            "bound predicate must reference the nested column: {bound}"
        );
    }

    /// An equality-delete task with `equality_ids: None` must surface a typed `DataInvalid` error
    /// naming the delete file, not panic the scan on `Option::unwrap`. The fixture uses a real
    /// eq-delete parquet file, so the missing equality_ids is the only defect.
    ///
    /// Mutation: restore `task.equality_ids.clone().unwrap()` and the load task panics, dropping the
    /// result sender. The loader channel then yields `RecvError` and the `.expect` below fails red.
    #[tokio::test]
    async fn test_eq_delete_missing_equality_ids_yields_typed_error_not_panic() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let table_location = tmp_dir.path().as_os_str().to_str().expect("utf-8 path");
        let file_io = FileIO::new_with_fs();

        let eq_delete_file_path = setup_write_equality_delete_file_1(table_location);

        let task = FileScanTaskDeleteFile {
            file_path: eq_delete_file_path.clone(),
            file_size_in_bytes: std::fs::metadata(&eq_delete_file_path)
                .map(|m| m.len())
                .unwrap_or(0),
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: None, // the corrupt input under test
            file_format: crate::spec::DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        };

        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let inner = loader
            .load_deletes(
                &[task],
                Arc::new(Schema::builder().build().expect("empty schema")),
            )
            .await
            .expect("loader channel must deliver a result, not panic the load task");
        let error = inner
            .expect_err("equality_ids: None must surface a typed error, not load successfully");

        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error.to_string().contains(&eq_delete_file_path),
            "error must name the delete file: {error}"
        );
        assert!(
            error.to_string().contains("equality_ids"),
            "error must name the missing equality_ids: {error}"
        );
    }

    /// The run-cached parse must build the same `HashMap<path, DeleteVector>` as a per-row reference.
    /// The fixture covers the edge cases the run cache must get right: contiguous runs of one path,
    /// several positions for one path, a path that recurs in a later non-contiguous run, and positions
    /// split across two batches. A `DeleteVector` is a set, so duplicate positions collapse.
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

        // Batch 1 is a, b, a, b, a. Both paths recur in non-contiguous runs, so the second flush of
        // each merges onto a non-empty map entry. It also includes a duplicate position for "a".
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

    /// Write a real positional-delete parquet file with the spec's `file_path` and `pos` columns.
    /// `pos` is nullable here so corrupt fixtures are expressible. The spec requires the column, so a
    /// conforming writer never emits a null there.
    fn write_pos_del_parquet(
        dir: &std::path::Path,
        file_name: &str,
        rows: &[(&str, Option<i64>)],
    ) -> String {
        write_pos_del_parquet_maybe_row(dir, file_name, rows, false)
    }

    /// Like [`write_pos_del_parquet`], optionally with the third optional `row` column Java may emit.
    fn write_pos_del_parquet_maybe_row(
        dir: &std::path::Path,
        file_name: &str,
        rows: &[(&str, Option<i64>)],
        with_row_column: bool,
    ) -> String {
        let mut fields = vec![
            simple_field("file_path", DataType::Utf8, false, "2147483546"),
            simple_field("pos", DataType::Int64, true, "2147483545"),
        ];
        if with_row_column {
            fields.push(simple_field("row", DataType::Utf8, true, "999"));
        }
        let schema = Arc::new(arrow_schema::Schema::new(fields));
        let paths: Vec<&str> = rows.iter().map(|(path, _)| *path).collect();
        let positions: Vec<Option<i64>> = rows.iter().map(|(_, pos)| *pos).collect();
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(Int64Array::from(positions)) as ArrayRef,
        ];
        if with_row_column {
            let rows_payload: Vec<Option<&str>> = rows.iter().map(|_| Some("payload")).collect();
            columns.push(Arc::new(StringArray::from(rows_payload)) as ArrayRef);
        }
        let batch =
            RecordBatch::try_new(schema.clone(), columns).expect("build positional-delete batch");

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

    /// The `load_deletes` path projects path and pos. It must still install correct positions when the
    /// file carries the optional third `row` column. This pins the loader, projection, and parse seam.
    #[tokio::test]
    async fn test_load_deletes_pos_delete_with_row_column_projects_and_applies() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let file_io = FileIO::new_with_fs();
        let data_file = format!("{}/data-1.parquet", tmp_dir.path().display());
        let pos_del_path = write_pos_del_parquet_maybe_row(
            tmp_dir.path(),
            "pos-with-row.parquet",
            &[
                (&data_file, Some(0)),
                (&data_file, Some(3)),
                (&data_file, Some(7)),
            ],
            true, // include optional `row` column
        );

        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let delete_filter = loader
            .load_deletes(
                &[parquet_pos_del_task(&pos_del_path)],
                Arc::new(Schema::builder().build().expect("empty schema")),
            )
            .await
            .expect("loader channel")
            .expect("load pos-delete with row column");

        // Build a minimal task pointing at data_file so get_delete_vector resolves the path.
        let task = crate::scan::FileScanTask {
            file_size_in_bytes: 0,
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: Arc::from(data_file.clone()),
            data_file_format: crate::spec::DataFileFormat::Parquet,
            schema: Arc::new(Schema::builder().build().expect("schema")),
            project_field_ids: Arc::from(vec![]),
            predicate: None,
            deletes: Arc::from(vec![parquet_pos_del_task(&pos_del_path)]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };
        let vector = delete_filter
            .get_delete_vector(&task)
            .expect("delete vector for data file");
        assert!(vector.contains(0), "pos 0 must be deleted");
        assert!(vector.contains(3), "pos 3 must be deleted");
        assert!(vector.contains(7), "pos 7 must be deleted");
        assert!(!vector.contains(1), "pos 1 must NOT be deleted");
        assert_eq!(
            vector.len(),
            3,
            "exactly three positions from the projected file"
        );
    }

    /// A negative position must fail closed with a typed `DataInvalid` error naming the delete file and
    /// the position. A `pos as u64` cast wraps -1 to `u64::MAX`, which matches no row, so the delete
    /// fails open and the deleted row resurrects. Java's `RoaringPositionBitmap.validatePosition` also
    /// throws below zero.
    ///
    /// Mutation: the negative row is the second row of a same-path run, so restoring `pos as u64` at
    /// the run-continuation site alone turns this test red through a successful load.
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

    /// The same fail-closed bar when the negative position is the first row of a path's run, which the
    /// new-path branch converts. Restoring `pos as u64` at that site alone turns this test red.
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

    /// A null position reaching the loader must surface as a typed `DataInvalid` error naming the
    /// delete file. It must never panic, and never skip the row. The spec requires the `pos` column,
    /// and Java's `Deletes.toPositionIndexes` also fails loud on a null there.
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

    /// A position-delete batch with fewer than two columns fails closed with a typed error naming the
    /// delete file, instead of panicking on `columns[0]` or `columns[1]`.
    ///
    /// Both arities run separately, because they are two distinct unchecked indexes. A one-column batch
    /// reaches only `columns[1]`, and a zero-column batch reaches only `columns[0]`. A delete file is
    /// untrusted input, and an index panic in a scan task aborts the engine process.
    #[tokio::test]
    async fn test_short_column_arity_yields_typed_error_not_panic() {
        use arrow_array::RecordBatchOptions;
        use futures::stream;

        // One column: `columns[1]` is the out-of-bounds index.
        let one_col_schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "file_path",
            DataType::Utf8,
            false,
        )]));
        let one_col = RecordBatch::try_new(one_col_schema, vec![Arc::new(StringArray::from(vec![
            "data-1.parquet",
        ])) as ArrayRef])
        .expect("build one-column batch");

        // Zero columns: `columns[0]` is the out-of-bounds index. A row count is required
        // because it cannot be inferred without columns.
        let no_col = RecordBatch::try_new_with_options(
            Arc::new(arrow_schema::Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .expect("build zero-column batch");

        for (batch, arity) in [(one_col, 1usize), (no_col, 0usize)] {
            let stream = Box::pin(stream::iter(vec![Ok(batch)])) as ArrowRecordBatchStream;
            let error = CachingDeleteFileLoader::parse_positional_deletes_record_batch_stream(
                "truncated-pos-dels.parquet",
                stream,
            )
            .await
            .expect_err("a short column arity must fail closed with a typed error");

            assert_eq!(error.kind(), ErrorKind::DataInvalid);
            assert!(
                error.to_string().contains("truncated-pos-dels.parquet"),
                "error must name the delete file (arity {arity}): {error}"
            );
            assert!(
                error.to_string().contains("column"),
                "error must say the column arity is wrong (arity {arity}): {error}"
            );
        }
    }

    /// The over-firing control for the arity guard. A batch with more than two columns must still parse
    /// from the first two.
    ///
    /// The spec's position-delete schema has an optional third `row` column, which Java writes when the
    /// writer keeps the deleted row. A guard demanding exactly two columns rejects those files, so the
    /// `..` in the arity pattern is load-bearing and this test holds it.
    #[tokio::test]
    async fn test_position_delete_batch_with_a_trailing_row_column_still_parses() {
        use futures::stream;

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("file_path", DataType::Utf8, false),
            Field::new("pos", DataType::Int64, false),
            Field::new("row", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["data-1.parquet", "data-1.parquet"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![3i64, 9i64])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("deleted-a"),
                Some("deleted-b"),
            ])) as ArrayRef,
        ])
        .expect("build three-column position delete batch");
        let stream = Box::pin(stream::iter(vec![Ok(batch)])) as ArrowRecordBatchStream;

        let result = CachingDeleteFileLoader::parse_positional_deletes_record_batch_stream(
            "pos-dels-with-row.parquet",
            stream,
        )
        .await
        .expect("a delete file carrying the optional `row` column must still parse");

        let deletes = result
            .get("data-1.parquet")
            .expect("the data file must be present in the parsed deletes");
        assert!(deletes.contains(3), "position 3 must be deleted");
        assert!(deletes.contains(9), "position 9 must be deleted");
        assert!(!deletes.contains(4), "position 4 must NOT be deleted");
    }

    /// A null `file_path` row fails closed with a typed error naming the delete file. Replacing the
    /// guard with an unwrap panics this test. The fixture is an in-memory batch, because the parquet
    /// fixture writer declares `file_path` non-nullable.
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

    /// The happy-path control for the fail-closed guards. The same fixture shape with valid positions,
    /// including the boundary `pos = 0`, must load and apply correctly. An over-broadened guard that
    /// rejects `pos <= 0` turns this test red. The negative tests alone cannot catch that.
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
        let tasks = [parquet_pos_del_task(&pos_del_path)];
        let delete_filter = loader
            .load_deletes(
                &tasks,
                Arc::new(Schema::builder().build().expect("empty schema")),
            )
            .await
            .expect("loader channel")
            .expect("valid positions (including the 0 boundary) must load cleanly");

        let vector = delete_filter
            .resolve_delete_vector(&tasks, &data_file)
            .expect("delete vector installed under the data file");
        let positions: Vec<u64> = vector.iter().collect();
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
        assert_eq!(result.len(), 12);

        let result = delete_filter.get_delete_vector(&file_scan_tasks[1]);
        assert!(result.is_none()); // no pos dels for file 3
    }

    /// `evolve_schema` must evolve only the equality_ids columns, not every table column. Per the
    /// [Iceberg spec](https://iceberg.apache.org/spec/#equality-delete-files), an equality delete file
    /// may hold a subset of the columns.
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

    /// Load a FileScanTask carrying both positional and equality deletes.
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
            data_file_path: Arc::from(format!(
                "{}/data-1.parquet",
                table_location.to_str().unwrap()
            )),
            data_file_format: DataFileFormat::Parquet,
            schema: data_file_schema.clone(),
            project_field_ids: Arc::from(vec![2, 3]),
            predicate: None,
            deletes: Arc::from(vec![pos_del, eq_del]),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        };

        // Load the deletes - should handle both types without error
        let delete_file_loader = CachingDeleteFileLoader::new(file_io.clone(), 10);
        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_task.deletes, file_scan_task.schema_ref())
            .await
            .unwrap()
            .unwrap();

        // This call is bounded on purpose. It can reach the eq-delete wait path, so a lost-wakeup
        // regression would hang the test binary forever instead of failing it.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            delete_filter.build_equality_delete_predicate(&file_scan_task),
        )
        .await
        .expect("build_equality_delete_predicate must not hang");
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

    /// Write a real Puffin file holding one `deletion-vector-v1` blob for `referenced_data_file`, and
    /// return the blob coordinates read back from the footer. A manifest's `DeleteFile` carries the
    /// same coordinates.
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

        // Read the blob coordinates back from the footer. `BaseDVFileWriter` records the same values.
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

    /// Write a real Puffin file holding one `deletion-vector-v1` blob per input pair, in order, and
    /// return the blob coordinates read back from the footer.
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

    /// The loader dispatch. A position delete in PUFFIN format must reach the blob decoder, not
    /// `parquet_to_batch_stream`, which fails on Puffin bytes. The decoded positions must land under
    /// the referenced data file and nowhere else, neither the Puffin path nor a sibling data file.
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
        let tasks = [dv_task(&puffin_path, &data_file_a, offset, length, 2)];
        let delete_filter = loader
            .load_deletes(&tasks, Arc::new(Schema::builder().build().unwrap()))
            .await
            .unwrap()
            .expect("DV load must succeed (parquet routing would fail here)");

        let vector = delete_filter
            .resolve_delete_vector(&tasks, &data_file_a)
            .expect("vector must be keyed by the referenced data file");
        let positions: Vec<u64> = vector.iter().collect();
        assert_eq!(positions, vec![1, 3]);

        assert!(
            delete_filter
                .resolve_delete_vector(&tasks, &puffin_path)
                .is_none(),
            "the vector must NOT be keyed by the Puffin file's own path"
        );
        assert!(
            delete_filter
                .resolve_delete_vector(&tasks, &data_file_b)
                .is_none(),
            "a DV for data file A must not leak onto sibling data file B"
        );
    }

    /// Cache-hit semantics. Loading one blob twice through one loader reuses the first decoded vector
    /// under the `{path}@{offset}` dedup, instead of decoding and unioning a second copy.
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

        let vector_1 = filter_1
            .resolve_delete_vector(&tasks, &data_file_a)
            .unwrap();
        let vector_2 = filter_2
            .resolve_delete_vector(&tasks, &data_file_a)
            .unwrap();
        assert!(
            Arc::ptr_eq(&vector_1, &vector_2),
            "the second load must reuse the cached vector"
        );
        assert_eq!(
            vector_1.len(),
            3,
            "re-loading must not union a second copy into the vector"
        );
        // After the first load the blob's claim key is terminally `Loaded`, so a re-load can only take
        // the `AlreadyLoaded` arm. A second decode needs a fresh `Load` claim, which this key no
        // longer offers.
        assert!(
            matches!(
                loader
                    .delete_filter
                    .try_start_pos_del_load(&format!("{puffin_path}@{offset}"))
                    .expect("a loaded blob must not error at claim time"),
                PosDelLoadAction::AlreadyLoaded
            ),
            "the second load must observe the first load's terminal state, not re-claim the blob"
        );
    }

    /// Two deletion vectors in one Puffin file, at different offsets and referencing different data
    /// files. This is the case the `{path}@{offset}` cache key exists for. A bare-file-path key marks
    /// blob 2 already loaded when blob 1 finishes, which drops its vector and resurrects its rows.
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
        let tasks = [
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
        ];
        let delete_filter = loader
            .load_deletes(&tasks, Arc::new(Schema::builder().build().unwrap()))
            .await
            .unwrap()
            .expect("both DV blobs in one Puffin file must load");

        let vector_a = delete_filter
            .resolve_delete_vector(&tasks, &data_file_a)
            .expect("blob 1 must land under data file A");
        let positions_a: Vec<u64> = vector_a.iter().collect();
        assert_eq!(positions_a, vec![1, 3]);

        let vector_b = delete_filter
            .resolve_delete_vector(&tasks, &data_file_b)
            .expect("blob 2 must land under data file B (not be marked already-loaded)");
        let positions_b: Vec<u64> = vector_b.iter().collect();
        assert_eq!(positions_b, vec![0, 2, 4]);
    }

    /// Two deletion vectors claiming one data file is an invalid table state. Java rejects it at
    /// index-build time, and the fork rejects it at the load door. Unioning them would over-delete, and
    /// keeping one would resurrect rows.
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

    /// The metadata validations at the load door, mirroring Java's `BaseDeleteLoader.validateDV`. A
    /// missing offset, an out-of-range size, and a missing referenced data file each reject by name.
    /// None of them panics, and none falls through to the parquet reader.
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

    /// The manifest's record_count is the vector's cardinality. A decoded bitmap that disagrees means
    /// the manifest and the blob diverge. Silent acceptance would hide the corruption.
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
            data_file_path: Arc::from(data_file_path),
            data_file_format: crate::spec::DataFileFormat::Parquet,
            schema,
            project_field_ids: Arc::from(vec![1]),
            predicate: None,
            deletes: Arc::from(deletes),
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: false,
            split_offsets: None,
            first_row_id: None,
            file_sequence_number: None,
        }
    }

    /// A deletion vector applied during a real Arrow read. The rows at the vector's positions are
    /// absent from the file it references, and a sibling file in the same scan is untouched. This
    /// proves the decoded vector reaches the reader's row selection.
    ///
    /// Mutation: key the vector by the Puffin file path and the deleted rows resurrect here.
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

    /// A delete file loaded for one task must not delete rows from another task's data file.
    ///
    /// Tasks A and B share one loader. Task B alone lists `foreign.parquet`, yet its rows name data
    /// file A. Task A has one delete of its own, so it consults the loader's shared state. Loading task
    /// B first lands `foreign`'s positions in that state before task A resolves.
    ///
    /// Java builds one `DeleteFilter` per task over `task.deletes()` only, so `foreign` can never reach
    /// task A. Merging every parsed row into one shared data-file-keyed map deletes id 30 wrongly.
    ///
    /// Mutation: union all installed contributions in `resolve_delete_vector` regardless of the task's
    /// delete list, and this test reads `[1, 2]` instead of `[1]`.
    #[tokio::test]
    async fn test_cross_task_pos_delete_does_not_leak_into_other_tasks_files() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let file_io = FileIO::new_with_fs();
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));

        let data_file_a = write_data_parquet(tmp_dir.path(), "data-a.parquet", &[10, 20, 30]);
        let data_file_b = write_data_parquet(tmp_dir.path(), "data-b.parquet", &[40, 50, 60]);

        // own-a deletes A's position 1 (id 20); own-b deletes B's position 1 (id 50); foreign —
        // listed by task B ONLY — names A's position 2 (id 30).
        let own_a =
            write_pos_del_parquet(tmp_dir.path(), "own-a.parquet", &[(&data_file_a, Some(1))]);
        let own_b =
            write_pos_del_parquet(tmp_dir.path(), "own-b.parquet", &[(&data_file_b, Some(1))]);
        let foreign = write_pos_del_parquet(tmp_dir.path(), "foreign.parquet", &[(
            &data_file_a,
            Some(2),
        )]);

        let task_a = data_scan_task(&data_file_a, schema.clone(), vec![parquet_pos_del_task(
            &own_a,
        )]);
        let task_b = data_scan_task(&data_file_b, schema.clone(), vec![
            parquet_pos_del_task(&own_b),
            parquet_pos_del_task(&foreign),
        ]);

        let loader = CachingDeleteFileLoader::new(file_io, 10);
        // Task B first: `foreign`'s parse is in the loader's shared state BEFORE task A resolves.
        let filter_b = loader
            .load_deletes(&task_b.deletes, schema.clone())
            .await
            .expect("loader channel for task B")
            .expect("task B's deletes must load");
        let filter_a = loader
            .load_deletes(&task_a.deletes, schema)
            .await
            .expect("loader channel for task A")
            .expect("task A's deletes must load");

        let vector_a = filter_a
            .get_delete_vector(&task_a)
            .expect("task A has a delete of its own");
        let positions_a: Vec<u64> = vector_a.iter().collect();
        assert_eq!(
            positions_a,
            vec![1],
            "task A's vector must hold ONLY its own delete's position — `foreign` (loaded for \
             task B, naming file A) must not leak position 2 (id 30) into task A"
        );

        let vector_b = filter_b
            .get_delete_vector(&task_b)
            .expect("task B has a delete of its own");
        let positions_b: Vec<u64> = vector_b.iter().collect();
        assert_eq!(
            positions_b,
            vec![1],
            "task B's vector must hold only positions its OWN deletes name for file B — \
             `foreign`'s rows name file A and contribute nothing to B"
        );
    }

    /// The legitimate-share control for per-task scoping. One delete file listed by two tasks must
    /// still apply in both, and each task must receive only the positions the file names for its own
    /// data file. Scoping must not drop a file that is in both lists, and must not broadcast one file's
    /// positions onto the other.
    ///
    /// It also pins that the parse cache survives the scoping. The second task's load observes the
    /// first's terminal `Loaded` claim, mirroring Java's per-delete-file cache in
    /// `BaseDeleteLoader.getOrReadPosDeletes`.
    ///
    /// Mutation: restrict a source's contribution to tasks whose data file is the only file it names,
    /// or skip sources listed by more than one task. Either turns this red.
    #[tokio::test]
    async fn test_pos_delete_shared_by_two_tasks_applies_in_both() {
        let tmp_dir = TempDir::new().expect("tempdir");
        let file_io = FileIO::new_with_fs();
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));

        let data_file_a = write_data_parquet(tmp_dir.path(), "data-a.parquet", &[10, 20, 30]);
        let data_file_b = write_data_parquet(tmp_dir.path(), "data-b.parquet", &[40, 50, 60]);

        // ONE delete file naming rows in BOTH data files: (A, 0) and (B, 2).
        let shared = write_pos_del_parquet(tmp_dir.path(), "shared.parquet", &[
            (&data_file_a, Some(0)),
            (&data_file_b, Some(2)),
        ]);

        let task_a = data_scan_task(&data_file_a, schema.clone(), vec![parquet_pos_del_task(
            &shared,
        )]);
        let task_b = data_scan_task(&data_file_b, schema.clone(), vec![parquet_pos_del_task(
            &shared,
        )]);

        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let filter_a = loader
            .load_deletes(&task_a.deletes, schema.clone())
            .await
            .expect("loader channel for task A")
            .expect("task A's deletes must load");
        let filter_b = loader
            .load_deletes(&task_b.deletes, schema)
            .await
            .expect("loader channel for task B")
            .expect("task B's deletes must load");

        // Parse-once: after task A's load, the shared file's claim is terminally `Loaded`; task
        // B's load can only have taken the `AlreadyLoaded`/`WaitFor` arm, never a second parse.
        assert!(
            matches!(
                loader
                    .delete_filter
                    .try_start_pos_del_load(&shared)
                    .expect("a loaded delete file must not error at claim time"),
                PosDelLoadAction::AlreadyLoaded
            ),
            "the shared delete file must be parsed once and reused, not re-claimed per task"
        );

        let positions_a: Vec<u64> = filter_a
            .get_delete_vector(&task_a)
            .expect("the shared delete names a row of file A")
            .iter()
            .collect();
        assert_eq!(
            positions_a,
            vec![0],
            "task A must receive the shared delete's position for file A"
        );

        let positions_b: Vec<u64> = filter_b
            .get_delete_vector(&task_b)
            .expect("the shared delete names a row of file B")
            .iter()
            .collect();
        assert_eq!(
            positions_b,
            vec![2],
            "task B must receive the shared delete's position for file B — per-task scoping must \
             not drop a file legitimately listed by two tasks"
        );
    }

    /// The positional-delete lost-wakeup class. A delete file whose load dies after claiming it must
    /// not strand the next `load_deletes` call on the same loader. Without the guard the failed claim
    /// stays `Loading` forever, and the second call parks on a notifier that can never fire. The scan
    /// then hangs with no error, no timeout, and no log line.
    ///
    /// Mutation: disarm the guard before it drops, so nothing publishes `Failed`. The second load hangs
    /// and the timeout below fires.
    #[tokio::test]
    async fn test_failed_pos_del_load_does_not_strand_the_next_load() {
        let tmp_dir = TempDir::new().unwrap();
        let missing = format!("{}/missing-pos-del.parquet", tmp_dir.path().display());
        let file_io = FileIO::new_with_fs();
        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let schema = Arc::new(Schema::builder().build().unwrap());
        let tasks = [FileScanTaskDeleteFile {
            file_path: missing.clone(),
            file_size_in_bytes: 1,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Parquet,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
            record_count: None,
        }];

        let first = loader
            .load_deletes(&tasks, schema.clone())
            .await
            .expect("the first load must deliver a result")
            .expect_err("an unreadable positional delete file must error");

        let second = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            loader.load_deletes(&tasks, schema),
        )
        .await
        .expect("a load after a failed load must not hang")
        .expect("the second load must deliver a result");

        let error = second.expect_err(
            "the file whose load died must error again, never silently resolve with no deletes",
        );
        assert!(
            error.to_string().contains(&missing),
            "the error must name the delete file, got: {error}"
        );
        // The claiming task records its own error into the terminal state, so this later caller learns
        // why the load died, not only that it died.
        assert!(
            error.to_string().contains(&first.to_string()),
            "the later caller's error must carry the original cause '{first}', got: {error}"
        );
    }

    /// The deletion-vector half of the same class. A blob whose load dies after claiming it must
    /// publish the terminal `PosDelState::Failed` under the `{puffin path}@{offset}` claim key. The
    /// next `load_deletes` then errors instead of parking on a notifier that can never fire.
    ///
    /// The composite key matters here. A guard that published under the bare file path would leave the
    /// real entry `Loading`.
    ///
    /// Mutation: leak the claim on the read-failure path, or disarm `PosDelLoadGuard::drop`. Either
    /// makes the second load hang and the timeout below fire.
    #[tokio::test]
    async fn test_failed_dv_load_does_not_strand_the_next_load() {
        let tmp_dir = TempDir::new().expect("temp dir for the DV fixture");
        let missing_puffin = format!("{}/missing.puffin", tmp_dir.path().display());
        let data_file = format!("{}/data-a.parquet", tmp_dir.path().display());
        let file_io = FileIO::new_with_fs();
        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));
        let tasks = [FileScanTaskDeleteFile {
            file_path: missing_puffin.clone(),
            file_size_in_bytes: 64,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
            file_format: DataFileFormat::Puffin,
            referenced_data_file: Some(data_file),
            content_offset: Some(4),
            content_size_in_bytes: Some(16),
            record_count: Some(2),
        }];

        let first = loader
            .load_deletes(&tasks, schema.clone())
            .await
            .expect("the first DV load must deliver a result")
            .expect_err("an unreadable Puffin DV must error");

        let second = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            loader.load_deletes(&tasks, schema),
        )
        .await
        .expect("a DV load after a failed DV load must not hang")
        .expect("the second DV load must deliver a result");

        let error =
            second.expect_err("a terminally failed DV must error, never resolve with no deletes");
        assert!(
            error.to_string().contains(&missing_puffin),
            "the error must name the Puffin file, got: {error}"
        );
        assert!(
            error.to_string().contains("@4"),
            "the error must name the blob claim key, got: {error}"
        );
        // As on the parquet path, the terminal state records the blob-read error, so the later caller
        // sees the cause.
        assert!(
            error.to_string().contains(&first.to_string()),
            "the later caller's error must carry the original cause '{first}', got: {error}"
        );
    }

    /// The `WaitFor` arm must consult the same `{puffin path}@{offset}` key it claimed under. Two
    /// concurrent `load_deletes` calls drive the real wait path. One waits, and both must end on the
    /// one shared vector rather than a spurious error.
    ///
    /// This test reads both loads after they join, so it cannot see when the waiter returned.
    /// `test_dv_waiter_does_not_return_before_the_vector_is_installed` pins that ordering instead.
    ///
    /// Mutation: look the state up under `&task.file_path` instead of `&cache_key`. The waiter then
    /// misses the terminal state and fails a healthy scan with the defensive error.
    #[tokio::test]
    async fn test_concurrent_dv_loads_share_one_claim() {
        let tmp_dir = TempDir::new().expect("temp dir for the DV fixture");
        let file_io = FileIO::new_with_fs();
        let data_file = format!("{}/data-a.parquet", tmp_dir.path().display());
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file, &[
                1, 3,
            ])
            .await;
        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let tasks = [dv_task(&puffin_path, &data_file, offset, length, 2)];
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));

        let a = loader.load_deletes(&tasks, schema.clone());
        let b = loader.load_deletes(&tasks, schema);
        let (ra, rb) = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            futures::future::join(a, b),
        )
        .await
        .expect("concurrent DV loads of one blob must not deadlock");

        let filter_a = ra
            .expect("load A must deliver a result")
            .expect("load A must succeed");
        let filter_b = rb
            .expect("load B must deliver a result")
            .expect("load B must succeed");
        let vector_a = filter_a
            .resolve_delete_vector(&tasks, &data_file)
            .expect("A must see the vector");
        let vector_b = filter_b
            .resolve_delete_vector(&tasks, &data_file)
            .expect("B must see the vector");
        assert!(
            Arc::ptr_eq(&vector_a, &vector_b),
            "both loads must observe the one shared delete vector"
        );
        assert_eq!(
            vector_a.len(),
            2,
            "the one shared vector must hold both deleted positions"
        );
    }

    /// The terminal `PosDelState::Failed` must carry the cause on every failure path that has one.
    /// Only the claiming task sees its own error. Every later consumer reads the state, so without the
    /// recorded cause an operator learns that a delete load died but never why.
    ///
    /// Three claim-then-die paths run here, each on its own loader: the cardinality check, the blob
    /// decoder, and the position-delete parser. The delete-file open and the ranged blob read are
    /// pinned by the two "does not strand the next load" tests.
    ///
    /// Mutation: drop any one of the three `note_failure` calls. That case's second load then reports
    /// only the generic reason, and its assertion fails.
    #[tokio::test]
    async fn test_failed_pos_del_load_reports_the_cause_to_later_callers() {
        let tmp_dir = TempDir::new().expect("temp dir for the fixtures");
        let file_io = FileIO::new_with_fs();
        let data_file = format!("{}/data-a.parquet", tmp_dir.path().display());
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file, &[
                1, 3,
            ])
            .await;
        let neg_pos_del_path = write_pos_del_parquet(tmp_dir.path(), "neg-pos.parquet", &[
            (&data_file, Some(0)),
            (&data_file, Some(-1)),
        ]);
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));

        // (a) the cardinality check: record_count claims 5 positions, the blob holds 2.
        // (b) the blob decoder: the claimed range is shifted one byte, so the bytes read back are
        //     not a `deletion-vector-v1` blob.
        // (c) the position-delete parser: the second row carries a negative position.
        let cases: [(&str, FileScanTaskDeleteFile, &str); 3] = [
            (
                "the DV cardinality check",
                dv_task(&puffin_path, &data_file, offset, length, 5),
                "cardinality",
            ),
            (
                "the DV blob decoder",
                dv_task(&puffin_path, &data_file, offset + 1, length, 2),
                "Invalid deletion vector",
            ),
            (
                "the position-delete parser",
                parquet_pos_del_task(&neg_pos_del_path),
                "negative position -1",
            ),
        ];

        for (path_under_test, task, expected_cause) in cases {
            let loader = CachingDeleteFileLoader::new(file_io.clone(), 10);
            let delete_file_path = task.file_path.clone();
            let tasks = [task];

            let first = loader
                .load_deletes(&tasks, schema.clone())
                .await
                .expect("the first load must deliver a result")
                .expect_err("the fixture must make the claiming task's load fail");
            assert!(
                first.to_string().contains(expected_cause),
                "{path_under_test}: the claiming task must see its own error, got: {first}"
            );

            let second = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                loader.load_deletes(&tasks, schema.clone()),
            )
            .await
            .expect("a load after a failed load must not hang")
            .expect("the second load must deliver a result")
            .expect_err("a terminally failed delete file must error for later callers too");

            assert_eq!(second.kind(), ErrorKind::Unexpected);
            assert!(
                second.to_string().contains(expected_cause),
                "{path_under_test}: the later caller's error must carry the ORIGINAL cause, got: \
                 {second}"
            );
            assert!(
                second.to_string().contains(&delete_file_path),
                "{path_under_test}: the later caller's error must still name the delete file, \
                 got: {second}"
            );
        }
    }

    /// Claim `key` on the loader's own delete filter, so its next load of that file takes the `WaitFor`
    /// arm. Publish the returned guard to release the waiter, or drop it to kill the claim.
    ///
    /// Driving the wait this way makes the two properties below observable. A race can only be read
    /// after both loads finish, by which point the claimant has published either way.
    fn claim_pos_del(loader: &CachingDeleteFileLoader, key: &str) -> PosDelLoadGuard {
        match loader
            .delete_filter
            .try_start_pos_del_load(key)
            .expect("a fresh claim on an unknown key must not error")
        {
            PosDelLoadAction::Load(guard) => guard,
            other => panic!("expected a fresh claim for '{key}', got {other:?}"),
        }
    }

    /// Assert that an in-flight `load_deletes` is parked on the claim above, and hand it back pending.
    async fn assert_parked_on_claim(waiting: &mut Receiver<Result<DeleteFilter>>, why: &str) {
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), waiting)
                .await
                .is_err(),
            "{why}"
        );
    }

    /// The ordering half of the deletion-vector wait contract. A `load_deletes` that finds the blob
    /// already claimed must not return until the claiming task installs the vector.
    /// `DeleteFilter::get_delete_vector` is synchronous, so an early return hands the reader an absent
    /// vector, and every row that vector deletes resurrects. That is silent under-delete, not a hang.
    ///
    /// Mutation: drop the `wait_for_pos_del_load` await on this path. The load then resolves at once
    /// with no delete vector.
    #[tokio::test]
    async fn test_dv_waiter_does_not_return_before_the_vector_is_installed() {
        let tmp_dir = TempDir::new().expect("temp dir for the DV fixture");
        let file_io = FileIO::new_with_fs();
        let data_file = format!("{}/data-a.parquet", tmp_dir.path().display());
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file, &[
                1, 3,
            ])
            .await;
        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let tasks = [dv_task(&puffin_path, &data_file, offset, length, 2)];
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));

        let cache_key = format!("{puffin_path}@{offset}");
        let guard = claim_pos_del(&loader, &cache_key);

        let mut waiting = loader.load_deletes(&tasks, schema);
        assert_parked_on_claim(
            &mut waiting,
            "the waiting load must still be pending while the claim is unpublished — returning \
             here hands the reader an absent delete vector and resurrects every deleted row",
        )
        .await;

        // Publish exactly as the production loader does: the contribution first, then the
        // terminal state.
        let mut delete_vector = DeleteVector::default();
        delete_vector.insert(1);
        delete_vector.insert(3);
        let filter = loader.delete_filter.clone();
        filter.install_pos_del_contribution(
            &guard,
            HashMap::from([(data_file.clone(), delete_vector)]),
        );
        guard.publish_loaded();

        let loaded = tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
            .await
            .expect("the waiting load must finish once the claim is published")
            .expect("the waiting load must deliver a result")
            .expect("the waiting load must succeed");
        let vector = loaded
            .resolve_delete_vector(&tasks, &data_file)
            .expect("the waiter must see the published vector");
        assert_eq!(
            vector.len(),
            2,
            "the waiter must observe the fully populated vector"
        );
    }

    /// The fail-loud half of the same contract. When the claiming task dies without publishing, the
    /// waiting load must surface the typed error, not proceed as though the deletes had loaded.
    /// Swallowing the wait result returns a `DeleteFilter` with no vector, so every row it deletes
    /// resurrects.
    ///
    /// It also pins the cause across the post-wake arm. `wait_for_pos_del_load` must render the reason
    /// the dead claimant recorded, not a generic one.
    ///
    /// Mutations: swallow the wait result, or hard-code the post-wake `Failed` reason. The first turns
    /// the whole test red, and the second turns the cause assertion red.
    #[tokio::test]
    async fn test_dv_waiter_surfaces_a_dead_claimants_error_instead_of_dropping_the_deletes() {
        let tmp_dir = TempDir::new().expect("temp dir for the DV fixture");
        let file_io = FileIO::new_with_fs();
        let data_file = format!("{}/data-a.parquet", tmp_dir.path().display());
        let (puffin_path, offset, length) =
            write_dv_puffin_file(&file_io, tmp_dir.path(), "deletes.puffin", &data_file, &[
                1, 3,
            ])
            .await;
        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let tasks = [dv_task(&puffin_path, &data_file, offset, length, 2)];
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));

        let cache_key = format!("{puffin_path}@{offset}");
        let mut guard = claim_pos_del(&loader, &cache_key);

        let mut waiting = loader.load_deletes(&tasks, schema);
        // Park the load on the notifier first. Otherwise the guard may already be dropped when the
        // loader claims, and the load takes the claim-time `Failed` arm, which is a different path.
        assert_parked_on_claim(
            &mut waiting,
            "the load must be parked on the claim's notifier before the claimant dies",
        )
        .await;

        // The claiming task records its cause and dies without publishing: `Drop` installs the
        // terminal failed state carrying that cause.
        let propagated = guard.note_failure(Error::new(
            ErrorKind::DataInvalid,
            "sentinel-cause: the claimant's own blob read failed",
        ));
        assert_eq!(
            propagated.kind(),
            ErrorKind::DataInvalid,
            "note_failure must hand the error back unchanged for `?` propagation"
        );
        drop(guard);

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
            .await
            .expect("a waiter whose claimant died must not hang")
            .expect("the waiting load must deliver a result");
        let error = result.expect_err(
            "a waiter whose claimant died must surface a typed error — succeeding here returns a \
             filter with NO delete vector and resurrects every deleted row",
        );
        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error.to_string().contains(&cache_key),
            "the error must name the blob claim key, got: {error}"
        );
        assert!(
            error.to_string().contains("sentinel-cause"),
            "the woken waiter's error must carry the cause the dead claimant recorded, got: \
             {error}"
        );
    }

    /// The parquet analogue of `test_dv_waiter_does_not_return_before_the_vector_is_installed`. This
    /// call site claims under the bare delete-file path, with the same under-delete consequence.
    ///
    /// Mutation: drop the `wait_for_pos_del_load` await on the parquet path.
    #[tokio::test]
    async fn test_parquet_pos_del_waiter_does_not_return_before_the_deletes_are_installed() {
        let tmp_dir = TempDir::new().expect("temp dir for the positional-delete fixture");
        let file_io = FileIO::new_with_fs();
        let data_file = format!("{}/data-a.parquet", tmp_dir.path().display());
        let pos_del_path = write_pos_del_parquet(tmp_dir.path(), "pos-del.parquet", &[
            (&data_file, Some(1)),
            (&data_file, Some(3)),
        ]);
        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let tasks = [parquet_pos_del_task(&pos_del_path)];
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));

        let guard = claim_pos_del(&loader, &pos_del_path);

        let mut waiting = loader.load_deletes(&tasks, schema);
        assert_parked_on_claim(
            &mut waiting,
            "the waiting load must still be pending while the claim is unpublished — returning \
             here hands the reader absent positional deletes and resurrects every deleted row",
        )
        .await;

        let mut delete_vector = DeleteVector::default();
        delete_vector.insert(1);
        delete_vector.insert(3);
        let filter = loader.delete_filter.clone();
        filter.install_pos_del_contribution(
            &guard,
            HashMap::from([(data_file.clone(), delete_vector)]),
        );
        guard.publish_loaded();

        let loaded = tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
            .await
            .expect("the waiting load must finish once the claim is published")
            .expect("the waiting load must deliver a result")
            .expect("the waiting load must succeed");
        let vector = loaded
            .resolve_delete_vector(&tasks, &data_file)
            .expect("the waiter must see the published positions");
        assert_eq!(
            vector.len(),
            2,
            "the waiter must observe the fully populated position set"
        );
    }

    /// The parquet analogue of
    /// `test_dv_waiter_surfaces_a_dead_claimants_error_instead_of_dropping_the_deletes`. This call site
    /// must propagate the wait error too.
    ///
    /// Mutation: swallow the `wait_for_pos_del_load` result on the parquet path.
    #[tokio::test]
    async fn test_parquet_pos_del_waiter_surfaces_a_dead_claimants_error() {
        let tmp_dir = TempDir::new().expect("temp dir for the positional-delete fixture");
        let file_io = FileIO::new_with_fs();
        let data_file = format!("{}/data-a.parquet", tmp_dir.path().display());
        let pos_del_path =
            write_pos_del_parquet(tmp_dir.path(), "pos-del.parquet", &[(&data_file, Some(1))]);
        let loader = CachingDeleteFileLoader::new(file_io, 10);
        let tasks = [parquet_pos_del_task(&pos_del_path)];
        let schema = Arc::new(Schema::builder().build().expect("empty schema"));

        let mut guard = claim_pos_del(&loader, &pos_del_path);

        let mut waiting = loader.load_deletes(&tasks, schema);
        assert_parked_on_claim(
            &mut waiting,
            "the load must be parked on the claim's notifier before the claimant dies",
        )
        .await;

        let propagated = guard.note_failure(Error::new(
            ErrorKind::DataInvalid,
            "sentinel-cause: the claimant's own parquet read failed",
        ));
        assert_eq!(
            propagated.kind(),
            ErrorKind::DataInvalid,
            "note_failure must hand the error back unchanged for `?` propagation"
        );
        drop(guard);

        let result = tokio::time::timeout(std::time::Duration::from_secs(5), waiting)
            .await
            .expect("a waiter whose claimant died must not hang")
            .expect("the waiting load must deliver a result");
        let error = result.expect_err(
            "a waiter whose claimant died must surface a typed error — succeeding here returns a \
             filter with NO positional deletes and resurrects every deleted row",
        );
        assert_eq!(error.kind(), ErrorKind::Unexpected);
        assert!(
            error.to_string().contains(&pos_del_path),
            "the error must name the delete file, got: {error}"
        );
        assert!(
            error.to_string().contains("sentinel-cause"),
            "the woken waiter's error must carry the cause the dead claimant recorded, got: \
             {error}"
        );
    }
}
