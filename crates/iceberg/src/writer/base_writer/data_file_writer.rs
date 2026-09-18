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

//! This module provide `DataFileWriter`.
//!
//! It also hosts `resolve_partition_spec_id` — the partition-spec-id stamping rule shared by all
//! three base writers (data, position-delete, equality-delete).

use std::borrow::Cow;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef as ArrowSchemaRef;

use crate::arrow::schema_to_arrow_schema;
use crate::spec::{DataContentType, DataFile, PartitionKey, PartitionSpec, SchemaRef};
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::{RollingFileWriter, RollingFileWriterBuilder};
use crate::writer::write_defaults::apply_write_defaults;
use crate::writer::{CurrentFileStatus, IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Resolve the `partition_spec_id` a base writer stamps on every file it produces, validating the
/// (spec, partition key) pair up front.
///
/// # Why this exists
///
/// Java takes the [`PartitionSpec`] as a REQUIRED constructor argument on every file builder —
/// `FileMetadata.Builder(spec)` (`core/.../FileMetadata.java`: `this.specId = spec.specId()`) and
/// `DataFiles.Builder(spec)` — and stamps `spec.specId()` unconditionally, so a Java-written file
/// always claims a spec that exists in the table. Rust's `DataFileBuilder` instead *defaults*
/// `partition_spec_id` to spec id 0, a fabricated value with no table behind
/// it. A file stamped 0 by that default is silently wrong whenever the spec it was actually written
/// under is not spec 0 — see `docs/ENGINE_CONTRACT.md` §7a for the observable outcomes (a same-arity
/// wrong-spec POSITION delete commits and then never applies; a keyless EQUALITY delete becomes a
/// GLOBAL delete instead; an unpartitioned current spec with a non-zero id cannot be written at all).
///
/// # Precedence
///
/// 1. **The [`PartitionKey`]'s own spec**, when a key is given. The key carries the partition tuple
///    *and* the spec that tuple was produced from, so it is authoritative — and a delete file must
///    claim the spec of the DATA FILES it deletes from, which is not necessarily the table's current
///    spec. A key whose spec differs from `configured_spec` is therefore legal, not an error.
/// 2. **The spec configured on the builder** (`with_partition_spec` / `unpartitioned`), when there
///    is no key.
/// 3. **Error** when neither is given. The old path stamped spec 0 silently. Call `unpartitioned()`
///    for a true unpartitioned table.
///
/// # The partitioned-without-a-key rejection
///
/// Case 2 with a spec that has partition fields is rejected with an [`ErrorKind::DataInvalid`]: the
/// file would carry the builder's default EMPTY partition tuple while claiming a spec whose
/// partition type has fields, which the commit path rejects anyway
/// (`SnapshotProducer::validate_partition_value`, Java `PartitionData` accessors) — failing here
/// names the actual mistake instead of surfacing an arity error one layer later.
///
/// The test is the spec's partition-field ARITY (`fields().is_empty()`), deliberately NOT
/// [`PartitionSpec::is_unpartitioned`], which is also `true` for an ALL-VOID spec (a V1 spec whose
/// fields were void-replaced). An all-void spec still has partition fields, so its partition type
/// still has that arity and a file under it still needs a tuple (of nulls) — `is_unpartitioned`
/// would wave it through into the same commit-time arity failure this check exists to prevent.
pub(crate) fn resolve_partition_spec_id(
    configured_spec: Option<&PartitionSpec>,
    partition_key: Option<&PartitionKey>,
) -> Result<i32> {
    match (partition_key, configured_spec) {
        // The key is authoritative — it carries both the tuple and the spec it came from.
        (Some(partition_key), _) => Ok(partition_key.spec().spec_id()),
        (None, Some(spec)) if !spec.fields().is_empty() => Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Partition spec {} has {} partition field(s) but the writer was built without a \
                 PartitionKey: a file written under a partitioned spec must carry its partition tuple",
                spec.spec_id(),
                spec.fields().len()
            ),
        )),
        (None, Some(spec)) => Ok(spec.spec_id()),
        (None, None) => Err(Error::new(
            ErrorKind::DataInvalid,
            "writer was built with neither a PartitionSpec nor a PartitionKey: call \
             unpartitioned() for an unpartitioned table, or with_partition_spec / a PartitionKey \
             for a partitioned one",
        )),
    }
}

/// Builder for `DataFileWriter`.
#[derive(Debug)]
pub struct DataFileWriterBuilder<B: FileWriterBuilder, L: LocationGenerator, F: FileNameGenerator> {
    inner: RollingFileWriterBuilder<B, L, F>,
    partition_spec: Option<PartitionSpec>,
    sort_order_id: Option<i32>,
}

impl<B, L, F> DataFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Create a new `DataFileWriterBuilder` using a `RollingFileWriterBuilder`.
    ///
    /// Prefer [`with_partition_spec`](Self::with_partition_spec) or [`unpartitioned`](Self::unpartitioned).
    /// `build(None)` with no spec now errors; see `resolve_partition_spec_id`.
    pub fn new(inner: RollingFileWriterBuilder<B, L, F>) -> Self {
        Self {
            inner,
            partition_spec: None,
            sort_order_id: None,
        }
    }

    /// Stamp `sort_order_id` on every produced file.
    pub fn with_sort_order_id(mut self, sort_order_id: i32) -> Self {
        self.sort_order_id = Some(sort_order_id);
        self
    }

    /// Stamp [`PartitionSpec::unpartition_spec`] (spec id 0, no fields).
    pub fn unpartitioned(self) -> Self {
        self.with_partition_spec(PartitionSpec::unpartition_spec())
    }

    /// Set the [`PartitionSpec`] the produced files are written under.
    ///
    /// This is the Rust counterpart of Java's REQUIRED `DataFiles.Builder(spec)` argument. It is used
    /// only when the writer is built WITHOUT a [`PartitionKey`]; a key always wins, because it
    /// carries the spec its tuple was produced from. See `resolve_partition_spec_id` for the full
    /// precedence and for why a partitioned spec with no key is rejected.
    ///
    /// **This writer OWNS `partition_spec_id` on every [`DataFile`] it emits.** `close()` sets the
    /// field unconditionally, so a custom [`FileWriter`](crate::writer::file_writer::FileWriter)
    /// that stamps it on the `DataFileBuilder` it returns will be overridden; give the spec to this
    /// builder instead. (No in-tree `FileWriter` stamps it — `ParquetWriter` leaves the field at its
    /// derive default.)
    pub fn with_partition_spec(mut self, partition_spec: PartitionSpec) -> Self {
        self.partition_spec = Some(partition_spec);
        self
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriterBuilder for DataFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    type R = DataFileWriter<B, L, F>;

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        let partition_spec_id =
            resolve_partition_spec_id(self.partition_spec.as_ref(), partition_key.as_ref())?;
        let arrow_schema = self
            .inner
            .iceberg_schema()
            .map(|schema| schema_to_arrow_schema(schema))
            .transpose()?
            .map(Arc::new);
        Ok(DataFileWriter {
            inner: Some(self.inner.build()),
            partition_key,
            partition_spec_id,
            sort_order_id: self.sort_order_id,
            schema: self.inner.iceberg_schema().cloned(),
            arrow_schema,
        })
    }
}

/// A writer write data is within one spec/partition.
#[derive(Debug)]
pub struct DataFileWriter<B: FileWriterBuilder, L: LocationGenerator, F: FileNameGenerator> {
    inner: Option<RollingFileWriter<B, L, F>>,
    partition_key: Option<PartitionKey>,
    /// The spec id stamped on every produced file, resolved once at build time by
    /// `resolve_partition_spec_id`.
    partition_spec_id: i32,
    sort_order_id: Option<i32>,
    schema: Option<SchemaRef>,
    arrow_schema: Option<ArrowSchemaRef>,
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriter for DataFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        let filled = match (&self.schema, &self.arrow_schema) {
            (Some(schema), Some(arrow_schema)) => {
                apply_write_defaults(schema, arrow_schema, &batch)?
            }
            _ => Cow::Borrowed(&batch),
        };
        if let Some(writer) = self.inner.as_mut() {
            writer.write(&self.partition_key, filled.as_ref()).await
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Writer is not initialized!",
            ))
        }
    }

    async fn close(&mut self) -> Result<Vec<DataFile>> {
        if let Some(writer) = self.inner.take() {
            writer
                .close()
                .await?
                .into_iter()
                .map(|mut res| {
                    res.content(DataContentType::Data);
                    // ALWAYS stamp the spec id (Java `DataFiles.Builder(spec)` does), never only when
                    // a partition key happens to be present — see `resolve_partition_spec_id`.
                    res.partition_spec_id(self.partition_spec_id);
                    if let Some(sort_order_id) = self.sort_order_id {
                        res.sort_order_id(sort_order_id);
                    }
                    if let Some(pk) = self.partition_key.as_ref() {
                        res.partition(pk.data().clone());
                    }
                    res.build().map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Failed to build data file: {e}"),
                        )
                    })
                })
                .collect()
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Data file writer has been closed.",
            ))
        }
    }
}

impl<B, L, F> CurrentFileStatus for DataFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    fn current_file_path(&self) -> String {
        // Post-`close()` the inner writer is taken; report empty rather than panicking on a
        // status query against a closed writer (same posture as `RollingFileWriter`).
        self.inner
            .as_ref()
            .map(|inner| inner.current_file_path())
            .unwrap_or_default()
    }

    fn current_row_num(&self) -> usize {
        self.inner
            .as_ref()
            .map(|inner| inner.current_row_num())
            .unwrap_or(0)
    }

    fn current_written_size(&self) -> usize {
        self.inner
            .as_ref()
            .map(|inner| inner.current_written_size())
            .unwrap_or(0)
    }
}
