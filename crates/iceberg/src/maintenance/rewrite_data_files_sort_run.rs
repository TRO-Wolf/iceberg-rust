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

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::future::Future;

use arrow_array::{RecordBatch, UInt32Array};
use arrow_schema::{ArrowError, SchemaRef as ArrowSchemaRef};
use bytes::Bytes;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::async_reader::{ParquetRecordBatchStream, ParquetRecordBatchStreamBuilder};
use parquet::arrow::async_writer::AsyncFileWriter as ArrowAsyncFileWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::arrow::ArrowFileReader;
use crate::error::{Error, ErrorKind, Result};
use crate::io::{FileIO, FileMetadata, FileWrite};
use crate::maintenance::rewrite_data_files_sort_key::KeyPlan;

pub(super) const SORT_MERGE_FAN_IN: usize = 16;

const MIN_SPILL_BATCH_BYTES: usize = 64 * 1024;
const MAX_SPILL_BATCH_ROWS: usize = 65536;
const MERGE_OUTPUT_ROWS: usize = 8192;

pub(super) trait SortedBatchSink {
    fn write_sorted(&mut self, batch: RecordBatch) -> impl Future<Output = Result<()>> + Send;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SortRunStats {
    pub(crate) spilled_runs: usize,
    pub(crate) merge_passes: usize,
    pub(crate) peak_sort_bytes: u64,
}

pub(super) struct ExternalSorter {
    file_io: FileIO,
    spill_prefix: String,
    budget: usize,
    arrow_schema: ArrowSchemaRef,
    plan: KeyPlan,
    run: Vec<RecordBatch>,
    run_bytes: usize,
    run_rows: usize,
    spills: Vec<SpillRun>,
    created_paths: Vec<String>,
    next_spill: usize,
    stats: SortRunStats,
}

struct SpillRun {
    path: String,
    rows: usize,
    batch_rows: usize,
}

impl ExternalSorter {
    pub(super) fn new(
        file_io: FileIO,
        spill_prefix: String,
        budget: u64,
        arrow_schema: ArrowSchemaRef,
        plan: KeyPlan,
    ) -> ExternalSorter {
        ExternalSorter {
            file_io,
            spill_prefix,
            budget: usize::try_from(budget).unwrap_or(usize::MAX),
            arrow_schema,
            plan,
            run: Vec::new(),
            run_bytes: 0,
            run_rows: 0,
            spills: Vec::new(),
            created_paths: Vec::new(),
            next_spill: 0,
            stats: SortRunStats::default(),
        }
    }

    pub(super) async fn push(&mut self, batch: RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.run_bytes += batch.get_array_memory_size();
        self.run_rows += batch.num_rows();
        self.run.push(batch);
        self.stats.peak_sort_bytes = self.stats.peak_sort_bytes.max(self.run_bytes as u64);
        if self.run_bytes >= self.budget {
            self.spill_current_run().await?;
        }
        Ok(())
    }

    pub(super) async fn finish<S: SortedBatchSink>(mut self, sink: &mut S) -> Result<SortRunStats> {
        let outcome = self.drain(sink).await;
        let cleanup = self.delete_spills().await;
        outcome?;
        cleanup?;
        Ok(self.stats)
    }

    async fn drain<S: SortedBatchSink>(&mut self, sink: &mut S) -> Result<()> {
        if self.spills.is_empty() {
            let Some(sorted) = self.sort_current_run()? else {
                return Ok(());
            };
            return sink.write_sorted(sorted).await;
        }
        if !self.run.is_empty() {
            self.spill_current_run().await?;
        }
        while self.spills.len() > SORT_MERGE_FAN_IN {
            let mut merged = Vec::new();
            let pending: Vec<SpillRun> = std::mem::take(&mut self.spills);
            for chunk in pending.chunks(SORT_MERGE_FAN_IN) {
                if chunk.len() == 1 {
                    merged.push(SpillRun {
                        path: chunk[0].path.clone(),
                        rows: chunk[0].rows,
                        batch_rows: chunk[0].batch_rows,
                    });
                    continue;
                }
                let rows: usize = chunk.iter().map(|run| run.rows).sum();
                let batch_rows = chunk
                    .iter()
                    .map(|run| run.batch_rows)
                    .max()
                    .unwrap_or(MERGE_OUTPUT_ROWS);
                let path = self.spill_path();
                let mut writer =
                    SpillWriter::create(&self.file_io, &path, &self.arrow_schema, batch_rows)
                        .await?;
                self.merge_into(chunk, &mut writer).await?;
                writer.close().await?;
                merged.push(SpillRun {
                    path,
                    rows,
                    batch_rows,
                });
            }
            self.spills = merged;
            self.stats.merge_passes += 1;
        }
        let runs: Vec<SpillRun> = std::mem::take(&mut self.spills);
        let result = self.merge_into(&runs, sink).await;
        self.stats.merge_passes += 1;
        result
    }

    async fn merge_into<S: SortedBatchSink>(&self, runs: &[SpillRun], sink: &mut S) -> Result<()> {
        let mut readers = Vec::with_capacity(runs.len());
        for run in runs {
            readers.push(SpillReader::open(&self.file_io, run, &self.plan).await?);
        }
        let mut heap: BinaryHeap<Reverse<(Vec<u8>, usize)>> = BinaryHeap::new();
        for (index, reader) in readers.iter_mut().enumerate() {
            if let Some(key) = reader.peek_key() {
                heap.push(Reverse((key, index)));
            }
        }
        let mut sources: Vec<RecordBatch> = Vec::new();
        let mut slots: Vec<Option<(usize, usize)>> = vec![None; readers.len()];
        let mut indices: Vec<(usize, usize)> = Vec::with_capacity(MERGE_OUTPUT_ROWS);
        while let Some(Reverse((_, index))) = heap.pop() {
            let reader = &mut readers[index];
            let epoch = reader.epoch;
            let (batch, row) = reader.take_row()?;
            let slot = match slots[index] {
                Some((slot, held)) if held == epoch => slot,
                _ => {
                    sources.push(batch);
                    slots[index] = Some((sources.len() - 1, epoch));
                    sources.len() - 1
                }
            };
            indices.push((slot, row));
            if indices.len() >= MERGE_OUTPUT_ROWS {
                flush_merged(&sources, &indices, sink).await?;
                indices.clear();
                sources.clear();
                slots.iter_mut().for_each(|slot| *slot = None);
            }
            let reader = &mut readers[index];
            if reader.exhausted_batch() {
                reader.load_next(&self.plan).await?;
            }
            if let Some(key) = reader.peek_key() {
                heap.push(Reverse((key, index)));
            }
        }
        if !indices.is_empty() {
            flush_merged(&sources, &indices, sink).await?;
        }
        Ok(())
    }

    fn sort_current_run(&mut self) -> Result<Option<RecordBatch>> {
        if self.run.is_empty() {
            return Ok(None);
        }
        let batches = std::mem::take(&mut self.run);
        self.run_bytes = 0;
        self.run_rows = 0;
        let batch = arrow_select::concat::concat_batches(&self.arrow_schema, &batches)
            .map_err(sort_run_err)?;
        drop(batches);
        let keys = self.plan.encode(&batch)?;
        let mut order: Vec<u32> = (0..keys.len())
            .map(|row| u32::try_from(row).unwrap_or(u32::MAX))
            .collect();
        order.sort_by(|left, right| keys[*left as usize].cmp(&keys[*right as usize]));
        let indices = UInt32Array::from(order);
        let sorted =
            arrow_select::take::take_record_batch(&batch, &indices).map_err(sort_run_err)?;
        Ok(Some(sorted))
    }

    async fn spill_current_run(&mut self) -> Result<()> {
        let rows = self.run_rows;
        let bytes = self.run_bytes;
        let Some(sorted) = self.sort_current_run()? else {
            return Ok(());
        };
        let batch_rows = spill_batch_rows(bytes, rows, self.budget);
        let path = self.spill_path();
        let mut writer =
            SpillWriter::create(&self.file_io, &path, &self.arrow_schema, batch_rows).await?;
        for offset in (0..sorted.num_rows()).step_by(batch_rows) {
            let length = batch_rows.min(sorted.num_rows() - offset);
            writer.write_sorted(sorted.slice(offset, length)).await?;
        }
        writer.close().await?;
        self.spills.push(SpillRun {
            path,
            rows: sorted.num_rows(),
            batch_rows,
        });
        self.stats.spilled_runs += 1;
        Ok(())
    }

    fn spill_path(&mut self) -> String {
        self.next_spill += 1;
        let path = format!("{}/run-{:05}.parquet", self.spill_prefix, self.next_spill);
        self.created_paths.push(path.clone());
        path
    }

    async fn delete_spills(&mut self) -> Result<()> {
        let mut failure = None;
        self.spills.clear();
        for path in std::mem::take(&mut self.created_paths) {
            if let Err(error) = self.file_io.delete(&path).await {
                failure = Some(error);
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

async fn flush_merged<S: SortedBatchSink>(
    sources: &[RecordBatch],
    indices: &[(usize, usize)],
    sink: &mut S,
) -> Result<()> {
    let refs: Vec<&RecordBatch> = sources.iter().collect();
    let batch =
        arrow_select::interleave::interleave_record_batch(&refs, indices).map_err(sort_run_err)?;
    sink.write_sorted(batch).await
}

fn spill_batch_rows(run_bytes: usize, run_rows: usize, budget: usize) -> usize {
    let bytes_per_row = (run_bytes / run_rows.max(1)).max(1);
    let target = (budget / (SORT_MERGE_FAN_IN + 1)).max(MIN_SPILL_BATCH_BYTES);
    (target / bytes_per_row).clamp(1, MAX_SPILL_BATCH_ROWS)
}

struct SpillWriter {
    writer: AsyncArrowWriter<SpillFileWriter>,
}

impl SpillWriter {
    async fn create(
        file_io: &FileIO,
        path: &str,
        arrow_schema: &ArrowSchemaRef,
        batch_rows: usize,
    ) -> Result<SpillWriter> {
        let output = file_io.new_output(path)?;
        let properties = WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_max_row_group_row_count(Some(batch_rows))
            .build();
        let writer = AsyncArrowWriter::try_new(
            SpillFileWriter(output.writer().await?),
            arrow_schema.clone(),
            Some(properties),
        )
        .map_err(sort_spill_err)?;
        Ok(SpillWriter { writer })
    }

    async fn close(self) -> Result<()> {
        self.writer.close().await.map_err(sort_spill_err)?;
        Ok(())
    }
}

impl SortedBatchSink for SpillWriter {
    async fn write_sorted(&mut self, batch: RecordBatch) -> Result<()> {
        self.writer.write(&batch).await.map_err(sort_spill_err)
    }
}

struct SpillFileWriter(Box<dyn FileWrite>);

impl ArrowAsyncFileWriter for SpillFileWriter {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, parquet::errors::Result<()>> {
        Box::pin(async {
            self.0
                .write(bs)
                .await
                .map_err(|error| parquet::errors::ParquetError::External(Box::new(error)))
        })
    }

    fn complete(&mut self) -> BoxFuture<'_, parquet::errors::Result<()>> {
        Box::pin(async {
            self.0
                .close()
                .await
                .map_err(|error| parquet::errors::ParquetError::External(Box::new(error)))
        })
    }
}

struct SpillReader {
    stream: ParquetRecordBatchStream<ArrowFileReader>,
    batch: Option<RecordBatch>,
    keys: Vec<Vec<u8>>,
    row: usize,
    epoch: usize,
}

impl SpillReader {
    async fn open(file_io: &FileIO, run: &SpillRun, plan: &KeyPlan) -> Result<SpillReader> {
        let input = file_io.new_input(&run.path)?;
        let size = input.metadata().await?.size;
        let reader = ArrowFileReader::new(FileMetadata { size }, input.reader().await?);
        let stream = ParquetRecordBatchStreamBuilder::new(reader)
            .await
            .map_err(sort_spill_err)?
            .with_batch_size(run.batch_rows)
            .build()
            .map_err(sort_spill_err)?;
        let mut reader = SpillReader {
            stream,
            batch: None,
            keys: Vec::new(),
            row: 0,
            epoch: 0,
        };
        reader.load_next(plan).await?;
        Ok(reader)
    }

    async fn load_next(&mut self, plan: &KeyPlan) -> Result<()> {
        loop {
            let next = self.stream.try_next().await.map_err(sort_spill_err)?;
            match next {
                None => {
                    self.batch = None;
                    self.keys.clear();
                    self.row = 0;
                    return Ok(());
                }
                Some(batch) if batch.num_rows() == 0 => continue,
                Some(batch) => {
                    self.keys = plan.encode(&batch)?;
                    self.batch = Some(batch);
                    self.row = 0;
                    self.epoch += 1;
                    return Ok(());
                }
            }
        }
    }

    fn peek_key(&self) -> Option<Vec<u8>> {
        let batch = self.batch.as_ref()?;
        if self.row >= batch.num_rows() {
            return None;
        }
        Some(self.keys[self.row].clone())
    }

    fn exhausted_batch(&self) -> bool {
        match &self.batch {
            None => true,
            Some(batch) => self.row >= batch.num_rows(),
        }
    }

    fn take_row(&mut self) -> Result<(RecordBatch, usize)> {
        let batch = self.batch.clone().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "A merged sort run reported a row after its last batch",
            )
        })?;
        let row = self.row;
        self.row += 1;
        Ok((batch, row))
    }
}

fn sort_run_err(error: ArrowError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Failed to order the rewritten rows by the requested sort order",
    )
    .with_source(error)
}

fn sort_spill_err(error: parquet::errors::ParquetError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Failed to spill a sorted run of the rewritten rows",
    )
    .with_source(error)
}
