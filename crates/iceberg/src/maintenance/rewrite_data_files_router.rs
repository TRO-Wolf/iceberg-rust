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

use std::collections::{HashMap, VecDeque};

use arrow_array::RecordBatch;

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{DataFile, PartitionKey, Struct};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};

pub(crate) struct BoundedPartitionRouter<B>
where B: IcebergWriterBuilder<RecordBatch, Vec<DataFile>>
{
    inner_builder: B,
    max_open: usize,
    writers: HashMap<Struct, B::R>,
    lru: VecDeque<Struct>,
    closed_files: Vec<DataFile>,
    peak_open: usize,
}

impl<B> BoundedPartitionRouter<B>
where B: IcebergWriterBuilder<RecordBatch, Vec<DataFile>>
{
    pub(crate) fn new(inner_builder: B, max_open: usize) -> Result<Self> {
        if max_open == 0 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "'max-open-partition-writers' is set to 0 but must be > 0",
            ));
        }
        Ok(Self {
            inner_builder,
            max_open,
            writers: HashMap::new(),
            lru: VecDeque::new(),
            closed_files: Vec::new(),
            peak_open: 0,
        })
    }

    pub(crate) fn peak_open_partition_writers(&self) -> usize {
        self.peak_open
    }

    pub(crate) async fn write(
        &mut self,
        partition_key: PartitionKey,
        batch: RecordBatch,
    ) -> Result<()> {
        let key = partition_key.data().clone();
        if !self.writers.contains_key(&key) {
            if self.writers.len() >= self.max_open {
                self.evict_least_recent().await?;
            }
            let writer = self.inner_builder.build(Some(partition_key)).await?;
            self.writers.insert(key.clone(), writer);
            let open = self.writers.len();
            if open > self.peak_open {
                self.peak_open = open;
            }
        }
        self.touch_lru(&key);
        let writer = self.writers.get_mut(&key).ok_or_else(|| {
            Error::new(ErrorKind::Unexpected, "partition writer missing after open")
        })?;
        writer.write(batch).await
    }

    pub(crate) async fn close(mut self) -> Result<Vec<DataFile>> {
        let mut files = std::mem::take(&mut self.closed_files);
        for (_, mut writer) in self.writers {
            files.extend(writer.close().await?);
        }
        Ok(files)
    }

    fn touch_lru(&mut self, key: &Struct) {
        self.lru.retain(|existing| existing != key);
        self.lru.push_back(key.clone());
    }

    async fn evict_least_recent(&mut self) -> Result<()> {
        let Some(key) = self.lru.pop_front() else {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "partition writer bound reached with no LRU key to evict",
            ));
        };
        let Some(mut writer) = self.writers.remove(&key) else {
            return Ok(());
        };
        self.closed_files.extend(writer.close().await?);
        Ok(())
    }
}
