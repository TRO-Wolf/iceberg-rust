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
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use super::{
    LegacyPositionDelete, close_touched_dv_containers_at,
    close_touched_dv_containers_with_partitions, load_legacy_positions,
};
use crate::io::{
    FileIOBuilder, FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorageFactory,
    OutputFile, Storage, StorageConfig, StorageFactory,
};
use crate::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use crate::spec::{DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Struct};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::{Catalog, CatalogBuilder, Result};

fn is_manifest(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.ends_with(".avro") && !name.starts_with("snap-")
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CountingStorage {
    #[serde(skip)]
    inner: Option<Arc<dyn Storage>>,
    #[serde(skip)]
    manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    bytes_read: Arc<AtomicU64>,
    #[serde(skip)]
    delete_manifest_paths: Arc<Mutex<HashSet<String>>>,
    #[serde(skip)]
    delete_manifest_reads: Arc<AtomicU64>,
}

impl CountingStorage {
    fn inner(&self) -> &Arc<dyn Storage> {
        self.inner.as_ref().expect("counting storage was built")
    }

    fn count(&self, path: &str) {
        if is_manifest(path) {
            self.manifest_reads.fetch_add(1, Ordering::Relaxed);
            if self
                .delete_manifest_paths
                .lock()
                .expect("delete-manifest path set")
                .contains(path)
            {
                self.delete_manifest_reads.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for CountingStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner().exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.inner().metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        self.count(path);
        let bytes = self.inner().read(path).await?;
        self.bytes_read
            .fetch_add(u64::try_from(bytes.len()).unwrap_or(0), Ordering::Relaxed);
        Ok(bytes)
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.count(path);
        let inner = self.inner().reader(path).await?;
        Ok(Box::new(CountingFileRead {
            inner,
            bytes_read: self.bytes_read.clone(),
        }))
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        self.inner().write(path, bs).await
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        self.inner().writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner().delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        self.inner().delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        self.inner().list(prefix).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        self.inner().new_output(path)
    }
}

struct CountingFileRead {
    inner: Box<dyn FileRead>,
    bytes_read: Arc<AtomicU64>,
}

#[async_trait]
impl FileRead for CountingFileRead {
    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        let bytes = self.inner.read(range).await?;
        self.bytes_read
            .fetch_add(u64::try_from(bytes.len()).unwrap_or(0), Ordering::Relaxed);
        Ok(bytes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CountingStorageFactory {
    #[serde(skip)]
    manifest_reads: Arc<AtomicU64>,
    #[serde(skip)]
    bytes_read: Arc<AtomicU64>,
    #[serde(skip)]
    delete_manifest_paths: Arc<Mutex<HashSet<String>>>,
    #[serde(skip)]
    delete_manifest_reads: Arc<AtomicU64>,
}

#[typetag::serde]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: Some(LocalFsStorageFactory.build(config)?),
            manifest_reads: self.manifest_reads.clone(),
            bytes_read: self.bytes_read.clone(),
            delete_manifest_paths: self.delete_manifest_paths.clone(),
            delete_manifest_reads: self.delete_manifest_reads.clone(),
        }))
    }
}

fn synthetic_data_file(path: &str) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(path.to_string())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition_spec_id(0)
        .partition(Struct::from_iter([Some(Literal::long(0))]))
        .build()
        .expect("build synthetic data file")
}

async fn append(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .fast_append()
        .add_data_files(vec![file])
        .apply(tx)
        .expect("apply fast append");
    tx.commit(catalog).await.expect("commit fast append")
}

struct Fixture {
    table: Table,
    paths: Vec<String>,
    manifest_reads: Arc<AtomicU64>,
    _warehouse: TempDir,
}

async fn three_data_manifests() -> Fixture {
    let warehouse = TempDir::new().expect("warehouse");
    let manifest_reads = Arc::new(AtomicU64::new(0));
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(CountingStorageFactory {
            manifest_reads: manifest_reads.clone(),
            bytes_read: Arc::new(AtomicU64::new(0)),
            delete_manifest_paths: Arc::new(Mutex::new(HashSet::new())),
            delete_manifest_reads: Arc::new(AtomicU64::new(0)),
        }))
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().expect("utf8").to_string(),
            )]),
        )
        .await
        .expect("catalog");
    let mut table = crate::transaction::tests::make_v3_minimal_table_in_catalog(&catalog).await;
    let mut paths = Vec::new();
    for index in 0..3 {
        let path = format!("{}/data/f{index}.parquet", table.metadata().location());
        table = append(&catalog, &table, synthetic_data_file(&path)).await;
        paths.push(path);
    }
    manifest_reads.store(0, Ordering::Relaxed);
    Fixture {
        table,
        paths,
        manifest_reads,
        _warehouse: warehouse,
    }
}

#[tokio::test]
async fn a_supplied_partition_map_reads_no_data_manifest() {
    let fixture = three_data_manifests().await;
    let new_positions = HashMap::from([(fixture.paths[0].clone(), vec![0u64])]);
    let known = HashMap::from([(
        fixture.paths[0].clone(),
        (0i32, Struct::from_iter([Some(Literal::long(0))])),
    )]);
    let close = close_touched_dv_containers_with_partitions(
        &fixture.table,
        &new_positions,
        None,
        &known,
        None,
    )
    .await
    .expect("close with a supplied partition map");
    assert_eq!(close.added.len(), 1);
    assert_eq!(
        fixture.manifest_reads.load(Ordering::Relaxed),
        0,
        "a supplied partition map must not walk any manifest"
    );
}

#[tokio::test]
async fn without_the_map_every_data_manifest_is_read_once() {
    let fixture = three_data_manifests().await;
    let new_positions = HashMap::from([(fixture.paths[0].clone(), vec![0u64])]);
    let close = close_touched_dv_containers_at(&fixture.table, &new_positions, None)
        .await
        .expect("close without a partition map");
    assert_eq!(close.added.len(), 1);
    assert_eq!(
        fixture.manifest_reads.load(Ordering::Relaxed),
        3,
        "each of the three data manifests is read exactly once"
    );
}

async fn write_pos_delete(table: &Table, deletes: &[(String, i64)]) -> DataFile {
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};

    use crate::spec::{MetricsConfig, PartitionKey};
    use crate::writer::base_writer::position_delete_writer::{
        PositionDeleteFileWriterBuilder, PositionDeleteWriterConfig,
        position_delete_writer_properties,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    let config = PositionDeleteWriterConfig::new().expect("pos-delete config");
    let location_gen =
        DefaultLocationGenerator::new(table.metadata().clone()).expect("location gen");
    let file_name_gen = DefaultFileNameGenerator::new(
        "pos-del".to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );
    let parquet_builder =
        ParquetWriterBuilder::new(position_delete_writer_properties(), config.schema().clone())
            .with_metrics_config(MetricsConfig::for_position_delete());
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_builder,
        table.file_io().clone(),
        location_gen,
        file_name_gen,
    );
    let partition_key = PartitionKey::new(
        table.metadata().default_partition_spec().as_ref().clone(),
        table.metadata().current_schema().clone(),
        Struct::from_iter([Some(Literal::long(0))]),
    )
    .expect("partition key");
    let mut writer = PositionDeleteFileWriterBuilder::new(rolling, config.clone())
        .build(Some(partition_key))
        .await
        .expect("build pos-delete writer");
    let paths: Vec<&str> = deletes.iter().map(|(path, _)| path.as_str()).collect();
    let positions: Vec<i64> = deletes.iter().map(|(_, pos)| *pos).collect();
    let batch = RecordBatch::try_new(config.arrow_schema().clone(), vec![
        Arc::new(StringArray::from(paths)) as ArrayRef,
        Arc::new(Int64Array::from(positions)) as ArrayRef,
    ])
    .expect("pos-delete batch");
    writer.write(batch).await.expect("write pos-delete");
    writer
        .close()
        .await
        .expect("close pos-delete")
        .into_iter()
        .next()
        .expect("one pos-delete file")
}

async fn commit_delete(catalog: &impl Catalog, table: &Table, file: DataFile) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .row_delta()
        .add_deletes(vec![file])
        .apply(tx)
        .expect("apply row delta");
    tx.commit(catalog).await.expect("commit row delta")
}

async fn upgrade_v3(catalog: &impl Catalog, table: &Table) -> Table {
    let tx = Transaction::new(table);
    let tx = tx
        .upgrade_table_version()
        .set_format_version(crate::spec::FormatVersion::V3)
        .apply(tx)
        .expect("apply upgrade");
    tx.commit(catalog).await.expect("commit upgrade")
}

async fn delete_manifest_count(table: &Table) -> usize {
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
    let list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    list.entries()
        .iter()
        .filter(|manifest| manifest.content == crate::spec::ManifestContentType::Deletes)
        .count()
}

struct DeleteManifestFixture {
    table: Table,
    path: String,
    manifest_reads: Arc<AtomicU64>,
    delete_manifest_reads: Arc<AtomicU64>,
    delete_manifests: usize,
    _warehouse: TempDir,
}

async fn n_delete_manifests(n: usize, with_legacy: bool) -> DeleteManifestFixture {
    let warehouse = TempDir::new().expect("warehouse");
    let manifest_reads = Arc::new(AtomicU64::new(0));
    let delete_manifest_reads = Arc::new(AtomicU64::new(0));
    let delete_manifest_paths = Arc::new(Mutex::new(HashSet::new()));
    let catalog = MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(CountingStorageFactory {
            manifest_reads: manifest_reads.clone(),
            bytes_read: Arc::new(AtomicU64::new(0)),
            delete_manifest_paths: delete_manifest_paths.clone(),
            delete_manifest_reads: delete_manifest_reads.clone(),
        }))
        .load(
            "memory",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                warehouse.path().to_str().expect("utf8").to_string(),
            )]),
        )
        .await
        .expect("catalog");
    let mut table = crate::transaction::tests::make_v2_minimal_table_in_catalog(&catalog).await;
    let path = format!("{}/data/f0.parquet", table.metadata().location());
    let other = format!("{}/data/f1.parquet", table.metadata().location());
    table = append(&catalog, &table, synthetic_data_file(&path)).await;
    table = append(&catalog, &table, synthetic_data_file(&other)).await;
    for index in 0..n {
        let target = if with_legacy { &path } else { &other };
        let pos_delete = write_pos_delete(&table, &[(
            target.clone(),
            i64::try_from(index).unwrap_or(0),
        )])
        .await;
        table = commit_delete(&catalog, &table, pos_delete).await;
    }
    table = upgrade_v3(&catalog, &table).await;
    let delete_manifests = delete_manifest_count(&table).await;
    let snapshot = table
        .metadata()
        .current_snapshot()
        .expect("current snapshot");
    let list = snapshot
        .load_manifest_list(table.file_io(), table.metadata())
        .await
        .expect("manifest list");
    let paths: HashSet<String> = list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == crate::spec::ManifestContentType::Deletes)
        .map(|manifest| manifest.manifest_path.clone())
        .collect();
    *delete_manifest_paths
        .lock()
        .expect("delete-manifest path set") = paths;
    manifest_reads.store(0, Ordering::Relaxed);
    delete_manifest_reads.store(0, Ordering::Relaxed);
    DeleteManifestFixture {
        table,
        path,
        manifest_reads,
        delete_manifest_reads,
        delete_manifests,
        _warehouse: warehouse,
    }
}

#[tokio::test]
async fn delete_manifests_are_read_once_without_legacy_deletes() {
    let fixture = n_delete_manifests(8, false).await;
    let expected = fixture.delete_manifests;
    assert!(
        expected >= 8,
        "fixture must write one delete manifest per commit"
    );
    let known = HashMap::from([(
        fixture.path.clone(),
        (0i32, Struct::from_iter([Some(Literal::long(0))])),
    )]);
    let new_positions = HashMap::from([(fixture.path.clone(), vec![0u64])]);
    let close = close_touched_dv_containers_with_partitions(
        &fixture.table,
        &new_positions,
        None,
        &known,
        None,
    )
    .await
    .expect("close without applicable legacy deletes");
    assert_eq!(close.added.len(), 1);
    assert!(
        close.legacy_deletes.is_empty(),
        "no legacy delete names the touched file"
    );
    assert_eq!(
        fixture.delete_manifest_reads.load(Ordering::Relaxed),
        u64::try_from(expected).expect("count fits"),
        "each delete manifest is read exactly once"
    );
}

#[tokio::test]
async fn delete_manifests_are_read_once_with_legacy_deletes() {
    let fixture = n_delete_manifests(8, true).await;
    let expected = fixture.delete_manifests;
    assert!(
        expected >= 8,
        "fixture must write one delete manifest per commit"
    );
    let known = HashMap::from([(
        fixture.path.clone(),
        (0i32, Struct::from_iter([Some(Literal::long(0))])),
    )]);
    let new_positions = HashMap::from([(fixture.path.clone(), vec![99u64])]);
    let close = close_touched_dv_containers_with_partitions(
        &fixture.table,
        &new_positions,
        None,
        &known,
        None,
    )
    .await
    .expect("close with legacy deletes");
    assert_eq!(close.added.len(), 1);
    assert_eq!(
        close.legacy_deletes.len(),
        8,
        "every file-scoped parquet delete names the touched file"
    );
    assert!(
        close.legacy_deletes.iter().all(|item| item.file_scoped),
        "equal file_path bounds make each delete file-scoped"
    );
    assert_eq!(
        fixture.delete_manifest_reads.load(Ordering::Relaxed),
        u64::try_from(expected).expect("count fits"),
        "each delete manifest is read exactly once"
    );
}

#[tokio::test]
async fn preloaded_manifest_list_skips_the_list_reread() {
    let fixture = n_delete_manifests(3, true).await;
    let snapshot = fixture
        .table
        .metadata()
        .current_snapshot()
        .expect("current snapshot")
        .clone();
    fixture.manifest_reads.store(0, Ordering::Relaxed);
    let list = snapshot
        .load_manifest_list(fixture.table.file_io(), fixture.table.metadata())
        .await
        .expect("preload manifest list");
    let list_reads = fixture.manifest_reads.load(Ordering::Relaxed);
    fixture.manifest_reads.store(0, Ordering::Relaxed);
    fixture.delete_manifest_reads.store(0, Ordering::Relaxed);
    let known = HashMap::from([(
        fixture.path.clone(),
        (0i32, Struct::from_iter([Some(Literal::long(0))])),
    )]);
    let new_positions = HashMap::from([(fixture.path.clone(), vec![0u64])]);
    let close = close_touched_dv_containers_with_partitions(
        &fixture.table,
        &new_positions,
        None,
        &known,
        Some(&list),
    )
    .await
    .expect("close with a pre-loaded manifest list");
    assert_eq!(close.added.len(), 1);
    assert_eq!(
        fixture.delete_manifest_reads.load(Ordering::Relaxed),
        u64::try_from(fixture.delete_manifests).expect("count fits"),
        "pre-loaded list must not be read again; delete manifests still once"
    );
    let _ = list_reads;
}

#[tokio::test]
async fn close_returns_touched_data_sequence_numbers() {
    let fixture = three_data_manifests().await;
    let new_positions = HashMap::from([(fixture.paths[0].clone(), vec![0u64])]);
    let close = close_touched_dv_containers_at(&fixture.table, &new_positions, None)
        .await
        .expect("close without a partition map");
    assert!(
        close.data_sequence_numbers.contains_key(&fixture.paths[0]),
        "close reports the touched file's data sequence number"
    );
}

#[tokio::test]
async fn load_legacy_positions_projects_past_the_row_column() {
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::file::properties::WriterProperties;

    use crate::metadata_columns::{
        RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
    };
    use crate::spec::DataFileBuilder;

    let warehouse = TempDir::new().expect("warehouse");
    let bytes_read = Arc::new(AtomicU64::new(0));
    let file_io = FileIOBuilder::new(Arc::new(CountingStorageFactory {
        manifest_reads: Arc::new(AtomicU64::new(0)),
        bytes_read: bytes_read.clone(),
        delete_manifest_paths: Arc::new(Mutex::new(HashSet::new())),
        delete_manifest_reads: Arc::new(AtomicU64::new(0)),
    }))
    .build();
    let del_path = format!(
        "{}/row-del.parquet",
        warehouse.path().to_str().expect("utf8")
    );
    let data_path = "s3://b/a.parquet";
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            RESERVED_FIELD_ID_DELETE_FILE_PATH.to_string(),
        )])),
        Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            RESERVED_FIELD_ID_DELETE_FILE_POS.to_string(),
        )])),
        Field::new("row", DataType::Utf8, true),
    ]));
    let n = 4_000i64;
    let paths: Vec<String> = vec![data_path.to_string(); usize::try_from(n).expect("n")];
    let positions: Vec<i64> = (0..n).collect();
    let rows: Vec<String> = (0..n).map(|index| format!("row-{index:0200}")).collect();
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(StringArray::from(paths)) as _,
        Arc::new(Int64Array::from(positions)) as _,
        Arc::new(StringArray::from(rows)) as _,
    ])
    .expect("row-column batch");
    {
        let file = std::fs::File::create(&del_path).expect("create parquet");
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close parquet");
    }
    let file_size = std::fs::metadata(&del_path).expect("metadata").len();
    let file = DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(del_path.clone())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(file_size)
        .record_count(u64::try_from(n).expect("n"))
        .partition_spec_id(0)
        .partition(Struct::empty())
        .referenced_data_file(Some(data_path.to_string()))
        .build()
        .expect("delete file");
    let delete = LegacyPositionDelete {
        file,
        touched: vec![data_path.to_string()],
        file_scoped: true,
        data_sequence_number: 1,
    };
    bytes_read.store(0, Ordering::Relaxed);
    let positions = load_legacy_positions(&file_io, &delete, data_path)
        .await
        .expect("load positions");
    assert_eq!(positions.len(), usize::try_from(n).expect("n"));
    let read = bytes_read.load(Ordering::Relaxed);
    assert!(
        read < file_size,
        "projected pos-only read {read} must be smaller than file size {file_size}"
    );
}

#[tokio::test]
#[ignore = "measurement, not a CI pin"]
async fn measure_close_at_8_and_192_delete_manifests() {
    for n in [8usize, 192usize] {
        let fixture = n_delete_manifests(n, false).await;
        let known = HashMap::from([(
            fixture.path.clone(),
            (0i32, Struct::from_iter([Some(Literal::long(0))])),
        )]);
        let new_positions = HashMap::from([(fixture.path.clone(), vec![0u64])]);
        let walk_start = std::time::Instant::now();
        {
            let snapshot = fixture
                .table
                .metadata()
                .current_snapshot()
                .expect("current snapshot");
            let list = snapshot
                .load_manifest_list(fixture.table.file_io(), fixture.table.metadata())
                .await
                .expect("manifest list");
            for manifest_file in list.entries() {
                if manifest_file.content == crate::spec::ManifestContentType::Deletes {
                    let _ = manifest_file
                        .load_manifest(fixture.table.file_io())
                        .await
                        .expect("delete manifest");
                }
            }
        }
        let walk = walk_start.elapsed();
        let start = std::time::Instant::now();
        let close = close_touched_dv_containers_with_partitions(
            &fixture.table,
            &new_positions,
            None,
            &known,
            None,
        )
        .await
        .expect("close");
        let elapsed = start.elapsed();
        println!(
            "F-22 n={n} delete_manifests={} close={elapsed:?} sequential_delete_walk={walk:?} added={}",
            fixture.delete_manifests,
            close.added.len()
        );
        assert_eq!(close.added.len(), 1);
    }
}
