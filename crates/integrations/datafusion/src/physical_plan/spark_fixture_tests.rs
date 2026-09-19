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

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::arrow::array::Int64Array;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};
use iceberg::io::{
    FileIO, FileIOBuilder, FileInfo, FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorage,
    OutputFile, Storage, StorageConfig, StorageFactory,
};
use iceberg::table::StaticTable;
use iceberg::{Result, TableIdent};
use serde::{Deserialize, Serialize};

use crate::physical_plan::scan_knobs::ensure_iceberg_scan_options;
use crate::table::IcebergStaticTableProvider;

const FIXTURE_PREFIX: &str = "/iceberg-fixtures/page-prune";

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../iceberg/testdata/interop/page_prune")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefixStorage {
    root: String,
}

impl PrefixStorage {
    fn map(&self, path: &str) -> String {
        match path.strip_prefix(FIXTURE_PREFIX) {
            Some(rest) => format!("{}{rest}", self.root),
            None => path.to_string(),
        }
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for PrefixStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        LocalFsStorage::new().exists(&self.map(path)).await
    }
    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        LocalFsStorage::new().metadata(&self.map(path)).await
    }
    async fn read(&self, path: &str) -> Result<Bytes> {
        LocalFsStorage::new().read(&self.map(path)).await
    }
    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        LocalFsStorage::new().reader(&self.map(path)).await
    }
    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        LocalFsStorage::new().write(&self.map(path), bs).await
    }
    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        LocalFsStorage::new().writer(&self.map(path)).await
    }
    async fn delete(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete(&self.map(path)).await
    }
    async fn delete_prefix(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete_prefix(&self.map(path)).await
    }
    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        LocalFsStorage::new().list(&self.map(prefix)).await
    }
    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }
    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefixStorageFactory {
    root: String,
}

#[typetag::serde]
impl StorageFactory for PrefixStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(PrefixStorage {
            root: self.root.clone(),
        }))
    }
}

fn fixture_io() -> FileIO {
    FileIOBuilder::new(Arc::new(PrefixStorageFactory {
        root: fixture_root().to_str().expect("utf8").to_string(),
    }))
    .build()
}

async fn fixture_table(name: &str, metadata_file: &str) -> iceberg::table::Table {
    StaticTable::from_metadata_file(
        &format!("{FIXTURE_PREFIX}/ns/{name}/metadata/{metadata_file}"),
        TableIdent::from_strs(["ns", name]).expect("ident"),
        fixture_io(),
    )
    .await
    .expect("static table")
    .into_table()
}

async fn fixture_session(name: &str, metadata_file: &str, row_selection: bool) -> SessionContext {
    let table = fixture_table(name, metadata_file).await;
    let provider = IcebergStaticTableProvider::try_new_from_table(table)
        .await
        .expect("provider");
    let mut config = SessionConfig::new().with_target_partitions(4);
    ensure_iceberg_scan_options(&mut config);
    config
        .options_mut()
        .set(
            "iceberg.row_selection_enabled",
            if row_selection { "true" } else { "false" },
        )
        .expect("set extension key");
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("t", Arc::new(provider))
        .expect("register table");
    ctx
}

async fn query_aggregates(ctx: &SessionContext, predicate: &str) -> (i64, i64, i64, i64) {
    let sql = format!("SELECT count(*), sum(id), min(id), max(id) FROM t WHERE {predicate}");
    let batches: Vec<RecordBatch> = ctx
        .sql(&sql)
        .await
        .expect("plan")
        .collect()
        .await
        .expect("collect");
    assert_eq!(batches.len(), 1, "aggregate query must emit one batch");
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 1);
    let int = |col: usize| {
        batch
            .column(col)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64")
            .value(0)
    };
    (int(0), int(1), int(2), int(3))
}

async fn assert_sql_matches_spark(
    name: &str,
    metadata_file: &str,
    predicate: &str,
    expected: (i64, i64, i64, i64),
) {
    let off = fixture_session(name, metadata_file, false).await;
    assert_eq!(
        query_aggregates(&off, predicate).await,
        expected,
        "{name}: {predicate}: OFF aggregates"
    );
    let on = fixture_session(name, metadata_file, true).await;
    assert_eq!(
        query_aggregates(&on, predicate).await,
        expected,
        "{name}: {predicate}: ON aggregates"
    );
}

const S: &str = "pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp";

#[tokio::test]
async fn spark_base_v2_sql_door() {
    let cases: [(String, (i64, i64, i64, i64)); 6] = [
        ("id BETWEEN 400 AND 520".to_string(), (121, 55660, 400, 520)),
        ("id = 1234".to_string(), (1, 1234, 1234, 1234)),
        ("i > 1800".to_string(), (199, 378100, 1801, 1999)),
        (
            "ts >= TIMESTAMP '2026-01-01 00:10:00' AND ts < TIMESTAMP '2026-01-01 00:12:00'"
                .to_string(),
            (240, 316680, 1200, 1439),
        ),
        (format!("s = '{S}001234'"), (1, 1234, 1234, 1234)),
        ("n != 900".to_string(), (1499, 1873350, 500, 1999)),
    ];
    for (predicate, expected) in cases {
        assert_sql_matches_spark("base_v2", "v2.metadata.json", &predicate, expected).await;
    }
}

#[tokio::test]
async fn spark_del_v3_sql_door() {
    let cases: [(String, (i64, i64, i64, i64)); 4] = [
        ("i > 1800".to_string(), (220, 399310, 1000, 1999)),
        ("id BETWEEN 400 AND 520".to_string(), (104, 47806, 400, 520)),
        ("id = 1234".to_string(), (1, 1234, 1234, 1234)),
        ("n != 900".to_string(), (1474, 1849102, 500, 1999)),
    ];
    for (predicate, expected) in cases {
        assert_sql_matches_spark("del_v3", "v6.metadata.json", &predicate, expected).await;
    }
}

#[tokio::test]
async fn spark_evo_v2_sql_door() {
    let cases: [(String, (i64, i64, i64, i64)); 5] = [
        ("i > 3000000000".to_string(), (300, 644850, 2000, 2299)),
        (format!("s2 = '{S}001234'"), (1, 1234, 1234, 1234)),
        ("addc IS NULL".to_string(), (2000, 1999000, 0, 1999)),
        ("dec > 1500.00".to_string(), (799, 1518100, 1501, 2299)),
        ("i > 1800".to_string(), (499, 1022950, 1801, 2299)),
    ];
    for (predicate, expected) in cases {
        assert_sql_matches_spark("evo_v2", "v10.metadata.json", &predicate, expected).await;
    }
}
