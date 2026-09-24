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

use super::*;

#[derive(Debug, Clone, Default)]
struct ListGate {
    armed: Arc<AtomicBool>,
    reached: Arc<Notify>,
    release: Arc<Notify>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GatedListStorage {
    #[serde(skip, default = "memory_storage")]
    inner: Arc<dyn Storage>,
    #[serde(skip)]
    gate: ListGate,
}

#[async_trait]
#[typetag::serde]
impl Storage for GatedListStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        self.inner.metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        self.inner.read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.inner.reader(path).await
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        self.inner.write(path, bs).await
    }

    async fn write_new(&self, path: &str, bs: Bytes) -> Result<()> {
        self.inner.write_new(path, bs).await
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        self.inner.writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        self.inner.delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        if self.gate.armed.swap(false, Ordering::SeqCst) {
            self.gate.reached.notify_one();
            self.gate.release.notified().await;
        }
        self.inner.list(prefix).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GatedListStorageFactory {
    #[serde(skip)]
    gate: ListGate,
}

#[typetag::serde]
impl StorageFactory for GatedListStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(GatedListStorage {
            inner: memory_storage(),
            gate: self.gate.clone(),
        }))
    }
}

#[tokio::test]
async fn hadoop_create_racing_a_drop_keeps_its_v1() {
    for version in [1, 2] {
        let gate = ListGate::default();
        let factory = GatedListStorageFactory { gate: gate.clone() };
        let catalog = Arc::new(
            load_catalog_with(Arc::new(factory), "memory:///warehouse", Some("hadoop"))
                .await
                .expect("load"),
        );
        let table = create(&catalog, HashMap::new()).await.expect("create");
        for value in 1..version {
            commit_property(&catalog, &value.to_string()).await;
        }
        let dir = metadata_dir(&table);

        gate.armed.store(true, Ordering::SeqCst);
        let dropper = tokio::spawn({
            let catalog = catalog.clone();
            async move { catalog.drop_table(&ident()).await }
        });
        gate.reached.notified().await;
        let creator = tokio::spawn({
            let catalog = catalog.clone();
            async move { create(&catalog, HashMap::new()).await }
        });
        for _ in 0..64 {
            tokio::task::yield_now().await;
        }
        let created_during_list = creator.is_finished();
        gate.release.notify_one();

        dropper.await.expect("drop task").expect("drop");
        let recreated = creator.await.expect("create task").expect("create");
        let loaded = catalog.load_table(&ident()).await.expect("load");
        assert_eq!(location(&loaded), format!("{dir}/v1.metadata.json"));
        assert_eq!(location(&recreated), location(&loaded));
        assert!(
            catalog
                .file_io
                .exists(location(&loaded))
                .await
                .expect("exists"),
            "v{version}"
        );
        assert_eq!(
            read_bytes(&catalog, &format!("{dir}/version-hint.text")).await,
            Bytes::from("1"),
            "v{version}"
        );
        assert!(
            !created_during_list,
            "v{version}: the create must wait for the drop's cleanup"
        );
    }
}
