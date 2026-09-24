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
use crate::io::MemoryStorageFactory;

#[track_caller]
fn assert_all_absent(dir: &Path, names: &[&str]) {
    for name in names {
        assert_absent(&dir.join(name));
    }
}

#[track_caller]
fn assert_all_files(dir: &Path, names: &[&str]) {
    for name in names {
        let path = dir.join(name);
        assert!(path.is_file(), "{} must exist", path.display());
    }
}

#[tokio::test]
async fn hadoop_drop_then_recreate_starts_at_v1() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    create(&catalog, HashMap::new()).await.expect("create");
    commit_property(&catalog, "a").await;
    let dropped = commit_property(&catalog, "b").await;
    assert!(location(&dropped).ends_with("/metadata/v3.metadata.json"));
    catalog.drop_table(&ident()).await.expect("drop");

    let recreated = create(&catalog, HashMap::new()).await.expect("recreate");
    assert!(
        location(&recreated).ends_with("/metadata/v1.metadata.json"),
        "{}",
        location(&recreated)
    );
    let loaded = catalog.load_table(&ident()).await.expect("load");
    assert_eq!(location(&loaded), location(&recreated));
    assert_eq!(hint(&loaded), "1");
    let committed = commit_property(&catalog, "c").await;
    assert!(
        location(&committed).ends_with("/metadata/v2.metadata.json"),
        "{}",
        location(&committed)
    );
    assert_eq!(hint(&committed), "2");
}

async fn assert_drop_then_recreate_at_v1(catalog: &MemoryCatalog) {
    let created = create(catalog, HashMap::new()).await.expect("create");
    commit_property(catalog, "a").await;
    commit_property(catalog, "b").await;
    let dir = metadata_dir(&created);
    catalog.drop_table(&ident()).await.expect("drop");
    for name in [
        "v1.metadata.json",
        "v2.metadata.json",
        "v3.metadata.json",
        "version-hint.text",
    ] {
        let path = format!("{dir}/{name}");
        assert!(
            !catalog.file_io.exists(&path).await.expect("exists"),
            "{path}"
        );
    }

    let recreated = create(catalog, HashMap::new()).await.expect("recreate");
    assert_eq!(location(&recreated), format!("{dir}/v1.metadata.json"));
    let committed = commit_property(catalog, "c").await;
    assert_eq!(location(&committed), format!("{dir}/v2.metadata.json"));
}

#[tokio::test]
async fn hadoop_drop_then_recreate_with_scheme_qualified_warehouses() {
    let warehouse = TempDir::new().expect("tempdir");
    let file_warehouse = format!("file://{}", warehouse.path().to_str().expect("utf8"));
    let catalog = load_catalog_with(
        Arc::new(LocalFsStorageFactory),
        &file_warehouse,
        Some("hadoop"),
    )
    .await
    .expect("load");
    assert_drop_then_recreate_at_v1(&catalog).await;

    let catalog = load_catalog_with(
        Arc::new(MemoryStorageFactory),
        "memory:///warehouse",
        Some("hadoop"),
    )
    .await
    .expect("load");
    assert_drop_then_recreate_at_v1(&catalog).await;
}

#[tokio::test]
async fn hadoop_drop_removes_chain_and_hint_keeps_data() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let table = create(&catalog, HashMap::new()).await.expect("create");
    let table_dir = Path::new(table.metadata().location()).to_path_buf();
    let data_file = table_dir.join("data/00000-0-data.parquet");
    std::fs::create_dir_all(data_file.parent().expect("parent")).expect("data dir");
    std::fs::write(&data_file, b"data").expect("data file");
    let manifest = table_dir.join("metadata/snap-1-1-manifest-list.avro");
    std::fs::write(&manifest, b"manifest").expect("manifest");
    commit_property(&catalog, "a").await;
    commit_property(&catalog, "b").await;
    let metadata_dir = table_dir.join("metadata");
    let chain = [
        "v1.metadata.json",
        "v2.metadata.json",
        "v3.metadata.json",
        "version-hint.text",
    ];
    assert_all_files(&metadata_dir, &chain);

    catalog.drop_table(&ident()).await.expect("drop");
    assert_all_absent(&metadata_dir, &chain);
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    assert_eq!(std::fs::read(&data_file).expect("data kept"), b"data");
    assert_eq!(
        std::fs::read(&manifest).expect("manifest kept"),
        b"manifest"
    );
    assert!(table_dir.is_dir());
    assert!(metadata_dir.is_dir());
}

async fn register_hand_placed(catalog: &MemoryCatalog, table_location: &Path, name: &str) {
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    let metadata = metadata_at(table_location, schema());
    let pointer = format!("{}/metadata/{name}", metadata.location());
    metadata
        .write_to(&catalog.file_io, &pointer)
        .await
        .expect("write");
    let registered = catalog
        .register_table(&ident(), pointer.clone())
        .await
        .expect("register");
    assert_eq!(location(&registered), pointer);
}

#[tokio::test]
async fn hadoop_drop_after_register_of_vn() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let table_location = warehouse.path().join("registered");
    register_hand_placed(&catalog, &table_location, "v5.metadata.json").await;
    let metadata_dir = table_location.join("metadata");
    assert_all_files(&metadata_dir, &["v5.metadata.json"]);
    assert_all_absent(&metadata_dir, &[
        "v1.metadata.json",
        "v2.metadata.json",
        "v3.metadata.json",
        "v4.metadata.json",
        "version-hint.text",
    ]);

    catalog.drop_table(&ident()).await.expect("drop");
    assert_all_absent(&metadata_dir, &["v5.metadata.json", "version-hint.text"]);
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    assert!(table_location.is_dir());
}

#[tokio::test]
async fn hadoop_drop_of_registered_huge_version_completes() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let table_location = warehouse.path().join("registered");
    let pointer = "v2000000000.metadata.json";
    register_hand_placed(&catalog, &table_location, pointer).await;
    let metadata_dir = table_location.join("metadata");
    std::fs::write(metadata_dir.join("version-hint.text"), "2000000000").expect("hint");
    std::fs::write(metadata_dir.join("v1.metadata.json"), "v1").expect("v1");

    let catalog = Arc::new(catalog);
    let dropper = catalog.clone();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let _ = sender.send(runtime.block_on(dropper.drop_table(&ident())));
    });
    tokio::time::timeout(Duration::from_secs(10), receiver)
        .await
        .expect("drop finishes within 10s")
        .expect("drop thread")
        .expect("drop");
    assert_all_absent(&metadata_dir, &[
        pointer,
        "version-hint.text",
        "v1.metadata.json",
    ]);
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
}

#[tokio::test]
async fn uuid_drop_leaves_hand_placed_hadoop_files() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, None).await.expect("load");
    let table = create(&catalog, HashMap::new()).await.expect("create");
    let committed = commit_property(&catalog, "a").await;
    let metadata_dir = Path::new(&metadata_dir(&table)).to_path_buf();
    let v1 = metadata_dir.join("v1.metadata.json");
    let hint = metadata_dir.join("version-hint.text");
    std::fs::write(&v1, b"v1").expect("v1");
    std::fs::write(&hint, b"1").expect("hint");

    catalog.drop_table(&ident()).await.expect("drop");
    assert_absent(Path::new(&location(&committed)));
    assert!(Path::new(&location(&table)).is_file());
    assert_eq!(std::fs::read(&v1).expect("v1 kept"), b"v1");
    assert_eq!(std::fs::read(&hint).expect("hint kept"), b"1");
}

#[tokio::test]
async fn hadoop_drop_leaves_near_miss_names() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let table = create(&catalog, HashMap::new()).await.expect("create");
    commit_property(&catalog, "a").await;
    commit_property(&catalog, "b").await;
    let metadata_dir = Path::new(&metadata_dir(&table)).to_path_buf();
    std::fs::create_dir(metadata_dir.join("sub")).expect("sub dir");
    let near_misses = [
        "v0.metadata.json",
        "V1.metadata.json",
        "v1.metadata.json.bak",
        "v4.metadata.json",
        "00001-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
        "sub/v1.metadata.json",
        "other.metadata.json",
    ];
    let parsed_as_chain = ["v01.metadata.json", "v2.gz.metadata.json"];
    for name in near_misses.iter().chain(&parsed_as_chain) {
        std::fs::write(metadata_dir.join(name), name).expect("hand-placed file");
    }

    catalog.drop_table(&ident()).await.expect("drop");
    assert_all_absent(&metadata_dir, &[
        "v1.metadata.json",
        "v2.metadata.json",
        "v3.metadata.json",
        "version-hint.text",
    ]);
    assert_all_absent(&metadata_dir, &parsed_as_chain);
    for name in near_misses {
        assert_eq!(
            std::fs::read_to_string(metadata_dir.join(name)).expect("near miss kept"),
            name
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ListFailingStorage {
    #[serde(skip, default = "memory_storage")]
    inner: Arc<dyn Storage>,
    #[serde(skip)]
    unsupported: bool,
}

#[async_trait]
#[typetag::serde]
impl Storage for ListFailingStorage {
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
        let kind = if self.unsupported {
            ErrorKind::FeatureUnsupported
        } else {
            ErrorKind::Unexpected
        };
        Err(Error::new(kind, format!("injected list failure: {prefix}")))
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(Arc::new(self.clone()), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(Arc::new(self.clone()), path.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ListFailingStorageFactory {
    #[serde(skip)]
    unsupported: bool,
}

#[typetag::serde]
impl StorageFactory for ListFailingStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(ListFailingStorage {
            inner: memory_storage(),
            unsupported: self.unsupported,
        }))
    }
}

async fn list_failing_catalog_at_v3(unsupported: bool) -> (MemoryCatalog, String) {
    let catalog = load_catalog_with(
        Arc::new(ListFailingStorageFactory { unsupported }),
        "memory:///warehouse",
        Some("hadoop"),
    )
    .await
    .expect("load");
    let table = create(&catalog, HashMap::new()).await.expect("create");
    commit_property(&catalog, "a").await;
    commit_property(&catalog, "b").await;
    let dir = metadata_dir(&table);
    (catalog, dir)
}

async fn file_exists(catalog: &MemoryCatalog, dir: &str, name: &str) -> bool {
    catalog
        .file_io
        .exists(format!("{dir}/{name}"))
        .await
        .expect("exists")
}

#[tokio::test]
async fn hadoop_drop_walks_versions_when_listing_is_unsupported() {
    let (catalog, dir) = list_failing_catalog_at_v3(true).await;
    catalog.drop_table(&ident()).await.expect("drop");
    for name in [
        "v1.metadata.json",
        "v2.metadata.json",
        "v3.metadata.json",
        "version-hint.text",
    ] {
        assert!(!file_exists(&catalog, &dir, name).await, "{name}");
    }
    let recreated = create(&catalog, HashMap::new()).await.expect("recreate");
    assert_eq!(location(&recreated), format!("{dir}/v1.metadata.json"));
}

#[tokio::test]
async fn hadoop_drop_propagates_other_list_errors() {
    let (catalog, dir) = list_failing_catalog_at_v3(false).await;
    let err = catalog.drop_table(&ident()).await.expect_err("list fails");
    assert_eq!(err.kind(), ErrorKind::Unexpected);
    assert!(err.message().starts_with("injected list failure"), "{err}");
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    assert!(!file_exists(&catalog, &dir, "v3.metadata.json").await);
    for name in ["v1.metadata.json", "v2.metadata.json", "version-hint.text"] {
        assert!(file_exists(&catalog, &dir, name).await, "{name}");
    }
}
