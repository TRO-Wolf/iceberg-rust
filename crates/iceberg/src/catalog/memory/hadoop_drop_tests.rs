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

use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;

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

async fn assert_drop_then_recreate_at_v1(catalog: &MemoryCatalog, form: &str) {
    let created = create(catalog, HashMap::new()).await.expect(form);
    commit_property(catalog, "a").await;
    commit_property(catalog, "b").await;
    let dir = metadata_dir(&created);
    catalog.drop_table(&ident()).await.expect(form);
    for name in [
        "v1.metadata.json",
        "v2.metadata.json",
        "v3.metadata.json",
        "version-hint.text",
    ] {
        let path = format!("{dir}/{name}");
        assert!(
            !catalog.file_io.exists(&path).await.expect(form),
            "{form}: {path} must not exist"
        );
    }

    let recreated = create(catalog, HashMap::new()).await.expect(form);
    assert_eq!(
        location(&recreated),
        format!("{dir}/v1.metadata.json"),
        "{form}"
    );
    let committed = commit_property(catalog, "c").await;
    assert_eq!(
        location(&committed),
        format!("{dir}/v2.metadata.json"),
        "{form}"
    );
}

#[tokio::test]
async fn hadoop_drop_then_recreate_over_every_accepted_warehouse_form() {
    for form in [
        "memory:///warehouse",
        "memory://warehouse",
        "memory:/warehouse",
        "/warehouse",
        "warehouse",
        "C:/warehouse",
        "C:\\warehouse",
    ] {
        let catalog = load_catalog_with(Arc::new(MemoryStorageFactory), form, Some("hadoop"))
            .await
            .expect(form);
        assert_drop_then_recreate_at_v1(&catalog, form).await;
    }

    let warehouse = TempDir::new().expect("tempdir");
    let absolute = warehouse.path().to_str().expect("utf8");
    let relative = absolute.trim_start_matches('/');
    for form in [
        absolute.to_string(),
        format!("file://{absolute}"),
        format!("file://{relative}"),
        format!("file:{absolute}"),
        format!("file:{relative}"),
    ] {
        let _ = std::fs::remove_dir_all(warehouse.path().join("ns"));
        let catalog = load_catalog_with(Arc::new(LocalFsStorageFactory), &form, Some("hadoop"))
            .await
            .expect(&form);
        assert_drop_then_recreate_at_v1(&catalog, &form).await;
    }
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
async fn uuid_mode_drop_of_registered_vn_pointer_deletes_only_the_pointer() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, None).await.expect("load");
    let table_location = warehouse.path().join("registered");
    register_hand_placed(&catalog, &table_location, "v3.metadata.json").await;
    let metadata_dir = table_location.join("metadata");
    let kept = [
        ("v1.metadata.json", "v1"),
        ("v2.metadata.json", "v2"),
        ("version-hint.text", "3"),
    ];
    for (name, content) in kept {
        std::fs::write(metadata_dir.join(name), content).expect(name);
    }

    catalog.drop_table(&ident()).await.expect("drop");
    assert_absent(&metadata_dir.join("v3.metadata.json"));
    for (name, content) in kept {
        assert_eq!(
            std::fs::read_to_string(metadata_dir.join(name)).expect(name),
            content
        );
    }
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
    let parsed_as_chain = [
        "v01.metadata.json",
        "v2.gz.metadata.json",
        "v1.metadata.json.gz",
        "v+1.metadata.json",
    ];
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
            std::fs::read_to_string(metadata_dir.join(name)).expect(name),
            name
        );
    }
}

const DELETE_BUDGET: usize = 64;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ListMode {
    #[default]
    PassThrough,
    Unsupported,
    Failing,
    Pending,
}

#[derive(Debug, Clone, Default)]
struct Counters {
    lists: Arc<AtomicUsize>,
    deletes: Arc<AtomicUsize>,
    failing_delete: Arc<Mutex<Option<&'static str>>>,
    list_prefixes: Arc<Mutex<Vec<String>>>,
}

impl Counters {
    fn lists(&self) -> usize {
        self.lists.load(Ordering::SeqCst)
    }

    fn deletes(&self) -> usize {
        self.deletes.load(Ordering::SeqCst)
    }

    fn list_prefixes(&self) -> Vec<String> {
        self.list_prefixes.lock().expect("lock").clone()
    }

    fn fail_delete_of(&self, name: &'static str) {
        *self.failing_delete.lock().expect("lock") = Some(name);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CountingStorage {
    #[serde(skip, default = "memory_storage")]
    inner: Arc<dyn Storage>,
    #[serde(skip)]
    mode: ListMode,
    #[serde(skip)]
    counters: Counters,
}

#[async_trait]
#[typetag::serde]
impl Storage for CountingStorage {
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
        if self.counters.deletes.fetch_add(1, Ordering::SeqCst) >= DELETE_BUDGET {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("delete budget exceeded: {path}"),
            ));
        }
        let failing = *self.counters.failing_delete.lock().expect("lock");
        if let Some(name) = failing
            && path.ends_with(&format!("/{name}"))
        {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("injected delete failure: {path}"),
            ));
        }
        self.inner.delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        self.inner.delete_prefix(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<FileInfo>> {
        self.counters.lists.fetch_add(1, Ordering::SeqCst);
        self.counters
            .list_prefixes
            .lock()
            .expect("lock")
            .push(prefix.to_string());
        let kind = match self.mode {
            ListMode::PassThrough => return self.inner.list(prefix).await,
            ListMode::Pending => return std::future::pending().await,
            ListMode::Unsupported => ErrorKind::FeatureUnsupported,
            ListMode::Failing => ErrorKind::Unexpected,
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
struct CountingStorageFactory {
    #[serde(skip)]
    mode: ListMode,
    #[serde(skip)]
    counters: Counters,
}

#[typetag::serde]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        Ok(Arc::new(CountingStorage {
            inner: memory_storage(),
            mode: self.mode,
            counters: self.counters.clone(),
        }))
    }
}

async fn counting_catalog(mode: ListMode) -> (MemoryCatalog, Counters) {
    let counters = Counters::default();
    let factory = CountingStorageFactory {
        mode,
        counters: counters.clone(),
    };
    let catalog = load_catalog_with(Arc::new(factory), "memory:///warehouse", Some("hadoop"))
        .await
        .expect("load");
    (catalog, counters)
}

async fn counting_catalog_at_v3(mode: ListMode) -> (MemoryCatalog, String, Counters) {
    let (catalog, counters) = counting_catalog(mode).await;
    let table = create(&catalog, HashMap::new()).await.expect("create");
    commit_property(&catalog, "a").await;
    commit_property(&catalog, "b").await;
    let dir = metadata_dir(&table);
    (catalog, dir, counters)
}

async fn file_exists(catalog: &MemoryCatalog, dir: &str, name: &str) -> bool {
    catalog
        .file_io
        .exists(format!("{dir}/{name}"))
        .await
        .expect("exists")
}

#[tokio::test]
async fn hadoop_drop_lists_metadata_once() {
    let (catalog, dir, counters) = counting_catalog_at_v3(ListMode::PassThrough).await;
    let (lists, deletes) = (counters.lists(), counters.deletes());
    let prefixes = counters.list_prefixes().len();
    catalog.drop_table(&ident()).await.expect("drop");
    assert_eq!(counters.lists() - lists, 1);
    assert_eq!(counters.deletes() - deletes, 4);
    assert_eq!(counters.list_prefixes()[prefixes..], [dir.as_str()]);
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
async fn hadoop_drop_without_listing_removes_only_current_file_and_hint() {
    let (catalog, dir, _) = counting_catalog_at_v3(ListMode::Unsupported).await;
    let v1 = format!("{dir}/v1.metadata.json");
    let v2 = format!("{dir}/v2.metadata.json");
    let v1_bytes = read_bytes(&catalog, &v1).await;
    let v2_bytes = read_bytes(&catalog, &v2).await;

    catalog.drop_table(&ident()).await.expect("drop");
    assert!(!file_exists(&catalog, &dir, "v3.metadata.json").await);
    assert!(!file_exists(&catalog, &dir, "version-hint.text").await);
    assert_eq!(read_bytes(&catalog, &v1).await, v1_bytes);
    assert_eq!(read_bytes(&catalog, &v2).await, v2_bytes);
    let err = create(&catalog, HashMap::new())
        .await
        .expect_err("re-create meets the leftover v1");
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    assert_eq!(read_bytes(&catalog, &v1).await, v1_bytes);
}

#[tokio::test]
async fn hadoop_drop_of_huge_registered_version_without_listing_is_bounded() {
    let (catalog, counters) = counting_catalog(ListMode::Unsupported).await;
    let pointer = "v2000000000.metadata.json";
    register_hand_placed(&catalog, Path::new("/warehouse/registered"), pointer).await;
    let (lists, deletes) = (counters.lists(), counters.deletes());

    catalog.drop_table(&ident()).await.expect("drop");
    assert_eq!(counters.deletes() - deletes, 2);
    assert_eq!(counters.lists() - lists, 1);
    assert!(!file_exists(&catalog, "/warehouse/registered/metadata", pointer).await);
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
}

#[tokio::test]
async fn hadoop_drop_propagates_other_list_errors() {
    let (catalog, dir, _) = counting_catalog_at_v3(ListMode::Failing).await;
    let err = catalog.drop_table(&ident()).await.expect_err("list fails");
    assert_eq!(err.kind(), ErrorKind::Unexpected);
    assert!(err.message().starts_with("injected list failure"), "{err}");
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    assert!(!file_exists(&catalog, &dir, "v3.metadata.json").await);
    for name in ["v1.metadata.json", "v2.metadata.json", "version-hint.text"] {
        assert!(file_exists(&catalog, &dir, name).await, "{name}");
    }
    assert_recreate_conflicts(&catalog).await;
}

async fn assert_recreate_conflicts(catalog: &MemoryCatalog) {
    let err = create(catalog, HashMap::new())
        .await
        .expect_err("re-create meets the leftover v1");
    assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
}

async fn drop_with_failing_delete_of(name: &'static str) -> (MemoryCatalog, String) {
    let (catalog, dir, counters) = counting_catalog_at_v3(ListMode::PassThrough).await;
    counters.fail_delete_of(name);
    let err = catalog
        .drop_table(&ident())
        .await
        .expect_err("delete fails");
    assert_eq!(err.kind(), ErrorKind::Unexpected);
    assert!(
        err.message().starts_with("injected delete failure: ")
            && err.message().ends_with(&format!("ns/t/metadata/{name}")),
        "{err}"
    );
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    let retry = catalog.drop_table(&ident()).await.expect_err("no pointer");
    assert_eq!(retry.kind(), ErrorKind::TableNotFound);
    (catalog, dir)
}

#[tokio::test]
async fn hadoop_drop_current_file_delete_error_leaves_chain_and_hint() {
    let (catalog, dir) = drop_with_failing_delete_of("v3.metadata.json").await;
    for name in [
        "v1.metadata.json",
        "v2.metadata.json",
        "v3.metadata.json",
        "version-hint.text",
    ] {
        assert!(file_exists(&catalog, &dir, name).await, "{name}");
    }
    assert_recreate_conflicts(&catalog).await;
}

#[tokio::test]
async fn hadoop_drop_chain_delete_error_leaves_rest_of_chain_and_hint() {
    let (catalog, dir) = drop_with_failing_delete_of("v1.metadata.json").await;
    assert!(!file_exists(&catalog, &dir, "v3.metadata.json").await);
    for name in ["v1.metadata.json", "version-hint.text"] {
        assert!(file_exists(&catalog, &dir, name).await, "{name}");
    }
    assert_recreate_conflicts(&catalog).await;
}

#[tokio::test]
async fn hadoop_drop_hint_delete_error_leaves_only_the_hint() {
    let (catalog, dir) = drop_with_failing_delete_of("version-hint.text").await;
    for name in ["v1.metadata.json", "v2.metadata.json", "v3.metadata.json"] {
        assert!(!file_exists(&catalog, &dir, name).await, "{name}");
    }
    assert_eq!(
        read_bytes(&catalog, &format!("{dir}/version-hint.text")).await,
        Bytes::from("3")
    );
    let recreated = create(&catalog, HashMap::new()).await.expect("recreate");
    assert_eq!(location(&recreated), format!("{dir}/v1.metadata.json"));
    assert_eq!(
        read_bytes(&catalog, &format!("{dir}/version-hint.text")).await,
        Bytes::from("1")
    );
}

#[tokio::test]
async fn hadoop_drop_cancelled_at_listing_leaves_chain_without_pointer() {
    let (catalog, dir, _) = counting_catalog_at_v3(ListMode::Pending).await;
    let outcome =
        tokio::time::timeout(Duration::from_millis(100), catalog.drop_table(&ident())).await;
    assert!(outcome.is_err(), "drop must still wait on the listing");
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    assert!(!file_exists(&catalog, &dir, "v3.metadata.json").await);
    for name in ["v1.metadata.json", "v2.metadata.json", "version-hint.text"] {
        assert!(file_exists(&catalog, &dir, name).await, "{name}");
    }
    let retry = catalog.drop_table(&ident()).await.expect_err("no pointer");
    assert_eq!(retry.kind(), ErrorKind::TableNotFound);
}

#[tokio::test]
async fn hadoop_drop_of_uuid_or_unparsable_pointer_deletes_only_the_pointer() {
    for (table, pointer) in [
        (
            "uuid",
            "00001-2cd22b57-5127-4198-92ba-e4e67c79821b.metadata.json",
        ),
        ("unparsable", "custom.json"),
    ] {
        let warehouse = TempDir::new().expect("tempdir");
        let catalog = load_catalog(&warehouse, Some("hadoop"))
            .await
            .expect("load");
        let table_location = warehouse.path().join(table);
        register_hand_placed(&catalog, &table_location, pointer).await;
        let metadata_dir = table_location.join("metadata");
        std::fs::write(metadata_dir.join("v1.metadata.json"), "v1").expect("v1");
        std::fs::write(metadata_dir.join("version-hint.text"), "1").expect("hint");

        catalog.drop_table(&ident()).await.expect("drop");
        assert_absent(&metadata_dir.join(pointer));
        assert_eq!(
            std::fs::read_to_string(metadata_dir.join("v1.metadata.json")).expect(table),
            "v1"
        );
        assert_eq!(
            std::fs::read_to_string(metadata_dir.join("version-hint.text")).expect(table),
            "1"
        );
    }
}

fn namespace(parts: &[&str]) -> NamespaceIdent {
    NamespaceIdent::from_strs(parts).expect("namespace")
}

async fn create_table_in(catalog: &MemoryCatalog, namespace: &NamespaceIdent) -> Table {
    catalog
        .create_table(
            namespace,
            TableCreation::builder()
                .name("t".to_string())
                .schema(schema())
                .build(),
        )
        .await
        .expect("create table")
}

#[tokio::test]
async fn hadoop_mode_drop_namespace_with_table_is_refused() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    let db = namespace(&["db"]);
    catalog
        .create_namespace(&db, HashMap::new())
        .await
        .expect("namespace");
    let table = create_table_in(&catalog, &db).await;

    let err = catalog
        .drop_namespace(&db)
        .await
        .expect_err("namespace holds a table");
    assert_eq!(err.kind(), ErrorKind::NamespaceNotEmpty);
    assert_eq!(err.message(), "Namespace db is not empty.");
    assert!(catalog.namespace_exists(&db).await.expect("exists"));
    let loaded = catalog
        .load_table(&TableIdent::new(db.clone(), "t".to_string()))
        .await
        .expect("table still loads");
    assert_eq!(location(&loaded), location(&table));
    assert!(location(&table).ends_with("/db/t/metadata/v1.metadata.json"));
    assert!(Path::new(&location(&table)).is_file());
}

async fn catalog_with_namespaces(
    warehouse: &TempDir,
    naming: Option<&str>,
    namespaces: &[&[&str]],
) -> MemoryCatalog {
    let catalog = load_catalog(warehouse, naming).await.expect("load");
    for parts in namespaces {
        catalog
            .create_namespace(&namespace(parts), HashMap::new())
            .await
            .expect("namespace");
    }
    catalog
}

#[tokio::test]
async fn hadoop_mode_drop_of_empty_namespace_succeeds() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = catalog_with_namespaces(&warehouse, Some("hadoop"), &[&["db"]]).await;
    catalog
        .drop_namespace(&namespace(&["db"]))
        .await
        .expect("drop");
    assert!(
        !catalog
            .namespace_exists(&namespace(&["db"]))
            .await
            .expect("exists")
    );
}

#[tokio::test]
async fn hadoop_mode_drop_namespace_refuses_a_table_in_a_descendant() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = catalog_with_namespaces(&warehouse, Some("hadoop"), &[&["a"], &["a", "b"]]).await;
    create_table_in(&catalog, &namespace(&["a", "b"])).await;

    for (parts, message) in [
        (&["a"][..], "Namespace a is not empty."),
        (&["a", "b"][..], "Namespace a.b is not empty."),
    ] {
        let err = catalog
            .drop_namespace(&namespace(parts))
            .await
            .expect_err(message);
        assert_eq!(err.kind(), ErrorKind::NamespaceNotEmpty);
        assert_eq!(err.message(), message);
    }
    for parts in [&["a"][..], &["a", "b"][..]] {
        assert!(
            catalog
                .namespace_exists(&namespace(parts))
                .await
                .expect("exists")
        );
    }
    assert!(
        catalog
            .table_exists(&TableIdent::new(namespace(&["a", "b"]), "t".to_string()))
            .await
            .expect("table exists")
    );
}

#[tokio::test]
async fn hadoop_mode_drop_namespace_with_only_an_empty_child_succeeds() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = catalog_with_namespaces(&warehouse, Some("hadoop"), &[&["a"], &["a", "b"]]).await;
    catalog
        .drop_namespace(&namespace(&["a"]))
        .await
        .expect("drop");
    for parts in [&["a"][..], &["a", "b"][..]] {
        assert!(
            !catalog
                .namespace_exists(&namespace(parts))
                .await
                .expect("exists")
        );
    }
}

#[tokio::test]
async fn hadoop_mode_drop_namespace_after_its_table_was_dropped_succeeds() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = catalog_with_namespaces(&warehouse, Some("hadoop"), &[&["db"]]).await;
    let db = namespace(&["db"]);
    create_table_in(&catalog, &db).await;
    catalog
        .drop_table(&TableIdent::new(db.clone(), "t".to_string()))
        .await
        .expect("drop table");
    catalog.drop_namespace(&db).await.expect("drop namespace");
    assert!(!catalog.namespace_exists(&db).await.expect("exists"));
}

#[tokio::test]
async fn uuid_mode_drop_namespace_with_table_still_succeeds() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = catalog_with_namespaces(&warehouse, None, &[&["db"]]).await;
    let db = namespace(&["db"]);
    let table = create_table_in(&catalog, &db).await;

    catalog.drop_namespace(&db).await.expect("drop namespace");
    assert!(!catalog.namespace_exists(&db).await.expect("exists"));
    let err = catalog
        .load_table(&TableIdent::new(db.clone(), "t".to_string()))
        .await
        .expect_err("pointer gone with the namespace");
    assert_eq!(err.kind(), ErrorKind::NamespaceNotFound);
    assert!(Path::new(&location(&table)).is_file());
}

#[tokio::test]
async fn hadoop_mode_drop_of_missing_namespace_keeps_the_not_found_error() {
    for parts in [&["missing"][..], &["db", "missing"][..]] {
        let hadoop_warehouse = TempDir::new().expect("tempdir");
        let hadoop = catalog_with_namespaces(&hadoop_warehouse, Some("hadoop"), &[&["db"]]).await;
        let uuid_warehouse = TempDir::new().expect("tempdir");
        let uuid = catalog_with_namespaces(&uuid_warehouse, None, &[&["db"]]).await;

        let err = hadoop
            .drop_namespace(&namespace(parts))
            .await
            .expect_err("missing namespace");
        let expected = uuid
            .drop_namespace(&namespace(parts))
            .await
            .expect_err("missing namespace");
        assert_eq!(err.kind(), ErrorKind::NamespaceNotFound);
        assert_eq!(err.message(), expected.message());
        assert_eq!(
            err.message(),
            format!("No such namespace: {:?}", namespace(parts))
        );
    }
}

#[tokio::test]
async fn hadoop_drop_keeps_nested_keys_holding_a_scheme_separator() {
    let catalog = load_catalog_with(
        Arc::new(MemoryStorageFactory),
        "memory:///warehouse",
        Some("hadoop"),
    )
    .await
    .expect("load");
    let table = create(&catalog, HashMap::new()).await.expect("create");
    commit_property(&catalog, "a").await;
    let dir = metadata_dir(&table);
    let bare = dir.trim_start_matches("memory:///");
    let nested = [
        format!("{dir}/x://{bare}/v1.metadata.json"),
        format!("{dir}/s3://b/v1.metadata.json"),
        format!("{dir}/x:/{bare}/v1.metadata.json"),
    ];
    for path in &nested {
        catalog
            .file_io
            .new_output(path)
            .expect("output")
            .write(Bytes::from(path.clone()))
            .await
            .expect("write nested key");
    }
    let listed: Vec<String> = catalog
        .file_io
        .list(&dir)
        .await
        .expect("list")
        .into_iter()
        .map(|file| file.location)
        .collect();
    for path in &nested {
        let key = path.trim_start_matches("memory:///");
        assert!(listed.iter().any(|location| location == key), "{key}");
    }

    catalog.drop_table(&ident()).await.expect("drop");
    for name in ["v1.metadata.json", "v2.metadata.json", "version-hint.text"] {
        assert!(!file_exists(&catalog, &dir, name).await, "{name}");
    }
    for path in &nested {
        assert_eq!(
            read_bytes(&catalog, path).await,
            Bytes::from(path.clone()),
            "{path}"
        );
    }
}
