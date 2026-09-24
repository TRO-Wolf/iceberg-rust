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

fn metadata_json_names(dir: &Path) -> Vec<String> {
    let version_file = Regex::new(r"^v[0-9]+\.metadata\.json$").expect("regex");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read metadata dir")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .into_string()
                .expect("utf8")
        })
        .filter(|name| version_file.is_match(name) || name == "version-hint.text")
        .collect();
    names.sort();
    names
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
    assert_eq!(metadata_json_names(&metadata_dir), [
        "v1.metadata.json",
        "v2.metadata.json",
        "v3.metadata.json",
        "version-hint.text"
    ]);

    catalog.drop_table(&ident()).await.expect("drop");
    assert!(metadata_json_names(&metadata_dir).is_empty());
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    assert_eq!(std::fs::read(&data_file).expect("data kept"), b"data");
    assert_eq!(
        std::fs::read(&manifest).expect("manifest kept"),
        b"manifest"
    );
    assert!(table_dir.is_dir());
    assert!(metadata_dir.is_dir());
}

#[tokio::test]
async fn hadoop_drop_after_register_of_vn() {
    let warehouse = TempDir::new().expect("tempdir");
    let catalog = load_catalog(&warehouse, Some("hadoop"))
        .await
        .expect("load");
    catalog
        .create_namespace(&ident().namespace, HashMap::new())
        .await
        .expect("namespace");
    let table_location = warehouse.path().join("registered");
    let metadata = metadata_at(&table_location, schema());
    let v5 = format!("{}/metadata/v5.metadata.json", metadata.location());
    metadata
        .write_to(&catalog.file_io, &v5)
        .await
        .expect("write");
    let registered = catalog
        .register_table(&ident(), v5.clone())
        .await
        .expect("register");
    assert_eq!(location(&registered), v5);
    let metadata_dir = table_location.join("metadata");
    assert_eq!(metadata_json_names(&metadata_dir), ["v5.metadata.json"]);

    catalog.drop_table(&ident()).await.expect("drop");
    assert!(metadata_json_names(&metadata_dir).is_empty());
    assert!(!catalog.table_exists(&ident()).await.expect("exists"));
    assert!(table_location.is_dir());
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
    let metadata_dir = Path::new(&metadata_dir(&table)).to_path_buf();
    let near_misses = ["v1.metadata.json.bak", "other.metadata.json"];
    for name in near_misses {
        std::fs::write(metadata_dir.join(name), name).expect("near miss");
    }

    catalog.drop_table(&ident()).await.expect("drop");
    assert!(metadata_json_names(&metadata_dir).is_empty());
    for name in near_misses {
        assert_eq!(
            std::fs::read_to_string(metadata_dir.join(name)).expect("near miss kept"),
            name
        );
    }
}
