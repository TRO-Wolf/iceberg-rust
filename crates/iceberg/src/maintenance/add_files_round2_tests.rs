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

use std::collections::HashMap;

use super::add_files::{AddFiles, AddFilesSource};
use super::add_files_datafile::unescape_hive_path_name;
use super::add_files_tests::{
    create_table, id_v_cat_schema, live_data_files, local_fs_catalog, long_column, source_root,
    string_column, write_source_file,
};
use crate::Catalog;
use crate::spec::{FormatVersion, Literal};

#[test]
fn the_hive_path_unescape_is_spark_s_unescape_path_name() {
    for (raw, expected) in [
        ("a b", "a b"),
        ("a+b", "a+b"),
        ("a%20b", "a b"),
        ("a%2Fb", "a/b"),
        ("a%25b", "a%b"),
        ("caf%C3%A9", "caf\u{c3}\u{a9}"),
        ("caf\u{e9}", "caf\u{e9}"),
        ("a%zzb", "a%zzb"),
        ("a%2", "a%2"),
        ("a%", "a%"),
        ("%41", "A"),
        ("a%2Gb", "a%2Gb"),
        ("100%", "100%"),
        ("%%20", "% "),
        ("a%2520b", "a%20b"),
        ("%2f", "/"),
        ("%00", "\u{0}"),
    ] {
        assert_eq!(
            unescape_hive_path_name(raw),
            expected,
            "ExternalCatalogUtils.unescapePathName({raw:?})"
        );
    }
}

async fn adopt_escaped_dirs(name: &str, dirs: &[&str]) -> Vec<(String, Option<Literal>)> {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, name);
    for (index, dir) in dirs.iter().enumerate() {
        write_source_file(&table, &format!("{root}/{dir}/part-00000.parquet"), &[
            ("id", long_column(&[index as i64])),
            ("v", string_column(&["a"])),
        ])
        .await;
    }
    AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .execute(&catalog)
        .await
        .expect("adopt the escaped hive directories");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    live_data_files(&table)
        .await
        .into_iter()
        .map(|file| {
            let directory = file
                .file_path()
                .strip_prefix(&format!("{root}/"))
                .and_then(|rest| rest.split('/').next())
                .expect("the adopted path stays under the source root")
                .to_string();
            let value = file.partition().iter().next().flatten().cloned();
            (directory, value)
        })
        .collect()
}

#[tokio::test]
async fn a_percent_escaped_hive_directory_adopts_spark_s_unescaped_value() {
    let adopted = adopt_escaped_dirs("escaped", &[
        "cat=a%20b",
        "cat=caf%C3%A9",
        "cat=a%2Fb",
        "cat=a+b",
        "cat=a%25b",
        "cat=a%zzb",
        "cat=a%2",
    ])
    .await;
    let by_directory: HashMap<String, Option<Literal>> = adopted.into_iter().collect();
    for (directory, expected) in [
        ("cat=a%20b", "a b"),
        ("cat=caf%C3%A9", "caf\u{c3}\u{a9}"),
        ("cat=a%2Fb", "a/b"),
        ("cat=a+b", "a+b"),
        ("cat=a%25b", "a%b"),
        ("cat=a%zzb", "a%zzb"),
        ("cat=a%2", "a%2"),
    ] {
        assert_eq!(
            by_directory.get(directory),
            Some(&Some(Literal::string(expected))),
            "oracle cell G: {directory} adopts {expected:?}"
        );
    }
}

async fn adopt_with_filter(name: &str, dirs: &[&str], filter: &str) -> Vec<String> {
    let (catalog, temp_dir) = local_fs_catalog().await;
    let table = create_table(&catalog, id_v_cat_schema(), Some("cat"), FormatVersion::V2).await;
    let root = source_root(&temp_dir, name);
    for (index, dir) in dirs.iter().enumerate() {
        write_source_file(&table, &format!("{root}/{dir}/part-00000.parquet"), &[
            ("id", long_column(&[index as i64])),
            ("v", string_column(&["a"])),
        ])
        .await;
    }
    AddFiles::new(table.clone(), AddFilesSource::Directory(root.clone()))
        .partition_filter(HashMap::from([("cat".to_string(), filter.to_string())]))
        .execute(&catalog)
        .await
        .expect("adopt under the partition filter");
    let table = catalog
        .load_table(table.identifier())
        .await
        .expect("reload table");
    live_data_files(&table)
        .await
        .into_iter()
        .map(|file| {
            file.file_path()
                .strip_prefix(&format!("{root}/"))
                .and_then(|rest| rest.split('/').next())
                .expect("the adopted path stays under the source root")
                .to_string()
        })
        .collect()
}

#[tokio::test]
async fn a_partition_filter_matches_the_unescaped_hive_value() {
    let dirs = ["cat=a%20b", "cat=a%2520b", "cat=a%2Fb"];
    assert_eq!(
        adopt_with_filter("filter-unescaped", &dirs, "a b").await,
        vec!["cat=a%20b".to_string()],
        "oracle cell B: map('cat','a b') selects the directory whose UNESCAPED value is 'a b'"
    );
    assert_eq!(
        adopt_with_filter("filter-raw", &dirs, "a%20b").await,
        vec!["cat=a%2520b".to_string()],
        "oracle cell B: map('cat','a%20b') selects cat=a%2520b, not the raw directory text"
    );
    assert_eq!(
        adopt_with_filter("filter-slash", &dirs, "a/b").await,
        vec!["cat=a%2Fb".to_string()],
        "oracle cell B: a value Spark had to escape is matched unescaped"
    );
}
