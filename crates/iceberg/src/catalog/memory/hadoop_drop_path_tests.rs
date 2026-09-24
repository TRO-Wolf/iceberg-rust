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

async fn register_chain_at(catalog: &MemoryCatalog, name: &str, table_location: &str) -> String {
    let metadata = metadata_at(Path::new(table_location), schema());
    let dir = format!("{}/metadata", metadata.location());
    for version in 1..=2 {
        metadata
            .write_to(&catalog.file_io, &format!("{dir}/v{version}.metadata.json"))
            .await
            .expect("write chain");
    }
    catalog
        .file_io
        .new_output(format!("{dir}/version-hint.text"))
        .expect("hint output")
        .write(Bytes::from("2"))
        .await
        .expect("hint");
    catalog
        .register_table(
            &TableIdent::new(ident().namespace, name.to_string()),
            format!("{dir}/v2.metadata.json"),
        )
        .await
        .expect("register");
    dir
}

const CHAIN: [&str; 3] = ["v1.metadata.json", "v2.metadata.json", "version-hint.text"];

#[tokio::test]
async fn hadoop_drop_keeps_a_scheme_variant_sibling_directory() {
    for (dropped, kept) in [
        ("/warehouse/ns/t", "file:/warehouse/ns/t"),
        ("file:/warehouse/ns/t", "/warehouse/ns/t"),
    ] {
        let catalog = load_catalog_with(
            Arc::new(MemoryStorageFactory),
            "memory:///other",
            Some("hadoop"),
        )
        .await
        .expect("load");
        catalog
            .create_namespace(&ident().namespace, HashMap::new())
            .await
            .expect("namespace");
        let dropped_dir = register_chain_at(&catalog, "dropped", dropped).await;
        let kept_dir = register_chain_at(&catalog, "kept", kept).await;
        let mut kept_bytes = Vec::new();
        for name in CHAIN {
            kept_bytes.push(read_bytes(&catalog, &format!("{kept_dir}/{name}")).await);
        }

        catalog
            .drop_table(&TableIdent::new(ident().namespace, "dropped".to_string()))
            .await
            .expect(dropped);
        for name in CHAIN {
            assert!(
                !file_exists(&catalog, &dropped_dir, name).await,
                "{dropped_dir}/{name}"
            );
        }
        for (name, bytes) in CHAIN.iter().zip(&kept_bytes) {
            assert_eq!(
                &read_bytes(&catalog, &format!("{kept_dir}/{name}")).await,
                bytes,
                "{kept_dir}/{name}"
            );
        }
        let kept_table = catalog
            .load_table(&TableIdent::new(ident().namespace, "kept".to_string()))
            .await
            .expect(kept);
        assert_eq!(
            location(&kept_table),
            format!("{kept_dir}/v2.metadata.json")
        );
    }
}
