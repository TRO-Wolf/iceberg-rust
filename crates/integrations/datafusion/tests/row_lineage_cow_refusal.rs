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
use std::sync::Arc;

use datafusion::error::DataFusionError;
use datafusion::execution::context::SessionContext;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    FormatVersion, NestedField, PrimitiveType, Schema, TableProperties, Transform, Type,
    UnboundPartitionSpec,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, NamespaceIdent, TableCreation, TableIdent,
};
use iceberg_datafusion::IcebergCatalogProvider;
use tempfile::TempDir;

fn leak_temp_path() -> String {
    let temp_dir = TempDir::new().expect("temp dir");
    let path = temp_dir.path().to_str().expect("utf8").to_string();
    std::mem::forget(temp_dir);
    path
}

async fn catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .with_storage_factory(Arc::new(iceberg::io::LocalFsStorageFactory))
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), leak_temp_path())]),
        )
        .await
        .expect("load catalog")
}

async fn v3_cow_ctx(ns: &str, tbl: &str) -> (SessionContext, Arc<MemoryCatalog>) {
    v3_cow_ctx_inner(ns, tbl, false, HashMap::new()).await
}

async fn v3_cow_ctx_with_format(
    ns: &str,
    tbl: &str,
    format: &str,
) -> (SessionContext, Arc<MemoryCatalog>) {
    let prop = TableProperties::PROPERTY_DEFAULT_FILE_FORMAT.to_string();
    v3_cow_ctx_inner(ns, tbl, false, HashMap::from([(prop, format.to_string())])).await
}

async fn v3_cow_ctx_inner(
    ns: &str,
    tbl: &str,
    partitioned: bool,
    properties: HashMap<String, String>,
) -> (SessionContext, Arc<MemoryCatalog>) {
    let iceberg_catalog = catalog().await;
    let namespace = NamespaceIdent::new(ns.to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");

    let mut fields =
        vec![NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into()];
    if partitioned {
        fields.push(
            NestedField::required(2, "category", Type::Primitive(PrimitiveType::String)).into(),
        );
        fields.push(NestedField::required(3, "val", Type::Primitive(PrimitiveType::String)).into());
    } else {
        fields.push(NestedField::required(2, "val", Type::Primitive(PrimitiveType::String)).into());
    }
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(fields)
        .build()
        .expect("schema");

    let location = leak_temp_path();
    let creation = if partitioned {
        let partition_spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(2, "category", Transform::Identity)
            .expect("identity(category)")
            .build();
        TableCreation::builder()
            .name(tbl.to_string())
            .location(location)
            .schema(schema)
            .partition_spec(partition_spec)
            .properties(properties)
            .format_version(FormatVersion::V3)
            .build()
    } else {
        TableCreation::builder()
            .name(tbl.to_string())
            .location(location)
            .schema(schema)
            .properties(properties)
            .format_version(FormatVersion::V3)
            .build()
    };
    iceberg_catalog
        .create_table(&namespace, creation)
        .await
        .expect("create v3 table");

    let client = Arc::new(iceberg_catalog);
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(client.clone())
            .await
            .expect("provider"),
    );
    let ctx = SessionContext::new();
    ctx.register_catalog("catalog", provider);
    (ctx, client)
}

async fn run_sql(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql)
        .await
        .unwrap_or_else(|error| panic!("plan `{sql}`: {error}"))
        .collect()
        .await
        .unwrap_or_else(|error| panic!("execute `{sql}`: {error}"));
}

async fn rewrite_refusal(
    ns: &str,
    format: Option<&str>,
    key: &str,
    value: &str,
) -> DataFusionError {
    let tbl = "t";
    let (ctx, client) = match format {
        None => v3_cow_ctx(ns, tbl).await,
        Some(name) => v3_cow_ctx_with_format(ns, tbl, name).await,
    };
    let planted = format!("INSERT INTO catalog.{ns}.{tbl} VALUES (1, 'a'), (2, 'b'), (3, 'c')");
    run_sql(&ctx, &planted).await;
    let ident = TableIdent::new(NamespaceIdent::new(ns.to_string()), tbl.to_string());
    let table = client.load_table(&ident).await.expect("load planted table");
    let tx = Transaction::new(&table);
    tx.update_table_properties()
        .set(key.to_string(), value.to_string())
        .apply(tx)
        .expect("stage property update")
        .commit(client.as_ref())
        .await
        .expect("commit property update");
    let delete = format!("DELETE FROM catalog.{ns}.{tbl} WHERE id = 2");
    let Err(err) = ctx.sql(&delete).await.expect("plan delete").collect().await else {
        panic!("delete over {key}={value} must refuse the rewrite");
    };
    err
}

#[tokio::test]
async fn cow_delete_refuses_bogus_parquet_codec_at_rewrite() {
    let err = rewrite_refusal(
        "lineage_cow_refuse_parquet_codec",
        None,
        TableProperties::PROPERTY_PARQUET_COMPRESSION_CODEC,
        "brotli",
    )
    .await;
    let DataFusionError::External(inner) = err else {
        panic!("bogus parquet codec must surface the iceberg refusal, got {err}");
    };
    let iceberg_err = inner
        .downcast_ref::<Error>()
        .expect("external wraps iceberg Error");
    assert_eq!(iceberg_err.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        iceberg_err.message(),
        "Invalid value for write.parquet.compression-codec: brotli"
    );
}

#[tokio::test]
async fn cow_delete_refuses_bogus_avro_codec_at_rewrite() {
    let err = rewrite_refusal(
        "lineage_cow_refuse_avro_codec",
        Some("avro"),
        "write.avro.compression-codec",
        "bogus",
    )
    .await;
    let DataFusionError::External(inner) = err else {
        panic!("bogus avro codec must surface the iceberg refusal, got {err}");
    };
    let iceberg_err = inner
        .downcast_ref::<Error>()
        .expect("external wraps iceberg Error");
    assert_eq!(iceberg_err.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        iceberg_err.message(),
        "Invalid value for write.avro.compression-codec: bogus"
    );
}

#[tokio::test]
async fn cow_delete_refuses_bogus_orc_stripe_at_rewrite() {
    let err = rewrite_refusal(
        "lineage_cow_refuse_orc_stripe",
        Some("orc"),
        TableProperties::PROPERTY_ORC_STRIPE_SIZE_BYTES,
        "abc",
    )
    .await;
    let DataFusionError::External(inner) = err else {
        panic!("bogus orc stripe must surface the iceberg refusal, got {err}");
    };
    let iceberg_err = inner
        .downcast_ref::<Error>()
        .expect("external wraps iceberg Error");
    assert_eq!(iceberg_err.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        iceberg_err.message(),
        "Invalid value for write.orc.stripe-size-bytes: abc"
    );
}

#[tokio::test]
async fn cow_delete_refuses_bogus_target_file_size_at_rewrite() {
    let err = rewrite_refusal(
        "lineage_cow_refuse_target_size",
        None,
        TableProperties::PROPERTY_WRITE_TARGET_FILE_SIZE_BYTES,
        "abc",
    )
    .await;
    let DataFusionError::External(inner) = err else {
        panic!("bogus target file size must surface the iceberg refusal, got {err}");
    };
    let iceberg_err = inner
        .downcast_ref::<Error>()
        .expect("external wraps iceberg Error");
    assert_eq!(iceberg_err.kind(), ErrorKind::DataInvalid);
    assert_eq!(iceberg_err.message(), "Invalid table properties");
}
