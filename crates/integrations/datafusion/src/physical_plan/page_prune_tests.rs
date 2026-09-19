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

use datafusion::arrow::array::Int32Array;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
use futures::TryStreamExt;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{NestedField, PrimitiveType, Schema as IcebergSchema, Type};
use iceberg::table::Table;
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use tempfile::TempDir;

use crate::IcebergCatalogProvider;
use crate::physical_plan::scan::{IcebergTableScan, ScanKnobs, build_table_scan};
use crate::physical_plan::scan_knobs::{ensure_iceberg_scan_options, scan_knobs_from_context};

const ROWS: i32 = 512;

struct Fixture {
    provider: Arc<IcebergCatalogProvider>,
    catalog: Arc<MemoryCatalog>,
    _warehouse: TempDir,
}

async fn fixture() -> Fixture {
    let warehouse = TempDir::new().expect("warehouse");
    let catalog = Arc::new(
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(
                    MEMORY_CATALOG_WAREHOUSE.to_string(),
                    warehouse.path().to_str().expect("utf8").to_string(),
                )]),
            )
            .await
            .expect("catalog"),
    );
    let namespace = NamespaceIdent::new("ns".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .expect("namespace");
    let schema = IcebergSchema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .expect("schema");
    catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("t".to_string())
                .location(format!("{}/t", warehouse.path().to_str().expect("utf8")))
                .schema(schema)
                .build(),
        )
        .await
        .expect("table");
    let provider = Arc::new(
        IcebergCatalogProvider::try_new(catalog.clone())
            .await
            .expect("catalog provider"),
    );
    Fixture {
        provider,
        catalog,
        _warehouse: warehouse,
    }
}

fn session(fixture: &Fixture, row_selection: bool) -> SessionContext {
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
    ctx.register_catalog("catalog", fixture.provider.clone());
    ctx
}

async fn insert_rows(ctx: &SessionContext, lo: i32, hi: i32) {
    let values = (lo..hi)
        .map(|i| format!("({i}, 'v{i}')"))
        .collect::<Vec<_>>()
        .join(", ");
    ctx.sql(&format!("INSERT INTO catalog.ns.t VALUES {values}"))
        .await
        .expect("insert plan")
        .collect()
        .await
        .expect("insert");
}

async fn sql_filtered_ids(ctx: &SessionContext) -> Vec<i32> {
    let batches = ctx
        .sql("SELECT id FROM catalog.ns.t WHERE id >= 400 ORDER BY id")
        .await
        .expect("query plan")
        .collect()
        .await
        .expect("query");
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id int")
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn multi_partition_filtered_scan_row_selection_on_matches_off() {
    let fixture = fixture().await;
    let seed = session(&fixture, false);
    insert_rows(&seed, 0, 300).await;
    insert_rows(&seed, 300, ROWS).await;
    drop(seed);

    let off = session(&fixture, false);
    let off_ids = sql_filtered_ids(&off).await;
    let on = session(&fixture, true);
    let on_ids = sql_filtered_ids(&on).await;

    assert_eq!(off_ids.len() as i32, ROWS - 400);
    assert_eq!(off_ids, on_ids);
}

#[tokio::test]
async fn single_stream_filtered_scan_row_selection_on_matches_off() {
    let fixture = fixture().await;
    let seed = session(&fixture, false);
    insert_rows(&seed, 0, ROWS).await;
    drop(seed);

    let table = fixture
        .catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new("ns".to_string()),
            "t".to_string(),
        ))
        .await
        .expect("load table");
    let arrow_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));
    let filters = [col("id").gt_eq(lit(400i32))];

    async fn single_stream_rows(
        table: &Table,
        arrow_schema: &SchemaRef,
        filters: &[Expr],
        row_selection: bool,
    ) -> usize {
        let scan = IcebergTableScan::new(
            table.clone(),
            None,
            arrow_schema.clone(),
            None,
            filters,
            None,
        )
        .expect("scan");
        let mut config = SessionConfig::new();
        ensure_iceberg_scan_options(&mut config);
        config
            .options_mut()
            .set(
                "iceberg.row_selection_enabled",
                if row_selection { "true" } else { "false" },
            )
            .expect("set extension key");
        let task_ctx = SessionStateBuilder::new()
            .with_config(config)
            .build()
            .task_ctx();
        let stream = scan.execute(0, task_ctx).expect("execute");
        let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect");
        batches.iter().map(|b| b.num_rows()).sum()
    }

    let off_rows = single_stream_rows(&table, &arrow_schema, &filters, false).await;
    let on_rows = single_stream_rows(&table, &arrow_schema, &filters, true).await;

    assert_eq!(off_rows as i32, ROWS - 400);
    assert_eq!(off_rows, on_rows);
}

#[tokio::test]
async fn single_stream_scan_builder_receives_row_selection_knob() {
    let fixture = fixture().await;
    let seed = session(&fixture, false);
    insert_rows(&seed, 0, ROWS).await;
    drop(seed);

    let table = fixture
        .catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new("ns".to_string()),
            "t".to_string(),
        ))
        .await
        .expect("load table");

    let scan_on = build_table_scan(
        &table,
        None,
        vec!["id".to_string(), "data".to_string()],
        None,
        ScanKnobs {
            row_selection_enabled: true,
            ..Default::default()
        },
    )
    .expect("scan on");
    assert!(scan_on.row_selection_enabled());

    let scan_off = build_table_scan(
        &table,
        None,
        vec!["id".to_string(), "data".to_string()],
        None,
        ScanKnobs {
            row_selection_enabled: false,
            ..Default::default()
        },
    )
    .expect("scan off");
    assert!(!scan_off.row_selection_enabled());
}

#[test]
fn scan_knobs_from_context_wires_row_selection_enabled() {
    let mut config = SessionConfig::new();
    ensure_iceberg_scan_options(&mut config);
    let state = SessionStateBuilder::new()
        .with_config(config.clone())
        .build();
    assert!(
        scan_knobs_from_context(&state.task_ctx()).row_selection_enabled,
        "row selection must default on"
    );

    config
        .options_mut()
        .set("iceberg.row_selection_enabled", "false")
        .expect("set extension key");
    let state = SessionStateBuilder::new().with_config(config).build();
    assert!(
        !scan_knobs_from_context(&state.task_ctx()).row_selection_enabled,
        "iceberg.row_selection_enabled=false must reach the knobs"
    );
}

#[tokio::test]
async fn plan_carries_row_selection_enabled_to_multi_partition_path() {
    let fixture = fixture().await;
    let table = fixture
        .catalog
        .load_table(&TableIdent::new(
            NamespaceIdent::new("ns".to_string()),
            "t".to_string(),
        ))
        .await
        .expect("load table");
    let arrow_schema =
        Arc::new(schema_to_arrow_schema(table.metadata().current_schema()).expect("arrow schema"));

    let scan_on = IcebergTableScan::plan(
        table.clone(),
        None,
        arrow_schema.clone(),
        None,
        &[],
        None,
        ScanKnobs {
            row_selection_enabled: true,
            ..Default::default()
        },
    )
    .await
    .expect("plan on");
    assert!(scan_on.row_selection_enabled);

    let scan_off = IcebergTableScan::plan(table, None, arrow_schema, None, &[], None, ScanKnobs {
        row_selection_enabled: false,
        ..Default::default()
    })
    .await
    .expect("plan off");
    assert!(!scan_off.row_selection_enabled);
}
