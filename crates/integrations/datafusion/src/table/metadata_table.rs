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

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::inspect::{
    EntriesTable, FilesTable, ManifestsTable, MetadataTableType, PartitionsTable,
    PositionDeletesTable,
};
use iceberg::table::Table;
use iceberg::{Error, ErrorKind, Result};

use crate::physical_plan::metadata_scan::IcebergMetadataScan;
use crate::to_datafusion_error;

/// Represents a [`TableProvider`] for the Iceberg [`Catalog`],
/// managing access to a [`MetadataTable`].
#[derive(Debug, Clone)]
pub struct IcebergMetadataTableProvider {
    pub(crate) table: Table,
    pub(crate) r#type: MetadataTableType,
    pub(crate) snapshot_id: Option<i64>,
    /// Arrow schema of the metadata table, converted eagerly at construction.
    ///
    /// The `TableProvider::schema` trait method is infallible, but the Iceberg → Arrow
    /// schema conversion is fallible. Resolving it here lets `schema()` return an
    /// already-validated schema instead of unwrapping the conversion (which would panic
    /// inside a trait method DataFusion calls).
    pub(crate) schema: ArrowSchemaRef,
}

impl IcebergMetadataTableProvider {
    /// Builds a metadata-table provider, resolving the Arrow schema for `r#type` up front.
    ///
    /// Returns an error if the metadata table's Iceberg schema cannot be represented in
    /// Arrow, so the panic surface never reaches the infallible [`TableProvider::schema`].
    pub(crate) fn try_new(
        table: Table,
        r#type: MetadataTableType,
        snapshot_id: Option<i64>,
    ) -> Result<Self> {
        let metadata_table = table.inspect();
        let schema = match r#type {
            MetadataTableType::Snapshots => metadata_table.snapshots().schema(),
            MetadataTableType::Manifests => metadata_table.manifests().schema(),
            MetadataTableType::Files => FilesTable::try_all(&table)?.schema(),
            MetadataTableType::DataFiles => FilesTable::try_data(&table)?.schema(),
            MetadataTableType::DeleteFiles => FilesTable::try_deletes(&table)?.schema(),
            MetadataTableType::Entries => EntriesTable::try_new(&table)?.schema(),
            MetadataTableType::AllFiles => FilesTable::try_all_files(&table)?.schema(),
            MetadataTableType::AllDataFiles => FilesTable::try_all_data_files(&table)?.schema(),
            MetadataTableType::AllDeleteFiles => FilesTable::try_all_delete_files(&table)?.schema(),
            MetadataTableType::AllEntries => EntriesTable::try_all(&table)?.schema(),
            MetadataTableType::History => metadata_table.history().schema(),
            MetadataTableType::Refs => metadata_table.refs().schema(),
            MetadataTableType::MetadataLogEntries => metadata_table.metadata_log_entries().schema(),
            MetadataTableType::Partitions => PartitionsTable::try_new(&table)?.schema(),
            MetadataTableType::AllManifests => metadata_table.all_manifests().schema(),
            MetadataTableType::PositionDeletes => PositionDeletesTable::try_new(&table)?.schema(),
        };
        let schema = Arc::new(schema_to_arrow_schema(&schema)?);
        Ok(Self {
            table,
            r#type,
            snapshot_id,
            schema,
        })
    }
}

#[async_trait]
impl TableProvider for IcebergMetadataTableProvider {
    fn schema(&self) -> ArrowSchemaRef {
        // Resolved (and validated) eagerly in `try_new`; this trait method must not fail.
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(IcebergMetadataScan::new(
            self.clone(),
            projection,
        )?))
    }
}

fn snapshot_scope_refused(table_name: &str) -> Error {
    Error::new(
        ErrorKind::FeatureUnsupported,
        format!("{table_name}: snapshot scope not yet served"),
    )
}

impl IcebergMetadataTableProvider {
    pub async fn scan(self) -> DFResult<BoxStream<'static, DFResult<RecordBatch>>> {
        let metadata_table = self.table.inspect();
        let snapshot_id = self.snapshot_id;
        let table_name = self.r#type.as_str().to_owned();
        let stream = match self.r#type {
            MetadataTableType::Snapshots => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.snapshots().scan().await,
            },
            MetadataTableType::Manifests => match snapshot_id {
                Some(id) => ManifestsTable::at_snapshot(&self.table, id).scan().await,
                None => metadata_table.manifests().scan().await,
            },
            MetadataTableType::Files => match snapshot_id {
                Some(id) => match FilesTable::try_all_at_snapshot(&self.table, id) {
                    Ok(scoped) => scoped.scan().await,
                    Err(err) => Err(err),
                },
                None => metadata_table.files().scan().await,
            },
            MetadataTableType::DataFiles => match snapshot_id {
                Some(id) => match FilesTable::try_data_at_snapshot(&self.table, id) {
                    Ok(scoped) => scoped.scan().await,
                    Err(err) => Err(err),
                },
                None => metadata_table.data_files().scan().await,
            },
            MetadataTableType::DeleteFiles => match snapshot_id {
                Some(id) => match FilesTable::try_deletes_at_snapshot(&self.table, id) {
                    Ok(scoped) => scoped.scan().await,
                    Err(err) => Err(err),
                },
                None => metadata_table.delete_files().scan().await,
            },
            MetadataTableType::Entries => match snapshot_id {
                Some(id) => match EntriesTable::try_at_snapshot(&self.table, id) {
                    Ok(scoped) => scoped.scan().await,
                    Err(err) => Err(err),
                },
                None => metadata_table.entries().scan().await,
            },
            MetadataTableType::AllFiles => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.all_files().scan().await,
            },
            MetadataTableType::AllDataFiles => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.all_data_files().scan().await,
            },
            MetadataTableType::AllDeleteFiles => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.all_delete_files().scan().await,
            },
            MetadataTableType::AllEntries => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.all_entries().scan().await,
            },
            MetadataTableType::History => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.history().scan().await,
            },
            MetadataTableType::Refs => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.refs().scan().await,
            },
            MetadataTableType::MetadataLogEntries => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.metadata_log_entries().scan().await,
            },
            MetadataTableType::Partitions => match snapshot_id {
                Some(id) => match PartitionsTable::try_at_snapshot(&self.table, id) {
                    Ok(scoped) => scoped.scan().await,
                    Err(err) => Err(err),
                },
                None => metadata_table.partitions().scan().await,
            },
            MetadataTableType::AllManifests => match snapshot_id {
                Some(_) => Err(snapshot_scope_refused(&table_name)),
                None => metadata_table.all_manifests().scan().await,
            },
            MetadataTableType::PositionDeletes => match snapshot_id {
                Some(id) => match PositionDeletesTable::try_at_snapshot(&self.table, id) {
                    Ok(scoped) => scoped.scan().await,
                    Err(err) => Err(err),
                },
                None => metadata_table.position_deletes().scan().await,
            },
        }
        .map_err(to_datafusion_error)?;
        let stream = stream.map_err(to_datafusion_error);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::datasource::TableProvider;
    use datafusion::execution::TaskContext;
    use datafusion::prelude::SessionContext;
    use futures::TryStreamExt;
    use iceberg::TableIdent;
    use iceberg::inspect::MetadataTableType;
    use iceberg::io::FileIO;
    use iceberg::table::{StaticTable, Table};

    use super::IcebergMetadataTableProvider;

    // Every `MetadataTableType` variant; kept exhaustive alongside the `try_new` match so a new
    // metadata table cannot silently skip the schema-resolution guard.
    const ALL_METADATA_TABLE_TYPES: [MetadataTableType; 15] = [
        MetadataTableType::Snapshots,
        MetadataTableType::Manifests,
        MetadataTableType::Files,
        MetadataTableType::DataFiles,
        MetadataTableType::DeleteFiles,
        MetadataTableType::Entries,
        MetadataTableType::AllFiles,
        MetadataTableType::AllDataFiles,
        MetadataTableType::AllDeleteFiles,
        MetadataTableType::AllEntries,
        MetadataTableType::History,
        MetadataTableType::Refs,
        MetadataTableType::MetadataLogEntries,
        MetadataTableType::Partitions,
        MetadataTableType::AllManifests,
    ];

    async fn test_table() -> Table {
        let metadata_file_path = format!(
            "{}/tests/test_data/{}",
            env!("CARGO_MANIFEST_DIR"),
            "TableMetadataV2Valid.json"
        );
        let file_io = FileIO::new_with_fs();
        let ident = TableIdent::from_strs(["ns", "t"]).unwrap();
        StaticTable::from_metadata_file(&metadata_file_path, ident, file_io)
            .await
            .unwrap()
            .into_table()
    }

    /// SAF-008: `TableProvider::schema()` is infallible, but the Iceberg → Arrow schema
    /// conversion is not. Every metadata-table type must resolve its Arrow schema at
    /// construction (`try_new`) so `schema()` returns an already-validated schema and never
    /// unwraps the conversion inside the trait method.
    ///
    /// MUTATION (drop the eager `try_new` resolution and restore
    /// `schema_to_arrow_schema(&schema).unwrap().into()` inside `schema()`): the fallible
    /// conversion moves back into the infallible trait method, reintroducing the `.unwrap()`
    /// panic surface this test guards against for all metadata-table types.
    #[tokio::test]
    async fn test_metadata_table_provider_schema_resolves_for_all_types() {
        let table = test_table().await;
        for r#type in ALL_METADATA_TABLE_TYPES {
            let provider =
                IcebergMetadataTableProvider::try_new(table.clone(), r#type.clone(), None)
                    .unwrap_or_else(|e| panic!("try_new failed for {type:?}: {e}"));
            assert!(
                !provider.schema().fields().is_empty(),
                "arrow schema for metadata table {type:?} must be non-empty",
            );
        }
    }

    async fn collect_scan(
        provider: &IcebergMetadataTableProvider,
        projection: Option<&Vec<usize>>,
    ) -> (
        datafusion::arrow::datatypes::SchemaRef,
        Vec<datafusion::arrow::array::RecordBatch>,
    ) {
        let ctx = SessionContext::new();
        let plan = TableProvider::scan(provider, &ctx.state(), projection, &[], None)
            .await
            .expect("metadata table scan must plan");
        let schema = plan.schema();
        let stream = plan
            .execute(0, Arc::new(TaskContext::default()))
            .expect("metadata table scan must execute");
        let batches: Vec<_> = stream
            .try_collect()
            .await
            .expect("metadata table scan must collect");
        (schema, batches)
    }

    fn total_rows(batches: &[datafusion::arrow::array::RecordBatch]) -> usize {
        batches.iter().map(|batch| batch.num_rows()).sum()
    }

    #[tokio::test]
    async fn test_metadata_table_scan_projects_subset_in_requested_order() {
        let table = test_table().await;
        let provider =
            IcebergMetadataTableProvider::try_new(table, MetadataTableType::Snapshots, None)
                .expect("snapshots metadata provider");
        let full_fields: Vec<String> = provider
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        assert!(
            full_fields.len() >= 2,
            "snapshots schema must have at least two columns"
        );
        let last = full_fields.len() - 1;
        let indices = vec![last, 0];

        let (projected_schema, projected_batches) = collect_scan(&provider, Some(&indices)).await;
        let projected_names: Vec<&str> = projected_schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        assert_eq!(projected_names, vec![
            full_fields[last].as_str(),
            full_fields[0].as_str()
        ]);

        let (full_schema, full_batches) = collect_scan(&provider, None).await;
        assert_eq!(full_schema.fields().len(), full_fields.len());
        assert_eq!(
            provider.schema().fields().len(),
            full_fields.len(),
            "TableProvider::schema must stay the full schema"
        );
        assert_eq!(
            total_rows(&projected_batches),
            total_rows(&full_batches),
            "a column subset must keep the snapshot row count"
        );
        assert!(
            !projected_batches.is_empty() && !full_batches.is_empty(),
            "the fixture must yield at least one snapshots batch"
        );
        assert_eq!(
            projected_batches[0].column(0).as_ref(),
            full_batches[0].column(last).as_ref()
        );
        assert_eq!(
            projected_batches[0].column(1).as_ref(),
            full_batches[0].column(0).as_ref()
        );
    }

    #[tokio::test]
    async fn test_metadata_table_scan_empty_projection_preserves_row_count() {
        let table = test_table().await;
        let provider =
            IcebergMetadataTableProvider::try_new(table, MetadataTableType::Snapshots, None)
                .expect("snapshots metadata provider");
        let (_, full_batches) = collect_scan(&provider, None).await;
        let full_rows = total_rows(&full_batches);
        assert!(full_rows > 0, "the fixture must have snapshot rows");

        let empty: Vec<usize> = Vec::new();
        let (empty_schema, empty_batches) = collect_scan(&provider, Some(&empty)).await;
        assert_eq!(empty_schema.fields().len(), 0);
        assert_eq!(
            total_rows(&empty_batches),
            full_rows,
            "SELECT count(*) empty projection must keep the snapshot row count"
        );
        assert!(
            empty_batches.iter().all(|batch| batch.num_columns() == 0),
            "every empty-projection batch must have zero columns"
        );
    }

    #[tokio::test]
    async fn test_metadata_table_scan_rejects_out_of_bounds_projection() {
        let table = test_table().await;
        let provider =
            IcebergMetadataTableProvider::try_new(table, MetadataTableType::Snapshots, None)
                .expect("snapshots metadata provider");
        let ctx = SessionContext::new();
        let err = TableProvider::scan(&provider, &ctx.state(), Some(&vec![999]), &[], None)
            .await
            .expect_err("index 999 must fail at plan time");
        let message = err.to_string();
        assert!(
            message.contains("999") || message.to_lowercase().contains("index"),
            "out-of-bounds projection must name the bad index, got: {message}"
        );
    }

    const SNAPSHOT_UNSCOPED_TYPES: [MetadataTableType; 9] = [
        MetadataTableType::Snapshots,
        MetadataTableType::History,
        MetadataTableType::Refs,
        MetadataTableType::MetadataLogEntries,
        MetadataTableType::AllFiles,
        MetadataTableType::AllDataFiles,
        MetadataTableType::AllDeleteFiles,
        MetadataTableType::AllEntries,
        MetadataTableType::AllManifests,
    ];

    const SNAPSHOT_SCOPED_TYPES: [MetadataTableType; 7] = [
        MetadataTableType::Files,
        MetadataTableType::DataFiles,
        MetadataTableType::DeleteFiles,
        MetadataTableType::Entries,
        MetadataTableType::Manifests,
        MetadataTableType::Partitions,
        MetadataTableType::PositionDeletes,
    ];

    const UNKNOWN_SNAPSHOT_ID: i64 = 999_999_999_999;

    fn datafusion_message(err: datafusion::error::DataFusionError) -> String {
        match err {
            datafusion::error::DataFusionError::External(inner) => {
                match inner.downcast_ref::<iceberg::Error>() {
                    Some(ice) => ice.message().to_owned(),
                    None => inner.to_string(),
                }
            }
            other => other.to_string(),
        }
    }

    fn iceberg_kind(err: &datafusion::error::DataFusionError) -> iceberg::ErrorKind {
        match err {
            datafusion::error::DataFusionError::External(inner) => inner
                .downcast_ref::<iceberg::Error>()
                .expect("external error wraps an iceberg error")
                .kind(),
            other => panic!("expected an external iceberg error, got: {other}"),
        }
    }

    async fn provider_scan_outcome(
        table: &Table,
        r#type: MetadataTableType,
        snapshot_id: Option<i64>,
    ) -> Result<String, String> {
        let provider = IcebergMetadataTableProvider::try_new(table.clone(), r#type, snapshot_id)
            .map_err(|e| e.message().to_owned())?;
        let batches: Vec<datafusion::arrow::array::RecordBatch> = provider
            .scan()
            .await
            .map_err(datafusion_message)?
            .try_collect()
            .await
            .map_err(datafusion_message)?;
        Ok(format!("{batches:?}"))
    }

    async fn accessor_scan_outcome(
        table: &Table,
        r#type: MetadataTableType,
    ) -> Result<String, String> {
        let metadata_table = table.inspect();
        let stream = match r#type {
            MetadataTableType::Snapshots => metadata_table.snapshots().scan().await,
            MetadataTableType::Manifests => metadata_table.manifests().scan().await,
            MetadataTableType::Files => metadata_table.files().scan().await,
            MetadataTableType::DataFiles => metadata_table.data_files().scan().await,
            MetadataTableType::DeleteFiles => metadata_table.delete_files().scan().await,
            MetadataTableType::Entries => metadata_table.entries().scan().await,
            MetadataTableType::AllFiles => metadata_table.all_files().scan().await,
            MetadataTableType::AllDataFiles => metadata_table.all_data_files().scan().await,
            MetadataTableType::AllDeleteFiles => metadata_table.all_delete_files().scan().await,
            MetadataTableType::AllEntries => metadata_table.all_entries().scan().await,
            MetadataTableType::History => metadata_table.history().scan().await,
            MetadataTableType::Refs => metadata_table.refs().scan().await,
            MetadataTableType::MetadataLogEntries => {
                metadata_table.metadata_log_entries().scan().await
            }
            MetadataTableType::Partitions => metadata_table.partitions().scan().await,
            MetadataTableType::AllManifests => metadata_table.all_manifests().scan().await,
            MetadataTableType::PositionDeletes => metadata_table.position_deletes().scan().await,
        };
        let batches: Vec<datafusion::arrow::array::RecordBatch> = stream
            .map_err(|e| e.message().to_owned())?
            .try_collect()
            .await
            .map_err(|e| e.message().to_owned())?;
        Ok(format!("{batches:?}"))
    }

    #[tokio::test]
    async fn test_snapshot_scope_none_matches_accessor_for_all_types() {
        let table = test_table().await;
        let mut matched = 0u32;
        for r#type in MetadataTableType::all_types() {
            let name = r#type.as_str().to_owned();
            let expected = accessor_scan_outcome(&table, r#type.clone()).await;
            let actual = provider_scan_outcome(&table, r#type, None).await;
            assert_eq!(
                actual, expected,
                "provider None outcome must equal the accessor outcome for {name}"
            );
            matched += 1;
        }
        assert_eq!(matched, 16);
    }

    #[tokio::test]
    async fn test_snapshot_scope_some_refused_for_unserved_tables() {
        let table = test_table().await;
        let snapshot_id = table
            .metadata()
            .current_snapshot_id()
            .expect("fixture has a current snapshot");
        let mut refused = 0u32;
        for r#type in SNAPSHOT_UNSCOPED_TYPES {
            let name = r#type.as_str().to_owned();
            let provider =
                IcebergMetadataTableProvider::try_new(table.clone(), r#type, Some(snapshot_id))
                    .unwrap_or_else(|e| panic!("try_new failed for {name}: {e}"));
            let err = match provider.scan().await {
                Ok(_) => panic!("scan with Some must refuse for {name}"),
                Err(err) => err,
            };
            assert_eq!(
                iceberg_kind(&err),
                iceberg::ErrorKind::FeatureUnsupported,
                "refusal kind for {name}"
            );
            let message = datafusion_message(err);
            assert!(
                message.contains(&name),
                "refusal must name the table {name}, got: {message}"
            );
            assert!(
                message.contains("snapshot scope not yet served"),
                "refusal must state snapshot scope is unserved, got: {message}"
            );
            refused += 1;
        }
        assert_eq!(refused, 9);
    }

    #[tokio::test]
    async fn test_snapshot_scope_some_unknown_id_fails_loud_for_served_tables() {
        let table = test_table().await;
        let mut failed = 0u32;
        for r#type in SNAPSHOT_SCOPED_TYPES {
            let name = r#type.as_str().to_owned();
            let provider = IcebergMetadataTableProvider::try_new(
                table.clone(),
                r#type,
                Some(UNKNOWN_SNAPSHOT_ID),
            )
            .unwrap_or_else(|e| panic!("try_new failed for {name}: {e}"));
            let err = match provider.scan().await {
                Ok(_) => panic!("unknown snapshot id must fail for {name}"),
                Err(err) => err,
            };
            assert_eq!(
                iceberg_kind(&err),
                iceberg::ErrorKind::DataInvalid,
                "unknown-id kind for {name}"
            );
            let message = datafusion_message(err);
            assert!(
                message.contains("Cannot find snapshot"),
                "unknown id must fail loud for {name}, got: {message}"
            );
            failed += 1;
        }
        assert_eq!(failed, 7);
    }

    #[tokio::test]
    async fn test_snapshot_scope_some_current_matches_none_for_served_tables() {
        let table = test_table().await;
        let snapshot_id = table
            .metadata()
            .current_snapshot_id()
            .expect("fixture has a current snapshot");
        let mut matched = 0u32;
        for r#type in SNAPSHOT_SCOPED_TYPES {
            let name = r#type.as_str().to_owned();
            let expected = provider_scan_outcome(&table, r#type.clone(), None).await;
            let actual = provider_scan_outcome(&table, r#type, Some(snapshot_id)).await;
            assert_eq!(actual, expected, "Some(current) must match None for {name}");
            matched += 1;
        }
        assert_eq!(matched, 7);
    }

    #[tokio::test]
    async fn test_snapshot_scope_some_serves_partitions_and_position_deletes() {
        let table = test_table().await;
        let snapshot_id = table
            .metadata()
            .current_snapshot_id()
            .expect("fixture has a current snapshot");
        for r#type in [
            MetadataTableType::Partitions,
            MetadataTableType::PositionDeletes,
        ] {
            let name = r#type.as_str().to_owned();
            let expected = provider_scan_outcome(&table, r#type.clone(), None).await;
            let actual = provider_scan_outcome(&table, r#type, Some(snapshot_id)).await;
            match &actual {
                Ok(_) => {}
                Err(message) => assert!(
                    !message.contains("snapshot scope not yet served"),
                    "Some must serve rows for {name}, got refusal: {message}"
                ),
            }
            assert_eq!(actual, expected, "Some(current) must match None for {name}");
        }
    }

    #[tokio::test]
    async fn test_snapshot_scope_some_refuses_all_star_tables() {
        let table = test_table().await;
        let snapshot_id = table
            .metadata()
            .current_snapshot_id()
            .expect("fixture has a current snapshot");
        let mut refused = 0u32;
        for r#type in [
            MetadataTableType::AllFiles,
            MetadataTableType::AllDataFiles,
            MetadataTableType::AllDeleteFiles,
            MetadataTableType::AllEntries,
            MetadataTableType::AllManifests,
        ] {
            let name = r#type.as_str().to_owned();
            let provider =
                IcebergMetadataTableProvider::try_new(table.clone(), r#type, Some(snapshot_id))
                    .unwrap_or_else(|e| panic!("try_new failed for {name}: {e}"));
            let err = match provider.scan().await {
                Ok(_) => panic!("scan with Some must refuse for {name}"),
                Err(err) => err,
            };
            assert_eq!(
                iceberg_kind(&err),
                iceberg::ErrorKind::FeatureUnsupported,
                "refusal kind for {name}"
            );
            let message = datafusion_message(err);
            assert!(
                message.contains(&name),
                "refusal must name the table {name}, got: {message}"
            );
            assert!(
                message.contains("snapshot scope not yet served"),
                "refusal must state snapshot scope is unserved, got: {message}"
            );
            refused += 1;
        }
        assert_eq!(refused, 5);
    }
}
