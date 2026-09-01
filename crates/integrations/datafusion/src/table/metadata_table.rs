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
use iceberg::Result;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::inspect::{
    EntriesTable, FilesTable, MetadataTableType, PartitionsTable, PositionDeletesTable,
};
use iceberg::table::Table;

use crate::physical_plan::metadata_scan::IcebergMetadataScan;
use crate::to_datafusion_error;

/// Represents a [`TableProvider`] for the Iceberg [`Catalog`],
/// managing access to a [`MetadataTable`].
#[derive(Debug, Clone)]
pub struct IcebergMetadataTableProvider {
    pub(crate) table: Table,
    pub(crate) r#type: MetadataTableType,
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
    pub(crate) fn try_new(table: Table, r#type: MetadataTableType) -> Result<Self> {
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

impl IcebergMetadataTableProvider {
    pub async fn scan(self) -> DFResult<BoxStream<'static, DFResult<RecordBatch>>> {
        let metadata_table = self.table.inspect();
        let stream = match self.r#type {
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
            // Schema-only table: refused loud in `inspect` (no async work to await).
            MetadataTableType::PositionDeletes => metadata_table.position_deletes().scan(),
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
            let provider = IcebergMetadataTableProvider::try_new(table.clone(), r#type.clone())
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
        let provider = IcebergMetadataTableProvider::try_new(table, MetadataTableType::Snapshots)
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
        let provider = IcebergMetadataTableProvider::try_new(table, MetadataTableType::Snapshots)
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
        let provider = IcebergMetadataTableProvider::try_new(table, MetadataTableType::Snapshots)
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
}
