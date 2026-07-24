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

use std::any::Any;
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
use iceberg::inspect::MetadataTableType;
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
            MetadataTableType::Files => metadata_table.files().schema(),
            MetadataTableType::DataFiles => metadata_table.data_files().schema(),
            MetadataTableType::DeleteFiles => metadata_table.delete_files().schema(),
            MetadataTableType::Entries => metadata_table.entries().schema(),
            MetadataTableType::AllFiles => metadata_table.all_files().schema(),
            MetadataTableType::AllDataFiles => metadata_table.all_data_files().schema(),
            MetadataTableType::AllDeleteFiles => metadata_table.all_delete_files().schema(),
            MetadataTableType::AllEntries => metadata_table.all_entries().schema(),
            MetadataTableType::History => metadata_table.history().schema(),
            MetadataTableType::Refs => metadata_table.refs().schema(),
            MetadataTableType::MetadataLogEntries => metadata_table.metadata_log_entries().schema(),
            MetadataTableType::Partitions => metadata_table.partitions().schema(),
            MetadataTableType::AllManifests => metadata_table.all_manifests().schema(),
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
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(IcebergMetadataScan::new(self.clone())))
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
        }
        .map_err(to_datafusion_error)?;
        let stream = stream.map_err(to_datafusion_error);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::datasource::TableProvider;
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
}
