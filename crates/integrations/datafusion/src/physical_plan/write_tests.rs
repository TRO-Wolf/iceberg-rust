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
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{
    DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::common::Result as DFResult;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::{StreamExt, stream};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::spec::{
    DataFile, DataFileFormat, NestedField, PrimitiveType, Schema, TableProperties, Type,
    deserialize_data_file_from_json,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, MemoryCatalog, NamespaceIdent, Result, TableCreation,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::basic::Encoding;
use parquet::file::reader::{FileReader, SerializedFileReader};
use tempfile::TempDir;

use super::*;

struct MockExecutionPlan {
    schema: ArrowSchemaRef,
    batches: Vec<RecordBatch>,
    properties: Arc<PlanProperties>,
}

impl MockExecutionPlan {
    fn new(schema: ArrowSchemaRef, batches: Vec<RecordBatch>) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));

        Self {
            schema,
            batches,
            properties,
        }
    }
}

impl Debug for MockExecutionPlan {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockExecutionPlan")
    }
}

impl DisplayAs for MockExecutionPlan {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "MockExecutionPlan")
            }
        }
    }
}

impl ExecutionPlan for MockExecutionPlan {
    fn name(&self) -> &str {
        "MockExecutionPlan"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let batches = self.batches.clone();
        let stream = stream::iter(batches.into_iter().map(Ok));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream.boxed(),
        )))
    }
}

fn temp_path() -> String {
    let temp_dir = TempDir::new().unwrap();
    temp_dir.path().to_str().unwrap().to_string()
}

async fn get_iceberg_catalog() -> MemoryCatalog {
    MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), temp_path())]),
        )
        .await
        .unwrap()
}

fn get_test_schema() -> Result<Schema> {
    Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
}

fn get_table_creation(
    location: impl ToString,
    name: impl ToString,
    schema: Schema,
) -> TableCreation {
    TableCreation::builder()
        .location(location.to_string())
        .name(name.to_string())
        .properties(HashMap::new())
        .schema(schema)
        .build()
}

#[tokio::test]
async fn test_iceberg_write_exec() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_namespace".to_string());

    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await?;

    let schema = get_test_schema()?;

    let table_name = "test_table";
    let table_location = temp_path();
    let creation = get_table_creation(table_location, table_name, schema);
    let table = iceberg_catalog.create_table(&namespace, creation).await?;

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]));

    let id_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
    let name_array = Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])) as ArrayRef;

    let batch =
        RecordBatch::try_new(arrow_schema.clone(), vec![id_array, name_array]).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Failed to create record batch: {e}"),
            )
        })?;

    let input_plan = Arc::new(MockExecutionPlan::new(arrow_schema.clone(), vec![
        batch.clone(),
    ]));

    let write_exec = IcebergWriteExec::new(
        table.clone(),
        input_plan,
        table.metadata().default_partition_spec().clone(),
        None,
    );

    assert_eq!(
        write_exec.schema().as_ref(),
        &ArrowSchema::new(vec![
            Field::new(DATA_FILES_COL_NAME, DataType::Utf8, false),
            Field::new(WRITE_PARTITION_INDEX_COL_NAME, DataType::UInt64, false),
        ]),
        "IcebergWriteExec must advertise its result schema"
    );

    let task_ctx = Arc::new(TaskContext::default());
    let stream = write_exec.execute(0, task_ctx).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to execute plan: {e}"),
        )
    })?;

    let mut results = vec![];
    let mut stream = stream;
    while let Some(batch) = stream.next().await {
        results.push(
            batch.map_err(|e| {
                Error::new(ErrorKind::Unexpected, format!("Failed to get batch: {e}"))
            })?,
        );
    }

    assert_eq!(results.len(), 1, "Expected one result batch");
    let result_batch = &results[0];

    assert_eq!(
        result_batch.schema().as_ref(),
        &ArrowSchema::new(vec![
            Field::new(DATA_FILES_COL_NAME, DataType::Utf8, false),
            Field::new(WRITE_PARTITION_INDEX_COL_NAME, DataType::UInt64, false),
        ])
    );

    assert_eq!(result_batch.num_rows(), 1, "Expected one data file");

    let data_file_json = result_batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray")
        .value(0);

    let partition_type = table.metadata().default_partition_type();
    let spec_id = table.metadata().default_partition_spec_id();
    let schema = table.metadata().current_schema();

    let data_file =
        deserialize_data_file_from_json(data_file_json, spec_id, partition_type, schema)
            .expect("Failed to deserialize data file JSON");

    assert_eq!(
        data_file.record_count(),
        3,
        "Expected 3 records in the data file"
    );
    assert!(
        data_file.file_size_in_bytes() > 0,
        "File size should be greater than 0"
    );
    assert_eq!(
        data_file.file_format(),
        DataFileFormat::Parquet,
        "Expected Parquet file format"
    );

    assert!(
        data_file.column_sizes().get(&1).unwrap() > &0,
        "Column 1 size should be greater than 0"
    );
    assert!(
        data_file.column_sizes().get(&2).unwrap() > &0,
        "Column 2 size should be greater than 0"
    );

    assert_eq!(
        *data_file.value_counts().get(&1).unwrap(),
        3,
        "Expected 3 values for column 1"
    );
    assert_eq!(
        *data_file.value_counts().get(&2).unwrap(),
        3,
        "Expected 3 values for column 2"
    );

    assert!(
        data_file.lower_bounds().contains_key(&1) || data_file.lower_bounds().contains_key(&2),
        "Expected lower bounds to contain at least one column"
    );
    assert!(
        data_file.upper_bounds().contains_key(&1) || data_file.upper_bounds().contains_key(&2),
        "Expected upper bounds to contain at least one column"
    );

    let file_path = data_file.file_path();
    assert!(!file_path.is_empty(), "File path should not be empty");

    let file_io = table.file_io();
    assert!(file_io.exists(file_path).await?, "Data file should exist");

    Ok(())
}

#[tokio::test]
async fn test_insert_honors_metrics_default_none() -> Result<()> {
    let iceberg_catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("test_namespace".to_string());
    iceberg_catalog
        .create_namespace(&namespace, HashMap::new())
        .await?;

    let schema = get_test_schema()?;
    let creation = TableCreation::builder()
        .location(temp_path())
        .name("metrics_none_table".to_string())
        .properties(HashMap::from([(
            "write.metadata.metrics.default".to_string(),
            "none".to_string(),
        )]))
        .schema(schema)
        .build();
    let table = iceberg_catalog.create_table(&namespace, creation).await?;

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "1".to_string(),
        )])),
        Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            "2".to_string(),
        )])),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])) as ArrayRef,
    ])
    .map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to create record batch: {e}"),
        )
    })?;

    let input_plan = Arc::new(MockExecutionPlan::new(arrow_schema, vec![batch]));
    let write_exec = IcebergWriteExec::new(
        table.clone(),
        input_plan,
        table.metadata().default_partition_spec().clone(),
        None,
    );
    let mut stream = write_exec
        .execute(0, Arc::new(TaskContext::default()))
        .map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Failed to execute plan: {e}"),
            )
        })?;

    let mut results = vec![];
    while let Some(batch) = stream.next().await {
        results.push(
            batch.map_err(|e| {
                Error::new(ErrorKind::Unexpected, format!("Failed to get batch: {e}"))
            })?,
        );
    }
    assert_eq!(results.len(), 1, "Expected one result batch");

    let data_file_json = results[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Expected StringArray")
        .value(0);
    let data_file = deserialize_data_file_from_json(
        data_file_json,
        table.metadata().default_partition_spec_id(),
        table.metadata().default_partition_type(),
        table.metadata().current_schema(),
    )
    .expect("Failed to deserialize data file JSON");

    assert_eq!(data_file.record_count(), 3);
    assert!(
        data_file.column_sizes().is_empty(),
        "metrics.default=none must write no column_sizes: {:?}",
        data_file.column_sizes()
    );
    assert!(
        data_file.value_counts().is_empty(),
        "metrics.default=none must write no value_counts"
    );
    assert!(
        data_file.null_value_counts().is_empty(),
        "metrics.default=none must write no null_value_counts"
    );
    assert!(
        data_file.nan_value_counts().is_empty(),
        "metrics.default=none must write no nan_value_counts"
    );
    assert!(
        data_file.lower_bounds().is_empty(),
        "metrics.default=none must write no lower_bounds"
    );
    assert!(
        data_file.upper_bounds().is_empty(),
        "metrics.default=none must write no upper_bounds"
    );

    Ok(())
}

async fn format_table(format: &str) -> Result<(MemoryCatalog, Table)> {
    format_table_with_properties(format, format, HashMap::new()).await
}

async fn format_table_with_properties(
    name: &str,
    format: &str,
    extra: HashMap<String, String>,
) -> Result<(MemoryCatalog, Table)> {
    let catalog = get_iceberg_catalog().await;
    let namespace = NamespaceIdent::new("format_ns".to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;
    let mut properties = HashMap::from([(
        TableProperties::PROPERTY_DEFAULT_FILE_FORMAT.to_string(),
        format.to_string(),
    )]);
    properties.extend(extra);
    let creation = TableCreation::builder()
        .location(temp_path())
        .name(format!("format_{name}_table"))
        .properties(properties)
        .schema(get_test_schema()?)
        .build();
    let table = catalog.create_table(&namespace, creation).await?;
    Ok((catalog, table))
}

fn plan_format_write(table: &Table) -> IcebergWriteExec {
    let meta = |id: &str| HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]);
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false).with_metadata(meta("1")),
        Field::new("name", DataType::Utf8, false).with_metadata(meta("2")),
    ]));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])) as ArrayRef,
    ])
    .expect("fixture batch builds");
    let input_plan = Arc::new(MockExecutionPlan::new(arrow_schema, vec![batch]));
    IcebergWriteExec::new(
        table.clone(),
        input_plan,
        table.metadata().default_partition_spec().clone(),
        None,
    )
}

async fn run_format_write(table: &Table) -> Result<Vec<DataFile>> {
    let mut stream = plan_format_write(table)
        .execute(0, Arc::new(TaskContext::default()))
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("execute: {e}")))?;
    let mut files = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.map_err(|e| Error::new(ErrorKind::Unexpected, format!("{e}")))?;
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("data_files column is Utf8");
        for row in 0..column.len() {
            files.push(deserialize_data_file_from_json(
                column.value(row),
                table.metadata().default_partition_spec_id(),
                table.metadata().default_partition_type(),
                table.metadata().current_schema(),
            )?);
        }
    }
    Ok(files)
}

async fn commit_format_files(
    catalog: &MemoryCatalog,
    table: &Table,
    files: Vec<DataFile>,
) -> Result<Table> {
    let tx = Transaction::new(table);
    let action = tx.fast_append().add_data_files(files);
    action.apply(tx)?.commit(catalog).await
}

async fn scan_format_rows(table: &Table) -> Result<Vec<(i32, String)>> {
    let mut stream = table
        .scan()
        .select(["id", "name"])
        .build()?
        .to_arrow()
        .await?;
    let mut rows = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id column is Int32");
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("name column is Utf8");
        for row in 0..batch.num_rows() {
            rows.push((ids.value(row), names.value(row).to_string()));
        }
    }
    rows.sort();
    Ok(rows)
}

async fn write_commit_scan(
    name: &str,
    format: DataFileFormat,
    check_bytes: impl Fn(&[u8]),
) -> Result<()> {
    let (catalog, table) = format_table(name).await?;
    let files = run_format_write(&table).await?;
    assert_eq!(files.len(), 1, "one write produces one data file");
    assert_eq!(files[0].file_format(), format);
    let path = files[0].file_path();
    let expected = format!(".{format}");
    assert!(
        path.ends_with(&expected),
        "data file path {path} ends with {expected}"
    );
    if format != DataFileFormat::Parquet {
        assert!(
            !path.ends_with(".parquet"),
            "non-parquet file path {path} keeps its own suffix"
        );
    }
    let bytes = table
        .file_io()
        .new_input(files[0].file_path())?
        .read()
        .await?;
    check_bytes(&bytes);
    let table = commit_format_files(&catalog, &table, files).await?;
    assert_eq!(scan_format_rows(&table).await?, vec![
        (1, "Alice".to_string()),
        (2, "Bob".to_string()),
        (3, "Charlie".to_string()),
    ]);
    Ok(())
}

#[tokio::test]
async fn test_iceberg_write_exec_orc_writes_orc_bytes() -> Result<()> {
    write_commit_scan("orc", DataFileFormat::Orc, |bytes| {
        assert!(
            bytes.len() > 4 && &bytes[bytes.len() - 4..bytes.len() - 1] == b"ORC",
            "orc tail magic"
        );
    })
    .await
}

#[tokio::test]
async fn test_iceberg_write_exec_avro_writes_avro_bytes() -> Result<()> {
    write_commit_scan("avro", DataFileFormat::Avro, |bytes| {
        assert!(bytes.starts_with(b"Obj\x01"), "avro OCF header");
    })
    .await
}

#[tokio::test]
async fn test_iceberg_write_exec_puffin_refuses_data_invalid() -> Result<()> {
    let (_catalog, table) = format_table("puffin").await?;
    let Err(err) = plan_format_write(&table).execute(0, Arc::new(TaskContext::default())) else {
        panic!("puffin is never a data file");
    };
    let DataFusionError::External(inner) = err else {
        panic!("expected External iceberg error, got {err}");
    };
    let iceberg_err = inner
        .downcast_ref::<Error>()
        .expect("external wraps iceberg Error");
    assert_eq!(iceberg_err.kind(), ErrorKind::DataInvalid);
    assert_eq!(
        iceberg_err.message(),
        "Cannot build a data-file writer for format puffin: a sidecar is never a data file"
    );
    Ok(())
}

fn parquet_columns_use_dictionary(bytes: bytes::Bytes) -> Vec<bool> {
    let reader = SerializedFileReader::new(bytes).expect("read the parquet footer");
    let metadata = reader.metadata();
    assert!(
        metadata.num_row_groups() > 0,
        "the written file must hold a row group"
    );
    let mut flags = Vec::new();
    for row_group in metadata.row_groups() {
        for column in row_group.columns() {
            flags.push(column.encodings().any(|encoding| {
                matches!(
                    encoding,
                    Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY
                )
            }));
        }
    }
    assert!(
        !flags.is_empty(),
        "the written file must hold column chunks"
    );
    flags
}

async fn written_parquet_dictionary(name: &str, dict: Option<&str>) -> Result<Vec<bool>> {
    let extra = dict
        .map(|value| HashMap::from([("parquet.enable.dictionary".to_string(), value.to_string())]))
        .unwrap_or_default();
    let (_catalog, table) = format_table_with_properties(name, "parquet", extra).await?;
    let files = run_format_write(&table).await?;
    assert_eq!(files.len(), 1, "one write produces one data file");
    let bytes = table
        .file_io()
        .new_input(files[0].file_path())?
        .read()
        .await?;
    Ok(parquet_columns_use_dictionary(bytes))
}

#[tokio::test]
async fn test_iceberg_write_exec_parquet_defaults_dictionary_off() -> Result<()> {
    let flags = written_parquet_dictionary("parquet_dict_off", None).await?;
    assert!(
        flags.iter().all(|flag| !flag),
        "a default write leaves no dictionary encoding"
    );
    Ok(())
}

#[tokio::test]
async fn test_iceberg_write_exec_parquet_property_enables_dictionary() -> Result<()> {
    let flags = written_parquet_dictionary("parquet_dict_on", Some("true")).await?;
    assert!(
        flags.iter().all(|flag| *flag),
        "a property write keeps dictionary encoding"
    );
    Ok(())
}

#[tokio::test]
async fn test_iceberg_write_exec_bogus_format_surfaces_from_str() -> Result<()> {
    let (_catalog, table) = format_table("csv").await?;
    let Err(err) = plan_format_write(&table).execute(0, Arc::new(TaskContext::default())) else {
        panic!("bogus format refuses");
    };
    let DataFusionError::External(inner) = err else {
        panic!("expected External iceberg error, got {err}");
    };
    let iceberg_err = inner
        .downcast_ref::<Error>()
        .expect("external wraps iceberg Error");
    assert_eq!(iceberg_err.kind(), ErrorKind::DataInvalid);
    assert_eq!(iceberg_err.message(), "Unsupported data file format: csv");
    Ok(())
}
