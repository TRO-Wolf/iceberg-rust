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
use std::pin::Pin;
use std::sync::Arc;
use std::vec;

use datafusion::arrow::array::{RecordBatch, RecordBatchOptions, new_null_array};
use datafusion::arrow::datatypes::SchemaRef as ArrowSchemaRef;
use datafusion::error::Result as DFResult;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::prelude::Expr;
use futures::{Stream, TryStreamExt};
use iceberg::expr::Predicate;
use iceberg::metadata_columns::is_metadata_column_name;
use iceberg::table::Table;
use iceberg::{Error, ErrorKind};

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// Manages the scanning process of an Iceberg [`Table`], encapsulating the
/// necessary details and computed properties required for execution planning.
#[derive(Debug)]
pub struct IcebergTableScan {
    /// A table in the catalog.
    table: Table,
    /// Snapshot of the table to scan.
    snapshot_id: Option<i64>,
    /// Stores certain, often expensive to compute,
    /// plan properties used in query optimization.
    plan_properties: PlanProperties,
    /// Projection column names, None means all columns
    projection: Option<Vec<String>>,
    /// The column names actually selected from the table: the advertised (output) columns MINUS
    /// any the scanned snapshot's schema does not carry. See [`IcebergTableScan::new`].
    scan_columns: Vec<String>,
    /// Filters to apply to the table scan
    predicates: Option<Predicate>,
    /// Optional limit on the number of rows to return
    limit: Option<usize>,
}

impl IcebergTableScan {
    /// Creates a new [`IcebergTableScan`] object.
    ///
    /// Returns a planning error when `projection` holds an index outside `schema`
    /// (previously an `unwrap` panic — SAF-004). The projected column names are derived
    /// from the projected schema itself, so they can never index out of bounds.
    ///
    /// # The advertised schema is a contract (BUG-011)
    ///
    /// `schema` is the schema DataFusion PLANNED this query against — `projection` indexes into
    /// it, and every parent operator was built against its projection. That projection is therefore
    /// the schema this node advertises, and the batches it emits MUST match it. Two things can make
    /// the table disagree with it:
    ///
    /// * the caller reloaded the table between planning and scanning (the catalog-backed provider
    ///   does exactly that), and
    /// * `ADD COLUMN` does not create a snapshot, so a table's CURRENT schema routinely has columns
    ///   the scanned snapshot's schema — the schema the core scan resolves names against — lacks.
    ///
    /// So the scan never says "give me everything the table has now" (`select_all`): it selects the
    /// advertised columns BY NAME, minus the ones the scanned snapshot's schema cannot resolve, and
    /// [`conform_batch`] restores those as NULL columns at execution — which is what Java does when
    /// a projected field is absent from the data it reads.
    pub(crate) fn new(
        table: Table,
        snapshot_id: Option<i64>,
        schema: ArrowSchemaRef,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Self> {
        let (output_schema, projection) = match projection {
            None => (schema, None),
            Some(indices) => {
                let projected_schema = Arc::new(schema.project(indices)?);
                let column_names = projected_schema
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect();
                (projected_schema, Some(column_names))
            }
        };
        let scan_columns = Self::resolve_scan_columns(&table, snapshot_id, &output_schema)?;
        let plan_properties = Self::compute_properties(output_schema);
        let predicates = convert_filters_to_predicate(filters);

        Ok(Self {
            table,
            snapshot_id,
            plan_properties,
            projection,
            scan_columns,
            predicates,
            limit,
        })
    }

    /// The advertised columns the scanned snapshot's schema can actually resolve.
    ///
    /// The acceptance test is deliberately the SAME one `TableScanBuilder::build` applies
    /// (`field_by_name`, with metadata columns exempt): a name this filter keeps is a name the core
    /// scan accepts, so the scan can never fail on a column the plan advertised. Names dropped here
    /// are null-filled by [`conform_batch`].
    ///
    /// When the scan has no snapshot to resolve against — an unknown snapshot id (the core scan
    /// reports that itself, with the better message) or a table with no snapshot at all (an empty
    /// scan) — every advertised name is passed through unchanged.
    fn resolve_scan_columns(
        table: &Table,
        snapshot_id: Option<i64>,
        output_schema: &ArrowSchemaRef,
    ) -> DFResult<Vec<String>> {
        let advertised = output_schema
            .fields()
            .iter()
            .map(|field| field.name().clone());

        let metadata = table.metadata();
        let snapshot = match snapshot_id {
            Some(snapshot_id) => metadata.snapshot_by_id(snapshot_id),
            None => metadata.current_snapshot(),
        };
        let Some(snapshot) = snapshot else {
            return Ok(advertised.collect());
        };
        let snapshot_schema = snapshot.schema(metadata).map_err(to_datafusion_error)?;

        Ok(advertised
            .filter(|name| {
                is_metadata_column_name(name) || snapshot_schema.field_by_name(name).is_some()
            })
            .collect())
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    pub fn snapshot_id(&self) -> Option<i64> {
        self.snapshot_id
    }

    pub fn projection(&self) -> Option<&[String]> {
        self.projection.as_deref()
    }

    pub fn predicates(&self) -> Option<&Predicate> {
        self.predicates.as_ref()
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Computes [`PlanProperties`] used in query optimization.
    fn compute_properties(schema: ArrowSchemaRef) -> PlanProperties {
        // TODO:
        // This is more or less a placeholder, to be replaced
        // once we support output-partitioning
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
    }
}

impl ExecutionPlan for IcebergTableScan {
    fn name(&self) -> &str {
        "IcebergTableScan"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan + 'static>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let fut = get_batch_stream(
            self.table.clone(),
            self.snapshot_id,
            self.scan_columns.clone(),
            self.predicates.clone(),
        );
        // Every emitted batch is conformed to the schema this node advertised — see
        // `IcebergTableScan::new` and `conform_batch`.
        let advertised_schema = self.schema();
        let stream = futures::stream::once(fut)
            .try_flatten()
            .and_then(move |batch| {
                futures::future::ready(conform_batch(batch, &advertised_schema))
            });

        // Apply limit if specified
        let limited_stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> =
            if let Some(limit) = self.limit {
                let mut remaining = limit;
                Box::pin(stream.try_filter_map(move |batch| {
                    futures::future::ready(if remaining == 0 {
                        Ok(None)
                    } else if batch.num_rows() <= remaining {
                        remaining -= batch.num_rows();
                        Ok(Some(batch))
                    } else {
                        let limited_batch = batch.slice(0, remaining);
                        remaining = 0;
                        Ok(Some(limited_batch))
                    })
                }))
            } else {
                Box::pin(stream)
            };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            limited_stream,
        )))
    }
}

impl DisplayAs for IcebergTableScan {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(
            f,
            "IcebergTableScan projection:[{}] predicate:[{}]",
            self.projection
                .clone()
                .map_or(String::new(), |v| v.join(",")),
            self.predicates
                .clone()
                .map_or(String::from(""), |p| format!("{p}"))
        )
    }
}

/// Asynchronously retrieves a stream of [`RecordBatch`] instances
/// from a given table.
///
/// This function initializes a [`TableScan`], builds it,
/// and then converts it into a stream of Arrow [`RecordBatch`]es.
async fn get_batch_stream(
    table: Table,
    snapshot_id: Option<i64>,
    column_names: Vec<String>,
    predicates: Option<Predicate>,
) -> DFResult<Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>>> {
    let scan_builder = match snapshot_id {
        Some(snapshot_id) => table.scan().snapshot_id(snapshot_id),
        None => table.scan(),
    };

    // Always an explicit selection — NEVER `select_all()`, which would read whatever column set the
    // table happens to have now rather than the one the plan advertised (BUG-011).
    let mut scan_builder = scan_builder.select(column_names);
    if let Some(pred) = predicates {
        scan_builder = scan_builder.with_filter(pred);
    }
    let table_scan = scan_builder.build().map_err(to_datafusion_error)?;

    let stream = table_scan
        .to_arrow()
        .await
        .map_err(to_datafusion_error)?
        .map_err(to_datafusion_error);
    Ok(Box::pin(stream))
}

/// Coerces a scanned batch to the schema the plan advertised (BUG-011).
///
/// A DataFusion operator addresses its input by ORDINAL against the schema its child advertised, so
/// a batch that merely happens to carry the right columns in a different order — or an extra one —
/// is silent corruption, not a nuisance. This resolves every advertised field BY NAME and rebuilds
/// the batch in the advertised order:
///
/// * present with the advertised type → taken as is (the common case; a batch that already equals
///   the advertised schema is returned untouched);
/// * absent and nullable → an all-NULL column, matching Java, whose readers null-fill a projected
///   field that the data it reads does not have (e.g. a column added after the scanned snapshot);
/// * absent and NOT nullable, or present with a different type → a typed error naming the column.
///   Silently coercing here would hand DataFusion data that contradicts its plan.
///
/// The row count is carried explicitly so a zero-column projection (`SELECT count(*)`) keeps it.
fn conform_batch(batch: RecordBatch, advertised: &ArrowSchemaRef) -> DFResult<RecordBatch> {
    if batch.schema_ref() == advertised {
        return Ok(batch);
    }

    let num_rows = batch.num_rows();
    let mut columns = Vec::with_capacity(advertised.fields().len());
    for field in advertised.fields() {
        match batch.column_by_name(field.name()) {
            Some(column) if column.data_type() == field.data_type() => columns.push(column.clone()),
            Some(column) => {
                return Err(to_datafusion_error(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "column '{}' was scanned as {} but the query plan expects {}: the table's \
                         type for this column changed after the query was planned; re-plan the query",
                        field.name(),
                        column.data_type(),
                        field.data_type()
                    ),
                )));
            }
            None if field.is_nullable() => {
                columns.push(new_null_array(field.data_type(), num_rows))
            }
            None => {
                return Err(to_datafusion_error(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "required column '{}' is absent from the scanned data and cannot be \
                         null-filled: the table's schema changed after the query was planned; \
                         re-plan the query",
                        field.name()
                    ),
                )));
            }
        }
    }

    RecordBatch::try_new_with_options(
        advertised.clone(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(|e| {
        datafusion::error::DataFusionError::ArrowError(
            Box::new(e),
            Some("failed to conform a scanned batch to the schema the plan advertised".to_string()),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use datafusion::arrow::datatypes::{
        DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
    };
    use iceberg::TableIdent;
    use iceberg::io::FileIO;
    use iceberg::spec::{
        FormatVersion, NestedField, PartitionSpec, PrimitiveType, Schema, SortOrder,
        TableMetadataBuilder, Type,
    };

    use super::*;

    fn create_test_table() -> Table {
        let schema = Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Long),
                )),
                Arc::new(NestedField::required(
                    2,
                    "data",
                    Type::Primitive(PrimitiveType::String),
                )),
            ])
            .build()
            .expect("test schema must build");

        let partition_spec = PartitionSpec::builder(schema.clone())
            .build()
            .expect("partition spec must build");
        let sort_order = SortOrder::builder()
            .build(&schema)
            .expect("sort order must build");
        let table_metadata = TableMetadataBuilder::new(
            schema,
            partition_spec,
            sort_order,
            "memory://test/table".to_string(),
            FormatVersion::V2,
            HashMap::new(),
        )
        .expect("metadata builder must construct")
        .build()
        .expect("table metadata must build");

        Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).expect("ident must parse"))
            .file_io(FileIO::new_with_memory())
            .metadata_location("memory://test/metadata.json".to_string())
            .build()
            .expect("table must build")
    }

    fn test_arrow_schema() -> ArrowSchemaRef {
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("data", ArrowDataType::Utf8, false),
        ]))
    }

    /// SAF-004 P5a: an out-of-bounds projection index must yield a PLANNING error —
    /// previously `schema.project(projection).unwrap()` panicked.
    #[test]
    fn test_scan_out_of_bounds_projection_is_error_not_panic() {
        let err = IcebergTableScan::new(
            create_test_table(),
            None,
            test_arrow_schema(),
            Some(&vec![0, 99]),
            &[],
            None,
        )
        .expect_err("projection index 99 on a 2-column schema must be a planning error");
        assert!(
            err.to_string().contains("99"),
            "the error should name the offending index: {err}"
        );
    }

    /// BUG-011: an advertised column the scan could not read is restored as NULL, and the columns
    /// are returned in the ADVERTISED order — DataFusion addresses them by ordinal.
    #[test]
    fn test_conform_batch_null_fills_and_reorders() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("data", ArrowDataType::Utf8, false),
                ArrowField::new("id", ArrowDataType::Int64, false),
            ])),
            vec![
                Arc::new(datafusion::arrow::array::StringArray::from(vec!["a", "b"])),
                Arc::new(datafusion::arrow::array::Int64Array::from(vec![1, 2])),
            ],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("data", ArrowDataType::Utf8, false),
            ArrowField::new("added_later", ArrowDataType::Int32, true),
        ]));

        let conformed = conform_batch(scanned, &advertised).expect("the batch must conform");
        assert_eq!(conformed.schema(), advertised);
        assert_eq!(conformed.num_rows(), 2);
        assert_eq!(
            conformed
                .column_by_name("added_later")
                .expect("the added column must be present")
                .null_count(),
            2,
            "a column the scan could not read must be all-NULL"
        );
        assert_eq!(
            conformed
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("column 0 must be the advertised `id`")
                .values(),
            &[1, 2]
        );
    }

    /// A zero-column projection (`SELECT count(*)`) must keep its row count — a rebuilt batch with
    /// no columns has no other way to carry it.
    #[test]
    fn test_conform_batch_preserves_row_count_with_no_columns() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
                1, 2, 3,
            ]))],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(Vec::<ArrowField>::new()));
        let conformed = conform_batch(scanned, &advertised).expect("the batch must conform");
        assert_eq!(conformed.num_columns(), 0);
        assert_eq!(
            conformed.num_rows(),
            3,
            "the row count must survive a zero-column projection"
        );
    }

    /// A column whose type changed after planning must be a loud, named error — handing DataFusion
    /// data that contradicts its plan is the corruption this whole path exists to prevent.
    #[test]
    fn test_conform_batch_rejects_a_changed_type() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
                1,
            ]))],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int32,
            false,
        )]));

        let err = conform_batch(scanned, &advertised)
            .expect_err("a type change must not be silently coerced");
        assert!(
            err.to_string().contains("id") && err.to_string().contains("Int32"),
            "the error must name the column and the expected type: {err}"
        );
    }

    /// An advertised NON-nullable column that the scan could not read cannot be null-filled, so it
    /// must error rather than fabricate values.
    #[test]
    fn test_conform_batch_rejects_absent_required_column() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
                1,
            ]))],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", ArrowDataType::Int64, false),
            ArrowField::new("required_new", ArrowDataType::Utf8, false),
        ]));

        let err = conform_batch(scanned, &advertised)
            .expect_err("an absent required column must not be null-filled");
        assert!(
            err.to_string().contains("required_new"),
            "the error must name the column: {err}"
        );
    }

    /// SAF-004 P5b (regression): a valid projection still produces the projected output schema
    /// and the projected column names, and no projection passes the schema through unchanged.
    #[test]
    fn test_scan_valid_projection_schema_and_names() {
        let projected = IcebergTableScan::new(
            create_test_table(),
            None,
            test_arrow_schema(),
            Some(&vec![1]),
            &[],
            None,
        )
        .expect("a valid projection must plan");
        assert_eq!(projected.projection(), Some(&["data".to_string()][..]));
        let output_schema = projected.schema();
        assert_eq!(output_schema.fields().len(), 1);
        assert_eq!(output_schema.field(0).name(), "data");
        assert_eq!(output_schema.field(0).data_type(), &ArrowDataType::Utf8);

        let unprojected = IcebergTableScan::new(
            create_test_table(),
            None,
            test_arrow_schema(),
            None,
            &[],
            None,
        )
        .expect("a scan without projection must plan");
        assert_eq!(unprojected.projection(), None);
        assert_eq!(unprojected.schema(), test_arrow_schema());
    }
}
