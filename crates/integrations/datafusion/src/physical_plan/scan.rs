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
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field as ArrowField, SchemaRef as ArrowSchemaRef};
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
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use super::expr_to_predicate::convert_filters_to_predicate;
use crate::to_datafusion_error;

/// How one advertised output column is produced from the scanned data.
///
/// Resolution is by FIELD ID, never by name: a column's name is mutable metadata in Iceberg
/// (`RENAME COLUMN` keeps the id and creates no snapshot), so a name-keyed binding silently reads
/// the wrong column — or fabricates NULLs over live data — the moment a rename lands.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ColumnSource {
    /// Take the scanned column with this name — the name the SCANNED snapshot's schema gives the
    /// advertised field's id, which is not necessarily the advertised name.
    Scanned(String),
    /// The advertised field's id is not in the scanned snapshot's schema at all (e.g. a column
    /// added after that snapshot): emit NULLs, as Java's readers do for an absent projected field.
    Absent,
}

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
    /// The column names actually selected from the table — the SCANNED snapshot schema's name for
    /// each advertised field whose id it carries. See [`IcebergTableScan::new`].
    scan_columns: Vec<String>,
    /// How each advertised output column is produced, parallel to the advertised schema's fields.
    sources: Vec<ColumnSource>,
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
    /// So the scan never says "give me everything the table has now" (`select_all`). It resolves
    /// each advertised field BY FIELD ID against the scanned snapshot's schema (see
    /// [`ColumnSource`]), selects the name that schema gives the id, and [`conform_batch`] renames,
    /// promotes or null-fills on the way out.
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
        let (scan_columns, sources) = resolve_projection(&table, snapshot_id, &output_schema)?;
        let plan_properties = Self::compute_properties(output_schema);
        let predicates = convert_filters_to_predicate(filters);

        Ok(Self {
            table,
            snapshot_id,
            plan_properties,
            projection,
            scan_columns,
            sources,
            predicates,
            limit,
        })
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
        let sources = self.sources.clone();
        let stream = futures::stream::once(fut)
            .try_flatten()
            .and_then(move |batch| {
                futures::future::ready(conform_batch(batch, &advertised_schema, &sources))
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

/// Binds each advertised output column to the scanned snapshot's schema BY FIELD ID, returning the
/// names to `select` from the table and the per-column [`ColumnSource`] bindings.
///
/// Names are not identities in Iceberg — `RENAME COLUMN` keeps the field id, rewrites the name and
/// creates no snapshot, so the schema the data is read with (the snapshot's) and the schema the plan
/// advertised routinely disagree on names for the SAME column. Field ids are the identities, and
/// they are already carried on the advertised Arrow fields as `PARQUET:field_id` metadata (written
/// by `schema_to_arrow_schema`). An advertised field without that metadata cannot be bound at all,
/// so it is a loud error rather than a name-based guess.
///
/// A name this returns is one `TableScanBuilder::build` accepts, because it came out of the very
/// schema that builder resolves against — the scan can never fail on a column the plan advertised.
///
/// When there is no snapshot to resolve against — an unknown snapshot id (the core scan reports
/// that itself, with the better message) or a table with no snapshot at all (an empty scan that
/// emits no batches) — the binding is the identity and field ids are not consulted.
fn resolve_projection(
    table: &Table,
    snapshot_id: Option<i64>,
    output_schema: &ArrowSchemaRef,
) -> DFResult<(Vec<String>, Vec<ColumnSource>)> {
    let metadata = table.metadata();
    let snapshot = match snapshot_id {
        Some(snapshot_id) => metadata.snapshot_by_id(snapshot_id),
        None => metadata.current_snapshot(),
    };
    let Some(snapshot) = snapshot else {
        let names: Vec<String> = output_schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let sources = names.iter().cloned().map(ColumnSource::Scanned).collect();
        return Ok((names, sources));
    };
    let snapshot_schema = snapshot.schema(metadata).map_err(to_datafusion_error)?;

    let mut scan_columns = Vec::with_capacity(output_schema.fields().len());
    let mut sources = Vec::with_capacity(output_schema.fields().len());
    for field in output_schema.fields() {
        // Reserved metadata columns (`_file`, `_pos`, ...) are not table fields; the core scan
        // resolves their reserved ids from the name itself.
        if is_metadata_column_name(field.name()) {
            scan_columns.push(field.name().clone());
            sources.push(ColumnSource::Scanned(field.name().clone()));
            continue;
        }

        let field_id = advertised_field_id(field)?;
        match snapshot_schema.name_by_field_id(field_id) {
            Some(name) => {
                scan_columns.push(name.to_string());
                sources.push(ColumnSource::Scanned(name.to_string()));
            }
            None => sources.push(ColumnSource::Absent),
        }
    }
    Ok((scan_columns, sources))
}

/// The Iceberg field id an advertised Arrow field carries, or a loud error.
fn advertised_field_id(field: &ArrowField) -> DFResult<i32> {
    let raw = field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .ok_or_else(|| {
            to_datafusion_error(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "column '{}' carries no `{PARQUET_FIELD_ID_META_KEY}` metadata, so it cannot be \
                     bound to a table field: an Iceberg column is identified by its field id, and \
                     matching on the name instead would read the wrong column after a rename",
                    field.name()
                ),
            ))
        })?;
    raw.parse::<i32>().map_err(|e| {
        to_datafusion_error(
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "column '{}' carries an unparsable `{PARQUET_FIELD_ID_META_KEY}` metadata value '{raw}'",
                    field.name()
                ),
            )
            .with_source(e),
        )
    })
}

/// Whether an Arrow type change is one of Iceberg's LEGAL type promotions.
///
/// This is the Arrow-side mirror of [`iceberg::spec::is_promotion_allowed`] (int → long,
/// float → double, decimal precision widening at equal scale, plus identity) — the Iceberg
/// primitives involved map one-to-one onto these Arrow types. The mirror is pinned against the
/// authority itself by `test_arrow_promotion_mirror_agrees_with_iceberg_rule`, so the two cannot
/// drift apart silently.
fn is_arrow_promotion_allowed(from: &DataType, to: &DataType) -> bool {
    if from == to {
        return true;
    }
    match (from, to) {
        (DataType::Int32, DataType::Int64) => true,
        (DataType::Float32, DataType::Float64) => true,
        (
            DataType::Decimal128(from_precision, from_scale),
            DataType::Decimal128(to_precision, to_scale),
        )
        | (
            DataType::Decimal256(from_precision, from_scale),
            DataType::Decimal256(to_precision, to_scale),
        ) => from_scale == to_scale && from_precision <= to_precision,
        _ => false,
    }
}

/// Coerces a scanned batch to the schema the plan advertised (BUG-011).
///
/// A DataFusion operator addresses its input by ORDINAL against the schema its child advertised, so
/// a batch that merely happens to carry the right columns in a different order — or an extra one —
/// is silent corruption, not a nuisance. This rebuilds the batch in the advertised order from the
/// field-id bindings [`resolve_projection`] computed:
///
/// * bound, same type → taken as is (the steady-state case, where the batch already equals the
///   advertised schema and is returned untouched);
/// * bound under a different NAME (`RENAME COLUMN`) → the same values, under the advertised name;
/// * bound with a legally PROMOTED type (`int` → `long`, ...) → cast to the advertised type;
/// * unbound and nullable → an all-NULL column, matching Java, whose readers null-fill a projected
///   field the data it reads does not have (e.g. a column added after the scanned snapshot);
/// * unbound and NOT nullable, or an illegal type change → a typed error naming the column.
///   Silently coercing here would hand DataFusion data that contradicts its plan.
///
/// The row count is carried explicitly so a zero-column projection (`SELECT count(*)`) keeps it.
fn conform_batch(
    batch: RecordBatch,
    advertised: &ArrowSchemaRef,
    sources: &[ColumnSource],
) -> DFResult<RecordBatch> {
    if batch.schema_ref() == advertised {
        return Ok(batch);
    }
    if sources.len() != advertised.fields().len() {
        return Err(datafusion::error::DataFusionError::Internal(format!(
            "the scan bound {} columns but advertises {}",
            sources.len(),
            advertised.fields().len()
        )));
    }

    let num_rows = batch.num_rows();
    let mut columns = Vec::with_capacity(advertised.fields().len());
    for (field, source) in advertised.fields().iter().zip(sources) {
        match source {
            ColumnSource::Scanned(name) => {
                let column = batch.column_by_name(name).ok_or_else(|| {
                    datafusion::error::DataFusionError::Internal(format!(
                        "the scan selected column '{name}' for advertised column '{}' but the \
                         scanned batch does not carry it",
                        field.name()
                    ))
                })?;
                if column.data_type() == field.data_type() {
                    columns.push(column.clone());
                } else if is_arrow_promotion_allowed(column.data_type(), field.data_type()) {
                    columns.push(cast(column, field.data_type()).map_err(|e| {
                        datafusion::error::DataFusionError::ArrowError(
                            Box::new(e),
                            Some(format!("promoting column '{}'", field.name())),
                        )
                    })?);
                } else {
                    return Err(to_datafusion_error(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "column '{}' is {} in the snapshot being scanned but {} in the schema \
                             this query was planned against, and that is not a legal Iceberg type \
                             promotion — the data cannot be read as the planned type",
                            field.name(),
                            column.data_type(),
                            field.data_type()
                        ),
                    )));
                }
            }
            ColumnSource::Absent if field.is_nullable() => {
                columns.push(new_null_array(field.data_type(), num_rows))
            }
            ColumnSource::Absent => {
                return Err(to_datafusion_error(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "required column '{}' has no field with its id in the snapshot being \
                         scanned, so there is no data to read for it and a required column cannot \
                         be null-filled",
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

        let sources = vec![
            ColumnSource::Scanned("id".to_string()),
            ColumnSource::Scanned("data".to_string()),
            ColumnSource::Absent,
        ];
        let conformed =
            conform_batch(scanned, &advertised, &sources).expect("the batch must conform");
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
        let conformed = conform_batch(scanned, &advertised, &[]).expect("the batch must conform");
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

        let err = conform_batch(scanned, &advertised, &[ColumnSource::Scanned(
            "id".to_string(),
        )])
        .expect_err("an illegal type change must not be silently coerced");
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

        let sources = vec![
            ColumnSource::Scanned("id".to_string()),
            ColumnSource::Absent,
        ];
        let err = conform_batch(scanned, &advertised, &sources)
            .expect_err("an absent required column must not be null-filled");
        assert!(
            err.to_string().contains("required_new"),
            "the error must name the column: {err}"
        );
    }

    /// S2-3: the Arrow promotion mirror must agree with the AUTHORITY,
    /// [`iceberg::spec::is_promotion_allowed`], on every pair of primitives — a drift between the
    /// two would either reject a legal promotion or, worse, cast something Iceberg forbids.
    #[test]
    fn test_arrow_promotion_mirror_agrees_with_iceberg_rule() {
        use iceberg::spec::{PrimitiveType as P, Type as T, is_promotion_allowed};

        let primitives = [
            P::Boolean,
            P::Int,
            P::Long,
            P::Float,
            P::Double,
            P::Decimal {
                precision: 9,
                scale: 2,
            },
            P::Decimal {
                precision: 18,
                scale: 2,
            },
            P::Decimal {
                precision: 18,
                scale: 3,
            },
            P::Date,
            P::Time,
            P::Timestamp,
            P::String,
            P::Binary,
            P::Uuid,
        ];

        // The Arrow form of each primitive, via the crate's own converter — so the mirror is checked
        // against the very mapping production uses.
        let arrow_of = |primitive: &P| -> ArrowDataType {
            let schema = Schema::builder()
                .with_fields(vec![Arc::new(NestedField::required(
                    1,
                    "c",
                    Type::Primitive(primitive.clone()),
                ))])
                .build()
                .expect("one-field schema must build");
            iceberg::arrow::schema_to_arrow_schema(&schema)
                .expect("schema must convert to arrow")
                .field(0)
                .data_type()
                .clone()
        };

        let mut checked = 0;
        for from in &primitives {
            for to in &primitives {
                let expected = is_promotion_allowed(&T::Primitive(from.clone()), to);
                let actual = is_arrow_promotion_allowed(&arrow_of(from), &arrow_of(to));
                assert_eq!(
                    actual,
                    expected,
                    "mirror disagrees for {from} -> {to} (arrow {:?} -> {:?})",
                    arrow_of(from),
                    arrow_of(to)
                );
                checked += 1;
            }
        }
        assert_eq!(checked, primitives.len() * primitives.len());
        // Non-vacuity: the matrix must contain at least one allowed NON-identity promotion.
        assert!(is_arrow_promotion_allowed(
            &ArrowDataType::Int32,
            &ArrowDataType::Int64
        ));
    }

    /// S2-3: a legally promoted column is cast to the advertised type, not rejected.
    #[test]
    fn test_conform_batch_promotes_a_legal_type_change() {
        let scanned = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "id",
                ArrowDataType::Int32,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int32Array::from(vec![
                7, -3,
            ]))],
        )
        .expect("the scanned batch must build");

        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int64,
            false,
        )]));

        let conformed = conform_batch(scanned, &advertised, &[ColumnSource::Scanned(
            "id".to_string(),
        )])
        .expect("int -> long is a legal Iceberg promotion");
        assert_eq!(conformed.schema(), advertised);
        assert_eq!(
            conformed
                .column(0)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .expect("the promoted column must be Int64")
                .values(),
            &[7, -3],
            "the values must survive the widening"
        );
    }

    /// A table whose CURRENT SNAPSHOT resolves to the 3-column schema `x`(1), `y`(2), `z`(3) — the
    /// committed V2 fixture, so the field-id binding is exercised against real metadata.
    async fn table_with_snapshot() -> Table {
        let metadata_file_path = format!(
            "{}/tests/test_data/TableMetadataV2Valid.json",
            env!("CARGO_MANIFEST_DIR")
        );
        iceberg::table::StaticTable::from_metadata_file(
            &metadata_file_path,
            iceberg::TableIdent::from_strs(["ns", "t"]).expect("ident must parse"),
            iceberg::io::FileIO::new_with_fs(),
        )
        .await
        .expect("the fixture metadata must load")
        .into_table()
    }

    fn field_with_id(name: &str, data_type: ArrowDataType, nullable: bool, id: i32) -> ArrowField {
        ArrowField::new(name, data_type, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    /// S1-1: an advertised field with no `PARQUET:field_id` cannot be bound to a table field, and
    /// guessing by name is exactly the bug field-id binding exists to prevent — so it errors.
    #[tokio::test]
    async fn test_resolve_projection_requires_a_field_id() {
        let table = table_with_snapshot().await;
        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int64,
            false,
        )]));

        let err = resolve_projection(&table, None, &advertised)
            .expect_err("a field without an id must not be bound by name");
        assert!(
            err.to_string().contains(PARQUET_FIELD_ID_META_KEY) && err.to_string().contains('x'),
            "the error must name the column and the missing metadata: {err}"
        );
    }

    /// S1-1: a renamed column binds to the SNAPSHOT schema's name for its FIELD ID — not to the
    /// advertised name — and an id the snapshot schema does not have becomes a null-fill.
    #[tokio::test]
    async fn test_resolve_projection_binds_by_field_id_not_by_name() {
        let table = table_with_snapshot().await;
        // The snapshot schema calls field 2 `y`; this plan advertises it as `renamed_y`.
        let advertised: ArrowSchemaRef = Arc::new(ArrowSchema::new(vec![
            field_with_id("renamed_y", ArrowDataType::Int64, false, 2),
            field_with_id("added_later", ArrowDataType::Int32, true, 99),
        ]));

        let (scan_columns, sources) =
            resolve_projection(&table, None, &advertised).expect("the projection must resolve");
        assert_eq!(
            scan_columns,
            vec!["y".to_string()],
            "the SNAPSHOT schema's name for field 2 is what gets selected"
        );
        assert_eq!(sources, vec![
            ColumnSource::Scanned("y".to_string()),
            ColumnSource::Absent,
        ]);
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
