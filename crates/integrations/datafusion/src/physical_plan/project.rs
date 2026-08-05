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

//! Partition value projection for Iceberg tables.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{ColumnarValue, ExecutionPlan};
use iceberg::arrow::{
    PROJECTED_PARTITION_VALUE_COLUMN, PartitionValueCalculator, schema_to_arrow_schema,
    strip_metadata_from_schema,
};
use iceberg::spec::PartitionSpec;
use iceberg::table::Table;

use crate::to_datafusion_error;

/// Nesting depth beyond which [`data_type_is_write_compatible`] stops walking and
/// falls back to plain structural equality (the pre-widening rule).
///
/// The input schema comes from a query plan, so the walk is user-influenced: a
/// pathologically nested type must not be able to overflow the thread stack
/// (AGENTS.md, "Recursion Safety"). The fallback is the STRICT comparison, so
/// exceeding the limit can only ever reject a write the widened rule would have
/// accepted — never accept one it would have rejected.
const MAX_WRITE_COMPATIBILITY_DEPTH: usize = 64;

/// Whether an input field may be written into the table field `expected`.
///
/// Identical to structural equality except for **one-directional nullability
/// widening**: a NON-nullable input value can always be stored in a nullable
/// (Iceberg OPTIONAL) target, so `required -> optional` is accepted. The reverse
/// is not: a nullable input into a required target may carry a NULL the table
/// forbids, and stays rejected. Names, types, field order and arity are
/// unchanged — strict.
///
/// This is the direction Java Iceberg gates writes on. Decoded from the
/// `iceberg-api` 1.10.0 bytecode (`org.apache.iceberg.types.CheckCompatibility`):
/// `writeCompatibilityErrors(readSchema, writeSchema)` visits the TABLE schema
/// with `checkNullability = true`, and `field()` records the nullability error
/// `"<name> should be required, but is optional"` on exactly one condition —
/// `readField.isRequired() && writeField.isOptional()`, i.e. optional incoming
/// data into a required table column. A required incoming field landing in an
/// optional table column produces no error at all. Java never gates a write on
/// whole-schema equality.
///
/// The fork's check stays STRICTER than Java's on every other axis (Java matches
/// by field id, tolerates missing optionals and extra fields, and permits type
/// promotion); this function relaxes the nullability axis only.
///
/// Comparing name + nullability + data type is EXHAUSTIVE here, not a subset of
/// `Field`'s own equality: both sides arrive from `strip_metadata_from_schema`,
/// which rebuilds every field at every level with `Field::new`, so the remaining
/// `Field` components (`metadata`, `dict_is_ordered`) are equal by construction.
/// Nullability is therefore the only axis this relaxes.
fn field_is_write_compatible(input: &Field, expected: &Field, depth: usize) -> bool {
    input.name() == expected.name()
        // The ONLY relaxation: reject exactly `nullable input -> required target`.
        && (!input.is_nullable() || expected.is_nullable())
        && data_type_is_write_compatible(input.data_type(), expected.data_type(), depth)
}

/// [`field_is_write_compatible`] for data types: recurses through the nested
/// kinds an Iceberg schema can produce (struct fields, list elements, map
/// key/value) so the widening applies at every level, and compares everything
/// else exactly.
fn data_type_is_write_compatible(input: &DataType, expected: &DataType, depth: usize) -> bool {
    if depth >= MAX_WRITE_COMPATIBILITY_DEPTH {
        // Depth guard: degrade to the strict pre-widening rule rather than
        // recursing further (see MAX_WRITE_COMPATIBILITY_DEPTH).
        return input == expected;
    }
    let depth = depth + 1;
    match (input, expected) {
        (DataType::Struct(input_fields), DataType::Struct(expected_fields)) => {
            input_fields.len() == expected_fields.len()
                && input_fields
                    .iter()
                    .zip(expected_fields.iter())
                    .all(|(input, expected)| field_is_write_compatible(input, expected, depth))
        }
        (DataType::List(input_element), DataType::List(expected_element))
        | (DataType::LargeList(input_element), DataType::LargeList(expected_element)) => {
            field_is_write_compatible(input_element, expected_element, depth)
        }
        (
            DataType::FixedSizeList(input_element, input_len),
            DataType::FixedSizeList(expected_element, expected_len),
        ) => {
            input_len == expected_len
                && field_is_write_compatible(input_element, expected_element, depth)
        }
        // The map's `key_value` entries field is a struct of {key, value}; the
        // recursion widens the VALUE field. The `sorted` flag is part of the
        // type and must match.
        (
            DataType::Map(input_entries, input_sorted),
            DataType::Map(expected_entries, expected_sorted),
        ) => {
            input_sorted == expected_sorted
                && field_is_write_compatible(input_entries, expected_entries, depth)
        }
        // Every primitive — and any nested kind not modelled above — must match
        // EXACTLY. Unknown shapes fall back to the strict rule rather than being
        // waved through.
        (input, expected) => input == expected,
    }
}

/// Extends an ExecutionPlan with partition value calculations for Iceberg tables.
///
/// This function takes an input ExecutionPlan and extends it with an additional column
/// containing calculated partition values based on the table's partition specification.
/// For unpartitioned tables, returns the original plan unchanged.
///
/// # Arguments
/// * `input` - The input ExecutionPlan to extend
/// * `table` - The Iceberg table with partition specification
///
/// # Returns
/// * `Ok(Arc<dyn ExecutionPlan>)` - Extended plan with partition values column
/// * `Err` - If partition spec is not found or transformation fails
pub fn project_with_partition(
    input: Arc<dyn ExecutionPlan>,
    table: &Table,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    let metadata = table.metadata();
    let partition_spec = metadata.default_partition_spec();
    let table_schema = metadata.current_schema();

    if partition_spec.is_unpartitioned() {
        return Ok(input);
    }

    let input_schema = input.schema();

    // Validate that input_schema matches the Iceberg table schema
    // Strip metadata from both schemas before comparison to ignore metadata differences
    let expected_arrow_schema =
        schema_to_arrow_schema(table_schema.as_ref()).map_err(to_datafusion_error)?;
    let input_schema_cleaned =
        strip_metadata_from_schema(&input_schema).map_err(to_datafusion_error)?;
    let expected_schema_cleaned =
        strip_metadata_from_schema(&expected_arrow_schema).map_err(to_datafusion_error)?;

    // Field-by-field rather than `!=` on the whole schema: the ONE tolerated
    // difference is safe-direction nullability widening (a non-nullable input
    // column into an OPTIONAL table column), applied recursively. Everything
    // else — names, types, arity, order, and `nullable input -> required target`
    // — stays strict and fails with the message below.
    let input_fields = input_schema_cleaned.fields();
    let expected_fields = expected_schema_cleaned.fields();
    let schemas_compatible = input_fields.len() == expected_fields.len()
        && input_fields
            .iter()
            .zip(expected_fields.iter())
            .all(|(input, expected)| field_is_write_compatible(input, expected, 0));

    if !schemas_compatible {
        return Err(DataFusionError::Plan(format!(
            "Input schema does not match Iceberg table schema.\n\
             Expected schema: {expected_schema_cleaned}\n\
             Input schema: {input_schema_cleaned}"
        )));
    }

    let calculator =
        PartitionValueCalculator::try_new(partition_spec.as_ref(), table_schema.as_ref())
            .map_err(to_datafusion_error)?;

    // One child per column of the (already-validated) input schema. The children
    // are how DataFusion's optimizers see — and rewrite — this expression's
    // inputs; hiding them (children() == vec![]) lets `ProjectionPushdown`
    // re-parent the expression verbatim onto a different plan node, computing
    // partition values from the wrong batch.
    let children: Vec<Arc<dyn PhysicalExpr>> = input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>)
        .collect();

    // The passthrough SELECT items ARE those same children — built once, so the
    // projection's columns and the partition expression's inputs cannot drift.
    let mut projection_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(children.len() + 1);
    for (child, field) in children.iter().zip(input_schema.fields().iter()) {
        projection_exprs.push((Arc::clone(child), field.name().clone()));
    }

    let partition_expr = Arc::new(PartitionExpr::new(
        calculator,
        partition_spec.clone(),
        children,
    ));
    projection_exprs.push((partition_expr, PROJECTED_PARTITION_VALUE_COLUMN.to_string()));

    let projection = ProjectionExec::try_new(projection_exprs, input)?;
    Ok(Arc::new(projection))
}

/// PhysicalExpr implementation for partition value calculation.
///
/// The expression reads EVERY table column: `children` carries one expression
/// per top-level column of the table schema (initially `Column` references into
/// the validated input, in table-schema order). Declaring the children honestly
/// is load-bearing for correctness, not decoration:
///
/// * `ProjectionPushdown`'s `try_unifying_projections` counts column references
///   through `children()` — with the children visible, its anti-fusion guard
///   (`count > 1 && !is_expr_trivial`) refuses to fuse this projection with a
///   non-trivial SELECT-list projection below it.
/// * When fusion or push-through IS legal (all-trivial inputs), `update_expr`
///   rewrites the children (`Column` → the child projection's expression), and
///   `with_new_children` rebuilds this expression around them — so `evaluate`
///   always computes partition values from the values this plan node actually
///   receives, never from a positional read of a re-parented batch.
#[derive(Debug, Clone)]
struct PartitionExpr {
    calculator: Arc<PartitionValueCalculator>,
    partition_spec: Arc<PartitionSpec>,
    /// One expression per top-level table column, in table-schema order.
    children: Vec<Arc<dyn PhysicalExpr>>,
}

impl PartitionExpr {
    fn new(
        calculator: PartitionValueCalculator,
        partition_spec: Arc<PartitionSpec>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self {
            calculator: Arc::new(calculator),
            partition_spec,
            children,
        }
    }
}

// Manual PartialEq/Eq implementations: pointer equality on the shared
// calculator/partition_spec instances, STRUCTURAL equality on the children —
// two PartitionExpr with the same calculator but different child expressions
// compute different values and must not compare equal.
impl PartialEq for PartitionExpr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.calculator, &other.calculator)
            && Arc::ptr_eq(&self.partition_spec, &other.partition_spec)
            && self.children == other.children
    }
}

impl Eq for PartitionExpr {}

impl PhysicalExpr for PartitionExpr {
    fn data_type(&self, _input_schema: &ArrowSchema) -> DFResult<DataType> {
        Ok(self.calculator.partition_arrow_type().clone())
    }

    fn nullable(&self, _input_schema: &ArrowSchema) -> DFResult<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> DFResult<ColumnarValue> {
        // Evaluate the children and feed THOSE arrays to the calculator —
        // never `batch.columns()`: after an optimizer rewrite the batch may
        // belong to a different plan node, and only the (rewritten) children
        // map this expression onto it correctly. A substituted child can
        // evaluate to a Scalar (e.g. a literal from a fused VALUES/SELECT
        // projection), so normalize each result to an array of the batch's
        // row count.
        let num_rows = batch.num_rows();
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.children.len());
        for child in &self.children {
            let value = child.evaluate(batch)?;
            columns.push(value.into_array(num_rows)?);
        }
        let array = self
            .calculator
            .calculate_from_columns(&columns, num_rows)
            .map_err(to_datafusion_error)?;
        Ok(ColumnarValue::Array(array))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.children.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        if children.len() != self.children.len() {
            return Err(DataFusionError::Internal(format!(
                "PartitionExpr::with_new_children expects exactly {} children \
                 (one per table column), got {}",
                self.children.len(),
                children.len()
            )));
        }
        Ok(Arc::new(PartitionExpr {
            calculator: Arc::clone(&self.calculator),
            partition_spec: Arc::clone(&self.partition_spec),
            children,
        }))
    }

    fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let field_names: Vec<String> = self
            .partition_spec
            .fields()
            .iter()
            .map(|pf| format!("{}({})", pf.transform, pf.name))
            .collect();
        write!(f, "iceberg_partition_values[{}]", field_names.join(", "))
    }
}

impl std::fmt::Display for PartitionExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let field_names: Vec<&str> = self
            .partition_spec
            .fields()
            .iter()
            .map(|pf| pf.name.as_str())
            .collect();
        write!(f, "iceberg_partition_values({})", field_names.join(", "))
    }
}

impl std::hash::Hash for PartitionExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Mirror PartialEq: pointer identity for the shared calculator and
        // partition_spec, structural hashing for the children.
        Arc::as_ptr(&self.calculator).hash(state);
        Arc::as_ptr(&self.partition_spec).hash(state);
        self.children.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{ArrayRef, Int32Array, StructArray};
    use datafusion::arrow::datatypes::{DataType, Field, Fields};
    use datafusion::physical_plan::empty::EmptyExec;
    use iceberg::spec::{NestedField, PrimitiveType, Schema, StructType, Transform, Type};

    use super::*;

    #[test]
    fn test_partition_calculator_basic() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = iceberg::spec::PartitionSpec::builder(Arc::new(table_schema.clone()))
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();

        // Verify partition type
        assert_eq!(calculator.partition_type().fields().len(), 1);
        assert_eq!(calculator.partition_type().fields()[0].name, "id_partition");
    }

    #[test]
    fn test_partition_expr_with_projection() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = Arc::new(
            iceberg::spec::PartitionSpec::builder(Arc::new(table_schema.clone()))
                .add_partition_field("id", "id_partition", Transform::Identity)
                .unwrap()
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let input = Arc::new(EmptyExec::new(arrow_schema.clone()));

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();

        let mut projection_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
            Vec::with_capacity(arrow_schema.fields().len() + 1);
        for (i, field) in arrow_schema.fields().iter().enumerate() {
            let column_expr = Arc::new(Column::new(field.name(), i));
            projection_exprs.push((column_expr, field.name().clone()));
        }

        let children: Vec<Arc<dyn PhysicalExpr>> = arrow_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, field)| Arc::new(Column::new(field.name(), i)) as Arc<dyn PhysicalExpr>)
            .collect();
        let partition_expr = Arc::new(PartitionExpr::new(calculator, partition_spec, children));
        projection_exprs.push((partition_expr, PROJECTED_PARTITION_VALUE_COLUMN.to_string()));

        let projection = ProjectionExec::try_new(projection_exprs, input).unwrap();
        let result = Arc::new(projection);

        assert_eq!(result.schema().fields().len(), 3);
        assert_eq!(result.schema().field(0).name(), "id");
        assert_eq!(result.schema().field(1).name(), "name");
        assert_eq!(result.schema().field(2).name(), "_partition");
    }

    #[test]
    fn test_partition_expr_evaluate() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = iceberg::spec::PartitionSpec::builder(Arc::new(table_schema.clone()))
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            Arc::new(datafusion::arrow::array::StringArray::from(vec![
                "a", "b", "c",
            ])),
        ])
        .unwrap();

        let partition_spec = Arc::new(partition_spec);
        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();
        let partition_type = calculator.partition_arrow_type().clone();
        let children: Vec<Arc<dyn PhysicalExpr>> = arrow_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, field)| Arc::new(Column::new(field.name(), i)) as Arc<dyn PhysicalExpr>)
            .collect();
        let expr = PartitionExpr::new(calculator, partition_spec, children);

        assert_eq!(expr.data_type(&arrow_schema).unwrap(), partition_type);
        assert!(!expr.nullable(&arrow_schema).unwrap());

        let result = expr.evaluate(&batch).unwrap();
        match result {
            ColumnarValue::Array(array) => {
                let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
                let id_partition = struct_array
                    .column_by_name("id_partition")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                assert_eq!(id_partition.value(0), 10);
                assert_eq!(id_partition.value(1), 20);
                assert_eq!(id_partition.value(2), 30);
            }
            _ => panic!("Expected array result"),
        }
    }

    /// Build the `{id int, data string}` identity(id) fixture used by the T7
    /// unit tests: (arrow schema, batch, partition spec, calculator-backed expr).
    fn t7_fixture() -> (Arc<ArrowSchema>, RecordBatch, Arc<PartitionExpr>) {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "data", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("build table schema");

        let partition_spec = Arc::new(
            iceberg::spec::PartitionSpec::builder(Arc::new(table_schema.clone()))
                .add_partition_field("id", "id_partition", Transform::Identity)
                .expect("add partition field")
                .build()
                .expect("build partition spec"),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            Arc::new(datafusion::arrow::array::StringArray::from(vec![
                "a", "b", "c",
            ])),
        ])
        .expect("build batch");

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema)
            .expect("build calculator");
        let children: Vec<Arc<dyn PhysicalExpr>> = arrow_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, field)| Arc::new(Column::new(field.name(), i)) as Arc<dyn PhysicalExpr>)
            .collect();
        let expr = Arc::new(PartitionExpr::new(calculator, partition_spec, children));
        (arrow_schema, batch, expr)
    }

    /// T7: the expression must declare one child per top-level input column,
    /// in schema order — this is what makes it visible to DataFusion's
    /// optimizer rewrites.
    #[test]
    fn test_partition_expr_children_one_per_input_column() {
        let (arrow_schema, _batch, expr) = t7_fixture();

        let children = expr.children();
        assert_eq!(
            children.len(),
            arrow_schema.fields().len(),
            "one child per input column"
        );
        for (i, (child, field)) in children.iter().zip(arrow_schema.fields()).enumerate() {
            let column = child
                .downcast_ref::<Column>()
                .expect("initial children are Column references");
            assert_eq!(column.name(), field.name().as_str(), "child {i} name");
            assert_eq!(column.index(), i, "child {i} index");
        }
    }

    /// T7: `with_new_children` must reject an arity mismatch with a typed
    /// error — never silently return the original expression.
    #[test]
    fn test_partition_expr_with_new_children_arity_mismatch_is_error() {
        let (_arrow_schema, _batch, expr) = t7_fixture();

        let one_child: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new("id", 0))];
        let result = Arc::clone(&expr).with_new_children(one_child);
        let err = result.expect_err("arity mismatch must be an error");
        assert!(
            err.to_string().contains("expects exactly 2 children"),
            "error must name the expected arity, got: {err}"
        );
    }

    /// T7: `with_new_children` must REBUILD the expression around the new
    /// children, and `evaluate` must compute from them — substituting a
    /// literal for the partition-source child must change the output.
    #[tokio::test]
    async fn test_partition_expr_with_new_children_rebuilds_and_evaluates() {
        use datafusion::common::ScalarValue;
        use datafusion::physical_expr::expressions::Literal;

        let (_arrow_schema, batch, expr) = t7_fixture();

        let new_children: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(Literal::new(ScalarValue::Int32(Some(7)))),
            Arc::new(Column::new("data", 1)),
        ];
        let rebuilt = Arc::clone(&expr)
            .with_new_children(new_children.clone())
            .expect("rebuild with matching arity");

        // The rebuilt expression must EXPOSE the new children...
        let rebuilt_children = rebuilt.children();
        assert_eq!(rebuilt_children.len(), 2);
        assert!(
            rebuilt_children[0].downcast_ref::<Literal>().is_some(),
            "rebuilt child 0 must be the substituted literal"
        );

        // ...and USE them: identity(id) over a literal 7 child (a Scalar
        // result, exercising the into_array normalization) is 7 on every row.
        let result = rebuilt.evaluate(&batch).expect("evaluate rebuilt expr");
        let array = match result {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => panic!("expected array result"),
        };
        let struct_array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("partition column is a struct");
        let id_partition = struct_array
            .column_by_name("id_partition")
            .expect("id_partition field")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id_partition is Int32");
        assert_eq!(id_partition.len(), 3, "one partition value per batch row");
        for i in 0..3 {
            assert_eq!(
                id_partition.value(i),
                7,
                "row {i}: partition value must come from the substituted literal child"
            );
        }
    }

    /// T7: equality and hashing must include the children — two expressions
    /// sharing the same calculator/spec instances but holding different
    /// children compute different values and must not compare equal (or the
    /// optimizer's common-subexpression machinery could substitute one for
    /// the other).
    #[test]
    fn test_partition_expr_eq_hash_include_children() {
        use std::hash::{DefaultHasher, Hash, Hasher};

        use datafusion::common::ScalarValue;
        use datafusion::physical_expr::expressions::Literal;

        let (_arrow_schema, _batch, expr) = t7_fixture();

        // Same calculator/spec Arcs, structurally identical children.
        let same_children: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(Column::new("id", 0)),
            Arc::new(Column::new("data", 1)),
        ];
        let rebuilt_same = Arc::clone(&expr)
            .with_new_children(same_children)
            .expect("rebuild with same-shaped children");
        let rebuilt_same = rebuilt_same
            .downcast_ref::<PartitionExpr>()
            .expect("rebuilt expr is a PartitionExpr")
            .clone();

        // Same calculator/spec Arcs, DIFFERENT children.
        let different_children: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(Literal::new(ScalarValue::Int32(Some(7)))),
            Arc::new(Column::new("data", 1)),
        ];
        let rebuilt_different = Arc::clone(&expr)
            .with_new_children(different_children)
            .expect("rebuild with different children");
        let rebuilt_different = rebuilt_different
            .downcast_ref::<PartitionExpr>()
            .expect("rebuilt expr is a PartitionExpr")
            .clone();

        assert_eq!(
            *expr, rebuilt_same,
            "same calculator/spec + structurally equal children => equal"
        );
        assert_ne!(
            *expr, rebuilt_different,
            "different children must make the expressions unequal even when \
             the calculator and partition spec instances are shared"
        );

        let hash_of = |e: &PartitionExpr| {
            let mut hasher = DefaultHasher::new();
            e.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(
            hash_of(&expr),
            hash_of(&rebuilt_same),
            "equal expressions must hash equally"
        );
        assert_ne!(
            hash_of(&expr),
            hash_of(&rebuilt_different),
            "children must contribute to the hash (deterministic here: the \
             pointer components are identical, only the children differ)"
        );
    }

    #[test]
    fn test_nested_partition() {
        let address_struct = StructType::new(vec![
            NestedField::required(3, "street", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::required(4, "city", Type::Primitive(PrimitiveType::String)).into(),
        ]);

        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "address", Type::Struct(address_struct)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = iceberg::spec::PartitionSpec::builder(Arc::new(table_schema.clone()))
            .add_partition_field("address.city", "city_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let struct_fields = Fields::from(vec![
            Field::new("street", DataType::Utf8, false),
            Field::new("city", DataType::Utf8, false),
        ]);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("address", DataType::Struct(struct_fields), false),
        ]));

        let street_array = Arc::new(datafusion::arrow::array::StringArray::from(vec![
            "123 Main St",
            "456 Oak Ave",
        ]));
        let city_array = Arc::new(datafusion::arrow::array::StringArray::from(vec![
            "New York",
            "Los Angeles",
        ]));

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("street", DataType::Utf8, false)),
                street_array as ArrayRef,
            ),
            (
                Arc::new(Field::new("city", DataType::Utf8, false)),
                city_array as ArrayRef,
            ),
        ]);

        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(struct_array),
        ])
        .unwrap();

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();
        let array = calculator.calculate(&batch).unwrap();

        let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
        let city_partition = struct_array
            .column_by_name("city_partition")
            .unwrap()
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .unwrap();

        assert_eq!(city_partition.value(0), "New York");
        assert_eq!(city_partition.value(1), "Los Angeles");
    }

    #[test]
    fn test_schema_validation_matching_schemas() {
        use iceberg::TableIdent;
        use iceberg::io::FileIO;
        use iceberg::spec::{FormatVersion, NestedField, PrimitiveType, Schema, Type};

        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = iceberg::spec::PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let sort_order = iceberg::spec::SortOrder::builder()
            .build(&table_schema)
            .unwrap();

        let table_metadata_builder = iceberg::spec::TableMetadataBuilder::new(
            (*table_schema).clone(),
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            std::collections::HashMap::new(),
        )
        .unwrap();

        let table_metadata = table_metadata_builder.build().unwrap();

        // Create Arrow schema matching the table schema
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let input = Arc::new(EmptyExec::new(arrow_schema));

        let table = iceberg::table::Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).unwrap())
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json".to_string())
            .build()
            .unwrap();

        let result = project_with_partition(input, &table);
        assert!(result.is_ok(), "Schema validation should pass");
    }

    #[test]
    fn test_schema_validation_mismatched_schemas() {
        use iceberg::TableIdent;
        use iceberg::io::FileIO;
        use iceberg::spec::{FormatVersion, NestedField, PrimitiveType, Schema, Type};

        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = iceberg::spec::PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let sort_order = iceberg::spec::SortOrder::builder()
            .build(&table_schema)
            .unwrap();

        let table_metadata_builder = iceberg::spec::TableMetadataBuilder::new(
            (*table_schema).clone(),
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            std::collections::HashMap::new(),
        )
        .unwrap();

        let table_metadata = table_metadata_builder.build().unwrap();

        // Create Arrow schema with different field name (mismatched)
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("different_name", DataType::Utf8, false), // Wrong field name
        ]));

        let input = Arc::new(EmptyExec::new(arrow_schema));

        let table = iceberg::table::Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).unwrap())
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json".to_string())
            .build()
            .unwrap();

        let result = project_with_partition(input, &table);
        assert!(
            result.is_err(),
            "Schema validation should fail for mismatched schemas"
        );
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Input schema does not match Iceberg table schema")
        );
    }

    /// T1 (hermetic): the partition column must be computed from the PROJECTED
    /// (computed) SELECT values even after `ProjectionPushdown` fuses the
    /// partition-bearing projection with the SELECT-list projection below it.
    ///
    /// Plan under test (hand-built, mirroring the provider's `insert_into` shape):
    ///
    /// ```text
    /// ProjectionExec[id, name, PartitionExpr]        <- project_with_partition
    ///   ProjectionExec[id@0 + 100 AS id, name@1]     <- computed SELECT item (non-trivial)
    ///     DataSourceExec[id=[1,2,3], name=[a,b,c]]
    /// ```
    ///
    /// `ProjectionPushdown` (runs twice in the real physical-optimizer pipeline)
    /// attempts `try_unifying_projections` on the adjacent pair. The assertion is on
    /// the EXECUTED `_partition` VALUES — never on plan shape — so it holds whether
    /// or not DataFusion fuses: the identity(id) partition value must equal the
    /// computed `id + 100` data value on every row.
    #[tokio::test]
    async fn test_partition_values_survive_projection_pushdown_fusion() {
        use datafusion::common::ScalarValue;
        use datafusion::common::config::ConfigOptions;
        use datafusion::datasource::memory::MemorySourceConfig;
        use datafusion::execution::TaskContext;
        use datafusion::logical_expr::Operator;
        use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
        use datafusion::physical_optimizer::PhysicalOptimizerRule;
        use datafusion::physical_optimizer::projection_pushdown::ProjectionPushdown;
        use datafusion::physical_plan::collect;
        use iceberg::TableIdent;
        use iceberg::io::FileIO;
        use iceberg::spec::FormatVersion;

        // Iceberg table {id int, name string} partitioned by identity(id).
        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .expect("build table schema"),
        );
        let partition_spec = iceberg::spec::PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id_partition", Transform::Identity)
            .expect("add identity(id) partition field")
            .build()
            .expect("build partition spec");
        let sort_order = iceberg::spec::SortOrder::builder()
            .build(&table_schema)
            .expect("build sort order");
        let table_metadata = iceberg::spec::TableMetadataBuilder::new(
            (*table_schema).clone(),
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            std::collections::HashMap::new(),
        )
        .expect("table metadata builder")
        .build()
        .expect("build table metadata");
        let table = iceberg::table::Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).expect("table ident"))
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json".to_string())
            .build()
            .expect("build table");

        // Source batch: id=[1,2,3], name=[a,b,c].
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(datafusion::arrow::array::StringArray::from(vec![
                "a", "b", "c",
            ])),
        ])
        .expect("build source batch");
        let source = MemorySourceConfig::try_new_exec(&[vec![batch]], arrow_schema, None)
            .expect("build memory source exec");

        // Inner projection: the computed SELECT list `id + 100 AS id, name`.
        let id_plus_100: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(100)))),
        ));
        let inner = Arc::new(
            ProjectionExec::try_new(
                vec![
                    (id_plus_100, "id".to_string()),
                    (
                        Arc::new(Column::new("name", 1)) as Arc<dyn PhysicalExpr>,
                        "name".to_string(),
                    ),
                ],
                source,
            )
            .expect("build inner (SELECT-list) projection"),
        );

        // Outer projection: passthroughs + the `_partition` expression.
        let plan = project_with_partition(inner, &table).expect("project_with_partition");

        // The optimizer pass that fuses adjacent projections (runs twice for real plans).
        let optimized = ProjectionPushdown::new()
            .optimize(plan, &ConfigOptions::default())
            .expect("run ProjectionPushdown");

        let batches = collect(optimized, Arc::new(TaskContext::default()))
            .await
            .expect("execute optimized plan");
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3, "all three source rows must flow through");

        let mut data_ids: Vec<i32> = Vec::new();
        let mut partition_ids: Vec<i32> = Vec::new();
        for batch in &batches {
            let schema = batch.schema();
            let id_idx = schema.index_of("id").expect("id column present");
            let part_idx = schema
                .index_of(PROJECTED_PARTITION_VALUE_COLUMN)
                .expect("_partition column present");
            let id_col = batch
                .column(id_idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id column is Int32");
            let part_col = batch
                .column(part_idx)
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("_partition column is a struct");
            let id_partition = part_col
                .column_by_name("id_partition")
                .expect("id_partition field present")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id_partition is Int32");
            for i in 0..batch.num_rows() {
                data_ids.push(id_col.value(i));
                partition_ids.push(id_partition.value(i));
            }
        }
        assert_eq!(
            data_ids,
            vec![101, 102, 103],
            "data column must carry the computed id + 100 values"
        );
        // The load-bearing assertion: identity(id) partition values must equal the
        // COMPUTED data values, not the raw source values [1, 2, 3].
        assert_eq!(
            partition_ids,
            vec![101, 102, 103],
            "identity(id) partition values must be computed from the projected \
             (id + 100) column, not from the raw source batch"
        );
    }

    // =======================================================================
    // G0 — write-compatibility (safe-direction nullability widening)
    // =======================================================================

    /// `{id int required (partition source), payload <ty> <nullable>}` as an
    /// Iceberg-shaped table plus a matching partitioned `Table`.
    ///
    /// The partition spec is `identity(id)`, so `payload` only ever exercises
    /// the schema validation — never the partition machinery.
    fn payload_table(payload: NestedField) -> iceberg::table::Table {
        use iceberg::TableIdent;
        use iceberg::io::FileIO;
        use iceberg::spec::FormatVersion;

        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    payload.into(),
                ])
                .build()
                .expect("build table schema"),
        );
        let partition_spec = iceberg::spec::PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id_partition", Transform::Identity)
            .expect("add identity(id) partition field")
            .build()
            .expect("build partition spec");
        let sort_order = iceberg::spec::SortOrder::builder()
            .build(&table_schema)
            .expect("build sort order");
        let table_metadata = iceberg::spec::TableMetadataBuilder::new(
            (*table_schema).clone(),
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            std::collections::HashMap::new(),
        )
        .expect("table metadata builder")
        .build()
        .expect("build table metadata");

        iceberg::table::Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).expect("table ident"))
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json".to_string())
            .build()
            .expect("build table")
    }

    /// Plan `project_with_partition` for [`payload_table`] against an input
    /// whose second column is `payload_field`.
    fn plan_with_payload_input(
        table: &iceberg::table::Table,
        payload_field: Field,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            payload_field,
        ]));
        project_with_partition(Arc::new(EmptyExec::new(arrow_schema)), table)
    }

    /// G0: a NON-nullable input column is accepted into an OPTIONAL (nullable)
    /// table column — the safe write direction, and the shape every FROM-less
    /// literal `INSERT` produces.
    #[test]
    fn test_schema_validation_accepts_required_input_into_optional_target() {
        let table = payload_table(NestedField::optional(
            2,
            "name",
            Type::Primitive(PrimitiveType::String),
        ));
        let result = plan_with_payload_input(&table, Field::new("name", DataType::Utf8, false));
        assert!(
            result.is_ok(),
            "a non-nullable input column must be accepted into an OPTIONAL \
             table column, got: {:?}",
            result.err()
        );
    }

    /// G0 NEGATIVE pin: the unsafe direction — a NULLABLE input column into a
    /// REQUIRED table column — keeps failing with the pre-existing message.
    #[test]
    fn test_schema_validation_rejects_nullable_input_into_required_target() {
        let table = payload_table(NestedField::required(
            2,
            "name",
            Type::Primitive(PrimitiveType::String),
        ));
        let result = plan_with_payload_input(&table, Field::new("name", DataType::Utf8, true));
        let err = result.expect_err("nullable input into a REQUIRED target must be rejected");
        assert!(
            err.to_string()
                .contains("Input schema does not match Iceberg table schema"),
            "the unsafe direction must keep failing loudly, got: {err}"
        );
    }

    /// G0: the widening is RECURSIVE — a struct whose child field is required
    /// in the input and optional in the table is accepted through the real
    /// `project_with_partition` entry point.
    #[test]
    fn test_schema_validation_accepts_nested_struct_widening() {
        let table = payload_table(NestedField::optional(
            2,
            "addr",
            Type::Struct(StructType::new(vec![
                NestedField::optional(3, "city", Type::Primitive(PrimitiveType::String)).into(),
            ])),
        ));
        // Input: same struct, but `city` is NOT nullable — safe direction.
        let input_struct = DataType::Struct(Fields::from(vec![Field::new(
            "city",
            DataType::Utf8,
            false,
        )]));
        let result = plan_with_payload_input(&table, Field::new("addr", input_struct, true));
        assert!(
            result.is_ok(),
            "a required NESTED field must be accepted into an optional nested \
             field, got: {:?}",
            result.err()
        );
    }

    /// G0 NEGATIVE pin (nested): a nullable nested input field into a required
    /// nested table field is still rejected — the recursion is one-directional
    /// at every level, not just the top.
    #[test]
    fn test_schema_validation_rejects_nested_struct_narrowing() {
        let table = payload_table(NestedField::optional(
            2,
            "addr",
            Type::Struct(StructType::new(vec![
                NestedField::required(3, "city", Type::Primitive(PrimitiveType::String)).into(),
            ])),
        ));
        let input_struct =
            DataType::Struct(Fields::from(vec![Field::new("city", DataType::Utf8, true)]));
        let result = plan_with_payload_input(&table, Field::new("addr", input_struct, true));
        let err =
            result.expect_err("nullable NESTED input into a required nested target is rejected");
        assert!(
            err.to_string()
                .contains("Input schema does not match Iceberg table schema"),
            "the unsafe nested direction must keep failing loudly, got: {err}"
        );
    }

    /// Build `{name, nullability}` field pairs for the comparator tests.
    fn utf8(name: &str, nullable: bool) -> Field {
        Field::new(name, DataType::Utf8, nullable)
    }

    /// A one-field struct type `{inner: Utf8 <nullable>}`.
    fn struct_of(inner: Field) -> DataType {
        DataType::Struct(Fields::from(vec![inner]))
    }

    /// An Iceberg-shaped Arrow map: `key_value: struct<key, value>`, entries
    /// field non-nullable.
    fn map_of(value: Field, sorted: bool) -> DataType {
        DataType::Map(
            Arc::new(Field::new(
                "key_value",
                DataType::Struct(Fields::from(vec![utf8("key", false), value])),
                false,
            )),
            sorted,
        )
    }

    /// G0 comparator: top-level nullability is one-directional; names and types
    /// stay strict.
    #[test]
    fn test_field_write_compatibility_top_level() {
        // required -> optional: the widening.
        assert!(field_is_write_compatible(
            &utf8("a", false),
            &utf8("a", true),
            0
        ));
        // equal nullability, both directions.
        assert!(field_is_write_compatible(
            &utf8("a", true),
            &utf8("a", true),
            0
        ));
        assert!(field_is_write_compatible(
            &utf8("a", false),
            &utf8("a", false),
            0
        ));
        // optional -> required: REJECTED.
        assert!(!field_is_write_compatible(
            &utf8("a", true),
            &utf8("a", false),
            0
        ));
        // names stay strict.
        assert!(!field_is_write_compatible(
            &utf8("a", false),
            &utf8("b", true),
            0
        ));
        // types stay strict, even in the widening direction.
        assert!(!field_is_write_compatible(
            &Field::new("a", DataType::Int32, false),
            &Field::new("a", DataType::Int64, true),
            0
        ));
    }

    /// G0 comparator: struct children widen in the safe direction only, and
    /// arity/name/type stay strict inside the struct.
    #[test]
    fn test_field_write_compatibility_nested_struct() {
        let required_inner = Field::new("s", struct_of(utf8("inner", false)), false);
        let optional_inner = Field::new("s", struct_of(utf8("inner", true)), false);

        assert!(field_is_write_compatible(
            &required_inner,
            &optional_inner,
            0
        ));
        assert!(!field_is_write_compatible(
            &optional_inner,
            &required_inner,
            0
        ));

        // Arity inside the struct stays strict.
        let two_fields = Field::new(
            "s",
            DataType::Struct(Fields::from(vec![
                utf8("inner", false),
                utf8("extra", true),
            ])),
            false,
        );
        assert!(!field_is_write_compatible(&two_fields, &optional_inner, 0));
        assert!(!field_is_write_compatible(&optional_inner, &two_fields, 0));

        // Nested field NAME stays strict.
        let renamed = Field::new("s", struct_of(utf8("other", true)), false);
        assert!(!field_is_write_compatible(&required_inner, &renamed, 0));
    }

    /// G0 comparator: list/large-list/fixed-size-list elements widen; the list
    /// KIND and the fixed size stay strict.
    #[test]
    fn test_field_write_compatibility_nested_list() {
        let required_element = Arc::new(utf8("element", false));
        let optional_element = Arc::new(utf8("element", true));

        let required_list = Field::new("l", DataType::List(required_element.clone()), true);
        let optional_list = Field::new("l", DataType::List(optional_element.clone()), true);
        assert!(field_is_write_compatible(&required_list, &optional_list, 0));
        assert!(!field_is_write_compatible(
            &optional_list,
            &required_list,
            0
        ));

        let required_large = Field::new("l", DataType::LargeList(required_element.clone()), true);
        let optional_large = Field::new("l", DataType::LargeList(optional_element.clone()), true);
        assert!(field_is_write_compatible(
            &required_large,
            &optional_large,
            0
        ));
        // List KIND is not interchangeable.
        assert!(!field_is_write_compatible(
            &required_large,
            &optional_list,
            0
        ));

        let required_fixed = Field::new(
            "l",
            DataType::FixedSizeList(required_element.clone(), 3),
            true,
        );
        let optional_fixed = Field::new(
            "l",
            DataType::FixedSizeList(optional_element.clone(), 3),
            true,
        );
        let optional_fixed_4 = Field::new("l", DataType::FixedSizeList(optional_element, 4), true);
        assert!(field_is_write_compatible(
            &required_fixed,
            &optional_fixed,
            0
        ));
        // The fixed size stays strict.
        assert!(!field_is_write_compatible(
            &required_fixed,
            &optional_fixed_4,
            0
        ));
    }

    /// G0 comparator: map VALUES widen; the `sorted` flag and the key stay
    /// strict.
    #[test]
    fn test_field_write_compatibility_nested_map() {
        let required_value = Field::new("m", map_of(utf8("value", false), false), true);
        let optional_value = Field::new("m", map_of(utf8("value", true), false), true);
        assert!(field_is_write_compatible(
            &required_value,
            &optional_value,
            0
        ));
        assert!(!field_is_write_compatible(
            &optional_value,
            &required_value,
            0
        ));

        // The `sorted` flag is part of the type.
        let optional_value_sorted = Field::new("m", map_of(utf8("value", true), true), true);
        assert!(!field_is_write_compatible(
            &required_value,
            &optional_value_sorted,
            0
        ));
    }

    /// G0: the recursion carries a depth limit; past it the comparison degrades
    /// to STRICT structural equality, so a deeper-than-limit widening is
    /// rejected while an identical deep type still compares equal. The guard can
    /// only ever reject — never accept something the strict rule would refuse.
    #[test]
    fn test_write_compatibility_depth_limit_falls_back_to_strict_equality() {
        /// Wrap `inner` in `depth` nested single-field structs.
        fn nest(inner: Field, depth: usize) -> Field {
            let mut field = inner;
            for _ in 0..depth {
                field = Field::new("s", struct_of(field), false);
            }
            field
        }

        let over = MAX_WRITE_COMPATIBILITY_DEPTH + 2;
        let deep_required = nest(utf8("leaf", false), over);
        let deep_optional = nest(utf8("leaf", true), over);

        // Identical deep types still compare equal (no regression).
        assert!(field_is_write_compatible(&deep_required, &deep_required, 0));
        // Beyond the limit the widening is NOT applied — strict equality wins.
        assert!(!field_is_write_compatible(
            &deep_required,
            &deep_optional,
            0
        ));
        // Just inside the limit the very same widening IS applied — proving the
        // rejection above comes from the depth guard, not from the shape.
        let shallow_required = nest(utf8("leaf", false), MAX_WRITE_COMPATIBILITY_DEPTH - 2);
        let shallow_optional = nest(utf8("leaf", true), MAX_WRITE_COMPATIBILITY_DEPTH - 2);
        assert!(field_is_write_compatible(
            &shallow_required,
            &shallow_optional,
            0
        ));
    }

    #[test]
    fn test_schema_validation_with_metadata_differences() {
        use std::collections::HashMap;

        use iceberg::TableIdent;
        use iceberg::io::FileIO;
        use iceberg::spec::{FormatVersion, NestedField, PrimitiveType, Schema, Type};

        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = iceberg::spec::PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let sort_order = iceberg::spec::SortOrder::builder()
            .build(&table_schema)
            .unwrap();

        let table_metadata_builder = iceberg::spec::TableMetadataBuilder::new(
            (*table_schema).clone(),
            partition_spec,
            sort_order,
            "/test/table".to_string(),
            FormatVersion::V2,
            std::collections::HashMap::new(),
        )
        .unwrap();

        let table_metadata = table_metadata_builder.build().unwrap();

        // Create Arrow schema with metadata (should be ignored in comparison)
        let mut metadata = HashMap::new();
        metadata.insert("extra".to_string(), "metadata".to_string());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(metadata.clone()),
            Field::new("name", DataType::Utf8, false).with_metadata(metadata),
        ]));

        let input = Arc::new(EmptyExec::new(arrow_schema));

        let table = iceberg::table::Table::builder()
            .metadata(table_metadata.metadata)
            .identifier(TableIdent::from_strs(["test", "table"]).unwrap())
            .file_io(FileIO::new_with_fs())
            .metadata_location("/test/metadata.json".to_string())
            .build()
            .unwrap();

        let result = project_with_partition(input, &table);
        assert!(
            result.is_ok(),
            "Schema validation should pass even with metadata differences"
        );
    }
}
