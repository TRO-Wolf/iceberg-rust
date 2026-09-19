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

use crate::error::Result;
use crate::spec::{NullOrder, SchemaRef, SortDirection, SortField, SortOrder, Transform};
use crate::table::Table;
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

/// Represents a sort field whose construction and validation are deferred until commit time.
/// This avoids the need to pass a `Table` reference into methods like `asc` or `desc` when
/// adding sort orders.
#[derive(Debug, PartialEq, Eq, Clone)]
struct PendingSortField {
    name: String,
    transform: Transform,
    direction: SortDirection,
    null_order: NullOrder,
}

impl PendingSortField {
    fn to_sort_field(&self, schema: &SchemaRef) -> Result<SortField> {
        self.check_supported_transform()?;

        let source_field = schema.field_by_name(self.name.as_str()).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot find field {} in table schema", self.name),
            )
        })?;

        if self.transform != Transform::Identity
            && self
                .transform
                .result_type(source_field.field_type.as_ref())
                .is_err()
        {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot bind: {} cannot transform {} values from '{}'",
                    self.transform, source_field.field_type, self.name
                ),
            ));
        }

        Ok(SortField::builder()
            .source_id(source_field.id)
            .transform(self.transform)
            .direction(self.direction)
            .null_order(self.null_order)
            .build())
    }

    fn check_supported_transform(&self) -> Result<()> {
        match self.transform {
            Transform::Void | Transform::Unknown => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "Transform is not supported: {}",
                    describe_sort_term(self.transform, &self.name)
                ),
            )),
            Transform::Bucket(width) | Transform::Truncate(width)
                if width == 0 || i32::try_from(width).is_err() =>
            {
                Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Unsupported width for transform: {}",
                        describe_sort_term(self.transform, &self.name)
                    ),
                ))
            }
            _ => Ok(()),
        }
    }
}

fn describe_sort_term(transform: Transform, name: &str) -> String {
    match transform {
        Transform::Bucket(num_buckets) => format!("bucket({num_buckets}, {name})"),
        Transform::Truncate(width) => format!("truncate({name}, {width})"),
        _ => format!("{transform}({name})"),
    }
}

/// Transaction action for replacing sort order.
pub struct ReplaceSortOrderAction {
    pending_sort_fields: Vec<PendingSortField>,
}

impl ReplaceSortOrderAction {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        ReplaceSortOrderAction {
            pending_sort_fields: vec![],
        }
    }

    /// Adds a field for sorting in ascending order.
    pub fn asc(self, name: &str, null_order: NullOrder) -> Self {
        self.add_sort_field(
            name,
            Transform::Identity,
            SortDirection::Ascending,
            null_order,
        )
    }

    /// Adds a field for sorting in descending order.
    pub fn desc(self, name: &str, null_order: NullOrder) -> Self {
        self.add_sort_field(
            name,
            Transform::Identity,
            SortDirection::Descending,
            null_order,
        )
    }

    #[allow(missing_docs)]
    pub fn sort_by(
        self,
        name: &str,
        transform: Transform,
        direction: SortDirection,
        null_order: NullOrder,
    ) -> Self {
        self.add_sort_field(name, transform, direction, null_order)
    }

    fn add_sort_field(
        mut self,
        name: &str,
        transform: Transform,
        sort_direction: SortDirection,
        null_order: NullOrder,
    ) -> Self {
        self.pending_sort_fields.push(PendingSortField {
            name: name.to_string(),
            transform,
            direction: sort_direction,
            null_order,
        });

        self
    }
}

impl Default for ReplaceSortOrderAction {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TransactionAction for ReplaceSortOrderAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let current_schema = table.metadata().current_schema();
        let sort_fields: Result<Vec<SortField>> = self
            .pending_sort_fields
            .iter()
            .map(|p| p.to_sort_field(current_schema))
            .collect();

        let bound_sort_order = SortOrder::builder()
            .with_fields(sort_fields?)
            .build(current_schema)?;

        let updates = vec![
            TableUpdate::AddSortOrder {
                sort_order: bound_sort_order,
            },
            TableUpdate::SetDefaultSortOrder { sort_order_id: -1 },
        ];

        let requirements = vec![
            TableRequirement::CurrentSchemaIdMatch {
                current_schema_id: current_schema.schema_id(),
            },
            TableRequirement::DefaultSortOrderIdMatch {
                default_sort_order_id: table.metadata().default_sort_order().order_id,
            },
        ];

        Ok(ActionCommit::new(updates, requirements))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use as_any::Downcast;

    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        NestedField, NullOrder, PrimitiveType, Schema, SortDirection, StructType, Transform, Type,
    };
    use crate::table::Table;
    use crate::transaction::sort_order::{PendingSortField, ReplaceSortOrderAction};
    use crate::transaction::tests::{make_v2_minimal_table_in_catalog, make_v2_table};
    use crate::transaction::{ApplyTransactionAction, Transaction, TransactionAction};
    use crate::{Catalog, ErrorKind, TableCreation, TableIdent};

    async fn commit_sort_order(
        catalog: &impl Catalog,
        table: &Table,
        action: ReplaceSortOrderAction,
    ) -> Table {
        let tx = Transaction::new(table);
        let tx = action.apply(tx).expect("apply sort order action");
        tx.commit(catalog).await.expect("commit sort order")
    }

    async fn make_xyz_table_in_catalog(catalog: &impl Catalog) -> Table {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "x", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "y", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(3, "z", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .expect("xyz schema");
        create_table_in_catalog(catalog, schema).await
    }

    async fn make_timestamp_table_in_catalog(catalog: &impl Catalog) -> Table {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "ts", Type::Primitive(PrimitiveType::Timestamp)).into(),
            ])
            .build()
            .expect("timestamp schema");
        create_table_in_catalog(catalog, schema).await
    }

    async fn make_nested_table_in_catalog(catalog: &impl Catalog) -> Table {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(
                    2,
                    "person",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String))
                            .into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .expect("nested schema");
        create_table_in_catalog(catalog, schema).await
    }

    async fn create_table_in_catalog(catalog: &impl Catalog, schema: Schema) -> Table {
        let table_ident =
            TableIdent::from_strs([format!("ns-{}", uuid::Uuid::new_v4()), "t".to_string()])
                .expect("table ident");
        catalog
            .create_namespace(table_ident.namespace(), HashMap::new())
            .await
            .expect("create namespace");
        catalog
            .create_table(
                table_ident.namespace(),
                TableCreation::builder()
                    .name(table_ident.name().to_string())
                    .schema(schema)
                    .build(),
            )
            .await
            .expect("create table")
    }

    #[test]
    fn test_replace_sort_order() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let replace_sort_order = tx.replace_sort_order();

        let tx = replace_sort_order
            .asc("x", NullOrder::First)
            .desc("y", NullOrder::Last)
            .apply(tx)
            .unwrap();

        let replace_sort_order = (*tx.actions[0])
            .downcast_ref::<ReplaceSortOrderAction>()
            .unwrap();

        assert_eq!(replace_sort_order.pending_sort_fields, vec![
            PendingSortField {
                name: String::from("x"),
                transform: Transform::Identity,
                direction: SortDirection::Ascending,
                null_order: NullOrder::First,
            },
            PendingSortField {
                name: String::from("y"),
                transform: Transform::Identity,
                direction: SortDirection::Descending,
                null_order: NullOrder::Last,
            }
        ]);
    }

    #[tokio::test]
    async fn test_sort_by_commits_transform_fields() {
        let catalog = new_memory_catalog().await;
        let table = make_xyz_table_in_catalog(&catalog).await;

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table)
                .replace_sort_order()
                .sort_by(
                    "x",
                    Transform::Bucket(4),
                    SortDirection::Ascending,
                    NullOrder::First,
                )
                .sort_by(
                    "z",
                    Transform::Truncate(10),
                    SortDirection::Descending,
                    NullOrder::Last,
                ),
        )
        .await;

        assert_eq!(table.metadata().default_sort_order_id(), 1);
        assert_eq!(table.metadata().sort_orders_iter().count(), 2);
        let fields = serde_json::to_value(&table.metadata().default_sort_order().fields)
            .expect("serialize sort fields");
        assert_eq!(
            fields,
            serde_json::json!([
                {
                    "transform": "bucket[4]",
                    "source-id": 1,
                    "direction": "asc",
                    "null-order": "nulls-first"
                },
                {
                    "transform": "truncate[10]",
                    "source-id": 3,
                    "direction": "desc",
                    "null-order": "nulls-last"
                }
            ])
        );
    }

    #[tokio::test]
    async fn test_sort_by_commits_temporal_transforms() {
        let catalog = new_memory_catalog().await;
        let table = make_timestamp_table_in_catalog(&catalog).await;

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table)
                .replace_sort_order()
                .sort_by(
                    "ts",
                    Transform::Day,
                    SortDirection::Ascending,
                    NullOrder::First,
                )
                .sort_by(
                    "ts",
                    Transform::Hour,
                    SortDirection::Descending,
                    NullOrder::First,
                ),
        )
        .await;

        let fields = serde_json::to_value(&table.metadata().default_sort_order().fields)
            .expect("serialize sort fields");
        assert_eq!(
            fields,
            serde_json::json!([
                {
                    "transform": "day",
                    "source-id": 2,
                    "direction": "asc",
                    "null-order": "nulls-first"
                },
                {
                    "transform": "hour",
                    "source-id": 2,
                    "direction": "desc",
                    "null-order": "nulls-first"
                }
            ])
        );
    }

    #[tokio::test]
    async fn test_reapplied_equal_sort_order_reuses_its_order_id() {
        let catalog = new_memory_catalog().await;
        let table = make_xyz_table_in_catalog(&catalog).await;

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table).replace_sort_order().sort_by(
                "x",
                Transform::Bucket(4),
                SortDirection::Ascending,
                NullOrder::First,
            ),
        )
        .await;
        assert_eq!(table.metadata().default_sort_order_id(), 1);

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table)
                .replace_sort_order()
                .asc("y", NullOrder::First),
        )
        .await;
        assert_eq!(table.metadata().default_sort_order_id(), 2);

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table).replace_sort_order().sort_by(
                "x",
                Transform::Bucket(4),
                SortDirection::Ascending,
                NullOrder::First,
            ),
        )
        .await;
        assert_eq!(
            table.metadata().default_sort_order_id(),
            1,
            "an order equal to an earlier one reuses that order's id"
        );
        assert_eq!(
            table.metadata().sort_orders_iter().count(),
            3,
            "reapplying an existing order adds no new sort-orders entry"
        );
    }

    #[tokio::test]
    async fn test_empty_sort_order_resets_default_to_unsorted() {
        let catalog = new_memory_catalog().await;
        let table = make_xyz_table_in_catalog(&catalog).await;

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table).replace_sort_order().sort_by(
                "x",
                Transform::Bucket(4),
                SortDirection::Ascending,
                NullOrder::First,
            ),
        )
        .await;
        assert_eq!(table.metadata().default_sort_order_id(), 1);

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table).replace_sort_order(),
        )
        .await;
        assert_eq!(
            table.metadata().default_sort_order_id(),
            0,
            "an action with no fields resets the default to the unsorted order"
        );
        assert_eq!(table.metadata().sort_orders_iter().count(), 2);
    }

    #[tokio::test]
    async fn test_sort_by_rejects_bad_transform_widths() {
        let table = make_v2_table();

        let error = Arc::new(Transaction::new(&table).replace_sort_order().sort_by(
            "x",
            Transform::Bucket(0),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .commit(&table)
        .await
        .err()
        .expect("bucket(0) must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            error.message(),
            "Unsupported width for transform: bucket(0, x)"
        );

        let error = Arc::new(Transaction::new(&table).replace_sort_order().sort_by(
            "z",
            Transform::Truncate(0),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .commit(&table)
        .await
        .err()
        .expect("truncate(0) must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            error.message(),
            "Unsupported width for transform: truncate(z, 0)"
        );

        let error = Arc::new(Transaction::new(&table).replace_sort_order().sort_by(
            "x",
            Transform::Bucket(2147483648),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .commit(&table)
        .await
        .err()
        .expect("bucket width above the Java int maximum must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            error.message(),
            "Unsupported width for transform: bucket(2147483648, x)"
        );
    }

    #[tokio::test]
    async fn test_sort_by_rejects_unsupported_transforms() {
        let table = make_v2_table();

        let error = Arc::new(Transaction::new(&table).replace_sort_order().sort_by(
            "x",
            Transform::Void,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .commit(&table)
        .await
        .err()
        .expect("void must be rejected");
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        assert_eq!(error.message(), "Transform is not supported: void(x)");

        let error = Arc::new(Transaction::new(&table).replace_sort_order().sort_by(
            "x",
            Transform::Unknown,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .commit(&table)
        .await
        .err()
        .expect("unknown must be rejected");
        assert_eq!(error.kind(), ErrorKind::FeatureUnsupported);
        assert_eq!(error.message(), "Transform is not supported: unknown(x)");
    }

    #[tokio::test]
    async fn test_sort_by_rejects_transform_type_mismatch() {
        let table = make_v2_table();

        let error = Arc::new(Transaction::new(&table).replace_sort_order().sort_by(
            "x",
            Transform::Day,
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .commit(&table)
        .await
        .err()
        .expect("day on a long column must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            error.message(),
            "Cannot bind: day cannot transform long values from 'x'"
        );
    }

    #[tokio::test]
    async fn test_sort_by_keeps_duplicate_terms() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table)
                .replace_sort_order()
                .sort_by(
                    "x",
                    Transform::Bucket(4),
                    SortDirection::Ascending,
                    NullOrder::First,
                )
                .sort_by(
                    "x",
                    Transform::Bucket(4),
                    SortDirection::Ascending,
                    NullOrder::First,
                ),
        )
        .await;

        let order = table.metadata().default_sort_order();
        assert_eq!(order.fields.len(), 2);
        assert_eq!(order.fields[0], order.fields[1]);
    }

    #[tokio::test]
    async fn test_sort_by_binds_nested_column_source() {
        let catalog = new_memory_catalog().await;
        let table = make_nested_table_in_catalog(&catalog).await;

        let table = commit_sort_order(
            &catalog,
            &table,
            Transaction::new(&table).replace_sort_order().sort_by(
                "person.name",
                Transform::Truncate(3),
                SortDirection::Ascending,
                NullOrder::First,
            ),
        )
        .await;

        let fields = serde_json::to_value(&table.metadata().default_sort_order().fields)
            .expect("serialize sort fields");
        assert_eq!(
            fields,
            serde_json::json!([{
                "transform": "truncate[3]",
                "source-id": 3,
                "direction": "asc",
                "null-order": "nulls-first"
            }])
        );
    }

    #[tokio::test]
    async fn test_sort_by_rejects_transform_on_struct_source() {
        let catalog = new_memory_catalog().await;
        let table = make_nested_table_in_catalog(&catalog).await;

        let error = Arc::new(Transaction::new(&table).replace_sort_order().sort_by(
            "person",
            Transform::Bucket(4),
            SortDirection::Ascending,
            NullOrder::First,
        ))
        .commit(&table)
        .await
        .err()
        .expect("bucket on a struct source must be rejected");
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(
            error
                .message()
                .starts_with("Cannot bind: bucket[4] cannot transform struct"),
            "unexpected message: {}",
            error.message()
        );
        assert!(
            error.message().ends_with("values from 'person'"),
            "unexpected message: {}",
            error.message()
        );
    }
}
