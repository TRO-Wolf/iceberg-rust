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

use crate::error::Result;
use crate::expr::visitors::expression_evaluator::ExpressionEvaluator;
use crate::expr::visitors::inclusive_metrics_evaluator::InclusiveMetricsEvaluator;
use crate::expr::visitors::inclusive_projection::InclusiveProjection;
use crate::expr::{Bind, BoundPredicate, Predicate};
use crate::spec::{DataFile, Schema};
use crate::table::Table;

/// Return the first file in `files` that COULD contain records matching `conflict_filter` — the shared
/// per-file conflict test behind the added-data, added-delete, and deleted-data validation walks.
/// Binds the filter once (`None` = `AlwaysTrue`); per file, the file's own spec partition projection
/// runs first, then [`InclusiveMetricsEvaluator`] (Java `ManifestGroup.filterData`). Unknown-spec files
/// stay conflicting. Returns the first match, or `None` when nothing can match.
pub(crate) fn first_conflicting_file(
    files: &[DataFile],
    current: &Table,
    conflict_filter: Option<&Predicate>,
    case_sensitive: bool,
) -> Result<Option<DataFile>> {
    if files.is_empty() {
        return Ok(None);
    }

    let schema = current.metadata().current_schema().clone();
    let bound_filter: BoundPredicate = conflict_filter
        .cloned()
        .unwrap_or(Predicate::AlwaysTrue)
        .rewrite_not()
        .bind(schema, case_sensitive)?;

    let mut partition_evaluators: HashMap<i32, ExpressionEvaluator> = HashMap::new();
    for file in files {
        if partition_might_match(current, &bound_filter, file, &mut partition_evaluators)?
            && InclusiveMetricsEvaluator::eval(&bound_filter, file, true)?
        {
            return Ok(Some(file.clone()));
        }
    }

    Ok(None)
}

fn partition_might_match(
    current: &Table,
    bound_filter: &BoundPredicate,
    file: &DataFile,
    partition_evaluators: &mut HashMap<i32, ExpressionEvaluator>,
) -> Result<bool> {
    if let Some(evaluator) = partition_evaluators.get(&file.partition_spec_id) {
        return evaluator.eval(file);
    }
    let Some(partition_spec) = current
        .metadata()
        .partition_spec_by_id(file.partition_spec_id)
    else {
        return Ok(true);
    };
    let schema = current.metadata().current_schema();
    let partition_type = partition_spec.partition_type(schema)?;
    let partition_schema = Arc::new(
        Schema::builder()
            .with_schema_id(partition_spec.spec_id())
            .with_fields(partition_type.fields().to_owned())
            .build()?,
    );
    let projected = InclusiveProjection::new(partition_spec.clone())
        .project(bound_filter)?
        .rewrite_not()
        .bind(partition_schema, true)?;
    let evaluator = ExpressionEvaluator::new(projected);
    let matches = evaluator.eval(file)?;
    partition_evaluators.insert(file.partition_spec_id, evaluator);
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::first_conflicting_file;
    use crate::expr::Reference;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Datum, ListType, Literal, NestedField,
        PrimitiveType, Schema, Struct, Type,
    };
    use crate::transaction::tests::make_v2_minimal_table_in_catalog;
    use crate::{Catalog, NamespaceIdent, TableCreation};

    #[tokio::test]
    async fn unknown_spec_id_stays_conflicting() {
        let catalog = new_memory_catalog().await;
        let table = make_v2_minimal_table_in_catalog(&catalog).await;
        let file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/unknown-spec.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(999)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .build()
            .expect("build unknown-spec file");
        let filter = Reference::new("x").equal_to(Datum::long(1));
        let conflicting = first_conflicting_file(&[file], &table, Some(&filter), true)
            .expect("an unknown spec must not error")
            .expect("an unknown spec must stay conflicting");
        assert_eq!(conflicting.file_path(), "test/unknown-spec.parquet");
    }

    async fn table_with_list_column(catalog: &impl Catalog) -> crate::table::Table {
        let namespace = NamespaceIdent::new("ns1".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .expect("namespace");
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                Arc::new(NestedField::optional(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                )),
                Arc::new(NestedField::optional(
                    2,
                    "xs",
                    Type::List(ListType::new(
                        NestedField::list_element(3, Type::Primitive(PrimitiveType::Int), false)
                            .into(),
                    )),
                )),
            ])
            .build()
            .expect("schema");
        catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .schema(schema)
                    .build(),
            )
            .await
            .expect("create table")
    }

    fn data_file(path: &str, upper_id: i32) -> crate::spec::DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .value_counts(HashMap::from([(1, 1u64)]))
            .upper_bounds(HashMap::from([(1, Datum::int(upper_id))]))
            .build()
            .expect("build data file")
    }

    #[tokio::test]
    async fn unbindable_conflict_filter_widens_instead_of_failing_to_bind() {
        let catalog = new_memory_catalog().await;
        let table = table_with_list_column(&catalog).await;
        let filter = Reference::new("id")
            .greater_than(Datum::int(1))
            .and(Reference::new("xs").is_null());

        let conflicting = first_conflicting_file(
            &[data_file("test/maybe.parquet", 4)],
            &table,
            Some(&filter),
            true,
        )
        .expect("an unbindable term must widen, not error")
        .expect("a file the sound conjunct might match stays conflicting");
        assert_eq!(conflicting.file_path(), "test/maybe.parquet");

        let none = first_conflicting_file(
            &[data_file("test/cannot.parquet", 1)],
            &table,
            Some(&filter),
            true,
        )
        .expect("an unbindable term must widen, not error");
        assert!(
            none.is_none(),
            "a file whose id upper bound is 1 cannot match id > 1"
        );
    }
}
