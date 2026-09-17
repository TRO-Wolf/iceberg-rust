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
        .bind(schema, case_sensitive)?;

    for file in files {
        if partition_might_match(current, &bound_filter, file)?
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
) -> Result<bool> {
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
    ExpressionEvaluator::new(projected).eval(file)
}
