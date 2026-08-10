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

pub(crate) mod aggregate_evaluator;
pub(crate) mod bound_predicate_visitor;
pub(crate) mod expression_evaluator;
pub(crate) mod inclusive_metrics_evaluator;
pub(crate) mod inclusive_projection;
pub(crate) mod manifest_evaluator;
pub(crate) mod page_index_evaluator;
/// The **unbound** predicate visitor is test-only as of the iterative `rewrite_not` rewrite.
///
/// `RewriteNotVisitor` was its only implementor and `Predicate::rewrite_not` its only caller;
/// that method is now an explicit-stack walk, because routing an infallible `pub` API through a
/// fallible visitor forced an `.expect()` that turned a typed depth error into a process panic.
/// Both modules are retained under `cfg(test)` as the **differential oracle** that pins the
/// iterative rewrites against the visitor semantics they replaced (see
/// `predicate::tests::rewrite_not_matches_the_visitor_oracle`). The **bound** visitor below is
/// unaffected — a dozen evaluators still drive it in production.
#[cfg(test)]
pub(crate) mod predicate_visitor;
pub(crate) mod residual_evaluator;
#[cfg(test)]
pub(crate) mod rewrite_not;
pub(crate) mod row_group_metrics_evaluator;
pub(crate) mod strict_metrics_evaluator;
pub(crate) mod strict_projection;
