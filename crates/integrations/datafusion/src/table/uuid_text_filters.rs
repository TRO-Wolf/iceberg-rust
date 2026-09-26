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

use datafusion::error::Result as DFResult;
use datafusion::logical_expr::expr::{InList, Like};
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use datafusion::scalar::ScalarValue;
use iceberg::spec::{PrimitiveType, Schema as IcebergSchema, Type as IcebergType};
use iceberg::{Error, ErrorKind};

use super::uuid_text::{is_canonical_uuid_text, parse_uuid_text, uuid_parse_error};
use crate::to_datafusion_error;

pub(crate) fn uuid_column_ids(schema: &IcebergSchema) -> HashMap<String, i32> {
    schema
        .as_struct()
        .fields()
        .iter()
        .filter(|field| {
            matches!(
                field.field_type.as_ref(),
                IcebergType::Primitive(PrimitiveType::Uuid)
            )
        })
        .map(|field| (field.name.clone(), field.id))
        .collect()
}

pub(crate) fn rewrite_uuid_text_filters(filters: &[Expr], schema: &IcebergSchema) -> Vec<Expr> {
    let uuid_columns = uuid_column_ids(schema);
    filters
        .iter()
        .map(|filter| rewrite_uuid_text_expr(filter, &uuid_columns, false))
        .collect()
}

fn rewrite_uuid_text_expr(expr: &Expr, uuid_columns: &HashMap<String, i32>, negated: bool) -> Expr {
    match expr {
        Expr::Not(inner) => Expr::Not(Box::new(rewrite_uuid_text_expr(
            inner,
            uuid_columns,
            !negated,
        ))),
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And | Operator::Or => Expr::BinaryExpr(BinaryExpr::new(
                Box::new(rewrite_uuid_text_expr(&binary.left, uuid_columns, negated)),
                binary.op,
                Box::new(rewrite_uuid_text_expr(&binary.right, uuid_columns, negated)),
            )),
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => rewrite_uuid_comparison(
                &binary.left,
                binary.op,
                &binary.right,
                uuid_columns,
                negated,
            )
            .unwrap_or_else(|| expr.clone()),
            _ => expr.clone(),
        },
        Expr::InList(inlist) => {
            rewrite_uuid_in_list(inlist, uuid_columns, negated).unwrap_or_else(|| expr.clone())
        }
        _ => expr.clone(),
    }
}

fn string_literal_value(value: &ScalarValue) -> Option<String> {
    match value {
        ScalarValue::Utf8(Some(text))
        | ScalarValue::LargeUtf8(Some(text))
        | ScalarValue::Utf8View(Some(text)) => Some(text.clone()),
        _ => None,
    }
}

fn uuid_byte_literal(bytes: [u8; 16]) -> Expr {
    Expr::Literal(ScalarValue::FixedSizeBinary(16, Some(bytes.to_vec())), None)
}

fn uuid_column_literal(
    left: &Expr,
    right: &Expr,
    uuid_columns: &HashMap<String, i32>,
) -> Option<(String, bool)> {
    let (name, text, swapped) = match (left, right) {
        (Expr::Column(column), Expr::Literal(value, _)) => {
            (&column.name, string_literal_value(value)?, false)
        }
        (Expr::Literal(value, _), Expr::Column(column)) => {
            (&column.name, string_literal_value(value)?, true)
        }
        _ => return None,
    };
    uuid_columns.contains_key(name).then_some((text, swapped))
}

fn rewrite_uuid_comparison(
    left: &Expr,
    op: Operator,
    right: &Expr,
    uuid_columns: &HashMap<String, i32>,
    negated: bool,
) -> Option<Expr> {
    let (text, swapped) = uuid_column_literal(left, right, uuid_columns)?;
    let bytes = parse_uuid_text(&text).ok()?;
    let effective_not_eq = (op == Operator::NotEq) != negated;
    let is_range = matches!(
        op,
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
    );
    if (is_range || effective_not_eq) && !is_canonical_uuid_text(&text) {
        return None;
    }
    let literal = uuid_byte_literal(bytes);
    if swapped {
        Some(Expr::BinaryExpr(BinaryExpr::new(
            Box::new(literal),
            op,
            Box::new(right.clone()),
        )))
    } else {
        Some(Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left.clone()),
            op,
            Box::new(literal),
        )))
    }
}

fn rewrite_uuid_in_list(
    inlist: &InList,
    uuid_columns: &HashMap<String, i32>,
    negated: bool,
) -> Option<Expr> {
    let Expr::Column(column) = inlist.expr.as_ref() else {
        return None;
    };
    if !uuid_columns.contains_key(&column.name) {
        return None;
    }
    let effective_negated = inlist.negated != negated;
    let mut literals = Vec::with_capacity(inlist.list.len());
    for item in &inlist.list {
        let Expr::Literal(value, _) = item else {
            return None;
        };
        let text = string_literal_value(value)?;
        if effective_negated && !is_canonical_uuid_text(&text) {
            return None;
        }
        literals.push(uuid_byte_literal(parse_uuid_text(&text).ok()?));
    }
    Some(Expr::InList(InList::new(
        inlist.expr.clone(),
        literals,
        inlist.negated,
    )))
}

pub(crate) fn refuse_unbindable_uuid_delete_filters(
    filters: &[Expr],
    schema: &IcebergSchema,
) -> DFResult<()> {
    let uuid_columns = uuid_column_ids(schema);
    filters
        .iter()
        .try_for_each(|filter| refuse_unbindable_uuid_expr(filter, &uuid_columns))
}

fn refuse_unbindable_uuid_expr(expr: &Expr, uuid_columns: &HashMap<String, i32>) -> DFResult<()> {
    match expr {
        Expr::Not(inner) => refuse_unbindable_uuid_expr(inner, uuid_columns),
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And | Operator::Or => {
                refuse_unbindable_uuid_expr(&binary.left, uuid_columns)?;
                refuse_unbindable_uuid_expr(&binary.right, uuid_columns)
            }
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => {
                match uuid_column_literal(&binary.left, &binary.right, uuid_columns) {
                    Some((text, _)) => refuse_unparsable(&text),
                    None => Ok(()),
                }
            }
            _ => Ok(()),
        },
        Expr::InList(inlist) => {
            let Expr::Column(column) = inlist.expr.as_ref() else {
                return Ok(());
            };
            if !uuid_columns.contains_key(&column.name) {
                return Ok(());
            }
            for item in &inlist.list {
                if let Expr::Literal(value, _) = item
                    && let Some(text) = string_literal_value(value)
                {
                    refuse_unparsable(&text)?;
                }
            }
            Ok(())
        }
        Expr::Like(like) => refuse_uuid_like(like, uuid_columns),
        _ => Ok(()),
    }
}

fn refuse_unparsable(text: &str) -> DFResult<()> {
    parse_uuid_text(text)
        .map(|_| ())
        .map_err(|message| uuid_parse_error(&message))
}

fn refuse_uuid_like(like: &Like, uuid_columns: &HashMap<String, i32>) -> DFResult<()> {
    if like.case_insensitive || like.escape_char.is_some() {
        return Ok(());
    }
    let Expr::Column(column) = like.expr.as_ref() else {
        return Ok(());
    };
    let Some(field_id) = uuid_columns.get(&column.name) else {
        return Ok(());
    };
    let Expr::Literal(value, _) = like.pattern.as_ref() else {
        return Ok(());
    };
    let Some(pattern) = string_literal_value(value) else {
        return Ok(());
    };
    let is_wildcard = |c: char| c == '%' || c == '_' || c == '\\';
    if !pattern.contains(is_wildcard) {
        return if like.negated {
            Ok(())
        } else {
            refuse_unparsable(&pattern)
        };
    }
    match pattern.strip_suffix('%') {
        Some(prefix) if !prefix.contains(is_wildcard) => Err(to_datafusion_error(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "Term for STARTS_WITH or NOT_STARTS_WITH must produce a string: ref(id={field_id}, accessor-type=uuid): uuid"
            ),
        ))),
        _ => Ok(()),
    }
}
