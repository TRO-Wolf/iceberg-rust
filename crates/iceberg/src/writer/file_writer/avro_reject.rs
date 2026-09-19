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

use crate::spec::{NestedField, PrimitiveType, StructType, Type};
use crate::{Error, ErrorKind, Result};

pub(super) fn reject_unsupported_types(struct_type: &StructType) -> Result<()> {
    for field in struct_type.fields() {
        reject_unsupported_field(field)?;
    }
    Ok(())
}

fn reject_unsupported_field(field: &NestedField) -> Result<()> {
    match field.field_type.as_ref() {
        Type::Variant => Err(unsupported_field_err(
            field,
            "variant type (the reader rejects it on read)",
        )),
        Type::Primitive(PrimitiveType::Unknown) => Err(unsupported_field_err(
            field,
            "unknown type yet (the always-null read path is deferred)",
        )),
        Type::Struct(s) => reject_unsupported_types(s),
        Type::List(l) => reject_unsupported_field(&l.element_field),
        Type::Map(m) => {
            reject_unsupported_field(&m.key_field)?;
            reject_unsupported_field(&m.value_field)
        }
        Type::Primitive(_) => Ok(()),
    }
}

fn unsupported_field_err(field: &NestedField, what: &str) -> Error {
    Error::new(
        ErrorKind::FeatureUnsupported,
        format!("Avro data writer does not support the {what}"),
    )
    .with_context("field_id", field.id.to_string())
    .with_context("field_name", field.name.clone())
}
