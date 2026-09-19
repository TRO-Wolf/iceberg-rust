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

use apache_avro::Schema as AvroSchema;
use apache_avro::schema::{
    DecimalSchema, FixedSchema, Name, RecordField as AvroRecordField, RecordSchema, UnionSchema,
};

use crate::avro::name::{iceberg_field_name, uniquified_avro_names};
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Copy)]
pub(crate) enum AvroNameCollision {
    Fail,
    Uniquify,
}

pub(crate) fn avro_record_schema(
    name: &str,
    mut fields: Vec<AvroRecordField>,
    collision: AvroNameCollision,
) -> Result<AvroSchema> {
    match collision {
        AvroNameCollision::Fail => {
            for (i, field) in fields.iter().enumerate() {
                if let Some(other) = fields[..i].iter().find(|f| f.name == field.name) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Iceberg fields '{}' and '{}' both produce Avro field name '{}'",
                            iceberg_field_name(other),
                            iceberg_field_name(field),
                            field.name
                        ),
                    ));
                }
            }
        }
        AvroNameCollision::Uniquify => {
            let entries: Vec<(String, String)> = fields
                .iter()
                .map(|f| (f.name.clone(), iceberg_field_name(f).to_string()))
                .collect();
            for (field, name) in fields.iter_mut().zip(uniquified_avro_names(&entries)) {
                field.name = name;
            }
        }
    }
    let lookup = fields
        .iter()
        .enumerate()
        .map(|f| (f.1.name.clone(), f.0))
        .collect();

    Ok(AvroSchema::Record(RecordSchema {
        name: Name::new(name)?,
        aliases: None,
        doc: None,
        fields,
        lookup,
        attributes: Default::default(),
    }))
}

pub(crate) fn avro_fixed_schema(len: usize) -> Result<AvroSchema> {
    Ok(AvroSchema::Fixed(FixedSchema {
        name: Name::new(format!("fixed_{len}").as_str())?,
        aliases: None,
        doc: None,
        size: len,
        attributes: Default::default(),
        default: None,
    }))
}

pub(crate) fn avro_decimal_schema(precision: usize, scale: usize) -> Result<AvroSchema> {
    Ok(AvroSchema::Decimal(DecimalSchema {
        precision,
        scale,
        inner: Box::new(AvroSchema::Fixed(FixedSchema {
            name: Name::new(&format!("decimal_{precision}_{scale}")).unwrap(),
            aliases: None,
            doc: None,
            size: crate::spec::Type::decimal_required_bytes(precision as u32)? as usize,
            attributes: Default::default(),
            default: None,
        })),
    }))
}

pub(crate) fn avro_optional(avro_schema: AvroSchema) -> Result<AvroSchema> {
    Ok(AvroSchema::Union(UnionSchema::new(vec![
        AvroSchema::Null,
        avro_schema,
    ])?))
}

pub(crate) fn is_avro_optional(avro_schema: &AvroSchema) -> bool {
    match avro_schema {
        AvroSchema::Union(union) => union.is_nullable(),
        _ => false,
    }
}
