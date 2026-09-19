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

use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, de};

use crate::spec::{NestedFieldRef, StructType};

impl<'de> Deserialize<'de> for StructType {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where D: Deserializer<'de> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Type,
            Fields,
        }

        struct StructTypeVisitor;

        impl<'de> Visitor<'de> for StructTypeVisitor {
            type Value = StructType;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<StructType, V::Error>
            where V: MapAccess<'de> {
                let mut fields = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Type => {
                            let type_val: String = map.next_value()?;
                            if type_val != "struct" {
                                return Err(serde::de::Error::custom(format!(
                                    "expected type 'struct', got '{type_val}'"
                                )));
                            }
                        }
                        Field::Fields => {
                            if fields.is_some() {
                                return Err(serde::de::Error::duplicate_field("fields"));
                            }
                            fields = Some(map.next_value()?);
                        }
                    }
                }
                let fields: Vec<NestedFieldRef> =
                    fields.ok_or_else(|| de::Error::missing_field("fields"))?;

                Ok(StructType::new(fields))
            }
        }

        const FIELDS: &[&str] = &["type", "fields"];
        deserializer.deserialize_struct("struct", FIELDS, StructTypeVisitor)
    }
}
