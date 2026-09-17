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

use std::borrow::Cow;
use std::sync::Arc;

use crate::Result;
use crate::expr::BoundReference;
use crate::spec::{
    DataFile, Datum, Literal, ManifestEntry, PartitionSpec, PrimitiveLiteral, PrimitiveType,
    Schema, Struct, StructType, TableMetadata, Type, is_promotion_allowed,
};

impl Datum {
    pub(crate) fn promoted_to(&self, target: &Type) -> Cow<'_, Datum> {
        match target {
            Type::Primitive(primitive)
                if primitive != self.data_type()
                    && is_promotion_allowed(
                        &Type::Primitive(self.data_type().clone()),
                        primitive,
                    ) =>
            {
                Cow::Owned(Datum::new(
                    primitive.clone(),
                    self.literal().promote_to(primitive),
                ))
            }
            _ => Cow::Borrowed(self),
        }
    }

    pub(crate) fn physical(field_type: &PrimitiveType, literal: PrimitiveLiteral) -> Datum {
        Datum::new(field_type.clone(), literal.promote_to(field_type))
    }
}

impl DataFile {
    pub(crate) fn promoted_lower_bound(
        &self,
        reference: &BoundReference,
    ) -> Option<Cow<'_, Datum>> {
        let bound = self.lower_bounds.get(&reference.field().id)?;
        Some(bound.promoted_to(&reference.field().field_type))
    }

    pub(crate) fn promoted_upper_bound(
        &self,
        reference: &BoundReference,
    ) -> Option<Cow<'_, Datum>> {
        let bound = self.upper_bounds.get(&reference.field().id)?;
        Some(bound.promoted_to(&reference.field().field_type))
    }
}

fn promotable_slot(slot: &Option<Literal>, field_type: &Type) -> Option<PrimitiveLiteral> {
    match (slot, field_type) {
        (Some(Literal::Primitive(literal)), Type::Primitive(target))
            if !target.compatible(literal) =>
        {
            let promoted = literal.promote_to(target);
            target.compatible(&promoted).then_some(promoted)
        }
        _ => None,
    }
}

impl Struct {
    pub(crate) fn promoted_to(&self, partition_type: &StructType) -> Option<Struct> {
        let fields = partition_type.fields();
        if fields.len() < self.fields().len()
            || !self
                .fields()
                .iter()
                .zip(fields)
                .any(|(slot, field)| promotable_slot(slot, &field.field_type).is_some())
        {
            return None;
        }
        Some(Struct::from_iter(self.fields().iter().zip(fields).map(
            |(slot, field)| match promotable_slot(slot, &field.field_type) {
                Some(promoted) => Some(Literal::Primitive(promoted)),
                None => slot.clone(),
            },
        )))
    }
}

impl PartitionSpec {
    pub(crate) fn validated_promoted_partition(
        &self,
        data: Struct,
        schema: &Schema,
    ) -> Result<Struct> {
        let data = match self.partition_type(schema) {
            Ok(partition_type) => data.promoted_to(&partition_type).unwrap_or(data),
            Err(_) => data,
        };
        self.validate_partition_data(&data, schema)?;
        Ok(data)
    }
}

impl TableMetadata {
    pub(crate) fn current_partition_key(&self, data_file: &DataFile) -> (i32, Struct) {
        let promoted = self
            .partition_spec_by_id(data_file.partition_spec_id)
            .and_then(|spec| spec.partition_type(self.current_schema()).ok())
            .and_then(|partition_type| data_file.partition.promoted_to(&partition_type));
        (
            data_file.partition_spec_id,
            promoted.unwrap_or_else(|| data_file.partition.clone()),
        )
    }
}

impl ManifestEntry {
    pub(crate) fn with_promoted_partition(
        entry: &Arc<ManifestEntry>,
        partition_type: Option<&StructType>,
    ) -> Arc<ManifestEntry> {
        let Some(partition) = partition_type
            .and_then(|partition_type| entry.data_file.partition.promoted_to(partition_type))
        else {
            return Arc::clone(entry);
        };
        let mut promoted = entry.as_ref().clone();
        promoted.data_file.partition = partition;
        Arc::new(promoted)
    }
}
