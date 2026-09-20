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
mod tests {
    use std::sync::Arc;

    use crate::expr::visitors::residual_evaluator::ResidualEvaluator;
    use crate::expr::{Bind, BoundPredicate, Reference};
    use crate::spec::{
        Literal, NestedField, PartitionSpec, PrimitiveType, Schema, SchemaRef, Struct, StructType,
        Transform, Type, UnboundPartitionField,
    };

    fn nested_name_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "category",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    Arc::new(NestedField::optional(
                        2,
                        "st",
                        Type::Struct(StructType::new(vec![Arc::new(NestedField::optional(
                            3,
                            "category",
                            Type::Primitive(PrimitiveType::String),
                        ))])),
                    )),
                ])
                .build()
                .expect("schema builds"),
        )
    }

    #[test]
    fn a_nested_residual_keeps_its_full_column_name() {
        let schema = nested_name_schema();
        let spec = identity_spec(schema.clone());
        let filter = Reference::new("st.category")
            .is_null()
            .bind(schema.clone(), true)
            .expect("binds the nested field");
        let evaluator = ResidualEvaluator::of(spec, &schema, filter, true).expect("evaluator");

        let residual = evaluator
            .residual_for(&Struct::from_iter([Some(Literal::int(5))]))
            .expect("residual");
        assert_eq!(
            residual.to_string(),
            "st.category IS NULL",
            "the residual must carry the nested column's full path, not its leaf name"
        );
        let rebound = residual
            .bind(schema.clone(), true)
            .expect("the residual rebinds");
        assert_ne!(
            rebound,
            BoundPredicate::AlwaysFalse,
            "the rebound residual must still be able to match rows"
        );
        assert_eq!(
            Reference::new("category")
                .is_null()
                .bind(schema, true)
                .expect("the leaf name binds to the top-level column"),
            BoundPredicate::AlwaysFalse,
            "the leaf name alone binds to the required top-level column and loses every row"
        );
    }

    fn identity_spec(schema: SchemaRef) -> Arc<PartitionSpec> {
        Arc::new(
            PartitionSpec::builder(schema)
                .with_spec_id(1)
                .add_unbound_field(
                    UnboundPartitionField::builder()
                        .source_id(1)
                        .name("category".to_string())
                        .field_id(1000)
                        .transform(Transform::Identity)
                        .build(),
                )
                .expect("add identity field")
                .build()
                .expect("spec builds"),
        )
    }
}
