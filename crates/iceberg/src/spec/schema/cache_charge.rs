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

use super::Schema;

impl Schema {
    pub(crate) fn alias_entries(&self) -> impl ExactSizeIterator<Item = (&String, &i32)> {
        self.alias_to_id.iter()
    }

    pub(crate) fn name_index_entries(&self) -> impl ExactSizeIterator<Item = (&String, &i32)> {
        self.name_to_id.iter()
    }

    pub(crate) fn lowercase_name_index_entries(
        &self,
    ) -> impl ExactSizeIterator<Item = (&String, &i32)> {
        self.lowercase_name_to_id.iter()
    }

    pub(crate) fn hidden_index_capacities(&self) -> (usize, usize, usize, usize) {
        (
            self.alias_to_id.capacity(),
            self.name_to_id.capacity(),
            self.lowercase_name_to_id.capacity(),
            self.field_id_to_accessor.capacity(),
        )
    }

    pub(crate) fn accessor_count(&self) -> usize {
        self.field_id_to_accessor.len()
    }

    pub(crate) fn identifier_storage_capacity(&self) -> usize {
        self.identifier_field_ids.capacity()
    }
}
