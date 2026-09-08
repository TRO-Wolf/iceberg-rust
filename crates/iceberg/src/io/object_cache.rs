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

use std::mem::size_of;
use std::sync::Arc;

use crate::expr::accessor::StructAccessor;
use crate::io::FileIO;
use crate::spec::{
    DataFile, Datum, FieldSummary, FormatVersion, Literal, Manifest, ManifestEntry, ManifestFile,
    ManifestList, NestedField, PrimitiveLiteral, Schema, SchemaId, SchemaRef, SnapshotRef,
    StructType, TableMetadataRef, Type, apply_manifest_list_context,
};
use crate::{Error, ErrorKind, Result};

const DEFAULT_CACHE_SIZE_BYTES: u64 = 32 * 1024 * 1024; // 32MB

fn clamp_cache_weight(bytes: u64) -> u32 {
    let clamped = bytes.clamp(1, u32::MAX as u64);
    clamped as u32
}

fn usize_charge(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn shallow_charge<T>() -> u64 {
    usize_charge(size_of::<T>())
}

fn sequence_charge<T>(capacity: usize) -> u64 {
    usize_charge(capacity).saturating_mul(shallow_charge::<T>())
}

fn hash_storage_charge<K, V>(capacity: usize) -> u64 {
    sequence_charge::<(K, V)>(capacity).saturating_add(usize_charge(capacity))
}

fn arc_allocation_charge<T>() -> u64 {
    shallow_charge::<T>().saturating_add(2u64.saturating_mul(shallow_charge::<usize>()))
}

fn primitive_literal_payload_charge(literal: &PrimitiveLiteral) -> u64 {
    match literal {
        PrimitiveLiteral::String(value) => usize_charge(value.capacity()),
        PrimitiveLiteral::Binary(value) => sequence_charge::<u8>(value.capacity()),
        _ => 0,
    }
}

fn literal_payload_charge(literal: &Literal) -> u64 {
    let mut charge = 0u64;
    let mut pending = vec![(literal, 1u64)];
    while let Some((value, copies)) = pending.pop() {
        match value {
            Literal::Primitive(value) => {
                charge = charge
                    .saturating_add(primitive_literal_payload_charge(value).saturating_mul(copies));
            }
            Literal::Struct(value) => {
                charge = charge.saturating_add(
                    sequence_charge::<Option<Literal>>(value.fields().len()).saturating_mul(copies),
                );
                pending.extend(value.fields().iter().flatten().map(|value| (value, copies)));
            }
            Literal::List(value) => {
                charge = charge.saturating_add(
                    sequence_charge::<Option<Literal>>(value.capacity()).saturating_mul(copies),
                );
                pending.extend(value.iter().flatten().map(|value| (value, copies)));
            }
            Literal::Map(value) => {
                let (index_capacity, pair_capacity) = value.storage_capacities();
                charge = charge
                    .saturating_add(
                        hash_storage_charge::<Literal, usize>(index_capacity)
                            .saturating_mul(copies),
                    )
                    .saturating_add(
                        sequence_charge::<(Literal, Option<Literal>)>(pair_capacity)
                            .saturating_mul(copies),
                    );
                for (key, map_value) in value.pairs() {
                    pending.push((key, copies.saturating_mul(2)));
                    pending.extend(map_value.iter().map(|value| (value, copies)));
                }
            }
        }
    }
    charge
}

fn schema_type_graph_charge(struct_type: &StructType) -> u64 {
    let fields = struct_type.fields();
    let mut charge = sequence_charge::<Arc<NestedField>>(fields.len())
        .saturating_add(hash_storage_charge::<i32, usize>(fields.len()))
        .saturating_add(hash_storage_charge::<String, usize>(fields.len()));
    let mut pending_fields: Vec<_> = fields.iter().collect();
    while let Some(field) = pending_fields.pop() {
        charge = charge
            .saturating_add(arc_allocation_charge::<NestedField>())
            .saturating_add(usize_charge(field.name.capacity()))
            .saturating_add(
                field
                    .doc
                    .as_ref()
                    .map_or(0, |value| usize_charge(value.capacity())),
            )
            .saturating_add(shallow_charge::<Type>())
            .saturating_add(
                field
                    .initial_default
                    .as_ref()
                    .map_or(0, literal_payload_charge),
            )
            .saturating_add(
                field
                    .write_default
                    .as_ref()
                    .map_or(0, literal_payload_charge),
            );
        match field.field_type.as_ref() {
            Type::Primitive(_) | Type::Variant => {}
            Type::Struct(value) => {
                let fields = value.fields();
                charge = charge
                    .saturating_add(sequence_charge::<Arc<NestedField>>(fields.len()))
                    .saturating_add(hash_storage_charge::<i32, usize>(fields.len()))
                    .saturating_add(hash_storage_charge::<String, usize>(fields.len()));
                pending_fields.extend(fields);
            }
            Type::List(value) => pending_fields.push(&value.element_field),
            Type::Map(value) => {
                pending_fields.push(&value.key_field);
                pending_fields.push(&value.value_field);
            }
        }
    }
    charge
}

fn schema_accessor_charge(schema: &Schema) -> u64 {
    let mut box_count = 0u64;
    let mut pending: Vec<_> = schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| (field.as_ref(), 0u64))
        .collect();
    while let Some((field, depth)) = pending.pop() {
        match field.field_type.as_ref() {
            Type::Primitive(_) => box_count = box_count.saturating_add(depth),
            Type::Struct(value) => {
                let child_depth = depth.saturating_add(1);
                pending.extend(
                    value
                        .fields()
                        .iter()
                        .map(|field| (field.as_ref(), child_depth)),
                );
            }
            Type::List(_) | Type::Map(_) | Type::Variant => {}
        }
    }
    usize_charge(schema.accessor_count())
        .saturating_mul(arc_allocation_charge::<StructAccessor>())
        .saturating_add(box_count.saturating_mul(shallow_charge::<StructAccessor>()))
}

fn schema_charge(schema: &Schema) -> u64 {
    let id_to_name = schema.field_id_to_name_map();
    let id_to_field = schema.field_id_to_fields();
    let identifier_capacity = schema.identifier_storage_capacity();
    let (alias_capacity, name_capacity, lowercase_name_capacity, accessor_capacity) =
        schema.hidden_index_capacities();
    let mut charge = arc_allocation_charge::<Schema>()
        .saturating_add(schema_type_graph_charge(schema.as_struct()))
        .saturating_add(hash_storage_charge::<i32, String>(id_to_name.capacity()))
        .saturating_add(hash_storage_charge::<String, i32>(name_capacity))
        .saturating_add(hash_storage_charge::<String, i32>(lowercase_name_capacity))
        .saturating_add(hash_storage_charge::<i32, Arc<NestedField>>(
            id_to_field.capacity(),
        ))
        .saturating_add(hash_storage_charge::<i32, Arc<StructAccessor>>(
            accessor_capacity,
        ))
        .saturating_add(schema_accessor_charge(schema))
        .saturating_add(hash_storage_charge::<i32, ()>(identifier_capacity))
        .saturating_add(hash_storage_charge::<String, i32>(alias_capacity).saturating_mul(2));
    for name in id_to_name.values() {
        charge = charge.saturating_add(usize_charge(name.capacity()));
    }
    for (name, _) in schema.name_index_entries() {
        charge = charge.saturating_add(usize_charge(name.capacity()));
    }
    for (name, _) in schema.lowercase_name_index_entries() {
        charge = charge.saturating_add(usize_charge(name.capacity()));
    }
    for (alias, _) in schema.alias_entries() {
        charge = charge.saturating_add(usize_charge(alias.capacity()));
    }
    charge
}

fn datum_payload_charge(datum: &Datum) -> u64 {
    primitive_literal_payload_charge(datum.literal())
}

fn data_file_payload_charge(data_file: &DataFile) -> u64 {
    let partition = data_file.partition();
    let mut charge = usize_charge(data_file.file_path.capacity())
        .saturating_add(sequence_charge::<Option<Literal>>(partition.fields().len()));
    for literal in partition.fields().iter().flatten() {
        charge = charge.saturating_add(literal_payload_charge(literal));
    }
    for map in [
        &data_file.column_sizes,
        &data_file.value_counts,
        &data_file.null_value_counts,
        &data_file.nan_value_counts,
    ] {
        charge = charge.saturating_add(hash_storage_charge::<i32, u64>(map.capacity()));
    }
    for map in [&data_file.lower_bounds, &data_file.upper_bounds] {
        charge = charge.saturating_add(hash_storage_charge::<i32, Datum>(map.capacity()));
        for datum in map.values() {
            charge = charge.saturating_add(datum_payload_charge(datum));
        }
    }
    charge
        .saturating_add(
            data_file
                .key_metadata
                .as_ref()
                .map_or(0, |value| sequence_charge::<u8>(value.capacity())),
        )
        .saturating_add(
            data_file
                .split_offsets
                .as_ref()
                .map_or(0, |value| sequence_charge::<i64>(value.capacity())),
        )
        .saturating_add(
            data_file
                .equality_ids
                .as_ref()
                .map_or(0, |value| sequence_charge::<i32>(value.capacity())),
        )
        .saturating_add(
            data_file
                .referenced_data_file
                .as_ref()
                .map_or(0, |value| usize_charge(value.capacity())),
        )
}

fn manifest_entry_charge(entry: &ManifestEntry) -> u64 {
    arc_allocation_charge::<ManifestEntry>()
        .saturating_add(data_file_payload_charge(entry.data_file()))
}

fn manifest_charge(manifest: &Manifest) -> u64 {
    let metadata = manifest.metadata();
    manifest.entries().iter().fold(
        arc_allocation_charge::<Manifest>()
            .saturating_add(sequence_charge::<Arc<ManifestEntry>>(
                manifest.entries().len(),
            ))
            .saturating_add(schema_charge(&metadata.schema))
            .saturating_add(sequence_charge::<crate::spec::PartitionField>(
                metadata.partition_spec.fields().len(),
            ))
            .saturating_add(
                metadata
                    .partition_spec
                    .fields()
                    .iter()
                    .fold(0u64, |charge, field| {
                        charge.saturating_add(usize_charge(field.name.capacity()))
                    }),
            ),
        |charge, entry| charge.saturating_add(manifest_entry_charge(entry)),
    )
}

fn field_summary_payload_charge(summary: &FieldSummary) -> u64 {
    summary
        .lower_bound
        .as_ref()
        .map_or(0, |value| usize_charge(value.len()))
        .saturating_add(
            summary
                .upper_bound
                .as_ref()
                .map_or(0, |value| usize_charge(value.len())),
        )
}

fn manifest_file_payload_charge(file: &ManifestFile) -> u64 {
    let partitions = file.partitions.as_ref().map_or(0, |summaries| {
        summaries.iter().fold(
            sequence_charge::<FieldSummary>(summaries.capacity()),
            |charge, summary| charge.saturating_add(field_summary_payload_charge(summary)),
        )
    });
    usize_charge(file.manifest_path.capacity())
        .saturating_add(partitions)
        .saturating_add(
            file.key_metadata
                .as_ref()
                .map_or(0, |value| sequence_charge::<u8>(value.capacity())),
        )
}

fn manifest_list_charge(list: &ManifestList) -> u64 {
    list.entries().iter().fold(
        arc_allocation_charge::<ManifestList>()
            .saturating_add(sequence_charge::<ManifestFile>(list.entries().len())),
        |charge, file| charge.saturating_add(manifest_file_payload_charge(file)),
    )
}

fn cache_key_charge(key: &CachedObjectKey) -> u64 {
    let path_capacity = match key {
        CachedObjectKey::ManifestList((path, _, _)) | CachedObjectKey::Manifest((path, _)) => {
            path.capacity()
        }
    };
    shallow_charge::<CachedObjectKey>().saturating_add(usize_charge(path_capacity))
}

fn cached_object_charge(key: &CachedObjectKey, item: &CachedItem) -> u32 {
    let item_charge = shallow_charge::<CachedItem>().saturating_add(match item {
        CachedItem::ManifestList(value) => manifest_list_charge(value),
        CachedItem::RawManifest(value) => manifest_charge(value),
    });
    clamp_cache_weight(cache_key_charge(key).saturating_add(item_charge))
}

#[derive(Clone, Debug)]
pub(crate) enum CachedItem {
    ManifestList(Arc<ManifestList>),
    RawManifest(Arc<Manifest>),
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub(crate) enum CachedObjectKey {
    ManifestList((String, FormatVersion, Option<SchemaId>)),
    /// Manifest path plus optional fallback schema id used when the embedded
    /// `"schema"` key fails strict parse (QD). Path-only keys are wrong once
    /// parse depends on caller-supplied fallback (C1-SEC-002).
    Manifest((String, Option<SchemaId>)),
}

/// Caches metadata objects deserialized from immutable files
#[derive(Clone, Debug)]
pub struct ObjectCache {
    cache: moka::future::Cache<CachedObjectKey, CachedItem>,
    file_io: FileIO,
    cache_disabled: bool,
}

impl ObjectCache {
    /// Creates a new [`ObjectCache`]
    /// with the default cache size
    pub(crate) fn new(file_io: FileIO) -> Self {
        Self::new_with_capacity(file_io, DEFAULT_CACHE_SIZE_BYTES)
    }

    /// Creates a new [`ObjectCache`] with a specific cache size, shareable across tables.
    pub fn new_with_capacity(file_io: FileIO, cache_size_bytes: u64) -> Self {
        if cache_size_bytes == 0 {
            Self::with_disabled_cache(file_io)
        } else {
            Self {
                cache: moka::future::Cache::builder()
                    .weigher(cached_object_charge)
                    .max_capacity(cache_size_bytes)
                    .build(),
                file_io,
                cache_disabled: false,
            }
        }
    }

    /// Creates a new [`ObjectCache`]
    /// with caching disabled
    pub(crate) fn with_disabled_cache(file_io: FileIO) -> Self {
        Self {
            cache: moka::future::Cache::new(0),
            file_io,
            cache_disabled: true,
        }
    }

    /// Retrieves an Arc [`Manifest`] from the cache
    /// or retrieves one from FileIO and parses it if not present.
    ///
    /// `schema_fallback` is the table/snapshot schema used when the manifest's embedded
    /// `"schema"` key fails strict parse (DuckDB malformation tolerance).
    pub(crate) async fn get_manifest(
        &self,
        manifest_file: &ManifestFile,
        schema_fallback: Option<SchemaRef>,
    ) -> Result<Arc<Manifest>> {
        if self.cache_disabled {
            return manifest_file
                .load_manifest_with_schema_fallback(&self.file_io, schema_fallback)
                .await
                .map(Arc::new);
        }

        let fallback_schema_id = schema_fallback.as_ref().map(|s| s.schema_id());
        let key =
            CachedObjectKey::Manifest((manifest_file.manifest_path.clone(), fallback_schema_id));

        let cache_entry = self
            .cache
            .entry_by_ref(&key)
            .or_try_insert_with(self.fetch_and_parse_manifest(manifest_file, schema_fallback))
            .await
            .map_err(|err| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to load manifest {}", manifest_file.manifest_path),
                )
                .with_source(err)
            })?
            .into_value();

        let raw_manifest = match cache_entry {
            CachedItem::RawManifest(arc_manifest) => arc_manifest,
            _ => {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!("cached object for key '{key:?}' is not a RawManifest"),
                ));
            }
        };

        let mut entries: Vec<ManifestEntry> = raw_manifest
            .entries()
            .iter()
            .map(|entry| entry.as_ref().clone())
            .collect();
        apply_manifest_list_context(&mut entries, manifest_file)?;

        Ok(Arc::new(Manifest::new(
            raw_manifest.metadata().clone(),
            entries,
        )))
    }

    /// Retrieves an Arc [`ManifestList`] from the cache
    /// or retrieves one from FileIO and parses it if not present
    pub(crate) async fn get_manifest_list(
        &self,
        snapshot: &SnapshotRef,
        table_metadata: &TableMetadataRef,
    ) -> Result<Arc<ManifestList>> {
        if self.cache_disabled {
            return snapshot
                .load_manifest_list(&self.file_io, table_metadata)
                .await
                .map(Arc::new);
        }

        // `Snapshot::schema_id` is `Option`: V1/legacy snapshots may omit it. The manifest-list
        // path already uniquely identifies the cache entry, so key on the `Option` directly
        // rather than unwrapping (which panicked on a schema-id-less snapshot).
        let key = CachedObjectKey::ManifestList((
            snapshot.manifest_list().to_string(),
            table_metadata.format_version,
            snapshot.schema_id(),
        ));
        let cache_entry = self
            .cache
            .entry_by_ref(&key)
            .or_try_insert_with(self.fetch_and_parse_manifest_list(snapshot, table_metadata))
            .await
            .map_err(|err| {
                Arc::try_unwrap(err).unwrap_or_else(|err| {
                    Error::new(
                        ErrorKind::Unexpected,
                        "Failed to load manifest list in cache",
                    )
                    .with_source(err)
                })
            })?
            .into_value();

        match cache_entry {
            CachedItem::ManifestList(arc_manifest_list) => Ok(arc_manifest_list),
            _ => Err(Error::new(
                ErrorKind::Unexpected,
                format!("cached object for path '{key:?}' is not a manifest list"),
            )),
        }
    }

    async fn fetch_and_parse_manifest(
        &self,
        manifest_file: &ManifestFile,
        schema_fallback: Option<SchemaRef>,
    ) -> Result<CachedItem> {
        let (metadata, entries) = manifest_file
            .load_manifest_parts_with_schema_fallback(&self.file_io, schema_fallback)
            .await?;

        Ok(CachedItem::RawManifest(Arc::new(Manifest::new(
            metadata, entries,
        ))))
    }

    async fn fetch_and_parse_manifest_list(
        &self,
        snapshot: &SnapshotRef,
        table_metadata: &TableMetadataRef,
    ) -> Result<CachedItem> {
        let manifest_list = snapshot
            .load_manifest_list(&self.file_io, table_metadata)
            .await?;

        Ok(CachedItem::ManifestList(Arc::new(manifest_list)))
    }
}

#[cfg(test)]
#[path = "object_cache_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "object_cache_charge_tests.rs"]
mod charge_tests;
