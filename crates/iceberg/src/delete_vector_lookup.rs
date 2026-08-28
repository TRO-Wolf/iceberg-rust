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

//! Previous-deletion-vector discovery from a table's current snapshot.

use std::collections::HashMap;

use crate::spec::{
    DataFile, ManifestContentType, is_deletion_vector, referenced_data_file_location,
};
use crate::table::Table;
use crate::{Error, ErrorKind, Result};

/// Live Puffin DVs of the current snapshot, keyed by referenced data-file path.
///
/// Merge with `DVFileWriter::with_previous_deletes`. Close a shared Puffin with
/// `close_touched_dv_containers`: path-keyed `remove_deletes` drops sibling blobs.
///
/// No snapshot yields an empty map. Missing referenced path or a duplicate is `DataInvalid`.
pub async fn live_deletion_vectors_by_data_file(
    table: &Table,
) -> Result<HashMap<String, DataFile>> {
    let mut out = HashMap::new();
    let metadata = table.metadata();
    let Some(snapshot) = metadata.current_snapshot() else {
        return Ok(out);
    };
    let manifest_list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await?;
    for manifest_file in manifest_list.entries() {
        if manifest_file.content != ManifestContentType::Deletes {
            continue;
        }
        let manifest = manifest_file.load_manifest(table.file_io()).await?;
        for entry in manifest.entries() {
            if !entry.is_alive() {
                continue;
            }
            let data_file = entry.data_file();
            if !is_deletion_vector(data_file) {
                continue;
            }
            record_live_dv(&mut out, data_file)?;
        }
    }
    Ok(out)
}

fn record_live_dv(out: &mut HashMap<String, DataFile>, data_file: &DataFile) -> Result<()> {
    let path = data_file.file_path();
    let referenced = referenced_data_file_location(data_file).ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("live deletion vector '{path}' has no referenced data file"),
        )
    })?;
    if let Some(existing) = out.get(&referenced) {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "snapshot has two live deletion vectors for '{referenced}': '{}' and '{path}'",
                existing.file_path()
            ),
        ));
    }
    out.insert(referenced, data_file.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use tempfile::TempDir;

    use super::live_deletion_vectors_by_data_file;
    use crate::io::FileIO;
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, NestedField,
        PrimitiveType, Schema, Struct, Type, is_deletion_vector, referenced_data_file_location,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::base_writer::deletion_vector_writer::DVFileWriter;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{Catalog, TableCreation, TableIdent};

    fn id_name_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .expect("schema")
    }

    #[tokio::test]
    async fn puffin_position_delete_is_a_deletion_vector() {
        let dir = TempDir::new().expect("temp dir");
        let file_io = FileIO::new_with_fs();
        let path = dir.path().join("dv.puffin");
        let output = file_io
            .new_output(path.to_str().expect("utf-8"))
            .expect("output");
        let mut writer = DVFileWriter::new(output).unpartitioned();
        writer.delete("s3://b/d.parquet", 1, None).expect("delete");
        let file = writer.close().await.expect("close").remove(0);
        assert!(
            is_deletion_vector(&file),
            "a Puffin position-delete file is a DV"
        );
        assert_eq!(
            referenced_data_file_location(&file).as_deref(),
            Some("s3://b/d.parquet")
        );
    }

    #[tokio::test]
    async fn live_deletion_vectors_empty_on_a_new_table() {
        let catalog = new_memory_catalog().await;
        let ident = TableIdent::from_strs(["ns-empty", "t"]).expect("ident");
        catalog
            .create_namespace(ident.namespace(), HashMap::new())
            .await
            .expect("namespace");
        let table = catalog
            .create_table(
                ident.namespace(),
                TableCreation::builder()
                    .schema(id_name_schema())
                    .name(ident.name().to_string())
                    .format_version(FormatVersion::V3)
                    .build(),
            )
            .await
            .expect("create table");
        let live = live_deletion_vectors_by_data_file(&table)
            .await
            .expect("discover");
        assert!(live.is_empty(), "a table with no deletes has no live DVs");
    }

    #[tokio::test]
    async fn live_deletion_vectors_finds_a_committed_dv() {
        let catalog = new_memory_catalog().await;
        let ident = TableIdent::from_strs(["ns-dv", "t"]).expect("ident");
        catalog
            .create_namespace(ident.namespace(), HashMap::new())
            .await
            .expect("namespace");
        let table = catalog
            .create_table(
                ident.namespace(),
                TableCreation::builder()
                    .schema(id_name_schema())
                    .name(ident.name().to_string())
                    .format_version(FormatVersion::V3)
                    .build(),
            )
            .await
            .expect("create table");

        let schema = table.metadata().current_schema().clone();
        let location_gen =
            DefaultLocationGenerator::new(table.metadata().clone()).expect("location");
        let file_name_gen = DefaultFileNameGenerator::new(
            "data".to_string(),
            None,
            crate::spec::DataFileFormat::Parquet,
        );
        let rolling = RollingFileWriterBuilder::new_with_default_file_size(
            ParquetWriterBuilder::new(
                parquet::file::properties::WriterProperties::builder().build(),
                schema.clone(),
            ),
            table.file_io().clone(),
            location_gen,
            file_name_gen,
        );
        let mut data_writer = DataFileWriterBuilder::new(rolling)
            .unpartitioned()
            .build(None)
            .await
            .expect("build data writer");
        let arrow_schema =
            Arc::new(crate::arrow::schema_to_arrow_schema(&schema).expect("arrow schema"));
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int64Array::from(vec![1i64])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
        ])
        .expect("batch");
        data_writer.write(batch).await.expect("write");
        let data_file = data_writer
            .close()
            .await
            .expect("close data")
            .into_iter()
            .next()
            .expect("one data file");
        let data_path = data_file.file_path().to_string();

        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .add_data_files(vec![data_file])
            .apply(tx)
            .expect("apply append")
            .commit(&catalog)
            .await
            .expect("commit data");

        let puffin = format!("{}/dv.puffin", table.metadata().location());
        let mut dv_writer =
            DVFileWriter::new(table.file_io().new_output(&puffin).expect("dv output"))
                .unpartitioned();
        dv_writer.delete(&data_path, 0, None).expect("dv delete");
        let dv_file = dv_writer
            .close()
            .await
            .expect("close dv")
            .into_iter()
            .next()
            .expect("one DV");

        let tx = Transaction::new(&table);
        let table = tx
            .row_delta()
            .add_deletes(vec![dv_file])
            .apply(tx)
            .expect("apply row delta")
            .commit(&catalog)
            .await
            .expect("commit dv");

        let live = live_deletion_vectors_by_data_file(&table)
            .await
            .expect("discover");
        assert_eq!(live.len(), 1, "the committed DV must be discoverable");
        assert!(
            live.contains_key(&data_path),
            "discovery keys by referenced data-file path, got {:?}",
            live.keys().collect::<Vec<_>>()
        );
        assert!(is_deletion_vector(&live[&data_path]));
    }

    fn puffin_dv(path: &str, referenced: Option<&str>) -> crate::spec::DataFile {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40));
        if let Some(referenced) = referenced {
            builder.referenced_data_file(Some(referenced.to_string()));
        }
        builder.build().expect("synthetic DV")
    }

    #[test]
    fn record_live_dv_errors_without_referenced_path() {
        let mut out = HashMap::new();
        let mut file = puffin_dv("puffin-a", Some("data.parquet"));
        file.referenced_data_file = None;
        let err = super::record_live_dv(&mut out, &file).expect_err("missing referenced path");
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        assert!(out.is_empty());
    }

    #[test]
    fn record_live_dv_errors_on_a_second_dv_for_the_same_data_file() {
        let mut out = HashMap::new();
        super::record_live_dv(&mut out, &puffin_dv("puffin-a", Some("data.parquet")))
            .expect("first DV");
        let err = super::record_live_dv(&mut out, &puffin_dv("puffin-b", Some("data.parquet")))
            .expect_err("second DV");
        assert_eq!(err.kind(), crate::ErrorKind::DataInvalid);
        assert!(err.to_string().contains("two live deletion vectors"));
        assert_eq!(out.len(), 1);
    }
}
