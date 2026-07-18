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

use std::fmt::Debug;
use std::io;

use anyhow::anyhow;
use hive_metastore::{
    ThriftHiveMetastoreAlterDatabaseException, ThriftHiveMetastoreCreateDatabaseException,
    ThriftHiveMetastoreCreateTableException, ThriftHiveMetastoreDropDatabaseException,
    ThriftHiveMetastoreDropTableException, ThriftHiveMetastoreGetDatabaseException,
    ThriftHiveMetastoreGetTableException,
};
use iceberg::{Error, ErrorKind, NamespaceIdent, Result, TableIdent};
use volo_thrift::MaybeException;

/// Format a thrift error into iceberg error.
///
/// Please only throw this error when you are sure that the error is caused by thrift.
pub fn from_thrift_error(error: impl std::error::Error) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Operation failed for hitting thrift error".to_string(),
    )
    .with_source(anyhow!("thrift error: {error:?}"))
}

/// Wrap a thrift *user* exception with no typed-kind equivalent (e.g. `MetaException`,
/// `InvalidObjectException`, `InvalidOperationException`) into an [`ErrorKind::Unexpected`]
/// iceberg error, preserving the exception as the source. This is the fallback for the
/// per-call-site mappers below and the body of [`from_thrift_exception`].
fn thrift_exception_error<E: Debug>(exception: E) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Operation failed for hitting thrift error".to_string(),
    )
    .with_source(anyhow!("thrift error: {exception:?}"))
}

/// Format a thrift exception into iceberg error.
///
/// The generic form cannot know the call-site semantics of the exception, so every exception
/// collapses to [`ErrorKind::Unexpected`]. Call sites that CAN distinguish not-found /
/// already-exists conditions from the thrift IDL use the typed mappers below instead.
pub fn from_thrift_exception<T, E: Debug>(value: MaybeException<T, E>) -> Result<T> {
    match value {
        MaybeException::Ok(v) => Ok(v),
        MaybeException::Exception(err) => Err(thrift_exception_error(err)),
    }
}

/// Format an io error into iceberg error.
pub fn from_io_error(error: io::Error) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Operation failed for hitting io error".to_string(),
    )
    .with_source(error)
}

/// Map a `get_database` outcome to a typed error.
///
/// The thrift `get_database` IDL declares `NoSuchObjectException o1, MetaException o2`, so `O1`
/// means the database (namespace) does not exist → [`ErrorKind::NamespaceNotFound`] (Java
/// `HiveCatalog` surfaces this as `NoSuchNamespaceException`). A `MetaException` (`O2`) is a
/// server-side failure with no not-found semantics → [`ErrorKind::Unexpected`].
pub fn from_get_database_exception<T>(
    namespace: &NamespaceIdent,
    value: MaybeException<T, ThriftHiveMetastoreGetDatabaseException>,
) -> Result<T> {
    match value {
        MaybeException::Ok(v) => Ok(v),
        MaybeException::Exception(exc @ ThriftHiveMetastoreGetDatabaseException::O1(_)) => {
            Err(Error::new(
                ErrorKind::NamespaceNotFound,
                format!("No such namespace: {namespace:?}"),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc) => Err(thrift_exception_error(exc)),
    }
}

/// Map a `get_table` outcome to a typed error.
///
/// The thrift `get_table` IDL declares `MetaException o1, NoSuchObjectException o2`, so `O2`
/// means the table does not exist → [`ErrorKind::TableNotFound`] (Java `HiveCatalog` surfaces
/// this as `NoSuchTableException`). A `MetaException` (`O1`) is a server-side failure with no
/// not-found semantics → [`ErrorKind::Unexpected`].
pub fn from_get_table_exception<T>(
    table: &TableIdent,
    value: MaybeException<T, ThriftHiveMetastoreGetTableException>,
) -> Result<T> {
    match value {
        MaybeException::Ok(v) => Ok(v),
        MaybeException::Exception(exc @ ThriftHiveMetastoreGetTableException::O2(_)) => {
            Err(Error::new(
                ErrorKind::TableNotFound,
                format!("No such table: {table:?}"),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc) => Err(thrift_exception_error(exc)),
    }
}

/// Map a `create_database` outcome to a typed error.
///
/// The thrift `create_database` IDL declares
/// `AlreadyExistsException o1, InvalidObjectException o2, MetaException o3`, so `O1` means the
/// namespace already exists → [`ErrorKind::NamespaceAlreadyExists`] (Java `HiveCatalog` surfaces
/// this as `AlreadyExistsException`). The other exceptions have no already-exists semantics →
/// [`ErrorKind::Unexpected`].
pub fn from_create_database_exception<T>(
    namespace: &NamespaceIdent,
    value: MaybeException<T, ThriftHiveMetastoreCreateDatabaseException>,
) -> Result<T> {
    match value {
        MaybeException::Ok(v) => Ok(v),
        MaybeException::Exception(exc @ ThriftHiveMetastoreCreateDatabaseException::O1(_)) => {
            Err(Error::new(
                ErrorKind::NamespaceAlreadyExists,
                format!("Namespace {namespace:?} already exists"),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc) => Err(thrift_exception_error(exc)),
    }
}

/// Map a `create_table` outcome to a typed error.
///
/// The thrift `create_table` IDL declares `AlreadyExistsException o1, InvalidObjectException o2,
/// MetaException o3, NoSuchObjectException o4`. `O1` means the table already exists →
/// [`ErrorKind::TableAlreadyExists`]; `O4` means the parent database (namespace) does not exist →
/// [`ErrorKind::NamespaceNotFound`]. The remaining exceptions have no such semantics →
/// [`ErrorKind::Unexpected`].
pub fn from_create_table_exception<T>(
    table: &TableIdent,
    value: MaybeException<T, ThriftHiveMetastoreCreateTableException>,
) -> Result<T> {
    match value {
        MaybeException::Ok(v) => Ok(v),
        MaybeException::Exception(exc @ ThriftHiveMetastoreCreateTableException::O1(_)) => {
            Err(Error::new(
                ErrorKind::TableAlreadyExists,
                format!("Table {table:?} already exists."),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc @ ThriftHiveMetastoreCreateTableException::O4(_)) => {
            Err(Error::new(
                ErrorKind::NamespaceNotFound,
                format!("No such namespace: {:?}", table.namespace()),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc) => Err(thrift_exception_error(exc)),
    }
}

/// Map a `drop_database` outcome to a typed error.
///
/// The thrift `drop_database` IDL declares `NoSuchObjectException o1, InvalidOperationException o2,
/// MetaException o3`, so `O1` means the namespace does not exist → [`ErrorKind::NamespaceNotFound`].
///
/// `O2` (`InvalidOperationException`) means the namespace is not empty →
/// [`ErrorKind::NamespaceNotEmpty`]. This flip is bytecode-grounded: Java
/// `org.apache.iceberg.hive.HiveCatalog.dropNamespace` (iceberg-hive-metastore 1.10.0, decoded with
/// `javap -p -c`) calls `IMetaStoreClient.dropDatabase(name, false, false, false)` — the trailing
/// `cascade=false` — and its exception table maps the caught Hive `InvalidOperationException`
/// (target 40) directly to `new NamespaceNotEmptyException("Namespace %s is not empty. One or more
/// tables exist.", ...)` (bytecode offsets 41–60). With `cascade=false`, non-empty is the sole
/// meaning of `InvalidOperationException` on this call, so the arm maps unambiguously.
///
/// A `MetaException` (`O3`) is a server-side failure with no typed semantics → [`ErrorKind::Unexpected`].
pub fn from_drop_database_exception<T>(
    namespace: &NamespaceIdent,
    value: MaybeException<T, ThriftHiveMetastoreDropDatabaseException>,
) -> Result<T> {
    match value {
        MaybeException::Ok(v) => Ok(v),
        MaybeException::Exception(exc @ ThriftHiveMetastoreDropDatabaseException::O1(_)) => {
            Err(Error::new(
                ErrorKind::NamespaceNotFound,
                format!("No such namespace: {namespace:?}"),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc @ ThriftHiveMetastoreDropDatabaseException::O2(_)) => {
            Err(Error::new(
                ErrorKind::NamespaceNotEmpty,
                format!("Namespace {namespace:?} is not empty. One or more tables exist."),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc) => Err(thrift_exception_error(exc)),
    }
}

/// Map a `drop_table` outcome to a typed error.
///
/// The thrift `drop_table` IDL declares `NoSuchObjectException o1, MetaException o3`, so `O1` means
/// the table does not exist → [`ErrorKind::TableNotFound`]. A `MetaException` (`O3`) has no
/// not-found semantics → [`ErrorKind::Unexpected`].
pub fn from_drop_table_exception<T>(
    table: &TableIdent,
    value: MaybeException<T, ThriftHiveMetastoreDropTableException>,
) -> Result<T> {
    match value {
        MaybeException::Ok(v) => Ok(v),
        MaybeException::Exception(exc @ ThriftHiveMetastoreDropTableException::O1(_)) => {
            Err(Error::new(
                ErrorKind::TableNotFound,
                format!("No such table: {table:?}"),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc) => Err(thrift_exception_error(exc)),
    }
}

/// Map an `alter_database` outcome to a typed error.
///
/// The thrift `alter_database` IDL declares `MetaException o1, NoSuchObjectException o2`, so `O2`
/// means the namespace does not exist → [`ErrorKind::NamespaceNotFound`]. A `MetaException` (`O1`)
/// has no not-found semantics → [`ErrorKind::Unexpected`].
pub fn from_alter_database_exception<T>(
    namespace: &NamespaceIdent,
    value: MaybeException<T, ThriftHiveMetastoreAlterDatabaseException>,
) -> Result<T> {
    match value {
        MaybeException::Ok(v) => Ok(v),
        MaybeException::Exception(exc @ ThriftHiveMetastoreAlterDatabaseException::O2(_)) => {
            Err(Error::new(
                ErrorKind::NamespaceNotFound,
                format!("No such namespace: {namespace:?}"),
            )
            .with_source(anyhow!("thrift error: {exc:?}")))
        }
        MaybeException::Exception(exc) => Err(thrift_exception_error(exc)),
    }
}

#[cfg(test)]
mod tests {
    use hive_metastore::{
        AlreadyExistsException, InvalidObjectException, InvalidOperationException, MetaException,
        NoSuchObjectException,
    };

    use super::*;

    fn namespace() -> NamespaceIdent {
        NamespaceIdent::new("db".into())
    }

    fn table() -> TableIdent {
        TableIdent::new(namespace(), "tbl".into())
    }

    /// `get_database` `O1` (`NoSuchObjectException`) must classify as `NamespaceNotFound`, and the
    /// thrift exception must survive as the error source (chain intact for debugging).
    #[test]
    fn get_database_no_such_object_maps_to_namespace_not_found() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreGetDatabaseException::O1(NoSuchObjectException::default()),
        );
        let err = from_get_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NamespaceNotFound);
        assert!(
            std::error::Error::source(&err)
                .expect("the thrift exception must survive as the source")
                .to_string()
                .contains("NoSuchObjectException"),
            "source chain must carry the underlying thrift exception"
        );
    }

    /// A `get_database` `MetaException` (`O2`) has no not-found semantics and must stay `Unexpected`
    /// — proving the mapper does not over-broaden every exception to a typed kind.
    #[test]
    fn get_database_meta_exception_stays_unexpected() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreGetDatabaseException::O2(MetaException::default()),
        );
        let err = from_get_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);
    }

    /// The `Ok` arm must pass the value through untouched.
    #[test]
    fn get_database_ok_passes_through() {
        let value: MaybeException<u8, ThriftHiveMetastoreGetDatabaseException> =
            MaybeException::Ok(7);
        assert_eq!(
            from_get_database_exception(&namespace(), value).expect("Ok must pass through"),
            7
        );
    }

    /// `get_table` `O2` (`NoSuchObjectException`) must classify as `TableNotFound`.
    #[test]
    fn get_table_no_such_object_maps_to_table_not_found() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreGetTableException::O2(NoSuchObjectException::default()),
        );
        let err = from_get_table_exception(&table(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TableNotFound);
    }

    /// A `get_table` `MetaException` (`O1`) must stay `Unexpected`.
    #[test]
    fn get_table_meta_exception_stays_unexpected() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreGetTableException::O1(MetaException::default()),
        );
        let err = from_get_table_exception(&table(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);
    }

    /// `create_database` `O1` (`AlreadyExistsException`) must classify as `NamespaceAlreadyExists`.
    #[test]
    fn create_database_already_exists_maps_to_namespace_already_exists() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreCreateDatabaseException::O1(AlreadyExistsException::default()),
        );
        let err = from_create_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NamespaceAlreadyExists);
    }

    /// A `create_database` `InvalidObjectException` (`O2`) has no already-exists semantics and must
    /// stay `Unexpected`.
    #[test]
    fn create_database_invalid_object_stays_unexpected() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreCreateDatabaseException::O2(InvalidObjectException::default()),
        );
        let err = from_create_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);
    }

    /// `create_table` `O1` (`AlreadyExistsException`) → `TableAlreadyExists`;
    /// `O4` (`NoSuchObjectException`, parent database missing) → `NamespaceNotFound`.
    #[test]
    fn create_table_distinguishes_already_exists_from_missing_parent() {
        let already: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreCreateTableException::O1(AlreadyExistsException::default()),
        );
        assert_eq!(
            from_create_table_exception(&table(), already)
                .unwrap_err()
                .kind(),
            ErrorKind::TableAlreadyExists
        );

        let missing_parent: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreCreateTableException::O4(NoSuchObjectException::default()),
        );
        assert_eq!(
            from_create_table_exception(&table(), missing_parent)
                .unwrap_err()
                .kind(),
            ErrorKind::NamespaceNotFound
        );
    }

    /// A `create_table` `MetaException` (`O3`) must stay `Unexpected`.
    #[test]
    fn create_table_meta_exception_stays_unexpected() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreCreateTableException::O3(MetaException::default()),
        );
        let err = from_create_table_exception(&table(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);
    }

    /// `drop_database` `O1` (`NoSuchObjectException`) must classify as `NamespaceNotFound`.
    #[test]
    fn drop_database_no_such_object_maps_to_namespace_not_found() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreDropDatabaseException::O1(NoSuchObjectException::default()),
        );
        let err = from_drop_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NamespaceNotFound);
    }

    /// A `drop_database` `InvalidOperationException` (`O2`) means the namespace is not empty and
    /// must classify as `NamespaceNotEmpty` — bytecode-grounded against Java
    /// `HiveCatalog.dropNamespace`, which maps this exception (from the `cascade=false`
    /// `dropDatabase`) to `NamespaceNotEmptyException`. The thrift exception must survive as the
    /// source so the chain stays intact for debugging.
    #[test]
    fn drop_database_invalid_operation_maps_to_namespace_not_empty() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreDropDatabaseException::O2(InvalidOperationException::default()),
        );
        let err = from_drop_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NamespaceNotEmpty);
        assert!(
            std::error::Error::source(&err)
                .expect("the thrift exception must survive as the source")
                .to_string()
                .contains("InvalidOperationException"),
            "source chain must carry the underlying thrift exception"
        );
    }

    /// A `drop_database` `MetaException` (`O3`) has no typed semantics and must stay `Unexpected`
    /// — proving the mapper does not over-broaden every exception to a typed kind.
    #[test]
    fn drop_database_meta_exception_stays_unexpected() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreDropDatabaseException::O3(MetaException::default()),
        );
        let err = from_drop_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);
    }

    /// `drop_table` `O1` (`NoSuchObjectException`) must classify as `TableNotFound`.
    #[test]
    fn drop_table_no_such_object_maps_to_table_not_found() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreDropTableException::O1(NoSuchObjectException::default()),
        );
        let err = from_drop_table_exception(&table(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TableNotFound);
    }

    /// A `drop_table` `MetaException` (`O3`) must stay `Unexpected`.
    #[test]
    fn drop_table_meta_exception_stays_unexpected() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreDropTableException::O3(MetaException::default()),
        );
        let err = from_drop_table_exception(&table(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);
    }

    /// `alter_database` `O2` (`NoSuchObjectException`) must classify as `NamespaceNotFound` — note
    /// the variant ordinal differs from `get_database`/`drop_database` (thrift IDL orders the
    /// exceptions per-method), which is exactly why the mapping is per-call-site.
    #[test]
    fn alter_database_no_such_object_maps_to_namespace_not_found() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreAlterDatabaseException::O2(NoSuchObjectException::default()),
        );
        let err = from_alter_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NamespaceNotFound);
    }

    /// An `alter_database` `MetaException` (`O1`) must stay `Unexpected`.
    #[test]
    fn alter_database_meta_exception_stays_unexpected() {
        let value: MaybeException<(), _> = MaybeException::Exception(
            ThriftHiveMetastoreAlterDatabaseException::O1(MetaException::default()),
        );
        let err = from_alter_database_exception(&namespace(), value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);
    }

    /// The generic [`from_thrift_exception`] still collapses to `Unexpected` (unchanged behavior)
    /// and preserves the source.
    #[test]
    fn generic_thrift_exception_stays_unexpected_with_source() {
        let value: MaybeException<(), ThriftHiveMetastoreGetTableException> =
            MaybeException::Exception(ThriftHiveMetastoreGetTableException::O2(
                NoSuchObjectException::default(),
            ));
        let err = from_thrift_exception(value).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert!(std::error::Error::source(&err).is_some());
    }
}
