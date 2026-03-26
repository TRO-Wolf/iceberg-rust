# iceberg_spark Implementation Plan

PySpark-compatible SQL interface backed by DataFusion + Iceberg Rust.
Zero JVM. Pure Python wrapper over existing Rust + Python infrastructure.

---

## Status

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 1 — Read Path | ✅ Complete | 94 passing |
| Phase 2 — Write Path + DDL | ✅ Complete | 136 passing (Phase 3: 170) |
| Phase 3 — Advanced Features | ✅ Complete | 170 passing |
| Phase 4 — DML (stretch) | ✅ Complete | 223 passing (DELETE FROM + UPDATE + MERGE INTO) |
| Partitioned Tables | ✅ Complete | 627 total (24 new partition tests) |
| Parity Sprints 1-4 + Quality | ✅ Complete | 802 total |

---

## Phase 1 — Read Path (Complete)

Core infrastructure built and tested.

### Files
- `iceberg_spark/session.py` — IcebergSession + IcebergSessionBuilder
- `iceberg_spark/dataframe.py` — DataFrame wrapping `datafusion.DataFrame`
- `iceberg_spark/column.py` — Column expression wrapper with operator overloads
- `iceberg_spark/row.py` — PySpark-compatible Row type
- `iceberg_spark/types.py` — Spark type system with Arrow conversion
- `iceberg_spark/functions.py` — 50+ PySpark-compatible functions
- `iceberg_spark/grouped_data.py` — GroupedData for .groupBy().agg()
- `iceberg_spark/writer.py` — DataFrameWriter (append mode only)
- `iceberg_spark/reader.py` — DataFrameReader (.table, .parquet, .csv, .json)
- `iceberg_spark/window.py` — WindowSpec / Window classes (API surface, not wired)
- `iceberg_spark/sql_preprocessor.py` — Spark SQL dialect → DataFusion translation
- `iceberg_spark/catalog_ops.py` — DDL dispatcher (CREATE/DROP TABLE/NAMESPACE)
- `iceberg_spark/_internal/display.py` — Spark-style table + schema formatting
- `iceberg_spark/_internal/type_mapping.py` — SQL type name parsing
- `iceberg_spark/_internal/catalog_factory.py` — PyIceberg catalog creation
- `iceberg_spark/_internal/table_registration.py` — FFI table registration

### SQL Transformations (Implemented)
- `USING iceberg` → stripped
- `TBLPROPERTIES(...)` → `OPTIONS(...)`
- `NVL(a, b)` → `COALESCE(a, b)`
- `RLIKE` → `~` (regex operator)
- `date_add(col, n)` → `col + interval 'n days'`
- `datediff(a, b)` → `date_diff('day', b, a)`
- `TIMESTAMP AS OF` / `VERSION AS OF` → intercepted as TIME_TRAVEL
- `SHOW TABLES [IN ns]` → intercepted as SHOW_TABLES
- `SHOW DATABASES/NAMESPACES` → intercepted as SHOW_DATABASES
- `DESCRIBE [TABLE] t` / `DESC t` → intercepted as DESCRIBE
- `CREATE TABLE` → intercepted as CREATE_TABLE
- `DROP TABLE` → intercepted as DROP_TABLE
- `CREATE DATABASE/SCHEMA` → intercepted as CREATE_NAMESPACE
- `DROP DATABASE` → intercepted as DROP_NAMESPACE
- `ALTER TABLE` → intercepted as ALTER_TABLE (ADD/DROP/RENAME COLUMN + SET/UNSET TBLPROPERTIES)

---

## Phase 2 — Write Path + DDL

### 2.1 SQL INSERT INTO Interception  ✅
**File:** `iceberg_spark/sql_preprocessor.py`, `iceberg_spark/session.py`

Add `INSERT_INTO` to `CommandType` enum. Intercept `INSERT INTO table VALUES (...)` and
`INSERT INTO table SELECT ...` in the preprocessor.

In `session.sql()`, route INSERT_INTO to `_insert_into_table()` using DataFusion to
execute the SELECT portion and PyIceberg `table.append()` to write.

```python
# Target API
session.sql("INSERT INTO db.t1 VALUES (1, 'hello'), (2, 'world')")
session.sql("INSERT INTO db.t1 SELECT * FROM db.t2 WHERE id > 5")
```

**Implementation steps:**
1. Add `INSERT_INTO` to `CommandType`
2. Add regex to detect `INSERT INTO <table> VALUES ...` and `INSERT INTO <table> SELECT ...`
3. In `session.sql()`: for `INSERT_INTO`, extract table name, ensure source table registered,
   execute with DataFusion (for SELECT) or parse VALUES directly, append via PyIceberg

### 2.2 INSERT OVERWRITE  ✅
**File:** `iceberg_spark/writer.py`, `iceberg_spark/session.py`

Overwrite = truncate then append. PyIceberg supports `table.overwrite()` (copy-on-write).

```python
df.write.mode("overwrite").saveAsTable("db.table")
```

PyIceberg API: `table.overwrite(df_arrow, overwrite_filter=AlwaysTrue())`

### 2.3 Write mode: error / ignore  ✅
**File:** `iceberg_spark/writer.py`

- `error` mode: check if table exists first, raise if it does
- `ignore` mode: check if table exists, skip write if it does

### 2.4 ALTER TABLE Support  ✅
**Files:** `iceberg_spark/sql_preprocessor.py`, `iceberg_spark/catalog_ops.py`

Extend `CommandType.ALTER_TABLE` handling from "raises NotImplementedError" to actual
dispatch via PyIceberg schema evolution API.

```python
session.sql("ALTER TABLE db.t1 ADD COLUMN new_col STRING")
session.sql("ALTER TABLE db.t1 DROP COLUMN old_col")
session.sql("ALTER TABLE db.t1 RENAME COLUMN old TO new")
```

PyIceberg API:
```python
with table.update_schema() as update:
    update.add_column("new_col", StringType())    # ADD
    update.delete_column("old_col")               # DROP
    update.rename_column("old", "new")            # RENAME
```

**SQL parsing:** regex to identify ADD COLUMN / DROP COLUMN / RENAME COLUMN patterns

### 2.5 TRUNCATE TABLE  ✅
**Files:** `iceberg_spark/sql_preprocessor.py`, `iceberg_spark/catalog_ops.py`

```python
session.sql("TRUNCATE TABLE db.t1")
```

PyIceberg: `table.overwrite(empty_df, overwrite_filter=AlwaysTrue())`
Or: use `table.delete()` if available.

### 2.6 DataFrameWriter.parquet / .csv / .json saves  ✅
**File:** `iceberg_spark/writer.py`

```python
df.write.format("parquet").save("/path/to/dir")
df.write.parquet("/path/to/dir")
```

Uses DataFusion `ctx.write_parquet()` / `ctx.write_csv()`.

---

## Phase 3 — Advanced Features

### 3.1 Window Functions  ✅
**Files:** `iceberg_spark/window.py`, `iceberg_spark/column.py`, `iceberg_spark/functions.py`

Wire up `WindowSpec` to produce DataFusion window expressions. Add `.over(window_spec)`
to `Column`. Add window functions to `functions.py`.

**Column.over():**
```python
def over(self, window_spec: WindowSpec) -> Column:
    # Build DataFusion window expression from partition_cols, order_cols, frame
    ...
```

**New functions in functions.py:**
```python
# Ranking
row_number()      # F.row_number()
rank()            # F.rank()
dense_rank()      # F.dense_rank()
percent_rank()    # F.percent_rank()
cume_dist()       # F.cume_dist()
ntile(n)          # F.ntile(n)

# Analytic
lag(col, offset, default)   # F.lag()
lead(col, offset, default)  # F.lead()
first_value(col)            # F.first_value()  [already exists as first()]
last_value(col)             # F.last_value()   [already exists as last()]

# Window aggregates (same funcs, used with .over())
sum, avg, min, max, count  # already exist, just need .over() wiring
```

**Usage:**
```python
from iceberg_spark.functions import row_number, rank
from iceberg_spark.window import Window

w = Window.partitionBy("dept").orderBy("salary")
df.withColumn("rank", rank().over(w))
df.withColumn("rn", row_number().over(w))
```

### 3.2 DataFrame: `.na` Accessor  ✅
**File:** `iceberg_spark/dataframe.py` (new class `DataFrameNaFunctions`)

```python
df.na.fill(0)                        # fill all numeric nulls with 0
df.na.fill({"col1": 0, "col2": ""}) # per-column fill
df.na.drop()                         # drop rows with any null
df.na.drop(subset=["col1", "col2"]) # drop rows with null in subset
df.na.replace(old, new)             # replace values
```

### 3.3 DataFrame: `.stat` Accessor  ✅
**File:** `iceberg_spark/dataframe.py` (new class `DataFrameStatFunctions`)

```python
df.stat.corr("col1", "col2")             # Pearson correlation
df.stat.cov("col1", "col2")              # Covariance
df.stat.crosstab("col1", "col2")         # Cross-tabulation
df.stat.freqItems(["col1"], support=0.1) # Frequent items
df.stat.sampleBy("col", {0: 0.5, 1: 1.0}) # Stratified sample
```

### 3.4 DataFrame: `sample()`, `toDF()`, `transform()`  ✅
**File:** `iceberg_spark/dataframe.py`

```python
df.sample(fraction=0.5, seed=42)           # random sample
df.sample(withReplacement=True, fraction=0.5)
df.toDF("new_col1", "new_col2")           # rename all columns
df.transform(lambda df: df.filter(...))   # apply function
df.hint("broadcast")                       # no-op compatibility hint
```

### 3.5 IcebergSession: `range()` + `createDataFrame` improvements  ✅
**File:** `iceberg_spark/session.py`

```python
session.range(10)             # DataFrame with id col 0..9
session.range(5, 15)          # DataFrame with id col 5..14
session.range(0, 100, 10)     # DataFrame with id col 0..90 step 10
```

Also improve `createDataFrame()` to handle `Row` objects as input.

### 3.6 DataFusion CatalogProvider Integration  ✅
**File:** `iceberg_spark/_internal/catalog_factory.py`, `iceberg_spark/session.py`

Register Iceberg catalog as a DataFusion `CatalogProvider` so fully qualified SQL works
without manual table registration:
```python
session.sql("SELECT * FROM my_catalog.my_db.my_table")
```

Uses `IcebergCatalogProvider` from `crates/integrations/datafusion/src/catalog.rs` if exposed
through pyiceberg-core, or implements in pure Python via lazy registration.

### 3.7 Time Travel (Proper Snapshot Support)  ✅
**File:** `iceberg_spark/session.py`, `iceberg_spark/_internal/table_registration.py`

Current implementation registers the current snapshot. Proper time travel requires
loading a specific snapshot.

Options:
1. Use PyIceberg's `table.snapshot_by_id()` + `table.scan(snapshot_id=...)` to create
   an Arrow table at a specific snapshot, then register as in-memory DataFusion table.
2. Expose `try_new_from_table_snapshot()` through pyiceberg-core PyO3 bindings (Rust side).

**PyIceberg approach (Python-only, no Rust changes):**
```python
snapshot = table.snapshot_by_id(snapshot_id)
scan = table.scan(snapshot_id=snapshot_id)
arrow_table = scan.to_arrow()
ctx.register_record_batches(tt_name, [arrow_table.to_batches()])
```

### 3.8 Metadata Tables  ✅
**File:** `iceberg_spark/session.py`, `iceberg_spark/sql_preprocessor.py`

```python
session.sql("SELECT * FROM db.t1.snapshots")
session.sql("SELECT * FROM db.t1.manifests")
session.sql("SELECT * FROM db.t1.entries")
session.sql("SELECT * FROM db.t1.files")
```

Intercept `<table>.<metadata_type>` pattern in preprocessor.
Use PyIceberg: `table.inspect.snapshots()`, `table.inspect.manifests()`, etc.

### 3.9 Spark Catalog API  ✅
**File:** `iceberg_spark/session.py` (new class `IcebergCatalogAPI`)

```python
session.catalog.listTables("db")
session.catalog.listDatabases()
session.catalog.tableExists("db.table")
session.catalog.databaseExists("db")
session.catalog.setCurrentDatabase("db")
session.catalog.currentDatabase()
```

---

## Phase 4 — DML (Complete)

All DML operations implemented via copy-on-write (no deletion vectors needed).

### 4.1 MERGE INTO  ✅
Copy-on-write: load target + source as Arrow, build UNION ALL query in DataFusion
combining matched updates/deletes + unmatched target rows + not-matched inserts,
overwrite target via `table.overwrite(AlwaysTrue())`.

```python
session.sql("""
    MERGE INTO db.target t USING db.source s ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET t.name = s.name, t.salary = s.salary
    WHEN NOT MATCHED THEN INSERT (id, name, salary) VALUES (s.id, s.name, s.salary)
""")
```

Supports: WHEN MATCHED THEN UPDATE/DELETE, WHEN NOT MATCHED THEN INSERT,
conditional WHEN clauses (AND condition), multiple WHEN clauses.

### 4.2 UPDATE  ✅
Copy-on-write: read table as Arrow, build `SELECT ... CASE WHEN (condition) THEN (expr) ELSE col END`
via DataFusion, overwrite with result via `table.overwrite(AlwaysTrue())`.

```python
session.sql("UPDATE db.t1 SET salary = salary * 1.1 WHERE dept = 'eng'")
session.sql("UPDATE db.t1 SET score = 0")  # no WHERE — update all
```

### 4.3 DELETE FROM  ✅
```python
session.sql("DELETE FROM db.t1 WHERE id > 100")
session.sql("DELETE FROM db.t1")  # no WHERE — delete all
```
Copy-on-write: scan table, `SELECT * WHERE NOT (condition)` via DataFusion, overwrite.

---

## Rust-Side Work Needed

### Partitioned Writes (Phase 2)
**File:** `crates/integrations/datafusion/src/table/mod.rs:183`

`insert_into()` currently returns `NotImplemented` for partitioned tables.
The projection and repartition infrastructure exists but isn't wired into the write path.
This is the single most important Rust-side gap to close for Phase 2.

### Snapshot-based Table Provider (Phase 3)
**File:** `crates/integrations/datafusion/src/table/mod.rs`

Expose `try_new_from_table_snapshot()` through PyO3 bindings in
`bindings/python/src/datafusion_table_provider.rs` to enable true time travel.

---

## Testing Strategy

### Unit Tests (no catalog required)
All in `tests/` — just need `datafusion` + `pyarrow`:
- `tests/test_row.py` ✅
- `tests/test_types.py` ✅
- `tests/test_sql_preprocessor.py` ✅
- `tests/test_display.py` ✅
- `tests/test_datafusion_integration.py` ✅
- `tests/test_window.py` ✅
- `tests/test_dataframe_enhancements.py` ✅

### Integration Tests (requires catalog)
In `tests/integration/` — needs PyIceberg + SQLite/REST catalog:
- `tests/integration/test_session_e2e.py` ✅
  - Create table, insert, select roundtrip
  - ALTER TABLE column operations
  - SHOW TABLES, DESCRIBE TABLE
  - Time travel queries
- `tests/integration/test_dml_e2e.py` ✅
- `tests/integration/test_time_travel.py` ✅
- `tests/integration/test_metadata_tables.py` ✅
- `tests/integration/test_partitioned_tables.py` ✅

### Smoke Test
```python
from iceberg_spark import IcebergSession

session = (IcebergSession.builder()
    .catalog("sql", uri="sqlite:///test.db", warehouse="file:///tmp/wh")
    .build())

session.sql("CREATE DATABASE IF NOT EXISTS test")
session.sql("CREATE TABLE IF NOT EXISTS test.t1 (id INT, name STRING) USING iceberg")
session.sql("INSERT INTO test.t1 VALUES (1, 'hello'), (2, 'world')")
session.sql("SELECT * FROM test.t1").show()
session.sql("ALTER TABLE test.t1 ADD COLUMN score DOUBLE")
session.sql("SELECT count(*), max(id) FROM test.t1").show()
```

---

## Key DataFusion v52 API Notes

- `lit()` is at `datafusion.lit`, NOT `datafusion.functions.lit`
- `col()` available at both `datafusion.col` and `datafusion.functions.col`
- `ctx.register_record_batches(name, [batches])` for in-memory tables
- `ctx.register_table_provider(name, provider)` for custom table providers
- `ctx.deregister_table(name)` to remove a table
- `df.aggregate(group_exprs, agg_exprs)` for groupBy + agg
- `df.with_column(name, expr)` for withColumn
- `df.with_column_renamed(old, new)` for withColumnRenamed
- PyIceberg `table.append(arrow_table)` for appending data
- PyIceberg `table.overwrite(arrow_table, overwrite_filter)` for overwrite
- PyIceberg `table.update_schema()` context manager for schema evolution

---

## Dependencies

```toml
[project]
dependencies = [
    "datafusion>=50",
    "pyiceberg[pyarrow]>=0.10.0",
    "pyiceberg-core>=0.7.0",  # pyiceberg-core from this fork for FFI bridge
    "pyarrow>=15.0",
]

[project.optional-dependencies]
sql-catalog = ["sqlalchemy>=2.0"]   # for SQLite/SQL catalog
```

Note: `pyiceberg-core` must be built from this fork (maturin build) to match the
iceberg-rust version. Standard PyPI pyiceberg-core targets a different iceberg-rust version.

---

## Partitioned Table Support (2026-03-23)

Full support for partitioned table creation, writes, and DML. All operations go through
the PyIceberg write path (not DataFusion's native `insert_into()`), which handles
partitioned file layout internally.

### Partition Transforms Supported
- `PARTITIONED BY (col)` — identity
- `PARTITIONED BY (bucket(N, col))` — hash bucket
- `PARTITIONED BY (year(col))` / `month(col)` / `day(col)` / `hour(col)` — temporal
- `PARTITIONED BY (truncate(N, col))` — truncation
- Multi-column: `PARTITIONED BY (region, category)`

### Features
- **CREATE TABLE ... PARTITIONED BY (...)** — SQL DDL with all transform types
- **CREATE TABLE ... PARTITIONED BY (...) AS SELECT ...** — CTAS with partition specs
- **INSERT INTO / INSERT OVERWRITE** — on partitioned tables
- **DELETE / UPDATE / MERGE INTO** — copy-on-write DML on partitioned tables
- **df.write.partitionBy("col").saveAsTable("t")** — auto-creates table with partition spec
- **df.writeTo("t").partitionedBy("col").create()** — WriterV2 with partition spec

### Key Implementation Details
- PyIceberg's `table.append()` / `table.overwrite()` handle partitioned file layout
  internally — no changes needed to the write path itself
- `_pa_schema_to_iceberg()` converts PyArrow schemas to PyIceberg Schemas with proper
  field IDs, required because PyIceberg's `assign_fresh_partition_spec_ids()` needs
  resolvable source IDs in the "old schema"
- `_parse_partition_by()` handles nested parentheses (e.g., `bucket(16, id)`) via
  balanced-paren scanning, not simple regex
- `_invalidate_table_cache()` properly deregisters tables from DataFusion after writes
  to prevent stale reads

### Files Modified
- `iceberg_spark/catalog_ops.py` — partition parsing, CTAS support, cache invalidation
- `iceberg_spark/writer.py` — `partitionBy()` wired into table creation
- `iceberg_spark/writer_v2.py` — `partitionedBy()` wired into `create()`
- `iceberg_spark/session.py` — CTAS pass-through, cache invalidation
- `iceberg_spark/sql_preprocessor.py` — CTAS regex updated for middle clause

### Known Limitations
- `overwrite(AlwaysTrue())` replaces ALL partitions; partition-scoped overwrites are a
  future optimization
- The Rust-side native write path (`insert_into()` in DataFusion) still blocks partitioned
  tables — the new TaskWriter (commit a970a0c) and partitioning node (commit 7ea6713)
  are available but not yet wired through. This is a future performance optimization.

---

## Quality Sprint (2026-03-23)

Hardening pass after partitioned table support. 802 tests, 0 deprecation warnings.

### Changes
- **Deprecation fix**: `register_table_provider()` → `register_table()` in table_registration.py
  — eliminates all 42 deprecation warnings from DataFusion v52
- **Type mappings**: `_arrow_to_iceberg_type()` expanded with BinaryType, TimeType,
  recursive ListType/MapType/StructType, non-UTC timestamp handling, warning on fallback
- **mapInPandas / mapInArrow**: Implemented as single-partition apply (was NotImplementedError).
  Handles empty results, multiple yields, StructType schema conversion, None-return guard.
- **README**: Removed false limitation claims (CTAS, subqueries), added Supported Features section

### Files Modified
- `iceberg_spark/_internal/table_registration.py` — register_table API migration
- `iceberg_spark/catalog_ops.py` — expanded type mappings
- `iceberg_spark/dataframe.py` — mapInPandas, mapInArrow, _resolve_schema helper
- `README.md` — updated limitations and features
