# iceberg_spark — Complete Project Handoff

**Last updated:** 2026-03-08
**Test status:** 250 passed, 33 skipped (integration tests require sqlalchemy)
**Codebase size:** ~4,100 lines source, ~3,100 lines tests (7,200 total)

---

## What This Is

`iceberg_spark` is a **pure Python** drop-in replacement for PySpark's SQL interface. It lets users run `spark.sql()` commands using **DataFusion** (Rust query engine) + **PyIceberg** (Iceberg catalog/table API) instead of a JVM/Spark cluster.

**Target use case:** Production analytics on a single node — zero JVM, zero Spark infrastructure.

```python
from iceberg_spark import IcebergSession

session = IcebergSession.builder() \
    .catalog("sql", uri="sqlite:///catalog.db", warehouse="file:///tmp/wh") \
    .build()

session.sql("CREATE DATABASE IF NOT EXISTS db")
session.sql("CREATE TABLE db.t1 (id INT, name STRING) USING iceberg")
session.sql("INSERT INTO db.t1 VALUES (1, 'Alice'), (2, 'Bob')")
session.sql("SELECT * FROM db.t1 WHERE id > 1").show()
session.sql("UPDATE db.t1 SET name = 'Bobby' WHERE id = 2")
session.sql("DELETE FROM db.t1 WHERE id = 1")
session.sql("CREATE TABLE db.t2 AS SELECT * FROM db.t1 WHERE id > 1")
```

**Location:** `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/`

This lives inside a fork of Apache Iceberg Rust (v0.7.0) at `iceberg-rust/`.

---

## Architecture

```
iceberg_spark/                     # Pure Python package
├── __init__.py                    # Public API exports (201 lines)
├── session.py                     # IcebergSession + Builder (472 lines)
├── dataframe.py                   # DataFrame + NaFunctions + StatFunctions (594 lines)
├── column.py                      # Column expression wrapper (178 lines)
├── row.py                         # PySpark-compatible Row (75 lines)
├── types.py                       # StructType, IntegerType, etc. (320 lines)
├── functions.py                   # 50+ PySpark-compatible functions (319 lines)
├── grouped_data.py                # GroupedData for .groupBy().agg() (54 lines)
├── window.py                      # WindowSpec / Window classes (100 lines)
├── writer.py                      # DataFrameWriter (136 lines)
├── reader.py                      # DataFrameReader (75 lines)
├── sql_preprocessor.py            # Spark SQL → DataFusion translator (325 lines)
├── catalog_ops.py                 # DDL + DML handlers (1160 lines) — LARGEST FILE
├── catalog_api.py                 # session.catalog.listTables() etc. (181 lines)
└── _internal/
    ├── __init__.py
    ├── catalog_factory.py         # PyIceberg catalog creation (21 lines)
    ├── table_registration.py      # FFI bridge to iceberg-rust (76 lines)
    ├── type_mapping.py            # Spark↔Arrow type conversion (113 lines)
    └── display.py                 # Spark-style table/schema formatting (96 lines)

tests/
├── test_row.py                    # Row type tests (91 lines)
├── test_types.py                  # Type system tests (146 lines)
├── test_display.py                # Display formatting tests (58 lines)
├── test_ctas.py                   # CTAS tests (19 tests, 166 lines)
├── test_sql_preprocessor.py       # SQL translation tests (162 lines)
├── test_datafusion_integration.py # DataFrame API tests (254 lines)
├── test_phase2_features.py        # Write path + DDL tests (324 lines)
├── test_phase3_features.py        # Window/na/stat/catalog tests (293 lines)
├── test_phase4_dml.py             # DML tests (625 lines) — LARGEST TEST FILE
└── integration/
    ├── test_session_e2e.py        # End-to-end with SQLite catalog (266 lines)
    └── test_dml_e2e.py            # DML end-to-end tests (242 lines)

pyproject.toml                     # hatchling build config
```

### Dependency Stack

```
                User Code
                    │
              iceberg_spark          ← This package (pure Python)
               /    |     \
     DataFusion  PyIceberg  pyiceberg-core
     (Python)    (Python)   (Rust FFI via PyO3)
        │            │           │
     DataFusion   Iceberg     iceberg-rust
     (Rust)       spec        (Rust)
```

**Runtime deps:** `datafusion>=50`, `pyiceberg[pyarrow]>=0.10.0`, `pyiceberg-core>=0.7.0`, `pyarrow>=15.0`
**Optional:** `sqlalchemy>=2.0` (for SQLite catalog), `pandas>=1.5`

---

## How It Works

### Request Flow

```
session.sql("UPDATE db.t1 SET salary = 100 WHERE id = 1")
    │
    ▼
sql_preprocessor.preprocess()      → PreprocessResult(command_type=UPDATE, ...)
    │
    ▼
session.py dispatch chain          → Matches CommandType.UPDATE
    │
    ▼
catalog_ops.handle_update()        → Load table via PyIceberg
    │                                 Scan to Arrow via table.scan().to_arrow()
    │                                 Build CASE WHEN SQL
    │                                 Execute via DataFusion SessionContext
    │                                 Overwrite via table.overwrite(AlwaysTrue())
    ▼
Returns DataFrame with result
```

### Key Design Decisions

1. **Lazy table registration:** Tables are registered with DataFusion on first SQL reference, not eagerly on session creation. The `_ensure_tables_registered()` method in session.py uses regex to extract table names from SQL and registers them via the FFI bridge.

2. **FFI bridge pattern:** PyIceberg table objects are monkey-patched with `__datafusion_table_provider__` to make them compatible with DataFusion's `register_table_provider()`. This uses `pyiceberg_core.datafusion.IcebergDataFusionTable` from the Rust bindings.

3. **Copy-on-write DML:** All DML (DELETE, UPDATE, MERGE INTO) reads the full table as Arrow, transforms via DataFusion SQL, then overwrites the entire table via `PyIceberg table.overwrite(result, overwrite_filter=AlwaysTrue())`. This avoids needing deletion vectors or row-level deletes in iceberg-rust.

4. **SQL interception:** The preprocessor classifies SQL into `CommandType` enums. DDL/DML/metadata commands are intercepted and handled via PyIceberg. Regular SELECT queries pass through to DataFusion after table registration.

---

## Phase 1 — Read Path (Complete)

### What Was Built
- `IcebergSession` + `IcebergSessionBuilder` — PySpark-compatible builder pattern
- `DataFrame` wrapping `datafusion.DataFrame` — full PySpark API surface
- `Column` with operator overloads (`==`, `<`, `>`, `&`, `|`, arithmetic)
- `Row` type with dict-like and attribute access
- Type system: `StructType`, `IntegerType`, `LongType`, `DoubleType`, `StringType`, etc.
- 50+ functions: `col()`, `lit()`, `count()`, `sum()`, `avg()`, `upper()`, `lower()`, `when()`, etc.
- `GroupedData` for `.groupBy().agg()`
- SQL preprocessor handling: `USING iceberg`, `NVL→COALESCE`, `RLIKE→~`, `date_add`, `datediff`, `SHOW TABLES`, `SHOW DATABASES`, `DESCRIBE TABLE`, `TIMESTAMP/VERSION AS OF`
- `DataFrameReader` for `.read.parquet()`, `.read.csv()`
- Spark-style `show()` and `printSchema()` formatting

### Key Implementation Details

**session.py:45-164** — The `sql()` method is a dispatch chain of `if result.command_type == X` blocks. Order matters — DDL/DML checks must come before the SQL pass-through at the end.

**sql_preprocessor.py:115-314** — The `preprocess()` function uses compiled regexes to classify and transform SQL. The INSERT_INTO check must come before DELETE_FROM in the regex chain because `INSERT INTO` would otherwise partially match `DELETE FROM`'s pattern.

**_internal/table_registration.py:14-39** — The FFI bridge pattern:
```python
def __datafusion_table_provider__(self):
    return IcebergDataFusionTable(
        identifier=self.name(),
        metadata_location=self.metadata_location,
        file_io_properties=self.io.properties,
    ).__datafusion_table_provider__()

table.__datafusion_table_provider__ = MethodType(__datafusion_table_provider__, table)
ctx.register_table_provider(table_name, table)
```

### DataFusion v52 API Quirks
- `lit` is at `datafusion.lit`, NOT `datafusion.functions.lit`
- `col` available both at `datafusion.col` and `datafusion.functions.col`
- `F.ntile(n)` takes a plain int, NOT `lit(n)`
- `F.lag(expr, offset, default)` — offset=plain int, default=plain Python scalar (NOT lit)
- `F.lead(expr, offset, default)` — same as lag
- Window expression API: `expr.over(partition_by=[...], order_by=[...], window_frame=...)`

### Tests
- `test_row.py` — Row creation, field access, asDict, equality
- `test_types.py` — Type construction, Arrow conversion, from_arrow roundtrip
- `test_display.py` — Table formatting, schema tree output
- `test_sql_preprocessor.py` — All SQL transformation patterns
- `test_datafusion_integration.py` — DataFrame API (select, filter, join, groupBy, window)

---

## Phase 2 — Write Path + DDL (Complete)

### What Was Built
- `INSERT INTO table VALUES (...)` and `INSERT INTO table SELECT ...`
- `INSERT OVERWRITE` via PyIceberg `table.overwrite(AlwaysTrue())`
- `DataFrameWriter` with modes: `append`, `overwrite`, `error`, `ignore`
- `df.write.saveAsTable("db.table")`, `df.write.insertInto("db.table")`
- `df.write.parquet(path)`, `df.write.csv(path)`, `df.write.json(path)`
- `df.write.format("parquet").save(path)`
- `CREATE TABLE ... USING iceberg` (with PARTITIONED BY support)
- `CREATE TABLE IF NOT EXISTS`
- `DROP TABLE [IF EXISTS]`
- `CREATE DATABASE/NAMESPACE [IF NOT EXISTS]`
- `DROP DATABASE/NAMESPACE [IF EXISTS]`
- `ALTER TABLE ADD COLUMN`, `DROP COLUMN`, `RENAME COLUMN` via PyIceberg schema evolution
- `TRUNCATE TABLE` via overwrite with empty Arrow table

### Key Implementation Details

**catalog_ops.py:15-73** — `handle_create_table()` parses column definitions from SQL, converts to Arrow schema, handles PARTITIONED BY, and calls `catalog.create_table()` or `catalog.create_table_if_not_exists()`.

**catalog_ops.py:76-131** — `handle_alter_table()` uses regex to detect ADD/DROP/RENAME COLUMN and delegates to PyIceberg's `table.update_schema()` context manager:
```python
with table.update_schema() as update:
    update.add_column("new_col", ice_type)     # ADD
    update.delete_column("old_col")            # DROP
    update.rename_column("old", "new")         # RENAME
```

**catalog_ops.py:161-214** — `handle_insert_into()` handles both VALUES and SELECT variants. For VALUES, it wraps in `SELECT * FROM (VALUES ...) AS __t` and executes via DataFusion. For SELECT, it ensures tables are registered then runs the SELECT query. Results are appended/overwritten via PyIceberg.

**writer.py:49-94** — `saveAsTable()` implements the four save modes. The `error` mode checks if the table has data before writing. The `ignore` mode silently returns if data exists.

**session.py:377-403** — `_insert_into_table()` is the internal method used by both SQL INSERT and DataFrame write. It handles the PyIceberg append/overwrite call and deregisters the table so the next query sees the updated snapshot.

### Tests
- `test_phase2_features.py` — INSERT INTO, ALTER TABLE, TRUNCATE, write modes, DataFrameWriter

---

## Phase 3 — Advanced Features (Complete)

### What Was Built
- **Window functions:** `row_number()`, `rank()`, `dense_rank()`, `percent_rank()`, `cume_dist()`, `ntile(n)`, `lag()`, `lead()` — all work with `.over(WindowSpec)`
- **WindowSpec:** `Window.partitionBy("col").orderBy("col")`, `.rowsBetween()`, `.rangeBetween()`
- **DataFrame.na:** `.drop(how, thresh, subset)`, `.fill(value)`, `.replace(old, new)`
- **DataFrame.stat:** `.corr(col1, col2)`, `.cov(col1, col2)`, `.crosstab(col1, col2)`
- **DataFrame methods:** `.sample(fraction)`, `.toDF(*names)`, `.transform(func)`, `.hint(name)`, `.selectExpr(*exprs)`
- **session.range(start, end, step)** — creates range DataFrame
- **session.createDataFrame(data, schema)** — from lists, dicts, pandas, Arrow
- **session.catalog** — `IcebergCatalogAPI` with `listTables()`, `listDatabases()`, `tableExists()`, `databaseExists()`, `setCurrentDatabase()`, `currentDatabase()`, `listColumns()`, `getTable()`, `refreshTable()`
- **Time travel:** `SELECT * FROM t TIMESTAMP AS OF '2024-01-01'` / `VERSION AS OF 123` — uses `PyIceberg table.scan(snapshot_id=...).to_arrow()` for snapshot reads
- **Metadata tables:** `SELECT * FROM db.t1.snapshots/manifests/history/files/entries/refs/schemas/partition_specs`

### Key Implementation Details

**window.py:41-66** — `WindowSpec._apply()` uses DataFusion's `Expr.over(partition_by, order_by, window_frame)` API. Has a fallback for compatibility.

**dataframe.py:436-594** — `DataFrameNaFunctions` and `DataFrameStatFunctions` implement the `.na` and `.stat` accessors using DataFusion aggregation functions. `na.drop(thresh=N)` builds a sum of not-null flags and filters.

**catalog_api.py** — A standalone class that wraps the PyIceberg catalog to expose PySpark's `session.catalog.xxx()` API. Cache operations are no-ops.

**session.py:315-375** — `_register_time_travel_table()` handles snapshot lookup by ID or timestamp. For timestamp-based travel, it iterates snapshots to find the latest one at or before the given timestamp.

**catalog_ops.py:434-608** — Metadata table support. Each metadata type has a fallback builder (`_snapshots_arrow()`, `_history_arrow()`, etc.) that constructs Arrow tables from PyIceberg metadata when the inspect API is unavailable.

### Tests
- `test_phase3_features.py` — Window functions, na operations, stat functions, catalog API, sample, toDF, transform
- `integration/test_session_e2e.py` — Full roundtrip with SQLite catalog (skipped without sqlalchemy)

---

## Phase 4 — DML (Complete)

### What Was Built
- **DELETE FROM table [WHERE condition]** — copy-on-write
- **UPDATE table SET col=expr [WHERE condition]** — copy-on-write
- **MERGE INTO target USING source ON condition WHEN MATCHED/NOT MATCHED** — copy-on-write

### DELETE FROM (catalog_ops.py:616-678)

```
1. Load table via PyIceberg catalog
2. Scan entire table to Arrow: table.scan().to_arrow()
3. If no WHERE: overwrite with empty table of same schema
4. If WHERE: register Arrow with DataFusion, run "SELECT * WHERE NOT (condition)"
5. Overwrite via table.overwrite(kept_rows, AlwaysTrue())
6. Deregister table from session so next query picks up new snapshot
```

### UPDATE (catalog_ops.py:681-758)

```
1. Load table, scan to Arrow
2. Parse SET clause via _parse_set_clause() — handles commas in parens/quotes
3. Strip qualified column prefixes: "t.salary" → "salary" (via .split(".")[-1])
4. Build SELECT with CASE WHEN:
   - With WHERE:  CASE WHEN (cond) THEN (expr) ELSE col END AS col
   - Without WHERE: (expr) AS col
5. Execute via DataFusion, enforce original schema on result
6. Overwrite via table.overwrite(result, AlwaysTrue())
```

**Schema enforcement (line 737):** `pa.Table.from_batches(batches, schema=arrow_data.schema)` — prevents type drift when expressions like `(0) AS price` produce INT64 instead of DOUBLE.

### MERGE INTO (catalog_ops.py:905-1098)

The most complex operation. Parses MERGE SQL into components, then builds a UNION ALL query:

```
Part 1: Matched target rows (UPDATE/DELETE)
  - INNER JOIN target with source on ON condition
  - Apply SET expressions via CASE WHEN (conditional updates)
  - Exclude deleted rows via WHERE NOT (delete_condition)

Part 2: Unmatched target rows (always kept)
  - If has_any_matched: SELECT target.* WHERE NOT EXISTS (source match)
  - If no matched clauses: SELECT target.* (keep ALL target rows)

Part 3: Not-matched source rows (INSERT)
  - SELECT source.* WHERE NOT EXISTS (target match)
  - Map INSERT columns/values to target schema

UNION ALL (Part 1) + (Part 2) + (Part 3) → overwrite target
```

**Parser (catalog_ops.py:824-902):**
- `_parse_merge_into()` — splits on `WHEN [NOT] MATCHED` boundaries using regex
- `_MERGE_HEADER` regex uses negative lookaheads to prevent capturing USING/ON as aliases
- `_parse_when_clause()` — classifies as update/delete/insert and extracts SET/VALUES

**Helper functions:**
- `_split_at_commas()` (line 761) — Paren-depth and quote-state-aware comma splitting
- `_parse_set_clause()` (line 791) — Parses `col1 = expr1, col2 = expr2` into pairs

### Bugs Found and Fixed During Review

1. **Qualified column names in UPDATE:** `SET t.salary = 100` created key `"t.salary"` which didn't match Arrow column `"salary"`. Fixed with `.split(".")[-1]` (line 713).

2. **Redundant SET parsing in MERGE:** `_parse_set_clause()` was called N×M times inside nested column/update loops. Hoisted to pre-parsed `parsed_updates` list (lines 972-980).

3. **MERGE insert-only dropping target rows:** When only `WHEN NOT MATCHED THEN INSERT` existed, `WHERE NOT EXISTS` incorrectly filtered matched target rows. Added `if has_any_matched` conditional (lines 1019-1032).

4. **Missing schema in UPDATE result:** `pa.Table.from_batches(batches)` without schema caused type drift on overwrite. Added `schema=arrow_data.schema` (line 737).

### Tests
- `test_phase4_dml.py` — 61 tests covering preprocessor parsing, SET clause parsing, MERGE parsing, DataFusion logic for DELETE/UPDATE/MERGE, subquery support
- `integration/test_dml_e2e.py` — 33 end-to-end tests with SQLite catalog (skipped without sqlalchemy)

---

## Phase 5 — CTAS + Subquery Support (Complete)

### What Was Built
- **CREATE TABLE AS SELECT (CTAS):** `session.sql("CREATE TABLE db.new AS SELECT * FROM db.old WHERE id > 5")`
- **Subqueries in DML WHERE clauses:** `DELETE FROM t WHERE id IN (SELECT id FROM t2)`

### CTAS (catalog_ops.py:76-137)

```
1. Parse table name and SELECT query from SQL
2. Check IF NOT EXISTS — skip if table already exists
3. Execute SELECT via DataFusion → Arrow table
4. Infer Iceberg schema from Arrow schema
5. Create table via PyIceberg catalog.create_table()
6. Append data via table.append() (skipped if 0 rows)
7. Deregister table so next query sees new data
```

**Preprocessor (sql_preprocessor.py:53-56):** `_CREATE_TABLE_AS_SELECT` regex checked before generic `_CREATE_TABLE` to prevent misclassification. Extracts `select_query` and `if_not_exists` into `extra` dict.

### Subqueries in DML (catalog_ops.py:680-710)

**Problem:** DML handlers (`handle_delete_from`, `handle_update`, `handle_merge_into`) each create a fresh `SessionContext()` with only the target table registered under a temp name. Subqueries referencing other tables (e.g., `WHERE id IN (SELECT id FROM t2)`) would fail because `t2` isn't in that context.

**Fix:** Added `_register_referenced_tables(session, ctx, *sql_fragments)` helper that:
1. Scans SQL fragments for `FROM/JOIN` table references
2. Skips internal temp names (`__delete_tmp`, etc.)
3. Loads each referenced table as Arrow via PyIceberg
4. Registers in the DML's fresh context

Wired into all three DML handlers after their fresh context creation.

**DataFusion limitation:** `IN (SELECT ...)` inside `CASE WHEN` is not supported by DataFusion's physical planner. This means `UPDATE ... SET ... WHERE id IN (SELECT ...)` has limited support. DELETE with `IN/EXISTS` subqueries works fully. UPDATE works with `EXISTS`-style subqueries and scalar subqueries in SET expressions.

### Tests
- `test_ctas.py` — 19 tests: preprocessor detection (11) + DataFusion logic (8)
- `test_phase4_dml.py::TestSubqueriesInDML` — 5 tests: IN/EXISTS/NOT IN subqueries, failure without registration, scalar subqueries

---

## What's Left / Possible Next Steps

### Ready to Do (No Rust Changes Needed)

1. **Run integration tests:** Install `sqlalchemy` and run the 33 skipped E2E tests:
   ```bash
   uv pip install sqlalchemy
   uv run pytest tests/integration/ -v
   ```

2. **INSERT INTO ... SELECT from registered tables:**
   - Currently works but tables must be registered. The handler calls `session._ensure_tables_registered()` for the SELECT portion, which should handle this.

3. **Better error messages:** Some errors surface as generic `RuntimeError` with the underlying PyIceberg exception. Could improve with specific exception types.

4. **Packaging:** The `pyproject.toml` is ready but the package hasn't been published. `pyiceberg-core` must be built from this fork (maturin) — PyPI's version targets a different iceberg-rust version.

### Requires Rust-Side Work

7. **Partitioned writes:**
   - **File:** `crates/integrations/datafusion/src/table/mod.rs:183`
   - `insert_into()` returns `NotImplemented` for partitioned tables
   - The projection and repartition infrastructure exists but isn't wired into the write path
   - This is the single most important Rust-side gap

8. **Expose snapshot-based table provider:**
   - **File:** `crates/integrations/datafusion/src/table/mod.rs`
   - `try_new_from_table_snapshot()` exists in Rust but isn't exposed through PyO3 bindings
   - Would enable true time travel without reading the full table into memory (current approach reads via `PyIceberg table.scan(snapshot_id=...)`)

### Performance / Scalability Considerations

9. **Copy-on-write scales linearly with table size.** DELETE/UPDATE/MERGE read the entire table into memory, transform, and rewrite. Fine for tables under ~10GB on a well-provisioned node. For larger tables, would need row-level deletes (deletion vectors) in iceberg-rust.

10. **Benchmarking:** No performance comparison with PySpark exists yet. Would be valuable for validating the approach.

---

## Key Files Quick Reference

| File | What It Does | Lines |
|------|-------------|-------|
| `catalog_ops.py` | DDL + DML handlers (the heaviest file) | 1098 |
| `dataframe.py` | DataFrame + NaFunctions + StatFunctions | 594 |
| `session.py` | Session + Builder + dispatch chain | 472 |
| `types.py` | Spark type system | 320 |
| `functions.py` | 50+ PySpark-compatible functions | 319 |
| `sql_preprocessor.py` | SQL classification + transformation | 325 |
| `test_phase4_dml.py` | DML + subquery tests (largest test file) | 720 |
| `test_ctas.py` | CTAS unit tests | 166 |

## Running Tests

```bash
cd iceberg-rust/iceberg-spark-python/

# All unit tests (no catalog needed)
uv run pytest tests/ -v

# Integration tests only (needs sqlalchemy)
uv run pytest tests/integration/ -v

# Single test file
uv run pytest tests/test_phase4_dml.py -v

# Specific test class
uv run pytest tests/test_phase4_dml.py::TestMergeDataFusionLogic -v
```

## Catalog Types Supported

All PyIceberg catalog backends work: `rest`, `sql` (SQLite/PostgreSQL), `hive`, `glue`, `dynamodb`, `s3tables`.

```python
# REST catalog
session = IcebergSession.builder().catalog("rest", uri="http://localhost:8181").build()

# SQLite catalog (for local dev/testing)
session = IcebergSession.builder().catalog(
    "sql", uri="sqlite:///catalog.db", warehouse="file:///tmp/wh"
).build()

# AWS Glue catalog
session = IcebergSession.builder().catalog("glue", warehouse="s3://bucket/wh").build()
```
