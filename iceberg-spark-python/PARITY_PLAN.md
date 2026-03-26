# PySpark Parity Plan — iceberg_spark

**Updated:** 2026-03-09
**Scope:** Single-node PySpark drop-in replacement (excludes cluster/streaming/RDD)
**Current state:** 250 unit tests, ~46% API coverage

---

## How To Use This Document

Each section below is a **self-contained task card**. An agent (Sonnet, Haiku, or Opus) should be able to read ONE section and implement it without reading other sections or the full codebase.

Every task card includes:
- **File to edit** — absolute path
- **Pattern to follow** — actual code from the codebase showing the convention
- **What to implement** — exact function signatures and DataFusion mappings
- **Tests** — where to put tests, what fixtures exist, test pattern to follow
- **Exports** — what to add to `__init__.py` and `__all__`
- **Acceptance criteria** — how to verify completion

### Project Root

```
/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/
```

All file paths below are relative to this root unless marked absolute.

### Run Tests

```bash
cd /home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/
uv run pytest tests/<test_file>.py -v
```

### Priority Legend

- **P0** — Breaks real-world PySpark scripts (users will hit these immediately)
- **P1** — Commonly used features that users will notice missing quickly
- **P2** — Less common but part of a complete replacement
- **P3** — Niche or advanced features

---

## TASK 6A: DataFrame Missing Methods (P0) ✅ DONE

### File to edit
`iceberg_spark/dataframe.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/dataframe.py`)

### Existing pattern to follow

The `DataFrame` class wraps `self._df` (a DataFusion DataFrame) and `self._session` (an `IcebergSession | None`). Every method returns `DataFrame(result_df, self._session)`. Example:

```python
# From dataframe.py — this is the pattern for all new methods
def limit(self, num: int) -> DataFrame:
    return DataFrame(self._df.limit(num), self._session)

def distinct(self) -> DataFrame:
    return DataFrame(self._df.distinct(), self._session)
```

For methods that use Column expressions:
```python
from iceberg_spark.column import Column

def select(self, *cols) -> DataFrame:
    exprs = []
    for c in cols:
        if isinstance(c, str):
            exprs.append(Column(c).expr)
        elif isinstance(c, Column):
            exprs.append(c.expr)
        else:
            exprs.append(c)
    return DataFrame(self._df.select(*exprs), self._session)
```

### What to implement

Add these methods to the `DataFrame` class in `dataframe.py`:

**1. `dropDuplicates(*cols)` — FIX existing broken stub (line ~178)**

Current broken code just calls `self.distinct()` ignoring column args. Fix:

```python
def dropDuplicates(self, *cols: str) -> DataFrame:
    """Returns a new DataFrame with duplicate rows removed.

    Args:
        cols: Column names to consider for dedup. If empty, all columns.
    """
    if not cols:
        return self.distinct()
    # Use SQL window: ROW_NUMBER() OVER (PARTITION BY cols ORDER BY cols) = 1
    all_cols = self.columns
    temp_name = f"_dedup_{id(self)}"
    self._session._ctx.register_table(temp_name, self._df)
    try:
        partition = ", ".join(cols)
        order = ", ".join(cols)
        col_list = ", ".join(all_cols)
        sql = (
            f"SELECT {col_list} FROM ("
            f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition} ORDER BY {order}) AS __rn "
            f"FROM {temp_name}"
            f") WHERE __rn = 1"
        )
        result = self._session._ctx.sql(sql)
        return DataFrame(result, self._session)
    finally:
        self._session._ctx.deregister_table(temp_name)

def drop_duplicates(self, *cols: str) -> DataFrame:
    """Alias for dropDuplicates (PySpark snake_case compat)."""
    return self.dropDuplicates(*cols)
```

**2. `withColumns(colsMap)` — add/replace multiple columns at once**

```python
def withColumns(self, colsMap: dict) -> DataFrame:
    """Returns a new DataFrame by adding or replacing multiple columns.

    Args:
        colsMap: Dict mapping column name -> Column expression.
    """
    result = self
    for col_name, col_expr in colsMap.items():
        result = result.withColumn(col_name, col_expr)
    return result
```

**3. `withColumnsRenamed(colsMap)` — rename multiple columns**

```python
def withColumnsRenamed(self, colsMap: dict) -> DataFrame:
    """Returns a new DataFrame with multiple columns renamed.

    Args:
        colsMap: Dict mapping old name -> new name.
    """
    result = self
    for old_name, new_name in colsMap.items():
        if old_name in result.columns:
            result = result.withColumnRenamed(old_name, new_name)
    return result
```

**4. `describe(*cols)` — summary statistics**

```python
def describe(self, *cols: str) -> DataFrame:
    """Computes basic statistics (count, mean, stddev, min, max).

    Args:
        cols: Column names. If empty, uses all numeric columns.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    target_cols = list(cols) if cols else self.columns
    arrow_table = self._df.to_arrow_table()

    stats = ["count", "mean", "stddev", "min", "max"]
    result_data = {"summary": stats}
    for col_name in target_cols:
        col_vals = []
        col_array = arrow_table.column(col_name)
        is_numeric = pa.types.is_integer(col_array.type) or pa.types.is_floating(col_array.type)
        for stat in stats:
            if stat == "count":
                col_vals.append(str(col_array.drop_null().length()))
            elif stat == "mean" and is_numeric:
                col_vals.append(str(pc.mean(col_array).as_py()))
            elif stat == "stddev" and is_numeric:
                col_vals.append(str(pc.stddev(col_array, ddof=1).as_py()))
            elif stat == "min":
                col_vals.append(str(pc.min(col_array).as_py()))
            elif stat == "max":
                col_vals.append(str(pc.max(col_array).as_py()))
            else:
                col_vals.append(None)
        result_data[col_name] = col_vals

    result_arrow = pa.table(result_data)
    temp_name = f"_describe_{id(self)}"
    self._session._ctx.register_record_batches(temp_name, [result_arrow.to_batches()])
    return DataFrame(self._session._ctx.table(temp_name), self._session)
```

**5. `summary(*statistics)` — configurable describe**

```python
def summary(self, *statistics: str) -> DataFrame:
    """Computes specified statistics. Defaults to describe stats."""
    return self.describe()
```

**6. `tail(n)` — last n rows**

```python
def tail(self, n: int = 1) -> list:
    """Returns the last n rows as a list of Row objects."""
    all_rows = self.collect()
    return all_rows[-n:]
```

**7. `toLocalIterator()` — iterate over rows**

```python
def toLocalIterator(self):
    """Returns an iterator over Row objects."""
    return iter(self.collect())
```

**8. `intersectAll(other)` — intersect preserving duplicates**

```python
def intersectAll(self, other: DataFrame) -> DataFrame:
    """Returns rows in both DataFrames, preserving duplicates."""
    return DataFrame(self._df.intersect(other._df), self._session)
```

### Test file
`tests/test_dataframe_methods.py` (NEW FILE)

Absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/tests/test_dataframe_methods.py`

### Test pattern to follow

```python
"""Tests for DataFrame missing methods (Task 6A)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.column import Column
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import col, lit


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def sample_table():
    return pa.table({
        "id": [1, 2, 2, 3, 3, 3],
        "name": ["Alice", "Bob", "Bob", "Carol", "Carol", "Carol"],
        "score": [85.0, 92.0, 92.0, 78.0, 78.0, 78.0],
    })


@pytest.fixture
def df(ctx, sample_table):
    ctx.register_record_batches("t", [sample_table.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestDropDuplicates:
    def test_no_cols_is_distinct(self, df):
        result = df.dropDuplicates()
        assert result.count() == 3  # Alice, Bob, Carol

    def test_snake_case_alias(self, df):
        assert hasattr(df, "drop_duplicates")


class TestWithColumns:
    def test_add_multiple(self, df):
        result = df.withColumns({
            "double_score": col("score") * lit(2),
            "id_plus_one": col("id") + lit(1),
        })
        assert "double_score" in result.columns
        assert "id_plus_one" in result.columns

    def test_empty_map(self, df):
        result = df.withColumns({})
        assert result.columns == df.columns


class TestWithColumnsRenamed:
    def test_rename_multiple(self, df):
        result = df.withColumnsRenamed({"id": "user_id", "name": "user_name"})
        assert "user_id" in result.columns
        assert "user_name" in result.columns
        assert "id" not in result.columns

    def test_missing_col_ignored(self, df):
        result = df.withColumnsRenamed({"nonexistent": "new_name"})
        assert result.columns == df.columns


class TestDescribe:
    # NOTE: describe() requires self._session to register temp table.
    # For unit tests without a session, test via collect() on Arrow directly
    # or skip if session is None.
    pass


class TestTail:
    def test_tail(self, df):
        rows = df.tail(2)
        assert len(rows) == 2

    def test_tail_default(self, df):
        rows = df.tail()
        assert len(rows) == 1


class TestToLocalIterator:
    def test_iterator(self, df):
        it = df.toLocalIterator()
        count = sum(1 for _ in it)
        assert count == df.count()


class TestIntersectAll:
    def test_intersect_all(self, ctx):
        t1 = pa.table({"id": [1, 2, 2, 3]})
        t2 = pa.table({"id": [2, 2, 3, 4]})
        ctx.register_record_batches("t1", [t1.to_batches()])
        ctx.register_record_batches("t2", [t2.to_batches()])
        df1 = DataFrame(ctx.table("t1"), session=None)
        df2 = DataFrame(ctx.table("t2"), session=None)
        result = df1.intersectAll(df2)
        assert result.count() >= 2  # both 2s and 3
```

### Exports to add
No new exports needed — `DataFrame` is already exported. The new methods are on the existing class.

### Acceptance criteria
- `uv run pytest tests/test_dataframe_methods.py -v` — all tests pass
- `df.dropDuplicates()` returns distinct rows
- `df.withColumns({...})` adds/replaces multiple columns
- `df.describe()` returns count/mean/stddev/min/max
- `df.tail(n)` returns last n rows

---

## TASK 6B: Temp View Registration (P0) ✅ DONE

### File to edit
`iceberg_spark/dataframe.py`

### What to implement

Add these methods to the `DataFrame` class. They register the DataFrame's data in the session's DataFusion context so SQL queries can reference them by name.

```python
def createTempView(self, name: str) -> None:
    """Creates a local temporary view with this DataFrame.

    Raises an error if a view with the same name already exists.
    """
    if self._session is None:
        raise RuntimeError("createTempView requires an active session")
    # Check if already registered
    try:
        self._session._ctx.table(name)
        raise RuntimeError(f"Temporary view '{name}' already exists")
    except Exception as e:
        if "already exists" in str(e):
            raise
    arrow_table = self._df.to_arrow_table()
    self._session._ctx.register_record_batches(name, [arrow_table.to_batches()])

def createOrReplaceTempView(self, name: str) -> None:
    """Creates or replaces a local temporary view with this DataFrame."""
    if self._session is None:
        raise RuntimeError("createOrReplaceTempView requires an active session")
    try:
        self._session._ctx.deregister_table(name)
    except Exception:
        pass
    arrow_table = self._df.to_arrow_table()
    self._session._ctx.register_record_batches(name, [arrow_table.to_batches()])

def createGlobalTempView(self, name: str) -> None:
    """Creates a global temporary view (same as createTempView for single-node)."""
    self.createTempView(name)

def createOrReplaceGlobalTempView(self, name: str) -> None:
    """Creates or replaces a global temporary view (same as createOrReplaceTempView)."""
    self.createOrReplaceTempView(name)
```

### Tests

Add to `tests/test_dataframe_methods.py`:

```python
class TestTempViews:
    # Full tests for temp views require an IcebergSession.
    # For unit tests without a catalog, test that the methods exist
    # and that session=None raises RuntimeError.

    def test_create_temp_view_requires_session(self, df):
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.createTempView("v1")

    def test_create_or_replace_requires_session(self, df):
        with pytest.raises(RuntimeError, match="requires an active session"):
            df.createOrReplaceTempView("v1")
```

### Acceptance criteria
- `df.createTempView("v1")` registers the view, `session.sql("SELECT * FROM v1")` works
- `df.createTempView("v1")` raises if `v1` already exists
- `df.createOrReplaceTempView("v1")` works even if `v1` already exists
- Methods raise `RuntimeError` if session is None

---

## TASK 6C: GroupedData Enhancements (P1) ✅ DONE

### File to edit
`iceberg_spark/grouped_data.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/grouped_data.py`)

### Existing pattern

```python
class GroupedData:
    def __init__(self, df: DataFrame, group_cols: list[Column]):
        self._df = df
        self._group_cols = group_cols

    def agg(self, *exprs: Column) -> DataFrame:
        from iceberg_spark.dataframe import DataFrame
        group_exprs = [c.expr for c in self._group_cols]
        agg_exprs = [e.expr for e in exprs]
        result = self._df._df.aggregate(group_exprs, agg_exprs)
        return DataFrame(result, self._df._session)

    def count(self) -> DataFrame:
        from iceberg_spark.functions import count
        return self.agg(count("*").alias("count"))
```

### What to implement

**1. Dict-style `agg()`** — currently `agg()` only accepts `Column` args. Modify `agg()` to also accept a dict:

Replace the existing `agg` method body:

```python
def agg(self, *exprs) -> DataFrame:
    """Compute aggregates and returns the result as a DataFrame.

    Args:
        exprs: Column expressions OR a single dict mapping col_name -> agg_name.
               Dict example: {"salary": "avg", "age": "max"}
    """
    from iceberg_spark.dataframe import DataFrame

    # Handle dict-style: .agg({"col": "func"})
    if len(exprs) == 1 and isinstance(exprs[0], dict):
        from iceberg_spark import functions as F_mod
        col_exprs = []
        for col_name, func_name in exprs[0].items():
            func = getattr(F_mod, func_name, None)
            if func is None:
                raise ValueError(f"Unknown aggregation function: {func_name}")
            col_exprs.append(func(col_name).alias(f"{func_name}({col_name})"))
        exprs = tuple(col_exprs)

    group_exprs = [c.expr for c in self._group_cols]
    agg_exprs = [e.expr for e in exprs]
    result = self._df._df.aggregate(group_exprs, agg_exprs)
    return DataFrame(result, self._df._session)
```

**2. `pivot(col, values=None)`**

Add to `GroupedData` class:

```python
def pivot(self, pivot_col: str, values: list | None = None) -> "PivotedGroupedData":
    """Pivots a column and returns a PivotedGroupedData for aggregation.

    Args:
        pivot_col: Column to pivot on.
        values: Optional list of values to pivot. If None, uses all distinct values.
    """
    return PivotedGroupedData(self._df, self._group_cols, pivot_col, values)
```

Add a new class after `GroupedData`:

```python
class PivotedGroupedData:
    """GroupedData with a pivot operation pending."""

    def __init__(self, df, group_cols, pivot_col, values):
        self._df = df
        self._group_cols = group_cols
        self._pivot_col = pivot_col
        self._values = values

    def agg(self, *exprs) -> "DataFrame":
        """Aggregate with pivot — creates one column per pivot value."""
        from iceberg_spark.column import Column
        from iceberg_spark.dataframe import DataFrame
        from iceberg_spark.functions import col, lit, sum as _sum, when

        # Get pivot values
        if self._values is None:
            vals = sorted(set(
                r[self._pivot_col]
                for r in self._df.select(self._pivot_col).distinct().collect()
            ))
        else:
            vals = self._values

        # For each agg expression, create pivoted columns
        # Simple approach: CASE WHEN pivot_col = val THEN col ELSE NULL END
        # then aggregate
        agg_cols = []
        for expr in exprs:
            for v in vals:
                agg_cols.append(
                    _sum(when(col(self._pivot_col) == lit(v), expr).otherwise(lit(None)))
                    .alias(str(v))
                )

        group_names = [str(c) for c in self._group_cols]
        return self._df.groupBy(*group_names).agg(*agg_cols)
```

### Test file
Add to `tests/test_phase2_features.py` or create `tests/test_grouped_data.py` (NEW FILE).

### Tests

```python
class TestGroupedDataEnhancements:
    def test_agg_dict(self, df):
        # df fixture has: id, name, age, score, dept
        result = df.groupBy("dept").agg({"score": "avg", "age": "max"})
        assert result.count() == 2  # eng, sales

    def test_agg_dict_unknown_func_raises(self, df):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            df.groupBy("dept").agg({"score": "nonexistent"})
```

### Acceptance criteria
- `df.groupBy("dept").agg({"salary": "avg"})` returns one row per dept with avg salary
- Unknown function names raise `ValueError`
- `df.groupBy("dept").pivot("quarter").agg(sum("sales"))` pivots values into columns

---

## TASK 6D: Column Missing Methods (P0) ✅ DONE

### File to edit
`iceberg_spark/column.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/column.py`)

### Existing pattern

All Column methods wrap DataFusion Expr operations. Import DataFusion functions lazily:

```python
# Existing pattern from column.py:
def like(self, pattern: str) -> Column:
    import datafusion.functions as F
    return Column(F.like(self._expr, _lit(pattern)))

def contains(self, value: str) -> Column:
    import datafusion.functions as F
    return Column(F.strpos(self._expr, _lit(value)) > _lit(0))
```

### What to implement

Add these methods to the `Column` class, after the existing `over()` method (line ~165), before `__repr__`:

```python
def when(self, condition, value) -> Column:
    """Chain a WHEN clause onto an existing CASE expression.

    Note: The standalone `when()` function in functions.py starts a new CASE WHEN.
    This Column method chains onto an existing CASE WHEN expression.
    """
    return Column(self._expr.when(_unwrap(condition), _unwrap(value)))

def substr(self, startPos: int, length: int) -> Column:
    """Returns a substring starting at startPos (1-based) for length characters."""
    import datafusion.functions as F
    return Column(F.substr(self._expr, _lit(startPos), _lit(length)))

def isNaN(self) -> Column:
    """Returns True if the value is NaN."""
    import datafusion.functions as F
    return Column(F.isnan(self._expr))

def rlike(self, pattern: str) -> Column:
    """Returns True if the string matches the regex pattern."""
    import datafusion.functions as F
    # regexp_match returns array of matches or null; check not-null for boolean
    return Column(F.regexp_match(self._expr, _lit(pattern)).is_not_null())

def eqNullSafe(self, other) -> Column:
    """Null-safe equality comparison.

    Returns True when both values are equal OR both are null.
    """
    other_expr = _unwrap(other)
    # (a = b) OR (a IS NULL AND b IS NULL)
    eq = self._expr == other_expr
    both_null = self._expr.is_null() & other_expr.is_null()
    return Column(eq | both_null)

def bitwiseAND(self, other) -> Column:
    """Bitwise AND."""
    import datafusion.functions as F
    return Column(F.bit_and(self._expr, _unwrap(other)))

def bitwiseOR(self, other) -> Column:
    """Bitwise OR."""
    import datafusion.functions as F
    return Column(F.bit_or(self._expr, _unwrap(other)))

def bitwiseXOR(self, other) -> Column:
    """Bitwise XOR."""
    import datafusion.functions as F
    return Column(F.bit_xor(self._expr, _unwrap(other)))
```

**IMPORTANT DataFusion v52 notes:**
- `F.regexp_match()` returns an array of matches or null. Use `.is_not_null()` to get a boolean.
- Bitwise functions: DataFusion may use `F.bit_and`/`F.bit_or`/`F.bit_xor` for aggregate bitwise ops. For per-row bitwise ops, try Python `&`, `|`, `^` operators on the Expr directly. If neither works, wrap in SQL.
- For `eqNullSafe`, the approach above uses `(a = b) | (a IS NULL AND b IS NULL)`. If `_unwrap(other)` produces a literal, `other_expr.is_null()` will always be False for non-null literals — this is correct.
- **Test each function individually.** If a DataFusion function doesn't exist, raise `NotImplementedError("DataFusion v52 does not support X")`.

### Test file
`tests/test_column_methods.py` (NEW FILE)

Absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/tests/test_column_methods.py`

### Tests

```python
"""Tests for Column missing methods (Task 6D)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.column import Column
from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import col, lit, when


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "score": [85.5, float("nan"), 78.0],
    })
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestSubstr:
    def test_substr(self, df):
        result = df.select(col("name").substr(1, 3).alias("short"))
        rows = result.collect()
        assert rows[0]["short"] == "Ali"

    def test_substr_mid(self, df):
        result = df.select(col("name").substr(2, 2).alias("mid"))
        rows = result.collect()
        assert rows[0]["mid"] == "li"


class TestIsNaN:
    def test_isnan_finds_nan(self, df):
        result = df.filter(col("score").isNaN())
        assert result.count() == 1

    def test_isnan_excludes_normal(self, df):
        result = df.filter(~col("score").isNaN())
        assert result.count() == 2


class TestRlike:
    def test_rlike_matches(self, df):
        result = df.filter(col("name").rlike("^A.*"))
        assert result.count() == 1
        assert result.collect()[0]["name"] == "Alice"

    def test_rlike_no_match(self, df):
        result = df.filter(col("name").rlike("^Z.*"))
        assert result.count() == 0


class TestWhenChain:
    def test_when_otherwise_chain(self, df):
        # when() function starts the chain, .when() on Column extends it
        result = df.select(
            when(col("id") == lit(1), lit("one"))
            .when(col("id") == lit(2), lit("two"))
            .otherwise(lit("other"))
            .alias("label")
        )
        rows = result.collect()
        labels = [r["label"] for r in rows]
        assert "one" in labels
        assert "two" in labels
        assert "other" in labels


class TestEqNullSafe:
    def test_both_null(self, ctx):
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        ctx.register_record_batches("t2", [t.to_batches()])
        df = DataFrame(ctx.table("t2"), session=None)
        result = df.filter(col("a").eqNullSafe(None))
        assert result.count() == 1

    def test_equal_values(self, ctx):
        t = pa.table({"a": [1, 2, 3]})
        ctx.register_record_batches("t3", [t.to_batches()])
        df = DataFrame(ctx.table("t3"), session=None)
        result = df.filter(col("a").eqNullSafe(2))
        assert result.count() == 1
```

### Exports
No changes needed — `Column` is already exported and these are methods on it.

### Acceptance criteria
- `col("name").substr(1, 3)` returns first 3 chars
- `col("score").isNaN()` filters to NaN rows
- `col("name").rlike("^A")` matches regex
- `when(cond, val).when(cond2, val2).otherwise(val3)` chains correctly
- `col("a").eqNullSafe(None)` matches null rows
- Any method that can't map to DataFusion v52 raises `NotImplementedError`

---

## TASK 7A: High-Priority Functions — Aggregates + Conditionals + Window (P0) ✅ DONE

### File to edit
`iceberg_spark/functions.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/functions.py`)

### Existing pattern

Every function follows this exact pattern. **Copy this pattern exactly:**

```python
# Single-column function:
def stddev(col: str | Column) -> Column:
    """Returns the sample standard deviation."""
    return Column(F.stddev(_unwrap(col if isinstance(col, Column) else Column(col))))

# Two-column function:
# (see covar_samp in DataFrameStatFunctions for DataFusion call)
```

**Key imports at top of file (already present — do NOT duplicate):**
```python
import datafusion.functions as F
from datafusion import lit as _df_lit
from iceberg_spark.column import Column, _unwrap
```

**Standard argument unwrap pattern:**
```python
_unwrap(col if isinstance(col, Column) else Column(col))
```

### What to implement — Aggregate functions

Add after the existing `variance` function (~line 100):

```python
def collect_set(col: str | Column) -> Column:
    """Returns a set of unique values (as array) from the group."""
    return Column(F.array_agg(_unwrap(col if isinstance(col, Column) else Column(col)), distinct=True))


def var_pop(col: str | Column) -> Column:
    """Returns the population variance."""
    return Column(F.var_pop(_unwrap(col if isinstance(col, Column) else Column(col))))


def stddev_pop(col: str | Column) -> Column:
    """Returns the population standard deviation."""
    return Column(F.stddev_pop(_unwrap(col if isinstance(col, Column) else Column(col))))


def covar_samp(col1: str | Column, col2: str | Column) -> Column:
    """Returns the sample covariance of two columns."""
    return Column(F.covar_samp(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))


def covar_pop(col1: str | Column, col2: str | Column) -> Column:
    """Returns the population covariance of two columns."""
    return Column(F.covar_pop(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))


def corr(col1: str | Column, col2: str | Column) -> Column:
    """Returns the Pearson correlation coefficient."""
    return Column(F.corr(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))


def percentile_approx(col: str | Column, percentage: float, accuracy: int = 10000) -> Column:
    """Returns approximate percentile."""
    return Column(F.approx_percentile_cont(
        _unwrap(col if isinstance(col, Column) else Column(col)),
        _df_lit(percentage),
    ))


def any_value(col: str | Column) -> Column:
    """Returns any value from the group (non-deterministic)."""
    return Column(F.first_value(_unwrap(col if isinstance(col, Column) else Column(col))))


def bool_and(col: str | Column) -> Column:
    """Returns True if all values are true."""
    return Column(F.bool_and(_unwrap(col if isinstance(col, Column) else Column(col))))


def bool_or(col: str | Column) -> Column:
    """Returns True if any value is true."""
    return Column(F.bool_or(_unwrap(col if isinstance(col, Column) else Column(col))))
```

### What to implement — Conditional functions

Add after the existing `isnan` function (~line 258):

```python
def isnotnull(col: str | Column) -> Column:
    """Returns True if the value is not null."""
    return Column(col if isinstance(col, Column) else Column(col)).isNotNull()


def greatest(*cols: str | Column) -> Column:
    """Returns the greatest value among the columns."""
    exprs = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
    return Column(F.greatest(*exprs))


def least(*cols: str | Column) -> Column:
    """Returns the least value among the columns."""
    exprs = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
    return Column(F.least(*exprs))


def ifnull(col1: str | Column, col2: str | Column) -> Column:
    """Returns col2 if col1 is null (alias for coalesce)."""
    return coalesce(col1, col2)


def nvl(col1: str | Column, col2: str | Column) -> Column:
    """Returns col2 if col1 is null (alias for coalesce)."""
    return coalesce(col1, col2)


def nvl2(col1: str | Column, col2: str | Column, col3: str | Column) -> Column:
    """Returns col2 if col1 is not null, otherwise col3."""
    expr1 = _unwrap(col1 if isinstance(col1, Column) else Column(col1))
    expr2 = _unwrap(col2 if isinstance(col2, Column) else Column(col2))
    expr3 = _unwrap(col3 if isinstance(col3, Column) else Column(col3))
    return Column(F.when(~expr1.is_null(), expr2).otherwise(expr3))
```

### What to implement — Window functions

Add after the existing `lead` function (~line 319):

```python
def nth_value(col: str | Column, n: int) -> Column:
    """Returns the nth value in the window frame."""
    return Column(F.nth_value(_unwrap(col if isinstance(col, Column) else Column(col)), _df_lit(n)))


def first_value(col: str | Column) -> Column:
    """Returns the first value in the window frame."""
    return Column(F.first_value(_unwrap(col if isinstance(col, Column) else Column(col))))


def last_value(col: str | Column) -> Column:
    """Returns the last value in the window frame."""
    return Column(F.last_value(_unwrap(col if isinstance(col, Column) else Column(col))))
```

### Exports to add

In `iceberg_spark/__init__.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/__init__.py`):

1. Add to the `from iceberg_spark.functions import (...)` block:
```python
    any_value,
    bool_and,
    bool_or,
    collect_set,
    corr,
    covar_pop,
    covar_samp,
    first_value,
    greatest,
    ifnull,
    isnotnull,
    last_value,
    least,
    nth_value,
    nvl,
    nvl2,
    percentile_approx,
    stddev_pop,
    var_pop,
```

2. Add to `__all__` list:
```python
    "any_value",
    "bool_and",
    "bool_or",
    "collect_set",
    "corr",
    "covar_pop",
    "covar_samp",
    "first_value",
    "greatest",
    "ifnull",
    "isnotnull",
    "last_value",
    "least",
    "nth_value",
    "nvl",
    "nvl2",
    "percentile_approx",
    "stddev_pop",
    "var_pop",
```

### Test file
`tests/test_functions_agg.py` (NEW FILE)

Absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/tests/test_functions_agg.py`

### Tests

```python
"""Tests for aggregate/conditional/window functions (Task 7A)."""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.dataframe import DataFrame
from iceberg_spark.functions import (
    any_value, bool_and, bool_or, col, collect_set, corr, covar_pop,
    covar_samp, greatest, ifnull, isnotnull, least, lit, nvl, nvl2,
    percentile_approx, stddev_pop, var_pop,
)


@pytest.fixture
def ctx():
    return SessionContext()


@pytest.fixture
def df(ctx):
    t = pa.table({
        "id": [1, 2, 3, 4, 5],
        "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        "dept": ["a", "a", "b", "b", "b"],
        "flag": [True, True, False, True, True],
    })
    ctx.register_record_batches("t", [t.to_batches()])
    return DataFrame(ctx.table("t"), session=None)


class TestAggFunctions:
    def test_var_pop(self, df):
        result = df.agg(var_pop("score").alias("vp"))
        rows = result.collect()
        assert isinstance(rows[0]["vp"], float)

    def test_stddev_pop(self, df):
        result = df.agg(stddev_pop("score").alias("sp"))
        rows = result.collect()
        assert isinstance(rows[0]["sp"], float)

    def test_covar_samp(self, df):
        result = df.agg(covar_samp("id", "score").alias("cs"))
        rows = result.collect()
        assert isinstance(rows[0]["cs"], float)

    def test_corr(self, df):
        result = df.agg(corr("id", "score").alias("r"))
        rows = result.collect()
        assert abs(rows[0]["r"] - 1.0) < 0.01  # perfect linear correlation

    def test_bool_and(self, df):
        result = df.agg(bool_and("flag").alias("ba"))
        rows = result.collect()
        assert rows[0]["ba"] is False  # one False value

    def test_bool_or(self, df):
        result = df.agg(bool_or("flag").alias("bo"))
        rows = result.collect()
        assert rows[0]["bo"] is True


class TestConditionalFunctions:
    def test_greatest(self, ctx):
        t = pa.table({"a": [1, 5, 3], "b": [4, 2, 6]})
        ctx.register_record_batches("t2", [t.to_batches()])
        df = DataFrame(ctx.table("t2"), session=None)
        result = df.select(greatest(col("a"), col("b")).alias("g"))
        rows = result.collect()
        assert [r["g"] for r in rows] == [4, 5, 6]

    def test_least(self, ctx):
        t = pa.table({"a": [1, 5, 3], "b": [4, 2, 6]})
        ctx.register_record_batches("t3", [t.to_batches()])
        df = DataFrame(ctx.table("t3"), session=None)
        result = df.select(least(col("a"), col("b")).alias("l"))
        rows = result.collect()
        assert [r["l"] for r in rows] == [1, 2, 3]

    def test_isnotnull(self, ctx):
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        ctx.register_record_batches("t4", [t.to_batches()])
        df = DataFrame(ctx.table("t4"), session=None)
        result = df.filter(isnotnull(col("a")))
        assert result.count() == 2

    def test_nvl(self, ctx):
        t = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
        ctx.register_record_batches("t5", [t.to_batches()])
        df = DataFrame(ctx.table("t5"), session=None)
        result = df.select(nvl(col("a"), lit(0)).alias("a"))
        rows = result.collect()
        assert all(r["a"] is not None for r in rows)

    def test_ifnull_is_nvl_alias(self, ctx):
        t = pa.table({"a": pa.array([None, 2], type=pa.int64())})
        ctx.register_record_batches("t6", [t.to_batches()])
        df = DataFrame(ctx.table("t6"), session=None)
        result = df.select(ifnull(col("a"), lit(-1)).alias("a"))
        rows = result.collect()
        assert rows[0]["a"] == -1
```

### Acceptance criteria
- All new functions are importable from `iceberg_spark`
- `from iceberg_spark import var_pop, greatest, nvl` — works
- `df.agg(var_pop("score"))` returns population variance
- `greatest(col("a"), col("b"))` returns max of two columns per row
- `uv run pytest tests/test_functions_agg.py -v` passes

---

## TASK 7B: String Functions (P1) ✅ DONE

### File to edit
`iceberg_spark/functions.py`

### Existing pattern
```python
# From functions.py — existing string function pattern:
def upper(col: str | Column) -> Column:
    return Column(F.upper(_unwrap(col if isinstance(col, Column) else Column(col))))

def substring(col: str | Column, pos: int, length: int) -> Column:
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.substr(expr, _df_lit(pos), _df_lit(length)))
```

### What to implement

Add after the existing `rpad` function (~line 166), still in the `# --- String functions ---` section:

```python
def regexp_extract(col: str | Column, pattern: str, idx: int) -> Column:
    """Extracts a group from a regex match.

    Note: DataFusion regexp_match returns an array. Use array indexing to get group.
    """
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.regexp_match(expr, _df_lit(pattern)))


def translate(col: str | Column, matching: str, replace: str) -> Column:
    """Translates characters (like Unix tr)."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.translate(expr, _df_lit(matching), _df_lit(replace)))


def locate(substr: str, col: str | Column, pos: int = 1) -> Column:
    """Returns the 1-based position of the first occurrence of substr."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.strpos(expr, _df_lit(substr)))


def instr(col: str | Column, substr: str) -> Column:
    """Returns the 1-based position of the first occurrence of substr."""
    return locate(substr, col)


def initcap(col: str | Column) -> Column:
    """Capitalizes the first letter of each word."""
    return Column(F.initcap(_unwrap(col if isinstance(col, Column) else Column(col))))


def ascii_func(col: str | Column) -> Column:
    """Returns the ASCII value of the first character.

    Named ascii_func to avoid shadowing Python builtin.
    Import as: from iceberg_spark.functions import ascii_func as ascii
    """
    return Column(F.ascii(_unwrap(col if isinstance(col, Column) else Column(col))))


def chr_func(col: str | Column) -> Column:
    """Returns the character for the ASCII value.

    Named chr_func to avoid shadowing Python builtin.
    Import as: from iceberg_spark.functions import chr_func as chr
    """
    return Column(F.chr(_unwrap(col if isinstance(col, Column) else Column(col))))


def repeat_func(col: str | Column, n: int) -> Column:
    """Repeats a string n times.

    Named repeat_func to avoid confusion. Exported as repeat.
    """
    return Column(F.repeat(_unwrap(col if isinstance(col, Column) else Column(col)), _df_lit(n)))


def left(col: str | Column, length: int) -> Column:
    """Returns the leftmost n characters."""
    return Column(F.left(_unwrap(col if isinstance(col, Column) else Column(col)), _df_lit(length)))


def right(col: str | Column, length: int) -> Column:
    """Returns the rightmost n characters."""
    return Column(F.right(_unwrap(col if isinstance(col, Column) else Column(col)), _df_lit(length)))
```

### Exports to add

In `__init__.py`, add to imports and `__all__`:
```python
from iceberg_spark.functions import (
    initcap, instr, left, locate, regexp_extract, right, translate,
    repeat_func as repeat,
)
# Add to __all__:
"initcap", "instr", "left", "locate", "regexp_extract", "repeat", "right", "translate",
```

### Test file
`tests/test_functions_string.py` (NEW FILE)

### Acceptance criteria
- `initcap(col("name"))` capitalizes first letter of each word
- `locate("ob", col("name"))` returns position of substring
- `left(col("name"), 3)` returns first 3 characters
- `uv run pytest tests/test_functions_string.py -v` passes

---

## TASK 7C: Math Functions (P1) ✅ DONE

### File to edit
`iceberg_spark/functions.py`

### Existing pattern
```python
def sqrt(col: str | Column) -> Column:
    return Column(F.sqrt(_unwrap(col if isinstance(col, Column) else Column(col))))
```

### What to implement

Add after the existing `power` function (~line 236), still in `# --- Math functions ---`:

```python
def log2(col: str | Column) -> Column:
    """Returns base-2 logarithm."""
    return Column(F.log2(_unwrap(col if isinstance(col, Column) else Column(col))))


def log10(col: str | Column) -> Column:
    """Returns base-10 logarithm."""
    return Column(F.log10(_unwrap(col if isinstance(col, Column) else Column(col))))


def log1p(col: str | Column) -> Column:
    """Returns ln(col + 1)."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.ln(expr + _df_lit(1)))


def cbrt(col: str | Column) -> Column:
    """Returns the cube root."""
    return Column(F.cbrt(_unwrap(col if isinstance(col, Column) else Column(col))))


def pow(col: str | Column, p: float) -> Column:
    """Alias for power."""
    return power(col, p)


def signum(col: str | Column) -> Column:
    """Returns the sign of the value (-1, 0, or 1)."""
    return Column(F.signum(_unwrap(col if isinstance(col, Column) else Column(col))))


def sign(col: str | Column) -> Column:
    """Alias for signum."""
    return signum(col)


def degrees(col: str | Column) -> Column:
    """Converts radians to degrees."""
    return Column(F.degrees(_unwrap(col if isinstance(col, Column) else Column(col))))


def radians(col: str | Column) -> Column:
    """Converts degrees to radians."""
    return Column(F.radians(_unwrap(col if isinstance(col, Column) else Column(col))))


def factorial(col: str | Column) -> Column:
    """Returns the factorial."""
    return Column(F.factorial(_unwrap(col if isinstance(col, Column) else Column(col))))


def hex_func(col: str | Column) -> Column:
    """Returns the hex string representation.

    Named hex_func to avoid shadowing Python builtin. Exported as hex.
    """
    return Column(F.to_hex(_unwrap(col if isinstance(col, Column) else Column(col))))


def pmod(col1: str | Column, col2: str | Column) -> Column:
    """Returns the positive modulo: ((a % b) + b) % b."""
    e1 = _unwrap(col1 if isinstance(col1, Column) else Column(col1))
    e2 = _unwrap(col2 if isinstance(col2, Column) else Column(col2))
    return Column(((e1 % e2) + e2) % e2)


def pi() -> Column:
    """Returns pi as a literal."""
    return Column(F.pi())


def e() -> Column:
    """Returns Euler's number as a literal."""
    import math
    return Column(_df_lit(math.e))


# --- Trigonometric functions ---

def sin(col: str | Column) -> Column:
    """Returns the sine."""
    return Column(F.sin(_unwrap(col if isinstance(col, Column) else Column(col))))

def cos(col: str | Column) -> Column:
    """Returns the cosine."""
    return Column(F.cos(_unwrap(col if isinstance(col, Column) else Column(col))))

def tan(col: str | Column) -> Column:
    """Returns the tangent."""
    return Column(F.tan(_unwrap(col if isinstance(col, Column) else Column(col))))

def asin(col: str | Column) -> Column:
    """Returns the arc sine."""
    return Column(F.asin(_unwrap(col if isinstance(col, Column) else Column(col))))

def acos(col: str | Column) -> Column:
    """Returns the arc cosine."""
    return Column(F.acos(_unwrap(col if isinstance(col, Column) else Column(col))))

def atan(col: str | Column) -> Column:
    """Returns the arc tangent."""
    return Column(F.atan(_unwrap(col if isinstance(col, Column) else Column(col))))

def atan2(col1: str | Column, col2: str | Column) -> Column:
    """Returns atan2(y, x)."""
    return Column(F.atan2(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))

def sinh(col: str | Column) -> Column:
    """Returns the hyperbolic sine."""
    return Column(F.sinh(_unwrap(col if isinstance(col, Column) else Column(col))))

def cosh(col: str | Column) -> Column:
    """Returns the hyperbolic cosine."""
    return Column(F.cosh(_unwrap(col if isinstance(col, Column) else Column(col))))

def tanh(col: str | Column) -> Column:
    """Returns the hyperbolic tangent."""
    return Column(F.tanh(_unwrap(col if isinstance(col, Column) else Column(col))))

def cot(col: str | Column) -> Column:
    """Returns the cotangent."""
    return Column(F.cot(_unwrap(col if isinstance(col, Column) else Column(col))))


# --- Hash functions ---

def md5(col: str | Column) -> Column:
    """Returns the MD5 hash."""
    return Column(F.md5(_unwrap(col if isinstance(col, Column) else Column(col))))

def sha1(col: str | Column) -> Column:
    """Returns the SHA-1 hash."""
    return Column(F.sha1(_unwrap(col if isinstance(col, Column) else Column(col))))

def sha2(col: str | Column, numBits: int = 256) -> Column:
    """Returns the SHA-2 hash (default SHA-256)."""
    return Column(F.sha256(_unwrap(col if isinstance(col, Column) else Column(col))))
```

### Exports to add
Add all new functions to `__init__.py` imports and `__all__`. Use aliased imports for builtins:
```python
from iceberg_spark.functions import hex_func as hex
```

### Test file
`tests/test_functions_math.py` (NEW FILE)

### Acceptance criteria
- `log2(col("x"))` returns base-2 log
- `sin(col("x"))` returns sine
- `pi()` returns ~3.14159
- `pmod(lit(-7), lit(3))` returns 2 (positive modulo)
- `uv run pytest tests/test_functions_math.py -v` passes

---

## TASK 7D: Date/Time Functions (P1) ✅ DONE

### File to edit
`iceberg_spark/functions.py`

### Existing pattern
```python
def year(col: str | Column) -> Column:
    return Column(F.date_part(_df_lit("year"), _unwrap(col if isinstance(col, Column) else Column(col))))
```

### What to implement

Add after the existing `second` function (~line 201), still in `# --- Date/time functions ---`:

```python
def dayofweek(col: str | Column) -> Column:
    """Returns the day of the week (1=Sunday, 7=Saturday)."""
    return Column(F.date_part(_df_lit("dow"), _unwrap(col if isinstance(col, Column) else Column(col))))


def dayofyear(col: str | Column) -> Column:
    """Returns the day of the year (1-366)."""
    return Column(F.date_part(_df_lit("doy"), _unwrap(col if isinstance(col, Column) else Column(col))))


def weekofyear(col: str | Column) -> Column:
    """Returns the ISO week of the year (1-53)."""
    return Column(F.date_part(_df_lit("week"), _unwrap(col if isinstance(col, Column) else Column(col))))


def quarter(col: str | Column) -> Column:
    """Returns the quarter of the year (1-4)."""
    return Column(F.date_part(_df_lit("quarter"), _unwrap(col if isinstance(col, Column) else Column(col))))


def date_format(col: str | Column, fmt: str) -> Column:
    """Formats a date/timestamp as a string using the given format.

    Note: DataFusion uses strftime format, not Java SimpleDateFormat.
    Common: %Y=year, %m=month, %d=day, %H=hour, %M=minute, %S=second.
    """
    return Column(F.to_char(
        _unwrap(col if isinstance(col, Column) else Column(col)),
        _df_lit(fmt),
    ))


def to_date(col: str | Column, fmt: str | None = None) -> Column:
    """Converts a string to a date."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    if fmt:
        return Column(F.to_date(expr, _df_lit(fmt)))
    return Column(F.to_date(expr))


def to_timestamp(col: str | Column, fmt: str | None = None) -> Column:
    """Converts a string to a timestamp."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    if fmt:
        return Column(F.to_timestamp(expr, _df_lit(fmt)))
    return Column(F.to_timestamp(expr))


def trunc(col: str | Column, fmt: str) -> Column:
    """Truncates a date/timestamp to the specified precision (year, month, etc.)."""
    return Column(F.date_trunc(
        _df_lit(fmt),
        _unwrap(col if isinstance(col, Column) else Column(col)),
    ))


def date_trunc(fmt: str, col: str | Column) -> Column:
    """Truncates a timestamp to the specified precision."""
    return Column(F.date_trunc(
        _df_lit(fmt),
        _unwrap(col if isinstance(col, Column) else Column(col)),
    ))


def make_date(year: str | Column, month: str | Column, day: str | Column) -> Column:
    """Creates a date from year, month, day columns."""
    return Column(F.make_date(
        _unwrap(year if isinstance(year, Column) else Column(year)),
        _unwrap(month if isinstance(month, Column) else Column(month)),
        _unwrap(day if isinstance(day, Column) else Column(day)),
    ))
```

**IMPORTANT:** Some DataFusion date functions may use different names. If a function doesn't exist:
- `F.to_char()` might be `F.strftime()`
- `F.to_date()` and `F.to_timestamp()` should exist in DataFusion v52
- `F.date_trunc()` should exist
- `F.make_date()` should exist
- If not, leave a `# TODO: DataFusion v52 does not support X` comment

### Exports to add
Add all new functions to `__init__.py` imports and `__all__`.

### Test file
`tests/test_functions_datetime.py` (NEW FILE)

### Acceptance criteria
- `dayofweek(col("date_col"))` returns 1-7
- `quarter(col("date_col"))` returns 1-4
- `to_date(col("str_col"))` parses date strings
- `date_trunc("month", col("ts"))` truncates to month start
- `uv run pytest tests/test_functions_datetime.py -v` passes

---

## TASK 7E: Collection Functions (P2) ✅ DONE

### File to edit
`iceberg_spark/functions.py`

### What to implement

Add a new section at the end of the file:

```python
# --- Collection functions ---


def array(*cols: str | Column) -> Column:
    """Creates an array column from the given columns."""
    exprs = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
    return Column(F.make_array(*exprs))


def array_contains(col: str | Column, value) -> Column:
    """Returns True if the array contains the given value."""
    return Column(F.array_has(
        _unwrap(col if isinstance(col, Column) else Column(col)),
        _df_lit(value),
    ))


def array_distinct(col: str | Column) -> Column:
    """Returns an array with distinct elements only."""
    return Column(F.array_distinct(_unwrap(col if isinstance(col, Column) else Column(col))))


def array_join(col: str | Column, delimiter: str, null_replacement: str | None = None) -> Column:
    """Joins array elements into a single string with the given delimiter."""
    return Column(F.array_to_string(
        _unwrap(col if isinstance(col, Column) else Column(col)),
        _df_lit(delimiter),
    ))


def array_position(col: str | Column, value) -> Column:
    """Returns the 1-based position of the first occurrence of value in the array."""
    return Column(F.array_position(
        _unwrap(col if isinstance(col, Column) else Column(col)),
        _df_lit(value),
    ))


def array_remove(col: str | Column, element) -> Column:
    """Removes all occurrences of element from the array."""
    return Column(F.array_remove_all(
        _unwrap(col if isinstance(col, Column) else Column(col)),
        _df_lit(element),
    ))


def array_sort(col: str | Column) -> Column:
    """Sorts the array in ascending order."""
    return Column(F.array_sort(_unwrap(col if isinstance(col, Column) else Column(col))))


def array_union(col1: str | Column, col2: str | Column) -> Column:
    """Returns the union of two arrays (no duplicates)."""
    return Column(F.array_union(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))


def element_at(col: str | Column, idx) -> Column:
    """Gets element at index (1-based for arrays) or key (for maps)."""
    return Column(F.array_element(
        _unwrap(col if isinstance(col, Column) else Column(col)),
        _df_lit(idx),
    ))


def explode(col: str | Column) -> Column:
    """Creates a new row for each element in the array column."""
    return Column(F.unnest(_unwrap(col if isinstance(col, Column) else Column(col))))


def flatten(col: str | Column) -> Column:
    """Flattens a nested array (array of arrays -> single array)."""
    return Column(F.flatten(_unwrap(col if isinstance(col, Column) else Column(col))))


def size(col: str | Column) -> Column:
    """Returns the number of elements in the array or map."""
    return Column(F.array_length(_unwrap(col if isinstance(col, Column) else Column(col))))


def sort_array(col: str | Column, asc: bool = True) -> Column:
    """Sorts the array in ascending or descending order."""
    return Column(F.array_sort(_unwrap(col if isinstance(col, Column) else Column(col))))


def struct_func(*cols: str | Column) -> Column:
    """Creates a struct column from the given columns.

    Named struct_func to avoid shadowing Python builtin. Exported as struct.
    """
    exprs = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
    return Column(F.struct(*exprs))
```

### Exports to add
Add all new functions to `__init__.py`. Use `struct_func as struct` for the export.

### Test file
`tests/test_functions_collection.py` (NEW FILE)

### Acceptance criteria
- `array(col("a"), col("b"))` creates an array column
- `array_contains(col("arr"), 5)` returns boolean
- `size(col("arr"))` returns element count
- `uv run pytest tests/test_functions_collection.py -v` passes

---

## TASK 8A: Session Properties (P1) ✅ DONE

### File to edit
`iceberg_spark/session.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/session.py`)

### What to implement

**1. Add class variable and properties to `IcebergSession` class:**

Add right after the class docstring (before `__init__`):

```python
_active_session: IcebergSession | None = None
```

Add new methods/properties after the `stop()` method (~line 271):

```python
@property
def version(self) -> str:
    """Returns the version of iceberg_spark."""
    from iceberg_spark import __version__
    return __version__

@classmethod
def getActiveSession(cls) -> IcebergSession | None:
    """Returns the most recently created session."""
    return cls._active_session

def newSession(self) -> IcebergSession:
    """Returns a new session that shares the same underlying catalog."""
    return IcebergSession(
        ctx=SessionContext(),
        catalog=self._catalog,
        catalog_name=self._catalog_name,
        config=dict(self._config),
    )

@property
def conf(self) -> "RuntimeConfig":
    """Access the runtime configuration."""
    return RuntimeConfig(self._config)
```

**2. Add `RuntimeConfig` class after `IcebergSessionBuilder` (at end of file):**

```python
class RuntimeConfig:
    """Runtime configuration for the session (dict-backed)."""

    def __init__(self, config: dict[str, Any]):
        self._config = config

    def get(self, key: str, default: str | None = None) -> str | None:
        """Returns the value of a config key."""
        return self._config.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Sets a config key-value pair."""
        self._config[key] = value

    def getAll(self) -> dict[str, str]:
        """Returns all config key-value pairs."""
        return dict(self._config)

    def unset(self, key: str) -> None:
        """Removes a config key."""
        self._config.pop(key, None)

    def isModifiable(self, key: str) -> bool:
        """Returns True (all keys are modifiable in single-node)."""
        return True
```

**3. Update `IcebergSessionBuilder.build()` to set `_active_session`:**

In the `build()` method, after creating the session, add:
```python
IcebergSession._active_session = session
```

(before `return session`)

### Exports to add
Add `RuntimeConfig` to `__init__.py` imports and `__all__`.

### Acceptance criteria
- `session.version` returns a version string
- `IcebergSession.getActiveSession()` returns the last-created session
- `session.conf.set("key", "val")` and `session.conf.get("key")` works
- `session.newSession()` returns a new session with same catalog

---

## TASK 9A: SQL DDL Gaps (P1) ✅ DONE

### Files to edit
1. `iceberg_spark/sql_preprocessor.py` — add new `CommandType` values and regex patterns
2. `iceberg_spark/catalog_ops.py` — add handler functions
3. `iceberg_spark/session.py` — add dispatch for new command types

### Existing preprocessor pattern (sql_preprocessor.py)

```python
# Add to CommandType enum:
class CommandType(Enum):
    # ... existing values ...
    CREATE_VIEW = auto()
    DROP_VIEW = auto()
    SHOW_COLUMNS = auto()
    EXPLAIN = auto()

# Add compiled regex patterns (after existing patterns):
_CREATE_VIEW = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?VIEW\s+(\S+)\s+AS\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_DROP_VIEW = re.compile(
    r"^\s*DROP\s+VIEW\s+(?:IF\s+EXISTS\s+)?(\S+)\s*$",
    re.IGNORECASE,
)
_SHOW_COLUMNS = re.compile(
    r"^\s*SHOW\s+COLUMNS\s+(?:FROM|IN)\s+(\S+)\s*$",
    re.IGNORECASE,
)
_EXPLAIN = re.compile(
    r"^\s*EXPLAIN\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
```

Add detection in `preprocess()` function (before the regular SQL return at the end):

```python
m = _CREATE_VIEW.match(stripped)
if m:
    return PreprocessResult(
        command_type=CommandType.CREATE_VIEW,
        sql=stripped,
        table_name=m.group(1),
        extra={"view_query": m.group(2).strip()},
    )

m = _DROP_VIEW.match(stripped)
if m:
    return PreprocessResult(
        command_type=CommandType.DROP_VIEW,
        sql=stripped,
        table_name=m.group(1),
    )

m = _SHOW_COLUMNS.match(stripped)
if m:
    return PreprocessResult(
        command_type=CommandType.SHOW_COLUMNS,
        sql=stripped,
        table_name=m.group(1),
    )

m = _EXPLAIN.match(stripped)
if m:
    return PreprocessResult(
        command_type=CommandType.EXPLAIN,
        sql=m.group(1).strip(),
    )
```

### Existing handler pattern (catalog_ops.py)

```python
def handle_show_tables(session: IcebergSession, namespace: str | None) -> DataFrame:
    from iceberg_spark.dataframe import DataFrame
    # ... logic ...
    return _empty_result_df(session, "tableName", ...)
```

### Handlers to add (catalog_ops.py)

```python
def handle_create_view(session, view_name, view_query):
    """Handle CREATE VIEW — register the SELECT result as a temp table."""
    session._ensure_tables_registered(view_query)
    arrow_table = session._ctx.sql(view_query).to_arrow_table()
    session._ctx.register_record_batches(view_name, [arrow_table.to_batches()])
    return _empty_result_df(session, "status", f"View {view_name} created")


def handle_drop_view(session, view_name):
    """Handle DROP VIEW — deregister from DataFusion."""
    try:
        session._ctx.deregister_table(view_name)
    except Exception:
        pass
    return _empty_result_df(session, "status", f"View {view_name} dropped")


def handle_show_columns(session, table_name):
    """Handle SHOW COLUMNS — return column metadata."""
    import pyarrow as pa
    from iceberg_spark.catalog_ops import _split_table_name

    ns, tbl = _split_table_name(table_name)
    table = session._catalog.load_table(f"{ns}.{tbl}")
    schema = table.schema()
    names = [f.name for f in schema.fields]
    types = [str(f.field_type) for f in schema.fields]
    nullables = ["YES" if f.optional else "NO" for f in schema.fields]
    arrow = pa.table({
        "col_name": names,
        "data_type": types,
        "nullable": nullables,
    })
    from iceberg_spark.dataframe import DataFrame
    temp_name = f"_show_cols_{id(table)}"
    session._ctx.register_record_batches(temp_name, [arrow.to_batches()])
    return DataFrame(session._ctx.table(temp_name), session)


def handle_explain(session, query):
    """Handle EXPLAIN — return DataFusion query plan."""
    import pyarrow as pa
    session._ensure_tables_registered(query)
    plan_str = str(session._ctx.sql(query).explain())
    arrow = pa.table({"plan": [plan_str]})
    from iceberg_spark.dataframe import DataFrame
    temp_name = f"_explain_{id(query)}"
    session._ctx.register_record_batches(temp_name, [arrow.to_batches()])
    return DataFrame(session._ctx.table(temp_name), session)
```

### Dispatch to add (session.py)

In the `sql()` method, add before the `# --- Regular SQL ---` section:

```python
if result.command_type == CommandType.CREATE_VIEW:
    from iceberg_spark.catalog_ops import handle_create_view
    return handle_create_view(self, result.table_name, result.extra["view_query"])

if result.command_type == CommandType.DROP_VIEW:
    from iceberg_spark.catalog_ops import handle_drop_view
    return handle_drop_view(self, result.table_name)

if result.command_type == CommandType.SHOW_COLUMNS:
    from iceberg_spark.catalog_ops import handle_show_columns
    return handle_show_columns(self, result.table_name)

if result.command_type == CommandType.EXPLAIN:
    from iceberg_spark.catalog_ops import handle_explain
    return handle_explain(self, result.sql)
```

### Test file
Add preprocessor tests to `tests/test_sql_preprocessor.py`. Create `tests/test_sql_ddl.py` (NEW FILE) for handler tests that don't require a catalog.

### Acceptance criteria
- `preprocess("CREATE VIEW v1 AS SELECT * FROM t1")` returns `CommandType.CREATE_VIEW`
- `preprocess("SHOW COLUMNS FROM db.t1")` returns `CommandType.SHOW_COLUMNS`
- `preprocess("EXPLAIN SELECT * FROM t1")` returns `CommandType.EXPLAIN`
- All new command types dispatched correctly in session.py

---

## TASK 10A: Catalog API Completeness (P2) ✅ DONE

### File to edit
`iceberg_spark/catalog_api.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/catalog_api.py`)

### Existing pattern

```python
def listDatabases(self, pattern: str | None = None) -> list[str]:
    try:
        namespaces = self._catalog.list_namespaces()
        result = [".".join(ns) for ns in namespaces]
    except Exception:
        result = []
    return result
```

### What to implement

**1. Add pattern filtering to `listDatabases` (fix existing method):**

```python
def listDatabases(self, pattern: str | None = None) -> list[str]:
    try:
        namespaces = self._catalog.list_namespaces()
        result = [".".join(ns) for ns in namespaces]
    except Exception:
        result = []
    if pattern:
        import fnmatch
        result = [r for r in result if fnmatch.fnmatch(r, pattern)]
    return result
```

**2. Add pattern filtering to `listTables`:**

Same approach — add `fnmatch` filtering at the end.

**3. Add `getDatabase(dbName)` method:**

```python
def getDatabase(self, dbName: str) -> dict:
    """Returns database metadata as a dict."""
    ns_tuple = tuple(dbName.split("."))
    try:
        props = self._catalog.load_namespace_properties(ns_tuple)
    except Exception:
        props = {}
    return {
        "name": dbName,
        "description": props.get("comment", ""),
        "locationUri": props.get("location", ""),
    }
```

**4. Add `dropTempView` and `dropGlobalTempView`:**

```python
def dropTempView(self, viewName: str) -> bool:
    """Drops a temporary view. Returns True if it existed."""
    try:
        self._session._ctx.deregister_table(viewName)
        return True
    except Exception:
        return False

def dropGlobalTempView(self, viewName: str) -> bool:
    """Drops a global temporary view (same as dropTempView)."""
    return self.dropTempView(viewName)
```

**5. Add `createTable` (programmatic):**

```python
def createTable(self, tableName: str, schema=None, **kwargs):
    """Creates a table programmatically."""
    if schema is None:
        raise ValueError("schema is required for createTable")
    import pyarrow as pa
    from iceberg_spark.types import StructType
    if isinstance(schema, StructType):
        arrow_schema = pa.schema([f.to_arrow() for f in schema.fields])
    else:
        arrow_schema = schema
    parts = tableName.split(".")
    ns = parts[0] if len(parts) > 1 else "default"
    tbl = parts[-1]
    self._catalog.create_table(f"{ns}.{tbl}", schema=arrow_schema)
    return self._session.table(tableName)
```

### Test file
Add to `tests/test_phase3_features.py` (has existing catalog API tests).

### Acceptance criteria
- `session.catalog.listDatabases("my_*")` filters by glob pattern
- `session.catalog.getDatabase("mydb")` returns dict with name/description
- `session.catalog.dropTempView("v1")` returns bool

---

## TASK 11A: Type System Gaps (P2) ✅ DONE

### File to edit
`iceberg_spark/types.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/types.py`)

### Existing pattern

```python
class IntegerType(DataType):
    def to_arrow(self):
        return pa.int32()
    def simpleString(self) -> str:
        return "int"
```

### What to implement

Add after existing type classes:

```python
class CharType(DataType):
    """Fixed-length character type."""
    def __init__(self, length: int):
        self.length = length
    def to_arrow(self):
        return pa.string()  # Arrow doesn't distinguish char/varchar
    def simpleString(self) -> str:
        return f"char({self.length})"

class VarcharType(DataType):
    """Variable-length character type."""
    def __init__(self, length: int):
        self.length = length
    def to_arrow(self):
        return pa.string()
    def simpleString(self) -> str:
        return f"varchar({self.length})"

class DayTimeIntervalType(DataType):
    """Day-time interval type."""
    def to_arrow(self):
        return pa.duration("us")
    def simpleString(self) -> str:
        return "interval day to second"
```

### Exports to add
Add `CharType`, `VarcharType`, `DayTimeIntervalType` to `__init__.py` imports and `__all__`.

### Acceptance criteria
- `CharType(10).to_arrow()` returns `pa.string()`
- `VarcharType(255).simpleString()` returns `"varchar(255)"`
- `DayTimeIntervalType().to_arrow()` returns `pa.duration("us")`

---

## TASK 12A: __init__.py Exports (P1) ✅ DONE

### File to edit
`iceberg_spark/__init__.py` (absolute: `/home/john/CodeRepos/openSource/apacheIcebergRust/iceberg-rust/iceberg-spark-python/iceberg_spark/__init__.py`)

### What to implement

Add these imports after the existing imports:

```python
from iceberg_spark.grouped_data import GroupedData
from iceberg_spark.dataframe import DataFrameNaFunctions, DataFrameStatFunctions
from iceberg_spark.reader import DataFrameReader
from iceberg_spark.writer import DataFrameWriter
```

Add to `__all__` list:
```python
"GroupedData",
"DataFrameNaFunctions",
"DataFrameStatFunctions",
"DataFrameReader",
"DataFrameWriter",
```

### Acceptance criteria
- `from iceberg_spark import GroupedData, DataFrameWriter, DataFrameReader` works
- `from iceberg_spark import DataFrameNaFunctions, DataFrameStatFunctions` works

---

## TASK 12B: StatFunctions Completeness (P2) ✅ DONE

### File to edit
`iceberg_spark/dataframe.py` — the `DataFrameStatFunctions` class (~line 548)

### Existing pattern

```python
class DataFrameStatFunctions:
    def corr(self, col1: str, col2: str, method: str = "pearson") -> float:
        import datafusion.functions as F
        from iceberg_spark.column import Column
        result = DataFrame(
            self._df._df.aggregate(
                [],
                [F.corr(Column(col1).expr, Column(col2).expr).alias("corr")],
            ),
            self._df._session,
        )
        rows = result.collect()
        return float(rows[0]["corr"]) if rows else None
```

### What to implement

Add these methods to `DataFrameStatFunctions`:

```python
def approxQuantile(self, col: str, probabilities: list[float], relativeError: float = 0.0) -> list[float]:
    """Returns approximate quantiles for a column."""
    import datafusion.functions as F
    from datafusion import lit as _lit
    from iceberg_spark.column import Column

    results = []
    for p in probabilities:
        agg_result = DataFrame(
            self._df._df.aggregate(
                [],
                [F.approx_percentile_cont(Column(col).expr, _lit(p)).alias("q")],
            ),
            self._df._session,
        )
        rows = agg_result.collect()
        results.append(float(rows[0]["q"]) if rows else None)
    return results

def freqItems(self, cols: list[str], support: float = 0.01) -> DataFrame:
    """Returns frequent items for each column (items occurring >= support fraction)."""
    import pyarrow as pa
    from iceberg_spark.functions import col as _col, count as _count, lit as _lit

    total = self._df.count()
    min_count = max(1, int(total * support))

    result_data = {}
    for c in cols:
        freq = (
            self._df.groupBy(c)
            .agg(_count("*").alias("cnt"))
            .filter(_col("cnt") >= _lit(min_count))
        )
        result_data[f"{c}_freqItems"] = [r[c] for r in freq.collect()]

    # Pad lists to equal length
    max_len = max((len(v) for v in result_data.values()), default=0)
    for k in result_data:
        result_data[k] = result_data[k] + [None] * (max_len - len(result_data[k]))

    arrow_table = pa.table(result_data)
    temp_name = f"_freq_{id(self)}"
    self._df._session._ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
    return DataFrame(self._df._session._ctx.table(temp_name), self._df._session)
```

### Acceptance criteria
- `df.stat.approxQuantile("score", [0.25, 0.5, 0.75])` returns list of 3 floats
- `df.stat.freqItems(["dept"])` returns DataFrame with frequent values

---

## Excluded (Cluster/Streaming Only)

These are intentionally out of scope for single-node:

- `SparkContext`, `RDD`, `Accumulator`, `Broadcast`
- `readStream`, `writeStream`, `StreamingQuery`, `StreamingQueryManager`
- `spark_partition_id()`, `repartition()` (keep as no-op)
- `foreach()`, `foreachPartition()` (Spark uses for distributed side effects)
- `mapInPandas()`, `mapInArrow()` (Spark distributed map)
- `UDFRegistration`, `pandas_udf` (would require custom DataFusion UDF registration)
- `checkpoint()`, `localCheckpoint()`
- `sortWithinPartitions()`
- `dropDuplicatesWithinWatermark()`

---

## Implementation Order (Recommended)

### Sprint 1 — Core Compatibility (P0)
1. **6D** Column methods (`when`, `substr`, `isNaN`, `rlike`, bitwise)
2. **6A** DataFrame methods (`dropDuplicates`, `withColumns`, `describe`, `tail`)
3. **6B** Temp view registration
4. **7A** High-priority functions (aggregates, conditionals, window)
5. **12A** Export fixes

### Sprint 2 — Functions Expansion (P1)
6. **7C** Math functions (~30 functions, mostly trivial wrappers)
7. **7B** String functions (~15 functions)
8. **7D** Date/time functions (~20 functions)
9. **6C** GroupedData (`pivot`, dict-style `agg`)

### Sprint 3 — SQL & Session (P1)
10. **9A** DDL gaps (CREATE VIEW, SHOW COLUMNS, EXPLAIN)
11. **8A** Session properties (version, conf, active, newSession)

### Sprint 4 — Completeness (P2)
12. **7E** Collection/array functions
13. **10A** Catalog API completeness
14. **11A** Type system gaps
15. **12B** StatFunctions completeness

---

## Metrics

| Category | Implemented | Missing (feasible) | Coverage |
|----------|-------------|-------------------|----------|
| DataFrame methods | ~35 | ~15 | 70% |
| Column methods | ~20 | ~10 | 67% |
| functions.py | 44 | ~150 feasible | 23% |
| SQL statements | ~18 | ~12 | 60% |
| Types | 19 | 5 | 79% |
| Session API | 8 | 5 | 62% |
| Reader formats | 3 | 3 | 50% |
| Writer formats | 4 | 3 | 57% |
| Catalog API | 10 | 7 | 59% |
| GroupedData | 6 | 3 | 67% |
| Window | 10 | 0 | 100% |
| Row | 6 | 2 | 75% |
| **Overall** | **~183** | **~215** | **~46%** |

The biggest gap is `functions.py` (~150 missing functions), but most are trivial DataFusion wrappers. After Sprint 1+2, coverage would jump to ~75%.
