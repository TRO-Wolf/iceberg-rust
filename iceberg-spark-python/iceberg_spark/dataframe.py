"""PySpark-compatible DataFrame wrapping a DataFusion DataFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    from iceberg_spark.session import IcebergSession


class DataFrame:
    """A distributed collection of data grouped into named columns.

    Wraps a DataFusion DataFrame and provides PySpark-compatible API.
    """

    def __init__(self, df, session: IcebergSession | None = None):
        self._df = df  # datafusion.DataFrame
        self._session = session

    # --- Schema and metadata ---

    @property
    def columns(self) -> list[str]:
        """Returns all column names as a list."""
        return [f.name for f in self._df.schema()]

    @property
    def schema(self):
        """Returns the schema of this DataFrame as a Spark StructType."""
        from iceberg_spark.types import from_arrow_schema

        arrow_schema = self._df.schema()
        return from_arrow_schema(arrow_schema)

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        """Returns all column names and their data types as a list of tuples."""
        from iceberg_spark._internal.type_mapping import arrow_type_to_spark_string

        return [
            (f.name, arrow_type_to_spark_string(f.type))
            for f in self._df.schema()
        ]

    def printSchema(self) -> None:
        """Prints the schema in tree format (Spark style)."""
        from iceberg_spark._internal.display import format_schema

        print(format_schema(self.schema))

    # --- Display ---

    def show(self, n: int = 20, truncate: bool | int = True, vertical: bool = False) -> None:
        """Prints the first n rows as a formatted table."""
        from iceberg_spark._internal.display import format_table

        table = self._df.limit(n).to_arrow_table()
        print(format_table(table, n=n, truncate=truncate))

    def __repr__(self):
        cols = ", ".join(self.columns)
        return f"DataFrame[{cols}]"

    # --- Collection ---

    def collect(self) -> list:
        """Returns all rows as a list of Row objects."""
        from iceberg_spark.row import Row

        table = self._df.to_arrow_table()
        fields = tuple(table.column_names)
        rows = []
        for batch in table.to_batches():
            for i in range(batch.num_rows):
                values = tuple(
                    batch.column(j)[i].as_py()
                    for j in range(batch.num_columns)
                )
                rows.append(Row._from_pairs(fields, values))
        return rows

    def toPandas(self) -> pd.DataFrame:
        """Returns the contents as a Pandas DataFrame."""
        return self._df.to_pandas()

    def toArrow(self) -> pa.Table:
        """Returns the contents as an Arrow Table."""
        return self._df.to_arrow_table()

    def to_arrow_table(self) -> pa.Table:
        """Returns the contents as an Arrow Table."""
        return self._df.to_arrow_table()

    def count(self) -> int:
        """Returns the number of rows."""
        result = self._df.count()
        return result

    def first(self):
        """Returns the first row."""
        rows = self.limit(1).collect()
        return rows[0] if rows else None

    def head(self, n: int = 1):
        """Returns the first n rows."""
        rows = self.limit(n).collect()
        if n == 1:
            return rows[0] if rows else None
        return rows

    def take(self, num: int) -> list:
        """Returns the first num rows as a list of Row."""
        return self.limit(num).collect()

    def isEmpty(self) -> bool:
        """Returns True if the DataFrame is empty."""
        return self.count() == 0

    # --- Transformations ---

    def select(self, *cols) -> DataFrame:
        """Projects a set of expressions and returns a new DataFrame."""
        from iceberg_spark.column import Column

        # Resolve _ToJsonColumn markers first — they need the DataFrame
        # schema to create properly-typed UDFs.  Must happen before any
        # other processing since _ToJsonColumn._expr is None.
        resolved_cols = list(cols)
        if any(getattr(c, '_is_to_json', False) for c in cols if isinstance(c, Column)):
            resolved_cols = self._resolve_to_json_columns(resolved_cols)

        # Check if any column needs SQL fallback (_SqlFuncColumn)
        has_sql_func = any(
            getattr(c, '_is_sql_func', False)
            for c in resolved_cols if isinstance(c, Column)
        )

        if has_sql_func:
            import re

            _DF_TYPE_RE = re.compile(
                r'(?:U?Int(?:8|16|32|64)|Float(?:16|32|64)'
                r'|Utf8(?:View)?|Boolean|Date(?:32|64)'
                r'|Timestamp|Binary|LargeBinary'
                r'|Decimal(?:128|256)?|Duration|Interval'
                r'|Null)\('
            )

            sql_parts = []
            for c in resolved_cols:
                if isinstance(c, str):
                    sql_parts.append('"' + c.replace('"', '""') + '"')
                elif getattr(c, '_is_sql_func', False):
                    sql_parts.append(c._sql_fragment)
                elif isinstance(c, Column):
                    name = c._expr.schema_name()
                    if _DF_TYPE_RE.search(name):
                        raise TypeError(
                            f"Cannot mix complex Column expressions "
                            f"(schema_name={name!r}) with SQL function "
                            f"columns in select(). Use selectExpr() "
                            f"instead for full SQL control."
                        )
                    sql_parts.append('"' + name.replace('"', '""') + '"')
                else:
                    sql_parts.append(str(c))
            return self.selectExpr(*sql_parts)

        exprs = []
        has_explode = False
        explode_indices = []
        for i, c in enumerate(resolved_cols):
            if isinstance(c, str):
                exprs.append(Column(c).expr)
            elif isinstance(c, Column):
                exprs.append(c.expr)
                if getattr(c, "_explode", False):
                    has_explode = True
                    explode_indices.append(i)
            else:
                exprs.append(c)
        result = self._df.select(*exprs)
        # Apply unnest for any explode-tagged columns
        if has_explode:
            import pyarrow as pa
            schema = result.schema()
            for field in schema:
                if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                    result = result.unnest_columns(field.name)
        return DataFrame(result, self._session)

    def _resolve_to_json_columns(self, cols: list) -> list:
        """Replace _ToJsonColumn markers with properly-typed UDF columns."""
        import json as _json

        import pyarrow as pa
        from datafusion.user_defined import ScalarUDF

        from iceberg_spark.column import Column

        schema = self._df.schema()

        def _make_json_udf(arrow_type):
            def _impl(arr):
                py_list = arr.to_pylist()
                results = [
                    _json.dumps(v, default=str) if v is not None else None
                    for v in py_list
                ]
                return pa.array(results, type=pa.utf8())

            name = f"_to_json_{id(arrow_type)}"
            return ScalarUDF.udf(_impl, [arrow_type], pa.utf8(), "volatile", name)

        resolved = []
        for c in cols:
            if isinstance(c, Column) and getattr(c, "_is_to_json", False):
                col_name = c._inner_expr.schema_name()
                alias_name = getattr(c, "_alias_name", None)

                # Try to find the column type from the DataFrame schema.
                # For computed expressions (e.g. struct(a, b)), the schema_name
                # won't match. In that case, evaluate the expression first to
                # determine its Arrow type.
                arrow_type = None
                for field in schema:
                    if field.name == col_name:
                        arrow_type = field.type
                        break

                if arrow_type is None:
                    # Expression not in schema — evaluate it to get the type
                    try:
                        probe = self._df.select(c._inner_expr)
                        arrow_type = probe.schema().field(0).type
                    except Exception:
                        pass

                if arrow_type is None or pa.types.is_string(
                    arrow_type
                ) or pa.types.is_large_string(arrow_type):
                    result_col = Column(c._inner_expr.cast(pa.utf8()))
                else:
                    udf = _make_json_udf(arrow_type)
                    result_col = Column(udf(c._inner_expr))
                if alias_name:
                    result_col = result_col.alias(alias_name)
                resolved.append(result_col)
            else:
                resolved.append(c)
        return resolved

    def selectExpr(self, *exprs: str) -> DataFrame:
        """Projects a set of SQL expressions and returns a new DataFrame."""
        import uuid

        if self._session is None:
            raise RuntimeError("selectExpr requires an active session")
        # Build a SQL query from expressions
        expr_list = ", ".join(exprs)
        # Create a temp view with a unique name to avoid collisions
        temp_name = f"_selectexpr_{uuid.uuid4().hex[:8]}"
        self._session._ctx.register_table(temp_name, self._df)
        try:
            result = self._session._ctx.sql(f"SELECT {expr_list} FROM {temp_name}")
            return DataFrame(result, self._session)
        finally:
            self._session._ctx.deregister_table(temp_name)

    def filter(self, condition) -> DataFrame:
        """Filters rows using the given condition."""
        from iceberg_spark.column import Column

        if isinstance(condition, str):
            # SQL expression string
            return DataFrame(self._df.filter(condition), self._session)
        elif isinstance(condition, Column):
            return DataFrame(self._df.filter(condition.expr), self._session)
        else:
            return DataFrame(self._df.filter(condition), self._session)

    def where(self, condition) -> DataFrame:
        """Alias for filter."""
        return self.filter(condition)

    def limit(self, num: int) -> DataFrame:
        """Returns a new DataFrame by taking the first num rows."""
        return DataFrame(self._df.limit(num), self._session)

    def distinct(self) -> DataFrame:
        """Returns a new DataFrame containing the distinct rows."""
        return DataFrame(self._df.distinct(), self._session)

    def dropDuplicates(self, *cols: str) -> DataFrame:
        """Returns a new DataFrame with duplicate rows removed.

        Args:
            cols: Column names to consider for dedup. If empty, all columns.
        """
        if not cols:
            return self.distinct()
        if self._session is None:
            raise RuntimeError(
                "dropDuplicates with specific columns requires an active session"
            )
        all_cols = self.columns
        # Quote column names with double-quotes to prevent SQL injection / reserved-word clashes
        def _quote(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        temp_name = f"_dedup_{id(self)}"
        self._session._ctx.register_table(temp_name, self._df)
        try:
            partition = ", ".join(_quote(c) for c in cols)
            order = ", ".join(_quote(c) for c in cols)
            col_list = ", ".join(_quote(c) for c in all_cols)
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

    def drop(self, *cols: str) -> DataFrame:
        """Returns a new DataFrame without the specified columns."""
        keep = [c for c in self.columns if c not in cols]
        return self.select(*keep)

    def withColumns(self, colsMap: dict) -> DataFrame:
        """Returns a new DataFrame by adding or replacing multiple columns.

        Args:
            colsMap: Dict mapping column name -> Column expression.
        """
        result = self
        for col_name, col_expr in colsMap.items():
            result = result.withColumn(col_name, col_expr)
        return result

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

    def withColumn(self, colName: str, col) -> DataFrame:
        """Returns a new DataFrame by adding or replacing a column."""
        from iceberg_spark.column import Column

        # Resolve _ToJsonColumn markers before accessing .expr
        if isinstance(col, Column) and getattr(col, "_is_to_json", False):
            resolved = self._resolve_to_json_columns([col])[0]
            expr = resolved.alias(colName).expr
        elif isinstance(col, Column):
            expr = col.alias(colName).expr
        else:
            expr = col
        return DataFrame(self._df.with_column(colName, expr), self._session)

    def withColumnRenamed(self, existing: str, new: str) -> DataFrame:
        """Returns a new DataFrame by renaming an existing column."""
        return DataFrame(self._df.with_column_renamed(existing, new), self._session)

    def sort(self, *cols, **kwargs) -> DataFrame:
        """Returns a new DataFrame sorted by the specified columns."""
        from iceberg_spark.column import Column

        sort_exprs = []
        ascending = kwargs.get("ascending", True)

        for c in cols:
            if isinstance(c, str):
                col_obj = Column(c)
                if not ascending:
                    sort_exprs.append(col_obj.desc().expr)
                else:
                    sort_exprs.append(col_obj.asc().expr)
            elif isinstance(c, Column):
                sort_exprs.append(c.expr)
            else:
                sort_exprs.append(c)
        return DataFrame(self._df.sort(*sort_exprs), self._session)

    def orderBy(self, *cols, **kwargs) -> DataFrame:
        """Alias for sort."""
        return self.sort(*cols, **kwargs)

    def alias(self, name: str) -> DataFrame:
        """Returns a new DataFrame with an alias set."""
        # DataFusion doesn't have a direct alias for DataFrames,
        # but we can register as a temp table and reference by name
        return DataFrame(self._df, self._session)

    def join(
        self,
        other: DataFrame,
        on=None,
        how: str = "inner",
    ) -> DataFrame:
        """Joins with another DataFrame.

        Args:
            other: Right side of the join.
            on: Column name(s) to join on (str, list of str, or Column expression).
            how: Join type - 'inner', 'left', 'right', 'full', 'cross', 'semi', 'anti'.
        """
        from iceberg_spark.column import Column

        # Map Spark join types to DataFusion
        join_type_map = {
            "inner": "inner",
            "outer": "full",
            "full": "full",
            "full_outer": "full",
            "fullouter": "full",
            "left": "left",
            "left_outer": "left",
            "leftouter": "left",
            "right": "right",
            "right_outer": "right",
            "rightouter": "right",
            "cross": "cross",
            "semi": "semi",
            "left_semi": "semi",
            "leftsemi": "semi",
            "anti": "anti",
            "left_anti": "anti",
            "leftanti": "anti",
        }
        df_how = join_type_map.get(how.lower(), how)

        if on is None:
            result = self._df.join(other._df, join_type=df_how)
        elif isinstance(on, str):
            result = self._df.join(other._df, join_keys=([on], [on]), how=df_how)
        elif isinstance(on, (list, tuple)):
            if all(isinstance(c, str) for c in on):
                result = self._df.join(other._df, join_keys=(list(on), list(on)), how=df_how)
            else:
                # Column expressions - join on condition
                from iceberg_spark.column import _unwrap
                combined = on[0]
                for c in on[1:]:
                    combined = combined & c
                result = self._df.join(other._df, on=_unwrap(combined), how=df_how)
        elif isinstance(on, Column):
            result = self._df.join(other._df, on=on.expr, how=df_how)
        else:
            result = self._df.join(other._df, on=on, how=df_how)

        return DataFrame(result, self._session)

    def crossJoin(self, other: DataFrame) -> DataFrame:
        """Returns the cartesian product with another DataFrame."""
        return self.join(other, how="cross")

    def union(self, other: DataFrame) -> DataFrame:
        """Returns a new DataFrame containing union of rows."""
        return DataFrame(self._df.union(other._df), self._session)

    def unionAll(self, other: DataFrame) -> DataFrame:
        """Alias for union."""
        return self.union(other)

    def unionByName(self, other: DataFrame, allowMissingColumns: bool = False) -> DataFrame:
        """Returns union by column name."""
        return self.union(other)

    def intersect(self, other: DataFrame) -> DataFrame:
        """Returns rows in both DataFrames."""
        return DataFrame(self._df.intersect(other._df), self._session)

    def subtract(self, other: DataFrame) -> DataFrame:
        """Returns rows in this DataFrame but not in another."""
        return DataFrame(self._df.except_all(other._df), self._session)

    def exceptAll(self, other: DataFrame) -> DataFrame:
        """Returns rows in this DataFrame but not in another."""
        return self.subtract(other)

    def intersectAll(self, other: DataFrame) -> DataFrame:
        """Returns rows in both DataFrames, preserving duplicates."""
        return DataFrame(self._df.intersect(other._df), self._session)

    def tail(self, n: int = 1) -> list:
        """Returns the last n rows as a list of Row objects."""
        if n <= 0:
            return []
        # Use DataFusion's native tail(), which avoids full materialization
        return DataFrame(self._df.tail(n), self._session).collect()

    def toLocalIterator(self):
        """Returns an iterator over Row objects."""
        return iter(self.collect())

    def randomSplit(self, weights: list[float], seed: int | None = None) -> list[DataFrame]:
        """Randomly split this DataFrame into multiple DataFrames by weight.

        Args:
            weights: List of weights (will be normalized to sum to 1.0).
            seed: Random seed for reproducibility.
        """
        import random

        total = sum(weights)
        normalized = [w / total for w in weights]

        rows = self.collect()
        rng = random.Random(seed)
        rng.shuffle(rows)

        results = []
        start = 0
        for i, frac in enumerate(normalized):
            if i == len(normalized) - 1:
                end = len(rows)
            else:
                end = start + round(frac * len(rows))
            subset = rows[start:end]
            start = end

            if subset:
                import pyarrow as pa

                columns = {}
                for col_name in self.columns:
                    columns[col_name] = [r[col_name] for r in subset]
                arrow_table = pa.table(columns)
            else:
                arrow_table = self._df.to_arrow_table().slice(0, 0)

            temp_name = f"_split_{id(self)}_{i}"
            if self._session:
                self._session._ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
                results.append(DataFrame(self._session._ctx.table(temp_name), self._session))
            else:
                from datafusion import SessionContext

                ctx = SessionContext()
                ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
                results.append(DataFrame(ctx.table(temp_name), None))
        return results

    def toJSON(self) -> DataFrame:
        """Returns a DataFrame of JSON string representations of each row."""
        import json
        import pyarrow as pa

        rows = self.collect()
        json_strings = [json.dumps(r.asDict()) for r in rows]
        arrow_table = pa.table({"value": json_strings})

        if self._session:
            temp_name = f"_json_{id(self)}"
            self._session._ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
            return DataFrame(self._session._ctx.table(temp_name), self._session)
        else:
            from datafusion import SessionContext

            ctx = SessionContext()
            temp_name = f"_json_{id(self)}"
            ctx.register_record_batches(temp_name, [arrow_table.to_batches()])
            return DataFrame(ctx.table(temp_name), None)

    def foreach(self, f) -> None:
        """Applies a function to each row."""
        for row in self.collect():
            f(row)

    def foreachPartition(self, f) -> None:
        """Applies a function to each partition (single partition in single-node mode)."""
        f(iter(self.collect()))

    def describe(self, *cols: str) -> DataFrame:
        """Computes basic statistics (count, mean, stddev, min, max).

        Args:
            cols: Column names. If empty, uses all columns.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        target_cols = list(cols) if cols else self.columns
        arrow_table = self._df.to_arrow_table()

        stats = ["count", "mean", "stddev", "min", "max"]
        result_data: dict = {"summary": stats}
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
        # Reuse session context if available; otherwise create a throwaway context.
        # Both paths return a DataFrame wrapping a one-time context — callers are
        # expected to collect() or show() the result, not re-register tables against it.
        ctx = self._session._ctx if self._session is not None else None
        if ctx is None:
            from datafusion import SessionContext as _Ctx
            ctx = _Ctx()
        temp_name = f"_describe_{id(self)}"
        ctx.register_record_batches(temp_name, [result_arrow.to_batches()])
        return DataFrame(ctx.table(temp_name), self._session)

    def summary(self, *statistics: str) -> DataFrame:
        """Computes specified statistics. Defaults to describe stats."""
        return self.describe()

    # --- Temp view registration ---

    def createTempView(self, name: str) -> None:
        """Creates a local temporary view with this DataFrame.

        Raises RuntimeError if a view with the same name already exists.
        """
        if self._session is None:
            raise RuntimeError("createTempView requires an active session")
        try:
            self._session._ctx.table(name)
            raise RuntimeError(f"Temporary view '{name}' already exists")
        except RuntimeError:
            raise
        except Exception:
            pass
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

    # --- Aggregation ---

    def groupBy(self, *cols) -> "GroupedData":
        """Groups the DataFrame using the specified columns."""
        from iceberg_spark.column import Column
        from iceberg_spark.grouped_data import GroupedData

        group_cols = []
        for c in cols:
            if isinstance(c, str):
                group_cols.append(Column(c))
            elif isinstance(c, Column):
                group_cols.append(c)
        return GroupedData(self, group_cols)

    def groupby(self, *cols) -> "GroupedData":
        """Alias for groupBy."""
        return self.groupBy(*cols)

    def cube(self, *cols) -> "GroupedData":
        """Creates a multi-dimensional cube for the given columns.

        Returns a GroupedData whose ``agg()`` produces rows for every
        combination of grouping columns (including subtotals and a grand
        total).  Missing grouping column values are represented as NULL.
        """
        from iceberg_spark.column import Column
        from iceberg_spark.grouped_data import CubeGroupedData

        group_cols = []
        for c in cols:
            if isinstance(c, str):
                group_cols.append(Column(c))
            elif isinstance(c, Column):
                group_cols.append(c)
        return CubeGroupedData(self, group_cols, mode="cube")

    def rollup(self, *cols) -> "GroupedData":
        """Creates a hierarchical rollup for the given columns.

        Returns a GroupedData whose ``agg()`` produces rows for each
        prefix of the grouping columns (including a grand total).
        Missing grouping column values are represented as NULL.
        """
        from iceberg_spark.column import Column
        from iceberg_spark.grouped_data import CubeGroupedData

        group_cols = []
        for c in cols:
            if isinstance(c, str):
                group_cols.append(Column(c))
            elif isinstance(c, Column):
                group_cols.append(c)
        return CubeGroupedData(self, group_cols, mode="rollup")

    def agg(self, *exprs) -> DataFrame:
        """Aggregate without groups (across entire DataFrame)."""
        from iceberg_spark.column import Column
        from iceberg_spark.grouped_data import GroupedData

        return GroupedData(self, []).agg(*exprs)

    # --- Explain ---

    def explain(self, extended: bool = False, mode: str | None = None) -> None:
        """Prints the physical plan."""
        plan = self._df.explain()
        # DataFusion explain returns a string, just print it
        print(plan)

    def writeTo(self, table: str) -> "DataFrameWriterV2":
        """Returns a DataFrameWriterV2 for writing to a table."""
        from iceberg_spark.writer_v2 import DataFrameWriterV2
        return DataFrameWriterV2(self, table)

    def inputFiles(self) -> list[str]:
        """Returns the list of files that were read to compute this DataFrame.

        Stub — returns empty list in single-node mode.
        """
        return []

    @property
    def storageLevel(self):
        """Returns the storage level. Stub in single-node mode."""
        return StorageLevel("MEMORY_AND_DISK")

    def observe(self, observation, *exprs) -> "DataFrame":
        """Defines metrics to observe. No-op in single-node mode."""
        return self

    def sameSemantics(self, other: "DataFrame") -> bool:
        """Returns True if the two DataFrames have the same schema."""
        return self.columns == other.columns and self.dtypes == other.dtypes

    def semanticHash(self) -> int:
        """Returns a hash based on the schema."""
        return hash(str(self.dtypes))

    def mapInPandas(self, func, schema) -> "DataFrame":
        """Apply a function that takes and returns an iterator of Pandas DataFrames.

        In single-node mode the entire DataFrame is treated as one partition.

        Args:
            func: Takes an iterator of pandas.DataFrame, returns an iterator of pandas.DataFrame.
            schema: Output schema (PyArrow Schema or StructType).
        """
        import pandas as pd
        import pyarrow as pa
        from datafusion import SessionContext

        pandas_df = self._df.to_pandas()
        result_iter = func(iter([pandas_df]))
        result_parts = list(result_iter) if result_iter is not None else []

        arrow_schema = self._resolve_schema(schema)
        if result_parts:
            result_df = pd.concat(result_parts, ignore_index=True)
            if arrow_schema is not None:
                arrow_table = pa.Table.from_pandas(result_df, schema=arrow_schema)
            else:
                arrow_table = pa.Table.from_pandas(result_df)
        else:
            # Empty result — build empty table with the correct schema
            if arrow_schema is None:
                arrow_schema = self._df.to_arrow_table().schema
            empty_arrays = [pa.array([], type=f.type) for f in arrow_schema]
            arrow_table = pa.table(
                {f.name: arr for f, arr in zip(arrow_schema, empty_arrays)},
                schema=arrow_schema,
            )

        ctx = SessionContext()
        batches = arrow_table.to_batches()
        if not batches:
            empty_arrays = [pa.array([], type=f.type) for f in arrow_table.schema]
            batches = [pa.record_batch(empty_arrays, schema=arrow_table.schema)]
        ctx.register_record_batches("_map_result", [batches])
        return DataFrame(ctx.table("_map_result"), self._session)

    def mapInArrow(self, func, schema) -> "DataFrame":
        """Apply a function that takes and returns an iterator of Arrow RecordBatches.

        In single-node mode the entire DataFrame is treated as one partition.

        Args:
            func: Takes an iterator of pyarrow.RecordBatch, returns an iterator of pyarrow.RecordBatch.
            schema: Output schema (PyArrow Schema or StructType).
        """
        import pyarrow as pa
        from datafusion import SessionContext

        arrow_table = self._df.to_arrow_table()
        batches = arrow_table.to_batches()
        if not batches:
            empty_arrays = [pa.array([], type=f.type) for f in arrow_table.schema]
            batches = [pa.record_batch(empty_arrays, schema=arrow_table.schema)]

        result_iter = func(iter(batches))
        result_batches = list(result_iter) if result_iter is not None else []

        arrow_schema = self._resolve_schema(schema)
        if arrow_schema is None:
            if result_batches:
                arrow_schema = result_batches[0].schema
            else:
                arrow_schema = arrow_table.schema

        result_table = pa.Table.from_batches(result_batches, schema=arrow_schema)
        ctx = SessionContext()
        out_batches = result_table.to_batches()
        if not out_batches:
            empty_arrays = [pa.array([], type=f.type) for f in arrow_schema]
            out_batches = [pa.record_batch(empty_arrays, schema=arrow_schema)]
        ctx.register_record_batches("_map_result", [out_batches])
        return DataFrame(ctx.table("_map_result"), self._session)

    @staticmethod
    def _resolve_schema(schema):
        """Convert a schema argument to a PyArrow Schema, or return None."""
        import pyarrow as pa

        from iceberg_spark.types import StructType as SparkStructType

        if isinstance(schema, pa.Schema):
            return schema
        if isinstance(schema, SparkStructType):
            arrow_type = schema.to_arrow()
            # StructType.to_arrow() returns pa.StructType, convert to pa.Schema
            if isinstance(arrow_type, pa.StructType):
                return pa.schema(list(arrow_type))
            return arrow_type
        return None

    # --- Write ---

    @property
    def write(self):
        """Returns a DataFrameWriter for saving the contents."""
        from iceberg_spark.writer import DataFrameWriter

        return DataFrameWriter(self)

    # --- Cache (no-op for single-node) ---

    def cache(self) -> DataFrame:
        """No-op in single-node DataFusion."""
        return self

    def persist(self) -> DataFrame:
        """No-op in single-node DataFusion."""
        return self

    def unpersist(self) -> DataFrame:
        """No-op in single-node DataFusion."""
        return self

    # --- Repartition (no-op for single-node) ---

    def repartition(self, numPartitions: int, *cols) -> DataFrame:
        """No-op in single-node DataFusion."""
        return self

    def coalesce(self, numPartitions: int) -> DataFrame:
        """No-op in single-node DataFusion."""
        return self

    def toDF(self, *col_names: str) -> DataFrame:
        """Returns a new DataFrame with columns renamed to the given names."""
        current = self.columns
        if len(col_names) != len(current):
            raise ValueError(
                f"Number of column names ({len(col_names)}) must match "
                f"number of columns ({len(current)})"
            )
        result = self
        for old, new in zip(current, col_names):
            if old != new:
                result = result.withColumnRenamed(old, new)
        return result

    def sample(
        self,
        withReplacement: bool = False,
        fraction: float | None = None,
        seed: int | None = None,
    ) -> DataFrame:
        """Returns an approximate sampled subset via RANDOM() < fraction filtering."""
        if fraction is None:
            raise ValueError("fraction is required for sample()")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0.0 and 1.0")
        import datafusion.functions as F
        return DataFrame(self._df.filter(F.random() < fraction), self._session)

    def hint(self, name: str, *parameters) -> DataFrame:
        """No-op compatibility hint."""
        return self

    def transform(self, func) -> DataFrame:
        """Applies a function to this DataFrame and returns the result."""
        return func(self)

    # --- unpivot / melt ---

    def unpivot(
        self,
        ids: list[str],
        values: list[str],
        variableColumnName: str = "variable",
        valueColumnName: str = "value",
    ) -> DataFrame:
        """Unpivot (melt) value columns into rows.

        For each value column, produces a row with the id columns, the column
        name as *variableColumnName*, and the column value as *valueColumnName*.

        Args:
            ids: Column names to keep as identifiers.
            values: Column names to unpivot into rows.
            variableColumnName: Name for the column containing original column names.
            valueColumnName: Name for the column containing the values.
        """
        if self._session is None:
            raise RuntimeError("unpivot requires an active session")
        if not values:
            raise ValueError("values must contain at least one column")

        import uuid

        def _quote(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        temp_name = f"_unpivot_{uuid.uuid4().hex[:8]}"
        self._session._ctx.register_table(temp_name, self._df)
        try:
            id_cols = ", ".join(_quote(c) for c in ids)
            parts = []
            for v in values:
                escaped = v.replace("'", "''")
                sel = (
                    f"SELECT {id_cols}, '{escaped}' AS {_quote(variableColumnName)}, "
                    f"CAST({_quote(v)} AS DOUBLE) AS {_quote(valueColumnName)} "
                    f"FROM {temp_name}"
                )
                parts.append(sel)
            sql = " UNION ALL ".join(parts)
            result = self._session._ctx.sql(sql)
            return DataFrame(result, self._session)
        finally:
            self._session._ctx.deregister_table(temp_name)

    def melt(
        self,
        ids: list[str],
        values: list[str],
        variableColumnName: str = "variable",
        valueColumnName: str = "value",
    ) -> DataFrame:
        """Alias for :meth:`unpivot`."""
        return self.unpivot(ids, values, variableColumnName, valueColumnName)

    # --- approxQuantile convenience (delegates to stat) ---

    def approxQuantile(
        self,
        col: str,
        probabilities: list[float],
        relativeError: float = 0.0,
    ) -> list[float | None]:
        """Convenience wrapper — delegates to ``self.stat.approxQuantile``."""
        return self.stat.approxQuantile(col, probabilities, relativeError)

    # --- Convenience delegates to .na / .stat ---

    def corr(self, col1: str, col2: str, method: str = "pearson") -> float:
        """Returns the Pearson correlation coefficient between two columns."""
        return self.stat.corr(col1, col2)

    def cov(self, col1: str, col2: str) -> float:
        """Returns the sample covariance between two columns."""
        return self.stat.cov(col1, col2)

    def freqItems(self, cols: list[str], support: float = 0.01) -> "DataFrame":
        """Returns frequent items for the given columns."""
        return self.stat.freqItems(cols, support)

    def dropna(self, how: str = "any", thresh: int | None = None, subset: list[str] | None = None) -> "DataFrame":
        """Returns a new DataFrame omitting rows with null values."""
        return self.na.drop(how=how, thresh=thresh, subset=subset)

    def fillna(self, value, subset: list[str] | None = None) -> "DataFrame":
        """Replaces null values with the given value."""
        return self.na.fill(value, subset=subset)

    def replace(self, to_replace, value=None, subset: list[str] | None = None) -> "DataFrame":
        """Replaces values matching ``to_replace`` with ``value``."""
        return self.na.replace(to_replace, value, subset=subset)

    def sortWithinPartitions(self, *cols, **kwargs) -> "DataFrame":
        """Sorts within each partition. Equivalent to ``sort()`` in single-node mode."""
        return self.sort(*cols, **kwargs)

    # --- Simple stubs (correct for single-node) ---

    def isLocal(self) -> bool:
        """Returns True — always local in single-node mode."""
        return True

    def isStreaming(self) -> bool:
        """Returns False — no streaming support."""
        return False

    def checkpoint(self, eager: bool = True) -> DataFrame:
        """No-op checkpoint in single-node mode."""
        return self

    def localCheckpoint(self, eager: bool = True) -> DataFrame:
        """No-op local checkpoint in single-node mode."""
        return self

    def withWatermark(self, eventTime: str, delayThreshold: str) -> DataFrame:
        """No-op watermark in single-node mode."""
        return self

    def colRegex(self, colName: str) -> DataFrame:
        """Selects columns matching a regex pattern.

        Args:
            colName: Regex pattern (backtick-delimited patterns have backticks stripped).
        """
        import re

        pattern = re.compile(colName.strip("`"))
        matching = [c for c in self.columns if pattern.search(c)]
        return self.select(*matching) if matching else self.select(*self.columns[:0])

    @property
    def na(self) -> DataFrameNaFunctions:
        """Returns a DataFrameNaFunctions for handling missing values."""
        return DataFrameNaFunctions(self)

    @property
    def stat(self) -> DataFrameStatFunctions:
        """Returns a DataFrameStatFunctions for statistics."""
        return DataFrameStatFunctions(self)


class DataFrameNaFunctions:
    """Methods for handling missing (null/NaN) values in a DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def drop(
        self,
        how: str = "any",
        thresh: int | None = None,
        subset: list[str] | None = None,
    ) -> DataFrame:
        """Returns a new DataFrame with rows containing nulls removed.

        Args:
            how: 'any' (drop if any null) or 'all' (drop only if all null).
            thresh: Minimum number of non-null values required to keep a row.
            subset: Column names to check. Defaults to all columns.
        """
        from iceberg_spark.column import Column

        cols = subset or self._df.columns

        if thresh is not None:
            import datafusion.functions as F
            # Build a sum of not-null flags and filter >= thresh
            not_null_exprs = [
                F.when(Column(c).isNotNull().expr, 1).otherwise(0)
                for c in cols
            ]
            # Sum them up via repeated addition
            total = not_null_exprs[0]
            for e in not_null_exprs[1:]:
                total = total + e
            return DataFrame(self._df._df.filter(total >= thresh), self._df._session)

        if how == "any":
            condition = None
            for c in cols:
                cond = Column(c).isNotNull().expr
                condition = cond if condition is None else condition & cond
        else:  # all
            condition = None
            for c in cols:
                cond = Column(c).isNotNull().expr
                condition = cond if condition is None else condition | cond

        if condition is None:
            return self._df
        return DataFrame(self._df._df.filter(condition), self._df._session)

    def fill(self, value, subset: list[str] | None = None) -> DataFrame:
        """Returns a new DataFrame with nulls filled.

        Args:
            value: Fill value (scalar) or dict mapping column → fill value.
            subset: Columns to fill (only when value is a scalar).
        """
        from iceberg_spark.column import Column
        from iceberg_spark.functions import coalesce, lit

        if isinstance(value, dict):
            result = self._df
            for col_name, fill_val in value.items():
                result = result.withColumn(
                    col_name, coalesce(Column(col_name), lit(fill_val))
                )
            return result

        cols = subset or self._df.columns
        result = self._df
        for c in cols:
            result = result.withColumn(c, coalesce(Column(c), lit(value)))
        return result

    def replace(
        self,
        to_replace,
        value=None,
        subset: list[str] | None = None,
    ) -> DataFrame:
        """Returns a new DataFrame replacing specific values.

        Args:
            to_replace: Value or dict/list of values to replace.
            value: Replacement value or list of replacements.
            subset: Columns to apply replacement to.
        """
        from iceberg_spark.column import Column
        from iceberg_spark.functions import lit, when

        if isinstance(to_replace, dict):
            pairs = list(to_replace.items())
        elif isinstance(to_replace, list):
            value = value if isinstance(value, list) else [value] * len(to_replace)
            pairs = list(zip(to_replace, value))
        else:
            pairs = [(to_replace, value)]

        cols = subset or self._df.columns
        result = self._df

        for col_name in cols:
            col_obj = Column(col_name)
            expr = col_obj
            for old_val, new_val in pairs:
                expr = when(col_obj == lit(old_val), lit(new_val)).otherwise(expr)
            result = result.withColumn(col_name, expr)

        return result


class DataFrameStatFunctions:
    """Statistical methods for a DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def corr(self, col1: str, col2: str, method: str = "pearson") -> float:
        """Computes Pearson correlation between two columns."""
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

    def cov(self, col1: str, col2: str) -> float:
        """Computes sample covariance between two columns."""
        import datafusion.functions as F
        from iceberg_spark.column import Column

        result = DataFrame(
            self._df._df.aggregate(
                [],
                [F.covar_samp(Column(col1).expr, Column(col2).expr).alias("cov")],
            ),
            self._df._session,
        )
        rows = result.collect()
        return float(rows[0]["cov"]) if rows else None

    def crosstab(self, col1: str, col2: str) -> DataFrame:
        """Computes a cross-tabulation (pivot) of two columns."""
        from iceberg_spark.functions import lit, sum as _sum, when
        from iceberg_spark.column import Column

        col2_vals = [r[col2] for r in self._df.select(col2).distinct().collect()]
        agg_exprs = [
            _sum(when(Column(col2) == lit(v), lit(1)).otherwise(lit(0))).alias(str(v))
            for v in col2_vals
        ]
        return self._df.groupBy(col1).agg(*agg_exprs)

    def approxQuantile(
        self,
        col: str,
        probabilities: list[float],
        relativeError: float = 0.0,
    ) -> list[float | None]:
        """Returns approximate quantiles for a column.

        Args:
            col: Column name.
            probabilities: List of quantile probabilities in [0, 1].
            relativeError: Ignored — DataFusion always uses approximate algorithm.

        Returns:
            List of approximate quantile values, one per probability.
            Elements are None when the column is empty.
        """
        import datafusion.functions as F
        from iceberg_spark.column import Column

        results: list[float | None] = []
        for p in probabilities:
            agg_result = DataFrame(
                self._df._df.aggregate(
                    [],
                    [F.approx_percentile_cont(Column(col).expr, p).alias("q")],
                ),
                self._df._session,
            )
            rows = agg_result.collect()
            # aggregate() always returns one row; q is None when table is empty
            val = rows[0]["q"] if rows else None
            results.append(float(val) if val is not None else None)
        return results

    def freqItems(self, cols: list[str], support: float = 0.01) -> DataFrame:
        """Returns frequent items for each column.

        An item is "frequent" if it appears in at least `support` fraction of rows.

        Args:
            cols: Column names to compute frequent items for.
            support: Minimum fraction of rows an item must appear in (0.0–1.0).

        Returns:
            DataFrame with one column per input column named ``<col>_freqItems``.

        Raises:
            RuntimeError: If no active session is set on this DataFrame.
        """
        if self._df._session is None:
            raise RuntimeError("freqItems requires an active session")

        import math
        import pyarrow as pa
        from iceberg_spark.functions import col as _col, count as _count, lit as _lit

        total = self._df.count()
        min_count = max(1, math.ceil(total * support))

        result_data: dict[str, list] = {}
        for c in cols:
            freq = (
                self._df.groupBy(c)
                .agg(_count("*").alias("cnt"))
                .filter(_col("cnt") >= _lit(min_count))
            )
            result_data[f"{c}_freqItems"] = [r[c] for r in freq.collect()]

        # Pad lists to equal length so pa.table() accepts them
        max_len = max((len(v) for v in result_data.values()), default=0)
        for k in result_data:
            result_data[k] = result_data[k] + [None] * (max_len - len(result_data[k]))

        # Build with explicit string schema to avoid null-type inference for
        # empty lists; use record_batch (not table.to_batches()) so DataFusion
        # always receives exactly one batch per partition, even for 0-row results.
        # (register_record_batches panics if the inner list is empty.)
        fields = [pa.field(k, pa.string()) for k in result_data]
        arrow_schema = pa.schema(fields)
        batch = pa.record_batch(result_data, schema=arrow_schema)

        temp_name = f"_freq_{id(self)}"
        self._df._session._ctx.register_record_batches(temp_name, [[batch]])
        return DataFrame(self._df._session._ctx.table(temp_name), self._df._session)


class StorageLevel:
    """Stub StorageLevel for PySpark compatibility."""

    def __init__(self, description: str = "MEMORY_AND_DISK"):
        self._description = description

    def __repr__(self) -> str:
        return f"StorageLevel({self._description})"

    def __str__(self) -> str:
        return self._description

    def __eq__(self, other) -> bool:
        if isinstance(other, StorageLevel):
            return self._description == other._description
        return False
