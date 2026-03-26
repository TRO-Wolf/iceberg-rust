"""PySpark-compatible GroupedData for aggregation operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iceberg_spark.column import Column
    from iceberg_spark.dataframe import DataFrame


class GroupedData:
    """A set of methods for aggregations on a DataFrame, created by DataFrame.groupBy()."""

    def __init__(self, df: DataFrame, group_cols: list[Column]):
        self._df = df
        self._group_cols = group_cols

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

    def pivot(self, pivot_col: str, values: list | None = None) -> "PivotedGroupedData":
        """Pivots a column and returns a PivotedGroupedData for aggregation.

        Args:
            pivot_col: Column to pivot on.
            values: Optional list of values to pivot. If None, uses all distinct values.
        """
        return PivotedGroupedData(self._df, self._group_cols, pivot_col, values)

    def count(self) -> DataFrame:
        """Counts the number of rows for each group."""
        from iceberg_spark.functions import count

        return self.agg(count("*").alias("count"))

    def sum(self, *cols: str) -> DataFrame:
        """Computes the sum for each numeric column for each group."""
        from iceberg_spark.functions import sum as _sum

        return self.agg(*[_sum(c).alias(f"sum({c})") for c in cols])

    def avg(self, *cols: str) -> DataFrame:
        """Computes the average for each numeric column for each group."""
        from iceberg_spark.functions import avg as _avg

        return self.agg(*[_avg(c).alias(f"avg({c})") for c in cols])

    def mean(self, *cols: str) -> DataFrame:
        """Computes the average for each numeric column for each group (alias for ``avg``)."""
        return self.avg(*cols)

    def min(self, *cols: str) -> DataFrame:
        """Computes the minimum value for each column for each group."""
        from iceberg_spark.functions import min as _min

        return self.agg(*[_min(c).alias(f"min({c})") for c in cols])

    def max(self, *cols: str) -> DataFrame:
        """Computes the maximum value for each column for each group."""
        from iceberg_spark.functions import max as _max

        return self.agg(*[_max(c).alias(f"max({c})") for c in cols])


class PivotedGroupedData:
    """GroupedData with a pivot operation pending."""

    def __init__(self, df, group_cols, pivot_col, values):
        self._df = df
        self._group_cols = group_cols
        self._pivot_col = pivot_col
        self._values = values

    def agg(self, *exprs) -> "DataFrame":
        """Aggregate with pivot — creates one column per pivot value.

        Uses SQL-based conditional aggregation (SUM(CASE WHEN ...)) because
        DataFusion's Expr API cannot nest column references inside CASE WHEN
        inside aggregate functions.
        """
        import uuid

        from iceberg_spark.dataframe import DataFrame

        # Get pivot values (excluding None/null entries, which can't be column names)
        if self._values is None:
            raw = set(
                r[self._pivot_col]
                for r in self._df.select(self._pivot_col).distinct().collect()
            )
            vals = sorted(v for v in raw if v is not None)
        else:
            vals = self._values

        # Build SQL with CASE WHEN for each pivot value + aggregate combo
        group_names = [c._expr.schema_name() for c in self._group_cols]
        group_sql = ", ".join(f'"{g}"' for g in group_names)

        pivot_col_q = f'"{self._pivot_col}"'

        import re

        agg_parts = []
        for expr in exprs:
            # Extract aggregate function and column from the Expr string.
            # schema_name() returns "sum(revenue)" for unaliased or "total"
            # for aliased exprs.  str(expr) returns "Expr(sum(revenue) AS total)".
            # We try schema_name first, then fall back to parsing str(expr).
            schema = expr._expr.schema_name()
            m = re.match(r"(\w+)\((.+)\)", schema)
            if m:
                agg_func = m.group(1).upper()
                agg_col = m.group(2)
            else:
                # Aliased — parse the full Expr string to get the function
                expr_str = str(expr._expr)
                m2 = re.match(r"Expr\((\w+)\((.+?)\)\s+AS\s+", expr_str)
                if m2:
                    agg_func = m2.group(1).upper()
                    agg_col = m2.group(2)
                else:
                    agg_func = "SUM"
                    agg_col = schema

            for v in vals:
                if isinstance(v, str):
                    val_sql = f"'{v}'"
                else:
                    val_sql = str(v)
                col_alias = str(v)
                agg_parts.append(
                    f'{agg_func}(CASE WHEN {pivot_col_q} = {val_sql} '
                    f'THEN "{agg_col}" ELSE NULL END) AS "{col_alias}"'
                )

        agg_sql = ", ".join(agg_parts)

        # Register source as temp table and run SQL
        temp_name = f"_pivot_{uuid.uuid4().hex[:8]}"

        # Use session context if available, otherwise create a temporary one
        if self._df._session:
            ctx = self._df._session._ctx
        else:
            from datafusion import SessionContext
            ctx = SessionContext()

        ctx.register_table(temp_name, self._df._df)
        try:
            sql = f"SELECT {group_sql}, {agg_sql} FROM {temp_name} GROUP BY {group_sql}"
            result = ctx.sql(sql)
            return DataFrame(result, self._df._session)
        finally:
            ctx.deregister_table(temp_name)


class CubeGroupedData:
    """GroupedData for cube() or rollup() operations.

    Produces subtotal and grand-total rows by running the aggregation for
    every combination (cube) or prefix (rollup) of grouping columns and
    combining the results with UNION ALL.
    """

    def __init__(self, df, group_cols, mode: str = "cube"):
        self._df = df
        self._group_cols = group_cols
        self._mode = mode  # "cube" or "rollup"

    def agg(self, *exprs) -> "DataFrame":
        """Compute aggregates for each sub-grouping and combine with UNION ALL."""
        import uuid
        from itertools import combinations

        from iceberg_spark.dataframe import DataFrame

        import re as _re

        group_names = [c._expr.schema_name() for c in self._group_cols]

        # Build agg SQL fragments.  We extract the function name and column
        # from schema_name() or the full Expr string, then reconstruct as
        # valid SQL.  Needed because str(Expr) uses DataFusion internals
        # like "Int64(1)" that aren't valid SQL.
        agg_schemas = []
        for expr in exprs:
            schema = expr._expr.schema_name()
            m = _re.match(r"(\w+)\((.+)\)", schema)
            if m:
                agg_func = m.group(1).upper()
                agg_col = m.group(2)
            else:
                # Aliased — parse full Expr string
                expr_str = str(expr._expr)
                m2 = _re.match(r"Expr\((\w+)\((.+?)\)\s+AS\s+", expr_str)
                if m2:
                    agg_func = m2.group(1).upper()
                    agg_col = m2.group(2)
                else:
                    agg_func = "SUM"
                    agg_col = schema

            # Reconstruct valid SQL — handle special cases
            if agg_func == "COUNT" and ("Int64" in agg_col or agg_col == "*"):
                agg_sql_frag = "COUNT(*)"
            else:
                agg_sql_frag = f'{agg_func}("{agg_col}")'

            alias = schema if not m else None  # use schema_name as alias only for aliased exprs
            if alias and alias != agg_sql_frag:
                agg_schemas.append(f'{agg_sql_frag} AS "{alias}"')
            else:
                agg_schemas.append(agg_sql_frag)

        # Build the list of grouping-column subsets
        if self._mode == "cube":
            # All subsets from full set down to empty
            subsets = []
            for r in range(len(group_names), -1, -1):
                for combo in combinations(group_names, r):
                    subsets.append(list(combo))
        else:
            # Rollup: prefixes only — (a,b,c), (a,b), (a), ()
            subsets = [group_names[: i] for i in range(len(group_names), -1, -1)]

        # Build SQL: one SELECT per subset, UNION ALL
        agg_sql = ", ".join(agg_schemas)

        temp_name = f"_{self._mode}_{uuid.uuid4().hex[:8]}"
        if self._df._session:
            ctx = self._df._session._ctx
        else:
            from datafusion import SessionContext
            ctx = SessionContext()
        ctx.register_table(temp_name, self._df._df)
        try:
            parts = []
            all_cols_q = ", ".join(f'"{g}"' for g in group_names)
            for subset in subsets:
                select_cols = []
                for g in group_names:
                    if g in subset:
                        select_cols.append(f'"{g}"')
                    else:
                        select_cols.append(f"NULL AS \"{g}\"")
                select_str = ", ".join(select_cols)
                if subset:
                    group_by = "GROUP BY " + ", ".join(f'"{g}"' for g in subset)
                else:
                    group_by = ""
                parts.append(
                    f"SELECT {select_str}, {agg_sql} FROM {temp_name} {group_by}"
                )

            sql = " UNION ALL ".join(parts)
            result = ctx.sql(sql)
            return DataFrame(result, self._df._session)
        finally:
            ctx.deregister_table(temp_name)
