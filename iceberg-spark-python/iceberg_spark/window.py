"""PySpark-compatible Window specification for window functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iceberg_spark.column import Column


class WindowSpec:
    """A window specification for window functions."""

    def __init__(self, partition_cols=None, order_cols=None, frame=None):
        self._partition_cols = partition_cols or []
        self._order_cols = order_cols or []
        self._frame = frame  # ("rows"|"range", start, end) or None

    def partitionBy(self, *cols) -> WindowSpec:
        """Defines the partitioning columns for the window."""
        from iceberg_spark.column import Column

        partition_cols = []
        for c in cols:
            partition_cols.append(Column(c) if isinstance(c, str) else c)
        return WindowSpec(partition_cols, self._order_cols, self._frame)

    def orderBy(self, *cols) -> WindowSpec:
        """Defines the ordering columns for the window."""
        from iceberg_spark.column import Column

        order_cols = []
        for c in cols:
            order_cols.append(Column(c) if isinstance(c, str) else c)
        return WindowSpec(self._partition_cols, order_cols, self._frame)

    def rowsBetween(self, start: int, end: int) -> WindowSpec:
        """Defines the window frame boundaries based on row offsets."""
        return WindowSpec(self._partition_cols, self._order_cols, ("rows", start, end))

    def rangeBetween(self, start: int, end: int) -> WindowSpec:
        """Defines the window frame boundaries based on value range."""
        return WindowSpec(self._partition_cols, self._order_cols, ("range", start, end))

    def _apply(self, col: Column) -> Column:
        """Wrap *col* (a window function expr) with this WindowSpec's OVER clause.

        Uses Expr.over(Window) for aggregate functions (first_value, last_value, etc.)
        and the ExprFuncBuilder pattern for window functions (row_number, rank, etc.).
        """
        from datafusion.expr import SortExpr, Window as DFWindow
        from iceberg_spark.column import Column

        partition_exprs = [c.expr for c in self._partition_cols]

        # Convert order columns to SortExpr objects
        order_sort_exprs = []
        for c in self._order_cols:
            if isinstance(c.expr, SortExpr):
                order_sort_exprs.append(c.expr)
            else:
                # Default: ascending, nulls first
                order_sort_exprs.append(c.expr.sort())

        # Try Expr.over(Window) first — works for aggregate exprs like first_value
        try:
            wf = self._build_window_frame()
            df_window = DFWindow(
                partition_by=partition_exprs or None,
                order_by=order_sort_exprs or None,
                window_frame=wf,
            )
            return Column(col._expr.over(df_window))
        except Exception:
            pass

        # Fallback: ExprFuncBuilder pattern — works for window exprs like row_number
        try:
            builder = col._expr.partition_by(*partition_exprs)
            if order_sort_exprs:
                builder = builder.order_by(*order_sort_exprs)
            wf = self._build_window_frame()
            if wf is not None:
                builder = builder.window_frame(wf)
            return Column(builder.build())
        except Exception:
            # Last resort: return the raw expression unchanged
            return col

    def _build_window_frame(self):
        """Build a DataFusion WindowFrame from the frame spec, or None for default."""
        if self._frame is None:
            return None
        try:
            from datafusion import WindowFrame
            frame_type, start, end = self._frame
            # Convert sentinel values to None (unbounded)
            if start == Window.unboundedPreceding:
                start = None
            if end == Window.unboundedFollowing:
                end = None
            return WindowFrame(frame_type, start, end)
        except Exception:
            return None


class Window:
    """Utility for defining window specifications."""

    unboundedPreceding: int = -(2**63)
    unboundedFollowing: int = 2**63 - 1
    currentRow: int = 0

    @staticmethod
    def partitionBy(*cols) -> WindowSpec:
        """Creates a WindowSpec with the given partitioning columns."""
        return WindowSpec().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols) -> WindowSpec:
        """Creates a WindowSpec with the given ordering columns."""
        return WindowSpec().orderBy(*cols)

    @staticmethod
    def rowsBetween(start: int, end: int) -> WindowSpec:
        """Creates a WindowSpec with a row-based window frame."""
        return WindowSpec().rowsBetween(start, end)

    @staticmethod
    def rangeBetween(start: int, end: int) -> WindowSpec:
        """Creates a WindowSpec with a value-range-based window frame."""
        return WindowSpec().rangeBetween(start, end)
