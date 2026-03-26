"""PySpark-compatible Column expression wrapper over DataFusion expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datafusion import lit as _lit

if TYPE_CHECKING:
    from datafusion import Expr


class Column:
    """A column expression, compatible with PySpark's Column interface.

    Wraps a DataFusion Expr and provides PySpark-compatible operator overloads
    and methods.
    """

    def __init__(self, expr):
        """Create a Column from a DataFusion Expr or column name string."""
        if isinstance(expr, str):
            import datafusion.functions as F

            self._expr = F.col(expr)
            self._name = expr
        else:
            self._expr = expr
            self._name = str(expr)

    @property
    def expr(self) -> Expr:
        """Returns the underlying DataFusion expression."""
        return self._expr

    # --- Comparison operators ---

    def __eq__(self, other):
        return Column(self._expr == _unwrap(other))

    def __ne__(self, other):
        return Column(self._expr != _unwrap(other))

    def __lt__(self, other):
        return Column(self._expr < _unwrap(other))

    def __le__(self, other):
        return Column(self._expr <= _unwrap(other))

    def __gt__(self, other):
        return Column(self._expr > _unwrap(other))

    def __ge__(self, other):
        return Column(self._expr >= _unwrap(other))

    # --- Boolean operators ---

    def __and__(self, other):
        return Column(self._expr & _unwrap(other))

    def __or__(self, other):
        return Column(self._expr | _unwrap(other))

    def __invert__(self):
        return Column(~self._expr)

    # --- Arithmetic operators ---

    def __add__(self, other):
        return Column(self._expr + _unwrap(other))

    def __sub__(self, other):
        return Column(self._expr - _unwrap(other))

    def __mul__(self, other):
        return Column(self._expr * _unwrap(other))

    def __truediv__(self, other):
        return Column(self._expr / _unwrap(other))

    def __mod__(self, other):
        return Column(self._expr % _unwrap(other))

    def __neg__(self):
        return Column(-self._expr)

    # --- PySpark Column methods ---

    def alias(self, name: str) -> Column:
        """Returns this column renamed to *name*."""
        result = Column(self._expr.alias(name))
        # Preserve explode tag through alias
        if getattr(self, "_explode", False):
            result._explode = True
        return result

    def name(self, name: str) -> Column:
        """Returns this column renamed to *name* (alias for ``alias()``)."""
        return self.alias(name)

    def cast(self, dataType) -> Column:
        """Casts the column to the specified data type."""
        from iceberg_spark.types import DataType

        if isinstance(dataType, DataType):
            arrow_type = dataType.to_arrow()
        elif isinstance(dataType, str):
            from iceberg_spark._internal.type_mapping import spark_type_from_name

            arrow_type = spark_type_from_name(dataType).to_arrow()
        else:
            arrow_type = dataType
        return Column(self._expr.cast(arrow_type))

    def isNull(self) -> Column:
        """Returns True if the column value is null."""
        return Column(self._expr.is_null())

    def isNotNull(self) -> Column:
        """Returns True if the column value is not null."""
        return Column(~self._expr.is_null())

    def isin(self, *values) -> Column:
        """Returns True if the value is in the given list."""
        import datafusion.functions as F

        lit_values = [_lit(v) for v in values]
        return Column(self._expr.in_list(lit_values, negated=False))

    def between(self, lower, upper) -> Column:
        """Returns True if the value is between *lower* and *upper* inclusive."""
        return (self >= lower) & (self <= upper)

    def like(self, pattern: str) -> Column:
        """Returns True if the string matches the SQL LIKE pattern."""
        import datafusion.functions as F

        return Column(F.like(self._expr, _lit(pattern)))

    def startswith(self, prefix: str) -> Column:
        """Returns True if the string starts with the given prefix."""
        import datafusion.functions as F

        return Column(F.starts_with(self._expr, _lit(prefix)))

    def endswith(self, suffix: str) -> Column:
        """Returns True if the string ends with the given suffix."""
        import datafusion.functions as F

        return Column(F.ends_with(self._expr, _lit(suffix)))

    def contains(self, value: str) -> Column:
        """Returns True if the string contains the given value."""
        import datafusion.functions as F

        return Column(F.strpos(self._expr, _lit(value)) > _lit(0))

    def asc(self) -> Column:
        """Returns a sort expression for ascending order."""
        return Column(self._expr.sort(ascending=True))

    def desc(self) -> Column:
        """Returns a sort expression for descending order."""
        return Column(self._expr.sort(ascending=False))

    def asc_nulls_first(self) -> Column:
        """Returns a sort expression for ascending order with nulls first."""
        return Column(self._expr.sort(ascending=True, nulls_first=True))

    def desc_nulls_first(self) -> Column:
        """Returns a sort expression for descending order with nulls first."""
        return Column(self._expr.sort(ascending=False, nulls_first=True))

    def asc_nulls_last(self) -> Column:
        """Returns a sort expression for ascending order with nulls last."""
        return Column(self._expr.sort(ascending=True, nulls_first=False))

    def desc_nulls_last(self) -> Column:
        """Returns a sort expression for descending order with nulls last."""
        return Column(self._expr.sort(ascending=False, nulls_first=False))

    def otherwise(self, value) -> Column:
        """Returns the value for the default case in a CASE WHEN chain."""
        return Column(self._expr.otherwise(_unwrap(value)))

    def over(self, window_spec) -> Column:
        """Apply this column expression as a window function over the given WindowSpec."""
        return window_spec._apply(self)

    def when(self, condition, value) -> Column:
        """Chain a WHEN clause onto an existing CASE expression.

        Note: The standalone `when()` function in functions.py starts a new CASE WHEN.
        This Column method chains onto an existing CASE WHEN expression.
        """
        return Column(self._expr.when(_unwrap(condition), _unwrap(value)))

    def substr(self, startPos: int, length: int) -> Column:
        """Returns a substring starting at startPos (1-based) for length characters."""
        import datafusion.functions as F
        return Column(F.substring(self._expr, _lit(startPos), _lit(length)))

    def isNaN(self) -> Column:
        """Returns True if the value is NaN."""
        import datafusion.functions as F
        return Column(F.isnan(self._expr))

    def rlike(self, pattern: str) -> Column:
        """Returns True if the string matches the regex pattern."""
        import datafusion.functions as F
        return Column(F.regexp_match(self._expr, _lit(pattern)).is_not_null())

    def eqNullSafe(self, other) -> Column:
        """Null-safe equality comparison.

        Returns True when both values are equal OR both are null.
        """
        other_expr = _unwrap(other)
        eq = self._expr == other_expr
        both_null = self._expr.is_null() & other_expr.is_null()
        return Column(eq | both_null)

    def bitwiseAND(self, other) -> Column:
        """Bitwise AND (per-row, using & operator on DataFusion Expr)."""
        return Column(self._expr & _unwrap(other))

    def bitwiseOR(self, other) -> Column:
        """Bitwise OR (per-row, using | operator on DataFusion Expr)."""
        return Column(self._expr | _unwrap(other))

    def bitwiseXOR(self, other) -> Column:
        """Bitwise XOR (per-row, using ^ operator on DataFusion Expr)."""
        return Column(self._expr ^ _unwrap(other))

    def __repr__(self):
        return f"Column<'{self._name}'>"

    def __str__(self):
        return self._name


class _SqlFuncColumn(Column):
    """Column wrapping a SQL function call not available in the Python API.

    When used in DataFrame.select(), triggers SQL-based evaluation via
    selectExpr instead of native DataFusion Expr projection.

    Because there is no underlying DataFusion Expr, most Column operations
    (comparison, arithmetic, cast, etc.) are not supported. Attempting to
    use them raises a clear TypeError.
    """

    def __init__(self, sql_fragment: str):
        # Bypass Column.__init__ — we have no DataFusion Expr for this.
        self._sql_fragment = sql_fragment
        self._expr = None  # No DataFusion Expr available
        self._name = sql_fragment
        self._is_sql_func = True

    def _no_expr_op(self, method_name: str):
        raise TypeError(
            f"Cannot call {method_name}() on a SQL function column "
            f"({self._sql_fragment}). Use this column only in "
            f"df.select() or df.selectExpr()."
        )

    # --- Comparison operators ---

    def __eq__(self, other):
        self._no_expr_op("__eq__")

    def __ne__(self, other):
        self._no_expr_op("__ne__")

    def __lt__(self, other):
        self._no_expr_op("__lt__")

    def __le__(self, other):
        self._no_expr_op("__le__")

    def __gt__(self, other):
        self._no_expr_op("__gt__")

    def __ge__(self, other):
        self._no_expr_op("__ge__")

    # --- Boolean operators ---

    def __and__(self, other):
        self._no_expr_op("__and__")

    def __or__(self, other):
        self._no_expr_op("__or__")

    def __invert__(self):
        self._no_expr_op("__invert__")

    # --- Arithmetic operators ---

    def __add__(self, other):
        self._no_expr_op("__add__")

    def __sub__(self, other):
        self._no_expr_op("__sub__")

    def __mul__(self, other):
        self._no_expr_op("__mul__")

    def __truediv__(self, other):
        self._no_expr_op("__truediv__")

    def __mod__(self, other):
        self._no_expr_op("__mod__")

    def __neg__(self):
        self._no_expr_op("__neg__")

    # --- Methods that use self._expr ---

    def cast(self, dataType):
        self._no_expr_op("cast")

    def isNull(self):
        self._no_expr_op("isNull")

    def isNotNull(self):
        self._no_expr_op("isNotNull")

    def isNaN(self):
        self._no_expr_op("isNaN")

    def isin(self, *values):
        self._no_expr_op("isin")

    def between(self, lower, upper):
        self._no_expr_op("between")

    def like(self, pattern):
        self._no_expr_op("like")

    def startswith(self, prefix):
        self._no_expr_op("startswith")

    def endswith(self, suffix):
        self._no_expr_op("endswith")

    def contains(self, value):
        self._no_expr_op("contains")

    def asc(self):
        self._no_expr_op("asc")

    def desc(self):
        self._no_expr_op("desc")

    def otherwise(self, value):
        self._no_expr_op("otherwise")

    def over(self, window_spec):
        self._no_expr_op("over")

    def when(self, condition, value):
        self._no_expr_op("when")

    def substr(self, startPos, length):
        self._no_expr_op("substr")

    def rlike(self, pattern):
        self._no_expr_op("rlike")

    def eqNullSafe(self, other):
        self._no_expr_op("eqNullSafe")

    def bitwiseAND(self, other):
        self._no_expr_op("bitwiseAND")

    def bitwiseOR(self, other):
        self._no_expr_op("bitwiseOR")

    def bitwiseXOR(self, other):
        self._no_expr_op("bitwiseXOR")

    @property
    def expr(self):
        raise TypeError(
            f"SQL function column ({self._sql_fragment}) has no DataFusion Expr. "
            f"Use only in df.select() or df.selectExpr()."
        )

    def alias(self, name: str) -> _SqlFuncColumn:
        return _SqlFuncColumn(f"{self._sql_fragment} AS \"{name}\"")

    def __repr__(self):
        return f"Column<'{self._sql_fragment}'>"

    def __str__(self):
        return self._sql_fragment


def _unwrap(value):
    """Unwrap a Column to its DataFusion Expr, or convert a literal value."""
    if isinstance(value, Column):
        return value._expr
    return _lit(value)
