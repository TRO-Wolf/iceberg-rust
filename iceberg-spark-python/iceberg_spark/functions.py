"""PySpark-compatible SQL functions mapped to DataFusion functions."""

from __future__ import annotations

from typing import Any

import datafusion.functions as F
from datafusion import lit as _df_lit

from iceberg_spark.column import Column, _unwrap


# --- Column reference and literals ---


def col(name: str) -> Column:
    """Returns a Column based on the given column name."""
    return Column(name)


def column(name: str) -> Column:
    """Returns a Column based on the given column name (alias for col)."""
    return Column(name)


def lit(value: Any) -> Column:
    """Creates a Column of literal value."""
    return Column(_df_lit(value))


# --- Aggregation functions ---


def count(col_or_star: str | Column = "*") -> Column:
    """Returns the number of rows."""
    if col_or_star == "*":
        return Column(F.count(_df_lit(1)))
    return Column(F.count(_unwrap(col_or_star if isinstance(col_or_star, Column) else Column(col_or_star))))


def sum(col: str | Column) -> Column:
    """Returns the sum of values."""
    return Column(F.sum(_unwrap(col if isinstance(col, Column) else Column(col))))


def avg(col: str | Column) -> Column:
    """Returns the average of values."""
    return Column(F.avg(_unwrap(col if isinstance(col, Column) else Column(col))))


def mean(col: str | Column) -> Column:
    """Alias for avg."""
    return avg(col)


def min(col: str | Column) -> Column:
    """Returns the minimum value."""
    return Column(F.min(_unwrap(col if isinstance(col, Column) else Column(col))))


def max(col: str | Column) -> Column:
    """Returns the maximum value."""
    return Column(F.max(_unwrap(col if isinstance(col, Column) else Column(col))))


def first(col: str | Column, ignorenulls: bool = False) -> Column:
    """Returns the first value in a group."""
    return Column(F.first_value(_unwrap(col if isinstance(col, Column) else Column(col))))


def last(col: str | Column, ignorenulls: bool = False) -> Column:
    """Returns the last value in a group."""
    return Column(F.last_value(_unwrap(col if isinstance(col, Column) else Column(col))))


def count_distinct(col: str | Column) -> Column:
    """Returns the number of distinct values."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.count(expr, distinct=True))


def approx_count_distinct(col: str | Column) -> Column:
    """Returns approximate distinct count."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.approx_distinct(expr))


def collect_list(col: str | Column) -> Column:
    """Returns a list of all values in the group."""
    return Column(F.array_agg(_unwrap(col if isinstance(col, Column) else Column(col))))


def stddev(col: str | Column) -> Column:
    """Returns the sample standard deviation."""
    return Column(F.stddev(_unwrap(col if isinstance(col, Column) else Column(col))))


def variance(col: str | Column) -> Column:
    """Returns the sample variance."""
    return Column(F.var_sample(_unwrap(col if isinstance(col, Column) else Column(col))))


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
        percentage,
        num_centroids=accuracy,
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


# --- String functions ---


def upper(col: str | Column) -> Column:
    """Converts a string column to upper case."""
    return Column(F.upper(_unwrap(col if isinstance(col, Column) else Column(col))))


def lower(col: str | Column) -> Column:
    """Converts a string column to lower case."""
    return Column(F.lower(_unwrap(col if isinstance(col, Column) else Column(col))))


def trim(col: str | Column) -> Column:
    """Trims leading and trailing whitespace."""
    return Column(F.trim(_unwrap(col if isinstance(col, Column) else Column(col))))


def ltrim(col: str | Column) -> Column:
    """Trims leading whitespace."""
    return Column(F.ltrim(_unwrap(col if isinstance(col, Column) else Column(col))))


def rtrim(col: str | Column) -> Column:
    """Trims trailing whitespace."""
    return Column(F.rtrim(_unwrap(col if isinstance(col, Column) else Column(col))))


def length(col: str | Column) -> Column:
    """Returns the character length of a string column."""
    return Column(F.character_length(_unwrap(col if isinstance(col, Column) else Column(col))))


def concat(*cols: str | Column) -> Column:
    """Concatenates multiple string columns."""
    exprs = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
    return Column(F.concat(*exprs))


def concat_ws(sep: str, *cols: str | Column) -> Column:
    """Concatenates multiple string columns using the given separator."""
    exprs = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
    return Column(F.concat_ws(_df_lit(sep), *exprs))


def substring(col: str | Column, pos: int, length: int) -> Column:
    """Returns a substring starting at *pos* (1-based) for *length* characters."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.substr(expr, _df_lit(pos), _df_lit(length)))


def regexp_replace(col: str | Column, pattern: str, replacement: str) -> Column:
    """Replaces all substrings matching the regex pattern with the replacement."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.regexp_replace(expr, _df_lit(pattern), _df_lit(replacement)))


def split(col: str | Column, pattern: str) -> Column:
    """Splits a string column by the given pattern and returns an array."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.string_to_array(expr, _df_lit(pattern)))


def reverse(col: str | Column) -> Column:
    """Reverses a string or array column."""
    return Column(F.reverse(_unwrap(col if isinstance(col, Column) else Column(col))))


def lpad(col: str | Column, length: int, pad: str = " ") -> Column:
    """Left-pads a string column to the given length with the pad character."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.lpad(expr, _df_lit(length), _df_lit(pad)))


def rpad(col: str | Column, length: int, pad: str = " ") -> Column:
    """Right-pads a string column to the given length with the pad character."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.rpad(expr, _df_lit(length), _df_lit(pad)))


def regexp_extract(col: str | Column, pattern: str, idx: int) -> Column:
    """Extracts a group from a regex match.

    DataFusion regexp_match returns an array of capture groups (or null if no match).
    idx=0 returns the whole match (wraps pattern in a capture group),
    idx>=1 returns the nth capture group from the original pattern.
    """
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    if idx == 0:
        # Wrap the whole pattern in a capture group to get the full match
        match_arr = F.regexp_match(expr, _df_lit(f"({pattern})"))
    else:
        match_arr = F.regexp_match(expr, _df_lit(pattern))
    # Extract the element at the appropriate 1-based index
    target_idx = 1 if idx == 0 else idx
    return Column(F.array_element(match_arr, _df_lit(target_idx)))


def translate(col: str | Column, matching: str, replace: str) -> Column:
    """Translates characters (like Unix tr)."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.translate(expr, _df_lit(matching), _df_lit(replace)))


def locate(substr: str, col: str | Column, pos: int = 1) -> Column:
    """Returns the 1-based position of the first occurrence of substr.

    Args:
        substr: Substring to search for.
        col: Column to search in.
        pos: Starting position (1-based). Default 1 searches from the beginning.
    """
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    if pos <= 1:
        return Column(F.strpos(expr, _df_lit(substr)))
    # Search from position `pos`: extract substring from pos onward, find within it,
    # then adjust the result back to the original string's position.
    sub_expr = F.substr(expr, _df_lit(pos))
    found = F.strpos(sub_expr, _df_lit(substr))
    # If found (> 0), adjust by adding pos-1; otherwise return 0
    return Column(
        F.when(found > _df_lit(0), found + _df_lit(pos - 1)).otherwise(_df_lit(0))
    )


def instr(col: str | Column, substr: str) -> Column:
    """Returns the 1-based position of the first occurrence of substr."""
    return locate(substr, col)


def initcap(col: str | Column) -> Column:
    """Capitalizes the first letter of each word."""
    return Column(F.initcap(_unwrap(col if isinstance(col, Column) else Column(col))))


def ascii_func(col: str | Column) -> Column:
    """Returns the ASCII value of the first character.

    Named ascii_func to avoid shadowing Python builtin.
    """
    return Column(F.ascii(_unwrap(col if isinstance(col, Column) else Column(col))))


def chr_func(col: str | Column) -> Column:
    """Returns the character for the ASCII value.

    Named chr_func to avoid shadowing Python builtin.
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


# --- Date/time functions ---


def current_date() -> Column:
    """Returns the current date as a date column."""
    return Column(F.current_date())


def current_timestamp() -> Column:
    """Returns the current timestamp."""
    return Column(F.current_time())


def year(col: str | Column) -> Column:
    """Extracts the year from a date or timestamp column."""
    return Column(F.date_part(_df_lit("year"), _unwrap(col if isinstance(col, Column) else Column(col))))


def month(col: str | Column) -> Column:
    """Extracts the month from a date or timestamp column."""
    return Column(F.date_part(_df_lit("month"), _unwrap(col if isinstance(col, Column) else Column(col))))


def dayofmonth(col: str | Column) -> Column:
    """Extracts the day of the month from a date or timestamp column."""
    return Column(F.date_part(_df_lit("day"), _unwrap(col if isinstance(col, Column) else Column(col))))


def hour(col: str | Column) -> Column:
    """Extracts the hour from a timestamp column."""
    return Column(F.date_part(_df_lit("hour"), _unwrap(col if isinstance(col, Column) else Column(col))))


def minute(col: str | Column) -> Column:
    """Extracts the minute from a timestamp column."""
    return Column(F.date_part(_df_lit("minute"), _unwrap(col if isinstance(col, Column) else Column(col))))


def second(col: str | Column) -> Column:
    """Extracts the second from a timestamp column."""
    return Column(F.date_part(_df_lit("second"), _unwrap(col if isinstance(col, Column) else Column(col))))


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

    Builds a formatted string from date_part extractions.
    Supports: %Y (year), %m (month), %d (day), %H (hour), %M (minute), %S (second).
    """
    import pyarrow as pa
    expr = _unwrap(col if isinstance(col, Column) else Column(col))

    # Build the formatted string by replacing format specifiers.
    # Coalesce adjacent literal characters into single literal strings.
    parts = []
    literal_buf = []
    i = 0
    while i < len(fmt):
        if fmt[i] == "%" and i + 1 < len(fmt):
            # Flush accumulated literal characters
            if literal_buf:
                parts.append(_df_lit("".join(literal_buf)))
                literal_buf = []
            spec = fmt[i + 1]
            if spec == "Y":
                parts.append(F.lpad(F.date_part(_df_lit("year"), expr).cast(pa.utf8()), _df_lit(4), _df_lit("0")))
            elif spec == "m":
                parts.append(F.lpad(F.date_part(_df_lit("month"), expr).cast(pa.utf8()), _df_lit(2), _df_lit("0")))
            elif spec == "d":
                parts.append(F.lpad(F.date_part(_df_lit("day"), expr).cast(pa.utf8()), _df_lit(2), _df_lit("0")))
            elif spec == "H":
                parts.append(F.lpad(F.date_part(_df_lit("hour"), expr).cast(pa.utf8()), _df_lit(2), _df_lit("0")))
            elif spec == "M":
                parts.append(F.lpad(F.date_part(_df_lit("minute"), expr).cast(pa.utf8()), _df_lit(2), _df_lit("0")))
            elif spec == "S":
                parts.append(F.lpad(F.date_part(_df_lit("second"), expr).cast(pa.utf8()), _df_lit(2), _df_lit("0")))
            else:
                literal_buf.append(f"%{spec}")
            i += 2
        else:
            literal_buf.append(fmt[i])
            i += 1

    # Flush remaining literal characters
    if literal_buf:
        parts.append(_df_lit("".join(literal_buf)))

    if len(parts) == 1:
        return Column(parts[0])
    return Column(F.concat(*parts))


def to_date(col: str | Column, fmt: str | None = None) -> Column:
    """Converts a string to a date.

    Uses cast to Date32 since DataFusion's F.to_date is not in the Python API.
    """
    import pyarrow as pa
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(expr.cast(pa.date32()))


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


def unix_timestamp(col: str | Column | None = None, fmt: str | None = None) -> Column:
    """Converts a date/timestamp column to Unix epoch seconds.

    If called with no arguments, returns the current Unix timestamp.
    """
    import pyarrow as pa
    if col is None:
        return Column(F.date_part(_df_lit("epoch"), F.now()))
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    ts_expr = expr.cast(pa.timestamp("us"))
    return Column(F.date_part(_df_lit("epoch"), ts_expr))


def from_unixtime(col: str | Column, fmt: str | None = None) -> Column:
    """Converts Unix epoch seconds to a timestamp string."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    ts = F.to_timestamp_seconds(expr)
    if fmt:
        return date_format(Column(ts), fmt)
    return Column(ts)


def date_add_func(col: str | Column, days: int) -> Column:
    """Adds days to a date column.

    Named date_add_func internally; exported as date_add.
    """
    import datetime as _dt
    import pyarrow as pa
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    ts = expr.cast(pa.timestamp("us"))
    result = ts + _df_lit(_dt.timedelta(days=days))
    return Column(result.cast(pa.date32()))


def date_sub(col: str | Column, days: int) -> Column:
    """Subtracts days from a date column."""
    return date_add_func(col, -days)


def add_months(col: str | Column, months: int) -> Column:
    """Adds months to a date column.

    Uses year/month/day extraction and make_date reconstruction.
    """
    import pyarrow as pa
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    y = F.date_part(_df_lit("year"), expr).cast(pa.int32())
    m = F.date_part(_df_lit("month"), expr).cast(pa.int32())
    d = F.date_part(_df_lit("day"), expr).cast(pa.int32())
    # Convert to 0-based month index (0=Jan), add months, then convert back
    # This handles negative months correctly
    total_months_0 = (y * _df_lit(12)) + (m - _df_lit(1)) + _df_lit(months)
    new_year = F.floor(total_months_0 / _df_lit(12)).cast(pa.int32())
    new_month = (total_months_0 - new_year * _df_lit(12) + _df_lit(1)).cast(pa.int32())
    return Column(F.make_date(new_year, new_month, d))


def months_between(col1: str | Column, col2: str | Column, roundOff: bool = True) -> Column:
    """Returns the number of months between two dates."""
    e1 = _unwrap(col1 if isinstance(col1, Column) else Column(col1))
    e2 = _unwrap(col2 if isinstance(col2, Column) else Column(col2))
    y1 = F.date_part(_df_lit("year"), e1)
    m1 = F.date_part(_df_lit("month"), e1)
    d1 = F.date_part(_df_lit("day"), e1)
    y2 = F.date_part(_df_lit("year"), e2)
    m2 = F.date_part(_df_lit("month"), e2)
    d2 = F.date_part(_df_lit("day"), e2)
    result = (y1 - y2) * _df_lit(12) + (m1 - m2) + (d1 - d2) / _df_lit(31.0)
    if roundOff:
        return Column(F.round(result, _df_lit(8)))
    return Column(result)


def last_day(col: str | Column) -> Column:
    """Returns the last day of the month for the given date."""
    import datetime as _dt
    import pyarrow as pa
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    y = F.date_part(_df_lit("year"), expr).cast(pa.int32())
    m = F.date_part(_df_lit("month"), expr).cast(pa.int32())
    next_m = ((m % _df_lit(12)) + _df_lit(1)).cast(pa.int32())
    next_y = (y + F.floor(m / _df_lit(12))).cast(pa.int32())
    first_of_next = F.make_date(next_y, next_m, _df_lit(1))
    ts = first_of_next.cast(pa.timestamp("us"))
    result = ts - _df_lit(_dt.timedelta(days=1))
    return Column(result.cast(pa.date32()))


def next_day(col: str | Column, dayOfWeek: str) -> Column:
    """Returns the first date after the given date that falls on the specified day of week.

    Args:
        col: Date column.
        dayOfWeek: Day name like 'Mon', 'Monday', 'Tue', 'Tuesday', etc.
    """
    import datetime as _dt
    import pyarrow as pa
    day_map = {
        "sun": 1, "sunday": 1,
        "mon": 2, "monday": 2,
        "tue": 3, "tuesday": 3,
        "wed": 4, "wednesday": 4,
        "thu": 5, "thursday": 5,
        "fri": 6, "friday": 6,
        "sat": 7, "saturday": 7,
    }
    target = day_map.get(dayOfWeek.lower())
    if target is None:
        raise ValueError(f"Unknown day of week: {dayOfWeek}")

    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    dow = F.date_part(_df_lit("dow"), expr)
    target_0 = target - 1  # 0-based (Sunday=0)
    # Build a CASE WHEN chain for each possible diff (1..7 days)
    ts = expr.cast(pa.timestamp("us"))
    # diff = (target_0 - dow + 7) % 7, replace 0 with 7
    chain = F.when(
        dow == _df_lit(float(target_0)),
        ts + _df_lit(_dt.timedelta(days=7)),
    )
    for offset in range(1, 7):
        # dow value that gives this offset: dow = (target_0 - offset + 7) % 7
        dow_val = (target_0 - offset + 7) % 7
        chain = chain.when(
            dow == _df_lit(float(dow_val)),
            ts + _df_lit(_dt.timedelta(days=offset)),
        )
    # Fallback to 7 days (same day of week)
    return Column(chain.otherwise(ts + _df_lit(_dt.timedelta(days=7))).cast(pa.date32()))


# --- Math functions ---


def abs(col: str | Column) -> Column:
    """Returns the absolute value."""
    return Column(F.abs(_unwrap(col if isinstance(col, Column) else Column(col))))


def ceil(col: str | Column) -> Column:
    """Returns the ceiling of a numeric value."""
    return Column(F.ceil(_unwrap(col if isinstance(col, Column) else Column(col))))


def floor(col: str | Column) -> Column:
    """Returns the floor of a numeric value."""
    return Column(F.floor(_unwrap(col if isinstance(col, Column) else Column(col))))


def round(col: str | Column, scale: int = 0) -> Column:
    """Rounds a numeric column to the given number of decimal places."""
    return Column(F.round(_unwrap(col if isinstance(col, Column) else Column(col)), _df_lit(scale)))


def sqrt(col: str | Column) -> Column:
    """Returns the square root."""
    return Column(F.sqrt(_unwrap(col if isinstance(col, Column) else Column(col))))


def log(col: str | Column) -> Column:
    """Returns the natural logarithm."""
    return Column(F.ln(_unwrap(col if isinstance(col, Column) else Column(col))))


def exp(col: str | Column) -> Column:
    """Returns Euler's number e raised to the power of the column value."""
    return Column(F.exp(_unwrap(col if isinstance(col, Column) else Column(col))))


def power(col: str | Column, p: float) -> Column:
    """Returns the value raised to the power of *p*."""
    return Column(F.power(_unwrap(col if isinstance(col, Column) else Column(col)), _df_lit(p)))


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
    """Returns the SHA-224 hash as hex string.

    Note: DataFusion does not support SHA-1. Uses SHA-224 as the closest alternative.
    """
    return Column(F.encode(F.sha224(_unwrap(col if isinstance(col, Column) else Column(col))), _df_lit("hex")))


def sha2(col: str | Column, numBits: int = 256) -> Column:
    """Returns the SHA-2 hash as hex string (default SHA-256)."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    if numBits == 224:
        return Column(F.encode(F.sha224(expr), _df_lit("hex")))
    elif numBits == 384:
        return Column(F.encode(F.sha384(expr), _df_lit("hex")))
    elif numBits == 512:
        return Column(F.encode(F.sha512(expr), _df_lit("hex")))
    else:
        return Column(F.encode(F.sha256(expr), _df_lit("hex")))


# --- Conditional functions ---


def when(condition: Column, value: Any) -> Column:
    """Evaluates conditions and returns values (start of a CASE WHEN chain)."""
    return Column(F.when(_unwrap(condition), _unwrap(value if isinstance(value, Column) else lit(value))))


def coalesce(*cols: str | Column) -> Column:
    """Returns the first non-null value among the columns."""
    exprs = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
    return Column(F.coalesce(*exprs))


def isnull(col: str | Column) -> Column:
    """Returns True if the column value is null."""
    return Column(col if isinstance(col, Column) else Column(col)).isNull()


def isnan(col: str | Column) -> Column:
    """Returns True if the column value is NaN."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.isnan(expr))


# --- Null handling ---


def nullif(col1: str | Column, col2: str | Column) -> Column:
    """Returns null if col1 equals col2, otherwise returns col1."""
    return Column(F.nullif(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))


def isnotnull(col: str | Column) -> Column:
    """Returns True if the value is not null."""
    c = col if isinstance(col, Column) else Column(col)
    return c.isNotNull()


def greatest(*cols: str | Column) -> Column:
    """Returns the greatest value among the columns.

    DataFusion v52 has no F.greatest(); implemented via chained CASE WHEN.
    NULL-safe: returns NULL only when ALL inputs are NULL.
    """
    resolved = [col if isinstance(col, Column) else Column(col) for col in cols]
    if len(resolved) == 1:
        return resolved[0]
    result = resolved[0]
    for c in resolved[1:]:
        r_expr = _unwrap(result)
        c_expr = _unwrap(c)
        # NULL-safe: COALESCE handles the case where one side is NULL
        result = Column(F.coalesce(
            F.when(r_expr >= c_expr, r_expr).otherwise(c_expr),
            r_expr,
            c_expr,
        ))
    return result


def least(*cols: str | Column) -> Column:
    """Returns the least value among the columns.

    DataFusion v52 has no F.least(); implemented via chained CASE WHEN.
    NULL-safe: returns NULL only when ALL inputs are NULL.
    """
    resolved = [col if isinstance(col, Column) else Column(col) for col in cols]
    if len(resolved) == 1:
        return resolved[0]
    result = resolved[0]
    for c in resolved[1:]:
        r_expr = _unwrap(result)
        c_expr = _unwrap(c)
        # NULL-safe: COALESCE handles the case where one side is NULL
        result = Column(F.coalesce(
            F.when(r_expr <= c_expr, r_expr).otherwise(c_expr),
            r_expr,
            c_expr,
        ))
    return result


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


# --- Window functions ---
# These return Column objects wrapping the underlying DataFusion window function.
# Apply them with .over(window_spec) to produce a window expression.


def row_number() -> Column:
    """Returns the row number within the window partition."""
    return Column(F.row_number())


def rank() -> Column:
    """Returns the rank of rows within a window partition (with gaps)."""
    return Column(F.rank())


def dense_rank() -> Column:
    """Returns the rank with no gaps."""
    return Column(F.dense_rank())


def percent_rank() -> Column:
    """Returns the relative rank (0.0 to 1.0) within the window partition."""
    return Column(F.percent_rank())


def cume_dist() -> Column:
    """Returns the cumulative distribution of values within the window partition."""
    return Column(F.cume_dist())


def ntile(n: int) -> Column:
    """Divides rows into n equal buckets and returns the bucket number."""
    return Column(F.ntile(n))


def lag(col: str | Column, offset: int = 1, default: Any = None) -> Column:
    """Returns the value of *col* from *offset* rows before the current row."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    if default is not None:
        return Column(F.lag(expr, offset, default))
    return Column(F.lag(expr, offset))


def lead(col: str | Column, offset: int = 1, default: Any = None) -> Column:
    """Returns the value of *col* from *offset* rows after the current row."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    if default is not None:
        return Column(F.lead(expr, offset, default))
    return Column(F.lead(expr, offset))


def nth_value(col: str | Column, n: int) -> Column:
    """Returns the nth value in the window frame.

    Must be used with .over(window_spec).
    """
    return Column(F.nth_value(
        _unwrap(col if isinstance(col, Column) else Column(col)), n,
    ))


def first_value(col: str | Column) -> Column:
    """Returns the first value in the window frame.

    Must be used with .over(window_spec).
    """
    return Column(F.first_value(
        _unwrap(col if isinstance(col, Column) else Column(col)),
    ))


def last_value(col: str | Column) -> Column:
    """Returns the last value in the window frame.

    Must be used with .over(window_spec).
    """
    return Column(F.last_value(
        _unwrap(col if isinstance(col, Column) else Column(col)),
    ))


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
    """Creates a new row for each element in the array column.

    Returns a tagged Column that DataFrame.select() will detect and
    unnest via DataFusion's unnest_columns().
    """
    c = col if isinstance(col, Column) else Column(col)
    result = Column(c.expr)
    result._explode = True  # Tag for DataFrame.select() to detect
    return result


def flatten(col: str | Column) -> Column:
    """Flattens a nested array (array of arrays -> single array)."""
    return Column(F.flatten(_unwrap(col if isinstance(col, Column) else Column(col))))


def size(col: str | Column) -> Column:
    """Returns the number of elements in the array or map."""
    return Column(F.array_length(_unwrap(col if isinstance(col, Column) else Column(col))))


def sort_array(col: str | Column, asc: bool = True) -> Column:
    """Sorts the array in ascending or descending order."""
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    # DataFusion array_sort(array, descending: bool, null_first: bool)
    return Column(F.array_sort(expr, descending=not asc, null_first=False))


def struct_func(*cols: str | Column) -> Column:
    """Creates a struct column from the given columns.

    Named struct_func to avoid shadowing Python builtin. Exported as struct.
    """
    exprs = [_unwrap(c if isinstance(c, Column) else Column(c)) for c in cols]
    return Column(F.struct(*exprs))


# --- Misc functions (Sprint 1) ---


def expr(expression: str) -> Column:
    """Parses a SQL expression string into a Column.

    For simple column references (e.g., ``expr("col_name")``), this is equivalent
    to ``col("col_name")``. For complex expressions (e.g., ``expr("a + 1")``),
    DataFusion resolves the expression at query execution time.
    """
    return Column(expression)


def monotonically_increasing_id() -> Column:
    """Returns a column with monotonically increasing 64-bit integers.

    In single-node mode this is equivalent to ``row_number() - 1`` applied
    over the entire DataFrame.  Must be used with ``.over(Window.orderBy(...))``.
    """
    return Column(F.row_number())


def spark_partition_id() -> Column:
    """Returns the partition ID.  Always 0 in single-node mode."""
    return lit(0)


def broadcast(df):
    """Marks a DataFrame for broadcast join.  No-op in single-node mode."""
    return df


def typedLit(value: Any) -> Column:
    """Creates a Column of literal value with inferred type (alias for lit)."""
    return lit(value)


def input_file_name() -> Column:
    """Returns the name of the file being read.  Stub — returns empty string."""
    return lit("")


def nanvl(col1: str | Column, col2: str | Column) -> Column:
    """Returns col1 if it is not NaN, otherwise col2."""
    e1 = col1 if isinstance(col1, Column) else Column(col1)
    e2 = col2 if isinstance(col2, Column) else Column(col2)
    return when(isnan(e1), e2).otherwise(e1)


def bitwise_not(col: str | Column) -> Column:
    """Returns the bitwise NOT of a column.

    Implemented as ``0 - col - 1`` which equals ``~col`` in two's complement.
    DataFusion has no scalar bitwise NOT function.
    """
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(_df_lit(0) - expr - _df_lit(1))


# --- Format functions (Sprint 2) ---


def format_number(col: str | Column, d: int) -> Column:
    """Formats a number with d decimal places."""
    import pyarrow as pa
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(F.round(expr, _df_lit(d)).cast(pa.utf8()))


def format_string(fmt: str, *cols: str | Column) -> Column:
    """Formats the arguments using printf-style format string.

    Supports %s and %d specifiers.
    """
    import re
    parts = re.split(r'(%[sd])', fmt)
    col_iter = iter(cols)
    exprs = []
    for part in parts:
        if part in ('%s', '%d'):
            c = next(col_iter)
            import pyarrow as pa
            exprs.append(_unwrap(c if isinstance(c, Column) else Column(c)).cast(pa.utf8()))
        elif part:
            exprs.append(_df_lit(part))
    if len(exprs) == 1:
        return Column(exprs[0])
    return Column(F.concat(*exprs))


# --- Collection function gaps (Sprint 2) ---


def array_except(col1: str | Column, col2: str | Column) -> Column:
    """Returns elements in col1 but not in col2."""
    return Column(F.array_except(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))


def array_intersect(col1: str | Column, col2: str | Column) -> Column:
    """Returns elements present in both arrays."""
    return Column(F.array_intersect(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))


def arrays_overlap(col1: str | Column, col2: str | Column) -> Column:
    """Returns true if the arrays share any elements."""
    return Column(F.array_has_any(
        _unwrap(col1 if isinstance(col1, Column) else Column(col1)),
        _unwrap(col2 if isinstance(col2, Column) else Column(col2)),
    ))


def map_keys(col: str | Column) -> Column:
    """Returns the keys of a map column."""
    from iceberg_spark.column import _SqlFuncColumn

    col_ref = col if isinstance(col, str) else col._expr.schema_name()
    return _SqlFuncColumn(f"map_keys({col_ref})")


def map_values(col: str | Column) -> Column:
    """Returns the values of a map column."""
    from iceberg_spark.column import _SqlFuncColumn

    col_ref = col if isinstance(col, str) else col._expr.schema_name()
    return _SqlFuncColumn(f"map_values({col_ref})")


def create_map(*cols: str | Column) -> Column:
    """Creates a map from alternating key-value column pairs."""
    from iceberg_spark.column import _SqlFuncColumn

    parts = []
    for c in cols:
        if isinstance(c, str):
            parts.append(f'"{c}"')
        elif isinstance(c, Column):
            parts.append(c._expr.schema_name())
        else:
            parts.append(str(c))
    return _SqlFuncColumn(f"make_map({', '.join(parts)})")


# --- JSON helpers ---


def _parse_ddl_struct(ddl: str):
    """Parse a Spark DDL schema string into a ``pa.DataType``.

    Supports forms like ``"struct<name:string,age:bigint>"`` as well as the
    bare field list ``"name:string,age:bigint"`` and simple ``"name type, ..."``
    pairs.  Correctly handles nested complex types such as
    ``"struct<items:array<string>,data:map<string,int>>"``.
    """
    import pyarrow as pa

    from iceberg_spark._internal.type_mapping import (
        _split_top_level,
        spark_type_from_name,
    )

    ddl = ddl.strip()
    # Strip optional struct<...> wrapper
    if ddl.lower().startswith("struct<") and ddl.endswith(">"):
        ddl = ddl[len("struct<"):-1]

    fields: list[pa.Field] = []
    for part in _split_top_level(ddl, ","):
        part = part.strip()
        if not part:
            continue
        # "name:type" or "name type"
        if ":" in part:
            name, type_str = part.split(":", 1)
        else:
            tokens = part.split(None, 1)
            if len(tokens) < 2:
                raise ValueError(f"Cannot parse DDL field: {part!r}")
            name, type_str = tokens
        name = name.strip().strip("`\"'")
        type_str = type_str.strip()
        spark_type = spark_type_from_name(type_str)
        fields.append(pa.field(name, spark_type.to_arrow()))

    return pa.struct(fields)


# --- JSON functions (Sprint 2) ---


def from_json(col: str | Column, schema=None, options: dict | None = None) -> Column:
    """Parses a JSON string column into a struct.

    Uses a DataFusion ScalarUDF to parse each JSON string row into a struct
    according to the given schema.

    Args:
        col: Column name or Column containing JSON strings.
        schema: A StructType, a ``pa.DataType`` (struct), or a DDL schema
            string such as ``"struct<name:string,age:bigint>"``.  Required.
        options: Optional parsing options (currently ignored).

    Returns:
        A Column containing parsed struct values.
    """
    if schema is None:
        raise ValueError("schema is required for from_json()")

    import json as _json

    import pyarrow as pa
    from datafusion.user_defined import ScalarUDF

    from iceberg_spark.types import StructType as SparkStructType

    # Resolve *schema* to a pa.DataType (struct).
    if isinstance(schema, SparkStructType):
        struct_type = schema.to_arrow()
    elif isinstance(schema, pa.DataType):
        struct_type = schema
    elif isinstance(schema, str):
        struct_type = _parse_ddl_struct(schema)
    else:
        raise ValueError(f"Unsupported schema type: {type(schema)}")

    field_names = [f.name for f in struct_type]
    field_types = [f.type for f in struct_type]

    def _parse_json_array(str_arr):
        """Parse JSON strings into a struct array."""
        import io

        import pyarrow.json as pj

        n = len(str_arr)

        # Fast path: batch parse via PyArrow JSON reader
        valid_indices = []
        lines = []
        for i in range(n):
            if str_arr[i].is_valid:
                lines.append(str_arr[i].as_py())
                valid_indices.append(i)

        if not lines:
            # All nulls — return null struct array
            return pa.StructArray.from_arrays(
                [pa.nulls(n, type=ft) for ft in field_types],
                names=field_names,
            )

        try:
            buf = io.BytesIO("\n".join(lines).encode("utf-8"))
            parsed_table = pj.read_json(buf)

            # Scatter parsed values back into full-length arrays with nulls
            result_arrays: dict[str, list] = {}
            for name, ft in zip(field_names, field_types):
                if name in parsed_table.column_names:
                    parsed_col = parsed_table.column(name)
                else:
                    parsed_col = pa.nulls(len(lines), type=ft)
                full: list = [None] * n
                for j, idx in enumerate(valid_indices):
                    full[idx] = parsed_col[j].as_py()
                result_arrays[name] = full

            arrays = [
                pa.array(result_arrays[name], type=ft)
                for name, ft in zip(field_names, field_types)
            ]
            return pa.StructArray.from_arrays(arrays, names=field_names)
        except Exception:
            # Fallback: row-by-row parsing for malformed JSON
            result_arrays_fb: dict[str, list] = {name: [] for name in field_names}
            for i in range(n):
                if str_arr[i].is_valid:
                    try:
                        parsed = _json.loads(str_arr[i].as_py())
                        for name in field_names:
                            result_arrays_fb[name].append(parsed.get(name))
                    except (_json.JSONDecodeError, TypeError):
                        for name in field_names:
                            result_arrays_fb[name].append(None)
                else:
                    for name in field_names:
                        result_arrays_fb[name].append(None)

            arrays = [
                pa.array(result_arrays_fb[name], type=ft)
                for name, ft in zip(field_names, field_types)
            ]
            return pa.StructArray.from_arrays(arrays, names=field_names)

    udf_name = f"_from_json_{id(schema)}"
    scalar_udf = ScalarUDF.udf(
        _parse_json_array,
        [pa.utf8()],
        struct_type,
        "volatile",
        udf_name,
    )

    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return Column(scalar_udf(expr))


class _ToJsonColumn(Column):
    """Deferred to_json column — resolved at DataFrame.select()/withColumn() time.

    DataFusion's ScalarUDF requires exact input types at registration, but
    ``to_json()`` doesn't know the column type until the DataFrame schema
    is available.  This marker column carries the inner expression and gets
    resolved by ``DataFrame.select()`` / ``withColumn()`` which inspect the
    schema and create a properly-typed UDF.

    ``_expr`` is set to ``None`` to prevent silent misuse — any function that
    calls ``_unwrap()`` on this column will get ``None`` and fail loudly
    instead of silently using the inner expression.
    """

    def __init__(self, inner_expr, alias_name=None):
        self._inner_expr = inner_expr
        self._expr = None  # prevent silent bypass via _unwrap()
        self._alias_name = alias_name
        self._name = f"to_json({inner_expr})"
        self._is_to_json = True

    def alias(self, name: str) -> "_ToJsonColumn":
        return _ToJsonColumn(self._inner_expr, alias_name=name)

    def _no_expr_op(self, method_name):
        raise TypeError(
            f"Cannot call {method_name}() on to_json() before it is "
            f"resolved. Use to_json() only inside df.select() or "
            f"df.withColumn()."
        )

    @property
    def expr(self):
        raise TypeError(
            "to_json() column must be resolved via df.select() or "
            "df.withColumn() before accessing the expression."
        )

    def __eq__(self, other): self._no_expr_op("__eq__")
    def __ne__(self, other): self._no_expr_op("__ne__")
    def __lt__(self, other): self._no_expr_op("__lt__")
    def __le__(self, other): self._no_expr_op("__le__")
    def __gt__(self, other): self._no_expr_op("__gt__")
    def __ge__(self, other): self._no_expr_op("__ge__")
    def __add__(self, other): self._no_expr_op("__add__")
    def __sub__(self, other): self._no_expr_op("__sub__")
    def __mul__(self, other): self._no_expr_op("__mul__")
    def __truediv__(self, other): self._no_expr_op("__truediv__")
    def __and__(self, other): self._no_expr_op("__and__")
    def __or__(self, other): self._no_expr_op("__or__")
    def __invert__(self): self._no_expr_op("__invert__")
    def cast(self, dataType): self._no_expr_op("cast")
    def isNull(self): self._no_expr_op("isNull")
    def isNotNull(self): self._no_expr_op("isNotNull")
    def over(self, window): self._no_expr_op("over")


def to_json(col: str | Column, options: dict | None = None) -> Column:
    """Converts a column to a JSON string.

    Works for all column types: structs produce ``{"key":"value",...}``,
    arrays produce ``[1,2,3]``, maps produce ``{"k":"v",...}``, and
    primitives produce their JSON literal.
    """
    expr = _unwrap(col if isinstance(col, Column) else Column(col))
    return _ToJsonColumn(expr)


def schema_of_json(json_str: str, options: dict | None = None) -> Column:
    """Infers schema from a JSON string literal.

    Returns a Column containing the DDL schema string (e.g.
    ``"struct<age:bigint,name:string>"``).

    Args:
        json_str: A JSON string to infer the schema from.
        options: Optional parsing options (currently ignored).
    """
    import io

    import pyarrow.json as pj

    from iceberg_spark._internal.type_mapping import arrow_type_to_spark_string

    buf = io.BytesIO(json_str.encode("utf-8"))
    table = pj.read_json(buf)

    fields = ",".join(
        f"{field.name}:{arrow_type_to_spark_string(field.type)}"
        for field in table.schema
    )
    ddl = f"struct<{fields}>"

    return Column(_df_lit(ddl))


# --- UDF support ---


def udf(f=None, returnType=None, inputTypes=None, batch_mode=False):
    """Create a UDF from a Python function.

    Can be used as a decorator::

        @udf(returnType=StringType())
        def my_upper(s):
            return s.upper() if s else None

    Or as a function::

        my_udf = udf(my_func, StringType())

    When *batch_mode* is ``True`` the function receives ``pyarrow.Array``
    arguments directly and must return a ``pyarrow.Array``.  The per-row
    wrapper is skipped, yielding much better performance::

        @udf(returnType=LongType(), inputTypes=[LongType()], batch_mode=True)
        def double_vals(arr):
            import pyarrow.compute as pc
            return pc.multiply(arr, 2)

    The returned callable accepts Column arguments and produces a Column.
    For SQL usage, also call ``session.udf.register()``.

    Args:
        f: A Python callable.  When using ``udf`` as a decorator, omit this.
        returnType: PySpark DataType instance, type-name string, or
            ``pyarrow.DataType``.  Defaults to ``StringType()``.
        inputTypes: Optional list of PySpark DataType / type-name / pa.DataType
            for the UDF input columns.  When *None* (the default), all
            inputs are declared as ``pa.utf8()``.
        batch_mode: When ``True``, the user function receives
            ``pyarrow.Array`` arguments directly and must return a
            ``pyarrow.Array``.  Defaults to ``False``.

    Returns:
        A ``_UserDefinedFunction`` (or a decorator that produces one).
    """
    import inspect
    import uuid

    import pyarrow as pa
    from datafusion.user_defined import ScalarUDF

    from iceberg_spark.session import (
        _UserDefinedFunction,
        _make_vectorized_wrapper,
        _resolve_return_type,
    )

    def _build(func, ret_type):
        arrow_return_type = _resolve_return_type(ret_type)

        try:
            sig = inspect.signature(func)
            n_args = len(sig.parameters)
        except (ValueError, TypeError):
            n_args = 1

        if inputTypes is not None:
            input_types = [_resolve_return_type(t) for t in inputTypes]
        else:
            input_types = [pa.utf8()] * n_args
        name = getattr(func, "__name__", None) or f"udf_{uuid.uuid4().hex[:8]}"

        if batch_mode:
            # In batch mode pass the user function directly — it already
            # operates on pa.Array objects and returns a pa.Array.
            callable_fn = func
        else:
            callable_fn = _make_vectorized_wrapper(func, n_args, arrow_return_type, name=name)

        scalar_udf = ScalarUDF.udf(callable_fn, input_types, arrow_return_type, "volatile", name)
        return _UserDefinedFunction(name, scalar_udf)

    if f is not None:
        # Called as: udf(my_func, StringType())
        return _build(f, returnType)

    # Called as decorator: @udf(returnType=StringType())
    def _decorator(func):
        return _build(func, returnType)

    return _decorator


def udaf(f=None, returnType=None, inputTypes=None):
    """Create a UDAF (aggregate UDF) from a Python function or Accumulator class.

    Can be used as a decorator::

        @udaf(returnType=LongType(), inputTypes=[LongType()])
        def my_sum(values):
            return sum(values)

    Or as a function::

        my_sum = udaf(lambda values: sum(values), LongType(), [LongType()])

    The returned callable accepts Column arguments and produces a Column that
    represents an aggregate expression.

    For SQL usage, also call ``session.udf.register_udaf()``.

    Args:
        f: A Python callable that takes a list of values and returns a scalar,
           or an Accumulator class.  When using ``udaf`` as a decorator, omit this.
        returnType: PySpark DataType instance, type-name string, or
            ``pyarrow.DataType``.  Defaults to ``LongType()``.
        inputTypes: List of PySpark DataType / type-name / pa.DataType for
            the input columns.  Defaults to ``[LongType()]``.

    Returns:
        A ``_UserDefinedAggregateFunction`` (or a decorator that produces one).
    """
    import uuid

    import pyarrow as pa
    from datafusion.user_defined import Accumulator, AggregateUDF

    from iceberg_spark.session import (
        _SimpleAccumulator,
        _UserDefinedAggregateFunction,
        _resolve_return_type,
    )

    def _build(func, ret_type):
        arrow_return_type = _resolve_return_type(ret_type)

        if inputTypes is not None:
            input_types = [_resolve_return_type(t) for t in inputTypes]
        else:
            input_types = [pa.int64()]

        name = getattr(func, "__name__", None) or f"udaf_{uuid.uuid4().hex[:8]}"
        if name == "<lambda>":
            name = f"udaf_{uuid.uuid4().hex[:8]}"

        # Determine if func is an Accumulator class/factory or a plain callable
        is_accumulator_class = False
        try:
            if isinstance(func, type) and issubclass(func, Accumulator):
                is_accumulator_class = True
        except (TypeError, Exception):
            pass

        if is_accumulator_class:
            accumulator_factory = func
            state_types = [arrow_return_type]
        else:
            captured_func = func
            captured_return_type = arrow_return_type
            captured_input_type = input_types[0]
            state_types = [pa.list_(captured_input_type)]

            def accumulator_factory():
                return _SimpleAccumulator(captured_func, captured_return_type, captured_input_type)

        aggregate_udf = AggregateUDF(
            name=name,
            accumulator=accumulator_factory,
            input_types=input_types,
            return_type=arrow_return_type,
            state_type=state_types,
            volatility="volatile",
        )
        return _UserDefinedAggregateFunction(name, aggregate_udf)

    if f is not None:
        # Called as: udaf(my_func, LongType(), [LongType()])
        return _build(f, returnType)

    # Called as decorator: @udaf(returnType=LongType(), inputTypes=[LongType()])
    def _decorator(func):
        return _build(func, returnType)

    return _decorator
