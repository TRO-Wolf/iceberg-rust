"""PySpark-compatible Row type."""

from __future__ import annotations


class Row:
    """A row in a DataFrame, compatible with PySpark's Row interface.

    Supports attribute access (row.col), item access (row["col"]),
    and index access (row[0]).
    """

    __slots__ = ("_fields", "_values")

    def __init__(self, **kwargs):
        if kwargs:
            self._fields = tuple(kwargs.keys())
            self._values = tuple(kwargs.values())
        else:
            self._fields = ()
            self._values = ()

    @classmethod
    def _from_pairs(cls, fields: tuple[str, ...], values: tuple) -> Row:
        row = object.__new__(cls)
        row._fields = fields
        row._values = values
        return row

    def __getattr__(self, name: str):
        try:
            idx = self._fields.index(name)
            return self._values[idx]
        except ValueError:
            raise AttributeError(f"Row has no field '{name}'")

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._values[item]
        if isinstance(item, str):
            try:
                idx = self._fields.index(item)
                return self._values[idx]
            except ValueError:
                raise KeyError(f"Row has no field '{item}'")
        raise TypeError(f"Row indices must be int or str, not {type(item).__name__}")

    def __len__(self):
        return len(self._values)

    def __contains__(self, item):
        return item in self._fields

    def __iter__(self):
        return iter(self._values)

    def __eq__(self, other):
        if not isinstance(other, Row):
            return NotImplemented
        return self._fields == other._fields and self._values == other._values

    def __hash__(self):
        return hash((self._fields, self._values))

    def __repr__(self):
        pairs = ", ".join(f"{f}={v!r}" for f, v in zip(self._fields, self._values))
        return f"Row({pairs})"

    def asDict(self, recursive: bool = False) -> dict:
        """Returns the row as a dictionary mapping field names to values."""
        d = dict(zip(self._fields, self._values))
        if recursive:
            for k, v in d.items():
                if isinstance(v, Row):
                    d[k] = v.asDict(recursive=True)
        return d
