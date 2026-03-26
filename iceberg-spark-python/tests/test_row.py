"""Tests for the PySpark-compatible Row type."""

from iceberg_spark.row import Row


def test_row_creation():
    row = Row(id=1, name="Alice", age=30)
    assert row.id == 1
    assert row.name == "Alice"
    assert row.age == 30


def test_row_item_access():
    row = Row(id=1, name="Alice")
    assert row["id"] == 1
    assert row["name"] == "Alice"
    assert row[0] == 1
    assert row[1] == "Alice"


def test_row_len():
    row = Row(a=1, b=2, c=3)
    assert len(row) == 3


def test_row_contains():
    row = Row(id=1, name="Alice")
    assert "id" in row
    assert "name" in row
    assert "age" not in row


def test_row_iter():
    row = Row(a=1, b=2, c=3)
    assert list(row) == [1, 2, 3]


def test_row_equality():
    row1 = Row(id=1, name="Alice")
    row2 = Row(id=1, name="Alice")
    row3 = Row(id=2, name="Bob")
    assert row1 == row2
    assert row1 != row3


def test_row_hash():
    row1 = Row(id=1, name="Alice")
    row2 = Row(id=1, name="Alice")
    assert hash(row1) == hash(row2)


def test_row_repr():
    row = Row(id=1, name="Alice")
    assert repr(row) == "Row(id=1, name='Alice')"


def test_row_as_dict():
    row = Row(id=1, name="Alice", age=30)
    d = row.asDict()
    assert d == {"id": 1, "name": "Alice", "age": 30}


def test_row_as_dict_recursive():
    inner = Row(city="NYC", state="NY")
    outer = Row(id=1, address=inner)
    d = outer.asDict(recursive=True)
    assert d == {"id": 1, "address": {"city": "NYC", "state": "NY"}}


def test_row_from_pairs():
    row = Row._from_pairs(("id", "name"), (1, "Alice"))
    assert row.id == 1
    assert row.name == "Alice"


def test_row_attribute_error():
    row = Row(id=1)
    try:
        _ = row.nonexistent
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_row_key_error():
    row = Row(id=1)
    try:
        _ = row["nonexistent"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
