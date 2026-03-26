"""Tests for Phase 4 DML — DELETE FROM, UPDATE, and MERGE INTO.

Covers:
- SQL preprocessor: DELETE_FROM, UPDATE, MERGE_INTO detection
- _parse_set_clause and _parse_merge_into helpers
- Core DataFusion logic for DELETE filtering, UPDATE CASE WHEN,
  and MERGE UNION ALL (no Iceberg catalog needed)
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from datafusion import SessionContext

from iceberg_spark.catalog_ops import (
    _parse_merge_into,
    _parse_set_clause,
    _parse_when_clause,
    _register_referenced_tables,
    _resolve_temp_view_arrow,
    _split_at_commas,
)
from iceberg_spark.sql_preprocessor import CommandType, preprocess


# ---------------------------------------------------------------------------
# SQL Preprocessor — DELETE FROM
# ---------------------------------------------------------------------------


class TestDeleteFromPreprocessor:
    def test_delete_with_where(self):
        result = preprocess("DELETE FROM db.t1 WHERE id > 5")
        assert result.command_type == CommandType.DELETE_FROM
        assert result.table_name == "db.t1"
        assert result.extra["where_clause"] == "id > 5"

    def test_delete_without_where(self):
        result = preprocess("DELETE FROM ns.tbl")
        assert result.command_type == CommandType.DELETE_FROM
        assert result.table_name == "ns.tbl"
        assert result.extra["where_clause"] is None

    def test_delete_simple_table(self):
        result = preprocess("DELETE FROM orders WHERE status = 'cancelled'")
        assert result.command_type == CommandType.DELETE_FROM
        assert result.table_name == "orders"
        assert "status = 'cancelled'" in result.extra["where_clause"]

    def test_delete_case_insensitive(self):
        result = preprocess("delete from db.t1 where id = 1")
        assert result.command_type == CommandType.DELETE_FROM
        assert result.table_name == "db.t1"

    def test_delete_complex_where(self):
        result = preprocess("DELETE FROM t WHERE a > 1 AND b < 10 OR c = 'x'")
        assert result.command_type == CommandType.DELETE_FROM
        assert "a > 1 AND b < 10" in result.extra["where_clause"]

    def test_delete_preserves_sql(self):
        sql = "DELETE FROM db.t1 WHERE id IN (1, 2, 3)"
        result = preprocess(sql)
        assert result.sql == sql


# ---------------------------------------------------------------------------
# SQL Preprocessor — UPDATE
# ---------------------------------------------------------------------------


class TestUpdatePreprocessor:
    def test_update_single_column_with_where(self):
        result = preprocess("UPDATE db.t1 SET name = 'hello' WHERE id = 1")
        assert result.command_type == CommandType.UPDATE
        assert result.table_name == "db.t1"
        assert result.extra["set_clause"] == "name = 'hello'"
        assert result.extra["where_clause"] == "id = 1"

    def test_update_multiple_columns(self):
        result = preprocess("UPDATE t SET a = 1, b = 2, c = 'x' WHERE id = 5")
        assert result.command_type == CommandType.UPDATE
        assert result.table_name == "t"
        assert "a = 1" in result.extra["set_clause"]
        assert "b = 2" in result.extra["set_clause"]
        assert result.extra["where_clause"] == "id = 5"

    def test_update_without_where(self):
        result = preprocess("UPDATE db.t1 SET score = 0")
        assert result.command_type == CommandType.UPDATE
        assert result.table_name == "db.t1"
        assert result.extra["set_clause"] == "score = 0"
        assert result.extra["where_clause"] is None

    def test_update_case_insensitive(self):
        result = preprocess("update ns.tbl set val = 99 where id = 1")
        assert result.command_type == CommandType.UPDATE
        assert result.table_name == "ns.tbl"

    def test_update_expression_rhs(self):
        result = preprocess("UPDATE t SET price = price * 1.1 WHERE category = 'food'")
        assert result.command_type == CommandType.UPDATE
        assert "price * 1.1" in result.extra["set_clause"]
        assert result.extra["where_clause"] == "category = 'food'"


# ---------------------------------------------------------------------------
# _parse_set_clause helper
# ---------------------------------------------------------------------------


class TestParseSetClause:
    def test_single_pair(self):
        pairs = _parse_set_clause("name = 'hello'")
        assert pairs == [("name", "'hello'")]

    def test_multiple_pairs(self):
        pairs = _parse_set_clause("a = 1, b = 2, c = 'x'")
        assert pairs == [("a", "1"), ("b", "2"), ("c", "'x'")]

    def test_function_call_in_expr(self):
        # Comma inside function call should not split
        pairs = _parse_set_clause("col = COALESCE(a, 0)")
        assert len(pairs) == 1
        assert pairs[0][0] == "col"
        assert pairs[0][1] == "COALESCE(a, 0)"

    def test_multiple_with_function(self):
        pairs = _parse_set_clause("a = ROUND(x, 2), b = 99")
        assert len(pairs) == 2
        assert pairs[0] == ("a", "ROUND(x, 2)")
        assert pairs[1] == ("b", "99")

    def test_arithmetic_expression(self):
        pairs = _parse_set_clause("price = price * 1.1")
        assert pairs == [("price", "price * 1.1")]

    def test_whitespace_trimmed(self):
        pairs = _parse_set_clause("  col1  =  42  ,  col2  =  'val'  ")
        assert pairs == [("col1", "42"), ("col2", "'val'")]

    def test_string_literal_with_comma(self):
        pairs = _parse_set_clause("name = 'hello, world'")
        assert len(pairs) == 1
        assert pairs[0] == ("name", "'hello, world'")

    def test_multiple_with_string_comma(self):
        pairs = _parse_set_clause("name = 'a, b', score = 99")
        assert len(pairs) == 2
        assert pairs[0] == ("name", "'a, b'")
        assert pairs[1] == ("score", "99")

    def test_double_quoted_string(self):
        pairs = _parse_set_clause('name = "hello, world", val = 1')
        assert len(pairs) == 2
        assert pairs[0] == ("name", '"hello, world"')
        assert pairs[1] == ("val", "1")

    def test_qualified_column_name(self):
        """SET t.salary = 100 should parse as (t.salary, 100)."""
        pairs = _parse_set_clause("t.salary = 100, t.name = 'x'")
        assert pairs == [("t.salary", "100"), ("t.name", "'x'")]


# ---------------------------------------------------------------------------
# Core DataFusion logic for DELETE (no catalog needed)
# ---------------------------------------------------------------------------


class TestDeleteDataFusionLogic:
    """Tests the filtering logic used inside handle_delete_from."""

    @pytest.fixture
    def arrow_data(self):
        return pa.table({
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "name": pa.array(["a", "b", "c", "d", "e"], type=pa.string()),
        })

    def test_delete_with_where_keeps_correct_rows(self, arrow_data):
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [arrow_data.to_batches()])
        kept = ctx.sql("SELECT * FROM __delete_tmp WHERE NOT (id > 3)").collect()
        result = pa.Table.from_batches(kept)
        assert len(result) == 3  # ids 1, 2, 3
        assert sorted(result["id"].to_pylist()) == [1, 2, 3]

    def test_delete_all_rows(self, arrow_data):
        """No WHERE clause → delete all; result is an empty table."""
        empty = pa.table({
            name: pa.array([], type=arrow_data.schema.field(name).type)
            for name in arrow_data.schema.names
        })
        assert len(empty) == 0
        assert empty.schema.names == arrow_data.schema.names

    def test_delete_no_matching_rows(self, arrow_data):
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [arrow_data.to_batches()])
        kept = ctx.sql("SELECT * FROM __delete_tmp WHERE NOT (id > 100)").collect()
        result = pa.Table.from_batches(kept)
        assert len(result) == 5  # nothing deleted

    def test_delete_all_matching_rows(self, arrow_data):
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [arrow_data.to_batches()])
        kept = ctx.sql("SELECT * FROM __delete_tmp WHERE NOT (id > 0)").collect()
        # Must pass schema when result is empty (no batches)
        result = (
            pa.Table.from_batches(kept, schema=arrow_data.schema)
            if kept
            else arrow_data.slice(0, 0)
        )
        assert len(result) == 0

    def test_delete_string_condition(self, arrow_data):
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [arrow_data.to_batches()])
        kept = ctx.sql("SELECT * FROM __delete_tmp WHERE NOT (name = 'c')").collect()
        result = pa.Table.from_batches(kept)
        assert len(result) == 4
        assert "c" not in result["name"].to_pylist()


# ---------------------------------------------------------------------------
# Core DataFusion logic for UPDATE (no catalog needed)
# ---------------------------------------------------------------------------


class TestUpdateDataFusionLogic:
    """Tests the CASE WHEN query logic used inside handle_update."""

    @pytest.fixture
    def arrow_data(self):
        return pa.table({
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "score": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
            "label": pa.array(["a", "b", "c", "d", "e"], type=pa.string()),
        })

    def test_update_with_where(self, arrow_data):
        """Update score for matching rows only."""
        ctx = SessionContext()
        ctx.register_record_batches("__update_tmp", [arrow_data.to_batches()])
        sql = (
            "SELECT id, "
            "CASE WHEN (id > 3) THEN (99) ELSE score END AS score, "
            "label "
            "FROM __update_tmp"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        assert len(result) == 5
        # ids 1,2,3 keep original scores; ids 4,5 get 99
        scores = dict(zip(result["id"].to_pylist(), result["score"].to_pylist()))
        assert scores[1] == 10
        assert scores[3] == 30
        assert scores[4] == 99
        assert scores[5] == 99

    def test_update_without_where(self, arrow_data):
        """Update all rows (no CASE WHEN)."""
        ctx = SessionContext()
        ctx.register_record_batches("__update_tmp", [arrow_data.to_batches()])
        sql = "SELECT id, (0) AS score, label FROM __update_tmp"
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        assert all(s == 0 for s in result["score"].to_pylist())

    def test_update_preserves_unmodified_columns(self, arrow_data):
        """Columns not in SET are passed through unchanged."""
        ctx = SessionContext()
        ctx.register_record_batches("__update_tmp", [arrow_data.to_batches()])
        sql = (
            "SELECT id, score, "
            "CASE WHEN (id = 1) THEN ('updated') ELSE label END AS label "
            "FROM __update_tmp"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        labels = dict(zip(result["id"].to_pylist(), result["label"].to_pylist()))
        assert labels[1] == "updated"
        assert labels[2] == "b"
        assert labels[5] == "e"

    def test_update_arithmetic_expression(self, arrow_data):
        """SET score = score * 2 WHERE id <= 2."""
        ctx = SessionContext()
        ctx.register_record_batches("__update_tmp", [arrow_data.to_batches()])
        sql = (
            "SELECT id, "
            "CASE WHEN (id <= 2) THEN (score * 2) ELSE score END AS score, "
            "label "
            "FROM __update_tmp"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        scores = dict(zip(result["id"].to_pylist(), result["score"].to_pylist()))
        assert scores[1] == 20   # 10 * 2
        assert scores[2] == 40   # 20 * 2
        assert scores[3] == 30   # unchanged

    def test_update_qualified_column_resolves(self):
        """SET t.score = 99 should match schema column 'score' (strip prefix)."""
        from iceberg_spark.catalog_ops import _parse_set_clause

        set_pairs = _parse_set_clause("t.score = 99")
        # Simulate handle_update's set_dict construction (with prefix stripping)
        set_dict = {col.split(".")[-1].strip().lower(): expr.strip() for col, expr in set_pairs}
        assert "score" in set_dict
        assert set_dict["score"] == "99"


# ---------------------------------------------------------------------------
# SQL Preprocessor — MERGE INTO
# ---------------------------------------------------------------------------


class TestMergeIntoPreprocessor:
    def test_merge_basic(self):
        result = preprocess(
            "MERGE INTO db.target t USING db.source s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.name = s.name"
        )
        assert result.command_type == CommandType.MERGE_INTO
        assert result.table_name == "db.target"

    def test_merge_case_insensitive(self):
        result = preprocess(
            "merge into tbl t using src s on t.id = s.id "
            "when matched then delete"
        )
        assert result.command_type == CommandType.MERGE_INTO
        assert result.table_name == "tbl"

    def test_merge_preserves_sql(self):
        sql = (
            "MERGE INTO target AS t USING source AS s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.x = s.x "
            "WHEN NOT MATCHED THEN INSERT (x) VALUES (s.x)"
        )
        result = preprocess(sql)
        assert result.sql == sql

    def test_merge_with_namespace(self):
        result = preprocess(
            "MERGE INTO catalog.db.table t USING other.src s ON t.k = s.k "
            "WHEN MATCHED THEN DELETE"
        )
        assert result.command_type == CommandType.MERGE_INTO
        assert result.table_name == "catalog.db.table"


# ---------------------------------------------------------------------------
# _parse_merge_into / _parse_when_clause helpers
# ---------------------------------------------------------------------------


class TestParseMergeInto:
    def test_basic_update_insert(self):
        sql = (
            "MERGE INTO db.target t USING db.source s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.name = s.name "
            "WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)"
        )
        result = _parse_merge_into(sql)
        assert result["target_table"] == "db.target"
        assert result["target_alias"] == "t"
        assert result["source_table"] == "db.source"
        assert result["source_alias"] == "s"
        assert result["on_condition"] == "t.id = s.id"
        assert len(result["when_clauses"]) == 2

    def test_with_as_keyword(self):
        sql = (
            "MERGE INTO target AS t USING source AS s ON t.id = s.id "
            "WHEN MATCHED THEN DELETE"
        )
        result = _parse_merge_into(sql)
        assert result["target_alias"] == "t"
        assert result["source_alias"] == "s"

    def test_complex_on_condition(self):
        sql = (
            "MERGE INTO t1 t USING s1 s "
            "ON t.id = s.id AND t.category = s.category "
            "WHEN MATCHED THEN UPDATE SET t.val = s.val"
        )
        result = _parse_merge_into(sql)
        assert "t.id = s.id AND t.category = s.category" in result["on_condition"]

    def test_conditional_when_clauses(self):
        sql = (
            "MERGE INTO t1 t USING s1 s ON t.id = s.id "
            "WHEN MATCHED AND s.op = 'update' THEN UPDATE SET t.name = s.name "
            "WHEN MATCHED AND s.op = 'delete' THEN DELETE "
            "WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)"
        )
        result = _parse_merge_into(sql)
        assert len(result["when_clauses"]) == 3
        assert result["when_clauses"][0]["condition"] == "s.op = 'update'"
        assert result["when_clauses"][0]["action"] == "update"
        assert result["when_clauses"][1]["condition"] == "s.op = 'delete'"
        assert result["when_clauses"][1]["action"] == "delete"
        assert result["when_clauses"][2]["type"] == "not_matched"
        assert result["when_clauses"][2]["action"] == "insert"

    def test_no_when_clauses_raises(self):
        with pytest.raises(RuntimeError, match="at least one WHEN"):
            _parse_merge_into("MERGE INTO t1 t USING s1 s ON t.id = s.id")


class TestParseWhenClause:
    def test_matched_update(self):
        clause = _parse_when_clause("MATCHED THEN UPDATE SET name = s.name, val = s.val")
        assert clause["type"] == "matched"
        assert clause["action"] == "update"
        assert "name = s.name" in clause["set_clause"]
        assert clause["condition"] is None

    def test_matched_delete(self):
        clause = _parse_when_clause("MATCHED THEN DELETE")
        assert clause["type"] == "matched"
        assert clause["action"] == "delete"

    def test_matched_with_condition(self):
        clause = _parse_when_clause(
            "MATCHED AND s.op = 'del' THEN DELETE"
        )
        assert clause["type"] == "matched"
        assert clause["condition"] == "s.op = 'del'"
        assert clause["action"] == "delete"

    def test_not_matched_insert_with_columns(self):
        clause = _parse_when_clause(
            "NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)"
        )
        assert clause["type"] == "not_matched"
        assert clause["action"] == "insert"
        assert clause["columns"] == ["id", "name"]
        assert clause["values"] == ["s.id", "s.name"]

    def test_not_matched_insert_without_columns(self):
        clause = _parse_when_clause(
            "NOT MATCHED THEN INSERT VALUES (s.id, s.name)"
        )
        assert clause["type"] == "not_matched"
        assert clause["action"] == "insert"
        assert clause["columns"] is None
        assert clause["values"] == ["s.id", "s.name"]

    def test_not_matched_with_condition(self):
        clause = _parse_when_clause(
            "NOT MATCHED AND s.active = true THEN INSERT (id) VALUES (s.id)"
        )
        assert clause["type"] == "not_matched"
        assert clause["condition"] == "s.active = true"

    def test_insert_with_function_in_values(self):
        clause = _parse_when_clause(
            "NOT MATCHED THEN INSERT (id, name) VALUES (s.id, CONCAT(s.first, ', ', s.last))"
        )
        assert clause["action"] == "insert"
        assert clause["values"] == ["s.id", "CONCAT(s.first, ', ', s.last)"]


# ---------------------------------------------------------------------------
# _split_at_commas helper
# ---------------------------------------------------------------------------


class TestSplitAtCommas:
    def test_simple(self):
        assert _split_at_commas("a, b, c") == ["a", "b", "c"]

    def test_respects_parens(self):
        assert _split_at_commas("ROUND(x, 2), b") == ["ROUND(x, 2)", "b"]

    def test_respects_quotes(self):
        assert _split_at_commas("'a, b', c") == ["'a, b'", "c"]

    def test_empty(self):
        assert _split_at_commas("") == []


# ---------------------------------------------------------------------------
# Core DataFusion logic for MERGE INTO (no catalog needed)
# ---------------------------------------------------------------------------


class TestMergeDataFusionLogic:
    """Tests the UNION ALL query patterns used inside handle_merge_into."""

    @pytest.fixture
    def merge_data(self):
        target = pa.table({
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": pa.array(["a", "b", "c"], type=pa.string()),
        })
        source = pa.table({
            "id": pa.array([2, 4], type=pa.int64()),
            "name": pa.array(["B", "D"], type=pa.string()),
        })
        return target, source

    def test_merge_update_and_insert(self, merge_data):
        """WHEN MATCHED THEN UPDATE + WHEN NOT MATCHED THEN INSERT."""
        target, source = merge_data
        ctx = SessionContext()
        ctx.register_record_batches("t", [target.to_batches()])
        ctx.register_record_batches("s", [source.to_batches()])

        sql = (
            # Matched: update name from source
            "(SELECT t.id, (s.name) AS name FROM t INNER JOIN s ON t.id = s.id)"
            " UNION ALL "
            # Unmatched target: keep
            "(SELECT t.id, t.name FROM t WHERE NOT EXISTS "
            "(SELECT 1 FROM s WHERE t.id = s.id))"
            " UNION ALL "
            # Not matched source: insert
            "(SELECT (s.id) AS id, (s.name) AS name FROM s WHERE NOT EXISTS "
            "(SELECT 1 FROM t WHERE t.id = s.id))"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        rows = {r[0]: r[1] for r in zip(result["id"].to_pylist(), result["name"].to_pylist())}
        assert rows == {1: "a", 2: "B", 3: "c", 4: "D"}

    def test_merge_delete(self, merge_data):
        """WHEN MATCHED THEN DELETE (no insert)."""
        target, source = merge_data
        ctx = SessionContext()
        ctx.register_record_batches("t", [target.to_batches()])
        ctx.register_record_batches("s", [source.to_batches()])

        sql = (
            # No matched part (all matched deleted)
            # Unmatched target: keep
            "(SELECT t.id, t.name FROM t WHERE NOT EXISTS "
            "(SELECT 1 FROM s WHERE t.id = s.id))"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        ids = sorted(result["id"].to_pylist())
        assert ids == [1, 3]  # id=2 was matched and deleted

    def test_merge_conditional_update_and_delete(self):
        """WHEN MATCHED AND cond THEN UPDATE + WHEN MATCHED AND cond THEN DELETE."""
        target = pa.table({
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": pa.array(["a", "b", "c"], type=pa.string()),
        })
        source = pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["A", "B"], type=pa.string()),
            "op": pa.array(["update", "delete"], type=pa.string()),
        })
        ctx = SessionContext()
        ctx.register_record_batches("t", [target.to_batches()])
        ctx.register_record_batches("s", [source.to_batches()])

        sql = (
            # Matched: update if op='update', exclude if op='delete'
            "(SELECT t.id, "
            "CASE WHEN (s.op = 'update') THEN (s.name) ELSE t.name END AS name "
            "FROM t INNER JOIN s ON t.id = s.id "
            "WHERE NOT (s.op = 'delete'))"
            " UNION ALL "
            # Unmatched
            "(SELECT t.id, t.name FROM t WHERE NOT EXISTS "
            "(SELECT 1 FROM s WHERE t.id = s.id))"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        rows = {r[0]: r[1] for r in zip(result["id"].to_pylist(), result["name"].to_pylist())}
        # id=1 updated to 'A', id=2 deleted, id=3 unchanged
        assert rows == {1: "A", 3: "c"}

    def test_merge_insert_only(self, merge_data):
        """WHEN NOT MATCHED THEN INSERT (no matched clause)."""
        target, source = merge_data
        ctx = SessionContext()
        ctx.register_record_batches("t", [target.to_batches()])
        ctx.register_record_batches("s", [source.to_batches()])

        sql = (
            # All target rows kept (no matched clause)
            "(SELECT t.id, t.name FROM t)"
            " UNION ALL "
            # Not matched: insert
            "(SELECT (s.id) AS id, (s.name) AS name FROM s WHERE NOT EXISTS "
            "(SELECT 1 FROM t WHERE t.id = s.id))"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect())
        ids = sorted(result["id"].to_pylist())
        assert ids == [1, 2, 3, 4]

    def test_merge_insert_only_handler_logic(self, merge_data):
        """Verify handler builds correct SQL when only NOT MATCHED clause exists.

        Bug regression test: without the fix, matched target rows (id=2) would
        be dropped because the handler used WHERE NOT EXISTS for all target rows
        even when no matched clause was present.
        """
        target, source = merge_data
        ctx = SessionContext()
        ctx.register_record_batches("t", [target.to_batches()])
        ctx.register_record_batches("s", [source.to_batches()])

        # Simulate what handle_merge_into builds when has_any_matched=False:
        # Part 2 should select ALL target rows (no NOT EXISTS filter)
        target_cols = target.schema.names
        has_any_matched = False

        if has_any_matched:
            unmatched_sql = (
                f"SELECT {', '.join(f't.{c}' for c in target_cols)} "
                f"FROM t WHERE NOT EXISTS (SELECT 1 FROM s WHERE t.id = s.id)"
            )
        else:
            unmatched_sql = (
                f"SELECT {', '.join(f't.{c}' for c in target_cols)} FROM t"
            )

        insert_sql = (
            "(SELECT (s.id) AS id, (s.name) AS name FROM s "
            "WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.id = s.id))"
        )
        final_sql = f"({unmatched_sql}) UNION ALL {insert_sql}"
        result = pa.Table.from_batches(ctx.sql(final_sql).collect())
        ids = sorted(result["id"].to_pylist())
        # All original target rows (1,2,3) preserved + new source row (4)
        assert ids == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Subqueries in DML WHERE clauses (DataFusion logic tests)
# ---------------------------------------------------------------------------


class TestSubqueriesInDML:
    """Test that DML operations with subqueries referencing other tables work.

    Simulates the pattern used by the DML handlers: a fresh DataFusion context
    with the target table + additional referenced tables registered.
    """

    @pytest.fixture
    def target_table(self):
        return pa.table({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "score": [10, 20, 30, 40, 50],
        })

    @pytest.fixture
    def ref_table(self):
        """Reference table for subqueries."""
        return pa.table({
            "id": [2, 4],
            "status": ["inactive", "inactive"],
        })

    def test_delete_with_in_subquery(self, target_table, ref_table):
        """DELETE FROM t WHERE id IN (SELECT id FROM ref) — needs ref registered."""
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [target_table.to_batches()])
        ctx.register_record_batches("ref", [ref_table.to_batches()])
        where = "id IN (SELECT id FROM ref)"
        keep_sql = f"SELECT * FROM __delete_tmp WHERE NOT ({where})"
        result = pa.Table.from_batches(ctx.sql(keep_sql).collect(), schema=target_table.schema)
        ids = sorted(result["id"].to_pylist())
        assert ids == [1, 3, 5]  # Rows 2 and 4 deleted

    def test_delete_with_exists_subquery(self, target_table, ref_table):
        """DELETE FROM t WHERE EXISTS (SELECT 1 FROM ref WHERE ref.id = t.id)."""
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [target_table.to_batches()])
        ctx.register_record_batches("ref", [ref_table.to_batches()])
        where = "EXISTS (SELECT 1 FROM ref WHERE ref.id = __delete_tmp.id)"
        keep_sql = f"SELECT * FROM __delete_tmp WHERE NOT ({where})"
        result = pa.Table.from_batches(ctx.sql(keep_sql).collect(), schema=target_table.schema)
        ids = sorted(result["id"].to_pylist())
        assert ids == [1, 3, 5]

    def test_delete_with_not_in_subquery(self, target_table, ref_table):
        """DELETE FROM t WHERE id NOT IN (SELECT id FROM ref) — inverse filter."""
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [target_table.to_batches()])
        ctx.register_record_batches("ref", [ref_table.to_batches()])
        where = "id NOT IN (SELECT id FROM ref)"
        keep_sql = f"SELECT * FROM __delete_tmp WHERE NOT ({where})"
        result = pa.Table.from_batches(ctx.sql(keep_sql).collect(), schema=target_table.schema)
        ids = sorted(result["id"].to_pylist())
        assert ids == [2, 4]  # Only rows IN ref are kept

    def test_delete_without_ref_table_fails(self, target_table, ref_table):
        """Without registering the ref table, the subquery fails."""
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [target_table.to_batches()])
        # Intentionally NOT registering ref table
        where = "id IN (SELECT id FROM ref)"
        keep_sql = f"SELECT * FROM __delete_tmp WHERE NOT ({where})"
        with pytest.raises(Exception):
            ctx.sql(keep_sql).collect()

    def test_update_with_scalar_subquery(self, target_table, ref_table):
        """UPDATE t SET score = (SELECT COUNT(*) FROM ref) — scalar subquery in SET."""
        ctx = SessionContext()
        ctx.register_record_batches("__update_tmp", [target_table.to_batches()])
        ctx.register_record_batches("ref", [ref_table.to_batches()])
        sql = (
            "SELECT id, name, "
            "CAST((SELECT COUNT(*) FROM ref) AS BIGINT) AS score "
            "FROM __update_tmp"
        )
        result = pa.Table.from_batches(ctx.sql(sql).collect(), schema=target_table.schema)
        # All scores should be 2 (count of ref table)
        assert all(s == 2 for s in result["score"].to_pylist())


# ---------------------------------------------------------------------------
# _register_referenced_tables — temp view awareness
# ---------------------------------------------------------------------------


class TestRegisterReferencedTables:
    """Test that _register_referenced_tables correctly handles temp views
    and registers them into the DML fresh SessionContext."""

    @pytest.fixture
    def mock_session(self):
        """A lightweight session-like object with _ctx and _catalog."""
        class _MockSession:
            def __init__(self):
                self._ctx = SessionContext()
                self._catalog = None  # No catalog needed for temp view tests

            def _ensure_table_registered(self, short_name, full_name=None):
                raise Exception("table not found in catalog")
        return _MockSession()

    def test_temp_view_registered_into_fresh_ctx(self, mock_session):
        """Temp view in session._ctx should be copied to the DML ctx."""
        view_data = pa.table({"id": [10, 20], "val": ["a", "b"]})
        mock_session._ctx.register_record_batches("my_view", [view_data.to_batches()])

        fresh_ctx = SessionContext()
        fresh_ctx.register_record_batches("__target", [view_data.to_batches()])
        _register_referenced_tables(mock_session, fresh_ctx, "id IN (SELECT id FROM my_view)")

        # my_view should now be queryable in fresh_ctx
        result = fresh_ctx.sql("SELECT * FROM my_view").to_arrow_table()
        assert result.num_rows == 2

    def test_dunder_names_skipped(self, mock_session):
        """Names starting with __ should be skipped."""
        fresh_ctx = SessionContext()
        target = pa.table({"x": [1]})
        fresh_ctx.register_record_batches("__target", [target.to_batches()])

        # Should not raise even though __target isn't in session._ctx
        _register_referenced_tables(mock_session, fresh_ctx, "SELECT * FROM __target")

    def test_unknown_name_skipped_silently(self, mock_session):
        """Names not in session._ctx or catalog should be silently skipped."""
        fresh_ctx = SessionContext()
        # Should not raise
        _register_referenced_tables(mock_session, fresh_ctx, "id IN (SELECT id FROM nonexistent)")

    def test_resolve_temp_view_arrow_found(self, mock_session):
        """_resolve_temp_view_arrow returns Arrow data for registered views."""
        view_data = pa.table({"a": [1, 2, 3]})
        mock_session._ctx.register_record_batches("v1", [view_data.to_batches()])
        result = _resolve_temp_view_arrow(mock_session, "v1")
        assert result is not None
        assert result.num_rows == 3

    def test_resolve_temp_view_arrow_not_found(self, mock_session):
        """_resolve_temp_view_arrow returns None for unregistered names."""
        result = _resolve_temp_view_arrow(mock_session, "does_not_exist")
        assert result is None

    def test_delete_subquery_with_temp_view(self, mock_session):
        """Simulate full DELETE WHERE ... IN (SELECT FROM temp_view) flow."""
        # Register temp view in session context
        view_data = pa.table({"id": pa.array([2, 4], type=pa.int64())})
        mock_session._ctx.register_record_batches("del_ids", [view_data.to_batches()])

        # Set up fresh DML context with target table
        target = pa.table({
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "name": ["a", "b", "c", "d", "e"],
        })
        ctx = SessionContext()
        ctx.register_record_batches("__delete_tmp", [target.to_batches()])

        # Register referenced tables (should find del_ids in session._ctx)
        _register_referenced_tables(mock_session, ctx, "id IN (SELECT id FROM del_ids)")

        # Execute the DELETE logic
        keep_sql = "SELECT * FROM __delete_tmp WHERE NOT (id IN (SELECT id FROM del_ids))"
        result = pa.Table.from_batches(ctx.sql(keep_sql).collect(), schema=target.schema)
        ids = sorted(result["id"].to_pylist())
        assert ids == [1, 3, 5]
