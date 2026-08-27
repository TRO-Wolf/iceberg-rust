#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Regression tests for the Rust source-file size gate."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_rust_file_size import DEFAULT_CEILING, check_repository


class RustFileSizeCheckTests(unittest.TestCase):
    """Pin the gate boundary and every fail-closed scan condition."""

    def make_repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        """Create a temporary repository with an empty ``crates`` directory."""

        temporary_directory = tempfile.TemporaryDirectory()
        repo = Path(temporary_directory.name)
        (repo / "crates").mkdir()
        return temporary_directory, repo

    def write_rust_file(self, repo: Path, relative_path: str, lines: int) -> Path:
        """Write a Rust fixture with the requested logical line count."""

        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// line\n" * lines, encoding="utf-8")
        return path

    def test_file_at_default_ceiling_passes(self) -> None:
        """Pin the inclusive default boundary so 1,000 lines remain legal."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        self.write_rust_file(repo, "crates/exact.rs", DEFAULT_CEILING)

        result = check_repository(repo, {})

        self.assertEqual(1, result.checked)
        self.assertEqual((), result.errors)

    def test_file_above_default_ceiling_fails_with_actionable_error(self) -> None:
        """Reject a new file as soon as it crosses the default ceiling."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        self.write_rust_file(repo, "crates/too_large.rs", DEFAULT_CEILING + 1)

        result = check_repository(repo, {})

        self.assertEqual(1, len(result.errors))
        self.assertIn("crates/too_large.rs is 1001 lines", result.errors[0])
        self.assertIn("ceiling 1000", result.errors[0])
        self.assertIn("Do not raise a ceiling", result.errors[0])

    def test_legacy_ceiling_freezes_existing_overage(self) -> None:
        """Allow the measured legacy size but reject one additional line."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        path = self.write_rust_file(repo, "crates/legacy.rs", DEFAULT_CEILING + 2)
        legacy_ceilings = {"crates/legacy.rs": DEFAULT_CEILING + 2}

        clean_result = check_repository(repo, legacy_ceilings)
        path.write_text("// line\n" * (DEFAULT_CEILING + 3), encoding="utf-8")
        grown_result = check_repository(repo, legacy_ceilings)

        self.assertEqual((), clean_result.errors)
        self.assertEqual(1, len(grown_result.errors))
        self.assertIn("grew from its legacy ceiling 1002", grown_result.errors[0])

    def test_legacy_ceiling_must_drop_after_file_shrinks(self) -> None:
        """Reject stale headroom so a shortened file cannot regrow later."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        path = self.write_rust_file(repo, "crates/legacy.rs", DEFAULT_CEILING + 1)

        stale_result = check_repository(
            repo, {"crates/legacy.rs": DEFAULT_CEILING + 2}
        )
        lowered_result = check_repository(
            repo, {"crates/legacy.rs": DEFAULT_CEILING + 1}
        )
        path.write_text("// line\n" * (DEFAULT_CEILING + 2), encoding="utf-8")
        regrown_result = check_repository(
            repo, {"crates/legacy.rs": DEFAULT_CEILING + 1}
        )

        self.assertTrue(stale_result.is_configuration_error)
        self.assertIn("Lower the ceiling to 1001", stale_result.errors[0])
        self.assertEqual((), lowered_result.errors)
        self.assertIn(
            "grew from its legacy ceiling 1001 to 1002", regrown_result.errors[0]
        )

    def test_stale_legacy_path_fails_closed(self) -> None:
        """Reject a ceiling row that no longer names an on-disk file."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        self.write_rust_file(repo, "crates/live.rs", 1)

        result = check_repository(repo, {"crates/missing.rs": DEFAULT_CEILING + 1})

        self.assertTrue(result.is_configuration_error)
        self.assertEqual(1, len(result.errors))
        self.assertIn("legacy ceiling path has no file", result.errors[0])

    def test_legacy_path_outside_crates_fails_closed(self) -> None:
        """Reject a ceiling row that escapes the recursive scan root."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        self.write_rust_file(repo, "crates/live.rs", 1)
        (repo / "outside.rs").write_text("// line\n", encoding="utf-8")

        result = check_repository(repo, {"outside.rs": DEFAULT_CEILING + 1})

        self.assertTrue(result.is_configuration_error)
        self.assertEqual(1, len(result.errors))
        self.assertIn("invalid legacy ceiling path", result.errors[0])

    def test_redundant_legacy_ceiling_fails_closed(self) -> None:
        """Reject a legacy row that does not exceed the default."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        self.write_rust_file(repo, "crates/small.rs", DEFAULT_CEILING)

        result = check_repository(repo, {"crates/small.rs": DEFAULT_CEILING + 2})

        self.assertEqual(1, len(result.errors))
        self.assertIn("within the default. Remove the row", result.errors[0])

    def test_empty_scan_set_fails_closed(self) -> None:
        """Reject a repository where the recursive Rust scan finds no files."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)

        result = check_repository(repo, {})

        self.assertTrue(result.is_configuration_error)
        self.assertEqual(0, result.checked)
        self.assertIn("scan set is empty", result.errors[0])

    def test_missing_crates_directory_fails_closed(self) -> None:
        """Reject a repository that has no Rust source scan root."""

        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        repo = Path(temporary_directory.name)

        result = check_repository(repo, {})

        self.assertTrue(result.is_configuration_error)
        self.assertEqual(0, result.checked)
        self.assertEqual(("ERROR: crates/ not found",), result.errors)

    def test_unreadable_source_fails_closed(self) -> None:
        """Reject a source file whose contents cannot be read."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        self.write_rust_file(repo, "crates/unreadable.rs", 1)

        with patch("check_rust_file_size.Path.read_text", side_effect=OSError("denied")):
            result = check_repository(repo, {})

        self.assertEqual(1, len(result.errors))
        self.assertIn("unreadable UTF-8 source", result.errors[0])

    def test_invalid_utf8_source_fails_closed(self) -> None:
        """Reject a Rust file that cannot be decoded as UTF-8."""

        temporary_directory, repo = self.make_repo()
        self.addCleanup(temporary_directory.cleanup)
        path = repo / "crates/invalid.rs"
        path.write_bytes(b"\xff\n")

        result = check_repository(repo, {})

        self.assertEqual(1, len(result.errors))
        self.assertIn("unreadable UTF-8 source", result.errors[0])


if __name__ == "__main__":
    unittest.main()
