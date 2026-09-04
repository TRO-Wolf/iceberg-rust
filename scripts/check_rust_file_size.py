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

"""Enforce line ceilings over every Rust source file under ``crates/``.

The default applies to new and currently compliant files. Legacy ceilings
freeze existing debt and only move down when a file is split or shortened.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping


DEFAULT_CEILING = 1000

# Measured on 2026-08-26. These exact ceilings keep the inherited tree green
# without permitting more legacy file-size debt. Delete a row once its file is
# at or below DEFAULT_CEILING. Keep keys sorted.
LEGACY_CEILINGS: dict[str, int] = {
    "crates/catalog/glue/src/catalog.rs": 1212,
    "crates/catalog/hms/src/catalog.rs": 1063,
    "crates/catalog/rest/src/catalog.rs": 4970,
    "crates/catalog/rest/src/client.rs": 1483,
    "crates/catalog/rest/src/types.rs": 1128,
    "crates/catalog/s3tables/src/catalog.rs": 1403,
    "crates/catalog/sql/src/catalog.rs": 3947,
    "crates/iceberg/src/arrow/avro_reader.rs": 1256,
    "crates/iceberg/src/arrow/avro_reader_tests.rs": 1097,
    "crates/iceberg/src/arrow/caching_delete_file_loader.rs": 3270,
    "crates/iceberg/src/arrow/delete_filter.rs": 2938,
    "crates/iceberg/src/arrow/reader.rs": 10246,
    "crates/iceberg/src/arrow/record_batch_transformer.rs": 2475,
    "crates/iceberg/src/arrow/schema.rs": 3711,
    "crates/iceberg/src/arrow/value.rs": 2165,
    "crates/iceberg/src/avro/schema.rs": 2095,
    "crates/iceberg/src/catalog/memory/catalog.rs": 3401,
    "crates/iceberg/src/catalog/mod.rs": 3074,
    "crates/iceberg/src/delete_file_index.rs": 2229,
    "crates/iceberg/src/delete_vector.rs": 1476,
    "crates/iceberg/src/expr/expression_parser.rs": 1267,
    "crates/iceberg/src/expr/predicate.rs": 2676,
    "crates/iceberg/src/expr/visitors/aggregate_evaluator.rs": 1117,
    "crates/iceberg/src/expr/visitors/inclusive_metrics_evaluator.rs": 2191,
    "crates/iceberg/src/expr/visitors/manifest_evaluator.rs": 1753,
    "crates/iceberg/src/expr/visitors/page_index_evaluator.rs": 1365,
    "crates/iceberg/src/expr/visitors/residual_evaluator.rs": 1641,
    "crates/iceberg/src/expr/visitors/row_group_metrics_evaluator.rs": 1941,
    "crates/iceberg/src/expr/visitors/strict_metrics_evaluator.rs": 1928,
    "crates/iceberg/src/expr/visitors/strict_projection.rs": 3234,
    "crates/iceberg/src/inspect/data_file.rs": 1353,
    "crates/iceberg/src/inspect/entries.rs": 1281,
    "crates/iceberg/src/inspect/files.rs": 1785,
    "crates/iceberg/src/inspect/partitions.rs": 1508,
    "crates/iceberg/src/io/storage/local_fs.rs": 1063,
    "crates/iceberg/src/maintenance/compute_table_stats.rs": 1248,
    "crates/iceberg/src/maintenance/partition_stats.rs": 4665,
    "crates/iceberg/src/maintenance/remove_dangling_delete_files.rs": 1804,
    "crates/iceberg/src/maintenance/rewrite_data_files.rs": 2659,
    "crates/iceberg/src/maintenance/rewrite_position_delete_files_tests.rs": 4716,
    "crates/iceberg/src/maintenance/tests.rs": 1524,
    "crates/iceberg/src/puffin/metadata.rs": 1125,
    "crates/iceberg/src/scan/incremental.rs": 2978,
    "crates/iceberg/src/scan/mod.rs": 6892,
    "crates/iceberg/src/scan/task.rs": 1674,
    "crates/iceberg/src/spec/datatypes.rs": 2039,
    "crates/iceberg/src/spec/manifest/mod.rs": 1266,
    "crates/iceberg/src/spec/manifest/writer.rs": 1058,
    "crates/iceberg/src/spec/manifest_list.rs": 2331,
    "crates/iceberg/src/spec/partition.rs": 3501,
    "crates/iceberg/src/spec/partitioning.rs": 1055,
    "crates/iceberg/src/spec/schema/id_reassigner.rs": 1669,
    "crates/iceberg/src/spec/schema/mod.rs": 2048,
    "crates/iceberg/src/spec/snapshot.rs": 1056,
    "crates/iceberg/src/spec/snapshot_summary.rs": 1660,
    "crates/iceberg/src/spec/table_metadata.rs": 4575,
    "crates/iceberg/src/spec/table_metadata_builder.rs": 4105,
    "crates/iceberg/src/spec/transform.rs": 1283,
    "crates/iceberg/src/spec/values/datum.rs": 1321,
    "crates/iceberg/src/spec/values/tests.rs": 2367,
    "crates/iceberg/src/spec/view_metadata_builder.rs": 1661,
    "crates/iceberg/src/transaction/append.rs": 1542,
    "crates/iceberg/src/transaction/cherry_pick.rs": 2132,
    "crates/iceberg/src/transaction/delete_files.rs": 2261,
    "crates/iceberg/src/transaction/expire_cleanup.rs": 1857,
    "crates/iceberg/src/transaction/expire_snapshots.rs": 1388,
    "crates/iceberg/src/transaction/manage_snapshots.rs": 1254,
    "crates/iceberg/src/transaction/merge_append.rs": 1662,
    "crates/iceberg/src/transaction/mod.rs": 1948,
    "crates/iceberg/src/transaction/overwrite_files.rs": 3429,
    "crates/iceberg/src/transaction/replace_partitions.rs": 2801,
    "crates/iceberg/src/transaction/rewrite_files.rs": 2461,
    "crates/iceberg/src/transaction/rewrite_manifests.rs": 1915,
    "crates/iceberg/src/transaction/row_delta.rs": 6366,
    "crates/iceberg/src/transaction/snapshot.rs": 3490,
    "crates/iceberg/src/transaction/staged_table.rs": 1229,
    "crates/iceberg/src/transaction/update_partition_spec.rs": 1263,
    "crates/iceberg/src/transaction/update_schema.rs": 3606,
    "crates/iceberg/src/transform/bucket.rs": 1214,
    "crates/iceberg/src/transform/temporal.rs": 2796,
    "crates/iceberg/src/variant/tests.rs": 3262,
    "crates/iceberg/src/variant/write.rs": 1090,
    "crates/iceberg/src/writer/base_writer/deletion_vector_writer.rs": 1400,
    "crates/iceberg/src/writer/base_writer/equality_delete_writer.rs": 1032,
    "crates/iceberg/src/writer/file_writer/avro_writer.rs": 1422,
    "crates/iceberg/src/writer/file_writer/parquet_writer.rs": 3391,
    "crates/iceberg/tests/interop_inspection_manifests.rs": 2289,
    "crates/iceberg/tests/interop_partition_stats.rs": 2605,
    "crates/iceberg/tests/interop_remove_dangling.rs": 1021,
    "crates/iceberg/tests/interop_scan_exec.rs": 2592,
    "crates/iceberg/tests/interop_scan_plan.rs": 1028,
    "crates/iceberg/tests/interop_write_data.rs": 2115,
    "crates/integrations/datafusion/src/catalog.rs": 1589,
    "crates/integrations/datafusion/src/physical_plan/delete.rs": 1164,
    "crates/integrations/datafusion/src/physical_plan/project.rs": 1506,
    "crates/integrations/datafusion/src/physical_plan/scan.rs": 1999,
    "crates/integrations/datafusion/src/table/mod.rs": 2251,
    "crates/integrations/datafusion/tests/integration_datafusion_test.rs": 6874,
    "crates/sketches/src/theta.rs": 1024,
    "crates/storage/opendal/src/lib.rs": 2080,
}


@dataclass(frozen=True)
class CheckResult:
    """Hold the number of scanned files and every detected violation."""

    checked: int
    errors: tuple[str, ...]
    is_configuration_error: bool = False


def count_lines(path: Path) -> int:
    """Read one UTF-8 source file and return its logical line count."""

    return len(path.read_text(encoding="utf-8").splitlines())


def check_file(
    path: Path,
    repo: Path,
    line_count: int,
    legacy_ceilings: Mapping[str, int],
) -> list[str]:
    """Check one source file and return its file-size violations."""

    relative_path = path.relative_to(repo).as_posix()
    ceiling = legacy_ceilings.get(relative_path, DEFAULT_CEILING)
    if line_count <= ceiling:
        return []

    return [
        f"ERROR: {relative_path} is {line_count} lines (ceiling {ceiling}). "
        "Split the file, or lower another legacy ceiling after extracting code. "
        "Do not raise a ceiling."
    ]


def check_repository(
    repo: Path,
    legacy_ceilings: Mapping[str, int] = LEGACY_CEILINGS,
) -> CheckResult:
    """Check every ``crates/**/*.rs`` file and fail closed on invalid scan state."""

    crates_root = repo / "crates"
    if not crates_root.is_dir():
        return CheckResult(0, ("ERROR: crates/ not found",), True)

    configuration_errors: list[str] = []
    invalid_legacy_paths: set[str] = set()
    for relative_path, ceiling in sorted(legacy_ceilings.items()):
        posix_path = PurePosixPath(relative_path)
        if (
            posix_path.is_absolute()
            or ".." in posix_path.parts
            or not posix_path.parts
            or posix_path.parts[0] != "crates"
            or posix_path.suffix != ".rs"
        ):
            configuration_errors.append(
                f"ERROR: invalid legacy ceiling path: {relative_path}"
            )
            invalid_legacy_paths.add(relative_path)
        legacy_path = repo / relative_path
        if ceiling <= DEFAULT_CEILING:
            configuration_errors.append(
                f"ERROR: legacy ceiling for {relative_path} is {ceiling}; "
                f"remove it because the default is {DEFAULT_CEILING}"
            )
            invalid_legacy_paths.add(relative_path)
        if not legacy_path.is_file():
            configuration_errors.append(
                f"ERROR: legacy ceiling path has no file on disk: {relative_path}"
            )
            invalid_legacy_paths.add(relative_path)

    paths = sorted(path for path in crates_root.rglob("*.rs") if path.is_file())
    if not paths:
        configuration_errors.append(
            "ERROR: crates/**/*.rs scan set is empty; refusing to pass"
        )
        return CheckResult(0, tuple(configuration_errors), True)

    line_counts: dict[str, int] = {}
    for path in paths:
        relative_path = path.relative_to(repo).as_posix()
        try:
            line_counts[relative_path] = count_lines(path)
        except (OSError, UnicodeError) as error:
            configuration_errors.append(
                f"ERROR: {relative_path}: unreadable UTF-8 source ({error})"
            )

    for relative_path, ceiling in sorted(legacy_ceilings.items()):
        if relative_path in invalid_legacy_paths or relative_path not in line_counts:
            continue
        line_count = line_counts[relative_path]
        if line_count <= DEFAULT_CEILING:
            configuration_errors.append(
                f"ERROR: legacy ceiling for {relative_path} is obsolete; the file is "
                f"{line_count} lines, within the default. Remove the row."
            )
            invalid_legacy_paths.add(relative_path)
        elif ceiling > line_count:
            configuration_errors.append(
                f"ERROR: legacy ceiling for {relative_path} is {ceiling}, but the file "
                f"shrunk to {line_count}. Lower the ceiling to {line_count}."
            )
            invalid_legacy_paths.add(relative_path)
        elif ceiling < line_count:
            configuration_errors.append(
                f"ERROR: {relative_path} grew from its legacy ceiling {ceiling} to "
                f"{line_count} lines. Split the file; do not raise the ceiling."
            )
            invalid_legacy_paths.add(relative_path)

    errors = list(configuration_errors)
    for path in paths:
        relative_path = path.relative_to(repo).as_posix()
        if relative_path in invalid_legacy_paths or relative_path not in line_counts:
            continue
        errors.extend(
            check_file(path, repo, line_counts[relative_path], legacy_ceilings)
        )

    return CheckResult(len(paths), tuple(errors), bool(configuration_errors))


def main() -> int:
    """Run the repository check and return a process exit status."""

    repo = Path(__file__).resolve().parent.parent
    result = check_repository(repo)
    if result.errors:
        for error in result.errors:
            print(error, file=sys.stderr)
        print(
            f"rust-file-size: FAIL — {len(result.errors)} violation(s) "
            f"across {result.checked} files",
            file=sys.stderr,
        )
        return 2 if result.is_configuration_error else 1

    print(
        f"rust-file-size: {result.checked} files clean "
        f"({len(LEGACY_CEILINGS)} legacy ceilings)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
