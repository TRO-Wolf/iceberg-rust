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
    "crates/catalog/glue/src/catalog.rs": 1256,
    "crates/catalog/hms/src/catalog.rs": 1063,
    "crates/catalog/rest/src/catalog.rs": 5230,
    "crates/catalog/rest/src/client.rs": 1726,
    "crates/catalog/rest/src/types.rs": 1289,
    "crates/catalog/s3tables/src/catalog.rs": 1635,
    "crates/catalog/sql/src/catalog.rs": 4214,
    "crates/iceberg/src/arrow/avro_reader.rs": 1256,
    "crates/iceberg/src/arrow/avro_reader_tests.rs": 1097,
    "crates/iceberg/src/arrow/caching_delete_file_loader.rs": 3487,
    "crates/iceberg/src/arrow/delete_filter.rs": 3229,
    "crates/iceberg/src/arrow/reader.rs": 10826,
    "crates/iceberg/src/arrow/record_batch_transformer.rs": 2816,
    "crates/iceberg/src/arrow/schema.rs": 3790,
    "crates/iceberg/src/arrow/value.rs": 2165,
    "crates/iceberg/src/avro/schema.rs": 2095,
    "crates/iceberg/src/catalog/memory/catalog.rs": 3444,
    "crates/iceberg/src/catalog/mod.rs": 3172,
    "crates/iceberg/src/delete_file_index.rs": 2311,
    "crates/iceberg/src/delete_vector.rs": 1608,
    "crates/iceberg/src/expr/expression_parser.rs": 1393,
    "crates/iceberg/src/expr/predicate.rs": 2934,
    "crates/iceberg/src/expr/visitors/aggregate_evaluator.rs": 1117,
    "crates/iceberg/src/expr/visitors/inclusive_metrics_evaluator.rs": 2191,
    "crates/iceberg/src/expr/visitors/manifest_evaluator.rs": 1755,
    "crates/iceberg/src/expr/visitors/page_index_evaluator.rs": 1365,
    "crates/iceberg/src/expr/visitors/residual_evaluator.rs": 1730,
    "crates/iceberg/src/expr/visitors/row_group_metrics_evaluator.rs": 1941,
    "crates/iceberg/src/expr/visitors/strict_metrics_evaluator.rs": 1928,
    "crates/iceberg/src/expr/visitors/strict_projection.rs": 3234,
    "crates/iceberg/src/inspect/data_file.rs": 1353,
    "crates/iceberg/src/inspect/entries.rs": 1281,
    "crates/iceberg/src/inspect/files.rs": 1829,
    "crates/iceberg/src/inspect/partitions.rs": 1508,
    "crates/iceberg/src/io/storage/local_fs.rs": 1063,
    "crates/iceberg/src/maintenance/compute_table_stats.rs": 1333,
    "crates/iceberg/src/maintenance/partition_stats.rs": 5188,
    "crates/iceberg/src/maintenance/remove_dangling_delete_files.rs": 1945,
    "crates/iceberg/src/maintenance/rewrite_data_files.rs": 3287,
    "crates/iceberg/src/maintenance/rewrite_position_delete_files.rs": 1581,
    "crates/iceberg/src/maintenance/rewrite_position_delete_files_tests.rs": 4908,
    "crates/iceberg/src/maintenance/rewrite_table_path.rs": 1040,
    "crates/iceberg/src/maintenance/tests.rs": 1598,
    "crates/iceberg/src/puffin/metadata.rs": 1125,
    "crates/iceberg/src/scan/incremental.rs": 3343,
    "crates/iceberg/src/scan/mod.rs": 7167,
    "crates/iceberg/src/scan/task.rs": 1970,
    "crates/iceberg/src/spec/datatypes.rs": 2122,
    "crates/iceberg/src/spec/manifest/mod.rs": 1271,
    "crates/iceberg/src/spec/manifest/writer.rs": 1093,
    "crates/iceberg/src/spec/manifest_list.rs": 2355,
    "crates/iceberg/src/spec/partition.rs": 3673,
    "crates/iceberg/src/spec/partitioning.rs": 1055,
    "crates/iceberg/src/spec/schema/id_reassigner.rs": 1865,
    "crates/iceberg/src/spec/schema/mod.rs": 2048,
    "crates/iceberg/src/spec/snapshot.rs": 1056,
    "crates/iceberg/src/spec/snapshot_summary.rs": 1662,
    "crates/iceberg/src/spec/table_metadata.rs": 4654,
    "crates/iceberg/src/spec/table_metadata_builder.rs": 4324,
    "crates/iceberg/src/spec/transform.rs": 1283,
    "crates/iceberg/src/spec/values/datum.rs": 1505,
    "crates/iceberg/src/spec/values/tests.rs": 2495,
    "crates/iceberg/src/spec/view_metadata_builder.rs": 1661,
    "crates/iceberg/src/transaction/append.rs": 1576,
    "crates/iceberg/src/transaction/cherry_pick.rs": 2202,
    "crates/iceberg/src/transaction/delete_files.rs": 2577,
    "crates/iceberg/src/transaction/expire_cleanup.rs": 2062,
    "crates/iceberg/src/transaction/expire_snapshots.rs": 1459,
    "crates/iceberg/src/transaction/manage_snapshots.rs": 1254,
    "crates/iceberg/src/transaction/merge_append.rs": 1727,
    "crates/iceberg/src/transaction/mod.rs": 2074,
    "crates/iceberg/src/transaction/overwrite_files.rs": 3928,
    "crates/iceberg/src/transaction/replace_partitions.rs": 2903,
    "crates/iceberg/src/transaction/rewrite_files.rs": 2760,
    "crates/iceberg/src/transaction/rewrite_manifests.rs": 1953,
    "crates/iceberg/src/transaction/row_delta.rs": 7331,
    "crates/iceberg/src/transaction/snapshot.rs": 3948,
    "crates/iceberg/src/transaction/staged_table.rs": 1229,
    "crates/iceberg/src/transaction/update_partition_spec.rs": 1263,
    "crates/iceberg/src/transaction/update_schema.rs": 3718,
    "crates/iceberg/src/transform/bucket.rs": 1214,
    "crates/iceberg/src/transform/temporal.rs": 2796,
    "crates/iceberg/src/variant/tests.rs": 3338,
    "crates/iceberg/src/variant/write.rs": 1206,
    "crates/iceberg/src/writer/base_writer/deletion_vector_writer.rs": 1554,
    "crates/iceberg/src/writer/base_writer/equality_delete_writer.rs": 1053,
    "crates/iceberg/src/writer/base_writer/position_delete_writer.rs": 1486,
    "crates/iceberg/src/writer/file_writer/avro_writer.rs": 1422,
    "crates/iceberg/src/writer/file_writer/parquet_writer.rs": 3541,
    "crates/iceberg/tests/interop_inspection_manifests.rs": 2424,
    "crates/iceberg/tests/interop_partition_stats.rs": 2806,
    "crates/iceberg/tests/interop_remove_dangling.rs": 1079,
    "crates/iceberg/tests/interop_scan_exec.rs": 2818,
    "crates/iceberg/tests/interop_scan_plan.rs": 1174,
    "crates/iceberg/tests/interop_write_data.rs": 2357,
    "crates/integrations/datafusion/src/catalog.rs": 1775,
    "crates/integrations/datafusion/src/physical_plan/commit.rs": 1245,
    "crates/integrations/datafusion/src/physical_plan/delete.rs": 2832,
    "crates/integrations/datafusion/src/physical_plan/project.rs": 1506,
    "crates/integrations/datafusion/src/physical_plan/scan.rs": 2161,
    "crates/integrations/datafusion/src/table/mod.rs": 2488,
    "crates/integrations/datafusion/tests/integration_datafusion_test.rs": 7159,
    "crates/sketches/src/theta.rs": 1137,
    "crates/storage/opendal/src/lib.rs": 2162,
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
