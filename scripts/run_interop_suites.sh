#!/bin/bash
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
#
# run_interop_suites.sh — the nightly Java-interop DRIVER (2026-07-10, bundle unit G3).
#
# The Java oracle under dev/java-interop/ is the parity mandate's ground truth, but until this
# driver existed every suite ran only when a human remembered to invoke it. This script turns the
# suite set into a standing regression net: it dynamically DISCOVERS every
# dev/java-interop/run-interop-*.sh at run time (zero maintenance when suites are added), runs
# EACH one to completion (continue-on-failure ACROSS suites, so one red suite cannot hide the
# others), prints a per-suite PASS/FAIL summary table, and exits non-zero if ANY suite failed.
# Invoked nightly by .github/workflows/nightly_interop.yml via `make interop`.
#
# House doctrine encoded here (CLAUDE.md working conventions + the G3 brief):
#   * HARD-FAIL, never SKIP. Missing prerequisites (mvn / the JDK-11 home / cargo) are an error
#     before anything runs — a skipped suite is a false-green.
#   * NO SILENT CAPS. There is no per-suite timeout (a hung suite is bounded only by the CI
#     job-level `timeout-minutes`, which is visible in the workflow file). The only bounding
#     mechanism, `--only`, LOGS every excluded suite and brands the run as a non-certifying
#     subset; the nightly workflow never passes it.
#   * DISCOVERY FLOOR. Discovery must find at least SUITE_FLOOR_DEFAULT suites (the count at
#     authoring time), or the run fails — an accidentally-emptied glob cannot green. THE FLOOR
#     RATCHETS UP: when you add a suite, bump SUITE_FLOOR_DEFAULT in the same change (see
#     dev/java-interop/map.md). Lowering it is a conscious act that must accompany a suite's
#     deliberate removal.
#
# Usage:
#   scripts/run_interop_suites.sh                 # the FULL discovered set (what CI runs)
#   scripts/run_interop_suites.sh --list          # print the discovered set and exit
#   scripts/run_interop_suites.sh --only a.sh,b.sh  # LOCAL USE ONLY: run a subset; every
#                                                 # excluded suite is logged by name
#   scripts/run_interop_suites.sh --selftest      # sabotage battery on the driver mechanics
#                                                 # (fake suites in a temp dir; ~seconds)
#
# Environment overrides — SELF-TEST HOOKS, not configuration. Each prints a loud TEST-MODE
# banner when set. The suites themselves default to exactly these paths (47 of the 48 hardcode
# /opt/maven/bin/mvn + JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 outright — 29 via an MVN=
# variable, 18 inline; run-interop-aggregate.sh alone reads $MVN/$JAVA_HOME from the
# environment), so overriding these only moves the driver's prerequisite check — a real run
# still needs the defaults present:
#   ICEBERG_INTEROP_SUITES_DIR   directory to discover suites in (default dev/java-interop)
#   ICEBERG_INTEROP_SUITE_FLOOR  discovery floor (default SUITE_FLOOR_DEFAULT below)
#   ICEBERG_INTEROP_MVN          mvn path the prereq check probes (default /opt/maven/bin/mvn)
#   ICEBERG_INTEROP_JAVA_HOME    JDK home the prereq check probes
#                                (default /usr/lib/jvm/java-11-openjdk-amd64)
#   ICEBERG_INTEROP_CARGO        cargo the prereq check probes (default: `cargo` via PATH)
#
# NOT discovered (documented, not silent): dev/java-interop/run.sh (the original
# metadata-evolution oracle pass) and run-inspection-manifests.sh do not match the
# run-interop-*.sh glob and are outside the nightly set — a named deferral, tracked in
# dev/java-interop/map.md.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# The discovery floor: the number of run-interop-*.sh suites (49 as of 2026-07-16, when
# run-interop-staged-txn.sh was added for R158). RATCHET THIS UP when a suite is added (same
# change); lower it only with a deliberate removal.
SUITE_FLOOR_DEFAULT=49

SUITES_DIR_DEFAULT="${REPO_ROOT}/dev/java-interop"
MVN_DEFAULT="/opt/maven/bin/mvn"
JAVA_HOME_DEFAULT="/usr/lib/jvm/java-11-openjdk-amd64"

SUITES_DIR="${ICEBERG_INTEROP_SUITES_DIR:-${SUITES_DIR_DEFAULT}}"
SUITE_FLOOR="${ICEBERG_INTEROP_SUITE_FLOOR:-${SUITE_FLOOR_DEFAULT}}"
MVN_BIN="${ICEBERG_INTEROP_MVN:-${MVN_DEFAULT}}"
JAVA_HOME_DIR="${ICEBERG_INTEROP_JAVA_HOME:-${JAVA_HOME_DEFAULT}}"
CARGO_BIN="${ICEBERG_INTEROP_CARGO:-cargo}"

LOG_DIR="${REPO_ROOT}/target/interop-nightly/logs"

test_mode_banner() {
    if [[ -n "${ICEBERG_INTEROP_SUITES_DIR:-}" || -n "${ICEBERG_INTEROP_SUITE_FLOOR:-}" \
        || -n "${ICEBERG_INTEROP_MVN:-}" || -n "${ICEBERG_INTEROP_JAVA_HOME:-}" \
        || -n "${ICEBERG_INTEROP_CARGO:-}" ]]; then
        echo "##############################################################################"
        echo "# TEST-MODE OVERRIDE ACTIVE — one or more ICEBERG_INTEROP_* env hooks are set."
        echo "# This run does NOT certify the real interop suite set."
        echo "##############################################################################"
    fi
}

# ---------------------------------------------------------------------------
# Prerequisites — hard-fail, never skip. Every missing prerequisite is listed
# before exiting so one fix round covers them all.
# ---------------------------------------------------------------------------
check_prerequisites() {
    local missing=()
    if [[ ! -x "${MVN_BIN}" ]]; then
        missing+=("mvn: '${MVN_BIN}' is not an executable (the suites hardcode ${MVN_DEFAULT})")
    fi
    if [[ ! -x "${JAVA_HOME_DIR}/bin/java" ]]; then
        missing+=("java: '${JAVA_HOME_DIR}/bin/java' is not an executable (the suites hardcode JAVA_HOME=${JAVA_HOME_DEFAULT})")
    fi
    if [[ "${CARGO_BIN}" == */* ]]; then
        if [[ ! -x "${CARGO_BIN}" ]]; then
            missing+=("cargo: '${CARGO_BIN}' is not an executable")
        fi
    elif ! command -v "${CARGO_BIN}" >/dev/null 2>&1; then
        missing+=("cargo: '${CARGO_BIN}' not found on PATH")
    fi
    if (( ${#missing[@]} > 0 )); then
        echo "ERROR: missing prerequisite(s) — the interop run HARD-FAILS, it never skips:" >&2
        local m
        for m in "${missing[@]}"; do
            echo "  - ${m}" >&2
        done
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Discovery — dynamic glob, so future suites are picked up with zero
# maintenance; the floor assertion keeps an emptied glob from greening.
# ---------------------------------------------------------------------------
discover_suites() {
    DISCOVERED=()
    local f
    while IFS= read -r f; do
        DISCOVERED+=("$(basename "${f}")")
    done < <(find "${SUITES_DIR}" -maxdepth 1 -type f -name 'run-interop-*.sh' | LC_ALL=C sort)

    local count=${#DISCOVERED[@]}
    echo "==> Discovery: ${count} suite(s) matching run-interop-*.sh in ${SUITES_DIR} (floor: ${SUITE_FLOOR})"
    if (( count < SUITE_FLOOR )); then
        echo "ERROR: discovery found ${count} suite(s), below the floor of ${SUITE_FLOOR}." >&2
        echo "       Either the glob broke / suites were removed by accident (fix that), or a suite" >&2
        echo "       was deliberately removed — then lower SUITE_FLOOR_DEFAULT in this script in the" >&2
        echo "       same change. An emptied glob must never green." >&2
        exit 1
    fi
    if (( count > SUITE_FLOOR )); then
        echo "NOTE: discovery found ${count} > floor ${SUITE_FLOOR}. New suites still RUN (no cap);"
        echo "      ratchet SUITE_FLOOR_DEFAULT up to ${count} so a future regression cannot hide."
    fi
}

# ---------------------------------------------------------------------------
# The run loop — continue-on-failure ACROSS suites; every outcome is recorded
# and reported, so one red suite cannot mask the others.
# ---------------------------------------------------------------------------
run_suites() {
    local -a run_set=("$@")
    # A run that runs NOTHING must never green — "0 passed, 0 failed" is not a pass. This is
    # reachable only through misuse (an emptied --only list, or a test-mode floor of 0 over an
    # empty directory), and both must hard-fail rather than certify an empty set.
    if (( ${#run_set[@]} == 0 )); then
        echo "ERROR: the run set is EMPTY — running zero suites is never a green run." >&2
        return 1
    fi
    mkdir -p "${LOG_DIR}"

    local -a result_names=() result_status=() result_secs=()
    local suite rc started elapsed log failed=0
    for suite in "${run_set[@]}"; do
        log="${LOG_DIR}/${suite%.sh}.log"
        echo "==> RUN ${suite} (log: ${log})"
        started=${SECONDS}
        rc=0
        bash "${SUITES_DIR}/${suite}" >"${log}" 2>&1 || rc=$?
        elapsed=$(( SECONDS - started ))
        result_names+=("${suite}")
        result_secs+=("${elapsed}")
        if (( rc == 0 )); then
            result_status+=("PASS")
            echo "    PASS in ${elapsed}s"
        else
            result_status+=("FAIL(rc=${rc})")
            failed=$(( failed + 1 ))
            echo "    FAIL (exit ${rc}) in ${elapsed}s — log tail:"
            tail -n 40 "${log}" | sed 's/^/    | /'
        fi
    done

    # -- summary table ------------------------------------------------------
    local total=${#run_set[@]} passed=$(( ${#run_set[@]} - failed ))
    echo
    echo "=================== interop suite summary ==================="
    local i
    for i in "${!result_names[@]}"; do
        printf '%-55s %-12s %6ss\n' "${result_names[${i}]}" "${result_status[${i}]}" "${result_secs[${i}]}"
    done
    echo "--------------------------------------------------------------"
    echo "TOTAL: ${passed} passed, ${failed} failed, ${total} run, ${#DISCOVERED[@]} discovered (floor ${SUITE_FLOOR})"

    # -- GitHub step summary (markdown), when running under Actions ----------
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
            echo "## Nightly interop suites"
            echo
            echo "| Suite | Result | Seconds |"
            echo "|---|---|---|"
            for i in "${!result_names[@]}"; do
                echo "| ${result_names[${i}]} | ${result_status[${i}]} | ${result_secs[${i}]} |"
            done
            echo
            echo "**TOTAL:** ${passed} passed, ${failed} failed, ${total} run, ${#DISCOVERED[@]} discovered (floor ${SUITE_FLOOR})"
        } >>"${GITHUB_STEP_SUMMARY}"
    fi

    if (( failed > 0 )); then
        echo "ERROR: ${failed} suite(s) failed — see the summary above and the logs in ${LOG_DIR}." >&2
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# --selftest: the sabotage battery on the DRIVER mechanics, over fake suites
# in a temp dir (seconds, no mvn/java invoked by the fakes). Each case is a
# risk pin — proven RED (or GREEN for the control) through the real driver
# code path via the documented env hooks.
# ---------------------------------------------------------------------------
run_selftest() {
    local self="${REPO_ROOT}/scripts/run_interop_suites.sh"
    local tdir
    tdir="$(mktemp -d)"
    trap 'rm -rf "${tdir}"' EXIT

    # Fake prerequisites: the battery is HERMETIC — it proves DRIVER mechanics, so it must not
    # depend on a real /opt/maven or JDK 11 existing on this machine. Each fake is a real
    # executable at a path the prerequisite check probes (and ST3 below proves the check is
    # still load-bearing by pointing individual hooks at nonexistent paths).
    local fake_mvn="${tdir}/prereqs/maven/bin/mvn"
    local fake_java_home="${tdir}/prereqs/jdk"
    local fake_cargo="${tdir}/prereqs/cargo"
    mkdir -p "${tdir}/prereqs/maven/bin" "${fake_java_home}/bin"
    printf '#!/bin/bash\nexit 0\n' >"${fake_mvn}"
    printf '#!/bin/bash\nexit 0\n' >"${fake_java_home}/bin/java"
    printf '#!/bin/bash\nexit 0\n' >"${fake_cargo}"
    chmod +x "${fake_mvn}" "${fake_java_home}/bin/java" "${fake_cargo}"

    # Fake suites: each records that it RAN via a marker file, then exits 0/1.
    local markers="${tdir}/markers"
    mkdir -p "${tdir}/all_pass" "${tdir}/one_fail" "${markers}"
    local name body
    for name in run-interop-fake-pass-a.sh run-interop-fake-pass-b.sh; do
        body="#!/bin/bash
touch '${markers}/'\"\$(basename \"\$0\")\".ran
exit 0"
        printf '%s\n' "${body}" >"${tdir}/all_pass/${name}"
        printf '%s\n' "${body}" >"${tdir}/one_fail/${name}"
    done
    # The failing fake MUST sort FIRST (aa- < fake-pass-*): only a failure BEFORE the passing
    # suites makes their .ran markers pin continue-after-failure. With the failure sorted last
    # (the pre-R2 'zz' name), an abort-on-first-failure mutation ran a/b first and produced an
    # IDENTICAL summary + markers — the battery greened 9/9 (reproduced in remediation R2).
    printf '%s\n' "#!/bin/bash
touch '${markers}/'\"\$(basename \"\$0\")\".ran
exit 1" >"${tdir}/one_fail/run-interop-aa-fail.sh"
    # A third passer so all_pass also has 3 suites (matches the floor used below).
    printf '%s\n' "#!/bin/bash
touch '${markers}/'\"\$(basename \"\$0\")\".ran
exit 0" >"${tdir}/all_pass/run-interop-fake-pass-c.sh"

    local failures=0
    st_case() {
        local label="$1"
        local expect="$2" # zero | nonzero
        local rc="$3"
        shift 3 # remaining args: needle(s) that must appear in the captured output
        local ok=1
        if [[ "${expect}" == "zero" && "${rc}" != "0" ]]; then ok=0; fi
        if [[ "${expect}" == "nonzero" && "${rc}" == "0" ]]; then ok=0; fi
        local needle
        for needle in "$@"; do
            if ! grep -qF -- "${needle}" "${tdir}/out.txt"; then
                echo "    missing expected output: ${needle}"
                ok=0
            fi
        done
        if (( ok == 1 )); then
            echo "PASS selftest(${label})"
        else
            echo "FAIL selftest(${label}) — exit=${rc}, expected ${expect}; output was:"
            sed 's/^/    | /' "${tdir}/out.txt"
            failures=$(( failures + 1 ))
        fi
    }
    drive() { # drive <floor> <dir> [extra args...] — captures output + rc
        local floor="$1" dir="$2"
        shift 2
        rm -f "${markers}"/*.ran
        local rc=0
        ICEBERG_INTEROP_SUITES_DIR="${tdir}/${dir}" \
            ICEBERG_INTEROP_SUITE_FLOOR="${floor}" \
            ICEBERG_INTEROP_MVN="${fake_mvn}" \
            ICEBERG_INTEROP_JAVA_HOME="${fake_java_home}" \
            ICEBERG_INTEROP_CARGO="${fake_cargo}" \
            bash "${self}" "$@" >"${tdir}/out.txt" 2>&1 || rc=$?
        echo "${rc}"
    }

    echo "==> Driver selftest (sabotage battery over fake suites in ${tdir})"

    # ST1 — RISK: a failing suite greens the nightly, or aborts/masks the other suites.
    # The failing fake sorts FIRST (see above), so the a/b marker check below directly pins
    # "a passing suite still runs AFTER a failure" — independent of any summary wording.
    local rc
    rc="$(drive 3 one_fail)"
    st_case "failing-suite-fails-run-and-others-still-run" nonzero "${rc}" \
        "run-interop-aa-fail.sh" "FAIL(rc=1)" \
        "run-interop-fake-pass-a.sh" "run-interop-fake-pass-b.sh" \
        "TOTAL: 2 passed, 1 failed, 3 run"
    if [[ ! -f "${markers}/run-interop-fake-pass-a.sh.ran" || ! -f "${markers}/run-interop-fake-pass-b.sh.ran" ]]; then
        echo "FAIL selftest(failing-suite-continue-across): a passing suite did not RUN after the failure"
        failures=$(( failures + 1 ))
    fi

    # ST2 — RISK: an emptied/shrunk glob greens (discovery below the floor must fail, run nothing).
    rc="$(drive 4 one_fail)"
    st_case "floor-breach-fails-before-running" nonzero "${rc}" "below the floor of 4"
    if compgen -G "${markers}/*.ran" >/dev/null; then
        echo "FAIL selftest(floor-breach-runs-nothing): suites ran despite the floor breach"
        failures=$(( failures + 1 ))
    fi

    # ST3 — RISK: a missing prerequisite silently skips instead of hard-failing. Each case
    # isolates ONE missing prerequisite (the other two hooks point at the working fakes), so
    # a pass proves the probe for exactly that prerequisite is load-bearing.
    rm -f "${markers}"/*.ran
    rc=0
    ICEBERG_INTEROP_SUITES_DIR="${tdir}/all_pass" ICEBERG_INTEROP_SUITE_FLOOR=3 \
        ICEBERG_INTEROP_MVN="/nonexistent/mvn-for-selftest" \
        ICEBERG_INTEROP_JAVA_HOME="${fake_java_home}" \
        ICEBERG_INTEROP_CARGO="${fake_cargo}" \
        bash "${self}" >"${tdir}/out.txt" 2>&1 || rc=$?
    st_case "missing-mvn-hard-fails-never-skips" nonzero "${rc}" "missing prerequisite" "/nonexistent/mvn-for-selftest"
    rc=0
    ICEBERG_INTEROP_SUITES_DIR="${tdir}/all_pass" ICEBERG_INTEROP_SUITE_FLOOR=3 \
        ICEBERG_INTEROP_MVN="${fake_mvn}" \
        ICEBERG_INTEROP_JAVA_HOME="${fake_java_home}" \
        ICEBERG_INTEROP_CARGO="/nonexistent/cargo-for-selftest" \
        bash "${self}" >"${tdir}/out.txt" 2>&1 || rc=$?
    st_case "missing-cargo-hard-fails-never-skips" nonzero "${rc}" "missing prerequisite" "/nonexistent/cargo-for-selftest"
    rc=0
    ICEBERG_INTEROP_SUITES_DIR="${tdir}/all_pass" ICEBERG_INTEROP_SUITE_FLOOR=3 \
        ICEBERG_INTEROP_MVN="${fake_mvn}" \
        ICEBERG_INTEROP_JAVA_HOME="/nonexistent/jdk-for-selftest" \
        ICEBERG_INTEROP_CARGO="${fake_cargo}" \
        bash "${self}" >"${tdir}/out.txt" 2>&1 || rc=$?
    st_case "missing-java-hard-fails-never-skips" nonzero "${rc}" "missing prerequisite" "/nonexistent/jdk-for-selftest/bin/java"
    if compgen -G "${markers}/*.ran" >/dev/null; then
        echo "FAIL selftest(prereq-runs-nothing): suites ran despite a missing prerequisite"
        failures=$(( failures + 1 ))
    fi

    # ST4 — CONTROL: an all-green set exits 0 with a correct summary (the battery above is
    # meaningful only if the clean path passes).
    rc="$(drive 3 all_pass)"
    st_case "all-green-control-passes" zero "${rc}" "TOTAL: 3 passed, 0 failed, 3 run"

    # ST5 — RISK: the subset flag silently caps the run (exclusions must be LOGGED by name).
    rc="$(drive 3 all_pass --only run-interop-fake-pass-a.sh)"
    st_case "subset-logs-exclusions" zero "${rc}" \
        "SUBSET RUN" "EXCLUDED (2)" \
        "run-interop-fake-pass-b.sh" "run-interop-fake-pass-c.sh" \
        "TOTAL: 1 passed, 0 failed, 1 run"
    if [[ -f "${markers}/run-interop-fake-pass-b.sh.ran" ]]; then
        echo "FAIL selftest(subset-excluded-must-not-run): an excluded suite ran"
        failures=$(( failures + 1 ))
    fi

    # ST6 — RISK: a typo'd --only name silently runs nothing it was asked for.
    rc="$(drive 3 all_pass --only run-interop-no-such-suite.sh)"
    st_case "unknown-only-name-hard-fails" nonzero "${rc}" "not in the discovered set"

    # ST7 — RISK: `--only ""` silently ignores the flag and runs the FULL set (the pre-fix
    # behavior: an empty value failed the -n gate, so a bounded-run request became an
    # unbounded run). It must hard-fail before running anything.
    rc="$(drive 3 all_pass --only "")"
    st_case "empty-only-value-hard-fails" nonzero "${rc}" "NON-EMPTY"
    if compgen -G "${markers}/*.ran" >/dev/null; then
        echo "FAIL selftest(empty-only-runs-nothing): suites ran despite the empty --only"
        failures=$(( failures + 1 ))
    fi

    # ST8 — RISK: a run that runs ZERO suites greens ("0 passed, 0 failed" is not a pass).
    # Reachable via the test-mode hooks (floor 0 over an empty directory); the empty-run-set
    # guard in run_suites must hard-fail it.
    mkdir -p "${tdir}/empty"
    rc="$(drive 0 empty)"
    st_case "zero-suite-run-never-greens" nonzero "${rc}" "run set is EMPTY"

    echo
    if (( failures > 0 )); then
        echo "SELFTEST FAILED: ${failures} case(s) — the driver mechanics are NOT proven." >&2
        exit 1
    fi
    echo "SELFTEST PASSED: all driver-mechanics risk pins hold."
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing + main.
# ---------------------------------------------------------------------------
ONLY_CSV=""
LIST_ONLY=0
while (( $# > 0 )); do
    case "$1" in
        --selftest)
            run_selftest
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        --only)
            # An EMPTY value must hard-fail: with `--only ""` the old `-n` gate below would
            # silently ignore the flag and run the FULL set — the caller asked for a bounded
            # run and got an unbounded one (found in remediation round 1, mutation-pinned ST7).
            if (( $# < 2 )) || [[ -z "$2" ]]; then
                echo "ERROR: --only requires a NON-EMPTY comma-separated list of suite names" >&2
                exit 1
            fi
            if [[ -n "${ONLY_CSV}" ]]; then
                ONLY_CSV="${ONLY_CSV},$2"
            else
                ONLY_CSV="$2"
            fi
            shift 2
            ;;
        --help|-h)
            # Print the header doc block by its MARKERS, not line numbers — a hardcoded line
            # range silently drifts the moment the header is edited (found in remediation
            # round 1: an earlier header edit had already shifted it).
            awk '/^# run_interop_suites\.sh /{doc=1} /^set -euo pipefail/{exit} doc' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1' (see --help)" >&2
            exit 1
            ;;
    esac
done

test_mode_banner

DISCOVERED=()
discover_suites

if (( LIST_ONLY == 1 )); then
    printf '%s\n' "${DISCOVERED[@]}"
    exit 0
fi

check_prerequisites

RUN_SET=()
if [[ -n "${ONLY_CSV}" ]]; then
    # LOCAL-USE subset: validate every requested name, then LOG the exclusions —
    # a bounded run must say exactly what it dropped (no silent caps).
    IFS=',' read -r -a requested <<<"${ONLY_CSV}"
    for want in "${requested[@]}"; do
        found=0
        for suite in "${DISCOVERED[@]}"; do
            if [[ "${suite}" == "${want}" ]]; then
                found=1
                break
            fi
        done
        if (( found == 0 )); then
            echo "ERROR: --only '${want}' is not in the discovered set (see --list)" >&2
            exit 1
        fi
        RUN_SET+=("${want}")
    done
    EXCLUDED=()
    for suite in "${DISCOVERED[@]}"; do
        keep=0
        for want in "${RUN_SET[@]}"; do
            if [[ "${suite}" == "${want}" ]]; then
                keep=1
                break
            fi
        done
        if (( keep == 0 )); then
            EXCLUDED+=("${suite}")
        fi
    done
    echo "##############################################################################"
    echo "# SUBSET RUN (--only): LOCAL USE ONLY — this is NOT a full interop"
    echo "# certification. The nightly workflow always runs the FULL discovered set."
    echo "##############################################################################"
    echo "EXCLUDED (${#EXCLUDED[@]}) — the suites this bounded run DROPS:"
    for suite in "${EXCLUDED[@]}"; do
        echo "  - ${suite}"
    done
else
    RUN_SET=("${DISCOVERED[@]}")
fi

run_suites "${RUN_SET[@]}"
