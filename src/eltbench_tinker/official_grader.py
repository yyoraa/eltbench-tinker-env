"""Hidden grader for official ELT-Bench tasks; warehouse-agnostic.

Tiers run cheapest first and stop as soon as the outcome is decided:

  T0 stage 1  official raw-table existence and row counts
  T1 stage 2  official comparator on the rollout's final warehouse state
  T2 replay   rerun the submitted pipeline twice in a clone (pipeline really
              produces the state, and is idempotent), then perturb a source row
              (output really derives from the source)

Reward (end_to_end):     0.2*S1 + 0.5*S2_state + 0.3*S2_replay
Reward (transform_only): (0.5*S2_state + 0.3*S2_replay) / 0.8
S1 is the raw-table pass fraction. S2 is the matched-column fraction over gold
columns (1.0 is exactly the official per-table "match"). transform_only starts
from cloned raw tables, so stage 1 is a precondition there, not earned credit;
that is what keeps nop at exactly 0.0 in both modes.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .destination import Destination, create_destination
from .official_stage2 import check_corretness, sort_by_keys

MODES = ("end_to_end", "transform_only")
REPLAY_SUFFIX = "_REPLAY"


@dataclass(frozen=True)
class GradingSpec:
    task: str
    mode: str
    raw_tables: dict[str, int]
    target_tables: tuple[str, ...]
    sort_keys: dict[str, list[str]]
    ground_truth_dir: Path
    canary_sql: str | None = None
    canary_check_sql: str | None = None

    @classmethod
    def load(cls, path: Path) -> "GradingSpec":
        data = json.loads(Path(path).read_text())
        return cls(
            task=data["task"],
            mode=data["mode"],
            raw_tables=data["raw_tables"],
            target_tables=tuple(data["target_tables"]),
            sort_keys=data["sort_keys"],
            ground_truth_dir=Path(path).parent / data["ground_truth_dir"],
            canary_sql=data.get("canary_sql"),
            canary_check_sql=data.get("canary_check_sql"),
        )


@dataclass
class GradeReport:
    task: str
    mode: str
    reward: float = 0.0
    tier_reached: str = "none"
    stage1_fraction: float = 0.0
    stage1_failures: dict[str, Any] = field(default_factory=dict)
    stage2_state: float = 0.0
    stage2_replay: float = 0.0
    official_stage2_match: bool = False
    tables: dict[str, Any] = field(default_factory=dict)
    idempotent: bool | None = None
    source_derived: bool | None = None
    submission_found: bool = False
    violations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)


def stage1(destination: Destination, expected: dict[str, int]) -> tuple[float, dict[str, Any]]:
    """Official stage 1: every raw table exists with the exact row count."""
    if not expected:
        return 1.0, {}
    sizes = {name.upper(): size for name, size in destination.table_sizes().items()}
    failures: dict[str, Any] = {}
    for table, wanted in expected.items():
        size = sizes.get(table.upper())
        if size is None:
            failures[table] = "not found"
        elif size != wanted:
            failures[table] = {"rows": size, "expected": wanted}
    return (len(expected) - len(failures)) / len(expected), failures


def compare_table(
    destination: Destination, table: str, ground_truth: Path, keys: list[str]
) -> dict[str, Any]:
    """Official stage 2 for one table, through the official CSV round trip."""
    names, rows = destination.fetch_table(table)
    buffer = io.StringIO()
    pd.DataFrame(rows, columns=names).to_csv(buffer, index=False)
    buffer.seek(0)
    actual = pd.read_csv(buffer, dtype=str, keep_default_na=True)
    gold = pd.read_csv(ground_truth, dtype=str, keep_default_na=True)
    result = check_corretness(sort_by_keys(gold, keys), sort_by_keys(actual, keys))
    columns = len(gold.columns)
    return {
        "match": bool(result["match"]),
        "fraction": len(result["matched"]) / columns if columns else 0.0,
        "unmatched": result["unmatched"],
        "missed": result["missed"],
        "rows": len(actual),
        "expected_rows": len(gold),
    }


def stage2(destination: Destination, spec: GradingSpec) -> tuple[float, bool, dict[str, Any]]:
    present = {name.upper() for name in destination.list_tables()}
    tables: dict[str, Any] = {}
    for table in spec.target_tables:
        if table.upper() not in present:
            tables[table] = {"match": False, "fraction": 0.0, "missing": True}
            continue
        try:
            tables[table] = compare_table(
                destination,
                table,
                spec.ground_truth_dir / f"{table}.csv",
                spec.sort_keys.get(table, []),
            )
        except Exception as error:
            # An unreadable table is the rollout's failure, not an infrastructure one.
            tables[table] = {"match": False, "fraction": 0.0, "error": type(error).__name__}
    fraction = sum(item["fraction"] for item in tables.values()) / max(1, len(tables))
    return fraction, all(item["match"] for item in tables.values()), tables


def compute_reward(mode: str, stage1_fraction: float, state: float, replay: float) -> float:
    if mode == "end_to_end":
        value = 0.2 * stage1_fraction + 0.5 * state + 0.3 * replay
    elif mode == "transform_only":
        value = (0.5 * state + 0.3 * replay) / 0.8
    else:
        raise ValueError(f"unknown grading mode {mode!r}")
    return round(value, 8)


def _replay(
    destination: Destination, spec: GradingSpec, pipeline: str, report: GradeReport
) -> None:
    clone = destination.clone(REPLAY_SUFFIX)
    try:
        for table in spec.target_tables:
            clone.execute(f"DROP TABLE IF EXISTS {table}")
        clone.execute(pipeline)
        clone.execute(pipeline)
        fraction, match, tables = stage2(clone, spec)
        report.stage2_replay = fraction
        report.idempotent = match
        report.tables = {name: {**report.tables.get(name, {}), "replay": item} for name, item in tables.items()}
        if fraction > 0 and spec.canary_sql and spec.canary_check_sql:
            clone.execute(spec.canary_sql)
            clone.execute(pipeline)
            report.source_derived = int(clone.query(spec.canary_check_sql)[0][0]) >= 1
            if not report.source_derived:
                report.stage2_replay = 0.0
    finally:
        clone.cleanup()


def grade_episode(
    destination: Destination, spec: GradingSpec, submission: str | None
) -> GradeReport:
    """Grade the warehouse state, then the submitted pipeline. Never raises for
    a bad rollout; infrastructure errors propagate so a caller can tell them apart."""
    report = GradeReport(task=spec.task, mode=spec.mode)
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        report.seconds[name] = round(now - clock, 3)
        clock = now

    report.stage1_fraction, report.stage1_failures = stage1(destination, spec.raw_tables)
    report.tier_reached = "stage1"
    lap("stage1")
    if report.stage1_fraction < 1.0:
        # Official semantics: stage 2 only runs for schemas that passed stage 1.
        report.reward = compute_reward(spec.mode, report.stage1_fraction, 0.0, 0.0)
        return report

    report.stage2_state, report.official_stage2_match, report.tables = stage2(destination, spec)
    report.tier_reached = "stage2"
    lap("stage2")

    report.submission_found = bool(submission and submission.strip())
    if report.stage2_state > 0 and report.submission_found:
        assert submission is not None
        report.violations = destination.submission_violations(submission)
        if not report.violations:
            try:
                _replay(destination, spec, submission, report)
                report.tier_reached = "replay"
            except Exception as error:
                report.errors.append(f"replay failed: {type(error).__name__}")
                report.stage2_replay = 0.0
                report.tier_reached = "replay-failed"
            lap("replay")

    report.reward = compute_reward(
        spec.mode, report.stage1_fraction, report.stage2_state, report.stage2_replay
    )
    return report


def write_reward(report: GradeReport, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "grading.json").write_text(json.dumps(asdict(report), indent=1, default=str))
    (log_dir / "reward.txt").write_text(f"{report.reward}\n")


def main() -> int:
    """Entry point used by tests/test.sh. Fails closed: any crash means reward 0."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    (args.logs / "reward.txt").write_text("0.0\n")
    spec = GradingSpec.load(args.spec)
    destination = create_destination(args.destination)
    started = time.perf_counter()
    try:
        submission = args.submission.read_text() if args.submission.is_file() else None
        report = grade_episode(destination, spec, submission)
    except Exception as error:
        report = GradeReport(task=spec.task, mode=spec.mode)
        report.errors.append(f"grader error: {type(error).__name__}")
        report.tier_reached = "grader-error"
        report.seconds["total"] = round(time.perf_counter() - started, 3)
        write_reward(report, args.logs)
        return 2
    report.seconds["total"] = round(time.perf_counter() - started, 3)
    write_reward(report, args.logs)
    # The rollout's schema outlives grading (the runner drops it); only the session ends here.
    close = getattr(destination, "close", None)
    if close:
        close()
    return 0
