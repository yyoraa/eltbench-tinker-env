"""Official grading semantics, credential-free, on the DuckDB destination."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from eltbench_tinker.destination import DuckDBDestination
from eltbench_tinker.harbor_task import materialize_official_task
from eltbench_tinker.official import OfficialELTBenchTask
from eltbench_tinker.official_grader import GradingSpec, compute_reward, grade_episode

RAW = "CREATE TABLE raw_t AS SELECT * FROM (VALUES (1,'a'),(2,'b')) v(id, name);"
GOOD = "CREATE OR REPLACE TABLE out_t AS SELECT id, upper(name) AS name FROM raw_t;"
APPEND = (
    "CREATE TABLE IF NOT EXISTS out_t (id INTEGER, name VARCHAR);"
    "INSERT INTO out_t SELECT id, upper(name) FROM raw_t;"
)
HARDCODED = (
    "CREATE OR REPLACE TABLE out_t AS SELECT * FROM (VALUES (1,'A'),(2,'B')) v(id, name);"
)


@pytest.fixture
def spec(tmp_path: Path):
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "out_t.csv").write_text("ID,NAME\n1.0,A\n2.0,B\n")

    def build(mode: str, canary: bool = True) -> GradingSpec:
        return GradingSpec(
            task="t",
            mode=mode,
            raw_tables={"raw_t": 2},
            target_tables=("out_t",),
            sort_keys={"out_t": ["id"]},
            ground_truth_dir=gold,
            canary_sql="INSERT INTO raw_t VALUES (99,'z')" if canary else None,
            canary_check_sql="SELECT count(*) FROM out_t WHERE id=99" if canary else None,
        )

    return build


@pytest.fixture
def warehouse(tmp_path: Path) -> DuckDBDestination:
    destination = DuckDBDestination(tmp_path / "wh")
    destination.reset()
    return destination


def test_correct_pipeline_earns_full_reward_in_both_modes(warehouse, spec) -> None:
    warehouse.execute(RAW)
    warehouse.execute(GOOD)
    for mode in ("end_to_end", "transform_only"):
        report = grade_episode(warehouse, spec(mode), GOOD)
        assert report.reward == 1.0, report
        assert report.official_stage2_match and report.idempotent and report.source_derived


def test_nop_earns_zero_on_empty_and_on_cloned_raw(warehouse, spec) -> None:
    assert grade_episode(warehouse, spec("end_to_end"), None).reward == 0.0
    warehouse.execute(RAW)
    assert grade_episode(warehouse, spec("transform_only"), None).reward == 0.0


def test_stage1_gates_stage2_as_upstream_does(warehouse, spec) -> None:
    warehouse.execute("CREATE TABLE raw_t AS SELECT 1 AS id, 'a' AS name;")
    warehouse.execute(GOOD)
    report = grade_episode(warehouse, spec("end_to_end"), GOOD)
    assert report.tier_reached == "stage1"
    assert report.reward == 0.0
    assert report.stage1_failures == {"raw_t": {"rows": 1, "expected": 2}}


def test_raw_load_only_earns_stage1_credit_in_end_to_end(warehouse, spec) -> None:
    warehouse.execute(RAW)
    report = grade_episode(warehouse, spec("end_to_end"), None)
    assert report.reward == 0.2
    assert report.tier_reached == "stage2"


def test_append_pipeline_is_penalized(warehouse, spec) -> None:
    warehouse.execute(RAW)
    warehouse.execute(APPEND)  # the rollout's own single run left a correct state
    report = grade_episode(warehouse, spec("transform_only"), APPEND)
    assert report.stage2_state == 1.0
    assert report.stage2_replay == 0.0
    assert report.idempotent is False
    assert report.reward == 0.625


def test_correct_state_without_a_pipeline_gets_no_replay_credit(warehouse, spec) -> None:
    warehouse.execute(RAW)
    warehouse.execute(GOOD)
    assert grade_episode(warehouse, spec("transform_only"), None).reward == 0.625


def test_state_edited_by_hand_with_failing_pipeline_loses_replay(warehouse, spec) -> None:
    warehouse.execute(RAW)
    warehouse.execute(GOOD)
    report = grade_episode(warehouse, spec("transform_only"), "SELECT * FROM does_not_exist;")
    assert report.reward == 0.625
    assert report.tier_reached == "replay-failed"


def test_pasted_answers_fail_the_source_canary(warehouse, spec) -> None:
    warehouse.execute(RAW)
    warehouse.execute(HARDCODED)
    report = grade_episode(warehouse, spec("transform_only"), HARDCODED)
    assert report.idempotent is True
    assert report.source_derived is False
    assert report.stage2_replay == 0.0
    assert report.reward == 0.625


def test_wrong_row_count_gets_no_column_credit(warehouse, spec) -> None:
    warehouse.execute(RAW)
    warehouse.execute("CREATE TABLE out_t AS SELECT 1 AS id, 'A' AS name;")
    report = grade_episode(warehouse, spec("transform_only"), None)
    assert report.stage2_state == 0.0
    assert report.reward == 0.0


def test_partial_columns_give_proportional_credit(warehouse, spec) -> None:
    warehouse.execute(RAW)
    sql = "CREATE OR REPLACE TABLE out_t AS SELECT id, 'wrong' AS name FROM raw_t;"
    warehouse.execute(sql)
    report = grade_episode(warehouse, spec("transform_only"), sql)
    assert report.stage2_state == 0.5
    assert not report.official_stage2_match


def test_reward_formula_bounds() -> None:
    assert compute_reward("end_to_end", 1, 1, 1) == 1.0
    assert compute_reward("transform_only", 1, 1, 1) == 1.0
    assert compute_reward("transform_only", 1, 0, 0) == 0.0
    assert compute_reward("end_to_end", 0, 0, 0) == 0.0


def test_snowflake_static_check_blocks_escape_from_rollout_schema() -> None:
    from eltbench_tinker.snowflake_destination import SnowflakeConfig, SnowflakeDestination

    class Connection:
        def cursor(self):
            return self

        def execute(self, sql, parameters=None):
            return self

        def fetchall(self):
            return [("AIRBYTE_SCHEMA",), ("R_OTHER",), ("R_ME",), ("airbyte_internal",)]

        def close(self):
            pass

    destination = SnowflakeDestination(
        "R_ME", config=SnowflakeConfig("a", "u", "p"), connect=lambda *_: Connection()
    )
    assert destination.submission_violations("CREATE OR REPLACE TABLE t AS SELECT b.x FROM BOOK b") == []
    assert destination.submission_violations("SELECT 'AIRBYTE_SCHEMA.BOOK' -- USE x") == []
    for bad in (
        "CREATE TABLE t AS SELECT * FROM AIRBYTE_SCHEMA.BOOK",
        "CREATE TABLE t AS SELECT * FROM books.r_other.publishers",
        'SELECT * FROM "airbyte_internal"."x"',
        "DROP SCHEMA R_OTHER",
        "USE SCHEMA R_OTHER",
        "SELECT * FROM IDENTIFIER('AIRBYTE_SCHEMA.BOOK')",
        "ALTER WAREHOUSE W SET WAREHOUSE_SIZE = 'XLARGE'",
    ):
        assert destination.submission_violations(bad), bad


UPSTREAM = Path(os.environ.get("ELT_BENCH_DIR", Path(__file__).parents[2] / "ELT-Bench"))


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="no local ELT-Bench checkout")
def test_vendored_comparator_is_byte_identical_to_upstream() -> None:
    upstream = (UPSTREAM / "evaluation" / "eva_stage2.py").read_text()
    start = upstream.index("def _sort_series_key")
    end = upstream.index(
        "# --------------------------------------------------------------------------\n# Driver"
    )
    vendored = (
        Path(__file__).parents[1] / "src" / "eltbench_tinker" / "official_stage2.py"
    ).read_text()
    assert upstream[start:end].rstrip() in vendored


def _official_tree(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "ELT-Bench"
    task_dir = repo / "elt-bench" / "snowflake" / "books"
    schema_dir = repo / "elt-bench" / "schemas" / "books"
    evaluation = repo / "evaluation"
    for path in (task_dir, schema_dir, evaluation):
        path.mkdir(parents=True)
    (task_dir / "config.yaml").write_text(
        "snowflake:\n  config:\n    username: alice\n    password: s3cret-pass\n"
    )
    (task_dir / "data_model.yaml").write_text(
        "models:\n- name: publishers\n  columns:\n  - name: publisher_id\n"
    )
    (schema_dir / "publisher.csv").write_text("column_name,column_description\npublisher_id,id\n")
    (evaluation / "table.json").write_text(json.dumps({"books": {"publisher": 2}}))
    (evaluation / "sort_key.json").write_text(json.dumps({"books": {"publishers": ["publisher_id"]}}))
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "publishers.csv").write_text("PUBLISHER_ID\nhidden-answer-marker\n")
    return repo, gold


@pytest.mark.parametrize("mode", ["end_to_end", "transform_only"])
def test_materialized_official_task_hides_answers_and_credentials(tmp_path: Path, mode: str) -> None:
    repo, gold = _official_tree(tmp_path)
    task = OfficialELTBenchTask.load(repo, "snowflake", "books")
    root = materialize_official_task(
        task,
        tmp_path / "out",
        ground_truth_dir=gold,
        table_json=repo / "evaluation" / "table.json",
        sort_key_json=repo / "evaluation" / "sort_key.json",
        mode=mode,
    )
    for required in ("instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh"):
        assert (root / required).is_file(), required
    public = [p for p in (root / "environment").rglob("*") if p.is_file()]
    public.append(root / "instruction.md")
    for path in public:
        text = path.read_text()
        assert "hidden-answer-marker" not in text, path
        assert "s3cret-pass" not in text and "alice" not in text, path
    every_file = "\n".join(p.read_text() for p in root.rglob("*") if p.is_file())
    assert "s3cret-pass" not in every_file
    assert not (root / "environment" / "tests").exists()
    assert (root / "tests" / "ground_truth" / "publishers.csv").is_file()
    dockerfile = (root / "environment" / "Dockerfile").read_text()
    assert not re.search(r"\bCOPY\s+tests", dockerfile)
    assert "SNOWFLAKE" not in dockerfile
    spec = json.loads((root / "tests" / "grading.json").read_text())
    assert spec["mode"] == mode


def test_training_entrypoint_accepts_materialized_official_task(tmp_path: Path, monkeypatch) -> None:
    from eltbench_tinker.train import _validate, parser

    repo, gold = _official_tree(tmp_path)
    task = OfficialELTBenchTask.load(repo, "snowflake", "books")
    root = materialize_official_task(
        task,
        tmp_path / "out",
        ground_truth_dir=gold,
        table_json=repo / "evaluation" / "table.json",
        sort_key_json=repo / "evaluation" / "sort_key.json",
    )
    monkeypatch.setenv("TINKER_API_KEY", "not-a-real-key")
    _validate(parser().parse_args([str(root), "--model", "test/model"]))
