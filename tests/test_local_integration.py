from pathlib import Path

import duckdb
import pytest

from eltbench_tinker.destination import DuckDBDestination
from eltbench_tinker.local_env import LocalELTEnvironment
from eltbench_tinker.harbor_task import materialize_local_harbor_task
from eltbench_tinker.official_grader import GradingSpec, grade_episode
from eltbench_tinker.official import OfficialELTBenchTask
from eltbench_tinker.official_compare import rows_match, values_match
from eltbench_tinker.task import ELTTask, ExpectedTable
from eltbench_tinker.train import _validate, parser


FIXTURE = Path(__file__).parent / "fixtures" / "customer_revenue"


def make_task() -> ELTTask:
    return ELTTask(
        task_id="customer-revenue",
        instruction=(
            "Load customers.csv and orders.csv and create customer_revenue "
            "with customer_id BIGINT, name VARCHAR, revenue DECIMAL(38,2)."
        ),
        source_dir=FIXTURE,
        expected_tables=(
            ExpectedTable(
                name="customer_revenue",
                columns=(
                    ("customer_id", "BIGINT"),
                    ("name", "VARCHAR"),
                    ("revenue", "DECIMAL(38,2)"),
                ),
                rows=((1, "Ada", 20.00), (2, "Linus", 9.00), (3, "Grace", 0.00)),
                order_by=("customer_id",),
            ),
        ),
    )


def test_credential_free_episode_and_hidden_grader() -> None:
    env = LocalELTEnvironment(make_task())
    try:
        assert "customer_revenue" in env.reset()
        result = env.run(
            "cat > solution.sql <<'SQL'\n"
            "CREATE OR REPLACE TABLE customer_revenue AS\n"
            "SELECT c.customer_id, c.name, "
            "CAST(COALESCE(SUM(o.amount), 0) AS DECIMAL(38,2)) AS revenue\n"
            "FROM read_csv_auto('customers.csv') c\n"
            "LEFT JOIN read_csv_auto('orders.csv') o USING (customer_id)\n"
            "GROUP BY c.customer_id, c.name;\n"
            "SQL\n"
        )
        assert result.exit_code == 0
        env.execute_sql_file("solution.sql")
        grade = env.grade()
        assert grade.reward == 1.0, grade.errors
        assert grade.components == {
            "table_existence": 1.0,
            "schema": 1.0,
            "content": 1.0,
        }
    finally:
        env.cleanup()


def test_partial_reward_does_not_confuse_existence_with_correctness() -> None:
    env = LocalELTEnvironment(make_task())
    try:
        env.reset()
        env.destination.execute(
            "CREATE TABLE customer_revenue(customer_id BIGINT, name VARCHAR, revenue DOUBLE);"
        )
        grade = env.grade()
        assert grade.reward == 0.1
        assert grade.components["table_existence"] == 1.0
        assert grade.components["schema"] == 0.0
        assert grade.components["content"] == 0.0
    finally:
        env.cleanup()


def test_reset_removes_previous_warehouse_state() -> None:
    env = LocalELTEnvironment(make_task())
    try:
        env.reset()
        env.destination.execute("CREATE TABLE leaked(value INTEGER);")
        assert env.destination.table_exists("leaked")
        env.reset()
        assert not env.destination.table_exists("leaked")
    finally:
        env.cleanup()


def test_materialized_harbor_hidden_grader(tmp_path: Path) -> None:
    task = make_task()
    task_dir = materialize_local_harbor_task(task, tmp_path)
    assert (task_dir / "environment" / "Dockerfile").is_file()
    assert (task_dir / "tests" / "test.sh").is_file()
    assert "expected" not in (task_dir / "instruction.md").read_text().lower()

    env = LocalELTEnvironment(task)
    pipeline = (
        "CREATE OR REPLACE TABLE customer_revenue AS "
        "SELECT c.customer_id::BIGINT AS customer_id, c.name::VARCHAR AS name, "
        "CAST(COALESCE(SUM(o.amount), 0) AS DECIMAL(38,2)) AS revenue "
        "FROM read_csv_auto('customers.csv') c "
        "LEFT JOIN read_csv_auto('orders.csv') o USING (customer_id) "
        "GROUP BY c.customer_id, c.name"
    )
    try:
        env.reset()
        env.destination.execute(pipeline)
        spec = GradingSpec.load(task_dir / "tests" / "grading.json")
        assert grade_episode(env.destination, spec, pipeline).reward == 1.0
    finally:
        env.cleanup()


def test_duckdb_pipeline_cannot_touch_files_outside_its_directory(tmp_path: Path) -> None:
    secret = tmp_path / "answers.csv"
    secret.write_text("a\n1\n")
    destination = DuckDBDestination(tmp_path / "wh")
    destination.reset()
    for attack in (
        f"CREATE TABLE stolen AS SELECT * FROM read_csv_auto('{secret}')",
        f"COPY (SELECT 1.0) TO '{tmp_path}/reward.txt'",
    ):
        with pytest.raises(duckdb.Error):
            destination.execute(attack)
    assert not (tmp_path / "reward.txt").exists()


def test_official_books_importer_redacts_credentials(tmp_path: Path) -> None:
    repo = tmp_path / "ELT-Bench"
    task_dir = repo / "elt-bench" / "snowflake" / "books"
    schema_dir = repo / "elt-bench" / "schemas" / "books"
    task_dir.mkdir(parents=True)
    schema_dir.mkdir(parents=True)
    (task_dir / "config.yaml").write_text(
        "snowflake:\n  config:\n    username: alice\n    password: secret\n"
    )
    (task_dir / "data_model.yaml").write_text(
        "models:\n- name: publishers\n  columns:\n  - name: publisher_id\n"
    )
    (schema_dir / "publisher.csv").write_text(
        "column_name,column_description\npublisher_id,identifier\n"
    )
    task = OfficialELTBenchTask.load(repo, "snowflake", "books")
    assert task.config["snowflake"]["config"]["password"] == "<injected-at-runtime>"
    assert task.config["snowflake"]["config"]["username"] == "<injected-at-runtime>"
    public = task.write_public_bundle(tmp_path / "public")
    exposed = (public / "config.yaml").read_text()
    assert "secret" not in exposed
    assert "alice" not in exposed
    assert "publishers" in task.instruction()


def test_submit_is_idempotent_and_terminates_episode() -> None:
    env = LocalELTEnvironment(make_task())
    try:
        env.reset()
        result = env.run(
            "cat > solution.sql <<'SQL'\n"
            "CREATE OR REPLACE TABLE customer_revenue AS\n"
            "SELECT c.customer_id, c.name, "
            "CAST(COALESCE(SUM(o.amount), 0) AS DECIMAL(38,2)) AS revenue\n"
            "FROM read_csv_auto('customers.csv') c\n"
            "LEFT JOIN read_csv_auto('orders.csv') o USING (customer_id)\n"
            "GROUP BY c.customer_id, c.name;\n"
            "SQL\n"
        )
        assert result.exit_code == 0
        submission = env.submit("solution.sql")
        assert submission.reward == 1.0
        assert submission.idempotent
        assert submission.transition_seconds >= 0
        assert submission.grade_seconds >= 0
        try:
            env.run("true")
        except RuntimeError as error:
            assert "terminated" in str(error)
        else:
            raise AssertionError("terminated episode accepted another action")
    finally:
        env.cleanup()


def test_nop_and_append_only_solution_cannot_get_full_reward(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    task = ELTTask(
        task_id="append-cheat",
        instruction="Create counters with one row containing value 1.",
        source_dir=empty,
        expected_tables=(
            ExpectedTable(
                name="counters",
                columns=(("value", "INTEGER"),),
                rows=((1,),),
            ),
        ),
    )
    env = LocalELTEnvironment(task)
    try:
        env.reset()
        assert env.grade().reward == 0.0  # nop must fail
        assert env.workspace is not None
        (env.workspace / "append.sql").write_text(
            "CREATE TABLE IF NOT EXISTS counters(value INTEGER);"
            "INSERT INTO counters VALUES (1);"
        )
        submission = env.submit("append.sql")
        assert not submission.idempotent
        assert submission.reward < 1.0
    finally:
        env.cleanup()


def test_official_comparison_semantics() -> None:
    assert values_match("0E-9", "0.0")
    assert values_match(" New York ", "new york")
    assert values_match("100.5", "100.0", rtol=1e-2)
    assert not values_match(None, "")
    assert rows_match(
        [(2, "B", "20.000"), (1, "A", "10")],
        [(1, "a", 10.0), (2, "b", 20)],
        key_indices=(0,),
    )


def test_training_entrypoint_fails_closed_without_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    args = parser().parse_args(
        [str(tmp_path), "--model", "test/model", "--steps", "1"]
    )
    try:
        _validate(args)
    except SystemExit as error:
        assert "TINKER_API_KEY" in str(error)
    else:
        raise AssertionError("training accepted a missing Tinker API key")
