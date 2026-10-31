"""A fake third warehouse proves the environment and verifier are warehouse-agnostic."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from eltbench_tinker.destination import (
    Destination,
    DuckDBDestination,
    create_destination,
    register_destination,
)
from eltbench_tinker.local_env import LocalELTEnvironment
from eltbench_tinker.snowflake_destination import SnowflakeConfig, SnowflakeDestination
from eltbench_tinker.task import ELTTask, ExpectedTable


class SQLiteDestination:
    """Third destination, defined only here: nothing in src/ knows it exists."""

    name = "fake-sqlite"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "fake.sqlite"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def reset(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)

    def execute(self, sql: str) -> None:
        with self._connect() as connection:
            connection.executescript(sql)

    def query(self, sql, parameters=()):
        with self._connect() as connection:
            return connection.execute(sql, parameters).fetchall()

    def list_tables(self):
        return [r[0] for r in self.query("SELECT name FROM sqlite_master WHERE type='table'")]

    def table_exists(self, table):
        return table in self.list_tables()

    def columns(self, table):
        return tuple((r[1], r[2].upper()) for r in self.query(f"PRAGMA table_info('{table}')"))

    def table_size(self, table):
        return self.query(f'SELECT count(*) FROM "{table}"')[0][0]

    def table_sizes(self):
        return {table: self.table_size(table) for table in self.list_tables()}

    def fetch_table(self, table):
        with self._connect() as connection:
            cursor = connection.execute(f'SELECT * FROM "{table}"')
            return [c[0] for c in cursor.description], cursor.fetchall()

    def clone(self, suffix):
        target = SQLiteDestination(self.root.parent / f"{self.root.name}{suffix}")
        target.reset()
        target.path.write_bytes(self.path.read_bytes())
        return target

    def runtime_env(self):
        return {"FAKE_SQLITE_PATH": str(self.path)}

    def submission_violations(self, sql):
        return ["attach forbidden"] if "ATTACH" in sql.upper() else []

    def cleanup(self):
        self.path.unlink(missing_ok=True)


@pytest.fixture
def task(tmp_path: Path) -> ELTTask:
    source = tmp_path / "source"
    source.mkdir()
    return ELTTask(
        task_id="fake",
        instruction="Create numbers(value INTEGER) holding 1 and 2.",
        source_dir=source,
        expected_tables=(
            ExpectedTable("numbers", (("value", "INTEGER"),), ((1,), (2,)), ("value",)),
        ),
    )


def test_protocol_is_satisfied_by_all_backends(tmp_path: Path) -> None:
    config = SnowflakeConfig("a", "u", "p")
    for backend in (
        DuckDBDestination(tmp_path),
        SQLiteDestination(tmp_path),
        SnowflakeDestination("R_X", config=config, connect=lambda *_: None),
    ):
        assert isinstance(backend, Destination)


def test_fake_destination_runs_full_episode_without_engine_changes(task: ELTTask) -> None:
    env = LocalELTEnvironment(task, destination_factory=SQLiteDestination)
    try:
        env.reset()
        assert env.run('echo "$FAKE_SQLITE_PATH"').stdout.strip().endswith("fake.sqlite")
        (env.workspace / "good.sql").write_text(
            "DROP TABLE IF EXISTS numbers; CREATE TABLE numbers(value INTEGER);"
            "INSERT INTO numbers VALUES (2),(1);"
        )
        result = env.submit("good.sql")
        assert result.reward == 1.0
        assert result.idempotent
    finally:
        env.cleanup()


def test_fake_destination_penalizes_append_and_nop(task: ELTTask) -> None:
    env = LocalELTEnvironment(task, destination_factory=SQLiteDestination)
    try:
        env.reset()
        (env.workspace / "nop.sql").write_text("SELECT 1;")
        assert env.submit("nop.sql").reward == 0.0
        env.reset()
        (env.workspace / "append.sql").write_text(
            "CREATE TABLE IF NOT EXISTS numbers(value INTEGER);"
            "INSERT INTO numbers VALUES (1),(2);"
        )
        result = env.submit("append.sql")
        assert not result.idempotent and result.reward < 1.0
    finally:
        env.cleanup()


def test_registry_builds_a_destination_added_at_runtime(tmp_path: Path) -> None:
    register_destination("fake-sqlite", SQLiteDestination)
    assert create_destination("fake-sqlite", root=tmp_path).name == "fake-sqlite"
    with pytest.raises(ValueError):
        create_destination("not-a-warehouse")


def test_engine_modules_import_no_concrete_warehouse() -> None:
    src = Path(__file__).parents[1] / "src" / "eltbench_tinker"
    for module in ("local_env.py", "verifier.py"):
        names: set[str] = set()
        for node in ast.walk(ast.parse((src / module).read_text())):
            if isinstance(node, ast.ImportFrom):
                names |= {alias.name for alias in node.names}
            elif isinstance(node, ast.Import):
                names |= {alias.name for alias in node.names}
        assert not names & {"DuckDBDestination", "SnowflakeDestination", "duckdb", "snowflake"}


class RecordingConnection:
    def __init__(self):
        self.statements: list[str] = []

    def cursor(self):
        return self

    def execute(self, sql, parameters=None):
        self.statements.append(sql)
        self.description = [("X",)]
        return self

    def fetchall(self):
        return []

    def close(self):
        pass


def _snowflake(schema: str, **kwargs) -> tuple[SnowflakeDestination, RecordingConnection]:
    connection = RecordingConnection()
    destination = SnowflakeDestination(
        schema, config=SnowflakeConfig("a", "u", "p"), connect=lambda *_: connection, **kwargs
    )
    return destination, connection


@pytest.mark.parametrize("schema", ["AIRBYTE_SCHEMA", "INFORMATION_SCHEMA", "airbyte_internal"])
def test_snowflake_refuses_benchmark_schemas(schema: str) -> None:
    with pytest.raises(ValueError):
        _snowflake(schema)


def test_snowflake_seed_schema_can_not_be_reset_or_dropped() -> None:
    destination, connection = _snowflake("SEED_BOOKS_V1")
    with pytest.raises(ValueError):
        destination.reset()
    destination.cleanup()
    assert not any("DROP" in sql for sql in connection.statements)


def test_snowflake_reset_clones_seed_and_cleanup_drops_only_rollout() -> None:
    destination, connection = _snowflake("r_abc", seed_schema="SEED_BOOKS_V1")
    destination.reset()
    destination.cleanup()
    assert connection.statements[:2] == [
        "CREATE OR REPLACE SCHEMA BOOKS.R_ABC",
        "SHOW TABLES IN SCHEMA BOOKS.SEED_BOOKS_V1",
    ]
    assert connection.statements[-1] == "DROP SCHEMA IF EXISTS BOOKS.R_ABC"


def test_snowflake_credentials_do_not_leak_through_repr() -> None:
    assert "hunter2" not in repr(SnowflakeConfig("acct", "someone", "hunter2"))
    assert "someone" not in repr(SnowflakeConfig("acct", "someone", "hunter2"))
