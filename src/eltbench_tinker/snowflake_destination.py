"""Snowflake destination: one disposable schema per rollout in one database."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .destination import Row

ROLLOUT_SCHEMA = re.compile(r"^R_[A-Z0-9_]{1,200}$")
SEED_SCHEMA = re.compile(r"^SEED_[A-Z0-9_]{1,200}$")
RAW_SCHEMA = '"airbyte_internal"'

# Statements a submitted pipeline has no business running. The pipeline executes
# with the shared role, so this is an accident and cheat filter, not a security
# boundary; see docs/design.md for what a restricted role would add.
_FORBIDDEN = re.compile(
    r"\b(USE|GRANT|REVOKE|ALTER|CALL|PUT|GET|UNDROP|EXECUTE|COPY|REMOVE|"
    r"CREATE\s+(OR\s+REPLACE\s+)?(DATABASE|SCHEMA|WAREHOUSE|ROLE|USER|STAGE|PIPE|TASK|PROCEDURE|FUNCTION)|"
    r"DROP\s+(DATABASE|SCHEMA|WAREHOUSE|ROLE|USER|STAGE))\b",
    re.IGNORECASE,
)
_DYNAMIC = re.compile(r"\b(IDENTIFIER\s*\(|SYSTEM\$)", re.IGNORECASE)
_QUALIFIED = re.compile(r'("?[A-Za-z_][A-Za-z0-9_$]*"?)\s*\.\s*"?[A-Za-z_]')


def strip_sql_literals(sql: str) -> str:
    """Remove comments and single-quoted strings so scans see only SQL tokens."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return re.sub(r"--[^\n]*|//[^\n]*", " ", sql)


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str
    user: str
    password: str
    role: str | None = None
    warehouse: str | None = None
    host: str | None = None

    def __repr__(self) -> str:  # never let credentials reach a log line
        return f"SnowflakeConfig(role={self.role!r}, warehouse={self.warehouse!r})"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SnowflakeConfig":
        source = os.environ if env is None else env
        missing = [
            key
            for key in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
            if not source.get(key)
        ]
        if missing:
            raise RuntimeError(f"missing runtime Snowflake settings: {missing}")
        return cls(
            account=source["SNOWFLAKE_ACCOUNT"],
            user=source["SNOWFLAKE_USER"],
            password=source["SNOWFLAKE_PASSWORD"],
            role=source.get("SNOWFLAKE_ROLE") or None,
            warehouse=source.get("SNOWFLAKE_WAREHOUSE") or None,
            host=source.get("SNOWFLAKE_HOST") or None,
        )


def _default_connect(config: SnowflakeConfig, database: str) -> Any:
    import snowflake.connector

    kwargs: dict[str, Any] = {
        "account": config.account,
        "user": config.user,
        "password": config.password,
        "database": database,
        "login_timeout": 60,
        "network_timeout": 300,
        "session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": 300, "QUERY_TAG": "eltbench-tinker"},
    }
    if config.role:
        kwargs["role"] = config.role
    if config.warehouse:
        kwargs["warehouse"] = config.warehouse
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return snowflake.connector.connect(**kwargs)


class SnowflakeDestination:
    """Destination bound to one rollout schema.

    ``reset`` either creates an empty schema (the model must ingest) or, when
    ``seed_schema`` is given, a zero-copy clone of it. Only ``R_*`` schemas can
    ever be reset or dropped, so the benchmark's own schema cannot be destroyed
    through this class.
    """

    name = "snowflake"

    def __init__(
        self,
        schema: str,
        *,
        database: str = "BOOKS",
        config: SnowflakeConfig | None = None,
        seed_schema: str | None = None,
        connect: Callable[[SnowflakeConfig, str], Any] = _default_connect,
    ):
        schema = schema.upper()
        if not (ROLLOUT_SCHEMA.match(schema) or SEED_SCHEMA.match(schema)):
            raise ValueError(f"refusing non-rollout schema {schema!r}")
        self.schema = schema
        self.database = database.upper()
        self.seed_schema = seed_schema
        self._config = config if config is not None else SnowflakeConfig.from_env()
        self._connect = connect
        self._connection: Any = None

    @classmethod
    def from_env(cls, **kwargs: Any) -> "SnowflakeDestination":
        """Build from the sandbox/runner environment (``ELT_TARGET_*``)."""
        return cls(
            kwargs.pop("schema", os.environ["ELT_TARGET_SCHEMA"]),
            database=kwargs.pop("database", os.environ.get("ELT_TARGET_DATABASE", "BOOKS")),
            **kwargs,
        )

    @property
    def fqn(self) -> str:
        return f"{self.database}.{self.schema}"

    def _cursor(self) -> Any:
        if self._connection is None:
            self._connection = self._connect(self._config, self.database)
        return self._connection.cursor()

    def _run(self, sql: str, parameters: tuple[object, ...] = ()) -> Any:
        cursor = self._cursor()
        cursor.execute(sql, parameters or None)
        return cursor

    def reset(self) -> None:
        if not ROLLOUT_SCHEMA.match(self.schema):
            raise ValueError(f"reset only applies to R_* schemas, not {self.schema!r}")
        if self.seed_schema:
            self._run(
                f"CREATE OR REPLACE SCHEMA {self.fqn} CLONE {self.database}.{self.seed_schema}"
            )
        else:
            self._run(f"CREATE OR REPLACE SCHEMA {self.fqn}")
        self._run(f"USE SCHEMA {self.fqn}")

    def execute(self, sql: str) -> None:
        self._cursor()
        self._run(f"USE SCHEMA {self.fqn}")
        for cursor in self._connection.execute_string(sql, remove_comments=True):
            cursor.close()

    def query(self, sql: str, parameters: tuple[object, ...] = ()) -> list[Row]:
        self._run(f"USE SCHEMA {self.fqn}")
        return [tuple(row) for row in self._run(sql, parameters).fetchall()]

    def list_tables(self) -> list[str]:
        rows = self.query(
            f"SELECT table_name FROM {self.database}.information_schema.tables "
            "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
            (self.schema,),
        )
        return [str(row[0]) for row in rows]

    def table_exists(self, table: str) -> bool:
        return table.upper() in {name.upper() for name in self.list_tables()}

    def columns(self, table: str) -> tuple[tuple[str, str], ...]:
        rows = self.query(
            f"SELECT column_name, data_type FROM {self.database}.information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (self.schema, table.upper()),
        )
        return tuple((str(row[0]), str(row[1]).upper()) for row in rows)

    def table_size(self, table: str) -> int:
        return int(self.query(f"SELECT COUNT(*) FROM {self.fqn}.{table}")[0][0])

    def fetch_table(self, table: str) -> tuple[list[str], list[Row]]:
        self._run(f"USE SCHEMA {self.fqn}")
        cursor = self._run(f"SELECT * FROM {self.fqn}.{table}")
        names = [item[0] for item in cursor.description]
        return names, [tuple(row) for row in cursor.fetchall()]

    def clone(self, suffix: str) -> "SnowflakeDestination":
        target = SnowflakeDestination(
            f"{self.schema}{suffix}",
            database=self.database,
            config=self._config,
            seed_schema=self.schema,
            connect=self._connect,
        )
        target.reset()
        return target

    def runtime_env(self) -> dict[str, str]:
        env = {
            "SNOWFLAKE_ACCOUNT": self._config.account,
            "SNOWFLAKE_USER": self._config.user,
            "SNOWFLAKE_PASSWORD": self._config.password,
            "ELT_TARGET_DATABASE": self.database,
            "ELT_TARGET_SCHEMA": self.schema,
        }
        for key, value in (
            ("SNOWFLAKE_ROLE", self._config.role),
            ("SNOWFLAKE_WAREHOUSE", self._config.warehouse),
            ("SNOWFLAKE_HOST", self._config.host),
        ):
            if value:
                env[key] = value
        return env

    def submission_violations(self, sql: str) -> list[str]:
        tokens = strip_sql_literals(sql)
        violations = [f"forbidden statement: {m.group(0).upper()}" for m in _FORBIDDEN.finditer(tokens)]
        violations += [f"forbidden dynamic SQL: {m.group(0)}" for m in _DYNAMIC.finditer(tokens)]
        known = {row[0].upper() for row in self._schema_names()} | {self.database}
        for match in _QUALIFIED.finditer(tokens):
            qualifier = match.group(1).strip('"').upper()
            if qualifier in known and qualifier != self.schema:
                violations.append(f"reference outside the rollout schema: {qualifier}")
        return sorted(set(violations))

    def _schema_names(self) -> list[Row]:
        cursor = self._run(f"SELECT schema_name FROM {self.database}.information_schema.schemata")
        return [tuple(row) for row in cursor.fetchall()]

    def drop_airbyte_raw_tables(self) -> int:
        """Drop the Airbyte raw tables this rollout's namespace created."""
        pattern = f"{self.schema}_raw__stream_".replace("_", "\\\\_") + "%"
        rows = self._run(
            f"SHOW TABLES LIKE '{pattern}' IN SCHEMA {self.database}.{RAW_SCHEMA}"
        ).fetchall()
        prefix = f"{self.schema}_raw__stream_"
        names = [str(row[1]) for row in rows if str(row[1]).startswith(prefix)]
        for name in names:
            self._run(f'DROP TABLE IF EXISTS {self.database}.{RAW_SCHEMA}."{name}"')
        return len(names)

    def warehouse_state(self) -> str:
        warehouse = self._config.warehouse
        if not warehouse:
            return "unknown"
        cursor = self._run(f"SHOW WAREHOUSES LIKE '{warehouse}'")
        names = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
        return str(dict(zip(names, rows[0]))["state"]) if rows else "unknown"

    def suspend_warehouse(self) -> str:
        """Try to suspend. Returns ok, already, or a no-privilege marker."""
        warehouse = self._config.warehouse
        if not warehouse:
            return "no-warehouse-configured"
        try:
            self._run(f"ALTER WAREHOUSE {warehouse} SUSPEND")
            return "suspended"
        except Exception as error:  # driver errors are not importable without the extra
            text = str(error)
            if "Insufficient privileges" in text:
                return "no-privilege"
            if "cannot be suspended" in text or "already suspended" in text.lower():
                return "already-suspended"
            return "error"

    def cleanup(self) -> None:
        try:
            if ROLLOUT_SCHEMA.match(self.schema):
                self._run(f"DROP SCHEMA IF EXISTS {self.fqn}")
        finally:
            self.close()

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None
