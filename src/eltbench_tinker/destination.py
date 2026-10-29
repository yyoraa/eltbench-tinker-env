from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass
class DuckDBDestination:
    """Credential-free destination used by local integration tests.

    The API is intentionally destination-specific. A Snowflake adapter can
    implement the same methods without changing the environment or verifier.
    """

    root: Path

    @property
    def db_path(self) -> Path:
        return self.root / "warehouse.duckdb"

    def reset(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.initialize()

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path.unlink(missing_ok=True)
        duckdb.connect(str(self.db_path)).close()

    def execute(self, sql: str) -> None:
        with duckdb.connect(str(self.db_path)) as connection:
            connection.execute("SET file_search_path = ?", (str(self.root),))
            connection.execute(sql)

    def query(
        self, sql: str, parameters: tuple[object, ...] = ()
    ) -> list[tuple[object, ...]]:
        with duckdb.connect(str(self.db_path), read_only=True) as connection:
            return connection.execute(sql, parameters).fetchall()

    def columns(self, table: str) -> tuple[tuple[str, str], ...]:
        rows = self.query(f"PRAGMA table_info('{table}')")
        return tuple((str(row[1]), str(row[2]).upper()) for row in rows)

    def table_exists(self, table: str) -> bool:
        rows = self.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_name=?",
            (table,),
        )
        return bool(rows)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
