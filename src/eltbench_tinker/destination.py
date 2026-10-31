"""Warehouse destination interface and the credential-free DuckDB backend.

The environment and verifier only depend on the ``Destination`` protocol. Adding
a warehouse means implementing this protocol and calling
``register_destination``; ``local_env.py``, ``verifier.py`` and
``official_grader.py`` do not change (``tests/test_destination_interface.py``
proves this with a fake SQLite destination).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

Row = tuple[Any, ...]


@runtime_checkable
class Destination(Protocol):
    """Everything the environment and grader may ask of a warehouse."""

    name: str

    def reset(self) -> None:
        """Make this rollout's namespace exist and contain only its initial state."""
        ...

    def execute(self, sql: str) -> None:
        """Run one or more statements inside this rollout's namespace."""
        ...

    def query(self, sql: str, parameters: tuple[object, ...] = ()) -> list[Row]:
        """Run a read query inside this rollout's namespace."""
        ...

    def list_tables(self) -> list[str]:
        """Base tables in this namespace, in the warehouse's native case."""
        ...

    def table_exists(self, table: str) -> bool:
        ...

    def columns(self, table: str) -> tuple[tuple[str, str], ...]:
        """(name, upper-cased type) pairs in ordinal order."""
        ...

    def table_size(self, table: str) -> int:
        ...

    def table_sizes(self) -> dict[str, int]:
        """Row count of every base table, as cheaply as the warehouse allows."""
        ...

    def fetch_table(self, table: str) -> tuple[list[str], list[Row]]:
        """Column names and all rows. Row order is not defined."""
        ...

    def clone(self, suffix: str) -> "Destination":
        """A disposable, independent copy of this namespace (for replay grading)."""
        ...

    def runtime_env(self) -> dict[str, str]:
        """Environment variables a rollout needs to reach this namespace."""
        ...

    def submission_violations(self, sql: str) -> list[str]:
        """Reasons a submitted pipeline may not be executed by the grader."""
        ...

    def cleanup(self) -> None:
        """Remove this rollout's namespace and release connections."""
        ...


_REGISTRY: dict[str, Callable[..., Destination]] = {}


def register_destination(name: str, factory: Callable[..., Destination]) -> None:
    _REGISTRY[name] = factory


def create_destination(name: str, **kwargs: Any) -> Destination:
    if name not in _REGISTRY and name == "snowflake":
        # Imported on demand so credential-free runs need no warehouse driver.
        from .snowflake_destination import SnowflakeDestination

        register_destination("snowflake", SnowflakeDestination.from_env)
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown destination {name!r}; have {sorted(_REGISTRY)}") from None
    return factory(**kwargs)


def _sort_token(value: object) -> tuple[int, float, str]:
    """Total order over mixed warehouse values; NULLs last, numbers numerically."""
    if value is None:
        return (2, 0.0, "")
    if not isinstance(value, bool):
        try:
            return (0, float(value), "")  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    return (1, 0.0, str(value))


def canonical_rows(rows: Sequence[Sequence[object]], first: Sequence[int] = ()) -> list[Row]:
    """Deterministically sort rows: ``first`` column indices, then the remainder."""

    def key(row: Sequence[object]) -> list[tuple[int, float, str]]:
        rest = [index for index in range(len(row)) if index not in first]
        return [_sort_token(row[index]) for index in [*first, *rest]]

    return [tuple(row) for row in sorted(rows, key=key)]


def state_fingerprint(destination: Destination) -> str:
    """Stable hash of all tables, schemas and canonically sorted rows."""
    digest = hashlib.sha256()
    for table in sorted(destination.list_tables()):
        digest.update(table.encode())
        digest.update(repr(destination.columns(table)).encode())
        _, rows = destination.fetch_table(table)
        digest.update(repr(canonical_rows(rows)).encode())
    return digest.hexdigest()


@dataclass
class DuckDBDestination:
    """Credential-free destination used by local integration tests."""

    root: Path
    name = "duckdb"

    @property
    def db_path(self) -> Path:
        return self.root / "warehouse.duckdb"

    def _connect(self, read_only: bool = False) -> Any:
        import duckdb

        if read_only and not self.db_path.exists():
            # Nothing was ever written (a nop rollout): an empty warehouse, not an error.
            self.root.mkdir(parents=True, exist_ok=True)
            duckdb.connect(str(self.db_path)).close()
        return duckdb.connect(str(self.db_path), read_only=read_only)

    def reset(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path.unlink(missing_ok=True)
        self._connect().close()

    def execute(self, sql: str) -> None:
        with self._connect() as connection:
            # A submitted pipeline runs as root during grading. Allow exactly its
            # own source files so it cannot read answers or write the reward file.
            files = [path for path in self.root.iterdir() if path.is_file()]
            allowed = [str(path.resolve()) for path in files] + [path.name for path in files]
            connection.execute("SET file_search_path = ?", (str(self.root),))
            connection.execute("SET allowed_paths = ?", (allowed,))
            connection.execute("SET enable_external_access = false")
            connection.execute(sql)

    def query(self, sql: str, parameters: tuple[object, ...] = ()) -> list[Row]:
        with self._connect(read_only=True) as connection:
            return connection.execute(sql, parameters).fetchall()

    def list_tables(self) -> list[str]:
        rows = self.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' ORDER BY table_name"
        )
        return [str(row[0]) for row in rows]

    def table_exists(self, table: str) -> bool:
        return table in self.list_tables()

    def columns(self, table: str) -> tuple[tuple[str, str], ...]:
        rows = self.query(f"PRAGMA table_info('{table}')")
        return tuple((str(row[1]), str(row[2]).upper()) for row in rows)

    def table_size(self, table: str) -> int:
        return int(self.query(f'SELECT count(*) FROM "{table}"')[0][0])

    def table_sizes(self) -> dict[str, int]:
        return {table: self.table_size(table) for table in self.list_tables()}

    def fetch_table(self, table: str) -> tuple[list[str], list[Row]]:
        with self._connect(read_only=True) as connection:
            cursor = connection.execute(f'SELECT * FROM "{table}"')
            names = [item[0] for item in cursor.description]
            return names, cursor.fetchall()

    def clone(self, suffix: str) -> "DuckDBDestination":
        target = self.root.parent / f"{self.root.name}{suffix}"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        # Source files travel with the database: pipelines read them by relative path.
        for path in self.root.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)
        return DuckDBDestination(target)

    def runtime_env(self) -> dict[str, str]:
        return {"ELT_DUCKDB_PATH": str(self.db_path)}

    def submission_violations(self, sql: str) -> list[str]:
        # The DuckDB file is private to the rollout; there is nothing to escape to.
        return []

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _duckdb_from_env(root: Path | None = None) -> DuckDBDestination:
    if root is None:
        root = Path(os.environ.get("ELT_DUCKDB_PATH", "/workspace/warehouse.duckdb")).parent
    return DuckDBDestination(Path(root))


register_destination("duckdb", _duckdb_from_env)
