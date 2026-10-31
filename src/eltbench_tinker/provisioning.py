"""Per-rollout warehouse provisioning, selected from the task's hidden spec."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .airbyte import AirbyteClient
from .docker_sandbox import Provisioned, read_grading_spec
from .official_grader import REPLAY_SUFFIX
from .snowflake_destination import SnowflakeConfig, SnowflakeDestination

SANDBOX_AIRBYTE_URL = "http://airbyte-proxy:8000"
DEFAULT_SANDBOX_NETWORK = "airbyte-compose-06311_airbyte_public"
DEFAULT_SEED_SCHEMA = "SEED_BOOKS_V1"


def load_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines into a dict without touching ``os.environ``."""
    values: dict[str, str] = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def seed_problems(sizes: Mapping[str, int], expected: Mapping[str, int]) -> dict[str, Any]:
    actual = {name.upper(): size for name, size in sizes.items()}
    return {
        table: actual.get(table.upper())
        for table, rows in expected.items()
        if actual.get(table.upper()) != rows
    }


@dataclass
class SnowflakeProvisioner:
    """Reset one rollout schema: empty for end_to_end, seed clone for transform_only."""

    credentials: Mapping[str, str]
    database: str = "BOOKS"
    seed_schema: str = DEFAULT_SEED_SCHEMA
    airbyte_url: str = "http://localhost:8000"
    sandbox_airbyte_url: str = SANDBOX_AIRBYTE_URL
    airbyte_destination_id: str | None = None
    airbyte_username: str = "airbyte"
    airbyte_password: str = "password"
    teardown_log: list[dict[str, Any]] = field(default_factory=list)

    def airbyte(self) -> AirbyteClient:
        return AirbyteClient(self.airbyte_url, self.airbyte_username, self.airbyte_password)

    def __call__(self, task_dir: Path) -> Provisioned:
        spec = read_grading_spec(task_dir)
        schema = f"R_{uuid.uuid4().hex[:12].upper()}"
        config = SnowflakeConfig.from_env(self.credentials)
        seed = self.seed_schema if spec["mode"] == "transform_only" else None
        destination = SnowflakeDestination(
            schema, database=self.database, config=config, seed_schema=seed
        )
        clock = time.perf_counter()
        if seed:
            self._check_seed(config, seed, spec["raw_tables"])
        destination.reset()
        reset_seconds = round(time.perf_counter() - clock, 3)
        destination.close()
        env = destination.runtime_env()
        env.update(
            {
                "AIRBYTE_URL": self.sandbox_airbyte_url,
                "AIRBYTE_USERNAME": self.airbyte_username,
                "AIRBYTE_PASSWORD": self.airbyte_password,
            }
        )
        if self.airbyte_destination_id:
            env["AIRBYTE_DESTINATION_ID"] = self.airbyte_destination_id

        def teardown() -> None:
            self.teardown_log.append(self.teardown(schema, config))

        return Provisioned(
            env=env,
            teardown=teardown,
            seconds={"reset_seconds": reset_seconds, "reset_used_clone": float(seed is not None)},
        )

    def _check_seed(self, config: SnowflakeConfig, seed: str, expected: dict[str, int]) -> None:
        """Refuse to clone a seed an earlier rollout could have damaged (metadata only)."""
        reader = SnowflakeDestination(seed, database=self.database, config=config)
        try:
            problems = seed_problems(reader.table_sizes(), expected)
        finally:
            reader.close()
        if problems:
            raise RuntimeError(f"seed schema {seed} failed its row-count check: {problems}")

    def teardown(self, schema: str, config: SnowflakeConfig) -> dict[str, Any]:
        """Remove everything the rollout created: connections, raw tables, schemas."""
        clock = time.perf_counter()
        report: dict[str, Any] = {"schema": schema}
        client = self.airbyte()
        connections = client.connections_for_namespace(schema)
        for connection in connections:
            client.delete_connection(connection)
        report["connections_deleted"] = len(connections)
        destination = SnowflakeDestination(schema, database=self.database, config=config)
        try:
            report["raw_tables_dropped"] = destination.drop_airbyte_raw_tables()
            destination._run(f"DROP SCHEMA IF EXISTS {destination.fqn}{REPLAY_SUFFIX}")
            destination._run(f"DROP SCHEMA IF EXISTS {destination.fqn}")
            report["suspend"] = destination.suspend_warehouse()
        finally:
            destination.close()
        report["teardown_seconds"] = round(time.perf_counter() - clock, 3)
        return report


def wait_for_warehouse_suspended(
    credentials: Mapping[str, str], database: str = "BOOKS", limit: int = 240
) -> dict[str, Any]:
    """Poll warehouse state. SHOW WAREHOUSES does not itself resume a warehouse."""
    destination = SnowflakeDestination(
        "R_STATE_POLL", database=database, config=SnowflakeConfig.from_env(credentials)
    )
    clock = time.perf_counter()
    state = "unknown"
    try:
        while time.perf_counter() - clock < limit:
            state = destination.warehouse_state()
            if state == "SUSPENDED":
                break
            time.sleep(5)
    finally:
        destination.close()
    return {"state": state, "seconds_until_observed": round(time.perf_counter() - clock, 1)}


def provisioner_for_tasks(
    snowflake: Callable[[Path], Provisioned] | None,
) -> Callable[[Path], Provisioned]:
    """Dispatch on the destination named by a task's grading spec."""

    def provision(task_dir: Path) -> Provisioned:
        path = Path(task_dir) / "tests" / "grading.json"
        destination = read_grading_spec(task_dir).get("destination") if path.is_file() else "duckdb"
        if destination == "snowflake":
            if snowflake is None:
                raise RuntimeError(f"task {task_dir} needs Snowflake credentials")
            return snowflake(task_dir)
        return Provisioned()

    return provision


def docker_host_default() -> str | None:
    return os.environ.get("DOCKER_HOST") or (
        f"unix:///run/user/{os.getuid()}/docker.sock"
        if Path(f"/run/user/{os.getuid()}/docker.sock").exists()
        else None
    )
