"""Scripted policies: reference solution and adversaries.

They act through the same interface as a model: bash commands sent to the
sandbox with the Cookbook bash tool's ``workdir="/"``, so they run as ``agent``.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from .airbyte import task_streams

BOOKS_TYPE_OVERRIDE = "book.title=string"
ORACLE_SQL = Path(__file__).parents[2] / "oracle" / "books_pipeline.sql"
_TABLE = re.compile(r"CREATE OR REPLACE TABLE (\w+) AS\n(.*?);\s*(?=CREATE|\Z)", re.DOTALL)


def append_variant(sql: str) -> str:
    """Same logic, but rerunning duplicates every row."""
    parts = []
    for table, body in _TABLE.findall(sql):
        parts.append(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM (\n{body}\n) WHERE FALSE;")
        parts.append(f"INSERT INTO {table}\n{body};")
    if not parts:
        raise ValueError("no CREATE OR REPLACE TABLE statements found")
    return "\n".join(parts) + "\n"


def _write(path: str, content: str) -> str:
    return f"cat > {shlex.quote(path)} <<'__ELT_EOF__'\n{content}\n__ELT_EOF__"


SUBMISSION = "/workspace/submission/pipeline.sql"


async def _bash(sandbox: Any, command: str, *, timeout: int = 120) -> Any:
    return await sandbox.run_command(command, workdir="/", timeout=timeout)


async def _ingest_like_a_policy(sandbox: Any, task_dir: Path, log: list[dict[str, Any]]) -> None:
    """Load the raw tables with the sandbox's own Airbyte tool, one call per step."""
    config = yaml.safe_load((task_dir / "environment" / "assets" / "config.yaml").read_text())
    jobs: list[int] = []
    for source, streams in task_streams(config).items():
        command = (
            f"elt-airbyte connect {shlex.quote(source)} --streams {','.join(streams)} "
            f"--type {BOOKS_TYPE_OVERRIDE}"
        )
        result = await _bash(sandbox, command)
        log.append({"command": f"connect {source}", "exit_code": result.exit_code})
        if result.exit_code != 0:
            raise RuntimeError(f"connect failed for {source}: {result.stderr[-500:]}")
        connection = json.loads(result.stdout)["connection_id"]
        result = await _bash(sandbox, f"elt-airbyte sync {connection}")
        if result.exit_code != 0:
            raise RuntimeError(f"sync failed for {source}: {result.stderr[-500:]}")
        jobs.append(int(json.loads(result.stdout)["job_id"]))
    for job in jobs:
        for _ in range(40):
            result = await _bash(sandbox, f"elt-airbyte wait {job} --timeout 100")
            if result.exit_code != 0:
                raise RuntimeError(f"wait failed for job {job}: {result.stderr[-500:]}")
            status = json.loads(result.stdout)["status"]
            if status != "timeout":
                break
        log.append({"command": f"wait {job}", "status": status})
        if status != "succeeded":
            raise RuntimeError(f"job {job} ended as {status}")


async def run_policy(name: str, sandbox: Any, task_dir: Path) -> dict[str, Any]:
    """Run a scripted policy. Returns a step log without any command text secrets."""
    log: list[dict[str, Any]] = []
    spec = json.loads((task_dir / "tests" / "grading.json").read_text())
    if name == "nop":
        return {"policy": name, "steps": log}
    if spec["mode"] == "end_to_end":
        await _ingest_like_a_policy(sandbox, task_dir, log)
    sql = ORACLE_SQL.read_text()
    if name == "oracle":
        pipeline = sql
        script = _write(SUBMISSION, pipeline) + f"\nelt-sql -f {SUBMISSION}"
    elif name == "append":
        pipeline = append_variant(sql)
        script = _write(SUBMISSION, pipeline) + f"\nelt-sql -f {SUBMISSION}"
    elif name == "state_only":
        script = _write("/tmp/scratch.sql", sql) + "\nelt-sql -f /tmp/scratch.sql"
    elif name == "ingest_only":
        return {"policy": name, "steps": log}
    else:
        raise ValueError(f"unknown policy {name!r}")
    result = await _bash(sandbox, script)
    log.append({"command": "run pipeline", "exit_code": result.exit_code, "stderr": result.stderr[-500:]})
    return {"policy": name, "steps": log}
