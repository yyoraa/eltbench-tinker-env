from __future__ import annotations

import json
import shutil
from pathlib import Path

from .official import OfficialELTBenchTask
from .task import ELTTask


def _write_common_task_files(root: Path, instruction: str) -> None:
    (root / "instruction.md").write_text(instruction.rstrip() + "\n")
    (root / "task.toml").write_text(
        'version = "1.0"\n[metadata]\ncategory = "elt-bench"\n'
    )


def materialize_local_harbor_task(task: ELTTask, output_dir: Path) -> Path:
    """Materialize a complete credential-free Harbor task and hidden grader."""
    root = Path(output_dir) / task.task_id
    shutil.rmtree(root, ignore_errors=True)
    env_dir = root / "environment"
    tests_dir = root / "tests"
    assets_dir = env_dir / "assets"
    assets_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    for source in task.source_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, assets_dir / source.name)
    _write_common_task_files(root, task.instruction)
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "RUN pip install --no-cache-dir duckdb==1.4.1\n"
        "WORKDIR /workspace\n"
        "COPY assets/ /workspace/\n"
        "ENV ELT_DUCKDB_PATH=/workspace/warehouse.duckdb\n"
    )
    hidden = {
        "tables": [
            {
                "name": table.name,
                "columns": table.columns,
                "rows": table.rows,
                "order_by": table.order_by,
            }
            for table in task.expected_tables
        ]
    }
    (tests_dir / "expected.json").write_text(json.dumps(hidden, default=str))
    grader_source = Path(__file__).with_name("harbor_grader.py")
    shutil.copy2(grader_source, tests_dir / "harbor_grader.py")
    (tests_dir / "test.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        "python /tests/harbor_grader.py "
        "/workspace/warehouse.duckdb /tests/expected.json "
        "/logs/verifier/reward.txt\n"
    )
    (tests_dir / "test.sh").chmod(0o755)
    return root


def materialize_official_public_task(
    task: OfficialELTBenchTask, output_dir: Path
) -> Path:
    """Materialize the public half of an official task without credentials/GT.

    A credentialed runner must mount destination credentials into the sandbox
    and provide a destination-specific hidden ``tests/test.sh`` at rollout time.
    """
    root = Path(output_dir) / task.task_id
    shutil.rmtree(root, ignore_errors=True)
    env_dir = root / "environment"
    assets_dir = env_dir / "assets"
    assets_dir.mkdir(parents=True)
    task.write_public_bundle(assets_dir)
    _write_common_task_files(root, task.instruction())
    (env_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "RUN pip install --no-cache-dir dbt-core pyyaml\n"
        "WORKDIR /workspace\nCOPY assets/ /workspace/\n"
    )
    return root
