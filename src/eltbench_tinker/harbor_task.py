from __future__ import annotations

import json
import shutil
from pathlib import Path

from .official import OfficialELTBenchTask
from .official_grader import MODES
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


SNOWFLAKE_IMAGE_PACKAGES = (
    "snowflake-connector-python==3.17.3 pandas==2.2.3 numpy==2.2.6 pyyaml==6.0.2"
)
GRADER_MODULES = (
    "destination.py",
    "snowflake_destination.py",
    "official_stage2.py",
    "official_grader.py",
)
TEST_SH = """#!/bin/bash
# Hidden grader. Uploaded only after the rollout ends; runs as root, the policy does not.
set -u
pkill -KILL -u agent 2>/dev/null || true
chmod -R go-rwx /tests
rm -rf /logs/verifier
mkdir -p /logs/verifier
exec /opt/elt/venv/bin/python -sE -c "
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/tests')
from eltbench_grader.official_grader import main
sys.exit(main())
" --spec /tests/grading.json --destination {destination} \\
  --submission /workspace/submission/pipeline.sql --logs /logs/verifier
"""


def _official_dockerfile() -> str:
    return (
        "FROM python:3.11-slim\n"
        "RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends procps "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN python -m venv /opt/elt/venv && /opt/elt/venv/bin/pip install -q --no-cache-dir "
        f"{SNOWFLAKE_IMAGE_PACKAGES}\n"
        "COPY tools/ /opt/elt/tools/\n"
        "RUN printf '#!/bin/sh\\nexec /opt/elt/venv/bin/python -sE /opt/elt/tools/elt_sql.py \"$@\"\\n' "
        "> /usr/local/bin/elt-sql "
        "&& printf '#!/bin/sh\\nexec /opt/elt/venv/bin/python -sE /opt/elt/tools/airbyte.py \"$@\"\\n' "
        "> /usr/local/bin/elt-airbyte "
        "&& chmod 755 /usr/local/bin/elt-sql /usr/local/bin/elt-airbyte\n"
        "RUN useradd --create-home --uid 1000 agent && mkdir -p /logs/verifier\n"
        "COPY assets/ /workspace/\n"
        "RUN mkdir -p /workspace/submission && chown -R agent:agent /workspace\n"
        "WORKDIR /workspace\n"
    )


def materialize_official_task(
    task: OfficialELTBenchTask,
    output_dir: Path,
    *,
    ground_truth_dir: Path,
    table_json: Path,
    sort_key_json: Path,
    mode: str = "end_to_end",
    canary_sql: str | None = None,
    canary_check_sql: str | None = None,
) -> Path:
    """Materialize a complete official Harbor task: public image, hidden tests/.

    ``tests/`` holds ground truth and is uploaded only after the rollout, so the
    image and prompt never contain it. No credential is written anywhere.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    suffix = "" if mode == "end_to_end" else "-transform"
    root = Path(output_dir) / f"{task.task_id}{suffix}"
    shutil.rmtree(root, ignore_errors=True)
    env_dir = root / "environment"
    assets_dir = env_dir / "assets"
    tools_dir = env_dir / "tools"
    tests_dir = root / "tests"
    assets_dir.mkdir(parents=True)
    tools_dir.mkdir()
    (tests_dir / "ground_truth").mkdir(parents=True)
    package = Path(__file__).parent
    task.write_public_bundle(assets_dir)
    shutil.copy2(package / "airbyte.py", tools_dir / "airbyte.py")
    shutil.copy2(package / "elt_sql.py", tools_dir / "elt_sql.py")
    (env_dir / "Dockerfile").write_text(_official_dockerfile())
    _write_common_task_files(root, task.instruction(mode=mode))

    targets = [str(model["name"]) for model in task.data_model["models"]]
    gold_tables = []
    for table in targets:
        gold = Path(ground_truth_dir) / f"{table}.csv"
        if not gold.is_file():
            raise FileNotFoundError(f"ground truth missing for {table}: {gold}")
        shutil.copy2(gold, tests_dir / "ground_truth" / gold.name)
        gold_tables.append(table)
    grader_dir = tests_dir / "eltbench_grader"
    grader_dir.mkdir()
    (grader_dir / "__init__.py").write_text("")
    for module in GRADER_MODULES:
        shutil.copy2(package / module, grader_dir / module)
    (tests_dir / "grading.json").write_text(
        json.dumps(
            {
                "task": task.name,
                "mode": mode,
                "raw_tables": json.loads(Path(table_json).read_text())[task.name],
                "target_tables": gold_tables,
                "sort_keys": json.loads(Path(sort_key_json).read_text())[task.name],
                "ground_truth_dir": "ground_truth",
                "canary_sql": canary_sql,
                "canary_check_sql": canary_check_sql,
            },
            indent=1,
        )
    )
    (tests_dir / "test.sh").write_text(TEST_SH.format(destination=task.destination))
    (tests_dir / "test.sh").chmod(0o755)
    return root
