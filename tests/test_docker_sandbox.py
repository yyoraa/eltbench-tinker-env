"""Credential-free integration: real Docker sandbox, real hidden grader, real rewards.

Skipped only when Docker is unavailable. Uses the DuckDB task, so no warehouse,
no Tinker account and no network beyond the first image build.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from eltbench_tinker.docker_sandbox import DockerSandbox, DockerSandboxFactory
from eltbench_tinker.harbor_task import materialize_local_harbor_task
from eltbench_tinker.task import ELTTask, ExpectedTable

ROOTLESS = f"unix:///run/user/{os.getuid()}/docker.sock"
HOST = os.environ.get("DOCKER_HOST") or (ROOTLESS if Path(ROOTLESS[7:]).exists() else None)
GOOD = (
    "CREATE OR REPLACE TABLE customer_revenue AS "
    "SELECT c.customer_id, c.name, CAST(COALESCE(SUM(o.amount), 0) AS DECIMAL(38,2)) AS revenue "
    "FROM read_csv_auto('customers.csv') c LEFT JOIN read_csv_auto('orders.csv') o "
    "USING (customer_id) GROUP BY c.customer_id, c.name;"
)
APPEND = (
    "CREATE TABLE IF NOT EXISTS customer_revenue "
    "(customer_id BIGINT, name VARCHAR, revenue DECIMAL(38,2));"
    "INSERT INTO customer_revenue SELECT c.customer_id, c.name, "
    "CAST(COALESCE(SUM(o.amount), 0) AS DECIMAL(38,2)) FROM read_csv_auto('customers.csv') c "
    "LEFT JOIN read_csv_auto('orders.csv') o USING (customer_id) GROUP BY c.customer_id, c.name;"
)


def docker_works() -> bool:
    if not shutil.which("docker"):
        return False
    env = {**os.environ, **({"DOCKER_HOST": HOST} if HOST else {})}
    return subprocess.run(["docker", "info"], env=env, capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not docker_works(), reason="Docker daemon unavailable")


@pytest.fixture(scope="module")
def task_dir(tmp_path_factory) -> Path:
    fixture = Path(__file__).parent / "fixtures" / "customer_revenue"
    task = ELTTask(
        task_id="customer-revenue",
        instruction="Create customer_revenue(customer_id, name, revenue) from the two CSV files.",
        source_dir=fixture,
        expected_tables=(
            ExpectedTable(
                "customer_revenue",
                (("customer_id", "BIGINT"), ("name", "VARCHAR"), ("revenue", "DECIMAL(38,2)")),
                ((1, "Ada", 20.00), (2, "Linus", 9.00), (3, "Grace", 0.00)),
                ("customer_id",),
            ),
        ),
    )
    return materialize_local_harbor_task(task, tmp_path_factory.mktemp("tasks"))


def sandbox_for(task_dir: Path, **env: str) -> DockerSandbox:
    factory = DockerSandboxFactory(docker_host=HOST, extra_env=env)
    return asyncio.run(factory(task_dir / "environment", 300))


async def grade(sandbox: DockerSandbox, task_dir: Path) -> float:
    await sandbox.run_command("mkdir -p /tests", workdir="/")
    for path in (task_dir / "tests").rglob("*"):
        if path.is_file():
            target = f"/tests/{path.relative_to(task_dir / 'tests')}"
            await sandbox.write_file(target, path.read_text(), executable=path.suffix == ".sh")
    result = await sandbox.run_command("bash /tests/test.sh", workdir="/root", timeout=120)
    assert result.exit_code == 0, result.stderr
    return float((await sandbox.read_file("/logs/verifier/reward.txt")).stdout)


def rollout(task_dir: Path, pipeline: str | None) -> tuple[float, str]:
    sandbox = sandbox_for(task_dir)

    async def run() -> tuple[float, str]:
        try:
            if pipeline is not None:
                write = (
                    "cat > /workspace/submission/pipeline.sql <<'EOF'\n" + pipeline + "\nEOF\n"
                    "cd /workspace && elt-sql -f /workspace/submission/pipeline.sql"
                )
                done = await sandbox.run_command(write, workdir="/")
                assert done.exit_code == 0, done.stderr
            reward = await grade(sandbox, task_dir)
            report = (await sandbox.read_file("/logs/verifier/grading.json")).stdout
            return reward, report
        finally:
            await sandbox.cleanup()

    return asyncio.run(run())


def test_oracle_earns_full_reward(task_dir: Path) -> None:
    reward, report = rollout(task_dir, GOOD)
    assert reward == 1.0, report


def test_nop_earns_zero(task_dir: Path) -> None:
    reward, report = rollout(task_dir, None)
    assert reward == 0.0, report


def test_append_is_penalized(task_dir: Path) -> None:
    reward, report = rollout(task_dir, APPEND)
    assert reward == 0.625, report


def test_policy_runs_unprivileged_and_grader_as_root(task_dir: Path) -> None:
    sandbox = sandbox_for(task_dir)

    async def run() -> None:
        try:
            who = await sandbox.run_command("id -un", workdir="/")
            assert who.stdout.strip() == "agent"
            root = await sandbox.run_command("id -un", workdir="/root")
            assert root.stdout.strip() == "root"
        finally:
            await sandbox.cleanup()

    asyncio.run(run())


def test_policy_cannot_forge_reward_or_tamper_with_the_grader(task_dir: Path) -> None:
    sandbox = sandbox_for(task_dir)

    async def run() -> None:
        try:
            for attack in (
                "mkdir -p /logs/verifier && echo 1.0 > /logs/verifier/reward.txt",
                "echo 'echo 1.0' > /opt/elt/venv/bin/python",
                "mkdir /tests",
            ):
                result = await sandbox.run_command(attack, workdir="/")
                assert result.exit_code != 0, attack
            # A process left running by the policy must not survive to race the grader.
            await sandbox.run_command(
                "nohup sh -c 'while true; do cp /tests/ground_truth/* /tmp/ 2>/dev/null; sleep 0.2; done' "
                ">/dev/null 2>&1 &",
                workdir="/",
                timeout=5,
            )
            assert await grade(sandbox, task_dir) == 0.0
            left = await sandbox.run_command(
                "ps -u agent -o pid,stat,cmd --no-headers; echo ---; ls /tmp | wc -l", workdir="/root"
            )
            print(left.stdout)
            processes, stolen = left.stdout.split("---")
            assert processes.strip() == ""
            assert stolen.strip() == "0"
            read = await sandbox.run_command("cat /tests/ground_truth/customer_revenue.csv", workdir="/")
            assert read.exit_code != 0
        finally:
            await sandbox.cleanup()

    asyncio.run(run())


def test_secrets_are_scrubbed_and_absent_from_container_config(task_dir: Path) -> None:
    marker = "not-a-real-secret-value-8675309"
    sandbox = sandbox_for(task_dir, SNOWFLAKE_PASSWORD=marker)

    async def run() -> None:
        try:
            shown = await sandbox.run_command('echo "pw=$SNOWFLAKE_PASSWORD"; env', workdir="/")
            assert marker not in shown.stdout
            assert "pw=[REDACTED]" in shown.stdout
            env = {**os.environ, **({"DOCKER_HOST": HOST} if HOST else {})}
            inspect = subprocess.run(
                ["docker", "inspect", sandbox.sandbox_id], env=env, capture_output=True, text=True
            )
            assert inspect.returncode == 0 and marker not in inspect.stdout
        finally:
            await sandbox.cleanup()

    asyncio.run(run())


def test_cookbook_contract_is_satisfied(task_dir: Path) -> None:
    pytest.importorskip("chz")
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook.sandbox import SandboxInterface

    sandbox = sandbox_for(task_dir)
    try:
        assert isinstance(sandbox, SandboxInterface)
    finally:
        asyncio.run(sandbox.cleanup())
