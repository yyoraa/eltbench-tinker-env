"""Credentialed one-command rollout for an official task.

reset rollout schema -> start sandbox -> (ingest) -> pipeline -> grade -> drop
schema and connections -> suspend warehouse. Every latency is measured. Credentials
are read from the env file at runtime and handed only to the Snowflake driver and
the sandbox's per-exec environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .docker_sandbox import DockerSandboxFactory
from .policies import BOOKS_TYPE_OVERRIDE, run_policy
from .provisioning import (
    DEFAULT_SANDBOX_NETWORK,
    DEFAULT_SEED_SCHEMA,
    SnowflakeProvisioner,
    docker_host_default,
    load_env_file,
    wait_for_warehouse_suspended,
)


async def run_rollout(args: argparse.Namespace) -> dict[str, Any]:
    from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward

    credentials = load_env_file(args.env_file)
    provisioner = SnowflakeProvisioner(
        credentials,
        seed_schema=args.seed_schema,
        airbyte_destination_id=args.airbyte_destination,
    )
    factory = DockerSandboxFactory(
        provisioner=provisioner,
        network=args.network,
        docker_host=args.docker_host or docker_host_default(),
    )
    task_dir = args.task_dir.resolve()
    result: dict[str, Any] = {"task": task_dir.name, "policy": args.policy}
    started = time.perf_counter()
    sandbox = await factory(task_dir / "environment", args.sandbox_timeout)
    result["seconds"] = dict(factory.last_timings)
    try:
        clock = time.perf_counter()
        result["steps"] = (await run_policy(args.policy, sandbox, task_dir))["steps"]
        result["seconds"]["policy_seconds"] = round(time.perf_counter() - clock, 3)

        clock = time.perf_counter()
        reward_fn = HarborReward(
            tests_dir=task_dir / "tests", sandbox=sandbox, grader_timeout=args.grader_timeout
        )
        reward, metrics = await reward_fn([])
        result["seconds"]["grade_seconds"] = round(time.perf_counter() - clock, 3)
        result["reward"] = reward
        result["reward_metrics"] = metrics
        grading = await sandbox.read_file("/logs/verifier/grading.json")
        if grading.exit_code == 0:
            result["grading"] = json.loads(grading.stdout)
    finally:
        clock = time.perf_counter()
        await sandbox.cleanup()
        result["seconds"]["cleanup_seconds"] = round(time.perf_counter() - clock, 3)
        result["teardown"] = provisioner.teardown_log
    result["seconds"]["total_seconds"] = round(time.perf_counter() - started, 3)
    result["warehouse_after"] = wait_for_warehouse_suspended(credentials)
    return result


def build_seed(args: argparse.Namespace) -> dict[str, Any]:
    """Load the raw tables once into the seed schema that rollouts clone."""
    import yaml

    from .airbyte import AirbyteClient, ingest, task_streams
    from .official_grader import stage1
    from .snowflake_destination import SnowflakeConfig, SnowflakeDestination

    credentials = load_env_file(args.env_file)
    config = yaml.safe_load((args.elt_bench / "elt-bench/snowflake/books/config.yaml").read_text())
    expected = json.loads((args.elt_bench / "evaluation/table.json").read_text())["books"]
    client = AirbyteClient()
    field, _, json_type = BOOKS_TYPE_OVERRIDE.partition("=")
    report = ingest(
        client,
        task_streams(config),
        args.seed_schema,
        destination=args.airbyte_destination,
        type_overrides={field: json_type},
    )
    for connection in report["connections"]:
        client.delete_connection(connection["connection_id"])
    seed = SnowflakeDestination(
        args.seed_schema, config=SnowflakeConfig.from_env(credentials)
    )
    try:
        fraction, failures = stage1(seed, expected)
        report["stage1_fraction"] = fraction
        report["stage1_failures"] = failures
        report["raw_tables_dropped"] = seed.drop_airbyte_raw_tables()
    finally:
        seed.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("~/.config/elt-bench/snowflake.env"))
    parser.add_argument("--seed-schema", default=DEFAULT_SEED_SCHEMA)
    parser.add_argument("--airbyte-destination", help="Airbyte destination id or name")
    parser.add_argument("--output", type=Path, help="append the JSON result to this file")
    commands = parser.add_subparsers(dest="command", required=True)

    rollout = commands.add_parser("rollout")
    rollout.add_argument("task_dir", type=Path)
    rollout.add_argument(
        "--policy", default="oracle", choices=["oracle", "nop", "append", "state_only", "ingest_only"]
    )
    rollout.add_argument("--network", default=DEFAULT_SANDBOX_NETWORK)
    rollout.add_argument("--docker-host")
    rollout.add_argument("--sandbox-timeout", type=int, default=3600)
    rollout.add_argument("--grader-timeout", type=int, default=300)

    seed = commands.add_parser("build-seed")
    seed.add_argument("--elt-bench", type=Path, required=True)
    args = parser.parse_args()

    result = asyncio.run(run_rollout(args)) if args.command == "rollout" else build_seed(args)
    text = json.dumps(result, indent=1, default=str)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as handle:
            handle.write(json.dumps(result, default=str) + "\n")


if __name__ == "__main__":
    main()
