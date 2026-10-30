from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run execution-reward RL on materialized ELT Harbor tasks."
    )
    result.add_argument("task_dirs", nargs="+", type=Path)
    result.add_argument("--model", required=True)
    result.add_argument("--steps", type=int, default=1)
    result.add_argument("--group-size", type=int, default=2)
    result.add_argument("--groups-per-batch", type=int, default=1)
    result.add_argument("--max-turns", type=int, default=10)
    result.add_argument("--max-tokens", type=int, default=4096)
    result.add_argument("--learning-rate", type=float, default=1e-5)
    result.add_argument("--lora-rank", type=int, default=32)
    result.add_argument("--log-path", type=Path, default=Path("runs/eltbench-one-step"))
    result.add_argument("--base-url")
    return result


def _validate(args: argparse.Namespace) -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is required and must be injected at runtime")
    if args.steps < 1 or args.group_size < 2 or args.groups_per_batch < 1:
        raise SystemExit("steps >= 1, group-size >= 2, and groups-per-batch >= 1")
    for task_dir in args.task_dirs:
        required = (
            task_dir / "instruction.md",
            task_dir / "task.toml",
            task_dir / "environment" / "Dockerfile",
            task_dir / "tests" / "test.sh",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"invalid Harbor task {task_dir}: missing {missing}")


async def _run(args: argparse.Namespace) -> None:
    try:
        from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
        from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main
    except ImportError as error:
        raise SystemExit("install the 'tinker' extra before training") from error

    tasks = [
        HarborTask(
            task_name=task_dir.name,
            instruction=(task_dir / "instruction.md").read_text(),
            task_dir=task_dir.resolve(),
            config={},
        )
        for task_dir in args.task_dirs
    ]
    config = CLIConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        log_path=str(args.log_path),
        base_url=args.base_url,
        max_steps=args.steps,
        eval_every=0,
        save_every=max(args.steps, 1),
        behavior_if_log_dir_exists="delete",
    )
    await cli_main(config, tasks)


def main() -> None:
    args = parser().parse_args()
    _validate(args)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
