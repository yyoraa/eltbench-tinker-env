from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedHarborTask:
    """Minimal Harbor task contract consumed by Tinker Cookbook Harbor RL."""

    task_name: str
    instruction: str
    task_dir: Path
    config: dict[str, object]


def as_tinker_harbor_task(task_dir: Path) -> object:
    """Return Tinker's real HarborTask when the optional dependency exists."""
    instruction = (task_dir / "instruction.md").read_text()
    try:
        from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborTask
    except ImportError as error:
        raise RuntimeError(
            "Install the 'tinker' extra to create training environments"
        ) from error
    return HarborTask(
        task_name=task_dir.name,
        instruction=instruction,
        task_dir=task_dir,
        config={},
    )


def build_dataset_builder(
    task_dirs: list[Path],
    *,
    model_name: str,
    batch_size: int = 1,
    group_size: int = 4,
) -> object:
    """Wire ELT Harbor tasks into Tinker's EnvGroupBuilder and RL loop."""
    try:
        from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborDatasetBuilder
    except ImportError as error:
        raise RuntimeError(
            "Install the 'tinker' extra to build the RL dataset"
        ) from error
    return HarborDatasetBuilder(
        tasks=[as_tinker_harbor_task(path) for path in task_dirs],
        batch_size=batch_size,
        group_size=group_size,
        model_name=model_name,
    )

