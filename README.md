# ELT-Bench Tinker Environment

Prototype bridge from an ELT benchmark task to an executable, verifiable RL
environment. It deliberately separates:

- the public task instruction and source files;
- a destination-specific warehouse adapter;
- the mutable rollout workspace;
- hidden expected warehouse state and grading;
- Tinker Cookbook's existing Harbor `EnvGroupBuilder` and RL training loop.

## What works

- Credential-free DuckDB episode reset and cleanup.
- Multi-turn command workspace plus explicit SQL execution.
- Hidden final-state verification of table existence, exact schema, and rows.
- Bounded reward where outcome correctness dominates shaping.
- Adapter that feeds Harbor-formatted ELT tasks into Tinker's existing
  `HarborDatasetBuilder` / `HarborEnvGroupBuilder`.
- Integration tests for success, partial reward, and state isolation.

## Run the local integration test

```bash
uv sync --extra test
uv run pytest -q
```

The local backend is only for trusted development and CI. Untrusted model
rollouts must use Harbor's container/Modal sandbox backend.

## Tinker wiring

Install Tinker Cookbook and materialize each official ELT task as:

```text
task-name/
├── environment/Dockerfile
├── tests/test.sh
├── instruction.md
└── task.toml
```

Then:

```python
from pathlib import Path
from eltbench_tinker.tinker_adapter import build_dataset_builder

builder = build_dataset_builder(
    [Path("tasks/customer-revenue")],
    model_name="<tinker-model>",
    group_size=4,
)
```

Pass `builder` to Tinker Cookbook's standard RL `Config`. The cookbook creates
one fresh sandbox per rollout, gives the agent a bash tool, uploads hidden tests
only after the episode, parses `/logs/verifier/reward.txt`, computes group
advantages, and performs the optimizer update.

## Still required for the official submission

The paper points to `https://github.com/uiuc-kang-lab/ETL`, but that repository
currently returns `Repository not found`. An official task cannot be truthfully
ported or run until Daniel's task assets/repository access and warehouse
credentials are available. Once provided, add:

1. an official ELT-Bench task importer;
2. the official destination adapter (for example Snowflake);
3. hidden verifier fixtures derived from the benchmark's expected models;
4. one credentialed official rollout;
5. one Tinker optimizer step with execution-derived rewards.

Do not place warehouse credentials in task images, prompts, logs, or verifier
artifacts.
