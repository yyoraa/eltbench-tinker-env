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
- Explicit `submit()` terminal transition: execute the submitted pipeline,
  rerun it to test idempotence, grade the final state, and reject later actions.
- Hidden final-state verification of table existence, exact schema, and rows.
- Bounded reward where outcome correctness dominates shaping.
- ELT-Bench-compatible value/row comparison helpers: numeric tolerance,
  case-insensitive strings, null handling, deterministic key-first sorting.
- Transition and grading latency measurements on each local submission.
- Adapter that feeds Harbor-formatted ELT tasks into Tinker's existing
  `HarborDatasetBuilder` / `HarborEnvGroupBuilder`.
- Importer for the real `uiuc-kang-lab/ELT-Bench` repository. It reads a
  destination/task pair, target `data_model.yaml`, and shared source schemas,
  while stripping credentials from the agent-visible bundle.
- Harbor task materializer with a real `environment/Dockerfile`, immutable
  public assets, and a hidden `tests/test.sh` grader that writes Harbor's
  `/logs/verifier/reward.txt` contract.
- Integration tests for oracle success, nop failure, append/idempotence failure,
  terminal-state enforcement, partial reward, state isolation, official import
  redaction, and official comparison semantics.

## Run the local integration test

```bash
uv sync --extra test
uv run pytest -q
```

The local backend is only for trusted development and CI. Untrusted model
rollouts must use Harbor's container/Modal sandbox backend.

## Materialize the official books task

```bash
uv run python scripts/materialize_books.py \
  --elt-bench ../ELT-Bench --output build/tasks
```

The public materializer intentionally does not create `tests/`: the official
hidden verifier and warehouse credentials must be mounted by the credentialed
runner only after the rollout. For a completely runnable first optimizer-step
smoke, materialize the included local fixture from Python with
`materialize_local_harbor_task`.

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

The repository also provides a guarded one-command training entrypoint. It
refuses to start without the runtime API key and validates the hidden grader
contract before contacting Tinker:

```bash
uv run python scripts/materialize_local_smoke.py --output build/tasks
uv sync --extra tinker
TINKER_API_KEY=... uv run eltbench-tinker-train \
  build/tasks/customer-revenue \
  --model <tinker-model> --steps 1 --group-size 2 --groups-per-batch 1
```

This calls Tinker Cookbook's real Harbor RL recipe with `max_steps=1`. See
[`docs/design.md`](docs/design.md) for the environment/reward design and
[`docs/evidence.md`](docs/evidence.md) for what was and was not run, and
[`docs/deployment.md`](docs/deployment.md) for fixes needed on rootless Docker.

## Official ELT-Bench import

```python
from pathlib import Path
from eltbench_tinker.official import OfficialELTBenchTask
from eltbench_tinker.harbor_task import materialize_official_public_task

task = OfficialELTBenchTask.load(
    Path("../ELT-Bench"), destination="snowflake", name="books"
)
task_dir = materialize_official_public_task(task, Path("build/tasks"))
```

This creates the agent-visible half of the task only. The official repo's
ground-truth CSVs and destination credentials must stay outside the image and
prompt. A credentialed runner mounts credentials into the sandbox and supplies
the hidden verifier at grading time. This mirrors Tinker Harbor's behavior:
`tests/` is uploaded only after the agent episode ends.

## Still required for the official submission

The correct upstream repository is
`https://github.com/uiuc-kang-lab/ELT-Bench`. The importer and public Harbor
task contract are implemented and tested against its `snowflake/books` task.
The remaining external executions are:

1. mount one official warehouse credential into the sandbox and run one
   official task end to end;
2. keep the upstream stage-1 checks (raw table existence and row counts) and
   stage-2 checks (ground-truth target data, key sorting, numeric tolerance,
   case-insensitive strings) in the hidden verifier;
3. provide Tinker API access and run one optimizer step with the execution
   reward.

Before submission, the credentialed runner should also run the same three
adversarial acceptance checks used locally: oracle succeeds, nop fails, and a
non-idempotent append-only pipeline cannot receive full reward.

Do not place warehouse credentials in task images, prompts, logs, or verifier
artifacts.
