# ELT-Bench Tinker RL environment

A Tinker RLVR recipe for [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench). A policy
gets a sandbox with a bash tool, Airbyte and a warehouse; it loads sources, writes
transformations and submits a pipeline. A hidden grader executes against the warehouse
and returns the reward. Tasks are Harbor directories run by Tinker Cookbook's own recipe.

- [Design](docs/design.md): contract, reward, defenses, latency, status.
- [Evidence](docs/evidence.md): exactly what was run, with numbers, and what was not.
- [Deployment](docs/deployment.md): fixes needed on rootless Docker.

**Status:** real-warehouse rollouts use scripted policies. No language model has acted in
this environment, and no real Tinker optimizer step has run. The Tinker test is a mock.

## Layout

| Path | Purpose |
|---|---|
| `src/eltbench_tinker/destination.py` | `Destination` protocol, registry, DuckDB backend |
| `src/eltbench_tinker/snowflake_destination.py` | Snowflake backend, one schema per rollout |
| `src/eltbench_tinker/official_grader.py` | Tiered grader and reward (runs as `tests/test.sh`) |
| `src/eltbench_tinker/official_stage2.py` | Upstream comparator, copied verbatim |
| `src/eltbench_tinker/docker_sandbox.py` | Local Docker `SandboxInterface` and `SandboxFactory` |
| `src/eltbench_tinker/airbyte.py`, `elt_sql.py` | Ingestion code and sandbox tools |
| `src/eltbench_tinker/runner.py`, `policies.py` | Credentialed runner and scripted policies |
| `src/eltbench_tinker/train.py` | Training entrypoint on the Cookbook's recipe |
| `oracle/books_pipeline.sql` | Reference solution; never enters an image or `tests/` |

## 1. Credential-free local tests

Needs Docker for the integration tests, which skip if the daemon is unreachable. No
warehouse, no Tinker account. The first run builds a small image.

```bash
uv sync --extra test
uv run python -m pytest -q
```

The mock Tinker test needs the training stack too. **It is a mock: it fakes the Tinker
service and changes no weights.** It runs the Cookbook's real loop, a Docker sandbox and
the hidden grader, and asserts the optimizer call receives advantages of +0.5 and -0.5
from rewards of 1.0 and 0.0.

```bash
uv venv /path/to/venv --python 3.13
uv pip install --python /path/to/venv/bin/python -e '.[snowflake,test]' \
  'tinker-cookbook[modal] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
/path/to/venv/bin/python -m pytest -q
```

The `modal` package is needed because the Cookbook imports it. A Modal account is not.

## 2. Materialize official tasks

The checkout's `ground_truth_legacy/` disagrees with `evaluation/table.json`, so use the
official v5 ground truth:

```bash
hf download tttjjj/elt_bench --repo-type dataset \
  --include "gt_snowflake/books/**" --local-dir /path/to/gt
python scripts/materialize_books.py --elt-bench ../ELT-Bench \
  --ground-truth /path/to/gt/gt_snowflake/books --output build/tasks
```

This writes `build/tasks/snowflake-books` (`end_to_end`) and
`build/tasks/snowflake-books-transform` (`transform_only`). `build/` is git-ignored because
`tests/` holds answers.

## 3. Credentialed rollout

Needs ELT-Bench sources and Airbyte running (see [deployment](docs/deployment.md)) and
a file of `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_ROLE`,
`SNOWFLAKE_WAREHOUSE`, read only at runtime. One command resets a rollout schema in
`BOOKS`, runs the policy in a sandbox, grades, drops the schema and connections, and
handles the warehouse:

```bash
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
RUN="python -m eltbench_tinker.runner --env-file ~/.config/elt-bench/snowflake.env \
  --airbyte-destination <destination-id>"

# Once: load raw tables into the seed that transform_only clones (about 5 minutes).
$RUN build-seed --elt-bench ../ELT-Bench

$RUN rollout build/tasks/snowflake-books-transform --policy oracle      # 1.0
$RUN rollout build/tasks/snowflake-books-transform --policy nop         # 0.0
$RUN rollout build/tasks/snowflake-books-transform --policy append      # 0.625
$RUN rollout build/tasks/snowflake-books-transform --policy state_only  # 0.625
$RUN rollout build/tasks/snowflake-books --policy oracle                # 1.0, with Airbyte
$RUN rollout build/tasks/snowflake-books --policy nop                   # 0.0
```

Each prints a JSON report with rewards, tiers and latencies. The sandbox joins the Airbyte
compose network (`--network`, default `airbyte-compose-06311_airbyte_public`).

## 4. Tinker training

Not run: the account used here returns HTTP 402. This is the command that would run one
optimizer step with execution-derived rewards:

```bash
TINKER_API_KEY=... eltbench-tinker-train build/tasks/snowflake-books-transform \
  --model <tinker-model> --steps 1 --group-size 2 --groups-per-batch 1 \
  --env-file ~/.config/elt-bench/snowflake.env \
  --airbyte-destination <destination-id>
```

It refuses to start without the key and validates each task's contract first.

## Adding a warehouse

Implement `Destination` and call `register_destination("name", factory)`. Materialize a
task whose `tests/grading.json` names it, and add a provisioner to
`provisioner_for_tasks` if rollouts need per-episode state. The environment and verifier
do not change; `tests/test_destination_interface.py` shows a complete fake backend.

Never put warehouse credentials in task images, prompts, logs or verifier artifacts.
