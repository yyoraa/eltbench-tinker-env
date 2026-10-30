# End-to-End ELT-Bench RL Environment

## Contract

Each example is a Harbor task with four pieces: an immutable source-data image,
a public natural-language target schema, a mutable workspace, and a hidden
execution grader. The policy receives a bash tool and may inspect data, write
SQL/dbt code, and run it. Submission ends the episode. The grader is uploaded
only after the rollout, so neither target rows nor credentials enter the prompt
or image.

The official `snowflake/books` example preserves ELT-Bench's heterogeneous
sources: Postgres, MongoDB, S3, flat files, and a declarative API connector. A
credentialed deployment performs ingestion with Airbyte and transformations in
Snowflake. The credential-free fixture uses DuckDB for fast CI while enforcing
the same reset, terminal-submission, idempotence, schema, and row contracts.

## Reward

The terminal reward is bounded in `[0, 1]`:

- 0.10: every requested target table exists;
- 0.20: exact requested column names and compatible types;
- 0.70: complete final contents match hidden ground truth.

Content comparison is deterministic and uses declared keys first, numeric
tolerance for floating-point values, case-insensitive trimmed strings, and
explicit null semantics. Full reward additionally requires the submitted
pipeline to succeed twice from the same post-reset state. Thus `CREATE TABLE IF
NOT EXISTS ... INSERT` cannot earn full reward by duplicating rows on rerun.

This outcome-heavy weighting prevents an agent from optimizing formatting or
tool activity while failing the ELT task. The small existence/schema components
provide learning signal before exact correctness.

## Reset and isolation

Every rollout gets a fresh sandbox and destination namespace. Public assets are
read-only. The local backend deletes and recreates its temporary workspace and
DuckDB database. The official runner must create a rollout-specific Snowflake
schema and drop it during cleanup. A terminal submission rejects further agent
actions.

## Reward-hacking defenses

1. Ground truth and grader are absent during the rollout and mounted afterward.
2. Credentials are runtime-only and never copied into task images or prompts.
3. The verifier checks warehouse state, not agent-authored logs or claims.
4. Nop, partial-schema, and append-only/idempotence adversaries are tests.
5. Exact tables, schema, row multiplicities, nulls, and deterministic ordering
   are checked; creating a plausibly named empty table earns at most 0.10.
6. Rollout and grading timeouts bound resource denial and infinite retries.

## Tinker integration

`eltbench-tinker-train` passes materialized tasks to Tinker Cookbook's real
`HarborDatasetBuilder` and `Config`. For each group it creates independent
sandboxes, obtains execution rewards from `/logs/verifier/reward.txt`, computes
within-group advantages, and calls the Tinker optimizer. `--steps 1` is a real
optimizer update, not a mocked training loop. The API key is mandatory and is
read only from the process environment.

## Scaling and efficiency

Use the DuckDB fixture for unit tests, then one credentialed `books` rollout as
the deployment gate. Cache immutable source images and connector images, but
never cache mutable warehouses. Parallelize groups across sandboxes while using
one schema per rollout. Start with group size 2 and one group per batch; increase
only after measuring success rate, rollout latency, verifier latency, Snowflake
credits, and reward variance. Suspend the X-Small warehouse after each batch.
