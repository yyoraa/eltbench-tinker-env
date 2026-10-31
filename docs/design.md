# ELT-Bench RL environment: implementation design

Evidence for every number here is in [evidence.md](evidence.md). A real model
rollout and a real Tinker optimizer step have **not** been run; see section 7.

## 1. Goal and scope

Train a language model with Tinker to solve official ELT-Bench tasks end to end,
rewarded by executing what it built. Implemented for one official task,
`snowflake/books`, with DuckDB as the credential-free destination. Warehouses are
pluggable through the `Destination` protocol (`destination.py`): the environment and
verifier import no concrete warehouse, and a test runs a full episode on a SQLite
destination defined only in the test file. A task is a Harbor directory, accepted
unchanged by the Cookbook's `HarborDatasetBuilder`.

## 2. Task and episode contract

- **Prompt** (`official.py`): target models, source tables, tool usage, submission
  rule. Credentials and ground truth are not in it.
- **Modes**: `end_to_end` starts from an empty schema; `transform_only` starts from a
  clone of loaded raw tables, for fast iteration.
- **Tools**: the Cookbook's single `bash` tool, in a container with `elt-airbyte`
  (sources, streams, connect, sync, wait) and `elt-sql`. SQL errors and sync states
  return as tool output, which is the execution feedback.
- **Limit of "end to end"**: the model creates connections on sources and a destination
  already configured in Airbyte. It does not create those. Nothing stops it calling the
  API to do so, but that path is untested.
- **Terminal conditions** (Cookbook `AgentToolMessageEnv`): a reply with no tool call,
  the turn limit (default 40 here), or the token budget. A submission is the file
  `/workspace/submission/pipeline.sql`. At the first hidden upload the sandbox kills
  the policy's processes and refuses its further commands.

## 3. State and reset

Each rollout gets a fresh container and a schema `R_<uuid>` in database `BOOKS`.
`SnowflakeDestination` can only reset or drop `R_*` schemas. `transform_only` clones
each table from `SEED_BOOKS_V1` concurrently with zero-copy `CLONE` (the role permits
it): 3.1 s for 15 tables, against 15 s for `CREATE SCHEMA ... CLONE`. Provisioning first
checks the seed's row counts. Teardown deletes the rollout's Airbyte connections, its
raw tables, the replay schema and the schema. The warehouse: see section 6.

## 4. Verifier and reward

A **correct pipeline** loads every raw table with the official row count, produces target
tables that match official ground truth, and reproduces them when rerun from raw tables.

`tests/test.sh` runs `official_grader.py`, cheapest tier first:

1. **Stage 1**: official raw-table row counts. Failure stops here, as upstream does.
2. **Stage 2**: the upstream comparator, copied verbatim and tested against the checkout
   (key sort, 1% numeric tolerance, case-insensitive strings, NULL handling, CSV round
   trip). `S2` is the matched fraction of gold columns; `S2 = 1` is the official match.
3. **Replay**: clone the schema, drop the targets, run the pipeline twice, compare again,
   then insert a canary source row, run once more and require it in the output.

```
end_to_end:      0.2*S1 + 0.5*S2_state + 0.3*S2_replay
transform_only: (0.5*S2_state + 0.3*S2_replay) / 0.8
```

Outcome is the warehouse state (`S1`, `S2_state`). Process is only whether the submitted
pipeline reproduces it (`S2_replay`). Nothing rewards tool use, turn count or messages.
`S1` is not credited in `transform_only`, because the rollout did not earn it. Any
grader crash writes reward 0 and exits non-zero.

Measured: oracle 1.0, nop 0.0, append 0.625, correct state with no pipeline 0.625.

## 5. Reward-hacking defenses

| Attack | Defense | Checked by |
|---|---|---|
| Read answers from image or prompt | `tests/` uploaded after the rollout; never in the image | test on materialized tree |
| Background process reads answers during upload | First upload kills the policy's processes; `/tests` is mode 700 | Docker test (found this hole) |
| Forge `reward.txt`, replace the interpreter | Policy runs as `agent`; grader as root with an absolute interpreter path | Docker test |
| Claim success in messages | Reward ignores history and reads warehouse state | grader code |
| Create plausible empty or wrong tables | Row count must match or the column scores 0 | unit test |
| Hand-edit state, no real pipeline | Replay from raw tables in a clone | Snowflake run (`state_only`) |
| Non-idempotent `INSERT` | Pipeline runs twice before comparison | Snowflake run (`append`) |
| Paste answer rows | Canary source row must appear | DuckDB unit test only |
| Copy from another schema | Static check rejects qualified references to other schemas | unit test |
| Leak credentials through output | Per-exec env; literal values scrubbed from output | Docker test |

**Known gaps.** The sandbox has unrestricted internet, and the ground truth is a public
dataset, so a model could download answers; pasting them would still lose all replay
credit to the canary. The static check is a filter, not a boundary: the model holds the shared
role's password, so it could damage `AIRBYTE_SCHEMA` or the seed, which only the seed's
row-count check would notice. A real fix needs a restricted per-rollout role, which was
out of scope here. Scrubbing misses encoded secrets. Upstream's comparator also ignores
extra columns and tolerates 1% numeric error, which we kept.

## 6. Tinker and efficiency

`eltbench-tinker-train` calls the Cookbook's `cli_main` with `DockerSandboxFactory`, so
the Cookbook's own env, reward parser, advantage centring and training loop run
unmodified. Modal is not used, though the Cookbook imports the `modal` package.

Measured on an X-Small warehouse, one rollout each unless noted:

| Step | Seconds |
|---|---|
| Reset, `transform_only` (seed check + login + clone; 6 runs) | 4.4-5.7 |
| Reset, `end_to_end` (empty schema) | 1.0-1.2 |
| Container start | 0.5-0.6 |
| Policy time, `end_to_end` oracle: Airbyte ingestion of 15 tables plus pipeline | 341 |
| Oracle pipeline via `elt-sql` | 12-13 |
| Grading: early exit at stage 1 (nop, `end_to_end`) | 9.6 |
| Grading: through stage 2 (`state_only`) | 12.7 |
| Grading: full replay (oracle) | 37-40 |
| Whole rollout, `transform_only` oracle | 57-62 |
| Whole rollout, `end_to_end` oracle | 389 |

Tiering is what the early exits buy: a failing rollout costs about a quarter of a passing
one, and stage 1 uses metadata-only `SHOW TABLES` that wakes no warehouse. Ingestion is
the bottleneck, which is why `transform_only` exists. A training plan would start there
and sample `end_to_end` sparsely; that schedule is not implemented. About 3.9 s of every
grade is driver import and login, and about 5 s is the Cookbook uploading `tests/` one
file at a time.

**Warehouse.** `AIRBYTE_ROLE` lacks OPERATE, so `ALTER WAREHOUSE ... SUSPEND` fails with
insufficient privileges. The runner tries, records `no-privilege`, then polls until the
60 s auto-suspend takes effect; it observed `SUSPENDED` after every recorded run, within
62 s. Grants were not changed, as instructed.

## 7. Implementation status and limitations

**Done, verified on real Snowflake and Airbyte:** the runner; both modes; oracle, nop,
append and `state_only` outcomes; Airbyte ingestion from inside the sandbox.

**Done, verified locally only:** destination protocol with a fake third backend; Docker
sandbox isolation; canary and static-check rejection; the rollout-to-optimizer path, using
a **mock** Tinker client on a DuckDB task.

**Not done:**
- No language model has ever acted in this environment. All policies are scripted, so
  whether the prompt, turn limit or token budget suffice for a real model is unknown.
- No real Tinker optimizer step: the account returns HTTP 402.
- Only `snowflake/books`. `task_streams` encodes this deployment's source naming.
- Group rollouts share one warehouse and have not been run concurrently.

Comparison with other work is limited to what was read: the Cookbook's Harbor recipe
supplies the task format and training loop reused here, and uses Modal by default.
