# Reproduction evidence

Everything below was run on this machine on 2026-10-31 unless it says otherwise. Runs used
scripted policies. **No language model was involved in any run.**

## Credential-free tests

| Command | Environment | Result |
|---|---|---|
| `.venv/bin/python -m pytest -q` | repo venv (no `chz`, no Tinker stack) | 44 passed, 2 skipped (Tinker-stack tests) |
| `python -m pytest -q` | training venv | 46 passed |

The Docker tests build a real image and run the hidden `tests/test.sh` in a real sandbox:
oracle 1.0, nop 0.0, append 0.625, plus forged-reward and tampering attempts.

## Mock Tinker wiring (a mock, not an optimizer step)

`tests/test_mock_tinker.py` replaces `tinker.ServiceClient`, the sampling client and the
tokenizer with fakes. No model is sampled and no weights change. Real parts: this repo's
`train.py`, the Cookbook's `cli_main` and `rl.train.main`, its Harbor env and `HarborReward`,
a Docker sandbox, the grader on DuckDB. Asserted: rewards `[0.0, 1.0]`, calls in the order
`forward_backward, optim_step`, loss `importance_sampling`, advantages `[-0.5, +0.5]`.

## Ground truth available for `books`

- `ground_truth_legacy/gt/books` in the checkout: authors 9,237 rows and publishers 2,270,
  against 9,235 and 2,264 in `evaluation/table.json`. It contains extra rows. Not used.
- `gt_snowflake/books` from the public `tttjjj/elt_bench` dataset named by the upstream
  README: 9,235 / 2,000 / 2,264 rows, agreeing with the raw tables. Used as the hidden answer.

Stage 2 is the upstream comparator, copied verbatim; a test fails if it drifts.

## Finding: `BOOKS.AIRBYTE_SCHEMA` passes stage 1 but cannot pass stage 2

The earlier "15/15 raw tables correct" result is row counts only. In that schema
`BOOK.TITLE` has type FLOAT, because Airbyte sampled numeric titles from MongoDB (`1984`,
`1776`). It is NULL in 11,122 of 11,127 rows, each flagged `DESTINATION_TYPECAST_ERROR` in
`_AIRBYTE_META`. Every target that reports a title depends on it. The schema was left
untouched as instructed.

A rollout's own connection can override the type (`--type book.title=string`). Done that
way, `BOOK.TITLE` is TEXT with 11,127 non-null values and no flagged rows in any of the 15
tables. `SEED_BOOKS_V1` was built like this. The prompt tells the model about the
mechanism, not about this column.

## Real warehouse rollouts

Command: `python -m eltbench_tinker.runner ... rollout <task> --policy <policy>`; database
`BOOKS`, X-Small warehouse, real Docker sandbox, real `tests/test.sh`. Raw results were kept
in `/data/yiqi/eltrun/runs/results.jsonl`, outside this repo.

| Mode | Policy | Reward | S1 | S2 state | S2 replay | Stops at | Total s | Reset s | Grade s |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| end_to_end | oracle (real Airbyte) | **1.0** | 1.0 | 1.0 | 1.0 | replay | 388.8 | 1.2 | 39.1 |
| end_to_end | nop | **0.0** | 0.0 | 0.0 | 0.0 | stage 1 | 13.3 | 1.0 | 9.6 |
| transform_only | oracle (run 1) | **1.0** | 1.0 | 1.0 | 1.0 | replay | 62.1 | 5.2 | 40.0 |
| transform_only | oracle (run 2) | **1.0** | 1.0 | 1.0 | 1.0 | replay | 57.3 | 5.2 | 36.8 |
| transform_only | oracle (run 3, rebuilt seed) | **1.0** | 1.0 | 1.0 | 1.0 | replay | 57.7 | 5.7 | 37.1 |
| transform_only | nop | **0.0** | 1.0 | 0.0 | 0.0 | stage 2 | 15.7 | 4.4 | 8.8 |
| transform_only | append | **0.625** | 1.0 | 1.0 | 0.0 | replay | 54.6 | 4.8 | 32.8 |
| transform_only | state_only | **0.625** | 1.0 | 1.0 | 0.0 | stage 2 | 32.9 | 5.7 | 12.7 |

Oracle runs: official stage-2 match on all 3 tables, idempotent, canary row reproduced.
Append: official match on its single run, then duplicated rows in replay. The end_to_end
oracle's policy ran through `elt-airbyte` inside the sandbox: 6 connections, 15 tables, 6
successful syncs; teardown deleted those 6 connections and 15 raw tables. `append` and
`state_only` were not run in end_to_end mode, to save warehouse time.

Three earlier transform_only runs (nop 93 s including a first image build, oracle 84 s,
append 85 s; rewards 0.0, 1.0, 0.625) used slower schema-level CLONE and per-table COUNT.

### Reset and stage-1 latency

Measured once by `clone_probe`, single connection unless noted, 15 tables:

| Operation | Seconds |
|---|---:|
| Login | 0.68 |
| `CREATE SCHEMA ... CLONE` | 15.5, 14.7 |
| 15 `CREATE TABLE ... CLONE`, serial | 19.2 |
| 15 `CREATE TABLE ... CLONE`, 8 threads (used) | 3.13 |
| `SHOW TABLES` with row counts (used for stage 1) | 0.11 |
| `DROP SCHEMA` | 0.16-0.25 |
| One `CREATE TABLE AS SELECT` of 22,348 rows, for contrast | 1.93 |

The role can clone. Stage-1 time in the table (about 4 s) is mostly driver import and login
in the grader process. Grade time also includes about 5 s of uploading `tests/`.

### Warehouse suspension

Requested suspend, but `ALTER WAREHOUSE ... SUSPEND` fails: `AIRBYTE_ROLE` lacks OPERATE.
Grants were not changed. Every run was followed by a poll of `SHOW WAREHOUSES`, which reported
`SUSPENDED` within 62 s (auto-suspend is 60 s). Runs that never woke it reported within 1 s.

### Seed

`build-seed` (6 concurrent Airbyte connections into `SEED_BOOKS_V1`, 15 raw tables, type
override applied): 286 s, all 6 succeeded, stage-1 fraction 1.0, then its 15 Airbyte raw tables
were dropped. Slowest syncs: order status file 274 s and API 274 s (concurrent, so wall
clock includes worker queueing). A transform_only oracle on the rebuilt seed: reward 1.0 in 57.7 s.

After all runs: `/data/yiqi/eltrun/check.py` reported `BOOKS.AIRBYTE_SCHEMA` 15/15; the database
held only `AIRBYTE_SCHEMA`, `SEED_BOOKS_V1`, `airbyte_internal`, `INFORMATION_SCHEMA`; the
original 6 Airbyte connections remained and no rollout connection; no warehouse was created
and all were suspended. Rollouts do add rows to `airbyte_internal._airbyte_destination_state`,
which teardown does not clean.

## Not run

- **Real Tinker optimizer step:** account returns HTTP 402; no paid call was made.
- **Any rollout by a language model**, so the prompt and limits are unvalidated.
- **Concurrent rollouts**, on Snowflake or in a group.
- **Modal backend.**
- **Model-created Airbyte sources or destination:** rollouts create only connections.
- **Other official tasks and warehouses.**

## Deployment issues found

- Rootless Docker without CPU delegation rejects `--cpus`; see `deployment.md`.
- Upstream REST API ignores `limit`/`offset`; see `deployment.md`.
- Airbyte reports `incomplete` for an attempt it is about to retry; the old `sync.py` treated
  that as final. Job 20 (order status file) failed its first attempt on a source error and
  succeeded on Airbyte's retry.
