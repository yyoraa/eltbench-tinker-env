# Reproduction Evidence

## Completed locally

- Credential-free oracle episode earns `1.0`.
- Nop earns `0.0`; an incomplete schema earns only partial reward.
- Append-only SQL fails the idempotence gate and cannot earn full reward.
- Reset removes prior warehouse state and submission terminates the episode.
- Official `snowflake/books` metadata imports with credentials redacted.
- Official row comparison semantics are covered by tests.
- Rootless Airbyte Compose deployment is healthy on Linux.
- The official ELT-Bench declarative API manifest was published; its `books`
  connection check succeeded.
- Books source data were loaded into Postgres (5 tables), MongoDB (3
  collections), S3 (2 objects), and the API service (3 endpoints).

Run the deterministic checks with `uv sync --extra test && uv run pytest -q`.

## Credentialed Snowflake ingestion (2026-10-31)

Real Airbyte syncs into a Snowflake `books` database on the official
`snowflake/books` task:

- Destination connection check succeeded.
- All 6 connections (Postgres, MongoDB, S3, 2 flat files, declarative API) synced.
- All 15 required raw tables match the official `evaluation/table.json` row
  counts exactly, for example `order_history` 22,348, `book_author` 17,642,
  `book` 11,127.

This verifies the extract-and-load stage against real warehouse state. It was
run by the operator, not by a model rollout.

## Not implemented yet

The repository has no Snowflake destination adapter, no grader or credentialed
runner for official tasks, and no way for a sandboxed model to reach Airbyte or
Snowflake. The ingestion above was done with operator scripts outside this repo.

## Not run

No Tinker optimizer step has been executed. `eltbench-tinker-train` is written
and refuses to start without `TINKER_API_KEY`, but the account used here was
blocked by billing (HTTP 402). The Cookbook default sandbox also needs a Modal
account, which was not configured. So there is no end-to-end model rollout or
execution-derived optimizer update to report yet, and neither should be assumed
to pass until run with those credentials.

The transformation stage and the official stage-2 verifier on Snowflake were
also not run; hidden-grader behaviour is covered by the local DuckDB tests.

Run the Tinker step with the command in the README using `--steps 1`, and retain
the run metadata plus the reward artifact.

## Deployment issues found

Rootless Docker without CPU cgroup delegation rejects the `--cpus` flag Airbyte
adds to connector containers (exit 125). The official REST API service ignores
`limit`/`offset`, so the published manifest's `OffsetIncrement` paginator repeats
forever on any table with at least 9,973 rows (`book_author`). See
`docs/deployment.md`.
