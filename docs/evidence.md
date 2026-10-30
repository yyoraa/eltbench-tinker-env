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

## Credentialed gates

These must remain explicitly separate from local evidence:

1. Snowflake destination check and `books` sync require the `books` database
   plus `USAGE, CREATE SCHEMA` granted to the dedicated Airbyte role.
2. A real optimizer update requires a runtime `TINKER_API_KEY` and a configured
   Harbor sandbox backend. Run the command in the README with `--steps 1` and
   retain the Tinker run metadata plus reward artifact.

Do not report either gate as completed solely because local tests pass.
