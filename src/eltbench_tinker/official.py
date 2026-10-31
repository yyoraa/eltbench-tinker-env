from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_SECRET_KEYS = {
    "password",
    "secret",
    "secret_access_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "connection_string",
    "access_key_id",
    "client_id",
    "username",
    "user",
    "account",
    "host",
    "hostname",
    "http_path",
    "workspace_id",
}


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("<injected-at-runtime>" if key.lower() in _SECRET_KEYS else _redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


@dataclass(frozen=True)
class OfficialELTBenchTask:
    """Credential-free public portion of one official ELT-Bench task."""

    repo_root: Path
    destination: str
    name: str
    config: dict[str, Any]
    data_model: dict[str, Any]
    schemas: dict[str, str]

    @classmethod
    def load(
        cls, repo_root: Path, destination: str, name: str
    ) -> "OfficialELTBenchTask":
        repo_root = Path(repo_root).resolve()
        task_dir = repo_root / "elt-bench" / destination / name
        schema_dir = repo_root / "elt-bench" / "schemas" / name
        if not task_dir.is_dir():
            raise FileNotFoundError(f"official task does not exist: {task_dir}")
        if not schema_dir.is_dir():
            raise FileNotFoundError(f"official schemas do not exist: {schema_dir}")
        config = yaml.safe_load((task_dir / "config.yaml").read_text())
        data_model = yaml.safe_load((task_dir / "data_model.yaml").read_text())
        schemas = {
            path.stem: path.read_text()
            for path in sorted(schema_dir.glob("*.csv"))
        }
        if not schemas:
            raise ValueError(f"official task has no source schemas: {schema_dir}")
        if not data_model.get("models"):
            raise ValueError(f"official task has no target models: {task_dir}")
        return cls(
            repo_root=repo_root,
            destination=destination,
            name=name,
            config=_redact_secrets(config),
            data_model=data_model,
            schemas=schemas,
        )

    @property
    def task_id(self) -> str:
        return f"{self.destination}-{self.name}"

    def instruction(self, *, mode: str = "end_to_end", runtime: str = "") -> str:
        source_names = ", ".join(sorted(self.schemas))
        targets = []
        for model in self.data_model["models"]:
            columns = ", ".join(column["name"] for column in model.get("columns", []))
            targets.append(f"- {model['name']}: {columns}")
        lines = [
            f"Complete the official ELT-Bench task '{self.name}' for {self.destination}.",
            "",
            "Files (read these first, use `cd /workspace`):",
            "- data_model.yaml: every target table, its columns, and how each is defined.",
            "- config.yaml: the task's sources and which tables each one provides. "
            "Credentials in it are redacted.",
            "- schemas/<table>.csv: source column documentation.",
            "",
        ]
        if mode == "end_to_end":
            lines += [
                "Your warehouse schema starts empty. Extract and load the sources with Airbyte, "
                "then build the targets. Load only the tables config.yaml lists:",
                "- `elt-airbyte sources` lists the configured sources.",
                "- `elt-airbyte streams SOURCE [--only a,b]` shows streams and inferred types.",
                "- `elt-airbyte connect SOURCE --streams a,b [--type stream.field=string]` writes "
                "them into your schema. Airbyte infers types from a sample and nulls values that "
                "do not conform; check _AIRBYTE_META on the loaded tables and override the type "
                "if a column came through wrong.",
                "- `elt-airbyte sync CONNECTION_ID`, then `elt-airbyte wait JOB_ID` (repeat until "
                "it reports a final status; a sync takes minutes).",
                "",
            ]
        else:
            lines += [
                "The raw source tables are already loaded in your warehouse schema. Do not "
                "reload them; build the targets from them.",
                "",
            ]
        lines += [
            "Run SQL with `elt-sql -e 'SELECT ...'` or `elt-sql -f file.sql`. It runs inside your "
            "schema, so use unqualified table names. Credentials and your schema name are in the "
            "environment; never print or copy them.",
            "",
            "Submit by writing the final idempotent transformation to "
            "/workspace/submission/pipeline.sql: statements only, unqualified table names, "
            "each target rebuilt from the raw tables. It must not depend on anything else you "
            "created. The episode ends when you reply without calling a tool. Then the warehouse "
            "state is graded and your pipeline is rerun twice on a copy of your schema.",
            "",
            f"Source tables: {source_names}",
            "",
            "Target models:",
            *targets,
        ]
        if runtime:
            lines += ["", runtime]
        return "\n".join(lines)

    def write_public_bundle(self, output_dir: Path) -> Path:
        """Write only agent-visible assets; ground truth never enters this tree."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "schemas").mkdir(exist_ok=True)
        (output_dir / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        (output_dir / "data_model.yaml").write_text(
            yaml.safe_dump(self.data_model, sort_keys=False)
        )
        for table, contents in self.schemas.items():
            (output_dir / "schemas" / f"{table}.csv").write_text(contents)
        return output_dir
