from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_SECRET_KEYS = {
    "password",
    "secret",
    "secret_access_key",
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

    def instruction(self) -> str:
        source_names = ", ".join(sorted(self.schemas))
        targets = []
        for model in self.data_model["models"]:
            columns = ", ".join(column["name"] for column in model.get("columns", []))
            targets.append(f"- {model['name']}: {columns}")
        return (
            f"Complete the official ELT-Bench task '{self.name}' for {self.destination}.\n"
            "Use the supplied source configuration and schema documentation to extract/load "
            "the sources and build the requested target models. Credentials are injected only "
            "at runtime and must never be printed or copied.\n\n"
            f"Source tables: {source_names}\n\nTarget models:\n" + "\n".join(targets)
        )

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
