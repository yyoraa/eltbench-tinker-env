from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .destination import DuckDBDestination
from .task import ELTTask
from .verifier import ELTVerifier, VerificationResult


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


class LocalELTEnvironment:
    """Credential-free episode runtime for development and CI.

    This backend is deliberately local and must not be used for untrusted
    models. Production rollouts use the Harbor sandbox adapter.
    """

    def __init__(self, task: ELTTask):
        self.task = task
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self.workspace: Path | None = None
        self.destination: DuckDBDestination | None = None

    def reset(self) -> str:
        self.cleanup()
        self._tmp = tempfile.TemporaryDirectory(prefix=f"elt-{self.task.task_id}-")
        root = Path(self._tmp.name)
        self.workspace = root / "workspace"
        shutil.copytree(self.task.source_dir, self.workspace)
        # Keep the local database beside the exposed source files so relative
        # paths in SQL behave exactly as they do inside a task container.
        self.destination = DuckDBDestination(self.workspace)
        self.destination.initialize()
        return self.task.instruction

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        if self.workspace is None or self.destination is None:
            raise RuntimeError("reset() must be called before run()")
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.workspace),
            "ELT_WORKSPACE": str(self.workspace),
            "ELT_DUCKDB_PATH": str(self.destination.db_path),
        }
        completed = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", command],
            cwd=self.workspace,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def execute_sql_file(self, relative_path: str) -> None:
        if self.workspace is None or self.destination is None:
            raise RuntimeError("reset() must be called before execute_sql_file()")
        path = (self.workspace / relative_path).resolve()
        if self.workspace.resolve() not in path.parents:
            raise ValueError("SQL path escapes workspace")
        self.destination.execute(path.read_text())

    def grade(self) -> VerificationResult:
        if self.destination is None:
            raise RuntimeError("reset() must be called before grade()")
        return ELTVerifier(self.task, self.destination).grade()

    def cleanup(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = None
        self.workspace = None
        self.destination = None
