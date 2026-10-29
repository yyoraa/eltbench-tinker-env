from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
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


@dataclass(frozen=True)
class SubmissionResult:
    reward: float
    verification: VerificationResult
    idempotent: bool
    transition_seconds: float
    grade_seconds: float


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
        self._done = False

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
        self._done = False
        return self.task.instruction

    def run(self, command: str, timeout: int = 60) -> CommandResult:
        if self.workspace is None or self.destination is None:
            raise RuntimeError("reset() must be called before run()")
        if self._done:
            raise RuntimeError("episode has terminated; call reset()")
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

    def submit(self, relative_sql_path: str) -> SubmissionResult:
        """Execute once, re-execute to verify idempotence, grade, then terminate.

        The second execution is part of hidden outcome verification. A solution
        that appends duplicates can be content-correct after its first run but
        cannot receive the idempotence component.
        """
        if self._done:
            raise RuntimeError("episode has already been submitted")
        started = time.perf_counter()
        self.execute_sql_file(relative_sql_path)
        assert self.destination is not None
        first_state = self.destination.state_fingerprint()
        self.execute_sql_file(relative_sql_path)
        second_state = self.destination.state_fingerprint()
        transition_seconds = time.perf_counter() - started
        grade_started = time.perf_counter()
        verification = self.grade()
        grade_seconds = time.perf_counter() - grade_started
        idempotent = first_state == second_state
        reward = round(0.9 * verification.reward + (0.1 if idempotent else 0.0), 8)
        self._done = True
        return SubmissionResult(
            reward=reward,
            verification=verification,
            idempotent=idempotent,
            transition_seconds=transition_seconds,
            grade_seconds=grade_seconds,
        )

    def cleanup(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = None
        self.workspace = None
        self.destination = None
        self._done = False
