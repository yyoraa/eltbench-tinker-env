"""Local Docker backend for ``tinker_cookbook.sandbox.SandboxInterface``.

Lets the Cookbook's Harbor recipe run without Modal, including under rootless
Docker. The policy's commands run as the unprivileged ``agent`` user; the
Cookbook's grader runs as root, so a policy cannot rewrite the interpreter,
read uploaded answers, or pre-write the reward file.

Who is who follows from the Cookbook's own calls: its bash tool always passes
``workdir="/"`` and its ``HarborReward`` runs the grader with ``workdir="/root"``.
The policy controls the command string but never this argument. The first upload
under ``/tests`` also kills everything the policy left running and ends its access.

Credentials reach the container only as per-exec environment: ``docker exec -e
NAME`` takes the value from the client's environment, so it does not appear in a
process listing, ``docker inspect``, or the image.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from tinker_cookbook.sandbox.sandbox_interface import SandboxResult, SandboxTerminatedError
except ImportError:  # credential-free runs do not need the training stack

    @dataclass
    class SandboxResult:  # type: ignore[no-redef]
        stdout: str
        stderr: str
        exit_code: int
        metrics: dict[str, Any] = field(default_factory=dict)

    class SandboxTerminatedError(RuntimeError):  # type: ignore[no-redef]
        pass


DEFAULT_MAX_OUTPUT = 128 * 1024
GRADER_WORKDIR = "/root"
AGENT_USER = "agent"
HIDDEN_PREFIX = "/tests"
SECRET_ENV_KEYS = ("PASSWORD", "ACCOUNT", "HOST", "SNOWFLAKE_USER", "TOKEN", "SECRET", "API_KEY")
_REDACTED = "[REDACTED]"


def _is_secret(key: str, value: str) -> bool:
    # The local Airbyte default password is the English word; hiding it would only
    # corrupt tool output. Host names and short values are not credentials either.
    if key == "AIRBYTE_PASSWORD" and value == "password":
        return False
    if key == "AIRBYTE_URL":
        return False
    return len(value) >= 4 and any(marker in key.upper() for marker in SECRET_ENV_KEYS)


async def _drain(stream: asyncio.StreamReader | None, limit: int) -> bytes:
    if stream is None:
        return b""
    kept = bytearray()
    while chunk := await stream.read(65536):
        if len(kept) < limit:
            kept.extend(chunk[: limit - len(kept)])
    return bytes(kept)


class DockerSandbox:
    def __init__(
        self,
        container: str,
        *,
        client_env: Mapping[str, str],
        runtime_env: Mapping[str, str],
        on_cleanup: Callable[[], Any] | None = None,
    ):
        self._container = container
        self._client_env = dict(client_env)
        self._runtime = dict(runtime_env)
        self._on_cleanup = on_cleanup
        self._closed = False
        self._sealed = False
        self._secrets = sorted(
            {value for key, value in self._runtime.items() if _is_secret(key, value)},
            key=len,
            reverse=True,
        )

    @property
    def sandbox_id(self) -> str:
        return self._container

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, _REDACTED)
        return text

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def _exec(
        self,
        command: str,
        *,
        user: str,
        workdir: str | None,
        timeout: int,
        max_output_bytes: int | None = None,
        stdin: bytes | None = None,
    ) -> SandboxResult:
        limit = max_output_bytes or DEFAULT_MAX_OUTPUT
        argv = ["docker", "exec", "-u", user]
        if stdin is not None:
            argv.append("-i")
        if workdir:
            argv += ["-w", workdir]
        home = "/root" if user == "root" else f"/home/{user}"
        argv += ["-e", f"HOME={home}"]
        for name in self._runtime:
            argv += ["-e", name]
        argv += [
            self._container, "timeout", "-k", "5", str(timeout),
            "bash", "--noprofile", "--norc", "-c", command,
        ]  # fmt: skip
        environ = {**self._client_env, **self._runtime}
        process = await asyncio.create_subprocess_exec(
            *argv,
            env=environ,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if stdin is not None and process.stdin is not None:
            process.stdin.write(stdin)
            await process.stdin.drain()
            process.stdin.close()
        try:
            out, err = await asyncio.wait_for(
                asyncio.gather(_drain(process.stdout, limit), _drain(process.stderr, limit)),
                timeout + 15,
            )
            code = await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()
            return SandboxResult(stdout="", stderr=f"timed out after {timeout}s", exit_code=124)
        stderr = err.decode(errors="replace")
        if code in (125, 126, 137) and ("No such container" in stderr or "is not running" in stderr):
            raise SandboxTerminatedError(f"container {self._container} is gone")
        return SandboxResult(
            stdout=self._redact(out.decode(errors="replace")),
            stderr=self._redact(stderr),
            exit_code=code,
        )

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        user = "root" if workdir == GRADER_WORKDIR else AGENT_USER
        if user == AGENT_USER and self._sealed:
            return SandboxResult(stdout="", stderr="episode has ended", exit_code=1)
        return await self._exec(
            command, user=user, workdir=workdir, timeout=timeout, max_output_bytes=max_output_bytes
        )

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        quoted = shlex.quote(path)
        command = f"head -c {int(max_bytes)} {quoted}" if max_bytes else f"cat {quoted}"
        return await self._exec(command, user="root", workdir=None, timeout=timeout)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        data = content.encode() if isinstance(content, str) else content
        quoted = shlex.quote(path)
        await self.seal_if_hidden(path)
        command = f'umask 077; mkdir -p "$(dirname {quoted})" && cat > {quoted}'
        if executable:
            command += f" && chmod +x {quoted}"
        return await self._exec(command, user="root", workdir=None, timeout=timeout, stdin=data)

    async def seal_if_hidden(self, path: str) -> None:
        """End the policy before the first hidden file lands in the container.

        The Cookbook uploads tests/ file by file, so a process the policy left
        behind could otherwise read answers before test.sh gets to run.
        """
        if self._sealed or not path.startswith(HIDDEN_PREFIX):
            return
        self._sealed = True
        await self._exec(
            f"pkill -KILL -u {AGENT_USER}; mkdir -p -m 700 {HIDDEN_PREFIX}; chmod 700 {HIDDEN_PREFIX}",
            user="root",
            workdir=None,
            timeout=30,
        )

    async def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", self._container,
            env=self._client_env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        await process.wait()
        if self._on_cleanup is not None:
            await asyncio.to_thread(self._on_cleanup)


@dataclass
class Provisioned:
    """Warehouse state prepared for one rollout."""

    env: dict[str, str] = field(default_factory=dict)
    teardown: Callable[[], Any] | None = None
    seconds: dict[str, float] = field(default_factory=dict)


def _docker_client_env(docker_host: str | None) -> dict[str, str]:
    env = {key: os.environ[key] for key in ("PATH", "HOME", "XDG_RUNTIME_DIR") if key in os.environ}
    host = docker_host or os.environ.get("DOCKER_HOST")
    if host:
        env["DOCKER_HOST"] = host
    return env


def context_digest(env_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(env_dir).rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(env_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


class DockerSandboxFactory:
    """Cookbook ``SandboxFactory``: ``await factory(env_dir, timeout)``.

    ``provisioner(task_dir)`` prepares warehouse state for a new rollout and is
    selected per task, so the factory itself knows nothing about any warehouse.
    Images are built once per context digest and reused across rollouts.
    """

    def __init__(
        self,
        *,
        provisioner: Callable[[Path], Provisioned] | None = None,
        network: str | None = None,
        docker_host: str | None = None,
        memory: str = "4g",
        pids: int = 512,
        extra_env: Mapping[str, str] | None = None,
    ):
        self._provisioner = provisioner
        self._network = network
        self._client_env = _docker_client_env(docker_host)
        self._memory = memory
        self._pids = pids
        self._extra_env = dict(extra_env or {})
        self._locks: dict[str, asyncio.Lock] = {}
        self.last_timings: dict[str, float] = {}

    async def _docker(self, *args: str, timeout: int = 900) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            "docker", *args,
            env=self._client_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )  # fmt: skip
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
        return process.returncode or 0, output.decode(errors="replace")

    async def ensure_image(self, env_dir: Path) -> str:
        tag = f"eltbench-sandbox:{context_digest(env_dir)}"
        lock = self._locks.setdefault(tag, asyncio.Lock())
        async with lock:
            code, _ = await self._docker("image", "inspect", tag)
            if code != 0:
                code, output = await self._docker("build", "-q", "-t", tag, str(env_dir))
                if code != 0:
                    raise RuntimeError(f"docker build failed: {output[-2000:]}")
        return tag

    async def __call__(self, env_dir: Path, timeout: int) -> DockerSandbox:
        timings: dict[str, float] = {}
        clock = time.perf_counter()
        image = await self.ensure_image(Path(env_dir))
        timings["image_seconds"] = round(time.perf_counter() - clock, 3)

        provisioned = Provisioned()
        if self._provisioner is not None:
            clock = time.perf_counter()
            provisioned = await asyncio.to_thread(self._provisioner, Path(env_dir).parent)
            timings["provision_seconds"] = round(time.perf_counter() - clock, 3)
            timings.update(provisioned.seconds)

        clock = time.perf_counter()
        name = f"eltbench-{uuid.uuid4().hex[:12]}"
        argv = [
            "run", "-d", "--rm", "--name", name, "--init",
            "--memory", self._memory, "--pids-limit", str(self._pids),
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--cap-add", "KILL", "--cap-add", "DAC_OVERRIDE",
            "--cap-add", "FOWNER", "--cap-add", "CHOWN",
        ]  # fmt: skip
        if self._network:
            argv += ["--network", self._network]
        argv += [image, "sleep", str(timeout)]
        try:
            code, output = await self._docker(*argv, timeout=120)
            if code != 0:
                raise RuntimeError(f"docker run failed: {output[-500:]}")
        except Exception:
            if provisioned.teardown is not None:
                await asyncio.to_thread(provisioned.teardown)
            raise
        timings["container_seconds"] = round(time.perf_counter() - clock, 3)
        self.last_timings = timings
        return DockerSandbox(
            name,
            client_env=self._client_env,
            runtime_env={**self._extra_env, **provisioned.env},
            on_cleanup=provisioned.teardown,
        )


def read_grading_spec(task_dir: Path) -> dict[str, Any]:
    return json.loads((Path(task_dir) / "tests" / "grading.json").read_text())
