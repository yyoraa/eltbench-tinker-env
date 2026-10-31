#!/usr/bin/env python3
"""Airbyte extract/load, usable from the runner and as ``elt-airbyte`` in a sandbox.

Standard library only so the same file is copied into task images. Every
connection targets one rollout namespace and selects only the task's streams.
The benchmark's own sources are reused; only connections are created.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# "incomplete" is an attempt that Airbyte is about to retry, not an end state.
TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled"}


class AirbyteClient:
    def __init__(self, url: str | None = None, username: str | None = None, password: str | None = None):
        self.url = (url or os.environ.get("AIRBYTE_URL") or "http://localhost:8000").rstrip("/")
        user = username or os.environ.get("AIRBYTE_USERNAME") or "airbyte"
        secret = password or os.environ.get("AIRBYTE_PASSWORD") or "password"
        token = base64.b64encode(f"{user}:{secret}".encode()).decode()
        self._headers = {"Content-Type": "application/json", "Authorization": f"Basic {token}"}

    def post(self, path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.url}/api/v1/{path}", json.dumps(payload).encode(), self._headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read()[:500].decode(errors="replace")
            raise RuntimeError(f"Airbyte {path} -> HTTP {error.code}: {detail}") from None
        return json.loads(body) if body else {}

    def workspace_id(self) -> str:
        workspace = os.environ.get("AIRBYTE_WORKSPACE_ID")
        return workspace or self.post("workspaces/list", {})["workspaces"][0]["workspaceId"]

    def sources(self) -> list[dict[str, Any]]:
        return self.post("sources/list", {"workspaceId": self.workspace_id()})["sources"]

    def destinations(self) -> list[dict[str, Any]]:
        return self.post("destinations/list", {"workspaceId": self.workspace_id()})["destinations"]

    def connections(self) -> list[dict[str, Any]]:
        return self.post("connections/list", {"workspaceId": self.workspace_id()})["connections"]

    def source_id(self, name_or_id: str) -> str:
        for source in self.sources():
            if name_or_id in (source["name"], source["sourceId"]):
                return source["sourceId"]
        raise RuntimeError(f"no Airbyte source named {name_or_id!r}")

    def destination_id(self, name_or_id: str | None = None) -> str:
        wanted = name_or_id or os.environ.get("AIRBYTE_DESTINATION_ID")
        for destination in self.destinations():
            if wanted in (None, destination["name"], destination["destinationId"]):
                return destination["destinationId"]
        raise RuntimeError(f"no Airbyte destination {wanted!r}")

    def discover(self, source_id: str) -> dict[str, Any]:
        return self.post("sources/discover_schema", {"sourceId": source_id}, timeout=600)["catalog"]

    def create_connection(
        self,
        source: str,
        streams: list[str],
        namespace: str,
        *,
        destination: str | None = None,
        type_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a manual connection writing only ``streams`` into ``namespace``.

        ``type_overrides`` maps ``stream.field`` to a JSON type. Airbyte infers
        types by sampling, and a wrong guess nulls every non-conforming value
        (see docs/evidence.md for the case that exists in books).
        """
        source_id = self.source_id(source)
        catalog = self.discover(source_id)
        selected = []
        available = {entry["stream"]["name"]: entry for entry in catalog["streams"]}
        missing = sorted(set(streams) - set(available))
        if missing:
            raise RuntimeError(f"source {source!r} has no streams {missing}")
        for name in streams:
            entry = available[name]
            entry["config"].update(
                {
                    "selected": True,
                    "syncMode": "full_refresh",
                    "destinationSyncMode": "overwrite",
                }
            )
            properties = entry["stream"]["jsonSchema"].get("properties", {})
            for target, json_type in (type_overrides or {}).items():
                stream_name, _, field = target.partition(".")
                if stream_name == name:
                    properties[field] = {"type": json_type}
            selected.append(entry)
        return self.post(
            "connections/create",
            {
                "name": f"{namespace} {source}",
                "sourceId": source_id,
                "destinationId": self.destination_id(destination),
                "syncCatalog": {"streams": selected},
                "status": "active",
                "scheduleType": "manual",
                "namespaceDefinition": "customformat",
                "namespaceFormat": namespace,
                "nonBreakingChangesPreference": "ignore",
            },
        )

    def sync(self, connection_id: str) -> int:
        return int(self.post("connections/sync", {"connectionId": connection_id})["job"]["id"])

    def job(self, job_id: int) -> dict[str, Any]:
        return self.post("jobs/get", {"id": job_id}, timeout=60)["job"]

    def wait(self, job_id: int, timeout: int = 1800, poll: float = 5.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.job(job_id)["status"]
            if status in TERMINAL_JOB_STATES:
                return status
            time.sleep(poll)
        return "timeout"

    def delete_connection(self, connection_id: str) -> None:
        self.post("connections/delete", {"connectionId": connection_id})

    def connections_for_namespace(self, namespace: str) -> list[str]:
        return [
            item["connectionId"]
            for item in self.connections()
            if item.get("namespaceFormat") == namespace
        ]


def task_streams(config: dict[str, Any]) -> dict[str, list[str]]:
    """Map Airbyte source names to the streams an official config.yaml requires.

    Source names follow this deployment's convention for the books database (see
    docs/deployment.md). The custom declarative source serves all tasks, hence
    its explicit stream selection.
    """
    database = config["postgres"]["config"]["database"] if "postgres" in config else "books"
    plan: dict[str, list[str]] = {}
    if "postgres" in config:
        plan[f"{database}-postgres"] = list(config["postgres"]["config"]["tables"])
    if "mongodb" in config:
        plan[f"{database}-mongodb"] = list(config["mongodb"]["config"]["tables"])
    if "aws_s3" in config:
        plan[f"{database}-s3"] = [item["table"] for item in config["aws_s3"]["data"]]
    for item in config.get("flat_files", []):
        table = item["table"]
        plan[f"{database}-file-{table.replace('_', '-')}"] = [table]
    if "custom_api" in config:
        plan["ELT Bench"] = list(config["custom_api"]["config"]["tables"])
    return plan


def ingest(
    client: AirbyteClient,
    plan: dict[str, list[str]],
    namespace: str,
    *,
    destination: str | None = None,
    type_overrides: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Create one connection per source, sync them concurrently, and wait."""
    started = time.perf_counter()

    def run(item: tuple[str, list[str]]) -> dict[str, Any]:
        source, streams = item
        began = time.perf_counter()
        connection = client.create_connection(
            source, streams, namespace, destination=destination, type_overrides=type_overrides
        )
        connected = time.perf_counter()
        job_id = client.sync(connection["connectionId"])
        status = client.wait(job_id, timeout=timeout)
        return {
            "source": source,
            "streams": streams,
            "connection_id": connection["connectionId"],
            "job_id": job_id,
            "status": status,
            "connect_seconds": round(connected - began, 2),
            "sync_seconds": round(time.perf_counter() - connected, 2),
        }

    with ThreadPoolExecutor(max_workers=max(1, len(plan))) as pool:
        results = list(pool.map(run, plan.items()))
    return {
        "namespace": namespace,
        "ok": all(item["status"] == "succeeded" for item in results),
        "seconds": round(time.perf_counter() - started, 2),
        "connections": results,
    }


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=1, default=str))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="elt-airbyte", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sources", help="list sources available in this workspace")
    commands.add_parser("connections", help="list connections for ELT_TARGET_SCHEMA")
    streams = commands.add_parser("streams", help="list a source's streams and JSON types")
    streams.add_argument("source")
    streams.add_argument("--only", help="comma-separated stream names to show columns for")
    connect = commands.add_parser("connect", help="create a connection into ELT_TARGET_SCHEMA")
    connect.add_argument("source")
    connect.add_argument("--streams", required=True, help="comma-separated")
    connect.add_argument(
        "--type", action="append", default=[], metavar="STREAM.FIELD=TYPE",
        help="override an inferred JSON type, e.g. orders.note=string",
    )
    sync = commands.add_parser("sync", help="start a connection's sync; prints the job id")
    sync.add_argument("connection_id")
    status = commands.add_parser("status", help="report a job's state")
    status.add_argument("job_id", type=int)
    wait = commands.add_parser("wait", help="block (bounded) until a job ends")
    wait.add_argument("job_id", type=int)
    wait.add_argument("--timeout", type=int, default=100)
    args = parser.parse_args(argv)

    client = AirbyteClient()
    namespace = os.environ.get("ELT_TARGET_SCHEMA", "")
    if args.command == "sources":
        _emit([{"name": s["name"], "connector": s["sourceName"]} for s in client.sources()])
    elif args.command == "connections":
        _emit(
            [
                {"connection_id": c["connectionId"], "name": c["name"]}
                for c in client.connections()
                if c.get("namespaceFormat") == namespace
            ]
        )
    elif args.command == "streams":
        catalog = client.discover(client.source_id(args.source))
        only = set(args.only.split(",")) if args.only else None
        names = [entry["stream"]["name"] for entry in catalog["streams"]]
        if only is None:
            _emit(names)
        else:
            _emit(
                {
                    entry["stream"]["name"]: {
                        field: spec.get("type") or spec.get("airbyte_type")
                        for field, spec in entry["stream"]["jsonSchema"].get("properties", {}).items()
                    }
                    for entry in catalog["streams"]
                    if entry["stream"]["name"] in only
                }
            )
    elif args.command == "connect":
        if not namespace:
            sys.exit("ELT_TARGET_SCHEMA is not set")
        overrides = dict(item.split("=", 1) for item in args.type)
        created = client.create_connection(
            args.source, args.streams.split(","), namespace, type_overrides=overrides
        )
        _emit({"connection_id": created["connectionId"], "namespace": namespace})
    elif args.command == "sync":
        _emit({"job_id": client.sync(args.connection_id)})
    elif args.command == "status":
        job = client.job(args.job_id)
        _emit({"status": job["status"], "stats": job.get("aggregatedStats", {})})
    elif args.command == "wait":
        _emit({"status": client.wait(args.job_id, timeout=args.timeout)})


if __name__ == "__main__":
    main()
