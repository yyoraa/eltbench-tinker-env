from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExpectedTable:
    name: str
    columns: tuple[tuple[str, str], ...]
    rows: tuple[tuple[object, ...], ...]
    order_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class ELTTask:
    """Immutable episode specification.

    Hidden expectations live outside the agent workspace. Only ``instruction``
    and source files are exposed during a rollout.
    """

    task_id: str
    instruction: str
    source_dir: Path
    expected_tables: tuple[ExpectedTable, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "expected_tables": [
                {
                    "name": table.name,
                    "columns": table.columns,
                    "rows": table.rows,
                    "order_by": table.order_by,
                }
                for table in self.expected_tables
            ],
            "metadata": self.metadata,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

