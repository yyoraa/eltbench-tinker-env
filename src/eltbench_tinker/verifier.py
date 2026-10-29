from __future__ import annotations

from dataclasses import dataclass, field

from .destination import DuckDBDestination
from .task import ELTTask


@dataclass(frozen=True)
class VerificationResult:
    reward: float
    components: dict[str, float] = field(default_factory=dict)
    errors: tuple[str, ...] = ()


class ELTVerifier:
    """Grades final warehouse state; never trusts agent-reported success."""

    def __init__(self, task: ELTTask, destination: DuckDBDestination):
        self.task = task
        self.destination = destination

    def grade(self) -> VerificationResult:
        total = max(1, len(self.task.expected_tables))
        existence = schema = content = 0.0
        errors: list[str] = []

        for expected in self.task.expected_tables:
            if not self.destination.table_exists(expected.name):
                errors.append(f"missing table: {expected.name}")
                continue
            existence += 1 / total

            actual_columns = self.destination.columns(expected.name)
            normalized_expected = tuple((name, dtype.upper()) for name, dtype in expected.columns)
            if actual_columns != normalized_expected:
                errors.append(
                    f"schema mismatch for {expected.name}: "
                    f"expected={normalized_expected!r} actual={actual_columns!r}"
                )
                continue
            schema += 1 / total

            order = ", ".join(expected.order_by) if expected.order_by else "ALL"
            rows = tuple(self.destination.query(f'SELECT * FROM "{expected.name}" ORDER BY {order}'))
            if rows != expected.rows:
                errors.append(
                    f"content mismatch for {expected.name}: "
                    f"expected={expected.rows!r} actual={rows!r}"
                )
                continue
            content += 1 / total

        # Outcome dominates shaping: full content correctness is worth 0.7.
        components = {
            "table_existence": round(existence, 8),
            "schema": round(schema, 8),
            "content": round(content, 8),
        }
        reward = 0.1 * existence + 0.2 * schema + 0.7 * content
        return VerificationResult(round(reward, 8), components, tuple(errors))

