from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb


def grade(db_path: Path, expected_path: Path) -> float:
    expected = json.loads(expected_path.read_text())
    tables = expected["tables"]
    if not tables:
        return 0.0
    existence = schema = content = 0
    with duckdb.connect(str(db_path), read_only=True) as connection:
        for table in tables:
            name = table["name"]
            found = connection.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='main' AND table_name=?", [name]
            ).fetchone()[0]
            if not found:
                continue
            existence += 1
            actual_columns = [
                [str(row[1]), str(row[2]).upper()]
                for row in connection.execute(f"PRAGMA table_info('{name}')").fetchall()
            ]
            wanted_columns = [[c[0], c[1].upper()] for c in table["columns"]]
            if actual_columns != wanted_columns:
                continue
            schema += 1
            order = ", ".join(table["order_by"]) or "ALL"
            actual_rows = connection.execute(
                f'SELECT * FROM "{name}" ORDER BY {order}'
            ).fetchall()
            if [list(row) for row in actual_rows] == table["rows"]:
                content += 1
    total = len(tables)
    return round(0.1 * existence / total + 0.2 * schema / total + 0.7 * content / total, 8)


def main() -> None:
    reward = grade(Path(sys.argv[1]), Path(sys.argv[2]))
    output = Path(sys.argv[3])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{reward}\n")


if __name__ == "__main__":
    main()
