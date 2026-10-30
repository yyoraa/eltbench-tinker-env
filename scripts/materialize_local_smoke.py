#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from eltbench_tinker.harbor_task import materialize_local_harbor_task
from eltbench_tinker.task import ELTTask, ExpectedTable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/tasks"))
    args = parser.parse_args()
    source_dir = Path(__file__).parents[1] / "tests" / "fixtures" / "customer_revenue"
    task = ELTTask(
        task_id="customer-revenue",
        instruction=(
            "Load customers.csv and orders.csv and create customer_revenue "
            "with customer_id BIGINT, name VARCHAR, revenue DECIMAL(38,2)."
        ),
        source_dir=source_dir,
        expected_tables=(
            ExpectedTable(
                name="customer_revenue",
                columns=(
                    ("customer_id", "BIGINT"),
                    ("name", "VARCHAR"),
                    ("revenue", "DECIMAL(38,2)"),
                ),
                rows=((1, "Ada", 20.00), (2, "Linus", 9.00), (3, "Grace", 0.00)),
                order_by=("customer_id",),
            ),
        ),
    )
    print(materialize_local_harbor_task(task, args.output))


if __name__ == "__main__":
    main()
