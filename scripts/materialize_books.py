#!/usr/bin/env python3
"""Materialize official snowflake/books tasks for Tinker's Harbor recipe.

Ground truth: `gt_snowflake/books` from the public tttjjj/elt_bench dataset (see
README). The checkout's ground_truth_legacy/ has duplicated rows and does not
agree with evaluation/table.json, so it is not used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eltbench_tinker.harbor_task import materialize_official_task
from eltbench_tinker.official import OfficialELTBenchTask

# Adds one publisher to the raw table in the replay clone. A pipeline that really
# transforms the source must then emit it; one that pastes rows cannot.
CANARY_ID = 987654321
CANARY_SQL = (
    f"INSERT INTO PUBLISHER SELECT * REPLACE ({CANARY_ID} AS PUBLISHER_ID, "
    "'canary publisher' AS PUBLISHER_NAME) FROM PUBLISHER LIMIT 1"
)
CANARY_CHECK_SQL = f"SELECT COUNT(*) FROM PUBLISHERS WHERE PUBLISHER_ID = {CANARY_ID}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elt-bench", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True, help="dir with <table>.csv")
    parser.add_argument("--output", type=Path, default=Path("build/tasks"))
    args = parser.parse_args()
    task = OfficialELTBenchTask.load(args.elt_bench, "snowflake", "books")
    for mode in ("end_to_end", "transform_only"):
        print(
            materialize_official_task(
                task,
                args.output,
                ground_truth_dir=args.ground_truth,
                table_json=args.elt_bench / "evaluation" / "table.json",
                sort_key_json=args.elt_bench / "evaluation" / "sort_key.json",
                mode=mode,
                canary_sql=CANARY_SQL,
                canary_check_sql=CANARY_CHECK_SQL,
            )
        )


if __name__ == "__main__":
    main()
