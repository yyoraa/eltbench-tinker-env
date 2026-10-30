#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from eltbench_tinker.harbor_task import materialize_official_public_task
from eltbench_tinker.official import OfficialELTBenchTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elt-bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/tasks"))
    args = parser.parse_args()
    task = OfficialELTBenchTask.load(args.elt_bench, "snowflake", "books")
    print(materialize_official_public_task(task, args.output))


if __name__ == "__main__":
    main()
