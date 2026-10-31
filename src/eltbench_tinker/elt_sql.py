#!/usr/bin/env python3
"""``elt-sql``: run SQL inside the rollout schema from a sandbox. Prints CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path


def connect():
    import snowflake.connector

    kwargs = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "database": os.environ["ELT_TARGET_DATABASE"],
        "schema": os.environ["ELT_TARGET_SCHEMA"],
        "login_timeout": 60,
        "network_timeout": 120,
        "session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": 110, "QUERY_TAG": "eltbench-agent"},
    }
    if os.environ.get("SNOWFLAKE_ROLE"):
        kwargs["role"] = os.environ["SNOWFLAKE_ROLE"]
    if os.environ.get("SNOWFLAKE_WAREHOUSE"):
        kwargs["warehouse"] = os.environ["SNOWFLAKE_WAREHOUSE"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return snowflake.connector.connect(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(prog="elt-sql", description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-e", dest="sql", help="SQL text")
    group.add_argument("-f", dest="path", type=Path, help="file of SQL statements")
    parser.add_argument("--max-rows", type=int, default=50)
    args = parser.parse_args()
    sql = args.sql if args.sql is not None else args.path.read_text()
    connection = connect()
    status = 0
    writer = csv.writer(sys.stdout)
    try:
        for cursor in connection.execute_string(sql, remove_comments=True):
            if cursor.description:
                writer.writerow([item[0] for item in cursor.description])
                for row in cursor.fetchmany(args.max_rows):
                    writer.writerow(row)
                if cursor.rowcount is not None and cursor.rowcount > args.max_rows:
                    print(f"... {cursor.rowcount} rows in total, first {args.max_rows} shown")
    except Exception as error:
        # Execution feedback for the model. Driver text carries the SQL error, not credentials.
        print(f"SQL error: {getattr(error, 'msg', None) or type(error).__name__}", file=sys.stderr)
        status = 1
    finally:
        connection.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
