"""Check the current Python, official submission files, and optional built tables."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from preprocess import TABLES, PARQUET_TYPES, load_cohort

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_tables(directory):
    frames, report = {}, {}
    for name, spec in TABLES.items():
        path = directory / f"{name}.parquet"
        table = pq.read_table(path)
        expected = [column for column, _ in spec["columns"]]
        require(table.column_names == expected, f"{name}: unexpected columns")
        for column, kind in spec["columns"]:
            require(table.schema.field(column).type == PARQUET_TYPES[kind],
                    f"{name}.{column}: unexpected Arrow type")
        df = table.to_pandas()
        require(np.array_equal(df[spec["pk"]].to_numpy(), np.arange(len(df))),
                f"{name}: primary keys are not consecutive row numbers")
        times = df[spec["time"]]
        require(times.notna().all() and times.is_monotonic_increasing,
                f"{name}: missing or unsorted row times")
        frames[name] = df
        report[name] = {"rows": len(df), "row_time": spec["time"],
                        "min_time_utc": None if df.empty else str(times.min()),
                        "max_time_utc": None if df.empty else str(times.max())}
    for name, spec in TABLES.items():
        for column, target in spec["fks"].items():
            values = frames[name][column].dropna()
            require(values.isin(frames[target][TABLES[target]["pk"]]).all(),
                    f"{name}.{column}: invalid foreign keys to {target}")
    cohort = load_cohort(ROOT / "scripts/cohort.txt")
    users = frames["Users"]
    require(np.array_equal(users["github_user_id"].isin(cohort).to_numpy(),
                           users["is_cohort"].to_numpy()), "Users: inconsistent is_cohort")
    expected = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"]
    report["coverage"] = {
        "cohort_users_present": int(users["is_cohort"].sum()), "cohort_users_total": len(cohort),
        "test_users_present": int(expected.isin(users["github_user_id"]).sum()),
        "test_users_total": len(expected),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(f"Python: {sys.executable}\nVersion: {sys.version.split()[0]}", flush=True)
    for package in ("numpy", "pandas", "pyarrow"):
        print(f"{package}: {importlib.metadata.version(package)}", flush=True)
    cohort = load_cohort(ROOT / "scripts/cohort.txt")
    test = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"]
    require(len(cohort) == 9000, "Expected 9000 cohort users")
    require(test.is_unique and test.isin(cohort).all(), "Invalid test users or cohort")
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
                    "--pred", str(ROOT / "submission/sample_submission.csv"),
                    "--input", str(ROOT / "submission/test_input.csv")], check=True)
    if args.tables:
        report = check_tables(args.tables)
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS: all 8 tables, Arrow schemas, primary/foreign keys, row times, and cohort flags")
    else:
        print("PASS: environment and official submission files")


if __name__ == "__main__":
    main()
