import csv
import os
import pathlib
import re
import sys
from collections import defaultdict


def latest_summary():
    root = pathlib.Path("tune_logs")
    if not root.exists():
        raise SystemExit("tune_logs/ not found, run tune_math.py first.")
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Math_")]
    if not candidates:
        raise SystemExit("No Math_* folders in tune_logs/.")
    return max(candidates, key=lambda p: p.name) / "summary.csv"


def best_score(log_path: pathlib.Path):
    vals = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m1 = re.search(r"curr accuracy is\s+([0-9\.]+)", line)
        if m1:
            vals.append(float(m1.group(1)))
            continue
        m2 = re.search(r"find better score\s+([0-9\.]+)", line)
        if m2:
            vals.append(float(m2.group(1)))
    return sum(vals) / len(vals) if vals else None


def load_summary(path: pathlib.Path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["rc"] != "0":
                continue
            rows.append(row)
    return rows


def main():
    summary_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else latest_summary()
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found")
    rows = load_summary(summary_path)

    grouped = defaultdict(list)
    for r in rows:
        log_path = pathlib.Path(r["log_path"])
        if not log_path.exists():
            continue
        s = best_score(log_path)
        if s is None:
            continue
        key = (
            r["lr"],
            r["edge_gate_l1"],
            r["batch_size"],
            r["use_gcnn"],
        )
        grouped[key].append(s)

    scored = []
    for k, vals in grouped.items():
        scored.append((sum(vals) / len(vals), len(vals)) + k)

    scored.sort(key=lambda x: x[0])  # lower score better
    print(f"Using summary: {summary_path}")
    print("mean_score  n  lr  edge_gate_l1  batch  use_gcnn  (lower is better; score来自curr accuracy)")
    for row in scored[:10]:
        mean_s, n, lr, l1, bs, gcnn = row
        print(f"{mean_s:8.4f}  {n:2d}  {lr}  {l1}  {bs}  {gcnn}")


if __name__ == "__main__":
    main()
