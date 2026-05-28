import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Any


DEFAULT_DATASETS = [
    "Cli",
    "Codec",
    "Compress",
    "Csv",
    "Gson",
    "JacksonCore",
    "Jsoup",
    "Lang",
    "Math",
    "Mockito",
    "Time",
]


def python_exe() -> str:
    return sys.executable or "python"


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_one(dataset: str, idx: int, total: int, log_dir: str) -> Dict[str, Any]:
    pkl_path = f"{dataset}.pkl"
    if not os.path.exists(pkl_path):
        msg = f"[{idx}/{total}] {dataset} skipped (missing {pkl_path})"
        print(msg)
        return {"dataset": dataset, "status": "skipped", "rc": None, "seconds": 0.0}

    log_path = os.path.join(log_dir, f"{dataset}.log")
    start = time.time()
    print(f"[{idx}/{total}] {dataset} start (GCNN baseline)")
    sys.stdout.flush()

    env = os.environ.copy()
    env["USE_GCNN_BASELINE"] = "1"

    with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(f"dataset={dataset}\nvariant=gcnn_baseline\nstart={datetime.now().isoformat()}\n")
        f.flush()
        proc = subprocess.Popen(
            [python_exe(), "runtotal.py", dataset],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        rc = proc.wait()
        f.write(f"\nrc={rc}\nend={datetime.now().isoformat()}\n")

    seconds = time.time() - start
    print(f"[{idx}/{total}] {dataset} done rc={rc} ({seconds:.1f}s)")
    sys.stdout.flush()
    return {
        "dataset": dataset,
        "status": "ok" if rc == 0 else "failed",
        "rc": rc,
        "seconds": seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GCNN baseline (code-only) over datasets and collect logs."
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to run (default: 11 preset)",
    )
    parser.add_argument(
        "--out-dir",
        default="s3_gcnn",
        help="Base directory to save logs (default: s3_gcnn)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag for log folder name; if omitted uses timestamp",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets: List[str] = args.datasets
    stamp = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.out_dir
    ensure_dir(base_dir)
    log_dir = os.path.join(base_dir, stamp)
    ensure_dir(log_dir)

    summary_path = os.path.join(log_dir, "summary.txt")
    results = []
    total = len(datasets)
    for i, name in enumerate(datasets, 1):
        res = run_one(name, i, total, log_dir)
        results.append(res)

    with open(summary_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write("dataset,status,rc,seconds\n")
        for r in results:
            f.write(f"{r['dataset']},{r['status']},{r['rc']},{r['seconds']:.2f}\n")

    print("All done. Logs in:", log_dir)


if __name__ == "__main__":
    main()
