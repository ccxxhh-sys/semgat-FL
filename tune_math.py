import itertools
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime

from tqdm import tqdm

# Hyperparameter grid
LRS = [1e-3, 5e-3, 1e-2]
EDGE_GATE_L1S = [1e-5, 1e-4, 5e-4]
BATCH_SIZES = [40, 60]
SEEDS = [0, 1]
EDGE_GATE_WARMUP = 5
USE_GCNN = [False, True]  # True means fallback baseline without GATv2

PROJECT = "Math"


def _python_exe():
    return sys.executable if sys.executable else "python"


def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def _load_test_ids():
    data = pickle.load(open(f"{PROJECT}.pkl", "rb"))
    return list(range(len(data)))


def main():
    test_ids = _load_test_ids()
    # To keep tuning成本可控，只跑前两个样本；需要全量可自行改为 test_ids
    test_ids = test_ids[:2]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("tune_logs", f"{PROJECT}_{stamp}")
    _ensure_dir(base_dir)
    summary_path = os.path.join(base_dir, "summary.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            "run_id,test_id,lr,edge_gate_l1,batch_size,seed,edge_gate_warmup,use_gcnn,rc,seconds,log_path\n"
        )

    combos = list(
        itertools.product(LRS, EDGE_GATE_L1S, BATCH_SIZES, SEEDS, USE_GCNN)
    )
    run_id = 0
    for lr, edge_l1, bs, seed, use_gcnn in tqdm(combos, desc="grid"):
        for tid in test_ids:
            run_id += 1
            log_path = os.path.join(base_dir, f"run_{run_id}_tid{tid}.log")
            args = [
                _python_exe(),
                "run.py",
                tid,
                PROJECT,
                lr,
                seed,
                bs,
                "--edge-gate-l1",
                edge_l1,
                "--edge-gate-warmup",
                EDGE_GATE_WARMUP,
            ]
            if use_gcnn:
                args.append("--use-gcnn-baseline")

            start = time.time()
            with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
                lf.write(f"cmd={' '.join(map(str, args))}\n")
                lf.flush()
                proc = subprocess.Popen(
                    list(map(str, args)),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                rc = proc.wait()
            seconds = time.time() - start

            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{run_id},{tid},{lr},{edge_l1},{bs},{seed},{EDGE_GATE_WARMUP},{int(use_gcnn)},{rc},{seconds:.1f},{log_path}\n"
                )

    print("Grid search finished. Summary:", summary_path)


if __name__ == "__main__":
    main()
