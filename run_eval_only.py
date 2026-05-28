import os
import sys
import subprocess

def main():
    if len(sys.argv) < 5:
        print("Usage: python run_eval_only.py <project> <seed> <lr> <batch_size>")
        sys.exit(1)
    project = sys.argv[1]
    seed = sys.argv[2]
    lr = sys.argv[3]
    batch_size = sys.argv[4]

    res_path = f"{project}res_{seed}_{lr}_{batch_size}.pkl"
    has_merged = os.path.exists(res_path)
    has_parts = any(
        name.startswith(f"{project}res") and name.endswith(f"_{seed}_{lr}_{batch_size}.pkl")
        for name in os.listdir(".")
    )
    if not has_merged and not has_parts:
        print(f"No result files found for {project} seed={seed} lr={lr} batch_size={batch_size}")
        sys.exit(1)

    subprocess.run([sys.executable, "sum.py", project, str(seed), str(lr), str(batch_size)], check=True)
    subprocess.run([sys.executable, "watch.py", project, str(seed), str(lr), str(batch_size)], check=True)

if __name__ == "__main__":
    main()
