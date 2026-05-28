import subprocess
from tqdm import tqdm
import time
import os, sys
import re
import pickle
project = sys.argv[1]
pp = sys.argv[1]

# Check if the file exists, if yes, delete it
if os.path.exists(f'{pp}_timing_data.txt'):
    os.remove(f'{pp}_timing_data.txt')

card = [0]
lst = list(range(len(pickle.load(open(project + '.pkl', 'rb')))))
singlenums = {'Time':1, 'Math':2, "Lang":1, "Chart":3, "Mockito":4, "Closure":1, "Codec":1, 'Compress':1, 'Gson':1, 'Cli':1, 'Jsoup':1, 'Csv':1, 'JacksonCore':1, 'JacksonXml':1, 'Collections':1}
singlenum = singlenums[project]
totalnum = len(card) * singlenum
lr = 1e-2
seed = 0
batch_size = 40
edge_gate_l1 = 1e-5
edge_gate_warmup = 5
# No dataset-specific overrides; use tuned hyper-params across all datasets


def _python_exe():
    return sys.executable if sys.executable else "python"


def _spawn(script, args, env=None):
    cmd = [_python_exe(), script] + [str(a) for a in args]
    return subprocess.Popen(cmd, env=env)


for i in tqdm(range(int(len(lst) / totalnum) + 1)):
    jobs = []
    for j in range(totalnum):
        if totalnum * i + j >= len(lst):
            continue
        cardn =int(j / singlenum)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(card[cardn])
        args = [
            lst[totalnum * i + j],
            project,
            lr,
            seed,
            batch_size,
            "--edge-gate-l1",
            edge_gate_l1,
            "--edge-gate-warmup",
            edge_gate_warmup,
        ]
        p = _spawn("run.py", args, env=env)
        jobs.append(p)
        time.sleep(10)
    for p in jobs:
        p.wait()



p = _spawn("sum.py", [project, seed, lr, batch_size])
p.wait()
w = _spawn("watch.py", [project, seed, lr, batch_size])
w.wait()
# After all subprocesses are complete
training_times = []
testing_times = []

# Read the timing data from the file
with open(f'{pp}_timing_data.txt', 'r') as f:
    lines = f.readlines()

    for line in lines:
        match = re.search(r"TIMING_INFO: Training Time: (\d+.\d+), Testing Time: (\d+.\d+)", line)
        if match:
            training_times.append(float(match.group(1)))
            testing_times.append(float(match.group(2)))

# Calculate the total training and testing time
total_training_time = sum(training_times)
total_testing_time = sum(testing_times)

print(f"The overall training time is {total_training_time} seconds.")
print(f"The overall testing time is {total_testing_time} seconds.")
