import os
import subprocess
import shlex
import numpy as np
import time

_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)
            
def set_gpu(gpu):
    print("set gpu:", gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def get_free_gpu():
    gpu_info = subprocess.Popen(
        shlex.split("nvidia-smi -q -d Memory"), stdout=subprocess.PIPE, text=True
    )
    grep1 = subprocess.Popen(
        shlex.split("grep -A4 GPU"),
        stdin=gpu_info.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )
    grep2 = subprocess.Popen(
        shlex.split("grep Used"), stdin=grep1.stdout, stdout=subprocess.PIPE, text=True
    )
    output, error = grep2.communicate()
    memory_available = np.array(
        [int(x.split(":")[1].strip().split()[0]) for x in output.split("\n")[0:-1]]
    )
    return np.argsort(memory_available)



def time_str(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    if t >= 60:
        return "{:.1f}m".format(t / 60)
    return "{:.1f}s".format(t)


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v