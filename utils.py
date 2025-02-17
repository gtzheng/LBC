import os
import subprocess
import shlex
import numpy as np
import time
import torch

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename="log.txt"):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            print(obj, file=f)


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


class BestMetric:
    def __init__(self):
        self.best_val = -1.0
        self.best_test = 0.0

    def add(self, val, test):
        if self.best_val < val:
            self.best_val = val
            self.best_test = test
            return 1
        else:
            return 0

    def get(self):
        return self.best_val, self.best_test


class BestMetricGroup(BestMetric):
    def __init__(self):
        super(BestMetricGroup, self).__init__()

    def str(self):
        test_avg_acc = self.best_test[0]
        test_unbiased_accs = self.best_test[1]
        test_worst_acc = self.best_test[2]

        val_acc = self.best_val
        res_str = f"val: {val_acc:.6f} test: avg_acc {test_avg_acc:.6f} worst_acc {test_worst_acc:.6f} test_unbiased {test_unbiased_accs:.6f} val_avg {self.best_test[3]}"
        return res_str


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
