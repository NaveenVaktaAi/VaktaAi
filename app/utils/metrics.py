import time
from collections import Counter


_metrics = Counter()


def incr(key: str, n: int = 1):
    _metrics[key] += n


def snapshot():
    return dict(_metrics)