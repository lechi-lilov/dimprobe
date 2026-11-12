"""Integer searches over monotone predicates. Solver-agnostic by design.

Monotonicity (e.g. "if d works, d+1 works") is an ASSUMPTION for optimizer-reachable
quantities, not a fact. Every evaluated point is recorded so violations can be inspected.
"""
from __future__ import annotations

from typing import Callable


class Recorder:
    def __init__(self, pred: Callable[[int], bool]):
        self.pred = pred
        self.calls: dict[int, bool] = {}

    def __call__(self, x: int) -> bool:
        if x not in self.calls:
            self.calls[x] = bool(self.pred(x))
        return self.calls[x]


def largest_true(pred: Callable[[int], bool], lo: int, hi: int) -> tuple[int | None, bool, dict]:
    """Largest x in [lo, hi] with pred(x) True, assuming True...True False...False.

    Returns (x, censored, calls). censored=True means pred(hi) is True (answer >= hi).
    x is None if pred(lo) is False.
    """
    r = Recorder(pred)
    if r(hi):
        return hi, True, r.calls
    if not r(lo):
        return None, False, r.calls
    a, b = lo, hi  # pred(a) True, pred(b) False
    while b - a > 1:
        mid = (a + b) // 2
        if r(mid):
            a = mid
        else:
            b = mid
    return a, False, r.calls


def smallest_true(pred: Callable[[int], bool], lo: int, hi: int) -> tuple[int | None, bool, dict]:
    """Smallest x in [lo, hi] with pred(x) True, assuming False...False True...True.

    Returns (x, censored, calls). censored=True means pred(lo) is True (answer <= lo).
    x is None if pred(hi) is False.
    """
    r = Recorder(pred)
    if r(lo):
        return lo, True, r.calls
    if not r(hi):
        return None, False, r.calls
    a, b = lo, hi  # pred(a) False, pred(b) True
    while b - a > 1:
        mid = (a + b) // 2
        if r(mid):
            b = mid
        else:
            a = mid
    return b, False, r.calls
