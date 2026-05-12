"""
Procfs-based CPU sampler.

Reads /proc/<pid>/stat (and recursively for children) to compute total CPU
seconds consumed by a process tree across a window. Linux-only — that's
fine because the benchmark targets Docker / CI containers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Set

_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _read_stat(pid: int) -> tuple[int, int]:
    """Returns (utime_ticks, stime_ticks) for pid, or (0, 0) if gone."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            data = f.read()
    except (FileNotFoundError, ProcessLookupError):
        return 0, 0
    # The comm field can contain spaces & parens — find the last ')'.
    rparen = data.rfind(")")
    fields = data[rparen + 2 :].split()
    # Fields after comm: state(0) ppid(1) ... utime(11) stime(12)
    return int(fields[11]), int(fields[12])


def _children(pid: int) -> List[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r") as f:
            return [int(x) for x in f.read().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


def _walk(pid: int, seen: Set[int]) -> None:
    if pid in seen:
        return
    seen.add(pid)
    for child in _children(pid):
        _walk(child, seen)


def total_cpu_seconds(pid: int) -> float:
    """Sum utime + stime across the whole process tree rooted at pid."""
    seen: Set[int] = set()
    _walk(pid, seen)
    ticks = 0
    for p in seen:
        u, s = _read_stat(p)
        ticks += u + s
    return ticks / _CLK_TCK


@dataclass
class CpuWindow:
    pid: int
    start_seconds: float
    end_seconds: float
    wall_seconds: float

    @property
    def cpu_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def avg_cores(self) -> float:
        return self.cpu_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0


def sample_window(pid: int, wall_seconds: float) -> CpuWindow:
    """Block for `wall_seconds` and report CPU consumed by the pid's tree."""
    t0 = time.perf_counter()
    start = total_cpu_seconds(pid)
    time.sleep(wall_seconds)
    end = total_cpu_seconds(pid)
    elapsed = time.perf_counter() - t0
    return CpuWindow(
        pid=pid, start_seconds=start, end_seconds=end, wall_seconds=elapsed
    )
