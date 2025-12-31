from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


@dataclass
class Scheduler:
    """
    高并发扫描驱动：按标的并行执行任务。
    """
    max_workers: int = 8

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers)

    def submit(self, fn: Callable, *args, **kwargs):
        return self._pool.submit(fn, *args, **kwargs)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
