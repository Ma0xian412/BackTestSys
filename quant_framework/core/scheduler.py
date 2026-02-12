"""基于堆的事件调度器。"""

from __future__ import annotations

import heapq
from typing import List

from .runtime import Event


class HeapScheduler:
    """最小堆调度器，按 (time, priority, seq) 稳定排序。"""

    def __init__(self) -> None:
        self._heap: List[Event] = []

    def clear(self) -> None:
        self._heap.clear()

    def push(self, e: Event) -> None:
        heapq.heappush(self._heap, e)

    def pushAll(self, es: List[Event]) -> None:
        for event in es:
            heapq.heappush(self._heap, event)

    def nextDueTimeOrDefault(self, defaultTime: int) -> int:
        if not self._heap:
            return defaultTime
        return self._heap[0].time

    def popAllAtTime(self, t: int) -> List[Event]:
        out: List[Event] = []
        while self._heap and self._heap[0].time == t:
            out.append(heapq.heappop(self._heap))
        return out

    def isEmpty(self) -> bool:
        return not self._heap
