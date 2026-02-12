"""事件循环内核（Kernel）。"""

from __future__ import annotations

from typing import Any, Dict

from .data_structure import (
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    Event,
    NormalizedSnapshot,
    RuntimeContext,
    reset_event_seq,
)
from .scheduler import HeapScheduler


class EventLoopKernel:
    """核心事件循环。

    负责：
    - 驱动区间循环
    - 与执行场所进行 step 协作推进时间
    - 通过 Dispatcher 处理事件并调度新事件
    """

    def __init__(self, scheduler: HeapScheduler | None = None) -> None:
        self._scheduler = scheduler or HeapScheduler()
        self._t_cur = 0

    def run(self, ctx: RuntimeContext) -> Dict[str, Any]:
        reset_event_seq()

        ctx.feed.reset()
        ctx.venue.startSession()
        self._scheduler.clear()

        prev_snapshot = ctx.feed.next()
        if prev_snapshot is None:
            ctx.obs.on_run_end(final_time=0, error="No data")
            return ctx.obs.get_run_result()

        first_t = int(prev_snapshot.ts_recv)
        self._t_cur = first_t
        self._scheduler.push(
            Event(
                time=first_t,
                kind=EVENT_KIND_SNAPSHOT_ARRIVAL,
                priority=ctx.eventSpec.priorityOf(EVENT_KIND_SNAPSHOT_ARRIVAL),
                payload=prev_snapshot,
            )
        )

        while True:
            curr_snapshot = ctx.feed.next()
            if curr_snapshot is None:
                break

            self._runInterval(ctx, prev_snapshot, curr_snapshot)
            prev_snapshot = curr_snapshot

        ctx.obs.on_run_end(final_time=self._t_cur, error=None)
        return ctx.obs.get_run_result()

    def _runInterval(
        self,
        ctx: RuntimeContext,
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
    ) -> None:
        t_a = int(prev.ts_recv)
        t_b = int(curr.ts_recv)
        if t_b <= t_a:
            return

        ctx.venue.beginInterval(prev, curr)

        self._scheduler.push(
            Event(
                time=t_b,
                kind=EVENT_KIND_SNAPSHOT_ARRIVAL,
                priority=ctx.eventSpec.priorityOf(EVENT_KIND_SNAPSHOT_ARRIVAL),
                payload=curr,
            )
        )

        t_cur = t_a
        while t_cur < t_b:
            # 先耗尽当前刻度事件（包含同刻新增事件）
            while True:
                events_at_t = self._scheduler.popAllAtTime(t_cur)
                if not events_at_t:
                    break
                for event in events_at_t:
                    emitted = ctx.dispatcher.dispatch(event, ctx) or []
                    if emitted:
                        self._scheduler.pushAll(emitted)

            t_ext = self._scheduler.nextDueTimeOrDefault(t_b)
            t_limit = min(t_ext, t_b)
            if t_limit <= t_cur:
                t_limit = t_b
                if t_limit <= t_cur:
                    break

            outcome = ctx.venue.step(t_cur, t_limit)
            next_time = self._clampTime(int(outcome.next_time), t_cur, t_limit)

            for receipt in outcome.receipts_generated:
                ctx.obs.on_receipt_generated(receipt)
                t_deliver = ctx.timeModel.delayin(int(receipt.timestamp))
                self._scheduler.push(
                    Event(
                        time=self._clampTime(int(t_deliver), next_time, t_b),
                        kind=EVENT_KIND_RECEIPT_DELIVERY,
                        priority=ctx.eventSpec.priorityOf(EVENT_KIND_RECEIPT_DELIVERY),
                        payload=receipt,
                    )
                )

            t_cur = next_time

        # 与旧语义保持一致：在区间边界 t_b 处理一批已到期事件，
        # 但不递归处理该批事件新产生的同刻事件。
        events_at_tb = self._scheduler.popAllAtTime(t_b)
        for event in events_at_tb:
            emitted = ctx.dispatcher.dispatch(event, ctx) or []
            if emitted:
                self._scheduler.pushAll(emitted)

        interval_stats = ctx.venue.endInterval(curr)
        ctx.obs.on_interval_end(interval_stats)
        self._t_cur = t_b

    @staticmethod
    def _clampTime(t: int, t_cur: int, t_max: int | None = None) -> int:
        """保证时间单调不回退。"""
        out = t if t >= t_cur else t_cur
        if t_max is not None and out > t_max:
            return t_max
        return out
