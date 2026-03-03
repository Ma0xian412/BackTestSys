"""事件循环内核（Kernel）。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .data_structure import (
    EVENT_KIND_MDARRIVE,
    EVENT_KIND_RECEIPT_DELIVERY,
    Event,
    RuntimeContext,
    reset_event_seq,
)
from .run_control import DEFAULT_INTERRUPT_REASON, RunControl
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

    def run(self, ctx: RuntimeContext, run_control: Optional[RunControl] = None) -> Dict[str, Any]:
        reset_event_seq()
        self._t_cur = 0
        self._scheduler.clear()
        ctx.obs.on_run_started({})

        if self._should_stop(run_control):
            return self._finish_run(ctx, run_control=run_control, interrupted=True)

        ctx.feed.reset()
        ctx.venue.start_run()

        prev_data = ctx.feed.next()
        if prev_data is None:
            return self._finish_run(ctx, run_control=run_control, error="No data")

        first_t = self._extract_tick(prev_data)
        self._t_cur = first_t
        self._push_market_data_event(ctx, payload=prev_data, event_time=first_t)
        prev_time = first_t

        interrupted = False
        while True:
            if self._should_stop(run_control):
                interrupted = True
                break

            ctx.venue.start_session()
            curr_data = ctx.feed.next()
            if curr_data is None:
                break

            curr_time = self._extract_tick(curr_data)
            self._push_market_data_event(ctx, payload=curr_data, event_time=curr_time)

            interval_completed = self._run_interval(
                ctx,
                prev_time=prev_time,
                curr_time=curr_time,
                run_control=run_control,
            )
            if not interval_completed:
                interrupted = True
                break
            prev_time = curr_time

        return self._finish_run(ctx, run_control=run_control, interrupted=interrupted)

    def _run_interval(
        self,
        ctx: RuntimeContext,
        prev_time: int,
        curr_time: int,
        run_control: Optional[RunControl] = None,
    ) -> bool:
        t_a = int(prev_time)
        t_b = int(curr_time)
        if t_b <= t_a:
            return True

        t_cur = t_a
        while t_cur < t_b:
            self._drain_events_at_time(ctx, t_cur)
            if self._should_stop(run_control):
                self._t_cur = t_cur
                return False

            next_time = self._step_and_schedule_receipts(ctx, t_cur=t_cur, t_b=t_b)
            if next_time <= t_cur:
                break
            t_cur = next_time

        self._dispatch_boundary_events(ctx, t_b)
        self._close_interval(ctx, t_b)
        return True

    def _drain_events_at_time(self, ctx: RuntimeContext, t_cur: int) -> None:
        while True:
            events_at_t = self._scheduler.popAllAtTime(t_cur)
            if not events_at_t:
                return
            for event in events_at_t:
                emitted = ctx.dispatcher.dispatch(event, ctx) or []
                if emitted:
                    self._scheduler.pushAll(emitted)

    def _step_and_schedule_receipts(self, ctx: RuntimeContext, t_cur: int, t_b: int) -> int:
        t_ext = self._scheduler.nextDueTimeOrDefault(t_b)
        t_limit = min(t_ext, t_b)
        if t_limit <= t_cur:
            t_limit = t_b
            if t_limit <= t_cur:
                return t_cur

        receipts = list(ctx.venue.step(t_limit) or [])
        receipt_time = int(receipts[0].timestamp) if receipts else int(t_limit)
        next_time = self._clampTime(receipt_time, t_cur, t_limit)
        for receipt in receipts:
            if receipt.receipt_type == "NONE":
                continue
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
        return next_time

    def _dispatch_boundary_events(self, ctx: RuntimeContext, t_b: int) -> None:
        # 与旧语义保持一致：在区间边界 t_b 处理一批已到期事件，
        # 但不递归处理该批事件新产生的同刻事件。
        events_at_tb = self._scheduler.popAllAtTime(t_b)
        for event in events_at_tb:
            emitted = ctx.dispatcher.dispatch(event, ctx) or []
            if emitted:
                self._scheduler.pushAll(emitted)

    def _close_interval(self, ctx: RuntimeContext, t_b: int) -> None:
        interval_stats = ctx.venue.flush_window()
        ctx.obs.on_interval_end(interval_stats)
        self._t_cur = t_b

    def _finish_run(
        self,
        ctx: RuntimeContext,
        run_control: Optional[RunControl],
        interrupted: bool = False,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        status = "interrupted" if interrupted else "completed"
        interrupt_reason = self._interrupt_reason(run_control) if interrupted else None
        run_context: Dict[str, Any] = {
            "final_time": self._t_cur,
            "oms_view": ctx.oms.view(),
            "status": status,
            "interrupted": interrupted,
            "interrupt_reason": interrupt_reason,
        }
        if error is not None:
            run_context["error"] = error
        ctx.obs.on_run_end(run_context)
        return ctx.obs.get_run_result()

    @staticmethod
    def _should_stop(run_control: Optional[RunControl]) -> bool:
        return bool(run_control is not None and run_control.should_stop())

    @staticmethod
    def _interrupt_reason(run_control: Optional[RunControl]) -> str:
        if run_control is None:
            return DEFAULT_INTERRUPT_REASON
        return run_control.stop_reason or DEFAULT_INTERRUPT_REASON

    def _push_market_data_event(self, ctx: RuntimeContext, payload: object, event_time: int) -> None:
        self._scheduler.push(
            Event(
                time=event_time,
                kind=EVENT_KIND_MDARRIVE,
                priority=ctx.eventSpec.priorityOf(EVENT_KIND_MDARRIVE),
                payload=payload,
            )
        )

    @staticmethod
    def _extract_tick(data: object) -> int:
        return int(data.ts_recv)

    @staticmethod
    def _clampTime(t: int, t_cur: int, t_max: int | None = None) -> int:
        """保证时间单调不回退。"""
        out = t if t >= t_cur else t_cur
        if t_max is not None and out > t_max:
            return t_max
        return out
