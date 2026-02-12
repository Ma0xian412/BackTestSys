"""执行场所端口适配器实现。"""

from __future__ import annotations

from typing import List, Optional

from ...core.data_structure import Action, ActionType, CancelRequest, NormalizedSnapshot, Order, OrderReceipt, StepOutcome
from ...core.port import IExecutionVenue, IIntervalModel
from .simulator import FIFOExchangeSimulator


class ExecutionVenue_Impl(IExecutionVenue):
    """将 FIFOExchangeSimulator 适配为 IExecutionVenue。"""

    def __init__(self, simulator: FIFOExchangeSimulator, tape_builder: IIntervalModel) -> None:
        self._simulator = simulator
        self._tape_builder = tape_builder
        self._tape = []
        self._seg_idx = 0
        self._prev_snapshot: Optional[NormalizedSnapshot] = None
        self._interval_start = 0
        self._interval_end = 0

    def startSession(self) -> None:
        self._simulator.full_reset()
        self._tape = []
        self._seg_idx = 0
        self._prev_snapshot = None
        self._interval_start = 0
        self._interval_end = 0

    def beginInterval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        self._prev_snapshot = prev
        self._interval_start = int(prev.ts_recv)
        self._interval_end = int(curr.ts_recv)
        self._tape = self._tape_builder.build(prev, curr)
        self._seg_idx = 0
        self._simulator.reset()
        self._simulator.set_tape(self._tape, self._interval_start, self._interval_end)

    def onActionArrival(self, action: Action, t_arrive: int) -> List[OrderReceipt]:
        if action.action_type == ActionType.PLACE_ORDER:
            return self.execute_place_order(action.payload, t_arrive)
        if action.action_type == ActionType.CANCEL_ORDER:
            return self.execute_cancel_order(action.payload, t_arrive)
        raise ValueError(f"Unsupported action type: {action.action_type!r}")

    def step(self, t_cur: int, t_limit: int) -> StepOutcome:
        if t_limit <= t_cur:
            return StepOutcome(next_time=t_cur, receipts_generated=[])
        if not self._tape:
            return StepOutcome(next_time=t_limit, receipts_generated=[])
        seg_idx = self._find_segment_idx(t_cur)
        if seg_idx >= len(self._tape):
            return StepOutcome(next_time=t_limit, receipts_generated=[])
        seg = self._tape[seg_idx]
        seg_limit = min(int(t_limit), int(seg.t_end))
        receipts, t_stop = self._simulator.advance(int(t_cur), seg_limit, seg)
        t_stop = int(t_stop)
        if t_stop < t_cur:
            t_stop = int(t_cur)
        if t_stop > t_limit:
            t_stop = int(t_limit)
        return StepOutcome(next_time=t_stop, receipts_generated=receipts or [])

    def endInterval(self, snapshot_end: NormalizedSnapshot) -> object:
        self._simulator.align_at_boundary(snapshot_end)
        return {
            "interval_start": self._interval_start,
            "interval_end": self._interval_end,
            "segment_count": len(self._tape),
        }

    def execute_place_order(self, order: Order, t_arrive: int) -> List[OrderReceipt]:
        market_qty = self._market_qty_at_price(order)
        receipt = self._simulator.on_order_arrival(order, t_arrive, market_qty)
        return [receipt] if receipt else []

    def execute_cancel_order(self, request: CancelRequest, t_arrive: int) -> List[OrderReceipt]:
        try:
            receipt = self._simulator.on_cancel_arrival(request.order_id, t_arrive)
        except ValueError:
            receipt = OrderReceipt(
                order_id=request.order_id,
                receipt_type="REJECTED",
                timestamp=t_arrive,
            )
        return [receipt]

    def _find_segment_idx(self, t: int) -> int:
        while self._seg_idx < len(self._tape) and int(t) >= int(self._tape[self._seg_idx].t_end):
            self._seg_idx += 1
        return self._seg_idx

    def _market_qty_at_price(self, order: Order) -> int:
        snapshot = self._prev_snapshot
        if snapshot is None:
            return 0
        levels = snapshot.bids if order.side.value == "BUY" else snapshot.asks
        target_price = float(order.price)
        for level in levels:
            if abs(float(level.price) - target_price) < 1e-12:
                return int(level.qty)
        return 0
