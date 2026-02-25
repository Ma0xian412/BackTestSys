"""执行场所端口适配器实现。"""

from __future__ import annotations

from typing import List, Optional

from ...core.data_structure import Action, ActionType, CancelRequest, NormalizedSnapshot, Order, OrderReceipt, StepOutcome
from ...core.port import IExecutionVenue, IIntervalModel, IMarketDataFeed
from .simulator import FIFOExchangeSimulator


class ExecutionVenue_Impl(IExecutionVenue):
    """将 FIFOExchangeSimulator 适配为 IExecutionVenue。"""

    def __init__(
        self,
        simulator: FIFOExchangeSimulator,
        tape_builder: IIntervalModel,
        market_data_feed: Optional[IMarketDataFeed] = None,
    ) -> None:
        self._simulator = simulator
        self._tape_builder = tape_builder
        self._market_data_feed = market_data_feed
        self._tape = []
        self._seg_idx = 0
        self._window_start_data: Optional[NormalizedSnapshot] = None
        self._window_end_data: Optional[NormalizedSnapshot] = None
        self._current_time = 0
        self._interval_start = 0
        self._interval_end = 0

    def bind_market_data_feed(self, market_data_feed: IMarketDataFeed) -> None:
        self._market_data_feed = market_data_feed

    def start_session(self) -> None:
        self._simulator.full_reset()
        self._tape = []
        self._seg_idx = 0
        self._window_start_data = None
        self._window_end_data = None
        self._current_time = 0
        self._interval_start = 0
        self._interval_end = 0

    def set_time_window(self, t_start: int, t_end: int) -> None:
        self._interval_start = int(t_start)
        self._interval_end = int(t_end)
        self._current_time = int(t_start)
        self._seg_idx = 0
        self._tape = []
        self._window_start_data = None
        self._window_end_data = None

        self._simulator.reset()
        if self._interval_end <= self._interval_start:
            return

        if self._market_data_feed is None:
            raise RuntimeError("ExecutionVenue_Impl requires a market_data_feed for set_time_window().")

        window_data = self._market_data_feed.query_data(self._interval_start, self._interval_end)
        if not window_data:
            return

        self._window_start_data = window_data[0]
        self._window_end_data = window_data[-1]
        if self._window_start_data is None or self._window_end_data is None:
            return

        if int(self._window_end_data.ts_recv) <= int(self._window_start_data.ts_recv):
            return

        self._tape = self._tape_builder.build(self._window_start_data, self._window_end_data)
        if self._tape:
            self._simulator.set_tape(self._tape, self._interval_start, self._interval_end)

    def on_action(self, action: Action) -> List[OrderReceipt]:
        t_arrive = int(action.create_time) if int(action.create_time) > 0 else int(self._current_time)
        if action.action_type == ActionType.PLACE_ORDER:
            return self.execute_place_order(action.payload, t_arrive)
        if action.action_type == ActionType.CANCEL_ORDER:
            return self.execute_cancel_order(action.payload, t_arrive)
        raise ValueError(f"Unsupported action type: {action.action_type!r}")

    def step(self, until_time: int) -> StepOutcome:
        t_cur = int(self._current_time)
        t_limit = int(until_time)
        if t_limit <= t_cur:
            return StepOutcome(next_time=t_cur, receipts_generated=[])
        if not self._tape:
            self._current_time = t_limit
            return StepOutcome(next_time=t_limit, receipts_generated=[])
        seg_idx = self._find_segment_idx(t_cur)
        if seg_idx >= len(self._tape):
            self._current_time = t_limit
            return StepOutcome(next_time=t_limit, receipts_generated=[])
        seg = self._tape[seg_idx]
        seg_limit = min(int(t_limit), int(seg.t_end))
        receipts, t_stop = self._simulator.advance(int(t_cur), seg_limit, seg)
        t_stop = int(t_stop)
        if t_stop < t_cur:
            t_stop = int(t_cur)
        if t_stop > t_limit:
            t_stop = int(t_limit)
        self._current_time = t_stop
        return StepOutcome(next_time=t_stop, receipts_generated=receipts or [])

    def flush_window(self) -> object:
        if self._window_end_data is not None:
            self._simulator.align_at_boundary(self._window_end_data)
        return {
            "interval_start": self._interval_start,
            "interval_end": self._interval_end,
            "segment_count": len(self._tape),
        }

    # --- backward compatibility helpers ---
    def startSession(self) -> None:
        self.start_session()

    def beginInterval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        self._window_start_data = prev
        self._window_end_data = curr
        self._interval_start = int(prev.ts_recv)
        self._interval_end = int(curr.ts_recv)
        self._current_time = self._interval_start
        self._tape = self._tape_builder.build(prev, curr)
        self._seg_idx = 0
        self._simulator.reset()
        if self._tape:
            self._simulator.set_tape(self._tape, self._interval_start, self._interval_end)

    def onActionArrival(self, action: Action, t_arrive: int) -> List[OrderReceipt]:
        action.create_time = int(t_arrive)
        return self.on_action(action)

    def endInterval(self, snapshot_end: NormalizedSnapshot) -> object:
        self._window_end_data = snapshot_end
        return self.flush_window()

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
        data_at_window_start = self._window_start_data
        if data_at_window_start is None:
            return 0
        levels = data_at_window_start.bids if order.side.value == "BUY" else data_at_window_start.asks
        target_price = float(order.price)
        for level in levels:
            if abs(float(level.price) - target_price) < 1e-12:
                return int(level.qty)
        return 0
