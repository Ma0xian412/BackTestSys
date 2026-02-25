"""撮合算法层：窗口准备 + 动作下沉 + step 推进。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Mapping, Optional

from ...core.data_structure import (
    CancelRequest,
    NormalizedSnapshot,
    Order,
    OrderReceipt,
    StepOutcome,
    TapeSegment,
)
from ...core.port import IIntervalModel, IMarketDataFeed
from .fifo_exchange import FIFOExchangeSimulator


EPSILON = 1e-12


class IMatchAlgorithm(ABC):
    """撮合算法接口。"""

    @abstractmethod
    def set_market_data_feed(self, market_data_feed: IMarketDataFeed) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_session(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_context(self, t_start: int, t_end: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_order_action_impl(
        self,
        order: Order,
        t_arrive: int,
        active_orders: Mapping[str, Order],
    ) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def on_cancel_action_impl(
        self,
        request: CancelRequest,
        t_arrive: int,
        active_orders: Mapping[str, Order],
    ) -> List[OrderReceipt]:
        raise NotImplementedError

    @abstractmethod
    def on_step(
        self,
        active_orders: Mapping[str, Order],
        start_time: int,
        until_time: int,
    ) -> StepOutcome:
        raise NotImplementedError

    @abstractmethod
    def flush_window(self) -> object:
        raise NotImplementedError


class SegmentMatchAlgorithm(IMatchAlgorithm):
    """区间撮合算法编排器。"""

    def __init__(
        self,
        exchange_simulator: FIFOExchangeSimulator,
        tape_builder: IIntervalModel,
        market_data_feed: Optional[IMarketDataFeed] = None,
    ) -> None:
        self._simulator = exchange_simulator
        self._tape_builder = tape_builder
        self._market_data_feed = market_data_feed
        self._tape: List[TapeSegment] = []
        self._seg_idx = 0
        self._window_start_data: Optional[NormalizedSnapshot] = None
        self._window_end_data: Optional[NormalizedSnapshot] = None
        self._interval_start = 0
        self._interval_end = 0

    def set_market_data_feed(self, market_data_feed: IMarketDataFeed) -> None:
        self._market_data_feed = market_data_feed

    def start_session(self) -> None:
        self._simulator.full_reset()
        self._tape = []
        self._seg_idx = 0
        self._window_start_data = None
        self._window_end_data = None
        self._interval_start = 0
        self._interval_end = 0

    def prepare_context(self, t_start: int, t_end: int) -> None:
        self._interval_start = int(t_start)
        self._interval_end = int(t_end)
        self._seg_idx = 0
        self._tape = []
        self._window_start_data = None
        self._window_end_data = None

        self._simulator.reset()
        if self._interval_end <= self._interval_start:
            return

        if self._market_data_feed is None:
            raise RuntimeError("SegmentMatchAlgorithm requires market_data_feed for prepare_context().")

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

    def on_order_action_impl(
        self,
        order: Order,
        t_arrive: int,
        active_orders: Mapping[str, Order],
    ) -> List[OrderReceipt]:
        del active_orders
        market_qty = self._market_qty_at_price(order)
        receipt = self._simulator.on_order_arrival(order, t_arrive, market_qty)
        return [receipt] if receipt else []

    def on_cancel_action_impl(
        self,
        request: CancelRequest,
        t_arrive: int,
        active_orders: Mapping[str, Order],
    ) -> List[OrderReceipt]:
        del active_orders
        try:
            receipt = self._simulator.on_cancel_arrival(request.order_id, t_arrive)
        except ValueError:
            receipt = OrderReceipt(
                order_id=request.order_id,
                receipt_type="REJECTED",
                timestamp=t_arrive,
            )
        return [receipt]

    def on_step(
        self,
        active_orders: Mapping[str, Order],
        start_time: int,
        until_time: int,
    ) -> StepOutcome:
        del active_orders
        t_cur = int(start_time)
        t_limit = int(until_time)
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

    def flush_window(self) -> object:
        if self._window_end_data is not None:
            self._simulator.align_at_boundary(self._window_end_data)
        return {
            "interval_start": self._interval_start,
            "interval_end": self._interval_end,
            "segment_count": len(self._tape),
        }

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
            if abs(float(level.price) - target_price) < EPSILON:
                return int(level.qty)
        return 0
