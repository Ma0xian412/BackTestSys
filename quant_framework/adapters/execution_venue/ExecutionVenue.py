"""执行场所端口适配器实现。"""

from __future__ import annotations

from typing import Dict, List, Optional

from ...core.data_structure import (
    Action,
    ActionType,
    CancelRequest,
    NormalizedSnapshot,
    Order,
    OrderReceipt,
    OrderStatus,
)
from ...core.port import IExecutionVenue, IIntervalModel, IMarketDataFeed, IMatchAlgorithm
from .simulator import FIFOExchangeSimulator


def _none_receipt(timestamp: int, order_id: str = "") -> OrderReceipt:
    return OrderReceipt(order_id=order_id, receipt_type="NONE", timestamp=int(timestamp))


class SegmentBaseAlgorithm(IMatchAlgorithm):
    """基于 Segment/Tape + FIFOExchangeSimulator 的撮合算法实现。"""

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
        self._current_time = 0
        self._window_start_data: Optional[NormalizedSnapshot] = None
        self._window_end_data: Optional[NormalizedSnapshot] = None
        self._pending_receipts: List[OrderReceipt] = []
        self._session_initialized = False

    def set_market_data_feed(self, market_data_feed: IMarketDataFeed) -> None:
        self._market_data_feed = market_data_feed

    def prepare_context(self, t_start: int, t_end: int) -> None:
        t_start = int(t_start)
        t_end = int(t_end)
        prev_window_end = self._window_end_data
        self._seg_idx = 0
        self._tape = []
        self._pending_receipts.clear()
        self._window_start_data = None
        self._window_end_data = None

        if self._market_data_feed is None:
            raise RuntimeError("SegmentBaseAlgorithm requires a market_data_feed.")

        # 新会话（或时间回拨）时完全重置；同会话跨窗口时仅做边界对齐。
        if (not self._session_initialized) or (t_start < self._current_time):
            self._simulator.full_reset()
            self._session_initialized = True
        else:
            # 先对齐到新区间左边界，随后 reset 清空区间态缓存。
            if prev_window_end is not None:
                self._simulator.align_at_boundary(prev_window_end)
            self._simulator.reset()

        self._current_time = t_start
        if t_end <= t_start:
            return

        window_data = self._market_data_feed.query_data(t_start, t_end)
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
            self._simulator.set_tape(self._tape, t_start, t_end)

    def on_order_action_impl(self, action: Action, current_time: int) -> OrderReceipt:
        t_arrive = int(current_time)
        if t_arrive > self._current_time:
            self._current_time = t_arrive

        if action.action_type == ActionType.PLACE_ORDER:
            order: Order = action.payload
            market_qty = self._market_qty_at_price(order)
            receipt = self._simulator.on_order_arrival(order, t_arrive, market_qty)
            return receipt if receipt is not None else _none_receipt(t_arrive, order.order_id)

        if action.action_type == ActionType.CANCEL_ORDER:
            request: CancelRequest = action.payload
            try:
                return self._simulator.on_cancel_arrival(request.order_id, t_arrive)
            except ValueError:
                return OrderReceipt(
                    order_id=request.order_id,
                    receipt_type="REJECTED",
                    timestamp=t_arrive,
                )

        raise ValueError(f"Unsupported action type: {action.action_type!r}")

    def on_step(self, until_time: int) -> OrderReceipt:
        t_limit = int(until_time)

        if self._pending_receipts:
            receipt = self._pending_receipts.pop(0)
            self._current_time = max(self._current_time, int(receipt.timestamp))
            return receipt

        if t_limit <= self._current_time:
            return _none_receipt(self._current_time)

        while self._current_time < t_limit:
            if not self._tape:
                self._current_time = t_limit
                return _none_receipt(t_limit)

            seg_idx = self._find_segment_idx(self._current_time)
            if seg_idx >= len(self._tape):
                self._current_time = t_limit
                return _none_receipt(t_limit)

            seg = self._tape[seg_idx]
            seg_limit = min(t_limit, int(seg.t_end))
            t_from = int(self._current_time)
            receipts, t_stop = self._simulator.advance(t_from, seg_limit, seg)
            t_stop = max(t_from, min(int(t_stop), seg_limit))
            self._current_time = t_stop

            if receipts:
                receipts_sorted = sorted(receipts, key=lambda r: int(r.timestamp))
                if len(receipts_sorted) > 1:
                    self._pending_receipts.extend(receipts_sorted[1:])
                return receipts_sorted[0]

            if t_stop <= t_from:
                # 防御性保护：避免在异常情况下死循环。
                self._current_time = t_limit
                return _none_receipt(t_limit)

        return _none_receipt(t_limit)

    def _find_segment_idx(self, t: int) -> int:
        while self._seg_idx < len(self._tape) and int(t) >= int(self._tape[self._seg_idx].t_end):
            self._seg_idx += 1
        return self._seg_idx

    def _market_qty_at_price(self, order: Order) -> int:
        md_start = self._window_start_data
        if md_start is None:
            return 0
        levels = md_start.bids if order.side.value == "BUY" else md_start.asks
        target_price = float(order.price)
        for level in levels:
            if abs(float(level.price) - target_price) < 1e-12:
                return int(level.qty)
        return 0


class ExecutionVenue_Impl(IExecutionVenue):
    """模拟交易所框架：维护时钟与活跃订单，调用撮合算法。"""

    def __init__(self, match_algorithm: IMatchAlgorithm) -> None:
        self._match_algo = match_algorithm
        self._current_time = 0
        self._active_orders: Dict[str, Order] = {}

    def set_time_window(self, t_start: int, t_end: int) -> None:
        self._current_time = int(t_start)
        self._match_algo.prepare_context(int(t_start), int(t_end))

    def on_action(self, action: Action) -> OrderReceipt:
        t_arrive = int(action.create_time) if int(action.create_time) > 0 else int(self._current_time)
        if action.action_type == ActionType.PLACE_ORDER:
            self._on_order_action(action.payload, t_arrive)
        elif action.action_type == ActionType.CANCEL_ORDER:
            self._on_cancel_action(action.payload)
        else:
            raise ValueError(f"Unsupported action type: {action.action_type!r}")

        receipt = self._match_algo.on_order_action_impl(action, t_arrive)
        receipt.timestamp = max(int(receipt.timestamp), int(t_arrive))
        self._current_time = int(receipt.timestamp)
        self._apply_receipt(receipt)
        return receipt

    def step(self, until_time: int) -> OrderReceipt:
        t_limit = max(int(until_time), int(self._current_time))
        receipt = self._match_algo.on_step(t_limit)
        if receipt is None:
            receipt = _none_receipt(t_limit)

        if receipt.receipt_type == "NONE":
            receipt.timestamp = t_limit
        else:
            receipt.timestamp = max(int(self._current_time), min(int(receipt.timestamp), t_limit))

        self._current_time = int(receipt.timestamp)
        self._apply_receipt(receipt)
        return receipt

    def _on_order_action(self, order: Order, t_arrive: int) -> None:
        order.create_time = int(t_arrive)
        self._active_orders[order.order_id] = order

    def _on_cancel_action(self, request: CancelRequest) -> None:
        _ = request

    def _apply_receipt(self, receipt: OrderReceipt) -> None:
        if receipt.receipt_type == "NONE":
            return

        order = self._active_orders.get(receipt.order_id)
        if order is None:
            return

        if receipt.receipt_type == "PARTIAL":
            order.filled_qty = min(order.qty, order.filled_qty + max(0, int(receipt.fill_qty)))
            if order.filled_qty >= order.qty:
                order.status = OrderStatus.FILLED
                self._active_orders.pop(order.order_id, None)
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
        elif receipt.receipt_type == "FILL":
            order.filled_qty = min(order.qty, order.filled_qty + max(0, int(receipt.fill_qty)))
            order.status = OrderStatus.FILLED
            self._active_orders.pop(order.order_id, None)
        elif receipt.receipt_type == "CANCELED":
            order.status = OrderStatus.CANCELED
            self._active_orders.pop(order.order_id, None)
        elif receipt.receipt_type == "REJECTED":
            order.status = OrderStatus.REJECTED
            self._active_orders.pop(order.order_id, None)
