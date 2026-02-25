"""模拟交易所框架层（状态机）。"""

from __future__ import annotations

from typing import Dict, List

from ...core.data_structure import (
    Action,
    ActionType,
    CancelRequest,
    Order,
    OrderReceipt,
    OrderStatus,
    StepOutcome,
)
from ...core.port import ISimulator, IMarketDataFeed
from .match_algorithm import IMatchAlgorithm


class Simulator_Impl(ISimulator):
    """执行框架：维护时钟与活跃订单，调用撮合算法。"""

    def __init__(self, match_algo: IMatchAlgorithm) -> None:
        self._match_algo = match_algo
        self._current_time = 0
        self._interval_start = 0
        self._interval_end = 0
        self._active_orders: Dict[str, Order] = {}

    def set_market_data_feed(self, market_data_feed: IMarketDataFeed) -> None:
        self._match_algo.set_market_data_feed(market_data_feed)

    def start_session(self) -> None:
        self._match_algo.start_session()
        self._current_time = 0
        self._interval_start = 0
        self._interval_end = 0
        self._active_orders.clear()

    def set_time_window(self, t_start: int, t_end: int) -> None:
        self._interval_start = int(t_start)
        self._interval_end = int(t_end)
        self._current_time = int(t_start)
        self._match_algo.prepare_context(self._interval_start, self._interval_end)

    def on_action(self, action: Action) -> List[OrderReceipt]:
        t_arrive = int(action.create_time) if int(action.create_time) > 0 else int(self._current_time)
        if t_arrive < self._current_time:
            t_arrive = int(self._current_time)
        self._current_time = t_arrive

        if action.action_type == ActionType.PLACE_ORDER:
            return self._on_order_action(action.payload, t_arrive)
        if action.action_type == ActionType.CANCEL_ORDER:
            return self._on_cancel_action(action.payload, t_arrive)
        raise ValueError(f"Unsupported action type: {action.action_type!r}")

    def step(self, until_time: int) -> StepOutcome:
        t_cur = int(self._current_time)
        t_limit = int(until_time)
        if t_limit <= t_cur:
            return StepOutcome(next_time=t_cur, receipts_generated=[])

        outcome = self._match_algo.on_step(self._active_orders, t_cur, t_limit)
        next_time = self._clamp_time(int(outcome.next_time), t_cur, t_limit)
        receipts = list(outcome.receipts_generated or [])
        self._apply_receipts(receipts)
        self._current_time = next_time
        return StepOutcome(next_time=next_time, receipts_generated=receipts)

    def flush_window(self) -> object:
        return self._match_algo.flush_window()

    def _on_order_action(self, order: Order, t_arrive: int) -> List[OrderReceipt]:
        internal_order = self._clone_order(order, t_arrive)
        self._active_orders[internal_order.order_id] = internal_order
        receipts = self._match_algo.on_order_action_impl(internal_order, t_arrive, self._active_orders) or []
        self._apply_receipts(receipts)
        return list(receipts)

    def _on_cancel_action(self, request: CancelRequest, t_arrive: int) -> List[OrderReceipt]:
        receipts = self._match_algo.on_cancel_action_impl(request, t_arrive, self._active_orders) or []
        self._apply_receipts(receipts)
        return list(receipts)

    def _apply_receipts(self, receipts: List[OrderReceipt]) -> None:
        for receipt in receipts:
            order = self._active_orders.get(receipt.order_id)
            if order is None:
                continue

            if receipt.receipt_type == "PARTIAL":
                fill_qty = max(0, int(receipt.fill_qty))
                if fill_qty > 0:
                    order.filled_qty = min(order.qty, order.filled_qty + fill_qty)
                if order.remaining_qty <= 0 or int(receipt.remaining_qty) <= 0:
                    order.status = OrderStatus.FILLED
                    self._active_orders.pop(order.order_id, None)
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
            elif receipt.receipt_type == "FILL":
                fill_qty = max(0, int(receipt.fill_qty))
                if fill_qty > 0:
                    order.filled_qty = min(order.qty, order.filled_qty + fill_qty)
                else:
                    order.filled_qty = order.qty
                order.status = OrderStatus.FILLED
                self._active_orders.pop(order.order_id, None)
            elif receipt.receipt_type == "CANCELED":
                if int(receipt.fill_qty) > order.filled_qty:
                    order.filled_qty = min(order.qty, int(receipt.fill_qty))
                order.status = OrderStatus.CANCELED
                self._active_orders.pop(order.order_id, None)
            elif receipt.receipt_type == "REJECTED":
                # 取消被拒绝不改变活跃订单状态；是否移除由后续回执决定。
                continue

    @staticmethod
    def _clone_order(order: Order, t_arrive: int) -> Order:
        return Order(
            order_id=order.order_id,
            side=order.side,
            price=order.price,
            qty=order.qty,
            type=order.type,
            tif=order.tif,
            filled_qty=order.filled_qty,
            status=order.status,
            create_time=order.create_time,
            arrival_time=t_arrive,
        )

    @staticmethod
    def _clamp_time(t: int, t_cur: int, t_max: int) -> int:
        out = t if t >= t_cur else t_cur
        if out > t_max:
            return t_max
        return out
