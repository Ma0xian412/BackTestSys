"""模拟交易所框架层（状态机与订单簿生命周期管理）。"""

from __future__ import annotations

from typing import Dict

from ...core.data_structure import (
    Action,
    ActionType,
    CancelRequest,
    Order,
    OrderReceipt,
    Result,
    ShadowOrder,
)
from ...core.port import IMatchAlgorithm, IMarketDataFeed, ISimulator


class Simulator_Impl(ISimulator):
    """执行框架：管理 Active Orders 与时间游标。"""

    def __init__(self, match_algo: IMatchAlgorithm) -> None:
        self._match_algo = match_algo
        self._current_time = 0
        self._interval_start = 0
        self._interval_end = 0
        self._active_orders: Dict[str, ShadowOrder] = {}

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

    def on_action(self, action: Action) -> Result:
        t_arrive = int(action.create_time) if int(action.create_time) > 0 else int(self._current_time)
        if t_arrive < self._current_time:
            t_arrive = int(self._current_time)
        self._current_time = t_arrive

        if action.action_type in (ActionType.ORDER_NEW, ActionType.PLACE_ORDER):
            return self._on_order_action(action, t_arrive)
        if action.action_type in (ActionType.ORDER_CANCEL, ActionType.CANCEL_ORDER):
            return self._on_cancel_action(action, t_arrive)
        raise ValueError(f"Unsupported action type: {action.action_type!r}")

    def step(self, until_time: int) -> Result:
        t_cur = int(self._current_time)
        t_limit = int(until_time)
        if t_limit <= t_cur:
            return self._none_result(timestamp=t_cur)

        result = self._match_algo.on_step(self._active_orders, t_cur, t_limit)
        if not result.receipts:
            result = self._none_result(timestamp=t_limit)

        self._apply_result_to_shadow_orders(result)
        self._advance_time_from_result(result, floor=t_cur, ceiling=t_limit)
        return result

    def flush_window(self) -> object:
        return self._match_algo.flush_window()

    def _on_order_action(self, action: Action, t_arrive: int) -> Result:
        order = self._extract_order_from_action(action)
        shadow = ShadowOrder(
            create_time=t_arrive,
            order_id=order.order_id,
            side=order.side,
            price=float(order.price),
            pos=0,
            init_vol=int(order.qty),
            now_vol=int(order.qty),
        )

        # 按时序图约束：先入 Active_Orders，再交给算法。
        self._active_orders[shadow.order_id] = shadow

        result = self._match_algo.on_order_action_impl(shadow, t_arrive)
        if not result.receipts:
            result = self._none_result(timestamp=t_arrive, order_id=shadow.order_id, pos=shadow.pos)

        if result.pos > 0 and shadow.order_id in self._active_orders:
            self._active_orders[shadow.order_id].pos = int(result.pos)

        self._apply_result_to_shadow_orders(result)
        self._advance_time_from_result(result, floor=t_arrive)
        return result

    def _on_cancel_action(self, action: Action, t_arrive: int) -> Result:
        request = self._extract_cancel_from_action(action)
        shadow = self._active_orders.pop(request.order_id, None)
        if shadow is None:
            receipt = OrderReceipt(
                order_id=request.order_id,
                receipt_type="REJECTED",
                timestamp=t_arrive,
                fill_qty=0,
                fill_price=0.0,
                remaining_qty=0,
            )
            result = Result(consumed_vol=0, pos=0, receipts=[receipt])
            self._advance_time_from_result(result, floor=t_arrive)
            return result

        traded = max(0, int(shadow.init_vol - shadow.now_vol))
        receipt = OrderReceipt(
            order_id=shadow.order_id,
            receipt_type="CANCELED",
            timestamp=t_arrive,
            fill_qty=traded,
            fill_price=float(shadow.price),
            remaining_qty=0,
        )
        result = Result(consumed_vol=0, pos=int(shadow.pos), receipts=[receipt])
        self._advance_time_from_result(result, floor=t_arrive)
        return result

    def _apply_result_to_shadow_orders(self, result: Result) -> None:
        for receipt in result.receipts:
            if receipt.receipt_type == "NONE":
                continue

            shadow = self._active_orders.get(receipt.order_id)
            if shadow is None:
                continue

            if receipt.receipt_type == "PARTIAL":
                remain = int(receipt.remaining_qty)
                if remain <= 0:
                    self._active_orders.pop(receipt.order_id, None)
                else:
                    shadow.now_vol = remain
            elif receipt.receipt_type in ("FILLED", "FILL"):
                shadow.now_vol = 0
                self._active_orders.pop(receipt.order_id, None)
            elif receipt.receipt_type == "CANCELED":
                shadow.now_vol = 0
                self._active_orders.pop(receipt.order_id, None)
            elif receipt.receipt_type == "REJECTED":
                # REJECTED 不改变 active orders。
                continue

    def _advance_time_from_result(self, result: Result, floor: int, ceiling: int | None = None) -> None:
        t_next = result.first_receipt_time()
        if t_next is None:
            t_next = int(floor)
        if t_next < floor:
            t_next = int(floor)
        if ceiling is not None and t_next > ceiling:
            t_next = int(ceiling)
        self._current_time = int(t_next)

    @staticmethod
    def _extract_order_from_action(action: Action) -> Order:
        payload = action.payload
        if isinstance(payload, Order):
            return payload
        raise TypeError(f"ORDER_NEW payload must be Order, got {type(payload)!r}")

    @staticmethod
    def _extract_cancel_from_action(action: Action) -> CancelRequest:
        payload = action.payload
        if isinstance(payload, CancelRequest):
            return payload
        if isinstance(payload, str):
            return CancelRequest(order_id=payload, create_time=int(action.create_time))
        raise TypeError(f"ORDER_CANCEL payload must be CancelRequest or str, got {type(payload)!r}")

    @staticmethod
    def _none_result(timestamp: int, order_id: str = "", pos: int = 0) -> Result:
        return Result(
            consumed_vol=0,
            pos=int(pos),
            receipts=[
                OrderReceipt(
                    order_id=order_id,
                    receipt_type="NONE",
                    timestamp=int(timestamp),
                    fill_qty=0,
                    fill_price=0.0,
                    remaining_qty=0,
                )
            ],
        )
