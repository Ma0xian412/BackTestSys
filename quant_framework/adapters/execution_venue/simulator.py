"""模拟交易所框架层（状态机与订单簿生命周期管理）。"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ...core.data_structure import (
    Action,
    ActionType,
    CancelRequest,
    Order,
    OrderReceipt,
    ShadowOrder,
)
from ...core.port import IMatchAlgorithm, IMarketDataQuery, IMarketDataStream, ISimulator


logger = logging.getLogger(__name__)


class Simulator_Impl(ISimulator):
    """执行框架：管理 Active Orders 与时间游标。"""

    def __init__(self, match_algo: IMatchAlgorithm) -> None:
        self._match_algo = match_algo
        self._current_time = 0
        self._interval_start = 0
        self._interval_end = 0
        self._active_orders: Dict[str, ShadowOrder] = {}
        self._market_data_stream: Optional[IMarketDataStream] = None

    def set_market_data_stream(self, market_data_stream: IMarketDataStream) -> None:
        self._market_data_stream = market_data_stream

    def set_market_data_query(self, market_data_query: IMarketDataQuery) -> None:
        self._match_algo.set_market_data_query(market_data_query)

    def start_run(self) -> None:
        self._match_algo.start_run()
        self._current_time = 0
        self._interval_start = 0
        self._interval_end = 0
        self._active_orders.clear()

    def start_session(self, t_start: int, t_end: int) -> None:
        self._interval_start = int(t_start)
        self._interval_end = int(t_end)
        self._current_time = int(t_start)
        self._match_algo.start_session(self._interval_start, self._interval_end)

    def on_action(self, action: Action) -> List[OrderReceipt]:
        raw_arrive = int(action.create_time) if int(action.create_time) > 0 else int(self._current_time)
        if raw_arrive != int(self._current_time):
            logger.warning(
                "Simulator.on_action time mismatch: current_time=%s action_time=%s action_type=%s",
                int(self._current_time),
                raw_arrive,
                action.action_type,
            )

        t_arrive = max(int(self._current_time), raw_arrive)
        self._current_time = t_arrive

        if action.action_type in (ActionType.ORDER_NEW, ActionType.PLACE_ORDER):
            return self._on_order_action(action, t_arrive)
        if action.action_type in (ActionType.ORDER_CANCEL, ActionType.CANCEL_ORDER):
            return self._on_cancel_action(action, t_arrive)
        raise ValueError(f"Unsupported action type: {action.action_type!r}")

    def step(self, until_time: int) -> List[OrderReceipt]:
        t_cur = int(self._current_time)
        t_limit = int(until_time)
        if t_limit <= t_cur:
            return [self._none_receipt(timestamp=t_cur)]

        receipts = list(self._match_algo.on_step(self._active_orders, t_cur, t_limit) or [])
        if not receipts:
            receipts = [self._none_receipt(timestamp=t_limit)]

        self._apply_receipts_to_shadow_orders(receipts)
        self._advance_time_from_receipts(receipts, floor=t_cur, ceiling=t_limit)
        return receipts

    def flush_window(self) -> object:
        return self._match_algo.flush_window()

    def _on_order_action(self, action: Action, t_arrive: int) -> List[OrderReceipt]:
        order = self._extract_order_from_action(action)
        if order.order_id in self._active_orders:
            return [
                OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="REJECTED",
                    timestamp=t_arrive,
                    fill_qty=0,
                    fill_price=float(order.price),
                    remaining_qty=0,
                    pos=int(self._active_orders[order.order_id].pos),
                )
            ]

        shadow = ShadowOrder(
            create_time=t_arrive,
            order_id=order.order_id,
            side=order.side,
            price=float(order.price),
            pos=0,
            init_vol=int(order.qty),
            now_vol=int(order.qty),
        )

        # 新语义：先计算到达结果，再决定是否入 Active Orders。
        receipts = list(self._match_algo.on_order_action_impl(shadow, t_arrive) or [])
        if not receipts:
            receipts = [self._none_receipt(timestamp=t_arrive, order_id=shadow.order_id, pos=0)]

        market_pos = self._extract_market_pos(receipts)
        final_remain = self._extract_remaining_qty(shadow, receipts)
        should_queue = final_remain > 0 and not self._is_terminal_receipt(receipts)
        queued_pos = 0

        if should_queue:
            shadow.now_vol = int(final_remain)
            shadow.pos = self._allocate_order_pos(shadow, market_pos=market_pos)
            queued_pos = int(shadow.pos)
            self._active_orders[shadow.order_id] = shadow

        self._rewrite_order_pos(receipts, order_id=shadow.order_id, queued_pos=queued_pos, market_pos=market_pos)
        self._apply_receipts_to_shadow_orders(receipts)
        return receipts

    def _on_cancel_action(self, action: Action, t_arrive: int) -> List[OrderReceipt]:
        request = self._extract_cancel_from_action(action)
        shadow = self._active_orders.pop(request.order_id, None)
        if shadow is None:
            return [
                OrderReceipt(
                    order_id=request.order_id,
                    receipt_type="REJECTED",
                    timestamp=t_arrive,
                    fill_qty=0,
                    fill_price=0.0,
                    remaining_qty=0,
                    pos=0,
                )
            ]

        traded = max(0, int(shadow.init_vol - shadow.now_vol))
        return [
            OrderReceipt(
                order_id=shadow.order_id,
                receipt_type="CANCELED",
                timestamp=t_arrive,
                fill_qty=traded,
                fill_price=float(shadow.price),
                remaining_qty=0,
                pos=int(shadow.pos),
            )
        ]

    def _apply_receipts_to_shadow_orders(self, receipts: List[OrderReceipt]) -> None:
        for receipt in receipts:
            if receipt.receipt_type == "NONE":
                continue

            shadow = self._active_orders.get(receipt.order_id)
            if shadow is None:
                continue

            if receipt.pos >= 0:
                shadow.pos = int(receipt.pos)

            if receipt.receipt_type == "PARTIAL":
                remain = int(receipt.remaining_qty)
                if remain <= 0:
                    self._active_orders.pop(receipt.order_id, None)
                else:
                    shadow.now_vol = remain
            elif receipt.receipt_type in ("FILLED", "FILL", "CANCELED"):
                shadow.now_vol = 0
                self._active_orders.pop(receipt.order_id, None)
            elif receipt.receipt_type == "REJECTED":
                continue

    def _advance_time_from_receipts(self, receipts: List[OrderReceipt], floor: int, ceiling: int | None = None) -> None:
        if not receipts:
            t_next = int(floor)
        else:
            t_next = int(receipts[0].timestamp)
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
    def _none_receipt(timestamp: int, order_id: str = "", pos: int = 0) -> OrderReceipt:
        return OrderReceipt(
            order_id=order_id,
            receipt_type="NONE",
            timestamp=int(timestamp),
            fill_qty=0,
            fill_price=0.0,
            remaining_qty=0,
            pos=int(pos),
        )

    @staticmethod
    def _extract_market_pos(receipts: List[OrderReceipt]) -> int:
        for receipt in receipts:
            if receipt.receipt_type in ("NONE", "PARTIAL", "FILL", "FILLED"):
                return max(0, int(receipt.pos))
        return 0

    @staticmethod
    def _extract_remaining_qty(order: ShadowOrder, receipts: List[OrderReceipt]) -> int:
        remain = int(order.now_vol)
        for receipt in receipts:
            if receipt.order_id != order.order_id:
                continue
            if receipt.receipt_type in ("FILL", "FILLED", "CANCELED"):
                remain = 0
            elif receipt.receipt_type == "PARTIAL":
                remain = max(0, int(receipt.remaining_qty))
            elif receipt.receipt_type == "NONE":
                remain = max(0, int(order.now_vol))
        return max(0, int(remain))

    @staticmethod
    def _is_terminal_receipt(receipts: List[OrderReceipt]) -> bool:
        for receipt in receipts:
            if receipt.receipt_type in ("FILL", "FILLED", "CANCELED", "REJECTED"):
                return True
        return False

    @staticmethod
    def _rewrite_order_pos(
        receipts: List[OrderReceipt],
        order_id: str,
        queued_pos: int,
        market_pos: int,
    ) -> None:
        for receipt in receipts:
            if receipt.order_id != order_id:
                continue
            if receipt.receipt_type in ("NONE", "PARTIAL") and receipt.remaining_qty > 0:
                receipt.pos = int(queued_pos)
            elif receipt.pos <= 0:
                receipt.pos = int(market_pos)

    def _allocate_order_pos(self, order: ShadowOrder, market_pos: int) -> int:
        same_level_tail = 0
        for existing in self._active_orders.values():
            if existing.order_id == order.order_id:
                continue
            if existing.side != order.side:
                continue
            if abs(float(existing.price) - float(order.price)) > 1e-8:
                continue
            threshold = int(existing.pos) + max(0, int(existing.now_vol))
            if threshold > same_level_tail:
                same_level_tail = threshold
        return max(int(market_pos), same_level_tail)
