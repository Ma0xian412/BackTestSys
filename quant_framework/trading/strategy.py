"""策略模块（新架构 single-entry）。"""

from typing import List

from ..core.model import (
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    Action,
    ActionType,
    StrategyContext,
)
from ..core.port import IStrategy
from ..core.types import Order, Side


class SimpleStrategyImpl(IStrategy):
    """简单示例策略：每 10 个快照下买单，成交后尝试卖出。"""

    def __init__(self, name: str = "Simple"):
        self.name = name
        self.order_count = 0
        self.last_fill_time = 0

    def on_event(self, e, ctx: StrategyContext) -> List[Action]:
        if e.kind == EVENT_KIND_SNAPSHOT_ARRIVAL:
            return self._on_snapshot(e, ctx)
        if e.kind == EVENT_KIND_RECEIPT_DELIVERY:
            return self._on_receipt(e, ctx)
        return []

    def _on_snapshot(self, e, ctx: StrategyContext) -> List[Action]:
        # 简单逻辑：每10个快照下一单
        self.order_count += 1

        if self.order_count % 10 != 0:
            return []

        # 使用快照便捷属性获取最优买价
        snapshot = e.payload
        best_bid = snapshot.best_bid
        if best_bid is None:
            return []

        # 可以查询当前活跃订单数量（只读操作）
        active_orders = ctx.omsView.get_active_orders()
        # 如果活跃订单超过5个，暂停下单
        if len(active_orders) >= 5:
            return []

        # 在最优买价下一个小买单
        order = Order(
            order_id=f"{self.name}-{self.order_count}",
            side=Side.BUY,
            price=best_bid,
            qty=1,
        )
        return [Action(action_type=ActionType.PLACE_ORDER, create_time=0, payload=order)]

    def _on_receipt(self, e, ctx: StrategyContext) -> List[Action]:
        receipt = e.payload
        # 响应成交
        if receipt.receipt_type in ["FILL", "PARTIAL"]:
            self.last_fill_time = receipt.timestamp

            # 使用DTO的便捷属性获取最优卖价
            best_ask = ctx.snapshot.best_ask if ctx.snapshot is not None else None
            if best_ask is None:
                return []

            # 可以查询持仓信息（只读操作）
            portfolio = ctx.omsView.get_portfolio()
            # 如果没有持仓，不下卖单
            if portfolio.position <= 0:
                return []

            # 在最优卖价下一个卖单
            self.order_count += 1
            order = Order(
                order_id=f"{self.name}-fill-{self.order_count}",
                side=Side.SELL,
                price=best_ask,
                qty=receipt.fill_qty,
            )
            return [Action(action_type=ActionType.PLACE_ORDER, create_time=0, payload=order)]

        return []