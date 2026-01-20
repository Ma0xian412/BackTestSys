"""策略模块。

本模块包含策略的实现：
- SimpleStrategy: 策略实现（使用IStrategy接口，通过DTO通信）
"""

from typing import List
from ..core.interfaces import IStrategy
from ..core.types import Order, Side, OrderReceipt
from ..core.dto import SnapshotDTO, ReadOnlyOMSView


class SimpleStrategy(IStrategy):
    """简单策略（实现IStrategy接口，通过DTO通信）。

    该策略使用DTO进行所有模块间通信：
    - SnapshotDTO: 不可变的行情快照
    - ReadOnlyOMSView: 只读的OMS视图

    示例策略功能：
    - 每10个快照下一个买单
    - 收到成交回执后下一个卖单
    - 可以查询当前活跃订单和持仓信息
    """

    def __init__(self, name: str = "Simple"):
        self.name = name
        self.order_count = 0
        self.last_fill_time = 0

    def on_snapshot(self, snapshot: SnapshotDTO, oms_view: ReadOnlyOMSView) -> List[Order]:
        """快照到达时回调（使用DTO）。

        Args:
            snapshot: 行情快照DTO（不可变）
            oms_view: OMS只读视图（只能查询）

        Returns:
            要提交的新订单列表
        """
        # 简单逻辑：每10个快照下一单
        self.order_count += 1

        if self.order_count % 10 != 0:
            return []

        # 使用DTO的便捷属性获取最优买价
        best_bid = snapshot.best_bid
        if best_bid is None:
            return []

        # 可以查询当前活跃订单数量（只读操作）
        active_orders = oms_view.get_active_orders()
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

        return [order]

    def on_receipt(self, receipt: OrderReceipt, snapshot: SnapshotDTO, oms_view: ReadOnlyOMSView) -> List[Order]:
        """订单回执到达时回调（使用DTO）。

        Args:
            receipt: 订单回执（成交、撤单等）
            snapshot: 当前行情快照DTO（不可变）
            oms_view: OMS只读视图（只能查询）

        Returns:
            要提交的新订单列表
        """
        # 响应成交
        if receipt.receipt_type in ["FILL", "PARTIAL"]:
            self.last_fill_time = receipt.timestamp

            # 使用DTO的便捷属性获取最优卖价
            best_ask = snapshot.best_ask
            if best_ask is None:
                return []

            # 可以查询持仓信息（只读操作）
            portfolio = oms_view.get_portfolio()
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

            return [order]

        return []