"""策略模块。

本模块包含策略的实现：
- LadderTestStrategy: 旧版策略实现（使用IStrategy接口）
- SimpleNewStrategy: 新版策略实现（使用IStrategyNew接口）
- SimpleDTOStrategy: 使用DTO的策略实现（使用IStrategyDTO接口，完全解耦）
"""

from typing import List
from ..core.interfaces import IStrategy, IStrategyNew, IStrategyDTO
from ..core.types import Order, Side, NormalizedSnapshot, OrderReceipt
from ..core.dto import SnapshotDTO, ReadOnlyOMSView
from ..market.book import BookView


class LadderTestStrategy(IStrategy):
    """旧版阶梯测试策略。"""

    def __init__(self, name="Ladder"):
        self.name = name
        self.count = 0

    def on_market_tick(self, book: BookView, oms) -> List[Order]:
        """行情tick回调。"""
        if not book.cur:
            return []
        self.count += 1
        if self.count % 50 != 0:
            return []

        # 简单示例：只挂单不撤单
        bid = book.get_best_price(Side.BUY)
        return [Order(f"{self.name}-{self.count}", Side.BUY, bid, 1)] if bid else []


class SimpleNewStrategy(IStrategyNew):
    """新版简单策略（实现IStrategyNew接口）。

    示例策略功能：
    - 每10个快照下一个买单
    - 收到成交回执后下一个卖单
    """

    def __init__(self, name: str = "SimpleNew"):
        self.name = name
        self.order_count = 0
        self.last_fill_time = 0

    def on_snapshot(self, snapshot: NormalizedSnapshot, oms) -> List[Order]:
        """快照到达时回调。

        注意：虽然oms参数可以访问，但策略应只使用其查询方法，
        不应调用submit等修改方法。订单通过返回值提交。

        Args:
            snapshot: 新的行情快照
            oms: 订单管理器（仅用于查询）

        Returns:
            要提交的新订单列表
        """
        # 简单逻辑：每10个快照下一单
        self.order_count += 1

        if self.order_count % 10 != 0:
            return []

        # 从快照获取最优买价
        if not snapshot.bids:
            return []

        best_bid = max(level.price for level in snapshot.bids)

        # 在最优买价下一个小买单
        order = Order(
            order_id=f"{self.name}-{self.order_count}",
            side=Side.BUY,
            price=best_bid,
            qty=1,
        )

        return [order]

    def on_receipt(self, receipt: OrderReceipt, snapshot: NormalizedSnapshot, oms) -> List[Order]:
        """订单回执到达时回调。

        Args:
            receipt: 订单回执（成交、撤单等）
            snapshot: 当前行情快照
            oms: 订单管理器（仅用于查询）

        Returns:
            要提交的新订单列表
        """
        # 响应成交
        if receipt.receipt_type in ["FILL", "PARTIAL"]:
            self.last_fill_time = receipt.timestamp

            # 简单响应：成交后下一个卖单
            if not snapshot.asks:
                return []

            best_ask = min(level.price for level in snapshot.asks)

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


class SimpleDTOStrategy(IStrategyDTO):
    """使用DTO的简单策略（实现IStrategyDTO接口，完全解耦）。

    该策略使用只读视图，确保：
    - 无法修改快照数据（SnapshotDTO是不可变的）
    - 无法直接操作OMS（ReadOnlyOMSView只提供查询方法）
    - 所有订单通过返回值提交

    示例策略功能：
    - 每10个快照下一个买单
    - 收到成交回执后下一个卖单
    - 可以查询当前活跃订单和持仓信息
    """

    def __init__(self, name: str = "SimpleDTO"):
        self.name = name
        self.order_count = 0
        self.last_fill_time = 0

    def on_snapshot(self, snapshot: SnapshotDTO, oms_view: ReadOnlyOMSView) -> List[Order]:
        """快照到达时回调（使用只读视图）。

        Args:
            snapshot: 行情快照DTO（不可变，无法修改）
            oms_view: OMS只读视图（只能查询，无法操作）

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
        """订单回执到达时回调（使用只读视图）。

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