"""订单管理模块。

本模块包含订单管理相关的实现：
- Portfolio: 投资组合，管理现金和持仓
- OrderManager: 订单管理器，实现IOrderManager接口
"""

from typing import Dict, List, Callable, Optional
from ..core.types import Order, Fill, OrderId, OrderStatus, OrderReceipt
from ..core.interfaces import IOrderManager


class Portfolio:
    """投资组合。

    管理现金余额和持仓数量。

    Attributes:
        cash: 现金余额
        position: 持仓数量
        realized_pnl: 已实现盈亏
    """

    def __init__(self, cash=100000.0):
        """初始化投资组合。

        Args:
            cash: 初始现金（默认100000.0）
        """
        self.cash = cash
        self.position = 0
        self.realized_pnl = 0.0

    def update(self, fill: Fill):
        """根据成交记录更新持仓（旧版接口）。

        Args:
            fill: 成交记录
        """
        cost = fill.price * fill.qty
        if fill.side.value == "BUY":
            self.cash -= cost
            self.position += fill.qty
        else:
            self.cash += cost
            self.position -= fill.qty

    def update_from_receipt(self, receipt: OrderReceipt, order: Order):
        """根据订单回执更新持仓（新版接口）。

        Args:
            receipt: 订单回执
            order: 对应的订单
        """
        if receipt.receipt_type in ["FILL", "PARTIAL"] and receipt.fill_qty > 0:
            cost = receipt.fill_price * receipt.fill_qty
            if order.side.value == "BUY":
                self.cash -= cost
                self.position += receipt.fill_qty
            else:
                self.cash += cost
                self.position -= receipt.fill_qty


class OrderManager(IOrderManager):
    """订单管理器（同时实现旧版和新版接口）。

    职责：
    - 管理订单的生命周期
    - 处理订单回执
    - 维护投资组合状态
    - 提供订单查询接口

    注意：策略应通过ReadOnlyOMSView访问OrderManager，
    以确保策略只能查询而不能直接操作订单状态。
    """

    def __init__(self, portfolio: Portfolio = None):
        """初始化订单管理器。

        Args:
            portfolio: 投资组合（可选，默认创建新的）
        """
        self.orders: Dict[OrderId, Order] = {}
        self.portfolio = portfolio or Portfolio()
        self.fill_cb: List[Callable] = []
        self.new_cb: List[Callable] = []
        self.receipt_cb: List[Callable] = []

    # ========================================================================
    # 旧版接口方法
    # ========================================================================

    def subscribe_fill(self, cb):
        """订阅成交事件。"""
        self.fill_cb.append(cb)

    def subscribe_new(self, cb):
        """订阅新订单事件。"""
        self.new_cb.append(cb)

    def subscribe_receipt(self, cb):
        """订阅回执事件。"""
        self.receipt_cb.append(cb)

    def register_orders(self, orders: List[Order]):
        """注册订单（旧版方法）。

        Args:
            orders: 订单列表
        """
        for o in orders:
            if o.order_id not in self.orders:
                self.orders[o.order_id] = o
                for cb in self.new_cb:
                    cb(o)

    def process_fills(self, fills: List[Fill]):
        """处理成交记录（旧版方法）。

        Args:
            fills: 成交记录列表
        """
        for f in fills:
            if f.order_id in self.orders:
                order = self.orders[f.order_id]
                order.filled_qty += f.qty
                if order.filled_qty >= order.qty:
                    order.status = OrderStatus.FILLED
                elif order.filled_qty > 0:
                    order.status = OrderStatus.PARTIALLY_FILLED

                self.portfolio.update(f)
                for cb in self.fill_cb:
                    cb(f)

    # ========================================================================
    # 新版IOrderManager接口方法
    # ========================================================================

    def submit(self, order: Order, submit_time: int) -> None:
        """提交新订单。

        Args:
            order: 要提交的订单
            submit_time: 提交时间
        """
        order.create_time = submit_time
        if order.order_id not in self.orders:
            self.orders[order.order_id] = order
            for cb in self.new_cb:
                cb(order)

    def on_receipt(self, receipt: OrderReceipt) -> None:
        """处理订单回执。

        Args:
            receipt: 订单回执
        """
        order = self.orders.get(receipt.order_id)
        if not order:
            return

        # 根据回执类型更新订单状态
        if receipt.receipt_type == "FILL":
            order.filled_qty = receipt.fill_qty
            order.status = OrderStatus.FILLED
        elif receipt.receipt_type == "PARTIAL":
            order.filled_qty += receipt.fill_qty
            order.status = OrderStatus.PARTIALLY_FILLED
        elif receipt.receipt_type == "CANCELED":
            order.status = OrderStatus.CANCELED
        elif receipt.receipt_type == "REJECTED":
            order.status = OrderStatus.REJECTED

        # 更新投资组合
        if order:
            self.portfolio.update_from_receipt(receipt, order)

        # 通知回调
        for cb in self.receipt_cb:
            cb(receipt)

    def get_active_orders(self) -> List[Order]:
        """获取所有活跃订单。

        Returns:
            活跃订单列表
        """
        return [o for o in self.orders.values() if o.is_active]

    def get_order(self, order_id: str) -> Optional[Order]:
        """根据ID获取订单。

        Args:
            order_id: 订单ID

        Returns:
            订单（如果存在），否则返回None
        """
        return self.orders.get(order_id)