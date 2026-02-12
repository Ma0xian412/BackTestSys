"""订单管理模块（新架构 IOMS 实现）。"""

from typing import Dict, List, Callable, Optional

from ..core.dto import ReadOnlyOMSView
from ..core.interfaces import IOMS
from ..core.types import CancelRequest, Fill, Order, OrderId, OrderReceipt, OrderStatus


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


class OrderManager(IOMS):
    """订单状态机 OMS。"""

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
        
    def subscribe_fill(self, cb):
        """订阅成交事件。"""
        self.fill_cb.append(cb)

    def subscribe_new(self, cb):
        """订阅新订单事件。"""
        self.new_cb.append(cb)

    def subscribe_receipt(self, cb):
        """订阅回执事件。"""
        self.receipt_cb.append(cb)

    def submit_action(self, action: object, send_time: int) -> None:
        """登记动作（仅 Order 会落库；CancelRequest 由回执驱动状态更新）。"""
        if isinstance(action, CancelRequest):
            return
        if not isinstance(action, Order):
            return

        action.create_time = int(send_time)
        if action.order_id not in self.orders:
            self.orders[action.order_id] = action
            for cb in self.new_cb:
                cb(action)

    def apply_receipt(self, receipt: OrderReceipt) -> None:
        """应用回执并推进订单状态。"""
        order = self.orders.get(receipt.order_id)
        if not order:
            return

        if receipt.receipt_type == "FILL":
            order.filled_qty = min(order.qty, order.filled_qty + max(0, int(receipt.fill_qty)))
            order.status = OrderStatus.FILLED if order.filled_qty >= order.qty else OrderStatus.PARTIALLY_FILLED
        elif receipt.receipt_type == "PARTIAL":
            order.filled_qty = min(order.qty, order.filled_qty + max(0, int(receipt.fill_qty)))
            order.status = OrderStatus.PARTIALLY_FILLED
        elif receipt.receipt_type == "CANCELED":
            order.status = OrderStatus.CANCELED
        elif receipt.receipt_type == "REJECTED":
            order.status = OrderStatus.REJECTED

        self.portfolio.update_from_receipt(receipt, order)

        for cb in self.receipt_cb:
            cb(receipt)

    def view(self) -> ReadOnlyOMSView:
        return ReadOnlyOMSView(self)

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