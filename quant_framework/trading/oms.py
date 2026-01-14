from typing import Dict, List, Callable
from ..core.types import Order, Fill, OrderId, OrderStatus

class Portfolio:
    def __init__(self, cash=100000.0):
        self.cash = cash
        self.position = 0
        self.realized_pnl = 0.0

    def update(self, fill: Fill):
        # 简化 PnL
        cost = fill.price * fill.qty
        if fill.side.value == "BUY":
            self.cash -= cost
            self.position += fill.qty
        else:
            self.cash += cost
            self.position -= fill.qty

class OrderManager:
    def __init__(self, portfolio: Portfolio):
        self.orders: Dict[OrderId, Order] = {}
        self.portfolio = portfolio
        self.fill_cb: List[Callable] = []
        self.new_cb: List[Callable] = []

    def subscribe_fill(self, cb): self.fill_cb.append(cb)
    def subscribe_new(self, cb): self.new_cb.append(cb)

    def register_orders(self, orders: List[Order]):
        for o in orders:
            if o.order_id not in self.orders:
                self.orders[o.order_id] = o
                for cb in self.new_cb: cb(o)

    def process_fills(self, fills: List[Fill]):
        for f in fills:
            if f.order_id in self.orders:
                self.portfolio.update(f)
                for cb in self.fill_cb: cb(f)