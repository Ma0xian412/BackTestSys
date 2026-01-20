from typing import Dict, List, Callable, Optional
from ..core.types import Order, Fill, OrderId, OrderStatus, OrderReceipt
from ..core.interfaces import IOrderManager


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
    
    def update_from_receipt(self, receipt: OrderReceipt, order: Order):
        """Update portfolio from order receipt."""
        if receipt.receipt_type in ["FILL", "PARTIAL"] and receipt.fill_qty > 0:
            cost = receipt.fill_price * receipt.fill_qty
            if order.side.value == "BUY":
                self.cash -= cost
                self.position += receipt.fill_qty
            else:
                self.cash += cost
                self.position -= receipt.fill_qty


class OrderManager(IOrderManager):
    """Order manager implementing both legacy and new interfaces."""
    
    def __init__(self, portfolio: Portfolio = None):
        self.orders: Dict[OrderId, Order] = {}
        self.portfolio = portfolio or Portfolio()
        self.fill_cb: List[Callable] = []
        self.new_cb: List[Callable] = []
        self.receipt_cb: List[Callable] = []

    # Legacy interface methods
    def subscribe_fill(self, cb): 
        self.fill_cb.append(cb)
    
    def subscribe_new(self, cb): 
        self.new_cb.append(cb)
    
    def subscribe_receipt(self, cb):
        """Subscribe to receipt events."""
        self.receipt_cb.append(cb)

    def register_orders(self, orders: List[Order]):
        """Legacy method for registering orders."""
        for o in orders:
            if o.order_id not in self.orders:
                self.orders[o.order_id] = o
                for cb in self.new_cb: 
                    cb(o)

    def process_fills(self, fills: List[Fill]):
        """Legacy method for processing fills."""
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
    
    # New IOrderManager interface methods
    def submit(self, order: Order, submit_time: int) -> None:
        """Submit a new order.
        
        Args:
            order: The order to submit
            submit_time: Time of submission
        """
        order.create_time = submit_time
        if order.order_id not in self.orders:
            self.orders[order.order_id] = order
            for cb in self.new_cb:
                cb(order)
    
    def on_receipt(self, receipt: OrderReceipt) -> None:
        """Process an order receipt.
        
        Args:
            receipt: The order receipt to process
        """
        order = self.orders.get(receipt.order_id)
        if not order:
            return
        
        # Update order state based on receipt
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
        
        # Update portfolio
        if order:
            self.portfolio.update_from_receipt(receipt, order)
        
        # Notify callbacks
        for cb in self.receipt_cb:
            cb(receipt)
    
    def get_active_orders(self) -> List[Order]:
        """Get all currently active orders.
        
        Returns:
            List of active orders
        """
        return [o for o in self.orders.values() if o.is_active]
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID.
        
        Args:
            order_id: The order ID
            
        Returns:
            The order if found, None otherwise
        """
        return self.orders.get(order_id)