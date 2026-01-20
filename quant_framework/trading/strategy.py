from typing import List
from ..core.interfaces import IStrategy, IStrategyNew
from ..core.types import Order, Side, NormalizedSnapshot, OrderReceipt
from ..market.book import BookView


class LadderTestStrategy(IStrategy):
    """Legacy strategy implementation."""
    def __init__(self, name="Ladder"):
        self.name = name
        self.count = 0

    def on_market_tick(self, book: BookView, oms) -> List[Order]:
        if not book.cur: return []
        self.count += 1
        if self.count % 50 != 0: return []
        
        # 简单示例：只挂单不撤单
        bid = book.get_best_price(Side.BUY)
        return [Order(f"{self.name}-{self.count}", Side.BUY, bid, 1)] if bid else []


class SimpleNewStrategy(IStrategyNew):
    """Simple strategy implementing the new IStrategyNew interface.
    
    This is an example implementation that:
    - Places a buy order when seeing a snapshot
    - Reacts to fills by placing another order
    """
    
    def __init__(self, name: str = "SimpleNew"):
        self.name = name
        self.order_count = 0
        self.last_fill_time = 0
    
    def on_snapshot(self, snapshot: NormalizedSnapshot, oms) -> List[Order]:
        """Called when a new snapshot arrives.
        
        Args:
            snapshot: The new market snapshot
            oms: Order manager for querying order state
            
        Returns:
            List of new orders to submit
        """
        # Simple logic: place an order every 10 snapshots
        self.order_count += 1
        
        if self.order_count % 10 != 0:
            return []
        
        # Get best bid from snapshot
        if not snapshot.bids:
            return []
        
        best_bid = max(level.price for level in snapshot.bids)
        
        # Place a small buy order at best bid
        order = Order(
            order_id=f"{self.name}-{self.order_count}",
            side=Side.BUY,
            price=best_bid,
            qty=1,
        )
        
        return [order]
    
    def on_receipt(self, receipt: OrderReceipt, snapshot: NormalizedSnapshot, oms) -> List[Order]:
        """Called when an order receipt is received.
        
        Args:
            receipt: The order receipt (fill, cancel, etc.)
            snapshot: Current market snapshot
            oms: Order manager for querying order state
            
        Returns:
            List of new orders to submit
        """
        # React to fills
        if receipt.receipt_type in ["FILL", "PARTIAL"]:
            self.last_fill_time = receipt.timestamp
            
            # Simple reaction: place another order when filled
            if not snapshot.asks:
                return []
            
            best_ask = min(level.price for level in snapshot.asks)
            
            # Place a sell order at best ask
            self.order_count += 1
            order = Order(
                order_id=f"{self.name}-fill-{self.order_count}",
                side=Side.SELL,
                price=best_ask,
                qty=receipt.fill_qty,
            )
            
            return [order]
        
        return []