from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Any
from .types import Order, NormalizedSnapshot, Price, Qty, Side, TapeSegment, OrderReceipt
from .events import SimulationEvent

class IQueueModel(ABC):
    @abstractmethod
    def init_order(self, order: Order, level_qty: Qty) -> None: pass
    @abstractmethod
    def advance_on_trade(self, order: Order, trade_px: Price, trade_qty: Qty, before: Qty, after: Qty) -> Qty: pass

    # 可选：用于处理“无成交的盘口变化”（主要承载撤单/新增导致的排队位置变化）
    # 若实现未覆盖该方法，则默认不更新（保持与旧实现兼容）。
    def advance_on_quote(self, order: Order, before: Qty, after: Qty) -> None:
        return

class IMarketDataFeed(ABC):
    @abstractmethod
    def next(self) -> Optional[NormalizedSnapshot]: pass
    @abstractmethod
    def reset(self): pass # 支持多次回测重置

class ISimulationModel(ABC):
    """仿真模型接口：支持随机种子初始化"""
    @abstractmethod
    def generate_events(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot, context: Any = None) -> Iterator[SimulationEvent]: pass

class ITradeTapeReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, prev: Optional[NormalizedSnapshot], curr: NormalizedSnapshot) -> List[Tuple[Price, Qty]]: pass

class IStrategy(ABC):
    @abstractmethod
    def on_market_tick(self, book: Any, oms: Any) -> List[Order]: pass

# ============================================================================
# New unified architecture interfaces
# ============================================================================

class ITapeBuilder(ABC):
    """Build event tape from snapshot pairs (pure function, stateless)"""
    @abstractmethod
    def build(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> List[TapeSegment]:
        """Build tape segments from A/B snapshots.
        
        Args:
            prev: Previous snapshot (A)
            curr: Current snapshot (B)
            
        Returns:
            List of TapeSegments ordered by time
        """
        pass

class IExchangeSimulator(ABC):
    """Exchange simulator with FIFO queue management"""
    
    @abstractmethod
    def reset(self) -> None:
        """Reset simulator state for new interval"""
        pass
    
    @abstractmethod
    def on_order_arrival(self, order: Order, arrival_time: int, market_qty: Qty) -> Optional[OrderReceipt]:
        """Handle order arrival at exchange.
        
        Args:
            order: The arriving order
            arrival_time: Time of arrival at exchange
            market_qty: Current market queue depth at order price
            
        Returns:
            Optional receipt for immediate rejection (None if accepted)
        """
        pass
    
    @abstractmethod
    def on_cancel_arrival(self, order_id: str, arrival_time: int) -> OrderReceipt:
        """Handle cancel request arrival at exchange.
        
        Args:
            order_id: ID of order to cancel
            arrival_time: Time of cancel arrival
            
        Returns:
            Receipt for the cancel operation
        """
        pass
    
    @abstractmethod
    def advance(self, t_from: int, t_to: int, segment: TapeSegment) -> List[OrderReceipt]:
        """Advance simulation from t_from to t_to using tape segment.
        
        Args:
            t_from: Start time of slice
            t_to: End time of slice
            segment: Tape segment containing M and C for this interval
            
        Returns:
            List of receipts for fills during this period
        """
        pass
    
    @abstractmethod
    def align_at_boundary(self, snapshot: NormalizedSnapshot) -> None:
        """Align internal state at interval boundary.
        
        Args:
            snapshot: The snapshot at the boundary to align to
        """
        pass
    
    @abstractmethod
    def get_queue_depth(self, side: Side, price: Price) -> Qty:
        """Get current queue depth at a price level.
        
        Args:
            side: BUY or SELL
            price: Price level
            
        Returns:
            Queue depth at this level
        """
        pass

class IStrategyNew(ABC):
    """New strategy interface with mandatory receipt handling"""
    
    @abstractmethod
    def on_snapshot(self, snapshot: NormalizedSnapshot, oms: 'IOrderManager') -> List[Order]:
        """Called when a new snapshot arrives.
        
        Args:
            snapshot: The new market snapshot
            oms: Order manager for querying order state
            
        Returns:
            List of new orders to submit
        """
        pass
    
    @abstractmethod
    def on_receipt(self, receipt: OrderReceipt, snapshot: NormalizedSnapshot, oms: 'IOrderManager') -> List[Order]:
        """Called when an order receipt is received.
        
        Args:
            receipt: The order receipt (fill, cancel, etc.)
            snapshot: Current market snapshot
            oms: Order manager for querying order state
            
        Returns:
            List of new orders to submit
        """
        pass

class IOrderManager(ABC):
    """Order manager interface for the new architecture"""
    
    @abstractmethod
    def submit(self, order: Order, submit_time: int) -> None:
        """Submit a new order.
        
        Args:
            order: The order to submit
            submit_time: Time of submission
        """
        pass
    
    @abstractmethod
    def on_receipt(self, receipt: OrderReceipt) -> None:
        """Process an order receipt.
        
        Args:
            receipt: The order receipt to process
        """
        pass
    
    @abstractmethod
    def get_active_orders(self) -> List[Order]:
        """Get all currently active orders.
        
        Returns:
            List of active orders
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID.
        
        Args:
            order_id: The order ID
            
        Returns:
            The order if found, None otherwise
        """
        pass