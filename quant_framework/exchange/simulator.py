"""FIFO Exchange Simulator with no-impact queue management.

This module implements exchange matching with:
- FIFO queue for shadow orders
- No-impact assumption (your orders don't affect market queue)
- ahead tracking for anonymous market quantity
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.interfaces import IExchangeSimulator
from ..core.types import Order, OrderReceipt, NormalizedSnapshot, Price, Qty, Side, TapeSegment


@dataclass
class ShadowOrder:
    """A shadow order in the exchange queue."""
    order_id: str
    side: Side
    price: Price
    remaining_qty: Qty
    arrival_time: int
    status: str = "ACTIVE"  # ACTIVE, FILLED, CANCELED
    filled_qty: Qty = 0


@dataclass
class PriceLevelState:
    """State for a single price level."""
    side: Side
    price: Price
    ahead: int = 0  # Anonymous market quantity ahead
    queue: List[ShadowOrder] = field(default_factory=list)  # Shadow order FIFO queue
    
    def total_shadow_qty(self) -> int:
        """Total quantity in shadow orders."""
        return sum(o.remaining_qty for o in self.queue if o.status == "ACTIVE")


class FIFOExchangeSimulator(IExchangeSimulator):
    """No-impact FIFO exchange simulator.
    
    Maintains per-price-level queues with:
    - ahead: anonymous market quantity
    - queue: FIFO list of shadow orders
    """
    
    def __init__(self, cancel_front_ratio: float = 0.5):
        """Initialize the simulator.
        
        Args:
            cancel_front_ratio: Proportion of cancels in front (0=pessimistic, 1=optimistic)
        """
        self.cancel_front_ratio = cancel_front_ratio
        self._levels: Dict[Tuple[Side, Price], PriceLevelState] = {}
        self.current_time: int = 0
    
    def reset(self) -> None:
        """Reset simulator state."""
        self._levels.clear()
        self.current_time = 0
    
    def _get_level(self, side: Side, price: Price) -> PriceLevelState:
        """Get or create price level state."""
        key = (side, float(price))
        if key not in self._levels:
            self._levels[key] = PriceLevelState(side=side, price=float(price))
        return self._levels[key]
    
    def on_order_arrival(self, order: Order, arrival_time: int, market_qty: Qty) -> Optional[OrderReceipt]:
        """Handle order arrival at exchange.
        
        Args:
            order: The arriving order
            arrival_time: Time of arrival
            market_qty: Current market queue depth at price
            
        Returns:
            Optional receipt for immediate rejection
        """
        side = order.side
        price = float(order.price)
        
        level = self._get_level(side, price)
        
        # Create shadow order
        shadow = ShadowOrder(
            order_id=order.order_id,
            side=side,
            price=price,
            remaining_qty=order.remaining_qty,
            arrival_time=arrival_time,
        )
        
        # Initialize ahead if first order at this level
        if not level.queue:
            level.ahead = market_qty
        
        # Append to queue
        level.queue.append(shadow)
        
        return None
    
    def on_cancel_arrival(self, order_id: str, arrival_time: int) -> OrderReceipt:
        """Handle cancel request.
        
        Args:
            order_id: ID of order to cancel
            arrival_time: Time of cancel arrival
            
        Returns:
            Receipt for the cancel operation
        """
        # Find order in all levels
        for level in self._levels.values():
            for shadow in level.queue:
                if shadow.order_id == order_id:
                    if shadow.status == "FILLED":
                        return OrderReceipt(
                            order_id=order_id,
                            receipt_type="REJECTED",
                            timestamp=arrival_time,
                        )
                    
                    cancelled_qty = shadow.remaining_qty
                    shadow.remaining_qty = 0
                    shadow.status = "CANCELED"
                    
                    return OrderReceipt(
                        order_id=order_id,
                        receipt_type="CANCELED",
                        timestamp=arrival_time,
                        fill_qty=shadow.filled_qty,
                        remaining_qty=0,
                    )
        
        return OrderReceipt(
            order_id=order_id,
            receipt_type="REJECTED",
            timestamp=arrival_time,
        )
    
    def advance(self, t_from: int, t_to: int, segment: TapeSegment) -> List[OrderReceipt]:
        """Advance simulation using tape segment.
        
        Args:
            t_from: Start time
            t_to: End time
            segment: Tape segment with trades and cancels
            
        Returns:
            List of receipts for fills
        """
        if t_to <= t_from:
            return []
        
        receipts = []
        seg_duration = segment.t_end - segment.t_start
        
        if seg_duration <= 0:
            return []
        
        slice_start = max(t_from, segment.t_start)
        slice_end = min(t_to, segment.t_end)
        
        if slice_end <= slice_start:
            return []
        
        slice_frac = (slice_end - slice_start) / seg_duration
        
        # Process each side
        for side in [Side.BUY, Side.SELL]:
            best_price = segment.bid_price if side == Side.BUY else segment.ask_price
            
            # Get segment volumes at best price
            m_total = segment.trades.get((side, best_price), 0)
            c_total = segment.cancels.get((side, best_price), 0)
            
            # Scale to this slice
            m_slice = int(round(m_total * slice_frac))
            c_slice = int(round(c_total * slice_frac))
            
            level = self._get_level(side, best_price)
            
            # Process cancellations (advance ahead)
            if c_slice > 0:
                c_front = int(round(self.cancel_front_ratio * c_slice))
                level.ahead = max(0, level.ahead - c_front)
            
            # Process trades
            if m_slice > 0:
                fill_receipts = self._consume_trades(level, m_slice, slice_start, slice_end)
                receipts.extend(fill_receipts)
        
        self.current_time = t_to
        return receipts
    
    def _consume_trades(self, level: PriceLevelState, trade_qty: int, 
                        t_start: int, t_end: int) -> List[OrderReceipt]:
        """Consume trade volume using FIFO.
        
        Args:
            level: Price level state
            trade_qty: Total trade volume
            t_start: Start time
            t_end: End time
            
        Returns:
            List of fill receipts
        """
        receipts = []
        m_rem = trade_qty
        
        # First consume ahead
        consumed_ahead = min(m_rem, level.ahead)
        level.ahead -= consumed_ahead
        m_rem -= consumed_ahead
        
        # Then FIFO through shadow orders
        for shadow in level.queue:
            if m_rem <= 0:
                break
            if shadow.remaining_qty <= 0 or shadow.status != "ACTIVE":
                continue
            
            fill_qty = min(m_rem, shadow.remaining_qty)
            shadow.remaining_qty -= fill_qty
            shadow.filled_qty += fill_qty
            m_rem -= fill_qty
            
            # Calculate fill time (linear interpolation)
            fill_time = t_end if trade_qty == 0 else int(t_start + (t_end - t_start) * (trade_qty - m_rem) / trade_qty)
            
            if shadow.remaining_qty <= 0:
                shadow.status = "FILLED"
                receipts.append(OrderReceipt(
                    order_id=shadow.order_id,
                    receipt_type="FILL",
                    timestamp=fill_time,
                    fill_qty=shadow.filled_qty,
                    fill_price=shadow.price,
                    remaining_qty=0,
                ))
            else:
                receipts.append(OrderReceipt(
                    order_id=shadow.order_id,
                    receipt_type="PARTIAL",
                    timestamp=fill_time,
                    fill_qty=fill_qty,
                    fill_price=shadow.price,
                    remaining_qty=shadow.remaining_qty,
                ))
        
        return receipts
    
    def align_at_boundary(self, snapshot: NormalizedSnapshot) -> None:
        """Align state at interval boundary.
        
        Args:
            snapshot: Snapshot to align to
        """
        for (side, price), level in self._levels.items():
            if not level.queue:
                continue
            
            # Find queue depth in snapshot
            levels = snapshot.bids if side == Side.BUY else snapshot.asks
            observed_qty = 0
            for lvl in levels:
                if abs(float(lvl.price) - price) < 1e-9:
                    observed_qty = int(lvl.qty)
                    break
            
            # Clamp ahead
            level.ahead = min(max(level.ahead, 0), observed_qty)
    
    def get_queue_depth(self, side: Side, price: Price) -> Qty:
        """Get queue depth at a price level.
        
        Args:
            side: BUY or SELL
            price: Price level
            
        Returns:
            Total queue depth (ahead + shadow orders)
        """
        level = self._get_level(side, price)
        return level.ahead + level.total_shadow_qty()
