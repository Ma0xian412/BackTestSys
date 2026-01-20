"""FIFO Exchange Simulator with coordinate-axis queue model.

This module implements exchange matching with:
- X_s(p,t): Queue front consumption coordinate
- Tail coordinate for shadow order position
- No-impact assumption (your orders don't affect market queue)
- Piecewise linear fill time calculation
- Top-5 activation window enforcement
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from ..core.interfaces import IExchangeSimulator
from ..core.types import (
    Order, OrderReceipt, NormalizedSnapshot, Price, Qty, Side, 
    TapeSegment, TimeInForce, OrderStatus
)


EPSILON = 1e-12


@dataclass
class ShadowOrder:
    """A shadow order in the exchange queue.
    
    Uses coordinate-axis model:
    - pos: Starting position on X coordinate axis
    - Order occupies interval [pos, pos + qty)
    - Filled when X(t) >= pos + qty
    """
    order_id: str
    side: Side
    price: Price
    original_qty: Qty
    remaining_qty: Qty
    arrival_time: int
    pos: float  # Position on X coordinate axis
    status: str = "ACTIVE"  # ACTIVE, FILLED, CANCELED
    filled_qty: Qty = 0
    tif: TimeInForce = TimeInForce.GTC


@dataclass
class PriceLevelState:
    """State for a single price level using coordinate-axis model.
    
    Key concepts:
    - Q_mkt: Public market queue depth (not including shadow orders)
    - X: Cumulative consumption from queue front (trades + phi * cancels)
    - Tail = X + Q_mkt: Coordinate of queue tail
    - Shadow orders occupy [pos, pos + qty) after Tail
    """
    side: Side
    price: Price
    q_mkt: float = 0.0  # Public market queue depth
    x_coord: float = 0.0  # Cumulative front consumption
    queue: List[ShadowOrder] = field(default_factory=list)
    
    @property
    def tail_coord(self) -> float:
        """Get tail coordinate: X + Q_mkt"""
        return self.x_coord + self.q_mkt
    
    def total_shadow_qty(self) -> int:
        """Total quantity in active shadow orders."""
        return sum(o.remaining_qty for o in self.queue if o.status == "ACTIVE")
    
    def shadow_qty_at_time(self, t: int) -> int:
        """Shadow order qty at coordinate for orders arriving before t."""
        return sum(o.remaining_qty for o in self.queue 
                  if o.status == "ACTIVE" and o.arrival_time <= t)


class FIFOExchangeSimulator(IExchangeSimulator):
    """No-impact FIFO exchange simulator with coordinate-axis model.
    
    Implements the specification:
    - X_s(p,t): Queue front consumption coordinate
    - Tail_s(p,t) = X_s(p,t) + Q^mkt_s(p,t)
    - Shadow order position: pos = Tail + S^shadow (prior shadow orders)
    - Fill condition: X(t) >= pos + qty
    - Piecewise linear fill time calculation
    """
    
    def __init__(self, cancel_front_ratio: float = 0.5):
        """Initialize the simulator.
        
        Args:
            cancel_front_ratio: phi - proportion of cancels that advance queue front
                              (0 = pessimistic, 0.5 = neutral, 1 = optimistic)
        """
        self.cancel_front_ratio = cancel_front_ratio
        self._levels: Dict[Tuple[Side, Price], PriceLevelState] = {}
        self.current_time: int = 0
        self._current_tape: List[TapeSegment] = []
        self._current_seg_idx: int = 0
        self._interval_start: int = 0
        self._interval_end: int = 0
        
        # Precomputed X rates per segment for fast fill time calculation
        self._x_rates: Dict[Tuple[Side, Price, int], float] = {}
        self._x_at_seg_start: Dict[Tuple[Side, Price, int], float] = {}
    
    def reset(self) -> None:
        """Reset simulator state for new interval."""
        self._levels.clear()
        self.current_time = 0
        self._current_tape = []
        self._current_seg_idx = 0
        self._x_rates.clear()
        self._x_at_seg_start.clear()
    
    def _get_level(self, side: Side, price: Price) -> PriceLevelState:
        """Get or create price level state."""
        key = (side, round(float(price), 8))
        if key not in self._levels:
            self._levels[key] = PriceLevelState(side=side, price=float(price))
        return self._levels[key]
    
    def set_tape(self, tape: List[TapeSegment], t_a: int, t_b: int) -> None:
        """Set the tape for this interval and precompute X rates.
        
        Args:
            tape: List of tape segments
            t_a: Interval start time
            t_b: Interval end time
        """
        self._current_tape = tape
        self._interval_start = t_a
        self._interval_end = t_b
        self._current_seg_idx = 0
        
        # Precompute X rates for each (side, price, segment)
        self._x_rates.clear()
        self._x_at_seg_start.clear()
        
        for seg_idx, seg in enumerate(tape):
            seg_duration = seg.t_end - seg.t_start
            if seg_duration <= 0:
                continue
            
            # For each activated price in this segment
            for side in [Side.BUY, Side.SELL]:
                activation_set = seg.activation_bid if side == Side.BUY else seg.activation_ask
                best_price = seg.bid_price if side == Side.BUY else seg.ask_price
                
                for price in activation_set:
                    key = (side, round(price, 8), seg_idx)
                    
                    # M_{s,i}(p): trades at this price in this segment
                    m_si = seg.trades.get((side, price), 0)
                    
                    # C_{s,i}(p): cancels at this price in this segment
                    c_si = seg.cancels.get((side, price), 0)
                    
                    # X rate: (M + phi * C) / duration
                    x_rate = (m_si + self.cancel_front_ratio * c_si) / seg_duration
                    self._x_rates[key] = x_rate
    
    def _is_in_activation_window(self, side: Side, price: Price, seg_idx: int) -> bool:
        """Check if price is in activation window for given segment."""
        if seg_idx >= len(self._current_tape):
            return False
        seg = self._current_tape[seg_idx]
        activation_set = seg.activation_bid if side == Side.BUY else seg.activation_ask
        return round(price, 8) in {round(p, 8) for p in activation_set}
    
    def _get_x_coord(self, side: Side, price: Price, t: int) -> float:
        """Get X coordinate at time t for given side and price.
        
        X_s(p,t) = cumulative (M + phi * C) from interval start to t
        """
        level = self._get_level(side, price)
        
        if not self._current_tape or t <= self._interval_start:
            return level.x_coord
        
        # Find which segment t falls into
        x = level.x_coord
        for seg_idx, seg in enumerate(self._current_tape):
            if t <= seg.t_start:
                break
            
            seg_start = max(seg.t_start, self._interval_start)
            seg_end = min(seg.t_end, t)
            
            if seg_end <= seg_start:
                continue
            
            # Check activation
            if not self._is_in_activation_window(side, price, seg_idx):
                continue
            
            # Get rate for this segment
            key = (side, round(price, 8), seg_idx)
            rate = self._x_rates.get(key, 0.0)
            
            # Add contribution from this segment
            x += rate * (seg_end - seg_start)
            
            if t <= seg.t_end:
                break
        
        return x
    
    def _get_q_mkt(self, side: Side, price: Price, t: int) -> float:
        """Get market queue depth Q_mkt at time t.
        
        Q_mkt(t) = Q_mkt(t_{i-1}) + N_{s,i}(p) * z - M_{s,i}(p) * z
        where z = (t - t_{i-1}) / delta_t_i
        """
        level = self._get_level(side, price)
        
        if not self._current_tape or t <= self._interval_start:
            return max(0.0, level.q_mkt)
        
        q = level.q_mkt
        
        for seg_idx, seg in enumerate(self._current_tape):
            if t <= seg.t_start:
                break
            
            seg_duration = seg.t_end - seg.t_start
            if seg_duration <= 0:
                continue
            
            # Check activation
            if not self._is_in_activation_window(side, price, seg_idx):
                continue
            
            seg_start = max(seg.t_start, self._interval_start)
            seg_end = min(seg.t_end, t)
            
            if seg_end <= seg_start:
                continue
            
            # Segment progress
            z = (seg_end - seg.t_start) / seg_duration
            z = min(1.0, max(0.0, z))
            
            # N_{s,i}(p): net flow
            n_si = seg.net_flow.get((side, price), 0)
            
            # M_{s,i}(p): trades
            m_si = seg.trades.get((side, price), 0)
            
            # Q changes by (N - M) * z over the segment slice
            q += (n_si - m_si) * z
            
            if t <= seg.t_end:
                break
        
        return max(0.0, q)
    
    def on_order_arrival(self, order: Order, arrival_time: int, market_qty: Qty) -> Optional[OrderReceipt]:
        """Handle order arrival at exchange.
        
        Args:
            order: The arriving order
            arrival_time: Time of arrival (exchtime)
            market_qty: Current market queue depth at price (from snapshot)
            
        Returns:
            Optional receipt for immediate rejection (None if accepted)
        """
        side = order.side
        price = float(order.price)
        
        # Find current segment
        seg_idx = self._find_segment(arrival_time)
        
        # Check if in activation window
        if seg_idx >= 0 and not self._is_in_activation_window(side, price, seg_idx):
            # Price not in top-5 activation window
            # According to spec: "outside top5 - don't track"
            # Order still gets queued but won't have progress until activated
            pass
        
        level = self._get_level(side, price)
        
        # Initialize Q_mkt if first order at this level
        if not level.queue and level.q_mkt == 0:
            level.q_mkt = float(market_qty)
        
        # Calculate position using coordinate-axis model
        # pos = Tail + S^shadow (prior shadow orders' remaining qty)
        x_t = self._get_x_coord(side, price, arrival_time)
        q_mkt_t = self._get_q_mkt(side, price, arrival_time)
        
        # S^shadow: sum of remaining qty from earlier shadow orders
        s_shadow = sum(o.remaining_qty for o in level.queue 
                      if o.status == "ACTIVE" and o.arrival_time < arrival_time)
        
        pos = x_t + q_mkt_t + s_shadow
        
        # Create shadow order
        shadow = ShadowOrder(
            order_id=order.order_id,
            side=side,
            price=price,
            original_qty=order.qty,
            remaining_qty=order.remaining_qty,
            arrival_time=arrival_time,
            pos=pos,
            tif=order.tif,
        )
        
        # Append to queue
        level.queue.append(shadow)
        
        # Handle IOC orders
        if order.tif == TimeInForce.IOC:
            # Check for immediate fill
            if x_t >= pos + order.remaining_qty:
                # Full immediate fill
                shadow.filled_qty = order.remaining_qty
                shadow.remaining_qty = 0
                shadow.status = "FILLED"
                return OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="FILL",
                    timestamp=arrival_time,
                    fill_qty=order.remaining_qty,
                    fill_price=price,
                    remaining_qty=0,
                )
            elif x_t > pos:
                # Partial fill, cancel rest
                fill_qty = int(x_t - pos)
                shadow.filled_qty = fill_qty
                shadow.remaining_qty = 0
                shadow.status = "CANCELED"
                return OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="PARTIAL",
                    timestamp=arrival_time,
                    fill_qty=fill_qty,
                    fill_price=price,
                    remaining_qty=0,
                )
            else:
                # No fill, cancel
                shadow.status = "CANCELED"
                return OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="CANCELED",
                    timestamp=arrival_time,
                    fill_qty=0,
                    fill_price=0.0,
                    remaining_qty=0,
                )
        
        return None
    
    def on_cancel_arrival(self, order_id: str, arrival_time: int) -> OrderReceipt:
        """Handle cancel request.
        
        Args:
            order_id: ID of order to cancel
            arrival_time: Time of cancel arrival (exchtime)
            
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
                    
                    if shadow.status == "CANCELED":
                        return OrderReceipt(
                            order_id=order_id,
                            receipt_type="REJECTED",
                            timestamp=arrival_time,
                        )
                    
                    # Calculate fill up to cancel time
                    x_t = self._get_x_coord(shadow.side, shadow.price, arrival_time)
                    fill_at_cancel = max(0, min(shadow.original_qty, int(x_t - shadow.pos)))
                    
                    shadow.filled_qty = fill_at_cancel
                    shadow.remaining_qty = 0
                    shadow.status = "CANCELED"
                    
                    return OrderReceipt(
                        order_id=order_id,
                        receipt_type="CANCELED",
                        timestamp=arrival_time,
                        fill_qty=fill_at_cancel,
                        remaining_qty=0,
                    )
        
        return OrderReceipt(
            order_id=order_id,
            receipt_type="REJECTED",
            timestamp=arrival_time,
        )
    
    def _find_segment(self, t: int) -> int:
        """Find segment index containing time t."""
        for i, seg in enumerate(self._current_tape):
            if seg.t_start <= t < seg.t_end:
                return i
        return -1
    
    def _compute_fill_time(self, shadow: ShadowOrder, qty_to_fill: int) -> Optional[int]:
        """Compute exchtime when order reaches fill threshold.
        
        Uses piecewise linear X to find when X(t) >= pos + qty_to_fill.
        
        Args:
            shadow: The shadow order
            qty_to_fill: Quantity threshold (usually original_qty for full fill)
            
        Returns:
            Fill time (exchtime) or None if not fillable in interval
        """
        threshold = shadow.pos + qty_to_fill
        side = shadow.side
        price = shadow.price
        
        level = self._get_level(side, price)
        
        # Start X from level's base
        x_running = level.x_coord
        
        for seg_idx, seg in enumerate(self._current_tape):
            if seg.t_end <= shadow.arrival_time:
                # Skip segments before order arrival
                # But still need to accumulate X
                if self._is_in_activation_window(side, price, seg_idx):
                    key = (side, round(price, 8), seg_idx)
                    rate = self._x_rates.get(key, 0.0)
                    seg_duration = seg.t_end - seg.t_start
                    x_running += rate * seg_duration
                continue
            
            seg_duration = seg.t_end - seg.t_start
            if seg_duration <= 0:
                continue
            
            # Check activation
            if not self._is_in_activation_window(side, price, seg_idx):
                continue
            
            key = (side, round(price, 8), seg_idx)
            rate = self._x_rates.get(key, 0.0)
            
            # Effective start time for this segment
            effective_start = max(seg.t_start, shadow.arrival_time)
            
            # X at effective start
            if effective_start > seg.t_start:
                x_at_start = x_running + rate * (effective_start - seg.t_start)
            else:
                x_at_start = x_running
            
            # X at segment end
            x_at_end = x_running + rate * seg_duration
            
            # Check if threshold is crossed in this segment
            if x_at_start < threshold <= x_at_end and rate > EPSILON:
                # Solve: x_at_start + rate * (t - effective_start) = threshold
                delta_t = (threshold - x_at_start) / rate
                fill_time = int(effective_start + delta_t)
                return max(fill_time, effective_start)
            
            x_running = x_at_end
        
        return None
    
    def advance(self, t_from: int, t_to: int, segment: TapeSegment) -> List[OrderReceipt]:
        """Advance simulation from t_from to t_to using tape segment.
        
        Args:
            t_from: Start time
            t_to: End time
            segment: Tape segment containing M and C for this period
            
        Returns:
            List of receipts for fills during this period
        """
        if t_to <= t_from:
            return []
        
        receipts = []
        
        # Find segment index
        seg_idx = -1
        for i, seg in enumerate(self._current_tape):
            if seg.t_start <= t_from < seg.t_end:
                seg_idx = i
                break
        
        # Process each price level with active orders
        for (side, price), level in list(self._levels.items()):
            if not level.queue:
                continue
            
            # Check activation
            if seg_idx >= 0 and not self._is_in_activation_window(side, price, seg_idx):
                continue
            
            # Get X at t_to
            x_t_to = self._get_x_coord(side, price, t_to)
            
            # Check each shadow order
            for shadow in level.queue:
                if shadow.status != "ACTIVE":
                    continue
                if shadow.arrival_time > t_to:
                    continue
                
                # Fill threshold
                threshold = shadow.pos + shadow.original_qty
                
                # Check if threshold is crossed
                if x_t_to >= threshold:
                    # Full fill - compute exact fill time
                    fill_time = self._compute_fill_time(shadow, shadow.original_qty)
                    
                    if fill_time is not None and t_from < fill_time <= t_to:
                        shadow.filled_qty = shadow.original_qty
                        shadow.remaining_qty = 0
                        shadow.status = "FILLED"
                        
                        receipts.append(OrderReceipt(
                            order_id=shadow.order_id,
                            receipt_type="FILL",
                            timestamp=fill_time,
                            fill_qty=shadow.original_qty,
                            fill_price=shadow.price,
                            remaining_qty=0,
                        ))
                elif x_t_to > shadow.pos:
                    # Partial fill
                    current_fill = int(x_t_to - shadow.pos)
                    if current_fill > shadow.filled_qty:
                        new_fill = current_fill - shadow.filled_qty
                        shadow.filled_qty = current_fill
                        shadow.remaining_qty = shadow.original_qty - current_fill
                        
                        receipts.append(OrderReceipt(
                            order_id=shadow.order_id,
                            receipt_type="PARTIAL",
                            timestamp=t_to,
                            fill_qty=new_fill,
                            fill_price=shadow.price,
                            remaining_qty=shadow.remaining_qty,
                        ))
        
        self.current_time = t_to
        return receipts
    
    def align_at_boundary(self, snapshot: NormalizedSnapshot) -> None:
        """Align state at interval boundary.
        
        Updates Q_mkt from snapshot and resets X coordinate for next interval.
        
        Args:
            snapshot: Snapshot at the boundary to align to
        """
        for (side, price), level in self._levels.items():
            # Find queue depth in snapshot
            levels_list = snapshot.bids if side == Side.BUY else snapshot.asks
            observed_qty = 0
            for lvl in levels_list:
                if abs(float(lvl.price) - price) < EPSILON:
                    observed_qty = int(lvl.qty)
                    break
            
            # Update Q_mkt
            level.q_mkt = float(observed_qty)
            
            # Reset X for next interval
            # Note: active shadow orders keep their pos from current X coordinate
            # We need to adjust pos values relative to new X base
            current_x = self._get_x_coord(side, price, self.current_time)
            
            for shadow in level.queue:
                if shadow.status == "ACTIVE":
                    # Adjust pos relative to new X = 0
                    shadow.pos = shadow.pos - current_x
            
            # Reset X to 0
            level.x_coord = 0.0
    
    def get_queue_depth(self, side: Side, price: Price) -> Qty:
        """Get current queue depth at a price level.
        
        Returns Q_mkt + shadow_qty
        
        Args:
            side: BUY or SELL
            price: Price level
            
        Returns:
            Total queue depth
        """
        level = self._get_level(side, price)
        q_mkt = self._get_q_mkt(side, price, self.current_time)
        return int(q_mkt) + level.total_shadow_qty()
    
    def get_x_coord(self, side: Side, price: Price) -> float:
        """Get current X coordinate for diagnostics."""
        return self._get_x_coord(side, price, self.current_time)
    
    def get_shadow_orders(self) -> List[ShadowOrder]:
        """Get all shadow orders for diagnostics."""
        result = []
        for level in self._levels.values():
            result.extend(level.queue)
        return result
