"""FIFO Exchange Simulator with coordinate-axis queue model.

This module implements exchange matching with:
- X_s(p,t): Queue front consumption coordinate
- Tail coordinate for shadow order position
- No-impact assumption (your orders don't affect market queue)
- Piecewise linear fill time calculation
- Top-5 activation window enforcement
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from ..core.interfaces import IExchangeSimulator
from ..core.types import (
    Order, OrderReceipt, NormalizedSnapshot, Price, Qty, Side, 
    TapeSegment, TimeInForce, OrderStatus
)


# 设置模块级logger
logger = logging.getLogger(__name__)


EPSILON = 1e-12



@dataclass
class ShadowOrder:
    """A shadow order in the exchange queue.
    
    Uses coordinate-axis model:
    - pos: Starting position on X coordinate axis
    - Order occupies interval [pos, pos + qty)
    - Filled when X(t) >= pos + qty
    
    Post-crossing orders:
    - is_post_crossing: True if this order is a remainder after crossing
    - crossed_prices: List of (side, price) tuples that were crossed
    - This is retained for diagnostics only. Matching is now based on
      execution-best price priority, not post-crossing side paths.
    """
    order_id: str
    side: Side
    price: Price
    original_qty: Qty
    remaining_qty: Qty
    arrival_time: int
    pos: int  # Position on X coordinate axis (integer: number of lots ahead in queue)
    status: str = "ACTIVE"  # ACTIVE, FILLED, CANCELED
    filled_qty: Qty = 0
    tif: TimeInForce = TimeInForce.GTC
    is_post_crossing: bool = False  # True if this is remainder after crossing
    crossed_prices: List[Tuple[Side, Price]] = field(default_factory=list)  # Prices that were crossed


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
    _active_shadow_qty: int = 0  # Cached total of active shadow order qty
    
    @property
    def tail_coord(self) -> float:
        """Get tail coordinate: X + Q_mkt"""
        return self.x_coord + self.q_mkt
    
    def total_shadow_qty(self) -> int:
        """Total quantity in active shadow orders."""
        return self._active_shadow_qty
    
    def _recompute_active_qty(self) -> None:
        """Recompute cached active shadow qty (call after status changes)."""
        self._active_shadow_qty = sum(o.remaining_qty for o in self.queue if o.status == "ACTIVE")
    
    def shadow_qty_at_time(self, t: int) -> int:
        """Shadow order qty at coordinate for orders arriving before t."""
        return sum(o.remaining_qty for o in self.queue 
                  if o.status == "ACTIVE" and o.arrival_time < t)


class FIFOExchangeSimulator(IExchangeSimulator):
    """No-impact FIFO exchange simulator with coordinate-axis model.
    
    Implements the specification:
    - X_s(p,t): Queue front consumption coordinate
    - Tail_s(p,t) = X_s(p,t) + Q^mkt_s(p,t)
    - Shadow order position: pos = Tail + S^shadow (prior shadow orders)
    - Fill condition: X(t) >= pos + qty
    - Piecewise linear fill time calculation
    """
    
    def __init__(self, cancel_bias_k: float = 0.0):
        """Initialize the simulator.
        
        Args:
            cancel_bias_k: Cancellation bias parameter k, ranging from -1 to 1.
                          Controls position-dependent cancel probability p_k(x):
                          - k < 0: p_k(x) = x^(1+k), biased toward cancels from behind
                          - k = 0: p_k(x) = x, uniform distribution
                          - k > 0: p_k(x) = 1 - (1-x)^(1-k), biased toward cancels from front
                          Common values: k=0 (uniform), k=-0.8 (moderate bias), k=-0.95 (strong bias)
        """
        self.cancel_bias_k = cancel_bias_k
        self._levels: Dict[Tuple[Side, Price], PriceLevelState] = {}
        self.current_time: int = 0
        self._current_tape: List[TapeSegment] = []
        self._current_seg_idx: int = 0
        self._interval_start: int = 0
        self._interval_end: int = 0
        
        # Track order IDs that were fully filled immediately (crossing orders)
        # This allows cancel requests to return REJECTED instead of ValueError
        # Note: Order IDs are removed from this set after being referenced in a cancel,
        # preventing unbounded growth. full_reset() also clears this set.
        self._filled_order_ids: set = set()

        # Trade pause intervals for improvement-mode execution (per side)
        # These intervals suppress trade contribution to X while still allowing cancels.
        self._trade_pause_intervals: Dict[Side, List[Tuple[int, int]]] = {
            Side.BUY: [],
            Side.SELL: [],
        }

    def _compute_cancel_front_prob(self, x: float) -> float:
        """Compute position-dependent cancel probability p_k(x).
        
        When your order is at normalized position x in the queue (0=front, 1=tail),
        this returns the probability that a cancel happens in front of you.
        
        Model:
        - p(1) = 1: At queue tail, all cancels happen in front
        - p(0) = 0: At queue front, all cancels happen behind
        - Parameter k controls the bias:
          - k < 0: p_k(x) = x^(1+k), cancels more likely from behind (realistic)
          - k = 0: p_k(x) = x, uniform distribution
          - k > 0: p_k(x) = 1 - (1-x)^(1-k), cancels more likely from front
        
        Args:
            x: Normalized queue position (0 = front, 1 = tail)
            
        Returns:
            Probability of cancel happening in front of the order
        """
        # Clamp x to [0, 1]
        x = max(0.0, min(1.0, x))
        
        k = self.cancel_bias_k
        
        if k < 0:
            # p_k(x) = x^(1+k)
            exponent = 1.0 + k
            if exponent <= 0:
                return 1.0 if x > 0 else 0.0
            return x ** exponent
        elif k == 0:
            # Uniform: p(x) = x
            return x
        else:
            # k > 0: p_k(x) = 1 - (1-x)^(1-k)
            exponent = 1.0 - k
            if exponent <= 0:
                return 0.0 if x < 1 else 1.0
            return 1.0 - (1.0 - x) ** exponent

    def _add_trade_pause_interval(self, side: Side, t_start: int, t_end: int) -> None:
        """Record a trade-suppressed interval for a side."""
        if t_end <= t_start:
            return
        intervals = self._trade_pause_intervals[side]
        if intervals and t_start <= intervals[-1][1]:
            last_start, last_end = intervals[-1]
            intervals[-1] = (last_start, max(last_end, t_end))
        else:
            intervals.append((t_start, t_end))

    def _get_trade_active_duration(self, side: Side, t_start: int, t_end: int) -> float:
        """Get duration with trade active (not paused) between times."""
        if t_end <= t_start:
            return 0.0
        active = float(t_end - t_start)
        for pause_start, pause_end in self._trade_pause_intervals[side]:
            if pause_end <= t_start:
                continue
            if pause_start >= t_end:
                break
            overlap = min(t_end, pause_end) - max(t_start, pause_start)
            if overlap > 0:
                active -= overlap
        return max(0.0, active)

    def _compute_trade_rate_for_segment(self, side: Side, price: Price, seg_idx: int) -> float:
        """Compute trade rate for a segment (per time unit)."""
        seg = self._current_tape[seg_idx]
        seg_duration = seg.t_end - seg.t_start
        if seg_duration <= 0:
            return 0.0
        m_si = seg.trades.get((side, price), 0)
        return m_si / seg_duration

    def _compute_cancel_rate_for_segment(
        self, side: Side, price: Price, seg_idx: int, x_running: float, shadow_pos: int
    ) -> float:
        """Compute cancel contribution rate for a segment."""
        seg = self._current_tape[seg_idx]
        seg_duration = seg.t_end - seg.t_start
        if seg_duration <= 0:
            return 0.0

        c_si = seg.cancels.get((side, price), 0)
        if c_si == 0:
            return 0.0

        q_mkt_at_seg_start = self._get_q_mkt(side, price, seg.t_start)
        remaining_ahead = shadow_pos - x_running

        if q_mkt_at_seg_start > EPSILON and remaining_ahead > 0:
            x_norm = remaining_ahead / q_mkt_at_seg_start
            x_norm = min(1.0, max(0.0, x_norm))
            cancel_prob = self._compute_cancel_front_prob(x_norm)
        else:
            cancel_prob = 0.0

        return cancel_prob * c_si / seg_duration

    # ── Zone-aware cancel distribution helpers ──────────────────────────

    def _get_active_shadows_sorted(self, side: Side, price: Price) -> List[ShadowOrder]:
        """Get active shadow orders at a price level, sorted by pos."""
        level = self._get_level(side, price)
        return sorted(
            [s for s in level.queue if s.status == "ACTIVE" and s.remaining_qty > 0],
            key=lambda s: s.pos,
        )

    def _build_queue_zones(
        self, side: Side, price: Price, x_running: float, q_mkt: float
    ) -> List[Tuple[str, float, float, Optional[ShadowOrder]]]:
        """Build zone structure from current x position.

        Zones alternate between "public" (market orders) and "shadow" (our
        orders).  A trailing public zone is appended so that the sum of all
        public zone sizes equals *q_mkt* – this is needed for correct CDF-
        based cancel distribution.

        Returns:
            List of (zone_type, start_coord, size, shadow_or_None).
        """
        shadows = self._get_active_shadows_sorted(side, price)
        zones: List[Tuple[str, float, float, Optional[ShadowOrder]]] = []
        cursor = x_running

        for shadow in shadows:
            shadow_start = float(shadow.pos)
            shadow_end = float(shadow.pos + shadow.original_qty)

            # Already past this shadow
            if cursor >= shadow_end - EPSILON:
                continue

            # Currently inside this shadow
            if cursor >= shadow_start - EPSILON:
                remaining = shadow_end - cursor
                if remaining > EPSILON:
                    zones.append(("shadow", cursor, remaining, shadow))
                cursor = shadow_end
                continue

            # Public zone before this shadow
            pub_size = shadow_start - cursor
            if pub_size > EPSILON:
                zones.append(("public", cursor, pub_size, None))

            # Shadow zone
            shadow_size = shadow_end - shadow_start
            if shadow_size > EPSILON:
                zones.append(("shadow", shadow_start, shadow_size, shadow))
            cursor = shadow_end

        # Trailing public zone (orders behind all shadows)
        total_pub_in_zones = sum(sz for zt, _, sz, _ in zones if zt == "public")
        trailing = max(0.0, q_mkt - total_pub_in_zones)
        if trailing > EPSILON:
            zones.append(("public", cursor, trailing, None))

        return zones

    def _distribute_cancels_to_zones(
        self,
        zones: List[Tuple[str, float, float, Optional[ShadowOrder]]],
        total_cancels: float,
        q_mkt: float,
    ) -> Dict[int, float]:
        """Distribute cancels to public zones using cancel_prob CDF + cap.

        Algorithm:
        1. Compute cumulative public position for each public zone.
        2. Use cancel_prob CDF to derive raw share per zone.
        3. Front-to-back cap (zone cancels ≤ zone size) + overflow.

        Returns:
            dict  zone_index → cancel_count  (only public zones).
        """
        if total_cancels <= EPSILON or q_mkt <= EPSILON:
            return {}

        pub_info: List[Tuple[int, float]] = [
            (i, sz) for i, (zt, _, sz, _) in enumerate(zones) if zt == "public"
        ]
        if not pub_info:
            return {}

        sizes = [sz for _, sz in pub_info]

        # Cumulative public position → CDF boundaries
        cum = [0.0]
        for s in sizes:
            cum.append(cum[-1] + s)

        # Raw shares from CDF
        raw_shares: List[float] = []
        for j in range(len(sizes)):
            x_lo = min(1.0, max(0.0, cum[j] / q_mkt))
            x_hi = min(1.0, max(0.0, cum[j + 1] / q_mkt))
            p_lo = self._compute_cancel_front_prob(x_lo)
            p_hi = self._compute_cancel_front_prob(x_hi)
            raw_shares.append(total_cancels * max(0.0, p_hi - p_lo))

        # Front-to-back cap + overflow redistribution
        result: Dict[int, float] = {}
        excess = 0.0
        for j, (idx, sz) in enumerate(pub_info):
            available = raw_shares[j] + excess
            actual = min(available, max(0.0, sz))
            result[idx] = actual
            excess = max(0.0, available - actual)

        return result

    def _traverse_zones_for_x(
        self,
        x_start: float,
        zones: List[Tuple[str, float, float, Optional[ShadowOrder]]],
        cancel_dist: Dict[int, float],
        trade_rate: float,
        seg_duration: int,
        t_abs_start: int,
        dt_available: int,
        target_x: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Piece-wise zone traversal.

        * Public zones:  rate = trade_rate + cancel_rate_i
          (cancel_rate_i = zone_cancels / seg_duration).
          Pre-shrinkage from cancels that occurred before *t_abs_start +
          t_elapsed* is subtracted from the remaining zone size.
        * Shadow zones:  rate = trade_rate  (cancels skip shadows).

        If *target_x* is given the traversal stops as soon as *x* reaches
        that coordinate and the elapsed time is returned.

        Returns:
            (x_end, time_elapsed)
        """
        x = x_start
        t_elapsed = 0.0

        for i, (ztype, zone_start, zone_size, _shadow) in enumerate(zones):
            remaining_dt = dt_available - t_elapsed
            if remaining_dt <= EPSILON:
                break
            if target_x is not None and x >= target_x - EPSILON:
                break

            zone_end = zone_start + zone_size

            if ztype == "public":
                cancel_count = cancel_dist.get(i, 0.0)
                cancel_rate = cancel_count / seg_duration if seg_duration > EPSILON else 0.0

                # Zone may have pre-shrunk from cancels before our window
                t_abs = t_abs_start + t_elapsed
                pre_shrunk = cancel_rate * t_abs
                remaining = max(0.0, zone_size - pre_shrunk)

                if remaining <= EPSILON:
                    # Already consumed by cancels
                    x = zone_end
                    continue

                consumption_rate = trade_rate + cancel_rate
                if consumption_rate <= EPSILON:
                    # No way to advance; stop here
                    break

                # Check whether target_x falls inside this zone
                if target_x is not None and x < target_x <= zone_end + EPSILON:
                    dt_to_target = max(0.0, target_x - x) / consumption_rate
                    if dt_to_target <= remaining_dt + EPSILON:
                        t_elapsed += dt_to_target
                        x = target_x
                        break

                dt_to_clear = remaining / consumption_rate
                if dt_to_clear <= remaining_dt + EPSILON:
                    x = zone_end
                    t_elapsed += dt_to_clear
                else:
                    x += consumption_rate * remaining_dt
                    t_elapsed = dt_available
                    break

            else:  # shadow
                if trade_rate <= EPSILON:
                    break  # cannot advance through shadow without trades

                remaining_shadow = max(0.0, zone_end - x)
                if remaining_shadow <= EPSILON:
                    x = zone_end
                    continue

                # Check whether target_x falls inside this shadow
                if target_x is not None and x < target_x <= zone_end + EPSILON:
                    dt_to_target = max(0.0, target_x - x) / trade_rate
                    if dt_to_target <= remaining_dt + EPSILON:
                        t_elapsed += dt_to_target
                        x = target_x
                        break

                dt_to_clear = remaining_shadow / trade_rate
                if dt_to_clear <= remaining_dt + EPSILON:
                    x = zone_end
                    t_elapsed += dt_to_clear
                else:
                    x += trade_rate * remaining_dt
                    t_elapsed = dt_available
                    break

        return x, t_elapsed

    # ── End zone-aware helpers ────────────────────────────────────────

    def _validate_fill_delta(self, order_id: str, delta: int, filled_qty: int, original_qty: int) -> bool:
        """Validate fill delta to avoid negative fill quantities."""
        if delta < 0:
            logger.warning(
                f"[Exchange] Advance: skip negative fill delta for {order_id}, "
                f"filled_qty={filled_qty}, original_qty={original_qty}"
            )
            return False
        return True

    def _apply_shadow_fill(self, shadow: ShadowOrder, fill_qty: int, timestamp: int) -> OrderReceipt:
        """Apply a fill to a shadow order and generate a receipt."""
        level_key = (shadow.side, round(float(shadow.price), 8))
        level = self._levels.get(level_key)
        if level is None:
            level = self._get_level(shadow.side, shadow.price)

        level._active_shadow_qty -= fill_qty
        shadow.filled_qty += fill_qty
        shadow.remaining_qty -= fill_qty

        if shadow.remaining_qty <= 0:
            shadow.remaining_qty = 0
            shadow.status = "FILLED"
            receipt_type = "FILL"
        else:
            receipt_type = "PARTIAL"

        return OrderReceipt(
            order_id=shadow.order_id,
            receipt_type=receipt_type,
            timestamp=timestamp,
            fill_qty=fill_qty,
            fill_price=shadow.price,
            remaining_qty=shadow.remaining_qty,
        )
    
    def _find_order_by_id(self, order_id: str) -> Optional[ShadowOrder]:
        """Find a shadow order by its ID.
        
        Searches through all price levels to find the order.
        
        Args:
            order_id: The order ID to search for
            
        Returns:
            The ShadowOrder if found, None otherwise
            
        Note:
            This is O(n*m) where n is number of price levels and m is average queue size.
            This is acceptable because:
            1. Cancel operations are infrequent compared to fills
            2. Number of active price levels and orders is typically small
            3. Simplicity of single data source outweighs performance cost
            
            If cancels become a bottleneck, consider adding an order_id index.
        """
        for level in self._levels.values():
            for shadow in level.queue:
                if shadow.order_id == order_id:
                    return shadow
        return None
    
    def reset(self) -> None:
        """Reset simulator state for new interval.
        
        This resets interval-specific state (tape, coordinates, X rates) but
        preserves the price levels (_levels) including shadow orders, allowing
        orders to span multiple intervals naturally.
        
        The key insight is that align_at_boundary() has already:
        1. Updated q_mkt from the new snapshot
        2. Adjusted shadow order pos values relative to X=0
        3. Reset x_coord to 0
        
        So we only need to clear tape-related caches here.
        
        Note: This method is typically called after align_at_boundary() which
        already resets x_coord. The x_coord reset here is for completeness.
        """
        # Reset interval-specific state
        self.current_time = 0
        self._current_tape = []
        self._current_seg_idx = 0
        self._trade_pause_intervals[Side.BUY].clear()
        self._trade_pause_intervals[Side.SELL].clear()
        
        # Reset X coordinate for all levels
        # (align_at_boundary already does this, but included here for completeness)
        for level in self._levels.values():
            level.x_coord = 0.0
        
        # Note: _levels is intentionally NOT cleared to preserve shadow orders
        # across interval boundaries. Their pos values were adjusted by
        # align_at_boundary() at the end of the previous interval.
        # Note: _filled_order_ids is also NOT cleared as orders may be canceled
        # in later intervals
    
    def full_reset(self) -> None:
        """Fully reset simulator state including levels.
        
        Call this when starting a new backtest session to clear all state.
        """
        # First do interval reset
        self.reset()
        # Then clear persistent state
        self._levels.clear()
        self._filled_order_ids.clear()
    
    def _get_level(self, side: Side, price: Price) -> PriceLevelState:
        """Get or create price level state."""
        key = (side, round(float(price), 8))
        if key not in self._levels:
            self._levels[key] = PriceLevelState(side=side, price=float(price))
        return self._levels[key]

    def _ensure_base_q_mkt(self, side: Side, price: Price, market_qty: Qty) -> PriceLevelState:
        """Ensure base market queue depth is initialized from snapshot.
        
        Args:
            side: Queue side to initialize.
            price: Price level for the queue.
            market_qty: Snapshot market quantity used as base depth.
            
        Returns:
            The price level state after initialization.
        """
        level = self._get_level(side, price)
        if level.q_mkt == 0:
            level.q_mkt = float(market_qty)
        return level

    def _get_total_queue_depth(self, side: Side, price: Price, t: int) -> float:
        """Get total queue depth including market and shadow orders.
        
        Args:
            side: Queue side to query.
            price: Price level to query.
            t: Time for queue depth calculation.
            
        Returns:
            Total quantity in the queue at time t (market depth + shadow orders).
        """
        level = self._get_level(side, price)
        # _get_q_mkt uses time-based interpolation for market depth.
        return self._get_q_mkt(side, price, t) + level.shadow_qty_at_time(t)
    
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
        
        # No need to restore orders - reset() now preserves _levels
        # Shadow orders remain in their price levels across intervals
        
        # Validate segments
        for seg_idx, seg in enumerate(tape):
            seg_duration = seg.t_end - seg.t_start
            if seg_duration <= 0:
                raise ValueError(
                    f"Invalid segment {seg_idx} in set_tape: "
                    f"seg_duration={seg_duration} <= 0 (t_start={seg.t_start}, t_end={seg.t_end})"
                )
    
    def _is_in_activation_window(self, side: Side, price: Price, seg_idx: int) -> bool:
        """Check if price is in activation window for given segment."""
        if seg_idx >= len(self._current_tape):
            return False
        seg = self._current_tape[seg_idx]
        activation_set = seg.activation_bid if side == Side.BUY else seg.activation_ask
        return round(price, 8) in {round(p, 8) for p in activation_set}
    
    def _get_x_coord(self, side: Side, price: Price, t: int, shadow_pos: int = 0) -> float:
        """Get X coordinate at time t using zone-aware model.

        The queue is split into *public* zones (market orders) and *shadow*
        zones (our orders).  Cancels are distributed to public zones via the
        cancel_prob CDF + cap algorithm.  x advances through public zones at
        ``trade_rate + cancel_rate`` and through shadow zones at ``trade_rate``
        only (cancels skip shadows).

        The *shadow_pos* parameter is retained for API compatibility but is no
        longer used — zone structure considers ALL active shadow orders.
        """
        level = self._get_level(side, price)

        if not self._current_tape or t <= self._interval_start:
            return level.x_coord

        x = level.x_coord
        for seg_idx, seg in enumerate(self._current_tape):
            if t <= seg.t_start:
                break

            seg_start = max(seg.t_start, self._interval_start)
            seg_end = min(seg.t_end, t)

            if seg_end <= seg_start:
                continue

            if not self._is_in_activation_window(side, price, seg_idx):
                continue

            seg_duration = seg.t_end - seg.t_start
            if seg_duration <= 0:
                continue

            total_cancels = seg.cancels.get((side, price), 0)
            total_trades = seg.trades.get((side, price), 0)
            q_mkt = self._get_q_mkt(side, price, seg.t_start)

            # Effective trade rate (accounts for improvement-mode pauses)
            trade_rate_base = total_trades / seg_duration
            dt = seg_end - seg_start
            trade_active = self._get_trade_active_duration(side, seg_start, seg_end)
            effective_trade_rate = (
                trade_rate_base * trade_active / dt if dt > 0 else 0.0
            )

            # Build zones & distribute cancels
            zones = self._build_queue_zones(side, price, x, q_mkt)
            cancel_dist = self._distribute_cancels_to_zones(zones, total_cancels, q_mkt)

            # Piece-wise traversal
            t_abs_start = seg_start - seg.t_start
            x, _ = self._traverse_zones_for_x(
                x, zones, cancel_dist, effective_trade_rate,
                seg_duration, t_abs_start, dt,
            )

            if t <= seg.t_end:
                break

        return x
    
    def _get_q_mkt(self, side: Side, price: Price, t: int) -> float:
        """根据segment进度计算时刻t的市场队列深度Q_mkt。
        
        根据零约束和激活窗口，净增量(net_flow)在各segment的分配是已知的。
        如果该价位是segment中的最优档位，则交易量(trades/消耗量)也是已知的。
        根据时刻t所处segment的进度和初始状态，计算队列长度。
        
        计算公式:
        Q_mkt(t) = Q_mkt(T_A) + Σ(N_{s,i}(p) - M_{s,i}(p)) * z_i
        
        其中:
        - Q_mkt(T_A): level.q_mkt，区间起点的队列深度（基础值）
        - N_{s,i}(p): segment i 在价位p的净增量(net_flow)
        - M_{s,i}(p): segment i 在价位p的交易量(trades/消耗量)
        - z_i: segment i 的进度，z = (t - seg.t_start) / (seg.t_end - seg.t_start)
        
        例如：arrival_time=4，位于segment[2,5]中
        进度 z = (4-2)/(5-2) = 2/3
        
        Args:
            side: 买卖方向
            price: 价格档位
            t: 目标时刻
            
        Returns:
            时刻t的市场队列深度
        """
        level = self._get_level(side, price)
        
        if not self._current_tape or t <= self._interval_start:
            return max(0.0, level.q_mkt)
        
        q = level.q_mkt  # 基础值：区间起点T_A的队列深度
        
        for seg_idx, seg in enumerate(self._current_tape):
            if t <= seg.t_start:
                break
            
            seg_duration = seg.t_end - seg.t_start
            if seg_duration <= 0:
                raise ValueError(
                    f"Invalid segment {seg_idx} in _get_q_mkt: "
                    f"seg_duration={seg_duration} <= 0 (t_start={seg.t_start}, t_end={seg.t_end})"
                )
            
            # Check activation
            if not self._is_in_activation_window(side, price, seg_idx):
                continue
            
            seg_start = max(seg.t_start, self._interval_start)
            seg_end = min(seg.t_end, t)
            
            if seg_end <= seg_start:
                continue
            
            # Segment progress: z = (t - seg.t_start) / (seg.t_end - seg.t_start)
            # 例如：t=4在segment[2,5]中，z = (4-2)/(5-2) = 2/3
            z = (seg_end - seg.t_start) / seg_duration
            z = min(1.0, max(0.0, z))
            
            # N_{s,i}(p): 净增量(net flow)
            n_si = seg.net_flow.get((side, price), 0)
            
            # M_{s,i}(p): 交易量(trades/消耗量)
            m_si = seg.trades.get((side, price), 0)
            
            # Q变化量 = (净增量 - 交易量) * 进度
            q += (n_si - m_si) * z
            
            if t <= seg.t_end:
                break
        
        return max(0.0, q)
    
    def _get_positive_netflow_between(self, side: Side, price: Price, t_from: int, t_to: int) -> float:
        """计算t_from到t_to之间的正净流入累计量。
        
        用于计算两个shadow订单之间的队列增量。
        如果在某个时间段内净流入为负，则该段贡献为0（队列收缩不会增加后续订单的距离）。
        
        Args:
            side: 买卖方向
            price: 价格档位
            t_from: 起始时间
            t_to: 结束时间
            
        Returns:
            正净流入累计量（只计算正值，负值视为0）
        """
        if not self._current_tape or t_to <= t_from:
            return 0.0
        
        total_positive_netflow = 0.0
        
        for seg_idx, seg in enumerate(self._current_tape):
            if t_to <= seg.t_start:
                break
            if t_from >= seg.t_end:
                continue
            
            seg_duration = seg.t_end - seg.t_start
            if seg_duration <= 0:
                raise ValueError(
                    f"Invalid segment {seg_idx} in _get_positive_netflow_between: "
                    f"seg_duration={seg_duration} <= 0 (t_start={seg.t_start}, t_end={seg.t_end})"
                )
            
            # Check activation
            if not self._is_in_activation_window(side, price, seg_idx):
                continue
            
            # Calculate the overlap between [t_from, t_to] and [seg.t_start, seg.t_end]
            overlap_start = max(seg.t_start, t_from)
            overlap_end = min(seg.t_end, t_to)
            
            if overlap_end <= overlap_start:
                continue
            
            # Segment progress for the overlap period
            z = (overlap_end - overlap_start) / seg_duration
            
            # N_{s,i}(p): 净增量(net flow)
            n_si = seg.net_flow.get((side, price), 0)
            
            # Only count positive netflow (queue growth)
            # When queue shrinks (negative netflow), distance between orders doesn't increase
            if n_si > 0:
                total_positive_netflow += n_si * z
        
        return total_positive_netflow
    
    def on_order_arrival(self, order: Order, arrival_time: int, market_qty: Qty) -> Optional[OrderReceipt]:
        """Handle order arrival at exchange.
        
        处理流程：
        1. 首先检查订单是否会立即成交（crossing check）
           - BUY订单: 如果 price >= ask_best，可立即成交
           - SELL订单: 如果 price <= bid_best，可立即成交
        2. 如果会crossing，按对手方从最优档开始逐档吃掉流动性
        3. 对于IOC订单：吃完能吃的就结束，剩余直接取消
        4. 对于非IOC订单：如果还有剩余，按FIFO坐标轴模型挂到本方队列
        5. 如果不crossing，直接走现有的队列逻辑
        
        Args:
            order: The arriving order
            arrival_time: Time of arrival (exchtime)
            market_qty: Current market queue depth at price (from snapshot)
            
        Returns:
            Optional receipt for immediate action (None if order accepted to queue without immediate fill)
        """
        logger.debug(
            f"[Exchange] Order arrival: order_id={order.order_id}, "
            f"side={order.side.value}, price={order.price}, qty={order.qty}, "
            f"arrival_time={arrival_time}, market_qty={market_qty}"
        )
        
        side = order.side
        price = float(order.price)
        remaining_qty = order.remaining_qty
        
        # Find current segment
        seg_idx = self._find_segment(arrival_time)
        
        # Get opposite side best price from current segment
        opposite_best = self._get_opposite_best_price(side, seg_idx)
        
        # Check for crossing (immediate execution condition)
        is_crossing = self._check_crossing(side, price, opposite_best)
        
        logger.debug(
            f"[Exchange] Order {order.order_id}: opposite_best={opposite_best}, "
            f"is_crossing={is_crossing}"
        )
        
        immediate_fill_qty = 0
        immediate_fill_price = 0.0
        crossed_prices: List[Tuple[Side, Price]] = []  # 记录被crossed的价格
        
        if is_crossing and remaining_qty > 0:
            # 新增检查：如果本方有优先级更高的未成交shadow订单，则不能crossing
            # BUY: 检查是否有价格更高的BUY订单（更高价买单优先匹配）
            # SELL: 检查是否有价格更低的SELL订单（更低价卖单优先匹配）
            has_blocking_shadow = self._has_active_shadow_blocking_crossing(side, price)
            
            if has_blocking_shadow:
                # 有优先级更高的shadow订单，不能crossing，直接入队
                is_crossing = False
                logger.debug(f"[Exchange] Order {order.order_id}: blocked by higher priority shadow order")
            else:
                queue_depth = self._get_total_queue_depth(side, price, arrival_time)
                
                if queue_depth > 0:
                    is_crossing = False
                    logger.debug(f"[Exchange] Order {order.order_id}: queue_depth={queue_depth} > 0, no crossing")
                else:
                    # No blocking shadow orders or same-side depth, can execute crossing
                    # Execute immediately against opposite side liquidity
                    immediate_fill_qty, immediate_fill_price, crossed_prices = self._execute_crossing(
                        side, price, remaining_qty, arrival_time, seg_idx
                    )
                    remaining_qty -= immediate_fill_qty
                    logger.debug(
                        f"[Exchange] Order {order.order_id}: crossing executed, "
                        f"immediate_fill_qty={immediate_fill_qty}, fill_price={immediate_fill_price}"
                    )
        
        # Handle based on TIF and remaining quantity
        if order.tif == TimeInForce.IOC:
            # IOC: Any remaining after immediate fill is canceled
            if immediate_fill_qty > 0:
                if remaining_qty == 0:
                    # Full immediate fill
                    receipt = OrderReceipt(
                        order_id=order.order_id,
                        receipt_type="FILL",
                        timestamp=arrival_time,
                        fill_qty=immediate_fill_qty,
                        fill_price=immediate_fill_price,
                        remaining_qty=0,
                    )
                    logger.debug(f"[Exchange] IOC Order {order.order_id}: FILL receipt generated")
                    return receipt
                else:
                    # Partial fill, cancel rest
                    receipt = OrderReceipt(
                        order_id=order.order_id,
                        receipt_type="PARTIAL",
                        timestamp=arrival_time,
                        fill_qty=immediate_fill_qty,
                        fill_price=immediate_fill_price,
                        remaining_qty=0,  # Canceled, so remaining is 0
                    )
                    logger.debug(
                        f"[Exchange] IOC Order {order.order_id}: PARTIAL fill, "
                        f"remaining_qty=0 (rest canceled)"
                    )
                    return receipt
            else:
                # No immediate fill, cancel
                receipt = OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="CANCELED",
                    timestamp=arrival_time,
                    fill_qty=0,
                    fill_price=0.0,
                    remaining_qty=0,
                )
                logger.debug(f"[Exchange] IOC Order {order.order_id}: CANCELED (no fill)")
                return receipt
        
        # Non-IOC (GTC): Queue remaining if any
        if remaining_qty > 0:
            # Queue the remaining order using coordinate-axis model
            # Pass crossed_prices for diagnostics
            receipt = self._queue_order(
                order, arrival_time, market_qty, remaining_qty, immediate_fill_qty, crossed_prices
            )
            logger.debug(f"[Exchange] Order {order.order_id}: queued with remaining_qty={remaining_qty}")
            
            # If there was an immediate fill, return that receipt
            if immediate_fill_qty > 0:
                return OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="PARTIAL",
                    timestamp=arrival_time,
                    fill_qty=immediate_fill_qty,
                    fill_price=immediate_fill_price,
                    remaining_qty=remaining_qty,
                )
            return receipt
        else:
            # Fully filled immediately
            if immediate_fill_qty > 0:
                # Track this order as fully filled for cancel handling
                self._filled_order_ids.add(order.order_id)
                
                receipt = OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="FILL",
                    timestamp=arrival_time,
                    fill_qty=immediate_fill_qty,
                    fill_price=immediate_fill_price,
                    remaining_qty=0,
                )
                logger.debug(f"[Exchange] Order {order.order_id}: fully filled immediately")
                return receipt
        
        return None
    
    def _has_active_shadow_blocking_crossing(self, side: Side, price: Price) -> bool:
        """检查本方是否有阻止crossing的未成交shadow订单。
        
        用于crossing检查：
        - BUY订单在价格P: 检查是否有本方BUY订单在价格 > P（更高价的买单优先）
        - SELL订单在价格P: 检查是否有本方SELL订单在价格 < P（更低价的卖单优先）
        
        如果存在这样的订单，新订单不能crossing，因为市场会先匹配已有的更优订单。
        
        Args:
            side: 订单方向（本方方向）
            price: 订单价格
            
        Returns:
            True如果有阻止crossing的活跃shadow订单，否则False
        """
        price = round(float(price), 8)
        
        # 遍历所有本方的价格档位
        for (level_side, level_price), level in self._levels.items():
            if level_side != side:
                continue
            
            level_price = round(float(level_price), 8)
            
            # BUY: 检查价格 > P 的档位
            # SELL: 检查价格 < P 的档位
            should_check = False
            if side == Side.BUY and level_price > price:
                should_check = True
            elif side == Side.SELL and level_price < price:
                should_check = True
            
            if should_check:
                for shadow in level.queue:
                    if shadow.status == "ACTIVE" and shadow.remaining_qty > 0:
                        return True
        
        return False
    
    def _get_opposite_best_price(self, side: Side, seg_idx: int) -> Optional[Price]:
        """获取对手方最优价格。
        
        BUY订单的对手方是ask，SELL订单的对手方是bid。
        
        Args:
            side: 订单方向
            seg_idx: 当前段索引
            
        Returns:
            对手方最优价格，如果没有则返回None
        """
        if seg_idx < 0 or seg_idx >= len(self._current_tape):
            return None
        
        seg = self._current_tape[seg_idx]
        if side == Side.BUY:
            return seg.ask_price  # BUY看ask
        else:
            return seg.bid_price  # SELL看bid
    
    def _check_crossing(self, side: Side, order_price: Price, opposite_best: Optional[Price]) -> bool:
        """检查订单是否会crossing（可立即成交）。
        
        BUY订单：price >= ask_best 时crossing
        SELL订单：price <= bid_best 时crossing
        
        Args:
            side: 订单方向
            order_price: 订单限价
            opposite_best: 对手方最优价
            
        Returns:
            是否crossing
        """
        if opposite_best is None:
            return False
        
        # 使用精确比较，不需要EPSILON容差
        # BUY: price >= ask_best 时crossing
        # SELL: price <= bid_best 时crossing
        if side == Side.BUY:
            return order_price >= opposite_best
        else:
            return order_price <= opposite_best
    
    def _execute_crossing(
        self, 
        side: Side, 
        order_price: Price, 
        order_qty: Qty, 
        arrival_time: int,
        seg_idx: int
    ) -> Tuple[Qty, Price, List[Tuple[Side, Price]]]:
        """执行crossing（立即成交）。
        
        按对手方从最优档开始逐档吃掉流动性，直到：
        - 订单数量耗尽
        - 对手方档位耗尽
        - 触及订单限价边界
        
        Args:
            side: 订单方向
            order_price: 订单限价
            order_qty: 订单数量
            arrival_time: 到达时间
            seg_idx: 当前段索引
            
        Returns:
            (成交数量, 加权平均成交价, 被crossed的价格列表[(side, price)...])
        """
        if seg_idx < 0 or seg_idx >= len(self._current_tape):
            return 0, 0.0, []
        
        seg = self._current_tape[seg_idx]
        remaining = order_qty
        total_fill_qty = 0
        total_fill_value = 0.0
        crossed_prices: List[Tuple[Side, Price]] = []  # 记录被crossed的价格
        
        # 确定对手方档位（从activation set中按价格排序）
        opposite_side = Side.SELL if side == Side.BUY else Side.BUY
        
        if side == Side.BUY:
            # BUY吃ask，从最低ask开始
            opposite_activation = seg.activation_ask
            # 筛选出价格 <= order_price 的档位，按价格升序排列
            crossable_prices = sorted([p for p in opposite_activation if p <= order_price + EPSILON])
        else:
            # SELL吃bid，从最高bid开始
            opposite_activation = seg.activation_bid
            # 筛选出价格 >= order_price 的档位，按价格降序排列
            crossable_prices = sorted([p for p in opposite_activation if p >= order_price - EPSILON], reverse=True)
        
        # 逐档吃掉对手方流动性
        for cross_price in crossable_prices:
            if remaining <= 0:
                break
            
            # 获取该档位可用的流动性（使用Q_mkt）
            available_qty = self._get_q_mkt(opposite_side, cross_price, arrival_time)
            
            if available_qty <= 0:
                continue
            
            # 成交数量
            fill_qty = min(remaining, int(available_qty))
            if fill_qty > 0:
                total_fill_qty += fill_qty
                total_fill_value += fill_qty * cross_price
                remaining -= fill_qty
                
                # 记录被crossed的价格（对手方侧）
                crossed_prices.append((opposite_side, cross_price))
                
                # 更新对手方档位的X坐标（消耗了流动性）
                # 注：这里简化处理，实际可能需要更复杂的状态更新
                opposite_level = self._get_level(opposite_side, cross_price)
                opposite_level.x_coord += fill_qty
        
        # 计算加权平均价格
        avg_price = total_fill_value / total_fill_qty if total_fill_qty > 0 else 0.0
        
        return total_fill_qty, avg_price, crossed_prices
    
    def _queue_order(
        self, 
        order: Order, 
        arrival_time: int, 
        market_qty: Qty,
        remaining_qty: Qty,
        already_filled: Qty,
        crossed_prices: Optional[List[Tuple[Side, Price]]] = None
    ) -> Optional[OrderReceipt]:
        """将订单（剩余部分）排队到本方队列。
        
        使用坐标轴FIFO模型初始化队列位置。
        
        队列深度计算说明：
        - market_qty是快照在区间起点T_A时的队列深度（基础值）
        - 根据arrival_time所在segment的进度，结合净增量和交易量，计算当前队列深度
        - 例如：arrival_time=4，位于segment[2,5]中，进度=(4-2)/(5-2)=2/3
        - 当前队列深度 = 基础值 + Σ(net_flow - trades) * segment_progress
        
        Args:
            order: 原始订单
            arrival_time: 到达时间
            market_qty: 区间起点T_A时的市场队列深度（作为插值计算的基础值）
            remaining_qty: 需要排队的剩余数量
            already_filled: 已经立即成交的数量
            crossed_prices: 被crossed的价格列表（用于诊断）
            
        Returns:
            可选的回执（通常为None，表示已入队）
        """
        side = order.side
        price = float(order.price)
        seg_idx = self._find_segment(arrival_time)
        
        # Check if in activation window
        if seg_idx >= 0 and not self._is_in_activation_window(side, price, seg_idx):
            pass  # Still queue but won't have progress until activated
        
        level = self._get_level(side, price)
        
        # Calculate position based on whether crossing occurred
        if already_filled > 0:
            # 订单发生了crossing（吃掉了对手方流动性）
            # 剩余部分应该在队首位置
            # 
            # 修复：pos 应该等于到达时刻的 x_coord，而不是 0
            # 原因：如果 x_coord 已经推进到比如 100，而 pos=0，
            #       那么后续订单的 pos 会是 0+remaining_qty，
            #       这会导致它们在下一轮 advance 中被立即成交（因为 x > pos+qty）
            # 
            # 正确做法：pos = x_coord(arrival_time)，表示队首在坐标轴上的位置
            # 这样后续订单的 pos 会正确地排在当前 x_coord 之后
            # 
            # 注：由于 post-crossing 订单在本方队列没有前序 shadow 订单，
            #     shadow_pos 参数使用 0（对 X 计算无影响，因为队列为空时不涉及位置相关撤单）
            pos = int(round(self._get_x_coord(side, price, arrival_time, 0)))
        else:
            # 没有crossing，计算新订单在队列中的位置
            # 
            # FIFO保序修复：
            # 之前的算法：pos = q_mkt(t) + shadow_qty
            # 问题：当队列收缩时（trades > netflow），后到达的订单可能有更小的threshold，
            #       导致后到达的订单先成交，违反FIFO原则
            # 
            # 新算法：
            # - 第一个订单：pos = q_mkt(t)
            # - 后续订单：pos = 前一个shadow订单的(pos + qty) + 正净流入增量
            #   - 正净流入增量 = 从前一个订单到达到当前订单到达期间的正netflow累计
            #   - 负netflow不增加距离（队列收缩不会让后续订单超过前面的订单）
            # 
            # 这保证了FIFO：每个订单的threshold = pos + qty >= 前一个订单的threshold
            # 
            # Initialize Q_mkt with base value at interval start T_A
            # This serves as the starting point for interpolation
            if not level.queue and level.q_mkt == 0:
                level.q_mkt = float(market_qty)
            
            # 找到该价位上最后一个活跃的shadow订单
            last_active_shadow = None
            for shadow_order in reversed(level.queue):
                if shadow_order.status == "ACTIVE" and shadow_order.arrival_time < arrival_time:
                    last_active_shadow = shadow_order
                    break
            
            if last_active_shadow is not None:
                # 有前序shadow订单，基于前序订单计算位置
                # pos = 前序订单的threshold + 期间的正净流入
                prev_threshold = last_active_shadow.pos + last_active_shadow.original_qty
                
                # 计算从前序订单到当前订单期间的正净流入
                positive_netflow = self._get_positive_netflow_between(
                    side, price, last_active_shadow.arrival_time, arrival_time
                )
                
                # 新订单位置 = 前序订单的threshold + 正净流入（只有队列增长才增加距离）
                pos = int(round(prev_threshold + positive_netflow))
            else:
                # 没有前序shadow订单，使用原始逻辑
                # 根据arrival_time所在segment的进度计算当前队列深度
                # q_mkt_t = 基础值 + Σ(net_flow - trades) * segment_progress
                q_mkt_t = self._get_q_mkt(side, price, arrival_time)  # 插值计算的当前队列深度
                
                # 新订单位置 = 当前队列深度
                # 注意：不包含X坐标，X坐标只用于成交推进计算
                # 手数必须是整数，所以需要取整
                pos = int(round(q_mkt_t))
        
        # Create shadow order with remaining qty
        # Mark as post-crossing if there was an immediate fill (crossing occurred)
        shadow = ShadowOrder(
            order_id=order.order_id,
            side=side,
            price=price,
            original_qty=remaining_qty,  # Only the remaining part
            remaining_qty=remaining_qty,
            arrival_time=arrival_time,
            pos=pos,
            tif=order.tif,
            filled_qty=0,  # Already filled part is tracked separately
            is_post_crossing=(already_filled > 0),  # True if this is remainder after crossing
            crossed_prices=crossed_prices or [],  # Store crossed prices for diagnostics
        )
        
        level.queue.append(shadow)
        level._active_shadow_qty += remaining_qty
        
        return None
    
    def on_cancel_arrival(self, order_id: str, arrival_time: int) -> OrderReceipt:
        """Handle cancel request.
        
        Args:
            order_id: ID of order to cancel
            arrival_time: Time of cancel arrival (exchtime)
            
        Returns:
            Receipt for the cancel operation
            
        Raises:
            ValueError: If the order_id is not found in price levels
                and was not previously fully filled immediately.
                This indicates a bug in order management or an invalid cancel request.
        """
        logger.debug(f"[Exchange] Cancel arrival: order_id={order_id}, arrival_time={arrival_time}")
        
        # Check if order was fully filled immediately (crossing order)
        if order_id in self._filled_order_ids:
            logger.debug(f"[Exchange] Cancel {order_id}: REJECTED (already filled immediately)")
            # Remove from set after handling to prevent unbounded growth
            self._filled_order_ids.discard(order_id)
            return OrderReceipt(
                order_id=order_id,
                receipt_type="REJECTED",
                timestamp=arrival_time,
            )
        
        # Look up order in price levels
        shadow = self._find_order_by_id(order_id)
        
        if shadow is None:
            logger.error(f"[Exchange] Cancel {order_id}: order not found")
            raise ValueError(f"Order not found: {order_id}")
        
        if shadow.status == "FILLED":
            logger.debug(f"[Exchange] Cancel {order_id}: REJECTED (already filled)")
            return OrderReceipt(
                order_id=order_id,
                receipt_type="REJECTED",
                timestamp=arrival_time,
            )
        
        if shadow.status == "CANCELED":
            logger.debug(f"[Exchange] Cancel {order_id}: REJECTED (already canceled)")
            return OrderReceipt(
                order_id=order_id,
                receipt_type="REJECTED",
                timestamp=arrival_time,
            )
        
        # Calculate fill up to cancel time
        # Use shadow.filled_qty which represents fills that have already been
        # reported via PARTIAL receipts, rather than recalculating from x_t - shadow.pos
        # This ensures consistency: if an order has partial fills, they should have been
        # reported via PARTIAL receipts during advance(). The cancel receipt should use
        # the same filled_qty that was accumulated from those PARTIAL receipts.
        fill_at_cancel = shadow.filled_qty
        
        # Save remaining_qty before updating for cache update
        old_remaining_qty = shadow.remaining_qty
        
        # Update shadow order status (filled_qty already contains the correct value)
        shadow.remaining_qty = 0
        shadow.status = "CANCELED"
        
        # Update level cache
        level_key = (shadow.side, round(float(shadow.price), 8))
        if level_key in self._levels:
            level = self._levels[level_key]
            if shadow in level.queue:
                level._active_shadow_qty -= old_remaining_qty
        
        logger.debug(
            f"[Exchange] Cancel {order_id}: CANCELED successfully, "
            f"fill_at_cancel={fill_at_cancel}"
        )
        return OrderReceipt(
            order_id=order_id,
            receipt_type="CANCELED",
            timestamp=arrival_time,
            fill_qty=fill_at_cancel,
            remaining_qty=0,
        )
    
    def _find_segment(self, t: int) -> int:
        """Find segment index containing time t."""
        for i, seg in enumerate(self._current_tape):
            if seg.t_start <= t < seg.t_end:
                return i
        return -1

    def _get_best_active_price(self, side: Side, t_from: int) -> Optional[Price]:
        """Get best active order price for a side."""
        best_price: Optional[Price] = None
        for (level_side, level_price), level in self._levels.items():
            if level_side != side:
                continue
            for shadow in level.queue:
                if shadow.status == "ACTIVE" and shadow.remaining_qty > 0 and shadow.arrival_time <= t_from:
                    price = level.price
                    if best_price is None:
                        best_price = price
                    elif side == Side.BUY and price > best_price:
                        best_price = price
                    elif side == Side.SELL and price < best_price:
                        best_price = price
                    break
        return best_price

    def _get_first_active_shadow(self, side: Side, price: Price, t_from: int) -> Optional[ShadowOrder]:
        """Get FIFO-first active shadow order at a price."""
        level = self._get_level(side, price)
        for shadow in level.queue:
            if shadow.status == "ACTIVE" and shadow.remaining_qty > 0 and shadow.arrival_time <= t_from:
                return shadow
        return None

    def _compute_full_fill_time_at_best(
        self,
        shadow: ShadowOrder,
        side: Side,
        price: Price,
        seg_idx: int,
        t_from: int,
        t_to: int,
        segment: TapeSegment,
    ) -> Optional[int]:
        """Compute full fill time within [t_from, t_to] using zone-aware model."""
        if t_to <= t_from or seg_idx < 0:
            return None

        if not self._is_in_activation_window(side, price, seg_idx):
            return None

        seg_duration = segment.t_end - segment.t_start
        if seg_duration <= 0:
            raise ValueError(
                f"Invalid segment duration in _compute_full_fill_time_at_best: seg_duration={seg_duration}"
            )

        threshold = shadow.pos + shadow.original_qty
        x_start = self._get_x_coord(side, price, t_from, shadow.pos)
        if x_start >= threshold - EPSILON:
            return t_from

        # Segment parameters
        total_cancels = segment.cancels.get((side, price), 0)
        total_trades = segment.trades.get((side, price), 0)
        q_mkt = self._get_q_mkt(side, price, segment.t_start)
        trade_rate = total_trades / seg_duration if seg_duration > 0 else 0.0

        if trade_rate <= EPSILON and total_cancels <= EPSILON:
            return None

        # Build zones from x_start & distribute cancels
        zones = self._build_queue_zones(side, price, x_start, q_mkt)
        cancel_dist = self._distribute_cancels_to_zones(zones, total_cancels, q_mkt)

        t_abs_start = t_from - segment.t_start
        dt_available = t_to - t_from

        x_end, time_elapsed = self._traverse_zones_for_x(
            x_start, zones, cancel_dist, trade_rate,
            seg_duration, t_abs_start, dt_available,
            target_x=threshold,
        )

        if x_end >= threshold - EPSILON and time_elapsed <= dt_available + EPSILON:
            fill_time = int(t_from + time_elapsed)
            fill_time = max(fill_time, t_from)
            if fill_time > t_to:
                return None
            return fill_time

        return None
    
    def _compute_rate_for_segment(
        self, side: Side, price: Price, seg_idx: int, x_running: float, shadow_pos: int
    ) -> float:
        """Compute X rate for a segment using position-dependent cancel probability.
        
        Args:
            side: Order side
            price: Price level
            seg_idx: Segment index
            x_running: Current X coordinate at segment start
            shadow_pos: Position of the shadow order
            
        Returns:
            Rate (X advancement per time unit)
        """
        seg = self._current_tape[seg_idx]
        seg_duration = seg.t_end - seg.t_start
        if seg_duration <= 0:
            return 0.0

        trade_rate = self._compute_trade_rate_for_segment(side, price, seg_idx)
        cancel_rate = self._compute_cancel_rate_for_segment(side, price, seg_idx, x_running, shadow_pos)
        return trade_rate + cancel_rate
    
    def advance(self, t_from: int, t_to: int, segment: TapeSegment) -> Tuple[List[OrderReceipt], int]:
        """Advance simulation from t_from to t_to using tape segment.
        
        Returns the earliest fill receipts (if any) and the stop time.
        """
        if t_to <= t_from:
            return [], t_to

        seg_idx = self._find_segment(t_from)
        if seg_idx < 0:
            try:
                seg_idx = self._current_tape.index(segment)
            except ValueError:
                seg_idx = -1

        seg_duration = segment.t_end - segment.t_start
        if seg_duration <= 0:
            raise ValueError(
                f"Invalid segment duration in advance: seg_duration={seg_duration}"
            )

        side_state: Dict[Side, Dict[str, object]] = {}
        full_fill_candidates: List[Tuple[int, Side]] = []

        for side in (Side.BUY, Side.SELL):
            mkt_best = segment.bid_price if side == Side.BUY else segment.ask_price
            best_active = self._get_best_active_price(side, t_from)
            if best_active is None:
                side_state[side] = {}
                continue

            if side == Side.BUY:
                exec_best = max(mkt_best, best_active)
                improvement = exec_best > mkt_best + EPSILON
            else:
                exec_best = min(mkt_best, best_active)
                improvement = exec_best < mkt_best - EPSILON

            shadow = self._get_first_active_shadow(side, exec_best, t_from)
            if shadow is None:
                side_state[side] = {}
                continue

            eligible = True
            if not improvement:
                if seg_idx < 0 or not self._is_in_activation_window(side, exec_best, seg_idx):
                    eligible = False

            side_state[side] = {
                "mkt_best": mkt_best,
                "exec_best": exec_best,
                "improvement": improvement,
                "shadow": shadow,
                "eligible": eligible,
            }

            if not eligible:
                continue

            if improvement:
                trade_qty = segment.trades.get((side, mkt_best), 0)
                if trade_qty > 0:
                    trade_rate = trade_qty / seg_duration
                    if trade_rate > EPSILON:
                        t_fill = int(t_from + (shadow.remaining_qty / trade_rate))
                        t_fill = max(t_fill, t_from)
                        if t_fill <= t_to:
                            full_fill_candidates.append((t_fill, side))
            else:
                t_fill = self._compute_full_fill_time_at_best(
                    shadow, side, exec_best, seg_idx, t_from, t_to, segment
                )
                if t_fill is not None and t_fill <= t_to:
                    full_fill_candidates.append((t_fill, side))

        receipts: List[OrderReceipt] = []

        if full_fill_candidates:
            t_stop = min(t_fill for t_fill, _ in full_fill_candidates)

            for side, state in side_state.items():
                if state.get("improvement"):
                    self._add_trade_pause_interval(side, t_from, t_stop)

            for t_fill, side in full_fill_candidates:
                if t_fill != t_stop:
                    continue
                state = side_state.get(side, {})
                shadow = state.get("shadow")
                if shadow is None:
                    continue
                fill_qty = shadow.remaining_qty
                if fill_qty <= 0:
                    continue
                receipt = self._apply_shadow_fill(shadow, fill_qty, t_stop)
                logger.debug(
                    f"[Exchange] Advance: FILL for {shadow.order_id}, "
                    f"fill_qty={fill_qty}, price={shadow.price}, time={t_stop}"
                )
                receipts.append(receipt)

            self.current_time = t_stop
            return receipts, t_stop

        t_stop = t_to

        for side, state in side_state.items():
            if state.get("improvement"):
                self._add_trade_pause_interval(side, t_from, t_stop)

        for side, state in side_state.items():
            if not state or not state.get("eligible"):
                continue

            shadow: ShadowOrder = state["shadow"]
            if shadow.remaining_qty <= 0:
                continue

            if state.get("improvement"):
                trade_qty = segment.trades.get((side, state["mkt_best"]), 0)
                if trade_qty <= 0:
                    continue
                trade_rate = trade_qty / seg_duration
                if trade_rate <= EPSILON:
                    continue
                virtual_volume = trade_rate * (t_stop - t_from)
                fill_qty = min(shadow.remaining_qty, int(virtual_volume))
                if fill_qty <= 0:
                    continue
                receipt = self._apply_shadow_fill(shadow, fill_qty, t_stop)
                if receipt.receipt_type == "PARTIAL":
                    logger.debug(
                        f"[Exchange] Advance: improvement PARTIAL fill for {shadow.order_id}, "
                        f"fill_qty={fill_qty}, remaining_qty={receipt.remaining_qty}, "
                        f"price={shadow.price}, time={t_stop}"
                    )
                else:
                    logger.debug(
                        f"[Exchange] Advance: improvement FILL for {shadow.order_id}, "
                        f"fill_qty={fill_qty}, price={shadow.price}, time={t_stop}"
                    )
                receipts.append(receipt)
            else:
                exec_best = state["exec_best"]
                x_t_stop = self._get_x_coord(side, exec_best, t_stop, shadow.pos)
                current_fill = int(x_t_stop - shadow.pos)
                if current_fill > shadow.filled_qty:
                    new_fill = current_fill - shadow.filled_qty
                    if new_fill <= 0:
                        continue
                    if not self._validate_fill_delta(
                        shadow.order_id,
                        new_fill,
                        shadow.filled_qty,
                        shadow.original_qty,
                    ):
                        continue
                    if new_fill > shadow.remaining_qty:
                        new_fill = shadow.remaining_qty
                    receipt = self._apply_shadow_fill(shadow, new_fill, t_stop)
                    if receipt.receipt_type == "PARTIAL":
                        logger.debug(
                            f"[Exchange] Advance: PARTIAL fill for {shadow.order_id}, "
                            f"fill_qty={new_fill}, remaining_qty={receipt.remaining_qty}, "
                            f"price={shadow.price}, time={t_stop}"
                        )
                    else:
                        logger.debug(
                            f"[Exchange] Advance: FILL for {shadow.order_id}, "
                            f"fill_qty={new_fill}, price={shadow.price}, time={t_stop}"
                        )
                    receipts.append(receipt)

        self.current_time = t_stop
        return receipts, t_stop
    
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
            # Find first active shadow for position-dependent X calculation
            first_active_shadow_pos = None
            for shadow in level.queue:
                if shadow.status == "ACTIVE":
                    first_active_shadow_pos = shadow.pos
                    break
            
            if first_active_shadow_pos is not None:
                current_x = self._get_x_coord(side, price, self.current_time, first_active_shadow_pos)
            else:
                current_x = 0.0  # No shadow orders, X is irrelevant
            
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
        """Get current X coordinate for diagnostics.
        
        Uses the first active shadow order's position for calculation.
        Returns 0.0 if no active shadow orders exist.
        """
        level = self._get_level(side, price)
        first_active_shadow_pos = None
        for shadow in level.queue:
            if shadow.status == "ACTIVE":
                first_active_shadow_pos = shadow.pos
                break
        
        if first_active_shadow_pos is not None:
            return self._get_x_coord(side, price, self.current_time, first_active_shadow_pos)
        return 0.0
    
    def get_shadow_orders(self) -> List[ShadowOrder]:
        """Get all shadow orders for diagnostics."""
        result = []
        for level in self._levels.values():
            result.extend(level.queue)
        return result
