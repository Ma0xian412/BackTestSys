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


class OrderNotFoundError(Exception):
    """Exception raised when attempting to cancel an order that doesn't exist.
    
    This exception is raised during cancel operations when the order_id
    cannot be found in the exchange's order registry. This typically indicates
    a bug in order management or an invalid cancel request.
    """
    def __init__(self, order_id: str, message: str = None):
        self.order_id = order_id
        if message is None:
            message = f"Order not found: {order_id}"
        super().__init__(message)


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
    - Post-crossing orders are filled based on the SUM of net increments N
      across all crossed price levels:
      - If N >= 0: fill min(remaining_qty, N) over the segment
      - If N < 0: no fill
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
        
        # Order registry: maps order_id to ShadowOrder
        # This persists across reset() to allow order lookup for cancellation
        # even after interval boundaries when levels are cleared
        self._orders: Dict[str, ShadowOrder] = {}

    def _validate_fill_delta(self, order_id: str, delta: int, filled_qty: int, original_qty: int) -> bool:
        """Validate fill delta to avoid negative fill quantities."""
        if delta < 0:
            logger.warning(
                f"[Exchange] Advance: skip negative fill delta for {order_id}, "
                f"filled_qty={filled_qty}, original_qty={original_qty}"
            )
            return False
        return True
    
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
        """
        # Reset interval-specific state
        self.current_time = 0
        self._current_tape = []
        self._current_seg_idx = 0
        self._x_rates.clear()
        self._x_at_seg_start.clear()
        
        # Reset X coordinate for all levels (already done in align_at_boundary,
        # but do it here too for safety when reset is called standalone)
        for level in self._levels.values():
            level.x_coord = 0.0
        
        # Note: _levels is intentionally NOT cleared to preserve shadow orders
        # across interval boundaries. Their pos values were adjusted by
        # align_at_boundary() at the end of the previous interval.
        # Note: _orders is also preserved for order lookup (e.g., cancellation)
    
    def full_reset(self) -> None:
        """Fully reset simulator state including levels and order registry.
        
        Call this when starting a new backtest session to clear all state
        including the order registry.
        """
        self._levels.clear()
        self._orders.clear()
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
                continue
            
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
                continue
            
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
                # New check: if same-side queue still has depth, cannot execute immediately
                self._ensure_base_q_mkt(side, price, market_qty)
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
                    logger.debug(f"[Exchange] IOC Order {order.order_id}: PARTIAL receipt (rest canceled)")
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
            # Pass crossed_prices for post-crossing fill calculation
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
            crossed_prices: 被crossed的价格列表（用于post-crossing fill计算）
            
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
            # 剩余部分应该在队首，position为0
            # 逻辑：如果我的订单在px上发生crossing，说明ask方在≤px有流动性
            # 如果ask@px有流动性，那么bid@px不可能有订单（否则早就撮合了）
            # 因此bid@px上不可能有之前的shadow订单，position直接为0
            pos = 0
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
            crossed_prices=crossed_prices or [],  # Store crossed prices for post-crossing fill
        )
        
        level.queue.append(shadow)
        level._active_shadow_qty += remaining_qty
        
        # Register order in the order registry for cross-interval lookup
        self._orders[order.order_id] = shadow
        
        return None
    
    def on_cancel_arrival(self, order_id: str, arrival_time: int) -> OrderReceipt:
        """Handle cancel request.
        
        Args:
            order_id: ID of order to cancel
            arrival_time: Time of cancel arrival (exchtime)
            
        Returns:
            Receipt for the cancel operation
            
        Raises:
            OrderNotFoundError: If the order_id is not found in the order registry.
                This indicates a bug in order management or an invalid cancel request.
        """
        logger.debug(f"[Exchange] Cancel arrival: order_id={order_id}, arrival_time={arrival_time}")
        
        # Look up order in the order registry (persists across reset())
        shadow = self._orders.get(order_id)
        
        if shadow is None:
            logger.error(f"[Exchange] Cancel {order_id}: order not found in registry")
            raise OrderNotFoundError(order_id)
        
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
        x_t = self._get_x_coord(shadow.side, shadow.price, arrival_time)
        fill_at_cancel = max(0, min(shadow.original_qty, int(x_t - shadow.pos)))
        
        # Save remaining_qty before updating for cache update
        old_remaining_qty = shadow.remaining_qty
        
        # Update shadow order status
        shadow.filled_qty = fill_at_cancel
        shadow.remaining_qty = 0
        shadow.status = "CANCELED"
        
        # Try to update level cache if level still exists
        level_key = (shadow.side, round(float(shadow.price), 8))
        if level_key in self._levels:
            level = self._levels[level_key]
            # Find shadow in level queue and update cache
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
    
    def _compute_post_crossing_fill(
        self, 
        shadow: ShadowOrder, 
        seg: TapeSegment, 
        t_from: int, 
        t_to: int
    ) -> Tuple[Qty, Optional[int]]:
        """计算post-crossing订单的成交量和成交时间。
        
        Post-crossing订单的成交逻辑：
        - 获取所有被crossed的对手方价位的净增量之和N
        - 如果N >= 0：成交量 = min(剩余数量, N)
        - 如果N < 0：不成交
        - 成交时间是消耗完剩余数量或消耗完N的时刻
        
        Args:
            shadow: post-crossing的shadow订单
            seg: 当前segment
            t_from: segment开始时间
            t_to: segment结束时间
            
        Returns:
            (成交数量, 成交时间)，如果不成交则返回(0, None)
        """
        # 获取聚合净增量N（所有crossed价位的净增量之和）
        n = self._compute_aggregated_net_increment(shadow, seg)
        
        # 如果N < 0，不成交
        if n < 0:
            return 0, None
        
        # 成交量 = min(剩余数量, N)
        fill_qty = min(shadow.remaining_qty, n)
        
        if fill_qty <= 0:
            return 0, None
        
        # 计算成交时间
        # 假设净增量在segment内均匀分布
        # 成交时间是消耗完fill_qty的时刻
        seg_duration = t_to - t_from
        if seg_duration <= 0:
            return fill_qty, t_to
        
        # 如果remaining_qty <= N，则成交时间是消耗完remaining_qty的时刻
        # 如果remaining_qty > N，则成交时间是消耗完N的时刻（即segment结束）
        if shadow.remaining_qty <= n:
            # 消耗完remaining_qty的时刻
            # 进度 = remaining_qty / N
            progress = shadow.remaining_qty / n if n > 0 else 1.0
            fill_time = int(t_from + seg_duration * progress)
        else:
            # 消耗完N的时刻（即segment结束）
            fill_time = t_to
        
        # 确保fill_time在有效范围内
        fill_time = max(t_from, min(t_to, fill_time))
        
        # 确保fill_time晚于订单到达时间
        fill_time = max(fill_time, shadow.arrival_time)
        
        return fill_qty, fill_time
    
    def _compute_aggregated_net_increment(
        self,
        shadow: ShadowOrder,
        seg: TapeSegment
    ) -> int:
        """计算所有crossed价位的聚合净增量N。
        
        对于post-crossing订单，N应该是所有被crossed的对手方价位的净增量之和。
        
        Args:
            shadow: post-crossing的shadow订单
            seg: 当前segment
            
        Returns:
            聚合净增量N（所有crossed价位的净增量之和）
        """
        if not shadow.crossed_prices:
            # 如果没有记录crossed_prices，回退到单价位逻辑
            opposite_side = Side.SELL if shadow.side == Side.BUY else Side.BUY
            return seg.net_flow.get((opposite_side, shadow.price), 0)
        
        # 汇总所有crossed价位的净增量
        total_n = 0
        for crossed_side, crossed_price in shadow.crossed_prices:
            n = seg.net_flow.get((crossed_side, crossed_price), 0)
            total_n += n
        
        return total_n
    
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
            
            # Check activation for this level
            # Note: Post-crossing orders may need processing even if not in activation window
            in_activation = seg_idx < 0 or self._is_in_activation_window(side, price, seg_idx)
            
            # Get X at t_to (only if in activation)
            x_t_to = self._get_x_coord(side, price, t_to) if in_activation else 0
            
            # Check each shadow order
            for shadow in level.queue:
                if shadow.status != "ACTIVE":
                    continue
                if shadow.arrival_time > t_to:
                    continue
                
                # Handle post-crossing orders differently
                # Post-crossing orders are filled based on opposite-side net increment
                # They don't require the price to be in activation window
                if shadow.is_post_crossing:
                    fill_qty, fill_time = self._compute_post_crossing_fill(
                        shadow, segment, t_from, t_to
                    )
                    
                    if fill_qty > 0 and fill_time is not None:
                        # Update cache before changing
                        level._active_shadow_qty -= fill_qty
                        
                        shadow.filled_qty += fill_qty
                        shadow.remaining_qty -= fill_qty
                        
                        if shadow.remaining_qty <= 0:
                            shadow.status = "FILLED"
                            receipt = OrderReceipt(
                                order_id=shadow.order_id,
                                receipt_type="FILL",
                                timestamp=fill_time,
                                fill_qty=fill_qty,
                                fill_price=shadow.price,
                                remaining_qty=0,
                            )
                            logger.debug(
                                f"[Exchange] Advance: post-crossing FILL for {shadow.order_id}, "
                                f"fill_qty={fill_qty}, price={shadow.price}, time={fill_time}"
                            )
                            receipts.append(receipt)
                        else:
                            receipt = OrderReceipt(
                                order_id=shadow.order_id,
                                receipt_type="PARTIAL",
                                timestamp=fill_time,
                                fill_qty=fill_qty,
                                fill_price=shadow.price,
                                remaining_qty=shadow.remaining_qty,
                            )
                            logger.debug(
                                f"[Exchange] Advance: post-crossing PARTIAL for {shadow.order_id}, "
                                f"fill_qty={fill_qty}, remaining={shadow.remaining_qty}"
                            )
                            receipts.append(receipt)
                    continue
                
                # Normal fill logic for non-post-crossing orders
                # Skip if not in activation window
                if not in_activation:
                    continue
                
                # Fill threshold
                threshold = shadow.pos + shadow.original_qty
                
                # Check if threshold is crossed
                if x_t_to >= threshold:
                    # Full fill - compute exact fill time
                    fill_time = self._compute_fill_time(shadow, shadow.original_qty)
                    
                    if fill_time is not None and t_from < fill_time <= t_to:
                        # Update cache before changing
                        level._active_shadow_qty -= shadow.remaining_qty
                        
                        remaining_to_fill = shadow.original_qty - shadow.filled_qty
                        if remaining_to_fill <= 0:
                            continue
                        if not self._validate_fill_delta(
                            shadow.order_id,
                            remaining_to_fill,
                            shadow.filled_qty,
                            shadow.original_qty,
                        ):
                            continue
                        
                        shadow.filled_qty = shadow.original_qty
                        shadow.remaining_qty = 0
                        shadow.status = "FILLED"
                        
                        # Emit only the remaining delta to avoid double-counting in multi-partial fills
                        receipt = OrderReceipt(
                            order_id=shadow.order_id,
                            receipt_type="FILL",
                            timestamp=fill_time,
                            fill_qty=remaining_to_fill,
                            fill_price=shadow.price,
                            remaining_qty=0,
                        )
                        logger.debug(
                            f"[Exchange] Advance: FILL for {shadow.order_id}, "
                            f"fill_qty={remaining_to_fill}, price={shadow.price}, time={fill_time}"
                        )
                        receipts.append(receipt)
                elif x_t_to > shadow.pos:
                    # Partial fill
                    current_fill = int(x_t_to - shadow.pos)
                    if current_fill > shadow.filled_qty:
                        new_fill = current_fill - shadow.filled_qty
                        
                        # Validate fill delta before checking completion
                        if new_fill <= 0:
                            continue
                        if not self._validate_fill_delta(
                            shadow.order_id,
                            new_fill,
                            shadow.filled_qty,
                            shadow.original_qty,
                        ):
                            continue
                        
                        # If this fill completes the order, emit a FILL receipt
                        # Cap final fill to remaining qty if interpolation overshoots remaining depth
                        completes_order = new_fill >= shadow.remaining_qty
                        if completes_order:
                            new_fill = shadow.remaining_qty
                        
                        # Update cache for the qty change
                        level._active_shadow_qty -= new_fill
                        
                        if completes_order:
                            shadow.filled_qty += new_fill
                            shadow.remaining_qty = 0
                            shadow.status = "FILLED"
                            
                            receipt = OrderReceipt(
                                order_id=shadow.order_id,
                                receipt_type="FILL",
                                timestamp=t_to,
                                fill_qty=new_fill,
                                fill_price=shadow.price,
                                remaining_qty=0,
                            )
                            logger.debug(
                                f"[Exchange] Advance: FILL for {shadow.order_id}, "
                                f"fill_qty={new_fill}, price={shadow.price}, time={t_to}"
                            )
                            receipts.append(receipt)
                            continue
                        
                        shadow.filled_qty = current_fill
                        shadow.remaining_qty = shadow.original_qty - current_fill
                        
                        receipt = OrderReceipt(
                            order_id=shadow.order_id,
                            receipt_type="PARTIAL",
                            timestamp=t_to,
                            fill_qty=new_fill,
                            fill_price=shadow.price,
                            remaining_qty=shadow.remaining_qty,
                        )
                        logger.debug(
                            f"[Exchange] Advance: PARTIAL for {shadow.order_id}, "
                            f"fill_qty={new_fill}, remaining={shadow.remaining_qty}"
                        )
                        receipts.append(receipt)
        
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
