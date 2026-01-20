"""Unified Tape-Based Simulation Model.

This module implements the unified framework for backtest simulation as specified
in the problem statement. The model constructs an "event tape" from Prev/Next
snapshots (A/B) and lastvolsplit data, then simulates exchange matching with:

1. **Market consistency** - Total volume conservation between snapshots
2. **No-impact assumption** - Your orders don't affect real market data
3. **FIFO queue management** - Unified queue for shadow orders

Key Components:
- Event Tape Construction: Segments from A/B + lastvolsplit
- Optimal Price Path: Minimal displacement bid/ask paths
- Segment Duration Iteration: 2-round volume-weighted duration computation
- Exchange Simulator: No-impact FIFO matching with ahead tracking
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from ...core.events import EventType, SimulationEvent
from ...core.interfaces import ISimulationModel
from ...core.types import Level, NormalizedSnapshot, Order, Price, Qty, Side


# =============================================================================
# Constants
# =============================================================================

# Numerical tolerance for floating point comparisons
EPSILON = 1e-12

# Minimum argument value for log function to avoid domain errors
MIN_LOG_ARG = 1e-12


# =============================================================================
# Configuration Parameters (Section 1.3)
# =============================================================================


@dataclass
class UnifiedTapeConfig:
    """Configuration parameters for the unified tape model.
    
    These parameters must be explicitly set with defaults as per specification.
    """
    
    # --- A. lastvolsplit -> single-side mapping ("ghost state" rule) ---
    # Options: "symmetric", "proportion", "single_bid", "single_ask"
    ghost_rule: str = "symmetric"
    # For "proportion" rule: E_bid(p) = alpha * E(p), E_ask(p) = (1-alpha) * E(p)
    ghost_alpha: float = 0.5
    
    # --- B. Segment duration iteration parameters ---
    # epsilon > 0: "no-trade baseline weight" for segment duration
    epsilon: float = 1.0
    # Number of iterations (fixed at 2 as recommended)
    segment_iterations: int = 2
    
    # --- C. Time scaling parameters (optional) ---
    # lambda: real progress u -> scaled progress u'
    # |lambda| < 1e-6 means no scaling (u' = u)
    time_scale_lambda: float = 0.0
    
    # --- D. Active window (top5 visibility) ---
    # Number of price levels to consider (default 5)
    active_levels: int = 5
    # Fallback strategy: "freeze", "expand", "low_confidence"
    invisible_fallback: str = "freeze"
    
    # --- E. Cancellation front-push ratio (no-information closure) ---
    # phi in [0,1]: proportion of cancellations occurring in queue front
    # phi=0 pessimistic, phi=1 optimistic, default 0.5
    cancel_front_ratio: float = 0.5
    
    # --- F. Network delays ---
    # Delay_out: strategy -> exchange (can be constant or callable)
    delay_out: int = 0
    # Delay_in: exchange -> strategy
    delay_in: int = 0
    
    # --- Crossing order handling ---
    # Options: "reject", "adjust", "passive"
    crossing_order_policy: str = "passive"
    
    # --- Price alignment ---
    # Options: "round", "reject"
    price_alignment_policy: str = "round"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TapeSegment:
    """A single segment in the event tape.
    
    Each segment has constant best bid/ask prices within the segment.
    """
    index: int  # Segment index (1-based)
    
    # Best prices for this segment
    bid_price: Price
    ask_price: Price
    
    # Segment boundaries in scaled progress space [0,1]
    u_prime_start: float
    u_prime_end: float
    
    # Segment boundaries in real time
    t_start: int
    t_end: int
    
    # Trade volumes for this segment (per-side, at best price)
    bid_trade_volume: Qty = 0  # M_{bid,i}
    ask_trade_volume: Qty = 0  # M_{ask,i}
    
    # Cancellation volumes for this segment (per-side, per-price)
    bid_cancels: Dict[Price, Qty] = field(default_factory=dict)  # C_{bid,i}(p)
    ask_cancels: Dict[Price, Qty] = field(default_factory=dict)  # C_{ask,i}(p)
    
    # Net flow (adds - cancels) for diagnostics
    bid_adds: Dict[Price, Qty] = field(default_factory=dict)
    ask_adds: Dict[Price, Qty] = field(default_factory=dict)


@dataclass
class ShadowOrder:
    """A shadow order in the exchange simulator's queue.
    
    These orders don't exist in the real market queue (no-impact assumption)
    but are tracked for FIFO fill determination.
    """
    order_id: str
    side: Side
    price: Price
    remaining_qty: Qty
    arrival_time: int
    status: str = "ACTIVE"  # ACTIVE, FILLED, CANCELED
    
    # Fill tracking
    filled_qty: Qty = 0
    fills: List[Tuple[int, Qty]] = field(default_factory=list)  # (time, qty)


@dataclass
class PriceLevelState:
    """State for a single price level in the exchange simulator.
    
    Maintains the "ahead" count and shadow order FIFO queue.
    """
    side: Side
    price: Price
    
    # Ahead: anonymous market quantity in front of our earliest shadow order
    ahead: int = 0
    
    # Shadow order FIFO queue
    queue: List[ShadowOrder] = field(default_factory=list)
    
    def total_external(self) -> int:
        """Total external (market) quantity at this level."""
        return max(0, self.ahead)


# =============================================================================
# Event Tape Construction (Section 4)
# =============================================================================


class EventTapeBuilder:
    """Builds the event tape from A/B snapshots and lastvolsplit.
    
    This is the "single source of truth" for all matching - the matching
    layer cannot change the tape's total volumes.
    """
    
    def __init__(self, config: UnifiedTapeConfig, tick_size: float = 1.0):
        self.config = config
        self.tick_size = tick_size
    
    def build(
        self,
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
    ) -> List[TapeSegment]:
        """Build the complete event tape for the interval [T_A, T_B].
        
        Args:
            prev: Previous snapshot (A) at time T_A
            curr: Current snapshot (B) at time T_B
            
        Returns:
            List of TapeSegments ordered by time
        """
        t_a = int(prev.ts_exch)
        t_b = int(curr.ts_exch)
        
        if t_b <= t_a:
            # Invalid interval - return single segment with no activity
            bid_price = self._best_price(prev, "bid")
            ask_price = self._best_price(prev, "ask")
            return [TapeSegment(
                index=1,
                bid_price=bid_price if bid_price is not None else 0.0,
                ask_price=ask_price if ask_price is not None else 0.0,
                u_prime_start=0.0,
                u_prime_end=1.0,
                t_start=t_a,
                t_end=t_b,
            )]
        
        # 4.1 Extract endpoint best prices
        bid_a = self._best_price(prev, "bid")
        ask_a = self._best_price(prev, "ask")
        bid_b = self._best_price(curr, "bid")
        ask_b = self._best_price(curr, "ask")
        
        # Handle missing price levels with fallbacks
        # If no levels, use 0.0 as placeholder (edge case)
        if bid_a is None:
            bid_a = bid_b if bid_b is not None else 0.0
        if ask_a is None:
            ask_a = ask_b if ask_b is not None else 0.0
        if bid_b is None:
            bid_b = bid_a
        if ask_b is None:
            ask_b = ask_a
        
        # 4.2 Get lastvolsplit prices and volumes
        last_vol_split = curr.last_vol_split or []
        price_set = {p for p, q in last_vol_split if q > 0}
        
        if not price_set:
            # No trades in interval - single segment with linear interpolation
            return [TapeSegment(
                index=1,
                bid_price=bid_a,
                ask_price=ask_a,
                u_prime_start=0.0,
                u_prime_end=1.0,
                t_start=t_a,
                t_end=t_b,
            )]
        
        p_min = min(price_set)
        p_max = max(price_set)
        
        # Map lastvolsplit to per-side volumes using ghost rule
        e_bid, e_ask = self._apply_ghost_rule(last_vol_split)
        
        # 4.3 Build discrete optimal price paths (minimal total displacement)
        bid_path = self._build_price_path(bid_a, bid_b, p_min, p_max, "bid")
        ask_path = self._build_price_path(ask_a, ask_b, p_min, p_max, "ask")
        
        # 4.4 Merge paths into global segments
        segments = self._merge_paths_to_segments(bid_path, ask_path, t_a, t_b)
        
        if not segments:
            return [TapeSegment(
                index=1,
                bid_price=bid_a,
                ask_price=ask_a,
                u_prime_start=0.0,
                u_prime_end=1.0,
                t_start=t_a,
                t_end=t_b,
            )]
        
        # 4.5 Allocate lastvolsplit to segments with 2-iteration refinement
        segments = self._allocate_volumes_iterative(segments, e_bid, e_ask)
        
        # 4.6 Derive cancellations from A/B snapshot conservation
        segments = self._derive_cancellations(segments, prev, curr, e_bid, e_ask)
        
        # Map segment times using time scaling
        segments = self._apply_time_mapping(segments, t_a, t_b)
        
        return segments
    
    def _best_price(self, snap: NormalizedSnapshot, side: str) -> Optional[Price]:
        """Extract best price from snapshot.
        
        Args:
            snap: The snapshot to extract from
            side: "bid" or "ask"
            
        Returns:
            Best price for the side, or None if no levels exist.
            Note: Caller must handle None case appropriately.
        """
        levels = snap.bids if side == "bid" else snap.asks
        if not levels:
            return None
        if side == "bid":
            return float(max(l.price for l in levels))
        return float(min(l.price for l in levels))
    
    def _apply_ghost_rule(
        self,
        last_vol_split: List[Tuple[Price, Qty]],
    ) -> Tuple[Dict[Price, Qty], Dict[Price, Qty]]:
        """Apply ghost state rule to map lastvolsplit to per-side volumes.
        
        Since lastvolsplit has no direction info, we must make assumptions.
        """
        e_bid: Dict[Price, Qty] = {}
        e_ask: Dict[Price, Qty] = {}
        
        rule = self.config.ghost_rule
        alpha = self.config.ghost_alpha
        
        for p, q in last_vol_split:
            if q <= 0:
                continue
            p = float(p)
            q = int(q)
            
            if rule == "symmetric":
                # Both sides get full volume (conservative)
                e_bid[p] = q
                e_ask[p] = q
            elif rule == "proportion":
                # Split proportionally
                e_bid[p] = int(round(alpha * q))
                e_ask[p] = int(round((1 - alpha) * q))
            elif rule == "single_bid":
                # All to bid side
                e_bid[p] = q
                e_ask[p] = 0
            elif rule == "single_ask":
                # All to ask side
                e_bid[p] = 0
                e_ask[p] = q
            else:
                # Default to symmetric
                e_bid[p] = q
                e_ask[p] = q
        
        return e_bid, e_ask
    
    def _build_price_path(
        self,
        p_start: Price,
        p_end: Price,
        p_min: Price,
        p_max: Price,
        side: str,
    ) -> List[Price]:
        """Build discrete optimal price path with at most one reversal.
        
        Tries both candidate paths and picks the one with smaller total displacement:
        - Path A: start -> p_min -> p_max -> end
        - Path B: start -> p_max -> p_min -> end
        """
        tick = self.tick_size
        
        def expand_path(waypoints: List[Price]) -> List[Price]:
            """Expand waypoints to tick-by-tick path."""
            path = []
            for i in range(len(waypoints) - 1):
                start = waypoints[i]
                end = waypoints[i + 1]
                if abs(start - end) < EPSILON:
                    if not path or abs(path[-1] - start) > EPSILON:
                        path.append(start)
                    continue
                direction = 1 if end > start else -1
                steps = int(round(abs(end - start) / tick)) + 1
                for j in range(steps):
                    p = start + j * direction * tick
                    if not path or abs(path[-1] - p) > EPSILON:
                        path.append(p)
            return path
        
        def total_displacement(waypoints: List[Price]) -> float:
            total = 0.0
            for i in range(len(waypoints) - 1):
                total += abs(waypoints[i + 1] - waypoints[i])
            return total
        
        # Two candidate paths
        path_a = [p_start, p_min, p_max, p_end]
        path_b = [p_start, p_max, p_min, p_end]
        
        disp_a = total_displacement(path_a)
        disp_b = total_displacement(path_b)
        
        chosen = path_a if disp_a <= disp_b else path_b
        
        # Expand to tick-level discrete path
        discrete = expand_path(chosen)
        
        # Align to tick grid if needed
        if self.config.price_alignment_policy == "round":
            discrete = [round(p / tick) * tick for p in discrete]
        
        # Remove consecutive duplicates
        result = []
        for p in discrete:
            if not result or abs(result[-1] - p) > EPSILON:
                result.append(p)
        
        return result if result else [p_start]
    
    def _merge_paths_to_segments(
        self,
        bid_path: List[Price],
        ask_path: List[Price],
        t_a: int,
        t_b: int,
    ) -> List[TapeSegment]:
        """Merge bid/ask paths into global segments.
        
        Each segment has constant bid and ask best prices.
        Uses double-pointer / merge approach.
        """
        # Build change points for each path
        bid_changes = [(i, bid_path[i]) for i in range(len(bid_path))]
        ask_changes = [(i, ask_path[i]) for i in range(len(ask_path))]
        
        # Normalize indices to [0, 1] progress
        bid_n = max(1, len(bid_path) - 1)
        ask_n = max(1, len(ask_path) - 1)
        
        # Collect all change points with their progress values
        events: List[Tuple[float, str, Price]] = []
        for i, p in bid_changes:
            u = i / bid_n if bid_n > 0 else 0.0
            events.append((u, "bid", p))
        for i, p in ask_changes:
            u = i / ask_n if ask_n > 0 else 0.0
            events.append((u, "ask", p))
        
        # Sort by progress, then by side
        events.sort(key=lambda x: (x[0], x[1]))
        
        # Build segments
        segments: List[TapeSegment] = []
        current_bid = bid_path[0] if bid_path else 0.0
        current_ask = ask_path[0] if ask_path else 0.0
        last_u = 0.0
        seg_idx = 1
        
        for u, side, price in events:
            if u > last_u + EPSILON:
                # Create segment from last_u to u
                segments.append(TapeSegment(
                    index=seg_idx,
                    bid_price=current_bid,
                    ask_price=current_ask,
                    u_prime_start=last_u,
                    u_prime_end=u,
                    t_start=0,  # Will be set later
                    t_end=0,
                ))
                seg_idx += 1
                last_u = u
            
            # Update current price
            if side == "bid":
                current_bid = price
            else:
                current_ask = price
        
        # Final segment to u=1
        if last_u < 1.0 - EPSILON:
            segments.append(TapeSegment(
                index=seg_idx,
                bid_price=current_bid,
                ask_price=current_ask,
                u_prime_start=last_u,
                u_prime_end=1.0,
                t_start=0,
                t_end=0,
            ))
        
        return segments
    
    def _allocate_volumes_iterative(
        self,
        segments: List[TapeSegment],
        e_bid: Dict[Price, Qty],
        e_ask: Dict[Price, Qty],
    ) -> List[TapeSegment]:
        """Allocate lastvolsplit volumes to segments using 2-iteration refinement.
        
        Section 4.5.1: Self-consistent iteration for segment lengths.
        """
        n = len(segments)
        if n == 0:
            return segments
        
        eps = self.config.epsilon
        num_iter = self.config.segment_iterations
        
        # Initialize segment lengths uniformly
        delta_u = np.ones(n) / n
        
        for iteration in range(num_iter):
            # For each price, find visiting segments and allocate volume
            e_seg_bid = np.zeros(n)
            e_seg_ask = np.zeros(n)
            
            # Bid side allocation
            for price, total_vol in e_bid.items():
                if total_vol <= 0:
                    continue
                # Find segments that visit this price
                visiting = [i for i, seg in enumerate(segments) 
                           if abs(seg.bid_price - price) < EPSILON]
                if not visiting and segments:
                    # Fallback: assign to nearest price segment
                    dists = [abs(seg.bid_price - price) for seg in segments]
                    visiting = [int(np.argmin(dists))]
                
                if not visiting:
                    continue
                
                # Weight by segment length
                weights = np.array([delta_u[i] for i in visiting])
                weights_sum = weights.sum()
                if weights_sum > EPSILON:
                    weights = weights / weights_sum
                else:
                    weights = np.ones(len(visiting)) / len(visiting)
                
                for j, i in enumerate(visiting):
                    e_seg_bid[i] += total_vol * weights[j]
            
            # Ask side allocation
            for price, total_vol in e_ask.items():
                if total_vol <= 0:
                    continue
                visiting = [i for i, seg in enumerate(segments)
                           if abs(seg.ask_price - price) < EPSILON]
                if not visiting and segments:
                    dists = [abs(seg.ask_price - price) for seg in segments]
                    visiting = [int(np.argmin(dists))]
                
                if not visiting:
                    continue
                
                weights = np.array([delta_u[i] for i in visiting])
                weights_sum = weights.sum()
                if weights_sum > EPSILON:
                    weights = weights / weights_sum
                else:
                    weights = np.ones(len(visiting)) / len(visiting)
                
                for j, i in enumerate(visiting):
                    e_seg_ask[i] += total_vol * weights[j]
            
            # Update segment lengths based on volumes
            e_total = e_seg_bid + e_seg_ask
            w = eps + e_total
            delta_u = w / w.sum()
        
        # Update segment boundaries
        cumsum = np.cumsum(delta_u)
        cumsum = np.insert(cumsum, 0, 0.0)
        
        for i, seg in enumerate(segments):
            seg.u_prime_start = float(cumsum[i])
            seg.u_prime_end = float(cumsum[i + 1])
            seg.bid_trade_volume = int(round(e_seg_bid[i]))
            seg.ask_trade_volume = int(round(e_seg_ask[i]))
        
        return segments
    
    def _derive_cancellations(
        self,
        segments: List[TapeSegment],
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
        e_bid: Dict[Price, Qty],
        e_ask: Dict[Price, Qty],
    ) -> List[TapeSegment]:
        """Derive cancellation volumes from A/B snapshot conservation.
        
        Section 4.6: Use snapshot queue conservation to estimate cancellations.
        """
        # Build price -> qty maps for snapshots
        def qty_map(snap: NormalizedSnapshot, side: str) -> Dict[Price, Qty]:
            levels = snap.bids if side == "bid" else snap.asks
            return {float(l.price): int(l.qty) for l in levels}
        
        q_bid_a = qty_map(prev, "bid")
        q_bid_b = qty_map(curr, "bid")
        q_ask_a = qty_map(prev, "ask")
        q_ask_b = qty_map(curr, "ask")
        
        # Get all relevant prices
        all_bid_prices = set(q_bid_a.keys()) | set(q_bid_b.keys()) | set(e_bid.keys())
        all_ask_prices = set(q_ask_a.keys()) | set(q_ask_b.keys()) | set(e_ask.keys())
        
        # For each price, compute net flow and allocate to active segments
        def process_side(
            segments: List[TapeSegment],
            side: str,
            q_a: Dict[Price, Qty],
            q_b: Dict[Price, Qty],
            e_s: Dict[Price, Qty],
            all_prices: set,
        ):
            for price in all_prices:
                qty_a = q_a.get(price, 0)
                qty_b = q_b.get(price, 0)
                m_total = e_s.get(price, 0)
                
                # Delta Q = Q_B - Q_A
                delta_q = qty_b - qty_a
                # Net flow = Delta Q + M (conservation)
                n_total = delta_q + m_total
                
                # Find active segments for this price (within top-K)
                active_segs = []
                for i, seg in enumerate(segments):
                    # Check if price is in active window
                    best_p = seg.bid_price if side == "bid" else seg.ask_price
                    tick = self.tick_size
                    k = self.config.active_levels
                    if side == "bid":
                        active_range = [best_p - j * tick for j in range(k)]
                    else:
                        active_range = [best_p + j * tick for j in range(k)]
                    
                    if any(abs(price - ap) < EPSILON for ap in active_range):
                        active_segs.append(i)
                
                if not active_segs:
                    # Fallback based on policy
                    if self.config.invisible_fallback == "freeze":
                        continue
                    # For "expand", just use all segments
                    active_segs = list(range(len(segments)))
                
                # Allocate net flow to active segments by length
                total_length = sum(
                    segments[i].u_prime_end - segments[i].u_prime_start
                    for i in active_segs
                )
                if total_length < EPSILON:
                    total_length = 1.0
                
                for i in active_segs:
                    seg = segments[i]
                    weight = (seg.u_prime_end - seg.u_prime_start) / total_length
                    n_seg = n_total * weight
                    
                    # Split into adds/cancels
                    adds = max(0, int(round(n_seg)))
                    cancels = max(0, int(round(-n_seg)))
                    
                    if side == "bid":
                        seg.bid_adds[price] = seg.bid_adds.get(price, 0) + adds
                        seg.bid_cancels[price] = seg.bid_cancels.get(price, 0) + cancels
                    else:
                        seg.ask_adds[price] = seg.ask_adds.get(price, 0) + adds
                        seg.ask_cancels[price] = seg.ask_cancels.get(price, 0) + cancels
        
        process_side(segments, "bid", q_bid_a, q_bid_b, e_bid, all_bid_prices)
        process_side(segments, "ask", q_ask_a, q_ask_b, e_ask, all_ask_prices)
        
        return segments
    
    def _apply_time_mapping(
        self,
        segments: List[TapeSegment],
        t_a: int,
        t_b: int,
    ) -> List[TapeSegment]:
        """Map segment boundaries from scaled progress to real time.
        
        Section 5: Time scaling with optional lambda parameter.
        """
        lam = self.config.time_scale_lambda
        dt = t_b - t_a
        
        # Threshold for considering lambda as effectively zero
        lambda_threshold = 1e-6
        
        def u_prime_to_u(u_prime: float) -> float:
            """Inverse function: u' -> u"""
            if abs(lam) < lambda_threshold:
                return u_prime
            a = 1 - math.exp(-lam)
            # u = -1/lambda * ln(1 - a * u')
            arg = 1 - a * u_prime
            if arg <= 0:
                arg = MIN_LOG_ARG  # Numerical protection
            return -math.log(arg) / lam
        
        for seg in segments:
            u_start = u_prime_to_u(seg.u_prime_start)
            u_end = u_prime_to_u(seg.u_prime_end)
            seg.t_start = int(t_a + u_start * dt)
            seg.t_end = int(t_a + u_end * dt)
        
        # Ensure final segment ends at t_b
        if segments:
            segments[-1].t_end = t_b
        
        return segments


# =============================================================================
# Exchange Simulator (Section 6)
# =============================================================================


class ExchangeSimulator:
    """No-impact exchange simulator with FIFO queue management.
    
    Key principles (Section 6):
    - Your orders don't exist in real market queue (no-impact)
    - `ahead` tracks anonymous quantity in front of your earliest order
    - Same trade volume can only be consumed once across all your orders
    """
    
    def __init__(self, config: UnifiedTapeConfig, tick_size: float = 1.0):
        self.config = config
        self.tick_size = tick_size
        
        # Per-price level state: (side, price) -> PriceLevelState
        self._levels: Dict[Tuple[Side, Price], PriceLevelState] = {}
        
        # Current simulation time
        self.current_time: int = 0
        
        # Pending receipts: (time, order_id, receipt_type, data)
        self._pending_receipts: List[Tuple[int, str, str, Any]] = []
    
    def _get_level(self, side: Side, price: Price) -> PriceLevelState:
        """Get or create price level state."""
        key = (side, float(price))
        if key not in self._levels:
            self._levels[key] = PriceLevelState(side=side, price=float(price))
        return self._levels[key]
    
    def reset(self) -> None:
        """Reset simulator state for new interval."""
        self._levels.clear()
        self._pending_receipts.clear()
        self.current_time = 0
    
    # -------------------------------------------------------------------------
    # Order Arrival (Section 6.2)
    # -------------------------------------------------------------------------
    
    def on_order_arrival(
        self,
        order: Order,
        arrival_time: int,
        market_qty_at_price: int,
    ) -> Optional[Tuple[str, str, Any]]:
        """Handle order arrival at exchange.
        
        Section 6.2: When queue is empty, initialize ahead = market_qty.
        When queue is non-empty, just append to FIFO (don't reset ahead).
        
        Args:
            order: The arriving order
            arrival_time: Time of arrival at exchange
            market_qty_at_price: Current market queue depth at order price
            
        Returns:
            Optional receipt (order_id, status, data) for immediate rejection
        """
        side = order.side
        price = float(order.price)
        
        level = self._get_level(side, price)
        
        # Check for crossing order (Section 7)
        if self._is_crossing_order(order):
            policy = self.config.crossing_order_policy
            if policy == "reject":
                return (order.order_id, "REJECTED", {"reason": "crossing"})
            elif policy == "adjust":
                # Adjust price to best passive
                # (would need market data to implement properly)
                pass
            # "passive" - proceed as normal passive order
        
        # Create shadow order
        shadow = ShadowOrder(
            order_id=order.order_id,
            side=side,
            price=price,
            remaining_qty=order.remaining_qty,
            arrival_time=arrival_time,
        )
        
        # Initialize ahead if this is first order at this level (Section 6.2)
        if not level.queue:
            level.ahead = market_qty_at_price
        
        # Append to FIFO queue
        level.queue.append(shadow)
        
        return None
    
    def _is_crossing_order(self, order: Order) -> bool:
        """Check if order would cross the spread (market order).
        
        This is a placeholder implementation. In a full implementation, this would:
        - For BUY orders: Check if order.price >= best_ask
        - For SELL orders: Check if order.price <= best_bid
        
        Currently returns False (assumes all orders are passive) because:
        1. Market data access would require additional state tracking
        2. The crossing_order_policy config handles the behavior when True
        3. Most L2 backtests assume passive limit orders
        
        To enable crossing order detection, extend this class to track
        current best bid/ask prices and implement the comparison logic.
        
        Returns:
            False (placeholder - always assumes passive orders)
        """
        # TODO: Implement crossing order detection when market state is available
        # Implementation would look like:
        # if order.side == Side.BUY and self._best_ask is not None:
        #     return order.price >= self._best_ask
        # if order.side == Side.SELL and self._best_bid is not None:
        #     return order.price <= self._best_bid
        return False
    
    # -------------------------------------------------------------------------
    # Cancel Handling (Section 6.3)
    # -------------------------------------------------------------------------
    
    def on_cancel_arrival(
        self,
        order_id: str,
        side: Side,
        price: Price,
        arrival_time: int,
    ) -> Tuple[str, str, Any]:
        """Handle cancel request arrival at exchange.
        
        Args:
            order_id: ID of order to cancel
            side: Order side
            price: Order price
            arrival_time: Time of cancel arrival
            
        Returns:
            Receipt (order_id, status, data)
        """
        level = self._get_level(side, price)
        
        # Find order in queue
        for shadow in level.queue:
            if shadow.order_id == order_id:
                if shadow.status == "FILLED":
                    return (order_id, "REJECT_CANCEL", {"reason": "already_filled"})
                
                # Cancel remaining
                cancelled_qty = shadow.remaining_qty
                shadow.remaining_qty = 0
                shadow.status = "CANCELED"
                
                return (order_id, "CANCELED", {
                    "cancelled_qty": cancelled_qty,
                    "filled_qty": shadow.filled_qty,
                    "cancel_time": arrival_time,
                })
        
        return (order_id, "REJECT_CANCEL", {"reason": "order_not_found"})
    
    # -------------------------------------------------------------------------
    # Time Slice Processing (Section 6.4)
    # -------------------------------------------------------------------------
    
    def advance(
        self,
        t_from: int,
        t_to: int,
        segment: TapeSegment,
    ) -> List[Tuple[str, str, Any]]:
        """Advance simulation from t_from to t_to using tape segment.
        
        Section 6.4: Process cancellations then trades within time slice.
        
        Args:
            t_from: Start time of slice
            t_to: End time of slice
            segment: Tape segment containing M and C for this interval
            
        Returns:
            List of receipts (order_id, status, data) for fills
        """
        if t_to <= t_from:
            return []
        
        receipts = []
        
        # Calculate time slice fraction within segment
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
            side_str = "bid" if side == Side.BUY else "ask"
            best_price = segment.bid_price if side == Side.BUY else segment.ask_price
            
            # Get segment volumes (only at best price for this segment)
            m_total = segment.bid_trade_volume if side == Side.BUY else segment.ask_trade_volume
            c_dict = segment.bid_cancels if side == Side.BUY else segment.ask_cancels
            
            # Scale to this time slice
            m_slice = int(round(m_total * slice_frac))
            
            level = self._get_level(side, best_price)
            
            # Step A: Cancellation advances ahead (Section 6.4)
            c_at_price = c_dict.get(best_price, 0)
            c_slice = int(round(c_at_price * slice_frac))
            if c_slice > 0:
                c_front = int(round(self.config.cancel_front_ratio * c_slice))
                level.ahead = max(0, level.ahead - c_front)
            
            # Step B: Trade volume consumption (Section 6.4)
            if m_slice > 0:
                fill_receipts = self._consume_trades(
                    level, m_slice, slice_start, slice_end, seg_duration
                )
                receipts.extend(fill_receipts)
        
        self.current_time = t_to
        return receipts
    
    def _consume_trades(
        self,
        level: PriceLevelState,
        trade_qty: int,
        t_start: int,
        t_end: int,
        seg_duration: int,
    ) -> List[Tuple[str, str, Any]]:
        """Consume trade volume at a price level using FIFO.
        
        Section 6.4 Step B: First consume ahead, then FIFO through queue.
        
        Args:
            level: Price level state
            trade_qty: Total trade volume to consume
            t_start: Start time of slice
            t_end: End time of slice
            seg_duration: Total segment duration
            
        Returns:
            List of fill receipts
        """
        receipts = []
        m_rem = trade_qty
        used_trade = 0  # Track consumed volume for fill time calculation
        dt = t_end - t_start
        
        # 1) First consume ahead (anonymous front queue)
        x = min(m_rem, level.ahead)
        level.ahead = level.ahead - x
        m_rem -= x
        used_trade += x
        
        # 2) Then FIFO through shadow orders
        for shadow in level.queue:
            if m_rem <= 0:
                break
            if shadow.remaining_qty <= 0 or shadow.status != "ACTIVE":
                continue
            
            f = min(m_rem, shadow.remaining_qty)
            shadow.remaining_qty -= f
            shadow.filled_qty += f
            m_rem -= f
            
            # Calculate fill time (Section 6.5: linear interpolation)
            if trade_qty > 0:
                z = (used_trade + f) / trade_qty
            else:
                z = 1.0
            t_fill = int(t_start + z * dt)
            
            shadow.fills.append((t_fill, f))
            used_trade += f
            
            # Generate fill receipt
            if shadow.remaining_qty <= 0:
                shadow.status = "FILLED"
                receipts.append((shadow.order_id, "FILLED", {
                    "fill_qty": shadow.filled_qty,
                    "fill_time": t_fill,
                    "price": shadow.price,
                }))
            else:
                receipts.append((shadow.order_id, "PARTIAL_FILL", {
                    "fill_qty": f,
                    "fill_time": t_fill,
                    "remaining_qty": shadow.remaining_qty,
                    "price": shadow.price,
                }))
        
        # 3) Remaining m_rem consumed by anonymous behind queue (no effect on us)
        
        return receipts
    
    # -------------------------------------------------------------------------
    # Interval Boundary Alignment (Section 8.3)
    # -------------------------------------------------------------------------
    
    def align_at_boundary(
        self,
        snapshot: NormalizedSnapshot,
    ) -> None:
        """Align internal state at interval boundary.
        
        Section 8.3: Clamp ahead values to snapshot observations.
        """
        for (side, price), level in self._levels.items():
            if not level.queue:
                continue
            
            # Find queue depth in snapshot
            levels = snapshot.bids if side == Side.BUY else snapshot.asks
            observed_qty = 0
            for lvl in levels:
                if abs(float(lvl.price) - price) < EPSILON:
                    observed_qty = int(lvl.qty)
                    break
            
            # Clamp ahead (Section 8.3)
            level.ahead = min(max(level.ahead, 0), observed_qty)


# =============================================================================
# Main Model Class (Section 10)
# =============================================================================


class UnifiedTapeModel(ISimulationModel):
    """Unified tape-based simulation model.
    
    Implements the complete framework from input to output as specified
    in Section 10. This model:
    
    1. Constructs event tape from A/B snapshots + lastvolsplit
    2. Runs exchange simulator with no-impact FIFO matching
    3. Generates events for the execution engine
    
    Args:
        seed: Random seed (for future stochastic extensions)
        config: Configuration parameters
        tick_size: Minimum price increment
    """
    
    def __init__(
        self,
        seed: int = 0,
        config: Optional[UnifiedTapeConfig] = None,
        tick_size: Optional[float] = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.config = config or UnifiedTapeConfig()
        
        # Tick size: can be set explicitly or inferred from data
        self._tick_size_override = float(tick_size) if tick_size else None
        
        # Internal components
        self._tape_builder: Optional[EventTapeBuilder] = None
        self._exchange: Optional[ExchangeSimulator] = None
        
        # Current interval tape cache
        self._current_tape: List[TapeSegment] = []
    
    def set_tick_size(self, tick_size: float) -> None:
        """Set tick size explicitly."""
        self._tick_size_override = float(tick_size) if tick_size and tick_size > 0 else None
    
    def _get_tick_size(
        self,
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
        context: Any,
    ) -> float:
        """Get tick size from config, context, or infer from data."""
        # 1) Explicit override
        if self._tick_size_override is not None:
            return self._tick_size_override
        
        # 2) From context/market view
        if context is not None:
            market_view = getattr(context, "market", None)
            if market_view is not None:
                tick_ctx = getattr(market_view, "tick_size", None)
                if tick_ctx is not None and float(tick_ctx) > 0:
                    return float(tick_ctx)
        
        # 3) Infer from snapshot prices
        prices: List[float] = []
        for snap in (prev, curr):
            for lvl in snap.bids + snap.asks:
                prices.append(float(lvl.price))
        
        prices = sorted(set(prices))
        diffs = [prices[i+1] - prices[i] for i in range(len(prices)-1)
                if prices[i+1] - prices[i] > EPSILON]
        
        if diffs:
            return float(min(diffs))
        return 1.0
    
    def generate_events(
        self,
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
        context: Any = None,
    ) -> Iterator[SimulationEvent]:
        """Generate simulation events for the interval [prev, curr].
        
        Section 10.2: Main event loop implementation.
        
        Args:
            prev: Previous snapshot (A)
            curr: Current snapshot (B)
            context: Optional interval context with order info
            
        Yields:
            SimulationEvents for this interval
        """
        t_a = int(prev.ts_exch)
        t_b = int(curr.ts_exch)
        
        if t_b <= t_a:
            yield SimulationEvent(t_b, EventType.SNAPSHOT_ARRIVAL, curr)
            return
        
        # Get tick size
        tick = self._get_tick_size(prev, curr, context)
        
        # Build event tape (Section 10.1)
        self._tape_builder = EventTapeBuilder(self.config, tick)
        tape = self._tape_builder.build(prev, curr)
        self._current_tape = tape
        
        # Initialize exchange simulator
        self._exchange = ExchangeSimulator(self.config, tick)
        
        if not tape:
            yield SimulationEvent(t_b, EventType.SNAPSHOT_ARRIVAL, curr)
            return
        
        # Generate events along the tape
        # For each segment, generate TRADE_TICK events at appropriate times
        for seg in tape:
            # Generate trade events within segment
            seg_duration = seg.t_end - seg.t_start
            if seg_duration <= 0:
                continue
            
            # Generate events at segment boundaries for simplicity
            # (More sophisticated: interpolate within segment)
            total_trades = seg.bid_trade_volume + seg.ask_trade_volume
            
            if total_trades > 0:
                # Generate trade tick at segment midpoint
                t_mid = (seg.t_start + seg.t_end) // 2
                
                # Bid side trades (market sells hitting bid)
                if seg.bid_trade_volume > 0:
                    yield SimulationEvent(
                        t_mid,
                        EventType.TRADE_TICK,
                        (seg.bid_price, seg.bid_trade_volume),
                    )
                
                # Ask side trades (market buys hitting ask)
                if seg.ask_trade_volume > 0:
                    yield SimulationEvent(
                        t_mid,
                        EventType.TRADE_TICK,
                        (seg.ask_price, seg.ask_trade_volume),
                    )
            
            # Generate quote update at segment end
            quote_snap = self._interpolate_snapshot(
                prev, curr, seg.t_end, t_a, t_b, seg
            )
            if quote_snap is not None and seg.t_end < t_b:
                yield SimulationEvent(
                    seg.t_end,
                    EventType.QUOTE_UPDATE,
                    quote_snap,
                )
        
        # Final snapshot arrival
        yield SimulationEvent(t_b, EventType.SNAPSHOT_ARRIVAL, curr)
    
    def _interpolate_snapshot(
        self,
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
        t: int,
        t_a: int,
        t_b: int,
        segment: TapeSegment,
    ) -> Optional[NormalizedSnapshot]:
        """Create interpolated snapshot at time t within segment.
        
        Uses segment best prices and interpolated queue depths.
        """
        if t_b <= t_a:
            return None
        
        # Progress fraction
        u = (t - t_a) / (t_b - t_a)
        u = max(0.0, min(1.0, u))
        
        # Interpolate queue depths
        def interp_levels(
            prev_levels: List[Level],
            curr_levels: List[Level],
            best_price: Price,
            side: str,
        ) -> List[Level]:
            tick = self._get_tick_size(prev, curr, None)
            result = []
            
            # Start from segment's best price
            for i in range(5):
                if side == "bid":
                    price = best_price - i * tick
                else:
                    price = best_price + i * tick
                
                # Find qty in prev/curr
                qty_prev = 0
                qty_curr = 0
                for lvl in prev_levels:
                    if abs(float(lvl.price) - price) < EPSILON:
                        qty_prev = int(lvl.qty)
                        break
                for lvl in curr_levels:
                    if abs(float(lvl.price) - price) < EPSILON:
                        qty_curr = int(lvl.qty)
                        break
                
                # Linear interpolation
                qty = int(round(qty_prev * (1 - u) + qty_curr * u))
                # Note: We use max(1, qty) to ensure non-zero depth for display.
                # In a real order book, zero-depth levels would not be shown,
                # but for interpolated snapshots we maintain 5 levels for consistency.
                qty = max(1, qty)
                
                result.append(Level(price, qty))
            
            return result
        
        bids = interp_levels(prev.bids, curr.bids, segment.bid_price, "bid")
        asks = interp_levels(prev.asks, curr.asks, segment.ask_price, "ask")
        
        return NormalizedSnapshot(
            ts_exch=t,
            bids=bids,
            asks=asks,
            last_vol_split=[],
        )
    
    # -------------------------------------------------------------------------
    # Public API for external access to tape/exchange state
    # -------------------------------------------------------------------------
    
    def get_current_tape(self) -> List[TapeSegment]:
        """Get the current interval's tape segments."""
        return self._current_tape
    
    def get_exchange_simulator(self) -> Optional[ExchangeSimulator]:
        """Get the exchange simulator (for advanced use cases)."""
        return self._exchange
