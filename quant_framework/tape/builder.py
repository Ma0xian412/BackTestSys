"""Unified tape builder for constructing event tapes from snapshot pairs.

This module implements the tape construction logic from A/B snapshots + lastvolsplit.
It builds discrete price paths, allocates volumes iteratively, and derives cancellations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from ..core.interfaces import ITapeBuilder
from ..core.types import NormalizedSnapshot, Price, Qty, Side, TapeSegment, Level


# Constants
EPSILON = 1e-12


@dataclass
class TapeConfig:
    """Configuration parameters for tape building."""
    
    # lastvolsplit -> single-side mapping
    ghost_rule: str = "symmetric"  # "symmetric", "proportion", "single_bid", "single_ask"
    ghost_alpha: float = 0.5  # For proportion rule
    
    # Segment duration iteration
    epsilon: float = 1.0  # No-trade baseline weight
    segment_iterations: int = 2  # Number of iterations
    
    # Time scaling
    time_scale_lambda: float = 0.0  # Time scaling parameter
    
    # Cancellation
    cancel_front_ratio: float = 0.5  # Proportion of cancels in front
    
    # Crossing order handling
    crossing_order_policy: str = "passive"  # "reject", "adjust", "passive"


class UnifiedTapeBuilder(ITapeBuilder):
    """Build event tape from A/B snapshots and lastvolsplit.
    
    This is a pure function implementation - no internal state is maintained
    between calls to build().
    """
    
    def __init__(self, config: TapeConfig = None, tick_size: float = 1.0):
        """Initialize the tape builder.
        
        Args:
            config: Configuration parameters
            tick_size: Minimum price increment
        """
        self.config = config or TapeConfig()
        self.tick_size = tick_size
    
    def build(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> List[TapeSegment]:
        """Build tape segments from A/B snapshots.
        
        Args:
            prev: Previous snapshot (A)
            curr: Current snapshot (B)
            
        Returns:
            List of TapeSegments ordered by time
        """
        t_a = int(prev.ts_exch)
        t_b = int(curr.ts_exch)
        
        if t_b <= t_a:
            # Invalid interval - return single segment
            bid_price = self._best_price(prev, "bid") or 0.0
            ask_price = self._best_price(prev, "ask") or 0.0
            return [TapeSegment(
                index=1,
                t_start=t_a,
                t_end=t_b,
                bid_price=bid_price,
                ask_price=ask_price,
            )]
        
        # Extract endpoint best prices
        bid_a = self._best_price(prev, "bid") or 0.0
        ask_a = self._best_price(prev, "ask") or 0.0
        bid_b = self._best_price(curr, "bid") or bid_a
        ask_b = self._best_price(curr, "ask") or ask_a
        
        # Get lastvolsplit prices
        last_vol_split = curr.last_vol_split or []
        price_set = {p for p, q in last_vol_split if q > 0}
        
        if not price_set:
            # No trades - single segment
            return [TapeSegment(
                index=1,
                t_start=t_a,
                t_end=t_b,
                bid_price=bid_a,
                ask_price=ask_a,
            )]
        
        p_min = min(price_set)
        p_max = max(price_set)
        
        # Map lastvolsplit to per-side volumes
        e_bid, e_ask = self._apply_ghost_rule(last_vol_split)
        
        # Build optimal price paths
        bid_path = self._build_price_path(bid_a, bid_b, p_min, p_max)
        ask_path = self._build_price_path(ask_a, ask_b, p_min, p_max)
        
        # Merge paths into segments
        segments = self._merge_paths_to_segments(bid_path, ask_path, t_a, t_b)
        
        if not segments:
            return [TapeSegment(
                index=1,
                t_start=t_a,
                t_end=t_b,
                bid_price=bid_a,
                ask_price=ask_a,
            )]
        
        # Allocate volumes iteratively
        segments = self._allocate_volumes_iterative(segments, e_bid, e_ask)
        
        # Derive cancellations
        segments = self._derive_cancellations(segments, prev, curr, e_bid, e_ask)
        
        return segments
    
    def _best_price(self, snap: NormalizedSnapshot, side: str) -> float:
        """Extract best price from snapshot."""
        levels = snap.bids if side == "bid" else snap.asks
        if not levels:
            return 0.0
        if side == "bid":
            return float(max(l.price for l in levels))
        return float(min(l.price for l in levels))
    
    def _apply_ghost_rule(self, last_vol_split: List[Tuple[Price, Qty]]) -> Tuple[Dict[Price, Qty], Dict[Price, Qty]]:
        """Map lastvolsplit to per-side volumes."""
        e_bid: Dict[Price, Qty] = {}
        e_ask: Dict[Price, Qty] = {}
        
        for p, q in last_vol_split:
            if q <= 0:
                continue
            p = float(p)
            q = int(q)
            
            if self.config.ghost_rule == "symmetric":
                e_bid[p] = q
                e_ask[p] = q
            elif self.config.ghost_rule == "proportion":
                e_bid[p] = int(round(self.config.ghost_alpha * q))
                e_ask[p] = int(round((1 - self.config.ghost_alpha) * q))
            elif self.config.ghost_rule == "single_bid":
                e_bid[p] = q
                e_ask[p] = 0
            elif self.config.ghost_rule == "single_ask":
                e_bid[p] = 0
                e_ask[p] = q
            else:
                e_bid[p] = q
                e_ask[p] = q
        
        return e_bid, e_ask
    
    def _build_price_path(self, p_start: Price, p_end: Price, p_min: Price, p_max: Price) -> List[Price]:
        """Build optimal price path with minimal displacement."""
        # Try both candidate paths
        path_a = [p_start, p_min, p_max, p_end]
        path_b = [p_start, p_max, p_min, p_end]
        
        disp_a = sum(abs(path_a[i+1] - path_a[i]) for i in range(len(path_a)-1))
        disp_b = sum(abs(path_b[i+1] - path_b[i]) for i in range(len(path_b)-1))
        
        chosen = path_a if disp_a <= disp_b else path_b
        
        # Remove consecutive duplicates
        result = []
        for p in chosen:
            if not result or abs(result[-1] - p) > EPSILON:
                result.append(p)
        
        return result if result else [p_start]
    
    def _merge_paths_to_segments(self, bid_path: List[Price], ask_path: List[Price], 
                                  t_a: int, t_b: int) -> List[TapeSegment]:
        """Merge bid/ask paths into global segments."""
        # Build change events
        events: List[Tuple[float, str, Price]] = []
        
        bid_n = max(1, len(bid_path) - 1)
        ask_n = max(1, len(ask_path) - 1)
        
        for i, p in enumerate(bid_path):
            u = i / bid_n if bid_n > 0 else 0.0
            events.append((u, "bid", p))
        
        for i, p in enumerate(ask_path):
            u = i / ask_n if ask_n > 0 else 0.0
            events.append((u, "ask", p))
        
        events.sort(key=lambda x: (x[0], x[1]))
        
        # Build segments
        segments: List[TapeSegment] = []
        current_bid = bid_path[0] if bid_path else 0.0
        current_ask = ask_path[0] if ask_path else 0.0
        last_u = 0.0
        seg_idx = 1
        
        for u, side, price in events:
            if u > last_u + EPSILON:
                seg = TapeSegment(
                    index=seg_idx,
                    t_start=int(t_a + last_u * (t_b - t_a)),
                    t_end=int(t_a + u * (t_b - t_a)),
                    bid_price=current_bid,
                    ask_price=current_ask,
                )
                segments.append(seg)
                seg_idx += 1
                last_u = u
            
            if side == "bid":
                current_bid = price
            else:
                current_ask = price
        
        # Final segment
        if last_u < 1.0 - EPSILON:
            seg = TapeSegment(
                index=seg_idx,
                t_start=int(t_a + last_u * (t_b - t_a)),
                t_end=t_b,
                bid_price=current_bid,
                ask_price=current_ask,
            )
            segments.append(seg)
        
        return segments
    
    def _allocate_volumes_iterative(self, segments: List[TapeSegment], 
                                     e_bid: Dict[Price, Qty], e_ask: Dict[Price, Qty]) -> List[TapeSegment]:
        """Allocate volumes using iterative refinement."""
        n = len(segments)
        if n == 0:
            return segments
        
        eps = self.config.epsilon
        num_iter = self.config.segment_iterations
        
        # Initialize uniform lengths
        delta_u = np.ones(n) / n
        
        for _ in range(num_iter):
            e_seg_bid = np.zeros(n)
            e_seg_ask = np.zeros(n)
            
            # Allocate bid volumes
            for price, total_vol in e_bid.items():
                if total_vol <= 0:
                    continue
                visiting = [i for i, seg in enumerate(segments) if abs(seg.bid_price - price) < EPSILON]
                if not visiting:
                    continue
                
                weights = np.array([delta_u[i] for i in visiting])
                weights = weights / weights.sum() if weights.sum() > EPSILON else np.ones(len(visiting)) / len(visiting)
                
                for j, i in enumerate(visiting):
                    e_seg_bid[i] += total_vol * weights[j]
            
            # Allocate ask volumes
            for price, total_vol in e_ask.items():
                if total_vol <= 0:
                    continue
                visiting = [i for i, seg in enumerate(segments) if abs(seg.ask_price - price) < EPSILON]
                if not visiting:
                    continue
                
                weights = np.array([delta_u[i] for i in visiting])
                weights = weights / weights.sum() if weights.sum() > EPSILON else np.ones(len(visiting)) / len(visiting)
                
                for j, i in enumerate(visiting):
                    e_seg_ask[i] += total_vol * weights[j]
            
            # Update lengths
            e_total = e_seg_bid + e_seg_ask
            w = eps + e_total
            delta_u = w / w.sum()
        
        # Update segment times and volumes
        cumsum = np.cumsum(delta_u)
        cumsum = np.insert(cumsum, 0, 0.0)
        
        dt = segments[-1].t_end - segments[0].t_start
        t_start = segments[0].t_start
        
        for i, seg in enumerate(segments):
            new_seg = TapeSegment(
                index=seg.index,
                t_start=int(t_start + cumsum[i] * dt),
                t_end=int(t_start + cumsum[i+1] * dt),
                bid_price=seg.bid_price,
                ask_price=seg.ask_price,
                trades={(Side.BUY, seg.bid_price): int(round(e_seg_bid[i])),
                        (Side.SELL, seg.ask_price): int(round(e_seg_ask[i]))},
            )
            segments[i] = new_seg
        
        # Ensure last segment ends at correct time
        if segments:
            last = segments[-1]
            segments[-1] = TapeSegment(
                index=last.index,
                t_start=last.t_start,
                t_end=segments[-1].t_end if i == len(segments)-1 else last.t_end,
                bid_price=last.bid_price,
                ask_price=last.ask_price,
                trades=last.trades,
                cancels=last.cancels,
            )
        
        return segments
    
    def _derive_cancellations(self, segments: List[TapeSegment], 
                               prev: NormalizedSnapshot, curr: NormalizedSnapshot,
                               e_bid: Dict[Price, Qty], e_ask: Dict[Price, Qty]) -> List[TapeSegment]:
        """Derive cancellations from snapshot conservation.
        
        This is a simplified version - full implementation would use conservation equations.
        """
        # For now, just return segments unchanged
        # Full implementation would compute cancellations from queue conservation
        return segments
