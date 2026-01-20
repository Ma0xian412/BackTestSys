"""Unified tape builder for constructing event tapes from snapshot pairs.

This module implements the complete tape construction logic from the specification:
- A/B snapshots + lastvolsplit -> Event Tape
- Discrete price paths with minimal displacement
- Iterative volume allocation
- Conservation-based cancellation derivation
- Top-5 activation window enforcement
- Time scaling with lambda parameter
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Set, Optional
import math

from ..core.interfaces import ITapeBuilder
from ..core.types import NormalizedSnapshot, Price, Qty, Side, TapeSegment, Level


# Constants
EPSILON = 1e-12
LAMBDA_THRESHOLD = 1e-6


@dataclass
class TapeConfig:
    """Configuration parameters for tape building."""
    
    # lastvolsplit -> single-side mapping
    ghost_rule: str = "symmetric"  # "symmetric", "proportion", "single_bid", "single_ask"
    ghost_alpha: float = 0.5  # For proportion rule
    
    # Segment duration iteration
    epsilon: float = 1.0  # No-trade baseline weight (prevents zero-length segments)
    segment_iterations: int = 2  # Number of iterations for volume allocation
    
    # Time scaling (u' axis)
    time_scale_lambda: float = 0.0  # Lambda for early/late event distribution
    
    # Cancellation handling
    cancel_front_ratio: float = 0.5  # phi: proportion of cancels in front (0=pessimistic, 1=optimistic)
    
    # Crossing order handling
    crossing_order_policy: str = "passive"  # "reject", "adjust", "passive"
    
    # Top-5 constraint
    top_k: int = 5  # Number of price levels to track


class UnifiedTapeBuilder(ITapeBuilder):
    """Build event tape from A/B snapshots and lastvolsplit.
    
    This is a pure function implementation - no internal state is maintained
    between calls to build().
    
    Implements the complete specification including:
    - Symmetric/proportion ghost rules for lastvolsplit
    - Optimal price path construction (minimal displacement, single reversal)
    - Two-round iterative segment width allocation
    - Conservation-based queue evolution (N = delta_Q + M)
    - Top-5 activation window enforcement
    - Time scaling via lambda parameter
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
            prev: Previous snapshot (A) at time T_A
            curr: Current snapshot (B) at time T_B
            
        Returns:
            List of TapeSegments ordered by time
        """
        t_a = int(prev.ts_exch)
        t_b = int(curr.ts_exch)
        
        if t_b <= t_a:
            # Invalid interval - return single segment
            bid_price = self._best_price(prev, Side.BUY) or 0.0
            ask_price = self._best_price(prev, Side.SELL) or 0.0
            return [TapeSegment(
                index=1,
                t_start=t_a,
                t_end=t_b,
                bid_price=bid_price,
                ask_price=ask_price,
                activation_bid=self._compute_activation_set(bid_price, Side.BUY),
                activation_ask=self._compute_activation_set(ask_price, Side.SELL),
            )]
        
        # Extract endpoint best prices
        bid_a = self._best_price(prev, Side.BUY) or 0.0
        ask_a = self._best_price(prev, Side.SELL) or 0.0
        bid_b = self._best_price(curr, Side.BUY) or bid_a
        ask_b = self._best_price(curr, Side.SELL) or ask_a
        
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
                activation_bid=self._compute_activation_set(bid_a, Side.BUY),
                activation_ask=self._compute_activation_set(ask_a, Side.SELL),
            )]
        
        p_min = min(price_set)
        p_max = max(price_set)
        
        # Map lastvolsplit to per-side volumes: E_bid(p), E_ask(p)
        e_bid, e_ask = self._apply_ghost_rule(last_vol_split)
        
        # Build optimal price paths for bid and ask (包含lastvolsplit中所有价位)
        bid_path = self._build_price_path(bid_a, bid_b, p_min, p_max, price_set)
        ask_path = self._build_price_path(ask_a, ask_b, p_min, p_max, price_set)
        
        # Merge paths into global segments
        segments = self._merge_paths_to_segments(bid_path, ask_path, t_a, t_b)
        
        if not segments:
            return [TapeSegment(
                index=1,
                t_start=t_a,
                t_end=t_b,
                bid_price=bid_a,
                ask_price=ask_a,
                activation_bid=self._compute_activation_set(bid_a, Side.BUY),
                activation_ask=self._compute_activation_set(ask_a, Side.SELL),
            )]
        
        # Compute activation sets for each segment
        segments = self._add_activation_sets(segments, prev, curr)
        
        # Two-round iterative volume allocation
        segments = self._allocate_volumes_iterative(segments, e_bid, e_ask, t_a, t_b)
        
        # Derive cancellations from conservation equations
        segments = self._derive_cancellations(segments, prev, curr, e_bid, e_ask)
        
        # Derive net flow for each segment
        segments = self._derive_net_flow(segments, prev, curr, e_bid, e_ask)
        
        return segments
    
    def _best_price(self, snap: NormalizedSnapshot, side: Side) -> Optional[float]:
        """Extract best price from snapshot."""
        levels = snap.bids if side == Side.BUY else snap.asks
        if not levels:
            return None
        if side == Side.BUY:
            return float(max(l.price for l in levels))
        return float(min(l.price for l in levels))
    
    def _compute_activation_set(self, best_price: float, side: Side) -> Set[Price]:
        """计算激活集（从最优价起的top-K档位）。

        这是基本的激活集计算，不考虑AB快照的交集。
        用于无快照信息时的回退情况。
        """
        if best_price <= 0:
            return set()

        result = set()
        for k in range(self.config.top_k):
            if side == Side.BUY:
                # Bid: best - k * tick_size
                p = best_price - k * self.tick_size
            else:
                # Ask: best + k * tick_size
                p = best_price + k * self.tick_size
            if p > 0:
                result.add(round(p, 8))  # Round to avoid floating point issues
        return result

    def _compute_activation_set_with_snapshots(
        self,
        best_price: float,
        side: Side,
        prev_snapshot: NormalizedSnapshot,
        curr_snapshot: NormalizedSnapshot
    ) -> Set[Price]:
        """计算激活集（考虑AB快照的交集）。

        激活集包含满足以下条件的价位：
        1. 在最优价之下（买方）或之上（卖方）的top-K档位
        2. 同时在A快照和B快照中都存在的价位

        这样可以避免某些激进档位突然从有效变为无效的情况。
        """
        if best_price <= 0:
            return set()

        # 获取A和B快照中的价位集合
        prev_levels = prev_snapshot.bids if side == Side.BUY else prev_snapshot.asks
        curr_levels = curr_snapshot.bids if side == Side.BUY else curr_snapshot.asks

        prev_prices = {round(float(lvl.price), 8) for lvl in prev_levels}
        curr_prices = {round(float(lvl.price), 8) for lvl in curr_levels}

        # AB快照中都出现的价位
        common_prices = prev_prices & curr_prices

        result = set()

        # 首先添加标准的top-K档位
        for k in range(self.config.top_k):
            if side == Side.BUY:
                p = best_price - k * self.tick_size
            else:
                p = best_price + k * self.tick_size
            if p > 0:
                result.add(round(p, 8))

        # 然后添加在最优价之下（买方）或之上（卖方）且在AB都出现的价位
        for price in common_prices:
            if side == Side.BUY:
                # 买方：价位应该在最优价之下或等于最优价
                if price <= best_price + EPSILON:
                    result.add(round(price, 8))
            else:
                # 卖方：价位应该在最优价之上或等于最优价
                if price >= best_price - EPSILON:
                    result.add(round(price, 8))

        return result
    
    def _apply_ghost_rule(self, last_vol_split: List[Tuple[Price, Qty]]) -> Tuple[Dict[Price, Qty], Dict[Price, Qty]]:
        """Map lastvolsplit to per-side volumes using ghost rule.
        
        Implements symmetric rule: E_bid(p) = E_ask(p) = E(p)
        """
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
                # Default to symmetric
                e_bid[p] = q
                e_ask[p] = q
        
        return e_bid, e_ask
    
    def _build_price_path(self, p_start: Price, p_end: Price, p_min: Price, p_max: Price, 
                          price_set: Set[Price] = None) -> List[Price]:
        """构建完整的价格路径（包含lastvolsplit中所有价位）。

        候选路径方向:
        - A: p_start -> 向下到p_min -> 向上到p_max -> p_end
        - B: p_start -> 向上到p_max -> 向下到p_min -> p_end

        选择总位移较小的路径方向，并确保路径包含lastvolsplit中的所有价位。
        """
        # 尝试两条候选路径方向
        path_a = [p_start, p_min, p_max, p_end]
        path_b = [p_start, p_max, p_min, p_end]

        # 计算总位移，选择较小的
        disp_a = sum(abs(path_a[i+1] - path_a[i]) for i in range(len(path_a)-1))
        disp_b = sum(abs(path_b[i+1] - path_b[i]) for i in range(len(path_b)-1))

        # 选择路径方向
        if disp_a <= disp_b:
            # 路径A: start -> min -> max -> end
            direction_down_first = True
        else:
            # 路径B: start -> max -> min -> end
            direction_down_first = False

        # 如果没有price_set，使用简单路径
        if not price_set:
            chosen = path_a if direction_down_first else path_b
            result = []
            for p in chosen:
                if not result or abs(result[-1] - p) > EPSILON:
                    result.append(p)
            return result if result else [p_start]

        # 将所有需要访问的价位收集起来
        all_prices = set(price_set)
        all_prices.add(p_start)
        all_prices.add(p_end)

        # 按价格排序
        sorted_prices = sorted(all_prices)

        # 根据选择的方向构建完整路径
        result = [p_start]

        if direction_down_first:
            # 路径: start -> 向下到min -> 向上到max -> end
            # 第一段: start -> min (向下)
            prices_below_start = [p for p in sorted_prices if p < p_start - EPSILON]
            for p in reversed(prices_below_start):  # 从高到低
                if abs(result[-1] - p) > EPSILON:
                    result.append(p)

            # 第二段: min -> max (向上，包含所有中间价位)
            for p in sorted_prices:
                if p > result[-1] + EPSILON:
                    result.append(p)

        else:
            # 路径: start -> 向上到max -> 向下到min -> end
            # 第一段: start -> max (向上)
            prices_above_start = [p for p in sorted_prices if p > p_start + EPSILON]
            for p in prices_above_start:  # 从低到高
                if abs(result[-1] - p) > EPSILON:
                    result.append(p)

            # 第二段: max -> min (向下，包含所有中间价位)
            for p in reversed(sorted_prices):
                if p < result[-1] - EPSILON:
                    result.append(p)

        # 确保终点在路径中
        if abs(result[-1] - p_end) > EPSILON:
            result.append(p_end)

        # 移除连续重复点
        final_result = []
        for p in result:
            if not final_result or abs(final_result[-1] - p) > EPSILON:
                final_result.append(round(p, 8))

        return final_result if final_result else [p_start]
    
    def _merge_paths_to_segments(self, bid_path: List[Price], ask_path: List[Price], 
                                  t_a: int, t_b: int) -> List[TapeSegment]:
        """Merge bid/ask paths into global segments.
        
        Creates segments at each price change point to ensure P_bid[i] and P_ask[i]
        are well-defined within each segment.
        """
        # Build change events (normalized progress u in [0,1])
        events: List[Tuple[float, str, Price]] = []
        
        bid_n = max(1, len(bid_path) - 1)
        ask_n = max(1, len(ask_path) - 1)
        
        for i, p in enumerate(bid_path):
            u = i / bid_n if bid_n > 0 else 0.0
            events.append((u, "bid", p))
        
        for i, p in enumerate(ask_path):
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
    
    def _add_activation_sets(self, segments: List[TapeSegment], 
                             prev: NormalizedSnapshot = None, 
                             curr: NormalizedSnapshot = None) -> List[TapeSegment]:
        """为每个段添加激活集。
        
        如果提供了prev和curr快照，则使用考虑AB交集的激活集计算方法。
        """
        result = []
        for seg in segments:
            if prev is not None and curr is not None:
                # 使用考虑AB快照交集的方法
                activation_bid = self._compute_activation_set_with_snapshots(
                    seg.bid_price, Side.BUY, prev, curr
                )
                activation_ask = self._compute_activation_set_with_snapshots(
                    seg.ask_price, Side.SELL, prev, curr
                )
            else:
                # 回退到基本方法
                activation_bid = self._compute_activation_set(seg.bid_price, Side.BUY)
                activation_ask = self._compute_activation_set(seg.ask_price, Side.SELL)
            
            new_seg = replace(
                seg,
                activation_bid=activation_bid,
                activation_ask=activation_ask,
            )
            result.append(new_seg)
        return result
    
    def _u_to_u_prime(self, u: float) -> float:
        """Convert real progress u to scaled progress u'.
        
        u' = (1 - e^(-lambda*u)) / (1 - e^(-lambda))  if |lambda| >= threshold
        u' = u                                         otherwise
        """
        lam = self.config.time_scale_lambda
        if abs(lam) < LAMBDA_THRESHOLD:
            return u
        return (1 - math.exp(-lam * u)) / (1 - math.exp(-lam))
    
    def _u_prime_to_u(self, u_prime: float) -> float:
        """Convert scaled progress u' back to real progress u.
        
        u = -ln(1 - (1 - e^(-lambda)) * u') / lambda  if |lambda| >= threshold
        u = u'                                          otherwise
        """
        lam = self.config.time_scale_lambda
        if abs(lam) < LAMBDA_THRESHOLD:
            return u_prime
        inner = 1 - (1 - math.exp(-lam)) * u_prime
        if inner <= 0:
            return 1.0
        return -math.log(inner) / lam
    
    def _allocate_volumes_iterative(self, segments: List[TapeSegment], 
                                     e_bid: Dict[Price, Qty], e_ask: Dict[Price, Qty],
                                     t_a: int, t_b: int) -> List[TapeSegment]:
        """Allocate volumes using two-round iterative refinement.
        
        Algorithm:
        1. Initialize uniform segment widths in u' space
        2. For each iteration:
           a. Compute visiting sets V_s(p) for each price/side
           b. Allocate E_s(p) to segments proportionally by width
           c. Update segment widths based on total allocated volume
        3. Map u' boundaries back to real time
        """
        n = len(segments)
        if n == 0:
            return segments
        
        eps = self.config.epsilon
        num_iter = self.config.segment_iterations
        
        # Initialize uniform widths in u' space
        delta_u_prime = [1.0 / n] * n
        
        # Track allocated volumes per segment
        m_bid_seg = [0.0] * n  # M_{bid,i} (at best bid)
        m_ask_seg = [0.0] * n  # M_{ask,i} (at best ask)
        
        for iteration in range(num_iter):
            # Reset allocations
            m_bid_seg = [0.0] * n
            m_ask_seg = [0.0] * n
            
            # Allocate bid volumes
            for price, total_vol in e_bid.items():
                if total_vol <= 0:
                    continue
                # V_bid(p) = {i | P_bid[i] == p}
                visiting = [i for i, seg in enumerate(segments) 
                           if abs(seg.bid_price - price) < EPSILON]
                if not visiting:
                    continue
                
                # Weight by segment width
                weights = [delta_u_prime[i] for i in visiting]
                total_weight = sum(weights)
                if total_weight < EPSILON:
                    weights = [1.0 / len(visiting)] * len(visiting)
                    total_weight = 1.0
                
                for j, i in enumerate(visiting):
                    m_bid_seg[i] += total_vol * weights[j] / total_weight
            
            # Allocate ask volumes
            for price, total_vol in e_ask.items():
                if total_vol <= 0:
                    continue
                # V_ask(p) = {i | P_ask[i] == p}
                visiting = [i for i, seg in enumerate(segments) 
                           if abs(seg.ask_price - price) < EPSILON]
                if not visiting:
                    continue
                
                weights = [delta_u_prime[i] for i in visiting]
                total_weight = sum(weights)
                if total_weight < EPSILON:
                    weights = [1.0 / len(visiting)] * len(visiting)
                    total_weight = 1.0
                
                for j, i in enumerate(visiting):
                    m_ask_seg[i] += total_vol * weights[j] / total_weight
            
            # Update segment widths based on total volume
            e_total = [m_bid_seg[i] + m_ask_seg[i] for i in range(n)]
            w = [eps + e for e in e_total]
            total_w = sum(w)
            delta_u_prime = [wi / total_w for wi in w]
        
        # Compute cumulative u' boundaries
        u_prime_cumsum = [0.0]
        for d in delta_u_prime:
            u_prime_cumsum.append(u_prime_cumsum[-1] + d)
        
        # Map u' back to u (real progress)
        u_cumsum = [self._u_prime_to_u(up) for up in u_prime_cumsum]
        
        # Update segment times and volumes
        dt = t_b - t_a
        result = []
        for i, seg in enumerate(segments):
            new_t_start = int(t_a + u_cumsum[i] * dt)
            new_t_end = int(t_a + u_cumsum[i+1] * dt)
            
            # Build trades dict: only at best price
            trades: Dict[Tuple[Side, Price], Qty] = {}
            if m_bid_seg[i] > 0:
                trades[(Side.BUY, seg.bid_price)] = int(round(m_bid_seg[i]))
            if m_ask_seg[i] > 0:
                trades[(Side.SELL, seg.ask_price)] = int(round(m_ask_seg[i]))
            
            new_seg = replace(
                seg,
                t_start=new_t_start,
                t_end=new_t_end,
                trades=trades,
            )
            result.append(new_seg)
        
        return result
    
    def _derive_cancellations(self, segments: List[TapeSegment], 
                               prev: NormalizedSnapshot, curr: NormalizedSnapshot,
                               e_bid: Dict[Price, Qty], e_ask: Dict[Price, Qty]) -> List[TapeSegment]:
        """Derive cancellations from snapshot conservation.
        
        For each price p in the price universe:
        - delta_Q = Q^B(p) - Q^A(p)
        - M(p) = sum of trades at p across all segments
        - N(p) = delta_Q + M(p)  (conservation: Q_B = Q_A + N - M)
        - If N < 0: there are net cancels at this price
        - Distribute cancels across segments where price is in activation window
        """
        n = len(segments)
        if n == 0:
            return segments
        
        # Build price universe (only activated prices)
        price_universe_bid: Set[Price] = set()
        price_universe_ask: Set[Price] = set()
        for seg in segments:
            price_universe_bid.update(seg.activation_bid)
            price_universe_ask.update(seg.activation_ask)
        
        # Get queue depths from snapshots
        def get_qty_at_price(snap: NormalizedSnapshot, side: Side, price: Price) -> int:
            levels = snap.bids if side == Side.BUY else snap.asks
            for lvl in levels:
                if abs(float(lvl.price) - price) < EPSILON:
                    return int(lvl.qty)
            return 0
        
        # Initialize cancels dict for each segment
        cancels_per_seg: List[Dict[Tuple[Side, Price], Qty]] = [{} for _ in range(n)]
        
        # Process bid side
        for price in price_universe_bid:
            q_a = get_qty_at_price(prev, Side.BUY, price)
            q_b = get_qty_at_price(curr, Side.BUY, price)
            
            # Total trades at this price
            m_total = sum(
                seg.trades.get((Side.BUY, price), 0) 
                for seg in segments
            )
            
            # Conservation: N = delta_Q + M
            delta_q = q_b - q_a
            n_total = delta_q + m_total
            
            # If N < 0, we have net cancels
            if n_total < 0:
                total_cancels = abs(n_total)
                # Distribute across segments where price is activated
                active_segs = [i for i, seg in enumerate(segments) 
                              if price in seg.activation_bid]
                if active_segs:
                    # Weight by segment duration
                    durations = [segments[i].t_end - segments[i].t_start for i in active_segs]
                    total_dur = sum(durations) or 1
                    for j, i in enumerate(active_segs):
                        alloc = int(round(total_cancels * durations[j] / total_dur))
                        if alloc > 0:
                            cancels_per_seg[i][(Side.BUY, price)] = alloc
        
        # Process ask side
        for price in price_universe_ask:
            q_a = get_qty_at_price(prev, Side.SELL, price)
            q_b = get_qty_at_price(curr, Side.SELL, price)
            
            m_total = sum(
                seg.trades.get((Side.SELL, price), 0) 
                for seg in segments
            )
            
            delta_q = q_b - q_a
            n_total = delta_q + m_total
            
            if n_total < 0:
                total_cancels = abs(n_total)
                active_segs = [i for i, seg in enumerate(segments) 
                              if price in seg.activation_ask]
                if active_segs:
                    durations = [segments[i].t_end - segments[i].t_start for i in active_segs]
                    total_dur = sum(durations) or 1
                    for j, i in enumerate(active_segs):
                        alloc = int(round(total_cancels * durations[j] / total_dur))
                        if alloc > 0:
                            cancels_per_seg[i][(Side.SELL, price)] = alloc
        
        # Update segments with cancels
        result = []
        for i, seg in enumerate(segments):
            new_seg = replace(seg, cancels=cancels_per_seg[i])
            result.append(new_seg)
        
        return result
    
    def _derive_net_flow(self, segments: List[TapeSegment], 
                          prev: NormalizedSnapshot, curr: NormalizedSnapshot,
                          e_bid: Dict[Price, Qty], e_ask: Dict[Price, Qty]) -> List[TapeSegment]:
        """Derive net flow (N = Adds - Cancels) for each segment.
        
        This enables Q_mkt(t) computation during simulation.
        """
        n = len(segments)
        if n == 0:
            return segments
        
        # Build price universe
        price_universe_bid: Set[Price] = set()
        price_universe_ask: Set[Price] = set()
        for seg in segments:
            price_universe_bid.update(seg.activation_bid)
            price_universe_ask.update(seg.activation_ask)
        
        def get_qty_at_price(snap: NormalizedSnapshot, side: Side, price: Price) -> int:
            levels = snap.bids if side == Side.BUY else snap.asks
            for lvl in levels:
                if abs(float(lvl.price) - price) < EPSILON:
                    return int(lvl.qty)
            return 0
        
        # Initialize net_flow for each segment
        net_flow_per_seg: List[Dict[Tuple[Side, Price], Qty]] = [{} for _ in range(n)]
        
        for side, price_universe in [(Side.BUY, price_universe_bid), (Side.SELL, price_universe_ask)]:
            for price in price_universe:
                q_a = get_qty_at_price(prev, side, price)
                q_b = get_qty_at_price(curr, side, price)
                
                m_total = sum(
                    seg.trades.get((side, price), 0) 
                    for seg in segments
                )
                
                delta_q = q_b - q_a
                n_total = delta_q + m_total  # Total net flow
                
                # Distribute N across activated segments
                active_segs = [i for i, seg in enumerate(segments) 
                              if price in (seg.activation_bid if side == Side.BUY else seg.activation_ask)]
                
                if active_segs:
                    durations = [segments[i].t_end - segments[i].t_start for i in active_segs]
                    total_dur = sum(durations) or 1
                    for j, i in enumerate(active_segs):
                        alloc = n_total * durations[j] / total_dur
                        net_flow_per_seg[i][(side, price)] = int(round(alloc))
        
        # Update segments
        result = []
        for i, seg in enumerate(segments):
            new_seg = replace(seg, net_flow=net_flow_per_seg[i])
            result.append(new_seg)
        
        return result
