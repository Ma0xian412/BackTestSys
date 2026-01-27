"""Unified tape builder for constructing event tapes from snapshot pairs.

This module implements the complete tape construction logic from the specification:
- A/B snapshots + lastvolsplit -> Event Tape
- Discrete price paths with minimal displacement
- Price-level based volume allocation
- Conservation-based cancellation derivation
- Top-5 activation window enforcement
- Time scaling with lambda parameter

All timestamps use the unified recv timeline (ts_recv) in tick units (100ns per tick).
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Set, Optional
import math

from ..core.interfaces import ITapeBuilder
from ..core.types import NormalizedSnapshot, Price, Qty, Side, TapeSegment, Level, TICK_PER_MS, SNAPSHOT_MIN_INTERVAL_TICK


# Constants
EPSILON = 1e-12
LAMBDA_THRESHOLD = 1e-6


def _largest_remainder_round(values: List[float], total: int) -> List[int]:
    """使用最大余数法将浮点数列表取整，同时保证总和不变。
    
    该方法首先将每个值向0取整，然后按照余数从大到小分配剩余的单位，
    确保取整后的总和等于指定的总数。
    
    Args:
        values: 需要取整的浮点数列表
        total: 期望的取整后总和
        
    Returns:
        取整后的整数列表，总和等于total
    """
    if not values:
        return []
    
    # 向0取整并计算余数
    floored = [math.trunc(v) for v in values]
    remainders = [v - f for v, f in zip(values, floored)]
    
    # 计算需要分配的剩余单位
    current_sum = sum(floored)
    diff = total - current_sum
    
    if diff == 0:
        return floored
    
    # 按余数大小排序，获取索引
    if diff > 0:
        # 需要增加diff个单位，选余数最大的
        indices_by_remainder = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)
        for i in range(min(diff, len(indices_by_remainder))):
            floored[indices_by_remainder[i]] += 1
    else:
        # 需要减少|diff|个单位，选余数最小的（最接近向下取整的）
        indices_by_remainder = sorted(range(len(remainders)), key=lambda i: remainders[i])
        for i in range(min(-diff, len(indices_by_remainder))):
            floored[indices_by_remainder[i]] -= 1
    
    return floored


@dataclass
class TapeConfig:
    """Configuration parameters for tape building.
    
    All time values are in tick units (100ns per tick).
    """
    
    # lastvolsplit -> single-side mapping
    ghost_rule: str = "symmetric"  # "symmetric", "proportion", "single_bid", "single_ask"
    ghost_alpha: float = 0.5  # For proportion rule
    
    # Volume allocation configuration
    epsilon: float = 1.0  # No-trade baseline weight for time allocation (prevents zero-length segments)
    segment_iterations: int = 2  # Deprecated: kept for backward compatibility
    
    # Time scaling (u' axis)
    time_scale_lambda: float = 0.0  # Lambda for early/late event distribution
    
    # Cancellation handling
    cancel_front_ratio: float = 0.5  # phi: proportion of cancels in front (0=pessimistic, 1=optimistic)
    
    # Crossing order handling
    crossing_order_policy: str = "passive"  # "reject", "adjust", "passive"
    
    # Top-5 constraint
    top_k: int = 5  # Number of price levels to track
    
    # 非均匀快照推送配置
    # Snapshot推送最小间隔（tick单位）：快照实际上是在T_B-min_interval到T_B之间产生的变化
    # 所以A快照的时间被视为T_B - min_interval
    # 500ms = 5_000_000 ticks (100ns per tick)
    snapshot_min_interval_tick: int = SNAPSHOT_MIN_INTERVAL_TICK


class UnifiedTapeBuilder(ITapeBuilder):
    """Build event tape from A/B snapshots and lastvolsplit.
    
    This is a pure function implementation - no internal state is maintained
    between calls to build().
    
    Implements the complete specification including:
    - Symmetric/proportion ghost rules for lastvolsplit
    - Optimal price path construction (minimal displacement, single reversal)
    - Price-level based volume allocation with time distribution
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
        
        从A/B快照构建tape段。使用统一的recv timeline (ts_recv)。
        
        非均匀快照时间处理（默认启用）：
        快照只在关键字段变化时推送，最小间隔500ms (5_000_000 ticks)。
        将A快照的时间视为T_B - 500ms，所有变化归因到[T_B-500ms, T_B]区间。
        
        Args:
            prev: Previous snapshot (A) at time T_A / 前一个快照（A），在T_A时刻
            curr: Current snapshot (B) at time T_B / 当前快照（B），在T_B时刻
            
        Returns:
            List of TapeSegments ordered by time / 按时间排序的TapeSegment列表
            
        Raises:
            ValueError: 当 t_b <= t_a 时抛出，因为快照时间必须严格递增
        """
        # 使用ts_recv作为主时间线
        t_a = int(prev.ts_recv)
        t_b = int(curr.ts_recv)
        
        # 快照时间必须严格递增，否则抛出ValueError
        # Snapshot timestamps must be strictly increasing
        if t_b <= t_a:
            raise ValueError(
                f"Snapshot timestamps must be strictly increasing: t_b ({t_b}) <= t_a ({t_a}). "
                f"快照时间必须严格递增。"
            )
        
        # 非均匀快照时间处理（默认启用）
        # 将A快照的时间视为T_B - min_interval
        # 这样所有变化都归因到最后min_interval内
        min_interval = self.config.snapshot_min_interval_tick
        effective_t_a = t_b - min_interval
        # 确保effective_t_a不小于原始t_a（处理间隔小于min_interval的情况）
        if effective_t_a < t_a:
            effective_t_a = t_a
        
        # Extract endpoint best prices
        bid_a = self._best_price(prev, Side.BUY) or 0.0
        ask_a = self._best_price(prev, Side.SELL) or 0.0
        bid_b = self._best_price(curr, Side.BUY) or bid_a
        ask_b = self._best_price(curr, Side.SELL) or ask_a
        
        # Get lastvolsplit prices
        last_vol_split = curr.last_vol_split or []
        price_set = {p for p, q in last_vol_split if q > 0}
        
        if not price_set:
            # No trades - single segment from effective_t_a to t_b
            return [TapeSegment(
                index=1,
                t_start=effective_t_a,
                t_end=t_b,
                bid_price=bid_a,
                ask_price=ask_a,
                activation_bid=self._compute_activation_set(bid_a, Side.BUY),
                activation_ask=self._compute_activation_set(ask_a, Side.SELL),
            )]
        
        p_min = min(price_set)
        p_max = max(price_set)
        
        # Map lastvolsplit to per-side volumes: E_bid(p), E_ask(p)
        # 将lastvolsplit映射到每个方向的成交量
        e_bid, e_ask = self._apply_ghost_rule(last_vol_split)
        
        # 使用区间扩张DP算法构建公共的"meeting sequence"
        # 确保bid和ask的中间路径完全一致（bid=ask同步的相遇价位）
        meeting_seq = self._build_meeting_sequence(price_set, p_min, p_max)
        
        # 如果起点价格（bid_a或ask_a）是成交价，但不在meeting_seq开头，
        # 则在meeting_seq前插入该起点价格，使得起点价格的成交不被误归因为撤单
        meeting_seq = self._prepend_starting_trade_prices(meeting_seq, bid_a, ask_a, price_set)
        
        # 用公共meeting序列构建对齐的bid和ask路径
        # 确保bid_path和ask_path长度相同，segment数量一致
        bid_path, ask_path = self._build_aligned_paths(bid_a, bid_b, ask_a, ask_b, meeting_seq)
        
        # 合并路径为全局段 - 使用effective_t_a作为开始时间
        segments = self._merge_paths_to_segments(bid_path, ask_path, effective_t_a, t_b)
        
        if not segments:
            segments = [TapeSegment(
                index=1,
                t_start=effective_t_a,
                t_end=t_b,
                bid_price=bid_a,
                ask_price=ask_a,
                activation_bid=self._compute_activation_set(bid_a, Side.BUY),
                activation_ask=self._compute_activation_set(ask_a, Side.SELL),
            )]
        
        # Compute activation sets for each segment
        segments = self._add_activation_sets(segments, prev, curr)
        
        # Volume allocation based on price-level distribution
        segments = self._allocate_volumes(segments, e_bid, e_ask, effective_t_a, t_b)
        
        # Derive cancellations and net flow using queue-zero constraint at price transitions
        segments = self._derive_cancellations_and_net_flow(segments, prev, curr)
        
        return segments
    
    def _best_price(self, snap: NormalizedSnapshot, side: Side) -> Optional[float]:
        """Extract best price from snapshot."""
        levels = snap.bids if side == Side.BUY else snap.asks
        if not levels:
            return None
        price_fn = max if side == Side.BUY else min
        return float(price_fn(l.price for l in levels))
    
    def _compute_activation_set(
        self,
        best_price: float,
        side: Side,
        prev_snapshot: NormalizedSnapshot = None,
        curr_snapshot: NormalizedSnapshot = None
    ) -> Set[Price]:
        """计算激活集（从最优价起的top-K档位）。

        Args:
            best_price: 当前最优价格
            side: 买卖方向（Side.BUY 或 Side.SELL）
            prev_snapshot: 前一个快照（可选），用于计算AB交集
            curr_snapshot: 当前快照（可选），用于计算AB交集

        Returns:
            激活价位集合

        行为说明：
        - 总是包含从最优价起的top-K档位（由config.top_k控制）
        - 如果同时提供prev_snapshot和curr_snapshot，还会包含在两个快照中
          都出现的价位（且满足方向约束：买方价位<=最优价，卖方价位>=最优价）
        - 如果不提供快照参数，则只返回标准的top-K档位
        """
        if best_price <= 0:
            return set()

        result = set()
        price_direction = -1 if side == Side.BUY else 1

        # 添加标准的top-K档位
        for k in range(self.config.top_k):
            p = best_price + price_direction * k * self.tick_size
            if p > 0:
                result.add(round(p, 8))

        # 如果提供了快照，添加在AB都出现的价位
        if prev_snapshot is not None and curr_snapshot is not None:
            prev_levels = prev_snapshot.bids if side == Side.BUY else prev_snapshot.asks
            curr_levels = curr_snapshot.bids if side == Side.BUY else curr_snapshot.asks

            prev_prices = {round(float(lvl.price), 8) for lvl in prev_levels}
            curr_prices = {round(float(lvl.price), 8) for lvl in curr_levels}
            common_prices = prev_prices & curr_prices

            for price in common_prices:
                if side == Side.BUY:
                    if price <= best_price + EPSILON:
                        result.add(round(price, 8))
                else:
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
    
    def _build_meeting_sequence(self, price_set: Set[Price], p_min: Price, p_max: Price) -> List[Price]:
        """使用区间扩张DP算法构建公共的"meeting sequence"（相遇价位序列）。
        
        这是bid和ask共享的中间路径，确保两条路径的中间点完全一致。
        meeting sequence是成交分解给出的可达/必经相遇价位。
        
        算法思路：区间扩张DP
        - 从[p_min, p_max]这个区间出发
        - 找出所有需要访问的价位（lastvolsplit中的价位）
        - 通过最小位移的方式遍历所有价位
        
        Args:
            price_set: lastvolsplit中所有有成交的价位集合
            p_min: 最小成交价
            p_max: 最大成交价
            
        Returns:
            公共相遇价位序列，按访问顺序排列。
            如果price_set为空，返回空列表（表示没有中间相遇点，路径只包含起点和终点）。
        """
        # 如果没有成交价位，返回空列表
        # 调用方会生成只包含起点和终点的路径，这是正确的行为
        if not price_set:
            return []
        
        # 收集所有需要访问的价位
        prices_to_visit = sorted(price_set)
        
        if len(prices_to_visit) <= 1:
            return list(prices_to_visit)
        
        # 区间扩张DP：从某个起点开始，通过扩张区间来访问所有价位
        # 核心思想：维护当前已访问的区间[lo, hi]，每次选择扩张到下一个未访问的价位
        # 最终路径包含所有价位
        
        # 策略：从中间开始向两边扩张，或从一端开始单调遍历
        # 简单实现：找到最小总位移的遍历顺序
        
        # 方案1：从p_min开始单调递增到p_max
        path_ascending = prices_to_visit[:]
        
        # 方案2：从p_max开始单调递减到p_min
        path_descending = list(reversed(prices_to_visit))
        
        # 方案3：区间扩张 - 从中位数开始，交替向两边扩张
        n = len(prices_to_visit)
        mid_idx = n // 2
        
        path_expand = [prices_to_visit[mid_idx]]
        lo_idx, hi_idx = mid_idx, mid_idx
        
        while lo_idx > 0 or hi_idx < n - 1:
            # 计算向下扩张和向上扩张的代价
            cost_lo = abs(prices_to_visit[lo_idx - 1] - path_expand[-1]) if lo_idx > 0 else float('inf')
            cost_hi = abs(prices_to_visit[hi_idx + 1] - path_expand[-1]) if hi_idx < n - 1 else float('inf')
            
            if cost_lo <= cost_hi and lo_idx > 0:
                lo_idx -= 1
                path_expand.append(prices_to_visit[lo_idx])
            elif hi_idx < n - 1:
                hi_idx += 1
                path_expand.append(prices_to_visit[hi_idx])
            else:
                break
        
        # 计算各路径的总位移
        def total_displacement(path: List[Price]) -> float:
            if len(path) <= 1:
                return 0.0
            return sum(abs(path[i+1] - path[i]) for i in range(len(path) - 1))
        
        disp_asc = total_displacement(path_ascending)
        disp_desc = total_displacement(path_descending)
        disp_expand = total_displacement(path_expand)
        
        # 选择总位移最小的路径
        min_disp = min(disp_asc, disp_desc, disp_expand)
        
        if min_disp == disp_asc:
            meeting_seq = path_ascending
        elif min_disp == disp_desc:
            meeting_seq = path_descending
        else:
            meeting_seq = path_expand
        
        # 移除连续重复点
        result = []
        for p in meeting_seq:
            if not result or abs(result[-1] - p) > EPSILON:
                result.append(round(p, 8))
        
        return result
    
    def _prepend_starting_trade_prices(
        self,
        meeting_seq: List[Price],
        bid_a: Price,
        ask_a: Price,
        price_set: Set[Price]
    ) -> List[Price]:
        """在meeting序列前插入起点成交价，使起点价格的成交不被误归因为撤单。
        
        当某一方的起点价格（bid_a或ask_a）本身也是成交价时，根据meeting_seq的方向
        决定插入哪个起点价格：
        - 如果meeting_seq先向下（第一个元素 < bid_a），说明bid先移动，插入bid_a
        - 如果meeting_seq先向上（第一个元素 > ask_a），说明ask先移动，插入ask_a
        
        这样可以让需要移动的一方先停留一拍，在起点价格上进行成交，
        避免起点价格的数量变化被错误地全部归因为撤单。
        
        例如：
        - bid_a=6, ask_a=7, meeting_seq=[5,6,7], price_set={5,6,7}
        - meeting_seq[0]=5 < bid_a=6，说明bid先向下移动
        - bid_a=6 是成交价 -> 插入bid_a: [6,5,6,7]
        - 结果: bid路径=[6,6,5,6,7,5], ask路径=[7,6,5,6,7,7]
        - 第二个segment: bid=6, ask=6，成交可以分配到价格6
        
        Args:
            meeting_seq: 原始的meeting序列
            bid_a: bid侧起点价格
            ask_a: ask侧起点价格
            price_set: 所有成交价位集合
            
        Returns:
            调整后的meeting序列
        """
        if not meeting_seq:
            # 如果没有meeting序列，检查起点是否是成交价
            result = []
            # 先检查bid_a
            if any(abs(bid_a - p) < EPSILON for p in price_set):
                result.append(round(bid_a, 8))
            # 再检查ask_a（如果与bid_a不同）
            if abs(ask_a - bid_a) > EPSILON and any(abs(ask_a - p) < EPSILON for p in price_set):
                result.append(round(ask_a, 8))
            return result
        
        result = list(meeting_seq)
        first_meeting = result[0]
        
        # 根据meeting_seq的方向决定插入哪个起点价格
        # 如果第一个meeting点低于bid_a，说明bid需要先向下移动（向下方向）
        # 如果第一个meeting点高于ask_a，说明ask需要先向上移动（向上方向）
        
        goes_down_first = first_meeting < bid_a - EPSILON  # bid先向下移动
        goes_up_first = first_meeting > ask_a + EPSILON    # ask先向上移动
        
        if goes_down_first:
            # bid先向下移动，检查bid_a是否是成交价
            bid_a_is_trade = any(abs(bid_a - p) < EPSILON for p in price_set)
            if bid_a_is_trade:
                # 在前面插入bid_a，让bid在起点停留一拍
                result.insert(0, round(bid_a, 8))
        elif goes_up_first:
            # ask先向上移动，检查ask_a是否是成交价
            ask_a_is_trade = any(abs(ask_a - p) < EPSILON for p in price_set)
            if ask_a_is_trade:
                # 在前面插入ask_a，让ask在起点停留一拍
                result.insert(0, round(ask_a, 8))
        
        return result
    
    def _build_aligned_paths(
        self, 
        bid_a: Price, bid_b: Price,
        ask_a: Price, ask_b: Price,
        meeting_seq: List[Price]
    ) -> Tuple[List[Price], List[Price]]:
        """构建对齐的bid和ask价格路径。
        
        确保bid_path和ask_path具有完全相同的长度，并且在meeting价位处对齐，
        使得每个segment中如果有成交，bid和ask都能参与（双边成交）。
        
        核心思想：
        - 两条路径都保留起点和终点
        - 中间经过meeting序列（成交价位），且两条路径在meeting价位处同步
        - 这确保了在meeting价位对应的segment中，bid_price == ask_price == meeting_price
        
        路径结构：
        [start] -> [meeting_1, meeting_1] -> [meeting_2, meeting_2] -> ... -> [end]
        
        其中每个meeting价位在两条路径中都出现在相同位置，确保双边成交。
        
        Args:
            bid_a: A快照的最优买价
            bid_b: B快照的最优买价
            ask_a: A快照的最优卖价
            ask_b: B快照的最优卖价
            meeting_seq: 公共相遇价位序列（成交价位）
            
        Returns:
            (bid_path, ask_path) 两条对齐的价格路径，长度相同
        """
        if not meeting_seq:
            # 没有成交，返回简单的两点路径
            bid_path = [round(bid_a, 8)]
            ask_path = [round(ask_a, 8)]
            if abs(bid_a - bid_b) > EPSILON:
                bid_path.append(round(bid_b, 8))
            if abs(ask_a - ask_b) > EPSILON:
                ask_path.append(round(ask_b, 8))
            # 确保路径长度相同
            return self._pad_paths_to_same_length(bid_path, ask_path, bid_b, ask_b)
        
        # 构建对齐的路径，确保meeting价位在两条路径中同步出现
        # 路径结构：[start] -> [meeting prices aligned] -> [end]
        
        bid_path = [round(bid_a, 8)]
        ask_path = [round(ask_a, 8)]
        
        # 所有meeting价位都必须在两条路径中同步出现
        # 这确保了在对应的segment中 bid_price == ask_price == meeting_price
        for meeting_price in meeting_seq:
            rounded_price = round(meeting_price, 8)
            # 总是添加meeting价位到两条路径，即使与前一个相同也要添加
            # 这确保两条路径在meeting价位处同步
            bid_path.append(rounded_price)
            ask_path.append(rounded_price)
        
        # 添加终点
        bid_path.append(round(bid_b, 8))
        ask_path.append(round(ask_b, 8))
        
        # 此时两条路径应该已经长度相同（start + meetings + end）
        # 但为了安全，还是调用padding
        return self._pad_paths_to_same_length(bid_path, ask_path, bid_b, ask_b)
    
    def _pad_paths_to_same_length(
        self, 
        bid_path: List[Price], 
        ask_path: List[Price],
        bid_end: Price,
        ask_end: Price
    ) -> Tuple[List[Price], List[Price]]:
        """将两条路径填充到相同长度。
        
        如果路径长度不同，用最后一个价格填充较短的路径。
        
        Args:
            bid_path: bid价格路径
            ask_path: ask价格路径
            bid_end: bid终点价格（用于填充）
            ask_end: ask终点价格（用于填充）
            
        Returns:
            (bid_path, ask_path) 长度相同的两条路径
        """
        len_bid = len(bid_path)
        len_ask = len(ask_path)
        
        if len_bid == len_ask:
            return bid_path, ask_path
        
        if len_bid > len_ask:
            # 填充ask_path
            padding_value = ask_path[-1] if ask_path else round(ask_end, 8)
            while len(ask_path) < len_bid:
                ask_path.append(padding_value)
        else:
            # 填充bid_path
            padding_value = bid_path[-1] if bid_path else round(bid_end, 8)
            while len(bid_path) < len_ask:
                bid_path.append(padding_value)
        
        return bid_path, ask_path
    
    def _merge_paths_to_segments(self, bid_path: List[Price], ask_path: List[Price], 
                                  t_a: int, t_b: int) -> List[TapeSegment]:
        """Merge bid/ask paths into global segments.
        
        Creates segments at each price change point to ensure P_bid[i] and P_ask[i]
        are well-defined within each segment.
        """
        # Build change events (normalized progress u in [0,1])
        # Use len(path) as divisor so N path points generate N segments
        # (each point starts a segment that ends at the next point's u or at u=1)
        events: List[Tuple[float, str, Price]] = []
        
        bid_n = max(1, len(bid_path))
        ask_n = max(1, len(ask_path))
        
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
            activation_bid = self._compute_activation_set(
                seg.bid_price, Side.BUY, prev, curr
            )
            activation_ask = self._compute_activation_set(
                seg.ask_price, Side.SELL, prev, curr
            )
            
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
    
    def _allocate_volumes(self, segments: List[TapeSegment], 
                          e_bid: Dict[Price, Qty], e_ask: Dict[Price, Qty],
                          t_a: int, t_b: int) -> List[TapeSegment]:
        """Allocate volumes based on price-level volume distribution.
        
        Algorithm:
        1. Count occurrences of each price in the path (bilateral segments)
        2. Distribute total volume for each price evenly across its occurrences
        3. For prices with no volume, use epsilon as minimum weight
        4. Time allocation is proportional to allocated volume weights
        
        Example:
        - Path: 0, 1, 2, 3, 2, 1, 4
        - Volume at price 1: 100 total (appears twice -> 50 each)
        - Volume at price 2: 200 total (appears twice -> 100 each)
        - Volume at price 3: 200 total (appears once -> 200)
        - Prices 0 and 4 have no volume -> use epsilon
        
        Time allocation for total duration N:
        - Total weight = 2*epsilon + 100 + 200 + 200 = 2*epsilon + 500
        - Time for price 0 = (N / total_weight) * epsilon
        - Time for each price 1 occurrence = (N / total_weight) * 50
        
        重要约束：成交必须是双边的
        - 只有当segment的bid_price == ask_price == 成交价时，才能分配成交量
        - 确保每个segment中如果有BUY成交，就必须有对应的SELL成交
        """
        n = len(segments)
        if n == 0:
            return segments
        
        eps = self.config.epsilon
        
        # Track allocated volumes per segment (bilateral - same for both sides)
        m_seg = [0.0] * n  # M_i (bilateral trade volume at price)
        
        # Step 1: Count occurrences of each price (bilateral segments only)
        # Price is considered bilateral when bid_price == ask_price
        price_segment_indices: Dict[Price, List[int]] = {}
        for i, seg in enumerate(segments):
            if abs(seg.bid_price - seg.ask_price) < EPSILON:
                price = seg.bid_price
                if price not in price_segment_indices:
                    price_segment_indices[price] = []
                price_segment_indices[price].append(i)
        
        # Step 2: Distribute volumes evenly for each price across its occurrences
        # Use _largest_remainder_round to preserve total volume when dividing
        for price, total_vol in e_bid.items():
            if total_vol <= 0:
                continue
            
            if price not in price_segment_indices:
                continue
            
            indices = price_segment_indices[price]
            count = len(indices)
            if count == 0:
                continue
            
            # Calculate even distribution and use largest remainder method for rounding
            vol_per_segment_float = total_vol / count
            float_volumes = [vol_per_segment_float] * count
            rounded_volumes = _largest_remainder_round(float_volumes, int(total_vol))
            
            for j, i in enumerate(indices):
                m_seg[i] += rounded_volumes[j]
        
        # Step 3: Calculate segment weights for time allocation
        # Weight = allocated volume, or eps (config.epsilon) if no volume
        # Note: EPSILON (1e-12) is used for floating point comparison, 
        # eps (config.epsilon, default 1.0) is used as minimum weight for no-trade segments
        weights = []
        for i in range(n):
            if m_seg[i] > EPSILON:
                weights.append(m_seg[i])
            else:
                weights.append(eps)
        
        total_weight = sum(weights)
        
        # Guard against division by zero (should not happen with eps > 0)
        if total_weight < EPSILON:
            total_weight = n  # Fallback to uniform distribution
            weights = [1.0] * n
        
        # Step 4: Calculate time proportions based on weights
        # delta_u_prime[i] = weights[i] / total_weight
        delta_u_prime = [w / total_weight for w in weights]
        
        # Compute cumulative u' boundaries
        u_prime_cumsum = [0.0]
        for d in delta_u_prime:
            u_prime_cumsum.append(u_prime_cumsum[-1] + d)
        
        # Map u' back to u (real progress) - apply lambda transformation
        u_cumsum = [self._u_prime_to_u(up) for up in u_prime_cumsum]
        
        # Update segment times and volumes
        dt = t_b - t_a
        result = []
        for i, seg in enumerate(segments):
            new_t_start = int(t_a + u_cumsum[i] * dt)
            new_t_end = int(t_a + u_cumsum[i+1] * dt)
            
            # Build trades dict: bilateral trades at the matching price
            trades: Dict[Tuple[Side, Price], Qty] = {}
            if m_seg[i] > 0:
                trade_price = seg.bid_price  # bid_price == ask_price in bilateral segments
                trade_qty = int(m_seg[i])  # Already integer from _largest_remainder_round
                trades[(Side.BUY, trade_price)] = trade_qty
                trades[(Side.SELL, trade_price)] = trade_qty
            
            new_seg = replace(
                seg,
                t_start=new_t_start,
                t_end=new_t_end,
                trades=trades,
            )
            result.append(new_seg)
        
        return result
    
    def _get_activation_set_for_side(self, seg: TapeSegment, side: Side) -> Set[Price]:
        """获取指定方向的激活集。"""
        return seg.activation_bid if side == Side.BUY else seg.activation_ask

    def _get_best_price_for_side(self, seg: TapeSegment, side: Side) -> Price:
        """获取指定方向的最优价格。"""
        return seg.bid_price if side == Side.BUY else seg.ask_price

    def _find_price_transition_segments(
        self, segments: List[TapeSegment], side: Side
    ) -> Dict[Price, List[Tuple[int, int, bool]]]:
        """找出每个价位作为best price的连续段范围。
        
        返回: {price: [(start_idx, end_idx, ends_with_transition), ...]}
        ends_with_transition: True表示该段组结束时价格转换（队列清空），False表示保持或最终段
        """
        result: Dict[Price, List[Tuple[int, int, bool]]] = {}
        n = len(segments)
        if not segments:
            return result
        
        # 获取每段的best price
        best_prices = [self._get_best_price_for_side(seg, side) for seg in segments]
        
        # 找出连续段组
        i = 0
        while i < n:
            price = best_prices[i]
            start = i
            # 找到该价位的连续段结束位置
            while i < n and abs(best_prices[i] - price) < EPSILON:
                i += 1
            end = i - 1  # 包含的最后一个段索引
            
            # 判断是否是价格转换（bid下降或ask上升表示队列清空）
            ends_with_transition = False
            if i < n:  # 还有后续段
                next_price = best_prices[i]
                if side == Side.BUY and next_price < price - EPSILON:
                    ends_with_transition = True
                elif side == Side.SELL and next_price > price + EPSILON:
                    ends_with_transition = True
            
            if price not in result:
                result[price] = []
            result[price].append((start, end, ends_with_transition))
        
        return result

    def _process_side_transitions(
        self,
        side: Side,
        segments: List[TapeSegment],
        transitions: Dict[Price, List[Tuple[int, int, bool]]],
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
        cancels_per_seg: List[Dict[Tuple[Side, Price], Qty]],
        net_flow_per_seg: List[Dict[Tuple[Side, Price], Qty]],
    ) -> None:
        """处理指定方向的价格转换和流量分配（best-price价位）。
        
        采用动态追踪剩余净增量N的方法，按段顺序逐段分配，确保任何时刻队列长度不为负。
        
        算法流程：
        1. 首先计算全局净增量 N_total = (Q_curr - Q_prev) + M_total
        2. 按段顺序遍历，维护一个动态的"剩余待分配量" N_remaining，初始值为 N_total
        3. 对于每一段：
           - 计算该段的起始队列深度 Q_start（第一段为 Q_prev，后续段根据前面段的变化累积得出）
           - 如果是归零段（价格转换离开该档位）：该段必须消耗 Q_start 使队列归零，
             净流入量 = M_group - Q_start，N_remaining += Q_start（归零释放的量加回待分配池）
           - 如果是有成交段：净流入量 = M_group（需要支撑成交），从N_remaining中消耗
           - 如果是无成交的普通段：按时长比例分配部分 N_remaining
        
        Args:
            side: 买卖方向
            segments: 所有TapeSegment列表
            transitions: 价格转换信息，由_find_price_transition_segments生成
            prev: 前一个快照（用于获取初始队列深度Q_A）
            curr: 当前快照（用于获取最终队列深度Q_B）
            cancels_per_seg: 撤单量输出字典列表（会被修改）
            net_flow_per_seg: 净流入量输出字典列表（会被修改）
        """
        def get_qty_at_price(snap: NormalizedSnapshot, price: Price) -> int:
            levels = snap.bids if side == Side.BUY else snap.asks
            for lvl in levels:
                if abs(float(lvl.price) - price) < EPSILON:
                    return int(lvl.qty)
            return 0

        for price, groups in transitions.items():
            q_a = get_qty_at_price(prev, price)
            q_b = get_qty_at_price(curr, price)
            
            # 计算该价位的全局成交量
            m_total_at_price = sum(
                seg.trades.get((side, price), 0) for seg in segments
            )
            
            # 计算全局净增量 N_total = (Q_B - Q_A) + M_total
            delta_q = q_b - q_a
            n_total = delta_q + m_total_at_price
            
            # 动态追踪剩余待分配量，初始值为 N_total
            n_remaining = float(n_total)
            
            # 动态追踪当前队列深度，初始值为 Q_A
            q_current = float(q_a)
            
            for group_idx, (start_idx, end_idx, ends_with_transition) in enumerate(groups):
                m_group = sum(
                    segments[i].trades.get((side, price), 0)
                    for i in range(start_idx, end_idx + 1)
                )
                
                # 该段组的起始队列深度
                q_start = q_current
                
                if ends_with_transition:
                    # 归零段：队列必须从 q_start 归零
                    # 净流入量 = M_group - Q_start（为了使队列归零）
                    n_group = m_group - q_start
                    
                    # 更新n_remaining：消耗n_group
                    n_remaining -= n_group
                    
                    # 段结束后队列归零
                    q_current = 0.0
                else:
                    # 非归零段（保持或最终段）
                    if m_group > 0:
                        # 有成交段：必须分配足够的净流入量来支撑成交
                        # 净流入量 = M_group（确保队列从0开始能支撑M_group的成交）
                        # 但受限于剩余可分配量n_remaining
                        n_group = min(m_group, n_remaining) if n_remaining > 0 else m_group
                        
                        # 如果起始队列q_start > 0，则实际需要的净流入量可以减少
                        # 因为队列已有的深度可以支撑部分成交
                        if q_start > 0:
                            # 需要的净流入量 = max(0, M_group - Q_start)
                            # 但为了保持守恒，我们需要消耗n_remaining
                            needed = max(0.0, m_group - q_start)
                            n_group = min(needed, n_remaining) if n_remaining > 0 else needed
                        
                        # 更新剩余待分配量
                        n_remaining -= n_group
                        
                        # 更新当前队列深度
                        q_current = q_start + n_group - m_group
                        if q_current < 0:
                            q_current = 0.0
                    else:
                        # 无成交的普通段：按时长比例分配部分 N_remaining
                        # 计算该段组在所有活跃段中的时长比例
                        all_active_segs = [
                            i for i, seg in enumerate(segments)
                            if price in self._get_activation_set_for_side(seg, side)
                        ]
                        group_segs = list(range(start_idx, end_idx + 1))
                        
                        group_dur = sum(segments[i].t_end - segments[i].t_start for i in group_segs)
                        total_dur = sum(segments[i].t_end - segments[i].t_start for i in all_active_segs) or 1
                        
                        # 按时长比例分配剩余待分配量
                        n_group = n_remaining * group_dur / total_dur
                        
                        # 确保队列不为负：n_group >= -q_start
                        if n_group < -q_start:
                            n_group = -q_start
                        
                        # 更新剩余待分配量
                        n_remaining -= n_group
                        
                        # 更新当前队列深度
                        q_current = q_start + n_group - m_group
                        if q_current < 0:
                            q_current = 0.0
                
                group_segs = list(range(start_idx, end_idx + 1))
                durations = [segments[i].t_end - segments[i].t_start for i in group_segs]
                total_dur = sum(durations) or 1
                
                # 筛选出在activation集中的段
                active_in_group = [
                    (j, seg_idx) for j, seg_idx in enumerate(group_segs)
                    if price in self._get_activation_set_for_side(segments[seg_idx], side)
                ]
                
                if active_in_group:
                    # 计算每个活跃段的分配比例
                    active_durations = [durations[j] for j, _ in active_in_group]
                    active_total_dur = sum(active_durations) or 1
                    alloc_values = [n_group * d / active_total_dur for d in active_durations]
                    
                    # 使用最大余数法取整，保证总和等于n_group（取整后）
                    n_group_int = int(round(n_group))
                    rounded_allocs = _largest_remainder_round(alloc_values, n_group_int)
                    
                    for k, (j, seg_idx) in enumerate(active_in_group):
                        net_flow_per_seg[seg_idx][(side, price)] = rounded_allocs[k]
                        
                        if rounded_allocs[k] < 0:
                            cancels_per_seg[seg_idx][(side, price)] = abs(rounded_allocs[k])

    def _process_non_best_prices(
        self,
        side: Side,
        price_universe: Set[Price],
        segments: List[TapeSegment],
        transitions: Dict[Price, List[Tuple[int, int, bool]]],
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
        cancels_per_seg: List[Dict[Tuple[Side, Price], Qty]],
        net_flow_per_seg: List[Dict[Tuple[Side, Price], Qty]],
    ) -> None:
        """处理非best-price但在activation中的价位（使用守恒方程）。
        
        对于那些从未作为best-price但在某些段的activation集中出现的价位，
        使用守恒方程 N = delta_Q + M 计算净流入量，并按段时长比例分配。
        
        Args:
            side: 买卖方向
            price_universe: 所有在activation集中出现过的价位
            segments: 所有TapeSegment列表
            transitions: 已处理的best-price价位（需要跳过）
            prev: 前一个快照（用于获取初始队列深度Q_A）
            curr: 当前快照（用于获取最终队列深度Q_B）
            cancels_per_seg: 撤单量输出字典列表（会被修改）
            net_flow_per_seg: 净流入量输出字典列表（会被修改）
        """
        def get_qty_at_price(snap: NormalizedSnapshot, price: Price) -> int:
            levels = snap.bids if side == Side.BUY else snap.asks
            for lvl in levels:
                if abs(float(lvl.price) - price) < EPSILON:
                    return int(lvl.qty)
            return 0

        for price in price_universe:
            if price in transitions:
                continue
            
            q_a = get_qty_at_price(prev, price)
            q_b = get_qty_at_price(curr, price)
            
            m_total = sum(seg.trades.get((side, price), 0) for seg in segments)
            delta_q = q_b - q_a
            n_total = delta_q + m_total
            
            active_segs = [
                i for i, seg in enumerate(segments)
                if price in self._get_activation_set_for_side(seg, side)
            ]
            if not active_segs:
                continue
            
            durations = [segments[i].t_end - segments[i].t_start for i in active_segs]
            total_dur = sum(durations) or 1
            
            # 计算每个活跃段的分配比例
            alloc_values = [n_total * d / total_dur for d in durations]
            
            # 使用最大余数法取整，保证总和等于n_total（取整后）
            n_total_int = int(round(n_total))
            rounded_allocs = _largest_remainder_round(alloc_values, n_total_int)
            
            for j, i in enumerate(active_segs):
                if (side, price) not in net_flow_per_seg[i]:
                    net_flow_per_seg[i][(side, price)] = rounded_allocs[j]
                
                if rounded_allocs[j] < 0:
                    alloc_cancel = abs(rounded_allocs[j])
                    if alloc_cancel > 0 and (side, price) not in cancels_per_seg[i]:
                        cancels_per_seg[i][(side, price)] = alloc_cancel

    def _derive_cancellations_and_net_flow(
        self,
        segments: List[TapeSegment],
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
    ) -> List[TapeSegment]:
        """Derive cancellations and net flow from snapshot conservation.

        For each price p in the activated universe:
        - delta_Q = Q^B(p) - Q^A(p)
        - M(p) = sum of trades at p across all segments
        - N(p) = delta_Q + M(p)  (conservation: Q_B = Q_A + N - M)
        - Distribute N sequentially across active segments based on duration
          while ensuring queue depth never goes negative. If a segment is the
          last active one before the price deactivates (bid down / ask up),
          force the queue to zero at the end of that segment.
        """
        n = len(segments)
        if n == 0:
            return segments

        price_universe_bid: Set[Price] = set()
        price_universe_ask: Set[Price] = set()
        for seg in segments:
            price_universe_bid.update(seg.activation_bid)
            price_universe_ask.update(seg.activation_ask)

        cancels_per_seg: List[Dict[Tuple[Side, Price], Qty]] = [{} for _ in range(n)]
        net_flow_per_seg: List[Dict[Tuple[Side, Price], Qty]] = [{} for _ in range(n)]

        for side, price_universe in [(Side.BUY, price_universe_bid), (Side.SELL, price_universe_ask)]:
            for price in price_universe:
                q_a = get_qty_at_price(prev, side, price)
                q_b = get_qty_at_price(curr, side, price)

                m_total = sum(seg.trades.get((side, price), 0) for seg in segments)
                delta_q = q_b - q_a
                n_total = delta_q + m_total

                active_segs = [
                    i
                    for i, seg in enumerate(segments)
                    if price in (seg.activation_bid if side == Side.BUY else seg.activation_ask)
                ]
                if not active_segs:
                    continue

                durations = [
                    max(0, segments[i].t_end - segments[i].t_start) for i in active_segs
                ]
                if sum(durations) <= 0:
                    durations = [1 for _ in active_segs]

                n_remaining = int(n_total)
                q_start = int(q_a)
                net_flow_allocations: List[Tuple[int, int]] = []

                for j, i in enumerate(active_segs):
                    remaining_dur = sum(durations[j:])
                    if remaining_dur <= 0:
                        remaining_dur = max(1, len(durations) - j)

                    if j == len(active_segs) - 1:
                        alloc_net = float(n_remaining)
                    else:
                        alloc_net = n_remaining * durations[j] / remaining_dur

                    seg = segments[i]
                    m_seg = int(seg.trades.get((side, price), 0))

                    next_active = False
                    if i < n - 1:
                        next_seg = segments[i + 1]
                        next_set = next_seg.activation_bid if side == Side.BUY else next_seg.activation_ask
                        next_active = price in next_set

                    is_zeroing = (i < n - 1) and (not next_active)

                    min_needed = m_seg - q_start
                    if is_zeroing:
                        n_seg = min_needed
                    else:
                        n_seg = max(alloc_net, min_needed)

                    n_seg_int = int(round(n_seg))
                    if not is_zeroing and n_seg_int < min_needed:
                        n_seg_int = min_needed

                    net_flow_per_seg[i][(side, price)] = n_seg_int

                    if n_seg_int < 0:
                        cancels_per_seg[i][(side, price)] = abs(n_seg_int)

                    net_flow_allocations.append((i, n_seg_int))

                    n_remaining -= n_seg_int
                    q_start = q_start + n_seg_int - m_seg

                if net_flow_allocations:
                    netflow_sum = sum(
                        net_flow_per_seg[idx].get((side, price), 0)
                        for idx, _ in net_flow_allocations
                    )
                    if netflow_sum != n_total:
                        adjust_idx, adjust_value = net_flow_allocations[-1]
                        delta = n_total - netflow_sum
                        adjusted = adjust_value + delta
                        net_flow_per_seg[adjust_idx][(side, price)] = adjusted
                        if adjusted < 0:
                            cancels_per_seg[adjust_idx][(side, price)] = abs(adjusted)
                        else:
                            cancels_per_seg[adjust_idx].pop((side, price), None)

        result = []
        for i, seg in enumerate(segments):
            new_seg = replace(
                seg,
                cancels=cancels_per_seg[i],
                net_flow=net_flow_per_seg[i],
            )
            result.append(new_seg)

        return result
