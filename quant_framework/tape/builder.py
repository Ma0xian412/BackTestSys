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
    
    # 非均匀快照推送配置
    # Snapshot推送最小间隔（毫秒）：快照实际上是在T_B-min_interval_ms到T_B之间产生的变化
    # 所以A快照的时间被视为T_B - min_interval_ms
    snapshot_min_interval_ms: int = 500


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
        
        从A/B快照构建tape段。
        
        非均匀快照时间处理（默认启用）：
        快照只在关键字段变化时推送，最小间隔500ms。
        将A快照的时间视为T_B - 500ms，所有变化归因到[T_B-500ms, T_B]区间。
        
        Args:
            prev: Previous snapshot (A) at time T_A / 前一个快照（A），在T_A时刻
            curr: Current snapshot (B) at time T_B / 当前快照（B），在T_B时刻
            
        Returns:
            List of TapeSegments ordered by time / 按时间排序的TapeSegment列表
            
        Raises:
            ValueError: 当 t_b <= t_a 时抛出，因为快照时间必须严格递增
        """
        t_a = int(prev.ts_exch)
        t_b = int(curr.ts_exch)
        
        # 快照时间必须严格递增，否则抛出ValueError
        # Snapshot timestamps must be strictly increasing
        if t_b <= t_a:
            raise ValueError(
                f"Snapshot timestamps must be strictly increasing: t_b ({t_b}) <= t_a ({t_a}). "
                f"快照时间必须严格递增。"
            )
        
        # 非均匀快照时间处理（默认启用）
        # 将A快照的时间视为T_B - min_interval_ms
        # 这样所有变化都归因到最后min_interval_ms内
        min_interval = self.config.snapshot_min_interval_ms
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
        
        # Two-round iterative volume allocation
        segments = self._allocate_volumes_iterative(segments, e_bid, e_ask, effective_t_a, t_b)
        
        # Derive cancellations and net flow using queue-zero constraint at price transitions
        segments = self._derive_cancellations_and_net_flow(segments, prev, curr)
        
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
    
    def _build_path_with_meeting_sequence(self, p_start: Price, p_end: Price, 
                                           meeting_seq: List[Price]) -> List[Price]:
        """使用公共meeting序列构建完整的价格路径。
        
        构建规则：
        - bid_path = [bidA] + meeting_seq + [bidB]
        - ask_path = [askA] + meeting_seq + [askB]
        
        这样可以确保bid和ask路径的中间部分完全一致，只有起点和终点不同。
        
        注意：当价格从一个点移动到下一个点时，会包含中间的所有价位（按tick_size）。
        例如：从3318到3316会包含3317作为中间步骤。
        
        Args:
            p_start: 起始价格（bidA或askA）
            p_end: 结束价格（bidB或askB）
            meeting_seq: 公共相遇价位序列
            
        Returns:
            完整的价格路径
        """
        result = [p_start]
        
        # 添加meeting序列，包含中间价位
        for p in meeting_seq:
            # 只添加与上一个价位不同的点
            if abs(result[-1] - p) > EPSILON:
                # 插入中间价位
                self._add_intermediate_prices(result, result[-1], p)
                result.append(round(p, 8))
        
        # 添加终点，包含中间价位
        if abs(result[-1] - p_end) > EPSILON:
            self._add_intermediate_prices(result, result[-1], p_end)
            result.append(round(p_end, 8))
        
        return result
    
    def _build_aligned_paths(
        self, 
        bid_a: Price, bid_b: Price,
        ask_a: Price, ask_b: Price,
        meeting_seq: List[Price]
    ) -> Tuple[List[Price], List[Price]]:
        """构建对齐的bid和ask价格路径。
        
        确保bid_path和ask_path具有完全相同的长度，使得每个segment中bid和ask的
        成交价格和成交数量能够正确对齐。
        
        核心思想：
        - 两条路径都保留起点和终点
        - 中间经过meeting序列（成交价位）
        - 通过padding确保两条路径长度相同
        - 当ask价格高于meeting价位时，ask在该价位的队列深度为0（taker）
        
        示例：
        - prev: bid1=3317, ask1=3319
        - curr: bid1=3317, ask1=3318
        - meeting_seq: [3318]
        
        结果：
        - bid_path: [3317, 3318, 3317]  -- 起点 -> meeting -> 终点
        - ask_path: [3319, 3318, 3318]  -- 起点 -> meeting -> 终点
        
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
        
        # 构建保留起点和终点的统一路径
        # 路径结构：[start] -> [meeting_1] -> [meeting_2] -> ... -> [meeting_n] -> [end]
        
        bid_path = [round(bid_a, 8)]
        ask_path = [round(ask_a, 8)]
        
        # 添加所有meeting价位到两条路径
        # 注意：meeting序列对bid和ask都是相同的
        for meeting_price in meeting_seq:
            rounded_price = round(meeting_price, 8)
            # 避免连续重复
            if abs(bid_path[-1] - rounded_price) > EPSILON:
                bid_path.append(rounded_price)
            if abs(ask_path[-1] - rounded_price) > EPSILON:
                ask_path.append(rounded_price)
        
        # 添加终点（如果与最后一个价位不同）
        # Bid终点
        if abs(bid_path[-1] - bid_b) > EPSILON:
            bid_path.append(round(bid_b, 8))
        
        # Ask终点
        if abs(ask_path[-1] - ask_b) > EPSILON:
            ask_path.append(round(ask_b, 8))
        
        # 确保路径长度相同（通过填充最后一个价格）
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
    
    def _add_intermediate_prices(self, result: List[Price], p_from: Price, p_to: Price) -> None:
        """在result中添加从p_from到p_to之间的中间价位（不包含p_from和p_to本身）。
        
        例如：p_from=3318, p_to=3316, tick_size=1.0
        则添加 3317 到result中
        
        Args:
            result: 价格路径列表，会被修改
            p_from: 起始价格
            p_to: 目标价格
        """
        if abs(p_to - p_from) <= self.tick_size + EPSILON:
            # 相邻价位，无需添加中间点
            return
        
        if p_to > p_from:
            # 向上移动：从p_from + tick_size到p_to - tick_size
            current = p_from + self.tick_size
            while current < p_to - EPSILON:
                result.append(round(current, 8))
                current += self.tick_size
        else:
            # 向下移动：从p_from - tick_size到p_to + tick_size
            current = p_from - self.tick_size
            while current > p_to + EPSILON:
                result.append(round(current, 8))
                current -= self.tick_size
    
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
    
    def _derive_cancellations_and_net_flow(
        self,
        segments: List[TapeSegment],
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
    ) -> List[TapeSegment]:
        """使用队列清空约束来分配净流入量和撤单量。

        核心约束：当价格从P1变化到P2时（bid下降或ask上升），
        P1在变化时刻的队列深度必须为0。

        算法：
        1. 识别每个价位作为best price的连续段组
        2. 对于价格转换点，应用约束：Q_A + N_total - M_total = 0
           即 N_total = M_total - Q_A（在该价位作为best price期间）
        3. 对于最后仍是best price的价位，使用守恒方程：N = delta_Q + M
        4. 按段时长比例分配N到各段
        5. 如果N < 0，计算撤单量
        """
        n = len(segments)
        if n == 0:
            return segments

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

        cancels_per_seg: List[Dict[Tuple[Side, Price], Qty]] = [{} for _ in range(n)]
        net_flow_per_seg: List[Dict[Tuple[Side, Price], Qty]] = [{} for _ in range(n)]

        # 找出价格转换点
        def find_price_transition_segments(side: Side) -> Dict[Price, List[Tuple[int, int, bool]]]:
            """找出每个价位作为best price的连续段范围。
            
            返回: {price: [(start_idx, end_idx, ends_with_transition), ...]}
            ends_with_transition: True表示该段组结束时价格转换（队列清空），False表示保持或最终段
            """
            result: Dict[Price, List[Tuple[int, int, bool]]] = {}
            
            if not segments:
                return result
            
            # 获取每段的best price
            best_prices = []
            for seg in segments:
                if side == Side.BUY:
                    best_prices.append(seg.bid_price)
                else:
                    best_prices.append(seg.ask_price)
            
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
                        # bid价格下降，说明当前价位队列清空
                        ends_with_transition = True
                    elif side == Side.SELL and next_price > price + EPSILON:
                        # ask价格上升，说明当前价位队列清空
                        ends_with_transition = True
                
                if price not in result:
                    result[price] = []
                result[price].append((start, end, ends_with_transition))
            
            return result
        
        # 处理bid侧
        bid_transitions = find_price_transition_segments(Side.BUY)
        for price, groups in bid_transitions.items():
            q_a = get_qty_at_price(prev, Side.BUY, price)
            q_b = get_qty_at_price(curr, Side.BUY, price)
            
            for group_idx, (start_idx, end_idx, ends_with_transition) in enumerate(groups):
                # 计算这组段中在该价位的总成交量
                m_group = sum(
                    segments[i].trades.get((Side.BUY, price), 0)
                    for i in range(start_idx, end_idx + 1)
                )
                
                # 判断是否是首次访问该价位
                # 首次访问：使用Q_A作为初始队列
                # 重访：初始队列为0（因为之前离开时已归零）
                is_first_visit = (group_idx == 0)
                initial_queue = q_a if is_first_visit else 0
                
                # 计算净流入总量
                if ends_with_transition:
                    # 队列清空约束：Q_initial + N - M = 0 => N = M - Q_initial
                    n_group = m_group - initial_queue
                else:
                    # 使用守恒方程：N = delta_Q + M
                    # 但这只适用于最后一组（价位在区间结束时仍为best price）
                    # 对于中间不转换的组，暂时按比例分配
                    m_total_at_price = sum(
                        seg.trades.get((Side.BUY, price), 0) for seg in segments
                    )
                    delta_q = q_b - q_a
                    n_total = delta_q + m_total_at_price
                    
                    # 按这组段在总激活时长中的比例分配
                    all_active_segs = [
                        i for i, seg in enumerate(segments)
                        if price in seg.activation_bid
                    ]
                    group_segs = list(range(start_idx, end_idx + 1))
                    
                    group_dur = sum(segments[i].t_end - segments[i].t_start for i in group_segs)
                    total_dur = sum(segments[i].t_end - segments[i].t_start for i in all_active_segs) or 1
                    
                    n_group = n_total * group_dur / total_dur
                
                # 在组内按段时长比例分配
                group_segs = list(range(start_idx, end_idx + 1))
                durations = [segments[i].t_end - segments[i].t_start for i in group_segs]
                total_dur = sum(durations) or 1
                
                for j, seg_idx in enumerate(group_segs):
                    if price in segments[seg_idx].activation_bid:
                        alloc = n_group * durations[j] / total_dur
                        net_flow_per_seg[seg_idx][(Side.BUY, price)] = int(round(alloc))
                        
                        if alloc < 0:
                            cancels_per_seg[seg_idx][(Side.BUY, price)] = int(round(abs(alloc)))
        
        # 处理ask侧（类似逻辑）
        ask_transitions = find_price_transition_segments(Side.SELL)
        for price, groups in ask_transitions.items():
            q_a = get_qty_at_price(prev, Side.SELL, price)
            q_b = get_qty_at_price(curr, Side.SELL, price)
            
            for group_idx, (start_idx, end_idx, ends_with_transition) in enumerate(groups):
                m_group = sum(
                    segments[i].trades.get((Side.SELL, price), 0)
                    for i in range(start_idx, end_idx + 1)
                )
                
                # 判断是否是首次访问该价位
                is_first_visit = (group_idx == 0)
                initial_queue = q_a if is_first_visit else 0
                
                if ends_with_transition:
                    # 队列清空约束：Q_initial + N - M = 0 => N = M - Q_initial
                    n_group = m_group - initial_queue
                else:
                    m_total_at_price = sum(
                        seg.trades.get((Side.SELL, price), 0) for seg in segments
                    )
                    delta_q = q_b - q_a
                    n_total = delta_q + m_total_at_price
                    
                    all_active_segs = [
                        i for i, seg in enumerate(segments)
                        if price in seg.activation_ask
                    ]
                    group_segs = list(range(start_idx, end_idx + 1))
                    
                    group_dur = sum(segments[i].t_end - segments[i].t_start for i in group_segs)
                    total_dur = sum(segments[i].t_end - segments[i].t_start for i in all_active_segs) or 1
                    
                    n_group = n_total * group_dur / total_dur
                
                group_segs = list(range(start_idx, end_idx + 1))
                durations = [segments[i].t_end - segments[i].t_start for i in group_segs]
                total_dur = sum(durations) or 1
                
                for j, seg_idx in enumerate(group_segs):
                    if price in segments[seg_idx].activation_ask:
                        alloc = n_group * durations[j] / total_dur
                        net_flow_per_seg[seg_idx][(Side.SELL, price)] = int(round(alloc))
                        
                        if alloc < 0:
                            cancels_per_seg[seg_idx][(Side.SELL, price)] = int(round(abs(alloc)))
        
        # 处理非best-price但在activation中的价位（使用原始守恒方程）
        for side, price_universe in [(Side.BUY, price_universe_bid), (Side.SELL, price_universe_ask)]:
            transitions = bid_transitions if side == Side.BUY else ask_transitions
            
            for price in price_universe:
                # 跳过已处理的best-price价位
                if price in transitions:
                    continue
                
                q_a = get_qty_at_price(prev, side, price)
                q_b = get_qty_at_price(curr, side, price)
                
                m_total = sum(seg.trades.get((side, price), 0) for seg in segments)
                delta_q = q_b - q_a
                n_total = delta_q + m_total
                
                active_segs = [
                    i for i, seg in enumerate(segments)
                    if price in (seg.activation_bid if side == Side.BUY else seg.activation_ask)
                ]
                if not active_segs:
                    continue
                
                durations = [segments[i].t_end - segments[i].t_start for i in active_segs]
                total_dur = sum(durations) or 1
                
                for j, i in enumerate(active_segs):
                    alloc_net = n_total * durations[j] / total_dur
                    if (side, price) not in net_flow_per_seg[i]:
                        net_flow_per_seg[i][(side, price)] = int(round(alloc_net))
                    
                    if alloc_net < 0:
                        alloc_cancel = int(round(abs(alloc_net)))
                        if alloc_cancel > 0 and (side, price) not in cancels_per_seg[i]:
                            cancels_per_seg[i][(side, price)] = alloc_cancel

        result = []
        for i, seg in enumerate(segments):
            new_seg = replace(
                seg,
                cancels=cancels_per_seg[i],
                net_flow=net_flow_per_seg[i],
            )
            result.append(new_seg)

        return result
