"""Queue-zero constraint resolver for tape segments."""

from dataclasses import replace
from typing import Dict, List, Tuple, Set

from ..core.types import NormalizedSnapshot, Price, Qty, Side, TapeSegment

EPSILON = 1e-12


class QueueConstraintResolver:
    """Apply queue-zero constraints to derive net flow and cancellations."""

    def __init__(self, epsilon: float = EPSILON):
        self.epsilon = epsilon

    def derive_cancellations_and_net_flow(
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

        eps = self.epsilon

        price_universe_bid: Set[Price] = set()
        price_universe_ask: Set[Price] = set()
        for seg in segments:
            price_universe_bid.update(seg.activation_bid)
            price_universe_ask.update(seg.activation_ask)

        def get_qty_at_price(snap: NormalizedSnapshot, side: Side, price: Price) -> int:
            levels = snap.bids if side == Side.BUY else snap.asks
            for lvl in levels:
                if abs(float(lvl.price) - price) < eps:
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
                while i < n and abs(best_prices[i] - price) < eps:
                    i += 1
                end = i - 1  # 包含的最后一个段索引

                # 判断是否是价格转换（bid下降或ask上升表示队列清空）
                ends_with_transition = False
                if i < n:  # 还有后续段
                    next_price = best_prices[i]
                    if side == Side.BUY and next_price < price - eps:
                        # bid价格下降，说明当前价位队列清空
                        ends_with_transition = True
                    elif side == Side.SELL and next_price > price + eps:
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
