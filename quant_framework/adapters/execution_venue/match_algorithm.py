"""SegmentBaseAlgorithm: 基于 Segment 的撮合算法，实现 FIFO 坐标轴模型。

核心模型：
- X_s(p,t): 队列前沿消耗坐标（trades + cancel_bias 贡献）
- Tail_s(p,t) = X_s(p,t) + Q^mkt_s(p,t): 队列尾部坐标
- Shadow order 占据 [pos, pos + qty)，当 X(t) >= pos + qty 时全部成交
- Zone-aware cancel 分配：public zone 按 CDF 分配 cancel，shadow zone 跳过
- Improvement 模式：订单价格优于市场最优时，使用对手方 trade_rate 推进
- Trade pause：improvement 模式下抑制同侧 trade 贡献，避免重复计
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ...core.data_structure import (
    NormalizedSnapshot,
    OrderReceipt,
    ShadowOrder,
    Side,
    TapeSegment,
)
from ...core.port import IIntervalModel, IMarketDataQuery, IMatchAlgorithm


logger = logging.getLogger(__name__)

EPSILON = 1e-12


@dataclass
class PriceLevelState:
    """单个价位的坐标轴状态（算法独有）。

    只保存 q_mkt（市场队列深度基础值）和 x_coord（X 坐标重置基线）。
    Shadow order 的队列信息通过 active_orders 参数传入，不在此处存储。
    """
    side: Side
    price: float
    q_mkt: float = 0.0
    x_coord: float = 0.0


class SegmentBaseAlgorithm(IMatchAlgorithm):
    """FIFO 坐标轴撮合算法。

    约束：
    - Shadow order 生命周期由 Simulator 维护，通过 active_orders 传入。
    - 算法持有 PriceLevelState（q_mkt, x_coord）和 segment buffer。
    - Trade pause intervals 作为算法内部状态维护。
    """

    def __init__(
        self,
        cancel_bias_k: float = 0.0,
        tape_builder: Optional[IIntervalModel] = None,
        market_data_query: Optional[IMarketDataQuery] = None,
    ) -> None:
        self.cancel_bias_k = float(cancel_bias_k)
        self._tape_builder = tape_builder
        self._market_data_query = market_data_query
        self._segment_buffer: List[TapeSegment] = []
        self._window_start: Optional[NormalizedSnapshot] = None
        self._window_end: Optional[NormalizedSnapshot] = None
        self._prev_snapshot: Optional[NormalizedSnapshot] = None
        self._pending_prev_snapshot: Optional[NormalizedSnapshot] = None
        self._levels: Dict[Tuple[Side, float], PriceLevelState] = {}
        self._trade_pause_intervals: Dict[Side, List[Tuple[int, int]]] = {
            Side.BUY: [],
            Side.SELL: [],
        }
        self._t_start = 0
        self._t_end = 0

    # ── IMatchAlgorithm 接口实现 ────────────────────────────────

    def set_market_data_query(self, market_data_query: IMarketDataQuery) -> None:
        self._market_data_query = market_data_query

    def start_run(self) -> None:
        self._segment_buffer = []
        self._window_start = None
        self._window_end = None
        self._prev_snapshot = None
        self._pending_prev_snapshot = None
        self._levels.clear()
        self._trade_pause_intervals[Side.BUY].clear()
        self._trade_pause_intervals[Side.SELL].clear()
        self._t_start = 0
        self._t_end = 0
        if self._market_data_query is not None:
            raw = self._market_data_query.query_data(1) or []
            first = self._decode_raw_md(raw[0]) if raw else None
            if first is not None:
                self._prev_snapshot = first

    def start_session(self) -> None:
        if self._pending_prev_snapshot is not None:
            self._prev_snapshot = self._pending_prev_snapshot
        self._pending_prev_snapshot = None
        self._segment_buffer = []
        self._window_start = None
        self._window_end = None
        self._trade_pause_intervals[Side.BUY].clear()
        self._trade_pause_intervals[Side.SELL].clear()
        self._t_start = 0
        self._t_end = 0
        if self._market_data_query is None:
            raise RuntimeError("SegmentBaseAlgorithm requires market_data_query.")
        if self._tape_builder is None:
            raise RuntimeError("SegmentBaseAlgorithm requires tape_builder.")

        raw_list = self._market_data_query.query_data(1) or []
        curr_snapshot = self._decode_raw_md(raw_list[0]) if raw_list else None
        if curr_snapshot is None:
            return

        if self._prev_snapshot is None:
            self._prev_snapshot = curr_snapshot
            return

        if int(curr_snapshot.ts_recv) <= int(self._prev_snapshot.ts_recv):
            return

        self._window_start = self._prev_snapshot
        self._window_end = curr_snapshot
        self._t_start = int(self._window_start.ts_recv)
        self._t_end = int(self._window_end.ts_recv)
        if self._t_end <= self._t_start:
            return

        self._segment_buffer = self._tape_builder.build(self._window_start, self._window_end)
        self._init_base_depth(self._window_start)
        for level in self._levels.values():
            level.x_coord = 0.0
        self._pending_prev_snapshot = curr_snapshot

    def on_order_action_impl(
        self,
        order: ShadowOrder,
        active_orders: Mapping[str, ShadowOrder],
    ) -> List[OrderReceipt]:
        t = int(order.create_time)
        qty = int(max(0, order.now_vol))
        if qty <= 0:
            return [self._none_receipt(t, order.order_id)]

        seg = self._segment_at_time(t)
        seg_idx = self._segment_index_at_time(t)

        opposite_best = self._get_opposite_best_price(order.side, seg)
        is_crossing = self._check_crossing(order.side, float(order.price), opposite_best)

        immediate_fill_qty = 0
        fill_price = 0.0

        if is_crossing and qty > 0:
            if self._has_blocking_shadow(order.side, float(order.price), active_orders):
                is_crossing = False
            elif self._same_side_queue_depth(order.side, float(order.price), t, active_orders) > 0:
                is_crossing = False
            else:
                immediate_fill_qty, fill_price = self._execute_crossing(
                    order.side, float(order.price), qty, t, seg
                )

        remaining = qty - immediate_fill_qty

        if immediate_fill_qty > 0 and remaining > 0:
            market_pos = self._compute_queue_position_post_crossing(
                order.side, float(order.price), t
            )
            return [
                OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="PARTIAL",
                    timestamp=t,
                    fill_qty=immediate_fill_qty,
                    fill_price=float(fill_price),
                    remaining_qty=remaining,
                    pos=market_pos,
                )
            ]

        if immediate_fill_qty > 0 and remaining == 0:
            return [
                OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="FILL",
                    timestamp=t,
                    fill_qty=immediate_fill_qty,
                    fill_price=float(fill_price),
                    remaining_qty=0,
                    pos=0,
                )
            ]

        market_pos = self._compute_queue_position(
            order.side, float(order.price), t, active_orders
        )
        return [
            OrderReceipt(
                order_id=order.order_id,
                receipt_type="NONE",
                timestamp=t,
                fill_qty=0,
                fill_price=float(order.price),
                remaining_qty=remaining,
                pos=market_pos,
            )
        ]

    def on_step(
        self,
        active_orders: Mapping[str, ShadowOrder],
        start_time: int,
        until_time: int,
    ) -> List[OrderReceipt]:
        t_from = int(start_time)
        t_to = int(until_time)
        if t_to <= t_from:
            return [self._none_receipt(t_from)]

        seg = self._segment_at_time(t_from)
        seg_idx = self._segment_index_at_time(t_from)
        if seg is None:
            return [self._none_receipt(t_to)]

        seg_duration = int(seg.t_end) - int(seg.t_start)
        if seg_duration <= 0:
            return [self._none_receipt(t_to)]

        seg_limit = min(t_to, int(seg.t_end))

        side_state: Dict[Side, Dict[str, object]] = {}
        full_fill_candidates: List[Tuple[int, Side]] = []

        for side in (Side.BUY, Side.SELL):
            mkt_best = float(seg.bid_price) if side == Side.BUY else float(seg.ask_price)
            best_active = self._get_best_active_price(side, t_from, active_orders)
            if best_active is None:
                side_state[side] = {}
                continue

            if side == Side.BUY:
                exec_best = max(mkt_best, best_active)
                improvement = exec_best > mkt_best + EPSILON
            else:
                exec_best = min(mkt_best, best_active)
                improvement = exec_best < mkt_best - EPSILON

            shadow = self._get_first_active_shadow(side, exec_best, t_from, active_orders)
            if shadow is None:
                side_state[side] = {}
                continue

            eligible = True
            if not improvement:
                if seg_idx < 0 or not self._is_price_active(seg, side, exec_best):
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
                trade_qty = self._seg_value(seg.trades, side, mkt_best)
                if trade_qty > 0:
                    trade_rate = trade_qty / seg_duration
                    if trade_rate > EPSILON:
                        t_fill = int(t_from + (shadow.now_vol / trade_rate))
                        t_fill = max(t_fill, t_from)
                        if t_fill <= seg_limit:
                            full_fill_candidates.append((t_fill, side))
            else:
                t_fill = self._compute_full_fill_time(
                    shadow, side, exec_best, seg, seg_idx,
                    t_from, seg_limit, active_orders,
                )
                if t_fill is not None and t_fill <= seg_limit:
                    full_fill_candidates.append((t_fill, side))

        receipts: List[OrderReceipt] = []

        if full_fill_candidates:
            t_stop = min(t_f for t_f, _ in full_fill_candidates)

            for side, state in side_state.items():
                if state.get("improvement"):
                    self._add_trade_pause(side, t_from, t_stop)

            for t_f, side in full_fill_candidates:
                if t_f != t_stop:
                    continue
                state = side_state.get(side, {})
                shadow = state.get("shadow")
                if shadow is None or shadow.now_vol <= 0:
                    continue
                receipts.append(OrderReceipt(
                    order_id=shadow.order_id,
                    receipt_type="FILL",
                    timestamp=t_stop,
                    fill_qty=int(shadow.now_vol),
                    fill_price=float(shadow.price),
                    remaining_qty=0,
                    pos=int(shadow.pos),
                ))

            if not receipts:
                return [self._none_receipt(t_stop)]
            return receipts

        t_stop = seg_limit

        for side, state in side_state.items():
            if state.get("improvement"):
                self._add_trade_pause(side, t_from, t_stop)

        for side, state in side_state.items():
            if not state or not state.get("eligible"):
                continue

            shadow = state["shadow"]
            if shadow.now_vol <= 0:
                continue

            if state.get("improvement"):
                mkt_best_price = state["mkt_best"]
                trade_qty = self._seg_value(seg.trades, side, mkt_best_price)
                if trade_qty <= 0:
                    continue
                trade_rate = trade_qty / seg_duration
                if trade_rate <= EPSILON:
                    continue
                virtual_volume = trade_rate * (t_stop - t_from)
                fill_qty = min(int(shadow.now_vol), int(virtual_volume))
                if fill_qty <= 0:
                    continue
                remain = int(shadow.now_vol) - fill_qty
                receipts.append(OrderReceipt(
                    order_id=shadow.order_id,
                    receipt_type="FILL" if remain == 0 else "PARTIAL",
                    timestamp=t_stop,
                    fill_qty=fill_qty,
                    fill_price=float(shadow.price),
                    remaining_qty=remain,
                    pos=int(shadow.pos),
                ))
            else:
                exec_best = state["exec_best"]
                x_to = self._get_x_coord(side, exec_best, t_stop, shadow, active_orders)
                current_fill = max(0, int(x_to) - int(shadow.pos))
                if current_fill > int(shadow.init_vol):
                    current_fill = int(shadow.init_vol)
                already_reported = int(shadow.init_vol) - int(shadow.now_vol)
                new_fill = current_fill - already_reported
                if new_fill <= 0:
                    continue
                if new_fill > int(shadow.now_vol):
                    new_fill = int(shadow.now_vol)
                remain = int(shadow.now_vol) - new_fill
                receipts.append(OrderReceipt(
                    order_id=shadow.order_id,
                    receipt_type="FILL" if remain == 0 else "PARTIAL",
                    timestamp=t_stop,
                    fill_qty=new_fill,
                    fill_price=float(shadow.price),
                    remaining_qty=remain,
                    pos=int(shadow.pos),
                ))

        if not receipts:
            return [self._none_receipt(t_stop)]
        return receipts

    def flush_window(self) -> object:
        if self._pending_prev_snapshot is not None:
            self._prev_snapshot = self._pending_prev_snapshot
        self._pending_prev_snapshot = None
        return {
            "interval_start": self._t_start,
            "interval_end": self._t_end,
            "segment_count": len(self._segment_buffer),
        }

    # ── PriceLevelState 管理 ─────────────────────────────────────

    def _get_level(self, side: Side, price: float) -> PriceLevelState:
        key = (side, round(float(price), 8))
        if key not in self._levels:
            self._levels[key] = PriceLevelState(side=side, price=float(price))
        return self._levels[key]

    def _init_base_depth(self, snapshot: NormalizedSnapshot) -> None:
        for lvl in snapshot.bids:
            level = self._get_level(Side.BUY, float(lvl.price))
            level.q_mkt = float(lvl.qty)
        for lvl in snapshot.asks:
            level = self._get_level(Side.SELL, float(lvl.price))
            level.q_mkt = float(lvl.qty)

    # ── Segment 辅助 ─────────────────────────────────────────────

    @staticmethod
    def _norm_price(price: float) -> float:
        return round(float(price), 8)

    def _seg_value(self, mapping: Dict[Tuple[Side, float], int], side: Side, price: float) -> int:
        p = self._norm_price(price)
        direct = mapping.get((side, p))
        if direct is not None:
            return int(direct)
        for (s, mp), value in mapping.items():
            if s == side and abs(float(mp) - p) <= 1e-8:
                return int(value)
        return 0

    def _is_price_active(self, seg: TapeSegment, side: Side, price: float) -> bool:
        p = self._norm_price(price)
        active = seg.activation_bid if side == Side.BUY else seg.activation_ask
        return any(abs(float(a) - p) <= 1e-8 for a in active)

    def _segment_at_time(self, t: int) -> Optional[TapeSegment]:
        if not self._segment_buffer:
            return None
        for seg in self._segment_buffer:
            if int(seg.t_start) <= t < int(seg.t_end):
                return seg
        if t >= int(self._segment_buffer[-1].t_end):
            return self._segment_buffer[-1]
        return self._segment_buffer[0]

    def _segment_index_at_time(self, t: int) -> int:
        for i, seg in enumerate(self._segment_buffer):
            if int(seg.t_start) <= t < int(seg.t_end):
                return i
        return -1

    def _decode_raw_md(self, raw: Any) -> Optional[NormalizedSnapshot]:
        if isinstance(raw, NormalizedSnapshot):
            return raw
        if hasattr(raw, "ts_recv") and hasattr(raw, "bids") and hasattr(raw, "asks"):
            return raw  # type: ignore[return-value]
        return None

    # ── Q_mkt 插值 ──────────────────────────────────────────────

    def _get_q_mkt(self, side: Side, price: float, t: int) -> float:
        level = self._get_level(side, price)
        if not self._segment_buffer or t <= self._t_start:
            return max(0.0, level.q_mkt)

        q = level.q_mkt
        for seg_idx, seg in enumerate(self._segment_buffer):
            if t <= int(seg.t_start):
                break
            seg_duration = int(seg.t_end) - int(seg.t_start)
            if seg_duration <= 0:
                continue
            if not self._is_price_active(seg, side, price):
                continue
            seg_start = max(int(seg.t_start), self._t_start)
            seg_end = min(int(seg.t_end), int(t))
            if seg_end <= seg_start:
                continue
            z = (seg_end - int(seg.t_start)) / seg_duration
            z = min(1.0, max(0.0, z))
            n_si = self._seg_value(seg.net_flow, side, price)
            m_si = self._seg_value(seg.trades, side, price)
            q += (n_si - m_si) * z
            if t <= int(seg.t_end):
                break
        return max(0.0, q)

    def _get_positive_netflow_between(
        self, side: Side, price: float, t_from: int, t_to: int
    ) -> float:
        if not self._segment_buffer or t_to <= t_from:
            return 0.0
        total = 0.0
        for seg in self._segment_buffer:
            if t_to <= int(seg.t_start):
                break
            if t_from >= int(seg.t_end):
                continue
            seg_duration = int(seg.t_end) - int(seg.t_start)
            if seg_duration <= 0:
                continue
            if not self._is_price_active(seg, side, price):
                continue
            overlap_start = max(int(seg.t_start), t_from)
            overlap_end = min(int(seg.t_end), t_to)
            if overlap_end <= overlap_start:
                continue
            z = (overlap_end - overlap_start) / seg_duration
            n_si = self._seg_value(seg.net_flow, side, price)
            if n_si > 0:
                total += n_si * z
        return total

    # ── Cancel bias 模型 ────────────────────────────────────────

    def _compute_cancel_front_prob(self, x: float) -> float:
        x = max(0.0, min(1.0, x))
        k = self.cancel_bias_k
        if k < 0:
            exponent = 1.0 + k
            if exponent <= 0:
                return 1.0 if x > 0 else 0.0
            return x ** exponent
        elif k == 0:
            return x
        else:
            exponent = 1.0 - k
            if exponent <= 0:
                return 0.0 if x < 1 else 1.0
            return 1.0 - (1.0 - x) ** exponent

    # ── Trade pause ─────────────────────────────────────────────

    def _add_trade_pause(self, side: Side, t_start: int, t_end: int) -> None:
        if t_end <= t_start:
            return
        intervals = self._trade_pause_intervals[side]
        if intervals and t_start <= intervals[-1][1]:
            last_start, last_end = intervals[-1]
            intervals[-1] = (last_start, max(last_end, t_end))
        else:
            intervals.append((t_start, t_end))

    def _get_trade_active_duration(self, side: Side, t_start: int, t_end: int) -> float:
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

    # ── Crossing 检测与执行 ─────────────────────────────────────

    @staticmethod
    def _get_opposite_best_price(side: Side, seg: Optional[TapeSegment]) -> Optional[float]:
        if seg is None:
            return None
        if side == Side.BUY:
            return float(seg.ask_price)
        return float(seg.bid_price)

    @staticmethod
    def _check_crossing(side: Side, order_price: float, opposite_best: Optional[float]) -> bool:
        if opposite_best is None:
            return False
        if side == Side.BUY:
            return order_price >= opposite_best
        return order_price <= opposite_best

    def _has_blocking_shadow(
        self, side: Side, price: float, active_orders: Mapping[str, ShadowOrder]
    ) -> bool:
        price = round(float(price), 8)
        for shadow in active_orders.values():
            if shadow.side != side or shadow.now_vol <= 0:
                continue
            sp = round(float(shadow.price), 8)
            if side == Side.BUY and sp > price:
                return True
            if side == Side.SELL and sp < price:
                return True
        return False

    def _same_side_queue_depth(
        self, side: Side, price: float, t: int, active_orders: Mapping[str, ShadowOrder]
    ) -> float:
        q_mkt = self._get_q_mkt(side, price, t)
        shadow_qty = sum(
            int(s.now_vol) for s in active_orders.values()
            if s.side == side
            and abs(float(s.price) - price) < EPSILON
            and int(s.now_vol) > 0
        )
        return q_mkt + shadow_qty

    def _execute_crossing(
        self, side: Side, order_price: float, order_qty: int, t: int,
        seg: Optional[TapeSegment],
    ) -> Tuple[int, float]:
        if seg is None:
            return 0, 0.0

        opposite_side = Side.SELL if side == Side.BUY else Side.BUY

        if side == Side.BUY:
            crossable = sorted(
                [p for p in seg.activation_ask if p <= order_price + EPSILON]
            )
        else:
            crossable = sorted(
                [p for p in seg.activation_bid if p >= order_price - EPSILON],
                reverse=True,
            )

        remaining = order_qty
        total_fill = 0
        total_value = 0.0

        for cross_price in crossable:
            if remaining <= 0:
                break
            available = self._get_q_mkt(opposite_side, cross_price, t)
            if available <= 0:
                continue
            fill = min(remaining, int(available))
            if fill > 0:
                total_fill += fill
                total_value += fill * cross_price
                remaining -= fill

        avg_price = total_value / total_fill if total_fill > 0 else 0.0
        return total_fill, avg_price

    # ── 队列位置计算 ────────────────────────────────────────────

    def _compute_queue_position_post_crossing(
        self, side: Side, price: float, t: int
    ) -> int:
        return 0

    def _compute_queue_position(
        self, side: Side, price: float, t: int,
        active_orders: Mapping[str, ShadowOrder],
    ) -> int:
        last_shadow = None
        for s in active_orders.values():
            if (s.side == side
                    and abs(float(s.price) - price) < EPSILON
                    and int(s.now_vol) > 0
                    and int(s.create_time) < t):
                if last_shadow is None or int(s.pos) + int(s.now_vol) > int(last_shadow.pos) + int(last_shadow.now_vol):
                    last_shadow = s

        if last_shadow is not None:
            prev_threshold = int(last_shadow.pos) + int(last_shadow.now_vol)
            positive_netflow = self._get_positive_netflow_between(
                side, price, int(last_shadow.create_time), t
            )
            return int(round(prev_threshold + positive_netflow))

        q_mkt = self._get_q_mkt(side, price, t)
        return int(round(q_mkt))

    # ── Zone-aware X 坐标计算 ───────────────────────────────────

    def _get_shadows_at_price(
        self, side: Side, price: float, active_orders: Mapping[str, ShadowOrder]
    ) -> List[ShadowOrder]:
        return sorted(
            [s for s in active_orders.values()
             if s.side == side
             and abs(float(s.price) - price) < EPSILON
             and int(s.now_vol) > 0],
            key=lambda s: int(s.pos),
        )

    def _build_queue_zones(
        self, side: Side, price: float, x_running: float, q_mkt: float,
        active_orders: Mapping[str, ShadowOrder],
    ) -> List[Tuple[str, float, float, Optional[ShadowOrder]]]:
        shadows = self._get_shadows_at_price(side, price, active_orders)
        zones: List[Tuple[str, float, float, Optional[ShadowOrder]]] = []
        cursor = x_running

        for shadow in shadows:
            shadow_start = float(shadow.pos)
            shadow_end = float(shadow.pos + shadow.now_vol)

            if cursor >= shadow_end - EPSILON:
                continue

            if cursor >= shadow_start - EPSILON:
                remaining = shadow_end - cursor
                if remaining > EPSILON:
                    zones.append(("shadow", cursor, remaining, shadow))
                cursor = shadow_end
                continue

            pub_size = shadow_start - cursor
            if pub_size > EPSILON:
                zones.append(("public", cursor, pub_size, None))

            shadow_size = shadow_end - shadow_start
            if shadow_size > EPSILON:
                zones.append(("shadow", shadow_start, shadow_size, shadow))
            cursor = shadow_end

        total_pub = sum(sz for zt, _, sz, _ in zones if zt == "public")
        trailing = max(0.0, q_mkt - total_pub)
        if trailing > EPSILON:
            zones.append(("public", cursor, trailing, None))

        return zones

    def _distribute_cancels_to_zones(
        self,
        zones: List[Tuple[str, float, float, Optional[ShadowOrder]]],
        total_cancels: float,
        q_mkt: float,
    ) -> Dict[int, float]:
        if total_cancels <= EPSILON or q_mkt <= EPSILON:
            return {}

        pub_info = [
            (i, sz) for i, (zt, _, sz, _) in enumerate(zones) if zt == "public"
        ]
        if not pub_info:
            return {}

        sizes = [sz for _, sz in pub_info]
        cum = [0.0]
        for s in sizes:
            cum.append(cum[-1] + s)

        raw_shares: List[float] = []
        for j in range(len(sizes)):
            x_lo = min(1.0, max(0.0, cum[j] / q_mkt))
            x_hi = min(1.0, max(0.0, cum[j + 1] / q_mkt))
            p_lo = self._compute_cancel_front_prob(x_lo)
            p_hi = self._compute_cancel_front_prob(x_hi)
            raw_shares.append(total_cancels * max(0.0, p_hi - p_lo))

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
        dt_available: float,
        target_x: Optional[float] = None,
    ) -> Tuple[float, float]:
        x = x_start
        t_elapsed = 0.0

        for i, (ztype, zone_start, zone_size, _shadow) in enumerate(zones):
            remaining_dt = dt_available - t_elapsed
            if remaining_dt <= EPSILON:
                break
            if target_x is not None and x >= target_x - EPSILON:
                break

            zone_end = zone_start + zone_size

            if x >= zone_end - EPSILON:
                x = max(x, zone_end)
                continue

            if ztype == "public":
                cancel_count = cancel_dist.get(i, 0.0)
                cancel_rate = cancel_count / seg_duration if seg_duration > EPSILON else 0.0
                consumption_rate = trade_rate + cancel_rate
                if consumption_rate <= EPSILON:
                    break

                effective_zone = max(0.0, zone_end - x)
                if target_x is not None and x < target_x <= zone_end + EPSILON:
                    dt_to_target = max(0.0, target_x - x) / consumption_rate
                    if dt_to_target <= remaining_dt + EPSILON:
                        t_elapsed += dt_to_target
                        x = target_x
                        break

                dt_to_clear = effective_zone / consumption_rate
                if dt_to_clear <= remaining_dt + EPSILON:
                    x = zone_end
                    t_elapsed += dt_to_clear
                else:
                    x += consumption_rate * remaining_dt
                    t_elapsed = dt_available
                    break
            else:
                if trade_rate <= EPSILON:
                    break
                effective_zone = max(0.0, zone_end - x)
                if effective_zone <= EPSILON:
                    x = zone_end
                    continue

                if target_x is not None and x < target_x <= zone_end + EPSILON:
                    dt_to_target = max(0.0, target_x - x) / trade_rate
                    if dt_to_target <= remaining_dt + EPSILON:
                        t_elapsed += dt_to_target
                        x = target_x
                        break

                dt_to_clear = effective_zone / trade_rate
                if dt_to_clear <= remaining_dt + EPSILON:
                    x = zone_end
                    t_elapsed += dt_to_clear
                else:
                    x += trade_rate * remaining_dt
                    t_elapsed = dt_available
                    break

        return x, t_elapsed

    def _get_x_coord(
        self, side: Side, price: float, t: int,
        ref_shadow: ShadowOrder,
        active_orders: Mapping[str, ShadowOrder],
    ) -> float:
        level = self._get_level(side, price)
        if not self._segment_buffer or t <= self._t_start:
            return level.x_coord

        x = level.x_coord
        for seg_idx, seg in enumerate(self._segment_buffer):
            if t <= int(seg.t_start):
                break
            seg_start = max(int(seg.t_start), self._t_start)
            seg_end = min(int(seg.t_end), int(t))
            if seg_end <= seg_start:
                continue
            if not self._is_price_active(seg, side, price):
                continue

            seg_duration = int(seg.t_end) - int(seg.t_start)
            if seg_duration <= 0:
                continue

            total_cancels = self._seg_value(seg.cancels, side, price)
            total_trades = self._seg_value(seg.trades, side, price)
            q_mkt = self._get_q_mkt(side, price, int(seg.t_start))

            dt = seg_end - seg_start
            trade_rate_base = total_trades / seg_duration
            trade_active = self._get_trade_active_duration(side, seg_start, seg_end)
            effective_trade_rate = (
                trade_rate_base * trade_active / dt if dt > 0 else 0.0
            )

            zones = self._build_queue_zones(side, price, x, q_mkt, active_orders)
            cancel_dist = self._distribute_cancels_to_zones(zones, total_cancels, q_mkt)
            x, _ = self._traverse_zones_for_x(
                x, zones, cancel_dist, effective_trade_rate, seg_duration, float(dt),
            )

            if t <= int(seg.t_end):
                break
        return x

    # ── Fill 时间计算 ───────────────────────────────────────────

    def _compute_full_fill_time(
        self,
        shadow: ShadowOrder,
        side: Side,
        price: float,
        seg: TapeSegment,
        seg_idx: int,
        t_from: int,
        t_to: int,
        active_orders: Mapping[str, ShadowOrder],
    ) -> Optional[int]:
        if t_to <= t_from or seg_idx < 0:
            return None
        if not self._is_price_active(seg, side, price):
            return None

        seg_duration = int(seg.t_end) - int(seg.t_start)
        if seg_duration <= 0:
            return None

        x_start = self._get_x_coord(side, price, t_from, shadow, active_orders)
        threshold = shadow.pos + shadow.init_vol
        if x_start >= threshold - EPSILON:
            return t_from

        total_cancels = self._seg_value(seg.cancels, side, price)
        total_trades = self._seg_value(seg.trades, side, price)
        q_mkt = self._get_q_mkt(side, price, int(seg.t_start))
        trade_rate = total_trades / seg_duration if seg_duration > 0 else 0.0

        if trade_rate <= EPSILON and total_cancels <= EPSILON:
            return None

        zones = self._build_queue_zones(side, price, x_start, q_mkt, active_orders)
        cancel_dist = self._distribute_cancels_to_zones(zones, total_cancels, q_mkt)

        dt_available = float(t_to - t_from)
        x_end, time_elapsed = self._traverse_zones_for_x(
            x_start, zones, cancel_dist, trade_rate, seg_duration, dt_available,
            target_x=threshold,
        )

        if x_end >= threshold - EPSILON and time_elapsed <= dt_available + EPSILON:
            fill_time = int(t_from + time_elapsed)
            fill_time = max(fill_time, t_from)
            if fill_time > t_to:
                return None
            return fill_time
        return None

    # ── Active orders 查询 ──────────────────────────────────────

    @staticmethod
    def _get_best_active_price(
        side: Side, t_from: int, active_orders: Mapping[str, ShadowOrder]
    ) -> Optional[float]:
        best: Optional[float] = None
        for s in active_orders.values():
            if s.side != side or int(s.now_vol) <= 0 or int(s.create_time) > t_from:
                continue
            p = float(s.price)
            if best is None:
                best = p
            elif side == Side.BUY and p > best:
                best = p
            elif side == Side.SELL and p < best:
                best = p
        return best

    @staticmethod
    def _get_first_active_shadow(
        side: Side, price: float, t_from: int,
        active_orders: Mapping[str, ShadowOrder],
    ) -> Optional[ShadowOrder]:
        best: Optional[ShadowOrder] = None
        for s in active_orders.values():
            if (s.side == side
                    and abs(float(s.price) - price) < EPSILON
                    and int(s.now_vol) > 0
                    and int(s.create_time) <= t_from):
                if best is None or int(s.pos) < int(best.pos):
                    best = s
        return best

    # ── 通用辅助 ────────────────────────────────────────────────

    @staticmethod
    def _none_receipt(timestamp: int, order_id: str = "") -> OrderReceipt:
        return OrderReceipt(
            order_id=order_id,
            receipt_type="NONE",
            timestamp=int(timestamp),
            fill_qty=0,
            fill_price=0.0,
            remaining_qty=0,
            pos=0,
        )
