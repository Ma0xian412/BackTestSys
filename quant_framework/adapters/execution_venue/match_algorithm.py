"""SegmentBaseAlgorithm: 基于 Segment 的无状态计算算法。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from ...core.data_structure import (
    NormalizedSnapshot,
    OrderReceipt,
    ShadowOrder,
    Side,
    TapeSegment,
)
from ...core.port import IIntervalModel, IMarketDataQuery, IMatchAlgorithm


EPSILON = 1e-12


class SegmentBaseAlgorithm(IMatchAlgorithm):
    """纯 Segment 计算算法。

    约束：
    - 不持有 shadow order/price level 等订单簿状态。
    - 只持有算法上下文（segment buffer 与窗口快照）。
    - 订单生命周期状态由 Simulator 维护并通过 active_orders 传入。
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
        self._base_depth: Dict[Tuple[Side, float], float] = {}
        self._t_start = 0
        self._t_end = 0

    def set_market_data_query(self, market_data_query: IMarketDataQuery) -> None:
        self._market_data_query = market_data_query

    def start_run(self) -> None:
        self._segment_buffer = []
        self._window_start = None
        self._window_end = None
        self._prev_snapshot = None
        self._pending_prev_snapshot = None
        self._base_depth = {}
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
        self._base_depth = {}
        self._t_start = 0
        self._t_end = 0
        if self._market_data_query is None:
            raise RuntimeError("SegmentBaseAlgorithm requires market_data_query for start_session().")
        if self._tape_builder is None:
            raise RuntimeError("SegmentBaseAlgorithm requires tape_builder for start_session().")

        raw_list = self._market_data_query.query_data(1) or []
        if raw_list is None:
            return
        curr_snapshot = self._decode_raw_md(raw_list[0])
        if not curr_snapshot:
            raise ValueError("Decode Error")

        if self._prev_snapshot is None:
            raise ValueError("SegmentbaseAlgo require prev_snapshot for start_session()")

        self._window_start = self._prev_snapshot
        self._window_end = curr_snapshot
        self._t_start = int(self._window_start.ts_recv)
        self._t_end = int(self._window_end.ts_recv)
        if self._t_end <= self._t_start:
            return

        self._segment_buffer = self._tape_builder.build(self._window_start, self._window_end)
        self._base_depth = self._build_base_depth(self._window_start)
        self._pending_prev_snapshot = curr_snapshot

    def on_order_action_impl(self, order: ShadowOrder, current_time: int) -> List[OrderReceipt]:
        t = int(current_time)
        market_pos = self._estimate_market_position(order.side, order.price, t)
        consumed, fill_price = self._compute_immediate_fill(order, t)

        if consumed <= 0:
            return [
                OrderReceipt(
                    order_id=order.order_id,
                    receipt_type="NONE",
                    timestamp=t,
                    fill_qty=0,
                    fill_price=float(order.price),
                    remaining_qty=int(max(0, order.now_vol)),
                    pos=market_pos,
                )
            ]

        remain = max(0, int(order.now_vol) - consumed)
        receipt_type = "FILL" if remain == 0 else "PARTIAL"
        return [
            OrderReceipt(
                order_id=order.order_id,
                receipt_type=receipt_type,
                timestamp=t,
                fill_qty=consumed,
                fill_price=float(fill_price),
                remaining_qty=remain,
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

        best_receipt: Optional[OrderReceipt] = None
        best_time = t_to + 1
        best_order_id = ""

        for order_id, order in active_orders.items():
            projected = self._project_order_event(order, t_from, t_to)
            if projected is None:
                continue
            evt_time = int(projected.timestamp)
            if evt_time < best_time or (evt_time == best_time and order_id < best_order_id):
                best_receipt = projected
                best_time = evt_time
                best_order_id = order_id

        if best_receipt is None:
            return [self._none_receipt(t_to)]
        return [best_receipt]

    def flush_window(self) -> object:
        if self._pending_prev_snapshot is not None:
            self._prev_snapshot = self._pending_prev_snapshot
        self._pending_prev_snapshot = None
        return {
            "interval_start": self._t_start,
            "interval_end": self._t_end,
            "segment_count": len(self._segment_buffer),
        }

    @staticmethod
    def _select_curr_snapshot(
        snapshots: List[NormalizedSnapshot],
        prev_snapshot: NormalizedSnapshot,
    ) -> Optional[NormalizedSnapshot]:
        prev_t = int(prev_snapshot.ts_recv)
        for snap in snapshots:
            if int(snap.ts_recv) > prev_t:
                return snap
        return None

    def _decode_raw_md(self, raw: Any) -> Optional[NormalizedSnapshot]:
        if isinstance(raw, NormalizedSnapshot):
            return raw
        if hasattr(raw, "ts_recv") and hasattr(raw, "bids") and hasattr(raw, "asks"):
            return raw  # type: ignore[return-value]
        return None

    @staticmethod
    def _norm_price(price: float) -> float:
        return round(float(price), 8)

    def _build_base_depth(self, snapshot: NormalizedSnapshot) -> Dict[Tuple[Side, float], float]:
        out: Dict[Tuple[Side, float], float] = {}
        for lvl in snapshot.bids:
            out[(Side.BUY, self._norm_price(lvl.price))] = float(lvl.qty)
        for lvl in snapshot.asks:
            out[(Side.SELL, self._norm_price(lvl.price))] = float(lvl.qty)
        return out

    def _segment_value(self, mapping: Dict[Tuple[Side, float], int], side: Side, price: float) -> int:
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

    def _queue_depth(self, side: Side, price: float, t: int) -> float:
        q = float(self._base_depth.get((side, self._norm_price(price)), 0.0))
        if not self._segment_buffer or t <= self._t_start:
            return max(0.0, q)

        for seg in self._segment_buffer:
            if t <= int(seg.t_start):
                break

            overlap_start = max(int(seg.t_start), self._t_start)
            overlap_end = min(int(seg.t_end), int(t))
            if overlap_end <= overlap_start:
                continue
            if not self._is_price_active(seg, side, price):
                continue

            seg_duration = int(seg.t_end) - int(seg.t_start)
            if seg_duration <= 0:
                continue
            progress = (overlap_end - overlap_start) / seg_duration

            net_flow = self._segment_value(seg.net_flow, side, price)
            trades = self._segment_value(seg.trades, side, price)
            q += (net_flow - trades) * progress

            if t <= int(seg.t_end):
                break

        return max(0.0, q)

    def _estimate_market_position(self, side: Side, price: float, t: int) -> int:
        return int(round(self._queue_depth(side, price, t)))

    def _best_prices_at(self, t: int) -> Tuple[Optional[float], Optional[float]]:
        seg = self._segment_at_time(t)
        if seg is not None:
            return float(seg.bid_price), float(seg.ask_price)
        if self._window_start is None:
            return None, None
        bid = self._window_start.best_bid
        ask = self._window_start.best_ask
        return (float(bid) if bid is not None else None, float(ask) if ask is not None else None)

    def _compute_immediate_fill(self, order: ShadowOrder, t: int) -> Tuple[int, float]:
        bid, ask = self._best_prices_at(t)
        qty = int(max(0, order.now_vol))
        if qty <= 0:
            return 0, float(order.price)

        if order.side == Side.BUY and ask is not None and float(order.price) >= ask - EPSILON:
            avail = int(self._queue_depth(Side.SELL, ask, t))
            fill = max(0, min(qty, avail))
            return fill, float(ask)

        if order.side == Side.SELL and bid is not None and float(order.price) <= bid + EPSILON:
            avail = int(self._queue_depth(Side.BUY, bid, t))
            fill = max(0, min(qty, avail))
            return fill, float(bid)

        return 0, float(order.price)

    def _cancel_front_factor(self) -> float:
        # k=-1 => 1.0 (撤单更偏前，前沿推进更快)
        # k=+1 => 0.0 (撤单更偏后，前沿推进更慢)
        return max(0.0, min(1.0, 0.5 * (1.0 - self.cancel_bias_k)))

    def _depletion_rate(self, seg: TapeSegment, side: Side, price: float) -> float:
        duration = int(seg.t_end) - int(seg.t_start)
        if duration <= 0:
            return 0.0
        if not self._is_price_active(seg, side, price):
            return 0.0

        m = self._segment_value(seg.trades, side, price) / duration
        n = self._segment_value(seg.net_flow, side, price) / duration

        if n < 0:
            # n<0 表示净流出（撤单主导），对前沿推进贡献按 bias 折算。
            rate = m + self._cancel_front_factor() * (-n)
        else:
            # n>0 表示净流入，压制前沿推进。
            rate = m - n
        return max(0.0, rate)

    def _depletion_between(self, side: Side, price: float, t_from: int, t_to: int) -> float:
        if t_to <= t_from or not self._segment_buffer:
            return 0.0
        dep = 0.0

        for seg in self._segment_buffer:
            if t_to <= int(seg.t_start):
                break
            overlap_start = max(int(seg.t_start), int(t_from))
            overlap_end = min(int(seg.t_end), int(t_to))
            if overlap_end <= overlap_start:
                continue

            rate = self._depletion_rate(seg, side, price)
            if rate <= EPSILON:
                continue
            dep += rate * (overlap_end - overlap_start)

        return max(0.0, dep)

    def _filled_volume_at(self, order: ShadowOrder, t: int) -> int:
        dep = self._depletion_between(order.side, float(order.price), int(order.create_time), int(t))
        filled = int(dep - int(order.pos))
        if filled < 0:
            filled = 0
        if filled > int(order.init_vol):
            filled = int(order.init_vol)
        return filled

    def _solve_hit_time(self, order: ShadowOrder, t_from: int, t_to: int, target_dep: float) -> int:
        dep = self._depletion_between(order.side, float(order.price), int(order.create_time), int(t_from))
        if dep >= target_dep - EPSILON:
            return int(t_from)

        for seg in self._segment_buffer:
            if t_to <= int(seg.t_start):
                break
            overlap_start = max(int(seg.t_start), int(t_from))
            overlap_end = min(int(seg.t_end), int(t_to))
            if overlap_end <= overlap_start:
                continue

            rate = self._depletion_rate(seg, order.side, float(order.price))
            if rate <= EPSILON:
                continue

            span = overlap_end - overlap_start
            need = target_dep - dep
            gain = rate * span
            if dep + gain >= target_dep - EPSILON:
                dt = max(0.0, need / rate)
                return int(overlap_start + dt)
            dep += gain

        return int(t_to)

    def _project_order_event(self, order: ShadowOrder, start_time: int, until_time: int) -> Optional[OrderReceipt]:
        if int(order.now_vol) <= 0:
            return None

        eval_start = max(int(start_time), int(order.create_time))
        if until_time <= eval_start:
            return None

        filled_before = max(0, int(order.init_vol) - int(order.now_vol))
        filled_end = self._filled_volume_at(order, int(until_time))
        if filled_end <= filled_before:
            return None

        target_fill = filled_before + 1
        target_dep = float(int(order.pos) + target_fill)
        t_hit = self._solve_hit_time(order, eval_start, int(until_time), target_dep)

        filled_hit = self._filled_volume_at(order, t_hit)
        if filled_hit <= filled_before:
            filled_hit = min(int(order.init_vol), filled_before + 1)

        fill_delta = max(0, filled_hit - filled_before)
        if fill_delta <= 0:
            return None

        remain = max(0, int(order.now_vol) - fill_delta)
        receipt_type = "FILL" if remain == 0 else "PARTIAL"
        bid, ask = self._best_prices_at(t_hit)
        if order.side == Side.BUY:
            fill_price = ask if ask is not None else float(order.price)
        else:
            fill_price = bid if bid is not None else float(order.price)

        return OrderReceipt(
            order_id=order.order_id,
            receipt_type=receipt_type,
            timestamp=int(t_hit),
            fill_qty=int(fill_delta),
            fill_price=float(fill_price),
            remaining_qty=int(remain),
            pos=int(order.pos),
        )

    @staticmethod
    def _none_receipt(timestamp: int) -> OrderReceipt:
        return OrderReceipt(
            order_id="",
            receipt_type="NONE",
            timestamp=int(timestamp),
            fill_qty=0,
            fill_price=0.0,
            remaining_qty=0,
            pos=0,
        )
