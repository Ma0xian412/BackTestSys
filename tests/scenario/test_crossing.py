"""Crossing（穿越价格）场景测试（新架构）。"""

from __future__ import annotations

from typing import Dict, List, Tuple

from quant_framework.adapters.execution_venue import SegmentBaseAlgorithm, Simulator_Impl
from quant_framework.core.data_structure import (
    Action,
    ActionType,
    Order,
    Side,
    TICK_PER_MS,
    TapeSegment,
)
from tests.conftest import create_test_snapshot


class _StaticQueryFeed:
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self._idx = 0

    def next(self):
        if self._idx >= len(self._snapshots):
            return None
        out = self._snapshots[self._idx]
        self._idx += 1
        return out

    def reset(self):
        self._idx = 0

    def query_data(self, n: int):
        n = int(n)
        if n <= 0 or self._idx >= len(self._snapshots):
            return []
        right = min(len(self._snapshots), self._idx + n)
        return self._snapshots[self._idx:right]


class _WindowBuilder:
    def __init__(self, mapping: Dict[Tuple[int, int], List[TapeSegment]]):
        self._mapping = dict(mapping)

    def build(self, prev, curr):
        return list(self._mapping.get((int(prev.ts_recv), int(curr.ts_recv)), []))


def _make_sim(
    snapshots,
    window_segments: Dict[Tuple[int, int], List[TapeSegment]],
    *,
    cancel_bias_k: float = 0.0,
) -> Simulator_Impl:
    feed = _StaticQueryFeed(snapshots)
    algo = SegmentBaseAlgorithm(
        cancel_bias_k=cancel_bias_k,
        tape_builder=_WindowBuilder(window_segments),
        market_data_query=feed,
    )
    sim = Simulator_Impl(match_algo=algo)
    sim.set_market_data_query(feed)
    sim.start_run()
    feed.next()  # 模拟 kernel 先消费 prev 快照
    sim._test_feed = feed  # type: ignore[attr-defined]
    return sim


def _seg(t0: int, t1: int, bid: float, ask: float) -> TapeSegment:
    return TapeSegment(
        index=1,
        t_start=t0,
        t_end=t1,
        bid_price=bid,
        ask_price=ask,
        trades={},
        cancels={},
        net_flow={},
        activation_bid={bid},
        activation_ask={ask},
    )


def _place(order_id: str, side: Side, price: float, qty: int, t: int) -> Action:
    return Action(
        action_type=ActionType.PLACE_ORDER,
        create_time=t,
        payload=Order(order_id=order_id, side=side, price=price, qty=qty),
    )


def test_immediate_execution():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    s0 = create_test_snapshot(t0, 100.0, 101.0, bid_qty=50, ask_qty=60)
    s1 = create_test_snapshot(t1, 100.0, 101.0, bid_qty=50, ask_qty=60)
    sim = _make_sim([s0, s1], {(t0, t1): [_seg(t0, t1, 100.0, 101.0)]})
    sim.start_session()

    buy_cross = sim.on_action(_place("buy-cross", Side.BUY, 101.0, 10, t0 + 10 * TICK_PER_MS))[0]
    sell_cross = sim.on_action(_place("sell-cross", Side.SELL, 100.0, 15, t0 + 20 * TICK_PER_MS))[0]
    passive = sim.on_action(_place("passive-1", Side.BUY, 99.0, 20, t0 + 30 * TICK_PER_MS))[0]

    assert buy_cross.receipt_type in {"PARTIAL", "FILL"}
    assert sell_cross.receipt_type in {"PARTIAL", "FILL"}
    assert passive.receipt_type == "NONE"


def test_partial_fill_position_zero():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    s0 = create_test_snapshot(t0, 100.0, 101.0, bid_qty=50, ask_qty=100)
    s1 = create_test_snapshot(t1, 100.0, 101.0, bid_qty=50, ask_qty=100)
    sim = _make_sim([s0, s1], {(t0, t1): [_seg(t0, t1, 100.0, 101.0)]})
    sim.start_session()

    receipt = sim.on_action(_place("after-crossing", Side.BUY, 101.0, 150, t0 + 10 * TICK_PER_MS))[0]
    assert receipt.receipt_type == "PARTIAL"
    assert receipt.fill_qty == 100
    assert receipt.remaining_qty == 50
    assert receipt.pos == 0


def test_blocked_by_existing_shadow():
    """新架构下验证同价位 FIFO 尾部排队。"""
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    s0 = create_test_snapshot(t0, 100.0, 101.0, bid_qty=20, ask_qty=100)
    s1 = create_test_snapshot(t1, 100.0, 101.0, bid_qty=20, ask_qty=100)
    sim = _make_sim([s0, s1], {(t0, t1): [_seg(t0, t1, 100.0, 101.0)]})
    sim.start_session()

    r1 = sim.on_action(_place("buy-1", Side.BUY, 100.0, 30, t0 + 10 * TICK_PER_MS))[0]
    r2 = sim.on_action(_place("buy-2", Side.BUY, 100.0, 10, t0 + 20 * TICK_PER_MS))[0]

    assert r1.receipt_type == "NONE"
    assert r2.receipt_type == "NONE"
    assert r2.pos >= r1.pos + 30


def test_blocked_by_queue_depth():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    s0 = create_test_snapshot(t0, 101.0, 102.0, bid_qty=20, ask_qty=60)
    s1 = create_test_snapshot(t1, 101.0, 102.0, bid_qty=20, ask_qty=60)
    sim = _make_sim([s0, s1], {(t0, t1): [_seg(t0, t1, 101.0, 102.0)]})
    sim.start_session()

    receipt = sim.on_action(_place("buy-q-depth", Side.BUY, 101.0, 10, t0 + 10 * TICK_PER_MS))[0]
    assert receipt.receipt_type == "NONE"
    assert receipt.pos >= 20


def test_post_crossing_pos_uses_x_coord():
    """新架构下改为验证跨区间后同价位 tail 分配。"""
    t0, t1, t2 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS, 2000 * TICK_PER_MS
    s0 = create_test_snapshot(t0, 100.0, 101.0, bid_qty=0, ask_qty=100)
    s1 = create_test_snapshot(t1, 100.0, 101.0, bid_qty=0, ask_qty=100)
    s2 = create_test_snapshot(t2, 100.0, 102.0, bid_qty=0, ask_qty=100)
    mapping = {
        (t0, t1): [_seg(t0, t1, 100.0, 101.0)],
        (t1, t2): [_seg(t1, t2, 100.0, 102.0)],
    }
    sim = _make_sim([s0, s1, s2], mapping)

    sim.start_session()
    r1 = sim.on_action(_place("post-cross", Side.BUY, 101.0, 150, t0 + 10 * TICK_PER_MS))[0]
    assert r1.receipt_type == "PARTIAL"
    assert r1.pos == 0
    assert r1.remaining_qty == 50

    sim._test_feed.next()  # type: ignore[attr-defined]
    sim.start_session()
    r2 = sim.on_action(_place("subsequent", Side.BUY, 101.0, 10, t1 + 10 * TICK_PER_MS))[0]
    assert r2.receipt_type == "NONE"
    assert r2.pos >= 50
