"""Simulator + SegmentBaseAlgorithm（新架构）单元测试。"""

from __future__ import annotations

from typing import Dict, List, Tuple

from quant_framework.adapters.execution_venue import SegmentBaseAlgorithm, Simulator_Impl
from quant_framework.core.data_structure import (
    Action,
    ActionType,
    CancelRequest,
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

    def query_data(self):
        if self._idx >= len(self._snapshots):
            return []
        return [self._snapshots[self._idx]]


class _StaticBuilder:
    def __init__(self, mapping: Dict[Tuple[int, int], List[TapeSegment]]):
        self._mapping = dict(mapping)

    def build(self, prev, curr):
        return list(self._mapping.get((int(prev.ts_recv), int(curr.ts_recv)), []))


def _segment(t0: int, t1: int, *, bid: float = 100.0, ask: float = 101.0, trades=None, net_flow=None) -> TapeSegment:
    return TapeSegment(
        index=1,
        t_start=t0,
        t_end=t1,
        bid_price=bid,
        ask_price=ask,
        trades=dict(trades or {}),
        cancels={},
        net_flow=dict(net_flow or {}),
        activation_bid={bid},
        activation_ask={ask},
    )


def _make_simulator(
    *,
    t0: int,
    t1: int,
    bid: float = 100.0,
    ask: float = 101.0,
    bid_qty: int = 30,
    ask_qty: int = 40,
    segment: TapeSegment | None = None,
    cancel_bias_k: float = 0.0,
) -> Simulator_Impl:
    snap_a = create_test_snapshot(t0, bid, ask, bid_qty=bid_qty, ask_qty=ask_qty)
    snap_b = create_test_snapshot(t1, bid, ask, bid_qty=bid_qty, ask_qty=ask_qty)
    feed = _StaticQueryFeed([snap_a, snap_b])
    seg = segment or _segment(t0, t1, bid=bid, ask=ask)
    builder = _StaticBuilder({(t0, t1): [seg]})

    algo = SegmentBaseAlgorithm(
        cancel_bias_k=cancel_bias_k,
        tape_builder=builder,
        market_data_query=feed,
    )
    sim = Simulator_Impl(match_algo=algo)
    sim.set_market_data_stream(feed)
    sim.set_market_data_query(feed)
    sim.start_run()
    feed.next()  # 模拟 kernel 先消费首个 prev 快照
    sim.start_session()
    return sim


def _place(order_id: str, side: Side, price: float, qty: int, t: int) -> Action:
    return Action(
        action_type=ActionType.PLACE_ORDER,
        create_time=t,
        payload=Order(order_id=order_id, side=side, price=price, qty=qty),
    )


def _cancel(order_id: str, t: int) -> Action:
    return Action(
        action_type=ActionType.CANCEL_ORDER,
        create_time=t,
        payload=CancelRequest(order_id=order_id, create_time=t),
    )


def test_basic_order_arrival():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    sim = _make_simulator(t0=t0, t1=t1, bid_qty=30, ask_qty=40)

    receipts = sim.on_action(_place("test-1", Side.BUY, 100.0, 10, t0 + 10 * TICK_PER_MS))
    assert len(receipts) == 1
    assert receipts[0].receipt_type == "NONE"
    assert receipts[0].pos >= 30


def test_ioc_order():
    """新架构中以撤单路径验证“立即取消”语义。"""
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    sim = _make_simulator(t0=t0, t1=t1, bid_qty=20, ask_qty=20)

    sim.on_action(_place("ioc-like", Side.BUY, 99.0, 10, t0 + 10 * TICK_PER_MS))
    receipts = sim.on_action(_cancel("ioc-like", t0 + 20 * TICK_PER_MS))

    assert receipts[0].receipt_type == "CANCELED"
    assert receipts[0].order_id == "ioc-like"


def test_coordinate_axis():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    sim = _make_simulator(t0=t0, t1=t1, bid_qty=30, ask_qty=30)

    r1 = sim.on_action(_place("o1", Side.BUY, 100.0, 20, t0 + 10 * TICK_PER_MS))
    r2 = sim.on_action(_place("o2", Side.BUY, 100.0, 10, t0 + 20 * TICK_PER_MS))

    assert r1[0].receipt_type == "NONE"
    assert r2[0].receipt_type == "NONE"
    assert r1[0].pos < r2[0].pos


def test_fill():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    seg = _segment(
        t0,
        t1,
        trades={(Side.BUY, 100.0): 80},
        net_flow={(Side.BUY, 100.0): 0},
    )
    sim = _make_simulator(t0=t0, t1=t1, bid_qty=0, ask_qty=50, segment=seg)
    sim.on_action(_place("fill-test", Side.BUY, 100.0, 5, t0 + TICK_PER_MS))

    total_fill = 0
    for _ in range(16):
        receipts = sim.step(t1)
        total_fill += sum(r.fill_qty for r in receipts if r.receipt_type in {"PARTIAL", "FILL"})
        if any(r.receipt_type == "FILL" for r in receipts):
            break
        if receipts and receipts[0].receipt_type == "NONE":
            break

    assert total_fill > 0


def test_multi_partial_to_fill():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    seg = _segment(
        t0,
        t1,
        trades={(Side.BUY, 100.0): 120},
        net_flow={(Side.BUY, 100.0): 0},
    )
    sim = _make_simulator(t0=t0, t1=t1, bid_qty=0, ask_qty=50, segment=seg)
    sim.on_action(_place("multi-fill", Side.BUY, 100.0, 3, t0 + TICK_PER_MS))

    total_fill = 0
    saw_fill = False
    for _ in range(32):
        receipts = sim.step(t1)
        total_fill += sum(r.fill_qty for r in receipts if r.receipt_type in {"PARTIAL", "FILL"})
        if any(r.receipt_type == "FILL" for r in receipts):
            saw_fill = True
            break
        if receipts and receipts[0].receipt_type == "NONE":
            break

    assert saw_fill
    assert total_fill == 3


def test_fill_priority_fifo():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    seg = _segment(
        t0,
        t1,
        trades={(Side.BUY, 100.0): 200},
        net_flow={(Side.BUY, 100.0): 0},
    )
    sim = _make_simulator(t0=t0, t1=t1, bid_qty=0, ask_qty=50, segment=seg)

    sim.on_action(_place("order1", Side.BUY, 100.0, 2, t0 + 10 * TICK_PER_MS))
    sim.on_action(_place("order2", Side.BUY, 100.0, 2, t0 + 20 * TICK_PER_MS))

    fill_order = []
    for _ in range(16):
        receipts = sim.step(t1)
        fill_order.extend([r.order_id for r in receipts if r.receipt_type in {"PARTIAL", "FILL"}])
        if len(fill_order) >= 2:
            break
        if receipts and receipts[0].receipt_type == "NONE":
            break

    assert fill_order
    assert fill_order[0] == "order1"
    if "order2" in fill_order:
        assert fill_order.index("order1") < fill_order.index("order2")


def test_multiple_orders_same_price():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    sim = _make_simulator(t0=t0, t1=t1, bid_qty=10, ask_qty=100)

    r1 = sim.on_action(_place("buy-1", Side.BUY, 100.0, 10, t0 + 10 * TICK_PER_MS))[0]
    r2 = sim.on_action(_place("buy-2", Side.BUY, 100.0, 10, t0 + 20 * TICK_PER_MS))[0]
    r3 = sim.on_action(_place("buy-3", Side.BUY, 100.0, 10, t0 + 30 * TICK_PER_MS))[0]

    assert r1.pos < r2.pos < r3.pos


def test_improvement_mode_fill():
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    sim = _make_simulator(t0=t0, t1=t1, bid=100.0, ask=100.5, bid_qty=0, ask_qty=6)

    r1 = sim.on_action(_place("buy-improve", Side.BUY, 100.5, 6, t0 + 10 * TICK_PER_MS))
    r2 = sim.on_action(_place("buy-base", Side.BUY, 100.0, 10, t0 + 20 * TICK_PER_MS))

    assert r1[0].receipt_type == "FILL"
    assert r1[0].fill_qty == 6
    assert r2[0].receipt_type == "NONE"
