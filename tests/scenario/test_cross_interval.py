"""跨区间场景测试（新架构）。"""

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

    def query_data(self, n: int):
        n = int(n)
        if n <= 0 or self._idx >= len(self._snapshots):
            return []
        right = min(len(self._snapshots), self._idx + n)
        return self._snapshots[self._idx:right]


class _BuilderByWindow:
    def __init__(self, mapping: Dict[Tuple[int, int], List[TapeSegment]]):
        self._mapping = dict(mapping)

    def build(self, prev, curr):
        return list(self._mapping.get((int(prev.ts_recv), int(curr.ts_recv)), []))


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


def test_cancel_across_interval():
    t0, t1, t2 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS, 2000 * TICK_PER_MS
    snapshots = [
        create_test_snapshot(t0, 100.0, 101.0, bid_qty=30, ask_qty=100),
        create_test_snapshot(t1, 100.0, 101.0, bid_qty=30, ask_qty=100),
        create_test_snapshot(t2, 100.0, 101.0, bid_qty=30, ask_qty=100),
    ]
    seg01 = TapeSegment(
        index=1,
        t_start=t0,
        t_end=t1,
        bid_price=100.0,
        ask_price=101.0,
        trades={},
        cancels={},
        net_flow={},
        activation_bid={100.0},
        activation_ask={101.0},
    )
    seg12 = TapeSegment(
        index=1,
        t_start=t1,
        t_end=t2,
        bid_price=100.0,
        ask_price=101.0,
        trades={},
        cancels={},
        net_flow={},
        activation_bid={100.0},
        activation_ask={101.0},
    )
    feed = _StaticQueryFeed(snapshots)
    algo = SegmentBaseAlgorithm(
        cancel_bias_k=0.0,
        tape_builder=_BuilderByWindow({(t0, t1): [seg01], (t1, t2): [seg12]}),
        market_data_query=feed,
    )
    sim = Simulator_Impl(match_algo=algo)
    sim.set_market_data_stream(feed)
    sim.set_market_data_query(feed)
    sim.start_run()
    feed.next()

    sim.start_session()
    sim.on_action(_place("cancel-1", Side.BUY, 100.0, 10, t0 + 10 * TICK_PER_MS))

    feed.next()
    sim.start_session()
    canceled = sim.on_action(_cancel("cancel-1", t1 + 10 * TICK_PER_MS))[0]
    rejected = sim.on_action(_cancel("non-existent", t1 + 20 * TICK_PER_MS))[0]

    assert canceled.receipt_type == "CANCELED"
    assert rejected.receipt_type == "REJECTED"


def test_fill_across_interval():
    t0, t1, t2 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS, 2000 * TICK_PER_MS
    snapshots = [
        create_test_snapshot(t0, 100.0, 101.0, bid_qty=0, ask_qty=100),
        create_test_snapshot(t1, 100.0, 101.0, bid_qty=0, ask_qty=100),
        create_test_snapshot(t2, 100.0, 101.0, bid_qty=0, ask_qty=100),
    ]
    seg01 = TapeSegment(
        index=1,
        t_start=t0,
        t_end=t1,
        bid_price=100.0,
        ask_price=101.0,
        trades={},
        cancels={},
        net_flow={},
        activation_bid={100.0},
        activation_ask={101.0},
    )
    seg12 = TapeSegment(
        index=1,
        t_start=t1,
        t_end=t2,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): 120},
        cancels={},
        net_flow={(Side.BUY, 100.0): 0},
        activation_bid={100.0},
        activation_ask={101.0},
    )
    feed = _StaticQueryFeed(snapshots)
    algo = SegmentBaseAlgorithm(
        cancel_bias_k=0.0,
        tape_builder=_BuilderByWindow({(t0, t1): [seg01], (t1, t2): [seg12]}),
        market_data_query=feed,
    )
    sim = Simulator_Impl(match_algo=algo)
    sim.set_market_data_stream(feed)
    sim.set_market_data_query(feed)
    sim.start_run()
    feed.next()

    sim.start_session()
    sim.on_action(_place("cross-fill", Side.BUY, 100.0, 5, t0 + TICK_PER_MS))
    assert sim.step(t1)[0].receipt_type == "NONE"

    feed.next()
    sim.start_session()
    seen_fill = False
    for _ in range(32):
        receipts = sim.step(t2)
        if any(r.receipt_type in {"PARTIAL", "FILL"} for r in receipts):
            seen_fill = True
        if any(r.receipt_type == "FILL" for r in receipts):
            break
        if receipts and receipts[0].receipt_type == "NONE":
            break

    cancel_after = sim.on_action(_cancel("cross-fill", t2 - TICK_PER_MS))[0]
    assert seen_fill
    assert cancel_after.receipt_type in {"REJECTED", "CANCELED"}
