"""Cancel bias 场景测试（新架构）。"""

from __future__ import annotations

from typing import Dict, List, Tuple

from quant_framework.adapters.execution_venue import SegmentBaseAlgorithm, Simulator_Impl
from quant_framework.core.data_structure import Action, ActionType, Order, Side, TICK_PER_MS, TapeSegment
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


def _place(order_id: str, price: float, qty: int, t: int) -> Action:
    return Action(
        action_type=ActionType.PLACE_ORDER,
        create_time=t,
        payload=Order(order_id=order_id, side=Side.BUY, price=price, qty=qty),
    )


def _single_segment_scenario(cancel_bias_k: float, *, trades: int, net_flow: int, qty: int = 5) -> int:
    t0, t1 = 1000 * TICK_PER_MS, 1500 * TICK_PER_MS
    snapshots = [
        create_test_snapshot(t0, 100.0, 101.0, bid_qty=20, ask_qty=200),
        create_test_snapshot(t1, 100.0, 101.0, bid_qty=20, ask_qty=200),
    ]
    cancels_dict = {(Side.BUY, 100.0): abs(net_flow)} if net_flow < 0 else {}
    seg = TapeSegment(
        index=1,
        t_start=t0,
        t_end=t1,
        bid_price=100.0,
        ask_price=101.0,
        trades={(Side.BUY, 100.0): trades} if trades > 0 else {},
        cancels=cancels_dict,
        net_flow={(Side.BUY, 100.0): net_flow},
        activation_bid={100.0},
        activation_ask={101.0},
    )

    feed = _StaticQueryFeed(snapshots)
    algo = SegmentBaseAlgorithm(
        cancel_bias_k=cancel_bias_k,
        tape_builder=_BuilderByWindow({(t0, t1): [seg]}),
        market_data_query=feed,
    )
    sim = Simulator_Impl(match_algo=algo)
    sim.set_market_data_stream(feed)
    sim.set_market_data_query(feed)
    sim.start_run()
    feed.next()
    sim.start_session()
    sim.on_action(_place(f"bias-{cancel_bias_k}", 100.0, qty, t0 + TICK_PER_MS))

    total_fill = 0
    for _ in range(64):
        receipts = sim.step(t1)
        total_fill += sum(r.fill_qty for r in receipts if r.receipt_type in {"PARTIAL", "FILL"})
        if receipts and receipts[0].receipt_type == "NONE":
            break
        if any(r.receipt_type == "FILL" for r in receipts):
            break
    return total_fill


def test_negative_bias_overfill():
    fill_neg = _single_segment_scenario(-0.8, trades=1, net_flow=-80, qty=5)
    assert fill_neg >= 1
    assert fill_neg <= 5


def test_zero_trades_no_fill():
    fill_zero = _single_segment_scenario(-0.8, trades=0, net_flow=0, qty=5)
    assert fill_zero == 0


def test_uniform_bias_control():
    fill_neg = _single_segment_scenario(-0.8, trades=1, net_flow=-80, qty=5)
    fill_uniform = _single_segment_scenario(0.0, trades=1, net_flow=-80, qty=5)
    assert fill_neg >= fill_uniform
    assert fill_uniform <= 5


def test_positive_bias_behavior():
    fill_uniform = _single_segment_scenario(0.0, trades=1, net_flow=-80, qty=5)
    fill_pos = _single_segment_scenario(0.8, trades=1, net_flow=-80, qty=5)
    assert fill_pos <= fill_uniform

    # 零成交时，正偏置也应保持稳定且不超订单量
    fill_pos_zero = _single_segment_scenario(0.8, trades=0, net_flow=0, qty=5)
    assert 0 <= fill_pos_zero <= 5


def test_multi_segment_cumulative():
    t0, t1, t2, t3 = 1000 * TICK_PER_MS, 1150 * TICK_PER_MS, 1300 * TICK_PER_MS, 1500 * TICK_PER_MS
    snapshots = [
        create_test_snapshot(t0, 100.0, 101.0, bid_qty=20, ask_qty=200),
        create_test_snapshot(t1, 100.0, 101.0, bid_qty=20, ask_qty=200),
        create_test_snapshot(t2, 100.0, 101.0, bid_qty=20, ask_qty=200),
        create_test_snapshot(t3, 100.0, 101.0, bid_qty=20, ask_qty=200),
    ]
    segs = [
        TapeSegment(
            index=1,
            t_start=t0,
            t_end=t1,
            bid_price=100.0,
            ask_price=101.0,
            trades={},
            cancels={(Side.BUY, 100.0): 40},
            net_flow={(Side.BUY, 100.0): -40},
            activation_bid={100.0},
            activation_ask={101.0},
        ),
        TapeSegment(
            index=2,
            t_start=t1,
            t_end=t2,
            bid_price=100.0,
            ask_price=101.0,
            trades={},
            cancels={(Side.BUY, 100.0): 40},
            net_flow={(Side.BUY, 100.0): -40},
            activation_bid={100.0},
            activation_ask={101.0},
        ),
        TapeSegment(
            index=3,
            t_start=t2,
            t_end=t3,
            bid_price=100.0,
            ask_price=101.0,
            trades={(Side.BUY, 100.0): 1},
            cancels={(Side.BUY, 100.0): 20},
            net_flow={(Side.BUY, 100.0): -20},
            activation_bid={100.0},
            activation_ask={101.0},
        ),
    ]
    feed = _StaticQueryFeed([snapshots[0], snapshots[-1]])
    algo = SegmentBaseAlgorithm(
        cancel_bias_k=-0.8,
        tape_builder=_BuilderByWindow({(t0, t3): segs}),
        market_data_query=feed,
    )
    sim = Simulator_Impl(match_algo=algo)
    sim.set_market_data_stream(feed)
    sim.set_market_data_query(feed)
    sim.start_run()
    feed.next()
    sim.start_session()
    sim.on_action(_place("multi-seg", 100.0, 5, t0 + TICK_PER_MS))

    receipts_all = []
    for _ in range(128):
        rs = sim.step(t3)
        receipts_all.extend(rs)
        if rs and rs[0].receipt_type == "NONE":
            break
        if any(r.receipt_type == "FILL" for r in rs):
            break

    total_fill = sum(r.fill_qty for r in receipts_all if r.receipt_type in {"PARTIAL", "FILL"})
    assert 0 <= total_fill <= 5
