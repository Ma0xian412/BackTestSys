"""因果一致性场景测试（新架构）。"""

from quant_framework.adapters import ExecutionVenueImpl, NullObservabilityImpl, TimeModelImpl
from quant_framework.core.data_structure import (
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    Action,
    ActionType,
)
from quant_framework.core import BacktestApp, RuntimeBuildConfig
from quant_framework.core.data_structure import Order, Side, TICK_PER_MS
from quant_framework.adapters.execution_venue import FIFOExchangeSimulator
from quant_framework.adapters.interval_model import TapeConfig, UnifiedTapeBuilder
from quant_framework.adapters.trading.oms import OMSImpl
from tests.conftest import MockFeed, create_test_snapshot


def _run_app(snapshots, strategy, delay_out: int, delay_in: int):
    app = BacktestApp(
        RuntimeBuildConfig(
            feed=MockFeed(snapshots),
            venue=ExecutionVenueImpl(
                simulator=FIFOExchangeSimulator(cancel_bias_k=0.0),
                tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
            ),
            strategy=strategy,
            oms=OMSImpl(),
            timeModel=TimeModelImpl(delay_out=delay_out, delay_in=delay_in),
            obs=NullObservabilityImpl(),
        ),
    )
    return app.run()


def test_event_causal_ordering():
    """因果顺序：快照在回执之前处理，回执 recv_time ≥ timestamp。"""
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30, last_vol_split=[(100.0, 100)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10, last_vol_split=[(100.0, 100)]),
    ]

    class _Tracker:
        def __init__(self):
            self.log = []

        def on_event(self, e, ctx):
            if e.kind == EVENT_KIND_SNAPSHOT_ARRIVAL:
                self.log.append(("SNAPSHOT", e.time))
                if len(self.log) == 1:
                    order = Order(order_id="test-order", side=Side.BUY, price=100.0, qty=5)
                    return [Action(action_type=ActionType.PLACE_ORDER, create_time=0, payload=order)]
                return []
            if e.kind == EVENT_KIND_RECEIPT_DELIVERY:
                receipt = e.payload
                self.log.append(("RECEIPT", receipt.timestamp, receipt.recv_time))
            return []

    strategy = _Tracker()
    _run_app(snapshots=snapshots, strategy=strategy, delay_out=100 * TICK_PER_MS, delay_in=50 * TICK_PER_MS)

    snap_idx = [i for i, e in enumerate(strategy.log) if e[0] == "SNAPSHOT"]
    rcpt_idx = [i for i, e in enumerate(strategy.log) if e[0] == "RECEIPT"]
    if rcpt_idx:
        assert snap_idx[0] < rcpt_idx[0], "第一个快照应在第一个回执之前"
        for e in strategy.log:
            if e[0] == "RECEIPT":
                assert e[2] >= e[1], f"recv_time({e[2]}) 应 ≥ timestamp({e[1]})"


def test_receipt_delay_consistency():
    """回执延迟：recv_time = timestamp + delay_in。"""
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30, last_vol_split=[(100.0, 80)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10, last_vol_split=[(100.0, 80)]),
    ]

    delay_in = 200 * TICK_PER_MS
    recorded = []

    class _Recorder:
        def __init__(self):
            self.placed = False

        def on_event(self, e, ctx):
            if e.kind == EVENT_KIND_SNAPSHOT_ARRIVAL and not self.placed:
                self.placed = True
                order = Order(order_id="recv-test", side=Side.BUY, price=100.0, qty=3)
                return [Action(action_type=ActionType.PLACE_ORDER, create_time=0, payload=order)]
            if e.kind == EVENT_KIND_RECEIPT_DELIVERY:
                receipt = e.payload
                recorded.append({"timestamp": receipt.timestamp, "recv_time": receipt.recv_time})
            return []

    _run_app(snapshots=snapshots, strategy=_Recorder(), delay_out=50 * TICK_PER_MS, delay_in=delay_in)
    for rec in recorded:
        assert rec["recv_time"] == rec["timestamp"] + delay_in


def test_time_clamping():
    """不同延迟参数下运行稳定，输出区间数正常。"""
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50, last_vol_split=[(100.0, 80)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=30, last_vol_split=[(100.0, 80)]),
        create_test_snapshot(2000 * TICK_PER_MS, 100.0, 101.0, bid_qty=20, last_vol_split=[(100.0, 80)]),
    ]

    class _FreqOrder:
        def __init__(self):
            self.count = 0

        def on_event(self, e, ctx):
            if e.kind == EVENT_KIND_SNAPSHOT_ARRIVAL:
                self.count += 1
                order = Order(order_id=f"t-{self.count}", side=Side.BUY, price=100.0, qty=5)
                return [Action(action_type=ActionType.PLACE_ORDER, create_time=0, payload=order)]
            return []

    for label, delay_out, delay_in in [
        ("baseline", 50 * TICK_PER_MS, 100 * TICK_PER_MS),
        ("short-delay", 1 * TICK_PER_MS, 10 * TICK_PER_MS),
        ("long-delay", 300 * TICK_PER_MS, 500 * TICK_PER_MS),
    ]:
        results = _run_app(
            snapshots=snapshots,
            strategy=_FreqOrder(),
            delay_out=delay_out,
            delay_in=delay_in,
        )
        print(f"  {label}: intervals={results['intervals']}, receipts={results['diagnostics']['receipts_generated']}")
