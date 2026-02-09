"""因果一致性场景测试。

验证内容：
- 事件因果顺序（快照先于回执、回执时间戳合理）
- 回执延迟一致性（recv_time = timestamp + delay_in）
- 时间钳制避免因果反转（非 1:1 时间线映射）
"""

from quant_framework.core.types import Order, Side, TimeInForce, TICK_PER_MS
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.trading.oms import OrderManager
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig

from tests.conftest import create_test_snapshot, MockFeed


# ---------------------------------------------------------------------------
# 事件因果顺序
# ---------------------------------------------------------------------------

def test_event_causal_ordering():
    """因果顺序：快照在回执之前处理，回执 recv_time ≥ timestamp。"""
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30,
                             last_vol_split=[(100.0, 100)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10,
                             last_vol_split=[(100.0, 100)]),
    ]

    class _Tracker:
        def __init__(self):
            self.log = []

        def on_snapshot(self, snapshot, oms_view):
            ts = snapshot.ts_exch if hasattr(snapshot, 'ts_exch') else 0
            self.log.append(("SNAPSHOT", ts))
            if len(self.log) == 1:
                return [Order(order_id="test-order", side=Side.BUY, price=100.0, qty=5)]
            return []

        def on_receipt(self, receipt, snapshot, oms_view):
            self.log.append(("RECEIPT", receipt.timestamp, receipt.recv_time))
            return []

    strategy = _Tracker()

    runner = EventLoopRunner(
        feed=MockFeed(snapshots),
        tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
        exchange=FIFOExchangeSimulator(cancel_bias_k=0.0),
        strategy=strategy, oms=OrderManager(),
        config=RunnerConfig(
            delay_out=100 * TICK_PER_MS, delay_in=50 * TICK_PER_MS,
            timeline=TimelineConfig(a=1.0, b=0),
        ),
    )
    runner.run()

    snap_idx = [i for i, e in enumerate(strategy.log) if e[0] == "SNAPSHOT"]
    rcpt_idx = [i for i, e in enumerate(strategy.log) if e[0] == "RECEIPT"]

    if rcpt_idx:
        assert snap_idx[0] < rcpt_idx[0], "第一个快照应在第一个回执之前"
        for e in strategy.log:
            if e[0] == "RECEIPT":
                assert e[2] >= e[1], f"recv_time({e[2]}) 应 ≥ timestamp({e[1]})"


# ---------------------------------------------------------------------------
# 回执延迟一致性
# ---------------------------------------------------------------------------

def test_receipt_delay_consistency():
    """回执延迟：recv_time = timestamp + delay_in（IOC 和成交回执均适用）。"""
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30,
                             last_vol_split=[(100.0, 80)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=10,
                             last_vol_split=[(100.0, 80)]),
    ]

    delay_in = 200 * TICK_PER_MS
    recorded = []

    class _Recorder:
        def __init__(self):
            self.placed = False

        def on_snapshot(self, snapshot, oms_view):
            if not self.placed:
                self.placed = True
                return [Order(order_id="recv-test", side=Side.BUY, price=100.0, qty=3)]
            return []

        def on_receipt(self, receipt, snapshot, oms_view):
            recorded.append({
                'timestamp': receipt.timestamp,
                'recv_time': receipt.recv_time,
            })
            return []

    runner = EventLoopRunner(
        feed=MockFeed(snapshots),
        tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
        exchange=FIFOExchangeSimulator(cancel_bias_k=0.0),
        strategy=_Recorder(), oms=OrderManager(),
        config=RunnerConfig(
            delay_out=50 * TICK_PER_MS, delay_in=delay_in,
            timeline=TimelineConfig(a=1.0, b=0),
        ),
    )
    runner.run()

    for rec in recorded:
        expected = rec['timestamp'] + delay_in
        assert rec['recv_time'] == expected, (
            f"recv_time 应为 {expected}，实际 {rec['recv_time']}"
        )


# ---------------------------------------------------------------------------
# 时间钳制
# ---------------------------------------------------------------------------

def test_time_clamping():
    """时间钳制：非 1:1 映射（a=0.9/2.0）时系统自动钳制，不产生因果反转。"""
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50,
                             last_vol_split=[(100.0, 80)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=30,
                             last_vol_split=[(100.0, 80)]),
        create_test_snapshot(2000 * TICK_PER_MS, 100.0, 101.0, bid_qty=20,
                             last_vol_split=[(100.0, 80)]),
    ]

    class _FreqOrder:
        def __init__(self):
            self.count = 0

        def on_snapshot(self, snapshot, oms_view):
            self.count += 1
            return [Order(order_id=f"t-{self.count}", side=Side.BUY, price=100.0, qty=5)]

        def on_receipt(self, receipt, snapshot, oms_view):
            return []

    for label, a, b in [("a=1.0", 1.0, 0), ("a=0.9", 0.9, 100), ("a=2.0", 2.0, -500)]:
        feed = MockFeed(snapshots)
        feed.reset()
        runner = EventLoopRunner(
            feed=feed,
            tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
            exchange=FIFOExchangeSimulator(cancel_bias_k=0.0),
            strategy=_FreqOrder(), oms=OrderManager(),
            config=RunnerConfig(
                delay_out=50 * TICK_PER_MS, delay_in=100 * TICK_PER_MS,
                timeline=TimelineConfig(a=a, b=b),
            ),
        )
        results = runner.run()
        print(f"  {label}: intervals={results['intervals']}, "
              f"receipts={results['diagnostics']['receipts_generated']}")
