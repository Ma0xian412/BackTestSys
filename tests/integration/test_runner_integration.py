"""BacktestApp 集成测试（新架构）。"""

import os
import tempfile

from quant_framework.adapters import DelayTimeModel, FIFOExecutionVenue, ReceiptLoggerSink
from quant_framework.core import BacktestApp, RuntimeBuildConfig
from quant_framework.core.runtime import EVENT_KIND_RECEIPT_DELIVERY, EVENT_KIND_SNAPSHOT_ARRIVAL
from quant_framework.core.types import Level, NormalizedSnapshot, Order, Side, TICK_PER_MS
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.replay_strategy import ReplayStrategy
from quant_framework.trading.receipt_logger import ReceiptLogger

from tests.conftest import create_test_snapshot, print_tape_path, MockFeed


# ---------------------------------------------------------------------------
# 基本管线
# ---------------------------------------------------------------------------

def test_basic_pipeline():
    """基本管线：BacktestApp 驱动组件协同。"""
    builder = UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0)
    exchange = FIFOExecutionVenue(FIFOExchangeSimulator(cancel_bias_k=0.0), builder)
    oms = OrderManager()
    receipt_logger = ReceiptLogger()

    class _OneShot:
        def __init__(self):
            self.sent = False

        def on_event(self, e, ctx):
            if e.kind == EVENT_KIND_SNAPSHOT_ARRIVAL and not self.sent:
                self.sent = True
                return [Order(order_id="one-shot", side=Side.BUY, price=100.0, qty=1)]
            return []

    strategy = _OneShot()

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5)
    tape = builder.build(prev, curr)
    print_tape_path(tape)

    app = BacktestApp()
    result = app.run(
        RuntimeBuildConfig(
            feed=MockFeed([prev, curr]),
            venue=exchange,
            strategy=strategy,
            oms=oms,
            timeModel=DelayTimeModel(delay_out=0, delay_in=0),
            obs=ReceiptLoggerSink(receipt_logger),
        )
    )
    assert result["intervals"] == 1
    assert result["diagnostics"]["orders_submitted"] == 1


# ---------------------------------------------------------------------------
# 带延迟管线
# ---------------------------------------------------------------------------

def test_pipeline_with_delays():
    """带延迟管线：验证 delay_out / delay_in 生效，区间数正确。"""
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0, bid_qty=50,
                             last_vol_split=[(100.0, 30)]),
        create_test_snapshot(1500 * TICK_PER_MS, 100.0, 101.0, bid_qty=40,
                             last_vol_split=[(100.0, 40)]),
        create_test_snapshot(2000 * TICK_PER_MS, 100.0, 101.0, bid_qty=30,
                             last_vol_split=[(100.0, 50)]),
    ]

    class _FrequentStrategy:
        def __init__(self):
            self.count = 0
            self.snapshots_received = []
            self.receipts_received = []

        def on_event(self, e, ctx):
            if e.kind == EVENT_KIND_SNAPSHOT_ARRIVAL:
                self.count += 1
                self.snapshots_received.append(e.time)
                if ctx.snapshot and ctx.snapshot.bids:
                    return [Order(
                        order_id=f"order-{self.count}",
                        side=Side.BUY,
                        price=ctx.snapshot.bids[0].price,
                        qty=5,
                    )]
                return []
            if e.kind == EVENT_KIND_RECEIPT_DELIVERY:
                self.receipts_received.append((e.receipt.timestamp, e.receipt.recv_time))
            return []

    strategy = _FrequentStrategy()
    app = BacktestApp()
    result = app.run(
        RuntimeBuildConfig(
            feed=MockFeed(snapshots),
            venue=FIFOExecutionVenue(
                simulator=FIFOExchangeSimulator(cancel_bias_k=0.0),
                tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
            ),
            strategy=strategy,
            oms=OrderManager(),
            timeModel=DelayTimeModel(
                delay_out=10 * TICK_PER_MS,
                delay_in=5 * TICK_PER_MS,
            ),
            obs=ReceiptLoggerSink(ReceiptLogger()),
        )
    )
    assert result['intervals'] == 2, f"应处理 2 个区间，实际 {result['intervals']}"


# ---------------------------------------------------------------------------
# ReplayStrategy 集成
# ---------------------------------------------------------------------------

def test_replay_pipeline():
    """ReplayStrategy 集成：订单和撤单通过 BacktestApp 正确处理。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        order_file = os.path.join(tmpdir, "orders.csv")
        with open(order_file, 'w') as f:
            f.write("OrderId,LimitPrice,Volume,OrderDirection,SentTime\n")
            f.write("1,100.0,10,Buy,1000\n")
            f.write("2,101.0,5,Sell,1100\n")

        cancel_file = os.path.join(tmpdir, "cancels.csv")
        with open(cancel_file, 'w') as f:
            f.write("OrderId,CancelSentTime\n")
            f.write("1,1500\n")

        snapshots = [
            NormalizedSnapshot(
                ts_recv=1000, bids=[Level(100.0, 100)], asks=[Level(101.0, 100)],
                last_vol_split=[(100.0, 50)],
            ),
            NormalizedSnapshot(
                ts_recv=2000, bids=[Level(100.0, 80)], asks=[Level(101.0, 90)],
                last_vol_split=[(100.0, 30)],
            ),
        ]

        receipt_logger = ReceiptLogger()
        app = BacktestApp()
        results = app.run(
            RuntimeBuildConfig(
                feed=MockFeed(snapshots),
                venue=FIFOExecutionVenue(
                    simulator=FIFOExchangeSimulator(cancel_bias_k=0.0),
                    tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
                ),
                strategy=ReplayStrategy(
                    name="TestReplay",
                    order_file=order_file,
                    cancel_file=cancel_file,
                ),
                oms=OrderManager(portfolio=Portfolio(cash=100000.0)),
                timeModel=DelayTimeModel(delay_out=0, delay_in=0),
                obs=ReceiptLoggerSink(receipt_logger),
            )
        )
        assert results['diagnostics']['orders_submitted'] == 2, (
            f"应提交 2 个订单，实际 {results['diagnostics']['orders_submitted']}"
        )
        assert results['diagnostics']['cancels_submitted'] == 1, (
            f"应有 1 个撤单，实际 {results['diagnostics']['cancels_submitted']}"
        )
