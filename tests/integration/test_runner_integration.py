"""BacktestApp 集成测试（新架构）。"""

import os
import tempfile

from quant_framework.adapters import ExecutionVenue_Impl, ReceiptLogger_Impl, TimeModel_Impl
from quant_framework.core.data_structure import (
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    Action,
    ActionType,
)
from quant_framework.core import BacktestApp, RuntimeBuildConfig
from quant_framework.core.data_structure import Level, NormalizedSnapshot, Order, Side, TICK_PER_MS
from quant_framework.adapters.interval_model import UnifiedIntervalModel_impl, TapeConfig
from quant_framework.adapters.execution_venue import FIFOExchangeSimulator
from quant_framework.adapters.IOMS.oms import OMS_Impl, Portfolio
from quant_framework.adapters.IStrategy.Replay_Strategy import ReplayStrategy_Impl
from quant_framework.adapters.observability.ReceiptLogger_Impl import ReceiptLogger_Impl

from tests.conftest import create_test_snapshot, print_tape_path, MockFeed


# ---------------------------------------------------------------------------
# 基本管线
# ---------------------------------------------------------------------------

def test_basic_pipeline():
    """基本管线：BacktestApp 驱动组件协同。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)
    exchange = ExecutionVenue_Impl(FIFOExchangeSimulator(cancel_bias_k=0.0), builder)
    oms = OMS_Impl()
    receipt_logger = ReceiptLogger_Impl()

    class _OneShot:
        def __init__(self):
            self.sent = False

        def on_event(self, e, ctx):
            if e.kind == EVENT_KIND_SNAPSHOT_ARRIVAL and not self.sent:
                self.sent = True
                order = Order(order_id="one-shot", side=Side.BUY, price=100.0, qty=1)
                return [Action(action_type=ActionType.PLACE_ORDER, create_time=0, payload=order)]
            return []

    strategy = _OneShot()

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5)
    tape = builder.build(prev, curr)
    print_tape_path(tape)

    app = BacktestApp(
        RuntimeBuildConfig(
            feed=MockFeed([prev, curr]),
            venue=exchange,
            strategy=strategy,
            oms=oms,
            timeModel=TimeModel_Impl(delay_out=0, delay_in=0),
            obs=receipt_logger,
        ),
    )
    result = app.run()
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
                    order = Order(
                        order_id=f"order-{self.count}",
                        side=Side.BUY,
                        price=ctx.snapshot.bids[0].price,
                        qty=5,
                    )
                    return [
                        Action(action_type=ActionType.PLACE_ORDER, create_time=0, payload=order)
                    ]
                return []
            if e.kind == EVENT_KIND_RECEIPT_DELIVERY:
                receipt = e.payload
                self.receipts_received.append((receipt.timestamp, receipt.recv_time))
            return []

    strategy = _FrequentStrategy()
    app = BacktestApp(
        RuntimeBuildConfig(
            feed=MockFeed(snapshots),
            venue=ExecutionVenue_Impl(
                simulator=FIFOExchangeSimulator(cancel_bias_k=0.0),
                tape_builder=UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0),
            ),
            strategy=strategy,
            oms=OMS_Impl(),
            timeModel=TimeModel_Impl(
                delay_out=10 * TICK_PER_MS,
                delay_in=5 * TICK_PER_MS,
            ),
            obs=ReceiptLogger_Impl(),
        ),
    )
    result = app.run()
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

        receipt_logger = ReceiptLogger_Impl()
        app = BacktestApp(
            RuntimeBuildConfig(
                feed=MockFeed(snapshots),
                venue=ExecutionVenue_Impl(
                    simulator=FIFOExchangeSimulator(cancel_bias_k=0.0),
                    tape_builder=UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0),
                ),
                strategy=ReplayStrategy_Impl(
                    name="TestReplay",
                    order_file=order_file,
                    cancel_file=cancel_file,
                ),
                oms=OMS_Impl(portfolio=Portfolio(cash=100000.0)),
                timeModel=TimeModel_Impl(delay_out=0, delay_in=0),
                obs=receipt_logger,
            ),
        )
        results = app.run()
        assert results['diagnostics']['orders_submitted'] == 2, (
            f"应提交 2 个订单，实际 {results['diagnostics']['orders_submitted']}"
        )
        assert results['diagnostics']['cancels_submitted'] == 1, (
            f"应有 1 个撤单，实际 {results['diagnostics']['cancels_submitted']}"
        )
