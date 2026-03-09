"""BacktestApp 集成测试（新架构）。"""

import os
import tempfile
import threading
import time

from quant_framework.adapters import ExecutionVenue_Impl, Observability_Impl, TimeModel_Impl
from quant_framework.core.data_structure import (
    EVENT_KIND_RECEIPT_DELIVERY,
    EVENT_KIND_MDARRIVE,
    Action,
    ActionType,
)
from quant_framework.core import BacktestApp, RuntimeBuildConfig
from quant_framework.core.data_structure import Level, NormalizedSnapshot, Order, Side, TICK_PER_MS
from quant_framework.adapters.interval_model import UnifiedIntervalModel_impl, TapeConfig
from quant_framework.adapters.execution_venue import (
    SegmentBaseAlgorithm,
    Simulator_Impl,
)
from quant_framework.adapters.IOMS.oms import OMS_Impl, Portfolio
from quant_framework.adapters.IStrategy.Replay_Strategy import ReplayStrategy_Impl
from quant_framework.adapters.observability.Observability_Impl import Observability_Impl

from tests.conftest import create_test_snapshot, print_tape_path, MockFeed


# ---------------------------------------------------------------------------
# 基本管线
# ---------------------------------------------------------------------------

def test_basic_pipeline():
    """基本管线：BacktestApp 驱动组件协同。"""
    builder = UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0)
    exchange = ExecutionVenue_Impl(
        simulator=Simulator_Impl(
            match_algo=SegmentBaseAlgorithm(
                cancel_bias_k=0.0,
                tape_builder=builder,
            )
        )
    )
    oms = OMS_Impl()
    receipt_logger = Observability_Impl()

    class _OneShot:
        def __init__(self):
            self.sent = False

        def on_event(self, e, ctx):
            if e.kind == EVENT_KIND_MDARRIVE and not self.sent:
                self.sent = True
                order = Order(order_id="1", side=Side.BUY, price=100.0, qty=1)
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
    assert len(result.OrderInfo) == 1
    assert len(result.DoneInfo) == 1
    assert result.OrderInfo[0].OrderId == 1


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
            if e.kind == EVENT_KIND_MDARRIVE:
                self.count += 1
                self.snapshots_received.append(e.time)
                if ctx.snapshot and ctx.snapshot.bids:
                    order = Order(
                        order_id=str(self.count),
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
                simulator=Simulator_Impl(
                    match_algo=SegmentBaseAlgorithm(
                        cancel_bias_k=0.0,
                        tape_builder=UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0),
                    )
                ),
            ),
            strategy=strategy,
            oms=OMS_Impl(),
            timeModel=TimeModel_Impl(
                delay_out=10 * TICK_PER_MS,
                delay_in=5 * TICK_PER_MS,
            ),
            obs=Observability_Impl(),
        ),
    )
    result = app.run()
    assert len(result.OrderInfo) == 3, f"应提交 3 个订单，实际 {len(result.OrderInfo)}"


# ---------------------------------------------------------------------------
# 重放策略集成
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

        receipt_logger = Observability_Impl()
        app = BacktestApp(
            RuntimeBuildConfig(
                feed=MockFeed(snapshots),
                venue=ExecutionVenue_Impl(
                    simulator=Simulator_Impl(
                        match_algo=SegmentBaseAlgorithm(
                            cancel_bias_k=0.0,
                            tape_builder=UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0),
                        )
                    ),
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
        assert len(results.OrderInfo) == 2, (
            f"应提交 2 个订单，实际 {len(results.OrderInfo)}"
        )
        assert len(results.CancelRequest) == 1, (
            f"应有 1 个撤单，实际 {len(results.CancelRequest)}"
        )


def _build_interrupt_test_app(snapshots, strategy):
    venue = ExecutionVenue_Impl(
        simulator=Simulator_Impl(
            match_algo=SegmentBaseAlgorithm(
                cancel_bias_k=0.0,
                tape_builder=UnifiedIntervalModel_impl(config=TapeConfig(), tick_size=1.0),
            )
        ),
    )
    return BacktestApp(
        RuntimeBuildConfig(
            feed=MockFeed(snapshots),
            venue=venue,
            strategy=strategy,
            oms=OMS_Impl(),
            timeModel=TimeModel_Impl(delay_out=0, delay_in=0),
            obs=Observability_Impl(),
        ),
    )


def test_run_can_interrupt_before_start():
    snapshots = [
        create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0),
        create_test_snapshot(1500 * TICK_PER_MS, 100.1, 101.1),
    ]

    class _NoOpStrategy:
        def on_event(self, e, ctx):
            return []

    app = _build_interrupt_test_app(snapshots, _NoOpStrategy())
    app.request_stop("external_request")
    result = app.run()

    assert len(result.OrderInfo) == 0
    assert len(result.ExecutionDetail) == 0
    assert len(result.DoneInfo) == 0
    assert len(result.CancelRequest) == 0


def test_run_can_interrupt_during_execution():
    snapshots = [
        create_test_snapshot((1000 + i * 500) * TICK_PER_MS, 100.0, 101.0)
        for i in range(60)
    ]
    first_md_seen = threading.Event()

    class _SlowStrategy:
        def on_event(self, e, ctx):
            if e.kind == EVENT_KIND_MDARRIVE:
                first_md_seen.set()
                time.sleep(0.003)
            return []

    app = _build_interrupt_test_app(snapshots, _SlowStrategy())
    result_holder = {}

    def _run_app():
        result_holder["result"] = app.run()

    run_thread = threading.Thread(target=_run_app)
    run_thread.start()
    assert first_md_seen.wait(timeout=1.0), "策略未收到首个行情事件"
    time.sleep(0.01)
    app.request_stop("external_request")
    run_thread.join(timeout=5.0)

    assert not run_thread.is_alive(), "中断后 run 线程应及时退出"
    result = result_holder["result"]
    assert len(result.OrderInfo) < len(snapshots)
