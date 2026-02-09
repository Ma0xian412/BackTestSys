"""EventLoopRunner 集成测试。

验证多组件协同工作：
- 基本管线（Tape → Exchange → OMS → Strategy）
- 带延迟的管线
- ReplayStrategy 集成
"""

import os
import tempfile

from quant_framework.core.types import (
    Order, Side, TimeInForce, NormalizedSnapshot, Level, TICK_PER_MS,
)
from quant_framework.core.dto import to_snapshot_dto, ReadOnlyOMSView
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.strategy import SimpleStrategy
from quant_framework.trading.replay_strategy import ReplayStrategy
from quant_framework.trading.receipt_logger import ReceiptLogger
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig, TimelineConfig

from tests.conftest import create_test_snapshot, print_tape_path, MockFeed


# ---------------------------------------------------------------------------
# 基本管线
# ---------------------------------------------------------------------------

def test_basic_pipeline():
    """基本管线：Tape 构建 → 交易所设置 → 策略下单 → 交易所推进。"""
    builder = UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0)
    exchange = FIFOExchangeSimulator()
    oms = OrderManager()
    oms_view = ReadOnlyOMSView(oms)
    strategy = SimpleStrategy(name="TestStrategy")

    prev = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    curr = create_test_snapshot(1500 * TICK_PER_MS, 100.5, 101.5)
    prev_dto = to_snapshot_dto(prev)

    tape = builder.build(prev, curr)
    print_tape_path(tape)

    exchange.set_tape(tape, 1000 * TICK_PER_MS, 1500 * TICK_PER_MS)

    orders = strategy.on_snapshot(prev_dto, oms_view)
    for o in orders:
        oms.submit(o, 1000 * TICK_PER_MS)

    for o in oms.get_active_orders():
        receipt = exchange.on_order_arrival(o, 1100 * TICK_PER_MS, market_qty=50)
        if receipt:
            oms.on_receipt(receipt)

    if tape:
        seg = tape[0]
        receipts, _ = exchange.advance(seg.t_start, seg.t_end, seg)
        for r in receipts:
            oms.on_receipt(r)


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

        def on_snapshot(self, snapshot, oms_view):
            self.count += 1
            self.snapshots_received.append(
                snapshot.ts_exch if hasattr(snapshot, 'ts_exch') else snapshot
            )
            if snapshot.bids:
                return [Order(
                    order_id=f"order-{self.count}",
                    side=Side.BUY,
                    price=snapshot.bids[0].price,
                    qty=5,
                )]
            return []

        def on_receipt(self, receipt, snapshot, oms_view):
            self.receipts_received.append((receipt.timestamp, receipt.recv_time))
            return []

    strategy = _FrequentStrategy()

    runner = EventLoopRunner(
        feed=MockFeed(snapshots),
        tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
        exchange=FIFOExchangeSimulator(cancel_bias_k=0.0),
        strategy=strategy,
        oms=OrderManager(),
        config=RunnerConfig(
            delay_out=10 * TICK_PER_MS,
            delay_in=5 * TICK_PER_MS,
            timeline=TimelineConfig(a=1.0, b=0),
        ),
    )
    results = runner.run()

    assert results['intervals'] == 2, f"应处理 2 个区间，实际 {results['intervals']}"


# ---------------------------------------------------------------------------
# ReplayStrategy 集成
# ---------------------------------------------------------------------------

def test_replay_pipeline():
    """ReplayStrategy 集成：订单和撤单通过 EventLoopRunner 正确处理。"""
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
        runner = EventLoopRunner(
            feed=MockFeed(snapshots),
            tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
            exchange=FIFOExchangeSimulator(cancel_bias_k=0.0),
            strategy=ReplayStrategy(
                name="TestReplay",
                order_file=order_file,
                cancel_file=cancel_file,
            ),
            oms=OrderManager(portfolio=Portfolio(cash=100000.0)),
            config=RunnerConfig(delay_out=0, delay_in=0),
            receipt_logger=receipt_logger,
        )
        results = runner.run()

        assert results['diagnostics']['orders_submitted'] == 2, (
            f"应提交 2 个订单，实际 {results['diagnostics']['orders_submitted']}"
        )
        assert results['diagnostics']['cancels_submitted'] == 1, (
            f"应有 1 个撤单，实际 {results['diagnostics']['cancels_submitted']}"
        )
