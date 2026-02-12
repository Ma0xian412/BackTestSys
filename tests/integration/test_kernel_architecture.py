"""新 Kernel/Dispatcher/Handlers 架构集成测试。"""

import os
import tempfile

from quant_framework.adapters import (
    DelayTimeModel,
    FIFOExecutionVenue,
    NullObservabilitySinks,
)
from quant_framework.core.app import BacktestApp, RuntimeBuildConfig
from quant_framework.core.runtime import EVENT_KIND_SNAPSHOT_ARRIVAL
from quant_framework.core.types import Level, NormalizedSnapshot, Order, Side
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.tape.builder import TapeConfig, UnifiedTapeBuilder
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.replay_strategy import ReplayStrategy
from tests.conftest import MockFeed, create_test_snapshot


class _FrequentOrderStrategy:
    """用于验证新架构的 on_event 策略回调。"""

    def __init__(self) -> None:
        self._seq = 0

    def on_event(self, e, ctx):
        if e.kind != EVENT_KIND_SNAPSHOT_ARRIVAL or ctx.snapshot is None or not ctx.snapshot.bids:
            return []
        self._seq += 1
        return [Order(order_id=f"snap-{self._seq}", side=Side.BUY, price=ctx.snapshot.bids[0].price, qty=1)]

def _build_basic_app(strategy, snapshots):
    builder = UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0)
    venue = FIFOExecutionVenue(FIFOExchangeSimulator(cancel_bias_k=0.0), builder)
    oms = OrderManager()

    app = BacktestApp()
    config = RuntimeBuildConfig(
        feed=MockFeed(snapshots),
        venue=venue,
        strategy=strategy,
        oms=oms,
        timeModel=DelayTimeModel(delay_out=0, delay_in=0),
        obs=NullObservabilitySinks(),
    )
    return app, config


def test_backtest_app_on_event_strategy():
    snapshots = [
        create_test_snapshot(1000, 100.0, 101.0),
        create_test_snapshot(2000, 100.0, 101.0),
        create_test_snapshot(3000, 100.0, 101.0),
    ]
    app, cfg = _build_basic_app(_FrequentOrderStrategy(), snapshots)

    result = app.run(cfg)

    assert result["intervals"] == 2
    assert result["diagnostics"]["orders_submitted"] >= 2


def test_backtest_app_replay_strategy_cancel_compat():
    with tempfile.TemporaryDirectory() as tmpdir:
        order_file = os.path.join(tmpdir, "orders.csv")
        with open(order_file, "w", encoding="utf-8") as f:
            f.write("OrderId,LimitPrice,Volume,OrderDirection,SentTime\n")
            f.write("1,100.0,10,Buy,1000\n")
            f.write("2,101.0,5,Sell,1100\n")

        cancel_file = os.path.join(tmpdir, "cancels.csv")
        with open(cancel_file, "w", encoding="utf-8") as f:
            f.write("OrderId,CancelSentTime\n")
            f.write("1,1500\n")

        snapshots = [
            NormalizedSnapshot(
                ts_recv=1000,
                bids=[Level(100.0, 100)],
                asks=[Level(101.0, 100)],
                last_vol_split=[(100.0, 20)],
            ),
            NormalizedSnapshot(
                ts_recv=2000,
                bids=[Level(100.0, 90)],
                asks=[Level(101.0, 95)],
                last_vol_split=[(100.0, 30)],
            ),
        ]

        replay = ReplayStrategy(
            name="ReplayCompat",
            order_file=order_file,
            cancel_file=cancel_file,
        )
        venue = FIFOExecutionVenue(
            simulator=FIFOExchangeSimulator(cancel_bias_k=0.0),
            tape_builder=UnifiedTapeBuilder(config=TapeConfig(), tick_size=1.0),
        )
        oms = OrderManager(portfolio=Portfolio(cash=100000.0))

        app = BacktestApp()
        cfg = RuntimeBuildConfig(
            feed=MockFeed(snapshots),
            venue=venue,
            strategy=replay,
            oms=oms,
            timeModel=DelayTimeModel(delay_out=0, delay_in=0),
            obs=NullObservabilitySinks(),
        )

        result = app.run(cfg)

        assert result["diagnostics"]["orders_submitted"] == 2
        assert result["diagnostics"]["cancels_submitted"] == 1
