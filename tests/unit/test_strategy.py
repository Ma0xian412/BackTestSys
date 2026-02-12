"""策略单元测试。

验证内容：
- SimpleStrategy 按节奏下单
- ReplayStrategy CSV 加载与订单生成
"""

import os
import tempfile

from quant_framework.core.runtime import (
    ReceiptStrategyEvent,
    SnapshotStrategyEvent,
    StrategyContext,
)
from quant_framework.core.types import Order, OrderReceipt, Side, TICK_PER_MS
from quant_framework.core.dto import (
    to_snapshot_dto, ReadOnlyOMSView, SnapshotDTO, LevelDTO,
)
from quant_framework.trading.strategy import SimpleStrategy
from quant_framework.trading.replay_strategy import ReplayStrategy
from quant_framework.trading.oms import OrderManager

from tests.conftest import create_test_snapshot


def test_simple_strategy():
    """SimpleStrategy：on_event 下每 10 个快照生成 1 个订单。"""
    strategy = SimpleStrategy(name="TestStrategy")
    oms = OrderManager()
    view = ReadOnlyOMSView(oms)

    snapshot = create_test_snapshot(1000 * TICK_PER_MS, 100.0, 101.0)
    dto = to_snapshot_dto(snapshot)

    all_orders = []
    for _ in range(15):
        ev = SnapshotStrategyEvent(time=dto.ts_recv, snapshot=dto)
        ctx = StrategyContext(t=dto.ts_recv, snapshot=dto, omsView=view)
        all_orders.extend(strategy.on_event(ev, ctx))

    assert len(all_orders) == 1, f"15 个快照应生成 1 个订单，实际 {len(all_orders)}"
    assert all_orders[0].side == Side.BUY
    assert all_orders[0].price == 100.0


def test_replay_strategy():
    """ReplayStrategy：首次 SnapshotArrival 返回全部订单和撤单动作。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 订单文件
        order_file = os.path.join(tmpdir, "PubOrderLog_T_D_Id12345.csv")
        with open(order_file, 'w') as f:
            f.write("OrderId,LimitPrice,Volume,OrderDirection,SentTime\n")
            f.write("1001,100.5,10,Buy,1000\n")
            f.write("1002,99.0,20,Sell,1100\n")
            f.write("1003,101.0,15,Buy,1200\n")

        # 撤单文件
        cancel_file = os.path.join(tmpdir, "PubOrderCancelRequestLog_T_D_Id12345.csv")
        with open(cancel_file, 'w') as f:
            f.write("OrderId,CancelSentTime\n")
            f.write("1001,1500\n")
            f.write("1002,1600\n")

        strategy = ReplayStrategy(
            name="TestReplay",
            order_file=order_file,
            cancel_file=cancel_file,
        )

        # 验证加载
        assert len(strategy.pending_orders) == 3, "应加载 3 个订单"
        assert len(strategy.pending_cancels) == 2, "应加载 2 个撤单"

        # 按时间排序
        order_times = [t for t, _ in strategy.pending_orders]
        assert order_times == sorted(order_times), "订单应按时间排序"
        cancel_times = [t for t, _ in strategy.pending_cancels]
        assert cancel_times == sorted(cancel_times), "撤单应按时间排序"

        class _MockOMSView:
            def get_active_orders(self):
                return []

            def get_portfolio(self):
                return None

        snap = SnapshotDTO(
            ts_recv=1000,
            bids=(LevelDTO(100.0, 100),),
            asks=(LevelDTO(101.0, 100),),
        )
        sev = SnapshotStrategyEvent(time=snap.ts_recv, snapshot=snap)
        sctx = StrategyContext(t=snap.ts_recv, snapshot=snap, omsView=_MockOMSView())
        actions = strategy.on_event(sev, sctx)
        assert len(actions) == 5, "第一次快照应返回 3 个订单 + 2 个撤单动作"
        assert sum(1 for a in actions if isinstance(a, Order)) == 3

        # 后续不再返回
        assert len(strategy.on_event(sev, sctx)) == 0

        # 回执事件不触发新动作
        rev = ReceiptStrategyEvent(
            time=snap.ts_recv,
            receipt=OrderReceipt(order_id="x", receipt_type="FILL", timestamp=snap.ts_recv, recv_time=snap.ts_recv),
        )
        assert strategy.on_event(rev, sctx) == []

        # 统计
        stats = strategy.get_statistics()
        assert stats['total_orders'] == 3
        assert stats['total_cancels'] == 2
